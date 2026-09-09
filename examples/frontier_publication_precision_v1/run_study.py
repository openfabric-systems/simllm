"""Qualify complete publication-interval comparisons without repricing services."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE_COMMIT = "d88c52005be7e0b9a9a0ae2142f099ee640d5f04"
SCHEMA = "simllm-frontier-publication-precision-evaluation-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_new(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value))


def fraction(value):
    return F(value["numerator"], value["denominator"])


def wire(value):
    if isinstance(value, F):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [wire(item) for item in value]
    return value


def projection(value):
    return {
        **wire(asdict(value)),
        "axis": value.threshold.axis,
        "units": value.threshold.units,
        "quotients": wire(value.quotients),
        "quotient_bounds": wire(value.quotient_bounds),
        "possibly_infeasible": value.possibly_infeasible,
        "agreement_verdict": value.verdict,
    }


def conditional_enclosure(p):
    """Declared binary64 profile, never verified historical export provenance."""
    if not F(1) <= p <= F(1023):
        raise ValueError("conditional publication profile requires 1 <= p <= 1023")
    q, u = F(1, 1000), F(1, 2**53)
    guard = 4 * u * (p + q)
    return p - q / 2 - guard, p + q / 2 + guard, guard


class Evaluation:
    def __init__(self):
        from simllm import deploy

        self.api = deploy
        self.freeze = read(HERE / "expectations.json")
        self.freeze_digest = digest(HERE / "expectations.json")
        self.data = {
            "schema": SCHEMA,
            "expectations_commit": FREEZE_COMMIT,
            "expectations_sha256": self.freeze_digest,
            "fatal_guards": [],
            "exact_oracles": [],
            "behavioral_relations": [],
            "synthetic_cells": [],
            "boundary_cells": [],
            "historical_agreement": [],
        }

    def guard(self, identity, passed, **detail):
        self.data["fatal_guards"].append({"id": identity, "passed": bool(passed), **wire(detail)})

    def reject(self, identity, function):
        try:
            function()
        except (ValueError, TypeError):
            self.guard(identity, True)
        else:
            self.guard(identity, False)

    def source(self, text, row_id, interval, *, conditional=False):
        return self.api.PublishedThreshold(
            text,
            self.freeze_digest,
            row_id,
            interval,
            self.api.PublicationBasis.CONDITIONAL
            if conditional
            else self.api.PublicationBasis.DECLARED,
            "conditional binary64 enclosure, historical profile unverified"
            if conditional
            else "declared exact synthetic geometry, no hardware measurement",
        )

    @staticmethod
    def coordinate(point):
        return point[1], point[2]

    @staticmethod
    def identity(point):
        return point[0]

    def compare(self, points, threshold, reference=F(100)):
        return self.api.compare_published_frontier(
            points,
            threshold,
            reference,
            coordinate=self.coordinate,
            identity=self.identity,
            quotient_band=tuple(F(value) for value in self.freeze["agreement_band"]),
        )

    def guard_projection(self, identity, points, result):
        snapshot = {point[0]: point for point in points}
        segments = result.segments
        maximum_y = max((point[2] for point in points), default=F())
        joined = True
        exact = True
        for segment in segments:
            choice = segment.selection
            if choice is not None:
                joined &= snapshot.get(choice.point_id) == (choice.point_id, choice.x, choice.y)
                joined &= 0 < choice.y <= maximum_y
            bounds = segment.interval
            probes = {(bounds.lower + bounds.upper) / 2}
            probes.update(point[1] for point in points if bounds.contains(point[1]))
            if bounds.lower_closed:
                probes.add(bounds.lower)
            if bounds.upper_closed:
                probes.add(bounds.upper)
            for probe in probes:
                feasible = [p for p in points if p[1] >= probe]
                selected = max(feasible, key=lambda p: (p[2], p[0]), default=None)
                actual = None if choice is None else (choice.point_id, choice.x, choice.y)
                exact &= actual == selected
        self.guard(f"{identity}:source-coordinate-join", joined)
        self.guard(f"{identity}:legal-complete-selection", exact)
        interval = result.threshold.interval
        coverage = (
            segments[0].interval.lower == interval.lower
            and segments[0].interval.lower_closed == interval.lower_closed
            and segments[-1].interval.upper == interval.upper
            and segments[-1].interval.upper_closed == interval.upper_closed
            and all(
                a.interval.upper == b.interval.lower
                and a.interval.upper_closed != b.interval.lower_closed
                for a, b in itertools.pairwise(segments)
            )
            and all(a.selection != b.selection for a, b in itertools.pairwise(segments))
        )
        self.guard(f"{identity}:maximal-partition-coverage", coverage)
        reversed_result = self.compare(
            tuple(reversed(points)), result.threshold, result.reference_y
        )
        self.guard(
            f"{identity}:permutation-identity", projection(result) == projection(reversed_result)
        )

    def preservation(self, when):
        for item in self.freeze["preservation"]:
            actual = digest(ROOT / item["path"])
            self.guard(
                f"preservation:{when}:{item['path']}", actual == item["sha256"], sha256=actual
            )

    def admission(self):
        interval = self.api.ThresholdInterval(F(3, 2), F(5, 2))
        threshold = self.source("2.000", "admission", interval)
        points = (("a", F(2), F(100)), ("b", F(3), F(40)))
        result = self.compare(points, threshold)
        self.reject(
            "reject:duplicate-point-id", lambda: self.compare((points[0], points[0]), threshold)
        )
        self.reject(
            "reject:float-coordinate", lambda: self.compare((("a", 2.0, F(100)),), threshold)
        )
        self.reject(
            "reject:open-singleton", lambda: self.api.ThresholdInterval(F(2), F(2), False, True)
        )
        self.reject("reject:missing-assumptions", lambda: replace(threshold, assumptions=""))
        self.reject("reject:invalid-source-digest", lambda: replace(threshold, source_sha256="bad"))
        self.reject(
            "reject:exact-nonsingleton",
            lambda: replace(threshold, basis=self.api.PublicationBasis.SOURCE_EXACT),
        )
        for name, selection in [
            ("suboptimal", self.api.FrontierSelection("b", F(3), F(40))),
            ("false-none", None),
            ("false-coordinate", self.api.FrontierSelection("a", F(3), F(100))),
        ]:
            segment = self.api.FrontierSelectionSegment(interval, selection)
            self.reject(
                f"reject:coherent-result-{name}",
                lambda segment=segment: replace(result, segments=(segment,)),
            )
        self.reject(
            "reject:result-source-replacement",
            lambda: replace(result, threshold=replace(threshold, source_sha256="b" * 64)),
        )
        self.reject(
            "reject:result-band-replacement",
            lambda: replace(result, quotient_band=(F(1, 10), F(10))),
        )
        self.reject("reject:direct-result-construction", self.api.FrontierPublicationComparison)
        self.reject("reject:conditional-range-low", lambda: conditional_enclosure(F(1, 2)))
        self.reject("reject:conditional-range-high", lambda: conditional_enclosure(F(1024)))

    def boundaries(self):
        for case in self.freeze["boundary_oracles"]:
            identity = case["id"]
            points = tuple((name, F(x), F(y)) for name, x, y in case["points"])
            interval = self.api.ThresholdInterval(
                F(case["lower"]), F(case["upper"]), case["lower_closed"], case["upper_closed"]
            )
            threshold = self.source("2.000", identity, interval)
            result = self.compare(points, threshold, F(case["reference_y"]))
            choices = [s.selection.point_id if s.selection else None for s in result.segments]
            self.data["exact_oracles"].append(
                {
                    "id": identity,
                    "family": "boundary",
                    "passed": choices == case["expected_choice_sequence"]
                    and result.verdict == case["expected_verdict"],
                    "observed_choices": choices,
                    "expected_choices": case["expected_choice_sequence"],
                    "observed_verdict": result.verdict,
                    "expected_verdict": case["expected_verdict"],
                }
            )
            self.data["boundary_cells"].append(
                {"id": identity, "points": wire(points), "comparison": projection(result)}
            )
            self.guard_projection(identity, points, result)
            if interval.lower == interval.upper:
                exact = self.api.frontier_at_threshold(
                    points, interval.lower, coordinate=self.coordinate, identity=self.identity
                )
                self.guard(
                    f"{identity}:singleton-exact-identity", result.segments[0].selection == exact
                )

    def synthetic(self):
        grid = self.freeze["synthetic_grid"]
        for center, places, offset_text in itertools.product(
            grid["centers"], grid["decimal_places"], grid["offsets"]
        ):
            c, delta, q = F(center), F(offset_text), F(1, 10**places)
            identity = f"c{center}-d{places}-offset{offset_text}"
            points = (
                ("primary", c + delta, F(grid["primary_y"])),
                ("alternate", c + grid["alternate_x_delta"], F(grid["alternate_y"])),
            )
            interval = self.api.ThresholdInterval(c - q / 2, c + q / 2)
            result = self.compare(
                points, self.source(str(center), identity, interval), F(grid["reference_y"])
            )
            expected = "PASS" if delta >= q / 2 else "FAIL" if delta < -q / 2 else "INDETERMINATE"
            expected_ids = (
                ["primary"]
                if expected == "PASS"
                else ["alternate"]
                if expected == "FAIL"
                else ["primary", "alternate"]
            )
            observed_ids = [s.selection.point_id for s in result.segments]
            self.data["behavioral_relations"].append(
                {
                    "id": identity,
                    "family": "precision-and-coordinate-position",
                    "passed": result.verdict == expected and observed_ids == expected_ids,
                    "expected_verdict": expected,
                    "observed_verdict": result.verdict,
                    "expected_choices": expected_ids,
                    "observed_choices": observed_ids,
                }
            )
            self.data["synthetic_cells"].append(
                {
                    "id": identity,
                    "center": center,
                    "decimal_places": places,
                    "offset": offset_text,
                    "points": wire(points),
                    "comparison": projection(result),
                }
            )
            self.guard_projection(identity, points, result)

    def historical(self):
        config = self.freeze["historical"]
        path = ROOT / config["point_record"]
        record = read(path)
        original = encoded(record)
        family = record["families"]["F"]
        raw_points = family["ideal_frontier"]
        points = tuple(
            (
                p["candidate_key"],
                fraction(p["x_tokens_per_second_per_user"]),
                fraction(p["y_tokens_per_second_per_gpu"]),
            )
            for p in raw_points
        )
        input_snapshot = copy.deepcopy(raw_points)
        by_key = {p["candidate_key"]: p for p in raw_points}
        self.data["historical_point_snapshot"] = raw_points
        with (ROOT / config["external_csv"]).open(newline="", encoding="utf-8") as stream:
            external = list(csv.DictReader(stream))
        self.guard("historical:source-row-count", len(external) == len(config["rows"]))
        composition = read(ROOT / "examples/matched_seam_frontier_v1/study_config.json")[
            "composition"
        ]
        rate = F.from_float(float.fromhex(composition["decode_rate_matching_degradation_hex"]))
        for point in family["ideal_points"]:
            x = fraction(point["x_tokens_per_second_per_user"])
            y = fraction(point["y_tokens_per_second_per_gpu"])
            capacity = fraction(point["request_capacity_per_second"])
            declaration = point["configuration"]
            external_row = external[point["row"] - 1]
            batch = int(external_row["(d)bs"])
            gpu_count = (
                declaration["decode_workers"] * declaration["decode_tp"]
                + declaration["prefill_workers"] * declaration["prefill_tp"]
            )
            joined = (
                declaration["decode_batch"] == batch
                and declaration["decode_tp"] == int(external_row["(d)tp"])
                and declaration["decode_workers"] == int(external_row["(d)workers"])
                and declaration["prefill_tp"] == int(external_row["(p)tp"])
                and declaration["prefill_workers"] == int(external_row["(p)workers"])
                and gpu_count == declaration["used_gpus"] == int(external_row["num_total_gpus"])
            )
            self.guard(f"historical:configuration-join:{point['row']}", joined)
            for phase in ("prefill", "decode"):
                stamp = self.api.estimate_stamp_from_json(point[f"{phase}_stamp"])
                self.guard(
                    f"historical:{phase}-stamp-join:{point['row']}",
                    stamp.candidate_key == point["candidate_key"],
                )
            ceiling = batch * declaration["decode_workers"] * rate * x / gpu_count
            valid = (
                x == F(10**12, point["decode_step_ps"])
                and y == capacity * 500 / gpu_count
                and 0 < y <= ceiling
            )
            self.guard(
                f"historical:physical-axes:{point['row']}",
                valid,
                x=x,
                y=y,
                decode_capacity_ceiling=ceiling,
            )
        for index, external_row in enumerate(external, 1):
            identity = f"historical-row-{index:02d}"
            x, y = F(external_row["tokens/s/user"]), F(external_row["tokens/s/gpu"])
            answer = self.api.frontier_at_threshold(
                points, x, coordinate=self.coordinate, identity=self.identity
            )
            answer_id = "none" if answer is None else by_key[answer.point_id]["candidate_id"]
            quotient = F() if answer is None else answer.y / y
            oracle = next(row for row in family["bracket_rows"] if row["row"] == index)
            self.data["exact_oracles"].append(
                {
                    "id": identity,
                    "family": "historical-exact-threshold",
                    "passed": answer_id == oracle["frontier_answer"]
                    and quotient == fraction(oracle["quotient"]),
                    "observed_choice": answer_id,
                    "expected_choice": oracle["frontier_answer"],
                    "observed_quotient": wire(quotient),
                    "expected_quotient": oracle["quotient"],
                }
            )
            lower, upper, guard = conditional_enclosure(x)
            threshold = replace(
                self.source(
                    external_row["tokens/s/user"],
                    identity,
                    self.api.ThresholdInterval(lower, upper),
                    conditional=True,
                ),
                source_sha256=digest(ROOT / config["external_csv"]),
            )
            result = self.compare(points, threshold, y)
            self.guard_projection(identity, points, result)
            self.guard(
                f"{identity}:conditional-evidence-retained",
                result.threshold.basis is self.api.PublicationBasis.CONDITIONAL,
            )
            row_points = [p for p in family["ideal_points"] if p["row"] == index]
            self.guard(f"{identity}:unique-config-join", len(row_points) == 1)
            matching = row_points[0]
            self.data["historical_agreement"].append(
                {
                    "id": identity,
                    "external_row": index,
                    "source_row": external_row,
                    "exact_threshold_choice": answer_id,
                    "exact_threshold_quotient": wire(quotient),
                    "historical_profile_verified": False,
                    "guard": wire(guard),
                    "interpretation": "choices over a conditional enclosure, not the exact exporter preimage",
                    "matched_candidate_key": matching["candidate_key"],
                    "matched_coordinate_in_enclosure": threshold.interval.contains(
                        fraction(matching["x_tokens_per_second_per_user"])
                    ),
                    "matched_candidate_is_possible_selection": any(
                        s.selection is not None
                        and s.selection.point_id == matching["candidate_key"]
                        for s in result.segments
                    ),
                    "comparison": projection(result),
                }
            )
        self.guard(
            "historical:point-and-stamp-identity",
            raw_points == input_snapshot and encoded(record) == original,
        )

    def finish(self):
        grid = self.freeze["synthetic_grid"]
        expected = {
            f"c{c}-d{d}-offset{o}"
            for c, d, o in itertools.product(
                grid["centers"], grid["decimal_places"], grid["offsets"]
            )
        }
        for key in ["synthetic_cells", "behavioral_relations"]:
            ids = [row["id"] for row in self.data[key]]
            self.guard(f"complete:{key}", set(ids) == expected and len(ids) == grid["cell_count"])
        for family, expected_ids in [
            ("boundary", {r["id"] for r in self.freeze["boundary_oracles"]}),
            (
                "historical-exact-threshold",
                {f"historical-row-{i:02d}" for i in self.freeze["historical"]["rows"]},
            ),
        ]:
            ids = [row["id"] for row in self.data["exact_oracles"] if row["family"] == family]
            self.guard(
                f"complete:{family}", set(ids) == expected_ids and len(ids) == len(expected_ids)
            )
        for key, expected_ids in (
            ("boundary_cells", {r["id"] for r in self.freeze["boundary_oracles"]}),
            (
                "historical_agreement",
                {f"historical-row-{i:02d}" for i in self.freeze["historical"]["rows"]},
            ),
        ):
            ids = [row["id"] for row in self.data[key]]
            self.guard(
                f"complete:{key}", set(ids) == expected_ids and len(ids) == len(expected_ids)
            )
        oracle_pairs = [(row["family"], row["id"]) for row in self.data["exact_oracles"]]
        expected_pairs = {("boundary", r["id"]) for r in self.freeze["boundary_oracles"]} | {
            ("historical-exact-threshold", f"historical-row-{i:02d}")
            for i in self.freeze["historical"]["rows"]
        }
        self.guard(
            "complete:exact-oracle-families",
            set(oracle_pairs) == expected_pairs and len(oracle_pairs) == len(expected_pairs),
        )
        self.guard("retained-cell-oracle-joins", retained_oracle_joins(self.data))
        ids = [row["id"] for row in self.data["fatal_guards"]]
        self.guard("unique-fatal-identities", len(ids) == len(set(ids)))
        fatal = [row["id"] for row in self.data["fatal_guards"] if not row["passed"]]
        scored = self.data["behavioral_relations"]
        self.data.update(
            {
                "state": "VOID"
                if fatal
                else "PASS"
                if all(row["passed"] for row in [*scored, *self.data["exact_oracles"]])
                else "FAIL",
                "fatal_findings": fatal,
                "behavioral_score": None
                if fatal
                else {"passed": sum(row["passed"] for row in scored), "instances": len(scored)},
                "historical_agreement_is_scored_for_study": False,
            }
        )
        return self.data


def worker(output):
    started = time.monotonic()
    runtime = {"gpu_imports": 0, "backend_processes": 0}

    def audit(event, args):
        if event == "import" and str(args[0]).split(".")[0] in {"torch", "cupy", "triton"}:
            runtime["gpu_imports"] += 1
            raise RuntimeError("GPU imports are outside the publication comparison")
        if event == "subprocess.Popen":
            runtime["backend_processes"] += 1
            raise RuntimeError("comparison worker starts no subprocess")

    sys.addaudithook(audit)
    evaluation = Evaluation()
    try:
        evaluation.preservation("before")
        for method in (
            evaluation.admission,
            evaluation.boundaries,
            evaluation.synthetic,
            evaluation.historical,
        ):
            method()
        evaluation.preservation("after")
    except (
        ArithmeticError,
        AssertionError,
        AttributeError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        evaluation.guard(
            "unexpected-worker-exception", False, exception_type=type(exc).__name__, detail=str(exc)
        )
    evaluation.guard(
        "no-gpu-imports-or-backend-processes", all(value == 0 for value in runtime.values())
    )
    value = evaluation.finish()
    value["runtime_calls"] = runtime
    write_new(output, value)
    write_new(
        output.with_suffix(".process.json"),
        {"process_id": os.getpid(), "elapsed_wall_seconds": time.monotonic() - started},
    )
    return 0 if value["state"] == "PASS" else 1


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def coordinator(output):
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("output must be outside the repository")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("commit the source before running the prospective study")
    source = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", FREEZE_COMMIT, source)
    paths = [
        f"examples/frontier_publication_precision_v1/expectations.{suffix}"
        for suffix in ("md", "json")
    ]
    if git("diff", "--name-only", FREEZE_COMMIT, "HEAD", "--", *paths):
        raise ValueError("frozen expectations changed")
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "configuration.json",
        {"source_commit": source, "expectations_commit": FREEZE_COMMIT, "workers": 2},
    )
    workers = []
    for index in range(2):
        stdout = (output / f"worker-{index}.stdout").open("xb")
        stderr = (output / f"worker-{index}.stderr").open("xb")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "examples.frontier_publication_precision_v1.run_study",
                "--worker-output",
                str(output / f"evaluation-{index}.json"),
            ],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
        )
        workers.append((process, stdout, stderr))
    codes = []
    for process, stdout, stderr in workers:
        codes.append(process.wait())
        stdout.close()
        stderr.close()
    files = [output / f"evaluation-{i}.json" for i in range(2)]
    findings = []
    try:
        values = [read(path) for path in files]
        metadata = [read(path.with_suffix(".process.json")) for path in files]
        identities = [m["process_id"] for m in metadata]
        if len(set(identities)) != 2 or any(type(pid) is not int or pid <= 0 for pid in identities):
            findings.append("two-fresh-process-identities")
        for i, value in enumerate(values):
            if not valid_worker(value):
                findings.append(f"invalid-worker-summary:{i}")
            elif value["state"] == "VOID":
                findings.append(f"worker-fatal:{i}")
            if codes[i] != (0 if value.get("state") == "PASS" else 1):
                findings.append(f"worker-state-exit:{i}")
        equal = files[0].read_bytes() == files[1].read_bytes()
        if not equal:
            findings.append("complete-evaluation-byte-identity")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        values = []
        equal = False
        findings.append("malformed-worker-evidence")
    if git("rev-parse", "HEAD") != source or git(
        "status", "--porcelain", "--untracked-files=normal"
    ):
        findings.append("source-changed-during-run")
    state = "VOID" if findings else "PASS" if all(v["state"] == "PASS" for v in values) else "FAIL"
    summary = {
        "schema": "simllm-frontier-publication-precision-summary-v1",
        "state": state,
        "source_commit": source,
        "expectations_commit": FREEZE_COMMIT,
        "fatal_findings": findings,
        "behavioral_score": None if findings else values[0]["behavioral_score"],
        "evaluation_sha256": [digest(p) if p.exists() else None for p in files],
        "deterministic_bytes_equal": equal,
        "worker_exit_codes": codes,
        "historical_agreement_is_study_score": False,
        "historical_profile_verified": False,
        "historical_runs_rescored": False,
    }
    write_new(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if state == "PASS" else 1


def retained_oracle_joins(value):
    """Require each retained comparison to carry its matching exact-oracle result."""
    try:
        oracles = {(row["family"], row["id"]): row for row in value["exact_oracles"]}
        for cell in value["boundary_cells"]:
            oracle = oracles[("boundary", cell["id"])]
            comparison = cell["comparison"]
            choices = [
                None if row["selection"] is None else row["selection"]["point_id"]
                for row in comparison["segments"]
            ]
            if (
                choices != oracle["observed_choices"]
                or comparison["agreement_verdict"] != oracle["observed_verdict"]
            ):
                return False
        for cell in value["historical_agreement"]:
            oracle = oracles[("historical-exact-threshold", cell["id"])]
            if (
                cell["id"] != f"historical-row-{cell['external_row']:02d}"
                or cell["exact_threshold_choice"] != oracle["observed_choice"]
                or cell["exact_threshold_quotient"] != oracle["observed_quotient"]
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def valid_worker(value):
    """Admit only typed, complete worker summaries with consistent qualification."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return False
    for key in ["fatal_guards", "exact_oracles", "behavioral_relations"]:
        rows = value.get(key)
        if not isinstance(rows, list) or not rows:
            return False
        if any(
            not isinstance(r, dict)
            or not isinstance(r.get("id"), str)
            or type(r.get("passed")) is not bool
            for r in rows
        ):
            return False
    fatal = [r["id"] for r in value["fatal_guards"] if not r["passed"]]
    relations = value["behavioral_relations"]
    if "behavioral_score" not in value:
        return False
    if not fatal:
        supplied_score = value["behavioral_score"]
        if (
            not isinstance(supplied_score, dict)
            or set(supplied_score) != {"passed", "instances"}
            or any(type(count) is not int or count < 0 for count in supplied_score.values())
        ):
            return False
        freeze = read(HERE / "expectations.json")
        grid = freeze["synthetic_grid"]
        expected_relations = {
            f"c{c}-d{d}-offset{o}"
            for c, d, o in itertools.product(
                grid["centers"], grid["decimal_places"], grid["offsets"]
            )
        }
        if (
            {row["id"] for row in relations} != expected_relations
            or len(relations) != len(expected_relations)
            or any(row.get("family") != "precision-and-coordinate-position" for row in relations)
        ):
            return False
        for family, expected_ids in (
            ("boundary", {row["id"] for row in freeze["boundary_oracles"]}),
            (
                "historical-exact-threshold",
                {f"historical-row-{i:02d}" for i in freeze["historical"]["rows"]},
            ),
        ):
            ids = [r["id"] for r in value["exact_oracles"] if r.get("family") == family]
            if set(ids) != expected_ids or len(ids) != len(expected_ids):
                return False
        if len(value["exact_oracles"]) != 25:
            return False
        for name, expected_ids in (
            ("synthetic_cells", expected_relations),
            ("boundary_cells", {r["id"] for r in freeze["boundary_oracles"]}),
            (
                "historical_agreement",
                {f"historical-row-{i:02d}" for i in freeze["historical"]["rows"]},
            ),
        ):
            rows = value.get(name)
            if (
                not isinstance(rows, list)
                or len(rows) != len(expected_ids)
                or any(
                    not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows
                )
                or {row["id"] for row in rows} != expected_ids
            ):
                return False
        if not retained_oracle_joins(value):
            return False
        if any(
            not isinstance(row, dict) or row.get("historical_profile_verified") is not False
            for row in value["historical_agreement"]
        ):
            return False
    if value.get("historical_agreement_is_scored_for_study") is not False:
        return False
    passed = all(r["passed"] for r in [*relations, *value["exact_oracles"]])
    state = "VOID" if fatal else "PASS" if passed else "FAIL"
    score = (
        None
        if fatal
        else {"passed": sum(r["passed"] for r in relations), "instances": len(relations)}
    )
    return (
        value.get("state") == state
        and value.get("fatal_findings") == fatal
        and value.get("behavioral_score") == score
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-root", type=Path)
    group.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    return worker(args.worker_output) if args.worker_output else coordinator(args.output_root)


if __name__ == "__main__":
    raise SystemExit(main())
