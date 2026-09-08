"""Audit historical measurement inferences using packet-free counterexamples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
FREEZE = "06a88de"
HISTORICAL = ROOT / "examples/nvlink_incast_validation_v1"


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*arguments):
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, check=True).stdout


def rational(value):
    return {"exact": str(value), "decimal": float(value)}


def load_scorer():
    path = HISTORICAL / "score_study.py"
    spec = importlib.util.spec_from_file_location("measurement_audit_historical_scorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packet_free_pair(size, rate, overhead_ps):
    if min(size, rate) <= 0 or overhead_ps < 0:
        raise ValueError("positive size/rate and nonnegative overhead are required")
    rows = []
    for length in (size, 2 * size):
        floor = Fraction(length * 10**12, rate)
        duration = floor + overhead_ps
        measured_goodput = Fraction(length * 10**12, duration)
        error = (rate - measured_goodput) / measured_goodput
        rows.append(
            {
                "size_bytes": length,
                "duration_ps": duration,
                "floor_ps": floor,
                "error": error,
            }
        )
    return rows


def classifier_rows(pair, degrees):
    return [
        {
            "degree": degree,
            "size_bytes": row["size_bytes"],
            "aggregate_signed_relative_error": float(row["error"]),
            "hardware_completion_us_by_source": [float(row["duration_ps"] / 10**6)] * degree,
            "simulation_completion_us_by_source": [float(row["floor_ps"] / 10**6)] * degree,
        }
        for degree, row in product(degrees, pair)
    ]


def phase_case(degree, duration_ps, stagger_ps):
    if degree < 1 or duration_ps <= 0 or stagger_ps < 0:
        raise ValueError("invalid phase geometry")
    starts = [index * stagger_ps for index in range(degree)]
    finishes = [start + duration_ps for start in starts]
    local_max = max(finish - start for start, finish in zip(starts, finishes, strict=True))
    phase = max(finishes) - min(starts)
    return {
        "degree": degree,
        "duration_ps": duration_ps,
        "stagger_ps": stagger_ps,
        "start_ps": starts,
        "finish_ps": finishes,
        "maximum_local_duration_ps": local_max,
        "common_phase_ps": phase,
        "goodput_overstatement": rational(Fraction(phase, local_max)),
    }


def historical_audit(frozen):
    result = json.loads((HISTORICAL / "results_run2.json").read_bytes())
    expectations = json.loads((HISTORICAL / "expectations_run2.json").read_bytes())
    budget = expectations["hardware_arm"]["launch_skew"]["per_additional_sender_budget_ps"]
    rows = []
    findings = []
    for sample, published in zip(
        result["hardware_samples"], result["fatal_guards"]["launch_skew_rows"], strict=True
    ):
        derived = (sample["degree"] - 1) * budget / (min(sample["completion_us_by_source"]) * 10**6)
        if (
            sample["degree"] != published["degree"]
            or sample["size_bytes"] != published["size_bytes"]
            or abs(derived - published["launch_skew_fraction"]) > 1e-12
        ):
            findings.append("historical launch-budget fraction does not reconstruct")
        rows.append(
            {
                "degree": sample["degree"],
                "size_bytes": sample["size_bytes"],
                "repetition": sample["repetition"],
                "declared_budget_fraction": derived,
                "source_start_observation": None,
            }
        )
    if any(
        "start" in key.lower() or "epoch" in key.lower()
        for sample in result["hardware_samples"]
        for key in sample
    ):
        findings.append("historical timing schema changed; review the new observations")
    pacing = []
    config = frozen["producer_source_probe"]
    for size, pattern in product(config["sizes_bytes"], config["patterns"]):
        checksum = 1469598103934665603
        for byte in pattern.encode():
            checksum = ((checksum ^ byte) * 1099511628211) % 2**64
        message = (config["first_warp_ordinal"] + checksum) % (size // config["payload_bytes"])
        deadline_ns = Fraction(message * config["payload_bytes"], config["offered_rate_percent"])
        pacing.append(
            {
                "size_bytes": size,
                "pattern": pattern,
                "first_message_address_index": message,
                "nominal_deadline_ns": rational(deadline_ns),
                "evidence_class": "derived_source_coordinate_not_measured_wait",
            }
        )
    return {
        "evidence_class": "post_specified_historical_inference_audit",
        "original_status": result["status"],
        "qualification_verdict": "VOID_UNDECIDABLE_ALIGNMENT_PRECONDITION",
        "hardware_behavioral_score": None,
        "responsible_mechanism": "unidentified",
        "declared_budget_ps_per_additional_sender": budget,
        "budget_reconstructions": rows,
        "producer_pacing_coordinates": pacing,
        "source_boundary_findings": [
            "The scorer divides a declared start budget by a local device-event duration.",
            "Maximum local duration is not a measured common first-release to last-completion span.",
            "Pacing uses the shuffled message address, so pattern labels change offered timing.",
            "The byte loop and access_width argument do not identify compiled memory instructions.",
            "Final per-address values establish coverage, not temporal destination arrival order.",
            "The host batch interval contains submission and synchronization work.",
            "Device cycle counters are not an established common epoch across cards.",
        ],
        "unchanged_capture_row_count": len(rows),
    }, findings


def run_study(output_root):
    own_path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    if Path(__file__).read_bytes() != git("show", f"HEAD:{own_path}"):
        raise ValueError("commit the audit runner before executing the study")
    git("merge-base", "--is-ancestor", FREEZE, "HEAD")
    for name in ("expectations.md", "expectations.json"):
        relative = (HERE / name).relative_to(ROOT).as_posix()
        if (HERE / name).read_bytes() != git("show", f"{FREEZE}:{relative}"):
            raise ValueError("audit expectations changed after the freeze")
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    findings, pairs, phases, oracles, relations = [], [], [], [], []
    preserved = {path: digest((ROOT / path).read_bytes()) for path in frozen["preserved_sha256"]}
    if preserved != frozen["preserved_sha256"]:
        findings.append("historical artifact or source preservation failed")

    def relation(family, instance, observed, expected):
        relations.append(
            {
                "family": family,
                "instance": instance,
                "observed": rational(observed),
                "expected": rational(expected),
                "verdict": "PASS" if observed == expected else "REFUTED",
            }
        )

    history = None
    if not findings:
        scorer = load_scorer()
        config = frozen["packet_free"]
        by_parameters = {}
        for size, rate, overhead in product(
            config["base_payload_bytes"],
            config["rates_bytes_per_second"],
            config["fixed_overhead_ps"],
        ):
            pair = packet_free_pair(size, rate, overhead)
            by_parameters[size, rate, overhead] = pair
            drop = pair[0]["error"] - pair[1]["error"]
            expected = Fraction(rate * overhead, 2 * size * 10**12)
            labels = scorer.attribute_misses(
                classifier_rows(pair, config["classifier_degree_labels"])
            )
            should_label = expected > Fraction(config["attribution_drop_threshold"])
            if any((label == "packetization") != should_label for label in labels.values()):
                findings.append("historical classifier differs from frozen threshold behavior")
            if any(row["duration_ps"] < row["floor_ps"] for row in pair):
                findings.append("synthetic packet-free duration violates the serialization floor")
            if overhead == 0 and any(row["error"] != 0 for row in pair):
                findings.append("zero-overhead identity control failed")
            oracles.append(
                {"kind": "packet_free_error_drop", "residual": rational(drop - expected)}
            )
            pairs.append(
                {
                    "base_size_bytes": size,
                    "rate_bytes_per_second": rate,
                    "fixed_overhead_ps": overhead,
                    "packet_overhead_bytes": 0,
                    "rows": [
                        {
                            key: rational(value) if isinstance(value, Fraction) else value
                            for key, value in row.items()
                        }
                        for row in pair
                    ],
                    "historical_labels": labels,
                    "false_packet_identification": should_label,
                }
            )
            if overhead:
                relation(
                    "payload_scaling",
                    f"{size}-{rate}-{overhead}",
                    pair[0]["error"],
                    2 * pair[1]["error"],
                )
        for size, rate in product(config["base_payload_bytes"], config["rates_bytes_per_second"]):
            low = by_parameters[size, rate, 5_000_000][0]["error"]
            high = by_parameters[size, rate, 10_000_000][0]["error"]
            relation("overhead_scaling", f"{size}-{rate}", high, 2 * low)
        phase_config = frozen["clock_boundary"]
        phase_map = {}
        for degree, duration, stagger in product(
            phase_config["degrees"], phase_config["duration_ps"], phase_config["start_stagger_ps"]
        ):
            row = phase_case(degree, duration, stagger)
            phases.append(row)
            phase_map[degree, duration, stagger] = row
            expected = duration + (degree - 1) * stagger
            oracles.append(
                {
                    "kind": "common_phase_ps",
                    "residual": rational(Fraction(row["common_phase_ps"] - expected)),
                }
            )
            if row["common_phase_ps"] < duration or (
                stagger == 0 and row["common_phase_ps"] != duration
            ):
                findings.append("common-phase floor or zero-stagger identity failed")
        for degree in phase_config["degrees"]:
            low = (
                Fraction(phase_map[degree, 100_000_000, 20_000_000]["common_phase_ps"], 100_000_000)
                - 1
            )
            high = (
                Fraction(phase_map[degree, 200_000_000, 20_000_000]["common_phase_ps"], 200_000_000)
                - 1
            )
            relation("phase_duration_scaling", str(degree), low, 2 * high)
        for duration in phase_config["duration_ps"]:
            low = phase_map[2, duration, 20_000_000]["common_phase_ps"] - duration
            high = phase_map[3, duration, 20_000_000]["common_phase_ps"] - duration
            relation("phase_degree_scaling", str(duration), Fraction(high), Fraction(2 * low))
        history, historical_findings = historical_audit(frozen)
        findings.extend(historical_findings)
    if any(row["residual"]["exact"] != "0" for row in oracles):
        findings.append("independent exact-arithmetic oracle failed")
    if any(row["verdict"] != "PASS" for row in relations):
        findings.append("frozen exact relation failed")
    result = {
        "schema": "simllm-nvlink-measurement-boundary-result-v1",
        "verdict": "VOID" if findings else "PASS",
        "evidence_class": "synthetic_inference_counterexample",
        "hardware_capture": False,
        "expectations_commit": git("rev-parse", FREEZE).decode().strip(),
        "implementation_commit": git("rev-parse", "HEAD").decode().strip(),
        "source_sha256": digest(Path(__file__).read_bytes()),
        "preserved_sha256": preserved,
        "fatal_findings": findings,
        "packet_free_row_evaluation_count": 2 * len(pairs),
        "packet_free_distinct_configuration_count": len(
            {
                (row["size_bytes"], pair["rate_bytes_per_second"], pair["fixed_overhead_ps"])
                for pair in pairs
                for row in pair["rows"]
            }
        ),
        "phase_configuration_count": len(phases),
        "exact_oracle_count": len(oracles),
        "exact_oracles": oracles,
        "behavioral_family_count": len({row["family"] for row in relations}),
        "behavioral_instance_count": len(relations),
        "behavioral_score": None
        if findings
        else {"passed": len(relations), "total": len(relations)},
        "behavioral_relations": relations,
        "packet_free_pairs": pairs,
        "false_packet_identification_pair_count": sum(
            row["false_packet_identification"] for row in pairs
        ),
        "phase_cases": phases,
        "historical_audit": history,
        "closure_scope": "TRAF-91 inference correction only",
        "remaining_task": "TRAF-86",
    }
    (output_root / "summary.json").write_bytes(canonical(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "verdict",
                    "false_packet_identification_pair_count",
                    "exact_oracle_count",
                    "behavioral_family_count",
                    "behavioral_instance_count",
                    "fatal_findings",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
