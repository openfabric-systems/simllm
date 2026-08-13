"""Run the frozen HTSIM UEC validation-bound inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPECTED_SOURCE_COMMIT = "1ae1215758b96a52c1709a538204f5a73a05c5a9"
AUTHORSHIP_CONTENT_COMMIT = "896cc765aabce04b0707a42eacf8774275a5d771"
PLAN_NAMES = (
    "validate_uec_sender.txt",
    "validate_uec_rcv.txt",
    "validate_uec_both.txt",
    "validate_load_balancing_snd.txt",
    "validate_load_balancing_rcv.txt",
    "validate_load_balancing_failed_snd.txt",
    "validate_load_balancing_failed_rcv.txt",
    "validate_uec_connreuse.txt",
)
PLAN_COUNTS = (15, 15, 15, 10, 10, 9, 9, 12)
TARGETS = (
    ("validate_uec_sender.txt", "1024 node incast"),
    ("validate_uec_sender.txt", "3 to 1 incast with long running flow"),
    ("validate_uec_sender.txt", "Small permutation, INC (16 nodes)"),
    ("validate_uec_rcv.txt", "outcast incast"),
    ("validate_uec_rcv.txt", "Small permutation, INC (16 nodes)"),
    ("validate_uec_both.txt", "Small permutation, INC (16 nodes)"),
    (
        "validate_load_balancing_snd.txt",
        "Large permutation, oblivious, large queues (1024 nodes)",
    ),
    (
        "validate_load_balancing_rcv.txt",
        "Large permutation, oblivious, large queues (1024 nodes)",
    ),
    (
        "validate_load_balancing_failed_snd.txt",
        "Large permutation, oblivious (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_snd.txt",
        "Large permutation, oblivious, large queues (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_snd.txt",
        "Large permutation, bitmap (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_snd.txt",
        "Large permutation, bitmap, large queues (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_snd.txt",
        "Large permutation, REPS, 4SACK/packet (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_rcv.txt",
        "Large permutation, oblivious (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_rcv.txt",
        "Large permutation, oblivious, large queues (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_rcv.txt",
        "Large permutation, bitmap, large queues (1024 nodes), 8 core links running at 10% capacity",
    ),
    (
        "validate_load_balancing_failed_rcv.txt",
        "Large permutation, REPS, 1SACK/packet , large queues (1024 nodes), 8 core links running at 10% capacity",
    ),
)
EXPERIMENT_RE = re.compile(r"^!Experiment\s+(?P<name>.+?)\s*$")
TAIL_RE = re.compile(
    r"^\[(?P<status>PASS|FAIL)\] Tail FCT (?P<value>[0-9.]+) us .* target of (?P<target>[0-9.]+) us$"
)
PER_FLOW_RE = re.compile(
    r"^\[(?P<status>PASS|FAIL)\] FCT (?P<value>[0-9.]+) us for flow (?P<flow>\S+) .* target of (?P<target>[0-9.]+) us$"
)
SPREAD_RE = re.compile(
    r"^FCT Spread (?P<minimum>[0-9.]+) -> (?P<maximum>[0-9.]+) "
    r"ratio (?P<ratio>[0-9.]+)$"
)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_output(source_root: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=source_root,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return process.stdout


def _assert_source_commit(source_root: Path) -> str:
    observed = _git_output(source_root, "rev-parse", "HEAD").decode().strip()
    if observed != EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            "HTSIM_SOURCE_ROOT must be a detached worktree at "
            f"{EXPECTED_SOURCE_COMMIT}, observed {observed}"
        )
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
    )
    if symbolic.returncode == 0:
        reference = symbolic.stdout.decode("utf-8", errors="replace").strip()
        raise ValueError(f"HTSIM_SOURCE_ROOT must have detached HEAD, found {reference}")
    if symbolic.returncode != 1:
        detail = symbolic.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot verify detached HTSIM_SOURCE_ROOT: {detail}")
    return observed


def _split_plan_bytes(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split only LF and CRLF rows while retaining every terminator byte."""
    rows = []
    start = 0
    while True:
        newline = data.find(b"\n", start)
        if newline < 0:
            break
        if newline > start and data[newline - 1] == 13:
            body = data[start : newline - 1]
            terminator = b"\r\n"
        else:
            body = data[start:newline]
            terminator = b"\n"
        if b"\r" in body:
            raise ValueError("plan contains a bare carriage return")
        rows.append((body, terminator))
        start = newline + 1
    if start < len(data):
        body = data[start:]
        if b"\r" in body:
            raise ValueError("plan contains a bare carriage return")
        rows.append((body, b""))
    if not rows:
        raise ValueError("plan is empty")
    return rows


def _decode_plan_body(body: bytes) -> str:
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("plan row is not UTF-8") from error


def _quoted_directive(prefix: str, path: Path) -> bytes:
    return f"{prefix}{shlex.quote(str(path))}".encode()


def _project_plan_bytes(
    *, source: bytes, binary: Path, log_path: Path, datacenter: Path
) -> tuple[bytes, dict[str, Any], tuple[Path, ...]]:
    """Project authorized path bodies without normalizing any source row."""
    projected: list[bytes] = []
    references: list[Path] = []
    active_experiment = False
    binary_seen = False
    counts = {"binary": 0, "log": 0, "matrix": 0, "topology": 0}
    for body, terminator in _split_plan_bytes(source):
        text = _decode_plan_body(body)
        stripped = text.strip()
        if stripped.startswith("!Param -o"):
            raise ValueError("source plan already contains !Param -o")
        if stripped and not stripped.startswith("!") and not stripped.startswith("#"):
            if active_experiment and not binary_seen:
                raise ValueError("experiment has no Binary row")
            active_experiment = True
            binary_seen = False
            resolved = (datacenter / stripped).resolve()
            projected.append(str(resolved).encode("utf-8") + terminator)
            references.append(resolved)
            counts["matrix"] += 1
            continue
        if stripped.startswith("!") and not active_experiment:
            raise ValueError("plan directive appears before a traffic matrix")
        if stripped.startswith("!Binary"):
            if not stripped.startswith("!Binary ") or not stripped.removeprefix(
                "!Binary "
            ).strip():
                raise ValueError("malformed or empty Binary row")
            if binary_seen:
                raise ValueError("experiment has duplicate Binary rows")
            if not terminator:
                raise ValueError("unterminated Binary row cannot receive a log row")
            projected.append(_quoted_directive("!Binary ", binary) + terminator)
            projected.append(_quoted_directive("!Param -o ", log_path) + terminator)
            binary_seen = True
            counts["binary"] += 1
            counts["log"] += 1
            continue
        if stripped.startswith("!Param -topo"):
            if not stripped.startswith("!Param -topo "):
                raise ValueError("malformed topology row")
            topology = stripped.removeprefix("!Param -topo ").strip()
            if not topology:
                raise ValueError("empty topology row")
            resolved = (datacenter / topology).resolve()
            projected.append(
                _quoted_directive("!Param -topo ", resolved) + terminator
            )
            references.append(resolved)
            counts["topology"] += 1
            continue
        projected.append(body + terminator)
    if active_experiment and not binary_seen:
        raise ValueError("final experiment has no Binary row")
    if counts["matrix"] != counts["binary"] or counts["binary"] != counts["log"]:
        raise ValueError(f"plan structure counts disagree: {counts}")
    projected_bytes = b"".join(projected)
    ledger = {
        "binary_replacements": counts["binary"],
        "exact_round_trip": True,
        "log_insertions": counts["log"],
        "matrix_replacements": counts["matrix"],
        "projected_final_lf": projected_bytes.endswith(b"\n"),
        "projected_sha256": _sha256(projected_bytes),
        "source_final_lf": source.endswith(b"\n"),
        "source_sha256": _sha256(source),
        "topology_replacements": counts["topology"],
    }
    if ledger["source_final_lf"] != ledger["projected_final_lf"]:
        raise ValueError("projector changed the final-LF state")
    return projected_bytes, ledger, tuple(references)


def _audit_projection(
    *, source: bytes, projected: bytes, binary: Path, log_path: Path, datacenter: Path
) -> dict[str, Any]:
    """Independently synchronize source and projected rows and reject drift."""
    source_rows = _split_plan_bytes(source)
    projected_rows = _split_plan_bytes(projected)
    projected_index = 0
    reconstructed: list[bytes] = []
    counts = {"binary": 0, "log": 0, "matrix": 0, "topology": 0}

    def consume(expected_body: bytes, expected_terminator: bytes) -> None:
        nonlocal projected_index
        if projected_index >= len(projected_rows):
            raise ValueError("projected plan is missing a row")
        actual_body, actual_terminator = projected_rows[projected_index]
        projected_index += 1
        if actual_body != expected_body or actual_terminator != expected_terminator:
            raise ValueError(
                f"unauthorized projected row change at row {projected_index}"
            )

    for source_body, source_terminator in source_rows:
        reconstructed.append(source_body + source_terminator)
        text = _decode_plan_body(source_body)
        stripped = text.strip()
        if stripped.startswith("!Param -o"):
            raise ValueError("source plan already contains !Param -o")
        if stripped and not stripped.startswith("!") and not stripped.startswith("#"):
            resolved = (datacenter / stripped).resolve()
            consume(str(resolved).encode("utf-8"), source_terminator)
            counts["matrix"] += 1
            continue
        if stripped.startswith("!Binary"):
            if not source_terminator:
                raise ValueError("unterminated Binary row cannot receive a log row")
            consume(_quoted_directive("!Binary ", binary), source_terminator)
            consume(_quoted_directive("!Param -o ", log_path), source_terminator)
            counts["binary"] += 1
            counts["log"] += 1
            continue
        if stripped.startswith("!Param -topo"):
            if not stripped.startswith("!Param -topo "):
                raise ValueError("malformed topology row")
            topology = stripped.removeprefix("!Param -topo ").strip()
            if not topology:
                raise ValueError("empty topology row")
            resolved = (datacenter / topology).resolve()
            consume(_quoted_directive("!Param -topo ", resolved), source_terminator)
            counts["topology"] += 1
            continue
        consume(source_body, source_terminator)
    if projected_index != len(projected_rows):
        raise ValueError("projected plan contains an extra row")
    if b"".join(reconstructed) != source:
        raise ValueError("projection audit did not reconstruct the source exactly")
    if source.endswith(b"\n") != projected.endswith(b"\n"):
        raise ValueError("projection audit found a final-LF change")
    return {
        "binary_replacements": counts["binary"],
        "exact_round_trip": True,
        "log_insertions": counts["log"],
        "matrix_replacements": counts["matrix"],
        "projected_final_lf": projected.endswith(b"\n"),
        "projected_sha256": _sha256(projected),
        "source_final_lf": source.endswith(b"\n"),
        "source_sha256": _sha256(source),
        "topology_replacements": counts["topology"],
    }


def _require_matching_projection_ledgers(
    projected: dict[str, Any], audited: dict[str, Any]
) -> None:
    if projected != audited:
        raise ValueError(
            f"projector and independent audit ledgers disagree: {projected} != {audited}"
        )


def _source_root() -> Path:
    raw = os.environ.get("HTSIM_SOURCE_ROOT")
    if not raw:
        raise ValueError("HTSIM_SOURCE_ROOT must name the backend source checkout")
    root = Path(raw).resolve()
    required = (
        root / "htsim" / "sim" / "datacenter" / "commit_check.sh",
        root / "htsim" / "sim" / "datacenter" / "validate.py",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(f"HTSIM_SOURCE_ROOT is not an HTSIM checkout: {root}")
    return root


def _candidates(values: list[str]) -> dict[str, Path]:
    candidates: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("each --candidate must be LABEL=PATH")
        if label in candidates or not re.fullmatch(r"[a-z0-9_-]+", label):
            raise ValueError(f"invalid or duplicate candidate label: {label!r}")
        candidates[label] = Path(raw_path).resolve()
    if not candidates:
        raise ValueError("at least one --candidate is required")
    return candidates


def _experiment_names(plan: Path) -> list[str]:
    names = []
    for line in plan.read_text(encoding="utf-8").splitlines():
        match = EXPERIMENT_RE.match(line)
        if match is not None:
            names.append(match.group("name"))
    return names


def _inventory(datacenter: Path) -> dict[str, list[str]]:
    inventory = {name: _experiment_names(datacenter / name) for name in PLAN_NAMES}
    actual_counts = tuple(len(inventory[name]) for name in PLAN_NAMES)
    if actual_counts != PLAN_COUNTS:
        raise ValueError(f"plan counts changed: expected {PLAN_COUNTS}, observed {actual_counts}")
    for plan, experiment in TARGETS:
        if inventory[plan].count(experiment) != 1:
            raise ValueError(f"target identity is not unique: {plan}: {experiment}")
    return inventory


def _content_manifest(source_root: Path) -> tuple[list[dict[str, Any]], tuple[Path, ...]]:
    datacenter = source_root / "htsim" / "sim" / "datacenter"
    plan_paths = tuple(datacenter / name for name in PLAN_NAMES)
    referenced: set[Path] = set()
    for plan_path in plan_paths:
        _, _, plan_references = _project_plan_bytes(
            source=plan_path.read_bytes(),
            binary=source_root / "projection-check" / "htsim_uec",
            log_path=source_root / "projection-check" / f"{plan_path.name}.logout",
            datacenter=datacenter,
        )
        referenced.update(plan_references)
    validator = datacenter / "validate.py"
    authored_paths = tuple(sorted((*plan_paths, *referenced), key=lambda path: str(path)))
    rows = []
    for path in authored_paths:
        if not path.is_file():
            raise ValueError(f"referenced input is absent: {path}")
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError as error:
            raise ValueError(f"referenced input escapes HTSIM_SOURCE_ROOT: {path}") from error
        current = path.read_bytes()
        authored = _git_output(
            source_root, "show", f"{AUTHORSHIP_CONTENT_COMMIT}:{relative}"
        )
        if current != authored:
            raise ValueError(
                f"{relative} differs from the bound-authorship content at "
                f"{AUTHORSHIP_CONTENT_COMMIT}"
            )
        rows.append(
            {
                "authorship_sha256": _sha256(authored),
                "kind": "plan" if path in plan_paths else "input",
                "path": relative,
                "sha256": _sha256(current),
            }
        )
    validator_relative = validator.relative_to(source_root).as_posix()
    validator_bytes = validator.read_bytes()
    committed_validator = _git_output(
        source_root, "show", f"{EXPECTED_SOURCE_COMMIT}:{validator_relative}"
    )
    if validator_bytes != committed_validator:
        raise ValueError(f"{validator_relative} differs from {EXPECTED_SOURCE_COMMIT}")
    rows.append(
        {
            "commit_sha256": _sha256(committed_validator),
            "kind": "comparator",
            "path": validator_relative,
            "sha256": _sha256(validator_bytes),
        }
    )
    all_paths = (*authored_paths, validator)
    return rows, all_paths


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths}


def _assert_snapshot(snapshot: dict[Path, bytes], *, boundary: str) -> None:
    for path, expected in snapshot.items():
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise ValueError(f"{boundary}: cannot reread {path}: {error}") from error
        if observed != expected:
            raise ValueError(f"{boundary}: input changed during the run: {path}")


def _check_only(*, source_root: Path, out: Path, candidates: dict[str, Path]) -> None:
    datacenter = source_root / "htsim" / "sim" / "datacenter"
    inventory = _inventory(datacenter)
    content_manifest, content_paths = _content_manifest(source_root)
    validator = datacenter / "validate.py"
    content_snapshot = _snapshot(content_paths)
    dry_runs = []
    for name in PLAN_NAMES:
        _assert_snapshot(content_snapshot, boundary=f"check-only before {name}")
        run = subprocess.run(
            [sys.executable, str(validator), "-dryrun", str(datacenter / name)],
            cwd=datacenter,
            check=False,
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            raise ValueError(f"validator dry-run failed for {name}: {run.stdout}{run.stderr}")
        _assert_snapshot(content_snapshot, boundary=f"check-only after {name}")
        dry_runs.append(
            {
                "command_count": run.stdout.count("\nRunning "),
                "plan": name,
                "returncode": run.returncode,
            }
        )
    projection_audits = []
    for label, binary in candidates.items():
        for name, expected_count in zip(PLAN_NAMES, PLAN_COUNTS):
            source = (datacenter / name).read_bytes()
            log_path = out / label / "run" / f"{name}.logout"
            projected, projection_ledger, _ = _project_plan_bytes(
                source=source,
                binary=binary,
                log_path=log_path,
                datacenter=datacenter,
            )
            audited = _audit_projection(
                source=source,
                projected=projected,
                binary=binary,
                log_path=log_path,
                datacenter=datacenter,
            )
            _require_matching_projection_ledgers(projection_ledger, audited)
            if audited["matrix_replacements"] != expected_count:
                raise ValueError(
                    f"{name}: expected {expected_count} projected experiments, "
                    f"observed {audited['matrix_replacements']}"
                )
            projection_audits.append({"candidate": label, "plan": name, **audited})
    _assert_snapshot(content_snapshot, boundary="check-only completion")
    _assert_source_commit(source_root)
    plan = {
        "artifacts_created": False,
        "candidate_labels": sorted(candidates),
        "candidate_paths_required_only_for_measured_run": True,
        "content_manifest": content_manifest,
        "dry_runs": dry_runs,
        "experiments": sum(len(names) for names in inventory.values()),
        "mutant": {
            "file": "validate_uec_connreuse.txt",
            "from_us": 18,
            "occurrence": 1,
            "to_us": 5,
        },
        "out": str(out),
        "plans": list(PLAN_NAMES),
        "projection_audits": projection_audits,
        "source_root": str(source_root),
        "targets": len(TARGETS),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def _parse_observations(stdout: str) -> list[dict[str, Any]]:
    observations = []
    current: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if line.startswith("Experiment: "):
            current = {
                "experiment": line.removeprefix("Experiment: "),
                "per_flow": [],
            }
            observations.append(current)
            continue
        if current is None:
            continue
        tail = TAIL_RE.match(line)
        if tail is not None:
            current["tail"] = {
                "passed": tail.group("status") == "PASS",
                "target_us": float(tail.group("target")),
                "value_us": float(tail.group("value")),
            }
            continue
        per_flow = PER_FLOW_RE.match(line)
        if per_flow is not None:
            current["per_flow"].append(
                {
                    "flow": per_flow.group("flow"),
                    "passed": per_flow.group("status") == "PASS",
                    "target_us": float(per_flow.group("target")),
                    "value_us": float(per_flow.group("value")),
                }
            )
        spread = SPREAD_RE.match(line)
        if spread is not None:
            current["spread"] = {
                "maximum_us": float(spread.group("maximum")),
                "minimum_us": float(spread.group("minimum")),
                "ratio": float(spread.group("ratio")),
            }
        if line.startswith("[PASS] Connection count "):
            current["connection_count_passed"] = True
        elif line.startswith("[FAIL] Total connections "):
            current["connection_count_passed"] = False
        elif line.startswith("Summary: "):
            current["packet_summary_present"] = True
    return observations


def _run_candidate(
    *,
    label: str,
    binary: Path,
    source_root: Path,
    out: Path,
    content_snapshot: dict[Path, bytes],
) -> dict[str, Any]:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"candidate binary is absent or not executable: {binary}")
    binary_bytes = binary.read_bytes()
    datacenter = source_root / "htsim" / "sim" / "datacenter"
    validator = datacenter / "validate.py"
    candidate_out = out / label
    candidate_out.mkdir()
    plans_out = candidate_out / "plans"
    plans_out.mkdir()
    run_out = candidate_out / "run"
    run_out.mkdir()
    plan_rows = []
    for plan_name in PLAN_NAMES:
        source_path = datacenter / plan_name
        source = content_snapshot[source_path]
        plan_out = plans_out / plan_name
        log_path = run_out / f"{plan_name}.logout"
        projected, projection_ledger, references = _project_plan_bytes(
            source=source,
            binary=binary,
            log_path=log_path,
            datacenter=datacenter,
        )
        for reference in references:
            if reference not in content_snapshot:
                raise ValueError(f"unregistered projected input: {reference}")
        audited_before = _audit_projection(
            source=source,
            projected=projected,
            binary=binary,
            log_path=log_path,
            datacenter=datacenter,
        )
        _require_matching_projection_ledgers(projection_ledger, audited_before)
        plan_out.write_bytes(projected)
        if plan_out.read_bytes() != projected:
            raise ValueError(f"projected plan changed while writing: {plan_out}")
        if binary.read_bytes() != binary_bytes:
            raise ValueError(f"candidate binary changed before {label}/{plan_name}")
        _assert_snapshot(content_snapshot, boundary=f"before {label}/{plan_name}")
        env = os.environ.copy()
        env["HTSIM_TRACE_FLOW_COMPLETIONS"] = "1"
        process = subprocess.run(
            [sys.executable, str(validator), str(plan_out)],
            cwd=run_out,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        _assert_snapshot(content_snapshot, boundary=f"after {label}/{plan_name}")
        if binary.read_bytes() != binary_bytes:
            raise ValueError(f"candidate binary changed during {label}/{plan_name}")
        projected_after = plan_out.read_bytes()
        if projected_after != projected:
            raise ValueError(f"projected plan changed during the run: {plan_out}")
        audited_after = _audit_projection(
            source=source,
            projected=projected_after,
            binary=binary,
            log_path=log_path,
            datacenter=datacenter,
        )
        _require_matching_projection_ledgers(projection_ledger, audited_after)
        stdout_path = candidate_out / f"{plan_name}.stdout"
        stderr_path = candidate_out / f"{plan_name}.stderr"
        stdout_path.write_text(process.stdout, encoding="utf-8", newline="\n")
        stderr_path.write_text(process.stderr, encoding="utf-8", newline="\n")
        observations = _parse_observations(process.stdout)
        plan_rows.append(
            {
                "observations": observations,
                "plan": plan_name,
                "projection_audit_after": audited_after,
                "projection_audit_before": audited_before,
                "projected_plan_sha256": _sha256(projected_after),
                "returncode": process.returncode,
                "source_plan_sha256": _sha256(source),
                "stderr_sha256": _sha256(process.stderr.encode()),
                "stdout_sha256": _sha256(process.stdout.encode()),
            }
        )
    return {
        "binary": str(binary),
        "binary_sha256": _sha256(binary_bytes),
        "label": label,
        "plans": plan_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source_root = _source_root()
    observed_commit = _assert_source_commit(source_root)
    candidates = _candidates(args.candidate)
    if args.check_only:
        _check_only(source_root=source_root, out=args.out, candidates=candidates)
        return
    if args.out.exists():
        parser.error("--out must not exist")
    _inventory(source_root / "htsim" / "sim" / "datacenter")
    content_manifest, content_paths = _content_manifest(source_root)
    content_snapshot = _snapshot(content_paths)
    args.out.mkdir(parents=True)
    rows = [
        _run_candidate(
            label=label,
            binary=binary,
            source_root=source_root,
            out=args.out,
            content_snapshot=content_snapshot,
        )
        for label, binary in candidates.items()
    ]
    _assert_snapshot(content_snapshot, boundary="study completion")
    if _assert_source_commit(source_root) != observed_commit:
        raise ValueError("source commit changed during the study")
    final_manifest, _ = _content_manifest(source_root)
    if final_manifest != content_manifest:
        raise ValueError("committed input manifest changed during the study")
    summary = {
        "authored_against_htsim_commit": "1f2c124c9738edcfa0f6044b4667c230e75a542c",
        "candidates": rows,
        "content_manifest": content_manifest,
        "observed_htsim_commit": observed_commit,
        "plan_count": len(PLAN_NAMES),
        "schema": "simllm-htsim-uec-bounds-study-v1",
        "target_count": len(TARGETS),
        "total_experiments_per_candidate": sum(PLAN_COUNTS),
    }
    (args.out / "summary.json").write_bytes(_canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
