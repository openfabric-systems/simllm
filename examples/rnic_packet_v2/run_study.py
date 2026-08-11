"""Run the frozen NetworkPort ABI v2 packet-event study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rnic_live_v1.tier_a_acceptance import check_observations

EXPECTATIONS_PATH = Path(__file__).with_name("expectations.json")
TIER_A_EXPECTATIONS = (
    REPO_ROOT / "examples" / "rnic_live_v1" / "tier_a_expectations.json"
)
SIMLLM_BASE_COMMIT = "b74629b4b4da1addda9ff21226cfabf5c09aad87"
HTSIM_BASE_COMMIT = "edb28c3015c173b4251abc5858c587df325e1ebc"
SIMLLM_EXPECTATION_COMMIT = "506f87af93687ccf0df85f6b5307b71a20ed3762"
HTSIM_EXPECTATION_COMMIT = "6ece8bdd908496dadfd4df809e3a4eb660d6cc26"
FIX_ROUND_EXPECTATION_COMMIT = "07521786020e41f56196d13718c62169d47ad70d"
PAYLOAD_BYTES = (4096, 1048576)
LINK_RATE_GBPS = (200, 400)
DOORBELL_SERVICE_PS = (0, 1000)
PACKET_WIRE_QUANTUM_BYTES = 4096
PACKET_EVENT_FIELDS = (
    "attempt_token",
    "extent_token",
    "wqe_id",
    "event_kind",
    "event_time_ps",
    "extent_index",
    "packet_index",
    "transmission_attempt",
    "payload_offset_bytes",
    "payload_bytes",
    "wire_bytes",
    "packet_kind",
)
PACKET_EVENT_KINDS = (
    "packet_tx_started",
    "packet_tx_finished",
    "packet_rx_arrived",
    "delivered",
    "dropped",
)
CONTROL_EVENT_KINDS = (
    "ecn_marked",
    "cnp_received",
    "eligibility_updated",
    "rate_updated",
    "pfc_frame_submitted",
    "pfc_paused",
    "pfc_resumed",
    "link_state_changed",
)
V1_ARTIFACT_SHA256 = {
    "raw_observations.json": (
        "37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a"
    ),
    "summary.json": (
        "00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004"
    ),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wave3_root() -> Path:
    configured = os.environ.get("SIMLLM_WAVE3_RUN_ROOT")
    if not configured:
        raise RuntimeError(
            "SIMLLM_WAVE3_RUN_ROOT must name the external wave-3 run root"
        )
    return Path(configured).resolve()


def _git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _validate_registry(
    htsim_source: Path, v1_reference_dir: Path, out: Path
) -> None:
    expectations = _load_json(EXPECTATIONS_PATH)
    if expectations["schema"] != "simllm-rnic-packet-v2-expectations-v1":
        raise AssertionError("packet-v2 expectation schema drifted")
    if expectations["simllm_base_commit"] != SIMLLM_BASE_COMMIT:
        raise AssertionError("SimLLM audit commit drifted")
    if expectations["htsim_base_commit"] != HTSIM_BASE_COMMIT:
        raise AssertionError("htsim audit commit drifted")
    if not re.fullmatch(r"[0-9a-f]{40}", SIMLLM_BASE_COMMIT):
        raise AssertionError("SimLLM audit commit must be a full hash")
    if not re.fullmatch(r"[0-9a-f]{40}", HTSIM_BASE_COMMIT):
        raise AssertionError("htsim audit commit must be a full hash")
    subprocess.run(
        ["git", "cat-file", "-e", f"{SIMLLM_BASE_COMMIT}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "cat-file", "-e", f"{HTSIM_BASE_COMMIT}^{{commit}}"],
        cwd=htsim_source,
        check=True,
        capture_output=True,
    )
    cells = set(product(PAYLOAD_BYTES, LINK_RATE_GBPS, DOORBELL_SERVICE_PS))
    if len(cells) != 8:
        raise AssertionError("packet-v2 single-WQE registry must have eight cells")
    if len(set(product(LINK_RATE_GBPS, DOORBELL_SERVICE_PS))) != 4:
        raise AssertionError("packet-v2 FIFO registry must have four cells")
    if expectations["packet_instances"] != {
        "tx_d_additivity": 4,
        "rx_d_additivity": 4,
        "multi_packet_inverse_rate_span": 2,
    }:
        raise AssertionError("packet-v2 scored family counts drifted")
    if tuple(expectations["packet_event_fields"]) != PACKET_EVENT_FIELDS:
        raise AssertionError("packet-v2 event field registry drifted")
    if tuple(expectations["packet_event_kinds"]) != PACKET_EVENT_KINDS:
        raise AssertionError("packet-v2 packet-event kinds drifted")
    if tuple(expectations["control_event_kinds"]) != CONTROL_EVENT_KINDS:
        raise AssertionError("packet-v2 control-event kinds drifted")
    for name, expected in V1_ARTIFACT_SHA256.items():
        actual = _digest(v1_reference_dir / name)
        if actual != expected:
            raise AssertionError(
                f"accepted ABI v1 reference drifted for {name}: {actual}"
            )
    try:
        out.resolve().relative_to(_wave3_root())
    except ValueError as error:
        raise ValueError(
            "packet-v2 output must remain under SIMLLM_WAVE3_RUN_ROOT"
        ) from error


def _producer(build_dir: Path) -> Path:
    candidates = (
        build_dir / "htsim_rnic_tier_a",
        build_dir / "htsim_rnic_tier_a.exe",
        build_dir / "Release" / "htsim_rnic_tier_a",
        build_dir / "Release" / "htsim_rnic_tier_a.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("htsim_rnic_tier_a was not produced by the build")


def _build(htsim_source: Path, build_dir: Path) -> tuple[Path, dict[str, int]]:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(htsim_source / "htsim" / "sim"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DENABLE_TESTS=ON",
            "-DHTSIM_ENABLE_SIMLLM_RNIC=ON",
            f"-DSIMLLM_REPOSITORY_ROOT={REPO_ROOT}",
            "-DHTSIM_CREATE_SOURCE_SYMLINKS=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        check=True,
    )
    completed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    match = re.search(
        r"100% tests passed, (\d+) tests failed out of (\d+)",
        completed.stdout,
    )
    if match is None:
        raise RuntimeError("could not parse the complete htsim CTest summary")
    failed = int(match.group(1))
    total = int(match.group(2))
    return _producer(build_dir), {
        "passed": total - failed,
        "failed": failed,
        "total": total,
    }


def _run_v1(producer: Path, run_dir: Path) -> dict[str, Any]:
    local_producer = run_dir / "build" / producer.name
    local_producer.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(producer, local_producer)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples/rnic_live_v1/tier_a_acceptance.py"),
            "--factory",
            "htsim",
            "--producer",
            str(local_producer),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
    )
    return {
        name: _digest(run_dir / name) for name in V1_ARTIFACT_SHA256
    }


def _run_v2(producer: Path, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    raw = run_dir / "raw_observations.json"
    subprocess.run(
        [
            str(producer),
            "--factory",
            "htsim",
            "--network-abi-version",
            "2",
            "--expectations",
            str(TIER_A_EXPECTATIONS),
            "--observations",
            str(raw),
        ],
        check=True,
    )
    observations = _load_json(raw)
    return _validate_v2_observations(observations)


def _v1_projection(value: Any) -> Any:
    if isinstance(value, list):
        return [_v1_projection(item) for item in value]
    if not isinstance(value, dict):
        return value
    removed = {
        "network_abi_version",
        "packet_events",
        "first_packet_at_ps",
        "last_packet_at_ps",
        "first_rx_at_ps",
        "last_rx_at_ps",
    }
    projected = {
        key: _v1_projection(item)
        for key, item in value.items()
        if key not in removed
    }
    if projected.get("schema") == "simllm-rnic-tier-a-observations-v2":
        projected["schema"] = "simllm-rnic-tier-a-observations-v1"
    return projected


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(cell["payload_bytes"]),
        int(cell["link_rate_gbps"]),
        int(cell["doorbell_service_ps"]),
    )


def _packet_times(
    cell: dict[str, Any], ordinal: int, kind: str
) -> list[int]:
    wqe = cell["wqes"][ordinal]
    wqe_id = int(wqe["wqe_id"])
    rows = [
        row
        for row in cell["port"]["packet_events"]
        if int(row["wqe_id"]) == wqe_id and row["event_kind"] == kind
    ]
    return [int(row["event_time_ps"]) for row in rows]


def _validate_packet_cell(cell: dict[str, Any]) -> None:
    payload = int(cell["payload_bytes"])
    rate = int(cell["link_rate_gbps"])
    doorbell = int(cell["doorbell_service_ps"])
    expected_packets = (payload - 1) // PACKET_WIRE_QUANTUM_BYTES + 1
    events = cell["port"]["packet_events"]
    for row in events:
        if tuple(row) != PACKET_EVENT_FIELDS:
            raise AssertionError("packet-v2 raw event schema drifted")
    for ordinal, wqe in enumerate(cell["wqes"]):
        tx_starts = _packet_times(cell, ordinal, "packet_tx_started")
        tx_finishes = _packet_times(cell, ordinal, "packet_tx_finished")
        rx_arrivals = _packet_times(cell, ordinal, "packet_rx_arrived")
        terminals = _packet_times(cell, ordinal, "delivered")
        if not all(
            len(values) == expected_packets
            for values in (tx_starts, tx_finishes, rx_arrivals, terminals)
        ):
            raise AssertionError("packet-v2 event cardinality is not conserved")
        if tx_starts != sorted(tx_starts):
            raise AssertionError("packet-v2 TX starts are not monotonic")
        if tx_finishes != rx_arrivals or rx_arrivals != terminals:
            raise AssertionError("zero-propagation packet boundaries disagree")
        if any(tx_finishes[index] != tx_starts[index + 1]
               for index in range(expected_packets - 1)):
            raise AssertionError("packet-v2 serializer has a gap or overlap")
        if int(wqe["first_packet_at_ps"]) != min(tx_starts):
            raise AssertionError("first packet did not derive from TX start")
        if int(wqe["last_packet_at_ps"]) != max(tx_starts):
            raise AssertionError("last packet did not derive from TX start")
        if int(wqe["first_rx_at_ps"]) != min(rx_arrivals):
            raise AssertionError("first RX did not derive from RX arrival")
        if int(wqe["last_rx_at_ps"]) != max(rx_arrivals):
            raise AssertionError("last RX did not derive from RX arrival")
        if tx_starts[0] < doorbell:
            raise AssertionError("packet-v2 TX issue predates native eligibility")
        total_payload = sum(
            int(row["payload_bytes"])
            for row in events
            if int(row["wqe_id"]) == int(wqe["wqe_id"])
            and row["event_kind"] == "packet_tx_started"
        )
        if total_payload != payload:
            raise AssertionError("packet-v2 payload ledger does not close")
        packet_issue_base = int(wqe["port_tx_at_ps"])
        if tx_starts[0] != packet_issue_base:
            raise AssertionError("packet-v2 first TX issue is not exact")
        expected_last_tx = packet_issue_base + (
            (payload - min(payload, PACKET_WIRE_QUANTUM_BYTES))
            * 8
            * 1000
            // rate
        )
        if tx_starts[-1] != expected_last_tx:
            raise AssertionError("packet-v2 last TX issue is not exact")
        expected_terminal = (
            packet_issue_base + payload * 8 * 1000 // rate
        )
        if rx_arrivals[-1] != expected_terminal:
            raise AssertionError("packet-v2 final RX boundary is not exact")


def _validate_v2_observations(observations: dict[str, Any]) -> dict[str, Any]:
    if observations.get("schema") != "simllm-rnic-tier-a-observations-v2":
        raise AssertionError("ABI v2 producer returned the wrong schema")
    if observations.get("network_abi_version") != 2:
        raise AssertionError("ABI v2 producer returned the wrong version")
    single = {
        _cell_key(cell): cell for cell in observations["single_wqe"]
    }

    # Fix-round 1 makes the packet relations operationally independent of the
    # exact oracles: evaluate the raw projections before any entailing pin.
    tx_additive = 0
    rx_additive = 0
    inverse_span = 0
    for payload, rate in product(PAYLOAD_BYTES, LINK_RATE_GBPS):
        low = single[(payload, rate, 0)]["wqes"][0]
        high = single[(payload, rate, 1000)]["wqes"][0]
        if not all(
            int(high[field]) - int(low[field]) == 1000
            for field in ("first_packet_at_ps", "last_packet_at_ps")
        ):
            raise AssertionError("packet-v2 TX D-additivity failed")
        tx_additive += 1
        if not all(
            int(high[field]) - int(low[field]) == 1000
            for field in ("first_rx_at_ps", "last_rx_at_ps")
        ):
            raise AssertionError("packet-v2 RX D-additivity failed")
        rx_additive += 1
    for doorbell in DOORBELL_SERVICE_PS:
        slow = single[(1048576, 200, doorbell)]["wqes"][0]
        fast = single[(1048576, 400, doorbell)]["wqes"][0]
        slow_tx = int(slow["last_packet_at_ps"]) - int(
            slow["first_packet_at_ps"]
        )
        fast_tx = int(fast["last_packet_at_ps"]) - int(
            fast["first_packet_at_ps"]
        )
        slow_rx = int(slow["last_rx_at_ps"]) - int(slow["first_rx_at_ps"])
        fast_rx = int(fast["last_rx_at_ps"]) - int(fast["first_rx_at_ps"])
        if slow_tx != 2 * fast_tx or slow_rx != 2 * fast_rx:
            raise AssertionError("packet-v2 inverse-rate packet span failed")
        inverse_span += 1

    tier_a_expectations = _load_json(TIER_A_EXPECTATIONS)
    inherited = check_observations(
        _v1_projection(observations), tier_a_expectations, "htsim"
    )
    fifo = observations["fifo"]
    for cell in [*single.values(), *fifo]:
        _validate_packet_cell(cell)

    mutant = copy.deepcopy(next(iter(single.values())))
    mutant["port"]["packet_events"] = [
        row
        for row in mutant["port"]["packet_events"]
        if row["event_kind"] != "packet_tx_started"
    ]
    mutant_rejected = False
    try:
        _validate_packet_cell(mutant)
    except (AssertionError, ValueError):
        mutant_rejected = True
    if not mutant_rejected:
        raise AssertionError("packet-v2 missing-TX-event mutant was accepted")

    return {
        "inherited_tier_a": inherited.as_dict(),
        "packet_families": {
            "tx_d_additivity": {"passed": tx_additive, "total": 4},
            "rx_d_additivity": {"passed": rx_additive, "total": 4},
            "multi_packet_inverse_rate_span": {
                "passed": inverse_span,
                "total": 2,
            },
        },
        "evidence_accounting": {
            "correction": "post_specified_fix_round_1",
            "original_oracle_first_packet_family": {
                "classification": "withdrawn_as_entailed",
                "independently_scored_relations": 0,
                "reported_relations": 10,
            },
            "fix_round_1_packet_family": {
                "evaluation": "raw_observations_before_exact_oracles",
                "independently_scored_relations": 10,
                "scored_surface": {
                    "multi_packet_inverse_rate_span": 2,
                    "rx_d_additivity": 4,
                    "tx_d_additivity": 4,
                },
            },
            "exact_oracles": "fatal_unscored",
        },
        "missing_tx_event_mutant_rejected": mutant_rejected,
    }


def _run(
    htsim_source: Path, v1_reference_dir: Path, out: Path
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    producer, ctest = _build(htsim_source, out / "build")
    v1_digests = _run_v1(producer, out / "abi-v1")
    v1_identity = {
        name: digest == V1_ARTIFACT_SHA256[name]
        for name, digest in v1_digests.items()
    }
    if not all(v1_identity.values()):
        raise AssertionError("NetworkPort ABI v1 Tier A bytes changed")
    v2 = _run_v2(producer, out / "abi-v2")
    report = {
        "schema": "simllm-rnic-packet-v2-results-v1",
        "simllm_revision": _git_commit(REPO_ROOT),
        "htsim_revision": _git_commit(htsim_source),
        "expectation_commits": {
            "simllm": SIMLLM_EXPECTATION_COMMIT,
            "htsim": HTSIM_EXPECTATION_COMMIT,
            "simllm_fix_round_1": FIX_ROUND_EXPECTATION_COMMIT,
        },
        "ctest": ctest,
        "abi_v1_artifact_identity": v1_identity,
        "abi_v2": v2,
        "genuine_risk": {
            "inherited_tier_a": {"plausible_failures": 12, "relations": 12},
            "packet_families": {
                "evaluation": "raw_observations_before_exact_oracles",
                "plausible_failures": 10,
                "relations": 10,
            },
            "abi_v1_identity": {"plausible_failures": 2, "relations": 2},
        },
        "v1_reference_sha256": {
            name: _digest(v1_reference_dir / name)
            for name in V1_ARTIFACT_SHA256
        },
    }
    (out / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--htsim-source", type=Path, required=True)
    parser.add_argument("--v1-reference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    _validate_registry(
        arguments.htsim_source.resolve(),
        arguments.v1_reference_dir.resolve(),
        arguments.out.resolve(),
    )
    if arguments.check_only:
        print("packet-v2 study registry check passed; no artifacts were produced")
        return
    report = _run(
        arguments.htsim_source.resolve(),
        arguments.v1_reference_dir.resolve(),
        arguments.out.resolve(),
    )
    packet = report["abi_v2"]["packet_families"]
    passed = sum(value["passed"] for value in packet.values())
    total = sum(value["total"] for value in packet.values())
    print(f"packet-v2 study passed {passed}/{total} new packet relations")


if __name__ == "__main__":
    main()
