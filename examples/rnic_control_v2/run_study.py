"""Run the frozen physical transport-control producer study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.rnic_packet_v2 import run_study as packet_study
from simllm.backends.rnic_records import (
    BypassArtifacts,
    assert_bypass_artifact_identity,
    canonical_bypass_parameters,
)

EXPECTATIONS_PATH = Path(__file__).with_name("expectations.json")
SIMLLM_BASE_COMMIT = "90ada43070adb3b1e624b6819aff34d8620e8571"
HTSIM_BASE_COMMIT = "4885c647eecdfdf81479d1df052223c016ad086b"
EXPECTATION_COMMIT = "51af85937d6b1d3c36f6d841c6445d98ef84c2d3"
V1_ARTIFACT_SHA256 = {
    "raw_observations.json": (
        "37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a"
    ),
    "summary.json": (
        "00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004"
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wave5_root() -> Path:
    configured = os.environ.get("SIMLLM_WAVE5_RUN_ROOT")
    if not configured:
        raise RuntimeError("SIMLLM_WAVE5_RUN_ROOT must be configured")
    return Path(configured).resolve()


def _validate_commit(repo: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise AssertionError("frozen revision must be a full hash")
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _validate_registry(
    htsim_source: Path, v1_reference_dir: Path, out: Path
) -> None:
    expectations = _load_json(EXPECTATIONS_PATH)
    if expectations["schema"] != "simllm-rnic-control-v2-expectations-v1":
        raise AssertionError("control-v2 expectation schema drifted")
    if expectations["simllm_base_commit"] != SIMLLM_BASE_COMMIT:
        raise AssertionError("SimLLM base commit drifted")
    if expectations["htsim_base_commit"] != HTSIM_BASE_COMMIT:
        raise AssertionError("htsim base commit drifted")
    _validate_commit(REPO_ROOT, SIMLLM_BASE_COMMIT)
    _validate_commit(htsim_source, HTSIM_BASE_COMMIT)

    congestion = expectations["congestion_cells"]
    pfc = expectations["pfc_cells"]
    dynamic = expectations["dynamic_link_cells"]
    if [row["flow_count"] for row in congestion] != [4, 8]:
        raise AssertionError("congestion sweep drifted")
    if [row["payload_bytes"] for row in pfc] != [65536, 131072]:
        raise AssertionError("PFC sweep drifted")
    if [row["up_at_ps"] for row in dynamic] != [201000, 401000]:
        raise AssertionError("dynamic-link sweep drifted")
    scored = expectations["scored_relations"]
    if sum(int(row["instances"]) for row in scored.values()) != 15:
        raise AssertionError("control-v2 scored denominator drifted")

    topology = htsim_source / expectations["topology"]
    if not topology.is_file():
        raise FileNotFoundError(f"frozen topology is unavailable: {topology}")
    for name, expected in V1_ARTIFACT_SHA256.items():
        actual = _digest(v1_reference_dir / name)
        if actual != expected:
            raise AssertionError(
                f"accepted ABI-v1 reference drifted for {name}: {actual}"
            )
    expected_parent = (
        _wave5_root() / "codex" / "htsim1516_control_producers"
    ).resolve()
    try:
        out.resolve().relative_to(expected_parent)
    except ValueError as error:
        raise ValueError(
            "control-v2 output must remain under the branch wave-5 directory"
        ) from error


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _control_probe(build_dir: Path) -> Path:
    candidates = (
        build_dir / "simllm_rnic_control_probe",
        build_dir / "simllm_rnic_control_probe.exe",
        build_dir / "Release" / "simllm_rnic_control_probe",
        build_dir / "Release" / "simllm_rnic_control_probe.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("simllm_rnic_control_probe was not produced by the build")


def _run_probe(
    probe: Path,
    topology: Path,
    condition: str,
    variant: str,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    raw = run_dir / "raw_observations.json"
    subprocess.run(
        [
            str(probe),
            "--topology",
            str(topology),
            "--condition",
            condition,
            "--variant",
            variant,
            "--observations",
            str(raw),
        ],
        check=True,
    )
    return _load_json(raw)


def _cell_by_name(expectations: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [
        *expectations["congestion_cells"],
        *expectations["pfc_cells"],
        *expectations["dynamic_link_cells"],
    ]
    return {str(row["name"]): row for row in rows}


def _identity_artifacts(
    observations: dict[str, Any],
    cell: dict[str, Any],
    topology: bytes,
) -> BypassArtifacts:
    physical_inputs = {
        "cell": cell,
        "flow_count": observations["flow_count"],
        "payload_bytes": observations["payload_bytes"],
    }
    return BypassArtifacts(
        goal_text=_canonical(physical_inputs),
        goal_binary=_canonical(
            {
                "destination": 63,
                "flows": [
                    {
                        "flow_id": 1000 + source,
                        "source": source,
                        "payload_bytes": observations["payload_bytes"],
                    }
                    for source in range(int(observations["flow_count"]))
                ],
            }
        ),
        topology=topology,
        profile="dcqcn",
        seed=9,
        baseline_parameters=canonical_bypass_parameters(
            {
                "condition": str(observations["condition"]),
                "endpoint_link_bps": 400000000000,
                "flow_count": int(observations["flow_count"]),
                "payload_bytes": int(observations["payload_bytes"]),
                "policy_context_token": 9001,
            }
        ),
        completion_csv=_canonical(observations["completions"]),
        canonical_completion=_canonical(observations["issued"]),
        step_results=_canonical(observations["terminals"]),
        replay_summary=_canonical(
            {
                "authority_counters": observations["authority_counters"],
                "packet_events": observations["packet_events"],
                "physical_counters": observations["physical_counters"],
                "quiescent": observations["quiescent"],
            }
        ),
    )


def _events(
    observations: dict[str, Any], kind: str
) -> list[dict[str, Any]]:
    return [
        event
        for event in observations["control_events"]
        if event["kind"] == kind
    ]


def _first_reduced_rate(observations: dict[str, Any]) -> dict[str, Any]:
    cnps = _events(observations, "cnp_received")
    if not cnps:
        raise AssertionError("congestion cell emitted no physical CNP")
    first_cnp = cnps[0]
    candidates = [
        event
        for event in _events(observations, "rate_updated")
        if event["parent_token"] == first_cnp["parent_token"]
        and int(event["event_time_ps"]) >= int(first_cnp["event_time_ps"])
        and int(event["effective_rate_bps"]) < 400000000000
    ]
    if not candidates:
        raise AssertionError("CNP did not induce a reduced effective rate")
    return candidates[0]


def _matching_pause_interval(
    observations: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    for pause in _events(observations, "pfc_paused"):
        resumes = [
            event
            for event in _events(observations, "pfc_resumed")
            if event["link_id"] == pause["link_id"]
            and event["priority"] == pause["priority"]
            and event["source"] == pause["source"]
            and event["destination"] == pause["destination"]
            and int(event["event_time_ps"]) > int(pause["event_time_ps"])
        ]
        if resumes:
            resume = resumes[0]
            return (
                pause,
                resume,
                int(resume["event_time_ps"]) - int(pause["event_time_ps"]),
            )
    raise AssertionError("PFC cell has no matching physical pause and resume")


def _score_raw_relations(
    expectations: dict[str, Any],
    enabled: dict[str, dict[str, Any]],
    disabled: dict[str, dict[str, Any]],
    no_transition: dict[str, dict[str, Any]],
    topology: bytes,
    v1_identity: dict[str, bool],
) -> dict[str, Any]:
    rate_rows = []
    rate_band = expectations["scored_relations"]["cnp_rate_delta_bps"][
        "band"
    ]
    for cell in expectations["congestion_cells"]:
        name = str(cell["name"])
        rate = _first_reduced_rate(enabled[name])
        delta = int(rate["effective_rate_bps"]) - int(
            expectations["endpoint_link_bps"]
        )
        if not int(rate_band[0]) <= delta <= int(rate_band[1]):
            raise AssertionError(f"{name} CNP rate delta is outside its band")
        if disabled[name]["control_events"]:
            raise AssertionError(f"{name} disabled path emitted control events")
        rate_rows.append(
            {
                "condition": name,
                "cnp_at_ps": next(
                    int(event["event_time_ps"])
                    for event in _events(enabled[name], "cnp_received")
                    if event["parent_token"] == rate["parent_token"]
                    and int(event["event_time_ps"])
                    <= int(rate["event_time_ps"])
                ),
                "rate_at_ps": int(rate["event_time_ps"]),
                "effective_rate_bps": int(rate["effective_rate_bps"]),
                "signed_delta_bps": delta,
                "band": rate_band,
            }
        )

    pause_rows = []
    pause_band = expectations["scored_relations"][
        "pfc_pause_duration_ps"
    ]["band"]
    for cell in expectations["pfc_cells"]:
        name = str(cell["name"])
        pause, resume, duration = _matching_pause_interval(enabled[name])
        if not int(pause_band[0]) <= duration <= int(pause_band[1]):
            raise AssertionError(f"{name} PFC interval is outside its band")
        if disabled[name]["control_events"]:
            raise AssertionError(f"{name} disabled path emitted control events")
        pause_rows.append(
            {
                "condition": name,
                "link_id": int(pause["link_id"]),
                "pause_at_ps": int(pause["event_time_ps"]),
                "resume_at_ps": int(resume["event_time_ps"]),
                "duration_ps": duration,
                "band": pause_band,
            }
        )

    dynamic_rows = []
    dynamic_delta_by_name: dict[str, int] = {}
    for cell in expectations["dynamic_link_cells"]:
        name = str(cell["name"])
        completion = max(
            int(row["completion_at_ps"])
            for row in enabled[name]["completions"]
        )
        baseline = max(
            int(row["completion_at_ps"])
            for row in no_transition[name]["completions"]
        )
        delta = completion - baseline
        band = cell["completion_delta_ps_band"]
        if not int(band[0]) <= delta <= int(band[1]):
            raise AssertionError(
                f"{name} dynamic completion delta is outside its band"
            )
        dynamic_delta_by_name[name] = delta
        dynamic_rows.append(
            {
                "condition": name,
                "completion_at_ps": completion,
                "no_transition_completion_at_ps": baseline,
                "signed_delta_ps": delta,
                "band": band,
            }
        )
    spacing = (
        dynamic_delta_by_name["link_hold_long"]
        - dynamic_delta_by_name["link_hold_short"]
    )
    spacing_band = expectations["scored_relations"][
        "dynamic_link_duration_spacing_ps"
    ]["band"]
    if not int(spacing_band[0]) <= spacing <= int(spacing_band[1]):
        raise AssertionError("dynamic-link hold spacing is outside its band")

    identity_rows = []
    cells = _cell_by_name(expectations)
    for name in cells:
        comparison = assert_bypass_artifact_identity(
            _identity_artifacts(enabled[name], cells[name], topology),
            _identity_artifacts(disabled[name], cells[name], topology),
        )
        identity_rows.append(
            {
                "condition": name,
                "changed_inputs": list(comparison.changed_inputs),
                "changed_artifacts": list(comparison.changed_artifacts),
                "input_matches": dict(comparison.input_matches),
                "behavioral_matches": dict(comparison.behavioral_matches),
            }
        )

    if not all(v1_identity.values()):
        raise AssertionError("control producer changed accepted ABI-v1 bytes")
    return {
        "cnp_rate_delta_bps": {
            "passed": len(rate_rows),
            "total": 2,
            "observations": rate_rows,
        },
        "pfc_pause_duration_ps": {
            "passed": len(pause_rows),
            "total": 2,
            "observations": pause_rows,
        },
        "dynamic_link_completion_delta_ps": {
            "passed": len(dynamic_rows),
            "total": 2,
            "observations": dynamic_rows,
        },
        "dynamic_link_duration_spacing_ps": {
            "passed": 1,
            "total": 1,
            "signed_spacing_ps": spacing,
            "band": spacing_band,
        },
        "control_disabled_identity": {
            "passed": len(identity_rows),
            "total": 6,
            "observations": identity_rows,
        },
        "abi_v1_identity": {
            "passed": sum(int(value) for value in v1_identity.values()),
            "total": 2,
            "artifacts": v1_identity,
        },
    }


def _validate_common(
    observations: dict[str, Any],
    condition: str,
    variant: str,
) -> dict[str, int]:
    if observations.get("schema") != "simllm-rnic-control-probe-v1":
        raise AssertionError("control probe schema drifted")
    if observations.get("condition") != condition:
        raise AssertionError("control probe condition drifted")
    if observations.get("variant") != variant:
        raise AssertionError("control probe variant drifted")
    if observations.get("quiescent") is not True:
        raise AssertionError("control probe did not report quiescence")
    flow_count = int(observations["flow_count"])
    if len(observations["issued"]) != flow_count:
        raise AssertionError("control probe issue cardinality drifted")
    if len(observations["terminals"]) != flow_count:
        raise AssertionError("control probe terminal cardinality drifted")
    if len(observations["completions"]) != flow_count:
        raise AssertionError("control probe completion cardinality drifted")
    if int(observations["physical_counters"]["completed_flows"]) != flow_count:
        raise AssertionError("physical completion counter drifted")
    if any(row["kind"] != "delivered" for row in observations["terminals"]):
        raise AssertionError("control study unexpectedly dropped a flow extent")
    if int(observations["physical_counters"]["dropped_packets"]) != 0:
        raise AssertionError("control study unexpectedly dropped a packet")

    issued_tokens = {int(row["token"]) for row in observations["issued"]}
    terminal_tokens = {int(row["token"]) for row in observations["terminals"]}
    if len(issued_tokens) != flow_count or terminal_tokens != issued_tokens:
        raise AssertionError("extent token conservation failed")

    packet_groups: dict[int, list[dict[str, Any]]] = {}
    for event in observations["packet_events"]:
        if int(event["token"]) == 0 or int(event["parent_token"]) not in issued_tokens:
            raise AssertionError("packet event has invalid token correlation")
        packet_groups.setdefault(int(event["token"]), []).append(event)
    if not packet_groups:
        raise AssertionError("physical packet producer emitted no attempts")
    expected_kinds = [
        "packet_tx_started",
        "packet_tx_finished",
        "packet_rx_arrived",
        "delivered",
    ]
    for events in packet_groups.values():
        if [event["kind"] for event in events] != expected_kinds:
            raise AssertionError("packet attempt lifecycle is not exact")
        if [int(event["event_time_ps"]) for event in events] != sorted(
            int(event["event_time_ps"]) for event in events
        ):
            raise AssertionError("packet attempt time moved backwards")
        fixed = (
            "parent_token",
            "wqe_id",
            "extent_index",
            "packet_index",
            "transmission_attempt",
            "payload_offset_bytes",
            "payload_bytes",
            "wire_bytes",
            "packet_kind",
        )
        if any(
            any(event[field] != events[0][field] for field in fixed)
            for event in events[1:]
        ):
            raise AssertionError("packet attempt geometry changed in flight")
    for issued in observations["issued"]:
        payload = sum(
            int(events[0]["payload_bytes"])
            for events in packet_groups.values()
            if int(events[0]["parent_token"]) == int(issued["token"])
            and int(events[0]["transmission_attempt"]) == 0
        )
        if payload != int(issued["payload_bytes"]):
            raise AssertionError("packet payload ledger did not close")
    return {
        "flow_extents": flow_count,
        "packet_attempts": len(packet_groups),
        "packet_events": len(observations["packet_events"]),
    }


def _expected_capabilities(condition: str, variant: str) -> dict[str, Any]:
    enabled = variant == "enabled"
    congestion = condition in {
        "four_flow_ecn",
        "eight_flow_ecn",
        "pfc_64k",
        "pfc_128k",
    }
    pfc = condition in {"pfc_64k", "pfc_128k"}
    dynamic = condition in {"link_hold_short", "link_hold_long"}
    return {
        "abi_version": 2,
        "packet_attempt_events": True,
        "ecn_cnp_events": enabled and congestion,
        "policy_update_events": enabled and congestion,
        "pfc_events": enabled and pfc,
        "dynamic_link_events": enabled and dynamic,
    }


def _validate_congestion(observations: dict[str, Any]) -> dict[str, int]:
    packet_by_token = {
        int(event["token"]): event
        for event in observations["packet_events"]
        if event["kind"] == "delivered"
    }
    terminal_by_wqe = {
        int(row["wqe_id"]): row for row in observations["terminals"]
    }
    ecn = _events(observations, "ecn_marked")
    cnps = _events(observations, "cnp_received")
    rates = _events(observations, "rate_updated")
    eligibility = _events(observations, "eligibility_updated")
    if not ecn or not cnps or not rates or not eligibility:
        raise AssertionError("physical congestion vocabulary is incomplete")
    for event in ecn:
        packet = packet_by_token.get(int(event["token"]))
        if packet is None or event["ecn_marked"] is not True:
            raise AssertionError("ECN mark lost packet-attempt correlation")
        for field in (
            "parent_token",
            "wqe_id",
            "packet_index",
            "transmission_attempt",
            "payload_offset_bytes",
            "payload_bytes",
            "wire_bytes",
            "source",
            "destination",
        ):
            if event[field] != packet[field]:
                raise AssertionError("ECN mark packet geometry drifted")
    late_cnp = 0
    for event in cnps:
        delivered = packet_by_token.get(int(event["token"]))
        terminal = terminal_by_wqe.get(int(event["wqe_id"]))
        if delivered is None or terminal is None:
            raise AssertionError("CNP lost packet or extent correlation")
        if int(delivered["event_time_ps"]) <= int(event["event_time_ps"]) < int(
            terminal["at_ps"]
        ):
            late_cnp += 1
    if late_cnp == 0:
        raise AssertionError("study did not exercise the completed-packet tombstone")
    for event in [*rates, *eligibility]:
        if int(event["policy_context_token"]) != 9001:
            raise AssertionError("policy update lost the descriptor token")
        if event["has_effective_rate"] is not True:
            raise AssertionError("policy update omitted its effective rate")
        if int(event["effective_at_ps"]) != int(event["event_time_ps"]):
            raise AssertionError("policy update effective time drifted")
    if int(observations["physical_counters"]["ecn_marked_packets"]) != len(ecn):
        raise AssertionError("ECN observation and physical counter disagree")
    return {
        "ecn_marks": len(ecn),
        "cnps": len(cnps),
        "rate_updates": len(rates),
        "eligibility_updates": len(eligibility),
        "late_cnp_after_packet_delivery": late_cnp,
    }


def _validate_pfc(observations: dict[str, Any]) -> dict[str, int]:
    submitted = _events(observations, "pfc_frame_submitted")
    paused = _events(observations, "pfc_paused")
    resumed = _events(observations, "pfc_resumed")
    counters = observations["physical_counters"]
    if len(paused) != int(counters["pfc_pauses"]):
        raise AssertionError("PFC pause observation count drifted")
    if len(resumed) != int(counters["pfc_resumes"]):
        raise AssertionError("PFC resume observation count drifted")
    if not paused or len(paused) != len(resumed):
        raise AssertionError("PFC pause and resume sequence did not balance")
    if len(submitted) != len(paused) + len(resumed):
        raise AssertionError("PFC submission sequence did not balance")
    for event in [*submitted, *paused, *resumed]:
        if int(event["link_id"]) == 0:
            raise AssertionError("PFC event omitted its physical link")
        if int(event["priority"]) != 0:
            raise AssertionError("PFC priority drifted")
    if any(int(event["pause_quanta"]) == 0 for event in paused):
        raise AssertionError("PFC pause omitted its nonzero quanta")
    if any(int(event["pause_quanta"]) != 0 for event in resumed):
        raise AssertionError("PFC resume retained pause quanta")
    for arrival in [*paused, *resumed]:
        causal = [
            event
            for event in submitted
            if event["link_id"] == arrival["link_id"]
            and event["source"] == arrival["source"]
            and event["destination"] == arrival["destination"]
            and event["pause_quanta"] == arrival["pause_quanta"]
            and int(event["event_time_ps"]) < int(arrival["event_time_ps"])
        ]
        if not causal:
            raise AssertionError("PFC arrival has no earlier real submission")

    pause, resume, _ = _matching_pause_interval(observations)
    eligibility = _events(observations, "eligibility_updated")
    pause_updates = [
        event
        for event in eligibility
        if event["source"] == pause["source"]
        and int(event["event_time_ps"]) == int(pause["event_time_ps"])
        and int(event["effective_rate_bps"]) == 0
    ]
    resume_updates = [
        event
        for event in eligibility
        if event["source"] == resume["source"]
        and int(event["event_time_ps"]) == int(resume["event_time_ps"])
        and int(event["effective_rate_bps"]) > 0
    ]
    if not pause_updates or not resume_updates:
        raise AssertionError("PFC eligibility did not pause and resume the source")
    return {
        "submitted": len(submitted),
        "paused": len(paused),
        "resumed": len(resumed),
        "paired_eligibility_updates": 2,
    }


def _validate_dynamic(
    observations: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any]:
    transitions = _events(observations, "link_state_changed")
    expected = [
        (int(cell["down_at_ps"]), "down"),
        (int(cell["up_at_ps"]), "up"),
    ]
    actual = [
        (int(event["event_time_ps"]), str(event["link_state"]))
        for event in transitions
    ]
    if actual != expected:
        raise AssertionError("dynamic-link transition rows drifted")
    for event in transitions:
        if int(event["link_id"]) != int(cell["link_id"]):
            raise AssertionError("dynamic-link identity drifted")
        if int(event["source"]) != int(cell["source"]):
            raise AssertionError("dynamic-link source drifted")
        if int(event["effective_at_ps"]) != int(event["event_time_ps"]):
            raise AssertionError("dynamic-link effective time drifted")
    return {"transitions": len(transitions), "rows": actual}


def _validate_exact_oracles(
    expectations: dict[str, Any],
    enabled: dict[str, dict[str, Any]],
    disabled: dict[str, dict[str, Any]],
    no_transition: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cells = _cell_by_name(expectations)
    common: dict[str, Any] = {}
    for name in cells:
        common[name] = {}
        for variant, observations in (
            ("enabled", enabled[name]),
            ("disabled", disabled[name]),
        ):
            common[name][variant] = _validate_common(
                observations, name, variant
            )
            if observations["capabilities"] != _expected_capabilities(
                name, variant
            ):
                raise AssertionError(f"{name} {variant} capability set drifted")
            if variant == "disabled" and observations["control_events"]:
                raise AssertionError(f"{name} disabled control projection is not empty")
        if name in no_transition:
            common[name]["no_transition"] = _validate_common(
                no_transition[name], name, "no_transition"
            )
            if no_transition[name]["capabilities"] != _expected_capabilities(
                name, "no_transition"
            ):
                raise AssertionError(f"{name} no-transition capability set drifted")
            if no_transition[name]["control_events"]:
                raise AssertionError(
                    f"{name} no-transition control projection is not empty"
                )

    congestion = {
        str(cell["name"]): _validate_congestion(enabled[str(cell["name"])])
        for cell in [
            *expectations["congestion_cells"],
            *expectations["pfc_cells"],
        ]
    }
    pfc = {
        str(cell["name"]): _validate_pfc(enabled[str(cell["name"])])
        for cell in expectations["pfc_cells"]
    }
    dynamic = {
        str(cell["name"]): _validate_dynamic(
            enabled[str(cell["name"])], cell
        )
        for cell in expectations["dynamic_link_cells"]
    }
    return {
        "classification": "fatal_unscored",
        "common_packet_and_extent_ledgers": common,
        "congestion": congestion,
        "pfc": pfc,
        "dynamic_link": dynamic,
        "unsupported_capability_rejection": (
            "covered by the full htsim CTest suite"
        ),
    }


def _run(
    htsim_source: Path,
    v1_reference_dir: Path,
    out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    expectations = _load_json(EXPECTATIONS_PATH)
    producer, ctest = packet_study._build(htsim_source, out / "build")
    probe = _control_probe(out / "build")
    topology = htsim_source / expectations["topology"]

    enabled: dict[str, dict[str, Any]] = {}
    disabled: dict[str, dict[str, Any]] = {}
    no_transition: dict[str, dict[str, Any]] = {}
    cells = _cell_by_name(expectations)
    for name in cells:
        enabled[name] = _run_probe(
            probe, topology, name, "enabled", out / "cells" / name / "enabled"
        )
        disabled[name] = _run_probe(
            probe,
            topology,
            name,
            "disabled",
            out / "cells" / name / "disabled",
        )
    for cell in expectations["dynamic_link_cells"]:
        name = str(cell["name"])
        no_transition[name] = _run_probe(
            probe,
            topology,
            name,
            "no_transition",
            out / "cells" / name / "no_transition",
        )

    v1_digests = packet_study._run_v1(producer, out / "abi-v1")
    v1_identity = {
        name: digest == V1_ARTIFACT_SHA256[name]
        for name, digest in v1_digests.items()
    }

    # Every scored family is evaluated from generated raw observations here,
    # before capability, token, geometry, counter, or exact schedule oracles.
    scored = _score_raw_relations(
        expectations,
        enabled,
        disabled,
        no_transition,
        topology.read_bytes(),
        v1_identity,
    )
    exact = _validate_exact_oracles(
        expectations, enabled, disabled, no_transition
    )

    plausible = sum(int(value["passed"]) for value in scored.values())
    relations = sum(int(value["total"]) for value in scored.values())
    if plausible != 15 or relations != 15:
        raise AssertionError("control study scored denominator drifted")
    report = {
        "schema": "simllm-rnic-control-v2-results-v1",
        "simllm_revision": packet_study._git_commit(REPO_ROOT),
        "htsim_revision": packet_study._git_commit(htsim_source),
        "expectation_commit": EXPECTATION_COMMIT,
        "ctest": ctest,
        "scored_relations": scored,
        "genuine_risk": {
            name: {
                "plausible_failures": int(value["passed"]),
                "relations": int(value["total"]),
            }
            for name, value in scored.items()
        }
        | {
            "overall": {
                "plausible_failures": plausible,
                "relations": relations,
            }
        },
        "entailment_analysis": {
            "evaluation_order": (
                "all scored relations use generated raw observations before "
                "the fatal exact-oracle pass"
            ),
            "cnp_rate_delta_bps": (
                "the real CNP can arrive while the subsequent rate remains "
                "unchanged or outside the frozen signed band"
            ),
            "pfc_pause_duration_ps": (
                "the real fabric can emit a zero or out-of-band pause interval"
            ),
            "dynamic_link_relations": (
                "exact transition rows do not force a completion-time response"
            ),
            "control_disabled_identity": (
                "the observation callback can perturb timing, tokens, counters, "
                "ordering, or samples after the run starts"
            ),
            "abi_v1_identity": (
                "accepted semantic output can still differ byte for byte"
            ),
            "exact_oracles": "fatal_unscored",
            "reference_digests": "unscored change-set guards validated at entry",
        },
        "exact_oracles": exact,
        "abi_v1_sha256": v1_digests,
        "raw_cells": {
            name: {
                "enabled": f"cells/{name}/enabled/raw_observations.json",
                "disabled": f"cells/{name}/disabled/raw_observations.json",
                **(
                    {
                        "no_transition": (
                            f"cells/{name}/no_transition/raw_observations.json"
                        )
                    }
                    if name in no_transition
                    else {}
                ),
            }
            for name in cells
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
        print("control-v2 study registry check passed; no artifacts were produced")
        return
    report = _run(
        arguments.htsim_source.resolve(),
        arguments.v1_reference_dir.resolve(),
        arguments.out.resolve(),
    )
    risk = report["genuine_risk"]["overall"]
    print(
        "control-v2 study passed "
        f"{risk['plausible_failures']}/{risk['relations']} "
        "genuine-risk relations"
    )


if __name__ == "__main__":
    main()
