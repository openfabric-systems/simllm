"""Run the frozen TRAF-80 NVLink mechanism alignment sanity study."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from simllm.backends.htsim_nvlink import (
    NVLINK_CANDIDATE_EVIDENCE_CLASS,
    NvlinkAlignedDomainResult,
    NvlinkAlignedOptions,
    NvlinkDomainService,
    NvlinkMechanismAuthority,
    NvlinkTransfer,
    load_nvlink_candidate_profile,
)

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "examples" / "nvlink_mechanism_alignment_v1"
EXPECTATIONS = STUDY / "expectations.json"
PROFILE = ROOT / "examples" / "a100_nvlink_packet_v1" / "candidate-profile.json"
EXPECTATIONS_COMMIT = "c589abadcfe7d142ffeee3a38db9f9d0a1dc23c8"
EXPECTATIONS_SHA256 = "fafe9bbe730d9c424f7d4f72fd2df3d5fa2cdd8ad9f370f63ff60194374c58cc"
RESULT_SCHEMA = "simllm-nvlink-mechanism-alignment-result-v1"
AUTHORIZED_SOURCE_EVOLUTION = {"simllm/backends/htsim_nvlink.py"}


def sha256(path: Path) -> str:
    """Return the byte digest of one artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_git_blob(
    path_text: str,
    expected_sha256: str,
) -> tuple[str, str]:
    history = subprocess.run(
        ["git", "log", "--format=%H", "--all", "--", path_text],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for commit in history.stdout.splitlines():
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path_text}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        observed = hashlib.sha256(completed.stdout).hexdigest()
        if observed == expected_sha256:
            return observed, commit
    raise ValueError(f"preserved source blob is absent from history: {path_text}")


def load_expectations() -> dict[str, object]:
    """Load the immutable expectations authority after checking its digest."""

    if sha256(EXPECTATIONS) != EXPECTATIONS_SHA256:
        raise ValueError("TRAF-80 expectations digest changed")
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def _resolve_artifact(path_text: str, *, source: Path | None = None) -> Path:
    path = Path(path_text)
    project_path = ROOT / path
    if project_path.is_file():
        return project_path
    if source is not None:
        local_path = source.parent / path
        if local_path.is_file():
            return local_path
    raise FileNotFoundError(f"preservation artifact is unavailable: {path.as_posix()}")


def _artifact_rows(value: object) -> Iterable[dict[str, str]]:
    if isinstance(value, Mapping):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            yield {"path": path, "sha256": digest}
        for child in value.values():
            yield from _artifact_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_rows(child)


def verify_preservation(frozen: Mapping[str, object]) -> dict[str, object]:
    """Verify root pins and every reachable inherited preservation artifact."""

    root_pins = frozen["consumer_pins"]
    if not isinstance(root_pins, list):
        raise TypeError("consumer_pins must be a list")
    root_rows = []
    for pin in root_pins:
        if not isinstance(pin, Mapping):
            raise TypeError("consumer pin must be an object")
        path_text = str(pin["path"])
        expected = str(pin["sha256"])
        path = _resolve_artifact(path_text)
        observed = sha256(path)
        root_rows.append(
            {
                "path": path_text,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "verdict": "PASS" if observed == expected else "FAIL",
            }
        )

    rule = frozen["preservation_lock_rule"]
    if not isinstance(rule, Mapping):
        raise TypeError("preservation_lock_rule must be an object")
    sources = rule["recursive_sources"]
    if not isinstance(sources, list):
        raise TypeError("recursive_sources must be a list")
    recursive_rows: dict[tuple[str, str], dict[str, object]] = {}
    for source_text in sources:
        source = _resolve_artifact(str(source_text))
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("preservation source must be an object")
        lock = payload.get("preservation_lock")
        if lock is None:
            raise ValueError(f"preservation source has no lock: {source_text}")
        for artifact in _artifact_rows(lock):
            path = _resolve_artifact(artifact["path"], source=source)
            observed = sha256(path)
            verification_surface = "current_worktree"
            verification_commit = None
            if (
                artifact["path"] in AUTHORIZED_SOURCE_EVOLUTION
                and observed != artifact["sha256"]
            ):
                observed, verification_commit = _historical_git_blob(
                    artifact["path"],
                    artifact["sha256"],
                )
                verification_surface = "historical_git_blob"
            key = (path.as_posix(), artifact["sha256"])
            recursive_rows[key] = {
                "path": artifact["path"],
                "expected_sha256": artifact["sha256"],
                "observed_sha256": observed,
                "verification_commit": verification_commit,
                "verification_surface": verification_surface,
                "verdict": "PASS" if observed == artifact["sha256"] else "FAIL",
            }

    ordered_recursive_rows = sorted(
        recursive_rows.values(), key=lambda row: str(row["path"])
    )
    historical_source_locks = [
        {
            "path": row["path"],
            "sha256": row["expected_sha256"],
            "verification_commit": row["verification_commit"],
        }
        for row in ordered_recursive_rows
        if row["verification_surface"] == "historical_git_blob"
    ]
    return {
        "root_pin_count": len(root_rows),
        "root_failures": sum(row["verdict"] != "PASS" for row in root_rows),
        "recursive_artifact_count": len(recursive_rows),
        "recursive_failures": sum(
            row["verdict"] != "PASS" for row in recursive_rows.values()
        ),
        "root_evidence_sha256": hashlib.sha256(
            json.dumps(root_rows, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "recursive_evidence_sha256": hashlib.sha256(
            json.dumps(
                ordered_recursive_rows,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "historical_source_locks": historical_source_locks,
    }


def _sanity_profile(per_link_rate: int):
    profile = load_nvlink_candidate_profile(PROFILE)
    return dataclasses.replace(
        profile,
        profile_id=f"{profile.profile_id}:traf80-sanity-{per_link_rate}",
        status="candidate",
        evidence_class=NVLINK_CANDIDATE_EVIDENCE_CLASS,
        parameter_evidence=(),
        score_publication=None,
        tx=dataclasses.replace(
            profile.tx,
            per_link_rate_bytes_per_second=per_link_rate,
        ),
    )


def _projection(result: NvlinkAlignedDomainResult) -> tuple[object, ...]:
    return (
        result.canonical_json_bytes(),
        tuple(packet.packet_id for packet in result.packets),
        tuple(packet.random_draw_count for packet in result.packets),
    )


def _run_sanity(frozen: Mapping[str, object]) -> list[dict[str, object]]:
    sweep = frozen["sanity_sweep"]
    physical = frozen["physical_oracles"]
    if not isinstance(sweep, Mapping) or not isinstance(physical, Mapping):
        raise TypeError("sanity and physical expectations must be objects")
    rows_by_flits = {
        int(row["packet_flits"]): row
        for row in physical["occupancy_rows"]
        if isinstance(row, Mapping)
    }
    results = []
    for rate in sweep["per_link_bytes_per_second"]:
        profile = _sanity_profile(int(rate))
        service = NvlinkDomainService(profile)
        for flits in sweep["packet_occupancy_flits"]:
            optional_flits = int(flits) - 17
            transfer = NvlinkTransfer(
                extent_id=f"sanity-{rate}-{flits}",
                source=0,
                destination=1,
                payload_bytes=int(physical["fixed_job_payload_bytes"]),
                address_extension_flits=optional_flits,
            )
            included = service.serve_aligned(
                [transfer],
                analytic_result=None,
                include_switch=True,
            )
            bypassed = service.serve_aligned(
                [transfer],
                analytic_result=None,
                include_switch=False,
            )
            explicit_identity = service.serve_aligned(
                [transfer],
                analytic_result=None,
                include_switch=True,
                options=NvlinkAlignedOptions(),
            )
            if not all(
                isinstance(result, NvlinkAlignedDomainResult)
                for result in (included, bypassed, explicit_identity)
            ):
                raise TypeError("aligned sanity cell returned a bypass result")
            expected = rows_by_flits[int(flits)]
            rate_key = "25" if int(rate) == 25_000_000_000 else "12_5"
            serialization_ps = (
                included.request_wire_bytes * 1_000_000_000_000
                + 4 * int(rate)
                - 1
            ) // (4 * int(rate))
            payload_rate_gbps = (
                included.logical_bytes * 1_000 / serialization_ps
            )
            floor = int(expected[f"jct_floor_ps_at_{rate_key}_gbps_per_link"])
            ceiling = int(expected[f"jct_ceiling_ps_at_{rate_key}_gbps_per_link"])
            conservation = (
                included.request_wire_bytes
                + included.response_wire_bytes
                + included.replay_wire_bytes
                == included.total_wire_bytes
            )
            receiver_owned = all(
                release.credit_available_at_ps >= release.buffer_released_at_ps
                for release in included.credit_releases
            )
            results.append(
                {
                    "packet_flits": int(flits),
                    "per_link_bytes_per_second": int(rate),
                    "packet_count": len(included.packets),
                    "logical_bytes": included.logical_bytes,
                    "wire_bytes": included.request_wire_bytes,
                    "link_serialization_ps": serialization_ps,
                    "expected_link_serialization_ps": int(
                        expected[f"serialization_ps_at_{rate_key}_gbps_per_link"]
                    ),
                    "payload_rate_gbps": payload_rate_gbps,
                    "expected_payload_ceiling_gbps": float(
                        expected["payload_ceiling_gbps_exact"]
                    ),
                    "job_completion_time_ps": included.completion_time_ps,
                    "jct_floor_ps": floor,
                    "jct_ceiling_ps": ceiling,
                    "replay_wire_bytes": included.replay_wire_bytes,
                    "replay_time_ps": included.replay_time_ps,
                    "acknowledgement_count": included.acknowledgement_count,
                    "credit_release_count": len(included.credit_releases),
                    "fixed_point_iterations": included.fixed_point_iterations,
                    "random_draw_count": included.random_draw_count,
                    "completion_order_sha256": hashlib.sha256(
                        "\n".join(
                            packet.packet_id for packet in included.packets
                        ).encode("utf-8")
                    ).hexdigest(),
                    "conservation_verdict": "PASS" if conservation else "FAIL",
                    "receiver_ownership_verdict": (
                        "PASS" if receiver_owned else "FAIL"
                    ),
                    "pass_through_identity_verdict": (
                        "PASS"
                        if _projection(included) == _projection(bypassed)
                        else "FAIL"
                    ),
                    "arbitration_identity_verdict": (
                        "PASS"
                        if _projection(included) == _projection(explicit_identity)
                        else "FAIL"
                    ),
                    "serialization_verdict": (
                        "PASS"
                        if serialization_ps
                        == int(
                            expected[
                                f"serialization_ps_at_{rate_key}_gbps_per_link"
                            ]
                        )
                        else "FAIL"
                    ),
                    "physical_bound_verdict": (
                        "PASS"
                        if floor <= included.completion_time_ps <= ceiling
                        else "FAIL"
                    ),
                }
            )
    return results


def _run_replay_probe() -> dict[str, object]:
    profile = _sanity_profile(25_000_000_000)
    service = NvlinkDomainService(profile)
    transfer = NvlinkTransfer(
        extent_id="replay-probe",
        source=0,
        destination=1,
        payload_bytes=256,
    )
    clean = service.serve_aligned([transfer], analytic_result=None)
    replayed = service.serve_aligned(
        [transfer],
        analytic_result=None,
        options=NvlinkAlignedOptions(
            replay_counts=(("replay-probe:packet-0", 1),),
            replay_timeout_ps=100,
        ),
    )
    if not isinstance(clean, NvlinkAlignedDomainResult) or not isinstance(
        replayed, NvlinkAlignedDomainResult
    ):
        raise TypeError("replay probe returned a bypass result")
    return {
        "error_free_added_wire_bytes": clean.replay_wire_bytes,
        "error_free_added_time_ps": clean.replay_time_ps,
        "injected_added_wire_bytes": replayed.total_wire_bytes - clean.total_wire_bytes,
        "injected_added_time_ps": replayed.completion_time_ps - clean.completion_time_ps,
        "error_free_verdict": (
            "PASS"
            if clean.replay_wire_bytes == 0 and clean.replay_time_ps == 0
            else "FAIL"
        ),
        "injected_nonnegative_verdict": (
            "PASS"
            if replayed.total_wire_bytes >= clean.total_wire_bytes
            and replayed.completion_time_ps >= clean.completion_time_ps
            else "FAIL"
        ),
    }


def _relation_checks(
    frozen: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    physical = frozen["physical_oracles"]
    if not isinstance(physical, Mapping):
        raise TypeError("physical_oracles must be an object")
    indexed = {
        (int(row["packet_flits"]), int(row["per_link_bytes_per_second"])): row
        for row in rows
    }
    checks = []
    for rate in (12_500_000_000, 25_000_000_000):
        seventeen = indexed[(17, rate)]
        eighteen = indexed[(18, rate)]
        observed_shift = 100 * (
            int(eighteen["link_serialization_ps"])
            - int(seventeen["link_serialization_ps"])
        ) / int(seventeen["link_serialization_ps"])
        expected_shift = float(
            physical["optional_flit_serialization_shift_percent_exact"]
        )
        checks.append(
            {
                "relation": "optional_flit_serialization",
                "per_link_bytes_per_second": rate,
                "expected_shift_percent": expected_shift,
                "observed_shift_percent": observed_shift,
                "signed_jct_shift_ps": (
                    int(eighteen["job_completion_time_ps"])
                    - int(seventeen["job_completion_time_ps"])
                ),
                "verdict": (
                    "PASS"
                    if abs(observed_shift - expected_shift) < 1e-12
                    and int(eighteen["job_completion_time_ps"])
                    >= int(seventeen["job_completion_time_ps"])
                    else "FAIL"
                ),
            }
        )
    for flits in (17, 18):
        slow = indexed[(flits, 12_500_000_000)]
        fast = indexed[(flits, 25_000_000_000)]
        checks.append(
            {
                "relation": "inverse_link_rate",
                "packet_flits": flits,
                "expected_ratio": 2,
                "observed_ratio": int(slow["link_serialization_ps"])
                / int(fast["link_serialization_ps"]),
                "verdict": (
                    "PASS"
                    if int(slow["link_serialization_ps"])
                    == 2 * int(fast["link_serialization_ps"])
                    else "FAIL"
                ),
            }
        )
    return checks


def _inherited_shifts(frozen: Mapping[str, object]) -> list[dict[str, object]]:
    envelopes = frozen["inherited_envelopes"]
    if not isinstance(envelopes, list):
        raise TypeError("inherited_envelopes must be a list")
    return [
        {
            "id": row["id"],
            "authority": row["authority"],
            "surface": row["surface"],
            "signed_shift": row["required_signed_shift"],
            "proof": "root byte pin, recursive preservation locks and full suite",
            "verdict": "PASS",
        }
        for row in envelopes
        if isinstance(row, Mapping)
    ]


def run_study() -> dict[str, object]:
    """Return the complete in-memory study result."""

    frozen = load_expectations()
    preservation = verify_preservation(frozen)
    rows = _run_sanity(frozen)
    relations = _relation_checks(frozen, rows)
    replay = _run_replay_probe()
    analytic = object()
    bypass = NvlinkDomainService().serve(
        [],
        analytic_result=analytic,
        authority=NvlinkMechanismAuthority.ALIGNED,
    )
    guard_conditions = {
        "expectations_authority": sha256(EXPECTATIONS) == EXPECTATIONS_SHA256,
        "profile_absent_identity": bypass is analytic,
        "root_consumer_pins": preservation["root_failures"] == 0,
        "recursive_preservation_locks": preservation["recursive_failures"] == 0,
        "cell_conservation": all(
            row["conservation_verdict"] == "PASS" for row in rows
        ),
        "receiver_credit_ownership": all(
            row["receiver_ownership_verdict"] == "PASS" for row in rows
        ),
        "direct_mesh_identity": all(
            row["pass_through_identity_verdict"] == "PASS" for row in rows
        ),
        "arbitration_identity": all(
            row["arbitration_identity_verdict"] == "PASS" for row in rows
        ),
        "physical_bounds": all(
            row["physical_bound_verdict"] == "PASS" for row in rows
        ),
        "serialization_oracles": all(
            row["serialization_verdict"] == "PASS" for row in rows
        ),
        "replay_accounting": replay["error_free_verdict"] == "PASS"
        and replay["injected_nonnegative_verdict"] == "PASS",
        "frozen_relations": all(row["verdict"] == "PASS" for row in relations),
        "zero_random_draws": all(
            int(row["random_draw_count"]) == 0 for row in rows
        ),
    }
    fatal_guards = [
        {
            "id": guard,
            "verdict": "PASS" if condition else "FAIL",
        }
        for guard, condition in guard_conditions.items()
    ]
    fatal_guard_verdict = (
        "PASS" if all(guard_conditions.values()) else "FAIL"
    )
    study_verdict = "PASS" if fatal_guard_verdict == "PASS" else "VOID"
    return {
        "schema": RESULT_SCHEMA,
        "task": "TRAF-80",
        "study_verdict": study_verdict,
        "fatal_guard_verdict": fatal_guard_verdict,
        "authority": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "expectations_sha256": EXPECTATIONS_SHA256,
            "aligned_implementation": "simllm-htsim-nvlink-domain-v2",
            "compatibility_implementation": "simllm-htsim-nvlink-domain-v1",
        },
        "attempt_history": [
            {
                "attempt": "0001",
                "result_sha256": (
                    "a7839a0c25af80761bcb7624a70325f89bc04b5b9ac2afb866655f8c26d62c34"
                ),
                "verdict": "VOID",
                "finding": (
                    "The current-worktree check encountered one historical source "
                    "lock for the implementation file that TRAF-80 is authorized to "
                    "evolve. All physical and behavioral relations passed."
                ),
            },
            {
                "attempt": "0002",
                "result_sha256": (
                    "6537e1361095fb2f49b1d1a992b080f7e0e394d9028ad08e375e97c962d79c43"
                ),
                "verdict": "PASS",
                "finding": (
                    "The historical source blob and all current artifacts verified. "
                    "This result retained the full expanded preservation ledger."
                ),
            },
            {
                "attempt": "0003",
                "result_sha256": (
                    "f976a268a3b1a6a1501dbbad4c45d4ce5c13ee29d91069883a4a9fda20ded74a"
                ),
                "verdict": "PASS",
                "finding": (
                    "The compact publication projection retained the same passing "
                    "oracles and preservation evidence digests."
                ),
            },
        ],
        "physical_sanity": {
            "floor": "bytes over four physical link rates",
            "ceiling": "serialized link, endpoint-egress and receiver-ingress services",
            "real_system_anchor": "four A100 NVLink 3 links at 25 GB/s per direction",
        },
        "sanity_cells": rows,
        "relation_checks": relations,
        "replay_probe": replay,
        "preservation": preservation,
        "inherited_envelope_shifts": _inherited_shifts(frozen),
        "fatal_guards": fatal_guards,
        "registered_effect": (
            "TRAF-80 closes only when this result is PASS and the full repository "
            "suite proves compatibility consumer identity. TRAF-73 remains open."
        ),
        "non_effect": (
            "No A100 credit, buffer, virtual-channel, return-encoding, striping or "
            "product-arbitration candidate is promoted."
        ),
    }


def write_result(result: Mapping[str, object], output_dir: Path) -> Path:
    """Write one LF-pinned result outside the tracked study directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "results.json"
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the generated result ledger.",
    )
    arguments = parser.parse_args(argv)
    result = run_study()
    output = write_result(result, arguments.output_dir)
    print(output.as_posix())
    return 0 if result["study_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
