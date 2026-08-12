"""Dry-run registry for the frozen CORE-36 precision-surface study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SEAMS = (
    "workload",
    "request_outcome",
    "framework",
    "compute",
    "dependency",
    "locality",
    "network",
    "rnic_hardware",
)
LEVELS = {
    "workload": ("fixed-trace", "poisson-arrivals"),
    "request_outcome": (
        "fabricated",
        "preplay-oracle",
        "framework-cpu-oracle",
    ),
    "framework": ("recorded-steps", "executor-rpc", "model-runner"),
    "compute": ("fixed", "roofline", "profile-table"),
    "dependency": ("serial", "observed-framework-schedule"),
    "locality": ("all-remote", "analytic-nvlink"),
    "network": ("rnic-nn-fluid", "packet-level"),
    "rnic_hardware": ("timing-neutral-bypass", "composed-native"),
}
NETWORK_HARDWARE_MATRIX = (
    ("rnic-nn-fluid", "timing-neutral-bypass", "accept"),
    ("packet-level", "timing-neutral-bypass", "accept"),
    ("packet-level", "composed-native", "accept"),
    ("rnic-nn-fluid", "composed-native", "refuse"),
)
REFUSAL_DIAGNOSTIC = (
    "precision.rnic_hardware='composed-native' is incompatible with "
    "precision.network='rnic-nn-fluid'; select "
    "rnic_hardware='timing-neutral-bypass' or network='packet-level'"
)
PRECISION_SCHEMA = "simllm-precision-config-v1"
RUN_PROVENANCE_SCHEMA = "simllm-run-provenance-v1"
SOURCE_SCHEMA = "atlahs-closed-loop-step-v1"
SOURCE_RECORD = {
    "schema": SOURCE_SCHEMA,
    "step_index": 0,
    "virtual_time_ps": 0,
    "scheduled": [],
    "preempted_request_ids": [],
    "finished_request_ids": [],
}
SOURCE_SHA256 = "499a5aee695b8269b1ffb5263f62fee6a00207416f7d62d1b0af64f543a68dca"
LEGAL_CONFIG = {
    "schema": PRECISION_SCHEMA,
    "workload": "fixed-trace",
    "request_outcome": "fabricated",
    "framework": "recorded-steps",
    "compute": "roofline",
    "dependency": "serial",
    "locality": "all-remote",
    "network": "packet-level",
    "rnic_hardware": "composed-native",
}
LEGAL_PRECISION_SHA256 = (
    "8e65df0c5296334800755254cb73c4c4f9cb2c090a2b8805a6409bdc3fbe7d45"
)
LEGAL_PROVENANCE_BYTES = 515
LEGAL_PROVENANCE_SHA256 = (
    "9eea24bf89de06325ee492cba345a22c0245c3a806bdd14da0fdbbd77871978d"
)
FROZEN_SCORED_FAMILIES = 1
FROZEN_SCORED_INSTANCES = 1


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _legal_provenance() -> dict[str, Any]:
    return {
        "schema": RUN_PROVENANCE_SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "source_sha256": SOURCE_SHA256,
        "precision": LEGAL_CONFIG,
        "precision_sha256": LEGAL_PRECISION_SHA256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def check_only(args: argparse.Namespace) -> None:
    if tuple(LEVELS) != SEAMS or len(SEAMS) != 8:
        raise AssertionError("eight-seam registry drifted")
    if any(not values or len(values) != len(set(values)) for values in LEVELS.values()):
        raise AssertionError("level registry is empty or duplicated")
    if NETWORK_HARDWARE_MATRIX != (
        ("rnic-nn-fluid", "timing-neutral-bypass", "accept"),
        ("packet-level", "timing-neutral-bypass", "accept"),
        ("packet-level", "composed-native", "accept"),
        ("rnic-nn-fluid", "composed-native", "refuse"),
    ):
        raise AssertionError("network and RNIC hardware matrix drifted")
    if sum(outcome == "refuse" for _, _, outcome in NETWORK_HARDWARE_MATRIX) != 1:
        raise AssertionError("matrix must contain exactly one refusal")
    if REFUSAL_DIAGNOSTIC != (
        "precision.rnic_hardware='composed-native' is incompatible with "
        "precision.network='rnic-nn-fluid'; select "
        "rnic_hardware='timing-neutral-bypass' or network='packet-level'"
    ):
        raise AssertionError("refusal diagnostic drifted")
    if set(LEGAL_CONFIG) != {"schema", *SEAMS}:
        raise AssertionError("legal configuration is not exact")
    for seam in SEAMS:
        if LEGAL_CONFIG[seam] not in LEVELS[seam]:
            raise AssertionError(f"legal configuration has unknown {seam} level")
    source = _canonical(SOURCE_RECORD, newline=True)
    if len(source) != 143 or _sha256(source) != SOURCE_SHA256:
        raise AssertionError("source record canonical identity drifted")
    precision = _canonical(LEGAL_CONFIG)
    if _sha256(precision) != LEGAL_PRECISION_SHA256:
        raise AssertionError("legal precision hash drifted")
    provenance = _canonical(_legal_provenance(), newline=True)
    if (
        len(provenance) != LEGAL_PROVENANCE_BYTES
        or _sha256(provenance) != LEGAL_PROVENANCE_SHA256
    ):
        raise AssertionError("legal provenance canonical identity drifted")
    if (FROZEN_SCORED_FAMILIES, FROZEN_SCORED_INSTANCES) != (1, 1):
        raise AssertionError("evidence denominator drifted")
    if not str(args.out):
        raise AssertionError("output path must be nonempty")
    print(
        "check-only "
        f"out={args.out} seams={len(SEAMS)} matrix={len(NETWORK_HARDWARE_MATRIX)} "
        f"scored={FROZEN_SCORED_FAMILIES}/{FROZEN_SCORED_INSTANCES}"
    )


# --- result mode ----------------------------------------------------------
#
# Nothing below runs in check-only mode and every SimLLM import is local to a
# function, so the pre-freeze dry run stays free of the implementation.

DIMS_KWARGS = {
    "num_layers": 2,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_heads": 4,
    "num_kv_heads": 4,
    "head_size": 16,
    "vocab_size": 256,
    "dtype_bytes": 2,
}
LINKSPEED_BPS = 400_000_000_000
#: one measured entry, enough for the fixed record this study lowers
PROFILE_TABLE = {
    (
        "llm_step",
        (("new_tokens", 1), ("sampled", 1), ("kv_tokens", 128)),
        "b100",
    ): 2_000_000,
}


def _raw_outcome(call) -> dict[str, Any]:
    """Record what a call did, with no expectation applied."""

    try:
        value = call()
    except Exception as error:  # noqa: BLE001 - the raw class is the observation
        return {
            "raised": type(error).__name__,
            "message": str(error),
            "value": None,
        }
    return {"raised": None, "message": None, "value": value}


def _precision(**levels: str):
    from simllm.core import precision_config_from_json

    payload = {"schema": PRECISION_SCHEMA}
    for seam in SEAMS:
        payload[seam] = levels.get(seam, LEGAL_CONFIG[seam])
    return precision_config_from_json(payload)


def observe_matrix() -> list[dict[str, Any]]:
    """Raw constructor outcome for all four cells, before any check."""

    from simllm.core import precision_config_to_json

    rows = []
    for network, rnic_hardware, _frozen in NETWORK_HARDWARE_MATRIX:
        outcome = _raw_outcome(
            lambda network=network, rnic_hardware=rnic_hardware: precision_config_to_json(
                _precision(network=network, rnic_hardware=rnic_hardware)
            )
        )
        rows.append(
            {
                "network": network,
                "rnic_hardware": rnic_hardware,
                "raised": outcome["raised"],
                "message": outcome["message"],
                "accepted": outcome["value"],
            }
        )
    return rows


def observe_stamp() -> dict[str, Any]:
    from simllm.core import (
        STEP_SCHEMA,
        RunProvenance,
        precision_config_sha256,
        precision_config_to_bytes,
        run_provenance_from_bytes,
        run_provenance_to_bytes,
    )

    precision = _precision()
    source = _canonical(SOURCE_RECORD, newline=True)
    provenance = RunProvenance.from_source_bytes(
        source_schema=SOURCE_SCHEMA,
        source_bytes=source,
        precision=precision,
    )
    payload = run_provenance_to_bytes(provenance)
    reparsed = run_provenance_to_bytes(run_provenance_from_bytes(payload))
    return {
        "source_schema_constant": STEP_SCHEMA,
        "precision_json": precision_config_to_bytes(precision).decode("utf-8"),
        "precision_sha256": precision_config_sha256(precision),
        "provenance_bytes": len(payload),
        "provenance_sha256": _sha256(payload),
        "round_trip_identical": reparsed == payload,
        "round_trip_sha256": _sha256(reparsed),
    }


def observe_reader() -> list[dict[str, Any]]:
    """Raw rejection outcome for each corrupted provenance payload."""

    from simllm.core import run_provenance_from_json

    payload = _legal_provenance()
    cases = {
        "unknown_field": {**payload, "extra": 1},
        "missing_field": {
            name: value for name, value in payload.items() if name != "source_schema"
        },
        "unsupported_schema": {**payload, "schema": "simllm-run-provenance-v2"},
        "malformed_source_hash": {**payload, "source_sha256": "not-a-digest"},
        "mismatched_precision_hash": {**payload, "precision_sha256": "0" * 64},
        "invalid_precision_combination": {
            **payload,
            "precision": {**payload["precision"], "network": "rnic-nn-fluid"},
        },
    }
    rows = []
    for name, corrupted in cases.items():
        outcome = _raw_outcome(
            lambda corrupted=corrupted: run_provenance_from_json(corrupted)
        )
        rows.append(
            {
                "case": name,
                "raised": outcome["raised"],
                "message": outcome["message"],
            }
        )
    return rows


def observe_selectors() -> list[dict[str, Any]]:
    """Which level each audited legacy spelling resolves to today."""

    from simllm.adapters.vllm.executor import SimExecutorConfig
    from simllm.compute import ProfileTableProvider, RooflineProvider
    from simllm.core import (
        ExecutionObservations,
        RnicAuthorityMode,
        compute_level_for_provider,
        dependency_level_for_observations,
        locality_level_for_placement,
        network_level_for_profile,
        rnic_hardware_level_for_authority_mode,
    )
    from simllm.placement import PlacementManifest, RankPlacement

    manifest = PlacementManifest(
        ranks=[RankPlacement(0, "node-a", 0), RankPlacement(1, "node-b", 0)]
    )
    observations = ExecutionObservations(operations=(), completion_operation_ids=())

    def _level(value) -> str | None:
        return None if value is None else value.value

    rows: list[dict[str, Any]] = []
    for profile in ("rnic-nn-fluid", "rnic-nn", "rnic-cn", "dcqcn"):
        rows.append(
            {
                "seam": "network",
                "spelling": f"profile={profile!r}",
                "level": _level(network_level_for_profile(profile)),
            }
        )
    for mode in (RnicAuthorityMode.BYPASS, RnicAuthorityMode.STRUCTURAL):
        rows.append(
            {
                "seam": "rnic_hardware",
                "spelling": f"authority_mode={mode.value!r}",
                "level": _level(rnic_hardware_level_for_authority_mode(mode)),
            }
        )
    rows.append(
        {
            "seam": "locality",
            "spelling": "placement_manifest=None",
            "level": _level(locality_level_for_placement(None)),
        }
    )
    rows.append(
        {
            "seam": "locality",
            "spelling": "placement_manifest=PlacementManifest",
            "level": _level(locality_level_for_placement(manifest)),
        }
    )
    rows.append(
        {
            "seam": "dependency",
            "spelling": "observations=None",
            "level": _level(dependency_level_for_observations(None)),
        }
    )
    rows.append(
        {
            "seam": "dependency",
            "spelling": "observations=ExecutionObservations",
            "level": _level(dependency_level_for_observations(observations)),
        }
    )
    rows.append(
        {
            "seam": "compute",
            "spelling": "provider=RooflineProvider",
            "level": _level(compute_level_for_provider(RooflineProvider())),
        }
    )
    rows.append(
        {
            "seam": "compute",
            "spelling": "provider=ProfileTableProvider",
            "level": _level(compute_level_for_provider(ProfileTableProvider(PROFILE_TABLE))),
        }
    )
    rows.append(
        {
            "seam": "compute",
            "spelling": "provider=caller-defined, level undeclared",
            "level": _level(compute_level_for_provider(object())),
        }
    )
    for replay, expected_spelling in (
        (None, "replay_run_path=None"),
        ("run.json", "replay_run_path='run.json'"),
    ):
        levels = SimExecutorConfig(replay_run_path=replay).selected_precision_levels()
        rows.append(
            {
                "seam": "request_outcome",
                "spelling": expected_spelling,
                "level": _level(levels["request_outcome"]),
            }
        )
    for schedule in ("off", "granite-dbo"):
        levels = SimExecutorConfig(observed_schedule=schedule).selected_precision_levels()
        rows.append(
            {
                "seam": "dependency",
                "spelling": f"SIMLLM_VLLM_OBSERVED_SCHEDULE={schedule!r}",
                "level": _level(levels["dependency"]),
            }
        )
    return rows


def _install_stub_backend() -> list[dict[str, Any]]:
    """Replace the native backend with a deterministic in-process stub.

    The submodule is not initialized, so no native binary exists. The bytes
    under comparison are rendered by SimLLM, so a stub is enough to reach a
    complete ``StepResult``. It is never a fidelity claim: this study reads no
    duration as a modeled result.

    The stub is deliberately sensitive to every backend-visible input. Its
    completion time is the payload serialization at the configured link rate,
    and it records the full argument vector of each call. A study whose stub
    ignored the backend arguments could not detect a configuration difference
    that the real backend would act on, so the compatibility comparison would
    silently lose its power.
    """

    import re

    from simllm.backends import step_sink as step_sink_module
    from simllm.backends.htsim_rnic import (
        FlowCompletion,
        RnicRunResult,
        build_htsim_rnic_command,
    )

    send_pattern = re.compile(r": send (\d+)b to (\d+) tag (\d+)\b")
    calls: list[dict[str, Any]] = []
    step_sink_module.to_binary = lambda path: path

    def backend(config):
        goal_text = Path(config.goal_bin).read_text()
        sends = send_pattern.findall(goal_text)
        argv = build_htsim_rnic_command(Path("htsim_rnic"), config)
        calls.append(
            {
                "profile": config.profile,
                "linkspeed_bps": config.linkspeed_bps,
                "topology": None if config.topology is None else config.topology.name,
                "extra_flags": dict(config.extra_flags),
                "argv_tail": [
                    part for part in argv if not part.endswith((".goal", ".csv", ".bin"))
                ],
            }
        )
        flows = []
        for index, (payload_bytes, destination, tag) in enumerate(sends):
            payload = int(payload_bytes)
            serialization_ps = (payload * 8 * 1_000_000_000_000) // config.linkspeed_bps
            flows.append(
                FlowCompletion(
                    profile=config.profile,
                    flow_id=index,
                    source=0,
                    destination=int(destination),
                    tag=int(tag),
                    payload_bytes=payload,
                    start_time_ps=0,
                    completion_time_ps=serialization_ps,
                    fct_ps=serialization_ps,
                )
            )
        return RnicRunResult(
            flows=flows, manifest=["stub completion backend"], quiescent=True
        )

    step_sink_module.run_htsim_rnic = backend
    return calls


def _sink_artifacts(
    workdir: Path,
    profile: str,
    precision,
    provider,
    manifest,
    tp_ranks: tuple[int, ...],
    backend_calls: list[dict[str, Any]],
):
    """Render and execute one step, returning its frozen byte classes."""

    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.backends.rnic_records import BypassArtifacts, canonical_bypass_parameters
    from simllm.compute import ModelDims
    from simllm.core import (
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        step_record_to_json,
        step_result_to_json,
    )

    record = StepRecord(
        step_index=4,
        virtual_time_ps=4_000_000,
        scheduled=[
            ScheduledRequest("request-4", RequestPhase.DECODE, 1, context_length=128)
        ],
    )
    values: dict[str, Any] = {
        "profile": profile,
        "tp_ranks": tp_ranks,
        "dims": ModelDims(**DIMS_KWARGS),
        "workdir": workdir,
        "linkspeed_bps": LINKSPEED_BPS,
        "precision": precision,
    }
    if provider is not None:
        values["provider"] = provider
    if manifest is not None:
        values["placement_manifest"] = manifest
    first_call = len(backend_calls)
    sink = HtsimStepSink(HtsimStepSinkConfig(**values))
    result = sink(record)
    invocations = backend_calls[first_call:]
    goals = tuple(sorted(workdir.glob("step-*.goal")))
    artifacts = BypassArtifacts(
        goal_text=b"".join(path.read_bytes() for path in goals),
        # No LogGOPSim compiler is available without the submodule, so the
        # binary slot carries the rendered artifact identity together with the
        # canonical source step-record bytes that produced it.
        goal_binary=_canonical(
            {
                "artifact_names": [path.name for path in goals],
                "step_record": step_record_to_json(record),
            }
        ),
        # Every backend argument vector the run actually issued. This is the
        # input class that a difference in a backend flag lands in.
        topology=_canonical(invocations),
        profile=profile,
        seed=0,
        baseline_parameters=canonical_bypass_parameters(
            {
                "linkspeed_bps": LINKSPEED_BPS,
                "num_layers": DIMS_KWARGS["num_layers"],
                "tp_ranks": ",".join(str(rank) for rank in tp_ranks),
            }
        ),
        completion_csv=_canonical(
            [
                {
                    "step_index": outcome.step_index,
                    "makespan_ps": outcome.makespan_ps,
                    "num_flows": outcome.num_flows,
                    "compute_estimate_ps": outcome.compute_estimate_ps,
                    "layer_calc_ns": list(outcome.layer_calc_ns),
                    "quiescent": outcome.quiescent,
                    "routing_mode": outcome.routing_mode,
                }
                for outcome in sink.outcomes
            ]
        ),
        canonical_completion=_canonical(
            [
                {
                    "step_index": outcome.step_index,
                    "authority": outcome.authority,
                    "compute_service_ps": outcome.compute_service_ps,
                    "nvlink_service_ps": outcome.nvlink_service_ps,
                    "fabric_directed_bytes": outcome.fabric_directed_bytes,
                    "nvlink_directed_bytes": outcome.nvlink_directed_bytes,
                    "fabric_phase_service_ps": list(outcome.fabric_phase_service_ps),
                    "composed_phase_service_ps": list(
                        outcome.composed_phase_service_ps
                    ),
                    "ordering_authority": outcome.ordering_authority,
                }
                for outcome in sink.locality_outcomes
            ]
        ),
        step_results=_canonical(
            None if result is None else step_result_to_json(result)
        ),
        replay_summary=_canonical(
            {"artifact_names": [path.name for path in goals], "profile": profile}
        ),
    )
    return artifacts


def observe_compatibility(out: Path) -> list[dict[str, Any]]:
    """Legacy against explicitly configured runs, for every routed spelling."""

    from simllm.backends.htsim_rnic import HtsimRnicConfig, build_htsim_rnic_command
    from simllm.backends.rnic_records import assert_bypass_artifact_identity
    from simllm.compute import ProfileTableProvider, RooflineProvider
    from simllm.core import (
        CoarseDeviceRuntime,
        DependencyLevel,
        RnicAuthorityMode,
        RnicHardwareLevel,
        compute_level_for_provider,
        locality_level_for_placement,
        network_level_for_profile,
        resolve_precision_config,
    )
    from simllm.placement import PlacementManifest, RankPlacement

    backend_calls = _install_stub_backend()
    # Two ranks per node, so the analytic split classifies both NVLink and
    # fabric segments instead of degenerating into one of the two.
    manifest = PlacementManifest(
        ranks=[
            RankPlacement(0, "node-a", 0),
            RankPlacement(1, "node-a", 1),
            RankPlacement(2, "node-b", 0),
            RankPlacement(3, "node-b", 1),
        ]
    )
    split_ranks = (0, 1, 2, 3)
    cells = [
        ("profile=rnic-nn", "rnic-nn", None, None, (0, 1)),
        ("profile=rnic-nn-fluid", "rnic-nn-fluid", None, None, (0, 1)),
        ("provider=RooflineProvider", "rnic-nn", RooflineProvider(), None, (0, 1)),
        (
            "provider=ProfileTableProvider",
            "rnic-nn",
            ProfileTableProvider(PROFILE_TABLE),
            None,
            (0, 1),
        ),
        ("placement_manifest=absent", "rnic-nn", None, None, split_ranks),
        ("placement_manifest=present", "rnic-nn", None, manifest, split_ranks),
    ]
    rows: list[dict[str, Any]] = []
    for index, (name, profile, provider, placement, tp_ranks) in enumerate(cells):
        explicit = resolve_precision_config(
            compute=compute_level_for_provider(
                RooflineProvider() if provider is None else provider
            ),
            dependency=DependencyLevel.SERIAL,
            locality=locality_level_for_placement(placement),
            network=network_level_for_profile(profile),
        )
        legacy = _sink_artifacts(
            out / f"compat-{index}-legacy",
            profile,
            None,
            provider,
            placement,
            tp_ranks,
            backend_calls,
        )
        configured = _sink_artifacts(
            out / f"compat-{index}-explicit",
            profile,
            explicit,
            provider,
            placement,
            tp_ranks,
            backend_calls,
        )

        def _identity(legacy=legacy, configured=configured):
            comparison = assert_bypass_artifact_identity(legacy, configured)
            return {
                "input_matches": [list(pair) for pair in comparison.input_matches],
                "behavioral_matches": [
                    list(pair) for pair in comparison.behavioral_matches
                ],
            }

        outcome = _raw_outcome(_identity)
        rows.append(
            {
                "spelling": name,
                "kind": "bypass_artifacts",
                "raised": outcome["raised"],
                "message": outcome["message"],
                "input_matches": (
                    None if outcome["value"] is None else outcome["value"]["input_matches"]
                ),
                "behavioral_matches": (
                    None
                    if outcome["value"] is None
                    else outcome["value"]["behavioral_matches"]
                ),
                "goal_bytes": len(legacy.goal_text),
                "backend_argv_bytes": len(legacy.topology),
                "step_result_bytes": len(legacy.step_results),
            }
        )

    for profile in ("rnic-nn", "rnic-nn-fluid", "rnic-cn"):
        explicit = resolve_precision_config(network=network_level_for_profile(profile))

        def _argv(precision, profile=profile):
            return build_htsim_rnic_command(
                Path("htsim_rnic"),
                HtsimRnicConfig(
                    goal_bin=out / "step.bin",
                    profile=profile,
                    linkspeed_bps=LINKSPEED_BPS,
                    completion_csv=out / "completions.csv",
                    precision=precision,
                ),
            )

        rows.append(
            {
                "spelling": f"command argv, profile={profile}",
                "kind": "command_arguments",
                "raised": None,
                "message": None,
                "identical": _argv(None) == _argv(explicit),
                "argv": _argv(explicit),
            }
        )

    mode = RnicAuthorityMode.BYPASS
    bypass_surface = resolve_precision_config(
        rnic_hardware=RnicHardwareLevel.TIMING_NEUTRAL_BYPASS
    )
    legacy_runtime = CoarseDeviceRuntime(authority_mode=mode)
    explicit_runtime = CoarseDeviceRuntime(authority_mode=mode, precision=bypass_surface)
    rows.append(
        {
            "spelling": f"authority_mode={mode.value}",
            "kind": "runtime_authority",
            "raised": None,
            "message": None,
            "identical": (
                legacy_runtime.authority_name == explicit_runtime.authority_name
                and (legacy_runtime.bypass_ledger is None)
                == (explicit_runtime.bypass_ledger is None)
            ),
            "authority_name": legacy_runtime.authority_name,
        }
    )
    return rows


def observe_conflicts(out: Path) -> list[dict[str, Any]]:
    """Raw refusal outcome when an explicit surface contradicts a spelling."""

    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.backends.htsim_rnic import HtsimRnicConfig
    from simllm.compute import ModelDims
    from simllm.core import CoarseDeviceRuntime, RnicAuthorityMode

    legal = _precision()
    guarded = out / "never-created"

    def _sink():
        return HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn-fluid",
                tp_ranks=(0, 1),
                dims=ModelDims(**DIMS_KWARGS),
                workdir=guarded,
                precision=legal,
            )
        )

    sink_outcome = _raw_outcome(_sink)
    rows = [
        {
            "component": "HtsimStepSinkConfig",
            "raised": sink_outcome["raised"],
            "message": sink_outcome["message"],
            "output_path_created": guarded.exists(),
        }
    ]
    rnic_outcome = _raw_outcome(
        lambda: HtsimRnicConfig(
            goal_bin=out / "step.bin",
            profile="rnic-nn-fluid",
            linkspeed_bps=LINKSPEED_BPS,
            precision=legal,
        )
    )
    rows.append(
        {
            "component": "HtsimRnicConfig",
            "raised": rnic_outcome["raised"],
            "message": rnic_outcome["message"],
            "output_path_created": False,
        }
    )
    runtime_outcome = _raw_outcome(
        lambda: CoarseDeviceRuntime(
            authority_mode=RnicAuthorityMode.BYPASS, precision=legal
        )
    )
    rows.append(
        {
            "component": "CoarseDeviceRuntime",
            "raised": runtime_outcome["raised"],
            "message": runtime_outcome["message"],
            "output_path_created": False,
        }
    )
    return rows


def score(observations: dict[str, Any]) -> dict[str, Any]:
    """Score the refusal family from raw observations, then run fatal guards.

    The scored family is evaluated first and independently: a permissive
    validator accepts the refused cell and reaches every later oracle, so no
    exact-stamp or compatibility check entails it.
    """

    refused = [
        row
        for row in observations["matrix"]
        if (row["network"], row["rnic_hardware"]) == ("rnic-nn-fluid", "composed-native")
    ]
    if len(refused) != 1:
        raise AssertionError("matrix lost its single refused cell")
    cell = refused[0]
    scored_pass = (
        cell["raised"] == "ValueError"
        and cell["message"] == REFUSAL_DIAGNOSTIC
        and cell["accepted"] is None
    )

    findings: list[str] = []

    def _guard(name: str, condition: bool) -> None:
        if not condition:
            findings.append(name)

    expected_outcomes = {
        (network, hardware): outcome
        for network, hardware, outcome in NETWORK_HARDWARE_MATRIX
    }
    for row in observations["matrix"]:
        expected = expected_outcomes[(row["network"], row["rnic_hardware"])]
        if expected == "accept":
            _guard(
                f"matrix accepts {row['network']}/{row['rnic_hardware']}",
                row["raised"] is None and row["accepted"] is not None,
            )

    stamp = observations["stamp"]
    _guard("stamp source schema constant", stamp["source_schema_constant"] == SOURCE_SCHEMA)
    _guard("stamp precision hash", stamp["precision_sha256"] == LEGAL_PRECISION_SHA256)
    _guard("stamp byte length", stamp["provenance_bytes"] == LEGAL_PROVENANCE_BYTES)
    _guard("stamp digest", stamp["provenance_sha256"] == LEGAL_PROVENANCE_SHA256)
    _guard("stamp round trip", stamp["round_trip_identical"] is True)

    for row in observations["reader"]:
        _guard(f"reader rejects {row['case']}", row["raised"] in ("ValueError", "TypeError"))

    for row in observations["selectors"]:
        _guard(
            f"selector {row['spelling']} resolves",
            row["level"] is not None
            or row["spelling"] == "provider=caller-defined, level undeclared",
        )

    for row in observations["compatibility"]:
        if row["kind"] == "bypass_artifacts":
            _guard(
                f"bypass identity for {row['spelling']}",
                row["raised"] is None
                and bool(row["input_matches"])
                and all(match for _, match in row["input_matches"])
                and all(match for _, match in row["behavioral_matches"]),
            )
        else:
            _guard(f"identity for {row['spelling']}", row["identical"] is True)

    for row in observations["conflicts"]:
        _guard(
            f"{row['component']} refuses a contradiction",
            row["raised"] == "ValueError" and "conflicts with" in (row["message"] or ""),
        )
        _guard(
            f"{row['component']} creates no output first",
            row["output_path_created"] is False,
        )

    return {
        "scored_families": FROZEN_SCORED_FAMILIES,
        "scored_instances": FROZEN_SCORED_INSTANCES,
        "scored_passed": 1 if scored_pass else 0,
        "refused_cell": cell,
        "fatal_violation_count": len(findings),
        "fatal_findings": findings,
        "void": bool(findings),
    }


def stamp_own_run(out: Path) -> Path:
    """Write this run's own fidelity configuration beside its observations.

    The study is itself a result, so it carries the precision that produced
    it. The observation file is the source artifact, and its schema and hash
    sit next to the resolved selection in one canonical record.
    """

    from simllm.core import RunProvenance, run_provenance_to_bytes

    observations_path = out / "observations.json"
    provenance = RunProvenance.from_source_bytes(
        source_schema="simllm-precision-surface-observations-v1",
        source_bytes=observations_path.read_bytes(),
        precision=_precision(),
    )
    target = out / "run-provenance.json"
    target.write_bytes(run_provenance_to_bytes(provenance))
    return target


def run(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    observations = {
        "matrix": observe_matrix(),
        "stamp": observe_stamp(),
        "reader": observe_reader(),
        "selectors": observe_selectors(),
        "compatibility": observe_compatibility(out),
        "conflicts": observe_conflicts(out),
    }
    (out / "observations.json").write_bytes(_canonical(observations, newline=True))
    results = score(observations)
    (out / "results.json").write_bytes(_canonical(results, newline=True))
    stamp_own_run(out)
    if results["void"]:
        print("VOID: fatal guards violated")
        for finding in results["fatal_findings"]:
            print(f"  {finding}")
        print(f"scored (uninterpretable): {results['scored_passed']}/1")
        return 1
    print(
        "valid run: genuine-risk refusal family "
        f"{results['scored_passed']}/{FROZEN_SCORED_INSTANCES}"
    )
    print(f"observations: {out / 'observations.json'}")
    print(f"run provenance: {out / 'run-provenance.json'}")
    return 0 if results["scored_passed"] == FROZEN_SCORED_INSTANCES else 2


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return
    raise SystemExit(run(args.out))


if __name__ == "__main__":
    main()
