"""Run the frozen vLLM observed-overlap study.

Three arms share one live vLLM observation stream. ``serial`` withholds the
observations so the sink delegates to the serial compatibility lowering.
``control`` takes the observed operation tuple and adds only cross-microbatch
serialization edges. ``observed`` takes the tuple as emitted. Overlap is
measured against the control, so the TRAF-9 layer-ordering and terminal
frontier differences cancel exactly instead of being assumed away.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

CAPTURE_SHA256 = "5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6"
CAPTURE_ROWS = 120
REPLAY_RUN_SHA256 = "b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e"
ROUTED_EXPERTS_SHA256 = (
    "24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f"
)
VLLM_VERSION = "0.26.0"
VLLM_AUTHORED_AGAINST_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"
EXPECTATION_BOUNDARY_COMMIT = "4e1be35af5327c27db53ed002dc420e1de6f613b"

DP_SIZE = 8
TP_SIZE = 1
PP_SIZE = 1
NUM_LAYERS = 24
NUM_MICROBATCHES = 2
SEMANTIC_MOE_SITES = 48
DBO_MOE_INVOCATIONS = 96
DBO_DECODE_THRESHOLD = 2
DBO_PREFILL_THRESHOLD = 512

#: every nonempty translated step of the frozen replay
NONEMPTY_STEPS = 32
#: steps 1 through 23 carry r0's TPOT intervals and select DBO
R0_DECODE_STEP_INDICES = tuple(range(1, 24))
#: step 0 is the 54-token prefill; steps 24 through 31 are one-request decodes
SINGLE_BATCH_STEP_INDICES = (0, *range(24, 32))
#: 24 per-layer merge edges times 24 layers plus 8 per layer boundary
CONTROL_EDGES_PER_DBO_STEP = 760
CONTROL_EDGES_TOTAL = CONTROL_EDGES_PER_DBO_STEP * len(R0_DECODE_STEP_INDICES)

NVLINK_RATE_BPS = 3_600_000_000_000
RNIC_RATE_BPS = 400_000_000_000

#: summed maximum per-source egress over every DBO collective invocation of
#: r0's 23 TPOT steps, from the frozen routed table under the landed TRAF-25
#: token-ownership correction
DBO_DECODE_EGRESS_BYTES = 15_071_232
#: summed maximum per-source egress of the 54-token prefill step
PREFILL_EGRESS_BYTES = 15_249_408

#: ceiling on the realizable overlap: a step cannot save more than the
#: communication it contains
COMM_CEILING_PS = {
    "single-node": 1_456_158,
    "cross-node": 13_105_420,
}
OVERLAP_BAND_LOW_FRACTION = Fraction(3, 4)
OVERLAP_BAND_PS = {
    "single-node": (1_092_119, 1_456_158),
    "cross-node": (9_829_065, 13_105_420),
}
OVERLAP_RATE_RATIO_BAND = (Fraction(15, 2), Fraction(21, 2))
TERMINAL_RATE_RATIO_BAND = (Fraction(19, 20), Fraction(21, 20))
OVERLAP_CEILING_ALLOWANCE = Fraction(21, 20)

#: 553,648,128 resident weight and LM-head bytes at 8 TB/s
COMPUTE_FLOOR_PS = 69_206_016
DECODE_TPOT_CEILING_PS = 150_000_000
PREFILL_TTFT_CEILING_PS = 500_000_000

EXPECTED_GENUINE_RISK_INSTANCES = 6

SERIAL_GRAPH_BYTES = 4_127
SERIAL_GRAPH_SHA256 = (
    "aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d"
)
SERIAL_GOAL_BYTES = 1_880
SERIAL_GOAL_SHA256 = (
    "7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6"
)

SOURCE_HASHES = {
    "model_executor/models/granitemoe.py": (
        "b60e452c3f28b25aa104c88869daa25c06a7fb6ed45bd34e908fa6a8395efda1"
    ),
    "config/parallel.py": (
        "a6581c267ab265e24905d2f5caa514482c28359f71380c6f894ceab25aa22541"
    ),
    "v1/worker/gpu_model_runner.py": (
        "81b7627fbe81f7aaa2f77b4bf085faa353c69d03662ebfe369536a9773bb70d0"
    ),
    "v1/worker/dp_utils.py": (
        "2ba84bbf92a25e756576918bfb215c1fb387b006899885d811bdb2f774e843a9"
    ),
    "v1/worker/ubatch_utils.py": (
        "0b727aaa1c7072152e25f684ddc2fc9790c430eddd862e610c97a8e3e9febdc4"
    ),
    "v1/worker/gpu_ubatch_wrapper.py": (
        "4eae50c929f3ba873072c13291c7140be3dd00d4a5b623170dff44754519c021"
    ),
    "v1/worker/ubatching.py": (
        "40391241c564feb5f16c77898ae6ae152ed6e71a4682e2a406387785d8de02d7"
    ),
    "model_executor/layers/fused_moe/modular_kernel.py": (
        "f78ae626babfd69f3c6ba37eef9c8f5186f28cd9064f566e341ca0c9e0fdb9b9"
    ),
    "model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py": (
        "465cdf1d6cee91b2ee8c2e43abbea6e8408976e3048c10f44c089f34b415bc60"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _check_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if _sha256(path) != expected_sha256:
        raise SystemExit(f"{label} SHA-256 does not match the frozen input")


def _ceiling_ps(byte_count: int, steps: int, rate_bps: int) -> int:
    """Round the exact serialization ceiling up to whole picoseconds."""

    exact = Fraction(byte_count * 8, rate_bps * steps) * 10**12
    return -((-exact.numerator) // exact.denominator)


def check_expectation_registry(args: argparse.Namespace) -> None:
    """Validate frozen inputs and literals without importing target code."""

    _check_file(args.capture, CAPTURE_SHA256, "capture")
    if _line_count(args.capture) != CAPTURE_ROWS:
        raise SystemExit("capture row count does not match the frozen input")
    _check_file(args.replay_run, REPLAY_RUN_SHA256, "replay run")
    _check_file(args.routed_experts, ROUTED_EXPERTS_SHA256, "routed experts")

    if not args.vllm_source.is_dir():
        raise SystemExit(f"vLLM source is not a directory: {args.vllm_source}")
    for relative, expected in SOURCE_HASHES.items():
        _check_file(args.vllm_source / relative, expected, f"vLLM source {relative}")

    if not args.vllm_python.is_file():
        raise SystemExit(f"vLLM Python is not a file: {args.vllm_python}")
    observed_version = subprocess.run(
        [str(args.vllm_python), "-c", "import vllm; print(vllm.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_version != VLLM_VERSION:
        raise SystemExit(
            f"vLLM version must be {VLLM_VERSION}, got {observed_version!r}"
        )

    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")

    driver = (
        Path(__file__).resolve().parents[1]
        / "vllm_observed_schedule_v1"
        / "live_driver.py"
    )
    if not driver.is_file():
        raise SystemExit(f"reused live driver is missing: {driver}")

    assert len(EXPECTATION_BOUNDARY_COMMIT) == 40
    assert len(VLLM_AUTHORED_AGAINST_COMMIT) == 40
    assert (TP_SIZE, DP_SIZE, PP_SIZE) == (1, 8, 1)
    assert SEMANTIC_MOE_SITES == 2 * NUM_LAYERS
    assert DBO_MOE_INVOCATIONS == SEMANTIC_MOE_SITES * NUM_MICROBATCHES
    assert DBO_DECODE_THRESHOLD == 2
    assert DBO_PREFILL_THRESHOLD > 54

    # step partition: 23 DBO steps plus 9 single-batch steps
    assert len(R0_DECODE_STEP_INDICES) == 23
    assert len(SINGLE_BATCH_STEP_INDICES) == 9
    assert not set(R0_DECODE_STEP_INDICES) & set(SINGLE_BATCH_STEP_INDICES)
    assert (
        len(R0_DECODE_STEP_INDICES) + len(SINGLE_BATCH_STEP_INDICES)
        == NONEMPTY_STEPS
    )

    # control edges: three merge rules per layer plus one per layer boundary
    assert CONTROL_EDGES_PER_DBO_STEP == (
        3 * DP_SIZE * NUM_LAYERS + DP_SIZE * (NUM_LAYERS - 1)
    )
    assert CONTROL_EDGES_TOTAL == 17_480

    # the ceilings are arithmetic over the frozen routed table, not fitted
    for placement, rate in (
        ("single-node", NVLINK_RATE_BPS),
        ("cross-node", RNIC_RATE_BPS),
    ):
        expected_ceiling = _ceiling_ps(
            DBO_DECODE_EGRESS_BYTES,
            len(R0_DECODE_STEP_INDICES),
            rate,
        )
        if expected_ceiling != COMM_CEILING_PS[placement]:
            raise SystemExit(
                f"{placement} communication ceiling literal "
                f"{COMM_CEILING_PS[placement]} disagrees with the arithmetic "
                f"{expected_ceiling}"
            )
        low, high = OVERLAP_BAND_PS[placement]
        assert high == COMM_CEILING_PS[placement]
        assert low == -(
            (-(OVERLAP_BAND_LOW_FRACTION * high).numerator)
            // (OVERLAP_BAND_LOW_FRACTION * high).denominator
        )
        assert 0 < low < high
    assert (
        _ceiling_ps(PREFILL_EGRESS_BYTES, 1, RNIC_RATE_BPS) == 304_988_160
    )
    assert Fraction(NVLINK_RATE_BPS, RNIC_RATE_BPS) == 9
    assert (
        OVERLAP_RATE_RATIO_BAND[0] < 9 < OVERLAP_RATE_RATIO_BAND[1]
    )
    assert TERMINAL_RATE_RATIO_BAND[0] < 1 < TERMINAL_RATE_RATIO_BAND[1]
    assert OVERLAP_CEILING_ALLOWANCE > 1

    assert 0 < COMPUTE_FLOOR_PS < DECODE_TPOT_CEILING_PS
    assert DECODE_TPOT_CEILING_PS < PREFILL_TTFT_CEILING_PS
    assert EXPECTED_GENUINE_RISK_INSTANCES == 6
    assert SERIAL_GRAPH_BYTES > SERIAL_GOAL_BYTES > 0
    assert len(SERIAL_GRAPH_SHA256) == len(SERIAL_GOAL_SHA256) == 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--routed-experts", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--vllm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _canonical_identity(value: object) -> dict[str, object]:
    wire = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {
        "bytes": len(wire),
        "sha256": hashlib.sha256(wire).hexdigest(),
    }


def _edge_free_identity(graph_json: dict[str, object]) -> dict[str, object]:
    """Identity of a lowered graph with every dependency edge removed.

    Two arms that differ only by added edges must agree on this identity, so
    it is what proves the control arm changed nothing else.
    """

    stripped = copy.deepcopy(graph_json)
    for operation in stripped.get("operations", []):
        operation.pop("depends_on", None)
        operation.pop("participant_local_depends_on", None)
    stripped.pop("collective_plans", None)
    # every arm keeps its own clock, so an arm that runs faster releases later
    # steps earlier; the frozen field list does not include the release time
    stripped["released_at_ps"] = 0
    return _canonical_identity(stripped)


def _fraction_to_json(value: Fraction | None) -> dict[str, int | float] | None:
    if value is None:
        return None
    return {
        "denominator": value.denominator,
        "float_ps": float(value),
        "numerator": value.numerator,
    }


def _fraction_from_json(value: dict[str, object]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _model_dims():
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=1_024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_155,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=32 // DP_SIZE,
    )


def _routed_supply(path: Path):
    from simllm.preplay import read_routed_experts
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % DP_SIZE)
            for layer in range(NUM_LAYERS)
            for expert in range(32)
        ),
    )
    return RoutedMoeSupply(
        placements=(placement,),
        step_placement_epochs=tuple((step, 0) for step in range(4_096)),
        routed_experts=read_routed_experts(path),
        engine_rank=0,
    )


def _profile(placement: str):
    from simllm.core import CoarseDeviceProfile

    class _OneRankPerNodeProfile(CoarseDeviceProfile):
        def node_gpu(self, rank: int) -> tuple[int, int]:
            super().node_gpu(rank)
            return rank, 0

    if placement == "single-node":
        return CoarseDeviceProfile(nvlink_rate_bps=NVLINK_RATE_BPS)
    if placement == "cross-node":
        return _OneRankPerNodeProfile(rnic_rate_bps=RNIC_RATE_BPS)
    raise ValueError(f"unknown placement {placement!r}")


def _observations_from_source(source: object):
    from simllm.core import ExecutionObservations, execution_graph_from_json

    if source is None:
        return None
    graph = execution_graph_from_json(source)
    return ExecutionObservations(
        operations=graph.operations,
        completion_operation_ids=graph.completion_operation_ids,
    )


def _control_observations(observations, step_index: int):
    """Serialize the two microbatches without touching anything else.

    Both shared logical queues already impose FIFO in submission order, so the
    only total order the control can impose is the interleaving of those two
    chains: pre-dispatch zero, pre-dispatch one, dispatch zero, experts zero,
    dispatch one, experts one, combine zero, combine one, next layer. The
    edges below are the ones that order is missing.
    """

    from simllm.core import ExecutionObservations

    if observations is None:
        return None, 0

    def pre(microbatch: int, layer: int, rank: int) -> str:
        return (
            f"step-{step_index}:ubatch-{microbatch}:layer-{layer}"
            f":rank-{rank}:pre-dispatch"
        )

    def experts(microbatch: int, layer: int, rank: int) -> str:
        return (
            f"step-{step_index}:ubatch-{microbatch}:layer-{layer}"
            f":rank-{rank}:experts"
        )

    def dispatch(microbatch: int, layer: int) -> str:
        return f"step-{step_index}:ubatch-{microbatch}:layer-{layer}:ep-dispatch"

    def combine(microbatch: int, layer: int) -> str:
        return f"step-{step_index}:ubatch-{microbatch}:layer-{layer}:ep-combine"

    operation_ids = {operation.operation_id for operation in observations.operations}
    if dispatch(1, 0) not in operation_ids:
        # a single-batch step has no second microbatch to serialize
        return observations, 0

    ranks = range(DP_SIZE)
    additions: dict[str, list[str]] = {}
    for layer in range(NUM_LAYERS):
        additions.setdefault(dispatch(0, layer), []).extend(
            pre(1, layer, rank) for rank in ranks
        )
        additions.setdefault(dispatch(1, layer), []).extend(
            experts(0, layer, rank) for rank in ranks
        )
        additions.setdefault(combine(0, layer), []).extend(
            experts(1, layer, rank) for rank in ranks
        )
        if layer + 1 < NUM_LAYERS:
            for rank in ranks:
                additions.setdefault(pre(0, layer + 1, rank), []).append(
                    combine(1, layer)
                )
    unknown = sorted(
        {
            operation_id
            for target, sources in additions.items()
            for operation_id in (target, *sources)
        }
        - operation_ids
    )
    if unknown:
        raise RuntimeError(f"control arm references absent operations: {unknown}")

    added = 0
    operations = []
    for operation in observations.operations:
        extra = additions.get(operation.operation_id)
        if extra:
            if set(extra) & set(operation.depends_on):
                raise RuntimeError("control edge already exists in the observed tuple")
            operation = replace(operation, depends_on=(*operation.depends_on, *extra))
            added += len(extra)
        operations.append(operation)
    return (
        ExecutionObservations(
            operations=tuple(operations),
            completion_operation_ids=observations.completion_operation_ids,
        ),
        added,
    )


def _tuple_fields_agree(observed, control) -> bool:
    """Whether two tuples agree on everything except whole-operation edges."""

    if observed.completion_operation_ids != control.completion_operation_ids:
        return False
    if len(observed.operations) != len(control.operations):
        return False
    for source, target in zip(observed.operations, control.operations, strict=True):
        if (
            source.operation_id != target.operation_id
            or source.rank != target.rank
            or source.logical_queue != target.logical_queue
            or source.work != target.work
            or source.correlation != target.correlation
            or source.not_before_ps != target.not_before_ps
            or source.priority != target.priority
            or source.placement_epoch != target.placement_epoch
            or source.participant_local_depends_on != target.participant_local_depends_on
        ):
            return False
        if not set(source.depends_on) <= set(target.depends_on):
            return False
    return True


def _schedule_fields_match(source, lowered) -> bool:
    from simllm.core import CollectiveWork, ComputeWork

    if source.completion_operation_ids != lowered.completion_operation_ids:
        return False
    if len(source.operations) != len(lowered.operations):
        return False
    for observed, bound in zip(source.operations, lowered.operations, strict=True):
        if (
            observed.operation_id != bound.operation_id
            or observed.rank != bound.rank
            or observed.logical_queue != bound.logical_queue
            or observed.depends_on != bound.depends_on
            or observed.participant_local_depends_on
            != bound.participant_local_depends_on
            or observed.not_before_ps != bound.not_before_ps
            or observed.priority != bound.priority
            or observed.correlation != bound.correlation
        ):
            return False
        if isinstance(observed.work, ComputeWork):
            if observed.work != bound.work:
                return False
        elif isinstance(observed.work, CollectiveWork):
            if not isinstance(bound.work, CollectiveWork):
                return False
            if (
                observed.work.collective != bound.work.collective
                or observed.work.ranks != bound.work.ranks
                or observed.work.channel_hint != bound.work.channel_hint
            ):
                return False
        else:
            return False
    return True


def _request_pair_inventory(graph) -> tuple[tuple[object, ...], ...]:
    from simllm.core import CollectiveWork

    return tuple(
        sorted(
            (
                operation.correlation.layer,
                operation.work.channel_hint,
                request_id,
                source,
                destination,
                size,
            )
            for operation in graph.operations
            if isinstance(operation.work, CollectiveWork)
            for request_id, source, destination, size in (
                operation.work.request_pair_payload_bytes
            )
        )
    )


def _merged_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, finish in sorted(spans):
        if merged and start <= merged[-1][1]:
            if finish > merged[-1][1]:
                merged[-1] = (merged[-1][0], finish)
            continue
        merged.append((start, finish))
    return merged


def _cross_microbatch_overlaps(report) -> int:
    """Count intersecting busy intervals of microbatch zero and one.

    Zero exactly when no queue visit of one microbatch overlaps in realized
    time with any queue visit of the other.
    """

    sides: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for visit in report.visits:
        if visit.finished_at_ps <= visit.started_at_ps:
            continue
        if ":ubatch-0:" in visit.operation_id:
            sides[0].append((visit.started_at_ps, visit.finished_at_ps))
        elif ":ubatch-1:" in visit.operation_id:
            sides[1].append((visit.started_at_ps, visit.finished_at_ps))
    left = _merged_spans(sides[0])
    right = _merged_spans(sides[1])
    index = 0
    overlaps = 0
    for start, finish in left:
        while index < len(right) and right[index][1] <= start:
            index += 1
        cursor = index
        while cursor < len(right) and right[cursor][0] < finish:
            overlaps += 1
            cursor += 1
    return overlaps


def _phase_timestamps(outcome) -> tuple[int, int]:
    """Return the last collective completion and the step completion."""

    from simllm.core import CollectiveWork

    collective_ids = {
        operation.operation_id
        for operation in outcome.graph.operations
        if isinstance(operation.work, CollectiveWork)
    }
    completed = outcome.execution_result.completed_at_ps
    moe_end = max(
        (
            record.completed_at_ps
            for record in outcome.runtime_report.operations
            if record.operation_id in collective_ids
        ),
        default=outcome.graph.released_at_ps,
    )
    return moe_end, completed


def _critical_path_split(outcome) -> tuple[int, int]:
    """Split the realized critical path into collective and compute time."""

    from simllm.core import CollectiveWork

    collective_ids = {
        operation.operation_id
        for operation in outcome.graph.operations
        if isinstance(operation.work, CollectiveWork)
    }
    latency_by_key = {
        (segment.operation_id, segment.participant_rank): (
            segment.breakdown.operation_latency_ps
        )
        for record in outcome.runtime_report.operations
        for segment in record.critical_segments
    }
    communication = 0
    compute = 0
    for key in outcome.runtime_report.realized_critical_path_segments:
        duration = latency_by_key[key]
        if key[0] in collective_ids:
            communication += duration
        else:
            compute += duration
    return communication, compute


def _mode_run(
    placement: str,
    mode: str,
    live_path: str,
    routed_experts: str,
    raw_output: str,
) -> dict[str, object]:
    from simllm.backends import (
        DeviceRuntimeStepSink,
        SerialStepLowerer,
        SerialStepLowererConfig,
    )
    from simllm.compute import RooflineProvider
    from simllm.core import (
        CoarseDeviceRuntime,
        CompletionReducer,
        VirtualClock,
        execution_graph_to_json,
        execution_result_to_json,
        step_record_from_json,
        step_result_to_json,
    )
    from simllm.traffic import project_execution_graph_goal, render_step_goal

    profile = _profile(placement)
    config = SerialStepLowererConfig(
        _model_dims(),
        (0,),
        ep_ranks=tuple(range(DP_SIZE)),
        provider=RooflineProvider(efficiency=0.7),
        routed_moe_supply=_routed_supply(Path(routed_experts)),
    )
    clock = VirtualClock()
    sink = DeviceRuntimeStepSink(config, runtime=CoarseDeviceRuntime(profile))
    sink.bind_clock(clock)
    direct_clock = VirtualClock()
    direct_reducer = CompletionReducer(direct_clock)
    direct_runtime = CoarseDeviceRuntime(profile)
    direct_lowerer = SerialStepLowerer(config)

    raw_rows: list[dict[str, object]] = []
    last_r0 = None
    control_edges_by_step: dict[str, int] = {}
    tuple_field_checks: list[bool] = []
    schedule_checks: list[bool] = []
    serial_checks: list[bool] = []
    live_checks: list[bool] = []
    serial_goal_identities = []
    serial_projection_identities = []
    request_pair_rows = 0
    request_pair_bytes = 0
    request_metric_ids: list[list[str]] = []

    with Path(live_path).open("r", encoding="utf-8") as stream:
        live_rows = [json.loads(line) for line in stream if line.strip()]
    for live_row in live_rows:
        source_record = step_record_from_json(live_row["record"])
        record = replace(source_record, virtual_time_ps=clock.now_ps)
        source_observations = _observations_from_source(live_row["source_graph"])
        observations = source_observations
        edges = 0
        if mode == "serial":
            observations = None
        elif mode == "control":
            observations, edges = _control_observations(
                source_observations,
                record.step_index,
            )
            control_edges_by_step[str(record.step_index)] = edges
            if source_observations is not None:
                tuple_field_checks.append(
                    _tuple_fields_agree(source_observations, observations)
                )
        elif mode != "observed":
            raise ValueError(f"unknown execution mode {mode!r}")

        step_result = sink(record, observations)
        outcome = sink.outcomes[-1]
        graph_json = execution_graph_to_json(outcome.graph)
        execution_json = execution_result_to_json(outcome.execution_result)
        step_json = step_result_to_json(step_result)
        graph_identity = _canonical_identity(graph_json)
        execution_identity = _canonical_identity(execution_json)
        pair_inventory = _request_pair_inventory(outcome.graph)
        request_pair_rows += len(pair_inventory)
        request_pair_bytes += sum(int(row[-1]) for row in pair_inventory)
        moe_end_ps, completed_ps = _phase_timestamps(outcome)
        communication_ps, compute_ps = _critical_path_split(outcome)

        if observations is not None:
            schedule_checks.append(
                _schedule_fields_match(observations, outcome.graph)
            )

        if mode == "serial":
            if direct_clock.now_ps != record.virtual_time_ps:
                raise RuntimeError("serial direct clock drifted before comparison")
            direct_graph = direct_lowerer.lower(record)
            direct_execution = direct_runtime.execute(direct_graph)
            direct_step = direct_reducer.reduce(
                record,
                direct_graph,
                direct_execution,
                direct_runtime.last_report,
            )
            projection = project_execution_graph_goal(direct_graph)
            serial_projection_identities.append(
                {
                    "artifacts": [
                        {
                            "bytes": len(artifact.trace.render().encode()),
                            "sha256": hashlib.sha256(
                                artifact.trace.render().encode()
                            ).hexdigest(),
                        }
                        for artifact in projection.artifacts
                    ],
                    "boundaries": len(projection.boundaries),
                    "serialized_edges": len(projection.serialized_edges),
                    "step_index": record.step_index,
                }
            )
            if record.total_new_tokens > 0:
                goal = render_step_goal(
                    record,
                    config.dims,
                    config.tp_ranks,
                    per_layer_calc_ns=direct_lowerer.timing(record).layer_calc_ns,
                    ep_ranks=config.ep_ranks,
                    routed_supply=config.routed_moe_supply,
                ).render().encode()
                serial_goal_identities.append(
                    {
                        "bytes": len(goal),
                        "sha256": hashlib.sha256(goal).hexdigest(),
                        "step_index": record.step_index,
                    }
                )
            serial_checks.append(
                graph_json == execution_graph_to_json(direct_graph)
                and execution_json == execution_result_to_json(direct_execution)
                and step_json == step_result_to_json(direct_step)
            )

        if placement == "cross-node" and mode == "observed":
            live_checks.append(
                graph_identity == live_row["lowered_graph_identity"]
                and execution_identity == live_row["execution_result_identity"]
                and step_json == live_row["step_result"]
            )

        for metric in step_result.request_metrics:
            if metric.request_id == "r0":
                last_r0 = metric
        request_metric_ids.append(
            [metric.request_id for metric in step_result.request_metrics]
        )
        raw_rows.append(
            {
                "control_edges": edges,
                "critical_path_communication_ps": communication_ps,
                "critical_path_compute_ps": compute_ps,
                "cross_microbatch_overlaps": _cross_microbatch_overlaps(
                    outcome.runtime_report
                ),
                "edge_free_graph_identity": _edge_free_identity(graph_json),
                "execution_result_identity": execution_identity,
                "graph_identity": graph_identity,
                "latency_ps": completed_ps - record.virtual_time_ps,
                "moe_phase_ps": moe_end_ps - record.virtual_time_ps,
                "released_at_ps": record.virtual_time_ps,
                "request_pair_identity": _canonical_identity(pair_inventory),
                "step_index": record.step_index,
                "step_result": step_json,
                "terminal_ps": completed_ps - moe_end_ps,
            }
        )

    if last_r0 is None or last_r0.tpot_ps is None:
        raise RuntimeError(f"{placement} {mode} produced no r0 TPOT")
    output = Path(raw_output)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in raw_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "control_edges_by_step": control_edges_by_step,
        "control_tuple_field_checks": tuple_field_checks,
        "live_identity_checks": live_checks,
        "mode": mode,
        "placement": placement,
        "r0_tpot_ps": _fraction_to_json(last_r0.tpot_ps),
        "r0_ttft_ps": last_r0.ttft_ps,
        "raw_rows": raw_rows,
        "request_metric_ids": request_metric_ids,
        "request_pair_bytes": request_pair_bytes,
        "request_pair_rows": request_pair_rows,
        "schedule_preservation_checks": schedule_checks,
        "serial_direct_checks": serial_checks,
        "serial_goal_identities": serial_goal_identities,
        "serial_projection_identities": serial_projection_identities,
    }


def _run_live_driver(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[2]
    driver = (
        Path(__file__).resolve().parents[1]
        / "vllm_observed_schedule_v1"
        / "live_driver.py"
    )
    log_path = args.output_dir / "live_driver.log"
    environment = dict(os.environ)
    previous_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo)
        if not previous_pythonpath
        else str(repo) + os.pathsep + previous_pythonpath
    )
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    command = [
        str(args.vllm_python),
        str(driver),
        "--capture",
        str(args.capture),
        "--replay-run",
        str(args.replay_run),
        "--routed-experts",
        str(args.routed_experts),
        "--output-dir",
        str(args.output_dir),
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"live vLLM driver failed with {completed.returncode}; see {log_path}"
        )
    live_path = args.output_dir / "live_observations.jsonl"
    if not live_path.is_file():
        raise RuntimeError("live vLLM driver produced no observation stream")
    return live_path


def _run_modes(
    args: argparse.Namespace,
    live_path: Path,
) -> dict[str, dict[str, object]]:
    jobs = [
        (placement, mode)
        for placement in COMM_CEILING_PS
        for mode in ("serial", "control", "observed")
    ]
    results: dict[str, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        pending = {
            executor.submit(
                _mode_run,
                placement,
                mode,
                str(live_path),
                str(args.routed_experts),
                str(args.output_dir / f"raw_{placement}_{mode}.jsonl"),
            ): (placement, mode)
            for placement, mode in jobs
        }
        for future in as_completed(pending):
            placement, mode = pending[future]
            results[f"{placement}:{mode}"] = future.result()
    return results


def _rows_by_step(result: dict[str, object]) -> dict[int, dict[str, object]]:
    return {int(row["step_index"]): row for row in result["raw_rows"]}


def _mean(values: list[int]) -> Fraction:
    return Fraction(sum(values), len(values))


def _decomposition(modes: dict[str, dict[str, object]]) -> dict[str, object]:
    """Split serial minus observed into structure and overlap, per placement."""

    rows = []
    for placement in COMM_CEILING_PS:
        serial = _rows_by_step(modes[f"{placement}:serial"])
        control = _rows_by_step(modes[f"{placement}:control"])
        observed = _rows_by_step(modes[f"{placement}:observed"])
        decode = list(R0_DECODE_STEP_INDICES)
        rows.append(
            {
                "layer_ordering_ps": _fraction_to_json(
                    _mean([serial[i]["moe_phase_ps"] for i in decode])
                    - _mean([control[i]["moe_phase_ps"] for i in decode])
                ),
                "mean_control_latency_ps": _fraction_to_json(
                    _mean([control[i]["latency_ps"] for i in decode])
                ),
                "mean_observed_latency_ps": _fraction_to_json(
                    _mean([observed[i]["latency_ps"] for i in decode])
                ),
                "mean_serial_latency_ps": _fraction_to_json(
                    _mean([serial[i]["latency_ps"] for i in decode])
                ),
                "overlap_latency_ps": _fraction_to_json(
                    _mean([control[i]["latency_ps"] for i in decode])
                    - _mean([observed[i]["latency_ps"] for i in decode])
                ),
                "placement": placement,
                "structure_latency_ps": _fraction_to_json(
                    _mean([serial[i]["latency_ps"] for i in decode])
                    - _mean([control[i]["latency_ps"] for i in decode])
                ),
                "terminal_control_ps": _fraction_to_json(
                    _mean([control[i]["terminal_ps"] for i in decode])
                ),
                "terminal_observed_ps": _fraction_to_json(
                    _mean([observed[i]["terminal_ps"] for i in decode])
                ),
                "terminal_serial_ps": _fraction_to_json(
                    _mean([serial[i]["terminal_ps"] for i in decode])
                ),
                "ttft_observed_ps": observed[0]["latency_ps"],
                "ttft_serial_ps": serial[0]["latency_ps"],
                "critical_path_communication_ps": {
                    arm: _fraction_to_json(
                        _mean(
                            [
                                cells[i]["critical_path_communication_ps"]
                                for i in decode
                            ]
                        )
                    )
                    for arm, cells in (
                        ("control", control),
                        ("observed", observed),
                        ("serial", serial),
                    )
                },
            }
        )
    return {"rows": rows}


def _evaluate_scored(
    modes: dict[str, dict[str, object]],
    decomposition: dict[str, object],
) -> dict[str, object]:
    rows = []
    passed = 0
    overlap_by_placement: dict[str, Fraction] = {}
    terminal_by_placement: dict[str, Fraction] = {}
    for placement, (low, high) in OVERLAP_BAND_PS.items():
        control_tpot = _fraction_from_json(modes[f"{placement}:control"]["r0_tpot_ps"])
        observed_tpot = _fraction_from_json(
            modes[f"{placement}:observed"]["r0_tpot_ps"]
        )
        serial_tpot = _fraction_from_json(modes[f"{placement}:serial"]["r0_tpot_ps"])
        overlap = control_tpot - observed_tpot
        overlap_by_placement[placement] = overlap
        band_passed = low <= overlap <= high
        passed += int(band_passed)

        serial_ttft = modes[f"{placement}:serial"]["r0_ttft_ps"]
        observed_ttft = modes[f"{placement}:observed"]["r0_ttft_ps"]
        ttft_passed = serial_ttft > observed_ttft
        passed += int(ttft_passed)

        terminal = next(
            _fraction_from_json(row["terminal_observed_ps"])
            for row in decomposition["rows"]
            if row["placement"] == placement
        )
        terminal_by_placement[placement] = terminal
        rows.append(
            {
                "b1_overlap_band_passed": band_passed,
                "b4_ttft_direction_passed": ttft_passed,
                "control_tpot_ps": _fraction_to_json(control_tpot),
                "observed_tpot_ps": _fraction_to_json(observed_tpot),
                "observed_ttft_ps": observed_ttft,
                "overlap_band_ps": [low, high],
                "overlap_fraction_of_ceiling": float(
                    Fraction(overlap) / COMM_CEILING_PS[placement]
                ),
                "overlap_tpot_ps": _fraction_to_json(overlap),
                "placement": placement,
                "serial_tpot_ps": _fraction_to_json(serial_tpot),
                "serial_ttft_ps": serial_ttft,
                "terminal_observed_ps": _fraction_to_json(terminal),
            }
        )

    overlap_ratio = (
        overlap_by_placement["cross-node"] / overlap_by_placement["single-node"]
        if overlap_by_placement["single-node"]
        else None
    )
    terminal_ratio = (
        terminal_by_placement["cross-node"] / terminal_by_placement["single-node"]
        if terminal_by_placement["single-node"]
        else None
    )
    overlap_ratio_passed = overlap_ratio is not None and (
        OVERLAP_RATE_RATIO_BAND[0] <= overlap_ratio <= OVERLAP_RATE_RATIO_BAND[1]
    )
    terminal_ratio_passed = terminal_ratio is not None and (
        TERMINAL_RATE_RATIO_BAND[0] <= terminal_ratio <= TERMINAL_RATE_RATIO_BAND[1]
    )
    passed += int(overlap_ratio_passed) + int(terminal_ratio_passed)
    return {
        "b2_overlap_rate_ratio": None if overlap_ratio is None else float(overlap_ratio),
        "b2_overlap_rate_ratio_passed": overlap_ratio_passed,
        "b3_terminal_rate_ratio": (
            None if terminal_ratio is None else float(terminal_ratio)
        ),
        "b3_terminal_rate_ratio_passed": terminal_ratio_passed,
        "evaluation_order": "raw arm latencies before every exact and fatal guard",
        "genuine_risk_executed": EXPECTED_GENUINE_RISK_INSTANCES,
        "genuine_risk_passed": passed,
        "rows": rows,
    }


def _producer_inventory(live_path: Path) -> dict[str, object]:
    from simllm.backends import SerialStepLowerer, SerialStepLowererConfig
    from simllm.compute import RooflineProvider
    from simllm.core import (
        CollectiveWork,
        ComputeWork,
        execution_graph_from_json,
        step_record_from_json,
    )

    rows = [
        json.loads(line)
        for line in live_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    nonempty = 0
    emitted = 0
    unique_site_counts = []
    invocation_counts = []
    ordered_layer_checks = []
    completion_checks = []
    compute_conservation = []
    request_partitions = []
    lowerer = SerialStepLowerer(
        SerialStepLowererConfig(
            _model_dims(),
            (0,),
            ep_ranks=tuple(range(DP_SIZE)),
            provider=RooflineProvider(efficiency=0.7),
        )
    )
    for row in rows:
        record = step_record_from_json(row["record"])
        if record.total_new_tokens <= 0:
            continue
        nonempty += 1
        source = row["source_graph"]
        if source is None:
            continue
        emitted += 1
        graph = execution_graph_from_json(source)
        collectives = [
            operation
            for operation in graph.operations
            if isinstance(operation.work, CollectiveWork)
        ]
        sites = {
            (operation.correlation.layer, operation.work.channel_hint)
            for operation in collectives
        }
        unique_site_counts.append(len(sites))
        invocation_counts.append(len(collectives))
        ordered_layer_checks.append(
            sites
            == {
                (layer, site)
                for layer in range(NUM_LAYERS)
                for site in ("dispatch", "combine")
            }
        )
        microbatches = sorted(
            {operation.correlation.microbatch for operation in collectives},
            key=lambda value: -1 if value is None else value,
        )
        expected_completions = 1 if microbatches == [None] else len(microbatches)
        completion_checks.append(
            len(graph.completion_operation_ids) == expected_completions
            and all(
                any(
                    operation.operation_id == completion_id
                    and operation.correlation.request_ids
                    for operation in graph.operations
                )
                for completion_id in graph.completion_operation_ids
            )
        )
        correlated = {
            microbatch: next(
                operation.correlation.request_ids
                for operation in collectives
                if operation.correlation.microbatch == microbatch
            )
            for microbatch in microbatches
        }
        flattened = tuple(
            request_id
            for microbatch in microbatches
            for request_id in correlated[microbatch]
        )
        request_partitions.append(
            flattened == tuple(request.request_id for request in record.scheduled)
            and sum(len(value) for value in correlated.values()) == len(set(flattened))
        )
        rank_zero_compute = sum(
            operation.work.nominal_duration_ps or 0
            for operation in graph.operations
            if operation.rank == 0 and isinstance(operation.work, ComputeWork)
        )
        represented_serial = sum(lowerer.timing(record).layer_calc_ns) * 1_000
        compute_conservation.append(rank_zero_compute == represented_serial)

    schedule_source = (
        Path(__file__).resolve().parents[2]
        / "simllm"
        / "adapters"
        / "vllm"
        / "schedule.py"
    ).read_text(encoding="utf-8")
    forbidden_knobs = [
        token
        for token in (
            "overlap_fraction",
            "overlap_percent",
            "overlap_percentage",
            "overlap_ps",
        )
        if token in schedule_source
    ]
    return {
        "completion_frontier_checks": completion_checks,
        "compute_conservation_checks": compute_conservation,
        "dbo_invocation_rows": sum(
            count == DBO_MOE_INVOCATIONS for count in invocation_counts
        ),
        "emitted_observation_steps": emitted,
        "forbidden_overlap_knobs": forbidden_knobs,
        "invocation_counts": invocation_counts,
        "nonempty_translated_steps": nonempty,
        "ordered_layer_checks": ordered_layer_checks,
        "request_partition_checks": request_partitions,
        "single_batch_invocation_rows": sum(
            count == SEMANTIC_MOE_SITES for count in invocation_counts
        ),
        "unique_semantic_site_counts": unique_site_counts,
    }


def _serial_fixture() -> dict[str, object]:
    from simllm.backends import (
        ObservedStepLowerer,
        SerialStepLowerer,
        SerialStepLowererConfig,
    )
    from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
    from simllm.core import (
        RequestPhase,
        ScheduledRequest,
        StepRecord,
        execution_graph_to_json,
    )
    from simllm.traffic import render_step_goal

    class _FlopProvider(ComputeProvider):
        def estimate(self, kernel, gpu):
            return DurationEstimate(duration_ps=int(kernel.flops), bound="compute")

    dims = ModelDims(2, 64, 128, 4, 4, 16, 256, 2)
    record = StepRecord(
        0,
        0,
        [
            ScheduledRequest(
                "p",
                RequestPhase.PREFILL,
                4,
                num_cached_tokens=4,
                context_length=8,
            ),
            ScheduledRequest("d", RequestPhase.DECODE, 1, context_length=32),
        ],
    )
    # the accepted fixture is the absent-plan wire form; the plan-attached
    # default must differ from it by the plan alone
    config = SerialStepLowererConfig(
        dims,
        (0, 1),
        provider=_FlopProvider(),
        attach_collective_plan=False,
    )
    default_config = replace(config, attach_collective_plan=True)
    graph = ObservedStepLowerer(config).lower(record, None)
    default = ObservedStepLowerer(default_config).lower(record)
    direct = SerialStepLowerer(config).lower(record)
    wire = (
        json.dumps(
            execution_graph_to_json(graph),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    goal = render_step_goal(
        record,
        dims,
        config.tp_ranks,
        per_layer_calc_ns=SerialStepLowerer(config).timing(record).layer_calc_ns,
    ).render().encode()
    return {
        "default_differs_only_by_plan": (
            replace(default, collective_plans=()) == graph
            and len(default.collective_plans) > 0
        ),
        "direct_graph_equal": execution_graph_to_json(graph)
        == execution_graph_to_json(direct),
        "goal_bytes": len(goal),
        "goal_sha256": hashlib.sha256(goal).hexdigest(),
        "graph_bytes": len(wire),
        "graph_sha256": hashlib.sha256(wire).hexdigest(),
    }


def _fatal_guards(
    modes: dict[str, dict[str, object]],
    scored: dict[str, object],
    inventory: dict[str, object],
    fixture: dict[str, object],
) -> dict[str, object]:
    placements = tuple(COMM_CEILING_PS)
    edge_free_agrees = True
    edge_counts_agree = True
    control_disjoint = True
    single_batch_zero = True
    decomposition_exact = True
    tpot_in_bounds = True
    ttft_in_bounds = True
    overlap_in_ceiling = True

    for placement in placements:
        serial = _rows_by_step(modes[f"{placement}:serial"])
        control = _rows_by_step(modes[f"{placement}:control"])
        observed = _rows_by_step(modes[f"{placement}:observed"])
        for step_index, row in control.items():
            expected = (
                CONTROL_EDGES_PER_DBO_STEP
                if step_index in R0_DECODE_STEP_INDICES
                else 0
            )
            edge_counts_agree = edge_counts_agree and row["control_edges"] == expected
            edge_free_agrees = (
                edge_free_agrees
                and row["edge_free_graph_identity"]
                == observed[step_index]["edge_free_graph_identity"]
            )
            control_disjoint = (
                control_disjoint and row["cross_microbatch_overlaps"] == 0
            )
            structure = serial[step_index]["latency_ps"] - row["latency_ps"]
            overlap = row["latency_ps"] - observed[step_index]["latency_ps"]
            total = serial[step_index]["latency_ps"] - observed[step_index]["latency_ps"]
            decomposition_exact = decomposition_exact and total == structure + overlap
            split = (
                serial[step_index]["moe_phase_ps"] - row["moe_phase_ps"]
            ) - (row["terminal_ps"] - serial[step_index]["terminal_ps"])
            decomposition_exact = decomposition_exact and split == structure
            if step_index in SINGLE_BATCH_STEP_INDICES:
                single_batch_zero = single_batch_zero and overlap == 0

        for arm in ("serial", "control", "observed"):
            tpot = _fraction_from_json(modes[f"{placement}:{arm}"]["r0_tpot_ps"])
            ttft = modes[f"{placement}:{arm}"]["r0_ttft_ps"]
            tpot_in_bounds = tpot_in_bounds and (
                COMPUTE_FLOOR_PS <= tpot <= DECODE_TPOT_CEILING_PS
            )
            ttft_in_bounds = ttft_in_bounds and (
                COMPUTE_FLOOR_PS <= ttft <= PREFILL_TTFT_CEILING_PS
            )

        overlap_tpot = next(
            _fraction_from_json(row["overlap_tpot_ps"])
            for row in scored["rows"]
            if row["placement"] == placement
        )
        overlap_in_ceiling = overlap_in_ceiling and (
            overlap_tpot <= OVERLAP_CEILING_ALLOWANCE * COMM_CEILING_PS[placement]
        )

    pair_identities = [
        [row["request_pair_identity"] for row in modes[f"{placement}:{mode}"]["raw_rows"]]
        for placement in placements
        for mode in ("serial", "control", "observed")
    ]
    active_metric_ids = [
        ids
        for placement in placements
        for mode in ("serial", "control", "observed")
        for ids in modes[f"{placement}:{mode}"]["request_metric_ids"]
    ]
    return {
        "control_differs_only_by_edges": (
            edge_free_agrees
            and all(
                check
                for placement in placements
                for check in modes[f"{placement}:control"]["control_tuple_field_checks"]
            )
        ),
        "control_edge_counts": (
            edge_counts_agree
            and all(
                sum(modes[f"{placement}:control"]["control_edges_by_step"].values())
                == CONTROL_EDGES_TOTAL
                for placement in placements
            )
        ),
        "control_has_no_cross_microbatch_concurrency": control_disjoint,
        "decode_tpot_bounds": tpot_in_bounds,
        "decomposition_identity": decomposition_exact,
        "live_cross_node_identity": bool(
            modes["cross-node:observed"]["live_identity_checks"]
        )
        and all(modes["cross-node:observed"]["live_identity_checks"]),
        "overlap_within_ceiling": overlap_in_ceiling,
        "overlap_zero_on_single_batch_steps": single_batch_zero,
        "prefill_ttft_bounds": ttft_in_bounds,
        "producer_completion_frontiers": all(inventory["completion_frontier_checks"]),
        "producer_compute_conservation": all(inventory["compute_conservation_checks"]),
        "producer_every_nonempty_step": (
            inventory["nonempty_translated_steps"] == NONEMPTY_STEPS
            and inventory["emitted_observation_steps"] == NONEMPTY_STEPS
        ),
        "producer_layer_and_site_inventory": (
            all(inventory["ordered_layer_checks"])
            and all(
                count == SEMANTIC_MOE_SITES
                for count in inventory["unique_semantic_site_counts"]
            )
            and inventory["dbo_invocation_rows"] == len(R0_DECODE_STEP_INDICES)
            and inventory["single_batch_invocation_rows"]
            == len(SINGLE_BATCH_STEP_INDICES)
        ),
        "producer_no_overlap_knob": not inventory["forbidden_overlap_knobs"],
        "producer_request_partitions": all(inventory["request_partition_checks"]),
        "request_pair_rebinding_exact": (
            bool(modes["single-node:serial"]["request_pair_rows"])
            and all(identity == pair_identities[0] for identity in pair_identities[1:])
            and all(
                set(ids) <= {"r0", "r1", "r2"} and ids
                for ids in active_metric_ids
            )
        ),
        "schedule_fields_survive_lowering": all(
            check
            for placement in placements
            for mode in ("control", "observed")
            for check in modes[f"{placement}:{mode}"]["schedule_preservation_checks"]
        ),
        "serial_direct_all_steps": all(
            check
            for placement in placements
            for check in modes[f"{placement}:serial"]["serial_direct_checks"]
        ),
        "serial_fixture": (
            fixture["default_differs_only_by_plan"]
            and fixture["direct_graph_equal"]
            and fixture["graph_bytes"] == SERIAL_GRAPH_BYTES
            and fixture["graph_sha256"] == SERIAL_GRAPH_SHA256
            and fixture["goal_bytes"] == SERIAL_GOAL_BYTES
            and fixture["goal_sha256"] == SERIAL_GOAL_SHA256
        ),
    }


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_study(args: argparse.Namespace) -> dict[str, object]:
    live_path = _run_live_driver(args)
    modes = _run_modes(args, live_path)

    # The raw arm latencies and the decomposition come first, before any exact
    # identity, physical ceiling or other fatal guard is interpreted.
    decomposition = _decomposition(modes)
    scored = _evaluate_scored(modes, decomposition)
    inventory = _producer_inventory(live_path)
    fixture = _serial_fixture()
    fatal = _fatal_guards(modes, scored, inventory, fixture)
    summary = {
        key: {
            field: value
            for field, value in result.items()
            if field != "raw_rows"
        }
        for key, result in modes.items()
    }
    return {
        "behavioral_evidence": scored,
        "capture": {
            "rows": _line_count(args.capture),
            "sha256": _sha256(args.capture),
        },
        "decomposition": decomposition,
        "fatal_unscored_guards": fatal,
        "mode_results": summary,
        "producer_inventory": inventory,
        "provenance": {
            "expectation_boundary_commit": EXPECTATION_BOUNDARY_COMMIT,
            "repository_commit_observed": _git_head(),
            "vllm_authored_against_commit": VLLM_AUTHORED_AGAINST_COMMIT,
            "vllm_observed_file_sha256": SOURCE_HASHES,
            "vllm_observed_version": VLLM_VERSION,
        },
        "schema": "simllm-vllm-observed-overlap-study-v1",
        "serial_fixture": fixture,
    }


def main() -> None:
    args = _parse_args()
    check_expectation_registry(args)
    if args.check_only:
        print("check-only: frozen expectation registry passed; no artifacts written")
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = _run_study(args)
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    behavior = report["behavioral_evidence"]
    print(
        "genuine-risk instances: "
        f"{behavior['genuine_risk_passed']}/{behavior['genuine_risk_executed']}"
    )
    failed = sorted(
        name for name, value in report["fatal_unscored_guards"].items() if not value
    )
    print("fatal guards violated: " + (", ".join(failed) if failed else "none"))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
