"""Run the frozen framework trace v2 join study.

Every input, cost-model constant, scheduling rule, sweep cell, bound and
relation this script evaluates is frozen in ``expectations.md``, which was
committed before any of the code below existed and before the first cell ran.

The chain is: three observed SGLang captures of Granite 3.0 1B A400M, the new
version 2 join, the resulting ``simllm-preplay-replay-run-v1`` record fed to
the accepted vLLM replay seam, an analytical roofline step cost over the real
model geometry, and per-request TTFT and TPOT out of the live step stream. The
declared arm runs the accepted absent-replay path with a workload-declared
output length; the joined arm replaces that length with the capture's own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from simllm.adapters.vllm import (
    SimExecutorConfig,
    SimWorker,
    configure,
    reset_configuration,
)
from simllm.compute import (
    GPU_ENVELOPES,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
    step_kernel,
)
from simllm.core import (
    RequestBookkeeper,
    StepRecord,
    StepResult,
    VirtualClock,
)
from simllm.preplay import (
    FrameworkPreplayTrace,
    RequestArrival,
    framework_kv_reconciliation_to_json,
    join_framework_arrivals,
    preplay_replay_run_to_json,
    project_preplay_routing,
    read_framework_preplay_trace,
    write_framework_kv_reconciliation,
    write_framework_preplay_trace,
    write_preplay_replay_run,
)

FROZEN_TRACE_SHA256 = {
    "short": "7d979e20516bec19062d55bd4ce5ba6512c71a729c20f45e5f827fb28b6be298",
    "long": "1c310fae51e98f309e805cd088c5c6869f8a32d9966eeeb601d8cd370184536e",
    "preempt": "da0096b696564f365003d11565f4cbc5bed33ab47f7236c52b3ca7c989fd4982",
}

#: frozen roofline constants, derived by hand in expectations.md
WEIGHT_BYTES = 2_566_914_048
LM_HEAD_BYTES = 100_663_296
KV_BYTES_PER_CONTEXT_TOKEN = 49_152
ROOFLINE_EFFICIENCY = 0.7
PS_PER_SECOND = 1_000_000_000_000
ENVELOPES = ("b100", "h200")

#: frozen workload cells: trace key, request IDs, prompt, oracle and declared
#: output length, and scheduler capacity
CELLS = {
    "short": ("short", ("s0",), 8, 4, 8, 1),
    "long": ("long", ("l0",), 96, 8, 16, 1),
    "preempt4": ("preempt", tuple(f"p{i}" for i in range(8)), 8, 20, 24, 4),
    "preempt8": ("preempt", tuple(f"p{i}" for i in range(8)), 8, 20, 24, 8),
}

#: frozen literal step counts and admission steps, from expectations.md
FROZEN_SCHEDULE = {
    ("short", "joined"): ((0,), 4),
    ("short", "declared"): ((0,), 8),
    ("long", "joined"): ((0,), 8),
    ("long", "declared"): ((0,), 16),
    ("preempt8", "joined"): ((0, 1, 2, 3, 4, 5, 6, 7), 27),
    ("preempt8", "declared"): ((0, 1, 2, 3, 4, 5, 6, 7), 31),
    ("preempt4", "joined"): ((0, 1, 2, 3, 20, 21, 22, 23), 43),
    ("preempt4", "declared"): ((0, 1, 2, 3, 24, 25, 26, 27), 51),
}

#: frozen TTFT ratio bands for the requests the join admits earlier
FROZEN_TTFT_BANDS = {
    "p4": (0.836, 0.844),
    "p5": (0.841, 0.849),
    "p6": (0.847, 0.855),
    "p7": (0.853, 0.861),
}

TPOT_TOLERANCE = 0.01
BANDWIDTH_TOLERANCE = 1e-6


@dataclass
class FakeNewRequest:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: object | None = None
    num_computed_tokens: int = 0


@dataclass
class FakeCachedRequests:
    req_ids: list[str] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: list[FakeNewRequest] = field(default_factory=list)
    scheduled_cached_reqs: FakeCachedRequests = field(default_factory=FakeCachedRequests)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    finished_req_ids: set = field(default_factory=set)
    preempted_req_ids: set | None = None
    has_structured_output_requests: bool = False

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())


class FakeDType:
    itemsize = 2


class FakeIrOpPriority:
    @staticmethod
    def set_default():
        return


class FakeModelConfig:
    runner_type = "generate"
    dtype = FakeDType()
    hf_text_config = SimpleNamespace(intermediate_size=512)
    max_model_len = 4096

    @staticmethod
    def get_hidden_size():
        return 1024

    @staticmethod
    def get_num_layers(parallel_config):
        return 24

    @staticmethod
    def get_num_attention_heads(parallel_config):
        return 16

    @staticmethod
    def get_num_kv_heads(parallel_config):
        return 8

    @staticmethod
    def get_head_size():
        return 64

    @staticmethod
    def get_vocab_size():
        return 49_152

    @staticmethod
    def get_total_num_hidden_layers():
        return 24


def fake_vllm_config():
    return SimpleNamespace(
        model_config=FakeModelConfig(),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto", num_gpu_blocks=None),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            world_size=1,
            rank=0,
        ),
        scheduler_config=SimpleNamespace(),
        device_config=SimpleNamespace(device="cuda"),
        speculative_config=None,
        lora_config=None,
        load_config=None,
        observability_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
        kernel_config=SimpleNamespace(ir_op_priority=FakeIrOpPriority()),
        profiler_config=SimpleNamespace(profiler=None),
        quant_config=None,
        use_v2_model_runner=False,
    )


def granite_dims() -> ModelDims:
    """The captured geometry, one rank, every expert resident."""

    return ModelDims(
        num_layers=24,
        hidden_size=1024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_152,
        dtype_bytes=2,
        num_experts=32,
        top_k=8,
        moe_intermediate_size=512,
        local_num_experts=32,
    )


def sampling_params(max_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(max_tokens=max_tokens, min_tokens=0, stop_token_ids=[])


@contextmanager
def skeleton_environment():
    names = {"SIMLLM_VLLM_WORKER_MODE": "skeleton", "SIMLLM_VLLM_MODE": "virtual"}
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(names)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class RooflineStepSink:
    """The frozen analytical step cost, recording each step's roofline bound."""

    def __init__(self, dims: ModelDims, envelope: str) -> None:
        self.dims = dims
        self.gpu = GPU_ENVELOPES[envelope]
        self.provider = RooflineProvider(ROOFLINE_EFFICIENCY)
        self.host_model = HostInitiationModel.ideal()
        self.bounds: list[str] = []
        self.context_tokens: list[int] = []

    def __call__(self, record: StepRecord) -> StepResult:
        kernel = step_kernel(self.dims, record, record.num_sampled)
        estimate = self.provider.estimate(kernel, self.gpu)
        represented = self.host_model.represented_estimate(estimate, self.gpu)
        self.bounds.append(estimate.bound)
        self.context_tokens.append(
            sum(request.context_length for request in record.scheduled)
        )
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=represented.duration_ps,
            completed_at_ps=record.virtual_time_ps + represented.duration_ps,
        )


def build_schedule(
    request_ids: tuple[str, ...],
    prompt_length: int,
    serving_length: int,
    capacity: int,
) -> tuple[list[FakeSchedulerOutput], dict[str, int]]:
    """Generate the frozen policy's schedule for one serving length."""

    steps: list[FakeSchedulerOutput] = []
    admissions: dict[str, int] = {}
    waiting = list(request_ids)
    produced: dict[str, int] = {}
    pending_finished: list[str] = []
    prompt = list(range(prompt_length))
    while waiting or produced or pending_finished:
        step_index = len(steps)
        new_requests: list[FakeNewRequest] = []
        cached = FakeCachedRequests()
        scheduled: dict[str, int] = {}
        finished = set(pending_finished)
        pending_finished = []
        admitted: str | None = None
        if waiting and len(produced) < capacity:
            admitted = waiting.pop(0)
            produced[admitted] = 0
            admissions[admitted] = step_index
            new_requests.append(
                FakeNewRequest(admitted, list(prompt), sampling_params(serving_length))
            )
            scheduled[admitted] = prompt_length
        for request_id, tokens in produced.items():
            if request_id == admitted:
                continue
            cached.req_ids.append(request_id)
            cached.num_computed_tokens.append(prompt_length + tokens - 1)
            cached.num_output_tokens.append(tokens)
            scheduled[request_id] = 1
        steps.append(
            FakeSchedulerOutput(
                scheduled_new_reqs=new_requests,
                scheduled_cached_reqs=cached,
                num_scheduled_tokens=scheduled,
                finished_req_ids=finished,
            )
        )
        for request_id in scheduled:
            produced[request_id] += 1
            if produced[request_id] == serving_length:
                del produced[request_id]
                pending_finished.append(request_id)
    return steps, admissions


def predicted_admissions(
    request_count: int, capacity: int, serving_length: int
) -> tuple[int, ...]:
    """Closed-form admission step of every request under the frozen policy."""

    admissions: list[int] = []
    for index in range(request_count):
        if index < capacity:
            admissions.append(index)
        else:
            admissions.append(admissions[index - capacity] + serving_length)
    return tuple(admissions)


def predicted_step_latencies(
    admissions: tuple[int, ...],
    prompt_length: int,
    serving_length: int,
    envelope: str,
) -> tuple[int, ...]:
    """Closed-form latency of every scheduled step, from the frozen formula.

    Independent of the schedule generator and of ``step_kernel``: the running
    set, the context sum and the memory-roof duration are all recomputed here
    with plain arithmetic over the constants frozen in expectations.md.
    """

    bandwidth = GPU_ENVELOPES[envelope].mem_bandwidth
    last_step = max(start + serving_length - 1 for start in admissions)
    latencies: list[int] = []
    for step in range(last_step + 1):
        context = 0
        for start in admissions:
            if start <= step <= start + serving_length - 1:
                context += prompt_length if start == step else prompt_length + step - start
        total_bytes = WEIGHT_BYTES + LM_HEAD_BYTES + KV_BYTES_PER_CONTEXT_TOKEN * context
        seconds = total_bytes / (bandwidth * ROOFLINE_EFFICIENCY)
        latencies.append(int(seconds * PS_PER_SECOND))
    return tuple(latencies)


def drive_cell(
    run_dir: Path,
    cell: str,
    envelope: str,
    treatment: str,
    replay_run_path: Path | None,
    request_ids: tuple[str, ...],
    prompt_length: int,
    serving_length: int,
    capacity: int,
) -> dict:
    """Run one sweep cell through the live adapter and return its evidence."""

    label = f"{cell}_{envelope}_{treatment}"
    schedule, admissions = build_schedule(
        request_ids, prompt_length, serving_length, capacity
    )
    sink = RooflineStepSink(granite_dims(), envelope)
    token_times: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
    served: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
    reset_configuration()
    try:
        configure(
            step_sink=sink,
            config=SimExecutorConfig(
                mode="virtual",
                token_id=512,
                step_records_path=str(run_dir / f"{label}_steps.jsonl"),
                replay_run_path=None if replay_run_path is None else str(replay_run_path),
            ),
        )
        with skeleton_environment():
            worker = SimWorker(
                fake_vllm_config(),
                local_rank=0,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:1",
                is_driver_worker=True,
                _simllm_clock=VirtualClock(),
            )
            worker.init_device()
            for scheduler_output in schedule:
                output = worker.execute_model(scheduler_output)
                if scheduler_output.num_scheduled_tokens:
                    if output is not None:
                        raise AssertionError("a nonempty skeleton step must split sampling")
                    output = worker.sample_tokens(None)
                if output is None:
                    continue
                for request_id, token_ids in zip(
                    output.req_ids, output.sampled_token_ids or (), strict=True
                ):
                    if not token_ids:
                        continue
                    served[request_id].extend(token_ids)
                    token_times[request_id].append(worker.clock.now_ps)
            step_latencies = tuple(result.step_latency_ps for result in worker.step_results)
            final_clock_ps = worker.clock.now_ps
    finally:
        reset_configuration()

    ttft = {request_id: times[0] for request_id, times in token_times.items()}
    tpot = {
        request_id: Fraction(
            sum(later - earlier for earlier, later in pairwise(times)), len(times) - 1
        )
        for request_id, times in token_times.items()
    }
    return {
        "cell": cell,
        "envelope": envelope,
        "treatment": treatment,
        "serving_length": serving_length,
        "capacity": capacity,
        "admissions": admissions,
        "scheduled_step_count": sum(
            1 for step in schedule if step.num_scheduled_tokens
        ),
        "record_count": len(step_latencies),
        "step_latencies_ps": step_latencies,
        "context_tokens": tuple(sink.context_tokens),
        "bounds": tuple(sink.bounds),
        "served": {request_id: tuple(tokens) for request_id, tokens in served.items()},
        "token_times_ps": {
            request_id: tuple(times) for request_id, times in token_times.items()
        },
        "ttft_ps": ttft,
        "tpot_ps": tpot,
        "final_clock_ps": final_clock_ps,
    }


def evaluate(run_dir: Path, traces: dict[str, Path]) -> dict:
    """Execute every frozen relation and return the whole result record."""

    fatal: list[str] = []
    exact: list[tuple[str, bool, str]] = []
    scored: list[tuple[str, bool, str]] = []
    joins: dict[str, dict] = {}

    for key, path in traces.items():
        trace = read_framework_preplay_trace(path)
        bookkeeper = RequestBookkeeper()
        request_ids = tuple(request.request_id for request in trace.requests)
        joined = join_framework_arrivals(
            tuple(
                RequestArrival(request_id=request_id, arrived_at_ps=0)
                for request_id in request_ids
            ),
            path,
            bookkeeper,
        )
        run_path = write_preplay_replay_run(joined.run, run_dir / f"{key}_replay-run.json")
        write_framework_kv_reconciliation(joined.kv, run_dir / f"{key}_kv.json")

        binding_ok = True
        for request in trace.requests:
            pinned_request = joined.run.by_request_id(request.request_id)
            if (
                pinned_request.output_token_ids != request.output_token_ids
                or pinned_request.output_length != request.output_length
                or pinned_request.stop_reason is not request.stop_reason
            ):
                binding_ok = False
                fatal.append(f"G3 {key}/{request.request_id}: join does not match the capture")
        exact.append(
            (
                f"E2 join binding ({key})",
                binding_ok,
                f"{len(trace.requests)} requests match on length, stop reason and tokens",
            )
        )

        routed = project_preplay_routing(joined.run)
        rows = 0
        routing_ok = True
        for request in trace.requests:
            projected = routed.by_request_id(request.request_id)
            source = (*request.prefill_dispatch, *request.decode_dispatch)
            if len(projected.tokens) != len(source):
                routing_ok = False
                break
            for token, dispatch in zip(projected.tokens, source, strict=True):
                rows += len(token.layers)
                if (
                    token.phase is not dispatch.phase
                    or token.token_index != dispatch.token_index
                    or token.token_id != dispatch.token_id
                    or tuple(layer.expert_ids for layer in token.layers)
                    != tuple(layer.expert_ids for layer in dispatch.routing)
                ):
                    routing_ok = False
                    break
        exact.append(
            (
                f"E3 routing projection ({key})",
                routing_ok,
                f"{rows} token-layer rows copied in capture order",
            )
        )
        if not routing_ok:
            fatal.append(f"E3 {key}: routing projection lost the capture order")

        payload = preplay_replay_run_to_json(joined.run)
        reconciliation = framework_kv_reconciliation_to_json(joined.kv)
        joins[key] = {
            "run_path": run_path.name,
            "trace_sha256": joined.run.trace.sha256,
            "request_count": len(joined.run.requests),
            "run_json_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "kv": reconciliation,
        }

        capacity = trace.provenance.kv_token_capacity
        defects = [defect.code.value for defect in joined.kv.defects]
        per_request = {
            request.request_id: joined.kv.by_request_id(request.request_id)
            for request in trace.requests
        }
        if key == "preempt":
            clean = [
                r
                for r in per_request.values()
                if r.preemption_event_count == 0 and r.prefix_hit_token_count == 0
            ]
            scored.append(
                (
                    f"KV1 ({key})",
                    all(
                        r.allocated_token_count == r.forwarded_token_count for r in clean
                    )
                    and bool(clean),
                    f"{len(clean)} unpreempted requests, allocation totals "
                    + ",".join(str(r.allocated_token_count) for r in clean),
                )
            )
            preempted = [r for r in per_request.values() if r.preemption_event_count > 0]
            scored.append(
                (
                    f"KV2 ({key})",
                    all(
                        r.allocated_token_count >= r.forwarded_token_count
                        for r in preempted
                    ),
                    f"{len(preempted)} preempted requests, agreements "
                    + ",".join(r.agreement.value for r in preempted),
                )
            )
        else:
            exact.append(
                (
                    f"KV1 post-specified ({key})",
                    all(
                        r.allocated_token_count == r.forwarded_token_count
                        for r in per_request.values()
                    ),
                    "allocation totals "
                    + ",".join(str(r.allocated_token_count) for r in per_request.values()),
                )
            )
        scored.append(
            (
                f"KV3 ({key})",
                all(
                    r.preemption_event_count == r.declared_preemption_count
                    for r in per_request.values()
                ),
                "preemption events "
                + ",".join(str(r.preemption_event_count) for r in per_request.values()),
            )
        )
        scored.append(
            (
                f"KV4 ({key})",
                all(
                    r.prefix_hit_token_count == r.declared_cached_tokens
                    for r in per_request.values()
                ),
                "prefix-hit tokens "
                + ",".join(str(r.prefix_hit_token_count) for r in per_request.values()),
            )
        )
        within = joined.kv.peak_live_token_count <= capacity
        expected_within = key != "preempt"
        scored.append(
            (
                f"KV5 ({key})",
                within == expected_within,
                f"peak {joined.kv.peak_live_token_count} tokens against capacity "
                f"{capacity}, predicted "
                + ("within" if expected_within else "over"),
            )
        )
        if defects:
            fatal.append(f"KV defects ({key}): {','.join(sorted(set(defects)))}")

        stripped_path = run_dir / f"{key}_no-kv.jsonl"
        write_framework_preplay_trace(
            stripped_path,
            FrameworkPreplayTrace(
                provenance=trace.provenance, requests=trace.requests, kv_events=()
            ),
        )
        stripped = join_framework_arrivals(
            tuple(
                RequestArrival(request_id=request_id, arrived_at_ps=0)
                for request_id in request_ids
            ),
            stripped_path,
            RequestBookkeeper(),
        )

        def pinned(run):
            return [
                (
                    request.request_id,
                    request.arrived_at_ps,
                    request.output_length,
                    request.stop_reason.value,
                    request.output_token_ids,
                    request.routing_reference.request_id,
                    request.bookkeeping_object_id,
                )
                for request in run.requests
            ]

        if pinned(joined.run) != pinned(stripped.run):
            fatal.append(f"G7 {key}: the KV stream changed a joined request")

    results: dict[tuple[str, str, str], dict] = {}
    for cell, (trace_key, request_ids, prompt, oracle, declared, capacity) in CELLS.items():
        for envelope in ENVELOPES:
            for treatment, serving in (("declared", declared), ("joined", oracle)):
                replay_path = None
                if treatment == "joined":
                    replay_path = run_dir / f"{trace_key}_replay-run.json"
                cell_dir = run_dir / "cells"
                cell_dir.mkdir(parents=True, exist_ok=True)
                results[(cell, envelope, treatment)] = drive_cell(
                    cell_dir,
                    cell,
                    envelope,
                    treatment,
                    replay_path,
                    request_ids,
                    prompt,
                    serving,
                    capacity,
                )

    for (cell, envelope, treatment), row in results.items():
        if any(bound != "memory" for bound in row["bounds"]):
            fatal.append(f"G1 {cell}/{envelope}/{treatment}: a step was compute bound")
        frozen_admissions, frozen_steps = FROZEN_SCHEDULE[(cell, treatment)]
        request_ids = CELLS[cell][1]
        observed = tuple(row["admissions"][request_id] for request_id in request_ids)
        if observed != frozen_admissions or row["scheduled_step_count"] != frozen_steps:
            fatal.append(
                f"G-schedule {cell}/{treatment}: admissions {observed} and "
                f"{row['scheduled_step_count']} steps against the frozen "
                f"{frozen_admissions} and {frozen_steps}"
            )
        prompt, oracle = CELLS[cell][2], CELLS[cell][3]
        if treatment == "joined":
            trace_key = CELLS[cell][0]
            trace = read_framework_preplay_trace(traces[trace_key])
            for request in trace.requests:
                if request.request_id not in row["served"]:
                    continue
                if row["served"][request.request_id] != request.output_token_ids:
                    fatal.append(
                        f"G3 {cell}/{envelope}: {request.request_id} served the wrong tokens"
                    )
        predicted = predicted_step_latencies(
            predicted_admissions(len(request_ids), CELLS[cell][5], row["serving_length"]),
            prompt,
            row["serving_length"],
            envelope,
        )
        measured = row["step_latencies_ps"][: len(predicted)]
        exact.append(
            (
                f"E4 closed form ({cell}/{envelope}/{treatment})",
                tuple(measured) == predicted,
                f"{len(predicted)} step latencies against the frozen formula",
            )
        )
        row["predicted_step_latencies_ps"] = predicted

    for envelope in ENVELOPES:
        for cell in CELLS:
            declared_row = results[(cell, envelope, "declared")]
            joined_row = results[(cell, envelope, "joined")]
            for request_id in CELLS[cell][1]:
                declared_ttft = declared_row["ttft_ps"][request_id]
                joined_ttft = joined_row["ttft_ps"][request_id]
                if declared_row["admissions"][request_id] == joined_row["admissions"][request_id]:
                    if declared_ttft != joined_ttft:
                        fatal.append(
                            f"G4 {cell}/{envelope}/{request_id}: TTFT moved from "
                            f"{declared_ttft} to {joined_ttft} with the same admission step"
                        )
                elif request_id in FROZEN_TTFT_BANDS:
                    low, high = FROZEN_TTFT_BANDS[request_id]
                    ratio = joined_ttft / declared_ttft
                    scored.append(
                        (
                            f"B1 ({cell}/{envelope}/{request_id})",
                            low <= ratio <= high,
                            f"TTFT ratio {ratio:.6f} against [{low}, {high}]",
                        )
                    )
                declared_tpot = declared_row["tpot_ps"][request_id]
                joined_tpot = joined_row["tpot_ps"][request_id]
                ratio = float(joined_tpot / declared_tpot)
                scored.append(
                    (
                        f"B2 ({cell}/{envelope}/{request_id})",
                        joined_tpot <= declared_tpot and 1.0 - ratio < TPOT_TOLERANCE,
                        f"TPOT ratio {ratio:.6f}",
                    )
                )

    for cell in CELLS:
        for treatment in ("declared", "joined"):
            slow = results[(cell, "h200", treatment)]
            fast = results[(cell, "b100", treatment)]
            ratios = []
            for request_id in CELLS[cell][1]:
                ratios.append(slow["ttft_ps"][request_id] / fast["ttft_ps"][request_id])
                ratios.append(
                    float(slow["tpot_ps"][request_id] / fast["tpot_ps"][request_id])
                )
            worst = max(abs(ratio - 5 / 3) for ratio in ratios)
            scored.append(
                (
                    f"B3 ({cell}/{treatment})",
                    worst < BANDWIDTH_TOLERANCE,
                    f"worst deviation {worst:.3e} from 5/3 over {len(ratios)} values",
                )
            )

    return {
        "joins": joins,
        "cells": {
            f"{cell}/{envelope}/{treatment}": {
                "serving_length": row["serving_length"],
                "capacity": row["capacity"],
                "scheduled_step_count": row["scheduled_step_count"],
                "admissions": row["admissions"],
                "first_step_latency_ps": row["step_latencies_ps"][0],
                "final_clock_ps": row["final_clock_ps"],
                "ttft_ps": row["ttft_ps"],
                "tpot_ps": {
                    request_id: str(value) for request_id, value in row["tpot_ps"].items()
                },
                "bounds": sorted(set(row["bounds"])),
            }
            for (cell, envelope, treatment), row in results.items()
        },
        "fatal": fatal,
        "exact": [
            {"name": name, "passed": passed, "detail": detail} for name, passed, detail in exact
        ],
        "scored": [
            {"name": name, "passed": passed, "detail": detail} for name, passed, detail in scored
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-trace", type=Path, required=True)
    parser.add_argument("--long-trace", type=Path, required=True)
    parser.add_argument("--preempt-trace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)

    traces = {
        "short": args.short_trace,
        "long": args.long_trace,
        "preempt": args.preempt_trace,
    }
    for key, path in traces.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != FROZEN_TRACE_SHA256[key]:
            raise SystemExit(f"{key} trace SHA-256 {digest} is not the frozen value")
        print(f"{key}: {digest}")
    if args.check_only:
        print("check-only: resolved every frozen input, no cells executed")
        return 0
    if args.run_dir is None:
        raise SystemExit("--run-dir is required outside --check-only")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)

    record = evaluate(run_dir, traces)
    (run_dir / "results.json").write_text(
        json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    for name, passed, detail in (
        (row["name"], row["passed"], row["detail"]) for row in record["exact"]
    ):
        print(f"exact  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    for name, passed, detail in (
        (row["name"], row["passed"], row["detail"]) for row in record["scored"]
    ):
        print(f"scored {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    for message in record["fatal"]:
        print(f"FATAL  {message}")
    scored_pass = sum(1 for row in record["scored"] if row["passed"])
    exact_pass = sum(1 for row in record["exact"] if row["passed"])
    print(
        f"exact {exact_pass}/{len(record['exact'])}, "
        f"scored {scored_pass}/{len(record['scored'])}, "
        f"fatal violations {len(record['fatal'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
