"""Run the frozen composed realistic-deployment SGLang study (SGL-27).

The chain is the live one: a real SGLang ``Scheduler``, ``RadixCache`` and
token pools driven step by step in the process that installed the sink,
arrival-gated admission on the worker's own virtual clock, SGLang's captured
post-selection expert ids driving the traffic, and per-request TTFT and TPOT
coming back out through the medium-aware reducer.

What this study adds over ``examples/sglang_end_to_end_v1`` is composition. The
same chain is priced as two declared deployments, one whose every segment stays
on NVLink inside a node and one whose every segment crosses the fabric, under
three named arms of a per-collective fixed-cost envelope and two host-cost
arms, so the reported numbers arrive as brackets with their transfer stated
rather than as one silently chosen constant.

Every band, relation, guard and entailment answer this script evaluates is
frozen in ``expectations.md``, committed before any of this code existed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: the expectations-only commit that froze every literal below
FREEZE_COMMIT = "dd026c0"

# --------------------------------------------------------------------------
# Frozen literals. Quoted from expectations.md; never edited to fit a result.
# --------------------------------------------------------------------------

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / MODEL_REVISION
)
SGLANG_PINNED_COMMIT = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"

NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 512
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_SIZE = 64
VOCAB_SIZE = 49155
DTYPE_BYTES = 2
NUM_EXPERTS = 32
TOP_K = 8
MOE_INTERMEDIATE_SIZE = 512
VECTOR_BYTES = HIDDEN_SIZE * DTYPE_BYTES

ENGINE_RANK = 0
EP_WORLD = 8
ARRIVAL_SPACING_PS = 1_000_000_000
PROFILE = "rnic-nn-fluid"
ROOFLINE_EFFICIENCY = 0.7

SGLANG_DEVICE = "cpu"
SGLANG_DTYPE = "float32"
SGLANG_SEED = 173
SGLANG_PAGE_SIZE = 1
SGLANG_CONTEXT_LENGTH = 512
SGLANG_MAX_TOTAL_TOKENS = 4096
SGLANG_MAX_RUNNING_REQUESTS = 8
#: -1 resolves to "chunked prefill disabled" at the pinned commit
SGLANG_CHUNKED_PREFILL_SIZE = -1
TORCH_NUM_THREADS = 8

REQUEST_IDS = ("p0", "p1", "p2", "p3")
PROMPT_TOKENS = 8
MAX_NEW_TOKENS = 12

INTRA_ENVELOPE = "intra-node-fixed-cost-v1"
CROSS_ENVELOPE = "cross-node-fixed-cost-provisional-v1"
ARMS = ("off", "lower", "upper")
HOST_ARMS = {"ideal": "ideal", "turing": "turing-cuda-graph"}
TURING_LAUNCH_COUNT = 440

LINK_400G = 400_000_000_000
LINK_100G = 100_000_000_000

#: cell name -> (topology, link bits per second, collective arm, host arm)
CELLS: dict[str, tuple[str, int, str, str]] = {}
for _host in ("ideal", "turing"):
    for _arm in ARMS:
        CELLS[f"intra-{_arm}-{_host}"] = ("intra", LINK_400G, _arm, _host)
        CELLS[f"cross400-{_arm}-{_host}"] = ("cross", LINK_400G, _arm, _host)
        CELLS[f"cross100-{_arm}-{_host}"] = ("cross", LINK_100G, _arm, _host)

#: the executed BACK-44 negative control, not a priced cell
BACK44_STAGE = "back44"
BACK44_MESSAGE = "graph cannot be represented by ordered GOAL artifacts"

# Exact constants of the closed form, frozen in expectations.md.
B200_LOCAL_BASE_PS = 30_128_029
CROSS_PROVISIONAL_BASE_PS = 49_487_789
TURING_LAUNCH_FLOOR_PS = 356_094_640
TURING_COMPUTE_SERVICE_PS = 356_095_000
INTRA_TP_ALLREDUCES = 24
INTRA_RING_PHASES = 14
MOE_ALLTOALLS = 48
INTRA_ARTIFACTS = NUM_LAYERS + INTRA_TP_ALLREDUCES * INTRA_RING_PHASES + MOE_ALLTOALLS
CROSS_ARTIFACTS = NUM_LAYERS + MOE_ALLTOALLS
CROSS_ROUTED_BYTES = 35_696_640
INTRA_TP_BYTES = 52_297_728
INTRA_ROUTED_BYTES = INTRA_TP_BYTES + CROSS_ROUTED_BYTES
MAX_STEP_TOKENS = len(REQUEST_IDS) * PROMPT_TOKENS
MAX_CRITICAL_ENDPOINT_BYTES = MAX_STEP_TOKENS * (EP_WORLD - 1) * VECTOR_BYTES
PROFILE_ENDPOINT_MAX_BYTES = 2 * (EP_WORLD - 1) * (262_144 // EP_WORLD)

#: frozen per-cell step-latency bands, picoseconds, from expectations.md
STEP_BANDS_PS: dict[str, tuple[int, int]] = {
    "intra-off-ideal": (53_274_000, 130_627_000),
    "intra-lower-ideal": (55_482_000, 429_091_000),
    "intra-upper-ideal": (2_224_700_000, 2_598_309_000),
    "intra-off-turing": (356_671_000, 411_439_000),
    "intra-lower-turing": (358_879_000, 709_903_000),
    "intra-upper-turing": (2_528_097_000, 2_879_121_000),
    "cross400-off-ideal": (167_222_000, 635_339_000),
    "cross400-lower-ideal": (1_613_367_000, 2_081_484_000),
    "cross400-upper-ideal": (2_542_636_000, 3_010_753_000),
    "cross400-off-turing": (454_061_000, 892_497_000),
    "cross400-lower-turing": (1_900_206_000, 2_338_642_000),
    "cross400-upper-turing": (2_829_475_000, 3_267_911_000),
    "cross100-off-ideal": (173_120_000, 1_956_545_000),
    "cross100-lower-ideal": (1_619_266_000, 3_402_690_000),
    "cross100-upper-ideal": (2_548_534_000, 4_331_959_000),
    "cross100-off-turing": (459_959_000, 2_213_703_000),
    "cross100-lower-turing": (1_906_105_000, 3_659_848_000),
    "cross100-upper-turing": (2_835_373_000, 4_589_117_000),
}
#: a request waits behind at most three other prompts plus its own
TTFT_CEILING_STEPS = 5
#: after its first token a request gets one token per decode step, and at most
#: three other prefills can interleave into its eleven remaining intervals
TPOT_CEILING_STEPS = 1.5

#: the accepted sglang_end_to_end_v1 ep8-400g artifact, quoted from that
#: study's RESULTS.md, to the published precision of 0.01 microseconds
ACCEPTED_E2E_STEPS = 26
ACCEPTED_E2E_TTFT_US = {"p0": 270.05, "p1": 358.38, "p2": 483.60, "p3": 421.38}
ACCEPTED_E2E_TPOT_US = {"p0": 262.04, "p1": 268.70, "p2": 244.41, "p3": 214.95}
E2_TOLERANCE_US = 0.005

#: scored denominators, split by evidence class and never summed
EXPECTED_EXACT_RELATIONS = 2
EXPECTED_BEHAVIORAL_RELATIONS = 6
EXPECTED_FATAL_GUARDS = 11

PS_PER_SECOND = 1_000_000_000_000
PS_PER_US = 1_000_000

GPU_COMPUTE_CODE = "g"
NVLINK_CODE = "n"


def prompt_token_ids(index: int) -> tuple[int, ...]:
    """The capture's own pressure-prompt rule, reproduced exactly."""

    return tuple(1000 + 100 * index + step for step in range(PROMPT_TOKENS))


def cell_names() -> tuple[str, ...]:
    return tuple(sorted(CELLS))


def intra_cells() -> tuple[str, ...]:
    return tuple(name for name in cell_names() if CELLS[name][0] == "intra")


def cross_cells() -> tuple[str, ...]:
    return tuple(name for name in cell_names() if CELLS[name][0] == "cross")


def arm_base_latency_ps(name: str) -> int:
    """The exact per-collective surcharge this cell's arm charges."""

    topology, _, arm, _ = CELLS[name]
    if arm == "off":
        return 0
    if topology == "intra":
        return 0 if arm == "lower" else B200_LOCAL_BASE_PS
    return B200_LOCAL_BASE_PS if arm == "lower" else CROSS_PROVISIONAL_BASE_PS


def collective_count(name: str) -> int:
    topology = CELLS[name][0]
    return INTRA_TP_ALLREDUCES + MOE_ALLTOALLS if topology == "intra" else MOE_ALLTOALLS


# ------------------------------------------------------------ input checks ---


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _python_version(executable: Path) -> tuple[int, ...]:
    result = subprocess.run(
        [
            str(executable),
            "-c",
            "import sys; print('.'.join(map(str, sys.version_info[:3])))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(int(part) for part in result.stdout.strip().split("."))


def _trace_header(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "header":
                raise SystemExit(f"{path}: first row is not a trace header")
            provenance = row.get("provenance")
            if not isinstance(provenance, dict):
                raise SystemExit(f"{path}: header carries no provenance object")
            return provenance
    raise SystemExit(f"{path}: trace is empty")


def check_trace_provenance(path: Path) -> dict[str, object]:
    """Fatal guard G1, also run as part of ``--check-only``."""

    provenance = _trace_header(path)
    expected = {
        "schema": "simllm-preplay-trace-v2",
        "framework": "sglang",
        "routing_source": "observed-dispatch",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "expert_count": NUM_EXPERTS,
        "top_k": TOP_K,
        "observed_source": SGLANG_PINNED_COMMIT,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise SystemExit(
                f"{path}: provenance {field}={provenance.get(field)!r}, "
                f"expected {value!r}"
            )
    if list(provenance.get("moe_layer_indices") or ()) != list(range(NUM_LAYERS)):
        raise SystemExit(f"{path}: MoE layer indices are not 0..{NUM_LAYERS - 1}")
    return provenance


def _check_frozen_registry() -> None:
    """Refuse to run if any frozen literal drifted from expectations.md."""

    if len(CELLS) != 18:
        raise AssertionError("the frozen cell matrix changed")
    if len(intra_cells()) != 6 or len(cross_cells()) != 12:
        raise AssertionError("the frozen topology split changed")
    if set(STEP_BANDS_PS) != set(CELLS):
        raise AssertionError("the frozen step bands do not cover the cell matrix")
    for name, (floor_ps, ceiling_ps) in STEP_BANDS_PS.items():
        if floor_ps >= ceiling_ps:
            raise AssertionError(f"the {name} step band is inverted")
    if INTRA_ARTIFACTS != 408 or CROSS_ARTIFACTS != 72:
        raise AssertionError("the frozen artifact inventory changed")
    if INTRA_TP_ALLREDUCES * B200_LOCAL_BASE_PS != 723_072_696:
        raise AssertionError("the at-capture surcharge split changed")
    if MOE_ALLTOALLS * B200_LOCAL_BASE_PS != 1_446_145_392:
        raise AssertionError("the transferred surcharge split changed")
    if MOE_ALLTOALLS * CROSS_PROVISIONAL_BASE_PS != 2_375_413_872:
        raise AssertionError("the cross-node upper surcharge changed")
    if TURING_LAUNCH_COUNT * 809_306 != TURING_LAUNCH_FLOOR_PS:
        raise AssertionError("the transferred launch floor changed")
    if TURING_COMPUTE_SERVICE_PS != -(-TURING_LAUNCH_FLOOR_PS // 1000) * 1000:
        raise AssertionError("the enclosed host compute service changed")
    if MAX_CRITICAL_ENDPOINT_BYTES != PROFILE_ENDPOINT_MAX_BYTES:
        raise AssertionError(
            "the zero-margin endpoint admissibility statement no longer holds"
        )
    if INTRA_ROUTED_BYTES != 87_994_368 or CROSS_ROUTED_BYTES != 35_696_640:
        raise AssertionError("the predicted directed-byte totals changed")
    if 8_000_000_000_000 // LINK_400G != 20 or 8_000_000_000_000 // LINK_100G != 80:
        raise AssertionError("the picoseconds-per-byte literals changed")
    if EXPECTED_EXACT_RELATIONS != 2 or EXPECTED_BEHAVIORAL_RELATIONS != 6:
        raise AssertionError("scored relation denominators changed")
    if EXPECTED_FATAL_GUARDS != 11:
        raise AssertionError("the fatal guard roster changed")
    if MAX_NEW_TOKENS - 1 > 19:
        raise AssertionError("output length exceeds the captured decode extent")
    if SGLANG_CHUNKED_PREFILL_SIZE > 0:
        raise AssertionError("chunked prefill must stay disabled, see G3")


def check_only(args: argparse.Namespace) -> None:
    """Validate every frozen input without producing an artifact."""

    _check_frozen_registry()
    if not args.run_dir.is_absolute():
        raise SystemExit("run directory must be an explicit absolute path")
    if args.run_dir.resolve() == REPOSITORY_ROOT:
        raise SystemExit("run directory must be outside the repository")
    model = args.cache_dir / MODEL_RELATIVE_PATH
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {model}")
    if not args.sglang_python.is_file() or not os.access(args.sglang_python, os.X_OK):
        raise SystemExit(
            f"SGLang Python is missing or not executable: {args.sglang_python}"
        )
    if _python_version(args.sglang_python) < (3, 10):
        raise SystemExit("SGLang Python is too old")
    observed = _git_head(args.sglang_source)
    if observed != SGLANG_PINNED_COMMIT:
        raise SystemExit(
            f"SGLang source is at {observed}, expected {SGLANG_PINNED_COMMIT}"
        )
    if not args.htsim_rnic.is_file() or not args.htsim_rnic.stat().st_mode & 0o111:
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    txt2bin = os.environ.get("SIMLLM_TXT2BIN")
    if txt2bin is not None:
        converter = Path(txt2bin)
        if not converter.is_file() or not converter.stat().st_mode & 0o111:
            raise SystemExit(f"SIMLLM_TXT2BIN is not an executable: {converter}")
    if not args.routing_trace.is_file():
        raise SystemExit(f"routing trace is missing: {args.routing_trace}")
    check_trace_provenance(args.routing_trace)
    print(
        json.dumps(
            {
                "check_only": True,
                "artifacts_written": 0,
                "freeze_commit": FREEZE_COMMIT,
                "cells": list(cell_names()),
                "exact_total": EXPECTED_EXACT_RELATIONS,
                "behavioral_total": EXPECTED_BEHAVIORAL_RELATIONS,
                "fatal_guards": EXPECTED_FATAL_GUARDS,
                "run_dir": str(args.run_dir),
            },
            sort_keys=True,
        )
    )


# ----------------------------------------------------------- deployment ------


def _dims(tp_world: int) -> Any:
    """The per-rank sharded geometry the deployment declares."""

    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE // tp_world,
        num_heads=NUM_HEADS // tp_world,
        num_kv_heads=max(NUM_KV_HEADS // tp_world, 1),
        head_size=HEAD_SIZE,
        vocab_size=VOCAB_SIZE,
        dtype_bytes=DTYPE_BYTES,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        local_num_experts=NUM_EXPERTS // EP_WORLD,
    )


def _physical_manifest(hostnames: tuple[str, ...]) -> Any:
    from simllm.placement import PlacementManifest, RankPlacement

    counts: dict[str, int] = {}
    ranks = []
    for global_rank, hostname in enumerate(hostnames):
        local_rank = counts.get(hostname, 0)
        counts[hostname] = local_rank + 1
        ranks.append(
            RankPlacement(
                global_rank=global_rank,
                hostname=hostname,
                local_rank=local_rank,
            )
        )
    return PlacementManifest(ranks=ranks)


def _deployment(name: str) -> dict[str, Any]:
    """Resolve one cell name into the deployment it declares."""

    topology, link_bps, arm, host = CELLS[name]
    if topology == "intra":
        return {
            "topology": topology,
            "link_bps": link_bps,
            "arm": arm,
            "host": host,
            "tp_ranks": tuple(range(EP_WORLD)),
            "ep_ranks": tuple(range(EP_WORLD)),
            "tp_world": EP_WORLD,
            "hostnames": ("node-0",) * EP_WORLD,
            "envelope": INTRA_ENVELOPE,
        }
    return {
        "topology": topology,
        "link_bps": link_bps,
        "arm": arm,
        "host": host,
        "tp_ranks": (ENGINE_RANK,),
        "ep_ranks": tuple(range(EP_WORLD)),
        "tp_world": 1,
        "hostnames": tuple(f"node-{rank}" for rank in range(EP_WORLD)),
        "envelope": CROSS_ENVELOPE,
    }


def _declare_arrivals(bookkeeper: Any) -> dict[str, int]:
    """Append one framework-request object per declared arrival."""

    import hashlib

    from simllm.core import (
        BookkeepingScope,
        CreatedObjectKind,
        CreatedObjectRecord,
        CreatedObjectRef,
        ObjectOwner,
        OperationCorrelation,
    )

    arrivals = {
        request_id: index * ARRIVAL_SPACING_PS
        for index, request_id in enumerate(REQUEST_IDS)
    }
    facts = []
    for request_id, arrived_at_ps in arrivals.items():
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        facts.append(
            CreatedObjectRecord(
                ref=CreatedObjectRef(
                    kind=CreatedObjectKind.FRAMEWORK_REQUEST,
                    object_id=f"sglang-composed-deployment-v1:{digest}",
                ),
                owner=ObjectOwner.FRAMEWORK,
                created_at_ps=arrived_at_ps,
                scope=BookkeepingScope(
                    correlation=OperationCorrelation(request_ids=(request_id,)),
                ),
                native_id=request_id,
                metadata=(
                    ("declared_arrival_ps", arrived_at_ps),
                    ("declared_max_new_tokens", MAX_NEW_TOKENS),
                ),
            )
        )
    bookkeeper.extend(facts)
    return arrivals


def _coarse_json(attribution: Any) -> dict[str, int]:
    return {
        "queue_ps": attribution.queue_ps,
        "kv_ps": attribution.kv_ps,
        "kernel_ps": attribution.kernel_ps,
        "dma_ps": attribution.dma_ps,
        "collective_ps": attribution.collective_ps,
        "nic_ps": attribution.nic_ps,
        "control_ps": attribution.control_ps,
        "total_ps": attribution.total_ps,
    }


def _media_json(media: Any) -> dict[str, int]:
    return {
        "queue_ps": media.queue_ps,
        "kernel_ps": media.kernel_ps,
        "nvlink_ps": media.nvlink_ps,
        "fabric_ps": media.fabric_ps,
        "co_critical_ps": media.co_critical_ps,
        "collective_base_ps": media.collective_base_ps,
        "control_ps": media.control_ps,
        "total_ps": media.total_ps,
    }


# ------------------------------------------------------------- one cell ------


def _cell(args: argparse.Namespace, name: str) -> None:
    """Drive one cell in a child interpreter that owns SGLang and torch."""

    import time

    import torch

    from simllm.adapters.sglang import install
    from simllm.adapters.sglang.host import select_sglang_host_model
    from simllm.adapters.sglang.pump import (
        SglangSchedulerPump,
        build_in_process_scheduler,
        tokenized_generate_request,
    )
    from simllm.adapters.sglang.worker import SimWorkerConfig, configure, latest_worker
    from simllm.backends import (
        HtsimRequestMetricReducer,
        HtsimStepSink,
        HtsimStepSinkConfig,
        attribute_step_detail,
    )
    from simllm.backends.step_sink import NVLINK_MEDIUM
    from simllm.core import RequestBookkeeper, framework_request_arrivals
    from simllm.preplay import project_framework_routing
    from simllm.traffic import (
        ExpertPlacementSnapshot,
        RoutedMoeSupply,
        layer_tp_allreduce_sites,
        step_moe_alltoalls,
        step_tp_allreduces,
    )
    from simllm.workload import AdmissionMode, RequestAdmissionGate

    torch.set_num_threads(TORCH_NUM_THREADS)
    plan = _deployment(name)
    cell_dir = args.run_dir / "cells" / name
    cell_dir.mkdir(parents=True, exist_ok=False)

    routed = project_framework_routing(args.routing_trace)
    bookkeeper = RequestBookkeeper()
    declared = _declare_arrivals(bookkeeper)
    arrivals = {
        arrival.request_id: arrival.arrived_at_ps
        for arrival in framework_request_arrivals(bookkeeper.snapshot())
    }
    if arrivals != declared:
        raise AssertionError("bookkeeping arrivals disagree with the declared mapping")

    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % EP_WORLD)
            for layer in range(NUM_LAYERS)
            for expert in range(NUM_EXPERTS)
        ),
    )
    supply = RoutedMoeSupply(
        engine_rank=ENGINE_RANK,
        routed_experts=routed,
        placements=(placement,),
        step_placement_epochs=tuple((step, 0) for step in range(4096)),
    )
    dims = _dims(plan["tp_world"])
    selection = select_sglang_host_model(
        HOST_ARMS[plan["host"]],
        launch_count=None if plan["host"] == "ideal" else TURING_LAUNCH_COUNT,
        efficiency=ROOFLINE_EFFICIENCY,
    )
    sink = HtsimStepSink(
        HtsimStepSinkConfig(
            profile=PROFILE,
            tp_ranks=plan["tp_ranks"],
            dims=dims,
            workdir=cell_dir / "htsim",
            ep_ranks=plan["ep_ranks"],
            linkspeed_bps=plan["link_bps"],
            routed_moe_supply=supply,
            placement_manifest=_physical_manifest(plan["hostnames"]),
            collective_fixed_cost_envelope=plan["envelope"],
            collective_fixed_cost_arm=plan["arm"],
            **selection.sink_overrides(),
        )
    )
    envelope = sink.config.resolved_collective_fixed_cost_envelope
    if envelope is None:
        raise AssertionError("the named fixed-cost envelope did not resolve")

    install()
    configure(
        step_sink=sink,
        config=SimWorkerConfig(
            mode="virtual",
            efficiency=ROOFLINE_EFFICIENCY,
            step_records_path=str(cell_dir / "steps.jsonl"),
        ),
        **selection.worker_overrides(),
    )
    started = time.time()
    scheduler = build_in_process_scheduler(
        model_path=str(args.cache_dir / MODEL_RELATIVE_PATH),
        device=SGLANG_DEVICE,
        dtype=SGLANG_DTYPE,
        tp_size=1,
        page_size=SGLANG_PAGE_SIZE,
        context_length=SGLANG_CONTEXT_LENGTH,
        max_total_tokens=SGLANG_MAX_TOTAL_TOKENS,
        max_running_requests=SGLANG_MAX_RUNNING_REQUESTS,
        chunked_prefill_size=SGLANG_CHUNKED_PREFILL_SIZE,
        random_seed=SGLANG_SEED,
    )
    worker = latest_worker()
    if worker is None:
        raise AssertionError("the simulated SGLang worker was not constructed")
    if worker.step_sink is not sink:
        raise AssertionError("the worker did not receive this driver's step sink")
    if worker.host_model != selection.host_model:
        raise AssertionError("the worker did not receive this cell's host model")
    pump = SglangSchedulerPump(scheduler)
    gate = RequestAdmissionGate(
        worker.clock,
        bookkeeper,
        mode=AdmissionMode.ARRIVAL_GATED,
    )

    def submit(arrival: Any) -> None:
        index = REQUEST_IDS.index(arrival.request_id)
        pump.submit(
            [
                tokenized_generate_request(
                    request_id=arrival.request_id,
                    input_token_ids=prompt_token_ids(index),
                    max_new_tokens=MAX_NEW_TOKENS,
                    tokenizer=getattr(scheduler, "tokenizer", None),
                )
            ]
        )

    batch_runs = 0
    idle_steps = 0
    step_shapes: list[dict[str, Any]] = []
    while gate.has_pending or pump.has_unfinished_requests:
        gate.admit_ready(submit)
        if pump.has_unfinished_requests:
            before = len(worker.step_records)
            outcome = pump.step()
            delta = len(worker.step_records) - before
            if outcome.ran_batch:
                batch_runs += 1
                if delta != 1:
                    raise AssertionError(
                        f"one scheduler batch produced {delta} step records"
                    )
                step_shapes.append(
                    {
                        "step_ordinal": outcome.step_ordinal,
                        "batch_size": outcome.batch_size,
                        "forward_mode": outcome.forward_mode,
                    }
                )
            else:
                idle_steps += 1
                if delta:
                    raise AssertionError("an idle scheduler step emitted a record")
                if pump.has_unfinished_requests:
                    raise AssertionError(
                        "the scheduler ran no batch while requests were unfinished"
                    )
        elif gate.has_pending:
            gate.advance_to_next_arrival()
    wall_seconds = time.time() - started

    records = tuple(worker.step_records)
    results = tuple(worker.step_results)
    if len(records) != len(results):
        raise AssertionError("step record and result cardinality differ")
    locality_by_step = {
        outcome.step_index: outcome for outcome in sink.locality_outcomes
    }
    network_by_step = {outcome.step_index: outcome for outcome in sink.outcomes}
    if len(locality_by_step) != len(sink.locality_outcomes):
        raise AssertionError("locality outcomes repeat a step index")

    reducer = HtsimRequestMetricReducer(arrivals)
    step_rows: list[dict[str, Any]] = []
    inventory = {
        "tp_allreduce_counts": set(),
        "tp_site_sets": set(),
        "moe_counts": set(),
        "artifact_counts": set(),
        "backend_runs": 0,
    }
    conservation = {
        "intervals": 0,
        "interval_conservation_failures": 0,
        "makespan_conservation_failures": 0,
        "artifact_partition_failures": 0,
        "mid_prompt_extend_rows": 0,
        "unsampled_rows": 0,
        "inactive_component_violations": 0,
        "base_sum_failures": 0,
        "compute_service_failures": 0,
        "medium_projection_missing": 0,
    }
    total_routed_bytes = 0
    total_nvlink_bytes = 0
    total_fabric_bytes = 0
    observed_base_values: list[int] = []

    for record, result in zip(records, results, strict=True):
        locality = locality_by_step.get(record.step_index)
        network = network_by_step.get(record.step_index)
        if locality is None or network is None:
            raise AssertionError(
                f"step {record.step_index} was not simulated by this sink"
            )
        for scheduled in record.scheduled:
            if (
                scheduled.phase.value == "prefill"
                and scheduled.context_length != PROMPT_TOKENS
            ):
                conservation["mid_prompt_extend_rows"] += 1
        if record.num_sampled != len(record.scheduled):
            conservation["unsampled_rows"] += 1

        sites = layer_tp_allreduce_sites(
            record, dims, ep_ranks=plan["ep_ranks"], routed_supply=supply
        )
        allreduces = step_tp_allreduces(
            record,
            dims,
            plan["tp_ranks"],
            ep_ranks=plan["ep_ranks"],
            routed_supply=supply,
        )
        alltoalls = step_moe_alltoalls(
            record, dims, plan["ep_ranks"], routed_supply=supply
        )
        inventory["tp_site_sets"].add(tuple(sites))
        inventory["tp_allreduce_counts"].add(len(allreduces))
        inventory["moe_counts"].add(len(alltoalls))
        inventory["artifact_counts"].add(locality.artifact_count)
        inventory["backend_runs"] += locality.backend_runs

        if len(locality.composed_phase_service_ps) != locality.artifact_count:
            conservation["artifact_partition_failures"] += 1
        if not any(
            (
                locality.local_phase_service_ps,
                locality.base_phase_latency_ps,
                locality.local_phase_medium,
            )
        ):
            conservation["medium_projection_missing"] += 1
        expected_base_ps = collective_count(name) * arm_base_latency_ps(name)
        if sum(locality.base_phase_latency_ps) != expected_base_ps:
            conservation["base_sum_failures"] += 1
        if sum(locality.composed_phase_service_ps) != result.step_latency_ps:
            conservation["makespan_conservation_failures"] += 1
        gpu_service_ps = sum(
            composed
            for composed, medium in zip(
                locality.composed_phase_service_ps,
                locality.local_phase_medium,
                strict=True,
            )
            if medium != NVLINK_MEDIUM
        )
        if gpu_service_ps != locality.compute_service_ps:
            conservation["compute_service_failures"] += 1

        detail = attribute_step_detail(result, locality)
        metrics = reducer.consume(record, result, locality)
        conservation["intervals"] += len(metrics)
        for metric in metrics:
            if metric.attribution.total_ps != metric.latency_ps:
                conservation["interval_conservation_failures"] += 1
            if (
                metric.attribution.kv_ps
                or metric.attribution.dma_ps
                or metric.attribution.nic_ps
                or metric.attribution.control_ps
            ):
                conservation["inactive_component_violations"] += 1

        total_routed_bytes += locality.total_directed_bytes
        total_nvlink_bytes += locality.nvlink_directed_bytes
        total_fabric_bytes += locality.fabric_directed_bytes
        observed_base_values.extend(
            value for value in locality.base_phase_latency_ps if value
        )

        step_rows.append(
            {
                "step_index": record.step_index,
                "virtual_time_ps": record.virtual_time_ps,
                "step_latency_ps": result.step_latency_ps,
                "completed_at_ps": result.completed_at_ps,
                "total_new_tokens": record.total_new_tokens,
                "num_sampled": record.num_sampled,
                "scheduled": [
                    {
                        "request_id": scheduled.request_id,
                        "phase": scheduled.phase.value,
                        "num_new_tokens": scheduled.num_new_tokens,
                        "context_length": scheduled.context_length,
                        "num_cached_tokens": scheduled.num_cached_tokens,
                    }
                    for scheduled in record.scheduled
                ],
                "compute_service_ps": locality.compute_service_ps,
                "nvlink_service_ps": locality.nvlink_service_ps,
                "artifact_count": locality.artifact_count,
                "backend_runs": locality.backend_runs,
                "total_directed_bytes": locality.total_directed_bytes,
                "nvlink_directed_bytes": locality.nvlink_directed_bytes,
                "fabric_directed_bytes": locality.fabric_directed_bytes,
                "routing_mode": network.routing_mode,
                "placement_epoch": network.placement_epoch,
                "quiescent": network.quiescent,
                "host_profile": network.host_profile,
                "host_launch_count": network.host_launch_count,
                "host_launch_floor_ps": network.host_launch_floor_ps,
                "provider_compute_ps": network.provider_compute_ps,
                "exposed_host_ps": network.exposed_host_ps,
                "composed": list(locality.composed_phase_service_ps),
                "fabric": list(locality.fabric_phase_service_ps),
                "local": list(locality.local_phase_service_ps),
                "base": list(locality.base_phase_latency_ps),
                "medium": "".join(
                    NVLINK_CODE if value == NVLINK_MEDIUM else GPU_COMPUTE_CODE
                    for value in locality.local_phase_medium
                ),
                "attribution": _coarse_json(detail.attribution),
                "media": _media_json(detail.media),
                "masked_nvlink_ps": detail.masked.nvlink_ps,
                "masked_fabric_ps": detail.masked.fabric_ps,
            }
        )

    totals = {row.request_id: row for row in reducer.totals()}
    completions = {row.request_id: row for row in pump.completions}
    request_rows = []
    for index, request_id in enumerate(REQUEST_IDS):
        total = totals.get(request_id)
        completion = completions.get(request_id)
        tpot = None if total is None or total.tpot_ps is None else total.tpot_ps
        request_rows.append(
            {
                "request_id": request_id,
                "arrival_ps": index * ARRIVAL_SPACING_PS,
                "finish_reason": None if completion is None else completion.finish_reason,
                "framework_output_tokens": (
                    None if completion is None else completion.output_token_count
                ),
                "framework_cached_tokens": (
                    None if completion is None else completion.cached_token_count
                ),
                "retraction_count": (
                    None if completion is None else completion.retraction_count
                ),
                "token_count": None if total is None else total.token_count,
                "first_token_at_ps": None if total is None else total.first_token_at_ps,
                "last_token_at_ps": None if total is None else total.last_token_at_ps,
                "ttft_ps": None if total is None else total.ttft_ps,
                "tpot_numerator": None if tpot is None else tpot.numerator,
                "tpot_denominator": None if tpot is None else tpot.denominator,
                "ttft_attribution": (
                    None if total is None else _coarse_json(total.ttft_attribution)
                ),
                "decode_attribution": (
                    None if total is None else _coarse_json(total.decode_attribution)
                ),
                "ttft_media": None if total is None else _media_json(total.ttft_media),
                "decode_media": (
                    None if total is None else _media_json(total.decode_media)
                ),
            }
        )

    payload = {
        "name": name,
        "topology": plan["topology"],
        "link_bps": plan["link_bps"],
        "arm": plan["arm"],
        "host_arm": plan["host"],
        "envelope": {
            "envelope_id": envelope.envelope_id,
            "claim": envelope.claim,
            "arm_evidence_class": envelope.arm_evidence_class(plan["arm"]),
            "arm_evidence_note": envelope.arm_evidence_note(plan["arm"]),
            "arm_base_latency_ps": arm_base_latency_ps(name),
            "bracket_ps": list(envelope.bracket_ps(EP_WORLD)),
            "realized_bracket_ps": list(envelope.realized_bracket_ps(EP_WORLD)),
            "resolved_evidence_class": sink.config.resolved_collective_evidence_class,
            "resolved_profile_id": (
                None
                if sink.config.resolved_collective_latency_profile is None
                else sink.config.resolved_collective_latency_profile.profile_id
            ),
        },
        "host": {
            "profile": selection.profile,
            "launch_count": selection.launch_count,
            "launch_floor_ps": selection.launch_floor_ps,
            "device_key": selection.gpu.name,
            "provider_envelope": selection.provider_envelope,
            "is_default": selection.is_default,
            "transfer_disclosure": selection.transfer_disclosure,
            "describe": selection.describe(),
        },
        "wall_seconds": round(wall_seconds, 2),
        "scheduler_steps": batch_runs,
        "idle_steps": idle_steps,
        "step_records": len(records),
        "simulated_steps": len(sink.locality_outcomes),
        "network_outcomes": len(sink.outcomes),
        "htsim_invocations": inventory["backend_runs"],
        "total_routed_bytes": total_routed_bytes,
        "total_nvlink_bytes": total_nvlink_bytes,
        "total_fabric_bytes": total_fabric_bytes,
        "inventory": {
            "tp_site_sets": sorted(list(value) for value in inventory["tp_site_sets"]),
            "tp_allreduce_counts": sorted(inventory["tp_allreduce_counts"]),
            "moe_counts": sorted(inventory["moe_counts"]),
            "artifact_counts": sorted(inventory["artifact_counts"]),
        },
        "identity": {
            "driver_pid": os.getpid(),
            "sink_is_worker_sink": worker.step_sink is sink,
            "host_model_agrees": worker.host_model == selection.host_model,
            "admitted": list(gate.admitted_request_ids),
            "submitted": list(pump.submitted_request_ids),
            "finished": list(pump.finished_request_ids),
            "scheduler_ids_subset": all(
                scheduled.request_id in REQUEST_IDS
                for record in records
                for scheduled in record.scheduled
            ),
            "worker_class": type(worker).__name__,
            "scheduler_class": type(scheduler).__name__,
            "tree_cache_class": type(getattr(scheduler, "tree_cache", None)).__name__,
        },
        "step_shapes": step_shapes,
        "requests": request_rows,
        "steps": step_rows,
        "conservation": conservation,
        "observed_base_values": sorted(set(observed_base_values)),
    }
    (cell_dir / "cell.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _back44(args: argparse.Namespace) -> None:
    """Execute the BACK-44 negative control and record its refusal."""

    from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
    from simllm.core import RequestPhase, ScheduledRequest, StepRecord
    from simllm.preplay import project_framework_routing
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply

    routed = project_framework_routing(args.routing_trace)
    mixed_ep = (0, 1, 2, 3)
    placement = ExpertPlacementSnapshot(
        placement_epoch=0,
        expert_owners=tuple(
            (layer, expert, expert % len(mixed_ep))
            for layer in range(NUM_LAYERS)
            for expert in range(NUM_EXPERTS)
        ),
    )
    supply = RoutedMoeSupply(
        engine_rank=ENGINE_RANK,
        routed_experts=routed,
        placements=(placement,),
        step_placement_epochs=((0, 0),),
    )
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(
                REQUEST_IDS[0],
                RequestPhase.PREFILL,
                PROMPT_TOKENS,
                context_length=PROMPT_TOKENS,
            )
        ],
        num_sampled=1,
    )
    control_dir = args.run_dir / "cells" / BACK44_STAGE
    control_dir.mkdir(parents=True, exist_ok=False)
    refused = False
    message = ""
    try:
        sink = HtsimStepSink(
            HtsimStepSinkConfig(
                profile=PROFILE,
                tp_ranks=(0, 1),
                dims=_dims(1),
                workdir=control_dir / "htsim",
                ep_ranks=mixed_ep,
                linkspeed_bps=LINK_400G,
                routed_moe_supply=supply,
                placement_manifest=_physical_manifest(
                    ("node-0", "node-0", "node-1", "node-1")
                ),
            )
        )
        sink(record)
    except ValueError as error:
        refused = True
        message = str(error)
    (control_dir / "cell.json").write_text(
        json.dumps(
            {
                "name": BACK44_STAGE,
                "refused": refused,
                "message": message,
                "tp_ranks": [0, 1],
                "ep_ranks": list(mixed_ep),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


# --------------------------------------------- independent recomputation -----


def _step_media(step: dict[str, Any]) -> dict[str, int]:
    """Recompute one step's medium partition with the standard library only.

    Deliberately duplicates the ownership rule rather than calling it: an
    artifact's realized service is the maximum of its two media, the medium
    whose own service equals that maximum owns the whole realized service, the
    base latency is owned by no resource, and a tie is co-critical.
    """

    kernel_ps = 0
    nvlink_ps = 0
    fabric_ps = 0
    co_critical_ps = 0
    base_ps = 0
    for composed, fabric, local, base, medium in zip(
        step["composed"],
        step["fabric"],
        step["local"],
        step["base"],
        step["medium"],
        strict=True,
    ):
        if composed != base + max(local, fabric):
            raise AssertionError("artifact composition disagrees with its own terms")
        base_ps += base
        realized = max(local, fabric)
        if realized == 0:
            continue
        if medium == GPU_COMPUTE_CODE:
            kernel_ps += realized
        elif local == fabric:
            co_critical_ps += realized
        elif local > fabric:
            nvlink_ps += realized
        else:
            fabric_ps += realized
    return {
        "kernel_ps": kernel_ps,
        "nvlink_ps": nvlink_ps,
        "fabric_ps": fabric_ps,
        "co_critical_ps": co_critical_ps,
        "collective_base_ps": base_ps,
    }


def _independent_request_metrics(cell: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Recompute TTFT, TPOT and both medium partitions from the per-step rows."""

    steps = sorted(cell["steps"], key=lambda step: step["step_index"])
    arrivals = {row["request_id"]: row["arrival_ps"] for row in cell["requests"]}
    components = ("kernel_ps", "nvlink_ps", "fabric_ps", "co_critical_ps",
                  "collective_base_ps")
    state: dict[str, dict[str, Any]] = {}
    for step in steps:
        media = _step_media(step)
        if sum(media.values()) != step["step_latency_ps"]:
            raise AssertionError("recomputed step partition does not conserve")
        for scheduled in step["scheduled"]:
            request_id = scheduled["request_id"]
            row = state.setdefault(
                request_id,
                {
                    "accounted_through_ps": arrivals[request_id],
                    "first_token_at_ps": None,
                    "last_token_at_ps": None,
                    "token_count": 0,
                    "inter_token_sum_ps": 0,
                    "inter_token_count": 0,
                    "ttft": {"queue_ps": 0, **dict.fromkeys(components, 0)},
                    "decode": {"queue_ps": 0, **dict.fromkeys(components, 0)},
                },
            )
            queue_ps = step["virtual_time_ps"] - row["accounted_through_ps"]
            row["accounted_through_ps"] = step["completed_at_ps"]
            row["token_count"] += 1
            target = "ttft" if row["first_token_at_ps"] is None else "decode"
            row[target]["queue_ps"] += queue_ps
            for component in components:
                row[target][component] += media[component]
            if row["first_token_at_ps"] is None:
                row["first_token_at_ps"] = step["completed_at_ps"]
            else:
                row["inter_token_sum_ps"] += (
                    step["completed_at_ps"] - row["last_token_at_ps"]
                )
                row["inter_token_count"] += 1
            row["last_token_at_ps"] = step["completed_at_ps"]
    metrics: dict[str, dict[str, Any]] = {}
    for request_id, row in state.items():
        metrics[request_id] = {
            "token_count": row["token_count"],
            "first_token_at_ps": row["first_token_at_ps"],
            "last_token_at_ps": row["last_token_at_ps"],
            "ttft_ps": row["first_token_at_ps"] - arrivals[request_id],
            "tpot": (
                Fraction(row["inter_token_sum_ps"], row["inter_token_count"])
                if row["inter_token_count"]
                else None
            ),
            "ttft_media": row["ttft"],
            "decode_media": row["decode"],
        }
    return metrics


# ------------------------------------------------------------- analysis ------


def _read_cells(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    cells = {}
    for name in cell_names():
        path = args.run_dir / "cells" / name / "cell.json"
        cells[name] = json.loads(path.read_text(encoding="utf-8"))
    return cells


def _families(kind: str) -> dict[str, tuple[str, ...]]:
    """Group cell names into the frozen comparison families."""

    groups: dict[str, list[str]] = {}
    for name, (topology, link_bps, arm, host) in CELLS.items():
        prefix = name.rsplit(f"-{arm}-", 1)[0]
        if kind == "arm":
            key = f"{prefix}-{host}"
        elif kind == "host":
            key = f"{prefix}-{arm}"
        else:
            raise ValueError(f"unknown family kind {kind!r}")
        groups.setdefault(key, []).append(name)
    return {
        key: tuple(
            sorted(names, key=lambda value: ARMS.index(CELLS[value][2]))
            if kind == "arm"
            else sorted(names, key=lambda value: CELLS[value][3])
        )
        for key, names in groups.items()
    }


def _max_batch(cell: dict[str, Any]) -> int:
    return max((len(step["scheduled"]) for step in cell["steps"]), default=0)


def _sum_ttft_ps(cell: dict[str, Any]) -> int:
    return sum(row["ttft_ps"] for row in cell["requests"])


def _median_step_ps(cell: dict[str, Any]) -> int:
    values = sorted(step["step_latency_ps"] for step in cell["steps"])
    return values[len(values) // 2]


def _score_e1(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compared = 0
    failures: list[dict[str, Any]] = []
    for name, cell in cells.items():
        independent = _independent_request_metrics(cell)
        for row in cell["requests"]:
            request_id = row["request_id"]
            expected = independent.get(request_id)
            if expected is None:
                failures.append({"cell": name, "request_id": request_id, "why": "absent"})
                continue
            compared += 1
            observed_tpot = (
                None
                if row["tpot_numerator"] is None
                else Fraction(row["tpot_numerator"], row["tpot_denominator"])
            )
            mismatches = []
            for field in ("token_count", "first_token_at_ps", "last_token_at_ps",
                          "ttft_ps"):
                if expected[field] != row[field]:
                    mismatches.append(field)
            if expected["tpot"] != observed_tpot:
                mismatches.append("tpot")
            for field in ("ttft_media", "decode_media"):
                published = row[field]
                for component, value in expected[field].items():
                    if published[component] != value:
                        mismatches.append(f"{field}.{component}")
            if mismatches:
                failures.append(
                    {"cell": name, "request_id": request_id, "fields": mismatches}
                )
    return {
        "passed": compared > 0 and not failures,
        "compared": compared,
        "failures": failures[:20],
        "failure_count": len(failures),
    }


def _score_e2(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cell = cells["cross400-off-ideal"]
    rows = {row["request_id"]: row for row in cell["requests"]}
    observed_ttft = {
        request_id: row["ttft_ps"] / PS_PER_US for request_id, row in rows.items()
    }
    observed_tpot = {
        request_id: (
            None
            if row["tpot_numerator"] is None
            else row["tpot_numerator"] / row["tpot_denominator"] / PS_PER_US
        )
        for request_id, row in rows.items()
    }
    mismatches = []
    if cell["scheduler_steps"] != ACCEPTED_E2E_STEPS:
        mismatches.append(
            f"scheduler_steps {cell['scheduler_steps']} != {ACCEPTED_E2E_STEPS}"
        )
    for request_id, accepted in ACCEPTED_E2E_TTFT_US.items():
        if abs(observed_ttft[request_id] - accepted) > E2_TOLERANCE_US:
            mismatches.append(f"ttft[{request_id}] {observed_ttft[request_id]:.5f}")
    for request_id, accepted in ACCEPTED_E2E_TPOT_US.items():
        value = observed_tpot[request_id]
        if value is None or abs(value - accepted) > E2_TOLERANCE_US:
            mismatches.append(f"tpot[{request_id}] {value}")
    return {
        "passed": not mismatches,
        "scheduler_steps": cell["scheduler_steps"],
        "ttft_us": {key: round(value, 5) for key, value in observed_ttft.items()},
        "tpot_us": {
            key: None if value is None else round(value, 5)
            for key, value in observed_tpot.items()
        },
        "mismatches": mismatches,
    }


def _score_b1(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rounds = []
    for family, names in sorted(_families("arm").items()):
        steps = [cells[name]["scheduler_steps"] for name in names]
        monotone = all(left >= right for left, right in pairwise(steps))
        strict = steps[0] > steps[-1]
        rounds.append(
            {
                "family": family,
                "arms": list(names),
                "scheduler_steps": steps,
                "non_increasing": monotone,
                "strict_off_to_upper": strict,
                "passed": monotone and strict,
            }
        )
    return {"passed": all(entry["passed"] for entry in rounds), "families": rounds}


def _score_b2(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rounds = []
    for family, names in sorted(_families("arm").items()):
        batches = [_max_batch(cells[name]) for name in names]
        monotone = all(
            left <= right for left, right in pairwise(batches)
        )
        full = batches[-1] == len(REQUEST_IDS)
        rounds.append(
            {
                "family": family,
                "arms": list(names),
                "max_batch": batches,
                "non_decreasing": monotone,
                "upper_is_full": full,
                "passed": monotone and full,
            }
        )
    return {"passed": all(entry["passed"] for entry in rounds), "families": rounds}


def _score_b3(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rounds = []
    for family, names in sorted(_families("host").items()):
        ideal, turing = (cells[name]["scheduler_steps"] for name in names)
        rounds.append(
            {
                "family": family,
                "arms": list(names),
                "ideal_steps": ideal,
                "turing_steps": turing,
                "passed": turing <= ideal,
                "strict": turing < ideal,
            }
        )
    return {
        "passed": all(entry["passed"] for entry in rounds)
        and any(entry["strict"] for entry in rounds),
        "families": rounds,
    }


def _score_b4(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from simllm.traffic import arm_ratio_envelope

    rounds = []
    for host in ("ideal", "turing"):
        rows = [
            (
                arm,
                _sum_ttft_ps(cells[f"cross100-{arm}-{host}"]),
                _sum_ttft_ps(cells[f"cross400-{arm}-{host}"]),
            )
            for arm in ARMS
        ]
        envelope = arm_ratio_envelope(
            f"cross-node 100G over 400G, host {host}",
            "summed TTFT at 100 Gbit/s",
            "summed TTFT at 400 Gbit/s",
            rows,
        )
        ratios = [ratio for _, ratio in envelope.arm_ratios]
        rounds.append(
            {
                "host": host,
                "arm_ratios": list(envelope.arm_ratios),
                "minimum": envelope.minimum,
                "maximum": envelope.maximum,
                "brackets_unity": envelope.brackets_unity,
                "width": envelope.width,
                "all_above_one": all(ratio > 1.0 for ratio in ratios),
                "non_increasing": all(
                    left >= right for left, right in pairwise(ratios)
                ),
                "passed": (
                    all(ratio > 1.0 for ratio in ratios)
                    and not envelope.brackets_unity
                    and all(
                        left >= right for left, right in pairwise(ratios)
                    )
                ),
            }
        )
    return {"passed": all(entry["passed"] for entry in rounds), "hosts": rounds}


def _score_b5(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from simllm.traffic import arm_ratio_envelope

    rounds = []
    for host in ("ideal", "turing"):
        by_name = arm_ratio_envelope(
            f"intra-node over cross400, arm-name matched, host {host}",
            "summed intra-node TTFT",
            "summed cross-node TTFT at 400 Gbit/s",
            [
                (
                    arm,
                    _sum_ttft_ps(cells[f"intra-{arm}-{host}"]),
                    _sum_ttft_ps(cells[f"cross400-{arm}-{host}"]),
                )
                for arm in ARMS
            ],
        )
        by_constant = arm_ratio_envelope(
            f"intra-node over cross400, constant matched, host {host}",
            "summed intra-node TTFT",
            "summed cross-node TTFT at 400 Gbit/s",
            [
                (
                    "0 ps",
                    _sum_ttft_ps(cells[f"intra-off-{host}"]),
                    _sum_ttft_ps(cells[f"cross400-off-{host}"]),
                ),
                (
                    f"{B200_LOCAL_BASE_PS} ps",
                    _sum_ttft_ps(cells[f"intra-upper-{host}"]),
                    _sum_ttft_ps(cells[f"cross400-lower-{host}"]),
                ),
            ],
        )
        matched_high = by_constant.arm_ratios[-1][1]
        rounds.append(
            {
                "host": host,
                "arm_name_matched": {
                    "arm_ratios": list(by_name.arm_ratios),
                    "minimum": by_name.minimum,
                    "maximum": by_name.maximum,
                    "brackets_unity": by_name.brackets_unity,
                },
                "constant_matched": {
                    "arm_ratios": list(by_constant.arm_ratios),
                    "minimum": by_constant.minimum,
                    "maximum": by_constant.maximum,
                    "brackets_unity": by_constant.brackets_unity,
                },
                "passed": (
                    not by_name.brackets_unity
                    and by_constant.brackets_unity
                    and matched_high > 1.0
                ),
            }
        )
    return {"passed": all(entry["passed"] for entry in rounds), "hosts": rounds}


def _score_b6(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rounds = []
    for prefix in ("intra", "cross400", "cross100"):
        base = {
            row["request_id"]: row["ttft_ps"]
            for row in cells[f"{prefix}-off-ideal"]["requests"]
        }
        loaded = {
            row["request_id"]: row["ttft_ps"]
            for row in cells[f"{prefix}-upper-turing"]["requests"]
        }
        ratios = {
            request_id: loaded[request_id] / base[request_id]
            for request_id in REQUEST_IDS[1:]
        }
        anchored = loaded["p0"] / base["p0"]
        rounds.append(
            {
                "topology": prefix,
                "scored_ratios": {key: round(value, 5) for key, value in ratios.items()},
                "anchored_p0_ratio": round(anchored, 5),
                "all_at_least_two": all(value >= 2.0 for value in ratios.values()),
                "min_ratio": min(ratios.values()),
            }
        )
    intra = next(entry for entry in rounds if entry["topology"] == "intra")
    cross = next(entry for entry in rounds if entry["topology"] == "cross400")
    ordering = intra["min_ratio"] > cross["min_ratio"]
    return {
        "passed": all(entry["all_at_least_two"] for entry in rounds) and ordering,
        "topologies": rounds,
        "intra_multiplier_exceeds_cross400": ordering,
    }


def _bands(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for name, cell in cells.items():
        floor_ps, ceiling_ps = STEP_BANDS_PS[name]
        latencies = [step["step_latency_ps"] for step in cell["steps"]]
        ttfts = [row["ttft_ps"] for row in cell["requests"]]
        tpots = [
            row["tpot_numerator"] / row["tpot_denominator"]
            for row in cell["requests"]
            if row["tpot_numerator"] is not None
        ]
        rows[name] = {
            "step_min_ps": min(latencies),
            "step_max_ps": max(latencies),
            "step_band_ps": [floor_ps, ceiling_ps],
            "step_in_band": all(floor_ps <= value <= ceiling_ps for value in latencies),
            "ttft_min_ps": min(ttfts),
            "ttft_max_ps": max(ttfts),
            "ttft_in_band": all(
                floor_ps <= value <= TTFT_CEILING_STEPS * ceiling_ps for value in ttfts
            ),
            "tpot_min_ps": min(tpots),
            "tpot_max_ps": max(tpots),
            "tpot_in_band": all(
                floor_ps <= value <= TPOT_CEILING_STEPS * ceiling_ps for value in tpots
            ),
        }
    return rows


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    _check_frozen_registry()
    cells = _read_cells(args)
    control = json.loads(
        (args.run_dir / "cells" / BACK44_STAGE / "cell.json").read_text(
            encoding="utf-8"
        )
    )
    fatal: dict[str, dict[str, Any]] = {}

    def guard(name: str, ok: bool, detail: dict[str, Any]) -> None:
        fatal[name] = {"held": bool(ok), **detail}

    provenance = check_trace_provenance(args.routing_trace)
    guard(
        "G1",
        _git_head(args.sglang_source) == SGLANG_PINNED_COMMIT,
        {
            "framework": provenance.get("framework"),
            "routing_source": provenance.get("routing_source"),
            "model_revision": provenance.get("model_revision"),
            "observed_source": provenance.get("observed_source"),
            "sglang_source_head": _git_head(args.sglang_source),
        },
    )
    guard(
        "G2",
        all(
            cell["identity"]["sink_is_worker_sink"]
            and cell["scheduler_steps"]
            == cell["step_records"]
            == cell["simulated_steps"]
            == cell["network_outcomes"]
            and cell["identity"]["scheduler_ids_subset"]
            and cell["identity"]["worker_class"] == "SimTpModelWorker"
            and cell["identity"]["tree_cache_class"] == "RadixCache"
            for cell in cells.values()
        ),
        {
            name: {
                "scheduler_steps": cell["scheduler_steps"],
                "step_records": cell["step_records"],
                "simulated_steps": cell["simulated_steps"],
                "network_outcomes": cell["network_outcomes"],
            }
            for name, cell in cells.items()
        },
    )
    guard(
        "G3",
        all(
            cell["conservation"]["mid_prompt_extend_rows"] == 0
            and cell["conservation"]["unsampled_rows"] == 0
            and all(row["retraction_count"] == 0 for row in cell["requests"])
            for cell in cells.values()
        ),
        {
            "mid_prompt_extend_rows": sum(
                cell["conservation"]["mid_prompt_extend_rows"] for cell in cells.values()
            ),
            "unsampled_rows": sum(
                cell["conservation"]["unsampled_rows"] for cell in cells.values()
            ),
            "retractions": sum(
                row["retraction_count"] or 0
                for cell in cells.values()
                for row in cell["requests"]
            ),
        },
    )
    intervals = sum(cell["conservation"]["intervals"] for cell in cells.values())
    guard(
        "G4",
        intervals > 0
        and all(
            cell["conservation"]["makespan_conservation_failures"] == 0
            and cell["conservation"]["interval_conservation_failures"] == 0
            and cell["conservation"]["inactive_component_violations"] == 0
            and cell["conservation"]["artifact_partition_failures"] == 0
            and cell["conservation"]["medium_projection_missing"] == 0
            and cell["conservation"]["compute_service_failures"] == 0
            and all(
                step["completed_at_ps"] == step["virtual_time_ps"] + step["step_latency_ps"]
                for step in cell["steps"]
            )
            and all(row["ttft_ps"] > 0 for row in cell["requests"])
            for cell in cells.values()
        ),
        {"intervals": intervals},
    )
    guard(
        "G5",
        all(
            cell["total_fabric_bytes"] == 0
            and cell["htsim_invocations"] == 0
            and cell["total_nvlink_bytes"] > 0
            for name, cell in cells.items()
            if CELLS[name][0] == "intra"
        )
        and all(
            cell["total_nvlink_bytes"] == 0
            and cell["total_fabric_bytes"] > 0
            and all(
                step["routing_mode"] == "captured"
                and step["placement_epoch"] == 0
                and step["quiescent"] is True
                for step in cell["steps"]
            )
            for name, cell in cells.items()
            if CELLS[name][0] == "cross"
        ),
        {
            "intra_backend_runs": sum(
                cells[name]["htsim_invocations"] for name in intra_cells()
            ),
            "cross_backend_runs": sum(
                cells[name]["htsim_invocations"] for name in cross_cells()
            ),
        },
    )
    guard(
        "G6",
        all(
            row["finish_reason"] == "length"
            and row["framework_output_tokens"] == MAX_NEW_TOKENS
            and row["token_count"] == MAX_NEW_TOKENS
            for cell in cells.values()
            for row in cell["requests"]
        ),
        {"requests": sum(len(cell["requests"]) for cell in cells.values())},
    )
    guard(
        "G7",
        all(
            cell["inventory"]["tp_site_sets"] == [["attention"]]
            and cell["inventory"]["tp_allreduce_counts"] == [INTRA_TP_ALLREDUCES]
            and cell["inventory"]["moe_counts"] == [MOE_ALLTOALLS]
            and cell["inventory"]["artifact_counts"] == [INTRA_ARTIFACTS]
            for name, cell in cells.items()
            if CELLS[name][0] == "intra"
        )
        and all(
            cell["inventory"]["tp_allreduce_counts"] == [0]
            and cell["inventory"]["moe_counts"] == [MOE_ALLTOALLS]
            and cell["inventory"]["artifact_counts"] == [CROSS_ARTIFACTS]
            for name, cell in cells.items()
            if CELLS[name][0] == "cross"
        ),
        {
            name: cell["inventory"]
            for name, cell in cells.items()
            if name in ("intra-off-ideal", "cross400-off-ideal")
        },
    )
    guard(
        "G8",
        all(cell["conservation"]["base_sum_failures"] == 0 for cell in cells.values())
        and all(
            set(cell["observed_base_values"]) <= {arm_base_latency_ps(name)}
            for name, cell in cells.items()
        ),
        {
            "max_representable_endpoint_bytes": MAX_CRITICAL_ENDPOINT_BYTES,
            "profile_envelope_max_bytes": PROFILE_ENDPOINT_MAX_BYTES,
            "max_observed_step_tokens": max(
                step["total_new_tokens"]
                for cell in cells.values()
                for step in cell["steps"]
            ),
        },
    )
    guard(
        "G9",
        all(
            cell["identity"]["host_model_agrees"]
            and (
                cell["host"]["launch_count"] == 0
                and cell["host"]["transfer_disclosure"] is None
                and all(step["exposed_host_ps"] == 0 for step in cell["steps"])
                if CELLS[name][3] == "ideal"
                else cell["host"]["launch_count"] == TURING_LAUNCH_COUNT
                and cell["host"]["launch_floor_ps"] == TURING_LAUNCH_FLOOR_PS
                and cell["host"]["device_key"] == "gtx1660-ti-sm75"
                and cell["host"]["provider_envelope"] == "b100"
                and bool(cell["host"]["transfer_disclosure"])
                and all(
                    step["compute_service_ps"] == TURING_COMPUTE_SERVICE_PS
                    for step in cell["steps"]
                )
            )
            for name, cell in cells.items()
        ),
        {
            "turing_compute_service_ps": TURING_COMPUTE_SERVICE_PS,
            "disclosure_present": all(
                bool(cells[name]["host"]["transfer_disclosure"])
                for name in cells
                if CELLS[name][3] == "turing"
            ),
        },
    )
    guard(
        "G10",
        bool(control["refused"]) and BACK44_MESSAGE in control["message"],
        {"message": control["message"]},
    )
    intra_bytes = {cells[name]["total_routed_bytes"] for name in intra_cells()}
    cross_bytes = {cells[name]["total_routed_bytes"] for name in cross_cells()}
    guard(
        "G11",
        intra_bytes == {INTRA_ROUTED_BYTES} and cross_bytes == {CROSS_ROUTED_BYTES},
        {
            "intra_bytes": sorted(intra_bytes),
            "cross_bytes": sorted(cross_bytes),
            "predicted_intra": INTRA_ROUTED_BYTES,
            "predicted_cross": CROSS_ROUTED_BYTES,
        },
    )

    exact = {"E1": _score_e1(cells), "E2": _score_e2(cells)}
    behavioral = {
        "B1": _score_b1(cells),
        "B2": _score_b2(cells),
        "B3": _score_b3(cells),
        "B4": _score_b4(cells),
        "B5": _score_b5(cells),
        "B6": _score_b6(cells),
    }
    if len(exact) != EXPECTED_EXACT_RELATIONS:
        raise AssertionError("exact relation denominator changed")
    if len(behavioral) != EXPECTED_BEHAVIORAL_RELATIONS:
        raise AssertionError("behavioral relation denominator changed")
    if len(fatal) != EXPECTED_FATAL_GUARDS:
        raise AssertionError("fatal guard roster changed")

    void = [name for name, detail in fatal.items() if not detail["held"]]
    decode_rates = {
        name: [
            PS_PER_SECOND / (row["tpot_numerator"] / row["tpot_denominator"])
            for row in cell["requests"]
            if row["tpot_numerator"] is not None
        ]
        for name, cell in cells.items()
    }
    return {
        "freeze_commit": FREEZE_COMMIT,
        "fatal_guards": fatal,
        "void": bool(void),
        "violated_fatal_guards": void,
        "exact_relations": exact,
        "exact_passed": sum(1 for detail in exact.values() if detail["passed"]),
        "exact_total": len(exact),
        "behavioral_relations": behavioral,
        "behavioral_passed": sum(1 for detail in behavioral.values() if detail["passed"]),
        "behavioral_total": len(behavioral),
        "bands": _bands(cells),
        "scale": {
            name: {
                "wall_seconds": cell["wall_seconds"],
                "scheduler_steps": cell["scheduler_steps"],
                "htsim_invocations": cell["htsim_invocations"],
                "total_routed_bytes": cell["total_routed_bytes"],
                "max_batch": _max_batch(cell),
                "median_step_ps": _median_step_ps(cell),
            }
            for name, cell in cells.items()
        },
        "cells": {
            name: {
                "topology": cell["topology"],
                "link_bps": cell["link_bps"],
                "arm": cell["arm"],
                "host_arm": cell["host_arm"],
                "envelope": cell["envelope"],
                "host": {
                    key: value
                    for key, value in cell["host"].items()
                    if key != "transfer_disclosure"
                },
                "scheduler_steps": cell["scheduler_steps"],
                "surcharge_per_step_ps": collective_count(name)
                * arm_base_latency_ps(name),
                "requests": [
                    {
                        "request_id": row["request_id"],
                        "arrival_ps": row["arrival_ps"],
                        "ttft_ps": row["ttft_ps"],
                        "tpot_ps": (
                            None
                            if row["tpot_numerator"] is None
                            else row["tpot_numerator"] / row["tpot_denominator"]
                        ),
                        "ttft_media": row["ttft_media"],
                        "decode_media": row["decode_media"],
                    }
                    for row in cell["requests"]
                ],
            }
            for name, cell in cells.items()
        },
        "diagnostics": {
            "decode_rate_tokens_per_second": {
                name: {"min": min(values), "max": max(values)}
                for name, values in decode_rates.items()
                if values
            },
            "surcharge_split_ps": {
                "intra_upper_at_capture": INTRA_TP_ALLREDUCES * B200_LOCAL_BASE_PS,
                "intra_upper_transferred": MOE_ALLTOALLS * B200_LOCAL_BASE_PS,
                "cross_lower_transferred": MOE_ALLTOALLS * B200_LOCAL_BASE_PS,
                "cross_upper_transferred": MOE_ALLTOALLS * CROSS_PROVISIONAL_BASE_PS,
            },
            "back44": control,
            "step_shapes": {name: cell["step_shapes"] for name, cell in cells.items()},
        },
    }


# -------------------------------------------------------------- driver -------


def _child_command(args: argparse.Namespace, mode: str) -> list[str]:
    return [
        str(args.sglang_python),
        str(Path(__file__).resolve()),
        "--cache-dir",
        str(args.cache_dir),
        "--sglang-python",
        str(args.sglang_python),
        "--sglang-source",
        str(args.sglang_source),
        "--routing-trace",
        str(args.routing_trace),
        "--htsim-rnic",
        str(args.htsim_rnic),
        "--run-dir",
        str(args.run_dir),
        "--internal",
        mode,
    ]


def _child_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic)
    return environment


def _run_children(args: argparse.Namespace, modes: list[str]) -> None:
    """Run child stages, at most ``args.jobs`` at a time."""

    environment = _child_environment(args)
    log_root = args.run_dir / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(modes)
    running: list[tuple[str, Any, Any]] = []
    while pending or running:
        while pending and len(running) < max(1, args.jobs):
            mode = pending.pop(0)
            log_path = log_root / f"{mode.replace(':', '-')}.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                _child_command(args, mode),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((mode, process, handle))
        mode, process, handle = running.pop(0)
        code = process.wait()
        handle.close()
        if code != 0:
            for _, other, other_handle in running:
                other.wait()
                other_handle.close()
            raise SystemExit(
                f"child stage {mode!r} failed with code {code}; see logs/"
                f"{mode.replace(':', '-')}.log"
            )


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    _check_frozen_registry()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    _run_children(args, [f"cell:{name}" for name in cell_names()] + [BACK44_STAGE])
    summary = _summarize(args)
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--routing-trace", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "child stages to run concurrently. Only the reported wall column is "
            "affected: every cell owns its work directory, scheduler and "
            "deterministic backend."
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--internal", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return 0
    if args.internal:
        import sys

        if str(REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(REPOSITORY_ROOT))
        if args.internal == BACK44_STAGE:
            _back44(args)
            return 0
        if args.internal.startswith("cell:"):
            _cell(args, args.internal.removeprefix("cell:"))
            return 0
        if args.internal != "summarize":
            raise SystemExit(f"unknown internal stage {args.internal!r}")
        summary = _summarize(args)
        (args.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        summary = run_study(args)
    print(
        json.dumps(
            {
                "void": summary["void"],
                "violated_fatal_guards": summary["violated_fatal_guards"],
                "exact": [summary["exact_passed"], summary["exact_total"]],
                "behavioral": [
                    summary["behavioral_passed"],
                    summary["behavioral_total"],
                ],
                "scale": summary["scale"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
