"""Run the frozen SGLang end-to-end closed-loop study.

The chain is: a real SGLang ``Scheduler`` driven step by step in the same
process that installed the sink, arrival-gated admission on the worker's own
virtual clock, SGLang's own captured post-selection expert ids driving the
fabric, ``htsim_rnic`` timing every dispatch and combine, and the per-request
TTFT and TPOT coming back out with a conserved seven-component partition.

Every acceptance clause, bound, band and relation this script evaluates is
frozen in ``expectations.md`` and was committed before the mechanism existed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / MODEL_REVISION
)

#: the SGLang commit the adapter is authored against
SGLANG_PINNED_COMMIT = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"

#: geometry of the captured model, frozen in expectations.md
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

#: the one rank that dispatches scheduled tokens
ENGINE_RANK = 0
#: declared arrival spacing, one millisecond between successive requests
ARRIVAL_SPACING_PS = 1_000_000_000
#: frozen backend profile and compute derate
PROFILE = "rnic-nn-fluid"
ROOFLINE_EFFICIENCY = 0.7

#: frozen SGLang server arguments that have no other home
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

#: request identity, prompt token ids and output length, frozen
REQUEST_IDS = ("p0", "p1", "p2", "p3")
PROMPT_TOKENS = 8
MAX_NEW_TOKENS = 12

BANDWIDTHS_BPS = (100_000_000_000, 200_000_000_000, 400_000_000_000)

#: cell name, expert-parallel world, link rate
CELLS = (
    ("ep8-400g", 8, 400_000_000_000),
    ("ep8-200g", 8, 200_000_000_000),
    ("ep8-100g", 8, 100_000_000_000),
    ("ep4-400g", 4, 400_000_000_000),
)
#: sink-free control: the same pump with no backend, reported never scored
CONTROL_CELL = "control-nosink"

#: physical sanity bounds from expectations.md, in bytes or picoseconds
S1_MOE_BYTES_FLOOR = PROMPT_TOKENS * NUM_LAYERS * 2 * 1 * VECTOR_BYTES
S1_MOE_BYTES_CEILING = PROMPT_TOKENS * NUM_LAYERS * 2 * 7 * VECTOR_BYTES
S2_SERIALIZATION_FLOOR_PS = 15_728_640
S2_SERIALIZATION_CEILING_PS = 110_100_480
S3_COMPUTE_FLOOR_PS = 60_000_000
S3_COMPUTE_CEILING_PS = 200_000_000
S4_PREFILL_FLOOR_PS = 80_000_000
S4_PREFILL_CEILING_PS = 2_000_000_000
S5_DECODE_FLOOR_PS = 100_000_000
S5_DECODE_CEILING_PS = 600_000_000

#: B1 three-point linearity tolerance
B1_ABSOLUTE_TOLERANCE_PS = 1_000
B1_RELATIVE_TOLERANCE = 1e-6
#: B4 expert-parallel compute band around the 1.5439 first-principles value
B4_COMPUTE_RATIO_FLOOR = 1.45
B4_COMPUTE_RATIO_CEILING = 1.65

#: scored relation counts, split by evidence class and never summed
EXPECTED_EXACT_RELATIONS = 5
EXPECTED_BEHAVIORAL_RELATIONS = 4

PS_PER_SECOND = 1_000_000_000_000


def prompt_token_ids(index: int) -> tuple[int, ...]:
    """The capture's own pressure-prompt rule, reproduced exactly."""

    return tuple(1000 + 100 * index + step for step in range(PROMPT_TOKENS))


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


def check_only(args: argparse.Namespace) -> None:
    """Validate every frozen input without producing an artifact."""

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

    if re.fullmatch(r"[0-9a-f]{40}", SGLANG_PINNED_COMMIT) is None:
        raise AssertionError("pinned SGLang provenance is malformed")
    if VECTOR_BYTES != 2048:
        raise AssertionError("routed hidden-vector size changed")
    if NUM_EXPERTS % 8 or NUM_EXPERTS % 4:
        raise AssertionError("expert count no longer divides both frozen EP worlds")
    if S1_MOE_BYTES_FLOOR != 786_432 or S1_MOE_BYTES_CEILING != 5_505_024:
        raise AssertionError("first prefill routed-byte bounds changed")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[2] != 20:
        raise AssertionError("400 Gbit/s byte serialization literal changed")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[1] != 40:
        raise AssertionError("200 Gbit/s byte serialization literal changed")
    if 8_000_000_000_000 // BANDWIDTHS_BPS[0] != 80:
        raise AssertionError("100 Gbit/s byte serialization literal changed")
    if BANDWIDTHS_BPS[1] != 2 * BANDWIDTHS_BPS[0]:
        raise AssertionError("bandwidth ladder must retain its 2x spacing")
    if BANDWIDTHS_BPS[2] != 2 * BANDWIDTHS_BPS[1]:
        raise AssertionError("bandwidth ladder must retain its 2x spacing")
    if len({name for name, *_ in CELLS}) != len(CELLS):
        raise AssertionError("cell names must be distinct")
    if {bandwidth for _, ep, bandwidth in CELLS if ep == 8} != set(BANDWIDTHS_BPS):
        raise AssertionError("the three-point bandwidth ladder lost a cell")
    if CONTROL_CELL in {name for name, *_ in CELLS}:
        raise AssertionError("the sink-free control must not be a scored cell")
    if S2_SERIALIZATION_FLOOR_PS != 48 * PROMPT_TOKENS * 1 * VECTOR_BYTES * 20:
        raise AssertionError("the 400G serialization floor literal changed")
    if S2_SERIALIZATION_CEILING_PS != 48 * PROMPT_TOKENS * 7 * VECTOR_BYTES * 20:
        raise AssertionError("the 400G serialization ceiling literal changed")
    for floor, ceiling, label in (
        (S3_COMPUTE_FLOOR_PS, S3_COMPUTE_CEILING_PS, "compute service"),
        (S4_PREFILL_FLOOR_PS, S4_PREFILL_CEILING_PS, "prefill makespan"),
        (S5_DECODE_FLOOR_PS, S5_DECODE_CEILING_PS, "decode makespan"),
    ):
        if floor >= ceiling:
            raise AssertionError(f"{label} bounds are inverted")
    if not 0.0 < ROOFLINE_EFFICIENCY <= 1.0:
        raise AssertionError("roofline derate left its valid range")
    if not B4_COMPUTE_RATIO_FLOOR < 1.5439 < B4_COMPUTE_RATIO_CEILING:
        raise AssertionError("the B4 band no longer contains its own prediction")
    if SGLANG_CHUNKED_PREFILL_SIZE > 0:
        raise AssertionError("chunked prefill must stay disabled, see G3")
    if len(REQUEST_IDS) != len(set(REQUEST_IDS)):
        raise AssertionError("request identities must be distinct")
    if any(len(prompt_token_ids(index)) != PROMPT_TOKENS for index in range(4)):
        raise AssertionError("frozen prompt shape changed")
    if MAX_NEW_TOKENS - 1 > 19:
        raise AssertionError("output length exceeds the captured decode extent")
    if EXPECTED_EXACT_RELATIONS != 5 or EXPECTED_BEHAVIORAL_RELATIONS != 4:
        raise AssertionError("scored relation denominators changed")
    print(
        json.dumps(
            {
                "check_only": True,
                "artifacts_written": 0,
                "cells": [name for name, *_ in CELLS] + [CONTROL_CELL],
                "requests": list(REQUEST_IDS),
                "exact_total": EXPECTED_EXACT_RELATIONS,
                "behavioral_total": EXPECTED_BEHAVIORAL_RELATIONS,
                "sglang_pinned_commit": SGLANG_PINNED_COMMIT,
                "run_dir": str(args.run_dir),
            },
            sort_keys=True,
        )
    )


# ------------------------------------------------------- raw trace reading ---


def read_raw_trace(path: Path) -> dict[str, Any]:
    """Re-read the v2 framework trace with the standard library only.

    Nothing from ``simllm`` participates. This is the independent side of every
    routing-fidelity relation, so it deliberately duplicates the reader rather
    than calling one.
    """

    header: dict[str, Any] = {}
    requests: dict[str, dict[str, Any]] = {}
    forwards: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row["row_type"]
            if kind == "header":
                header = row["provenance"]
            elif kind == "request":
                requests[row["request_id"]] = {
                    "input_token_ids": list(row["input_token_ids"]),
                    "output_token_ids": list(row["output_token_ids"]),
                    "output_length": row["output_length"],
                }
            elif kind == "observed-dispatch":
                key = (row["request_id"], row["phase"])
                forwards.setdefault(key, []).append(
                    {
                        "token_index": row["token_index"],
                        "token_id": row["token_id"],
                        "routing": {
                            entry["layer_index"]: tuple(entry["expert_ids"])
                            for entry in row["routing"]
                        },
                    }
                )
            elif kind not in ("kv-event", "footer"):
                raise AssertionError(f"unexpected trace row type {kind!r}")
    for key, rows in forwards.items():
        rows.sort(key=lambda row: row["token_index"])
        if [row["token_index"] for row in rows] != list(range(len(rows))):
            raise AssertionError(f"trace forward rows for {key} are not contiguous")
    return {"header": header, "requests": requests, "forwards": forwards}


def _goal_sends(path: Path) -> dict[tuple[int, int], int]:
    """Parse one executed GOAL artifact into ``(source, dest) -> bytes``."""

    result: dict[tuple[int, int], int] = {}
    rank = -1
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("rank "):
            rank = int(line.split()[1])
        if ": send " not in line:
            continue
        words = line.split()
        pair = (rank, int(words[4]))
        result[pair] = result.get(pair, 0) + int(words[2].removesuffix("b"))
    return result


# ------------------------------------------------------------------- cells ---


def _cell_spec(name: str) -> tuple[int, int]:
    for cell_name, ep_world, bandwidth in CELLS:
        if cell_name == name:
            return ep_world, bandwidth
    if name == CONTROL_CELL:
        return 8, BANDWIDTHS_BPS[2]
    raise SystemExit(f"unknown cell {name!r}")


def _dims(ep_world: int) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        vocab_size=VOCAB_SIZE,
        dtype_bytes=DTYPE_BYTES,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        local_num_experts=NUM_EXPERTS // ep_world,
    )


def _declare_arrivals(bookkeeper: Any) -> dict[str, int]:
    """Append one framework-request object per declared arrival.

    ``join_preplay_arrivals`` is the v1 projection and this study's routing
    authority is a v2 framework trace, so the arrivals are declared here and
    the same mapping is handed to the admission gate and to the reducer.
    """

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
                    object_id=f"sglang-end-to-end-v1:{digest}",
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


def _attribution_json(attribution: Any) -> dict[str, int]:
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


def _independent_request_pairs(
    record: Any,
    raw: dict[str, Any],
    ep_world: int,
) -> dict[str, dict[tuple[str, int, int], int]]:
    """Recompute the per-request directed byte table straight from the trace.

    Returns ``operation_id -> {(request_id, source, destination): bytes}`` with
    the operation identity spelled exactly as ``simllm.traffic`` spells it, so a
    disagreement about which layer or phase a byte belongs to is visible rather
    than absorbed.
    """

    tables: dict[str, dict[tuple[str, int, int], int]] = {}
    for scheduled in record.scheduled:
        if scheduled.num_new_tokens == 0:
            continue
        request_id = scheduled.request_id
        prompt_token_count = len(raw["requests"][request_id]["input_token_ids"])
        if scheduled.phase.value == "prefill":
            end = scheduled.context_length
            start = end - scheduled.num_new_tokens
            rows = raw["forwards"][(request_id, "prefill")]
        else:
            start = scheduled.context_length - scheduled.num_new_tokens - prompt_token_count
            end = start + scheduled.num_new_tokens
            rows = raw["forwards"][(request_id, "decode")]
        if start < 0 or end > len(rows) or start >= end:
            raise AssertionError(
                f"independent slice [{start}, {end}) for {request_id} is outside the trace"
            )
        for row in rows[start:end]:
            for layer in range(NUM_LAYERS):
                experts = row["routing"][layer]
                if len(experts) != TOP_K:
                    raise AssertionError("trace row does not carry the frozen top-k")
                destinations = {expert % ep_world for expert in experts}
                destinations.discard(ENGINE_RANK)
                for destination in destinations:
                    dispatch_id = f"step-{record.step_index}:layer-{layer}:ep-dispatch"
                    combine_id = f"step-{record.step_index}:layer-{layer}:ep-combine"
                    dispatch = tables.setdefault(dispatch_id, {})
                    combine = tables.setdefault(combine_id, {})
                    dispatch_key = (request_id, ENGINE_RANK, destination)
                    combine_key = (request_id, destination, ENGINE_RANK)
                    dispatch[dispatch_key] = dispatch.get(dispatch_key, 0) + VECTOR_BYTES
                    combine[combine_key] = combine.get(combine_key, 0) + VECTOR_BYTES
    return tables


def _library_request_pairs(
    record: Any,
    dims: Any,
    ep_ranks: tuple[int, ...],
    supply: Any,
) -> dict[str, dict[tuple[str, int, int], int]]:
    from simllm.traffic import step_moe_alltoalls

    operations = step_moe_alltoalls(record, dims, ep_ranks, routed_supply=supply)
    tables: dict[str, dict[tuple[str, int, int], int]] = {}
    for operation in operations:
        operation_id = f"step-{record.step_index}:layer-{operation.layer}:ep-{operation.phase}"
        tables[operation_id] = {
            (request_id, source, destination): size
            for request_id, source, destination, size in operation.request_pair_payload_bytes
        }
    return tables


def _check_step_routing(
    record: Any,
    locality: Any,
    raw: dict[str, Any],
    dims: Any,
    ep_ranks: tuple[int, ...],
    supply: Any,
    workdir: Path,
    routing: dict[str, Any],
) -> None:
    """Score E2, E3 and E4 for one simulated step."""

    independent = _independent_request_pairs(record, raw, len(ep_ranks))
    library = _library_request_pairs(record, dims, ep_ranks, supply)
    for operation_id in sorted(set(independent) | set(library)):
        expected = independent.get(operation_id, {})
        actual = library.get(operation_id, {})
        routing["request_pair_rows"] += len(expected)
        if expected != actual:
            routing["request_pair_mismatches"].append(
                {
                    "operation_id": operation_id,
                    "expected_rows": len(expected),
                    "actual_rows": len(actual),
                }
            )
        layer = int(operation_id.split(":")[1].split("-")[1])
        for (request_id, source, _destination), size in expected.items():
            routing["layer_bytes"][str(layer)] += size
            routing["request_bytes"][request_id] += size
            routing["rank_egress_bytes"][str(source)] += size

    for index, operation_ids in enumerate(locality.artifact_operation_ids):
        moe_ids = [
            operation_id for operation_id in operation_ids if ":ep-" in operation_id
        ]
        if not moe_ids:
            continue
        if len(moe_ids) != 1:
            raise AssertionError("one executed artifact carries several MoE operations")
        operation_id = moe_ids[0]
        goal_path = workdir / f"step-{record.step_index:06d}.artifact-{index:04d}.goal"
        if not goal_path.is_file():
            routing["goal_mismatches"].append(
                {"operation_id": operation_id, "reason": "missing artifact"}
            )
            continue
        observed = _goal_sends(goal_path)
        expected_pairs: dict[tuple[int, int], int] = {}
        for (_request_id, source, destination), size in independent.get(
            operation_id, {}
        ).items():
            pair = (source, destination)
            expected_pairs[pair] = expected_pairs.get(pair, 0) + size
        routing["goal_rows"] += len(expected_pairs)
        if observed != expected_pairs:
            routing["goal_mismatches"].append(
                {
                    "operation_id": operation_id,
                    "artifact": goal_path.name,
                    "expected_pairs": len(expected_pairs),
                    "observed_pairs": len(observed),
                }
            )
        dispatch = operation_id.endswith(":ep-dispatch")
        for source, destination in observed:
            owner = source if dispatch else destination
            if owner != ENGINE_RANK:
                routing["ownership_violations"].append(
                    {
                        "operation_id": operation_id,
                        "source": source,
                        "destination": destination,
                    }
                )


def _cell(args: argparse.Namespace, name: str) -> None:
    """Drive one cell in a child interpreter that owns SGLang and torch."""

    import time

    import torch

    from simllm.adapters.sglang import install
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
        attribute_step,
    )
    from simllm.compute import HostInitiationModel, RooflineProvider
    from simllm.core import RequestBookkeeper, framework_request_arrivals
    from simllm.preplay import project_framework_routing
    from simllm.traffic import ExpertPlacementSnapshot, RoutedMoeSupply
    from simllm.workload import AdmissionMode, RequestAdmissionGate

    torch.set_num_threads(TORCH_NUM_THREADS)
    simulated = name != CONTROL_CELL
    ep_world, bandwidth = _cell_spec(name)
    ep_ranks = tuple(range(ep_world))
    cell_dir = args.run_dir / "cells" / name
    cell_dir.mkdir(parents=True, exist_ok=False)

    raw = read_raw_trace(args.routing_trace)
    routed = project_framework_routing(args.routing_trace)
    missing = [request_id for request_id in REQUEST_IDS if request_id not in raw["requests"]]
    if missing:
        raise AssertionError(f"trace does not carry every request of this cell: {missing}")

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
            (layer, expert, expert % ep_world)
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
    dims = _dims(ep_world)
    workdir = cell_dir / "htsim"
    host_model = HostInitiationModel.ideal()
    sink = None
    if simulated:
        sink = HtsimStepSink(
            HtsimStepSinkConfig(
                profile=PROFILE,
                tp_ranks=(0,),
                dims=dims,
                workdir=workdir,
                ep_ranks=ep_ranks,
                linkspeed_bps=bandwidth,
                provider=RooflineProvider(ROOFLINE_EFFICIENCY),
                host_model=host_model,
                routed_moe_supply=supply,
            )
        )

    install()
    configure(
        step_sink=sink,
        host_model=host_model,
        config=SimWorkerConfig(
            mode="virtual",
            efficiency=ROOFLINE_EFFICIENCY,
            step_records_path=str(cell_dir / "steps.jsonl"),
        ),
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
    locality_by_step = (
        {outcome.step_index: outcome for outcome in sink.locality_outcomes}
        if sink is not None
        else {}
    )
    network_by_step = (
        {outcome.step_index: outcome for outcome in sink.outcomes} if sink is not None else {}
    )
    if sink is not None and len(locality_by_step) != len(sink.locality_outcomes):
        raise AssertionError("locality outcomes repeat a step index")

    reducer = HtsimRequestMetricReducer(arrivals)
    step_rows: list[dict[str, Any]] = []
    routing = {
        "token_cells_compared": 0,
        "token_cells_equal": 0,
        "request_pair_rows": 0,
        "request_pair_mismatches": [],
        "goal_rows": 0,
        "goal_mismatches": [],
        "ownership_violations": [],
        "layer_bytes": {str(layer): 0 for layer in range(NUM_LAYERS)},
        "request_bytes": dict.fromkeys(REQUEST_IDS, 0),
        "rank_egress_bytes": {str(rank): 0 for rank in ep_ranks},
    }
    conservation = {
        "intervals": 0,
        "interval_conservation_failures": 0,
        "inactive_component_violations": 0,
        "artifact_partition_failures": 0,
        "makespan_conservation_failures": 0,
        "mid_prompt_extend_rows": 0,
    }
    htsim_invocations = 0
    total_routed_bytes = 0

    for record, result in zip(records, results, strict=True):
        locality = locality_by_step.get(record.step_index)
        network = network_by_step.get(record.step_index)
        for scheduled in record.scheduled:
            if (
                scheduled.phase.value == "prefill"
                and scheduled.context_length != PROMPT_TOKENS
            ):
                conservation["mid_prompt_extend_rows"] += 1
        if locality is not None and (
            locality.compute_service_ps
            + sum(
                composed
                for composed, fabric in zip(
                    locality.composed_phase_service_ps,
                    locality.fabric_phase_service_ps,
                    strict=True,
                )
                if fabric
            )
            != result.step_latency_ps
        ):
            conservation["makespan_conservation_failures"] += 1
        step_attribution = attribute_step(result, locality)
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

        row: dict[str, Any] = {
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
            "attribution": _attribution_json(step_attribution),
            "simulated": locality is not None,
        }
        if locality is not None and network is not None:
            if len(locality.composed_phase_service_ps) != locality.artifact_count:
                conservation["artifact_partition_failures"] += 1
            row.update(
                {
                    "compute_service_ps": locality.compute_service_ps,
                    "artifact_count": locality.artifact_count,
                    "total_directed_bytes": locality.total_directed_bytes,
                    "nvlink_directed_bytes": locality.nvlink_directed_bytes,
                    "fabric_phase_service_ps": list(locality.fabric_phase_service_ps),
                    "composed_phase_service_ps": list(locality.composed_phase_service_ps),
                    "routing_mode": network.routing_mode,
                    "placement_epoch": network.placement_epoch,
                    "quiescent": network.quiescent,
                    "num_flows": network.num_flows,
                    "backend_runs": locality.backend_runs,
                }
            )
            htsim_invocations += locality.backend_runs
            total_routed_bytes += locality.total_directed_bytes
            _check_step_routing(
                record, locality, raw, dims, ep_ranks, supply, workdir, routing
            )
        step_rows.append(row)

    for (request_id, phase), rows in raw["forwards"].items():
        if request_id not in REQUEST_IDS:
            continue
        projected = routed.by_request_id(request_id)
        tokens = projected.prefill_tokens if phase == "prefill" else projected.decode_tokens
        if len(tokens) != len(rows):
            raise AssertionError(f"routed projection lost {phase} tokens for {request_id}")
        for token, row in zip(tokens, rows, strict=True):
            for layer in range(NUM_LAYERS):
                routing["token_cells_compared"] += 1
                if tuple(token.layers[layer].expert_ids) == row["routing"][layer]:
                    routing["token_cells_equal"] += 1

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
                "prompt_tokens": PROMPT_TOKENS,
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
                    None if total is None else _attribution_json(total.ttft_attribution)
                ),
                "decode_attribution": (
                    None if total is None else _attribution_json(total.decode_attribution)
                ),
            }
        )

    payload = {
        "name": name,
        "simulated": simulated,
        "ep_world": ep_world,
        "bandwidth_bps": bandwidth if simulated else None,
        "wall_seconds": round(wall_seconds, 2),
        "scheduler_steps": batch_runs,
        "idle_steps": idle_steps,
        "step_records": len(records),
        "simulated_steps": 0 if sink is None else len(sink.locality_outcomes),
        "network_outcomes": 0 if sink is None else len(sink.outcomes),
        "htsim_invocations": htsim_invocations,
        "total_routed_bytes": total_routed_bytes,
        "identity": {
            "driver_pid": os.getpid(),
            "sink_is_worker_sink": sink is not None and worker.step_sink is sink,
            "sink_installed": sink is not None,
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
        "routing": routing,
        "conservation": conservation,
    }
    (cell_dir / "cell.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------- analysis ---


def _read_cells(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    cells = {}
    for name in [entry[0] for entry in CELLS] + [CONTROL_CELL]:
        path = args.run_dir / "cells" / name / "cell.json"
        cells[name] = json.loads(path.read_text(encoding="utf-8"))
    return cells


def _composition_key(step: dict[str, Any]) -> tuple[tuple[str, str, int, int], ...]:
    """Identify a step by what it schedules rather than by its ordinal.

    A faster fabric reaches the next declared arrival after a different number
    of steps, so the same step index carries different work in different cells.
    Every cross-cell relation is therefore evaluated over steps that schedule
    exactly the same requests, phases, token counts and context lengths.
    """

    return tuple(
        (
            scheduled["request_id"],
            scheduled["phase"],
            scheduled["num_new_tokens"],
            scheduled["context_length"],
        )
        for scheduled in step["scheduled"]
    )


def _steps_by_composition(cell: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    rows: dict[Any, dict[str, Any]] = {}
    for step in cell["steps"]:
        if not step.get("simulated"):
            continue
        key = _composition_key(step)
        if key in rows:
            raise AssertionError("two simulated steps share a scheduling composition")
        rows[key] = step
    return rows


def _independent_request_metrics(cell: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Recompute TTFT, TPOT and both attributions with the standard library.

    Reads only the per-step rows and the declared arrivals. Every scheduled row
    samples a token under guard G3, so a request's token times are the
    completion times of the steps that scheduled it. The attribution rule is the
    frozen one: the gap since the request's last accounted point is queue time,
    an artifact with nonzero fabric service is collective time, and an artifact
    with zero fabric service is kernel time.
    """

    steps = [step for step in cell["steps"] if step.get("simulated")]
    steps.sort(key=lambda step: step["step_index"])
    arrivals = {row["request_id"]: row["arrival_ps"] for row in cell["requests"]}
    state: dict[str, dict[str, Any]] = {}
    for step in steps:
        kernel_ps = 0
        collective_ps = 0
        for composed, fabric in zip(
            step["composed_phase_service_ps"],
            step["fabric_phase_service_ps"],
            strict=True,
        ):
            if fabric:
                collective_ps += composed
            else:
                kernel_ps += composed
        for scheduled in step["scheduled"]:
            request_id = scheduled["request_id"]
            row = state.setdefault(
                request_id,
                {
                    "accounted_through_ps": arrivals[request_id],
                    "first_token_at_ps": None,
                    "last_token_at_ps": None,
                    "token_count": 0,
                    "ttft_kernel_ps": 0,
                    "ttft_collective_ps": 0,
                    "ttft_queue_ps": 0,
                    "decode_kernel_ps": 0,
                    "decode_collective_ps": 0,
                    "decode_queue_ps": 0,
                },
            )
            queue_ps = step["virtual_time_ps"] - row["accounted_through_ps"]
            row["accounted_through_ps"] = step["completed_at_ps"]
            row["token_count"] += 1
            if row["first_token_at_ps"] is None:
                row["first_token_at_ps"] = step["completed_at_ps"]
                row["ttft_kernel_ps"] = kernel_ps
                row["ttft_collective_ps"] = collective_ps
                row["ttft_queue_ps"] = queue_ps
            else:
                row["decode_kernel_ps"] += kernel_ps
                row["decode_collective_ps"] += collective_ps
                row["decode_queue_ps"] += queue_ps
            row["last_token_at_ps"] = step["completed_at_ps"]
    metrics: dict[str, dict[str, Any]] = {}
    for request_id, row in state.items():
        span = row["last_token_at_ps"] - row["first_token_at_ps"]
        tpot = Fraction(span, row["token_count"] - 1) if row["token_count"] > 1 else None
        metrics[request_id] = {
            "token_count": row["token_count"],
            "first_token_at_ps": row["first_token_at_ps"],
            "last_token_at_ps": row["last_token_at_ps"],
            "ttft_ps": row["first_token_at_ps"] - arrivals[request_id],
            "tpot": tpot,
            "ttft_attribution": {
                "queue_ps": row["ttft_queue_ps"],
                "kernel_ps": row["ttft_kernel_ps"],
                "collective_ps": row["ttft_collective_ps"],
            },
            "decode_attribution": {
                "queue_ps": row["decode_queue_ps"],
                "kernel_ps": row["decode_kernel_ps"],
                "collective_ps": row["decode_collective_ps"],
            },
        }
    return metrics


def _score_e1(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    compared = 0
    failures: list[dict[str, Any]] = []
    for name, cell in cells.items():
        if not cell["simulated"]:
            continue
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
            if expected["token_count"] != row["token_count"]:
                mismatches.append("token_count")
            if expected["first_token_at_ps"] != row["first_token_at_ps"]:
                mismatches.append("first_token_at_ps")
            if expected["last_token_at_ps"] != row["last_token_at_ps"]:
                mismatches.append("last_token_at_ps")
            if expected["ttft_ps"] != row["ttft_ps"]:
                mismatches.append("ttft_ps")
            if expected["tpot"] != observed_tpot:
                mismatches.append("tpot")
            for field, attribution in (
                ("ttft_attribution", expected["ttft_attribution"]),
                ("decode_attribution", expected["decode_attribution"]),
            ):
                published = row[field]
                for component, value in attribution.items():
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


def _linearity(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_bandwidth = {
        cells[name]["bandwidth_bps"]: _steps_by_composition(cells[name])
        for name in ("ep8-100g", "ep8-200g", "ep8-400g")
    }
    shared = set.intersection(*(set(rows) for rows in by_bandwidth.values()))
    compared = 0
    failures = []
    worst = 0.0
    for key in sorted(shared, key=lambda value: (len(value), value)):
        services = {
            bandwidth: rows[key]["fabric_phase_service_ps"]
            for bandwidth, rows in by_bandwidth.items()
        }
        lengths = {len(value) for value in services.values()}
        if len(lengths) != 1:
            failures.append({"composition": list(key), "reason": "artifact count differs"})
            continue
        for index in range(next(iter(lengths))):
            slow = services[100_000_000_000][index]
            middle = services[200_000_000_000][index]
            fast = services[400_000_000_000][index]
            if not slow and not middle and not fast:
                continue
            compared += 1
            residual = abs(slow - 3 * middle + 2 * fast)
            tolerance = max(B1_ABSOLUTE_TOLERANCE_PS, B1_RELATIVE_TOLERANCE * slow)
            worst = max(worst, residual / slow if slow else 0.0)
            if residual > tolerance:
                failures.append(
                    {
                        "composition": list(key),
                        "artifact": index,
                        "s100": slow,
                        "s200": middle,
                        "s400": fast,
                        "residual_ps": residual,
                    }
                )
    return {
        "matched_steps": len(shared),
        "compared": compared,
        "failures": failures[:20],
        "failure_count": len(failures),
        "worst_relative_residual": worst,
        "passed": compared > 0 and not failures,
    }


def _halving_response(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = (("ep8-400g", "ep8-200g"), ("ep8-200g", "ep8-100g"))
    rounds = []
    for fast_name, slow_name in pairs:
        fast_steps = _steps_by_composition(cells[fast_name])
        slow_steps = _steps_by_composition(cells[slow_name])
        shared = set(fast_steps) & set(slow_steps)
        ratios = [
            slow_steps[key]["step_latency_ps"] / fast_steps[key]["step_latency_ps"]
            for key in shared
        ]
        violations = [ratio for ratio in ratios if not 1.0 < ratio < 2.0]
        rounds.append(
            {
                "from": fast_name,
                "to": slow_name,
                "matched_steps": len(shared),
                "ratio_min": min(ratios, default=None),
                "ratio_max": max(ratios, default=None),
                "violations": violations,
                "passed": bool(ratios) and not violations,
            }
        )
    return {"passed": all(entry["passed"] for entry in rounds), "halvings": rounds}


def _closed_loop(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ladder = ("ep8-400g", "ep8-200g", "ep8-100g")
    steps = [cells[name]["scheduler_steps"] for name in ladder]
    monotone = all(left >= right for left, right in pairwise(steps))
    strict = any(left > right for left, right in pairwise(steps))
    token_counts = {
        name: sorted(row["token_count"] for row in cells[name]["requests"])
        for name in ladder
    }
    identical_tokens = (
        len({tuple(value) for value in token_counts.values()}) == 1
        and token_counts[ladder[0]] == [MAX_NEW_TOKENS] * len(REQUEST_IDS)
    )
    compositions = {
        name: [_composition_key(step) for step in cells[name]["steps"]] for name in ladder
    }
    return {
        "passed": bool(monotone and strict and identical_tokens),
        "scheduler_steps": dict(zip(ladder, steps, strict=True)),
        "monotone": monotone,
        "strict_somewhere": strict,
        "identical_token_counts": identical_tokens,
        "distinct_step_sequences": len({tuple(value) for value in compositions.values()}),
        "control_steps": cells[CONTROL_CELL]["scheduler_steps"],
    }


def _compute_response(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def mean_compute(name: str) -> float | None:
        values = [
            step["compute_service_ps"]
            for step in cells[name]["steps"]
            if step.get("simulated")
        ]
        return sum(values) / len(values) if values else None

    ep8 = mean_compute("ep8-400g")
    ep4 = mean_compute("ep4-400g")
    ratio = None if not ep8 or ep4 is None else ep4 / ep8
    return {
        "passed": ratio is not None
        and B4_COMPUTE_RATIO_FLOOR <= ratio <= B4_COMPUTE_RATIO_CEILING,
        "ratio": ratio,
        "ep8_mean_ps": ep8,
        "ep4_mean_ps": ep4,
        "prediction": 857_217_024 / 555_227_136,
    }


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    cells = _read_cells(args)
    simulated = {name: cell for name, cell in cells.items() if cell["simulated"]}
    exact: dict[str, dict[str, Any]] = {}
    fatal: dict[str, dict[str, Any]] = {}

    def score(name: str, ok: bool, detail: dict[str, Any]) -> None:
        exact[name] = {"passed": bool(ok), **detail}

    def guard(name: str, ok: bool, detail: dict[str, Any]) -> None:
        fatal[name] = {"held": bool(ok), **detail}

    provenance = check_trace_provenance(args.routing_trace)
    guard(
        "G1",
        True,
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
            for cell in simulated.values()
        ),
        {
            name: {
                "driver_pid": cell["identity"]["driver_pid"],
                "scheduler_steps": cell["scheduler_steps"],
                "step_records": cell["step_records"],
                "sink_locality_outcomes": cell["simulated_steps"],
                "sink_network_outcomes": cell["network_outcomes"],
            }
            for name, cell in simulated.items()
        },
    )
    guard(
        "G3",
        all(
            cell["conservation"]["mid_prompt_extend_rows"] == 0
            and all(row["retraction_count"] == 0 for row in cell["requests"])
            for cell in cells.values()
        ),
        {
            "mid_prompt_extend_rows": sum(
                cell["conservation"]["mid_prompt_extend_rows"] for cell in cells.values()
            ),
            "retractions": sum(
                row["retraction_count"] or 0
                for cell in cells.values()
                for row in cell["requests"]
            ),
        },
    )
    intervals = sum(cell["conservation"]["intervals"] for cell in simulated.values())
    guard(
        "G4",
        intervals > 0
        and all(
            cell["conservation"]["makespan_conservation_failures"] == 0
            and cell["conservation"]["interval_conservation_failures"] == 0
            and cell["conservation"]["inactive_component_violations"] == 0
            and cell["conservation"]["artifact_partition_failures"] == 0
            and all(
                step["completed_at_ps"] == step["virtual_time_ps"] + step["step_latency_ps"]
                for step in cell["steps"]
            )
            and all(row["ttft_ps"] > 0 for row in cell["requests"])
            for cell in simulated.values()
        ),
        {"intervals": intervals},
    )
    guard(
        "G5",
        all(
            step.get("routing_mode") == "captured"
            and step.get("placement_epoch") == 0
            and step.get("quiescent") is True
            and step.get("nvlink_directed_bytes") == 0
            for cell in simulated.values()
            for step in cell["steps"]
            if step.get("simulated")
        ),
        {"simulated_steps": sum(cell["simulated_steps"] for cell in simulated.values())},
    )
    guard(
        "G6",
        all(
            row["finish_reason"] == "length"
            and row["framework_output_tokens"] == MAX_NEW_TOKENS
            and (not cell["simulated"] or row["token_count"] == MAX_NEW_TOKENS)
            for cell in cells.values()
            for row in cell["requests"]
        ),
        {"requests": sum(len(cell["requests"]) for cell in cells.values())},
    )
    ep8_bytes = cells["ep8-400g"]["total_routed_bytes"]
    ep4_bytes = cells["ep4-400g"]["total_routed_bytes"]
    guard(
        "G7",
        0 < ep4_bytes <= ep8_bytes,
        {
            "ep8_bytes": ep8_bytes,
            "ep4_bytes": ep4_bytes,
            "ratio": ep4_bytes / ep8_bytes if ep8_bytes else None,
        },
    )

    fast = cells["ep8-400g"]
    prefill = [
        step
        for step in fast["steps"]
        if step.get("simulated")
        and any(scheduled["phase"] == "prefill" for scheduled in step["scheduled"])
    ]
    decode = [
        step
        for step in fast["steps"]
        if step.get("simulated")
        and step["scheduled"]
        and all(scheduled["phase"] == "decode" for scheduled in step["scheduled"])
    ]
    ep8_compute = [
        step["compute_service_ps"] for step in fast["steps"] if step.get("simulated")
    ]
    step0_bytes = prefill[0]["total_directed_bytes"] if prefill else 0
    step0_fabric = sum(prefill[0]["fabric_phase_service_ps"]) if prefill else 0
    decode_rates = [
        PS_PER_SECOND / (row["tpot_numerator"] / row["tpot_denominator"])
        for cell in simulated.values()
        for row in cell["requests"]
        if row["tpot_numerator"] is not None
    ]
    bounds = {
        "S1": {
            "value": step0_bytes,
            "floor": S1_MOE_BYTES_FLOOR,
            "ceiling": S1_MOE_BYTES_CEILING,
            "held": S1_MOE_BYTES_FLOOR <= step0_bytes <= S1_MOE_BYTES_CEILING,
        },
        "S2": {
            "value": step0_fabric,
            "floor": S2_SERIALIZATION_FLOOR_PS,
            "ceiling": None,
            "held": step0_fabric >= S2_SERIALIZATION_FLOOR_PS,
            "note": (
                "floor only: the fluid model adds a per-artifact propagation term "
                "on top of serialization, so the S2 ceiling bounds the "
                "serialization component and not the artifact service"
            ),
        },
        "S3": {
            "values": [min(ep8_compute, default=0), max(ep8_compute, default=0)],
            "floor": S3_COMPUTE_FLOOR_PS,
            "ceiling": S3_COMPUTE_CEILING_PS,
            "held": bool(ep8_compute)
            and all(
                S3_COMPUTE_FLOOR_PS <= value <= S3_COMPUTE_CEILING_PS
                for value in ep8_compute
            ),
        },
        "S4": {
            "values": [step["step_latency_ps"] for step in prefill],
            "floor": S4_PREFILL_FLOOR_PS,
            "ceiling": S4_PREFILL_CEILING_PS,
            "held": bool(prefill)
            and all(
                S4_PREFILL_FLOOR_PS <= step["step_latency_ps"] <= S4_PREFILL_CEILING_PS
                for step in prefill
            ),
        },
        "S5": {
            "min": min((step["step_latency_ps"] for step in decode), default=0),
            "max": max((step["step_latency_ps"] for step in decode), default=0),
            "floor": S5_DECODE_FLOOR_PS,
            "ceiling": S5_DECODE_CEILING_PS,
            "held": bool(decode)
            and all(
                S5_DECODE_FLOOR_PS <= step["step_latency_ps"] <= S5_DECODE_CEILING_PS
                for step in decode
            ),
        },
    }
    for bound_name, detail in bounds.items():
        guard(bound_name, detail["held"], detail)

    e1 = _score_e1(simulated)
    score("E1", e1["passed"], e1)
    pair_rows = sum(cell["routing"]["request_pair_rows"] for cell in simulated.values())
    pair_bad = sum(
        len(cell["routing"]["request_pair_mismatches"]) for cell in simulated.values()
    )
    score(
        "E2",
        pair_rows > 0 and pair_bad == 0,
        {
            "rows": pair_rows,
            "mismatched_operations": pair_bad,
            "layer_bytes": {
                name: cell["routing"]["layer_bytes"] for name, cell in simulated.items()
            },
            "request_bytes": {
                name: cell["routing"]["request_bytes"] for name, cell in simulated.items()
            },
        },
    )
    goal_rows = sum(cell["routing"]["goal_rows"] for cell in simulated.values())
    goal_bad = sum(len(cell["routing"]["goal_mismatches"]) for cell in simulated.values())
    score("E3", goal_rows > 0 and goal_bad == 0, {"rows": goal_rows, "mismatches": goal_bad})
    ownership_bad = sum(
        len(cell["routing"]["ownership_violations"]) for cell in simulated.values()
    )
    score("E4", ownership_bad == 0, {"violations": ownership_bad})
    token_cells = sum(cell["routing"]["token_cells_compared"] for cell in cells.values())
    token_equal = sum(cell["routing"]["token_cells_equal"] for cell in cells.values())
    score(
        "E5",
        token_cells > 0 and token_cells == token_equal,
        {"compared": token_cells, "equal": token_equal},
    )

    behavioral: dict[str, dict[str, Any]] = {
        "B1": _linearity(cells),
        "B2": _halving_response(cells),
        "B3": _closed_loop(cells),
        "B4": _compute_response(cells),
    }

    void = [name for name, detail in fatal.items() if not detail["held"]]
    summary = {
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
        "scale": {
            name: {
                "wall_seconds": cell["wall_seconds"],
                "scheduler_steps": cell["scheduler_steps"],
                "htsim_invocations": cell["htsim_invocations"],
                "total_routed_bytes": cell["total_routed_bytes"],
                "peak_rank_egress_bytes": max(
                    (int(value) for value in cell["routing"]["rank_egress_bytes"].values()),
                    default=0,
                ),
            }
            for name, cell in cells.items()
        },
        "cells": {
            name: {
                "simulated": cell["simulated"],
                "scheduler_steps": cell["scheduler_steps"],
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
                        "token_count": row["token_count"],
                        "ttft_attribution": row["ttft_attribution"],
                        "decode_attribution": row["decode_attribution"],
                    }
                    for row in cell["requests"]
                ],
            }
            for name, cell in cells.items()
        },
        "diagnostics": {
            "decode_rate_tokens_per_second": {
                "min": min(decode_rates, default=None),
                "max": max(decode_rates, default=None),
            },
            "control_cell": {
                "scheduler_steps": cells[CONTROL_CELL]["scheduler_steps"],
                "reads_link_rate": False,
            },
            "step_shapes": {
                name: cell["step_shapes"] for name, cell in cells.items()
            },
        },
    }
    if len(exact) != EXPECTED_EXACT_RELATIONS:
        raise AssertionError("exact relation denominator changed")
    if len(behavioral) != EXPECTED_BEHAVIORAL_RELATIONS:
        raise AssertionError("behavioral relation denominator changed")
    return summary


FREEZE_COMMIT = "8907c53"


# ------------------------------------------------------------------ driver ---


def _run_child(args: argparse.Namespace, mode: str) -> None:
    command = [
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
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["SIMLLM_HTSIM_RNIC"] = str(args.htsim_rnic)
    log_path = args.run_dir / "logs" / f"{mode.replace(':', '-')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command, check=False, env=environment, stdout=handle, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        raise SystemExit(
            f"child stage {mode!r} failed with code {completed.returncode}; "
            f"see {log_path.name}"
        )


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=False)
    for name in [entry[0] for entry in CELLS] + [CONTROL_CELL]:
        _run_child(args, f"cell:{name}")
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
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--internal", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return 0
    if args.internal and args.internal.startswith("cell:"):
        import sys

        if str(REPOSITORY_ROOT) not in sys.path:
            sys.path.insert(0, str(REPOSITORY_ROOT))
        _cell(args, args.internal.removeprefix("cell:"))
        return 0
    if args.internal == "summarize":
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
                "behavioral": [summary["behavioral_passed"], summary["behavioral_total"]],
                "scale": summary["scale"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
