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
from pathlib import Path

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--routing-trace", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.check_only:
        check_only(args)
        return 0
    raise SystemExit(
        "the measuring half of this study is not implemented in the freeze "
        "commit; run with --check-only"
    )


if __name__ == "__main__":
    raise SystemExit(main())
