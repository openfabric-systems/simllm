"""Run the frozen end-to-end per-request replay study.

This module currently carries the frozen constants and the check-only input
gate. The implementation of the study lands in a later commit, after this
expectations-only freeze.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / MODEL_REVISION
)

#: geometry of the captured model, frozen in expectations.md
NUM_LAYERS = 24
HIDDEN_SIZE = 1024
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

BANDWIDTHS_BPS = (100_000_000_000, 200_000_000_000, 400_000_000_000)

#: case name to expert-parallel world and link rate
CELLS = (
    ("a-ep8-400g", "a", 8, 400_000_000_000),
    ("a-ep8-200g", "a", 8, 200_000_000_000),
    ("a-ep8-100g", "a", 8, 100_000_000_000),
    ("a-ep4-400g", "a", 4, 400_000_000_000),
    ("b-ep8-400g", "b", 8, 400_000_000_000),
)

#: physical sanity bounds from expectations.md, all in picoseconds or bytes
CASE_A_PROMPT_TOKENS = 54
S1_MOE_BYTES_FLOOR = CASE_A_PROMPT_TOKENS * NUM_LAYERS * 2 * 1 * VECTOR_BYTES
S1_MOE_BYTES_CEILING = CASE_A_PROMPT_TOKENS * NUM_LAYERS * 2 * 7 * VECTOR_BYTES
S3_COMPUTE_FLOOR_PS = 60_000_000
S3_COMPUTE_CEILING_PS = 200_000_000
S4_PREFILL_FLOOR_PS = 270_000_000
S4_PREFILL_CEILING_PS = 3_000_000_000
S5_DECODE_FLOOR_PS = 150_000_000
S5_DECODE_CEILING_PS = 600_000_000
S6_COMPUTE_SPREAD = 0.10
S7_DECODE_RATE_FLOOR = 1_000.0
S7_DECODE_RATE_CEILING = 20_000.0
C5_4_COMPUTE_RATIO_FLOOR = 1.35
C5_4_COMPUTE_RATIO_CEILING = 1.75
C5_1_ABSOLUTE_TOLERANCE_PS = 1_000
C5_1_RELATIVE_TOLERANCE = 1e-6

#: scored relation counts, split by evidence class and never summed
EXPECTED_EXACT_ORACLE_RELATIONS = 13
EXPECTED_BEHAVIORAL_RELATIONS = 4


def check_only(args: argparse.Namespace) -> None:
    """Validate every frozen input without producing an artifact."""

    model = args.cache_dir / MODEL_RELATIVE_PATH
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {model}")
    if importlib.metadata.version("vllm") != "0.26.0":
        raise SystemExit("this study requires vLLM 0.26.0")
    if not args.htsim_rnic.is_file() or not args.htsim_rnic.stat().st_mode & 0o111:
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    txt2bin = os.environ.get("SIMLLM_TXT2BIN")
    if txt2bin is not None:
        converter = Path(txt2bin)
        if not converter.is_file() or not converter.stat().st_mode & 0o111:
            raise SystemExit(f"SIMLLM_TXT2BIN is not an executable: {converter}")
    if VECTOR_BYTES != 2048:
        raise AssertionError("routed hidden-vector size changed")
    if NUM_EXPERTS % 8 or NUM_EXPERTS % 4:
        raise AssertionError("expert count no longer divides both frozen EP worlds")
    if S1_MOE_BYTES_FLOOR != 5_308_416 or S1_MOE_BYTES_CEILING != 37_158_912:
        raise AssertionError("case A step-0 routed-byte bounds changed")
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
    if {bandwidth for name, case, ep, bandwidth in CELLS if case == "a" and ep == 8} != set(
        BANDWIDTHS_BPS
    ):
        raise AssertionError("the three-point linearity test lost a bandwidth cell")
    if S4_PREFILL_FLOOR_PS >= S4_PREFILL_CEILING_PS:
        raise AssertionError("prefill makespan bounds are inverted")
    if S5_DECODE_FLOOR_PS >= S5_DECODE_CEILING_PS:
        raise AssertionError("decode makespan bounds are inverted")
    if S3_COMPUTE_FLOOR_PS >= S3_COMPUTE_CEILING_PS:
        raise AssertionError("compute service bounds are inverted")
    if not 0.0 < ROOFLINE_EFFICIENCY <= 1.0:
        raise AssertionError("roofline derate left its valid range")
    if EXPECTED_EXACT_ORACLE_RELATIONS != 3 + 4 + 5 + 1:
        raise AssertionError("exact-oracle relation denominator changed")
    if EXPECTED_BEHAVIORAL_RELATIONS != 4:
        raise AssertionError("behavioral relation denominator changed")
    print(
        f"check-only run-dir={args.run_dir}; validated the frozen end-to-end "
        "inputs and produced no artifacts"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_only(args)
    if args.check_only:
        return
    raise SystemExit(
        "the end-to-end study implementation lands after this expectations freeze"
    )


if __name__ == "__main__":
    main()
