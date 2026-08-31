#!/usr/bin/env python3
"""Run the frozen binary-free frontier comparison study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn

from simllm.deploy import (
    BudgetSpec,
    DeploymentCandidate,
    EnvelopeSpec,
    EstimatorInputs,
    FabricSpec,
    ModelRef,
    ModelWork,
    PoolSpec,
    SlaSpec,
    WorkloadPoint,
    candidate_key,
    candidate_to_json,
    check_feasibility,
    estimate_decode_step,
    estimate_prefill_request,
    estimate_stamp_to_json,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.md"
EXTERNAL_DIGESTS_PATH = STUDY_DIR / "external.sha256"
EXTERNAL_FRONTIER_PATH = STUDY_DIR / "external" / "disagg_pareto.csv"
INVENTORY_PATH = (
    REPOSITORY_ROOT
    / "offline"
    / "calibration"
    / "model-inventories"
    / "c8832ba8ba21e49517b6b74e89554c2abdb0d9e76530f647a7849f3f8448ec56.json"
)
SUITE_PATH = (
    REPOSITORY_ROOT
    / "offline"
    / "calibration"
    / "suites"
    / "qwen3-32b-fp8-text-v1-frameworks-2026-08-28"
    / "suite.json"
)

RESULT_SCHEMA = "simllm-frontier-comparison-study-v1"
EXPECTATIONS_COMMIT = "83bb28176c2873d94b1d94872ddbfdb96f7270fa"
EXPECTATIONS_SHA256 = "68b9263e3c236de591be12711c1f4c50707195991e9512f23910674026b70d2a"
EXTERNAL_ARCHIVE_MANIFEST_SHA256 = (
    "645b0b206f5af38ec1cc22cbef08d8cb7685af28b02f4e4c4c4480d84e080f5d"
)
INVENTORY_SHA256 = "c8832ba8ba21e49517b6b74e89554c2abdb0d9e76530f647a7849f3f8448ec56"
SUITE_SHA256 = "f0830d3692029dca5464af6932f273d7147258d72edaf5986aada37a0ba25435"
MODEL_REVISION = "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
ARCHITECTURE_LITERALS = {
    "layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_heads": 64,
    "num_kv_heads": 8,
    "head_size": 128,
    "vocab_size": 151936,
}

H200_PEAK_FP8_FLOPS_PER_SECOND = 1_979_000_000_000_000
H200_HBM_BYTES_PER_SECOND = 4_800_000_000_000
H200_CAPACITY_BYTES = 141_000_000_000
EFFICIENCY_ARMS = (Decimal("0.6"), Decimal("0.8"), Decimal("1.0"))
TP_WIDTHS = (2, 4, 8)
BATCH_LADDER = (1, 2, 4, 8, 9, 16, 20, 26, 32, 48, 56, 64, 96, 112, 128)
REFERENCE_TP = 4
GPU_BUDGET = 32
UNCACHED_PROMPT_TOKENS = 3500
OUTPUT_TOKENS = 500
AVERAGE_DECODE_CONTEXT_TOKENS = 4250
TPOT_TARGET_PS = 10_000_000_000
TTFT_TARGET_PS = 300_000_000_000
PICOSECONDS_PER_SECOND = 1_000_000_000_000

EXTERNAL_TOOL = "aiconfigurator"
EXTERNAL_VERSION = "0.11.0"
EXTERNAL_DATABASE = "h200_sxm trtllm 1.3.0rc10"
EXTERNAL_BEST_TPOT_PS = 9_179_000_000
EXTERNAL_BEST_TTFT_PS = 196_423_000_000


class ProcessCreationBlocked(RuntimeError):
    """Raised if the pricing lane attempts to start a process."""


class ProcessGuard(AbstractContextManager["ProcessGuard"]):
    """Intercept process creation only while the estimator sweep runs."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._popen: Any = None
        self._posix_spawn: Any = None

    def _blocked(self, *args: object, **kwargs: object) -> NoReturn:
        detail = f"args={args!r}, kwargs={kwargs!r}"
        self.attempts.append(detail)
        raise ProcessCreationBlocked(f"process creation attempted: {detail}")

    def __enter__(self) -> ProcessGuard:  # noqa: PYI034
        self._popen = subprocess.Popen
        subprocess.Popen = self._blocked  # type: ignore[assignment]
        if hasattr(os, "posix_spawn"):
            self._posix_spawn = os.posix_spawn
            os.posix_spawn = self._blocked  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        subprocess.Popen = self._popen
        if self._posix_spawn is not None:
            os.posix_spawn = self._posix_spawn  # type: ignore[assignment]
        return False


@dataclass(frozen=True, slots=True)
class WorkDerivation:
    """Inventory-derived logical work at the frozen workload point."""

    static_parameter_bytes: int
    decode_total_flops_per_batch_item: int
    decode_kv_bytes: int
    prefill_total_flops_per_request: int
    prefill_kv_bytes: int
    decode_attention_score_flops_per_context_token: int
    decode_attention_score_flops_per_token_pair: int
    prefill_attention_score_flops_per_token_pair: int
    kv_bytes_per_context_token: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name}: expected a JSON object")
    return value


def _fraction(value: str) -> Fraction:
    return Fraction(Decimal(value))


def _fraction_json(value: Fraction) -> dict[str, int | float]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _projection(case: dict[str, Any], family_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in case["kernel_projections"]
        if item["family_id"] == family_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"case {case['case_id']!r}: expected one {family_id!r} projection"
        )
    return matches[0]


def _case(inventory: dict[str, Any], case_id: str) -> dict[str, Any]:
    matches = [case for case in inventory["cases"] if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"inventory: expected one case {case_id!r}")
    return matches[0]


def derive_work(inventory: dict[str, Any]) -> WorkDerivation:
    """Derive reference-TP ModelWork coefficients from inventory rows."""

    decode = _case(inventory, "db-train-b1-c2048")
    prefill = _case(inventory, "cp-train-r4-t128")
    decode_context = 2048
    prefill_tokens = 128
    prefill_requests = 4

    static_parameter_bytes = sum(
        item["aggregate_hbm_bytes"]
        for item in decode["kernel_projections"]
        if item["family_id"] != "kv_read"
    )
    decode_fixed_flops = sum(
        item["aggregate_flops"]
        for item in decode["kernel_projections"]
        if item["family_id"] != "attn_score"
    )
    decode_score_flops = _projection(decode, "attn_score")["aggregate_flops"]
    score_per_context_token = decode_score_flops // decode_context
    decode_score_per_pair = decode_score_flops // (decode_context - 1)
    kv_bytes_per_token = (
        _projection(decode, "kv_read")["aggregate_hbm_bytes"] // decode_context
    )
    decode_flops = (
        decode_fixed_flops
        + score_per_context_token * AVERAGE_DECODE_CONTEXT_TOKENS
    )
    decode_kv_bytes = kv_bytes_per_token * AVERAGE_DECODE_CONTEXT_TOKENS

    linear_prefill_flops = sum(
        item["aggregate_flops"]
        for item in prefill["kernel_projections"]
        if item["family_id"] in {"attn_gemm", "mlp_gemm"}
    ) // prefill_tokens
    prefill_score_per_pair = (
        _projection(prefill, "attn_score")["aggregate_flops"]
        // (prefill_tokens * prefill_tokens)
    )
    lm_head_per_request = (
        _projection(prefill, "lm_head")["aggregate_flops"] // prefill_requests
    )
    prefill_flops = (
        linear_prefill_flops * UNCACHED_PROMPT_TOKENS
        + prefill_score_per_pair * UNCACHED_PROMPT_TOKENS**2
        + lm_head_per_request
    )
    prefill_kv_bytes = kv_bytes_per_token * UNCACHED_PROMPT_TOKENS
    return WorkDerivation(
        static_parameter_bytes=static_parameter_bytes,
        decode_total_flops_per_batch_item=decode_flops,
        decode_kv_bytes=decode_kv_bytes,
        prefill_total_flops_per_request=prefill_flops,
        prefill_kv_bytes=prefill_kv_bytes,
        decode_attention_score_flops_per_context_token=score_per_context_token,
        decode_attention_score_flops_per_token_pair=decode_score_per_pair,
        prefill_attention_score_flops_per_token_pair=prefill_score_per_pair,
        kv_bytes_per_context_token=kv_bytes_per_token,
    )


def _model_work(
    derivation: WorkDerivation,
    *,
    phase: str,
    tensor_parallel: int,
) -> ModelWork:
    if phase == "decode":
        flops = derivation.decode_total_flops_per_batch_item
        kv_bytes = derivation.decode_kv_bytes
    elif phase == "prefill":
        flops = derivation.prefill_total_flops_per_request
        kv_bytes = derivation.prefill_kv_bytes
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return ModelWork(
        kernel_name=f"qwen3-32b-fp8-{phase}-tp{tensor_parallel}",
        flops_per_batch_item=flops // tensor_parallel,
        static_logical_hbm_bytes=(
            derivation.static_parameter_bytes // tensor_parallel
        ),
        dynamic_hbm_bytes_per_batch_item=kv_bytes // tensor_parallel,
        logical_collective_bytes_per_gpu_per_batch_item=0,
        inventory_sha256=INVENTORY_SHA256,
        source=(
            "Qwen3-32B inventory whole-model logical work divided by the "
            "tensor-parallel width per rank; FP8 weights use one byte per "
            "parameter; KV bytes use the frozen workload context"
        ),
    )


def _envelope(efficiency: Decimal) -> EnvelopeSpec:
    return EnvelopeSpec(
        device="h200",
        peak_flops_per_second=H200_PEAK_FP8_FLOPS_PER_SECOND,
        hbm_bytes_per_second=H200_HBM_BYTES_PER_SECOND,
        efficiency=float(efficiency),
        source=(
            "DECLARED NVIDIA H200 SXM public specification: dense FP8 is "
            "1.979e15 flop/s after removing the documented sparsity factor; "
            "HBM is 4.8e12 byte/s"
        ),
    )


def make_candidate(
    *,
    prefill_tp: int,
    prefill_workers: int,
    decode_tp: int,
    decode_workers: int,
    decode_batch: int,
    candidate_id: str | None = None,
) -> DeploymentCandidate:
    """Build one exact disaggregated candidate under the frozen workload."""

    identity = candidate_id or (
        f"p-tp{prefill_tp}-w{prefill_workers}-"
        f"d-tp{decode_tp}-w{decode_workers}-b{decode_batch}"
    )
    return DeploymentCandidate(
        candidate_id=identity,
        model=ModelRef(
            framework="vllm",
            model_id="Qwen/Qwen3-32B-FP8",
            inventory_sha256=INVENTORY_SHA256,
        ),
        pools=(
            PoolSpec(
                role="prefill",
                engines=prefill_workers,
                gpus_per_engine=prefill_tp,
                tensor_parallel=prefill_tp,
                pipeline_parallel=1,
                expert_parallel=1,
                data_parallel=1,
                device="h200",
            ),
            PoolSpec(
                role="decode",
                engines=decode_workers,
                gpus_per_engine=decode_tp,
                tensor_parallel=decode_tp,
                pipeline_parallel=1,
                expert_parallel=1,
                data_parallel=1,
                device="h200",
            ),
        ),
        fabric=FabricSpec(
            inter_node_bits_per_second=400_000_000_000,
            intra_node_bytes_per_second=900_000_000_000,
        ),
        workload=WorkloadPoint(
            arrival_rate_rps=None,
            prompt_tokens=4000,
            output_tokens=OUTPUT_TOKENS,
            kv_context_tokens=AVERAGE_DECODE_CONTEXT_TOKENS,
        ),
        sla=SlaSpec(
            tpot_target_ps=TPOT_TARGET_PS,
            ttft_target_ps=TTFT_TARGET_PS,
        ),
        budget=BudgetSpec(max_gpus=GPU_BUDGET, max_nodes=None),
    )


def _prefill_pricing_candidate(tensor_parallel: int) -> DeploymentCandidate:
    return DeploymentCandidate(
        candidate_id=f"prefill-pricing-tp{tensor_parallel}",
        model=ModelRef(
            framework="vllm",
            model_id="Qwen/Qwen3-32B-FP8",
            inventory_sha256=INVENTORY_SHA256,
        ),
        pools=(
            PoolSpec(
                role="combined",
                engines=1,
                gpus_per_engine=tensor_parallel,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=1,
                expert_parallel=1,
                data_parallel=1,
                device="h200",
            ),
        ),
        fabric=FabricSpec(400_000_000_000, 900_000_000_000),
        workload=WorkloadPoint(None, 4000, OUTPUT_TOKENS, 0),
        sla=SlaSpec(TPOT_TARGET_PS, TTFT_TARGET_PS),
        budget=BudgetSpec(GPU_BUDGET, None),
    )


def _price_candidate(
    candidate: DeploymentCandidate,
    *,
    decode_batch: int,
    efficiency: Decimal,
    derivation: WorkDerivation,
) -> dict[str, Any]:
    prefill_pool = next(pool for pool in candidate.pools if pool.role == "prefill")
    decode_pool = next(pool for pool in candidate.pools if pool.role == "decode")
    envelope = _envelope(efficiency)

    prefill_work = _model_work(
        derivation,
        phase="prefill",
        tensor_parallel=prefill_pool.tensor_parallel,
    )
    prefill_floor = estimate_decode_step(
        _prefill_pricing_candidate(prefill_pool.tensor_parallel),
        1,
        EstimatorInputs(
            model_work=prefill_work,
            envelopes={"h200": envelope},
        ),
    ).kernel_floor
    prefill = estimate_prefill_request(
        candidate,
        EstimatorInputs(
            model_work=prefill_work,
            envelopes={"h200": envelope},
            prefill_service=prefill_floor,
            handoff_ps=0,
            handoff_source=(
                "DECLARED device-roofline comparison excludes a packetized "
                "prefill-to-decode handoff; X4 carries the regime scope"
            ),
        ),
    )

    decode_work = _model_work(
        derivation,
        phase="decode",
        tensor_parallel=decode_pool.tensor_parallel,
    )
    decode = estimate_decode_step(
        candidate,
        decode_batch,
        EstimatorInputs(
            model_work=decode_work,
            envelopes={"h200": envelope},
        ),
    )
    used_gpus = sum(pool.engines * pool.gpus_per_engine for pool in candidate.pools)
    prefill_capacity = Fraction(
        prefill_pool.engines * PICOSECONDS_PER_SECOND,
        prefill.request_ps,
    )
    decode_capacity = Fraction(
        decode_pool.engines * decode_batch * PICOSECONDS_PER_SECOND,
        OUTPUT_TOKENS * decode.step_ps,
    )
    request_capacity = min(prefill_capacity, decode_capacity)
    x = Fraction(PICOSECONDS_PER_SECOND, decode.step_ps)
    y = request_capacity * OUTPUT_TOKENS / used_gpus
    sla_pass = (
        decode.step_ps <= TPOT_TARGET_PS and prefill.request_ps <= TTFT_TARGET_PS
    )
    return {
        "candidate_key": candidate_key(candidate),
        "candidate_id": candidate.candidate_id,
        "configuration": {
            "prefill_tp": prefill_pool.tensor_parallel,
            "prefill_workers": prefill_pool.engines,
            "decode_tp": decode_pool.tensor_parallel,
            "decode_workers": decode_pool.engines,
            "decode_batch": decode_batch,
            "used_gpus": used_gpus,
        },
        "efficiency": float(efficiency),
        "point_class": "ESTIMATE",
        "decode_step_ps": decode.step_ps,
        "prefill_request_ps": prefill.request_ps,
        "x_tokens_per_second_per_user": _fraction_json(x),
        "y_tokens_per_second_per_gpu": _fraction_json(y),
        "request_capacity_per_second": _fraction_json(request_capacity),
        "sla_pass": sla_pass,
        "decode_stamp": estimate_stamp_to_json(decode.stamp),
        "prefill_stamp": estimate_stamp_to_json(prefill.stamp),
    }


def _point_xy(point: dict[str, Any]) -> tuple[Fraction, Fraction]:
    x = point["x_tokens_per_second_per_user"]
    y = point["y_tokens_per_second_per_gpu"]
    return (
        Fraction(x["numerator"], x["denominator"]),
        Fraction(y["numerator"], y["denominator"]),
    )


def pareto_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable, coordinate-deduplicated upper-right frontier."""

    by_coordinate: dict[tuple[Fraction, Fraction], dict[str, Any]] = {}
    for point in points:
        coordinate = _point_xy(point)
        prior = by_coordinate.get(coordinate)
        if prior is None or point["candidate_key"] < prior["candidate_key"]:
            by_coordinate[coordinate] = point
    unique = list(by_coordinate.values())
    frontier = []
    for point in unique:
        x, y = _point_xy(point)
        dominated = any(
            other_x >= x
            and other_y >= y
            and (other_x > x or other_y > y)
            for other in unique
            if other is not point
            for other_x, other_y in (_point_xy(other),)
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: (*_point_xy(point), point["candidate_key"]))


def _external_rows() -> list[dict[str, Any]]:
    rows = []
    with EXTERNAL_FRONTIER_PATH.open(encoding="utf-8", newline="") as stream:
        for index, raw in enumerate(csv.DictReader(stream), start=1):
            rows.append(
                {
                    "row": index,
                    "evidence_class": "MEASURED-EXTERNAL",
                    "tool": EXTERNAL_TOOL,
                    "tool_version": EXTERNAL_VERSION,
                    "database": EXTERNAL_DATABASE,
                    "x_tokens_per_second_per_user": raw["tokens/s/user"],
                    "y_tokens_per_second_per_gpu": raw["tokens/s/gpu"],
                    "concurrency": int(raw["concurrency"]),
                    "request_rate": raw["request_rate"],
                    "ttft_ms": raw["ttft"],
                    "tpot_ms": raw["tpot"],
                    "configuration": {
                        "prefill_tp": int(raw["(p)tp"]),
                        "prefill_workers": int(raw["(p)workers"]),
                        "decode_tp": int(raw["(d)tp"]),
                        "decode_workers": int(raw["(d)workers"]),
                        "decode_batch": int(raw["(d)bs"]),
                        "used_gpus": int(raw["num_total_gpus"]),
                    },
                }
            )
    return rows


def _external_hashes() -> tuple[list[dict[str, Any]], bool]:
    rows = []
    all_match = True
    for line in EXTERNAL_DIGESTS_PATH.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = STUDY_DIR / relative
        observed = sha256_file(path)
        matched = observed == expected
        all_match &= matched
        rows.append(
            {
                "path": f"examples/frontier_comparison_v1/{relative}",
                "expected_sha256": expected,
                "observed_sha256": observed,
                "matched": matched,
            }
        )
    return rows, all_match


def _chronology_holds() -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _architecture_check(inventory: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    observed = inventory["model"]["geometry"]
    exact = {name: observed[name] for name in ARCHITECTURE_LITERALS}
    return exact == ARCHITECTURE_LITERALS, exact


def _x1(derivation: WorkDerivation, parameter_count: int) -> dict[str, Any]:
    first = make_candidate(
        prefill_tp=4,
        prefill_workers=5,
        decode_tp=4,
        decode_workers=3,
        decode_batch=64,
        candidate_id="external-best-p5-tp4-d3-tp4-b64",
    )
    second = make_candidate(
        prefill_tp=4,
        prefill_workers=5,
        decode_tp=4,
        decode_workers=3,
        decode_batch=64,
        candidate_id="external-best-p5-tp4-d3-tp4-b64",
    )
    rank_bytes = parameter_count // 4
    feasibility = check_feasibility(
        first,
        static_rank_bytes_per_pool={"prefill": rank_bytes, "decode": rank_bytes},
        device_hbm_capacity_bytes={"h200": H200_CAPACITY_BYTES},
    )
    exact_shape = (
        [(pool.role, pool.engines, pool.tensor_parallel) for pool in first.pools]
        == [("prefill", 5, 4), ("decode", 3, 4)]
        and sum(pool.engines * pool.gpus_per_engine for pool in first.pools) == 32
    )
    stable_key = candidate_key(first) == candidate_key(second)
    rows = [
        {"id": "X1a", "passed": exact_shape, "detail": "exact pool structure"},
        {
            "id": "X1b",
            "passed": feasibility.accepted,
            "detail": f"FP8 rank bytes {rank_bytes} below {H200_CAPACITY_BYTES}",
        },
        {"id": "X1c", "passed": stable_key, "detail": "two keys agree"},
    ]
    return {
        "passed": sum(row["passed"] for row in rows),
        "denominator": len(rows),
        "rows": rows,
        "candidate": candidate_to_json(first),
        "candidate_key": candidate_key(first),
        "fp8_parameter_count": parameter_count,
        "fp8_rank_bytes_tp4": rank_bytes,
        "inventory_matrix_bytes": derivation.static_parameter_bytes,
        "feasibility": {
            "accepted": feasibility.accepted,
            "reasons": list(feasibility.reasons),
        },
    }


def _x2(derivation: WorkDerivation) -> dict[str, Any]:
    candidate = make_candidate(
        prefill_tp=4,
        prefill_workers=5,
        decode_tp=4,
        decode_workers=3,
        decode_batch=64,
        candidate_id="external-best-p5-tp4-d3-tp4-b64",
    )
    point = _price_candidate(
        candidate,
        decode_batch=64,
        efficiency=Decimal("1.0"),
        derivation=derivation,
    )
    decode_ps = point["decode_step_ps"]
    prefill_ps = point["prefill_request_ps"]
    decode_e_star = Fraction(decode_ps, EXTERNAL_BEST_TPOT_PS)
    prefill_e_star = Fraction(prefill_ps, EXTERNAL_BEST_TTFT_PS)
    rows = [
        {
            "id": "X2a",
            "passed": decode_ps <= EXTERNAL_BEST_TPOT_PS,
            "predicted_ps": decode_ps,
            "external_ps": EXTERNAL_BEST_TPOT_PS,
        },
        {
            "id": "X2b",
            "passed": prefill_ps <= EXTERNAL_BEST_TTFT_PS,
            "predicted_ps": prefill_ps,
            "external_ps": EXTERNAL_BEST_TTFT_PS,
        },
        {
            "id": "X2c-decode",
            "passed": Fraction(2, 5) <= decode_e_star <= 1,
            "e_star": _fraction_json(decode_e_star),
        },
        {
            "id": "X2c-prefill",
            "passed": Fraction(2, 5) <= prefill_e_star <= 1,
            "e_star": _fraction_json(prefill_e_star),
        },
    ]
    return {
        "passed": sum(row["passed"] for row in rows),
        "denominator": len(rows),
        "rows": rows,
        "decode_e_star": _fraction_json(decode_e_star),
        "prefill_e_star": _fraction_json(prefill_e_star),
        "e_star_band": {"minimum": 0.4, "maximum": 1.0},
        "e_star_policy": "reported only and never installed",
        "matched_point": point,
    }


def _candidate_family() -> list[DeploymentCandidate]:
    candidates = []
    for prefill_tp in TP_WIDTHS:
        for decode_tp in TP_WIDTHS:
            for prefill_workers in range(1, GPU_BUDGET // prefill_tp + 1):
                for decode_workers in range(1, GPU_BUDGET // decode_tp + 1):
                    used = (
                        prefill_tp * prefill_workers
                        + decode_tp * decode_workers
                    )
                    if used > GPU_BUDGET:
                        continue
                    for batch in BATCH_LADDER:
                        candidates.append(
                            make_candidate(
                                prefill_tp=prefill_tp,
                                prefill_workers=prefill_workers,
                                decode_tp=decode_tp,
                                decode_workers=decode_workers,
                                decode_batch=batch,
                            )
                        )
    return candidates


def _frontier_is_monotone(frontier: list[dict[str, Any]]) -> bool:
    coordinates = [_point_xy(point) for point in frontier]
    return all(
        left_x < right_x and left_y > right_y
        for (left_x, left_y), (right_x, right_y) in pairwise(coordinates)
    )


def _external_is_monotone(rows: list[dict[str, Any]]) -> bool:
    coordinates = sorted(
        (
            _fraction(row["x_tokens_per_second_per_user"]),
            _fraction(row["y_tokens_per_second_per_gpu"]),
        )
        for row in rows
    )
    return all(
        left_x < right_x and left_y > right_y
        for (left_x, left_y), (right_x, right_y) in pairwise(coordinates)
    )


def _configuration_tuple(configuration: dict[str, Any]) -> tuple[int, ...]:
    return (
        configuration["prefill_tp"],
        configuration["prefill_workers"],
        configuration["decode_tp"],
        configuration["decode_workers"],
        configuration["decode_batch"],
    )


def _x3b_frontier_answer(
    frontier: list[dict[str, Any]], external_x: Fraction
) -> tuple[Fraction, dict[str, Any] | None]:
    """Return the frozen step-frontier value and the point that supplies it."""

    eligible = [
        (index, point)
        for index, point in enumerate(frontier, start=1)
        if _point_xy(point)[0] >= external_x
    ]
    if not eligible:
        return Fraction(0), None
    index, point = max(eligible, key=lambda indexed: _point_xy(indexed[1])[1])
    point_x, point_y = _point_xy(point)
    first_x, _ = _point_xy(frontier[0])
    last_x, _ = _point_xy(frontier[-1])
    if external_x < first_x:
        mechanism = "left-endpoint-clamp"
    elif index == len(frontier) and external_x <= last_x:
        mechanism = "right-endpoint"
    else:
        mechanism = "first-frontier-point-at-or-above-external-x"
    return point_y, {
        "frontier_index": index,
        "candidate_id": point["candidate_id"],
        "x_tokens_per_second_per_user": float(point_x),
        "y_tokens_per_second_per_gpu": float(point_y),
        "selection_mechanism": mechanism,
    }


def _x3(
    derivation: WorkDerivation,
    external_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, list[str]]:
    candidates = _candidate_family()
    arm_records: dict[str, dict[str, Any]] = {}
    process_attempts: list[str] = []
    started = time.perf_counter()
    with ProcessGuard() as guard:
        for efficiency in EFFICIENCY_ARMS:
            points = [
                _price_candidate(
                    candidate,
                    decode_batch=int(candidate.candidate_id.rsplit("b", 1)[1]),
                    efficiency=efficiency,
                    derivation=derivation,
                )
                for candidate in candidates
            ]
            eligible = [point for point in points if point["sla_pass"]]
            frontier = pareto_points(eligible)
            arm_records[str(efficiency)] = {
                "point_class": "ESTIMATE",
                "candidate_count": len(points),
                "service_feasible_count": len(eligible),
                "frontier_count": len(frontier),
                "frontier": frontier,
                "points": points,
            }
        process_attempts.extend(guard.attempts)
    elapsed = time.perf_counter() - started

    x3a_rows = [
        {
            "id": "X3a-external",
            "passed": _external_is_monotone(external_rows),
        }
    ]
    for efficiency in EFFICIENCY_ARMS:
        key = str(efficiency)
        x3a_rows.append(
            {
                "id": f"X3a-e{key}",
                "passed": _frontier_is_monotone(arm_records[key]["frontier"]),
            }
        )

    full_frontier = arm_records["1.0"]["frontier"]
    points_by_arm_and_configuration: dict[
        tuple[str, tuple[int, ...]], dict[str, Any]
    ] = {}
    for arm, record in arm_records.items():
        for point in record["points"]:
            points_by_arm_and_configuration[
                (arm, _configuration_tuple(point["configuration"]))
            ] = point

    comparisons = []
    x3b_rows = []
    x3c_rows = []
    for external in external_rows:
        external_x = _fraction(external["x_tokens_per_second_per_user"])
        external_y = _fraction(external["y_tokens_per_second_per_gpu"])
        frontier_y, frontier_answer = _x3b_frontier_answer(
            full_frontier, external_x
        )
        x3b_pass = frontier_y >= external_y

        configuration = _configuration_tuple(external["configuration"])
        low = points_by_arm_and_configuration[("0.6", configuration)]
        high = points_by_arm_and_configuration[("1.0", configuration)]
        low_y = _point_xy(low)[1]
        high_y = _point_xy(high)[1]
        x3c_pass = low_y <= external_y <= high_y
        x3b_rows.append(
            {
                "row": external["row"],
                "passed": x3b_pass,
                "external_y": float(external_y),
                "frontier_y_at_or_above_external_x": float(frontier_y),
                "frontier_answer": frontier_answer,
            }
        )
        x3c_rows.append(
            {
                "row": external["row"],
                "passed": x3c_pass,
                "low_y": float(low_y),
                "external_y": float(external_y),
                "high_y": float(high_y),
                "miss_direction": (
                    None
                    if x3c_pass
                    else "below-0.6" if external_y < low_y else "above-1.0"
                ),
            }
        )
        comparisons.append(
            {
                **external,
                "x3b": x3b_rows[-1],
                "x3c": x3c_rows[-1],
            }
        )

    published_arms = {
        arm: {name: value for name, value in record.items() if name != "points"}
        for arm, record in arm_records.items()
    }

    return (
        {
            "X3a": {
                "passed": sum(row["passed"] for row in x3a_rows),
                "denominator": len(x3a_rows),
                "rows": x3a_rows,
            },
            "X3b": {
                "passed": sum(row["passed"] for row in x3b_rows),
                "denominator": len(x3b_rows),
                "rows": x3b_rows,
            },
            "X3c": {
                "passed": sum(row["passed"] for row in x3c_rows),
                "denominator": len(x3c_rows),
                "acceptance_minimum": 8,
                "rows": x3c_rows,
            },
            "arms": published_arms,
            "external_rows": comparisons,
            "candidate_enumeration": {
                "tensor_parallel_widths": list(TP_WIDTHS),
                "decode_batch_ladder": list(BATCH_LADDER),
                "positive_role_workers": True,
                "gpu_budget": GPU_BUDGET,
                "normalization": "used GPUs, with at most 32 GPUs",
                "frontier_filter": "TTFT <= 300 ms and TPOT <= 10 ms",
            },
            "elapsed_seconds": elapsed,
        },
        len(process_attempts),
        process_attempts,
    )


def _study_verdict(nonvoid: bool, acceptance: dict[str, bool]) -> str:
    if not nonvoid:
        return "VOID"
    return "PASS" if all(acceptance.values()) else "MIXED"


def run_study() -> dict[str, Any]:
    """Return the complete frozen study result."""

    inventory = _load_json(INVENTORY_PATH)
    suite = _load_json(SUITE_PATH)
    external_rows = _external_rows()
    derivation = derive_work(inventory)
    external_hashes, external_hashes_match = _external_hashes()
    architecture_holds, observed_architecture = _architecture_check(inventory)
    chronology_holds = _chronology_holds()
    expectations_match = sha256_file(EXPECTATIONS_PATH) == EXPECTATIONS_SHA256
    inventory_match = sha256_file(INVENTORY_PATH) == INVENTORY_SHA256
    suite_match = sha256_file(SUITE_PATH) == SUITE_SHA256

    x1 = _x1(derivation, suite["reference_model"]["parameter_count"])
    x2 = _x2(derivation)
    x3, process_count, process_attempts = _x3(derivation, external_rows)
    x3c_holds = x3["X3c"]["passed"] >= x3["X3c"]["acceptance_minimum"]
    wall_holds = x3["elapsed_seconds"] <= 120.0
    guards = [
        {
            "id": "FG-1",
            "held": external_hashes_match,
            "detail": "tracked external rows match external.sha256",
        },
        {
            "id": "FG-2",
            "held": architecture_holds,
            "detail": "inventory geometry matches all seven frozen literals",
        },
        {
            "id": "FG-3",
            "held": all(
                row["evidence_class"] == "MEASURED-EXTERNAL"
                for row in external_rows
            ),
            "detail": "external rows are display and comparison inputs only",
        },
        {
            "id": "FG-4",
            "held": process_count == 0,
            "detail": "pricing-lane process interception count is zero",
        },
        {
            "id": "FG-5",
            "held": chronology_holds and expectations_match,
            "detail": "83bb281 is an ancestor and the frozen bytes match",
        },
    ]
    findings = [guard["id"] for guard in guards if not guard["held"]]
    nonvoid = not findings
    acceptance = {
        "nonvoid": nonvoid,
        "x1_pass": x1["passed"] == x1["denominator"],
        "x2_pass": x2["passed"] == x2["denominator"],
        "x3a_pass": x3["X3a"]["passed"] == x3["X3a"]["denominator"],
        "x3b_pass": x3["X3b"]["passed"] == x3["X3b"]["denominator"],
        "x3c_pass": x3c_holds,
        "wall_time_pass": wall_holds,
    }
    verdict = _study_verdict(nonvoid, acceptance)
    external_concurrency = [row["concurrency"] for row in external_rows]
    external_request_rates = [
        _fraction(row["request_rate"]) for row in external_rows
    ]
    external_ttft_values = sorted({row["ttft_ms"] for row in external_rows})

    return {
        "schema": RESULT_SCHEMA,
        "verdict": verdict,
        "nonvoid": nonvoid,
        "void_findings": findings,
        "chronology": {
            "expectations_commit": EXPECTATIONS_COMMIT,
            "expectations_sha256": EXPECTATIONS_SHA256,
            "expectations_bytes_match": expectations_match,
            "expectations_commit_is_ancestor": chronology_holds,
        },
        "frozen_inputs": {
            "external_archive_manifest_sha256": EXTERNAL_ARCHIVE_MANIFEST_SHA256,
            "external_sha256_governs": True,
            "external_files": external_hashes,
            "inventory": {
                "path": INVENTORY_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": INVENTORY_SHA256,
                "matched": inventory_match,
            },
            "suite": {
                "path": SUITE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": SUITE_SHA256,
                "matched": suite_match,
            },
        },
        "extraction": {
            "model": inventory["model"],
            "framework": inventory["framework"],
            "observed_architecture": observed_architecture,
            "expected_architecture": ARCHITECTURE_LITERALS,
            "architecture_exact": architecture_holds,
            "companion_sglang_inventory_sha256": (
                "51740b52625002a964e75fddb679e9f8394a08a7d7c62556d2535c3bc60515e3"
            ),
            "case_count": len(inventory["cases"]),
            "family_count": len(inventory["kernel_families"]),
            "logical_visits_per_case": 257,
        },
        "declared_h200_envelope": {
            "evidence_class": "DECLARED",
            "peak_dense_fp8_flops_per_second": H200_PEAK_FP8_FLOPS_PER_SECOND,
            "hbm_bytes_per_second": H200_HBM_BYTES_PER_SECOND,
            "capacity_bytes": H200_CAPACITY_BYTES,
            "efficiency_arms": [float(value) for value in EFFICIENCY_ARMS],
            "source": "https://www.nvidia.com/en-us/data-center/h200/",
            "sparsity_note": (
                "NVIDIA publishes 3.958 PFLOP/s FP8 with sparsity; the frozen "
                "dense envelope removes the 2:1 sparsity factor"
            ),
        },
        "workload": {
            "input_tokens": 4000,
            "shared_prefix_tokens": 500,
            "uncached_prompt_tokens": UNCACHED_PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "average_decode_context_tokens": AVERAGE_DECODE_CONTEXT_TOKENS,
            "ttft_target_ps": TTFT_TARGET_PS,
            "tpot_target_ps": TPOT_TARGET_PS,
            "gpu_budget": GPU_BUDGET,
        },
        "model_work_derivation": {
            "reference_tensor_parallel": REFERENCE_TP,
            "static_parameter_bytes": derivation.static_parameter_bytes,
            "decode_total_flops_per_batch_item": (
                derivation.decode_total_flops_per_batch_item
            ),
            "decode_flops_per_batch_item_per_rank_tp4": (
                derivation.decode_total_flops_per_batch_item // REFERENCE_TP
            ),
            "decode_logical_kv_bytes_per_batch_item": derivation.decode_kv_bytes,
            "prefill_total_flops_per_request": (
                derivation.prefill_total_flops_per_request
            ),
            "prefill_flops_per_request_per_rank_tp4": (
                derivation.prefill_total_flops_per_request // REFERENCE_TP
            ),
            "prefill_logical_kv_bytes_per_request": derivation.prefill_kv_bytes,
            "attention_score_flops_per_decode_context_token": (
                derivation.decode_attention_score_flops_per_context_token
            ),
            "attention_score_flops_per_prefill_token_pair": (
                derivation.prefill_attention_score_flops_per_token_pair
            ),
            "attention_score_projection_inconsistency": {
                "decode_flops_per_token_pair": (
                    derivation.decode_attention_score_flops_per_token_pair
                ),
                "prefill_flops_per_token_pair": (
                    derivation.prefill_attention_score_flops_per_token_pair
                ),
                "decode_over_prefill": 8,
                "residual_task": "COMP-81",
                "frozen_inventories_changed": False,
            },
            "kv_bytes_per_context_token": derivation.kv_bytes_per_context_token,
            "statement": (
                "Whole-model FLOPs and logical HBM bytes come from the "
                "inventory's per-layer work and both divide by the candidate's "
                "tensor-parallel width for per-rank pricing. FP8 uses one byte "
                "per parameter. KV read bytes are evaluated at 4,250 decode "
                "context tokens and 3,500 uncached prefill tokens."
            ),
        },
        "fatal_guards": guards,
        "families": {
            "X1": x1,
            "X2": x2,
            "X3": x3,
            "X4": {
                "evidence_class": "DECLARED-FROM-LADDER-STUDY",
                "scored": False,
                "contention_free_point_to_point_packet_over_ideal": 1.016,
                "eight_into_one_packet_over_ideal": 8.0,
                "workload_regime": (
                    "intra-node tensor parallel and one prefill-to-decode "
                    "transfer are contention-free point-to-point legs"
                ),
                "packet_execution_in_this_study": False,
                "source": "examples/frontier_ladder_v1/RESULTS.md",
                "zero_collective_bytes_identity": {
                    "logical_collective_bytes_per_gpu_per_batch_item": 0,
                    "applies_to_every_candidate": True,
                    "bound_scope": (
                        "the 1 to 2 percent contention-free ideal-versus-packet "
                        "bound applies only to represented legs, not to omitted "
                        "tensor-parallel collective service"
                    ),
                },
            },
            "W": {
                "passed": int(wall_holds),
                "denominator": 1,
                "elapsed_seconds": x3["elapsed_seconds"],
                "limit_seconds": 120.0,
                "process": "single",
                "external_context_seconds": 11.0,
                "external_context_scored": False,
            },
        },
        "score_classes": {
            "exact_oracles": {"X1": x1},
            "behavioral_relations": {
                "X2": {
                    "passed": x2["passed"],
                    "denominator": x2["denominator"],
                },
                "X3a": x3["X3a"],
                "X3b": x3["X3b"],
                "X3c": x3["X3c"],
                "W": {
                    "passed": int(wall_holds),
                    "denominator": 1,
                },
            },
            "fatal_unscored": ["FG-1", "FG-2", "FG-3", "FG-4", "FG-5", "X4"],
        },
        "pricing_subprocess_count": process_count,
        "pricing_subprocess_attempts": process_attempts,
        "external_rows_entered_pricing": False,
        "honesty": {
            "external_calibration_advantage": (
                "The external numbers interpolate a measured per-operation "
                "database for real H200 silicon. On absolute kernel "
                "throughput, that side is better calibrated today."
            ),
            "our_claims": (
                "The defensible precision claims are the X4 network-mechanism "
                "envelope, evidence-class labeling on every number and exact "
                "accounting gates. Nothing broader is claimed."
            ),
            "e_star_installed": False,
            "version_drift": (
                "The local aiconfigurator 0.11.0 best row is 602.586 tokens/s/GPU "
                "at 108.944 tokens/s/user. The published README snapshot is "
                "684.79 at 100.31 with a different topology. Neither is "
                "preferred or fitted."
            ),
            "external_ttft_semantics": {
                "status": "candidate-explanation-without-rescoring",
                "ttft_ms_values": external_ttft_values,
                "concurrency_min": min(external_concurrency),
                "concurrency_max": max(external_concurrency),
                "request_rate_min": float(min(external_request_rates)),
                "request_rate_max": float(max(external_request_rates)),
                "explanation": (
                    "The external TTFT column is attached to operating points "
                    "that carry concurrency and request_rate, while the SimLLM "
                    "prefill value is isolated service. The frozen matched-point "
                    "premise therefore conflates queueing with service."
                ),
                "residual_task": "DEPLOY-12",
                "rescored": False,
            },
            "conduct_deviation": {
                "status": "recorded-without-history-rewrite",
                "commits": [
                    {
                        "commit": "8a96b3f",
                        "nonconforming_prefix": "feat:",
                    },
                    {
                        "commit": "11db813",
                        "nonconforming_prefix": "feat:",
                    },
                ],
            },
        },
        "acceptance": acceptance,
    }


def _outside_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return True
    return False


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _portable_argv() -> list[str]:
    return [
        ".venv/bin/python",
        "examples/frontier_comparison_v1/run_study.py",
        "--run-dir",
        "${SIMLLM_FRONTIER_COMPARISON_RUN_ROOT}",
    ]


def _begin_attempt(run_root: Path) -> tuple[Path, Path | None]:
    root = run_root.resolve()
    if not _outside_repository(root):
        raise SystemExit("--run-dir must be outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[int, Path]] = []
    for path in root.glob("attempt-*"):
        match = re.fullmatch(r"attempt-(\d+)", path.name)
        if match is not None and path.is_dir():
            attempts.append((int(match.group(1)), path))
    incomplete = [path for _, path in attempts if not (path / "verdict.json").is_file()]
    if incomplete:
        names = ", ".join(path.name for path in sorted(incomplete))
        raise SystemExit(
            f"cannot start a later attempt while verdict records are missing: {names}"
        )
    number = max((number for number, _ in attempts), default=0) + 1
    attempt_dir = root / f"attempt-{number}"
    attempt_dir.mkdir(parents=False, exist_ok=False)
    previous = None if not attempts else max(attempts)[1]
    _write_json_exclusive(
        attempt_dir / "attempt.json",
        {
            "schema": "simllm-frontier-comparison-attempt-v1",
            "attempt_id": attempt_dir.name,
            "portable_argv": _portable_argv(),
            "started_unix_time_ns": time.time_ns(),
        },
    )
    return attempt_dir, previous


def deterministic_projection(result: dict[str, Any]) -> dict[str, Any]:
    """Return every result quantity that is deterministic across full reruns."""

    projected = json.loads(json.dumps(result))
    projected.pop("attempt_evidence", None)
    projected.pop("verdict", None)
    projected["families"]["X3"].pop("elapsed_seconds", None)
    projected["families"]["W"].pop("elapsed_seconds", None)
    projected["families"]["W"].pop("passed", None)
    projected["score_classes"]["behavioral_relations"].pop("W", None)
    projected["acceptance"].pop("wall_time_pass", None)
    return projected


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    attempt_dir, previous_attempt = _begin_attempt(args.run_dir)
    verdict_path = attempt_dir / "verdict.json"
    try:
        result = run_study()
        projection_sha256 = _canonical_sha256(deterministic_projection(result))
        previous_result_sha256 = None
        previous_projection_sha256 = None
        reproduction_matched = None
        if previous_attempt is not None:
            previous_verdict = previous_attempt / "verdict.json"
            previous_result_sha256 = sha256_file(previous_verdict)
            previous_result = _load_json(previous_verdict)
            previous_projection_sha256 = _canonical_sha256(
                deterministic_projection(previous_result)
            )
            reproduction_matched = previous_projection_sha256 == projection_sha256
        result["attempt_evidence"] = {
            "attempt_id": attempt_dir.name,
            "policy": (
                "each full run uses a fresh attempt-N directory and refuses a "
                "later attempt until every earlier attempt has a verdict"
            ),
            "portable_argv": _portable_argv(),
            "previous_attempt_id": (
                None if previous_attempt is None else previous_attempt.name
            ),
            "previous_result_sha256": previous_result_sha256,
            "deterministic_projection_sha256": projection_sha256,
            "previous_deterministic_projection_sha256": (
                previous_projection_sha256
            ),
            "deterministic_reproduction_matched": reproduction_matched,
            "excluded_from_reproduction": (
                "wall-clock elapsed values, their W outcome, the overall verdict "
                "that includes W, and attempt metadata"
            ),
        }
    except BaseException as error:
        _write_json_exclusive(
            verdict_path,
            {
                "schema": "simllm-frontier-comparison-attempt-v1",
                "verdict": "ERROR",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    _write_json_exclusive(verdict_path, result)
    print(
        json.dumps(
            {
                "attempt": attempt_dir.name,
                "verdict": result["verdict"],
                "nonvoid": result["nonvoid"],
                "X1": [
                    result["families"]["X1"]["passed"],
                    result["families"]["X1"]["denominator"],
                ],
                "X2": [
                    result["families"]["X2"]["passed"],
                    result["families"]["X2"]["denominator"],
                ],
                "X3a": [
                    result["families"]["X3"]["X3a"]["passed"],
                    result["families"]["X3"]["X3a"]["denominator"],
                ],
                "X3b": [
                    result["families"]["X3"]["X3b"]["passed"],
                    result["families"]["X3"]["X3b"]["denominator"],
                ],
                "X3c": [
                    result["families"]["X3"]["X3c"]["passed"],
                    result["families"]["X3"]["X3c"]["denominator"],
                ],
                "deterministic_reproduction_matched": result[
                    "attempt_evidence"
                ]["deterministic_reproduction_matched"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["nonvoid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
