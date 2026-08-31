# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simllm.calibration.external_db import (
    ExternalDatabaseIdentityError,
    ExternalOperationDatabase,
)
from simllm.calibration.external_nccl import ExternalNcclDatabase
from simllm.calibration.external_pass import ExternalModelConfig, ExternalPassModel

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples/minimax_ep_scaling_v1/study_config.json"


@pytest.fixture(scope="module")
def study_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("width_index", range(4))
def test_minimax_pass_matches_frozen_live_oracles(
    study_config: dict[str, object],
    width_index: int,
) -> None:
    model_record = study_config["model"]
    operating_point = study_config["operating_point"]
    frozen = study_config["widths"][width_index]
    assert isinstance(model_record, dict)
    assert isinstance(operating_point, dict)
    assert isinstance(frozen, dict)
    model = ExternalModelConfig.from_mapping(
        model_record,
        architecture="moe",
        tensor_parallel=1,
        pipeline_parallel=1,
        expert_parallel=int(frozen["expert_parallel"]),
        workload_distribution=str(operating_point["workload_distribution"]),
        gemm_quant_mode=str(model_record["gemm_quant_mode"]),
        attention_quant_mode=str(model_record["kv_cache_quant_mode"]),
    )
    result = ExternalPassModel(
        ExternalOperationDatabase.load(),
        model,
        nccl_database=ExternalNcclDatabase.load(),
    ).run_generation(batch_size=1, isl=256, osl=2, stride=32)
    operations = result.operation_latencies()
    pre_dispatch = operations["generation_moe_pre_dispatch"]
    post_dispatch = operations["generation_moe_post_dispatch"]

    assert pre_dispatch.hex() == frozen["live_pre_dispatch_hex"]
    assert post_dispatch.hex() == frozen["live_post_dispatch_hex"]
    assert (pre_dispatch + post_dispatch).hex() == frozen["live_dispatch_hex"]
    assert result.total.hex == frozen["live_decode_step_hex"]
    dispatch_terms = [
        entry
        for entry in result.operations
        if entry.operation in {
            "generation_moe_pre_dispatch",
            "generation_moe_post_dispatch",
        }
    ]
    assert {entry.source.backend for entry in dispatch_terms} == {"nccl"}
    assert len(result.operations) == 13


def test_model_config_requires_explicit_nextn() -> None:
    model = {
        "model_id": "missing-nextn",
        "num_hidden_layers": 1,
        "hidden_size": 128,
        "intermediate_size": 64,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "vocab_size": 256,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
    }
    with pytest.raises(ExternalDatabaseIdentityError, match="explicit nextn"):
        ExternalModelConfig.from_mapping(
            model,
            architecture="moe",
            tensor_parallel=1,
            pipeline_parallel=1,
            expert_parallel=2,
        )
