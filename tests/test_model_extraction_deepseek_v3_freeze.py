"""Lock the expectations-only COMP-67 DeepSeek-V3 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/model_extraction_deepseek_v3_v1"
EXPECTATIONS = STUDY / "expectations.json"
SUITE = (
    ROOT
    / "offline/calibration/suites/"
    "deepseek-v3-text-v1-frameworks-2026-08-25/suite.json"
)
SUITE_SHA256 = "88f718c94ad35bb0c74314811680b5ebff7e5df70759096dc0b640f84f47bd69"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _case_axes(cell: dict[str, object]) -> tuple[int, int, int, int, int]:
    if cell["phase"] == "prefill":
        requests = int(cell["requests"])
        per_request = int(cell["prompt_tokens_per_request"])
        tokens = int(cell["total_prompt_tokens"])
        return tokens, requests, tokens, requests * per_request**2 // 2, requests

    batch = int(cell["batch"])
    context = int(cell["context_tokens"])
    new_tokens = int(cell["new_tokens_per_request"])
    return (
        batch * new_tokens,
        batch,
        batch * context,
        batch * (context - new_tokens),
        batch,
    )


def _fp8_matrix_bytes(rows: int, columns: int) -> int:
    return (
        rows * columns
        + 4 * ((rows + 127) // 128) * ((columns + 127) // 128)
    )


def test_deepseek_v3_freeze_contains_authored_expectations_only() -> None:
    freeze = _load(EXPECTATIONS)

    assert freeze["schema"] == (
        "simllm-model-extraction-deepseek-v3-expectations-v1"
    )
    assert freeze["study"] == "model-extraction-deepseek-v3-v1"
    assert freeze["authored_after_commit"] == (
        "dc350b6996215adf69384c23335b496440042fe7"
    )
    assert "results" not in freeze
    assert "observed_inventories" not in freeze
    assert (STUDY / ".gitattributes").read_text(encoding="utf-8") == (
        "* text eol=lf\n"
    )
    assert SUITE.parent.joinpath(".gitattributes").read_text(
        encoding="utf-8"
    ) == "* text eol=lf\n"


def test_deepseek_v3_checkpoint_and_manifest_are_exact() -> None:
    freeze = _load(EXPECTATIONS)
    suite = _load(SUITE)
    model = suite["reference_model"]
    choice = freeze["checkpoint_choice"]
    shards = model["weight_shards"]

    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == SUITE_SHA256
    assert freeze["suite"]["sha256"] == SUITE_SHA256
    assert choice["selected"] == "deepseek-ai/DeepSeek-V3"
    assert choice["revision"] == model["revision"]
    assert choice["r1_scope"].startswith("excluded")
    assert choice["v3_0324_structural_relation"].startswith(
        "all model and quantization configuration fields are identical"
    )
    assert len(shards) == 163
    assert [row["name"] for row in shards] == sorted(
        row["name"] for row in shards
    )
    assert sum(row["bytes"] for row in shards) == 688_586_727_753
    assert hashlib.sha256(_canonical_bytes(shards)).hexdigest() == (
        "ec8b878368c5fdb9f3288bd3a36a723a1637ec76464135a3f5b2e9aeff4072b4"
    )
    assert model["local_weight_byte_verification"] is False


def test_deepseek_v3_historical_inputs_are_byte_locked() -> None:
    freeze = _load(EXPECTATIONS)

    for relative, expected_sha256 in freeze["historical_byte_locks"].items():
        assert hashlib.sha256(ROOT.joinpath(relative).read_bytes()).hexdigest() == (
            expected_sha256
        )


def test_deepseek_v3_geometry_schedule_and_family_order_are_exact() -> None:
    freeze = _load(EXPECTATIONS)
    suite = _load(SUITE)
    geometry = freeze["geometry_symbols"]
    contract = freeze["inventory_contract"]
    families = {row["id"]: row for row in contract["families"]}
    stack = suite["reference_model"]["deepseek_stack"]

    assert contract["ordered_families"] == list(families)
    assert len(families) == 14
    assert geometry == {
        "H": 7168,
        "I": 18432,
        "Nq": 128,
        "Qr": 1536,
        "KVr": 512,
        "Dn": 128,
        "Dr": 64,
        "Dv": 128,
        "L": 61,
        "Ld": 3,
        "Lm": 58,
        "E": 256,
        "K": 8,
        "M": 2048,
        "V": 129280,
        "Lmtp": 1,
    }
    assert stack["first_k_dense_replace"] == geometry["Ld"]
    assert stack["moe_layer_freq"] == 1
    assert geometry["L"] - geometry["Ld"] == geometry["Lm"]
    assert [families[name]["layers"] for name in contract["ordered_families"]] == [
        61,
        61,
        61,
        61,
        61,
        61,
        61,
        61,
        3,
        58,
        58,
        58,
        1,
        1,
    ]


def test_deepseek_v3_mla_integer_forms_are_exact() -> None:
    freeze = _load(EXPECTATIONS)
    g = freeze["geometry_symbols"]
    contract = freeze["inventory_contract"]
    f = {row["id"]: row for row in contract["families"]}
    h, heads = g["H"], g["Nq"]
    q_rank, kv_rank = g["Qr"], g["KVr"]
    nope, rope, value = g["Dn"], g["Dr"], g["Dv"]

    assert f["mla_q_compression"]["flops_per_new_token_per_layer"] == (
        2 * h * q_rank
    )
    assert f["mla_q_compression"]["static_hbm_bytes_per_layer"] == (
        _fp8_matrix_bytes(q_rank, h)
    )
    assert f["mla_q_decompression"]["flops_per_new_token_per_layer"] == (
        2 * q_rank * heads * (nope + rope)
    )
    assert f["mla_q_decompression"]["static_hbm_bytes_per_layer"] == (
        _fp8_matrix_bytes(heads * (nope + rope), q_rank)
    )
    assert f["mla_kv_compression"]["flops_per_new_token_per_layer"] == (
        2 * h * (kv_rank + rope)
    )
    assert f["mla_kv_decompression"]["flops_per_new_token_per_layer"] == (
        2 * kv_rank * heads * (nope + value)
    )
    assert f["mla_rotary_split"]["flops_per_new_token_per_layer"] == (
        3 * rope * (heads + 1)
    )
    assert f["mla_attention"][
        "prefill_flops_per_attention_pair_per_layer"
    ] == 2 * heads * ((nope + rope) + value)
    assert f["mla_attention"][
        "decode_flops_per_attention_pair_per_layer"
    ] == 2 * heads * (2 * kv_rank + rope)
    assert f["mla_compressed_kv_read"][
        "hbm_bytes_per_kv_token_per_layer"
    ] == 2 * (kv_rank + rope)


def test_deepseek_v3_moe_dense_and_mtp_integer_forms_are_exact() -> None:
    freeze = _load(EXPECTATIONS)
    g = freeze["geometry_symbols"]
    contract = freeze["inventory_contract"]
    f = {row["id"]: row for row in contract["families"]}
    h, dense, expert = g["H"], g["I"], g["M"]
    experts, top_k, vocab = g["E"], g["K"], g["V"]
    expert_bytes = _fp8_matrix_bytes(2 * expert, h) + _fp8_matrix_bytes(
        h, expert
    )

    assert f["dense_early_mlp"]["flops_per_new_token_per_layer"] == (
        2 * 3 * h * dense
    )
    assert f["moe_router"]["flops_per_new_token_per_layer"] == (
        2 * h * experts + 2 * experts + 3 * top_k - 1
    )
    assert f["moe_router"]["static_hbm_bytes_per_layer"] == (
        2 * h * experts + 4 * experts
    )
    assert f["moe_shared_expert"]["flops_per_new_token_per_layer"] == (
        2 * 3 * h * expert
    )
    assert f["moe_shared_expert"]["static_hbm_bytes_per_layer"] == expert_bytes
    assert f["moe_routed_experts"]["flops_per_new_token_per_layer"] == (
        top_k * 2 * 3 * h * expert
    )
    assert f["moe_routed_experts"]["static_hbm_bytes_per_layer"] == (
        experts * expert_bytes
    )
    assert f["lm_head"]["flops_per_sampled_token"] == 2 * h * vocab
    assert f["multi_token_prediction_head"]["launch_scale_axis"] == (
        "mtp_enabled"
    )
    assert contract["logical_visit_count_without_mtp"] == (
        61 * 8 + 3 + 58 * 3 + 1
    )
    assert contract["logical_visit_count_with_mtp"] == (
        contract["logical_visit_count_without_mtp"] + 1
    )


def test_deepseek_v3_all_case_oracles_follow_independent_forms() -> None:
    freeze = _load(EXPECTATIONS)
    suite = _load(SUITE)
    contract = freeze["inventory_contract"]
    oracles = freeze["exact_case_oracles"]

    assert len(oracles) == len(suite["graph_cells"]) == 20
    for cell, oracle in zip(suite["graph_cells"], oracles, strict=True):
        tokens, sequences, kv_tokens, pairs, sampled = _case_axes(cell)
        mtp = int(cell["mtp_enabled"])
        pair_coefficient = contract[
            f"{cell['phase']}_flops_per_attention_pair"
        ]
        mtp_pair_coefficient = contract[
            f"mtp_{cell['phase']}_flops_per_attention_pair"
        ]

        assert oracle["case_id"] == cell["id"]
        assert (
            oracle["new_tokens"],
            oracle["sequences"],
            oracle["kv_tokens"],
            oracle["attention_pairs"],
            oracle["sampled"],
            oracle["mtp_enabled"],
        ) == (tokens, sequences, kv_tokens, pairs, sampled, mtp)
        assert oracle["logical_visit_count"] == 666 + mtp
        assert oracle["aggregate_flops"] == (
            contract["base_fixed_flops_per_new_token"] * tokens
            + pair_coefficient * pairs
            + contract["flops_per_sampled_token"] * sampled
            + mtp
            * (
                contract["mtp_fixed_flops_per_new_token"] * tokens
                + mtp_pair_coefficient * pairs
                + contract["mtp_flops_per_sampled_token"] * sampled
            )
        )
        assert oracle["aggregate_hbm_bytes"] == (
            contract["base_static_hbm_bytes"]
            + contract["base_hbm_bytes_per_kv_token"] * kv_tokens
            + mtp
            * (
                contract["mtp_static_hbm_bytes"]
                + contract["mtp_hbm_bytes_per_kv_token"] * kv_tokens
            )
        )


def test_deepseek_v3_deployment_rank_classes_conserve_exactly() -> None:
    projection = _load(EXPECTATIONS)["deployment_projection_contract"]
    logical = projection["logical_expert_count"]
    physical = projection["physical_expert_slots"]
    per_expert_bytes = projection[
        "per_expert_static_hbm_bytes_all_base_moe_layers"
    ]

    for classes in projection["rank_classes"].values():
        assert sum(
            row["rank_count"] * row["logical_experts_per_rank"]
            for row in classes
        ) == logical
        assert sum(
            row["rank_count"] * row["physical_slots_per_rank"]
            for row in classes
        ) == physical
        assert sum(
            row["rank_count"] * row["redundant_slots_per_rank"]
            for row in classes
        ) == physical - logical
        assert sum(
            row["rank_count"]
            * row["logical_experts_per_rank"]
            * per_expert_bytes
            for row in classes
        ) == logical * per_expert_bytes

    for oracle in projection["disclosed_case_oracles"]:
        assert oracle["global_routed_visits_per_moe_layer"] == (
            oracle["expert_parallel"]
            * oracle["local_new_tokens_per_rank"]
            * 8
        )
        assert oracle["routed_visits_per_unique_expert_per_moe_layer"] == (
            oracle["global_routed_visits_per_moe_layer"] // logical
        )


def test_deepseek_v3_physical_sanity_uses_byte_over_rate_floors() -> None:
    sanity = _load(EXPECTATIONS)["physical_sanity"]
    rate = sanity["h100_or_h800_hbm_bytes_per_second"]

    for ep in (32, 72, 144):
        assert sanity[f"ep{ep}_one_pass_hbm_floor_picoseconds"] == (
            sanity[f"ep{ep}_physical_static_hbm_bytes_per_rank"]
            * 10**12
            // rate
        )
    assert sanity["base_inventory_hbm_bytes"] < (
        sanity["enabled_mtp_inventory_hbm_bytes"]
    )
    assert sanity["enabled_mtp_inventory_hbm_bytes"] < (
        sanity["checkpoint_payload_bytes"]
    )
