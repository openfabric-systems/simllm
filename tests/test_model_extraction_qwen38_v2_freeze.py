"""Lock the expectations-only COMP-62 completion study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "examples/model_extraction_qwen38_v2"
EXPECTATIONS = STUDY / "expectations.json"
SUITES = ROOT / "offline/calibration/suites"
CURRENT_SUITE = (
    SUITES / "qwen3.8-27b-text-v1-frameworks-2026-08-25/suite.json"
)
HISTORICAL_SUITE = SUITES / "qwen3.8-27b-text-v1/suite.json"
CURRENT_SUITE_SHA256 = (
    "7be24843ffae71de65a1eab243eab9f592ce614097d701d5234eabd0c5980a9c"
)


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


def test_qwen38_v2_freeze_contains_authored_expectations_only() -> None:
    freeze = _load(EXPECTATIONS)

    assert freeze["schema"] == (
        "simllm-model-extraction-qwen38-completion-expectations-v1"
    )
    assert freeze["study"] == "model-extraction-qwen38-v2"
    assert freeze["authored_after_commit"] == (
        "76d389f1fa3dde5b7935d5cc0b85401849fe3026"
    )
    assert "results" not in freeze
    assert "observed_inventories" not in freeze
    assert (STUDY / ".gitattributes").read_text(encoding="utf-8") == (
        "* text eol=lf\n"
    )
    assert CURRENT_SUITE.parent.joinpath(".gitattributes").read_text(
        encoding="utf-8"
    ) == "* text eol=lf\n"


def test_qwen38_v2_current_suite_changes_only_declared_surfaces() -> None:
    current_raw = CURRENT_SUITE.read_bytes()
    current = json.loads(current_raw)
    historical = _load(HISTORICAL_SUITE)
    freeze = _load(EXPECTATIONS)

    assert hashlib.sha256(current_raw).hexdigest() == CURRENT_SUITE_SHA256
    assert freeze["suite"]["sha256"] == CURRENT_SUITE_SHA256
    assert _canonical_bytes(current["reference_model"]) == _canonical_bytes(
        historical["reference_model"]
    )
    assert _canonical_bytes(current["graph_cells"]) == _canonical_bytes(
        historical["graph_cells"]
    )
    assert current["suite"] == "qwen3.8-27b-text-v1-frameworks-2026-08-25"
    assert current["phase_scope"]["unsupported_mechanism_policy"] == (
        "require-complete-gated-delta-net-inventory"
    )
    assert [framework["version"] for framework in current["frameworks"]] == [
        "0.27.1",
        "0.5.19.dev345+gbfeae4e79",
    ]


def test_qwen38_v2_historical_and_granite_inputs_are_byte_locked() -> None:
    freeze = _load(EXPECTATIONS)

    for relative, expected_sha256 in freeze["historical_byte_locks"].items():
        assert hashlib.sha256(ROOT.joinpath(relative).read_bytes()).hexdigest() == (
            expected_sha256
        )


def test_qwen38_v2_family_geometry_and_schedule_are_exact() -> None:
    freeze = _load(EXPECTATIONS)
    contract = freeze["inventory_contract"]
    families = {family["id"]: family for family in contract["families"]}
    geometry = freeze["geometry_symbols"]
    h = geometry["H"]
    intermediate = geometry["I"]
    key_width = geometry["K"]
    value_width = geometry["V"]
    conv_width = geometry["W"]
    value_heads = geometry["Nv"]
    value_head_width = geometry["Dv"]
    key_head_width = geometry["Dk"]

    assert contract["ordered_families"] == list(families)
    assert [families[name]["layers"] for name in contract["ordered_families"]] == [
        16,
        16,
        16,
        48,
        48,
        48,
        48,
        48,
        48,
        48,
        64,
        1,
    ]
    assert families["gdn_input_projection"]["flops_per_new_token_per_layer"] == (
        2 * h * (2 * key_width + 2 * value_width + 2 * value_heads)
    )
    assert families["gdn_short_convolution"][
        "flops_per_new_token_per_layer"
    ] == 2 * (2 * key_width + value_width) * conv_width
    assert families["gdn_state_read"]["hbm_bytes_per_sequence_per_layer"] == (
        4 * value_heads * value_head_width * key_head_width
    )
    assert families["gdn_state_update"]["flops_per_new_token_per_layer"] == (
        geometry["Nk"] * (7 * key_head_width + 2)
        + value_heads * (7 * value_head_width * key_head_width + 2 * value_head_width + 7)
    )
    assert families["gdn_gated_norm"]["flops_per_new_token_per_layer"] == (
        value_heads * (7 * value_head_width + 2)
    )
    assert families["gdn_output_projection"][
        "flops_per_new_token_per_layer"
    ] == 2 * value_width * h
    assert families["mlp_gemm"]["flops_per_new_token_per_layer"] == (
        2 * 3 * h * intermediate
    )
    assert sum(family["layers"] for family in families.values()) == 449
    assert contract["fixed_flops_per_new_token"] == (
        families["attn_gemm"]["flops_per_new_token_per_layer"] * geometry["Lf"]
        + families["gdn_input_projection"]["flops_per_new_token_per_layer"]
        * geometry["Ll"]
        + families["gdn_short_convolution"]["flops_per_new_token_per_layer"]
        * geometry["Ll"]
        + families["gdn_state_update"]["flops_per_new_token_per_layer"]
        * geometry["Ll"]
        + families["gdn_gated_norm"]["flops_per_new_token_per_layer"]
        * geometry["Ll"]
        + families["gdn_output_projection"]["flops_per_new_token_per_layer"]
        * geometry["Ll"]
        + families["mlp_gemm"]["flops_per_new_token_per_layer"] * geometry["L"]
    )
    assert contract["flops_per_attention_pair"] == (
        families["attn_score"]["flops_per_attention_pair_per_layer"]
        * geometry["Lf"]
    )
    assert contract["hbm_bytes_per_sequence"] == (
        families["gdn_short_convolution"][
            "state_read_write_hbm_bytes_per_sequence_per_layer"
        ]
        * geometry["Ll"]
        + families["gdn_state_read"]["hbm_bytes_per_sequence_per_layer"]
        * geometry["Ll"]
        + families["gdn_state_write"]["hbm_bytes_per_sequence_per_layer"]
        * geometry["Ll"]
    )
    assert contract["hbm_bytes_per_kv_token"] == (
        families["kv_read"]["hbm_bytes_per_kv_token_per_layer"] * geometry["Lf"]
    )


def test_qwen38_v2_all_case_oracles_follow_independent_integer_forms() -> None:
    freeze = _load(EXPECTATIONS)
    suite = _load(CURRENT_SUITE)
    contract = freeze["inventory_contract"]
    oracles = freeze["exact_case_oracles"]

    assert len(oracles) == len(suite["graph_cells"]) == 15
    for cell, oracle in zip(suite["graph_cells"], oracles, strict=True):
        tokens, sequences, kv_tokens, pairs, sampled = _case_axes(cell)
        assert oracle["case_id"] == cell["id"]
        assert (
            oracle["new_tokens"],
            oracle["sequences"],
            oracle["kv_tokens"],
            oracle["attention_pairs"],
            oracle["sampled"],
        ) == (tokens, sequences, kv_tokens, pairs, sampled)
        assert oracle["aggregate_flops"] == (
            contract["fixed_flops_per_new_token"] * tokens
            + contract["flops_per_attention_pair"] * pairs
            + contract["flops_per_sampled_token"] * sampled
        )
        assert oracle["aggregate_hbm_bytes"] == (
            contract["static_hbm_bytes"]
            + contract["hbm_bytes_per_sequence"] * sequences
            + contract["hbm_bytes_per_kv_token"] * kv_tokens
        )


def test_qwen38_v2_physical_sanity_uses_byte_over_rate_floors() -> None:
    sanity = _load(EXPECTATIONS)["physical_sanity"]
    rate = sanity["a100_hbm_bytes_per_second"]

    assert sanity["batch_1_hbm_floor_picoseconds"] == (
        sanity["recurrent_read_write_bytes_batch_1_per_layer"] * 10**12 // rate
    )
    assert sanity["batch_16_hbm_floor_picoseconds"] == (
        sanity["recurrent_read_write_bytes_batch_16_per_layer"] * 10**12 // rate
    )
    assert sanity["static_inventory_hbm_bytes"] < (
        sanity["bf16_checkpoint_payload_ceiling_bytes"]
    )
