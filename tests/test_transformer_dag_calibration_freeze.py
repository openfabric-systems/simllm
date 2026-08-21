"""Lock the authored transformer-dag-v1 calibration expectations.

These checks read authored data only. They do not import SimLLM, invoke a
subprocess, access hardware or a network, or write an output artifact.
"""

from __future__ import annotations

import ast
import json
import re
from itertools import pairwise
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SUITE_DIR = REPOSITORY / "offline" / "calibration" / "suites" / "transformer-dag-v1"
SUITE_JSON = SUITE_DIR / "suite.json"
EXPECTATIONS_JSON = SUITE_DIR / "expectations.json"
EXPECTATIONS_MD = SUITE_DIR / "expectations.md"
THIS_TEST = Path(__file__).resolve()

EM_DASH = "\u2014"
EXPECTED_SUITE_FILES = {"suite.json", "expectations.json", "expectations.md"}
FORBIDDEN_OUTPUT_KEYS = {
    "result",
    "results",
    "measurements",
    "samples",
    "durations_ps",
    "simulated_cycles",
    "residual",
    "record_id",
    "instance_graph_sha256",
    "template_graph_sha256",
}


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate authored key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def test_freeze_directory_contains_authored_inputs_only() -> None:
    assert SUITE_DIR.is_dir()
    assert {path.name for path in SUITE_DIR.iterdir()} == EXPECTED_SUITE_FILES
    assert all(path.is_file() for path in SUITE_DIR.iterdir())


def test_authored_json_has_no_duplicate_keys() -> None:
    for path in (SUITE_JSON, EXPECTATIONS_JSON):
        assert isinstance(_load(path), dict)

    try:
        json.loads('{"duplicate":1,"duplicate":2}', object_pairs_hook=_reject_duplicate_pairs)
    except ValueError as error:
        assert "duplicate authored key: duplicate" in str(error)
    else:
        raise AssertionError("strict authored JSON loader accepted a duplicate key")


def test_freeze_chronology_and_nonclosure_are_explicit() -> None:
    freeze = _load(EXPECTATIONS_JSON)
    chronology = freeze["chronology"]
    assert chronology == {
        "class": "expectations-only",
        "authored_before_implementation": True,
        "authored_before_hardware_execution": True,
        "authored_before_simulator_execution": True,
        "contains_generated_values": False,
        "contains_measured_values": False,
    }
    expected_tasks = [
        "COMP-1",
        "COMP-4",
        "COMP-5",
        "COMP-6",
        "COMP-10",
        "COMP-13",
        "COMP-17",
        "COMP-22",
        "COMP-23",
        "COMP-24",
        "COMP-25",
        "COMP-35",
        "COMP-41",
        "COMP-43",
        "COMP-45",
        "COMP-47",
        "COMP-48",
        "COMP-49",
        "COMP-50",
        "COMP-51",
        "COMP-52",
        "CORE-8",
        "CORE-11",
        "CORE-12",
        "CORE-13",
        "CORE-26",
        "CORE-27",
        "CORE-45",
        "CORE-50",
        "VLLM-12",
        "SGL-10",
        "SGL-24",
    ]
    assert freeze["closes"] == []
    assert freeze["tasks"] == expected_tasks
    assert freeze["does_not_close"] == expected_tasks


def test_machine_freezes_use_no_float_tokens_or_output_fields() -> None:
    for path in (SUITE_JSON, EXPECTATIONS_JSON):
        document = _load(path)
        assert not any(isinstance(value, float) for value in _walk(document))
        keys = {key for key in _keys(document) if key in FORBIDDEN_OUTPUT_KEYS}
        assert not keys


def test_reference_model_and_framework_identities_are_exact() -> None:
    suite = _load(SUITE_JSON)
    model = suite["reference_model"]
    assert suite["schema"] == "simllm-transformer-dag-suite-v1"
    assert suite["suite"] == "transformer-dag-v1"
    assert suite["state"] == "authored-inputs-only"
    assert model == {
        "name": "ibm-granite/granite-3.0-1b-a400m-instruct",
        "revision": "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445",
        "config_sha256": ("ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b"),
        "weight_sha256": ("f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc"),
        "weight_bytes": 2_669_283_096,
        "dtype": "bfloat16",
        "quantization": "none",
        "geometry": {
            "layers": 24,
            "hidden_size": 1024,
            "intermediate_size": 512,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_size": 64,
            "num_experts": 32,
            "top_k": 8,
            "vocab_size": 49_155,
        },
    }
    assert suite["frameworks"] == [
        {
            "id": "vllm",
            "version": "0.26.0",
            "source_commit": "568afb3a13806beb53bb2e6bd518269357b237c0",
        },
        {
            "id": "sglang",
            "version": "0.0.0.dev1+g8f2a3ad6d",
            "source_commit": "8f2a3ad6d7d68c58ae65b61a75bb2115449addca",
            "source_tree": "5be26db1f559064c0f9e724e78c1a8f619754867",
        },
    ]
    assert suite["launch_modes"] == [
        {"id": "eager", "requirement": "required"},
        {
            "id": "captured-graph",
            "requirement": "required-when-backend-reports-supported",
            "vendor_spelling": "recorded-by-collector",
        },
    ]


def test_observation_expansion_is_exact() -> None:
    suite = _load(SUITE_JSON)
    assert suite["observation_expansion"] == {
        "context_axes": ["target", "framework", "launch_mode"],
        "target_rule": (
            "every-capability-matrix-target-using-its-supported-real-silicon-collector"
        ),
        "framework_rule": "every-frozen-framework-row",
        "launch_mode_rule": (
            "eager-required-and-captured-graph-required-when-backend-reports-supported"
        ),
        "graph_rule": "every-graph-cell-in-every-required-context",
        "communication_rule": (
            "every-communication-cell-in-every-required-context-with-authored-"
            "participant-capability-ready"
        ),
        "mixed_rule": (
            "every-mixed-cell-in-every-required-context-where-all-member-capabilities-are-ready"
        ),
        "unavailable_rule": (
            "blocked-not-applicable-or-rejected-produces-no-observation-and-never-a-zero"
        ),
        "claim_rule": (
            "an-unobserved-required-context-cannot-contribute-to-a-validated-capability"
        ),
        "denominator_rule": (
            "graph-communication-and-mixed-denominators-count-authored-"
            "topology-cells-not-expanded-observations"
        ),
    }
    assert suite["parameterized_target_slots"] == [
        {
            "id": "amd-rocm-target",
            "binding_rule": (
                "bind-to-one-exact-immutable-concrete-target-and-operating-"
                "envelope-before-the-first-observation"
            ),
            "isolation_rule": (
                "different-concrete-targets-cannot-share-an-authored-"
                "denominator-evidence-split-or-device-model"
            ),
        }
    ]


def test_token_fixture_and_routing_contract_are_exact() -> None:
    assert _load(SUITE_JSON)["token_fixture_contract"] == {
        "schema": "simllm-calibration-token-fixture-v1",
        "case_ordinal_rule": "zero-based-graph-cells-array-order",
        "request_ordinal_rule": "zero-based-within-case",
        "request_id_rule": "<case-id>:request:<request-ordinal>",
        "request_count_rule": ("requests-for-compute-prefill-and-batch-for-decode-families"),
        "sequence_length_rule": (
            "prompt-tokens-per-request-for-compute-prefill-and-context-tokens-for-decode-families"
        ),
        "token_formula": (
            "token_id = 256 + ((104729 * case_ordinal + 8191 * "
            "request_ordinal + 131 * position) mod (vocab_size - 256))"
        ),
        "content_address_rule": (
            "external-fixture-id-is-lowercase-sha256-of-the-canonical-token-"
            "fixture-record-and-is-not-an-authored-output-in-this-freeze"
        ),
        "attention_rule": "all-tokens-active",
        "position_rule": ("position-ids-are-zero-through-sequence-length-minus-one"),
        "padding_rule": "no-padding-beyond-authored-shape",
        "sampling_rule": ("greedy-temperature-zero-with-no-random-number-generator"),
        "decode_rule": ("one-decode-forward-and-output-token-is-not-recursively-consumed"),
        "routing_sidecar_schema": "simllm-moe-routing-sidecar-v1",
        "routing_sidecar_fields": [
            "case_id",
            "request_id",
            "layer_ordinal",
            "token_position",
            "ordered_expert_ids",
        ],
        "routing_rule": (
            "capture-records-exact-top-k-moe-routing-and-divergence-within-one-"
            "claimed-envelope-is-fatal-or-requires-a-separate-envelope"
        ),
    }


def test_graph_cell_ids_axes_and_splits_are_frozen() -> None:
    suite = _load(SUITE_JSON)
    source_cells = suite["graph_cells"]
    assert len(source_cells) == 15
    assert len({cell["id"] for cell in source_cells}) == len(source_cells)
    cells = {cell["id"]: cell for cell in source_cells}
    expected = {
        "cp-train-r4-t128": ("compute-prefill", "train", 128),
        "cp-train-r4-t768": ("compute-prefill", "train", 768),
        "cp-train-r4-t2048": ("compute-prefill", "train", 2048),
        "cp-validation-r4-t512": ("compute-prefill", "validation", 512),
        "cp-test-r4-t1024": ("compute-prefill", "test", 1024),
        "md-train-b4-c128": ("memory-decode", "train", 128),
        "md-train-b4-c1024": ("memory-decode", "train", 1024),
        "md-train-b4-c8192": ("memory-decode", "train", 8192),
        "md-validation-b4-c512": ("memory-decode", "validation", 512),
        "md-test-b4-c2048": ("memory-decode", "test", 2048),
        "mc-train-b1-c2048": ("moe-communication-decode", "train", 1),
        "mc-train-b16-c2048": ("moe-communication-decode", "train", 16),
        "mc-train-b64-c2048": ("moe-communication-decode", "train", 64),
        "mc-validation-b4-c2048": (
            "moe-communication-decode",
            "validation",
            4,
        ),
        "mc-test-b8-c2048": ("moe-communication-decode", "test", 8),
    }
    assert set(cells) == set(expected)
    for identifier, (family, split, axis_value) in expected.items():
        cell = cells[identifier]
        assert cell["family"] == family
        assert cell["split"] == split
        if family == "compute-prefill":
            assert cell["total_prompt_tokens"] == axis_value
            assert cell["requests"] == 4
            assert cell["prompt_tokens_per_request"] * 4 == axis_value
        elif family == "memory-decode":
            assert cell["context_tokens"] == axis_value
            assert cell["batch"] == 4
        else:
            assert cell["batch"] == axis_value
            assert cell["context_tokens"] == 2048
            assert cell["expert_participants"] == 4
            assert cell["parallelism_override"] == {"expert": 4}


def test_communication_matrix_has_three_axes_and_no_duplicate_cell() -> None:
    suite = _load(SUITE_JSON)
    assert suite["communication_contract"] == {
        "supported_operation_algorithms": [
            {
                "operation": "all-reduce",
                "algorithm_hint": "ring",
                "scalar_payload_convention": "full-reduced-input-bytes-per-rank",
            },
            {
                "operation": "all-to-allv",
                "algorithm_hint": "pairwise",
                "scalar_payload_convention": ("bytes-per-ordered-source-destination-pair"),
            },
        ],
        "pair_payload_bytes_rule": ("empty-table-uses-the-scalar-payload-convention"),
        "byte_floor_rule": (
            "derive-peer-port-service-bytes-from-operation-algorithm-and-"
            "payload-convention-before-EQ4"
        ),
        "rank_rule": "ranks-are-zero-through-participants-minus-one",
        "placement_rule": "rank-i-maps-to-node-i-and-local-gpu-zero",
        "locality": "cross-node",
        "transport_class": "vendor-native-gpu-direct-rdma",
        "unsupported_rule": (
            "a-campaign-without-the-required-placement-or-transport-declares-"
            "communication-unsupported"
        ),
        "channel_count_rule": ("every-cell-declares-a-positive-integer-channel-count"),
        "channel_axis_rule": (
            "at-four-participants-and-1048576-payload-bytes-train-uses-one-and-"
            "four-channels-validation-uses-two-and-test-uses-three"
        ),
    }
    cells = suite["communication_cells"]
    assert len(cells) == 20
    assert len({cell["id"] for cell in cells}) == 20
    assert {cell["id"] for cell in cells} == {
        "comm-all-reduce-p4-b65536-ch2",
        "comm-all-reduce-p4-b16777216-ch2",
        "comm-all-reduce-p4-b1048576-ch2",
        "comm-all-reduce-p4-b4194304-ch2",
        "comm-all-reduce-p2-b1048576-ch2",
        "comm-all-reduce-p8-b1048576-ch2",
        "comm-all-reduce-p6-b1048576-ch2",
        "comm-all-reduce-p4-b1048576-ch1",
        "comm-all-reduce-p4-b1048576-ch4",
        "comm-all-reduce-p4-b1048576-ch3",
        "comm-all-to-allv-p4-b65536-ch2",
        "comm-all-to-allv-p4-b16777216-ch2",
        "comm-all-to-allv-p4-b1048576-ch2",
        "comm-all-to-allv-p4-b4194304-ch2",
        "comm-all-to-allv-p2-b1048576-ch2",
        "comm-all-to-allv-p8-b1048576-ch2",
        "comm-all-to-allv-p6-b1048576-ch2",
        "comm-all-to-allv-p4-b1048576-ch1",
        "comm-all-to-allv-p4-b1048576-ch4",
        "comm-all-to-allv-p4-b1048576-ch3",
    }
    assert all(cell["channel_count"] > 0 for cell in cells)
    expected_algorithms = {"all-reduce": "ring", "all-to-allv": "pairwise"}
    for operation, algorithm_hint in expected_algorithms.items():
        rows = [cell for cell in cells if cell["operation"] == operation]
        assert len(rows) == 10
        assert {cell["algorithm_hint"] for cell in rows} == {algorithm_hint}
        if operation == "all-to-allv":
            assert all(cell["pair_payload_bytes"] == [] for cell in rows)
        else:
            assert all("pair_payload_bytes" not in cell for cell in rows)
        payload_rows = [cell for cell in rows if "payload" in cell["axis_roles"]]
        participant_rows = [cell for cell in rows if "participants" in cell["axis_roles"]]
        assert {
            (row["participants"], row["payload_bytes"], row["split"]) for row in payload_rows
        } == {
            (4, 65_536, "train"),
            (4, 16_777_216, "train"),
            (4, 1_048_576, "validation"),
            (4, 4_194_304, "test"),
        }
        payloads = sorted(row["payload_bytes"] for row in payload_rows)
        assert all(high % low == 0 for low, high in pairwise(payloads))
        assert {
            (row["participants"], row["payload_bytes"], row["split"]) for row in participant_rows
        } == {
            (2, 1_048_576, "train"),
            (8, 1_048_576, "train"),
            (4, 1_048_576, "validation"),
            (6, 1_048_576, "test"),
        }
        channel_rows = [cell for cell in rows if "channel_count" in cell["axis_roles"]]
        assert {
            (row["participants"], row["payload_bytes"], row["channel_count"], row["split"])
            for row in channel_rows
        } == {
            (4, 1_048_576, 1, "train"),
            (4, 1_048_576, 4, "train"),
            (4, 1_048_576, 2, "validation"),
            (4, 1_048_576, 3, "test"),
        }


def test_mixed_matrix_fixes_all_arms_widths_and_splits() -> None:
    suite = _load(SUITE_JSON)
    matrix = suite["mixed_matrix"]
    assert matrix["start_pattern"] == "simultaneous"
    assert matrix["canonical_member_order"] == [
        "compute",
        "memory",
        "communication",
    ]
    assert [row["value"] for row in matrix["widths"]] == [1, 4, 2, 3]
    assert [row["split"] for row in matrix["widths"]] == [
        "train",
        "train",
        "validation",
        "test",
    ]
    assert matrix["arms"] == [
        {"id": "mix-compute", "members": ["compute"]},
        {"id": "mix-memory", "members": ["memory"]},
        {"id": "mix-communication", "members": ["communication"]},
        {"id": "mix-compute-memory", "members": ["compute", "memory"]},
        {
            "id": "mix-compute-communication",
            "members": ["compute", "communication"],
        },
        {
            "id": "mix-memory-communication",
            "members": ["memory", "communication"],
        },
        {
            "id": "mix-compute-memory-communication",
            "members": ["compute", "memory", "communication"],
        },
    ]
    assert matrix["cell_id_rule"] == "<arm-id>-w<width>"
    assert matrix["width_semantics"] == "number-of-copies-per-arm-member"
    assert matrix["copy_id_rule"] == "<cell-id>:<member>:<rep>"
    assert matrix["flattening_order"] == (
        "member-major-in-canonical-member-order-then-zero-based-repetition-ordinal"
    )
    assert matrix["timestamp_rule"] == ("all-copies-have-identical-submitted-at-and-eligible-at")
    assert matrix["device_rule"] == "all-copies-share-one-device-instance"
    assert matrix["batch_order_rule"] == (
        "flattened-copy-tuple-is-the-exact-BatchKernelService-request-order"
    )
    assert matrix["representative_selection"] == {
        "compute": "greatest-authored-flops-among-captured-dense-gemm-bindings",
        "memory": (
            "greatest-authored-compulsory-bytes-among-captured-decode-attention-"
            "or-streaming-bindings"
        ),
        "communication": ("greatest-logical-payload-among-captured-collective-bindings"),
        "tie_break": "canonical-implementation-identity",
    }
    identities = {
        f"{arm['id']}-w{width['value']}" for arm in matrix["arms"] for width in matrix["widths"]
    }
    assert len(identities) == 28


def test_split_rules_keep_repetitions_and_units_isolated() -> None:
    rules = _load(SUITE_JSON)["split_rules"]
    assert rules == {
        "kernel_unit": ["implementation_id", "shape_vector"],
        "kernel_membership_precedence": ["train", "validation", "test"],
        "duplicate_rule": (
            "assign-every-occurrence-to-the-first-present-split-in-precedence-order"
        ),
        "repetition_rule": "all-repetitions-of-one-cell-stay-in-one-split",
        "graph_unit": ["template_graph_sha256", "case_id"],
        "support_rule": "derive-support-from-train-only",
        "validation_rule": "select-model-form-source-policy-and-uncertainty-only",
        "test_rule": "open-only-after-fit-and-validation-choices-are-frozen",
        "tail_rule": (
            "stable-singleton-exact-entry-only-with-no-interpolation-or-concurrent-sharing"
        ),
        "mandatory_test_rule": "at-least-one-unique-test-unit-per-claimed-family",
    }


def test_support_matrix_and_upstream_pins_are_exact() -> None:
    freeze = _load(EXPECTATIONS_JSON)
    upstream = freeze["upstream_accel_sim"]
    assert upstream["qualified_development_pin"] == ("3016c658f810bdae9a14bf4534ee99e9945eedae")
    assert upstream["release_v1_3_0_pin"] == ("c5296df152c99a28dd64e5d9560bd58a8fd2e774")
    assert upstream["statistics_archive_pin"] == ("ee21104be44ad55dfde789111d3b94372be8435f")
    assert upstream["gpgpu_sim_pin"] == ("6c3cf4ff32110908386d605a7034fc67666a92de")
    assert upstream["a100_reproduction_gate"] == (
        "archive-golden-conformance-until-site-local-inputs-are-acquired-and-hash-locked"
    )
    assert freeze["selective_fill_expectation"] == {
        "case_id": "a100-sm80-exact-only-gap",
        "target": "nvidia-a100-sm80",
        "query_rule": (
            "inside-train-defined-target-and-qualified-sm80-envelopes-and-"
            "strictly-between-real-silicon-anchors"
        ),
        "exact_silicon_row": "absent",
        "implementation_region_policy": "exact-only-categorical",
        "silicon_fit_applicability": (
            "inapplicable-with-no-validated-fit-entry-for-the-exact-region"
        ),
        "source_precedence": [
            "exact-silicon",
            "validated-silicon-fit",
            "qualified-accel-sim",
        ],
        "expected_selected_source": "qualified-accel-sim",
        "counterfactual_rule": (
            "a-valid-silicon-fit-for-that-exact-region-outranks-qualified-accel-sim"
        ),
        "promotion_rule": (
            "qualified-accel-sim-fill-alone-never-promotes-and-validated-status-"
            "additionally-requires-qualified-sidecar-replay-and-correlation-"
            "untouched-physical-test-evidence-and-live-ttft-tpot-evidence"
        ),
        "extrapolation_rule": "forbidden",
    }
    assert freeze["capability_matrix"] == [
        {
            "target": "nvidia-a100-sm80",
            "cuda_silicon_collector": "supported",
            "rocm_silicon_collector": "reject",
            "accel_sim": "conditional-supported-cells-only",
            "compact_model_compiler": "supported",
        },
        {
            "target": "nvidia-h100-sm90",
            "cuda_silicon_collector": "supported",
            "rocm_silicon_collector": "reject",
            "accel_sim": "reject",
            "compact_model_compiler": "supported",
        },
        {
            "target": "amd-rocm-target",
            "cuda_silicon_collector": "reject",
            "rocm_silicon_collector": "supported",
            "accel_sim": "reject",
            "compact_model_compiler": "supported",
        },
    ]


def test_canonical_byte_and_model_identity_contracts_are_exact() -> None:
    freeze = _load(EXPECTATIONS_JSON)
    assert freeze["canonical_byte_contract"] == {
        "schema": "simllm-calibration-canonical-bytes-v1",
        "encoding": "utf-8-without-bom",
        "unicode_normalization": "NFC",
        "object_key_order": ("lexicographic-by-normalized-unicode-scalar-sequence"),
        "array_order": "schema-defined-and-preserved",
        "duplicate_key_rule": "reject-before-and-after-unicode-normalization",
        "string_scalar_rule": "reject-unpaired-surrogates",
        "string_escape_rule": (
            "escape-quote-backslash-and-five-json-short-controls-use-lowercase-"
            "\\u00xx-for-other-u0000-through-u001f-never-escape-slash-and-emit-"
            "other-scalars-as-utf8"
        ),
        "integer_rule": "minimal-base-10-with-no-plus-leading-zero-or-negative-zero",
        "boolean_null_rule": "lowercase-json-literals",
        "whitespace_rule": "no-insignificant-whitespace",
        "terminal_newline": False,
        "exact_quantity_rule": (
            "picoseconds-cycles-bytes-counts-and-flops-are-json-integer-tokens-"
            "parsed-losslessly-before-schema-bounds"
        ),
        "decimal_rule": (
            "fitted-coefficients-are-precision-declared-decimal-strings-with-"
            "no-exponent-plus-redundant-zero-or-negative-zero"
        ),
        "record_identity_rule": ("lowercase-hex-sha256-of-exact-canonical-record-bytes"),
        "identity_member_rule": "external-record-identifier-is-not-a-json-member",
        "full_domain_authority": ("python-with-locked-python-and-unicode-database-versions"),
        "python_runtime": "CPython-3.10",
        "unicode_database_version": "13.0.0",
        "ascii_subset_rule": (
            "independent-cpp17-verifier-rejects-nonascii-and-covers-structural-"
            "types-duplicate-rejection-control-escapes-arbitrary-integers-and-"
            "project-sha256"
        ),
    }
    assert freeze["model_identity_contract"] == {
        "acceptance_status_values": ["candidate", "validated"],
        "target_basis_values": ["target-silicon", "architecture-derived"],
        "wire_value_rule": "closed-exact-lowercase-values-with-no-aliases",
        "cross_field_rule": (
            "architecture-derived-requires-candidate-and-validated-plus-"
            "architecture-derived-rejects"
        ),
        "runtime_selection_rule": (
            "validated-by-default-and-candidate-only-under-explicit-experimental-opt-in"
        ),
        "run_provenance_rule": ("copies-acceptance-status-and-target-basis-unchanged"),
    }


def test_dispatch_context_and_signature_contracts_are_exact() -> None:
    assert _load(EXPECTATIONS_JSON)["dispatch_contract"] == {
        "trait_schema": "simllm-typed-dispatch-trait-v1",
        "trait_fields": ["trait_id", "value_type", "value"],
        "trait_value_type_values": ["integer", "string", "boolean"],
        "trait_value_rule": "value-matches-value-type-and-trait-id-is-nonblank",
        "trait_tuple_rule": "sorted-unique-by-trait-id",
        "signature_schema": "simllm-dispatch-signature-v1",
        "signature_fields": [
            "schema",
            "framework_id",
            "framework_version",
            "backend_id",
            "backend_version",
            "kernel_library_id",
            "kernel_library_version",
            "algorithm_policy_id",
            "device_isa",
            "numeric_traits",
            "layout_traits",
        ],
        "signature_string_rule": "all-string-fields-are-nonblank",
        "signature_trait_rule": ("numeric-traits-and-layout-traits-are-strict-typed-trait-tuples"),
        "signature_launch_mode_rule": "launch-mode-is-forbidden",
        "context_schema": "simllm-device-dispatch-context-v1",
        "context_fields": [
            "schema",
            "instance_graph_sha256",
            "rank_device_assignments",
            "selected_device_models",
        ],
        "rank_assignment_fields": ["rank", "device_instance_id"],
        "rank_assignment_rule": (
            "strict-records-sorted-unique-by-rank-and-total-over-graph-participant-ranks"
        ),
        "selected_model_fields": [
            "device_instance_id",
            "device_model_id",
            "device_model_sha256",
            "dispatch_signature_sha256",
        ],
        "selected_model_rule": (
            "strict-records-sorted-unique-by-device-instance-id-with-exactly-the-"
            "used-devices-and-no-extras"
        ),
        "resolved_set_reference_rule": (
            "operation-and-collective-resolved-sets-reference-the-sha256-of-the-"
            "exact-same-dispatch-context-record"
        ),
        "closure_cross_validation_rule": (
            "binding-closure-and-both-present-resolved-sets-have-one-identical-"
            "instance-graph-sha256-and-the-present-sets-have-one-identical-"
            "dispatch-context-sha256"
        ),
        "provenance_cross_validation_rule": (
            "run-provenance-device-model-tuples-match-context-selections-and-"
            "every-resolved-set-member-device-model-pair"
        ),
    }


def test_template_graph_canonicalization_contract_is_exact() -> None:
    freeze = _load(EXPECTATIONS_JSON)
    assert freeze["interfaces"]["execution_graph_template"] == (
        "simllm-execution-graph-template-v1"
    )
    assert freeze["template_graph_contract"] == {
        "instance_schema": "simllm-execution-graph-v1",
        "instance_hash_rule": (
            "instance-graph-sha256-hashes-exact-canonical-unbound-instance-bytes"
        ),
        "instance_writer_rule": (
            "calibration-canonical-writer-serializes-the-exact-strict-simllm-"
            "execution-graph-v1-json-object"
        ),
        "unbound_compute_null_fields": [
            "nominal_duration_ps",
            "uncertainty_fraction",
        ],
        "calibration_float_rule": (
            "reject-any-float-valued-ComputeWork-config-scalar-because-the-"
            "calibration-canonical-grammar-has-no-float-spelling"
        ),
        "graph_reader_rule": ("existing-simllm-execution-graph-v1-reader-remains-unchanged"),
        "template_schema": "simllm-execution-graph-template-v1",
        "writer": "calibration-canonical-writer",
        "top_level_fields": [
            "schema",
            "operations",
            "completion_operation_ordinals",
            "collective_plans",
        ],
        "empty_collective_rule": (
            "collective-plans-is-present-as-an-empty-tuple-when-no-plan-exists"
        ),
        "operation_fields": [
            "rank_ordinal",
            "logical_queue_ordinal",
            "priority",
            "work",
            "depends_on_operation_ordinals",
            "participant_local_depends_on_operation_ordinals",
        ],
        "operation_ordinal_rule": (
            "array-position-follows-original-execution-graph-operation-tuple-order"
        ),
        "rank_normalization_rule": (
            "ascending-union-of-operation-anchors-collective-participants-"
            "control-destinations-accepted-dma-endpoints-and-plan-ranks-maps-to-"
            "zero-through-rank-count-minus-one"
        ),
        "queue_normalization_rule": (
            "within-each-normalized-rank-logical-queues-map-to-ordinals-by-"
            "first-operation-occurrence"
        ),
        "dependency_rule": ("rewrite-both-dependency-tuples-to-sorted-unique-operation-ordinals"),
        "completion_frontier_rule": (
            "empty-source-frontier-normalizes-to-all-operation-ordinals-and-an-"
            "explicit-frontier-to-a-sorted-unique-tuple"
        ),
        "work_variant_fields": {
            "compute": ["kind", "kernel"],
            "kv-cache": ["kind", "action"],
            "dma": ["kind", "source_role", "destination_role"],
            "collective": [
                "kind",
                "collective",
                "algorithm_hint",
                "rank_ordinals",
                "channel_ordinal",
                "pair_rank_ordinals",
            ],
            "control": [
                "kind",
                "mode",
                "message",
                "destination_rank_ordinals",
            ],
        },
        "work_variant_rule": "closed-strict-union-with-kind-as-the-exact-discriminator",
        "algorithm_hint_rule": "nullable-member-remains-present",
        "semantic_rank_tuple_rule": ("collective-and-control-rank-tuples-preserve-source-order"),
        "collective_pair_rule": (
            "each-pair-is-an-exact-two-integer-array-of-source-rank-ordinal-and-"
            "destination-rank-ordinal-in-source-aggregate-pair-order-with-bytes-"
            "removed-and-an-empty-table-remains-empty"
        ),
        "effective_channel_rule": (
            "channel-hint-or-canonical-default-maps-to-a-graph-global-ordinal-"
            "by-first-collective-operation-occurrence-and-preserves-channel-"
            "equality-while-removing-spelling"
        ),
        "dma_endpoint_role_schema": "simllm-device-endpoint-role-v1",
        "dma_endpoint_role_values": [
            "host",
            "host:pinned",
            "host:pageable",
            "gpu:<rank>",
            "gpu:<rank>:hbm",
            "cuda:<rank>",
        ],
        "dma_rank_rule": (
            "gpu-and-cuda-rank-tokens-rewrite-through-the-same-rank-map-and-must-resolve"
        ),
        "dma_unknown_role_rule": "reject",
        "collective_plan_fields": [
            "operation_ordinal",
            "algorithm",
            "channel_ordinal",
            "rank_order",
            "rounds",
            "actions",
            "extents",
            "entry_action_ordinals",
            "terminal_action_ordinals",
        ],
        "collective_plan_order_rule": (
            "source-plan-tuple-order-which-graph-validation-aligns-to-collective-operation-order"
        ),
        "plan_rank_order_rule": ("normalized-rank-ordinals-preserving-source-rank-order"),
        "plan_channel_rule": (
            "channel-ordinal-repeats-the-semantic-collective-effective-channel-as-a-loss-check"
        ),
        "transfer_channel_rule": (
            "round-transfer-channels-use-a-separate-graph-global-ordinal-"
            "namespace-by-first-plan-and-round-occurrence"
        ),
        "round_fields": ["transfer_channel_ordinal"],
        "round_ordinal_rule": "array-position-in-the-source-round-tuple",
        "action_fields": [
            "rank_ordinal",
            "kind",
            "extent_ordinal",
            "depends_on_action_ordinals",
        ],
        "action_dependency_rule": ("depends-on-action-ordinals-is-a-sorted-unique-tuple"),
        "extent_fields": [
            "round_ordinal",
            "source_rank_ordinal",
            "destination_rank_ordinal",
            "send_action_ordinal",
            "receive_action_ordinal",
        ],
        "frontier_fields": ["rank_ordinal", "action_ordinals"],
        "frontier_rule": (
            "entry-and-terminal-maps-follow-plan-rank-order-and-each-action-tuple-is-sorted"
        ),
        "plan_ordinal_rule": ("action-and-extent-array-positions-follow-their-source-tuple-order"),
        "plan_reference_rule": (
            "every-operation-round-action-extent-entry-and-terminal-reference-is-"
            "ordinal-rewritten-and-total"
        ),
        "excluded_fields": [
            "execution_identities",
            "step_identities",
            "operation_identities",
            "queue_identities",
            "channel_identities",
            "action_identities",
            "extent_identities",
            "request_identities",
            "pool_identities",
            "block_identities",
            "descriptor_identities",
            "correlation_identities",
            "release_timestamps",
            "not_before_timestamps",
            "placement_epochs",
            "compute_config",
            "shape_values",
            "compute_flops",
            "compute_hbm_bytes",
            "compute_nominal_duration",
            "compute_uncertainty",
            "other_kv_fields",
            "dma_bytes",
            "collective_payloads",
            "control_payloads",
            "plan_payloads",
            "extent_payloads",
            "service_values",
            "demand_values",
            "request_attribution",
            "round_tags",
            "round_indices",
            "integrity_hashes",
        ],
        "invariance_guard": (
            "excluded-only-changes-consistent-ordinalized-identity-renames-"
            "dependency-completion-or-frontier-action-set-tuple-permutations-"
            "empty-versus-explicit-all-completion-and-null-versus-default-"
            "channel-spelling-preserve-the-template-hash"
        ),
        "sensitivity_guard": (
            "changing-any-retained-source-tuple-order-except-explicitly-sorted-"
            "dependency-completion-and-frontier-action-set-fields-or-changing-"
            "priority-dependency-scope-effective-completion-retained-rank-or-"
            "queue-equivalence-work-kind-or-family-dma-role-collective-rank-"
            "order-sparse-pair-support-channel-sharing-plan-presence-or-any-"
            "retained-round-action-extent-or-frontier-edge-changes-the-template-hash"
        ),
        "rank_relabel_rule": (
            "rank-relabeling-is-invariant-only-when-source-rank-order-is-preserved"
        ),
        "projection_guard": (
            "projection-is-idempotent-leaks-no-raw-identity-and-rejects-an-"
            "unknown-role-or-unresolved-reference"
        ),
        "selection_rule": "template-hash-groups-splits-and-never-selects-service",
    }


def test_collective_device_stage_and_binding_closure_contracts_are_exact() -> None:
    freeze = _load(EXPECTATIONS_JSON)
    assert freeze["interfaces"]["operation_implementation_binding"] == (
        "OperationImplementationBinding"
    )
    assert freeze["interfaces"]["resolved_operation_service_binding_set"] == (
        "ResolvedOperationServiceBindingSet"
    )
    assert freeze["interfaces"]["collective_device_stage_binding"] == (
        "CollectiveDeviceStageBinding"
    )
    assert freeze["interfaces"]["resolved_collective_device_stage_set"] == (
        "ResolvedCollectiveDeviceStageSet"
    )
    assert freeze["interfaces"]["resolved_device_binding_closure"] == (
        "ResolvedDeviceBindingClosure"
    )
    assert freeze["operation_binding_contract"] == {
        "binding_type": "OperationImplementationBinding",
        "binding_fields": [
            "instance_graph_sha256",
            "operation_id",
            "launch_ordinal",
            "implementation_ref",
            "shape_vector",
        ],
        "binding_identity_fields": [
            "instance_graph_sha256",
            "operation_id",
            "launch_ordinal",
        ],
        "observed_fields": ["implementation_ref", "shape_vector"],
        "binding_forbidden_fields": [
            "measured_demand",
            "resource_vector",
            "fixed_floor_ps",
            "service_entry_id",
        ],
        "binding_rule": "capture-evidence-only-with-no-demand-or-service-entry",
        "resolved_set_schema": ("simllm-resolved-operation-service-binding-set-v1"),
        "resolved_set_fields": [
            "schema",
            "instance_graph_sha256",
            "dispatch_context_sha256",
            "bindings",
        ],
        "resolved_type": "ResolvedOperationServiceBinding",
        "resolved_fields": [
            "instance_graph_sha256",
            "operation_id",
            "launch_ordinal",
            "device_instance_id",
            "device_model_sha256",
            "semantic_key",
            "shape_vector",
            "implementation_ref",
            "service_entry_id",
            "resolution_source",
            "observed_implementation_binding_sha256",
        ],
        "resolution_source_values": ["observed-binding", "selector"],
        "observed_binding_rule": (
            "observed-implementation-binding-sha256-is-a-required-nullable-field-"
            "required-nonnull-for-observed-binding-and-null-for-selector"
        ),
        "binding_order_rule": "graph-operation-tuple-order-then-launch-ordinal",
        "resolver_selected_models_rule": (
            "resolver-consumes-a-total-selected-model-tuple-keyed-by-device-"
            "instance-and-every-binding-device-model-pair-matches-it"
        ),
        "set_rule": "fresh-immutable-total-set-resolves-before-runtime-state-mutates",
    }
    assert freeze["collective_device_stage_contract"] == {
        "binding_type": "CollectiveDeviceStageBinding",
        "binding_fields": [
            "instance_graph_sha256",
            "collective_operation_id",
            "collective_plan_integrity_sha256",
            "rank",
            "launch_ordinal",
            "implementation_ref",
            "shape_vector",
        ],
        "binding_identity_fields": [
            "instance_graph_sha256",
            "collective_operation_id",
            "collective_plan_integrity_sha256",
            "rank",
            "launch_ordinal",
        ],
        "observed_fields": ["implementation_ref", "shape_vector"],
        "binding_forbidden_fields": [
            "measured_demand",
            "resource_vector",
            "fixed_floor_ps",
            "service_entry_id",
            "eligibility_frontiers",
            "consumer_frontiers",
        ],
        "binding_rule": ("observed-binding-carries-no-demand-frontier-or-service-entry"),
        "resolved_set_schema": ("simllm-resolved-collective-device-stage-set-v1"),
        "resolved_set_fields": [
            "schema",
            "instance_graph_sha256",
            "dispatch_context_sha256",
            "stages",
            "rank_frontiers",
        ],
        "resolved_type": "ResolvedCollectiveDeviceStage",
        "resolved_fields": [
            "instance_graph_sha256",
            "collective_operation_id",
            "collective_plan_integrity_sha256",
            "rank",
            "launch_ordinal",
            "device_instance_id",
            "device_model_sha256",
            "implementation_ref",
            "shape_vector",
            "service_entry_id",
            "resolution_source",
        ],
        "resolution_source_values": ["selector"],
        "resolution_rule": (
            "fresh-online-resolution-requires-no-historical-observed-binding-and-"
            "validation-compares-observed-and-resolved-records-separately"
        ),
        "selection_authority": "device-model-stage-selector-only",
        "selector_inputs": [
            "semantic_collective",
            "traffic_owned_plan_topology",
            "dispatch_context",
        ],
        "resolver_selected_models_rule": (
            "resolver-consumes-a-total-selected-model-tuple-keyed-by-device-"
            "instance-and-every-stage-device-model-pair-matches-it"
        ),
        "stage_order_rule": (
            "graph-collective-tuple-order-then-plan-rank-order-then-launch-ordinal"
        ),
        "frontier_order_rule": ("graph-collective-tuple-order-then-plan-rank-order"),
        "set_presence_rule": (
            "set-is-nonempty-and-a-graph-with-no-resolved-collective-stage-"
            "omits-it-and-uses-null-in-the-binding-closure"
        ),
        "v1_stage_cardinality_rule": (
            "exactly-one-resolved-resident-stage-per-plan-rank-and-multistage-rejects"
        ),
        "rank_frontier_type": "CollectiveDeviceRankFrontier",
        "rank_frontier_fields": [
            "collective_operation_id",
            "collective_plan_integrity_sha256",
            "rank",
            "ordered_stage_ordinals",
            "entry_action_ids",
            "terminal_action_ids",
        ],
        "rank_frontier_rule": (
            "exactly-one-per-plan-rank-with-one-ordered-stage-ordinal-and-entry-"
            "and-terminal-action-ids-copied-byte-identically-from-the-plan"
        ),
        "validation_rule": (
            "every-plan-rank-and-referenced-action-resolves-every-stage-is-used-"
            "exactly-once-and-no-extra-rank-action-or-stage-is-admitted"
        ),
        "v1_resource_rule": (
            "positive-stage-demand-is-permitted-only-on-device-internal-axes-"
            "and-rejects-on-peer-port-or-data-mover-axes"
        ),
        "submitted_at_rule": "parent-collective-launch-completion",
        "eligible_at_equation": (
            "eligible_at_ps = max(submitted_at_ps, rank_local_graph_predecessor_ready_at_ps)"
        ),
        "device_grant_rule": (
            "device-grant-releases-the-existing-plan-entry-actions-for-that-rank"
        ),
        "device_work_finished_rule": (
            "device-work-finished-at-is-when-throughput-demands-and-epoch-floors-"
            "finish-while-compute-retains-lifetime-residency-and-exclusive-"
            "reservations"
        ),
        "traffic_authority_rule": ("traffic-alone-owns-plan-actions-extents-and-network-service"),
        "traffic_terminal_equation": (
            "traffic_terminal_at_ps = max({eligible_at_ps} union "
            "{terminal_action_completed_at_ps for terminal_action_id in "
            "terminal_action_ids})"
        ),
        "rank_release_equation": (
            "rank_release_at_ps = max(device_work_finished_at_ps, traffic_terminal_at_ps)"
        ),
        "reservation_release_rule": (
            "collective-stage-uses-incremental-external-frontier-runtime-"
            "advances-through-rank-release-at-and-calls-compute-owned-release-"
            "held-which-returns-the-final-ServiceFact-directly-and-traffic-never-"
            "mutates-the-reservations"
        ),
        "rank_completion_equation": "rank_completed_at_ps = rank_release_at_ps",
        "parent_completion_equation": (
            "collective_completed_at_ps = max(rank_completed_at_ps for all plan ranks)"
        ),
        "acyclicity_rule": "no-plan-action-depends-on-rank-release",
        "authority_rule": (
            "device-stage-is-neither-ComputeWork-nor-an-independent-"
            "CompletionEvent-and-emits-no-graph-completion"
        ),
        "queue_visit_operation_rule": (
            "authoritative-stage-QueueVisit-remains-under-parent-collective-operation-id"
        ),
        "queue_visit_subject_rule": ("<operation-id>:rank:<rank>:stage:<launch-ordinal>"),
        "queue_visit_finished_rule": "finished-at-equals-rank-release-at",
        "queue_visit_completed_rule": "completed-at-equals-rank-release-at",
        "queue_visit_resource_ref": "legacy-GPU_SCHEDULER-ResourceRef",
        "axis_projection_rule": ("internal-service-axes-never-become-separate-queue-visits"),
        "service_event_rule": (
            "exactly-one-authoritative-composite-QueueVisit-and-no-independent-graph-completion"
        ),
        "lease_tail_rule": (
            "device-work-finished-at-to-rank-release-at-is-lease-held-occupancy-"
            "evidence-and-never-an-additive-kernel-or-network-latency-term"
        ),
        "bypass_rule": (
            "all-active-axis-demands-are-known-zero-in-every-epoch-and-every-"
            "epoch-floor-is-null-or-zero-or-disabled-bypass-emits-no-device-"
            "visit-delays-no-entry-and-preserves-accepted-artifacts-and-"
            "timestamps-byte-for-byte"
        ),
        "positive_floor_rule": "a-positive-epoch-floor-prevents-zero-demand-bypass",
    }
    assert freeze["binding_closure_contract"] == {
        "type": "ResolvedDeviceBindingClosure",
        "schema": "simllm-resolved-device-binding-closure-v1",
        "fields": [
            "schema",
            "instance_graph_sha256",
            "operation_service_binding_set_sha256",
            "collective_device_stage_set_sha256",
        ],
        "collective_stage_set_rule": (
            "nullable-and-null-when-absent-otherwise-hash-the-nonempty-"
            "resolved-collective-device-stage-set"
        ),
        "hash_rule": ("external-closure-id-is-sha256-of-the-exact-canonical-four-field-record"),
    }
    assert freeze["run_provenance_contract"] == {
        "schema": "simllm-run-provenance-v2",
        "byte_contract": ("core-v1-family-compact-utf8-json-plus-exactly-one-terminal-lf"),
        "step_result_ref_rule": (
            "StepResult-run-provenance-ref-sha256-hashes-the-full-provenance-"
            "bytes-including-the-terminal-lf"
        ),
        "calibration_record_distinction": (
            "calibration-canonical-records-remain-without-a-terminal-newline"
        ),
        "compatibility_rule": "preserve-all-run-provenance-v1-fields-unchanged",
        "compact_device_source_rule": (
            "inherited-source-schema-is-simllm-execution-graph-v1-and-inherited-"
            "source-sha256-equals-instance-graph-sha256-and-any-disagreement-rejects"
        ),
        "added_top_level_fields": [
            "instance_graph_sha256",
            "resolved_device_binding_closure_sha256",
            "device_models",
        ],
        "device_model_entry_fields": [
            "device_instance_id",
            "device_model_id",
            "device_model_sha256",
            "acceptance_status",
            "target_basis",
            "operating_envelope_sha256",
        ],
        "device_model_entry_rule": "exact-fields-with-no-extras",
        "device_model_order": "canonical-by-device-instance-id",
        "device_model_cardinality": "exactly-one-entry-per-device-instance",
        "copy_rule": (
            "acceptance-status-and-target-basis-copy-unchanged-from-the-selected-device-model"
        ),
        "heterogeneous_rule": "heterogeneous-device-model-entries-are-legal",
        "artifact_closure_rule": (
            "selected-device-model-and-operating-envelope-records-are-reachable-"
            "and-verified-in-the-result-artifact-closure"
        ),
    }


def test_batch_kernel_service_contract_is_exact() -> None:
    assert _load(EXPECTATIONS_JSON)["batch_kernel_service_contract"] == {
        "signature": (
            "dispatch_batch(requests: tuple[ResolvedDeviceServiceRequest,...], "
            "common_start_ps: int, snapshot: DeviceServiceSnapshot) -> "
            "BatchKernelServiceResult"
        ),
        "input_fields": ["requests", "common_start_ps", "snapshot"],
        "input_rule": ("inputs-are-immutable-requests-are-ordered-and-request-order-is-preserved"),
        "off_path_rule": (
            "scalar-and-legacy-off-paths-preserve-service-calls-composition-"
            "cursors-barriers-visits-reports-result-bytes-and-timestamps-byte-"
            "for-byte"
        ),
        "request_fields": ["subject_key", "service_entry_id", "release_mode"],
        "request_identity_rule": (
            "exact-fields-with-no-extras-subject-key-is-stable-service-entry-id-"
            "is-exact-and-release-mode-is-closed"
        ),
        "release_mode_values": ["work-finish", "external-frontier"],
        "batch_release_mode_rule": ("batch-accepts-only-work-finish-and-rejects-external-frontier"),
        "snapshot_fields": [
            "device_instance_id",
            "device_model_sha256",
            "registry_sha256",
            "resident_states",
        ],
        "snapshot_rule": ("exact-fields-with-no-extras-and-runtime-composition-cursors-excluded"),
        "resident_state_order": ["admission_sequence", "subject_key"],
        "v1_resident_rule": (
            "input-resident-states-and-successful-next-snapshot-resident-states-are-both-empty"
        ),
        "admission_rule": (
            "at-common-start-and-after-every-release-release-completed-"
            "reservations-first-then-scan-pending-in-original-request-order-and-"
            "admit-each-feasible-request-while-skipping-infeasible-requests"
        ),
        "same_time_rule": (
            "drain-zero-duration-completions-and-newly-feasible-requests-to-a-"
            "finite-same-time-fixed-point-before-time-advances"
        ),
        "preflight_rule": (
            "reject-a-request-whose-lifetime-reservation-maxima-exceed-device-capacity"
        ),
        "result_fields": [
            "service_facts",
            "device_accounting",
            "next_snapshot",
        ],
        "result_field_types": {
            "service_facts": "tuple[ServiceFact,...]",
            "device_accounting": "DeviceAccounting",
            "next_snapshot": "DeviceServiceSnapshot",
        },
        "service_fact_fields": [
            "subject_key",
            "epoch_index",
            "submitted_at",
            "eligible_at",
            "started_at",
            "work_finished_at",
            "finished_at",
            "completed_at",
        ],
        "service_fact_rule": "exact-fields-with-no-extras",
        "fact_coverage_rule": (
            "every-input-service-entry-epoch-is-contiguous-and-produces-exactly-"
            "one-fact-with-no-extra-subject-or-epoch-and-every-input-completes-"
            "exactly-once"
        ),
        "fact_order_rule": "request-tuple-order-then-ascending-epoch-index",
        "fact_time_rule": (
            "submitted-at-is-no-later-than-eligible-at-which-is-no-later-than-"
            "started-at-which-is-no-later-than-work-finished-at-which-is-no-"
            "later-than-finished-at-which-is-no-later-than-completed-at"
        ),
        "ordinary_batch_fact_rule": "work-finished-at-equals-finished-at",
        "first_epoch_rule": (
            "submitted-at-equals-eligible-at-equals-common-start-ps-and-started-"
            "at-equals-admission-grant"
        ),
        "later_epoch_rule": (
            "submitted-at-equals-eligible-at-equals-started-at-equals-prior-epoch-work-finished-at"
        ),
        "intermediate_epoch_rule": ("finished-at-equals-completed-at-equals-work-finished-at"),
        "ordinary_final_epoch_rule": ("finished-at-equals-completed-at-equals-work-finished-at"),
        "device_accounting_fields": [
            "registry_sha256",
            "admitted_throughput",
            "served_throughput",
            "acquired_reservations",
            "released_reservations",
            "held_reservation_ps",
        ],
        "device_accounting_rule": (
            "exact-fields-with-no-extras-throughput-vectors-are-aligned-reduced-"
            "rationals-and-reservation-vectors-are-aligned-integers"
        ),
        "accounting_zero_rule": (
            "throughput-totals-are-zero-on-nonthroughput-axes-and-reservation-"
            "and-held-reservation-totals-are-zero-on-throughput-axes"
        ),
        "held_reservation_rule": ("held-reservation-is-demand-units-times-lease-duration"),
        "accounting_conservation_rule": (
            "served-throughput-equals-admitted-throughput-released-reservations-"
            "equal-acquired-reservations-and-all-vectors-agree-with-input-"
            "service-entries"
        ),
        "next_snapshot_identity_fields": [
            "device_instance_id",
            "device_model_sha256",
            "registry_sha256",
        ],
        "next_snapshot_rule": "identities-equal-the-input-snapshot",
        "resident_leak_rule": "no-resident-state-leakage",
        "snapshot_authority_rule": ("compute-owned-with-exact-outer-fields-and-no-runtime-cursors"),
        "purity_rule": "no-input-mutation-callback-or-graph-completion",
        "commit_rule": (
            "runtime-validates-the-complete-result-before-atomically-adopting-"
            "next-snapshot-and-service-facts"
        ),
        "failure_rule": "failure-leaves-live-state-unchanged",
        "later_arrival_rule": (
            "later-arrival-methods-exist-only-on-the-incremental-device-service-transaction"
        ),
    }
    assert _load(EXPECTATIONS_JSON)["incremental_device_service_contract"] == {
        "begin_signature": (
            "IncrementalDeviceService.begin(snapshot) -> IncrementalDeviceServiceTransaction"
        ),
        "begin_rule": (
            "snapshot-is-the-only-begin-argument-and-owns-the-device-model-and-registry-identity"
        ),
        "admissible_signature": "admissible(request, now_ps) -> Feasibility",
        "dispatch_granted_signature": ("dispatch_granted(request, admission_sequence, now_ps)"),
        "peek_signature": "peek_next_event_ps() -> int | None",
        "advance_signature": "advance(to_ps) -> tuple[DeviceServiceEvent,...]",
        "event_union": ["WorkFinishedEvent", "ServiceFactEvent"],
        "work_finished_event_fields": [
            "kind",
            "subject_key",
            "epoch_index",
            "at_ps",
        ],
        "work_finished_event_kind": "work-finished",
        "service_fact_event_fields": ["kind", "fact"],
        "service_fact_event_kind": "service-fact",
        "service_fact_event_rule": "fact-is-one-strict-ServiceFact",
        "release_held_signature": ("release_held(subject_key, release_at_ps) -> ServiceFact"),
        "accounting_signature": "accounting() -> DeviceAccounting",
        "transaction_methods": ["prepare()", "commit()", "abort()"],
        "time_rule": (
            "transaction-time-never-decreases-and-dispatch-now-ps-equals-"
            "runtime-current-logical-time"
        ),
        "epoch_event_rule": (
            "intermediate-epochs-and-ordinary-final-epochs-emit-ServiceFactEvent-at-work-finish"
        ),
        "work_finish_rule": ("work-finish-releases-normally-and-emits-the-final-ServiceFactEvent"),
        "external_frontier_rule": (
            "external-frontier-retains-lifetime-reservations-after-work-finish-"
            "and-advance-emits-only-WorkFinishedEvent-instead-of-the-final-"
            "ServiceFactEvent-for-its-final-epoch"
        ),
        "release_held_rule": (
            "release-held-is-valid-exactly-once-only-for-an-external-frontier-"
            "final-epoch-and-only-after-that-subjects-WorkFinishedEvent-and-"
            "returns-the-final-ServiceFact-directly-with-no-staged-same-time-advance"
        ),
        "release_time_rule": (
            "release-at-ps-equals-transaction-current-time-after-runtime-"
            "advances-through-that-boundary-and-is-no-earlier-than-work-finish"
        ),
        "released_fact_rule": (
            "work-finished-at-preserves-the-earlier-boundary-and-finished-at-"
            "equals-completed-at-equals-release-at-ps"
        ),
        "traffic_order_rule": (
            "if-the-traffic-terminal-is-already-known-release-follows-work-"
            "finish-immediately-otherwise-runtime-advances-to-the-later-terminal-"
            "before-release"
        ),
        "held_wait_rule": (
            "peek-next-event-ps-none-with-a-held-subject-means-an-external-wait-"
            "and-never-global-quiescence-or-a-fatal-device-dead-state"
        ),
        "event_order_rule": (
            "events-sort-by-time-admission-sequence-epoch-index-and-kind-with-"
            "work-finished-before-service-fact"
        ),
        "collective_rule": (
            "collective-stages-use-external-frontier-and-runtime-calls-release-"
            "held-at-rank-release-after-reading-the-traffic-terminal-traffic-"
            "never-mutates-compute-reservations"
        ),
        "grant_authority_rule": "runtime-alone-selects-a-grant",
    }


def test_resource_and_numeric_contracts_are_exact() -> None:
    freeze = _load(EXPECTATIONS_JSON)
    assert freeze["resource_contract"] == {
        "registry_schema": "simllm-device-resource-registry-v1",
        "registry_fields": ["schema", "device_kind_id", "active_axis_ids", "axes"],
        "axis_order": "lexicographic-by-axis-id",
        "axis_identity_rule": "axis-ids-are-unique",
        "active_axis_order_rule": "sorted-unique-subset-of-axis-ids",
        "axis_classes": ["throughput", "residency", "exclusive"],
        "axis_fields": [
            "axis_id",
            "axis_class",
            "service_scope",
            "base_unit",
            "clock_domain_id",
            "capacity_source_id",
            "rate",
            "residency_capacity",
            "exclusive_capacity",
        ],
        "capacity_member_rule": (
            "all-three-capacity-members-are-required-and-exactly-the-class-"
            "appropriate-one-is-nonnull"
        ),
        "rate_fields": ["numerator", "denominator"],
        "rate_rule": ("reduced-nonnegative-integer-numerator-and-positive-integer-denominator"),
        "service_scope_values": ["device-internal", "peer-port", "data-mover"],
        "service_scope_rule": ("explicit-closed-wire-value-and-never-inferred-from-axis-id"),
        "throughput_capacity": (
            "reduced-nonnegative-rational-rate-with-clock-domain-required-only-"
            "for-cycle-denominator"
        ),
        "residency_capacity": "nonnegative-integer-units-with-no-rate",
        "exclusive_capacity": "positive-integer-slot-count-with-no-rate",
        "nonthroughput_rate_rule": "reject-nonnull-rate",
        "registry_hash_rule": ("registry-sha256-hashes-the-exact-canonical-registry-record"),
        "vector_fields": [
            "registry_sha256",
            "device_kind_id",
            "values",
            "known",
        ],
        "vector_alignment": ("values-and-known-cover-complete-registry-axis-order"),
        "vector_value_rule": ("known-values-are-nonnegative-integers-and-negative-values-reject"),
        "active_axis_rule": (
            "every-active-axis-is-known-known-positive-means-demand-and-known-"
            "zero-means-no-entry-demand"
        ),
        "inactive_axis_rule": ("inactive-axis-known-bit-is-false-with-canonical-zero-placeholder"),
        "unknown_rule": "unknown-active-axis-rejects-and-never-becomes-known-zero",
        "service_entry_fields": [
            "implementation_id",
            "shape_vector",
            "epochs",
        ],
        "service_entry_rule": (
            "immutable-key-and-ordered-nonempty-tuple-of-immutable-service-epoch-definitions"
        ),
        "service_epoch_definition_fields": [
            "resource_vector",
            "fixed_floor_ps",
        ],
        "service_epoch_definition_rule": (
            "resource-vector-aligns-to-registry-and-fixed-floor-is-null-or-"
            "nonnegative-integer-picoseconds"
        ),
        "runtime_resident_state_fields": [
            "current_epoch_index",
            "started_at_ps",
            "remaining_demands",
        ],
        "runtime_resident_state_rule": (
            "owns-current-epoch-start-and-aligned-remaining-demand-after-admission"
        ),
        "runtime_rate_rule": (
            "derive-rates-from-registry-capacity-current-residency-and-declared-"
            "interaction-law-never-from-service-entry-or-epoch-definition"
        ),
        "interaction_contract_fields": ["interaction_law", "interaction_terms"],
        "interaction_law_values": ["independent-resource-v1"],
        "interaction_terms_rule": "required-empty-and-nonempty-rejects",
        "independent_rate_rule": (
            "each-throughput-axis-divides-its-one-registry-capacity-equally-"
            "among-resident-epochs-with-positive-remaining-demand-on-that-axis-"
            "using-exact-rational-arithmetic"
        ),
        "axis_independence_rule": "resource-axes-progress-independently",
        "event_cost_rule": "O(kR)-for-k-resident-entries-and-R-active-axes",
        "epoch_throughput_rule": (
            "throughput-components-are-consumable-remaining-work-and-decrement"
        ),
        "epoch_held_requirement_rule": (
            "residency-and-exclusive-components-are-held-requirements-and-never-remaining-work"
        ),
        "admission_reservation_rule": (
            "reserve-per-axis-maxima-across-all-epochs-for-residency-and-"
            "exclusive-for-entry-lifetime-before-service"
        ),
        "epoch_completion_rule": (
            "advance-in-order-only-when-all-throughput-demands-reach-zero-and-"
            "fixed-floor-has-elapsed"
        ),
        "final_completion_rule": (
            "ordinary-entries-release-all-residency-and-exclusive-reservations-"
            "at-final-work-completion-and-collective-stages-release-at-the-frozen-"
            "rank-lease-boundary"
        ),
        "liveness_rule": (
            "load-rejects-an-active-axis-with-any-positive-accepted-entry-demand-"
            "unless-class-appropriate-capacity-is-positive-before-admission-zero-"
            "throughput-or-residency-capacity-is-legal-only-with-no-accepted-"
            "demand-or-an-inactive-axis-and-exclusive-capacity-is-always-positive"
        ),
        "core_reference_union": ["ResourceRef", "RegisteredDeviceResourceRef"],
        "registered_reference_fields": [
            "registry_sha256",
            "device_kind_id",
            "device_instance_id",
            "axis_id",
            "resource_instance_id",
            "latency_owner",
        ],
    }
    assert freeze["numeric_contract"] == {
        "integer_domain": "signed-128",
        "rate_form": "reduced-nonnegative-rational",
        "overflow_rule": "reject-on-load-or-evaluation",
        "comparison_rule": "exact-cross-product",
        "rounding_rule": ("one-ceiling-at-complete-externally-visible-picosecond-boundary"),
        "mechanistic_lookup_rule": (
            "DeviceServiceEntry-resource-demands-and-reservations-are-exact-"
            "cell-only-and-never-interpolated"
        ),
        "interpolation_scope": "optional-scalar-duration-profile-table-only",
        "scalar_axis_rule": (
            "at-most-one-declared-integer-shape-axis-with-every-other-axis-pinned"
        ),
        "scalar_support_rule": (
            "support-includes-the-canonical-endpoints-and-interpolation-applies-"
            "only-between-canonical-cells-x0-less-than-x-less-than-x1"
        ),
        "scalar_exact_hit_rule": "exact-hits-precede-interpolation",
        "scalar_tie_rule": "a-bracketing-tie-selects-the-lower-cell-id",
        "scalar_arithmetic_rule": (
            "evaluate-EQ7-with-reduced-rational-arithmetic-and-no-floating-"
            "point-logarithm-or-exponential"
        ),
        "scalar_ceiling_rule": (
            "apply-one-ceiling-only-when-y-becomes-an-externally-visible-"
            "integer-picosecond-duration"
        ),
        "scalar_lookup_cost_rule": (
            "exact-lookup-is-expected-constant-time-and-declared-one-axis-bracketing-is-logarithmic"
        ),
        "multi_axis_rule": (
            "generic-multi-axis-interpolation-is-unavailable-and-a-query-"
            "differing-on-two-or-more-axes-fails-closed"
        ),
    }


def test_measurement_and_physical_sanity_contracts_are_exact() -> None:
    freeze = _load(EXPECTATIONS_JSON)
    assert freeze["measurement_protocol"] == {
        "warmup_launches": 10,
        "retained_timing_repetitions": 41,
        "timeline_pass": "low-overhead-activity-without-per-kernel-events",
        "counter_pass": ("faithful-isolated-replay-never-used-as-concurrency-timeline"),
        "dynamic_instruction_pass": "nvidia-selected-supported-cells-only",
        "mixed_pass": "isolated-pairwise-triple-and-full-graph-makespans",
        "code_cache_state": "steady-warm",
        "data_buffer_rotation_rule": (
            "rotate-data-buffers-across-a-working-set-at-least-two-times-the-measured-l2-capacity"
        ),
        "l2_capacity_rule": (
            "record-measured-l2-capacity-in-the-controlled-environment-before-cell-execution"
        ),
        "memory_counter_rule": (
            "counter-pass-validates-compulsory-hbm-bytes-and-cache-and-hbm-"
            "counters-for-the-authored-memory-bucket"
        ),
        "memory_bucket_rejection_rule": (
            "reject-a-requested-hbm-bucket-whose-observed-data-working-set-is-l2-resident"
        ),
        "clock_policy": "exact-policy-and-observed-band-recorded-per-environment",
        "timer_unit": "integer-picoseconds",
    }
    assert freeze["physical_sanity_contract"] == {
        "finite_campaign_envelope_rule": (
            "content-address-and-freeze-before-the-first-observation"
        ),
        "finite_campaign_envelope_fields": [
            "minimum_compute_rate",
            "minimum_hbm_rate",
            "minimum_peer_port_rate",
            "minimum_transport_rate",
            "maximum_host_launch_ps",
            "maximum_device_fixed_ps",
            "maximum_transport_fixed_ps_per_action",
        ],
        "finite_rate_rule": (
            "minimum-rates-are-positive-reduced-rationals-in-declared-base-units-per-second"
        ),
        "finite_fixed_rule": ("maximum-fixed-terms-are-nonnegative-integer-picoseconds"),
        "finite_evidence_rule": (
            "every-minimum-rate-and-maximum-fixed-term-cites-preexisting-"
            "qualified-evidence-and-never-the-current-cell-outcome"
        ),
        "kernel_ceiling_equation": (
            "kernel_ceiling_ps = maximum_host_launch_ps + maximum_device_fixed_ps "
            "+ ceil(flops * minimum_compute_rate_den * 10^12 / "
            "minimum_compute_rate_num) + ceil(hbm_bytes * minimum_hbm_rate_den * "
            "10^12 / minimum_hbm_rate_num) + ceil(peer_bytes * "
            "minimum_peer_port_rate_den * 10^12 / minimum_peer_port_rate_num)"
        ),
        "communication_ceiling_equation": (
            "communication_ceiling_ps = maximum_host_launch_ps + "
            "maximum_device_fixed_ps + ceil(derived_peer_bytes * "
            "minimum_peer_port_rate_den * 10^12 / minimum_peer_port_rate_num) + "
            "sum(maximum_transport_fixed_ps_per_action + ceil(action_bytes * "
            "minimum_transport_rate_den * 10^12 / minimum_transport_rate_num))"
        ),
        "graph_ceiling_rule": (
            "sum-applicable-kernel-and-communication-ceilings-in-a-fully-"
            "serialized-topological-order"
        ),
        "mixed_physical_ceiling_rule": (
            "sum-the-applicable-member-ceilings-for-every-authored-copy"
        ),
        "mixed_floor_rule": "maximum-of-the-applicable-member-copy-floors",
        "mixed_control_rule": (
            "simultaneous-makespan-is-also-no-greater-than-the-measured-serialized-"
            "same-members-control-whose-own-timing-must-pass-its-physical-ceiling"
        ),
        "ceiling_state_values": ["finite", "unbounded"],
        "unbounded_rule": (
            "an-isolated-or-graph-cell-with-no-defensible-finite-bound-declares-"
            "unbounded-before-observation-without-borrowing-a-measured-value"
        ),
        "cell_rule": (
            "materialize-each-applicable-floor-and-finite-or-explicitly-unbounded-"
            "stratum-ceiling-before-reading-any-cell-timing"
        ),
    }


def test_equation_relation_guard_and_matrix_denominators_are_exact() -> None:
    suite = _load(SUITE_JSON)
    freeze = _load(EXPECTATIONS_JSON)
    equations = freeze["equations"]
    relations = freeze["expected_relations"]
    guards = freeze["fatal_guards"]
    assert [(row["id"], row["name"], row["expression"]) for row in equations] == [
        (
            "EQ1",
            "record-identity",
            "record_id = lowercase_hex(SHA256(canonical_record_bytes))",
        ),
        (
            "EQ2",
            "compute-floor",
            ("compute_floor_ps = ceil(flops * compute_rate_den * 10^12 / compute_rate_num)"),
        ),
        (
            "EQ3",
            "memory-floor",
            ("memory_floor_ps = ceil(hbm_bytes * hbm_rate_den * 10^12 / hbm_rate_num)"),
        ),
        (
            "EQ4",
            "peer-floor",
            ("peer_floor_ps = ceil(peer_bytes * peer_rate_den * 10^12 / peer_rate_num)"),
        ),
        (
            "EQ5",
            "isolated-floor",
            (
                "isolated_floor_ps = max(compute_floor_ps, memory_floor_ps, "
                "peer_floor_ps, kernel_floor_ps)"
            ),
        ),
        (
            "EQ6",
            "graph-floor",
            ("graph_floor_ps = longest_dependency_path_sum(applicable_stage_floors_ps)"),
        ),
        (
            "EQ7",
            "optional-scalar-duration-one-axis-interpolation",
            "y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)",
        ),
        (
            "EQ8",
            "absolute-percentage-error",
            "ape = abs(predicted_ps - silicon_ps) / silicon_ps",
        ),
        (
            "EQ9",
            "support-coverage",
            ("support_coverage = supported_required_test_units / all_required_test_units"),
        ),
    ]
    assert [(row["id"], row["family"], row["claim"]) for row in relations] == [
        (
            "R1",
            "physical-floor-monotonicity",
            (
                "For one implementation, increasing an authored work quantity "
                "cannot lower the corresponding physical floor."
            ),
        ),
        (
            "R2",
            "capacity-scaling",
            (
                "Halving one identified throughput capacity doubles that resource "
                "service term exactly and cannot reduce complete service."
            ),
        ),
        (
            "R3",
            "train-defined-support",
            (
                "A validated claimed capability supports every required test unit "
                "inside the support envelope defined from train rows only."
            ),
        ),
        (
            "R4",
            "kernel-accuracy",
            (
                "Untouched supported kernel test rows meet the frozen median and "
                "p95 absolute-percentage-error bars."
            ),
        ),
        (
            "R5",
            "phase-and-step-accuracy",
            (
                "Untouched phase rows and the compute-only full step meet their "
                "separately frozen error bars."
            ),
        ),
        (
            "R6",
            "mixed-resource-bounds",
            (
                "Every simultaneous mixed-resource makespan lies between its "
                "applicable floor and its deliberately serialized control."
            ),
        ),
        (
            "R7",
            "mixed-resource-accuracy",
            (
                "Mixed completion and queue wait stay within the larger of two GPU "
                "cycles or ten percent on untouched cells."
            ),
        ),
        (
            "R8",
            "communication-scaling",
            (
                "At fixed participants, scaling payload by an authored integer "
                "ratio scales the peer-port service floor by that exact ratio and "
                "cannot reduce completion."
            ),
        ),
        (
            "R9",
            "selective-a100-simulator-fill",
            (
                "Inside the train-defined target and qualified SM80 envelopes, "
                "an exact-only categorical region with no exact silicon row or "
                "validated silicon-fit entry selects qualified Accel-Sim only "
                "between real anchors; a later valid silicon fit for that exact "
                "region outranks it, and extrapolation remains forbidden."
            ),
        ),
        (
            "R10",
            "live-and-disabled-paths",
            (
                "On a serial compute-only graph, changing exactly one critical-path "
                "compact-service duration by signed delta_ps while holding every "
                "other term fixed changes StepResult completion and prefill TTFT or "
                "decode TPOT by exactly delta_ps; the disabled path preserves every "
                "accepted byte and timestamp exactly."
            ),
        ),
        (
            "R11",
            "host-launch-residual-identification",
            (
                "On untouched launch-mode cells, measured host initiation predicts "
                "the observed residual within the larger of two GPU cycles or ten "
                "percent while kernel service remains identical."
            ),
        ),
        (
            "R12",
            "channel-repartitioning",
            (
                "At fixed operation, algorithm, participants and payload, changing "
                "channel count preserves logical bytes and the peer-port "
                "serialization floor exactly; no completion monotonicity is "
                "assumed, and untouched channel-count-three completion and queue "
                "wait meet the larger of two GPU cycles or ten percent."
            ),
        ),
    ]
    assert [(row["id"], row["claim"]) for row in guards] == [
        (
            "G1",
            (
                "Target, framework, model, toolchain, launch mode and environment "
                "identities equal the declared run envelope."
            ),
        ),
        (
            "G2",
            (
                "Before and after device UUID, SKU, partition, clock policy, power "
                "policy and foreign-process state agree."
            ),
        ),
        (
            "G3",
            (
                "Every execution graph is strict simllm-execution-graph-v1, "
                "acyclic and has a complete completion frontier; its token fixture "
                "equals the authored generator, every referenced collective plan "
                "has exact rank membership and valid entry and terminal action "
                "identities, and MoE routing does not diverge inside one claimed "
                "envelope."
            ),
        ),
        (
            "G4",
            (
                "Activity and correlation joins are total: every noncollective "
                "physical launch has exactly one operation binding and every bound "
                "noncollective operation has exactly one physical launch; a semantic "
                "noncollective operation hiding two or more launches is unsupported. "
                "Every collective physical launch has exactly one stage binding, and "
                "neither ledger has a missing, duplicate or extra launch."
            ),
        ),
        (
            "G5",
            (
                "Every supported capture has a total observed binding for each "
                "noncollective operation and collective device stage; every online "
                "modeled operation and collective plan rank has exactly one resolved "
                "service entry, v1 has exactly one resident stage per rank, and "
                "validation never rewrites an observed implementation."
            ),
        ),
        (
            "G6",
            (
                "No kernel, graph or repetition unit crosses the frozen train, "
                "validation and test boundary."
            ),
        ),
        (
            "G7",
            (
                "Every timing lies between its precomputed physical floor and "
                "frozen stratum ceiling, and every simultaneous mixed-resource "
                "makespan also lies at or below its deliberately serialized "
                "same-members control."
            ),
        ),
        (
            "G8",
            (
                "Every external raw-blob digest, byte count and recomputed compact "
                "summary agrees with its evidence record."
            ),
        ),
        (
            "G9",
            (
                "Task, operation, launch, collective rank, plan action, device stage "
                "and byte accounting conserve exactly with no hidden loss, "
                "duplication or extra member."
            ),
        ),
        (
            "G10",
            (
                "No GPU-port, collective, RNIC or fabric interval or byte is "
                "charged by two timing authorities; version-1 collective device "
                "stages permit positive demand only on explicit device-internal "
                "axes and reject it on peer-port or data-mover axes, which traffic "
                "alone owns."
            ),
        ),
        (
            "G11",
            (
                "Timeline, counter and dynamic-instruction passes remain separate, "
                "and counter replay is never used as the concurrency timeline."
            ),
        ),
        (
            "G12",
            (
                "No per-kernel timing event perturbs a short production kernel, "
                "device-side event tracing is disabled unless separately frozen, "
                "code and cache are steady-warm, data rotation spans at least "
                "twice measured L2, and compulsory-HBM plus cache and HBM counters "
                "reject an L2-resident requested HBM bucket."
            ),
        ),
        (
            "G13",
            (
                "Resource vectors reject unknown active or negative known demand; "
                "demanded active axes have positive capacity; only independent-"
                "resource-v1 with empty interaction terms loads; epoch reservation, "
                "progress, ordered advance and final release follow the frozen "
                "semantics."
            ),
        ),
        (
            "G14",
            (
                "The content-addressed record graph is acyclic, complete and "
                "consistent with the one canonical byte contract."
            ),
        ),
        (
            "G15",
            (
                "Exact integer and reduced-rational arithmetic applies one final "
                "picosecond ceiling and remains inside the declared overflow bound."
            ),
        ),
        (
            "G16",
            (
                "Accel-Sim serves only supported SM80 cells and rejects H100, AMD "
                "and unsupported SM80 features."
            ),
        ),
        (
            "G17",
            ("Accel-Sim supplies no communication-stratum evidence or fabric timing."),
        ),
        (
            "G18",
            (
                "Calibration-off, absent-profile, absent-submodule, batch scalar "
                "and legacy off paths, disabled collective-stage and collective-"
                "stage paths whose every active-axis demand is known zero in every "
                "epoch and every epoch floor is null or zero preserve service "
                "calls, composition cursors, barriers, visits, reports, result "
                "bytes and timestamps exactly; a positive floor prevents bypass, "
                "a bypass emits no device visit and delays no plan entry, and a "
                "rejected batch result leaves live state unchanged."
            ),
        ),
        (
            "G19",
            (
                "ResolvedDeviceBindingClosure, model identity, acceptance status, "
                "target basis and operating envelope reach StepResult, TTFT and TPOT "
                "provenance without disagreement."
            ),
        ),
        (
            "G20",
            (
                "One fatal violation makes the affected device campaign void, "
                "retains its evidence and closes no task."
            ),
        ),
    ]
    assert freeze["survivable_fatal_guards"] == []
    assert freeze["denominators"] == {
        "graph_cells": len(suite["graph_cells"]),
        "communication_cells": len(suite["communication_cells"]),
        "mixed_cells": len(suite["mixed_matrix"]["arms"]) * len(suite["mixed_matrix"]["widths"]),
        "equations": len(equations),
        "expected_relation_families": len(relations),
        "fatal_guards_unscored": len(guards),
    }


def test_acceptance_bars_are_exact_rationals() -> None:
    bars = _load(EXPECTATIONS_JSON)["acceptance_bars"]
    expected = {
        "supported_graph_operation_coverage": [1, 1],
        "supported_graph_physical_launch_coverage": [1, 1],
        "validated_test_support_coverage": [1, 1],
        "controlled_environment_population_cv_lt": [1, 50],
        "kernel_test_median_ape_lt": [1, 10],
        "kernel_test_p95_ape_lt": [1, 5],
        "phase_test_median_ape_lt": [1, 20],
        "phase_test_p95_ape_lt": [1, 10],
        "compute_only_step_ape_lt": [1, 20],
        "mixed_relative_error_le": [1, 10],
        "mixed_absolute_error_gpu_cycles_le": 2,
        "host_launch_residual_relative_error_le": [1, 10],
        "host_launch_residual_absolute_gpu_cycles_le": 2,
        "percentile_rule": "nearest-rank",
        "comparison_rule": ("strict-lt-where-key-ends-lt-and-inclusive-le-where-key-ends-le"),
    }
    assert bars == expected


def test_machine_ids_are_all_present_in_the_prose() -> None:
    suite = _load(SUITE_JSON)
    freeze = _load(EXPECTATIONS_JSON)
    prose = EXPECTATIONS_MD.read_text(encoding="utf-8")
    for section in ("equations", "expected_relations", "fatal_guards"):
        for row in freeze[section]:
            assert f"**{row['id']}" in prose, row["id"]
    for section in ("graph_cells", "communication_cells"):
        for row in suite[section]:
            assert f"`{row['id']}`" in prose, row["id"]


def test_freeze_files_have_no_em_dash_or_nonportable_path() -> None:
    posix_absolute = re.compile(
        r"(?<![)/:A-Za-z0-9_.<>{}])/(?!/|\.\.?/)"
        r"[A-Za-z0-9_.@+~-]+(?:/[A-Za-z0-9_.@+~-]+)*/?"
    )
    windows_drive = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
    home_shortcut = re.compile(r"(?<![A-Za-z0-9_])~[\\/]")
    url = re.compile(r"https?://[^\s`\"]+")
    for path in (SUITE_JSON, EXPECTATIONS_JSON, EXPECTATIONS_MD):
        text = path.read_text(encoding="utf-8")
        assert EM_DASH not in text
        masked = url.sub("", text)
        assert not posix_absolute.search(masked)
        assert not windows_drive.search(masked)
        assert not home_shortcut.search(masked)


def test_freeze_lock_has_a_read_only_standard_library_dependency_boundary() -> None:
    source = THIS_TEST.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"write_text", "write_bytes", "unlink"}
    assert imported == {"__future__", "ast", "itertools", "json", "re", "pathlib"}
