from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import pytest

from simllm.calibration.canonical import canonical_sha256
from simllm.calibration.kernel_cycle_lut import (
    KERNEL_CYCLE_PRICING_PROVENANCE_SCHEMA,
    KernelCycleLookupBinding,
    analyze_kernel_cycle_capture,
    compile_session_profile_provider,
)
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    DurationEstimate,
    KernelRequestShape,
    KernelSpec,
    ModelDims,
    step_kernel,
)
from simllm.core import RequestPhase, ScheduledRequest, StepRecord

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kernel_cycle_lut_v1"
EXPECTED_RECORD_SHA256 = (
    "e495f3ca5d0858cf371b19205ae6b7747d633695020d10f58645c5f245086070"
)
EXPECTED_ENTRY_SHA256 = (
    "d178524fd1bbc3719f7d45065e3a35b41ef9abfec3965d1db453152a30e303e1"
)


class FixedComparator(ComputeProvider):
    precision_compute_level = "roofline"

    def __init__(self, duration_ps: int = 123) -> None:
        self.duration_ps = duration_ps

    def estimate(self, kernel: KernelSpec, gpu) -> DurationEstimate:
        return DurationEstimate(self.duration_ps, "comparison", 0.25)


@pytest.fixture(scope="module")
def record():
    return analyze_kernel_cycle_capture(FIXTURE)


def _kernel(*prior_contexts: int) -> KernelSpec:
    return KernelSpec(
        name="llm_step",
        flops=17,
        bytes_moved=23,
        request_shapes=tuple(
            KernelRequestShape(
                num_new_tokens=1,
                context_length=prior_context + 1,
            )
            for prior_context in prior_contexts
        ),
    )


def _provider(record, *, pool: str = "decode", comparator=None):
    return compile_session_profile_provider(
        record.canonical,
        expected_sha256=record.record_id,
        pool=pool,
        comparator=comparator,
    )


def test_exact_decode_shape_selects_compiled_candidate_row(record) -> None:
    provider = _provider(record, comparator=FixedComparator())
    binding = provider.lookup_binding
    assert isinstance(binding, KernelCycleLookupBinding)

    selection = binding.selection_for(_kernel(16))
    estimate = provider.estimate(_kernel(16), GPU_ENVELOPES["b100"])
    provenance = provider.pricing_provenance()

    assert selection == {
        "query_key_sha256": EXPECTED_ENTRY_SHA256,
        "selected": True,
        "selected_entry_key_sha256": EXPECTED_ENTRY_SHA256,
        "implementation_id": "vllm.granite.tp1.decode.cuda-graph.partial-v1",
    }
    assert estimate == DurationEstimate(2_047_488_000, "measured", 0.007681)
    assert provenance == {
        "schema": KERNEL_CYCLE_PRICING_PROVENANCE_SCHEMA,
        "record_sha256": EXPECTED_RECORD_SHA256,
        "acceptance_status": "candidate",
        "campaign_id": "retained-granite-tp1-b1-candidate",
        "coverage": "partial-kernel-subset",
        "record_device_kind_id": "nvidia-a100-sm80",
        "pool": "decode",
        "selected_entry_key_sha256": EXPECTED_ENTRY_SHA256,
        "selected_entry_key_sha256s": [EXPECTED_ENTRY_SHA256],
        "lookup_hits": 1,
        "lookup_misses": 0,
        "calibration_claim": False,
    }


@pytest.mark.parametrize("prior_context", [8, 15, 17])
def test_uncovered_decode_shapes_preserve_comparator_estimate(
    record,
    prior_context: int,
) -> None:
    comparator = FixedComparator(duration_ps=456)
    provider = _provider(record, comparator=comparator)
    binding = provider.lookup_binding
    assert isinstance(binding, KernelCycleLookupBinding)

    selection = binding.selection_for(_kernel(prior_context))
    estimate = provider.estimate(_kernel(prior_context), GPU_ENVELOPES["b100"])

    assert selection["selected"] is False
    assert selection["selected_entry_key_sha256"] is None
    assert estimate == comparator.estimate(_kernel(prior_context), GPU_ENVELOPES["b100"])
    assert provider.pricing_provenance()["lookup_misses"] == 1


def test_prefill_shape_is_formed_but_candidate_has_no_prefill_row(record) -> None:
    comparator = FixedComparator(duration_ps=789)
    provider = _provider(record, pool="prefill", comparator=comparator)
    binding = provider.lookup_binding
    assert isinstance(binding, KernelCycleLookupBinding)
    kernel = KernelSpec(
        name="llm_step",
        flops=17,
        bytes_moved=23,
        request_shapes=(KernelRequestShape(16, 16),),
    )

    selection = binding.selection_for(kernel)
    estimate = provider.estimate(kernel, GPU_ENVELOPES["b100"])

    assert selection["query_key_sha256"] is not None
    assert selection["selected"] is False
    assert estimate.duration_ps == 789
    provenance = provider.pricing_provenance()
    assert provenance["pool"] == "prefill"
    assert provenance["selected_entry_key_sha256"] is None


def test_kernel_without_request_projection_uses_comparator(record) -> None:
    provider = _provider(record, comparator=FixedComparator(duration_ps=321))
    binding = provider.lookup_binding
    assert isinstance(binding, KernelCycleLookupBinding)
    kernel = KernelSpec("legacy", 1, 1)

    assert binding.selection_for(kernel)["query_key_sha256"] is None
    assert provider.estimate(kernel, GPU_ENVELOPES["b100"]).duration_ps == 321


def test_content_address_mismatch_rejects_before_pricing(record) -> None:
    with pytest.raises(ValueError, match="content address disagrees"):
        compile_session_profile_provider(
            record.canonical,
            expected_sha256="0" * 64,
            pool="decode",
        )


def test_compiled_provider_round_trips_across_spawn_serialization(record) -> None:
    provider = _provider(record, comparator=FixedComparator())

    restored = pickle.loads(pickle.dumps(provider))
    estimate = restored.estimate(_kernel(16), GPU_ENVELOPES["b100"])

    assert estimate.duration_ps == 2_047_488_000
    assert restored.pricing_provenance()["record_sha256"] == record.record_id
    assert restored.pricing_provenance()["lookup_hits"] == 1


def test_duplicate_complete_keys_reject_instead_of_using_file_order(record) -> None:
    payload = json.loads(record.canonical)
    duplicate = copy.deepcopy(payload["entries"][0])
    duplicate["implementation_id"] = (
        "vllm.granite.tp1.decode.cuda-graph.partial-v2"
    )
    payload["entries"].append(duplicate)
    provider = compile_session_profile_provider(
        payload,
        expected_sha256=canonical_sha256(payload),
        pool="decode",
        comparator=FixedComparator(),
    )

    with pytest.raises(ValueError, match="duplicate rows for complete key"):
        provider.estimate(_kernel(16), GPU_ENVELOPES["b100"])


def test_hopper_record_data_swaps_by_content_address_without_code_change(record) -> None:
    payload = json.loads(record.canonical)
    payload["campaign_id"] = "future-hopper-data-only-control"
    payload["device"].update(
        {
            "device_kind_id": "nvidia-hopper-sm90",
            "gpu_name": "Hopper data-only control",
            "gpu_uuid": "GPU-data-only-control",
            "architecture": "sm90",
        }
    )
    content_address = canonical_sha256(payload)

    provider = compile_session_profile_provider(
        payload,
        expected_sha256=content_address,
        pool="decode",
        comparator=FixedComparator(),
    )

    assert provider.estimate(_kernel(16), GPU_ENVELOPES["b100"]).duration_ps == (
        2_047_488_000
    )
    provenance = provider.pricing_provenance()
    assert provenance["record_sha256"] == content_address
    assert provenance["record_device_kind_id"] == "nvidia-hopper-sm90"
    assert provenance["acceptance_status"] == "candidate"
    assert provenance["calibration_claim"] is False


def test_step_kernel_carries_ordered_request_shapes_without_changing_config() -> None:
    dims = ModelDims(
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=1,
        num_kv_heads=1,
        head_size=8,
        vocab_size=32,
        dtype_bytes=2,
    )
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest("a", RequestPhase.DECODE, 1, context_length=17),
            ScheduledRequest("b", RequestPhase.DECODE, 1, context_length=9),
        ],
        num_sampled=2,
    )

    kernel = step_kernel(dims, record, num_sampled=2)

    assert kernel.config == (
        ("new_tokens", 2),
        ("sampled", 2),
        ("kv_tokens", 26),
    )
    assert kernel.request_shapes == (
        KernelRequestShape(1, 17),
        KernelRequestShape(1, 9),
    )
    assert [shape.prior_context_tokens for shape in kernel.request_shapes] == [
        16,
        8,
    ]


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((True, 1), "integer"),
        ((0, 1), "positive"),
        ((2, 1), "include every new token"),
    ],
)
def test_request_shape_rejects_ambiguous_values(args, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        KernelRequestShape(*args)
