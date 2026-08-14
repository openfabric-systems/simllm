"""Selectable per-step host cost for the SGLang chain (SGL-23).

Until this module existed no SGL task owned choosing a host model. Every
SGLang study built :meth:`~simllm.compute.HostInitiationModel.ideal` by hand,
so the per-step host term on this chain was exactly zero while the vLLM chain
already selected the calibrated profiles through a study-local resolver.
:func:`select_sglang_host_model` is that resolver, owned by the adapter instead
of by one study, and ``ideal`` remains the default and the exact off path.

Three facts have to travel with any nonideal selection, because none of the
constants it composes was measured on SGLang:

- the per-launch point ``g`` is a CUDA-graph node replay time or an eager
  host-bound launch measured on one GTX 1660 Ti with an AMD Ryzen 9 3950X host
  in ``examples/host_step_cost_v1``. COMP-1 refuses to treat it as a
  calibration for any other device;
- the launch count ``N`` is a static enumeration of vLLM 0.26.0 sources for the
  pinned Granite MoE geometry, frozen as the bracket ``[440, 567]`` in
  ``examples/compute_fidelity_v1``. SGLang's model runner, its fused MoE path
  and the pump's unrolled ``event_loop_normal`` issue their own launches and
  nobody has counted them. SGL-24 owns that measurement; until it lands, using
  vLLM's count here is a third-party surrogate;
- the compute envelope stays ``b100`` while the calibrated host profiles refuse
  every GPU key but ``gtx1660-ti-sm75``. The selection therefore presents the
  Turing key to the host model and pins the provider to the accepted envelope,
  exactly as ``examples/end_to_end_replay_v1`` does. Every enabled row is a
  disclosed device hybrid and never a device-consistent prediction.

:data:`SGLANG_HOST_TRANSFER_DISCLOSURE` is that statement in one string, and a
selection carries it so a study cannot report an enabled cell without it.

This module imports no SGLang and needs no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    DurationEstimate,
    GpuSpec,
    HostInitiationModel,
    KernelSpec,
    RooflineProvider,
)

__all__ = [
    "SGLANG_HOST_PROFILES",
    "SGLANG_HOST_TRANSFER_DISCLOSURE",
    "SGLANG_TRANSFERRED_LAUNCH_COUNTS",
    "SglangHostSelection",
    "select_sglang_host_model",
]

#: Profiles an SGLang study may select. ``ideal`` is the default, contributes
#: exactly zero, and is the only one that preserves an accepted artifact.
SGLANG_HOST_PROFILES: tuple[str, ...] = (
    "ideal",
    "turing-cuda-graph",
    "turing-eager-host",
)

#: Launch counts transferred from the vLLM 0.26.0 static enumeration. They are
#: a surrogate for SGLang's own unmeasured launch demand (SGL-24), offered as a
#: convenience so a caller cannot invent a bracket endpoint by accident.
SGLANG_TRANSFERRED_LAUNCH_COUNTS: dict[str, int] = {
    "turing-cuda-graph": 440,
    "turing-eager-host": 567,
}

#: The provider envelope every SGLang study prices compute against.
SGLANG_PROVIDER_ENVELOPE = "b100"

SGLANG_HOST_TRANSFER_DISCLOSURE = (
    "This host cost is transferred, not calibrated, and no part of it was "
    "measured on SGLang. The per-launch point is a GTX 1660 Ti (Turing, "
    "sm75) capture taken with an AMD Ryzen 9 3950X host in "
    "examples/host_step_cost_v1, and its sample-limited empirical range is "
    "not a confidence interval. The launch count is a static enumeration of "
    "vLLM 0.26.0 sources for the pinned Granite MoE geometry "
    "(examples/compute_fidelity_v1); SGLang's own per-step launch demand is "
    "unmeasured and is registered as SGL-24. Compute is priced against the "
    "b100 envelope while the device key presented to the host model is "
    "gtx1660-ti-sm75, so every enabled row is a three-source device hybrid. "
    "Nothing selected here is a calibration of the SGLang chain or a "
    "prediction for any deployment."
)


class _PinnedEnvelopeProvider(ComputeProvider):
    """Price every kernel against one fixed envelope, ignoring the argument.

    A calibrated host profile refuses every GPU key but its own, so a cell that
    selects one has to present that device key to the host model. The compute
    input must not move for that reason alone. This provider keeps the accepted
    envelope while the configuration carries the calibrated device key, which
    is what makes the enabled rows an explicitly disclosed device hybrid rather
    than a silent change of GPU.
    """

    def __init__(self, envelope: str, efficiency: float) -> None:
        if envelope not in GPU_ENVELOPES:
            known = ", ".join(sorted(GPU_ENVELOPES))
            raise ValueError(f"unknown compute envelope {envelope!r}; known: {known}")
        self._pinned = GPU_ENVELOPES[envelope]
        self._inner = RooflineProvider(efficiency)
        self.envelope = envelope
        self.efficiency = float(efficiency)
        self.precision_compute_level = getattr(
            self._inner, "precision_compute_level", None
        )

    def estimate(self, kernel: KernelSpec, gpu: GpuSpec) -> DurationEstimate:
        del gpu
        return self._inner.estimate(kernel, self._pinned)

    def estimate_layers(
        self,
        kernel: KernelSpec,
        gpu: GpuSpec,
        num_layers: int,
    ) -> tuple[DurationEstimate, ...] | None:
        del gpu
        return self._inner.estimate_layers(kernel, self._pinned, num_layers)


@dataclass(frozen=True)
class SglangHostSelection:
    """One resolved per-step host cost, with everything it forces.

    ``host_model``, ``gpu`` and ``provider`` must be applied together: the
    adapter's own check (`_validate_host_model_selection` in
    :mod:`simllm.adapters.sglang.worker`) requires the worker and the step sink
    to select the same host model, and the calibrated profiles reject any GPU
    key but their own. :meth:`overrides` returns the three under the keyword
    names both `configure` and ``HtsimStepSinkConfig`` use.

    ``is_default`` is what keeps every accepted SGLang artifact byte identical:
    the ideal selection resolves to the same three objects a pre-SGL-23 study
    built by hand, so a study can write the extra disclosure row only when a
    caller actually selected something else.
    """

    profile: str
    host_model: HostInitiationModel
    gpu: GpuSpec
    provider: ComputeProvider
    provider_envelope: str
    efficiency: float
    launch_count: int
    launch_floor_ps: int
    is_default: bool
    transfer_disclosure: str | None

    def overrides(self) -> dict[str, Any]:
        """The three objects a study hands to the worker and to the sink."""

        return {
            "host_model": self.host_model,
            "gpu": self.gpu,
            "provider": self.provider,
        }

    def describe(self) -> str:
        """One line naming the profile, the launch demand and the transfer."""

        if self.is_default:
            return "host profile ideal: no per-step host cost is modeled"
        return (
            f"host profile {self.profile} with {self.launch_count} launches, "
            f"a {self.launch_floor_ps} ps launch floor on device key "
            f"{self.gpu.name}, compute pinned to {self.provider_envelope}. "
            f"{self.transfer_disclosure}"
        )


def select_sglang_host_model(
    profile: str = "ideal",
    *,
    launch_count: int | None = None,
    provider_envelope: str = SGLANG_PROVIDER_ENVELOPE,
    efficiency: float = 0.7,
) -> SglangHostSelection:
    """Resolve one SGLang per-step host cost and everything it forces.

    ``profile`` defaults to ``ideal``, which returns the exact pre-SGL-23
    configuration: a zero host model, the ``provider_envelope`` GPU under its
    own key and a plain :class:`~simllm.compute.RooflineProvider`. That path
    takes no launch count and carries no disclosure, because it transfers
    nothing.

    A calibrated profile requires a positive ``launch_count``. Passing ``None``
    takes the transferred vLLM bracket endpoint for that profile, which is a
    surrogate for SGLang's own unmeasured launch demand (SGL-24), not a
    measurement of it. The returned selection carries
    :data:`SGLANG_HOST_TRANSFER_DISCLOSURE` so a caller cannot report an
    enabled cell without saying where its constants came from.
    """

    if profile not in SGLANG_HOST_PROFILES:
        known = ", ".join(SGLANG_HOST_PROFILES)
        raise ValueError(f"unknown SGLang host profile {profile!r}; known: {known}")
    if provider_envelope not in GPU_ENVELOPES:
        known = ", ".join(sorted(GPU_ENVELOPES))
        raise ValueError(
            f"unknown compute envelope {provider_envelope!r}; known: {known}"
        )
    if isinstance(efficiency, bool) or not isinstance(efficiency, (int, float)):
        raise TypeError("efficiency must be a number")
    if not 0.0 < float(efficiency) <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")

    if profile == "ideal":
        if launch_count not in (None, 0):
            raise ValueError("the ideal host profile takes no launch count")
        return SglangHostSelection(
            profile="ideal",
            host_model=HostInitiationModel.ideal(),
            gpu=GPU_ENVELOPES[provider_envelope],
            provider=RooflineProvider(efficiency),
            provider_envelope=provider_envelope,
            efficiency=float(efficiency),
            launch_count=0,
            launch_floor_ps=0,
            is_default=True,
            transfer_disclosure=None,
        )

    resolved = (
        SGLANG_TRANSFERRED_LAUNCH_COUNTS[profile]
        if launch_count is None
        else launch_count
    )
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise TypeError("launch_count must be an integer")
    if resolved <= 0:
        raise ValueError("a calibrated host profile needs a positive launch count")

    builder = (
        HostInitiationModel.turing_cuda_graph
        if profile == "turing-cuda-graph"
        else HostInitiationModel.turing_eager_host
    )
    host_model = builder(resolved)
    gpu = GPU_ENVELOPES[host_model.device_key]
    return SglangHostSelection(
        profile=profile,
        host_model=host_model,
        gpu=gpu,
        provider=_PinnedEnvelopeProvider(provider_envelope, efficiency),
        provider_envelope=provider_envelope,
        efficiency=float(efficiency),
        launch_count=resolved,
        launch_floor_ps=host_model.launch_floor_ps,
        is_default=False,
        transfer_disclosure=SGLANG_HOST_TRANSFER_DISCLOSURE,
    )
