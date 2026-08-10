"""Closed-loop step sink: one packet-level ``htsim_rnic`` run per step.

:class:`HtsimStepSink` implements the adapters' step-sink contract (a
callable ``StepRecord -> StepResult | None``): for every scheduler step it
renders the step's tensor-parallel GOAL program
(:func:`simllm.traffic.render_step_goal`), converts it with ``txt2bin``,
executes it on a configured ``htsim_rnic`` profile, and returns the
simulated makespan as the step latency. Plugged into
``simllm.adapters.vllm.configure(step_sink=...)`` or
``simllm.adapters.sglang.configure(step_sink=...)`` this closes the loop:
the network's completion time advances the virtual clock the frontend
scheduler sees.

Per-step subprocess invocation is the documented *diagnostic* mode of the
closed loop (docs/modules/core.md): every step pays a process spawn and a
full GOAL parse, which is fine for validation runs of tens of steps. The
persistent co-simulator that amortizes this is BRIDGE-1 and needs the
incremental flow-injection mode on the htsim side; this module deliberately
does not attempt it.

A step with no collective work returns ``None``: the TP world has size 1
(or the dims declare no experts, or no EP group is configured) and the
record is a drain record with zero new tokens. ``None`` tells the adapter
that its own compute-only estimate stands, which is exactly right when
there is no network work to simulate. With MoE dims and ``ep_ranks``
configured, the per-step GOAL additionally carries the dispatch and
combine all-to-alls of every layer
(:func:`simllm.traffic.step_moe_alltoalls`); a non-MoE configuration
renders byte-identically to the pre-MoE sink.

Modeling approximations (numbered in docs/modules/backends.md):

- BACK-5: the whole-step compute estimate is split evenly across layers,
  ``per_layer_calc_ns = estimate_step_latency_ps(...) // (L * 1000)`` (the
  division also truncates ps to whole GOAL ns units). Real per-layer times
  differ (the LM head and sampling live in the last layer's share).
- BACK-6: ``num_sampled`` is approximated as the number of scheduled
  requests; a mid-prompt chunked prefill does not actually sample. The LM
  head term this inflates is small against the step total.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from simllm.backends.htsim_rnic import RNIC_PROFILES, HtsimRnicConfig, run_htsim_rnic
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
    estimate_step_latency_ps,
)
from simllm.core import StepRecord, StepResult
from simllm.goal import to_binary
from simllm.traffic import render_step_goal, step_moe_alltoalls, step_tp_allreduces


@dataclass
class HtsimStepSinkConfig:
    """One closed-loop deployment under simulation.

    ``tp_ranks`` are the GOAL ranks of the tensor-parallel group (e.g.
    ``manifest.group_ranks(0, "tp")`` of a declared manifest under the
    gpu-rank mapping); ``dims`` is the per-rank sharded geometry the same
    deployment declares. ``ep_ranks`` is the optional expert-parallel
    group: when the dims declare experts (``dims.num_experts > 0``) and
    ``ep_ranks`` are given, every step's GOAL includes the per-layer MoE
    dispatch and combine all-to-alls over these ranks; leaving it ``None``
    (or using dense dims) keeps the per-step GOAL byte-identical to the
    pre-MoE sink. ``topology`` is optional: the null-network profiles
    (``rnic-nn``, ``rnic-nn-fluid``) run on the generated manifold,
    ``rnic-cn`` takes a Clos topology file.
    """

    profile: str
    tp_ranks: Sequence[int]
    dims: ModelDims
    workdir: Path
    ep_ranks: Sequence[int] | None = None
    linkspeed_bps: int = 400_000_000_000
    topology: Path | None = None
    #: explicit GOAL rank count for topology padding; None keeps inferred sizing
    num_goal_ranks: int | None = None
    provider: ComputeProvider = field(default_factory=lambda: RooflineProvider(efficiency=0.7))
    gpu: GpuSpec = GPU_ENVELOPES["b100"]
    host_model: HostInitiationModel = field(default_factory=HostInitiationModel)
    #: first GOAL tag; each allreduce takes a disjoint 2(W-1)-tag block
    base_tag: int = 1000

    def __post_init__(self) -> None:
        if self.profile not in RNIC_PROFILES:
            raise ValueError(f"profile must be one of {RNIC_PROFILES}")


@dataclass(frozen=True)
class StepNetworkOutcome:
    """Bookkeeping for one simulated step, kept for reporting."""

    step_index: int
    #: the adapter-equivalent compute-only whole-step estimate, ps
    compute_estimate_ps: int
    #: the even-split per-layer calc cost handed to GOAL, ns (BACK-5)
    per_layer_calc_ns: int
    #: simulated makespan of the step's GOAL program, ps
    makespan_ps: int
    num_flows: int

    def network_share_for(self, num_layers: int) -> float:
        """1 - (L * per-layer calc) / makespan, the step's network fraction."""
        calc_ps = num_layers * max(self.per_layer_calc_ns, 1) * 1000
        return 1.0 - calc_ps / self.makespan_ps


class HtsimStepSink:
    """Step sink that simulates each step's TP traffic on ``htsim_rnic``."""

    def __init__(self, config: HtsimStepSinkConfig) -> None:
        self.config = config
        self.config.workdir.mkdir(parents=True, exist_ok=True)
        #: one entry per simulated (non-None) step, in call order
        self.outcomes: list[StepNetworkOutcome] = []

    def compute_estimate_ps(self, record: StepRecord) -> int:
        """The compute-only whole-step estimate the sink splits into calcs."""
        return estimate_step_latency_ps(
            self.config.dims,
            record,
            num_sampled=len(record.scheduled),
            provider=self.config.provider,
            gpu=self.config.gpu,
            host_model=self.config.host_model,
        )

    def __call__(self, record: StepRecord) -> StepResult | None:
        cfg = self.config
        tp_ops = step_tp_allreduces(record, cfg.dims, cfg.tp_ranks)
        moe_ops = step_moe_alltoalls(
            record, cfg.dims, cfg.ep_ranks if cfg.ep_ranks is not None else []
        )
        if not tp_ops and not moe_ops:
            return None
        estimate_ps = self.compute_estimate_ps(record)
        per_layer_calc_ns = estimate_ps // (cfg.dims.num_layers * 1000)
        trace = render_step_goal(
            record,
            cfg.dims,
            cfg.tp_ranks,
            per_layer_calc_ns,
            ep_ranks=cfg.ep_ranks,
            num_goal_ranks=cfg.num_goal_ranks,
            base_tag=cfg.base_tag,
        )
        name = f"step-{record.step_index:06d}"
        goal_bin = to_binary(trace.write(cfg.workdir / f"{name}.goal"))
        run = run_htsim_rnic(
            HtsimRnicConfig(
                goal_bin=goal_bin,
                profile=cfg.profile,
                linkspeed_bps=cfg.linkspeed_bps,
                completion_csv=cfg.workdir / f"{name}.{cfg.profile}.csv",
                topology=cfg.topology,
            )
        )
        makespan_ps = run.job_completion_time_ps()
        self.outcomes.append(
            StepNetworkOutcome(
                step_index=record.step_index,
                compute_estimate_ps=estimate_ps,
                per_layer_calc_ns=per_layer_calc_ns,
                makespan_ps=makespan_ps,
                num_flows=len(run.flows),
            )
        )
        return StepResult(
            step_index=record.step_index,
            step_latency_ps=makespan_ps,
            completed_at_ps=record.virtual_time_ps + makespan_ps,
        )
