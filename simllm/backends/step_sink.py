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

Per-step subprocess invocation remains the documented *diagnostic* mode and
the default. :class:`HtsimPersistentStepSink` is the opt-in acceleration for a
finite replay whose records are known before the scheduler consumes them. It
keeps a local worker pool alive, prepares isolated diagnostic runs
concurrently, then serves their exact results in record order. The pinned
backend still accepts only one GOAL per process, so this mode deliberately
preserves per-step reset semantics. A stateful online simulator session needs
the separate backend protocol tracked as HTSIM-18.

A step with no collective work returns ``None``: the TP world has size 1
(or the dims declare no experts, or no EP group is configured) and the
record is a drain record with zero new tokens. ``None`` tells the adapter
that its own compute-only estimate stands, which is exactly right when
there is no network work to simulate. With MoE dims and ``ep_ranks``
configured, the per-step GOAL additionally carries the dispatch and
combine all-to-alls of every layer
(:func:`simllm.traffic.step_moe_alltoalls`); a non-MoE configuration
renders byte-identically to the pre-MoE sink.

Providers may opt into an exact per-layer duration breakdown. The sink checks
that it sums to the fused estimate and emits the unequal layer costs. Existing
providers inherit the optional hook's ``None`` result, retaining the original
even split byte for byte. An optional exact sample count on the record prices
the LM head correctly; its absence retains the historical scheduled-request
approximation.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Self

from simllm.backends.htsim_rnic import RNIC_PROFILES, HtsimRnicConfig, run_htsim_rnic
from simllm.compute import (
    GPU_ENVELOPES,
    ComputeProvider,
    GpuSpec,
    HostInitiationModel,
    ModelDims,
    RooflineProvider,
    step_kernel,
)
from simllm.core import StepRecord, StepResult
from simllm.goal import to_binary
from simllm.traffic import (
    RoutedMoeSupply,
    render_step_goal,
    step_moe_alltoalls,
    step_tp_allreduces,
)


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
    pre-MoE sink. ``routed_moe_supply`` optionally replaces uniform pair
    sizes with captured request routing at an explicit expert-placement
    epoch; its absence retains the uniform GOAL bytes. ``topology`` is
    optional: the null-network profiles
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
    provider: ComputeProvider = field(default_factory=lambda: RooflineProvider(efficiency=0.7))
    gpu: GpuSpec = GPU_ENVELOPES["b100"]
    host_model: HostInitiationModel = field(default_factory=HostInitiationModel)
    #: first GOAL tag; each allreduce takes a disjoint 2(W-1)-tag block
    base_tag: int = 1000
    #: explicit GOAL rank count for topology padding; None keeps inferred sizing
    num_goal_ranks: int | None = None
    #: optional captured routing and explicit placement-epoch supply
    routed_moe_supply: RoutedMoeSupply | None = None

    def __post_init__(self) -> None:
        if self.profile not in RNIC_PROFILES:
            raise ValueError(f"profile must be one of {RNIC_PROFILES}")
        if self.routed_moe_supply is not None and not isinstance(
            self.routed_moe_supply, RoutedMoeSupply
        ):
            raise TypeError("routed_moe_supply must be RoutedMoeSupply or None")


@dataclass(frozen=True)
class StepNetworkOutcome:
    """Bookkeeping for one simulated step, kept for reporting."""

    step_index: int
    #: the adapter-equivalent compute-only whole-step estimate, ps
    compute_estimate_ps: int
    #: uniform calc cost handed to GOAL, or None for an unequal breakdown
    per_layer_calc_ns: int | None
    #: simulated makespan of the step's GOAL program, ps
    makespan_ps: int
    num_flows: int
    #: emitted GOAL calc units in layer order, ns; empty on legacy construction
    layer_calc_ns: tuple[int, ...] = ()
    #: sample count used for the fused estimate
    num_sampled: int = 0
    #: whether num_sampled came from an exact record field
    sample_count_exact: bool = False
    #: whether the backend wrapper verified physical quiescence
    quiescent: bool = False
    #: realized MoE traffic mode: ``none``, ``uniform`` or ``captured``
    routing_mode: str = "uniform"
    #: selected expert placement epoch, present only for captured traffic
    placement_epoch: int | None = None

    def network_share_for(self, num_layers: int) -> float:
        """One minus represented calc time over makespan."""
        if not self.layer_calc_ns:
            if self.per_layer_calc_ns is None:
                raise ValueError("outcome has neither uniform nor ordered layer calcs")
            calc_ps = num_layers * max(self.per_layer_calc_ns, 1) * 1000
            return 1.0 - calc_ps / self.makespan_ps
        if len(self.layer_calc_ns) != num_layers:
            raise ValueError(
                f"outcome has {len(self.layer_calc_ns)} layer calcs, expected {num_layers}"
            )
        calc_ps = sum(max(calc_ns, 1) for calc_ns in self.layer_calc_ns) * 1000
        return 1.0 - calc_ps / self.makespan_ps


@dataclass(frozen=True)
class _PlannedStep:
    """Immutable handoff from serial record lowering to one backend worker."""

    step_index: int
    virtual_time_ps: int
    goal_path: Path
    completion_csv: Path
    compute_estimate_ps: int
    num_sampled: int
    sample_count_exact: bool
    per_layer_calc_ns: int | None
    layer_calc_ns: tuple[int, ...]
    routing_mode: str
    placement_epoch: int | None
    profile: str
    linkspeed_bps: int
    topology: Path | None


@dataclass(frozen=True)
class _SimulatedStep:
    """One unpublished result, safe to retain until its record is consumed."""

    result: StepResult | None
    outcome: StepNetworkOutcome | None


@dataclass(frozen=True)
class _PreparedStep:
    """A copied input record joined to its unpublished simulation result."""

    record: StepRecord
    simulation: _SimulatedStep


class HtsimStepSink:
    """Step sink that simulates each step's TP traffic on ``htsim_rnic``."""

    def __init__(self, config: HtsimStepSinkConfig) -> None:
        self.config = config
        self.config.workdir.mkdir(parents=True, exist_ok=True)
        #: one entry per simulated (non-None) step, in call order
        self.outcomes: list[StepNetworkOutcome] = []

    @staticmethod
    def _num_sampled(record: StepRecord) -> int:
        if record.num_sampled is not None:
            return record.num_sampled
        return len(record.scheduled)

    def _compute_estimate(
        self, record: StepRecord
    ) -> tuple[int, tuple[int, ...] | None]:
        """Return the whole-step estimate and optional exact layer durations."""
        cfg = self.config
        num_sampled = self._num_sampled(record)
        kernel = step_kernel(cfg.dims, record, num_sampled=num_sampled)
        fused = cfg.provider.estimate(kernel, cfg.gpu)
        host_delay_ps = cfg.host_model.delay_ps()
        estimate_ps = fused.duration_ps + host_delay_ps
        estimates = cfg.provider.estimate_layers(kernel, cfg.gpu, cfg.dims.num_layers)
        if estimates is None:
            return estimate_ps, None

        if len(estimates) != cfg.dims.num_layers:
            raise ValueError(
                "provider layer breakdown length "
                f"{len(estimates)} does not match num_layers={cfg.dims.num_layers}"
            )
        layer_ps = tuple(estimate.duration_ps for estimate in estimates)
        if any(duration_ps < 0 for duration_ps in layer_ps):
            raise ValueError("provider layer breakdown durations must be nonnegative")
        if sum(layer_ps) != fused.duration_ps:
            raise ValueError(
                "provider layer breakdown sum "
                f"{sum(layer_ps)} ps does not match fused estimate {fused.duration_ps} ps"
            )
        return estimate_ps, (layer_ps[0] + host_delay_ps, *layer_ps[1:])

    def compute_estimate_ps(self, record: StepRecord) -> int:
        """The compute-only whole-step estimate represented by the sink."""
        estimate_ps, _ = self._compute_estimate(record)
        return estimate_ps

    @staticmethod
    def _to_goal_layer_calc_ns(layer_duration_ps: Sequence[int]) -> tuple[int, ...]:
        """Truncate cumulative layer boundaries to whole GOAL nanoseconds."""
        previous_boundary_ns = 0
        cumulative_ps = 0
        layer_calc_ns = []
        for duration_ps in layer_duration_ps:
            cumulative_ps += duration_ps
            boundary_ns = cumulative_ps // 1000
            layer_calc_ns.append(boundary_ns - previous_boundary_ns)
            previous_boundary_ns = boundary_ns
        return tuple(layer_calc_ns)

    def _plan_step(self, record: StepRecord) -> _PlannedStep | None:
        """Lower and render one record without invoking either native tool."""

        cfg = self.config
        tp_ops = step_tp_allreduces(record, cfg.dims, cfg.tp_ranks)
        moe_ops = step_moe_alltoalls(
            record,
            cfg.dims,
            cfg.ep_ranks if cfg.ep_ranks is not None else [],
            routed_supply=cfg.routed_moe_supply,
        )
        if not tp_ops and not moe_ops:
            return None
        if not moe_ops:
            routing_mode = "none"
        elif all(operation.pair_payload_bytes for operation in moe_ops):
            routing_mode = "captured"
        elif all(not operation.pair_payload_bytes for operation in moe_ops):
            routing_mode = "uniform"
        else:
            raise AssertionError("one step cannot mix uniform and captured MoE traffic")
        estimate_ps, layer_duration_ps = self._compute_estimate(record)
        if layer_duration_ps is None:
            per_layer_calc_ns = estimate_ps // (cfg.dims.num_layers * 1000)
            layer_calc_ns = (per_layer_calc_ns,) * cfg.dims.num_layers
            rendered_calc_ns: int | Sequence[int] = per_layer_calc_ns
        else:
            layer_calc_ns = self._to_goal_layer_calc_ns(layer_duration_ps)
            per_layer_calc_ns = (
                layer_calc_ns[0]
                if all(value == layer_calc_ns[0] for value in layer_calc_ns)
                else None
            )
            rendered_calc_ns = layer_calc_ns
        trace = render_step_goal(
            record,
            cfg.dims,
            cfg.tp_ranks,
            rendered_calc_ns,
            ep_ranks=cfg.ep_ranks,
            routed_supply=cfg.routed_moe_supply,
            num_goal_ranks=cfg.num_goal_ranks,
            base_tag=cfg.base_tag,
        )
        name = f"step-{record.step_index:06d}"
        goal_path = trace.write(cfg.workdir / f"{name}.goal")
        return _PlannedStep(
            step_index=record.step_index,
            virtual_time_ps=record.virtual_time_ps,
            goal_path=goal_path,
            completion_csv=cfg.workdir / f"{name}.{cfg.profile}.csv",
            compute_estimate_ps=estimate_ps,
            num_sampled=self._num_sampled(record),
            sample_count_exact=record.num_sampled is not None,
            per_layer_calc_ns=per_layer_calc_ns,
            layer_calc_ns=layer_calc_ns,
            routing_mode=routing_mode,
            placement_epoch=(
                moe_ops[0].placement_epoch if routing_mode == "captured" else None
            ),
            profile=cfg.profile,
            linkspeed_bps=cfg.linkspeed_bps,
            topology=cfg.topology,
        )

    @staticmethod
    def _execute_plan(plan: _PlannedStep) -> _SimulatedStep:
        """Compile and execute one plan using the accepted diagnostic path."""

        goal_bin = to_binary(plan.goal_path)
        run = run_htsim_rnic(
            HtsimRnicConfig(
                goal_bin=goal_bin,
                profile=plan.profile,
                linkspeed_bps=plan.linkspeed_bps,
                completion_csv=plan.completion_csv,
                topology=plan.topology,
            )
        )
        makespan_ps = run.job_completion_time_ps()
        outcome = StepNetworkOutcome(
            step_index=plan.step_index,
            compute_estimate_ps=plan.compute_estimate_ps,
            num_sampled=plan.num_sampled,
            sample_count_exact=plan.sample_count_exact,
            per_layer_calc_ns=plan.per_layer_calc_ns,
            layer_calc_ns=plan.layer_calc_ns,
            makespan_ps=makespan_ps,
            num_flows=len(run.flows),
            quiescent=run.quiescent,
            routing_mode=plan.routing_mode,
            placement_epoch=plan.placement_epoch,
        )
        result = StepResult(
            step_index=plan.step_index,
            step_latency_ps=makespan_ps,
            completed_at_ps=plan.virtual_time_ps + makespan_ps,
        )
        return _SimulatedStep(result=result, outcome=outcome)

    def _simulate_step(self, record: StepRecord) -> _SimulatedStep:
        plan = self._plan_step(record)
        if plan is None:
            return _SimulatedStep(result=None, outcome=None)
        return self._execute_plan(plan)

    def _publish(self, simulation: _SimulatedStep) -> StepResult | None:
        if simulation.outcome is not None:
            self.outcomes.append(simulation.outcome)
        return simulation.result

    def __call__(self, record: StepRecord) -> StepResult | None:
        return self._publish(self._simulate_step(record))


class HtsimPersistentStepSink(HtsimStepSink):
    """Opt-in prepared replay using a persistent local worker pool.

    ``prepare`` copies and lowers a finite record sequence before the scheduler
    consumes it. Native compilation and the unchanged one-GOAL simulator path
    then run concurrently. Calls must replay the prepared values in exact
    order; no prediction, fallback, or record substitution is permitted.

    The executor survives across fully consumed batches until ``close``. Use a
    context manager when possible so worker shutdown belongs to the timed
    end-to-end boundary.
    """

    def __init__(self, config: HtsimStepSinkConfig, *, max_workers: int) -> None:
        super().__init__(config)
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="simllm-htsim",
        )
        self._prepared: deque[_PreparedStep] = deque()
        self._state_lock = Lock()
        self._preparing = False
        self._closed = False

    @property
    def prepared_steps_remaining(self) -> int:
        with self._state_lock:
            return len(self._prepared)

    def prepare(self, records: Sequence[StepRecord]) -> None:
        """Prepare one finite replay atomically from the caller's perspective."""

        copied_records = tuple(copy.deepcopy(record) for record in records)
        if not copied_records:
            raise ValueError("records must not be empty")
        if any(not isinstance(record, StepRecord) for record in copied_records):
            raise TypeError("records must contain StepRecord values")
        step_indices = [record.step_index for record in copied_records]
        if len(step_indices) != len(set(step_indices)):
            raise ValueError("prepared records must have unique step indices")

        with self._state_lock:
            if self._closed:
                raise RuntimeError("persistent step sink is closed")
            if self._preparing:
                raise RuntimeError("a preparation is already in progress")
            if self._prepared:
                raise RuntimeError("consume the prepared replay before preparing another")
            self._preparing = True

        futures: list[Future[_SimulatedStep] | None] = []
        try:
            plans = tuple(self._plan_step(record) for record in copied_records)
            futures = [
                self._executor.submit(self._execute_plan, plan)
                if plan is not None
                else None
                for plan in plans
            ]
            wait(tuple(future for future in futures if future is not None))
            simulations = tuple(
                future.result()
                if future is not None
                else _SimulatedStep(result=None, outcome=None)
                for future in futures
            )
        except BaseException:
            submitted = tuple(future for future in futures if future is not None)
            for future in submitted:
                future.cancel()
            wait(submitted)
            with self._state_lock:
                self._preparing = False
            raise

        prepared = deque(
            _PreparedStep(record, simulation)
            for record, simulation in zip(copied_records, simulations, strict=True)
        )
        with self._state_lock:
            self._prepared = prepared
            self._preparing = False

    def __call__(self, record: StepRecord) -> StepResult | None:
        with self._state_lock:
            if self._preparing:
                raise RuntimeError("prepared results are not available yet")
            if not self._prepared:
                raise RuntimeError("prepare records before calling the persistent sink")
            prepared = self._prepared[0]
            if record != prepared.record:
                raise ValueError(
                    "record does not match the next prepared step "
                    f"{prepared.record.step_index}"
                )
            self._prepared.popleft()
            return self._publish(prepared.simulation)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> Self:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("persistent step sink is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
