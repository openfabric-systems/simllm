"""Flow and phase completion-time analysis for identical GOAL workloads.

Raw FCT is a debug metric. ``normalized_fct`` matches the physical profile
against ``rnic-nn`` without changing or filtering either flow population.
For aligned-start flows it is a lower-bound comparison only when a flow does
not share its bottleneck link with another aligned flow, including the
uncontended chained handoff/echo cases. A ratio meaningfully below one there
indicates a modeling bug. Under shared receiver service, a fair allocation
can complete an individual flow later than a different packet scheduler;
per-flow normalization is diagnostic (BACK-68, aligned_baseline_v1).

The reviewable shared-phase quantity is ``normalized_phase_makespan``:
physical over ideal, first start to last completion of the identical GOAL
phase. Check its baseline floor and ``earliest_completion_byte_floors`` on
each receiver link. These conservation guards are separate from behavioral
slowdown targets. Callers must verify identical GOAL phases, link capacity,
path propagation and successful quiescence; completion rows cannot establish
those facts. A byte floor is necessary, not sufficient, for correct service.

Model-dependent start stagger makes per-flow ratios ill-posed as lower bounds
(M1 finding F1). Phase normalization still uses each run's own first start.
Do not compare arbitrary subsets as if they were the complete GOAL phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from simllm.backends.htsim_rnic import FlowCompletion


@dataclass(frozen=True)
class NormalizedFct:
    source: int
    destination: int
    tag: int
    payload_bytes: int
    fct_ps: int
    baseline_fct_ps: int

    @property
    def slowdown(self) -> float:
        return self.fct_ps / self.baseline_fct_ps


def _keyed(flows: list[FlowCompletion]) -> dict[tuple[int, int, int], list[FlowCompletion]]:
    keyed: dict[tuple[int, int, int], list[FlowCompletion]] = {}
    for f in flows:
        keyed.setdefault((f.source, f.destination, f.tag), []).append(f)
    for group in keyed.values():
        group.sort(key=lambda f: f.start_time_ps)
    return keyed


def normalized_fct(
    flows: list[FlowCompletion], baseline: list[FlowCompletion]
) -> list[NormalizedFct]:
    """Match flows to their baseline counterparts by (source, destination, tag).

    Groups with several flows are paired in start-time order. The two runs
    must come from the same GOAL: any unmatched flow is an error, silence
    would hide a lost or duplicated transfer.
    """
    keyed, base_keyed = _keyed(flows), _keyed(baseline)
    if keyed.keys() != base_keyed.keys():
        raise ValueError(
            f"flow sets differ: only-run={sorted(keyed.keys() - base_keyed.keys())} "
            f"only-baseline={sorted(base_keyed.keys() - keyed.keys())}"
        )
    out = []
    for key, group in sorted(keyed.items()):
        base_group = base_keyed[key]
        if len(group) != len(base_group):
            raise ValueError(f"flow multiplicity differs at {key}")
        for f, b in zip(group, base_group):
            if f.payload_bytes != b.payload_bytes:
                raise ValueError(f"payload differs at {key}: {f.payload_bytes} vs {b.payload_bytes}")
            out.append(NormalizedFct(
                source=f.source, destination=f.destination, tag=f.tag,
                payload_bytes=f.payload_bytes,
                fct_ps=f.fct_ps, baseline_fct_ps=b.fct_ps,
            ))
    return out


@dataclass(frozen=True)
class NormalizedPhaseMakespan:
    makespan_ps: int
    baseline_makespan_ps: int

    @property
    def slowdown(self) -> float:
        return self.makespan_ps / self.baseline_makespan_ps


@dataclass(frozen=True)
class CompletionByteFloor:
    k: int
    source: int
    destination: int
    tag: int
    flow_id: int
    cumulative_bytes: int
    elapsed_ps: int
    floor_ps: int

    @property
    def slack_ps(self) -> int:
        return self.elapsed_ps - self.floor_ps

    @property
    def ok(self) -> bool:
        return self.slack_ps >= 0


def _phase_span(flows: list[FlowCompletion]) -> int:
    if not flows:
        raise ValueError("phase must contain completions")
    if any(f.start_time_ps < 0 or f.payload_bytes <= 0 or f.fct_ps <= 0 or
           f.completion_time_ps - f.start_time_ps != f.fct_ps for f in flows):
        raise ValueError("phase requires positive payloads and consistent timestamps")
    return max(f.completion_time_ps for f in flows) - min(f.start_time_ps for f in flows)


def normalized_phase_makespan(
    flows: list[FlowCompletion], baseline: list[FlowCompletion]
) -> NormalizedPhaseMakespan:
    """Normalize full phase spans, validating message identities, bytes and counts.

    Matching uses the same start-order multiplicity contract as normalized_fct.
    The caller establishes identical GOAL semantics, including dependencies.
    Start times may differ between runs; neither span is a sum of flow FCTs.
    """
    span, baseline_span = _phase_span(flows), _phase_span(baseline)
    normalized_fct(flows, baseline)
    return NormalizedPhaseMakespan(span, baseline_span)


def earliest_completion_byte_floors(
    flows: list[FlowCompletion], *, link_rate_bps: int, propagation_ps: int
) -> list[CompletionByteFloor]:
    """Check every completion prefix on one receiver's single ingress link.

    k completed messages require at least their total payload service plus
    one common propagation delay after the first phase start. Use the minimum
    path propagation for mixed paths, never the largest. This conservative
    bound also supports staggered releases. Payload service is rounded up to
    an integer ps using exact arithmetic. Headers and partly served messages
    are omitted, so these can only strengthen the true lower bound.

    Ties sort by identity and flow ID, retaining a deterministic row for every
    k, including the full simultaneous-completion prefix. Invalid inputs raise;
    physically impossible completions are returned with ok=False for auditing.
    """
    if type(link_rate_bps) is not int or link_rate_bps <= 0:
        raise ValueError("link_rate_bps must be a positive integer")
    if type(propagation_ps) is not int or propagation_ps < 0:
        raise ValueError("propagation_ps must be a nonnegative integer")
    _phase_span(flows)
    if len({f.destination for f in flows}) != 1:
        raise ValueError("prefix floors require one receiver link")
    start = min(f.start_time_ps for f in flows)
    total, rows = 0, []
    for k, f in enumerate(sorted(flows, key=lambda f: (
        f.completion_time_ps, f.source, f.destination, f.tag, f.flow_id)), 1):
        total += f.payload_bytes
        service = (total * 8_000_000_000_000 + link_rate_bps - 1) // link_rate_bps
        rows.append(CompletionByteFloor(
            k, f.source, f.destination, f.tag, f.flow_id, total,
            f.completion_time_ps - start, service + propagation_ps,
        ))
    return rows
