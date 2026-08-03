"""Flow-completion-time analysis.

Raw FCT is the intermediate/debug metric of every packet-level run. For the
physical profiles (`rnic-cn`, and later DCQCN), the reviewable form is the
per-flow FCT *normalized to the `rnic-nn` baseline* of the identical GOAL:
the null-network manifold gives each flow its ideal max-min completion time,
so the ratio isolates what the fabric and the protocol added.

Scope of validity: for chained flows whose start times agree across the two
models (activation handoffs, echoes), a per-flow ratio meaningfully below 1
indicates a modeling bug, not a fast network. For phases whose members start
at model-dependent times (e.g. an incast under dynamic collective
membership, where early joiners receive large grants), per-flow ratios are
ill-posed by construction; the reviewable quantity is the phase makespan
ratio, first start to last completion (M1 finding F1, see
examples/m1/RESULTS.md).
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
