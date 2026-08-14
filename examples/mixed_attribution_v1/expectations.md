# Mixed attribution v1 expectations

This document freezes the acceptance contract for BACK-43 before the behavior
exists, before the harness that produces a measured number exists, and before
any result-producing run. It is the pre-run record required by the validation
discipline.

## Working-tree status at freeze time

The worktree is on branch `codex/back43_mixed_attribution` at base commit
`e18b9b0102808e9b8e0f276c2b82c51ed8c5b51d`, with no modified, added or deleted
tracked file. The only files this commit adds are this document and the
BACK-43 registry entry in `docs/modules/backends.md`. No implementation of the
behavior, no harness, no measured value and no generated result is present.

Two probes ran against the unmodified base before this freeze, and both are
disclosed here as run-configuration evidence rather than as results:

1. a structural probe of the fixture below, which reported only how many
   communication phases of each step are local-only, fabric-only or mixed
   under the chosen placement, and no service time, latency or attribution;
2. a feasibility and timing probe of one step of a configuration that is not
   a cell of this study (vector 1,024 rather than 2,048), which reported that
   the sink runs such a step in about 4 seconds with 24 backend invocations.

Neither probe read a quantity this document predicts, and neither probe
invoked the per-request reducer, which refuses every NVLink-bearing step on
the base commit.

## The blocker, stated from the source

`simllm/backends/step_attribution.py:73` refuses the whole step:

```python
    if locality.nvlink_directed_bytes or locality.nvlink_service_ps:
        raise ValueError(
            "packet-level request attribution requires the all-remote level; "
            "a mixed NVLink and fabric artifact has no single-resource owner"
        )
```

Its stated reason is in the module docstring at
`simllm/backends/step_attribution.py:20` to `:23`: the composed service of a
mixed artifact "is a maximum over two resources and this module would have to
choose one of them without evidence". The consequence is that a physically
honest placement, i.e. any placement in which two ranks share a node, takes
the per-request TTFT and TPOT reducer offline. Nothing downstream of it can
reach the reported metric chain, so intra-node NVLink work is unreachable
evidence today.

The reason was correct about the evidence, and wrong about the conclusion.
The evidence exists one layer up, and this study's implementation half
publishes it rather than approximating it.

### The composition the attribution must partition

`simllm/backends/step_sink.py:1150` to `:1159` composes and sums:

```python
            fabric_services.append(fabric_service_ps)
            composed_services.append(
                artifact.collective_base_latency_ps
                + max(artifact.local_service_ps, fabric_service_ps)
            )
            artifact_offset_ps += composed_services[-1]
        ...
        makespan_ps = represented_compute_ps + sum(composed_services)
```

Three facts follow, and they are the whole design:

- **The artifacts are serial.** The step's realized elapsed interval is the
  ordered concatenation of one disjoint subinterval per artifact. Partitioning
  by artifact is therefore critical-path selection, not a sum over
  concurrent visits, and it satisfies the additive-decomposition rule in
  AGENTS.md rather than evading it.
- **Within one artifact the two media compose by maximum, not by sum.** The
  artifact ends when the slower of the two resources drains. The resource
  whose own service equals that maximum is the one that decided the finish
  time; it is the artifact's owner, and its whole realized duration lies on
  the critical path. The other medium's service is masked, is off the
  critical path, and must never be added to TTFT.
- **`local_service_ps` carries two different physical meanings.** For a
  compute artifact it is GPU kernel service, `simllm/backends/step_sink.py:823`
  and `:925` and `:994` (`max(duration, 1_000)` over the artifact's non
  collective operations). For a collective phase it is the analytic NVLink
  endpoint serializer, `simllm/backends/step_sink.py:959`
  (`local_service_ps=phase.nvlink_service_ps`). A per-artifact medium label is
  therefore required evidence, not a convenience: without it the attribution
  cannot tell a kernel from an NVLink serializer.

### What the locality projection already charges

`simllm/traffic/locality.py:62` to `:84` builds an explicit per-endpoint ledger
of both directions, and `simllm/traffic/locality.py:254` to `:266` charges each
endpoint `ceil(max(egress_bytes, ingress_bytes) * 1e9 / bandwidth)` whole
nanoseconds, the serial phase taking the largest endpoint.

This refutes one premise of the task brief. The brief states that
`locality.py` "charges max source egress with no ingress term". That was true
before CORE-41 (`78d8c14`, "Charge the maximum local endpoint load") and is
not true at the base commit: the ingress term is present, the ledger is
validated to conserve local bytes in both directions
(`simllm/traffic/locality.py:228` to `:233`), and the full-duplex maximum is
the declared model with the half-duplex sum explicitly rejected in the module
docstring at `simllm/traffic/locality.py:22` to `:32`.

The interaction that does remain, and that this study names rather than builds
on silently, is on the other side of the comparison: the cross-node path still
has no destination-ingress serializer, which is CORE-48 in
`docs/modules/core.md:840` and is recorded again at
`docs/modules/traffic.md:261`. The ownership rule compares an ingress-aware
NVLink term against a fabric term that under-charges a converging combine, so
an artifact near the crossing point can be assigned to NVLink when a fully
modeled fabric would have owned it. Every cell of this study sits far from
that crossing point, by the napkin bounds below, and the residual is
registered rather than absorbed.

### Where today's behavior is locked

- `tests/test_step_attribution.py:99` locks the refusal itself and must be
  replaced by the tests of the new contract.
- `tests/test_step_attribution.py:72` to `:159` lock the all-remote partition
  and the reducer's conserved intervals with exact literals.
- `examples/end_to_end_replay_v1/run_study.py:783` and
  `examples/sglang_end_to_end_v1/run_study.py:812` call `attribute_step` on the
  all-remote path, and `examples/host_step_cost_v1/run_study.py:454` drives the
  reducer. Their accepted results report zero NVLink bytes in every simulated
  step (`examples/sglang_end_to_end_v1/RESULTS.md`, guard G5), so they exercise
  exactly the branch that must not move.

## Frozen contract for the implementation half

The names below are frozen now so the result report cannot rename a component
after seeing a number.

A step's realized interval is partitioned into these components, and the same
component names carry through the per-request TTFT and decode partitions:

| component | owner |
|---|---|
| `queue_ps` | scheduler gap before the step was released; no resource |
| `kernel_ps` | GPU compute artifacts |
| `nvlink_ps` | artifacts whose NVLink endpoint serializer decided the finish |
| `fabric_ps` | artifacts whose packet-level fabric service decided the finish |
| `co_critical_ps` | artifacts where both media realized the same duration |
| `collective_base_ps` | semantic collective fixed cost; no resource owner |
| `control_ps` | steps the sink did not simulate |

Masked service, i.e. `min(local, fabric)` over collective artifacts, is
published under its own name as an additive work sum, is never part of any
total, and may exceed wall time. The coarse `LatencyAttribution` keeps its
existing meaning, so `collective_ps` remains the roll-up of `nvlink_ps`,
`fabric_ps`, `co_critical_ps` and `collective_base_ps`, and the all-remote
path keeps its exact accepted values.

## Frozen fixture

One tracked capture, one request, one geometry, three placements, two NVLink
rates.

| element | selection |
|---|---|
| routing capture | `examples/preplay_trace_v1/granite_length_cap.jsonl`, request `length-cap`, 22 prefill tokens, 24 layers, top-8 of 32 experts |
| step records | 3 steps, each one prefill token of that capture, `context_length` 1, 2, 3, `num_sampled` 1, arrival 0, each step released at the previous step's completion |
| geometry | `hidden_size` 1,024 with `dtype_bytes` 2, so one hidden vector is 2,048 bytes |
| parallelism | `tp_ranks=(0,)`, `ep_ranks=(0, 1, 2, 3)`, engine rank 0, 4 GOAL ranks |
| compute | declared fixed 24,000 ps per step, i.e. 24 compute artifacts of 1,000 ps; a mechanism fixture, never a device calibration |
| fabric | `rnic-nn-fluid` on `htsim_rnic`, 400 Gbit/s |
| NVLink rate | 450,000,000,000 and 225,000,000,000 bytes per second per endpoint direction |

Two expert layouts, both declared fixtures:

- `uniform`: rank `r` owns experts `[8r, 8r+8)` at every layer, the layout of
  `examples/nvlink_locality_v1`.
- `node-local-even`: at even layers rank 1 owns exactly the eight experts the
  capture's first prefill token selects at that layer, and the remaining 24 are
  split eight each over ranks 0, 2 and 3; odd layers keep the uniform layout.
  This layout is chosen, not measured, so that a single step carries both
  fully intra-node phases and fabric-crossing phases. The structural probe
  above confirmed the phase counts it produces before this freeze.

Three host layouts: `AAAA` (one node), `AABB` (two nodes of two ranks) and
`ABCD` (four single-rank nodes).

### Cells

| label | hosts | expert layout | NVLink bytes/s | backend |
|---|---|---|---:|---|
| `all-remote-450` | `ABCD` | uniform | 450,000,000,000 | `htsim_rnic` |
| `all-local-450` | `AAAA` | uniform | 450,000,000,000 | none, analytic |
| `all-local-225` | `AAAA` | uniform | 225,000,000,000 | none, analytic |
| `mixed-450` | `AABB` | node-local-even | 450,000,000,000 | `htsim_rnic` |
| `mixed-225` | `AABB` | node-local-even | 225,000,000,000 | `htsim_rnic` |

Two parameters vary, as the validation discipline requires: placement locality
across three host layouts, and NVLink bandwidth across a factor of two.

## Napkin bounds, written before any digit is read

Byte inventory, from the fixture alone. One token, top-8 of 32 experts, one
2,048-byte hidden vector per distinct destination rank. A dispatch phase
therefore moves at least 2,048 and at most 6,144 bytes out of rank 0, its
combine phase mirrors that into rank 0, and a step has at most 48 phases. Per
step the directed payload is between 98,304 and 294,912 bytes.

**NVLink floor and ceiling.** One endpoint's service is
`ceil(peak_endpoint_bytes * 1e9 / rate)` whole nanoseconds. At 450 GB/s a
phase costs at least `ceil(2048 * 1e9 / 450e9) = 5` ns and at most
`ceil(6144 * 1e9 / 450e9) = 14` ns, so a 48-phase all-local step carries
between 240,000 and 672,000 ps of NVLink service. No all-local step can
complete faster than 264,000 ps including the 24,000 ps of declared compute,
and none should exceed 696,000 ps.

**Fabric floor and ceiling.** A 2,048-byte message cannot cross a 400 Gbit/s
link faster than 40.96 ns, and three of them out of one source cannot beat
122.88 ns. The ideal-network model adds a propagation term measured
independently at 2.000 us per artifact and published in
`examples/sglang_end_to_end_v1/RESULTS.md`. A fabric-bearing phase therefore
costs between about 2.041 and 2.123 us, an all-remote step with 48 such phases
between about 98.0 and 101.9 us, and this document registers the wider
interval `[96,000,000, 110,000,000]` ps for the all-remote step service to
absorb any propagation term that is not exactly 2.000 us. That interval is
stated against the base commit's propagation constant; wave-14 work on the
collective intercept would move the interval without touching any ownership
relation in this document.

**The crossing point, and why every cell sits far from it.** An artifact is
NVLink-owned only when its NVLink service exceeds its fabric service. With a
per-phase fabric service near 2.0 us and a per-phase NVLink service near 5 to
14 ns, a mixed-media artifact would need roughly 150 to 400 times more local
bytes than it carries, or an NVLink rate roughly 150 to 400 times lower, to
change owner. No cell approaches that. The physical prediction that follows is
sharp and is scored below: an artifact that carries any fabric segment is
fabric-owned in every cell, and NVLink ownership arises only from phases that
carry no fabric segment at all.

**System plausibility.** These are microsecond-scale steps for a single token
of a 400M-active-parameter model, because compute is a declared 24,000 ps
fixture rather than a device model. A real B100 decode step for this geometry
prices near 99 us of compute, an order of magnitude above the whole all-local
step here. This study therefore makes no claim about the realistic ratio of
kernel to collective time; it claims only that the components are the right
ones and that they add up.

## Registered fatal guards, unscored

Fatal means void, not a lost point. One violated guard voids the run for the
purpose of closing BACK-43, and no behavioral fraction is then interpretable.
These are never reported as a fraction.

- **G1 all-remote byte identity.** For `all-remote-450`, the reducer output
  computed from the enriched locality outcome equals, field for field, the
  output computed from the same outcome with the three new per-artifact tuples
  cleared, which is exactly the input shape the base commit's code sees. Every
  step attribution, every published metric row, and every per-request total
  including TTFT, TPOT as an exact `Fraction`, token counts and both
  attributions must be equal. Any difference voids the run.
- **G2 conservation.** In every interval of every cell the medium partition
  totals the same picoseconds as the coarse attribution, which totals the
  request's realized elapsed interval, with no remainder. The step partition
  totals `step_latency_ps`.
- **G3 roll-up agreement.** In every interval `kernel_ps` agrees between the
  two views, and `collective_ps` equals
  `nvlink_ps + fabric_ps + co_critical_ps + collective_base_ps`.
- **G4 inactive components.** `kv_ps`, `dma_ps`, `nic_ps` and `control_ps` are
  exactly zero in every interval of every cell. Configuration forced.
- **G5 backend health.** Every simulated step of every cell reports routing
  mode `captured`, placement epoch 0 and backend quiescence, and every cell
  reduces exactly three steps with strictly positive TTFT.
- **G6 projection agreement.** For every step the four per-artifact tuples have
  equal length, and the sum of local services over NVLink-medium artifacts
  equals the published `nvlink_service_ps`. The implementation raises when
  this fails, so a violation appears as a failed cell rather than a wrong
  number.
- **G7 all-remote has no NVLink component.** For `all-remote-450` the NVLink
  and masked-NVLink components are exactly zero in every interval. This is
  entailed by G6 plus the ownership rule, which is why it is registered as a
  fatal guard here rather than scored below.
- **G8 all-local runs no backend.** `all-local-450` and `all-local-225` report
  zero backend invocations, zero fabric bytes and zero fabric service.
  Configuration forced.

## Registered scored relations

Two evidence classes, never summed with each other and never with the guards
above. The pre-freeze entailment question is answered for each relation: given
the guards already registered, can this relation fail?

### Exact-oracle relations

- **E1 independent recomputation.** For every cell, every step and every
  reduced interval, a recomputation that uses only the Python standard library,
  the published per-artifact tuples and the declared arrivals reproduces the
  reducer's owner for every artifact, its seven medium components, its coarse
  attribution, TTFT, TPOT as an exact `Fraction`, and both per-request
  partitions, exactly.

  **Can E1 fail?** Yes. It shares the per-artifact inputs with the reducer, so
  it does not test the sink's composition, and that limit is stated here rather
  than discovered later. It does test every step the reducer performs on top of
  those inputs: owner selection, masking, the pending-interval carry across a
  step that samples no token, the queue-gap charge, and the accumulation into
  TTFT and decode partitions. A defect in any of those changes a published
  number without changing an input.

### Behavioral relations

- **F1 all-local composition.** In both all-local cells, for every step, the
  fabric component is exactly zero, the kernel component is exactly 24,000 ps,
  the NVLink component lies in `[240,000, 672,000]` ps, and the step's realized
  latency equals kernel plus NVLink exactly.

  **Can F1 fail?** Yes. No registered guard pins the NVLink component's value.
  The interval is derived from the per-phase byte inventory and the ceiling
  rule; a floor instead of a ceiling, an egress-only ledger, a per-segment
  instead of per-endpoint charge, or a sum instead of a maximum over endpoints
  each move the measured value out of the interval or off the exact equality.

- **F2 the mixed-owner step.** For `mixed-450`, in the first reduced interval,
  i.e. the TTFT interval of step 0, the NVLink component is exactly 120,000 ps,
  the fabric component is strictly positive, the kernel component is exactly
  24,000 ps, and the three plus the zero queue gap total the request's TTFT.
  The same step reports at least 24 NVLink-owned artifacts and at least 24
  fabric-owned artifacts.

  **Can F2 fail?** Yes, and it is the relation that decides whether BACK-43 is
  solved. Under the base commit's coarse rule the interval does not exist at
  all. Under any rule that keeps step-level rather than artifact-level
  ownership the NVLink component is zero. The exact value follows from 24
  fully intra-node phases each carrying one 2,048-byte vector in each
  direction, `ceil(2048 * 1e9 / 450e9) = 5` ns per phase: a half-duplex
  `egress + ingress` charge would give 240,000 ps, a per-segment charge or a
  different tie rule would give something else again.

- **F3 masked service is not additive.** `mixed-225` and `mixed-450` differ by
  exactly 120,000 ps in TTFT and in the NVLink component, their fabric
  components are byte-identical, and no artifact changes owner. In both
  all-local cells the NVLink component at 225 GB/s lies in
  `[2 * n450 - 48,000, 2 * n450]` ps where `n450` is the same cell's NVLink
  component at 450 GB/s, and the latency difference equals the NVLink component
  difference exactly.

  **Can F3 fail?** Yes, and this is the relation that catches the most
  tempting wrong implementation. Every mixed artifact's masked NVLink service
  also doubles when the rate halves. If masked service were added into the
  total instead of being reported separately, the mixed delta would exceed
  120,000 ps by the masked doubling. If the fabric service depended on the
  NVLink constant, the fabric components would differ. If the ceiling
  quantization were mishandled, the all-local bracket would be missed.

- **F4 locality ordering.** Across the three 450 GB/s cells the TTFT order is
  `all-local-450 < mixed-450 < all-remote-450`, the fabric component is
  strictly increasing in that order with an exact zero in the first cell, and
  `all-remote-450`'s step service lies in `[96,000,000, 110,000,000]` ps.

  **Can F4 fail?** Yes. The ordering is a physical expectation, not a guard:
  nothing in the implementation forces a placement with fewer fabric bytes to
  produce a smaller fabric component, and the registered all-remote interval is
  an independent first-principles bound that the measured value can miss in
  either direction.

## Failure and disclosure rules, frozen now

- Every attempted run is reported, including runs abandoned for a harness
  defect, with the reason.
- A violated fatal guard is reported as a void run with findings; the scored
  relations are then reported as observations without a fraction, and BACK-43
  stays open.
- No frozen literal in this document is edited after a run. A relation that
  fails is reported as failed and its refutation is the result.
- If the mixed cells produce no NVLink-owned artifact, F2 fails and the honest
  finding is that the chosen fixture does not exhibit the case; the
  implementation contract is unchanged, and the residual is registered rather
  than repaired by re-picking the fixture.
