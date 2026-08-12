# Endpoint versus fabric serializer cross-check, expectations

This is the expectations-only record for CORE-43. It freezes the sweep, the
closed-form agreement band and every derived literal before the study runs.
Nothing here is an implementation of the comparison, and no measured fabric
value appears below.

CORE-43 asks one question: two independently written serializers, the analytic
intra-node endpoint charge in `simllm/traffic/locality.py` and the packet-level
fluid manifold in the htsim backend, are handed the same directed traffic. Do
they agree?

## What is being compared, and why it can disagree

The analytic charge is a closed-form Python expression. For one serial
communication phase it builds a per-endpoint ledger of egress and ingress
bytes, takes the full-duplex load `max(egress, ingress)` at every endpoint,
converts the largest of them to whole GOAL nanoseconds and charges that as the
phase service. It has no propagation term, no packet, and no contention state.

The fluid manifold is a C++ progressive max-min allocator. Every active flow
shares its source uplink and its destination downlink; the ideal max-min rate
vector is solved in exact rational arithmetic and then each rate is rounded
down independently to whole bits per second; a flow's remaining service debt is
divided by its rate and rounded up to whole picoseconds; one fixed propagation
delay is added after the last serviced bit.

Nothing in either implementation refers to the other. They can disagree for at
least these reasons, and the point of the study is to find out whether they do:

- The analytic model could charge the wrong direction, the wrong endpoint or
  the wrong aggregate (source egress only, or total group bytes).
- The fluid manifold could fail to keep the bottleneck endpoint busy, in which
  case its makespan would exceed the bytes-over-rate floor by a margin far
  larger than its rounding.
- The live composition could place either serializer on the wrong artifact, so
  that a correct component number never reaches a step latency.

## Sweep

Two parameters vary, as the validation discipline requires.

| Parameter | Values |
|---|---|
| Physical placement | all-local (one host, analytic serializer), all-remote (eight hosts, fluid serializer) |
| Link rate | 400 Gbit/s, 200 Gbit/s |

The traffic is fixed and is the real Granite MoE capture at EP width eight:
all 32 recorded scheduler steps, 48 collective phases per step, replayed from
the recorded step records and the captured routing projection under
`$SIMLLM_MOE_E2E_ROOT`. The rank-to-host assignment is the only difference
between the two placements; the graph, the segment set and the compute term are
identical.

Two further arms are executed for narrower purposes:

- One all-local arm at the declared NVLink rate of 450,000,000,000 bytes per
  second, which is the deployment-realistic single-node configuration, used
  only to report the live TTFT and TPOT effect of running this traffic on
  NVLink instead of the fabric.
- One all-remote identity control at 400 Gbit/s in which the analytic NVLink
  bandwidth is changed from 50,000,000,000 to 450,000,000,000 bytes per second.
  The all-remote path has no local segment, so every artifact, timestamp, byte
  count and flow row must be identical. This is by construction and is
  fatal-unscored.

## Rate matching, which is what makes the comparison meaningful

The two serializers are compared at exactly the same rate. The fluid manifold
is configured in bits per second and the analytic charge in bytes per second,
so the matched pairs are

| Link rate | Fluid `linkspeed_bps` | Analytic bytes/s | Picoseconds per byte |
|---:|---:|---:|---:|
| 400 Gbit/s | 400,000,000,000 | 50,000,000,000 | 20 |
| 200 Gbit/s | 200,000,000,000 | 25,000,000,000 | 40 |

Both picosecond-per-byte constants are exact integers, so no rounding enters
the comparison from the rate itself.

## Notation

For one phase `p`:

- `L_p` is the peak endpoint load in bytes, the largest of
  `max(egress, ingress)` over that phase's endpoints.
- `n_p` is the number of directed segments in the phase.
- `k` is the picoseconds per byte of the matched rate, 20 or 40.
- `A_p` is the analytic phase service in picoseconds.
- `F_p` is the fluid artifact completion for that phase in picoseconds.
- `PROP` is the fixed propagation delay the fluid manifold adds once after the
  last serviced bit, `2,000,000` ps, the htsim default
  `fixed_propagation_delay_ps`. It is a property of the manifold, not of the
  rate, so it is identical at 400 and 200 Gbit/s.

## Registered agreement band, derived from arithmetic

**Analytic side, exact.** GOAL calc units are whole nanoseconds and `ceil` is
monotone, so the maximum over endpoints of the per-endpoint ceiling equals the
ceiling of the maximum:

    A_p = ceil(k * L_p / 1000) * 1000            (exact identity)
    k * L_p  <=  A_p  <=  k * L_p + 999

**Fluid side, floor.** No link serves faster than its capacity, so the
bottleneck endpoint alone forces

    F_p - PROP  >=  k * L_p                       (physical floor)

**Fluid side, ceiling.** Two rounding rules can only delay the manifold. Each
rate is rounded down by less than one bit per second, which over a phase of
`k * L_p` picoseconds costs less than `k * L_p * n_p / (C - n_p)` picoseconds,
below one picosecond for every cell in this sweep. Each allocation epoch ends
at a picosecond ceiling, costing at most one picosecond, and a phase has at
most `n_p` epochs because an epoch ends only when a flow completes. Therefore

    F_p - PROP  <=  k * L_p + n_p

**The registered band** is the difference of the two, per phase:

    -n_p  <=  A_p - (F_p - PROP)  <=  999

This is the whole claim. It is tight: at `k = 20` and a phase of half a
megabyte the two serializers are being required to agree to better than one
part in ten thousand, and the ceiling `n_p` is at most 7 picoseconds on a value
of order ten microseconds.

If the band is violated the study does not widen it. It reports which side left
its own bound: a violation of the physical floor is a fabric-side defect, a
violation above `999` with the floor intact is an analytic-side defect, and a
violation of the fluid ceiling with the floor intact is a manifold rounding or
scheduling defect to be attributed to the allocator.

## Input characterization, computed before this freeze

The directed traffic is an input, not an outcome. It was computed before this
freeze with pure Python from the recorded step records and captured routing,
with no backend, no artifact and no timing. It is recorded here so the derived
literals below are auditable, and it is **not** scored: an input that the
freeze already fixes cannot also be evidence.

- 32 steps, 48 phases per step (24 MoE layers, one dispatch and one combine
  phase each), 1,536 phases in total.
- Every phase is a star hubbed on the engine rank 0: a dispatch phase has rank
  0 as its only source, a combine phase has rank 0 as its only destination.
  Consequently `L_p` equals the phase's total directed bytes.
- At most 7 directed segments per phase, so `n_p <= 7` everywhere.
- Prefill step 0: 336 segments, `sum(L_p) = 25,563,136` bytes, and the
  superseded source-egress-only aggregate `sum(peak egress) = 15,249,408`
  bytes. Their ratio is 1.676336, which is the capture-scale undercharge CORE-43
  names.
- All 32 steps: 9,108 segments, `sum(L_p) = 54,218,752` bytes and
  `sum(peak egress) = 32,567,296` bytes, ratio 1.664822.
- The correction is confined to the 24 combine phases of each step. Over step 0
  the dispatch phases contribute 12,781,568 bytes to both aggregates, while the
  combine phases contribute 12,781,568 against 2,467,840.

## Derived literals, frozen

Serialization at the matched rates, from `sum(L_p) * k`:

| Cell | `sum(L_p)` bytes | 400 Gbit/s ideal ps | 200 Gbit/s ideal ps |
|---|---:|---:|---:|
| Step 0 | 25,563,136 | 511,262,720 | 1,022,525,440 |
| All 32 steps | 54,218,752 | 1,084,375,040 | 2,168,750,080 |

Analytic charge, from `sum(ceil(k * L_p / 1000) * 1000)`:

| Cell | 400 Gbit/s analytic ps | 200 Gbit/s analytic ps |
|---|---:|---:|
| Step 0 | 511,290,000 | 1,022,550,000 |
| All 32 steps | 1,084,962,000 | 2,169,586,000 |

Propagation, from `48 * 2,000,000`: 96,000,000 ps per step at both rates, and
3,072,000,000 ps over all 32 steps.

The step-0 400 Gbit/s serialization literal 511,262,720 ps is the same number
the CORE-41 record quotes from the 2026-08-12 end-to-end run. It is recorded
here as an independently recomputed cross-reference, not as an assertion that
the current code reproduces that run's step latency: the graph-artifact
decomposition has changed since, so that run's TTFT is expected **not** to
reproduce and is reported as provenance only.

## Scored behavioral relations

Every scored relation is evaluated from raw analytic and fluid observations
before any exact oracle, conservation guard, artifact digest or physical-bound
check runs. No earlier fatal oracle pins any of them: the fatal guards below
constrain the byte population and the identity of the two arms, and none of
them constrains a fluid completion time.

### CORE-F1, two-serializer agreement, 3,072 instances

One instance per phase per matched rate: 48 phases times 32 steps times 2
rates. Each instance passes when

    -n_p  <=  A_p - (F_p - PROP)  <=  999

with `A_p` read from the all-local arm and `F_p` from the all-remote arm of the
same rate, on a phase pair the fatal guards have already shown carries an
identical segment multiset.

### CORE-F2, rate scaling, 64 instances

One instance per step per arm: 32 steps times 2 arms (analytic, fluid). Halving
the link rate must double the serialization term exactly and leave propagation
untouched. Registered:

- Fluid: `sum_p (F_p^200 - PROP) == 2 * sum_p (F_p^400 - PROP)` within
  `+/- sum_p n_p` picoseconds per step, and the fluid propagation total is
  exactly 96,000,000 ps at both rates.
- Analytic: `sum_p A_p^200 == 2 * sum_p A_p^400` within `+/- 48 * 999`
  picoseconds per step, the quantization allowance.

A term that moved by 1.05 or by 4 would fail. The registered direction is a
factor of exactly 2 on the serialization term and exactly 1 on propagation.

### CORE-F3, live composition, 64 instances

One instance per step per rate. The registered closed form for a step's
latency is

    local  = C_step + sum_p A_p
    remote = C_step + sum_p (F_p)
           = C_step + sum_p (k * L_p + r_p) + 48 * PROP,   0 <= r_p <= n_p

with `C_step` the identical compute term of both arms, so

    95,952,048  <=  remote - local  <=  96,000,000 + sum_p n_p

per step, where the lower bound is `96,000,000 - 48 * 999`. This is the
relation that requires both serializers to reach a live `StepResult` on the
supported metric chain, not to stop at a component ledger. An implementation
that computed both component numbers correctly and composed them onto the wrong
artifact fails CORE-F3 while passing CORE-F1.

## Fatal-unscored guards

Each is fatal: a single violation voids the run, and none of them is ever
reported as a fraction.

1. **Byte population identity.** For every step and every phase, the all-local
   arm's NVLink segment multiset equals the all-remote arm's fabric segment
   multiset, and the per-step totals satisfy
   `nvlink_directed_bytes(local) == fabric_directed_bytes(remote)` with the
   complementary term zero in each arm.
2. **Endpoint ledger conservation.** In every phase the ledger's egress sum,
   its ingress sum and the phase's directed bytes are equal, the ledger is
   sorted and rank-unique, and it is reproduced by an independently built
   ledger inside the study rather than compared with itself.
3. **Analytic quantization identity.** `A_p == ceil(k * L_p / 1000) * 1000` in
   every phase, at both rates.
4. **Structural star identity.** Every phase has at most 7 directed segments,
   one hub endpoint, and `L_p` equal to the phase's directed bytes.
5. **Backend quiescence.** Every htsim run reports verified physical
   quiescence, and the flow count of each fabric artifact equals `n_p`.
6. **All-remote exactness under the analytic rate.** Between the two all-remote
   arms that differ only in `nvlink_bandwidth_bytes_per_second`, every step
   latency, artifact inventory entry, artifact digest, flow row and locality
   outcome is identical.
7. **Compute identity.** The per-step compute term is identical across all
   arms and both rates.
8. **Phase inventory.** 48 phases and 72 graph artifacts per step in both arms,
   48 backend runs per step in the all-remote arm and 0 in the all-local arm.

## Physical sanity, stated before any measurement is read

- Floor: no phase can complete faster than its peak endpoint bytes over the
  link rate. At 400 Gbit/s that is `20 * L_p` picoseconds, and the fabric arm
  additionally cannot beat one propagation delay, so its floor is
  `20 * L_p + 2,000,000`.
- Ceiling: the analytic arm cannot exceed its floor by a whole nanosecond per
  phase, and the fluid arm cannot exceed it by more than `n_p` picoseconds.
- Scaling: halving the rate must move the serialization term by exactly 2 and
  the propagation term by exactly 1.
- System plausibility: prefill step 0 carries 25.6 MB of peak endpoint load. On
  a 400 Gbit/s NIC that is 511 microseconds of pure serialization, which for a
  54-token prefill of a 1B-parameter 400M-active MoE is implausible as a
  deployment number and is exactly why the single-node NVLink placement is the
  realistic one. The reported TTFT and TPOT effect is expected to be large and
  in the direction of the local placement being faster.

## Registered acceptance clauses for CORE-43

1. The capture-scale traffic is run all-local and all-remote at EP width eight
   over all 48 Granite phases, and the two serializers are compared on
   identical directed traffic.
2. The analytic charge and the fluid serialization term agree inside the
   preregistered band at every phase, or the study identifies which side is
   wrong instead of widening the band.
3. The effect on a live TTFT and TPOT is reported through the supported metric
   chain.
4. The all-remote path is exact: it is unaffected by the analytic serializer's
   configuration.

## Production commands

`SIMLLM_WAVE10_RUN_ROOT` names this branch's external run root, and the runner
refuses to write outside it. `SIMLLM_MOE_E2E_ROOT` names the recorded capture
tree. `SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN` name the two native tools.

```bash
.venv/bin/python examples/endpoint_fabric_crosscheck_v1/run_study.py \
  --out "$SIMLLM_WAVE10_RUN_ROOT/endpoint_fabric_crosscheck_v1" \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

The dry run registered with this freeze is the same command with
`--check-only`, which validates every frozen literal above, imports no SimLLM
implementation, reads no input file, invokes no native executable and writes no
artifact.

## Amendment, before any run

Two defects in the section above were found while reviewing the freeze against
its own arithmetic, before the study was implemented and before any result
existed. Both are corrected here rather than after a run, and the original text
is left standing so the correction is visible.

**1. The CORE-F2 analytic allowance was derived wrongly.** The registered
allowance `48 * 999` is the bound for the distance between a quantized value
and its own ideal. CORE-F2's analytic instance instead compares two different
quantizations of the same load. Writing `x = k * L_p / 1000` at the faster
rate, the per-phase residual is

    1000 * ceil(2x) - 2 * 1000 * ceil(x)

and `ceil(2x) - 2 * ceil(x)` takes the values `-1` and `0`, so the per-phase
residual spans a full quantum and the correct bound over 48 phases is
`48 * 1000 = 48,000` picoseconds, not `47,952`. This follows from the identity
alone and needs no measurement. The corrected allowance is `48 * 1000`.

**2. The CORE-F2 analytic instances are entailed and must not be scored.** The
analytic charge is a deterministic closed form over the endpoint ledger, and
that ledger is the input characterization this freeze already fixes. A relation
whose value the freeze already pins carries no genuine risk, which the
entailment rule forbids counting as behavioral evidence. The 32 analytic
instances of CORE-F2 are therefore reclassified as fatal-unscored, and CORE-F2
keeps its 32 fluid instances as its scored population. The fluid instances are
genuine risk: nothing in the freeze pins what the max-min allocator does when
its capacity is halved.

The revised scored population is CORE-F1 with 3,072 instances, CORE-F2 with 32,
and CORE-F3 with 64, for 3,168 scored instances across three families.

The CORE-F1 and CORE-F3 bounds are unaffected. Both compare a quantized value
with its own unquantized ideal, where `999` is the correct bound.
