# nccl_registration_v1: frozen expectations

This document is the pre-run freeze for the NCCL/RCCL channel-and-buffer
registration study. It is committed before the registration mechanism exists,
before the live wiring exists, and before any cell has been executed. It
contains no measured value and no result.

The subject is the interim collective-completion contract recorded as a
standing decision on 2026-08-18: collective work is the one exception to the
kernel-time determinism ruling. Until the packetized path over the GPU's
NVLink, xGMI or UALink ports lands, collectives complete through the
deterministic ATLAHS/htsim chain with a no-tail constant completion, gated on
the destination memory being registered, and the NCCL/RCCL buffer-channel
registration carries an explicit modeled time cost.

This study asks one question about that cost: does it behave as a one-time
per-identity charge on the live metric chain, and does its absence leave the
accepted baseline untouched.

## What is being built, in one paragraph

A `(communicator, generation, channel, buffer)` identity is registered at most
once. The first collective that needs an unregistered identity pays a declared
registration cost, serialized ahead of that collective's own completion. Every
later collective on a registered identity pays nothing. Three events force a
re-registration: a new buffer, a new peer set, and a communicator rebuild. The
cost is declared configuration with provenance, not a measurement; a request
for a calibrated value that does not exist fails closed. The mechanism is off
by default, and the off path is byte-identical to the accepted baseline.

## Declared configuration, frozen before the run

| Name | Value | Status |
|---|---|---|
| declared registration cost | 20,000,000 ps (20 us) per identity | declared, never measured |
| evidence class | `declared` | fail closed on a calibrated request |
| default state | disabled | opt-in per study |
| identity | `(communicator, generation, channel, buffer)` | see the rules below |

The 20 us constant is a declared configuration value. Nothing in this
repository measures it. Its provenance record states the structural facts it
does rest on, which come from the net plugin ABI recorded in
`docs/papers/amd-gpu-fabric.md`: `regMr` and `regMrDmaBuf` are members of the
documented `ncclNet_v6` struct, NCCL calls them so an RDMA NIC can prepare a
buffer, and RCCL exposes the same ABI, so one seam serves both stacks. The
existence of the registration, its one-time nature and its per-buffer scope
are taken from that ABI. The duration is not. Any consumer that asks this cost
for a calibrated value must be refused rather than served the declared number
under a calibrated label.

## Identity and re-registration rules, frozen

The identity is derived from the semantic collective, not from the step:

- **communicator**: the collective kind together with its participant rank
  tuple. Two different rank sets are two different communicators, which is how
  a new peer forces re-registration.
- **generation**: an integer that a communicator rebuild increments. Every
  channel and buffer of that communicator re-registers on next use.
- **channel**: `0 .. channels_per_communicator - 1`. A communicator with two
  channels registers each channel separately, so its first collective pays
  twice.
- **buffer**: the collective's semantic site, which is the operation identity
  with its `step-<n>:` prefix removed. This is stable across steps, so a
  deployment that runs the same layer's all-reduce every step registers that
  buffer once and never again.

A collective that spans several executed artifacts (an operation split into
locality phases) pays its registration exactly once, on its first artifact.

## Cells

Two live cells, both driven through the repository's own metric chain:
`StepRecord` into `HtsimStepSink`, an authoritative `ExecutionGraph`, the
locality projection, `StepResult`, `attribute_step_detail`, then
`HtsimRequestMetricReducer` into per-request TTFT and TPOT.

### Cell `local-tp2`

A 7B-class dense geometry with tensor parallelism 2, both ranks on one host,
so no fabric artifact is produced and the whole cell is analytic.

| Parameter | Value |
|---|---|
| `num_layers` | 32 |
| `hidden_size` | 4096 |
| `intermediate_size` | 11008 |
| `num_heads` / `num_kv_heads` / `head_size` | 32 / 32 / 128 |
| `vocab_size` | 32000 |
| `dtype_bytes` | 2 |
| `tp_ranks` | (0, 1) |
| hosts | `a`, `a` |
| profile | `rnic-nn-fluid` |
| provider | `RooflineProvider(efficiency=0.7)` on `b100` |
| steps | step 0 prefill of 512 prompt tokens, steps 1 and 2 decode |
| collectives per step | 32 layers x 2 sites = 64 |
| registered identities, one channel | 64 |

Arms: `off` (registration absent), `on` (one channel), `on-2ch` (two
channels), `on-rebuild` (one channel, with an explicit communicator rebuild
issued after step 1).

### Cell `mixed-tp4`

A small geometry with tensor parallelism 4 across two hosts, so every
collective carries both an NVLink phase and a fabric phase and the fabric
phases run on the real `htsim_rnic` binary.

| Parameter | Value |
|---|---|
| `num_layers` | 2 |
| `hidden_size` | 1024 |
| `intermediate_size` | 4096 |
| `num_heads` / `num_kv_heads` / `head_size` | 16 / 16 / 64 |
| `vocab_size` | 32000 |
| `dtype_bytes` | 2 |
| `tp_ranks` | (0, 1, 2, 3) |
| hosts | `a`, `a`, `b`, `b` |
| profile | `rnic-nn-fluid` |
| steps | step 0 prefill of 64 prompt tokens, step 1 decode |
| collectives per step | 2 layers x 2 sites = 4 |
| registered identities, one channel | 4 |

Arms: `off` and `on`.

## Fatal guards, unscored

A violated fatal guard voids the run. It is never reported as a fraction and
never enters a behavioral denominator. The run is then reported as void with
findings, the evidence is kept, and every task stays open. No guard below is
declared survivable.

- **G1 default equals disabled.** The arm built with the registration
  parameter absent and the arm built with it explicitly disabled produce
  identical step latencies, identical locality outcomes, identical medium
  partitions and identical per-request totals, field by field.
- **G2 off-path artifact identity.** Every GOAL artifact SHA-256 the `on` arm
  writes equals the one the `off` arm writes, for both cells. Registration is
  a time cost and must not change a single emitted byte.
- **G3 off-path timing identity.** In both cells the `off` arm's step
  latencies, makespans, fabric services, local services, base latencies and
  attribution partitions are identical to the `off` arm computed with the
  feature entirely absent.
- **G4 conservation.** Every step's medium partition conserves its makespan
  exactly, and every request's TTFT and decode partitions conserve their
  spans exactly, in every arm.
- **G5 fail-closed calibration.** Asking a declared cost for a calibrated
  value raises. A cost cannot be constructed as `calibrated` without a
  measurement locator. An unknown named cost selector raises.
- **G6 one charge per collective.** In every arm, no collective operation is
  charged registration on more than one executed artifact, and the sum of
  per-artifact registration charges equals the ledger's own charged total for
  that step.
- **G7 mirrored-seam off path.** The `simllm.compute.nccl_stack` event stream
  of the accepted `nccl_stack_v1` sequences is unchanged when no registration
  is requested: same event count, same sequence numbers, same function names,
  same JSON.
- **G8 manifest lock.** The identity block that `results.json` publishes
  (declared cost, evidence class, both cell geometries, arm list, identity
  counts) equals this freeze exactly.
- **G9 ledger construction facts.** These hold by construction of a
  set-membership ledger and are therefore guards, not scores: a first charge
  on an unregistered identity returns the declared cost; a second charge on
  the same identity returns zero; a new buffer key, a new peer set and a
  bumped generation each miss the ledger and charge again; a disabled ledger
  charges zero for every identity.

## Scored relations

Each relation states the entailment question first: could this relation fail
while the code still runs and the guards still hold. Only relations whose
answer is yes are scored.

### Exact-oracle rows

An exact-oracle row is a single number with a closed form known before the
run. Denominator: 6.

- **O1 (`local-tp2`)** `TTFT_on - TTFT_off == 1,280,000,000 ps`, which is
  64 identities x 20,000,000 ps. Entailment: no. The charge could be dropped
  by the composition, absorbed by the `max(local, fabric)` term, charged per
  artifact instead of per collective, or charged again on the decode steps.
  Scored.
- **O2 (`local-tp2`)** `makespan_on(step 1) - makespan_off(step 1) == 0` and
  the same for step 2 in the `on` arm. Entailment: no. A ledger rebuilt per
  step or per plan would re-charge. Scored.
- **O3 (`local-tp2`)** `TTFT_on-2ch - TTFT_off == 2,560,000,000 ps`, exactly
  twice O1. Entailment: no. The channel loop could register once per
  communicator instead of once per channel. Scored.
- **O4 (`local-tp2`)** In the `on-rebuild` arm,
  `makespan(step 2) - makespan_off(step 2) == 1,280,000,000 ps`, and that
  arm's TTFT equals the plain `on` arm's TTFT exactly. Entailment: no. A
  rebuild that failed to invalidate, or that invalidated too early, breaks
  one of the two halves. Scored.
- **O5 (`mixed-tp4`)** `TTFT_on - TTFT_off == 80,000,000 ps`, which is 4
  identities x 20,000,000 ps, even though every collective is split into
  several executed artifacts and a real backend process decides the fabric
  term. Entailment: no. This is the phase-split double-charge risk in its
  live form. Scored.
- **O6 (both cells)** The registration component the medium partition
  publishes for the first step equals the ledger's charged total for that
  step, and equals zero for every later step of the `on` arm. Entailment: no.
  The partition could conserve the makespan while attributing the charge to
  the wrong component. Scored.

### Behavioral relation families

Denominator: 3 families over 7 instances. Reported separately from the
exact-oracle rows and never summed with them.

- **B1 linearity in identities (3 instances).** Charged registration is
  exactly `identities x 20,000,000 ps` at identity counts 4, 64 and 128,
  where the counts come from the two cells and the two-channel arm.
  Entailment: no. A per-communicator or per-step charge is linear in neither.
- **B2 registration is additive, not maximal (2 instances).** In both cells
  the first step's makespan under `on` equals its makespan under `off` plus
  the charged total, with no term absorbed. Entailment: no. Composing the
  charge inside the existing `max(local, fabric)` would hide it whenever the
  fabric term is larger, which is exactly the `mixed-tp4` situation.
- **B3 later steps are untouched (2 instances).** In both cells every step
  after the registering step has an identical makespan under `on` and `off`,
  and identical GOAL artifacts. Entailment: no.

## Physical sanity bounds, stated before reading any number

- **Floor.** A registration charge cannot be negative and cannot be smaller
  than zero identities times the cost. The smallest nonzero charge in this
  study is one identity, 20,000,000 ps.
- **Ceiling.** The charge on any step cannot exceed the number of distinct
  collectives in that step times the channel count times 20,000,000 ps:
  1,280,000,000 ps for `local-tp2` at one channel, 2,560,000,000 ps at two,
  and 80,000,000 ps for `mixed-tp4`.
- **Scaling check.** Doubling the channel count must double the charge and
  must leave every other term of the step exactly where it was. If the total
  moves by a factor that is not 2 while the collective payloads are
  unchanged, the relation is not the one this model claims.
- **Plausibility against the real system.** In `local-tp2` the charge is a
  fixed 1.28 ms added to a prefill whose collective and compute terms are
  themselves in the millisecond range, so registration is a visible minority
  of TTFT rather than the whole of it. In `mixed-tp4` the geometry is a toy
  and the 80 us charge is expected to dominate a step whose real work is
  microseconds; that is a statement about the size of the cell, not evidence
  about a production deployment, and the results report must say so. Real
  NCCL registers user buffers around communicator setup and the whole setup
  costs tens of milliseconds, so a per-buffer cost of tens of microseconds is
  the right order of magnitude and is still not a measurement.

## What this study does not establish

- It does not calibrate the registration cost. The number is declared.
- It does not demonstrate a packetized registration handshake. The charge is
  a serialized constant, not port traffic.
- It does not demonstrate a live new-peer re-registration. The composed
  replay has a fixed rank set, so new-peer re-registration is covered only by
  the ledger guard G9.
- It does not model deregistration, registration cache eviction, or the
  cost's dependence on buffer size.
- It says nothing about whether the deterministic no-tail constant completion
  is the right model for collectives. It implements the interim contract the
  maintainer ruled; the packetized destiny is registered as open work.

## Chronology

This file and `expectations.json` are committed first, with no implementation
of the registration mechanism, no live wiring, no runner and no result. The
results report must cite the commit of this freeze and must state the actual
order of commits.
