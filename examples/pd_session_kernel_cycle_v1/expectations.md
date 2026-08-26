# Disaggregated session kernel-cycle binding expectations

This is the expectations-only record for CORE-53. It freezes the exact
candidate lookup identity, provider binding, prompt and handoff grid, signed
movement arithmetic, roofline comparator identity and closure rule before the
binding exists or any scored run executes.

## Chronology

The source state is commit
`4f7a316926ecd55fb00d376e5aae1bcfc01c1929`. Immediately before this freeze,
the tracked worktree was clean. The required sizing note was already present
in the ignored working layer. The accepted `pd_session_v1` result and the
accepted `kernel_cycle_lut_v1` candidate record both predate this freeze. No
CORE-53 binding implementation or scored CORE-53 run existed.

The final result must name this expectations-only commit. Any change to the
movement oracle after a run is post-specified and cannot be called part of
this freeze.

## Physical mechanism

The GPU receives one scheduler step. The step names the requests that share
the launch, how many new tokens each computes and each request's context after
that work. A prefill lookup uses total new work and prior context. A decode
lookup uses the ordered prior context of every request in the batch. Prior
context is `context_length - num_new_tokens`, because those bytes exist before
the selected kernel stream begins.

The lookup record remains evidence, not a second timing authority. A provider
validates its canonical bytes and required SHA-256, forms the complete record
key, and selects exactly one row. It compiles that row through the existing
profile-table provider. A miss calls the explicit roofline comparator. The
step sink, shared virtual clock and request timeline remain the only timing
chain from provider service to time to first token (TTFT) and time per output
token (TPOT).

When no record is supplied, the session receives the existing roofline
provider object directly. It must not receive an empty lookup wrapper, a new
provenance field or another serialization branch. That is the byte-identity
off path.

## Candidate record

The accepted retained-fixture record has SHA-256
`e495f3ca5d0858cf371b19205ae6b7747d633695020d10f58645c5f245086070`.
It is candidate status, covers a partial kernel subset and carries one decode
row: batch one, prior KV length 16, CUDA graph launch, tensor parallel one on
an A100 under the captured vLLM 0.26 identity. The row reconstructs
2,047,488,000 ps exactly.

The record is not a Hopper calibration and is not a complete session table.
Its A100 device, vLLM 0.26 framework, tensor-parallel-one configuration,
missing route split, insufficient replay count and absent prefill rows must
remain visible in provenance or result scope. Candidate status is never
promoted to validated.

The binding study deliberately answers a narrower mechanism question: when a
caller explicitly supplies this accepted candidate record and its exact
record-owned selection context, does the live provider chain choose the only
matching dynamic shape, carry that candidate provenance and move the final
metrics by the exact selected-row delta? This does not assert that the
candidate number calibrates the B100 session.

## Frozen grid and selection

The request grid is unchanged from `pd_session_v1`:

| Parameter | Values |
|---|---|
| Original prompt tokens | 8, 16 |
| Declared KV handoff | 100,000,000 ps, 200,000,000 ps |
| Client-visible decode tokens | 4 |

Every prefill step forms a prefill key and misses because the candidate record
contains no prefill row. Every decode step forms a decode key. The first
client-visible decode step after a 16-token original prompt has prior context
16 and selects the candidate row. Later decode steps have prior contexts 17,
18 and 19 and miss. Every prompt-8 step misses. The exact expectation is two
lookup hits, both decode, over the four cells.

An exact key match means equality of the complete canonical entry key. A
partial tuple match is not a hit. More than one matching row rejects instead
of choosing by file order. A supplied expected record digest that differs
from the canonical record bytes rejects before any step prices.

## Signed movement oracle

The B100 roofline provider returns 75,292,525 ps for the KV-16 fused step. The
existing scalar layer renderer exposes 75,288,000 ps after whole-nanosecond
layer boundaries. The candidate duration is divisible exactly across the same
24-layer renderer and exposes 2,047,488,000 ps. The signed selected-step
movement is therefore:

```text
2,047,488,000 ps - 75,288,000 ps = +1,972,200,000 ps
```

The selected row is the first client-visible decode step. It therefore adds
exactly 1,972,200,000 ps to TTFT and exactly 0 ps to TPOT. TPOT begins at the
first client-visible completion and averages only the three later intervals.
The prompt-8 cells select no row, so both signed movements are zero.

| Prompt | Handoff (ps) | Expected TTFT (ps) | Signed TTFT (ps) | Expected TPOT (ps) | Signed TPOT (ps) |
|---:|---:|---:|---:|---:|---:|
| 8 | 100,000,000 | 273,376,000 | 0 | 77,952,000 | 0 |
| 8 | 200,000,000 | 373,376,000 | 0 | 77,952,000 | 0 |
| 16 | 100,000,000 | 2,265,112,000 | +1,972,200,000 | 77,976,000 | 0 |
| 16 | 200,000,000 | 2,365,112,000 | +1,972,200,000 | 77,976,000 | 0 |

Doubling the declared handoff must still add exactly 100,000,000 ps to TTFT
at both prompt lengths. It must change lookup selection and TPOT by zero.

## Off-path identity

All six accepted tracked files under `examples/pd_session_v1` are locked by
SHA-256 in the JSON freeze. Two independent record-absent observations must
equal the accepted four compact cells and each other. Their complete request
result serializations, including KV bytes and every timestamp, must also be
byte-identical. The new provenance member must be absent rather than null on
this arm.

This identity is a fatal guard and is not a scored behavioral point. The two
independent observations form a separate behavioral family only for
reproducibility after the fatal accepted-baseline check has held.

## Future Hopper record control

The provider receives canonical record bytes and a required content address.
It contains no digest allowlist, campaign-specific conditional or device-name
branch. A test must construct a different key-compatible record whose device
is Hopper, load it through the identical call, select its exact row and expose
the new content address and acceptance status. The test may change data only.
Changing production code, renaming the implementation or adding a special
Hopper selector fails this requirement.

This control proves interchangeability of the content-addressed binding. It
does not stand in for the campaign record or make a Hopper calibration claim.

## Physical bounds before modeled values

The candidate partial kernel stream must be positive and cannot exceed its
2,323,678,000 ps enclosing measured decode step. Its frozen 2,047,488,000 ps
lies 276,190,000 ps below that ceiling. Every live nonempty session step must
remain between 1,000,000 and 100,000,000,000 ps.

The roofline comparator still has the existing per-rank weight-read floor.
The candidate number comes from an A100 partial capture, so it is not compared
to the B100 roof as though it were B100 service. The only valid cross-arm
claim is exact mechanism movement after explicit selection. End-to-end decode
cadence must remain within 10 through 100,000 client-visible tokens per
second. These bounds can reject an impossible number; they cannot validate a
candidate calibration.

## Evidence accounting

Four per-cell TTFT decompositions are exact-oracle rows. Three scored
behavioral families contain eight instances: four signed movement rows, two
handoff-orthogonality rows and two record-absent reproducibility rows. Nine
fatal guards remain outside that denominator. Native vLLM engine construction
is separate evidence.

A fatal guard violation voids the run. It is never reported as a lost point.

## Closure rule

CORE-53 closes only if its registered entry is literal against this candidate
record, including the required prefill and decode pricing coverage. The known
record has one partial decode row and no prefill row. Passing the binding and
movement mechanism therefore does not by itself close CORE-53. If the run
confirms that limit, the exact missing record coverage is registered under an
allocated residual ID and CORE-53 stays open on it.

No result from this study may claim B100 or Hopper calibration, complete
prefill or decode coverage, a validated routing distribution or completion of
the COMP-64 campaign.
