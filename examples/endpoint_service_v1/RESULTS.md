# Endpoint service v1 results

CORE-41 is complete. The analytic intra-node serializer charges the maximum
endpoint load over both source egress and destination ingress, from an explicit
per-endpoint byte ledger rather than from a source-side surrogate.

The primary result is **4/4 genuine-risk families and 20/20 parameterized
instances**. Every fatal guard passed. Fatal guards are not reported as a
fraction.

The correction changed only how local service is charged. Byte population,
segment order, phase boundaries, the routed renderer and the whole fabric path
are byte-identical across the baseline and corrected runs.

## Declared modeling choice

The modeled endpoint is **full duplex**: one endpoint's load is
`max(egress_bytes, ingress_bytes)`.

The named and rejected alternative is a **shared half-duplex port** charged
`egress_bytes + ingress_bytes`. It is rejected against the hardware being
imitated, not against convenience. NVIDIA's *DGX GB Rack Scale Systems User
Guide*, section 1.3 "NVLink Switch Trays", states fifth-generation NVLink
switch bandwidth as full duplex, and the *NVIDIA NVSwitch Technical Overview*
(April 2018) states that each NVSwitch port supports bandwidth in each
direction over a nonblocking crossbar. Transmit and receive lanes are
independent, so summing them would double a symmetric exchange that the
hardware serves concurrently.

Two consequences follow directly and are both measured below. A symmetric
phase, where every endpoint's egress equals its ingress, must keep its previous
answer exactly, because the maximum of two equal directions is either one of
them. A many-to-one combine phase must increase, because the home rank's
ingress is the sum of W-1 peer egresses while no single source's egress moved.

The half-duplex rule would have failed both: it doubles the symmetric case and
overcharges the star by the one direction that is idle. It is not selectable;
this is a declared choice, not a configurable policy.

These sources establish the duplex structure. They do not calibrate the flat
450 GB/s rate or the zero propagation term, which remain uncalibrated first-cut
parameters owned by TRAF-11.

## Chronology and provenance

The expectations-only commit is `3879fb01a7249bbe92fe4342ad9e163570c2da1d`. It
froze the duplex decision, the fixture sweep, the exact signed combine changes,
the exact symmetric and dispatch preservation, the live JCT relation, the
physical floor and ceiling table and the dependency-authority refreeze table,
and it registered both production commands with a `--check-only` dry run that
imported no SimLLM implementation, read no input file, invoked no native
executable and wrote no artifact. It preceded the implementation and every
result-producing run. Nothing in it was edited afterwards.

The implementation sequence was:

1. `a1f13b9c876c141b5023b843dd494510f0feed20` completed the study runner with no
   implementation of the endpoint charge, so the baseline run observed the
   pre-correction serializer.
2. `78d8c14f52e7bfa3a45adcf61b24b3e77038ae0e` replaced the source-egress
   surrogate with the per-endpoint ledger and the full-duplex maximum.
3. `43ffeb87b3d4877f9a491d55a83ddd33254b3923` refroze the two affected
   dependency-authority rows to the values predicted before the correction
   existed, and preceded the rerun that tested them.

One defect was found and fixed in the runner before the first production run,
during a development probe of the live fixture. Baseline observations reach the
corrected run through `summary.json`, where a tuple written by
`dataclasses.asdict` returns as a list. Comparing a live corrected observation
against a reloaded baseline would then have reported a difference that does not
exist, and the all-remote identity guard would have voided a sound run. Every
observation is now canonicalized once onto the value shapes a JSON reload
returns. This was corrected before any production run; no run was discarded.

| Provenance field | Observed value |
|---|---|
| Expectations-only commit | `3879fb01a7249bbe92fe4342ad9e163570c2da1d` |
| Evidence authored against | `76223875557a552deb5aa2c2c529a07f000135ba` |
| Baseline run revision | `a1f13b9c876c141b5023b843dd494510f0feed20` |
| Baseline `summary.json` SHA-256 | `dfd6c5dac284a32e4815fcdc4625ee9219b9146d8e8d7321d129819bdea13591` |
| Corrected run revision | `78d8c14f52e7bfa3a45adcf61b24b3e77038ae0e` |
| Corrected `summary.json` SHA-256 | `394a88676331cf9544e084ba995e1158442396ae5ccec0848ed83e4bcf2e182c` |
| htsim gitlink observed by both runs | `fc4400e4ca619223481536632074045cb6af2756` |
| `htsim_rnic` SHA-256 | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| Runtime | Python 3.12.12 on Linux x86-64 |

The observed htsim gitlink is recorded as provenance alongside the revision the
evidence was authored against. Neither is asserted equal to the other, and no
frozen literal is compared with the live submodule pin.

## Physical sanity before exact comparison

The floor no local flow can beat is its peak endpoint bytes over the declared
one-direction rate. The analytic model quantizes upward to whole nanoseconds,
so its strict ceiling is that floor plus 1,000 ps. Every corrected combine-star
service was checked against those bounds before its exact integer oracle was
read.

| Payload bytes | EP width | Peak endpoint bytes | Floor ps | Strict ceiling ps | Measured ps | Position |
|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 2 | 1,024 | 2,275.556 | 3,275.556 | 3,000 | inside |
| 1,024 | 4 | 3,072 | 6,826.667 | 7,826.667 | 7,000 | inside |
| 1,024 | 8 | 7,168 | 15,928.889 | 16,928.889 | 16,000 | inside |
| 2,048 | 2 | 2,048 | 4,551.111 | 5,551.111 | 5,000 | inside |
| 2,048 | 4 | 6,144 | 13,653.333 | 14,653.333 | 14,000 | inside |
| 2,048 | 8 | 14,336 | 31,857.778 | 32,857.778 | 32,000 | inside |

All six sit strictly inside their bounds. Being inside is not proof of
correctness; being outside would have been proof of a defect.

Three independent framings were used rather than three passes over the same
reasoning.

**Serialization physics.** Doubling payload must double both peak endpoint
bytes and the ideal floor exactly, and it does at every width: 1,024 to 2,048,
3,072 to 6,144, 7,168 to 14,336. The measured service doubles exactly at widths
four and eight (7,000 to 14,000 and 16,000 to 32,000). At width two it moves
from 3,000 to 5,000, a factor of 1.667 rather than 2. That deviation is
explained, not tolerated: the whole-nanosecond quantum rounds 2.276 ns up to 3
and 4.551 ns up to 5, and the quantization overhead is a larger share of a
smaller number. The unquantized floors do double exactly, 2,275.556 to
4,551.111.

**Group bytes versus endpoint load.** This is the distinction TRAF-25
established and that two earlier reviewers got wrong, so it is checked
directly. The symmetric fixture carries `W(W-1)P` total group bytes and each
star carries only `(W-1)P`, a factor of W between them. All three nevertheless
have the same peak directional endpoint load `(W-1)P`, and all three measured
identical service at every one of the six cells. Widening from four to eight
moves peak endpoint load by `7/3 = 2.333`, and the measured service moves by
`16,000/7,000 = 2.286` after quantization. It does not move by the symmetric
total-byte factor `56/12 = 4.667`. Charging total group bytes would have
produced the 4.667 and is refuted.

**Agreement with the fabric the model sits beside.** The defect never reached
the fabric path, because htsim's max-min manifold already serializes a combine
incast on the home rank's ingress link. On the 2026-08-12 Granite end-to-end
run the fluid makespan decomposed as 99,360,000 ps compute plus 96,000,000 ps
propagation plus 511,262,720 ps serialization, and 511,262,720 ps is exactly
25,563,136 bytes at 20 ps/byte, the full rank-0 endpoint total rather than any
per-source share. The corrected analytic local charge now applies the same
accounting principle as the fabric model it composes with, so the two
serializers no longer disagree about what an endpoint absorbs.

## Fixture sweep and scored relations

The sweep is payload `P` in {1,024, 2,048} bytes crossed with EP width `W` in
{2, 4, 8} at 450,000,000,000 bytes/s, over three all-local fixtures: symmetric
(every ordered pair carries `P`), dispatch star (rank 0 sends `P` to each of the
other `W-1` ranks) and combine star (each of the other `W-1` ranks sends `P` to
rank 0). Two parameters vary, as the validation discipline requires.

All four families were evaluated from raw baseline and corrected observations
before any exact ledger, conservation, physical-bound, artifact or dependency
guard ran. No earlier fatal oracle pins a scored instance.

### CORE-B1, symmetric preservation: 6/6

Corrected minus baseline is exactly zero in all six cells.

| Payload bytes | EP width | Baseline ps | Corrected ps | Signed change ps | Registered | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 2 | 3,000 | 3,000 | 0 | 0 | pass |
| 1,024 | 4 | 7,000 | 7,000 | 0 | 0 | pass |
| 1,024 | 8 | 16,000 | 16,000 | 0 | 0 | pass |
| 2,048 | 2 | 5,000 | 5,000 | 0 | 0 | pass |
| 2,048 | 4 | 14,000 | 14,000 | 0 | 0 | pass |
| 2,048 | 8 | 32,000 | 32,000 | 0 | 0 | pass |

These are genuine-risk instances and not a formality. A half-duplex
implementation, or any implementation that sums the two endpoint directions,
fails all six even when its byte ledger conserves exactly. The compatibility
property is checked here rather than assumed.

### CORE-B2, dispatch preservation: 6/6

Corrected minus baseline is exactly zero in all six cells, with the same
service values as the symmetric table. This protects the already correct
one-to-many path independently of the symmetric fixture: in a dispatch star the
critical endpoint's load is its egress, so an implementation that charged
ingress only, or that charged the largest ingress rather than the largest
endpoint maximum, would report 3,000 ps and 5,000 ps here and fail.

### CORE-B3, combine response: 4/4

The registered signed changes at widths four and eight were derived from the
frozen arithmetic before anything ran, and all four matched exactly.

| Payload bytes | EP width | Baseline ps | Corrected ps | Signed change ps | Registered ps | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 4 | 3,000 | 7,000 | +4,000 | +4,000 | pass |
| 1,024 | 8 | 3,000 | 16,000 | +13,000 | +13,000 | pass |
| 2,048 | 4 | 5,000 | 14,000 | +9,000 | +9,000 | pass |
| 2,048 | 8 | 5,000 | 32,000 | +27,000 | +27,000 | pass |

The baseline column is the defect in raw form: the old charge is 3,000 ps at
every width because it saw only one peer's egress and never the home rank's
fan-in. A source-only implementation, an aggregate-group serializer and a
half-duplex implementation each produce a different response and each fail.

Width two is the degenerate transpose where one source and one destination
carry equal load. Its zero change is a fatal-unscored compatibility guard and
is deliberately excluded from this family rather than counted as a positive
response.

### CORE-B4, live JCT: 4/4

The same signed change had to reach a live `StepResult` through the supported
metric chain, not stop at a component number. The live fixture is one uniform
MoE layer with one engine rank, its dispatch star and combine transpose, fixed
compute and an all-local placement, executed through `StepRecord`, the serial
graph lowerer, checked graph artifacts, `HtsimStepSink` and `StepResult`.

| Payload bytes | EP width | Baseline JCT ps | Corrected JCT ps | Signed change ps | Registered ps | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1,024 | 4 | 20,000 | 24,000 | +4,000 | +4,000 | pass |
| 1,024 | 8 | 29,000 | 42,000 | +13,000 | +13,000 | pass |
| 2,048 | 4 | 29,000 | 38,000 | +9,000 | +9,000 | pass |
| 2,048 | 8 | 47,000 | 74,000 | +27,000 | +27,000 | pass |

CORE-B4 is decision-relevant independently of CORE-B3. An implementation could
compute a correct component ledger and still fail to place its service on the
executed graph artifact, in which case CORE-B3 passes and CORE-B4 fails. The
JCT decomposes exactly as 10,000 ps compute plus the dispatch phase plus the
combine phase, and only the combine term moved.

## Fatal-unscored guards

All passed. A single violation would have voided the run.

| Guard family | Rows | Outcome |
|---|---:|---|
| Endpoint ledger, byte conservation, exact service, ledger and legacy surfaces | 18 | all passed |
| Physical floor and strict quantization ceiling | 6 | all passed |
| Live metric exactness, replay equality, `StepResult` reachability | 6 | all passed |
| All-remote explicit versus omitted identity, cross-mode identity, quiescence | 6 | all passed |
| Renderer identity: segments, group bytes, independent ledger | 18 | all passed |
| Native tool identity across modes | 1 | passed |

Each phase's ledger was rebuilt independently inside the study from the
classified local segments and required to reproduce the production ledger
exactly, so the production ledger is checked against a second construction
rather than against itself. Ledger egress sum, ledger ingress sum and local
directed bytes were required equal in every cell. Endpoint coverage, sort
order, byte nonnegativity and the local-versus-fabric partition were exact.

All-remote placements were preserved exactly, as required, in both the explicit
distinct-host form and the omitted-manifest compatibility form. Their rendered
GOAL artifacts, binary artifacts, completion CSVs, flow rows, timestamps,
`StepResult`, TTFT and TPOT are byte-identical between the baseline and
corrected runs. They never enter the analytic local path, and the measurement
confirms it rather than assuming it.

Width-two identity and the fixed-replay metric equalities
(`TTFT == TPOT == JCT`) are fatal-unscored: three equal controlled steps make
them algebraically entailed, so they establish reachability without adding
behavioral evidence. By-construction guards never entered the behavioral
denominator, and counts from different evidence classes are not summed.

## Dependency-authority consequence

`examples/dependency_authority_v1` recorded its two all-local `AAAA` values as
baseline observations rather than precision oracles precisely because this fix
was pending. Exactly those two rows moved and nothing else did.

| Vector bytes | Service old ps | Service new ps | Signed change ps | JCT, TTFT, TPOT old ps | JCT, TTFT, TPOT new ps |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 4,538,000 | 6,652,000 | +2,114,000 | 4,562,000 | 6,676,000 |
| 2,048 | 9,047,000 | 13,286,000 | +4,239,000 | 9,071,000 | 13,310,000 |

Both new values were predicted twice before the correction existed: by the
TRAF-27 refreeze expectations and again by this study's freeze. The refreeze
commit `43ffeb87b3d4877f9a491d55a83ddd33254b3923` recorded them and preceded
the run that tested them. The rerun observed exactly those values, matched
every unaffected row, and left the study at 2/2 families and 3/3 instances with
all fatal guards passing. The rerun `summary.json` has SHA-256
`95286d67fa033bc66e2e054b4aab9c53976a2bf90ada7e7e31501dbe2586eee4` and observed
revision `43ffeb87b3d4877f9a491d55a83ddd33254b3923`.

The ratio at that fixture's EP width four is `6,652,000 / 4,538,000 = 1.466`,
independently reproducing the maintainer's recomputed width-four undercharge of
about 1.47 from a different fixture and a different code path.

The `AABB` cells are unaffected for a structural reason, not an empirical one:
each has one local pair in each direction per phase, so every local endpoint's
egress equals its ingress and the maximum is unchanged. The `ABCD` cells have
no local service at all. Both are measured unchanged.

The represented compute term stayed at 24,000 ps in both cells, and the
frozen-registry check now enforces that the refrozen service exceeds the
surrogate it replaced and that the compute term does not drift.

## Seqgen consequence

The corrected full-duplex rule is the safe first-principles floor that the
historical dispatch-sequence run should have assumed. Its synthetic home
endpoint carries 16,384 bytes in each direction, so `max(egress, ingress)` gives
655,360 ps at 200 Gbit/s and 327,680 ps at 400 Gbit/s. The historical freeze
summed both directions and used 1,310,720 ps and 655,360 ps, exactly twice the
corrected floors, and its fluid observations exceed the corrected values.

This makes a future refreeze recoverable. It does not retroactively unvoid or
score the historical run, and it does not alter that run's chronology. TRAF-22
retains requalification and still owes a new expectations-only commit,
dependency-aware bounds and the missing 200 Gbit/s scaling cell.

## Contradiction sweep

Performed after closure over `README.md`, `docs/README_PRO.md` and
`docs/architecture.md`. Hits are reported here rather than edited there.

No statement in those three files contradicts this change. The nearest hits all
describe a different mechanism: `README.md` line 280, `docs/README_PRO.md`
lines 363, 377, 433, 466 and 503, and `docs/architecture.md` lines 561 and 563
describe the compute-side per-GPU NVLink egress cursor and NCCL ring kernel in
`simllm/compute`, which is a separate model owned by COMP-11 and COMP-15.
`README.md` line 280 explicitly lists ingress as planned there. That model still
has no ingress term, and CORE-41 did not give it one; it remains COMP-11's
scope, and this is recorded as an adjacent finding rather than a contradiction.

The module docs that describe the mechanism this task changed were updated in
the same change, since they would otherwise be factually wrong:
`docs/modules/traffic.md` interface and duration-model text, and
`docs/modules/core.md`.

Two published records keep their historical text and are not rewritten:
`examples/nvlink_locality_v1/RESULTS.md` and the pre-refreeze narrative in
`examples/dependency_authority_v1/RESULTS.md` both describe the superseded
per-source form as it stood when they were written.

One process observation, reported rather than acted on: the study ledger table
in `docs/README_PRO.md` has not carried a new row since before
`nvlink_locality_v1`, `token_ownership_v1` and `dependency_authority_v1`, so
this study follows the same pattern and adds none. The only `README_PRO.md` edit
here is the mandatory task-progress block regeneration.

## Registered acceptance clauses

| Clause | Evidence | Status |
|---|---|---|
| 1. Service derived from an explicit per-endpoint egress and ingress ledger, charging the full-duplex maximum, with the half-duplex alternative named and rejected against hardware evidence | `simllm/traffic/locality.py` builds and self-verifies the ledger in `__post_init__`; duplex choice and rejected alternative recorded above with two NVIDIA sources | met |
| 2. Payload and EP-width sweeps over three fixtures conserve bytes, preserve symmetric and dispatch service exactly, and produce the frozen positive combine changes | CORE-B1 6/6, CORE-B2 6/6, CORE-B3 4/4, conservation and exactness 18/18 | met |
| 3. The corrected charge changes a supported live `StepResult` JCT by the exact signed amount while fixed-replay TTFT and TPOT stay reachable | CORE-B4 4/4, live metric exactness 6/6 | met |
| 4. Routed byte output, phase order, width-two behavior and all-remote artifacts, timestamps and metrics remain exact | renderer identity 18/18, all-remote identity 6/6, width-two zero change in CORE-B1, CORE-B2 and the live rows | met |
| 5. The two dependency-authority `AAAA` rows are refrozen with old and new values, every unaffected row remains exact, and the seqgen consequence is assigned to TRAF-22 without retroactively changing its void chronology | refreeze commit plus rerun above; seqgen floors recorded and left with TRAF-22 | met |

All five clauses are met, so CORE-41 closes.

Residual work that this run did not demonstrate moves to new IDs rather than
keeping a completed task open:

- **CORE-42** requalifies `examples/nvlink_locality_v1`. Its two all-local
  `AAAA` cells are still frozen at the superseded 4,538,000 ps and 9,047,000 ps,
  and unlike the dependency-authority rows they are **scored** TRAF-B2
  instances. Editing a scored oracle needs its own expectations-only commit, so
  it is not folded into this change.
- **CORE-43** cross-validates the analytic endpoint charge against the fabric
  backend's realized per-endpoint serialization on identical traffic at EP width
  eight on the Granite capture, where the maintainer's recomputed undercharge is
  1.676 times. This study demonstrated width four on a real fixture and widths
  two, four and eight on synthetic fixtures; the width-eight capture-scale
  figure and the two-serializer agreement remain undemonstrated here.

## Reproduction

Bulk outputs are external to the repository. The two study runs occupy 1.1 MB
each and the dependency-authority rerun 9.5 MB, so nothing needed bounding on
size. `SIMLLM_WAVE6_RUN_ROOT` names the external run root, and the runner
refuses to write outside it.

```bash
.venv/bin/python examples/endpoint_service_v1/run_study.py \
  --mode baseline \
  --out "$SIMLLM_WAVE6_RUN_ROOT/endpoint_service_v1-baseline" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"

.venv/bin/python examples/endpoint_service_v1/run_study.py \
  --mode corrected \
  --baseline-summary \
    "$SIMLLM_WAVE6_RUN_ROOT/endpoint_service_v1-baseline/summary.json" \
  --out "$SIMLLM_WAVE6_RUN_ROOT/endpoint_service_v1-corrected" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

The baseline command must run at a revision that predates the correction; it
observes the superseded serializer by construction. Both commands require a
clean worktree so the recorded SimLLM revision identifies the executed source.
