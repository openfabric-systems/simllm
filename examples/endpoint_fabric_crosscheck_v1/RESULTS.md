# Endpoint versus fabric serializer cross-check results

CORE-43 is complete. The analytic intra-node endpoint charge and the
`rnic-nn-fluid` manifold were handed the same directed traffic, the real
Granite MoE capture at EP width eight over all 48 phases of all 32 recorded
steps, at matched rates. They agree.

The primary result is **2/2 genuine-risk families and 3,104/3,104 instances**.
Every fatal guard passed. Fatal guards are not reported as a fraction.

The headline number is the one CORE-43 existed to produce. On the prefill step
at 400 Gbit/s the analytic serializer charges 511,290,000 ps and the fluid
manifold realizes 511,262,768 ps of serialization on the same bytes. The two
independently written implementations differ by 27,232 ps out of 511 million,
5.3 parts in one hundred thousand, and every picosecond of that difference is
accounted for: the analytic model quantizes each phase up to a whole GOAL
nanosecond, and the fluid manifold rounds each allocation epoch up to a whole
picosecond.

## Chronology and provenance

The expectations-only commit is `9f2cb0999c3ddbb79561f7c4ce760e73072b1804`. It
froze the sweep, the matched rate pairs, the closed-form agreement band, the
input characterization and every derived literal, and it registered the
production command with a `--check-only` dry run that imported no SimLLM
implementation, read no input file, invoked no native executable and wrote no
artifact. It preceded the implementation and every result-producing run.

Two things about that freeze are disclosed rather than smoothed over.

**An arithmetic defect in the freeze, corrected before the run.** Commit
`dd87ecebbf125b9c0473af6c99dbab1f84a18ce1` corrected the CORE-F2 analytic
allowance from `48 * 999` to `48 * 1000` and reclassified the 32 analytic
scaling instances as fatal-unscored. Both corrections follow from arithmetic
alone. The allowance was derived as the distance between a quantized value and
its own ideal, but that instance compares two different quantizations of the
same load, whose per-phase residual `ceil(2x) - 2 * ceil(x)` spans a full
quantum. And the analytic charge is a closed form over the endpoint ledger that
the freeze already fixes as an input, so it is entailed and may not be scored.
Both landed before the implementation and before any run. The corrected
allowance was needed: the observed analytic residual reaches exactly -48,000 ps,
which the original literal would have failed by 48 ps.

**A post-run evidence-accounting correction.** Integrator review found that
CORE-F3 is entailed by CORE-F1 plus the compute-identity fatal guard. The live
step latency sums the same composed phase-service arrays that supply CORE-F1,
so summing CORE-F1's 48 per-phase bands reproduces CORE-F3's registered band
exactly. Its 64 rows remain fatal-unscored live-composition evidence and no
longer increase the behavioral score.

**A textual fix for CI.** Commit `eb9dd4ff1d3c3173c731ac55993dd24b5a5a615b`
spelled a two-sided tolerance sign in words because the tracked-markdown
portability gate reads it as an absolute path. No registered value changed.

The implementation landed at `614ceb359ba063b9e7b4b6e0c543da2d672d1b5a`. Two
executions were performed. The first, retained at run root `crosscheck-run1`,
executed the source at `eb9dd4f` and passed every relation and every guard, but
the CORE-47 closure was committed while it ran, so its recorded `run_head`
names a revision later than the one it executed. It is therefore not the
accepted evidence. The accepted run is `crosscheck-run2`, which started and
finished on a clean worktree at the recorded revision.

The two runs produced **byte-identical** `summary.json` files, all 21,543,766
bytes of them. That is a stronger reproducibility statement than the digest
alone: two executions at two different source revisions, one of which changed
an unrelated study and the module docs, produced the same 3,072 fluid
measurements to the picosecond.

| Provenance field | Observed value |
|---|---|
| Expectations-only commit | `9f2cb0999c3ddbb79561f7c4ce760e73072b1804` |
| Pre-run amendment | `dd87ecebbf125b9c0473af6c99dbab1f84a18ce1` |
| Implementation commit | `614ceb359ba063b9e7b4b6e0c543da2d672d1b5a` |
| Accepted run revision | `ed4de3c2380dc732f7d8f7abab8515c055219920` |
| Accepted `summary.json` bytes | 21,543,766 |
| Accepted `summary.json` SHA-256 | `b33e7929a4301887ef894cbc2f0d224df2f09b7333e3e775a351a0a9c7db7713` |
| htsim gitlink observed by the run | `fc4400e4ca619223481536632074045cb6af2756` |
| `htsim_rnic` SHA-256 | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| `txt2bin` SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| Recorded steps SHA-256 | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |
| Captured routing SHA-256 | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |

The observed htsim gitlink is recorded as provenance beside the revision the
evidence was authored against. Neither is asserted equal to the other, and no
frozen literal is compared with the live submodule pin.

## Physical sanity before the exact comparison

Every bound below was stated in the freeze before any value was read.

**Floor.** No phase can complete faster than its peak endpoint bytes over the
link rate: `20 * L_p` picoseconds at 400 Gbit/s, `40 * L_p` at 200 Gbit/s. The
fabric arm additionally cannot beat one propagation delay. All 3,072 fluid
instances sit at or above their floor, and 402 of them sit exactly on it.

**Ceiling.** The analytic charge cannot exceed its floor by a whole nanosecond,
and the fluid manifold cannot exceed it by more than `n_p` picoseconds, at most
7 here. The observed fluid excess is **0 or 1 picosecond on every one of the
3,072 instances**, never more, at both rates: 1,134 phases at +1 and 402 at +0
in each rate. The registered ceiling was a factor of seven loose and the
mechanism is tighter than the bound.

**Scaling.** Halving the rate must move the serialization term by exactly 2 and
propagation by exactly 1. Fluid serialization over all 32 steps moves from
1,084,376,174 ps to 2,168,751,214 ps, a ratio of 1.99999895, and the residual is
the +1 ps epoch rounding at each rate, not a modeling difference. Propagation is
96,000,000 ps per step at both rates, identically.

**Deployment plausibility.** Prefill step 0 carries 25.6 MB of peak endpoint
load. At 400 Gbit/s that is 511 microseconds of pure serialization for a
54-token prefill of a 1B-parameter, 400M-active MoE, a substantial placement
penalty that favors the single-node NVLink placement. The measured live TTFT
confirms the direction: 706.6 microseconds across the fabric against 156.2
microseconds on NVLink. Decode TPOT on NVLink is 101.9 microseconds, of which
99.36 microseconds is the roofline compute term, so the decode step is
weight-read bound and the network is 2.5 percent of it. At the corrected
microsecond scale both TTFTs are physically plausible; the deployment argument
is the 4.52 times fabric penalty, not an implausible absolute latency.

Three independent framings were used rather than three passes over the same
reasoning: serialization physics against the byte ledger, the fluid allocator's
own rounding rules read out of the C++ source, and end-to-end plausibility
against what a weight-read-bound decode step should cost.

## Sweep and matched rates

The traffic is fixed: all 32 recorded scheduler steps, 48 phases each, replayed
from the recorded step records and the captured routing projection. The only
difference between the two placements is the rank-to-host assignment. Two
parameters vary, placement and link rate, and two further arms serve narrower
purposes.

| Arm | Placement | Fluid bits/s | Analytic bytes/s | Backend runs per step |
|---|---|---:|---:|---:|
| `local-400` | one host | not used | 50,000,000,000 | 0 |
| `remote-400` | eight hosts | 400,000,000,000 | not used | 48 |
| `local-200` | one host | not used | 25,000,000,000 | 0 |
| `remote-200` | eight hosts | 200,000,000,000 | not used | 48 |
| `local-nvlink` | one host | not used | 450,000,000,000 | 0 |
| `remote-400-control` | eight hosts | 400,000,000,000 | 450,000,000,000, unread | 48 |

At the matched pairs both serializers are charged exactly 20 and exactly 40
picoseconds per byte, so no rounding enters the comparison from the rate itself.

## Input characterization, frozen and not scored

The directed traffic was computed before the freeze with pure Python, no
backend and no timing, and the freeze records it. It is an input, so under the
entailment rule it cannot also be evidence, and none of it is counted below.

Every phase is a star hubbed on the engine rank 0, with at most 7 directed
segments, so the peak endpoint load equals the phase's directed bytes. Prefill
step 0 carries 336 segments and `sum(L_p) = 25,563,136` bytes against the
superseded source-egress-only aggregate of 15,249,408 bytes, the ratio 1.676336
that CORE-43 named. Over all 32 steps the totals are 9,108 segments,
54,218,752 bytes and 32,567,296 bytes, ratio 1.664822. The correction is
confined to the 24 combine phases of each step: over step 0 the dispatch phases
contribute 12,781,568 bytes to both aggregates while the combine phases
contribute 12,781,568 against 2,467,840, an undercharge of 5.18 times on the
half of the traffic where the home rank's fan-in is the bottleneck.

Two arithmetic consequences follow and are reported rather than scored, because
the ledger above already determines them. At 400 Gbit/s the corrected analytic
charge for the prefill step is 511,290,000 ps against 305,012,000 ps under the
superseded surrogate, a factor of 1.676295. Carried onto the live step latency
with its identical 99,336,000 ps compute term, that is 610,626,000 ps against
404,348,000 ps, so the CORE-41 correction moves this capture's live TTFT by a
factor of **1.510**.

## Scored behavioral relations

Both families were evaluated from raw analytic and fluid observations
before any exact ledger, conservation, physical-bound, artifact or identity
guard ran. No earlier fatal oracle pins a scored instance: the fatal guards
constrain the byte population and the identity of the two arms, and not one of
them constrains a fluid completion time.

### CORE-F1, two-serializer agreement, 3,072/3,072

One instance per phase per matched rate. Each passes when
`-n_p <= A_p - (F_p - PROP) <= 999`.

| Rate | Instances | Fluid excess over ideal | Difference `A_p - (F_p - PROP)` | Band |
|---|---:|---|---|---|
| 400 Gbit/s | 1,536 | 0 to 1 ps | 119 to 959 ps | -7 to 999 ps |
| 200 Gbit/s | 1,536 | 0 to 1 ps | 39 to 959 ps | -7 to 999 ps |

Aggregated to the cells the freeze named:

| Cell | Rate | Ideal `k * sum(L_p)` ps | Analytic ps | Fluid serialization ps | Analytic minus fluid ps |
|---|---|---:|---:|---:|---:|
| Prefill step 0 | 400 Gbit/s | 511,262,720 | 511,290,000 | 511,262,768 | 27,232 |
| Prefill step 0 | 200 Gbit/s | 1,022,525,440 | 1,022,550,000 | 1,022,525,488 | 24,512 |
| All 32 steps | 400 Gbit/s | 1,084,375,040 | 1,084,962,000 | 1,084,376,174 | 585,826 |
| All 32 steps | 200 Gbit/s | 2,168,750,080 | 2,169,586,000 | 2,168,751,214 | 834,786 |

Two things in that table matter more than the pass count.

First, the fluid manifold reproduces the ideal `k * sum(L_p)` to within 48 ps on
the prefill step and 1,134 ps over all 32 steps, and both residuals are exactly
one picosecond per phase that had a nonzero epoch remainder. The fabric model
is not merely close to bytes over rate on this traffic; it is bytes over rate
plus a countable rounding.

Second, the 511,262,720 ps figure that the CORE-41 record quotes from the
2026-08-12 end-to-end run was arithmetic, `25,563,136` bytes at 20 ps per byte.
This study measured it: 511,262,768 ps. The maintainer's recomputation was right
to within 48 picoseconds, and it is now an observation rather than a derivation.

The extremes bound the claim from both ends. The largest phase,
`layer-20:ep-dispatch` of step 0 at 595,968 bytes over 7 segments, charges
11,920,000 ps analytically against a fluid serialization of 11,919,361 ps on an
ideal of 11,919,360 ps. The smallest, `layer-2:ep-dispatch` of step 24 at 6,144
bytes over 3 segments, charges 123,000 ps against 122,881 ps on an ideal of
122,880 ps. Across a hundredfold range of phase size the disagreement stays
inside one nanosecond because it is quantization, not a model difference.

**Which side is wrong, if either.** Neither. The band was registered so that a
failure would name a side: below the physical floor is a fabric defect, above
999 with the floor intact is an analytic defect. No instance left the band in
either direction, and the entire observed disagreement is the analytic model's
declared whole-nanosecond GOAL calc quantum. The analytic serializer is the
coarser of the two by construction, never the wrong one.

### CORE-F2, rate scaling, 32/32

One instance per step, on the fluid arm. The analytic instances were
reclassified as entailed before the run and appear under the fatal guards.

Fluid serialization at 200 Gbit/s against twice its value at 400 Gbit/s, summed
over the 48 phases of each step, sits within the registered per-step allowance
of `sum(n_p)` picoseconds in every step. Step 0 is 1,022,525,488 ps against
1,022,525,536 ps, a residual of -48 ps on an allowance of 336. Over all 32 steps
the ratio is 1,084,376,174 to 2,168,751,214, or 1.99999895.

The residual has a mechanism, not a tolerance. Each rate carries the same +1 ps
epoch rounding on the same 1,134 phases, so doubling the faster arm doubles that
rounding while the slower arm carries it once. The registered direction, a
factor of exactly 2 on serialization and exactly 1 on propagation, holds: the
propagation total is 96,000,000 ps per step at both rates.

## Fatal-unscored live composition identity

### CORE-F3, 64 rows

One instance per step per rate, on live `StepResult` latencies through the
supported metric chain.

| Rate | Instances | `remote - local` ps | Registered band ps |
|---|---:|---|---|
| 400 Gbit/s | 32 | 95,971,648 to 95,991,128 | 95,952,048 to 96,000,336 |
| 200 Gbit/s | 32 | 95,962,440 to 95,991,248 | 95,952,048 to 96,000,336 |

These rows demonstrate that both serializers reach a live step latency, but
they do not add an independent behavioral relation. `step_latency_ps` is the
represented compute term plus the sum of the same composed phase-service
arrays that supply CORE-F1. With compute identity holding, `remote - local` is
identically `sum(F_p) - sum(A_p)`, and summing CORE-F1's per-phase band over 48
phases reproduces the registered CORE-F3 band exactly. The prefill step
decomposes exactly:
610,626,000 ps locally as 99,336,000 ps of compute plus 511,290,000 ps of
analytic service, and 706,598,768 ps remotely as the same compute plus
511,262,768 ps of fluid serialization plus 96,000,000 ps of propagation.

## Live TTFT and TPOT

Reported through `StepRecord`, the serial graph lowerer, the checked graph
artifacts, `HtsimStepSink` and `StepResult`, for request `r0` of the capture.

| Arm | TTFT ps | TPOT ps |
|---|---:|---:|
| `local-nvlink`, 450 GB/s | 156,168,000 | 101,907,826 |
| `local-400`, matched 50 GB/s | 610,626,000 | 121,177,304 |
| `remote-400`, 400 Gbit/s | 706,598,768 | 217,156,175 |
| `local-200`, matched 25 GB/s | 1,121,886,000 | 142,865,391 |
| `remote-200`, 200 Gbit/s | 1,217,861,488 | 238,836,481 |
| `remote-400-control` | 706,598,768 | 217,156,175 |

Three readings, in decreasing order of how much they say:

- **Placement dominates TPOT.** At the same 20 ps per byte, moving this traffic
  off the node costs 95.98 microseconds of TTFT and 95.98 microseconds of TPOT,
  dominated by the 96-microsecond total from the 48 fixed propagation delays,
  with only a nanosecond-scale serializer quantization residual. At the
  deployment NVLink rate the local arm's TPOT falls to 101.9 microseconds, of
  which the compute term is 99.36 microseconds, so the network is 2.5 percent
  of a decode step and the fabric arm's is 54 percent.
- **Rate scaling reaches the live metric.** Halving the link rate adds
  511.3 microseconds to the local TTFT and 511.3 microseconds to the remote
  TTFT, which is the serialization term doubling while the compute and
  propagation terms do not move.
- **The control arm is bit-identical to its reference.** Every TTFT and TPOT
  digit is the same, which is the live face of the all-remote exactness guard.

## Fatal-unscored guards

All passed. A single violation would have voided the run.

| Guard family | Rows | Outcome |
|---|---:|---|
| Byte population identity: per-arm phase counts, artifact counts, backend runs, and identical segment multisets between the two arms of each rate | 256 | all passed |
| Endpoint ledger conservation, rebuilt independently inside the study | 9,216 | all passed |
| Analytic quantization identity against a second closed form | 4,608 | all passed |
| Structural star identity: at most 7 segments, one hub, hub rank 0 | 9,216 | all passed |
| Backend quiescence and per-artifact flow count equal to `n_p` | 96 | all passed |
| All-remote exactness under a changed analytic rate | 3 | all passed |
| Compute identity across all six arms | 32 | all passed |
| Analytic projection identity: executed artifact service equals the independently classified phase service | 96 | all passed |
| Analytic scaling identity, entailed and reclassified before the run | 32 | all passed |
| CORE-F3 live composition identity, entailed by CORE-F1 and compute identity | 64 | all passed |

Each phase was lowered and classified a second time inside the study, so the
endpoint ledger, the peak endpoint load and the segment multiset were checked
against an independent construction rather than against the production values
that produced them. The analytic charge was recomputed from its own closed form
for the same reason.

The all-remote exactness guard is the one that answers the CORE-43 clause
directly. Between `remote-400` and `remote-400-control`, which differ only in
the analytic NVLink bandwidth, all 4,608 artifact digests, every step latency,
every network and locality outcome and every fabric phase service are
identical. The fabric path does not read the analytic serializer's
configuration, and this is measured rather than assumed.

The entailed analytic scaling rows are reported here rather than scored, per
the pre-run amendment. Their residual runs from -48,000 ps to 0 against the
corrected allowance of 48,000 ps, so the corrected literal was necessary and
the original `47,952` would have failed by 48 ps.

## Registered acceptance clauses

| Clause | Evidence | Status |
|---|---|---|
| 1. The capture-scale traffic is run all-local and all-remote at EP width eight over all 48 Granite phases, and the two serializers are compared on identical directed traffic | Six arms over all 32 recorded steps; byte-population identity 256/256 proves the segment multisets match phase by phase | met |
| 2. The analytic charge and the fluid serialization term agree inside the preregistered band at every phase, or the study identifies which side is wrong instead of widening the band | CORE-F1 3,072/3,072 inside the band; the fluid excess is 0 or 1 ps against a registered ceiling of 7; the entire disagreement is the analytic model's declared nanosecond quantum | met |
| 3. The effect on a live TTFT and TPOT is reported through the supported metric chain | CORE-F3 has 64 fatal-unscored live-composition rows and the TTFT and TPOT table above states the 1.510 times capture-scale TTFT effect of the CORE-41 correction | met |
| 4. The all-remote path is exact: it is unaffected by the analytic serializer's configuration | All-remote exactness 3/3 over 4,608 artifact digests and every timestamp | met |

All four clauses are met, so CORE-43 closes. Nothing it registered was left
undemonstrated, so **no new ID is registered**.

Two observations are recorded here as prose rather than as new IDs, because no
registered clause claimed either of them:

- The comparison is strong on star traffic and has not been exercised on a
  contended one. Every phase of this capture has a single hub, because the
  capture has one engine rank, so the fluid allocator never had to resolve a
  many-to-many bottleneck. The agreement demonstrated here is that both
  implementations charge the correct endpoint at the correct rate, not that they
  agree under max-min contention among several loaded endpoints.
- The 2026-08-12 exploratory run's TTFT of 974,838,253 ps does not reproduce and
  was not expected to. That run rendered one GOAL artifact per step; the graph
  projection now emits 72 artifacts per step, of which 48 carry a collective. Its
  serialization figure does reproduce, to 48 ps, which is the part of it this
  study was registered against.

## Contradiction sweep

Performed after closure over `README.md`, `docs/README_PRO.md` and
`docs/architecture.md`. Hits are reported here rather than edited there.

No statement in those three files contradicts this result. `README.md` line 269
and `docs/architecture.md` line 125 describe intra-node traffic riding an
NVLink-class resource and staying off the fabric, which is exactly the split
this study measured from both sides. `docs/README_PRO.md` line 296 lists the
intra-node NVLink split as a landed deterministic analytic seam and line 297
keeps its calibration open under TRAF-11; this study matched the analytic rate
to the fabric rate deliberately, to compare the two serializers, and says
nothing about the calibrated 450 GB/s value, which remains TRAF-11's.

One adjacent finding, reported and not acted on: `README.md` line 280 describes
the compute-side per-GPU NVLink egress serializer as a first cut with ingress
planned. That is a different model, owned by COMP-11, and this study did not
give it an ingress term.

## Reproduction

Bulk outputs are external to the repository. The accepted run occupies
75 MB, of which the `summary.json` is the bulk; the six arms together write
13,824 backend artifacts totalling about 41 MB. `SIMLLM_WAVE10_RUN_ROOT` names
the external run root and the runner refuses to write outside it.

```bash
.venv/bin/python examples/endpoint_fabric_crosscheck_v1/run_study.py \
  --out "$SIMLLM_WAVE10_RUN_ROOT/endpoint_fabric_crosscheck_v1" \
  --source-root "$SIMLLM_MOE_E2E_ROOT" \
  --htsim-rnic "$SIMLLM_HTSIM_RNIC" \
  --txt2bin "$SIMLLM_TXT2BIN"
```

The command requires a clean worktree so the recorded SimLLM revision
identifies the executed source.
