# Collective latency floor v1 results

This study replaces the optional local 450 GB/s byte-rate surrogate with a
calibrated bandwidth, participant-latency and endpoint-concurrency form, and
adds the same separately reportable non-propagation floor to the fluid fabric
path. The historical `None` and explicit `legacy` selectors remain the exact
compatibility path.

## Outcome

Production attempt one is **void**. Its raw sensitivity cells returned the
upper endpoint of the fluid backend's published one-picosecond completion
quantization range, while the first freeze admitted only the lower endpoint.
The `sensitivity_fixture_identity` and `propagation_rate_cancellation` fatal
guards failed. The raw observations are retained, no behavioral fraction is
reported for that attempt, and it closes nothing.

Production attempt two is **not void**, but it is insufficient for closure.
Its fatal harness did not execute a mixed-placement collective, and its stored
4/4 headline incorrectly counted the mathematically entailed C3 ratio as
genuine risk. Its raw measurements remain valid, but neither task closes on
that run.

The final classification replay of production attempt three is **not void**.
Every fatal guard held, including a two-node mixed-placement collective with
simultaneous local and fabric service. All **2 of 2 non-entailed genuine-risk
families (C1, C2)** passed, for **100 percent genuine risk**. C1 contains three
held-out instances, which are reported separately from the family count. C3
and C4 also pass, but both are exact-unscored because exact guards and fixed
fixture construction entail their predicates. `step_service_conservation`,
not C4's predicate, carries the TTFT and TPOT reach claim. TRAF-11 and COMP-11
each close only on their own registered clauses: the undemonstrated
point-to-point source clause moves to TRAF-31, and the undemonstrated
detailed-mechanism clauses move to COMP-31.

## Freeze chronology and two-sided integrity

The original expectations were frozen in `2a14acb` and clarified in
`81badd4`, before implementation commit `cec7109` and before any
result-producing run. The original expectations SHA-256 was
`9bcbdc22bbd4525cbbc51782f1574d1f0097f6793c0cb075dc4be5c11a5d52a8`.
The original freeze classified four families as scored, and included the
calibration parameters, physical bounds and closure scope. Independent closure
review found that C3's exact fatal oracles entail its ratio. The B1 fix-round
then found that C4's exact guards and fixture construction likewise entail its
predicate. This report post-specifically removes both from the genuine-risk
denominator without discarding or weakening either registered relation.

Attempt one ran at `cec7109adeb9656de92f9ef5ea54572accdc3208`. Its summary
SHA-256 is
`eeb59dd67d6e7349b6b299396e5d86a4e60393101ef294dd2fb6f993430f327a`.
The failure was an evidence-harness defect: the mission report had already
published a 2,000,000 to 2,000,001 ps rate-cancelled range, but the freeze and
harness required exactly 2,000,000 ps.

Attempt two was explicitly refrozen after that observation in `3a6126e`.
This part is post-specified and is not presented as a preregistered oracle. Its
expectations SHA-256 is
`986c5952099bb6200736618a33580217b58ab3bbf9cb9ec97df1efc8b6962a28`.
The check-only command then passed without creating its requested directory.
Commit `a7bca21` made the matching harness correction, and attempt two ran at
`a7bca21de7ecfcf3abd056335f71a21acc7808ce`. Its summary SHA-256 is
`89394441c47a32a5e54be7fec4f702c10e8345ce104ea677ec2af7e1539e1a35`.

Independent closure review then found the entailed C3 classification and the
missing mixed-placement coverage. Commit `4401401` corrected both evidence
defects without changing modeled behavior. Attempt three initially ran at
`4401401c9bafd995050aa5cf83fff50c8ccf3b75`; its summary SHA-256 is
`fe55272c7a45b69874ae10b4848649d5eb80624f44ef5c5f0f300a24959cee81`.
The B1 fix-round then found that C4 was also entailed. Commit `5eb8041` moved
C4 beside C3 as exact-unscored evidence and added a classification regression.
A clean classification replay of attempt three ran at
`5eb8041db9260e32f4da82dc9143958e255a6a65`; its `summary.json` SHA-256 is
`eb34411f3db57383490833eeed12828ceb90cee5afe1ef071c60f07231c16485`.
All 1,350 non-summary files are byte-identical to the initial attempt-three
record, and all raw C1 through C4 values remain identical to attempt two. The
specific mixed cell is post-specified coverage of the frozen no-duplicate
charge clause. It is fatal-unscored evidence, not a new behavioral relation.

Every commit after the first measured run is classified here:

| Commit | Classification | Modeled behavior before and after | Measurement before and after |
|---|---|---|---|
| `3a6126e` | Refreezes a fatal harness premise after a void run | unchanged | raw C1 through C4 values unchanged; the accepted cancellation oracle changes from an invalid 2,000,000 ps literal to the executed 2,000,001 ps quantized value |
| `a7bca21` | Fixes the evidence harness to implement that refreeze | unchanged | 6,014,081 and 10,028,161 ps before and after; only the fatal comparison changes |
| `4401401` | Fixes the first evidence classification defect and adds missing mixed-placement coverage | unchanged | C1 through C4 are identical before and after; the headline changes from invalid 4/4 to still-invalid 3/3, and the new mixed cell measures one previously unexercised frozen guard |
| `702976e` | Reconciles documentation and the ledger and adds an LF portability lock | unchanged | no measurement rerun and no measured value changes |
| `d3cdbcd` | Merges current main before the fix round | unchanged for this study | the later clean replay reproduces all 1,350 non-summary files exactly |
| `5eb8041` | Fixes the remaining C4 classification defect and adds a regression lock | unchanged | the valid headline changes from 3/3 to 2/2; all raw C1 through C4 values and all 1,350 non-summary files remain identical |
| fix-round report commit | Reconciles the accepted replay and composition disclosures | unchanged | no measurement rerun and no measured value changes |

No post-measurement commit changes this study's execution graph, lowerer,
collective plan, backend call, calibrated parameter or service equation. The
physical propagation reference remains 2,000,000 ps. The study-specific later
commits change only a fatal quantization oracle, evidence classification, guard
coverage and reporting.

## What ran

Attempts one and two used 9 sink cells, 6 all-local calibration cells and one
unsupported-width preflight. Attempt three added mixed-placement off,
explicit-legacy and enabled cells. Each of its two classification records has
19 cell directories comprising 12 backend sink configurations, 6 all-local
calibration cells and the unsupported-width preflight. The accepted replay's
backend sink configurations executed 18 steps through 450 backend invocations
and produced 450 GOAL, 450 binary and 450 CSV artifacts. The mixed cells place
four ranks on each of two nodes. The accepted replay occupies 5.9 MiB, and the
complete retained wave output, including both earlier attempts, both
attempt-three classification records and the source attachment, occupies
about 24 MiB of allocated disk.

The non-void record observed:

| Provenance item | Value |
|---|---|
| SimLLM revision | `5eb8041db9260e32f4da82dc9143958e255a6a65` |
| attempt-two refreeze retained for attempt three | `3a6126e174e859d5c222e137dc9e2d94ead6db29` |
| htsim gitlink observed | `fc4400e4ca619223481536632074045cb6af2756` |
| `htsim_rnic` SHA-256 | `32035c778e40e9b11dd32d081350a36a92872855a97dc4b5f217c634420c0816` |
| converter SHA-256 | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |
| Python | 3.12.12 |

The htsim value is observed provenance, not a frozen equality requirement.

## Sources and calibration

The physical capacity bound comes from NVIDIA's
[DGX B200 system guide](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html),
which describes eight B200 GPUs and 14.4 TB/s aggregate fifth-generation
NVLink switch bandwidth. NVIDIA's
[DGX GB200 guide](https://docs.nvidia.com/dgx/dgxgb200-user-guide/dgxgb200-user-guide.pdf)
identifies the fifth-generation switch figure as full duplex. The resulting
one-direction ceiling is 900 GB/s per GPU. This is a capacity bound, not the
calibrated effective rate.

The timing anchor is the public `nccl-tests`
[issue 333](https://github.com/NVIDIA/nccl-tests/issues/333) attachment
[`nccl-test-result.zip`](https://github.com/user-attachments/files/21326711/nccl-test-result.zip).
It records an eight-B200 system, driver 570.158.01, CUDA 12.9, NCCL
2.27.0a0, and local buffer registration. The attachment SHA-256 is
`91629a3b4a6eff4ac2e8bbc2261a928dbcca42f07c02d7f1fe15f9d981d0713f`;
the selected `nccl-b200.local.tsv` SHA-256 is
`639348d43e625d8b7199c45db29ec7a848165974142016fab7b85ca34564a8f3`.
This is a user-supplied public capture hosted with an NVIDIA issue, not a
vendor guarantee and not a local measurement by this worker.

Following the repository's
[published-figure calibration precedent](../../docs/papers/msg-size-vs-bandwidth.md),
the source role, extraction, fit interval and uncertainty stay separate. The
fit uses `out_time` rows from 8 B through 256 KiB at participant widths 2, 4
and 8, excluding 4 KiB at every width. The three 4 KiB rows are holdouts. One
shared endpoint-byte slope and one intercept per observed width produce the
smallest form these data identify:

```text
collective_service_ps(E, W)
    = participant_latency_ps[W]
    + ceil(E * 1e12 / effective_bandwidth_bytes_per_second)
```

Concurrency retains the traffic model's full-duplex per-endpoint ledgers and
prices a phase by their maximum, not their sum. The public `out_time` rows
identify that endpoint completion form, but do not identify a separate channel
count or per-link topology parameter.

| Parameter | Calibrated value |
|---|---:|
| effective endpoint bandwidth | 70,027,079,100 B/s |
| width-2 base latency | 10,722,112 ps |
| width-4 base latency | 15,745,167 ps |
| width-8 base latency | 30,128,029 ps |
| supported source payload envelope | 8 B to 262,144 B |

The effective rate is 7.78 percent of the 900 GB/s one-direction capacity
ceiling. The model rejects participant widths other than 2, 4 and 8 because
three measured widths do not identify a defensible interpolation law. The
intercepts are reduced-form collective floors. They do not separately resolve
launch, rendezvous, protocol setup, reduction arithmetic and propagation.

No same-generation point-to-point payload capture or B200 all-to-all capture
was available. The point-to-point omission is the exact TRAF-31 residual. The
all-reduce anchor's use for the mission's all-to-all remains an uncertainty
carried in the error budget, not a claim that the two collectives are equal.

## Physical sanity before precision

These bounds were written before either production value was read.

1. **Capacity floor and capture ceiling.** At 900 GB/s, endpoint loads 4,096,
   6,144 and 7,168 B cannot serialize faster than 4.552, 6.827 and 7.965 ns.
   The selected real B200 capture spans roughly 9.56 to 38.83 us over the fit
   interval, so the registered small-collective defect envelope is 5 to 50 us.
2. **Collective-count floor and ceiling.** Forty-eight small collectives at the
   [mission study's comparable-system 15 to 30 us band](../end_to_end_replay_v1/RESULTS.md#error-budget)
   contribute 0.72 to 1.44 ms. Applying the registered 10 percent calibration
   uncertainty to the configured point gives 1.302 to 1.591 ms, with 1.60 ms
   as the duplicate-charge tripwire.
3. **Bandwidth response.** At fixed 200,704 endpoint bytes, halving 400 to
   200 Gbit/s must add 4.014080 us of serialization. A real additive floor must
   move the total ratio toward one rather than scale with the link rate.
4. **End-to-end deployment envelope.** The
   [mission error budget](../end_to_end_replay_v1/RESULTS.md#error-budget) is
   1.1 to 4.5 ms per decode step. The old simulated point is 0.205 ms, and the
   configured floor gives about 1.651 ms when added to that rounded published
   literal, before resolving composition with the same-wave host-cost work.

The measured values are shown only after those bounds:

| Width | Capacity floor ps | Registered ceiling ps | Public capture ps | Model ps | Position |
|---:|---:|---:|---:|---:|---|
| 2 | 4,552 | 50,000,000 | 10,520,000 | 10,780,604 | inside |
| 4 | 6,827 | 50,000,000 | 16,180,000 | 15,832,905 | inside |
| 8 | 7,965 | 50,000,000 | 30,310,000 | 30,230,390 | inside |

The configured 1.446145392 ms addition is 0.4 percent above the mission
budget's nominal 1.44 ms upper endpoint, although it remains inside the frozen
1.302 to 1.591 ms calibration range and below the 1.60 ms duplicate-charge
tripwire. Its width-8 intercept comes from a DGX B200 intra-node NVLink
ALL-REDUCE and is applied unchanged to the reference step's cross-node pairwise
ALL-TO-ALLV operations. That operation mismatch can bias the residual in
either direction, so this record does not claim that the floor is settled. The
1.651145392 ms mission point is arithmetic on main's published literals,
`0.205000000 + 1.446145392 ms`; it is not a measured composed run. The
bandwidth increment is exactly 4.014080 us, and the arithmetic mission point
sits inside the 1.1 to 4.5 ms real-system envelope as a plausibility check.
These three angles are independent: link capacity, a real collective capture
and an end-to-end comparable-deployment budget.

## C1: held-out completion error

The registered bar is at most 10 percent or 1 microsecond absolute error,
whichever is larger.

| Width | Capture us | Prediction us | Absolute error us | Relative error | Allowed us |
|---:|---:|---:|---:|---:|---:|
| 2 | 10.520000 | 10.780604 | 0.260604 | 2.4772% | 1.052000 |
| 4 | 16.180000 | 15.832905 | 0.347095 | 2.1452% | 1.618000 |
| 8 | 30.310000 | 30.230390 | 0.079610 | 0.2627% | 3.031000 |

C1 passes at every held-out width. The three rows are instances of one
genuine-risk family, not three additional families.

## C2: live participant-width response

The all-local sink path ran one layer with two semantic collectives. Both the
enabled completion and the enabled-minus-off change increase strictly with
participant width:

| Width | Enabled step us | Enabled minus off us |
|---:|---:|---:|
| 2 | 21.566224 | 21.544224 |
| 4 | 31.672334 | 31.634334 |
| 8 | 60.482058 | 60.424058 |

C2 passes. Every local cell emitted zero fabric bytes. The active local rate
was 70,027,079,100 B/s, while the off cells retained 450,000,000,000 B/s.

## C3: exact sensitivity relation, unscored

| Link rate | Raw transport us | With width-8 base us |
|---:|---:|---:|
| 400 Gbit/s | 6.014081 | 36.142110 |
| 200 Gbit/s | 10.028161 | 40.156190 |

The off-path slow-to-fast ratio is 1.6674469466. With the calibrated floor it
is 1.1110637979, inside `[1.10, 1.12]` and strictly closer to one. Halving the
rate adds exactly 4.014080 us to both paths. Rate cancellation returns
2.000001 us: the separately reported 2.000000 us propagation reference plus
one picosecond of registered backend completion quantization. C3 passes.

This relation is not genuine-risk evidence. The exact sensitivity-fixture
guard pins both raw transports, the parameter-identity guard pins the width-8
base, and the artifact-equation guard pins enabled service to base plus raw
transport. Together they uniquely determine both ratios. The original freeze
incorrectly argued that evaluating the ratio first avoided entailment;
evaluation order cannot restore mathematical independence. C3 is therefore
retained as an exact-unscored mechanism check and removed from the behavioral
numerator and denominator.

## C4: exact TTFT and TPOT reach relation, unscored

The fixture defines TTFT as its first prefill-shaped `StepResult` and TPOT as
the mean of two equal decode-shaped `StepResult` values.

| Metric | Off ms | Enabled ms | Absolute increase ms | Relative increase |
|---|---:|---:|---:|---:|
| TTFT | 0.635809968 | 2.081955360 | 1.446145392 | 227.45% |
| TPOT | 0.209194608 | 1.655340000 | 1.446145392 | 691.29% |

Both metrics rise by exactly
`48 * 30,128,029 = 1,446,145,392 ps`. This relation is exact-unscored.
`operation_inventory`, `one_base_charge_per_semantic_collective`,
`artifact_field_equations`, `step_service_conservation`, and the enabled GOAL,
backend-artifact and backend-outcome identity guards fix both absolute deltas
before C4 evaluates. With equal deltas, the relative-order clause reduces to
`ttft_off > tpot_off`, which the fixed 32-token prefill at 458,752 endpoint
bytes and one-token decode at 14,336 endpoint bytes makes true by construction.
C4 therefore contributes no behavioral numerator or denominator. TTFT and
TPOT reach remains established by `step_service_conservation`, which connects
the collective artifacts to scheduler-visible `StepResult` values.

## Fatal guards

Fatal guards are not a score. Every declared fatal guard held in attempt three,
so its behavioral result is interpretable. They established:

- exact default-`None` versus explicit-`legacy` results, locality rows, backend
  outcomes and generated artifact bytes;
- exact omitted-placement versus explicit all-remote identity with the floor
  off and enabled;
- byte-identical GOAL and backend CSV artifacts between rate-matched off and
  enabled cells, including raw flow rows, timestamps, bytes and order;
- exact `None` versus `legacy` identity for a two-node mixed placement, plus an
  enabled mixed result with 229,376 fabric bytes and 172,032 local bytes in
  total across its two semantic operations, simultaneous positive local and
  fabric service in each operation, and exactly one base charge per operation
  outside `max(local, fabric)`;
- one base charge per semantic collective, artifact equations and exact step
  service conservation;
- distinct profile, bandwidth, participant table, base, raw transport,
  propagation-reference and composed-service report fields;
- the exact 200,704-byte sensitivity fixture, transport increment and
  quantized rate cancellation;
- unsupported-width rejection before file creation or published-result
  mutation; and
- operation inventory, byte conservation, quiescence, positive time and exact
  source and parameter identity.

Attempt one violated two of these premises and is therefore reported only as
void with findings, even though its raw behavioral values happen to equal
the final attempt-three values.

## Evidence classes and entailment

Evidence classes remain separate:

- **Fatal-unscored:** all declared guards held in attempt three. They carry no
  numerator or denominator.
- **Scored behavioral:** **2 of 2 non-entailed families**, or **100 percent
  genuine risk**. These are C1 and C2.
- **Exact-unscored relations:** C3 and C4 pass, but their predicates are
  entailed and carry no numerator or denominator.
- **Parameterized instances:** three C1 holdouts inside one family, all within
  their bar.
- **Native regression executables:** reported under validation, never added to
  the behavioral count.
- **Change-set provenance:** frozen hashes and parameter identity are unscored.

C1 reads production-model predictions rather than merely echoing the frozen
integer prediction, so a formula or rounding defect can fail it while the
reported parameter fields remain correct. C2 reads live sink completion in the
local-width cells. Those cells sit outside `active_cells`, so the fatal
base-charge and step-service guards do not determine their enabled latencies or
deltas. C3's exact transport, parameter and artifact guards determine both
ratios. C4's identity, inventory, charge, artifact-equation and
step-conservation guards determine both deltas, while the fixed fixture
determines their relative order. C3 and C4 are therefore exact-unscored.
Artifact sums, identity, the exact 48-charge delta and unsupported-width
rejection are likewise exact, configuration-forced or by-construction evidence
and are deliberately not scored.

## Error budget

This branch moves mission error-budget item 2, the collective latency floor.
It does not itself implement the fixed host term or change the compute
calibration term. The same-wave host-cost branch creates a composition question
that is disclosed below.

| Term | Before | After | Change |
|---|---:|---:|---:|
| representative network budget arithmetic | 0.106000000 ms | 1.552145392 ms | +1.446145392 ms, 14.6429x total |
| representative whole-step budget arithmetic | 0.205000000 ms | 1.651145392 ms | +1.446145392 ms, 8.0544x total; not a measured composition |
| fixed per-step host cost in this branch | 0 ms | 0 ms | unchanged here; same-wave profile branch discussed below |
| compute model | flat 0.7 roofline on default B100 | same | unchanged |

`codex/comp2_host_step_cost` lands fixed host cost in the same integration
wave. Under the overlap semantics in that branch's `simllm/compute/host.py`, a
whole-step composition of `max(C + network, N * g)` gives 1.650672126 ms,
rounded to 1.650672 ms, for all four profiles. The exact inputs are
`C = 0.099024000 ms`, raw network service of `0.105502734 ms` and this branch's
`1.446145392 ms` floor. The largest launch demand is 1.340532585 ms, so it is
smaller than the collective-bearing step service. The separate 1.651145392 ms
table value instead adds the floor to main's rounded 0.205000000 ms literal.
An additive host and network composition gives different values. Neither
branch resolved or measured whether these terms compose additively or by
whole-step overlap, so this report leaves that choice unresolved.

Against the unchanged 1.1 to 4.5 ms comparable-deployment band, the old budget
arithmetic was optimistic by about 5.37x to 21.95x. The new arithmetic point
leaves a real-to-model ratio of 0.67x to 2.73x. The lower endpoint below one is
not an accuracy win. Transferring an ALL-REDUCE intercept to cross-node
ALL-TO-ALLV can overprice or underprice the mission operation, and the host
composition is unresolved, so the direction of the residual error is
ambiguous. The result supports the presence, separation and live effect of the
floor, not absolute deployment timing.

## Batched closure scope

### TRAF-11

The registered clause was:

> "calibrate the current flat 450 GB/s, zero-propagation, per-endpoint NVLink
> surrogate against same-generation point-to-point and collective captures.
> Sweep payload and participant count on the reference eight-GPU node, hold out
> at least one payload per participant width, and replace the constant with the
> smallest identifiable bandwidth, latency and concurrency form whose held-out
> phase completion error is at most 10 percent or 1 microsecond, whichever is
> larger. Report the before/after TTFT and TPOT effect and retain the exact
> all-remote identity path."

Evidence mapping:

| Acceptance clause | Evidence |
|---|---|
| collective capture | pinned public eight-B200 all-reduce source and digests |
| point-to-point capture | not demonstrated; moved exactly to TRAF-31 |
| payload and participant sweep | fit rows from 8 B through 256 KiB at widths 2, 4 and 8 |
| one holdout per width | C1's 4 KiB row at every width |
| smallest identifiable form | one shared endpoint slope, concurrent full-duplex endpoint ledgers priced by their maximum, and three observed-width intercepts; unsupported widths reject |
| held-out error bar | C1, every error below the larger of 10 percent or 1 us |
| TTFT and TPOT effect | fatal-unscored `step_service_conservation` connects the 48 calibrated charges to TTFT and TPOT `StepResult` values; C4 retains the exact +1.446145392 ms observation as entailed, unscored evidence |
| exact all-remote identity | fatal off and enabled placement guards plus artifact identity |

**TRAF-11 closes**, with only its missing same-generation point-to-point
source clause retained as TRAF-31.

### COMP-11

The registered clause was:

> "Add peer topology and per-link routing, ingress service and its interaction
> with the receiving GPU's HBM, reduction lanes so a collective's arithmetic is
> priced, and proxy operations. Calibrate the egress latency and bandwidth from
> real captures rather than the current synthetic profiles, and reconcile the
> intra-node split with TRAF-10 so one collective is never counted both here and
> on the fabric backend."

Evidence mapping:

| Acceptance clause | Evidence |
|---|---|
| calibrated egress latency and bandwidth | public B200 collective capture, C1 holdouts and C2 live local sweep |
| reconcile local and fabric authority | two-node mixed cell with simultaneous positive local and fabric service, one-charge guard and `base + max(local, fabric)` conservation |
| preserve raw fabric authority | backend artifact identity and separately reported raw transport and propagation reference |
| peer topology and per-link routing | not demonstrated; moved to COMP-31 |
| ingress and receiving-HBM interaction | not demonstrated; moved to COMP-31 |
| priced reduction lanes and proxy operations | not demonstrated; moved to COMP-31 |

**COMP-11 closes**, with exactly those detailed-mechanism clauses retained as
COMP-31. The strong TRAF-11 calibration result is not used to claim those
unexecuted COMP-11 mechanisms.

## Registered residual IDs

Exactly **two** new IDs are registered, both because a quoted acceptance clause
was not demonstrated:

- **TRAF-31** carries the missing same-generation point-to-point payload
  capture, joint refit and held-out validation.
- **COMP-31** carries peer topology, per-link routing, ingress and receiving
  HBM interaction, reduction lanes and proxy operations.

TRAF-32 remains unused. No ID is created for an adjacent improvement or for
the all-reduce versus all-to-all uncertainty, because neither is an additional
undemonstrated registered clause beyond the two residuals above.

Ledger reconciliation removes TRAF-11 and COMP-11 from their module open-task
registries, adds them to `docs/task-ledger.json`, registers TRAF-31 and COMP-31
in the owning module docs, and regenerates the progress block.

## Contradiction sweep

The owning module docs were corrected. The following integrator-owned stale
statements are reported and deliberately not edited, except that the generated
task-progress block in `docs/README_PRO.md` was regenerated as mandatory ledger
reconciliation:

1. `README.md:280` still calls the NVLink egress model a flat first cut and
   links the remaining peer-topology work to COMP-11 rather than COMP-31.
2. `docs/README_PRO.md:244` still assigns the deeper NCCL work to COMP-11.
3. `docs/README_PRO.md:297` still labels measured NVLink bandwidth, latency and
   concurrency as registered under TRAF-11 rather than calibrated with the
   point-to-point residual under TRAF-31.
4. `docs/architecture.md:571` still assigns peer topology, ingress service and
   reduction lanes to COMP-11 rather than COMP-31.

The historical mission-study row in `docs/README_PRO.md` still reports the
2.000 us propagation term that its own run measured. That is historical fact,
not a contradiction: the new profile adds a distinct non-propagation floor.

## Validation and reproduction

The repository gates are:

```text
.venv/bin/ruff check .
python3 scripts/check_docs_format.py
python3 scripts/task_progress.py --check
.venv/bin/pytest -q
```

Their final outcomes, retained as native-test and repository-gate evidence
rather than added to the behavioral score, were:

| Gate | Outcome |
|---|---|
| Ruff | all checks passed |
| module-doc format | 10 module docs matched; 28 legacy untagged entries remain outside this task |
| task-progress reconciliation | generated block and module-status open counts current |
| Pytest | 1,498 passed, 7 skipped |
| staged-diff hygiene | `git diff --check` clean |

The production runner requires explicit paths and confines output under
`SIMLLM_COLLECTIVE_FLOOR_RUN_ROOT`:

```bash
SIMLLM_COLLECTIVE_FLOOR_RUN_ROOT=<output-root> \
SIMLLM_TXT2BIN=<path-to-txt2bin> \
.venv/bin/python examples/collective_latency_floor_v1/run_study.py \
  --run-dir <output-root>/production-v4 \
  --htsim-rnic <path-to-htsim_rnic>
```

Adding `--check-only` validates the frozen inputs and arithmetic, invokes no
backend, and creates no output directory. Raw outputs remain outside Git. No
submodule pin changed, and no test requires initialized submodules.
