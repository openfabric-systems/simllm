# CORE-5 completion reduction results

## Outcome and chronology

All original CORE-5 relations and all integration-review regressions passed.
The original expectations-only commit is `fc3836d` (`Freeze completion
reduction and Tier B expectations`). It contains only `expectations.md` and
the separate Tier B expectations file. Its commit message records that the
then-untracked `run_study.py` encoded only frozen literals and check-only
validation. Both original check-only commands printed their registry
confirmations by design and produced no artifacts.

The first implementation was `cb3c982`. Integrator probes then exposed the
replay, zero-sample, JSON-null and evidence-accounting defects documented in
the review. Before correcting them or running corrected evidence, commit
`067cbfb` (`Supplement Tier B review expectations`) froze the additive Tier B
producer/schema contract and the post-specified CORE-5 regression checks. It
did not change either frozen Tier A file or weaken the original Tier B file.
Immediately before that freeze, the index held only expectation records,
their declarative JSON and its LF attribute; the tracked working tree had no
unstaged changes. The sole untracked file was
`examples/rnic_live_v1/tier_b_review_check.py`, which encoded only frozen
literals and check-only validation. Both review check-only commands printed
registry confirmations by design and produced no artifacts.

The corrections, executable review checker, tests and measured-evidence
records followed `067cbfb`. The full lint and Python gates passed before the
final measured run. The three final check-only commands again printed their
registry confirmations and produced no artifacts. The resolved machine-local
historical directory is intentionally omitted. The command below is a
post-specified portable reproduction rendering with the same executable and
options; it writes a new run under the configured external data root:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/core5_reduction/review-final-2"
```

The external `results.json` has SHA-256
`38830b099166c46369e933162a3c2f88b857c69196927d8698bdf3a616493a7c`.
No measured artifact or bulk output is tracked in Git.

## Exact two-request rows

Both requests ran one prefill and two decode steps from `T0 = 7,000 ps`.
Every step in a cell had the same exact J, so each request had `TTFT = J`, both
inter-token intervals equaled J, and exact rational `TPOT = J/1` after either
decode.

| Shape | R (Gbit/s) | J (ps) | Final clock after 3 steps (ps) | Residual (ps) |
|---|---:|---:|---:|---:|
| parallel | 200 | 356,680 | 1,077,040 | 0 |
| serial | 200 | 366,680 | 1,107,040 | 0 |
| parallel | 400 | 192,840 | 585,520 | 0 |
| serial | 400 | 202,840 | 615,520 | 0 |

The dependency effect was exactly

```text
J(serial, R) - J(parallel, R) = +10,000 ps
```

at both rates. The rate effect was exactly

```text
J(shape, 200) - J(shape, 400) = +163,840 ps
```

in both dependency shapes. All four signed effects matched the frozen 0 ps
residual band. These effects were subtracted from measured
`StepResult.step_latency_ps` rows. Frozen JCT values appeared only on the
expected side of each scored record.

## Conserved request components

Each request row matched the frozen component vector exactly:

| Shape | R | queue | KV | kernel | DMA | collective | NIC | control | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| parallel | 200 | 163,840 | 0 | 20,000 | 0 | 8,000 | 163,840 | 1,000 | 356,680 |
| serial | 200 | 163,840 | 0 | 20,000 | 10,000 | 8,000 | 163,840 | 1,000 | 366,680 |
| parallel | 400 | 81,920 | 0 | 20,000 | 0 | 8,000 | 81,920 | 1,000 | 192,840 |
| serial | 400 | 81,920 | 0 | 20,000 | 10,000 | 8,000 | 81,920 | 1,000 | 202,840 |

The table passed for both requests in all four cells. The conservation
identity itself is fatal and unscored. The eight exact component rows are
scored because an implementation can retain the correct total while losing
the serial DMA segment, duplicating the NIC tail or substituting a work sum
for one selected wait.

KV remains zero because the fixture deliberately uses the CORE-3-off,
zero-byte lifecycle path. That inactive value did not add a behavioral pass.

## Additive work stayed separate

Each request projected 21 physical visits; the graph projected 42. Dependency
shape changed the realized path but not physical visit work.

| R | Request queue wait | Request service | Request additive total | Graph queue wait | Graph service |
|---:|---:|---:|---:|---:|---:|
| 200 | 163,840 | 385,680 | 549,520 | 327,680 | 771,360 |
| 400 | 81,920 | 221,840 | 303,760 | 163,840 | 443,680 |

Visibility work was 0 ps in every ideal-profile cell. The request additive
totals exceeded both parallel and serial request latency at each rate. The
graph additive queue wait was twice the selected request queue component.
Neither value entered TTFT, TPOT or the seven-component conservation sum.

## Synchronous and asynchronous boundaries

The required compute anchor was 10,000 ps and background physical work was
20,971,520 ps.

| Work kind | Required mode | Step latency (ps) | Completion (ps) | Quiescence (ps) |
|---|---|---:|---:|---:|
| control | asynchronous handoff | 10,000 | 17,000 | 20,978,520 |
| control | synchronous delivery | 20,971,520 | 20,978,520 | 20,978,520 |
| collective | background subset | 10,000 | 17,000 | 20,978,520 |
| collective | required subset | 20,971,520 | 20,978,520 | 20,978,520 |

Both synchronous-minus-asynchronous changes were exactly `+20,961,520 ps`.
The asynchronous event streams retained events after framework completion,
but `VirtualClock` stopped at 17,000 ps. The synchronous rows advanced it to
20,978,520 ps.

## Evidence classes

Evidence remained separated as frozen:

- run configurations: 4;
- exact JCT oracle rows: 4;
- scored behavioral instances: 18 of 18 passed;
- live in-harness structural predicates: 60 of 60 passed, unscored;
- expected validator rejection probes: 2 of 2 passed, unscored;
- compatibility acceptance probes: 2 of 2 passed, unscored;
- full Python suite: 578 passed and 4 skipped;
- lint gate: passed.

The scored total is the sum of only the five behavioral families below. It
does not include configurations, exact row publication, conservation,
inactive fields, event membership, callback identity, unit tests or gates.
Every family total and genuine-risk numerator was derived from its executed
check records. Each record carries measured observations, frozen expected
values or predicates, its pass result and its genuine-risk classification.

The corrected structural count is 12 executed graphs times five recorded
predicates: callback count, callback object identity, event-phase membership,
graph additive queue wait and selected critical-path queue wait. The former
48 count came from incrementing four after a combined callback expression; it
was not an honest count of separately evaluated predicates. Validator raises
are now a different evidence class: duplicate zero-latency replay and
ambiguous partial sampling both rejected atomically. Zero sampled decode rows
and explicit JSON null loaded successfully in the separate compatibility
class. None of these four probes enters a behavioral denominator.

| Behavioral family | Passed | Genuine-risk fraction | Why failure was plausible |
|---|---:|---:|---|
| dependency shape | 2/2 | 2/2 | A missing realized predecessor segment erases the DMA penalty. |
| inverse-rate tail | 2/2 | 2/2 | A lost or duplicated NIC tail changes the signed rate delta. |
| request metrics and components | 8/8 | 8/8 | Correct total latency does not guarantee the correct seven owners. |
| additive-work separation | 4/4 | 4/4 | Visit work can be substituted for a selected path while every visit remains valid. |
| asynchronous boundaries | 2/2 | 2/2 | Quiescence can be mistaken for completion, or required work can be released early. |

## Gates and residual scope

The pre-portability correction gates were:

```text
.venv/bin/ruff check .
All checks passed!

.venv/bin/pytest -q
578 passed, 4 skipped in 13.83s
```

After merging portability commit `561b4d0`, the post-specified path-only round
passed the new scanner and the full merged-state gates:

```text
.venv/bin/pytest -q tests/test_path_portability.py
6 passed in 0.30s

.venv/bin/ruff check .
All checks passed!

.venv/bin/pytest -q
592 passed, 4 skipped in 14.17s
```

The runner's explicit `--out` behavior and every study relation are unchanged,
so this portability-only round did not regenerate the measured artifact.

No C++ source changed, so native CMake and CTest gates are not applicable.

CORE-17 (Completeness; P1; M) remains for framework adapters to populate exact
sampled request identities in a mixed partial-sampling batch. CORE-5 fails
closed for that ambiguous count-only case while preserving zero-sample,
all-sample and absent-count legacy behavior. CORE-18 and CORE-19 were not used
for residual work and are not claimed as open tasks.

Tier B was frozen but not executed. The composed HTSIM-9 and CORE-15 producer
does not exist in this worktree, and this result makes no structural native
live-reachability claim. Its existing residual owners are CORE-15
(Completeness; P1; L) and HTSIM-9 (Completeness; P1; L); duplicating those gaps
under CORE-18 or CORE-19 would make the module registry inexact. The Tier B
check-only registries pass with the exact producer invocation, raw schema,
four retained bypass profiles, two objective doorbell-owner mappings,
preceding-release completion form, zero-service SimLLM profile and two-WQE
FIFO relation frozen without changing the Tier A files.

No framework adapter was changed, no composed binary was fabricated, and no
Tier B result was scored. No C++ source changed, so native CMake and CTest
gates remain deliberately inapplicable.
