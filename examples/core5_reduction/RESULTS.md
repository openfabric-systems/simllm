# CORE-5 completion reduction results

## Outcome and chronology

All frozen CORE-5 checks passed. The expectations-only commit is `fc3836d`
(`Freeze completion reduction and Tier B expectations`). It contains only
`expectations.md` and the separate Tier B expectations file. Its commit message
records that the then-untracked `run_study.py` encoded only frozen literals and
check-only validation. Both registered check-only commands ran before the
freeze, printed their registry confirmations by design, and produced no
artifacts.

The completion reducer, runtime attribution projection, tests and
result-producing study were implemented after `fc3836d`. The first measured
run then used the registered command:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "$SIMLLM_CORE5_RUN_ROOT"
```

The external `results.json` has SHA-256
`9283ce48a8768f25703ca3086cac22d3f2670976ce208bbe6cb3b4f68c5a5421`.
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
residual band.

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
- fatal structural guards: 48 of 48 passed, unscored;
- full Python suite: 575 passed and 4 skipped;
- lint gate: passed.

The scored total is the sum of only the five behavioral families below. It
does not include configurations, exact row publication, conservation,
inactive fields, event membership, callback identity, unit tests or gates.

| Behavioral family | Passed | Genuine-risk fraction | Why failure was plausible |
|---|---:|---:|---|
| dependency shape | 2/2 | 2/2 | A missing realized predecessor segment erases the DMA penalty. |
| inverse-rate tail | 2/2 | 2/2 | A lost or duplicated NIC tail changes the signed rate delta. |
| request metrics and components | 8/8 | 8/8 | Correct total latency does not guarantee the correct seven owners. |
| additive-work separation | 4/4 | 4/4 | Visit work can be substituted for a selected path while every visit remains valid. |
| asynchronous boundaries | 2/2 | 2/2 | Quiescence can be mistaken for completion, or required work can be released early. |

## Gates and residual scope

The final repository gates were:

```text
.venv/bin/ruff check .
All checks passed!

.venv/bin/pytest -q
575 passed, 4 skipped in 14.13s
```

No C++ source changed, so native CMake and CTest gates are not applicable.

CORE-17 (Completeness; P1; M) remains for framework adapters to populate exact
sampled request identities in a mixed partial-sampling batch. CORE-5 fails
closed for that ambiguous count-only case while preserving zero-sample,
all-sample and absent-count legacy behavior.

Tier B was frozen but not executed. The composed HTSIM-9 and CORE-15 producer
does not exist in this worktree, and this result makes no structural native
live-reachability claim. The Tier B check-only registry passes and is ready for
that producer without changing the frozen Tier A files.
