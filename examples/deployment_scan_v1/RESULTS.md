# Deployment scan v1 results

## Verdict

**PASS.** The corrected scoring record is non-void. Every runtime fatal
precondition held, every corrected frozen family passed, and the largest
compatibility error across both 18-cell step-time reproductions was exactly
**0 ps**. The scan used one process and no process-creation interception fired.

The result validates the backend-free deployment planning rung as the exact
installed projection frozen by this study. It does not validate the absolute
accuracy of a deployment estimate against hardware.

## Post-specified corrections

The publication at commit `372d077` remains the run of record for the published
point quantities. The rerun using scorer commit
`ee48d2a66b271fc3475632bf879c36aafb24f64b` is the scoring record. The
original B1-exact result of 36 of 36 was not decided by the frozen predicate:
it compared each point with terms that the point constructor had already
required to compose to that value. That result is superseded by the corrected
36-cell evaluation below.

| Finding | Class | What changed | What did not change |
|---|---|---|---|
| B1-exact tautology | BLOCKER | C1 independently recomputes the 18 anchors at 400 Gbit/s. B1 independently reads the pinned maximum remote-flow bytes and local logical bytes, recomputes the 36 floors at 200 and 100 Gbit/s, compares each emitted fabric term with that floor, and composes each step only from the pinned kernel and recomputed network floors. All 36 corrected cells passed. | The 18 analytical and 18 simulation-derived point values stayed byte-identical. |
| Stamp guard theater | MAJOR | FG-2 is labeled enforced by construction and carries no runtime evidential weight. `EstimateStamp.schema` fixes the schema, while `FrontierPoint.__post_init__` enforces the point class. FG-4 now flips one serialized point class and proves that strict parsing rejects the mutation. | The schema and point-class contract did not change. |
| Chronology hard-coded PASS | MINOR | FG-6 is labeled verified out of process. The CI test uses `git cat-file` and `git show` to verify that the cited commit changed only `expectations.md` and that its tree has no `simllm/deploy/`; it skips with an explicit shallow-clone reason if the commit is absent. | The frozen expectations commit and chronology did not change. |
| Unchecked spot literals | MINOR | C1 checks all four frozen analytical integers. C2 checks the frozen B100 batch-32 simulated integer and proves it is the unique simulated-differs-from-analytical point. | Every compatibility error remains 0 ps. |
| Missing B1 anchor conditional | MINOR | Both B1 rows become `UNEVALUATED` if C1 fails, so an invalid kernel anchor cannot produce an interpretable B1 score. | C1 passed in the corrected rerun, so both B1 rows were evaluated and passed. |
| Incomplete W1a predicate | MINOR | W1a now requires both at most 10 seconds and at least 64 priced points. | The frozen primary width remains 72 points. |
| Partial evidence coverage | MINOR | FG-3 now checks the analytical, simulated, bandwidth, rejected-candidate, post-specified and 6,000-point throughput records, 6,073 points in total. It also asserts `DECLARED` separately for handoff and prefill service, `DECLARED` for the synthetic surface and `MEASURED` for the measured probe. | No term source or evidence label changed. |
| D4 construction review | STRUCTURAL | D4 remains a runtime negative control because the scorer observes a rejected candidate, its stable `pipeline-parallel-unpriced` reason and zero emitted points. It is not reclassified as constructor-only evidence. | D4 remains fatal and unscored. |

### Post-specified floor-division regression

This check is labeled **POST-SPECIFIED**, kept outside every frozen score and
not added to any frozen denominator. For `h100-two-node-serialized` batch 1,
the pinned maximum flow is 6,651,904 bytes. At 300 Gbit/s, direct floor
division gives 177,384,106 ps. Rounded scaling of the 400 Gbit/s floor gives
177,384,107 ps. The installed estimator emitted 177,384,106 ps, so the check
passed and would have caught a rounded-scaling implementation.

### Artifact comparison with the published run

- `results.json` changed from
  `4e70a01a1f5c229db8b1807c049ddd35c9632eea42a4dd07b8a18a8c1ffde2f2`
  to `05a6f627a770054687d3e01d5edbd867577b0345fb1bf5a559f43e5a1cfe2ffd`.
  The changed top-level fields are only the corrected fatal-guard ledger, score
  fields and verdict, post-specified regression, wall times and implementation
  provenance.
- `results.csv` is byte-identical at
  `e6d3f43fe76f2c23cc2ad099e7b3931d32803ee1de668bfd9b9e320b8e861a37`.
  The complete analytical, simulated and bandwidth frontier records are also
  structurally identical in the two JSON files.
- `expectations.md` remains byte-identical at
  `619e80027700669393e7477d5674d3a8758a0897b5a18bc89650122bc67760d8`.
  The figure files were not regenerated. This report changed only to disclose
  the scoring correction and corrected rerun.

## Fatal preconditions

Runtime fatal rows are preconditions, not scored instances. Any failure would
void the entire run. Construction and out-of-process rows are labeled plainly
and add no runtime evidential weight.

| Guard | Outcome | Evidence |
|---|---|---|
| FG-1 zero subprocess | PASS | `subprocess.Popen` and `os.posix_spawn` were intercepted around every scan and estimator call; zero interceptions fired |
| FG-2 stamp | ENFORCED BY CONSTRUCTION | `EstimateStamp.schema` fixes the v1 tag and `FrontierPoint.__post_init__` enforces the point class; FG-4 carries the observable mutation control |
| FG-3 evidence | PASS | all sourced terms across seven records and 6,073 points used the frozen classes; handoff and prefill service were each `DECLARED` |
| FG-4 strict schemas | PASS | every candidate, estimate stamp and frontier record round-tripped; unknown fields were rejected; a serialized point-class mutation was rejected |
| FG-5 input pinning | PASS | all three frozen SHA-256 digests matched before evaluation |
| FG-6 chronology | VERIFIED OUT OF PROCESS | the CI test and integrator verify expectations commit `15ee956e2ba54a851884d2cba5d6abd7ca0cdd8d`; the runner makes no Git or process call |
| D4 structural refusal | PASS | the runtime negative-control scan returned `pipeline-parallel-unpriced` and no points |

## Scored evidence

The classes below remain separate. Their instance counts are not added into a
single headline score.

### Compatibility exact oracles

| Family | Outcome | Instances | Deciding result |
|---|---|---:|---|
| C1 analytical reproduction | PASS | 18 of 18 | maximum absolute error 0 ps; four frozen spot literals and all independent 400 Gbit/s component recomputations matched |
| C2 simulation-derived reproduction | PASS | 18 of 18 | maximum absolute error 0 ps; the frozen simulated spot matched and B100 batch 32 was the unique analytical difference |
| C3 exact coordinates | PASS | 18 of 18 | zero fraction mismatches |

### Synthetic exact oracles

| Family | Outcome | Instances | Observed literal |
|---|---|---:|---|
| E1 roofline | PASS | 1 of 1 | 1,250,000,000 ps |
| E2 fabric floor | PASS | 1 of 1 | 10,000,000,000 ps |
| E3 intra-node floor | PASS | 1 of 1 | 2,000,000,000 ps |
| E4 surface interpolation | PASS | 1 of 1 | 400,000,000 ps, `DECLARED` |
| E5 deterministic queue | PASS | 1 of 1 | capacity 2,500 request/s; occupancies 2, 7 and 8; final wait 4,725,000,000 ps |
| E6 rate match | PASS | 1 of 1 | five prefill engines and one decode engine |

### Behavioral relations

| Family | Outcome | Instances | Deciding result |
|---|---|---:|---|
| B1 exact bandwidth composition | PASS | 36 of 36 | zero independent fabric-floor or step-composition mismatches from pinned bytes |
| B1 direction | PASS | 1 of 1 | 100 Gbit/s was never faster than 200 Gbit/s, which was never faster than 400 Gbit/s; one binding comparison was strict |
| S1 service-level agreement membership | PASS | 2 of 2 | exact five-point and twelve-point sets |
| P1 Pareto literal | PASS | 1 of 1 | exactly the six B100 points |
| D1 request-speed direction | PASS | 3 of 3 configurations | nonincreasing with batch |
| D2 per-GPU throughput direction | PASS | 3 of 3 configurations | nondecreasing with batch |
| D3 target nesting | PASS | 1 of 1 | tighter targets produced subsets |

### Wall time

The W1 bands are generous by construction. They test the order of magnitude of
the in-process planning path, not a tuned performance ceiling. Time was measured
only inside scan and estimator calls with one Python process.

| Family | Outcome | Width | Observed | Frozen ceiling |
|---|---|---:|---:|---:|
| W1a complete study pricing | PASS | 72 points | 0.036835543 s | 10 s and at least 64 points |
| W1c throughput grid | PASS | 6,000 points | 2.082205866 s | 60 s |

W1b is reported and unscored: the primary scan reached 1,954.63 point/s and
the 1,000-candidate grid reached 2,881.56 point/s. The machine was Linux
5.14.21 on x86-64, an AMD Ryzen 9 3950X with 32 logical CPUs, using Python
3.12.12. The run still used one process.

## Exact membership and front

At a time per output token (TPOT) target of 4,000,000,000 ps, the admitted set
was B100 batches 1, 2, 4, 8 and 16. At 8,500,000,000 ps, it was all six B100
points plus batches 1, 2 and 4 of both H100 configurations. The Pareto front
was exactly B100 batches 1, 2, 4, 8, 16 and 32.

## Physical sanity before exactness

The physical bounds were stated before reading the final values:

- Floor: B100 batch 1 moves 27,587,187,040 logical high-bandwidth-memory bytes
  through an 8 TB/s envelope, so it cannot beat 3,448,398,380 ps.
- Conservative ceiling: if the B100 batch-32 kernel and simulated intra-node
  service serialized instead of overlapping, their sum would be
  8,516,304,727 ps.

The B100 batch-32 result was 4,523,298,348 ps. It lies above its
4,257,218,560 ps intra-node serialization floor and below the conservative
serialized ceiling. Three independent checks support physical plausibility:

1. Memory physics: the B100 batch-1 value equals bytes divided by bandwidth
   under the declared integer projection.
2. Network physics: each 100, 200 and 400 Gbit/s cell recomputed bytes divided
   by link rate with floor division. Every binding term followed the frozen
   inverse-rate direction.
3. System scale: H100 batch 32 produced 104.87 token/s/request and
   3,355.87 token/s/GPU. The paired published anchor is 87.04 and 2,785.25,
   respectively, so the floor-style estimate is 20.49 percent faster on the
   request axis rather than implausibly below the measured reference.

Being inside these bounds is necessary, not sufficient, for hardware accuracy.

## Figure and artifacts

The [frontier figure](figures/deployment-scan-frontier.pdf) shows analytical
lines, hollow `ESTIMATE` points, filled `SIMULATED` points, the emphasized
Pareto front, the paired measured diamond and the frozen y-only production
anchor. The y-only anchor remains a horizontal line and receives no invented x
coordinate. A [PNG rendering](figures/deployment-scan-frontier.png) is included
for direct inspection.

Generated data are in [results.json](results.json) and
[results.csv](results.csv). Their byte locks for this run are:

- `results.json`: `05a6f627a770054687d3e01d5edbd867577b0345fb1bf5a559f43e5a1cfe2ffd`
- `results.csv`: `e6d3f43fe76f2c23cc2ad099e7b3931d32803ee1de668bfd9b9e320b8e861a37`

No scored miss exists, so there is no miss ledger and no DEPLOY-10 through
DEPLOY-12 residual is registered.

## Provenance

- Frozen expectations commit:
  `15ee956e2ba54a851884d2cba5d6abd7ca0cdd8d`
- Candidate and feasibility implementation:
  `6c8957935e217ebc2c588f816fd6fcecc717d0d0`, reviewed by
  `beeaa6d6549a07867a7ab97a5b2c4972b690b2a7`
- Estimator implementation:
  `6e16070fb4d8a772ed738a9490998b30e44183a4`, hardened by
  `110020e80226fd02cd12f037b8c51652e12a27be`
- Frontier implementation:
  `4ec8538823a944b22a20b24f808dea28ecaddb66`
- Study runner and renderer:
  `aebd7ad5ea058a1019c0d2274c8275a53f26a980`,
  `b898e8a22ca38826df76a859edee7a5e24819eca`, and
  `a0ec74c8ba335092d098c734b908c617c6d350e5`; the fatal schema-order audit is
  `069ac9d409c31d6f4f883d34079199e88fc2257a`
- Scoring-fidelity correction:
  `ee48d2a66b271fc3475632bf879c36aafb24f64b`
- Frozen input SHA-256 values:
  `54295c81cebe36ee32d12b8ab1432c9fc060094ddf98403152b0d619cc37438f`,
  `ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2`,
  and `f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad`

## Project consequence

What ran: `examples/deployment_scan_v1` priced the frozen 72-point primary grid
and the 6,000-point throughput grid through the installed candidate, estimator
and frontier contracts with process creation blocked.

What came out: the corrected scoring record passed every frozen family, with a
deciding maximum compatibility error of 0 ps, 36 of 36 independently
recomputed B1 cells and no runtime fatal finding. The separate post-specified
discrimination cell also passed.

What it changes: DEPLOY-1 remains closed. The deployment planner remains a validated
backend-free rung with exact CORE-62 compatibility, strict evidence stamps and
a deterministic frontier artifact.

What it does not change: the result does not establish absolute prediction
accuracy, does not price unsupported parallel widths, does not render physical
host placement, does not promote a point into simulation, and does not change
`PrecisionConfig`, a backend, a GPU path or a serving-framework adapter. Those
optional integrations remain under DEPLOY-2 through DEPLOY-8. The correction
does not change any published point quantity or figure.
