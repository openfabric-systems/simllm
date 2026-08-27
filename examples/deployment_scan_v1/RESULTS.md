# Deployment scan v1 results

## Verdict

**PASS.** The run is non-void. All fatal preconditions held, every frozen
scored family passed, and the largest compatibility error across both 18-cell
step-time reproductions was exactly **0 ps**. The scan used one process and no
process-creation interception fired.

The result validates the backend-free deployment planning rung as the exact
installed projection frozen by this study. It does not validate the absolute
accuracy of a deployment estimate against hardware.

## Fatal preconditions

Fatal rows are preconditions, not scored instances. Any failure would void the
entire run.

| Guard | Outcome | Evidence |
|---|---|---|
| FG-1 zero subprocess | PASS | `subprocess.Popen` and `os.posix_spawn` were intercepted around every scan and estimator call; zero interceptions fired |
| FG-2 stamp | PASS | every point carried `simllm-deployment-estimate-v1`; analytic points were `ESTIMATE` and tracked-excess points were `SIMULATED` |
| FG-3 evidence | PASS | all duration terms used one allowed evidence class and a nonempty source |
| FG-4 strict schemas | PASS | every candidate, estimate stamp and frontier record round-tripped; unknown fields were rejected |
| FG-5 input pinning | PASS | all three frozen SHA-256 digests matched before evaluation |
| FG-6 chronology | PASS | expectations commit `15ee956e2ba54a851884d2cba5d6abd7ca0cdd8d` contains only `expectations.md` |
| D4 structural refusal | PASS | pipeline parallelism greater than one returned `pipeline-parallel-unpriced` and no points |

## Scored evidence

The classes below remain separate. Their instance counts are not added into a
single headline score.

### Compatibility exact oracles

| Family | Outcome | Instances | Deciding result |
|---|---|---:|---|
| C1 analytical reproduction | PASS | 18 of 18 | maximum absolute error 0 ps |
| C2 simulation-derived reproduction | PASS | 18 of 18 | maximum absolute error 0 ps; every point was `SIMULATED` |
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
| B1 exact bandwidth composition | PASS | 36 of 36 | zero recomposition mismatches |
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
| W1a complete study pricing | PASS | 72 points | 0.030626657 s | 10 s |
| W1c throughput grid | PASS | 6,000 points | 2.084882394 s | 60 s |

W1b is reported and unscored: the primary scan reached 2,350.89 point/s and
the 1,000-candidate grid reached 2,877.86 point/s. The machine was Linux
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

- `results.json`: `4e70a01a1f5c229db8b1807c049ddd35c9632eea42a4dd07b8a18a8c1ffde2f2`
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
- Frozen input SHA-256 values:
  `54295c81cebe36ee32d12b8ab1432c9fc060094ddf98403152b0d619cc37438f`,
  `ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2`,
  and `f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad`

## Project consequence

What ran: `examples/deployment_scan_v1` priced the frozen 72-point primary grid
and the 6,000-point throughput grid through the installed candidate, estimator
and frontier contracts with process creation blocked.

What came out: the run passed every frozen family, with a deciding maximum
compatibility error of 0 ps and no fatal finding.

What it changes: DEPLOY-1 closes. The deployment planner becomes a validated
backend-free rung with exact CORE-62 compatibility, strict evidence stamps and
a deterministic frontier artifact.

What it does not change: the result does not establish absolute prediction
accuracy, does not price unsupported parallel widths, does not render physical
host placement, does not promote a point into simulation, and does not change
`PrecisionConfig`, a backend, a GPU path or a serving-framework adapter. Those
optional integrations remain under DEPLOY-2 through DEPLOY-8.
