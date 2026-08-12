# Arrival admission v1 results

WORK-3, CORE-31 and PLAY-12 are complete for in-process vLLM replay. A
request can now remain outside `LLMEngine.add_request` until its bookkeeping
arrival becomes eligible on the shared virtual clock. Once handed off, vLLM
remains the sole batching authority. The completion reducer can independently
seed first-token history from the same creation fact, so reported TTFT includes
the arrival-to-first-release interval.

Both scored behavioral families passed. The genuine-risk result is 2/2
families and 8/8 instances. Exact decomposition, admission facts, artifact
identity, source identities, the anchor control and the existing fixture lock
are fatal-unscored evidence and do not increase that denominator.

## Provenance and chronology

The complete expectations were frozen at commit
`f369b719aecfd176ccf37e5b5a27eff07f273b87`. The registered `--check-only`
command passed before that commit using a parser-only harness. It checked the
vLLM version and source hashes, the tracked Granite fixture, the model cache
and the output target. It constructed no model, produced no artifacts and was
removed before the freeze. The freeze therefore preceded implementation and
every result-producing run.

The implementation landed at
`a9975d585b1094b725f81b6c07e16832d77d5e13`. The observed run chronology was:

1. The first full run completed all four gated cells and began the identity
   comparison. It then rejected direct all-at-once request `r1` because the
   study had incorrectly supplied the arrival bookkeeping snapshot to the
   reducer even though the direct legacy loop scheduled every request at time
   zero. The run retained its fixed trace and nine completed cell artifacts,
   but wrote no `summary.json` or `metrics.csv`, so it contributes no pass.
2. Commit `3988eed91f44572edf0baa8fba263c8775831f0b` corrected the study wiring.
   Arrival-based metric seeding is now selected only with arrival-gated
   admission; both direct and disabled all-at-once cells use the legacy
   reducer origin. No expectation or implementation contract changed.
3. A fresh run completed every gated and identity cell. Its final line was
   `arrival admission study passed 4/4 spacing and 4/4 load relations`.

The final `$SIMLLM_DATA_ROOT/arrival_admission_v1-run2/summary.json` has
SHA-256 `84ef703784182499b0a0c01c10bb253c4e9c7f9111afb08c26c78a7993537828`.
The paired `metrics.csv` has SHA-256
`bf88d40b0e5cb095e065db28d51ef9aaf6269200e5cd17a4bcd64b0dd568b4b2`.
The derived fixed replay trace has SHA-256
`27c1ea1160de6ea33b7909dbc3e5fc4d074b68d08ae71859787a681286276e5b`.

The run observed vLLM 0.26.0 and the five source hashes registered in the
expectations. These are provenance and fatal configuration checks, not
behavioral evidence. The study records no equality requirement between a
live submodule pin and a frozen literal, and it does not touch a submodule.

## Configuration

The live engine used `max_num_seqs=2`, greedy replay, two prompt tokens per
request, no chunked prefill, no asynchronous scheduling and no prefix caching.
Request `r0` replayed tokens `(38, 39, 40, 41)`. Followers `r1`, `r2` and `r3`
replayed `(61,)`, `(62,)` and `(63,)`. Every request ended for the framework's
length reason, corresponding to the captured length-cap stop reason.

The sweep crossed two arrival offsets, 750,000 ps and 1,250,000 ps, with two
offered loads. The one-follower cell contained `r0` and `r1`; the
three-follower cell contained `r0`, `r1`, `r2` and `r3`. Every nonempty live
engine step became one core execution graph with exactly 1,000,000 ps of
synthetic compute service. That service is an exact study input, not a Granite
latency or hardware calibration claim.

The real scheduler produced these request sets, with one list per step:

| Load | Offset (ps) | Live vLLM batches |
|---|---:|---|
| one follower | 750,000 | `r0`; `r0,r1`; `r0`; `r0` |
| one follower | 1,250,000 | `r0`; `r0`; `r0,r1`; `r0` |
| three followers | 750,000 | `r0`; `r0,r1`; `r0,r2`; `r0,r3` |
| three followers | 1,250,000 | `r0`; `r0`; `r0,r1`; `r0,r2`; `r3` |

These shapes establish live scheduler reachability but are structural and
unscored. The harness neither proposes nor chooses any batch.

## Scored behavioral relations

### A1: arrival-stagger movement

All four paired follower instances moved by the exact registered signed
amount when arrival increased from 750,000 ps to 1,250,000 ps:

| Load | Request | Queue delta (ps) | Service delta (ps) | TTFT delta (ps) | Tokens and finish |
|---|---|---:|---:|---:|---|
| one follower | `r1` | +500,000 | 0 | +500,000 | unchanged |
| three followers | `r1` | +500,000 | 0 | +500,000 | unchanged |
| three followers | `r2` | +500,000 | 0 | +500,000 | unchanged |
| three followers | `r3` | +500,000 | 0 | +500,000 | unchanged |

A1 passed 4/4 instances, all 4/4 at genuine risk. The relation was evaluated
from raw arrival, first-release and first-completion observations before the
exact decomposition check.

### A2: offered-load queue slope

The three-follower burst exceeded the one available slot beside the running
anchor. At each offset, both successive follower queue differences were the
registered positive 1,000,000 ps:

| Offset (ps) | Pair | Queue delta (ps) |
|---:|---|---:|
| 750,000 | `r1` to `r2` | +1,000,000 |
| 750,000 | `r2` to `r3` | +1,000,000 |
| 1,250,000 | `r1` to `r2` | +1,000,000 |
| 1,250,000 | `r2` to `r3` | +1,000,000 |

A2 passed 4/4 instances, all 4/4 at genuine risk. It is independent of A1:
A1 compares one request across offsets, while A2 compares different requests
at one offset and can fail if the gate bypasses the framework waiting queue.

## Raw first-token metrics

| Load | Offset (ps) | Request | Queue (ps) | Service (ps) | TTFT (ps) | Tokens |
|---|---:|---|---:|---:|---:|---:|
| one | 750,000 | `r0` | 0 | 1,000,000 | 1,000,000 | 4 |
| one | 750,000 | `r1` | 250,000 | 1,000,000 | 1,250,000 | 1 |
| one | 1,250,000 | `r0` | 0 | 1,000,000 | 1,000,000 | 4 |
| one | 1,250,000 | `r1` | 750,000 | 1,000,000 | 1,750,000 | 1 |
| three | 750,000 | `r0` | 0 | 1,000,000 | 1,000,000 | 4 |
| three | 750,000 | `r1` | 250,000 | 1,000,000 | 1,250,000 | 1 |
| three | 750,000 | `r2` | 1,250,000 | 1,000,000 | 2,250,000 | 1 |
| three | 750,000 | `r3` | 2,250,000 | 1,000,000 | 3,250,000 | 1 |
| three | 1,250,000 | `r0` | 0 | 1,000,000 | 1,000,000 | 4 |
| three | 1,250,000 | `r1` | 750,000 | 1,000,000 | 1,750,000 | 1 |
| three | 1,250,000 | `r2` | 1,750,000 | 1,000,000 | 2,750,000 | 1 |
| three | 1,250,000 | `r3` | 2,750,000 | 1,000,000 | 3,750,000 | 1 |

All requests retained their exact captured token tuples and length finish
reason. This table is the raw observation set used by A1 and A2.

## Entailment analysis

The harness stores raw live-engine observations and evaluates A1 and A2
before exact TTFT decomposition, admission-fact equality, anchor controls or
artifact identity. No earlier fatal oracle pins either scored outcome.

The later decomposition requires reported queue, service and TTFT to equal
the same raw quantities exactly. It would entail a relation over those values
if evaluated first, so all 12 decomposition instances are explicitly fatal
and unscored. Admission timestamps, source and fixture hashes, the anchor's
zero queue, direct-versus-disabled artifact equality and configuration echoes
are likewise fatal-unscored. None contributes to the 8/8 denominator.

Each scored instance carried genuine risk. A gate could release early, advance
past a ready framework step, reorder the tied burst, seed TTFT at scheduling
rather than arrival, or insert followers without using vLLM's waiting queue.
Those defects could preserve token replay and still fail A1 or A2.

## Fatal exact and identity evidence

All 12 request rows satisfied
`TTFT_ps = queue_ps + service_ps` from independent raw and reducer values.
All four gated cells admitted no request before eligibility and appended one
scheduler-entry fact per successful handoff at the observed virtual time.
Equal-time followers remained in bookkeeping sequence order. Unit tests also
proved that callback failure adds no fact and that scheduling before arrival
rejects without advancing reducer time or history.

For both loads, the direct legacy loop and the default all-at-once gate had
identical final clocks, output tables, replay runs, replay snapshots,
bookkeeping ledgers and streamed step bytes. Their artifact hashes were:

| Load | Bookkeeping | Outputs | Replay run | Replay snapshot | Step stream |
|---|---|---|---|---|---|
| one | `e27c3928b20a9963ff35f35643e5fdce196b213d0ded39360b1b20c017b59a1f` | `be9f68900ad3e407feb35d13a3efadb1088bf7fdf96d1eab4989850fe9076e6a` | `ffcef020373be1a099b6c42fdd66748b9427a201b572971073c8bbcd7c141d5e` | `e22c2ea6af92466c1796a3455b6b304abbd44d994d9aa53df2a0cc336b79c972` | `0956faa386072943f4af32bb137f373d84247800e48b8d2e049277fcd0950e16` |
| three | `3b6b4868d2bf0deac528ee287ca4d6de18bc3c1934df1d911bfa49155763baf9` | `a7df309a9dbb75488ecd62c565fcc3e936e791607b7ce219f548a82f707fb554` | `c6559e7654915ad36b2ba863e758244e13c87896a81ed3312483823ba7684702` | `36a25ecd8d385acd7e84d4077b89d8029151da758324b13f0a2fadcd5a70a6a7` | `b99f562f323adabc631a31e422dcba705f6c4bff7ea79eb82fe8d19d3a5ebb64` |

The existing no-replay adapter fixture retained SHA-256
`71862c9a49814bef3fc830f647f1b439d9c4d6ad0ef9707be6597528adb1808a`.
These identity and byte-lock checks passed but are by-construction or
change-set guards, so they remain unscored.

## Placement and authority of the gate

`RequestAdmissionGate` lives in `simllm.workload` and is driven by the
in-process replay harness. It consumes framework-request creation facts from
core bookkeeping, advances the shared clock only when the engine has no
admitted work to step, and calls the supplied framework `add_request` callback
for the eligible prefix. A successful call adds a scheduler-entry projection.
It never inspects scheduler candidates, capacity or batch composition.

This is the honest seam for the supported stepping loop because that caller
owns when `add_request` and `step` are invoked. A server has an independently
running wall-time ingress and does not expose the same caller-owned loop. No
server-mode timing claim follows from this result; WORK-4 remains the ingress
coordinator needed there. Likewise this task adds no framework admission
control or scheduling policy. CORE-32 owns rejection, rate limits, caps and
policy-driven deferral after arrival eligibility.

## Closure scope

WORK-3 registered this acceptance:

> Consume framework-request creation timestamps through an opt-in in-process
> admission gate. Hold each request outside `add_request` until the shared
> virtual clock reaches its arrival, retain stable bookkeeping order for ties,
> and leave batching entirely to the framework scheduler. Acceptance must
> sweep arrival offset and a burst that exceeds one available scheduler slot,
> reproduce exact per-request queue and TTFT movement, and keep the all-at-once
> path byte-identical.

The gate and four admission-fact checks cover withholding, clock eligibility
and stable order. The live batch table demonstrates scheduler authority. A1
and A2 cover both sweep dimensions, exact queue and TTFT movement, and a burst
above one available slot. The two full artifact comparisons cover the identity
off path. Every WORK-3 clause is demonstrated.

CORE-31 registered this acceptance:

> Seed first-token metric history from the exact framework-request creation
> timestamp in a supplied bookkeeping snapshot. The interval from arrival to
> the first scheduled graph release must enter critical-path queue attribution,
> and TTFT must equal that queue interval plus first-token service exactly.
> Acceptance must reject scheduling before arrival atomically and preserve
> every existing reducer result when no bookkeeping snapshot is supplied.

All 12 decomposition rows cover the arrival origin, queue attribution and
conservation identity. Focused tests cover atomic rejection. The direct and
disabled study cells plus legacy completion tests cover the absent-bookkeeping
identity. Every CORE-31 clause is demonstrated.

PLAY-12 registered this acceptance:

> Drive a joined vLLM replay through the in-process arrival gate using the
> bookkeeping timestamps written by the pre-play join. Acceptance must use a
> fixed capture, sweep arrival offset and offered burst load through the real
> scheduler, preserve each request's captured token count and stop reason,
> demonstrate exact queue plus service TTFT decomposition, and compare the
> all-at-once bypass artifacts byte for byte with the direct legacy loop.

The derived fixed Granite replay trace and bookkeeping extraction cover the
joined capture. The four live vLLM cells cover both sweeps. A1, the raw metrics
and all 12 decompositions cover tokens, finish reasons and TTFT. The identity
hashes cover the bypass. Every PLAY-12 clause is demonstrated.

WORK-2 remains open for bursty or MMPP arrival generation. WORK-4 remains open
for server-mode ingress. CORE-32 remains open for optional admission-control
policy, and PLAY-7 remains open for SGLang replay. The present gate deliberately
models none of those paths.

## Contradiction sweep

The required post-closure sweep inspected every arrival, admission and
queue-related hit in `README.md`, `docs/README_PRO.md` and
`docs/architecture.md`. No hit says that arrivals remain bookkeeping-only or
that requests cannot wait for simulated admission. The architecture statement
that the virtual clock orders request arrivals is now backed by this live
path. The README and developer guide retain higher-level arrival-process and
pre-play-join descriptions; their omission of the new gate is not a
contradiction. No narrative text in those integrator-owned files was changed.
Only the generated task-progress block and machine-checked open counts in
`docs/README_PRO.md` were reconciled as required by the closure ledger.

The tracked ID sweep found the three closed IDs only in this evidence, their
module completion status and `docs/task-ledger.json`. One older routed-supply
result mentions PLAY-12 only as part of an unused preallocated ID range. It is
a historical authorship statement, not an open-task or behavior claim, and was
left unchanged.

## Validation

- The final implementation-side `--check-only` command passed without
  constructing an engine or writing output.
- `ruff check .` passed.
- `pytest -q` passed with 921 tests and 7 skips.
- `python3 scripts/task_progress.py --check` reported that the progress block
  and module-status open counts are current.
- `git diff --check` passed.

This checkout does not contain the locally documented standalone
`scripts/check_docs_format.py`; the full test suite and task-progress checker
therefore provide the available tracked documentation gates.
