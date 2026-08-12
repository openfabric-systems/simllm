# vLLM observed schedule v1 results

VLLM-22 and TRAF-13 are complete at the frozen Granite vLLM v0.26.0
boundary. A real eight-rank vLLM replay emitted source-backed
`ExecutionObservations` for all 32 nonempty scheduler steps. Every step carried
all 24 ordered MoE layers and 48 unique semantic collective sites. DBO steps
carried 96 collective invocations across two request-correlated microbatches.
The observations reached traffic rebinding, `CoarseDeviceRuntime`, completion
events, request-attributed `StepResult`, TTFT and TPOT.

The behavioral result is `2/4 = 50%`. Both registered serial-to-observed TPOT
reductions passed their signed bands. Both dependency perturbations changed
TPOT in the registered positive direction on every applicable decode step,
which proves the observations were metric-live, but their magnitudes missed
the frozen minimums. Thirteen of 14 fatal unscored guards passed. The failed
guard was the frozen exact-TTFT assumption, which the source-backed rank-local
DeepEP completion semantics refuted. The misses are retained and defended
below. TRAF-23 owns measured frontier and perturbation-magnitude precision;
VLLM-23 owns the deliberately rejected schedule shapes outside this frozen
boundary.

## Chronology, provenance and reproduction

The complete source audit and expectations were frozen in commit
`a91ac0672b0bb1f6eedd03bbb581e221fb298793` before implementation or a
result-producing run. The registered command passed with `--check-only`
before that commit, imported no SimLLM target module, created no output
directory, and wrote no artifact.

The producer and traffic implementation landed in commit
`d9be55dc4a4577d25ddabd29f984c3d6e8312da2`. The first registered run stopped
at cross-node profile construction before the live engine or any behavioral
cell ran. That attempt is `0/0, blocked before behavioral execution`; its
directory was retained. Commit `e395837d0012b60e01b75fed0d3cd55f75b2803d`
fixed only the study's physical rank projection. No metric had been observed
before that fix.

The final result record observed commit
`e395837d0012b60e01b75fed0d3cd55f75b2803d`. This is run provenance, not an
assertion that a future checkout must equal that commit. The external result
is at
`$SIMLLM_VLLM22_RUN_ROOT/qualification-2026-08-12-rerun1/results.json` with
SHA-256
`7beeb818e6dfa19bb3e0df1064defa418af37ee90c3f8c6fbad2c8282b9fb51d`.
The live observation stream has SHA-256
`e72ba66845047cdce1fdc781c616220527c724aa4b0bea1e6a5779386641e413`.

The evidence was authored against official vLLM v0.26.0 commit
`568afb3a13806beb53bb2e6bd518269357b237c0`. The run independently observed
vLLM version `0.26.0` and all nine source-file hashes frozen in
[the expectations](expectations.md#pinned-source-audit). Authored-against and
observed identities remain separate provenance fields, with no equality
assumption against a live package or repository pin. The 120-row capture was
observed at its frozen SHA-256
`5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`.

Configure the machine-local roots, then reproduce from the repository root.
The selected output directory must not already exist:

```bash
.venv/bin/python examples/vllm_observed_schedule_v1/run_study.py \
  --capture "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/capture/granite-greedy.jsonl" \
  --replay-run "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/run.json" \
  --routed-experts "${SIMLLM_MOE_E2E_ROOT:?configure SIMLLM_MOE_E2E_ROOT}/replay-400g-nvlink/routed-experts.json" \
  --vllm-source "${SIMLLM_VLLM_SOURCE:?configure SIMLLM_VLLM_SOURCE}" \
  --vllm-python "${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" \
  --output-dir "${SIMLLM_VLLM22_RUN_ROOT:?configure SIMLLM_VLLM22_RUN_ROOT}/qualification-reproduction"
```

Add `--check-only` to repeat the complete artifact-free registry validation.

## Evidence classes

The evidence classes answer different questions and are not added together.

| Evidence class | Count | Outcome |
|---|---:|---|
| Physical placement configurations | 2 | Single-node NVLink and one-rank-per-node 400 Gbit/s |
| Raw runtime cells | 6 | Serial, observed and perturbed on both placements, 32 steps each |
| Scored genuine-risk instances | 4 | 2 passed, 2 failed, `50%` |
| Fatal unscored guards | 14 | 13 passed, 1 failed |
| Live framework ranks | 8 | All exited successfully |
| Live nonempty scheduler steps | 32 | 32 emitted observations |
| Direct serial per-step comparisons | 64 | 64 graph, runtime, event, result and metric identities passed |
| Fixed serial compatibility fixture | 1 | Both frozen byte identities passed |
| Full Python regression | 1 final closure invocation | 1,039 passed, 7 skipped |
| Repository lint | 1 final closure invocation | Passed |

Source hashes, fixed configuration values, byte conservation, serial
identities and the absence of an overlap knob are fatal or change-set guards.
They are unscored. Unit tests and the live framework run remain separate from
the four behavioral instances.

## Live producer inventory

All eight vLLM ranks ran the frozen Granite model revision through the actual
v0.26.0 scheduler and `SimWorker` model-runner seam. Rank zero was the sole
schedule, sink and virtual-clock authority. The other ranks participated in
vLLM's DP coordination without advancing a second simulated authority.

The source-derived operation inventory was:

| Step shape | Steps | Operations per step | Collective invocations | Unique `(layer, site)` values | Completion endpoints |
|---|---:|---:|---:|---:|---:|
| Single batch | 9 | 441 | 48 | 48 | 1 |
| Two-way DBO | 23 | 874 | 96 | 48 | 2 |

The single-batch rows comprise the 54-token prefill and the eight one-request
decode-tail steps. The 23 steps contributing to `r0` TPOT all selected the
two-way DBO path. Every row covered layers 0 through 23 and exactly one
`dispatch` and one `combine` site per layer. Every DBO row invoked each site
once per microbatch.

For a DBO layer, tuple order was pre-dispatch compute for microbatch zero and
one, dispatch zero and one, expert compute zero and one, then combine zero and
one. Rank-local compute used `cuda:<rank>:compute`; both microbatches used the
shared `cuda:0:comm:ep` semantic communication queue. Queue FIFO plus explicit
participant-local event edges represented the audited shared streams and
DeepEP event waits. Each microbatch's next layer waited on its own rank-local
combine frontier. Final per-rank logits waited on both final combines, and
each request-visible endpoint waited on every rank's logits.

These edges come from the source paths cited before the freeze in
[the pinned-source audit](expectations.md#pinned-source-audit): vLLM's two
cooperative DBO model threads, shared compute and communication streams,
DeepEP high-throughput yields, and CUDA event waits. Neither the producer nor
the lowerer contains an overlap percentage, duration discount, random choice,
or reconstruction from the serial compatibility graph.

All 128 observed and perturbed lowering checks preserved tuple position,
logical queue, whole-operation edges, participant-local edges, gates,
priority, correlation and completion IDs. The live cross-node sink and the
independent cross-node observed replay matched all 32 lowered graph,
execution-result and `StepResult` identities.

## Traffic and request attribution

Traffic binding treated the adapter's zero-byte all-to-allv rows as semantic
sites, then supplied the captured routed work for each correlated
microbatch. The microbatch request slices were disjoint and source ordered.
Their request rows and aggregate pair rows recombined exactly to every
full-step routed table.

Across the 32 steps, each of the six cells contained 121,646 request-pair
rows and 440,115,200 directed bytes. Every per-step request-pair identity was
the same in serial, observed and perturbed modes and on both placements.
`StepResult` retained the original `r0`, `r1` and `r2` identities while those
requests remained active. These are fatal unscored conservation and identity
checks, not overlap evidence.

## Raw TPOT relations

`r0` TPOT is the exact mean of its 23 decode intervals. The runner wrote all
six raw step streams and evaluated the four relations before interpreting any
later fatal guard.

| Placement | Serial TPOT (ps) | Observed TPOT (ps) | Serial minus observed (ps) | Frozen band (ps) | Outcome |
|---|---:|---:|---:|---:|---|
| `single-node` | 102,567,669.435 | 99,517,385.696 | 3,050,283.739 | 1,000,000 to 5,000,000 | Passed |
| `cross-node` | 127,300,132.174 | 99,851,905.565 | 27,448,226.609 | 20,000,000 to 130,000,000 | Passed |

The observed schedule reduced TPOT by 2.97 percent of serial on the
single-node placement and 21.56 percent on the cross-node placement. The
cross-node reduction was 8.999 times the single-node reduction, passing the
separate frozen requirement that it be at least five times larger.

## Dependency perturbation finding

The perturbation added exactly one whole-operation dependency on each of the
23 DBO steps, from microbatch zero's layer-12 combine to rank zero's
microbatch-one layer-12 expert operation. No work, byte, queue, tuple position,
correlation or completion endpoint changed.

| Placement | Observed TPOT (ps) | Perturbed TPOT (ps) | Increase (ps) | Frozen minimum (ps) | Direction | Scored outcome |
|---|---:|---:|---:|---:|---|---|
| `single-node` | 99,517,385.696 | 99,549,249.696 | 31,864.000 | 100,000 | Positive | Failed minimum |
| `cross-node` | 99,851,905.565 | 100,138,704.826 | 286,799.261 | 5,000,000 | Positive | Failed minimum |

Every single-node DBO step increased by exactly 31,864 ps. Every cross-node
DBO step increased by 286,720 through 288,292 ps. A lowerer that ignored the
observations would have produced zero change, so both direction checks are
decision-relevant and passed. Each frozen instance also required its minimum,
however, so neither instance is counted as passed.

A post-specified critical-path diagnostic explained the miss without tuning
the implementation. On representative decode step 1, the original
single-node target became eligible at 207,590,872 ps, while the added
predecessor completed at 207,622,736 ps. The difference is exactly 31,864 ps.
On the cross-node cell, those times were 739,538,034 and 739,828,718 ps, a
290,684 ps local eligibility change whose realized end-to-end effect was
286,720 ps. The frozen minimums overestimated the removable critical slack by
about 3.1 times and 17.4 times. TRAF-23 retains quantitative frontier and
perturbation precision rather than hiding these misses or inserting a fitted
overlap knob.

## TTFT finding

The frozen expectation that a non-DBO prefill must equal the serial
compatibility path was wrong:

| Placement | Serial TTFT (ps) | Observed TTFT (ps) | Serial minus observed (ps) | Exact expectation |
|---|---:|---:|---:|---|
| `single-node` | 167,543,635 | 164,630,939 | 2,912,696 | Failed |
| `cross-node` | 713,203,520 | 696,566,212 | 16,637,308 | Failed |

DBO was disabled on prefill, but the source does not impose a global barrier
after each DeepEP combine. Each rank waits on its local communication event and
can enter the next layer before the slowest rank's combine frontier. The
producer represented that as a participant-local edge, while the serial
compatibility graph waits on whole-operation completion.

The post-specified layer-zero diagnostic made the distinction observable. In
the single-node cell, combine completed globally at 6,525,728 ps, but rank
one's local completion and next-layer pre-dispatch start were 5,346,991 ps.
In the cross-node cell the corresponding values were 31,611,447 and
21,002,807 ps. The completion endpoint still waited on all ranks' final
logits, so no request was released early. This is a source-fidelity finding,
not a failure of the absent-observation serial path. It remains a failed fatal
expectation in the immutable result record and motivates TRAF-23's measured
frontier validation.

## Physical sanity

Three independent checks were applied before accepting the mechanism-level
interpretation.

First, compute and memory physics give a 69.206 microsecond floor from
553,648,128 resident weight and LM-head bytes at 8 TB/s. The 0.7-efficiency
roofline predicts roughly 99.3 microseconds. Observed TPOT was 99.52
microseconds on the single node and 99.85 microseconds cross-node, above the
floor and close to the active compute term.

Second, network serialization gives 3.33 microseconds for the representative
1.5 MB peak-rank decode egress at 450 GB/s, and 30 microseconds at 400 Gbit/s.
The serial-to-observed reductions were 3.05 and 27.45 microseconds. Their
8.999 ratio matches the ninefold byte-rate ratio to within 0.02 percent. For
prefill, 27.1 MB through a 400 Gbit/s rank requires at least 542 microseconds;
the observed cross-node TTFT was 696.57 microseconds. The values are inside
the frozen conservative decode ceiling of 332 microseconds and the applicable
prefill serialization floor.

Third, the observed TPOTs imply about 10,048 and 10,015 tokens per second for
`r0`. As a broad external plausibility reference, NVIDIA reports more than
1,000 tokens per second per user for a much larger 400-billion-parameter
Llama 4 Maverick deployment on eight B200 GPUs in its
[Blackwell inference report](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/).
The comparison does not calibrate this 400-million-active-parameter Granite
model, but it does not make the roughly 10,000 token/s ideal-model result
physically impossible. Absolute accuracy is not claimed: the current run uses
an analytic roofline and coarse collective service, and TRAF-11, TRAF-14 and
TRAF-23 retain their calibration gaps.

## Exact serial off path

With observations absent, `DeviceRuntimeStepSink` delegated to
`SerialStepLowerer`. Both placements passed all 32 per-step direct comparisons
of graph JSON, operation timestamps, completion-event order, execution result,
`StepResult` and request metrics. The accepted fixture remained:

| Artifact | Bytes | SHA-256 | Outcome |
|---|---:|---|---|
| Canonical execution graph JSON plus LF | 4,127 | `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d` | Passed |
| Legacy direct diagnostic GOAL | 1,880 | `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6` | Passed |

The active graph path now projects that fixture into the already accepted six
causal artifacts, with three required boundaries and 12 other serialized
edges. The 1,880-byte form is retained as the independent legacy direct
renderer diagnostic. This distinction reconciles the frozen wording with the
repository's current graph-authoritative GOAL contract. It does not change
either byte identity.

## Entailment and genuine-risk accounting

The six raw mode workers completed before `_evaluate_scored` compared the two
TPOT reductions and two perturbation increases. Producer inventory, source
searches, physical bounds, graph identities, serial fixture hashes and the
other fatal guards were interpreted afterward. Direct serial checks and the
live cross-node replay comparison pin only repeatability of their own raw
runs; they contain no frozen metric value and cannot entail a signed band.
No exact check pins the perturbed metric.

Each scored instance could fail after reaching execution. A serial and
observed result outside either band would fail the first family. Ignoring the
observation tuple or the added dependency would produce a zero perturbation
and fail the second family. The two observed nonzero perturbations therefore
remain genuine-risk evidence even though their minimums failed.

The 14 fatal guards are unscored. Several are conservation identities,
configuration-forced facts or fixed hashes. The bandwidth-scaling guard is a
secondary relation over already scored raw reductions and is not counted
again. The producer counts, request-pair rows, live identities, serial
identities, physical bounds and no-knob search do not increase the behavioral
denominator.

| Scored family | Executed | Passed | Fraction |
|---|---:|---:|---:|
| Serial minus observed TPOT band | 2 | 2 | 100% |
| Dependency perturbation direction plus minimum | 2 | 0 | 0% |
| Aggregate genuine-risk result | 4 | 2 | 50% |

## VLLM-22 closure map

The registered VLLM-22 clauses map as follows.

> "add a real source-backed vLLM `ExecutionObservations` producer for each
> translated step."

The real vLLM v0.26.0 replay reached the `SimWorker` model-forward boundary
and emitted observations for all 32 of 32 nonempty translated steps. The
explicit `off` mode emitted `None` and retained the legacy coordinator call.

> "observe all 24 ordered layers and exactly 48 semantic MoE sites, one
> dispatch and one combine per layer, with exact submission order, logical
> streams, program-order and event-wait dependencies, request correlation,
> and completion frontier."

Every source inventory row covered 24 layers and 48 unique sites. The 23 DBO
rows carried 96 invocations across two disjoint request slices. Operation
counts, tuple order, both shared queues, both dependency scopes, correlation
and one or two request-visible endpoints are reported above. All 128 lowering
preservation checks passed.

> "Name the active source mechanism that makes every claimed concurrency
> legal, and derive no edge from an overlap percentage or compatibility
> schedule."

The mechanism is vLLM DBO's cooperative model threads, shared compute and
communication streams, and DeepEP CUDA event waits. The audit cites each
source location before the freeze. The producer starts from the translated
step and active vLLM configuration and contains no overlap knob or serial
graph rewrite.

> "Extend the per-request regression to the adapter-emitted schedule, proving
> that traffic rebinding preserves every request-pair byte and that completion
> reduction returns the correct request identities."

All six cells retained the same 121,646 request-pair rows and 440,115,200
bytes, with exact per-step identities. Completion reduction returned the
original request IDs.

> "With the producer absent, preserve the legacy sink call, serial graph and
> GOAL bytes, timestamps, and completion order exactly."

The focused adapter regression preserves the legacy call. Sixty-four
per-step direct serial comparisons and both accepted fixture hashes passed.
VLLM-22 therefore closes. VLLM-23 records only schedule shapes intentionally
rejected outside the frozen TP1, PP1, uniform-decode DBO boundary.

## TRAF-13 closure map

The registered TRAF-13 clauses map as follows.

> "connect at least one real framework schedule producer to
> `ObservedStepLowerer`"

The live vLLM producer passed observations through `DeviceRuntimeStepSink` and
`ObservedStepLowerer` on every nonempty step.

> "Replay a fixed captured step through the traffic binding, `DeviceRuntime`,
> `CompletionEvent`, `StepResult`, TTFT and TPOT; require every captured order
> and dependency fact to survive exactly"

The frozen capture ran end to end on both placements. The live cross-node
identity, 128 schedule-field checks, request-pair identities, 32-step metric
streams and completion frontiers passed.

> "and show that one observed legal overlap changes the live metric in its
> registered direction."

Observed TPOT was below serial TPOT on both placements and passed both
registered reduction bands. The added dependency increased all 46 applicable
placement-step results, so a lowering that ignored observations cannot pass.
The self-registered perturbation minimums failed and move to TRAF-23 as
quantitative precision rather than weakening this direction evidence.

> "Disabling the producer must select the serial lowerer and preserve every
> accepted serial graph, GOAL byte, timestamp and completion order exactly."

The explicit off path selected direct serial lowering. All 64 Granite
per-step comparisons and the accepted fixture passed. TRAF-13 therefore
closes at this framework, model and schedule boundary.

## Residuals and deliberate limits

- VLLM-23 (Completeness; P2; L) owns TP greater than one, PP greater than one,
  explicit `ubatch_size`, padded DBO and multi-token microbatch correlation.
  These shapes fail explicitly today. Their refusal is the identity off path.
- TRAF-23 (Precision; P1; L) owns measured per-rank DeepEP event frontiers and
  perturbation magnitude. It retains both failed frozen minimums and the
  refuted TTFT equality assumption as calibration evidence.
- TRAF-11 still owns the flat NVLink rate, TRAF-14 owns immutable physical
  collective expansion, and VLLM-12 owns general device-schedule templates.

VLLM-24 and CORE-38 are deliberately unused. No general vLLM model family,
parallel shape, GPU-present worker, calibrated DeepEP timing or packet-level
overlap claim is made.

## Verification evidence

The post-implementation registered command passed with `--check-only` and
created no artifact. Before the result run, full-tree Ruff passed and the
Python suite passed 1,039 tests with 7 skips. The final closure tree repeated
both full gates with the same outcome. Focused adapter, traffic, step-lowering
and device-runtime regressions passed 131 tests. No C++ source, native
executable or submodule pin changed.

## Integrator-owned contradiction sweep

`README.md` and `docs/architecture.md` contain no stale VLLM-22 or TRAF-13
state. `docs/README_PRO.md` contains one prose hit in the fidelity-level table:
it says the observed framework schedule is landed under TRAF-7 but that the
live producer is TRAF-13. The generated progress block and module open counts
are mechanically reconciled for ledger CI, but that prose row is left for the
integrator. Historical expectations and results retain their original
chronology and are not rewritten.
