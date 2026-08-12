# vLLM observed schedule v1 results

VLLM-22 and TRAF-13 remain open at the frozen Granite vLLM v0.26.0 boundary.
A real eight-rank vLLM replay emitted
`ExecutionObservations` for all 32 nonempty scheduler steps. Every step carried
all 24 ordered MoE layers and 48 unique semantic collective sites. DBO steps
carried 96 collective invocations across two request-correlated microbatches.
The observations reached traffic rebinding, `CoarseDeviceRuntime`, completion
events, request-attributed `StepResult`, TTFT and TPOT.

The run is **void with findings** because one fatal unscored guard,
`ttft_exact_single_batch`, was violated. The raw rows still show two
serial-to-observed TPOT reductions inside their frozen bands and positive TPOT
movement under both dependency perturbations, but no behavioral pass fraction
is interpretable after the fatal violation. The evidence is retained and both
tasks remain open.

The failed guard tested the premise that the serial and observed arms differ
only by DBO. That same premise is required to attribute the two in-band TPOT
reductions to DBO, so the violation is not orthogonal to the behavioral result.
The retained DBO-off decode control below bounds the non-DBO structural
residual at about 1.2 percent of the mean reduction on both placements. This
makes the confound quantitatively small, but it does not make the void run
scorable.

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
| Behavioral relations | 4 | Raw findings retained; no score because the run is void |
| Fatal unscored guard set | 1 violation | `ttft_exact_single_batch`; run void |
| Live framework ranks | 8 | All exited successfully |
| Live nonempty scheduler steps | 32 | 32 emitted observations |
| Direct serial per-step comparisons | 64 | 64 graph, runtime, event, result and metric identities passed |
| Fixed serial compatibility fixture | 1 | Both frozen byte identities passed |
| Full Python regression | 1 retained result invocation | 1,039 passed, 7 skipped |
| Repository lint | 1 retained result invocation | Passed |

Source hashes, fixed configuration values, byte conservation, serial
identities and the absence of an overlap knob are fatal or change-set guards.
They are unscored. Unit tests and the live framework run remain separate from
the four behavioral relations. The fatal guard set is not reported as a pass
fraction.

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
participant-local edges represented the shared streams and the event waits
inferred from the audited vLLM wrapper. Each microbatch's next layer waited on
its own participant-local combine frontier. Final per-rank logits waited on
both final combines, and each request-visible endpoint waited on every rank's
logits.

The source paths cited before the freeze in
[the pinned-source audit](expectations.md#pinned-source-audit) establish
vLLM's two cooperative DBO model threads, shared compute and communication
streams, wrapper-level high-throughput yields, and CUDA event waits. The
audited vLLM wrapper contains no rank-global barrier at that seam, but the
`deep_ep` implementation itself was not installed or read. Participant-local
DeepEP completion is therefore an inference from wrapper behavior, not a
directly source-backed DeepEP semantic claim. Neither the producer nor the
lowerer contains an overlap percentage, duration discount, random choice, or
reconstruction from the serial compatibility graph.

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

Across the 32 steps, each of the six cells contained 121,646 request-pair rows
and 440,115,200 directed bytes. The byte figure is a pre-TRAF-25 conservation
identity over the known source-multiplied routed table. It is inflated by
`_routed_moe_alltoalls` sourcing every token from all eight EP ranks and is not
portable across the pending TRAF-25 fix. Every per-step request-pair identity
was the same in serial, observed and perturbed modes and on both placements.
`StepResult` retained the original `r0`, `r1` and `r2` identities while those
requests remained active. These are fatal unscored conservation and identity
checks, not overlap evidence.

The collective duration model does not scale from total bytes. In
`classify_communication_phase`, each rank's `source_egress_bytes` is divided by
bandwidth and the phase uses the maximum source service. Under TRAF-25,
per-layer dispatch egress from the owning rank is unchanged while combine
collapses, so the communication term changes by roughly a factor of two, not
by the factor of eight by which total bytes change. Both retained reduction
bands scale with that communication term and are therefore not portable across
TRAF-25.

`_validate_microbatch_partition` only proves that microbatch tables recombine
to the full table produced by the same traffic planner. It conserves the
inflated table against itself and is structurally incapable of detecting the
source-ownership defect, so it is not a byte-correctness guard. The adapter
emits zero-byte semantic all-to-allv markers, and
`_validate_observed_collective` permits those markers and empty pair tables
before replacing them with planned work. The observed path consequently
cross-checks no independently observed routed-MoE byte count. VLLM-24 records
that validation gap.

## Raw TPOT relations

`r0` TPOT is the exact mean of its 23 decode intervals. The runner wrote all
six raw step streams and evaluated the four relations before interpreting any
later fatal guard.

| Placement | Serial TPOT (ps) | Observed TPOT (ps) | Serial minus observed (ps) | Frozen band (ps) | Retained finding |
|---|---:|---:|---:|---:|---|
| `single-node` | 102,567,669.435 | 99,517,385.696 | 3,050,283.739 | 1,000,000 to 5,000,000 | Inside band |
| `cross-node` | 127,300,132.174 | 99,851,905.565 | 27,448,226.609 | 20,000,000 to 130,000,000 | Inside band |

The observed schedule reduced TPOT by 2.97 percent of serial on the
single-node placement and 21.56 percent on the cross-node placement. The
cross-node reduction was 8.999 times the single-node reduction, which is above
the frozen fivefold threshold. These raw reductions are pre-TRAF-25 and are not
portable because both scale with the communication term affected by the
pending token-ownership fix.

## DBO-off control from retained raw rows

Steps 24 through 31 are single-request decode steps for which the frozen vLLM
configuration forces DBO off. Their serial-minus-observed latency is therefore
a direct control for the non-DBO structural difference between the two arms.
The 23 earlier decode steps used by `r0` TPOT all selected DBO.

| Placement | DBO-off residual range (ps) | Percent of control step | Mean residual (ps) | DBO reduction range (ps) | Percent of DBO step | Mean DBO reduction (ps) | Mean residual / reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single-node` | 31,864 to 40,968 | 0.03% to 0.04% | 37,554 | 2,608,046 to 3,996,106 | 2.55% to 3.86% | 3,050,283.739 | 1.231% |
| `cross-node` | 286,720 to 368,640 | 0.26% to 0.33% | 337,920 | 23,470,080 to 35,962,880 | 19.05% to 26.45% | 27,448,226.609 | 1.231% |

The placement-invariant 1.231 percent ratio bounds the measured non-DBO
confound at about 1.2 percent of the mean reduction whose bands were evaluated.
It is the quantitative reason to describe the guard failure as a small
incidental residual rather than evidence that the DBO mechanism is mis-scaled.
It does not restore the failed premise or make the run non-void.

## Dependency perturbation finding

The perturbation added exactly one whole-operation dependency on each of the
23 DBO steps, from microbatch zero's layer-12 combine to `rank-0`'s
microbatch-one layer-12 expert operation. This is one rank's expert compute
endpoint only, not all eight ranks. No work, byte, queue, tuple position,
correlation or completion endpoint changed.

| Placement | Observed TPOT (ps) | Perturbed TPOT (ps) | Mean increase (ps) | Per-step increase range (ps) | Frozen minimum (ps) | Retained finding |
|---|---:|---:|---:|---:|---:|---|
| `single-node` | 99,517,385.696 | 99,549,249.696 | 31,864.000 | 31,864 | 100,000 | Positive; below minimum |
| `cross-node` | 99,851,905.565 | 100,138,704.826 | 286,799.261 | 286,720 to 288,292 | 5,000,000 | Positive; below minimum |

A lowerer that ignored the observations would have produced zero movement, so
the positive directions remain useful findings. The magnitude criteria were
unreachable for the perturbation that was actually implemented:

| Placement | Increase / same-cell reduction | One of 96 invocations | Frozen minimum / reduction |
|---|---:|---:|---:|
| `single-node` | 1.0446% | 1.0417% | 3.2784% |
| `cross-node` | 1.0449% | 1.0417% | 18.2161% |

One added dependency edge can delay a step by at most one targeted
collective's service time. For this uniform 96-invocation DBO shape, removing
one overlap therefore costs about one ninety-sixth of the total reduction.
The measured 1.045 percent on both placements matches that 1.042 percent
ceiling, while the frozen minima demanded 3.3 and 18.2 percent. The mechanism
is direction-right; the magnitude expectation was mis-specified.

The frozen wording named "microbatch one's layer-12 expert compute" without
stating a rank. The study chose the defensible singular interpretation and
gated only `rank-0`. This implementation detail is outcome-relevant. Gating all
eight expert operations would be of order eight times larger and could
plausibly have cleared the single-node minimum, although shared critical-path
effects prevent treating eightfold scaling as exact. TRAF-23 refiles future
frontier criteria against the measured per-collective service ceiling rather
than a whole-step perturbation minimum.

## TTFT finding

The frozen expectation that a non-DBO prefill must equal the serial
compatibility path was wrong:

| Placement | Serial TTFT (ps) | Observed TTFT (ps) | Serial minus observed (ps) | Exact expectation |
|---|---:|---:|---:|---|
| `single-node` | 167,543,635 | 164,630,939 | 2,912,696 | Failed |
| `cross-node` | 713,203,520 | 696,566,212 | 16,637,308 | Failed |

DBO was disabled on prefill in both arms, but participant-local dependency
scope is not the difference. `SerialStepLowerer` uses each rank's tail for the
next-layer compute and for both EP collectives, just as the observed producer
uses participant-local dependencies around pre-dispatch, dispatch, expert and
combine work. Collective operations expose all ranks as participants, so both
arms receive per-rank frontiers. In a direct two-layer serial lowering, the
only whole-operation edges are logical-queue FIFO edges.

The actual structural mismatch is the open TRAF-9 approximation plus the
observed arm's terminal operations. The serial arm models one whole-layer
compute followed by its collectives, with dispatch and combine back to back.
The observed arm splits pre-dispatch and expert compute around those
collectives, then adds per-rank logits and a `requests-visible`
whole-operation fan-in from every rank's logits. The retained raw rows do not
separate which of those structural differences contributes each picosecond of
the TTFT residual. TRAF-9 remains the owner of the layer-ordering approximation;
TRAF-23 is limited to measured completion frontiers.

The vLLM wrapper files cited in the frozen audit show CUDA event waits and no
wrapper-level rank-global barrier. The `deep_ep` implementation itself was not
installed, so the absence of a barrier below the wrapper is inferred rather
than directly source-backed. The earlier claim of source-backed rank-local
DeepEP completion semantics was too strong.

This guard was not merely an unrelated TTFT check. It tested whether the two
arms differed only by DBO, which is also the premise needed to attribute the
in-band decode reductions to DBO. The DBO-off control bounds the measured
structural residual at about 1.2 percent of the mean reduction, but the fatal
violation still makes the run void and keeps VLLM-22 and TRAF-13 open.

## Physical sanity

Three independent checks were recorded before interpreting the mechanism-level
findings. They do not override the fatal guard.

First, compute and memory physics give a 69.206 microsecond floor from
553,648,128 resident weight and LM-head bytes at 8 TB/s. The 0.7-efficiency
roofline predicts roughly 99.3 microseconds. Observed TPOT was 99.52
microseconds on the single node and 99.85 microseconds cross-node, above the
floor and close to the active compute term.

Second, the pre-TRAF-25 routed table gives 3.33 microseconds for its 1.5 MB
peak-rank decode egress at 450 GB/s, and 30 microseconds at 400 Gbit/s. The
serial-to-observed reductions were 3.05 and 27.45 microseconds, and their 8.999
ratio matches the ninefold byte-rate ratio to within 0.02 percent. The same
pre-fix table gives 27.1 MB of peak-rank prefill egress, whose 400 Gbit/s
serialization floor is 542 microseconds; observed cross-node TTFT was 696.57
microseconds. These checks establish internal physical consistency with the
source-multiplied table only. The 1.5 MB and 27.1 MB figures, their floors and
both reduction bands are not portable across TRAF-25.

Third, the observed TPOTs imply about 10,048 and 10,015 tokens per second for
`r0`. As a broad external plausibility reference, NVIDIA reports more than
1,000 tokens per second per user for a much larger 400-billion-parameter
Llama 4 Maverick deployment on eight B200 GPUs in its
[Blackwell inference report](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/).
The comparison does not calibrate this 400-million-active-parameter Granite
model, but it does not make the roughly 10,000 token/s ideal-model result
physically impossible. Absolute accuracy is not claimed: the run is void, the
routed traffic is pre-TRAF-25, the compute path uses an analytic roofline, and
TRAF-11, TRAF-14 and TRAF-23 retain their calibration gaps.

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

## Entailment and void accounting

The six raw mode workers completed before `_evaluate_scored` compared the two
TPOT reductions and two perturbation increases. Producer inventory, source
searches, physical bounds, graph identities, serial fixture hashes and the
other fatal guards were interpreted afterward. Direct serial checks and the
live cross-node replay comparison pin only repeatability of their own raw
runs; they contain no frozen metric value and cannot entail a signed band.
No exact check pins the perturbed metric.

Each behavioral instance could fail after reaching execution. A serial and
observed result outside either band would miss the first family. Ignoring the
observation tuple or the added dependency would produce zero perturbation and
miss the second family. Those properties establish genuine risk in how the raw
relations were evaluated; they do not survive a fatal guard as a behavioral
score.

The fatal guards are unscored. Several are conservation identities,
configuration-forced facts or fixed hashes. The bandwidth-scaling guard is a
secondary relation over the raw reductions and is not counted again. The
producer counts, request-pair rows, live identities, serial identities,
physical bounds and no-knob search do not increase a behavioral denominator.

| Frozen behavioral family | Instances evaluated | Retained raw finding |
|---|---:|---|
| Serial minus observed TPOT band | 2 | Both reductions were inside their bands |
| Dependency perturbation direction plus minimum | 2 | Both were positive and both were below their frozen minima |

The `ttft_exact_single_batch` violation voids the run, so neither a family pass
fraction nor an aggregate behavioral percentage is reported.

## VLLM-22 evidence map

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

The vLLM wrapper source establishes DBO's cooperative model threads, shared
compute and communication streams, CUDA event waits and wrapper-level absence
of a rank-global barrier. The audit cites those locations before the freeze.
The separate `deep_ep` implementation was not installed, so no lower-level
completion semantic is claimed as directly source-backed. The producer starts
from the translated step and active vLLM configuration and contains no overlap
knob or serial graph rewrite.

> "Extend the per-request regression to the adapter-emitted schedule, proving
> that traffic rebinding preserves every request-pair byte and that completion
> reduction returns the correct request identities."

All six cells retained the same 121,646 request-pair rows and the same
440,115,200 pre-TRAF-25 directed bytes, and completion reduction returned the
original request IDs. The byte identity is not independent evidence: the
microbatch checker conserves the planner's inflated table against itself, and
the adapter's zero-byte semantic markers bypass adapter-to-plan routed-byte
agreement. This clause is therefore not demonstrated as a byte-correctness
claim. VLLM-24 owns the missing independent cross-check.

> "With the producer absent, preserve the legacy sink call, serial graph and
> GOAL bytes, timestamps, and completion order exactly."

The focused adapter regression preserves the legacy call. Sixty-four
per-step direct serial comparisons and both accepted fixture hashes passed.
VLLM-22 remains open because the run is void and its routed-byte clause lacks
an independent pre-TRAF-25 oracle. VLLM-23 records schedule shapes
intentionally rejected outside the frozen TP1, PP1, uniform-decode DBO
boundary.

## TRAF-13 evidence map

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

Observed TPOT was below serial TPOT on both placements and both reductions
were inside their registered bands. The added dependency increased all 46
applicable placement-step results, so a lowering that ignored observations
cannot produce these raw findings. The frozen perturbation minima were
mis-specified for one targeted edge. TRAF-23 refiles future criteria against a
measured per-collective service ceiling.

> "Disabling the producer must select the serial lowerer and preserve every
> accepted serial graph, GOAL byte, timestamp and completion order exactly."

The explicit off path selected direct serial lowering. All 64 Granite
per-step comparisons and the accepted fixture passed. TRAF-13 remains open
because the fatal TTFT guard voids this qualification.

## Residuals and deliberate limits

- VLLM-23 (Completeness; P2; L) owns TP greater than one, PP greater than one,
  explicit `ubatch_size`, padded DBO and multi-token microbatch correlation.
  These shapes fail explicitly today. Their refusal is the identity off path.
- VLLM-22 (Completeness; P0; L) and TRAF-13 (Completeness; P0; L) remain open
  with this void evidence package and its DBO-off decomposition.
- VLLM-24 (Precision; P0; M) owns the missing independent adapter-to-plan
  routed-byte cross-check after TRAF-25.
- TRAF-23 (Precision; P1; L) owns measured per-rank completion frontiers. A
  future one-edge criterion must be bounded by one measured collective service
  time.
- TRAF-9 remains the owner of the serial whole-layer MoE ordering
  approximation.
- TRAF-11 still owns the flat NVLink rate, TRAF-14 owns immutable physical
  collective expansion, and VLLM-12 owns general device-schedule templates.

CORE-38 is deliberately unused. No general vLLM model family, parallel shape,
GPU-present worker, calibrated DeepEP timing or packet-level overlap claim is
made.

## Verification evidence

The post-implementation registered command passed with `--check-only` and
created no artifact. Before the result run, full-tree Ruff passed and the
Python suite passed 1,039 tests with 7 skips. The retained result tree repeated
both full gates with the same outcome. Focused adapter, traffic, step-lowering
and device-runtime regressions passed 131 tests. No C++ source, native
executable or submodule pin changed.

This record-only correction did not rerun or rescore the study. It repeated
full-tree Ruff and the Python suite, which again passed with 1,039 tests and 7
skips, and verified that the task-progress block and module open counts are
current.

## Integrator-owned contradiction sweep

`README.md` and `docs/architecture.md` contain no stale VLLM-22 or TRAF-13
state. `docs/README_PRO.md` contains one prose hit in the fidelity-level table:
it says the observed framework schedule is landed under TRAF-7 but that the
live producer is TRAF-13. The generated progress block and module open counts
are mechanically reconciled for ledger CI, but that prose row is left for the
integrator. Historical expectations and results retain their original
chronology and are not rewritten.
