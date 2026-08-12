# Compute and communication overlap v1 expectations

These expectations are frozen before the TRAF-7 schedule-aware lowering,
focused tests, study implementation, or first result-producing run. The study
starts from the accepted CORE-4 resource executor and changes only the graph
shape presented to that executor. No overlap percentage, fitted overlap
constant, or elapsed-phase discount is an input.

## Decision and source audit

The audited repository state is commit `6973bd0`. Before this freeze:

- `simllm/core/execution.py:201-220` defines graph operations on framework-level
  FIFO logical queues with explicit whole-operation and participant-local
  dependency edges. `simllm/core/execution.py:223-255` says tuple order carries
  observed submission order and that different queues may proceed without an
  edge.
- `simllm/backends/step_lowerer.py:105-139` accepts standard
  `ExecutionObservations`, but its no-observation path remains the serial
  compatibility schedule. `simllm/backends/step_lowerer.py:141-270` lowers one
  whole-layer compute per rank followed by both tensor-parallel collectives and
  makes the next layer wait for the collective tail.
- `simllm/traffic/step_comm.py:72-90` is the tensor-parallel payload authority.
  It emits attention and MLP ring all-reduces for every layer.
  `simllm/traffic/step_comm.py:326-452` is the accepted serial GOAL identity
  path and explicitly chains the next layer's calc to the preceding
  collective frontier.
- `simllm/core/runtime.py:924-1005` releases only dependency-ready operations,
  `simllm/core/runtime.py:1140-1220` applies logical-queue FIFO and both edge
  kinds, and `simllm/core/runtime.py:1641-1803` expands semantic ring work into
  NCCL-channel visits and semantic sends. `simllm/core/runtime.py:1805-1832`
  owns the per-rank, per-channel contention cursor.
- `simllm/core/completion.py:256-393` is the live reduction from
  `ExecutionGraph`, `ExecutionResult`, and `RuntimeReport` into `StepResult`,
  TTFT, and exact rational TPOT.
- `examples/core4_runtime/RESULTS.md:74-96` records the accepted component
  baseline: independent compute and DMA matched `max(C, D)`, while an edge
  matched `C + D`. This study does not rescore that old fixture. It applies the
  same executor to step-derived ring collectives and then traverses the live
  reducer.
- `simllm/compute/nccl.py:1-15` and `docs/modules/compute.md:491-500` show that a
  GPU-resident NCCL egress model exists but is not yet the collective service
  used by `CoarseDeviceRuntime`. `docs/modules/compute.md:523-551` keeps real
  NCCL software, proxy, ingress, GPU-initiated, and runtime projection work
  open under COMP-15.

The fixed schedules below are synthetic conformance fixtures. They do not
mirror a claimed vLLM, SGLang, NCCL, or silicon trace, so they carry no
external-system timing expectation. Future framework adapters must supply
their observed operation order, logical queues, and dependency edges. The
traffic lowering may validate the semantic collective site and bind the ring
algorithm and bytes, but it may not invent the framework schedule.

The design decision is whether the step model admits an explicit
observation-driven graph path while retaining serial compatibility as the
off path. If the independent, serial, or pipeline relation below fails, that
path is rejected and the existing serial schedule remains the only accepted
step lowering. A tuned overlap fraction is not an admissible fallback.

## Fixed workload and exact service terms

Each step schedules one request and has two layers, tensor-parallel ranks
`(0, 8)`, one token, hidden size 262,144, and two-byte activations. Every
attention or MLP all-reduce therefore carries 524,288 bytes. Ranks 0 and 8
are on distinct nodes in the accepted fixed profile. At exactly 400 Gbit/s,
a two-rank ring has two rounds of 262,144 bytes per all-reduce, so one
all-reduce takes exactly 10,485,760 ps. Four all-reduces give the fixed total

```text
D = 4 * 10,485,760 ps = 41,943,040 ps.
```

All ideal-family launch, NCCL-channel, completion-delivery, and propagation
constants are zero. Compute carries zero HBM bytes so the accepted independent
GPU and RNIC resources are intentionally isolated. Sweep total compute `C` at
`20,971,520 ps` and `83,886,080 ps`, crossing `D` from `C/D = 1/2` to
`C/D = 2`. Each layer carries `C/2` on each rank.

The adapter-authored schedule shapes are:

1. `independent`: each rank's two compute operations are FIFO on its compute
   queue, and all four collectives are FIFO on the communication queue. There
   is no cross-queue edge. Completion waits for both queue tails.
2. `serial`: layer 0 compute releases its attention and MLP collectives; layer
   1 compute waits for the layer 0 MLP completion; layer 1 collectives wait for
   layer 1 compute. This is the strict compatibility order.
3. `pipeline`: layer 0 compute releases layer 0 communication and layer 1
   compute independently. Layer 1 communication waits for layer 1 compute and
   for communication-queue FIFO. This is a two-stage pipeline shape, not a
   measured claim about a specific engine.

For `c = C/2` and `d = D/2`, the exact pipeline closed form is

```text
pipeline(C, D) = c + max(c, d) + d.
```

The frozen step-latency values are:

| C (ps) | C/D | Independent (ps) | Pipeline (ps) | Serial (ps) |
|---:|---:|---:|---:|---:|
| 20,971,520 | 1/2 | 41,943,040 | 52,428,800 | 62,914,560 |
| 83,886,080 | 2 | 83,886,080 | 104,857,600 | 125,829,120 |

## A. Decision-relevant dependency-shape relations

For each compute ratio, collect all three raw runtime results before applying
an exact literal check. The raw observations must satisfy:

```text
independent = max(C, D)
serial = C + D
pipeline = C/2 + max(C/2, D/2) + D/2
independent < pipeline < serial
```

The independent-to-pipeline and pipeline-to-serial gaps are both 10,485,760 ps
when `C/D = 1/2`, and both 20,971,520 ps when `C/D = 2`. These eight exact-form
instances plus the two strict-between instances are scored behavioral
relations. The six fixed table cells are then checked as a separate exact
oracle class.

This family can fail under a competent implementation. A lowerer can add a
cross-stream edge accidentally, omit the framework completion frontier, use
operation order as a global serializer, or release a participant before its
rank-local predecessor. Crossing the compute-to-communication ratio prevents
one accidentally dominant resource from hiding those defects.

## B. Live TTFT and TPOT relation

For each ratio and schedule shape, run three consecutive copies of the fixed
step through one `CoarseDeviceRuntime`, its `CompletionEvent` stream,
`CompletionReducer`, and `VirtualClock`. Step 0 is prefill and steps 1 and 2
are decode. Every step samples the same request once. With no scheduler gap,
the first step's TTFT and each later exact rational TPOT must equal that
shape's raw step latency.

For both ratios, changing only `serial` to `pipeline` must reduce TTFT and TPOT
to exactly `5/6` of the serial value. The signed reductions are 10,485,760 ps
at `C/D = 1/2` and 20,971,520 ps at `C/D = 2`. The four signed TTFT/TPOT
instances are scored. Per-step exact TTFT and TPOT literals remain a separate
exact-oracle class.

Decision consequence: if graph JCT changes but TTFT or TPOT does not change by
the same signed amount, the lowering is not live-reachable and TRAF-7 cannot
close.

## C. First contention mechanism, NCCL channel occupancy

Use two dependency-independent two-rank ring operations on distinct logical
queues, each with a two-byte payload, an 8 Tbit/s RNIC rate, zero launch and
delivery service, and 1,000 ps of NCCL channel service per ring round. One
payload-byte ring chunk serializes in exactly 1 ps.

Run one graph with both operations on physical NCCL channel `shared`, and one
with channels `a` and `b`. In the shared-channel graph the second operation's
first channel visit starts at 2,001 ps, exactly when the first operation's last
channel visit releases. Its JCT is 4,003 ps. In the split-channel graph both
operations' first channel visits start at 0 ps; RNIC contention still leaves a
JCT of 3,004 ps. Changing only channel identity therefore reduces JCT by
exactly 999 ps.

The raw shared-versus-split start and JCT differences are scored before their
fixed literals are checked. Queue timestamp order and resource identity are
fatal structural guards. This family can fail if logical queue independence
is ignored, the NCCL cursor is node-global instead of rank-and-channel local,
or the channel cursor is bypassed while WQEs still happen to serialize.

This is the only newly isolated contention family in this study. The ideal
family still includes real coarse RNIC serialization, while all other coarse
service constants are zero.

## D. Serial identity off path

The opt-in observation-aware lowerer must delegate an absent-observation call
to the existing `SerialStepLowerer` without reconstructing it. The accepted
two-layer B1 compatibility fixture uses `FlopProvider`, TP ranks `(0, 1)`, and
the absent exact-sample-count record from `tests/test_step_lowerer.py`. Its
canonical compact sorted execution-graph JSON plus one LF is 4,127 bytes with
SHA-256 `aa3c836fe559973a7bf0940384c2e8a84e6af84e0fbd2c02d3b89774ee0c8e2d`.
Its serial graph-only GOAL text is 1,880 bytes with the already accepted
SHA-256 `7087db6780f7e34f5a559a6505eeccc15d984c7b478cd8f0bc5838053825d4b6`.

Both bytes and hashes must remain exact. The existing dense, TP=1, MoE,
captured-routing, and legacy GOAL regression tests must also remain green.
These are identity and change-set guards, fatal but unscored. They do not
increase a behavioral pass denominator.

## Fatal structural guards

The following fail the study but remain unscored:

- observation tuple order, operation IDs, logical queues, explicit dependency
  edges, timing gates, priorities, completion IDs, and compute work are
  preserved exactly by lowering;
- every planned tensor-parallel and MoE collective appears exactly once, with
  no missing, duplicate, unknown-site, wrong-group, wrong-payload, or
  wrong-layer observation;
- traffic lowering selects ring all-reduce and pairwise all-to-allv work and
  preserves routed pair bytes and placement epoch;
- graph validation rejects cycles, unknown dependencies, and an operation that
  names one predecessor through both edge modes;
- no public configuration or result field contains an overlap percentage or
  overlap duration discount;
- completion events, graph boundaries, and request metric attribution conserve
  the runtime result exactly;
- check-only mode creates no result directory or artifact;
- no test requires an initialized `third_party` path.

Author-defined operation sequences and the fixed zero-overhead profile are
configuration guards, not scored behavior.

## Entailment and evidence accounting

The runner must evaluate every scored relation against raw observations before
checking any fixed literal that entails the same relation. The later exact
oracles are useful regression pins, but they do not make the earlier relation
family genuine-risk evidence retroactively. `RESULTS.md` must state the order
actually used and report genuine-risk fractions separately for families A, B,
and C. Structural guards, identity hashes, configuration echoes, conservation,
and by-construction zeros are fatal and unscored.

The report keeps these evidence classes separate:

- run configurations;
- exact-oracle rows;
- scored behavioral relation families and parameterized instances;
- fatal structural and identity guards;
- focused Python tests;
- the full repository regression suite.

Counts from different classes are never added into one headline total.

## Scope and honesty boundary

This freeze covers the generic traffic-to-graph binding, explicit serial off
path, coarse GPU-versus-RNIC overlap, NCCL channel FIFO, RNIC service, the live
completion chain, and exact TTFT/TPOT effects. It does not claim that a current
vLLM or SGLang adapter emits the studied schedule. VLLM-19 and SGL-10 already
own framework runtime projection and captured stream/event schedules.

It also does not claim GPU-resident NCCL kernel occupancy, shared SM or HBM
contention between compute and collective kernels, copy-engine or GPUDirect
DMA contention, calibrated channel service, proxy progress, ingress and
reduction work, or packet-level overlap. CORE-11, CORE-13, COMP-11, COMP-12,
and COMP-15 already own those mechanisms. The TRAF-7 closure report must quote
its original clauses, map only demonstrated clauses to evidence, and register
any uncovered traffic or core acceptance clause under a new pre-allocated ID
rather than broadening the claim.

## Registered command and pre-freeze dry run

The registered result-producing command from the repository root is:

```bash
.venv/bin/python examples/compute_comm_overlap_v1/run_study.py \
  --output-dir "${SIMLLM_OVERLAP_RUN_ROOT:?configure SIMLLM_OVERLAP_RUN_ROOT}"
```

Before this freeze, the same command is run with `--check-only`. Check-only
parses the complete CLI and validates only the frozen integer tables, formulas,
hash shapes, and evidence-family names. It imports no target implementation,
executes no graph or runtime behavior, and creates no output directory or
artifact. It prints one confirmation line by design.
