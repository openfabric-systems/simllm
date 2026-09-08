# Receiver contention reaches request latency

The coarse receiver study completes all **204 candidate configurations** and
their 612 request steps without a fatal or numerical finding. Eight sources
each sending 65536 bytes to one 400 Gbit/s receiver now require **10.48576
microseconds**, exactly the receiver's byte-service floor. The source-only
baseline reports 1.31072 microseconds. Both time to first token (TTFT) and
time per output token (TPOT) reflect the eightfold correction; the three-step
job completion time (JCT) is 31.45728 microseconds.

The consumer evidence and full gates close CORE-48 for the declared coarse
joint-reservation model. A supported `CoarseDeviceRuntime(receiver_ingress=True)`
path now carries receiver contention through completion events and step results
to request metrics. All 102 disabled snapshots remain byte-identical to the
pre-change runtime. The default remains the explicit source-only compatibility
model. This result does not calibrate packet queues or deployment token speed,
close CORE-8, BACK-38 or TRAF-8, or change CORE-41's intra-node service and
HTSIM-40/41's packet-recovery evidence.

## One transmission, two occupied ports

A source can offer its next transfer only after its previous reservation
finishes. The receiver can accept that transfer only after its own previous
reservation finishes. The model grants one interval during which both ports
carry the same bytes. It charges that interval once to request latency.
Opposite send and receive directions have separate cursors, so full-duplex
traffic remains independent.

For source eligibility e, source offer t, joint grant s, release f and
visibility delay D:

```text
d = ceil(payload_bytes * 8 * 10^12 / rate_bps)
t = max(e, source_available)
s = max(t, receiver_available)
f = s + d
source_available = receiver_available = f
completion = f + D
```

The existing `AtlahsWqeLedger` owns both cursors in the same transaction.
`ReceiverIngressWqeProjection` extends the immutable work queue entry (WQE)
projection only when receiver serialization is enabled. The legacy record
keeps exactly its prior serialized fields. Source and receiver byte ledgers
and port reservations are read-only projections joined by WQE identity.

| Reported stage | Eligibility | Grant | Release | Bytes | Additive contribution |
|---|---|---|---|---:|---|
| Source send-queue admission | e | t | t | 0 | Source ordering t-e |
| Receiver admission and joint transmission | t | s | f | B | Receiver wait s-t, service f-s |
| Completion visibility | f | | f+D | B once at completion | D |

Each enabled WQE has nine checked lifecycle events across the two admission
stages and its one completion queue. Source and destination ports, send and
receive queues, completion queue and queue pair are bound to their declared
node and GPU rail. A consistently mislabeled projection is rejected before
any runtime or bookkeeping state commits. Missing or duplicated receiver
events also cause rollback, with the next valid execution matching a fresh
runtime.

This is an indivisible coarse reservation with receiver backpressure and
deterministic submission order. A blocked transfer delays later transfers
from its source. Earlier reservations keep their order even if a later
submission has earlier eligibility. The model makes no claim of independent
buffered egress, packet interleaving, work-conserving arbitration or calibrated
switch-buffer behavior. Structural native RNIC selection rejects an additional
coarse receiver authority. Intra-node work keeps its existing path.

## Bounds before observations

The main sweep varies payload over 3, 4096 and 65536 bytes, fan-in over 1,
2, 4 and 8, and link rate over 200 and 400 Gbit/s. All ranks are on separate
nodes. Every configuration executes one prefill and two decode steps with
the same network work and zero compute. These are controlled network-isolation
fixtures, not predictions of a real model's token throughput.

Before observations, the enabled phase floor is the larger of the total
payload sent by one source and received by one destination, divided by link
rate. Ring predecessor rounds strengthen this bound. For a finite isolated
fixture, serializing every extent and every declared channel stage supplies
a conservative ceiling. Under external contention the unconditional ceiling
is unbounded.

For eight 65536-byte arrivals, 524288 bytes over 400 Gbit/s gives a floor
and isolated star ceiling of 10.48576 microseconds. At 200 Gbit/s both become
20.97152 microseconds. The observed enabled values equal these bounds.
The four-rank complete all-to-all with 4096-byte extents has a 0.24576
microsecond capacity floor and 0.98304 microsecond fully serial ceiling; its
declared reservation order predicts and produces 0.40960 microseconds.

The ring's per-send bytes are q=max(1,floor(B/W)), with chunk service d_q.
For four ranks, 4096 payload bytes, 400G and 7000 ps channel service, six
rounds require 0.12288 microseconds of wire service and 0.042 microseconds of
channel service. The measured 0.16488 microseconds equals their sum. The
accepted three-byte sentinel keeps TTFT=TPOT=120 ps at 400G and 240 ps at
200G. These tiny fixtures exercise exact arithmetic under the inherited
coarse assumptions; they include no physical propagation or packet-header cost.

## Measured receiver scaling

All times below are microseconds. Payload is 65536 bytes per source.
Each TTFT equals TPOT in these identical-step fixtures.

| Sources | Source-only TTFT, 400G | Receiver TTFT, 400G | Receiver TTFT, 200G | Receiver JCT, 400G |
|---|---:|---:|---:|---:|
| 1 | 1.31072 | 1.31072 | 2.62144 | 3.93216 |
| 2 | 1.31072 | 2.62144 | 5.24288 | 7.86432 |
| 4 | 1.31072 | 5.24288 | 10.48576 | 15.72864 |
| 8 | 1.31072 | 10.48576 | 20.97152 | 31.45728 |

At fan-in eight and 400G, the selected receiver wait is 9.17504 microseconds
and the final wire service is 1.31072 microseconds. Their sum is 10.48576.
The sum of waits over all receiver visits is 36.70016 microseconds, larger
than elapsed time because several transfers wait together. It remains a
separate work-accounting total and never enters an additive TTFT or TPOT.

Halving bandwidth doubles the measured serialization-bound metric. Raising
payload from 4096 to 65536 bytes multiplies it by sixteen. Doubling fan-in
doubles combine and dispatch duration; disjoint duplex pairs retain one
transfer duration. The 114 nonzero scaling instances comprise 18 receiver
latency increases, 36 bandwidth relations, 24 payload relations and 36 fan-in
relations. Every instance holds. Independent review confirms the same
relations in TPOT and three-step JCT; these supporting checks do not multiply
the behavioral count.

Dispatch, duplex, the sixteen accepted ring cases, both sentinels and the
remaining unchanged fixtures provide 78 enabled timing identities. The
asymmetric three-byte plus five-byte combine changes from 100 to 160 ps at
400G and from 200 to 320 ps at 200G; dispatch remains 160/320 ps. The complete
four-rank source-major all-to-all changes from 3*d to 5*d. Its old source-only
time was not a protected physical timing result; the frozen contract states
this correction explicitly. No traffic plan is reordered to retain a time.

## Evidence and actual chronology

The mechanism, parameter grid and primary scaling relations were frozen in
expectations-only commit `5de4db26abb64188e27b56c3e3700588cc0df03d`, before the
first implementation edit. Two later clarifications are disclosed in
[expectations.md](expectations.md), with no history rewrite:

- `4226cb3ec398d8536c8f4ff61ce7cedf1715a865` clarifies ring chunk notation and
  the sentinel sentence after the first implementation edit, before any unit
  or consumer execution. These explicit rewritten assertions are
  post-specified regression checks.
- `859af6a03e8929c715e97573f9b97290055c920c` defines enabled workload-graph
  identity after initial unit execution, before the first consumer run.
  It sets only `released_at_ps` to zero for the comparison, because subsequent
  step releases follow the changed preceding completion. Every other graph
  field and every GOAL byte remain exact. Disabled historical snapshots receive
  no normalization. This projection is a post-specified regression check.

The immutable baseline runtime is `429acf0fbd733006c8978d00cf18c0090584dc86`.
The candidate executes source `0f6c63c` after implementation `d4c88d5` and
endpoint-identity hardening `f90ca91`. Exact full commits, runner and source
hashes, input hashes, raw-result hashes and the final expectation digest are
in [results.json](results.json). Raw observations precede exact-oracle guards.
Bulk snapshots retain the full reports, WQEs, events, bookkeeping and step
results; the published JSON is a compact projection of those records.

The primary evidence consists of 102 immutable baseline executions and 204
candidate executions, or 918 request steps. There are 102 full disabled
snapshot comparisons and 204 candidate exact fixture/request-timing oracle rows.
Byte conservation, directional interval nonoverlap, causal ordering, source
and receiver identity, event joins and graph identity are fatal unscored guards.
No fatal guard is violated. Configuration counts, exact oracles, unit test
cases and the four behavioral families are not added into one score.

Before the first candidate run, independent review required a stricter
baseline loader: a missing row could previously skip its identity comparison.
The runner now requires the exact complete population, each declared case,
its mode and three steps, clean outcome, matching runner and expectations,
and the actual canonical snapshot digest. A first 102-case baseline capture
is retained separately. The hardened runner repeats it in a fresh directory;
all 102 complete canonical snapshots compare byte for byte. This additional
baseline capture is identity evidence outside the primary 306 executions.

Independent review reconstructs all 2088 enabled WQE reservations from raw
snapshots without importing the runtime or harness. It confirms both directional
port capacities, exact bytes, no receiver service before source offer, one
wire-service charge, bookkeeping joins and request causality. All 102 disabled
snapshots also compare directly against both baseline captures. No numerical
or integrity finding remains.

## Reproduction and gates

Use an immutable checkout at the baseline commit and configure its location
as `SIMLLM_RECEIVER_BASELINE_SOURCE` in ignored local configuration. Configure
`SIMLLM_DATA_ROOT` as an external artifact directory. The runner's Python
environment must import the intended checkout:

```sh
PYTHONPATH="$SIMLLM_RECEIVER_BASELINE_SOURCE" python \
  examples/receiver_ingress_v1/run_study.py --mode baseline \
  --out "$SIMLLM_DATA_ROOT/receiver_baseline"
PYTHONPATH=. python examples/receiver_ingress_v1/run_study.py --mode candidate \
  --baseline "$SIMLLM_DATA_ROOT/receiver_baseline" \
  --out "$SIMLLM_DATA_ROOT/receiver_candidate"
```

Both output directories must be new. A changed runner requires a fresh
baseline capture. The runtime source must be committed before execution.
Input bounds are saved before measurements; incomplete runs have null request
metrics and cannot be reconstructed from completed subsets.

The final full Python suite passes on candidate source `0f6c63c`:

```text
4745 passed, 29 skipped in 1261.67s (0:21:01)
```

The focused runtime gate passes 238 tests, including 94 new receiver cases
and 144 retained cases. The independent study-harness gate passes 22 tests,
including malformed and incomplete baseline rejection. Documentation format,
path portability, task progress and commit identity pass their 29-test gate.
`ruff check .`, `scripts/check_docs_format.py` and
`scripts/task_progress.py --check` also pass. These test populations overlap
the full suite and are not additional behavioral evidence.

An earlier full-suite invocation was interrupted before endpoint-identity
hardening; its retained log is not a gate. The completed full suite above
uses the final committed runtime and harness. The inherited packet-model
binary remains byte-identical to the separately validated HTSIM-41 binary;
the publication pin also includes its test-portability repair.

CORE-48 closes for this executable coarse model, including receiver-bound
TTFT/TPOT effects and exact compatibility. CORE-8 retains general cross-layer
queue contracts and calibration. BACK-38 and TRAF-8 retain physical serving
integration; this CPU-only study supplies no calibrated packet or GPU timing.
