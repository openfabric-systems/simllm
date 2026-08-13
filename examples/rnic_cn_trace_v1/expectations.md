# HTSIM-2 rnic-cn trace flag expectations

This is the expectations-only record for HTSIM-2. It precedes every backend
implementation change and every result-producing command of this study.

HTSIM-2 registers "goodput/state/queue trace flags for `rnic-cn`; they need
trace hooks in the reviewed runtime first". The object under test is
observation, not timing. A trace flag that changed a modeled number would be a
defect, so the acceptance shape is deliberately the opposite of a precision
refinement: the accepted `rnic-cn` baseline must survive byte for byte, and
each new trace must agree exactly with an authority that already exists.

This study makes no TTFT or TPOT claim and does not recalibrate any mechanism.

## Source audit, authored against a recorded commit

The evidence is authored against HTSIM commit
`fc4400e4ca619223481536632074045cb6af2756`. The result report records the
commit each run actually observed. It does not assert that a live SimLLM
submodule pin equals this literal.

The audit found that two of the three trace mechanisms already exist as
shared, profile-neutral components, and that the third has a hook but no
consumer:

- `htsim/sim/datacenter/atlahs_goodput_trace.h` already bins delivered payload
  bytes by flow and writes
  `bin_start_ps,bin_end_ps,flow_id,source,destination,delivered_payload_bytes,goodput_bps`.
  Only the DCQCN runtime records into it.
- `htsim/sim/datacenter/atlahs_state_trace.h` already carries sparse sender
  state as
  `time_ps,flow_id,source,destination,event,configured_rate_bps,effective_rate_bps,alpha,paused,new_packets_sent,rtx_packets_sent,acked_packets`,
  and its own comment says it is shared by the physical comparison profiles.
  Only the DCQCN runtime appends to it.
- `htsim/sim/datacenter/ns_tm3_switch.h` already publishes `NsTm3QueueObserver`
  and the exact `NsTm3QueueObservation` boundary record. No CSV consumer
  exists for it in any profile.

So the work is to record into the two shared traces from the reviewed
`rnic-cn` runtime, to add the missing queue-trace consumer as a third shared
component in the same shape, and to expose all three behind `rnic-cn` CLI
flags. No new parallel interface is introduced. `alpha` and `paused` stay
empty for `rnic-cn`: they are DCQCN policy state, and the shared header
forbids inventing samples to fill a column.

## Frozen trace schemas

The queue trace is new and its header is frozen here:

```text
time_ps,transition,switch_tier,switch_id,ingress_id,egress_id,priority,
flow_id,packet_id,packet_bytes,egress_buffered_bytes,
egress_in_service_bytes,egress_backlog_bytes,shared_buffer_occupancy_bytes
```

written as one physical line. The two existing headers are unchanged, so the
DCQCN artifacts keep their exact bytes.

## Frozen flag surface

```text
-rnic_cn_goodput_trace_csv FILE   -rnic_cn_goodput_trace_bin_ps PS
-rnic_cn_state_trace_csv FILE
-rnic_cn_queue_trace_csv FILE     -rnic_cn_queue_trace_max_rows N
```

Every flag is off by default. The goodput pair must be supplied together, as
the DCQCN gate already requires of its own pair. The queue row cap exists
because an event-driven queue trace grows with packets times hops times
transitions; exceeding it fails the run explicitly rather than truncating
silently. All five are rejected for the profiles that cannot produce them,
matching the existing `-rnic_nn_propagation_ps is valid only for NN profiles`
rule.

## Fixed fixture and sweep

One GOAL, eight ranks, rank `r` sending 262144 bytes to rank `(r + 1) % 8`
with no dependency edges, so no rank can block another. Eight physical
endpoints satisfy the generated two-tier ns-tm3 Clos predicate. The seed and
every other `rnic-cn` option keep their defaults.

Two parameters vary:

| parameter | values |
|---|---|
| `-linkspeed_bps` | 400000000000, 200000000000 |
| `-rnic_cn_goodput_trace_bin_ps` | 1000000, 500000 |

## Physical sanity, stated before any measurement

- Floor: a receiver cannot absorb more than its link rate. In one goodput bin
  of width `W` picoseconds at `C` bits per second, no destination may show
  more than `W * C / 8 / 1e12` bytes. A bin above that ceiling proves a
  misassigned bin or a wrong time base.
- Floor: the run makespan cannot beat the busiest endpoint's own
  serialization. Each rank sends 262144 bytes, so at 400 Gbit/s that is at
  least 5.24 microseconds of egress, and at 200 Gbit/s at least 10.49
  microseconds, before any header, control packet, hop latency or queueing.
- Ceiling and scaling: halving the link rate doubles every serialization term
  while hop latency and propagation stay constant, and per-flow control
  overhead is additive rather than proportional, so the makespan ratio must
  land below two and not far below it. The frozen band is `[1.5, 2.05]`. A
  ratio near 1.05 or above 2.5 refutes the model regardless of any exact
  match elsewhere.
- Plausibility against the real system: 2 MiB moved across eight 400 Gbit/s
  endpoints should complete in tens of microseconds. A result in nanoseconds
  or in milliseconds is wrong whatever the internal consistency.

## F1: accepted baseline preserved, fatal and unscored

With every trace flag absent, the changed `htsim_rnic` must reproduce the
pre-change binary exactly for the same GOAL and options: byte-identical
completion CSV, and an identical set of `[RNIC manifest]` lines. The harness
runs a separately built baseline binary from the audited commit for this
comparison.

With the trace flags present, the completion CSV and the manifest lines other
than the new trace manifest line must still be byte-identical to the same
run with the flags absent. Observation must not perturb the model.

This is a by-construction bypass guard. It is fatal when violated and never
scored, and a violation voids the run for the purpose of closing HTSIM-2.

## F2: conservation identities, fatal and unscored

- Per flow, the sum of `delivered_payload_bytes` across all goodput bins
  equals that flow's `payload_bytes` in the completion CSV.
- Per switch egress port, the count of `Enqueued` queue-trace rows equals the
  count of `Dequeued` plus `Dropped` rows, and the final
  `egress_buffered_bytes` observed at that port is zero.
- The state trace contains at least one row per flow and no row for a flow
  absent from the completion CSV.

Conservation and closure identities. Fatal when violated, never scored.

## S1: goodput bin bracket against the completion authority

For every flow, the last goodput bin that carries any of its payload must
contain that flow's `completion_time_ps` from the completion CSV:

```text
bin_start_ps <= completion_time_ps < bin_end_ps
```

Genuine risk. The trace is binned on receiver delivery time while the
completion record is produced by a different code path, and a trace that
records at transmit, at acceptance or at retirement instead of at in-order
delivery release fails this while still conserving bytes. Instances: one per
flow, eight at each of the two link rates, sixteen scored instances.

## S2: goodput never exceeds the receiver link

No goodput row may report `delivered_payload_bytes` greater than
`bin_width_ps * linkspeed_bps / 8 / 1e12`, and none may report a
`goodput_bps` greater than `linkspeed_bps`. Genuine risk: an off-by-one bin
boundary, a wrong bin width or a double-counted retransmission all break it.
Instances: one per (link rate, bin width) configuration, four scored
instances.

## S3: bin refinement conserves and refines

Halving `-rnic_cn_goodput_trace_bin_ps` at a fixed link rate must leave every
per-flow total unchanged and must not decrease the number of distinct bin
start times. Genuine risk: a trace that binned on a cached or rounded time
would drift. Instances: one per link rate, two scored instances.

## S4: queue trace is rate-invariant in count and rate-sensitive in depth

The packet population and the routes do not depend on the link rate, so at a
fixed GOAL and seed:

- the number of `Enqueued` queue-trace rows must be exactly equal at
  400 Gbit/s and at 200 Gbit/s;
- the maximum `egress_backlog_bytes` over all rows at 200 Gbit/s must be
  greater than or equal to the maximum at 400 Gbit/s.

Genuine risk: an observer attached after arbitration, or one that misses
drops, breaks the equality; a backlog read at the wrong event boundary breaks
the ordering. Two scored instances.

## S5: state trace rates follow the configured link

Between the 400 Gbit/s and the 200 Gbit/s run, at a fixed GOAL and seed:

- every `configured_rate_bps` value must halve exactly, because the field
  carries the endpoint access wire capacity;
- every `effective_rate_bps` value `r` observed at 200 Gbit/s and its
  counterpart `R` at 400 Gbit/s must satisfy `2 * r <= R <= 2 * r + 3`. The
  slack of three is the worst case of the two nested integer floors in the
  margin derating and the nflow fraction, and nothing else.

Genuine risk: a state trace that recorded a stale snapshot, the shared ledger
rate rather than the flow's own rate, or a rate captured before activation
fails one of the two. Two scored instances.

## S6: makespan scaling

The makespan, taken as the maximum `completion_time_ps` over the completion
CSV, must grow by a factor inside `[1.5, 2.05]` when the link rate halves, and
each run's makespan must exceed the serialization floor stated above. One
scored instance.

S1 through S6 give twenty-seven scored instances. Planned genuine-risk
fraction: `27/27`.

### Entailment analysis

Every scored relation is evaluated against raw parsed rows before any other
check reads them. F1 constrains only that the traced run equals the untraced
run; it says nothing about what the traces contain, so it cannot entail S1
through S5. F2 pins per-flow byte totals and port-level counts; totals do not
determine which bin a delivery lands in, so S1 and S3 stay open, and F2 says
nothing about backlog depth or rate values. S2's ceiling is looser by orders
of magnitude than S1's bracket and neither implies the other. S4's count
equality and S5's rate relations compare two independent runs, which no
single-run identity pins. The remaining genuine risk is the specific one this
task exists to retire: a trace hook placed at the wrong point in the reviewed
runtime.

## Closure scope

HTSIM-2 registers these clauses.

| registered acceptance clause | frozen evidence |
|---|---|
| "goodput ... trace flags for `rnic-cn`" | The flag pair, F2 conservation, S1, S2 and S3. |
| "state ... trace flags for `rnic-cn`" | The flag, F2 presence, and S5. |
| "queue trace flags for `rnic-cn`" | The flag and cap, F2 port conservation, and S4. |
| "they need trace hooks in the reviewed runtime first" | The recording points added to the reviewed `rnic-cn` runtime, exercised by the complete backend ctest suite plus F1. |

Any clause the run does not demonstrate moves to `HTSIM-27` with a
categorized priority and difficulty tag, quoting the clause it failed. An idea
for later work that no registered clause claimed is recorded in prose instead.

A note on the registered category rather than a new identifier: HTSIM-2 is
tagged Precision, but a trace flag is observation and must not move a metric.
The correct reading of the "metrics must be live-reachable" rule here is that
the traces are read-only projections of the runtime authority, and their
acceptance is exact agreement with existing authorities plus untouched
baseline bytes.

## Registered commands and check-only dry run

Bulk outputs remain outside Git. The registered command is:

```bash
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_RNIC:?configure the traced htsim_rnic}" \
SIMLLM_HTSIM_RNIC_BASELINE="${SIMLLM_HTSIM_RNIC_BASELINE:?configure the pre-change htsim_rnic}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the txt2bin executable}" \
.venv/bin/python examples/rnic_cn_trace_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/rnic_cn_trace_v1"
```

The same command with `--check-only` must run before this expectations commit.
Check-only validates all three executables, the fixture, the frozen sweep, the
frozen flag spellings and the external output placement. It prints the plan and
creates no artifacts. The harness present at freeze time encodes only these
frozen literals, orchestration and check-only validation. It contains no
backend change and no observed outcome.
