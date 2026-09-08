# Pipeline queue service and receiver holding

The 96-execution packet study is **valid and refutes the all-depth request
penalty hypothesis**. Expert traffic occupies pipeline output queues at every
tested depth, but the two-stage request gains **0 ps** of latency. Four- and
eight-stage requests gain about 2.54 milliseconds because lost pipeline packets
wait for recovery and ordered delivery. These are different observed mechanisms.
Every fatal guard holds, all 24 trace on/off pairs are exact, and all 48
historical completion-CSV controls retain their accepted bytes.

TRAF-88 stays open under its frozen all-instance acceptance rule. Its loss,
trace and topology-dependent receiver-window prerequisites are complete; the
remaining qualification must distinguish the demonstrated holding and recovery
regimes before generalizing a pipeline contention penalty. The original
[contention study](../pp_rail_contention_v1/RESULTS.md) remains void. This result
does not close CORE-8's cross-layer authority contract, BACK-38's persistent
physical execution or TRAF-8's captured serving integration, and supplies no
time-per-output-token or hardware calibration result.

## Physical scope and bounds

One declared request computes for one microsecond at each of two, four or eight
stages and sends a 65536-byte activation between successive stages. Separate
network interfaces exchange expert-parallel (EP) traffic at widths 0, 8 or 32.
The pipeline-parallel (PP) interfaces attach by rail or by node. Each of eight
leaves has eight or two 400 Gbit/s uplinks; each link adds one microsecond of
propagation. The source graphs, placement, endpoint mapping and GOAL inputs
retain their predecessor identities. This is a forward request with fixed
synthetic stage service, not a measurement of GPU execution or serving speedup.

The following bounds were written and emitted before execution:

- A 65536-byte payload needs at least 1.31072 microseconds at 400 Gbit/s.
  Including propagation, a PP hop needs at least 3.31072 microseconds on a rail
  path or 5.31072 microseconds on a node-local path. The request floor is
  `P*1 us + (P-1)*hop_floor`. At depths 2, 4 and 8 the node-local floors are
  7.31072, 19.93216 and 45.17504 microseconds.
- There is no unconditional finite flow-completion ceiling with finite buffers
  and retries. The separate engineering budget is
  `9*S_work + 8*(40 us + 70.99776 us)`, where `S_work` is the largest declared
  receiver, cut or pipeline service floor. Its metric is the complete flow
  phase, from first flow start to last delivery, not request latency or drain.
- At EP width 32, the eight-spine EP phase floor is 589.20256 microseconds.
  Two-spine floors are 1.17840512 milliseconds for node-local attachment and
  1.76560768 milliseconds for rail attachment. Their complete-phase engineering
  budgets are 11.45762816 and 16.7424512 milliseconds, respectively.

Every physical observation stays above its applicable byte and causal floors
and below its frozen complete-phase budget. The audit also checks the complete
physical phase against its identical ideal-profile phase and checks cumulative
bytes for each receiver's earliest completions. Shared-flow ratios and phases
with model-dependent starts are diagnostic; an individual physical/ideal flow
ratio is not used as a lower bound there.

The main receiver window is fixed at 10.5664 microseconds on both topologies.
An on-time packet becomes eligible at expected arrival plus that window,
rounded up to 16-nanosecond ticks. Later arrival inside the same release window
reduces receiver holding. A late admitted retry instead uses actual-arrival
release. Receiver serialization and ordered delivery then select the actual
completion boundary. Control headroom and exponential DATA recovery are
selected explicitly; initial-send budgets remain disabled.

## Request outcome and measured queue work

All times in this table are microseconds. Queue work sums EP DATA service ahead
of every physical PP packet visit, including retries. Its visits can overlap,
so this sum is not elapsed time or an additive request penalty. In particular,
the two-stage sum exceeds the entire two-stage request duration.

| Node-local, two spines | Unloaded TTFT | EP-width-32 TTFT | TTFT increase | EP service-ahead work | Independent EP work bound |
|---|---:|---:|---:|---:|---:|
| 2 stages | 18.249 | 18.249 | 0 | 29.92708 | 29.92708 |
| 4 stages | 53.404 | 2594.085 | 2540.681 | 305.05204 | 305.05204 |
| 8 stages | 122.969 | 2663.033 | 2540.064 | 304.37484 | 304.29164 |

Time to first token (TTFT) is the final PP stage's completion event projected
through the existing singleton step result. Stage compute, actual hop flow
completion times (FCTs), GOAL gates and final nanosecond truncation reconstruct
it exactly. The reported hop maximum uses only 1, 3 or 7 hops; it is not a
statistical latency-tail estimate. At depth two, EP width eight already adds
0.95076 microseconds of queue work while leaving TTFT unchanged.

The independent EP bound counts strictly earlier queued DATA bytes at one
physical output, at 20 ps per wire byte, plus only the residual service of an
already active EP packet. The eight-stage exact work exceeds that conservative
subset by one 4160-byte packet's 0.0832-microsecond service. Same-time arrivals
are excluded from the subset; their actual service remains in the separately
reconstructed interval sum. Every PP switch wait equals the exact partitioned
service intersections on its output.

The rail control has zero EP service ahead at every tested depth. All 16, 48
and 112 original PP packets preserve source transmission, expected and actual
arrival, receiver release, receiver service and delivery under EP widths zero
and 32. TTFT is exactly 16.082, 46.329 and 107.237 microseconds at depths 2, 4
and 8, even as the corresponding background phases reach 6.4945728,
6.5184832 and 6.5950944 milliseconds.

## Why two stages absorb delay

On the two-spine, two-stage node-local path, every original PP packet retains
its source transmission, expected arrival, receiver release, receiver service
and delivery under EP width 32. Packet index 14, lifecycle 1622, arrives
4.10876 microseconds later. Its receiver holding falls from 10.57212 to
6.46336 microseconds, exactly the same change, while release remains at
17.072 microseconds. Its delivery time is unchanged.

The selected receiver predecessor chain confirms the mechanism: its on-time
release inputs use expected arrival plus the fixed window. Actual arrival does
not become a selected release dependency in this case. This is evidence of
real fabric contention with an exactly zero request effect, not absence of
contention and not a reason to lower the window after observing the result.

## Why deeper requests wait for recovery

The second hop, semantic ranks 8 to 16, dominates both deeper requests. In the
four-stage case it takes 2557.2572 microseconds. Packet index 14 loses its
original and attempts 1 through 6 in finite switch storage. Independent raw
queue rows confirm all seven drops. One retry even encounters an empty target
output but cannot fit in the shared pool: 1044928 bytes are already occupied,
leaving less than its 4160-byte wire frame.

The receiver's negative acknowledgement authorizes retry 1 at 38.86336
microseconds; transmission begins at 40.0192 microseconds. The intervening
1.15584 microseconds is an observed scheduling interval whose internal source
eligibility remains opaque. Attempts 2 through 7 start exactly at the preceding
transmission end plus 40, 80, 160, 320, 640 and 1280 microseconds. Those six
observed timeout intervals sum to **2.52 milliseconds**.

Retry 7, lifecycle 1217876, starts at 2560.5184 microseconds and completes
receiver service at 2575.5072 microseconds. It arrives only 640 ps after its
fresh expected arrival, waits just 640 ps in switch queues, and uses the
on-time expected-arrival release branch. The original final packet, index 15,
has already finished receiver service at 34.8192 microseconds. Ordered delivery
holds it another 2540.688 microseconds until index 14 closes the hole. The final
logical packet and the packet that triggers completion are therefore distinct.

The eight-stage slow hop takes 2556.6332 microseconds. Its completion trigger is
index 11, retry 7, lifecycle 1215955; its switch wait is 0.1696 microseconds.
Its original and attempts 1 through 6 are also fabric-dropped. The six probe
windows again sum to 2.52 milliseconds, with separately observed scheduling
intervals of 0.4928 microseconds after the first negative acknowledgement and
0.0832 microseconds after retry 3's authorization.
The final logical packet is index 15, retry 3, which has already completed
receiver service at 178.2272 microseconds and waits for ordered delivery until
2574.8832 microseconds. That final index-15 packet is late-admitted and uses
actual-arrival release. The completion-triggering index-11 retry 7 instead uses
fresh expected-arrival release; its raw arrival is not a selected critical input.

These witnesses establish loss recovery and ordered delivery on the request
path. They do not turn a trigger-packet timeline into a complete resource
causal graph, or assign the summed EP queue work as a marginal TTFT cost.
Source eligibility and recursively selected competing switch dependencies remain
outside the trace's scope. A future direct-arrival penalty must survive the
receiver and delivery dependency chain; recovery can instead shift source
transmission and expected arrival, as it does here.

## Window and cut comparisons

All four unloaded low/high window pairs obey the rounded release rule.
Increasing the window from 6.5728 to 10.5664 microseconds leaves source timing
and actual arrival unchanged. Individual releases increase by 3.984 or
4.000 microseconds. After receiver serialization, every hop FCT increases by
exactly 3.9936 microseconds. Final GOAL truncation makes the TTFT increase
3.993 microseconds for rail attachment and 3.994 microseconds for node-local
attachment. At a fixed window, two and eight spines give identical unloaded
PP timing, separating the predecessor's control-window effect from contention.

| EP width 32, two stages | Eight-spine EP phase (ms) | Two-spine EP phase (ms) | Observed ratio | Two-spine physical floor (ms) | Complete-phase budget (ms) |
|---|---:|---:|---:|---:|---:|
| Rail | 0.8951072 | 6.4945728 | 7.255637 | 1.76560768 | 16.7424512 |
| Node-local | 0.9117152 | 3.6848192 | 4.041634 | 1.17840512 | 11.45762816 |

The bare directional cut service changes by exactly four when the spine count
falls from eight to two. Observed phases include endpoint service and recovery,
so the freeze does not predict a fourfold completion ratio. Both observed
ratios are interpretable diagnostics. The deeper loaded node-local EP phases
are 3.7189312 and 3.7130464 milliseconds, inside the same two-spine budget.
Physical quiescence follows delivery separately and is identical with tracing
enabled or disabled in every pair.

## Frozen chronology, evidence classes and reproduction

The [expectations-only contract](expectations.md), commit
`d13a593ce3f347f6b3ed4b499d170411d609f39e`, precedes the implementation and
first new native execution. The executed simLLM source is
`ed5ad4ce84f147e2f40a2b500695573c3042e0ea`; backend source is
`3593ccbf77d9e7f9e553e2a1ac1a5ae01d8c5c0a`. Its Release binary SHA-256 is
`a0722e32822da8b02461dc057d94a4fe4a4a4cbcb8b9d15354f7a487f06bb491`.
The publication pin `dd0a343c52f03a0af75d441afe5a6e75a93a2ddf` adds only the
native trace documentation. No parameter, hypothesis or earlier result changes
after execution. Publication adds a compact file projection, not a new run.

The evidence classes remain separate:

- Configurations: 24 physical configurations executed with tracing off and on,
  12 legacy physical controls and 36 ideal controls, totaling 96 executions.
- Fatal unscored checks: all 24 trace-selection identities and 48 historical
  completion-CSV identities are exact. All 72 input sets retain their locked
  identities. Packet conservation, queue service, receiver release, request
  causality, capacity floors, completion budgets and quiescence hold.
- Behavioral relations: all four R1 receiver-window instances hold; all three
  R2 positive-service instances hold; R3 is refuted at depth two and holds at
  depths four and eight. Thus its all-instance hypothesis is refuted.
- Diagnostics: the two R4 cut-response ratios are reported without a behavioral
  score. Native executables and Python unit cases are separate software gates.

The full native suite passes 501 tests. The unchanged legacy gate passes eight
plans containing 95 experiments. The Python suite reports 4915 passed and
29 skipped; the new-binary wrapper has 70 passing cases, and the focused study
and trace-corruption checks have 170. These counts are not combined with the
behavioral study populations.

[results.json](results.json) retains every configuration metric, exact oracle,
behavioral relation and diagnostic. Explicit audit projections keep flow
completion witnesses, selected trigger-attempt chains, packet outcome counts
and queue-work totals. Complete packet/egress inventories remain in external
`trace_audit.json` files bound by SHA-256, alongside raw CSVs, manifests, logs
and the full result's hash. Every execution retains raw FCT in `completion.csv`;
physical-profile executions additionally retain `per_flow_normalization.csv`.

Configure `SIMLLM_HTSIM_RNIC`, `SIMLLM_HTSIM_SOURCE` and `SIMLLM_TXT2BIN` for the
native build. Set `SIMLLM_PP_REFERENCE` to the retained predecessor evidence and
`SIMLLM_STUDY_OUT` to a fresh directory outside the checkout. From committed
source, run:

```sh
PYTHONPATH=. python -m examples.pp_rail_contention_v2.run_study \
  --reference "$SIMLLM_PP_REFERENCE" --out "$SIMLLM_STUDY_OUT" --workers 4
python examples/pp_rail_contention_v2/publish_results.py \
  --source "$SIMLLM_STUDY_OUT/results.json" \
  --out "$SIMLLM_STUDY_OUT/public-results.json"
```

The publisher preserves acceptance and checks each complete audit against the
full result before projecting it. It refuses to overwrite an existing output.
The physical request-penalty qualification remains owned by TRAF-88; no serving
milestone advances through this synthetic forward-only study.
