# Live peer packet runtime result

The declared packet path reaches request time to first token (TTFT) and time
per output token (TPOT) with **zero picosecond residual across 136 exact timing
checks**. One routed request runs a prefill and two decode steps through the
original execution graph. All 36 instances in four timing-relation families
match the frozen expectations, and no fatal guard is violated.

BACK-48 and TRAF-45 close. The common packet vocabulary now serves both native
wire and GPU peer observations, and local physical service reaches the reported
metric chain. COMP-40 retains host-port emission. This advances the local
communication part of M4; it does not close the large-model deployment frontier.
No GPU, model weights or hardware allocation was used.

## Numerical result

Every extent carries 1,024 payload bytes in four declared 272-byte wire packets.
The source and attachment rate is 25 GB/s. Receiver ingress is 12.5 or 25 GB/s.
One or three donor GPUs dispatch and combine through direct links or a switch
with actual shared attachments. The fixed 37,000 ps compute term is a synthetic
control. It is not a prediction for any model kernel.

| Donors | Receiver GB/s | Route | TTFT ps | TPOT ps | Three-step completion ps |
|---|---|---|---:|---:|---:|
| 1 | 25 | Direct | 147800 | 147800 | 443400 |
| 1 | 25 | Switched | 171560 | 171560 | 514680 |
| 1 | 12.5 | Direct | 234840 | 234840 | 704520 |
| 1 | 12.5 | Switched | 258600 | 258600 | 775800 |
| 3 | 25 | Direct | 321880 | 321880 | 965640 |
| 3 | 25 | Switched | 345640 | 345640 | 1036920 |
| 3 | 12.5 | Direct | 495960 | 495960 | 1487880 |
| 3 | 12.5 | Switched | 519720 | 519720 | 1559160 |

The independent floors and ceilings are recorded before executing each cell.
For three donors at 25 GB/s, the common receiver must accept 3,264 wire bytes,
which takes at least 130,560 ps. A direct first packet adds 11,880 ps of
transport, so the combine floor is 142,440 ps. The observed combine equals
that floor. The two-phase service is 284,880 ps, below the 594,240 ps
conservative ceiling that serializes every packet through every stage and
bounded return. The compute control then gives exactly 321,880 ps per token.

Three independent checks constrain that result:

- Link bytes and propagation determine the first arrival and attachment floors.
  A switched route adds exactly 23,760 ps to the two-phase request service.
- Receiver bytes and finite ownership constrain converging service. Halving
  receiver bandwidth adds 87,040 ps with one donor and 174,080 ps with three.
  Shared switch input storage and shared receiver capacity never multiply with
  virtual queues or the number of attachments.
- Original-graph operation and extent events agree with the returned step and
  published request metrics. Compute is identical in both arms. Packet minus
  analytic token latency equals packet communication minus the accepted
  6,000 or 14,000 ps analytic service, with no fitted coefficient.

## Ownership and retained state

A source feed and its selected attachment acquire one coupled grant. Their
release times differ when the source is faster than the link. Switch input,
output and output attachment likewise share a grant. Completed `QueueVisit`
records retain submission, eligibility, grant, release and visibility; partial
packet/path observations and buffer claims retain unfinished visits. Their
resource sums overlap and are never added wholesale to TTFT or TPOT.

The one-credit control separates visibility from retirement. Its first packet
is visible at 22,760 ps. With 100,000 ps additional credit-return processing,
the next packet starts at 123,760 ps and becomes visible at 146,520 ps. The
200,000 ps acknowledgement processing tail remains on the same calendar, which
finally drains at 336,640 ps. Both physical hop acknowledgements are retained
for switched packets. Observing an idle interval does not inflate drain time.

Native packet `Delivered` means transport retirement; it need not mean consumer
visibility. Native `PacketTxStarted` may mean producer issue. Immutable context
retains those boundaries without changing the native wire grammar. GPU logical
completion projects actual destination visibility, and its parent transport
terminal follows every child terminal. Packet identities stay in the sidecar;
only logical extent and operation events cross the graph completion interface.

## Compatibility and evidence classes

The accepted `nvlink_locality_v1` and `mixed_attribution_v1` matrices execute
at baseline `4b7042681a8b5d942fc21f96dd9786f887203a46` and the new source with
packet selection absent. Their canonical artifact inventories contain 2,593
and 1,129 entries respectively, all byte-identical. Separately compiled native
producers emit identical full version 1 and version 2 observation artifacts.
The native model's nine existing test executables also pass.

The only excluded JSON fields are explicitly listed source and platform/build
provenance. Every modeled value and semantic identifier remains in the
comparison. Before either compatibility run, the mixed study's separate host
stopwatch input is fixed to zero; its `wall_seconds` field stays present and
identical. This comparison makes no host wall-performance claim. Raw generated
files, canonical bundles, exclusion pointers and digests remain separate.

Evidence classes are not added together:

| Class | Evidence |
|---|---|
| Request configurations | 8 configurations, each with packet and analytic arms, 3 steps per request |
| Retention configurations | 2 additional return delays |
| Attachment configurations | 4 rate/capacity combinations, each with baseline, reversed-order and class-permutation controls |
| Exact timing oracles | 136, maximum absolute residual 0 ps |
| Behavioral relations | 4 families, 36 matching instances |
| Fatal structural guards | 253 checked conditions, no findings, unscored |
| Compatibility families | 4 exact families, unscored |
| Native component executables | 9 pass, component evidence only |
| Hardware measurements | 0 |

## Chronology and reproduction

The final expectations-only commit is
`319cf7f3341ad7d204454810a096981d8cc216c7`, preceding implementation
`69a3ab0` and every run. The accepted executed source is
`e13bf1feffe3a339820d40f6d72380ee78dc8cc5`. Frozen parameters and relations
remain unchanged. [results.json](results.json) records the compact evidence,
raw digests and preserved runs.

The first compatibility invocation aborted before simulation because its
harness called the wrong entry point. Corrected invocations reproduce all
four families. `frozen-v1` is **VOID**, with behavioral score null: a malformed
harness field rejected direct domains and interrupted the retained control.
Its raw outputs and rejection records are preserved. `frozen-v2` passes the
initial checker. `frozen-v3` also checks the already-frozen TPOT relations on
published request metrics and literal packet geometry. All eight request raw
artifacts are byte-identical between those last two runs. No expectation was
rewritten after observing a result.

Set `SIMLLM_PEER_RUN_ROOT` to an external output directory. Build the same
pinned native backend against the baseline and current SimLLM roots, then run
`compatibility.py produce` for each, passing `--source-root`, `--output`,
`--htsim`, `--txt2bin` and `--native-producer`. The producer runs the complete
accepted studies and records native observations. Compare those directories:

```bash
python examples/local_peer_packet_runtime_v1/compatibility.py compare \
  --before "$SIMLLM_PEER_RUN_ROOT/before" \
  --after "$SIMLLM_PEER_RUN_ROOT/after" \
  --output "$SIMLLM_PEER_RUN_ROOT/comparison"
python examples/local_peer_packet_runtime_v1/run_study.py \
  --output "$SIMLLM_PEER_RUN_ROOT/frozen" \
  --compatibility "$SIMLLM_PEER_RUN_ROOT/comparison/comparison.json"
```

The runner requires committed source, the original frozen inputs and a new
output directory. Fatal findings void the entire run and clear its score.

## Supported boundary and remaining work

The ordinary serial step sink supports peer writes on declared direct or
switched NVLink domains. Its existing stateless remote backends retain their
usual restrictions, including rejection of multiple stateful physical backend
artifacts. Every selected remote result must match the rendered message
inventory, timing and quiescence. Retained physical remote sessions combined
with locality remain rejected under BACK-72. Parallel speculative preparation
cannot clone the local calendar. Detailed packet critical-path reporting stays
explicitly disabled under BACK-73.

TRAF-54 retains the collective protocol above these peer extents. COMP-40
retains host-port emission, and COMP-31/35 retain further local mechanisms and
vendor generalization. Reverse credit and acknowledgement controls use
propagation plus declared processing, without assigning unidentified control
packet bytes or link occupancy. TRAF-65/73/86 retain hardware identification;
this study does not qualify an NVSwitch product or an undocumented credit,
buffer or arbitration field. COMP-54/59 and CORE-65/66 retain Kimi K3 structure,
physical workload coverage and DeepSeek-V3 deployment evidence. None of those
large-model or hardware tasks closes with this synthetic loop check.
