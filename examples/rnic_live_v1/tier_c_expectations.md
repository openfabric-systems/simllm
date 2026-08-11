# Live RNIC composition Tier C expectations

## Freeze status and closure scope

This is an additive expectations-only record for the HTSIM-9 closure run. It
does not edit or replace any Tier A, Tier B or packet-v2 frozen file. It
precedes the Tier C producer, checker implementation and every result-producing
Tier C invocation in this worktree.

Tier C repeats the landed Tier B live chain under NetworkPort ABI v2:

```text
explicit packet TX start
  -> native WQE first and last packet timeline
  -> ComposedRnicSession
  -> ExecutionGraph and CompletionEvent
  -> ExecutionResult and StepResult
  -> request TTFT and TPOT
```

The closure claim is limited to the registered isolated fixture. Network
acceptance and whole-flow terminal events cannot satisfy packet-issue
evidence. ABI v1 remains the exact off path.

## Audited sources before freeze

The SimLLM source audit is commit
`90ada43070adb3b1e624b6819aff34d8620e8571`. The htsim source audit is pinned
backend commit `4885c647eecdfdf81479d1df052223c016ad086b`.

- `simllm/backends/rnic/src/work_queue.cpp:1030-1094` validates ABI-v2 packet
  envelopes and updates `first_packet_at_ps` and `last_packet_at_ps` only for
  data or retransmission `PacketTxStarted` events.
- `examples/rnic_live_v1/native/tier_a_producer.cpp:774-843` publishes the
  native packet timeline separately from network acceptance, whole-flow
  terminal time and explicit packet event rows.
- `htsim/sim/simllm_htsim_network_port.cpp:398-468` at the audited backend
  commit emits unbound ABI-v2 TX start and finish events at integer serializer
  boundaries, followed by RX and packet terminal events.
- `htsim/sim/rnic_packetized_manifold_runtime.cpp:349-488` at the audited
  backend commit emits TX start and finish from committed source serializer
  reservations in the bound packetized runtime.
- `htsim/sim/simllm_htsim_network_port.cpp:617-696` at the audited backend
  commit relays the bound runtime event time and packet identity into an ABI-v2
  packet-attempt event without substituting flow acceptance.
- `simllm/backends/composed_rnic.py:66-82`, `105-176` and `421-490` are the
  surrogate being replaced for ABI v2. The audited adapter retains only
  `port_tx_at_ps` plus whole-flow terminal time, so it cannot project the two
  native packet fields into the live chain.
- `simllm/core/runtime.py:1989-2047` maps the projected WQE start into the WQE
  subject `CompletionEvent.STARTED`, then maps completion into the CQ event.
- `examples/rnic_live_v1/tier_b_producer.py:269-360` runs one prefill and two
  decode graphs through `CoarseDeviceRuntime`, `CompletionReducer`,
  `StepResult`, TTFT and TPOT.

No hardware timing parameter is inferred from an observed Tier C result. The
closed forms below follow from those audited serializer boundaries, the frozen
zero-header and zero-propagation fixture, and the already accepted Tier B
reduction.

## Registered matrix and closed forms

Retain the Tier B structural matrix:

- payload `P` in `{4096, 1048576}` bytes;
- link rate `R` in `{200, 400}` Gbit/s;
- native doorbell service `D` in `{0, 1000}` ps;
- one prefill and two decode steps released from `T0 = 7000 ps`;
- one-WQE cells for both payloads and two-WQE FIFO cells at 4096 bytes;
- zero headers, propagation, congestion, control frames and every other
  SimLLM service term.

Let the packet payload quantum be `Q = 4096` bytes. Every registered payload is
an exact multiple of Q. For a single WQE, relative to its graph release:

```text
network acceptance A(P,R,D) = D
first TX issue F(P,R,D)      = D
last TX issue K(P,R,D)       = D + (P - Q) * 8 * 1000 / R
whole-flow terminal T        = D + P * 8 * 1000 / R
live step latency J          = T
```

The exact one-WQE serialization terms are 163,840 and 81,920 ps for 4096
bytes at 200 and 400 Gbit/s. They are 41,943,040 and 20,971,520 ps for
1,048,576 bytes. The exact last-minus-first packet spans for the large payload
are 41,779,200 and 20,889,600 ps.

The 1 MiB cells are the acceptance-separation cells. Their last packet issue
is strictly later than network acceptance and strictly earlier than the
whole-flow terminal. Therefore copying acceptance or terminal time into both
packet fields cannot pass the exact packet ledger.

The FIFO cells retain the accepted capacity-one equations. For service
`L = 4096 * 8 * 1000 / R`, W0 first and last packet issue occur at `D`, W0
terminates at `D + L`, W1 first and last packet issue occur at `D + L`, and W1
terminates at `D + 2L`. The live request endpoint is W1.

## Scored family 1: doorbell packet-to-live chain

There are four instances, one per payload and rate. Compare D equal to 1,000
ps against D equal to zero. In every one of the three request steps:

- first and last packet offsets from that step's graph release move by exactly
  `+1000 ps`;
- the WQE subject `CompletionEvent.STARTED` offset moves by exactly `+1000 ps`;
- `StepResult.step_latency_ps`, request TTFT and each defined TPOT move by
  exactly `+1000 ps`; and
- absolute `StepResult.completed_at_ps` moves by `+1000`, `+2000` and
  `+3000 ps` for steps zero, one and two because each later release consumes
  the preceding result.

Each payload-rate comparison is one conjunctive scored instance. A failure of
any packet, event or live request field fails that instance. This is the
decision-relevant hardware-service relation: increasing native doorbell
service must delay explicit packet issue and must reach the reported request
metrics with the frozen sign and magnitude.

## Scored family 2: link-rate packet-to-live chain

There are four instances, one per payload and D. Compare 200 Gbit/s against
400 Gbit/s. First packet offset has exact signed slow-minus-fast change zero.
For 4096 bytes, last packet offset also has change zero because the WQE is one
packet, while the live step, TTFT and defined TPOT changes are exactly
`+81920 ps`. For 1 MiB, last packet offset changes by exactly
`+20889600 ps`, while the live step, TTFT and defined TPOT changes are exactly
`+20971520 ps`.

Each payload-D comparison is one conjunctive scored instance. This separates
the last packet issue boundary from flow acceptance and checks that changing
link rate still reaches the live request metric boundary.

## Entailment analysis and checker order

Both scored families must execute against raw Tier C observations before the
checker invokes the inherited Tier B exact oracle or any per-cell packet
closed-form oracle. Schema and type parsing may run first. The adapter may
also require that the native packet fields equal the minimum and maximum
explicit data or retransmission TX-start rows for the same WQE. That origin
guard does not entail either scored family: it pins equality within one cell,
but it does not pin the cross-cell D effect, rate effect, CompletionEvent
projection or live metric movement.

Every scored instance can fail in a run that reaches it. A valid monotonic TX
event stream can omit D from one cell, apply the wrong link rate, project a
constant start into `CompletionEvent`, or reach the native timeline without
moving `StepResult`, TTFT or TPOT. Each defect passes basic parsing and fails
the relevant raw-observation relation before a later exact oracle can entail
the result.

Per-cell closed forms, origin equality, packet cardinality, payload closure,
event identity, authority exclusivity, request conservation, queue ordering,
quiescence and ABI-v1 digests are fatal and unscored. They never increase a
behavioral denominator.

## Negative controls

The deployed checker must reject three author-defined mutants. They are fatal
and unscored:

1. `acceptance_surrogate` replaces first and last packet fields with the WQE's
   network acceptance time. The 1 MiB link-rate relation or later origin
   oracle must reject it.
2. `producer_constant` supplies a release-relative constant for packet issue
   across the D grid. The doorbell packet-to-live relation must reject it.
3. `missing_tx_start` removes the explicit TX-start rows while retaining the
   producer packet fields. The origin guard must reject it.

The controls demonstrate checker sensitivity. They are not sampled runtime
behavior and do not enter genuine-risk fractions.

## ABI-v1 off-path and native gates

Before Tier C is accepted, regenerate the accepted ABI-v1 Tier A and Tier B
artifacts with the audited backend pin and the current SimLLM tree. Their
SHA-256 values must remain exactly:

| Artifact | SHA-256 |
|---|---|
| Tier A `raw_observations.json` | `37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a` |
| Tier A `summary.json` | `00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004` |
| Tier B `raw_observations.json` | `acaca5c57134848a314a92d223c283a7dc63f1c3ef964f65f7dea75487d6dfa1` |
| Tier B `results.json` | `3755bf5c2b37e9c30f90f97e3d6920841c70d052ff9164434d26c4f56773f0ed` |

These identity checks are fatal off-path regression evidence, not scored Tier
C families. The complete htsim CTest suite from the composed build, the
standalone SimLLM native CTest suite, ruff and pytest must also pass. Native
test executable counts remain a separate evidence class.

## Registered command and pre-freeze dry run

Load machine-local paths from `.env.local.sh`, then run:

```bash
.venv/bin/python examples/rnic_live_v1/tier_c_run.py \
  --htsim-source "${SIMLLM_HTSIM_SOURCE:?configure SIMLLM_HTSIM_SOURCE}" \
  --tier-b-reference-rnic \
    "${SIMLLM_TIER_B_REFERENCE_RNIC:?configure SIMLLM_TIER_B_REFERENCE_RNIC}" \
  --tier-b-reference-dcqcn \
    "${SIMLLM_TIER_B_REFERENCE_DCQCN:?configure SIMLLM_TIER_B_REFERENCE_DCQCN}" \
  --out \
    "${SIMLLM_WAVE5_RUN_ROOT:?configure SIMLLM_WAVE5_RUN_ROOT}/codex/htsim9_packet_closure"
```

The runner configures the audited backend with
`HTSIM_ENABLE_SIMLLM_RNIC=ON` and `SIMLLM_REPOSITORY_ROOT` naming the current
repository. The Tier C producer is invoked with exactly `--factory htsim`,
`--expectations` and `--observations`; it invokes the built native producer
with `--network-abi-version 2` in addition to the accepted Tier A arguments.

Before this freeze, execute the registered outer command with `--check-only`
appended. Check-only mode validates the source revision literals, output-root
shape, complete 8-cell single-WQE and 4-cell FIFO matrices, both four-instance
scored families, exact signed magnitudes, negative-control names, ABI-v1
digests and internal argument vectors. It treats the future producer as
opaque, creates no directory, builds no source, invokes no simulator and
publishes no measured artifact.
