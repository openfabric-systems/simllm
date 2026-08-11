# RNIC packet-event ABI v2 expectations

## Freeze status and scope

This is the expectations-only record for BACK-25 and BACK-26. It precedes
the NetworkPort ABI v2 implementation, the htsim runtime-event relay, every
new native test and every result-producing run of this study. The companion
runner contains frozen literals, build orchestration and validation logic.
Its `--check-only` path does not compile or import the future ABI.

This remains component evidence. The composed wrapper reaches native WQE and
CQE state, but it does not create the Tier B
`ExecutionGraph -> DeviceRuntime -> CompletionEvent -> ExecutionResult ->
StepResult -> TTFT/TPOT` chain. CORE-15 is the registered successor for that
live-chain relation. HTSIM-9 remains open until Tier B carries packet-issue
evidence through that chain.

## Source audit before freeze

The audit was completed before this freeze against SimLLM commit
`b74629b4b4da1addda9ff21226cfabf5c09aad87` and htsim commit
`edb28c3015c173b4251abc5858c587df325e1ebc`. No ABI v2 implementation or
result-producing command was run before this record.

- SimLLM's complete ABI v1 event enum contains only `Delivered` and
  `Dropped` at
  `simllm/backends/rnic/include/simllm/rnic/network_port.h:14-25`. The
  descriptor and terminal record are at the same file's lines 42-61 and
  101-110.
- The work queue creates the descriptor and records acceptance at
  `simllm/backends/rnic/src/work_queue.cpp:670-718`. Its terminal-only event
  path is at lines 784-866. The native packet fields are deliberately unset
  and documented as explicit-TX-event projections at
  `simllm/backends/rnic/include/simllm/rnic/work_queue.h:142-160`.
- The htsim compatibility port currently records `port_tx_at_ps = now_ps`
  when the flow is accepted at
  `htsim/sim/simllm_htsim_network_port.cpp:175-251`. A runtime completion is
  reduced to one flow terminal at lines 254-279. Neither location is a
  packet TX issue source.
- The packetized manifold commits an exact transmission with source start,
  source finish and destination boundaries at
  `htsim/sim/rnic_packetized_manifold_runtime.cpp:334-413`. The immutable
  transmission accessors are at
  `htsim/sim/rnic_packetized_manifold.h:28-58`. These are the audited source
  of packet TX and RX observations.
- Source service settles only at the committed source-finish boundary at
  `htsim/sim/rnic_packetized_manifold_runtime.cpp:461-500`; destination
  delivery settles at lines 503-545. A wrapper may relay these observations
  but may not replace them with flow acceptance or completion.
- The current htsim runtime interface has only setup, send, completion and
  physical-drainage surfaces at `htsim/sim/atlahs_flow_runtime.h:70-105`.
  ABI v2 therefore needs one read-only event relay on this existing narrow
  interface, not a second transport authority.
- The audited control sources are ECN marking at
  `htsim/sim/datacenter/ns_tm3_dcqcn_policy.cpp:232-243`, CNP production and
  consumption at `htsim/sim/dcqcn.cpp:366-403,307-315`, rate cuts and
  recovery at `htsim/sim/dcqcn.cpp:122-158,177-209`, and PFC pause or resume
  production at
  `htsim/sim/datacenter/ns_tm3_dcqcn_policy.cpp:319-363`.
- Static failed links are removed before runtime at
  `htsim/sim/datacenter/fat_tree_topology.cpp:1392-1408`. There is no audited
  timestamped dynamic-link producer. ABI v2 must represent a negotiated
  transition, while a runtime without that capability rejects enablement
  before mutation.

These sources identify event boundaries and fields. They do not authorize a
second hardware rate or PFC authority. SimLLM remains the consumer and
hardware-gate owner; htsim remains the transport-policy and fabric authority.

## Version and lifecycle contract

ABI v1 remains the default. A v1 NetworkPort requires no new override, emits
the existing descriptor and terminal bytes, and never populates packet
timestamps. ABI v2 is negotiated before the first submission. A session may
not mix descriptor or event versions.

One admitted logical extent retains its flow token. Every packet transmission
attempt has a separate nonzero session-unique attempt token and points back to
that extent token. Its identity carries logical extent index, packet index,
transmission-attempt index, payload offset, payload bytes, wire bytes and
data or control kind. TX start, TX finish and native RX arrival are
intermediate observations and never consume either token. `Delivered` and
`Dropped` are the only attempt terminals. The extent terminal remains the
only event that retires the native WQE. At quiescence, every issued extent and
attempt token has exactly one matching terminal and no token remains live.

The native `first_packet_at_ps` and `last_packet_at_ps` fields equal the
minimum and maximum event times among explicit data `PacketTxStarted` events
for that WQE. No acceptance, TX-finish, RX, attempt-terminal or flow-terminal
event may populate them. Removing every TX-start observation while retaining
the timeline must make the checker fail.

ABI v2 also carries these discriminated control forms:

- packet-keyed ECN mark and CNP feedback;
- policy-context-keyed eligibility and rate updates with an effective time;
- PFC frame submission and pause or resume reception with endpoint or stable
  link identity, priority and quanta or duration; and
- capability-negotiated link transitions with stable link identity, state,
  transition time and optional effective rate.

Busy remains resource backpressure. A per-attempt link failure remains
`DropReason::LinkDown`. Control and dynamic-link capabilities are explicit;
disabled capabilities preserve every ABI v1 timestamp, byte, token order and
random draw. A requested dynamic transition without a timestamped producer
rejects before runtime mutation.

## Frozen Tier A matrix and exact packet oracle

The v2 run reuses the complete Tier A matrix. Single-WQE rows sweep payload
`P` in `{4096, 1048576}` bytes, rate `R` in `{200, 400}` Gbit/s and native
doorbell service `D` in `{0, 1000}` ps. FIFO rows sweep the same rates and
doorbells for two 4096-byte WQEs. Header and propagation are zero. The packet
wire quantum is 4096 bytes.

For one WQE, let:

```text
N(P)       = ceil(P / 4096)
L(P, R)    = P * 8 * 1000 / R ps
Q(P, R)    = 4096 * 8 * 1000 / R ps, except the final packet uses its exact bytes
first_tx   = D
last_tx    = D + (P - final_packet_bytes) * 8 * 1000 / R
first_rx   = D + 4096 * 8 * 1000 / R, capped to P for a short packet
last_rx    = D + L(P, R)
terminal   = last_rx
```

Every packet has ordered TX-start, TX-finish, RX-arrival and Delivered
observations. Packet `i` starts where packet `i - 1` finishes. Payload offsets
and byte sums close exactly. `first_packet_at_ps = first_tx` and
`last_packet_at_ps = last_tx`.

The inherited Tier A D-additivity, inverse-rate and FIFO families must still
pass under v2, with four instances in each family. All eight exact single-WQE
rows, the four FIFO rows, controlled drop, authority exclusivity, terminal
atomicity, wrapper-bypass sensitivity and quiescence remain required. They
retain their existing evidence-class separation.

## New scored behavioral relations

The new packet families are scored separately:

1. For each `(P, R)` pair, raising D from 0 to 1000 ps moves native first and
   last TX issue by exactly `+1000 ps`. The signed quantitative band is
   `[1000, 1000]` ps. There are four parameterized instances. This is the
   formerly unscorable relation 1 of `rnic_live_v1`.
2. For each `(P, R)` pair, the same D change moves first and last RX arrival
   by exactly `+1000 ps`, also with band `[1000, 1000]` ps. There are four
   parameterized instances.
3. For the multi-packet 1 MiB rows at each D, the last-minus-first TX issue
   span and last-minus-first RX span at 200 Gbit/s are exactly twice their
   400 Gbit/s values. There are two parameterized instances. The one-packet
   zero spans are fatal exact-oracle checks and do not enter this denominator.

The inherited twelve Tier A instances and the ten new packet instances are
reported as distinct behavioral families. Exact rows, token conservation,
event ordering, capability validation and author-defined sequence checks are
fatal unscored evidence.

## ABI v1 byte identity

The accepted htsim Tier A reference was produced for backend implementation
`f88d9fd24eb294944fd0c90c955c8924a00c5106`, cited by backend result commit
`7e0bac5357065ee19553bcee9755dfb3e3e2815d`. The frozen external artifacts are:

| Artifact | SHA-256 |
|---|---|
| `raw_observations.json` | `37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a` |
| `summary.json` | `00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004` |

The post-change v1 run must match both files byte for byte. This is a separate
two-instance scored compatibility family with exact zero-byte-difference
band. The documented legacy `htsim_uec` scenario and the complete backend
CTest suite remain fatal compatibility gates and are not added to that
denominator.

## Registered command and pre-freeze dry run

Set `SIMLLM_WAVE3_RUN_ROOT` to the external wave-3 output root and
`SIMLLM_HTSIM_SOURCE` to the paired htsim checkout. The result-producing
command is:

```bash
.venv/bin/python examples/rnic_packet_v2/run_study.py \
  --htsim-source "${SIMLLM_HTSIM_SOURCE:?configure SIMLLM_HTSIM_SOURCE}" \
  --v1-reference-dir \
    "${SIMLLM_WAVE3_RUN_ROOT:?configure SIMLLM_WAVE3_RUN_ROOT}/htsim9/fix-round-f88d9fd" \
  --out \
    "${SIMLLM_WAVE3_RUN_ROOT}/codex/back2526_packet_vocabulary/packet-v2"
```

Before this freeze, the same command was executed with `--check-only`
appended. That mode parses the complete CLI, validates both source commits,
the 12-cell matrix, ten new scored instances, event schema, exact v1 artifact
digests and external-output rule. It prints a registry confirmation by design.
It does not configure CMake, inspect a future producer, create an output
directory or produce a result artifact.
