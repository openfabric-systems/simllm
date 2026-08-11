# Physical transport-control producer expectations

## Freeze scope

This is the expectations-only record for HTSIM-15 and HTSIM-16. It was
written before implementation, new native tests, and every result-producing
run. The companion runner's `--check-only` path validates only frozen
literals, source revisions, accepted reference digests, and output placement.
It neither builds nor imports a future producer and creates no artifacts.

This is component evidence at the composed native RNIC boundary. It does not
claim the later `ExecutionGraph` through `CompletionEvent`, `StepResult`, TTFT
and TPOT chain. HTSIM-9 and CORE-15 remain the registered successors for that
live-chain evidence.

## External-source audit before freeze

The audit used SimLLM commit
`90ada43070adb3b1e624b6819aff34d8620e8571` and htsim commit
`4885c647eecdfdf81479d1df052223c016ad086b`. No new control-producing run was
executed before this record.

- The landed relay maps all physical control kinds and keeps completed packet
  correlation until the parent extent retires at
  `htsim/sim/simllm_htsim_network_port.cpp:617-760`. Its capability check
  rejects a requested vocabulary that the runtime does not advertise at
  `htsim/sim/simllm_htsim_network_port.cpp:217-230`.
- The SimLLM work queue accepts ECN or CNP only against a live or completed
  packet attempt, validates the policy token and effective rate, and validates
  PFC and link payloads at
  `simllm/backends/rnic/src/work_queue.cpp:1161-1245`.
- Real ECN marking occurs during ns-tm3 egress selection at
  `htsim/sim/datacenter/ns_tm3_dcqcn_policy.cpp:232-243`. The sample includes
  policy seed, switch domain, wire flow, packet sequence, egress, and ingress
  at the same file's lines 261-295.
- The DCQCN receiver creates CNP feedback after receiving marked data at
  `htsim/sim/dcqcn.cpp:366-403`; the source consumes it at lines 307-315.
  Rate cuts and recovery update the actual pacing rate at lines 122-158 and
  177-209.
- The lossless policy submits a PFC pause or resume at
  `htsim/sim/datacenter/ns_tm3_dcqcn_policy.cpp:319-363`. Its dedicated real
  reverse serializer commits the frame arrival at lines 48-105, so submission
  and upstream pause reception are distinct physical observations.
- The existing eight-flow incast test drives the actual DCQCN runtime and
  asserts nonzero ECN and balanced PFC pause or resume counters at
  `htsim/sim/datacenter/dcqcn_atlahs_runtime_test.cpp:17-70`. The runtime does
  not implement `eventCapabilities` or `setEventHandler` at
  `htsim/sim/datacenter/dcqcn_atlahs_runtime.h:49-64`, so no physical control
  event currently reaches the relay.
- Current failed links are static topology mutations performed before runtime
  at `htsim/sim/datacenter/fat_tree_topology.cpp:1392-1408`. There is no
  timestamped transition source.

These locations are the only producer authorities used by the study. The
implementation may add read-only observers and a scheduled endpoint-link
gate, but it may not duplicate ECN, CNP, DCQCN rate, or PFC decisions.

## Frozen producer contract

The physical runtime uses the 64-node `clos_64_400g.topo`, 400 Gbit/s endpoint
links, 4,096-byte maximum wire packets, 64-byte data headers, and policy token
9,001. ECN uses `kmin=0`, `kmax=4096`, `pmax=1000000 ppm`, and seed 9.

Packet observations are taken from actual host serialization and sink arrival.
An ECN event uses the exact attempt marked by ns-tm3. A CNP event uses the
marked attempt that caused feedback even when its Delivered event has already
completed the attempt. Rate and eligibility updates carry the descriptor's
policy token and the real effective pacing rate. A rate must never be inferred
from an event count.

PFC submission is observed where the policy hands the real frame to its
reverse link. Pause and resume are observed when that reverse link delivers
the frame upstream. Each row carries a stable nonzero physical-link identity,
priority, source and destination. Pause has nonzero quanta or duration;
resume has neither.

The dynamic source controls source endpoint link 1. A transition at timestamp
`T` emits exactly one `LinkStateChanged` observation at `T`. Down prevents a
new low-priority data serialization from starting, without truncating a packet
already in service. Up restores 400 Gbit/s and resumes queued data. The source
exists and the capability is advertised only when the transition list is
nonempty.

Capabilities are independent. Packet attempts are advertised only when their
real observation path is enabled. ECN/CNP and policy-update capabilities are
advertised only with their real observers. PFC is advertised only when both
the physical PFC policy and its observer are enabled. Dynamic link is
advertised only with a nonempty validated transition schedule. A requested
unsupported capability rejects before setup or submission changes state.

## Frozen cells and scored relations

The congestion sweep has four-flow and eight-flow 64 KiB incasts from sources
starting at zero into destination 63, with PFC physically disabled. For each
cell, the first CNP-induced effective rate update is compared with the fixed
400 Gbit/s starting rate. Its signed delta must be in
`[-300000000000, -1] bit/s`. There are two scored instances. The event must be
absent when the control relay is disabled.

The PFC sweep has eight 64 KiB flows and eight 128 KiB flows, both with low
and high thresholds 4,096 and 8,192 bytes. For each cell, raw arrival times of
a matching pause and resume produce a strictly positive paused interval in
`[1, 1000000000] ps`. The paired eligibility observations must show zero while
paused and a positive real rate when resumed. Pause counts and exact sequence
checks are fatal unscored evidence, not the scored relation.

The dynamic sweep sends one 64 KiB flow from source 0 to destination 63. Both
cells transition link 1 down at 1,000 ps. The short cell returns up at 201,000
ps and must increase completion time over its no-transition control by
`[100000, 250000] ps`. The long cell returns up at 401,000 ps and must increase
completion time by `[300000, 450000] ps`. The long-minus-short completion
increase must be `[190000, 210000] ps`. These are three scored timing
instances, while exact transition rows are fatal unscored evidence.

Every one of the six enabled cells has an observation-disabled twin with the
same real physical policy and transition schedule. After removing only the
negotiated control-event projection, the existing `BypassArtifacts` record and
checker must find zero changed input or behavioral byte classes. These six
identity instances cover completions, packet timestamps and tokens, policy
counters, ordering, and deterministic samples. The accepted ABI-v1 Tier A
artifacts are regenerated and compared byte for byte as two additional scored
compatibility instances.

## Entailment and evidence accounting

The runner evaluates rate deltas, pause intervals, dynamic completion deltas,
and enabled-versus-disabled identity directly from raw observations before it
runs any exact event, token, capability, or counter oracle. A first CNP rate
can be wrong while the run reaches the relation, a PFC interval can be zero or
outside its band, and a transition can leave completion unchanged. None is
entailed by an earlier fatal check.

Exact packet geometry, event ordering, late-CNP token correlation, one terminal
per attempt and extent, exact link rows, balanced PFC sequence, capability
rejection, quiescence, and author-defined schedules are fatal unscored. The
reference digest checks performed before a run are change-set guards and are
also unscored. The result reports each scored family's genuine-risk numerator
and denominator separately.

## Registered command and pre-freeze dry run

Set `SIMLLM_WAVE5_RUN_ROOT`, `SIMLLM_WAVE3_RUN_ROOT`, and
`SIMLLM_HTSIM_SOURCE`. The result-producing command is:

```bash
.venv/bin/python examples/rnic_control_v2/run_study.py \
  --htsim-source "${SIMLLM_HTSIM_SOURCE:?configure SIMLLM_HTSIM_SOURCE}" \
  --v1-reference-dir \
    "${SIMLLM_WAVE3_RUN_ROOT:?configure SIMLLM_WAVE3_RUN_ROOT}/htsim9/fix-round-f88d9fd" \
  --out \
    "${SIMLLM_WAVE5_RUN_ROOT:?configure SIMLLM_WAVE5_RUN_ROOT}/codex/htsim1516_control_producers/control-v2"
```

Before the freeze commit, this exact command was run with `--check-only`
appended. That path validates the two commits, all six cells, all fifteen
scored instances, accepted ABI-v1 digests, topology availability, and the
external output rule. It does not create the output directory.
