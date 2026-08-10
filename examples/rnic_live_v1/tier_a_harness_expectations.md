# Tier A preparation harness expectations

## Freeze status

This is the expectations-only record for the SimLLM-side HTSIM-9 preparation
harness. It is separate from the frozen composed-run gate in
[`expectations.md`](expectations.md), which remains byte unchanged. This freeze
contains the acceptance checker, its declarative matrix and its command-line
contract. It contains no `RnicDevice` construction, port factory, event loop,
raw observation or measured result.

The harness is component evidence. The successor that makes it live-reachable
is HTSIM-9, which must run this same contract against the composed htsim binary
by replacing only the port factory. The preparation harness does not claim a
`StepRecord -> StepResult -> TTFT/TPOT` result.

## Source audit before freeze

The htsim source referent is the pinned gitlink
`8c3f8b231a6a9311ffc1e7969a003dcba724b50d`. Its
`AtlahsFlowRequest` contains flow, endpoint, payload, start-time and tag fields
at `third_party/htsim/htsim/sim/atlahs_flow_runtime.h:20-27`, and the outer
runtime contract is `setup`, `send` and physical-work drainage at
`third_party/htsim/htsim/sim/atlahs_flow_runtime.h:37-57`.

The landed SimLLM source referent is
`6aa3a7622f57b63c35e030667bad24948c6a0e0e`. Its relevant surfaces were
audited before this freeze:

- `NetworkTxDescriptor` carries opaque WQE, WR, flow, policy, endpoint,
  traffic-class, extent and eligibility fields at
  `simllm/backends/rnic/include/simllm/rnic/network_port.h:42-61`.
- A port returns Accepted, Busy or Rejected at
  `simllm/backends/rnic/include/simllm/rnic/network_port.h:63-99`; its current
  terminal vocabulary is only Delivered or Dropped at
  `simllm/backends/rnic/include/simllm/rnic/network_port.h:22-25,101-110`.
- `RnicDevice` injects an externally owned port at
  `simllm/backends/rnic/include/simllm/rnic/rnic_device.h:47-70`, and requires
  same-time external events before device progress at
  `simllm/backends/rnic/include/simllm/rnic/rnic_device.h:122-128`.
- Scalar doorbell service is the DMA-off `doorbell_service_ps` field at
  `simllm/backends/rnic/include/simllm/rnic/work_queue.h:67-85`. The fixture
  does not route D through the port factory.
- The scalar doorbell path computes observation as the serialized cursor plus
  D at `simllm/backends/rnic/src/work_queue.cpp:307-310`, then the zero-service
  fetch, QPC and scheduler path preserves that time at
  `simllm/backends/rnic/src/work_queue.cpp:360-395`.
- Network submission visits the ready SQ head and retains it on Busy at
  `simllm/backends/rnic/src/work_queue.cpp:428-545`. Ordered retirement and
  CQE construction follow SQ sequence at
  `simllm/backends/rnic/src/work_queue.cpp:969-1039`, and CQ polling stamps the
  caller time at `simllm/backends/rnic/src/work_queue.cpp:636-655`.
- The existing deterministic fake port supplies capacity, fixed latency,
  future Busy retry, token allocation and ordered due-event extraction at
  `simllm/backends/rnic/tests/fake_network.h:25-68,87-139`.
- The current WQE timeline explicitly forbids treating flow acceptance or
  delivery as first-packet issue at
  `simllm/backends/rnic/include/simllm/rnic/work_queue.h:120-139`.
- Pinned htsim makes its DATA header explicit and permits zero header bytes at
  `third_party/htsim/htsim/sim/rnic_packet_extent.h:37-81`. Its packet ledger
  computes total wire bytes as payload plus packet count times header bytes at
  `third_party/htsim/htsim/sim/rnic_packetized_manifold_runtime.cpp:242-259`.
- Pinned htsim computes integer picoseconds per byte as `8 * 10^12 / bitrate`
  at `third_party/htsim/htsim/sim/queue.cpp:11-14`, multiplies packet bytes by
  that value at `third_party/htsim/htsim/sim/queue.h:54-56`, and schedules the
  serializer completion by that duration at
  `third_party/htsim/htsim/sim/queue.cpp:130-159`. The future htsim factory
  must select the frozen single-serializer test topology. A routed topology
  with additional serializers requires a separate expectation amendment.

Therefore `port_tx_at_ps` below is a fake-port probe observation. It is kept
distinct from `network_accepted_at_ps` and is not claimed to populate native
`first_packet_at_ps`. BACK-25 owns the missing versioned packet-attempt and
issue-event vocabulary before the real composed gate can claim that native
field.

## Registered commands and pre-freeze dry run

The fake command contract is:

```bash
.venv/bin/python examples/rnic_live_v1/tier_a_acceptance.py \
  --factory fake \
  --producer /data3/yifeng/simllm-dev/wave2-runs/codex/htsim9_prep_harness/tier_a/fake/build/simllm_rnic_tier_a \
  --run-dir /data3/yifeng/simllm-dev/wave2-runs/codex/htsim9_prep_harness/tier_a/fake
```

The future composed command changes the factory and producer only:

```bash
.venv/bin/python examples/rnic_live_v1/tier_a_acceptance.py \
  --factory htsim \
  --producer /data3/yifeng/simllm-dev/wave2-runs/codex/htsim9_prep_harness/tier_a/htsim/build/htsim_rnic_tier_a \
  --run-dir /data3/yifeng/simllm-dev/wave2-runs/codex/htsim9_prep_harness/tier_a/htsim
```

Before this freeze, both exact commands are run with `--check-only` appended.
That mode parses the CLI, validates the complete declarative matrix and raw
schema contract, and validates that every output path is under the external
wave-2 run root. It does not inspect the producer path, construct a port or
device, create the run directory, open an observation file or emit a result.

After implementation, the checker invokes the producer with the frozen
arguments `--factory`, `--expectations` and `--observations`. The producer must
emit raw observations only. It must not emit expected values, PASS fields or
behavioral counts. Immediately before atomically publishing the observation
file, the producer must call native `RnicDevice::validateInvariants()` for
every constructed session. Any failed invariant makes the producer exit
nonzero without publishing an observation file. Successful validation is not
self-certified in the raw schema.

## Fixture and exact oracle

For the single-WQE grid, sweep payload P in `{4096, 1048576}` bytes, link rate
R in `{200, 400}` Gbit/s and scalar doorbell service D in `{0, 1000}` ps. All
other native services are zero, the deterministic port has capacity one, and
network service is

```text
L(P, R) = P * 8 * 1000 / R ps
```

This formula is not an accidental fake-port convention. Both factories use
the frozen exact fixture with zero DATA-header bytes, zero propagation, no
control frames and no congestion, so total wire bytes equal payload bytes.
The controlled-drop case is the only baseline override. A broader composed
packetization or control-plane study requires its own expectations-only
amendment and cannot reuse these exact rows silently.

The registered values are exact integers:

| Payload | L at 200 Gbit/s | L at 400 Gbit/s |
|---:|---:|---:|
| 4 KiB | 163,840 ps | 81,920 ps |
| 1 MiB | 41,943,040 ps | 20,971,520 ps |

For each of the eight structural single-WQE rows:

```text
eligible       = D
port_tx        = D
terminal       = D + L(P, R)
CQE visible    = D + L(P, R)
CQ poll        = D + L(P, R)
JCT            = D + L(P, R)
```

The separate FIFO grid posts two signaled 4 KiB WQEs, W0 then W1, in one
doorbell to the same SQ. For every R in `{200, 400}` and D in `{0, 1000}`:

```text
eligible(W0) = D
eligible(W1) = D
port_tx(W0)  = D
terminal(W0) = D + L
port_tx(W1)  = D + L
terminal(W1) = D + 2 * L
wait(W1)     = port_tx(W1) - eligible(W1) = L
JCT          = D + 2 * L
CQE order    = W0, W1
```

## Scored behavioral relations

The checker scores three families separately, with four parameterized
instances in each family:

1. D-additivity over payload by rate. For each `(P, R)`, raising D from 0 to
   1,000 ps increases eligibility, fake-port TX issue, terminal time, CQE
   visibility, poll time and JCT by exactly 1,000 ps. The quantitative band is
   `[1000, 1000]` ps and the signed direction is positive. Network service
   `terminal - port_tx` is unchanged.
2. Inverse-rate serialization over payload by D. For each `(P, D)`, the
   observed network service at 200 Gbit/s is exactly twice the service at
   400 Gbit/s.
3. Two-WQE FIFO over rate by D. Each of the four rows satisfies the frozen
   eligibility, TX, terminal, W1 wait and JCT timing equations. CQE ordering is
   a fatal structural check against the landed ordered-retirement source, not
   a scored relation.

The eight single-WQE exact row-oracle checks are reported separately and are
not added to the 12-instance behavioral denominator. FIFO ordering and CQE
boundary checks are fatal structural evidence and likewise do not enter that
denominator.

## Decision-relevant sensitivity control

The producer must run a wrapper-bypass mutant with the same valid-looking raw
shape as one structural `(P, R)` D pair. The mutant deliberately discards the
requested D when constructing the D=1,000 ps device, so its raw timestamp
delta is zero. The acceptance checker applies the exact same D-additivity
predicate used for scored rows and must reject the pair. It must not inspect a
mode label, configuration echo or producer-provided PASS flag to reject it.

This control is fatal and unscored. If it does not fail the checker, the Tier A
gate cannot distinguish a bypassed wrapper from a composed wrapper. That
failure changes the design decision: the composition acceptance gate must be
redesigned before HTSIM-9 can be accepted.

## Fatal unscored invariants

Every structural row reports one native session, N native posts, no legacy
ledger construction and zero legacy mutations. The bypass authority case
reports no native session or posts, one legacy ledger and N legacy posts. A
dual-authority construction attempt records its caught exception type and
message plus the raw counter snapshots before and after. The checker derives
rejection and atomicity from those observations. No producer verdict field is
accepted.

For every accepted structural session, the checker independently requires:

- every issued token is nonzero and unique for the whole session;
- the frozen v1 one-extent fixture issues exactly one token for each WQE;
- every issued token has exactly one Delivered or Dropped terminal;
- every terminal names the issued token and its WQE;
- each issue row's acceptance, fake-port TX probe and payload agree with the
  corresponding native WQE and cell projections;
- `issued = delivered + dropped + live` throughout the final snapshot;
- device accepted, delivered and dropped counters equal the port ledger;
- quiescence has zero live tokens, zero pending physical work, empty SQ and CQ,
  and a successful native invariant validation.

The controlled-drop case uses one unsignaled SEND. It must produce exactly one
`TransportError` CQE, no Success CQE, one native NetworkDrop evidence entry,
one accepted and dropped token, no delivery and zero live tokens. Error CQE
behavior is grounded in
`simllm/backends/rnic/src/work_queue.cpp:593-633,1004-1039` and the landed
directed test at `simllm/backends/rnic/tests/work_queue_test.cpp:522-556`.

Duplicate, unknown and cross-WQE terminal controls must reject before port
ledger, device lifecycle, counter or time mutation. The current direct queue
validation is at `simllm/backends/rnic/src/work_queue.cpp:551-615`, but the
post-specified fixture audit found that `RnicDevice::onNetworkEvent` observes
caller time first at `simllm/backends/rnic/src/rnic_device.cpp:407-413`.
BACK-24 owns that correctness repair. The harness wrapper must prevalidate the
terminal ledger and demonstrate atomic rejection, but that does not close
BACK-24's direct-device defect. Each control starts from a completed two-WQE,
capacity-two fixture, records the caught exception type and message, and
records exact nested snapshots before and after. A raw progress call at
150,000 ps after the rejected 200,000 ps event must return zero changes and no
exception. The checker derives both rejection and caller-clock preservation;
the producer supplies no `rejected`, `accepted` or PASS boolean.

Authority, token conservation, terminal atomicity, controlled drop,
quiescence, native FIFO and CQE ordering, the wrapper-bypass sensitivity
control and exact rows remain fatal and unscored. Run configuration is
reported but unscored. Native executable counts are component evidence and
never enter a behavioral total.

## Plausible failure mechanisms frozen before the run

- D-additivity can fail if the outer wrapper drops D, charges it twice or
  applies it after network service.
- Inverse-rate serialization can fail through bit/byte or Gbit/ps conversion,
  integer truncation or by including D in the network term.
- FIFO can fail if the capacity-one port advertises the wrong retry time, the
  event loop progresses before delivering a same-time terminal, or a later WQE
  bypasses the SQ head.

These are mechanism-bearing relations, not configuration echoes. The
[preparation results](tier_a_harness_results.md) must report, per scored
family, how many instances a competent implementation could plausibly have
failed and why.
