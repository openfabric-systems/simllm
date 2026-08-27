# NVLink domain model

The candidate NVLink domain is a queue-level decomposition of one directed
peer transfer into three independently parameterized services: TX, switch and
RX. This note records the merged model, its exact analytic bypass, the frozen
A100 case-to-parameter identification map and the evidence class of every
current claim. It documents the implementation; it does not promote the
candidate or execute a study.

## Model and evidence at a glance

![Queue-level TX, switch and RX NVLink domain model](../../resources/figures/nvlink-domain-model.png)

*Figure 1. The queue-level domain in `simllm.backends.htsim_nvlink`. Every
numeric module parameter is a DECLARED CANDIDATE. The two rates at the bottom
are PUBLISHED-MEASUREMENT CHECKS against the earlier A100 envelope, not
parameter-identification evidence. The first TRAF-65 hardware run voided its
capture procedure, so it promoted nothing. The corrected TRAF-70 capture is in
progress and is the gate before any candidate value or evidence class may be
promoted.*

The diagram uses the same module palette as the README device view: green for
the endpoint TX side, orange for the intervening fabric service and blue for
the endpoint RX side. The switch is always present in the composition. Its
selected A100 mode is an identity operation rather than an omitted box.

## Authorities and scope

Two merged files are authoritative for this study:

- `simllm/backends/htsim_nvlink.py` defines the TX, switch and RX contracts,
  their composition, the timestamps and the analytic bypass.
- `examples/a100_nvlink_packet_v1/candidate-profile.json` binds the A100
  candidate values and preserves the frozen expectations digest
  `212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571`.

The profile schema is `simllm-htsim-nvlink-candidate-profile-v1`; its evidence
class is `declared_candidate_not_hardware_measurement`. Loading code rejects a
different class or a status other than `candidate`.

This module is an additive htsim-style handoff. It is not the default
intra-node timing authority. The active traffic path still uses the analytic
per-endpoint serializer unless a caller explicitly selects the candidate. The
candidate can support component studies and downstream integration work, but
that does not make its parameters measured or put them on the live TTFT and
TPOT chain.

## Composition and queue boundaries

`NvlinkDomainService.serve` applies the modules in one fixed order:

```text
NvlinkTransfer
    -> TX packetization, staging, credits and bonded-link service
    -> switch pass-through or queued service
    -> RX ingress buffering, reassembly and delivery
    -> NvlinkDomainResult
```

The result carries a stable packet ledger, request and response payload and
wire-byte totals, the last delivery time and the maximum observed RX
occupancy. Logical bytes are counted from the input extents. Wire bytes are
counted after packet headers have been added. This separation prevents a
packet overhead hypothesis from silently changing the application payload.

The module timestamps are htsim-style local evidence, not core `QueueVisit`
records. Their mapping is:

| Boundary | Representation | Meaning |
|---|---|---|
| Logical release | `released_at_ps` | The extent, and then each packet, may enter TX |
| TX grant | `tx_started_at_ps` | Destination credit, selected link and endpoint egress are all available |
| TX release | `tx_finished_at_ps` | Wire serialization on the selected bonded link completes |
| Switch grant and release | `switch_started_at_ps`, `switch_finished_at_ps` | Present only for the queued mode; absent under pass-through |
| RX grant | `rx_started_at_ps` | The destination ingress cursor can accept the arrived packet |
| RX release and visibility | `rx_finished_at_ps`, `delivered_at_ps` | Ingress serialization finishes and that packet becomes visible |

There is no separate `eligible_at` field and no emitted core `QueueVisit` in
this candidate module. Downstream work must map these boundaries to the shared
queue-visit contract before using them in an additive critical-path metric.

## TX module

The TX service is per endpoint. It owns staging, packetization, credit gating,
the shared endpoint egress cursor and the physical-link bond.

### Per-destination staging

Each extent is packetized into an ordered packet group. The composition helper
round-robins ready groups in caller order, while link cursors and credit slots
are keyed by `(source, destination)`. That is the v1 per-destination staging
contract. It is represented by deterministic packet ordering and keyed state,
not by a separately exported queue record.

All destinations at one source also meet one endpoint egress cursor. Its
declared rate is 300 GB/s. A destination therefore has its own bond and credit
window while still contending for the source endpoint's total egress service.

### Packetization and direction

The maximum payload is 256 bytes and every packet adds a 16-byte header. A
short final packet keeps the same header. The resulting wire count is always
`payload_bytes + header_bytes` and construction rejects any disagreement.

Direction is a transaction property, separate from endpoint orientation:

- a peer write emits its data packets from source to destination as requests;
- a peer read first emits a zero-payload request packet from the initiating
  source, then emits data packets in the reverse endpoint direction as
  responses.

Request and response payload and wire bytes remain separate fields in the
domain result. The direction rule is declared candidate policy, not an
observation from the void hardware capture.

### Destination credits

Each directed source-destination pair has 256 credit slots. One slot covers a
272-byte maximum wire packet. Packet sequence modulo 256 selects the slot, so
the next packet mapped to that slot cannot start until its recorded return
time.

The v1 abstraction makes a slot reusable at `tx_finished_at_ps + 200,000 ps`.
It does not create a credit packet, wait on the modeled RX finish or expose a
separate reverse-link serialization. The blue return path in Figure 1 names
the causal ownership without claiming a more detailed protocol than the code
implements.

### Four-link bonding

Each directed peer pair has four link cursors, each at a declared 25 GB/s.
For every packet, TX chooses the link with the earliest cursor; a tie chooses
the lowest link index. The start time is the maximum of logical release, that
link cursor, the source endpoint cursor and the selected credit-slot return.
The link cursor advances by wire serialization at 25 GB/s, while the endpoint
cursor advances by serialization at 300 GB/s.

This is earliest-available packet striping. It is not round-robin striping,
and the four physical links are scoped per directed peer pair rather than
globally across the endpoint.

## Switch module

The switch is one parameterized module with two explicit modes.

### A100 direct-mesh pass-through

The A100 `NV4` candidate selects `pass_through`. `NvlinkSwitch.forward`
returns the exact packet tuple it received. It adds no packet, byte, timestamp,
reordering, queue visit or service delay. The behavior is therefore
byte-identical and object-identical to direct TX-to-RX composition.

Pass-through refuses FIFO placement, service rate, buffer capacity,
arbitration and head-of-line fields. That refusal matters: the absence of a
physical switch in the direct mesh cannot accidentally acquire a queue through
a default value. The identity is a structural topology invariant, not an A100
hardware measurement.

### NVSwitch-class queued parameterization

The same `NvlinkSwitch` box also supports `queued`. The input-placement form
shown in Figure 1 has one cursor per source port feeding a contention point.
The implementation may instead key the cursor by output destination or use one
shared cursor. FIFO arbitration is the only v1 arbitration policy. Enabling or
disabling head-of-line blocking controls whether an extent receives a separate
cursor under the selected placement.

A queued profile must supply placement, service rate, buffer capacity,
arbitration and head-of-line policy together. V1 verifies that an individual
packet fits the declared switch buffer and serializes the selected cursor; it
does not maintain a time-varying switch-buffer occupancy ledger.

This is the parameterization seam for H100 and GH200 NVSwitch-class paths.
No H100 or GH200 queued profile, buffer value, arbitration measurement or
service value is shipped here. Those architectures must supply their own
profile rather than inherit an A100 number.

## RX module

The RX service is per endpoint and keyed by packet destination. It owns ingress
serialization, the finite buffer ledger, sequence enforcement and delivery.

### Ingress FIFO and occupancy

Each destination has one ingress cursor at a declared 300 GB/s and one
occupancy queue. On packet arrival, RX first retires buffered wire bytes whose
ingress service has finished, then admits the new packet. Admission rejects a
single packet larger than the declared 1 MiB capacity and rejects aggregate
live occupancy above that capacity. The domain result reports the maximum
occupancy observed across destinations.

RX rate and capacity are independent declarations. Equal TX and RX endpoint
rates do not make them one parameter and do not let evidence for one identify
the other.

### Extent sequence and delivery

For each extent, packet sequence numbers must be strictly increasing in the
stream presented to RX. The destination ingress cursor then serializes packets
in that order. Each packet becomes visible at its RX finish, and the result's
completion is the latest visible packet. The policy names are
`extent_sequence` reassembly and `per_extent` delivery.

The candidate does not expose a separate completed-extent object. Its exact
packet ledger, sequence check, separate direction totals and final completion
time are the current conservation surface.

## Analytic-bypass contract

No profile means no packet-domain side effect. `NvlinkDomainService` returns
the caller's `analytic_result` by Python object identity, not a copy or a
numerically reconstructed equivalent. Thus enabling the service seam with no
candidate selected cannot change bytes, timestamps, ordering, provenance or
any other field held by the analytic result.

With a profile selected, `include_switch=False` is a module-level diagnostic
bypass. It passes the transmitted packet tuple directly to RX. For the A100
pass-through profile, including or excluding the switch is exact tuple
identity. This is distinct from the profile-absent analytic bypass, which
returns before packetization and preserves the caller's original result.

## Frozen case-to-parameter identification map

The merged freeze is
`examples/a100_nvlink_packet_v1/expectations.json`, with the SHA-256 recorded
above. Its 80 cases are five families of 16. Case names describe interventions;
they do not confer an evidence class. A parameter is identified only if the
required observed counters, ordering records, controls and frozen decision
rules are satisfied.

| Frozen family | Cases | Parameters the freeze can identify |
|---|---|---|
| Packetization | `CORNER_NVPKT_001` to `016` | TX maximum payload, header and packet granularity; request and response direction in the peer-write, peer-read and producer controls; RX reassembly; switch pass-through byte identity |
| Bond and wire | `CORNER_NVBOND_017` to `032` | Links per peer, per-link rate, earliest-available bonding, endpoint egress and direction independence; destination fan-in also separates RX ingress from TX egress; switch identity |
| Incast and destination FIFO | `CORNER_NVINC_033` to `048` | TX destination-credit scope; RX ingress rate, effective buffer or merge scope and delivery order; switch identity |
| Credit depletion and return | `CORNER_NVCRD_049` to `064` | TX credit unit, destination window and destination scope; RX return latency and effective capacity; request and response controls; switch identity |
| FIFO partition and head of line | `CORNER_NVHOL_065` to `080` | TX egress-queue scope and bond policy; RX ingress-queue or merge scope and delivery order; same-pair, other-peer, incast, direction and memory-region controls localize blocking; switch identity |
| Direct-mesh invariant | all 80 cases | Switch pass-through may only retain exact bytes, time and order; no A100 case can identify a physical switch FIFO, virtual-channel count, arbitration rule or buffer depth |

The first hardware execution completed all 86 scheduled cells, including the
80 isolated cases, five ordered corner frames and the all-corners frame. Its
row schema did not record the observed per-row raw and data deltas, per-link
and per-direction counters, recovery and replay deltas, destination-byte
checksum or ordering ledger required by the freeze. Several sweep controls
were parsed but not applied. Under the frozen fatal-guard rule, the whole
capture is `COMPLETE_VOID_86_OF_86` and identifies no TX or RX parameter.

## Current evidence-class table

| Surface | Current value or claim | Evidence class today | Consequence |
|---|---|---|---|
| TX packet geometry and direction | 256-byte maximum payload; 16-byte header; write data in request; read control in request and read data in response | **DECLARED CANDIDATE**, `declared_candidate_not_hardware_measurement` | No packet-format or direction parameter is calibrated |
| TX bond and rates | Four links per peer; 25 GB/s per link; 300 GB/s endpoint egress; earliest-available packet striping | **DECLARED CANDIDATE**, `declared_candidate_not_hardware_measurement` | The published rates constrain an envelope but do not identify bonding |
| TX credits | 256 per destination; 272 bytes per credit | **DECLARED CANDIDATE**, `declared_candidate_not_hardware_measurement` | Effective unit and window remain unmeasured |
| A100 switch | Exact pass-through with zero byte and time effect | **STRUCTURAL DIRECT-MESH INVARIANT, NOT MEASURED** | It stands because the selected topology has no switch, not because TRAF-65 measured it |
| NVSwitch-class switch | Queued interface with input, output or shared placement, FIFO arbitration and optional head-of-line partitioning | **PARAMETERIZED INTERFACE, NO SHIPPED PROFILE VALUES** | H100 and GH200 need architecture-specific queue and service evidence |
| RX ingress and delivery | 300 GB/s ingress; 1 MiB capacity; 200,000 ps credit return; extent-sequence reassembly; per-extent delivery | **DECLARED CANDIDATE**, `declared_candidate_not_hardware_measurement` | No RX rate, capacity, return or ordering parameter is calibrated |
| Ordered-pair envelope | 94.056 GB/s composed candidate against 94.00 to 94.07 GB/s measured | **PUBLISHED-MEASUREMENT CHECK, NOT IDENTIFICATION** | Passes the registered envelope check; cannot distinguish packet overhead from copy-engine coalescing |
| Three-way fan-out envelope | 281.699 GB/s composed candidate against 281.65 GB/s measured | **PUBLISHED-MEASUREMENT CHECK, NOT IDENTIFICATION** | Passes the registered envelope check; does not identify endpoint or ingress queue service |
| TRAF-65 hardware capture | `COMPLETE_VOID_86_OF_86` | **VOID CAPTURE, NO MEASUREMENT EVIDENCE** | Candidate values and classes remain unchanged; TRAF-65 remains open |
| TRAF-70 corrected capture | Corrected identification capture in progress | **PENDING IDENTIFICATION EVIDENCE** | It is the gate before any candidate parameter or evidence class can move |

The two envelope checks use 524,288 payload bytes per destination. The
composed candidate reaches 94.05638991723997 GB/s for one ordered pair and
281.6991815868504 GB/s for three-way fan-out, within the registered 10 percent
relative-error limit against the measurements published before TRAF-65. Those
checks validate only that the declared composition falls inside the known
envelope. They do not select the composition's internal explanation.

## Nonclaims and next gate

This documentation study performs no simulation, scored comparison or
registry transition. It does not change TRAF-65, TRAF-70 or the candidate
profile. It also does not treat the H100 or GH200 switch path as populated.

TRAF-70 owns the corrected capture procedure. Until its frozen guards are
decidable and its required observed ledgers are complete, every numeric TX and
RX value in Figure 1 remains a declared candidate. Later integration must
also preserve the analytic identity bypass and connect packet-domain
timestamps to the repository's shared queue-visit and live-metric contracts
before the mechanism can support a TTFT or TPOT precision claim.
