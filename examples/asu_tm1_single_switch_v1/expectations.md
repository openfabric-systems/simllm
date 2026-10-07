# Single Tomahawk1 ASU allocation study: frozen expectations

Freeze before implementation and execution. This is a packet-level component
study, not a GPU workload, deployed controller, ASIC calibration or RDMA RC
validation. The paired backend starts at main commit 41f5c60.

## Physical contract

One Cisco Nexus C3232C connects four sending endpoints and one receiving
endpoint. Compare four full-duplex 25 Gb/s lanes per endpoint with one
100 Gb/s lane. All endpoint lanes connect to the same switch. Explicit lane
addresses select physical destination ports; changing a UDP hash does not
create another switch path. Repeat selected rate cases at half link capacity.

Tomahawk1 admission uses 208-byte cells and four independent XPE domains of
20165 advertised cells each. Admit each whole packet into its configured XPE
mask. Expose masks, unavailable pool cells and internal accounting overhead
as inputs, because the installed NX-OS mapping and reservations are unknown.
Sweep zero and 64 bytes of internal overhead and one versus two accounting
domains. Do not subtract physical CFAP reservations from advertised capacity
without a verified accounting relation. Run zero and 4096 unavailable cells
per domain as sensitivities, not measured settings.

The proposed data cap is 262144 bytes for EACH physical destination lane's
data queue, rounded down to 1260 cells (262080 bytes). It is not a cap per
flow or an aggregate cap across all four lanes. Control has its own 8192-byte
queue. A shared alpha=0.5 mode is a separate comparison, not a simultaneous
static-cap interpretation. Occupancy includes the transmitting packet until
its final bit; this conservative model lifetime is explicit.

Packets serialize nonpreemptively. HIGH control precedes LOW data. One output
has one serializer and independent outputs can overlap. Source MAC service
also serializes, preventing impossible simultaneous emissions on one lane.
The store-forward sensitivity receives the complete source frame before
450 ns of switch processing and a second serialization. Cable propagation
is 100 ns in each direction. Actual switch cut-through is known, but this
model must not claim exact cut-through latency from an uncalibrated surrogate.

IP MTUs are 1500 and 9000 bytes. Wire size includes Ethernet header, FCS,
preamble/SFD and IFG: IP size + 38 bytes with no VLAN. Cell charge excludes
preamble/SFD and IFG: ceil((IP size + 18 + internal overhead)/208).
Synthetic application data is UDP with an explicit 36-byte experiment header,
so full data payload is MTU - 64 bytes. No RoCE header or retry behavior is
implied by this synthetic payload.

## Scheduling contract and parameter sweep

Keep allocation window W distinct from rnic-cn one-way deadline K (`dwnd`).
The study supplies a preinstalled demand/calendar from known ASU byte counts.
It does not emulate the reference controller's instantaneous shared ledger
as a causally valid distributed network exchange.

Each sender has 1 MiB of application data for the same receiver. Balanced
lane assignment stripes across the receiver's four addressed lane ports.
Compare W = 8, 16, 32 and 64 us, tick = 8 ns and 64 ns, data wire budget
fractions 0.90 and 0.95, MTU = 1500 and 9000, seeds 1, 7 and 31. Exact
integer packet cost is charged to BOTH source and destination lane budgets.
Carry fractional byte deficits across windows. Rate budgets are wire rates,
not application goodput. Deterministic interleaving is the candidate baseline;
PRBS selects only among legal remaining quota slots with independent
node/lane seeds. Include independent unbounded PRBS lottery pacing, unpaced
incast and a one-destination-lane collision as explanatory controls.
Tick rounds launch times upwards and never permits a launch before eligibility.

## Expected relations and fatal guards

| Family | Expectation before the run |
| :-- | :-- |
| Physical floor | Receiver makespan cannot beat total delivered wire bits / aggregate receiver rate, plus forward propagation and processing. In a dropped run this is a delivered-byte floor, not completion of the original task. |
| Rate scaling | Halving lane rate doubles pure serialization. Fixed cable/processing terms remain constant. Check an unloaded exact-oracle packet separately from loaded calendars. |
| Buffer geometry | No admitted data queue exceeds 1260 cells in static mode. No XPE exceeds its configured available cells. A frame that cannot fit is dropped whole and its allocation is not leaked. |
| Conservation | Generated frames = delivered + dropped + pending at end; all routes drain before destruction. Delivered payload equals requested payload only in loss-free cases. These are fatal guards, not scored behaviors. |
| Port independence | Four balanced 25G lanes have 100G aggregate service but each frame serializes four times slower than on a 100G lane. A single collided 25G destination has at most 25G service and no hash changes can remove that configured collision. |
| MTU | Full 1500/9000 IP packets occupy 1538/9038 wire bytes and serialize in 492.16/2892.16 ns at 25G. Jumbo frames reduce overhead but increase nonpreemptive HIGH blocking by 2400 ns. |
| Control | Isolated HIGH traffic waits at most the residual of one active LOW frame, plus older HIGH traffic and configured pipeline delay. Continuing unpoliced HIGH traffic can starve LOW; finite buffer size is not a LOW residence-time bound. |
| Window | Smaller W cannot create more physical link capacity. W changes allocation granularity, quantization and readiness delay. Jumbo packet service can exceed an 8 us sender share of a four-way incast, so whole-packet deficit carry is required. |
| Quota vs lottery | Exact quotas bound count/allocation error by a packet plus explicit deficit carry. Unbounded lottery has statistical tails and supplies no deterministic cap guarantee. No promised strict improvement of all seeds is made. |
| Optimum | Select a practical configuration from measured utilization, queue headroom, control blocking and quota accuracy. Report tradeoffs; no global mathematical optimality claim is supported by a finite sweep. |

Expected static-buffer drain ceiling with no new arrivals is
262144*8/25e9 = 83.88608 us per lane, or 20.97152 us at 100G, before packet
rounding. A 32 us window budgets 100000 wire bytes per 25G lane at full rate
and 95000 at 95 percent. This does not guarantee safety if independent
senders burst their complete quotas together; the calendar must bound bursts.

Bound the deployed control deadline separately. A candidate reserved/policed
control class can target K = 8 us and W = 32 us, but these are model design
inputs awaiting measurements, not current hardware guarantees. With K_d =
K_g = 8 us, allocation and arm work 1 us each, and guard 1 us, lead is 19 us;
boundary-only declarations target ceil(19/32) = 1 future window. A same-class
control path subject to the full 256 KiB data drain cannot use that K.

## Acceptance and evidence

Backend unit tests must exercise cell boundaries, caps, domain isolation,
nonpreemptive priority, serialization and accounting. Run the complete ctest
suite before pushing. The study runner calls the backend binary directly,
keeps raw CSV and manifests outside Git, emits a compact report, and evaluates
the frozen families without combining oracle, behavioral and fatal counts.
Legacy runtime defaults are untouched. A complete backend suite verifies the
off path. Standard ECN/DCQCN and RDMA RC require their real transport and
notification chain; an ECN marker or this UDP calendar alone does not validate
that comparator. PTP queue isolation does not prove hardware timestamping.
