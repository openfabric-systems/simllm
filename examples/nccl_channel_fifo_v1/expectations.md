# Channel FIFO execution: frozen expectations

This file freezes the first executable slice before implementation or a new
study. The companion JSON fixes the model-side matrix. It does not contain
measured values or fitted costs. Previously published captures remain
retrospective diagnostics. This is local prior registration, not a claim of
public preregistration.

## Source and scope

NCCL source commit `7b83616df3ae082a1f32bb74c27458bfe8153a13` (2.31.2):
`src/device/all_reduce.h`, `prims_ll.h`, `prims_ll128.h`, `prims_simple.h`,
`src/include/device.h`, `src/transport/p2p.cc` and `src/enqueue.cc`.
The existing analytic protocol profile is the explicit compatibility bypass.
Implement Ring float32 sum on two/four-rank direct peer meshes. Connection
identity persists across calls and protocols. Expose protocol, realized channel
partition, threads and buffer mode explicitly. Scope the first timed path to
buffered peer writes, inline LL/LL128 flags, and Simple separate counters.
Selected direct-read, registered, proxy and switched paths reject until their
own source dependencies exist. They remain TRAF-54 work. Source conformance
is distinct from matching measured GPU latency.

Use the existing physical event calendar, not an independent network clock.
GPU resource completions and protocol callbacks schedule events on that
calendar. Reserve source software slots separately from transport credits.
Keep producer reservation, receiver consumption and cached returned head
separate; modulo addressing never replaces absolute sequence numbers.

## Physical bounds before accuracy

Floor: no directed physical attachment can serve more bytes than its rate
times its service interval; useful collective bytes cannot precede their
source, flag/counter visibility or final required output writes.
Ceiling: no finite latency ceiling follows without bounded stalls. Controlled
fixtures below bound every delay and resource service explicitly.

Costs in this first model-side study are declared intervention inputs. They
are not A100/GH200 calibrations. Plot their units and provenance. Do not reuse
old composite coefficients as independently measured instruction costs.

## Fatal guards, unscored

Source identity, unique calendar authority, monotone absolute sequences,
reserved minus returned capacity at most eight, no overwritten live slot,
source-correct partial payload and encoded traffic, matched send/receive
intervals, no consumption before readiness, publication before reuse, exact
byte and operation membership, output completion before collective completion,
no pending required GPU work at completion, and exact disabled-path identity.
An unsupported selected branch must fail before admission. Invalid input must
not alter a retained session. Any violated guard voids the affected run.

LL128 source warp stripes contain 1920 useful bytes in 2048 encoded bytes.
The source sends the whole stripe for a partial participating warp. The
120/128 line ratio is only a floor, not exact partial-warp work. LL encodes
each participating 8-byte pair into 16 bytes. Empty Simple slices reserve
and publish their prescribed progress without inventing payload bytes.
LL flag-wrap cleanup and Simple constructor alignment need explicit fixtures.

## Behavioral relation families

1. Isolated serialization: rates 0.5, 1 and 2 times baseline change only byte
   service inversely, within one picosecond of each independently rounded
   service. Keep propagation and executed GPU work unchanged.
2. Residency: identical independent block jobs on a declared one-block-per-SM
   fixture complete in exactly `ceil(channels / available_sms)` service waves.
   This is a fixture, not a universal NCCL occupancy assumption. General
   admission respects block, warp, register and shared-memory limits.
3. Shared work: fixed per-channel work increases aggregate requested GPU and
   memory service in exact proportion to channels. A shared serializer cannot
   gain capacity from additional channel labels. Fixed total work can fall
   initially as independent SMs participate, then saturate.
4. Finite window: with eight steps of useful capacity K and reuse delay R, a
   deliberately window-limited producer has steady useful rate at most K/R.
   The ninth one-step reservation must await a returned head. A two-step
   Simple reservation needs two free steps. Delay only causal dependants.
5. Protocol state: sequential LL, Simple and LL128 calls share connection
   progress. Simple aligns to four steps, LL/LL128 do not reset it. Flags,
   buffer selection and post operations follow the selected branch.
6. Tails: source payload boundaries plus/minus four bytes change useful byte
   counts exactly. Padding is protocol work, not application work. Retain all
   participating partial channels and the empty Simple control slices.
7. Live metrics: use original execution graphs with two collectives per step
   and fixed compute. In the serial fixture, time-to-first-token and
   time-per-output-token changes equal the sum of changed collective times
   exactly. Change physical rate and available SMs independently. Packet and
   GPU evidence join the original operation; no analytic collective charge
   remains on the structural path. The explicit bypass is byte-identical.

Each family has its own independent closed-form oracle or source fixture.
Count parameter instances separately from families and fatal guards.
Trace the realized critical path; summed resource waits are work accounting
and must not be relabeled as additive token latency.

## Identification and validation chronology

The design document's channel, protocol, warp, resource, receiver-delay and
timing contrasts remain the hardware plan. Before any executable hardware
pilot or measurement, freeze its concrete manifest and capability/rejection
rules in a separate expectations-only commit. Read-only SSH, scheduler and
source inventory can precede that freeze. Do not turn unavailable SM controls
into assumed controls or substitute channel count for available SM count.

First publish the controlled model relations and diagnostic plots for review.
Then identify costs with realized controls and publish sensitivity rank and
unresolved parameter combinations. Lock parameters and bands before inspecting
new validation observations. No hardware-accuracy claim follows from these
model-side checks. TRAF-54 and TRAF-43 remain open until their registered
source, live-path, identification and accuracy obligations hold.
