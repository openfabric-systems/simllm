# Causal NVLink service expectations

TRAF-90 replaces the aligned packet domain's iterative batch scheduling with
one causal event calendar. This file and the adjacent parameter matrix are
committed before implementation and before this study's first execution.
The study qualifies model correctness, not an A100 or NVSwitch calibration.
TRAF-45 separately owns connection to the supported request metric chain.

A sender cannot consume a receiver buffer that has no space. A response cannot
exist before its request becomes visible at the target. A future submission
cannot reserve an earlier idle link. These are causal constraints independent
of the undisclosed NVLink credit encoding or arbitration implementation.

## Frozen model boundary

Use the existing generation-scoped 16-byte flit packetizer and declared
profile interface. Writes carry 256-byte payload packets and one header flit.
A read has a 16-byte request followed by response data. Its additional target
processing time is explicitly zero, a lower-envelope assumption rather than
measured memory service. Request visibility gates response eligibility.

The enabled engine has one event calendar. Direct links reserve destination
receive capacity before transmission. A queued switch instead owns the first
hop's buffer and credit return; its output reserves the destination receive
capacity before a crossbar grant. Each reservation covers in-flight and
resident bytes until downstream release; advertised space returns after the
declared return latency. Separate incoming links retain separate credit
pools. The candidate receive byte pool is shared by destination and virtual
channel. The switch byte pool is per input, virtual channel and destination.
These pool scopes are explicit hypotheses, not product identification.

Every queue keeps a deterministic baseline order. Mandatory causal, credit,
capacity and port-legality gates filter candidates before arbitration. The
identity policy ignores class labels. Distinct sources may transmit at the
same time; the two directions of a pair have distinct link service. Packets
retain byte, extent, link, virtual-channel, replay and visibility identity.
No capacity or credit appears outside a recorded owner.

## Independent exact oracles

The direct oracle uses one link per direction, 100 GB/s endpoint egress and
100 GB/s receiver service. Sweep per-link rates 12.5 and 25 GB/s and payload
lengths 256 and 1024 bytes. Let N be payload length divided by 256,
L = ceil(272 * 1e12 / link_rate), R = 2720 ps,
Q = ceil(16 * 1e12 / link_rate) + 160 ps, and D the return latency.
Use integer arithmetic independently of the production serializer.

- With four credits and four-packet byte capacity, write completion is
  N * L + R. Read completion is Q + N * L + R.
- With one credit or a one-packet byte capacity, completion is
  N * (L + R) + (N - 1) * D. Sweep D at 0 and 10000 ps. Changing byte
  capacity alone must enforce this same bound when the credit count is four.
- Doubling the link rate halves each link serialization term exactly; it
  leaves receiver serialization unchanged. Quadrupling payload multiplies
  only the repeated packet terms by four. Compare the terms, not an assumed
  fourfold wall-time ratio with a fixed pipeline tail.

Floor: completion is at least all wire bytes divided by the active pair's
link rate, and a read also waits for its request and target acceptance.
Ceiling: in these no-background cells, fully serial service of every packet
through link and receiver, with one declared return between packets, bounds
completion from above. For the queued probe add one switch serialization per
packet to the serial ceiling. Bounds are specified before any study value.

## Behavioral families

1. Future insertion: submit a one-packet early write and an independent
   future extent from the same source, in both caller orders, at offsets
   1000000 and 2000000 ps. The early extent's complete packet timing equals
   its isolated execution; the later extent never starts before its release.
2. Read dependency: at both rates and payloads, every response grant is at
   or after request visibility, and the exact read oracle holds. A write
   contending at the target cannot let a response bypass that dependency.
3. Backpressure: across the rate, payload, capacity and return grids above,
   receive capacity alone can cause the exact one-packet recurrence. A queued
   switch with one-packet buffers and a slower receiver completes without
   overflow, fictitious unbuffered arrivals or iterative nonconvergence.
4. Independent resources: adding a disjoint transfer changes none of an
   extent's packet times. Opposite directions can start together. In a
   simultaneous fan-in, destination ingress visits never overlap.
5. Replay: one injected replay adds exactly one packet's wire bytes and a
   nonnegative link delay, consumes one receive reservation, returns each
   credit once, and cannot expose duplicate payload or visibility.

These are behavioral relation families; their parameterized instances are
reported separately from exact-oracle rows. No count is merged with another
evidence class.

## Fatal guards and controls

Any duplicate or missing packet, byte disagreement, early response, future
reservation, timestamp reversal, overlapping exclusive grant, capacity or
credit violation, unowned arrival, double return, missing terminal, nonfinite
drain or identity-policy class dependence voids the entire enabled study.
A void run has no interpretable behavioral score and closes nothing. Keep its
evidence and fix the defect before an independently identified repetition.

The profile-absent route returns the exact caller object. The compatibility
authority preserves its packet bytes, timestamps, order and accounting for
writes, reads, simultaneous fan-in and staggered releases. Capture reference
bytes by importing the frozen base Git blob, not by rewriting historical
artifacts. The old aligned implementation remains historical evidence and is
not an exact control for the defective cases. Existing uncontended aligned
serialization oracles remain physical checks. The obsolete fixed-point
iteration diagnostic becomes zero when the event engine is selected.

The report gives the freeze commit, source identity, configuration count,
exact-oracle rows, behavioral families and instances, and fatal verdict as
separate evidence. It states which correctness task closes and that TRAF-45,
TRAF-73, TRAF-86 and end-to-end deployment qualification remain independent.
