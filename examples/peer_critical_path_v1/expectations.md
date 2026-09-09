# Local packet critical-path expectations

This expectations-only commit precedes implementation and campaign execution.
It defines BACK-73 reporting over the existing retained physical peer-write
path. The calendar continues to own every packet, grant, capacity reservation,
credit return and consumer arrival. A recorder stores immutable projections
of those decisions. It never changes eligibility, selection, event ordering,
service, packetization or completion.

## Causal contract

Every recorded event has a stable identity, timestamp, packet and physical
resource join. A positive service or propagation interval has one predecessor
at its actual start. A zero-duration grant or visibility join names its actual
gating predecessors, whose latest timestamp equals the join timestamp exactly.
Every predecessor is recorded earlier in causal order, including equal-time
events. An unrelated wake event is not an explanation for a delayed grant.

Record source eligibility, source-feed and directed-link release, finite
downstream buffer availability, returned link credits, switch input/output
grants, receiver service and ordered consumer visibility. Reuse the calendar's
selected route and grant; do not reschedule from final timestamps. Native
switch grants supply the same observations without making Python's inert
switch-port clocks a second authority. Source feed and link transmission
overlap. Switch forwarding and its output attachment overlap. Only the
limiting branch enters a selected critical chain; ties use a stable declared
order and retain the other co-critical predecessors.

For a completed phase, start at the last required packet visibility and follow
the latest actual predecessor. Clip the chain at that phase's eligibility.
Retained control tails can cross that boundary; preserve their original
packet identities and clip only the report interval. Positive intervals cover
the phase from eligibility to visibility exactly once, with no gap or overlap.
Zero-duration dependency joins remain explicit. Acyclicity, parent totality,
timestamp agreement and exact coverage are fatal unscored guards.

Keep per-resource visit sums separate. Queue wait remains grant minus
eligibility for each visit, and can exceed wall duration when visits overlap.
The report never turns that sum into request latency. The existing step
breakdown uses executed local duration, and its per-request reduction must
agree with the original graph completion and StepResult. Detailed local
predecessors remain joined to that same phase and packet inventory.

Reporting is explicit and has an identity off path. The absent selection
preserves every accepted serialized observation, graph, completion event,
packet field, queue visit, byte, order and request metric. Selected reporting
adds its sidecar and changes no existing data field. Unsupported component
read or replay envelopes reject rather than inventing a partial chain.

## Frozen model grid

Use direct and single-switch physical fabrics, common directional service
rates of 12.5 and 25 billion bytes per second, one or three donors, and
256 or 1024 payload bytes per donor. Physical propagation p is 1000 ps per
hop. All source-feed, switch and receiver rates equal the selected rate.
The existing packet format carries 256 payload bytes in 272 wire bytes.
Buffers are finite at 65536 bytes and the credit pool is 256 units. Additional
credit and acknowledgement processing is zero in this grid.

These sixteen component configurations submit aligned donors to one receiver.
Let m be payload/256 and q be the exact 272-byte serialization in picoseconds.
Before reading an outcome:

- Floor: no endpoint beats its cumulative wire bytes divided by its rate;
  the first packet also requires every serial service and propagation hop.
- Ceiling: serializing all wire-service visits and their propagation bounds
  the complete finite phase; no packet finishes before its own dependencies.
- Exact direct completion is (donors*m+1)*q+p.
- Exact switched completion is (donors*m+2)*q+2*p.

Sixteen live configurations use the same transport grid in the standard step
sink. One request selects every donor's expert. Hidden width is 128 or 512,
with two-byte elements, so each directed expert vector is 256 or 1024 bytes.
The existing fixed provider assigns 37000 ps per step. One prefill and two
decode steps execute attention, dispatch and combine through the accepted
serial compatibility schedule. This study changes reporting, not that
schedule or its compute approximation.

The dispatch fan-out and combine fan-in each have the component completion
above. Therefore TTFT and TPOT equal 37000 ps plus twice that completion,
and last completion equals three such steps. These declared synthetic prices
exercise the request chain and make no physical model-performance claim.

Execute every component and live configuration through the frozen predecessor
at 861d7a45ff2e3b8b8ed438d79ca514960d52e438 with reporting absent, the candidate
with reporting absent, and the candidate with Python reporting enabled.
Additionally execute all eight switched configurations with the native
allocator and reporting enabled. This is 56 component runs and 56 serving
sessions, with 168 actual step calls. Preserve the complete source identity
and loaded origins of both implementations. Bind native source, build command,
compiler and resulting library before its first execution.

Evidence classes remain separate:

- Thirty-two exact oracle vectors cover sixteen component completions and
  sixteen live TTFT/TPOT/last-completion tuples.
- Twenty-four behavioral instances in three families use the live Python
  reporting arm: eight rate comparisons halve the network term after
  subtracting compute and propagation; eight payload comparisons add
  exactly 6*donors*q to TTFT and TPOT; eight donor-count comparisons add
  exactly 4*m*q to TTFT and TPOT.
- All baseline/on/native equalities, path coverage, capacities, byte
  conservation and software corruption controls are fatal and unscored.
  Converging multi-packet cases must expose a resource-work sum greater
  than wall time while retaining exact critical coverage.

## Stress and admission controls

Software tests exercise a one-packet credit pool, one-packet receiver and
switch capacity, long credit processing, unequal source and link service,
slow receiver service, bidirectional/disjoint pairs, visibility reordering,
both existing switch arbitration policies and multi-chip A100/H100 presets.
Retain two or more phases without draining control tails between them.
Every enabled stress result must have complete causal coverage and match its
reporting-off packet and visibility evidence exactly.

Corrupt recorded evidence independently: missing/duplicate node, missing or
forward parent, wrong parent timestamp, invented grant delay, changed packet,
resource or phase identity, missing interval, overlapping interval and altered
terminal visibility. Admission rejects each without publishing a valid score.
Preserve original failure before any reporting failure. Never repair a failed
campaign in place or reinterpret its denominator.

Freeze and source checks precede the first model call. Retain each actual
input, original graph, packet observation, causal report, StepResult and metric
history outside Git. Write an append-only flushed receipt before admitting a
file; recheck its bytes at final publication. Preserve exact initial and final
file domains, source/package origins, all required stage identities and the
first receipt journal. Any fatal failure makes the run VOID with a null
behavioral score, even if other configurations completed.

A valid complete result closes BACK-73 for the supported local packet path.
It qualifies explanations of declared model time, not physical link, buffer,
credit or switch calibration. TRAF-92, BACK-74 and TRAF-54 retain those
distinct obligations. The packet and request timings must remain unchanged.
