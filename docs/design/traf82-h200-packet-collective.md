# H200 packet collective design boundary

## Decision

TRAF-82 owns the packetized H200 intra-node collective that remains after the
aggregate completion authority reaches its accuracy bar. The current H200
source identifies one opaque completion per operation, rank count and payload.
It does not identify credits, ports, queues, switch service or arbitration.
Those internal values cannot be fitted independently from one completion
number, so the packet family remains unexecuted rather than assigning several
mechanisms to the same observation.

TRAF-76 retains the aggregate authority as the sole timing owner until this
design is identified and implemented. The nonzero-fan-in packet path remains
an explicitly transferred local component.

## Frozen comparison cells

| Family | Participants | Receiver fan-in | Payloads per sender |
|---|---:|---:|---:|
| PZ | 2 and 8 | 0 | 65,536 and 1,048,576 bytes |
| PN | 4 and 8 | 3 and 7 | 65,536 and 1,048,576 bytes |

PZ and PN are separate evidence families. Neither has a denominator until an
independently matched H200 observation exists for every frozen cell.

## Required authority

The packet path is one mutable timing authority. It advances a collective's
flits, credits, switch visits, receive visibility and completion. The
aggregate completion row becomes a read-only comparison and contributes zero
service whenever the packet path is enabled.

The path includes:

- generation-scoped 16-byte flits with explicit data and optional control
  flits;
- link-local acknowledgement and replay identity;
- explicit traffic class and virtual channel;
- receiver-owned credit release and receive-order visibility;
- H200 product link and port geometry;
- NVSwitch input ports, virtual output queues and crossbar outputs;
- a two-sided arbitration policy seam whose identity policy preserves the
  deterministic baseline order.

TRAF-80 supplies the generation-independent packet, credit and switch
structure. TRAF-82 identifies the H200 product values, binds that structure to
the collective execution path and validates the frozen cells.

## Evidence needed before implementation

Each numeric credit quantum, credit pool, virtual-channel count, buffer depth,
link and port count, switch queue, crossbar service and arbitration choice
names independent H200 evidence. Product documentation, implementation logs,
counter captures or controlled hardware experiments may identify a value.
The opaque aggregate completion table cannot.

At minimum, the hardware campaign records per-cell phase completion, endpoint
application bytes, link wire bytes, participant placement and receiver
fan-in. Credit and switch counters are required to distinguish endpoint wait
from switch wait. Arbitration needs at least one simultaneous two-input,
two-output case that distinguishes input selection from output selection.

## Acceptance

Before a result-producing run, freeze expected directions, exact relations and
the larger of 10 percent or two H200 GPU cycles for phase completion. For every
PZ and PN cell report:

- aggregate phase completion before and packet phase completion after;
- relative before and after error against the matched H200 observation;
- application bytes and wire bytes;
- completion order, credit waits, switch waits and arbitration grants;
- the single timing owner and every disabled charge.

A fatal guard voids the run if one mechanism lacks independent identity, the
TRAF-80 structure is absent, any byte is lost or duplicated, two authorities
advance one object, or a disabled aggregate charge survives. Two fresh
processes must match exactly apart from a named wall-time field.

The identity off path preserves every accepted pre-wave timestamp,
application and wire byte count, completion order, backend invocation order
and random draw exactly. Class-label permutation also preserves the baseline
when identity arbitration is selected.
