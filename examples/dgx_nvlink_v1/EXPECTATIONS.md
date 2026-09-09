# DGX NVLink topology and native switch expectations

This freeze precedes implementation and the first run. It qualifies declared
software behavior, not measured A100 or H100 timing. The study changes one
selected peer domain through the existing physical packet and request-metric
interfaces. An absent native selection preserves the existing Python path.

## Physical bounds

A100 has eight GPUs, six parallel switch chips and twelve links per GPU, two
to each chip. H100 has eight GPUs, four chips and eighteen links per GPU,
distributed 4, 4, 5 and 5. Each physical link has a nominal 25 GB/s capacity in
each direction. Thus the endpoint one-way ceilings are 300 and 450 GB/s.
All 56 directed pairs cross exactly one switch chip. Links and destination
capacity are shared across peers; adding ranks never creates physical ports.
Merlin four-GPU direct meshes are different topologies and evidence domains.

A transfer cannot complete before its wire bytes have traversed the bottleneck
attachment. The declared model waits for a complete packet at switch ingress,
then acquires an input and output together. For one uncontended packet of W
wire bytes, endpoint feed F, input rate R, switch rate X, receiver rate D and
one-way per-link propagation P, visibility is exactly
max(ceil(W/F), ceil(W/R)) + P + max(ceil(W/X), ceil(W/R)) + P + ceil(W/D),
where every division is converted to picoseconds before integer ceiling.
This is also a conservative uncontended ceiling for this declared model;
it is not an estimate of the product's internal forwarding mode.

## Configuration matrix

Run both A100 and H100 presets, both Python and native switch selection,
12.5 and 25 GB/s attachment rates, and 256 and 4096 byte payloads. Component
patterns include an isolated pair, bidirectional pairs, disjoint pairs and
fan-in with 1, 3 and 7 donors. Use switch and receiver capacities of 272 and
65536 bytes in separate pressure cases and fixed 256-byte payload packets
with a 16-byte header, explicitly declared rather than identified A100 format.
Exercise identity and rotating-candidate arbitration as model policies.

For live request evidence run tensor-parallel all-reduce at widths 2, 4, 8
and expert-parallel dispatch/combine at widths 2, 4, 8 on a single eight-GPU
board, using both uniform and concentrated routing. Use a fixed compute
provider, one prefill and two decode steps, with a single retained packet
calendar. Compare two attachment rates and both implementation selections.
Do not use a cross-node substitute or measure GPU hardware in this study.

## Scored behavioral relations

1. At identical inputs Python and native grants, packet timestamps, buffer
   visits, credit returns, ordered delivery and request metrics agree exactly.
   Native implementation identity may differ only in its named provenance.
2. The single-packet result matches the independent equation above exactly.
   Halving a serialization rate changes only terms containing that rate.
3. Disjoint input/output ports, including separate switch chips, can receive
   simultaneous grants. Same-output contenders serialize on that output;
   a blocked destination cannot consume another destination's free capacity.
4. Under the fixed source and receiver services, increasing wire rate cannot
   increase phase makespan in isolated and disjoint cases. Fan-in and live
   comparisons record the change without imposing monotonicity on arbitrary
   packet scheduling. At least one transport-limited live case must change
   request latency when rate changes; compute service stays identical.
5. Reducing finite buffer capacity or delaying credit returns creates visible
   backpressure in deliberately credit-limited cases, then makes progress when
   credits return. The next phase retains control events from the prior phase.

## Fatal guards and evidence classes

No packet loss, duplication, negative credit, buffer overflow, early visibility,
shared-port overlap or cross-chip path is permitted. Source and receiver byte
floors hold for every phase, and elapsed times fit unsigned 64-bit picoseconds.
Invalid native inputs reject before any state mutation; native selection must
fail explicitly if its library is missing. The unselected path imports or
loads no native library and preserves existing serialized outputs exactly.
These are fatal guards, not points in the behavioral score. Configuration
rows, structural checks, component relations, native tests and live metric
relations remain separate evidence classes. A violated fatal guard voids
closure; results are retained and an amended freeze must precede another run.

## Closure boundary

PLACE-6 owns the product wiring; BACK-74 owns native switch development;
TRAF-92 retains hardware qualification. This slice implements switch port
allocation and service in C/C++ while reusing the existing Python-owned
queues, credits and event calendar. It does not claim a fully native GPU
endpoint, NVIDIA RTL equivalence, calibrated switch depths, H100 reduction
offload, or completion of the remaining collective protocol and attribution
work. Register residual work in the owning module before merging.
