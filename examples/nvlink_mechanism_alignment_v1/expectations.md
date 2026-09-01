# TRAF-80 NVLink mechanism alignment freeze

This file and `expectations.json` are the expectations-only authority for the
TRAF-80 structural replacement. They precede the aligned implementation and
the first simulation run. The mechanism source is the TRAF-79 public-document
record. No cluster, model weight or web access participates.

## Authority boundary

Every run selects one packet and timing authority.

- The `simllm-htsim-nvlink-domain-v1` compatibility authority owns every
  inherited merged consumer. It must reproduce the accepted canonical bytes.
- The `simllm-htsim-nvlink-domain-v2` aligned authority owns the new flit,
  link-reliability, receiver-credit, ordering and switch ledgers. The sanity
  study selects this authority explicitly.
- The two authorities never advance the same packet. Selecting one disables
  the other.

The profile-absent path returns its input object by identity under both
selectors. The direct NV4 switch returns the exact packet tuple without adding
time, bytes, state or random draws. An identity arbitration seam must preserve
every accepted timestamp, wire byte, random draw and completion order.

## Candidate boundary

The aligned structure carries explicit parameter provenance. It does not turn
the following unidentified A100 or NVSwitch values into hardware facts:

- credit quantum, pool scope or pool depth;
- virtual-channel count or traffic-class map;
- receiver-buffer depth;
- credit-return encoding or transport;
- bonded-link striping granularity;
- deployed product arbitration.

Every default for one of these values is labeled `DECLARED_CANDIDATE` at its
definition and cites the TRAF-79 evidence boundary. TRAF-73 remains the
measurement owner.

## Packet and reliability oracles

The documented family unit is one 16-byte flit. The sanity job repeats 4,096
packets, each with a 256-byte payload, for one fixed 1,048,576-byte posted peer
write over four links.

At 25 GB/s per link, four links expose a 100 GB/s aggregate wire ceiling:

- 17 flits occupy 272 bytes. The exact payload ceiling is
  94.11764705882354 GB/s and publishes as 94.117647 GB/s, with a tolerance of
  plus or minus 0.0000005 GB/s. The repeated-packet link serialization term is
  exactly 11,141,120 ps.
- 18 flits occupy 288 bytes. The exact payload ceiling is
  88.88888888888889 GB/s and publishes as 88.888889 GB/s, with the same
  tolerance. The repeated-packet link serialization term is exactly
  11,796,480 ps.
- The optional flit raises serialization by exactly 5.882352941176471 percent,
  which publishes as a nonnegative 5.882 percent shift. Payload throughput
  shifts by exactly -5.555555555555555 percent.

At 12.5 GB/s per link, the corresponding serialization terms are 22,282,240
ps and 23,592,960 ps. Halving link rate doubles the serialization term exactly.

The 25 GB/s job-completion bounds are 11,141,120 through 23,449,396 ps for the
17-flit job and 11,796,480 through 24,828,773 ps for the 18-flit job. The
12.5 GB/s bounds are 22,282,240 through 34,590,516 ps and 23,592,960 through
36,625,253 ps. The floor is bytes over aggregate physical link rate. The
ceiling serializes the link, declared endpoint-egress and declared
receiver-ingress services without overlap.

Every credit becomes returnable only when its owning receiver buffer releases.
Any modeled return transport follows that event. Error-free acknowledgement
and replay add exactly zero bytes and zero time. An injected error may add
bytes and time, but neither quantity may be negative.

## Sanity study

The study crosses packet occupancy 17 and 18 flits with per-link rates 12.5
and 25 GB/s. It measures the job completion time of the same fixed write in all
four cells. It checks the physical floors and ceilings, exact two-to-one link
rate scaling, the exact optional-flit serialization shift, nonnegative job
completion movement, byte conservation, receiver ownership, ordering and
error-free identity.

## Inherited envelopes

Every inherited consumer stays on the explicit compatibility authority and
must remain byte-identical. The result publishes these signed shifts:

- `nvlink_flow_dynamics_v1`: +0 ps and +0 GB/s at every transition, flow
  completion and incast coordinate.
- `nvlink_rnic_comparison_v1`: +0 ps for every NVLink flow completion.
- `nvlink_rnic_comparison_v2`: +0 ps and +0 fairness movement at physical and
  simulated mesh degrees.
- `nvlink_incast_validation_v1`: +0 ps and +0 GB/s in all six frozen
  predictions.
- `deployment_frontier_v1`: +0 ps at all 18 intra-node points and +0
  bottleneck classifications.
- `deployment_curve_v1` run 3: +0 ps and +0 price. Its H100 substitution of
  the A100 packet candidate is already frozen rejected, so the aligned path is
  unreachable there.

The root pins in `expectations.json` lock each expectation, result and report.
The runner also expands every preservation lock reachable from the five study
expectation roots and verifies every artifact digest without executing or
rewriting any prior study.

## Void rule

Conservation, authority, ownership, ordering, identity and physical bounds are
fatal guards. One violation makes the result VOID. A void result reports its
findings, closes nothing and publishes no inherited shift as valid.
