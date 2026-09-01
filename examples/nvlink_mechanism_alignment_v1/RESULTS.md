# TRAF-80 NVLink mechanism alignment result

## Outcome

What ran: the expectations-only TRAF-80 study selected the aligned NVLink
authority for one fixed 1 MiB peer write across a two-by-two sweep of packet
occupancy and link rate, then checked replay, credit ownership, direct-mesh
identity and every inherited consumer lock. No cluster or hardware ran.

What came out: the final run is **PASS**. Repeated 17-flit packets deliver the
frozen 94.117647 GB/s link payload rate, and repeated 18-flit packets deliver
88.888889 GB/s, on four 25 GB/s links. The extra flit increases the exact link
serialization term by 5.882352941176471 percent. All 13 fatal guards pass. The
first attempt remains recorded as **VOID** because it compared a historical
source lock to the implementation file that this task was authorized to
evolve; the corrected check finds that exact locked blob in Git history and
still verifies all current result artifacts and executable behavior.

What it changes: the technical acceptance for TRAF-80 is literal. The
three-module domain now has an explicit aligned authority with
generation-scoped flits, link acknowledgement and replay, traffic-class and
virtual-channel state, receiver-owned credit release, ordered visibility, and
NVSwitch input-port, virtual-output-queue and two-sided crossbar service. Every
inherited envelope remains on the authority it ran under and publishes a signed
zero shift. The integrator can close TRAF-80 and unblock TRAF-73 when this
implementation lands with the generated registry projections.

What it does not change: TRAF-73 stays open. TRAF-80 also stays open in this
worker branch. The task-progress block and module-status count in the
integrator-owned developer README need the same landing commit as the registry
closure. No A100
credit quantum, pool scope, virtual-channel count, buffer depth, credit-return
encoding, striping granularity or product arbiter is promoted. The H200
collective family, MiniMax studies, deployment-curve scored lineage, hardware
records, profile defaults and time-to-first-token or time-per-output-token
claims do not move.

## Structural replacement

One run selects one mutable packet and timing authority:

- `simllm-htsim-nvlink-domain-v1` is the compatibility authority. It retains
  the merged packet cursor and owns every inherited study that ran on it.
- `simllm-htsim-nvlink-domain-v2` is the aligned authority. It owns flit
  packetization, link reliability, receiver capacity, ordering and switch
  state. The compatibility ledger is disabled in this mode.

The aligned TX module records one 16-byte family flit as the physical unit. It
counts header, address extension, byte enable, payload and padding flits
separately. Every packet carries transaction direction, traffic class, virtual
channel and ordering domain. An acknowledgement releases the replay-buffer
entry. Explicit error injection adds a replay of the same wire occupancy.

The RX module admits wire bytes into a destination and virtual-channel buffer.
It alone records when those bytes release. Sender credit becomes usable only
at or after that release plus the declared return transport. A fixed-point
solver couples the resulting release times back to link selection without
creating a second authority. Ordered visibility waits for earlier packet
sequence members even when physical arrivals reorder.

The direct NV4 path still returns the exact packet tuple and has zero switch
grants. The queued NVSwitch path separates each input, virtual channel and
output into a virtual output queue. A grant interval uses one input and one
output at most once. The identity policy ignores class labels and retains
baseline timestamps, bytes, random draws and completion order. Round robin is
available only as a declared candidate policy.

## Physical sanity and frozen oracles

The first-principles link floor is wire bytes divided by aggregate link rate.
Four 25 GB/s links provide 100 GB/s of directional wire rate. The conservative
ceiling serializes the link, measured endpoint-egress and measured
receiver-ingress services without overlap.

| Flits | Per-link rate | Link payload rate | Link serialization | Job completion | Frozen JCT bounds | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 17 | 12.5 GB/s | 47.058824 GB/s | 22,282,240 ps | 22,288,630 ps | 22,282,240 to 34,590,516 ps | PASS |
| 18 | 12.5 GB/s | 44.444444 GB/s | 23,592,960 ps | 23,599,727 ps | 23,592,960 to 36,625,253 ps | PASS |
| 17 | 25 GB/s | 94.117647 GB/s | 11,141,120 ps | 11,147,510 ps | 11,141,120 to 23,449,396 ps | PASS |
| 18 | 25 GB/s | 88.888889 GB/s | 11,796,480 ps | 11,803,247 ps | 11,796,480 to 24,828,773 ps | PASS |

Halving link rate doubles both serialization terms exactly. At 25 GB/s per
link, the optional flit shifts job completion by +655,737 ps. At 12.5 GB/s it
shifts job completion by +1,311,097 ps. Both signs match the frozen
nonnegative direction.

The clean replay probe adds 0 bytes and 0 ps. One injected replay adds 272
wire bytes and 10,980 ps. Every one of the 4,096 packets in each sanity cell
has one acknowledgement and one receiver-owned credit-release record. No
credit becomes available before its buffer release. Random draws remain zero.

## Pinned consumers and signed envelope shifts

The study verifies 22 root artifacts and 95 recursively inherited preservation
artifacts with zero failures. The historical implementation source lock resolves
to commit `9898a66dc215fd853d10492c6b852009326e376e`; current expectations,
results, reports and figures retain their exact digests. The full repository
suite exercises the compatibility paths without rewriting any frozen record.

| Inherited envelope | Pinned authority | Published signed shift |
|---|---|---|
| Flow dynamics | compatibility v1 | +0 ps and +0 GB/s at every published coordinate |
| First transport comparison | compatibility v1 | +0 ps in every flow-completion-time coordinate |
| Corrected transport and incast mesh comparison | compatibility v1 | +0 ps and +0 in every fairness coordinate |
| NV4 incast validation | compatibility v1 | +0 ps and +0 GB/s in every frozen prediction |
| Analytical frontier and two-network bottleneck map | compatibility v1 | +0 ps at all 18 intra-node points and +0 classifications |
| Deployment-curve run 3 pricing | analytic non-packet path | +0 ps and +0 price; A100 packet substitution remains rejected |

## Evidence and chronology

The expectations authority is commit
`c589abadcfe7d142ffeee3a38db9f9d0a1dc23c8`; its SHA-256 digest is
`fafe9bbe730d9c424f7d4f72fd2df3d5fa2cdd8ad9f370f63ff60194374c58cc`.
It precedes the aligned implementation and all four attempts.

Attempt 0001 is retained with digest
`a7839a0c25af80761bcb7624a70325f89bc04b5b9ac2afb866655f8c26d62c34`.
It is void with one preservation finding and closes nothing. Attempt 0002
passes with the expanded preservation evidence. Attempt 0003 passes with the
compact publication projection. Attempt 0004 is the byte-identical final
publication at `results.json`, digest
`579acea13bd8c899e1c7c00a752dc23397136d44a95827abccf6a8294283e32d`.

Fatal guards are unscored preconditions. The packet-occupancy and link-rate
relations are behavioral evidence. Artifact digests are preservation evidence.
These classes are reported separately and are not added into one score.
