# Disaggregated target topology result

## Outcome

What ran: `examples/disaggregated_target_topology_v1`, the frozen PLACE-5
structural qualification, built both the one-prefill plus one-decode cell and
the 16-prefill plus 40-decode target from implementation commit `faeaae9`,
round-tripped their `simllm-fabric-topology-v1` records, rendered one Group
Operation Assembly Language (GOAL) message per endpoint and resolved every
message through the same physical graph. The expectations-only commit is
`241ba81`.

What came out: the result is `PASS` with no finding. The deciding target count
is 448 of 448 ranks, GPUs and network interface controllers (NICs) conserved
and reachable, with exactly 128 prefill and 320 decode roles. All nine fatal
structural guard classes held. There is no scored behavioral denominator
because PLACE-5 qualifies a declared topology, not packet timing.

What it changes for the project: PLACE-5 closes, which also makes PLACE-4's
remaining physical-location clause literal and closes PLACE-4. TRAF-62's
bounded packet mechanism had already met its exact timing and conservation
bar; its sole PLACE-5 dependency is now literal, so TRAF-62 closes. TRAF-64 is
unblocked and remains the owner of target-scale key-value cache handoff,
packet completion and the live time-to-first-token (TTFT) and
time-per-output-token (TPOT) qualification. TRAF-61 stays open until TRAF-64
qualifies that target path.

What it does not change: this run does not execute a packet backend, move a
TTFT or TPOT, calibrate a fabric, discover live inventory, implement general
`unique-nic` mapping, close TRAF-61 or TRAF-64, or change either backend
repository. The topology's rates and delays are declared, not measured.

## Conservation and topology

| Cell | Ranks | GPUs | NICs | Prefill | Decode | Nodes | Leaves | Spines | Links | Switch ports |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one-plus-one | 16 | 16 | 16 | 8 | 8 | 2 | 2 | 8 | 32 | 48 |
| target | 448 | 448 | 448 | 128 | 320 | 56 | 56 | 8 | 896 | 1,344 |

Every logical rank joined to exactly one unique GPU and one unique affine NIC.
Every NIC joined to one leaf-switch port and one endpoint link. Every leaf had
one 400,000,000,000 bit/s link to each of eight spines. Every physical link
carried the declared 1,000,000 ps propagation delay and the switch latency was
zero. The graph had one undirected component: traversal from rank zero reached
16 of 16 endpoints in the small cell and 448 of 448 in the target, which
proves pairwise reachability for all endpoints.

The serialized fabric records round-tripped byte for byte. The small record
was 21,379 bytes with SHA-256
`36527abf437eb875ce8db097b301bd1c25789f98fd38271378d65f6416361367`.
The target record was 573,301 bytes with SHA-256
`5f372f96b73e9b0c2c632ae4ffd4062f92620c4d1efb2b67c2cd2e0e1da70c75`.

## GOAL reachability

The witness mapped each rank `r` to `(r + 8) mod R` with a 4,096-byte message.
It rendered 16 messages and 65,536 bytes at one-plus-one, then 448 messages
and 1,835,008 bytes at the target. In each cell every rank appeared exactly
once as source and once as destination. Every message resolved to four
physical links through spine zero, with a 400,000,000,000 bit/s bottleneck and
4,000,000 ps aggregate declared propagation.

A 4,096-byte message needs at least 81,920 ps to enter a 400 Gbit/s link. The
frozen no-queue cross-leaf interval was therefore 4,081,920 ps for cut-through
forwarding through 4,327,680 ps for store-and-forward service on all four
links. The graph observation supplied the four-link path, bottleneck rate and
propagation needed by that bound. It supplied no completion timestamp and
made no transport-calibration claim.

## Disabled-path identity

Physical rendering was then disabled through the builder's explicit flag. The
compatibility fabric retained no switches or links. The enabled and disabled
placement JSON bytes were identical to each other and to the pre-change
PLACE-4 anchors:

| Cell | Bytes | SHA-256 |
|---|---:|---|
| one-plus-one | 15,772 | `019f818e02252407e560b37415da12151c8f6ca0ff01bcb7ff8aabfead47f286` |
| target | 2,639,042 | `48029d871293762007ab33082d59a7b5a4efb22583394e718c97e733717fd709` |

The raw run result has SHA-256
`634d74143275acf875aed942d752efcad5bf252d9ac6ac1fe8bd85f17e793c7f`.
Bulk placement, fabric and GOAL artifacts remain outside Git under the
configured PLACE-5 run root. The tracked [result summary](results.json)
contains the exact closure numbers without embedding a machine-specific path.
