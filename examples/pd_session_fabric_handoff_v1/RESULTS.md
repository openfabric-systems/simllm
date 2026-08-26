# Fabric-rendered KV handoff result

The bounded packet KV handoff passes every frozen guard and relation. Four
live vLLM session cells conserve every byte and endpoint through GOAL and the
packet backend, charge the 20,000,000 ps PCIe term before packet service, and
complete at the last required arrival. The maximum signed TTFT residual is 0
ps and decode TPOT is unchanged exactly. TRAF-62 remains open only because its
registered PLACE-5 dependency is not literal; TRAF-64 owns that topology-
qualified closure.

## What ran

One eight-rank vLLM prefill engine and one eight-rank decode engine shared the
CORE-51 session clock. The explicit packet arm crossed 8-token and 16-token
contexts with 200 and 400 Gbit/s endpoint links. Each prefill local rank sent
one KV shard to the same decode local rank, giving pairs `(0, 8)` through
`(7, 15)`. The same session also ran the accepted 100,000,000 ps constant arm
and zero-duration off arm as controls.

The packet arm rendered GOAL with eight parallel 20,000 ns source calculations
before eight sends, converted it with the executable tracked at the frozen
htsim gitlink, and ran the pinned `rnic-nn` packet backend directly. The raw
result is retained with SHA-256
`311d20615e49e5d8610bd519650b4e818179408d1784ebdf6380d0f8b5889189`.

## What came out

All fatal guards held. Each request has eight positive chunks and eight flow
completions. The 8-token rows carry 393,216 bytes as eight 49,152-byte chunks;
the 16-token rows carry 786,432 bytes as eight 98,304-byte chunks. Sources 0
through 7 and destinations 8 through 15 each appear exactly once in both GOAL
messages and backend rows. Every backend endpoint, tag and byte count joins to
exactly one rendered message.

| Context | Link | Packet service | Complete handoff | Packet TTFT | Constant TTFT | Signed difference | TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 200 Gbit/s | 4,163,200 ps | 24,163,200 ps | 197,539,200 ps | 273,376,000 ps | -75,836,800 ps | 77,952,000 ps |
| 8 | 400 Gbit/s | 3,081,600 ps | 23,081,600 ps | 196,457,600 ps | 273,376,000 ps | -76,918,400 ps | 77,952,000 ps |
| 16 | 200 Gbit/s | 6,160,000 ps | 26,160,000 ps | 219,072,000 ps | 292,912,000 ps | -73,840,000 ps | 77,976,000 ps |
| 16 | 400 Gbit/s | 4,080,000 ps | 24,080,000 ps | 216,992,000 ps | 292,912,000 ps | -75,920,000 ps | 77,976,000 ps |

For every row, `packet TTFT - constant TTFT` equals `packet handoff duration -
100,000,000 ps` with 0 ps residual. The signed difference is negative because
the bounded packet arm completes in 23.08 to 26.16 microseconds, below the
accepted 100 microsecond declared comparator. The sign is part of the frozen
relation and is not converted to an absolute difference.

The constant and off arms changed the packet-artifact count by exactly zero.
Only the four packet cells emitted request directories, each containing one
GOAL text, GOAL binary, completion CSV and packet manifest. All six tracked
CORE-51 artifact SHA-256 values remain exact.

## Physical sanity

At 400 Gbit/s, one 49,152-byte shard cannot serialize faster than 983,040 ps;
the observed 8-token service is 3,081,600 ps. One 98,304-byte shard has a
1,966,080 ps floor; the observed 16-token service is 4,080,000 ps. At 200
Gbit/s those floors double to 1,966,080 and 3,932,160 ps, below the observed
4,163,200 and 6,160,000 ps. Every row is also far below its deliberately broad
57,864,320 to 81,457,280 ps ceiling.

Halving bandwidth increases service for both contexts. Doubling context
increases service at both bandwidths. Neither relation is exactly twofold
because packet framing and fixed overhead remain visible. The 20,000,000 ps
PCIe term is outside packet service and appears once on the request critical
path, not eight times.

## What it changes for the project

The packet mechanism, GOAL and flow projection, PCIe ordering, last-arrival
completion and live TTFT effect required by TRAF-62 are delivered for the
one-plus-one role-aware cell. TRAF-64 is registered for the remaining
PLACE-5-dependent target-topology qualification. TRAF-62 and its TRAF-61
umbrella stay open until that dependency lands, so no topology claim is
silently inferred from the passing bounded cell.

## What it does not change

This result does not calibrate KV transfer against hardware, validate the
448-rank physical target, add backend transport behavior, change decode TPOT,
or change the accepted constant and off arms. It does not close PLACE-5,
TRAF-61 or TRAF-62.
