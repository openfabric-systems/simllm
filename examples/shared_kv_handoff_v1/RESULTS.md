# Shared packet handoff through native serving

**All 40 requests qualify in the complete 18-process retry.** Independent
prefill engines submit modeled key/value-cache shards to one persistent
packet runtime. Decode waits for its last required shard, so receiving-port
contention changes the live time to first token (TTFT). CORE-71 closes within
the frozen ideal endpoint and simulated-worker envelope.

At 200 Gbit/s, two eight-token requests sharing a receiving port complete
their shards in 5.9936 and 6.1600 microseconds. Separate receiving ports take
4.1632 microseconds for both requests. Sharing therefore delays the earliest
first token by exactly 1.8304 microseconds. The extra delay comes from packet
serialization and reaches the request through the actual decode release.

## Physical bounds and observed schedules

Each request carries 393,216 or 786,432 cache bytes across eight rank pairs.
A shard contains twelve or twenty-four packets, each with 4,096 payload bytes
and 64 header bytes. One full packet takes `q = 4160 * 8 / rate` seconds,
or 0.1664 microseconds at 200 Gbit/s and 0.0832 at 400 Gbit/s. Submission adds
20 microseconds and propagation adds two microseconds. Neither scales with
link rate.

Floor: an isolated shard requires `(n + 1) * q + propagation` after
eligibility, including the last packet's destination serialization. Every
receiver's earliest-completion prefix must also accommodate all its completed
wire bytes. Ceiling: all sixteen shards serialized through one link, plus one
startup packet and propagation, bounds this finite loss-free endpoint graph.
These bounds were frozen before the campaign.

The actual disjoint-port durations equal the isolated floor. With two equal
transfers sharing a receiver, the expected completion envelope is
`2 * n * q + propagation` through `(2 * n + 1) * q + propagation`.
The two captured durations attain those endpoints. Each request waits for
the maximum of its eight shard completions, without summing parallel work.

| Decode engines | Prompt tokens | Rate (Gbit/s) | Shard FCT, request 1 / 2 | TTFT, request 1 / 2 | Last request completion |
|---|---:|---:|---:|---:|---:|
| 1, shared ports | 8 | 200 | 5.9936 / 6.1600 | 199.3696 / 511.1776 | 745.0336 |
| 1, shared ports | 8 | 400 | 3.9968 / 4.0800 | 197.3728 / 509.1808 | 743.0368 |
| 1, shared ports | 16 | 200 | 9.9872 / 10.1536 | 222.8992 / 534.8032 | 768.7312 |
| 1, shared ports | 16 | 400 | 5.9936 / 6.0768 | 218.9056 / 530.8096 | 764.7376 |
| 2, disjoint ports | 8 | 200 | 4.1632 / 4.1632 | 197.5392 / 197.5392 | 431.3952 |
| 2, disjoint ports | 8 | 400 | 3.0816 / 3.0816 | 196.4576 / 196.4576 | 430.3136 |
| 2, disjoint ports | 16 | 200 | 6.1600 / 6.1600 | 219.0720 / 219.0720 | 453.0000 |
| 2, disjoint ports | 16 | 400 | 4.0800 / 4.0800 | 216.9920 / 216.9920 | 450.9200 |

All timing columns are microseconds. Flow completion time (FCT) starts at
shard eligibility; TTFT and last completion start at request admission.
The FCT column applies to every shard of the named request. Requests arrive
together and emit four decode tokens each. The second request on one decode
engine additionally waits for that engine; its larger TTFT is not attributed
to the network.

Doubling the link rate halves serialization while keeping submission and
propagation fixed. Doubling the payload doubles packet work while retaining
the startup packet term. All four rate and four size comparisons satisfy the
frozen additive residual bounds. All four receiver-sharing comparisons move
the earliest TTFT by the exact difference between the earliest shard joins,
between `(n - 1) * q` and `n * q`.

The separate memory check starts from the unchanged compute surrogate's
40.108032-microsecond resident-weight streaming floor. Selected prefill
services are 95.424 and 114.936 microseconds; decode services are 77.952 and
77.976 microseconds. Each exceeds that conditional floor. COMP-7 owns demand
from selected experts instead of all resident experts, so this check does not
qualify real GPU service.

The causal serving check bounds completion by prefill, submission, the
conservative network ceiling and eight decode intervals. Every request lies
inside its frozen range. In the eight-token 200-Gbit/s shared cell, the last
completion is 745.0336 microseconds, below the 773.1552-microsecond ceiling.
Two independent decode engines start at their own shard joins. One decode
engine also waits for its prior request to finish. The native input records,
completion events, step results and four token times match these schedules
exactly. Time per output token (TPOT) remains the selected decode service;
that configuration-forced identity is unscored.

## Evidence, ownership and chronology

The executed source is `be67cd7da3dc6b22ac7340cedc0a0356bc2a1331`.
Original expectation commit `2ad72f72d947e636ea384bfa40f365da27e1ec5a`
precedes implementation and all shared campaigns. Deadline amendment
`0e0e71aa586b2ccd2d786534d6b686d888e300b0` and identity amendment
`4c0cf5f4494d4534f4cdd481543827bbb3528032` precede their corresponding
implementation and the first campaign. The final pre-run expectation commit,
`9b520748dfe9a7ba9d17c72a98e71148fa0660be`, follows the first campaign's
failed service-vector guard and precedes its correction and this fresh retry.
That reader regression is post-specified relative to the failed observation;
it is not presented as an original prospective prediction.

The [first campaign remains VOID](VOID_RESULTS.md), with null behavioral
score and unchanged receipts. It exposed a checker that assigned another
serialized engine's waiting interval to the current request's service.
The corrected reader uses the already frozen reference's individual starts
and finishes. No production timing, workload, physical relation, tolerance
or resource cap changes, and the old execution is never rescored.

The complete evidence classes remain separate:

- Eight primary configurations cover sixteen requests; two persistent
  configurations cover eight requests across two batches each; eight
  source-paired compatibility processes cover sixteen requests.
- All 22 required stages finish. All 2,637,380 fatal guards hold, with no
  ownership, timestamp, source, conservation or receipt-integrity violation.
- All twenty exact vectors agree: four complete compatibility pairs, four
  disjoint-port packet vectors and twelve complete serving schedules.
- All twelve behavioral instances pass in three families, four each for
  rate, payload size and receiving-port sharing.
- Twenty component contracts, six alias-corruption controls and twenty-three
  service-vector regressions remain unscored. The software executable reports
  253 passed and three skipped cases; these are not native behavioral points.

The eight primary packet cells retain 128 flows and 512 native lifecycle
events. Their exact payload/header geometry accounts for 2,304 DATA packets
and 9,584,640 wire bytes. Packet counts are derived from that admitted geometry,
not presented as separately captured packet callbacks. Native lifecycle rows
retain accepted, queued, started and completed timestamps. The interface
supplies no resource-release timestamp, so no release event or queue-service
duration is invented.

Each repeated-batch process retains the same child, policy owner and endpoint
bindings. Its accepted sequences continue from 1 through 16 to 17 through 32.
The second batch starts at 1,664 microseconds, after the first batch's frozen
completion ceiling and on the same packet-calendar phase. Relative packet
and request times match exactly while identities remain distinct.

The slowest process finishes in 124.353145 seconds, below the 1,800-second
limit. Maximum sampled current resident memory is 1,141,304 KiB, below sixteen
gibibytes. These observations establish this campaign's host feasibility;
they are not controlled host speedup measurements.

The [portable publication](results.json) retains receipts for 1,429 raw files,
totaling 347,757,328 bytes, plus 29 root receipts. First and final inventories
match exactly, and publication checks rehash every retained file. The raw
summary SHA-256 is
`f5b9e667d207a895218ecb4aed1391debb964d30c36d069d067d6e04154d7dd0`.
The source-manifest digest and every process envelope remain in the publication.

## Project effect and limits

CORE-71's shared transfer-time claim is literal for in-process vLLM 0.27.1,
one scheduled sequence per engine, virtual workers and the pinned ideal
packet endpoint. One engine runtime owns the public clock; one packet runtime
owns all cache-transfer contention. The all-shard join releases actual decode
work, connecting packet completion to the reported TTFT and TPOT chain.

CORE-70 retains native tensor-buffer retention, shared compute/collective
composition and broader native modes. The connector here is tensor-free.
TRAF-64 retains target-topology and physical-fabric qualification; this
topology-free endpoint profile does not model a Clos or NVLink switch.
CORE-52 retains the full 448-worker host-scale campaign. CORE-51 and CORE-54
retain the large-model deployment and explainable frontier obligations.
No GPU kernels, model weights or physical network were measured. This result
qualifies the causal composition, without establishing calibrated DeepSeek-V3
or Kimi K3 deployment performance.
