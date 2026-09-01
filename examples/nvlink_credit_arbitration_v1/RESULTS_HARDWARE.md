# TRAF-73 NV4 credit and arbitration identification result

## Identification verdicts

- H1, credit window and return: **VOID**. No directed pair had one break at the same payload on every timed pass. This has the frozen no-break direction, but the void run identifies neither a window nor a return delay.
- H2, pool scope: **VOID**. No sender count had the repeated knees required by the frozen constant-aggregate shared-pool or growing-aggregate per-link-pool selector.
- H3, arbitration: **VOID**. The unscored shape gave each small sender 58.785 to 59.545 GB/s, which has the frozen release-aware fair direction, but the receiver-ceiling guard voided classification.

## Aggregate outstanding bytes discriminator

| Senders | Per-sender knees, B | Aggregate outstanding, B |
|---:|---|---:|
| 1 | none | none |
| 2 | none | none |
| 3 | none | none |

## H3 unscored achieved-rate shape

| Greedy source | Achieved raw GB/s by source order | Aggregate raw GB/s | Frozen shape before fatal guards |
|---:|---|---:|---|
| 0 | 93.646419968, 58.922897408, 58.784913408 | 211.354230784 | release_aware_round_robin |
| 1 | 94.103437312, 59.460319232, 59.545339904 | 213.109096448 | release_aware_round_robin |
| 2 | 93.463478272, 59.302037504, 59.143274496 | 211.908790272 | release_aware_round_robin |

## What ran

Jobs 202778, 202796 and 202813 ran the frozen H1, H2 and H3
families serially on one qualified four-A100 NV4 node through the
corrected TRAF-70 producer lineage.

## What came out

The hardware result is **VOID**. The largest H3 aggregate was 213.109096448 GB/s against the frozen 207.101921876 GB/s fatal ceiling. Independently, the largest H1 or H2 completion was 7.587253223 times the frozen loose service ceiling.

## What it changes for the project

TRAF-73 stays open. Its credit-window, pool-scope and arbitration
identifications do not become literal, and no milestone moves.
TRAF-85 is not registered because void evidence cannot promote a
model value; its exact promotion-cell set is empty.

## What it does not change

This result does not edit the aligned module, candidate profile or any
README. It promotes no declared window, return, pool scope or arbitration
policy. Degrees 4, 8 and 16 remain simulated mesh extrapolations.

## Fatal guards

Fatal-guard verdict: **VOID**. A failed fatal guard voids the result, so these findings are not presented as a pass fraction.

- FG15: an ordered-pair raw rate exceeds 100 GB/s or aggregate raw rate exceeds 207.101921876 GB/s beyond one chunk of quantization.

Physical sanity: the fastest completion was
258.363969110 times the packetized 100 GB/s wire floor, while the slowest relative case reached 7.587253223 times the frozen loose ceiling.
