# CORE-63 decode expert-residency result

Status: **PROTOCOL_VOID_CALIBRATION_ONLY_UNDERCORRECTION**. This is an honest calibration-only numerical
finding, not a clean-protocol closure.

## Residency arithmetic, corrected step and movement

Uniform routing gives one EP72 rank an expected
`256 tokens/node x top 8 x 4 resident slots / 288 slots = 256/9` routed
expert-token assignments. The TP1 batch-32 capture represents `32 x 8 = 256`
assignments, so the frozen routed-expert scale is exactly `1/9`. The signed
direction was frozen first: the step must decrease and throughput must increase.

The 46 retained noncollective rows total
1744159511.000000 ps of retained
repeatable work plus 131520000.000000 ps of
routed `fused_moe_kernel` work, with the independently retained fixed term of
489.000000 ps kept once. Therefore:

```text
T63 = 489 + 61/4 x (1,744,159,511 + 131,520,000 / 9)
    = 26821286365.083333 ps
```

The published round-half-up step is
**26,821,286,365 ps**. The
standard-decode prediction moves from the published 8,949.76 display to
**9544.657796 tokens/s/node**, a
signed increase of **594.898111**
tokens/s/node (6.647085 percent). Against the
published calibration anchor of 22,282 tokens/s/node, the signed residual is
**-57.164268 percent**, a
2.669860-percentage-point movement.
The calibration classification is **UNDERCORRECTION**.

## Component and mechanism ruling

Only the one row containing the preregistered marker `fused_moe_kernel` is
scaled. Attention, MLA, router/top-k work, the shared expert, dense early MLP,
normalization, elementwise work and every other noncollective row stay at
scale one. The kernel summary reconstructs the retained 1,875,680,000 ps step
exactly, and the record components independently reconstruct it from compute,
memory and the 489 ps fixed term. There are zero fitted or free constants.

No communication term enters the current decode price. Decode-side overlap is
therefore not binding here and remains a follow-on only after a decode
communication service term exists.

## Protocol, access and preservation

The final successful tranche contains exactly two logged field-addressed
accesses. Across schema discovery and the successful tranche, the append-only
reader ledger contains 11 entries:
6 PASS and 5
REJECTED. Its held-out ledger is empty.

CORE-63 is nevertheless **protocol void**. Before the committed reader/freeze,
an ambient direct range inspection exposed a forbidden held-out MTP numeric
value and a retained historical record was inspected without a contemporaneous
access row. A later broad registry inspection re-exposed that held-out value.
None of those values entered the residency arithmetic and no MTP comparison or
score was performed, but the literal no-read and every-access-logged rules
cannot be restored after exposure. The CSV selector also required streaming
all 13,985 cataloged bytes. It never materialized the whole record or decoded
unselected payload fields, but that still fails the literal no-whole-file-read
clause. The incident ledger in the JSON publication records these failures
without reproducing the held-out number.

All 93 preservation-lock artifacts
remain byte-identical. No prior scored artifact changed, no model weights were
downloaded, no web page was fetched and no scored run was performed.

## Registry verdict

CORE-63 stays open for a genuinely exposure-free repetition. CORE-64 is opened
only for the exact remaining standard-decode undercorrection residual reported
above; its conclusion remains conditional on a clean CORE-63 repetition.
