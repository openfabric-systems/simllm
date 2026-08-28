# CORE-63 clean expert-residency repetition

Status: **PASS_CLEAN_REPRODUCTION_UNDERCORRECTION**. The clean repetition independently reproduces
an honest **UNDERCORRECTION** calibration-only result.

## Empty forbidden-access ledger

The forbidden-access ledger is exactly `[]`. No held-out MTP numeric value was
read, copied, compared, or scored. The fifth scored run retains sole ownership
of scoring.

## Reproduced residency step and signed movement

The frozen assignment arithmetic is `256 * 8 * 4 / 288 = 256/9`, relative to
the 256-assignment capture, so routed expert work alone is scaled by `1/9`.
The fixed service is kept once and every other noncollective family stays at
scale one.

```text
T63 = F + 61/4 * (retained_4 + routed_4/9)
    = 26821286365.083333 ps
```

The round-half-up corrected step is **26,821,286,365
ps**. The standard-decode prediction moves by
**+594.898111 tokens/s/node** to
**9544.657796 tokens/s/node**. The
signed residual moves by **+2.669860
percentage points** to **-57.164268 percent**.
The result remains an undercorrection, so the entry's literal acceptance bar
governs how far CORE-63 may move.

## Component classification

The clean reader selected 46 standard-decode
noncollective rows. Exactly 1 row matched
the frozen case-insensitive marker `fused_moe_kernel` and was classified as
routed expert work. Every attention, MLA, router/top-k, shared-expert, dense
MLP, normalization, elementwise, and other nonmatching row stayed retained at
scale one. The committed JSON companion carries the complete classification
ledger and the independently recomputed exact fractions.

## Access and preservation

All 6 allowlisted field accesses have contemporaneous
`BEGIN` and `END` events. Every completed byte count is strictly below the
source size, and the final CSV selector left the terminal record byte unread.
The earlier forward and header-plus-reverse preflights were both safely
rejected at 13,984 of 13,985 bytes before full coverage. A third preflight
passed the terminal-byte CSV selector and then rejected an over-specific
registry spelling before EOF. Across all tranches there were
16 logged accesses and
32 contemporaneous events. Whole-file
semantic streams: **0**.

All 93 inherited preservation artifacts are
byte-identical. Hash verification decoded no artifact values. The void
result's post-derivation numbers were not used as inputs, and no parameter was
amended or refit.

## Registry disposition

The exact pre-run CORE-63 and CORE-64 entries are retained in the JSON result
for literal acceptance review. This clean reproduction stands as candidate
evidence. CORE-64's attention-and-MLA-family gap may be registered
unconditionally because the clean repetition preserved all nonmatching
families and still produced the standard-decode undercorrection.
