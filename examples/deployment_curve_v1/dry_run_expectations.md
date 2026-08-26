# Granite two-parameter dry-run expectations

This freeze precedes the two-parameter granite dry run but follows the scaffold
implementation and an earlier 8-token-only dry run. It is therefore a
post-specified dry-run regression, not a public pre-registration. It cannot
score CORE-54 or change any disclosure anchor.

## Sweep and relations

Both configurations use one eight-GPU prefill node plus one eight-GPU decode
node, the granite roofline-bootstrap column and bootstrap pricing. They differ
only in prompt length, 8 or 16 tokens. Each runs 8 requests asking for 4 output
tokens at 8,000, 16,000 and 32,000 requests per second.

Within each prompt curve, aggregated output throughput is nondecreasing with
offered load. At each matched load, the 16-token prompt does not reduce
per-token request delay relative to the 8-token prompt. Doubling the prompt
doubles key-value handoff bytes from 393,216 to 786,432 exactly. Every point
conserves 8 admissions, 8 terminals and 32 output tokens.

## Physical bounds before the run

The named 400-million-active-parameter granite model carries about 800 million
bytes of active bfloat16 weights. Perfect tensor-parallel sharding gives 100
million bytes per rank. At the declared B100 bootstrap bandwidth of 8 trillion
bytes per second, a complete weight read takes at least 12,500,000 ps. The
corresponding memory-only ceiling is 80,000 decode tokens per second per
request. The observed client rate must remain between 10 and 100,000 tokens
per second per request, the previously accepted broad session bound.

At 400 Gbit/s over eight equal rank-pair links, 393,216 handoff bytes need at
least 983,040 ps of serialization and 786,432 bytes need at least 1,966,080 ps.
The selected 100,000,000 ps constant is above both wire floors and inside its
declared envelope.

## Scope

The run demonstrates a second varied parameter and multi-curve rendering. It
does not calibrate granite, compare equivalent models or hardware, fit a
constant, read a held-out anchor, score a DeepSeek point or close CORE-54.
