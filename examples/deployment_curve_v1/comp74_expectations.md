# COMP-74 expectations freeze

## Chronology and access boundary

This is an expectations-only freeze authored against `7a906bfa322685316a5973f4c40cfb9e555f3d3b`.
No COMP-74 runner, interval, result or publication existed when it was written.
No retained repetition value beyond the values already disclosed by the merged
studies was inspected. The committed field reader must precede every additional
record access.

The reader allowlists three repository-relative sources and returns only the
selectors written verbatim in `comp74_expectations.json`. Each access appends a
`simllm-deployment-curve-comp74-field-access-v1` JSONL row. It never loads a
whole record or returns an unselected value. No published-throughput field is
allowlisted, and no published-throughput value may select, estimate, adjust or
propagate a distribution.

## Frozen repeat statistic

Each of the four priced DeepSeek keys is its own estimation population. The
immutable `published_point_ps` and its `independent_repeat_ps` form that key's
retained observation set. The scored center remains `published_point_ps` and is
never replaced by a mean, median or fitted value.

For point service `p` and observations `x_i`, the exact relative half-width is

`h = max_i(abs(x_i - p)) / p`.

The key-local observed-repeat service envelope is `p * [1 - h, 1 + h]`. It is
nonzero exactly when at least one observation varies. This envelope contains
all retained observations, uses no distributional family, and is not a
confidence interval. Two observations support an observed-repeat envelope, not
a broad stability claim.

There is no pooling across roles, prompt lengths, MTP modes or implementation
suffixes. The exact mapping is:

| Retained implementation suffix | Flagship anchor | Role and shape |
|---|---|---|
| `ep32-prefill-r16-l1024-t16384` | `sglang_prefill_1k` | EP32 prefill, 1K prompt |
| `ep32-prefill-r8-l2048-t16384` | `sglang_prefill_2k` | EP32 prefill, 2K prompt |
| `ep32-prefill-r4-l4096-t16384` | `sglang_prefill_4k` | EP32 prefill, 4K prompt |
| `ep72-decode-b32-c2000` | `sglang_decode_standard` | EP72 standard decode, batch 32, remote KV 2000 |

## Frozen interval propagation

The existing `deterministic-additive-interval-v1` engine remains authoritative.
For an inherited prediction interval `[L, P, U]`, enabled distribution
propagation performs the Minkowski addition

`[L - P*h, U + P*h]`.

The physical record and constant-envelope contributions remain in `[L, U]`;
the repeat-derived contribution is new and separately labeled. The standard
decode interval is then passed through the existing capacity-to-curve engine,
so the flagship load curves inherit the wider capacity band.

With distribution propagation disabled, the implementation must return every
inherited interval object without recomputation and reproduce every current
scored point prediction exactly. Propagation never rescores. The run-3 prefill
PASS rows and the run-4 simulated-MTP REFUTED row retain their frozen verdicts.
If a wider interval touches either edge of a closed 5 percent bar, the result
reports that contact as context only.

The simulated-MTP row remains single-seed and receives no standard-decode
spread. Borrowing that spread would pool MTP modes and violate this freeze.

## Evidence and closure

The successor lookup record remains `candidate`, its evidence-class ledger is
preserved exactly, and neither predecessor nor successor is mutable. All prior
CORE-54 scored publication artifacts named in the preservation class must keep
their frozen SHA-256 identities.

COMP-74 closes only if all four priced keys retain at least two independent
observations, varying rows receive nonzero intervals, OFF reproduces every
current point exactly, and the new study artifact restates rather than changes
the verdicts. `COMP-79` is reserved for single-seed DeepSeek keys including
simulated MTP. `COMP-80` is reserved for the Granite arm's absent repetitions.
