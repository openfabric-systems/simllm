# TRAF-66 source allowlist extension

Status: frozen before TRAF-66 source inspection

This record extends, but does not modify, the byte-locked
`notes/2026-08-26-comp75-source-allowlist.md` protocol. The ranges below were
selected only from citations already committed in CORE-60 and COMP-75 records.
No external SGLang source was read to choose them.

Pinned source tree: `<SGLANG_SOURCE_ROOT>/`, where the configured external root
must have the leaf name `sglang-source-bfeae4e79`.

Pinned source commit: `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`

## Allowed TRAF-66 source ranges

Only the following inclusive, one-based line ranges may be read from the
pinned SGLang tree for TRAF-66:

| Purpose | Relative implementation path | Lines |
|---|---|---:|
| Two-child stage executor and offset interleave | `python/sglang/srt/batch_overlap/operations.py` | 38-71 |
| Per-layer dispatch and combine yield boundaries | `python/sglang/srt/batch_overlap/operations_strategy.py` | 89-132 |
| Child split, operation invocation and output merge | `python/sglang/srt/batch_overlap/two_batch_overlap.py` | 880-941, 1085-1134 |

The `two_batch_overlap.py` ranges narrow TRAF-66 inspection within the complete
file bound that COMP-75 preregistered. The other two rows are additive files.
No other range is implied by an import, symbol reference or comment in these
ranges. No amendment is permitted after TRAF-66 inspection begins.

## Denied sources and payloads

Everything not enumerated above is denied for TRAF-66 source inspection. This
includes:

- framework evaluation, benchmark, test and result tables;
- every anchor payload, held-out row, held-out payload and scored-run artifact;
- CORE-60 and COMP-75 external-source values outside their committed records;
- model weights and model downloads;
- decode pricing, SGL-38 residuals, TRAF-65 material, htsim and the NVLink
  model;
- all web pages, URLs, network fetches and remote source material.

Existing in-repository component compute and packet records remain permitted
inputs. Their prior bytes are preservation-locked and they may not be rewritten
or replaced.
