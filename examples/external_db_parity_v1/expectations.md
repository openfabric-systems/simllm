# External database parity expectations

These expectations freeze the import of the external planning tool's
measured operation database and the parity bar the maintainer set: at the
matched seam the numbers are identical, so every remaining difference
between the two stacks is a mechanism, not a timing base. They are
committed before any importer implementation exists. The design basis is
the executed source audit of the installed tool (its findings are restated
here as frozen facts; the audit's oracle values were produced by the
external sdk, not by any simllm code).

## Frozen external identity (fatal)

- Packages: aiconfigurator 0.11.0 with aiconfigurator-core 0.11.0, read
  from a local installation whose root is supplied by
  SIMLLM_EXTERNAL_AIC_VENV; the run records both package versions and
  refuses any other.
- Slice: system h200_sxm, backend trtllm, version 1.3.0rc10,
  DatabaseMode SILICON, shared_layer False, the Python estimator surface,
  strict_provenance False with every manifest's legacy provenance flag
  recorded.
- Data-slice hash (the audit's executed recipe over the 27 slice files,
  sorted path-and-content manifest):
  `85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284`.
- Pricing-closure hash (slice plus systems/h200_sxm.yaml):
  `d559d6694f30ad269ecbf697e0193c7d95e4aba1cfb929836d381a46b675876f`.
- systems/h200_sxm.yaml SHA-256
  `142584d6bddd98207fd04e844029b0ba5d6fcd4c6f41016c5e77f0cbe4053614`;
  bundled Qwen--Qwen3-32B-FP8_config.json SHA-256
  `e546dacd2c772660270233f5579e9ab923cc2a7ec5ed3c58c27c2bc62cbf5169`.

## Fatal guards

- FG-1 the local installation's slice reproduces the data-slice hash
  before conversion, and the tracked converted artifact's manifest
  records it.
- FG-2 licensing: the converted artifact directory carries the Apache 2.0
  license, the preserved SPDX and copyright notices, and a modified-file
  statement; the repository NOTICE is updated; a test checks all three.
- FG-3 no donor rows: the converted artifact contains rows from exactly
  version 1.3.0rc10 (shared_layer stays off end to end); a version scan
  over every converted row is the guard.
- FG-4 evidence class: every value served from the import carries the
  MEASURED-EXTERNAL class with tool, package versions, system, backend,
  database version and slice hash in its source identity; nothing from
  the import can be served under MEASURED.
- FG-5 load-time mutation parity: the audited SOL clamps (GEMM and
  generation attention raised to their analytical floor at load) are
  applied by the importer; one frozen cell whose raw value sits below its
  SOL floor must serve the clamped value on both sides.
- FG-6 determinism: every scored comparison executes twice in fresh
  processes with bit-equal results.
- FG-7 composite honesty: the family mapping ships as a declared table
  (exact, composite, gap per the audit); a composite's fused external
  record and its constituent families are never both priced in one
  composition; gaps fail closed naming the mapping entry.
- FG-8 chronology.

## Family I1: importer identity (exact, scored)

The converted artifact round-trips the audit's representative raw records
byte-exactly as IEEE-754 doubles, including at least: GEMM
{bfloat16, m=4353, n=65536, k=51200} at 41.66388193766276 ms and
{fp8, m=1024, n=32, k=32} at 0.012622933586438498 ms; generation GQA
{kv=fp8, kv_heads=8, head_dim=64, heads=96, batch=64, total_seq=2} at
0.007925333455204964 ms; MoE {bfloat16, tokens=256, hidden=4096,
inter=1536, topk=8, experts=128, tp=4, ep=4, balanced} at
0.10713600118954976 ms; custom all-reduce {tp=8, bytes=536870912} at
4.007393798828125 ms; GDN fused decode {batch=512, d_model=2048} at
0.33931519985198977 ms; compute scale {m=32768, k=51200, fp8} at
2.4173152923583987 ms. Row counts per table equal the audit inventory
(gemm 101,010; context attention 50,574; generation attention 24,438;
moe 74,358; the full 284,717 total).

## Family I2: resolver parity (exact, scored)

For a frozen list of at least twenty query points spanning every audited
resolution rule (exact nested hit returned verbatim; GEMM linear-in-m
inside a site curve; GEMM cross-site blend in log2 space with the 2.0
distance cap; context sqrt-latency sequence interpolation; the context
prefix formula; generation five-point smoothing over 0.9 s to 1.1 s;
one-dimensional curves for MoE tokens, all-reduce bytes and GDN batch;
boundary utilization hold beyond a range; quantize clamp behaviors), the
simllm resolver's value is bit-equal to the external sdk's query executed
live from the pinned installation. The query list is frozen in the study
configuration at implementation time before the first comparison run,
and every point names the rule it exercises.

## Family P1: pass-composition parity (exact, scored)

Reproducing the external Python composition (context and generation
phase rules, effective-token accounting, stride sampling, repeat counts)
over the imported database, the four audited pass oracles land bit-equal
on the frozen IEEE-754 values:

| Oracle | Arguments | Frozen value |
|---|---|---|
| prefill | batch 1, isl 3500, osl 1, static_ctx | 99.2826112094288 ms, hex 0x1.8d2164d537eb3p+6 |
| decode | batch 64, isl 4000, osl 2, static_gen, stride 32 | 11.10212868689609 ms, hex 0x1.6344a3614677ep+3 |
| prefill | batch 2, isl 1750, osl 1, static_ctx | 94.83179372633029 ms, hex 0x1.7b53c1bc0e6d2p+6 |
| decode | batch 32, isl 8000, osl 2, static_gen, stride 32 | 11.207792286589099 ms, hex 0x1.66a63c02685c1p+3 |

A one-ULP or larger miss is published as a finding with the diverging
term isolated (the bar is bit-equality on the pinned Python surface; the
audited compiled surface differs by one ULP and is out of scope).

## Family W: wall time (scored, generous)

Conversion of the slice plus the complete I1, I2 and P1 evaluation
completes in at most 120 s single process, machine disclosed.

## Closure

A full pass establishes the matched-seam identity the maintainer
directed: simllm pricing from the imported database is numerically the
external tool's pricing. It does not reproduce their end-to-end pareto
rows (their serving composition above this seam), does not compare
mechanisms, and does not import any other system or version; those are
the follow-on freeze. Scored families are I1, I2, P1 and W, in their
classes, never summed with fatal rows.
