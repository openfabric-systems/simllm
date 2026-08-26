# COMP-75 preregistered source allowlist

Status: frozen before source inspection

Pinned source tree:
`/data3/yifeng/simllm-dev/wave-runs/ds67/sglang-source-bfeae4e79/`

Pinned source commit: `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`

## Allowed source ranges

Only the following inclusive, one-based line ranges may be read from the
pinned SGLang tree:

| Purpose | Relative implementation path | Lines |
|---|---|---:|
| FP8 dispatch and destination-rank incidence | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | 1-1041 |
| Two-batch stage interleaving | `python/sglang/srt/batch_overlap/two_batch_overlap.py` | 1-1157 |

The complete-file bounds are preregistered because narrower semantic ranges
cannot be selected without inspecting the source text. Inspection may quote or
derive evidence only from the smallest relevant subranges found inside these
frozen bounds.

## Denied sources and payloads

Everything not enumerated above is denied. This includes:

- framework evaluation, benchmark, test, and result tables;
- all CORE-60 void-run external-source values and evidence payloads;
- every forbidden anchor, flagship score, held-out row, and held-out payload;
- model weights and model downloads;
- decode pricing, SGL-38 residuals, traffic-module TRAF-65 material, htsim, and
  the NVLink model;
- all web pages, URLs, network fetches, and remote source material.

Existing in-repository component evidence may be used only after this
allowlist commit, subject to the COMP-75 spec and its exposure rules. No source
allowlist amendment is permitted after inspection begins.
