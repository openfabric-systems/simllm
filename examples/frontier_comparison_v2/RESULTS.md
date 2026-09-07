# Attention pair successor result

## What ran

The COMP-81 successor is non-void and the external comparison remains
**MIXED**. The expectations-only commit is
`37676c4f443026fc20944126f643c77eb8705b90`, with frozen expectation digest
`ead11b885885a01905e2af32ddc9e9d3c8aa90c29188148424e9f9faee21e688`.

Config-only extraction read the pinned Qwen3-32B-FP8 config at revision
`aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df` through the existing
`extract_model_inventory` path with `attention_shape_version=2`. The run used
`HF_HUB_OFFLINE=1` and loaded no weights. Framework source identities and
bindings were replayed from the retained suite; this is not a fresh native
vLLM or SGLang configuration inspection. Two repetitions per framework
produced identical canonical inventory bytes and identical step-record
streams, retained in [extraction.json](extraction.json).

| Framework | Successor inventory SHA-256 |
|---|---|
| vLLM 0.27.1 | `b3ec33edcab6a81868c00efdc2147df4d41b633bd932cb6c4831ea282a1d79fe` |
| SGLang 0.5.19.dev345+gbfeae4e79 | `d25471742a987e31d872dc6fbf8c60ee8746ba17eb25e95a245689064ed3a8d7` |

Each inventory retains the same 15 cases, five families and 257 logical
visits per case. Only the attention invocation schema and its projection
vectors differ from its predecessor. Family FLOPs, HBM bytes, case identities,
model identity, graph identities and source provenance are unchanged.
[predecessors.json](predecessors.json) pins both old inventories, the suite,
and every tracked v1 study file. All those bytes remain unchanged.

The estimator swept 180 prompt/context/TP/efficiency cells: prompt lengths
128, 512, 2048, 3500 and 4096; decode contexts 128, 2048, 4250 and 8192;
TP 2, 4 and 8; efficiencies 0.6, 0.8 and 1.0. Each cell prices batch-64
decode and batch-1 prefill through the live deployment estimator. The
unchanged frontier family scanned 5,070 candidates per efficiency arm,
15,210 total, in 24.202 seconds on the retained publication run. Wall time is
diagnostic and excluded from the deterministic result record.

## What came out

The pair coefficient is **2,097,152 FLOPs in both phases in all 30
framework/case rows**. Every pair axis matches `step_shape`; every family
FLOP and byte sum matches the fused kernel exactly. Ten fatal guards pass.
The exact oracles are structural evidence, separately reported from the
behavioral comparisons. A fatal failure yields VOID and null scores.

The existing coefficient was already correct in both old inventories.
The v1 comparison divided the four-sequence prefill row by the aggregate
128-token square, losing its per-sequence structure. That divisor is 16,384;
the modeled pair count is 2,048. The new invocation schema
`simllm-attn-score-invocation-shape-v2` carries
`(new_tokens, kv_tokens, attention_pairs)` with units `(tokens, tokens, pairs)`.
It preserves `step_shape`'s analytical half-square convention:
`sum(n * max(context - n, 0) + n * n // 2)` across sequences. Rounding happens
per sequence. This is a causal approximation, not the inclusive discrete
triangle; an uncached one-token sequence contributes zero pairs and cannot
identify a per-pair coefficient on its own.

The coefficient is `4 * layers * query_heads * head_size`, covering QK and
PV across all represented layers and query heads. GQA shares KV storage
without removing query-head products. At one sequence of U uncached tokens,
the successor prices `2,097,152 * (U * U // 2)`. Relative to v1's
`262,144 * U * U`, the attention term rises 4x while its coefficient rises 8x.

| Work quantity | Old whole-model FLOPs | Successor whole-model FLOPs |
|---|---:|---:|
| Decode per batch item, context 4250 | 72,875,612,160 | 72,877,867,008 |
| Prefill per request, U=3500 | 221,652,172,144,640 | 231,285,964,144,640 |

Successor TP4 decode owns 18,219,466,752 FLOPs per item or
1,166,045,872,128 per batch-64 rank. Successor TP4 prefill owns
57,821,491,036,160 FLOPs per request. Every integer matches the freeze.
Decode's additional 2,254,848 FLOPs per item come from using 4,249 pairs
instead of scaling the old rounded per-context coefficient.

| X2 row | Old value | Successor value | Old / successor verdict |
|---|---:|---:|---|
| X2a decode versus 9.179 ms | 5.379515733 ms | 5.379515733 ms | PASS / PASS |
| X2b prefill versus 196.423 ms | 28.000527052 ms | 29.217529578 ms | PASS / PASS |
| X2c decode e-star | 0.5860677343 | 0.5860677343 | PASS / PASS |
| X2c prefill e-star | 0.1425521810 | 0.1487480060 | FAIL / FAIL |

**The prefill correction does not change the X2c-prefill verdict.** Its total
work and compute-bound service rise about 4.346 percent at U=3500. Attention
is 5.554 percent of successor prefill FLOPs there. At U=2048, attention costs
4,398,046,511,104 out of 132,217,829,064,704 FLOPs, **3.326 percent**.
Neither implied efficiency is installed as a model parameter.

The [plain comparison figure](figures/attention-pair-successor.png), also
available as [PDF](figures/attention-pair-successor.pdf), places old and
successor X3 frontiers together on logarithmic axes at efficiencies 1.0,
0.8 and 0.6: dashed lines are old, solid lines are successor, and black
crosses are external rows. The right panel shows X2c implied efficiencies
as open circles (old) and crosses (successor). Decode overlaps at 0.5861;
prefill rises from 0.1426 to 0.1487, both below the dotted frozen floor of
0.4. The pair coefficient is 2,097,152 floating-point operations (FLOPs).
Whole-model work rises 4.346 percent at 3,500 uncached prefill tokens and
0.003 percent at decode context 4,250. All three successor frontiers remain
monotone. Matched throughput never rises.
X3a remains 4/4, X3b 10/10 and X3c 3/10 against its minimum of eight.
The modest attention correction does not close the calibration gap.

## Physical sanity and evidence scope

The pre-written bounds hold. TP4 batch-64 decode moves 25,821,675,520 bytes:
its HBM floor is 5.379516 ms, versus a compute floor of 0.589210 ms. Its
e=0.4 envelope ceiling is 13.448789 ms. Prefill moves 8,225,259,520 bytes:
its compute floor is 29.217530 ms, versus a memory floor of 1.713596 ms;
its e=0.4 ceiling is 73.043824 ms. Estimator picoseconds truncate fractions,
so reported service can be less than the exact rational floor by under one
picosecond. Static TP4 matrix storage is 7,995,883,520 bytes, below the
frozen 10 GB bound and declared 141 GB capacity.

Decode grows affinely with context; prefill grows quadratically with prompt
length. At fixed work, service is nonincreasing in TP and efficiency, and
all 180 cells remain within their first-principles floor/ceiling envelope.
These are ESTIMATE services with DECLARED H200 envelopes. The external
columns are MEASURED-EXTERNAL operation-database estimates, not measurements
of complete serving requests. Zero subprocesses entered the estimator sweep.
No GPU, network, native framework runtime or external executable ran.

## What it changes

COMP-81 is complete: both framework inventories agree after removing source
provenance, the same pair has the same phase-independent coefficient, family
conservation is exact, and the successor comparison names both new hashes.
The owning open entry is removed. No new stable task ID is registered.
Native extraction also exposes `--attention-shape-version 2`; its default v1
mode preserves historical identities. The axis is an explicit inventory
projection of the unchanged `step_shape` authority. `transformer.py` remains
byte-identical because older studies freeze that source file by digest. Successor pricing requires the v2 pair
property and rejects an old two-axis projection instead of guessing its
missing denominator.

## What it does not change

The old inventories and entire v1 study remain immutable, including their
historical prose. This successor corrects that prose's attribution of the 8x
mismatch: the error was in the comparison's divisor, not inventory FLOPs.
The external TTFT operating-point versus isolated-service confound remains
DEPLOY-12's scope. Physical launch identity, omitted families, tensor-parallel
collective service and silicon calibration remain outside COMP-81. No
residual COMP-81 work remains; no other task entry changes.

## Reproduction

Use the worktree's environment and Python. Bulk paths are supplied by the
caller and never serialized into tracked records:

```bash
. ./.env.local.sh
HF_HUB_OFFLINE=1 .venv/bin/python examples/frontier_comparison_v2/extract_inventories.py \
  --checkpoint-root "$SIMLLM_QWEN3_32B_CHECKPOINT_ROOT" \
  --output-root "$SIMLLM_DATA_ROOT/frontier_comparison_v2/reproduction-extraction"
.venv/bin/python examples/frontier_comparison_v2/run_study.py \
  --output-root "$SIMLLM_DATA_ROOT/frontier_comparison_v2/reproduction-study"
```

The study consumes the two committed content-addressed inventories and
[results.json](results.json) records the exact live-estimator projection.
Extraction emits the same objects plus its repetition receipt. Every new
text artifact is written as LF bytes; digest checks accept raw or
LF-normalized input for Windows checkouts. Tests exercise both schemas,
all committed inventories, GQA, unequal sequences, odd lengths, zero pairs,
mutated axes/FLOPs, extraction nonconservation, VOID behavior, and direct
estimator reproduction of every sweep row and X2 stamp.
