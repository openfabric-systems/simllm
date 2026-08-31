# Frontier comparison result

## Verdict

**MIXED, non-void.** The corrected scoring record passes
X1, X3a, X3b, W and fails X2, X3c. X2c-prefill is outside the frozen
[0.40, 1.00] band at e-star
**0.142552**, so X2 fails. X3c passes **3 of
10** rows against the frozen minimum of
8, so X3c fails; the misses are rows
4, 5, 6, 7, 8, 9, 10. The bands are unchanged and every miss remains in the scoring
record.

The result is non-void because all five fatal guards held. Evidence classes
remain separate and are not added into one score.

| Family | Tally | Acceptance | Verdict |
|---|---:|---:|---|
| X1 | 3 / 3 | all | PASS |
| X2 | 3 / 4 | all | FAIL |
| X3a | 4 / 4 | all | PASS |
| X3b | 10 / 10 | all | PASS |
| X3c | 3 / 10 | >= 8 | FAIL |
| W | 1 / 1 | <= 120 s | PASS |

## Qwen3-32B FP8 extraction column

The config-only extraction binds `Qwen/Qwen3-32B-FP8` to exact Hugging Face
revision `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`. The vLLM 0.27.1 inventory is
`c8832ba8ba21e49517b6b74e89554c2abdb0d9e76530f647a7849f3f8448ec56` and the companion SGLang
inventory is `51740b52625002a964e75fddb679e9f8394a08a7d7c62556d2535c3bc60515e3`. Both contain
15 cases, 5 logical families
and 257 visits per case. The tracked tests
verify that each committed inventory is canonical at its content-addressed
filename and that their framework-neutral content agrees exactly after source
provenance is removed. No tracked second extraction run exists, so this report
does not claim repeat-run evidence.

FG-2 confirms the fatal architecture literals exactly: 64 layers, hidden size
5120, intermediate size 25600, 64 attention heads, 8 key-value heads, head
dimension 128 and vocabulary size 151936. The FP8 checkpoint has
32,762,123,264 logical parameters, so TP4 owns
8,190,530,816 weight bytes per rank at one byte per parameter.
That is 5.8 percent of the declared 141 GB H200 capacity.

## Work derivation and matched-point pricing

The pricing record derives its work from the inventory's per-layer
projections. Decode carries 72,875,612,160
whole-model FLOPs per batch item and a TP4 rank owns
18,218,903,040; a 3,500-token
uncached prefill carries 221,652,172,144,640
whole-model FLOPs and a TP4 rank owns
55,413,043,036,160. Logical weight,
key-value bytes and FLOPs now all divide by tensor-parallel width exactly once.
This is the physical ownership correction that replaced the prior whole-model
FLOP charge on every GPU.

At efficiency 1.0, the exact external-best topology prices decode batch 64 to
**5.379516 ms**, below the external
9.179 ms, and prices the uncached prefill request to
**28.000527 ms**, below the external
196.423 ms. The implied efficiencies are **0.586068 for decode** and
**0.142552 for prefill**. Decode is inside the frozen band;
prefill is outside. Both remain report-only and neither is
installed as a model parameter.

| X2 row | Corrected predicate value | Verdict |
|---|---:|---|
| X2a | 5.379516 ms <= 9.179000 ms | PASS |
| X2b | 28.000527 ms <= 196.423000 ms | PASS |
| X2c-decode | e-star 0.586068 | PASS |
| X2c-prefill | e-star 0.142552 | FAIL |

The frozen inventories also expose an unresolved convention mismatch:
decode `attn_score` projects 2,097,152 FLOPs
per token pair while prefill projects
262,144, exactly
8x lower. COMP-81 owns the successor
reconciliation; neither frozen inventory changed in this repair.

## Frontier overlay

[PDF](figures/frontier-comparison.pdf) and
[PNG](figures/frontier-comparison.png) show the three SimLLM ESTIMATE
frontiers and all 10 MEASURED-EXTERNAL rows on logarithmic per-user speed and
per-GPU throughput axes. The upper-right corner is better. External row labels
match the table below.

X3b compares the efficiency-1.0 service-feasible frontier at or above each
external per-user speed. X3c separately compares each external row's
throughput with the 0.6 and 1.0 estimates for that row's exact prefill/decode
topology and batch, including comparison points that the 10 ms frontier filter
excludes. The X3b frontier value and X3c matched-topology value have separate
columns so one cannot be read as the other.

| Row | External user tok/s | X3b frontier e=1.0 tok/s/GPU | X3b | X3c matched e=0.6 | External tok/s/GPU | X3c matched e=1.0 | X3c |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 56.032 | 1919.512 | PASS | 527.866 | 773.212 | 879.776 | PASS |
| 2 | 59.324 | 1919.512 | PASS | 514.404 | 765.613 | 857.339 | PASS |
| 3 | 84.006 | 1919.512 | PASS | 606.593 | 644.344 | 1010.989 | PASS |
| 4 | 108.944 | 1919.512 | PASS | 669.205 | 602.586 | 1115.342 | FAIL, below e=0.6 |
| 5 | 111.906 | 1913.229 | PASS | 640.856 | 541.607 | 1068.093 | FAIL, below e=0.6 |
| 6 | 119.638 | 1913.229 | PASS | 892.274 | 441.140 | 1487.123 | FAIL, below e=0.6 |
| 7 | 127.064 | 1913.229 | PASS | 767.837 | 386.606 | 1279.728 | FAIL, below e=0.6 |
| 8 | 146.002 | 1913.229 | PASS | 616.752 | 343.650 | 1027.920 | FAIL, below e=0.6 |
| 9 | 168.131 | 1913.229 | PASS | 707.628 | 258.328 | 1179.381 | FAIL, below e=0.6 |
| 10 | 202.168 | 1913.229 | PASS | 462.741 | 157.234 | 771.235 | FAIL, below e=0.6 |

X3b uses the frozen step-frontier rule rather than interpolation.
frontier point F1 answers external rows 1, 2, 3, 4 through the left endpoint clamp rule; frontier point F2 answers external rows 5, 6, 7, 8, 9, 10 through the first frontier point at or above external x rule. The complete answer identity is:

| External row | X3b selection | Answering frontier point (user tok/s, tok/s/GPU) | Candidate |
|---:|---|---|---|
| 1 | left endpoint clamp | F1 (109.972, 1919.512) | `p-tp2-w5-d-tp4-w3-b128` |
| 2 | left endpoint clamp | F1 (109.972, 1919.512) | `p-tp2-w5-d-tp4-w3-b128` |
| 3 | left endpoint clamp | F1 (109.972, 1919.512) | `p-tp2-w5-d-tp4-w3-b128` |
| 4 | left endpoint clamp | F1 (109.972, 1919.512) | `p-tp2-w5-d-tp4-w3-b128` |
| 5 | first frontier point at or above external x | F2 (244.954, 1913.229) | `p-tp2-w6-d-tp8-w2-b112` |
| 6 | first frontier point at or above external x | F2 (244.954, 1913.229) | `p-tp2-w6-d-tp8-w2-b112` |
| 7 | first frontier point at or above external x | F2 (244.954, 1913.229) | `p-tp2-w6-d-tp8-w2-b112` |
| 8 | first frontier point at or above external x | F2 (244.954, 1913.229) | `p-tp2-w6-d-tp8-w2-b112` |
| 9 | first frontier point at or above external x | F2 (244.954, 1913.229) | `p-tp2-w6-d-tp8-w2-b112` |
| 10 | first frontier point at or above external x | F2 (244.954, 1913.229) | `p-tp2-w6-d-tp8-w2-b112` |

The X4 scope comes from the
[frontier ladder study](../frontier_ladder_v1/RESULTS.md). Its ideal-network
class tracks the packet rung within about 1.6 percent on contention-free
point-to-point legs and is about 8x optimistic at the frozen eight-into-one
fan-in cell. This workload uses intra-node tensor parallel and one
prefill-to-decode transfer, so its declared regime is contention-free. No
packet run executes here, and the 1.6 percent statement is a regime-scoped
mechanism result rather than an absolute-accuracy claim.

Every candidate declares zero logical collective bytes per GPU per batch item;
the X4 1 to 2 percent bound applies only to ideal-versus-packet pricing of the
represented contention-free legs, not to the omitted tensor-parallel
collective service.

## Honesty and version drift

The external rows interpolate a measured per-operation database for real H200
silicon. Our rows use a declared roofline and declared envelopes until the
calibration campaigns close. **On absolute kernel throughput, their side is
better calibrated today.** The defensible precision claims are exactly the X4
network-mechanism envelope, the evidence-class label on every number and the
exact accounting gates. Nothing broader is claimed.

The local run used aiconfigurator 0.11.0 and TensorRT-LLM database
h200_sxm 1.3.0rc10. Its best disaggregated row is 602.586 tokens/s/GPU at
108.944 tokens/s/user, with five TP4 prefill workers and three TP4 decode
workers at batch 64. The published README snapshot reports 684.79 at 100.31
with four replicas of a different TP2/TP4 topology and decode batch 68. This is
external-tool version drift. Neither anchor is preferred, and neither was used
to fit any SimLLM parameter.

The corrected prefill miss has a candidate semantic explanation, but it is not
used to rescore this frozen study. All 10 external rows carry operating-point
fields: concurrency ranges from 27 to
288, request rate ranges from
10.063 to 41.238, and
their TTFT column is the same 196.423 ms value. The
external TTFT is therefore an operating-point quantity at concurrency, while
the SimLLM value is isolated prefill service. The frozen matched-point premise
conflates queueing with service. DEPLOY-12 owns a v2 comparison freeze that
must clarify the external semantics before any new score is defined.

Commits `8a96b3f`, `11db813` used the nonconforming `feat:` prefix. This conduct
deviation is recorded without rewriting history.

## Physical sanity

Per TP4 rank, decode batch 64 executes 1,166,009,794,560 FLOPs and moves
25,821,675,520 bytes. The compute floor is
0.589191 ms and the memory floor is
5.379516 ms, so the corrected ideal service sits exactly on
the larger memory floor. The frozen e=0.4 edge gives a
13.448789 ms envelope ceiling; the external
9.179 ms TPOT lies inside that 5.379516 to
13.448789 ms bracket.

Per TP4 rank, the uncached prefill executes 55,413,043,036,160 FLOPs and
moves 8,225,259,520 bytes. Its compute floor is
28.000527 ms while its memory floor is only
1.713596 ms, so compute is decisive. The frozen e=0.4 edge
gives a 70.001318 ms service-envelope ceiling;
the external 196.423 ms TTFT is
2.806x above it.
That outside result is exactly what X2c was frozen to detect.

At system level, the external best point and our matched point both fit 32
GPUs and the same five-plus-three TP4 pool structure. Our ideal decode and
prefill values are 58.6 and 14.3 percent of
their external columns. Decode remains a clean service comparison inside the
frozen bracket. Prefill does not: its operating-point-versus-service semantic
confound is the DEPLOY-12 residual, not a reason to widen or reinterpret the
v1 band.

## Fatal guards

| Guard | Outcome | Predicate |
|---|---|---|
| FG-1 | PASS | tracked external rows match external.sha256 |
| FG-2 | PASS | inventory geometry matches all seven frozen literals |
| FG-3 | PASS | external rows are display and comparison inputs only |
| FG-4 | PASS | pricing-lane process interception count is zero |
| FG-5 | PASS | 83bb281 is an ancestor and the frozen bytes match |

The pricing lane called only `simllm.deploy` and triggered zero process
interceptions. The external program did not run. All three efficiency arms
scanned 5,070 candidates each in one process
and completed in 19.664 seconds, below the frozen 120 s
limit. The external tool's observed 11 s search remains unscored context.

The scoring record is attempt-2. Its full predecessor is
attempt-1, and the two deterministic projections have
the same SHA-256 `f4eef6643e3324a77bd513ae257a83ef500e04489afa536f388f850e62a92112`. The comparison
excludes only wall-clock values, their W outcome, the overall verdict that
includes W, and attempt metadata.

## Project consequence

What ran: `examples/frontier_comparison_v1`, a config-only extraction-backed,
binary-free comparison of the SimLLM deployment estimator against the 10
frozen aiconfigurator 0.11.0 disaggregated H200 rows.

What came out: the run is non-void and the corrected result is
MIXED. X2 is 3 of 4 because
prefill e-star is 0.142552 outside the frozen band, and X3c is
3 of 10 against a minimum of
8. Decode e-star remains 0.586068 and
inside the band. Every X3c row and miss direction is published above.

What it changes for the project: the comparison study now delivers a validated
decode bracket and a refuted prefill matched-point premise. DEPLOY-12 registers
the v2 external-semantics freeze and COMP-81 registers the 8x attention-score
projection inconsistency. No implementation or calibration task closes; this
change closes nothing beyond publication of the corrected scoring record.

What it does not change: COMP-54 remains open for Kimi K3, DEPLOY-9 through
DEPLOY-11 remain open, TRAF-20 remains open on its separate speed
qualification, no packet-level validity claim expands, no H200 efficiency is
installed, neither tool is validated against a live serving deployment, and
the frozen expectations, anchors, earlier studies and inventories remain
unchanged.
