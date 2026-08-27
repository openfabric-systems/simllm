# Local Hopper kernel-cycle candidate result

## Campaign successor, 2026-08-27

Successor record digest:
`d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107`.
The flagship MTP cell measures 2.033951 ms of non-collective service from 119
GPU kernel records selected by 109 runtime correlations at the exact
`generation_16(32)` NVTX boundary. Its service class moves from `ABSENT` to
`MEASURED`, with its component split `DISCLOSED`; the freeze still forbids an
MTP lookup price.

The campaign result is `PARTIAL_CAMPAIGN_RECOMPILED`. The exact DeepSeek MTP,
base and decode submissions ran as jobs 198967, 198985 and 198987 on the
`gh-hourly` partition, paced one at a time. No model weight was downloaded.
The base and decode runs retain a second independent observation for every
priced physical key. No repeat is pooled and no distribution is changed
before COMP-74.

The staged compact analyzer passed the decode job but blocked after the MTP
and base captures because its all-cells decode boundary did not match those
runtime labels. Both blocked outputs are preserved. A fail-closed recovery
reads the exact registered profile cases and NVTX runtime correlations from
their retained SQLite sources, rejects any weight file, and records each raw
source byte count and digest.

The 1,212-cell Granite plan rendered canonically with SHA-256
`19a6d6fb93d7b3f4faa1ad9cdd94724ecddb9fdae147f71c0f8403a5c0b19d36`.
Neither pinned framework target executable named by
`SIMLLM_VLLM_KERNEL_CAPTURE_TARGET` and
`SIMLLM_SGLANG_KERNEL_CAPTURE_TARGET` was staged on Merlin, so no real
Granite cell could start. The digest-complete prefix therefore remains the
empty sequence with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
byte for byte. The first incomplete cell remains
`sglang-decode-cuda-graph-te1-pi1-da1-ex1-b1-kv1311-deliberately-fragmented`.

CORE-61 was not submitted because its frozen earliest time,
`2026-08-28T06:30` in `Europe/Zurich`, had not arrived. The preregistered
3,751,359,511 ps depth-8 prediction therefore retains no measurement or
signed residual. COMP-72 stays open and COMP-78 owns this exact Granite and
CORE-61 remainder.

## Frozen predecessor

The immutable predecessor digest is
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`.
It remains the original `candidate` `simllm-kernel-cycle-lut-v1` artifact with
20 service entries and is not a validated device release. The successor
preserves every predecessor service point and distribution; its content
address changes because the new retained source digests are attached.

What was frozen: the expectations-only commit `748e6b6` fixed the one-shot
Merlin reachability verdict, every retained source digest, the exact Nsys cell
inventory, the calibration and held-out split, distribution and ratio bands,
KV-slope tolerances, physical floors, the only allowed DeepSeek depth
arithmetic, and the deferred commands. The probe command was:

```text
timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 merlin hostname
```

It exited 255 with this verbatim output:

```text
Connection timed out during banner exchange
Connection to UNKNOWN port 65535 timed out
```

It was not retried.

What ran for the predecessor: after the freeze, the local compiler verified
the byte count and SHA-256 of every retained report and derived CSV,
cross-checked the selected cell values against those CSVs, verified the
tracked DeepSeek projection, compiled the lookup, and projected it through
the existing profile-table and device-service compilers. The portable local
command is:

```text
SIMLLM_KERNELPROBE_ROOT=$SIMLLM_KERNELPROBE_ROOT \
  .venv/bin/python examples/hopper_kernel_cycle_candidate_v1/run_study.py \
  --output-dir $SIMLLM_CAMPAIGN_RUN_ROOT/hopper-kernel-cycle-candidate-v1
```

What came out then: `CANDIDATE_COMPILED`. All frozen Granite distribution, ratio
and physical-floor checks passed. The DeepSeek cells stayed
`insufficient-replays`, as frozen, and no stability result is inferred from a
single retained observation. Held-out entries underwent the same identity
transform as calibration entries; the compiler has no fitted parameter that a
held-out value could influence.

What this changes now: consumers can select the successor by its new content
address, the unpriced MTP anchor has measured GH200 Nsys service, and COMP-74
has two independent observations for all four priced keys.

What this does not change: no frozen lookup point or distribution changes, no
MTP row is priced, no full-depth DeepSeek model ran, no Granite campaign cell
ran, and no candidate is promoted. COMP-1, COMP-5, COMP-64, COMP-72 and
CORE-54 remain open.

## Evidence-class ledger

Service and component rows are separate, non-additive ledgers. A row with
`MEASURED` service and `DISCLOSED` components counts once in each named axis,
not twice as service.

| Family | Lookup MEASURED | Lookup DECLARED | Lookup DISCLOSED | Requested physical MEASURED | Requested physical ABSENT |
|---|---:|---:|---:|---:|---:|
| Granite 3.0 1B A400M | 12 | 0 | 12 | 0 | 1,212 campaign cells |
| DeepSeek-V3 COMP-72 cells | 4 reduced-depth | 4 full-depth projections | 8 | 5 | 0 |

The only class movement is the registered MTP physical cell from `ABSENT` to
`MEASURED` plus one `DISCLOSED` component overlay. The four priced keys remain
`MEASURED` at reduced depth and `DECLARED` at full depth; their new independent
repetitions extend provenance but do not reclassify or reprice them. The
CORE-61 depth-8 held-out cell remains separately `ABSENT` under its time gate.

Every emitted lookup, profile-table and device-service entry carries its
service class, component class, split and source digests. Granite and DeepSeek
routing vectors were not retained, so routed keys say `not-captured`; no
expert loads are invented. The A100 counter evidence remains a `DISCLOSED`
component bound. It does not replace the GH200 Nsys service authority.

The remaining absent evidence is:

- all 1,212 registered Granite campaign cells;
- the CORE-61 depth-8 held-out physical cell and signed residual;
- GH200 program-counter or counter-pass attribution;
- full-depth DeepSeek silicon service;
- tensor-parallel widths not present in the lean pull; and
- retained per-cell expert-load vectors.

## DeepSeek values for flagship pricing

The retained physical envelope has four layers: the three dense layers and
the first MoE layer. `MEASURED` below means exact retained GH200 Nsys service
for that four-layer physical envelope. `DECLARED` means only the frozen
`61 / 4` arithmetic. It does not mean full-depth silicon measurement.

| Per-rank cell | Four-layer rank step, MEASURED | Per-layer basis, MEASURED | 61-layer rank step, DECLARED | Flagship entry class |
|---|---:|---:|---:|---|
| EP32 prefill, 16 x 1,024 input tokens | 89.393440 ms | 22.348360 ms | 1,363.249960 ms | DECLARED |
| EP32 prefill, 8 x 2,048 input tokens | 93.134208 ms | 23.283552 ms | 1,420.296672 ms | DECLARED |
| EP32 prefill, 4 x 4,096 input tokens | 104.598911 ms | 26.14972775 ms | 1,595.13339275 ms | DECLARED, held-out |
| EP72 decode, batch 32 at KV 2,000 | 1.875680 ms | 0.468920 ms | 28.604120 ms | DECLARED |
| EP72 MTP decode, batch 16 at KV 4,000 | 2.033951 ms | not derived | not priced | MEASURED, unpriced |

The per-layer values are reported to expose the arithmetic consumed by the
flagship. The actual lookup keys select the 61-layer per-rank entries, not a
new per-layer pricing authority.

The independent priced-key repetitions are retained as COMP-74 inputs:

| Physical key | Frozen point | Independent repeat | Signed repeat minus point |
|---|---:|---:|---:|
| EP32 prefill, 16 x 1,024 | 89.393440 ms | 91.249600 ms | +1.856160 ms |
| EP32 prefill, 8 x 2,048 | 93.134208 ms | 94.656736 ms | +1.522528 ms |
| EP32 prefill, 4 x 4,096 | 104.598911 ms | 104.294464 ms | -0.304447 ms |
| EP72 decode, batch 32 at KV 2,000 | 1.875680 ms | 1.883392 ms | +0.007712 ms |

These are raw signed observations only. COMP-74 owns the preregistered
statistic and interval rule.

## Granite checks

The 12 exact TP1 cells cover graph and eager service at decode batches 1, 8
and 32 with KV 16, and prefill lengths 128, 512 and 2,048. The largest shape
per pool and launch mode was held out.

- Graph decode retained 300 replays per cell, one 1,980 MHz clock bin and
  trimmed CV from 0.154 to 0.423 percent. All three cells satisfy the frozen
  tight-single-peak and 0.5 percent ceiling.
- Graph prefill and all eager cells retained only 30 repetitions. Their CVs
  are below 0.5 percent, but their verdict remains `insufficient-replays`
  where the protocol demands more evidence.
- A100 over GH200 service ratios remained inside the frozen family envelopes:
  1.776 to 1.853 for graph decode, 1.805 to 2.577 for graph prefill, 1.916 to
  2.014 for eager decode, and 1.969 to 2.789 for eager prefill.
- Every service exceeded its GH200 roof-derived floor: 0.199 ms for decode
  and length-128 prefill, 0.383 ms at length 512, and 1.531 ms at length
  2,048.

The retained A100 KV bounds remain context, not GH200 measurements: Granite
TP1 FlashAttention is 1.450 ns per token per layer with the frozen 1.305 to
1.595 tolerance. Qwen TP1 and TP4 bounds are retained as cross-family checks
only. No DeepSeek MLA KV slope was present.

## Published artifacts

| Artifact | SHA-256 | Bytes | Status |
|---|---|---:|---|
| successor `candidate-record.json` | `d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107` | 60,675 | candidate lookup |
| successor `profile-table.json` | `ff5f4d5376a90a82fd3588355c3671e02c06778290dc906416374b496f2e08f1` | 33,587 | candidate profile-table projection |
| successor `device-service-entries.json` | `1a8815dcdf3d8df4f8143b845c19be9c3b8a93846e1696cf713eaba2767d29b5` | 49,255 | candidate device-service projection |
| successor `result.json` | `e38f08dd231b6dec681b90135a5ad9c9588f4627df9778680379572c2ae0a20d` | 3,629 | partial campaign result |
| successor `artifact-manifest.json` | `e5a68bdcedc375996b99ab38f7f9b951b883a6929be57416cc82b7dade25051e` | 704 | artifact ledger |
| immutable predecessor `candidate-record.json` | `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52` | 57,417 | immutable history |

The manifest SHA-256 was computed after the four payload artifacts. It is not
included in its own payload list.

## Run integrity and deferral

The COMP-72 remote campaign is retained under the registered campaign root,
and the lean local pull retains control digests, exact profiles, derived
kernel tables and recovery outputs under the external bulk root. Full Nsys
SQLite and report sources remain remote with their byte counts and SHA-256
digests embedded in each recovery output.

Three earlier stopped local attempts are preserved under the bulk run root.
The first stopped before a result because a zero-count ledger key was omitted.
The second was rejected during publication audit because its Granite label and
family did not match the retained report. The third used an explanatory
synthetic name instead of the exact full-depth DeepSeek identity and was also
rejected before publication. None is published. Regression tests cover the
ledger and exact model identities, and the final digest above binds both the
corrected Granite 3.0 1B A400M routed identity and the exact disclosed
DeepSeek-V3 target identity.

COMP-78 now owns the literal COMP-72 remainder under `expectations.json`. It
resumes the rendered Granite plan at the first cell without a digest-complete
directory:

```text
.venv/bin/python offline/calibration/kernel_cycle_capture.py run-cell \
  --plan $SIMLLM_CAMPAIGN_RUN_ROOT/granite-plan.json \
  --cell-id $SIMLLM_CAMPAIGN_CELL_ID \
  --output-dir $SIMLLM_CAMPAIGN_RUN_ROOT/cells/$SIMLLM_CAMPAIGN_CELL_ID
```

The same freeze carries the exact short `gh-hourly` DeepSeek base, decode and
MTP submissions. Completed cells are never overwritten, and any new SSH loss
stops the run cleanly.
