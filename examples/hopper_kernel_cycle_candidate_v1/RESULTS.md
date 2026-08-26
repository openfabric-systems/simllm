# Local Hopper kernel-cycle candidate result

## Outcome

Record digest: `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`.
The record is a `candidate` `simllm-kernel-cycle-lut-v1` artifact with 20
service entries. It is not a validated device release.

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

What ran: after the freeze, the local compiler verified the byte count and
SHA-256 of every retained report and derived CSV, cross-checked the selected
cell values against those CSVs, verified the tracked DeepSeek projection,
compiled the lookup, and projected it through the existing profile-table and
device-service compilers. No model weight was downloaded. The portable local
command is:

```text
SIMLLM_KERNELPROBE_ROOT=$SIMLLM_KERNELPROBE_ROOT \
  .venv/bin/python examples/hopper_kernel_cycle_candidate_v1/run_study.py \
  --output-dir $SIMLLM_CAMPAIGN_RUN_ROOT/hopper-kernel-cycle-candidate-v1
```

What came out: `CANDIDATE_COMPILED`. All frozen Granite distribution, ratio
and physical-floor checks passed. The DeepSeek cells stayed
`insufficient-replays`, as frozen, and no stability result is inferred from a
single retained observation. Held-out entries underwent the same identity
transform as calibration entries; the compiler has no fitted parameter that a
held-out value could influence.

What this changes: CORE-54 and the CORE-53 binding seam can identify this
candidate by content address and consume its exact per-rank durations once the
binding lands. Current main does not contain the CORE-53 seam, so the proof is
against the landed kernel-cycle record, profile and device-service contracts.
The contract test validates the content address, exact profile duration,
candidate acceptance status and one-for-one device-service projection without
a binding adapter.

What this does not change: no registered Merlin campaign cell ran, no GH200
counter pass exists, no full-depth DeepSeek model ran, no MTP row is priced,
and no task closes. COMP-1, COMP-5, COMP-64 and CORE-54 remain open.

## Evidence-class ledger

Service and component rows are separate, non-additive ledgers. A row with
`MEASURED` service and `DISCLOSED` components counts once in each named axis,
not twice as service.

| Family | MEASURED service | DECLARED service | DISCLOSED component bounds | ABSENT requested cells |
|---|---:|---:|---:|---:|
| Granite 3.0 1B A400M | 12 | 0 | 12 | 1,212 registered campaign cells |
| DeepSeek-V3 | 4 reduced-depth | 4 full-depth projections | 8 | 1 MTP physical cell |

Every emitted lookup, profile-table and device-service entry carries its
service class, component class, split and source digests. Granite and DeepSeek
routing vectors were not retained, so routed keys say `not-captured`; no
expert loads are invented. The A100 counter evidence remains a `DISCLOSED`
component bound. It does not replace the GH200 Nsys service authority.

The remaining absent evidence is:

- all 1,212 registered Granite campaign cells;
- DeepSeek EP72 MTP batch 16 at KV 4,000;
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
| EP72 MTP decode, batch 16 at KV 4,000 | ABSENT | ABSENT | ABSENT | ABSENT |

The per-layer values are reported to expose the arithmetic consumed by the
flagship. The actual lookup keys select the 61-layer per-rank entries, not a
new per-layer pricing authority.

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
| `candidate-record.json` | `ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52` | 57,417 | candidate lookup |
| `profile-table.json` | `116f8fc58d82a92d87653a0e5d6203a5a6ede1fdd31759493367657ea85033f4` | 32,963 | candidate profile-table projection |
| `device-service-entries.json` | `bf1f4a2a9fcba459de864b1f6511d33eab77e656330d002d44cee5452eb508a3` | 48,631 | candidate device-service projection |
| `result.json` | `8112afe70fdd5b8f9c47f4e3af4f567f848862dc967832f142abd39fdab77e6e` | 3,091 | scored result |
| `artifact-manifest.json` | `3777d98495985312cdd8c411adf606a5f1d1cfdad7f85c571c4d4a8225e3c93f` | 704 | artifact ledger |

The manifest SHA-256 was computed after the four payload artifacts. It is not
included in its own payload list.

## Run integrity and deferral

Three stopped local attempts are preserved under the bulk run root. The first
stopped before a result because a zero-count ledger key was omitted. The
second was rejected during publication audit because its Granite label and
family did not match the retained report. The third used an explanatory
synthetic name instead of the exact full-depth DeepSeek identity and was also
rejected before publication. None is published. Regression tests cover the
ledger and exact model identities, and the final digest above binds both the
corrected Granite 3.0 1B A400M routed identity and the exact disclosed
DeepSeek-V3 target identity.

COMP-72 owns the deferred Merlin run under `expectations.json`. It starts by
rendering the Granite plan with the exact command in that freeze, then resumes
at the first cell without a digest-complete directory:

```text
.venv/bin/python offline/calibration/kernel_cycle_capture.py run-cell \
  --plan $SIMLLM_CAMPAIGN_RUN_ROOT/granite-plan.json \
  --cell-id $SIMLLM_CAMPAIGN_CELL_ID \
  --output-dir $SIMLLM_CAMPAIGN_RUN_ROOT/cells/$SIMLLM_CAMPAIGN_CELL_ID
```

The same freeze carries the exact short `gh-hourly` DeepSeek base, decode and
MTP submissions. Completed cells are never overwritten, and any new SSH loss
stops the run cleanly.
