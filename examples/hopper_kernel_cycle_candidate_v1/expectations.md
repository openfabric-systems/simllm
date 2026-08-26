# Hopper kernel-cycle candidate freeze

This is the expectations-only freeze for the local arm of the registered
Hopper campaign. It precedes every scored compilation from the retained
evidence. It does not contain a candidate record or a compilation result.

## Reachability verdict

The one permitted probe was:

```text
timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 merlin hostname
```

It returned exit status 255 and exactly:

```text
Connection timed out during banner exchange
Connection to UNKNOWN port 65535 timed out
```

No retry was made. The local arm therefore applies.

## Evidence authority and scope

The retained A100 report is configured as
`$SIMLLM_KERNELPROBE_ROOT/REPORT.md`, SHA-256
`95c56afd0cf5f974d748a9789129b2594a765cdbcc4170a76e3e9ba3b75e95f8`.
It establishes that Nsight Systems kernel start and end timestamps are the
service authority. Nsight Compute counter passes contribute component ratios
and counters only. They do not replace service time.

The GH200 lane report is configured as
`$SIMLLM_KERNELPROBE_ROOT/gh200lane/REPORT.md`, SHA-256
`8c4f21f0fdb99fd6b007b21fa3a631523294be616f4078b08d2a1934507cb798`.
It records a ready Nsight Systems lane and denied GH200 counters under
`ERR_NVGPUCTRPERM`. The candidate may therefore use retained GH200 aggregate
kernel service. It may carry an A100 component bound as `DISCLOSED`, but it
must not claim GH200 counter attribution.

The DeepSeek shape authority is the tracked deployment projection with
content address
`ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2`.

Successful raw Nsight Systems containers remain remote. The lean local pull
contains derived Nsys CSV tables, source hashes, clock samples and reports.
This freeze calls those retained Nsys evidence without claiming the raw
containers are present locally.

## Exact retained cell inventory

Granite has retained Nsys evidence for 12 exact vLLM TP1 cells:

- Graph decode at batch 1, 8 and 32 with KV length 16.
- Graph prefill at lengths 128, 512 and 2,048.
- Eager decode at batch 1, 8 and 32 with KV length 16.
- Eager prefill at lengths 128, 512 and 2,048.

The graph batch-32 cell comes from
`gh200lane/granite-repeats-198869-tp1-graph`. The other five graph cells come
from `gh200lane/granite-repeats-198874-tp1-graph`. The six eager cells come
from `gh200lane/granite-repeats-198878-tp1-eager`.

The initialization failure in job 198853 and the partial batch-32 trace in
job 198858 are absent. More importantly, none of the 1,212 rendered cells in
the registered `kernel-cycle-v1` suite has run as a registered GH200
campaign. The 12 retained scouting anchors do not replace that grid.

DeepSeek-V3 has retained Nsys evidence for four reduced-depth TP1 physical
envelope cells:

- EP32 dynamic prefill ranks 16 by 1,024, 8 by 2,048 and 4 by 4,096, each
  carrying 16,384 new tokens per rank, from job 198883.
- EP72 dynamic decode batch 32 at context 2,000, selected by the final runtime
  correlation boundary from job 198891.

Job 198883's capture-level compact-analysis verdict stayed blocked, so only
the three report-accepted prefill cell summaries are eligible. Jobs 198886
and 198889 do not provide the exact batch-32 decode boundary. Job 198880
failed before timing. The EP72 MTP batch-16, context-4,000 arm did not run and
is `ABSENT`.

## Evidence classes

Every emitted lookup row and both compiled projections must name two axes:

- `MEASURED` service is the retained GH200 Nsys sum of noncollective kernels
  at the exact physical envelope and shape.
- `DISCLOSED` component evidence is an A100 counter-derived bound transferred
  only as a bound. It cannot replace GH200 service or become a GH200 counter
  claim.
- `DECLARED` service is explicit arithmetic. The only permitted instance is
  the DeepSeek full-depth value computed as the four-layer service times
  `61 / 4`, rounded half up in integer picoseconds.
- `ABSENT` means no lookup, profile-table or device-service row is emitted.

The service and component axes are not added. Granite expects 12 `MEASURED`
service entries with 12 `DISCLOSED` component overlays, while all 1,212
registered campaign cells remain `ABSENT`. DeepSeek expects four `MEASURED`
reduced-depth entries, four `DECLARED` full-depth entries and eight
`DISCLOSED` component overlays. Of its five requested physical cells, four
are measured and the MTP cell is absent.

## Calibration and held-out split

Granite calibration uses batch 1 and 8 decode plus length 128 and 512 prefill
in each launch mode. Its batch-32 decode and length-2,048 prefill cells are
held out in each launch mode.

DeepSeek calibration uses prefill 16 by 1,024, prefill 8 by 2,048 and the
batch-32 decode cell. Prefill 4 by 4,096 is held out. The unexecuted MTP cell
also remains held out, but it is a fatal absence rather than a failed score.
No fitting path may read a held-out value.

## Frozen distribution verdicts

The Granite graph-decode family must retain a single clock bin, at least 256
replays and a trimmed coefficient of variation no larger than 0.5 percent.
Its expected record verdict is `tight-single-peak`.

Granite graph prefill and all eager cells have only 30 retained repeats. Their
observed service coefficients remain below 0.5 percent, but the landed record
requires 256 graph or 64 eager replays. Their record verdict is therefore
`insufficient-replays`.

Each retained DeepSeek shape is a seed execution, not a repeat distribution.
Its record verdict is `insufficient-replays`, and a one-sample structural zero
coefficient never becomes a stability claim.

## Frozen ratio envelopes

The retained GH200 lane compares different pinned software stacks, so these
are bounded A100-over-GH200 indicators, not pure architecture ratios:

| Family | Frozen envelope | Retained range |
|---|---:|---:|
| Granite graph decode | 1.75x to 1.90x | 1.776x to 1.853x |
| Granite graph prefill | 1.75x to 2.65x | 1.805x to 2.577x |
| Granite eager decode | 1.85x to 2.10x | 1.916x to 2.014x |
| Granite eager prefill | 1.90x to 2.90x | 1.969x to 2.789x |

No matched A100 DeepSeek capture exists at the retained implementation and
shapes. The DeepSeek A100-over-GH200 ratio is `ABSENT`, never inferred from
Granite.

## Frozen KV slope bounds

The A100 counter and light-timeline split supplies these disclosed bounds:

| Family | Center | Frozen 10 percent bound | Use |
|---|---:|---:|---|
| Granite TP1 FlashAttention | 1.450 ns per token per layer | 1.305 to 1.595 ns | Granite component bound |
| Qwen TP1 full attention | 2.493 ns per token per layer | 2.244 to 2.742 ns | Cross-family context only |
| Qwen TP4 full attention | 0.904 ns per token per layer | 0.814 to 0.994 ns | TP-width-specific context only |

The Qwen bounds must not be applied to DeepSeek multi-head latent attention.
One retained DeepSeek decode context and no MTP cell cannot identify a
DeepSeek KV slope, so that slope stays `ABSENT`.

The page-placement evidence is shape dependent. Granite TP1 batch 1 at KV
2,048 is scatter-insensitive within noise. Qwen TP4 batch 1 has a 2.162
percent split-KV penalty with a 2.059 to 2.255 percent interval, while batch
32 has no full-step penalty. No GH200 fragmentation row exists, so the
candidate emits no GH200 fragmentation correction.

## Physical floors

The GH200 report fixes a 4,022.78 GB/s HBM roof and about 1.070 PFLOP/s dense
BF16 tensor roof for these checks. Before accepting a compiled duration:

- Granite decode batch 1, 8 and 32 must each be at least 0.199 ms per step.
- Granite prefill length 128, 512 and 2,048 must be at least 0.199, 0.383 and
  1.531 ms.
- DeepSeek prefill 16 by 1,024, 8 by 2,048 and 4 by 4,096 must be at least
  37.05, 38.32 and 40.89 ms.
- DeepSeek decode batch 32 at context 2,000 must be at least 1.115 ms.

These are rejection floors, not predictions.

## DeepSeek depth projection

The physical capture retains four layers: the three early dense layers and
the first mixture-of-experts layer. The only allowed full-depth arithmetic is

```text
per_layer_service_ps = reduced_service_ps / 4
full_service_ps = round_half_up(reduced_service_ps * 61 / 4)
```

Every full-depth row is `DECLARED`. This average does not isolate dense from
mixture-of-experts layers. It does not claim full-depth measurement,
distributed expert residency, expert-parallel communication, MTP timing or
rank-class-specific service.

## Compilation guards and consequences

Compilation is void if a configured source digest differs, a physical floor
is crossed, a row lacks its evidence classes, a held-out row enters fitting,
an absent cell appears in any output, a collective enters service, or the
lookup and compiled durations differ. The lookup must validate as candidate
under `simllm-kernel-cycle-lut-v1`. Its profile-table and device-service forms
must select by the same content address.

No result can claim validated status, full GH200 coverage, GH200 counter
attribution, MTP pricing, or closure of COMP-1, COMP-5, COMP-64 or CORE-54.

## Deferred Merlin entry point

COMP-72 owns the deferred registered execution. It runs under this freeze and
uses the exact commands stored in `expectations.json`. The Granite plan is
rendered by the landed capture driver. Each cell resumes by setting
`SIMLLM_CAMPAIGN_CELL_ID` to the first cell without a digest-complete output
directory. Completed cells are never overwritten. DeepSeek base, exact
decode and MTP use the pinned short `gh-hourly` submissions stored beside the
plan commands. Any new SSH loss stops execution cleanly.
