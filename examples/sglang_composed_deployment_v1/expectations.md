# Expectations: the composed realistic-deployment SGLang study (SGL-27)

This file is the pre-run freeze. It lands before any driver code exists and
before any cell is executed. Every band, relation, guard and entailment answer
below is written against the mechanisms as they stand at base commit
`2f0745d`, and nothing in it is edited after a measurement is read. A relation
that fails is reported as failed.

The study runs the live in-process SGLang chain of
`examples/sglang_end_to_end_v1` as two realistically structured deployments and
prices both with all four wave-14 mechanisms at once: the corrected
tensor-parallel allreduce site inventory, the per-collective fixed-cost
envelope with named arms, the per-request medium attribution under mixed
NVLink and fabric locality, and the SGLang host-model selection seam.

Allocated task IDs: SGL-27, SGL-28, SGL-29, and TRAF-42 if a traffic-side
deferral is genuinely discovered. No other ID may be created by this work.

## Chronology, stated plainly

1. The four wave-14 mechanisms landed on `main` before this file.
2. A premise probe was run before this freeze, as the task brief requires. It
   executed one synthetic single-record step per cell shape, outside the live
   chain, purely to answer the structural questions in "Premises verified"
   below. Its observed values are reproduced there so that no later reader can
   mistake them for predictions. They anchor the closed form; they determine no
   live-chain quantity, because the probe never ran a scheduler.
3. This file is committed with no implementation and no results.
4. Only then is the driver written and the matrix executed.

This is a genuine pre-registration of the scored relations, and the probe
disclosure above is what keeps it genuine rather than a claim that hides a
partial look at the answer.

## External-source audit

Every constant this study charges that was not produced inside this repository,
with the file and line where the repository records it, and what the transfer
does to it.

| constant | value | recorded at | class |
|---|---|---|---|
| B200 intra-node NCCL all-reduce intercept, width 8 | 30,128,029 ps | `simllm/traffic/collective_latency.py:387-421` | calibrated at source, applied here to two different operation shapes |
| its declared uncertainty band, width 8 | [30,048,029, 30,208,029] ps | `simllm/traffic/collective_latency.py:415-419` | sample-limited fit residual, not a confidence interval |
| its endpoint bandwidth | 70,027,079,100 B/s | `simllm/traffic/collective_latency.py:389` | fitted slope of the same capture |
| capture identity | nccl-tests issue 333 attachment, 8x B200, driver 570.158.01, CUDA 12.9, NCCL 2.27.0a0, widths 2/4/8, `-R 1` | `examples/collective_latency_floor_v1/expectations.md:70-84` | public user-supplied capture, not an NVIDIA guarantee, not a local measurement |
| cross-node provisional intercept, width 8 | 49,487,789 ps | `simllm/traffic/collective_latency.py:448-505` | provisional-transferred, no cross-node measurement exists |
| its ring-step anchors | 2.000 / 3.000 / 5.000 us | `simllm/traffic/collective_latency.py:376-378` | see the two rows below |
| Kalia et al. ATC'16 commodity RDMA round trip | about 2 us | `docs/papers/msg-size-vs-bandwidth.md:89-91` | halved to 1.000 us and added to the fluid propagation reference for the point estimate |
| UCCL Table 2 p50 ACK turnaround, Light columns | 2.0 to 3.0 us | `docs/papers/msg-size-vs-bandwidth.md:33-42` | the 3.0 us top is added for the upper edge; the Heavy columns reach 7.0 us and would move the edge, and the repository already says so at that line |
| fluid per-collective propagation | 2,000,000 ps | `simllm/traffic/collective_latency.py:370` | measured inside this repository from its own backend |
| structural floor arm | 0 ps surcharge | `simllm/traffic/collective_latency.py:423-446` | structural-floor, no collective can beat one propagation delay |
| CUDA-graph node replay point | 809,306 ps per launch | `simllm/compute/host.py:167-171`, captured in `examples/host_step_cost_v1/RESULTS.md:213, 236` | measured on one GTX 1660 Ti (Turing, sm75) with an AMD Ryzen 9 3950X host; not SGLang, not a datacenter GPU |
| its empirical range | [624,665, 809,306] ps | `examples/host_step_cost_v1/RESULTS.md:236` | five samples, explicitly not a confidence interval |
| launch count per step | 440 | `simllm/adapters/sglang/host.py:69-85` | static enumeration of vLLM 0.26.0 sources, `examples/compute_fidelity_v1/RESULTS.md:336`; SGLang's own count is unmeasured and is SGL-24 |
| compute envelope | b100, 8.0e12 B/s, 1.8e15 FLOP/s | `simllm/compute/transformer.py:36` `GPU_ENVELOPES` | vendor envelope used as a physical bound |
| model geometry and routing | granite-3.0-1b-a400m-instruct at `ffec3c35`, SGLang at `8f2a3ad6` | `examples/sglang_end_to_end_v1/run_study.py:27-50` | captured by this repository from the pinned framework |

Three of these are already known to be misapplied in a specific, named way,
and this study inherits all three rather than fixing any of them:

- the B200 intercept is an ALL-REDUCE capture. In the intra-node cell it prices
  24 all-reduces per step at their captured operation shape and interconnect,
  which no earlier study in this repository could say, and 48 all-to-allvs per
  step by transfer. In the cross-node cell it prices 48 all-to-allvs by
  transfer and across an interconnect it was not captured on. The envelope's
  own point-of-use downgrade collapses this to one class per cell
  (`simllm/traffic/collective_latency.py:677-720`); this study reports the
  split itself.
- the host constant is a Turing consumer GPU with a desktop host, presented to
  the model under the `gtx1660-ti-sm75` device key while compute is priced
  against b100. Every enabled row is a three-source device hybrid, and
  `SGLANG_HOST_TRANSFER_DISCLOSURE`
  (`simllm/adapters/sglang/host.py:90-103`) is carried verbatim in the results.
- the launch count is vLLM's, for a model runner SGLang does not use.

Nothing in this study is a calibration. The claim language is fixed in advance:
arms, envelopes, brackets, transfers. Never "the collective costs X".

## Premises verified before this freeze

Each was checked against the code and, where structural, against a synthetic
single-record probe. File and line references are to base commit `2f0745d`.

**P1. The end-to-end driver's construction points.** The sink is built at
`examples/sglang_end_to_end_v1/run_study.py:650-665`, the worker receives it
through `configure(step_sink=sink, ...)` at `:668-676`, the in-process
scheduler is built at `:678-689`, the pump wraps it at `:695`, and the
arrival gate is built on the worker's own clock at `:696-700`. The loop that
alternates admission and stepping is `:718-746`. **Holds.**

**P2. A declared placement manifest routes the intra-node cell's segments to
the NVLink serializer rather than to htsim.** `HtsimStepSinkConfig` builds a
`RankMapper` from the manifest and takes the compatibility fast path only when
the manifest is absent or the step carries no NVLink bytes
(`simllm/backends/step_sink.py:870-872`). Directed segments are classified by
the manifest and local ones are served by the analytic per-endpoint serializer
of `simllm/traffic/locality.py:1-33`. **Holds, and more strongly than the brief
assumed:** with all eight ranks on one host the probe reports
`backend_runs = 0`, so the intra-node cell invokes `htsim_rnic` zero times and
its `linkspeed_bps` is inert. That is registered as a fatal guard below rather
than left implicit.

**P3. The intra-node collective inventory is 24 + 48, not 96.**
`layer_tp_allreduce_sites` returns the attention site alone whenever
`renders_expert_combine` is true (`simllm/traffic/step_comm.py:158-223`), and
a captured `RoutedMoeSupply` makes it unconditionally true for a step with
tokens (`simllm/traffic/step_comm.py:150-151`). The probe reports 24 `TpAllReduce` operations, 48
`MoeAllToAll` operations, sites `("attention",)`. **Holds.**

**P4. The envelope selection surface.** `collective_fixed_cost_envelope` plus
`collective_fixed_cost_arm` are validated at
`simllm/backends/step_sink.py:265-305`, are mutually exclusive with the bare
`collective_latency_profile` spelling (`:281-288`), and the `off` arm resolves
to no profile at all (`simllm/traffic/collective_latency.py:647-658`). The
profile gate that applies to this study is at
`simllm/backends/step_sink.py:315-322`: any resolved profile requires
`profile="rnic-nn-fluid"`, which is the profile the end-to-end study already
runs, so the gate is satisfied by the inherited configuration and not by a
change made for it. **Holds.**

**P5. A refuted sub-premise inside P4.** The brief describes the arms as a
fixed-cost bracket. For the cross-node cell that is exactly what they are: the
arms differ only in an additive per-collective constant, because that cell has
no local segments. For the **intra-node** cell it is not. A resolved profile
also replaces the declared NVLink endpoint rate with the profile's own fitted
bandwidth (`simllm/backends/step_sink.py:809-813`), so moving from `off` to
`lower` charges no surcharge and still slows every NVLink endpoint from
450,000,000,000 B/s to 70,027,079,100 B/s. The intra-node `lower` arm is
therefore a bandwidth arm, not a null arm. This study runs all three intra-node
arms precisely so the bandwidth change and the surcharge change are separated
rather than confounded, and it reports the `lower` arm under that name.

**P6. The w14d selection seam contract.** `SglangHostSelection.worker_overrides`
returns `host_model`, `gpu` and `compute_provider` for `configure`, and
`sink_overrides` returns `host_model`, `gpu` and `provider` for
`HtsimStepSinkConfig` (`simllm/adapters/sglang/host.py:174-190`). The adapter
requires the worker and the sink to select the same host model
(`simllm/adapters/sglang/worker.py:151-170`); splatting one selection into both
accessors satisfies it by construction because both carry the same object.
`ideal()` is the byte-identical off arm (`simllm/adapters/sglang/host.py:243-257`).
**Holds.**

**P7. The end-to-end study's fatal guards under this matrix.** Chunked prefill
stays disabled (`chunked_prefill_size=-1`), so the pump's narrowed
chunked-prefill gate stays inert and no prefill row can carry a context length
below the whole prompt. Radix zero-hit and no-retraction are properties of the
same four requests with the same distinct prompts, and this study changes only
the priced deployment, not the request set. All three are re-asserted here as
fatal guards rather than assumed.

**P8. BACK-44 is real and this matrix deliberately avoids it.** The probe
reproduced the refusal verbatim for `tp_ranks=(0, 1)` with
`ep_ranks=(0, 1, 2, 3)`:

```
ValueError: graph cannot be represented by ordered GOAL artifacts:
'step-0:layer-0:tp-attention' does not depend on 'step-0:layer-0:rank-2:compute'
```

The refusal fires when the tensor-parallel group is a strict subset of the
expert-parallel group, because the allreduce of a layer then does not depend on
the compute of the expert-parallel ranks outside it. Both cells of this study
avoid it by construction: the intra-node cell sets `tp_ranks == ep_ranks ==
(0..7)`, so every rank's compute precedes the allreduce and the ordering
exists, and the cross-node cell sets `tp_ranks = (0,)`, so there is no
tensor-parallel collective at all. The canonical realistic composition, a
tensor-parallel group inside a node with an expert-parallel group across nodes,
remains unrepresentable and remains BACK-44's. This study neither closes nor
weakens it, and the negative control is executed as a fatal guard so the
refusal is evidence rather than a recollection.

**P9. Probe values, recorded so they cannot later be mistaken for predictions.**
One synthetic 8-token prefill record, request `p0`, on each cell shape:

| shape | step latency, ps | artifacts | backend runs | compute service, ps | surcharge total, ps |
|---|---|---|---|---|---|
| intra, off, ideal | 85,316,000 | 408 | 0 | 75,264,000 | 0 |
| intra, upper, ideal | 2,308,228,088 | 408 | 0 | 75,264,000 | 2,169,218,088 |
| intra, upper, turing | 2,589,059,088 | 408 | 0 | 356,095,000 | 2,169,218,088 |
| cross, off, ideal | 270,048,688 | 72 | 48 | 98,928,000 | 0 |
| cross, off, turing | 527,215,688 | 72 | 48 | 356,095,000 | 0 |

The cross-node off/ideal row equals the first prefill step of the accepted
`sglang_end_to_end_v1` `ep8-400g` cell, which is why E2 below can be scored
against a published artifact.

## The frozen deployment matrix

Model and request set are inherited unchanged from
`examples/sglang_end_to_end_v1`: granite-3.0-1b-a400m-instruct at `ffec3c35`,
CPU, float32, `tp_size=1` inside SGLang, chunked prefill disabled, four
requests `p0` to `p3` with eight distinct prompt tokens each, twelve new
tokens each, arrivals one millisecond apart on the worker's virtual clock,
routing from the SGL-16 strict v2 framework trace, backend profile
`rnic-nn-fluid`, `RooflineProvider(0.7)`.

Two declared topologies, both eight ranks:

**Intra-node.** One host. `tp_ranks = (0..7)` and `ep_ranks = (0..7)`. Dims are
the per-rank sharded geometry of that deployment: 2 attention heads, 1 KV head,
dense intermediate 64, and 4 resident experts. Every directed segment is
NVLink. The renderer emits 24 attention allreduces and 48 MoE all-to-alls per
full-model step, so 72 collectives and, with the 14 ring phases of each
allreduce and the 24 per-layer compute artifacts, 408 executed artifacts.
Collective arms: `off`, `lower`, `upper` of `intra-node-fixed-cost-v1`.

**Cross-node.** Eight hosts. `tp_ranks = (0,)` and `ep_ranks = (0..7)`. Dims
are the unsharded attention geometry with 4 resident experts, exactly as the
end-to-end study declares. Every directed segment crosses the fabric. 48 MoE
all-to-alls and 24 compute artifacts, so 72 executed artifacts and 48
`htsim_rnic` runs per step. Link rates 400 and 100 Gbit/s. Collective arms:
`off`, `lower`, `upper` of `cross-node-fixed-cost-provisional-v1`.

Host arms in both topologies: `ideal` and `turing-cuda-graph` at 440 launches,
both resolved through `select_sglang_host_model`.

That is 6 intra-node cells and 12 cross-node cells, 18 in total. The matrix is
small on purpose and is not extended after the run.

The two topologies are two different deployments, not one controlled
experiment. They differ in locality, in the presence of a tensor-parallel
group, and therefore in per-rank compute. Every intra-node versus cross-node
number below is reported as a deployment comparison and never as an isolated
locality effect.

## Closed-form step composition

Let `C` be the represented compute service, `W = 8` the collective width,
`L = 24` layers, `b` the selected arm's per-collective base latency, and `p`
the picoseconds per byte of the declared link, which is 20 at 400 Gbit/s and 80
at 100 Gbit/s.

**Cross-node.** Every collective artifact composes as `b + max(0, fabric)`, and
the fabric term of the fluid profile is one propagation delay plus the
bottleneck endpoint's serialization:

```
step = C + 48 * b + sum_{i=1..48} ( 2,000,000 ps + p * endpoint_bytes_i )
```

**Intra-node.** Every collective artifact composes as `b + max(nvlink, 0)`, the
base is charged once per collective and not once per ring round, and the local
term is the whole-nanosecond endpoint serialization at the effective NVLink
rate `R`:

```
step = C + 72 * b
     + 24 * sum_{r=1..14} ceil( chunk_bytes * 1e9 / R ) ns
     + sum_{j=1..48} ceil( endpoint_bytes_j * 1e9 / R ) ns
```

with `chunk_bytes = total_new_tokens * hidden * dtype / W`, `R = 450e9` on the
`off` arm and `R = 70,027,079,100` on both other arms.

**Host term.** `C = 1000 * ceil( max(C_provider, N * g) / 1000 )` picoseconds,
per `simllm/compute/host.py:280-343` composed with the whole-nanosecond
enclosure of `simllm/backends/step_lowerer.py:247-262`. With `N * g = 440 *
809,306 = 356,094,640 ps` and a provider service that cannot exceed about
100 us at this geometry, the maximum always selects the launch floor, so every
`turing` cell reports `C = 356,095,000 ps` on every step and every `ideal` cell
reports the provider's own enclosed service.

**Constant terms, exact.**

| term | value, ps |
|---|---|
| intra-node `upper` surcharge, per step | 2,169,218,088 |
| of which at its captured operation and interconnect, 24 allreduces | 723,072,696 |
| of which transferred to all-to-allv, 48 phases | 1,446,145,392 |
| cross-node `lower` surcharge, per step | 1,446,145,392 |
| cross-node `upper` surcharge, per step | 2,375,413,872 |
| `turing-cuda-graph` compute service, every step | 356,095,000 |

## Napkin floors and ceilings, before any measured digit

Stated from first principles, one line each, so a value outside them proves a
defect somewhere regardless of internal consistency.

- **Weight-read floor, cross-node.** 554,047,488 resident bytes over the b100
  envelope's 8.0e12 B/s is 69.256 us, which no step's compute can beat, and
  98.937 us after the provider's 0.7 derate.
- **Weight-read floor, intra-node.** The TP=8 shard leaves 421,582,848 bytes,
  so 52.698 us at peak and 75.283 us after the derate.
- **Serialization floor, cross-node.** Bytes over link rate is a floor no flow
  can beat. The smallest representable step, one decode token reaching one
  remote destination, moves 2,048 bytes per artifact, so 1.966 us over 48
  artifacts at 400 Gbit/s and 7.864 us at 100 Gbit/s.
- **Propagation floor, cross-node.** 48 collectives at one 2.000 us fluid
  propagation each is 96.000 us that no step can beat.
- **Surcharge floor and ceiling.** A selected arm's surcharge is exactly its
  constant times the collective count; it cannot vary with the workload.
- **Largest representable step.** Chunked prefill is disabled and forward modes
  do not mix, so the largest step is four co-scheduled eight-token prefills,
  32 new tokens. Its critical endpoint load is `32 * 7 * 2048 = 458,752` bytes,
  which is **exactly** the maximum of the profile's own source payload envelope,
  `2 * (8 - 1) * (262,144 / 8) = 458,752`
  (`simllm/traffic/collective_latency.py:70-77, 319-343`). The matrix is
  admissible with zero margin. A single token more in one prefill batch would
  make `validate_endpoint_bytes` raise and void the affected cells. This is
  written down before the run so that, if it happens, it is a recorded
  prediction and not a discovery.

Per-cell step bands follow from the closed form at the two token-shape
extremes, one decode token to one destination and 32 prefill tokens to seven
destinations. TTFT is bounded above by five step ceilings, because a request
waits behind at most the three other prompts plus its own, and TPOT by
one and a half step ceilings, because after its first token a request receives
one token per decode step and at most three other prefills can interleave into
its eleven remaining intervals. Both are bounded below by one step floor,
because every token costs one step.

| cell | step floor, us | step ceiling, us | TTFT band, us | TPOT band, us |
|---|---|---|---|---|
| intra-off-ideal | 53.274 | 130.627 | [53.274, 653.133] | [53.274, 195.940] |
| intra-lower-ideal | 55.482 | 429.091 | [55.482, 2145.453] | [55.482, 643.636] |
| intra-upper-ideal | 2224.700 | 2598.309 | [2224.700, 12991.544] | [2224.700, 3897.463] |
| intra-off-turing | 356.671 | 411.439 | [356.671, 2057.195] | [356.671, 617.159] |
| intra-lower-turing | 358.879 | 709.903 | [358.879, 3549.515] | [358.879, 1064.854] |
| intra-upper-turing | 2528.097 | 2879.121 | [2528.097, 14395.605] | [2528.097, 4318.682] |
| cross400-off-ideal | 167.222 | 635.339 | [167.222, 3176.695] | [167.222, 953.008] |
| cross400-lower-ideal | 1613.367 | 2081.484 | [1613.367, 10407.422] | [1613.367, 3122.227] |
| cross400-upper-ideal | 2542.636 | 3010.753 | [2542.636, 15053.764] | [2542.636, 4516.129] |
| cross400-off-turing | 454.061 | 892.497 | [454.061, 4462.485] | [454.061, 1338.745] |
| cross400-lower-turing | 1900.206 | 2338.642 | [1900.206, 11693.212] | [1900.206, 3507.963] |
| cross400-upper-turing | 2829.475 | 3267.911 | [2829.475, 16339.554] | [2829.475, 4901.866] |
| cross100-off-ideal | 173.120 | 1956.545 | [173.120, 9782.724] | [173.120, 2934.817] |
| cross100-lower-ideal | 1619.266 | 3402.690 | [1619.266, 17013.451] | [1619.266, 5104.035] |
| cross100-upper-ideal | 2548.534 | 4331.959 | [2548.534, 21659.793] | [2548.534, 6497.938] |
| cross100-off-turing | 459.959 | 2213.703 | [459.959, 11068.513] | [459.959, 3320.554] |
| cross100-lower-turing | 1906.105 | 3659.848 | [1906.105, 18299.240] | [1906.105, 5489.772] |
| cross100-upper-turing | 2835.373 | 4589.117 | [2835.373, 22945.583] | [2835.373, 6883.675] |

These bands are wide because the token shape varies by a factor of 32 inside
one run. They are reported as bands and never as predictions of a value.

## Predicted ratio envelopes

Ratios are reported through `arm_ratio_envelope`
(`simllm/traffic/collective_latency.py:902-936`) over exact picosecond
quotients. An envelope that brackets one means the ordering is undetermined by
this evidence, and the study says so plainly rather than picking an arm.

**Bandwidth sensitivity, cross-node, per arm.** The 100 Gbit/s cell over the
400 Gbit/s cell. Predicted from the closed form at the observed mean token
shape: `off` about 1.84, `lower` about 1.13, `upper` about 1.09. Predicted
envelope roughly [1.08, 1.85], which does **not** bracket one: 100 Gbit/s is
slower under every arm. Predicted direction: the ratio falls as the arm's
constant grows, because a fixed cost dilutes the bandwidth-sensitive term.

**Intra-node versus cross-node at 400 Gbit/s, arm-name matched.** Predicted per
arm: `off` about 0.32, `lower` about 0.08, `upper` about 0.87. Predicted
envelope roughly [0.08, 0.87], which does **not** bracket one. The study will
state that this apparent verdict is an artifact of the pairing: the two
envelopes do not have the same arms. `intra lower` charges zero surcharge while
`cross lower` charges the full 30,128,029 ps intercept, so the arm names line
up and the physics does not.

**Intra-node versus cross-node at 400 Gbit/s, constant matched.** The honest
pairing compares equal per-collective constants. At a zero constant, intra-node
over cross-node is predicted about 0.32. At the 30,128,029 ps constant, that is
`intra upper` over `cross lower`, predicted about 1.35, above one, because the
intra-node cell pays 72 surcharges against the cross-node cell's 48. The
predicted envelope over the two matched constants therefore **does bracket
one**, and the frozen conclusion is that the ordering of the two deployments is
undetermined: it flips with the size of the per-collective constant, and no
evidence in this repository fixes that constant for an all-to-allv. TRAF-36
hardware measurement is what would tighten it.

## Fatal guards, void and never scored

A violated fatal guard voids the affected cells for the purpose of closing
anything. Fatal guards are never reported as a fraction. None of them is
declared survivable.

- **G1 provenance.** The SGLang source tree HEAD equals `8f2a3ad6...`, the
  routing trace header carries `framework=sglang`,
  `routing_source=observed-dispatch`, model revision `ffec3c35...`, 32 experts,
  top-k 8 and MoE layers 0 to 23, and the pinned model snapshot is present.
- **G2 identity.** In every cell the worker's step sink is this driver's sink
  object in the same process, the worker is `SimTpModelWorker`, the tree cache
  is `RadixCache`, and scheduler batch runs, worker step records, sink locality
  outcomes and sink network outcomes are all equal in count. No step is settled
  by the adapter's own fallback.
- **G3 shape.** No retraction, no prefill row whose context length differs from
  the whole prompt, and every scheduled row samples a token.
- **G4 conservation.** Every reduced interval's coarse and medium partitions
  conserve the elapsed time, every step's completion equals its release plus
  its makespan, every artifact's composed service equals its base plus the
  maximum of its two media, and every TTFT is strictly positive.
- **G5 locality.** Intra-node cells report zero fabric directed bytes, zero
  backend runs and positive NVLink directed bytes. Cross-node cells report zero
  NVLink directed bytes, positive fabric directed bytes, `routing_mode`
  `captured`, placement epoch 0 and a quiescent backend.
- **G6 completion.** Every request in every cell finishes for reason `length`
  with exactly 12 output tokens and 12 reduced token intervals.
- **G7 inventory.** Intra-node cells report exactly 24 tensor-parallel
  allreduces at the attention site, exactly 48 MoE all-to-alls and exactly 408
  executed artifacts per step. Cross-node cells report zero tensor-parallel
  allreduces, exactly 48 MoE all-to-alls and exactly 72 executed artifacts per
  step.
- **G8 envelope admissibility.** Every collective's critical endpoint load lies
  inside the selected profile's source payload envelope, and each cell's
  summed base latency equals its collective count times the arm's constant
  exactly, which is the check that the base is charged once per collective and
  not once per ring round.
- **G9 host agreement.** The worker and the sink select the same host model in
  every cell. Every `ideal` cell reports launch count 0 and zero exposed host
  picoseconds. Every `turing` cell reports launch count 440, launch floor
  356,094,640 ps, device key `gtx1660-ti-sm75`, compute pinned to `b100`, and
  carries `SGLANG_HOST_TRANSFER_DISCLOSURE` verbatim.
- **G10 BACK-44 negative control.** The mixed configuration
  `tp_ranks=(0, 1)` with `ep_ranks=(0, 1, 2, 3)` is refused with the recorded
  ordered-GOAL message. Executed, not recalled.
- **G11 byte conservation across arms.** Total directed bytes are identical
  across all six intra-node cells and identical across all twelve cross-node
  cells, because every token is forwarded exactly once no matter how the
  scheduler batches. Predicted exactly: 35,696,640 bytes for every cross-node
  cell, and 52,297,728 tensor-parallel plus 35,696,640 MoE, so 87,994,368
  bytes, for every intra-node cell.

## Scored relations and their entailment answers

The scored set is deliberately small. With a closed-form fluid manifold
underneath and the guards above holding, most exact rows are entailed, and an
entailed row is not evidence. Each relation below is followed by the question
"could this fail while every guard holds", answered before the run.

### Exact relations

> **E1.** In all 18 cells, an independent standard-library recomputation of
> per-request TTFT, TPOT and the medium components, taken only from the
> per-step rows and the declared arrivals, agrees exactly with
> `HtsimRequestMetricReducer`.

*Entailment: no.* The reducer raises when a total fails to conserve, so a
conservation break aborts the cell rather than failing E1. What E1 catches is
everything that conserves and is still wrong: an interval charged to the wrong
request, a first interval started at the wrong endpoint, a pending attribution
carried across the wrong step, or a realized service assigned to `nvlink_ps`
when the fabric owned it. The intra-node cells drive `nvlink_ps` and
`collective_base_ps` through a live reducer for the first time in this
repository, so the risk is real and new. *Declared narrow now, not later:*
`co_critical_ps` and `control_ps` are configuration-forced zeros under G4, G5
and G2, so E1 scores the five reachable components and not seven.

> **E2.** The `cross400-off-ideal` cell reproduces the accepted
> `sglang_end_to_end_v1` `ep8-400g` cell: 26 scheduler steps, and per-request
> TTFT of 270.05, 358.38, 483.60, 421.38 us and TPOT of 262.04, 268.70, 244.41,
> 214.95 us for `p0` to `p3`, each to the published precision of 0.01 us.

*Entailment: no.* Four things differ from the accepted run: a different trace
file carrying the same routing rows, a declared eight-host placement manifest
where the accepted run declared none, the host model built through the w14d
selection seam instead of by hand, and this study's own driver loop. Any one of
them perturbing a timestamp fails E2. It is the anchor that makes every other
cell interpretable, because it shows the composed harness did not move the
baseline.

### Behavioral relations

> **B1.** Within each of the six (topology, link, host) families, the scheduler
> step count is non-increasing across the arms in the order `off`, `lower`,
> `upper`, and is strictly smaller at `upper` than at `off` in all six.

*Entailment: no.* The arm adds a constant to every step and nothing else. The
step count is a discrete function of how that constant interacts with the one
millisecond arrival spacing and with SGLang's own batching decisions. A longer
step batches more work and should need fewer steps, but it can equally push an
arrival across a batching boundary and produce more. This is the relation that
distinguishes a live closed loop from a replay, and it is falsifiable in both
directions.

> **B2.** The largest number of requests co-scheduled in one step is
> non-decreasing across the arms in the order `off`, `lower`, `upper` in all
> six families, and equals 4 at the `upper` arm in all six.

*Entailment: no.* Nothing forces the scheduler to reach a batch of four. The
prediction is that a step longer than the arrival spacing lets all four
requests accumulate before the next batch is formed, which is a claim about
SGLang's admission and batching, not about arithmetic.

> **B3.** Within each of the nine (topology, link, collective arm) families,
> the `turing` cell takes no more scheduler steps than the `ideal` cell, and
> strictly fewer in at least one family.

*Entailment: no.* Same mechanism as B1 with a different constant. Note that the
`intra-off` family is expected to be the hardest case: 356 us per step is still
far below the one millisecond arrival spacing, so the extra host cost may not
change the count at all there.

> **B4.** For the cross-node cell at each arm, the ratio of summed TTFT at
> 100 Gbit/s over summed TTFT at 400 Gbit/s exceeds one; the three-arm envelope
> does not bracket one; and the ratio is non-increasing as the arm's constant
> grows.

*Entailment: no.* The direction is entailed for a **fixed** step sequence, but
the two link rates produce different step sequences, so the realized TTFT ratio
is taken over different batchings and can leave the per-step prediction. The
non-increasing clause is the risky half.

> **B5.** The intra-node over cross400 envelope of summed TTFT, at matched host
> arm, does not bracket one under arm-name matching, and does bracket one under
> constant matching, with the 30,128,029 ps pairing above one.

*Entailment: no.* The per-step closed form predicts 1.35 for the
constant-matched pairing with a margin of 35 percent, which the live step
sequences can erase or reverse. This is the relation that tests whether the
corrected 24-plus-48 inventory actually makes the intra-node deployment more
expensive per collective constant than the cross-node one.

> **B6.** For `p1`, `p2` and `p3`, in both topologies and at each link rate,
> enabling both the `upper` collective arm and the `turing` host arm multiplies
> TTFT by at least 2 relative to the `off`/`ideal` cell of the same topology
> and link, and the multiplier is larger in the intra-node cell than in the
> cross400 cell.

*Entailment: partly, and the entailed part is excluded.* `p0`'s TTFT is exactly
the first step's latency in both arms, so its ratio follows from the closed
form and is reported as anchored rather than scored. `p1` to `p3` carry
queueing that depends on the realized step sequence, which is what B6 scores.

### Relations deliberately removed from the scored set because they are entailed

Each is kept as a guard or as a reported diagnostic, and none of them
contributes to a scored denominator.

1. *TTFT and TPOT conserve against their attributions.* `RequestLatencyTotals`
   and `HtsimRequestMetricReducer.consume` raise instead of returning a
   disagreement (`simllm/backends/step_attribution.py:485-513, 655-667`). Kept
   as G4.
2. *Every `turing` cell reports 356,095,000 ps of compute service on every
   step.* Entailed once the resident-byte count keeps provider service under
   the launch floor. Kept as G9 and reported.
3. *Each cell's summed base equals its collective count times the arm
   constant.* Entailed by the sink charging the base only at ring phase zero
   (`simllm/backends/step_sink.py:1083-1087`). Kept as G8.
4. *Intra-node requests carry zero fabric time and cross-node requests carry
   zero NVLink time.* Forced by the declared manifests. Kept as G5.
5. *Total directed bytes are identical across arms within a topology.* Forced
   by every token being forwarded exactly once. Kept as G11 and reported with
   its exact predicted values.
6. *The intra-node inventory is 24 plus 48 rather than 96.* Forced by
   `layer_tp_allreduce_sites` under a captured supply. Kept as G7.

**Genuine-risk fraction, declared now:** 8 of the 8 scored relations are
genuine risk, with E1's component coverage narrowed to five of seven components
by declaration above rather than after the fact.

## What this study will not claim

- No calibration of anything. The upper arms are provenance-transferred, the
  host term is a three-source device hybrid, and the launch count is vLLM's.
- No absolute TTFT or TPOT prediction for any real deployment. SGL-4 owns the
  silicon comparison.
- No closure of BACK-44, TRAF-36, SGL-24 or SGL-4.
- No claim that the intra-node deployment is faster or slower than the
  cross-node one, unless the constant-matched envelope fails to bracket one.
- No mixing of evidence classes. Guard counts, exact counts and behavioral
  counts are reported separately and never summed.

## Plausibility check to be reported

The end-to-end study found that `HostInitiationModel.ideal()` made the
simulated decode rate optimistic by roughly one order of magnitude against a
real serve of a 400M-active-parameter MoE, which decodes a single request at
roughly `10^2` tokens per second. The `turing` arm adds 356 us of fixed host
cost per step, which alone caps the decode rate near 2,800 tokens per second,
and the `upper` collective arms push the step past 2 ms, which caps it near 400
tokens per second. The results will state where each arm's implied
tokens-per-second lands against that `10^2` anchor, and will say plainly that
landing near it is not evidence of accuracy, because the constants that got it
there were transferred from a consumer Turing GPU and an NVLink all-reduce
capture and neither was measured on this chain.
