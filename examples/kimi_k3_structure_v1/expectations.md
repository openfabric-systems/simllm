# Kimi K3 structural extraction expectations

This prospective study freezes the text-only structure of the exact Kimi K3
checkpoint through the pinned vLLM and SGLang configuration seams. COMP-54
owns the complete logical inventory. COMP-59 retains physical implementation
and reduced-depth capture, and CORE-54 retains deployment frontier evidence.
No model weights, GPU execution or hardware reservation belong to this study.

## Sources and authority

The [official configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)
is an input with SHA256
`9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`.
Its 93 text layers contain 69 Kimi Delta Attention (KDA) layers and 24
multi-head latent attention (MLA) layers. The first layer has a dense feed
forward network; the other 92 have latent routed experts and a separate
full-width shared-expert branch. The final layer is MLA even though its index
is outside the preceding four-layer cadence.

The [vLLM implementation](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/models/kimi_k3/nvidia/model.py)
and [SGLang implementation](https://github.com/sgl-project/sglang/blob/bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3/python/sglang/srt/models/kimi_k3.py)
provide independent native configuration bindings. Neither a common dictionary
manufactured for both drivers nor a generic uniform transformer projection is
evidence that both native seams work. Each driver loads its native configuration,
verifies its pinned architecture binding and normalizes the full typed geometry.
The two normalized specifications must agree exactly. The checkpoint identity
carries this heterogeneous geometry rather than inventing a uniform head size.

One typed specification owns the original layer indices, tensor shapes,
recurrent and history-cache shapes, latent expert widths and residual bank.
One logical operation graph owns the per-case inventory. Every inventory visit
joins to that graph, with no independent reconstruction of family totals.
Neither projection changes the original layer numbering.

## Logical work and physical binding

A logical operation is a mathematical region, not an observed GPU launch.
KDA implementations can fuse convolution, recurrent update and normalization;
MLA implementations can absorb matrices into a different attention algorithm.
The inventory preserves complete logical regions and their declared algorithm
variant, while observed launches and code objects remain absent by design.

Matrix multiply-accumulate work counts two floating-point operations per pair.
Unknown scalar algorithm work remains explicitly unbound. Stored checkpoint
bytes, declared state capacity and bytes served from high-bandwidth memory (HBM)
are distinct quantities. No checkpoint encoding or logical state touch silently
becomes a per-step HBM byte count. Unbound memory demand is null, never zero.

The shared compute-work contract gains an explicit logical-operator scope.
Execution, GOAL emission and bottleneck reporting reject that scope or unknown
demand before publishing a metric or advancing mutable state, including when a
duration or concurrent service is supplied. A separately identified synthetic
binding may assign every region a declared service and zero memory arbitration
solely for the diagnostic request loop. Its graph identity and provenance differ
from the unbound inventory. It is never a Kimi K3 speed prediction.

The initial supported matrix contains pure cold prefill and pure decode. It
declares expanded new-token MLA for cold prefill and absorbed MLA for decode.
Prefix, mixed-batch, speculative, vision, projector, multi-token prediction and
undeclared implementation-dispatch inputs reject explicitly. This boundary
does not infer every backend's dispatch from the word prefill or decode.

The pinned logical partition is 17 regions per KDA layer, 13 per expanded
MLA layer, 14 per absorbed MLA layer, four for the dense feed forward network,
and 16 per latent mixture of experts. Three common residual regions surround
each layer's attention and feed forward work. Four outer regions handle
embedding, final residual mixture, final normalization and the language head.
That gives exactly 3,244 prefill and 3,268 decode visits, and dependency
depths of 1,812 and 1,860 with unit region service and independent grants.
The ordered region vocabulary is in [expectations.json](expectations.json).

All independent branches use independent semantic queues. The routed branch
starts from a 3,584-wide latent projection, while shared experts consume the
original 7,168-wide normalized input. Dispatch waits for both token selection
and the latent input. Final addition waits for both shared and routed outputs.
The prefill attention product consumes expanded new-token tensors directly;
it does not wait for compressed-cache storage. The step completion frontier
retains that separate cache commit. Decode attention includes its current
token and therefore waits for that commit before reading compressed state.

The bounded KDA forget gate is per key channel. It multiplies a sigmoid by
the declared negative bound, with the learned exponential scale inside that
sigmoid, and then exponentially decays state columns. A clamp or ordinary
softplus is a different operation. The inventory records the complete logical
update and query read as one region without claiming its fused scalar cost.

Final normalization has an explicit framework token axis: vLLM normalizes
the sampled rows passed to its logits entry point; SGLang normalizes all rows
at its forward tail. Both structures preserve the same checkpoint geometry.
Their prefill normalization invocation shapes intentionally differ, and the
comparison must explain that difference rather than force identical records.

## Exact geometry and capacity relations

Let hidden width H=7168, latent expert width E=3584, expert intermediate
width I=3072, routed experts X=896, selected experts K=16, heads A=96 and
KDA head width D=128. The dense intermediate width is 33792. There are two
shared experts with intermediate width 3072, and an untied vocabulary of 163840.

One routed expert has 3 E I = 33,030,144 weight values. With packed four-bit
values and one byte of scale per group of 32, its encoding floor is
17,547,264 bytes. All routed experts have 2,722,740,830,208 values and an
encoding floor of 1,446,456,066,048 bytes. The shared branches have
3 H (2 I) values per routed layer and 24,310,185,984 BF16 bytes across all
92 routed layers. These are storage statements and do not count HBM traffic.

The two latent input/output matrices have 2 H E = 51,380,224 values per
routed layer. The router has H X = 6,422,528 values. The dense first layer
has 3 H 33792 = 726,663,168 values. A token visits exactly 16 routed experts;
resident experts, selected expert visits and the union of active weight tensors
are separate counts.

Each sequence retains 69 A D D FP32 recurrent values, or 434,110,464 bytes.
Its three KDA convolution histories retain 69 (3 A D) (4-1) BF16 values,
or 15,261,696 additional bytes. Checkpoint convolution weights and padded
gate parameters do not determine those runtime cache layouts.

Each historical token in all 24 MLA layers retains (512+64) BF16 values,
or 27,648 bytes. Skipping rotary position transformation does not delete
the 64-value component. For B equally shaped sequences with C cached tokens,
the MLA history capacity is exactly B C 27,648 bytes. KDA persistent capacity
depends on B and is invariant to C.

For n new tokens after c prior tokens, inclusive causal attention has
n c + n(n+1)/2 query-key pairs per sequence. A cold singleton has one pair.
No floor(n squared/2) approximation is accepted on this new path.

Residual snapshots occur before attention at original zero-based layers
0, 12, 24, 36, 48, 60, 72 and 84. The maximum snapshot bank holds
8 H BF16 values, or 114,688 bytes per current token, separately from the
current working stream. Snapshot timing, prefix reset and both per-layer
aggregations remain causal graph dependencies.

## Parameter sweep and diagnostic loop

The structural grid crosses batch sizes 1 and 3 with cold prefill lengths
1, 4 and 16, and separately with decode context lengths 1, 17 and 257.
Decode context includes the newly computed token. Each framework extracts the
same complete grid twice. A separate routed-load projection preserves all
896 resident experts and accounts for exactly 16 visits per new token.

Expected relations precede the run:

- Multiplying batch size by three triples matrix arithmetic, total selected
  expert visits and per-sequence persistent capacity, at fixed sequence shape.
- Cold prefill pair work follows n(n+1)/2 exactly. Linear projections scale
  with n. Their different shapes must remain separately observable.
- Decode attention pair work grows linearly with total context. Persistent
  KDA capacity stays fixed at fixed batch, while MLA history capacity grows
  by exactly 27,648 bytes per added cached token and sequence.
- Changing the declared synthetic service from 1000 to 2000 ps per token
  doubles the graph and request completion time exactly. With one serialized
  diagnostic compute resource, completion equals the sum of every bound
  region's declared service. Prompt lengths 1 and 4 cross both service values.
  Published time to first token and time per output token must agree with
  the original graph's completions over a prefill and two decode steps.
- A separate join control provides 64 diagnostic compute slots, 1,000 ps for
  ordinary regions and 10,000 ps for each shared-expert region. Its shared
  branch then controls each routed layer's join. Completion is exactly
  3,652,000 ps for prefill and 3,700,000 ps for decode. These come from
  892 or 940 ordinary-depth units plus 276 shared-depth units. Missing the
  shared dependency or serializing independent branches fails this control.

Before reading each diagnostic completion, its floor is the longest causal
path's declared service and its ceiling is the sum of all region services.
The selected serialized resource must achieve the ceiling. Three independent
checks cover tensor arithmetic, bytes required by retained state, and original
graph to request timing. No synthetic timing is compared to a hardware anchor.

## Fatal guards, compatibility and closure

Source/checkpoint identity disagreement, missing or duplicated original layers,
missing logical families, graph/inventory disagreement, invalid causal order,
invented physical identity, unsafe unknown-demand execution and loss of an
accepted bypass artifact each void the run. A void run retains its evidence
and has no behavioral score. Guards and disabled-path identities are unscored;
they are never added to behavioral relations or exact arithmetic checks.

Every previously tracked model inventory and authored legacy suite remains
byte-identical. Regenerate the accepted two-framework extraction controls with
the same native environments where available, and compare canonical records
exactly. Ordinary imports remain free of framework imports. Malformed new
inputs reject without changing the old path.

COMP-54 closes only when both native configuration seams, the complete graph
and inventory, negative controls, compatibility and the diagnostic request loop
pass the frozen bar. A missing native seam or partial inventory leaves it open.
COMP-59, COMP-64 and CORE-54 retain physical launches, real implementation costs
and deployment qualification. No measured model-throughput claim follows from
this structural study.
