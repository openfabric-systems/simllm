# Matched-seam aggregate-arm expectations

This expectations-only freeze defines the DEPLOY-22 aggregate arm before its
implementation or first SimLLM evaluation. It adds no outcome to the protected
matched-seam publications and changes no prior score, record, result, runner,
plotter, or figure. The external source audit and the 25-row external aggregate
table are frozen givens. Nobody knows the SimLLM aggregate quotients at this
commit.

## Protected prior evidence

The aggregate runner verifies these bytes before and after both fresh-process
evaluations. A mismatch is fatal and voids the aggregate run.

| Path | SHA-256 |
|---|---|
| `DEPLOY12_RESULTS.md` | `502c835fd33fd5bd0abee11ae2548eaf099e39653671d9a1a3c993a76530c6c3` |
| `RESULTS.md` | `fa1170277fa8f3b9f1a14df353add3dbd4e8e490aeb4847748dd2120d4434e62` |
| `deploy12_record.json` | `c2b4fa9b8e8c2401d01a36731e9e1989ef27918b5bb170813b436c0e61ab630f` |
| `deploy12_results.csv` | `4057d5f321ae60bd7e34bd8b3e9ca663694f189788632c34806b4bfe1b7bc4a8` |
| `expectations.md` | `fc5af307fee560fc7050011543e18e1cf77030d0aa6a13e6c5a014cb159a5726` |
| `expectations_deploy12.md` | `ed784f7514fe766c509b02ed591391370129b84c63cc51552e278f5fcee44812` |
| `expectations_v2.md` | `fe403500575d674a25c8b7c6c59eb41aec65fce6cc29024609fa995b29585f35` |
| `external_adjustments.json` | `c6778a81cdc6078ce74f06733e4bce9d99a92b4ab3eccba4a83d14e7d063a09e` |
| `figure_addendum.md` | `cc4dcb8c82bbcd5e542457b56d91ddf172af2cbe05e6bac5c865535dcc307762` |
| `plot_publication.py` | `a98514cb985a9980a679357285a11dbe52418e774a55d69a6c9f30ba9ddda53d` |
| `plot_study.py` | `d4fe430f1fede23bcbcbb21834d98a51d3563c4b4e4c21dc887c7b8c837a7e4f` |
| `record.json` | `bddd7cb040a3c0f0ec8afd7ea836d873fa22cad2131f98ff36e38da5441b2d50` |
| `results.csv` | `4113ab2413084b7da957de60002abc4a4f8530bbb89837a5a5f73b9852f4448d` |
| `run_deploy12_arm.py` | `683bd65e48539bfaf657b3b1f1aaa0dbcae809bbc39de815be50544fcba03a41` |
| `run_study.py` | `242b5f1ae46ac18ac2cb474ad6fa24acc4dba21c4b8ff1d6683137163fec3182` |
| `study_config.json` | `64c8e16de53e194e98f5ca7c9b27d533d4c7f7ca32311841a62e3c6cece21f17` |
| `figures/matched-seam-frontier-publication.pdf` | `511a0fb869d3397a87664d28c6b0c1d5adc17738dd84543973f66c7fcfd764cb` |
| `figures/matched-seam-frontier-publication.png` | `d79b5099cbbfeed9e4272a64d7007512ed1889a08fc3438c9f2eef41a28986d1` |
| `figures/matched-seam-frontier.pdf` | `4ecc3bf2822f916bfd53107b55d1344406efea01fd0b1ad7a417019391712dbb` |
| `figures/matched-seam-frontier.png` | `852378a01d3c9e0aeab74423259afe86b456dca0b193e27c23e48256322069c4` |

The scored external table is
`examples/frontier_comparison_v1/external/agg_pareto.csv`, SHA-256
`89b062634eacb75acacf7a6935e00d42992d112359a302eb8a998992f52ab1f3`.
It contains 25 aggregate operating points and remains external display and
comparison evidence, never a SimLLM result.

## Frozen external aggregate semantics

The pinned packages are aiconfigurator 0.11.0 and aiconfigurator-core 0.11.0.
The operation slice remains H200 SXM, TensorRT-LLM 1.3.0rc10,
Qwen3-32B-FP8, SILICON mode, shared layer off, Python estimator surface, slice
SHA-256
`85e72f990f00ea457de522d0b773e678f5e067740689912df5646f6296273284`.

The complete aggregate composition source is
`aiconfigurator_core/sdk/backends/base_backend.py`, SHA-256
`0ad5647459114a726dcce5783bce52a6069bcefcb7b175202a4f6161cbc57f75`.
Lines 994 to 1128 construct one mixed step from the combined non-attention
pass, context attention, and generation attention. Lines 1130 to 1193
construct a generation-only step. Lines 1246 to 1505 construct the in-flight
schedule, TTFT, TPOT, throughput, axes, and the counters carried by the
external CSV.

For batch `b`, input length `I`, output length `O`, prefix `P`, and per-step
context budget `C`, the frozen schedule is:

```text
mix_steps = ceil(I * b / C)
mix_gen_requests = max(1, b - ceil(C / I))
genonly_steps = O - mix_steps
genonly_requests = b
ctx_requests = ceil(C / I)
gen_requests = b - ctx_requests
```

All frozen rows have `mix_steps < O`. TP4 rows use `C=4000`; TP8 rows use
`C=4000` or `C=8000` exactly as declared by each table row. The table fields
`balance_score`, `num_ctx_reqs`, `num_gen_reqs`, `num_tokens`, `ctx_tokens`,
and `gen_tokens` must reproduce these identities exactly.

TensorRT-LLM's TPOT correction is in
`aiconfigurator_core/sdk/backends/trtllm_backend.py:68-72`, SHA-256
`6a2d13b74a7296ad94d0f2c16e8d579f28a20ab9f95b48b70180a29b004e89e0`:

```text
tpot_mix_steps = max(1, mix_steps - 3)
tpot = (mix_step_ms * tpot_mix_steps
        + genonly_step_ms * genonly_steps)
       / (tpot_mix_steps + genonly_steps)
```

The three-step reduction is an empirical pipeline-drain correction. It is not
kernel service and must never be hidden inside an imported operation duration.

Output throughput and the two plotted axes are frozen as:

```text
total_schedule_ms = mix_steps * mix_step_ms
                    + genonly_steps * genonly_step_ms
tokens_per_second = 1000 / total_schedule_ms * b * (O - 1)
tokens_per_second_per_gpu = tokens_per_second / tensor_parallel
tokens_per_second_per_user = 1000 / tpot
```

## External adjustments and aggregate reachability

The protected eight-row table remains byte-identical. The additive
`external_adjustments_agg.json` records aggregate reachability and the two
aggregate-only empirical mechanisms. Of the original eight factors, exactly
three reach this arm: the H200 memory-bandwidth scale 0.8, the 3 microsecond
memory-operation constant, and the 1.1 context-attention extra-latency
correction. The disaggregated prefill and decode latency corrections, both
rate-matching factors, and the 1.8 autoscale correction do not reach aggregate
composition.

The aggregate path adds exactly two declared empirical mechanisms: the
concurrency-dependent TTFT queueing heuristic at
`base_backend.py:127-136,1341-1349`, and the TensorRT-LLM three-step TPOT
reduction above. The identity hooks for mixed-step efficiency, dispatch
overhead, and throughput caps have values 1, 0, and identity, respectively;
they are recorded as inactive structural hooks rather than applied factors.

## Family AR: aggregate TPOT reproduction (scored)

For each of the 25 external aggregate rows, SimLLM composes the mixed and
generation-only services from the imported measured database, applies the
declared three-step count correction, and reports:

```text
AR quotient = SimLLM composed aggregate TPOT / published aggregate TPOT
```

Every quotient must lie in **[0.98, 1.02]**. The expected result is near 1
because the operation database, mixed-step construction, schedule, and one
applied count correction are shared. A miss is published with the residual
split among mixed-step service, generation-only service, count arithmetic,
and three-decimal table reconciliation. The band is not widened. Family AR is
25 rows and is never summed with any other evidence class.

## Aggregate TTFT decomposition (unscored, mandatory)

External aggregate TTFT is an operating-point value. It includes queueing and
varies with concurrency. It is never compared with an isolated prefill pass.
For every row publish:

```text
pure_prefill_step_ms = mixed-step service before any mixed-step efficiency
prefill_passes_per_request = ceil(I / C)
base_prefill_ms = pure_prefill_step_ms * prefill_passes_per_request
queue_factor = min(2 + (mix_steps - 3) / 20, 4)
queueing_component_ms = base_prefill_ms * (queue_factor - 1)
composed_ttft_ms = base_prefill_ms + queueing_component_ms
publication_residual_ms = published_ttft_ms - composed_ttft_ms
```

The pure prefill term, the queueing component, the total, and the rounding
residual are separate fields. A residual outside plus or minus 0.0005 ms is a
fatal source-semantics mismatch. This register is unscored and never becomes a
TPOT pass.

## Aggregate adjustment sensitivity (unscored, mandatory)

For all ten rows in the combined adjustment declaration, publish a remove-one
table with the aggregate TPOT quotient range, aggregate TTFT quotient range,
and reachability for both metrics. Removing a multiplicative factor replaces
it with 1.0; removing the 3 microsecond constant replaces it with zero;
removing the three-step reduction uses all mixed steps; removing the TTFT
queueing heuristic uses factor 1.0. An unreachable factor must reproduce the
complete baseline aggregate projection byte for byte. Sensitivity rows are
disclosure evidence and are not added to Family AR.

## Network arms and co-location identity

Each aggregate candidate has one `combined` pool. Prefill and decode execute
on the same TP worker, so the external-fabric P/D handoff ledger contains zero
bytes and zero flows. The unpriced-network and packet-network aggregate arms
must therefore have identical scheduling, TPOT, TTFT, throughput, axes, and
completion order. The packet arm starts no packet process. The imported
operation timings already include the measured TP collectives inside each
forward pass; re-pricing those collectives as a P/D fabric handoff would double
charge them and is fatal.

This identity is specific to the co-located traffic definition. It does not
claim that the GPU's internal collective network costs zero.

## Physical sanity before precision

These bounds are frozen before a SimLLM aggregate result is read. Passing them
is necessary and is not proof of composition parity.

- Weight-stream floor: one FP8 pass over 32 billion parameter bytes costs at
  least 1.666667 ms at TP4 and 0.833333 ms at TP8 on 4.8 TB/s HBM. Every
  generation-only step and every TPOT must stay above its width's floor.
- Mixed-step compute floor: a mixed step with `N = ctx_tokens + gen_requests`
  tokens cannot beat `2 * 32e9 * N / tensor_parallel / 1.978e15` seconds.
  Every mixed-step service must stay above this row-specific floor.
- Causal ceiling: mixed-step service, generation-only service, TTFT, and TPOT
  must each be positive and no larger than the row's published total request
  latency. This deliberately loose ceiling catches unit and repeat-count
  failures without asserting silicon accuracy.
- Scaling cross-check: TP4 and TP8 use independent HBM and compute floors, and
  the reported throughput must satisfy both axis identities and the exact
  request-rate identity `tokens_per_second / (O - 1)`.
- System plausibility: per-GPU output throughput must not exceed
  `batch * 4.8e12 / 32e9` tokens/s/GPU, the optimistic peak-HBM bound when one
  weight read serves the whole batch.

## Fatal guards

A failed guard makes the aggregate run void. Fatal guards are never converted
to a score or included in a behavioral denominator.

- FG-AGG-1a, no SimLLM-authored timing: every positive forward-pass duration
  comes from the frozen imported database through the external pass model. No
  SimLLM roofline, declared efficiency, fitted constant, fitted curve, or
  authored duration reaches TPOT, TTFT, throughput, or either axis.
- FG-AGG-1b, declared adjustments: the applied aggregate factors equal the five
  reachable factors in the additive table exactly, every source hash and line
  verifies in the pinned installation, and no undeclared factor is applied.
- FG-AGG-1c, sensitivity completeness: every one of the ten declared factors
  has both TPOT and TTFT remove-one disclosure, with unreachable projections
  byte-identical to baseline.
- FG-AGG-2, protected evidence: all prior publication bytes, the external
  aggregate table, and the parity record are byte-identical before and after
  both evaluations.
- FG-AGG-3, external identity: package versions, slice, system, backend,
  database version, quantization, source hashes, and source line ranges match
  this freeze.
- FG-AGG-4, strategy and traffic identity: every record and figure series names
  aggregate or disaggregated strategy and its traffic definition. Both
  aggregate network arms have zero P/D handoff bytes; the packet arm starts no
  native process and equals the unpriced arm byte for byte.
- FG-AGG-5, no naive TTFT match: no score or claim equates published aggregate
  TTFT with isolated prefill service. Every row carries the mandatory
  operating-point decomposition and stays within the publication residual
  bound.
- FG-AGG-6, determinism: two complete scored-evaluation JSON records produced
  in fresh processes are byte-identical. Only `elapsed_seconds` and `W-1` are
  excluded, by those exact names. No service, schedule, row, quotient,
  sensitivity, source, or traffic field is excluded.
- FG-AGG-7, schedule and axis identities: all external scheduling counters,
  row identities, axis formulas, and request-rate identities reproduce their
  declared arithmetic without row omission, duplication, or reordering.
- FG-AGG-8, physical bounds: every row satisfies all predeclared HBM, compute,
  causal, scaling, and throughput bounds.
- FG-AGG-9, chronology: this freeze commit precedes product and study
  implementation and the first SimLLM aggregate evaluation.

## Family W and evidence separation

The complete coordinator, including two fresh-process evaluations, source
audit, all sensitivities, both renderings, and record generation, must finish
within 600 seconds. W is one scored row in its own register. Fatal guards,
Family AR, TTFT decomposition, sensitivity, network identities, and W remain
separate evidence classes.

## Figure contract

Create two new aggregate-qualified PDF/PNG pairs. The existing figure pairs and
panels remain byte-identical. Both new figures retain the established log-log
frontier, network-mechanism zoom, and matched-decode panel where present, then
add the SimLLM aggregate unpriced and packet series with distinct markers at
their exact coordinates. Because the two aggregate arms are an identity, both
markers must remain legible without moving either series off its data.

Every legend entry names its strategy and traffic definition: aggregate or
disaggregated, co-located or split P/D pools, and unpriced or packet P/D
handoff. The caption states that every SimLLM compute duration comes from the
same imported measured database, that aggregate differences are composition
rather than kernel timing, and that co-location makes the packet P/D handoff
zero bytes. No aggregate agreement or disagreement is attributed to kernel
timing.

## Closure

DEPLOY-22 closes only if the run is nonvoid, all 25 Family AR rows and W pass,
the TTFT and sensitivity tables are complete, the two aggregate network arms
meet their exact identity, and both new figure pairs pass visual inspection.
Closure establishes composition parity or publishes a named composition
refutation at the matched measured seam. It does not validate either planner
against hardware, calibrate H200 silicon, repair the protected disaggregated
F-2-09 miss, or change any prior task disposition. Any remaining work receives
a new stable task ID rather than leaving DEPLOY-22 partially closed.
