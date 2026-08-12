# Serving-framework CPU oracle results

Expectations were frozen in commit `2bba046` before the first source build,
framework execution, implementation, or comparison run. The frozen study is
in [expectations.md](expectations.md). CPU latency is deliberately absent from
the outcome. This oracle is about per-request decisions and framework-owned
state, not GPU timing.

## Result

vLLM 0.26.0 built and ran successfully as a CPU distribution on this host.
It is the selected oracle. SGLang and other runtimes were therefore excluded
from the behavioral denominator.

The behavioral result is **7/8 passed**. All eight scored relations were at
genuine risk, so the genuine-risk pass fraction is also **7/8**. The three
requests matched exactly on output tokens, output length and normalized stop
reason. Every aligned routing tuple differed, but all differences were
`order-only`: the selected expert sets were identical and changed all-to-all
bytes were zero. Prefix reuse passed. The pressure relation failed because
the four registered pressure requests were admitted or serialized without a
preemption in either capacity cell.

## Isolated vLLM build

The build used a new Python 3.12 environment below the configured external
run root. It never installed into or otherwise modified the repository's
existing vLLM environment. The source archive SHA-256 was
`c1ded7e5af7e7fd27c6abcce1ecfaf795bc927a0f747f10f1e5a7a13f2c4a6a9`.
The observed and authored-against source identity was
`568afb3a13806beb53bb2e6bd518269357b237c0`.

The retained build ladder was:

| Attempt | Exit | Decisive result | Retained log |
|---|---:|---|---|
| Unpacked archive, system toolchain | 1 | `LookupError: setuptools-scm was unable to detect version for` | `vllm-build-system-gcc.log` |
| Versioned source, system GCC 12.2.1 | 1 | `X86 backend requires gcc/g++ >= 12.3` | `vllm-build-system-gcc-global-version.log` |
| Task-local GCC 13.4, system CMake 3.20.4 | 1 | `CMake 3.26 or higher is required. You are running version 3.20.4` | `vllm-build-gcc13.log` |
| Task-local GCC 13.4 and CMake 4.4.2 | 1 | `fatal error: numa.h: No such file or directory` | `vllm-build-gcc13-cmake442.log` |
| Task-local GCC 13.4, CMake 4.4.2 and numactl headers | 0 | Built and installed `vllm-0.26.0+cpu` | `vllm-build-gcc13-numa.log` |

The successful wheel was 101,905,798 bytes with SHA-256
`cc1c19717bcc40c2be0eb6c5ab4a9498e4d376f1f0ac4574f196b94b5f20445b`.
The precise additional host requirements beyond the audited installation were
GCC 12.3 or newer, CMake 3.26 or newer, and libnuma development headers. The
host is AMD Zen 2 with AVX2 and no AVX-512. A direct generic extension import
outside vLLM's platform selection terminated with `Illegal instruction`, but
the supported `CpuPlatform` loader selected the AVX2 extension and completed
both qualification and study runs.

The qualified runtime reported:

| Check | Observation |
|---|---|
| Distribution | `0.26.0+cpu` |
| Torch | `2.11.0+cpu` |
| Platform and device | `CpuPlatform`, `cpu` |
| Required operator | `torch.ops._C.init_cpu_memory_env` present |
| Worker and runner | `CPUWorker`, `CPUModelRunner` |
| Model | `GraniteMoeForCausalLM`, all parameters on CPU |
| CUDA allocation, 64-token cell | 0 bytes before, 0 bytes after |
| CUDA allocation, 256-token cell | 0 bytes before, 0 bytes after |

The import and one-request runtime transcripts are retained as
`vllm-qualified-import-probe.log` and `vllm-cpu-qualification.log`. The
machine-readable build ladder is `vllm-build-observation.json`. It records the
source observed by the run separately from the source identity against which
the evidence was authored. No live submodule pin equality is asserted.

## Request outcomes

The framework received the exact prompt token IDs produced for the
Transformers capture. Both used greedy sampling and seed 173.

| Request | Transformers output IDs | vLLM output IDs | Length | Stop reason | Result |
|---|---|---|---:|---|---|
| `eos-brief` | `[2950, 32, 0]` | `[2950, 32, 0]` | 3 | `eos` | PASS |
| `length-cap` | `[38]` | `[38]` | 1 | `length-cap` | PASS |
| `stop-string` | `[2123, 1679, 21062, 81, 15707]` | `[2123, 1679, 21062, 81, 15707]` | 5 | `stop-string` | PASS |

Each complete token sequence and normalized stop reason is one scored
instance. Length is diagnostic because token-sequence equality already pins
it. Text equality is not used as a substitute for token equality.

## Observed dispatch

The stock vLLM CPU `enable_return_routed_experts` allocation was zero-filled:
its built-in callback is attached to the modular MoE path, while Granite on
CPU used monolithic `CPUFusedMOE`. The oracle plugin therefore observed the
expert-ID tensor returned by `cpu_fused_moe.select_experts` immediately before
the unchanged expert kernel consumed it. It did not recalculate top-k and did
not replace dispatch.

| Request | Aligned token-layer rows | Exact tuples | Taxonomy | Changed all-to-all bytes | Result |
|---|---:|---:|---|---:|---|
| `eos-brief` | 408 | 0 | 408 `order-only` | 0 | PASS |
| `length-cap` | 528 | 0 | 528 `order-only` | 0 | PASS |
| `stop-string` | 576 | 0 | 576 `order-only` | 0 | PASS |
| Total | 1,512 | 0 | 1,512 `order-only` | 0 | diagnostic |

Classification coverage was 1,512 of 1,512 differing rows. No row was
`expert-id-only`, `byte-changing`, `output-cascade`, or `unaligned`. The tuple
ordering difference is real, but the expert sets are identical. Under the
frozen two-rank placement, dispatch and combine destinations therefore remain
identical. All three changed-byte relations passed at zero bytes, below the
4,096-byte materiality quantum.

This is not entailed by classification coverage. The harness first computed
the raw per-token, per-layer, per-destination byte vectors and their absolute
differences. It classified raw tuple differences separately. The later fatal
coverage guard only proves that no difference escaped the taxonomy.

## Framework KV behavior

Both cells used vLLM's reported 16-token page size and exact configured token
capacity. Prefix matching, allocation, release, cached-block eviction and
preemption were observed at the stock manager, pool and scheduler calls that
own those decisions.

| Relation | 64-token cell | 256-token cell | Result |
|---|---:|---:|---|
| Cold prefix hit | 0 tokens | 0 tokens | diagnostic |
| Warm prefix hit | 16 tokens | 16 tokens | PASS at 256 |
| Evictions after pressure-group submission | 8 | 1 | diagnostic |
| Pressure-group preemptions | 0 | 0 | FAIL |

The warm prompt contained 20 tokens. The frozen minimum was the largest whole
page not exceeding 19 tokens, namely 16, so the prefix relation passed exactly
at its bound. The pressure relation required at least one low-capacity
preemption. It failed even though low capacity produced more evictions. The
observed mechanism was admission or serialization, not scheduler recompute,
for that four-request group. PLAY-9 owns a new, pre-registered workload that
must distinguish those mechanisms without rewriting this negative result.

No live preemption occurred anywhere in the two final cells. The observation
hook wraps the scheduler's actual recompute path and has direct unit coverage,
but this final study does not promote that component check into live evidence.
PLAY-9 requires a workload that reaches the path.

During an earlier non-final execution, an initial conservation check
incorrectly summed every prefix lookup for a preempted request and compared it
with `RequestOutput.num_cached_tokens`. vLLM reports that field for the first
prefill only; a later post-preemption prefix lookup remains visible only in
the event ledger. The correction uses the first prefill for that response
field reconciliation and retains all later events. This is a post-specified,
fatal-unscored correction. It changed no scored outcome, routing, prefix,
eviction, or pressure result.

## Trace contract and guards

`simllm-preplay-trace-v2` is a separate strict JSONL schema. It records
framework provenance, complete request outcomes, post-selection routing and a
globally sequenced KV event ledger. The two canonical traces contained nine
requests each, with 91 KV events at capacity 64 and 33 at capacity 256. Both
strict read and write round trips were byte-identical.

All fatal unscored guards passed:

- vLLM CPU build, platform, operator, stock worker, stock runner, model device,
  dispatch-source and zero CUDA allocation-delta qualification;
- source hashes and observed versus authored-against provenance;
- complete routing classification and per-request identity coverage;
- request cached-token and preemption-counter reconciliation;
- strict v2 validation, footer conservation and canonical round trips;
- strict v1 reading and the frozen 1,368-byte v1 writer fixture;
- missing-dependency and unsupported-worker rejection before the canonical v2
  writer, covered by unit tests.

These guards are by-construction or structural evidence and are not added to
the behavioral pass denominator.

## Evidence accounting

| Evidence class | Executed | Passed | Genuine-risk executed | Genuine-risk passed |
|---|---:|---:|---:|---:|
| Request outcome comparisons | 3 | 3 | 3 | 3 |
| Changed-byte routing relations | 3 | 3 | 3 | 3 |
| Prefix relation | 1 | 1 | 1 | 1 |
| Pressure relation | 1 | 0 | 1 | 0 |
| Behavioral total | 8 | 7 | 8 | 7 |
| Fatal unscored guards | separate | all | not scored | not scored |
| Native executables | none | none | separate | separate |

The three output relations were not pinned by schema or qualification. The
three routing-byte relations were not pinned by exact output equality because
identical tokens can select different experts under different dispatch
kernels. The prefix and pressure relations depended on live scheduler and KV
state. Thus all eight scored instances were genuine-risk. Raw observations
were evaluated before the fatal source, schema, footer and reconciliation
checks. No guard was promoted into a behavioral pass.

The wheel build and one-request framework qualification are component
evidence, not native behavioral executables and not part of the 7/8 headline.
After intermediate vLLM build failures, a one-request SGLang CPU fallback
probe reached stock `TpModelWorker`, stock `ModelRunner`, Granite dispatch,
Radix prefix matching and token-slot allocation. Once vLLM qualified, the
fallback was not run in the scored cells. Its CPU-only compatibility hook
uses an unpinned CPU tensor for SGLang's dispatch-capture buffer because the
upstream pinned allocation initializes CUDA even in the CPU engine. It does
not change the selected IDs. No other CPU runtime survey was needed.

## Closure mapping

PLAY-5 registered this remaining acceptance clause:

> Run the frozen requests and seed through the PLAY-1 runner and a vLLM
> 0.26.0 CPU build, then compare lengths, stop reasons and routing with every
> divergence classified and none silently accepted. The runtime must select
> `CpuPlatform`, export `torch.ops._C.init_cpu_memory_env`, construct the stock
> `CPUWorker` and `CPUModelRunner`, load the pinned Granite model entirely on
> CPU, and show no CUDA allocation increase. The routed replay half is
> complete with 13/13 scored live relations and must retain those accepted
> results when this task closes.

Evidence mapping:

- The three frozen requests and seed 173 ran through both live runners. The
  outcome and dispatch tables above cover every compared request and all 1,512
  routing rows.
- The build qualification table demonstrates `CpuPlatform`, the required
  operator, the stock classes, CPU-only parameters and a 0-to-0 CUDA allocator
  measurement in both cells.
- The taxonomy classified every non-exact routing row. No difference was
  silently accepted.
- The prior routed replay implementation and its 13/13 result were untouched;
  the full repository test gate retained that path.

PLAY-6 registered this acceptance clause:

> Add an optional framework CPU backend runner that captures the same artifact
> through vLLM or SGLang on CPU, exercising the deployment framework's
> sampler. The Transformers runner remains the supported baseline and must
> stay byte-identical when no framework runner is selected. Missing framework
> dependencies and unsupported CPU backends must be rejected before a trace
> writer opens.

Evidence mapping:

- `VllmCpuRunner` and `SglangCpuRunner` are explicit opt-in classes. The
  selected vLLM runner used the framework sampler, actual dispatch and actual
  KV owners and wrote the new strict v2 artifact.
- v1 remains a distinct contract. Its reader passed, and the frozen writer
  fixture remained byte-identical at 1,368 bytes.
- Unit tests remove each optional framework import and assert that no v2 trace
  or raw response exists. Runtime qualification rejects the wrong worker,
  runner, model, parameter device, dispatch source, KV seam, page size or
  capacity before calling the canonical writer.

PLAY-5 and PLAY-6 are therefore removed from the open registry and recorded in
the closed-task ledger. PLAY-8 owns live v2 join and replay consumption.
PLAY-9 owns the failed pressure-family follow-up. SGL-16 owns replacement of
the fallback's Granite model-order layer-label surrogate.

## Reproduction

Configure machine-local paths in the ignored local environment, then run the
registered dry check before the live study:

```bash
RUN_ROOT="${SIMLLM_WAVE6_RUN_ROOT:?configure SIMLLM_WAVE6_RUN_ROOT}/codex/playcpu_framework_oracle"

python examples/framework_oracle_v1/run_study.py \
  --cache-dir "${SIMLLM_HF_CACHE:?configure SIMLLM_HF_CACHE}" \
  --run-dir "${RUN_ROOT}" \
  --vllm-source "${RUN_ROOT}/source-build-gcc13" \
  --sglang-source "${SIMLLM_SGLANG_SOURCE:?configure SIMLLM_SGLANG_SOURCE}" \
  --sglang-python "${SIMLLM_SGLANG_PYTHON:?configure SIMLLM_SGLANG_PYTHON}" \
  --check-only
```

The dry check emitted `"artifacts_written":0`, verified the frozen source
hashes and parameter families, and performed no framework run. Remove
`--check-only` for the registered execution. Its process exit status is zero
when all fatal guards pass even if a scored behavioral expectation fails. The
machine-readable `summary.json` therefore retains `"status":"FAIL"` for the
7/8 behavioral result without misrepresenting the run as a harness failure.

## Contradiction sweep

Per the closure contract, the sweep reports rather than edits top-layer prose:

- `README.md:283-284` still calls the independent CPU comparison the one open
  half, and `README.md:301` says that comparison is still open.
- `docs/README_PRO.md:420`, `:464`, and `:533-535` still describe the
  framework oracle as blocked on a CPU-capable vLLM build and list PLAY-5 and
  PLAY-6 as remaining.
- `docs/architecture.md:492` links to the CPU pre-play module but makes no
  contradictory blocked or open-task claim.

The generated task-progress block and mechanically checked open-task counts in
`docs/README_PRO.md` are regenerated for ledger consistency. Its stale prose
claims above remain reported hits, not outcome-dependent edits.
