# Dense vLLM pipeline execution

The real vLLM 0.27.1 batch queue completes all nine declared-runtime cases,
including TP4 x PP2. Decreasing link bandwidth increases the resident-KV
decode window in all four paired configurations. This closes VLLM-10's FIFO
and per-stage execution path. VLLM-52 owns physical timing calibration and
per-request PP attribution; TRAF-8 retains packet-backed serving qualification.
These results do not establish MI210 or CX6 performance accuracy.

## Scope and chronology

Expectations commit `575f94c7` precedes implementation and every study run.
It freezes PP2/PP4, TP1/TP2, 100/200 Gb/s, an uneven split, chunked prefill,
an eight-rank smoke, output identity and compute/communication relations.
The additional common resident-KV window follows the maintainer's decode-only
scope clarification. Those window checks are additional post-specified
regressions, not a rewritten expectation freeze. All fatal guards pass.

The driver constructs a local five-layer Llama configuration: hidden width
128, MLP width 256, eight query and KV heads, head width 16, vocabulary 1024
and BF16 tensors. No weights or physical GPU are used. The real scheduler,
KV allocator and batch queue produce the steps; SimExecutor fabricates token
512. The selected compute envelope is explicitly named `declared-study`,
100 TFLOP/s and 1 TB/s, with efficiency one. Its tiny-model times must not be
read as a prediction for a production checkpoint.

Each stage occupies one coarse eight-slot node, with consecutive TP lanes
within that node. Idle slots are unused. The runtime's declared local link
rate stays fixed while the cross-node rate varies. This is not the four-GPU
MI210/xGMI topology, which needs an explicit calibrated profile.

## Decode results

First, four 48-token prompts each generate eight tokens through the real
engine, exercising multiple prefill chunks. A second fixed cohort finishes
prefill before a common decode window opens. Its requests retain KV pages,
admit no new arrivals and have no preemption or prefill inside the window.
Each request advances at least eight tokens; the exact increments and initial
cached lengths are retained in [results.json](results.json). Initialization
TTFT remains a diagnostic. Aggregated versus disaggregated PD is a separate
future experiment.

| PP | TP | Link (Gb/s) | Resident decode window (ps) | Mean request token interval (ps) |
|---|---|---|---|---|
| 2 | 1 | 100 | 19,730,720 | 2,466,340 |
| 2 | 1 | 200 | 19,084,920 | 2,385,615 |
| 2 | 2 | 100 | 11,418,384 | 1,427,298 |
| 2 | 2 | 200 | 11,172,624 | 1,396,578 |
| 4 | 1 | 100 | 19,267,160 | 2,408,395 |
| 4 | 1 | 200 | 18,325,080 | 2,290,635 |
| 4 | 2 | 100 | 11,141,704 | 1,392,713 |
| 4 | 2 | 200 | 10,866,496 | 1,358,312 |
| 2 | 4 | 100 | 7,794,240 | 974,280 |

The four paired bandwidth instances belong to one behavioral relation family.
Decode does not become twice as fast when bandwidth doubles because compute,
local collectives and queue occupancy remain. The engine fills the configured
two- or four-ticket queue; output histories contain exactly the requested
tokens and preserve request identity.

## Independent checks

**Compute/memory floor.** Five layers contain
`5 * (4 * 128^2 + 3 * 128 * 256)` non-embedding matrix parameters.
At two bytes per parameter and 1 TB/s, the sum of per-stage weight-read
floors plus the declared final-stage LM head is 1,900,544 ps at TP1,
1,081,344 ps at TP2 and 671,744 ps at TP4. Every measured mean request
interval exceeds its floor, even before adding KV reads and activation
traffic. The final-stage LM head retains the compatibility model's full
vocabulary proxy; VLLM-52 owns captured final-stage service. Being above this
floor is a necessary condition, not a calibration result.

**Communication and overlap oracle.** An independent recurrence grants each
stage at `max(incoming_arrival, previous_source_release)`, adds fixed compute
service, and holds the source through its outgoing send. Eight cases vary
PP2/PP4, two/four batches and 100/200 Gb/s. Every completion equals this
recurrence exactly and lies between its causal chain floor and independent
serial execution ceiling. A 256-byte tensor takes 20,480 ps at 100 Gb/s;
two boundary tensors take 40,960 ps. At 200 Gb/s that term is exactly
20,480 ps. For example PP4, four batches, 100 Gb/s completes at 315,760 ps,
between its 162,880 ps causal floor and 651,520 ps serial ceiling. The separate
compute-only four-stage/three-batch fixture gives 60,000 ps for 10,000 ps
stages, exactly `(P + M - 1) * C`. Oracle cases are not added to behavioral
relation counts.

**Engine reachability.** Lowered graphs enter the existing runtime, produce
CompletionEvents and StepResults, and retire through the future consumed by
the unmodified engine. Link-rate changes reach returned token times and the
common decode window. The exact PP1 immediate path remains an explicit
bypass. This establishes mechanism reachability, not agreement with silicon.

## Reproduce

Use the pinned vLLM 0.27.1 CPU environment and install this checkout without
changing its dependency pin. No tokenizer, checkpoint download or GPU is
required:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
VLLM_CPU_KVCACHE_SPACE=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
python examples/vllm_pipeline_v1/run_study.py \
    --output-dir "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/pipeline-study"
```

The output directory must be fresh. The full report includes every stage's
release, start and completion evidence; the committed compact result records
its SHA-256. Unit fixtures additionally cover draining, capacity, shutdown,
failure propagation, callbacks, shard conservation, source-send fences,
unsupported configurations and repeated future consumption.

The resolved-class API entry point also passes a localhost HTTP smoke using
the pinned CPU engine: two 48-token prompts return four token-512 outputs
each. This exercises the multiprocessing EngineCore, API class resolution,
batch queue and teardown. A first environment-only launch failed because the
Unix socket path exceeded the operating system limit; shortening the IPC
temporary path resolved it without a code change.
