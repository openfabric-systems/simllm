# Surrogate conformance expectations

These expectations freeze the acceptance surface for the surrogate serving
loop: an engine-shaped continuous-batching estimator that emits step
records and KV cache work into the existing ledger and metric chain, as a
registered estimator model class and never a framework precision level.
They are committed before the surrogate exists. The design basis is an
executed source audit of the pinned vLLM v1 scheduler; its central ruling
is frozen here: with virtual-time admission, explicit stable request
identities, asynchronous scheduling off, pinned source hashes and a pinned
KV geometry, the scheduler's decisions are deterministic, so conformance
is scored as exact comparisons that can genuinely fail, and bands exist
only for wall-clock phenomena. Two authoring rules from the previous wave
bind this freeze: no guard forbids the study's own purpose, and no frozen
clause presumes the surrogate and the live engine decide alike; measuring
that is the study.

## Fixed configuration (fatal when violated)

- The pinned vLLM distribution 0.26.0 with the scheduler source SHA-256
  `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`
  recorded and matching at run time; any source-hash mismatch voids the
  run (version drift is never a band).
- In-process engine, VLLM_ENABLE_V1_MULTIPROCESSING=0, v1 engine,
  asynchronous scheduling off, prefill schedule interval one, no
  speculative decoding, no LoRA, no multimodal input, explicit
  num_gpu_blocks_override, explicit stable request identifiers with
  virtual arrival times through the admission gate (the object-identity
  tie-break fallback is prohibited by a uniqueness guard).
- The causal tuple recorded per cell: resolved scheduled-token budget,
  max_num_seqs, chunked-prefill enablement, long-prefill threshold,
  max_model_len, queue policy, scheduler block size, block count, reserve
  mode and watermark. For non-speculative cells the resolved budget must
  equal max_num_batched_tokens.
- The live side records BOTH the native scheduler output and its
  step-record projection; conformance never compares two projections of
  one translator alone.
- Engine-internal constants are declared with source file and line; the
  fitted-constants prohibition applies only to constants of this
  repository's authorship.

## Families (scored exact unless stated; a miss is a published finding)

- F1 budget by sequence cap: at least two scheduled-token budgets crossed
  with two max_num_seqs values on prefix-free workloads under a fatal
  no-allocation-failure guard scoping this family only. Exact: nonempty
  step count, ordered per-step request sets, per-request scheduled token
  counts, batch totals, phases, post-step contexts, sampled identities
  and admission order, surrogate versus live.
- F2 chunk boundaries: prompts at budget minus one, budget, budget plus
  one, and at multiples of the long-prefill threshold, chunking enabled
  and disabled. Exact chunk sizes, step counts, and the stop-at-head
  behavior when chunking is off.
- F3 capacity, preemption and recompute: at least two block capacities
  and two concurrency shapes around the admission threshold. Exact
  allocation outcomes, preemption victim and step, recomputed token
  interval, resumed position, final step count.
- F4 prefix cache: from an empty pinned pool, repeated prefixes of zero,
  one and several full hash blocks. Exact hit-token counts, the
  full-hit last-token recompute, eviction order, and block lifecycle
  equivalence under one stable block-identity bijection (never numeric
  bands).
- F5 admission: the arrival_admission_v1 frozen workloads replayed
  through both sides. Exact gate admission order, first release step,
  batch composition, queue time and token counts for both arrival
  offsets and both load shapes.
- F6 metric reachability: identical records fed through the same lowerer
  and device runtime must yield identical StepResult, TTFT, TPOT and KV
  accounting; the surrogate's own records then flow end to end and each
  request's TTFT and TPOT equal the values computed from the live
  engine's records priced identically. This family separates translator
  fidelity from scheduler fidelity.
- F7 KV work stream: both sides emit ordered KV cache work (the live
  side through the normalization bridge, the surrogate natively);
  exact equivalence of action sequences, token intervals and per-request
  block counts under the same block-identity bijection as F4.
- W wall time (banded, the only banded family): after separating engine
  construction from the steady loop, the surrogate's median loop wall
  time over seven runs is at most one hundredth of the live engine's on
  the largest frozen workload (the one-hundred-times relation chosen
  before any run); virtual timestamps and TTFT/TPOT never enter this
  band. The absolute times and machine are disclosed.

## Fatal guards

Setup guards only, never decision-agreement presumptions: source hashes;
configuration tuple recording; identifier uniqueness; capacity pinning;
inactive features verified inactive; token conservation on both sides;
the F1-scoped no-allocation-failure guard; append-only per-attempt
evidence with native outputs retained; chronology. A violated guard voids
the run with findings.

## Closure

A full pass certifies the surrogate as a faithful stand-in for the pinned
scheduler on the frozen surface and establishes the wall-time class that
makes candidate-space scans affordable; the certification is re-earned at
every framework pin bump. Misses are published findings that bound the
surrogate's envelope; nothing here claims silicon accuracy, and the
estimator stamp keeps every downstream number labeled. The registered
tasks that own the groundwork (sampled identities emission, the KV
normalization bridge as the first VLLM-11 slice) are cited by the
implementation, and the surrogate itself lives under the deploy module's
estimator model class per the standing architecture ruling.
