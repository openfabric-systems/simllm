# SGL-38 remote KV decode shape result

## Selected-key proof

The exact candidate key
`05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa`
selected once with zero comparator misses in each of two repetitions. Each
repetition let the scheduler-shaped engine seam author one ordered batch of 32
standard-decode requests. The enabled driver join registered remote prefix
length 2000 for each pool-local identity, and the worker translation produced
32 `KernelRequestShape(1, 2001)` values, each with prior context 2000.

The lookup used candidate entry 3 from record
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`,
implementation `deepseek-v3-full61-vllm-ep72-decode-b32-c2000`. Each
repetition reported exactly:

- `lookup_hits`: 1
- `lookup_misses`: 0
- `selected_entry_key_sha256s`:
  `[05d1c33cdef9c12e25eb9159adc9dc80f1cd57b6333778f9efb5fb24cd6a74aa]`

This was an import-free contract qualification, not a scored flagship rerun.
It used the merged candidate record without loading or downloading model
weights.

## Byte-identity proof

The feature-disabled session omitted the remote-prefix member from every
decode submission. Its canonical projection over every prompt token, all KV
handoff fields, the bootstrap and decode tokens, and every lifecycle timestamp
retained the pre-change digest
`48a655471cf6fd72fa42e3f5bb70355b5b12c9b8397dd96ae4ff6bacd01fc094`.

The enabled session ran twice. The preregistered CORE-58 projection was equal
across repetitions. It included stable timeline, handoff, engine, token, step,
join and pricing fields, removed only process identifiers from join metadata,
and did not compare pool-local request IDs or complete serialized result bytes.

## Mechanism and authority

`SglangPdSessionConfig.project_remote_kv_length` is default-off. When enabled,
the parent adds the prompt length to the decode submit RPC without modifying
the one-token bootstrap request. The child registers that length immutably by
pool-local request identity. `SglStepTranslator` adds it only to the logical
`ScheduledRequest.context_length` that becomes a `KernelRequestShape`.

The projection does not allocate or claim KV tensors, increment
`num_cached_tokens`, alter token-pool state, or form a batch. SGLang's scheduler
remains the only batching authority.

## Verification

- Expectations-only commit: `c765c0c`.
- Implementation commit: `fc4ef11`.
- Focused adapter and session tests: 87 passed, 1 skipped.
- Implementation-slice Ruff: passed.
- Implementation-slice full pytest: 3,011 passed, 11 skipped, direct exit 0.
- Scored flagship rerun: not run, reserved for later integrator dispatch.

SGL-38 is complete. CORE-56 remains open until the integrator publishes the
enabled live binding qualification. CORE-59 pricing mechanisms remain outside
this work.
