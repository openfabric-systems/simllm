# VLLM-48 request-identity expectations amendment 1

This is a post-attempt-2 harness-reality amendment to the expectations frozen
at commit `2808a04`. It changes only the request-identity fatal guard. It does
not change the collective population, metadata oracle, behavioral denominator,
timing interpretation, mutation rules or any scored expectation.

## Reason and pinned authority

The original guard treated the string returned by `LLMEngine.add_request()` as
the caller-supplied logical request ID. Pinned vLLM stores that caller value as
`external_req_id`, then assigns the engine-core request an internal value of
`f"{request.external_req_id}-{random_uuid():.8}"`
(`vllm/v1/engine/input_processor.py:232-249`). The pinned `random_uuid()` emits
16 lowercase hexadecimal characters, of which the assignment retains eight
(`vllm/utils/__init__.py:11-12`). A suffix is therefore required by the stock
path and is not evidence that logical request identity changed.

The accompanying lock records the exact pinned hashes of both source files.

## Amended fatal guard

For each frozen logical ID, the assigned internal ID must match the regular
expression formed by the exact escaped logical ID, one hyphen and exactly eight
lowercase hexadecimal characters. Order must be conserved. The first captured
step must carry those exact internal IDs at the scheduler boundary. Completion
and output maps must still carry the original logical IDs exactly. A missing,
extra, reordered, malformed or cross-associated ID remains fatal and voids the
run.

No result observed before this amendment may qualify under this identity rule.
The qualifying evidence must come from two fresh live processes in a new
append-only attempt directory created after the amendment commit. The original
source-frozen `gather` expectation remains unchanged and may still fail as a
scored M1 predicate.
