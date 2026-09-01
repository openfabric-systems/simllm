# VLLM-48 live collective timing expectations

This freeze defines the mechanism-only evidence for the optional in-situ
collective timing seam. It precedes implementation and the first live run.
Timing values from this host are recorded but never used to calibrate an A100
or H200 floor. VLLM-49 owns that matched-hardware evidence.

## Frozen source and workload

The live authority is vLLM 0.27.1 at commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`, built from source for CPU. The
model is `ibm-granite/granite-3.0-1b-a400m-instruct` at revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`. The engine uses tensor parallel
size two, gloo, eager execution, disabled V1 engine multiprocessing, explicit
request IDs and a manual `llm_engine.step()` loop.

Each fresh run admits the two frozen prompt-token vectors together. Both use
greedy sampling, ignore end-of-sequence termination and produce exactly two
tokens. With chunked prefill disabled and the frozen batch budget, the first
engine step processes both prompts and the second processes both decode tokens.

Pinned source inspection predicts exactly 50 tensor-parallel communicator
invocations on driver rank zero per model step: one embedding all-reduce, one
attention output all-reduce and one mixture-of-experts output all-reduce for
each of 24 decoder layers, then one logits gather. The run therefore has 100
expected calls. The source-derived sequence, tensor shape, dtype, group and
layer location are the independent oracle for kind, payload bytes, world size,
tag and layer metadata. A zero-call or single-rank run is void because it
cannot challenge a broken seam.

## Fatal guards

A fatal guard voids the complete run. It is never converted into a scored
failure or included in a pass denominator.

- The installed distribution is `0.27.1+cpu`; its source commit and the frozen
  hashes in `study_config.json` match; the selected worker and model runner are
  stock CPU classes; every parameter and collective tensor is on CPU.
- Tensor parallel size is exactly two, the communicator backend is gloo, V1
  engine multiprocessing is disabled, both explicit request IDs are preserved,
  both runs finish in exactly two manual engine steps, and no request fails.
- Each run contains exactly 100 independently predicted communicator calls and
  every call has a positive payload, world size two, a nonempty group tag, a
  positive monotonic-clock service value and one environment label. Missing or
  additional calls void the run before scoring.
- The two attempts use fresh processes and new output directories beneath the
  configured append-only root. Neither attempt overwrites or deletes earlier
  evidence.
- All mutation controls fire: dropping one call, changing one payload byte,
  changing one second-run kind, deleting the optional capture envelope,
  bypassing default environment refusal, or omitting the acknowledgement stamp
  must make its owning predicate fail.

## Scored behavioral families

Every predicate below can fail independently of JSON construction. Fatal
guards and schema structural checks remain separate from the behavioral count.

1. **M1, call and metadata conservation, 2 instances.** For each fresh run,
   join the 100 source-predicted calls one-to-one to captured calls by step and
   call ordinal. Kind, payload bytes, world size, group tag and layer metadata
   must match exactly, with no missing, duplicate or extra row.
2. **M2, shape determinism, 1 instance.** Remove service values and other wall
   observations, then compare the ordered capture projection from the two fresh
   runs byte for byte. Every step and collective shape field must match.
3. **M3, schema compatibility, 2 instances.** An accepted old record with no
   collective field must load and serialize to its exact original bytes. A new
   record containing the timing envelope must load and serialize to exact
   canonical bytes. The pre-existing byte-locked fixture is not rewritten.
4. **M4, comparison refusal, 1 instance.** A timing observation and aggregate
   floor with equal kind, payload bytes and ranks but different system/backend
   identities must raise the dedicated cross-environment refusal when no
   acknowledgement is supplied.
5. **M5, acknowledged comparison, 1 instance.** The same exact-coordinate
   comparison with deliberate acknowledgement must return a score whose
   cross-environment acknowledgement stamp is true. The stamp must appear on
   every accepted result, not only on a run summary.

The maximum behavioral score is seven instances. A nonvoid result must report
each evidence class separately. Local collective service magnitudes are
unscored diagnostics because CPU gloo does not identify A100 or H200 service.

## Interpretation

Passing M1 and M2 qualifies the live capture mechanism for this exact pinned
CPU path and Granite workload. Passing M3 preserves the optional schema seam.
Passing M4 and M5 qualifies the comparator's environment discipline. It does
not validate a GPU timing value, calibrate a floor, or establish a signed
time-to-first-token or time-per-output-token improvement. Those claims remain
open under VLLM-49.
