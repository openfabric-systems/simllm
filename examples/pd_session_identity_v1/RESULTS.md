# Complete native session identity result

**CORE-58 closes with zero projected-byte differences across the four frozen
prompt/handoff cells.** Two fresh native vLLM processes each run the same
prefill/decode session grid. Their full ordinary results differ at exactly two
root fields: the native prefill and decode request identifiers. Every other
serialized value matches, including bytes transferred, timing, token outputs
and metadata.

The new `VllmPdRequestResult.to_comparison_json()` method deep-copies the
ordinary result and declares those two exclusions. It validates each excluded
identifier first. It preserves nested identifiers and future members. The
study's deterministic JSON encoding preserves Unicode code points and scalar
types, avoiding the calibration-record encoder's string normalization.

## Native result and evidence classes

Both native processes reproduce every unchanged compact `pd_session_v1` cell.
The model computes the prompt, completes the declared key/value cache handoff,
and produces four decode tokens. Increasing handoff time delays the first
visible token by exactly that increase; it does not change the later cadence.

| Prompt tokens | Handoff (microseconds) | Cache bytes | Time to first token (microseconds) | Time per output token (microseconds) |
|---:|---:|---:|---:|---:|
| 8 | 100 | 393,216 | 273.376 | 77.952 |
| 8 | 200 | 393,216 | 373.376 | 77.952 |
| 16 | 100 | 786,432 | 292.912 | 77.976 |
| 16 | 200 | 786,432 | 392.912 | 77.976 |

The eight behavioral instances in two families satisfy the frozen handoff
and prompt-length relations. Sixteen separate exact oracle rows have zero
residual for cache bytes and first-token decomposition. All 105 fatal guards
hold. The 472 mutation controls are unscored discrimination evidence; they
are not added to the behavioral denominator.

Each cell retains the full ordinary serialization before and after projection,
typed snapshots of all result state, the comparison bytes, and native step
records. Per-call state remains unchanged, including fields represented only
by counts in the ordinary result. The two-process comparison covers the
complete serialized result. It does not assert that the complete execution
traces are byte-identical across processes.

The [compact publication](results.json) records all four comparison hashes,
exact raw difference paths and native artifact identities. The accepted raw
summary has SHA-256
`b795a3990df0c255986dc39f8f80ab80e59e64bd8e72e0ef6155e3e888fc4cd5`.

## Physical interpretation

At 400 gigabits per second, the two cache payloads need at least 7.86432 and
15.72864 microseconds of serialization. The frozen slower-path ceilings are
364.5728 and 679.1456 microseconds. Both declared handoffs lie inside their
respective intervals. The nonempty compute steps range from 77.952 to
114.936 microseconds, inside the frozen fixture bounds. Each first-token
latency equals the sum of its five causal timing components exactly.

This preserves the existing synthetic device model and its accepted values.
It measures no GPU kernel and runs no packet backend. Neither native process
initializes a CUDA context, the driver's state for executing GPU work.

## Chronology and project consequence

The first expectations-only commit is `04e206a`. The final freeze is
`ae0a9d7ed52e1f13be1faebb8676f2234ea1c211`; it clarifies that the older
restricted deployment-study helper is a different contract. Both precede
implementation and run commit `882c670c87153b9914397dfa6df7f67d5b705e33`.

The first attempt stops before engine construction because redirecting the
default cache hides the existing checkpoint. Its logs and void summary remain
retained. The accepted fresh run selects that existing cache explicitly,
keeps offline access mandatory and changes no source or frozen criterion.

CORE-58 supplies the identity acceptance boundary needed by CORE-53. CORE-53
still needs COMP-73's complete, key-compatible target record before a new
pricing acceptance run. Its original void study stays void. This result does
not qualify a GPU architecture, fill kernel coverage, rescore an earlier
study or establish deployment-frontier accuracy.

## Reproduction

Supply the pinned native environment, its package directory and the exact
Granite configuration from the existing offline model cache. Keep new runtime
outputs in an external directory and select a fresh run root:

```bash
HF_HUB_CACHE="$SIMLLM_MODEL_CACHE" \
XDG_CACHE_HOME="$SIMLLM_CACHE_ROOT" \
VLLM_CACHE_ROOT="$SIMLLM_CACHE_ROOT/vllm" \
python -m examples.pd_session_identity_v1.run_study \
  --output-root "$SIMLLM_RUN_ROOT" \
  --vllm-python "$SIMLLM_VLLM_PYTHON" \
  --vllm-source "$SIMLLM_VLLM_PACKAGE" \
  --model-config "$SIMLLM_MODEL_CONFIG"
```

The coordinator enforces offline native workers and returns status 2 with a
void record when an admission, fatal, oracle or behavioral condition fails.
