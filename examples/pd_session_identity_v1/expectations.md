# Native session result identity expectations

CORE-58 defines an explicit comparison boundary for independent vLLM
prefill/decode sessions. The existing CORE-53 run stays void: this new study
never replaces or rescores it. The new boundary precedes its implementation
and every run that claims acceptance against it.

## Authority and projection

The native session continues to own request execution, identifiers and timing.
Its existing `VllmPdRequestResult.to_json()` output remains unchanged. A named,
opt-in comparison projection wraps a deep copy of that complete serialized
result and removes exactly two root members: `prefill_internal_request_id`
and `decode_internal_request_id`. Both must exist and contain nonblank strings.
No recursive identifier removal, suffix trimming, rounding, field allowlist or
pricing normalization is permitted. Future serialized fields remain visible.

The projection carries schema `simllm-vllm-pd-request-comparison-v1` and its
fixed exclusion declaration. It owns no clock, queue, request or pricing state.
Calling it must not mutate the native result, nested transfer metadata or the
ordinary serialization. Equal projected bytes mean all serialized information
except those two declared opaque identifiers is equal.

## Independent native runs

Launch two separate native Python processes, each constructing fresh prefill
and decode engines with the pinned vLLM 0.27.1 Granite configuration. Each runs
the original prompt-length and handoff grid: prompt lengths 8 and 16, crossed
with declared handoffs of 100,000,000 and 200,000,000 picoseconds, four output
tokens per request. Both processes use the record-absent roofline path. Native
engine source, checkpoint configuration, prompt and accepted result artifacts
are hash-pinned in [expectations.json](expectations.json).

Within each cell, retain the full ordinary result before and after projection,
the projected canonical bytes, process provenance and native step records.
Compare the two independent raw results recursively. Their exact differing
paths must be the two declared root fields. The projected results must match
byte for byte. All four compact cells in both arms must equal the unchanged
accepted `pd_session_v1` artifact, including transferred key/value cache bytes,
prefill service, first decode service, time to first token (TTFT) and time per
output token (TPOT). Lookup provenance remains absent.

Corruption controls change every preserved scalar leaf and insert a future
metadata field. Every such change must alter the projected canonical bytes.
Changes confined to the two excluded root identifiers preserve them. A nested
field with the same name is preserved. Missing or invalid excluded fields
reject. These are fatal discrimination guards, never behavioral score points.

## Relations and physical bounds

The compute fixture, GPU envelope and pricing are unchanged accepted inputs.
At fixed prompt length, doubling the declared handoff adds exactly
100,000,000 picoseconds to TTFT and zero to TPOT. At fixed handoff, doubling
prompt length strictly increases prefill service and TTFT, and TPOT does not
decrease. These two behavioral families supply eight parameterized instances
across two independent processes. The exact cache-byte and TTFT decomposition
oracles supply sixteen separate rows.

The link floor is bytes divided by 400 gigabits per second: 7,864,320
picoseconds for the 8-token cache and 15,728,640 for the 16-token cache.
The ceiling is bytes divided by 10 gigabits per second plus 50,000,000
picoseconds: 364,572,800 and 679,145,600 respectively. Each declared handoff
must lie inside its interval. A nonempty compute step stays between the
accepted fixture bounds of 1,000,000 and 100,000,000,000 picoseconds. Timing
must remain causal, and the five-term TTFT decomposition must conserve exactly.
These are sanity constraints on the existing synthetic device model; no GPU
is measured and no physical kernel calibration is established.

## Consequence and failure policy

A nonvoid result closes CORE-58 only. CORE-53 still requires COMP-73's complete
key-compatible target record before a new pricing acceptance run. Its original
void result remains unchanged. All fatal guards are unscored. Any violation
voids the run with a null behavioral score; matching rows remain findings.
