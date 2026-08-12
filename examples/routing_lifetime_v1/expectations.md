# Routing lifetime v1 expectations

Tasks: PLAY-13 and CORE-34. BACK-39 records the deliberate packet boundary.

## Decision and claim boundary

The joined routing arena becomes the sole routing authority in an enabled run.
Its binary payload contains only uint8 expert identities in
`[token][MoE layer][top-k slot]` order. Gate weights stay in the offline
capture because no traffic calculation consumes them. The strict
`simllm-routed-experts-v1` object remains a validation-time and compatibility
form. It must not remain resident beside an enabled arena or advance routing
state independently.

One mutable core record carries a joined request from arrival through final
close. It owns request identity, join provenance, arrival, the arena token
offset and count, a monotonic unique-token cursor, the scheduler finish flag,
and dispatch and combine end masks. The only legal state path is
`JOINED -> ADMITTED -> EXECUTING -> FINISH_FLAGGED -> DRAINED -> CLOSED`.
`CLOSED` requires the scheduler flag, full masks for the model's actual MoE
layers, and cursor equal to captured token count. The request view is released
only after those conditions hold.

The decision-relevant representation relation is measured retained routing
bytes per forwarded token on the real Granite capture. The lifecycle relation
is clean closure versus deliberate loss of one final-token collective end
flag. If the packed form does not retain the expected reduction, if a clean
run leaves a live view, or if a missing layer can pass end-of-run audit, the
design is rejected.

This study claims routing storage, exact traffic preservation and request
close-out through `StepRecord -> ExecutionGraph -> CompletionEvent ->
CompletionReducer`. It does not claim a new timing model. TTFT, TPOT, physical
GOAL bytes and per-request traffic attribution must remain unchanged. Packet
attempts remain backend-private. BACK-39 names the byte-extent, packetization,
retry and terminal-event work required before any packet can be joined to a
request.

## Pre-freeze source audit

The audited repository state is commit
`b0438884273c27fa40c6d59da874289b5a5a41bf`.

- `simllm/preplay/join.py:422-484` reads and hashes the source trace, creates
  joined request records and appends their bookkeeping facts atomically. The
  join has no routing sidecar output today.
- `simllm/preplay/routing.py:28-95` defines the nested routed layer, token,
  request and run objects. Lines 165-210 validate every forwarded token.
  Lines 482-544 materialize the complete joined object graph after rereading
  the trace. No 256-expert ceiling is present.
- `simllm/traffic/routed_moe.py:110-151` requires that object graph in every
  `RoutedMoeSupply`. `simllm/traffic/step_comm.py:338-397` performs a linear
  request lookup, allocates prefill or decode token tuples, and slices them by
  scheduler context. Lines 412-512 dereference every selected layer and expert
  to derive the physical and per-request pair tables.
- `simllm/backends/step_lowerer.py:140-263` projects routed collectives into an
  execution graph. Each MoE operation carries request correlation and model
  layer, while `CollectiveWork.channel_hint` names dispatch or combine.
- `simllm/core/runtime.py:2097-2114` emits one subjectless completed event for
  each graph operation after finer-grained WQE events. The subjectless event is
  the request close-out input. WQE-subject events are not end flags.
- `simllm/core/completion.py:256-393` validates graph, runtime and completion
  evidence and reduces it into live request metrics. It currently has no
  joined-request lifecycle authority and never consumes scheduler finish IDs.
- `simllm/adapters/vllm/executor.py:1328-1342` records delayed scheduler
  finishes on empty drain steps. It also documents that the in-process
  `LLM.generate` loop can stop before the final drain. This study therefore
  supplies one explicit final completion-bearing drain for `r2`; it never
  infers the scheduler flag from token exhaustion.

The external source is supplied through `SIMLLM_MOE_E2E_ROOT`; no site path is
part of the tracked study. The pre-freeze audit observed:

- `capture/granite-greedy.jsonl`: 120 LF-terminated rows, SHA-256
  `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6`.
- `replay-400g/run.json`: 1,831 bytes, SHA-256
  `b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e`.
- `replay-400g/steps.jsonl`: 12,666 bytes and 32 records, SHA-256
  `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755`.
  Delayed finishes for `r1` and `r0` appear in steps 8 and 24. The final `r2`
  drain is absent, matching the audited in-process limitation.
- `replay-400g/routed-experts.json`: 159,957 bytes, SHA-256
  `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f`.
- `replay-400g/htsim/step-000000.goal`: 334,432 bytes, SHA-256
  `08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`.

The capture has 45 forwarded tokens for `r0`, 19 for `r1`, 51 for `r2`, 24
MoE layers and top-k 8, for 115 forwarded tokens and 22,080 packed expert-ID
bytes in total. These observed inputs are not frozen live-pin literals.

## Representation sweep

The retained Python size walk starts at one `RoutedExperts` root, follows
dicts, tuples, lists, sets and object `__dict__` values, counts each object
identity once and sums `sys.getsizeof`. It records the interpreter and every
raw byte count. This is retained object-graph storage, not RSS. The arena
observation is the binary payload length before any layout or digest oracle is
evaluated.

The pre-freeze Python 3.12.12 audit measured the following context values. The
study measures them again and does not substitute these values for its raw
observations.

| Joined request prefix | Forwarded tokens | Audited Python bytes/token | Expected packed bytes/token | Frozen reduction band |
|---:|---:|---:|---:|---:|
| 1 | 45 | 6,309.76 | 192 | 32x to 34x |
| 3 | 115 | 6,235.24 | 192 | 32x to 34x |

### MEM-B1: raw retained-routing reduction

For each request-count cell, first record raw legacy bytes, arena payload
bytes and token count. Then compute:

```text
legacy_bytes_per_token = legacy_retained_bytes / forwarded_tokens
arena_bytes_per_token = arena_payload_bytes / forwarded_tokens
reduction = legacy_retained_bytes / arena_payload_bytes
```

Both cells must have `legacy_bytes_per_token` in `[6,000, 6,600]`,
`arena_bytes_per_token <= 192`, and reduction in `[32, 34]`. The signed
direction is strictly less arena storage per token. These are two scored
instances. A builder that retains Python tokens, uses a wider element, adds
per-token padding or packs gate weights can reach and fail this relation.

The later exact uint8 layout check is fatal-unscored. It is evaluated only
after MEM-B1 and cannot add a scored pass.

## Lifecycle sweep

The one-request cell filters the recorded schedule to `r0` and ends after its
real delayed finish at step 24. The three-request cell consumes all 32 recorded
steps and appends an empty step 32 whose only scheduler observation is
`finished_request_ids=["r2"]`. The explicit drain is author-defined
configuration and unscored. It is required because the source run did not
observe the final in-process drain, and the implementation must not infer a
finish flag.

Each nonempty step is lowered through the serial graph path with the arena as
routing authority, executed by the coarse runtime, and reduced by
`CompletionReducer`. The lifecycle observer consumes only raw subjectless MoE
completion events. It records state, cursor, both masks and view-release state
after every step before final audit.

### LIFE-B1: clean close

The one-request and three-request cells must reach these raw exit observations:

| Cell | Closed requests | Live requests | Live arena views |
|---|---:|---:|---:|
| one request | 1 | 0 | 0 |
| three requests | 3 | 0 | 0 |

These are two scored instances. The final fatal audit runs only after the raw
counts are scored. A wrong cursor, delayed-finish join, phase mask, shared
operation association or release order can reach and fail the relation.

### LIFE-B2: suppressed final-token end flag

Replay the same raw graphs and completion events into a lifecycle observer
whose test subclass drops only the named end-bit update. It must not remove or
alter the runtime event itself.

| Cell | Request | Suppressed phase | Suppressed model layer | Required end-of-run diagnostic |
|---|---|---|---:|---|
| one request | `r0` | dispatch | 7 | names `r0`, dispatch and layer 7 |
| three requests | `r2` | combine | 19 | names `r2`, combine and layer 19 |

Each run must fail closed at end-of-run and retain its request view. These are
two scored instances. The raw runtime event remains present, so an earlier
completion oracle does not entail the lifecycle result.

## Fatal unscored traffic and structural oracles

Before any backend execution, compare validation-time `RoutedExperts` and
arena authorities over all 32 recorded steps. Every `MoeAllToAll` field,
aggregate pair, per-request pair and direct GOAL byte must match exactly. The
execution-graph JSON and graph-rendered GOAL must also remain identical for
the compared steps. These identities are fatal and unscored because they
protect the accepted model rather than test the storage decision.

The arena reader must reject an unknown schema, duplicate request, noncanonical
request order, overlapping or gapped token ranges, wrong payload length or
digest, changed trace provenance, token-count disagreement, expert identity
outside the declared range, 257 experts and more than 64 layers. Expert count
256 is accepted. The mmap is read-only, closing it with live views fails, and
no index or payload contains gate weights. Premature view release, cursor
overflow, skipped lifecycle transitions, unknown scheduler finishes, subject
WQE completion as an end flag and any nonclosed end-of-run record are fatal and
unscored.

The result report keeps run configuration, scored memory relations, scored
lifecycle relations, traffic exact oracles, structural guards and pytest
executables as separate evidence classes. Their counts are never added.

## Entailment and genuine-risk plan

MEM-B1 is evaluated from raw storage observations before exact payload length,
layout or digest checks. LIFE-B1 is evaluated from raw registry counts before
the end-of-run assertion. LIFE-B2 retains the raw runtime completion and
captures the independent close-out diagnostic after dropping only the
lifecycle bit update. No earlier fatal oracle pins any scored result.

All six scored instances can fail in a run that reaches them. None is an
author-defined sequence or configuration-forced zero. The planned genuine-risk
fractions are `2/2` for MEM-B1, `2/2` for LIFE-B1 and `2/2` for LIFE-B2. The
RESULTS report must revise those fractions if implementation or execution
shows an instance is unreachable, entailed or by construction.

## Registered command and pre-freeze dry run

Configure project-local paths in the gitignored local environment, then run:

```text
.venv/bin/python examples/routing_lifetime_v1/run_study.py --out "$SIMLLM_ROUTING_LIFETIME_RUN_ROOT" --source-root "$SIMLLM_MOE_E2E_ROOT"
```

`SIMLLM_ROUTING_LIFETIME_RUN_ROOT` must resolve under the required external
wave-6 run root for this branch. The pre-freeze dry run is the same command
with `--check-only`. It parses the complete CLI, validates only the frozen
registry, does not inspect either supplied path, imports no implementation
under study and produces no artifacts.
