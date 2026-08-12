# Routing lifetime v1 results

PLAY-13 and CORE-34 are complete. An enabled routed supply now selects one
read-only packed arena as routing authority, while the nested
`simllm-routed-experts-v1` object remains the explicit validation-time and
compatibility form. One core lifetime record carries each joined request until
its captured routing is consumed, its scheduler finish arrives, and both
final-token collective phase masks are complete.

The genuine-risk result is 3/3 behavioral families and 6/6 instances. The two
memory cells passed their observed retained-footprint and reduction bands. The
two clean lifecycle cells closed every request with zero live views. The two
fault cells retained the raw completion event but rejected close after one
lifecycle bit was suppressed. Exact uint8 layout, traffic identity, source
identity, view ownership and unit executables remain separate fatal-unscored
evidence classes.

The traffic byte and hash values below are the historical pre-TRAF-25
source-multiplied observations. The corrected single-engine traffic table is in
[the token ownership results](../token_ownership_v1/RESULTS.md#routing_lifetime_v1).

## Provenance and chronology

The expectations were frozen at commit
`6fa7c4acc059a16ac2b1054f9538358404dc74ce`. Its registered `--check-only`
command validated the complete sweep and produced no artifact. This preceded
implementation and every result-producing run.

The implementation and result chronology was retained without rewriting:

1. The first result-producing working-tree run completed the memory and all 32
   traffic comparisons. The clean one-request lifecycle also completed. The
   unmodified three-request serial graph then reached
   `CoarseDeviceRuntime`, whose additive report rejected a participant-local
   path that began before its one global critical predecessor. It raised
   `operation 'step-0:layer-1:rank-1:compute' has overlapping visits on its
   selected additive critical path`. No result JSON was produced. Its sidecars
   remain under `$SIMLLM_ROUTING_LIFETIME_RUN_ROOT/run-1-failed/`.
2. The study then tightened participant-local dependency frontiers to
   whole-operation barriers only for lifecycle execution. Operation work,
   request correlation, layer identity and completion boundaries were
   unchanged. Run 2 passed, but its suppression report counted every matching
   per-step completion rather than the final-token event alone.
3. Commit `d5fc0934f9c7fd174e87091118382f83b1be2505` landed the arena,
   lifecycle, traffic integration, tests and study harness. Run 3 passed with
   the final-token event counter corrected to one in both fault cells.
4. The public builder validates exact uint8 layout before returning. Commit
   `95603a6d9b33ba8919a9e63f7ff6ac4cd1ce6ad0` therefore moved the
   by-construction arena bound and direction out of the scored memory
   predicate while retaining every frozen threshold and raw value. Run 4 is
   the corrected evidence run.
5. Commit `896609b7452b38795cd55c6cae0d200f668ee73f` added rollback of
   newly published sidecars if the final bookkeeping append unexpectedly
   fails after prevalidation. Run 5 repeated the complete sweep.
6. The full portability gate then treated one combined forbidden-character
   literal as a personal Windows path. Commit
   `a434e6e98541b62ff8a4f6299eb04223bdf0ca53` spelled those characters
   separately without changing validation behavior. Run 6 is the final
   result.

The final `$SIMLLM_ROUTING_LIFETIME_RUN_ROOT/run-6/results.json` is 62,567
bytes with SHA-256
`0955f931b4a38b9221682e6a97dacf051929cb7831c970da5fa9864445e2a559`.
It records run head `a434e6e98541b62ff8a4f6299eb04223bdf0ca53`, Python 3.12.12 and observed
htsim gitlink `fc4400e4ca619223481536632074045cb6af2756`. The gitlink is provenance
only. The study records `gitlink_equality_required=false` and never compares
it with a frozen live-pin literal.

## Inputs and configurations

All external inputs retained their frozen identities:

| Input | Bytes | SHA-256 |
|---|---:|---|
| Granite capture | 656,205 | `5f55fccc265ee5519430a0f73d6631e49aa547ec07fcb81034e9fc2b4d9fead6` |
| Joined replay run | 1,831 | `b4d38a09011caf6de159c22133264d62a2727063496953f4337b17d79cfde93e` |
| 32 scheduler steps | 12,666 | `824cd9557293328bb42b593ac893b6a067302e545b087c9219195ccb8031d755` |
| Validation routing projection | 159,957 | `24e986e989e21f1bfe7e758d4470928c82c3bbaec06072a839743b9b17d7cf5f` |
| Archived prefill GOAL | 334,432 | `08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92` |

The memory sweep used joined prefixes of one and three requests. The traffic
and three-request lifecycle used all 32 recorded steps, 24 MoE layers, 32
experts, top-k eight, EP ranks 0 through 7 and owner `expert_id % 8`. The
three-request lifecycle added the frozen empty step 32 carrying only the
delayed `r2` scheduler finish. The one-request cell retained `r0` through its
recorded delayed finish in step 24.

The lifecycle dependency-barrier projection is a study configuration, not a
new timing model. CORE-35 records the original participant-local report gap.
No TTFT, TPOT or JCT relation is scored here.

## Scored behavioral relations

### MEM-B1: retained-routing reduction

Raw retained sizes were collected before the study's explicit layout oracle.
The Python walk counted every object identity once from one `RoutedExperts`
root. The arena observation is the payload file length.

| Requests | Tokens | Legacy bytes | Legacy bytes/token | Arena bytes | Arena bytes/token | Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45 | 283,939 | 6,309.76 | 8,640 | 192.0 | 32.863x |
| 3 | 115 | 717,061 | 6,235.31 | 22,080 | 192.0 | 32.476x |

Both legacy values lie in the frozen `[6,000, 6,600]` band, and both
reductions lie in `[32, 34]`. MEM-B1 passed 2/2 genuine-risk instances. The
builder never inspects the legacy object graph or its retained size, so either
row could reach this observation and fail those bands.

The 192-byte packed stride and the signed storage direction are not counted as
behavioral passes. Exact uint8 layout is a construction guard that runs inside
the public builder before it returns. Those observations are retained below as
fatal-unscored evidence.

### LIFE-B1: clean close

The raw registry counts were evaluated before `audit_closed()`:

| Cell | Closed | Live requests | Live views |
|---|---:|---:|---:|
| `r0` | 1 | 0 | 0 |
| `r0`, `r1`, `r2` | 3 | 0 | 0 |

LIFE-B1 passed 2/2 genuine-risk instances. The runtime completion stream did
not pin these counts: a wrong cursor, delayed-finish join, request association,
mask update or release order could leave a request live after the same raw
events. The later end-of-run audit passed both cells and is fatal-unscored.

### LIFE-B2: suppressed final-token end flag

Each fault run replayed the same graph, result and report evidence through a
fresh registry. The subclass suppressed only one `_mark_end_flag` call. It did
not remove or alter the runtime event.

| Request | Phase | Layer | Raw final-token logical completions | Exit state | View live | Diagnostic |
|---|---|---:|---:|---|---|---|
| `r0` | dispatch | 7 | 1 | `FINISH_FLAGGED` | yes | `dispatch missing layers [7]` |
| `r2` | combine | 19 | 1 | `FINISH_FLAGGED` | yes | `combine missing layers [19]` |

Both audits failed closed with the request, phase and model layer named. Both
arenas also rejected close while the request view remained live. LIFE-B2
passed 2/2 genuine-risk instances. The raw subjectless completion was present
before the lifecycle decision, so the runtime's positive completion oracle
does not entail whether a missing observer bit is detected.

## Fatal exact and structural evidence

The validation-time object and arena authority produced identical
`MoeAllToAll` records for all 32 real steps. Equality includes every aggregate
pair and every per-request pair row. Direct GOAL text, execution-graph JSON and
graph-rendered GOAL text were byte-identical on every step. Prefill step 0
matched the archived 334,432-byte GOAL and SHA-256
`08a0403af66ff8a9d6b18f93afd15ae0bc925cc85555acf8a0593438a3d7bc92`.
These are model-preservation invariants and add no behavioral pass.

Both packed cells had exactly `tokens * 24 * 8` payload bytes. The strict
reader and focused tests reject unknown schemas and fields, duplicate request
identities, noncanonical order, gaps, overlap, wrong lengths, extra bytes,
truncation, digest changes, out-of-range experts, 257 experts, more than 64
layers and changed join provenance. Exactly 256 experts is accepted. The mmap
is read-only, exports no long-lived Python memoryview, and rejects close with
live request views. The index and payload contain no gate weights.

Lifecycle tests separately reject premature release, cursor gaps and overflow,
unknown finishes, duplicate consumption and post-finish scheduling. A subject
WQE completion cannot set a request end bit. Cached-prefix admission and
idempotent recompute advance only monotonic unique coverage. These are fatal
guards, not scored relations.

The final evidence classes remain separate: two run configurations, three
scored behavioral families with six instances, 32 fatal exact traffic rows,
three fatal structural families and no native executable. Repository pytest is
reported separately in the closure commit.

## Entailment and genuine-risk analysis

The final genuine-risk fractions are `2/2` for MEM-B1, `2/2` for LIFE-B1 and
`2/2` for LIFE-B2. No conservation identity, inactive field, explicit drain,
exact layout, byte identity or forced zero increases those denominators.

The pre-freeze plan expected arena bytes per token to be part of MEM-B1. The
implemented public builder validates exact one-byte layout and payload length
before it returns, so that subrelation is unreachable as an independent raw
failure and is withdrawn from the behavioral denominator. The retained legacy
size and resulting reduction are still raw, unpinned observations, preserving
both memory instances as genuine risk.

LIFE-B1 reads raw counts before the fatal close audit. LIFE-B2 keeps the raw
completion and tests the independent registry decision after fault injection.
No earlier fatal oracle pins either lifecycle outcome.

## Contradiction sweep

The post-closure sweep found no relevant statement in `README.md` and no
relevant statement in `docs/architecture.md`. It found one stale authority
statement in the integrator-owned developer guide: the preplay module-status
row in `docs/README_PRO.md` says that "the per-token routing projection feeds
the traffic expansion." In an enabled arena run, that projection is now the
validation-time compatibility form and the mmap arena feeds expansion. The
row is reported here and intentionally not edited. The generated progress
block and module open counts in that file were regenerated as required by the
closure contract.

The packet-status row in `docs/README_PRO.md` also matched the sweep terms but
does not contradict this closure: it discusses an ABI-v2 packet-issue run and
makes no claim that packets carry request identity. The broader sweep found
the same stale projection wording in `docs/modules/traffic.md` and a future
compute task in `docs/modules/compute.md` that still names
`simllm-routed-experts-v1` as its input. Those docs are outside this task's
owning-doc set and are reported for integration rather than edited or assigned
an unallocated TRAF or COMP ID.

## Closure scope

PLAY-13 registered this acceptance:

> Replace the active captured-routing Python object graph with one joined-run
> routing arena. Pack expert identities as contiguous uint8 values in token,
> MoE-layer and top-k-slot order, reject models wider than 256 experts, and
> publish a strict request index containing token offset and count next to the
> replay run. A later process must open the payload read-only through mmap and
> retain only request views, while `simllm-routed-experts-v1` remains an
> explicit validation-time compatibility form rather than a second live
> authority. Acceptance measures retained routing bytes per token on the real
> Granite capture for one and three joined requests, preserves every routed
> pair, per-request attribution row and GOAL byte against the compatibility
> form, and proves malformed, truncated, overlapping and wider-than-uint8
> arenas fail before traffic expansion.

`build_routing_arena` and the optional join output cover the packed sidecar and
strict request index. `open_routing_arena` covers read-only later-process mmap.
`RoutedMoeSupply` enforces exactly one arena or compatibility authority.
MEM-B1 covers both measured request prefixes. The 32-step exact comparison
covers routed pairs, request rows, direct GOAL, graph JSON and graph GOAL.
Focused corruption tests cover every named rejection. Every PLAY-13 clause is
demonstrated.

CORE-34 registered this acceptance:

> Make one per-request routing-lifetime record the mutable completion authority
> for joined replay requests. It carries stable request and join provenance,
> arrival time, an arena offset/count view, a monotonic unique-token consumption
> cursor, scheduler-finish state and separate dispatch/combine end masks for at
> most 64 model layers. The only legal state path is `JOINED -> ADMITTED ->
> EXECUTING -> FINISH_FLAGGED -> DRAINED -> CLOSED`; the arena view may be
> released only at `CLOSED`, which requires the scheduler finish flag, full
> masks and cursor equal to captured token count. Join final-token collective
> completions through the existing `ExecutionGraph -> CompletionEvent ->
> CompletionReducer` path without treating WQE or packet events as request
> completions. Acceptance drives the real Granite replay records through clean
> one-request and three-request lifetimes with zero live views at exit,
> deliberately suppresses one dispatch and one combine end flag in separate
> runs and requires fatal diagnostics naming the request, phase and missing
> model layer, and rejects premature release or any non-closed end-of-run record
> without partial lifecycle mutation.

`RequestRoutingLifetime` contains every named field, and
`RequestLifetimeRegistry` owns its state transitions and atomic staged step
updates. `CompletionReducer(lifetimes=...)` is the deployed join. The clean and
fault tables cover all four real replay cells. Focused tests cover WQE-event
exclusion, early release, cursor failure and atomic rollback. Every CORE-34
clause is demonstrated under the report-compatible barrier configuration.

The unmodified participant-local serial graph remains a distinct runtime
report limitation and moves to CORE-35. Per-packet request attribution remains
deliberately absent and stays registered as BACK-39. Neither residual weakens
the routing storage or fail-closed lifetime results above.
