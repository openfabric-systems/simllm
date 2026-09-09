# Kimi K3 structural qualification protocol

This protocol qualifies the existing 93-layer logical workload against fresh
native configuration extractions. It owns the remaining COMP-54 structural
qualification. The final expectations-only commit must precede this protocol's
first execution. The implementation and corrected regression oracles already
exist: this is not preregistration of the original behavior or a claim that
these corrected assertions were unobserved.

The original `kimi_k3_structure_v1` run remains **VOID**, with a null behavioral
score. Its frozen slow-shared calculation omitted the final addition in each
of 92 expert layers. Preserve its original freeze, raw evidence and later
post-specified audit. No successor result revises that chronology.

## Physical and logical boundary

An expert layer sends the same input into a routed branch and a shared branch.
Both values must arrive before their sum can become the layer result. The
shared branch has two parallel projections, one activation and one output
projection: four visits, three sequential regions. The routed branch is nine
regions deep. A separate logical addition follows the two branches.

This graph describes data dependencies. Native framework streams can impose
extra ordering, and a framework can fuse the logical addition into another
physical kernel. Neither native scheduling nor an independently launched add
kernel is established by this qualification.

For positive logical services, a completion floor is the longest dependency
chain's service sum. A ceiling is the sum of all operation services under a
serial grant. No operation completes before every parent, and no state becomes
visible before its writer. These are logical bounds, not GPU throughput bounds.
The exact model's retained bytes and matrix dimensions are checked separately
against the inherited source-pinned geometry. Unknown physical memory demand
and service remain absent and reject physical execution.

## Independent shared-branch oracle

Use ordinary-region service `O=1000 ps`, shared-region service
`S in {1000, 3000, 10000} ps`, and final-add service `J in {1000, 2000} ps`.
For each framework and phase, predict the component completion before running:

`D = O * (4 + 3*93 + 6*69 + A*24 + 3) + 92 * (max(9*O, 3*S) + J)`

Here `A=8` for prefill and `A=10` for decode. The first term counts four outer
regions, three residual regions per layer, six sequential regions in each
Kimi Delta Attention (KDA) layer, the phase-specific Multi-head Latent Attention
(MLA) depth, and the first dense feed-forward layer. It is `892*O` or `940*O`.

There are 24 exact component cells, separate from all behavioral scoring.
The three shared-service settings cover routed dominance, a tie and shared
dominance. Doubling `J` adds exactly `92,000 ps` in each setting and phase.
At `S=10000` and `J=1000`, the known corrected depths are `3,744,000 ps` and
`3,792,000 ps`. These values are explicitly post-specified regression oracles.

For every expert layer, freeze exact branch parent sets, shared-fork topology,
the completion equation at the final addition, its direct residual successor
and its transitive reachability to the language head. A scalar match alone is
insufficient. Mutation controls separately remove each branch dependency,
serialize the shared projections and bypass the final addition. They preserve
operation counts and must be rejected by causal guards.

## State visibility and complete evidence

Every prefill cache commit is absent from its attention operation's transitive
ancestry. Every decode cache commit is a required attention ancestor. The
completion frontier contains exactly the language head followed by all 24
cache commits at their original layer indices. Controls inject a hidden prefill
cache wait, remove the decode cache wait and omit a frontier member. Each must
fail the owning guard without changing the operation count.

Keep the inherited shape grid: both frameworks, batches 1 and 3, cold-prefill
lengths 1, 4 and 16, and decode contexts 1, 17 and 257. This yields 24 graph
cells. Extract twice in fresh processes per framework. Require identical native
inventory bytes and input step-record bytes across repeats. Preserve the
frameworks' intentional final-normalization shape difference.

Reuse the existing partition, projection, native-record, lifecycle and request
audit helpers through explicit arguments. Do not call or reinterpret the old
whole-audit verdict. Reconstruct and join every retained graph, operation,
resource, event and request identity. Missing, duplicate or malformed evidence
is fatal. Frozen historical file hashes supply the existing evidence-name
manifests, not fresh numerical outcomes. The JSON contract names the exact
included prefixes, exclusions and successor domains. Check identities as sets,
not only row counts, and require every declared stage to finish.

Eight synthetic request cells vary prompt length 1 or 4 and uniform service
1000 or 2000 ps. Each produces one prefill and two decode steps through the
existing serial device runtime, completion events and step-result reducer.
Time to first token is exactly `3244 * prompt * service`; time per output token
is exactly `3268 * service`. Request completion is their prefill plus two
decode services. Doubling service doubles all three metrics; quadrupling the
prompt quadruples prefill time. These are synthetic mechanism checks.

The two inherited decode-prompt equalities follow directly from this binder's
one-new-token service rule. Move those named rows to fatal unscored guards.
The remaining inherited behavioral denominator is 62 instances in four
families. Matrix, storage, operation-service and shared-join exact oracles stay
separate. Conservation, disabled paths and corruption discrimination remain
unscored fatal guards. A failed fatal guard makes the entire run VOID and its
behavioral score null, even if all relations individually match.

## Native source and compatibility joins

Each fresh K3 extraction process emits its own proof alongside the canonical
inventory. Bind process identity, actual interpreter, framework, suite bytes,
configuration bytes, inventory and input step-record digests. Read the exact
11 semantic framework files before and after extraction. At coordinator
validation, re-read their actual files and all recorded import-origin files.
Required named modules and hashes are frozen in the JSON contract; a nonempty
origin list alone is insufficient. Reject an origin outside the installed
framework package, source drift or an identity mismatch.

The vLLM source commit is declared by its adapter. Its installed version and
selected implementation files are checked independently; this proves the
selected source envelope, not a whole clean vendor tree. The framework proof
may contain local interpreter and package paths only in external raw evidence.
Public evidence carries portable identities and hashes.

Re-extract all four inherited legacy suites in both frameworks and require
their accepted content addresses. Preserve the inherited suite and inventory
bytes, the old K3 study and its raw results. Unsupported state-type overrides,
unbound physical runtime, network emission and bottleneck interpretation must
continue to reject. No GPU, weight loading or hardware allocation is required.

## Promotion and consequence

Only a complete valid successor, independent review and software gates qualify
the two unchanged native inventory objects for publication at their existing
content-addressed paths. Keep the authored suite's `authored-inputs-only` state;
acceptance is joined through the report, result and coverage links. Never add
an acceptance flag or physical identity to canonical inventory bytes.

Success closes COMP-54's structural qualification. Physical capture remains
COMP-59, physical compute binding remains COMP-64, and distributed projection
remains CORE-54. This protocol cannot qualify GPU timing, native stream
scheduling or a K3 deployment frontier.

The finite successor stage list and each stage's check-name expansion are
frozen in JSON. Use only its framework, source, module, case, layer and grid
domains. Every stage finishes exactly once. Its evidence names have the form
`successor:STAGE:CHECK`; only `completion-oracle` is an exact-oracle row, and
all other successor rows are fatal guards. Missing, duplicated, extra or
mistyped names reject independently of the recorded verdict or row totals.
