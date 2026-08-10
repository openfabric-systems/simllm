# Tier B review-triggered supplementary freeze

## Status and precedence

This expectations-only supplement responds to integration review before any
result-producing Tier B run. It adds requirements to
[`tier_b_expectations.md`](tier_b_expectations.md) and never relaxes or
replaces an earlier relation. The machine-readable literals and raw schema are
frozen in [`tier_b_review_expectations.json`](tier_b_review_expectations.json).
The original Tier A files remain byte unchanged.

Tier B still cannot run until the composed HTSIM-9 and CORE-15 producer lands.
That producer and the checker must implement this supplement without another
agreement about invocation, observation fields, retained profiles, doorbell
ownership, or the FIFO relation.

## Pinned source anchors

The SimLLM base is commit
`fc282efc91573638de7dcfae2befee1cf022011b`. All runtime citations in the
original Tier B freeze are interpreted at that commit, specifically:

- `simllm/core/runtime.py:1813-1911` emits the immutable completion-event
  projection;
- `simllm/core/runtime.py:1913-2055` selects and checks the realized
  critical-path reduction; and
- `simllm/core/runtime.py:2056-2072` publishes the remaining runtime-report
  projection and additive visit total.

Later line movement does not change these referents. A semantic change from
the pinned implementation requires a new expectations-only supplement before
it is used in Tier B.

## Checker and producer invocation contract

The registered outer command remains:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "$SIMLLM_CORE5_RUN_ROOT/tier_b" \
  --tier-b-only \
  --tier-b-producer "$SIMLLM_RNIC_TIER_B_PRODUCER"
```

The checker resolves the producer and expectations paths, requires the
producer to reside under the fresh Tier B output directory, and invokes this
exact argument vector:

```text
<producer>
--factory htsim
--expectations <resolved tier_b_review_expectations.json>
--observations <resolved Tier B output>/raw_observations.json
```

This is the Tier A port-factory seam with only its expectations schema
changed. The producer emits raw observations only. It must not emit expected
values, residuals, pass fields, family counts, or a checker verdict. Before it
atomically publishes `raw_observations.json`, it calls native
`RnicDevice::validateInvariants()` for every structural session. A producer
failure, invariant failure, partial JSON file, pre-existing observation file,
or pre-existing result file is fatal. Only after independent validation may
the checker publish `results.json`.

The check-only command accepts the future producer path as opaque, validates
the complete frozen matrix, schema, invocation vector, and output-root shape,
prints its registry confirmation by design, and produces no artifacts. It
does not inspect or invoke the producer.

## Raw observation schema

The top-level object has schema `simllm-rnic-tier-b-observations-v1`, factory
`htsim`, the full pinned base commit, and exactly three observation arrays:
`structural_single_wqe`, `structural_fifo`, and `bypass`. Exact nested key
sets are listed in `tier_b_review_expectations.json`.

Each structural cell records only inputs and observations. Its three step
rows retain the graph identity and release, the callback-to-result event
indices, canonical completion-event rows, the `ExecutionResult` boundary,
the runtime report, the reduced `StepResult`, and the latest request summary.
Runtime operation rows contain the seven attributed components and the
critical predecessor. Runtime visit rows contain submitted, eligible,
started, finished, and completed timestamps, resource identity, subject
identity, service bytes, and a native stage label. WQE rows retain the native
doorbell and network stage boundaries needed for the owner criterion below.

The callback observation is an ordered list of indices into the one canonical
`completion_events` array. `ExecutionResult.event_indices` is a second ordered
list into that same array. This represents object reuse without serializing a
memory address. The checker requires both lists to equal the full integer
range in order. No producer-provided equality flag is accepted.

Exact rational TPOT uses either JSON null or an object with integer
`numerator` and positive integer `denominator`. Additive visit totals stay in
their own objects and never appear in the seven-component attribution object.

Each bypass row names one frozen profile and carries reference and candidate
bytes for the four behavioral artifacts. Binary data uses lower-case,
even-length hexadecimal. Canonical rows and metric tuples use canonical JSON
arrays. The checker compares the raw values itself. Hashes may accompany
diagnostics but cannot substitute for the bytes. GOAL text, GOAL binary,
topology, seed, and baseline argument identity are raw input guards.

## Explicit retained bypass set

The retained Tier B bypass profile set is exactly:

```text
rnic-nn-fluid
rnic-nn
rnic-cn
dcqcn
```

There are four scored bypass identity instances, one per name above. No
profile may be silently omitted, aliased, added to the denominator, or treated
as a structural row. A deliberately unsupported profile is outside this set
and must fail configuration explicitly.

## Structural zero-service configuration

Every structural single-WQE and FIFO graph uses the following SimLLM-side
configuration:

```text
gpus_per_node                 = 8
rnics_per_node                = 8
launch_service_ps             = 0
control_service_ps            = 0
nccl_channel_service_ps       = 0
completion_delivery_ps        = 0
copy_engines                  = empty
goal_base_tag                 = 1000
rnic_rate_bps                 = R * 1,000,000,000
nvlink_rate_bps               = 100,000,000,000, unused
```

The graphs contain no KV, compute, DMA, or intra-node collective service.
The control handoff that creates each native WQE has zero local service.
Doorbell D is configured only in the native RNIC session. The selected htsim
factory is the sole network-serialization authority for L, so the coarse
profile must not add another RNIC serialization term.

## General completion boundary

For every structural step, including a step whose release is not zero or
whose J differs from its predecessor:

```text
StepResult.completed_at_ps(i) = ExecutionGraph.released_at_ps(i) + J(i)
StepResult.step_latency_ps(i)  = J(i).
```

The first release is `T0 = 7,000 ps`; every later release is the preceding
`StepResult.completed_at_ps`. Therefore the fixed three-step replay in one
unchanged cell also satisfies `T0 + (i + 1) * J`, but that recurrence is a
deduction for this replay and is not the general boundary definition.

## Objective doorbell-owner selection

The native raw timeline always exposes a doorbell stage of duration D followed
by a network stage of duration L. Tier B admits both of the following resource
projections and no others:

1. `queue_owner`: the selected doorbell visit has resource
   `nic_send_queue`, `started_at_ps - eligible_at_ps = D`, and zero service.
   The request attribution is `queue_ps = D`, `nic_ps = L` for one WQE.
2. `nic_owner`: the selected doorbell visit has resource `nic`, zero queue
   wait, and `finished_at_ps - started_at_ps = D`. The request attribution is
   `queue_ps = 0`, `nic_ps = D + L` for one WQE.

The checker selects the mapping from the raw visit intervals in every
nonzero-D row. Exactly one predicate must hold, the same predicate must hold
for all nonzero-D single-WQE and FIFO rows, and the corresponding component
equations must pass. Zero-D rows inherit that selected mapping because their
two projections are numerically indistinguishable. The producer may not emit
an owner label, and observed component values may not choose a third mapping.
This objective selection replaces the earlier conditional amendment path.

In both mappings the seven-component conservation identity remains exact and
all inactive components are zero. Additive visit totals are reported
separately and cannot satisfy conservation.

## Two-WQE FIFO live-chain fixture

Add the Tier A FIFO grid to Tier B. For each rate R in `{200, 400}` Gbit/s and
D in `{0, 1000}` ps, each of three request steps posts two signaled 4 KiB WQEs,
W0 then W1, to one SQ in one doorbell with one capacity-one serializer. Both
control operations become legal at the graph release, and W1 is the required
request endpoint. With `L = 4096 * 8 * 1000 / R`:

```text
doorbell completes = release + D
W0 network start   = release + D
W0 completes       = release + D + L
W1 network start   = release + D + L
W1 completes       = release + D + 2 * L
W1 queue wait      = W1 started_at_ps - W1 eligible_at_ps = L
J_fifo             = D + 2 * L.
```

The live `ExecutionResult`, `StepResult`, TTFT, each inter-token interval, and
TPOT all use `J_fifo`; absolute completion uses the preceding-release form
above. W1 wait equals L in every step and is scored once for each of the four
rate-by-D cells. CQE order W0 then W1 and all conservation identities remain
fatal and unscored.

For FIFO attribution, the selected doorbell mapping contributes D to queue or
NIC exactly as above, W1 contention contributes L to queue, and W1 network
service contributes L to NIC:

```text
queue_owner: queue_ps = D + L, nic_ps = L
nic_owner:   queue_ps = L,     nic_ps = D + L
```

Both vectors sum to `D + 2 * L`. FIFO additive totals include both WQE visits
and remain outside this identity.

The FIFO behavioral family has four genuine-risk instances. A bypassed native
queue, incorrect eligibility boundary, capacity error, or selection of
quiescence instead of W1 completion can independently violate a row while the
single-WQE rows still pass.

## Supplement dry run

Before this supplement is committed, run both the registered outer Tier B
command and the literal/schema audit below with `--check-only`:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "$SIMLLM_CORE5_RUN_ROOT/tier_b" \
  --tier-b-only \
  --tier-b-producer "$SIMLLM_RNIC_TIER_B_PRODUCER" \
  --check-only

.venv/bin/python examples/rnic_live_v1/tier_b_review_check.py \
  --out "$SIMLLM_CORE5_RUN_ROOT/tier_b" \
  --producer "$SIMLLM_RNIC_TIER_B_PRODUCER" \
  --check-only
```

The second command is a freeze-time harness. At freeze it may be untracked and
may encode only the literals and validations in this supplement. Neither
command creates a directory, invokes the producer, or emits a measured
artifact in check-only mode. Both print registry confirmations by design.

## Post-specified filesystem portability note

This note was added after the supplementary freeze and changes no schema,
relation, producer argument, chronology, or historical dry run. The one-off
environment-variable spellings above remain frozen text. After loading
`.env.local.sh`, the current portable renderings are:

```bash
.venv/bin/python examples/core5_reduction/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/core5_reduction/tier_b" \
  --tier-b-only \
  --tier-b-producer "${SIMLLM_DATA_ROOT}/core5_reduction/tier_b/build/htsim_rnic_tier_b" \
  --check-only

.venv/bin/python examples/rnic_live_v1/tier_b_review_check.py \
  --out "${SIMLLM_DATA_ROOT}/core5_reduction/tier_b" \
  --producer "${SIMLLM_DATA_ROOT}/core5_reduction/tier_b/build/htsim_rnic_tier_b" \
  --check-only
```

The resolved historical machine-local paths are intentionally omitted.
