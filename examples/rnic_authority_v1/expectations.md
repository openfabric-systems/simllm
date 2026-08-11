# RNIC authority comparison v1 expectations

## Freeze status

This expectations-only study freezes the residual CORE-21 and BACK-31 gate.
It precedes the authority producer, live checker, positive and negative builds,
all result-producing executions, tests written for this study, and measured
results. The existing `examples/rnic_live_v1` files remain unchanged.

The SimLLM source anchor is
`90ada43070adb3b1e624b6819aff34d8620e8571`. The read-only htsim source anchor
is `4885c647eecdfdf81479d1df052223c016ad086b`.

Before this freeze, a read-only audit agent executed a diagnostic 400 Gbit/s
comparison using two separate control operations. That observation is excluded
from every evidence class and is not quoted here. This freeze deliberately
selects a different literal graph: one control operation with two destination
extents. That graph, its matrix, the link-disabled control, and their producer
have not been executed. The signed band below came from the deployed closed
forms before either graph ran.

## External-source audit before freeze

The signed relation below is a deduction from deployed code and the pinned
native fixture. It is not a band chosen from observed results.

- At the SimLLM anchor, `simllm/core/runtime.py:506-606` makes
  `AtlahsWqeLedger` the timing-neutral bypass WQE authority and serializes two
  same-RNIC WQEs at the configured link rate.
- `simllm/core/runtime.py:801-827` constructs exactly one of the bypass ledger
  and native session. `simllm/core/runtime.py:847-865` selects that object as
  the sole WQE authority for an execution.
- `simllm/core/runtime.py:996-1018` prepares and commits a native transaction
  only after all projections validate, and aborts an uncommitted transaction
  on failure.
- `simllm/core/runtime.py:1504-1591` lowers synchronous control work into a
  semantic send, while `simllm/core/runtime.py:1836-1906` submits it to the
  selected WQE authority and turns the returned native stages into visits.
- `simllm/core/completion.py:256-393` is the deployed
  `ExecutionResult -> StepResult` reduction. Its request state distinguishes a
  first-token TTFT from later exact rational TPOT.
- `simllm/backends/composed_rnic.py:370-524` consumes one immutable native cell
  through the transactional `NativeRnicSession` seam. A two-WQE cell must be
  consumed completely before its counters commit.
- In pinned htsim, `htsim/sim/CMakeLists.txt:6-10` defines the SimLLM RNIC link
  option with default OFF. `htsim/sim/CMakeLists.txt:182-224` adds the SimLLM
  sources, native library link and `htsim_rnic_tier_a` target only when the
  option is ON.
- `htsim/sim/datacenter/CMakeLists.txt:62-91` always builds `htsim_rnic`, but
  defines its SimLLM composition macro only for the ON build.
- `htsim/sim/datacenter/main_rnic.cpp:102-121` emits the structural native
  authority manifest only when the composed runtime is installed.
  `htsim/sim/datacenter/main_rnic.cpp:184-203` installs the composed runtime
  for non-fluid profiles in the ON build and the legacy runtime directly in
  the OFF build.

The existing Tier A source audit already freezes the zero-header,
zero-propagation, no-control-frame capacity-one serializer. The new producer
must independently invoke the pinned Tier A executable and consume its raw
two-WQE cells. It must not emit expected deltas, pass flags or a verdict.

## One literal contended graph

The history seed is an unscored 10,000 ps compute-only first token for request
`core21-decode`, released at 7,000 ps. It creates identical reducer history
without creating a WQE. Two independent reducers consume the same real seed
`ExecutionResult` and `RuntimeReport`; no request tuple is synthesized.

The decision step is one literal `ExecutionGraph` and one literal
`StepRecord`, each passed unchanged to separate bypass and structural
runtimes. The graph is released at 17,000 ps and contains one synchronous 4
KiB control operation from rank 0 to destination ranks 8 and 16. Its two
extents become two WQEs on one source RNIC, are both legal at release, and the
operation is the required completion endpoint. That operation correlates with:

- new request `core21-prefill`, whose metric is first-token TTFT; and
- primed request `core21-decode`, whose metric is second-token TPOT.

The exact graph fields are locked in `expectations.json`. The producer must
serialize the graph and step record once, then reuse those same objects on
both mode paths. In particular, a structural completion may not move the
bypass release time or otherwise change semantic input.

## Closed forms and signed bands

Let `P = 4096 bytes`, `R` be link rate in Gbit/s, `D` be native doorbell
service in ps, and

```text
L(P, R) = P * 8 * 1000 / R ps.
```

Two same-RNIC WQEs give:

```text
J_bypass     = 2L
J_structural = D + 2L
structural - bypass = D.
```

The exact cells are:

| R (Gbit/s) | D (ps) | bypass JCT (ps) | structural JCT (ps) |
|---:|---:|---:|---:|
| 200 | 0 | 327,680 | 327,680 |
| 200 | 1,000 | 327,680 | 328,680 |
| 400 | 0 | 163,840 | 163,840 |
| 400 | 1,000 | 163,840 | 164,840 |

For `D = 1,000 ps`, the signed structural-minus-bypass direction is positive
and the inclusive band is exactly `[+1,000, +1,000] ps` for each of fixed-graph
JCT, `core21-prefill` TTFT, and `core21-decode` TPOT. This is six scored
instances, three metrics at each of two rates.

For both authority modes and both D values, the 200-minus-400 Gbit/s band is
exactly `[163,840, 163,840] ps` for those same three metrics. This is twelve
scored instances. The rate family varies a second model parameter and checks
that only `2L` changes. The signed and rate families share raw cells and must
not be summed as independent risks.

At `D = 0`, authority identity is a configuration-forced zero relation. It is
fatal and unscored.

## Entailment analysis frozen before execution

Every scored instance can fail after its raw observation is reached. A dropped
native doorbell can make the signed delta zero; double-counted serialization
can move it outside the exact band; a scalar-derived bypass result can disagree
with the deployed reducer; and a stale or synthetic request summary can move
TTFT or TPOT independently of graph JCT.

The checker must parse only raw shape and types before evaluating both scored
families directly from the observed `StepResult` and request metrics. It must
evaluate them before any per-mode exact-oracle check that pins the same JCT,
TTFT or TPOT quantity. The Tier A checker may validate its native source cell
before the live run because that does not pin the later bypass result, runtime
adapter, `ExecutionResult` boundary or deployed reduction. The producer may
enforce transaction safety and atomic publication, but it may not evaluate a
scored relation. With this order, neither scored family is entailed by an
earlier fatal check.

## Authority, transaction and bypass guards

The bypass row must report `AtlahsWqeLedger`, retain a non-null ledger, use no
native session, and traverse `CoarseDeviceRuntime.execute` followed by
`CompletionReducer.reduce`. Its WQE count, completion events, `ExecutionResult`,
`RuntimeReport`, `StepResult` and request summary are raw observations. No
scalar JCT may be expanded into request tuples or summaries.

The structural row must report `SimllmNativeRnicSession`, have no bypass
ledger, commit exactly one two-WQE transaction, and expose the native
doorbell and network visits.

Before the successful structural run in every cell, the registered live
harness submits a one-WQE graph to that two-WQE session. The expected
transaction failure consumes no native observation, changes no session
counter, installs no runtime report or runtime execution ID, and is followed
by one successful retry of the fixed graph. These sequence and atomicity
checks are fatal and unscored.

The accepted bypass bundle is materialized from the real graph, completion,
`StepResult` and request summary bytes. It is rerun after the isolated negative
build and compared with `BypassArtifacts` and
`compare_bypass_artifacts`. This is a change-set guard, not a scored family.
Positive composed and bypass-capable binary hashes and accepted artifact
hashes are likewise checked before and after the negative run, and remain
fatal and unscored.

## Link-disabled negative control

Both build trees come from the same pinned, clean htsim source and are
disjoint. Both force `HTSIM_CREATE_SOURCE_SYMLINKS=OFF` so the read-only source
tree is untouched.

The positive tree sets `HTSIM_ENABLE_SIMLLM_RNIC=ON`, points
`SIMLLM_REPOSITORY_ROOT` at this checkout, and builds `htsim_rnic_tier_a`,
`htsim_rnic`, and `txt2bin`.

The negative tree sets `HTSIM_ENABLE_SIMLLM_RNIC=OFF`, still points
`SIMLLM_REPOSITORY_ROOT` at this checkout, and builds the unconditional
`htsim_rnic` and `txt2bin` targets. Its fresh cache must say OFF, its main
binary must exist, and no single- or multi-configuration candidate for
`htsim_rnic_tier_a` may exist.

The same registered producer is then invoked with the negative main binary on
a non-fluid `rnic-nn` GOAL preflight. The executable must finish normally but
lack the ON-only `hardware_mode=structural` and
`wqe_authority=simllm-native-rnic-session` manifest. The producer exits 2 with
`composed preflight lacks structural native authority` before resolving or
running a Tier A producer and before publishing raw observations. The same
live checker exits 2 with `authority observations do not exist`. Raw, result
and temporary publication paths all remain absent. This executable run, not a
mutated observation, is the BACK-31 negative control. It is fatal and
unscored.

## Evidence classes

The two raw behavioral relation families report genuine-risk fractions of
six and twelve instances. Exact graph identity, reducer use, sole authority,
transaction atomicity, D-zero identity, schema, conservation, native source
validation, build-cache state, missing target, rejected publication and
artifact hashes are fatal-unscored evidence. Native build completion is a
separate executable evidence class. Counts from these classes are never added.

## Registered commands and pre-freeze dry run

The outer command is:

```bash
.venv/bin/python examples/rnic_authority_v1/run_study.py \
  --out "${SIMLLM_RNIC_AUTHORITY_RUN_ROOT:?configure SIMLLM_RNIC_AUTHORITY_RUN_ROOT}" \
  --htsim-source "${SIMLLM_HTSIM_PIN_ROOT:?configure SIMLLM_HTSIM_PIN_ROOT}"
```

The outer runner configures and builds both trees, then invokes these exact
positive producer and checker interfaces:

```text
.venv/bin/python examples/rnic_authority_v1/produce_observations.py
--expectations <repo>/examples/rnic_authority_v1/expectations.json
--observations <out>/positive/raw_observations.json
--tier-a-producer <out>/positive/build/<configuration>/htsim_rnic_tier_a
--htsim-rnic <out>/positive/build/<configuration>/datacenter/htsim_rnic
--txt2bin <out>/positive/build/<configuration>/txt2bin

.venv/bin/python examples/rnic_authority_v1/check_results.py
--expectations <repo>/examples/rnic_authority_v1/expectations.json
--observations <out>/positive/raw_observations.json
--negative-evidence <out>/negative/evidence.json
--results <out>/positive/results.json
```

The negative producer and checker use the identical interfaces with negative
build and publication paths. They are expected to exit 2 as specified above.

Before this freeze, the outer, producer and checker vectors for both positive
and negative cases must each be dry-run with `--check-only`. At freeze the
check-only harness may be untracked and may encode only the literals and
validations in these two expectation files. Check-only validates commits,
matrix, arithmetic, graph identity, argument vectors, disjoint build paths and
fresh output shape. It does not configure CMake, invoke a native executable,
create an output directory, or emit a measured artifact.
