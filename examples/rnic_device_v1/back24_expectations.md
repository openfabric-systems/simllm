# RNIC device BACK-24 expectations

This expectations-only change precedes the BACK-24 implementation and every
corrective run. It contains no measured value, generated row or implementation.
The accepted RNIC device expectations and tracked artifacts remain unchanged.

## Defect boundary and decision relation

`RnicDevice` is the caller-clock authority around its composed `WorkQueue`.
An invalid external terminal must be rejected before the device commits that
terminal's timestamp. The work queue already validates terminal identity and
plans ordered retirement before its mutation boundary. BACK-24 therefore first
tests the narrow design in which the device validates monotonic time, delegates
the complete terminal transaction, and commits caller time only after success.

The decision-relevant relation is post-rejection clock continuity. If the
valid matched continuation at `20 ps` fails or differs from its paired control
after any invalid future terminal, the narrow ordering fix is inadequate. The
design must then change to an explicit WorkQueue terminal preview and commit
interface so the device can approve the complete retirement plan before either
authority mutates.

## Frozen matrix and signed quantitative relation

Each fixture establishes accepted caller time `10 ps`. It then supplies one
syntactically valid delivered terminal at either `110 ps` or `1010 ps`, giving
future deltas of `+100 ps` and `+1000 ps`. The invalid-terminal axis is:

1. an unknown token, which must throw exactly `std::logic_error` with
   `unknown or duplicate RNIC network token`;
2. a duplicate of a terminal already accepted once, with the same exact type
   and message; and
3. a live token paired with another real WQE, which must throw exactly
   `std::logic_error` with `RNIC network token/WQE mismatch`.

The scored clock-continuity family has six paired-control instances. After the
rejection, a valid matched terminal is supplied at `20 ps`, followed by device
progress at `20 ps`. The paired control receives the same setup and valid
terminal without the invalid injection. The continuation must succeed and its
native terminal timestamp, completion-visible timestamp and complete public
device snapshot must equal the control exactly. Every signed timestamp delta
is exactly `0 ps`, with quantitative band `[0 ps, 0 ps]`, for 6 of 6 matrix
cells. Any positive or negative delta fails the family. A competent
implementation can fail this relation by validating the WorkQueue correctly
while still committing the enclosing device clock too early.

## Fatal transactional guards

Immediately before and after each rejected terminal, the study compares one
canonical snapshot byte for byte. It contains every public WQE record field,
all counters, all evidence events, the complete fake-port history and inflight
ledger, and these physical-state observations: next event time, pending-work
flag, fatal flag, occupied SQ entries, CQ depth and unpublished WQE count. The
snapshot also contains PCIe generation and complete accounting if the fixture
uses DMA. `progress(10)` is a no-op both immediately before and after the
rejection. Exception type and message, immediate snapshot equality, both
no-op probes and invariant validation are fatal unscored guards. They do not
increase the six-instance scored denominator.

The three terminal identities and two future timestamps are independent axes.
An event is constructed or consumed at the external port before the
pre-rejection snapshot, so the comparison isolates device rejection and never
attributes legitimate port-side event delivery to the device.

Exact internal call order is not behavioral evidence. The existing rejection
identities are frozen structural referents at SimLLM commit `637c6a4` in
`simllm/backends/rnic/src/work_queue.cpp:579-585`. The existing transactional
device pattern is at
`simllm/backends/rnic/src/rnic_device.cpp:447-461` in the same commit.

## Portability and accepted baselines

The RNIC device study runner must have no machine-specific build path. An
explicit `SIMLLM_RNIC_DEVICE_BUILD_DIR` value is used exactly. Without that
override, the default is outside the repository and is deterministically keyed
by the resolved `examples/rnic_device_v1` output location, so two worktrees do
not share a CMake cache. Two distinct synthetic output locations must produce
distinct defaults and repeated resolution of one location must be stable.

Running the existing device study from a non-repository working directory with
no `--build-dir` must preserve these tracked artifacts byte for byte:

- `results.csv`, SHA-256
  `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934`;
- `native_tests.csv`, SHA-256
  `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d`.

Artifact identity and build-path checks are fatal unscored guards.

## Evidence boundary and external-source audit

This is a component study. It does not claim a live htsim session or a change
to `CompletionEvent`, `StepResult`, TTFT or TPOT. HTSIM-9 is the successor that
makes native terminal delivery live-reachable. CORE-4 and CORE-5 are the
successors that join the resulting completion into the reported metric chain.

No frozen relation mirrors vLLM, NCCL, hardware specifications or another
external runtime, so there is no external-source expectation to audit. The
accepted `rnic_wq_v1`, `rnic_pcie_v1` and `rnic_device_v1` artifacts and their
existing behavioral counts remain regression evidence and are not added to
the BACK-24 denominator.

## Registered command and dry run

The historical dry run used the same executable basename, script, options and
pinned inputs; resolved machine-local paths are intentionally omitted. The
following blocks are portable post-freeze renderings, not verbatim
transcripts. Source the local configuration first. The result-producing
rendering is:

```bash
.venv/bin/python examples/rnic_device_v1/run_back24_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/rnic_session_records_v1/back24"
```

Before this freeze, the historical resolved form of the same command was
executed in its non-result-producing mode. Its portable rendering is:

```bash
.venv/bin/python examples/rnic_device_v1/run_back24_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/rnic_session_records_v1/back24" \
  --check-only
```

`--check-only` parses the complete CLI and validates the six-cell registry,
exact exception identities, snapshot inventory, output-volume rule and frozen
artifact digests. It creates no directory, imports no not-yet-implemented
native behavior and produces no result artifact.
