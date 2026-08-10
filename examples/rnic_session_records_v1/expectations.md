# RNIC session records v1 expectations

## Freeze status and scope

This file is the expectations-only record for the BACK-8 session-record
remainder. It precedes implementation of the session configuration record,
result record, hardware-configuration hash, authority projection and reusable
bypass-artifact checker. It also precedes every result-producing run of this
study. The result report must cite this freeze commit and record the actual
chronology.

This is component scope. It does not claim a linked htsim session, a
`CompletionEvent`, a `StepResult`, TTFT or TPOT reachability result. HTSIM-9 is
the specific successor that binds the native port and writes these records
from a directly invoked simulator. CORE-4 and CORE-5 are the specific
successors that carry the selected authority through `CompletionEvent`,
`StepResult` and TTFT or TPOT. The Tier A and Tier B live gates remain frozen
in `examples/rnic_live_v1/expectations.md`.

## External-source audit before freeze

The audit used the SimLLM base commit `6aa3a76` and pinned HTSim commit
`8c3f8b231a6a9311ffc1e7969a003dcba724b50d`. No source was updated or run to
obtain an outcome before this freeze.

- SimLLM `simllm/backends/rnic/include/simllm/rnic/rnic_device.h:15-60`
  defines the independently versioned device, identity, work-queue, QPC, DMA
  and network configuration surfaces. The active device exposes its retained
  config and resolved PCIe binding at lines 143-149.
- SimLLM `simllm/backends/rnic/include/simllm/rnic/work_queue.h:67-107`
  defines the queue capacities, scalar services and PCIe binding fields.
  SimLLM `simllm/backends/rnic/include/simllm/rnic/pcie_fabric.h:77-160`
  defines every PCIe latency, credit, path and analytical-profile field.
- The pinned HTSim
  `htsim/sim/atlahs_wqe.h:50-80,94-100` defines the timing-neutral bypass
  ledger record and its post/complete mutation surface. Its implementation at
  `htsim/sim/atlahs_wqe.cpp:104-209` posts and dispatches at one timestamp,
  then posts and consumes the CQ at the completion timestamp.
- The pinned HTSim
  `htsim/sim/datacenter/main_rnic.cpp:38-66` sorts by flow ID and writes the
  accepted completion CSV header and column order. This is the external
  referent for the reusable byte checker and compatibility-row renderer.
- `examples/rnic_live_v1/expectations.md:83-100` freezes the four behavioral
  artifacts that bypass must compare byte for byte and excludes only the new
  configuration and run audit record from that comparison.

The hardware-hash relations below are SimLLM design requirements, not claims
copied from a vendor specification. The bypass row shape and byte order are
the only expectations that mirror an external runtime, and they cite its
pinned source above.

## Effective hardware configuration

The hash input is a canonical, schema-tagged projection of effective hardware
configuration. It includes every active capacity, service time, module-enable
choice, resolved PCIe ordering domain, PCIe transfer shape, credit limit,
fixed latency, analytical seed, path and active analytical-profile parameter.
It excludes transport-policy identity, session and workload identity, QP and
policy-context correlation identity, and fields belonging to a disabled
module. An inactive field remains validated by its owning config but cannot
change the effective-hardware bytes or hash.

The structural configuration record carries the canonical effective object
and its lowercase SHA-256 digest. A bypass configuration record carries no
effective native hardware object and no hardware hash. Both forms carry their
own schema, session ID, hardware mode, authority and transport-policy label.
Changing display order, object address, path spelling, build ID or wall time
cannot affect the hash.

## Decision-relevant hash relations

Use scalar structural devices with QPC and the external-network seam enabled,
DMA disabled, and otherwise fixed accepted values. Sweep SQ depth `Q` in
`{32, 64}`, scalar doorbell service `D` in `{0, 1000}` ps, and policy label in
`{rnic-nn, rnic-cn, dcqcn}`. This gives 12 run-configuration rows and four
hardware cells.

1. Within each `(Q, D)` cell, all three policy labels must produce exactly one
   hardware hash. The quantitative band is zero unequal policy pairs out of
   12 pairwise comparisons.
2. The four `(Q, D)` cells must produce four distinct hashes. Changing only Q
   or only D must change the digest in all four adjacent-axis comparisons.
   The quantitative band is four changes out of four and zero collisions.
3. A maintained sensitivity census constructs accepted scalar and DMA-on
   devices and changes each audited effective field, or the smallest valid
   tuple required to activate one analytical profile. Every mutation must
   change the canonical bytes and digest. The quantitative band is 100
   percent sensitivity, with the field label and before/after digest reported
   for every case. The census is checked against the configuration surfaces
   cited in the source audit, so omitting a newly effective field is a failure,
   not a smaller denominator.

These relations decide whether full-RNIC policy comparisons are valid. If
relation 1 fails, policy identity has leaked into hardware identity and
`rnic-nn`, `rnic-cn` and DCQCN rows must not be normalized as one hardware
comparison. If relation 2 or 3 fails, materially different devices can be
misclassified as equal and the comparison methodology must be blocked until
the hash projection is repaired. The design decision that would change is
therefore explicit: failed relations replace cross-policy comparison with a
hard configuration error, not a warning or a per-policy exception.

Policy-label permutation, session-ID changes, QP/correlation-ID changes and
mutations confined to disabled DMA payloads must leave the digest unchanged.
These exclusions and inactive-field checks are fatal structural guards and
are unscored.

## Authority and result records

The result record has schema `simllm-rnic-session-result-v1` and repeats the
session ID, mode, selected authority, policy and nullable hardware hash from
the configuration record. It contains these exact mode-exclusivity counters:

- `native_session_constructed`;
- `legacy_ledger_constructed`;
- `native_posts`; and
- `legacy_mutations`.

For a two-WQE structural fixture the tuple in that order is `(1, 0, 2, 0)`.
For a two-WQE bypass fixture with one ledger post and one completion mutation
per WQE it is `(0, 1, 0, 4)`. Selecting both authorities, neither authority,
posting through the inactive authority, or recording a legacy mutation in
structural mode must fail before lifecycle, audit-counter or caller-time
mutation. These are fatal unscored invariants.

The native device remains the only lifecycle authority. The result builder
reads immutable `RnicDevice` WQE records plus returned CQ entries and checks
identity, count, status and timestamp agreement. It does not advance a WQE,
choose between timestamps or maintain a shadow state machine. Its stable send
key is `(session_id, source endpoint, send, sq_id, sq_post_sequence)`. A send
projection names its local SQ and send CQ and leaves receive-WQ identity
absent. Projection loss, duplication, an unknown CQ entry, timestamp drift or
configuration-hash disagreement is fatal and unscored.

The schema-tagged bookkeeping projection retains the native timeline and the
stable key. The compatibility completion row uses native post time as the
legacy flow start boundary and native network outcome as the legacy flow
completion boundary; it never labels acceptance as first-packet issue. CQ
sequence fields come only from the returned native CQ entry. Rows are sorted
by flow ID and render with the pinned header and line endings. Exact projection
call order is a structural guard, not scored behavioral evidence.

## Reusable bypass identity relation

The checker accepts two immutable artifact bundles and compares these four
behavioral byte strings independently:

1. completion CSV;
2. canonical parsed completion rows plus final JCT record;
3. `StepResult` tuple sequence; and
4. replay TTFT and TPOT summary.

The GOAL text and binary are separate input-identity guards. Run-record bytes,
paths, build IDs, elapsed wall time and command spelling are not accepted as
behavioral artifacts. An equal candidate must pass all four behavioral
instances. Four negative controls each flip one byte in exactly one artifact;
the checker must reject all four and identify only the changed artifact. The
quantitative band is 4 of 4 equal instances accepted and 4 of 4 one-byte
mutations rejected. This relation has the pinned HTSim writer and the frozen
Tier A contract as author-independent referents, so it is scored.

The bypass result record is checked separately and must say
`hardware_mode=bypass`, `authority=AtlahsWqeLedger`, carry the bypass counter
tuple and omit native hardware/hash fields. Adding that audit record must not
change any of the four compared byte strings or add a default stdout line.

## Evidence classes and acceptance

Scored behavioral families are reported separately:

- policy-invariant hardware identity over the four hardware cells;
- active-field hardware-hash sensitivity over the two-axis grid and audited
  mutation census; and
- bypass byte identity and one-byte negative controls over four artifact
  classes.

Run configurations, exact digest values, reader round trips and compatibility
rows are reported but not added across evidence classes. Authority exclusion,
inactive fields, schema/version rejection, hash recomputation, projection
reconciliation, input identity and baseline artifact equality are fatal
unscored guards. Native executables and Python tests are component evidence.

The accepted `rnic_wq_v1`, `rnic_pcie_v1` and `rnic_device_v1` tracked result
bytes must remain identical before and after this study. Their existing
behavioral counts are not added to this study's denominator.

## Registered commands and dry run

The historical dry run used the same executable basename, script, options and
pinned inputs; resolved machine-local paths are intentionally omitted. The
following blocks are portable post-freeze renderings, not verbatim
transcripts. Source the local configuration first. The result-producing
rendering is:

```bash
.venv/bin/python examples/rnic_session_records_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/rnic_session_records_v1"
```

Before this freeze, the historical resolved form of the same registered
command was executed in its non-result-producing mode. Its portable rendering
is:

```bash
.venv/bin/python examples/rnic_session_records_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/rnic_session_records_v1" \
  --check-only
```

`--check-only` parses the complete CLI, validates the frozen matrix, artifact
inventory, output-volume rule and external-source pins, and neither creates
the output directory nor imports the not-yet-implemented record API.
