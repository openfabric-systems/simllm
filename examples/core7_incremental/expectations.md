# CORE-7 incremental bookkeeping validation expectations

Written and frozen before the CORE-7 implementation and before any run of
this study. The public ledger schema and accepted fact language stay unchanged.
The full-ledger validator remains the reference for immutable ledgers and wire
loads.

## System under test

`RequestBookkeeper.append` and `RequestBookkeeper.extend` will retain
incremental auxiliary state for the three cross-entry relations enforced by
the current validator:

- created objects indexed by portable object ID;
- the latest completion-event timestamp for each subject object;
- the set of WQEs that have reached their terminal `COMPLETED` event.

The auxiliary state is private. `BookkeepingLedger`, its v1 wire form, public
method signatures and query results must not change. Initial ledgers, snapshots
passed explicitly to `validate_bookkeeping_ledger`, and both wire directions
continue to receive a complete reference validation.

## E1: seeded behavioral equivalence

The deterministic seeds are `{7001, 7002, 7003, 7004, 7005, 7006}`. Each seed
generates valid object, stage and completion-event streams with sequence
lengths drawn from `{8, 32, 128}`. The streams include causal request objects,
batched execution-operation scopes that may narrow at their descendants,
reusable queue references, transport-free and physical WQEs, subjectless
events, subject timestamp progressions and terminal WQE completions.

For the single-fact family, each candidate is applied to two independently
maintained ledgers:

1. The reference path forms the complete candidate `BookkeepingLedger` and
   calls `validate_bookkeeping_ledger`.
2. The incremental path calls `RequestBookkeeper.append`.

For every candidate and every prefix, both paths must either accept or reject.
If they reject, the raised exception classes must be identical and the
incremental ledger must remain equal to the last accepted reference ledger. If
they accept, the returned entry and complete incremental snapshot must equal
the reference candidate exactly.

For the atomic-batch family, the same comparison is repeated with batch widths
drawn from `{0, 1, 2, 7, 32}`. The reference validates the whole candidate in
one call and the incremental path uses `extend`. If any fact fails, both paths
must reject with the same exception class and `extend` must retain its entire
pre-call state. Accepted return tuples and final snapshots must be exactly
equal.

Each seed supplies both an unmodified valid stream and invalid streams with a
mutation inserted at a seeded random position. Across the fixed seed set, the
mutation catalog covers every existing full-validator rule family:

- unsupported fact types and invalid fact timestamps;
- invalid enum, scope, correlation, reference, metadata and scalar types,
  including Boolean rejection for integer-only fields;
- duplicate object IDs, self-parenting, unknown or wrong-typed parents,
  child-before-parent time, causal-parent disagreement and request
  introduction;
- WQE queue cardinality, transport cardinality and `transport_kind` rules;
- stage enum, scope, object-reference and stage-before-object rules;
- completion field types, unknown subjects, subject scope mismatch,
  completion-before-creation and decreasing subject timestamps;
- WQE completion-queue identity, duplicate completion and strict terminality.

Acceptance is exact. There is no tolerated decision, exception-class, entry,
sequence or snapshot difference.

## E2: append scaling

Two valid fact mixes are measured so ledger size is not the only varied
parameter:

- `stage`: independent request-stage facts with no cross-object lookup;
- `wqe`: reusable queue setup followed by repeated operation, NCCL, WQE and
  subject-event chains, including one terminal completion per WQE.

For the incremental implementation, total time for N individual `append`
calls is measured at `N in {1000, 4000, 16000}`. The first complete sweep is a
warmup. Five subsequent sweeps are recorded and the median for each cell is
used for acceptance. Within each fact mix, both adjacent quadruplings must
satisfy:

```text
T_incremental(4N) / T_incremental(N) <= 6
```

The direction claim is near-linear total work, not an exact wall-clock model.
No exact duration or cross-machine comparison is registered.

The former behavior is reproduced by constructing and fully validating the
entire candidate ledger after every single fact. It is measured at
`N in {1000, 2000, 4000}` with one warmup sweep and three recorded sweeps. For
each fact mix, the endpoint relation must satisfy:

```text
T_reference(4000) / T_reference(1000) >= 8
```

The ideal quadratic ratio is 16. The looser lower bound registers only the
expected superlinear direction in the presence of interpreter and host noise.
The comparison demonstrates removal of the old trend; it does not claim a
stable speedup factor.

The frozen requirement is that raw per-repetition nanosecond measurements
remain outside Git in a machine-local external directory. The resolved
historical target is intentionally omitted. As a post-freeze portability
convention, new runs default to
`${SIMLLM_DATA_ROOT}/core7_incremental/`.

## Evidence classes and construction disclosures

E1 is one behavioral relation family with seeded parameterized instances. E2
is a separate complexity relation family with fact mix and ledger size as its
parameters. Their instance counts are reported separately and are never added
into one headline total.

The following checks are fatal structural guards but unscored:

- generated valid source streams pass the reference validator before mutation;
- both timing paths receive the same immutable fact tuples;
- appended sequence numbers are contiguous plain integers because the public
  API assigns them, while explicit malformed initial ledgers retain the
  reference validator's Boolean-sequence rejection test;
- rejection leaves the prior snapshot unchanged;
- benchmark processes finish with the expected ledger length and no validation
  error.

These construction-forced properties do not increase either behavioral pass
denominator. Any failure still invalidates the study.
