# RNIC session records v1 results

## Chronology and provenance

The local expectations-only freeze is commit
`3f1c6d898bc294549c0088f7d34b8ab2a7ff3549`. Its `--check-only` command
passed before implementation and produced no results. The freeze commit
records that the working tree then contained no implementation files for this
task. Implementation started afterward, and the first result-producing run
used the registered command from `expectations.md`. This is a local pre-run
freeze, not a claim of public pre-registration.

That first implementation run reported 53 active-field sensitivity cases and
kept only aggregate result counts. A post-run audit found incomplete field
census coverage and insufficient retained diagnostics. The checker and report
were then expanded to 72 cases, exact before/after hashes, named fatal guards,
all six bypass-input controls and exact frozen fixture cardinality, followed by
a fresh run. No frozen sweep, signed direction or quantitative band changed.
At freeze time, `--check-only` compared the audited HTSim commit to a literal;
the later audit strengthened this by deriving the gitlink from the frozen
SimLLM base commit. The external source and its pinned commit were audited
before the freeze in both cases.

The external-source audit was completed before the freeze against SimLLM base
`6aa3a76` and HTSim commit
`8c3f8b231a6a9311ffc1e7969a003dcba724b50d`. The study's registry check derives
the HTSim gitlink from that SimLLM base commit and rejects a mismatch. Exact
source files and lines are recorded in
[`expectations.md`](expectations.md).

The validated machine-readable result is [`results.json`](results.json), with
SHA-256 `d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6`.

## Scored component evidence

The policy-invariance family passed all 12 pairwise comparisons. Each of the
four `(SQ depth, doorbell service)` hardware cells produced one hash across
`rnic-nn`, `rnic-cn` and `dcqcn`; the four hardware cells produced four
distinct hashes. The four exact cell digests and all 12 policy rows are in
`results.json`.

The active-field sensitivity family passed all four adjacent-axis changes and
all 72 audited census mutations. The census contains 10 scalar, 16 resolved
DMA-binding, 28 PCIe-fabric, 4 path and 14 analytical-profile cases. Every row
retains its field label and exact before/after SHA-256 values. The audit also
proved that path declaration order, disabled DMA payload, disabled-path
payload, inactive non-posted data-credit placeholders, session identity,
policy identity and QP/correlation identity do not change the hash.

The reusable bypass-checker family accepted all four equal output artifact
instances and rejected all four one-byte output mutations, identifying only
the changed artifact each time. This is checker component evidence over a
synthetic immutable fixture. It is not a claim that accepted external-runtime
artifacts have already been compared. Six additional unscored input guards
rejected independent changes to GOAL text, GOAL binary, topology bytes,
profile, seed and canonical semantic baseline parameters.

## Fatal unscored guards

The two-WQE structural fixture reported authority counters `(1, 0, 2, 0)`.
The two-WQE bypass fixture reported `(0, 1, 0, 4)`. Both-authority,
neither-authority, wrong-native and wrong-legacy controls all failed before
audit-counter mutation, and failed calls left counters unchanged.

The native session projected two WQEs one to one into the structural
bookkeeping and compatibility completion rows. Stable keys, native terminal
timestamps, no fabricated structural `rq_id`, strict CSV header/order/LF,
configuration/result identity and exact CQ/WQE reconciliation all passed.
The strict Python reader also accepted an emitted DMA-enabled configuration
and recomputed its canonical hash. These checks are structural and are not
added to any scored denominator.

## Accepted-artifact byte identity

All tracked predecessor artifacts retained their exact pre-run bytes:

| Artifact | Before SHA-256 | After SHA-256 |
|---|---|---|
| `examples/rnic_wq_v1/results.csv` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` |
| `examples/rnic_pcie_v1/results.csv` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` |
| `examples/rnic_device_v1/results.csv` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` |
| `examples/rnic_device_v1/native_tests.csv` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` |

The predecessor harnesses also reported `rnic_wq_v1` at 11 of 11 rows,
`rnic_pcie_v1` at 35 of 35 exact rows, 10 of 10 behavioral families and 18 of
18 predicate instances, and `rnic_device_v1` at six behavioral rows with all
four discovered CTest entries passing.

## Validation gates

- Registered Release study build: 4 of 4 CTest entries passed.
- Independent Debug warnings-as-errors build: 4 of 4 CTest entries passed.
- Embedded and standalone native session-record checks: passed with warnings
  treated as errors.
- Focused Python reader/checker tests: 25 passed.
- `.venv/bin/ruff check .`: all checks passed.
- `.venv/bin/pytest -q`: 472 passed, 3 skipped.

## Genuine-risk fraction

Fractions are reported per scored family and are not combined across evidence
classes.

- Policy invariance: 12 of 12 relations were plausible failures for a
  competent generic run-record implementation because including session or
  transport-policy labels in the hash is a natural serialization mistake.
- Active sensitivity: 72 of 76 relations were plausible failures. The 72
  census cases cross conditional modules, resolved domains, both PCIe credit
  directions, path canonicalization and kind-dependent analytical fields. The
  four explicit SQ-depth and doorbell-axis changes are basic positive controls
  and were not counted as genuine-risk cases.
- Bypass byte identity: 4 of 8 relations were plausible failures. A checker
  can competently omit or misclassify one artifact class, which the four
  one-byte controls expose. Comparing each unchanged synthetic artifact to
  itself is a necessary equality guard but is not genuine-risk evidence.

## Reachability boundary and residual tasks

This study is component scope. It does not claim a live htsim session,
`CompletionEvent`, `StepResult`, TTFT or TPOT change. The residual registry
items are BACK-8 `(Completeness; P1; L)`, HTSIM-9
`(Completeness; P1; L)`, CORE-4 `(Completeness; P1; L)` and CORE-5
`(Completeness; P1; L)`. HTSIM-9 owns concrete port binding and live token
reconciliation. CORE-4 and CORE-5 own the request/execution join and final
metric reachability. The Tier A and Tier B gates remain the frozen authority
for that later evidence.

No new BACK-21 through BACK-23 item was needed. This slice deliberately did
not modify either backend submodule, the frozen `rnic_live_v1` expectations,
the acceptance-harness area, `README.md` or `docs/README_PRO.md`.
