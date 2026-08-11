# BACK-28 effective-hardware reader results

## Chronology and provenance

Expectations-only commit
`f50ed49581f072eb50d4c3e69445217cb2877c36` precedes every v2/v3 reader
implementation edit and every result-producing parity run. Its registered
`--check-only` command reconstructed the native-emitted bases, matched their
frozen hashes, validated the 100-entry mutation registry and created no output
directory. The commit message records the precise freeze-time working tree,
including the untracked dry-run harness and its frozen literals. This is a
local pre-run freeze, not a claim of public pre-registration.

The first result-producing run used the uncommitted post-freeze implementation
and passed all 100 rejection pairs. Only regression tests were added after
observing that result; the reader, native probe, harness, mutations, expected
directions and exact bands were not changed. Implementation commit
`34202e049fc365cdb26654067c07f5309e0b132c` then froze the executable behavior.
The registered command was rerun against that exact commit and again passed.

The pre-freeze external-source audit used SimLLM commit
`9923c9f0add6b6f23a0019382962931e1792bc47`. The native reader files, exact
line ranges, four emitted fixture hashes and audit method are recorded in
[expectations.md](expectations.md). The formal external summary is
`$SIMLLM_WAVE5_RUN_ROOT/back28/summary.json`, SHA-256
`4961cafc457227845446f089d273c4ef914bf546186035a72370eee596375f18`.

## Native and Python rejection parity

Every frozen mutation was rendered as canonical ASCII JSON. N001 through N098
received a recomputed object hash, so each reached its intended semantic
reader check. N099 and N100 kept the valid v2 object and changed only the
supplied hash. The native observation came from a thin CLI over the existing
`validateRnicSessionConfigRecord`; the study did not create a parallel native
comparator.

For each mutation N, the frozen relation was exact:

```text
A_native(N) = 0
A_python(N) = 0
A_python(N) - A_native(N) = 0
```

The formal run produced:

| Rejection area | Fixture coverage | Passed | Total |
|---|---|---:|---:|
| Root, modules and host-memory enablement | v2, v3 host | 9 | 9 |
| DMA fabric, credits, paths and binding | v2 | 27 | 27 |
| Submission shape, identities and sole CQ consumer | v3 host, proxy, GPU | 15 | 15 |
| Host-memory registry, allocation and ownership | v2 | 26 | 26 |
| Page geometry, allocation binding and descriptor ownership | v2, v3 host, proxy, GPU | 17 | 17 |
| Effective WQ and canonical hash | v2 | 6 | 6 |
| **Exact rejection family** | **75 v2, 9 v3 host, 9 v3 proxy, 7 v3 GPU** | **100** | **100** |

The native reader rejected all 100 objects and the Python reader independently
rejected all 100. The accepted/rejected bit difference was exactly zero for
every pair.

## Entailment and genuine risk

All 100 native and Python outcomes, diagnostics and exit statuses were written
to `raw_observations` before the harness evaluated any scored predicate. The
valid-base controls do not constrain a mutation outcome, and no fatal check
first asserts either reader's answer for a mutated object. A run can therefore
reach any scored instance with either implementation accepting it, which
would fail that instance. The relation is not entailed.

The genuine-risk fraction is 100 of 100, or 100 percent. This is one scored
evidence class and is not combined with acceptance, freezing, hash or native
test counts. An early blanket rejection could satisfy the mutation side, but
would fail the separate valid-base controls; it cannot claim successful v2/v3
ingestion.

## Fatal unscored controls

Both readers accepted all four unmodified native-emitted objects and the
Python projection retained every field and array value:

| Fixture | SHA-256 | Native | Python | Recursively frozen |
|---|---|---|---|---|
| v2 | `4a94c6ec23c0af9a18524d33dbb3127dd1d4cde4dcfced7e972fdb1dda5dfebf` | PASS | PASS | PASS |
| v3 host CPU driver | `4ffabebbb9c6f5aace706f241af030b95c286c607a6e4f9e39a146e5065dfa17` | PASS | PASS | PASS |
| v3 CPU proxy | `a9cc2fa1df269f75d6c8d48ba27bff81dd5c65f086ea9c4fe3f3eaadb264cee3` | PASS | PASS | PASS |
| v3 GPU initiated | `cd4da0c3635006ce3a02f1a19b1ecd1700ca92fa6758ee87a77bccca6f15c4ad` | PASS | PASS | PASS |

Attempted mutation of each frozen root mapping raised `TypeError`; attempted
append to each nested allocation tuple raised `AttributeError`. These are
structural controls and do not increase the scored denominator.

The v1 and bypass off paths retained their frozen values:

| Off path | Effective SHA-256 | Complete config SHA-256 | Parsed identity |
|---|---|---|---|
| v1 structural | `a9732c130d2ed0075668c7ee1f77c742492ca059f1c50b1ca35c078799deaa9c` | `69f20997fede3a9a00b386a5a0412f948dba5bc1b6eb0c7e93d6d6dd85e01d0c` | structural, `SimllmNativeRnicSession` |
| bypass | null | `c750be3ba90023987478e6ecd111ee70ad90c02f669470e547ef252e047afc2b` | bypass, `AtlahsWqeLedger`, null hardware/hash |

The frozen SHA values are change-set guards and are unscored. Parsed-record
identity and the v1 projected-object equality are fatal off-path controls.

## Closure scope

BACK-28 registered three acceptance clauses:

- "ingest and freeze the native strict effective-hardware v2 and v3 objects"
  is demonstrated by four native-emitted acceptance controls, exact field and
  array projection equality, and failed nested mapping and array mutations.
- "validate allocation and page geometry, submission-shape endpoint
  agreement, descriptor ownership, sole CQ consumer and canonical hashes with
  the same rejection set as the native reader" is demonstrated by the 100 of
  100 exact native/Python rejection pairs. The frozen corpus cites every
  native branch by source line and includes each named mechanism.
- "v1 structural and bypass ingestion remains the exact off path" is
  demonstrated by the two complete config hashes, v1 effective-object hash,
  exact projected v1 object, and unchanged mode, authority and nullability.

No acceptance clause remains undemonstrated, so no residual task ID is needed.
This is component evidence. It does not claim a live simulator path,
`CompletionEvent`, `StepResult`, TTFT or TPOT change. CORE-21 remains the
named live-chain successor.

## Validation gates

The registered Release build enabled tools and tests and treated warnings as
errors. CTest reported 6 of 6 passed, including the negative-service CLI test.
The focused Python reader and study suite reported 131 passed. The complete
repository gates on the final implementation tree reported:

```text
.venv/bin/ruff check .
All checks passed!

.venv/bin/pytest -q
793 passed, 5 skipped in 15.73s
```

Python 3.10 compiled the reader, harness and new tests, then completed the
registered check-only path without creating its output directory. No test
depends on a path under `third_party/`; the native authority is the tracked
`simllm/backends/rnic` source. No backend submodule, `README.md`,
`docs/README_PRO.md` or `docs/architecture.md` was changed.
