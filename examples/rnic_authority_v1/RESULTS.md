# RNIC authority comparison v1 results

## Status and chronology

The registered CORE-21 and BACK-31 study passed.

The expectation files were frozen in commit
`1871cfcc2e343205f9e78693c0aa87b7b34942e9` before the producer, checker,
runner, tests, frozen-graph execution, or link-disabled build existed. The
runtime and external-source audit anchor in that freeze is SimLLM
`90ada43070adb3b1e624b6819aff34d8620e8571` and htsim
`4885c647eecdfdf81479d1df052223c016ad086b`. The implementation was then
committed as `3f293cbc4f7a4d0354b7e066027a3063a5f474e4`, and the result-producing
command ran from that clean implementation commit. The raw record therefore
names `3f293cbc4f7a4d0354b7e066027a3063a5f474e4` as its SimLLM source while the
checker proves that it descends from the frozen audit anchor.

The pre-freeze two-operation diagnostic disclosed in
[expectations.md](expectations.md) remains excluded. The scored run used only
the frozen one-operation, two-extent graph.

The frozen expectation JSON has SHA-256
`6fd38b8edd046d9e741bfa741cf421ac1f8489a1cd3c0d735787ec603142a1aa`.
The external run artifacts have these hashes:

| Artifact | SHA-256 |
|---|---|
| positive raw observations | `ceada27fd21053c7ac7087e8c246007df57b31d6e0298cde0e7b825af35fcd63` |
| native Tier A observations | `37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a` |
| positive results | `548e234bfbce09341fc5963979c3ba5c60375b0f56fd79d7cb4081d734135227` |
| link-disabled evidence | `5bc8dec66c326d03e9a29d2b861e1759ec8b961c929f8146ab0d2a10750e6775` |

## Same-graph metric result

Every row passed the same fixed `ExecutionGraph` and `StepRecord` to the
timing-neutral `AtlahsWqeLedger` path and the composed
`SimllmNativeRnicSession` path. Both paths produced an `ExecutionResult`, a
live runtime report, and a deployed `CompletionReducer` result. JCT below is
`StepResult.step_latency_ps`; TTFT is the `core21-prefill` request metric; TPOT
is the exact rational metric for the primed `core21-decode` request.

| Rate (Gbit/s) | D (ps) | Mode | JCT (ps) | TTFT (ps) | TPOT (ps) |
|---:|---:|---|---:|---:|---:|
| 200 | 0 | bypass | 327,680 | 327,680 | 327,680 |
| 200 | 0 | structural | 327,680 | 327,680 | 327,680 |
| 200 | 1,000 | bypass | 327,680 | 327,680 | 327,680 |
| 200 | 1,000 | structural | 328,680 | 328,680 | 328,680 |
| 400 | 0 | bypass | 163,840 | 163,840 | 163,840 |
| 400 | 0 | structural | 163,840 | 163,840 | 163,840 |
| 400 | 1,000 | bypass | 163,840 | 163,840 | 163,840 |
| 400 | 1,000 | structural | 164,840 | 164,840 | 164,840 |

The decision-relevant signed family passed 6/6 genuine-risk instances. At
both rates, structural minus bypass was exactly `+1,000 ps` for JCT, TTFT and
TPOT, matching the frozen positive `[+1,000, +1,000] ps` band.

The inverse-rate family passed 12/12 genuine-risk instances. For bypass and
structural mode at both D values, 200 minus 400 Gbit/s was exactly
`163,840 ps` for JCT, TTFT and TPOT, matching the frozen
`[163,840, 163,840] ps` band.

These families share raw cells and are reported separately. They are not
summed into an 18-instance independence claim.

## Entailment analysis

The checker evaluation order was:

1. raw shape and scalar types;
2. signed authority delta;
3. inverse-rate delta;
4. fatal exact oracles and live invariants; and
5. the link-disabled negative control.

No exact per-mode JCT, TTFT or TPOT oracle ran before either scored family.
Each scored instance could therefore fail after reaching the checker. For
example, a lost doorbell would make the signed delta zero, a serialization
error would move the rate delta, and an incorrect live reduction could move
TTFT or TPOT without changing the native source fixture. Focused mutants
confirmed that a one-picosecond TTFT or TPOT change reaches and fails its
scored family before a later exact oracle.

The Tier A checker ran before the live matrix, but it pins only the immutable
native source cell. It does not pin the bypass runtime, the composed adapter,
the `ExecutionResult` boundary, or the deployed reduction, so it does not
entail either end-to-end relation.

The result record explicitly reports 6/6 and 12/12 as genuine-risk fractions.
D-zero identity, exact closed forms, authority labels, transaction safety,
schema checks, binary hashes and artifact preservation are fatal-unscored.
The binary and artifact hashes are change-set guards that cannot increase a
behavioral denominator.

## Live reduction and authority evidence

The raw matrix contains one canonical graph and one canonical `StepRecord`
across all four cells. The implementation passes the same in-memory graph and
record objects to the two mode reductions in each cell; the focused reducer
spy observed exactly those two calls. The unscored compute-only history seed
also traversed a real runtime and reducer twice, producing the common first
token for `core21-decode` before the decision step.

The bypass rows reported:

- authority `AtlahsWqeLedger`;
- two WQEs with that authority and no native timing stages;
- a real `ExecutionResult`, runtime report and `StepResult`; and
- request summaries derived from the returned `StepResult`, not from a scalar
  JCT.

The structural rows reported:

- authority `SimllmNativeRnicSession`;
- two WQEs with that authority;
- no bypass ledger;
- both native doorbell and native network visits; and
- one successful two-WQE transaction after the registered failure probe.

The selected bypass bundle was constructed from the canonical graph, actual
completion events, actual `StepResult` and actual request summary. After the
isolated negative build, a new real bypass execution was decoded into the
repository `BypassArtifacts` type and compared through
`compare_bypass_artifacts`. All six input fields and all four behavioral
fields matched exactly.

## Transaction failure in the live harness

Every registered cell first submitted a one-WQE graph to the selected
two-WQE native session. All four cells observed the same atomic sequence:

| Observation | Value |
|---|---|
| exception | `ValueError: graph did not consume every WQE in the selected native cell` |
| counters before failure | 0 committed transactions, 0 committed WQEs |
| counters after failure | 0 committed transactions, 0 committed WQEs |
| runtime report after failure | absent |
| bypass ledger in structural runtime | absent |
| counters after valid retry | 1 committed transaction, 2 committed WQEs |

This is live registered evidence, not a claim borrowed from
`tests/test_composed_rnic.py`.

## Link-disabled executable control

Both build trees used the same clean htsim source at
`4885c647eecdfdf81479d1df052223c016ad086b`, set source-symlink creation OFF,
and pointed CMake at the same clean SimLLM implementation commit. The positive
cache recorded `HTSIM_ENABLE_SIMLLM_RNIC=ON`; the negative cache recorded
`HTSIM_ENABLE_SIMLLM_RNIC=OFF`.

The OFF build produced the unconditional `htsim_rnic` and `txt2bin`
executables and no candidate `htsim_rnic_tier_a`. Its `htsim_rnic` SHA-256 was
`2e64eea9f1b6dc1e71406d0d27e6f053720e66c132a1e46d577a26b3bbf19161`.
The registered producer actually ran that OFF main executable on the frozen
non-fluid `rnic-nn` preflight. The executable completed, but its output lacked
the ON-only structural/native manifest, so the producer exited 2 with exactly:

```text
composed preflight lacks structural native authority
```

It failed before resolving a Tier A producer or publishing raw observations.
The same checker then exited 2 with exactly:

```text
authority observations do not exist
```

The negative raw observation, raw temporary, result and result temporary
paths were absent. Native-observation and native-observation temporary paths
were also absent. The positive `results.json` was published only after those
checks and the preservation guards passed. This was an executable link-level
control, not a mutated observation.

The positive preflight independently reported one quiescent flow and a
manifest containing `hardware_mode=structural`,
`wqe_authority=simllm-native-rnic-session`, `native_posts=1`, and
`physical_quiescence=verified`.

## Preserved positive assets

The runner hashed the positive assets before the OFF build and again after
the OFF producer and checker executions. Every value matched:

| Asset | SHA-256 |
|---|---|
| composed Tier A producer | `09925308faee0e082d8e1ce46a2a7d1cdda0a34b8b7db925ed2f061ea72b7ae4` |
| positive RNIC main | `0031a1b0284372bb097f9f1fe65f29fbb7b4b77bcae2021ce576559f5b9735f6` |
| positive `txt2bin` | `c0fa60292d2c4b8feacb45a35237aef60e8022819730296affe469bdcfc64240` |
| native observations | `37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a` |
| raw observations | `ceada27fd21053c7ac7087e8c246007df57b31d6e0298cde0e7b825af35fcd63` |
| bypass graph text | `d1b37acbba5e936eacbabb5ff203170435fc58df164aa570d219f761e7a98858` |
| bypass graph binary | `fa17721d635a19de1c15434bdb8df2d7b1d25dcf46fc566e3def164c9e319b69` |
| bypass topology | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| bypass completion CSV | `7132f6403a501e9e4aa9f68767c3e099c06caf0f3b709e78c5ae9c34033ba4ec` |
| bypass canonical completion | `07fb4314e0fc63242e082f8d3c2221c69d9c6eab84f8c320459888a6624500be` |
| bypass `StepResult` | `296f67bbabac85cfef83bcfbac64837b75a3efb2b3e88d5e17df1e3063e8f3bc` |
| bypass request summary | `035cfb0a30ff300d94c43e69ffb8f301a3141044d2194111d82f059e604b43d9` |

The same-graph bypass is a Python runtime authority, not a second native
executable. Its protected evidence is the byte-locked `BypassArtifacts`
bundle above. The positive RNIC main retains the native executable's composed
and compatibility-capable profiles. No existing core or backend behavior was
modified by this study.

## Closure scope

### CORE-21

| Registered acceptance clause | Evidence |
|---|---|
| `Freeze one fixed contended ExecutionGraph` | The expectation commit fixes one synchronous operation with two 4 KiB extents, release 17,000 ps, request identities and a four-cell matrix. |
| `run that same graph through the timing-neutral bypass authority and the composed native authority` | One canonical graph and record appear in all raw cells; each cell reports `AtlahsWqeLedger` and `SimllmNativeRnicSession`, and the focused object-identity test observes the same graph and record objects at both reducer calls. |
| `register the signed JCT, TTFT and TPOT change before execution` | The freeze registered a positive exact `+1,000 ps` band; the live result passed all six instances at exactly `+1,000 ps`. |
| `Both modes must use identical semantic inputs and the deployed ExecutionResult -> StepResult reduction` | Canonical graph and `StepRecord` bytes are identical, both raw modes contain their own execution result and live step result, and the reducer spy observes both deployed calls. |
| `the bypass side must not synthesize StepResult tuples or request summaries from a scalar JCT` | The bypass producer reduces its runtime result, serializes that returned `StepResult`, derives the summary from its request metrics, and locks those exact bytes into `BypassArtifacts`. |
| `Preserve the accepted bypass artifacts exactly` | The post-negative replay passed every repository comparator input and behavioral field; all selected bypass hashes were identical before and after the OFF build. |
| `prove the structural session remains the sole WQE authority when selected` | Every structural WQE and report names `SimllmNativeRnicSession`, native stages are present, the structural bypass ledger is absent, and the structural Tier A authority rows report zero legacy construction, posts and mutations. |
| `include transactional failure in the registered live harness rather than relying only on the existing unit test` | All four live cells recorded the 0/0 abort followed by exactly one 2-WQE commit. |

All registered CORE-21 clauses are demonstrated. CORE-21 closes without a
residual; CORE-25 and CORE-26 remain unused.

### BACK-31

| Registered acceptance clause | Evidence |
|---|---|
| `From the same pinned sources, build a candidate with the SimLLM native library link deliberately disabled` | Disjoint ON and OFF trees used htsim `4885c64`; the OFF cache says `HTSIM_ENABLE_SIMLLM_RNIC=OFF`, its main exists, and its ON-only producer target is absent. |
| `invoke the registered composed producer and live checker` | The runner invoked the same producer and checker interfaces used by the positive path, with the OFF main and negative publication paths. |
| `The run must fail before publishing an accepted result because native authority and the signed D relation are absent` | The producer rejected the absent native manifest with exit 2 before raw publication; the checker rejected the absent raw record with exit 2; positive acceptance was published only afterward. |
| `An observation-only mutant is not a substitute for this link-level test` | The negative main executable was freshly linked with the option OFF and actually ran the non-fluid GOAL preflight. No observation was created or mutated. |
| `Preserve the accepted composed and bypass binaries and artifacts byte for byte outside the negative build` | The separate positive tree, raw/native records, three positive executables and standard bypass bundle were hashed before and after the negative executions; all twelve hashes matched. |

All registered BACK-31 clauses are demonstrated. BACK-31 closes without a
residual; BACK-42 and BACK-43 remain unused.

## Reproduction

Configure a clean checkout of the frozen htsim commit and a fresh external
output directory, then run:

```bash
export SIMLLM_HTSIM_PIN_ROOT=<clean-htsim-checkout>
export SIMLLM_RNIC_AUTHORITY_RUN_ROOT=<fresh-external-output-directory>

.venv/bin/python examples/rnic_authority_v1/run_study.py \
  --out "${SIMLLM_RNIC_AUTHORITY_RUN_ROOT}" \
  --htsim-source "${SIMLLM_HTSIM_PIN_ROOT}"
```

The runner refuses a dirty SimLLM or htsim tree, an existing output root, a
source commit mismatch, overlapping source/build paths, frozen expectation
drift, or any negative-output publication. It leaves both source trees clean.

## Deliberate scope

This result closes only the registered same-graph authority comparison and
link-disabled executable control. It does not generalize the exact deltas
beyond the frozen capacity-one, zero-header, zero-propagation fixture. It does
not claim congestion calibration, packet issue timing, arbitrary application
graphs, or an online persistent session. No `simllm/core` or
`simllm/backends` implementation changed; the study exercises their deployed
interfaces as they existed at the frozen audit anchor.
