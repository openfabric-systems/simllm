# BRIDGE-1 prepared co-simulator results

Run on 2026-08-11. The registered relation passed: the opt-in prepared replay
preserved every simulated step and backend artifact byte for byte while cutting
real end-to-end wall time by 3.36x to 5.43x on the literal registered run.

## Chronology and provenance

The public expectations were frozen at commit
`aa6e92f882e4a2091493ebee68e117547fe60d53`. That commit precedes both the
implementation commit `d3551e0bf8468b901b7882a7c107c3dc12cbd195` and every
prepared-mode run.

Before the freeze, the pinned diagnostic path alone was timed to choose honest
wall-clock bands. Its eight-record vLLM replay took 60.338481569 seconds. A
separate component audit measured 0.011178458 seconds for `txt2bin` and
7.252140791 seconds for one simulator invocation. These values were calibration
inputs, not registered evidence, and no prepared implementation existed when
they were observed. The expectations file excludes the measured values as
required for an expectations-only commit.

The full external-source audit appears in
[expectations.md](expectations.md). It established before freeze that pinned
HTSIM commit `edb28c3015c173b4251abc5858c587df325e1ebc` accepts one GOAL,
creates one event list and runtime, writes one completion CSV and exits. It has
no batch or stdin session interface. The registered command was run with
`--check-only` before freeze; it printed the two fixture hashes, two worker
widths and six-cell matrix by design, and produced no artifacts. The freeze
commit message records the exact working-tree state: only the expectations file
was staged, while the untracked dry-run harness existed and encoded only frozen
literals.

The exact pinned composed binaries used for the study were:

| binary | SHA-256 |
|---|---|
| `htsim_rnic` | `dc2415b39dd890cc49ba0b8042d08bb57b5a978c00510cb33fe048bc84e89e8d` |
| `txt2bin` | `b922987a752ebb028f580ee713748e9b9dd9e34ddec5f8bd1cc315abd0a56db2` |

The official external summary is
`${SIMLLM_DATA_ROOT}/bridge_persistent_v1/summary.json`, SHA-256
`03ed74e2a31e21dbe2279f6675364fff0f294285ff6ad5b8fa15c870df834bcd`.
The host reported Linux x86-64, 32 logical CPUs and Python 3.12.12. Raw GOAL,
binary and CSV artifacts remain outside Git.

One earlier post-implementation run used the frozen matrix and binaries but an
alternate external output suffix. It passed 4/4 scored instances and every
identity check. That operational mismatch was noticed before this report, so
the literal registered command was rerun without changing code, expectations
or bands. The earlier summary remains as a disclosed replication at
`${SIMLLM_DATA_ROOT}/registered-study/summary.json`, SHA-256
`7c8c230046814035dc76d8b77a1a133e032943e05407df10263cb6a3460ce48b`.
It is not substituted for the official table below.

## Scored live wall-clock relation

Elapsed time includes sink and pool construction, GOAL rendering and
conversion, child invocation, simulation, CSV parsing, result delivery and
pool shutdown. Fixture loading, hashing, comparisons and summary writing stay
outside the timed boundary.

| replay | workers | diagnostic, s | prepared, s | diagnostic band, s | prepared band, s | speedup | bound | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| vLLM | 4 | 68.868529 | 20.505591 | `[45, 85]` | `[10, 45]` | 3.3585x | >= 1.5x | PASS |
| vLLM | 8 | 68.868529 | 12.678099 | `[45, 85]` | `[6, 35]` | 5.4321x | >= 2.0x | PASS |
| SGLang | 4 | 76.716722 | 22.753830 | `[50, 100]` | `[12, 50]` | 3.3716x | >= 1.5x | PASS |
| SGLang | 8 | 76.716722 | 20.703817 | `[50, 100]` | `[7, 38]` | 3.7054x | >= 2.0x | PASS |

All four live-runtime parameterized instances passed. Every instance could
genuinely fail through process scheduling, contention, serialization or native
tool failure, so the genuine-risk fraction is `4/4` (100 percent). Replay
identity and worker width are the two varied parameters. No monotonic relation
between four and eight workers was registered.

The disclosed alternate-output replication also passed: vLLM measured
61.301904, 17.403924 and 10.573489 seconds for diagnostic, four and eight
workers; SGLang measured 72.218455, 24.060923 and 16.985783 seconds. Its four
speedups ranged from 3.0015x to 5.7977x.

## Fatal unscored identity evidence

The author-defined fixture rows and order, result conservation, artifact
identity and quiescence are fatal invariants. They do not increase the scored
denominator and are not added to the 4/4 headline.

| evidence class | passed | total | disposition |
|---|---:|---:|---|
| canonical `StepResult` per step | 34 | 34 | fatal, unscored |
| canonical `StepNetworkOutcome` per step | 34 | 34 | fatal, unscored |
| complete ordered latency stream | 4 | 4 | fatal, unscored |
| GOAL text per step | 34 | 34 | fatal, unscored |
| GOAL binary per step | 34 | 34 | fatal, unscored |
| completion CSV per step | 34 | 34 | fatal, unscored |
| physical quiescence per run cell | 6 | 6 | fatal, unscored |

The diagnostic and both prepared modes shared these complete latency streams:

- vLLM: `[2584833280, 2309582080, 2456446720, 2327996160, 2309582080,
  2529815040, 2327964160, 2309582080]` ps.
- SGLang: `[2511432960, 2309582080, 2419714560, 2327964160, 2309582080,
  2493114880, 2327964160, 2309582080, 2309582080]` ps.

The canonical result streams had SHA-256
`756cf19b4ab987fcf82f4a5a65ebe30d272fc50aba63b7f2dad21b35d0faa7c0`
for vLLM and
`40f8baf16ee75dc45c0a1f0fcfc9496861a7b143515e850c815e9a13e0cc8c6c`
for SGLang in all three modes. GOAL text, GOAL binary and completion CSV stream
digests likewise matched within each replay. Since the complete step-latency
streams are identical, replay-derived simulated TTFT and TPOT are unchanged;
the measured movement is host wall time only.

## Scope and residuals

BRIDGE-1 closes for the pinned-binary prepared-replay scope. The delivered
pool accelerates a finite known replay while deliberately retaining the
accepted diagnostic model in which each step resets backend state. It does not
claim online scheduler prediction or cross-step network-policy state.

The remaining work is explicit:

- BRIDGE-2 `(Completeness; P1; L)` owns the online graph-level client carrying
  `ExecutionGraph`, `CompletionEvent`, `ExecutionResult`, `StepResult` and
  loss-checked bookkeeping append facts.
- CORE-24 `(Completeness; P1; M)` owns the missing strict full `StepResult` wire
  codec required by that client.
- HTSIM-18 `(Completeness; P1; L)` owns the genuine stateful stdin/stdout flow
  session in the backend; its precise frame, sequence, authority and drain
  contract is registered in `docs/modules/backends.md`.

No backend source was changed, no physical profile was reinterpreted, no
tracked raw output was added, and no BRIDGE-3 task was created because there is
no fourth distinct deferred scope.
