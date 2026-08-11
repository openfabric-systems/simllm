# BRIDGE-3 child-lifetime expectations

## Freeze status and scope

This is the expectations-only record for BRIDGE-3. It precedes the child
lifetime implementation and every result-producing lifecycle run. The
companion dry-run harness contains only the frozen matrix, time bounds, source
pins, command-line validation and check-only reporting. Its check-only path
does not import or exercise the future lifecycle implementation.

BRIDGE-3 is a P0 process-lifecycle correctness task. Its decision metric is the
number of targeted simulator descendants still present after their owning
SimLLM interpreter receives `SIGTERM`. TTFT and TPOT are not meaningful after
the owning run has been killed. The normal-run control therefore reuses the
complete BRIDGE-1 result and artifact identity checker to prove that successful
diagnostic and prepared runs retain their accepted step metrics and bytes.

## External-source audit before freeze

The audit was completed against SimLLM base commit
`90ada43070adb3b1e624b6819aff34d8620e8571`, pinned HTSIM commit
`4885c647eecdfdf81479d1df052223c016ad086b`, CPython 3.10.19 and Linux UAPI
headers before this freeze. No lifecycle implementation or result-producing
kill trial was run before this record.

- `simllm/backends/step_sink.py:320-359` compiles and executes every diagnostic
  plan synchronously. Prepared mode submits the same function to as many as
  `max_workers` threads at `simllm/backends/step_sink.py:425-445`.
- `simllm/backends/htsim_rnic.py:165-195` uses `subprocess.run` with a
  600-second timeout, but supplies no process group, parent-death mechanism or
  parent signal-and-reap handler.
- CPython `Lib/subprocess.py:2944-3056` implements `run` by communicating with one
  `Popen` object. Timeout and exceptional cleanup call `process.kill()` on that
  direct child, then wait. The `Popen` interface lists `start_new_session` as
  POSIX-only at `Lib/subprocess.py:3320-3396`.
- Linux `include/uapi/linux/prctl.h:7-10` defines `PR_SET_PDEATHSIG` as a signal
  delivered when the creating parent dies.
- The Windows SDK metadata defines the extended Job Object limit structure at
  `generation/WinSDK/RecompiledIdlHeaders/um/winnt.h:12507-12514` and the
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` flag at the same file's lines
  12829-12838.
- The pinned native CLI requires one GOAL at
  `third_party/htsim/htsim/sim/datacenter/rnic_atlahs_cli.cpp:441-452`.
  Its `main` constructs one runtime, executes it, writes at most one completion
  CSV, validates quiescence and exits at
  `third_party/htsim/htsim/sim/datacenter/main_rnic.cpp:152-221`.

The implementation strategy is therefore platform-specific behind one
invocation helper:

- Linux starts each simulator in a new session and uses a one-byte launch
  handshake. A single-threaded launcher sets `PR_SET_PDEATHSIG` before it
  acknowledges the launch and replaces itself with the simulator. A registered
  parent handler also terminates the owned process group and reaps the direct
  child for catchable shutdown.
- Other POSIX platforms start a new session and use the same registered
  signal-and-reap handler for `SIGTERM`, `SIGINT` and `SIGHUP`. They do not
  claim protection from uncatchable `SIGKILL` or host failure. A worker-thread
  launch must fail clearly if the main thread did not install the handler.
- Windows assigns the handshake-blocked launcher to a Job Object configured
  with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` before releasing it. Job creation,
  configuration or assignment failure aborts the launch. Closing or
  terminating the job owns descendants even when interpreter termination is
  abrupt.
- Any other platform rejects managed simulator launch with an actionable
  unsupported-platform error. No supported path silently falls back to an
  unowned child.

Every owned POSIX process has a newly created group whose ID is its registered
child PID. Cleanup may signal only a still-live registered `Popen` and its own
group. Windows cleanup may terminate only the per-invocation Job Object. This
is the unrelated-process exclusion rule.

## Frozen kill matrix and measurement protocol

The hermetic simulator stand-in sleeps for 30 seconds and writes no result. It
is invoked through real `HtsimStepSink` and `HtsimPersistentStepSink` calls;
only text-to-binary conversion is replaced by an in-process fixture copy so
that no second native tool is part of the kill timing. The two varied
parameters are invocation mode and prepared worker width:

| Cell | Invocation | `max_workers` | targeted children |
|---|---|---:|---:|
| D1 | diagnostic | not applicable | 1 |
| P2 | prepared | 2 | 2 |
| P4 | prepared | 4 | 4 |

Each cell runs twice. The negative run selects an internal, explicitly unsafe
regression-only switch that executes the pre-fix unowned invocation. The
enabled run uses the default managed path. The switch is not a supported
deployment mode and must never become the default.

The controller creates a unique run nonce and an empty marker directory. The
invocation layer records, during the live run, the owner PID, child PID,
command digest and Linux procfs start-time token before releasing the child
handshake. The controller waits at most 10 seconds for the exact targeted child
count, then sends one real `SIGTERM` to the owning interpreter. It waits at
most 10 seconds for that interpreter to exit.

After owner exit, the controller polls every registered `(pid, start-time)`
target every 20 milliseconds for at most 5 seconds. A target counts as
remaining if that exact process still exists in any state, including zombie
state. A surviving target whose recorded owner has exited is an orphan. PID
reuse or marker disagreement is a fatal unscored targeting failure, and the
controller never signals such a PID. Negative-control survivors are killed
only after their observation is recorded and their start-time token is
revalidated; their removal is then polled to the same bound.

## Scored live-runtime relation

For each of D1, P2 and P4, let `U` be the remaining targeted count with the
fix disabled and `M` the count with the managed default. The registered exact
relation is:

```text
U = targeted children
M = 0
M - U = -targeted children
```

Thus the expected rows are `(U, M) = (1, 0)`, `(2, 0)` and `(4, 0)`. The band
is exact, with no tolerance. All three instances are genuine-risk because the
unsafe subprocess may remain alive for the stand-in's full 30 seconds, while
the managed result depends on live parent-death or signal cleanup rather than
on an artifact validated before entry.

Entailment is checked from raw observations. The harness records owner exit,
every target state and both counts before applying the relation. It does not
abort merely because a negative count is zero or a managed count is nonzero.
Only after all raw rows exist does it score the three pairs and raise for a
failed relation. Marker validity, exact ready counts and negative-control
cleanup are fatal unscored checks; none pins the post-kill count tested by the
scored relation. The scored family therefore remains fail-able in every run
that reaches it.

## Real-binary corroboration

One additional diagnostic cell uses the freshly built `htsim_rnic` and
`txt2bin` from the pinned HTSIM commit. It lowers the first frozen vLLM M4
record, waits for the real simulator child marker, kills the owner and applies
the same 5-second poll. The managed remaining count must be exactly zero. This
is fatal corroborating evidence, reported separately from the hermetic scored
family because native completion time can vary by host.

## Normal, timeout and shutdown controls

The existing BRIDGE-1 study is rerun unchanged over both M4 captures and
worker widths 4 and 8. For each evidence class, all 34 prepared-versus-
diagnostic pairs must match exactly: `StepResult`, `StepNetworkOutcome`, GOAL
text, GOAL binary and completion CSV. Its four latency streams and six
physical-quiescence cells must also pass. These are fatal unscored identity
controls. They are evaluated by the existing checker, not by a second
comparator.

Unit and integration controls must additionally demonstrate:

- captured normal stdout, stderr and return code are byte-identical to a
  direct successful invocation;
- timeout cleanup removes and reaps the exact registered child group or Job
  Object descendants;
- repeated cleanup and repeated persistent-sink `close` calls are harmless;
- registration is removed after normal completion; and
- a platform primitive failure rejects before the simulator handshake rather
  than running an unowned child.

These cleanup and portability checks are fatal unscored evidence. The launch
handshake and static platform selection are change-set guards and do not enter
the scored denominator.

## Registered commands and pre-freeze dry runs

Local configuration sets `SIMLLM_WAVE5_RUN_ROOT` to this branch's external
run root and sets `SIMLLM_PINNED_HTSIM_RNIC` and `SIMLLM_PINNED_TXT2BIN` to
executables built from the pinned submodule. The lifecycle command is:

```bash
.venv/bin/python examples/bridge_lifecycle_v1/run_study.py \
  --out "$SIMLLM_WAVE5_RUN_ROOT/bridge3" \
  --real-htsim "$SIMLLM_PINNED_HTSIM_RNIC" \
  --real-txt2bin "$SIMLLM_PINNED_TXT2BIN"
```

The unchanged BRIDGE-1 identity command is:

```bash
SIMLLM_HTSIM_RNIC="$SIMLLM_PINNED_HTSIM_RNIC" \
SIMLLM_TXT2BIN="$SIMLLM_PINNED_TXT2BIN" \
.venv/bin/python examples/bridge_persistent_v1/run_study.py \
  --out "$SIMLLM_WAVE5_RUN_ROOT/bridge1-identity" \
  --fixtures vllm,sglang --workers 4,8
```

Before this freeze, both commands were run with `--check-only` appended. The
lifecycle dry run parses every option, validates the three-cell matrix, time
bounds, source pins, executable inputs and external-output rule. The existing
identity dry run validates its full 34-pair fixture plan. Neither command
creates its output directory or produces a result artifact in check-only
mode.
