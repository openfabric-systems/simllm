# BACK-2 LogGOPSim invocation helper expectations

This is the expectations-only record for BACK-2. It precedes the helper
implementation and every result-producing command of this study. BACK-2
registers one clause, "LogGOPSim invocation helper for fast flow-level
sweeps", so the object under test is the invocation seam and its result
parser, not a new network model and not a new fidelity level. TRAF-20 owns the
fluid fast level itself and is out of scope here.

The decision-relevant outcome is whether a caller can drive the unmodified
LogGOPSim binary from `simllm.backends` with the same discovery, argument and
parse discipline the htsim helpers already use, and get a completion time that
obeys the LogGOPS cost model exactly.

## Pre-freeze tooling probe, disclosed

A tooling probe ran before this freeze, so its cells are excluded from the
scored sweep and this study does not claim pre-registration for them.

The probe linked the LogGOPSim executable from the ATLAHS submodule's
prebuilt objects (`sim/LogGOPSim/LogGOPSim.o` plus `cmdline.o`, linked with
`g++` into an external directory so no submodule file is written), converted
two GOAL fixtures with an existing `txt2bin`, and ran the unmodified binary.

- An 8-rank cyclic fixture in which every rank receives before it sends
  reported every host finishing time as zero. That schedule deadlocks, so the
  simulator has nothing to advance. It is the reason this study asserts a
  nonzero finishing time as a fatal precondition.
- A 2-rank fixture sending 1048576 bytes reported `Maximum finishing time at
  host 0: 6297950` at the default parameters, and `12589400` with `-G 12`.

Those two observations identified the model constant below. The constant is
therefore post-specified. Every scored relation in this study is a difference
that eliminates the constant, and no scored cell repeats a probed cell.

## Frozen cost model

LogGOPSim's `sim/LogGOPSim/simulator.ggo` defines `L`, `o`, `g`, `O` in whole
nanoseconds and `G` in nanoseconds per byte, with `S` the eager to rendezvous
threshold in bytes. For one point-to-point message of `s` bytes with
`s > S`, on an otherwise empty schedule, the sender's maximum finishing time
in nanoseconds is

```text
T(s, G) = C + (s - 1) * G,     C = 2 * o + L + g
```

`C` is 6500 ns at the tool defaults `o = 1500`, `L = 2500`, `g = 1000`. The
byte term is the only term that depends on `s` or `G`.

### Physical sanity, stated before any measurement

- Floor: no message can finish before its own byte serialization,
  `(s - 1) * G` ns. Nothing in a LogGOPS run may beat that.
- Ceiling: the constant terms of a rendezvous exchange are three overheads,
  one latency and one message gap at most, so `floor + 15000` ns bounds every
  cell from above at the tool defaults.
- A measured value outside `[floor, floor + 15000]` proves a defect in the
  helper, the fixture or the reading, whatever the relations say.
- Scaling cross-check: `G` is an inverse bandwidth, so a serialization
  dominated cell must move by close to two when `G` doubles. The ratio
  `T(s, 6.0) / T(s, 3.0)` must lie in `[1.99, 2.01]`. A ratio near 1.05 or
  near 40 refutes the relation regardless of how exactly a single number
  matched.
- Calibration disclaimer: `G = 6` ns per byte is about 1.33 Gbit/s. These are
  sweep parameters chosen to exercise the seam, not a calibrated claim about
  any modern fabric. This study makes no TTFT or TPOT claim.

## Fixed fixtures and sweep

Every fixture is emitted by `simllm.goal.GoalTrace`, converted by the
repository's own `txt2bin` helper, and run through the new invocation helper.
Bulk artifacts stay outside Git.

The scored sweep varies two parameters, message size and the per-byte gap:

| cell | `payload_bytes` | `byte_gap_ns` |
|---|---|---|
| A | 262144 | 3.0 |
| B | 262144 | 6.0 |
| C | 524288 | 3.0 |
| D | 524288 | 6.0 |

All four sizes stay above the default `S = 65535`, so all four cells sit in
the same rendezvous regime. No cell repeats the probed 1048576-byte point.

## F1: run validity, fatal and unscored

For every cell the helper must return a parsed maximum finishing time that is
strictly positive, and the parsed rank count must be 2. A zero finishing time
means the fixture deadlocked, as the probe showed, and then no timing number
in this study means what it claims. Violation voids the run rather than
scoring a fraction.

Each cell must also sit inside its stated floor and ceiling. This is a
physical precondition, not a behavioral score.

## E1: exact-oracle class, argument and parse rows

Separate evidence class from the behavioral relations below, and never added
into one headline total.

- The helper must build exactly the frozen argument vector for a frozen
  configuration, including the `-f`, `-L`, `-o`, `-g`, `-G`, `-O`, `-S` and
  `-n` spellings that `simulator.ggo` defines, in that order, with the batch
  flag `-b` present only when batch mode is selected.
- The helper must parse a frozen captured stdout sample of each of the two
  output shapes the tool can print: the per-host `Times:` block that appears
  at 16 ranks or fewer without batch mode, and the single `Maximum finishing
  time at host` line that appears in batch mode.
- Nanosecond to picosecond conversion is exactly a factor of 1000, matching
  the existing conversion in `simllm/backends/htsim_rnic.py`.

## E2: explicit off path, fatal and unscored

BACK-2 is a completeness task, so its off path is part of acceptance. With no
LogGOPSim executable configured, discovery must return `None` and the runner
must raise an actionable error that names the `SIMLLM_LOGGOPSIM` environment
variable. This is a by-construction guard: fatal when violated, never scored.

The helper is additive. No existing module may change behavior because it
exists, and the complete existing test suite must stay green.

## R1: per-byte gap family

For each message size, holding the size fixed and moving the per-byte gap from
3.0 to 6.0 ns must add exactly the byte term and nothing else:

```text
T(s, 6.0) - T(s, 3.0) == (s - 1) * 3.0
```

| instance | expected difference, ns |
|---|---|
| s = 262144 | 786429 |
| s = 524288 | 1572861 |

Two scored instances.

## R2: message size family

For each per-byte gap, holding the gap fixed and doubling the message size
must add exactly the extra bytes and nothing else:

```text
T(524288, G) - T(262144, G) == 262144 * G
```

| instance | expected difference, ns |
|---|---|
| G = 3.0 | 786432 |
| G = 6.0 | 1572864 |

Two scored instances.

R1 and R2 together give four scored instances. Planned genuine-risk fraction:
`4/4`.

### Entailment analysis

The scored relations are evaluated against the four raw finishing times before
any other check reads them. No earlier fatal oracle pins those four numbers.
E1 pins the argument vector the helper builds and the parse of a frozen text
sample; neither constrains what the unmodified simulator returns for a live
fixture. F1 constrains only positivity and the physical bracket, which is far
looser than an exact difference. The genuine risk is real and specific: a
helper that drops `-G`, formats the float in a spelling gengetopt rejects,
converts the wrong stdout line, or fails to carry the emitted payload size
through `txt2bin` breaks R1 or R2 while still passing E1 and F1.

## C1: intercept invariance, fatal and unscored because entailed

`C = T - (s - 1) * G` must be the same integer in all four cells. This is
recorded, but it is entailed by R1 and R2 holding jointly on the two-by-two
grid, so it is not independent evidence and is not scored.

## Closure scope

BACK-2 registers one clause.

| registered acceptance clause | frozen evidence |
|---|---|
| "LogGOPSim invocation helper for fast flow-level sweeps" | The helper module with the repository's discovery, argument and parse discipline; E1 exact rows; E2 off path; F1 validity; R1 and R2 over the four-cell sweep. |

Anything the run does not demonstrate moves to a new identifier from the range
available to this branch, quoting the clause it failed. An idea for later work
that no registered clause claimed is recorded in prose instead.

## Registered command and check-only dry run

Bulk outputs remain outside Git. The registered command is:

```bash
SIMLLM_LOGGOPSIM="${SIMLLM_LOGGOPSIM:?configure the LogGOPSim executable}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the txt2bin executable}" \
.venv/bin/python examples/loggopsim_helper_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/loggopsim_helper_v1"
```

The same command with `--check-only` must run before this expectations commit.
Check-only validates both executables, the four sweep cells, the frozen
expected differences and the external output placement. It prints the plan and
creates no artifacts. The harness present at freeze time encodes only these
frozen literals, orchestration and check-only validation. It contains no
helper implementation and no observed outcome.
