# HTSIM-24 held-out persistent-session wall expectations

This is the expectations-only record for HTSIM-24. It precedes every
result-producing calibration and qualification command in this study. The
existing persistent-session implementation predates this study, so this is a
clean prospective requalification, not a claim of preimplementation
registration.

The study uses a two-stage freeze. This commit freezes the workload-selection
rule, complete timing boundaries, calibration repetitions, band formulas,
signed directions, physical bounds and fatal identity surface. A later band
lock may contain only isolated base-CLI observations and the mechanically
materialized bounds defined here. That lock must be committed before the
session option is invoked for the first time in this study.

## External-source audit and provenance

The audit used SimLLM commit
`76223875557a552deb5aa2c2c529a07f000135ba` and HTSIM commit
`fc4400e4ca619223481536632074045cb6af2756`. These are the commits against
which the evidence was authored. The calibration and qualification summaries
record the SimLLM and HTSIM source commits actually observed plus executable
digests. They do not assert equality between either authored literal and a
live `third_party/htsim` pin.

- HTSIM `htsim/sim/datacenter/main_rnic.cpp:180-197` selects the opt-in
  `--flow-session` process before the unchanged one-GOAL CLI, and lines
  199-201 define the exact base help CLI used for the persistent-work anchor.
- The same file at lines 204-260 owns the complete isolated boundary:
  parse one GOAL, construct one event authority and RNIC runtime, execute to
  quiescence, write the completion CSV, and exit.
- HTSIM `htsim/sim/datacenter/rnic_flow_session.cpp:1-30` composes the
  session protocol with the same event list and native RNIC runtime. The
  protocol acceptance and exact latency identity were already qualified by
  HTSIM-18; HTSIM-24 tests only its wall-clock family on new replay shapes.
- SimLLM `examples/persistent_session_v1/expectations.md:263-284` defines the
  accepted complete isolated and persistent timing boundaries and identifies
  latency-byte identity as independent of host elapsed time.
- `docs/modules/backends.md:959-968` assigns this held-out requalification to
  HTSIM-24 and forbids wave-5 session timing from choosing either workload or
  thresholds.

## Held-out workload selection and isolation guarantee

The two replays are generated from a topology rule, not selected from timing:

- use the first two odd endpoint cardinalities greater than one, 3 and 5;
- for each cardinality, emit every directed edge of the bidirectional ring in
  source order, clockwise before counterclockwise;
- clockwise payloads are 4096 bytes and counterclockwise payloads are 8192
  bytes; and
- inject successive flows 1,000,000,000 ps apart, far beyond their 400 Gbit/s
  payload serialization floors.

This yields the following immutable cells:

| replay | nodes | flow count | ordered `(source,destination,payload_bytes)` |
|---|---:|---:|---|
| odd-3-biring | 3 | 6 | `(0,1,4096)`, `(0,2,8192)`, `(1,2,4096)`, `(1,0,8192)`, `(2,0,4096)`, `(2,1,8192)` |
| odd-5-biring | 5 | 10 | `(0,1,4096)`, `(0,4,8192)`, `(1,2,4096)`, `(1,0,8192)`, `(2,3,4096)`, `(2,1,8192)`, `(3,4,4096)`, `(3,2,8192)`, `(4,0,4096)`, `(4,3,8192)` |

This rule guarantees independence from wave-5 timing in two ways. It was
frozen before any local timing command, and the runner accepts no wave-5
artifact or threshold input. Band materialization reads only the new
calibration summary produced by the exact configured base CLI. Existing
wave-5 result values cannot enter the dataflow.

## Complete timing boundaries

Each isolated replay measurement starts before rendering its first GOAL and
ends after parsing its final completion CSV. It includes every GOAL write,
every `txt2bin` conversion, every child-process start and reap, every simulator
run, and every CSV parse. It does not reuse a process or converted artifact.

Each persistent replay measurement starts before spawning `htsim_rnic
--flow-session` and ends after the close response, EOF, stderr read and process
reap. It includes framing, open, every inject, the inclusive advance, drain,
close, all simulation and all response parsing.

Source inspection, workload construction, executable hashing, calibration
loading and evidence comparison remain outside both timers.

## Base-only calibration and mechanical band lock

Calibration invokes only the exact configured one-GOAL and help CLIs. It must
not pass `--flow-session` or write any session frame. For each replay it takes
seven independent complete isolated samples. It also takes seven complete
`--help` process samples as an empirical anchor for the one process boundary
that persists in session mode.

For replay `r`, let `I_r` be its seven isolated nanosecond observations, `H`
the seven help-process observations, `n_r` its flow count, and `median` the
integer middle order statistic. The band lock is exactly:

- `isolated_lower_r = floor(min(I_r) / 2)`;
- `isolated_upper_r = 2 * max(I_r)`;
- `session_lower_r = max(1, floor(min(H) / 4))`;
- `session_upper_r = floor(9 * median(I_r) / 10)`;
- `minimum_speedup_r = 11 / 10`; and
- `maximum_speedup_r = isolated_upper_r / session_lower_r`.

The fractions are stored as integer numerators and denominators, not rounded
floats. The band-lock file records the calibration-summary SHA-256 and exact
base executable digests. Its commit is a calibration lock containing measured
base-only inputs, not an expectations-only commit. Any hand edit or formula
change after calibration invalidates the study.

The seven calibration observations and derived calibration envelope are
configuration evidence. Their self-derived fit is by construction and
unscored. Only a new isolated qualification observation taken after the band
lock is scored.

## Physical sanity before session observation

At 400 Gbit/s, payload serialization alone is a floor of 81,920 ps for 4096
bytes and 163,840 ps for 8192 bytes. Every isolated FCT must exceed its
payload floor and remain below the 1,000,000,000 ps causal gap. These are fatal
network-sanity preconditions, not behavioral scores.

Host wall time and simulated picoseconds are different quantities. The
session must still start and reap one copy of the same executable. Before any
session outcome is read, the persistent-work anchor therefore sets a positive
session lower band from `min(H) / 4`. Together with the frozen isolated upper
band it sets the finite speedup ceiling
`isolated_upper_r / session_lower_r`. The quarter-anchor is a predeclared
fourfold allowance for scheduler and timer noise. A faster observation is
treated as evidence of an incomplete timing boundary, not as unlimited
improvement.

## F1: latency and authority identity, fatal and unscored

For every replay, the canonical UTF-8 JSON bytes of the ordered isolated FCT
integer list must equal the persistent completion FCT list byte for byte.
Completion count, ordered source, destination, tag and payload identities,
hardware hash consistency, physical quiescence and a clean session exit must
also hold exactly.

Any F1 violation makes the wall run void. It is never reported as a lost point
or included in a fraction. These identities are accepted-authority guards and
do not raise the behavioral denominator.

## R1: held-out wall-clock family

After the band-lock commit, the qualification command takes one new complete
isolated observation and then the first session observation for each replay.
It records the raw wall values and evaluates the following conjunction before
applying F1:

1. isolated elapsed time is in its inclusive two-sided band;
2. session elapsed time is in its inclusive two-sided band; and
3. raw isolated divided by raw session elapsed time is inclusively between
   `11/10` and the frozen maximum speedup.

The signed expectation is lower complete-boundary wall time in persistent
mode. R1 has two scored instances, one per replay. Both are genuine risk: a
competent session can retain simulator state yet spend too long on framing,
parsing, startup or per-flow bookkeeping; an isolated boundary can drift; and
an incomplete timer can violate the lower band or physical ceiling. Planned
genuine-risk fraction: `2/2`.

### Entailment analysis

R1 is evaluated directly from raw `perf_counter_ns` observations before the
fatal latency-byte comparison. F1 fixes simulated completion values but places
no constraint on host process, conversion, framing, scheduling, parsing or
reaping time. Calibration does not invoke the session path, and its envelope
does not pin either later qualification observation. No earlier oracle entails
R1.

The maximum-speedup check is not counted as a separate instance because it is
already part of the same conjunction and follows from the two wall bands. The
physical serialization and causal-gap guards are likewise fatal-unscored.

## Closure scope

HTSIM-24 acceptance maps as follows:

| registered acceptance clause | frozen evidence |
|---|---|
| "held-out flow replay whose two-sided bands are frozen from the exact pinned base CLI before the session outcome is observed" | Initial workload/formula freeze, base-only calibration summary, committed band lock, then the first session invocation. |
| "time the complete isolated and persistent boundaries" | The exact timer boundaries above, emitted raw nanoseconds and command/artifact manifests. |
| "preserve the fatal latency-byte identity" | F1 canonical FCT bytes, identity tuples, quiescence and clean exit. |
| "pass every predeclared band and signed speedup instance" | Both R1 conjunctions must pass or the task remains open. |
| "state the entailment and genuine-risk analysis without using any wave-5 session timing" | The dataflow isolation guarantee, R1 analysis and final per-family fraction. |

Any unproved clause moves to HTSIM-25 or HTSIM-26 with a categorized priority
and difficulty tag.

## Registered commands and check-only dry run

Bulk outputs remain outside Git. The commands are:

```bash
HTSIM_SOURCE_ROOT="${HTSIM_SOURCE_ROOT:?configure the backend source}" \
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_RNIC:?configure the exact base binary}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the matching converter}" \
.venv/bin/python examples/persistent_session_wall_v1/run_study.py \
  --phase calibrate \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/persistent_session_wall_v1-calibration"
```

```bash
HTSIM_SOURCE_ROOT="${HTSIM_SOURCE_ROOT:?configure the backend source}" \
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_RNIC:?configure the exact base binary}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the matching converter}" \
.venv/bin/python examples/persistent_session_wall_v1/run_study.py \
  --phase lock \
  --calibration "${SIMLLM_DATA_ROOT:?configure the data root}/persistent_session_wall_v1-calibration/summary.json" \
  --bands examples/persistent_session_wall_v1/bands.json
```

```bash
HTSIM_SOURCE_ROOT="${HTSIM_SOURCE_ROOT:?configure the backend source}" \
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_RNIC:?configure the exact base binary}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the matching converter}" \
.venv/bin/python examples/persistent_session_wall_v1/run_study.py \
  --phase qualify \
  --calibration "${SIMLLM_DATA_ROOT:?configure the data root}/persistent_session_wall_v1-calibration/summary.json" \
  --bands examples/persistent_session_wall_v1/bands.json \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/persistent_session_wall_v1"
```

Before this expectations commit, both command shapes must run with
`--check-only`. Check-only validates executable paths, phase-specific inputs,
the immutable replay registry, all formulas, complete boundary definitions and
external output placement. It prints the plan and creates no artifacts. For
the prospective lock and qualification checks, in-memory schema-valid
placeholder calibration and mechanically matching band lock objects are used
only to validate parsing; they contain no observation. The untracked harness present at freeze
time encodes only frozen literals, orchestration and check-only validation. It
contains no backend implementation or observed timing.
