# HTSIM-24 held-out persistent-session wall results

Run on 2026-08-13. The held-out requalification passes. Both replays satisfy
every predeclared two-sided band and the signed speedup relation, and every
fatal identity guard holds, so the run is valid rather than void. The
wall-clock family passes `2/2` genuine-risk instances. HTSIM-24 closes.

## Chronology and provenance

The two-stage freeze was respected in order:

1. `522f1fdc7830fd378b15cc9177b764b299d21fec` froze the workload-selection
   rule, the immutable replay cells, the complete timing boundaries, the
   calibration repetitions, every band formula, the signed direction, the
   physical bounds and the fatal identity surface. All registered command
   shapes had run with `--check-only` before it.
2. Calibration then ran with the base CLI only. Its summary records
   `session_invocations: 0`, and the runner has no code path that passes
   `--flow-session` during that phase.
3. `7329946b09ae3804650086937a99e89c6e14ee11` committed the mechanically
   materialized band lock. It is a calibration lock carrying measured base-only
   inputs, not an expectations-only commit, and it says so.
4. Only after that commit was `--flow-session` invoked for the first time in
   this study.

| artifact | SHA-256 |
|---|---|
| calibration summary | `8e3114edfff38a24ba052b0e5dc9b0f6ce0335bcbebe824c80452f9c1c909022` |
| band lock `bands.json` | `06ef68db1152a107535082efe2822889ffd6f463136edbcd890362e20c2de0de` |
| qualification summary | `02a7501c1e75eb7a77e3c185092f2397ce03ddff52c3434c601e18a861f003b5` |
| `htsim_rnic` | `500c5afb8c29335d19bd7b77a166168c05d56f744fcf3d9df16be49ff2ee304f` |
| `txt2bin` | `cf5d85c1774222e500318b08d917b63bfad8aff97b786014032d3e15146fb6f7` |

Evidence was authored against SimLLM `7622387` and HTSIM `fc4400e4`. The runs
observed HTSIM `1f2c124c9738edcfa0f6044b4667c230e75a542c` and SimLLM
`7329946b09ae3804650086937a99e89c6e14ee11`. No equality between an authored
literal and a live `third_party/htsim` pin is asserted, and the pin was not
moved. Host: Linux x86-64, Python 3.12.12. Bulk GOAL, converted binary, CSV and
framed session artifacts stay outside Git under
`${SIMLLM_DATA_ROOT}/persistent_session_wall_v1-calibration/` and
`${SIMLLM_DATA_ROOT}/persistent_session_wall_v1/`.

Four unrelated long-running processes owned by the same account held roughly
four cores throughout, on a 32-core host. That load was present and steady for
both the calibration and the qualification, so it enters both sides of every
band identically rather than favoring one.

## How wave-5 session timing was kept out

The constraint is that no wave-5 session timing may select the held-out
workload or any threshold. Three independent mechanisms enforce it:

- The workload is generated, not chosen. The frozen rule takes the first two
  odd endpoint cardinalities above one, emits every directed edge of the
  bidirectional ring in source order, assigns 4096 bytes clockwise and 8192
  bytes counterclockwise, and spaces injections 1,000,000,000 ps apart. No
  observation participates in that rule, and it was fixed before any local
  timing command ran.
- The thresholds are mechanical. Every band is a closed-form function of the
  new calibration only, and the qualification recomputes the bands from the
  calibration and refuses to proceed if the committed lock disagrees by a
  single field.
- The dataflow admits no wave-5 input. The runner reads only
  `--calibration` and `--bands`, both produced in this study, and validates the
  schema of each. There is no code path that can load a wave-5 artifact or
  threshold.

## Held-out replays

| replay | nodes | flows | payloads |
|---|---:|---:|---|
| `odd-3-biring` | 3 | 6 | 4096 clockwise, 8192 counterclockwise |
| `odd-5-biring` | 5 | 10 | 4096 clockwise, 8192 counterclockwise |

## Physical bound before reading the measurement

Two bounds were set before the session outcome was observed.

The floor: a persistent session still starts and reaps one copy of the same
executable, so its complete boundary cannot fall below one process lifetime.
The frozen anchor sets that floor empirically at `min(help)/4`, a predeclared
fourfold allowance for scheduler and timer noise, giving 275,213 ns.

The ceiling: with the frozen isolated upper band this fixes the finite maximum
speedup `isolated_upper / session_lower`, i.e. 197.1x for `odd-3-biring` and
274.7x for `odd-5-biring`. That is a deliberately loose guard against an
incomplete timer rather than a prediction.

The realistic expectation sits far below that ceiling. The isolated boundary
spawns two processes per flow (`txt2bin` then `htsim_rnic`), so 12 processes
for `odd-3-biring` and 20 for `odd-5-biring`, while the session spawns one. The
session still pays per-flow simulation, framing and parsing cost, so the
speedup must land well under the 12x and 20x process-count ratios and well
above the frozen 1.1x minimum.

On the network side, 4096 bytes at 400 Gbit/s serialize in 81,920 ps and 8192
bytes in 163,840 ps. Every observed FCT must exceed its own payload floor and
stay under the 1,000,000,000 ps causal gap.

## F1: latency and authority identity, fatal and unscored

| replay | FCT bytes identical | completion identity | hardware hash matches calibration | quiescent | clean session exit | physical sanity |
|---|---|---|---|---|---|---|
| `odd-3-biring` | PASS | PASS | PASS | PASS | PASS | PASS |
| `odd-5-biring` | PASS | PASS | PASS | PASS | PASS | PASS |

The ordered isolated FCT list is byte-identical to the persistent list in both
replays: `2166400, 2249600` repeating, 6 entries for `odd-3-biring` and 10 for
`odd-5-biring`. Both distinct FCT values clear their serialization floors by
more than an order of magnitude (2,166,400 against 81,920 and 2,249,600 against
163,840) and sit far below the 1,000,000,000 ps causal gap, so no completion is
inside another step's interval. No fatal guard was violated, so the run is not
void and the scored numbers mean what they claim. These guards are never
reported as a fraction.

## R1: held-out wall-clock family, 2/2 genuine risk

Raw `perf_counter_ns` observations of the complete boundaries, evaluated before
the fatal identity comparison.

| replay | isolated band, ns | isolated, ns | session band, ns | session, ns | speedup | bands and signed speedup |
|---|---|---:|---|---:|---:|---|
| `odd-3-biring` | 7,346,619 to 54,254,414 | 25,835,257 | 275,213 to 22,772,361 | 4,194,369 | 6.16x | PASS |
| `odd-5-biring` | 11,155,499 to 75,610,724 | 43,140,144 | 275,213 to 26,672,875 | 7,238,252 | 5.96x | PASS |

Genuine-risk fraction: `2/2`. The signed expectation, lower complete-boundary
wall time in persistent mode, holds in both replays.

### Where the measurements sit against the bounds

Both speedups land between the frozen 1.1x minimum and the 12x and 20x
process-count ratios, which is where a competent session that still simulates
every flow should land. They are two orders of magnitude below the loose
197.1x and 274.7x ceilings, so no incomplete-timer signature appears.

The check that should scale with the primary number does. `odd-5-biring` has
`10/6 = 1.67x` the flows of `odd-3-biring`. Its isolated boundary measures
`43,140,144 / 25,835,257 = 1.67x` and its session boundary
`7,238,252 / 4,194,369 = 1.73x`. Both scale nearly linearly in flow count,
which is what per-flow-dominated cost predicts on both paths, and it is why the
two speedups are close to each other rather than diverging. A session whose
saving came from a fixed startup term would have shown a speedup that fell as
the flow count rose.

### Entailment analysis

R1 is evaluated directly from raw host nanosecond observations before the fatal
comparison runs. F1 fixes simulated completion values but constrains no host
process, conversion, framing, scheduling, parsing or reaping time, so it cannot
entail either wall instance. Calibration never invoked the session path, so it
places no constraint on any session observation, and its own seven-sample
envelope is a self-derived fit that is by construction and unscored. The
qualification isolated observation is new and taken after the lock, so the
isolated band is a genuine two-sided test rather than a restatement of its own
inputs. No earlier oracle entails R1.

The maximum-speedup check is not counted as a separate instance because it is
part of the same conjunction and follows from the two wall bands. The
serialization and causal-gap guards are fatal-unscored and are excluded from
the denominator.

Both instances carry real risk. A session can retain simulator state and still
lose the saving to framing, parsing, startup or per-flow bookkeeping; an
isolated boundary can drift outside a two-sided band frozen from an earlier
calibration; and an incomplete timer would breach the lower band or the
ceiling. None of these was ruled out before the observation.

## Closure scope

| registered acceptance clause | evidence | status |
|---|---|---|
| "held-out flow replay whose two-sided bands are frozen from the exact pinned base CLI before the session outcome is observed" | Topology-rule workload frozen at `522f1fd`, base-only calibration with `session_invocations: 0`, band lock committed at `7329946`, first session invocation only afterwards | DEMONSTRATED |
| "time the complete isolated and persistent boundaries" | Isolated timer spans every GOAL write, `txt2bin` conversion, child start and reap, simulator run and CSV parse with no process or artifact reuse; session timer spans spawn, framing, open, every inject, the inclusive advance, drain, close, EOF, stderr read and reap | DEMONSTRATED |
| "preserve the fatal latency-byte identity" | Canonical FCT byte equality, ordered identity tuples, hardware hash agreement, physical quiescence and clean exit, all PASS on both replays | DEMONSTRATED |
| "pass every predeclared band and signed speedup instance" | `2/2` conjunctions pass | DEMONSTRATED |
| "state the entailment and genuine-risk analysis without using any wave-5 session timing" | Three-mechanism isolation guarantee, entailment analysis, `2/2` fraction | DEMONSTRATED |

Every registered clause is demonstrated, so HTSIM-24 closes with no residual
moved to a new ID.

## Scope of the claim

This qualifies the wall-clock family of the persistent session on lightweight
flow replays. It is a host-time efficiency result, not a network-modeling
result: the simulated outcome is identical on both paths by fatal guard, which
is the point. It says nothing about session behavior on large step programs, on
profiles other than `rnic-nn`, or under concurrent sessions.

## Contradiction sweep

`README.md`, `docs/README_PRO.md` and `docs/architecture.md` describe the
persistent session as an open online-stateful-cosimulator item under BRIDGE-2,
CORE-24 and HTSIM-18 and make no wall-clock claim for it, so nothing in them
contradicts this result. No edit was made to those files.
