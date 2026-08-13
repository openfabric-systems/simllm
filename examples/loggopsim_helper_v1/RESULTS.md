# BACK-2 LogGOPSim invocation helper results

Pre-run expectation commit: `543aa62`, which contains
[expectations.md](expectations.md) and the check-only harness and no helper
implementation. The registered command ran with `--check-only` before that
commit and created no artifacts. The helper landed after it, and the sweep ran
after the helper.

Observed SimLLM commit for the run: `543aa62`. The study touches no submodule
and asserts no submodule pin literal.

## What landed

`simllm/backends/loggopsim.py` is the invocation seam and result parser for
the analytical flow-level simulator, in the same shape as the two htsim
helpers next to it: discovery through `simllm._native`, a configuration
dataclass, an argument builder, a pure parser and a runner. It adds no
fidelity level and no new interface. TRAF-20 still owns the fluid fast level.

LogGOPS parameters keep the tool's own units, with explicit `_ns` names, and
parsed times convert to picoseconds by exactly 1000, matching the conversion
already in `simllm/backends/htsim_rnic.py`.

## F1: run validity, fatal and unscored

All four cells passed. Every finishing time is strictly positive, every banner
reports two ranks, and every value sits inside its stated physical bracket.

| cell | payload bytes | `G` ns/byte | floor ns | measured ns | ceiling ns |
|---|---|---|---|---|---|
| A | 262144 | 3.0 | 786429 | 792929 | 801429 |
| B | 262144 | 6.0 | 1572858 | 1579358 | 1587858 |
| C | 524288 | 3.0 | 1572861 | 1579361 | 1587861 |
| D | 524288 | 6.0 | 3145722 | 3152222 | 3160722 |

Every cell sits 6500 ns above its own serialization floor, which is 0.83
percent of cell A and 0.21 percent of cell D. Nothing beats its floor, and
nothing approaches the ceiling.

The pre-freeze probe's deadlocked 8-rank fixture is the reason this guard
exists: that fixture reported every finishing time as zero, which passes any
difference relation trivially. F1 is what separates a real measurement from
that failure mode.

## E1: exact-oracle class

`tests/test_loggopsim.py` holds fifteen oracles, kept separate from the
behavioral relations and never added into one total. They pin the exact
argument vector for a default and a fully specified configuration, the
`-b` placement, valueless and valued extra flags, the double literal a float
gap renders to, both output shapes the tool can print, the exact factor of
1000, the nonfinite `Average FCT is -nan` case, the rejected configurations,
and the two runner error paths.

## E2: explicit off path, fatal and unscored

Passed. With no executable configured, `find_loggopsim` returns `None` and
`run_loggopsim` raises `FileNotFoundError` naming `SIMLLM_LOGGOPSIM`. The
helper is additive: the complete suite is 1214 passed and 7 skipped, up from
1199 passed and 7 skipped by exactly the fifteen new oracles, with no existing
test changed.

## R1: per-byte gap family

Passed, two of two.

| instance | expected difference ns | observed ns | doubling ratio |
|---|---|---|---|
| payload 262144 | 786429 | 786429 | 1.9918 |
| payload 524288 | 1572861 | 1572861 | 1.9959 |

Both ratios sit inside the frozen `[1.99, 2.01]` band. They are below two by
exactly the invariant 6500 ns constant, which is the correct sign: doubling an
inverse bandwidth doubles the serialization term and leaves the fixed
overheads alone, so the total must grow by slightly less than two.

## R2: message size family

Passed, two of two.

| instance | expected difference ns | observed ns |
|---|---|---|
| `G = 3.0` | 786432 | 786432 |
| `G = 6.0` | 1572864 | 1572864 |

Genuine-risk fraction across R1 and R2: `4/4`, as planned.

## C1: intercept invariance, entailed and unscored

`T - (s - 1) * G` is 6500 ns in all four cells, the single value the frozen
model predicted from `2 * o + L + g` at the tool defaults. Recorded, not
scored, because R1 and R2 holding jointly on the two-by-two grid already
determine it.

## Three independent review angles

- Serialization physics. The byte term dominates every cell and matches the
  frozen prediction to the nanosecond. The residual is a constant, not a
  proportional error, so the model's shape and not only its magnitude is
  confirmed.
- Protocol accounting. The measured constant decomposes exactly as two
  overheads, one latency and one message gap, which is the rendezvous
  handshake the tool's own threshold selects at these sizes. All four cells
  are above `S = 65535` and land in the same regime, so no cell is compared
  across a protocol boundary.
- System plausibility. `G = 6` ns per byte is about 1.33 Gbit/s and `G = 3`
  about 2.67 Gbit/s. Moving 256 KiB at 1.33 Gbit/s in 1.58 ms is what such a
  link does. These are sweep parameters that exercise the seam, not a
  calibrated fabric, and this study makes no TTFT or TPOT claim.

## Closure

BACK-2 registered one clause, "LogGOPSim invocation helper for fast
flow-level sweeps". The helper exists with the repository's discovery,
argument and parse discipline, its off path is explicit and tested, and a live
end-to-end sweep through `GoalTrace`, `txt2bin` and the unmodified simulator
reproduces the LogGOPS cost model exactly on four of four scored instances
with no fatal guard violated. BACK-2 closes.

Identifiers registered by this closure: zero. Every registered clause was
demonstrated, so nothing was carried forward.

## Recorded as prose, not as new identifiers

- The tool has no CMake target, so a checked-in build path for it does not
  exist in either repository. Discovery therefore falls back to the ATLAHS
  submodule's own make output. Nothing in the registry claimed a build target
  for it, and `HTSIM-4` already owns the analogous `txt2bin` build target.
- Building the tool from a clean checkout needs `gengetopt` and `re2c`, which
  are not present on this machine. This run linked the tool from the
  submodule's committed object files into an external directory, so no
  submodule file was written.
- The tool's unmatched-queue diagnostics are compiled out unless the tool is
  built with `LIST_MATCH`, so the helper's queue guard cannot be relied on as
  a drain check with a default build. The positive finishing-time guard is
  what catches a deadlocked schedule today.
- `sim/LogGOPSim/LogGOPSim.cpp` divides by the flow count when it prints
  `Average FCT`, so a schedule with no completed flow prints `-nan`. The
  helper treats a nonfinite average as absent rather than propagating it.

## Reproduction

```bash
SIMLLM_LOGGOPSIM="${SIMLLM_LOGGOPSIM:?configure the LogGOPSim executable}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the txt2bin executable}" \
.venv/bin/python examples/loggopsim_helper_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/loggopsim_helper_v1"
```
