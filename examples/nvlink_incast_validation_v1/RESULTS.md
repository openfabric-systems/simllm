# TRAF-74 NV4 long-flow incast validation result

## Hardware against simulation

| Degree | Flow | Hardware aggregate GB/s | Simulation aggregate GB/s | Signed error | Hardware completion us by source | Simulation completion us by source | Verdict | Responsible parameter |
|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 256 KiB | 2.226087 | 93.902216 | +4118.264% | 117.760003 | 2.791670 | VOID | `undecidable_under_void_run` |
| 2 | 256 KiB | 4.571428 | 187.527541 | +4002.165% | 114.688002, 100.351997 | 2.794478, 2.795792 | VOID | `undecidable_under_void_run` |
| 3 | 256 KiB | 6.620689 | 194.301255 | +2834.759% | 115.712002, 102.399997, 101.375997 | 4.044860, 4.046174, 4.047488 | VOID | `undecidable_under_void_run` |
| 1 | 512 KiB | 3.180124 | 94.009808 | +2856.168% | 164.864004 | 5.576950 | VOID | `undecidable_under_void_run` |
| 2 | 512 KiB | 6.131736 | 187.880751 | +2964.071% | 171.008006, 149.504006 | 5.579758, 5.581072 | VOID | `undecidable_under_void_run` |
| 3 | 512 KiB | 9.365854 | 194.562756 | +1977.363% | 167.935997, 155.647993, 158.720002 | 8.081468, 8.082782, 8.084096 | VOID | `undecidable_under_void_run` |

Signed relative error is `(simulation - hardware) / hardware`; the frozen
acceptance band is [-15%, +15%]. Per-flow goodput and all seven repetitions
remain in the compact JSON record, while the table leads with the receiver
aggregate and per-source completion values that decide each cell.
For a void run, those signed differences are arithmetic diagnostics only and
receive no behavioral acceptance interpretation.

## What ran

One short exclusive `a100-hourly` cell ran the unchanged corrected TRAF-70
persistent peer-write producer on one qualified four-A100 `NV4` node. It
covered 256 KiB and 512 KiB flows at incast degrees 1, 2 and 3 with seven
repetitions per cell. The comparison uses the six predictions frozen at commit
`092080e` before Merlin job `200456` ran.

## What came out

The run status is **VOID_FATAL_GUARD**. The deciding maximum
launch-skew fraction was 10.501 percent against the frozen
10.000 percent ceiling. Fatal guard FG11 therefore failed and
the whole run is void. None of the six hardware cells
receives a pass or miss verdict.

## What it changes for the project

TRAF-74 stays open because the run is void.
The observations exercise the scored NVLink
domain at the only incast degrees this node can realize, but the void result
validates no behavioral prediction. It refutes the frozen claim that the
256 KiB degree-3 launch skew is negligible in every repetition. A future
capture needs a new expectations-only freeze with larger long-flow rungs.

## What it does not change

Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware counterpart
on an NV4 node. This result covers long flows only. Agreement at degrees 1
to 3 supports but does not prove the higher-degree extrapolation, and no
small-flow hardware validity claim follows from it.

## Fatal guards and preservation

Fatal-guard verdict: **VOID**. All 59 merged
study and scored source artifacts remain byte-identical. The raw capture stays
outside Git; this study publishes its own compact score, comparison table and
figure.
