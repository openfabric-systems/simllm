# TRAF-74 NV4 long-flow incast second-capture result

## Hardware against simulation

| Degree | Flow | Hardware aggregate GB/s | Simulation aggregate GB/s | Signed error | Hardware completion us by source | Simulation completion us by source | Maximum launch skew | Budget | Verdict | Responsible parameter |
|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| 1 | 4 MiB | 4.461874 | 94.104154 | +2009.073% | 940.032005 | 44.570870 | 0.000% | 10.000% | MISS | `packetization` |
| 2 | 4 MiB | 8.914037 | 188.190903 | +2011.175% | 941.056013, 929.791987 | 44.573678, 44.574992 | 0.544% | 10.000% | MISS | `packetization` |
| 3 | 4 MiB | 13.760359 | 194.792148 | +1315.604% | 911.360025, 900.095999, 900.095999 | 64.593980, 64.595294, 64.596608 | 1.129% | 10.000% | MISS | `packetization` |
| 1 | 8 MiB | 4.561247 | 94.110900 | +1963.271% | 1839.104056 | 89.135350 | 0.000% | 10.000% | MISS | `packetization` |
| 2 | 8 MiB | 9.061947 | 188.213096 | +1976.961% | 1851.392031, 1828.863978 | 89.138158, 89.139472 | 0.275% | 10.000% | MISS | `packetization` |
| 3 | 8 MiB | 14.288373 | 194.808553 | +1263.406% | 1761.279941, 1747.967958, 1746.943951 | 129.179708, 129.181022, 129.182336 | 0.574% | 10.000% | MISS | `packetization` |

Signed relative error is `(simulation - hardware) / hardware`; the frozen
acceptance band is plus or minus 16 percent. Each cell requires its aggregate
and every per-source median to be inside that band. Fatal guards remain
separate and never enter the behavioral count.

## What ran

One short exclusive `a100-hourly` cell ran the unchanged corrected TRAF-70
persistent peer-write producer on one qualified four-A100 `NV4` node. It
covered 4 MiB and 8 MiB flows at incast degrees 1, 2 and 3 with seven
repetitions per cell. The comparison uses the six predictions frozen at
commit `b21ba82` before Merlin job `202466` ran. The scored module version is
`65593131a0448d2b33f51018d5972c918dad3493` with flow policy
`release_aware_round_robin`.

## What came out

The run status is **VALID_0_PASS_6_MISS**. The maximum observed
launch-skew fraction was 1.129 percent against the 10.000 percent ceiling.
The deciding worst absolute signed relative error was
2011.175 percent. 0 of 6 cells pass and 6 miss.
Every miss names `packetization` under the frozen size-dependent
attribution rule.

## Physical sanity before precision

Floor: packetized wire serialization sets frozen per-cell floors from
44.564480 to 129.108836 us. Hardware completion ranged from
900.095999 to 1851.392031 us and was never faster than
its floor. The closest sample was 13.531 times its floor.

Ceiling: every source completed below the frozen 5000 us observed-producer
ceiling. The slowest used 37.028 percent of that ceiling.

Byte scaling: doubling each source from 4 MiB to 8 MiB moved median
completion by 1.933 to 1.967 times, close
to the expected factor of two for sustained service.

End-to-end plausibility: measured per-source payload goodput ranged from
4.457 to 4.802 GB/s. That extends the
retained 2.2 to 3.5 GB/s short-rung trend after fixed launch work is amortized,
but it remains far below the model's packetized wire-rate prediction.

## What it changes for the project

TRAF-74 closes as a completed non-void validation; TRAF-86 owns the identified model precision residual.
The second capture supplies all six literal
per-cell verdicts at the only incast degrees an NV4 node can realize.

## What it does not change

Degrees 4, 8 and 16 remain DECLARED SIMULATION with no hardware counterpart
on an NV4 node. This result covers long flows only. Agreement at degrees 1
to 3 supports but does not prove the higher-degree extrapolation, and no
small-flow hardware validity claim follows. The first frozen capture remains
byte-identical and void; this result does not reinterpret job `200456`.

## Fatal guards and preservation

Fatal-guard verdict: **PASS**. All 71 inherited and first-capture
artifacts remain byte-identical. The digest-complete raw capture stays outside
Git and retains every checksum, ordering, per-link data and raw counter,
replay, recovery, throttle, topology and competing-process observation.
