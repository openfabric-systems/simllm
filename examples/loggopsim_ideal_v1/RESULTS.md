# LogGOPSim ideal-network level results

## Outcome

What ran: the frozen `loggopsim_ideal_v1` study executed the pinned
LogGOPSim binary over every E1 through E6 exact oracle, the L1 live metric
chain, the W wall-time cells, and fatal guards FG-1 through FG-6.

What came out: the corrected run passed, with 30 of 30 exact arithmetic
observables, 3 of
3 live identities and 3 of 3 wall-time ceilings in their separate evidence
classes. The deciding live number is 202,000 ps: the sink's independently
re-executed network makespan and the remote-minus-control time to first token
(TTFT) delta are both exactly 202,000 ps.

What it changes for the project: `loggopsim-ideal` is a selectable
`NetworkLevel` whose sink prices the existing graph-projected GOAL artifacts
and returns a standard `StepResult`. Its result therefore reaches TTFT and
time per output token (TPOT) through the unchanged metric reducer. The level
has exact declared parameter provenance, preserves the analytic intra-node
path, and refuses composed-native remote direct memory access (RNIC)
hardware. The corrected record replaces the under-scored record. TRAF-20
remains open for its packet-reference acceptance clauses.

What it does not change: this study does not validate packet contention,
congestion control, per-flow completion behavior, or error against a
packet-level reference. Existing network levels and their accepted byte
artifacts remain unchanged. No task closes and no milestone moves.

## Chronology and identities

The frozen expectations are commit
`6523a625ce67ba4f8522b42a6e1e7872ba3379d0`. The level implementation is
commit `ddf79f3df3c9818d02971a47f76b9cc550dc5bb1`, so the freeze precedes the
implementation. The run used the required binary SHA-256
`7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`.
Every scored native row records a portable complete argv and exact `G` string
in [results.json](results.json). The resolved argv lives with the retained
stdout in the external attempt directory.

## Scored families

The evidence classes are reported separately. Fatal guards are not included
in any denominator.

| Evidence class | Family | Passed | Registered |
|---|---|---:|---:|
| exact oracle | E1 single-send parameters | 11 | 11 |
| exact oracle | E2 400 Gbit/s quantization | 7 | 7 |
| exact oracle | E3 dependency semantics | 2 | 2 |
| exact oracle | E4 ring forms | 6 | 6 |
| exact oracle | E5 guarded all-to-allv | 2 | 2 |
| exact oracle | E6 scale closed forms | 2 | 2 |
| live identity | L1 sink and metric chain | 3 | 3 |
| wall time | W speed ceilings | 3 | 3 |

The spot literals held: E1 base was 175 ns. At the protocol boundary, `S=50`
produced hosts 0 and 1 at 277 ns, while `S=51` produced host 0 at 10 ns and
host 1 at 277 ns. The held E1 cell was 579 ns.
E2 crossed from 0 ns at 50 bytes to 1 ns at 51 bytes, E3 `requires` and
`irequires` were 140 ns and 123 ns, the two E4 all-reduces were 1,050 ns and
39,486 ns, E5 was 1,462 ns and 13,569 ns, and E6 was 4,422,432 ns for the
chain and 1,408,731 ns for the all-to-allv.

## L1 live metric chain

The remote step emitted one fabric artifact. Re-executing that artifact
independently with the recorded argv produced 202,000 ps, exactly equal to
the sink's network makespan. The remote step completed at 203,000 ps and its
zero-collective control at 1,000 ps, so their TTFT difference was exactly the
202,000 ps network makespan.

The derived per-byte gaps were exact strings: 400,000,000,000 bit/s mapped
to `0.02` ns/byte and 200,000,000,000 bit/s mapped to `0.04` ns/byte. All six
LogGP parameters carry `DECLARED` evidence. The defaults are `o=0`, `g=0`,
`O=0`, and `S=9223372036854775807`; the sink rejects a rendered payload above
`S` before invoking the binary.

## Wall-time class

Each value is the median of seven executions.

| Cell | Schedule | Median s | Ceiling s | Unscored ratio to 7.252 s |
|---|---|---:|---:|---:|
| W1 | four-rank 4 KiB ring all-reduce | 0.002313387 | 1 | 0.000319000 |
| W2 | 64-rank 1 MiB all-to-allv | 0.020375621 | 5 | 0.002809655 |
| W3 | 129,024-event chain | 0.325131915 | 30 | 0.044833414 |

The ratios are context only. The 7.252 s htsim invocation used a different
schedule and machine condition, so it is neither a speedup measurement nor a
packet-level error comparison.

The run host was `teferi.ethz.ch`, with an AMD Ryzen 9 3950X 16-Core
Processor, 16 physical cores, 32 logical CPUs and two threads per core. It
ran Linux 5.14 on x86-64 with Python 3.12.12.

## Fatal guards

All fatal guards held. Each mutation control invoked the same predicate as its
guard with the named input changed, and every predicate rejected its mutant.

| Guard | Enforcement | Outcome | Mutation control |
|---|---|---|---|
| FG-1 binary hash | runtime | held | predicate exercised, mutant rejected |
| FG-2 GOAL hashes | runtime | held | predicate exercised, mutant rejected |
| FG-3 full argv, exact `G`, maximum host finish | runtime | held | predicate exercised, mutant rejected |
| FG-4 byte-identical repeated stdout | runtime | held | predicate exercised, mutant rejected |
| FG-5 E5 separated-domain precondition | runtime | held | predicate exercised, mutant rejected |
| FG-6 expectation chronology | runtime | held | predicate exercised, mutant rejected |

The run would be void if any guard failed. These rows are therefore not a
score and are not added to the family denominators.

## Physical sanity

- Serialization physics: the 64-rank 1 MiB all-to-allv completed in
  1,408,731 ns, above its 1,321,173 ns serialization floor and below its
  1,730,673 ns serial ceiling.
- Bandwidth scaling: halving the declared rate from 400 to 200 Gbit/s changed
  `G` from `0.02` to `0.04` ns/byte, exactly a factor of two.
- End-to-end plausibility: the modeled all-to-allv takes 1.409 ms, close to
  the serialization-dominated lower bound and inside the independently
  stated physical bracket. This supports the ideal serialization mechanism,
  not packet-level fidelity.

## Corrections and prior void disclosure

The earlier scoring record counted only each E1 cell's host maximum. That made
the `S=50` rendezvous cell and `S=51` eager cell score the same 277 ns value.
The corrected record scores the four frozen host values individually. This
raises E1 from 9 to 11 observables and the exact class from 28 to 30. Every
modeled maximum, host time, live identity and wall-cell finish reproduced
identically.

The earlier run under FG-4 remains void and is not rescored. Its 129,024-event
chain produced the same modeled maximum twice, while the integer-second
performance banner differed between `Time: 0 s` and `Time: 1 s`. Contrary to
the prior RESULTS claim, those differing stdout bytes were not retained. Only
their SHA-256 hashes reached the committed record. Five of the six legacy run
directories also had no verdict of their own because later attempts overwrote
the single record destination. The prior retention claim is withdrawn.

The corrected protocol creates a fresh `attempt-N` directory, refuses to
start a later attempt until every earlier attempt has a verdict, and writes
stdout plus resolved-argv provenance as each native execution completes. The
corrected `attempt-1` contains 85 stdout files, 85 matching invocation records
and its own byte-identical `verdict.json`. Every repeated scored pair is byte
identical. The FG-4 control now passes one real captured stdout and a perturbed
copy through the byte comparator. FG-3 removes `-G` from a real recorded argv
and invokes the same argv validator. The FG-1 early-void path hashes an actual
perturbed binary byte sequence and invokes the binary-hash predicate.

Tracked argv now names `LogGOPSim` and records each GOAL path relative to the
attempt directory. Resolved machine paths stay only in the external attempt
provenance. Both `--binary` and `--txt2bin` reach the exact-oracle, live L1 and
wall paths; a flag-only process run with both discovery environment variables
removed passed this same 30-observable study.

## TRAF-20 verdict

TRAF-20 stays open. Its registered entry after this result is:

> - TRAF-20 (Precision; P2; M): qualify the delivered `loggopsim-ideal`
>   fast level for schedule-shape studies that do not need per-flow transport
>   behavior. The level prices the existing GOAL artifacts through LogGOPSim
>   and bypasses the event-driven RNIC path. Its exact arithmetic, live metric
>   wiring and generous speed ceilings are validated by
>   `examples/loggopsim_ideal_v1`. Remaining acceptance must measure wall-clock
>   gain and modeled error against the packet-level reference on identical
>   schedules, then define and enforce the envelope of questions the level
>   cannot answer.

The residual remains TRAF-20 because it is the unfulfilled acceptance surface
of that task, not a distinct deferred feature. No new TRAF identifier is
registered.

## Reproduction

The run record is [results.json](results.json), and the flattened evidence
table is [results.csv](results.csv). Bulk GOAL binaries, stdout, resolved argv
and per-attempt verdicts remain outside Git.

```bash
.venv/bin/python examples/loggopsim_ideal_v1/run_study.py \
  --run-dir "${SIMLLM_LOGGOPSIM_RUN_ROOT:?configure an external run root}" \
  --binary "${SIMLLM_LOGGOPSIM:?configure the pinned LogGOPSim binary}" \
  --txt2bin "${SIMLLM_TXT2BIN:?configure txt2bin}"
```
