# LogGOPSim ideal-network level results

## Outcome

What ran: the frozen `loggopsim_ideal_v1` study executed the pinned
LogGOPSim binary over every E1 through E6 exact oracle, the L1 live metric
chain, the W wall-time cells, and fatal guards FG-1 through FG-6.

What came out: the run passed, with 28 of 28 exact arithmetic oracles, 3 of
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
hardware. TRAF-20 remains open for its packet-reference acceptance clauses.

What it does not change: this study does not validate packet contention,
congestion control, per-flow completion behavior, or error against a
packet-level reference. Existing network levels and their accepted byte
artifacts remain unchanged. No task closes and no milestone moves.

## Chronology and identities

The frozen expectations are commit
`6523a625ce67ba4f8522b42a6e1e7872ba3379d0`. The level implementation is
commit `ce8f28ce4ae766f5b90f16da4128752266d361c1`, so the freeze precedes the
implementation. The run used the required binary SHA-256
`7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`.
Every scored native row records the complete argv and exact `G` string in
[results.json](results.json).

## Scored families

The evidence classes are reported separately. Fatal guards are not included
in any denominator.

| Evidence class | Family | Passed | Registered |
|---|---|---:|---:|
| exact oracle | E1 single-send parameters | 9 | 9 |
| exact oracle | E2 400 Gbit/s quantization | 7 | 7 |
| exact oracle | E3 dependency semantics | 2 | 2 |
| exact oracle | E4 ring forms | 6 | 6 |
| exact oracle | E5 guarded all-to-allv | 2 | 2 |
| exact oracle | E6 scale closed forms | 2 | 2 |
| live identity | L1 sink and metric chain | 3 | 3 |
| wall time | W speed ceilings | 3 | 3 |

The spot literals held: E1 base was 175 ns, the held E1 cell was 579 ns,
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
| W1 | four-rank 4 KiB ring all-reduce | 0.002033177 | 1 | 0.000280361 |
| W2 | 64-rank 1 MiB all-to-allv | 0.020580883 | 5 | 0.002837960 |
| W3 | 129,024-event chain | 0.516878711 | 30 | 0.071273954 |

The ratios are context only. The 7.252 s htsim invocation used a different
schedule and machine condition, so it is neither a speedup measurement nor a
packet-level error comparison.

The run host was `teferi.ethz.ch`, with an AMD Ryzen 9 3950X 16-Core
Processor, 16 physical cores, 32 logical CPUs and two threads per core. It
ran Linux 5.14 on x86-64 with Python 3.12.12.

## Fatal guards

All fatal guards held. Each guard also carried a mutation negative control
that failed the guard as intended.

| Guard | Enforcement | Outcome | Mutation control |
|---|---|---|---|
| FG-1 binary hash | runtime | held | detected |
| FG-2 GOAL hashes | runtime | held | detected |
| FG-3 full argv, exact `G`, maximum host finish | by construction | held | detected |
| FG-4 byte-identical repeated stdout | runtime | held | detected |
| FG-5 E5 separated-domain precondition | runtime | held | detected |
| FG-6 expectation chronology | runtime | held | detected |

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

## Determinism deviation and correction

An earlier retained run was void under FG-4. The 129,024-event chain produced
the same maximum host finishing time twice, but the binary's integer-second
performance banner printed `Time: 0 s` in one stdout and `Time: 1 s` in the
other. The run of record launches each repeated pair against one shared start
boundary within 20 ms after a wall-second rollover. Binary, argv, GOAL and
modeled finish time are unchanged. Every final scored stdout pair is byte
identical. The void evidence remains outside the repository.

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
table is [results.csv](results.csv). Bulk GOAL binaries and stdout remain
outside Git.

```bash
SIMLLM_LOGGOPSIM="${SIMLLM_LOGGOPSIM:?configure the pinned LogGOPSim binary}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure txt2bin}" \
.venv/bin/python examples/loggopsim_ideal_v1/run_study.py \
  --run-dir "${SIMLLM_LOGGOPSIM_RUN_ROOT:?configure an external run root}"
```
