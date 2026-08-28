# LogGOPSim ideal-network level expectations

These expectations freeze the first TRAF-20 slice: LogGOPSim as a selectable
ideal-network level, wired so the step sink can price the same GOAL
artifacts it renders today through the LogGOP cost model instead of the
packet backend, reaching TTFT and TPOT through the unchanged metric chain.
They are committed before any implementation of the level exists. The
tool's exact arithmetic was established by a pre-freeze source audit of the
pinned binary (option grammar, verbose state transitions, one-at-a-time
parameter fits, held-out prediction); this freeze pins the audited forms.

The level is an idealized network by construction: no contention beyond the
LogGOP resource model, no packets, no congestion control. Nothing here
claims packet-level fidelity; the packet rungs remain the authority for
contention effects.

## Pinned binary

- Executable identity: `goalsim 0.1`, SHA-256
  `7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`,
  GNU build ID `74845b64804d53600d13ca6d80d0829e40426074`. The run of
  record uses a binary with exactly this SHA-256, supplied via
  `SIMLLM_LOGGOPSIM`. A rebuilt binary from the backend submodule is
  admissible for a supplementary arm only if it reproduces every exact
  family below; any deviation is disclosed and the pinned binary decides.
- Network type is always `LogGP` (`-n LogGP`).
- The metric is the maximum host finishing time. `Average FCT` is never
  scored; the audit showed it does not track `G`.
- The `G` argument is preserved as its exact decimal string in every
  record. The printed banner rounds `G` to six decimals and must never be
  used to recover it.

## Frozen inputs

Tracked GOAL texts under `goals/` with SHA-256 pinned in
[goals.sha256](goals.sha256) (generated at freeze time; hash mismatch is
fatal). Binary GOALs are produced by `txt2bin` at run time. Two bulk
schedules are digest-pinned rather than tracked: the study re-renders them
with the repository GOAL writers from the frozen generator specs below and
their text SHA-256 must equal
`e96a335a11061bc2191834e02b11bd937a992dc2c43af4628454de506fb1235f`
(`alltoall-p64-s1048576`: 64 ranks, pairwise all-to-allv, 1,048,576 bytes
per pair, no dependencies, all operations on CPU 0 and NIC 0) and
`9ba680241997c0e1af2913e85fba48184b9a02d2d08c2c7f1ad88c9a9ea29338`
(`chain-p64-l24-s4096`: 8 independent TP8 groups, 24 layers, 2 ring
all-reduce sites per layer, 14 dependent ring rounds per collective,
4,096 bytes per send, 43,008 sends, 85,888 requires edges).

## Frozen arithmetic

With `s` the payload in bytes and integer nanosecond parameters
(`G` a binary64 in nanoseconds per byte):

    d_G = floor_binary64((s - 1) * G)
    d_O = (s - 1) * O
    single_send_ns = L + 2*o + g + d_G + 2*d_O

The floor applies once per message, never accumulated per byte and never
merged across a chain (`k` dependent steps cost `k` independent floors).
`requires` starts the dependent operation exactly at the predecessor's
finish and `irequires` at its start, with zero scheduling surcharge; only
shared-resource availability may delay further. `s <= S` selects eager
(sender completes locally), `s > S` rendezvous. The repository driver
multiplies host times by exactly 1000 to picoseconds.

## Fatal guards (violation voids the run)

- FG-1 binary hash equals the pinned SHA-256 before any cell runs.
- FG-2 every tracked and re-rendered GOAL matches its pinned SHA-256.
- FG-3 every scored value is a maximum host finishing time from a run
  whose full argv is recorded, including the exact `G` string.
- FG-4 determinism: each scored cell executes twice and both stdouts are
  byte-identical.
- FG-5 the all-to-allv separated-domain precondition
  `L + o + d_G >= 2*max(o, g + d_G) + o` holds in every E5 cell (the
  audited formula is invalid outside it).
- FG-6 chronology: RESULTS cites this file's commit, which precedes every
  implementation commit of the level.

## Family E1: single-send one-at-a-time (exact, scored)

On `pair-17.goal` (`s=17`) with base
`L=100, o=10, g=7, G=3, O=0, S=50`: 175 ns. Then exactly one change each:
`L=130` gives 205; `o=13` gives 181; `g=11` gives 179; `G=5` gives 207;
`O=2` gives 239. Protocol boundary on `pair-51.goal` (`s=51`) with the
base parameters: `S=50` (rendezvous) gives host maximum 277 with both
hosts at 277; `S=51` (eager) keeps host maximum 277 with host 0 at 10.
Held-out combined cell on `pair-64.goal`:
`L=137, o=19, g=11, G=0.25, O=3, S=64` gives exactly 579 ns
(prediction `137 + 38 + 11 + floor(63*0.25) + 2*63*3`).

## Family E2: 400 Gbit/s quantization (exact, scored)

With `L=0, o=0, g=0, O=0, S=10000000, G` passed as the exact string
`0.02` (the 400 Gbit/s per-byte gap in nanoseconds), host maximum equals
`floor((s-1)*0.02)` exactly:

| GOAL | s bytes | expected ns |
|---|---:|---:|
| pair-50 | 50 | 0 |
| pair-51 | 51 | 1 |
| pair-64 | 64 | 1 |
| pair-101 | 101 | 2 |
| pair-4096 | 4096 | 81 |
| pair-65536 | 65536 | 1310 |
| pair-1048576 | 1048576 | 20971 |

## Family E3: dependency semantics (exact, scored)

With `L=101, o=999, g=13, G=0.02, O=7, S=5`: `dep-requires.goal`
(calc 123 gating calc 17 on distinct CPUs) completes at exactly 140 ns;
`dep-irequires.goal` at exactly 123 ns. The 999 ns overhead must not
appear in either result.

## Family E4: ring forms (exact, scored)

With `O=0` define `C = L + 2*o + g + d_G`. Setting A
(`L=100, o=10, g=7, G=3, s=17`): `C=175`; `ring-round-s17` gives 175,
`ring-chain-s17` gives 700, `ring-allreduce-p4-s17` gives 1050. Setting B
(`L=2500, o=1500, g=1000, G=0.02, s=4096`): `d_G=81`, `C=6581`;
`ring-round-s4096` gives 6581, `ring-chain-s4096` gives 26324,
`ring-allreduce-p4-s4096` gives 39486.

## Family E5: guarded pairwise all-to-allv (exact, scored)

With `O=0`, `S >= s`, define `d=d_G`, `p=max(o, g+d)`, `r=o+g`,
`a=L+o+d`; under FG-5 the four-rank completion is
`a + max(2*p, r) + 2*r`. Setting A
(`L=1000, o=80, g=7, G=3, s=17, S=50`... `S=50 >= 17`): exactly 1462 ns.
Setting B (`L=10000, o=100, g=50, G=0.25, s=4096, S=65535`): exactly
13569 ns.

## Family E6: scale closed form (exact, scored)

With `L=2500, o=1500, g=1000, G=0.02, O=0, S=65535`: the re-rendered
`chain-p64-l24-s4096` (129,024 events) completes at exactly
`24 * 2 * 14 * 6581 = 4,422,432` ns, and the re-rendered
`alltoall-p64-s1048576` completes at exactly 1,408,731 ns. Physical
sanity, stated before reading: the all-to-allv sits between its
serialization floor `63 * floor(1048575*0.02) = 1,321,173` ns
(plus fixed terms) and the serial ceiling `63 * 27,471 = 1,730,673` ns.

## Family L1: live chain (exact identities, scored)

The new level is selected for the step sink on a fixed small step with one
remote collective (the implementation defines the step; the relations are
frozen):

- L1a: the `StepResult` network makespan equals the pinned binary's
  maximum host finishing time on the sink's own emitted GOAL artifact,
  re-executed independently by the harness with the recorded argv, to
  0 ps.
- L1b: the step's TTFT differs from an identical zero-collective control
  step by exactly that makespan.
- L1c: parameter derivation from the declared fabric: for
  400e9 bits per second the derived `G` string is exactly `0.02`, and for
  200e9 exactly `0.04` (`G_ns_per_byte = 8e9 / rate_bits_per_second`,
  rendered as an exact shortest decimal); the derivation is DECLARED
  evidence and the mapping is recorded in the level's provenance stamp.

## Family W: wall time (scored, generous, median of 7, machine disclosed)

- W1: `ring-allreduce-p4-s4096` completes in at most 1 s
  (audited median 2.46 ms).
- W2: the re-rendered 64-rank 1 MiB all-to-allv completes in at most 5 s
  (audited median 20.3 ms).
- W3: the re-rendered 129,024-event chain completes in at most 30 s
  (audited median 0.391 s with spread to 0.902 s).
- Reported unscored: the ratio of each to the 7.252 s htsim diagnostic
  invocation baseline, cited as context only; the schedules and machine
  conditions are not identical, and the record says so.

## Level contract (fatal unscored where by construction, else scored)

- Every existing network level's accepted artifacts stay byte-identical
  when the new level is not selected; the existing byte-lock tests must
  pass unchanged.
- Selecting the level with composed-native RNIC hardware is refused with
  a diagnostic naming both seams.
- The level's provenance records the binary SHA-256, full argv per
  invocation, and the exact `G` string.

## Closure

This study validates the LogGOPSim level's exact arithmetic on pinned
schedules, its wiring into the metric chain, its parameter derivation and
its speed class. It does not validate packet-level fidelity, does not
price contention, and does not by itself close TRAF-20 beyond the words
its registered entry actually contains; the implementation quotes that
entry and scopes the closure or residual accordingly. Scored families are
E1 through E6, L1 and W, reported in their classes and never summed with
fatal rows.
