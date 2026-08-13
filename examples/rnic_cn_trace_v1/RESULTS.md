# HTSIM-2 rnic-cn trace flag results

Pre-run expectation commit: `543aa62`, which contains
[expectations.md](expectations.md) and the check-only harness and no backend
change. The registered command ran with `--check-only` before that commit and
created no artifacts.

Backend commit the evidence was authored against:
`fc4400e4ca619223481536632074045cb6af2756`. Backend commit the traced binary
was built from: `9f0637e8409d5f4c9770c222e9b51d4444798048` on
`codex/htsim2_cn_traces`, whose parent is that authored-against commit. This
study asserts no submodule pin literal and moves no gitlink. Observed SimLLM
commit for the run: `ca605b4`.

## Chronology, including a void first run

1. `543aa62` froze the expectations and the check-only harness.
2. The backend change landed and the complete backend ctest suite passed at
   358 of 358.
3. A first run was **void**: F1 failed. Two causes, both machinery.
   - The changed binary printed its observation manifest line unconditionally,
     so an untraced run of it could not print exactly what the previous binary
     printed. The frozen guard demanded that it must. The implementation was
     corrected to match the frozen guard, not the other way round: the line is
     now printed only when a trace is configured.
   - The harness gave each run its own `-completion_csv` path, so the manifest
     lines it compared embedded different option strings. F1 compares the two
     binaries "for the same GOAL and options", which the harness therefore
     never actually executed. Every run now writes one shared completion path
     and the harness moves the artifact aside afterwards.
   - While correcting that, the trace-line filter was tightened from "any line
     containing the substring trace" to "the exact observation line prefix".
     The loose form had been silently dropping the model manifest line on both
     sides, because the output directory name itself contains `trace`.
4. The second run is the reported one. No frozen relation, bound, sweep cell or
   closure clause was changed at any point. All three corrections are
   machinery, and two of them make the checks strictly stricter.

## F1: accepted baseline preserved, fatal and unscored

Passed. The changed binary with no trace flag reproduces the pre-change binary
byte for byte on the completion CSV and line for line on the manifest, for the
same GOAL and the same options. With the trace flags present, the completion
CSV is still byte-identical and the only manifest difference is the one
observation line.

## F2: conservation identities, fatal and unscored

Passed on every traced configuration. Per flow the goodput bins sum to exactly
the completion CSV's `payload_bytes`. Every switch egress port has
`Enqueued == Dequeued + Dropped` and ends at zero buffered bytes. Every flow in
the completion CSV appears in the state trace and no other flow does.

## S1: goodput bin bracket against the completion authority

Passed, sixteen of sixteen. Every flow's last carrying bin contains that flow's
`completion_time_ps`. Example: at 400 Gbit/s, flow 0 completes at 9,417,600 ps
and its last bin is `[9,000,000, 10,000,000)`.

## S2: goodput never exceeds the receiver link

**Failed, two of four.** The frozen ceiling holds at 400 Gbit/s and is exceeded
at 200 Gbit/s.

| link | bin width ps | frozen ceiling B | worst bin B | excess B |
|---|---|---|---|---|
| 400 G | 500000 | 25000 | 24576 | -424 |
| 400 G | 1000000 | 50000 | 49152 | -848 |
| 200 G | 500000 | 12500 | 16384 | +3884 |
| 200 G | 1000000 | 25000 | 28672 | +3672 |

The frozen bound is wrong in form, and this is stated as a refutation rather
than rescored. The bound treated the goodput trace as a wire-arrival rate. It
is not: the trace is binned on the receiver's in-order release instant, which
is the very semantics S1 requires so that a bin can bracket the completion
record. Release can lag behind arrival while a gap is outstanding and then
catch up, so a release-time bin total is not bounded by the link's per-bin byte
budget. The excess is 3884 and 3672 bytes, both strictly below one maximum DATA
payload of 4096 bytes, so on this fixture the release burst was one packet
deep, but a correct bound has the form `bin_width * rate / 8 + burst`, not
`bin_width * rate / 8`. That correct bound was not pre-registered and is not
claimed here.

## S3: bin refinement conserves and refines

Passed, two of two. Halving the bin width leaves every per-flow total
unchanged and takes the distinct bin start count from 9 to 17 at 400 Gbit/s and
from 16 to 31 at 200 Gbit/s. Both are exactly `2n - 1`, which is what
contiguous coarse bins must produce.

## S4: queue trace under a halved link rate

**Failed, one of two.**

- Enqueued row equality: refuted. 1266 rows at 400 Gbit/s against 1796 at
  200 Gbit/s.
- Peak backlog ordering: passed. 4224 bytes at 400 Gbit/s against 8448 at
  200 Gbit/s, exactly double, and 4224 is one 4160-byte DATA packet plus one
  64-byte control packet.

The refutation was diagnosed exactly, on the retained artifacts. Splitting the
`Enqueued` rows by wire size gives 1024 DATA rows at **both** rates and 242
against 772 control rows. So the frozen reasoning, "the packet population and
the routes do not depend on the link rate", is exactly right for DATA and
wrong for control: rnic-cn's ACK and nflow feedback is time driven, the slower
run lasts 1.65 times longer, and its derived resequencing window is also twice
as wide, so it emits more periodic control. The 1024 DATA rows are themselves
an independent structural check: 8 flows times 64 packets times 2 switch egress
hops in this Clos is exactly 1024.

## S5: state trace rates follow the configured link

Passed, two of two. `configured_rate_bps` halves exactly, 400000000000 to
200000000000. The distinct `effective_rate_bps` values are `{0, 359999999999}`
at 400 Gbit/s and `{0, 179999999999}` at 200 Gbit/s. Twice the slow value is
359999999998, one below the fast value, inside the frozen `[2r, 2r + 3]` slack
and consistent with the two nested integer floors and nothing else. The zero
entry is the pre-declaration Idle rate.

## S6: makespan scaling

Passed. 11,910,400 ps at 400 Gbit/s and 19,686,400 ps at 200 Gbit/s, a ratio of
1.653 inside the frozen `[1.5, 2.05]`.

Genuine-risk fraction across S1 to S6: `22/27`. Two frozen relations were
refuted and are reported as refuted.

## Three independent review angles

- Network and serialization physics. Each rank sends 262144 bytes, so the
  egress floor is 5.24 us at 400 Gbit/s and 10.49 us at 200 Gbit/s. Both
  makespans sit above their floor, at 2.27 and 1.88 times it. The gap is
  dominated by the 10 us control deadline and the derived resequencing window,
  and the window scales as one over the link rate, which is why the ratio is
  1.65 rather than exactly 2 and why the constant term grew rather than
  staying fixed. Predicting a fixed constant would have been wrong here, and
  the frozen band was wide enough to say so.
- Queue physics. Peak backlog doubles exactly when the rate halves, and its
  absolute value decomposes as one DATA packet plus one control packet. A
  trace reading backlog at the wrong event boundary would not produce a clean
  factor of two nor an interpretable byte total.
- System plausibility. 2 MiB across eight 400 Gbit/s endpoints in 11.9 us is
  about 176 Gbit/s of per-endpoint delivered goodput on a ring where every
  link carries exactly one flow. That is the right order for a run whose
  makespan is half constant overhead at this message size, and it is neither
  nanoseconds nor milliseconds.

## Bounding

The traced artifacts total 1.4 MB for four configurations: 3798 queue rows, 56
goodput rows and 24 state rows at the coarse bin. The row cap was set to four
million and never approached.

## Closure

HTSIM-2 registers, verbatim, "goodput/state/queue trace flags for `rnic-cn`;
they need trace hooks in the reviewed runtime first".

| registered clause fragment | evidence |
|---|---|
| "goodput ... trace flags for `rnic-cn`" | The flag pair with its all-or-nothing gate, exact per-flow conservation, S1 at 16 of 16, S3 at 2 of 2. |
| "state ... trace flags for `rnic-cn`" | The flag, per-flow presence and closure, S5 at 2 of 2. |
| "queue trace flags for `rnic-cn`" | The flag with its explicit row cap, per-port conservation and drain, the backlog half of S4, and the 1024-row DATA structural identity. |
| "they need trace hooks in the reviewed runtime first" | Seven recording points in the reviewed `rnic-cn` runtime plus the switch observer, the complete backend ctest suite at 358 of 358, and F1. |

HTSIM-2 closes. Every fragment of the registered clause is demonstrated by
evidence that survived a fatal guard, and the traced run leaves the accepted
baseline byte for byte intact.

Identifiers registered by this closure: **zero**. The two refuted relations,
S2's ceiling form and S4's enqueue-count equality, are bounds this study
invented for itself. Neither is a registered acceptance clause, and the
wave-10 residual rule reserves a new identifier for a registered clause that
was not demonstrated. Both refutations are recorded above and in the module
doc narrative instead. A reviewer who reads the goodput clause as requiring
its frozen physical ceiling should treat that clause as open; every number
needed to make that call is in this report.

## Recorded as prose, not as new identifiers

- The registered category tag for HTSIM-2 is Precision, but a trace flag is
  observation and must not move a metric. The acceptance shape used here,
  untouched baseline bytes plus exact agreement with existing authorities, is
  the correct reading of the live-reachability rule for an observation-only
  change.
- Release-time goodput and wire-arrival goodput are different quantities in
  `rnic-cn`, because resequencing separates them. The trace records release,
  which is what the completion record uses. A study that wants a wire rate
  needs the arrival instant, which the packet-attempt vocabulary already
  carries.
- The `rnic-cn` control-packet population is time driven and therefore rate
  dependent, while DATA is not. Any future comparison across link rates must
  separate the two rather than expecting one invariant packet count.

## Reproduction

```bash
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_RNIC:?configure the traced htsim_rnic}" \
SIMLLM_HTSIM_RNIC_BASELINE="${SIMLLM_HTSIM_RNIC_BASELINE:?configure the pre-change htsim_rnic}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the txt2bin executable}" \
.venv/bin/python examples/rnic_cn_trace_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/rnic_cn_trace_v1"
```

The harness exits nonzero while any scored instance fails, so this command
still exits 1 on the two refuted relations. That is deliberate: the failures
are real and the command must keep saying so.
