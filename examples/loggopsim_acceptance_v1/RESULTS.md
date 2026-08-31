# LogGOPSim acceptance results

## Outcome

**The study is a valid REFUTATION.** The packet arm took 1.088866981 seconds
and the ideal arm took 0.029767114 seconds when each total is the sum of the
twelve per-shape medians of seven executions. The measured gain is therefore
36.579528x, below the frozen A-1 floor of 50x. A passes one of two scored
predicates, B passes all twelve packet anchors, and C passes all three envelope
predicates. All four fatal guards held and rejected their end-to-end mutants,
so the failed speed predicate is interpretable.

What ran: `loggopsim_acceptance_v1` rendered the twelve pinned frontier byte
partitions once per shape, converted each to one binary GOAL, and executed that
same binary seven times through both the `loggopsim-ideal` and `rnic-nn`
repository runners at the frozen 400 Gbit/s configuration.

What came out: the one deciding number is 36.579528x against a required 50x.
The packet completion of every shape exactly reproduced its pinned frontier
observation, and default refusal, acknowledged fan-in, and clean-path identity
all behaved as frozen.

What it changes for the project: TRAF-20 stays open because its wall-clock
qualification is refuted on the disclosed machine and binaries. Its packet
anchoring and enforcement clauses now have passing live evidence, but the
projected full speed qualification does not become literal and no milestone
moves.

What it does not change: the frontier ladder M-1, M-2, and M-3 modeled-error
results stand; TRAF-19 and the packet rungs remain the contention authority;
the default fan-in refusal remains active; and no packet fidelity or silicon
accuracy beyond the pinned frontier record is claimed.

## Frozen authority and execution

- Expectations-only commit: `30a9af9dd6e424b1458eff8a0f97598efe5ebd03`.
- Implementation commit: `6b3da37a3218273f40d4eaac5339bfa093a38d97`.
- Pinned frontier record SHA-256:
  `f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad`.
- LogGOPSim SHA-256:
  `7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`.
- htsim_rnic SHA-256:
  `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d`.
- txt2bin SHA-256:
  `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b`.
- Ideal parameters: `L=2000 ns`, `o=g=O=0`, eager `S`, and literal
  `G=0.02 ns/byte`.
- Packet parameters: `rnic-nn`, `linkspeed_bps=400000000000`.
- Bulk evidence identifier: `p2l-t2c/attempt-1`.

The attempt retains 87 ideal stdout/stderr pairs, 84 packet stdout/stderr
pairs, 171 portable argv manifests, all 84 packet completion CSV files, the
twelve GOAL texts and binaries, and its own verdict. The tracked result is
byte-identical to the attempt verdict.

## Family tallies

| Family | Passed | Scored | Verdict |
|---|---:|---:|---|
| A, wall-clock gain | 1 | 2 | REFUTED |
| B, packet-reference anchoring | 12 | 12 | PASS |
| C, enforced envelope | 3 | 3 | PASS |

Fatal guards are unscored. FG-1 through FG-4 all held, and every guard's
mutation control exercised the same predicate and was rejected.

## Family A: wall-clock gain

A-1 is the sole miss. A-2 passes because 0.029767114 seconds is below the
one-second ceiling. A-3 is reported and unscored.

| Shape | Ideal median (s) | Packet median (s) | Packet / ideal |
|---|---:|---:|---:|
| serialized-b1 | 0.002493461 | 0.033497749 | 13.434238 |
| serialized-b2 | 0.002300717 | 0.036671650 | 15.939227 |
| serialized-b4 | 0.002492242 | 0.039255219 | 15.750966 |
| serialized-b8 | 0.002446554 | 0.047252188 | 19.313773 |
| serialized-b16 | 0.002306817 | 0.061269309 | 26.560108 |
| serialized-b32 | 0.002511191 | 0.089867152 | 35.786665 |
| incast-b1 | 0.002437663 | 0.041912736 | 17.193819 |
| incast-b2 | 0.002532670 | 0.051587942 | 20.368995 |
| incast-b4 | 0.002560439 | 0.070440841 | 27.511236 |
| incast-b8 | 0.002597289 | 0.106312886 | 40.932251 |
| incast-b16 | 0.002625348 | 0.182431228 | 69.488398 |
| incast-b32 | 0.002462723 | 0.328368081 | 133.335369 |

The frozen context cited one earlier comparable packet invocation at 7.252
seconds. It was context rather than a bound. The present packet binary prices
the entire twelve-shape set in 1.088866981 seconds, so that context does not
describe this executable on this machine. The miss is not repaired or
reinterpreted after observation.

## Family B: packet-reference anchoring

Every re-executed packet completion equals its pinned observation, so every
published quotient is exactly 1.0 inside the frozen `[0.98, 1.02]` band.

| Shape | Packet completion (ps) | Pinned observation (ps) | Quotient |
|---|---:|---:|---:|
| serialized-b1 | 137201000 | 137201000 | 1.000000000 |
| serialized-b2 | 272317000 | 272317000 | 1.000000000 |
| serialized-b4 | 542551000 | 542551000 | 1.000000000 |
| serialized-b8 | 1083018000 | 1083018000 | 1.000000000 |
| serialized-b16 | 2163953000 | 2163953000 | 1.000000000 |
| serialized-b32 | 4325821000 | 4325821000 | 1.000000000 |
| incast-b1 | 242356000 | 242356000 | 1.000000000 |
| incast-b2 | 482629000 | 482629000 | 1.000000000 |
| incast-b4 | 963174000 | 963174000 | 1.000000000 |
| incast-b8 | 1924264000 | 1924264000 | 1.000000000 |
| incast-b16 | 3845860000 | 3845860000 | 1.000000000 |
| incast-b32 | 7689053000 | 7689053000 | 1.000000000 |

The tracked record also publishes all seven sample quotients for every row.
All seven are 1.0 in every shape.

## Family C: enforced envelope

- C-1 passes. Unacknowledged incast is refused before native execution. The
  diagnostic names the unmodeled receiver per-byte gap and cites the frontier
  ladder. The acknowledged-option mutation reaches the armed execution
  boundary, proving that the refusal is doing work.
- C-2 passes. Acknowledged incast executes, reports physical quiescence, and
  records both `fan_in_detected=true` and `acknowledged=true`.
- C-3 passes. The serialized shape executes with the option absent and present.
  Its canonical records are byte-identical after removing only
  `acknowledge_fan_in_option`; that option is the sole differing field.

The CI refusal cell needs no binary because the rejection occurs before GOAL
conversion or execution. The three executed C cells run conditionally when
the pinned LogGOPSim and txt2bin tools are present, otherwise pytest reports an
explicit skip reason.

## Physical sanity

The implementation commit fixed these checks before any measured value was
read.

- Floor: payload bytes times 20 ps/byte is the minimum serialization time at
  400 Gbit/s through the shared destination ingress.
- Ceiling: each 4096-byte payload packet occupies a 4160-byte wire slot, plus
  the declared 2 microseconds of propagation, at most one 83.2 ns slot per
  flow, and 1 ns of reported-time quantization.

All twelve packet observations lie between those cell-specific bounds. The
batch-32 over batch-16 incast ratios are 1.995790 on the ideal arm and 1.999307
on the packet arm, within the pre-run `[1.9, 2.1]` scaling check. At batch 32,
packet over ideal is 1.015637 for serialized traffic and 8.110405 for incast.
That separation is physically coherent with one clean link versus eight flows
sharing the receiver ingress, but it does not rescue the failed software
wall-clock criterion.

## TRAF-20 verdict

TRAF-20 stays open. The ladder M families continue to satisfy its modeled
error clause, and this study satisfies the packet anchoring and enforcement
clauses. The frozen 50x speed floor is not met, so the full registered
acceptance is refuted and the task cannot close from this result.
