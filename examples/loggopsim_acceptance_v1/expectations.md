# LogGOPSim acceptance expectations

These expectations freeze the two clauses that keep TRAF-20 open after the
ladder study: wall-clock gain measured against an actually executed
packet-level reference on identical flow sets, and the enforced validity
envelope demonstrated live. They are committed before the acceptance
implementation or any acceptance run exists. TRAF-20 closes if and only if
every scored family here passes on the entry's registered words; a miss
keeps it open with the miss published.

## Frozen inputs and arms

- The twelve ladder flow sets, rendered THIS TIME identically for both
  arms from the pinned deployment_frontier_v1 byte partitions (record
  SHA-256
  `f2f216068bf5ba914853c62a2ee965ede0ebfc0a6f29e3d11cfa5f45eac359ad`),
  including the mixed per-flow sizes of the incast shapes (four flows of
  N and four of N minus 1 where the pinned partition says so), so the two
  arms consume byte-identical GOAL texts per shape.
- Ideal arm: the landed loggopsim-ideal level, pinned binary SHA-256
  `7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf`,
  declared parameters L = 2000 ns, o = g = O = 0, S eager, G exactly
  `0.02`, fan-in acknowledgment set for the incast shapes with the stamp
  recorded.
- Packet arm: `htsim_rnic` profile `rnic-nn` through the repository
  backend runner on the same GOAL binaries, with the deployment-frontier
  study's declared 400 Gbit/s link configuration; the executable's
  SHA-256 is recorded at run time and pinned in the results record.

## Fatal guards

- FG-1 input pinning: the pinned record hash, the loggopsim binary hash,
  and per-shape GOAL text digests shared byte-identically by both arms.
- FG-2 provenance: every measurement row carries arm, argv (portable),
  the exact G string on the ideal arm, and median-of-seven timing with
  all seven samples retained in append-only attempt directories.
- FG-3 no closed form feeds any scored value; both arms execute.
- FG-4 chronology.

## Family A: wall-clock gain (scored)

Per arm, the total wall time to price all twelve shapes is the sum of
per-shape medians of seven executions.

- A-1 (scored, floor form): packet_total divided by ideal_total is at
  least 50. The band is deliberately generous; the observed ratio is
  reported with both totals and per-shape rows. Context, not a bound:
  the ladder's twelve ideal legs measured 0.0346 s total, and a prior
  htsim diagnostic invocation measured 7.252 s for one comparable run.
- A-2 (scored): ideal_total is at most 1 s.
- A-3 (reported, unscored): per-shape gain table and machine disclosure.

## Family B: packet-reference anchoring (scored)

Per shape, the packet arm's completion time divided by the pinned
frontier record's corresponding fabric observation (isolated for the
serialized shapes, concurrent for the incast shapes) lies in
[0.98, 1.02]. This anchors the acceptance packet reference to the pinned
observations through an independent re-execution; a miss outside the
band is published and voids only this family's dependent claim (the gain
in Family A stands on its own executions).

## Family C: enforced envelope (scored)

- C-1 the incast shape without acknowledgment is REFUSED by the level
  with the diagnostic naming the unmodeled receiver per-byte gap and
  citing the ladder envelope; the refusal happens before any execution.
- C-2 the acknowledged incast run proceeds and its record carries the
  fan-in stamp and the acknowledgment.
- C-3 the serialized shape runs identically with and without the
  acknowledgment option present (byte-identical record apart from the
  recorded option), demonstrating the clean-path identity.

## Closure

On a full pass, TRAF-20 closes with its registered acceptance satisfied:
modeled error against the packet reference (the ladder M families, cited),
wall-clock gain measured here against executed packet runs on
byte-identical flow sets, and the envelope both defined (ladder) and
enforced (Family C). No packet-fidelity claim beyond the pinned record is
made; TRAF-19 and the packet rungs stay the contention authority.
