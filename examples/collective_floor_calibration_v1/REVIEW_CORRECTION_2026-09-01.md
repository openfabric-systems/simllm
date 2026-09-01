# Collective floor evidence correction

This record maps the 2026-09-01 adversarial review findings onto the immutable
completion evidence. It introduces no new measurement, fit, model or score.
The detailed correction is appended to
[COMPLETION_RESULTS.md](COMPLETION_RESULTS.md).

## Superseded classification

The 63-of-63 Family H result is a post-specified regression on an adaptively
reused 63-cell evaluation set with training-cell-only numeric evaluation. It
is not an untouched holdout qualification. The training and evaluation
identities are mechanically disjoint, but the model form and targeted branches
were chosen after results from the same evaluation cells had been observed
across attempts.

The following immutable files retain their original bytes:

- `expectations_v6.md`
- `study_config_v4.json`
- `run_completion.py`
- `completion_record.json`
- `completion_results.csv`

Their `qualifying` and `holdout` labels describe the frozen attempt chronology
and mechanical split. This correction supersedes those labels as claims about
independence. The record's 63 passes, error statistics, fatal guards and
digests remain unchanged. TRAF-84 owns a future validation on genuinely
independent H200 cells.

## Previously undisclosed consumer consequence

The published MiniMax EP 8 PASS is valid only under its legacy authority pin.
Binding the successor authority at the exact rank-8, 196,608-byte,
65-repeat coordinate gives quotient 0.946736591 against the MiniMax frozen
packet-over-external floor of 1.0, so the successor refutes it. MiniMax Family
D is 0 of 3 under that binding. EP 32 and EP 128 remain refuted legacy rank-8
donor transfers, and the successor rejects their unfitted ranks.

The exact reproduction of all 16 published MiniMax queries proves legacy
preservation only. It does not validate the successor binding. TRAF-83 owns an
explicit supersede-and-republish event with unchanged bands. This correction
does not silently change the default authority.
