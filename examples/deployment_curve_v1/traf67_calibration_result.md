# TRAF-67 clean finite-boundary repetition result

Status: **PASS CLEAN REPETITION**. TRAF-67 reproduces the frozen TRAF-66
boundary literally and closes both tasks under the registry's clean-pass
clause.

## Access ledger

The visible-access ledger contains exactly one successful entry for
`calibration_rows[anchor_id=sglang_prefill_1k]` in the COMP-75 calibration
record. The committed streaming reader consumed 2,193 of 8,415 bytes and
stopped at the end of that single row. It did not load the whole record or
parse a second row. The reader and expectations were committed at `9177890`
before this access.

The held-out access ledger is empty. No 2K or 4K numeric value was accessed or
compared, and `candidate-record.json` was not opened by the TRAF-67 study.

## Reproduced signed movement

The independent signer reproduces lower predicted throughput and a more
negative visible residual. The prediction moves from 57,332.324550 to
44,164.630583 tokens/s/node. The 1K signed relative error moves from
-0.592425 percent to -23.423673 percent, a movement of -22.831248 percentage
points.

This is the frozen TRAF-66 movement, reproduced from the single visible row.
No coefficient was amended or refit.

## Preservation locks

All 27 prior-artifact SHA-256 locks pass byte-identically. The complete digest
ledger is in the external event ledger; the published result keeps the lock
count, status and mutation flag.

## Frozen boundary and events

The reused boundary is

`max(C, P) + min(C, P) / 2`.

The packet-dominant 1K branch therefore remains `P + C / 2`, with boundary
service 681,624,980,000 ps and total service envelope 2,967,804,740,360 to
2,967,804,742,680 ps. Across 58 sparse layers, the unchanged two-child ledger
conserves 116 dispatch launches and completions, 116 combine launches and
completions, 232 yields and 234 child-stage advances.

## Scope and disposition

No scored flagship was rerun. Decode pricing, TRAF-65 and the NVLink scope are
untouched. No model weight was downloaded and no web page was fetched. The
bulk access, event and calibration ledgers are retained under
`<TRAF67_RUN_ROOT>/visible-boundary-9177890/`.

The clean pass closes TRAF-66 and TRAF-67. The third scored run alone owns the
held-out comparison. TRAF-68 remains the next reserved traffic ID.
