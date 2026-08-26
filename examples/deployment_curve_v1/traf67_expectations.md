# TRAF-67 clean finite-boundary repetition expectations

Status: expectations only. No calibration record field has been read for this
repetition.

## Frozen repetition

TRAF-67 repeats the committed TRAF-66 finite two-child boundary literally. It
does not amend or refit the four pinned source ranges, the two-child event
ledger, the component-service envelope or the service operator:

`max(C, P) + min(C, P) / 2`.

The expected event ledger remains 116 dispatch launches and completions, 116
combine launches and completions, 232 yields and 234 child-stage advances
across 58 sparse layers. The expected preservation class remains exactly 27
prior artifacts.

The already frozen independent sign is lower predicted throughput and a more
negative visible residual. A literal reproduction must move the visible 1K
residual from -0.592425 percent to -23.423673 percent, a movement of
-22.831248 percentage points. These are repetition targets from the committed
TRAF-66 result, not newly observed values.

## Clean exposure protocol

Before any calibration-record field access, the field-addressed reader and
this expectations file must be committed. The reader is fixed to
`comp75_calibration_result.json` and the selector
`calibration_rows[anchor_id=sglang_prefill_1k]`. It returns exactly one row,
stops at that field boundary and logs the access. It refuses any other record
path or anchor, including `candidate-record.json`, and rejects a second row
without parsing it.

The visible-access log must contain exactly one successful entry. The held-out
access ledger must remain empty. The 2K and 4K values must not be accessed or
compared.

## Scope locks

The scored flagship is not rerun. Decode pricing, TRAF-65 and the NVLink scope
are untouched. No model weights are downloaded and no web page is fetched.
