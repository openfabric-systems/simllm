# TRAF-66 finite two-batch boundary result

Status: **PROTOCOL VOID**. The boundary mechanism, event conservation and
visible movement are derived, but TRAF-66 remains open because its held-out
access ledger is not empty.

## Derived boundary form

The finite two-child service is

`T2(C, P) = max(C, P) + min(C, P) / 2`.

The pinned schedule fixes two children and zero stage offset. The coefficient
one half is therefore derived from child conservation, not fitted. Under the
selected packet-dominant 1K envelope this becomes `P + C / 2`. The derived
boundary is 681,624,980,000 ps and the total envelope is
2,967,804,740,360 to 2,967,804,742,680 ps. No dispatch phase is subtracted.

## Event conservation proof

Each of the 58 sparse layers has, per child, one dispatch launch and
completion, one combine launch and completion, and two yields. The exact
ledger is:

| Event | Per child | Both children |
|---|---:|---:|
| Dispatch launches | 58 | 116 |
| Dispatch completions | 58 | 116 |
| Combine launches | 58 | 116 |
| Combine completions | 58 | 116 |
| Yield boundaries | 116 | 232 |
| Stage advances | 117 | 234 |

With `tbo_delta_stages = 0`, the executor makes zero leading advances, runs
117 central iterations as child A then child B, makes zero trailing advances,
and merges only after both children are done. At the service level, two
`C / 2` child compute ledgers sum to `C` and two `P / 2` child packet ledgers
sum to `P`. One unlike pair overlaps in the central maximum; the other
half-service is exposed across the finite prologue and epilogue.

## Signed visible movement

The sign frozen before comparison was confirmed independently. The positive
boundary increases service, so the prediction decreases from
57,332.324550 to 44,164.630583 tokens/s/node. The visible signed relative
error moves from -0.592425% to -23.423673%, a movement of -22.831248
percentage points. The service surplus grows from exactly
`390565749501320 / 28837` ps to
`20046585297761320 / 28837` ps.

This disagrees with the tempting one-dispatch-phase correction. The derived
boundary is 50.327044 times the prior service surplus and 50.827495 frozen
dispatch phases. The source event structure calls for adding the exposed
compute half, not subtracting a dispatch phase.

An independent module reconstructed the boundary and all signs directly from
the child count, raw component services, tokens and visible target. It does
not import the boundary implementation.

## Held-out structural prediction

For both the 2K and 4K compute rows, prompt-length dependence re-enters through
`compute_service_ps / 2`. While packet service remains dominant, their total
service is structurally `packet_service_ps + compute_service_ps / 2`, so a
larger prompt-dependent compute service lowers throughput. No held-out value
is computed or compared, and no third scored run is performed here.

## Preservation and scope

All 27 prior records in the preservation-lock class are byte-identical. No
scored artifact was changed, the flagship was not rerun, decode pricing was
not touched, and no TRAF-65 or NVLink file was touched. Bulk event and
calibration records are retained under `<TRAF66_RUN_ROOT>/visible-boundary-672afcf/`.

The expectations-only freeze preceded the visible comparison, and source
inspection stayed within the committed ranges without web access. However,
an earlier whole-file read of `candidate-record.json` exposed a held-out
component row. That value was neither used nor compared, but the ledger is
nonempty. TRAF-67 owns the exact clean-repetition remainder.
