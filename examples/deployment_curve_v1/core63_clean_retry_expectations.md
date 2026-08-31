# CORE-63 clean sparse-access retry freeze

Status: **EXPECTATIONS ONLY FOR ACCESS RETRY**. This extension is committed
before any retry. It changes only the byte-access pattern and does not amend
the frozen arithmetic, component rule, expected direction, or acceptance.

## Safe preflight disposition

The first committed reader logged three begin/end pairs contemporaneously.
The calibration-context and component projections stopped early and passed.
The forward CSV selector reached its structural guard and was rejected after
13,984 of 13,985 bytes, before the final byte could be read. It created no
basis or result, printed no value, reported no whole-file stream, and left the
forbidden-access ledger exactly empty.

This rejection is retained as preflight evidence. It is not an arithmetic
input and will not be reconstructed or removed.

## Frozen sparse selector

The retry reader obtains the CSV schema from the first line, then seeks to the
end and reads rows in reverse until it reaches the routing boundary immediately
before the selected standard-decode shape. It records physical and unique byte
counts. It rejects before the set of unique positions could cover the full
file. Selected rows are restored to original launch order before derivation.

The retry still has exactly six semantic selectors and twelve begin/end
events. Together with the safe preflight, the final evidence will contain nine
logged accesses and eighteen contemporaneous events. Whole-file streams and
held-out MTP accesses remain frozen at zero.
