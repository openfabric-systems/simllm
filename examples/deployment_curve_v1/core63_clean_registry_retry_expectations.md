# CORE-63 clean registry-selector retry freeze

Status: **EXPECTATIONS ONLY FOR REGISTRY SELECTOR RETRY**. This freeze exists
before the next access. It changes no arithmetic, field values, component
classification, signed direction, preservation lock, or acceptance rule.

## Third safe preflight

The terminal-byte CSV selector passed with exactly 13,984 of 13,985 unique
bytes and left the final byte untouched. The subsequent registry lookup
rejected at 99,635 of 99,636 bytes because the file does not spell CORE-63 as
a pipe-table cell. No result or basis was written, no source value was printed,
and no whole-file stream or MTP access occurred. Its eight-event ledger is
preserved unchanged.

## Frozen selector correction

The registry selector now stops at the first line containing the exact ASCII
task identifier `CORE-63` or `CORE-64`. The change removes only the unverified
pipe-table punctuation assumption. Registry text is not an arithmetic input.
The selector still has the one-byte-short whole-file guard and contemporaneous
begin/end logging.

The earlier statement that the terminal-byte plan was the last access attempt
is superseded only because that attempt self-terminated at the registry syntax
guard. Across three safe preflights and the final six-selector tranche, the
publication will contain sixteen logged accesses and thirty-two events. The
forbidden-access ledger remains empty and whole-file streams remain zero.
