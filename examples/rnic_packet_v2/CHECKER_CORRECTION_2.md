# BACK-34 checker correction 2

The first registered BACK-34 result run reached and passed all three raw
compatibility relations before the fatal-unscored partial-tail oracle ran:
the accepted ABI-v2 full-quantum projection and both accepted ABI-v1 artifact
identities were unchanged. The tail oracle then exposed that
`back34_expectations.json` used one `tx_started_at_ps` and one
`tx_finished_at_ps` field for two boundaries that the frozen prose already
distinguished.

The unbound Tier A producer serializes the second packet from 82,920 through
103,560 ps. The composed packetized runtime reserves a full source quantum,
right-aligns the short second packet, and serializes it from 144,200 through
164,840 ps. `BACK34_EXPECTATIONS.md` already states both sequences. This
correction replaces the ambiguous pair with explicit `tier_a_*` and
`composed_*` fields and routes the Tier A checker to the former.

No scored relation, band, denominator, payload geometry, RX boundary, WQE
boundary, implementation, or accepted reference changes. The failed run did
not write `results.json`; its complete external directory is retained as
`back34-failed-shared-tx-field`.
