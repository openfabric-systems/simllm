# Packet-v2 checker correction chronology

The expectation freeze is SimLLM commit
`506f87af93687ccf0df85f6b5307b71a20ed3762`. Its prose defines the exact
packet oracle for a single WQE and separately requires the inherited two-WQE
FIFO rows. The frozen checker mistakenly used the cell doorbell delay as the
TX base for both WQEs in a FIFO cell. The second WQE is not eligible for the
capacity-one port until the first extent terminates, so that assertion would
require a packet issue before network acceptance. It also conflicts with the
frozen inherited FIFO oracle, where each WQE's `port_tx_at_ps` is the serializer
grant for that extent.

A nonfinal ABI-v2 smoke exposed the checker defect after implementation had
started. The correction changes only the fatal packet-cell consistency check:
each WQE's exact TX and RX formulas now use its already frozen
`port_tx_at_ps`. The single-WQE matrix is unchanged because its port TX equals
the cell doorbell delay. The parameter sweep, four TX additivity instances,
four RX additivity instances, two inverse-rate span instances, quantitative
bands and genuine-risk denominators are unchanged.

This correction is post-specified test machinery. It is not claimed as a
pre-registered assertion or as additional behavioral evidence. The corrected
checker must still reject the frozen missing-TX-event mutant and pass every
inherited Tier A family.

Before the formal run, a read-through against the inherited Tier A path also
found that the study built its producer in a sibling of the ABI-v1 run
directory. The inherited checker deliberately requires the executable to
reside inside that run directory. The study now copies the already built
executable into an ABI-v1-local `build/` directory before invoking the
unchanged checker. This path correction changes no simulation input, output,
oracle or denominator and was made before the formal result-producing run.
