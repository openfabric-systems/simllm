# Pre-run physical accounting clarification

Independent source/physics review of freeze 33a44d6d identified a bound-label
error before any study execution. Keep the original immutable expectation
history and apply this clarification. Model implementation may have started
from that freeze; this is not represented as a new public pre-registration
preceding every implementation edit. It changes no simulation mechanism.

The original 83.88608 us value is a byte-cap approximation, not a formal
cell-charged wire drain ceiling. Wire preamble/IFG is not stored, and packet
padding/internal header charge changes the relation between storage and wire.
For a queue whose 1260 cells retain waiting plus active frames, the exact
no-new-arrivals drain is `sum(admitted wire bytes)*8/C`, with no new HIGH
traffic and with existing HIGH backlog charged separately.

With zero internal overhead and only full study DATA frames, 1500-byte IP
packets consume 8 cells each, allowing 157 frames and 241466 wire bytes:
77.26912 us at 25G. 9000-byte IP packets consume 44 cells each, allowing
28 frames and 253064 wire bytes: 80.98048 us. Tail packets change those
counts. For arbitrary IP lengths without descriptor limits, one-cell frames
can contain up to 228 wire bytes, giving 287280 wire bytes across 1260 cells
and 91.9296 us at 25G (22.9824 us at 100G). All are queue-content/service
bounds, not guarantees of lossless admission or DATA service under unpoliced
continuing HIGH arrivals.

For 1 MiB/sender and the stated synthetic headers, MTU1500 needs 731 frames
and 1123138 wire bytes/sender; MTU9000 needs 118 frames and 1060612 wire
bytes/sender. Four-sender aggregate 100G serialization floors are therefore
359.40416 and 339.39584 us. For four 25G receiver lanes the stronger floor
is the maximum actual routed lane wire total divided by 25G. Rotate lane
start phases across senders when comparing balanced service; a common start
phase otherwise produces a real jumbo tail imbalance. Dropped cases use
actual delivered lane wire totals and do not count as task completion.
