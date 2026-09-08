# Rail comparator correction: post-specified regression checks

These assertions follow the completed 72-execution study and its observed
void result. They are post-specified regression checks, not public
pre-registration or a replacement for the qualification freeze at
`540bf8ddc89b3c0fe23991fcccf100e3dbd7b49b`.

The original comparator compares complete pipeline hop dictionaries. Adding
expert traffic changes the renderer's pipeline tag assignment, so the
comparison can report unequal flow times when only tags differ. The frozen
rail guard requires equal start, completion and flow-completion times across
the loaded and unloaded populations. It does not require equal renderer tags
across different message inventories.

Before correcting the comparator, freeze these checks:

- Match pipeline hops uniquely by source, destination and payload bytes.
  Tags remain validated against each cell's own declared message inventory.
- A matched tag change alone preserves the rail timing guard. A change to
  start, completion or flow-completion time fails it, including one picosecond.
- Missing or duplicate hop correspondence fails closed. The complete
  original-packet timestamp and zero expert-service checks remain unchanged.
- The retained full result and its compact projection keep their original
  verdict, fatal findings and local behavioral observations byte-for-byte.
  Do not rerun or rescore the measured population under this correction.

The ideal calendar mismatch independently voids the study. The original
packet surviving in the four-stage, 80-nanosecond case independently
contradicts its local retry-shape condition. This correction cannot qualify
TRAF-88 or remove the need for a new prospective study.
