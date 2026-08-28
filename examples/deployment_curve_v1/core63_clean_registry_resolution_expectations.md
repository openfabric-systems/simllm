# CORE-63 clean registry resolution freeze

Status: **EXPECTATIONS ONLY FOR REGISTRY RESOLUTION**. The clean arithmetic is
complete and immutable. This bounded access exists only to resolve the actual
CORE-63 and CORE-64 registration paragraphs after the successful tranche's
forward selector stopped at earlier mentions.

The resolver starts one byte before the end of `<repo>/docs/modules/core.md`
and reads lines backward to the last entry-leading `CORE-63` and `CORE-64`
identifiers after only Markdown heading, list, table, emphasis, or numbering
punctuation. This excludes prose references such as "conditional on CORE-63."
It returns each matching line plus its bounded paragraph continuation. It does
not read the terminal byte, cannot cover the complete file, and logs begin/end
events contemporaneously with physical and unique byte counts.

Exactly two accesses and four events are permitted. Registry text is not an
arithmetic input. The corrected step, movement, undercorrection classification,
93-file preservation result, empty forbidden ledger, and MTP no-read status
cannot change as a result of this resolution.
