# VLLM-39 field-access protocol

Status: access protocol only. This is not the load-delay expectations freeze,
and no permitted candidate value was read to author it.

The allowlisted source is the landed candidate record at content address
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`.
The reader admits exactly entry 0 as Granite CUDA-graph decode batch 1 at KV 16
and entry 1 as the same cell at batch 8. Both must identify themselves as
MEASURED calibration evidence. It stops at the closing byte of entry 1.

The next Granite batch-32 row is the source study's held-out shape and must not
be read. DeepSeek rows and all later record fields must not be read. The reader
returns only the three required top-level provenance fields and the two
permitted rows, whose row-local coverage fields complete provenance. It never
returns the whole record and appends one LF-terminated access event for every
returned selector.

Initial reconnaissance violated the required ordering: a broad repository
`rg` scanned the candidate path family before this reader existed and surfaced
a DeepSeek summary row. That output is excluded from every freeze and verdict.
The incident is part of the protocol record, so this attempt cannot be called a
clean exposure repetition.
