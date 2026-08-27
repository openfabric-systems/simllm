# VLLM-39 field-access protocol

Status: access protocol only. This is not the load-delay expectations freeze,
and no permitted candidate value was read to author it.

The allowlisted source is the landed candidate record at content address
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`.
The candidate builder concatenates Granite and DeepSeek entries, then sorts by
implementation ID. That committed construction rule places Granite CUDA-graph
decode batch 1 at entry 14, held-out batch 32 at entry 15 and calibration
batch 8 at entry 16. The reader admits exactly entries 14 and 16 at KV 16.
Both must identify themselves as MEASURED calibration evidence. It stops at
the closing byte of entry 16.

The intervening Granite batch-32 row is the source study's held-out shape and
must not be decoded or captured. DeepSeek rows must not be decoded or captured,
and all later record fields must not be read. The reader traverses nonselected
row bytes with a raw brace-and-string skipper that performs no JSON decoding,
returns only the three required top-level provenance fields and the two
permitted rows, whose row-local coverage fields complete provenance. It never
returns the whole record and appends one LF-terminated access event for every
returned selector.

Initial reconnaissance violated the required ordering: a broad repository
`rg` scanned the candidate path family before this reader existed and surfaced
a DeepSeek summary row. That output is excluded from every freeze and verdict.
The incident is part of the protocol record, so this attempt cannot be called a
clean exposure repetition.

The first committed reader attempt stopped before entry 0 because coverage is
row-local. The second stopped after decoding entry 0 because positional entry
0 was not the expected Granite row. Both rejected attempts remain in the
external access ledger and contribute to the contamination ruling.
