# CORE-64 field-addressed access protocol

Status: **EXPECTATIONS ONLY**. This protocol, its reader, the component
classification, and every expected direction must be committed before the
reader accesses either protected record.

## Access boundary

Exactly three selectors are permitted:

1. The standard-decode calibration, residency decomposition, and frozen scope
   fields in `<repo>/examples/deployment_curve_v1/core63_clean_calibration_result.json`.
2. `units[1] -> attention_parallelism` in the content-addressed DeepSeek
   deployment projection.
3. `units[1] -> case_projections[0]` in that projection, which is the standard
   EP72 decode case.

The projection reader returns immediately after the first decode case. It
does not consume the delimiter before the second case, which is the forbidden
held-out MTP case. Each access writes a `BEGIN` event before opening the source
and an `END` event immediately after field selection. Every selected byte
count must be strictly below its source size.

## Forbidden boundary

The held-out MTP case at `units[1] -> case_projections[1]`, every fifth-run
artifact, and every MTP anchor value are forbidden. They must not be read,
decoded, copied, compared, or scored. Whole-file record streams are rejected
one byte before complete coverage. The forbidden-access ledger is frozen as
the empty JSON array `[]`.

No kernel-summary access is needed. CORE-63 already published the exact
physical decomposition. CORE-64 reads that bounded standard-decode result and
the framework-neutral logical family projection only.

## Frozen outcome protocol

The shape arithmetic, family classification, and signed directions are frozen
in `core64_expectations.json`. No service value may change unless a frozen
logical family has a deployment shape different from its capture shape. No
free or fitted constant is permitted. A null movement must publish as zero and
must retain the full signed calibration remainder.

No fifth scored run, model-weight download, web fetch, deployment-lane edit,
or decode-overlap term is permitted.
