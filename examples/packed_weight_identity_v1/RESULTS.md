# Packed weight identity result

The metadata admission study passes all 15 boundary cases. A declared envelope
for 512 MXFP4 weight values needs a minimum of **272 payload bytes**, including
scales, so a 320-byte envelope is admitted. The previous one-byte-per-value rule
rejected it. All five independent byte oracles have zero residual; an envelope
one byte below each floor is rejected.

COMP-91 closes this false-rejection defect. COMP-54 still owns complete Kimi K3
configuration projection, mixed-format workload and execution graph; COMP-59
still owns physical capture. This result downloads no weights, verifies no
weight file, measures no GPU and enables no generic K3 timing prediction.

## Frozen evidence

The final expectation commit is
`2bec2fd4c34bee2fbcbb9d2e2340b24e0ffb33f2`, before implementation and first
execution. The executed source is
`c283dceb27a916bab6a8467c82a6b2c9ec2168eb`. The study exercises
`load_extraction_suite` and its canonical `ModelCheckpointIdentity` projection.
[results.json](results.json) records every admission, source identity and
compatibility control. The first frozen execution passes without fatal findings.

| Weight values | Packed value bytes | Minimum scale bytes | Payload floor |
|---:|---:|---:|---:|
| 31 | 16 | 1 | 17 |
| 32 | 16 | 1 | 17 |
| 33 | 17 | 2 | 19 |
| 512 | 256 | 16 | 272 |
| 1024 | 512 | 32 | 544 |

Each parameter count is tested one byte below its floor, at the floor, and
with 48 additional container bytes. The floor is fixed before execution as
ceil(P/2)+ceil(P/32). The synthetic container ceiling is that floor plus its
declared extra bytes; it is not a ceiling on a real checkpoint. Splitting
tensors, padding groups or replacing packed values with values of at least two
bytes cannot lower this bound. The independent oracle counts whole groups and
a compact tail rather than calling the production calculation.

Five exact byte rows and seven storage-scaling relations are reported separately
from the 15 admission cases. These arithmetic, rejection and identity checks
are unscored. There is no timing-behavior denominator. Four existing authored
suite identities preserve their canonical bytes, and the focused tests retain
strict count, digest, shard-order and metadata-only verification guards.

## Format and implementation boundary

Only `mxfp4-e2m1-group32-e8m0` selects this packed floor. Its name specifies
four-bit E2M1 values, groups of 32 and eight-bit E8M0 scales. Other formats retain
their previous admission behavior. Parameter count denotes weight values and
excludes their additional scales. The floor is necessary for the declared
format; passing it does not prove that a valid file or tensor layout exists.

K3's publisher configuration declares that packing for selected linear weights,
with higher-precision exclusions. Binding the complete configuration, exact
matrix storage and every execution family remains COMP-54's work. The generic
homogeneous compute projection continues to reject the packed format even after
metadata admission. Stored file bytes, resident device bytes and bytes fetched
by active experts remain distinct quantities.

Run the synthetic study in a new bulk-output directory:

```bash
python examples/packed_weight_identity_v1/run_study.py --output-root "$SIMLLM_STUDY_OUTPUT"
```
