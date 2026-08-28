# CORE-63 clean access protocol

Status: **EXPECTATIONS ONLY**. This protocol and its field reader must be
committed before any protected source access.

## Purpose

This repetition independently derives the CORE-63 residency correction from
the frozen pre-run inputs. The merged void result's derived numbers are not
inputs. Its protocol section alone is allowlisted so the three known failure
modes remain explicit without exposing its numerical result.

## Access boundary

Exactly six selectors are permitted:

1. The frozen standard-decode calibration context in
   `<repo>/examples/deployment_curve_v1/core63_expectations.json`.
2. Entry 7's registered component projection in
   `<repo>/examples/hopper_kernel_cycle_candidate_v1/candidate-record.json`.
3. The standard-decode, device-zero, noncollective rows in
   `<kernelprobe-root>/gh200lane/capture-198891-deepseek-v3-tp1-graph-decode/analysis/kernel-summary.csv`.
4. The literal CORE-63 entry in `<repo>/docs/modules/core.md`.
5. The conditional CORE-64 entry in `<repo>/docs/modules/core.md`.
6. The protocol section in the merged void study at
   `<repo>/examples/deployment_curve_v1/core63_calibration_result.md`.

The reader refuses any other path. Each permitted access appends a `BEGIN`
event before the source is opened and an `END` event immediately after the
selector finishes. Every event carries the selector, byte count, source size,
whole-file status, and held-out MTP status.

## Whole-file rejection

Every source is wrapped by a byte-counting guard whose hard limit is one byte
short of the file size. A missing field, an absent boundary, an end-position
selector, or any other request that would reach the final source byte is
rejected before that byte is read. The CSV reader stops after reading only the
routing prefix of the first row outside the selected shape. It does not scan
to EOF and does not decode that boundary row's payload.

The reader returns values in memory to the clean derivation runner. It never
prints protected values and never creates an intermediate basis file.

## Forbidden boundary

All MTP numeric fields in
`<repo>/examples/deployment_curve_v1/expectations.json` are forbidden. They
must not be read, copied, compared, scored, or reproduced. The fifth scored
run owns all scoring. The forbidden-access ledger must remain the empty JSON
array `[]`.

The void JSON result and every numerical section of the void Markdown result
are also forbidden as inputs. No ambient `cat`, `sed`, `rg`, CSV parser, JSON
loader, test, diff viewer, or shell pipeline may inspect a protected source.

## Frozen arithmetic and direction

The clean freeze will preserve these pre-run terms without amendment or refit:

- Assignment formula: `256 * 8 * 4 / 288`.
- Routed-expert scale: exactly `1/9` relative to the 256-assignment capture.
- Component rule: only a kernel name containing `fused_moe_kernel`, compared
  case-insensitively, is routed-expert work. All other noncollective work is
  retained at scale one, and the independently retained fixed term is kept
  once.
- Expected direction: corrected step decreases, prediction increases, and the
  signed residual becomes less negative before any possible crossing.

The clean arithmetic must be recomputed from the selected fields. No free or
fitted constants are permitted.

## Publication and preservation

The clean publication will contain the access ledger, empty forbidden-access
ledger, component classification ledger, signed movement, and preservation
verification. Prior artifacts remain byte-identical. Repository paths in
committed evidence use `<repo>`, `<kernelprobe-root>`, and `<wave-runs>` labels.
