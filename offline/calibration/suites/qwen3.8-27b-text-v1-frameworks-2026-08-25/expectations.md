# Qwen3.8-27B current-framework extraction suite

This authored-input suite carries the exact Qwen3.8-27B reference model and
15 text-only graph cells frozen by `qwen3.8-27b-text-v1` to the current
framework identities qualified on 2026-08-25. It exists because framework
identity is part of an inventory's input provenance. The historical suite and
the rejection study that consumed it remain byte-identical.

The only semantic differences from `qwen3.8-27b-text-v1` are:

- the suite ID is
  `qwen3.8-27b-text-v1-frameworks-2026-08-25`;
- vLLM is 0.27.1 at source commit
  `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`;
- SGLang is `0.5.19.dev345+gbfeae4e79` at source commit
  `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3` and source tree
  `9ffe149f40e1cd5bff7dadc6806ad1927d312e69`;
- the mechanism policy requires a complete Gated DeltaNet inventory instead
  of the historical total rejection.

The canonical reference-model object has SHA-256
`dc40d4409ffa517026e155671ab15af4086baf122d396d899bf3343e8920668c` in
both suites. The canonical 15-cell array has SHA-256
`2c59b3446463cf1538e304f0408ec467db0cb38cdef1b97e972cf2c68a3db5cf` in
both suites. The new suite file has SHA-256
`7be24843ffae71de65a1eab243eab9f592ce614097d701d5234eabd0c5980a9c`.

The suite remains authored inputs only. It contains no extracted StepRecord,
inventory, measured duration, code-object identity or observed launch.
