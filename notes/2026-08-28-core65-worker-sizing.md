# CORE-65 physical-binding worker sizing

Date: 2026-08-28

## Scope

This worker will bind the retained standard-decode kernel stream to SGLang's
EP72 physical operation identities, or register the exact EP72-shaped capture
needed where retained evidence cannot decide the binding. The work includes a
total kernel inventory, layer-type and expert-population checks, static
weight-read accounting, counterpart classification, a calibration-only
movement against the 22,282 tokens/s/node anchor, registry updates, and
verification evidence.

## Expected change size

- Small field-addressed access reader and focused tests.
- Small protocol, expectations, source allowlist, and preservation manifests.
- Medium kernel-inventory and physical-binding derivation artifacts.
- Medium human-readable result and machine-readable companion report.
- Small mechanical registry and task-progress updates.

## Frozen qualitative directions

- Replacing a dense-only reduced-depth basis with the real three-dense plus
  58-MoE composition is expected to reduce priced service if the retained four
  layers are all dense. A correctly mixed basis is expected to produce no
  layer-composition movement.
- Replacing full-resident-expert iteration or full routed-expert weight reads
  with four EP72-local expert slots is expected to reduce priced service.
  Assignment-scaled expert compute may move less or remain unchanged, while
  expert-count or weight-byte service must use its own physical scale.
- Replacing excess captured per-rank static weight reads with the declared
  EP72-local volume is expected to reduce priced service. Equal volumes imply
  a null movement.
- Removing captured kernels with no EP72 counterpart is expected to reduce
  priced service. Adding required EP72 operations absent from the retained
  stream is expected to increase priced service.

These directions are frozen before any arithmetic or protected record access.
No candidate is assumed to explain the residual, and no calibration constant
will be fitted.

## Portable paths

- Repository root: `<repo>`
- CORE registry: `<repo>/docs/modules/core.md`
- Deployment study area: `<repo>/examples/deployment_curve_v1/`
- CORE-65 protocol and field reader: `<repo>/tools/`
- Bulk scratch and generated intermediates: `<wave-runs>/core65/`

No generated bulk data will be placed in the repository. Local absolute paths
will not be recorded in committed evidence.

## Isolation

This worker will not modify `simllm/deploy`, `worktrees/p2loggopsim`, any
`codex/deploy_p2_*` branch, the fifth scored run, or `nvcompare` traffic lanes.
No model weights or web pages will be downloaded.

## Access and verification budget

The field-addressed reader, source allowlist, frozen expectations, and protocol
will be committed before any retained capture or calibration record is
accessed. Every permitted field access will be logged contemporaneously with
byte accounting. Whole-file selectors will be rejected by construction, the
forbidden-access ledger must remain empty, and the held-out MTP value will
remain unread and uncompared.

Each commit will be checked with Ruff and the full pytest suite in the
worktree Python 3.10 virtual environment. Pytest's own exit status will be
checked directly without a pipe. End-of-line attributes and POSIX text
rendering will also be verified.
