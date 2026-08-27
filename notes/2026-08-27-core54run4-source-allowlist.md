# CORE-54 fourth-run source allowlist

Status: frozen before run-4 source inspection

Pinned source tree: `<SGLANG_SOURCE_ROOT>/`, where the configured external root
must have the leaf name `sglang-source-bfeae4e79`.

Pinned source commit: `bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`

Pinned source tree: `9ffe149f40e1cd5bff7dadc6806ad1927d312e69`

The local tree was resolved from the already configured external source root.
No network request, web page fetch or source-content read was used to select
these paths. Tracked filenames and Git object identities were the only inputs.

## Allowed source files

Only the following complete implementation files may be inspected. The first
inspection will narrow citations to the smallest relevant line ranges.

| Purpose | Relative implementation path |
|---|---|
| Accepted speculative-token accounting contract | `python/sglang/srt/speculative/spec_info.py` |
| EAGLE accepted-token and verified-output accounting | `python/sglang/srt/speculative/eagle_info.py` |
| Worker handoff from verification to scheduler output | `python/sglang/srt/speculative/eagle_worker_common.py` |
| Current worker realization of accepted output IDs | `python/sglang/srt/speculative/eagle_worker_v2.py` |
| Request output-length and completion accounting | `python/sglang/srt/managers/schedule_batch.py` |

## Denied sources and payloads

Everything not enumerated above is denied for external source inspection. In
particular, the run will not inspect:

- benchmark, evaluation, test, recipe, result or performance-table files;
- web pages, URLs or remote source material;
- model configurations or model weights;
- any held-out anchor or scored-run payload through the source tree;
- unrelated SGLang implementation files, including prefill overlap and
  dispatch files covered by earlier allowlists.

The in-repository disclosure digest, recovered profile boundary and successor
campaign compiler remain separate evidence. This allowlist does not authorize
an anchor read or a successor-record read. Those require the committed run-4
field reader and an external append-only access log.
