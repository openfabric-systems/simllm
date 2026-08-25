# Retained kernel-cycle fixture provenance

These are the smallest projections needed from the read-only exploratory
kernel probe. They are test fixtures, not calibration evidence and not a GPU
campaign result.

The retained source snapshots read during fixture extraction and their SHA-256
values are:

| Logical retained artifact | SHA-256 | Fixture projection |
|---|---|---|
| `REPORT.md` | `bd410bf51b4aece12be63f172d144e5fb3b6896a4a80bcdedc37f0b597abb579` | five rows from the Granite TP1 bounded Nsight Compute table |
| `raw/197735-granite-tp1-graph/analysis/ordered-kernels.csv` | `42b21f91bac61a20d0fe1a91431e218a8ce33bb467d5df97b963e5646a11069e` | one `fused_moe_kernel` launch at the same ordinal from each of 16 steady decode steps |
| `raw/197735-granite-tp1-graph/analysis/kernel-summary.csv` | `50975eb52e94c0589ba3f8c6ac33c2bfc5b301c6758ace31a580c838741f738a` | the five decode batch-1 families also seen by Nsight Compute |
| `raw/197735-granite-tp1-graph/clocks.csv` | `98ee726ff4290266ac033841b12d6f2cee7663b267a5cf58cd9e81c1b70a3765` | one active 1,410 MHz SM and 1,593 MHz memory-clock sample |

The ordered excerpt starts after the 383-row matched baseline prefix. Within
each steady step, it selects the first of the 48 `fused_moe_kernel` launches.
This keeps one duration per replay while avoiding a 7 MB fixture. The summary
projection preserves the source medians and per-step counts. The Nsight
Compute raw reports were not present in the supplied retained tree, so the
five metric rows are transcribed from the source report and carry the report
snapshot digest rather than claiming a missing raw-file digest. The external
report is allowed to evolve after extraction; its pinned snapshot digest keeps
this fixture's origin unambiguous.

The record stays candidate status. The retained component pass did not report
DRAM bytes and did not capture the per-expert route split. Those values remain
null, and the routed key says `not-captured`; a validated record rejects that
state.
