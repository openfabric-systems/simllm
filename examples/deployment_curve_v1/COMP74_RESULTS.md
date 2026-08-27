# COMP-74 repeat-derived distribution result

## Per-key intervals

The committed reader exposed only the preregistered successor fields after the
freeze commit. Each exact key retains two independent observations, every pair
varies, and every repeat-derived service interval is therefore nonzero.

| Flagship key | Frozen point (ps) | Independent repeat (ps) | Signed movement (ps) | Frozen service envelope (ps) | Relative half-width |
|---|---:|---:|---:|---:|---:|
| EP32 prefill, 1K | 89,393,440,000 | 91,249,600,000 | +1,856,160,000 | 87,537,280,000 to 91,249,600,000 | 2.076393973% |
| EP32 prefill, 2K | 93,134,208,000 | 94,656,736,000 | +1,522,528,000 | 91,611,680,000 to 94,656,736,000 | 1.634767754% |
| EP32 prefill, 4K | 104,598,911,000 | 104,294,464,000 | -304,447,000 | 104,294,464,000 to 104,903,358,000 | 0.291061348% |
| EP72 standard decode, batch 32, remote KV 2000 | 1,875,680,000 | 1,883,392,000 | +7,712,000 | 1,867,968,000 to 1,883,392,000 | 0.411157554% |

These are key-local observed-repeat envelopes. No role, prompt length,
implementation suffix or MTP mode is pooled. The immutable record point stays
the center, and the two observations do not establish a broad stability claim.

## Distribution-OFF reproduction proof

The OFF path returns each inherited interval object without recomputation.
All 15 current anchor-layer point predictions reproduce exactly: three layers
for each of the 1K, 2K and 4K prefill anchors, standard decode, and simulated
MTP. The two stored flagship curves also reproduce as exact objects across all
ten load points when rebuilt by the existing capacity interval engine.

| Reproduction surface | Result |
|---|---:|
| Current anchor-layer point predictions | 15 of 15 exact |
| Inherited interval objects | 15 of 15 exact |
| Stored flagship curves | 2 of 2 exact |
| Stored curve load points | 10 of 10 exact |

## Band movement per anchor

The ON path Minkowski-adds `point * relative_half_width` to the inherited
physical and constant envelope. The compact machine-readable band table is
[`comp74_band_table.csv`](comp74_band_table.csv). The table below summarizes
the scored layer, `physics_plus_boundary_plus_attenuation`; calibration rows
remain context only.

| Anchor | Distribution movement | Frozen point verdict | 5% bar context |
|---|---|---|---|
| 1K prefill | Nonzero 2.076393973% spread added | Calibration context | Wider scored-layer band does not touch a boundary |
| 2K prefill | Nonzero 1.634767754% spread added | PASS | Wider scored-layer band intersects the bar but newly touches neither boundary |
| 4K prefill | Nonzero 0.291061348% spread added | PASS | Wider scored-layer band intersects the bar but newly touches neither boundary |
| Standard decode | Nonzero 0.411157554% spread added and propagated through both five-point flagship curves | Calibration context | Wider band remains outside the bar |
| Simulated MTP | No spread borrowed; zero width retained as single-seed | REFUTED | Band remains outside the bar |

One wider non-scored layer does newly touch a boundary: the 2K prefill
`physics_only` band reaches the upper edge of the closed 5 percent bar. This is
reported only as context. It is not the scored attenuated layer, no point moved,
and no rescore was performed.

The frozen verdicts are unchanged and restated: run-3 prefill remains PASS,
run-4 simulated MTP remains REFUTED, and the combined every-anchor claim
remains `ALL_SCORABLE_HELD_OUT_REFUTED`. Band widening never flips a verdict.

## Evidence, preservation and closure

The successor remains candidate at
`d868a4f35d633032daa238168d00f42c2ab47fc569db649b19b907008072e107`.
Its evidence-class ledger remains exactly 4 DeepSeek `MEASURED`, 4 DeepSeek
`DECLARED`, and 12 Granite `MEASURED`; no promotion or ledger edit occurred.
All 18 prior CORE-54 scored publication artifacts in the COMP-74 preservation
class retain their frozen SHA-256 identities. The three source projections are
logged with `whole_record_loaded` false and no unselected value returned.

COMP-74 closes literally: four of four priced DeepSeek keys have two retained
independent observations, all varying keys have nonzero intervals, OFF is exact,
and ON propagates through the existing interval engine without rescoring.

Two honest residuals remain. COMP-79 owns single-seed DeepSeek keys, beginning
with simulated MTP, without cross-mode pooling. COMP-80 owns the absent Granite
repeat arm from the partial campaign. No model weights were loaded or
downloaded, no web page was fetched, no Merlin submission was made, and no
traffic or NVLink module was touched.
