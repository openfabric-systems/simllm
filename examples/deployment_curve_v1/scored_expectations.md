# CORE-54 scored flagship expectations

This is the expectations-only freeze for the scored DeepSeek-V3 deployment
curve. It precedes the scored runner, every fitted value, every held-out read,
the scored result and the flagship figure. The machine-readable authority is
`scored_expectations.json`.

An implementation preflight then found that the first standard-decode
powers-of-two grid exceeded the `2^12` factor in `10^12` ps, so its largest
rates did not have integer-picosecond interarrivals. Before any fit, held-out
read or scored observation, only the standard-decode and MTP grids were
replaced with exact decimal divisors. No anchor, prediction, envelope, split or
decision rule changed.

## Dependency and identity freeze

The run may start only from a commit containing the merged CORE-53 binding,
the local Hopper candidate and the SGL-33 session. The candidate record is
pinned at
`ff46f6d8a79ddae899da89d4db6eb34373f8042acd06cab50b6336c8fb9a8f52`
and must retain `candidate` status in pricing provenance. Selecting it never
promotes it to calibrated evidence.

Cross-run guards compare only the preregistered stable projection. That
projection contains the client request identity, all request and handoff
timestamps, KV bytes, token observations, deterministic engine identities,
step counts, stable join metadata and candidate lookup provenance. It excludes
frontend-owned pool-local request IDs, operating-system process IDs and whole
serialized request-result bytes. A whole-bytes comparison is forbidden because
fresh frontend suffixes are unrelated to timing, pricing or client identity.

## Separate disclosure experiments

The SGLang prefill and decode tables are separate experiments on the disclosed
12-node cluster. The prefill experiment configures four eight-GPU nodes at
EP32. The decode experiment configures nine eight-GPU nodes at EP72. They are
never added and called a 96-GPU simultaneous deployment. The retained 13-node,
104-rank render is a structural comparator only. Any joint prefill-plus-decode
shape appears only in the second legend as a declared what-if.

The live session scale is one eight-GPU prefill engine plus one eight-GPU
decode engine. This is the largest scale with an identified physical curve in
the landed session: its parent clock serializes scheduler processes, so adding
processes does not demonstrate parallel node service. Per-node benchmark
observables stay unchanged. Aggregate target throughput uses the exact factor
4 for prefill and 9 for decode. Full EP32, EP72 and 16-prefill plus 40-decode
rank sets come from the PLACE-5 placement authority and are retained separately
from the live scheduler scale.

## Expected shapes and load grid

At low load, throughput should rise with offered load. Near capacity it should
approach a plateau while queueing increases delay and therefore moves inverse
delay downward. The prefill plateaus should order 1K above 2K above 4K. The
standard decode curve should approach its exact priced capacity. The joint
16-prefill plus 40-decode curve is expected to be decode-limited and is context
only.

The exact request-rate sweeps are:

| Configuration | Offered requests/s |
|---|---|
| SGLang prefill 1K | 32, 64, 128, 256 |
| SGLang prefill 2K | 16, 32, 64, 128 |
| SGLang prefill 4K | 8, 16, 32, 64 |
| SGLang standard decode | 2,000, 4,000, 8,000, 16,000, 32,000 |
| SGLang simulated MTP | 1,000, 2,000, 4,000, 8,000, 16,000 |

Every rate has an exact integer-picosecond interarrival. The MTP grid is frozen
but is not executed until an exact MTP price exists.

## Constants and physical envelopes

Only the intra-node collective surcharge is identifiable from the exact
calibration throughput anchors. It is fitted as a parameter, never reported as
a measurement.

| Constant | Initial | Closed envelope | Use and physical basis |
|---|---:|---:|---|
| Intra-node collective surcharge | 15,064,014 ps | 0 to 30,128,029 ps per routed collective | Applied 116 times, from two routed collectives in each of 58 MoE layers. Zero is the propagation floor and the upper endpoint is the measured width-8 pessimistic selectable surcharge. |
| PCIe submission | 20,000,000 ps | 100,000 to 20,000,000 ps | Fixed, because no calibration-split anchor gives exact handoff latency. The interval is still propagated. |
| Physical link rate | 400 Gbit/s | 200 to 400 Gbit/s | Fixed at the PLACE-5 link rate with the existing 200 Gbit/s sensitivity edge propagated. |
| KV packet service | backend result | 400 Gbit/s point to 200 Gbit/s edge | Derived from last required arrival minus first packet start for identical bytes and endpoints. It is not tuned. |

The fit minimizes summed squared relative error across only
`sglang_prefill_1k` and `sglang_decode_standard`, at integer-picosecond
resolution inside the closed surcharge envelope. A tie selects the smaller
value. An unconstrained preference outside the envelope becomes an honest
calibration miss at the in-envelope minimizer. It never widens the envelope.

## Exact predictions before tuning

The point below uses the initial 15,064,014 ps surcharge. Each lower endpoint
uses 30,128,029 ps and each upper endpoint uses 0 ps. The formula is

`per-node tokens x 10^12 / (candidate service ps + 116 x surcharge ps)`.

| Anchor | Lower | Point | Upper | Published role |
|---|---:|---:|---:|---|
| SGLang prefill 1K | 95,900.858 | 96,023.627 | 96,146.711 | calibration |
| SGLang prefill 2K | 92,058.421 | 92,171.544 | 92,284.945 | held out |
| SGLang prefill 4K | 81,990.294 | 82,080.014 | 82,169.930 | held out |
| SGLang standard decode | 7,975.333 | 8,434.496 | 8,949.760 | calibration |
| SGLang simulated MTP | BLOCKED | BLOCKED | BLOCKED | held out |

These are deterministic component projections, not confidence intervals. The
single retained DeepSeek seed contributes its recorded zero distribution
half-width while remaining labeled `insufficient-replays`. No repeat evidence
is invented. Candidate, distribution, constant and packet contributions stay
separate in the result.

## Fit, one-shot score and MTP ruling

The fit may read only the 1K prefill and standard decode anchors. It is written
and content-addressed before the scoring function is allowed to load any
held-out value. The one-shot numeric score then reads only the priced 2K and 4K
prefill anchors. It reports each point error and their maximum. A scoped pass
requires the maximum point error to be at most 5 percent. Error bars are shown
but cannot convert a point-error miss into a pass.

The EP72 MTP batch-16 KV-4,000 price is absent. The runner must not impute it,
select the standard decode cell or read its published value in a pricing
calculation. Its result row is `BLOCKED`, carries no prediction and names
COMP-72 resumable Merlin execution as the exact dependency.

## Figure and disposition

The main panel uses aggregated output throughput rightward and inverse
per-token request delay upward, with the upper-right corner optimal. A separate
prefill panel shows the published and simulated 1K, 2K and 4K input-throughput
anchors because input throughput has no honest coordinate on the main output
axis. The second legend carries DeepSeek's H800 production profile and the
declared 16-prefill plus 40-decode PLACE-5 what-if. The figure has no dry-run
watermark and states both the maximum scorable held-out error and the MTP gap.

If either priced held-out point misses 5 percent, the published verdict is
`REFUTED` with the dominant mechanistic contributor and a registered residual.
If both pass, the verdict is explicitly scoped to those two priced anchors and
still names MTP as blocked. CORE-54 closes only if its literal registered
acceptance text is satisfied; a refuted or blocked required anchor keeps it
open.
