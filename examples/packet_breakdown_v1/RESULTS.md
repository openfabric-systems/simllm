# Packet step critical-path breakdown

## What ran

The registration is [expectations.md](expectations.md), frozen by the
expectations-only commit `7b39399`. The matrix contains 14 configurations and
57 scheduler steps: eight width-tail ideal steps, four eight-step breakdown
requests, and the complete eight-step vLLM and nine-step SGLang M4 fixtures.
The first width-64, 400G ring cell also ran once as a pilot outside the matrix.

Each configuration executes its ordered artifacts once on the native ideal
backend. Its actual completion rows then drive five deterministic replays: omitted selection, explicit off,
explicit on, prepared off and prepared on. Prepared replays use two batches
when the fixture contains multiple records and check that preparation
publishes nothing. This is a transport run followed by projection checks,
not six independent transport measurements. M4 scheduler decisions are kept;
release times are rebased to preceding simulated completions, as registered.

Bulk GOALs, binaries, completion CSVs, events, full projections and logs live
under `SIMLLM_DATA_ROOT/packet_breakdown_v1`. The binary digests in
[results.json](results.json) identify the executables selected by
`SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN`. Compact extrema/count fixtures retain
every executed artifact identity for deterministic CI replay. Input digests
accept raw or LF-normalized bytes; generated text is written as LF bytes.

## What came out

All 14 configurations and 57 steps completed with no void cells. The 9,456
visits project to 47,280 CompletionEvent rows. Every step and every reduced
TTFT/decode interval conserves at 0 ps, and the old semantic/media partitions
are reproduced at 0 ps. All 70 mode comparisons preserve the complete
accepted outcome snapshot, GOAL text, binary GOAL and completion CSV bytes.
The eight width cells additionally match the earlier accepted width-tail
GOAL hashes, step latency, media and masked service exactly.

The four width-share comparisons and four serialization-doubling comparisons
all pass: 8/8 relations, with 0 ps residual for every rate relation. Fatal
structural checks remain separate from this denominator. Launch, device
queue and completion delivery are zero in all registered ideal cells;
nonzero values below belong to synthetic tests.

The width cells retain 100,000,000 ps of external compute each. Their other
nonzero component is service. These are modeled ideal-network intervals,
not measured hardware request tails. The exact serialization comparison
subtracts 4(W-1)*2,000,000 ps for the two rings and 4,000,000 ps for expert
dispatch/combine before comparing 200G against twice the 400G remainder.

| Pattern | Width | Rate (Gbit/s) | Service (ps) | Step latency (ps) |
|---|---:|---:|---:|---:|
| Ring | 8 | 400 | 132,876,800 | 232,876,800 |
| Ring | 8 | 200 | 209,753,600 | 309,753,600 |
| Ring | 64 | 400 | 608,832,000 | 708,832,000 |
| Ring | 64 | 200 | 713,664,000 | 813,664,000 |
| All-to-all | 8 | 400 | 22,803,200 | 122,803,200 |
| All-to-all | 8 | 200 | 41,606,400 | 141,606,400 |
| All-to-all | 64 | 400 | 153,260,800 | 253,260,800 |
| All-to-all | 64 | 200 | 302,521,600 | 402,521,600 |

The four breakdown requests reproduce their earlier registered compute and
network components. The packetized network values equal the earlier point
predictions, which were originally accepted within a band.

| TP | Profile | Service (ps) | External compute (ps) | Request span (ps) |
|---|---|---:|---:|---:|
| 2 | rnic-nn-fluid | 23,596,236,800 | 21,986,784,000 | 45,583,020,800 |
| 2 | rnic-nn | 24,018,124,800 | 21,986,784,000 | 46,004,908,800 |
| 8 | rnic-nn-fluid | 52,045,414,400 | 6,482,240,000 | 58,527,654,400 |
| 8 | rnic-nn | 53,237,022,720 | 6,482,240,000 | 59,719,262,720 |

Physical bounds were computed before execution. At 400G and 200G, payload
serialization costs 20 and 40 ps/byte. Each ring's service floor is
2(W-1)*(S/W*ps_per_byte + 2,000,000); each expert transfer's floor is
D*65,536*ps_per_byte + 2,000,000, with D=7W/8. The registered packet-calendar
ceilings and per-step bounds are retained next to each measured interval in
results.json. Compute is bounded by its declared quantized input; every
component lies between zero and the step latency. A queued physical network
has no finite first-principles ceiling. No timeout is treated as one.

The synthetic nonzero-wait test has 74 ps of device queue, 20,928 ps of
service, 998 ps of delivery and 10,000 ps of external compute, totaling
32,000 ps exactly. Its two fabric visits each begin 37 ps after eligibility,
finish at 10,501 ps and become visible at 11,000 ps. Those are test inputs,
not physical measurements. Separate tests exercise registration, exposed
host cost, semantic base, local/fabric dominance, ties, pending samples,
scheduler gaps, strict JSON rejection and deferred publication.

## What it changes

BACK-69's acceptance is met and its Completeness entry is removed. The
packet-step projection is available with
`HtsimStepSinkConfig(emit_packet_breakdown=True)` on both sinks. The executed
visits in `sink.packet_breakdowns` are the authority. `breakdown`, `segments`,
`runtime_report`, `execution_result` and `detail` are read-only projections.
Artifact identities are aggregate execution boundaries; the report's segment lane numbered zero is not an observed individual-rank frontier.

The runtime's existing `_critical_path_breakdown` performs the timing
reduction. Registration and exposed host cost occupy launch visits. Semantic
base and aggregate floor remain separately named delivery intervals. GPU
compute visits retain kernel ownership while gating the communication path
as external dependency. Only the exposed part of a host floor already
charged in compute is partitioned into launch time; it is never charged twice.

Pass the same entry as `packet_breakdown=` to `attribute_step_detail` or
`HtsimRequestMetricReducer.consume`. The reducer checks the segment-derived
semantic/media projection against the historical locality projection and
publishes per-request TTFT/decode five-component totals through
`critical_path_breakdowns()`. A request history cannot mix present and
missing projections. Historical RequestMetric work fields remain unchanged;
artifact work sums are available separately in RuntimeReport.

`packet_step_to_json(result, projection)` adds the strict versioned
`simllm-packet-step-breakdown-v1` object beside the canonical step-result
fields. `packet_step_from_json` validates it, including its join to the step
boundary and its declared totals. Omitting the projection writes the
unchanged core result object; the reader preserves old bytes on round trip.
The execution result emits five CompletionEvent phases per visit: submitted,
queued at eligibility, started, progress at resource finish, and completed at
visibility. No core StepResult dataclass or default codec was changed.

## What it does not change

CORE-67 still owns request bottleneck classification. BACK-38 still owns
state-preserving multi-artifact rnic-cn execution; its guard remains active.
No physical rnic-cn step was substituted with an ideal or standalone result.

The first/last completion-row timestamps bound an aggregate fabric visit.
The backend describes its start as a WQE start; this projection does not
invent a finer first-byte probe. Queueing within that start-to-finish service
interval remains unresolved, so these records do not separate internal
switch queues from endpoint work. Mixed-medium masked service remains a work
sum and is never added to elapsed time. Request endpoints remain whole-step
attribution for co-scheduled requests, without per-request packet frontiers.

BACK-68, HTSIM-40, fct.py, README files, other branches, other worktrees and
submodules were not modified. The root AGENTS.md was absent; the supplied
wave contract governed this work. No new task IDs were registered.

## Validation and handoff boundary

Ruff and module formatting pass. The complete suite reports:

```text
3 failed, 4373 passed, 24 skipped in 1259.12s (0:20:59)
```

All 50 packet-breakdown tests pass, including all 14 retained fixture cases.
The three remaining failures are outside the packet implementation:

- Two optional collective-floor native identity checks expect the older
  calibration publication's binaries. The configured wave binaries have
  different digests. Those existing tests and their golden records were not
  modified. Without selecting native binaries, that module reports
  `13 passed, 3 skipped in 0.45s`; this is committed-evidence validation and
  does not claim native reproduction of the older calibration.
- Removing BACK-69 requires closing it in docs/task-ledger.json and
  regenerating the README_PRO progress counts. The supplied worker contract
  prohibits README_PRO edits and assigns cross-file reconciliation to the
  orchestrator. The observed failure is 52 open backend tasks versus the
  correct 51. A proposed closure patch is retained in
  `SIMLLM_DATA_ROOT/readme-progress-proposed.patch`; neither bookkeeping
  file has been changed.

This worker has not obtained a fully green full-suite signoff. The
orchestrator must reconcile closure bookkeeping and run the suite with the
appropriate native-tool selection. No gate has been waived or converted to
an attribution score.

```text
All checks passed!
OK: 11 module doc(s) match docs/modules/FORMAT.md
```

Integration note (2026-09-07): the collective width tail record named in the
fixture inputs was rebuilt by the BACK-68 landing with all 800 ideal numerical
fields unchanged. The fixture's input digest was refreshed by re-collecting the
tracked rows on the merged tree; results.json stayed byte-identical and every
cell, relation and artifact comparison reproduced.

## Reproduce

```sh
. ./.env.local.sh
.venv/bin/python examples/packet_breakdown_v1/run_study.py
.venv/bin/python examples/packet_breakdown_v1/run_study.py --collect
.venv/bin/ruff check .
.venv/bin/pytest -q --basetemp "$SIMLLM_DATA_ROOT/pytest-c4"
.venv/bin/python scripts/check_docs_format.py
```

The [PNG](figures/packet_breakdown.png) and [PDF](figures/packet_breakdown.pdf)
show the five critical-path components for the eight ideal rnic-nn width
cells. Green service is fabric time; purple external dependency is the fixed
100,000,000 ps (0.100 ms) compute of every step. Launch queue, device queue
and completion delivery are each 0 ps in every registered ideal cell and
have no visible height, as the legend states. All panels use the same
millisecond scale. Segment labels and step totals above the bars are rounded
to 0.001 ms; leader lines identify the two smallest service segments. These
are modeled step intervals, not measured hardware request tails.
