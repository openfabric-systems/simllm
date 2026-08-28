# Frontier comparison expectations

These expectations freeze the external comparison the maintainer directed:
the deployment planning rung's frontier for the exact workload of the
public aiconfigurator example, overlaid NV-style against that tool's own
locally executed frontier, with every difference attributed to a named
mechanism and every precision claim scoped to what this repository can
defend. They are committed before any implementation of the comparison
exists. Nothing here tunes any parameter to any external number; the one
fitted quantity (the implied efficiency) is reported against a frozen
plausibility band, never installed.

## External anchors (fatal-pinned)

- The locally executed external run: aiconfigurator 0.11.0, backend
  trtllm, performance database h200_sxm 1.3.0rc10, model
  Qwen/Qwen3-32B-FP8, 32 GPUs, ISL 4000, OSL 500, prefix 500, TTFT
  300 ms, TPOT 10 ms; archived at the integrator's external-anchors store
  with manifest SHA-256
  `645b0b206f5af38ec1cc22cbef08d8cb7685af28b02f4e4c4c4480d84e080f5d`
  (76 files, pareto.csv for both modes, best_config_topn.csv, stdout,
  provenance). Pinned best-disaggregated row: 602.586 tokens per second
  per GPU, 108.944 tokens per second per user, TTFT 196.423 ms, TPOT
  9.179 ms, concurrency 192, 5 prefill workers (tp4, batch 1) plus 3
  decode workers (tp4, batch 64) on 32 GPUs. The disaggregated frontier
  carries 10 rows and the aggregated 25.
- The published README example of the same tool, pinned as a second,
  version-drifted anchor: best disaggregated 684.79 tokens per second per
  GPU at 100.31 per user, 4 replicas of (2 prefill tp2 plus 1 decode tp4
  batch 68). The drift between the published snapshot and the local
  0.11.0 run is reported as an external-tool versioning fact; neither
  anchor is preferred and nothing is fitted to either.
- The study consumes copies of the archived pareto and best-config CSVs
  as tracked frozen inputs with their SHA-256 recorded in the study
  directory at implementation time; the manifest hash above governs.

## Our-side inputs

- A new Qwen3-32B extraction column through the landed COMP-54 machinery,
  both pinned frameworks, config-only with the model identity recorded
  (exact Hugging Face revision written into the record). Fatal
  cross-check: the extracted structure must state 64 layers, hidden size
  5120, intermediate size 25600, 64 attention heads, 8 KV heads, head
  dimension 128, vocabulary 151936; any disagreement with these literals
  (which match both the public config and the external tool's own loaded
  architecture line) voids the run.
- A declared h200 device envelope, evidence class DECLARED from NVIDIA
  public specifications: peak dense FP8 1.979e15 flops per second, HBM
  4.8e12 bytes per second, capacity 141e9 bytes. FP8 weights at one byte
  per parameter. An efficiency arm at {0.6, 0.8, 1.0}.
- Workload point: ISL 4000 with 500 shared prefix (3500 uncached), OSL
  500, service targets TTFT 300 ms and TPOT 10 ms, 32-GPU budget.

## Fatal guards

- FG-1 the external archive manifest hash verifies; the tracked CSV
  copies match it.
- FG-2 the extraction cross-check literals above.
- FG-3 every term stamped with its evidence class; the external rows
  carry class MEASURED-EXTERNAL with tool, version and database identity;
  nothing external enters any pricing path (display and comparison only).
- FG-4 zero subprocess in our pricing lane (the estimator scan); the
  external tool is not executed by this study (its archive is consumed).
- FG-5 chronology.

## Family X1: expressibility (exact, scored)

Their pinned best configuration is expressed as one of our candidates
exactly: prefill pool 5 engines of 4 GPUs (tp4), decode pool 3 engines of
4 GPUs (tp4), 32 GPUs total, the workload point above. Feasibility
accepts it (FP8 weights per tp4 rank about 8.2e9 bytes, far under the
declared capacity); the candidate key is stable across two constructions.

## Family X2: matched-point pricing (scored)

With the frozen mapping (a decode engine prices batch 64 at tp4; the
prefill request prices 3500 uncached tokens at tp4, batch 1):

- X2a direction: our decode step at efficiency 1.0 predicts TPOT less
  than or equal to their 9.179 ms (roofline optimism has a sign; a miss
  here is a structural accounting defect, not a calibration gap).
- X2b direction: our prefill request at efficiency 1.0 predicts TTFT
  less than or equal to their 196.423 ms.
- X2c implied efficiency: e-star equals our efficiency-1.0 prediction
  divided by their value, computed for decode and prefill separately;
  both must lie in the frozen plausibility band [0.40, 1.00]. This band
  is wide by construction: it separates a structurally wrong accounting
  (outside) from an uncalibrated but sane roofline (inside), and the
  study reports e-star without installing it anywhere.

## Family X3: frontier overlay (scored)

Our estimator sweeps the config family (tp in {2, 4, 8} per role, worker
splits and batch ladder under the 32-GPU budget, service-target
filtered) at each efficiency arm; their 10 disaggregated rows overlay as
MEASURED-EXTERNAL points on the frozen NV-style axes.

- X3a shape: both frontiers are monotone (per-user speed falls as
  per-GPU throughput rises along each frontier).
- X3b dominance direction: at efficiency 1.0, for each of their 10 rows,
  our frontier's per-GPU throughput at that row's per-user speed is
  greater than or equal to theirs.
- X3c bracket: at least 8 of their 10 rows lie between our 0.6 and 1.0
  efficiency curves (could fail in either direction; a miss is published
  with the row).

## Family X4: the mechanism scope statement (fatal-unscored, disclosed)

The comparison figure and record carry the frozen envelope from the
ladder study: an ideal-network class (theirs by construction, ours at the
loggopsim-ideal rung) tracks the packet rung within about 1.6 percent on
contention-free point-to-point legs and is about 8x optimistic under
eight-into-one fan-in at the frozen cell, the mechanism their stack
cannot express and our packet rung prices. For this workload's shapes
(intra-node tensor parallel, single prefill-to-decode transfers) the
legs are in the contention-free regime, so no packet execution is run
here and the claim travels as a regime-scoped statement with the ladder
study cited, never as an absolute-accuracy claim.

## Honesty disclosures (fatal-unscored)

Their numbers interpolate a measured per-operation database for real
H200 silicon; ours are declared roofline plus declared envelopes until
the calibration campaigns close. On absolute kernel throughput their
side is better calibrated today and the record says so plainly. The
defensible precision claims are exactly: the network-mechanism envelope
(X4), the evidence-class labeling on every number, and the exact
accounting gates; nothing broader.

## Family W: wall time (scored, generous)

Our full sweep (all arms) completes in at most 120 s single process;
the external tool's observed 11 s search is reported as context,
unscored.

## Closure

This study delivers the maintainer-directed comparison: the NV-style
overlay, similarity where similarity is expected (shape, bracket), and
justified difference where difference is structural (roofline optimism
with its sign; the fan-in mechanism envelope). It does not close the
calibration gap, does not execute packet runs, and does not validate
either tool against serving silicon.
