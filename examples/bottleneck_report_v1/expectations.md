# Bottleneck report v1 expectations

This commit freezes CORE-67 classification and the study before implementation
or execution. It reuses published study cells and their predictions; it is
not a new hardware measurement pre-registration. BACK-69 supplies packet paths.

## Record and conservation

The optional `bottleneck_report` beside StepResult has schema
`simllm-bottleneck-report-v1`, step identity, a step ranking, and request
rankings for time to first token (TTFT) and time per output token (TPOT).
Each ranking carries its integer span in picoseconds, a positive divisor,
and deterministically ordered classes with integer span contributions.
The divisor is one for a step or TTFT and the completed inter-token interval
count for TPOT. Class spans sum exactly to the ranking span. Dividing both
by the same divisor conserves fractional TPOT without rounding. Class shares
are exact rational span/total; zero latency has no positive-share classes.
Ties use class name and participant width, never traversal order.

The reader rejects unknown fields, unsupported versions, booleans masquerading
as integers, negative/nonfinite values, duplicate identities, misordered
rankings, inconsistent evidence and nonconservation. Old results omit the
field and remain readable. No explicit null is substituted for absence.

## Classification and evidence

Only a selected participant's predecessor chain contributes. Parallel visits,
masked medium work, unused endpoints and additive visit totals cannot enter
the partition. Resource owner and the five timing components describe the
same intervals; they are not two sums to add together.

- Launch queue, launch service and launch visibility belong to host launch.
- Selected GPU kernel service and visibility use arithmetic intensity
  `F/D` against ridge `P/R`, where P is device FLOPs/s and R is HBM bytes/s.
  Below the ridge is HBM-bound, equality and above are compute-bound.
  Missing work or roof metadata is explicitly unclassified kernel service.
- Device queueing belongs to the selected contended resource. Collective
  service and waits identify intra-node collective with participant width,
  or fabric collective/queueing for cross-node work. A mixed medium tie is
  explicitly co-critical; it is never counted twice.
- Scheduler admission gaps are batching queue. Remaining causal gaps are
  external dependency; KV, DMA and control retain their names when no finer
  resource evidence exists. Unknown service must not be called batching.
- Kernel records retain the provider/calibration kernel identity and config.
  A supplied measured ledger cell is joined by that identity, retains its
  cell identity, evidence class and void state, and reports achieved binding
  fraction `max(D/R,F/P)/measured_time`. With no measured cell the fraction
  and measurement are `absent-by-design`, never an invented model efficiency.
  Void cells remain diagnostic and cannot become accepted calibration.
- Packet flow completion time (FCT) tail share is diagnostic only: for each
  selected fabric artifact, `(max(FCT)-min(FCT))/max(FCT)` over its packet
  completion rows. Retain the range and number of flows. This does not assign
  additive FCT sums to wall latency or claim switch queue instrumentation.

## Physical bounds before observations

Kernel floor: unavoidable bytes divided by HBM rate and FLOPs divided by
arithmetic rate bound service below by their maximum (integer quantization
is explicitly retained for legacy nanosecond GOAL calculations).
Kernel ceiling: the selected deterministic provider duration plus its stated
rounding quantum bounds the modeled service, not arbitrary real hardware.
Fabric floor: each causal hop requires payload serialization plus propagation;
receiver sharing additionally requires cumulative received bytes/link rate.
Fabric ceiling: ideal ring rounds take at most the analytic payload time plus
propagation and one packet slot per round; expert traffic is bounded by serial
delivery of all declared packets plus propagation.
Request floor: it cannot complete before its selected dependency chain.
Request ceiling: the sum of all nonoverlapping serialized required services,
declared host and scheduler delays bounds these deterministic cells.
Share floor: zero; ceiling: one, since only disjoint selected intervals count.
These are model-side classifications; no deployment throughput or hardware
calibration claim follows from agreement with a simulator's own service model.

## Frozen sweep and behavioral predictions

1. m4: ideal `rnic-nn-fluid`, TP widths 2 and 8, prefill2048 and decode8x2048,
   rates 200 and 400 Gbit/s. Fabric ranks first at TP 8. At fixed compute,
   halving rate doubles only serialization; propagation remains fixed.
2. breakdown: its 2048-token prefill and seven decode steps, ideal fluid,
   TP 2 and 8, rates 100 and 400 Gbit/s. Aggregate fabric share rises from
   TP 2 to TP 8. Prefill kernel is compute-bound and decode kernels HBM-bound.
   Lower rate increases fabric share. Compare its frozen 400G/100G components
   exactly, with zero picosecond tolerance.
3. collective_width_tail_v1: ideal `rnic-nn` step cells, widths 8 and 64,
   rates 200 and 400 Gbit/s, ring and all-to-all. Width 64 at 400G gives
   fabric shares 85.892 percent for two rings and 60.515 percent for expert
   dispatch/combine, within 0.0005 percentage point of these rounded anchors.
   Fabric share rises with width and with lower rate; class width follows
   actual collective participants, including the expert group.
4. core5_reduction: the published overlap graph and its deterministic service
   constants at 200 and 400 Gbit/s, with an added declared admission gap of
   0 or 1,000,000,000 ps. In the delayed first-token cells batching queue
   ranks first. Added admission increases TTFT by exactly the gap and leaves
   step service and subsequent TPOT unchanged. This variant deliberately
   changes admission only; it is not the original study's baseline cell.

## Identity and fatal guards

Compare disabled default with explicit false and compare enabled timing
after removing only the optional report. Preserve every accepted step JSON,
GOAL, completion CSV, timestamp, metric, backend row and completion order
byte for byte. Enabling the report changes no timing. Request chunking,
unequal endpoints, skipped scheduling intervals and fractional TPOT must
conserve at zero picoseconds. Codec round trips preserve exact identity.

Source digests accept raw or LF-normalized bytes. New text artifacts are
written as LF bytes and pinned with `text eol=lf`. CPU tests use deterministic
retained packet fixtures or stubs; the live study uses the configured backend.

All conservation, selection, source identity, timing identity, physical bounds
and codec integrity guards are fatal and unscored. No fatal study guard is
survivable. A violated guard makes this study VOID and leaves CORE-67 open;
retain findings and evidence. An inherited void kernel measurement is
survivable only as a labelled diagnostic, never calibration acceptance.
Behavioral predictions are reported separately from configurations, exact
oracle rows, fatal guards and test executables. No combined pass denominator.
RESULTS.md cites this expectations commit, reports the four required outcome
statements and includes one plain ranked-share figure as PNG and PDF.
