# Host step cost v1 frozen expectations

The initial expectations-only commits for COMP-2 preceded the corrected
attempt-two capture, its harness, and every result-producing run on this
branch. Attempt two was nonvoid: all fatal guards held, so its measurements are
interpretable. It nevertheless missed CAL-1 because graph replay measured
809,068 ps against the frozen 600,000 to 700,000 ps band. Three of four scored
calibration relations passed, so the frozen phase gate rejected the attempt
before any host-step model was installed. The retained artifact is
`calibration_attempt2.json`.

This expectations-only revision discloses and refreezes attempt three. It
widens only CAL-1 to 600,000 to 1,000,000 ps using the now-known attempt-two
observation, incorporates that observation into the empirical uncertainty and
flagship bounds, and leaves every fatal predicate unchanged. This is not a
retroactive pass for attempt two. Attempt three first repeats the fixed-cost
capture on the same device under the corrected fatal oracle. Installation
starts only if the capture is nonvoid and all four scored calibration
relations pass. A nonvoid but rejected capture again stops the study and
requires a new disclosed freeze. Only then does the study install the
device-bound constants and follow them through `ExecutionGraph`, the serial
lowerer, the packet-level step sink, `StepResult`, TTFT, and TPOT. A B100 or
H100 query is not allowed to borrow the Turing values.

This expectations-only amendment freezes live attempt two after review of the
first implementation run. Calibration attempt three was accepted before the
model was installed. The first live run at clean observed commit `8a620e9`
was initially labeled accepted and recorded by `f5e2e74`. Review then compared
the represented GOAL compute service with the physical composition already
frozen below and found every calibrated step below
`F = max(C, N * g)`. Per-layer integer flooring lost 6,640 to 20,502 ps. That
violates a fatal physical precondition, so live attempt one is **VOID with
findings**. Its behavioral observations have no interpretable pass fraction
and cannot close COMP-2.

This amendment precedes the calibrated quantization repair and the live
attempt-two run. It does not widen LIVE-1, LIVE-2 or LIVE-3. The repair changes
only explicitly selected calibrated behavior from repeated per-layer floors to
the smallest whole-nanosecond enclosure. The ideal branch is excluded from the
repair and must remain byte-identical. A first five-cell ideal replay at
`f5e2e74` already reproduced the frozen aggregate digest and all five
step-record digests; the same check is repeated after the repair against the
final implementation.

Live attempt two subsequently completed nonvoid and reproduced every exact
repair row. A later entailment audit found that those exact rows were also
required by fatal LIVE-G3, LIVE-G5 and LIVE-G6. They therefore determine every
LIVE-1, LIVE-2 and LIVE-3 numerator before it is scored. Attempt two remains a
valuable nonvoid repair regression, but its 12 band observations are entailed,
unscored findings and its genuine-risk denominator is zero.

This expectations-only amendment freezes live holdout attempt three after that
audit. It changes no modeled behavior and does not reinterpret attempt two as
void. The holdout uses the previously unused accepted mission cell
`a-ep8-200g`; it freezes broad magnitude relations without freezing calibrated
timestamps or the exact represented service that would determine them. The
holdout is explicitly post-specified. Its purpose is to carry the magnitude
risk that attempt two cannot honestly carry.

## Chronology and prior observations

The following values were known before this freeze and therefore are not new
predictions:

- `examples/compute_fidelity_v1/RESULTS.md:3-8` declares the first run void
  because fatal guard XFER-G4 missed exact integer proportionality by 1 ps.
- `simllm/compute/provider.py:116-124` independently truncates every real-valued
  roofline duration to integer picoseconds. The registered single duration is
  mathematically 793,650,793.65 ps, so a doubled call can differ from twice the
  first integer by 1 ps without an additive term.
- The three prior captures observed graph-node costs of 624,665, 630,356 and
  630,124 ps; eager host-bound costs of 2,327,730, 2,337,286 and 2,331,958 ps;
  stamped device gaps of 1,280,480, 1,270,640 and 1,602,561 ps; and serialized
  launch-plus-synchronize costs of 5,021,995, 5,096,963 and 5,053,538 ps.
  They remain findings from void runs. Their ranges set the original
  replication target, not a confidence interval. Their `launch.csv` content
  identities are frozen in `expectations.json` as `initial`, `post_fix` and
  `registered`, so the three value rows cannot be silently reordered.
- Corrected attempt two, run from clean commit `13ae8a9`, held every fatal
  guard and observed graph replay 809,068 ps, eager host-bound launch 2,544,074
  ps, stamped device gap 1,573,280 ps and serialized launch 5,605,330 ps. Its
  raw launch content SHA-256 is frozen in `expectations.json`. The graph value
  alone missed its scored band, so the run is nonvoid but not accepted. These
  values are known inputs to attempt three, not new predictions.
- `examples/compute_fidelity_v1/RESULTS.md:372-390` retains 440 to 567 launches
  per 24-layer top-8 Granite MoE decode step and an omitted excess of 1.79 to
  12.31 times the 99,366,034 ps modeled compute.
- The accepted mission cell `a-ep8-400g` records an exact 204,526,734 ps decode
  step 1, partitioned into 99,024,000 ps compute and 105,502,734 ps collective
  service. Its first prefill step and request `r00` TTFT are exactly
  372,217,008 ps. These observed fixture literals are frozen here before the
  implementation run.
- Two archived accepted `end_to_end_replay_v1` executions already agreed on
  every deterministic field of cell `a-ep8-400g` and on its step-record bytes.
  The five-cell baseline digests in `expectations.json` were computed before
  this freeze. They are fatal compatibility oracles, not genuine-risk scores.
- Live attempt one represented graph launch counts 440 and 567 with compute
  service 356,088,000 and 458,856,000 ps, below their launch floors by 6,640
  and 20,502 ps. Its eager rows represented 1,040,256,000 and 1,340,520,000
  ps, below their floors by 16,200 and 12,585 ps. The retained void artifact
  is `live_attempt1.json`. All 12 behavioral observations are findings only.
- Live attempt two ran at clean commit `12d337e`, held every fatal guard and
  reproduced the frozen quantization repair. Its result was committed by
  `a484f38`. The result originally labeled 12 magnitude rows as scored, but
  fatal guards exact-checked their service, timestamps, TTFT and TPOT. Those
  fatal oracles entail all 12 relations, so this freeze reclassifies them as
  unscored findings. The run itself remains nonvoid.
- The final five-cell ideal replay was evaluated before this holdout freeze.
  OFF-G1 held: the mission was nonvoid, all 13 of its separate exact-oracle
  relations passed, and the aggregate plus all five step-stream identities
  matched. That fatal compatibility guard is unscored.

Calibration attempt three is therefore a disclosed replication after a
calibration-band miss. Its genuinely risky facts are whether
the device still lands in the predeclared ranges and whether the new model
composes to the predeclared end-to-end bands without changing the identity
path.

## External source audit

The launch bracket remains the static vLLM 0.26.0 audit from attempt one. The
source surfaces were re-read before this freeze:

- `vllm/model_executor/models/granitemoe.py:117-136` covers the gate and fused
  experts, `:209-219` covers attention, and `:264-282` covers the decoder layer.
- `vllm/model_executor/layers/fused_moe/fused_moe.py:1529-1570` is the fused
  experts entry and `:1592-1660` is the implementation boundary.
- `tools/compute_capture/gpu_fixed_cost_probe.cu:279-341` measures eager and
  graph launch classes, while `:343-429` independently stamps the real-kernel
  device gap.
- `simllm/backends/step_lowerer.py:172-229` is the one serial timing authority,
  `simllm/backends/step_sink.py:426-438` delegates the packet sink to it, and
  `simllm/adapters/vllm/executor.py:1103-1177` dispatches either the authoritative
  sink result or the compute fallback. No later layer may add the term again.
- `simllm/adapters/vllm/communicator.py:251-419` remains a zero-time structural
  coordinator observer. COMP-2 wires the profile through coordinator dispatch
  at `_SimStepRuntime.settle`; it does not time individual coordinator calls.
  VLLM-21 owns that distinct call-cost calibration.

## Corrected attempt-three calibration

The capture must identify exactly this environment before any measurement is
accepted:

| field | frozen value |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti |
| key | `gtx1660-ti-sm75` |
| UUID | `GPU-a90a812a-41bf-4f2f-c96d-d83e6eae6bd0` |
| compute capability | 7.5 |
| driver | 550.90.07 |
| CUDA compiler | 12.4.99, target `sm_75` |
| host CPU | AMD Ryzen 9 3950X 16-Core Processor |
| graph shape | 512 nodes, 200 replays |
| eager shape | 20,000 empty launches and one final synchronization |
| stamped gap shape | 400 back-to-back launches of the 262,144-item probe kernel |

The corrected XFER-G4 is fatal and unscored: a zero-flop, zero-byte
`RooflineProvider` query must return exactly 0 ps. Integer quantization cannot
confound zero. The positive-work single/double residual remains descriptive and
must still be reported, but it is not a fatal predicate in attempt three.

The incorrectly signed attempt-one FIX-2 is not silently reused. Attempt three
registers the quantity the probe directly measures: stamped batch wall time
minus summed in-kernel global-timer spans, divided by launch count, must be
strictly positive and inside its frozen band.

### Scored calibration relations

- **CAL-1**, one instance: graph replay is in `[600,000, 1,000,000]` ps per
  node. The upper bound is a disclosed attempt-three refreeze after attempt two
  observed 809,068 ps outside the original 700,000 ps ceiling.
- **CAL-2**, one instance: eager host-bound throughput is in
  `[2,000,000, 2,700,000]` ps per launch.
- **CAL-3**, one instance: the stamped device gap is in
  `[500,000, 3,000,000]` ps per launch.
- **CAL-4**, one instance: serialized launch plus synchronization is in
  `[3,000,000, 20,000,000]` ps per launch.

Each relation is evaluated directly on raw attempt-three capture output before
any exact inventory or provenance comparison. Device identity does not pin a
duration, the zero-work oracle does not pin a positive launch cost, and none of
CAL-1 through CAL-4 entails another. The calibration genuine-risk denominator
is four. Graph replay being cheaper than eager host-bound launching is still
checked, but it is a derived, unscored check because the disjoint CAL-1 and
CAL-2 bands already entail it.

### Fatal calibration guards

A violation makes attempt three void and COMP-2 stays open. These guards are
unscored:

- **CAL-G1** exact GPU model, key, UUID and compute capability.
- **CAL-G2** exact host CPU, driver and CUDA compiler identities.
- **CAL-G3** every parsed duration and count is positive.
- **CAL-G4** the corrected zero-work oracle returns exactly 0 ps.
- **CAL-G5** graph and eager costs do not exceed the same run's empty
  serialized launch-plus-synchronize enclosure. The positive stamped gap does
  not exceed the stamped real-kernel batch wall time divided by its 400
  launches. These are separate causal enclosures; empty-kernel synchronization
  is not claimed to enclose a real-kernel residual gap.
- **CAL-G6** the tracked and untracked worktree is clean before output creation,
  one observed HEAD is unchanged through capture, and the harness and probe
  content digests recorded at entry still match at exit. This records the live
  revision and content identities; it does not compare a frozen commit literal
  to a moving repository pin.

## Physical sanity before the attempt-three digits are read

The launch measurements have a causal floor of 0 ps. A negative host or device
gap is impossible.

For graph and eager measurements, the per-launch ceiling is the enclosing
empty serialized operation on the same run: one launch followed by device
synchronization. The four known enclosing values were 5.022 to 5.605 microseconds,
and the attempt-three serialized value has its own scored 3 to 20 microsecond
band. A graph node or pipelined eager launch above that enclosing path is a
harness defect. The stamped real-kernel gap instead has a causal floor of 0 ps
and a ceiling of the stamped batch wall time divided by 400 launches.

For a step with kernel service `C`, launch count `N`, and per-launch cost `g`,
the physical composition is frozen as

```text
launch_floor = N * g
represented_step_compute = max(C, launch_floor)
omitted_excess = max(0, launch_floor - C)
```

The model must never add `N * g` on top of `C`, because the host can run ahead
of device service. Across the four now-known captures, graph replay puts the
launch floor at 274.9 to 458.7 microseconds and eager launching at 1.024 to
1.442 milliseconds. Both lie below the serialized enclosure of
`N * serialized_launch`, and both lie above the 99.366 microsecond modeled
kernel service. Attempt two did too.
The attempt-three value must satisfy the same two inequalities before its
precision is discussed.

For the modeled number itself, the first-principles floor is
`max(C, N * g)`: neither the kernel service nor the serial launch demand can be
beaten. The deliberately loose ceiling is `C + N * serialized_launch`: even a
fully serialized sequence of the enclosing launch-and-synchronize operation
fits below it.

The fixed decode service input moves 554,631,168 bytes. On the source B100
roof, its memory-physics floor is therefore
`554,631,168 / 8 TB/s = 69,328,896` ps. A deliberately conservative
deployment-envelope ceiling assumes no less than one-eighth of that peak,
`554,631,168 / 1 TB/s = 554,631,168` ps. The frozen 99,024,000 ps source value
sits inside this range. This is a sanity enclosure for the fixed input, not a
new host-device transfer claim.

The live fixture has 48 serial dispatch or combine collectives per step. Its
network floor is therefore `48 * 2,000,000 = 96,000,000` ps before any byte is
serialized. The accepted source cell supplies 8,859,648 prefill bytes and
475,136 and 466,944 bytes in the two decode steps. Serializing all bytes at
400 Gbit/s and allowing one picosecond of integer enclosure per collective
gives network ceilings of 273,193,008 ps for prefill and 105,502,768 and
105,338,928 ps for decode. With fixed compute service of 99,024,000 ps for the
first two steps and 99,048,000 ps for the third, ideal prefill is physically
bounded by 195,024,000 to 372,217,008 ps. The decode steps are bounded by
195,024,000 to 204,526,768 ps and 195,048,000 to 204,386,928 ps. The source
observations, 372,217,008, 204,526,734 and 204,386,898 ps, sit within those
bounds before their exact digits are compared.

For each calibrated row, the decode-step floor is
`96,000,000 + max(C, N * g)`. Its deliberately loose ceiling is
`105,502,768 + C + N * serialized_launch`. TPOT is the mean of exactly two
declared decode intervals and uses the corresponding per-row bounds. TTFT is
exactly the one first-prefill interval, bounded below by
`96,000,000 + max(C, N * g)` and above by
`273,193,008 + C + N * serialized_launch`. These physical bounds are checked
before the registered multiplier digits.

Three independent sanity angles are required in the report:

1. host enqueue and eager throughput must remain close enough to identify a
   host-bound stream;
2. the independently stamped device gap must stay positive and below its batch
   wall time; and
3. the resulting 0.27 to 1.44 ms prior-observed Turing launch floor must be
   compared with the mission's independent generic 0.3 to 3 ms host-cost
   plausibility bracket. Its lower edge is about 0.025 ms below that generic
   floor and must be reported, while the rest overlaps. It is neither a B100
   calibration nor an absolute Turing prediction. Even the minimum accepted
   264 microsecond launch floor exceeds 1.3 times the roughly 99 microsecond
   compute term, so the flat-roofline derate cannot dominate this conditional
   sensitivity.

## Live attempt-two quantization repair

The calibrated point composition remains
`F = max(provider_duration_ps, N * g)`. GOAL carries whole nanoseconds, so its
first-principles floor is `F` and the narrowest representable ceiling is less
than `F + 1,000` ps. Live attempt two freezes the exact carrier value

```text
Q = ceil(F / 1,000) * 1,000 ps.
```

For every calibrated step, **LIVE-G5** requires
`compute_estimate_ps == F`, `compute_service_ps == Q`, and therefore
`F <= Q < F + 1,000`. It is fatal and unscored because integer enclosure is a
representation invariant, not magnitude evidence. The loose independent
ceiling `C + N * serialized_launch` remains in force. Ideal and legacy
profiles retain their old quantization path.

The repair is frozen to move the four calibrated rows as follows. Every step
within a row gets the same delta; completion timestamps after steps 0, 1 and 2
move by one, two and three times that delta. Network service, byte counts, flow
counts, operation order and completion order must not change.

| Profile | Launches | Attempt-one service (ps) | Frozen `F` (ps) | Attempt-two `Q` (ps) | Per-step delta (ps) |
|---|---:|---:|---:|---:|---:|
| CUDA graph | 440 | 356,088,000 | 356,094,640 | 356,095,000 | 7,000 |
| CUDA graph | 567 | 458,856,000 | 458,876,502 | 458,877,000 | 21,000 |
| Eager host | 440 | 1,040,256,000 | 1,040,272,200 | 1,040,273,000 | 17,000 |
| Eager host | 567 | 1,340,520,000 | 1,340,532,585 | 1,340,533,000 | 13,000 |

The resulting fixed-schedule metrics are frozen below. These exact rows check
the disclosed defect repair and stay fatal and unscored. The later entailment
audit established that they also determine LIVE-1 through LIVE-3. The 12
attempt-two band rows are therefore retained as unscored findings, not
genuine-risk evidence.

| Profile | Launches | TTFT (ps) | Decode step 1 (ps) | TPOT (ps) | Final completion (ps) |
|---|---:|---:|---:|---:|---:|
| CUDA graph | 440 | 629,288,008 | 461,597,734 | 461,515,816 | 1,552,319,640 |
| CUDA graph | 567 | 732,070,008 | 564,379,734 | 564,297,816 | 1,860,665,640 |
| Eager host | 440 | 1,313,466,008 | 1,145,775,734 | 1,145,693,816 | 3,604,853,640 |
| Eager host | 567 | 1,613,726,008 | 1,446,035,734 | 1,445,953,816 | 4,505,633,640 |

The observed vLLM schedule has a second projection invariant. **LIVE-G6**
requires `provider_compute_ps` to remain the raw provider value, 99,024,000 ps
for fixture steps 0 and 1 and 99,048,000 ps for step 2, while its separately
reported represented service equals the same `Q` as the serial lowerer. This
attribution check is fatal and unscored. Correcting the attribution moves no
timestamp.

LIVE-G3 is tightened to require exact component conservation with `Q`, exact
sequential releases and completions, exact request TTFT and TPOT partitions,
and one shared adapter and sink model. The ideal row must retain zero delta.
OFF-G1 repeats the full five-cell digest check after the repair.

## Installed model and uncertainty

Two calibrated profile classes are installed, each carrying the attempt-three
point value and a sample-limited empirical interval formed by the minimum and
maximum of the four prior observations plus attempt three:

- `turing-cuda-graph`, launch class `cuda-graph-node`;
- `turing-eager-host`, launch class `eager-host-bound`.

The profile also carries the device key, model, UUID, host CPU, driver, CUDA
version, source study, launch count and the empirical lower and upper
per-launch values. The interval is not called a confidence interval. Runtime
uses the attempt-three point value; reporting carries the empirical bounds.

The exact `ideal` profile remains 0 ps with 0 uncertainty. A nonideal profile
may execute only when `GpuSpec.name == "gtx1660-ti-sm75"`. Queries naming
`b100`, `h100`, or any other device fail before a graph, work directory or
timestamp is produced. The Turing values are a sensitivity profile for this
measured device and host, not a transferred production calibration.

`HostInitiationModel()`, `SerialStepLowererConfig` and `HtsimStepSinkConfig`
keep `HostInitiationModel.ideal()` as their exact default. Calibrated execution
requires an explicit profile and launch count. This preserves all accepted
studies while avoiding an impossible Turing-valued default on the default B100
envelope.

The flagship composition deliberately reuses the named mission fixture's
99,024,000 ps service input through a fixed-duration provider and names its
GPU `gtx1660-ti-sm75` solely so the calibrated host profile can be exercised.
It is an end-to-end mechanism and magnitude sensitivity, not a claim that the
fixed service value is a Turing model prediction. No B100 or H100 `GpuSpec`
ever receives the launch constant. The study must report this hybrid boundary
and separately demonstrate that the same host profile refuses both production
envelopes. A normal Turing roofline run may hide all launch demand behind its
slower kernel service; that adjacent fact is prose, not a registered clause.

The serial lowerer owns the one application of the term. The packet sink
delegates to that lowerer. Coordinator dispatch selects or validates the same
model and accepts the sink result without adding a second term. The vLLM
observed schedule applies it once when that path is explicitly selected.

## Flagship end-to-end relations

For the exact named mission decode fixture, the original retained finding
still derives this step-makespan range:

```text
1 + (99,024,000 / 204,526,734) * [1.79, 12.31]
  = [1.8666493447, 6.9600298512].
```

Attempt two then raised the known empirical high endpoint. Across all four
known captures and both launch-count endpoints, the direct composition is

```text
low  = (105,502,734 + max(99,024,000, 440 * 624,665)) / 204,526,734
     = 1.8596851696
high = (105,502,734 + max(99,024,000, 567 * 2,544,074)) / 204,526,734
     = 7.5686569757
```

The attempt-two acceptance band was `[1.80, 7.75]`. It encloses the four prior
empirical captures, including the disclosed calibration attempt-two
measurement. The live repair run swept two independent parameters: launch
class (graph and eager) and launch count (440 and 567). Its exact repair
oracles later made every observation in this band entailed, so this band is
historical regression context rather than holdout acceptance.

The attempt-two live fixture is the accepted mission cell `a-ep8-400g`,
request `r00`, with its exact captured routing. It replays source prefill step
0 and one-token
decode steps 1 and 2 under new sequential virtual timestamps, so scheduling is
fixed and host cost cannot alter batching. The model
is the source 24-layer, top-8, 32-expert, EP-8 geometry; the packet sink uses
`rnic-nn-fluid` at 400 Gbit/s. A fixed-duration provider supplies the source
step services 99,024,000, 99,024,000 and 99,048,000 ps, and a Turing-named
`GpuSpec` is used only for the host-profile compatibility check. The source `steps.jsonl` and
`routed-experts.json` digests are frozen in `expectations.json` and supplied by
the `SIMLLM_MISSION_BASELINE` input. This is a fixed-service Turing host
sensitivity, not a transfer of Turing time onto a B100 compute envelope.

The attempt-two ideal replay must reproduce prefill 372,217,008 ps, decode
steps 204,526,734 and 204,386,898 ps, and TPOT 204,456,816 ps exactly. Request
arrival is 0 ps, TTFT is the prefill completion, and TPOT is the exact mean of
the two distinct following decode intervals.

- **LIVE-1**, four entailed findings: every end-to-end decode-step multiplier
  is inside `[1.80, 7.75]`.
- **LIVE-2**, four entailed findings: TPOT rises by a multiplier inside the same
  band.
- **LIVE-3**, four entailed findings: TTFT increases strictly, but its relative
  increase is strictly smaller than TPOT's.

The exact step and request partitions are fatal and unscored. Contrary to the
first classification, the exact attempt-two service, timestamp, TTFT and TPOT
oracles do pin the raw multiplier relations. All 12 attempt-two LIVE rows are
entailed restatements and unscored. At fixed launch class, 567 launches costing
more than 440, and at fixed count, eager costing more than graph, are both
derived by construction from positive `N * g`. They remain checked as LIVE-D1
but are unscored.

Attempt two had six additional fatal, unscored preconditions:

- **LIVE-G1** each selected constant and empirical interval exactly matches the
  accepted calibration artifact and carries its declared provenance;
- **LIVE-G2** the calibrated profiles reject B100 and H100 before creating a
  graph, work directory, timestamp or sink call;
- **LIVE-G3** every step, request interval and TTFT or TPOT partition conserves
  exactly, the ideal replay matches the three source durations above, and the
  adapter and sink agree on one host model without double application;
- **LIVE-G4** the minimum and maximum over every actual profile and launch
  count at both plausible network endpoints stay inside `[1.35, 4.70]` and
  match direct integer arithmetic. This is derived from the accepted
  calibration bands and is not scored;
- **LIVE-G5** every calibrated GOAL service is exactly the narrow whole-ns
  enclosure `Q` and never undercuts `F`; and
- **LIVE-G6** observed-schedule attribution keeps raw provider service separate
  from the same represented service `Q`.

## Live attempt-three held-out magnitude

Attempt two cannot close the flagship relation because its fatal exact-row
oracles entail the reported multipliers. Live attempt three is post-specified
after that audit and uses the accepted mission cell `a-ep8-200g`, request
`r00`, steps 0 through 2. This cell has not been run with a calibrated host
profile. The accepted source observations are known before the holdout:

| Quantity | Step 0 | Step 1 | Step 2 |
|---|---:|---:|---:|
| fixed provider service (ps) | 99,024,000 | 99,024,000 | 99,048,000 |
| ideal step service (ps) | 549,409,968 | 214,029,454 | 213,725,778 |
| inferred network service (ps) | 450,385,968 | 115,005,454 | 114,677,778 |

The two-step micro-fixture TPOT is 213,877,616 ps. These are accepted-source
inputs, not new results. The source step stream and routing identities are
frozen in `expectations.json`. The holdout changes only the link rate from the
attempt-two cell's 400 Gbit/s to this cell's accepted 200 Gbit/s configuration.
It retains the same two launch classes and the 440 and 567 launch counts.

Before any calibrated holdout value is read, the physical network floor is
`48 * 2,000,000 = 96,000,000` ps per step. Serialization of the accepted byte
counts at 200 Gbit/s, with one picosecond of integer enclosure per collective,
gives ceilings of 450,385,968, 115,005,488 and 114,677,808 ps. The calibrated
compute carrier remains `Q = ceil(max(C, N * g) / 1,000) * 1,000`, so the
point values already identified by calibration and native conformance are
356,095,000, 458,877,000, 1,040,273,000 and 1,340,533,000 ps for graph 440,
graph 567, eager 440 and eager 567.

Adding the independent network floor to each `Q` gives whole-step floors of
452,095,000, 554,877,000, 1,136,273,000 and 1,436,533,000 ps. Adding the
per-step serialization ceilings gives these tighter physical ceilings:

| Profile and count | Step 0 ceiling (ps) | Step 1 ceiling (ps) | Step 2 ceiling (ps) |
|---|---:|---:|---:|
| CUDA graph, 440 | 806,480,968 | 471,100,488 | 470,772,808 |
| CUDA graph, 567 | 909,262,968 | 573,882,488 | 573,554,808 |
| Eager host, 440 | 1,490,658,968 | 1,155,278,488 | 1,154,950,808 |
| Eager host, 567 | 1,790,918,968 | 1,455,538,488 | 1,455,210,808 |

The separate loose causal ceiling remains
`network_source + C + N * 5,396,816` ps, using the accepted serialized
launch-plus-synchronize enclosure. For 440 launches its three values are
2,924,009,008, 2,588,628,494 and 2,588,324,818 ps; for 567 they are
3,609,404,640, 3,274,024,126 and 3,273,720,450 ps. Every measured step must
sit inside both applicable enclosures before its multiplier is considered.
The selected point host range, 0.356095 to 1.340533 ms, overlaps the
independent plausible 0.72 to 1.44 ms 48-collective network range and remains
below its upper endpoint. This is a Turing sensitivity and supplies no B100
host constant.

Point composition with the prior accepted network observations predicts the
following bands. Each band is rounded outward beyond the point projection;
neither a fatal guard nor a by-construction relation requires it to hold.

| Profile | Launches | Decode projection | TPOT projection | Frozen band for each |
|---|---:|---:|---:|---:|
| CUDA graph | 440 | 2.2011010410 | 2.2018976310 | `[2.17, 2.23]` |
| CUDA graph | 567 | 2.6813246648 | 2.6824621797 | `[2.65, 2.71]` |
| Eager host | 440 | 5.3977545259 | 5.4008205141 | `[5.34, 5.46]` |
| Eager host | 567 | 6.8006455504 | 6.8047074922 | `[6.73, 6.88]` |

- **LIVE-1**, four genuine-risk instances: decode step 1 is inside its
  profile-and-count band.
- **LIVE-2**, four genuine-risk instances: the micro-fixture TPOT is inside the
  same profile-and-count band.
- **LIVE-3**, four genuine-risk instances: `1 < TTFT multiplier < TPOT
  multiplier`, and `(TTFT multiplier - 1) / (TPOT multiplier - 1)` is inside
  `[0.38, 0.40]`. The point composition predicts 0.3892860131.

The three relation families stay separate. LIVE-D1 checks launch-count and
launch-class directions but is unscored by construction.

The holdout fatal guards are conditions, never a score:

- **LIVE-G1** requires accepted profile provenance, a clean stable revision
  and stable harness source.
- **LIVE-G2** requires B100 and H100 mismatch rejection before output.
- **LIVE-G3** requires exact source input identities and the exact ideal source
  row. Fixing the denominator does not fix a calibrated numerator.
- **LIVE-G4** requires structural component conservation, sequential release
  and completion, exact request reductions, raw provider attribution, bytes,
  flow counts and operation order. It does not require calibrated network
  service to equal the ideal network service.
- **LIVE-G5** requires every whole step to stay inside the physical bounds and
  retains exact `Q` as a component conformance invariant. Since calibrated
  network service remains free anywhere inside its physical interval, this
  does not determine a scored timestamp.
- **LIVE-G6** requires the already frozen conditional mission-budget
  arithmetic to remain exact. It does not use the 200 Gbit/s holdout
  timestamps.

Calibrated network service matching the ideal row is reported afterward as
LIVE-D2, a survivable unscored diagnostic. If it fails while the physical and
conservation guards hold, the raw multiplier observations remain
interpretable. Native exact-`Q` tests remain a separate conformance class.

The non-entailment check uses fatal-valid countermodels. Setting calibrated
network service to its 96,000,000 ps floor misses every LIVE-1 and LIVE-2
lower band while all fatal ranges can still hold. Setting prefill network
service to its floor and decode network service to its ceiling moves the
LIVE-3 increment ratio outside `[0.38, 0.40]`. Thus no earlier fatal oracle
pins any scored relation.

## Exact ideal off path

`examples/end_to_end_replay_v1` is the named accepted study. Its sink is changed
only to request `HostInitiationModel.ideal()` explicitly. A fresh five-cell run
must reproduce the frozen aggregate canonical digest
`5b51c31c1d83422cecfcbd975bf67690c6cccfd8ca4437ffef3e54985ee615fe`.
Canonicalization removes only `wall_seconds` from each `cell.json`, retains
every timestamp, request and operation order, byte count, service value and
completion, then serializes the five cells by name with sorted keys and compact
JSON separators plus one LF. Every `steps.jsonl` must also match its frozen
per-cell SHA-256 in `expectations.json` byte for byte.

This is **OFF-G1**, fatal and unscored. A mismatch voids the implementation
study. It cannot increase a behavioral denominator because the baseline bytes
were observed before the freeze and the ideal branch is an identity by
construction.

## Mission error budget after installation

This study must not fabricate a B100 constant. For a Turing-bound sensitivity
run, item 1 moves from zero to the measured launch floor. Correlating the same
launch term in the simulated and plausible-real expressions gives

```text
simulated = represented_launch_floor + 0.105502734 ms
plausible real = represented_launch_floor + [0.72, 1.44] ms
```

over the measured graph-to-eager launch range. The scored calibration bands
and launch-count endpoints bound this ratio in `[1.35, 4.70]` before the new
digits are read. The report must give the tighter interval recomputed from the
actual attempt-three value. Before live attempt two, the exact point interval
using `Q` is frozen as `[1.4249530295, 3.8910394651]`. Propagating the retained
sample-limited empirical endpoints gives `[1.3969639214, 4.5085504088]`. Both
are derived and unscored. This is a conditional Turing launch-throughput
sensitivity beside the mission's generic 5x to 22x budget, not a replacement
for that full budget. It assumes residual scheduler, sampler and Python costs
are zero. Those unmeasured residuals remain explicit unknowns, so the absolute
composed range stays unsupported even on Turing.
For the reference B100 configuration, item 1 becomes an explicit unknown and
the absolute composed optimism range remains unsupported until a B100 host
capture exists. The report must state both facts rather than transfer the
Turing result.

## Registered commands and dry-run contract

The calibration and attempt-two check-only commands validate their historical
registry structure, arithmetic, runtime inputs and output containment without
running CUDA or htsim. The holdout check-only command validates the newly
frozen cell identity, point projections, score bands, physical bounds,
countermodels, tools and output containment without invoking htsim. The
unchanged mission check-only command validates its pinned model, tool and
frozen mission inputs without running vLLM. Result-producing runs evaluate the
content digests and guards that require produced artifacts; check-only does not
claim to pre-evaluate those outcomes.
`${SIMLLM_WAVE12_RUN_ROOT}` is the branch-local
bulk-output root. `${SIMLLM_MISSION_BASELINE}` names the externally retained
accepted `a-ep8-400g` cell; its two required content digests are frozen above,
so its site location is not part of the tracked contract.
`${SIMLLM_MISSION_HOLDOUT}` names the corresponding accepted `a-ep8-200g`
cell and is checked by content identity rather than site location.

```bash
.venv/bin/python examples/host_step_cost_v1/run_calibration.py \
  --cuda-root "${SIMLLM_CUDA_ROOT}" \
  --out "${SIMLLM_WAVE12_RUN_ROOT}/host-step-calibration-attempt3" \
  --check-only

.venv/bin/python examples/host_step_cost_v1/run_study.py \
  --htsim-rnic "${SIMLLM_HTSIM_RNIC}" \
  --baseline-cell "${SIMLLM_MISSION_BASELINE}" \
  --out "${SIMLLM_WAVE12_RUN_ROOT}/host-step-cost-v1-attempt2" \
  --check-only

.venv/bin/python examples/host_step_cost_v1/run_study.py \
  --htsim-rnic "${SIMLLM_HTSIM_RNIC}" \
  --baseline-cell "${SIMLLM_MISSION_HOLDOUT}" \
  --out "${SIMLLM_WAVE12_RUN_ROOT}/host-step-cost-v1-attempt3" \
  --check-only

SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN}" \
PYTHONPATH=. "${SIMLLM_VLLM_PYTHON}" \
  examples/end_to_end_replay_v1/run_study.py \
  --cache-dir "${SIMLLM_MODEL_CACHE}" \
  --htsim-rnic "${SIMLLM_HTSIM_RNIC}" \
  --run-dir "${SIMLLM_WAVE12_RUN_ROOT}/end-to-end-ideal-offpath-attempt2" \
  --check-only
```

Calibration, attempt-two regression and the final ideal replay are retained
prior evidence. After this expectations-only commit, only the holdout command
is run in result-producing form. No output directory may already exist.

## Closure scope fixed before the run

COMP-2 currently registers “calibrated host-initiation profiles
(GPU-initiated vs CPU-proxy constants) for launch-path sensitivity studies.”
CUDA graph versus eager host launch is not a measurement of GPU-initiated
versus CPU-proxy network submission. If COMP-2 closes for the fixed per-step
scope demonstrated here, that untouched historical network-submission clause
moves to exactly one residual, COMP-28 `(Precision; P2; L)`, with this frozen
acceptance wording:

> After COMP-21 supplies device-bound structural captures for CPU-proxy and
> GPU-initiated network submission, fit and validate their scalar
> host-initiation projections for the analytical fallback used only while
> structural submission is disabled. Carry GPU, host, RNIC and submission-class
> provenance plus predeclared capture uncertainty; held-out
> ready-to-RNIC-visible latency must remain within that uncertainty. The ideal
> zero-cost profile remains the exact compatibility path.

This boundary is intentional. COMP-21 already owns the active producer-task
capture, including producer completion and RNIC-visible doorbell observables.
COMP-28 must consume that evidence instead of duplicating its hardware
campaign; it owns only the scalar compatibility projection while the
structural path is disabled.

COMP-2 closes only if every following clause is demonstrated: calibration
attempt three is nonvoid and accepted; each constant carries device and
launch-class provenance plus sample-limited uncertainty and refuses a
mismatched device; the host model, serial lowerer, packet sink and coordinator
dispatch apply one shared term exactly once; live holdout attempt three is
nonvoid and both profiles pass its decode-multiplier, TPOT and TTFT relations;
the named accepted study matches exactly on the ideal path; the mission budget
is recomputed; and the owning registry tag, bucket and ledger reconcile. A
miss on any fatal guard keeps COMP-2 open and does not register COMP-28. No ID
is registered for unknown B100 calibration, because the task explicitly
permits that result and no registered clause promises a B100 value. COMP-29
and COMP-30 remain unused.
