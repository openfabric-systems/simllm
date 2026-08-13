# Host step cost v1 frozen expectations

This is the expectations-only record for COMP-2. It precedes the attempt-two
capture, the corrected capture harness, the calibrated host-step model, and
every result-producing run on this branch. The first compute-fidelity run stays
void with findings. This study does not edit its expectations or results and
does not promote its behavioral score.

Attempt two has two ordered phases. First, it repeats the fixed-cost capture on
the same device under a corrected fatal oracle. Phase two starts only if the
capture is nonvoid and all four scored calibration relations pass. A nonvoid
but rejected capture stops the study and requires a new disclosed freeze before
installation. Second, it installs those device-bound constants and follows
them through `ExecutionGraph`, the serial
lowerer, the packet-level step sink, `StepResult`, TTFT, and TPOT. A B100 or
H100 query is not allowed to borrow the Turing values.

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
  They remain findings from void runs. Their ranges set an attempt-two
  replication target, not a confidence interval. Their `launch.csv` content
  identities are frozen in `expectations.json` as `initial`, `post_fix` and
  `registered`, so the three value rows cannot be silently reordered.
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

Attempt two is therefore a replication. Its genuinely risky facts are whether
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

## Corrected attempt-two calibration

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
must still be reported, but it is not a fatal predicate in attempt two.

The incorrectly signed attempt-one FIX-2 is not silently reused. Attempt two
registers the quantity the probe directly measures: stamped batch wall time
minus summed in-kernel global-timer spans, divided by launch count, must be
strictly positive and inside its frozen band.

### Scored calibration relations

- **CAL-1**, one instance: graph replay is in `[600,000, 700,000]` ps per node.
- **CAL-2**, one instance: eager host-bound throughput is in
  `[2,000,000, 2,700,000]` ps per launch.
- **CAL-3**, one instance: the stamped device gap is in
  `[500,000, 3,000,000]` ps per launch.
- **CAL-4**, one instance: serialized launch plus synchronization is in
  `[3,000,000, 20,000,000]` ps per launch.

Each relation is evaluated directly on raw attempt-two capture output before
any exact inventory or provenance comparison. Device identity does not pin a
duration, the zero-work oracle does not pin a positive launch cost, and none of
CAL-1 through CAL-4 entails another. The calibration genuine-risk denominator
is four. Graph replay being cheaper than eager host-bound launching is still
checked, but it is a derived, unscored check because the disjoint CAL-1 and
CAL-2 bands already entail it.

### Fatal calibration guards

A violation makes attempt two void and COMP-2 stays open. These guards are
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

## Physical sanity before the attempt-two digits are read

The launch measurements have a causal floor of 0 ps. A negative host or device
gap is impossible.

For graph and eager measurements, the per-launch ceiling is the enclosing
empty serialized operation on the same run: one launch followed by device
synchronization. The prior enclosing values were 5.022 to 5.097 microseconds,
and the attempt-two serialized value has its own scored 3 to 20 microsecond
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
of device service. At the old extrema, graph replay puts the launch floor at
274.9 to 357.4 microseconds and eager launching at 1.024 to 1.325 milliseconds.
Both lie below the serialized enclosure of `N * serialized_launch`, and both
lie above the 99.366 microsecond modeled kernel service. The attempt-two value
must satisfy the same two inequalities before its precision is discussed.

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
3. the resulting 0.28 to 1.33 ms prior-observed Turing launch floor must be
   compared with the mission's independent generic 0.3 to 3 ms host-cost
   plausibility bracket. Its lower edge is about 0.025 ms below that generic
   floor and must be reported, while the rest overlaps. It is neither a B100
   calibration nor an absolute Turing prediction. Even the minimum accepted
   264 microsecond launch floor exceeds 1.3 times the roughly 99 microsecond
   compute term, so the flat-roofline derate cannot dominate this conditional
   sensitivity.

## Installed model and uncertainty

Two calibrated profile classes are installed, each carrying the attempt-two
point value and a sample-limited empirical interval formed by the minimum and
maximum of the three prior observations plus attempt two:

- `turing-cuda-graph`, launch class `cuda-graph-node`;
- `turing-eager-host`, launch class `eager-host-bound`.

The profile also carries the device key, model, UUID, host CPU, driver, CUDA
version, source study, launch count and the empirical lower and upper
per-launch values. The interval is not called a confidence interval. Runtime
uses the attempt-two point value; reporting carries the empirical bounds.

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

For the exact named mission decode fixture, the representative multiplier
derived from the retained finding is

```text
1 + (99,024,000 / 204,526,734) * [1.79, 12.31]
  = [1.8666493447, 6.9600298512].
```

The acceptance band is `[1.80, 7.10]`. It encloses the three prior empirical
captures and deliberately leaves the attempt-two result at risk; the wider
CAL bands do not guarantee LIVE acceptance. The study sweeps two independent
parameters: launch class (graph and eager) and launch count (440 and 567).

The live fixture is the accepted mission cell `a-ep8-400g`, request `r00`, with
its exact captured routing. It replays source prefill step 0 and one-token
decode steps 1 and 2 under new sequential virtual timestamps, so scheduling is
fixed and host cost cannot alter batching. The model
is the source 24-layer, top-8, 32-expert, EP-8 geometry; the packet sink uses
`rnic-nn-fluid` at 400 Gbit/s. A fixed-duration provider supplies the source
step services 99,024,000, 99,024,000 and 99,048,000 ps, and a Turing-named
`GpuSpec` is used only for the host-profile compatibility check. The source `steps.jsonl` and
`routed-experts.json` digests are frozen in `expectations.json` and supplied by
the `SIMLLM_MISSION_BASELINE` input. This is a fixed-service Turing host
sensitivity, not a transfer of Turing time onto a B100 compute envelope.

The ideal replay must reproduce prefill 372,217,008 ps, decode steps
204,526,734 and 204,386,898 ps, and TPOT 204,456,816 ps exactly. The scored
rows are evaluated from raw calibrated timestamps before that exact baseline
guard. Request arrival is 0 ps, TTFT is the prefill completion, and TPOT is the
exact mean of the two distinct following decode intervals.

- **LIVE-1**, four instances: every end-to-end decode-step multiplier is
  inside `[1.80, 7.10]`.
- **LIVE-2**, four instances: TPOT rises by a multiplier inside the same band.
  The raw per-request TPOT is checked before component conservation guards.
- **LIVE-3**, four instances: TTFT increases strictly, but its relative
  increase is strictly smaller than TPOT's because the same exposed step cost
  is amortized by the 372,217,008 ps first-prefill path.

The exact step and request partitions are fatal and unscored. They do not pin
the raw multiplier bands before LIVE-1 through LIVE-3 are evaluated, so the 12
live instances are genuine risk rather than entailed restatements. At fixed
launch class, 567 launches costing more than 440, and at fixed count, eager
costing more than graph, are both derived by construction from positive
`N * g`. They remain checked as LIVE-D1 but are unscored.

The implementation run has four additional fatal, unscored preconditions:

- **LIVE-G1** each selected constant and empirical interval exactly matches the
  accepted calibration artifact and carries its declared provenance;
- **LIVE-G2** the calibrated profiles reject B100 and H100 before creating a
  graph, work directory, timestamp or sink call;
- **LIVE-G3** every step, request interval and TTFT or TPOT partition conserves
  exactly, the ideal replay matches the three source durations above, and the
  adapter and sink agree on one host model without double application; and
- **LIVE-G4** the minimum and maximum over every actual profile and launch
  count at both plausible network endpoints stay inside `[1.35, 4.70]` and
  match direct integer arithmetic. This is derived from the accepted
  calibration bands and is not scored.

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
simulated = launch_floor + 0.105502734 ms
plausible real = launch_floor + [0.72, 1.44] ms
```

over the measured graph-to-eager launch range. The scored calibration bands
and launch-count endpoints bound this ratio in `[1.35, 4.70]` before the new
digits are read. The report must give the tighter interval recomputed from the
actual attempt-two value. This is a conditional Turing launch-throughput
sensitivity beside the mission's generic 5x to 22x budget, not a replacement
for that full budget. It assumes residual scheduler, sampler and Python costs
are zero. Those unmeasured residuals remain explicit unknowns, so the absolute
composed range stays unsupported even on Turing.
For the reference B100 configuration, item 1 becomes an explicit unknown and
the absolute composed optimism range remains unsupported until a B100 host
capture exists. The report must state both facts rather than transfer the
Turing result.

## Registered commands and dry-run contract

The first two check-only commands validate the enumerated registry structure,
derived arithmetic, runtime identities or inputs, and output containment
without running a CUDA workload or htsim and without creating output. The
unchanged mission check-only command validates its pinned model, tool and
frozen mission inputs without running vLLM or creating output. Result-producing
runs evaluate the content digests and fatal identity guards that need produced
artifacts; check-only does not claim to pre-evaluate those outcomes.
`${SIMLLM_WAVE12_RUN_ROOT}` is the branch-local
bulk-output root. `${SIMLLM_MISSION_BASELINE}` names the externally retained
accepted `a-ep8-400g` cell; its two required content digests are frozen above,
so its site location is not part of the tracked contract.

```bash
.venv/bin/python examples/host_step_cost_v1/run_calibration.py \
  --cuda-root "${SIMLLM_CUDA_ROOT}" \
  --out "${SIMLLM_WAVE12_RUN_ROOT}/host-step-calibration-attempt2" \
  --check-only

.venv/bin/python examples/host_step_cost_v1/run_study.py \
  --htsim-rnic "${SIMLLM_HTSIM_RNIC}" \
  --baseline-cell "${SIMLLM_MISSION_BASELINE}" \
  --out "${SIMLLM_WAVE12_RUN_ROOT}/host-step-cost-v1" \
  --check-only

SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN}" \
PYTHONPATH=. "${SIMLLM_VLLM_PYTHON}" \
  examples/end_to_end_replay_v1/run_study.py \
  --cache-dir "${SIMLLM_MODEL_CACHE}" \
  --htsim-rnic "${SIMLLM_HTSIM_RNIC}" \
  --run-dir "${SIMLLM_WAVE12_RUN_ROOT}/end-to-end-ideal-offpath" \
  --check-only
```

The result-producing forms are the same commands without `--check-only`, in
this order: calibration first, implementation study second, ideal compatibility
run third. No output directory may already exist.

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

COMP-2 closes only if every following clause is demonstrated: the attempt-two
capture is nonvoid; each constant carries device and launch-class provenance
plus sample-limited uncertainty and refuses a mismatched device; the host model,
serial lowerer, packet sink and coordinator dispatch apply one shared term
exactly once; both profiles pass the decode-multiplier, TPOT and TTFT relations;
the named accepted study matches exactly on the ideal path; the mission budget
is recomputed; and the owning registry tag, bucket and ledger reconcile. A miss
on any fatal guard keeps COMP-2 open and does not register COMP-28. No ID is
registered for unknown B100 calibration, because the task explicitly permits
that result and no registered clause promises a B100 value. COMP-29 and COMP-30
remain unused.
