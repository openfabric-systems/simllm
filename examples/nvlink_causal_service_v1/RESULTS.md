# Causal NVLink service result

The causal packet study passes with a maximum exact-oracle residual of
**0 picoseconds**. It executes 84 model configurations over two link rates,
two payload sizes, finite credits and buffers, return latency, read dependencies,
replay and direct or queued routes. No GPU or hardware measurement participates.
TRAF-90 closes: one event calendar owns packet service and downstream capacity.
TRAF-45 still owns request-metric integration, and TRAF-73/TRAF-86 still own
hardware identification. This result does not qualify a deployed NVSwitch or a
large-model latency prediction.

## Provenance and verdict

The final preimplementation expectation commit is
`9787af3f30397f568adc5739192640ffe6e77d67`; it precedes implementation and the
first study execution. The executed implementation is
`a2700e9396c4a09b061ea5889983c4933f3b2115`. The first frozen execution passes.
[results.json](results.json) records source and raw-evidence digests, independent
oracles, timing relations and compatibility controls.

| Evidence class | Result |
|---|---|
| Configurations | 84 model executions |
| Exact completion oracles | 36 rows, maximum absolute residual 0 ps |
| Nonzero timing relations | 5 families, 22 passing instances |
| Fatal verdict | PASS, no findings |
| Compatibility controls | Four canonical result pairs are byte-identical |
| Profile-absent control | Exact caller object identity |

Future-release independence, causal legality, disjoint resources, class-label
identity, finite drain, capacity, conservation and compatibility are unscored
guards. The frozen bandwidth and payload relations receive their own timing
families; future insertion and resource independence contribute no behavioral
denominator. These evidence classes are never added together.

## Physical bounds and scaling

Before observing completion, the floor is wire bytes divided by the active
link rate; a read also waits for request serialization and target acceptance.
The ceiling is fully serial packet service through each declared resource,
with a declared credit return between packets. Every exact row lies inside
its independently calculated interval.

For a 1024-byte write, four 272-byte wire packets at 25 GB/s need at least
43,520 ps of link service. Fully serial link and receiver service takes
54,400 ps with immediate returns. Four credits and four-packet capacity give
46,240 ps, between those limits: four link services plus one 2720 ps receiver
tail. At 12.5 GB/s completion is 89,760 ps. The link term doubles exactly;
the receiver tail stays fixed. Neither the tail nor a read's request overhead
is incorrectly multiplied with payload size.

Reducing receiver capacity to one packet raises the 25 GB/s completion to
54,400 ps even with four link credits. Adding 10,000 ps to each return raises
it to 84,400 ps, exactly three intervening returns later. This distinguishes
finite downstream space from a credit-count-only model. Read completion for
the unconstrained 1024-byte case is 47,040 ps: the request adds 800 ps before
response service can begin.

The independent architectural checks ask three different questions: can the
source and link physically feed the packet by its finish time; does finite
receiver or switch storage own every arriving byte; and can the consumer see
data before its causal predecessor? Queued fan-in drains through a declared
one-packet switch buffer and slower receiver without over-admission. This is a
component plausibility result. Comparing complete model deployments with
published behavior remains a separate end-to-end requirement.

## Defects corrected and scope retained

An unscored diagnostic of the frozen base source finds that adding a future
extent shifts an earlier packet's visibility by 1,000,907 ps. Its response can
also precede request visibility by 694 ps. The new calendar passes both causal
guards; those historical values are diagnostics, not new calibration points.

Post-specified review probes, separate from the frozen matrix, caught slow
source feeding, overlapping replay source use and per-attempt rounding defects
before the first study execution. Regression tests retain those witnesses and
reject malformed capacity, credit, reservation and visibility projections.

The queued probe connects declared per-peer incoming links to a crossbar. It
does not establish a real NVSwitch attachment map. TRAF-45 binds explicit
physical ports and shared attachment calendars before using a switched route
for deployment claims. The source and link remain occupied throughout an
explicitly blocking replay train; this conservative policy does not identify
hardware retry arbitration. Read target processing has zero additional service
in this lower-envelope candidate. Numeric credit, buffer, return and arbitration
parameters keep their existing evidence classes. Compatibility consumers retain
their previous authority and results.

## Reproduction

Choose a new bulk-output directory through `SIMLLM_STUDY_OUTPUT`, then run:

```bash
python examples/nvlink_causal_service_v1/run_study.py --output-root "$SIMLLM_STUDY_OUTPUT"
```

The runner refuses an existing output directory, changed frozen parameters or
uncommitted study source. It writes each raw configuration and result before
checking its guards. Any fatal finding makes the whole run void and removes
its behavioral score; evidence remains available for a separately identified
repetition.
