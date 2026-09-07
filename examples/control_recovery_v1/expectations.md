# Control recovery: frozen expectations

This contract is committed before recovery implementation and before any new
study run. It tests HTSIM-40 using the identical binary GOAL inputs and topology
files in the prior collective-width and pipeline-rail bulk evidence. It does
not alter their freezes or reinterpret their void verdicts.

## Mechanism and sweep

Select `none` (default, fatal control loss) or `headroom` (additive control
storage in each ns-tm3 output domain). The latter uses 131072 bytes per egress,
32 outstanding control messages per flow for its declared sizing envelope,
and 64 wire bytes per control: admitted fan-in = floor(131072/(32*64)) = 64.
The shared data pool stays 1048576 bytes. Reservation changes neither packet
service priority nor admission thresholds for data. A protected packet still
traverses every physical link and consumes serialization time. No endpoint
learns of a drop from simulator bookkeeping.

Main matrix: all-to-all widths 8, 16, 32, 64; rates 400 and 200 Gbit/s;
recovery none and headroom; physical rnic-cn compared with the identical
rnic-nn input phase. These axes are width, rate, recovery selection and
profile. The ideal arm is a reference, not a recovery implementation.
Also execute the six former pipeline control-loss cells with headroom:
4:1 oversubscription, expert width 32, pipeline depths 2, 4, 8, both rail
and node-local attachments, preserving each saved rate and schedule.
Compatibility additionally replays every completed standalone physical cell
in the supplied collective-width and pipeline-rail evidence, with none and
headroom. Reuse identical saved GOAL binaries, topology, flags and input
identities; never modify the reference evidence. Ideal reference completion
files are read-only. The study also checks the untouched pin binary against
the fresh reference evidence for the main matrix and failed pipeline cells.

## Physical bounds, before reading new results

All-to-all flow floor: 65536 payload bytes * 8 / endpoint bit rate, plus at
least the route propagation. Its ceiling is unbounded: retries, scheduling
and control can delay completion arbitrarily.

At width W, D = 7W/8 remote sources feed each destination. Phase floor:
D*65536*8/rate + propagation. For width 64 physical Clos, propagation is
4 microseconds, so the floors are 77.400320 microseconds at 400 Gbit/s and
150.800640 microseconds at 200 Gbit/s. Ideal propagation is 2 microseconds.
Physical phase ceiling: no finite universal ceiling follows from the reserve;
it guarantees buffer admission only within its declared control envelope.

Physical/ideal phase ratio floor is 1 for identical complete aligned phases;
its physical ceiling is unbounded for the same reason. Predictive band, not a
physical bound: the width-64 ratio lies in [1.5, 3.0], extending the earlier
roughly twofold width-8-to-32 ratios. The 200/400 phase-time ratio lies in
[1.6, 2.2]: halved link rate doubles serialization while propagation remains
constant; feedback and retry scheduling can perturb exact scaling. At each
rate, physical phase makespan should grow strictly with width. A miss of a
predictive band is a finding, not permission to amend this freeze.

Pipeline phase floor: maximum total payload delivered to one receiver divided
by that receiver's link rate, plus at least one path propagation; strengthen
with the saved causal-path floor when available. Ceiling is unbounded.
Pipeline physical/ideal phase ratio has no universal floor of 1 when schedules
release flows at model-dependent times; report it diagnostically. Each
individual flow still obeys its payload and propagation floor. Median and
99th-percentile flow completion time use nearest-rank order statistics within
one cell; each is above the minimum flow floor and below that cell's phase
makespan when releases start at zero. They are not request-tail estimates.

Recovery admissions floor is zero; total protected admissions cannot exceed
all control packets times their traversed switch count. Concurrent reserve
occupancy cannot exceed the configured per-egress bytes. With reserve disabled
both occupancy and admissions equal zero by construction and are unscored.

## Fatal guards and evidence classes

Every headroom cell must exit successfully, verify physical quiescence, return
all intended flow identities exactly once, and have consistent nonnegative
timestamps. Every previously completed physical cell must have a completion
CSV byte-identical to the fresh external reference with recovery both off and
on. Input text digests accept raw or LF-normalized bytes; exact completion
CSV comparison remains byte-for-byte against the supplied fresh evidence.

Width-64 none cells must retain the exact control-loss exit class. This is an
expected identity outcome, never a behavioral pass. A missing expected exit
voids identity validation. Expected none-mode failure is not a violated fatal
guard and leaves no valid phase latency to score. No violated fatal guard is
survivable: one violation voids the study for closure, while raw evidence and
individual diagnostics are retained without a behavioral pass fraction.

Width-64 physical phase must be at least the ideal phase and the payload plus
propagation floor. For aligned all-to-all also check every receiver's k earliest
completions against cumulative delivered payload/rate plus one propagation
delay. Per-flow physical/ideal ratios under sharing are diagnostic, not floors.
Data admission retains the old base-pool thresholds; native fixtures force
both pool-overflow domains separately and test every control kind, exhausted
headroom, release accounting and no-loss identity. Direct link drops outside
the protected admission domain remain fatal; reserve is prevention, not retry.

Keep run configurations, exact CSV compatibility oracles, behavioral relation
families, fatal structural guards and native test executables separate. Do not
sum them into a headline score. No new request-level time to first token or
time per output token claim follows: physical multi-artifact execution still
belongs to BACK-38, and the two consumer reruns and pin bump belong to the
orchestrator. HTSIM-40 stays open for those integration consequences.

## Record and figure

Record the pre-run expectation commit, backend commit and binary SHA-256,
reference-input digests, per-cell phase makespan, median and 99th-percentile
flow completion time, physical/ideal phase ratio, headroom admissions by
control kind and switch drop lines. Failed cells have null timing values.
All bulk evidence goes under SIMLLM_DATA_ROOT; reference roots are explicit
command-line options or SIMLLM_* variables. Tracked text uses LF bytes.
One plain matplotlib figure, PNG and PDF: phase makespan and ratio against
width for each rate, missing none-mode width-64 cells marked as void and
headroom admission counts annotated. Scientific bounds and the conditional
behavior bands retain their distinct meanings in the report.
