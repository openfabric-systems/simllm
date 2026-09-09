# Frontier publication precision expectations

This prospective DEPLOY-13 study qualifies the representation and comparison
of a published throughput threshold. It adds no service time, capacity,
calibration or scheduler. Historical operating points are retained regression
inputs, not new predictions. Their original mixed and void results remain
unchanged. Exact inputs and intervention grids are frozen in
[expectations.json](expectations.json).

## Scope correction before implementation

The former DEPLOY-13 bar required a matched configuration to remain selected
throughout its rounding interval. That statement is false under exact
threshold feasibility: any part of an interval above the configuration's
coordinate excludes that configuration. The rightmost configuration can
also leave part of an interval without any feasible answer. This study
withdraws that premise before implementation or execution. It does not widen
the historical throughput quotient band, which stays [0.75, 1.35].

DEPLOY-13 instead requires a complete, honest comparison over the declared
coordinate uncertainty. DEPLOY-26 owns the unresolved exact historical
coordinate and agreement evidence. Passing this representation study does
not imply that a historical frontier comparison passes its numerical band.
The earlier F-2-09 quotient remains exactly as published.

## Source precision and interpretation

The pinned AIConfigurator 0.11.0 source rounds a worker summary to three
places before composing the disaggregated result. Its picking path copies
that worker's throughput per user, and its final table rounds again. The
raw result dictionary can retain values before summary rounding, but a new
execution or removing all rounding changes the evidence source. Neither
recovers an unrecorded value in an old archive. See the pinned
[summary implementation](https://github.com/ai-dynamo/aiconfigurator/blob/v0.11.0/aic-core/src/aiconfigurator_core/sdk/inference_summary.py),
[selection implementation](https://github.com/ai-dynamo/aiconfigurator/blob/v0.11.0/src/aiconfigurator/sdk/picking.py)
and [session implementation](https://github.com/ai-dynamo/aiconfigurator/blob/v0.11.0/src/aiconfigurator/sdk/inference_session.py).

The archive does not fully identify its array datatype and numeric exporter.
For the historical x axis only, declare a CONDITIONAL conservative enclosure.
For exact decimal p, publication unit q = 1/1000 and u = 2^-53, use the closed
interval [p - q/2 - g, p + q/2 + g], where g = 4*u*(p+q). Require 1 <= p <=
1023. The guard is derived without consulting any modeled coordinate.

Its explicit arithmetic profile assumes a normal, finite binary64 input,
round-to-nearest binary64 multiplication and division, nearest-integer
rounding, and CSV text which round-trips the rounded binary64 value. NumPy
[documents this rounding operation and its scaling error](https://numpy.org/doc/1.26/reference/generated/numpy.round.html).
With alpha = 2*u + u*u, the profile bounds absolute error by
((1+u)*q/2 + (3*u+u*u)*p)/(1-alpha), which is at most q/2 + g. In this range,
a second identical rounding recovers the same integer tick because its
scaling error is far below 1/2. This justifies one half-unit for the copied
x coordinate. It says nothing about y or first-token time, whose intermediate
arithmetic differs. Carry historical_profile_verified=false on every
historical interval. These are possible choices over the enclosure, not a
claim that every enclosed value belongs to the actual exporter preimage.

## Installed comparison contract

Extend the existing generic coordinate/identity frontier seam. Snapshot exact
positive Fraction coordinates and unique stable identities once; reject
floats, duplicate identities and malformed intervals. Preserve the source
point and estimator stamp objects. The comparison never prices a service,
selects a calibration, starts a backend or creates another timing authority.

A published threshold retains its original decimal text, source SHA-256, row
identity, throughput axis and units, explicit interval endpoint ownership,
and exact, conditional or declared evidence basis. An exact-source basis
requires a singleton interval and never arises implicitly from display text.
A conditional basis retains its assumptions. The source-specific enclosure
above belongs to this study, not a universal installed rounding rule.

Exact lookup remains x >= threshold, selecting the greatest (y, stable ID)
among feasible points, or no point when none is feasible. This includes the
old tie direction. Interval lookup partitions every candidate breakpoint and
boundary singleton, preserving which endpoint belongs to which selection.
Return all maximal constant-selection segments, including no-answer segments.
No midpoint, epsilon, matched-identity preference or favorable endpoint may
replace that complete answer. Every selected identity remains joinable to its
original configuration and stamp.

For a positive exact reference y, report the full set of segment quotients,
its conservative bounds and possible infeasibility. Infeasibility retains an
explicit no-answer marker and contributes zero to the comparison quotient,
as in the existing exact study. PASS means every possible quotient lies in
[0.75, 1.35]; FAIL means every possible quotient lies outside it;
INDETERMINATE means both occur. Bounds alone cannot replace the segments,
because a bound can span unattainable intermediate throughput values. An
agreement verdict and the threshold's evidence basis remain separate fields.

## Prospective experiments and independent oracles

Use two synthetic points: primary at c + delta with y=100 and alternate at
c+1 with y=40. Cross centers c in {10,100}, publication precisions d in
{1,2,3,4}, and fixed offsets delta in {-0.00075,-0.00025,0,0.00025,0.00075}.
These 40 configurations use the declared closed sensitivity interval
[c - 10^-d/2, c + 10^-d/2], without the conditional historical guard. No toy
value is labeled measured or physically calibrated.

Independent decision oracle: if primary x is at least the interval upper
bound, only primary is selected and agreement passes. If primary x is below
the lower bound, only alternate is selected and agreement fails. Otherwise
both choices occur and agreement is indeterminate. A primary exactly at the
lower bound still owns its singleton. With delta zero, increased decimal
precision never resolves the ambiguity. With nonzero delta, sufficiently
small intervals select only the corresponding side. These shapes must hold
without changing either point's throughput or the accepted band.

Freeze exact lower/at/upper breakpoint answers, both endpoint ownerships,
rightmost infeasibility, duplicate-coordinate tie direction, dominated and
unsorted inputs, and two selections outside the band whose numerical hull
crosses the band. The latter must FAIL, not INDETERMINATE. A singleton interval
must equal exact lookup. Input permutation, unchanged stamps, complete
partition coverage, legal selections, unique IDs and no mutation are fatal
unscored guards. They do not increase the behavioral denominator.

The historical arm consumes all ten retained ideal operating points and all
ten original external rows. Reproduce all ten old exact-threshold answers and
quotients unchanged. Then report complete conditional interval selections,
quotient sets, qualification and any no-answer ranges. Do not predetermine
which historical rows pass. Retain the matching configuration's presence or
absence as a disclosure, not a substitute selection rule. Preserve the packet
arm byte for byte without repricing it.

Floor: selected throughput is nonnegative, and an infeasible segment is
explicit. Ceiling: selected throughput cannot exceed the greatest retained
candidate throughput. Before reviewing each historical comparison, verify
x = 10^12/decode_step_ps and y = request_capacity*500/used_gpus exactly.
Also bound y by the decode pool's service capacity under its declared rate
factor. These are dimensional and physical preconditions, not additional
behavioral passes. Review source rounding, exact threshold geometry and the
service/capacity identity as three independent angles. No new hardware
accuracy follows from these checks.

## Evidence and qualification

Run two complete evaluations in fresh Python processes and compare complete
evaluation bytes. Keep process IDs and wall time outside that object. Freeze
and verify the expectation commit, clean implementation commit, all protected
historical bytes and source stability over execution. Retain raw outputs
under a fresh external root. No overwritten artifacts or hidden retries.

Keep 40 prospective behavioral instances, exact lookup regression oracles,
historical conditional agreement disclosures, fatal guards and software tests
in separate evidence classes. A fatal violation makes the study VOID with a
null behavioral score. Any behavioral or exact-oracle miss prevents DEPLOY-13
closure. A PASS study may contain historical INDETERMINATE or FAIL agreement
rows because reporting those rows faithfully is its representation task.
DEPLOY-26 stays open regardless, and no earlier study is rescored.
