# VLLM-42 batching-service expectations

Status: `EXPECTATIONS_ONLY`. No successor cell has run, and no observed
VLLM-41 batching-service or held-out value entered this freeze.

## Mechanism and derivation

The old component predictor made prefill and handoff zero-time boundaries.
That omission grants decode before some external arrivals are ready. Near a
discrete batch transition, the predicted decode batch is therefore too small.
Smaller batches cost more service per request-token, so the old prediction is
biased high exactly when the omitted positive phase changes batch membership.
Away from a transition the batch membership is unchanged, so the correction
has zero signed effect.

The replacement is service-only. Request `i` arrives at `i` times the floor
of 10^12 divided by `lambda` picoseconds. A ready prefill batch advances the shared clock by the
independently frozen prompt service, and decode becomes eligible after the
independently frozen handoff service. The next legal pool driver takes at
most eight ready requests. Decode batch `j` of size `b_j` advances the clock
by the independently measured and interpolated `S(b_j)`. The predicted field
is `sum_j S(b_j) / (64 * 4)`.

The lower and upper timing scenarios scale only the decode service clock by
plus or minus three times the largest independent trimmed coefficient of
variation. Every scenario is still priced with central `S(b_j)`, so the band
measures batch-transition uncertainty rather than changing the scored field.
There are no observed-curve inputs and no fitted parameters.

## Holdout and disclosure order

Load 240 requests per second is the held-out batch-transition load. Pool ratio
2:1 is the held-out ratio. Their union contains 30 cells. The other 48 cells
must be run, scored, published, and committed before any held-out cell runs or
is scored.

## Physical sanity and acceptance

The service floor is 236.603938 microseconds per request-token: the minimum `S(b)` divided by `b` over batch sizes one through eight.
The ceiling is 1110.576000 microseconds per request-token: the maximum `S(b)` divided by `b` over the same measured surface.
Every frozen band lies within those bounds. At the maximum studied load,
arrivals are four milliseconds apart; prompt service plus handoff is at most
0.214936 milliseconds, while a measured decode batch takes 1.110576 to
1.8928315 milliseconds. Those independent scales admit a transition from
single-request batches without permitting service outside the measured bounds.

A cell holds only when its observed amortized batching-service field lies in
the inclusive exact rational band in `expectations.json`. Any conservation,
identity, pricing, input-lock, preservation, or chronology failure makes the
run void. VLLM-42 closes only if all 78 cells hold and disclosure order holds.
Otherwise every miss publishes unchanged and the residual registers on
VLLM-50. Arrival-to-prefill and handoff-to-decode waits publish separately but
are unscored. The settled 210 to 220 requests per second onset and the 250 to
8,000 requests per second monotonic direction remain unscored.

## Frozen per-cell service bands

Values below are microseconds per request-token. `expectations.json` retains
the exact rational acceptance values.

| Prefill | Decode | Prompt | Load | Split | Predicted | Lower | Upper |
|---:|---:|---:|---:|:---|---:|---:|---:|
| 1 | 1 | 8 | 50 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 8 | 100 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 8 | 150 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 8 | 175 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 8 | 200 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 8 | 210 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 8 | 220 | non-held-out | 1110.576000 | 1103.587221 | 1110.576000 |
| 1 | 1 | 8 | 225 | non-held-out | 1093.104053 | 1079.126496 | 1103.587221 |
| 1 | 1 | 8 | 230 | non-held-out | 1068.643328 | 1054.665771 | 1082.620885 |
| 1 | 1 | 8 | 235 | non-held-out | 1047.676992 | 1033.699435 | 1058.160160 |
| 1 | 1 | 8 | 240 | held-out | 1026.710656 | 1012.733099 | 1037.193824 |
| 1 | 1 | 8 | 245 | non-held-out | 1005.744321 | 991.766763 | 1016.227488 |
| 1 | 1 | 8 | 250 | non-held-out | 984.777985 | 970.800427 | 995.261153 |
| 1 | 1 | 16 | 50 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 16 | 100 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 16 | 150 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 16 | 175 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 16 | 200 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 16 | 210 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 1 | 16 | 220 | non-held-out | 1110.576000 | 1096.598443 | 1110.576000 |
| 1 | 1 | 16 | 225 | non-held-out | 1086.115275 | 1075.632107 | 1100.092832 |
| 1 | 1 | 16 | 230 | non-held-out | 1065.148939 | 1051.171382 | 1079.126496 |
| 1 | 1 | 16 | 235 | non-held-out | 1040.688214 | 1030.205046 | 1054.665771 |
| 1 | 1 | 16 | 240 | held-out | 1019.721878 | 1009.238710 | 1033.699435 |
| 1 | 1 | 16 | 245 | non-held-out | 998.755542 | 988.272374 | 1012.733099 |
| 1 | 1 | 16 | 250 | non-held-out | 981.283595 | 967.306038 | 991.766763 |
| 1 | 2 | 8 | 50 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 8 | 100 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 8 | 150 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 8 | 175 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 8 | 200 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 8 | 210 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 8 | 220 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 8 | 225 | non-held-out | 1103.587221 | 1089.609664 | 1110.576000 |
| 1 | 2 | 8 | 230 | non-held-out | 1082.620885 | 1068.643328 | 1096.598443 |
| 1 | 2 | 8 | 235 | non-held-out | 1058.160160 | 1047.676992 | 1068.643328 |
| 1 | 2 | 8 | 240 | held-out | 1037.193824 | 1026.710656 | 1047.676992 |
| 1 | 2 | 8 | 245 | non-held-out | 1016.227488 | 1005.744321 | 1026.710656 |
| 1 | 2 | 8 | 250 | non-held-out | 998.755542 | 984.777985 | 1009.238710 |
| 1 | 2 | 16 | 50 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 16 | 100 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 16 | 150 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 16 | 175 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 16 | 200 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 16 | 210 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 16 | 220 | non-held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 1 | 2 | 16 | 225 | non-held-out | 1096.598443 | 1086.115275 | 1110.576000 |
| 1 | 2 | 16 | 230 | non-held-out | 1075.632107 | 1061.654550 | 1089.609664 |
| 1 | 2 | 16 | 235 | non-held-out | 1054.665771 | 1040.688214 | 1068.643328 |
| 1 | 2 | 16 | 240 | held-out | 1033.699435 | 1019.721878 | 1047.676992 |
| 1 | 2 | 16 | 245 | non-held-out | 1012.733099 | 998.755542 | 1023.216267 |
| 1 | 2 | 16 | 250 | non-held-out | 991.766763 | 977.789206 | 1005.744321 |
| 2 | 1 | 8 | 50 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 8 | 100 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 8 | 150 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 8 | 175 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 8 | 200 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 8 | 210 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 8 | 220 | held-out | 1110.576000 | 1103.587221 | 1110.576000 |
| 2 | 1 | 8 | 225 | held-out | 1093.104053 | 1079.126496 | 1103.587221 |
| 2 | 1 | 8 | 230 | held-out | 1068.643328 | 1054.665771 | 1082.620885 |
| 2 | 1 | 8 | 235 | held-out | 1047.676992 | 1033.699435 | 1058.160160 |
| 2 | 1 | 8 | 240 | held-out | 1026.710656 | 1012.733099 | 1037.193824 |
| 2 | 1 | 8 | 245 | held-out | 1005.744321 | 991.766763 | 1016.227488 |
| 2 | 1 | 8 | 250 | held-out | 984.777985 | 970.800427 | 995.261153 |
| 2 | 1 | 16 | 50 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 16 | 100 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 16 | 150 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 16 | 175 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 16 | 200 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 16 | 210 | held-out | 1110.576000 | 1110.576000 | 1110.576000 |
| 2 | 1 | 16 | 220 | held-out | 1110.576000 | 1096.598443 | 1110.576000 |
| 2 | 1 | 16 | 225 | held-out | 1086.115275 | 1075.632107 | 1100.092832 |
| 2 | 1 | 16 | 230 | held-out | 1065.148939 | 1051.171382 | 1079.126496 |
| 2 | 1 | 16 | 235 | held-out | 1040.688214 | 1030.205046 | 1054.665771 |
| 2 | 1 | 16 | 240 | held-out | 1019.721878 | 1009.238710 | 1033.699435 |
| 2 | 1 | 16 | 245 | held-out | 998.755542 | 988.272374 | 1012.733099 |
| 2 | 1 | 16 | 250 | held-out | 981.283595 | 967.306038 | 991.766763 |
