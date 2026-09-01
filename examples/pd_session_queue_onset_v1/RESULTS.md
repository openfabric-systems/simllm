# VLLM-41 scheduler queue-wait onset

Status: onset identified below 250 requests/s; VLLM-41 closed.

## Predicted versus observed onset

The surface-and-arrival-only model predicted a central first queue-dominated segment of 225 to 230 requests/s. Its frozen uncertainty admitted 220 to 225 or 225 to 230 requests/s. Every observed configuration instead begins at 210 to 220 requests/s.

| Configuration | Predicted central | Frozen admitted segments | Observed first segment | Prior segments not queue-dominated | Band |
|---|---:|---:|---:|---:|---|
| `(1,1,8)` | 225 to 230 | 220 to 225; 225 to 230 | 210 to 220 | 5 | MISSED |
| `(1,1,16)` | 225 to 230 | 220 to 225; 225 to 230 | 210 to 220 | 5 | MISSED |
| `(1,2,8)` | 225 to 230 | 220 to 225; 225 to 230 | 210 to 220 | 5 | MISSED |
| `(1,2,16)` | 225 to 230 | 220 to 225; 225 to 230 | 210 to 220 | 5 | MISSED |
| `(2,1,8)` | 225 to 230 | 220 to 225; 225 to 230 | 210 to 220 | 5 | MISSED |
| `(2,1,16)` | 225 to 230 | 220 to 225; 225 to 230 | 210 to 220 | 5 | MISSED |

The observed onset is common to all six configurations, is strictly below 250 requests/s, and has five preceding non-queue-dominated segments per configuration. This satisfies the literal VLLM-41 closure rule. The earlier frozen knees of 1,056.6 and 2,113.2 requests/s remain refuted and are replaced by the observed 210 to 220 requests/s bracket.

## Held-out band verdicts

All 30 of 30 scheduler-wait bands held. Only 14 of 30 batching-service bands held, so 14 of 30 joint component comparisons held and VLLM-42 is registered. No band was widened after observation.

Displayed values are milliseconds. Frozen comparisons use the exact inclusive rational picosecond bounds retained in `results.json`.

| Configuration | Load | Queue predicted | Queue band | Queue observed | Queue | Service predicted | Service band | Service observed | Service | Joint |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| `(1,1,8)` | 240 | 0.581351931 | [0.000000000, 4.479688490] | 1.541873021 | HELD | 1.044182603 | [1.044182603, 1.046823283] | 1.023216267 | MISSED | MISSED |
| `(1,1,16)` | 240 | 0.581351931 | [0.000000000, 4.479688490] | 1.534646021 | HELD | 1.044182603 | [1.044182603, 1.046823283] | 1.016227488 | MISSED | MISSED |
| `(1,2,8)` | 240 | 1.664737947 | [0.000000000, 5.498463725] | 2.512592021 | HELD | 1.054665771 | [1.054665771, 1.057439545] | 1.033699435 | MISSED | MISSED |
| `(1,2,16)` | 240 | 1.664737947 | [0.000000000, 5.498463725] | 2.548692521 | HELD | 1.054665771 | [1.054665771, 1.057439545] | 1.030205046 | MISSED | MISSED |
| `(2,1,8)` | 50 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,8)` | 100 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,8)` | 150 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,8)` | 175 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,8)` | 200 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,8)` | 210 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,8)` | 220 | 0.000000000 | [0.000000000, 3.833725778] | 1.009364332 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,8)` | 225 | 0.000000000 | [0.000000000, 4.428540384] | 1.524696014 | HELD | 1.110576000 | [1.096476127, 1.114059611] | 1.089609664 | MISSED | MISSED |
| `(2,1,8)` | 230 | 0.607573514 | [0.000000000, 4.447228890] | 1.548790041 | HELD | 1.089609664 | [1.089288332, 1.089609664] | 1.065148939 | MISSED | MISSED |
| `(2,1,8)` | 235 | 0.593958993 | [0.000000000, 4.438910825] | 1.537416338 | HELD | 1.068643328 | [1.065425906, 1.068643328] | 1.044182603 | MISSED | MISSED |
| `(2,1,8)` | 240 | 0.581351931 | [0.000000000, 4.479688490] | 1.541873021 | HELD | 1.044182603 | [1.044182603, 1.046823283] | 1.023216267 | MISSED | MISSED |
| `(2,1,8)` | 245 | 0.569958747 | [0.000000000, 4.405846408] | 1.531161430 | HELD | 1.023216267 | [1.023216267, 1.025590759] | 1.002249931 | MISSED | MISSED |
| `(2,1,8)` | 250 | 0.524685701 | [0.000000000, 4.390462664] | 1.496204250 | HELD | 1.005744321 | [1.003325464, 1.005744321] | 0.981283595 | MISSED | MISSED |
| `(2,1,16)` | 50 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,16)` | 100 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,16)` | 150 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,16)` | 175 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,16)` | 200 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,16)` | 210 | 0.000000000 | [0.000000000, 3.833725778] | 0.000000000 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.110576000 | HELD | HELD |
| `(2,1,16)` | 220 | 0.000000000 | [0.000000000, 3.833725778] | 1.410400582 | HELD | 1.110576000 | [1.096476127, 1.124675873] | 1.107081611 | HELD | HELD |
| `(2,1,16)` | 225 | 0.000000000 | [0.000000000, 4.428540384] | 1.526542764 | HELD | 1.110576000 | [1.096476127, 1.114059611] | 1.086115275 | MISSED | MISSED |
| `(2,1,16)` | 230 | 0.607573514 | [0.000000000, 4.447228890] | 1.531002291 | HELD | 1.089609664 | [1.089288332, 1.089609664] | 1.061654550 | MISSED | MISSED |
| `(2,1,16)` | 235 | 0.593958993 | [0.000000000, 4.438910825] | 1.547584088 | HELD | 1.068643328 | [1.065425906, 1.068643328] | 1.040688214 | MISSED | MISSED |
| `(2,1,16)` | 240 | 0.581351931 | [0.000000000, 4.479688490] | 1.534646021 | HELD | 1.044182603 | [1.044182603, 1.046823283] | 1.016227488 | MISSED | MISSED |
| `(2,1,16)` | 245 | 0.569958747 | [0.000000000, 4.405846408] | 1.527351430 | HELD | 1.023216267 | [1.023216267, 1.025590759] | 0.998755542 | MISSED | MISSED |
| `(2,1,16)` | 250 | 0.524685701 | [0.000000000, 4.390462664] | 1.506372000 | HELD | 1.005744321 | [1.003325464, 1.005744321] | 0.977789206 | MISSED | MISSED |

## Per-cell decomposition

Batching service per token is reported separately from arrival-to-prefill wait and handoff-to-decode admission wait. Scheduler wait is only the sum of those two wait fields; provider service is excluded.

All displayed component values are milliseconds.

| Configuration | Load | Arrival to prefill | Handoff to decode | Scheduler wait | Batch service/token | Max prefill batch | Max decode batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `(1,1,8)` | 50 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,8)` | 100 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,8)` | 150 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,8)` | 175 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,8)` | 200 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,8)` | 210 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,8)` | 220 | 0.170255082 | 0.839109250 | 1.009364332 | 1.110576000 | 1 | 1 |
| `(1,1,8)` | 225 | 0.606425514 | 0.918270500 | 1.524696014 | 1.089609664 | 1 | 2 |
| `(1,1,8)` | 230 | 0.598855041 | 0.949935000 | 1.548790041 | 1.065148939 | 1 | 2 |
| `(1,1,8)` | 235 | 0.571649088 | 0.965767250 | 1.537416338 | 1.044182603 | 1 | 2 |
| `(1,1,8)` | 240 | 0.576105771 | 0.965767250 | 1.541873021 | 1.023216267 | 1 | 2 |
| `(1,1,8)` | 245 | 0.549561930 | 0.981599500 | 1.531161430 | 1.002249931 | 1 | 2 |
| `(1,1,8)` | 250 | 0.514604750 | 0.981599500 | 1.496204250 | 0.981283595 | 1 | 2 |
| `(1,1,16)` | 50 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,16)` | 100 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,16)` | 150 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,16)` | 175 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,16)` | 200 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,16)` | 210 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,1,16)` | 220 | 0.555459082 | 0.854941500 | 1.410400582 | 1.107081611 | 1 | 2 |
| `(1,1,16)` | 225 | 0.592440014 | 0.934102750 | 1.526542764 | 1.086115275 | 1 | 2 |
| `(1,1,16)` | 230 | 0.581067291 | 0.949935000 | 1.531002291 | 1.061654550 | 1 | 2 |
| `(1,1,16)` | 235 | 0.581816838 | 0.965767250 | 1.547584088 | 1.040688214 | 1 | 2 |
| `(1,1,16)` | 240 | 0.568878771 | 0.965767250 | 1.534646021 | 1.016227488 | 1 | 2 |
| `(1,1,16)` | 245 | 0.545751930 | 0.981599500 | 1.527351430 | 0.998755542 | 1 | 2 |
| `(1,1,16)` | 250 | 0.524772500 | 0.981599500 | 1.506372000 | 0.977789206 | 1 | 2 |
| `(1,2,8)` | 50 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,8)` | 100 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,8)` | 150 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,8)` | 175 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,8)` | 200 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,8)` | 210 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,8)` | 220 | 0.170255082 | 0.839109250 | 1.009364332 | 1.110576000 | 1 | 1 |
| `(1,2,8)` | 225 | 0.899319264 | 1.231376000 | 2.130695264 | 1.103587221 | 1 | 2 |
| `(1,2,8)` | 230 | 1.022824041 | 1.370391000 | 2.393215041 | 1.075632107 | 1 | 2 |
| `(1,2,8)` | 235 | 1.073270088 | 1.428467750 | 2.501737838 | 1.054665771 | 1 | 2 |
| `(1,2,8)` | 240 | 1.071202521 | 1.441389500 | 2.512592021 | 1.033699435 | 1 | 2 |
| `(1,2,8)` | 245 | 1.038134430 | 1.477598500 | 2.515732930 | 1.012733099 | 1 | 2 |
| `(1,2,8)` | 250 | 1.045420250 | 1.480580500 | 2.526000750 | 0.991766763 | 1 | 2 |
| `(1,2,16)` | 50 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,16)` | 100 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,16)` | 150 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,16)` | 175 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,16)` | 200 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,16)` | 210 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(1,2,16)` | 220 | 0.528123082 | 0.854941500 | 1.383064582 | 1.110576000 | 1 | 1 |
| `(1,2,16)` | 225 | 0.899931014 | 1.281997750 | 2.181928764 | 1.096598443 | 1 | 2 |
| `(1,2,16)` | 230 | 1.060947291 | 1.367409000 | 2.428356291 | 1.075632107 | 1 | 2 |
| `(1,2,16)` | 235 | 1.093998588 | 1.428806000 | 2.522804588 | 1.051171382 | 1 | 2 |
| `(1,2,16)` | 240 | 1.102491771 | 1.446200750 | 2.548692521 | 1.030205046 | 1 | 2 |
| `(1,2,16)` | 245 | 1.086818430 | 1.483019500 | 2.569837930 | 1.009238710 | 1 | 2 |
| `(1,2,16)` | 250 | 1.038193250 | 1.486611250 | 2.524804500 | 0.988272374 | 1 | 2 |
| `(2,1,8)` | 50 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,8)` | 100 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,8)` | 150 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,8)` | 175 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,8)` | 200 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,8)` | 210 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,8)` | 220 | 0.170255082 | 0.839109250 | 1.009364332 | 1.110576000 | 1 | 1 |
| `(2,1,8)` | 225 | 0.606425514 | 0.918270500 | 1.524696014 | 1.089609664 | 1 | 2 |
| `(2,1,8)` | 230 | 0.598855041 | 0.949935000 | 1.548790041 | 1.065148939 | 1 | 2 |
| `(2,1,8)` | 235 | 0.571649088 | 0.965767250 | 1.537416338 | 1.044182603 | 1 | 2 |
| `(2,1,8)` | 240 | 0.576105771 | 0.965767250 | 1.541873021 | 1.023216267 | 1 | 2 |
| `(2,1,8)` | 245 | 0.549561930 | 0.981599500 | 1.531161430 | 1.002249931 | 1 | 2 |
| `(2,1,8)` | 250 | 0.514604750 | 0.981599500 | 1.496204250 | 0.981283595 | 1 | 2 |
| `(2,1,16)` | 50 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,16)` | 100 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,16)` | 150 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,16)` | 175 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,16)` | 200 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,16)` | 210 | 0.000000000 | 0.000000000 | 0.000000000 | 1.110576000 | 1 | 1 |
| `(2,1,16)` | 220 | 0.555459082 | 0.854941500 | 1.410400582 | 1.107081611 | 1 | 2 |
| `(2,1,16)` | 225 | 0.592440014 | 0.934102750 | 1.526542764 | 1.086115275 | 1 | 2 |
| `(2,1,16)` | 230 | 0.581067291 | 0.949935000 | 1.531002291 | 1.061654550 | 1 | 2 |
| `(2,1,16)` | 235 | 0.581816838 | 0.965767250 | 1.547584088 | 1.040688214 | 1 | 2 |
| `(2,1,16)` | 240 | 0.568878771 | 0.965767250 | 1.534646021 | 1.016227488 | 1 | 2 |
| `(2,1,16)` | 245 | 0.545751930 | 0.981599500 | 1.527351430 | 0.998755542 | 1 | 2 |
| `(2,1,16)` | 250 | 0.524772500 | 0.981599500 | 1.506372000 | 0.977789206 | 1 | 2 |

## Per-segment decomposition

The queue-dominated rule requires a positive scheduler-wait delta and a positive sum of scheduler-wait delta per four output tokens plus batch-service delta per token. Values below are milliseconds per token.

| Configuration | Segment | Wait delta/token | Service delta/token | Component sum | Predicted | Observed | Prediction |
|---|---:|---:|---:|---:|---|---|---|
| `(1,1,8)` | 50 to 100 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,8)` | 100 to 150 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,8)` | 150 to 175 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,8)` | 175 to 200 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,8)` | 200 to 210 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,8)` | 210 to 220 | +0.252341083 | 0.000000000 | +0.252341083 | not queue | queue | MISSED |
| `(1,1,8)` | 220 to 225 | +0.128832920 | -0.020966336 | +0.107866584 | not queue | queue | MISSED |
| `(1,1,8)` | 225 to 230 | +0.006023507 | -0.024460725 | -0.018437218 | queue | not queue | MISSED |
| `(1,1,8)` | 230 to 235 | -0.002843426 | -0.020966336 | -0.023809762 | not queue | not queue | HELD |
| `(1,1,8)` | 235 to 240 | +0.001114171 | -0.020966336 | -0.019852165 | not queue | not queue | HELD |
| `(1,1,8)` | 240 to 245 | -0.002677898 | -0.020966336 | -0.023644234 | not queue | not queue | HELD |
| `(1,1,8)` | 245 to 250 | -0.008739295 | -0.020966336 | -0.029705631 | not queue | not queue | HELD |
| `(1,1,16)` | 50 to 100 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,16)` | 100 to 150 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,16)` | 150 to 175 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,16)` | 175 to 200 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,16)` | 200 to 210 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,1,16)` | 210 to 220 | +0.352600146 | -0.003494389 | +0.349105756 | not queue | queue | MISSED |
| `(1,1,16)` | 220 to 225 | +0.029035545 | -0.020966336 | +0.008069209 | not queue | queue | MISSED |
| `(1,1,16)` | 225 to 230 | +0.001114882 | -0.024460725 | -0.023345843 | queue | not queue | MISSED |
| `(1,1,16)` | 230 to 235 | +0.004145449 | -0.020966336 | -0.016820887 | not queue | not queue | HELD |
| `(1,1,16)` | 235 to 240 | -0.003234517 | -0.024460725 | -0.027695242 | not queue | not queue | HELD |
| `(1,1,16)` | 240 to 245 | -0.001823648 | -0.017471947 | -0.019295594 | not queue | not queue | HELD |
| `(1,1,16)` | 245 to 250 | -0.005244858 | -0.020966336 | -0.026211194 | not queue | not queue | HELD |
| `(1,2,8)` | 50 to 100 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,8)` | 100 to 150 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,8)` | 150 to 175 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,8)` | 175 to 200 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,8)` | 200 to 210 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,8)` | 210 to 220 | +0.252341083 | 0.000000000 | +0.252341083 | not queue | queue | MISSED |
| `(1,2,8)` | 220 to 225 | +0.280332733 | -0.006988779 | +0.273343954 | not queue | queue | MISSED |
| `(1,2,8)` | 225 to 230 | +0.065629944 | -0.027955115 | +0.037674830 | queue | queue | HELD |
| `(1,2,8)` | 230 to 235 | +0.027130699 | -0.020966336 | +0.006164363 | queue | queue | HELD |
| `(1,2,8)` | 235 to 240 | +0.002713546 | -0.020966336 | -0.018252790 | queue | not queue | MISSED |
| `(1,2,8)` | 240 to 245 | +0.000785227 | -0.020966336 | -0.020181109 | not queue | not queue | HELD |
| `(1,2,8)` | 245 to 250 | +0.002566955 | -0.020966336 | -0.018399381 | not queue | not queue | HELD |
| `(1,2,16)` | 50 to 100 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,16)` | 100 to 150 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,16)` | 150 to 175 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,16)` | 175 to 200 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,16)` | 200 to 210 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(1,2,16)` | 210 to 220 | +0.345766146 | 0.000000000 | +0.345766146 | not queue | queue | MISSED |
| `(1,2,16)` | 220 to 225 | +0.199716045 | -0.013977557 | +0.185738488 | not queue | queue | MISSED |
| `(1,2,16)` | 225 to 230 | +0.061606882 | -0.020966336 | +0.040640546 | queue | queue | HELD |
| `(1,2,16)` | 230 to 235 | +0.023612074 | -0.024460725 | -0.000848651 | queue | not queue | MISSED |
| `(1,2,16)` | 235 to 240 | +0.006471983 | -0.020966336 | -0.014494353 | queue | not queue | MISSED |
| `(1,2,16)` | 240 to 245 | +0.005286352 | -0.020966336 | -0.015679984 | not queue | not queue | HELD |
| `(1,2,16)` | 245 to 250 | -0.011258358 | -0.020966336 | -0.032224694 | not queue | not queue | HELD |
| `(2,1,8)` | 50 to 100 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,8)` | 100 to 150 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,8)` | 150 to 175 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,8)` | 175 to 200 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,8)` | 200 to 210 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,8)` | 210 to 220 | +0.252341083 | 0.000000000 | +0.252341083 | not queue | queue | MISSED |
| `(2,1,8)` | 220 to 225 | +0.128832920 | -0.020966336 | +0.107866584 | not queue | queue | MISSED |
| `(2,1,8)` | 225 to 230 | +0.006023507 | -0.024460725 | -0.018437218 | queue | not queue | MISSED |
| `(2,1,8)` | 230 to 235 | -0.002843426 | -0.020966336 | -0.023809762 | not queue | not queue | HELD |
| `(2,1,8)` | 235 to 240 | +0.001114171 | -0.020966336 | -0.019852165 | not queue | not queue | HELD |
| `(2,1,8)` | 240 to 245 | -0.002677898 | -0.020966336 | -0.023644234 | not queue | not queue | HELD |
| `(2,1,8)` | 245 to 250 | -0.008739295 | -0.020966336 | -0.029705631 | not queue | not queue | HELD |
| `(2,1,16)` | 50 to 100 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,16)` | 100 to 150 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,16)` | 150 to 175 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,16)` | 175 to 200 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,16)` | 200 to 210 | 0.000000000 | 0.000000000 | 0.000000000 | not queue | not queue | HELD |
| `(2,1,16)` | 210 to 220 | +0.352600146 | -0.003494389 | +0.349105756 | not queue | queue | MISSED |
| `(2,1,16)` | 220 to 225 | +0.029035545 | -0.020966336 | +0.008069209 | not queue | queue | MISSED |
| `(2,1,16)` | 225 to 230 | +0.001114882 | -0.024460725 | -0.023345843 | queue | not queue | MISSED |
| `(2,1,16)` | 230 to 235 | +0.004145449 | -0.020966336 | -0.016820887 | not queue | not queue | HELD |
| `(2,1,16)` | 235 to 240 | -0.003234517 | -0.024460725 | -0.027695242 | not queue | not queue | HELD |
| `(2,1,16)` | 240 to 245 | -0.001823648 | -0.017471947 | -0.019295594 | not queue | not queue | HELD |
| `(2,1,16)` | 245 to 250 | -0.005244858 | -0.020966336 | -0.026211194 | not queue | not queue | HELD |

## Frozen derivation and refusal

- Lower offered-load ladder: 50, 100, 150, 175, 200, 210, 220, 225, 230, 235, 240, 245 and 250 requests/s.
- Six configurations: pool ratios 1:1, 1:2 and 2:1 crossed with 8-token and 16-token prompts; 64 requests and four decode tokens per cell.
- Held out: load 240 across the non-held-out ratios, plus the entire 2:1 pool ratio. Their union contains 30 component comparisons.
- The numerical queue model consumes only the imported batch-1 and batch-8 measured service/CV rows and deterministic interarrival times. Its shared virtual-clock simulation has no observed curve inputs and no fitted parameters.
- The isolated central onset is 225.108412 requests/s; the frozen surface envelope spans 222.286266 to 228.003140 requests/s. The envelope is three times the maximum measured CV, 12,696 ppm.
- The model and bands were committed before any VLLM-41 lower-ladder observation. The observed curve was never fitted, and the prior 250-to-8,000 requests/s direction was preserved without rescoring.

## Imported surface and logged access

The imported surface remains candidate evidence with calibration claim `false`. It was read through the committed field-addressed reader with five passing ledger events; no whole record, DeepSeek row or held-out batch-32 row was decoded or captured.

| Batch | Service (ps) | CV (ppm) | Replays | Evidence | Key SHA-256 |
|---:|---:|---:|---:|---|---|
| 1 | 1,110,576,000 | 4,232 | 300 | MEASURED calibration | `d8fdebd7051f530bed3232cfeb2e3d5c87a1ab9a22fe68db9cca51436853a502` |
| 8 | 1,892,831,500 | 1,538 | 300 | MEASURED calibration | `38978457b27e56dbc0e9ffbe2385b7d53538a2515c826284be4d6d414520c830` |

## Conservation and preservation

- Cells: 78; admissions, handoffs and terminals: 4,992 / 4,992 / 4,992.
- Terminal decode tokens: 19,968; maximum TTFT decomposition residual: 0 ps.
- Imported-surface candidate/no-calibration pricing held in every request record, and all pool-local identities remained unique.

| Preservation class | Files | Bytes | Manifest SHA-256 |
|---|---:|---:|---|
| Prior VLLM-39/VLLM-40 lineage | 17 | 279,928 | `ae964f9ccecc2554764f9ef69300ca06a84c4a8609682c678063f73c0d41538d` |
| CORE-51 one-request control | 6 | 61,248 | `092d79c35c7632e87427804cda11bc6fe0890d2c98ceec23ff30af2e1143ad4d` |
| Concurrent comparator | 9 | 56,495 | `d09202846afeba9efc019d4f44f881fde6639457c0398f1ed4ae8da1e0c804c3` |
| Scored flagship artifacts | 26 | 1,715,149 | `375d2359e0c9dff9cae98c576eaf8a9e24b0c7621b0af0dcfde187662c57955b` |

## Run evidence

- Scored HEAD: `9d1ad344d9c21fc46c1bfb1c379e692ac231e49f`; freeze commit: `b3e225e6a4b97280c86536bef136e9945cc239fb`.
- Raw result: `$SIMLLM_VLLM41_RUN_ROOT/qualified-sharded-v1/result.json`, 18,833,582 bytes, SHA-256 `0cdd0f2bf6244d7c3daf75cfbaee5e56fa3fcc95bfb4718bbb11f7e5beca0248`.
- Tracked compact result SHA-256: `27ec9540979302625a85d2f1f1866e885bb980df69947dc50ee39e52dad26488`.
- Shard runner exit status: 0; Python 3.10.18; vLLM 0.27.1; offline mode `true`.
- Expectations SHA-256: `859efc475534bd461761a0e34a039594bd52877520a232a91f2f9c4309c73308`; access ledger SHA-256: `8a61a6e0b58a213259a19a593b8b3f4ec08cea6a7c854f5481ccbd7bc2dc5914`; surface SHA-256: `26fc547d8b47ccec7108872e05fbedfe71ebb6229b88799ca254089d3f2b6e9d`.
- The complete run was offline; no model weights or web content were downloaded.
- An initial infrastructure-only directory stopped before constructing a session because the historical helper import resolved to the new queue-model namespace. It remains retained and unscored. A separate sequential duplicate was stopped after the complete sharded result; its partial external directory and log remain retained and unscored. The published evidence is only the complete 78-cell sharded registry above.

## Registry movement

- **VLLM-41 closed**: all six configurations identify the common 210 to 220 requests/s first queue-dominated segment, strictly below 250 with five preceding non-queue-dominated segments.
- **VLLM-42 registered**: 16 batching-service component bands missed without widening, despite all 30 scheduler-wait bands holding.
- **VLLM-43 unused**: all six configurations share one resolved onset segment.
- The validated monotonic direction over 250 to 8,000 requests/s was preserved and not reopened.
