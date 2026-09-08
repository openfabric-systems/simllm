# External congestion notification ownership

The repaired native interface and the frozen eight-configuration physical
comparison complete successfully. Turning packet and control observations on
changes none of the 24 paired flow completion rows or their order: the maximum
timestamp difference is **0 ps**. BACK-71 closes, restoring the composed native
validation gate and unblocking BACK-38's session qualification. This result
does not calibrate congestion control or establish serving latency; BACK-38
has its own [completion-boundary expectations](expectations.md).

## Mechanism and chronology

When a switch reports congestion, the external sender already owns the change
to its sending rate. The native work queue checks the notification's token,
payload extent and advertised capability, then observes that external action.
A regression instead required a second, native transmit reaction point for
every notification. The original physical fixture therefore failed before it
could finish its validation.

The repair routes a notification to the native reaction point only when the
native transmit pipeline owns the flow. External notifications retain their
existing validated work-queue path, including tokens retained after delivery.
A native pipeline with its reaction point disabled still rejects the event;
an enabled reaction point handles it exactly once. No controller, packet
policy, timing constant or topology changes.

The original failure preceded this work and is retained as diagnostic history.
Expectations-only commit `d6ba3d37b65f73e26c4364e98c18beebc99e0b61`
then froze the [ownership checks](cnp_owner_checks.md), before the routing
repair and first new matrix execution. The first repaired matrix and the
integrated rerun have identical completion rows, boundaries and control
counts. The [machine-readable record](cnp-results.json) retains all 24 paired
rows, all four comparison records, source identities and evidence hashes.

## Physical bounds and measured completions

Floor: the shared receiver needs at least total payload bytes divided by its
400 Gbit/s rate, plus two microseconds for the shortest two-link propagation
path. A byte takes 20 ps at that rate.

Ceiling: each finite fixture must complete and quiesce within one millisecond
and 100,000 callbacks. This is the frozen engineering guard, not a prediction
of the congestion controller's scaling law.

The matrix uses four or eight senders and 64 or 128 KiB per sender on the
existing 64-endpoint Clos. Explicit congestion marking uses seed nine;
routing uses seed one. Packet/control observation is absent or present for
each of those four inputs. The absent mode uses interface version one; the
present mode uses version two with the required packet observation capability.
Both retain the same external controller and native work-queue authority.

| Senders | Payload per sender | Physical floor | Last completion and quiescence | Callbacks, absent / present |
|---:|---:|---:|---:|---:|
| 4 | 64 KiB | 7.242880 us | 15.500160 us | 1,973 / 2,254 |
| 4 | 128 KiB | 12.485760 us | 24.254080 us | 3,765 / 4,309 |
| 8 | 64 KiB | 12.485760 us | 20.666240 us | 3,919 / 4,418 |
| 8 | 128 KiB | 22.971520 us | 33.759360 us | 7,511 / 8,555 |

Every boundary sits above its independently calculated payload/propagation
floor and below the frozen ceiling. The largest callback count is 8,555.
Observation adds callbacks while preserving every physical completion, which
is the required separation between observation and control ownership. The
matrix claims no monotone congestion-control law between its four load points.

## Validation and reproduction

All fatal population, identity, payload, observation, physical-floor,
quiescence and callback guards are clear. Evidence classes remain separate:

- Eight physical configurations form four observation comparisons with 24
  exact completion-row pairs. These are compatibility controls, not a
  behavioral pass score.
- The complete composed native suite passes all 587 discovered test cases,
  including the original congestion-marking, rate-change and pause/resume
  fixture. The eight matrix configurations are included in this suite; their
  count is not added to it.
- The complete standalone native interface suite passes all eight test
  executables. It covers external validation and native rejection and
  exactly-once reaction.

With the composed native build and external evidence paths configured:

```bash
ctest --test-dir "$HTSIM_BUILD" -V -R 'Back71/' > "$CNP_LOG"
python examples/completion_boundary_v1/check_cnp_observations.py \
  --log "$CNP_LOG" --output "$CNP_RESULT"
```

Each native configuration runs in its own process because the event list has
process lifetime. The checker rejects missing or duplicated records, changed
message identities or order, even a one-picosecond timing change, and every
physical or finite-budget violation. A violated guard writes a void result
without a partial score. Raw logs remain outside Git; their hashes in the
result include the original failure and both successful matrix executions.

The control-ownership defect is closed. BACK-38's retained-artifact request
study, BRIDGE-2's online framed serving client and hardware calibration keep
their separate acceptance bars.
