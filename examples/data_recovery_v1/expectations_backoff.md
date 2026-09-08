# Exponential tail probes after the constant-probe study

This is a prospective successor to [the initial contract](expectations.md).
It freezes an additional explicit recovery selection before its implementation
or first run. The earlier constant-probe study remains void and keeps its
original artifacts and implementation identity. Its results are not rescored
under this contract. Every unchanged mechanism, input, physical floor,
identity guard and task boundary in the initial contract still applies.

## Retry schedule and authority

The added `exponential` selection shares the existing DATA recovery authority,
late-retry admission and initial-window interface. `none` remains the exact
legacy default. `deadline` keeps constant probe spacing. No queue capacity,
receiver window, route, random generator or maximum retry count changes.

Let P=m*K, where K is the control deadline and m the configured probe multiple.
After physical source serialization of retry a, for 1 <= a < R, exponential
recovery makes retry a+1 eligible at end(a) + P*2^(a-1). R is the existing
maximum retry count, eight in every consumer cell. Each logical extent owns
its attempt history. Recovery of another extent cannot reset that history.
A receiver negative acknowledgement still requests its retry immediately;
backoff applies only to the speculative tail probe. No switch-drop callback,
mutable queue state or remote receive state drives the sender's decision.

At R, no further probe is armed because no next retry is legal. The existing
long timeout remains armed from end(R). A physical resolution may close the
extent before that timeout, including after a hypothetical next probe epoch.
If every copy remains lost, the original timeout fails explicitly without
sending retry R+1. Apply this terminal-timer correction to both enabled
recovery selections. It preserves the `none` path exactly.

At primary m=4 and K=10 microseconds, the seven intervals preceding retry eight
are 40, 80, 160, 320, 640, 1280 and 2560 microseconds. Their sum is 127*P,
or 5.08 milliseconds. These are eligibility intervals; source contention may
add time before actual dispatch. Arithmetic is checked before execution for
every legal configured attempt. Configuration that cannot represent its
largest interval in an unsigned 64-bit timestamp is rejected. Exponential
selection also requires the long timeout to exceed its largest probe interval,
so a configured fallback cannot silently preempt the selected schedule.

The design borrows the established distinction between a speculative probe
and a loss declaration, and the use of increasing probe intervals during
repeated loss, from [RFC 9002 sections 6.2 and 6.2.1](https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2).
This model does not claim to implement QUIC or calibrate any physical RNIC's
firmware policy. The interval starts from the configured control deadline;
it is not a measured round-trip-time estimator.

## Native relations and fatal boundaries

Freeze K in {5,10} microseconds, m in {2,4}, and lost retry prefixes of length
{1,3,7} after a dropped original. Under isolated service, consecutive physical
retry serialization starts differ by the preceding packet's serialization,
the exact exponential interval and the existing source opportunity
quantization. Summed intervals preceding retry n equal P*(2^(n-1)-1).
Doubling K or m doubles that timer contribution exactly. The first probe is
identical to the constant selection. Changing only the long timeout from
50 to 100 milliseconds keeps successful prompt completion and bytes exact.

Drop all eight retries and require failure at the eighth retry's physical
serialization end plus the unchanged long timeout, never an extra DATA
packet. Separately, deliver the final legal retry so its physical resolution
arrives at the next nominal constant probe boundary minus one picosecond,
equality and plus one. All three complete exactly once and quiesce. Derive
this boundary from one-packet forward and reverse serialization, configured
propagation and the receive release window before execution.

Retain the earlier older-attempt race, physically cancelled queued probe,
duplicate resolution, stale negative acknowledgement, two gaps in one flow,
and gaps in two flows. Add simultaneous lost-prefix flows under exponential
selection so one flow's resolution cannot cancel another's timer. Count only
physically dispatched probe packets and wire bytes. No-loss and dormant-window
identities include dispatch order, event timestamps, random-generator state
and exact bytes. Initial budget checks cover all DATA until actual grant
arrival, including retries. These are fatal invariants, not scored successes.

## Consumer acceptance and reporting

Repeat the same 96 configurations and 52 saved inputs. Recovery-only and
combined arms select `exponential`; disabled and window-only arms retain their
original selections. Record the selected policy and its interval rule in the
manifest, configuration lock and run provenance. The 44 legacy CSV identities
and the predeclared two dormant-recovery CSV identities remain fatal guards.

Keep every original physical floor, finite-storage guard, original engineering
budget and the width-64 submillisecond target unchanged. In particular, do not
increase the budget to absorb the longer retry intervals. Record 127*P
separately as the maximum cumulative probe-timer allowance before retry eight.
This allowance fits the original pipeline budgets but is not proof that the
finite buffers or deterministic arbitration must recover every packet.

Require all combined cells to complete and physically quiesce below the
original budgets. A fatal failure makes this successor void. A completed cell
outside a budget remains a behavioral finding and prevents closure. Compare
bandwidth and width behavior with serialization and fixed timer contributions
kept separate. Preserve raw observations before oracle checks, immutable
source/input/binary identities, and null timing for every failed phase.

Report the initial void study, the diagnostic replay and this successor as
separate evidence classes and chronology. Only a valid successor, paired
backend integration and full native/Python gates can close HTSIM-41 and its
HTSIM-40 remainder. TRAF-88 becomes unblocked only then. Neither physical
request calibration nor BACK-38, TRAF-8 or CORE-48 closes through this study.
