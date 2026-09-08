# Bounded initial sending and prompt data recovery

This successor contract freezes HTSIM-41 before implementation and before
its first simulation. The original control-recovery, collective-width and
pipeline-contention records keep their original verdicts and bytes.

## Mechanisms and authority

Two independent selections extend the existing collective runtime. Both
default to disabled. The initial-window selection permits at most U wire
bytes per flow before an in-band nonempty receiver grant arrives. Its input
declares an upper bound F on flows converging on one shared switch pool and
requires F*U <= B, where B is that pool's unchanged configured capacity.
Packets are indivisible: a head larger than the remaining budget waits for
the grant. U may be zero. A receiver whose declaration arrived in an empty
snapshot window sends its first nonempty ACCEPT at a subsequent control
boundary even when no DATA can yet arrive. Endpoint control packets retain
their physical routes, serialization and lifecycle.

The recovery selection uses a DATA retransmission as a physical tail probe
after m control deadlines from the preceding retry's source serialization
end. This is a separate timer and counted transmission, not a change to the
legacy long retransmission timeout. The legacy timer remains available as
fallback. Normal receiver RETIRE processing still detects a missing original
tail. No switch-drop callback, mutable switch occupancy or future completion
record authorizes a retry.

An authenticated retry for an exact receiver-owned missing extent may enter
the recovery admission path after its normal reorder window expired. The
packet keeps its carried timestamp and extent. Its logical release is the
first receive tick at or after actual arrival. It uses the same finite
receive storage and physical receive serializer as normal DATA. Originals
retain strict window admission. Recovery admissions and probe bytes are
reported separately. This does not certify the stronger lossless-window
invariant of the algorithm book.

Receipt of a physical gap-resolution control terminally resolves that
logical extent at the sender. It cancels all queued retries and deadlines for
that extent; already-routed duplicates still drain physically. A stale or
duplicate control cannot reopen a terminal extent or retire another extent.
The existing maximum of eight retransmissions remains unchanged. Permanent
loss must still fail explicitly.

## Identity boundary

The protected legacy set is the 44 completed physical configurations at the
pre-headroom reference pin: the completed collective-width inputs and the
completed pipeline-contention inputs. The formerly failing width-64 and six
oversubscribed pipeline configurations are outside that completed set.

The new binary with both selections disabled must reproduce every protected
completion CSV exactly, with the same control selection as its reference.
Recovery enabled must also reproduce no-recovery-needed fixtures exactly.
An enabled initial window must preserve exact packet order, timestamps,
random draws and bytes when its budget is dormant. The original acceptance
phrase "every cell that completed before" is thus made explicit before the
run: it does not require an active byte limit to leave the traffic it delays
unchanged, and it does not protect the long-tail completions that this task
must improve. A retrospective per-cell bypass based on observed success is
forbidden. Combined-enabled completion evidence is required separately.

## Native mechanism families

Freeze control deadlines K in {5, 10} microseconds and probe multiples m in
{2, 4}. Compare an original final-packet drop with original-plus-first-retry
loss. The second retry cannot serialize before the first retry's physical
serialization end plus m*K. Under isolated service, changing m or K moves
the probe epoch by the exact changed interval, with any completion residual
bounded by the existing receive-tick and source-opportunity quantization.
Changing only the legacy timeout from 50 to 100 milliseconds must leave
prompt-path completion, packet count and order exact. A disabled probe keeps
the legacy timeout behavior.

Freeze receive arrivals at timestamp-plus-window minus one, equality, plus
one and two windows later. Strict admission rejects late packets. Recovery
admits authenticated late retries only, cannot release before arrival, and
charges the original extent exactly once on the shared serializer. Vary
receive capacity over one and two maximum wire packets and packet extents
over maximum and short final sizes. Full storage rejects admission without
manufacturing delivery or exceeding the capacity.

Exercise an older successful attempt racing a newly authorized or routed
probe; closure before, at and after a probe boundary; duplicate DATA and
stale negative acknowledgements; two gaps in one flow and gaps in two flows;
and a permanently dropped logical extent. Require one completion, exact
payload conservation, independent timer cancellation and physical quiescence.
These are fatal structural and identity checks, not behavioral score points.

For the initial window, freeze U in {0, one packet minus one, one packet,
two packets, full flow}, with F and B values at the exact multiplication
boundary and one byte below it. Before a physical grant, serialized wire
bytes cannot exceed U. Zero budget must progress through physical ACCEPT.
An unconstraining enabled budget is exactly identical to disabled. Invalid
profile, incomplete configuration, negative/overflowed values and violated
sizing bounds must reject before execution.

## Consumer matrix and physical bounds

Use the unchanged saved GOAL binaries, topology files, seeds and ideal
completion references from the collective-width and pipeline-contention
studies. The main all-to-all matrix is widths {8,16,32,64}, endpoint rates
{200,400} Gbit/s, and arms {disabled, recovery-only, window-only, combined}.
Control headroom remains enabled for mechanism arms; its legacy none
selection remains an explicit identity control. Primary probe multiple is
4. For the combined/window arms, F is a conservative declared upper bound
derived from all input sends targeting each leaf, before any run, and
U=floor(B/F). This bound may overestimate simultaneous membership and must
be reported as an input assumption, not an observed occupancy.

Run the six 4:1, expert-width-32 pipeline cases at pipeline depths {2,4,8}
on both rail and node-local attachments with combined recovery. Retain the
same payloads, schedules, ideal references and seeds. The recovery-only arm
on these inputs supplies a mechanism contrast. Failed standalone window-only
or legacy arms are expected diagnostics with null phase values; failure in a
combined acceptance cell is fatal. No failed phase may be reconstructed from
its completed subset.

For every flow, the physical floor is payload bytes divided by endpoint byte
rate plus at least path propagation. Its unconditional ceiling is unbounded.
For aligned all-to-all at width W, one receiver receives D=7W/8 flows of
65536 bytes. Its phase floor is D*65536*8/rate + 4 microseconds. Every
receiver's k earliest completions must also exceed cumulative payload/rate
plus this propagation. Complete physical/ideal phase ratio has floor one.
For pipeline phases, retain the saved receiver and causal/cut floors; do not
apply a universal per-flow ideal floor under shared or dynamic membership.

A finite engineering acceptance budget is distinct from these physical
bounds. Let S be the maximum receiver serialization floor, strengthened by
the saved pipeline causal/cut floor. Let Q be three shared-buffer drains at
the slowest traversed data-link rate, plus eight propagation hops and one
maximum-packet source serialization. Define budget
T_budget = 9*S + 8*(m*K + Q).
The nine service terms allow an original plus eight attempts; the eight
recovery allowances include each probe interval and conditional queue drain.
This is a deliberately conservative registered workload budget assuming
bounded control interference and fair service under the fixed sender seed.
It is not a universal theorem about pseudorandom pacing or strict priority.
Record T_budget/ideal_phase before execution and require every combined
cell to complete below it. Width-64 must additionally finish before one
millisecond, reflecting submillisecond finite packet work and deadline-based
recovery; violating either budget is a behavioral finding, not permission
to revise the freeze. Holding K fixed does not imply exact inverse-bandwidth
scaling of total latency: serialization scales with inverse rate while
propagation and probe intervals stay constant.

## Fatal guards and reporting

Combined cells must quiesce with every input flow completed exactly once,
nonnegative causal timestamps, finite storage accounting and unchanged
maximum retry policy. Disabled identity and dormant-budget identity are
fatal. No fatal guard is survivable. A violated guard voids the successor
study for task closure, and its full evidence is retained without a
behavioral score fraction. Window-only failures and explicitly frozen
legacy failures are diagnostics, not fatal acceptance cells.

Keep configuration counts, exact CSV oracles, behavioral relation families,
native executables and fatal invariants separate. Evaluate behavioral
comparisons directly from observations before exact-oracle guards. Preserve
configuration, implementation and input digests; write all raw outputs to
SIMLLM_DATA_ROOT outside Git. Failed cells have null timing metrics.

This study can close HTSIM-41 and the HTSIM-40 recovery remainder only after
the paired backend integration, all native/Python gates and combined consumer
evidence satisfy the contract. TRAF-88 then becomes unblocked, not completed:
its load-dependent topology attribution and queue evidence remain separate.
No calibrated request-level latency claim or physical multi-artifact serving
closure follows; BACK-38 and TRAF-8 keep those scopes. CORE-48 is the next
separate task and is not changed by this study.
