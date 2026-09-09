# Shared session wall-deadline amendment

This expectations-only amendment follows source inspection at `ab21905` and
precedes its timeout implementation and every shared native campaign. The
original shared-handoff freeze remains the authority for modeled time, packet
geometry, workload, comparisons and evidence classes. Its statement that the
existing 60-second limit applies per packet call was incorrect: the current
binary-stream timer covers the complete child lifetime, including idle time
while Python executes serving work. No shared native outcome has been read.

Preserve that legacy default exactly. Add an optional complete-exchange bound
through the existing binary-stream and FlowSession interfaces. Shared serving
selects a 1,800-second child lifetime and a 60-second exchange deadline. The
outer native worker keeps its frozen 1,800-second and 16-GiB limits. The
lifetimes start at different events; neither refreshes or extends the other.
This amendment replaces only the earlier mapping of the 60-second setting to
the stream lifetime. It changes no simulated packet or serving timestamp.

One exchange budget spans request writing, response header and body reading,
and response validation. It never restarts between frame parts. Time spent
by the client between exchanges does not consume this exchange budget. The
fixed lifetime still caps every exchange. Overlapping or nested contexts
reject before changing their active deadline. A timed-out exchange poisons and
reaps its child. Cleanup preserves the first exception and retains partial
protocol evidence. Close and end-of-stream checking remain bounded.

Freeze an unscored component grid with exchange limits of one and two seconds
and client idle intervals of three and six seconds, under a thirty-second
lifetime. A deterministic clock fixture checks that the active deadline is
exactly the lesser of lifetime expiry and entry time plus exchange allowance.
A real echo child completes one response, remains the same live process over
the idle interval, then completes the same response again inside a new
exchange. The complete response bytes match the legacy default path. These
are host-I/O conformance cases, not packet or GPU measurements and not added
to the shared study's behavioral denominator. Separate controls cover a
stalled exchange, cumulative frame-part budgeting, fixed lifetime expiry,
overlap rejection, terminal cleanup and preservation of the original error.

The complete forty-request campaign still has to admit its frozen two-rate,
two-prompt and shared-destination sweeps, full source-pair legacy comparison
and exact packet-to-serving schedules. Component deadline evidence alone
cannot close CORE-71 or any deployment milestone. The first and amended
expectations commits and actual chronology are cited together in the result.
