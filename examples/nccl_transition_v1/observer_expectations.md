# Direct observer amendment, before its implementation and run

The timing freeze `c68b0754` and capture implementation `a1cbd6e4` precede
A100 job 204703. That job completes all 105 process records. Its nccl-tests
profiler loads the v7 interface, but the per-size tuning summaries contain
`N/A`. Those summaries do not identify a mechanism. This observation is the
reason for this prospective amendment, not an outcome to erase or relabel.

Keep every original timing, control, repeat and hypothesis unchanged. Add a
diagnostic-only observer implementing NCCL's published v7 profiler interface.
At each collective callback it records the communicator width, rank, sequence
number, operation, element count, datatype, in-place status, algorithm,
protocol, channel count, warp count and kernel variant. A locked append-only
file records complete callback rows; no aggregate readiness flag is consulted.
No callback timestamp or instrumented duration enters the timing figures.

Run the same 24 diagnostic arms, grids and five warmup/five timed iterations
as the initial freeze. Disable nccl-tests' static profiler override and select
the observer library explicitly. Add one diagnostic pass of the inherited
dense harness at both widths, with its original five/twenty iteration counts,
to identify choices in the exact timing method used by the older figure.
The first arm is a qualification of callback availability, not a performance
acceptance test. Missing callbacks stop the observer campaign and remain a
finding. No latency is fitted or rescaled during observer development.

An observer result qualifies only when every frozen payload and rank has an
all-reduce callback with float32 elements and nonempty algorithm/protocol/
channel/warp fields. For each payload, report all distinct tuples across
ranks and calls; disagreement forbids representing the selection as one
choice. Check in-place and out-of-place selections separately. A unique
out-of-place tuple supplies the diagnostic join to the timing table.
Matching inherited and nccl-tests tuples identifies transfer between their
timing methods; any mismatch is explicitly retained.

The already completed A100 timing allocation may require a separate observer
allocation. Record both GPU identities and require the same GPU generation,
participant placement and NVLink topology before transferring selections.
For GH200, the still-pending original job may be superseded before it runs by
a combined capture-and-observer job, preserving the timing plan exactly and
recording the new implementation commit. Never overwrite the earlier A100
capture or claim that the failed summary observer provided valid choices.

Fatal observer failures void its mechanism attribution only. They do not
invalidate the separate, uninstrumented timing lane. Frozen H3 and H4 remain
unresolved until qualified direct observations exist. Passing this amendment
closes no model-accuracy requirement; TRAF-43 stays open.
