# Transport-placement and per-channel peer observations

This amendment freezes a separate diagnostic capture before its implementation
and first invocation. The ordinary timing campaign and its manifest are
unchanged. Run diagnostics only in a new allocation after that architecture's
timing allocation completes. Diagnostic timings are never calibration data.

Use the same node, GPU identities, NCCL library, timing executable and observer
binary as the completed campaign. Capture two/four ranks; requested channels
`{0,1,8,22,23,24,32}`; LL, LL128 and Simple; payloads 1 MiB and 3.25 MiB;
one warmup and one timed iteration; ordinary streams, persistent workers,
64-MiB allocations and no rotation. Zero channels means default tuning. Other
channel requests set both channel limits and zero thread thresholds, as in
the original campaign. Every protocol uses its default thread request.

Enable informational INIT, GRAPH and P2P logging and the existing selection
observer. Save the Ring order printed for every communicator channel, the
directed connection log and its read/write placement, and the selected active
channel/warp counts. The transport's read suffix is a connection capability;
the pinned source uses sender-side FIFO storage only for Simple. LL and
LL128 keep receiver-side FIFO storage even on a read-capable connection.

For Simple only, additionally request `NCCL_P2P_READ_ENABLE=0` and `=1`.
The default is left unset. This yields 70 processes per architecture. The
source predicts that the default direct Ampere NVLink path selects read
placement, while the direct Hopper path selects write placement. Explicit
zero/one requests should select write/read respectively, unless the observed
transport is an unsupported intermediate or copy-engine path. Such a result
is a source-qualification finding, never an invitation to relabel the path.

Fatal diagnostic guards are library/probe/observer identity, empty initial
and final GPU process inventories, float32 sum correctness, finite positive
timers, complete rank and payload inventory, a complete permutation for each
printed Ring, and directed connections consistent with those Rings. A missing
or conflicting branch/ring record makes that configuration unqualified. It
does not retroactively invalidate otherwise qualified ordinary timing data,
but those timings cannot qualify a model that assumes an unobserved branch.

No timing direction or speedup is scored here. No fitted constant is inferred
from debug-enabled processes. This capture identifies which source path must
be modeled and supplies peer-map evidence. Actual source-slot/lane traces,
instruction costs and a fresh validation remain TRAF-54 and TRAF-43 work.
