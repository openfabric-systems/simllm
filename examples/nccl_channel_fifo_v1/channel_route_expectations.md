# Per-channel Ring peer bindings: expectations

This amendment precedes the optional per-channel permutation implementation
and its first run. The existing common-order descriptor is an explicit
declared fixture, not a captured Merlin channel map. Permit each channel to
carry its own permutation of the communicator's two or four participants.
The actual GPU rank, its position in that channel's Ring, the physical peer
and the software connection identity remain distinct.

For four participants, use channel orders `(0,1,2,3)`, `(0,2,1,3)` and
`(0,3,1,2)`. GPU 0 sends the three channels to GPUs 1, 2 and 3 respectively;
its returned-capacity destinations follow each channel's predecessor. Derive
chunk order from the channel-local Ring position. Useful bytes remain exactly
`2(n-1)S`, and each rank writes exactly S final output bytes. Repeat this
fixture for LL, LL128 and Simple and for one/four available SMs. Identity and
byte conservation are fatal guards, not scored behavior.

An omitted map and an explicit repetition of the existing common order have
identical timestamps, transfers and resource work. An invalid permutation,
wrong channel count or missing selected route rejects before any mutation of
the retained packet session. Snapshot input sequences so caller mutation
cannot alter a validated descriptor. Expose the optional map through the same
source-execution configuration and original-graph path.

This implements a binding supplied by an observer or a declared study. It
does not infer the hardware chooser's mapping from channel count and does
not establish that the current Merlin selection observer captured that map.
Captured bindings and source-transition traces remain TRAF-54 qualification
work; calibrated hardware curves remain TRAF-43 work.
