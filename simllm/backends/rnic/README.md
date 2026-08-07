# Native RNIC queue core

This directory contains the SimLLM-owned C++17 RNIC hardware core. The module
design, status and open-task registry remain in
[`docs/modules/backends.md`](../../../docs/modules/backends.md).

The implemented v1 slice is one finite SQ and CQ bound to one QP. It models
accepted-prefix WR posting, explicit doorbell batches, serialized fetch and
CQE-write service, ordered retirement, signaled and unsignaled reclaim, CQ
owner wrap, polling, network retry gates and controlled queue failures.

## Network boundary

`NetworkPort` is independent of Python, htsim and any congestion-control
algorithm. A submitted descriptor carries:

- opaque WQE/WR correlation IDs;
- GOAL flow ID and tag;
- one stable opaque policy-context token;
- source, destination, traffic class, payload extent and eligibility time.

The port returns a network-owned token and later returns one delivery or drop
event for that token. A Busy result retains the SQ head until its advertised
retry time. Completion of another token does not revoke that deadline.

This first port admits one flow extent per WQE. Network acceptance and outcome
times are real v1 observations. First/last packet timestamps stay unset until
HTSIM-9 adds explicit packet-issue events, so flow admission is never mislabeled
as NIC packet start.

At one timestamp, deliver network events to `onNetworkEvent` before retrying
the SQ with `progress`. CQ priority is then explicit call order. Calling
`progress(t)` before `pollCompletionQueue(t)` gives device CQE publication
priority. Polling first gives host consumption priority and sees only CQEs
strictly older than the timestamp; CQEs due exactly at that timestamp remain
host-first. A fatal CQ overrun remains non-quiescent but exposes no next event,
so event loops must test `fatal()` and abort rather than spin.

## Standalone build

```bash
cmake -S simllm/backends/rnic -B build/rnic \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON
cmake --build build/rnic --parallel
ctest --test-dir build/rnic --output-on-failure
```

When this directory is consumed with `add_subdirectory`, tests and validation
tools default off. The link target is `simllm::rnic`.
