# Local packet critical-path qualification

The retained packet and serving campaign passes: all 32 exact outcome vectors
and all 24 behavioral instances hold, with every fatal guard satisfied.
BACK-73 closes for the supported physical peer-write path. The report explains
the model's elapsed time from the decisions that produced it. Enabling that
report preserves the original packet data and request timing exactly.

This result qualifies an explanation, not hardware parameters. TRAF-92 retains
physical calibration, BACK-74 native queue and credit ownership, and TRAF-54
collective protocol. The fixed 37-nanosecond compute value is a synthetic
sentinel that makes transport accounting visible. These numbers are not
predictions of a real model's serving performance.

## What ran

Sixteen component configurations and sixteen live configurations vary direct
versus switched attachments, 12.5 versus 25 billion bytes per second, one
versus three donors, and 256 versus 1024 payload bytes per donor. Every cell
runs the frozen predecessor with reporting absent, the candidate with reporting
absent, and the candidate with Python reporting enabled. The eight switched
cells also run the native allocator with reporting enabled.

That is 56 component runs, 56 serving sessions and 168 actual step calls.
Each live session executes one prefill and two decode steps through the
original execution graph, completion events, StepResult and request reducer.
The two native build stages and 56 capture processes all complete. No GPU
allocation or hardware measurement runs.

The expectations-only commit is
`2919cf2dbf2811647b3c0e43a677586899f2b4eb`. It precedes implementation and the
first external campaign. The executed source is
`7293b6018b73ef3949bd6a5b186e8cccd5db0d1d`; the unchanged predecessor is
`861d7a45ff2e3b8b8ed438d79ca514960d52e438`.
The [frozen expectations](expectations.md) and [compact receipt](results.json)
carry the exact grid and evidence identity.

## Physical and causal checks

Floor: a receiver cannot finish before accepting its cumulative wire bytes
divided by its service rate. The first packet also pays each required serial
service and propagation stage.

Ceiling: serializing every resource-service visit and its propagation bounds
the entire finite phase. This is a deliberately loose work bound because
different resources operate at the same time.

A packet carries 256 payload bytes in 272 wire bytes. Write its serialization
as q and per-hop propagation as p. For d donors with m packets each, the
direct component completes at (d*m+1)*q+p and the switched component at
(d*m+2)*q+2*p. Each live step pays the fixed compute value and one dispatch
plus one combine phase. The original request records match those equations.

| Attachment | Rate, billion bytes/s | Donors | Payload/donor, bytes | TTFT and TPOT, ns | Last completion, ns |
|---|---:|---:|---:|---:|---:|
| Direct | 12.5 | 3 | 1024 | 604.76 | 1814.28 |
| Direct | 25 | 3 | 1024 | 321.88 | 965.64 |
| Switched | 12.5 | 3 | 1024 | 650.28 | 1950.84 |
| Switched | 25 | 3 | 1024 | 345.64 | 1036.92 |

TTFT means time to first token; TPOT means time per subsequent output token.
Doubling the rate halves serialization after subtracting fixed compute and
propagation. Increasing payload or donor count adds the exact newly required
receiver service. Eight rate, eight payload and eight donor comparisons hold.

The explanation follows actual resource owners. In the 25-billion-byte/s,
three-donor, 1024-byte component cases:

| Attachment | Phase wall duration, ns | Sum of visit queue waits, ns | Sum of service work, ns | Serial service-plus-propagation ceiling, ns |
|---|---:|---:|---:|---:|
| Direct | 142.44 | 913.92 | 391.68 | 403.68 |
| Switched | 154.32 | 1958.40 | 783.36 | 807.36 |

The visit sums exceed wall duration because they count simultaneous work or
waiting separately. Neither sum becomes an additive request-latency term.
Selected positive causal intervals instead cover the phase exactly once.
Source feed and wire transmission can tie; the stable first-parent rule
selects one branch while preserving the other co-critical predecessor.

The runtime records source eligibility and pacing, directed-link release,
finite buffer and credit return, switch grants, receiver service and ordered
consumer visibility. Native grants retain their native owner. Older control
tails retain their packet identity when the report clips an interval at a
later phase's release. The reader rejects an invented same-time service parent
even when a forged chain is internally time-consistent.

## Evidence and limits

The 32 exact oracle vectors, 24 behavioral instances in three families, and
13,756 fatal guards are separate evidence classes. Fatal guards include source,
receipt, physical, identity-off, native/Python and reporting consistency
conditions. They do not increase the behavioral score. No fatal guard fails.

The original 1,197 retained files total 83,546,550 bytes. Every file is rehashed
against its first receipt after execution, with the complete file domain
checked again before publication. Each child records its source before and
after execution, loaded origins, original inputs and outputs. Native source,
compiler, build commands and library are bound before the first native call.
The native compiler is GNU C++ 12.2.1. Raw files and builds remain outside Git.

| Receipt | SHA-256 |
|---|---|
| Original summary | `01b3b18f249b1f153eacf9b240643ed52de10503cb0ffa22fa8e639bfedd569f` |
| Original checks | `9aadd62fe1451bfe51413f7b253894579b616c7aa4bbef427df63a829bebc9cc` |

Software controls separately cover one-packet capacity and credits, retained
control tails, producer pacing, slow receivers, out-of-order arrivals, direct
and disjoint paths, both switch policies, and all 56 ordered GPU pairs on
each declared eight-GPU A100 and H100 preset. Capture corruption controls
reject changed files, source identities, packet times, causal parents, phase
membership, request histories and incomplete drains. Storage failures preserve
the original failure, and a failed run has a null behavioral score.

These controls do not identify deployed packet sizes, buffers, arbitration or
credits. Native/Python equality does not prove hardware fidelity. The Merlin
four-GPU direct meshes and an eight-GPU switched board remain separate
calibration targets under TRAF-92. No large-model performance claim changes.

## Reproduction

Configure SIMLLM_PEER_SOURCE and SIMLLM_PEER_BASELINE as clean checkouts of the
executed and predecessor commits above. Configure SIMLLM_PYTHON to the project
Python environment and SIMLLM_STUDY_OUTPUT to a new directory on bulk storage.
The command builds the optional native library within that output directory.

```bash
cd "$SIMLLM_PEER_SOURCE"
"$SIMLLM_PYTHON" -m examples.peer_critical_path_v1.run_study \
  --baseline-root "$SIMLLM_PEER_BASELINE" \
  --output "$SIMLLM_STUDY_OUTPUT"
```

The command refuses dirty source trees and an existing output directory. A
retry uses a new directory and retains any original VOID record unchanged.
