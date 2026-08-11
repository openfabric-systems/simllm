# RNIC packet-event ABI v2 fix-round 1 expectations

## Status and chronology

This is a separate expectations-only record for the first review correction
to BACK-25 and BACK-26. It does not edit the original freeze at
`506f87af93687ccf0df85f6b5307b71a20ed3762`. The audited implementation tips
are SimLLM `4effe2be9fb7308396debb60322d6e49bade267c` and htsim
`63e2eb6437ef15b4bb039ce94fe647b7b488dbde`. No fix-round implementation,
test, measured result or outcome-dependent assertion is present in this file.

These checks are post-specified review regressions. They are not described as
public pre-registration and do not retroactively change the chronology of the
original packet study.

## Source audit before the fix

- The packet study calls the inherited Tier A checker and then
  `_validate_packet_cell` for every row before evaluating its ten packet
  relations at `examples/rnic_packet_v2/run_study.py:369-427`. The exact
  per-packet oracle therefore entails every later relation in a run that
  reaches the reported packet-family counts.
- The frozen FIFO fatal checker used the cell doorbell time for every WQE. A
  post-freeze smoke exposed the conflict with W1's capacity-one serializer
  grant. The correction to each WQE's `port_tx_at_ps` landed in implementation
  commit `fad1dcf277bab950035e35cd76c83fe1ec3db4f2`, while result prose
  incorrectly attributed it to the later runner-path commit `c54d556`.
- A runtime-bound submission snapshots the token and event-sequence cursors at
  `htsim/sim/simllm_htsim_network_port.cpp:529-531`. Failure restores both at
  lines 596-600, but rollback at lines 814-840 erases scheduled events only
  when their token or parent token equals the provisional extent. A global PFC
  or link event can therefore survive while its sequence number is reused.
- Packet terminals erase their runtime correlation at
  `htsim/sim/simllm_htsim_network_port.cpp:676-684`. ECN and CNP lookup at
  lines 693-704 accepts only that live map. SimLLM similarly erases a packet
  attempt at `simllm/backends/rnic/src/work_queue.cpp:1097,1107` and validates
  feedback only against the live map at lines 1136-1145.
- The ABI-v1 unbound path chooses `drop_first` while scheduling the submitted
  extent at `htsim/sim/simllm_htsim_network_port.cpp:573-583`. The accepted
  pre-v2 implementation instead selected the first due terminal during
  consumption, so capacity greater than one can distinguish the two rules.
- `RnicPacketizedManifoldRuntime::eventCapabilities()` advertises only packet
  attempts at `htsim/sim/rnic_packetized_manifold_runtime.cpp:154-159`.
  ECN/CNP, rate, PFC and link events in the current relay test originate from
  `VocabularyFlowRuntime`, not a physical policy or fabric producer.

## Frozen fix-round regressions

### Evidence ordering

The ten packet relations retain their original formulas and counts, but the
runner must evaluate them directly from the raw ABI-v2 projections before it
invokes either the inherited Tier A exact oracle or the per-packet exact
oracle. Only that pre-oracle evaluation is scored. Exact timestamps, payload
closure, ordering and the missing-TX mutant remain fatal unscored checks.

The original matrix and quantitative bands do not change. The result report
must label this ordering change as a post-specified evidence-accounting
correction and state the corrected scored surface plainly.

### Transactional runtime rollback

Use a bound ABI-v2 runtime that advertises packet and PFC events. On its first
send, emit one uncorrelated PFC submission at time 0 and then throw. Required
post-failure state is exact: no live or issued token, no scheduled event, no
ready notification and no pending physical work. A second identical attempt
must throw the injected runtime error again, not an event-key collision. Its
provisional extent token and event-sequence range are reusable because the
first attempt did not commit.

### Late packet feedback

For one ABI-v2 packet attempt, deliver the packet terminal and then emit a CNP
for the same `(flow, packet index, transmission attempt)` before the logical
extent terminal. The wrapper and SimLLM consumer must accept the CNP with the
original attempt token. A bounded completed-attempt tombstone may live only
while its parent extent is live; the extent terminal purges it. Quiescence
retains no live attempt or tombstone.

### ABI-v1 consumption-time drop choice

Use one unbound ABI-v1 port at 400 Gbit/s with capacity two and `drop_first`
enabled. Submit an 8,192-byte extent first and a 4,096-byte extent second at
time 0. The second extent is due at 81,920 ps and must be the sole injected
drop. The first extent is due at 163,840 ps and must deliver. This restores the
accepted rule that the first due terminal, not the first submitted token, is
selected.

The existing capacity-one v1 raw observations and summary must retain their
exact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `raw_observations.json` | `37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a` |
| `summary.json` | `00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004` |

## Explicit residuals

The registered packet matrix and composed 8,192-byte directed test contain no
partial 4,096-byte final packet. A separate residual must own a physical
4,096-byte-quantum partial-tail cell. The current control vocabulary also has
no physical ECN/CNP, rate, PFC or link-state producer. The registry correction
must assign those producers separately from the already registered dynamic
link-transition source.

ABI negotiation must reject a v2 consumer paired with a v1 producer before
setup, handler installation, submission or authority mutation. There is no
implicit v2-to-v1 down-conversion. A caller that wants compatibility must
explicitly construct the v1 session.

## Registered validation and dry runs

The result-producing packet study remains the command registered in the
original expectations, with a new external output directory for this
fix-round reproduction. Source the local configuration first:

```bash
.venv/bin/python examples/rnic_packet_v2/run_study.py \
  --htsim-source "${SIMLLM_HTSIM_SOURCE:?configure SIMLLM_HTSIM_SOURCE}" \
  --v1-reference-dir \
    "${SIMLLM_WAVE3_RUN_ROOT:?configure SIMLLM_WAVE3_RUN_ROOT}/htsim9/fix-round-f88d9fd" \
  --out \
    "${SIMLLM_WAVE3_RUN_ROOT}/codex/back2526_packet_vocabulary/packet-v2-fix-round1"
```

Before this freeze, that command was run with `--check-only`. It printed its
registry confirmation by design and produced no artifacts. `ctest -N` also
inventoried the existing complete htsim and SimLLM native suites without
executing tests or producing result artifacts. The Python suite was collected
with `.venv/bin/pytest --collect-only -q`; ruff was run in its normal
read-only check mode.
