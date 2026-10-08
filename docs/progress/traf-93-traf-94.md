# TRAF-93 and TRAF-94 progress

This document is the working ledger for the Simple buffered-read implementation
and the standardized NCCL primitive experiment. It records what is implemented,
what was verified, and what remains open. Hardware results are not claimed until
the corresponding frozen campaign is complete and its guards pass.

## Scope and order

1. **TRAF-93:** implement `buffered_read` for Simple on the retained GPU/NVLink
   calendar while preserving the existing LL, LL128 and source-off paths.
2. **TRAF-94:** implement the manifest-driven primitive runner, capability and
   source-conformance diagnostics, raw result schema, and analysis gates.
3. Run hardware measurements only after the runner inventory and capability
   outcomes have been frozen as required by the existing v1 experiment design.

The normative inputs are:

- `examples/nccl_simple_read_v1/expectations.md` and `expectations.json`;
- `docs/design/nccl-primitive-identification-v1.md`;
- `examples/nccl_primitive_identification_v1/manifest.json`.

## TRAF-93 checklist

- [x] Read the frozen expectations and inspect the retained packet and NCCL
  runtime boundaries.
- [x] Confirm that component packetization already represents a peer read as a
  zero-payload request followed by payload-bearing response packets.
- [x] Admit peer reads on the retained packet calendar without weakening the
  existing peer-write preflight.
- [x] Add a one-shot response gate so owner-memory service becomes eligible
  only after request visibility and responses become eligible only after that
  service completes.
- [x] Extend `NcclRingProgram` and `NcclExecutionConfig` with explicit
  `buffered_read`, applying it only to Simple payload movement.
- [x] Emit read-specific source bindings that preserve requester, buffer owner,
  connection sequence, stripe, request and response identity.
- [x] Preserve empty-slice control work without inventing payload reads.
- [x] Reject placement changes on an existing communicator and unsupported
  registered/direct/proxy branches before admission.
- [x] Add exact isolated-delay, memory-rate, FIFO-reuse, source-bypass and
  collective/graph tests from the frozen expectations.
- [x] Add and run the `examples/nccl_simple_read_v1` mechanism study.

## TRAF-94 checklist

- [ ] Define the expanded-cell representation and deterministic manifest digest.
- [ ] Implement sequential stage expansion rather than a blind Cartesian
  product.
- [ ] Implement required result-row validation and explicit qualification
  reasons.
- [ ] Implement already-ready, delayed-publication, delayed-consumption,
  empty/nonempty, copy/reduction and working-set primitive probes.
- [ ] Add channel/warp/SM resource-sharing probes after the one-channel pilot.
- [ ] Keep diagnostic source-operation counts separate from ordinary timing.
- [ ] Capture requested and realized controls, source/build/device identity,
  local timer boundaries, correctness and process exits.
- [ ] Implement five-process completeness, paired-IQR resolution and held-out
  prediction checks.
- [ ] Freeze the capability-qualified expanded H100 inventory before ordinary
  timing. H100 is an extension; it is not silently labeled A100 or GH200.
- [ ] Run and publish the hardware campaign.

## Verification log

| Date | Branch | Evidence | Result |
|---|---|---|---|
| 2026-09-11 | `traf-93-simple-read` | Repository and frozen-design inspection | Packetizer read support exists; retained read admission and source-memory response gating are missing. |
| 2026-09-11 | `traf-93-simple-read` | Focused retained packet/NCCL tests | 221 tests passed while the implementation was being integrated. |
| 2026-09-11 | `traf-93-simple-read` | Expanded packet/NCCL regression set | 368 passed; one assertion expected the old validation wording and was corrected without changing behavior. |
| 2026-09-11 | `traf-93-simple-read` | Final combined packet/NCCL regression (`PYTHONPATH=. .venv/bin/pytest -q ...`) | 370 passed in 14.89 seconds. |
| 2026-09-11 | `traf-93-simple-read` | `nccl_simple_read_v1/run_study.py` | All fatal guards valid: 9 isolated-read, 6 remote-memory, 288 collective and 32 graph-metric rows. |
| 2026-09-11 | `traf-93-simple-read` | Ruff over every changed Python file; `git diff --check` | Passed. |

## Commands

Commands and exact results will be added here as implementation gates run. Raw
hardware captures must remain outside Git; only compact summaries and their
content-hash manifest belong in the repository.

## Handoff notes for future engineers and agents

- The retained packet engine is the single clock owner. Request-visibility
  callbacks may enqueue GPU work on that calendar but must never advance it
  recursively.
- A read response is intentionally ineligible until
  `release_read_response(extent_id)` is called. The call is one-shot and is
  legal only after the request is consumer-visible. Do not replace this gate
  with a duration added to packet transport: doing so loses owner-memory
  contention and causal attribution.
- For a Simple read, the resident block belongs to the requester while the
  memory cursor belongs to the source/buffer owner. `memory_rank` represents
  precisely this split; it does not imply a resident block on the owner.
- `buffered_read` is communicator placement, not a fourth protocol. LL and
  LL128 still execute writes on that communicator; only Simple payload motion
  changes direction.
- The critical-path packet-only reporter cannot describe an external GPU
  service gate and therefore rejects this path before admission. Extend its
  evidence model before relaxing that guard.
- TRAF-94 must not turn an unqualified H100 run into a result merely because
  this checkout happens to be on H100 hardware. Freeze the realized source,
  build, device, and expanded-cell inventory first.
