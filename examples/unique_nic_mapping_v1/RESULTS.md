# Unique-NIC mapping result

## Outcome

What ran: `examples/unique_nic_mapping_v1`, the frozen PLACE-2
qualification, built the mapper's `unique-nic` mode with fabric-backed NIC
selection, the declared shared-NIC fabric builder and the collapsed-pair tag
rule with its projection table from implementation commit `2fb5ff75`, and
drove them on `rnic-nn-fluid` at 400 Gbit/s through the step sink (cells U1
and U2) and through the traffic renderer and backend directly (cell U3). The
expectations-only commit is `d348c326`; the amendment that moved U3 to the
traffic layer, frozen before U3 was implemented, is `2e56b337`.

What came out: the result is `PASS` with no finding. The deciding rows are
the twelve U3 relation instances: with `g` GPUs behind one NIC, every
per-flow completion time and every phase makespan equals
`8 g * S * 20 ps + 2,000,000 ps`, so the `g = 2` and `g = 4` rows are exactly
two and four times the `g = 1` row after the propagation term, for both
payloads. The `g = 1` row is byte identical to the `gpu-rank` run. Both
identity families held, all ten rejection controls refused before any
workdir existed, the seven compatibility digests and the six m5 makespans
held under the default mapping, and all 128 completion rows of the U3 grid
joined their semantic segment exactly once.

What it changes for the project: PLACE-2 closes. The architecture doc's
`unique-nic` sentence becomes literal, and the mapper's NIC selection is read
from the fabric manifest rather than assumed from the rank number, which
makes PLACE-1's "general NIC selection in the mapper" clause literal for the
declared shared-NIC shape. The completion join under the default mapping and
the full-population expert-parallel traffic through the sink are registered
as PLACE-10.

What it does not change: the coarse device runtime keeps its fixed
eight-RNIC profile (CORE-14); the physical Clos projection still requires 64
endpoints; `unique-nic` is refused with the peer packet, flow session,
dependency cross-check and physical topology seams, all registered under
PLACE-10; and no timing is calibrated, since the fluid null network is the
oracle.

## Cells

| Cell | Result |
|---|---|
| U1 identity, one-plus-one deployment (16 GPUs, 16 NICs in GPU order) | `unique-nic` and `gpu-rank` produce byte-identical GOAL text and completion tables, equal `StepResult` rows and outcomes, and equal joined per-segment times on both m5 shapes |
| U2 permutation (two nodes of four, NICs in reverse order) | GOAL rank is `4 * node + (3 - local)`; the GOAL text differs from `gpu-rank` only in rank numbers; results, outcomes and joined times are identical |
| U3 shared endpoints (two nodes of eight, `g` in 1, 2, 4) | 12 of 12 relations exact (below) |
| U4 refusals | 10 of 10 refused before any workdir, GOAL artifact or backend process |
| U5 wire identity | the shared-NIC fabric round-trips with `goal_rank_mapping` in both spellings; 7 of 7 digests and 6 of 6 m5 makespans exact under the default |

U3, per-flow completion time equal to phase makespan:

| Payload S (bytes) | g = 1 (16 GOAL ranks, 8 flows per endpoint) | g = 2 (8 ranks, 16 flows) | g = 4 (4 ranks, 32 flows) |
|---:|---:|---:|---:|
| 65,536 | 12,485,760 ps | 22,971,520 ps | 43,943,040 ps |
| 1,048,576 | 169,772,160 ps | 337,544,320 ps | 673,088,640 ps |

Minus the 2,000,000 ps propagation term the columns are exactly 1 : 2 : 4.
Fabric bytes are identical across `g` (the collapse moves no byte) and NVLink
bytes are identical across `g` and across modes (the locality split is
untouched). The tag multiplier `M` is 1, 4 and 16 for `g` 1, 2 and 4.

## Physical sanity

A flow of `S` bytes cannot beat `S * 20 ps` of serialization plus one
2,000,000 ps propagation at 400 Gbit/s, and `n` equal flows sharing one
endpoint under fair sharing cannot finish before `n * S * 20 ps + 2,000,000 ps`,
the fluid rule the m1 scatter cells validated to 0 ps. The measured values
equal that bound with `n = 8 g`. The smallest cell, `g = 1` at 65,536 bytes,
sits at 12,485,760 ps, which is 8 times 1,310,720 ps of serialization plus
the propagation term, and no cell is below its bound.

## Evidence accounting

The twelve U3 relation instances are the scored behavioral family and the
U1 and U2 identities are the scored exact-oracle families. The projection
joins and U5 are structural exact guards, U4 is a rejection control family,
and the seven digests plus six makespans are fatal by-construction
identities. Counts from these classes are not added. No fatal guard was
violated. The tracked [results](results.json) hold every row; bulk GOAL
programs, completion tables and manifests stay under the data root.

## Deviations from the freeze

The completion join through the projection table is enforced under
`unique-nic` only. Enforcing it under `gpu-rank` broke 28 existing tests
because typed backend doubles and the LogGOPSim sink return completion rows
that are not one per GOAL message. Under `gpu-rank` the renderer still
builds the projection table, which is the identity there, but publishes no
join outcome; the harness reads the `gpu-rank` completion times from the
completion tables. This is post-specified to the implementation and is
registered in PLACE-10 together with the full-population expert-parallel
traffic the amendment set aside.

The default path does run the new projection code: under `gpu-rank` the
sink builds the step projection table (the identity) and hands it to the
renderer, and only the join is skipped. Two consequences are recorded here
rather than hidden. First, a cross-phase duplicate of `(source, destination,
tag)` is now fatal under `gpu-rank`; the per-operation tag blocks make that
unreachable today. Second, byte identity of the default path against the
pre-change tree is proven by the six m5 makespans, by an independent review
probe that found rendered GOAL text and sink GOAL files byte identical to
commit `26704c1c` in 24 configurations (m5 worlds 2, 4 and 8, a 16-rank MoE
and a 16-rank ring on two nodes, and an 8-rank tensor group), and by the
pre-change GOAL digest guard added to the tests after that review.

## Reproduction

```bash
source .env.local.sh
python examples/unique_nic_mapping_v1/run_study.py
python examples/unique_nic_mapping_v1/run_study.py --check
```

`SIMLLM_HTSIM_RNIC` and `SIMLLM_TXT2BIN` select the pinned binaries;
`SIMLLM_DATA_ROOT` or `--output-root` owns the bulk artifacts.
