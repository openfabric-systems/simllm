# RNIC submission-source v1 results

## Chronology and provenance

Expectations-only commit
`928f2a5d995a33f0fb846734d2a7207777f3f20c` precedes the submission-source
implementation, every new native test and every result-producing BACK-20 run.
Its registered `--check-only` command printed its registry confirmation by
design and produced no artifact. The commit message records the precise
freeze-time working tree: the only two untracked files were the expectations
and a dry-run harness containing frozen literals, build orchestration and
validation logic. No implementation edit or result artifact existed. This is
a local pre-run freeze, not a claim of public pre-registration.

Implementation began after the freeze. The first result-producing registered
command completed without a stopped attempt or outcome-dependent edit. The
frozen two-axis grid, row schema, expected ownership, exact-zero band and
artifact digests were unchanged. The external-source audit was completed
before freeze against SimLLM commit
`dba467984b9d82ba374dce5d64d687ca59074135` and official NVIDIA NCCL commit
`5067397c2676d5aed50042fc39e5c8ee96eb0027`. Exact source files and lines are
recorded in [expectations.md](expectations.md).

Integration review after implementation commit `805944f` found that the CSV
emitter projected `RnicProducerShape` into both `producer_shape` and
`producer_kind`. The GPU agent therefore appeared as `gpu_initiated` instead
of taxonomy kind `gpu`. This is a post-specified evidence correction, not a
frozen expectation or a pre-registered result. It changes only the two GPU
cells' `producer_kind` value and the result digest. The row schema, six-cell
grid, identities, translation relation, bands, counts and six predecessor
artifact digests are unchanged. The validator now requires kind `gpu`, and
the corrected study was rerun on the merged integration-review state.

The byte-locked native result is [results.csv](results.csv), SHA-256
`8f74c6fd92d012f2c70c1c2b09d6f49a4d99bcc35fd418a239f7b577777edbc7`.

## Decision-relevant translation asymmetry

All six `(producer shape, batch size)` cells pass. Every active QPC fetch uses
one `QpcIcm` transaction and has exactly zero QPC-attributed MKey, MPT or MTT
events. The study observes fifteen QPC fetches in total.

| Producer shape | Batch | QPC fetches | QpcIcm | QPC MKey/MPT/MTT |
|---|---:|---:|---:|---:|
| `host_cpu_driver` | 1 | 1 | 1 | 0/0/0 |
| `host_cpu_driver` | 4 | 4 | 4 | 0/0/0 |
| `cpu_proxy` | 1 | 1 | 1 | 0/0/0 |
| `cpu_proxy` | 4 | 4 | 4 | 0/0/0 |
| `gpu_initiated` | 1 | 1 | 1 | 0/0/0 |
| `gpu_initiated` | 4 | 4 | 4 | 0/0/0 |

This is 6 of 6 scored cells in the frozen exact-zero band. The positive
control also passes in every cell: each data access carries one MKey, one MPT
and one MTT event. These controls are fatal and unscored because their exact
sequence and counts are author-defined structure.

## Default-shape byte identity

All six accepted artifacts retain their frozen bytes:

| Artifact | SHA-256 | Result |
|---|---|---|
| `examples/rnic_wq_v1/results.csv` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` | PASS |
| `examples/rnic_pcie_v1/results.csv` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` | PASS |
| `examples/rnic_device_v1/results.csv` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` | PASS |
| `examples/rnic_device_v1/native_tests.csv` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` | PASS |
| `examples/rnic_session_records_v1/results.json` | `d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6` | PASS |
| `examples/rnic_hostmem_v1/results.csv` | `1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53` | PASS |

This is 6 of 6 scored byte-identity instances with exactly zero changed
bytes. The default producer is `host_cpu_driver`; zero compatibility
identities resolve to the QP number, retaining all accepted PCIe requester
IDs. DMA-off devices keep the strict v1 effective-hardware bytes. Enabled
host-memory devices use strict v3 records so the resolved producer, requester,
CQ consumer, descriptor queue and placement contribute to the hardware hash.
The native reader retains strict v2 compatibility.

## Fatal unscored evidence

The six rows match the frozen producer-shape matrix and corrected agent-kind
projection exactly. Across fifteen WQEs,
the component emits fifteen submission records and fifteen CQ-consumption
records. Every record joins the existing WQE or CQE identity and timestamp.
Each CQ uses one configured owner. Producer IDs 7101 through 7103, CQ consumer
IDs 8101 through 8103 and RNIC requester ID 9100 remain separate from QP 19.

Host CPU mode locates SQ, CQ, doorbell and data objects in pinned host memory.
CPU proxy mode registers GPU writer 7202's descriptor queue in host-visible
memory, keeps the NIC rings host-pinned and locates data in GPU memory.
GPU-initiated mode locates SQ, CQ, doorbell and data objects in GPU memory and
marks the MMIO UAR mapping as GPU-owned. QPC/ICM stays host-pinned in every
row.

Native directed checks additionally cover endpoint and shape disagreement,
GPU-resident QPC rejection, missing proxy descriptor rejection, active
submission fields with DMA off, invalid data-descriptor atomicity, empty CQ
polls, default host identity resolution, v3 session-record emission and strict
shape-versus-endpoint rejection. The native parser also accepts a canonical
strict v2 record after removing the v3 submission object and recomputing its
hash. Ledger conservation, sequence order, consumer uniqueness, transaction
requester attribution and queue, registry and fabric invariants also pass.
These are structural checks and do not increase either scored denominator.

## Native and study gates

The registered Release build treats warnings as errors. It reports:

```text
1/6 simllm_rnic_pcie_fabric_test passed
2/6 simllm_rnic_work_queue_test passed
3/6 simllm_rnic_host_memory_test passed
4/6 simllm_rnic_submission_test passed
5/6 simllm_rnic_device_test passed
6/6 simllm_rnic_wq_probe_rejects_negative_service passed
100% tests passed, 0 tests failed out of 6
BACK-20 passed 6/6 translation cells and 6/6 artifact identities
```

The read-only Tier A fake producer and checker from the baseline checkout were
also compiled against this tree's final Release library with warnings as
errors. With `SIMLLM_TIER_A_RUN_ROOT` pointing at the external wave-3 root,
the checker passes 8 exact-oracle rows; 4 of 4 D-additivity, 4 of 4 inverse-rate
serialization and 4 of 4 two-WQE FIFO instances; all seven fatal invariant
families; and the wrapper-bypass negative control. The external raw and
summary artifacts retain the baseline hashes
`5fb58e513f6313ebe23fc751ce05bafc07a51ef0a4892d9035c24cdff20fafbb`
and `6825d3ae34f079ec5cc5e3d91faa59948dd31ee724b89c400306a5fac5b869fb`.
This compatibility check remains separate from both BACK-20 scored families.

The portable reproduction command is:

```bash
.venv/bin/python examples/rnic_submission_v1/run_study.py \
  --out "$SIMLLM_WAVE3_RUN_ROOT/codex/rnic1920_hostmem_submission/back20"
```

## Genuine-risk fractions and boundary

Fractions remain separate by scored evidence class:

- QPC translation asymmetry: 6 of 6 cells, or 100 percent, are genuine-risk
  relations. Relaxing ring and data endpoints to GPU memory can naturally
  route QPC through the same generic MKey/MTT helper.
- Default artifact identity: 6 of 6 artifacts, or 100 percent, are
  genuine-risk relations. New config, requester identities, ledgers and
  schema fields can perturb each accepted producer independently.

This is component evidence. It creates no htsim composition,
`CompletionEvent`, `StepResult`, TTFT or TPOT claim. HTSIM-9 is the successor
for native RNIC to packet-simulator composition. BACK-27 is the wave-4
successor that submits GPU producer and CQ-owner work through the concurrent
compute service. BACK-28 owns strict Python ingestion of native effective-
hardware v2 and v3. BACK-11 retains QP lifecycle, pairing and cache residency.
No backend submodule, `README.md` or `docs/README_PRO.md` was changed.
