# RNIC host-memory v1 results

## Chronology and provenance

Expectations-only commit
`64335b0a48d11f160e2c576dcc41bd787643eda1` precedes the host-memory
implementation and every result-producing run. Its registered `--check-only`
command printed its registry confirmation by design and produced no artifact.
The commit message records the precise freeze-time working tree: the only two
untracked files were the expectations and a dry-run harness containing frozen
literals and check logic, with no implementation or other change present.
This is a local pre-run freeze, not a claim of public pre-registration.

The first result-producing attempt built the library, passed all five CTest
entries and emitted the four native rows, but then stopped while the Python
wrapper tried to merge the unchecked row list as a mapping. It produced the
external raw CSV and the local result CSV before stopping. Correction commit
`8714ff84ba1bca0d2e19f5695d301fcb0bab8500` routes those rows through the
already-frozen validator before assembling the summary. No scored result is
claimed from the stopped attempt. The registered command was rerun after that
machinery-only correction and again after the final session-record integration.
Neither the sweep nor any expected count, direction or band changed.

The external-source audit was completed before the freeze against SimLLM base
`fc282efc91573638de7dcfae2befee1cf022011b`, rdma-core commit
`c1c5bf1f480312c07ed4d23f0feecf8b5fd73289` and Linux commit
`db2ddb87143519e20a95aa36c60b36107b736a58`. Exact source files and lines are
recorded in [expectations.md](expectations.md).

The byte-locked native result is [results.csv](results.csv), SHA-256
`1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53`.

## Decision-relevant translation asymmetry

All four `(page size, batch size)` cells pass. Every active QPC fetch uses one
`QpcIcm` transaction and has exactly zero QPC-attributed MKey, MPT or MTT
events. The study observes ten QPC fetches in total.

| Page size (bytes) | Batch | QPC fetches | QpcIcm | QPC MKey/MPT/MTT |
|---:|---:|---:|---:|---:|
| 4096 | 1 | 1 | 1 | 0/0/0 |
| 4096 | 4 | 4 | 4 | 0/0/0 |
| 2097152 | 1 | 1 | 1 | 0/0/0 |
| 2097152 | 4 | 4 | 4 | 0/0/0 |

This is 4 of 4 scored cells in the frozen exact-zero band. The positive
controls also pass in every cell: each WQE has one SQ page-list event, one data
MKey/MPT/MTT chain and one CQ page-list event. These controls are fatal and
unscored because their exact sequence and counts are author-defined structure.

## Default byte identity

All five accepted artifacts retain their frozen bytes:

| Artifact | SHA-256 | Result |
|---|---|---|
| `examples/rnic_wq_v1/results.csv` | `598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682` | PASS |
| `examples/rnic_pcie_v1/results.csv` | `464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2` | PASS |
| `examples/rnic_device_v1/results.csv` | `7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934` | PASS |
| `examples/rnic_device_v1/native_tests.csv` | `969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d` | PASS |
| `examples/rnic_session_records_v1/results.json` | `d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6` | PASS |

This is 5 of 5 scored byte-identity instances with exactly zero changed bytes.
Default construction creates no registry, lifecycle event or memory-access
record and makes the new stages not applicable. An enabled host-memory device
uses the strict `simllm-rnic-effective-hardware-v2` projection, so allocation,
page and translation configuration contributes to its hardware hash. Disabled
devices retain the accepted v1 projection exactly.

## Fatal unscored evidence

The four rows match every frozen constructed count: `4 * B` `MttMpt`
transactions, `B` each of `WqeRead`, `PayloadRead` and `CqeWrite`, one
doorbell-record store, one UAR write, six registrations and six teardowns. The
data access selects page index one for both page geometries.

Native directed checks additionally cover transactional multi-object and
multi-batch registration, stale-plan rejection, duplicate and overlap
rejection, foreign-owner teardown, teardown evidence capacity, shared-registry
constructor rollback, data-descriptor rejection without queue, fabric, clock
or registry mutation, explicit use-after-teardown rejection, typed ownership,
path and extent validation, lifecycle conservation and effective-hardware hash
sensitivity. These are structural invariants and do not increase either
scored denominator.

## Native and study gates

The registered Release build treats warnings as errors. It reports:

```text
1/5 simllm_rnic_pcie_fabric_test passed
2/5 simllm_rnic_work_queue_test passed
3/5 simllm_rnic_host_memory_test passed
4/5 simllm_rnic_device_test passed
5/5 simllm_rnic_wq_probe_rejects_negative_service passed
100% tests passed, 0 tests failed out of 5
BACK-19 passed 4/4 translation cells and 5/5 artifact identities
```

The portable reproduction command is:

```bash
.venv/bin/python examples/rnic_hostmem_v1/run_study.py \
  --out "$SIMLLM_WAVE3_RUN_ROOT/codex/rnic1920_hostmem_submission/back19"
```

## Genuine-risk fractions and boundary

Fractions remain separate by scored evidence class:

- QPC translation asymmetry: 4 of 4 cells, or 100 percent, are genuine-risk
  relations. A conventional generic device-memory helper can naturally route
  QPC reads through the same MKey/MTT chain as payload and ring accesses.
- Default artifact identity: 5 of 5 artifacts, or 100 percent, are
  genuine-risk relations. Adding config fields, transactions or record fields
  can perturb each accepted producer independently.

This is component evidence. It creates no htsim composition,
`CompletionEvent`, `StepResult`, TTFT or TPOT claim. HTSIM-9 is the successor
for native RNIC to packet-simulator composition. The wave-4 compute coupling
is the successor for GPU-submitted producer work. BACK-11 retains QP lifecycle,
pairing and cache residency. No backend submodule, live acceptance harness,
`README.md` or `docs/README_PRO.md` was changed.
