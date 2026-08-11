# RNIC GPU producer coupling v1 results

## Chronology and provenance

Expectations-only commit
`43d3f39c4be6469c3cf2679d1eca3e89a3207db6` precedes the producer-task
builder, timed task admission, native task linkage, every new test and every
result-producing BACK-27 run. Immediately before that commit, the working tree
contained exactly two untracked files:
`examples/rnic_gpu_producer_v1/expectations.md` and
`examples/rnic_gpu_producer_v1/run_study.py`. The untracked dry-run harness
existed at freeze time and encoded only frozen literals, build orchestration
and validation logic. There were no tracked modifications, implementation
edits or result artifacts.

The registered check-only command printed its registry confirmation by design
and produced no artifacts. Commit
`74f463ae40290cb17fca1bae2b4cdfaac1b248aa` then corrected only one lint error
in dictionary iteration before implementation began. It changed no frozen
grid, source audit, relation, literal, command or artifact digest. This second
commit is the final pre-run expectation state. This is a local pre-run freeze,
not a claim of public pre-registration.

Implementation began after both commits. The first result-producing registered
command completed successfully without a stopped attempt or an
outcome-dependent edit. The external-source audit was completed before freeze
against SimLLM commit
`b74629b4b4da1addda9ff21226cfabf5c09aad87` and official NVIDIA NCCL commit
`5067397c2676d5aed50042fc39e5c8ee96eb0027`. Exact source files and lines are
recorded in [expectations.md](expectations.md).

The byte-locked result is [results.csv](results.csv), SHA-256
`5b397094ec9e942ab01915223dc6e80884ef62e67486ad1823d874ed2f2bf277`.

## Decision-relevant scheduler results

All 18 frozen `(coupling, producer shape, SM occupancy)` configurations pass.
The enabled non-host cells are:

| Producer shape | Occupancy | Task start | Task completion | Submission | Delay from cycle 16 |
|---|---|---:|---:|---:|---:|
| `cpu_proxy` | idle | 0 | 4 | 16 | 0 |
| `cpu_proxy` | half | 0 | 5 | 16 | 0 |
| `cpu_proxy` | saturated | 32 | 36 | 36 | +20 |
| `gpu_initiated` | idle | 0 | 7 | 16 | 0 |
| `gpu_initiated` | half | 0 | 8 | 16 | 0 |
| `gpu_initiated` | saturated | 32 | 39 | 39 | +23 |

The half-occupancy cells satisfy `C_half = P + 1` exactly for both producer
shapes. The producer shares the live issue and HBM service path with the
surrounding HBM-to-NVLink kernel, but caller slack keeps the submission at
cycle 16.

The saturated cells satisfy
`C_saturated = H + P`, with `H = 32`, exactly. Full-SM residency delays task
admission until cycle 32, which shifts CPU-proxy submission by +20 cycles and
GPU-initiated submission by +23 cycles. This is the registered positive-delay
decision relation.

Both idle producer timelines are byte-identical to their disabled
counterparts. Producer service completes inside the frozen caller slack and is
not charged again at the submission boundary. All six enabled and disabled
host-CPU cells construct no GPU task and invoke no scheduler. Every disabled
cell returns its baseline timestamp unchanged.

## Task shape and authority checks

The CPU-proxy task issues one 64-byte HBM descriptor store and one ordered
publication instruction. The GPU-initiated task issues one 64-byte HBM WQE
store, one 4-byte HBM doorbell-record store and one ordered publication
instruction. Their measured task counters are therefore 2 instructions and
64 HBM bytes, or 3 instructions and 68 HBM bytes. NVLink bytes remain zero for
the producer itself. These author-defined trace and conservation checks are
fatal and unscored.

For every enabled non-host cell, the compute estimate is the only mutable
producer-task timing authority. The native RNIC receives the task identity,
producer shape, GPU owner and submitted, eligible, started, finished and
completed timestamps after scheduling. It validates that projection before
any doorbell-side mutation and copies it into the existing submission record.
It never advances the task.

Native directed tests accept the CPU-proxy GPU descriptor writer and the
GPU-initiated producer as owners. They reject a host task link, the wrong GPU
owner, a nonmonotonic or late completion and a task identity reused across
doorbell batches. Rejection preserves queue, record and PCIe-fabric state. The
explicit caller-timestamp non-host bypass remains valid.

## Default and predecessor identity

All eleven frozen artifacts retain their exact SHA-256 digests:

| Artifact | Result |
|---|---|
| `examples/rnic_wq_v1/results.csv` | PASS |
| `examples/rnic_pcie_v1/results.csv` | PASS |
| `examples/rnic_device_v1/results.csv` | PASS |
| `examples/rnic_device_v1/native_tests.csv` | PASS |
| `examples/rnic_session_records_v1/results.json` | PASS |
| `examples/rnic_hostmem_v1/results.csv` | PASS |
| `examples/rnic_submission_v1/results.csv` | PASS |
| `examples/gpu_service_model/results.csv` | PASS |
| `examples/gpu_task_mix/results.csv` | PASS |
| `examples/gpu_task_mix/diagnostics.csv` | PASS |
| `examples/gpu_task_mix/nccl_convergence.csv` | PASS |

The registered native executable also regenerated the accepted BACK-20 CSV
byte for byte. Its SHA-256 remains
`8f74c6fd92d012f2c70c1c2b09d6f49a4d99bcc35fd418a239f7b577777edbc7`.
This live default-output check is separate from the frozen artifact inventory.

## Gates and reproduction

The registered Release build uses warnings as errors and reports:

```text
1/6 simllm_rnic_pcie_fabric_test passed
2/6 simllm_rnic_work_queue_test passed
3/6 simllm_rnic_host_memory_test passed
4/6 simllm_rnic_submission_test passed
5/6 simllm_rnic_device_test passed
6/6 simllm_rnic_wq_probe_rejects_negative_service passed
100% tests passed, 0 tests failed out of 6
```

The full repository gates report:

```text
All checks passed!
655 passed, 4 skipped in 14.76s
```

The portable reproduction command is:

```bash
.venv/bin/python examples/rnic_gpu_producer_v1/run_study.py \
  --out "$SIMLLM_WAVE3_RUN_ROOT/codex/back27_gpu_producer_coupling/back27"
```

Bulk build and run outputs remain under the configured external wave-3 run
root. Only the reviewed result CSV is tracked.

## Genuine-risk fractions and boundary

Fractions remain separate by scored evidence class:

- Half-occupancy issue sharing: 2 of 2 instances, or 100 percent, are
  genuine-risk. An isolated producer estimate could naturally miss the shared
  issue cycle.
- Saturated submission cadence: 2 of 2 instances, or 100 percent, are
  genuine-risk. A caller timestamp could naturally bypass SM admission.
- Idle timeline identity: 2 of 2 instances, or 100 percent, are genuine-risk.
  Producer service could naturally be charged both in compute and again at
  submission.
- Predecessor artifact identity: 11 of 11 instances, or 100 percent, are
  genuine-risk. New task times and native record fields could naturally
  perturb accepted output bytes.

The 18 configurations, exact task rows, structural invariants, four behavioral
families and six native executables are not added into one headline total.

This is component evidence. It stops at a compute-owned task completion joined
to the native RNIC submission record and makes no htsim, `CompletionEvent`,
`StepResult`, TTFT or TPOT claim. HTSIM-9 remains the packet-simulator
composition successor, and CORE-5 remains the final metric-reduction
successor.

COMP-21 (Precision; P1; L) owns hardware calibration of the synthetic producer
trace. BACK-37 (Completeness; P1; L) owns the GPU CQ-consumer and runner
callback half. Those are deliberate omissions. No backend submodule,
`README.md` or `docs/README_PRO.md` was changed.
