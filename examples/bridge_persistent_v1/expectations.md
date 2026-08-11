# BRIDGE-1 prepared co-simulator expectations

This file is frozen before the BRIDGE-1 implementation and before any run of
the registered study. It defines an opt-in prepared-replay mode for the pinned
`htsim_rnic` binary. The accepted `HtsimStepSink` diagnostic mode remains the
default and retains one isolated simulator invocation per step.

## External-source audit and design decision

The audit used SimLLM base commit `b74629b` and pinned HTSIM commit
`edb28c3015c173b4251abc5858c587df325e1ebc` before this freeze.

- `simllm/backends/step_sink.py:245-265` renders one step, invokes `txt2bin`,
  then synchronously invokes `htsim_rnic` before returning its `StepResult`.
- `simllm/backends/htsim_rnic.py:60-73` constructs a command with exactly one
  `-goal` input, while `simllm/backends/htsim_rnic.py:165-195` runs one child
  process and parses one completion CSV.
- `third_party/htsim/htsim/sim/datacenter/rnic_atlahs_cli.cpp:441-452` requires
  one `-goal` option, and its usage at lines 478-503 exposes neither a batch
  manifest nor a stream/session option.
- `third_party/htsim/htsim/sim/datacenter/main_rnic.cpp:152-217` constructs one
  event list and API, calls `start_lgs` once, writes at most one completion
  CSV, validates quiescence, and exits.
- `third_party/htsim/htsim/sim/logsim-interface.cpp:370-431` constructs one
  parser from that filename and configures one runtime from its binary header.
- Existing versioned objects needed by a later true session already have
  strict projections: `ExecutionGraph` at
  `simllm/core/execution_io.py:612-719`, `CompletionEvent` at
  `simllm/core/execution_io.py:746-815`, `ExecutionResult` beginning at
  `simllm/core/execution_io.py:844`, and the complete bookkeeping ledger at
  `simllm/core/bookkeeping_io.py:254-288`.

The pinned binary therefore cannot retain one mutable simulator authority
across steps without a backend interface change. BRIDGE-1 will use a persistent
SimLLM worker pool for finite, fully known replays. Each worker executes the
unchanged diagnostic step path in an isolated child process; independent steps
are prepared concurrently, retained in input order, and served later through
the existing callable `StepRecord -> StepResult | None` contract. This is an
exact acceleration of the accepted per-step-reset model, not a claim of
cross-step network state.

A pre-freeze diagnostic-only timing calibration of the exact pinned binary
informed the broad bands below. Its measured values are intentionally excluded
from this expectations-only commit and will be chronology-labeled in the
result report. The calibration is not study evidence.

## Fixed inputs and timed boundary

The study uses the two recorded M4 adapter captures without editing their
author-defined schedules:

| replay | records | SHA-256 |
|---|---:|---|
| `examples/m4/fixtures/vllm-m2-steps.jsonl` | 8 | `a226fcc17908844ba080587fe6607c5c8f34b178d17111fbd384819731b26fb7` |
| `examples/m4/fixtures/sglang-m3-steps.jsonl` | 9 | `656772148cd8fbda71a25af08215d806f38f3886abb068f72c9e0ddc8cb7c26f` |

Every cell uses the M4 llama-shaped dimensions, TP 8, one 400 Gbit/s link per
rank, and `rnic-nn-fluid`. The varied parameters are replay identity and
prepared worker count in `{4, 8}`. The diagnostic comparison has no worker
count because it is the existing serial default.

Elapsed wall time starts immediately before sink construction and ends after
all records have returned and, for prepared mode, the persistent worker pool
has shut down. It includes GOAL rendering, text-to-binary conversion, child
process invocation, simulation, CSV parsing, prepared-result delivery, and
worker startup and shutdown. It excludes fixture loading, fixture hashing,
artifact comparison, and summary serialization.

One invocation of the registered command runs each cell once. Wall time is not
an exact deterministic oracle; the two-sided bands and within-invocation
ratios are the registered acceptance surface.

## F1: simulated-result identity, fatal and unscored

For every replay and worker count, prepared mode must match the diagnostic
mode step by step and in order. The following artifacts must be byte-identical
for every step:

- canonical `StepResult` bytes, including `step_index`, `step_latency_ps`,
  `completed_at_ps`, request metrics, and additive visit totals;
- rendered GOAL text and compiled GOAL binary;
- the complete backend completion CSV.

The complete ordered latency-byte stream must also be identical. Every run
must report physical quiescence and must return one result for every recorded
step. A mismatch, missing row, reordering, worker exception, or partial
preparation invalidates the study. These are conservation and equivalence
invariants, so they are fatal but do not increase the scored denominator.

The diagnostic class must remain the default exported sink. Prepared mode must
reject an unprepared call, an out-of-order or changed record, duplicate step
indices within one batch, and preparation while earlier prepared results
remain. A failed preparation must publish no prepared result. These API guards
are structural and unscored.

## R1: live wall-clock relation

Let `D_f` be diagnostic elapsed seconds for replay `f`, and let `P_f,w` be
prepared elapsed seconds with `w` workers. A cell passes only if its diagnostic
band, prepared band, and speedup relation all pass.

| replay | workers | diagnostic band, s | prepared band, s | required speedup |
|---|---:|---:|---:|---:|
| vLLM | 4 | `[45, 85]` | `[10, 45]` | `D_vllm / P_vllm,4 >= 1.5` |
| vLLM | 8 | `[45, 85]` | `[6, 35]` | `D_vllm / P_vllm,8 >= 2.0` |
| SGLang | 4 | `[50, 100]` | `[12, 50]` | `D_sglang / P_sglang,4 >= 1.5` |
| SGLang | 8 | `[50, 100]` | `[7, 38]` | `D_sglang / P_sglang,8 >= 2.0` |

The signed expectation is lower wall time in prepared mode. No monotonic
four-to-eight-worker relation is required because process scheduling and
memory contention can make those two concurrent cells noisy. Both must still
beat their own serial replay by the registered factor.

R1 is one behavioral relation family with four live-runtime parameterized
instances. All four are genuine-risk instances: each executes the pinned
simulator, and pool startup, scheduler contention, subprocess failure, or
serialization can independently violate its band or ratio. The planned
genuine-risk fraction is therefore `4/4`. F1 and the API guards remain separate
fatal-unscored evidence classes.

## Registered command and dry-run rule

Bulk artifacts must remain outside Git. Run with configured project-local
binary variables and an external data root:

```bash
SIMLLM_HTSIM_RNIC="${SIMLLM_HTSIM_RNIC:?configure the pinned binary}" \
SIMLLM_TXT2BIN="${SIMLLM_TXT2BIN:?configure the matching converter}" \
.venv/bin/python examples/bridge_persistent_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/bridge_persistent_v1" \
  --fixtures vllm,sglang --workers 4,8
```

Before this freeze, the same command must be executed with `--check-only`.
Check-only validates argument shapes, binary executability, fixture hashes,
and the complete registered matrix; it prints the plan by design and produces
no artifacts.

