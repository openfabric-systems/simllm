# TRAF-65 local arm and Merlin remainder

## Current state

TRAF-65 is OPEN. The GPU-free local arm is
`VALID_LOCAL_MOCK_86_OF_86`: 80 isolated cells, five ordered
`corner_frame` cells, and one `all_corners_frame` cell produced 14,035
digest-complete rows. The tracked summary is
[local-validation.json](local-validation.json). It has
`measurement_claim: false`; it validates compilation, catalog expansion,
resumption, checksums, manifests, and the candidate service only.

The hardware remainder ran on 2026-08-27 as Merlin job `198968`. All 86 cells
are digest-complete on `gpu105`, with 14,035 hardware rows and an empty pending
set. Every scheduler task exited `0:0` and no scheduler stop record exists.

This does not promote the result to measurement evidence. The frozen capture
cannot evaluate five fatal guards, derives packet and raw-byte fields from the
candidate, leaves seven named sweep controls unapplied in hardware mode, and
enqueues one copy-engine operation per message. The scored state is
`COMPLETE_VOID_86_OF_86`; see [the hardware score](RESULTS.md). TRAF-65 remains
OPEN, and the profile remains a declared candidate with no parameter value
changed.

The written maintenance record said reservation `SD26082026` held Merlin GPU
nodes down until `2026-08-28T06:30`. On 2026-08-27 the integrator verified that
the reservation lifted early and the A100 partitions were visible in mixed and
allocated states. That verified state superseded the submission date only.
The `a100-hourly`, exclusive four-GPU, `%1`, six-minute, and task-indexed pacing
rules remained in force.

The final expectations digest is:

```text
212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571
```

The expectations-only commit is `d74b123`. The first local mock lineage
found and corrected a producer-index mapping defect before hardware
submission. Obsolete mock attempts remain under their obsolete digest and are
not publication inputs.

## Candidate profile and envelope

Every candidate value below is a declared model choice, not a TRAF-65 hardware
measurement. The only measured values are the previously published
`a100_hardware_envelope_v1` rows.

| Module | Parameter | Candidate | Published envelope or identification |
|---|---|---:|---|
| TX | packet payload | 256 bytes | NVPKT 001-016; unscored candidate |
| TX | packet header | 16 bytes | NVPKT 001-016; unscored candidate |
| TX | links per ordered pair | 4 | NVBOND 017-032; 25 GB/s hard ceiling per link |
| TX | per-link raw rate | 25 GB/s | 100 GB/s hard ceiling per ordered pair |
| TX | endpoint raw egress | 300 GB/s | measured copy-engine fan-out is 281.65 GB/s |
| TX | bond policy | earliest-available packet striping | NVBOND 017-032 |
| TX | direction | writes carry request data; reads carry a request control packet and response data | NVPKT 013-015, NVCRD and NVHOL direction cases |
| TX | effective destination credits | 256 units of 272 bytes | NVCRD 049-064; never a literal hardware-register claim |
| Switch | mode | pass-through | mandatory on the four-A100 direct mesh |
| Switch | bytes and time | exactly zero | byte-identical direct TX-to-RX conformance test |
| Switch | FIFO, arbitration and HOL | absent in pass-through | unidentifiable on A100; declared only for later switched profiles |
| RX | ingress rate | 300 GB/s | independent parameter; NVBOND 025 and NVINC 033-048 |
| RX | buffer capacity | 1 MiB | independent effective candidate; NVINC and NVCRD |
| RX | credit return latency | 200,000 ps | independent effective candidate; NVCRD 049-064 |
| RX | reassembly and delivery | extent sequence, per-extent order | packet, incast, and HOL conservation cases |

The candidate formula predicts 94.117647 GB/s for one full-packet ordered pair
and 282.352941 GB/s for three-way fan-out. Timing the actual TX-to-switch-to-RX
composition over 524,288 bytes per destination gives 94.056390 GB/s and
281.699182 GB/s, respectively. Those values are compared only against the
already measured 94.00 to 94.07 GB/s pair row and 281.65 GB/s fan-out row. The
comparison is post-specified validation, not a new measurement.

The hardware score did not identify the 256-byte payload, 16-byte header,
four-link striping, direction mapping, effective credits, RX buffer, credit
return latency, or reassembly policy. The direct-mesh switch pass-through
invariant stands structurally but is not a hardware measurement. The captured
elapsed-time bands refuted three bond cases and all 16 incast cases, but those
rows are void under the frozen guard contract and therefore refute this capture
as an identification procedure rather than the physical candidate constants.

## Exact local entry point

From the worktree root:

```bash
TRAF65_WORKTREE="$(git rev-parse --show-toplevel)"
TRAF65_DEVELOPMENT_ROOT="$(CDPATH= cd -- "${TRAF65_WORKTREE}/../.." && pwd)"
TRAF65_LOCAL_ROOT="${TRAF65_DEVELOPMENT_ROOT}/wave-runs/traf65"
examples/a100_nvlink_packet_v1/run_local.sh
.venv/bin/python examples/a100_nvlink_packet_v1/summarize_local.py \
  --root "${TRAF65_LOCAL_ROOT}" \
  --output examples/a100_nvlink_packet_v1/local-validation.json
```

Each result is under `TRAF65_LOCAL_ROOT`, followed by the freeze digest,
`cells`, the cell name, and an `attempt-NNNN` directory.
`COMPLETE.json` binds `manifest.json`, which binds every payload. A current,
complete attempt is skipped. An incomplete or obsolete attempt is retained and
a new attempt directory is created.

## Exact Merlin staging and resume entry point

These are the commands used from this worktree after the integrator's verified
early release of the reservation. They stage the exact local commit without
requiring a remote Git branch. The archive first lands as a `.part` file, so
SSH loss cannot create a stage that the submission command accepts.

```bash
TRAF65_HEAD="$(git rev-parse HEAD)"
TRAF65_FREEZE="212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571"
TRAF65_WORKTREE="$(git rev-parse --show-toplevel)"
TRAF65_DEVELOPMENT_ROOT="$(CDPATH= cd -- "${TRAF65_WORKTREE}/../.." && pwd)"
TRAF65_LOCAL_ROOT="${TRAF65_DEVELOPMENT_ROOT}/wave-runs/traf65"
TRAF65_ARCHIVE_ROOT="${TRAF65_LOCAL_ROOT}/stage"
TRAF65_ARCHIVE="${TRAF65_ARCHIVE_ROOT}/simllm-${TRAF65_HEAD}.tar"
TRAF65_REMOTE_ROOT="simllm-data/traf65"
TRAF65_REMOTE_NCCL="simllm-stage/a100_hw_envelope_v1/nccl/nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64/nvidia/nccl"
mkdir -p "${TRAF65_ARCHIVE_ROOT}"
git archive --format=tar --output="${TRAF65_ARCHIVE}" "${TRAF65_HEAD}"
TRAF65_ARCHIVE_SHA="$(sha256sum "${TRAF65_ARCHIVE}" | awk '{print $1}')"

ssh merlin "mkdir -p ${TRAF65_REMOTE_ROOT}/incoming"
scp "${TRAF65_ARCHIVE}" \
  "merlin:${TRAF65_REMOTE_ROOT}/incoming/simllm-${TRAF65_HEAD}.tar.part"
ssh merlin "set -eu; \
  test \"\$(sha256sum ${TRAF65_REMOTE_ROOT}/incoming/simllm-${TRAF65_HEAD}.tar.part | awk '{print \$1}')\" = \"${TRAF65_ARCHIVE_SHA}\"; \
  if [ ! -f ${TRAF65_REMOTE_ROOT}/incoming/simllm-${TRAF65_HEAD}.tar ]; then \
    mv ${TRAF65_REMOTE_ROOT}/incoming/simllm-${TRAF65_HEAD}.tar.part \
       ${TRAF65_REMOTE_ROOT}/incoming/simllm-${TRAF65_HEAD}.tar; \
  fi; \
  test \"\$(sha256sum ${TRAF65_REMOTE_ROOT}/incoming/simllm-${TRAF65_HEAD}.tar | awk '{print \$1}')\" = \"${TRAF65_ARCHIVE_SHA}\"; \
  mkdir -p ${TRAF65_REMOTE_ROOT}/stage/${TRAF65_HEAD}; \
  tar -xf ${TRAF65_REMOTE_ROOT}/incoming/simllm-${TRAF65_HEAD}.tar \
    -C ${TRAF65_REMOTE_ROOT}/stage/${TRAF65_HEAD}; \
  test \"\$(sha256sum ${TRAF65_REMOTE_ROOT}/stage/${TRAF65_HEAD}/examples/a100_nvlink_packet_v1/expectations.json | awk '{print \$1}')\" = \"${TRAF65_FREEZE}\""
```

The staged NCCL 2.31.2 x86 root already present on Merlin is the
`simllm-stage/a100_hw_envelope_v1/nccl/nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64/nvidia/nccl`
path relative to the login home.

Its `libnccl.so.2` SHA-256 is
`dba12e429fe11268b895d0531ba96a7f679f35227d5b1ec77c5febbcd02281bd`.
The hardware source compiled against CUDA 12.9.1 and that NCCL installation on
the Merlin login node. The resulting compile-check binary SHA-256 is
`96b4c544de54457d1fbed8e56b0a1cbe61344bcdab02d6445c07a0ab637277a4`.
This is build evidence only; it allocated no GPU and is not a measurement.

Compute the still-pending array indices and submit only those cells:

```bash
TRAF65_PENDING="$(ssh merlin "TRAF65_REMOTE_ABS=\$(cd ${TRAF65_REMOTE_ROOT} && pwd); python3 \
  \${TRAF65_REMOTE_ABS}/stage/${TRAF65_HEAD}/examples/a100_nvlink_packet_v1/run_study.py \
  --pending-indices \
  --output-root \${TRAF65_REMOTE_ABS} \
  --expected-head ${TRAF65_HEAD} \
  --freeze-sha256 ${TRAF65_FREEZE}")"

if [ -n "${TRAF65_PENDING}" ]; then
  ssh merlin "set -eu; \
    TRAF65_REMOTE_ABS=\$(cd ${TRAF65_REMOTE_ROOT} && pwd); \
    TRAF65_NCCL_ABS=\$(cd ${TRAF65_REMOTE_NCCL} && pwd); \
    sbatch -M gmerlin7 \
    --array=${TRAF65_PENDING}%1 \
    --export=ALL,TRAF65_STAGE_ROOT=\${TRAF65_REMOTE_ABS}/stage/${TRAF65_HEAD},TRAF65_DATA_ROOT=\${TRAF65_REMOTE_ABS},TRAF65_EXPECTED_HEAD=${TRAF65_HEAD},TRAF65_NCCL_ROOT=\${TRAF65_NCCL_ABS},TRAF65_CUDA_MODULE=cuda/12.9.1 \
    \${TRAF65_REMOTE_ABS}/stage/${TRAF65_HEAD}/examples/a100_nvlink_packet_v1/run_merlin_cell.sbatch"
fi
```

The array limit `%1`, six-minute cell wall, and task-indexed pause implement
the short paced occupancy rule. A submitted Slurm job is scheduler-owned, so
loss of the submitting SSH connection does not kill it. The batch script also
traps HUP, INT, and TERM, terminates its active child, writes a scheduler stop
record, and leaves the cell resumable.

Re-running the pending-index command and the same `sbatch` command is the
complete resume procedure. It never treats a directory as complete unless all
payload digests and the manifest digest verify.

The post-run pending-index command returns an empty string. Merlin's login-node
default `python3` is Python 3.6.15, which cannot parse the frozen runner. The
execution prepended the installed Python 3.10.16 binary and runtime-library
paths to the environment before invoking the exact pending and `sbatch`
commands; the staged source, archive, freeze, and batch script were unchanged.
The batch-built binary SHA-256 was
`992eaa12d5953806a1f21d12fce612d72f721a141d425a666404ffb26770c3e1`,
which differs from the earlier compile-check digest. Every cell plan binds the
single batch digest, and the score publishes the reproducibility mismatch.
