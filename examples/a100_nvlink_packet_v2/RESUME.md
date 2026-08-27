# TRAF-70 corrected A100 NVLink capture resume record

## Final outcome

The corrected capture is `COMPLETE_VALID_86_OF_86`. Merlin job `199957`
completed the all-corners frame at array index 85, and job `199960` completed
array indices 0 through 84. The final `sacct` audit reports `COMPLETED` with
exit code `0:0` for all 85 elements of job `199960` and for the accepted
`199957_85` task. The pending set is empty, no accepted attempt has a stop
record, and all 86 manifests bind one producer binary SHA-256:

```text
9c39603234969c5a7b01dd80008788a8649367f8ee003942e6462b344fa6c9d8
```

The lean local pull contains the accepted attempt payloads and the scheduler
logs for jobs `199957` and `199960`, with 1,118 files total and no CUDA build
tree. All 11,542 hardware rows verify against their manifests. The score is
[hardware-score.json](hardware-score.json), its readable report is
[RESULTS.md](RESULTS.md), and the score SHA-256 is:

```text
d08ee9032bc87fd9fbef99ec6027a8250f7c20cfc3d365f39c55ffd9b54a9b3f
```

All ten fatal guards are decidable passes. The explicit throttle verdict is
`CLEAR` on every one of the 11,542 result rows; `FG07` has zero failures and
zero missing observations. The flow-dynamics gate is `OPEN` because every
frozen prerequisite has a decidable non-void outcome.

## Frozen state

The expectations-only commit is `fd7bc6b`. The expectations SHA-256 is:

```text
f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6
```

The freeze contains 80 isolated cases, five ordered corner frames and one
all-corners frame. It defines 11 observation-to-parameter rules and 10 fatal
guards whose missing observables are fatal rather than undecidable.

The GPU-free arm is `VALID_LOCAL_MOCK_86_OF_86`: all 86 cells and 11,542 rows
are digest-complete. The mock validates expansion, the corrected schema,
batched copy-engine accounting, actual checksum fields, ordered-frame
preservation, manifests and resumption. It has `measurement_claim: false` and
does not score a hardware parameter.

TRAF-65 expectations remain SHA-256
`212a7a26f54e444c9b18f1e528bd0d00b5a28e4f9e005b0dc137f477ad642571`.
The protected pre-score A100 candidate is preserved in
[candidate-profile-pre-traf70.json](../a100_nvlink_packet_v1/candidate-profile-pre-traf70.json)
with SHA-256
`899712c4734f7a6b410d80231291663a404511528d46aab7497b73831e0e354f`.
The live profile changed only after score commit `7f729d7`, through
`publish_score.py`. Its scored SHA-256 is
`d33ef5b2c6fa87cc97e1e7b45a43a841a5da45f5462311e3981fbc903c56deb2`.

Merlin was verified available on 2026-08-27. The `a100-hourly` partition had
mixed and allocated qualified nodes, and the staged NCCL 2.31.2 tree was
present. Submission still requires one exclusive four-GPU allocation, the
qualified `NV4` topology, short twelve-minute cells and `%1` array pacing.

## Local entry point

From the worktree root:

```bash
TRAF70_WORKTREE="$(git rev-parse --show-toplevel)"
TRAF70_DEVELOPMENT_ROOT="$(CDPATH= cd -- "${TRAF70_WORKTREE}/../.." && pwd)"
TRAF70_LOCAL_ROOT="${TRAF70_DEVELOPMENT_ROOT}/wave-runs/traf70"
TRAF70_LOCAL_ROOT="${TRAF70_LOCAL_ROOT}" \
  examples/a100_nvlink_packet_v2/run_local.sh
.venv/bin/python examples/a100_nvlink_packet_v2/summarize_local.py \
  --root "${TRAF70_LOCAL_ROOT}" \
  --output examples/a100_nvlink_packet_v2/local-validation.json
```

A current complete attempt is skipped only when `COMPLETE.json` binds the
manifest digest and the manifest binds every payload size and SHA-256.
Incomplete and obsolete attempts remain in place and a new numbered attempt
is created.

## Atomic Merlin staging

Stage the committed implementation without a remote Git mutation:

```bash
TRAF70_HEAD="$(git rev-parse HEAD)"
TRAF70_FREEZE="f0ab026e054873a56614af63ab3a7ae3219dc0b045423808cb41522910fa6da6"
TRAF70_WORKTREE="$(git rev-parse --show-toplevel)"
TRAF70_DEVELOPMENT_ROOT="$(CDPATH= cd -- "${TRAF70_WORKTREE}/../.." && pwd)"
TRAF70_LOCAL_ROOT="${TRAF70_DEVELOPMENT_ROOT}/wave-runs/traf70"
TRAF70_ARCHIVE_ROOT="${TRAF70_LOCAL_ROOT}/stage"
TRAF70_ARCHIVE="${TRAF70_ARCHIVE_ROOT}/simllm-${TRAF70_HEAD}.tar"
TRAF70_REMOTE_ROOT="simllm-data/traf70"
TRAF70_REMOTE_NCCL="simllm-stage/a100_hw_envelope_v1/nccl/nvidia_nccl_cu12-2.31.2-py3-none-manylinux_2_18_x86_64/nvidia/nccl"
mkdir -p "${TRAF70_ARCHIVE_ROOT}"
git archive --format=tar --output="${TRAF70_ARCHIVE}" "${TRAF70_HEAD}"
TRAF70_ARCHIVE_SHA="$(sha256sum "${TRAF70_ARCHIVE}" | awk '{print $1}')"

ssh merlin "mkdir -p ${TRAF70_REMOTE_ROOT}/incoming"
scp "${TRAF70_ARCHIVE}" \
  "merlin:${TRAF70_REMOTE_ROOT}/incoming/simllm-${TRAF70_HEAD}.tar.part"
ssh merlin "set -eu; \
  test \"\$(sha256sum ${TRAF70_REMOTE_ROOT}/incoming/simllm-${TRAF70_HEAD}.tar.part | awk '{print \$1}')\" = \"${TRAF70_ARCHIVE_SHA}\"; \
  if [ ! -f ${TRAF70_REMOTE_ROOT}/incoming/simllm-${TRAF70_HEAD}.tar ]; then \
    mv ${TRAF70_REMOTE_ROOT}/incoming/simllm-${TRAF70_HEAD}.tar.part \
       ${TRAF70_REMOTE_ROOT}/incoming/simllm-${TRAF70_HEAD}.tar; \
  fi; \
  test \"\$(sha256sum ${TRAF70_REMOTE_ROOT}/incoming/simllm-${TRAF70_HEAD}.tar | awk '{print \$1}')\" = \"${TRAF70_ARCHIVE_SHA}\"; \
  mkdir -p ${TRAF70_REMOTE_ROOT}/stage/${TRAF70_HEAD}; \
  tar -xf ${TRAF70_REMOTE_ROOT}/incoming/simllm-${TRAF70_HEAD}.tar \
    -C ${TRAF70_REMOTE_ROOT}/stage/${TRAF70_HEAD}; \
  test \"\$(sha256sum ${TRAF70_REMOTE_ROOT}/stage/${TRAF70_HEAD}/examples/a100_nvlink_packet_v2/expectations.json | awk '{print \$1}')\" = \"${TRAF70_FREEZE}\""
```

The archive first lands with a `.part` suffix and becomes admissible only
after its digest verifies. An SSH loss cannot create an accepted partial
stage.

## Pending set and paced submission

Compute and submit only the exact pending indices:

```bash
TRAF70_PENDING="$(ssh merlin "set -eu; \
  module purge >/dev/null 2>&1; \
  module load Python/3.10.16; \
  TRAF70_REMOTE_ABS=\$(cd ${TRAF70_REMOTE_ROOT} && pwd); \
  python3 \${TRAF70_REMOTE_ABS}/stage/${TRAF70_HEAD}/examples/a100_nvlink_packet_v2/run_study.py \
    --pending-indices \
    --output-root \${TRAF70_REMOTE_ABS} \
    --expected-head ${TRAF70_HEAD} \
    --freeze-sha256 ${TRAF70_FREEZE}")"

if [ -n "${TRAF70_PENDING}" ]; then
  ssh merlin "set -eu; \
    TRAF70_REMOTE_ABS=\$(cd ${TRAF70_REMOTE_ROOT} && pwd); \
    TRAF70_NCCL_ABS=\$(cd ${TRAF70_REMOTE_NCCL} && pwd); \
    sbatch -M gmerlin7 \
      --array=${TRAF70_PENDING}%1 \
      --export=ALL,TRAF70_STAGE_ROOT=\${TRAF70_REMOTE_ABS}/stage/${TRAF70_HEAD},TRAF70_DATA_ROOT=\${TRAF70_REMOTE_ABS},TRAF70_EXPECTED_HEAD=${TRAF70_HEAD},TRAF70_NCCL_ROOT=\${TRAF70_NCCL_ABS},TRAF70_CUDA_MODULE=cuda/12.9.1,TRAF70_PYTHON_MODULE=Python/3.10.16 \
      \${TRAF70_REMOTE_ABS}/stage/${TRAF70_HEAD}/examples/a100_nvlink_packet_v2/run_merlin_cell.sbatch"
fi
```

The `%1` limit ensures that only one four-GPU cell occupies the qualified node
at a time. The task-indexed pause avoids scheduler-side bursts. Scheduler HUP,
INT and TERM signals terminate the active producer, write one stop record and
leave the cell resumable. A submitted Slurm job is scheduler-owned, so loss of
the submitting SSH connection does not create an ambiguous local child.

Repeat the pending command and submission across short windows until the
pending string is empty. Never overwrite a digest-complete attempt.

## Lean pull and literal scoring

After the pending set is empty, pull manifests, summaries, environment and
result rows into the external local bulk root. Do not pull CUDA builds or
unrelated remote data. Run:

```bash
.venv/bin/python examples/a100_nvlink_packet_v2/score_hardware.py \
  --bulk-root "${TRAF70_LOCAL_ROOT}" \
  --expected-head "${TRAF70_HEAD}" \
  --scheduler-job "<MERLIN_JOB_ID>" \
  --json-out examples/a100_nvlink_packet_v2/hardware-score.json \
  --markdown-out examples/a100_nvlink_packet_v2/RESULTS.md
```

Only the scorer's `profile_patch.changes` list may update the A100 profile.
Each listed entry names its rule and evidence class. An identified value that
differs from the candidate is an explicit refutation and replacement. An
inconclusive or void parameter remains byte-unchanged and declared.

## Published parameter result

The score identifies the TX endpoint egress rate as
`160795737454` bytes per second and the RX ingress rate as
`207101921876` bytes per second. Both replace the 300,000,000,000-byte per
second candidates with measured effective counter-plateau evidence. It also
confirms the existing request and response direction, `extent_sequence`
reassembly and `per_extent` delivery with their rule-specific measured
evidence classes.

Packet payload and header size, link count and per-link rate, bond policy,
credit unit and count, RX buffer and credit-return latency, and TX/RX queue
scope remain inconclusive declared candidates. The A100 direct-mesh switch
remains `pass_through` as a structural invariant, not a measurement. No other
runtime value or evidence class changed.
