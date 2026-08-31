# Merlin collective capture integrator runbook

This runbook stages and submits the harness frozen by commit `9c9a42e`. It
does no compute on the login node and uses no network package installation.
Run every command in order. Keep the first local shell open so its variables
remain defined.

The four jobs have this fixed coverage:

| Submitted script | Nodes | GPUs per node | Width | Concentration | Cells |
|---|---:|---:|---:|---|---:|
| `capture_w2_one_port.sbatch` | 2 | 1 | 2 | `one-port` | 4 operations x 22 payloads = 88 |
| `capture_w2_four_port.sbatch` | 2 | 1 | 2 | `four-port` | 4 operations x 22 payloads = 88 |
| `capture_w8_one_port.sbatch` | 2 | 4 | 8 | `one-port` | 4 operations x 22 payloads = 88 |
| `capture_w8_four_port.sbatch` | 2 | 4 | 8 | `four-port` | 4 operations x 22 payloads = 88 |

Each job runs the 8-byte consistency anchor first and stops before its other
cells if FG-4 misses the frozen factor-two band. A width-8 job checks both its
all-reduce and pairwise all-to-allv anchors before continuing.

## 1. Prepare and hash the local staging copy

Run on the integration host from this repository:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
LOCAL_HARNESS="${REPO_ROOT}/examples/merlin_collective_capture_v1/harness"
LOCAL_EVIDENCE_ROOT=/"data3/yifeng/simllm-dev/planmode-runs/traf77-t2"
LOCAL_STAGE="$(mktemp -d)"
mkdir -p "${LOCAL_EVIDENCE_ROOT}"
git merge-base --is-ancestor 9c9a42e HEAD
git diff --exit-code 9c9a42e -- examples/merlin_collective_capture_v1/expectations.md
git rev-parse HEAD | tee "${LOCAL_EVIDENCE_ROOT}/submitted_commit.txt"
rsync -a --checksum "${LOCAL_HARNESS}/" "${LOCAL_STAGE}/"
python3 "${LOCAL_STAGE}/hash_manifest.py" \
  --root "${LOCAL_STAGE}" \
  --output "${LOCAL_STAGE}/submitted_scripts.local.sha256"
cp "${LOCAL_STAGE}/submitted_scripts.local.sha256" \
  "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.local.sha256"
sha256sum "${LOCAL_STAGE}"/*.sbatch \
  | sed "s#${LOCAL_STAGE}/##" \
  | tee "${LOCAL_EVIDENCE_ROOT}/submitted_sbatch.local.sha256"
```

The flagship fence is external to this harness. Before any A100 submission,
place the actual ping and reply quotations in the evidence root, one line
beginning `ping_quote=` and one beginning `reply_quote=`. Then run:

```bash
test -s "${LOCAL_EVIDENCE_ROOT}/flagship_ping_and_reply.txt"
grep -q '^ping_quote=.' "${LOCAL_EVIDENCE_ROOT}/flagship_ping_and_reply.txt"
grep -q '^reply_quote=.' "${LOCAL_EVIDENCE_ROOT}/flagship_ping_and_reply.txt"
```

Do not replace this record with invented text. A missing record stops the
campaign at this point.

## 2. Create the remote directories and stage with rsync

The remote paths are under the required home-directory roots. This `ssh`
command creates directories only:

```bash
ssh merlin 'set -eu
mkdir -p "${HOME}/simllm-stage/merlin_collective_capture_v1"
mkdir -p "${HOME}/simllm-data/merlin_collective_capture_v1/slurm"'
```

Stage the exact local copy:

```bash
rsync -av --checksum "${LOCAL_STAGE}/" \
  'merlin:simllm-stage/merlin_collective_capture_v1/'
```

## 3. Select the existing wheel, create the linker symlink and verify hashes

This finds an already staged `nvidia-nccl-cu12` wheel payload. It never calls
pip and never contacts a package index. It creates `libnccl.so` beside the
wheel's `libnccl.so.2`, then records the exact reusable environment. The
tracked job runner loads `cuda/12.2.2` only after the GPU allocation starts:

```bash
ssh merlin 'set -eu
STAGE_ROOT="${HOME}/simllm-stage/merlin_collective_capture_v1"
DATA_ROOT="${HOME}/simllm-data"
NCCL_LIBRARY="$(find "${HOME}/simllm-stage" -type f \
  -path "*nvidia_nccl_cu12*" -name "libnccl.so.2" \
  -print -quit)"
test -n "${NCCL_LIBRARY}"
NCCL_LIB_DIR="$(dirname "${NCCL_LIBRARY}")"
NCCL_ROOT="$(dirname "${NCCL_LIB_DIR}")"
test -f "${NCCL_ROOT}/include/nccl.h"
if test ! -e "${NCCL_ROOT}/lib/libnccl.so"; then
  ln -s libnccl.so.2 "${NCCL_ROOT}/lib/libnccl.so"
fi
test -f "${NCCL_ROOT}/lib/libnccl.so"
{
  printf "export TRAF77_STAGE_ROOT=%q\n" "${STAGE_ROOT}"
  printf "export TRAF77_DATA_ROOT=%q\n" "${DATA_ROOT}"
  printf "export TRAF77_NCCL_ROOT=%q\n" "${NCCL_ROOT}"
} > "${STAGE_ROOT}/.env.local.sh"
python3 "${STAGE_ROOT}/hash_manifest.py" \
  --root "${STAGE_ROOT}" \
  --check "${STAGE_ROOT}/submitted_scripts.local.sha256"
python3 "${STAGE_ROOT}/hash_manifest.py" \
  --root "${STAGE_ROOT}" \
  --output "${STAGE_ROOT}/submitted_scripts.remote.sha256"
cmp "${STAGE_ROOT}/submitted_scripts.local.sha256" \
  "${STAGE_ROOT}/submitted_scripts.remote.sha256"
sha256sum "${STAGE_ROOT}"/*.sbatch \
  | sed "s#${STAGE_ROOT}/##" \
  > "${STAGE_ROOT}/submitted_sbatch.remote.sha256"'
```

Fetch the pre-submission remote hash records with `scp` and compare them on the
integration host:

```bash
scp merlin:simllm-stage/merlin_collective_capture_v1/submitted_scripts.remote.sha256 \
  "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.remote.pre_submit.sha256"
scp merlin:simllm-stage/merlin_collective_capture_v1/submitted_sbatch.remote.sha256 \
  "${LOCAL_EVIDENCE_ROOT}/submitted_sbatch.remote.pre_submit.sha256"
cmp "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.local.sha256" \
  "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.remote.pre_submit.sha256"
diff -u "${LOCAL_EVIDENCE_ROOT}/submitted_sbatch.local.sha256" \
  "${LOCAL_EVIDENCE_ROOT}/submitted_sbatch.remote.pre_submit.sha256"
```

## 4. Submit the four A100 jobs

Every submission names cluster `gmerlin7` and account `merlin` on the command
line even though the tracked batch files also carry the account. The Slurm
stdout and stderr paths are under the remote data root.

```bash
: > "${LOCAL_EVIDENCE_ROOT}/submitted_jobs.txt"
ssh merlin 'set -eu
. "${HOME}/simllm-stage/merlin_collective_capture_v1/.env.local.sh"
sbatch -M gmerlin7 --account=merlin \
  --output="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.out" \
  --error="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.err" \
  "${TRAF77_STAGE_ROOT}/capture_w2_one_port.sbatch"' \
  | tee -a "${LOCAL_EVIDENCE_ROOT}/submitted_jobs.txt"
ssh merlin 'set -eu
. "${HOME}/simllm-stage/merlin_collective_capture_v1/.env.local.sh"
sbatch -M gmerlin7 --account=merlin \
  --output="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.out" \
  --error="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.err" \
  "${TRAF77_STAGE_ROOT}/capture_w2_four_port.sbatch"' \
  | tee -a "${LOCAL_EVIDENCE_ROOT}/submitted_jobs.txt"
ssh merlin 'set -eu
. "${HOME}/simllm-stage/merlin_collective_capture_v1/.env.local.sh"
sbatch -M gmerlin7 --account=merlin \
  --output="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.out" \
  --error="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.err" \
  "${TRAF77_STAGE_ROOT}/capture_w8_one_port.sbatch"' \
  | tee -a "${LOCAL_EVIDENCE_ROOT}/submitted_jobs.txt"
ssh merlin 'set -eu
. "${HOME}/simllm-stage/merlin_collective_capture_v1/.env.local.sh"
sbatch -M gmerlin7 --account=merlin \
  --output="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.out" \
  --error="${TRAF77_DATA_ROOT}/merlin_collective_capture_v1/slurm/%x-%j.err" \
  "${TRAF77_STAGE_ROOT}/capture_w8_four_port.sbatch"' \
  | tee -a "${LOCAL_EVIDENCE_ROOT}/submitted_jobs.txt"
```

## 5. Poll with squeue and record terminal accounting

```bash
JOB_IDS="$(awk '$1 == "Submitted" && $2 == "batch" && $3 == "job" {print $4}' \
  "${LOCAL_EVIDENCE_ROOT}/submitted_jobs.txt" | paste -sd, -)"
test "$(tr ',' '\n' <<< "${JOB_IDS}" | grep -c .)" = 4
while :; do
  ssh merlin "squeue -M gmerlin7 -h -j ${JOB_IDS} -o '%i %T %M %N'" \
    | tee -a "${LOCAL_EVIDENCE_ROOT}/squeue_poll.txt"
  ACTIVE="$(ssh merlin "squeue -M gmerlin7 -h -j ${JOB_IDS} -o '%i'" \
    | grep -c . || true)"
  test "${ACTIVE}" = 0 && break
  sleep 30
done
ssh merlin "sacct -M gmerlin7 -j ${JOB_IDS} \
  --format=JobID,JobName,Partition,Account,State,ExitCode,Elapsed,NodeList" \
  | tee "${LOCAL_EVIDENCE_ROOT}/sacct.txt"
```

Do not resubmit a failed job into an existing attempt directory. A new Slurm
job ID creates a new append-only attempt and preserves the failed evidence.

## 6. Fetch the evidence and recheck every submitted hash

```bash
mkdir -p "${LOCAL_EVIDENCE_ROOT}/raw"
rsync -av --checksum --partial \
  'merlin:simllm-data/merlin_collective_capture_v1/' \
  "${LOCAL_EVIDENCE_ROOT}/raw/"
scp merlin:simllm-stage/merlin_collective_capture_v1/submitted_scripts.remote.sha256 \
  "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.remote.post_run.sha256"
cmp "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.local.sha256" \
  "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.remote.post_run.sha256"
for attempt in "${LOCAL_EVIDENCE_ROOT}/raw/attempts/"*; do
  manifest="${attempt}/submitted_scripts.remote.sha256"
  cmp "${LOCAL_EVIDENCE_ROOT}/submitted_scripts.local.sha256" "${manifest}"
  python3 "${LOCAL_HARNESS}/hash_manifest.py" \
    --root "${attempt}/submitted" \
    --check "${attempt}/submitted_scripts.remote.sha256"
done
```

## 7. Normalize offline and verify deterministic analysis

These commands run only on fetched evidence. They do not access Merlin and do
not perform T2B scoring:

```bash
env -u PYTHONPATH python3 "${LOCAL_HARNESS}/analyze_capture.py" \
  --capture-root "${LOCAL_EVIDENCE_ROOT}/raw" \
  --output "${LOCAL_EVIDENCE_ROOT}/normalized.1.json"
env -u PYTHONPATH python3 "${LOCAL_HARNESS}/analyze_capture.py" \
  --capture-root "${LOCAL_EVIDENCE_ROOT}/raw" \
  --output "${LOCAL_EVIDENCE_ROOT}/normalized.2.json"
cmp "${LOCAL_EVIDENCE_ROOT}/normalized.1.json" \
  "${LOCAL_EVIDENCE_ROOT}/normalized.2.json"
sha256sum "${LOCAL_EVIDENCE_ROOT}/normalized.1.json" \
  | tee "${LOCAL_EVIDENCE_ROOT}/normalized.sha256"
```

Retain both deterministic outputs, every failed or partial attempt, all Slurm
logs and the ping record. T2B decides whether the campaign is interpretable and
publishes hardware outcomes. Nothing from this runbook belongs in Git as a
capture result during T2A.
