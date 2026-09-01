#!/bin/bash
# Build and run one TRAF-77 width and concentration combination on allocated GPUs.

set -euo pipefail

# The site login and compute images default to python 3.6; the manifest
# tooling needs 3.7 or newer, so select the newest available interpreter.
PYTHON3="$(command -v python3.12 || command -v python3.11 || command -v python3)"

STAGE="${TRAF77_STAGE_ROOT:?set TRAF77_STAGE_ROOT}"
DATA_ROOT="${TRAF77_DATA_ROOT:?set TRAF77_DATA_ROOT}"
NCCL_ROOT="${TRAF77_NCCL_ROOT:?set TRAF77_NCCL_ROOT}"
WIDTH="${TRAF77_WIDTH:?set TRAF77_WIDTH}"
CONCENTRATION="${TRAF77_CONCENTRATION:?set TRAF77_CONCENTRATION}"
SUBMITTED_SCRIPT="${TRAF77_SUBMITTED_SCRIPT:?set TRAF77_SUBMITTED_SCRIPT}"

CONFIG="${STAGE}/study_config.json"
MANIFEST="${STAGE}/submitted_scripts.local.sha256"
ATTEMPT="w${WIDTH}_${CONCENTRATION}_job${SLURM_JOB_ID}"
OUT_ROOT="${DATA_ROOT}/merlin_collective_capture_v1/attempts"
OUT="${OUT_ROOT}/${ATTEMPT}"

log() { echo "[${ATTEMPT}] $*"; }
fail() { echo "[${ATTEMPT}] FATAL $*"; exit 1; }

mkdir -p "${OUT_ROOT}"
if ! mkdir "${OUT}"; then
  fail "append-only attempt directory already exists: ${OUT}"
fi
mkdir "${OUT}/submitted" "${OUT}/rank_identity" "${OUT}/counter_snapshots"

status=1
finish() {
  printf 'exit_status=%s\nfinished_at=%s\n' "${status}" "$(date -Is)" \
    >> "${OUT}/attempt_status.txt"
}
trap finish EXIT

[ -f "${CONFIG}" ] || fail "missing study configuration"
[ -f "${MANIFEST}" ] || fail "missing local submitted-source manifest"
"${PYTHON3}" "${STAGE}/hash_manifest.py" --root "${STAGE}" --check "${MANIFEST}" \
  || fail "staged source manifest does not match the integrator copy"
"${PYTHON3}" "${STAGE}/hash_manifest.py" --root "${STAGE}" \
  --output "${OUT}/submitted_scripts.remote.sha256"
cp "${MANIFEST}" "${OUT}/submitted_scripts.local.sha256"
cmp "${OUT}/submitted_scripts.local.sha256" \
  "${OUT}/submitted_scripts.remote.sha256" \
  || fail "local and remote submitted-source hashes differ"

while read -r _digest relative; do
  relative="${relative#\*}"
  mkdir -p "${OUT}/submitted/$(dirname "${relative}")"
  cp "${STAGE}/${relative}" "${OUT}/submitted/${relative}"
done < "${MANIFEST}"
"${PYTHON3}" "${OUT}/submitted/hash_manifest.py" --root "${OUT}/submitted" \
  --check "${MANIFEST}" || fail "attempt source copy failed verification"

SCRIPT_SHA256="$(awk -v name="${SUBMITTED_SCRIPT}" '$2 == name {print $1}' "${MANIFEST}")"
[ -n "${SCRIPT_SHA256}" ] || fail "submitted script is absent from the manifest"
CONFIG_SHA256="$(sha256sum "${CONFIG}" | awk '{print $1}')"
REPEATS="$("${PYTHON3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["measured_repeats"])' "${CONFIG}")"
WARMUPS="$("${PYTHON3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["excluded_warmups"])' "${CONFIG}")"
CHUNK_BYTES="$("${PYTHON3}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["chunk_limit_bytes"])' "${CONFIG}")"

if [ "${SLURM_NTASKS}" != "${WIDTH}" ]; then
  fail "allocation has ${SLURM_NTASKS} tasks for declared width ${WIDTH}"
fi
if [ "${SLURM_JOB_NUM_NODES}" != "2" ]; then
  fail "every frozen combination requires exactly two nodes"
fi

{
  echo "schema=simllm-merlin-collective-attempt-context-v1"
  echo "evidence_class=hardware-capture"
  echo "attempt=${ATTEMPT}"
  echo "job_id=${SLURM_JOB_ID}"
  echo "cluster=${SLURM_CLUSTER_NAME:-unset}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "account=${SLURM_JOB_ACCOUNT:-merlin}"
  echo "nodelist=${SLURM_JOB_NODELIST}"
  echo "nodes=${SLURM_JOB_NUM_NODES}"
  echo "tasks=${SLURM_NTASKS}"
  echo "tasks_per_node=${SLURM_NTASKS_PER_NODE:-unset}"
  echo "width=${WIDTH}"
  echo "concentration=${CONCENTRATION}"
  echo "submitted_script=${SUBMITTED_SCRIPT}"
  echo "submitted_script_sha256=${SCRIPT_SHA256}"
  echo "config_sha256=${CONFIG_SHA256}"
  echo "measured_repeats=${REPEATS}"
  echo "excluded_warmups=${WARMUPS}"
  echo "chunk_limit_bytes=${CHUNK_BYTES}"
  echo "started_at=$(date -Is)"
} > "${OUT}/job_context.txt"
cat "${OUT}/job_context.txt"

module purge 2>/dev/null || true
module load cuda/12.2.2 || fail "cuda/12.2.2 module load failed"
nvcc --version > "${OUT}/nvcc_version.txt" 2>&1
if grep -q 'release 13\.' "${OUT}/nvcc_version.txt"; then
  fail "CUDA 13 is forbidden on driver 565.57.01"
fi
grep -q 'release 12\.2' "${OUT}/nvcc_version.txt" \
  || fail "the loaded compiler is not CUDA 12.2"

[ -f "${NCCL_ROOT}/lib/libnccl.so.2" ] \
  || fail "the staged nvidia-nccl-cu12 wheel has no libnccl.so.2"
if [ ! -e "${NCCL_ROOT}/lib/libnccl.so" ]; then
  ln -s libnccl.so.2 "${NCCL_ROOT}/lib/libnccl.so"
fi
[ -f "${NCCL_ROOT}/lib/libnccl.so" ] \
  || fail "libnccl.so symlink creation failed"

{
  echo "=== uname ==="
  uname -a
  echo "=== operating system ==="
  test -r /etc/os-release && cat /etc/os-release
  echo "=== modules ==="
  module list 2>&1 || true
  echo "=== Slurm hostnames ==="
  scontrol show hostnames "${SLURM_JOB_NODELIST}"
  echo "=== driver ==="
  nvidia-smi --query-gpu=driver_version --format=csv,noheader
  echo "=== CUDA ==="
  cat "${OUT}/nvcc_version.txt"
  echo "=== NCCL wheel library ==="
  ls -l "${NCCL_ROOT}/lib/libnccl.so" "${NCCL_ROOT}/lib/libnccl.so.2"
  sha256sum "${NCCL_ROOT}/lib/libnccl.so.2"
} > "${OUT}/environment.txt" 2>&1

log "building the A100 lane"
nvcc -O3 -std=c++17 -arch=sm_80 \
  -I"${NCCL_ROOT}/include" \
  -o "${OUT}/merlin_collective_lane" \
  "${STAGE}/merlin_collective_lane.cu" \
  -L"${NCCL_ROOT}/lib" -lnccl \
  -Xlinker -rpath="${NCCL_ROOT}/lib" -lpthread \
  > "${OUT}/build.txt" 2>&1 \
  || { cat "${OUT}/build.txt"; fail "lane build failed"; }
sha256sum "${OUT}/merlin_collective_lane" > "${OUT}/binary.sha256"
ldd "${OUT}/merlin_collective_lane" > "${OUT}/linked_libraries.txt"

if [ "${CONCENTRATION}" = "one-port" ]; then
  export NCCL_SOCKET_IFNAME='=hsn0'
elif [ "${CONCENTRATION}" = "four-port" ]; then
  unset NCCL_SOCKET_IFNAME
else
  fail "unknown concentration ${CONCENTRATION}"
fi

log "recording placement and topology"
srun --ntasks="${SLURM_NTASKS}" "${STAGE}/capture_rank_identity.sh" \
  "${OUT}/rank_identity"

log "snapshotting job-level counters before"
srun --ntasks-per-node=1 "${PYTHON3}" "${STAGE}/snapshot_counters.py" \
  --out-dir "${OUT}/counter_snapshots" --tag before \
  --net-class-root /sys/class/net --interfaces hsn0 hsn1 hsn2 hsn3

log "running 88 frozen cells"
set +e
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING \
NCCL_DEBUG_FILE="${OUT}/nccl_debug.%h.%p.log" \
srun --ntasks="${SLURM_NTASKS}" \
  "${OUT}/merlin_collective_lane" \
    --out "${OUT}/capture.jsonl" \
    --id-path "${OUT}/nccl_unique_id.bin" \
    --attempt-id "${ATTEMPT}" \
    --concentration "${CONCENTRATION}" \
    --script-name "${SUBMITTED_SCRIPT}" \
    --script-sha256 "${SCRIPT_SHA256}" \
    --config-sha256 "${CONFIG_SHA256}" \
    --repeats "${REPEATS}" \
    --warmups "${WARMUPS}" \
    --chunk-bytes "${CHUNK_BYTES}" \
    --net-class-root /sys/class/net \
  > "${OUT}/run.txt" 2>&1
lane_status=$?
set -e

log "snapshotting job-level counters after"
srun --ntasks-per-node=1 "${PYTHON3}" "${STAGE}/snapshot_counters.py" \
  --out-dir "${OUT}/counter_snapshots" --tag after \
  --net-class-root /sys/class/net --interfaces hsn0 hsn1 hsn2 hsn3 \
  || true

grep -hE 'Using network|via NET/|Channel|coll channels|TUNING|Algo|Proto|GDR' \
  "${OUT}"/nccl_debug.*.log > "${OUT}/nccl_selection.txt" 2>/dev/null || true
grep -hoE 'Using network [A-Za-z]+|via NET/[A-Za-z0-9_/]+|GDR [0-9]' \
  "${OUT}"/nccl_debug.*.log 2>/dev/null | sort | uniq -c \
  > "${OUT}/transport_summary.txt" || true
grep -hiE 'GDR|GPU Direct' "${OUT}"/nccl_debug.*.log \
  > "${OUT}/gdr_state.txt" 2>/dev/null || true

tail -40 "${OUT}/run.txt" || true
cat "${OUT}/transport_summary.txt" || true
printf 'lane_status=%s\n' "${lane_status}" >> "${OUT}/attempt_status.txt"
if [ "${lane_status}" -ne 0 ]; then
  fail "lane exited ${lane_status}; all partial evidence is retained"
fi

line_count="$(wc -l < "${OUT}/capture.jsonl")"
[ "${line_count}" = "88" ] \
  || fail "capture has ${line_count} JSON lines, expected 88"

status=0
log "capture complete under ${OUT}"
