#!/bin/bash
# Build and run one frozen TRAF-81 rank cell inside its Slurm allocation.

set -u

stage_root="${TRAF81_STAGE_ROOT:?set TRAF81_STAGE_ROOT}"
data_root="${TRAF81_DATA_ROOT:?set TRAF81_DATA_ROOT}"
nccl_root="${TRAF81_NCCL_ROOT:?set TRAF81_NCCL_ROOT}"
cell="${TRAF81_CELL:?set TRAF81_CELL}"
expected_ranks="${TRAF81_EXPECTED_RANKS:?set TRAF81_EXPECTED_RANKS}"
batch_source="${TRAF81_BATCH_SOURCE:?set TRAF81_BATCH_SOURCE}"
freeze_commit="${TRAF81_FREEZE_COMMIT:?set TRAF81_FREEZE_COMMIT}"
harness_commit="${TRAF81_HARNESS_COMMIT:?set TRAF81_HARNESS_COMMIT}"
output_root="${data_root}/${cell}/${SLURM_JOB_ID}"
mkdir -p "${output_root}" || exit 1

log() { echo "[${cell}] $*"; }
fail() { echo "[${cell}] FATAL $*"; exit 1; }

commit_count=$(printf '%s\n%s\n' "${freeze_commit}" "${harness_commit}" |
  grep -Ec '^[0-9a-f]{40}$' || true)
if [ "${commit_count}" -ne 2 ]; then
  fail "freeze and harness commits must be full lowercase Git object IDs"
fi
if [ "${SLURM_NTASKS}" != "${expected_ranks}" ]; then
  fail "expected ${expected_ranks} tasks, received ${SLURM_NTASKS}"
fi

{
  echo "schema=simllm-collective-floor-job-context-v1"
  echo "job_id=${SLURM_JOB_ID}"
  echo "cell=${cell}"
  echo "freeze_commit=${freeze_commit}"
  echo "harness_commit=${harness_commit}"
  echo "date=$(date -Is)"
  echo "cluster=${SLURM_CLUSTER_NAME:-unset}"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "nodelist=${SLURM_JOB_NODELIST}"
  echo "nodes=${SLURM_JOB_NUM_NODES}"
  echo "tasks=${SLURM_NTASKS}"
  echo "tasks_per_node=${SLURM_NTASKS_PER_NODE:-unset}"
  echo "stage_root=${stage_root}"
  echo "data_root=${data_root}"
  echo "nccl_root=${nccl_root}"
  echo "output_root=${output_root}"
} >"${output_root}/job_context.txt"

module purge 2>/dev/null
module load cuda/12.2.2 || fail "cuda/12.2.2 module load failed"
nvcc --version >"${output_root}/nvcc_version.txt" 2>&1

if [ ! -f "${nccl_root}/include/nccl.h" ] ||
   [ ! -f "${nccl_root}/lib/libnccl.so.2" ]; then
  fail "staged NCCL headers or library are missing"
fi
if [ ! -e "${nccl_root}/lib/libnccl.so" ]; then
  fail "libnccl.so link is missing beside libnccl.so.2"
fi

cp "${stage_root}/collective_lane.cu" "${output_root}/"
cp "${stage_root}/collect_guard.sh" "${output_root}/"
cp "${stage_root}/run_merlin_cell.sh" "${output_root}/"
cp "${stage_root}/${batch_source}" "${output_root}/"
sha256sum "${output_root}/collective_lane.cu" \
  "${output_root}/collect_guard.sh" \
  "${output_root}/run_merlin_cell.sh" \
  "${output_root}/${batch_source}" \
  "${nccl_root}/include/nccl.h" \
  "${nccl_root}/lib/libnccl.so.2" >"${output_root}/inputs.sha256"

nvcc -O3 -std=c++17 -arch=sm_80 \
  -I"${nccl_root}/include" \
  -o "${output_root}/collective_lane" \
  "${output_root}/collective_lane.cu" \
  -L"${nccl_root}/lib" -lnccl \
  -Xlinker -rpath="${nccl_root}/lib" \
  >"${output_root}/build.txt" 2>&1 || {
    tail -100 "${output_root}/build.txt"
    fail "build failed"
  }
sha256sum "${output_root}/collective_lane" >>"${output_root}/inputs.sha256"

log "collecting pre-run guards"
srun --ntasks="${SLURM_NTASKS}" \
  "${output_root}/collect_guard.sh" "${output_root}" before ||
  fail "pre-run guard collection failed"

rank_directories=$(find "${output_root}" -mindepth 1 -maxdepth 1 \
  -type d -name 'rank_*' | wc -l)
if [ "${rank_directories}" != "${SLURM_NTASKS}" ]; then
  fail "rank directory count ${rank_directories} does not match ${SLURM_NTASKS}"
fi
if grep -h '^assigned_gpu=' "${output_root}"/rank_*/guards_before.txt |
   grep -qv 'NVIDIA A100-SXM4-80GB'; then
  fail "allocated target is not uniformly A100-SXM4-80GB"
fi
assigned_gpus=$(grep -h '^assigned_gpu=' \
  "${output_root}"/rank_*/guards_before.txt | cut -d, -f1 | sort -u | wc -l)
if [ "${assigned_gpus}" != "${SLURM_NTASKS}" ]; then
  fail "only ${assigned_gpus} distinct GPUs serve ${SLURM_NTASKS} ranks"
fi
for guard in "${output_root}"/rank_*/guards_before.txt; do
  process_count=$(sed -n '/^=== compute processes ===$/,/^=== high-speed ports ===$/p' \
    "${guard}" | grep -cE '^GPU-[^,]+, [0-9]+,' || true)
  if [ "${process_count}" -ne 0 ]; then
    fail "foreign compute process occupies the assigned GPU in ${guard}"
  fi
done
topology_rows=$(awk '
  $1 ~ /^GPU[0-3]$/ {
    count = 0
    for (field = 2; field <= NF; ++field) {
      if ($field == "NV4") count += 1
    }
    if (count == 3) valid += 1
  }
  END { print valid + 0 }
' "$(find "${output_root}" -name guards_before.txt | sort | head -1)")
if [ "${topology_rows}" -ne 4 ]; then
  fail "node does not expose the frozen four-GPU direct NV4 mesh"
fi
if grep -h '^cassini_port_count=' "${output_root}"/rank_*/guards_before.txt |
   grep -qv '^cassini_port_count=4$'; then
  fail "a node does not expose four Cassini ports"
fi

log "running frozen collectives"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING,COLL
export NCCL_DEBUG_FILE="${output_root}/nccl_debug.%h.%p.log"
srun --ntasks="${SLURM_NTASKS}" \
  "${output_root}/collective_lane" \
  --out "${output_root}/measurement.json" \
  --id-path "${output_root}/nccl_unique_id.bin" \
  >"${output_root}/run.txt" 2>&1
run_status=$?
echo "${run_status}" >"${output_root}/run.exit_status"

grep -hoE 'Using network [A-Za-z]+|via NET/[A-Za-z0-9_/]+|via P2P/[a-z]+|GDR [0-9]' \
  "${output_root}"/nccl_debug.*.log 2>/dev/null | sort -u \
  >"${output_root}/transport_summary.txt" || true

log "collecting post-run guards"
srun --ntasks="${SLURM_NTASKS}" \
  "${output_root}/collect_guard.sh" "${output_root}" after
post_guard_status=$?

if [ "${run_status}" -ne 0 ]; then
  tail -100 "${output_root}/run.txt"
  if ! grep -q 'first_target_timing_started' "${output_root}/run.txt"; then
    echo "BLOCKED pre-timing NCCL or CUDA failure" >"${output_root}/cell_state.txt"
  else
    echo "VOID failure after target timing began" >"${output_root}/cell_state.txt"
  fi
  exit "${run_status}"
fi

if [ "${post_guard_status}" -ne 0 ]; then
  echo "VOID post-run guard collection failed" >"${output_root}/cell_state.txt"
  exit "${post_guard_status}"
fi
for before in "${output_root}"/rank_*/guards_before.txt; do
  after="${before%before.txt}after.txt"
  before_gpu=$(grep '^assigned_gpu=' "${before}")
  after_gpu=$(grep '^assigned_gpu=' "${after}")
  if [ "${before_gpu}" != "${after_gpu}" ]; then
    echo "VOID assigned GPU changed during measurement" \
      >"${output_root}/cell_state.txt"
    exit 1
  fi
  process_count=$(sed -n '/^=== compute processes ===$/,/^=== high-speed ports ===$/p' \
    "${after}" | grep -cE '^GPU-[^,]+, [0-9]+,' || true)
  if [ "${process_count}" -ne 0 ]; then
    echo "VOID foreign process observed after measurement" \
      >"${output_root}/cell_state.txt"
    exit 1
  fi
done

echo "MEASURED" >"${output_root}/cell_state.txt"
sha256sum "${output_root}/measurement.json" \
  "${output_root}/job_context.txt" \
  "${output_root}/transport_summary.txt" >"${output_root}/outputs.sha256"
log "completed ${cell} under ${output_root}"
