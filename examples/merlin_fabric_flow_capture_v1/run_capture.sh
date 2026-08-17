#!/bin/bash
# Shared body of every homogeneous (single-partition) capture cell.
#
# Collects the fatal-guard evidence per rank before and after the run,
# snapshots interface byte counters, samples counters and established
# sockets at 1 Hz while the lane runs, builds the lane on the allocation,
# runs it with NCCL transport logging, and enforces the G1 checks the
# wave-16 runner enforced.
#
# Required environment, no personal-path defaults anywhere:
#   W18_STAGE_ROOT   staged sources
#   W18_DATA_ROOT    output root
#   W18_NCCL_ROOT    staged NCCL wheel root for this architecture
#   W18_CUDA_MODULE  cuda module (cuda/12.2.2 on A100, cuda/12.9.1 on GH)
#   W18_ARCH_FLAG    sm_80 on A100, sm_90 on GH200
#   W18_CELL         cell name from the freeze
#   W18_WINDOW_S     window seconds
#   W18_OFFSETS      comma-separated join offsets, one per source
#   W18_DEST_RANK    destination rank (default 0)
#   W18_SOURCES      comma-separated source ranks (default: all but dest)
#   W18_CHUNK_BYTES  frozen chunk size (default 8388608)

set -u

STAGE="${W18_STAGE_ROOT:?set W18_STAGE_ROOT}"
OUT_ROOT="${W18_DATA_ROOT:?set W18_DATA_ROOT}"
NCCL_ROOT="${W18_NCCL_ROOT:?set W18_NCCL_ROOT}"
CUDA_MODULE="${W18_CUDA_MODULE:?set W18_CUDA_MODULE}"
ARCH_FLAG="${W18_ARCH_FLAG:?set W18_ARCH_FLAG}"
CELL="${W18_CELL:?set W18_CELL}"
WINDOW_S="${W18_WINDOW_S:?set W18_WINDOW_S}"
OFFSETS="${W18_OFFSETS:?set W18_OFFSETS}"
DEST_RANK="${W18_DEST_RANK:-0}"
SOURCES="${W18_SOURCES:-}"
CHUNK_BYTES="${W18_CHUNK_BYTES:-8388608}"

OUT="${OUT_ROOT}/capture/${CELL}/${SLURM_JOB_ID}"
mkdir -p "${OUT}" || exit 1

log() { echo "[${CELL}] $*"; }
fail() { echo "[${CELL}] FATAL $*"; exit 1; }

log "job ${SLURM_JOB_ID} on $(hostname) at $(date -Is)"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "cell=${CELL}"
  echo "window_s=${WINDOW_S}"
  echo "offsets=${OFFSETS}"
  echo "dest_rank=${DEST_RANK}"
  echo "sources=${SOURCES:-default}"
  echo "chunk_bytes=${CHUNK_BYTES}"
  echo "date_iso=$(date -Is)"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "nodelist=${SLURM_JOB_NODELIST}"
  echo "nnodes=${SLURM_JOB_NUM_NODES}"
  echo "ntasks=${SLURM_NTASKS}"
  echo "nccl_root=${NCCL_ROOT}"
} > "${OUT}/job_context.txt"
cat "${OUT}/job_context.txt"

# Guard evidence per rank, the wave-16 battery.
cat > "${OUT}/guards.sh" <<'GUARDS'
#!/bin/bash
set -u
OUT="$1"; WHEN="$2"
RANK_OUT="${OUT}/rank_$(printf '%02d' "${SLURM_PROCID:-0}")_$(hostname)"
mkdir -p "${RANK_OUT}"
exec > "${RANK_OUT}/guards_${WHEN}.txt" 2>&1
echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "SLURM_PROCID=${SLURM_PROCID:-unset} SLURM_LOCALID=${SLURM_LOCALID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "=== G6 rank placement ==="
nvidia-smi -L
nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/^/gpu_uuid=/'
echo "=== G2 clocks ==="
nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv
echo "=== G4 foreign processes ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "=== G1 host NIC inventory ==="
lspci 2>/dev/null | grep -icE 'cassini' | sed 's/^/cassini_port_count=/'
echo "ib_device_count=$(ls /sys/class/infiniband 2>/dev/null | grep -c . || echo 0)"
echo "=== interfaces ==="
ip -br link 2>/dev/null | awk '{print $1, $2}'
GUARDS
chmod +x "${OUT}/guards.sh"

# Byte counters, tx side authoritative (see the freeze's discovery note).
cat > "${OUT}/counters.sh" <<'COUNTERS'
#!/bin/bash
set -u
OUT="$1"; TAG="$2"
NODE_OUT="${OUT}/node_$(hostname)"
mkdir -p "${NODE_OUT}"
{
  echo "epoch_ns=$(date +%s%N) tag=${TAG}"
  for ifdir in /sys/class/net/*; do
    ifname=$(basename "${ifdir}")
    [ "${ifname}" = "lo" ] && continue
    rx=$(cat "${ifdir}/statistics/rx_bytes" 2>/dev/null || echo 0)
    tx=$(cat "${ifdir}/statistics/tx_bytes" 2>/dev/null || echo 0)
    echo "iface=${ifname} rx_bytes=${rx} tx_bytes=${tx}"
  done
} >> "${NODE_OUT}/byte_counters.txt"
COUNTERS
chmod +x "${OUT}/counters.sh"

# 1 Hz sampler wrapper (local rank 0 per node).
cat > "${OUT}/sampled.sh" <<'SAMPLED'
#!/bin/bash
set -u
OUT="$1"; TAG="$2"; shift 2
NODE_OUT="${OUT}/node_$(hostname)"
mkdir -p "${NODE_OUT}"
SAMPLER_PID=""
if [ "${SLURM_LOCALID:-0}" = "0" ]; then
  (
    while true; do
      {
        echo "epoch_ns=$(date +%s%N) tag=${TAG}"
        for ifdir in /sys/class/net/hsn* /sys/class/net/nmn0; do
          [ -e "${ifdir}" ] || continue
          ifname=$(basename "${ifdir}")
          rx=$(cat "${ifdir}/statistics/rx_bytes" 2>/dev/null || echo 0)
          tx=$(cat "${ifdir}/statistics/tx_bytes" 2>/dev/null || echo 0)
          echo "iface=${ifname} rx_bytes=${rx} tx_bytes=${tx}"
        done
        echo "tcp_established=$(ss -tn state established 2>/dev/null | grep -c 172.30)"
      } >> "${NODE_OUT}/sampler_${TAG}.txt"
      sleep 1
    done
  ) &
  SAMPLER_PID=$!
fi
"$@"
RC=$?
[ -n "${SAMPLER_PID}" ] && kill "${SAMPLER_PID}" 2>/dev/null
exit "${RC}"
SAMPLED
chmod +x "${OUT}/sampled.sh"

module purge 2>/dev/null
module load "${CUDA_MODULE}" || fail "${CUDA_MODULE} module load failed"
nvcc --version > "${OUT}/nvcc_version.txt" 2>&1

[ -f "${NCCL_ROOT}/lib/libnccl.so" ] || fail "staged NCCL missing at ${NCCL_ROOT}"
cp "${STAGE}/fabric_flow_lane.cu" "${OUT}/" || fail "source copy failed"
sha256sum "${OUT}/fabric_flow_lane.cu" > "${OUT}/source.sha256"
sha256sum "${NCCL_ROOT}/lib/libnccl.so.2" >> "${OUT}/source.sha256"

log "building for ${ARCH_FLAG}"
nvcc -O3 -std=c++17 -arch="${ARCH_FLAG}" \
  -I"${NCCL_ROOT}/include" \
  -o "${OUT}/fabric_flow_lane" "${OUT}/fabric_flow_lane.cu" \
  -L"${NCCL_ROOT}/lib" -lnccl -Xlinker -rpath="${NCCL_ROOT}/lib" -lpthread \
  > "${OUT}/build.txt" 2>&1 || { cat "${OUT}/build.txt"; fail "build failed"; }
sha256sum "${OUT}/fabric_flow_lane" >> "${OUT}/source.sha256"
cat "${OUT}/source.sha256"

log "guard evidence before"
srun --ntasks="${SLURM_NTASKS}" "${OUT}/guards.sh" "${OUT}" before \
  || fail "guard collection failed"
if grep -h '^cassini_port_count=' "${OUT}"/rank_*/guards_before.txt \
    | grep -qv '^cassini_port_count=4$'; then
  fail "G1 violated: a node does not carry exactly four Cassini ports"
fi
if grep -h '^ib_device_count=' "${OUT}"/rank_*/guards_before.txt \
    | grep -qv '^ib_device_count=0$'; then
  fail "G1 violated: an InfiniBand device is present"
fi
for f in "${OUT}"/rank_*/guards_before.txt; do
  if [ "$(sed -n '/=== G4 foreign processes ===/,/=== G1 host NIC/p' "${f}" \
        | grep -cE '^[0-9]+,')" != "0" ]; then
    fail "G4 violated: a foreign compute process occupies an allocated GPU on ${f}"
  fi
done
rank_dirs=$(ls -d "${OUT}"/rank_*/ 2>/dev/null | grep -c .)
if [ "${rank_dirs}" != "${SLURM_NTASKS}" ]; then
  fail "G6 violated: ${rank_dirs} rank directories for ${SLURM_NTASKS} tasks"
fi
log "G1, G4 and G6 evidence collected"

srun --ntasks-per-node=1 "${OUT}/counters.sh" "${OUT}" before

SRC_ARG=""
[ -n "${SOURCES}" ] && SRC_ARG="--sources ${SOURCES}"

log "running the lane"
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING \
NCCL_DEBUG_FILE="${OUT}/nccl_debug.%h.%p.log" \
srun --ntasks="${SLURM_NTASKS}" \
  "${OUT}/sampled.sh" "${OUT}" run \
  "${OUT}/fabric_flow_lane" \
    --cell "${CELL}" \
    --out-prefix "${OUT}/${CELL}" \
    --id-dir "${OUT}" \
    --dest "${DEST_RANK}" \
    ${SRC_ARG} \
    --offsets "${OFFSETS}" \
    --window "${WINDOW_S}" \
    --chunk-bytes "${CHUNK_BYTES}" \
  > "${OUT}/run.txt" 2>&1
rc=$?
srun --ntasks-per-node=1 "${OUT}/counters.sh" "${OUT}" after
tail -20 "${OUT}/run.txt"
if [ "${rc}" -ne 0 ]; then
  log "lane exited ${rc}; keeping every log"
fi

grep -hoE 'Using network [A-Za-z]+|via NET/[A-Za-z0-9_/]+|GDR [0-9]' \
  "${OUT}"/nccl_debug.*.log 2>/dev/null | sort | uniq -c \
  > "${OUT}/transport_summary.txt"
cat "${OUT}/transport_summary.txt"
if ! grep -q 'Using network Socket' "${OUT}"/nccl_debug.*.log 2>/dev/null; then
  fail "G1 violated: NCCL did not select the Socket transport"
fi
if grep -q 'GDR 1' "${OUT}"/nccl_debug.*.log 2>/dev/null; then
  fail "G1 violated: GPUDirect RDMA is active"
fi

log "guard evidence after"
srun --ntasks="${SLURM_NTASKS}" "${OUT}/guards.sh" "${OUT}" after || true

log "cell ${CELL} complete under ${OUT}, rc ${rc}"
exit "${rc}"
