#!/bin/bash
# mx-pair: heterogeneous A100 plus GH200 capture cell body (freeze 1).
#
# Three phases in one het job, every phase with NCCL_SOCKET_IFNAME=hsn
# (the two architectures' management networks are not mutually routable,
# see the freeze's discovery section):
#   matrix     the three-payload pair probe (8 B, 128 KiB, 16 MiB)
#   mx-gh2a    60-second stream, destination on the A100 side
#   mx-a2gh    60-second stream, destination on the GH200 side
#
# Required environment: W18_STAGE_ROOT, W18_DATA_ROOT, W18_NCCL_ROOT
# (x86_64 wheel), W18_NCCL_ROOT_AARCH64 (aarch64 wheel).

set -u

STAGE="${W18_STAGE_ROOT:?set W18_STAGE_ROOT}"
OUT_ROOT="${W18_DATA_ROOT:?set W18_DATA_ROOT}"
NCCL_X86="${W18_NCCL_ROOT:?set W18_NCCL_ROOT}"
NCCL_ARM="${W18_NCCL_ROOT_AARCH64:?set W18_NCCL_ROOT_AARCH64}"
WINDOW_S="${W18_WINDOW_S:-60}"
CHUNK_BYTES="${W18_CHUNK_BYTES:-8388608}"

OUT="${OUT_ROOT}/capture/mx-pair/${SLURM_JOB_ID}"
mkdir -p "${OUT}" || exit 1

log() { echo "[mx-pair] $*"; }

log "het job ${SLURM_JOB_ID} at $(date -Is)"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "date_iso=$(date -Is)"
  echo "window_s=${WINDOW_S}"
  echo "chunk_bytes=${CHUNK_BYTES}"
  echo "nodelist_group0=${SLURM_JOB_NODELIST_HET_GROUP_0:-unset}"
  echo "nodelist_group1=${SLURM_JOB_NODELIST_HET_GROUP_1:-unset}"
} > "${OUT}/job_context.txt"
cat "${OUT}/job_context.txt"

cp "${STAGE}/fabric_flow_lane.cu" "${STAGE}/disco_lane.cu" "${OUT}/" || exit 1
sha256sum "${OUT}/fabric_flow_lane.cu" "${OUT}/disco_lane.cu" > "${OUT}/source.sha256"

cat > "${OUT}/mx_side.sh" <<'SIDE'
#!/bin/bash
set -u
OUT="$1"; ROLE="$2"; NCCL_ROOT="$3"; CUDA_MODULE="$4"; ARCH="$5"
WINDOW_S="$6"; CHUNK_BYTES="$7"
SIDE_OUT="${OUT}/${ROLE}_$(hostname)"
mkdir -p "${SIDE_OUT}"
exec > "${SIDE_OUT}/side.txt" 2>&1
echo "role=${ROLE} host=$(hostname) arch=$(uname -m) date=$(date -Is)"

echo "=== guard evidence before ==="
nvidia-smi -L
nvidia-smi --query-gpu=uuid,clocks.sm,clocks.mem --format=csv,noheader | sed 's/^/gpu=/'
nvidia-smi --query-compute-apps=pid,process_name --format=csv
echo "cassini_port_count=$(lspci 2>/dev/null | grep -icE 'cassini')"
echo "ib_device_count=$(ls /sys/class/infiniband 2>/dev/null | grep -c . || echo 0)"

snapshot() {
  echo "counters tag=$1 epoch_ns=$(date +%s%N)"
  for ifdir in /sys/class/net/hsn*; do
    [ -e "${ifdir}" ] || continue
    echo "iface=$(basename "${ifdir}") rx_bytes=$(cat "${ifdir}/statistics/rx_bytes") tx_bytes=$(cat "${ifdir}/statistics/tx_bytes")"
  done
}

module purge 2>/dev/null
module load "${CUDA_MODULE}" || { echo "FATAL module ${CUDA_MODULE}"; exit 1; }
for src in fabric_flow_lane disco_lane; do
  nvcc -O3 -std=c++17 -arch="${ARCH}" \
    -I"${NCCL_ROOT}/include" \
    -o "${SIDE_OUT}/${src}" "${OUT}/${src}.cu" \
    -L"${NCCL_ROOT}/lib" -lnccl -Xlinker -rpath="${NCCL_ROOT}/lib" -lpthread \
    || { echo "FATAL build ${src} failed"; exit 1; }
done
echo "builds ok"
sha256sum "${SIDE_OUT}/fabric_flow_lane" "${SIDE_OUT}/disco_lane"

export NCCL_SOCKET_IFNAME=hsn
export CAPT_WORLD=2 DISCO_WORLD=2
export CAPT_LOCALID=0 DISCO_LOCALID=0
export CAPT_ID_POLL_SECS=600 DISCO_ID_POLL_SECS=600

rc_total=0

# Phase 1: matrix probe (rank 0 on the A100 side).
if [ "${ROLE}" = "a100" ]; then export DISCO_RANK=0; else export DISCO_RANK=1; fi
snapshot before_matrix
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING \
NCCL_DEBUG_FILE="${SIDE_OUT}/nccl_debug_matrix.%h.%p.log" \
"${SIDE_OUT}/disco_lane" --mode matrix \
  --out "${OUT}/mx_matrix.json" --id-path "${OUT}/mx_id_matrix.bin" \
  || rc_total=$?
snapshot after_matrix
echo "phase matrix rc=${rc_total}"

# Phase 2: stream with the destination on the A100 side (GH sources).
if [ "${ROLE}" = "a100" ]; then export CAPT_RANK=0; else export CAPT_RANK=1; fi
snapshot before_gh2a
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING \
NCCL_DEBUG_FILE="${SIDE_OUT}/nccl_debug_gh2a.%h.%p.log" \
"${SIDE_OUT}/fabric_flow_lane" --cell mx-gh2a \
  --out-prefix "${OUT}/mx-gh2a" --id-dir "${OUT}/id_gh2a" \
  --dest 0 --sources 1 --offsets 0 \
  --window "${WINDOW_S}" --chunk-bytes "${CHUNK_BYTES}" \
  || rc_total=$?
snapshot after_gh2a
echo "phase gh2a rc=${rc_total}"

# Phase 3: stream with the destination on the GH side (A100 sources).
if [ "${ROLE}" = "a100" ]; then export CAPT_RANK=1; else export CAPT_RANK=0; fi
snapshot before_a2gh
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING \
NCCL_DEBUG_FILE="${SIDE_OUT}/nccl_debug_a2gh.%h.%p.log" \
"${SIDE_OUT}/fabric_flow_lane" --cell mx-a2gh \
  --out-prefix "${OUT}/mx-a2gh" --id-dir "${OUT}/id_a2gh" \
  --dest 0 --sources 1 --offsets 0 \
  --window "${WINDOW_S}" --chunk-bytes "${CHUNK_BYTES}" \
  || rc_total=$?
snapshot after_a2gh
echo "phase a2gh rc=${rc_total}"

echo "=== guard evidence after ==="
nvidia-smi --query-gpu=uuid,clocks.sm,clocks.mem --format=csv,noheader | sed 's/^/gpu=/'
echo "=== transport summary ==="
grep -hoE 'Using network [A-Za-z]+|via NET/[A-Za-z0-9_/]+|GDR [0-9]' \
  "${SIDE_OUT}"/nccl_debug_*.log 2>/dev/null | sort | uniq -c
if ! grep -q 'Using network Socket' "${SIDE_OUT}"/nccl_debug_*.log 2>/dev/null; then
  echo "FATAL G1: NCCL did not select the Socket transport"; rc_total=1
fi
if grep -q 'GDR 1' "${SIDE_OUT}"/nccl_debug_*.log 2>/dev/null; then
  echo "FATAL G1: GPUDirect RDMA is active"; rc_total=1
fi
exit "${rc_total}"
SIDE
chmod +x "${OUT}/mx_side.sh"
mkdir -p "${OUT}/id_gh2a" "${OUT}/id_a2gh"

log "launching both sides"
srun --het-group=0 --ntasks=1 \
  "${OUT}/mx_side.sh" "${OUT}" a100 "${NCCL_X86}" cuda/12.2.2 sm_80 \
  "${WINDOW_S}" "${CHUNK_BYTES}" \
  > "${OUT}/srun_a100.txt" 2>&1 &
PID_A=$!
srun --het-group=1 --ntasks=1 \
  "${OUT}/mx_side.sh" "${OUT}" gh200 "${NCCL_ARM}" cuda/12.9.1 sm_90 \
  "${WINDOW_S}" "${CHUNK_BYTES}" \
  > "${OUT}/srun_gh200.txt" 2>&1 &
PID_G=$!
wait "${PID_A}"; RC_A=$?
wait "${PID_G}"; RC_G=$?
log "a100 side rc=${RC_A} gh200 side rc=${RC_G}"
tail -30 "${OUT}"/*/side.txt 2>/dev/null
exit $((RC_A + RC_G))
