#!/bin/bash
# Shared body of every cross-node collective envelope job.
#
# Captures the fatal-guard evidence, builds the lane, runs it once per declared
# interface arm, and summarizes the transport NCCL selected. Called from the
# thin per-cell sbatch files, which own only the resource request.
#
# Required environment, with no personal-path defaults anywhere:
#   XNODE_STAGE_ROOT   directory holding this study's staged sources
#   XNODE_DATA_ROOT    directory the bulk outputs are written under
#   XNODE_NCCL_ROOT    root of the staged NCCL wheel (lib/ and include/)
#   XNODE_CELL         cell name, e.g. w2, w8, w4
#   XNODE_ARMS         space-separated interface arms: default, fournic

set -u

STAGE="${XNODE_STAGE_ROOT:?set XNODE_STAGE_ROOT}"
OUT_ROOT="${XNODE_DATA_ROOT:?set XNODE_DATA_ROOT}"
NCCL_ROOT="${XNODE_NCCL_ROOT:?set XNODE_NCCL_ROOT}"
CELL="${XNODE_CELL:?set XNODE_CELL}"
ARMS="${XNODE_ARMS:?set XNODE_ARMS}"

OUT="${OUT_ROOT}/${CELL}/${SLURM_JOB_ID}"
mkdir -p "${OUT}" || exit 1

log() { echo "[${CELL}] $*"; }
fail() { echo "[${CELL}] FATAL $*"; exit 1; }

log "job ${SLURM_JOB_ID} on $(hostname) at $(date -Is)"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "cell=${CELL}"
  echo "arms=${ARMS}"
  echo "date_iso=$(date -Is)"
  echo "partition=${SLURM_JOB_PARTITION}"
  echo "nodelist=${SLURM_JOB_NODELIST}"
  echo "nnodes=${SLURM_JOB_NUM_NODES}"
  echo "ntasks=${SLURM_NTASKS}"
  echo "ntasks_per_node=${SLURM_NTASKS_PER_NODE:-unset}"
  echo "nccl_root=${NCCL_ROOT}"
} > "${OUT}/job_context.txt"
cat "${OUT}/job_context.txt"

# ---------------------------------------------------------------------------
# Fatal-guard evidence, collected per node before anything is timed.
# ---------------------------------------------------------------------------
# One directory per rank, not per node: at width 8 there are four tasks on each
# node and a per-node path would have them clobber one file.
cat > "${OUT}/guards.sh" <<'GUARDS'
#!/bin/bash
set -u
OUT="$1"
WHEN="$2"
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
lspci 2>/dev/null | grep -iE 'cassini|mellanox|infiniband' || echo "(no high-speed NIC matched)"
echo "=== G1 sysfs infiniband class ==="
ls /sys/class/infiniband 2>/dev/null | sed 's/^/ib_device=/' || true
echo "ib_device_count=$(ls /sys/class/infiniband 2>/dev/null | grep -c . || echo 0)"
echo "=== G1 interfaces ==="
ip -br link 2>/dev/null | awk '{print $1, $2}'
GUARDS
chmod +x "${OUT}/guards.sh"

module purge 2>/dev/null
module load cuda/12.2.2 || fail "cuda/12.2.2 module load failed"
nvcc --version > "${OUT}/nvcc_version.txt" 2>&1

[ -f "${NCCL_ROOT}/lib/libnccl.so" ] || fail "staged NCCL missing at ${NCCL_ROOT}"
cp "${STAGE}/crossnode_lane.cu" "${OUT}/" || fail "source copy failed"
sha256sum "${OUT}/crossnode_lane.cu" > "${OUT}/source.sha256"
sha256sum "${NCCL_ROOT}/lib/libnccl.so.2" >> "${OUT}/source.sha256"

log "building"
nvcc -O3 -std=c++17 -arch=sm_80 \
  -I"${NCCL_ROOT}/include" \
  -o "${OUT}/crossnode_lane" "${OUT}/crossnode_lane.cu" \
  -L"${NCCL_ROOT}/lib" -lnccl -Xlinker -rpath="${NCCL_ROOT}/lib" -lpthread \
  > "${OUT}/build.txt" 2>&1 || { cat "${OUT}/build.txt"; fail "build failed"; }
sha256sum "${OUT}/crossnode_lane" >> "${OUT}/source.sha256"
cat "${OUT}/source.sha256"

log "collecting guard evidence before"
srun --ntasks="${SLURM_NTASKS}" "${OUT}/guards.sh" "${OUT}" before \
  || fail "guard collection failed"
if grep -h '^cassini_port_count=' "${OUT}"/rank_*/guards_before.txt \
    | grep -qv '^cassini_port_count=4$'; then
  grep -h '^cassini_port_count=' "${OUT}"/rank_*/guards_before.txt
  fail "G1 violated: a node does not carry exactly four Cassini ports"
fi
if grep -h '^ib_device_count=' "${OUT}"/rank_*/guards_before.txt \
    | grep -qv '^ib_device_count=0$'; then
  grep -h '^ib_device_count=' "${OUT}"/rank_*/guards_before.txt
  fail "G1 violated: an InfiniBand device is present"
fi
for f in "${OUT}"/rank_*/guards_before.txt; do
  if [ "$(sed -n '/=== G4 foreign processes ===/,/=== G1 host NIC/p' "${f}" \
        | grep -cE '^[0-9]+,')" != "0" ]; then
    fail "G4 violated: a foreign compute process occupies an allocated GPU on ${f}"
  fi
done
# G6: the ranks must occupy as many distinct physical GPUs as there are ranks.
# One rank per line of gpu_uuid= output would let two ranks share a card, and
# then every collective time would be measuring the wrong machine.
rank_dirs=$(ls -d "${OUT}"/rank_*/ 2>/dev/null | grep -c .)
distinct_gpus=$(grep -h '^gpu_uuid=' "${OUT}"/rank_*/guards_before.txt \
  | sort -u | grep -c .)
echo "rank_dirs=${rank_dirs} distinct_gpu_uuids=${distinct_gpus} ntasks=${SLURM_NTASKS}"
if [ "${rank_dirs}" != "${SLURM_NTASKS}" ]; then
  fail "G6 violated: ${rank_dirs} rank directories for ${SLURM_NTASKS} tasks"
fi
if [ "${distinct_gpus}" -lt "${SLURM_NTASKS}" ]; then
  fail "G6 violated: only ${distinct_gpus} distinct GPUs across ${SLURM_NTASKS} ranks"
fi
log "G1, G4 and G6 evidence collected"

# ---------------------------------------------------------------------------
# One run per declared interface arm.
# ---------------------------------------------------------------------------
rc_total=0
for arm in ${ARMS}; do
  ARM_OUT="${OUT}/${arm}"
  mkdir -p "${ARM_OUT}"
  ID_PATH="${ARM_OUT}/nccl_unique_id.bin"

  unset NCCL_SOCKET_IFNAME NCCL_SOCKET_NTHREADS NCCL_NSOCKS_PERTHREAD
  if [ "${arm}" = "fournic" ]; then
    export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_NSOCKS_PERTHREAD=2
  fi

  log "running arm ${arm}"
  # INIT, NET and GRAPH logging fires only while a communicator is being built,
  # so it records the transport decision without touching a timed loop.
  NCCL_DEBUG=INFO \
  NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING \
  NCCL_DEBUG_FILE="${ARM_OUT}/nccl_debug.%h.%p.log" \
  srun --ntasks="${SLURM_NTASKS}" \
    "${OUT}/crossnode_lane" \
      --out "${ARM_OUT}/crossnode_${CELL}_${arm}.json" \
      --id-path "${ID_PATH}" \
      --arm "${arm}" \
    > "${ARM_OUT}/run.txt" 2>&1
  rc=$?
  tail -20 "${ARM_OUT}/run.txt"
  if [ "${rc}" -ne 0 ]; then
    log "arm ${arm} exited ${rc}; keeping every log"
    rc_total=${rc}
    continue
  fi

  grep -hoE 'Using network [A-Za-z]+|via NET/[A-Za-z0-9_/]+|via P2P/[a-z]+|GDR [0-9]' \
    "${ARM_OUT}"/nccl_debug.*.log 2>/dev/null | sort | uniq -c \
    > "${ARM_OUT}/transport_summary.txt"
  cat "${ARM_OUT}/transport_summary.txt"
  if ! grep -q 'Using network Socket' "${ARM_OUT}"/nccl_debug.*.log 2>/dev/null; then
    fail "G1 violated in arm ${arm}: NCCL did not select the Socket transport"
  fi
  if grep -q 'GDR 1' "${ARM_OUT}"/nccl_debug.*.log 2>/dev/null; then
    fail "G1 violated in arm ${arm}: GPUDirect RDMA is active"
  fi
  log "arm ${arm} result bytes: $(stat -c %s "${ARM_OUT}/crossnode_${CELL}_${arm}.json")"
done

log "collecting guard evidence after"
srun --ntasks="${SLURM_NTASKS}" "${OUT}/guards.sh" "${OUT}" after || true

log "cell ${CELL} complete, output under ${OUT}"
exit "${rc_total}"
