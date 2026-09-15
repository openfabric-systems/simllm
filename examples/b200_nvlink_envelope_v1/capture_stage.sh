#!/usr/bin/env bash
# Run one stage of the B200 NVLink envelope study inside a rented container.
#
#   capture_stage.sh <output dir> <stage>
#
# The caller cds to the repository checkout first, so every path here is
# relative to the repository root. The stage order is fixed by the freeze:
# inventory, RDMA evidence, NVLink precheck, then the timed lanes. A precheck
# that finds no active NVLink exits 3 before any timed lane, so an instance on
# a PCIe-only board costs the boot time only.
set -u
OUT="${1:?usage: capture_stage.sh <output dir> <stage>}"
STAGE="${2:?usage: capture_stage.sh <output dir> <stage>}"

case "$STAGE" in
  1|2) ;;
  *) echo "stage must be 1 or 2, got $STAGE" >&2; exit 2 ;;
esac

mkdir -p "$OUT"

# 1. The inventory the topology capture study records, unchanged and not
#    duplicated here. That script cds into its own output directory and then
#    reuses the path it was given, so resolve the directory first and hand it
#    an absolute one; the caller may pass a relative output directory.
mkdir -p "$OUT/inventory"
INVENTORY=$(cd "$OUT/inventory" && pwd)
bash examples/nccl_topology_capture_v1/capture_container.sh "$INVENTORY"

# 2. The RDMA evidence the freeze lists for the PLACE-6 NIC clause.
{
  echo "--- ls /sys/class/infiniband ---"; ls /sys/class/infiniband 2>&1
  echo "--- ibdev2netdev ---"
  if command -v ibdev2netdev > /dev/null 2>&1; then ibdev2netdev 2>&1; else echo "not present"; fi
  echo "--- ibv_devinfo ---"
  if command -v ibv_devinfo > /dev/null 2>&1; then ibv_devinfo 2>&1; else echo "not present"; fi
  echo "--- nvidia-smi topo -m ---"; nvidia-smi topo -m 2>&1
} > "$OUT/rdma_evidence.txt"

# 3. NVLink precheck. Every visible GPU must present its full set of active
#    links, and the topology matrix must show NV18 on every pair the stage
#    uses: all of them in stage 2, GPU 0 to GPU 1 in stage 1. A board with one
#    dead GPU passes a total link count and fails here, which is what a rental
#    that cannot run the study should do before it costs anything.
nvidia-smi nvlink -s > "$OUT/nvlink_status.txt" 2>&1
nvidia-smi topo -m > "$OUT/topo_matrix.txt" 2>&1
GPU_COUNT=$(nvidia-smi -L | grep -c "^GPU ")
LINKS_PER_GPU=18
ACTIVE_LINKS=$(grep -c "GB/s" "$OUT/nvlink_status.txt" || true)
LINK_BLOCKS=$(grep -c "^GPU [0-9]*:" "$OUT/nvlink_status.txt" || true)
SHORT_GPUS=$(awk -v expected="$LINKS_PER_GPU" '
  /^GPU [0-9]+:/ {
    if (seen && links != expected) printf "GPU%s(%d) ", gpu, links
    gpu = $2; sub(":", "", gpu); links = 0; seen = 1; next
  }
  /GB\/s/ { links++ }
  END { if (seen && links != expected) printf "GPU%s(%d) ", gpu, links }
' "$OUT/nvlink_status.txt")
BAD_PAIRS=$(awk -v n="$GPU_COUNT" -v stage="$STAGE" '
  $1 ~ /^GPU[0-9]+$/ {
    row = $1; sub("GPU", "", row); r = row + 0
    for (i = 1; i <= n; i++) {
      c = i - 1
      if (c == r) continue
      if (stage == 1 && !((r == 0 && c == 1) || (r == 1 && c == 0))) continue
      if ($(i + 1) != "NV18") printf "%d->%d=%s ", r, c, $(i + 1)
    }
  }
' "$OUT/topo_matrix.txt")

if [ "${GPU_COUNT:-0}" -lt 2 ]; then
  echo "NVLINK PRECHECK FAILED: $GPU_COUNT GPUs visible, the study needs at least 2" >&2
  exit 3
fi
if [ "${LINK_BLOCKS:-0}" -ne "$GPU_COUNT" ]; then
  echo "NVLINK PRECHECK FAILED: nvidia-smi nvlink -s reported $LINK_BLOCKS blocks for" \
    "$GPU_COUNT visible GPUs" >&2
  exit 3
fi
if [ -n "$SHORT_GPUS" ]; then
  echo "NVLINK PRECHECK FAILED: these GPUs do not report $LINKS_PER_GPU active links:" \
    "$SHORT_GPUS" >&2
  exit 3
fi
if [ -n "$BAD_PAIRS" ]; then
  echo "NVLINK PRECHECK FAILED: these GPU pairs are not NV18 in the topology matrix:" \
    "$BAD_PAIRS" >&2
  exit 3
fi
PAIR_01=$(awk '$1 == "GPU0" {print $3; exit}' "$OUT/topo_matrix.txt")
echo "nvlink precheck: $GPU_COUNT GPUs, $ACTIVE_LINKS active links," \
  "$LINKS_PER_GPU per GPU, every checked pair NV18, GPU0 to GPU1 is $PAIR_01"

# 4. The timed lanes, one process per visible GPU.
#
#    Three guards keep a hang from eating the rental cap. The process group
#    carries a timeout, so a stuck collective raises. The watchdog aborts the
#    process on that timeout without changing how a completed collective is
#    waited on, which blocking wait would, and which would add a host round
#    trip to every timed iteration of the small-payload window this study
#    scores. The hard wall clock is the only guard that also bounds a hang
#    inside communicator bootstrap, where no collective exists to time out.
RANKS=$(python -c "import torch; print(torch.cuda.device_count())")
LANE_WALL_CLOCK_SECONDS=900
PG_TIMEOUT_SECONDS=120
if [ "$STAGE" = "2" ]; then
  # Lane P1 walks 56 ordered pairs while seven ranks wait in one barrier.
  PG_TIMEOUT_SECONDS=300
fi

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,TUNING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
timeout --signal=TERM "$LANE_WALL_CLOCK_SECONDS" \
  python -m torch.distributed.run --standalone --nproc_per_node="$RANKS" \
  examples/b200_nvlink_envelope_v1/bench_nvlink.py \
  --stage "$STAGE" --output "$OUT" --pg-timeout-seconds "$PG_TIMEOUT_SECONDS" 2>&1 \
  | tee "$OUT/nccl_bench.log" \
  | grep --line-buffered -E "^(LANE|FATAL|wrote )" || true
BENCH_STATUS="${PIPESTATUS[0]}"
if [ "$BENCH_STATUS" = "124" ]; then
  echo "BENCH TIMED OUT after $LANE_WALL_CLOCK_SECONDS s; see nccl_bench.log" >&2
fi

# 5. The listing the orchestrator reads back. STAGE_DONE means the stage
#    script reached its end; bench_exit carries the lane's own status.
{
  ls -la "$OUT"
  ls -la "$OUT/inventory"
  echo "stage: $STAGE"
  echo "ranks: $RANKS"
  echo "gpus: $GPU_COUNT"
  echo "active_nvlinks: $ACTIVE_LINKS"
  echo "gpu0_gpu1: $PAIR_01"
  echo "pg_timeout_seconds: $PG_TIMEOUT_SECONDS"
  echo "lane_wall_clock_seconds: $LANE_WALL_CLOCK_SECONDS"
  echo "bench_exit: $BENCH_STATUS"
  echo "STAGE_DONE"
} > "$OUT/listing.txt" 2>&1

exit "$BENCH_STATUS"
