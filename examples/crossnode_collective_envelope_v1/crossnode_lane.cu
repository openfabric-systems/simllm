// Cross-node collective envelope: one multi-process lane over two or more
// nodes.
//
// One process per rank, launched by the site scheduler, with ranks taken from
// the scheduler's own environment and the NCCL unique id shared through a file
// on the shared filesystem. There is no MPI dependency.
//
// Measures, at the frozen payload grid:
//   * a cross-node point-to-point send and receive ramp, at width 2 only,
//   * ring all-reduce,
//   * pairwise all-to-allv with a uniform per-pair payload.
//
// Every timed block is bracketed by CUDA events on the measured stream. The
// primary reading is the same back-to-back loop the intra-node A100 and GH200
// lanes used and that nccl-tests uses, so the numbers are comparable to the
// intra-node floors and to the capture the shipped B200 profile came from. A
// secondary isolated reading is taken for the latency-floor payloads only and
// is a diagnostic, never a profile input.
//
// The reported value for every cell is the maximum over ranks. It is reduced
// once, by a single ncclAllReduce with ncclMax over the whole result vector,
// after every timed block has completed, so the reduction cannot perturb any
// measurement.
//
// The frozen sweep, bounds, guards and relations live in expectations.md.

#include <cuda_runtime.h>
#include <nccl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err_ = (expr);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "FATAL cuda %s at %s:%d: %s\n", #expr, __FILE__,   \
                   __LINE__, cudaGetErrorString(err_));                       \
      std::exit(20);                                                          \
    }                                                                         \
  } while (0)

#define NCCL_CHECK(expr)                                                      \
  do {                                                                        \
    ncclResult_t res_ = (expr);                                               \
    if (res_ != ncclSuccess) {                                                \
      std::fprintf(stderr, "FATAL nccl %s at %s:%d: %s\n", #expr, __FILE__,   \
                   __LINE__, ncclGetErrorString(res_));                       \
      std::exit(21);                                                          \
    }                                                                         \
  } while (0)

struct Json {
  std::string body;
  void raw(const std::string& text) { body += text; }
  void key(const char* name) {
    if (!body.empty() && body.back() != '{' && body.back() != '[') body += ",";
    body += "\"";
    body += name;
    body += "\":";
  }
  void num(const char* name, double value) {
    key(name);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10g", value);
    body += buf;
  }
  void integer(const char* name, long long value) {
    key(name);
    body += std::to_string(value);
  }
  void str(const char* name, const std::string& value) {
    key(name);
    body += "\"" + value + "\"";
  }
  void open_arr(const char* name) {
    key(name);
    body += "[";
  }
  void close_obj() { body += "}"; }
  void close_arr() { body += "]"; }
  void item_sep() {
    if (!body.empty() && body.back() != '[') body += ",";
  }
};

__global__ void k_fill_float(float* p, size_t n, float value) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) p[i] = value;
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  return (n % 2 == 1) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::atoi(raw);
}

std::string env_str(const char* name) {
  const char* raw = std::getenv(name);
  return (raw == nullptr) ? std::string("unset") : std::string(raw);
}

// Rank 0 writes the unique id to a temporary path and renames it, so a reader
// never observes a partially written id. Every other rank polls for it.
void share_unique_id(ncclUniqueId* id, int rank, const std::string& path) {
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(id));
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "FATAL cannot write %s\n", tmp.c_str());
      std::exit(22);
    }
    if (std::fwrite(id, sizeof(*id), 1, f) != 1) {
      std::fprintf(stderr, "FATAL short write of the unique id\n");
      std::exit(22);
    }
    std::fclose(f);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
      std::fprintf(stderr, "FATAL cannot rename %s\n", tmp.c_str());
      std::exit(22);
    }
    return;
  }
  for (int attempt = 0; attempt < 900; ++attempt) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f != nullptr) {
      const size_t got = std::fread(id, sizeof(*id), 1, f);
      std::fclose(f);
      if (got == 1) return;
    }
    usleep(200000);
  }
  std::fprintf(stderr, "FATAL never saw the unique id at %s\n", path.c_str());
  std::exit(23);
}

// The frozen payload grid, in bytes.
const size_t kPayloads[] = {
    8,        64,       512,      4096,     16384,    65536,
    131072,   196608,   262144,   393216,   524288,   786432,
    1048576,  1572864,  2097152,  3145728,  4194304,
    8388608,  16777216, 33554432, 67108864, 134217728,
};
constexpr int kPayloadCount = static_cast<int>(sizeof(kPayloads) / sizeof(kPayloads[0]));

// The frozen isolated-reading cut: payloads at or below 64 KiB.
constexpr size_t kIsolatedMaxBytes = 65536;
constexpr int kIsolatedSamples = 20;

constexpr int kWarmup = 5;
constexpr int kBlockThreads = 256;
constexpr int kGridBlocks = 864;

int timed_iters(size_t bytes) {
  if (bytes <= (8ull << 20)) return 20;
  if (bytes <= (64ull << 20)) return 10;
  return 5;
}

enum class Op { PointToPoint, AllReduce, AllToAllv };

const char* op_name(Op op) {
  switch (op) {
    case Op::PointToPoint: return "p2p";
    case Op::AllReduce: return "allreduce";
    case Op::AllToAllv: return "alltoallv";
  }
  return "unknown";
}

// One recorded cell, before the cross-rank maximum is taken.
struct Cell {
  Op op;
  size_t bytes;
  double back_to_back_us;
  double isolated_us;  // negative when the frozen grid does not take one
  int iters;
  int mismatches;
};

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "crossnode_result.json";
  std::string id_path = "nccl_unique_id.bin";
  std::string arm = "default";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (std::strcmp(argv[i], "--id-path") == 0 && i + 1 < argc) {
      id_path = argv[++i];
    } else if (std::strcmp(argv[i], "--arm") == 0 && i + 1 < argc) {
      arm = argv[++i];
    }
  }

  const int rank = env_int("SLURM_PROCID", 0);
  const int world = env_int("SLURM_NTASKS", 1);
  const int local_rank = env_int("SLURM_LOCALID", 0);
  const int local_size = env_int("SLURM_NTASKS_PER_NODE", 1);
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);

  if (world < 2) {
    std::fprintf(stderr, "FATAL this lane needs at least two ranks\n");
    return 24;
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    std::fprintf(stderr, "FATAL rank %d sees no CUDA device\n", rank);
    return 25;
  }
  const int device = local_rank % device_count;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  char pci[64] = {0};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
  int nccl_version = 0;
  NCCL_CHECK(ncclGetVersion(&nccl_version));

  std::printf("[lane] rank %d of %d on %s local %d device %d (%s, %s) nccl %d arm %s\n",
              rank, world, host, local_rank, device, prop.name, pci,
              nccl_version, arm.c_str());
  std::fflush(stdout);

  ncclUniqueId id;
  std::memset(&id, 0, sizeof(id));
  share_unique_id(&id, rank, id_path);

  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  const size_t max_bytes = kPayloads[kPayloadCount - 1];

  // Barrier buffer, used only outside timed brackets.
  float* barrier_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&barrier_buf, sizeof(float)));
  CUDA_CHECK(cudaMemset(barrier_buf, 0, sizeof(float)));
  auto barrier = [&]() {
    NCCL_CHECK(ncclAllReduce(barrier_buf, barrier_buf, 1, ncclFloat, ncclSum,
                             comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  // All-reduce buffers.
  float* ar_send = nullptr;
  float* ar_recv = nullptr;
  CUDA_CHECK(cudaMalloc(&ar_send, max_bytes));
  CUDA_CHECK(cudaMalloc(&ar_recv, max_bytes));
  k_fill_float<<<kGridBlocks, kBlockThreads>>>(ar_send, max_bytes / sizeof(float), 1.0f);
  CUDA_CHECK(cudaMemset(ar_recv, 0, max_bytes));

  // Point-to-point buffers, width 2 only.
  float* p2p_send = nullptr;
  float* p2p_recv = nullptr;
  if (world == 2) {
    CUDA_CHECK(cudaMalloc(&p2p_send, max_bytes));
    CUDA_CHECK(cudaMalloc(&p2p_recv, max_bytes));
    k_fill_float<<<kGridBlocks, kBlockThreads>>>(p2p_send, max_bytes / sizeof(float), 3.5f);
    CUDA_CHECK(cudaMemset(p2p_recv, 0, max_bytes));
  }

  // All-to-allv buffers: one block per peer, self included so the indexing is
  // uniform, with the self block never posted.
  float* a2a_send = nullptr;
  float* a2a_recv = nullptr;
  CUDA_CHECK(cudaMalloc(&a2a_send, max_bytes * static_cast<size_t>(world)));
  CUDA_CHECK(cudaMalloc(&a2a_recv, max_bytes * static_cast<size_t>(world)));
  CUDA_CHECK(cudaMemset(a2a_recv, 0, max_bytes * static_cast<size_t>(world)));
  // The block this rank sends to peer p is filled with rank * world + p + 1,
  // so a receiver can check the identity of every source block exactly. The
  // largest value is world * world, exact in float32 for any world here.
  for (int peer = 0; peer < world; ++peer) {
    k_fill_float<<<kGridBlocks, kBlockThreads>>>(
        a2a_send + static_cast<size_t>(peer) * (max_bytes / sizeof(float)),
        max_bytes / sizeof(float),
        static_cast<float>(rank * world + peer + 1));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::vector<Cell> cells;

  auto issue_p2p = [&](size_t elems) {
    NCCL_CHECK(ncclGroupStart());
    if (rank == 0) {
      NCCL_CHECK(ncclSend(p2p_send, elems, ncclFloat, 1, comm, stream));
    } else {
      NCCL_CHECK(ncclRecv(p2p_recv, elems, ncclFloat, 0, comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
  };

  auto issue_allreduce = [&](size_t elems) {
    NCCL_CHECK(ncclAllReduce(ar_send, ar_recv, elems, ncclFloat, ncclSum, comm,
                             stream));
  };

  auto issue_a2a = [&](size_t elems) {
    const size_t stride = max_bytes / sizeof(float);
    NCCL_CHECK(ncclGroupStart());
    for (int peer = 0; peer < world; ++peer) {
      if (peer == rank) continue;
      NCCL_CHECK(ncclSend(a2a_send + static_cast<size_t>(peer) * stride, elems,
                          ncclFloat, peer, comm, stream));
      NCCL_CHECK(ncclRecv(a2a_recv + static_cast<size_t>(peer) * stride, elems,
                          ncclFloat, peer, comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
  };

  std::vector<Op> ops;
  if (world == 2) ops.push_back(Op::PointToPoint);
  ops.push_back(Op::AllReduce);
  ops.push_back(Op::AllToAllv);

  for (Op op : ops) {
    for (int p = 0; p < kPayloadCount; ++p) {
      const size_t bytes = kPayloads[p];
      const size_t elems = bytes / sizeof(float);
      if (elems == 0) continue;

      auto issue = [&]() {
        switch (op) {
          case Op::PointToPoint: issue_p2p(elems); break;
          case Op::AllReduce: issue_allreduce(elems); break;
          case Op::AllToAllv: issue_a2a(elems); break;
        }
      };

      const int iters = timed_iters(bytes);

      for (int it = 0; it < kWarmup; ++it) issue();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      barrier();

      CUDA_CHECK(cudaEventRecord(e0, stream));
      for (int it = 0; it < iters; ++it) issue();
      CUDA_CHECK(cudaEventRecord(e1, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      const double back_to_back_us = static_cast<double>(ms) * 1000.0 / iters;

      double isolated_us = -1.0;
      if (bytes <= kIsolatedMaxBytes) {
        std::vector<double> samples;
        samples.reserve(kIsolatedSamples);
        for (int s = 0; s < kIsolatedSamples; ++s) {
          barrier();
          CUDA_CHECK(cudaEventRecord(e0, stream));
          issue();
          CUDA_CHECK(cudaEventRecord(e1, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          float one = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&one, e0, e1));
          samples.push_back(static_cast<double>(one) * 1000.0);
        }
        isolated_us = median(samples);
      }

      // Value conservation, checked after the timed blocks so the check never
      // sits inside a bracket. Three probes per source block.
      int mismatches = 0;
      const size_t probes[3] = {0, elems / 2, elems - 1};
      if (op == Op::AllReduce) {
        const float expected = static_cast<float>(world);
        for (int q = 0; q < 3; ++q) {
          float got = 0.f;
          CUDA_CHECK(cudaMemcpy(&got, ar_recv + probes[q], sizeof(float),
                                cudaMemcpyDeviceToHost));
          if (got != expected) ++mismatches;
        }
      } else if (op == Op::AllToAllv) {
        const size_t stride = max_bytes / sizeof(float);
        for (int source = 0; source < world; ++source) {
          if (source == rank) continue;
          const float expected = static_cast<float>(source * world + rank + 1);
          for (int q = 0; q < 3; ++q) {
            float got = 0.f;
            CUDA_CHECK(cudaMemcpy(&got,
                                  a2a_recv + static_cast<size_t>(source) * stride +
                                      probes[q],
                                  sizeof(float), cudaMemcpyDeviceToHost));
            if (got != expected) ++mismatches;
          }
        }
      } else if (op == Op::PointToPoint && rank == 1) {
        for (int q = 0; q < 3; ++q) {
          float got = 0.f;
          CUDA_CHECK(cudaMemcpy(&got, p2p_recv + probes[q], sizeof(float),
                                cudaMemcpyDeviceToHost));
          if (got != 3.5f) ++mismatches;
        }
      }

      cells.push_back(Cell{op, bytes, back_to_back_us, isolated_us, iters,
                           mismatches});
      barrier();
    }
  }

  // One reduction over the whole result vector, after every timed block. Three
  // doubles per cell: back-to-back, isolated, mismatch count. Taking the
  // maximum of the mismatch count means one bad rank cannot hide behind a
  // clean one.
  const size_t reduce_count = cells.size() * 3;
  std::vector<double> local(reduce_count, 0.0);
  for (size_t c = 0; c < cells.size(); ++c) {
    local[c * 3 + 0] = cells[c].back_to_back_us;
    local[c * 3 + 1] = cells[c].isolated_us;
    local[c * 3 + 2] = static_cast<double>(cells[c].mismatches);
  }
  double* dev_reduce = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_reduce, reduce_count * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dev_reduce, local.data(), reduce_count * sizeof(double),
                        cudaMemcpyHostToDevice));
  NCCL_CHECK(ncclAllReduce(dev_reduce, dev_reduce, reduce_count, ncclDouble,
                           ncclMax, comm, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<double> reduced(reduce_count, 0.0);
  CUDA_CHECK(cudaMemcpy(reduced.data(), dev_reduce,
                        reduce_count * sizeof(double), cudaMemcpyDeviceToHost));

  if (rank == 0) {
    Json j;
    j.raw("{");
    j.str("study", "crossnode_collective_envelope_v1");
    j.str("arm", arm);
    j.integer("world", world);
    j.integer("tasks_per_node", local_size);
    j.integer("visible_device_count", device_count);
    j.integer("nccl_version", nccl_version);
    j.str("rank0_host", host);
    j.str("rank0_device_name", prop.name);
    j.str("rank0_pci_bus_id", pci);
    j.integer("warmup_iters", kWarmup);
    j.integer("isolated_samples", kIsolatedSamples);
    j.integer("isolated_max_bytes", static_cast<long long>(kIsolatedMaxBytes));
    j.str("env_nccl_socket_ifname", env_str("NCCL_SOCKET_IFNAME"));
    j.str("env_nccl_socket_nthreads", env_str("NCCL_SOCKET_NTHREADS"));
    j.str("env_nccl_nsocks_perthread", env_str("NCCL_NSOCKS_PERTHREAD"));
    j.str("env_slurm_job_id", env_str("SLURM_JOB_ID"));
    j.str("env_slurm_nodelist", env_str("SLURM_JOB_NODELIST"));

    j.open_arr("cells");
    for (size_t c = 0; c < cells.size(); ++c) {
      const double bb = reduced[c * 3 + 0];
      const double iso = reduced[c * 3 + 1];
      const double mism = reduced[c * 3 + 2];
      const double seconds = bb * 1e-6;
      j.item_sep();
      j.raw("{");
      j.str("op", op_name(cells[c].op));
      j.integer("width", world);
      j.integer("bytes", static_cast<long long>(cells[c].bytes));
      j.integer("timed_iters", cells[c].iters);
      j.num("time_us", bb);
      j.num("rank0_time_us", cells[c].back_to_back_us);
      if (iso >= 0.0) {
        j.num("isolated_time_us", iso);
        j.num("rank0_isolated_time_us", cells[c].isolated_us);
      }
      // Algorithm bandwidth is payload over time in every case. The bus
      // bandwidth factor is the nccl-tests one for a ring all-reduce; for the
      // point-to-point ramp and the all-to-allv the meaningful derived rate is
      // reported instead of a bus bandwidth that would not mean the same thing.
      j.num("algbw_gbps", static_cast<double>(cells[c].bytes) / seconds / 1e9);
      if (cells[c].op == Op::AllReduce) {
        const double factor = 2.0 * (world - 1) / world;
        j.num("busbw_factor", factor);
        j.num("busbw_gbps",
              static_cast<double>(cells[c].bytes) / seconds / 1e9 * factor);
      }
      if (cells[c].op == Op::AllToAllv) {
        j.num("per_rank_egress_bytes",
              static_cast<double>(cells[c].bytes) * (world - 1));
        j.num("per_rank_egress_gbps",
              static_cast<double>(cells[c].bytes) * (world - 1) / seconds / 1e9);
      }
      j.num("max_rank_mismatches", mism);
      j.close_obj();
    }
    j.close_arr();
    j.close_obj();
    j.raw("\n");

    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (f == nullptr) {
      std::fprintf(stderr, "FATAL cannot write %s\n", out_path.c_str());
      return 26;
    }
    std::fwrite(j.body.data(), 1, j.body.size(), f);
    std::fclose(f);
    std::printf("[lane] wrote %zu bytes to %s over %zu cells\n", j.body.size(),
                out_path.c_str(), cells.size());
  }

  // Report any conservation failure loudly on every rank, and fail the job.
  double worst_mismatch = 0.0;
  for (size_t c = 0; c < cells.size(); ++c) {
    worst_mismatch = std::max(worst_mismatch, reduced[c * 3 + 2]);
  }

  CUDA_CHECK(cudaFree(dev_reduce));
  CUDA_CHECK(cudaFree(a2a_send));
  CUDA_CHECK(cudaFree(a2a_recv));
  if (p2p_send != nullptr) CUDA_CHECK(cudaFree(p2p_send));
  if (p2p_recv != nullptr) CUDA_CHECK(cudaFree(p2p_recv));
  CUDA_CHECK(cudaFree(ar_send));
  CUDA_CHECK(cudaFree(ar_recv));
  CUDA_CHECK(cudaFree(barrier_buf));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(comm));

  if (worst_mismatch != 0.0) {
    std::fprintf(stderr,
                 "FATAL guard G3 violated: worst per-cell mismatch count is %g\n",
                 worst_mismatch);
    return 27;
  }
  std::printf("[lane] rank %d clean\n", rank);
  return 0;
}
