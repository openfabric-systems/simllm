// Wave-18 fabric flow capture: DISCOVERY lane. UNSCORED.
//
// Two modes, both explicitly unscored discovery for the wave-18 study
// (examples/merlin_fabric_flow_capture_v1). No relation is scored against
// anything this program prints. Its outputs inform the freeze; the freeze
// precedes every scored run.
//
//   --mode matrix   all-pairs point-to-point probe over the world
//                   communicator: 8 B ping-pong RTT (back-to-back and
//                   isolated), recv-side one-way completion, and a
//                   128 KiB and 16 MiB per-pair bandwidth point.
//   --mode jitter   fixed-chunk repetition ladder on a two-rank flow,
//                   using the exact per-chunk loop the capture harness
//                   will use (recv, stream sync, CPU timestamp, 4-byte
//                   probe readback, check), plus two tracer-floor
//                   blocks with no network in them.
//
// Chunk-size decision rule, stated BEFORE this program first runs: the
// capture chunk size will be frozen as the smallest ladder size whose
// per-chunk spread, measured as p95 minus p5 over the repetitions at that
// size on the destination rank, is below 1 percent of the median per-chunk
// time at that size. The spread and the median are both reported raw here
// and the choice is recorded in the freeze, not in this program.
//
// Rank bootstrap is the wave-16 pattern: ncclUniqueId through a file on the
// shared filesystem, ranks from Slurm, no MPI. DISCO_RANK / DISCO_WORLD /
// DISCO_LOCALID environment overrides exist so a heterogeneous or
// two-job-rendezvous cell can assign ranks explicitly.

#include <cuda_runtime.h>
#include <nccl.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
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

long long now_ns(clockid_t clk) {
  timespec ts{};
  clock_gettime(clk, &ts);
  return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

__global__ void k_write_probe(float* p, size_t mid, size_t last, float value) {
  p[0] = value;
  p[mid] = value;
  p[last] = value;
}

double percentile(std::vector<double> sorted, double q) {
  if (sorted.empty()) return 0.0;
  const double pos = q * (sorted.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = pos - lo;
  return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
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

void share_unique_id(ncclUniqueId* id, int rank, const std::string& path,
                     int poll_secs) {
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
  const int attempts = poll_secs * 5;
  for (int attempt = 0; attempt < attempts; ++attempt) {
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

struct Stats {
  double minv, p5, p25, p50, p75, p95, p99, maxv, mean, sd;
};

Stats stats_of(std::vector<double> v) {
  Stats s{};
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.minv = v.front();
  s.maxv = v.back();
  s.p5 = percentile(v, 0.05);
  s.p25 = percentile(v, 0.25);
  s.p50 = percentile(v, 0.50);
  s.p75 = percentile(v, 0.75);
  s.p95 = percentile(v, 0.95);
  s.p99 = percentile(v, 0.99);
  double sum = 0.0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  double var = 0.0;
  for (double x : v) var += (x - s.mean) * (x - s.mean);
  s.sd = (v.size() > 1) ? std::sqrt(var / (v.size() - 1)) : 0.0;
  return s;
}

void json_stats(FILE* f, const char* name, const Stats& s, int n) {
  std::fprintf(f,
               "\"%s\":{\"n\":%d,\"min_us\":%.4f,\"p5_us\":%.4f,"
               "\"p25_us\":%.4f,\"p50_us\":%.4f,\"p75_us\":%.4f,"
               "\"p95_us\":%.4f,\"p99_us\":%.4f,\"max_us\":%.4f,"
               "\"mean_us\":%.4f,\"sd_us\":%.4f}",
               name, n, s.minv, s.p5, s.p25, s.p50, s.p75, s.p95, s.p99,
               s.maxv, s.mean, s.sd);
}

void dump_samples(FILE* f, const char* name, const std::vector<double>& v) {
  std::fprintf(f, "\"%s\":[", name);
  for (size_t i = 0; i < v.size(); ++i) {
    std::fprintf(f, "%s%.3f", i ? "," : "", v[i]);
  }
  std::fprintf(f, "]");
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "matrix";
  std::string out_path = "disco_result.json";
  std::string id_path = "disco_unique_id.bin";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
    else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    else if (std::strcmp(argv[i], "--id-path") == 0 && i + 1 < argc) id_path = argv[++i];
  }

  const int rank = env_int("DISCO_RANK", env_int("SLURM_PROCID", 0));
  const int world = env_int("DISCO_WORLD", env_int("SLURM_NTASKS", 1));
  const int local_rank = env_int("DISCO_LOCALID", env_int("SLURM_LOCALID", 0));
  const int poll_secs = env_int("DISCO_ID_POLL_SECS", 180);
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);

  if (world < 2) {
    std::fprintf(stderr, "FATAL discovery needs at least two ranks\n");
    return 24;
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  const int device = local_rank % std::max(device_count, 1);
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int nccl_version = 0;
  NCCL_CHECK(ncclGetVersion(&nccl_version));

  std::printf("[disco] rank %d of %d on %s device %d (%s) nccl %d mode %s\n",
              rank, world, host, device, prop.name, nccl_version, mode.c_str());
  std::fflush(stdout);

  // Per-rank identity file so rank 0 can name every host in the output.
  {
    std::string p = out_path + ".rank" + std::to_string(rank) + ".host";
    FILE* f = std::fopen(p.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "%s %s\n", host, prop.name);
      std::fclose(f);
    }
  }

  ncclUniqueId id;
  std::memset(&id, 0, sizeof(id));
  share_unique_id(&id, rank, id_path, poll_secs);
  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  float* barrier_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&barrier_buf, sizeof(float)));
  CUDA_CHECK(cudaMemset(barrier_buf, 0, sizeof(float)));
  auto barrier = [&]() {
    NCCL_CHECK(ncclAllReduce(barrier_buf, barrier_buf, 1, ncclFloat, ncclSum,
                             comm, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  const size_t max_bytes = 16777216;  // 16 MiB, the largest probe payload
  float* buf_a = nullptr;
  float* buf_b = nullptr;
  CUDA_CHECK(cudaMalloc(&buf_a, max_bytes));
  CUDA_CHECK(cudaMalloc(&buf_b, max_bytes));
  CUDA_CHECK(cudaMemset(buf_a, 0, max_bytes));
  CUDA_CHECK(cudaMemset(buf_b, 0, max_bytes));

  FILE* out = nullptr;
  if (rank == 0) {
    out = std::fopen(out_path.c_str(), "w");
    if (out == nullptr) {
      std::fprintf(stderr, "FATAL cannot open %s\n", out_path.c_str());
      return 26;
    }
    std::fprintf(out, "{\"study\":\"merlin_fabric_flow_capture_v1_discovery\",");
    std::fprintf(out, "\"mode\":\"%s\",\"world\":%d,\"nccl_version\":%d,",
                 mode.c_str(), world, nccl_version);
    std::fprintf(out, "\"rank0_host\":\"%s\",\"rank0_device\":\"%s\",", host,
                 prop.name);
    std::fprintf(out, "\"slurm_job_id\":\"%s\",\"nodelist\":\"%s\",",
                 env_str("SLURM_JOB_ID").c_str(),
                 env_str("SLURM_JOB_NODELIST").c_str());
    std::fprintf(out, "\"epoch_realtime_ns_at_start\":%lld,",
                 now_ns(CLOCK_REALTIME));
  }

  if (mode == "matrix") {
    // All ordered pairs probed sequentially; every non-member idles at the
    // barrier that closes each block.
    const size_t payloads[3] = {8, 131072, 16777216};
    const int kRttIters = 100;
    const int kIsolated = 20;
    if (rank == 0) std::fprintf(out, "\"pairs\":[");
    bool first_pair = true;
    for (int a = 0; a < world; ++a) {
      for (int b = a + 1; b < world; ++b) {
        for (int pi = 0; pi < 3; ++pi) {
          const size_t bytes = payloads[pi];
          const size_t elems = bytes / sizeof(float);
          // Back-to-back ping-pong RTT, timed at rank a with CUDA events.
          double rtt_us = -1.0;
          barrier();
          if (rank == a) {
            for (int w = 0; w < 3; ++w) {  // warmup
              NCCL_CHECK(ncclSend(buf_a, elems, ncclFloat, b, comm, stream));
              NCCL_CHECK(ncclRecv(buf_b, elems, ncclFloat, b, comm, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
          } else if (rank == b) {
            for (int w = 0; w < 3; ++w) {
              NCCL_CHECK(ncclRecv(buf_b, elems, ncclFloat, a, comm, stream));
              NCCL_CHECK(ncclSend(buf_a, elems, ncclFloat, a, comm, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
          }
          barrier();
          if (rank == a) {
            CUDA_CHECK(cudaEventRecord(e0, stream));
            for (int it = 0; it < kRttIters; ++it) {
              NCCL_CHECK(ncclSend(buf_a, elems, ncclFloat, b, comm, stream));
              NCCL_CHECK(ncclRecv(buf_b, elems, ncclFloat, b, comm, stream));
            }
            CUDA_CHECK(cudaEventRecord(e1, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            rtt_us = static_cast<double>(ms) * 1000.0 / kRttIters;
          } else if (rank == b) {
            for (int it = 0; it < kRttIters; ++it) {
              NCCL_CHECK(ncclRecv(buf_b, elems, ncclFloat, a, comm, stream));
              NCCL_CHECK(ncclSend(buf_a, elems, ncclFloat, a, comm, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
          }
          barrier();
          // Isolated one-way, recv-side bracket at rank b (the wave-16
          // comparable reading): barrier, then a send and its recv, recv
          // bracketed by events on the receiver.
          std::vector<double> oneway;
          for (int s = 0; s < kIsolated; ++s) {
            barrier();
            if (rank == a) {
              NCCL_CHECK(ncclSend(buf_a, elems, ncclFloat, b, comm, stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
            } else if (rank == b) {
              CUDA_CHECK(cudaEventRecord(e0, stream));
              NCCL_CHECK(ncclRecv(buf_b, elems, ncclFloat, a, comm, stream));
              CUDA_CHECK(cudaEventRecord(e1, stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
              float ms = 0.f;
              CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
              oneway.push_back(static_cast<double>(ms) * 1000.0);
            }
          }
          barrier();
          // Rank b ships its median one-way to rank a through the wire so
          // rank 0 can print it (8 bytes, after all timed blocks per pair).
          double oneway_median = -1.0;
          if (!oneway.empty()) {
            std::sort(oneway.begin(), oneway.end());
            oneway_median = oneway[oneway.size() / 2];
          }
          double pair_row[2] = {rtt_us, oneway_median};
          {
            double* d = nullptr;
            CUDA_CHECK(cudaMalloc(&d, 2 * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d, pair_row, 2 * sizeof(double),
                                  cudaMemcpyHostToDevice));
            NCCL_CHECK(ncclAllReduce(d, d, 2, ncclDouble, ncclMax, comm,
                                     stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(pair_row, d, 2 * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d));
          }
          if (rank == 0) {
            std::fprintf(out,
                         "%s{\"a\":%d,\"b\":%d,\"bytes\":%zu,"
                         "\"rtt_back_to_back_us\":%.4f,"
                         "\"oneway_recv_isolated_median_us\":%.4f}",
                         first_pair ? "" : ",", a, b, bytes, pair_row[0],
                         pair_row[1]);
            first_pair = false;
          }
        }
      }
    }
    if (rank == 0) std::fprintf(out, "],");
  }

  if (mode == "jitter") {
    if (world != 2) {
      std::fprintf(stderr, "FATAL jitter mode needs exactly two ranks\n");
      return 24;
    }
    // Rank 0 is the destination (the capture harness convention); rank 1 the
    // source. The per-chunk loop below is the loop the capture harness will
    // use, byte for byte in its operations: recv, sync, CPU timestamp,
    // 4-byte probe readback, check.
    const size_t ladder[] = {4194304, 8388608, 16777216, 33554432,
                             67108864, 134217728};
    const int reps_of[] = {200, 200, 200, 200, 200, 100};
    const int n_sizes = 6;

    size_t big = ladder[n_sizes - 1];
    float* chunk_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&chunk_buf, big));
    CUDA_CHECK(cudaMemset(chunk_buf, 0, big));
    float* host_probe = nullptr;
    CUDA_CHECK(cudaMallocHost(&host_probe, sizeof(float)));

    if (rank == 0) std::fprintf(out, "\"jitter\":{");

    // Tracer floor A: launch + sync + 4 B readback + clock, no network.
    {
      const int reps = 1000;
      std::vector<double> t_us;
      t_us.reserve(reps);
      const size_t elems = 1024;
      for (int i = 0; i < 20; ++i) {  // warmup
        k_write_probe<<<1, 1, 0, stream>>>(chunk_buf, elems / 2, elems - 1,
                                           1.0f);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      for (int i = 0; i < reps; ++i) {
        const long long t0 = now_ns(CLOCK_MONOTONIC_RAW);
        k_write_probe<<<1, 1, 0, stream>>>(chunk_buf, elems / 2, elems - 1,
                                           static_cast<float>(i));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(host_probe, chunk_buf, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (*host_probe != static_cast<float>(i)) {
          std::fprintf(stderr, "FATAL tracer floor probe mismatch\n");
          return 27;
        }
        const long long t1 = now_ns(CLOCK_MONOTONIC_RAW);
        t_us.push_back(static_cast<double>(t1 - t0) / 1000.0);
      }
      if (rank == 0) {
        json_stats(out, "tracer_floor_launch_sync_read_us", stats_of(t_us),
                   reps);
        std::fprintf(out, ",");
      }
    }
    // Tracer floor B: the clock call itself.
    {
      const int reps = 100000;
      std::vector<double> t_us;
      t_us.reserve(reps);
      for (int i = 0; i < reps; ++i) {
        const long long t0 = now_ns(CLOCK_MONOTONIC_RAW);
        const long long t1 = now_ns(CLOCK_MONOTONIC_RAW);
        t_us.push_back(static_cast<double>(t1 - t0) / 1000.0);
      }
      if (rank == 0) {
        json_stats(out, "clock_gettime_pair_us", stats_of(t_us), reps);
        std::fprintf(out, ",");
      }
    }

    if (rank == 0) std::fprintf(out, "\"ladder\":[");
    for (int li = 0; li < n_sizes; ++li) {
      const size_t bytes = ladder[li];
      const size_t elems = bytes / sizeof(float);
      const int reps = reps_of[li];
      std::vector<double> t_us;
      t_us.reserve(reps);

      barrier();
      // Warmup: 5 chunks.
      for (int w = 0; w < 5; ++w) {
        if (rank == 1) {
          k_write_probe<<<1, 1, 0, stream>>>(chunk_buf, elems / 2, elems - 1,
                                             -2.0f);
          NCCL_CHECK(ncclSend(chunk_buf, elems, ncclFloat, 0, comm, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
          NCCL_CHECK(ncclRecv(chunk_buf, elems, ncclFloat, 1, comm, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
        }
      }
      barrier();
      const long long block_start = now_ns(CLOCK_REALTIME);
      std::printf("JITTER_BLOCK size=%zu start_epoch_ns=%lld\n", bytes,
                  block_start);
      std::fflush(stdout);
      for (int k = 0; k < reps; ++k) {
        if (rank == 1) {
          const long long t0 = now_ns(CLOCK_MONOTONIC_RAW);
          k_write_probe<<<1, 1, 0, stream>>>(chunk_buf, elems / 2, elems - 1,
                                             static_cast<float>(k));
          NCCL_CHECK(ncclSend(chunk_buf, elems, ncclFloat, 0, comm, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          const long long t1 = now_ns(CLOCK_MONOTONIC_RAW);
          t_us.push_back(static_cast<double>(t1 - t0) / 1000.0);
        } else {
          const long long t0 = now_ns(CLOCK_MONOTONIC_RAW);
          NCCL_CHECK(ncclRecv(chunk_buf, elems, ncclFloat, 1, comm, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaMemcpyAsync(host_probe, chunk_buf, sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          if (*host_probe != static_cast<float>(k)) {
            std::fprintf(stderr,
                         "FATAL jitter probe mismatch at size %zu rep %d: "
                         "got %f\n",
                         bytes, k, static_cast<double>(*host_probe));
            return 27;
          }
          const long long t1 = now_ns(CLOCK_MONOTONIC_RAW);
          t_us.push_back(static_cast<double>(t1 - t0) / 1000.0);
        }
      }
      const long long block_end = now_ns(CLOCK_REALTIME);
      std::printf("JITTER_BLOCK size=%zu end_epoch_ns=%lld\n", bytes,
                  block_end);
      std::fflush(stdout);
      barrier();

      // Rank 1 sends its sample vector to rank 0 so both sides land in the
      // output; the transfer happens after the timed block.
      std::vector<double> src_t(t_us.size(), 0.0);
      {
        double* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, reps * sizeof(double)));
        if (rank == 1) {
          CUDA_CHECK(cudaMemcpy(d, t_us.data(), reps * sizeof(double),
                                cudaMemcpyHostToDevice));
          NCCL_CHECK(ncclSend(d, reps, ncclDouble, 0, comm, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
          NCCL_CHECK(ncclRecv(d, reps, ncclDouble, 1, comm, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          CUDA_CHECK(cudaMemcpy(src_t.data(), d, reps * sizeof(double),
                                cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaFree(d));
      }

      if (rank == 0) {
        const Stats dst = stats_of(t_us);
        const Stats src = stats_of(src_t);
        std::fprintf(out, "%s{\"bytes\":%zu,\"reps\":%d,", li ? "," : "",
                     bytes, reps);
        json_stats(out, "dest_per_chunk", dst, reps);
        std::fprintf(out, ",");
        json_stats(out, "source_per_chunk", src, reps);
        std::fprintf(out, ",\"dest_spread_p95_minus_p5_us\":%.4f,",
                     dst.p95 - dst.p5);
        std::fprintf(out, "\"dest_spread_over_median\":%.6f,",
                     (dst.p50 > 0.0) ? (dst.p95 - dst.p5) / dst.p50 : -1.0);
        dump_samples(out, "dest_samples_us", t_us);
        std::fprintf(out, ",");
        dump_samples(out, "source_samples_us", src_t);
        std::fprintf(out, "}");
      }
    }
    if (rank == 0) std::fprintf(out, "]},");

    CUDA_CHECK(cudaFreeHost(host_probe));
    CUDA_CHECK(cudaFree(chunk_buf));
  }

  if (rank == 0) {
    std::fprintf(out, "\"epoch_realtime_ns_at_end\":%lld}\n",
                 now_ns(CLOCK_REALTIME));
    std::fclose(out);
    std::printf("[disco] wrote %s\n", out_path.c_str());
  }

  CUDA_CHECK(cudaFree(buf_a));
  CUDA_CHECK(cudaFree(buf_b));
  CUDA_CHECK(cudaFree(barrier_buf));
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(comm));
  std::printf("[disco] rank %d clean\n", rank);
  return 0;
}
