// Lane B of the GH200 hardware envelope study: four cards inside one node.
//
// Measures the NVLink peer bandwidth matrix, the NCCL collective latency and
// bandwidth model at participant widths two and four, and one cell where a
// collective and a dense GEMM contend for the same SMs and HBM.
//
// The communicator is single process, one host thread and one stream per
// device, built with ncclCommInitAll. Every timed block is bracketed by CUDA
// events on the measured stream. Multi-stream blocks use an event fork and
// join on one device so the reported span is the true makespan.
//
// This is the A100 lane B of commit 8509a2c with two changes: the fill grid
// scales with the SM count instead of assuming 108 SMs, and a missing peer
// link is recorded rather than fatal, because a Grace Hopper quad node is not
// guaranteed to carry direct GPU-to-GPU NVLink on every ordered pair. Every
// timing path is otherwise identical.
//
// The frozen sweep and the expected bounds live in expectations.md.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
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

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (expr);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "FATAL cublas %s at %s:%d: %d\n", #expr, __FILE__, \
                   __LINE__, static_cast<int>(st_));                          \
      std::exit(22);                                                          \
    }                                                                         \
  } while (0)

class Barrier {
 public:
  explicit Barrier(int count) : threshold_(count), remaining_(count), generation_(0) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const unsigned generation = generation_;
    if (--remaining_ == 0) {
      ++generation_;
      remaining_ = threshold_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [this, generation] { return generation != generation_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int threshold_;
  int remaining_;
  unsigned generation_;
};

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
  void open_obj(const char* name) {
    key(name);
    body += "{";
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

__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, float value) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const __nv_bfloat16 v = __float2bfloat16(value);
  for (; i < n; i += stride) p[i] = v;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n == 0) return 0.0;
  return (n % 2 == 1) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

constexpr int kBlockThreads = 256;
constexpr size_t kMaxBytes = 1ull << 30;  // 1 GiB, the largest frozen size
constexpr int kGemmDim = 8192;

enum class Op { AllReduce, AllGather, ReduceScatter, Broadcast };

const char* op_name(Op op) {
  switch (op) {
    case Op::AllReduce: return "allreduce";
    case Op::AllGather: return "allgather";
    case Op::ReduceScatter: return "reducescatter";
    case Op::Broadcast: return "broadcast";
  }
  return "unknown";
}

// nccl-tests bus bandwidth factor for one collective at one width.
double busbw_factor(Op op, int n) {
  switch (op) {
    case Op::AllReduce: return 2.0 * (n - 1) / n;
    case Op::AllGather:
    case Op::ReduceScatter: return static_cast<double>(n - 1) / n;
    case Op::Broadcast: return 1.0;
  }
  return 1.0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "lane_b_result.json";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  int nccl_version = 0;
  NCCL_CHECK(ncclGetVersion(&nccl_version));
  cudaDeviceProp prop0{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop0, 0));
  const int kGridBlocks = prop0.multiProcessorCount * 8;

  Json j;
  j.raw("{");
  j.str("lane", "B");
  j.str("study", "gh200_hardware_envelope_v1");
  j.integer("visible_device_count", device_count);
  j.integer("nccl_version", nccl_version);

  j.open_arr("devices");
  for (int d = 0; d < device_count; ++d) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, d));
    char pci[64] = {0};
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), d));
    j.item_sep();
    j.raw("{");
    j.integer("ordinal", d);
    j.str("name", prop.name);
    j.str("pci_bus_id", pci);
    j.integer("sm_count", prop.multiProcessorCount);
    j.close_obj();
  }
  j.close_arr();

  // Peer access for every ordered pair. A pair without direct peer access is
  // recorded and skipped rather than fatal: the collective lanes can still run
  // over whatever path the driver selects, and which pairs are direct is
  // itself an observation this study exists to make.
  std::vector<std::vector<int>> peer_ok(device_count, std::vector<int>(device_count, 0));
  for (int a = 0; a < device_count; ++a) {
    CUDA_CHECK(cudaSetDevice(a));
    for (int b = 0; b < device_count; ++b) {
      if (a == b) continue;
      int can = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can, a, b));
      if (!can) continue;
      const cudaError_t err = cudaDeviceEnablePeerAccess(b, 0);
      if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
        peer_ok[a][b] = 1;
      } else {
        std::fprintf(stderr, "WARN peer enable %d->%d: %s\n", a, b, cudaGetErrorString(err));
      }
    }
  }
  CUDA_CHECK(cudaGetLastError());

  j.open_arr("peer_access");
  for (int a = 0; a < device_count; ++a) {
    for (int b = 0; b < device_count; ++b) {
      if (a == b) continue;
      j.item_sep();
      j.raw("{");
      j.integer("src", a);
      j.integer("dst", b);
      j.integer("enabled", peer_ok[a][b]);
      j.close_obj();
    }
  }
  j.close_arr();

  // ---------------------------------------------------------------------------
  // B1 peer bandwidth matrix.
  // ---------------------------------------------------------------------------
  const int kP2pWarmup = 3;
  const int kP2pTimed = 10;
  const size_t p2p_bytes = kMaxBytes;

  std::vector<void*> p2p_src(device_count, nullptr);
  std::vector<void*> p2p_dst(device_count, nullptr);
  for (int d = 0; d < device_count; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    CUDA_CHECK(cudaMalloc(&p2p_src[d], p2p_bytes));
    CUDA_CHECK(cudaMalloc(&p2p_dst[d], p2p_bytes));
    CUDA_CHECK(cudaMemset(p2p_src[d], 0x11 + d, p2p_bytes));
    CUDA_CHECK(cudaMemset(p2p_dst[d], 0x00, p2p_bytes));
  }

  j.integer("p2p_bytes", static_cast<long long>(p2p_bytes));
  j.integer("p2p_warmup_iters", kP2pWarmup);
  j.integer("p2p_timed_iters", kP2pTimed);
  j.open_arr("p2p_unidirectional");
  for (int s = 0; s < device_count; ++s) {
    for (int d = 0; d < device_count; ++d) {
      if (s == d) continue;
      if (!peer_ok[s][d]) {
        j.item_sep();
        j.raw("{");
        j.integer("src", s);
        j.integer("dst", d);
        j.str("status", "skipped_no_peer_access");
        j.close_obj();
        continue;
      }
      CUDA_CHECK(cudaSetDevice(s));
      cudaStream_t stream;
      cudaEvent_t e0, e1;
      CUDA_CHECK(cudaStreamCreate(&stream));
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));
      std::vector<double> samples;
      for (int it = 0; it < kP2pWarmup + kP2pTimed; ++it) {
        CUDA_CHECK(cudaEventRecord(e0, stream));
        CUDA_CHECK(cudaMemcpyPeerAsync(p2p_dst[d], d, p2p_src[s], s, p2p_bytes, stream));
        CUDA_CHECK(cudaEventRecord(e1, stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (it >= kP2pWarmup) samples.push_back(ms);
      }
      const double ms = median(samples);
      j.item_sep();
      j.raw("{");
      j.integer("src", s);
      j.integer("dst", d);
      j.num("time_ms", ms);
      j.num("gbps", p2p_bytes / (ms * 1e-3) / 1e9);
      j.close_obj();
      CUDA_CHECK(cudaEventDestroy(e0));
      CUDA_CHECK(cudaEventDestroy(e1));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }
  j.close_arr();

  // Multi-stream makespan helper: fork from stream 0, join back into stream 0,
  // so the reported span comes from two events on one device.
  auto timed_multi_copy = [&](int owner, const std::vector<std::array<int, 2>>& transfers) {
    CUDA_CHECK(cudaSetDevice(owner));
    const size_t lanes = transfers.size();
    std::vector<cudaStream_t> streams(lanes);
    std::vector<cudaEvent_t> lane_done(lanes);
    for (size_t i = 0; i < lanes; ++i) {
      CUDA_CHECK(cudaStreamCreate(&streams[i]));
      CUDA_CHECK(cudaEventCreate(&lane_done[i]));
    }
    cudaEvent_t begin, end;
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventCreate(&end));

    std::vector<double> samples;
    for (int it = 0; it < kP2pWarmup + kP2pTimed; ++it) {
      CUDA_CHECK(cudaEventRecord(begin, streams[0]));
      for (size_t i = 1; i < lanes; ++i) CUDA_CHECK(cudaStreamWaitEvent(streams[i], begin, 0));
      for (size_t i = 0; i < lanes; ++i) {
        const int s = transfers[i][0];
        const int d = transfers[i][1];
        CUDA_CHECK(cudaMemcpyPeerAsync(p2p_dst[d], d, p2p_src[s], s, p2p_bytes, streams[i]));
      }
      for (size_t i = 0; i < lanes; ++i) CUDA_CHECK(cudaEventRecord(lane_done[i], streams[i]));
      for (size_t i = 1; i < lanes; ++i)
        CUDA_CHECK(cudaStreamWaitEvent(streams[0], lane_done[i], 0));
      CUDA_CHECK(cudaEventRecord(end, streams[0]));
      CUDA_CHECK(cudaEventSynchronize(end));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
      if (it >= kP2pWarmup) samples.push_back(ms);
    }
    for (size_t i = 0; i < lanes; ++i) {
      CUDA_CHECK(cudaEventDestroy(lane_done[i]));
      CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return median(samples);
  };

  j.open_obj("p2p_bidirectional_0_1");
  if (device_count >= 2 && peer_ok[0][1] && peer_ok[1][0]) {
    const std::vector<std::array<int, 2>> bidir = {{0, 1}, {1, 0}};
    const double ms = timed_multi_copy(0, bidir);
    j.str("status", "measured");
    j.num("time_ms", ms);
    j.num("aggregate_gbps", 2.0 * p2p_bytes / (ms * 1e-3) / 1e9);
    j.num("per_direction_gbps", p2p_bytes / (ms * 1e-3) / 1e9);
  } else {
    j.str("status", "skipped_no_peer_access");
  }
  j.close_obj();

  j.open_obj("p2p_fanout_0_to_all");
  if (device_count >= 4 && peer_ok[0][1] && peer_ok[0][2] && peer_ok[0][3]) {
    const std::vector<std::array<int, 2>> fanout = {{0, 1}, {0, 2}, {0, 3}};
    const double ms = timed_multi_copy(0, fanout);
    j.str("status", "measured");
    j.num("time_ms", ms);
    j.num("aggregate_gbps", 3.0 * p2p_bytes / (ms * 1e-3) / 1e9);
  } else {
    j.str("status", "skipped_no_peer_access");
  }
  j.close_obj();

  for (int d = 0; d < device_count; ++d) {
    CUDA_CHECK(cudaSetDevice(d));
    CUDA_CHECK(cudaFree(p2p_src[d]));
    CUDA_CHECK(cudaFree(p2p_dst[d]));
  }

  // ---------------------------------------------------------------------------
  // B2 NCCL collectives.
  // ---------------------------------------------------------------------------
  std::vector<size_t> sizes;
  sizes.push_back(8);
  for (int k = 10; k <= 30; ++k) sizes.push_back(1ull << k);

  const Op ops[] = {Op::AllReduce, Op::AllGather, Op::ReduceScatter, Op::Broadcast};
  const int widths[] = {2, 4};
  const int kWarmup = 5;

  j.integer("collective_warmup_iters", kWarmup);
  j.open_arr("collectives");

  for (int width : widths) {
    if (width > device_count) continue;

    std::vector<int> devs(width);
    for (int i = 0; i < width; ++i) devs[i] = i;
    std::vector<ncclComm_t> comms(width);
    NCCL_CHECK(ncclCommInitAll(comms.data(), width, devs.data()));

    std::vector<float*> send(width, nullptr), recv(width, nullptr);
    std::vector<cudaStream_t> streams(width);
    for (int r = 0; r < width; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaMalloc(&send[r], kMaxBytes));
      CUDA_CHECK(cudaMalloc(&recv[r], kMaxBytes));
      CUDA_CHECK(cudaStreamCreate(&streams[r]));
      k_fill_float<<<kGridBlocks, kBlockThreads>>>(send[r], kMaxBytes / sizeof(float),
                                                   static_cast<float>(r + 1));
      CUDA_CHECK(cudaMemset(recv[r], 0, kMaxBytes));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    const float expected_sum = static_cast<float>(width * (width + 1) / 2);

    for (Op op : ops) {
      for (size_t bytes : sizes) {
        const size_t elems_total = bytes / sizeof(float);
        // All-gather sends bytes/width and receives bytes; reduce-scatter is
        // the mirror. Both need a whole number of elements per rank.
        const bool needs_split = (op == Op::AllGather || op == Op::ReduceScatter);
        if (needs_split && (elems_total % static_cast<size_t>(width) != 0 || elems_total == 0)) {
          j.item_sep();
          j.raw("{");
          j.str("op", op_name(op));
          j.integer("width", width);
          j.integer("bytes", static_cast<long long>(bytes));
          j.str("status", "skipped_not_divisible");
          j.close_obj();
          continue;
        }
        if (elems_total == 0) continue;

        const int timed = (bytes > (64ull << 20)) ? 10 : 20;
        const size_t split_elems = needs_split ? elems_total / width : elems_total;

        Barrier barrier(width);
        std::vector<double> per_rank_ms(width, 0.0);
        std::vector<int> rank_mismatch(width, 0);

        auto worker = [&](int r) {
          CUDA_CHECK(cudaSetDevice(r));
          cudaEvent_t e0, e1;
          CUDA_CHECK(cudaEventCreate(&e0));
          CUDA_CHECK(cudaEventCreate(&e1));

          auto issue = [&]() {
            switch (op) {
              case Op::AllReduce:
                NCCL_CHECK(ncclAllReduce(send[r], recv[r], elems_total, ncclFloat, ncclSum,
                                         comms[r], streams[r]));
                break;
              case Op::AllGather:
                NCCL_CHECK(ncclAllGather(send[r], recv[r], split_elems, ncclFloat, comms[r],
                                         streams[r]));
                break;
              case Op::ReduceScatter:
                NCCL_CHECK(ncclReduceScatter(send[r], recv[r], split_elems, ncclFloat, ncclSum,
                                             comms[r], streams[r]));
                break;
              case Op::Broadcast:
                NCCL_CHECK(ncclBroadcast(send[r], recv[r], elems_total, ncclFloat, 0, comms[r],
                                         streams[r]));
                break;
            }
          };

          for (int it = 0; it < kWarmup; ++it) issue();
          CUDA_CHECK(cudaStreamSynchronize(streams[r]));
          barrier.wait();

          CUDA_CHECK(cudaEventRecord(e0, streams[r]));
          for (int it = 0; it < timed; ++it) issue();
          CUDA_CHECK(cudaEventRecord(e1, streams[r]));
          CUDA_CHECK(cudaStreamSynchronize(streams[r]));
          float ms = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
          per_rank_ms[r] = ms / timed;

          if (op == Op::AllReduce) {
            const size_t probe[3] = {0, elems_total / 2, elems_total - 1};
            float host[3] = {0.f, 0.f, 0.f};
            for (int p = 0; p < 3; ++p) {
              CUDA_CHECK(cudaMemcpy(&host[p], recv[r] + probe[p], sizeof(float),
                                    cudaMemcpyDeviceToHost));
              if (host[p] != expected_sum) rank_mismatch[r] = 1;
            }
          }

          CUDA_CHECK(cudaEventDestroy(e0));
          CUDA_CHECK(cudaEventDestroy(e1));
          barrier.wait();
        };

        std::vector<std::thread> threads;
        threads.reserve(width);
        for (int r = 0; r < width; ++r) threads.emplace_back(worker, r);
        for (std::thread& t : threads) t.join();

        double ms = 0.0;
        int mismatches = 0;
        for (int r = 0; r < width; ++r) {
          ms = std::max(ms, per_rank_ms[r]);
          mismatches += rank_mismatch[r];
        }
        const double seconds = ms * 1e-3;
        const double algbw = bytes / seconds / 1e9;
        const double busbw = algbw * busbw_factor(op, width);

        j.item_sep();
        j.raw("{");
        j.str("op", op_name(op));
        j.integer("width", width);
        j.integer("bytes", static_cast<long long>(bytes));
        j.integer("timed_iters", timed);
        j.str("status", "measured");
        j.num("time_us", ms * 1e3);
        j.num("algbw_gbps", algbw);
        j.num("busbw_gbps", busbw);
        j.num("busbw_factor", busbw_factor(op, width));
        j.integer("allreduce_mismatching_ranks", mismatches);
        j.close_obj();
      }
    }

    // -------------------------------------------------------------------------
    // B3 contention, run only at the full width.
    // -------------------------------------------------------------------------
    if (width == 4) {
      const size_t contend_bytes = 256ull << 20;
      const size_t contend_elems = contend_bytes / sizeof(float);
      const int kCommIters = 20;
      const int kGemmIters = 8;

      std::vector<__nv_bfloat16*> ga(width, nullptr), gb(width, nullptr), gc(width, nullptr);
      std::vector<cudaStream_t> gemm_streams(width);
      std::vector<cublasHandle_t> handles(width);
      const size_t gemm_elems = static_cast<size_t>(kGemmDim) * kGemmDim;
      for (int r = 0; r < width; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(&ga[r], gemm_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&gb[r], gemm_elems * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(&gc[r], gemm_elems * sizeof(__nv_bfloat16)));
        k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(ga[r], gemm_elems, 0.01f);
        k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(gb[r], gemm_elems, 0.02f);
        CUDA_CHECK(cudaStreamCreate(&gemm_streams[r]));
        CUBLAS_CHECK(cublasCreate(&handles[r]));
        CUBLAS_CHECK(cublasSetStream(handles[r], gemm_streams[r]));
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      const float alpha = 1.0f;
      const float beta = 0.0f;

      auto gemm_call = [&](int r) {
        CUBLAS_CHECK(cublasGemmEx(handles[r], CUBLAS_OP_N, CUBLAS_OP_N, kGemmDim, kGemmDim,
                                  kGemmDim, &alpha, ga[r], CUDA_R_16BF, kGemmDim, gb[r],
                                  CUDA_R_16BF, kGemmDim, &beta, gc[r], CUDA_R_16BF, kGemmDim,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      };
      auto comm_call = [&](int r) {
        NCCL_CHECK(ncclAllReduce(send[r], recv[r], contend_elems, ncclFloat, ncclSum, comms[r],
                                 streams[r]));
      };

      // Mode 0: collective alone. Mode 1: GEMM alone. Mode 2: both concurrent.
      double comm_alone_us = 0.0, gemm_alone_us = 0.0;
      double comm_together_us = 0.0, gemm_together_us = 0.0, makespan_us = 0.0;

      for (int mode = 0; mode < 3; ++mode) {
        Barrier barrier(width);
        std::vector<double> comm_ms(width, 0.0), gemm_ms(width, 0.0), span_ms(width, 0.0);

        auto worker = [&](int r) {
          CUDA_CHECK(cudaSetDevice(r));
          cudaEvent_t begin, comm_end, gemm_end;
          CUDA_CHECK(cudaEventCreate(&begin));
          CUDA_CHECK(cudaEventCreate(&comm_end));
          CUDA_CHECK(cudaEventCreate(&gemm_end));

          if (mode != 1) {
            for (int it = 0; it < 3; ++it) comm_call(r);
          }
          if (mode != 0) {
            for (int it = 0; it < 2; ++it) gemm_call(r);
          }
          CUDA_CHECK(cudaDeviceSynchronize());
          barrier.wait();

          CUDA_CHECK(cudaEventRecord(begin, streams[r]));
          CUDA_CHECK(cudaStreamWaitEvent(gemm_streams[r], begin, 0));
          const int rounds = std::max(kCommIters, kGemmIters);
          for (int it = 0; it < rounds; ++it) {
            if (mode != 1 && it < kCommIters) comm_call(r);
            if (mode != 0 && it < kGemmIters) gemm_call(r);
          }
          CUDA_CHECK(cudaEventRecord(comm_end, streams[r]));
          CUDA_CHECK(cudaEventRecord(gemm_end, gemm_streams[r]));
          CUDA_CHECK(cudaDeviceSynchronize());

          float ms_comm = 0.f, ms_gemm = 0.f;
          CUDA_CHECK(cudaEventElapsedTime(&ms_comm, begin, comm_end));
          CUDA_CHECK(cudaEventElapsedTime(&ms_gemm, begin, gemm_end));
          if (mode != 1) comm_ms[r] = ms_comm / kCommIters;
          if (mode != 0) gemm_ms[r] = ms_gemm / kGemmIters;
          double span = 0.0;
          if (mode != 1) span = std::max(span, static_cast<double>(ms_comm));
          if (mode != 0) span = std::max(span, static_cast<double>(ms_gemm));
          span_ms[r] = span;

          CUDA_CHECK(cudaEventDestroy(begin));
          CUDA_CHECK(cudaEventDestroy(comm_end));
          CUDA_CHECK(cudaEventDestroy(gemm_end));
          barrier.wait();
        };

        std::vector<std::thread> threads;
        threads.reserve(width);
        for (int r = 0; r < width; ++r) threads.emplace_back(worker, r);
        for (std::thread& t : threads) t.join();

        double c = 0.0, g = 0.0, s = 0.0;
        for (int r = 0; r < width; ++r) {
          c = std::max(c, comm_ms[r]);
          g = std::max(g, gemm_ms[r]);
          s = std::max(s, span_ms[r]);
        }
        if (mode == 0) comm_alone_us = c * 1e3;
        if (mode == 1) gemm_alone_us = g * 1e3;
        if (mode == 2) {
          comm_together_us = c * 1e3;
          gemm_together_us = g * 1e3;
          makespan_us = s * 1e3;
        }
      }

      j.item_sep();
      j.raw("{");
      j.str("op", "contention");
      j.integer("width", width);
      j.integer("bytes", static_cast<long long>(contend_bytes));
      j.integer("comm_iters", kCommIters);
      j.integer("gemm_iters", kGemmIters);
      j.integer("gemm_dim", kGemmDim);
      j.str("status", "measured");
      j.num("comm_alone_us", comm_alone_us);
      j.num("gemm_alone_us", gemm_alone_us);
      j.num("comm_together_us", comm_together_us);
      j.num("gemm_together_us", gemm_together_us);
      j.num("makespan_us", makespan_us);
      j.num("alone_sum_us", comm_alone_us * kCommIters + gemm_alone_us * kGemmIters);
      j.close_obj();

      for (int r = 0; r < width; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUBLAS_CHECK(cublasDestroy(handles[r]));
        CUDA_CHECK(cudaStreamDestroy(gemm_streams[r]));
        CUDA_CHECK(cudaFree(ga[r]));
        CUDA_CHECK(cudaFree(gb[r]));
        CUDA_CHECK(cudaFree(gc[r]));
      }
    }

    for (int r = 0; r < width; ++r) {
      CUDA_CHECK(cudaSetDevice(r));
      CUDA_CHECK(cudaStreamDestroy(streams[r]));
      CUDA_CHECK(cudaFree(send[r]));
      CUDA_CHECK(cudaFree(recv[r]));
      NCCL_CHECK(ncclCommDestroy(comms[r]));
    }
  }
  j.close_arr();
  j.raw("}");

  FILE* f = std::fopen(out_path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "FATAL cannot write %s\n", out_path.c_str());
    return 25;
  }
  std::fwrite(j.body.data(), 1, j.body.size(), f);
  std::fputc('\n', f);
  std::fclose(f);
  std::printf("lane B wrote %s (%zu bytes)\n", out_path.c_str(), j.body.size());
  return 0;
}
