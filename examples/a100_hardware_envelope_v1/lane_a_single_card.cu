// Lane A of the A100 hardware envelope study: one card.
//
// Measures HBM bandwidth, dense BF16 GEMM throughput across the roofline
// crossover, kernel launch cost and the host PCIe link. Every timed block is
// bracketed by CUDA events on the measured stream, except the launch-cost
// block, whose subject is the host-side launch path and which is therefore
// timed on the host clock by design.
//
// The frozen sweep and the expected bounds live in expectations.md. This file
// only produces numbers; it does not judge them.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err_ = (expr);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "FATAL cuda %s at %s:%d: %s\n", #expr, __FILE__,  \
                   __LINE__, cudaGetErrorString(err_));                      \
      std::exit(20);                                                         \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(expr)                                                   \
  do {                                                                       \
    cublasStatus_t st_ = (expr);                                             \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "FATAL cublas %s at %s:%d: %d\n", #expr, __FILE__, \
                   __LINE__, static_cast<int>(st_));                         \
      std::exit(21);                                                         \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Clock and power observation. Fatal guard F7 requires an observation
// immediately before and immediately after every timed block.
// ---------------------------------------------------------------------------

struct ClockSample {
  bool valid = false;
  unsigned sm_mhz = 0;
  unsigned mem_mhz = 0;
  unsigned power_mw = 0;
  unsigned temp_c = 0;
};

nvmlDevice_t g_nvml_device = nullptr;
bool g_nvml_ready = false;

void nvml_open(int cuda_ordinal) {
  if (nvmlInit_v2() != NVML_SUCCESS) return;
  char pci[64] = {0};
  if (cudaDeviceGetPCIBusId(pci, sizeof(pci), cuda_ordinal) != cudaSuccess) return;
  if (nvmlDeviceGetHandleByPciBusId_v2(pci, &g_nvml_device) != NVML_SUCCESS) return;
  g_nvml_ready = true;
}

ClockSample clock_sample() {
  ClockSample s;
  if (!g_nvml_ready) return s;
  unsigned v = 0;
  if (nvmlDeviceGetClockInfo(g_nvml_device, NVML_CLOCK_SM, &v) == NVML_SUCCESS) s.sm_mhz = v;
  if (nvmlDeviceGetClockInfo(g_nvml_device, NVML_CLOCK_MEM, &v) == NVML_SUCCESS) s.mem_mhz = v;
  if (nvmlDeviceGetPowerUsage(g_nvml_device, &v) == NVML_SUCCESS) s.power_mw = v;
  if (nvmlDeviceGetTemperature(g_nvml_device, NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS)
    s.temp_c = v;
  s.valid = true;
  return s;
}

// ---------------------------------------------------------------------------
// Minimal JSON emission.
// ---------------------------------------------------------------------------

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

void emit_clocks(Json& j, const char* name, const ClockSample& s) {
  j.open_obj(name);
  j.integer("valid", s.valid ? 1 : 0);
  j.integer("sm_mhz", s.sm_mhz);
  j.integer("mem_mhz", s.mem_mhz);
  j.integer("power_mw", s.power_mw);
  j.integer("temp_c", s.temp_c);
  j.close_obj();
}

// ---------------------------------------------------------------------------
// Bandwidth kernels. Each uses a grid-stride loop over float4 elements so the
// access pattern is fully coalesced and the launch shape is fixed across the
// whole size sweep.
// ---------------------------------------------------------------------------

__global__ void k_read(const float4* __restrict__ src, size_t n, float* sink) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  for (; i < n; i += stride) {
    const float4 v = src[i];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  // Never true for the data written below, but the compiler cannot prove it,
  // so the loads are not eliminated.
  if (acc.x == 1.2345678e30f) sink[0] = acc.x + acc.y + acc.z + acc.w;
}

__global__ void k_write(float4* __restrict__ dst, size_t n, float value) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const float4 v = make_float4(value, value, value, value);
  for (; i < n; i += stride) dst[i] = v;
}

__global__ void k_copy(const float4* __restrict__ src, float4* __restrict__ dst, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

__global__ void k_nop() {}

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
constexpr int kGridBlocks = 864;  // 108 SMs x 8 blocks, one full occupancy wave

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "lane_a_result.json";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  nvml_open(0);

  int driver_version = 0, runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

  Json j;
  j.raw("{");
  j.str("lane", "A");
  j.str("study", "a100_hardware_envelope_v1");
  j.integer("visible_device_count", device_count);
  j.str("device_name", prop.name);
  j.integer("sm_count", prop.multiProcessorCount);
  j.integer("sm_clock_khz", prop.clockRate);
  j.integer("mem_clock_khz", prop.memoryClockRate);
  j.integer("mem_bus_bits", prop.memoryBusWidth);
  j.integer("l2_bytes", prop.l2CacheSize);
  j.integer("compute_capability_major", prop.major);
  j.integer("compute_capability_minor", prop.minor);
  j.integer("driver_api_version", driver_version);
  j.integer("runtime_api_version", runtime_version);
  j.integer("grid_blocks", kGridBlocks);
  j.integer("block_threads", kBlockThreads);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // -------------------------------------------------------------------------
  // A1 HBM bandwidth.
  // -------------------------------------------------------------------------
  const int kHbmWarmup = 3;
  const int kHbmTimed = 10;
  const size_t hbm_sizes_mib[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  const size_t max_bytes = 4096ull * 1024ull * 1024ull;

  float4* buf_a = nullptr;
  float4* buf_b = nullptr;
  float* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&buf_a, max_bytes));
  CUDA_CHECK(cudaMalloc(&buf_b, max_bytes));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
  CUDA_CHECK(cudaMemset(buf_a, 0x3f, max_bytes));
  CUDA_CHECK(cudaMemset(buf_b, 0x3f, max_bytes));

  j.integer("hbm_warmup_iters", kHbmWarmup);
  j.integer("hbm_timed_iters", kHbmTimed);
  j.open_arr("hbm");
  for (size_t mib : hbm_sizes_mib) {
    const size_t bytes = mib * 1024ull * 1024ull;
    const size_t n4 = bytes / sizeof(float4);

    const ClockSample before = clock_sample();
    double ms_read = 0.0, ms_write = 0.0, ms_copy = 0.0;
    std::vector<double> read_ms, write_ms, copy_ms;

    for (int it = 0; it < kHbmWarmup + kHbmTimed; ++it) {
      CUDA_CHECK(cudaEventRecord(ev_start, stream));
      k_read<<<kGridBlocks, kBlockThreads, 0, stream>>>(buf_a, n4, sink);
      CUDA_CHECK(cudaEventRecord(ev_stop, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      if (it >= kHbmWarmup) read_ms.push_back(ms);
    }
    for (int it = 0; it < kHbmWarmup + kHbmTimed; ++it) {
      CUDA_CHECK(cudaEventRecord(ev_start, stream));
      k_write<<<kGridBlocks, kBlockThreads, 0, stream>>>(buf_b, n4, 1.5f);
      CUDA_CHECK(cudaEventRecord(ev_stop, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      if (it >= kHbmWarmup) write_ms.push_back(ms);
    }
    for (int it = 0; it < kHbmWarmup + kHbmTimed; ++it) {
      CUDA_CHECK(cudaEventRecord(ev_start, stream));
      k_copy<<<kGridBlocks, kBlockThreads, 0, stream>>>(buf_a, buf_b, n4);
      CUDA_CHECK(cudaEventRecord(ev_stop, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      if (it >= kHbmWarmup) copy_ms.push_back(ms);
    }
    const ClockSample after = clock_sample();

    ms_read = median(read_ms);
    ms_write = median(write_ms);
    ms_copy = median(copy_ms);

    j.item_sep();
    j.raw("{");
    j.integer("size_mib", static_cast<long long>(mib));
    j.integer("bytes", static_cast<long long>(bytes));
    j.num("read_ms", ms_read);
    j.num("write_ms", ms_write);
    j.num("copy_ms", ms_copy);
    j.num("read_gbps", bytes / (ms_read * 1e-3) / 1e9);
    j.num("write_gbps", bytes / (ms_write * 1e-3) / 1e9);
    j.num("copy_gbps", 2.0 * bytes / (ms_copy * 1e-3) / 1e9);
    emit_clocks(j, "clocks_before", before);
    emit_clocks(j, "clocks_after", after);
    j.close_obj();
  }
  j.close_arr();

  CUDA_CHECK(cudaFree(buf_a));
  CUDA_CHECK(cudaFree(buf_b));

  // -------------------------------------------------------------------------
  // A2 GEMM roofline crossover, BF16 in and out with FP32 accumulation.
  // -------------------------------------------------------------------------
  cublasHandle_t blas;
  CUBLAS_CHECK(cublasCreate(&blas));
  CUBLAS_CHECK(cublasSetStream(blas, stream));

  struct GemmPoint {
    int m, n, k, warmup, timed;
    const char* sweep;
  };
  std::vector<GemmPoint> gemm_points;
  const int decode_m[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  for (int m : decode_m) gemm_points.push_back({m, 8192, 8192, 5, 20, "decode"});
  const int square_n[] = {1024, 2048, 4096, 8192, 16384};
  for (int s : square_n) {
    const bool huge = (s == 16384);
    gemm_points.push_back({s, s, s, huge ? 2 : 5, huge ? 5 : 20, "square"});
  }

  size_t max_elems = 0;
  for (const GemmPoint& p : gemm_points) {
    max_elems = std::max(max_elems, static_cast<size_t>(p.m) * p.k);
    max_elems = std::max(max_elems, static_cast<size_t>(p.k) * p.n);
    max_elems = std::max(max_elems, static_cast<size_t>(p.m) * p.n);
  }
  __nv_bfloat16 *ga = nullptr, *gb = nullptr, *gc = nullptr;
  CUDA_CHECK(cudaMalloc(&ga, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gb, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gc, max_elems * sizeof(__nv_bfloat16)));
  k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(ga, max_elems, 0.01f);
  k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(gb, max_elems, 0.02f);
  CUDA_CHECK(cudaDeviceSynchronize());

  const float alpha = 1.0f;
  const float beta = 0.0f;

  j.open_arr("gemm");
  for (const GemmPoint& p : gemm_points) {
    const ClockSample before = clock_sample();
    std::vector<double> samples;
    for (int it = 0; it < p.warmup + p.timed; ++it) {
      CUDA_CHECK(cudaEventRecord(ev_start, stream));
      CUBLAS_CHECK(cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, p.m, p.n, p.k, &alpha, ga,
                                CUDA_R_16BF, p.m, gb, CUDA_R_16BF, p.k, &beta, gc, CUDA_R_16BF,
                                p.m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      CUDA_CHECK(cudaEventRecord(ev_stop, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      if (it >= p.warmup) samples.push_back(ms);
    }
    const ClockSample after = clock_sample();

    const double ms = median(samples);
    const double flops = 2.0 * p.m * p.n * static_cast<double>(p.k);
    const double bytes = 2.0 * (static_cast<double>(p.m) * p.k + static_cast<double>(p.k) * p.n +
                                static_cast<double>(p.m) * p.n);
    j.item_sep();
    j.raw("{");
    j.str("sweep", p.sweep);
    j.integer("m", p.m);
    j.integer("n", p.n);
    j.integer("k", p.k);
    j.integer("warmup_iters", p.warmup);
    j.integer("timed_iters", p.timed);
    j.num("time_ms", ms);
    j.num("flops", flops);
    j.num("bytes", bytes);
    j.num("tflops_achieved", flops / (ms * 1e-3) / 1e12);
    j.num("t_mem_ms", bytes / 2.03904e12 * 1e3);
    j.num("t_flop_ms", flops / 311.87e12 * 1e3);
    emit_clocks(j, "clocks_before", before);
    emit_clocks(j, "clocks_after", after);
    j.close_obj();
  }
  j.close_arr();

  CUBLAS_CHECK(cublasDestroy(blas));
  CUDA_CHECK(cudaFree(ga));
  CUDA_CHECK(cudaFree(gb));
  CUDA_CHECK(cudaFree(gc));

  // -------------------------------------------------------------------------
  // A3 kernel launch cost. The subject is the host launch path, so these three
  // blocks are timed on the host clock with a device synchronization inside
  // the measured region, as the freeze states.
  // -------------------------------------------------------------------------
  const int kPipelinedLaunches = 200000;
  const int kRoundtrips = 2000;
  const int kGraphNodes = 1000;
  const int kGraphReplays = 200;

  const ClockSample launch_before = clock_sample();

  for (int i = 0; i < 1000; ++i) k_nop<<<1, 1, 0, stream>>>();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kPipelinedLaunches; ++i) k_nop<<<1, 1, 0, stream>>>();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto t1 = std::chrono::steady_clock::now();
  const double pipelined_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kPipelinedLaunches;

  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kRoundtrips; ++i) {
    k_nop<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  t1 = std::chrono::steady_clock::now();
  const double roundtrip_us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / kRoundtrips;

  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < kGraphNodes; ++i) k_nop<<<1, 1, 0, stream>>>();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < 5; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kGraphReplays; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  t1 = std::chrono::steady_clock::now();
  const double graph_us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                          (static_cast<double>(kGraphReplays) * kGraphNodes);
  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));

  const ClockSample launch_after = clock_sample();

  j.open_obj("launch");
  j.integer("pipelined_launches", kPipelinedLaunches);
  j.integer("roundtrips", kRoundtrips);
  j.integer("graph_nodes", kGraphNodes);
  j.integer("graph_replays", kGraphReplays);
  j.num("pipelined_period_us", pipelined_us);
  j.num("roundtrip_us", roundtrip_us);
  j.num("graph_period_us", graph_us);
  emit_clocks(j, "clocks_before", launch_before);
  emit_clocks(j, "clocks_after", launch_after);
  j.close_obj();

  // -------------------------------------------------------------------------
  // A4 host link.
  // -------------------------------------------------------------------------
  const size_t pcie_bytes = 256ull * 1024ull * 1024ull;
  const int kPcieWarmup = 3;
  const int kPcieTimed = 10;
  void* host_pinned = nullptr;
  void* device_buf = nullptr;
  CUDA_CHECK(cudaHostAlloc(&host_pinned, pcie_bytes, cudaHostAllocDefault));
  CUDA_CHECK(cudaMalloc(&device_buf, pcie_bytes));
  std::memset(host_pinned, 0x5a, pcie_bytes);

  const ClockSample pcie_before = clock_sample();
  std::vector<double> h2d, d2h;
  for (int it = 0; it < kPcieWarmup + kPcieTimed; ++it) {
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_buf, host_pinned, pcie_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    if (it >= kPcieWarmup) h2d.push_back(ms);
  }
  for (int it = 0; it < kPcieWarmup + kPcieTimed; ++it) {
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_pinned, device_buf, pcie_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    if (it >= kPcieWarmup) d2h.push_back(ms);
  }
  const ClockSample pcie_after = clock_sample();

  j.open_obj("pcie");
  j.integer("bytes", static_cast<long long>(pcie_bytes));
  j.integer("warmup_iters", kPcieWarmup);
  j.integer("timed_iters", kPcieTimed);
  j.num("h2d_ms", median(h2d));
  j.num("d2h_ms", median(d2h));
  j.num("h2d_gbps", pcie_bytes / (median(h2d) * 1e-3) / 1e9);
  j.num("d2h_gbps", pcie_bytes / (median(d2h) * 1e-3) / 1e9);
  emit_clocks(j, "clocks_before", pcie_before);
  emit_clocks(j, "clocks_after", pcie_after);
  j.close_obj();

  CUDA_CHECK(cudaFreeHost(host_pinned));
  CUDA_CHECK(cudaFree(device_buf));
  CUDA_CHECK(cudaFree(sink));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  j.raw("}");

  FILE* f = std::fopen(out_path.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "FATAL cannot write %s\n", out_path.c_str());
    return 22;
  }
  std::fwrite(j.body.data(), 1, j.body.size(), f);
  std::fputc('\n', f);
  std::fclose(f);
  std::printf("lane A wrote %s (%zu bytes)\n", out_path.c_str(), j.body.size());
  if (g_nvml_ready) nvmlShutdown();
  return 0;
}
