// A100 graph launch v1: the stage-2 measurement harness.
//
// It separates three quantities that the campaign brief keeps apart, in the
// exact protocol frozen in expectations.md:
//
//   1. the back-to-back device period per kernel, P_mode(k), measured with
//      CUDA events around the whole chain and no events between kernels, in
//      both eager and CUDA-graph mode;
//   2. the host submission cost, measured with a monotonic host clock around
//      the launch loop only, taken before any synchronization and containing
//      none, so the host interval never contains device time and the device
//      events never contain host launch time;
//   3. the per-kernel in-graph time measured with inner event pairs, kept
//      separate because stage 1 showed that instrumentation costs about 2.3
//      microseconds of device time per boundary.
//
// It judges nothing. score_expectations.py evaluates the output.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (expr);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "FATAL cublas %s at %s:%d: %d\n", #expr, __FILE__, \
                   __LINE__, static_cast<int>(st_));                          \
      std::exit(21);                                                          \
    }                                                                         \
  } while (0)

constexpr int kBlocks = 12;         // timed blocks per cell
constexpr int kGraphReplays = 64;   // graph replays per host-submission block
constexpr int kBlockThreads = 256;
constexpr int kGridBlocks = 864;
constexpr float kFillA = 0.5f;
constexpr float kFillB = 0.25f;

// ---------------------------------------------------------------------------
// NVML.
// ---------------------------------------------------------------------------

struct ClockSample {
  int valid = 0;
  unsigned sm_mhz = 0;
  unsigned mem_mhz = 0;
  unsigned power_mw = 0;
  unsigned temp_c = 0;
  unsigned long long throttle = 0;
};

nvmlDevice_t g_nvml = nullptr;
bool g_nvml_ready = false;
std::string g_uuid;

void nvml_open(int ordinal) {
  if (nvmlInit_v2() != NVML_SUCCESS) return;
  char pci[64] = {0};
  if (cudaDeviceGetPCIBusId(pci, sizeof(pci), ordinal) != cudaSuccess) return;
  if (nvmlDeviceGetHandleByPciBusId_v2(pci, &g_nvml) != NVML_SUCCESS) return;
  char uuid[96] = {0};
  if (nvmlDeviceGetUUID(g_nvml, uuid, sizeof(uuid)) == NVML_SUCCESS) g_uuid = uuid;
  g_nvml_ready = true;
}

ClockSample clock_sample() {
  ClockSample s;
  if (!g_nvml_ready) return s;
  unsigned v = 0;
  if (nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &v) == NVML_SUCCESS) s.sm_mhz = v;
  if (nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_MEM, &v) == NVML_SUCCESS) s.mem_mhz = v;
  if (nvmlDeviceGetPowerUsage(g_nvml, &v) == NVML_SUCCESS) s.power_mw = v;
  if (nvmlDeviceGetTemperature(g_nvml, NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS) s.temp_c = v;
  unsigned long long reasons = 0;
  if (nvmlDeviceGetCurrentClocksThrottleReasons(g_nvml, &reasons) == NVML_SUCCESS) {
    s.throttle = reasons;
  }
  s.valid = 1;
  return s;
}

// ---------------------------------------------------------------------------
// JSON.
// ---------------------------------------------------------------------------

struct Json {
  std::string body;
  void raw(const std::string& t) { body += t; }
  void key(const char* n) {
    if (!body.empty() && body.back() != '{' && body.back() != '[') body += ",";
    body += "\"";
    body += n;
    body += "\":";
  }
  void num(const char* n, double v) {
    key(n);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    body += buf;
  }
  void integer(const char* n, long long v) {
    key(n);
    body += std::to_string(v);
  }
  void str(const char* n, const std::string& v) {
    key(n);
    body += "\"" + v + "\"";
  }
  void open_obj(const char* n) {
    key(n);
    body += "{";
  }
  void open_arr(const char* n) {
    key(n);
    body += "[";
  }
  void close_obj() { body += "}"; }
  void close_arr() { body += "]"; }
  void sep() {
    if (!body.empty() && body.back() != '[') body += ",";
  }
  void arr_num(double v) {
    sep();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.7g", v);
    body += buf;
  }
};
// ---------------------------------------------------------------------------
// Kernels and the kernel set.
// ---------------------------------------------------------------------------

__global__ void k_nop() {}

__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, float v) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const __nv_bfloat16 h = __float2bfloat16(v);
  for (; i < n; i += stride) p[i] = h;
}

cudaStream_t g_stream;
cublasHandle_t g_blas;
__nv_bfloat16 *g_ga = nullptr, *g_gb = nullptr, *g_gc = nullptr;
size_t g_gemm_elems = 0;

// The stage-1 repaired layout: Out[N,M] = W[N,K] * X[K,M], leading dimensions
// N, K and N, none of which depends on the token count M.
void gemm(int m, int n, int k) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, g_ga,
                            CUDA_R_16BF, n, g_gb, CUDA_R_16BF, k, &beta, g_gc,
                            CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

struct KernelSpec {
  const char* tag;
  bool is_gemm;
  int m, n, k;
  double flops;
  double bytes;
};

const KernelSpec kSet[] = {
    {"nop", false, 0, 0, 0, 0.0, 0.0},
    {"g1", true, 64, 2048, 1024, 2.0 * 64 * 2048 * 1024,
     2.0 * (64.0 * 1024 + 1024.0 * 2048 + 64.0 * 2048)},
    {"g2", true, 1024, 1024, 1024, 2.0 * 1024 * 1024 * 1024,
     2.0 * (1024.0 * 1024 + 1024.0 * 1024 + 1024.0 * 1024)},
    {"g4", true, 1, 8192, 8192, 2.0 * 1 * 8192 * 8192,
     2.0 * (1.0 * 8192 + 8192.0 * 8192 + 1.0 * 8192)},
};

void issue(const KernelSpec& spec) {
  if (spec.is_gemm) {
    gemm(spec.m, spec.n, spec.k);
  } else {
    k_nop<<<1, 1, 0, g_stream>>>();
  }
}

const char* kMixCycle[] = {"g1", "g2", "nop", "g4"};

const KernelSpec& lookup(const char* tag) {
  for (const KernelSpec& spec : kSet) {
    if (std::strcmp(spec.tag, tag) == 0) return spec;
  }
  std::fprintf(stderr, "FATAL unknown kernel tag %s\n", tag);
  std::exit(23);
}

// Issue one chain of `length` kernels for the given tag.
void issue_chain(const std::string& tag, int length) {
  if (tag == "mix") {
    for (int i = 0; i < length; ++i) issue(lookup(kMixCycle[i % 4]));
    return;
  }
  const KernelSpec& spec = lookup(tag.c_str());
  for (int i = 0; i < length; ++i) issue(spec);
}

double chain_flops(const std::string& tag, int length) {
  if (tag == "mix") {
    double total = 0.0;
    for (int i = 0; i < length; ++i) total += lookup(kMixCycle[i % 4]).flops;
    return total;
  }
  return lookup(tag.c_str()).flops * length;
}

double chain_bytes(const std::string& tag, int length) {
  if (tag == "mix") {
    double total = 0.0;
    for (int i = 0; i < length; ++i) total += lookup(kMixCycle[i % 4]).bytes;
    return total;
  }
  return lookup(tag.c_str()).bytes * length;
}

// ---------------------------------------------------------------------------
// Cell recording.
// ---------------------------------------------------------------------------

struct Cell {
  std::string tag;
  std::string mode;
  int length = 0;
  int graph_nodes = 0;
  double graph_instantiate_ms = 0.0;
  double flops = 0.0;
  double bytes = 0.0;
  std::vector<double> makespan_ms;      // device time for one chain or replay
  std::vector<double> host_ms;          // host submission time of the launch loop
  std::vector<int> host_before_sync;    // structural: 1 if closed before any sync
  std::vector<ClockSample> before;
  std::vector<ClockSample> after;
  std::vector<double> inner_kernel_ms;  // per-kernel, inner event pairs
};

std::vector<Cell> g_cells;

// One eager cell. The host interval closes before the single synchronization,
// which is the frozen construction that keeps host and device timing apart.
Cell measure_eager(const std::string& tag, int length) {
  Cell cell;
  cell.tag = tag;
  cell.mode = "eager";
  cell.length = length;
  cell.flops = chain_flops(tag, length);
  cell.bytes = chain_bytes(tag, length);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  issue_chain(tag, length);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  for (int block = 0; block < kBlocks; ++block) {
    const ClockSample before = clock_sample();
    issue_chain(tag, 1);  // priming launch, untimed
    CUDA_CHECK(cudaEventRecord(start, g_stream));
    const auto host_start = std::chrono::steady_clock::now();
    issue_chain(tag, length);
    const auto host_stop = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(stop, g_stream));
    // The host interval is already closed here, before the first sync call.
    cell.host_before_sync.push_back(1);
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const ClockSample after = clock_sample();
    cell.makespan_ms.push_back(ms);
    cell.host_ms.push_back(
        std::chrono::duration<double, std::milli>(host_stop - host_start).count());
    cell.before.push_back(before);
    cell.after.push_back(after);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return cell;
}

// One graph cell. The device makespan is one replay bracketed by events; the
// host submission cost is the loop that issues kGraphReplays replays, closed
// before any synchronization.
Cell measure_graph(const std::string& tag, int length, bool inner_events) {
  Cell cell;
  cell.tag = tag;
  cell.mode = inner_events ? "graph-instrumented" : "graph";
  cell.length = length;
  cell.flops = chain_flops(tag, length);
  cell.bytes = chain_bytes(tag, length);

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  std::vector<cudaEvent_t> inner;
  if (inner_events) {
    inner.resize(2 * static_cast<size_t>(length));
    for (auto& e : inner) CUDA_CHECK(cudaEventCreate(&e));
  }

  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  CUDA_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal));
  if (inner_events) {
    for (int i = 0; i < length; ++i) {
      CUDA_CHECK(cudaEventRecord(inner[2 * i], g_stream));
      if (tag == "mix") {
        issue(lookup(kMixCycle[i % 4]));
      } else {
        issue(lookup(tag.c_str()));
      }
      CUDA_CHECK(cudaEventRecord(inner[2 * i + 1], g_stream));
    }
  } else {
    issue_chain(tag, length);
  }
  CUDA_CHECK(cudaStreamEndCapture(g_stream, &graph));

  size_t node_count = 0;
  CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &node_count));
  cell.graph_nodes = static_cast<int>(node_count);

  const auto inst_start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  const auto inst_stop = std::chrono::steady_clock::now();
  cell.graph_instantiate_ms =
      std::chrono::duration<double, std::milli>(inst_stop - inst_start).count();

  for (int i = 0; i < 4; ++i) {
    CUDA_CHECK(cudaGraphLaunch(exec, g_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int block = 0; block < kBlocks; ++block) {
    const ClockSample before = clock_sample();

    // Device makespan of one replay, with the stream already busy.
    CUDA_CHECK(cudaGraphLaunch(exec, g_stream));
    CUDA_CHECK(cudaEventRecord(start, g_stream));
    CUDA_CHECK(cudaGraphLaunch(exec, g_stream));
    CUDA_CHECK(cudaEventRecord(stop, g_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cell.makespan_ms.push_back(ms);

    // Host submission cost of kGraphReplays replays, closed before any sync.
    const auto host_start = std::chrono::steady_clock::now();
    for (int i = 0; i < kGraphReplays; ++i) CUDA_CHECK(cudaGraphLaunch(exec, g_stream));
    const auto host_stop = std::chrono::steady_clock::now();
    cell.host_before_sync.push_back(1);
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    cell.host_ms.push_back(
        std::chrono::duration<double, std::milli>(host_stop - host_start).count());

    const ClockSample after = clock_sample();
    cell.before.push_back(before);
    cell.after.push_back(after);
  }

  if (inner_events) {
    CUDA_CHECK(cudaGraphLaunch(exec, g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    // Repair R6, post-specified and disclosed: an event recorded during stream
    // capture becomes a graph node whose elapsed time this driver refuses to
    // report, returning cudaErrorInvalidValue. The instrumented cells feed one
    // unscored record only, so a refusal is recorded as -1 and the run
    // continues rather than losing every scored measurement to it.
    for (int i = 0; i < length; ++i) {
      float ms = 0.f;
      const cudaError_t status =
          cudaEventElapsedTime(&ms, inner[2 * i], inner[2 * i + 1]);
      cell.inner_kernel_ms.push_back(status == cudaSuccess ? ms : -1.0);
      if (status != cudaSuccess) cudaGetLastError();
    }
    for (auto& e : inner) CUDA_CHECK(cudaEventDestroy(e));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return cell;
}

void emit_cell(Json& j, const Cell& cell) {
  j.sep();
  j.raw("{");
  j.str("tag", cell.tag);
  j.str("mode", cell.mode);
  j.integer("length", cell.length);
  j.integer("graph_nodes", cell.graph_nodes);
  j.integer("graph_replays", kGraphReplays);
  j.num("graph_instantiate_ms", cell.graph_instantiate_ms);
  j.num("flops", cell.flops);
  j.num("bytes", cell.bytes);
  j.open_arr("makespan_ms");
  for (double v : cell.makespan_ms) j.arr_num(v);
  j.close_arr();
  j.open_arr("host_ms");
  for (double v : cell.host_ms) j.arr_num(v);
  j.close_arr();
  j.open_arr("host_before_sync");
  for (int v : cell.host_before_sync) j.arr_num(v);
  j.close_arr();
  j.open_arr("inner_kernel_ms");
  for (double v : cell.inner_kernel_ms) j.arr_num(v);
  j.close_arr();
  j.open_arr("block_clocks");
  for (size_t i = 0; i < cell.makespan_ms.size(); ++i) {
    j.sep();
    j.raw("{");
    j.integer("sm_before", cell.before[i].sm_mhz);
    j.integer("sm_after", cell.after[i].sm_mhz);
    j.integer("mem_before", cell.before[i].mem_mhz);
    j.integer("mem_after", cell.after[i].mem_mhz);
    j.integer("th_before", static_cast<long long>(cell.before[i].throttle));
    j.integer("th_after", static_cast<long long>(cell.after[i].throttle));
    j.integer("pw_before", cell.before[i].power_mw);
    j.integer("tc_before", cell.before[i].temp_c);
    j.close_obj();
  }
  j.close_arr();
  j.close_obj();
}

double gemm_residual(const KernelSpec& spec) {
  gemm(spec.m, spec.n, spec.k);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  __nv_bfloat16 host = {};
  CUDA_CHECK(cudaMemcpy(&host, g_gc, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  const double expected = static_cast<double>(spec.k) * kFillA * kFillB;
  return std::fabs(__bfloat162float(host) - expected) / expected;
}

void preheat(double seconds) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<long long>(seconds * 1000.0));
  while (std::chrono::steady_clock::now() < deadline) {
    for (int i = 0; i < 8; ++i) gemm(4096, 4096, 4096);
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "graph_launch_result.json";
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

  CUDA_CHECK(cudaStreamCreate(&g_stream));
  CUBLAS_CHECK(cublasCreate(&g_blas));
  CUBLAS_CHECK(cublasSetStream(g_blas, g_stream));

  g_gemm_elems = 8192ull * 8192ull;
  CUDA_CHECK(cudaMalloc(&g_ga, g_gemm_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&g_gb, g_gemm_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&g_gc, g_gemm_elems * sizeof(__nv_bfloat16)));
  k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(g_ga, g_gemm_elems, kFillA);
  k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(g_gb, g_gemm_elems, kFillB);
  CUDA_CHECK(cudaDeviceSynchronize());

  // GG6: correctness of every GEMM in the set, outside every timed region.
  Json j;
  j.raw("{");
  j.str("study", "a100_graph_launch_v1");
  j.integer("stage", 2);
  j.integer("visible_device_count", device_count);
  j.str("device_name", prop.name);
  j.str("gpu_uuid", g_uuid);
  j.integer("sm_count", prop.multiProcessorCount);
  j.integer("l2_bytes", prop.l2CacheSize);
  j.integer("driver_api_version", driver_version);
  j.integer("runtime_api_version", runtime_version);
  j.integer("blocks_per_cell", kBlocks);
  j.integer("graph_replays", kGraphReplays);
  j.open_arr("correctness");
  for (const KernelSpec& spec : kSet) {
    if (!spec.is_gemm) continue;
    j.sep();
    j.raw("{");
    j.str("tag", spec.tag);
    j.num("residual", gemm_residual(spec));
    j.close_obj();
  }
  j.close_arr();

  preheat(3.0);

  const int lengths[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
  const char* tags[] = {"nop", "g1", "g2", "g4", "mix"};

  for (const char* tag : tags) {
    for (int length : lengths) {
      // The 90 microsecond kernel at length 256 would run for 23 ms per
      // replay; that is affordable, but the mixed chain repeats it too, so
      // both stay in the sweep and the wall limit covers them.
      g_cells.push_back(measure_eager(tag, length));
      std::fprintf(stderr, "[cell] eager %-4s K=%-4d\n", tag, length);
      g_cells.push_back(measure_graph(tag, length, false));
      std::fprintf(stderr, "[cell] graph %-4s K=%-4d nodes=%d\n", tag, length,
                   g_cells.back().graph_nodes);
    }
  }

  // Instrumented graph cells, for the reserved device-gap seed only. One
  // chain length per kernel keeps the event count bounded.
  for (const char* tag : tags) {
    g_cells.push_back(measure_graph(tag, 64, true));
    std::fprintf(stderr, "[cell] graph-instrumented %-4s K=64 nodes=%d\n", tag,
                 g_cells.back().graph_nodes);
  }

  j.open_arr("cells");
  for (const Cell& cell : g_cells) emit_cell(j, cell);
  j.close_arr();
  j.close_obj();

  FILE* f = std::fopen(out_path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "FATAL cannot write %s\n", out_path.c_str());
    return 22;
  }
  std::fwrite(j.body.data(), 1, j.body.size(), f);
  std::fputc('\n', f);
  std::fclose(f);
  std::fprintf(stderr, "[stage2] wrote %s with %zu cells (%zu bytes)\n", out_path.c_str(),
               g_cells.size(), j.body.size());
  return 0;
}
