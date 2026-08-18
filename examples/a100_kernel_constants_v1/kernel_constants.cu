// A100 kernel constants v1: the stage-1 measurement harness.
//
// It measures the clock-conditioned warm-state service constant of one kernel
// cell at a time, over five lanes, in the exact protocol frozen in
// expectations.md. It judges nothing: it produces series and identities, and
// score_expectations.py evaluates them against the freeze.
//
// Protocol per cell, frozen:
//   - one untimed shape-priming launch, then K = 20 discarded repetitions;
//   - a single-launch probe whose elapsed time sizes the batch G as the
//     smallest power of two with G * t_probe >= 200 us, capped at 256;
//   - 12 timed batches of G back-to-back launches in one stream, each batch
//     bracketed by CUDA events, with the NVML SM clock, memory clock and
//     throttle word read immediately before and immediately after;
//   - one diagnostic chain of R = 64 launches with one event per repetition;
//   - for cuBLAS cells, one correctness sample outside every timed region.
//
// Lane 1 runs first because every memory-bound expectation in lanes 2 to 5 is
// stated against the HBM roof it measures.

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

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t st_ = (expr);                                              \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "FATAL cublas %s at %s:%d: %d\n", #expr, __FILE__, \
                   __LINE__, static_cast<int>(st_));                          \
      std::exit(21);                                                          \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Frozen protocol constants.
// ---------------------------------------------------------------------------

constexpr int kWarmupDiscard = 20;
constexpr int kBatches = 12;
constexpr double kBatchMinUs = 200.0;
constexpr int kBatchCap = 256;
constexpr int kChainReps = 64;
constexpr int kBlockThreads = 256;
constexpr int kGridBlocks = 864;
constexpr long long kL2Bytes = 41943040LL;

// Fill values chosen so every cuBLAS product is exact in BF16 and FP32:
// 0.5 times 0.25 is 0.125, and the K-length sum is K * 0.125.
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
// Streaming kernels.
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

__global__ void k_triad(const float4* __restrict__ a, const float4* __restrict__ b,
                        float4* __restrict__ c, size_t n, float s) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    const float4 x = a[i];
    const float4 y = b[i];
    c[i] = make_float4(x.x + s * y.x, x.y + s * y.y, x.z + s * y.z, x.w + s * y.w);
  }
}

__global__ void k_scale(float4* __restrict__ p, size_t n, float s) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    float4 v = p[i];
    v.x *= s;
    v.y *= s;
    v.z *= s;
    v.w *= s;
    p[i] = v;
  }
}

__global__ void k_add(const float4* __restrict__ a, const float4* __restrict__ b,
                      float4* __restrict__ c, size_t n) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    const float4 x = a[i];
    const float4 y = b[i];
    c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

// Two-pass RMS normalization over rows of `width` floats. The activation is
// read twice and written once, which is the byte model the freeze states.
__global__ void k_rmsnorm(const float* __restrict__ src, const float* __restrict__ weight,
                          float* __restrict__ dst, int width) {
  const int row = blockIdx.x;
  const float* in = src + static_cast<size_t>(row) * width;
  float* out = dst + static_cast<size_t>(row) * width;
  __shared__ float reduction[kBlockThreads];
  float partial = 0.f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = in[i];
    partial += v * v;
  }
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) reduction[threadIdx.x] += reduction[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(reduction[0] / width + 1e-6f);
  for (int i = threadIdx.x; i < width; i += blockDim.x) out[i] = in[i] * inv * weight[i];
}

__global__ void k_nop() {}

__global__ void k_fill_bf16(__nv_bfloat16* p, size_t n, float v) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const __nv_bfloat16 h = __float2bfloat16(v);
  for (; i < n; i += stride) p[i] = h;
}

__global__ void k_fill_f32(float* p, size_t n, float v) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) p[i] = v;
}

// ---------------------------------------------------------------------------
// Decode attention: one block per (batch, query head), warps striding over the
// cache length with an online softmax. Head size must be a multiple of 32.
// ---------------------------------------------------------------------------

__global__ void k_decode_attn(const __nv_bfloat16* __restrict__ q,
                              const __nv_bfloat16* __restrict__ kcache,
                              const __nv_bfloat16* __restrict__ vcache,
                              __nv_bfloat16* __restrict__ out, int heads_q, int heads_kv,
                              int length, int head_size, float scale) {
  extern __shared__ float shared[];
  const int bh = blockIdx.x;
  const int b = bh / heads_q;
  const int h = bh % heads_q;
  const int hkv = h / (heads_q / heads_kv);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int per_lane = head_size / 32;

  float* qs = shared;                        // head_size floats
  float* wm = shared + head_size;            // warps floats
  float* wl = wm + warps;                    // warps floats
  float* wacc = wl + warps;                  // warps * head_size floats

  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    qs[i] = __bfloat162float(q[static_cast<size_t>(bh) * head_size + i]);
  }
  __syncthreads();

  // Repair R5: four independent online-softmax accumulators per warp, so four
  // cache rows are in flight at once and the walk is not held by one load
  // latency at a time.
  constexpr int kWays = 4;
  float qreg[8];
  float acc[kWays][8];
  float run_max[kWays];
  float run_sum[kWays];
  for (int e = 0; e < per_lane; ++e) qreg[e] = qs[e * 32 + lane];
  for (int w = 0; w < kWays; ++w) {
    run_max[w] = -INFINITY;
    run_sum[w] = 0.f;
    for (int e = 0; e < per_lane; ++e) acc[w][e] = 0.f;
  }

  const size_t kv_base =
      (static_cast<size_t>(b) * heads_kv + hkv) * static_cast<size_t>(length) * head_size;
  const int step = warps * kWays;
  for (int base = warp * kWays; base < length; base += step) {
    float dot[kWays];
    float vreg[kWays][8];
#pragma unroll
    for (int w = 0; w < kWays; ++w) dot[w] = 0.f;
#pragma unroll
    for (int w = 0; w < kWays; ++w) {
      const int t = base + w;
      if (t >= length) continue;
      const __nv_bfloat16* krow = kcache + kv_base + static_cast<size_t>(t) * head_size;
      const __nv_bfloat16* vrow = vcache + kv_base + static_cast<size_t>(t) * head_size;
      for (int e = 0; e < per_lane; ++e) {
        dot[w] += qreg[e] * __bfloat162float(krow[e * 32 + lane]);
        vreg[w][e] = __bfloat162float(vrow[e * 32 + lane]);
      }
    }
#pragma unroll
    for (int w = 0; w < kWays; ++w) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        dot[w] += __shfl_xor_sync(0xffffffffu, dot[w], offset);
      }
    }
#pragma unroll
    for (int w = 0; w < kWays; ++w) {
      if (base + w >= length) continue;
      const float score = dot[w] * scale;
      const float new_max = fmaxf(run_max[w], score);
      const float correction = __expf(run_max[w] - new_max);
      const float p = __expf(score - new_max);
      run_sum[w] = run_sum[w] * correction + p;
      for (int e = 0; e < per_lane; ++e) {
        acc[w][e] = acc[w][e] * correction + p * vreg[w][e];
      }
      run_max[w] = new_max;
    }
  }

  // Merge the four ways inside the warp before the cross-warp merge.
  float warp_max = -INFINITY;
  for (int w = 0; w < kWays; ++w) warp_max = fmaxf(warp_max, run_max[w]);
  float warp_sum = 0.f;
  float merged[8];
  for (int e = 0; e < per_lane; ++e) merged[e] = 0.f;
  for (int w = 0; w < kWays; ++w) {
    const float weight = __expf(run_max[w] - warp_max);
    warp_sum += run_sum[w] * weight;
    for (int e = 0; e < per_lane; ++e) merged[e] += acc[w][e] * weight;
  }

  if (lane == 0) {
    wm[warp] = warp_max;
    wl[warp] = warp_sum;
  }
  for (int e = 0; e < per_lane; ++e) wacc[warp * head_size + e * 32 + lane] = merged[e];
  __syncthreads();

  if (warp == 0) {
    float global_max = -INFINITY;
    for (int w = 0; w < warps; ++w) global_max = fmaxf(global_max, wm[w]);
    float total = 0.f;
    for (int w = 0; w < warps; ++w) total += wl[w] * __expf(wm[w] - global_max);
    const float inv = 1.0f / fmaxf(total, 1e-20f);
    for (int e = 0; e < per_lane; ++e) {
      float value = 0.f;
      for (int w = 0; w < warps; ++w) {
        value += wacc[w * head_size + e * 32 + lane] * __expf(wm[w] - global_max);
      }
      out[static_cast<size_t>(bh) * head_size + e * 32 + lane] =
          __float2bfloat16(value * inv);
    }
  }
}

// ---------------------------------------------------------------------------
// Cell recording.
// ---------------------------------------------------------------------------

cudaStream_t g_stream;
cublasHandle_t g_blas;
cudaEvent_t g_ev_a, g_ev_b;

struct Cell {
  std::string id;
  std::string lane;
  std::string family;
  std::string arm;
  long long m = 0, n = 0, k = 0, size_bytes = 0, rotate = 1, batch_count = 0, length = 0;
  double flops = 0.0;
  double total_bytes = 0.0;
  double distinct_bytes = 0.0;
  int group = 0;
  double probe_ms = 0.0;
  double correctness_residual = -1.0;
  std::vector<double> batch_ms;
  std::vector<double> batch_host_ms;  // repair R3
  std::vector<ClockSample> before;
  std::vector<ClockSample> after;
  std::vector<double> chain_ms;
  ClockSample chain_before;
  ClockSample chain_after;
};

std::vector<Cell> g_cells;

void emit_clocks(Json& j, const char* name, const ClockSample& s) {
  j.open_obj(name);
  j.integer("valid", s.valid);
  j.integer("sm", s.sm_mhz);
  j.integer("mem", s.mem_mhz);
  j.integer("pw", s.power_mw);
  j.integer("tc", s.temp_c);
  j.integer("th", static_cast<long long>(s.throttle));
  j.close_obj();
}

using Launch = std::function<void(int)>;

// Runs the frozen protocol for one cell. `launch(i)` issues repetition i.
void measure(Cell& cell, const Launch& launch) {
  // Shape priming plus the frozen warmup discard.
  launch(0);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  for (int i = 0; i < kWarmupDiscard; ++i) launch(i);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));

  // Probe one launch to size the batch.
  CUDA_CHECK(cudaEventRecord(g_ev_a, g_stream));
  launch(0);
  CUDA_CHECK(cudaEventRecord(g_ev_b, g_stream));
  CUDA_CHECK(cudaEventSynchronize(g_ev_b));
  float probe_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&probe_ms, g_ev_a, g_ev_b));
  cell.probe_ms = probe_ms;
  int group = 1;
  while (group < kBatchCap && group * probe_ms * 1000.0 < kBatchMinUs) group *= 2;
  cell.group = group;

  // Twelve timed batches of `group` back-to-back launches.
  //
  // Repair R2: one untimed priming launch precedes the opening event, so the
  // timed region begins with the stream already executing and measures
  // back-to-back kernels rather than one post-synchronization launch.
  // Repair R3: the host wall time of the launch loop is recorded before any
  // synchronization, so a host-issue-bound cell is identifiable.
  int issued = 0;
  for (int batch = 0; batch < kBatches; ++batch) {
    const ClockSample before = clock_sample();
    launch(issued++);
    CUDA_CHECK(cudaEventRecord(g_ev_a, g_stream));
    const auto host_start = std::chrono::steady_clock::now();
    for (int i = 0; i < group; ++i) launch(issued++);
    const auto host_stop = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(g_ev_b, g_stream));
    CUDA_CHECK(cudaEventSynchronize(g_ev_b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_ev_a, g_ev_b));
    const ClockSample after = clock_sample();
    cell.batch_ms.push_back(ms);
    cell.batch_host_ms.push_back(
        std::chrono::duration<double, std::milli>(host_stop - host_start).count());
    cell.before.push_back(before);
    cell.after.push_back(after);
  }
  cell.batch_count = kBatches;

  // Diagnostic chain: one event per repetition, no host synchronization
  // inside the chain.
  std::vector<cudaEvent_t> chain(kChainReps + 1);
  for (auto& e : chain) CUDA_CHECK(cudaEventCreate(&e));
  cell.chain_before = clock_sample();
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  CUDA_CHECK(cudaEventRecord(chain[0], g_stream));
  for (int i = 0; i < kChainReps; ++i) {
    launch(issued++);
    CUDA_CHECK(cudaEventRecord(chain[i + 1], g_stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  cell.chain_after = clock_sample();
  for (int i = 0; i < kChainReps; ++i) {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, chain[i], chain[i + 1]));
    cell.chain_ms.push_back(ms);
  }
  for (auto& e : chain) CUDA_CHECK(cudaEventDestroy(e));

  std::fprintf(stderr, "[cell] %-52s G=%-4d probe=%.4f ms sm=%u/%u\n", cell.id.c_str(),
               group, probe_ms, cell.before.front().sm_mhz, cell.after.back().sm_mhz);
}

void record(const Cell& cell) { g_cells.push_back(cell); }

// ---------------------------------------------------------------------------
// Repair R4: the instrumentation control. The same 64 launches are timed with
// one event every 1, 4, 16 and 64 repetitions, so the per-boundary cost of the
// event chain is measured rather than assumed. An event-only chain with no
// kernels between the events gives the bare event-record period.
// ---------------------------------------------------------------------------

struct ControlRow {
  std::string id;
  int stride;
  int reps;
  double per_kernel_ms;
};

std::vector<ControlRow> g_controls;
double g_event_only_period_ms = 0.0;

void instrumentation_control(const std::string& id, const Launch& launch) {
  const int reps = 64;
  for (int stride : {1, 4, 16, 64}) {
    for (int i = 0; i < 16; ++i) launch(i);
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    const int marks = reps / stride;
    std::vector<cudaEvent_t> ev(marks + 1);
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
    launch(0);  // prime, as in the batch protocol
    CUDA_CHECK(cudaEventRecord(ev[0], g_stream));
    for (int mark = 0; mark < marks; ++mark) {
      for (int s = 0; s < stride; ++s) launch(mark * stride + s);
      CUDA_CHECK(cudaEventRecord(ev[mark + 1], g_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    float total_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev[0], ev[marks]));
    for (auto& e : ev) CUDA_CHECK(cudaEventDestroy(e));
    g_controls.push_back({id, stride, reps, total_ms / reps});
  }
  std::fprintf(stderr, "[control] %s done\n", id.c_str());
}

void event_only_control() {
  const int marks = 64;
  std::vector<cudaEvent_t> ev(marks + 1);
  for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));
  k_nop<<<1, 1, 0, g_stream>>>();
  CUDA_CHECK(cudaEventRecord(ev[0], g_stream));
  for (int mark = 0; mark < marks; ++mark) CUDA_CHECK(cudaEventRecord(ev[mark + 1], g_stream));
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  float total_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev[0], ev[marks]));
  for (auto& e : ev) CUDA_CHECK(cudaEventDestroy(e));
  g_event_only_period_ms = total_ms / marks;
}

// ---------------------------------------------------------------------------
// GEMM helpers.
// ---------------------------------------------------------------------------

__nv_bfloat16 *g_ga = nullptr, *g_gb = nullptr, *g_gc = nullptr;
size_t g_gemm_elems = 0;

// Repair R1: the engine-natural layout. A serving engine holds a fixed weight
// matrix and varies the token count, so it issues Out[N,M] = W[N,K] * X[K,M]
// in column-major with leading dimensions N, K and N. No leading dimension
// depends on the token count M, which is what the first run got wrong.
void gemm_nn(int m, int n, int k) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, g_ga,
                            CUDA_R_16BF, n, g_gb, CUDA_R_16BF, k, &beta, g_gc,
                            CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Checks C[0] and C[last] against the exact expected K * 0.125.
double gemm_residual(int m, int n, int k) {
  gemm_nn(m, n, k);
  CUDA_CHECK(cudaStreamSynchronize(g_stream));
  __nv_bfloat16 host[2] = {};
  CUDA_CHECK(cudaMemcpy(&host[0], g_gc, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  const size_t last = static_cast<size_t>(m) * n - 1;
  CUDA_CHECK(cudaMemcpy(&host[1], g_gc + last, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  const double expected = static_cast<double>(k) * kFillA * kFillB;
  double worst = 0.0;
  for (int i = 0; i < 2; ++i) {
    const double got = __bfloat162float(host[i]);
    worst = std::max(worst, std::fabs(got - expected) / expected);
  }
  return worst;
}

void fill_gemm_operands() {
  k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(g_ga, g_gemm_elems, kFillA);
  k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(g_gb, g_gemm_elems, kFillB);
  CUDA_CHECK(cudaDeviceSynchronize());
}

double gemm_bytes(long long m, long long n, long long k) {
  return 2.0 * (static_cast<double>(m) * k + static_cast<double>(k) * n +
                static_cast<double>(m) * n);
}

Cell gemm_cell(const std::string& id, const std::string& lane, const std::string& family,
               const std::string& arm, int m, int n, int k, const std::string& split) {
  Cell cell;
  cell.id = id;
  cell.lane = lane;
  cell.family = family + ":" + split;
  cell.arm = arm;
  cell.m = m;
  cell.n = n;
  cell.k = k;
  cell.flops = 2.0 * m * n * k;
  cell.total_bytes = gemm_bytes(m, n, k);
  cell.distinct_bytes = cell.total_bytes;
  cell.correctness_residual = gemm_residual(m, n, k);
  measure(cell, [m, n, k](int) { gemm_nn(m, n, k); });
  return cell;
}

// ---------------------------------------------------------------------------
// Grid construction, exactly as frozen.
// ---------------------------------------------------------------------------

const double kGridFactors[] = {0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15,
                               1.30, 1.50, 1.80, 2.20, 3.00, 4.00};
const double kHoldoutFactors[] = {0.32, 0.62, 0.92, 1.07, 1.22, 1.65, 2.60};
const int kFixedM[] = {1, 2, 4, 8, 16, 32, 64, 1024, 2048, 4096, 8192};

double knee_nameplate(int n, int k) {
  const double peak = 108.0 * 1410e6 * 2048.0;
  const double roof = 1593e6 * 2.0 * 5120.0 / 8.0;
  return peak * k * n / (static_cast<double>(n) * k * roof - peak * (k + n));
}

std::vector<int> grid_for(int n, int k) {
  const double knee = knee_nameplate(n, k);
  std::vector<int> values(std::begin(kFixedM), std::end(kFixedM));
  for (double f : kGridFactors) values.push_back(static_cast<int>(std::lround(knee * f)));
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<int> holdout_for(int n, int k) {
  const double knee = knee_nameplate(n, k);
  const std::vector<int> grid = grid_for(n, k);
  std::vector<int> values;
  for (double f : kHoldoutFactors) {
    const int m = static_cast<int>(std::lround(knee * f));
    if (std::find(grid.begin(), grid.end(), m) == grid.end() &&
        std::find(values.begin(), values.end(), m) == values.end()) {
      values.push_back(m);
    }
  }
  std::sort(values.begin(), values.end());
  return values;
}

// ---------------------------------------------------------------------------
// Emission.
// ---------------------------------------------------------------------------

void emit_cell(Json& j, const Cell& cell) {
  j.sep();
  j.raw("{");
  j.str("id", cell.id);
  j.str("lane", cell.lane);
  j.str("family", cell.family);
  j.str("arm", cell.arm);
  j.integer("m", cell.m);
  j.integer("n", cell.n);
  j.integer("k", cell.k);
  j.integer("length", cell.length);
  j.integer("size_bytes", cell.size_bytes);
  j.integer("rotate", cell.rotate);
  j.integer("group", cell.group);
  j.integer("batches", cell.batch_count);
  j.num("flops", cell.flops);
  j.num("total_bytes", cell.total_bytes);
  j.num("distinct_bytes", cell.distinct_bytes);
  j.num("probe_ms", cell.probe_ms);
  j.num("correctness_residual", cell.correctness_residual);
  j.open_arr("batch_ms");
  for (double v : cell.batch_ms) j.arr_num(v);
  j.close_arr();
  j.open_arr("batch_host_ms");
  for (double v : cell.batch_host_ms) j.arr_num(v);
  j.close_arr();
  j.open_arr("batch_clocks");
  for (size_t i = 0; i < cell.batch_ms.size(); ++i) {
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
  j.open_arr("chain_ms");
  for (double v : cell.chain_ms) j.arr_num(v);
  j.close_arr();
  emit_clocks(j, "chain_before", cell.chain_before);
  emit_clocks(j, "chain_after", cell.chain_after);
  j.close_obj();
}

void preheat(double seconds) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<long long>(seconds * 1000.0));
  while (std::chrono::steady_clock::now() < deadline) {
    for (int i = 0; i < 8; ++i) gemm_nn(4096, 4096, 4096);
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_path = "kernel_constants_result.json";
  std::string arm = "boosted";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    if (std::strcmp(argv[i], "--arm") == 0 && i + 1 < argc) arm = argv[++i];
  }
  const bool boosted = (arm == "boosted");

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
  CUDA_CHECK(cudaEventCreate(&g_ev_a));
  CUDA_CHECK(cudaEventCreate(&g_ev_b));

  g_gemm_elems = 8192ull * 8192ull;
  CUDA_CHECK(cudaMalloc(&g_ga, g_gemm_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&g_gb, g_gemm_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&g_gc, g_gemm_elems * sizeof(__nv_bfloat16)));
  fill_gemm_operands();

  auto idle = [&](double seconds) {
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<long long>(seconds * 1000.0)));
  };

  if (boosted) preheat(3.0);

  // Instrumentation control (repair R4), boosted arm only. It runs before the
  // lanes so its cost cannot be attributed to a lane cell.
  if (boosted) {
    float4* ctrl_pool = nullptr;
    const size_t ctrl_bytes = 64ull * 1024ull * 1024ull;
    CUDA_CHECK(cudaMalloc(&ctrl_pool, ctrl_bytes));
    CUDA_CHECK(cudaMemset(ctrl_pool, 0x3f, ctrl_bytes));
    const size_t ctrl_n4 = ctrl_bytes / sizeof(float4);
    event_only_control();
    instrumentation_control("ctrl_empty_kernel",
                            [](int) { k_nop<<<1, 1, 0, g_stream>>>(); });
    instrumentation_control("ctrl_gemm_G1_m64", [](int) { gemm_nn(64, 2048, 1024); });
    instrumentation_control("ctrl_gemm_G4_m1", [](int) { gemm_nn(1, 8192, 8192); });
    instrumentation_control("ctrl_gemm_G2_m1024", [](int) { gemm_nn(1024, 1024, 1024); });
    instrumentation_control("ctrl_scale_64mib", [ctrl_pool, ctrl_n4](int) {
      k_scale<<<kGridBlocks, kBlockThreads, 0, g_stream>>>(ctrl_pool, ctrl_n4, 1.0000001f);
    });
    CUDA_CHECK(cudaFree(ctrl_pool));
  }

  // -------------------------------------------------------------------------
  // Lane 1, the measured HBM roof. Runs first.
  // -------------------------------------------------------------------------
  {
    const size_t max_bytes = 2048ull * 1024ull * 1024ull;
    float4 *a = nullptr, *b = nullptr, *c = nullptr;
    float* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&a, max_bytes));
    CUDA_CHECK(cudaMalloc(&b, max_bytes));
    CUDA_CHECK(cudaMalloc(&c, max_bytes));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    CUDA_CHECK(cudaMemset(a, 0x3f, max_bytes));
    CUDA_CHECK(cudaMemset(b, 0x3f, max_bytes));
    CUDA_CHECK(cudaMemset(c, 0x3f, max_bytes));

    const size_t sizes_mib[] = {256, 512, 1024, 2048};
    const char* kinds[] = {"read", "write", "copy", "triad"};
    for (size_t mib : sizes_mib) {
      const size_t bytes = mib * 1024ull * 1024ull;
      const size_t n4 = bytes / sizeof(float4);
      for (const char* kind : kinds) {
        if (!boosted && mib != 1024) continue;
        Cell cell;
        cell.id = std::string("hbm_") + kind + "_" + std::to_string(mib) + "mib";
        cell.lane = "1";
        cell.family = std::string("hbm_") + kind;
        cell.arm = arm;
        cell.size_bytes = static_cast<long long>(bytes);
        cell.flops = 0.0;
        if (std::strcmp(kind, "read") == 0) {
          cell.total_bytes = static_cast<double>(bytes);
          cell.distinct_bytes = static_cast<double>(bytes);
          if (!boosted) idle(3.0);
          measure(cell, [&, n4](int) {
            k_read<<<kGridBlocks, kBlockThreads, 0, g_stream>>>(a, n4, sink);
          });
        } else if (std::strcmp(kind, "write") == 0) {
          cell.total_bytes = static_cast<double>(bytes);
          cell.distinct_bytes = static_cast<double>(bytes);
          if (!boosted) idle(3.0);
          measure(cell, [&, n4](int) {
            k_write<<<kGridBlocks, kBlockThreads, 0, g_stream>>>(b, n4, 1.5f);
          });
        } else if (std::strcmp(kind, "copy") == 0) {
          cell.total_bytes = 2.0 * bytes;
          cell.distinct_bytes = 2.0 * bytes;
          if (!boosted) idle(3.0);
          measure(cell, [&, n4](int) {
            k_copy<<<kGridBlocks, kBlockThreads, 0, g_stream>>>(a, b, n4);
          });
        } else {
          cell.total_bytes = 3.0 * bytes;
          cell.distinct_bytes = 3.0 * bytes;
          if (!boosted) idle(3.0);
          measure(cell, [&, n4](int) {
            k_triad<<<kGridBlocks, kBlockThreads, 0, g_stream>>>(a, b, c, n4, 1.5f);
          });
        }
        record(cell);
      }
    }
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(sink));
  }

  // -------------------------------------------------------------------------
  // Lane 2, dense GEMM over the knee-anchored grid.
  // -------------------------------------------------------------------------
  struct Family {
    const char* id;
    int n, k;
  };
  const Family families[] = {
      {"G1", 2048, 1024}, {"G2", 1024, 1024}, {"G3", 1024, 512},
      {"G4", 8192, 8192}, {"G5", 4096, 4096},
  };
  const int base_arm_g4[] = {1, 64, 256, 1024};
  const int base_arm_g1[] = {32, 512};

  // The BASE arm measures exactly the frozen subset. Every one of its shapes is
  // also measured in the BOOSTED arm so the two-arm ratio expectations have a
  // pair; a BASE shape that is not a grid knot is tagged "pair" so it enters
  // neither the interpolation basis nor the held-out set.
  auto base_lane2 = [&](const Family& f) {
    std::vector<int> values;
    if (std::strcmp(f.id, "G4") == 0)
      values.assign(std::begin(base_arm_g4), std::end(base_arm_g4));
    if (std::strcmp(f.id, "G1") == 0)
      values.assign(std::begin(base_arm_g1), std::end(base_arm_g1));
    return values;
  };

  for (const Family& f : families) {
    const std::vector<int> grid = grid_for(f.n, f.k);
    const std::vector<int> base_values = base_lane2(f);
    if (!boosted) {
      for (int m : base_values) {
        if (static_cast<size_t>(m) * std::max(f.n, f.k) > g_gemm_elems) continue;
        idle(3.0);
        const bool on_grid = std::find(grid.begin(), grid.end(), m) != grid.end();
        record(gemm_cell(std::string("gemm_") + f.id + "_m" + std::to_string(m), "2", f.id,
                         arm, m, f.n, f.k, on_grid ? "grid" : "pair"));
      }
      continue;
    }
    for (int m : grid) {
      if (static_cast<size_t>(m) * std::max(f.n, f.k) > g_gemm_elems) continue;
      record(gemm_cell(std::string("gemm_") + f.id + "_m" + std::to_string(m), "2", f.id,
                       arm, m, f.n, f.k, "grid"));
    }
    for (int m : holdout_for(f.n, f.k)) {
      if (static_cast<size_t>(m) * std::max(f.n, f.k) > g_gemm_elems) continue;
      record(gemm_cell(std::string("gemm_") + f.id + "_holdout_m" + std::to_string(m), "2",
                       f.id, arm, m, f.n, f.k, "holdout"));
    }
    for (int m : base_values) {
      if (static_cast<size_t>(m) * std::max(f.n, f.k) > g_gemm_elems) continue;
      if (std::find(grid.begin(), grid.end(), m) != grid.end()) continue;
      record(gemm_cell(std::string("gemm_") + f.id + "_m" + std::to_string(m), "2", f.id,
                       arm, m, f.n, f.k, "pair"));
    }
  }

  // -------------------------------------------------------------------------
  // Lane 3, attention prefill and decode.
  // -------------------------------------------------------------------------
  {
    struct Geometry {
      const char* id;
      int heads, head_size;
    };
    const Geometry geometries[] = {{"granite", 16, 64}, {"synthetic", 64, 128}};
    const int seqs[] = {128, 256, 512, 1024, 2048, 4096};
    const size_t score_elems = 64ull * 4096ull * 4096ull;
    __nv_bfloat16 *qb = nullptr, *kb = nullptr, *sb = nullptr;
    CUDA_CHECK(cudaMalloc(&qb, 64ull * 4096ull * 128ull * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&kb, 64ull * 4096ull * 128ull * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&sb, score_elems * sizeof(__nv_bfloat16)));
    k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(qb, 64ull * 4096ull * 128ull, kFillA);
    k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(kb, 64ull * 4096ull * 128ull, kFillB);
    CUDA_CHECK(cudaDeviceSynchronize());

    const float alpha = 1.0f, beta = 0.0f;
    for (const Geometry& g : geometries) {
      for (int s : seqs) {
        if (!boosted && !(std::strcmp(g.id, "granite") == 0 && (s == 1024 || s == 4096)))
          continue;
        if (!boosted) idle(3.0);
        Cell cell;
        cell.id = std::string("attn_prefill_") + g.id + "_s" + std::to_string(s);
        cell.lane = "3";
        cell.family = std::string("attn_prefill_") + g.id;
        cell.arm = arm;
        cell.m = s;
        cell.n = g.heads;
        cell.k = g.head_size;
        cell.length = s;
        // score: [S,S] = Q^T[S,D] x K[D,S]; value: [S,D] = P[S,S] x V^T[S,D]
        const double score_flops = 2.0 * g.heads * s * s * g.head_size;
        const double value_flops = 2.0 * g.heads * s * g.head_size * s;
        cell.flops = score_flops + value_flops;
        const double qk_bytes = 2.0 * g.heads * (2.0 * s * g.head_size + 1.0 * s * s);
        const double pv_bytes = 2.0 * g.heads * (1.0 * s * s + 2.0 * s * g.head_size);
        cell.total_bytes = qk_bytes + pv_bytes;
        cell.distinct_bytes = cell.total_bytes;
        const int heads = g.heads, d = g.head_size;
        measure(cell, [&, heads, d, s](int) {
          CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              g_blas, CUBLAS_OP_T, CUBLAS_OP_N, s, s, d, &alpha, qb, CUDA_R_16BF, d,
              static_cast<long long>(s) * d, kb, CUDA_R_16BF, d,
              static_cast<long long>(s) * d, &beta, sb, CUDA_R_16BF, s,
              static_cast<long long>(s) * s, heads, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
          CUBLAS_CHECK(cublasGemmStridedBatchedEx(
              g_blas, CUBLAS_OP_N, CUBLAS_OP_T, s, d, s, &alpha, sb, CUDA_R_16BF, s,
              static_cast<long long>(s) * s, kb, CUDA_R_16BF, d,
              static_cast<long long>(s) * d, &beta, qb, CUDA_R_16BF, s,
              static_cast<long long>(s) * d, heads, CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT));
        });
        record(cell);
        k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(qb, 64ull * 4096ull * 128ull, kFillA);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }
    CUDA_CHECK(cudaFree(qb));
    CUDA_CHECK(cudaFree(kb));
    CUDA_CHECK(cudaFree(sb));

    // Decode attention.
    const int heads_q = 16, heads_kv = 8, head_size = 64;
    const int batches[] = {1, 4, 16, 64, 256};
    const int lengths[] = {128, 512, 2048, 8192};
    const size_t max_kv =
        256ull * heads_kv * 8192ull * head_size;  // elements per cache
    __nv_bfloat16 *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dq, 256ull * heads_q * head_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dk, max_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dv, max_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dout, 256ull * heads_q * head_size * sizeof(__nv_bfloat16)));
    k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(dq, 256ull * heads_q * head_size, 0.125f);
    k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(dk, max_kv, 0.125f);
    k_fill_bf16<<<kGridBlocks, kBlockThreads>>>(dv, max_kv, 0.25f);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int threads = 128;
    const int warps = threads / 32;
    const size_t shared = (head_size + 2 * warps + warps * head_size) * sizeof(float);
    for (int b : batches) {
      for (int l : lengths) {
        if (!boosted && !(b == 256 && (l == 2048 || l == 8192))) continue;
        if (!boosted) idle(3.0);
        Cell cell;
        cell.id = "attn_decode_b" + std::to_string(b) + "_l" + std::to_string(l);
        cell.lane = "3";
        cell.family = "attn_decode";
        cell.arm = arm;
        cell.m = b;
        cell.n = heads_q;
        cell.k = head_size;
        cell.length = l;
        const double kv_bytes = 2.0 * b * heads_kv * l * head_size * 2.0;
        cell.total_bytes = kv_bytes;
        cell.distinct_bytes = kv_bytes;
        cell.flops = 4.0 * b * heads_q * l * head_size;
        const int blocks = b * heads_q;
        measure(cell, [&, blocks, l](int) {
          k_decode_attn<<<blocks, threads, shared, g_stream>>>(
              dq, dk, dv, dout, heads_q, heads_kv, l, head_size, 0.125f);
        });
        record(cell);
      }
    }
    CUDA_CHECK(cudaFree(dq));
    CUDA_CHECK(cudaFree(dk));
    CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dout));
  }

  // -------------------------------------------------------------------------
  // Lane 4, MoE expert GEMM at captured expert loads.
  // -------------------------------------------------------------------------
  {
    const int loads[] = {1, 2, 4, 7, 11, 14, 18, 27, 54};
    struct Shape {
      const char* id;
      int n, k;
    };
    const Shape shapes[] = {{"expert_gate_up", 1024, 1024}, {"expert_down", 1024, 512}};
    for (const Shape& s : shapes) {
      for (int m : loads) {
        if (!boosted && !(m == 14 && std::strcmp(s.id, "expert_gate_up") == 0) &&
            !(m == 54 && std::strcmp(s.id, "expert_down") == 0))
          continue;
        if (!boosted) idle(3.0);
        record(gemm_cell(std::string("moe_") + s.id + "_m" + std::to_string(m), "4", s.id,
                         arm, m, s.n, s.k, "captured"));
      }
    }
  }

  // -------------------------------------------------------------------------
  // Lane 5, elementwise and normalization, warm and rotated.
  // -------------------------------------------------------------------------
  {
    const size_t pool_bytes = 4096ull * 1024ull * 1024ull;
    float4* pool = nullptr;
    float* weight = nullptr;
    CUDA_CHECK(cudaMalloc(&pool, pool_bytes));
    CUDA_CHECK(cudaMalloc(&weight, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMemset(pool, 0x3f, pool_bytes));
    k_fill_f32<<<1, kBlockThreads>>>(weight, 1024, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t sizes_mib[] = {4, 16, 64, 256};
    const char* kinds[] = {"scale", "add", "rmsnorm"};
    for (size_t mib : sizes_mib) {
      const size_t bytes = mib * 1024ull * 1024ull;
      const size_t n4 = bytes / sizeof(float4);
      // Rotation depth: enough distinct slots to exceed 8 x L2, and at least 2.
      const size_t needed =
          std::max<size_t>(2, (8ull * kL2Bytes + bytes - 1) / bytes);
      for (const char* kind : kinds) {
        const size_t slots_per_rep = (std::strcmp(kind, "scale") == 0) ? 1 : 3;
        const size_t max_slots = pool_bytes / (bytes * slots_per_rep);
        const size_t rotate_depth = std::min(needed, std::max<size_t>(1, max_slots));
        for (int rotated = 0; rotated < 2; ++rotated) {
          const size_t rotate = rotated ? rotate_depth : 1;
          if (rotated && rotate < 2) continue;
          if (!boosted && !(mib == 64 && rotated == 0 &&
                            (std::strcmp(kind, "scale") == 0 ||
                             std::strcmp(kind, "rmsnorm") == 0)))
            continue;
          if (!boosted) idle(3.0);
          Cell cell;
          cell.id = std::string("elem_") + kind + "_" + std::to_string(mib) + "mib_" +
                    (rotated ? "rot" : "warm");
          cell.lane = "5";
          cell.family = std::string("elem_") + kind;
          cell.arm = arm;
          cell.size_bytes = static_cast<long long>(bytes);
          cell.rotate = static_cast<long long>(rotate);
          cell.flops = 0.0;
          if (std::strcmp(kind, "scale") == 0) {
            cell.total_bytes = 2.0 * bytes;
            cell.distinct_bytes = static_cast<double>(bytes);
            measure(cell, [&, n4, rotate](int i) {
              const size_t slot = static_cast<size_t>(i) % rotate;
              k_scale<<<kGridBlocks, kBlockThreads, 0, g_stream>>>(pool + slot * n4, n4,
                                                                   1.0000001f);
            });
          } else if (std::strcmp(kind, "add") == 0) {
            cell.total_bytes = 3.0 * bytes;
            cell.distinct_bytes = 3.0 * bytes;
            measure(cell, [&, n4, rotate](int i) {
              const size_t slot = static_cast<size_t>(i) % rotate;
              k_add<<<kGridBlocks, kBlockThreads, 0, g_stream>>>(
                  pool + (3 * slot + 0) * n4, pool + (3 * slot + 1) * n4,
                  pool + (3 * slot + 2) * n4, n4);
            });
          } else {
            const int width = 1024;
            const size_t floats = bytes / sizeof(float);
            const int rows = static_cast<int>(floats / width);
            cell.total_bytes = 3.0 * bytes;
            cell.distinct_bytes = 2.0 * bytes;
            cell.n = width;
            measure(cell, [&, rows, width, floats, rotate](int i) {
              const size_t slot = static_cast<size_t>(i) % rotate;
              const float* src = reinterpret_cast<const float*>(pool) + (3 * slot) * floats;
              float* dst = reinterpret_cast<float*>(pool) + (3 * slot + 2) * floats;
              k_rmsnorm<<<rows, kBlockThreads, 0, g_stream>>>(src, weight, dst, width);
            });
          }
          record(cell);
        }
      }
    }
    CUDA_CHECK(cudaFree(pool));
    CUDA_CHECK(cudaFree(weight));
  }

  // -------------------------------------------------------------------------
  // Emit.
  // -------------------------------------------------------------------------
  Json j;
  j.raw("{");
  j.str("study", "a100_kernel_constants_v1");
  j.integer("stage", 1);
  j.str("arm", arm);
  j.integer("visible_device_count", device_count);
  j.str("device_name", prop.name);
  j.str("gpu_uuid", g_uuid);
  j.integer("sm_count", prop.multiProcessorCount);
  j.integer("l2_bytes", prop.l2CacheSize);
  j.integer("mem_clock_khz", prop.memoryClockRate);
  j.integer("sm_clock_khz", prop.clockRate);
  j.integer("mem_bus_bits", prop.memoryBusWidth);
  j.integer("compute_capability_major", prop.major);
  j.integer("compute_capability_minor", prop.minor);
  j.integer("driver_api_version", driver_version);
  j.integer("runtime_api_version", runtime_version);
  j.integer("warmup_discard", kWarmupDiscard);
  j.integer("batches_per_cell", kBatches);
  j.num("batch_min_us", kBatchMinUs);
  j.integer("batch_cap", kBatchCap);
  j.integer("chain_reps", kChainReps);
  j.integer("grid_blocks", kGridBlocks);
  j.integer("block_threads", kBlockThreads);
  j.num("event_only_period_ms", g_event_only_period_ms);
  j.open_arr("instrumentation_control");
  for (const ControlRow& row : g_controls) {
    j.sep();
    j.raw("{");
    j.str("id", row.id);
    j.integer("stride", row.stride);
    j.integer("reps", row.reps);
    j.num("per_kernel_ms", row.per_kernel_ms);
    j.close_obj();
  }
  j.close_arr();
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
  std::fprintf(stderr, "[stage1] wrote %s with %zu cells (%zu bytes)\n", out_path.c_str(),
               g_cells.size(), j.body.size());
  return 0;
}
