// Merlin fabric flow capture: the scored capture lane.
//
// Star of long-running greedy flows: N source ranks each stream fixed-size
// chunks to one destination rank, every flow on its own two-rank NCCL
// communicator, its own CUDA stream, and (at the destination) its own host
// thread, so flows progress independently and contend only in the host
// stacks and on the fabric. Join offsets stagger flow starts against a
// common barrier epoch. Per-chunk completion timestamps on the destination
// clock are the dataset.
//
// The frozen substrate, cells, guards and relations live in expectations.md
// (freeze 1). The per-chunk loop below is byte for byte the loop the
// discovery jitter ladder measured: post, synchronize, 4-byte probe
// readback and sequence check (destination), then the CLOCK_MONOTONIC_RAW
// timestamp, so the tracer cost inside each chunk is exactly the measured
// floor. Every rank re-runs the no-network tracer floor block before the
// window opens.
//
// Rank bootstrap is the wave-16 pattern: ncclUniqueId files on the shared
// filesystem, ranks from Slurm, no MPI. CAPT_RANK / CAPT_WORLD /
// CAPT_LOCALID overrides exist for the heterogeneous mixed cell.

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

void sleep_until_raw(long long target_ns) {
  for (;;) {
    const long long now = now_ns(CLOCK_MONOTONIC_RAW);
    if (now >= target_ns) return;
    const long long left = target_ns - now;
    if (left > 2000000LL) {
      usleep(static_cast<useconds_t>((left - 1000000LL) / 1000));
    } else {
      // Final approach: spin so the start lands within microseconds.
    }
  }
}

__global__ void k_fill_float(float* p, size_t n, float value) {
  size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) p[i] = value;
}

__global__ void k_write_probe(float* p, size_t mid, size_t last, float value) {
  p[0] = value;
  p[mid] = value;
  p[last] = value;
}

double percentile(const std::vector<double>& sorted, double q) {
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

void share_unique_id(ncclUniqueId* id, bool writer, const std::string& path,
                     int poll_secs) {
  if (writer) {
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

std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t comma = text.find(',', pos);
    if (comma == std::string::npos) comma = text.size();
    out.push_back(std::atoi(text.substr(pos, comma - pos).c_str()));
    pos = comma + 1;
  }
  return out;
}

std::vector<double> parse_double_list(const std::string& text) {
  std::vector<double> out;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t comma = text.find(',', pos);
    if (comma == std::string::npos) comma = text.size();
    out.push_back(std::atof(text.substr(pos, comma - pos).c_str()));
    pos = comma + 1;
  }
  return out;
}

struct FloorStats {
  double p5, p50, p95, p99, maxv, mean;
};

FloorStats measure_tracer_floor(cudaStream_t stream, float* dev_buf,
                                float* host_probe) {
  const int reps = 1000;
  const size_t elems = 1024;
  std::vector<double> t_us;
  t_us.reserve(reps);
  for (int i = 0; i < 20; ++i) {
    k_write_probe<<<1, 1, 0, stream>>>(dev_buf, elems / 2, elems - 1, 1.0f);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  for (int i = 0; i < reps; ++i) {
    const long long t0 = now_ns(CLOCK_MONOTONIC_RAW);
    k_write_probe<<<1, 1, 0, stream>>>(dev_buf, elems / 2, elems - 1,
                                       static_cast<float>(i));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(host_probe, dev_buf, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (*host_probe != static_cast<float>(i)) {
      std::fprintf(stderr, "FATAL tracer floor probe mismatch\n");
      std::exit(27);
    }
    const long long t1 = now_ns(CLOCK_MONOTONIC_RAW);
    t_us.push_back(static_cast<double>(t1 - t0) / 1000.0);
  }
  std::sort(t_us.begin(), t_us.end());
  double sum = 0.0;
  for (double x : t_us) sum += x;
  return FloorStats{percentile(t_us, 0.05), percentile(t_us, 0.50),
                    percentile(t_us, 0.95), percentile(t_us, 0.99),
                    t_us.back(), sum / t_us.size()};
}

// One flow's bookkeeping at the destination.
struct DestFlow {
  int flow_id = 0;
  int source_rank = 0;
  int device = 0;
  double offset_s = 0.0;
  ncclComm_t comm = nullptr;
  cudaStream_t stream{};
  float* buf[2] = {nullptr, nullptr};
  float* host_probe = nullptr;
  size_t chunk_bytes = 0;
  long long t0_raw = 0;
  double window_s = 0.0;
  std::vector<long long> completions_ns;  // relative to t0_raw
  long long sentinel_ns = -1;
  long long data_chunks = 0;
  int mismatches = 0;
  int final_probe_mismatches = 0;
};

void* dest_flow_thread(void* arg) {
  DestFlow* f = static_cast<DestFlow*>(arg);
  CUDA_CHECK(cudaSetDevice(f->device));
  const size_t elems = f->chunk_bytes / sizeof(float);
  const long long start_ns =
      f->t0_raw + static_cast<long long>(f->offset_s * 1e9);
  sleep_until_raw(start_ns);
  long long k = 0;
  for (;;) {
    float* buf = f->buf[k % 2];
    NCCL_CHECK(ncclRecv(buf, elems, ncclFloat, 1, f->comm, f->stream));
    CUDA_CHECK(cudaStreamSynchronize(f->stream));
    CUDA_CHECK(cudaMemcpyAsync(f->host_probe, buf, sizeof(float),
                               cudaMemcpyDeviceToHost, f->stream));
    CUDA_CHECK(cudaStreamSynchronize(f->stream));
    const float probe = *f->host_probe;
    const long long t = now_ns(CLOCK_MONOTONIC_RAW) - f->t0_raw;
    if (probe == -1.0f) {
      f->sentinel_ns = t;
      break;
    }
    if (probe != static_cast<float>(k)) ++f->mismatches;
    f->completions_ns.push_back(t);
    ++k;
  }
  f->data_chunks = k;
  // G3 final check: the last data chunk sits in the other buffer. Probe
  // elements 0, mid and last must carry the final index, and element 1 must
  // carry the flow's fill constant.
  if (k > 0) {
    float* last_buf = f->buf[(k - 1) % 2];
    const float expected = static_cast<float>(k - 1);
    const size_t probes[3] = {0, elems / 2, elems - 1};
    for (int q = 0; q < 3; ++q) {
      float got = 0.f;
      CUDA_CHECK(cudaMemcpy(&got, last_buf + probes[q], sizeof(float),
                            cudaMemcpyDeviceToHost));
      if (got != expected) ++f->final_probe_mismatches;
    }
    float fill = 0.f;
    CUDA_CHECK(cudaMemcpy(&fill, last_buf + 1, sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (fill != static_cast<float>(100 + f->flow_id))
      ++f->final_probe_mismatches;
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  std::string out_prefix = "capture";
  std::string id_dir = ".";
  std::string cell = "unnamed";
  std::string sources_arg;
  std::string offsets_arg;
  int dest_rank = 0;
  double window_s = 60.0;
  long long chunk_bytes = 8388608;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) out_prefix = argv[++i];
    else if (std::strcmp(argv[i], "--id-dir") == 0 && i + 1 < argc) id_dir = argv[++i];
    else if (std::strcmp(argv[i], "--cell") == 0 && i + 1 < argc) cell = argv[++i];
    else if (std::strcmp(argv[i], "--dest") == 0 && i + 1 < argc) dest_rank = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--sources") == 0 && i + 1 < argc) sources_arg = argv[++i];
    else if (std::strcmp(argv[i], "--offsets") == 0 && i + 1 < argc) offsets_arg = argv[++i];
    else if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) window_s = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--chunk-bytes") == 0 && i + 1 < argc) chunk_bytes = std::atoll(argv[++i]);
  }

  const int rank = env_int("CAPT_RANK", env_int("SLURM_PROCID", 0));
  const int world = env_int("CAPT_WORLD", env_int("SLURM_NTASKS", 1));
  const int local_rank = env_int("CAPT_LOCALID", env_int("SLURM_LOCALID", 0));
  const int poll_secs = env_int("CAPT_ID_POLL_SECS", 300);
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);

  std::vector<int> sources;
  if (sources_arg.empty()) {
    for (int r = 0; r < world; ++r)
      if (r != dest_rank) sources.push_back(r);
  } else {
    sources = parse_int_list(sources_arg);
  }
  std::vector<double> offsets(sources.size(), 0.0);
  if (!offsets_arg.empty()) {
    offsets = parse_double_list(offsets_arg);
    if (offsets.size() != sources.size()) {
      std::fprintf(stderr, "FATAL %zu offsets for %zu sources\n",
                   offsets.size(), sources.size());
      return 24;
    }
  }
  const int n_flows = static_cast<int>(sources.size());
  if (world < 2 || n_flows < 1) {
    std::fprintf(stderr, "FATAL need at least one source and one dest\n");
    return 24;
  }

  int my_flow = -1;  // flow index if this rank is a source
  for (int f = 0; f < n_flows; ++f)
    if (sources[f] == rank) my_flow = f;
  const bool is_dest = (rank == dest_rank);
  const bool is_idle = (!is_dest && my_flow < 0);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  const int device = local_rank % std::max(device_count, 1);
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  char pci[64] = {0};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), device));
  int nccl_version = 0;
  NCCL_CHECK(ncclGetVersion(&nccl_version));

  std::printf(
      "[lane] cell %s rank %d of %d on %s device %d (%s, %s) role %s nccl %d\n",
      cell.c_str(), rank, world, host, device, prop.name, pci,
      is_dest ? "dest" : (is_idle ? "idle" : "source"), nccl_version);
  std::fflush(stdout);

  // World communicator for barriers and the final conservation reduction.
  ncclUniqueId wid;
  std::memset(&wid, 0, sizeof(wid));
  share_unique_id(&wid, rank == 0, id_dir + "/world_id.bin", poll_secs);
  ncclComm_t world_comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&world_comm, world, wid, rank));

  cudaStream_t world_stream;
  CUDA_CHECK(cudaStreamCreate(&world_stream));
  float* barrier_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&barrier_buf, sizeof(float)));
  CUDA_CHECK(cudaMemset(barrier_buf, 0, sizeof(float)));
  auto world_barrier = [&]() {
    NCCL_CHECK(ncclAllReduce(barrier_buf, barrier_buf, 1, ncclFloat, ncclSum,
                             world_comm, world_stream));
    CUDA_CHECK(cudaStreamSynchronize(world_stream));
  };

  // Tracer floor, re-verified in the scored run on every rank.
  float* floor_buf = nullptr;
  CUDA_CHECK(cudaMalloc(&floor_buf, 4096 * sizeof(float)));
  float* host_probe_main = nullptr;
  CUDA_CHECK(cudaMallocHost(&host_probe_main, sizeof(float)));
  const FloorStats floor =
      measure_tracer_floor(world_stream, floor_buf, host_probe_main);
  std::printf("[lane] rank %d tracer floor p50 %.3f us p95 %.3f us\n", rank,
              floor.p50, floor.p95);

  const size_t elems = static_cast<size_t>(chunk_bytes) / sizeof(float);

  // Pair communicators, one per flow, built in ascending flow order.
  std::vector<DestFlow> dflows;
  ncclComm_t my_pair_comm = nullptr;
  cudaStream_t my_stream{};
  float* src_buf = nullptr;
  if (is_dest) dflows.resize(n_flows);
  for (int f = 0; f < n_flows; ++f) {
    const std::string id_path =
        id_dir + "/flow" + std::to_string(f) + "_id.bin";
    if (is_dest) {
      ncclUniqueId fid;
      std::memset(&fid, 0, sizeof(fid));
      share_unique_id(&fid, true, id_path, poll_secs);
      DestFlow& d = dflows[f];
      d.flow_id = f;
      d.source_rank = sources[f];
      d.offset_s = offsets[f];
      d.chunk_bytes = chunk_bytes;
      d.device = device;
      d.window_s = window_s;
      NCCL_CHECK(ncclCommInitRank(&d.comm, 2, fid, 0));
      CUDA_CHECK(cudaStreamCreate(&d.stream));
      CUDA_CHECK(cudaMalloc(&d.buf[0], chunk_bytes));
      CUDA_CHECK(cudaMalloc(&d.buf[1], chunk_bytes));
      // 0xFF bytes decode as NaN floats, so a chunk that never arrives can
      // never satisfy a probe equality.
      CUDA_CHECK(cudaMemset(d.buf[0], 0xFF, chunk_bytes));
      CUDA_CHECK(cudaMemset(d.buf[1], 0xFF, chunk_bytes));
      CUDA_CHECK(cudaMallocHost(&d.host_probe, sizeof(float)));
      d.completions_ns.reserve(1200000);
    } else if (my_flow == f) {
      ncclUniqueId fid;
      std::memset(&fid, 0, sizeof(fid));
      share_unique_id(&fid, false, id_path, poll_secs);
      NCCL_CHECK(ncclCommInitRank(&my_pair_comm, 2, fid, 1));
      CUDA_CHECK(cudaStreamCreate(&my_stream));
      CUDA_CHECK(cudaMalloc(&src_buf, chunk_bytes));
      k_fill_float<<<864, 256, 0, my_stream>>>(
          src_buf, elems, static_cast<float>(100 + f));
      CUDA_CHECK(cudaStreamSynchronize(my_stream));
    }
  }

  // Warm both directions of every flow with one 8-byte exchange so a join
  // is a data-plane arrival on an established connection.
  for (int f = 0; f < n_flows; ++f) {
    if (is_dest) {
      DestFlow& d = dflows[f];
      NCCL_CHECK(ncclSend(d.buf[0], 2, ncclFloat, 1, d.comm, d.stream));
      NCCL_CHECK(ncclRecv(d.buf[1], 2, ncclFloat, 1, d.comm, d.stream));
      CUDA_CHECK(cudaStreamSynchronize(d.stream));
      CUDA_CHECK(cudaMemset(d.buf[0], 0xFF, chunk_bytes));
      CUDA_CHECK(cudaMemset(d.buf[1], 0xFF, chunk_bytes));
    } else if (my_flow == f) {
      NCCL_CHECK(ncclRecv(src_buf, 2, ncclFloat, 0, my_pair_comm, my_stream));
      NCCL_CHECK(ncclSend(src_buf, 2, ncclFloat, 0, my_pair_comm, my_stream));
      CUDA_CHECK(cudaStreamSynchronize(my_stream));
      k_fill_float<<<864, 256, 0, my_stream>>>(
          src_buf, elems, static_cast<float>(100 + f));
      CUDA_CHECK(cudaStreamSynchronize(my_stream));
    }
  }

  // Common epoch.
  world_barrier();
  const long long t0_raw = now_ns(CLOCK_MONOTONIC_RAW);
  const long long t0_real = now_ns(CLOCK_REALTIME);
  std::printf("[lane] rank %d T0 raw %lld realtime %lld\n", rank, t0_raw,
              t0_real);
  std::fflush(stdout);

  // The window.
  std::vector<long long> src_completions_ns;
  long long src_chunks = 0;
  if (is_dest) {
    std::vector<pthread_t> threads(n_flows);
    for (int f = 0; f < n_flows; ++f) {
      dflows[f].t0_raw = t0_raw;
      if (pthread_create(&threads[f], nullptr, dest_flow_thread,
                         &dflows[f]) != 0) {
        std::fprintf(stderr, "FATAL pthread_create failed\n");
        return 28;
      }
    }
    for (int f = 0; f < n_flows; ++f) pthread_join(threads[f], nullptr);
  } else if (my_flow >= 0) {
    src_completions_ns.reserve(1200000);
    const long long start_ns =
        t0_raw + static_cast<long long>(offsets[my_flow] * 1e9);
    const long long end_ns =
        t0_raw + static_cast<long long>(window_s * 1e9);
    sleep_until_raw(start_ns);
    long long k = 0;
    while (now_ns(CLOCK_MONOTONIC_RAW) < end_ns) {
      k_write_probe<<<1, 1, 0, my_stream>>>(src_buf, elems / 2, elems - 1,
                                            static_cast<float>(k));
      NCCL_CHECK(ncclSend(src_buf, elems, ncclFloat, 0, my_pair_comm,
                          my_stream));
      CUDA_CHECK(cudaStreamSynchronize(my_stream));
      src_completions_ns.push_back(now_ns(CLOCK_MONOTONIC_RAW) - t0_raw);
      ++k;
    }
    src_chunks = k;
    // Sentinel.
    k_write_probe<<<1, 1, 0, my_stream>>>(src_buf, elems / 2, elems - 1,
                                          -1.0f);
    NCCL_CHECK(ncclSend(src_buf, elems, ncclFloat, 0, my_pair_comm,
                        my_stream));
    CUDA_CHECK(cudaStreamSynchronize(my_stream));
  }

  // Conservation reduction over the world: per flow, the destination's
  // count, the source's count, the mismatch counters and the sentinel
  // marker, combined with max so no rank can hide a defect.
  const int slots = n_flows * 5;
  std::vector<double> local(slots, 0.0);
  for (int f = 0; f < n_flows; ++f) {
    if (is_dest) {
      local[f * 5 + 0] = static_cast<double>(dflows[f].data_chunks);
      local[f * 5 + 2] = static_cast<double>(dflows[f].mismatches);
      local[f * 5 + 3] = static_cast<double>(dflows[f].final_probe_mismatches);
      local[f * 5 + 4] = (dflows[f].sentinel_ns >= 0) ? 1.0 : 0.0;
    } else if (my_flow == f) {
      local[f * 5 + 1] = static_cast<double>(src_chunks);
    }
  }
  world_barrier();
  double* dev_red = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_red, slots * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dev_red, local.data(), slots * sizeof(double),
                        cudaMemcpyHostToDevice));
  NCCL_CHECK(ncclAllReduce(dev_red, dev_red, slots, ncclDouble, ncclMax,
                           world_comm, world_stream));
  CUDA_CHECK(cudaStreamSynchronize(world_stream));
  std::vector<double> reduced(slots, 0.0);
  CUDA_CHECK(cudaMemcpy(reduced.data(), dev_red, slots * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dev_red));

  // Per-rank metadata JSON.
  {
    const std::string p =
        out_prefix + "_rank" + std::to_string(rank) + ".json";
    FILE* f = std::fopen(p.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f,
                   "{\"cell\":\"%s\",\"rank\":%d,\"world\":%d,"
                   "\"host\":\"%s\",\"device\":\"%s\",\"pci\":\"%s\","
                   "\"role\":\"%s\",\"nccl_version\":%d,"
                   "\"chunk_bytes\":%lld,\"window_s\":%.3f,"
                   "\"t0_monotonic_raw_ns\":%lld,\"t0_realtime_ns\":%lld,"
                   "\"tracer_floor_us\":{\"p5\":%.4f,\"p50\":%.4f,"
                   "\"p95\":%.4f,\"p99\":%.4f,\"max\":%.4f,\"mean\":%.4f},"
                   "\"slurm_job_id\":\"%s\",\"nodelist\":\"%s\","
                   "\"nccl_socket_ifname\":\"%s\"}\n",
                   cell.c_str(), rank, world, host, prop.name, pci,
                   is_dest ? "dest" : (is_idle ? "idle" : "source"),
                   nccl_version, chunk_bytes, window_s, t0_raw, t0_real,
                   floor.p5, floor.p50, floor.p95, floor.p99, floor.maxv,
                   floor.mean, env_str("SLURM_JOB_ID").c_str(),
                   env_str("SLURM_JOB_NODELIST").c_str(),
                   env_str("NCCL_SOCKET_IFNAME").c_str());
      std::fclose(f);
    }
  }

  // Series files, written after every timed path has closed.
  auto write_series = [&](const std::string& path,
                          const std::vector<long long>& t,
                          int flow_id, const char* side) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
      std::fprintf(stderr, "FATAL cannot write %s\n", path.c_str());
      std::exit(26);
    }
    std::fprintf(f, "cell,flow,side,chunk_idx,chunk_bytes,t_ns_since_t0\n");
    for (size_t k = 0; k < t.size(); ++k) {
      std::fprintf(f, "%s,%d,%s,%zu,%lld,%lld\n", cell.c_str(), flow_id,
                   side, k, chunk_bytes, t[k]);
    }
    std::fclose(f);
  };
  if (is_dest) {
    for (int f = 0; f < n_flows; ++f) {
      write_series(out_prefix + "_flow" + std::to_string(f) + "_dest.csv",
                   dflows[f].completions_ns, f, "dest");
    }
    // Destination summary JSON.
    const std::string p = out_prefix + "_summary.json";
    FILE* f = std::fopen(p.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "{\"cell\":\"%s\",\"world\":%d,\"dest_rank\":%d,",
                   cell.c_str(), world, dest_rank);
      std::fprintf(f, "\"chunk_bytes\":%lld,\"window_s\":%.3f,\"flows\":[",
                   chunk_bytes, window_s);
      for (int fl = 0; fl < n_flows; ++fl) {
        const DestFlow& d = dflows[fl];
        std::fprintf(f,
                     "%s{\"flow\":%d,\"source_rank\":%d,\"offset_s\":%.3f,"
                     "\"dest_chunks\":%lld,\"source_chunks\":%.0f,"
                     "\"mismatches\":%lld,\"final_probe_mismatches\":%lld,"
                     "\"sentinel_seen\":%.0f,\"sentinel_ns\":%lld}",
                     fl ? "," : "", fl, d.source_rank, d.offset_s,
                     d.data_chunks, reduced[fl * 5 + 1],
                     static_cast<long long>(d.mismatches),
                     static_cast<long long>(d.final_probe_mismatches),
                     reduced[fl * 5 + 4], d.sentinel_ns);
      }
      std::fprintf(f, "]}\n");
      std::fclose(f);
    }
  } else if (my_flow >= 0) {
    write_series(out_prefix + "_flow" + std::to_string(my_flow) + "_src.csv",
                 src_completions_ns, my_flow, "src");
  }

  // G3 verdict across ranks, in every rank's exit code.
  int g3_bad = 0;
  for (int f = 0; f < n_flows; ++f) {
    const double dest_n = reduced[f * 5 + 0];
    const double src_n = reduced[f * 5 + 1];
    const double mism = reduced[f * 5 + 2];
    const double final_mism = reduced[f * 5 + 3];
    const double sentinel = reduced[f * 5 + 4];
    if (dest_n != src_n || mism != 0.0 || final_mism != 0.0 ||
        sentinel != 1.0) {
      std::fprintf(stderr,
                   "FATAL guard G3 flow %d: dest %.0f src %.0f mism %.0f "
                   "final %.0f sentinel %.0f\n",
                   f, dest_n, src_n, mism, final_mism, sentinel);
      g3_bad = 1;
    }
  }

  if (is_dest) {
    for (int f = 0; f < n_flows; ++f) {
      CUDA_CHECK(cudaFree(dflows[f].buf[0]));
      CUDA_CHECK(cudaFree(dflows[f].buf[1]));
      CUDA_CHECK(cudaFreeHost(dflows[f].host_probe));
      CUDA_CHECK(cudaStreamDestroy(dflows[f].stream));
      NCCL_CHECK(ncclCommDestroy(dflows[f].comm));
    }
  } else if (my_flow >= 0) {
    CUDA_CHECK(cudaFree(src_buf));
    CUDA_CHECK(cudaStreamDestroy(my_stream));
    NCCL_CHECK(ncclCommDestroy(my_pair_comm));
  }
  CUDA_CHECK(cudaFreeHost(host_probe_main));
  CUDA_CHECK(cudaFree(floor_buf));
  CUDA_CHECK(cudaFree(barrier_buf));
  CUDA_CHECK(cudaStreamDestroy(world_stream));
  NCCL_CHECK(ncclCommDestroy(world_comm));

  if (g3_bad != 0) return 27;
  std::printf("[lane] rank %d clean\n", rank);
  return 0;
}
