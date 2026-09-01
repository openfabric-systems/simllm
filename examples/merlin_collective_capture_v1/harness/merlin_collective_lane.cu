// Merlin TRAF-77 collective capture lane.
//
// One Slurm task owns one GPU. Rank 0 shares the NCCL unique id through the
// allocated nodes' shared filesystem, so the lane has no MPI dependency. Each
// output record is exactly one JSON line for one operation and payload cell.
// Warmups are declared but excluded. Every measured repeat preserves each
// rank's CLOCK_MONOTONIC_RAW release, per-application-chunk completion and
// phase completion. Rank-local elapsed times are the comparable wall metric;
// raw clock epochs from different nodes are retained but never subtracted.

#include <cuda_runtime.h>
#include <nccl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err_ = (expr);                                                \
    if (err_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "FATAL cuda %s at %s:%d: %s\n", #expr, __FILE__,  \
                   __LINE__, cudaGetErrorString(err_));                       \
      std::exit(20);                                                          \
    }                                                                         \
  } while (0)

#define NCCL_CHECK(expr)                                                      \
  do {                                                                        \
    ncclResult_t result_ = (expr);                                            \
    if (result_ != ncclSuccess) {                                             \
      std::fprintf(stderr, "FATAL nccl %s at %s:%d: %s\n", #expr, __FILE__,  \
                   __LINE__, ncclGetErrorString(result_));                    \
      std::exit(21);                                                          \
    }                                                                         \
  } while (0)

constexpr size_t kPayloads[] = {
    8,        64,        512,       4096,      16384,     65536,
    131072,   196608,    262144,    393216,    524288,    786432,
    1048576,  1572864,   2097152,   3145728,   4194304,   8388608,
    16777216, 33554432,  67108864,  134217728,
};
constexpr int kPayloadCount =
    static_cast<int>(sizeof(kPayloads) / sizeof(kPayloads[0]));
constexpr const char* kInterfaces[] = {"hsn0", "hsn1", "hsn2", "hsn3"};
constexpr int kInterfaceCount = 4;
constexpr int kBlockThreads = 256;
constexpr int kGridBlocks = 864;
constexpr int kHostBytes = 128;

long long now_raw_ns() {
  timespec value{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
    std::perror("clock_gettime(CLOCK_MONOTONIC_RAW)");
    std::exit(22);
  }
  return static_cast<long long>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::atoi(raw);
}

std::string env_string(const char* name) {
  const char* raw = std::getenv(name);
  return raw == nullptr ? std::string("unset") : std::string(raw);
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default: escaped += ch; break;
    }
  }
  return escaped;
}

void write_string(FILE* output, const char* name, const std::string& value,
                  bool comma = true) {
  std::fprintf(output, "\"%s\":\"%s\"%s", name,
               json_escape(value).c_str(), comma ? "," : "");
}

void share_unique_id(ncclUniqueId* id, int rank, const std::string& path) {
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(id));
    const std::string temporary = path + ".tmp";
    FILE* output = std::fopen(temporary.c_str(), "wb");
    if (output == nullptr || std::fwrite(id, sizeof(*id), 1, output) != 1) {
      std::fprintf(stderr, "FATAL cannot write NCCL unique id\n");
      std::exit(23);
    }
    std::fclose(output);
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
      std::fprintf(stderr, "FATAL cannot publish NCCL unique id\n");
      std::exit(23);
    }
    return;
  }
  for (int attempt = 0; attempt < 900; ++attempt) {
    FILE* input = std::fopen(path.c_str(), "rb");
    if (input != nullptr) {
      const size_t count = std::fread(id, sizeof(*id), 1, input);
      std::fclose(input);
      if (count == 1) return;
    }
    usleep(200000);
  }
  std::fprintf(stderr, "FATAL never observed NCCL unique id\n");
  std::exit(23);
}

__global__ void fill_float(float* data, size_t count, float value) {
  size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; index < count; index += stride) data[index] = value;
}

enum class Operation { AllGather, ReduceScatter, AllReduce, AllToAllv };

const char* operation_name(Operation operation) {
  switch (operation) {
    case Operation::AllGather: return "all_gather";
    case Operation::ReduceScatter: return "reduce_scatter";
    case Operation::AllReduce: return "all_reduce";
    case Operation::AllToAllv: return "all_to_allv";
  }
  return "unknown";
}

const char* payload_semantics(Operation operation) {
  switch (operation) {
    case Operation::AllGather: return "input bytes per rank";
    case Operation::ReduceScatter: return "output bytes per rank";
    case Operation::AllReduce: return "input bytes per rank";
    case Operation::AllToAllv: return "bytes sent to each remote peer";
  }
  return "unknown";
}

struct CellSpec {
  Operation operation;
  size_t payload_bytes;
};

bool is_anchor(const CellSpec& cell, int width) {
  if (cell.payload_bytes != 8) return false;
  if (cell.operation == Operation::AllReduce) return width == 2 || width == 8;
  return width == 8 && cell.operation == Operation::AllToAllv;
}

long long anchor_ps(const CellSpec& cell, int width) {
  if (cell.payload_bytes != 8) return 0;
  if (cell.operation == Operation::AllReduce && width == 2) return 40140799LL;
  if (cell.operation == Operation::AllReduce && width == 8) return 50790000LL;
  if (cell.operation == Operation::AllToAllv && width == 8) return 89805000LL;
  return 0;
}

std::vector<CellSpec> ordered_cells(int width) {
  std::vector<CellSpec> cells;
  cells.push_back(CellSpec{Operation::AllReduce, 8});
  if (width == 8) cells.push_back(CellSpec{Operation::AllToAllv, 8});
  const Operation operations[] = {
      Operation::AllGather,
      Operation::ReduceScatter,
      Operation::AllReduce,
      Operation::AllToAllv,
  };
  for (Operation operation : operations) {
    for (int index = 0; index < kPayloadCount; ++index) {
      CellSpec cell{operation, kPayloads[index]};
      if (!is_anchor(cell, width)) cells.push_back(cell);
    }
  }
  return cells;
}

long long read_counter(const std::string& net_root, const char* interface,
                       const char* counter) {
  const std::string path = net_root + "/" + interface + "/statistics/" + counter;
  std::ifstream input(path);
  unsigned long long value = 0;
  if (!(input >> value)) {
    std::fprintf(stderr, "FATAL cannot read counter %s\n", path.c_str());
    std::exit(24);
  }
  return static_cast<long long>(value);
}

struct CounterSnapshot {
  long long captured_ns = 0;
  std::array<long long, kInterfaceCount> rx{};
  std::array<long long, kInterfaceCount> tx{};
};

CounterSnapshot read_counters(const std::string& net_root) {
  CounterSnapshot snapshot;
  snapshot.captured_ns = now_raw_ns();
  for (int index = 0; index < kInterfaceCount; ++index) {
    snapshot.rx[index] = read_counter(net_root, kInterfaces[index], "rx_bytes");
    snapshot.tx[index] = read_counter(net_root, kInterfaces[index], "tx_bytes");
  }
  return snapshot;
}

std::vector<long long> gather_int64(const std::vector<long long>& local, int world,
                                    ncclComm_t communicator,
                                    cudaStream_t stream) {
  long long* device_send = nullptr;
  long long* device_receive = nullptr;
  CUDA_CHECK(cudaMalloc(&device_send, local.size() * sizeof(long long)));
  CUDA_CHECK(cudaMalloc(&device_receive,
                        local.size() * static_cast<size_t>(world) *
                            sizeof(long long)));
  CUDA_CHECK(cudaMemcpy(device_send, local.data(),
                        local.size() * sizeof(long long),
                        cudaMemcpyHostToDevice));
  NCCL_CHECK(ncclAllGather(device_send, device_receive, local.size(), ncclInt64,
                           communicator, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<long long> gathered(local.size() * static_cast<size_t>(world));
  CUDA_CHECK(cudaMemcpy(gathered.data(), device_receive,
                        gathered.size() * sizeof(long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_send));
  CUDA_CHECK(cudaFree(device_receive));
  return gathered;
}

long long median_integer(std::vector<long long> values) {
  std::sort(values.begin(), values.end());
  if (values.size() % 2 == 1) return values[values.size() / 2];
  const size_t upper = values.size() / 2;
  return (values[upper - 1] + values[upper]) / 2;
}

struct Arguments {
  std::string output_path;
  std::string id_path;
  std::string attempt_id;
  std::string concentration;
  std::string script_name;
  std::string script_sha256;
  std::string config_sha256;
  std::string net_class_root;
  int repeats = 25;
  int warmups = 5;
  size_t chunk_bytes = 8388608;
};

Arguments parse_arguments(int argc, char** argv) {
  Arguments args;
  for (int index = 1; index < argc; ++index) {
    auto value = [&]() -> const char* {
      if (index + 1 >= argc) {
        std::fprintf(stderr, "FATAL missing value after %s\n", argv[index]);
        std::exit(25);
      }
      return argv[++index];
    };
    if (std::strcmp(argv[index], "--out") == 0) args.output_path = value();
    else if (std::strcmp(argv[index], "--id-path") == 0) args.id_path = value();
    else if (std::strcmp(argv[index], "--attempt-id") == 0) args.attempt_id = value();
    else if (std::strcmp(argv[index], "--concentration") == 0)
      args.concentration = value();
    else if (std::strcmp(argv[index], "--script-name") == 0)
      args.script_name = value();
    else if (std::strcmp(argv[index], "--script-sha256") == 0)
      args.script_sha256 = value();
    else if (std::strcmp(argv[index], "--config-sha256") == 0)
      args.config_sha256 = value();
    else if (std::strcmp(argv[index], "--net-class-root") == 0)
      args.net_class_root = value();
    else if (std::strcmp(argv[index], "--repeats") == 0)
      args.repeats = std::atoi(value());
    else if (std::strcmp(argv[index], "--warmups") == 0)
      args.warmups = std::atoi(value());
    else if (std::strcmp(argv[index], "--chunk-bytes") == 0)
      args.chunk_bytes = static_cast<size_t>(std::strtoull(value(), nullptr, 10));
    else {
      std::fprintf(stderr, "FATAL unknown argument %s\n", argv[index]);
      std::exit(25);
    }
  }
  if (args.output_path.empty() || args.id_path.empty() || args.attempt_id.empty() ||
      args.concentration.empty() || args.script_name.empty() ||
      args.script_sha256.empty() || args.config_sha256.empty() ||
      args.net_class_root.empty() || args.repeats < 2 || args.warmups < 0 ||
      args.chunk_bytes == 0 || args.chunk_bytes % sizeof(float) != 0) {
    std::fprintf(stderr, "FATAL incomplete or invalid lane arguments\n");
    std::exit(25);
  }
  return args;
}

}  // namespace

int main(int argc, char** argv) {
  const Arguments args = parse_arguments(argc, argv);
  const int rank = env_int("SLURM_PROCID", 0);
  const int world = env_int("SLURM_NTASKS", 1);
  const int local_rank = env_int("SLURM_LOCALID", 0);
  const int tasks_per_node = env_int("SLURM_NTASKS_PER_NODE", 1);
  if (world != 2 && world != 8) {
    std::fprintf(stderr, "FATAL frozen widths are 2 and 8, observed %d\n", world);
    return 26;
  }

  char host_buffer[kHostBytes] = {0};
  gethostname(host_buffer, sizeof(host_buffer) - 1);
  const std::string local_host(host_buffer);

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 1) {
    std::fprintf(stderr, "FATAL rank %d sees no GPU\n", rank);
    return 26;
  }
  const int device = local_rank % device_count;
  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  char pci_bus_id[64] = {0};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));
  int nccl_version = 0;
  NCCL_CHECK(ncclGetVersion(&nccl_version));

  std::printf("[lane] rank %d/%d host %s local %d GPU %s PCI %s NCCL %d\n",
              rank, world, local_host.c_str(), local_rank, properties.name,
              pci_bus_id, nccl_version);
  std::fflush(stdout);

  ncclUniqueId unique_id;
  std::memset(&unique_id, 0, sizeof(unique_id));
  share_unique_id(&unique_id, rank, args.id_path);
  ncclComm_t communicator = nullptr;
  NCCL_CHECK(ncclCommInitRank(&communicator, world, unique_id, rank));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  float* barrier_send = nullptr;
  float* barrier_receive = nullptr;
  CUDA_CHECK(cudaMalloc(&barrier_send, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&barrier_receive, sizeof(float)));
  const float one = 1.0f;
  CUDA_CHECK(cudaMemcpy(barrier_send, &one, sizeof(float), cudaMemcpyHostToDevice));
  auto barrier = [&]() {
    NCCL_CHECK(ncclAllReduce(barrier_send, barrier_receive, 1, ncclFloat,
                             ncclSum, communicator, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  char* device_host_send = nullptr;
  char* device_host_receive = nullptr;
  CUDA_CHECK(cudaMalloc(&device_host_send, kHostBytes));
  CUDA_CHECK(cudaMalloc(&device_host_receive,
                        static_cast<size_t>(world) * kHostBytes));
  CUDA_CHECK(cudaMemcpy(device_host_send, host_buffer, kHostBytes,
                        cudaMemcpyHostToDevice));
  NCCL_CHECK(ncclAllGather(device_host_send, device_host_receive, kHostBytes,
                           ncclChar, communicator, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<char> gathered_hosts(static_cast<size_t>(world) * kHostBytes, 0);
  CUDA_CHECK(cudaMemcpy(gathered_hosts.data(), device_host_receive,
                        gathered_hosts.size(), cudaMemcpyDeviceToHost));
  std::vector<std::string> hosts;
  for (int peer = 0; peer < world; ++peer) {
    hosts.emplace_back(gathered_hosts.data() + static_cast<size_t>(peer) * kHostBytes);
  }

  const size_t max_elements = args.chunk_bytes / sizeof(float);
  const size_t wide_elements = max_elements * static_cast<size_t>(world);
  float* collective_send = nullptr;
  float* collective_receive = nullptr;
  float* all_to_all_send = nullptr;
  float* all_to_all_receive = nullptr;
  CUDA_CHECK(cudaMalloc(&collective_send, wide_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&collective_receive, wide_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&all_to_all_send, wide_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&all_to_all_receive, wide_elements * sizeof(float)));
  fill_float<<<kGridBlocks, kBlockThreads>>>(
      collective_send, wide_elements, static_cast<float>(rank + 1));
  for (int peer = 0; peer < world; ++peer) {
    fill_float<<<kGridBlocks, kBlockThreads>>>(
        all_to_all_send + static_cast<size_t>(peer) * max_elements,
        max_elements, static_cast<float>(rank * world + peer + 1));
  }
  CUDA_CHECK(cudaMemset(collective_receive, 0xFF,
                        wide_elements * sizeof(float)));
  CUDA_CHECK(cudaMemset(all_to_all_receive, 0xFF,
                        wide_elements * sizeof(float)));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  FILE* output = nullptr;
  if (rank == 0) {
    output = std::fopen(args.output_path.c_str(), "wb");
    if (output == nullptr) {
      std::fprintf(stderr, "FATAL cannot open output JSONL\n");
      return 26;
    }
  }

  int exit_code = 0;
  const std::vector<CellSpec> cells = ordered_cells(world);
  for (const CellSpec& cell : cells) {
    const size_t chunk_count =
        (cell.payload_bytes + args.chunk_bytes - 1) / args.chunk_bytes;
    const int timing_slots = 2 + static_cast<int>(chunk_count);

    auto issue_chunk = [&](size_t chunk_payload_bytes) {
      const size_t elements = chunk_payload_bytes / sizeof(float);
      switch (cell.operation) {
        case Operation::AllGather:
          NCCL_CHECK(ncclAllGather(collective_send, collective_receive, elements,
                                   ncclFloat, communicator, stream));
          break;
        case Operation::ReduceScatter:
          NCCL_CHECK(ncclReduceScatter(collective_send, collective_receive,
                                       elements, ncclFloat, ncclSum,
                                       communicator, stream));
          break;
        case Operation::AllReduce:
          NCCL_CHECK(ncclAllReduce(collective_send, collective_receive, elements,
                                   ncclFloat, ncclSum, communicator, stream));
          break;
        case Operation::AllToAllv:
          NCCL_CHECK(ncclGroupStart());
          for (int peer = 0; peer < world; ++peer) {
            if (peer == rank) continue;
            NCCL_CHECK(ncclSend(
                all_to_all_send + static_cast<size_t>(peer) * max_elements,
                elements, ncclFloat, peer, communicator, stream));
            NCCL_CHECK(ncclRecv(
                all_to_all_receive + static_cast<size_t>(peer) * max_elements,
                elements, ncclFloat, peer, communicator, stream));
          }
          NCCL_CHECK(ncclGroupEnd());
          break;
      }
    };

    for (int warmup = 0; warmup < args.warmups; ++warmup) {
      for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const size_t offset = chunk * args.chunk_bytes;
        issue_chunk(std::min(args.chunk_bytes, cell.payload_bytes - offset));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      barrier();
    }

    barrier();
    const CounterSnapshot counters_before = read_counters(args.net_class_root);
    std::vector<long long> local_timing(
        static_cast<size_t>(args.repeats) * timing_slots, 0);
    for (int repeat = 0; repeat < args.repeats; ++repeat) {
      barrier();
      const size_t base = static_cast<size_t>(repeat) * timing_slots;
      local_timing[base] = now_raw_ns();
      for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const size_t offset = chunk * args.chunk_bytes;
        issue_chunk(std::min(args.chunk_bytes, cell.payload_bytes - offset));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        local_timing[base + 2 + chunk] = now_raw_ns();
      }
      local_timing[base + 1] = now_raw_ns();
    }
    const CounterSnapshot counters_after = read_counters(args.net_class_root);

    const std::vector<long long> gathered_timing =
        gather_int64(local_timing, world, communicator, stream);
    std::vector<long long> local_counters;
    local_counters.push_back(counters_before.captured_ns);
    local_counters.push_back(counters_after.captured_ns);
    for (int port = 0; port < kInterfaceCount; ++port) {
      local_counters.push_back(counters_before.rx[port]);
      local_counters.push_back(counters_before.tx[port]);
      local_counters.push_back(counters_after.rx[port]);
      local_counters.push_back(counters_after.tx[port]);
    }
    const std::vector<long long> gathered_counters =
        gather_int64(local_counters, world, communicator, stream);

    const size_t final_chunk_bytes =
        cell.payload_bytes % args.chunk_bytes == 0
            ? args.chunk_bytes
            : cell.payload_bytes % args.chunk_bytes;
    const size_t final_elements = final_chunk_bytes / sizeof(float);
    const size_t probes[3] = {0, final_elements / 2, final_elements - 1};
    int local_mismatches = 0;
    if (cell.operation == Operation::AllGather) {
      for (int source = 0; source < world; ++source) {
        const float expected = static_cast<float>(source + 1);
        for (size_t probe : probes) {
          float observed = 0.0f;
          CUDA_CHECK(cudaMemcpy(
              &observed,
              collective_receive + static_cast<size_t>(source) * final_elements +
                  probe,
              sizeof(float), cudaMemcpyDeviceToHost));
          if (observed != expected) ++local_mismatches;
        }
      }
    } else if (cell.operation == Operation::ReduceScatter ||
               cell.operation == Operation::AllReduce) {
      const float expected = static_cast<float>(world * (world + 1) / 2);
      for (size_t probe : probes) {
        float observed = 0.0f;
        CUDA_CHECK(cudaMemcpy(&observed, collective_receive + probe,
                              sizeof(float), cudaMemcpyDeviceToHost));
        if (observed != expected) ++local_mismatches;
      }
    } else {
      for (int source = 0; source < world; ++source) {
        if (source == rank) continue;
        const float expected = static_cast<float>(source * world + rank + 1);
        for (size_t probe : probes) {
          float observed = 0.0f;
          CUDA_CHECK(cudaMemcpy(
              &observed,
              all_to_all_receive + static_cast<size_t>(source) * max_elements +
                  probe,
              sizeof(float), cudaMemcpyDeviceToHost));
          if (observed != expected) ++local_mismatches;
        }
      }
    }

    int* device_mismatch = nullptr;
    CUDA_CHECK(cudaMalloc(&device_mismatch, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_mismatch, &local_mismatches, sizeof(int),
                          cudaMemcpyHostToDevice));
    NCCL_CHECK(ncclAllReduce(device_mismatch, device_mismatch, 1, ncclInt,
                             ncclMax, communicator, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int max_mismatches = 0;
    CUDA_CHECK(cudaMemcpy(&max_mismatches, device_mismatch, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_mismatch));

    std::vector<long long> max_durations;
    for (int repeat = 0; repeat < args.repeats; ++repeat) {
      long long maximum = 0;
      for (int peer = 0; peer < world; ++peer) {
        const size_t base =
            static_cast<size_t>(peer) * local_timing.size() +
            static_cast<size_t>(repeat) * timing_slots;
        maximum = std::max(
            maximum, gathered_timing[base + 1] - gathered_timing[base]);
      }
      max_durations.push_back(maximum);
    }
    const long long median_duration_ns = median_integer(max_durations);
    const long long reference_ps = anchor_ps(cell, world);
    const double anchor_ratio =
        reference_ps == 0
            ? 0.0
            : static_cast<double>(median_duration_ns) * 1000.0 / reference_ps;
    const bool anchor_held =
        reference_ps == 0 || (anchor_ratio >= 0.5 && anchor_ratio <= 2.0);

    if (rank == 0) {
      std::fprintf(output, "{");
      write_string(output, "schema", "simllm-merlin-collective-cell-v1");
      write_string(output, "evidence_class", "hardware-capture");
      write_string(output, "study", "merlin_collective_capture_v1");
      write_string(output, "attempt_id", args.attempt_id);
      write_string(output, "slurm_job_id", env_string("SLURM_JOB_ID"));
      write_string(output, "slurm_nodelist", env_string("SLURM_JOB_NODELIST"));
      write_string(output, "submitted_script", args.script_name);
      write_string(output, "submitted_script_sha256", args.script_sha256);
      write_string(output, "config_sha256", args.config_sha256);
      write_string(output, "concentration", args.concentration);
      write_string(output, "operation", operation_name(cell.operation));
      write_string(output, "payload_semantics", payload_semantics(cell.operation));
      write_string(output, "clock", "CLOCK_MONOTONIC_RAW");
      write_string(output, "clock_epoch_scope", "rank-local");
      write_string(output, "nccl_socket_ifname", env_string("NCCL_SOCKET_IFNAME"));
      std::fprintf(
          output,
          "\"width\":%d,\"tasks_per_node\":%d,\"payload_bytes\":%zu,"
          "\"measured_repeats\":%d,\"excluded_warmups\":%d,"
          "\"chunk_limit_bytes\":%zu,\"chunk_count\":%zu,"
          "\"nccl_version\":%d,\"max_rank_mismatches\":%d,",
          world, tasks_per_node, cell.payload_bytes, args.repeats, args.warmups,
          args.chunk_bytes, chunk_count, nccl_version, max_mismatches);
      if (reference_ps != 0) {
        std::fprintf(output,
                     "\"fg4_anchor_ps\":%lld,\"fg4_anchor_ratio\":%.12g,"
                     "\"fg4_anchor_held\":%s,",
                     reference_ps, anchor_ratio, anchor_held ? "true" : "false");
      }

      std::fprintf(output, "\"samples\":[");
      for (int repeat = 0; repeat < args.repeats; ++repeat) {
        if (repeat != 0) std::fprintf(output, ",");
        std::fprintf(output, "{\"repeat\":%d,\"max_rank_duration_ns\":%lld,",
                     repeat, max_durations[repeat]);
        std::fprintf(output, "\"ranks\":[");
        for (int peer = 0; peer < world; ++peer) {
          if (peer != 0) std::fprintf(output, ",");
          const size_t base =
              static_cast<size_t>(peer) * local_timing.size() +
              static_cast<size_t>(repeat) * timing_slots;
          const long long release = gathered_timing[base];
          const long long completion = gathered_timing[base + 1];
          std::fprintf(output,
                       "{\"rank\":%d,\"host\":\"%s\","
                       "\"release_monotonic_raw_ns\":%lld,"
                       "\"completion_monotonic_raw_ns\":%lld,"
                       "\"duration_ns\":%lld,\"chunk_completions\":[",
                       peer, json_escape(hosts[peer]).c_str(), release,
                       completion, completion - release);
          for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
            if (chunk != 0) std::fprintf(output, ",");
            const size_t offset = chunk * args.chunk_bytes;
            const size_t bytes =
                std::min(args.chunk_bytes, cell.payload_bytes - offset);
            const long long chunk_completion = gathered_timing[base + 2 + chunk];
            std::fprintf(output,
                         "{\"chunk_index\":%zu,\"payload_bytes\":%zu,"
                         "\"completion_monotonic_raw_ns\":%lld,"
                         "\"elapsed_ns\":%lld}",
                         chunk, bytes, chunk_completion,
                         chunk_completion - release);
          }
          std::fprintf(output, "]}");
        }
        std::fprintf(output, "]}");
      }
      std::fprintf(output, "],\"rank_counters\":[");
      for (int peer = 0; peer < world; ++peer) {
        if (peer != 0) std::fprintf(output, ",");
        const size_t base = static_cast<size_t>(peer) * local_counters.size();
        std::fprintf(output,
                     "{\"rank\":%d,\"host\":\"%s\",\"local_rank\":%d,"
                     "\"before_monotonic_raw_ns\":%lld,"
                     "\"after_monotonic_raw_ns\":%lld,\"ports\":[",
                     peer, json_escape(hosts[peer]).c_str(),
                     peer % tasks_per_node, gathered_counters[base],
                     gathered_counters[base + 1]);
        for (int port = 0; port < kInterfaceCount; ++port) {
          if (port != 0) std::fprintf(output, ",");
          const size_t entry = base + 2 + static_cast<size_t>(port) * 4;
          std::fprintf(output,
                       "{\"interface\":\"%s\",\"before_rx_bytes\":%lld,"
                       "\"before_tx_bytes\":%lld,\"after_rx_bytes\":%lld,"
                       "\"after_tx_bytes\":%lld}",
                       kInterfaces[port], gathered_counters[entry],
                       gathered_counters[entry + 1],
                       gathered_counters[entry + 2],
                       gathered_counters[entry + 3]);
        }
        std::fprintf(output, "]}");
      }
      std::fprintf(output, "]}\n");
      std::fflush(output);
    }

    if (max_mismatches != 0) {
      std::fprintf(stderr,
                   "FATAL value conservation failed for %s at %zu bytes: %d\n",
                   operation_name(cell.operation), cell.payload_bytes,
                   max_mismatches);
      exit_code = 27;
      break;
    }
    if (!anchor_held) {
      std::fprintf(stderr,
                   "FATAL FG-4 anchor miss for %s width %d: ratio %.6f\n",
                   operation_name(cell.operation), world, anchor_ratio);
      exit_code = 29;
      break;
    }
  }

  if (output != nullptr) std::fclose(output);
  CUDA_CHECK(cudaFree(all_to_all_receive));
  CUDA_CHECK(cudaFree(all_to_all_send));
  CUDA_CHECK(cudaFree(collective_receive));
  CUDA_CHECK(cudaFree(collective_send));
  CUDA_CHECK(cudaFree(device_host_receive));
  CUDA_CHECK(cudaFree(device_host_send));
  CUDA_CHECK(cudaFree(barrier_receive));
  CUDA_CHECK(cudaFree(barrier_send));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(communicator));

  if (exit_code == 0) {
    std::printf("[lane] rank %d completed %zu cells\n", rank, cells.size());
  }
  return exit_code;
}
