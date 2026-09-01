// TRAF-81 A100 all-gather and reduce-scatter measurement lane.
//
// One scheduler process owns one GPU. A unique NCCL identifier is exchanged
// through a shared file, so this harness needs no MPI runtime. Each frozen
// measurement is one CUDA-event-bracketed target collective. The duration of
// every repetition is reduced with NCCL maximum after the timed region, and
// rank zero retains the full 31-sample vector and its observed median.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    const cudaError_t error_ = (expr);                                          \
    if (error_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "FATAL cuda %s at %s:%d: %s\n", #expr, __FILE__,   \
                   __LINE__, cudaGetErrorString(error_));                       \
      std::exit(20);                                                           \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(expr)                                                        \
  do {                                                                         \
    const ncclResult_t result_ = (expr);                                        \
    if (result_ != ncclSuccess) {                                               \
      std::fprintf(stderr, "FATAL nccl %s at %s:%d: %s\n", #expr, __FILE__,  \
                   __LINE__, ncclGetErrorString(result_));                      \
      std::exit(21);                                                           \
    }                                                                          \
  } while (0)

constexpr int kWarmups = 10;
constexpr int kSamples = 31;
constexpr int kThreads = 256;
constexpr int kBlocks = 864;

constexpr size_t kBytes[] = {
    512,      2048,     8192,     32768,    65536,    131072,
    196608,   262144,   393216,   524288,   655360,   786432,
    917504,   1048576,  1179648,  1310720,  1572864,  2097152,
    3145728,  4194304,  8388608,  16777216, 33554432, 67108864,
};
constexpr size_t kByteCount = sizeof(kBytes) / sizeof(kBytes[0]);

enum class Operation { AllGather, ReduceScatter };

const char* operation_name(Operation operation) {
  return operation == Operation::AllGather ? "all_gather" : "reduce_scatter";
}

struct Measurement {
  Operation operation;
  size_t bytes;
  std::vector<double> samples_us;
  int max_rank_mismatches;
};

__global__ void fill_half(__half* values, size_t count, float value) {
  size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (; index < count; index += stride) {
    values[index] = __float2half(value);
  }
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
  std::ostringstream out;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character) << std::dec;
        } else {
          out << character;
        }
    }
  }
  return out.str();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.at(values.size() / 2);
}

void share_unique_id(ncclUniqueId* id, int rank, const std::string& path) {
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(id));
    const std::string temporary = path + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (file == nullptr || std::fwrite(id, sizeof(*id), 1, file) != 1) {
      std::fprintf(stderr, "FATAL cannot write NCCL unique id\n");
      std::exit(22);
    }
    std::fclose(file);
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
      std::fprintf(stderr, "FATAL cannot publish NCCL unique id\n");
      std::exit(22);
    }
    return;
  }

  for (int attempt = 0; attempt < 900; ++attempt) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file != nullptr) {
      const size_t count = std::fread(id, sizeof(*id), 1, file);
      std::fclose(file);
      if (count == 1) return;
    }
    usleep(200000);
  }
  std::fprintf(stderr, "FATAL never observed NCCL unique id\n");
  std::exit(23);
}

void write_result(const std::string& path, int world, int tasks_per_node,
                  int visible_devices, int nccl_version,
                  const std::string& host, const std::string& device_name,
                  const std::string& pci_bus_id,
                  const std::vector<Measurement>& measurements) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "FATAL cannot write result path\n");
    std::exit(26);
  }
  out << std::setprecision(12);
  out << "{\n"
      << "  \"schema\": \"simllm-collective-floor-measurement-v1\",\n"
      << "  \"study\": \"collective_floor_extrapolation_v1\",\n"
      << "  \"world\": " << world << ",\n"
      << "  \"tasks_per_node\": " << tasks_per_node << ",\n"
      << "  \"visible_device_count_rank0\": " << visible_devices << ",\n"
      << "  \"nccl_version\": " << nccl_version << ",\n"
      << "  \"rank0_host\": \"" << json_escape(host) << "\",\n"
      << "  \"rank0_device_name\": \"" << json_escape(device_name)
      << "\",\n"
      << "  \"rank0_pci_bus_id\": \"" << json_escape(pci_bus_id)
      << "\",\n"
      << "  \"slurm_job_id\": \""
      << json_escape(env_string("SLURM_JOB_ID")) << "\",\n"
      << "  \"slurm_nodelist\": \""
      << json_escape(env_string("SLURM_JOB_NODELIST")) << "\",\n"
      << "  \"warmup_iterations\": " << kWarmups << ",\n"
      << "  \"timed_repetitions\": " << kSamples << ",\n"
      << "  \"sample_reduction\": \"maximum-over-ranks\",\n"
      << "  \"aggregation\": \"observed-median\",\n"
      << "  \"measurements\": [\n";

  for (size_t index = 0; index < measurements.size(); ++index) {
    const Measurement& row = measurements[index];
    out << "    {\"operation\": \"" << operation_name(row.operation)
        << "\", \"operation_buffer_bytes\": " << row.bytes
        << ", \"median_us\": " << median(row.samples_us)
        << ", \"max_rank_mismatches\": " << row.max_rank_mismatches
        << ", \"samples_us\": [";
    for (size_t sample = 0; sample < row.samples_us.size(); ++sample) {
      if (sample != 0) out << ", ";
      out << row.samples_us[sample];
    }
    out << "]}";
    out << (index + 1 == measurements.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string output_path = "measurement.json";
  std::string id_path = "nccl_unique_id.bin";
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--out") == 0 && index + 1 < argc) {
      output_path = argv[++index];
    } else if (std::strcmp(argv[index], "--id-path") == 0 &&
               index + 1 < argc) {
      id_path = argv[++index];
    } else {
      std::fprintf(stderr, "usage: %s [--out PATH] [--id-path PATH]\n", argv[0]);
      return 2;
    }
  }

  const int rank = env_int("SLURM_PROCID", 0);
  const int world = env_int("SLURM_NTASKS", 1);
  const int local_rank = env_int("SLURM_LOCALID", 0);
  const int tasks_per_node = env_int("SLURM_NTASKS_PER_NODE", 1);
  if (world != 2 && world != 4 && world != 8 && world != 16) {
    std::fprintf(stderr, "FATAL unsupported frozen rank count %d\n", world);
    return 24;
  }

  char host_buffer[256] = {0};
  gethostname(host_buffer, sizeof(host_buffer) - 1);
  const std::string host(host_buffer);

  int visible_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&visible_devices));
  if (visible_devices < 1) {
    std::fprintf(stderr, "FATAL rank %d sees no CUDA device\n", rank);
    return 25;
  }
  const int device = local_rank % visible_devices;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  char pci_bus_id[64] = {0};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));
  int nccl_version = 0;
  NCCL_CHECK(ncclGetVersion(&nccl_version));
  std::printf("[lane] rank=%d world=%d host=%s local_rank=%d device=%d "
              "name=%s pci=%s nccl=%d\n",
              rank, world, host.c_str(), local_rank, device, properties.name,
              pci_bus_id, nccl_version);
  std::fflush(stdout);

  ncclUniqueId unique_id{};
  share_unique_id(&unique_id, rank, id_path);
  ncclComm_t communicator = nullptr;
  NCCL_CHECK(ncclCommInitRank(&communicator, world, unique_id, rank));
  std::printf("[lane] rank=%d communicator_initialized\n", rank);
  std::fflush(stdout);

  cudaStream_t stream;
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  int* barrier_value = nullptr;
  float* sample_value = nullptr;
  CUDA_CHECK(cudaMalloc(&barrier_value, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&sample_value, sizeof(float)));
  CUDA_CHECK(cudaMemset(barrier_value, 0, sizeof(int)));

  auto barrier = [&]() {
    NCCL_CHECK(ncclAllReduce(barrier_value, barrier_value, 1, ncclInt, ncclSum,
                             communicator, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  };

  const size_t maximum_bytes = kBytes[kByteCount - 1];
  const size_t maximum_elements = maximum_bytes / sizeof(__half);
  const size_t maximum_chunk = maximum_elements / static_cast<size_t>(world);

  __half* all_gather_send = nullptr;
  __half* all_gather_receive = nullptr;
  __half* reduce_scatter_send = nullptr;
  __half* reduce_scatter_receive = nullptr;
  CUDA_CHECK(cudaMalloc(&all_gather_send, maximum_chunk * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&all_gather_receive, maximum_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&reduce_scatter_send, maximum_elements * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&reduce_scatter_receive,
                        maximum_chunk * sizeof(__half)));

  fill_half<<<kBlocks, kThreads>>>(all_gather_send, maximum_chunk,
                                   static_cast<float>(rank + 1));
  fill_half<<<kBlocks, kThreads>>>(reduce_scatter_send, maximum_elements, 1.0f);
  CUDA_CHECK(cudaMemset(all_gather_receive, 0,
                        maximum_elements * sizeof(__half)));
  CUDA_CHECK(cudaMemset(reduce_scatter_receive, 0,
                        maximum_chunk * sizeof(__half)));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  std::vector<Measurement> measurements;
  const Operation operations[] = {Operation::AllGather,
                                  Operation::ReduceScatter};
  bool announced_timing = false;

  for (const Operation operation : operations) {
    for (const size_t bytes : kBytes) {
      const size_t total_elements = bytes / sizeof(__half);
      const size_t chunk_elements = total_elements / static_cast<size_t>(world);
      auto issue = [&]() {
        if (operation == Operation::AllGather) {
          NCCL_CHECK(ncclAllGather(all_gather_send, all_gather_receive,
                                   chunk_elements, ncclHalf, communicator,
                                   stream));
        } else {
          NCCL_CHECK(ncclReduceScatter(
              reduce_scatter_send, reduce_scatter_receive, chunk_elements,
              ncclHalf, ncclSum, communicator, stream));
        }
      };

      for (int warmup = 0; warmup < kWarmups; ++warmup) issue();
      CUDA_CHECK(cudaStreamSynchronize(stream));

      std::vector<double> samples;
      samples.reserve(kSamples);
      for (int sample = 0; sample < kSamples; ++sample) {
        barrier();
        if (!announced_timing) {
          std::printf("[lane] first_target_timing_started\n");
          std::fflush(stdout);
          announced_timing = true;
        }
        CUDA_CHECK(cudaEventRecord(start_event, stream));
        issue();
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float local_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&local_ms, start_event, stop_event));

        CUDA_CHECK(cudaMemcpyAsync(sample_value, &local_ms, sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        NCCL_CHECK(ncclAllReduce(sample_value, sample_value, 1, ncclFloat,
                                 ncclMax, communicator, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float maximum_ms = 0.0f;
        CUDA_CHECK(cudaMemcpy(&maximum_ms, sample_value, sizeof(float),
                              cudaMemcpyDeviceToHost));
        samples.push_back(static_cast<double>(maximum_ms) * 1000.0);
      }

      int local_mismatches = 0;
      const size_t probes[] = {0, chunk_elements / 2, chunk_elements - 1};
      if (operation == Operation::AllGather) {
        for (int source = 0; source < world; ++source) {
          const float expected = static_cast<float>(source + 1);
          for (const size_t probe : probes) {
            __half observed{};
            CUDA_CHECK(cudaMemcpy(
                &observed,
                all_gather_receive + static_cast<size_t>(source) * chunk_elements +
                    probe,
                sizeof(__half), cudaMemcpyDeviceToHost));
            if (__half2float(observed) != expected) ++local_mismatches;
          }
        }
      } else {
        const float expected = static_cast<float>(world);
        for (const size_t probe : probes) {
          __half observed{};
          CUDA_CHECK(cudaMemcpy(&observed, reduce_scatter_receive + probe,
                                sizeof(__half), cudaMemcpyDeviceToHost));
          if (__half2float(observed) != expected) ++local_mismatches;
        }
      }

      int* mismatch_value = nullptr;
      CUDA_CHECK(cudaMalloc(&mismatch_value, sizeof(int)));
      CUDA_CHECK(cudaMemcpy(mismatch_value, &local_mismatches, sizeof(int),
                            cudaMemcpyHostToDevice));
      NCCL_CHECK(ncclAllReduce(mismatch_value, mismatch_value, 1, ncclInt,
                               ncclMax, communicator, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      int maximum_mismatches = 0;
      CUDA_CHECK(cudaMemcpy(&maximum_mismatches, mismatch_value, sizeof(int),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(mismatch_value));

      measurements.push_back(
          Measurement{operation, bytes, samples, maximum_mismatches});
      if (rank == 0) {
        std::printf("[lane] op=%s bytes=%zu median_us=%.6f mismatches=%d\n",
                    operation_name(operation), bytes, median(samples),
                    maximum_mismatches);
        std::fflush(stdout);
      }
    }
  }

  if (rank == 0) {
    write_result(output_path, world, tasks_per_node, visible_devices,
                 nccl_version, host, properties.name, pci_bus_id, measurements);
  }

  int worst_mismatches = 0;
  for (const Measurement& measurement : measurements) {
    worst_mismatches =
        std::max(worst_mismatches, measurement.max_rank_mismatches);
  }

  CUDA_CHECK(cudaFree(all_gather_send));
  CUDA_CHECK(cudaFree(all_gather_receive));
  CUDA_CHECK(cudaFree(reduce_scatter_send));
  CUDA_CHECK(cudaFree(reduce_scatter_receive));
  CUDA_CHECK(cudaFree(barrier_value));
  CUDA_CHECK(cudaFree(sample_value));
  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaStreamDestroy(stream));
  NCCL_CHECK(ncclCommDestroy(communicator));

  if (worst_mismatches != 0) {
    std::fprintf(stderr, "FATAL value-conservation mismatch count %d\n",
                 worst_mismatches);
    return 27;
  }
  std::printf("[lane] rank=%d clean\n", rank);
  return 0;
}
