#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 16'777'216;
constexpr unsigned int kThreadsPerBlock = 256;
constexpr int kWarmupLaunches = 5;
constexpr int kDefaultMeasuredLaunches = 1;
constexpr std::uint64_t kFnvOffsetBasis = 14'695'981'039'346'656'037ULL;
constexpr std::uint64_t kFnvPrime = 1'099'511'628'211ULL;

static_assert(kElementCount % kThreadsPerBlock == 0);

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceFloatBuffer {
  public:
    explicit DeviceFloatBuffer(std::size_t elements) {
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("device allocation byte count overflow");
        }
        checkCuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(float)),
            "cudaMalloc");
    }

    DeviceFloatBuffer(const DeviceFloatBuffer&) = delete;
    DeviceFloatBuffer& operator=(const DeviceFloatBuffer&) = delete;

    ~DeviceFloatBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    float* get() const { return data_; }

    void release(const char* operation) {
        if (data_ == nullptr) {
            return;
        }
        float* allocation = data_;
        data_ = nullptr;
        checkCuda(cudaFree(allocation), operation);
    }

  private:
    float* data_ = nullptr;
};

extern "C" __global__ void simllm_a100_environment_probe_vector_add_fp32_v1(
    const float* left,
    const float* right,
    float* output,
    std::size_t element_count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = left[index] + right[index];
    }
}

struct ProbeConfig {
    int measured_launches = kDefaultMeasuredLaunches;
};

unsigned long long parsePositiveInteger(const char* spelling, const char* option) {
    if (spelling[0] == '\0' || spelling[0] == '-') {
        throw std::invalid_argument(std::string(option) + " must be a positive integer");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(spelling, &end, 10);
    if (errno == ERANGE || end == nullptr || *end != '\0' || parsed == 0) {
        throw std::invalid_argument(std::string(option) + " must be a positive integer");
    }
    return parsed;
}

ProbeConfig parseArguments(int argc, char** argv) {
    ProbeConfig config;
    bool saw_elements = false;
    bool saw_threads = false;
    bool saw_warmups = false;
    bool saw_measured = false;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument(
                std::string("usage: ") + argv[0] +
                " [--elements N] [--threads N] [--warmups N] [--measured N]");
        }
        const std::string option = argv[index];
        const unsigned long long value =
            parsePositiveInteger(argv[index + 1], option.c_str());
        if (option == "--elements") {
            if (saw_elements || value != kElementCount) {
                throw std::invalid_argument(
                    "--elements must appear at most once and equal 16777216");
            }
            saw_elements = true;
        } else if (option == "--threads") {
            if (saw_threads || value != kThreadsPerBlock) {
                throw std::invalid_argument(
                    "--threads must appear at most once and equal 256");
            }
            saw_threads = true;
        } else if (option == "--warmups") {
            if (saw_warmups || value != kWarmupLaunches) {
                throw std::invalid_argument(
                    "--warmups must appear at most once and equal 5");
            }
            saw_warmups = true;
        } else if (option == "--measured" || option == "--measured-launches") {
            if (saw_measured || value > INT_MAX) {
                throw std::invalid_argument(
                    "measured launch count must appear at most once and be in [1, INT_MAX]");
            }
            config.measured_launches = static_cast<int>(value);
            saw_measured = true;
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return config;
}

std::string deviceUuid(const cudaUUID_t& uuid) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string value = "GPU-";
    value.reserve(40);
    for (int index = 0; index < 16; ++index) {
        const unsigned int byte =
            static_cast<unsigned char>(uuid.bytes[index]);
        value.push_back(kHex[(byte >> 4U) & 0x0fU]);
        value.push_back(kHex[byte & 0x0fU]);
        if (index == 3 || index == 5 || index == 7 || index == 9) {
            value.push_back('-');
        }
    }
    return value;
}

std::uint32_t floatBits(float value) {
    static_assert(sizeof(value) == sizeof(std::uint32_t));
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t appendFnv1a(std::uint64_t hash, std::uint32_t bits) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        hash ^= (bits >> shift) & 0xffU;
        hash *= kFnvPrime;
    }
    return hash;
}

void launchProbe(
    const DeviceFloatBuffer& left,
    const DeviceFloatBuffer& right,
    const DeviceFloatBuffer& output,
    unsigned int grid_blocks) {
    simllm_a100_environment_probe_vector_add_fp32_v1
        <<<grid_blocks, kThreadsPerBlock>>>(
            left.get(), right.get(), output.get(), kElementCount);
    checkCuda(cudaGetLastError(), "vector-add kernel launch");
}

int runProbe(int measured_launches) {
    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count != 1) {
        throw std::runtime_error(
            "expected exactly one CUDA-visible device, observed " +
            std::to_string(device_count));
    }

    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    if (properties.major != 8 || properties.minor != 0) {
        throw std::runtime_error(
            "expected compute capability 8.0, observed " +
            std::to_string(properties.major) + "." +
            std::to_string(properties.minor));
    }

    char pci_bus_id[32]{};
    checkCuda(
        cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0),
        "cudaDeviceGetPCIBusId");

    int driver_version = 0;
    int runtime_version = 0;
    checkCuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    checkCuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");

    std::vector<float> host_left(kElementCount);
    std::vector<float> host_right(kElementCount);
    std::vector<float> host_output(kElementCount, 0.0F);
    for (std::size_t index = 0; index < kElementCount; ++index) {
        host_left[index] = static_cast<float>(index & 0x3ffU);
        host_right[index] =
            static_cast<float>((index * 17U + 3U) & 0x3ffU);
    }

    DeviceFloatBuffer device_left(kElementCount);
    DeviceFloatBuffer device_right(kElementCount);
    DeviceFloatBuffer device_output(kElementCount);
    const std::size_t buffer_bytes = kElementCount * sizeof(float);
    checkCuda(
        cudaMemcpy(
            device_left.get(), host_left.data(), buffer_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy left host-to-device");
    checkCuda(
        cudaMemcpy(
            device_right.get(), host_right.data(), buffer_bytes,
            cudaMemcpyHostToDevice),
        "cudaMemcpy right host-to-device");
    checkCuda(cudaMemset(device_output.get(), 0, buffer_bytes), "cudaMemset output");

    const unsigned int grid_blocks = static_cast<unsigned int>(
        (kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock);
    for (int launch = 0; launch < kWarmupLaunches; ++launch) {
        launchProbe(device_left, device_right, device_output, grid_blocks);
    }
    checkCuda(cudaDeviceSynchronize(), "warmup synchronization");

    checkCuda(cudaProfilerStart(), "cudaProfilerStart");
    for (int launch = 0; launch < measured_launches; ++launch) {
        launchProbe(device_left, device_right, device_output, grid_blocks);
    }
    checkCuda(cudaDeviceSynchronize(), "measured synchronization");
    checkCuda(cudaProfilerStop(), "cudaProfilerStop");

    checkCuda(
        cudaMemcpy(
            host_output.data(), device_output.get(), buffer_bytes,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy output device-to-host");

    std::size_t mismatch_count = 0;
    std::size_t first_mismatch = 0;
    std::uint32_t first_expected_bits = 0;
    std::uint32_t first_observed_bits = 0;
    std::uint64_t expected_checksum = kFnvOffsetBasis;
    std::uint64_t observed_checksum = kFnvOffsetBasis;
    for (std::size_t index = 0; index < kElementCount; ++index) {
        const std::uint32_t expected_bits =
            floatBits(host_left[index] + host_right[index]);
        const std::uint32_t observed_bits = floatBits(host_output[index]);
        expected_checksum = appendFnv1a(expected_checksum, expected_bits);
        observed_checksum = appendFnv1a(observed_checksum, observed_bits);
        if (expected_bits != observed_bits) {
            if (mismatch_count == 0) {
                first_mismatch = index;
                first_expected_bits = expected_bits;
                first_observed_bits = observed_bits;
            }
            ++mismatch_count;
        }
    }
    if (mismatch_count != 0 || observed_checksum != expected_checksum) {
        char detail[256]{};
        std::snprintf(
            detail,
            sizeof(detail),
            "full correctness check failed: mismatches=%zu first_index=%zu "
            "expected_bits=0x%08x observed_bits=0x%08x",
            mismatch_count,
            first_mismatch,
            first_expected_bits,
            first_observed_bits);
        throw std::runtime_error(detail);
    }

    device_output.release("cudaFree output");
    device_right.release("cudaFree right");
    device_left.release("cudaFree left");

    std::printf("probe=simllm-a100-environment-qualification-v1\n");
    std::printf(
        "target_kernel=simllm_a100_environment_probe_vector_add_fp32_v1\n");
    std::printf("device_count=%d\n", device_count);
    std::printf("device_name=%s\n", properties.name);
    std::printf("device_uuid=%s\n", deviceUuid(properties.uuid).c_str());
    std::printf("pci_bus_id=%s\n", pci_bus_id);
    std::printf("compute_capability=%d.%d\n", properties.major, properties.minor);
    std::printf("total_global_memory_bytes=%zu\n", properties.totalGlobalMem);
    std::printf("cuda_driver_version=%d\n", driver_version);
    std::printf("cuda_runtime_version=%d\n", runtime_version);
    std::printf("element_count=%zu\n", kElementCount);
    std::printf("threads_per_block=%u\n", kThreadsPerBlock);
    std::printf("grid_blocks=%u\n", grid_blocks);
    std::printf("warmup_launches=%d\n", kWarmupLaunches);
    std::printf("measured_launches=%d\n", measured_launches);
    std::printf("authored_fp32_adds_per_launch=%zu\n", kElementCount);
    std::printf("authored_fp32_reads_per_launch=%zu\n", 2 * kElementCount);
    std::printf("authored_fp32_writes_per_launch=%zu\n", kElementCount);
    std::printf("authored_read_bytes_per_launch=%zu\n", 2 * buffer_bytes);
    std::printf("authored_write_bytes_per_launch=%zu\n", buffer_bytes);
    std::printf("checked_elements=%zu\n", kElementCount);
    std::printf("mismatch_count=%zu\n", mismatch_count);
    std::printf("checksum_algorithm=fnv1a64-fp32-output-bits\n");
    std::printf(
        "expected_checksum=0x%016llx\n",
        static_cast<unsigned long long>(expected_checksum));
    std::printf(
        "output_checksum=0x%016llx\n",
        static_cast<unsigned long long>(observed_checksum));
    std::printf("correctness=PASS\n");
    std::printf("status=PASS\n");
    std::printf("PROBE_GPU_NAME=%s\n", properties.name);
    std::printf("PROBE_COMPUTE_CAPABILITY=%d.%d\n", properties.major, properties.minor);
    std::printf("PROBE_ELEMENTS=%zu\n", kElementCount);
    std::printf("PROBE_THREADS=%u\n", kThreadsPerBlock);
    std::printf("PROBE_WARMUPS=%d\n", kWarmupLaunches);
    std::printf("PROBE_MEASURED=%d\n", measured_launches);
    std::printf(
        "PROBE_CHECKSUM=0x%016llx\n",
        static_cast<unsigned long long>(observed_checksum));
    std::printf("PROBE_PASS=1\n");
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return runProbe(parseArguments(argc, argv).measured_launches);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "status=FAIL\n");
        std::fprintf(stderr, "error=%s\n", error.what());
        return EXIT_FAILURE;
    }
}
