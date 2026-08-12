#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;
constexpr std::size_t kWorkItemsPerShapeUnit = 262144;
constexpr int kWarmupLaunches = 10;
constexpr int kMeasuredLaunches = 41;
constexpr std::size_t kFlushBytes = 33554432;
constexpr std::array<int, 5> kShapes = {1, 4, 16, 2, 8};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
__device__ T multiplyAdd(T left, T right, T addend);

template <>
__device__ float multiplyAdd(float left, float right, float addend) {
    return fmaf(left, right, addend);
}

template <>
__device__ double multiplyAdd(double left, double right, double addend) {
    return fma(left, right, addend);
}

template <typename T>
__global__ void simllm_attn_gemm_kernel(
    T* output,
    const T* left,
    const T* right,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    T value = left[index];
#pragma unroll
    for (int iteration = 0; iteration < 8; ++iteration) {
        value = multiplyAdd(value, T(1.0001), right[index]);
    }
    output[index] = value;
}

template <typename T>
__global__ void simllm_attn_score_kernel(
    T* output,
    const T* left,
    const T* right,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    T value = left[index];
#pragma unroll
    for (int iteration = 0; iteration < 4; ++iteration) {
        value = multiplyAdd(value, T(0.9999), right[index]);
    }
    output[index] = value;
}

template <typename T>
__global__ void simllm_mlp_gemm_kernel(
    T* output,
    const T* left,
    const T* right,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    T value = left[index];
#pragma unroll
    for (int iteration = 0; iteration < 16; ++iteration) {
        value = multiplyAdd(value, T(1.0002), right[index]);
    }
    output[index] = value;
}

template <typename T>
__global__ void simllm_lm_head_kernel(
    T* output,
    const T* left,
    const T* right,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    T value = left[index];
#pragma unroll
    for (int iteration = 0; iteration < 12; ++iteration) {
        value = multiplyAdd(value, T(0.9998), right[index]);
    }
    output[index] = value;
}

template <typename T>
__global__ void simllm_kv_read_kernel(
    T* output,
    const T* input,
    const T*,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
    }
}

__global__ void simllm_cache_flush_kernel(
    std::uint32_t* buffer,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        buffer[index] = static_cast<std::uint32_t>(index);
    }
}

enum class Family {
    AttnGemm,
    AttnScore,
    MlpGemm,
    LmHead,
    KvRead,
};

constexpr std::array<Family, 5> kFamilies = {
    Family::AttnGemm,
    Family::AttnScore,
    Family::MlpGemm,
    Family::LmHead,
    Family::KvRead,
};

template <typename T>
struct Buffers {
    T* output = nullptr;
    T* left = nullptr;
    T* right = nullptr;
    std::size_t count = 0;

    explicit Buffers(std::size_t elements) : count(elements) {
        checkCuda(cudaMalloc(&output, count * sizeof(T)), "cudaMalloc output");
        checkCuda(cudaMalloc(&left, count * sizeof(T)), "cudaMalloc left");
        checkCuda(cudaMalloc(&right, count * sizeof(T)), "cudaMalloc right");
        checkCuda(cudaMemset(output, 0, count * sizeof(T)), "cudaMemset output");
        checkCuda(cudaMemset(left, 1, count * sizeof(T)), "cudaMemset left");
        checkCuda(cudaMemset(right, 2, count * sizeof(T)), "cudaMemset right");
    }

    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;

    ~Buffers() {
        cudaFree(output);
        cudaFree(left);
        cudaFree(right);
    }
};

template <typename T>
void launchTarget(Family family, Buffers<T>& buffers, std::size_t count) {
    const unsigned int blocks = static_cast<unsigned int>(
        (count + kThreads - 1) / kThreads);
    switch (family) {
        case Family::AttnGemm:
            simllm_attn_gemm_kernel<T><<<blocks, kThreads>>>(
                buffers.output, buffers.left, buffers.right, count);
            break;
        case Family::AttnScore:
            simllm_attn_score_kernel<T><<<blocks, kThreads>>>(
                buffers.output, buffers.left, buffers.right, count);
            break;
        case Family::MlpGemm:
            simllm_mlp_gemm_kernel<T><<<blocks, kThreads>>>(
                buffers.output, buffers.left, buffers.right, count);
            break;
        case Family::LmHead:
            simllm_lm_head_kernel<T><<<blocks, kThreads>>>(
                buffers.output, buffers.left, buffers.right, count);
            break;
        case Family::KvRead:
            simllm_kv_read_kernel<T><<<blocks, kThreads>>>(
                buffers.output, buffers.left, buffers.right, count);
            break;
    }
    checkCuda(cudaGetLastError(), "target kernel launch");
}

template <typename T>
void captureCell(
    Buffers<T>& buffers,
    Family family,
    int shape,
    std::uint32_t* flushBuffer) {
    const std::size_t flushWords = kFlushBytes / sizeof(std::uint32_t);
    const unsigned int flushBlocks = static_cast<unsigned int>(
        (flushWords + kThreads - 1) / kThreads);
    const std::size_t count = kWorkItemsPerShapeUnit * shape;

    for (int warmup = 0; warmup < kWarmupLaunches; ++warmup) {
        launchTarget(family, buffers, count);
    }
    checkCuda(cudaDeviceSynchronize(), "cell warmup synchronization");

    checkCuda(cudaProfilerStart(), "cudaProfilerStart");
    for (int sample = 0; sample < kMeasuredLaunches; ++sample) {
        simllm_cache_flush_kernel<<<flushBlocks, kThreads>>>(
            flushBuffer, flushWords);
        checkCuda(cudaGetLastError(), "cache flush kernel launch");
        launchTarget(family, buffers, count);
    }
    checkCuda(cudaDeviceSynchronize(), "cell capture synchronization");
    checkCuda(cudaProfilerStop(), "cudaProfilerStop");
}

template <typename T>
void captureMatrix(Buffers<T>& buffers, std::uint32_t* flushBuffer) {
    for (Family family : kFamilies) {
        for (int shape : kShapes) {
            captureCell(buffers, family, shape, flushBuffer);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        int device = 0;
        checkCuda(cudaSetDevice(device), "cudaSetDevice");
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, device),
                  "cudaGetDeviceProperties");
        if (properties.major != 7 || properties.minor != 5) {
            std::fprintf(stderr,
                         "expected compute capability 7.5, observed %d.%d\n",
                         properties.major,
                         properties.minor);
            return EXIT_FAILURE;
        }

        if (argc == 2 && std::string(argv[1]) == "--counter-smoke") {
            Buffers<float> smoke(kWorkItemsPerShapeUnit);
            launchTarget(Family::KvRead, smoke, kWorkItemsPerShapeUnit);
            checkCuda(cudaDeviceSynchronize(), "counter smoke synchronization");
            std::printf("counter smoke completed\n");
            return EXIT_SUCCESS;
        }
        if (argc != 1) {
            std::fprintf(stderr, "usage: %s [--counter-smoke]\n", argv[0]);
            return EXIT_FAILURE;
        }

        const std::size_t maxCount = kWorkItemsPerShapeUnit * 16;
        Buffers<float> fp32(maxCount);
        Buffers<double> fp64(maxCount);
        std::uint32_t* flushBuffer = nullptr;
        checkCuda(cudaMalloc(&flushBuffer, kFlushBytes),
                  "cudaMalloc cache flush buffer");

        captureMatrix(fp32, flushBuffer);
        captureMatrix(fp64, flushBuffer);

        checkCuda(cudaFree(flushBuffer), "cudaFree cache flush buffer");
        std::printf(
            "captured device=%s cells=50 target_launches=2050 warmups=500\n",
            properties.name);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
}
