#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

static void blas_check(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS operation failed: " + std::to_string(result));
}

__global__ void ordered_row_sum(const float* input, float* output, int rows, int columns) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    float sum = 0;
    for (int row = 0; row < rows; ++row) sum += input[row + column * rows];
    output[column] = sum;
}

static int positive_arg(const char* value) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (!end || *end || parsed <= 0 || parsed > 4096)
        throw std::runtime_error("dimensions and repetitions must be in [1,4096]");
    return static_cast<int>(parsed);
}

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 5)
            throw std::runtime_error("usage: sm70_serial_probe [rows columns inner repetitions]");
        const int rows = argc == 5 ? positive_arg(argv[1]) : 16;
        const int columns = argc == 5 ? positive_arg(argv[2]) : 128;
        const int inner = argc == 5 ? positive_arg(argv[3]) : 128;
        const int repetitions = argc == 5 ? positive_arg(argv[4]) : 4;
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, 0));
        if (properties.major != 7 || properties.minor != 0)
            throw std::runtime_error("this capture probe requires an actual SM70 GPU");

        std::vector<__half> left(rows * inner), right(inner * columns);
        for (size_t index = 0; index < left.size(); ++index)
            left[index] = __float2half((static_cast<int>(index % 7) - 3) / 32.0f);
        for (size_t index = 0; index < right.size(); ++index)
            right[index] = __float2half((static_cast<int>(index % 5) - 2) / 32.0f);
        std::vector<float> result(rows * columns), sum(columns);
        __half* device_left = nullptr;
        __half* device_right = nullptr;
        float* device_result = nullptr;
        float* device_sum = nullptr;
        cuda_check(cudaMalloc(&device_left, left.size() * sizeof(__half)));
        cuda_check(cudaMalloc(&device_right, right.size() * sizeof(__half)));
        cuda_check(cudaMalloc(&device_result, result.size() * sizeof(float)));
        cuda_check(cudaMalloc(&device_sum, sum.size() * sizeof(float)));
        cuda_check(cudaMemcpy(device_left, left.data(), left.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(device_right, right.data(), right.size() * sizeof(__half),
                              cudaMemcpyHostToDevice));
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cublasHandle_t handle;
        blas_check(cublasCreate(&handle));
        blas_check(cublasSetStream(handle, stream));
        blas_check(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        cudaEvent_t begin, after_gemm, after_sum;
        cuda_check(cudaEventCreate(&begin));
        cuda_check(cudaEventCreate(&after_gemm));
        cuda_check(cudaEventCreate(&after_sum));
        const float alpha = 1, beta = 0;
        std::vector<float> gemm_ms(repetitions), sum_ms(repetitions);
        for (int repeat = 0; repeat < repetitions; ++repeat) {
            cuda_check(cudaEventRecord(begin, stream));
            blas_check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   rows, columns, inner, &alpha,
                                   device_left, CUDA_R_16F, rows,
                                   device_right, CUDA_R_16F, inner, &beta,
                                   device_result, CUDA_R_32F, rows,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            cuda_check(cudaEventRecord(after_gemm, stream));
            ordered_row_sum<<<(columns + 127) / 128, 128, 0, stream>>>(
                device_result, device_sum, rows, columns);
            cuda_check(cudaGetLastError());
            cuda_check(cudaEventRecord(after_sum, stream));
            cuda_check(cudaEventSynchronize(after_sum));
            cuda_check(cudaEventElapsedTime(&gemm_ms[repeat], begin, after_gemm));
            cuda_check(cudaEventElapsedTime(&sum_ms[repeat], after_gemm, after_sum));
        }
        cuda_check(cudaMemcpy(result.data(), device_result, result.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(sum.data(), device_sum, sum.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double maximum_error = 0;
        for (int column = 0; column < columns; ++column) {
            float expected_sum = 0;
            for (int row = 0; row < rows; ++row) {
                if (!std::isfinite(result[row + column * rows]))
                    throw std::runtime_error("GPU matrix contains a nonfinite value");
                float expected = 0;
                for (int k = 0; k < inner; ++k)
                    expected += __half2float(left[row + k * rows]) *
                                __half2float(right[k + column * inner]);
                maximum_error = std::fmax(maximum_error,
                                          std::fabs(expected - result[row + column * rows]));
                expected_sum += expected;
            }
            if (!std::isfinite(sum[column]))
                throw std::runtime_error("GPU row sum contains a nonfinite value");
            maximum_error = std::fmax(maximum_error, std::fabs(expected_sum - sum[column]));
        }
        if (maximum_error > 0.00001) throw std::runtime_error("CPU numerical oracle failed");
        std::printf("{\"schema\":\"simllm-sm70-serial-probe-v1\",\"architecture\":\"sm70\","
                    "\"kernel_concurrency\":1,\"stream_count\":1,\"dtype\":\"fp16-input-fp32-output\","
                    "\"rows\":%d,\"columns\":%d,\"inner\":%d,\"repetitions\":%d,"
                    "\"evidence\":\"diagnostic-region-timing-including-first-use-and-host-gaps\","
                    "\"maximum_absolute_error\":%.9g,\"regions\":[",
                    rows, columns, inner, repetitions, maximum_error);
        for (int repeat = 0; repeat < repetitions; ++repeat)
            std::printf("%s{\"repeat\":%d,\"gemm_ms\":%.9g,\"ordered_sum_ms\":%.9g}",
                        repeat ? "," : "", repeat, gemm_ms[repeat], sum_ms[repeat]);
        std::printf("]}\n");
        cuda_check(cudaEventDestroy(begin));
        cuda_check(cudaEventDestroy(after_gemm));
        cuda_check(cudaEventDestroy(after_sum));
        blas_check(cublasDestroy(handle));
        cuda_check(cudaStreamDestroy(stream));
        cuda_check(cudaFree(device_left));
        cuda_check(cudaFree(device_right));
        cuda_check(cudaFree(device_result));
        cuda_check(cudaFree(device_sum));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
