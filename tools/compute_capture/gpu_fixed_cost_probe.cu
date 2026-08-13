// Turing probe for the compute-fidelity study.
//
// Two independent measurements the calibration capture cannot make:
//
//   --stability  replays the exact calibration cell that failed the frozen
//                2 percent coefficient-of-variation guard (attn_gemm FP32
//                shape 8) and records, per launch, both the wall duration and
//                the SM cycle span each block actually spent resident. A wall
//                excursion with a flat cycle count is a clock or occupancy
//                effect; a wall excursion whose cycle count rises with it is
//                the block being resident longer, i.e. the SM was shared.
//                The two are physically different and the calibration capture
//                cannot tell them apart from durations alone.
//
//   --launch     measures the fixed per-launch cost the modeled compute path
//                omits entirely: host enqueue cost, eager stream throughput,
//                launch-plus-synchronize latency, CUDA-graph replay cost, and
//                the device-side inter-kernel gap of a real kernel.
//
// Both modes print machine-readable text on stdout and nothing else, so the
// study harness can parse them without a profiler.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr std::size_t kStabilityWorkItems = 2097152;
constexpr std::size_t kLongWorkItems = 262144;
constexpr std::size_t kFlushBytes = 33554432;
constexpr int kStabilityWarmups = 100;
constexpr int kStatsPerLaunch = 5;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ unsigned long long globalTimerNs() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Body identical to simllm_attn_gemm_kernel<float> in
// compute_family_benchmark.cu, plus per-block clock64 and globaltimer spans.
__global__ void simllm_probe_attn_gemm_kernel(
    float* output,
    const float* left,
    const float* right,
    std::size_t count,
    unsigned long long* stats,
    int launchIndex) {
    const unsigned long long blockStartNs = globalTimerNs();
    const unsigned long long blockStartCycles =
        static_cast<unsigned long long>(clock64());

    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = left[index];
#pragma unroll
        for (int iteration = 0; iteration < 8; ++iteration) {
            value = fmaf(value, 1.0001f, right[index]);
        }
        output[index] = value;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const unsigned long long blockEndCycles =
            static_cast<unsigned long long>(clock64());
        const unsigned long long blockEndNs = globalTimerNs();
        unsigned long long* slot =
            stats + static_cast<unsigned long long>(kStatsPerLaunch) * launchIndex;
        atomicAdd(slot + 0, blockEndCycles - blockStartCycles);
        atomicAdd(slot + 1, blockEndNs - blockStartNs);
        atomicMin(slot + 2, blockStartNs);
        atomicMax(slot + 3, blockEndNs);
        atomicAdd(slot + 4, 1ULL);
    }
}

// Same body without instrumentation, used for the launch-cost measurements.
__global__ void simllm_probe_plain_kernel(
    float* output,
    const float* left,
    const float* right,
    std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = left[index];
#pragma unroll
    for (int iteration = 0; iteration < 8; ++iteration) {
        value = fmaf(value, 1.0001f, right[index]);
    }
    output[index] = value;
}

__global__ void simllm_probe_flush_kernel(std::uint32_t* buffer, std::size_t words) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < words) {
        buffer[index] = buffer[index] + 1u;
    }
}

__global__ void simllm_probe_empty_kernel() {}

__global__ void simllm_probe_stats_init_kernel(
    unsigned long long* stats,
    std::size_t slots) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= slots) {
        return;
    }
    const std::size_t field = index % kStatsPerLaunch;
    // sums start at zero, the minimum starts saturated, the maximum at zero
    stats[index] = (field == 2) ? ~0ULL : 0ULL;
}

struct DeviceBuffers {
    float* output = nullptr;
    float* left = nullptr;
    float* right = nullptr;
    std::size_t count = 0;

    explicit DeviceBuffers(std::size_t elements) : count(elements) {
        checkCuda(cudaMalloc(&output, count * sizeof(float)), "cudaMalloc output");
        checkCuda(cudaMalloc(&left, count * sizeof(float)), "cudaMalloc left");
        checkCuda(cudaMalloc(&right, count * sizeof(float)), "cudaMalloc right");
        checkCuda(cudaMemset(output, 0, count * sizeof(float)), "cudaMemset output");
        checkCuda(cudaMemset(left, 1, count * sizeof(float)), "cudaMemset left");
        checkCuda(cudaMemset(right, 2, count * sizeof(float)), "cudaMemset right");
    }

    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    ~DeviceBuffers() {
        cudaFree(output);
        cudaFree(left);
        cudaFree(right);
    }
};

unsigned int blocksFor(std::size_t items) {
    return static_cast<unsigned int>((items + kThreads - 1) / kThreads);
}

void printDeviceIdentity(const cudaDeviceProp& properties) {
    std::printf("device_name,%s\n", properties.name);
    const unsigned char* uuidBytes =
        reinterpret_cast<const unsigned char*>(properties.uuid.bytes);
    std::printf("device_uuid,GPU-");
    for (int index = 0; index < 16; ++index) {
        std::printf("%02x", static_cast<unsigned int>(uuidBytes[index]));
        if (index == 3 || index == 5 || index == 7 || index == 9) {
            std::printf("-");
        }
    }
    std::printf("\n");
    std::printf("compute_capability,%d.%d\n", properties.major, properties.minor);
    std::printf("multiprocessor_count,%d\n", properties.multiProcessorCount);
    std::printf("nominal_clock_khz,%d\n", properties.clockRate);
    std::printf("memory_clock_khz,%d\n", properties.memoryClockRate);
    std::printf("memory_bus_width_bits,%d\n", properties.memoryBusWidth);
}

int runStability(int launches) {
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");

    DeviceBuffers buffers(kStabilityWorkItems);
    std::uint32_t* flushBuffer = nullptr;
    checkCuda(cudaMalloc(&flushBuffer, kFlushBytes), "cudaMalloc flush buffer");
    checkCuda(cudaMemset(flushBuffer, 0, kFlushBytes), "cudaMemset flush buffer");
    const std::size_t flushWords = kFlushBytes / sizeof(std::uint32_t);

    const std::size_t slots =
        static_cast<std::size_t>(launches) * kStatsPerLaunch;
    unsigned long long* stats = nullptr;
    checkCuda(cudaMalloc(&stats, slots * sizeof(unsigned long long)),
              "cudaMalloc stats");
    simllm_probe_stats_init_kernel<<<blocksFor(slots), kThreads>>>(stats, slots);
    checkCuda(cudaGetLastError(), "stats init launch");
    checkCuda(cudaDeviceSynchronize(), "stats init synchronization");

    const unsigned int targetBlocks = blocksFor(kStabilityWorkItems);
    for (int warmup = 0; warmup < kStabilityWarmups; ++warmup) {
        simllm_probe_attn_gemm_kernel<<<targetBlocks, kThreads>>>(
            buffers.output, buffers.left, buffers.right, buffers.count, stats, 0);
    }
    checkCuda(cudaGetLastError(), "warmup launch");
    checkCuda(cudaDeviceSynchronize(), "warmup synchronization");
    // discard whatever the warmups accumulated into slot 0
    simllm_probe_stats_init_kernel<<<blocksFor(slots), kThreads>>>(stats, slots);
    checkCuda(cudaGetLastError(), "stats reinit launch");
    checkCuda(cudaDeviceSynchronize(), "stats reinit synchronization");

    std::vector<cudaEvent_t> starts(launches);
    std::vector<cudaEvent_t> stops(launches);
    for (int index = 0; index < launches; ++index) {
        checkCuda(cudaEventCreate(&starts[index]), "cudaEventCreate start");
        checkCuda(cudaEventCreate(&stops[index]), "cudaEventCreate stop");
    }

    for (int index = 0; index < launches; ++index) {
        simllm_probe_flush_kernel<<<blocksFor(flushWords), kThreads>>>(
            flushBuffer, flushWords);
        checkCuda(cudaEventRecord(starts[index]), "cudaEventRecord start");
        simllm_probe_attn_gemm_kernel<<<targetBlocks, kThreads>>>(
            buffers.output, buffers.left, buffers.right, buffers.count, stats, index);
        checkCuda(cudaEventRecord(stops[index]), "cudaEventRecord stop");
    }
    checkCuda(cudaGetLastError(), "stability launch");
    checkCuda(cudaDeviceSynchronize(), "stability synchronization");

    std::vector<unsigned long long> hostStats(slots, 0ULL);
    checkCuda(cudaMemcpy(hostStats.data(), stats,
                         slots * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
              "cudaMemcpy stats");

    printDeviceIdentity(properties);
    std::printf("mode,stability\n");
    std::printf("launches,%d\n", launches);
    std::printf("work_items,%zu\n", kStabilityWorkItems);
    std::printf("threads_per_block,%d\n", kThreads);
    std::printf("blocks,%u\n", targetBlocks);
    std::printf("warmups,%d\n", kStabilityWarmups);
    std::printf(
        "samples_begin,launch,event_duration_ns,kernel_span_ns,"
        "block_cycle_sum,block_resident_ns_sum,block_count\n");
    for (int index = 0; index < launches; ++index) {
        float milliseconds = 0.0F;
        checkCuda(cudaEventElapsedTime(&milliseconds, starts[index], stops[index]),
                  "cudaEventElapsedTime");
        const unsigned long long* slot =
            hostStats.data() + static_cast<std::size_t>(index) * kStatsPerLaunch;
        const unsigned long long spanNs =
            (slot[3] >= slot[2]) ? (slot[3] - slot[2]) : 0ULL;
        std::printf("sample,%d,%.0f,%llu,%llu,%llu,%llu\n",
                    index,
                    static_cast<double>(milliseconds) * 1.0e6,
                    spanNs,
                    slot[0],
                    slot[1],
                    slot[4]);
    }
    std::printf("samples_end\n");

    for (int index = 0; index < launches; ++index) {
        cudaEventDestroy(starts[index]);
        cudaEventDestroy(stops[index]);
    }
    cudaFree(stats);
    cudaFree(flushBuffer);
    return EXIT_SUCCESS;
}

double medianOf(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

int runLaunch(int iterations, int graphNodes, int graphReplays,
              int isolatedIterations, int backToBackIterations) {
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");

    // Empty-kernel launch costs, host timed.
    for (int warmup = 0; warmup < 1000; ++warmup) {
        simllm_probe_empty_kernel<<<1, 1>>>();
    }
    checkCuda(cudaGetLastError(), "empty warmup launch");
    checkCuda(cudaDeviceSynchronize(), "empty warmup synchronization");

    const auto enqueueStart = std::chrono::steady_clock::now();
    for (int index = 0; index < iterations; ++index) {
        simllm_probe_empty_kernel<<<1, 1>>>();
    }
    const auto enqueueEnd = std::chrono::steady_clock::now();
    checkCuda(cudaGetLastError(), "empty pipelined launch");
    checkCuda(cudaDeviceSynchronize(), "empty pipelined synchronization");
    const auto pipelinedEnd = std::chrono::steady_clock::now();

    const double enqueueNs =
        std::chrono::duration<double, std::nano>(enqueueEnd - enqueueStart).count()
        / iterations;
    const double pipelinedNs =
        std::chrono::duration<double, std::nano>(pipelinedEnd - enqueueStart).count()
        / iterations;

    const int serializedIterations = std::min(iterations, 5000);
    const auto serializedStart = std::chrono::steady_clock::now();
    for (int index = 0; index < serializedIterations; ++index) {
        simllm_probe_empty_kernel<<<1, 1>>>();
        checkCuda(cudaDeviceSynchronize(), "empty serialized synchronization");
    }
    const auto serializedEnd = std::chrono::steady_clock::now();
    const double serializedNs =
        std::chrono::duration<double, std::nano>(serializedEnd - serializedStart).count()
        / serializedIterations;

    // CUDA-graph replay cost for the same empty launches.
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
              "cudaStreamBeginCapture");
    for (int index = 0; index < graphNodes; ++index) {
        simllm_probe_empty_kernel<<<1, 1, 0, stream>>>();
    }
    cudaGraph_t graph = nullptr;
    checkCuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    cudaGraphExec_t graphExec = nullptr;
    checkCuda(cudaGraphInstantiate(&graphExec, graph, 0), "cudaGraphInstantiate");
    checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch warmup");
    checkCuda(cudaStreamSynchronize(stream), "graph warmup synchronization");

    const auto graphStart = std::chrono::steady_clock::now();
    for (int replay = 0; replay < graphReplays; ++replay) {
        checkCuda(cudaGraphLaunch(graphExec, stream), "cudaGraphLaunch");
    }
    checkCuda(cudaStreamSynchronize(stream), "graph synchronization");
    const auto graphEnd = std::chrono::steady_clock::now();
    const double graphNs =
        std::chrono::duration<double, std::nano>(graphEnd - graphStart).count()
        / (static_cast<double>(graphReplays) * graphNodes);

    // Device-side inter-kernel gap of a real kernel.
    DeviceBuffers buffers(kLongWorkItems);
    const unsigned int longBlocks = blocksFor(kLongWorkItems);
    for (int warmup = 0; warmup < 50; ++warmup) {
        simllm_probe_plain_kernel<<<longBlocks, kThreads>>>(
            buffers.output, buffers.left, buffers.right, buffers.count);
    }
    checkCuda(cudaGetLastError(), "long warmup launch");
    checkCuda(cudaDeviceSynchronize(), "long warmup synchronization");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate isolated start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate isolated stop");

    std::vector<double> isolated;
    isolated.reserve(isolatedIterations);
    for (int index = 0; index < isolatedIterations; ++index) {
        checkCuda(cudaEventRecord(start), "cudaEventRecord isolated start");
        simllm_probe_plain_kernel<<<longBlocks, kThreads>>>(
            buffers.output, buffers.left, buffers.right, buffers.count);
        checkCuda(cudaEventRecord(stop), "cudaEventRecord isolated stop");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize isolated");
        float milliseconds = 0.0F;
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop),
                  "cudaEventElapsedTime isolated");
        isolated.push_back(static_cast<double>(milliseconds) * 1.0e6);
    }
    const double isolatedNs = medianOf(isolated);

    checkCuda(cudaEventRecord(start), "cudaEventRecord batch start");
    for (int index = 0; index < backToBackIterations; ++index) {
        simllm_probe_plain_kernel<<<longBlocks, kThreads>>>(
            buffers.output, buffers.left, buffers.right, buffers.count);
    }
    checkCuda(cudaEventRecord(stop), "cudaEventRecord batch stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize batch");
    float batchMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&batchMilliseconds, start, stop),
              "cudaEventElapsedTime batch");
    const double backToBackNs =
        static_cast<double>(batchMilliseconds) * 1.0e6 / backToBackIterations;

    // The isolated event window necessarily contains the empty-queue launch
    // latency, so isolated minus back-to-back does not isolate the device-side
    // inter-kernel gap. Measure that gap directly instead: run the same shape
    // back to back through the instrumented kernel, which stamps its own
    // start and end from the always-on nanosecond global timer, and subtract
    // the summed in-kernel spans from the wall time of the whole batch.
    const std::size_t gapSlots =
        static_cast<std::size_t>(backToBackIterations) * kStatsPerLaunch;
    unsigned long long* gapStats = nullptr;
    checkCuda(cudaMalloc(&gapStats, gapSlots * sizeof(unsigned long long)),
              "cudaMalloc gap stats");
    simllm_probe_stats_init_kernel<<<blocksFor(gapSlots), kThreads>>>(
        gapStats, gapSlots);
    checkCuda(cudaGetLastError(), "gap stats init launch");
    checkCuda(cudaDeviceSynchronize(), "gap stats init synchronization");

    checkCuda(cudaEventRecord(start), "cudaEventRecord stamped batch start");
    for (int index = 0; index < backToBackIterations; ++index) {
        simllm_probe_attn_gemm_kernel<<<longBlocks, kThreads>>>(
            buffers.output, buffers.left, buffers.right, buffers.count,
            gapStats, index);
    }
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stamped batch stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stamped batch");
    float stampedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&stampedMilliseconds, start, stop),
              "cudaEventElapsedTime stamped batch");
    std::vector<unsigned long long> hostGapStats(gapSlots, 0ULL);
    checkCuda(cudaMemcpy(hostGapStats.data(), gapStats,
                         gapSlots * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy gap stats");
    double stampedSpanNs = 0.0;
    for (int index = 0; index < backToBackIterations; ++index) {
        const unsigned long long* slot =
            hostGapStats.data() + static_cast<std::size_t>(index) * kStatsPerLaunch;
        if (slot[3] >= slot[2]) {
            stampedSpanNs += static_cast<double>(slot[3] - slot[2]);
        }
    }
    const double stampedWallNs = static_cast<double>(stampedMilliseconds) * 1.0e6;
    const double stampedServiceNs = stampedSpanNs / backToBackIterations;
    const double stampedGapNs =
        (stampedWallNs - stampedSpanNs) / backToBackIterations;
    cudaFree(gapStats);

    printDeviceIdentity(properties);
    std::printf("mode,launch\n");
    std::printf("empty_iterations,%d\n", iterations);
    std::printf("empty_stream_cpu_enqueue_ns,%.3f\n", enqueueNs);
    std::printf("empty_stream_pipelined_ns,%.3f\n", pipelinedNs);
    std::printf("empty_serialized_iterations,%d\n", serializedIterations);
    std::printf("empty_stream_serialized_ns,%.3f\n", serializedNs);
    std::printf("graph_nodes,%d\n", graphNodes);
    std::printf("graph_replays,%d\n", graphReplays);
    std::printf("empty_graph_ns,%.3f\n", graphNs);
    std::printf("long_work_items,%zu\n", kLongWorkItems);
    std::printf("long_isolated_iterations,%d\n", isolatedIterations);
    std::printf("long_isolated_ns,%.3f\n", isolatedNs);
    std::printf("long_backtoback_iterations,%d\n", backToBackIterations);
    std::printf("long_backtoback_ns,%.3f\n", backToBackNs);
    std::printf("device_gap_ns,%.3f\n", backToBackNs - isolatedNs);
    std::printf("stamped_batch_wall_ns,%.3f\n", stampedWallNs);
    std::printf("stamped_kernel_service_ns,%.3f\n", stampedServiceNs);
    std::printf("stamped_device_gap_ns,%.3f\n", stampedGapNs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return EXIT_SUCCESS;
}

int parsePositive(const char* text, const char* name) {
    const long value = std::strtol(text, nullptr, 10);
    if (value <= 0 || value > 1000000) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        checkCuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        if (properties.major != 7 || properties.minor != 5) {
            std::fprintf(stderr,
                         "expected compute capability 7.5, observed %d.%d\n",
                         properties.major,
                         properties.minor);
            return EXIT_FAILURE;
        }

        if (argc >= 2 && std::strcmp(argv[1], "--identity") == 0) {
            printDeviceIdentity(properties);
            return EXIT_SUCCESS;
        }
        if (argc >= 2 && std::strcmp(argv[1], "--stability") == 0) {
            const int launches = (argc >= 3) ? parsePositive(argv[2], "launches") : 4000;
            return runStability(launches);
        }
        if (argc >= 2 && std::strcmp(argv[1], "--launch") == 0) {
            const int iterations = (argc >= 3) ? parsePositive(argv[2], "iterations") : 20000;
            const int graphNodes = (argc >= 4) ? parsePositive(argv[3], "graph nodes") : 512;
            const int graphReplays = (argc >= 5) ? parsePositive(argv[4], "graph replays") : 200;
            const int isolated = (argc >= 6) ? parsePositive(argv[5], "isolated") : 200;
            const int backToBack = (argc >= 7) ? parsePositive(argv[6], "back to back") : 400;
            return runLaunch(iterations, graphNodes, graphReplays, isolated, backToBack);
        }

        std::fprintf(stderr,
                     "usage: %s --identity\n"
                     "       %s --stability [launches]\n"
                     "       %s --launch [iterations] [graph_nodes] [graph_replays] "
                     "[isolated] [back_to_back]\n",
                     argv[0],
                     argv[0],
                     argv[0]);
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
}
