// TRAF-70 corrected NVLink capture with a GPU-free mock build.
//
// Mock compile:
//   c++ -x c++ -std=c++17 -DSIMLLM_NVLINK_MOCK nvlink_packet_lane.cu -o lane
// Hardware compile:
//   nvcc -std=c++17 -arch=sm_80 nvlink_packet_lane.cu -o lane
//
// The executable consumes one tab-separated plan and writes one JSON object
// per point. Hardware rows contain only observed counters and destination
// evidence. Candidate-derived packet counts and raw bytes are intentionally
// absent. Copy-engine transfers are one contiguous batch per flow and stream,
// never one host enqueue per logical message.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "sha256.h"

#ifndef SIMLLM_NVLINK_MOCK
#include <cuda_runtime.h>
#include <nvml.h>
#ifdef SIMLLM_NVLINK_ENABLE_NCCL
#include <nccl.h>
#endif
#endif

namespace {

constexpr std::uint64_t kCounterQuantumBytes = 1024;
constexpr std::size_t kMaximumStreams = 16;

struct Point {
  std::string case_name;
  std::string point_id;
  std::string producer;
  std::uint64_t payload_bytes = 0;
  std::uint64_t message_count = 0;
  int source = 0;
  int destination = 1;
  std::string sources;
  std::string destinations;
  int source_alignment = 0;
  int destination_alignment = 0;
  int access_width = 16;
  int active_lanes = 32;
  std::string lane_mask;
  int stride = 1;
  int stream_count = 1;
  int outstanding = 256;
  int burst_messages = 256;
  int gap_ns = 0;
  int offered_rate_percent = 100;
  std::string pattern;
};

struct Observation {
  double elapsed_us = 0.0;
  double completion_us = 0.0;
  double drain_us = 0.0;
  std::uint64_t logical_bytes = 0;
  std::uint64_t observed_data_bytes = 0;
  std::uint64_t observed_raw_bytes = 0;
  std::string expected_sha256;
  std::string observed_sha256;
  std::string expected_sequence_sha256;
  std::string observed_sequence_sha256;
  std::uint64_t terminal_extents = 0;
  std::uint64_t missing = 0;
  std::uint64_t duplicate = 0;
  std::uint64_t out_of_order = 0;
  std::string throttle_verdict = "NOT_APPLICABLE_MOCK";
  std::uint64_t throttle_reason_mask = 0;
  std::uint64_t copy_engine_host_enqueue_count = 0;
  std::string copy_engine_batch_mode = "not_copy_engine";
  std::string protocol = "transport_observation";
  std::vector<double> flow_completion_us;
};

struct LinkDelta {
  int gpu = 0;
  int link = 0;
  int remote_gpu = -1;
  std::uint64_t data_tx_kib_before = 0;
  std::uint64_t data_tx_kib_after = 0;
  std::uint64_t data_rx_kib_before = 0;
  std::uint64_t data_rx_kib_after = 0;
  std::uint64_t raw_tx_kib_before = 0;
  std::uint64_t raw_tx_kib_after = 0;
  std::uint64_t raw_rx_kib_before = 0;
  std::uint64_t raw_rx_kib_after = 0;
  std::array<std::uint64_t, 5> errors_before{};
  std::array<std::uint64_t, 5> errors_after{};
  std::array<int, 9> statuses{};
};

struct DeviceTelemetry {
  int gpu = 0;
  std::uint64_t clock_event_reasons = 0;
  unsigned int sm_clock_mhz = 0;
  unsigned int memory_clock_mhz = 0;
  unsigned int power_mw = 0;
  unsigned int temperature_c = 0;
  std::array<int, 5> statuses{};
};

struct CapturedObservation {
  Observation summary;
  std::vector<LinkDelta> links;
  std::vector<DeviceTelemetry> telemetry_before;
  std::vector<DeviceTelemetry> telemetry_after;
};

std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> fields;
  std::stringstream stream(text);
  std::string field;
  while (std::getline(stream, field, delimiter)) fields.push_back(field);
  if (!text.empty() && text.back() == delimiter) fields.emplace_back();
  return fields;
}

std::uint64_t parse_u64(const std::string& text, const std::string& name) {
  std::size_t consumed = 0;
  const auto value = std::stoull(text, &consumed);
  if (consumed != text.size()) throw std::runtime_error("invalid " + name + ": " + text);
  return value;
}

int parse_int(const std::string& text, const std::string& name) {
  const auto value = parse_u64(text, name);
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(name + " exceeds int range");
  }
  return static_cast<int>(value);
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const unsigned char character : text) {
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

#ifndef SIMLLM_NVLINK_MOCK
std::uint64_t stable_checksum(const std::string& text) {
  std::uint64_t value = 1469598103934665603ULL;
  for (const unsigned char byte : text) {
    value ^= byte;
    value *= 1099511628211ULL;
  }
  return value;
}
#endif

std::string control_canonical(const Point& point) {
  std::ostringstream out;
  out << point.case_name << '\n' << point.point_id << '\n' << point.producer << '\n'
      << point.payload_bytes << '\n' << point.message_count << '\n' << point.source << '\n'
      << point.destination << '\n' << point.sources << '\n' << point.destinations << '\n'
      << point.source_alignment << '\n' << point.destination_alignment << '\n'
      << point.access_width << '\n' << point.active_lanes << '\n' << point.lane_mask << '\n'
      << point.stride << '\n' << point.stream_count << '\n' << point.outstanding << '\n'
      << point.burst_messages << '\n' << point.gap_ns << '\n'
      << point.offered_rate_percent << '\n' << point.pattern << '\n';
  return out.str();
}

std::string control_sha256(const Point& point) {
  const auto text = control_canonical(point);
  return traf70::sha256_hex(text.data(), text.size());
}

#ifndef SIMLLM_NVLINK_MOCK
unsigned char expected_byte(std::uint64_t message, std::uint64_t offset,
                            std::uint64_t flow_index) {
  std::uint64_t value = message * 0x9e3779b97f4a7c15ULL;
  value ^= offset * 0xbf58476d1ce4e5b9ULL;
  value ^= flow_index * 0x94d049bb133111ebULL;
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  return static_cast<unsigned char>((value ^ (value >> 31)) & 0xffU);
}
#endif

std::uint64_t positive_delta(std::uint64_t before, std::uint64_t after) {
  return after >= before ? after - before : 0;
}

std::vector<Point> read_points(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open points file: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("points file is empty");
  const auto header = split(line, '\t');
  std::map<std::string, std::size_t> index;
  for (std::size_t i = 0; i < header.size(); ++i) index.emplace(header[i], i);
  const std::vector<std::string> required = {
      "case_name", "point_id", "producer", "payload_bytes", "message_count",
      "source", "destination", "sources", "destinations", "source_alignment",
      "destination_alignment", "access_width", "active_lanes", "lane_mask",
      "stride", "stream_count", "outstanding", "burst_messages", "gap_ns",
      "offered_rate_percent", "pattern"};
  for (const auto& name : required) {
    if (index.count(name) == 0) throw std::runtime_error("missing TSV field: " + name);
  }
  std::vector<Point> points;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto fields = split(line, '\t');
    if (fields.size() != header.size()) throw std::runtime_error("malformed TSV row");
    const auto value = [&](const std::string& name) -> const std::string& {
      return fields.at(index.at(name));
    };
    Point point;
    point.case_name = value("case_name");
    point.point_id = value("point_id");
    point.producer = value("producer");
    point.payload_bytes = parse_u64(value("payload_bytes"), "payload_bytes");
    point.message_count = parse_u64(value("message_count"), "message_count");
    point.source = parse_int(value("source"), "source");
    point.destination = parse_int(value("destination"), "destination");
    point.sources = value("sources");
    point.destinations = value("destinations");
    point.source_alignment = parse_int(value("source_alignment"), "source_alignment");
    point.destination_alignment =
        parse_int(value("destination_alignment"), "destination_alignment");
    point.access_width = parse_int(value("access_width"), "access_width");
    point.active_lanes = parse_int(value("active_lanes"), "active_lanes");
    point.lane_mask = value("lane_mask");
    point.stride = parse_int(value("stride"), "stride");
    point.stream_count = parse_int(value("stream_count"), "stream_count");
    point.outstanding = parse_int(value("outstanding"), "outstanding");
    point.burst_messages = parse_int(value("burst_messages"), "burst_messages");
    point.gap_ns = parse_int(value("gap_ns"), "gap_ns");
    point.offered_rate_percent =
        parse_int(value("offered_rate_percent"), "offered_rate_percent");
    point.pattern = value("pattern");
    if (point.payload_bytes == 0 || point.message_count == 0) {
      throw std::runtime_error("payload and message count must be positive");
    }
    points.push_back(std::move(point));
  }
  return points;
}

std::vector<int> parse_devices(const std::string& text) {
  std::vector<int> devices;
  for (const auto& item : split(text, ',')) {
    if (!item.empty()) devices.push_back(parse_int(item, "device list"));
  }
  return devices;
}

std::vector<std::pair<int, int>> flows_for(const Point& point) {
  auto sources = parse_devices(point.sources);
  auto destinations = parse_devices(point.destinations);
  if (sources.empty()) sources.push_back(point.source);
  if (destinations.empty()) destinations.push_back(point.destination);
  std::vector<std::pair<int, int>> flows;
  if (point.pattern == "full_mesh") {
    for (const int source : sources) {
      for (const int destination : destinations) {
        if (source != destination) flows.emplace_back(source, destination);
      }
    }
  } else if (sources.size() == destinations.size() && sources.size() > 1) {
    for (std::size_t i = 0; i < sources.size(); ++i) {
      if (sources[i] != destinations[i]) flows.emplace_back(sources[i], destinations[i]);
    }
  } else {
    for (const int source : sources) {
      for (const int destination : destinations) {
        if (source != destination) flows.emplace_back(source, destination);
      }
    }
  }
  if (flows.empty()) flows.emplace_back(point.source, point.destination);
  const bool same_pair_competition =
      point.pattern == "small_behind_large" ||
      point.pattern == "large_behind_small" ||
      point.pattern == "separate_streams" ||
      point.pattern == "alternating_sizes" ||
      point.pattern == "bimodal_mix" ||
      point.pattern == "same_pair_bulk" ||
      point.pattern == "write_write" ||
      point.pattern == "read_read" ||
      point.pattern == "same_direction_read_write" ||
      point.pattern == "distinct_regions" ||
      point.pattern == "shared_cache_line" ||
      point.pattern == "post_burst_drain";
  if (same_pair_competition && flows.size() == 1) flows.push_back(flows.front());
  if (point.pattern == "other_peer_bulk" && flows.size() == 1) {
    flows.emplace_back(flows.front().first, (flows.front().second + 1) % 4);
  }
  if (point.pattern == "remote_incast" && flows.size() == 1) {
    flows.emplace_back((flows.front().first + 2) % 4, flows.front().second);
  }
  const bool bidirectional =
      point.pattern.find("bidirectional") != std::string::npos ||
      point.pattern == "opposite_directions" ||
      point.pattern == "opposite_direction_read_write";
  if (bidirectional) {
    const auto forward = flows;
    for (const auto& flow : forward) {
      const auto reverse = std::make_pair(flow.second, flow.first);
      if (std::find(flows.begin(), flows.end(), reverse) == flows.end()) {
        flows.push_back(reverse);
      }
    }
  }
  return flows;
}

#ifdef SIMLLM_NVLINK_MOCK
CapturedObservation run_mock(const Point& point) {
  const auto flows = flows_for(point);
  const auto flow_count = static_cast<std::uint64_t>(flows.size());
  const auto logical_bytes = point.payload_bytes * point.message_count * flow_count;
  const auto synthetic_kib = (logical_bytes + kCounterQuantumBytes - 1) /
                             kCounterQuantumBytes;
  CapturedObservation captured;
  captured.summary.elapsed_us = static_cast<double>(logical_bytes) / 80.0e9 * 1.0e6 + 1.0;
  captured.summary.completion_us = captured.summary.elapsed_us;
  captured.summary.drain_us = captured.summary.elapsed_us;
  captured.summary.logical_bytes = logical_bytes;
  captured.summary.observed_data_bytes = synthetic_kib * kCounterQuantumBytes;
  captured.summary.observed_raw_bytes = synthetic_kib * kCounterQuantumBytes;
  const auto checksum_text = point.point_id + ":mock-destination-bytes";
  captured.summary.expected_sha256 =
      traf70::sha256_hex(checksum_text.data(), checksum_text.size());
  captured.summary.observed_sha256 = captured.summary.expected_sha256;
  const auto sequence_text = point.point_id + ":mock-order";
  captured.summary.expected_sequence_sha256 =
      traf70::sha256_hex(sequence_text.data(), sequence_text.size());
  captured.summary.observed_sequence_sha256 =
      captured.summary.expected_sequence_sha256;
  captured.summary.terminal_extents = point.message_count * flow_count;
  if (point.producer == "copy_engine_reference") {
    captured.summary.copy_engine_host_enqueue_count = flow_count * point.stream_count;
    captured.summary.copy_engine_batch_mode = "single_contiguous_batch_per_flow_stream";
  }
  captured.summary.protocol = point.producer == "nccl_send_receive_validation"
      ? "protocol_validation_only"
      : "mock_synthetic_no_measurement_claim";
  captured.summary.flow_completion_us.assign(flows.size(), captured.summary.elapsed_us);
  for (std::size_t index = 0; index < flows.size(); ++index) {
    LinkDelta delta;
    delta.gpu = flows[index].first;
    delta.link = static_cast<int>(index % 4);
    delta.remote_gpu = flows[index].second;
    delta.data_tx_kib_after = synthetic_kib / std::max<std::size_t>(1, flows.size());
    delta.raw_tx_kib_after = delta.data_tx_kib_after;
    captured.links.push_back(delta);
  }
  return captured;
}
#endif

#ifndef SIMLLM_NVLINK_MOCK

#define CUDA_CHECK(expression)                                                     \
  do {                                                                             \
    const cudaError_t error_ = (expression);                                        \
    if (error_ != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("CUDA failure: ") +                     \
                               cudaGetErrorString(error_));                         \
    }                                                                               \
  } while (0)

#define NVML_CHECK(expression)                                                     \
  do {                                                                             \
    const nvmlReturn_t error_ = (expression);                                      \
    if (error_ != NVML_SUCCESS) {                                                  \
      throw std::runtime_error(std::string("NVML failure: ") +                   \
                               nvmlErrorString(error_));                           \
    }                                                                               \
  } while (0)

#ifdef SIMLLM_NVLINK_ENABLE_NCCL
#define NCCL_CHECK(expression)                                                      \
  do {                                                                             \
    const ncclResult_t error_ = (expression);                                       \
    if (error_ != ncclSuccess) {                                                    \
      throw std::runtime_error(std::string("NCCL failure: ") +                     \
                               ncclGetErrorString(error_));                         \
    }                                                                               \
  } while (0)
#endif

__global__ void peer_copy_kernel(unsigned char* destination,
                                 const unsigned char* source,
                                 unsigned long long* order,
                                 std::uint64_t total_bytes,
                                 std::uint64_t payload_bytes,
                                 int access_width,
                                 int stride,
                                 unsigned int lane_mask,
                                 int stream_index,
                                 int stream_count,
                                 int burst_messages,
                                 int gap_ns,
                                 int offered_rate_percent,
                                 int clock_rate_khz,
                                 std::uint64_t pattern_seed,
                                 int operation) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if ((lane_mask & (1U << lane)) == 0) return;
  const int lanes_per_warp = __popc(lane_mask);
  const int lane_rank = __popc(lane_mask & ((1U << lane) - 1U));
  const std::uint64_t message_count = total_bytes / payload_bytes;
  const std::uint64_t active_warp =
      static_cast<std::uint64_t>(blockIdx.x * (blockDim.x / 32) + warp);
  const std::uint64_t active_warps =
      static_cast<std::uint64_t>(gridDim.x * (blockDim.x / 32));
  const std::uint64_t pattern_offset =
      message_count == 0 ? 0 : pattern_seed % message_count;
  const std::uint64_t pacing_start = clock64();
  for (std::uint64_t ordinal = active_warp; ordinal < message_count;
       ordinal += active_warps) {
    const std::uint64_t message = (ordinal + pattern_offset) % message_count;
    if (static_cast<int>(message % static_cast<std::uint64_t>(stream_count)) !=
        stream_index) {
      continue;
    }
    const std::uint64_t burst = message / static_cast<std::uint64_t>(burst_messages);
    const double target_ns =
        static_cast<double>(message * payload_bytes) /
            static_cast<double>(offered_rate_percent) +
        static_cast<double>(burst) * static_cast<double>(gap_ns);
    const std::uint64_t target_cycles = static_cast<std::uint64_t>(
        target_ns * static_cast<double>(clock_rate_khz) / 1.0e6);
    while (clock64() - pacing_start < target_cycles) {
    }
    for (std::uint64_t message_offset =
             static_cast<std::uint64_t>(lane_rank * access_width);
         message_offset < payload_bytes;
         message_offset +=
             static_cast<std::uint64_t>(lanes_per_warp * access_width)) {
      for (int byte = 0; byte < access_width; ++byte) {
        const std::uint64_t offset =
            message_offset + static_cast<std::uint64_t>(byte);
        if (offset >= payload_bytes) break;
        const std::uint64_t logical = message * payload_bytes + offset;
        const std::uint64_t physical = logical * static_cast<std::uint64_t>(stride);
        if (operation == 2) {
          if ((physical & 7U) == 0 && physical + 8 <= total_bytes * stride) {
            atomicAdd(reinterpret_cast<unsigned long long*>(destination + physical), 1ULL);
          }
        } else {
          destination[physical] = source[physical];
        }
      }
    }
    __syncwarp(lane_mask);
    if (lane_rank == 0) order[message] = message + 1;
  }
}

__global__ void mark_order_kernel(unsigned long long* order,
                                  std::uint64_t message_count,
                                  int stream_index,
                                  int stream_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < message_count &&
      static_cast<int>(index % static_cast<std::uint64_t>(stream_count)) == stream_index) {
    order[index] = index + 1;
  }
}

__global__ void busy_wait_kernel(std::uint64_t cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::uint64_t start = clock64();
  while (clock64() - start < cycles) {
  }
}

unsigned int lane_mask_for(const Point& point) {
  const int count = std::min(32, std::max(1, point.active_lanes));
  if (point.lane_mask == "contiguous") {
    return count == 32 ? 0xffffffffU : (1U << count) - 1U;
  }
  unsigned int mask = 0;
  if (point.lane_mask == "alternating") {
    for (int lane = 0; lane < count; ++lane) {
      const int position = lane < 16 ? lane * 2 : (lane - 16) * 2 + 1;
      mask |= 1U << position;
    }
  } else if (point.lane_mask == "split") {
    const int lower = (count + 1) / 2;
    const int upper = count - lower;
    for (int lane = 0; lane < lower; ++lane) mask |= 1U << lane;
    for (int lane = 0; lane < upper; ++lane) mask |= 1U << (32 - upper + lane);
  } else if (point.lane_mask == "seeded") {
    std::uint64_t seed = stable_checksum(point.point_id);
    while (__builtin_popcount(mask) < count) {
      seed = seed * 6364136223846793005ULL + 1;
      mask |= 1U << ((seed >> 32) & 31U);
    }
  } else {
    throw std::runtime_error("unknown lane-mask shape: " + point.lane_mask);
  }
  return mask;
}

std::uint64_t pacing_nanoseconds(const Point& point, std::uint64_t logical_bytes) {
  const auto bursts = (point.message_count + point.burst_messages - 1) /
                      point.burst_messages;
  const std::uint64_t gap = bursts > 0 ? (bursts - 1) * point.gap_ns : 0;
  const double target_seconds =
      static_cast<double>(logical_bytes) /
      (100.0e9 * static_cast<double>(point.offered_rate_percent) / 100.0);
  const double full_seconds = static_cast<double>(logical_bytes) / 100.0e9;
  const auto offered = target_seconds > full_seconds
      ? static_cast<std::uint64_t>((target_seconds - full_seconds) * 1.0e9)
      : 0;
  return gap + offered;
}

struct HardwareContext {
  static constexpr std::size_t kAllocationBytes = 320ULL << 20;
  static constexpr std::size_t kOrderEntries = 16ULL << 20;
  unsigned char* buffers[4] = {nullptr, nullptr, nullptr, nullptr};
  unsigned long long* orders[4] = {nullptr, nullptr, nullptr, nullptr};
  cudaStream_t streams[4][4][kMaximumStreams]{};
  std::array<nvmlDevice_t, 4> nvml_devices{};
  std::array<std::string, 4> pci_bus_ids{};
  std::array<int, 4> clock_rate_khz{};
#ifdef SIMLLM_NVLINK_ENABLE_NCCL
  std::array<ncclComm_t, 4> communicators{};
#endif

  HardwareContext() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count != 4) throw std::runtime_error("hardware mode requires exactly four GPUs");
    NVML_CHECK(nvmlInit_v2());
    for (int device = 0; device < 4; ++device) {
      CUDA_CHECK(cudaSetDevice(device));
      CUDA_CHECK(cudaMalloc(&buffers[device], kAllocationBytes));
      CUDA_CHECK(cudaMalloc(&orders[device], kOrderEntries * sizeof(unsigned long long)));
      cudaDeviceProp properties{};
      CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      clock_rate_khz[device] = properties.clockRate;
      char cuda_bus_id[32]{};
      CUDA_CHECK(cudaDeviceGetPCIBusId(cuda_bus_id, sizeof(cuda_bus_id), device));
      NVML_CHECK(nvmlDeviceGetHandleByPciBusId_v2(cuda_bus_id,
                                                 &nvml_devices[device]));
      nvmlPciInfo_t pci{};
      NVML_CHECK(nvmlDeviceGetPciInfo_v3(nvml_devices[device], &pci));
      pci_bus_ids[device] = pci.busId;
      for (int peer = 0; peer < 4; ++peer) {
        for (std::size_t stream = 0; stream < kMaximumStreams; ++stream) {
          CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][peer][stream],
                                               cudaStreamNonBlocking));
        }
        if (peer == device) continue;
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer));
        if (!can_access) throw std::runtime_error("CUDA peer access is unavailable");
        const auto error = cudaDeviceEnablePeerAccess(peer, 0);
        if (error != cudaSuccess && error != cudaErrorPeerAccessAlreadyEnabled) {
          CUDA_CHECK(error);
        }
        cudaGetLastError();
      }
    }
#ifdef SIMLLM_NVLINK_ENABLE_NCCL
    int devices[4] = {0, 1, 2, 3};
    NCCL_CHECK(ncclCommInitAll(communicators.data(), 4, devices));
#endif
  }

  ~HardwareContext() {
#ifdef SIMLLM_NVLINK_ENABLE_NCCL
    for (auto& communicator : communicators) ncclCommDestroy(communicator);
#endif
    for (int device = 0; device < 4; ++device) {
      cudaSetDevice(device);
      for (int peer = 0; peer < 4; ++peer) {
        for (std::size_t stream = 0; stream < kMaximumStreams; ++stream) {
          cudaStreamDestroy(streams[device][peer][stream]);
        }
      }
      cudaFree(orders[device]);
      cudaFree(buffers[device]);
    }
    nvmlShutdown();
  }
};

struct LinkSnapshot {
  int gpu = 0;
  int link = 0;
  int remote_gpu = -1;
  std::array<std::uint64_t, 4> throughput{};
  std::array<std::uint64_t, 5> errors{};
  std::array<int, 9> statuses{};
};

std::uint64_t field_unsigned(const nvmlFieldValue_t& field) {
  if (field.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG) return field.value.ullVal;
  if (field.valueType == NVML_VALUE_TYPE_UNSIGNED_INT) return field.value.uiVal;
  if (field.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG) return field.value.ulVal;
  throw std::runtime_error("NVML throughput field has a non-unsigned value type");
}

std::vector<LinkSnapshot> capture_links(const HardwareContext& context) {
  std::vector<LinkSnapshot> snapshots;
  const std::array<unsigned int, 4> fields = {
      NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
      NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX,
      NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX,
      NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX};
  for (int gpu = 0; gpu < 4; ++gpu) {
    for (unsigned int link = 0; link < NVML_NVLINK_MAX_LINKS; ++link) {
      nvmlEnableState_t active = NVML_FEATURE_DISABLED;
      const auto state_status =
          nvmlDeviceGetNvLinkState(context.nvml_devices[gpu], link, &active);
      if (state_status == NVML_ERROR_INVALID_ARGUMENT) break;
      if (state_status != NVML_SUCCESS || active != NVML_FEATURE_ENABLED) continue;
      LinkSnapshot snapshot;
      snapshot.gpu = gpu;
      snapshot.link = static_cast<int>(link);
      nvmlPciInfo_t remote{};
      const auto remote_status = nvmlDeviceGetNvLinkRemotePciInfo_v2(
          context.nvml_devices[gpu], link, &remote);
      if (remote_status == NVML_SUCCESS) {
        for (int candidate = 0; candidate < 4; ++candidate) {
          if (context.pci_bus_ids[candidate] == remote.busId) snapshot.remote_gpu = candidate;
        }
      }
      std::array<nvmlFieldValue_t, 4> values{};
      for (std::size_t index = 0; index < values.size(); ++index) {
        values[index].fieldId = fields[index];
        values[index].scopeId = link;
      }
      const auto field_status = nvmlDeviceGetFieldValues(
          context.nvml_devices[gpu], static_cast<int>(values.size()), values.data());
      for (std::size_t index = 0; index < values.size(); ++index) {
        snapshot.statuses[index] = field_status == NVML_SUCCESS
            ? static_cast<int>(values[index].nvmlReturn)
            : static_cast<int>(field_status);
        if (snapshot.statuses[index] == NVML_SUCCESS) {
          snapshot.throughput[index] = field_unsigned(values[index]);
        }
      }
      for (int counter = 0; counter < 5; ++counter) {
        unsigned long long value = 0;
        const auto error_status = nvmlDeviceGetNvLinkErrorCounter(
            context.nvml_devices[gpu], link,
            static_cast<nvmlNvLinkErrorCounter_t>(counter), &value);
        snapshot.statuses[4 + counter] = static_cast<int>(error_status);
        if (error_status == NVML_SUCCESS) snapshot.errors[counter] = value;
      }
      snapshots.push_back(snapshot);
    }
  }
  return snapshots;
}

std::vector<DeviceTelemetry> capture_telemetry(const HardwareContext& context) {
  std::vector<DeviceTelemetry> telemetry;
  for (int gpu = 0; gpu < 4; ++gpu) {
    DeviceTelemetry value;
    value.gpu = gpu;
    unsigned long long reasons = 0;
    value.statuses[0] = nvmlDeviceGetCurrentClocksThrottleReasons(
        context.nvml_devices[gpu], &reasons);
    value.clock_event_reasons = reasons;
    value.statuses[1] = nvmlDeviceGetClockInfo(
        context.nvml_devices[gpu], NVML_CLOCK_SM, &value.sm_clock_mhz);
    value.statuses[2] = nvmlDeviceGetClockInfo(
        context.nvml_devices[gpu], NVML_CLOCK_MEM, &value.memory_clock_mhz);
    value.statuses[3] = nvmlDeviceGetPowerUsage(context.nvml_devices[gpu], &value.power_mw);
    value.statuses[4] = nvmlDeviceGetTemperature(
        context.nvml_devices[gpu], NVML_TEMPERATURE_GPU, &value.temperature_c);
    telemetry.push_back(value);
  }
  return telemetry;
}

std::vector<LinkDelta> link_deltas(const std::vector<LinkSnapshot>& before,
                                   const std::vector<LinkSnapshot>& after) {
  if (before.size() != after.size()) throw std::runtime_error("NVLink set changed within row");
  std::vector<LinkDelta> deltas;
  for (std::size_t index = 0; index < before.size(); ++index) {
    const auto& first = before[index];
    const auto& last = after[index];
    if (first.gpu != last.gpu || first.link != last.link ||
        first.remote_gpu != last.remote_gpu) {
      throw std::runtime_error("NVLink mapping changed within row");
    }
    LinkDelta delta;
    delta.gpu = first.gpu;
    delta.link = first.link;
    delta.remote_gpu = first.remote_gpu;
    delta.data_tx_kib_before = first.throughput[0];
    delta.data_tx_kib_after = last.throughput[0];
    delta.data_rx_kib_before = first.throughput[1];
    delta.data_rx_kib_after = last.throughput[1];
    delta.raw_tx_kib_before = first.throughput[2];
    delta.raw_tx_kib_after = last.throughput[2];
    delta.raw_rx_kib_before = first.throughput[3];
    delta.raw_rx_kib_after = last.throughput[3];
    delta.errors_before = first.errors;
    delta.errors_after = last.errors;
    for (std::size_t status = 0; status < delta.statuses.size(); ++status) {
      delta.statuses[status] = first.statuses[status] != NVML_SUCCESS
          ? first.statuses[status]
          : last.statuses[status];
    }
    deltas.push_back(delta);
  }
  return deltas;
}

CapturedObservation run_hardware(const Point& point, HardwareContext& context) {
  if (point.stream_count > static_cast<int>(kMaximumStreams)) {
    throw std::runtime_error("stream count exceeds compiled maximum");
  }
  const auto flows = flows_for(point);
  const std::uint64_t bytes_per_flow = point.payload_bytes * point.message_count;
  const std::uint64_t span_per_flow = bytes_per_flow * point.stride + 512;
  if (span_per_flow * flows.size() > HardwareContext::kAllocationBytes) {
    throw std::runtime_error("point exceeds fixed data allocation");
  }
  if (point.message_count * flows.size() > HardwareContext::kOrderEntries) {
    throw std::runtime_error("point exceeds fixed order-ledger allocation");
  }
  std::vector<std::vector<unsigned char>> expected(flows.size());
  for (std::size_t flow_index = 0; flow_index < flows.size(); ++flow_index) {
    expected[flow_index].assign(span_per_flow, 0);
    for (std::uint64_t logical = 0; logical < bytes_per_flow; ++logical) {
      expected[flow_index][point.source_alignment + logical * point.stride] =
          expected_byte(logical / point.payload_bytes,
                        logical % point.payload_bytes, flow_index);
    }
    const int source_device = point.producer == "dependent_sm_peer_read"
        ? flows[flow_index].second
        : flows[flow_index].first;
    const int destination_device = point.producer == "dependent_sm_peer_read"
        ? flows[flow_index].first
        : flows[flow_index].second;
    const std::uint64_t region = flow_index * span_per_flow;
    CUDA_CHECK(cudaSetDevice(source_device));
    CUDA_CHECK(cudaMemcpy(context.buffers[source_device] + region,
                          expected[flow_index].data(), span_per_flow,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaSetDevice(destination_device));
    CUDA_CHECK(cudaMemset(context.buffers[destination_device] + region, 0,
                          span_per_flow));
    CUDA_CHECK(cudaMemset(context.orders[destination_device] +
                              flow_index * point.message_count,
                          0, point.message_count * sizeof(unsigned long long)));
  }
  for (int device = 0; device < 4; ++device) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  auto telemetry_before = capture_telemetry(context);
  const auto counters_before = capture_links(context);
  const auto batch_started = std::chrono::steady_clock::now();
  std::vector<cudaEvent_t> starts;
  std::vector<cudaEvent_t> stops;
  for (std::size_t flow_index = 0; flow_index < flows.size(); ++flow_index) {
    const int logical_source = flows[flow_index].first;
    const int logical_destination = flows[flow_index].second;
    const int issuer = point.producer == "dependent_sm_peer_read"
        ? logical_source
        : logical_source;
    const int data_source = point.producer == "dependent_sm_peer_read"
        ? logical_destination
        : logical_source;
    const int data_destination = point.producer == "dependent_sm_peer_read"
        ? logical_source
        : logical_destination;
    const std::uint64_t region = flow_index * span_per_flow;
    CUDA_CHECK(cudaSetDevice(issuer));
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    starts.push_back(start);
    stops.push_back(stop);
    CUDA_CHECK(cudaEventRecord(start, context.streams[issuer][logical_destination][0]));
    for (int stream_index = 0; stream_index < point.stream_count; ++stream_index) {
      auto stream = context.streams[issuer][logical_destination][stream_index];
      if (point.producer == "nccl_send_receive_validation") {
#ifdef SIMLLM_NVLINK_ENABLE_NCCL
        NCCL_CHECK(ncclGroupStart());
        CUDA_CHECK(cudaSetDevice(data_source));
        NCCL_CHECK(ncclSend(context.buffers[data_source] + region +
                                point.source_alignment,
                            bytes_per_flow, ncclUint8, data_destination,
                            context.communicators[data_source], stream));
        CUDA_CHECK(cudaSetDevice(data_destination));
        auto receive_stream =
            context.streams[data_destination][data_source][stream_index];
        NCCL_CHECK(ncclRecv(context.buffers[data_destination] + region +
                                point.destination_alignment,
                            bytes_per_flow, ncclUint8, data_source,
                            context.communicators[data_destination], receive_stream));
        NCCL_CHECK(ncclGroupEnd());
        const auto blocks = static_cast<unsigned int>((point.message_count + 255) / 256);
        mark_order_kernel<<<blocks, 256, 0, receive_stream>>>(
            context.orders[data_destination] + flow_index * point.message_count,
            point.message_count, stream_index, point.stream_count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(receive_stream));
#else
        throw std::runtime_error("NCCL validation requires SIMLLM_NVLINK_ENABLE_NCCL");
#endif
      } else if (point.producer == "copy_engine_reference") {
        const std::uint64_t first_message =
            point.message_count * stream_index / point.stream_count;
        const std::uint64_t last_message =
            point.message_count * (stream_index + 1) / point.stream_count;
        const std::uint64_t first_byte = first_message * point.payload_bytes;
        const std::uint64_t byte_count =
            (last_message - first_message) * point.payload_bytes;
        if (point.stride != 1) {
          throw std::runtime_error("copy-engine row requires unit stride");
        }
        CUDA_CHECK(cudaMemcpyPeerAsync(
            context.buffers[data_destination] + region + point.destination_alignment + first_byte,
            data_destination,
            context.buffers[data_source] + region + point.source_alignment + first_byte,
            data_source, byte_count, stream));
        const auto blocks = static_cast<unsigned int>((point.message_count + 255) / 256);
        mark_order_kernel<<<blocks, 256, 0, stream>>>(
            context.orders[data_destination] + flow_index * point.message_count,
            point.message_count, stream_index, point.stream_count);
        CUDA_CHECK(cudaGetLastError());
      } else {
        const int operation = point.producer == "persistent_sm_peer_atomic" ? 2 : 0;
        const int blocks = std::min(256, std::max(1, point.outstanding));
        peer_copy_kernel<<<blocks, 256, 0, stream>>>(
            context.buffers[data_destination] + region + point.destination_alignment,
            context.buffers[data_source] + region + point.source_alignment,
            context.orders[data_destination] + flow_index * point.message_count,
            bytes_per_flow, point.payload_bytes, point.access_width, point.stride,
            lane_mask_for(point), stream_index, point.stream_count,
            point.burst_messages, point.gap_ns, point.offered_rate_percent,
            context.clock_rate_khz[issuer], stable_checksum(point.pattern), operation);
        CUDA_CHECK(cudaGetLastError());
      }
      const auto delay_ns =
          point.producer == "copy_engine_reference" ||
                  point.producer == "nccl_send_receive_validation"
              ? pacing_nanoseconds(point, bytes_per_flow) /
                    static_cast<std::uint64_t>(point.stream_count)
              : 0;
      if (delay_ns > 0) {
        const auto cycles = delay_ns * context.clock_rate_khz[issuer] / 1000000ULL;
        busy_wait_kernel<<<1, 1, 0, stream>>>(cycles);
        CUDA_CHECK(cudaGetLastError());
      }
    }
    CUDA_CHECK(cudaSetDevice(issuer));
    for (int stream_index = 1; stream_index < point.stream_count; ++stream_index) {
      CUDA_CHECK(cudaStreamSynchronize(
          context.streams[issuer][logical_destination][stream_index]));
    }
    CUDA_CHECK(cudaEventRecord(stop, context.streams[issuer][logical_destination][0]));
  }
  double elapsed_us = 0.0;
  std::vector<double> flow_completion_us;
  for (std::size_t index = 0; index < starts.size(); ++index) {
    CUDA_CHECK(cudaSetDevice(flows[index].first));
    CUDA_CHECK(cudaEventSynchronize(stops[index]));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, starts[index], stops[index]));
    const double flow_elapsed_us = static_cast<double>(elapsed_ms) * 1000.0;
    flow_completion_us.push_back(flow_elapsed_us);
    elapsed_us = std::max(elapsed_us, flow_elapsed_us);
    CUDA_CHECK(cudaEventDestroy(starts[index]));
    CUDA_CHECK(cudaEventDestroy(stops[index]));
  }
  for (int device = 0; device < 4; ++device) {
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  const auto batch_completed = std::chrono::steady_clock::now();
  const auto counters_after = capture_links(context);
  auto telemetry_after = capture_telemetry(context);
  CapturedObservation captured;
  captured.links = link_deltas(counters_before, counters_after);
  captured.telemetry_before = std::move(telemetry_before);
  captured.telemetry_after = std::move(telemetry_after);
  const double batch_elapsed_us =
      std::chrono::duration<double, std::micro>(batch_completed - batch_started).count();
  captured.summary.elapsed_us = std::max(elapsed_us, batch_elapsed_us);
  captured.summary.completion_us = captured.summary.elapsed_us;
  captured.summary.drain_us = captured.summary.elapsed_us;
  captured.summary.logical_bytes = bytes_per_flow * flows.size();
  captured.summary.flow_completion_us = std::move(flow_completion_us);
  traf70::Sha256 expected_digest;
  traf70::Sha256 observed_digest;
  traf70::Sha256 expected_order_digest;
  traf70::Sha256 observed_order_digest;
  for (std::size_t flow_index = 0; flow_index < flows.size(); ++flow_index) {
    const int destination_device = point.producer == "dependent_sm_peer_read"
        ? flows[flow_index].first
        : flows[flow_index].second;
    const std::uint64_t region = flow_index * span_per_flow;
    std::vector<unsigned char> observed(span_per_flow);
    std::vector<unsigned long long> order(point.message_count);
    CUDA_CHECK(cudaSetDevice(destination_device));
    CUDA_CHECK(cudaMemcpy(observed.data(), context.buffers[destination_device] + region,
                          span_per_flow, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(order.data(),
                          context.orders[destination_device] +
                              flow_index * point.message_count,
                          point.message_count * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    for (std::uint64_t logical = 0; logical < bytes_per_flow; ++logical) {
      const auto expected_value = point.producer == "persistent_sm_peer_atomic"
          ? static_cast<unsigned char>(logical % 8 == 0 ? 1 : 0)
          : expected_byte(logical / point.payload_bytes,
                          logical % point.payload_bytes, flow_index);
      expected_digest.update(&expected_value, 1);
      const auto observed_value =
          observed[point.destination_alignment + logical * point.stride];
      observed_digest.update(&observed_value, 1);
    }
    for (std::uint64_t message = 0; message < point.message_count; ++message) {
      const std::uint64_t expected_value = message + 1;
      expected_order_digest.update(&expected_value, sizeof(expected_value));
      observed_order_digest.update(&order[message], sizeof(order[message]));
      if (order[message] == 0) ++captured.summary.missing;
      if (order[message] != 0 && order[message] != expected_value) {
        ++captured.summary.out_of_order;
      }
    }
  }
  captured.summary.expected_sha256 = expected_digest.finish_hex();
  captured.summary.observed_sha256 = observed_digest.finish_hex();
  captured.summary.expected_sequence_sha256 = expected_order_digest.finish_hex();
  captured.summary.observed_sequence_sha256 = observed_order_digest.finish_hex();
  captured.summary.terminal_extents =
      point.message_count * flows.size() - captured.summary.missing;
  const std::uint64_t fatal_throttle_mask =
      nvmlClocksThrottleReasonHwSlowdown |
      nvmlClocksThrottleReasonSwThermalSlowdown |
      nvmlClocksThrottleReasonHwThermalSlowdown |
      nvmlClocksThrottleReasonHwPowerBrakeSlowdown;
  for (const auto& telemetry : captured.telemetry_before) {
    captured.summary.throttle_reason_mask |= telemetry.clock_event_reasons;
  }
  for (const auto& telemetry : captured.telemetry_after) {
    captured.summary.throttle_reason_mask |= telemetry.clock_event_reasons;
  }
  const std::uint64_t known_throttle_mask =
      static_cast<std::uint64_t>(nvmlClocksThrottleReasonAll);
  const std::uint64_t unknown_throttle_mask =
      captured.summary.throttle_reason_mask & ~known_throttle_mask;
  captured.summary.throttle_verdict =
      (captured.summary.throttle_reason_mask & fatal_throttle_mask) == 0 &&
              unknown_throttle_mask == 0
      ? "CLEAR"
      : "FATAL_CLOCK_EVENT";
  if (point.producer == "copy_engine_reference") {
    captured.summary.copy_engine_host_enqueue_count =
        flows.size() * static_cast<std::uint64_t>(point.stream_count);
    captured.summary.copy_engine_batch_mode = "single_contiguous_batch_per_flow_stream";
  }
  captured.summary.protocol = point.producer == "nccl_send_receive_validation"
      ? "protocol_validation_only"
      : "transport_observation";
  std::uint64_t data_tx_kib = 0;
  std::uint64_t data_rx_kib = 0;
  std::uint64_t raw_tx_kib = 0;
  std::uint64_t raw_rx_kib = 0;
  for (const auto& link : captured.links) {
    data_tx_kib += positive_delta(link.data_tx_kib_before, link.data_tx_kib_after);
    data_rx_kib += positive_delta(link.data_rx_kib_before, link.data_rx_kib_after);
    raw_tx_kib += positive_delta(link.raw_tx_kib_before, link.raw_tx_kib_after);
    raw_rx_kib += positive_delta(link.raw_rx_kib_before, link.raw_rx_kib_after);
  }
  captured.summary.observed_data_bytes =
      std::max(data_tx_kib, data_rx_kib) * kCounterQuantumBytes;
  captured.summary.observed_raw_bytes =
      std::max(raw_tx_kib, raw_rx_kib) * kCounterQuantumBytes;
  return captured;
}

#endif

void write_telemetry(std::ofstream& output,
                     const std::vector<DeviceTelemetry>& telemetry) {
  output << '[';
  for (std::size_t index = 0; index < telemetry.size(); ++index) {
    if (index) output << ',';
    const auto& value = telemetry[index];
    output << "{\"gpu\":" << value.gpu
           << ",\"clock_event_reasons\":" << value.clock_event_reasons
           << ",\"sm_clock_mhz\":" << value.sm_clock_mhz
           << ",\"memory_clock_mhz\":" << value.memory_clock_mhz
           << ",\"power_mw\":" << value.power_mw
           << ",\"temperature_c\":" << value.temperature_c
           << ",\"statuses\":[";
    for (std::size_t status = 0; status < value.statuses.size(); ++status) {
      if (status) output << ',';
      output << value.statuses[status];
    }
    output << "]}";
  }
  output << ']';
}

void write_links(std::ofstream& output, const std::vector<LinkDelta>& links) {
  output << '[';
  for (std::size_t index = 0; index < links.size(); ++index) {
    if (index) output << ',';
    const auto& link = links[index];
    output << "{\"gpu\":" << link.gpu
           << ",\"link\":" << link.link
           << ",\"remote_gpu\":" << link.remote_gpu
           << ",\"data_tx_kib_before\":" << link.data_tx_kib_before
           << ",\"data_tx_kib_after\":" << link.data_tx_kib_after
           << ",\"data_tx_kib_delta\":"
           << positive_delta(link.data_tx_kib_before, link.data_tx_kib_after)
           << ",\"data_rx_kib_before\":" << link.data_rx_kib_before
           << ",\"data_rx_kib_after\":" << link.data_rx_kib_after
           << ",\"data_rx_kib_delta\":"
           << positive_delta(link.data_rx_kib_before, link.data_rx_kib_after)
           << ",\"raw_tx_kib_before\":" << link.raw_tx_kib_before
           << ",\"raw_tx_kib_after\":" << link.raw_tx_kib_after
           << ",\"raw_tx_kib_delta\":"
           << positive_delta(link.raw_tx_kib_before, link.raw_tx_kib_after)
           << ",\"raw_rx_kib_before\":" << link.raw_rx_kib_before
           << ",\"raw_rx_kib_after\":" << link.raw_rx_kib_after
           << ",\"raw_rx_kib_delta\":"
           << positive_delta(link.raw_rx_kib_before, link.raw_rx_kib_after)
           << ",\"errors\":{";
    static constexpr std::array<const char*, 5> names = {
        "replay", "recovery", "crc_flit", "crc_data", "ecc_data"};
    for (std::size_t error = 0; error < names.size(); ++error) {
      if (error) output << ',';
      output << '\"' << names[error] << "\":{\"before\":"
             << link.errors_before[error] << ",\"after\":"
             << link.errors_after[error] << ",\"delta\":"
             << positive_delta(link.errors_before[error], link.errors_after[error])
             << '}';
    }
    output << "},\"statuses\":[";
    for (std::size_t status = 0; status < link.statuses.size(); ++status) {
      if (status) output << ',';
      output << link.statuses[status];
    }
    output << "]}";
  }
  output << ']';
}

void write_error_deltas(std::ofstream& output,
                        const std::vector<LinkDelta>& links) {
  output << '[';
  for (std::size_t index = 0; index < links.size(); ++index) {
    if (index) output << ',';
    const auto& link = links[index];
    output << "{\"gpu\":" << link.gpu << ",\"link\":" << link.link
           << ",\"remote_gpu\":" << link.remote_gpu;
    static constexpr std::array<const char*, 5> names = {
        "replay", "recovery", "crc_flit", "crc_data", "ecc_data"};
    for (std::size_t error = 0; error < names.size(); ++error) {
      output << ",\"" << names[error] << "_delta\":"
             << positive_delta(link.errors_before[error], link.errors_after[error]);
    }
    output << '}';
  }
  output << ']';
}

void write_result(std::ofstream& output, const Point& point,
                  const CapturedObservation& captured,
                  const std::string& mode) {
  const auto& observation = captured.summary;
  const auto flows = flows_for(point);
  const auto flow_count = static_cast<std::uint64_t>(flows_for(point).size());
  const auto logical_bytes = point.payload_bytes * point.message_count * flow_count;
  const auto elapsed_seconds = observation.elapsed_us * 1.0e-6;
  const auto payload_gbps = elapsed_seconds > 0.0
      ? static_cast<double>(logical_bytes) / elapsed_seconds / 1.0e9
      : 0.0;
  output << "{"
         << "\"schema\":\"simllm-a100-nvlink-packet-observation-v2\","
         << "\"mode\":\"" << json_escape(mode) << "\","
         << "\"case_name\":\"" << json_escape(point.case_name) << "\","
         << "\"point_id\":\"" << json_escape(point.point_id) << "\","
         << "\"producer\":\"" << json_escape(point.producer) << "\","
         << "\"protocol_scope\":\"" << json_escape(observation.protocol) << "\","
         << "\"payload_bytes\":" << point.payload_bytes << ','
         << "\"message_count\":" << point.message_count << ','
         << "\"logical_bytes\":" << logical_bytes << ','
         << "\"observed_data_bytes\":" << observation.observed_data_bytes << ','
         << "\"observed_raw_bytes\":" << observation.observed_raw_bytes << ','
         << "\"elapsed_us\":" << std::fixed << std::setprecision(9)
         << observation.elapsed_us << ','
         << "\"completion_us\":" << observation.completion_us << ','
         << "\"drain_us\":" << observation.drain_us << ','
         << "\"drain_time_us\":" << observation.drain_us << ','
         << "\"payload_rate_gbps\":" << std::setprecision(6) << payload_gbps << ','
         << "\"destination_checksum\":{\"algorithm\":\"sha256\","
         << "\"expected_sha256\":\"" << observation.expected_sha256 << "\","
         << "\"observed_sha256\":\"" << observation.observed_sha256 << "\","
         << "\"matches\":"
         << (observation.expected_sha256 == observation.observed_sha256 ? "true" : "false")
         << "},"
         << "\"ordering_ledger\":{\"expected_sequence_sha256\":\""
         << observation.expected_sequence_sha256 << "\","
         << "\"observed_sequence_sha256\":\""
         << observation.observed_sequence_sha256 << "\","
         << "\"expected_extents\":" << point.message_count * flow_count << ','
         << "\"terminal_extents\":" << observation.terminal_extents << ','
         << "\"missing\":" << observation.missing << ','
         << "\"duplicate\":" << observation.duplicate << ','
         << "\"out_of_order\":" << observation.out_of_order << "},"
         << "\"checksum_ok\":"
         << (observation.expected_sha256 == observation.observed_sha256 &&
                     observation.expected_sequence_sha256 ==
                         observation.observed_sequence_sha256
                 ? "true"
                 : "false")
         << ','
         << "\"pattern\":\"" << json_escape(point.pattern) << "\","
         << "\"candidate_blind_fit_membership\":\""
         << (point.case_name.find("_016_") != std::string::npos
                 ? "held_out"
                 : "frozen_training_or_nonfit")
         << "\","
         << "\"offered_inflight_bytes\":"
         << point.payload_bytes * static_cast<std::uint64_t>(point.outstanding) * flow_count
         << ','
         << "\"throttle_verdict\":\"" << observation.throttle_verdict << "\","
         << "\"throttle_reason_mask\":" << observation.throttle_reason_mask << ','
         << "\"copy_engine_batch_mode\":\""
         << json_escape(observation.copy_engine_batch_mode) << "\","
         << "\"copy_engine_host_enqueue_count\":"
         << observation.copy_engine_host_enqueue_count << ','
         << "\"applied_control_sha256\":\"" << control_sha256(point) << "\","
         << "\"applied_controls\":{"
         << "\"payload_bytes\":" << point.payload_bytes << ','
         << "\"message_count\":" << point.message_count << ','
         << "\"source\":" << point.source << ','
         << "\"destination\":" << point.destination << ','
         << "\"sources\":\"" << json_escape(point.sources) << "\","
         << "\"destinations\":\"" << json_escape(point.destinations) << "\","
         << "\"source_alignment\":" << point.source_alignment << ','
         << "\"destination_alignment\":" << point.destination_alignment << ','
         << "\"access_width\":" << point.access_width << ','
         << "\"active_lanes\":" << point.active_lanes << ','
         << "\"lane_mask\":\"" << json_escape(point.lane_mask) << "\","
         << "\"stride\":" << point.stride << ','
         << "\"stream_count\":" << point.stream_count << ','
         << "\"outstanding\":" << point.outstanding << ','
         << "\"burst_messages\":" << point.burst_messages << ','
         << "\"gap_ns\":" << point.gap_ns << ','
         << "\"offered_rate_percent\":" << point.offered_rate_percent << ','
         << "\"pattern\":\"" << json_escape(point.pattern) << "\","
         << "\"effects\":{"
         << "\"payload_bytes\":\"kernel_extent_and_message_boundary\","
         << "\"message_count\":\"kernel_extent_and_terminal_ledger\","
         << "\"source\":\"flow_endpoint_selection\","
         << "\"destination\":\"flow_endpoint_selection\","
         << "\"sources\":\"multi_source_flow_expansion\","
         << "\"destinations\":\"multi_destination_flow_expansion\","
         << "\"source_alignment\":\"source_pointer_offset\","
         << "\"destination_alignment\":\"destination_pointer_offset\","
         << "\"access_width\":\"kernel_work_unit_width\","
         << "\"active_lanes\":\"active_lane_mask_population\","
         << "\"lane_mask\":\"active_lane_positions\","
         << "\"stride\":\"physical_byte_address_stride\","
         << "\"stream_count\":\"work_partition_across_cuda_streams\","
         << "\"outstanding\":\"resident_work_block_count\","
         << "\"burst_messages\":\"in_kernel_burst_schedule\","
         << "\"gap_ns\":\"in_kernel_inter_burst_delay\","
         << "\"offered_rate_percent\":\"in_kernel_byte_issue_schedule\","
         << "\"pattern\":\"pattern_specific_flow_set_and_unit_permutation\"}},"
         << "\"flow_rate_ledger\":[";
  for (std::size_t index = 0; index < flows.size(); ++index) {
    if (index) output << ',';
    output << "{\"source\":" << flows[index].first
           << ",\"destination\":" << flows[index].second
           << ",\"logical_bytes\":" << point.payload_bytes * point.message_count
           << '}';
  }
  output << "],\"observed_counter_deltas\":{"
         << "\"unit\":\"KiB\",\"per_gpu_per_link_per_direction\":";
  write_links(output, captured.links);
  output << "},\"replay_recovery_crc_ecc_deltas\":";
  write_error_deltas(output, captured.links);
  output << ",\"latency_flow_ledger\":[";
  if (!flows.empty()) {
    output << "{\"source\":" << flows.front().first
           << ",\"destination\":" << flows.front().second
           << ",\"logical_bytes\":" << point.payload_bytes * point.message_count
           << ",\"completion_us\":"
           << (observation.flow_completion_us.empty()
                   ? observation.completion_us
                   : observation.flow_completion_us.front())
           << '}';
  }
  output << "],\"bulk_flow_ledger\":[";
  for (std::size_t index = 1; index < flows.size(); ++index) {
    if (index > 1) output << ',';
    output << "{\"source\":" << flows[index].first
           << ",\"destination\":" << flows[index].second
           << ",\"logical_bytes\":" << point.payload_bytes * point.message_count
           << ",\"completion_us\":"
           << (index < observation.flow_completion_us.size()
                   ? observation.flow_completion_us[index]
                   : observation.completion_us)
           << '}';
  }
  output << "],\"telemetry_before\":";
  write_telemetry(output, captured.telemetry_before);
  output << ",\"telemetry_after\":";
  write_telemetry(output, captured.telemetry_after);
  output << ",\"measurement_claim\":"
         << (mode == "hardware" ? "\"unscored\"" : "false") << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string points_path;
    std::string output_path;
    std::string mode;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--points" && i + 1 < argc) points_path = argv[++i];
      else if (argument == "--output" && i + 1 < argc) output_path = argv[++i];
      else if (argument == "--mode" && i + 1 < argc) mode = argv[++i];
      else throw std::runtime_error("unknown or incomplete argument: " + argument);
    }
    if (points_path.empty() || output_path.empty() || mode.empty()) {
      throw std::runtime_error("--points, --output, and --mode are required");
    }
#ifdef SIMLLM_NVLINK_MOCK
    if (mode != "mock") throw std::runtime_error("mock binary accepts --mode mock only");
#else
    if (mode != "hardware") {
      throw std::runtime_error("CUDA binary accepts --mode hardware only");
    }
#endif
    const auto points = read_points(points_path);
    std::ofstream output(output_path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create output: " + output_path);
#ifndef SIMLLM_NVLINK_MOCK
    HardwareContext context;
#endif
    for (const auto& point : points) {
#ifdef SIMLLM_NVLINK_MOCK
      const auto observation = run_mock(point);
#else
      const auto observation = run_hardware(point, context);
#endif
      write_result(output, point, observation, mode);
    }
    output.close();
    if (!output) throw std::runtime_error("failed while writing result output");
    std::cout << "TRAF-70 " << mode << " rows=" << points.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TRAF-70 FATAL: " << error.what() << '\n';
    return 2;
  }
}
