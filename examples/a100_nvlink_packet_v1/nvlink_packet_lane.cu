// TRAF-65 three-producer NVLink harness with a GPU-free mock build.
//
// Mock compile:
//   c++ -x c++ -std=c++17 -DSIMLLM_NVLINK_MOCK nvlink_packet_lane.cu -o lane
// Hardware compile:
//   nvcc -std=c++17 -arch=sm_80 nvlink_packet_lane.cu -o lane
//
// The executable consumes one tab-separated plan and writes one JSON object
// per point. Persistent peer write and dependent peer read use one kernel per
// repeated point, not one launch per message. The copy engine enqueues the
// repeated transfers and synchronizes once after the batch. NCCL send/receive
// rows are conservation checks only and never feed packet-format fields.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SIMLLM_NVLINK_MOCK
#include <cuda_runtime.h>
#ifdef SIMLLM_NVLINK_ENABLE_NCCL
#include <nccl.h>
#endif
#endif

namespace {

constexpr std::uint64_t kCandidatePayloadBytes = 256;
constexpr std::uint64_t kCandidateHeaderBytes = 16;
constexpr double kPairRawBytesPerSecond = 100.0e9;

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
  std::uint64_t checksum = 0;
  bool checksum_ok = true;
  std::string protocol = "candidate_packet";
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

std::uint64_t stable_checksum(const std::string& text) {
  std::uint64_t value = 1469598103934665603ULL;
  for (const unsigned char byte : text) {
    value ^= byte;
    value *= 1099511628211ULL;
  }
  return value;
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

Observation run_mock(const Point& point) {
  const auto packets_per_message =
      (point.payload_bytes + kCandidatePayloadBytes - 1) / kCandidatePayloadBytes;
  const auto flow_count = static_cast<std::uint64_t>(flows_for(point).size());
  const auto logical_bytes = point.payload_bytes * point.message_count * flow_count;
  const auto raw_bytes = logical_bytes +
      packets_per_message * kCandidateHeaderBytes * point.message_count * flow_count;
  double producer_factor = 1.0;
  if (point.producer == "persistent_sm_peer_write") producer_factor = 0.97;
  if (point.producer == "dependent_sm_peer_read") producer_factor = 0.90;
  if (point.producer == "nccl_send_receive_validation") producer_factor = 0.72;
  const auto parallel_flows = static_cast<double>(flow_count);
  Observation observation;
  observation.elapsed_us =
      static_cast<double>(raw_bytes) / (kPairRawBytesPerSecond * producer_factor) * 1.0e6 /
      std::max(1.0, parallel_flows);
  observation.elapsed_us += 0.25;
  observation.checksum = stable_checksum(point.point_id + ":mock");
  observation.protocol = point.producer == "nccl_send_receive_validation"
      ? "protocol_validation_only"
      : "candidate_packet";
  return observation;
}

#ifndef SIMLLM_NVLINK_MOCK

#define CUDA_CHECK(expression)                                                     \
  do {                                                                             \
    const cudaError_t error_ = (expression);                                        \
    if (error_ != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("CUDA failure: ") +                     \
                               cudaGetErrorString(error_));                         \
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

__global__ void persistent_peer_write(unsigned char* destination,
                                      const unsigned char* source,
                                      std::uint64_t payload_bytes,
                                      std::uint64_t total_bytes,
                                      int active_lanes,
                                      int stride) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane >= active_lanes) return;
  const std::uint64_t active_rank =
      static_cast<std::uint64_t>(warp * active_lanes + lane);
  const std::uint64_t active_threads =
      static_cast<std::uint64_t>((blockDim.x / 32) * active_lanes);
  for (std::uint64_t logical = active_rank; logical < total_bytes;
       logical += active_threads) {
    const std::uint64_t offset = (logical * static_cast<std::uint64_t>(stride)) %
                                 payload_bytes;
    destination[offset] = source[offset];
  }
}

__global__ void dependent_peer_read(const unsigned char* source,
                                    std::uint64_t payload_bytes,
                                    std::uint64_t total_bytes,
                                    unsigned long long* checksum,
                                    int active_lanes) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane >= active_lanes) return;
  const std::uint64_t active_rank =
      static_cast<std::uint64_t>(warp * active_lanes + lane);
  const std::uint64_t active_threads =
      static_cast<std::uint64_t>((blockDim.x / 32) * active_lanes);
  std::uint64_t index = active_rank % payload_bytes;
  unsigned long long local = 0;
  for (std::uint64_t logical = active_rank; logical < total_bytes;
       logical += active_threads) {
    const unsigned char value = source[index];
    local += value;
    index = (index + static_cast<std::uint64_t>(value) + 1) % payload_bytes;
  }
  atomicAdd(checksum, local);
}

struct HardwareContext {
  static constexpr std::size_t kAllocationBytes = (2U << 20) + 512;
  unsigned char* buffers[4] = {nullptr, nullptr, nullptr, nullptr};
  unsigned long long* checksums[4] = {nullptr, nullptr, nullptr, nullptr};
  cudaStream_t streams[4][4]{};

  HardwareContext() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count != 4) throw std::runtime_error("hardware mode requires exactly four GPUs");
    for (int device = 0; device < 4; ++device) {
      CUDA_CHECK(cudaSetDevice(device));
      CUDA_CHECK(cudaMalloc(&buffers[device], kAllocationBytes));
      CUDA_CHECK(cudaMalloc(&checksums[device], sizeof(unsigned long long)));
      CUDA_CHECK(cudaMemset(buffers[device], 0x5a, kAllocationBytes));
      CUDA_CHECK(cudaMemset(checksums[device], 0, sizeof(unsigned long long)));
      for (int peer = 0; peer < 4; ++peer) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][peer], cudaStreamNonBlocking));
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
  }

  ~HardwareContext() {
    for (int device = 0; device < 4; ++device) {
      cudaSetDevice(device);
      for (int peer = 0; peer < 4; ++peer) cudaStreamDestroy(streams[device][peer]);
      cudaFree(checksums[device]);
      cudaFree(buffers[device]);
    }
  }
};

Observation run_nccl_validation(const Point& point, HardwareContext& context) {
#ifdef SIMLLM_NVLINK_ENABLE_NCCL
  int devices[4] = {0, 1, 2, 3};
  ncclComm_t communicators[4];
  NCCL_CHECK(ncclCommInitAll(communicators, 4, devices));
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, context.streams[0][1]));
  NCCL_CHECK(ncclGroupStart());
  CUDA_CHECK(cudaSetDevice(0));
  NCCL_CHECK(ncclSend(context.buffers[0], point.payload_bytes, ncclUint8, 1,
                      communicators[0], context.streams[0][1]));
  CUDA_CHECK(cudaSetDevice(1));
  NCCL_CHECK(ncclRecv(context.buffers[1], point.payload_bytes, ncclUint8, 0,
                      communicators[1], context.streams[1][0]));
  NCCL_CHECK(ncclGroupEnd());
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaEventRecord(stop, context.streams[0][1]));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  for (auto& communicator : communicators) ncclCommDestroy(communicator);
  Observation result;
  result.elapsed_us = elapsed_ms * 1000.0;
  result.checksum = stable_checksum(point.point_id + ":nccl");
  result.protocol = "protocol_validation_only";
  return result;
#else
  (void)point;
  (void)context;
  throw std::runtime_error("NCCL validation row requires SIMLLM_NVLINK_ENABLE_NCCL");
#endif
}

Observation run_hardware(const Point& point, HardwareContext& context) {
  if (point.producer == "nccl_send_receive_validation") {
    return run_nccl_validation(point, context);
  }
  if (point.payload_bytes + 256 > HardwareContext::kAllocationBytes) {
    throw std::runtime_error("payload exceeds the fixed harness allocation");
  }
  const auto flows = flows_for(point);
  const auto total_bytes = point.payload_bytes * point.message_count;
  std::vector<cudaEvent_t> starts;
  std::vector<cudaEvent_t> stops;
  starts.reserve(flows.size());
  stops.reserve(flows.size());
  for (const auto& flow : flows) {
    const int source = flow.first;
    const int destination = flow.second;
    CUDA_CHECK(cudaSetDevice(source));
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    starts.push_back(start);
    stops.push_back(stop);
    auto* source_pointer = context.buffers[source] + point.source_alignment;
    auto* destination_pointer =
        context.buffers[destination] + point.destination_alignment;
    CUDA_CHECK(cudaEventRecord(start, context.streams[source][destination]));
    if (point.producer == "persistent_sm_peer_write") {
      persistent_peer_write<<<1, 256, 0, context.streams[source][destination]>>>(
          destination_pointer, source_pointer, point.payload_bytes, total_bytes,
          point.active_lanes, point.stride);
      CUDA_CHECK(cudaGetLastError());
    } else if (point.producer == "dependent_sm_peer_read") {
      dependent_peer_read<<<1, 256, 0, context.streams[source][destination]>>>(
          destination_pointer, point.payload_bytes, total_bytes,
          context.checksums[source], point.active_lanes);
      CUDA_CHECK(cudaGetLastError());
    } else if (point.producer == "copy_engine_reference") {
      for (std::uint64_t message = 0; message < point.message_count; ++message) {
        CUDA_CHECK(cudaMemcpyPeerAsync(destination_pointer, destination, source_pointer,
                                       source, point.payload_bytes,
                                       context.streams[source][destination]));
      }
    } else {
      throw std::runtime_error("unknown producer: " + point.producer);
    }
    CUDA_CHECK(cudaEventRecord(stop, context.streams[source][destination]));
  }
  double elapsed_us = 0.0;
  for (std::size_t i = 0; i < flows.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(flows[i].first));
    CUDA_CHECK(cudaEventSynchronize(stops[i]));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, starts[i], stops[i]));
    elapsed_us = std::max(elapsed_us, static_cast<double>(elapsed_ms) * 1000.0);
    CUDA_CHECK(cudaEventDestroy(starts[i]));
    CUDA_CHECK(cudaEventDestroy(stops[i]));
  }
  Observation result;
  result.elapsed_us = elapsed_us;
  result.checksum = stable_checksum(point.point_id + ":hardware");
  return result;
}

#endif

void write_result(std::ofstream& output, const Point& point, const Observation& observation,
                  const std::string& mode) {
  const auto packets_per_message =
      (point.payload_bytes + kCandidatePayloadBytes - 1) / kCandidatePayloadBytes;
  const auto flow_count = static_cast<std::uint64_t>(flows_for(point).size());
  const auto logical_bytes = point.payload_bytes * point.message_count * flow_count;
  const auto candidate_raw_bytes = logical_bytes +
      packets_per_message * kCandidateHeaderBytes * point.message_count * flow_count;
  const auto elapsed_seconds = observation.elapsed_us * 1.0e-6;
  const auto payload_gbps = elapsed_seconds > 0.0
      ? static_cast<double>(logical_bytes) / elapsed_seconds / 1.0e9
      : 0.0;
  output << "{"
         << "\"schema\":\"simllm-a100-nvlink-packet-observation-v1\","
         << "\"mode\":\"" << json_escape(mode) << "\","
         << "\"case_name\":\"" << json_escape(point.case_name) << "\","
         << "\"point_id\":\"" << json_escape(point.point_id) << "\","
         << "\"producer\":\"" << json_escape(point.producer) << "\","
         << "\"protocol_scope\":\"" << json_escape(observation.protocol) << "\","
         << "\"payload_bytes\":" << point.payload_bytes << ','
         << "\"message_count\":" << point.message_count << ','
         << "\"logical_bytes\":" << logical_bytes << ','
         << "\"candidate_packet_count\":" << packets_per_message * point.message_count
         << ','
         << "\"candidate_raw_bytes\":" << candidate_raw_bytes << ','
         << "\"elapsed_us\":" << std::fixed << std::setprecision(9)
         << observation.elapsed_us << ','
         << "\"payload_rate_gbps\":" << std::setprecision(6) << payload_gbps << ','
         << "\"checksum\":" << observation.checksum << ','
         << "\"checksum_ok\":" << (observation.checksum_ok ? "true" : "false") << ','
         << "\"pattern\":\"" << json_escape(point.pattern) << "\","
         << "\"measurement_claim\":" << (mode == "hardware" ? "\"unscored\"" : "false")
         << "}\n";
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
    std::cout << "TRAF-65 " << mode << " rows=" << points.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TRAF-65 FATAL: " << error.what() << '\n';
    return 2;
  }
}
