// Drives the RNIC golden model's C facade over one cell of the slice-B grid
// and prints one CSV row of integers. Every derived quantity is left to the
// study script, so the probe itself has no floating-point arithmetic and its
// output is exactly reproducible.

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "simllm/rnic/rnic_cmodel_c.h"

namespace {

constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;

struct Options {
    std::string profile{"cx5_100g"};
    std::uint64_t size_bytes{8192};
    std::uint64_t depth{1};
    std::uint64_t mtu_bytes{0};
    std::uint64_t messages{64};
    std::uint64_t packetization{1};
    std::string trace_prefix;
    bool replay{false};
};

std::uint64_t parseUnsigned(const std::string& text, const char* option) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value, 10);
    if (parsed.ec == std::errc::result_out_of_range) {
        throw std::out_of_range(
            std::string(option) + " is outside the uint64 range");
    }
    if (text.empty() || parsed.ec != std::errc{}
        || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument(
            std::string(option) + " must be an unsigned integer");
    }
    return value;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--replay") {
            options.replay = true;
            continue;
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("every probe option needs a value");
        }
        const std::string value = argv[index + 1];
        ++index;
        if (option == "--profile") {
            options.profile = value;
        } else if (option == "--trace-prefix") {
            options.trace_prefix = value;
        } else if (option == "--size-bytes") {
            options.size_bytes = parseUnsigned(value, "--size-bytes");
        } else if (option == "--depth") {
            options.depth = parseUnsigned(value, "--depth");
        } else if (option == "--mtu-bytes") {
            options.mtu_bytes = parseUnsigned(value, "--mtu-bytes");
        } else if (option == "--messages") {
            options.messages = parseUnsigned(value, "--messages");
        } else if (option == "--packetization") {
            options.packetization = parseUnsigned(value, "--packetization");
        } else {
            throw std::invalid_argument("unknown probe option: " + option);
        }
    }
    if (options.depth == 0 || options.messages == 0) {
        throw std::invalid_argument("probe depth and message count must be positive");
    }
    return options;
}

// The wire behind the facade: it serializes each packet at the link rate with
// exact rational arithmetic, adds one fixed one-way latency in each
// direction, and acknowledges every packet. It drops nothing.
class ProbeWire {
public:
    ProbeWire(std::uint64_t link_bps, std::uint64_t one_way_latency_ps)
        : link_bps_(link_bps), latency_ps_(one_way_latency_ps) {
        if (link_bps_ == 0) {
            throw std::invalid_argument("probe wire needs a positive link rate");
        }
    }

    void accept(const rnic_cm_packet& packet) {
        const std::uint64_t start =
            std::max(packet.issued_at_ps, link_free_at_ps_);
        const std::uint64_t finish = start + serialize(packet.wire_bytes * 8);
        link_free_at_ps_ = finish;
        schedule(finish, packet.token, RNIC_CM_EVENT_PACKET_TX_FINISHED);
        schedule(
            finish + latency_ps_, packet.token,
            RNIC_CM_EVENT_PACKET_RX_ARRIVED);
        schedule(
            finish + 2 * latency_ps_, packet.token,
            RNIC_CM_EVENT_PACKET_DELIVERED);
    }

    std::optional<std::uint64_t> nextEventTime() const {
        if (pending_.empty()) {
            return std::nullopt;
        }
        return pending_.begin()->first.first;
    }

    std::vector<std::pair<std::uint64_t, rnic_cm_event_info>> takeDue(
        std::uint64_t now_ps) {
        std::vector<std::pair<std::uint64_t, rnic_cm_event_info>> due;
        while (!pending_.empty() && pending_.begin()->first.first <= now_ps) {
            due.push_back(std::make_pair(
                pending_.begin()->first.first, pending_.begin()->second));
            pending_.erase(pending_.begin());
        }
        return due;
    }

private:
    std::uint64_t serialize(std::uint64_t bits) {
        const std::uint64_t numerator =
            bits * kPicosecondsPerSecond + remainder_;
        remainder_ = numerator % link_bps_;
        return numerator / link_bps_;
    }

    void schedule(std::uint64_t when, std::uint64_t token, std::uint32_t kind) {
        rnic_cm_event_info event;
        std::memset(&event, 0, sizeof(event));
        event.kind = kind;
        event.token = token;
        pending_.emplace(std::make_pair(when, next_sequence_++), event);
    }

    std::uint64_t link_bps_;
    std::uint64_t latency_ps_;
    std::uint64_t link_free_at_ps_{0};
    std::uint64_t remainder_{0};
    std::uint64_t next_sequence_{1};
    std::map<std::pair<std::uint64_t, std::uint64_t>, rnic_cm_event_info>
        pending_;
};

struct Row {
    std::uint64_t messages{0};
    std::uint64_t completions{0};
    std::uint64_t errors{0};
    std::uint64_t packets{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes{0};
    std::uint64_t last_completion_ps{0};
    std::uint64_t first_packet_ps{0};
    std::uint64_t last_packet_ps{0};
    std::uint64_t late_releases{0};
    std::uint64_t window_stalls{0};
    std::uint64_t pacer_stalls{0};
    std::uint64_t posted{0};
    std::uint64_t delivered{0};
    std::uint64_t reclaimed{0};
    std::uint64_t cq_overruns{0};

    bool operator==(const Row& other) const {
        return std::memcmp(this, &other, sizeof(Row)) == 0;
    }
};

Row runCell(const Options& options, const std::string& trace_path) {
    rnic_cm_profile profile;
    if (rnic_cm_profile_preset(options.profile.c_str(), &profile)
        != RNIC_CM_OK) {
        throw std::invalid_argument("unknown profile: " + options.profile);
    }

    rnic_cm_config config;
    std::memset(&config, 0, sizeof(config));
    config.version = SIMLLM_RNIC_CM_ABI_VERSION;
    config.qpn = 1;
    config.source = 0;
    config.policy_context_token = 1;
    config.sq_depth = options.depth;
    config.cq_depth = options.depth * 2;
    config.packetization = options.packetization != 0 ? 1u : 0u;
    config.trace_enabled = trace_path.empty() ? 0u : 1u;
    config.max_inflight_wqes = options.depth;
    config.mtu_bytes = options.mtu_bytes;

    rnic_cm_device* device = rnic_cm_create(&profile, &config);
    if (device == nullptr) {
        throw std::runtime_error("facade construction failed");
    }

    ProbeWire wire(profile.link_bps, profile.wire_round_trip_floor_ps / 2);
    std::vector<rnic_cm_packet> packets(256);
    std::vector<rnic_cm_cqe> cqes(256);
    Row row;
    row.messages = options.messages;
    std::uint64_t next_wr = 1;
    std::uint64_t now_ps = 0;
    std::uint64_t outstanding = 0;
    std::uint64_t guard = 0;

    try {
        while (row.completions + row.errors < options.messages) {
            if (++guard > 200000000ULL) {
                throw std::runtime_error("probe loop did not converge");
            }
            for (const auto& due : wire.takeDue(now_ps)) {
                if (rnic_cm_event(device, &due.second, due.first)
                    != RNIC_CM_OK) {
                    throw std::runtime_error("facade refused a wire event");
                }
            }
            bool posted = false;
            while (next_wr <= options.messages && outstanding < options.depth) {
                rnic_cm_wqe request;
                std::memset(&request, 0, sizeof(request));
                request.wr_id = next_wr;
                request.destination = 1;
                request.payload_bytes = options.size_bytes;
                request.sge_count = 1;
                request.signaled = 1;
                std::uint64_t wqe_id = 0;
                const int status =
                    rnic_cm_post(device, &request, now_ps, &wqe_id);
                if (status == RNIC_CM_ERROR_SQ_FULL) {
                    break;
                }
                if (status != RNIC_CM_OK) {
                    throw std::runtime_error("facade refused a work request");
                }
                ++next_wr;
                ++outstanding;
                posted = true;
            }
            if (posted) {
                rnic_cm_doorbell_batch batch;
                if (rnic_cm_doorbell(device, now_ps, &batch) != RNIC_CM_OK) {
                    throw std::runtime_error("facade refused a doorbell");
                }
            }
            std::uint64_t changes = 0;
            if (rnic_cm_progress(device, now_ps, &changes) != RNIC_CM_OK) {
                throw std::runtime_error("facade refused to progress");
            }
            std::size_t drained = 0;
            do {
                if (rnic_cm_tx_next(
                        device, packets.data(), packets.size(), &drained)
                    != RNIC_CM_OK) {
                    throw std::runtime_error("facade refused a transmit drain");
                }
                for (std::size_t index = 0; index < drained; ++index) {
                    if (row.first_packet_ps == 0) {
                        row.first_packet_ps = packets[index].issued_at_ps;
                    }
                    row.last_packet_ps = packets[index].issued_at_ps;
                    wire.accept(packets[index]);
                }
            } while (drained == packets.size());

            std::size_t polled = 0;
            if (rnic_cm_poll(
                    device, cqes.data(), cqes.size(), now_ps, &polled)
                != RNIC_CM_OK) {
                throw std::runtime_error("facade refused a poll");
            }
            for (std::size_t index = 0; index < polled; ++index) {
                if (cqes[index].status == RNIC_CM_COMPLETION_SUCCESS) {
                    ++row.completions;
                } else {
                    ++row.errors;
                }
                row.last_completion_ps = cqes[index].polled_at_ps;
                --outstanding;
            }

            std::uint64_t device_next = 0;
            std::optional<std::uint64_t> next;
            if (rnic_cm_next_event_ps(device, &device_next) == RNIC_CM_OK) {
                next = device_next;
            }
            const auto wire_next = wire.nextEventTime();
            if (wire_next.has_value()
                && (!next.has_value() || *wire_next < *next)) {
                next = wire_next;
            }
            if (!next.has_value()) {
                if (next_wr <= options.messages) {
                    continue;
                }
                break;
            }
            now_ps = std::max(now_ps, *next);
        }

        rnic_cm_counter_set counters;
        if (rnic_cm_counters(device, &counters) != RNIC_CM_OK) {
            throw std::runtime_error("facade refused a counter read");
        }
        row.packets = counters.tx_packets;
        row.payload_bytes = counters.tx_payload_bytes;
        row.wire_bytes = counters.tx_wire_bytes;
        row.late_releases = counters.tx_late_releases;
        row.window_stalls = counters.tx_window_stalls;
        row.pacer_stalls = counters.tx_pacer_stalls;
        row.posted = counters.posted_wqes;
        row.delivered = counters.network_delivered;
        row.reclaimed = counters.sq_reclaimed_wqes;
        row.cq_overruns = counters.cq_overruns;
        if (!trace_path.empty()
            && rnic_cm_trace(device, trace_path.c_str()) != RNIC_CM_OK) {
            throw std::runtime_error("facade refused to write its trace");
        }
    } catch (...) {
        rnic_cm_destroy(device);
        throw;
    }
    rnic_cm_destroy(device);
    return row;
}

std::string readFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

int run(const Options& options) {
    std::string trace_a;
    std::string trace_b;
    if (!options.trace_prefix.empty()) {
        trace_a = options.trace_prefix + ".a.trace";
        trace_b = options.trace_prefix + ".b.trace";
    }
    const Row row = runCell(options, trace_a);
    std::uint64_t replay_identical = 1;
    if (options.replay) {
        const Row again = runCell(options, trace_b);
        replay_identical = (again == row) ? 1 : 0;
        if (!trace_a.empty() && readFile(trace_a) != readFile(trace_b)) {
            replay_identical = 0;
        }
    }

    std::cout
        << "profile,size_bytes,depth,mtu_bytes,messages,completions,errors,"
           "packets,payload_bytes,wire_bytes,last_completion_ps,"
           "first_packet_ps,last_packet_ps,late_releases,window_stalls,"
           "pacer_stalls,posted,delivered,reclaimed,cq_overruns,"
           "replay_identical\n"
        << options.profile << ',' << options.size_bytes << ','
        << options.depth << ',' << options.mtu_bytes << ',' << row.messages
        << ',' << row.completions << ',' << row.errors << ',' << row.packets
        << ',' << row.payload_bytes << ',' << row.wire_bytes << ','
        << row.last_completion_ps << ',' << row.first_packet_ps << ','
        << row.last_packet_ps << ',' << row.late_releases << ','
        << row.window_stalls << ',' << row.pacer_stalls << ',' << row.posted
        << ',' << row.delivered << ',' << row.reclaimed << ','
        << row.cq_overruns << ',' << replay_identical << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "simllm_rnic_cmodel_probe: " << error.what() << '\n';
        return 2;
    }
}
