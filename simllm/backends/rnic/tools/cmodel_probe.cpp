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
    // "tx" is the slice-B transmit cell and is the default, so an invocation
    // that predates the receive half runs exactly the code it ran before.
    // "gap", "ud", "incast" and "duplex" are the slice-C receive cells.
    std::string mode{"tx"};
    std::string profile{"cx5_100g"};
    std::uint64_t size_bytes{8192};
    std::uint64_t depth{1};
    std::uint64_t mtu_bytes{0};
    std::uint64_t messages{64};
    std::uint64_t packetization{1};
    std::string trace_prefix;
    bool replay{false};

    // Receive-side cells.
    std::uint64_t burst_messages{128};
    std::uint64_t gap_ps{0};
    std::uint64_t senders{1};
    std::uint64_t qps{1};
    std::uint64_t loss_ppm{0};
    std::uint64_t loss_period{0};
    std::uint64_t loss_seed{1};
    std::uint64_t offered_pps{0};
    std::uint64_t offered_bps{0};
    std::uint64_t fabric_queue_bytes{0};
    std::uint64_t firmware_variant{0};
    // Fitted receive parameters. Zero keeps the profile's value.
    std::uint64_t rx_ingress_bytes{0};
    std::uint64_t rx_drain_bps{0};
    std::uint64_t rx_pps_per_qp_rc{0};
    std::uint64_t rx_pps_per_qp_ud{0};
    std::uint64_t rx_pps_per_nic{0};
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
        } else if (option == "--mode") {
            options.mode = value;
        } else if (option == "--trace-prefix") {
            options.trace_prefix = value;
        } else if (option == "--burst-messages") {
            options.burst_messages = parseUnsigned(value, "--burst-messages");
        } else if (option == "--gap-ps") {
            options.gap_ps = parseUnsigned(value, "--gap-ps");
        } else if (option == "--senders") {
            options.senders = parseUnsigned(value, "--senders");
        } else if (option == "--qps") {
            options.qps = parseUnsigned(value, "--qps");
        } else if (option == "--loss-ppm") {
            options.loss_ppm = parseUnsigned(value, "--loss-ppm");
        } else if (option == "--loss-period") {
            options.loss_period = parseUnsigned(value, "--loss-period");
        } else if (option == "--loss-seed") {
            options.loss_seed = parseUnsigned(value, "--loss-seed");
        } else if (option == "--offered-pps") {
            options.offered_pps = parseUnsigned(value, "--offered-pps");
        } else if (option == "--offered-bps") {
            options.offered_bps = parseUnsigned(value, "--offered-bps");
        } else if (option == "--fabric-queue-bytes") {
            options.fabric_queue_bytes =
                parseUnsigned(value, "--fabric-queue-bytes");
        } else if (option == "--firmware-variant") {
            options.firmware_variant =
                parseUnsigned(value, "--firmware-variant");
        } else if (option == "--rx-ingress-bytes") {
            options.rx_ingress_bytes =
                parseUnsigned(value, "--rx-ingress-bytes");
        } else if (option == "--rx-drain-bps") {
            options.rx_drain_bps = parseUnsigned(value, "--rx-drain-bps");
        } else if (option == "--rx-pps-per-qp-rc") {
            options.rx_pps_per_qp_rc =
                parseUnsigned(value, "--rx-pps-per-qp-rc");
        } else if (option == "--rx-pps-per-qp-ud") {
            options.rx_pps_per_qp_ud =
                parseUnsigned(value, "--rx-pps-per-qp-ud");
        } else if (option == "--rx-pps-per-nic") {
            options.rx_pps_per_nic = parseUnsigned(value, "--rx-pps-per-nic");
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

int runTransmit(const Options& options) {
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

// ---------------------------------------------------------------------------
// Slice C: the receive cells.
//
// Every cell drives the same facade the transmit cells drive, with the receive
// half selected. The probe owns the wire: it serializes each packet on a
// per-direction link, propagates it, may lose it in the fabric, hands it to the
// responder's receive entry point, and carries the responder's ACK or NAK back.
// Nothing here reaches around the facade.
// ---------------------------------------------------------------------------

// A reproducible loss generator, matching the test fake's: deterministic is
// one in every `period`, Bernoulli is a seeded xorshift at a rate in parts per
// million, so a cell replays exactly.
class LossSource {
public:
    LossSource(std::uint64_t period, std::uint64_t rate_ppm, std::uint64_t seed)
        : period_(period), rate_ppm_(rate_ppm),
          state_(seed == 0 ? 1 : seed) {}

    bool losesNext() {
        ++offered_;
        if (period_ != 0) {
            return offered_ % period_ == 0;
        }
        if (rate_ppm_ == 0) {
            return false;
        }
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return (state_ % 1000000ULL) < rate_ppm_;
    }

    std::uint64_t losses() const noexcept { return losses_; }
    void noteLoss() noexcept { ++losses_; }

private:
    std::uint64_t period_;
    std::uint64_t rate_ppm_;
    std::uint64_t state_;
    std::uint64_t offered_{0};
    std::uint64_t losses_{0};
};

// One direction of wire. Serialization is exact rational arithmetic and a
// finite queue refuses an attempt, which is what makes two senders share a
// link fairly instead of each pretending it owns one.
class Direction {
public:
    Direction(
        std::uint64_t link_bps,
        std::uint64_t one_way_latency_ps,
        std::uint64_t queue_bytes)
        : link_bps_(link_bps), latency_ps_(one_way_latency_ps),
          queue_bytes_(queue_bytes) {}

    bool accepts(std::uint64_t now_ps, std::uint64_t wire_bytes) const {
        if (queue_bytes_ == 0) {
            return true;
        }
        return queuedBytes(now_ps) + wire_bytes <= queue_bytes_;
    }

    std::uint64_t serialize(std::uint64_t now_ps, std::uint64_t wire_bytes) {
        const std::uint64_t start = std::max(now_ps, free_at_ps_);
        const std::uint64_t numerator =
            wire_bytes * 8 * kPicosecondsPerSecond + remainder_;
        remainder_ = numerator % link_bps_;
        free_at_ps_ = start + numerator / link_bps_;
        return free_at_ps_;
    }

    std::uint64_t queuedBytes(std::uint64_t now_ps) const {
        if (free_at_ps_ <= now_ps) {
            return 0;
        }
        return (free_at_ps_ - now_ps) / 8 * link_bps_ / kPicosecondsPerSecond;
    }

    std::uint64_t latencyPs() const noexcept { return latency_ps_; }
    std::uint64_t freeAtPs() const noexcept { return free_at_ps_; }

private:
    std::uint64_t link_bps_;
    std::uint64_t latency_ps_;
    std::uint64_t queue_bytes_;
    std::uint64_t free_at_ps_{0};
    std::uint64_t remainder_{0};
};

struct RxRow {
    std::uint64_t messages{0};
    std::uint64_t completions{0};
    std::uint64_t errors{0};
    std::uint64_t packets_issued{0};
    std::uint64_t packets_retransmitted{0};
    std::uint64_t recovery_episodes{0};
    std::uint64_t timeouts{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes{0};
    std::uint64_t first_packet_ps{0};
    std::uint64_t last_completion_ps{0};
    std::uint64_t warm_start_ps{0};
    std::uint64_t warm_payload_bytes{0};
    std::uint64_t gap_time_ps{0};
    std::uint64_t injected_losses{0};
    std::uint64_t rx_packets_offered{0};
    std::uint64_t rx_packets_delivered{0};
    std::uint64_t rx_payload_bytes{0};
    std::uint64_t rx_bytes_phy{0};
    std::uint64_t rx_discards_phy{0};
    std::uint64_t rx_discards_meter{0};
    std::uint64_t rx_discards_rate{0};
    std::uint64_t rx_discards_sequence{0};
    std::uint64_t rx_high_watermark{0};
    std::uint64_t out_of_sequence{0};
    std::uint64_t packet_seq_err{0};
    std::uint64_t roce_adp_retrans{0};
    std::uint64_t local_ack_timeout_err{0};
    std::uint64_t np_ecn_marked{0};
    std::uint64_t tx_pause_ctrl_phy{0};
    std::uint64_t late_releases{0};
    std::uint64_t sender0_payload_bytes{0};
    std::uint64_t sender1_payload_bytes{0};

    bool operator==(const RxRow& other) const {
        return std::memcmp(this, &other, sizeof(RxRow)) == 0;
    }
};

void applyFittedProfile(const Options& options, rnic_cm_profile* profile) {
    if (options.rx_ingress_bytes != 0) {
        profile->rx_ingress_bytes = options.rx_ingress_bytes;
    }
    if (options.rx_drain_bps != 0) {
        profile->rx_drain_bps = options.rx_drain_bps;
    }
    if (options.rx_pps_per_qp_rc != 0) {
        profile->rx_pps_per_qp_rc = options.rx_pps_per_qp_rc;
    }
    if (options.rx_pps_per_qp_ud != 0) {
        profile->rx_pps_per_qp_ud = options.rx_pps_per_qp_ud;
    }
    if (options.rx_pps_per_nic != 0) {
        profile->rx_pps_per_nic = options.rx_pps_per_nic;
    }
}

rnic_cm_device* makeEndpoint(
    const rnic_cm_profile& profile,
    const Options& options,
    std::uint32_t qpn,
    std::uint32_t source,
    std::uint64_t depth,
    bool receives,
    bool trace) {
    rnic_cm_config config;
    std::memset(&config, 0, sizeof(config));
    config.version = SIMLLM_RNIC_CM_ABI_VERSION;
    config.qpn = qpn;
    config.source = source;
    config.policy_context_token = qpn;
    config.sq_depth = depth;
    config.cq_depth = depth * 2;
    config.packetization = 1;
    config.trace_enabled = trace ? 1u : 0u;
    config.receive = receives ? 1u : 0u;
    config.firmware_counter_variant =
        static_cast<std::uint8_t>(options.firmware_variant);
    config.max_inflight_wqes = depth;
    config.mtu_bytes = options.mtu_bytes;
    return rnic_cm_create(&profile, &config);
}

// The unreliable-datagram cell. The offered rate is an input, not a
// transmit-side result, because the measured offer is above the profile's own
// single-QP transmit message rate. Packets go straight into the responder's
// receive entry point at the offered spacing.
RxRow runUdCell(const Options& options) {
    rnic_cm_profile profile;
    if (rnic_cm_profile_preset(options.profile.c_str(), &profile)
        != RNIC_CM_OK) {
        throw std::invalid_argument("unknown profile: " + options.profile);
    }
    applyFittedProfile(options, &profile);
    if (options.offered_pps == 0) {
        throw std::invalid_argument("the ud cell needs --offered-pps");
    }

    rnic_cm_device* responder =
        makeEndpoint(profile, options, 1, 0, 16, true, false);
    if (responder == nullptr) {
        throw std::runtime_error("responder construction failed");
    }
    RxRow row;
    row.messages = options.messages;
    try {
        // One packet per message at the offered rate, spread evenly across
        // the configured queue pairs.
        const std::uint64_t spacing_ps =
            kPicosecondsPerSecond / options.offered_pps;
        for (std::uint64_t index = 0; index < options.messages; ++index) {
            rnic_cm_packet packet;
            std::memset(&packet, 0, sizeof(packet));
            packet.qpn = static_cast<std::uint32_t>(index % options.qps) + 1;
            packet.destination = 1;
            packet.psn = static_cast<std::uint32_t>(index / options.qps);
            packet.payload_bytes = options.size_bytes;
            packet.wire_bytes = options.size_bytes + profile.wire_header_bytes;
            packet.kind = RNIC_CM_PACKET_DATA;
            packet.service = RNIC_CM_SERVICE_UD;
            packet.last_of_message = 1;
            rnic_cm_rx_result result;
            const std::uint64_t now_ps = index * spacing_ps;
            if (rnic_cm_rx_packet(responder, &packet, now_ps, &result)
                != RNIC_CM_OK) {
                throw std::runtime_error("responder refused a datagram");
            }
            if (result.has_reply != 0) {
                throw std::runtime_error(
                    "an unreliable datagram was acknowledged");
            }
            row.last_completion_ps = now_ps + spacing_ps;
        }
        rnic_cm_nic_counter_set counters;
        if (rnic_cm_nic_counters(responder, &counters) != RNIC_CM_OK) {
            throw std::runtime_error("responder refused a counter read");
        }
        row.rx_packets_offered = counters.rx_packets_offered;
        row.rx_packets_delivered = counters.rx_packets_delivered;
        row.rx_payload_bytes = counters.rx_payload_bytes_delivered;
        row.rx_bytes_phy = counters.rx_bytes_phy;
        row.rx_discards_phy = counters.rx_discards_phy;
        row.rx_discards_meter = counters.rx_discards_meter;
        row.rx_discards_rate = counters.rx_discards_rate;
        row.rx_discards_sequence = counters.rx_discards_sequence;
        row.rx_high_watermark = counters.rx_ingress_high_watermark_bytes;
        row.out_of_sequence = counters.out_of_sequence;
        row.packet_seq_err = counters.packet_seq_err;
        row.roce_adp_retrans = counters.roce_adp_retrans;
        row.np_ecn_marked = counters.np_ecn_marked_roce_packets;
        row.tx_pause_ctrl_phy = counters.tx_pause_ctrl_phy;
        row.completions = counters.rx_packets_delivered;
    } catch (...) {
        rnic_cm_destroy(responder);
        throw;
    }
    rnic_cm_destroy(responder);
    return row;
}

// One reliable-connection sender, as the probe's wire sees it.
struct Sender {
    rnic_cm_device* handle{nullptr};
    std::uint32_t qpn{0};
    std::uint64_t posted{0};
    std::uint64_t completions{0};
    std::uint64_t errors{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t next_burst_at_ps{0};
    std::uint64_t in_burst{0};
    // Attempt token by sequence number, so the responder's verdict can be
    // turned back into an event for the right attempt.
    std::map<std::uint32_t, std::uint64_t> tokens;
};

struct WireArrival {
    std::uint64_t when_ps{0};
    std::size_t sender{0};
    std::uint32_t psn{0};
    std::uint64_t token{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes{0};
    std::uint8_t kind{0};
    std::uint8_t last_of_message{0};
    bool lost{false};
};

struct WireReply {
    std::uint64_t when_ps{0};
    std::size_t sender{0};
    std::uint32_t psn{0};
    std::uint8_t kind{0};
};

// The reliable-connection cells: the gap sweep, the depth pair, the incast and
// the duplex pair are all this one loop with different inputs.
RxRow runRcCell(const Options& options, const std::string& trace_path) {
    rnic_cm_profile profile;
    if (rnic_cm_profile_preset(options.profile.c_str(), &profile)
        != RNIC_CM_OK) {
        throw std::invalid_argument("unknown profile: " + options.profile);
    }
    applyFittedProfile(options, &profile);

    const std::uint64_t one_way_ps = profile.wire_round_trip_floor_ps / 2;
    // A contended fabric holds a standing egress queue in front of the
    // receiver, and its depth is what sets how far go-back-N has to replay.
    // It is a fabric constant, declared, not an endpoint parameter: the
    // endpoint cannot see it and cannot fit it.
    const std::uint64_t fabric_delay_ps = options.fabric_queue_bytes * 8
        * kPicosecondsPerSecond / profile.link_bps;
    const std::uint64_t mtu = options.mtu_bytes != 0 ? options.mtu_bytes
                                                     : profile.mtu_bytes;
    const std::uint64_t packets_per_message =
        options.size_bytes == 0 ? 1 : (options.size_bytes + mtu - 1) / mtu;

    std::vector<Sender> senders(options.senders);
    rnic_cm_device* responder = nullptr;
    RxRow row;
    row.messages = options.messages;

    Direction forward(
        profile.link_bps, one_way_ps, options.fabric_queue_bytes);
    Direction reverse(profile.link_bps, one_way_ps, 0);
    LossSource loss(options.loss_period, options.loss_ppm, options.loss_seed);
    // One wire queue, in time order. Three separate queues drained one after
    // another would hand a device an event stamped earlier than one it has
    // already seen, and a facade whose clock has moved past a timestamp
    // refuses it. With two senders that happens constantly.
    enum class WireKind { Arrival, Reply, Event };
    struct Pending {
        WireKind kind{WireKind::Arrival};
        WireArrival arrival;
        WireReply reply;
        std::size_t sender{0};
        std::uint64_t token{0};
        std::uint32_t event_kind{0};
        std::uint64_t when_ps{0};
    };
    std::map<std::pair<std::uint64_t, std::uint64_t>, Pending> wire;
    std::uint64_t sequence = 1;

    try {
        responder = makeEndpoint(profile, options, 1, 0, 16, true, false);
        if (responder == nullptr) {
            throw std::runtime_error("responder construction failed");
        }
        for (std::size_t index = 0; index < senders.size(); ++index) {
            senders[index].qpn = static_cast<std::uint32_t>(index + 2);
            senders[index].handle = makeEndpoint(
                profile,
                options,
                senders[index].qpn,
                static_cast<std::uint32_t>(index + 2),
                options.depth,
                true,
                !trace_path.empty() && index == 0);
            if (senders[index].handle == nullptr) {
                throw std::runtime_error("sender construction failed");
            }
        }

        // A sender that is asked for a fixed offered rate paces itself by
        // holding its next post until the offered spacing has elapsed. That is
        // how the duplex pair is driven at the measured per-direction rate.
        const std::uint64_t message_spacing_ps = options.offered_bps == 0
            ? 0
            : options.size_bytes * 8 * kPicosecondsPerSecond
                / options.offered_bps;

        std::vector<rnic_cm_packet> drained(256);
        std::vector<rnic_cm_cqe> cqes(256);
        std::uint64_t now_ps = 0;
        std::uint64_t guard = 0;
        const std::uint64_t total_messages =
            options.messages * options.senders;
        std::uint64_t completions = 0;
        std::uint64_t errors = 0;
        std::uint64_t warm_target = total_messages / 2;

        while (completions + errors < total_messages) {
            if (++guard > 400000000ULL) {
                throw std::runtime_error("receive probe did not converge");
            }

            while (!wire.empty() && wire.begin()->first.first <= now_ps) {
                const Pending pending = wire.begin()->second;
                wire.erase(wire.begin());
                if (pending.kind == WireKind::Event) {
                    rnic_cm_event_info info;
                    std::memset(&info, 0, sizeof(info));
                    info.kind = pending.event_kind;
                    info.token = pending.token;
                    if (rnic_cm_event(
                            senders[pending.sender].handle, &info,
                            pending.when_ps)
                        != RNIC_CM_OK) {
                        throw std::runtime_error("sender refused a wire event");
                    }
                    continue;
                }
                if (pending.kind == WireKind::Reply) {
                    rnic_cm_packet nak;
                    std::memset(&nak, 0, sizeof(nak));
                    nak.qpn = senders[pending.reply.sender].qpn;
                    nak.psn = pending.reply.psn;
                    nak.kind = RNIC_CM_PACKET_NAK;
                    nak.service = RNIC_CM_SERVICE_RC;
                    if (rnic_cm_rx_packet(
                            senders[pending.reply.sender].handle, &nak,
                            pending.reply.when_ps, nullptr)
                        != RNIC_CM_OK) {
                        throw std::runtime_error("requester refused a NAK");
                    }
                    continue;
                }
                const WireArrival arrival = pending.arrival;
                Sender& sender = senders[arrival.sender];
                if (arrival.lost) {
                    continue;
                }
                rnic_cm_event_info rx_arrived;
                std::memset(&rx_arrived, 0, sizeof(rx_arrived));
                rx_arrived.kind = RNIC_CM_EVENT_PACKET_RX_ARRIVED;
                rx_arrived.token = arrival.token;
                if (rnic_cm_event(sender.handle, &rx_arrived, arrival.when_ps)
                    != RNIC_CM_OK) {
                    throw std::runtime_error("sender refused an arrival");
                }

                rnic_cm_packet packet;
                std::memset(&packet, 0, sizeof(packet));
                packet.qpn = sender.qpn;
                packet.destination =
                    static_cast<std::uint32_t>(arrival.sender + 2);
                packet.psn = arrival.psn;
                packet.payload_bytes = arrival.payload_bytes;
                packet.wire_bytes = arrival.wire_bytes;
                packet.kind = arrival.kind;
                packet.service = RNIC_CM_SERVICE_RC;
                packet.last_of_message = arrival.last_of_message;
                rnic_cm_rx_result verdict;
                if (rnic_cm_rx_packet(
                        responder, &packet, arrival.when_ps, &verdict)
                    != RNIC_CM_OK) {
                    throw std::runtime_error("responder refused a packet");
                }
                // Any packet the responder acknowledges retires its attempt,
                // and it acknowledges a duplicate as well as a delivery. A
                // replay whose original had already arrived is a duplicate,
                // so leaving it unacknowledged would strand it on the timer.
                if (verdict.has_reply != 0
                    && verdict.reply_kind == RNIC_CM_PACKET_ACK) {
                    Pending owed;
                    owed.kind = WireKind::Event;
                    owed.when_ps = arrival.when_ps + one_way_ps;
                    owed.sender = arrival.sender;
                    owed.token = arrival.token;
                    owed.event_kind = RNIC_CM_EVENT_PACKET_DELIVERED;
                    wire.emplace(
                        std::make_pair(owed.when_ps, sequence++), owed);
                }
                if (verdict.has_reply != 0
                    && verdict.reply_kind == RNIC_CM_PACKET_NAK) {
                    Pending pending_reply;
                    pending_reply.kind = WireKind::Reply;
                    const std::uint64_t finish = reverse.serialize(
                        arrival.when_ps, verdict.reply_wire_bytes);
                    pending_reply.reply.when_ps = finish + one_way_ps;
                    pending_reply.reply.sender = arrival.sender;
                    pending_reply.reply.psn = verdict.reply_psn;
                    pending_reply.reply.kind = RNIC_CM_PACKET_NAK;
                    pending_reply.when_ps = pending_reply.reply.when_ps;
                    wire.emplace(
                        std::make_pair(pending_reply.when_ps, sequence++),
                        pending_reply);
                }
            }

            for (std::size_t index = 0; index < senders.size(); ++index) {
                Sender& sender = senders[index];
                bool posted = false;
                while (sender.posted < options.messages
                       && now_ps >= sender.next_burst_at_ps) {
                    if (options.burst_messages != 0 && options.gap_ps != 0
                        && sender.in_burst >= options.burst_messages) {
                        // The gap is between bursts, not inside one: it opens
                        // when the burst has fully drained, which is what
                        // gives the ingress meter its idle time.
                        if (sender.completions + sender.errors
                            < sender.posted) {
                            break;
                        }
                        sender.in_burst = 0;
                        sender.next_burst_at_ps = now_ps + options.gap_ps;
                        if (index == 0) {
                            row.gap_time_ps += options.gap_ps;
                        }
                        break;
                    }
                    rnic_cm_wqe request;
                    std::memset(&request, 0, sizeof(request));
                    request.wr_id = sender.posted + 1;
                    request.destination = 1;
                    request.payload_bytes = options.size_bytes;
                    request.sge_count = 1;
                    request.signaled = 1;
                    std::uint64_t wqe_id = 0;
                    const int status = rnic_cm_post(
                        sender.handle, &request, now_ps, &wqe_id);
                    if (status == RNIC_CM_ERROR_SQ_FULL) {
                        break;
                    }
                    if (status != RNIC_CM_OK) {
                        throw std::runtime_error("sender refused a request");
                    }
                    ++sender.posted;
                    ++sender.in_burst;
                    posted = true;
                    if (message_spacing_ps != 0) {
                        sender.next_burst_at_ps = now_ps + message_spacing_ps;
                        break;
                    }
                }
                if (posted) {
                    rnic_cm_doorbell_batch batch;
                    rnic_cm_doorbell(sender.handle, now_ps, &batch);
                }
                std::uint64_t changes = 0;
                if (rnic_cm_progress(sender.handle, now_ps, &changes)
                    != RNIC_CM_OK) {
                    throw std::runtime_error("sender refused to progress");
                }

                std::size_t count = 0;
                do {
                    if (rnic_cm_tx_next(
                            sender.handle, drained.data(), drained.size(),
                            &count)
                        != RNIC_CM_OK) {
                        throw std::runtime_error("sender refused a drain");
                    }
                    for (std::size_t slot = 0; slot < count; ++slot) {
                        const rnic_cm_packet& emitted = drained[slot];
                        if (row.first_packet_ps == 0) {
                            row.first_packet_ps = emitted.issued_at_ps;
                        }
                        const std::uint64_t finish = forward.serialize(
                            emitted.issued_at_ps, emitted.wire_bytes);
                        Pending finished;
                        finished.kind = WireKind::Event;
                        finished.when_ps = finish;
                        finished.sender = index;
                        finished.token = emitted.token;
                        finished.event_kind =
                            RNIC_CM_EVENT_PACKET_TX_FINISHED;
                        wire.emplace(
                            std::make_pair(finish, sequence++), finished);
                        WireArrival arrival;
                        arrival.when_ps = finish + one_way_ps + fabric_delay_ps;
                        arrival.sender = index;
                        arrival.psn = emitted.psn;
                        arrival.token = emitted.token;
                        arrival.payload_bytes = emitted.payload_bytes;
                        arrival.wire_bytes = emitted.wire_bytes;
                        arrival.kind = emitted.kind;
                        arrival.last_of_message = emitted.last_of_message;
                        arrival.lost = loss.losesNext();
                        if (arrival.lost) {
                            loss.noteLoss();
                        }
                        Pending pending_arrival;
                        pending_arrival.kind = WireKind::Arrival;
                        pending_arrival.arrival = arrival;
                        pending_arrival.when_ps = arrival.when_ps;
                        wire.emplace(
                            std::make_pair(arrival.when_ps, sequence++),
                            pending_arrival);
                    }
                } while (count == drained.size());

                std::size_t polled = 0;
                if (rnic_cm_poll(
                        sender.handle, cqes.data(), cqes.size(), now_ps,
                        &polled)
                    != RNIC_CM_OK) {
                    throw std::runtime_error("sender refused a poll");
                }
                for (std::size_t slot = 0; slot < polled; ++slot) {
                    if (cqes[slot].status == RNIC_CM_COMPLETION_SUCCESS) {
                        ++sender.completions;
                        ++completions;
                    } else {
                        ++sender.errors;
                        ++errors;
                    }
                    // The completion carries the message, so a completed
                    // message is exactly its offered byte count. Reading the
                    // count off the CQE would depend on which optional fields
                    // the queue chose to publish.
                    sender.payload_bytes += options.size_bytes;
                    row.last_completion_ps = cqes[slot].polled_at_ps;
                    if (completions + errors == warm_target) {
                        row.warm_start_ps = cqes[slot].polled_at_ps;
                        row.warm_payload_bytes = 0;
                        for (const Sender& other : senders) {
                            row.warm_payload_bytes += other.payload_bytes;
                        }
                    }
                }
            }

            std::optional<std::uint64_t> next;
            const auto consider = [&next](std::uint64_t when) {
                if (!next.has_value() || when < *next) {
                    next = when;
                }
            };
            for (const Sender& sender : senders) {
                std::uint64_t device_next = 0;
                if (rnic_cm_next_event_ps(sender.handle, &device_next)
                    == RNIC_CM_OK) {
                    consider(device_next);
                }
                if (sender.posted < options.messages
                    && sender.next_burst_at_ps > now_ps) {
                    consider(sender.next_burst_at_ps);
                }
            }
            if (!wire.empty()) {
                consider(wire.begin()->first.first);
            }
            if (!next.has_value()) {
                bool more = false;
                for (const Sender& sender : senders) {
                    more = more || sender.posted < options.messages;
                }
                if (more) {
                    continue;
                }
                break;
            }
            now_ps = std::max(now_ps, *next);
        }

        row.completions = completions;
        row.errors = errors;
        row.injected_losses = loss.losses();
        for (std::size_t index = 0; index < senders.size(); ++index) {
            rnic_cm_counter_set counters;
            rnic_cm_nic_counter_set nic;
            if (rnic_cm_counters(senders[index].handle, &counters) != RNIC_CM_OK
                || rnic_cm_nic_counters(senders[index].handle, &nic)
                    != RNIC_CM_OK) {
                throw std::runtime_error("sender refused a counter read");
            }
            row.packets_issued += counters.tx_packets;
            row.payload_bytes += counters.tx_payload_bytes;
            row.wire_bytes += counters.tx_wire_bytes;
            row.late_releases += counters.tx_late_releases;
            row.packets_retransmitted += nic.tx_packets_retransmitted;
            row.recovery_episodes += nic.tx_recovery_episodes;
            row.timeouts += nic.tx_timeouts;
            row.packet_seq_err += nic.packet_seq_err;
            row.roce_adp_retrans += nic.roce_adp_retrans;
            row.local_ack_timeout_err += nic.local_ack_timeout_err;
            if (index == 0) {
                row.sender0_payload_bytes = senders[index].payload_bytes;
            } else if (index == 1) {
                row.sender1_payload_bytes = senders[index].payload_bytes;
            }
        }
        rnic_cm_nic_counter_set responder_counters;
        if (rnic_cm_nic_counters(responder, &responder_counters)
            != RNIC_CM_OK) {
            throw std::runtime_error("responder refused a counter read");
        }
        row.rx_packets_offered = responder_counters.rx_packets_offered;
        row.rx_packets_delivered = responder_counters.rx_packets_delivered;
        row.rx_payload_bytes = responder_counters.rx_payload_bytes_delivered;
        row.rx_bytes_phy = responder_counters.rx_bytes_phy;
        row.rx_discards_phy = responder_counters.rx_discards_phy;
        row.rx_discards_meter = responder_counters.rx_discards_meter;
        row.rx_discards_rate = responder_counters.rx_discards_rate;
        row.rx_discards_sequence = responder_counters.rx_discards_sequence;
        row.rx_high_watermark =
            responder_counters.rx_ingress_high_watermark_bytes;
        row.out_of_sequence = responder_counters.out_of_sequence;
        row.np_ecn_marked = responder_counters.np_ecn_marked_roce_packets;
        row.tx_pause_ctrl_phy = responder_counters.tx_pause_ctrl_phy;
        if (!trace_path.empty()) {
            rnic_cm_trace(senders[0].handle, trace_path.c_str());
        }
        (void)packets_per_message;
    } catch (...) {
        for (Sender& sender : senders) {
            rnic_cm_destroy(sender.handle);
        }
        rnic_cm_destroy(responder);
        throw;
    }
    for (Sender& sender : senders) {
        rnic_cm_destroy(sender.handle);
    }
    rnic_cm_destroy(responder);
    return row;
}

int runReceive(const Options& options) {
    if (options.mode != "gap" && options.mode != "ud"
        && options.mode != "incast" && options.mode != "duplex") {
        throw std::invalid_argument("unknown probe mode: " + options.mode);
    }
    const bool datagram = options.mode == "ud";
    std::string trace_a;
    std::string trace_b;
    if (!options.trace_prefix.empty()) {
        trace_a = options.trace_prefix + ".a.trace";
        trace_b = options.trace_prefix + ".b.trace";
    }
    const RxRow row =
        datagram ? runUdCell(options) : runRcCell(options, trace_a);
    std::uint64_t replay_identical = 1;
    if (options.replay) {
        const RxRow again =
            datagram ? runUdCell(options) : runRcCell(options, trace_b);
        replay_identical = (again == row) ? 1 : 0;
        if (!trace_a.empty() && readFile(trace_a) != readFile(trace_b)) {
            replay_identical = 0;
        }
    }

    std::cout
        << "mode,profile,size_bytes,depth,gap_ps,burst_messages,senders,qps,"
           "loss_ppm,offered_pps,offered_bps,messages,completions,errors,"
           "packets_issued,packets_retransmitted,recovery_episodes,timeouts,"
           "payload_bytes,wire_bytes,first_packet_ps,last_completion_ps,"
           "warm_start_ps,warm_payload_bytes,gap_time_ps,injected_losses,"
           "rx_packets_offered,rx_packets_delivered,rx_payload_bytes,"
           "rx_bytes_phy,rx_discards_phy,rx_discards_meter,rx_discards_rate,"
           "rx_discards_sequence,rx_high_watermark,out_of_sequence,"
           "packet_seq_err,roce_adp_retrans,local_ack_timeout_err,"
           "np_ecn_marked,tx_pause_ctrl_phy,late_releases,"
           "sender0_payload_bytes,sender1_payload_bytes,replay_identical\n"
        << options.mode << ',' << options.profile << ',' << options.size_bytes
        << ',' << options.depth << ',' << options.gap_ps << ','
        << options.burst_messages << ',' << options.senders << ','
        << options.qps << ',' << options.loss_ppm << ','
        << options.offered_pps << ',' << options.offered_bps << ','
        << row.messages << ',' << row.completions << ',' << row.errors << ','
        << row.packets_issued << ',' << row.packets_retransmitted << ','
        << row.recovery_episodes << ',' << row.timeouts << ','
        << row.payload_bytes << ',' << row.wire_bytes << ','
        << row.first_packet_ps << ',' << row.last_completion_ps << ','
        << row.warm_start_ps << ',' << row.warm_payload_bytes << ','
        << row.gap_time_ps << ',' << row.injected_losses << ',' << row.rx_packets_offered << ','
        << row.rx_packets_delivered << ',' << row.rx_payload_bytes << ','
        << row.rx_bytes_phy << ',' << row.rx_discards_phy << ','
        << row.rx_discards_meter << ',' << row.rx_discards_rate << ','
        << row.rx_discards_sequence << ',' << row.rx_high_watermark << ','
        << row.out_of_sequence << ',' << row.packet_seq_err << ','
        << row.roce_adp_retrans << ',' << row.local_ack_timeout_err << ','
        << row.np_ecn_marked << ',' << row.tx_pause_ctrl_phy << ','
        << row.late_releases << ',' << row.sender0_payload_bytes << ','
        << row.sender1_payload_bytes << ',' << replay_identical << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.mode == "tx") {
            return runTransmit(options);
        }
        return runReceive(options);
    } catch (const std::exception& error) {
        std::cerr << "simllm_rnic_cmodel_probe: " << error.what() << '\n';
        return 2;
    }
}
