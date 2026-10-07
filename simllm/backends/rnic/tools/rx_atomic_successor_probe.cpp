#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

#include "simllm/rnic/rnic_cmodel_c.h"
#include "simllm/rnic/rnic_device.h"

namespace {
using namespace simllm::rnic;
using U64 = std::uint64_t;
std::string fixture;
std::filesystem::path output;
void require(bool value, const std::string& reason) {
    if (!value) throw std::runtime_error(fixture + ": " + reason);
}
class EmptyPort final : public NetworkPort {
public:
    NetworkPortCapabilities capabilities() const noexcept override {
        NetworkPortCapabilities caps;
        caps.abi_version = kNetworkPortAbiVersionV2;
        caps.packet_attempt_events = true;
        return caps;
    }
    NetworkSubmitResult trySubmit(const NetworkTxDescriptor&, Picoseconds) override {
        throw std::logic_error("receive-only fixture attempted transmission");
    }
};
struct Snapshot {
    U64 occupancy, offered, wire, delivered, meter, rate, sequence, duplicate;
    bool operator==(const Snapshot& other) const {
        return occupancy == other.occupancy && offered == other.offered
            && wire == other.wire && delivered == other.delivered
            && meter == other.meter && rate == other.rate
            && sequence == other.sequence && duplicate == other.duplicate;
    }
};
class Receiver {
public:
    Receiver(bool facade, U64 width, U64 rate = 96600000000ULL,
             U64 capacity = 262016, U64 pps = 0, unsigned fluid = 0,
             const std::string& suffix = "") : facade_(facade) {
        trace_path_ = output / (fixture + "_" + (facade ? "facade" : "native")
                               + "_w" + std::to_string(width) + suffix + ".trace");
        result_bytes_.open(trace_path_.string() + ".results.bin", std::ios::binary);
        if (!facade_) {
            RnicDeviceConfig config;
            config.identity.qpn = 7; config.identity.policy_context_token = 1;
            config.work_queue.qpn = 7; config.work_queue.policy_context_token = 1;
            config.network.enabled = true;
            config.network.abi_version = kNetworkPortAbiVersionV2;
            config.network.packetization.enabled = true;
            auto& rx = config.network.receive;
            rx.enabled = true; rx.ingress_bytes = capacity;
            rx.drain_bps = rate; rx.ud_pps_per_qp = pps;
            if (fluid == 0) {
                rx.service.mode = RnicRxServiceMode::SerializedPacketBeats;
                rx.service.beat_bytes = width; rx.service.period_ps = 5000;
                rx.service.phase_ps = 2500; rx.service.token_capacity_bytes = 4224;
            }
            RnicDeviceAttachments attachments; attachments.network_port = &port_;
            native_ = std::make_unique<RnicDevice>(config, attachments);
            native_trace_.open(trace_path_);
        } else {
            rnic_cm_profile profile{};
            require(rnic_cm_profile_preset("cx5_100g", &profile) == RNIC_CM_OK,
                    "profile construction");
            profile.rx_ingress_bytes = capacity; profile.rx_drain_bps = rate;
            profile.rx_pps_per_qp_rc = 0; profile.rx_pps_per_qp_ud = pps;
            profile.rx_pps_per_nic = 0;
            rnic_cm_config config{};
            config.version = SIMLLM_RNIC_CM_ABI_VERSION;
            config.qpn = 7; config.policy_context_token = 1;
            config.sq_depth = 16; config.cq_depth = 32;
            config.packetization = 1; config.receive = 1; config.trace_enabled = 1;
            rnic_cm_rx_service_config service{};
            service.version = SIMLLM_RNIC_CM_RX_SERVICE_VERSION;
            if (fluid == 0) {
                service.mode = RNIC_CM_RX_SERVICE_SERIALIZED_PACKET_BEATS;
                service.beat_bytes = width; service.period_ps = 5000;
                service.phase_ps = 2500; service.token_capacity_bytes = 4224;
            }
            handle_ = fluid == 1 ? rnic_cm_create(&profile, &config)
                : rnic_cm_create_with_rx_service(&profile, &config, &service);
            require(handle_ != nullptr, "facade construction");
        }
    }
    ~Receiver() {
        if (handle_) {
            rnic_cm_trace(handle_, trace_path_.string().c_str());
            rnic_cm_destroy(handle_);
        }
    }
    bool arrive(U64 bytes, U64 now, unsigned* outcome = nullptr,
                unsigned qpn = 1, bool rc = false, unsigned psn = 0) {
        if (facade_) {
            rnic_cm_packet p{};
            p.kind = RNIC_CM_PACKET_DATA; p.service = rc ? RNIC_CM_SERVICE_RC : RNIC_CM_SERVICE_UD;
            p.qpn = qpn; p.psn = psn; p.wire_bytes = bytes; p.payload_bytes = bytes;
            rnic_cm_rx_result result{};
            const auto status = rnic_cm_rx_packet(handle_, &p, now, &result);
            require(status == RNIC_CM_OK || status == RNIC_CM_ERROR_STATE,
                    "facade arrival uses the declared state-error translation");
            result_bytes_.write(reinterpret_cast<const char*>(&result), sizeof(result));
            if (outcome) *outcome = result.outcome;
            return status == RNIC_CM_OK;
        }
        RnicRxPacket p;
        p.qpn = qpn; p.psn = psn; p.wire_bytes = bytes; p.payload_bytes = bytes;
        p.service = rc ? RnicTransportService::ReliableConnected : RnicTransportService::Unreliable;
        try {
            const auto result = native_->onReceivedPacket(p, now);
            native_trace_ << now << " arrival " << bytes << " outcome "
                          << static_cast<unsigned>(result.outcome) << " occupancy "
                          << result.ingress_occupancy_bytes << '\n';
            rnic_cm_rx_result projection{};
            projection.outcome = static_cast<unsigned>(result.outcome);
            projection.has_reply = result.has_reply;
            projection.reply_kind = result.reply_kind == NetworkPacketKind::Nak
                ? RNIC_CM_PACKET_NAK : RNIC_CM_PACKET_ACK;
            projection.reply_psn = result.reply_psn;
            projection.reply_wire_bytes = result.reply_wire_bytes;
            projection.ingress_occupancy_bytes = result.ingress_occupancy_bytes;
            projection.has_cnp = result.has_cnp;
            projection.cnp_source = result.cnp_source; projection.cnp_qpn = result.cnp_qpn;
            result_bytes_.write(reinterpret_cast<const char*>(&projection), sizeof(projection));
            if (outcome) *outcome = static_cast<unsigned>(result.outcome);
            return true;
        } catch (const std::exception&) { return false; }
    }
    bool progress(U64 now) {
        if (facade_) {
            const auto status = rnic_cm_progress(handle_, now, nullptr);
            require(status == RNIC_CM_OK || status == RNIC_CM_ERROR_STATE,
                    "facade progress uses the declared state-error translation");
            return status == RNIC_CM_OK;
        }
        try {
            native_->progress(now);
            native_trace_ << now << " progress occupancy " << snapshot().occupancy << '\n';
            return true;
        } catch (const std::exception&) { return false; }
    }
    std::optional<U64> next(bool* error = nullptr) const {
        if (error) *error = false;
        if (facade_) {
            U64 time = 0;
            const auto status = rnic_cm_next_event_ps(handle_, &time);
            if (status == RNIC_CM_OK) return time;
            require(status == RNIC_CM_NO_EVENT || status == RNIC_CM_ERROR_STATE,
                    "facade query uses NO_EVENT or the declared state error");
            if (status != RNIC_CM_NO_EVENT && error) *error = true;
            return {};
        }
        try { return native_->nextEventTime(); }
        catch (const std::exception&) { if (error) *error = true; return {}; }
    }
    Snapshot snapshot() const {
        if (facade_) {
            rnic_cm_nic_counter_set c{};
            require(rnic_cm_nic_counters(handle_, &c) == RNIC_CM_OK, "counter read");
            return {c.rx_ingress_occupancy_bytes, c.rx_packets_offered, c.rx_bytes_phy,
                    c.rx_packets_delivered, c.rx_discards_meter, c.rx_discards_rate,
                    c.rx_discards_sequence, c.duplicate_request};
        }
        const auto& c = native_->rxPipeline()->counters();
        return {c.ingress_occupancy_bytes, c.packets_offered, c.wire_bytes_offered,
                c.packets_delivered, c.packets_discarded_meter, c.packets_discarded_rate,
                c.packets_discarded_sequence, c.packets_discarded_duplicate};
    }
    void invariants() const {
        if (native_) {
            native_->validateInvariants();
            require(native_->hasPendingPhysicalWork() == native_->rxPipeline()->hasPendingService(),
                    "candidate must not schedule physical work");
        }
        const auto c = snapshot();
        require(c.offered == c.delivered + c.meter + c.rate + c.sequence + c.duplicate,
                "public offered-packet partition");
    }
    void writeFluidBytes(const std::string& name) const {
        std::ofstream stream(output / name, std::ios::binary);
        if (handle_) {
            rnic_cm_nic_counter_set nic{}; rnic_cm_counter_set counters{};
            require(rnic_cm_nic_counters(handle_, &nic) == RNIC_CM_OK
                && rnic_cm_counters(handle_, &counters) == RNIC_CM_OK, "fluid counters");
            stream.write(reinterpret_cast<const char*>(&nic), sizeof(nic));
            stream.write(reinterpret_cast<const char*>(&counters), sizeof(counters));
        } else {
            const auto& counters = native_->rxPipeline()->counters();
            const auto nic = native_->nicCounters();
            stream.write(reinterpret_cast<const char*>(&counters), sizeof(counters));
            stream.write(reinterpret_cast<const char*>(&nic), sizeof(nic));
        }
    }
private:
    bool facade_;
    EmptyPort port_;
    std::unique_ptr<RnicDevice> native_;
    rnic_cm_device* handle_{nullptr};
    std::filesystem::path trace_path_;
    std::ofstream native_trace_;
    std::ofstream result_bytes_;
};
void row(const std::string& kind, bool facade, U64 width, U64 rate,
         U64 completed = 0, U64 events = 0, const std::string& times = "",
         const std::string& bytes = "") {
    std::cout << kind << ',' << (facade ? "facade" : "native") << ',' << fixture
              << ',' << width << ',' << rate << ',' << completed << ',' << events
              << ',' << times << ',' << bytes << ",PASS\n";
}
void exact(bool facade, U64 width, U64 arrival, U64 deadline, bool off_grid) {
    Receiver rx(facade, width);
    require(rx.arrive(21, 2500), "head admission");
    std::vector<U64> times, bytes;
    auto debit = [&] (U64 now) {
        const auto before = rx.snapshot().occupancy;
        require(rx.progress(now), "scheduled progress");
        const auto after = rx.snapshot().occupancy;
        if (after < before) { times.push_back(now); bytes.push_back(before - after); }
    };
    if (off_grid) { debit(7500); debit(7500); debit(7501); }
    else {
        for (unsigned guard = 0; rx.next() && *rx.next() <= arrival; ++guard) {
            require(guard < 4, "bounded pre-arrival walk"); debit(*rx.next());
        }
    }
    require(rx.arrive(64, arrival), "successor admission");
    for (unsigned guard = 0; rx.next(); ++guard) {
        require(guard < 4, "bounded service walk"); debit(*rx.next());
    }
    require(times == std::vector<U64>({7500, deadline})
            && bytes == std::vector<U64>({21, 64}), "literal debit times and useful bytes");
    rx.invariants();
    row("exact", facade, width, 96600000000ULL, times.back(), times.size(),
        std::to_string(times[0]) + ";" + std::to_string(times[1]), "21;64");
}
void matrix(bool facade, U64 width, U64 rate, U64 deadline, U64 beats) {
    Receiver rx(facade, width, rate);
    require(rx.arrive(21, 2500), "component head");
    const auto before = rx.snapshot().occupancy;
    require(rx.arrive(85, 7500), "component successor");
    const auto admitted_debit = before + 85 - rx.snapshot().occupancy;
    require(admitted_debit == 21, "component observes exactly21B before admission");
    U64 completion = 7500, events = admitted_debit > 0 ? 1 : 0;
    for (unsigned guard = 0; rx.next(); ++guard) {
        require(guard < 6, "bounded component walk");
        completion = *rx.next(); require(rx.progress(completion), "component progress"); ++events;
    }
    rx.invariants();
    require(completion == deadline && events == beats, "literal width/rate component row");
    row("component", facade, width, rate, completion, events);
}
void controls(bool facade, U64 width, const std::string& name) {
    if (name == "default_fluid_identity") {
        for (const unsigned fluid : {1U, 2U}) {
            Receiver rx(facade, width, 96600000000ULL, 262016, 0, fluid,
                        "_" + std::to_string(fluid));
            for (unsigned i = 0; i < 20; ++i)
                require(rx.arrive(4096, i * 2500, nullptr, 1, true, i / 2), "fluid arrival");
            require(rx.progress(1000000), "fluid progress"); rx.invariants();
            rx.writeFluidBytes(std::string(facade ? "facade" : "native")
                               + "_fluid" + std::to_string(fluid) + "_w"
                               + std::to_string(width) + ".bin");
        }
    } else if (name == "public_input_identity") {
        // Every stimulus here is a literal packet or configuration value.
        // The runner additionally pins and audits the receiver source inputs.
        Receiver rx(facade, width); rx.invariants();
    } else if (name == "zero_progress") {
        Receiver rx(facade, width);
        require(rx.progress(0) && rx.progress(0) && !rx.next()
                && rx.snapshot().offered == 0 && rx.snapshot().occupancy == 0, "empty zero progress");
        rx.invariants();
    } else if (name == "direct_vs_event_walk") {
        std::optional<Snapshot> reference;
        for (unsigned variant = 0; variant < 3; ++variant) {
            Receiver rx(facade, width, 96600000000ULL, 262016, 0, 0,
                        "_" + std::to_string(variant));
            require(rx.arrive(21, 2500), "order head");
            if (variant == 1) require(rx.progress(7500), "pre-admission progress");
            unsigned outcome = 99;
            require(rx.arrive(64, 7500, &outcome) && outcome == RNIC_CM_RX_DELIVERED,
                    "same-time successor outcome");
            if (variant == 2) require(rx.progress(7500), "post-admission progress");
            require(rx.next() == 12500 && rx.snapshot().occupancy == 64, "same order deadline");
            if (reference) require(rx.snapshot() == *reference, "ordering snapshot identity");
            else reference = rx.snapshot();
            rx.invariants();
        }
    } else if (name == "partial_terminal_horizon" || name == "fully_drained_terminal_horizon") {
        const auto maximum = std::numeric_limits<U64>::max();
        const auto last = maximum - (maximum - 2500) % 5000;
        Receiver rx(facade, width, 160000000000ULL);
        const bool partial = name == "partial_terminal_horizon" && width == 64;
        require(rx.arrive(name == "partial_terminal_horizon" ? 85 : 21, last - 1), "terminal arrival");
        require(rx.next() == last, "representable terminal clock");
        if (partial) {
            for (unsigned repeat = 0; repeat < 2; ++repeat) {
                require(!rx.progress(last) && rx.snapshot().occupancy == 21, "partial error retains21B");
                bool error = false; require(!rx.next(&error) && error, "partial remains pending error");
            }
        } else {
            require(rx.progress(last) && !rx.next(), "fully drained terminal quiescence");
            const auto before = rx.snapshot();
            require(!rx.arrive(64, last) && rx.snapshot() == before, "terminal rejection before counters");
        }
        rx.invariants();
    } else if (name == "admitted_sequence_rejection") {
        for (const unsigned psn : {0U, 3U}) {
            Receiver rx(facade, width, 96600000000ULL, 262016, 0, 0,
                        "_" + std::to_string(psn));
            require(rx.arrive(21, 2500, nullptr, 1, true), "RC head");
            unsigned outcome = 99;
            require(rx.arrive(64, 7500, &outcome, 1, true, psn)
                    && outcome == (psn == 0 ? RNIC_CM_RX_DISCARDED_DUPLICATE
                                            : RNIC_CM_RX_DISCARDED_OUT_OF_SEQUENCE), "RC disposition");
            require(rx.snapshot().occupancy == 64 && rx.next() == 12500, "RC rejection stays charged");
            require(rx.progress(12500), "RC charged progress"); rx.invariants();
        }
    } else if (name == "invalid_wire_reject_only" || name == "capacity_reject_only"
               || name == "rate_reject_only" || name == "reject_then_valid_exact_time") {
        for (unsigned variant = 0; variant < 3; ++variant) {
            if (name != "reject_then_valid_exact_time"
                && variant != (name == "invalid_wire_reject_only" ? 0U
                    : name == "capacity_reject_only" ? 1U : 2U)) continue;
            Receiver rx(facade, width, 96600000000ULL, variant == 1 ? 64 : 262016,
                        variant == 2 ? 1 : 0, 0, "_" + std::to_string(variant));
            require(rx.arrive(21, 2500), "rejection head");
            unsigned outcome = 99;
            const bool accepted = rx.arrive(variant == 0 ? 0 : variant == 1 ? 65 : 64, 7500, &outcome);
            require(variant == 0 ? !accepted : accepted && outcome == RNIC_CM_RX_DISCARDED_SILENTLY,
                    "rejected offer verdict");
            rx.invariants();
            require(!rx.next() && rx.snapshot().occupancy == 0, "rejection schedules no work");
            require(rx.snapshot().offered == (variant == 0 ? 1 : 2)
                    && rx.snapshot().meter == (variant == 1 ? 1 : 0)
                    && rx.snapshot().rate == (variant == 2 ? 1 : 0), "exact rejection counters");
            const auto same = name == "reject_then_valid_exact_time";
            require(rx.arrive(64, same ? 7500 : 12500, nullptr, 2), "subsequent valid admission");
            require(rx.next() == (same ? 12500 : 22500), "rejection never activates live credit");
            require(rx.progress(same ? 12500 : 22500), "valid successor service"); rx.invariants();
        }
    } else {
        Receiver rx(facade, width);
        require(rx.arrive(21, 2500) && rx.next() == 7500 && rx.next() == 7500, "head read-only queries");
        require(rx.progress(7500) && rx.progress(7500) && !rx.next() && !rx.next(), "empty repeated progress");
        rx.invariants();
        if (name == "regressed_time") {
            require(rx.progress(7501) && !rx.arrive(64, 7500), "regression rejected");
            require(rx.snapshot().offered == 1 && !rx.next(), "regression cannot revive candidate");
        } else if (name == "same_time_candidate_is_not_work") {
            require(rx.progress(7501) && !rx.next() && rx.snapshot().occupancy == 0, "positive empty expiry");
        } else {
            require(name == "repeat_query_progress", "unknown fixture");
            require(rx.arrive(64, 7500) && rx.progress(7500) && rx.progress(7500), "repeat successor progress");
            require(rx.snapshot().occupancy == 64 && rx.next() == 12500, "only head debits same clock");
        }
        rx.invariants();
    }
    row("fatal", facade, width, 96600000000ULL);
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) throw std::runtime_error("usage: rx_atomic_successor_probe OUTPUT [FIXTURE]");
        output = argv[1]; std::filesystem::create_directories(output);
        const std::string selected = argc == 3 ? argv[2] : "";
        std::ofstream abi(output / "abi.txt");
        abi << SIMLLM_RNIC_CM_ABI_VERSION << '\n' << sizeof(rnic_cm_config) << '\n'
            << offsetof(rnic_cm_config, receive) << '\n' << offsetof(rnic_cm_config, mtu_bytes) << '\n'
            << sizeof(rnic_cm_rx_result) << '\n' << offsetof(rnic_cm_rx_result, ingress_occupancy_bytes)
            << '\n' << sizeof(rnic_cm_profile) << '\n' << sizeof(rnic_cm_packet) << '\n'
            << sizeof(rnic_cm_counter_set) << '\n' << sizeof(rnic_cm_nic_counter_set) << '\n';
        std::cout << "class,frontend,name,width_bytes,rate_bps,completed_ps,service_debits,debit_times_ps,debit_bytes,verdict\n";
        const std::vector<std::string> exact_names = {"exact_replacement", "already_queued", "one_ps_before",
            "one_ps_after", "actual_idle_successor", "positive_off_grid_progress", "large_idle_jump"};
        const std::vector<U64> arrivals = {7500, 2500, 7499, 7501, 12500, 7501, 52500};
        const std::vector<U64> deadlines = {12500, 12500, 12500, 17500, 22500, 17500, 62500};
        const std::vector<std::string> fatal_names = {"zero_progress", "repeat_query_progress",
            "direct_vs_event_walk", "invalid_wire_reject_only", "capacity_reject_only", "rate_reject_only",
            "reject_then_valid_exact_time", "admitted_sequence_rejection", "regressed_time",
            "partial_terminal_horizon", "fully_drained_terminal_horizon", "default_fluid_identity",
            "public_input_identity", "same_time_candidate_is_not_work"};
        bool reached = false;
        for (const bool facade : {false, true}) for (const auto width : {64ULL, 128ULL}) {
            for (std::size_t i = 0; i < exact_names.size(); ++i) {
                fixture = exact_names[i];
                if (!selected.empty() && selected != fixture) continue;
                reached = true; exact(facade, width, arrivals[i], deadlines[i], i == 5);
            }
            for (const auto& name : fatal_names) {
                fixture = name;
                if (!selected.empty() && selected != fixture) continue;
                reached = true; controls(facade, width, name);
            }
            if (selected.empty()) for (const auto rate : {40000000000ULL, 96600000000ULL, 160000000000ULL}) {
                fixture = "width_rate";
                matrix(facade, width, rate, rate == 40000000000ULL ? 27500 : width == 64 ? 17500 : 12500,
                       width == 64 ? 3 : 2);
            }
        }
        require(reached, "selected fixture not found");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << fixture << ": " << error.what() << '\n';
        return 1;
    }
}
