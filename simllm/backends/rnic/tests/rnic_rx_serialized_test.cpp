#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "simllm/rnic/rnic_cmodel_c.h"
#include "simllm/rnic/rnic_rx_pipeline.h"

namespace {
using namespace simllm::rnic;
std::size_t failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}
template <class Callable>
void rejects(Callable call, const char* message) {
    try { call(); check(false, message); }
    catch (const std::exception&) {}
}
RnicRxPipelineConfig configuration(std::uint64_t width = 64,
                                 std::uint64_t rate = 96600000000ULL,
                                 std::uint64_t capacity = 262016) {
    RnicRxPipelineConfig config;
    config.enabled = true;
    config.ingress_bytes = capacity;
    config.drain_bps = rate;
    config.service.mode = RnicRxServiceMode::SerializedPacketBeats;
    config.service.beat_bytes = width;
    config.service.period_ps = 5000;
    config.service.phase_ps = 2500;
    config.service.token_capacity_bytes = 4224;
    return config;
}
RnicRxPacket packet(std::uint64_t bytes) {
    RnicRxPacket p;
    p.wire_bytes = bytes;
    p.payload_bytes = bytes;
    p.service = RnicTransportService::Unreliable;
    return p;
}
void scheduling() {
    RnicRxPipeline rx(configuration());
    rx.onPacket(packet(64), 2500);
    check(rx.nextServiceTime() == 12500, "60.375B/clock needs two clocks for 64B");
    rx.progress(7500);
    check(rx.ingressOccupancyBytes() == 64, "unaffordable full beat is indivisible");
    rx.progress(12500);
    check(rx.ingressOccupancyBytes() == 0, "second refill funds first beat");
    rx.onPacket(packet(64), 52500);
    check(rx.nextServiceTime() == 62500, "empty queue clears idle credit");
    rx.progress(62500);
    rx.validateInvariants();

    RnicRxPipeline start(configuration(64, 160000000000ULL));
    start.onPacket(packet(85), 2500);
    start.onPacket(packet(85), 2500);
    start.progress(7500);
    check(start.ingressOccupancyBytes() == 106, "first of two 85B packets debits 64B");
    start.progress(12500);
    check(start.ingressOccupancyBytes() == 85, "21B tail owns its clock, successor waits");
    start.onPacket(packet(64), 12500);
    check(start.ingressOccupancyBytes() == 149, "debit precedes same-clock arrival");
    check(start.nextServiceTime() == 17500, "no second beat at same timestamp");
    start.progress(1000000);
    start.validateInvariants();

    auto phase_config = configuration(128, 160000000000ULL);
    phase_config.service.phase_ps = 0;
    RnicRxPipeline phase(phase_config);
    phase.onPacket(packet(85), 0);
    check(phase.nextServiceTime() == 5000, "arrival at clock zero waits following clock");
    phase.progress(5000);
    check(phase.ingressOccupancyBytes() == 0, "128B beat drains whole 85B packet");

    auto cap_config = configuration(64, 160000000000ULL);
    cap_config.service.token_capacity_bytes = 64;
    RnicRxPipeline capped(cap_config);
    capped.onPacket(packet(128), 2500);
    capped.progress(7500);
    check(capped.ingressOccupancyBytes() == 64, "finite cap cannot fund multiple beats");
    capped.progress(12500);
    capped.validateInvariants();

    auto binding_cap = configuration(128, 160000000000ULL);
    binding_cap.service.token_capacity_bytes = 128;
    RnicRxPipeline clamped(binding_cap);
    clamped.onPacket(packet(4096), 2500);
    clamped.progress(12500);
    check(clamped.ingressOccupancyBytes() == 3968 && clamped.nextServiceTime() == 22500,
          "128B cap spills72B at first debit, requiring two new100B refills");
    clamped.progress(317500);
    check(clamped.ingressOccupancyBytes() == 128, "binding cap final beat still waits");
    clamped.progress(322500);
    check(clamped.ingressOccupancyBytes() == 0, "binding cap yields exactly64 service clocks");
    clamped.validateInvariants();

}
void capacityAndProcessing() {
    // Exactly 16384B, with an 85B head. The first useful debit is 64B or 85B.
    // A new 85B packet therefore fits only the wider serialized geometry.
    for (const auto width : {64ULL, 128ULL}) {
        RnicRxPipeline rx(configuration(width, 160000000000ULL, 16384));
        for (unsigned i = 0; i < 64; ++i) rx.onPacket(packet(85), 2500);
        for (unsigned i = 0; i < 171; ++i) rx.onPacket(packet(64), 2500);
        check(rx.ingressOccupancyBytes() == 16384, "capacity equality admits");
        const auto result = rx.onPacket(packet(85), 7500);
        check((result.outcome == RnicRxOutcome::Delivered) == (width == 128),
              "selected geometry controls capacity admission");
        check(rx.counters().packets_discarded_meter == (width == 64 ? 1 : 0),
              "selected capacity rejection has exact meter counter");
        check(rx.counters().ingress_high_watermark_bytes == 16384,
              "selected occupancy high-water never exceeds capacity");
        rx.progress(10000000);
        rx.validateInvariants();
    }
    auto fluid_config = configuration(64, 160000000000ULL, 16384);
    fluid_config.service = {};
    RnicRxPipeline fluid(fluid_config);
    for (unsigned i = 0; i < 64; ++i) fluid.onPacket(packet(85), 2500);
    for (unsigned i = 0; i < 171; ++i) fluid.onPacket(packet(64), 2500);
    check(fluid.onPacket(packet(85), 7500).outcome == RnicRxOutcome::Delivered,
          "fluid admission differs from 64B serialized capacity edge");

    RnicRxPipeline rc(configuration());
    auto p = packet(85);
    p.service = RnicTransportService::ReliableConnected;
    check(rc.onPacket(p, 2500).outcome == RnicRxOutcome::Delivered, "RC delivery retained");
    check(rc.onPacket(p, 2500).outcome == RnicRxOutcome::DiscardedDuplicate,
          "duplicate remains charged after sequence rejection");
    p.psn = 3;
    check(rc.onPacket(p, 2500).outcome == RnicRxOutcome::DiscardedOutOfSequence,
          "sequence reject remains charged");
    check(rc.ingressOccupancyBytes() == 255, "all three metered records remain queued");
    rc.progress(10000000);
    rc.validateInvariants();
    auto rate_config = configuration();
    rate_config.ud_pps_per_qp = 1;
    RnicRxPipeline rate(rate_config);
    rate.onPacket(packet(85), 2500);
    check(rate.onPacket(packet(85), 2500).outcome == RnicRxOutcome::DiscardedSilently,
          "packet-rate rejection retains existing semantics");
    check(rate.ingressOccupancyBytes() == 85, "rate rejection does not enqueue service");
    rate.validateInvariants();

    auto notify_config = configuration(64, 160000000000ULL, 128);
    notify_config.notification.enabled = true;
    notify_config.notification.threshold_bytes = 100;
    RnicRxPipeline notify(notify_config);
    notify.onPacket(packet(128), 2500);
    const auto result = notify.onPacket(packet(85), 7500);
    check(result.has_cnp && result.outcome == RnicRxOutcome::DiscardedSilently,
          "selected occupancy notifies even on capacity reject");
    notify.validateInvariants();
}
void arithmeticAndInvalid() {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    auto invalid = configuration();
    invalid.service.version = 2;
    rejects([&] { RnicRxPipeline rx(invalid); }, "unsupported version rejected");
    invalid = configuration(); invalid.service.mode = static_cast<RnicRxServiceMode>(2);
    rejects([&] { RnicRxPipeline rx(invalid); }, "unknown mode rejected");
    invalid = configuration(); invalid.service.ready = static_cast<RnicRxReadySemantics>(1);
    rejects([&] { RnicRxPipeline rx(invalid); }, "unsupported readiness rejected");
    invalid = configuration(); invalid.service.beat_bytes = 0;
    rejects([&] { RnicRxPipeline rx(invalid); }, "zero width rejected");
    invalid = configuration(); invalid.service.period_ps = 0;
    rejects([&] { RnicRxPipeline rx(invalid); }, "zero period rejected");
    invalid = configuration(); invalid.service.phase_ps = 5000;
    rejects([&] { RnicRxPipeline rx(invalid); }, "phase outside clock rejected");
    invalid = configuration(); invalid.service.token_capacity_bytes = 63;
    rejects([&] { RnicRxPipeline rx(invalid); }, "cap below full beat rejected");
    invalid = configuration(); invalid.service.token_capacity_bytes = maximum;
    rejects([&] { RnicRxPipeline rx(invalid); }, "token numerator overflow rejected");
    invalid = configuration(); invalid.drain_bps = maximum;
    rejects([&] { RnicRxPipeline rx(invalid); }, "clock refill overflow rejected");
    invalid = configuration(); invalid.drain_bps = 0;
    rejects([&] { RnicRxPipeline rx(invalid); }, "zero rate rejected");
    invalid = configuration(); invalid.service.mode = RnicRxServiceMode::Fluid;
    rejects([&] { RnicRxPipeline rx(invalid); }, "fluid cannot silently absorb geometry");

    RnicRxPipeline slow(configuration(64, 1));
    slow.onPacket(packet(64), 2500);
    check(slow.nextServiceTime() == 512000000002500ULL, "exact low-rate lookahead skips clocks");
    slow.progress(maximum);
    check(slow.ingressOccupancyBytes() == 0, "unbounded idle leap finishes finite backlog");
    slow.validateInvariants();
    rejects([&] { slow.onPacket(packet(64), maximum); }, "no future clock fails explicitly");
    RnicRxPipeline near_end(configuration());
    near_end.onPacket(packet(64), maximum - 5000);
    rejects([&] { near_end.nextServiceTime(); }, "out-of-range service fails, never quiesces");
    rejects([&] { near_end.progress(maximum); }, "progress fails on unaffordable terminal horizon");
    near_end.validateInvariants();
    rejects([&] { near_end.progress(0); }, "time regression remains rejected");
    const auto last_clock = maximum - (maximum - 2500) % 5000;
    RnicRxPipeline partial(configuration(64, 160000000000ULL));
    partial.onPacket(packet(85), last_clock - 1);
    check(partial.nextServiceTime() == last_clock, "last affordable clock is representable");
    rejects([&] { partial.progress(last_clock); }, "partial final beat reaches terminal horizon");
    check(partial.ingressOccupancyBytes() == 21
              && partial.counters().ingress_occupancy_bytes == 21,
          "successful final debit stays visible when successor clock overflows");
    partial.validateInvariants();

}
rnic_cm_config facadeConfiguration() {
    rnic_cm_config cfg{};
    cfg.version = SIMLLM_RNIC_CM_ABI_VERSION;
    cfg.qpn = 7; cfg.policy_context_token = 1; cfg.sq_depth = 16; cfg.cq_depth = 32;
    cfg.packetization = 1; cfg.receive = 1; cfg.trace_enabled = 1;
    return cfg;
}
void facade(const std::string& output) {
    static_assert(SIMLLM_RNIC_CM_ABI_VERSION == 1);
    static_assert(sizeof(rnic_cm_config) == 72);
    static_assert(offsetof(rnic_cm_config, receive) == 42);
    static_assert(offsetof(rnic_cm_config, mtu_bytes) == 64);
    static_assert(sizeof(rnic_cm_rx_result) == 48);
    static_assert(offsetof(rnic_cm_rx_result, ingress_occupancy_bytes) == 24);
    static_assert(sizeof(rnic_cm_rx_service_config) == 48);
    rnic_cm_profile profile{};
    check(rnic_cm_profile_preset("cx5_100g", &profile) == RNIC_CM_OK, "preset available");
    auto cfg = facadeConfiguration();
    rnic_cm_rx_service_config service{};
    service.version = SIMLLM_RNIC_CM_RX_SERVICE_VERSION;
    service.mode = RNIC_CM_RX_SERVICE_SERIALIZED_PACKET_BEATS;
    service.beat_bytes = 64; service.period_ps = 5000; service.phase_ps = 2500;
    service.token_capacity_bytes = 4224;
    auto invalid = service;
    invalid.reserved0 = 1;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "reserved input rejected");
    invalid = service; invalid.version = 2;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade version rejected");
    invalid = service; invalid.mode = 2;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade mode rejected");
    invalid = service; invalid.ready_semantics = 1;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade ready rejected");
    invalid = service; invalid.beat_bytes = 0;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade geometry rejected");
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, nullptr), "null opt-in rejected");
    invalid = service; invalid.period_ps = 0;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade zero period rejected");
    invalid = service; invalid.phase_ps = 5000;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade phase rejected");
    invalid = service; invalid.token_capacity_bytes = std::numeric_limits<std::uint64_t>::max();
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade capacity overflow rejected");
    invalid = service; invalid.mode = RNIC_CM_RX_SERVICE_FLUID;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &invalid), "facade fluid geometry rejected");
    cfg.receive = 0;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &service), "disabled receive rejected");
    cfg.receive = 1; cfg.packetization = 0;
    check(!rnic_cm_create_with_rx_service(&profile, &cfg, &service), "nonpacketized opt-in rejected");
    cfg.packetization = 1;
    auto* device = rnic_cm_create_with_rx_service(&profile, &cfg, &service);
    check(device != nullptr, "explicit serialized facade constructs");
    if (device) {
        rnic_cm_packet p{};
        p.kind = RNIC_CM_PACKET_DATA; p.service = RNIC_CM_SERVICE_UD;
        p.qpn = 7; p.wire_bytes = 64; p.payload_bytes = 64;
        rnic_cm_rx_result result{};
        check(rnic_cm_rx_packet(device, &p, 2500, &result) == RNIC_CM_OK, "facade accepts receive");
        std::uint64_t event = 0, changes = 0;
        check(rnic_cm_next_event_ps(device, &event) == RNIC_CM_OK && event == 12500,
              "device announces receive-only service event");
        check(rnic_cm_progress(device, event, &changes) == RNIC_CM_OK, "device advances receive event");
        rnic_cm_nic_counter_set counters{};
        rnic_cm_nic_counters(device, &counters);
        check(counters.rx_ingress_occupancy_bytes == 0, "facade selected occupancy progresses");
        check(rnic_cm_next_event_ps(device, &event) == RNIC_CM_NO_EVENT, "empty service has no event");
        rnic_cm_destroy(device);
    }
    auto* horizon = rnic_cm_create_with_rx_service(&profile, &cfg, &service);
    check(horizon != nullptr, "horizon fixture constructs");
    if (horizon) {
        rnic_cm_packet p{};
        p.kind = RNIC_CM_PACKET_DATA; p.service = RNIC_CM_SERVICE_UD;
        p.qpn = 7; p.wire_bytes = 64; p.payload_bytes = 64;
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t event = 0, changes = 0;
        check(rnic_cm_rx_packet(horizon, &p, maximum - 5000, nullptr) == RNIC_CM_OK,
              "terminal horizon retains accepted receive bytes");
        check(rnic_cm_next_event_ps(horizon, &event) == RNIC_CM_ERROR_STATE,
              "pending unrepresentable service reports ERROR_STATE, never NO_EVENT");
        check(rnic_cm_progress(horizon, maximum, &changes) == RNIC_CM_ERROR_STATE,
              "facade terminal horizon progress fails explicitly");
        rnic_cm_destroy(horizon);
    }
    auto fast_profile = profile;
    fast_profile.rx_drain_bps = 160000000000ULL;
    auto* partial = rnic_cm_create_with_rx_service(&fast_profile, &cfg, &service);
    check(partial != nullptr, "partial horizon facade constructs");
    if (partial) {
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        const auto last_clock = maximum - (maximum - 2500) % 5000;
        rnic_cm_packet p{};
        p.kind = RNIC_CM_PACKET_DATA; p.service = RNIC_CM_SERVICE_UD;
        p.qpn = 7; p.wire_bytes = 85; p.payload_bytes = 85;
        std::uint64_t changes = 0, event = 0;
        check(rnic_cm_rx_packet(partial, &p, last_clock - 1, nullptr) == RNIC_CM_OK,
              "partial horizon accepts public packet");
        check(rnic_cm_progress(partial, last_clock, &changes) == RNIC_CM_ERROR_STATE,
              "partial horizon error crosses facade explicitly");
        rnic_cm_nic_counter_set counters{};
        check(rnic_cm_nic_counters(partial, &counters) == RNIC_CM_OK
                  && counters.rx_ingress_occupancy_bytes == 21,
              "facade exposes exactly21B retained after successful final64B debit");
        check(rnic_cm_next_event_ps(partial, &event) == RNIC_CM_ERROR_STATE,
              "partial horizon remains a pending error, never NO_EVENT");
        rnic_cm_destroy(partial);
    }
    // Explicit fluid opt-in must not add a trace line or alter any counter.
    rnic_cm_rx_service_config fluid{};
    fluid.version = SIMLLM_RNIC_CM_RX_SERVICE_VERSION;
    auto* old = rnic_cm_create(&profile, &cfg);
    auto* explicit_fluid = rnic_cm_create_with_rx_service(&profile, &cfg, &fluid);
    check(old && explicit_fluid, "both fluid constructors available");
    if (old && explicit_fluid) {
        for (unsigned i = 0; i < 20; ++i) {
            rnic_cm_packet p{};
            p.kind = RNIC_CM_PACKET_DATA; p.service = RNIC_CM_SERVICE_RC;
            p.qpn = 7; p.psn = i / 2; p.wire_bytes = 4096; p.payload_bytes = 4032;
            rnic_cm_rx_result a{}, b{};
            check(rnic_cm_rx_packet(old, &p, i * 2500, &a) == RNIC_CM_OK
                  && rnic_cm_rx_packet(explicit_fluid, &p, i * 2500, &b) == RNIC_CM_OK
                  && std::memcmp(&a, &b, sizeof(a)) == 0, "fluid result bytes identical");
        }
        rnic_cm_nic_counter_set a{}, b{};
        rnic_cm_nic_counters(old, &a); rnic_cm_nic_counters(explicit_fluid, &b);
        check(std::memcmp(&a, &b, sizeof(a)) == 0, "fluid counter bytes identical");
        const auto legacy_path = (std::filesystem::path(output) / "legacy_fluid.trace").string();
        const auto explicit_path = (std::filesystem::path(output) / "explicit_fluid.trace").string();
        check(rnic_cm_trace(old, legacy_path.c_str()) == RNIC_CM_OK
              && rnic_cm_trace(explicit_fluid, explicit_path.c_str()) == RNIC_CM_OK,
              "fluid traces written");
        std::ifstream a_stream(legacy_path), b_stream(explicit_path);
        const std::string a_text((std::istreambuf_iterator<char>(a_stream)), {});
        const std::string b_text((std::istreambuf_iterator<char>(b_stream)), {});
        check(a_text == b_text && !a_text.empty(), "fluid trace bytes identical");
    }
    rnic_cm_destroy(old); rnic_cm_destroy(explicit_fluid);
}
}  // namespace
int main(int argc, char** argv) {
    try {
        scheduling(); capacityAndProcessing(); arithmeticAndInvalid();
        facade(argc == 2 ? argv[1] : ".");
    } catch (const std::exception& error) {
        check(false, error.what());
    }
    std::cout << "serialized receive fixtures: " << (failures == 0 ? "PASS" : "FAIL")
              << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
