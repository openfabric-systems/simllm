#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "simllm/rnic/rnic_cmodel_c.h"

namespace {
constexpr std::uint64_t denominator = 8000000000000ULL;
constexpr std::uint64_t period = 5000;
constexpr std::uint64_t phase = 2500;
std::uint64_t ceilRatio(std::uint64_t n, std::uint64_t d) {
    return n / d + (n % d == 0 ? 0 : 1);
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// An independent cumulative-budget bound, with a one-clock exclusion for each
// packet-aligned beat. No receiver occupancy, credit or next-event output is an
// input. For these constant-backlog rows, a 4224B cap cannot delay this bound:
// a capped clock already affords any beat, while a budget-limited clock never
// accumulates 4224B. The queue empties only after the final beat.
std::uint64_t expectedTick(std::uint64_t width, std::uint64_t length,
                           std::uint64_t count, std::uint64_t rate) {
    std::uint64_t tick = 0, prefix = 0;
    for (std::uint64_t p = 0; p < count; ++p) {
        for (std::uint64_t offset = 0; offset < length; offset += width) {
            prefix += std::min(width, length - offset);
            tick = std::max(tick + 1, ceilRatio(prefix * denominator, rate * period));
        }
    }
    return tick;
}
void row(std::uint64_t width, std::uint64_t rate, std::uint64_t length,
         std::uint64_t capacity) {
    rnic_cm_profile profile{};
    require(rnic_cm_profile_preset("cx5_100g", &profile) == RNIC_CM_OK, "preset failed");
    profile.rx_ingress_bytes = capacity;
    profile.rx_drain_bps = rate;
    profile.rx_pps_per_qp_rc = 0; profile.rx_pps_per_qp_ud = 0; profile.rx_pps_per_nic = 0;
    rnic_cm_config config{};
    config.version = SIMLLM_RNIC_CM_ABI_VERSION;
    config.qpn = 7; config.policy_context_token = 1;
    config.sq_depth = 16; config.cq_depth = 32;
    config.packetization = 1; config.receive = 1;
    rnic_cm_rx_service_config service{};
    service.version = SIMLLM_RNIC_CM_RX_SERVICE_VERSION;
    service.mode = RNIC_CM_RX_SERVICE_SERIALIZED_PACKET_BEATS;
    service.beat_bytes = width; service.period_ps = period; service.phase_ps = phase;
    service.token_capacity_bytes = 4224;
    auto* device = rnic_cm_create_with_rx_service(&profile, &config, &service);
    require(device != nullptr, "serialized construction failed");
    const auto count = std::min<std::uint64_t>(128, capacity / length);
    rnic_cm_packet packet{};
    packet.kind = RNIC_CM_PACKET_DATA; packet.service = RNIC_CM_SERVICE_UD;
    packet.qpn = 7; packet.wire_bytes = length; packet.payload_bytes = length;
    for (std::uint64_t p = 0; p < count; ++p) {
        rnic_cm_rx_result result{};
        require(rnic_cm_rx_packet(device, &packet, phase, &result) == RNIC_CM_OK
                && result.outcome == RNIC_CM_RX_DELIVERED, "unexpected admission/drop");
        require(result.ingress_occupancy_bytes == (p + 1) * length, "arrival conservation failed");
    }
    std::uint64_t completed_at = phase, events = 0;
    rnic_cm_nic_counter_set counters{};
    int status = RNIC_CM_OK;
    while ((status = rnic_cm_next_event_ps(device, &completed_at)) == RNIC_CM_OK) {
        std::uint64_t changes = 0;
        require(rnic_cm_progress(device, completed_at, &changes) == RNIC_CM_OK, "progress failed");
        require(++events <= count * ceilRatio(length, width), "unbounded service events");
    }
    require(status == RNIC_CM_NO_EVENT, "service event query failed before quiescence");
    require(events == count * ceilRatio(length, width), "packet beat event conservation failed");
    require(rnic_cm_nic_counters(device, &counters) == RNIC_CM_OK, "counter read failed");
    const auto expected_at = phase + expectedTick(width, length, count, rate) * period;
    require(completed_at == expected_at, "cumulative-budget deadline mismatch");
    require(counters.rx_ingress_occupancy_bytes == 0
            && counters.rx_ingress_high_watermark_bytes == count * length
            && counters.rx_packets_delivered == count && counters.rx_discards_meter == 0,
            "selected occupancy/admission conservation failed");
    const auto geometric_bps = length * denominator / (period * ceilRatio(length, width));
    std::cout << width << ',' << rate << ',' << length << ',' << capacity << ',' << count
              << ',' << completed_at << ',' << expected_at << ',' << events << ','
              << geometric_bps << ',' << counters.rx_ingress_high_watermark_bytes
              << ',' << counters.rx_discards_meter << ",PASS\n";
    rnic_cm_destroy(device);
}
}  // namespace
int main() {
    try {
        std::cout << "width_bytes,rate_bps,length_bytes,capacity_bytes,packets,completed_ps,"
                     "expected_ps,service_events,geometric_bps,high_water_bytes,meter_drops,verdict\n";
        for (const auto width : {64ULL, 128ULL})
            for (const auto rate : {40000000000ULL, 96600000000ULL, 160000000000ULL})
                for (const auto length : {64ULL, 85ULL, 4096ULL})
                    for (const auto capacity : {262016ULL, 16384ULL})
                        row(width, rate, length, capacity);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
