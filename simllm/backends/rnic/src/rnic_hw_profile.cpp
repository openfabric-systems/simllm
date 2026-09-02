#include "simllm/rnic/rnic_hw_profile.h"

#include <limits>
#include <stdexcept>
#include <string>

#include "simllm/rnic/session_record.h"

namespace simllm::rnic {
namespace {

void appendUnsigned(
    std::string& out,
    const char* key,
    std::uint64_t value,
    EvidenceClass evidence) {
    out += "\"";
    out += key;
    out += "\":{\"v\":";
    out += std::to_string(value);
    out += ",\"e\":\"";
    out += toString(evidence);
    out += "\"},";
}

void appendBool(
    std::string& out,
    const char* key,
    bool value,
    EvidenceClass evidence) {
    out += "\"";
    out += key;
    out += "\":{\"v\":";
    out += value ? "true" : "false";
    out += ",\"e\":\"";
    out += toString(evidence);
    out += "\"},";
}

void appendText(
    std::string& out,
    const char* key,
    const char* value,
    EvidenceClass evidence) {
    out += "\"";
    out += key;
    out += "\":{\"v\":\"";
    out += value;
    out += "\",\"e\":\"";
    out += toString(evidence);
    out += "\"},";
}

void appendPlainText(std::string& out, const char* key, const char* value) {
    out += "\"";
    out += key;
    out += "\":\"";
    out += value;
    out += "\",";
}

}  // namespace

const char* toString(EvidenceClass evidence) noexcept {
    switch (evidence) {
    case EvidenceClass::Documented:
        return "documented";
    case EvidenceClass::DriverInferred:
        return "driver-inferred";
    case EvidenceClass::CalibratedOpaque:
        return "calibrated-opaque";
    case EvidenceClass::Declared:
        return "declared";
    }
    return "invalid";
}

const char* toString(RnicRecoveryMode recovery) noexcept {
    switch (recovery) {
    case RnicRecoveryMode::GoBackN:
        return "go-back-n";
    case RnicRecoveryMode::SelectiveRepeat:
        return "selective-repeat";
    }
    return "invalid";
}

const char* toString(RnicEcnStamp stamp) noexcept {
    switch (stamp) {
    case RnicEcnStamp::NotEct:
        return "not-ect";
    case RnicEcnStamp::Ect0:
        return "ect0";
    case RnicEcnStamp::Ect1:
        return "ect1";
    }
    return "invalid";
}

const char* toString(RnicFirmwareCounterVariant variant) noexcept {
    switch (variant) {
    case RnicFirmwareCounterVariant::Fw1632:
        return "fw_16_32";
    case RnicFirmwareCounterVariant::Fw1631:
        return "fw_16_31";
    }
    return "invalid";
}

void validateRnicHwProfile(const RnicHwProfile& profile) {
    if (profile.version != kRnicHwProfileVersion) {
        throw std::invalid_argument("unsupported RNIC hardware profile version");
    }
    if (profile.name == nullptr || profile.name[0] == '\0') {
        throw std::invalid_argument("RNIC hardware profile needs a name");
    }
    if (profile.derived_link_factor != 0
        && (profile.derived_from == nullptr
            || profile.derived_from[0] == '\0')) {
        throw std::invalid_argument(
            "derived RNIC hardware profile must name its base");
    }
    if (profile.link_bps == 0 || profile.goodput_bps == 0
        || profile.goodput_bps > profile.link_bps) {
        throw std::invalid_argument(
            "RNIC hardware profile needs a positive goodput at or below the link");
    }
    if (profile.mtu_bytes == 0 || profile.wire_header_bytes == 0) {
        throw std::invalid_argument(
            "RNIC hardware profile needs a positive MTU and wire header");
    }
    if (profile.mtu_bytes
        > std::numeric_limits<std::uint64_t>::max() / 8
            - profile.wire_header_bytes) {
        throw std::out_of_range("RNIC hardware profile MTU overflows");
    }
    const Picoseconds stages = profile.doorbell_service_ps
        + profile.wqe_fetch_service_ps + profile.qpc_lookup_service_ps
        + profile.scheduler_service_ps + profile.cqe_write_service_ps;
    if (profile.wire_round_trip_floor_ps > profile.t_eff_ps
        || stages + profile.wire_round_trip_floor_ps != profile.t_eff_ps) {
        throw std::invalid_argument(
            "RNIC hardware profile stage split must reconstruct t_eff_ps "
            "together with the wire round-trip floor");
    }
    if (profile.sq_depth == 0) {
        throw std::invalid_argument(
            "RNIC hardware profile needs a positive send-queue depth");
    }
}

std::uint64_t effectiveWireBps(const RnicHwProfile& profile) {
    validateRnicHwProfile(profile);
    const std::uint64_t wire_unit =
        profile.mtu_bytes + profile.wire_header_bytes;
    if (profile.goodput_bps > std::numeric_limits<std::uint64_t>::max()
            / wire_unit) {
        throw std::out_of_range("RNIC effective wire rate overflows");
    }
    return profile.goodput_bps * wire_unit / profile.mtu_bytes;
}

std::uint64_t packetsForMessage(
    const RnicHwProfile& profile,
    std::uint64_t payload_bytes) {
    if (profile.mtu_bytes == 0) {
        throw std::invalid_argument("RNIC hardware profile needs a positive MTU");
    }
    if (payload_bytes == 0) {
        return 1;
    }
    return (payload_bytes + profile.mtu_bytes - 1) / profile.mtu_bytes;
}

std::string renderRnicHwProfileJson(const RnicHwProfile& profile) {
    validateRnicHwProfile(profile);
    std::string out = "{\"schema\":\"";
    out += kRnicHwProfileSchema;
    out += "\",\"version\":";
    out += std::to_string(profile.version);
    out += ",";
    appendPlainText(out, "name", profile.name);
    appendPlainText(
        out,
        "derived_from",
        profile.derived_from == nullptr ? "" : profile.derived_from);
    out += "\"derived_link_factor\":";
    out += std::to_string(profile.derived_link_factor);
    out += ",";

    const RnicHwProfileEvidence& e = profile.evidence;
    appendUnsigned(out, "link_bps", profile.link_bps, e.link_bps);
    appendUnsigned(out, "goodput_bps", profile.goodput_bps, e.goodput_bps);
    appendUnsigned(out, "mtu_bytes", profile.mtu_bytes, e.mtu_bytes);
    appendUnsigned(
        out, "wire_header_bytes", profile.wire_header_bytes,
        e.wire_header_bytes);

    appendUnsigned(out, "t_eff_ps", profile.t_eff_ps, e.t_eff_ps);
    appendUnsigned(
        out, "wire_round_trip_floor_ps", profile.wire_round_trip_floor_ps,
        e.wire_round_trip_floor_ps);
    appendUnsigned(
        out, "doorbell_service_ps", profile.doorbell_service_ps,
        e.doorbell_service_ps);
    appendUnsigned(
        out, "wqe_fetch_service_ps", profile.wqe_fetch_service_ps,
        e.wqe_fetch_service_ps);
    appendUnsigned(
        out, "qpc_lookup_service_ps", profile.qpc_lookup_service_ps,
        e.qpc_lookup_service_ps);
    appendUnsigned(
        out, "scheduler_service_ps", profile.scheduler_service_ps,
        e.scheduler_service_ps);
    appendUnsigned(
        out, "cqe_write_service_ps", profile.cqe_write_service_ps,
        e.cqe_write_service_ps);

    appendUnsigned(out, "sq_depth", profile.sq_depth, e.sq_depth);
    appendUnsigned(
        out, "max_inflight_bytes", profile.max_inflight_bytes,
        e.max_inflight_bytes);
    appendUnsigned(
        out, "max_inflight_packets", profile.max_inflight_packets,
        e.max_inflight_packets);

    appendUnsigned(out, "tx_pps_per_qp", profile.tx_pps_per_qp, e.tx_pps_per_qp);
    appendUnsigned(
        out, "tx_pps_per_nic", profile.tx_pps_per_nic, e.tx_pps_per_nic);
    appendUnsigned(
        out, "rx_pps_per_qp_rc", profile.rx_pps_per_qp_rc, e.rx_pps_per_qp_rc);
    appendUnsigned(
        out, "rx_pps_per_qp_ud", profile.rx_pps_per_qp_ud, e.rx_pps_per_qp_ud);
    appendUnsigned(
        out, "rx_pps_per_nic", profile.rx_pps_per_nic, e.rx_pps_per_nic);

    appendUnsigned(
        out, "rx_ingress_bytes", profile.rx_ingress_bytes, e.rx_ingress_bytes);
    appendUnsigned(out, "rx_drain_bps", profile.rx_drain_bps, e.rx_drain_bps);
    appendUnsigned(
        out, "internal_budget_bps", profile.internal_budget_bps,
        e.internal_budget_bps);
    appendBool(
        out, "loopback_priority", profile.loopback_priority,
        e.loopback_priority);

    appendText(out, "recovery", toString(profile.recovery), e.recovery);
    appendUnsigned(
        out, "selective_repeat_window", profile.selective_repeat_window,
        e.selective_repeat_window);
    appendUnsigned(out, "rto_ps", profile.rto_ps, e.rto_ps);
    appendBool(out, "ack_coalescing", profile.ack_coalescing, e.ack_coalescing);

    appendBool(out, "dcqcn_enabled", profile.dcqcn_enabled, e.dcqcn_enabled);
    appendText(out, "ecn_stamp", toString(profile.ecn_stamp), e.ecn_stamp);
    appendUnsigned(
        out, "cnp_min_interval_ps", profile.cnp_min_interval_ps,
        e.cnp_min_interval_ps);
    appendUnsigned(
        out, "dcqcn_alpha_update_ps", profile.dcqcn_alpha_update_ps,
        e.dcqcn_alpha_update_ps);
    appendUnsigned(
        out, "dcqcn_rate_reduce_ps", profile.dcqcn_rate_reduce_ps,
        e.dcqcn_rate_reduce_ps);
    appendUnsigned(
        out, "dcqcn_byte_reset", profile.dcqcn_byte_reset, e.dcqcn_byte_reset);
    appendUnsigned(
        out, "dcqcn_rate_step_bps", profile.dcqcn_rate_step_bps,
        e.dcqcn_rate_step_bps);
    appendUnsigned(
        out, "np_cnp_threshold_bytes", profile.np_cnp_threshold_bytes,
        e.np_cnp_threshold_bytes);
    appendUnsigned(
        out, "dcqcn_alpha_init_ppm", profile.dcqcn_alpha_init_ppm,
        e.dcqcn_alpha_init_ppm);
    appendUnsigned(
        out, "dcqcn_alpha_gain_ppm", profile.dcqcn_alpha_gain_ppm,
        e.dcqcn_alpha_gain_ppm);
    appendUnsigned(
        out, "dcqcn_rate_increase_step_bps",
        profile.dcqcn_rate_increase_step_bps,
        e.dcqcn_rate_increase_step_bps);
    appendUnsigned(
        out, "dcqcn_rate_increase_interval_ps",
        profile.dcqcn_rate_increase_interval_ps,
        e.dcqcn_rate_increase_interval_ps);
    appendUnsigned(
        out, "dcqcn_rate_floor_bps", profile.dcqcn_rate_floor_bps,
        e.dcqcn_rate_floor_bps);

    appendBool(out, "pfc_enabled", profile.pfc_enabled, e.pfc_enabled);
    appendBool(out, "global_pause_tx", profile.global_pause_tx, e.global_pause_tx);
    appendBool(
        out, "pause_propagates", profile.pause_propagates, e.pause_propagates);

    appendText(
        out, "firmware_counter_variant",
        toString(profile.firmware_counter_variant), e.firmware_counter_variant);

    if (!out.empty() && out.back() == ',') {
        out.pop_back();
    }
    out += "}";
    return out;
}

std::string rnicHwProfileSha256(const RnicHwProfile& profile) {
    return rnicSha256Hex(renderRnicHwProfileJson(profile));
}

RnicHwProfileRecord makeRnicHwProfileRecord(const RnicHwProfile& profile) {
    RnicHwProfileRecord record;
    record.version = profile.version;
    record.schema = kRnicHwProfileSchema;
    record.name = profile.name;
    record.profile_json = renderRnicHwProfileJson(profile);
    record.sha256 = rnicSha256Hex(record.profile_json);
    return record;
}

bool findRnicHwProfile(const std::string& name, RnicHwProfile* out) {
    if (out == nullptr) {
        return false;
    }
    if (name == kConnectX5_100G.name) {
        *out = kConnectX5_100G;
        return true;
    }
    if (name == kConnectX7_400G.name) {
        *out = kConnectX7_400G;
        return true;
    }
    return false;
}

}  // namespace simllm::rnic
