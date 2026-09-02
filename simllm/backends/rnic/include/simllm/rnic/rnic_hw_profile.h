#ifndef SIMLLM_RNIC_RNIC_HW_PROFILE_H
#define SIMLLM_RNIC_RNIC_HW_PROFILE_H

#include <cstdint>
#include <string>

#include "simllm/rnic/network_port.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicHwProfileVersion = 1;

// The profile is its own versioned record with its own schema string and its
// own hash. It is deliberately not mixed into the effective-hardware schemas
// or their hash inputs: those identify a composed device so a policy
// comparison cannot silently change hardware, and this identifies the
// hardware parameter set a golden-model run was calibrated with.
inline constexpr const char* kRnicHwProfileSchema =
    "simllm-rnic-hw-profile-v1";

// How a field's value was established. `declared` covers both an unmeasurable
// value and one derived by scaling another profile.
enum class EvidenceClass : std::uint8_t {
    Documented,
    DriverInferred,
    CalibratedOpaque,
    Declared,
};

enum class RnicRecoveryMode : std::uint8_t {
    GoBackN,
    SelectiveRepeat,
};

// The NIC forces one ECN codepoint on every RoCEv2 transmit regardless of the
// requested ToS, so the stamp is a hardware property, not a caller choice.
enum class RnicEcnStamp : std::uint8_t {
    NotEct,
    Ect0,
    Ect1,
};

// The two firmware revisions in the campaign disagree about
// local_ack_timeout_err, so counter semantics are part of the profile.
enum class RnicFirmwareCounterVariant : std::uint8_t {
    Fw1632,
    Fw1631,
};

const char* toString(EvidenceClass evidence) noexcept;
const char* toString(RnicRecoveryMode recovery) noexcept;
const char* toString(RnicEcnStamp stamp) noexcept;
const char* toString(RnicFirmwareCounterVariant variant) noexcept;

// One evidence class per profile field, mirroring the value struct field for
// field. A scaled field is `Declared` in the derived profile even when the
// base field was measured.
struct RnicHwProfileEvidence {
    EvidenceClass link_bps{EvidenceClass::Declared};
    EvidenceClass goodput_bps{EvidenceClass::Declared};
    EvidenceClass mtu_bytes{EvidenceClass::Declared};
    EvidenceClass wire_header_bytes{EvidenceClass::Declared};

    EvidenceClass t_eff_ps{EvidenceClass::Declared};
    EvidenceClass wire_round_trip_floor_ps{EvidenceClass::Declared};
    EvidenceClass doorbell_service_ps{EvidenceClass::Declared};
    EvidenceClass wqe_fetch_service_ps{EvidenceClass::Declared};
    EvidenceClass qpc_lookup_service_ps{EvidenceClass::Declared};
    EvidenceClass scheduler_service_ps{EvidenceClass::Declared};
    EvidenceClass cqe_write_service_ps{EvidenceClass::Declared};

    EvidenceClass sq_depth{EvidenceClass::Declared};
    EvidenceClass max_inflight_bytes{EvidenceClass::Declared};
    EvidenceClass max_inflight_packets{EvidenceClass::Declared};

    EvidenceClass tx_pps_per_qp{EvidenceClass::Declared};
    EvidenceClass tx_pps_per_nic{EvidenceClass::Declared};
    EvidenceClass rx_pps_per_qp_rc{EvidenceClass::Declared};
    EvidenceClass rx_pps_per_qp_ud{EvidenceClass::Declared};
    EvidenceClass rx_pps_per_nic{EvidenceClass::Declared};

    EvidenceClass rx_ingress_bytes{EvidenceClass::Declared};
    EvidenceClass rx_drain_bps{EvidenceClass::Declared};
    EvidenceClass internal_budget_bps{EvidenceClass::Declared};
    EvidenceClass loopback_priority{EvidenceClass::Declared};

    EvidenceClass recovery{EvidenceClass::Declared};
    EvidenceClass selective_repeat_window{EvidenceClass::Declared};
    EvidenceClass rto_ps{EvidenceClass::Declared};
    EvidenceClass ack_coalescing{EvidenceClass::Declared};

    EvidenceClass dcqcn_enabled{EvidenceClass::Declared};
    EvidenceClass ecn_stamp{EvidenceClass::Declared};
    EvidenceClass cnp_min_interval_ps{EvidenceClass::Declared};
    EvidenceClass dcqcn_alpha_update_ps{EvidenceClass::Declared};
    EvidenceClass dcqcn_rate_reduce_ps{EvidenceClass::Declared};
    EvidenceClass dcqcn_byte_reset{EvidenceClass::Declared};
    EvidenceClass dcqcn_rate_step_bps{EvidenceClass::Declared};
    EvidenceClass np_cnp_threshold_bytes{EvidenceClass::Declared};
    EvidenceClass dcqcn_alpha_init_ppm{EvidenceClass::Declared};
    EvidenceClass dcqcn_alpha_gain_ppm{EvidenceClass::Declared};
    EvidenceClass dcqcn_rate_increase_step_bps{EvidenceClass::Declared};
    EvidenceClass dcqcn_rate_increase_interval_ps{EvidenceClass::Declared};
    EvidenceClass dcqcn_rate_floor_bps{EvidenceClass::Declared};

    EvidenceClass pfc_enabled{EvidenceClass::Declared};
    EvidenceClass global_pause_tx{EvidenceClass::Declared};
    EvidenceClass pause_propagates{EvidenceClass::Declared};

    EvidenceClass firmware_counter_variant{EvidenceClass::Declared};
};

// Every rate is an integer of bits per second and every packet rate an
// integer of packets per second, so rate scaling stays exact and a rendered
// profile has no floating-point spelling to disagree about.
struct RnicHwProfile {
    std::uint32_t version{kRnicHwProfileVersion};
    // Stable identity, and the base profile a derived one was scaled from.
    const char* name{""};
    const char* derived_from{""};
    std::uint32_t derived_link_factor{0};

    // Link.
    std::uint64_t link_bps{0};
    std::uint64_t goodput_bps{0};
    std::uint64_t mtu_bytes{0};
    std::uint64_t wire_header_bytes{0};

    // Initiation. The five work-queue service stages plus the modelled wire
    // round-trip floor sum to the lumped measured `t_eff_ps`: the campaign
    // fitted one fixed offset that already contains the round trip, so a
    // model that charges the round trip explicitly must not also charge it
    // inside the lump. The split across the five stages is declared.
    Picoseconds t_eff_ps{0};
    Picoseconds wire_round_trip_floor_ps{0};
    Picoseconds doorbell_service_ps{0};
    Picoseconds wqe_fetch_service_ps{0};
    Picoseconds qpc_lookup_service_ps{0};
    Picoseconds scheduler_service_ps{0};
    Picoseconds cqe_write_service_ps{0};

    // Outstanding work. Zero means the bound is the queue itself.
    std::uint64_t sq_depth{0};
    std::uint64_t max_inflight_bytes{0};
    std::uint64_t max_inflight_packets{0};

    // Packet rate. Zero means no ceiling was established.
    std::uint64_t tx_pps_per_qp{0};
    std::uint64_t tx_pps_per_nic{0};
    std::uint64_t rx_pps_per_qp_rc{0};
    std::uint64_t rx_pps_per_qp_ud{0};
    std::uint64_t rx_pps_per_nic{0};

    // Ingress. `loopback_priority` false means wire ingress wins the shared
    // internal budget, which is what the campaign measured.
    std::uint64_t rx_ingress_bytes{0};
    std::uint64_t rx_drain_bps{0};
    std::uint64_t internal_budget_bps{0};
    bool loopback_priority{false};

    // Transport.
    RnicRecoveryMode recovery{RnicRecoveryMode::GoBackN};
    std::uint64_t selective_repeat_window{0};
    Picoseconds rto_ps{0};
    bool ack_coalescing{false};

    // Congestion control.
    bool dcqcn_enabled{false};
    RnicEcnStamp ecn_stamp{RnicEcnStamp::NotEct};
    Picoseconds cnp_min_interval_ps{0};
    Picoseconds dcqcn_alpha_update_ps{0};
    Picoseconds dcqcn_rate_reduce_ps{0};
    std::uint64_t dcqcn_byte_reset{0};
    std::uint64_t dcqcn_rate_step_bps{0};
    // The notification point's own parameters. The threshold is the ingress
    // occupancy at or above which an arriving packet is treated as having
    // observed congestion; on this fabric nothing else can observe it, because
    // the switch never marks.
    std::uint64_t np_cnp_threshold_bytes{0};
    // The reaction point's alpha recursion, in parts per million.
    std::uint64_t dcqcn_alpha_init_ppm{0};
    std::uint64_t dcqcn_alpha_gain_ppm{0};
    // The additive increase, per queue pair. The campaign measured a recovery
    // that is linear in time rather than the fast-recovery-then-additive shape
    // the vendor defaults imply, so this is one step over one interval.
    std::uint64_t dcqcn_rate_increase_step_bps{0};
    Picoseconds dcqcn_rate_increase_interval_ps{0};
    std::uint64_t dcqcn_rate_floor_bps{0};

    // Flow control.
    bool pfc_enabled{false};
    bool global_pause_tx{false};
    bool pause_propagates{false};

    // Observable state.
    RnicFirmwareCounterVariant firmware_counter_variant{
        RnicFirmwareCounterVariant::Fw1632};

    RnicHwProfileEvidence evidence;
};

// The measured ConnectX-5 Ex 100 GbE profile. Every value comes from the
// mlx5 campaign records; the evidence class states how.
constexpr RnicHwProfile connectX5_100G() {
    RnicHwProfile profile;
    profile.name = "cx5_100g";

    profile.link_bps = 100000000000ULL;
    profile.goodput_bps = 97100000000ULL;
    profile.mtu_bytes = 4096;
    profile.wire_header_bytes = 64;
    profile.evidence.link_bps = EvidenceClass::Documented;
    profile.evidence.goodput_bps = EvidenceClass::CalibratedOpaque;
    profile.evidence.mtu_bytes = EvidenceClass::Documented;
    profile.evidence.wire_header_bytes = EvidenceClass::DriverInferred;

    profile.t_eff_ps = 4480000;
    profile.wire_round_trip_floor_ps = 2100000;
    profile.doorbell_service_ps = 40000;
    profile.wqe_fetch_service_ps = 40000;
    profile.qpc_lookup_service_ps = 2220000;
    profile.scheduler_service_ps = 40000;
    profile.cqe_write_service_ps = 40000;
    profile.evidence.t_eff_ps = EvidenceClass::CalibratedOpaque;
    profile.evidence.wire_round_trip_floor_ps =
        EvidenceClass::CalibratedOpaque;
    profile.evidence.doorbell_service_ps = EvidenceClass::Declared;
    profile.evidence.wqe_fetch_service_ps = EvidenceClass::Declared;
    profile.evidence.qpc_lookup_service_ps = EvidenceClass::Declared;
    profile.evidence.scheduler_service_ps = EvidenceClass::Declared;
    profile.evidence.cqe_write_service_ps = EvidenceClass::Declared;

    profile.sq_depth = 1024;
    profile.max_inflight_bytes = 0;
    profile.max_inflight_packets = 0;
    profile.evidence.sq_depth = EvidenceClass::DriverInferred;
    profile.evidence.max_inflight_bytes = EvidenceClass::Declared;
    profile.evidence.max_inflight_packets = EvidenceClass::Declared;

    profile.tx_pps_per_qp = 3870000;
    profile.tx_pps_per_nic = 16700000;
    profile.rx_pps_per_qp_rc = 0;
    // Post-specified correction from the P6 fabric campaign, measured after
    // the slice-C expectations were frozen. The earlier 3.07e6 came from the
    // Collie engine and was that engine's receive path, not the NIC: on the
    // wire one unreliable receive queue pair absorbed 5.51 Mpps of 2 KiB
    // datagrams with only the 0.17 to 0.19 percent ingress floor, and four
    // queue pairs together were slightly worse rather than better. 5.51e6 is
    // therefore the highest per-QP receive rate the silicon was shown to
    // absorb, not a rate at which it was shown to break: at 2 KiB that offer
    // is already 100 Gb/s of wire, and at 4 KiB the link binds first at 2.98
    // Mpps, so no 100 GbE probe can push one queue pair past it.
    profile.rx_pps_per_qp_ud = 5510000;
    // Kept at the measured multi-queue aggregate. No P6 wire point contradicts
    // it, because none could reach it: 9.65 Mpps needs payloads near 1 KiB to
    // fit a 100 GbE port at all, and the campaign's aggregate row is exactly
    // that size. It shares its instrument with the re-attributed per-QP row,
    // so it is retained as an unconfirmed ceiling and re-measuring it on the
    // wire is BACK-56's multi-QP clause.
    profile.rx_pps_per_nic = 9650000;
    profile.evidence.tx_pps_per_qp = EvidenceClass::CalibratedOpaque;
    profile.evidence.tx_pps_per_nic = EvidenceClass::CalibratedOpaque;
    profile.evidence.rx_pps_per_qp_rc = EvidenceClass::Declared;
    profile.evidence.rx_pps_per_qp_ud = EvidenceClass::CalibratedOpaque;
    profile.evidence.rx_pps_per_nic = EvidenceClass::CalibratedOpaque;

    profile.rx_ingress_bytes = 262016;
    // Fitted by the slice-C study against the two measured drain-window
    // thresholds and the measured saturated equilibrium, over a declared
    // candidate grid. It is a wire-bit rate: the meter drains headers as well
    // as payload.
    profile.rx_drain_bps = 96600000000ULL;
    profile.internal_budget_bps = 197000000000ULL;
    profile.loopback_priority = false;
    profile.evidence.rx_ingress_bytes = EvidenceClass::Documented;
    profile.evidence.rx_drain_bps = EvidenceClass::CalibratedOpaque;
    profile.evidence.internal_budget_bps = EvidenceClass::CalibratedOpaque;
    profile.evidence.loopback_priority = EvidenceClass::CalibratedOpaque;

    profile.recovery = RnicRecoveryMode::GoBackN;
    profile.selective_repeat_window = 0;
    profile.rto_ps = 67108864000ULL;
    profile.ack_coalescing = false;
    profile.evidence.recovery = EvidenceClass::Documented;
    profile.evidence.selective_repeat_window = EvidenceClass::Documented;
    profile.evidence.rto_ps = EvidenceClass::DriverInferred;
    profile.evidence.ack_coalescing = EvidenceClass::Declared;

    profile.dcqcn_enabled = true;
    profile.ecn_stamp = RnicEcnStamp::Ect0;
    // Fitted by the slice-D study over a declared candidate grid, and a
    // correction of the vendor default that stood here: the notification point
    // was measured raising 283 per second per congested queue pair, which is
    // one per 3.53 ms and seventy times slower than a 50 us limiter allows.
    profile.cnp_min_interval_ps = 3530000000;
    // Fitted with it. Alpha has to decay far enough between two notifications
    // for the loop to hold its measured operating point, and 50 us is what
    // does that at the fitted notification interval.
    profile.dcqcn_alpha_update_ps = 50000000;
    profile.dcqcn_rate_reduce_ps = 4000000;
    profile.dcqcn_byte_reset = 33554432;
    profile.dcqcn_rate_step_bps = 5000000000ULL;
    profile.evidence.dcqcn_enabled = EvidenceClass::CalibratedOpaque;
    profile.evidence.ecn_stamp = EvidenceClass::CalibratedOpaque;
    // The DCQCN parameter block is not readable on the campaign hosts: the
    // inbox driver exposes no ecn/ sysfs tree and firmware NV config needs
    // root, so every value here is opaque. The two the slice-D study could fit
    // against measured dynamics are fitted; the rest are the vendor 100 G
    // defaults, declared.
    profile.evidence.cnp_min_interval_ps = EvidenceClass::CalibratedOpaque;
    profile.evidence.dcqcn_alpha_update_ps = EvidenceClass::CalibratedOpaque;
    profile.evidence.dcqcn_rate_reduce_ps = EvidenceClass::Declared;
    profile.evidence.dcqcn_byte_reset = EvidenceClass::Declared;
    profile.evidence.dcqcn_rate_step_bps = EvidenceClass::Declared;
    // The notification and reaction parameters, fitted by the slice-D study
    // over declared candidate grids against the measured notification rate and
    // the measured transient. The threshold is half the ingress buffer, which
    // is where the notification rate lands closest to the measured 283 per
    // second per congested queue pair. The additive step is per queue pair, so
    // a sender of four of them recovers at four times it, which is the rate
    // the campaign measured at the host.
    profile.np_cnp_threshold_bytes = 131008;
    profile.dcqcn_alpha_init_ppm = 500000;
    profile.dcqcn_rate_increase_step_bps = 27500000;
    // The gain, the increase interval and the floor are not measurable from
    // outside and no cell of the study separates them, so they stay declared.
    profile.dcqcn_alpha_gain_ppm = 3906;
    profile.dcqcn_rate_increase_interval_ps = 1000000000;
    profile.dcqcn_rate_floor_bps = 1000000000;
    profile.evidence.np_cnp_threshold_bytes = EvidenceClass::CalibratedOpaque;
    profile.evidence.dcqcn_alpha_init_ppm = EvidenceClass::CalibratedOpaque;
    profile.evidence.dcqcn_rate_increase_step_bps =
        EvidenceClass::CalibratedOpaque;
    profile.evidence.dcqcn_alpha_gain_ppm = EvidenceClass::Declared;
    profile.evidence.dcqcn_rate_increase_interval_ps = EvidenceClass::Declared;
    profile.evidence.dcqcn_rate_floor_bps = EvidenceClass::Declared;

    profile.pfc_enabled = false;
    profile.global_pause_tx = true;
    profile.pause_propagates = false;
    profile.evidence.pfc_enabled = EvidenceClass::Documented;
    profile.evidence.global_pause_tx = EvidenceClass::Documented;
    profile.evidence.pause_propagates = EvidenceClass::CalibratedOpaque;

    profile.firmware_counter_variant = RnicFirmwareCounterVariant::Fw1632;
    profile.evidence.firmware_counter_variant = EvidenceClass::Documented;
    return profile;
}

// Derive a same-architecture profile at a higher line rate. Link, goodput,
// packet-rate and threshold fields scale; initiation, MTU, header, transport,
// outstanding-work and flow-control fields are kept. Every scaled field is
// marked `declared`, because scaling is a derivation, not a measurement.
constexpr RnicHwProfile scaleProfile(
    const RnicHwProfile& base,
    std::uint32_t link_factor) {
    RnicHwProfile scaled = base;
    scaled.derived_from = base.name;
    scaled.derived_link_factor = link_factor;
    scaled.name = "";

    const auto scale = [link_factor](std::uint64_t value) -> std::uint64_t {
        return value * link_factor;
    };
    scaled.link_bps = scale(base.link_bps);
    scaled.goodput_bps = scale(base.goodput_bps);
    scaled.tx_pps_per_qp = scale(base.tx_pps_per_qp);
    scaled.tx_pps_per_nic = scale(base.tx_pps_per_nic);
    scaled.rx_pps_per_qp_rc = scale(base.rx_pps_per_qp_rc);
    scaled.rx_pps_per_qp_ud = scale(base.rx_pps_per_qp_ud);
    scaled.rx_pps_per_nic = scale(base.rx_pps_per_nic);
    scaled.rx_ingress_bytes = scale(base.rx_ingress_bytes);
    scaled.rx_drain_bps = scale(base.rx_drain_bps);
    scaled.internal_budget_bps = scale(base.internal_budget_bps);
    scaled.dcqcn_byte_reset = scale(base.dcqcn_byte_reset);
    scaled.dcqcn_rate_step_bps = scale(base.dcqcn_rate_step_bps);
    scaled.np_cnp_threshold_bytes = scale(base.np_cnp_threshold_bytes);
    scaled.dcqcn_rate_increase_step_bps =
        scale(base.dcqcn_rate_increase_step_bps);
    scaled.dcqcn_rate_floor_bps = scale(base.dcqcn_rate_floor_bps);

    if (link_factor != 1) {
        scaled.evidence.link_bps = EvidenceClass::Declared;
        scaled.evidence.goodput_bps = EvidenceClass::Declared;
        scaled.evidence.tx_pps_per_qp = EvidenceClass::Declared;
        scaled.evidence.tx_pps_per_nic = EvidenceClass::Declared;
        scaled.evidence.rx_pps_per_qp_rc = EvidenceClass::Declared;
        scaled.evidence.rx_pps_per_qp_ud = EvidenceClass::Declared;
        scaled.evidence.rx_pps_per_nic = EvidenceClass::Declared;
        scaled.evidence.rx_ingress_bytes = EvidenceClass::Declared;
        scaled.evidence.rx_drain_bps = EvidenceClass::Declared;
        scaled.evidence.internal_budget_bps = EvidenceClass::Declared;
        scaled.evidence.dcqcn_byte_reset = EvidenceClass::Declared;
        scaled.evidence.dcqcn_rate_step_bps = EvidenceClass::Declared;
        scaled.evidence.np_cnp_threshold_bytes = EvidenceClass::Declared;
        scaled.evidence.dcqcn_rate_increase_step_bps = EvidenceClass::Declared;
        scaled.evidence.dcqcn_rate_floor_bps = EvidenceClass::Declared;
    }
    return scaled;
}

constexpr RnicHwProfile connectX7_400G() {
    RnicHwProfile profile = scaleProfile(connectX5_100G(), 4);
    profile.name = "cx7_400g";
    return profile;
}

inline constexpr RnicHwProfile kConnectX5_100G = connectX5_100G();
inline constexpr RnicHwProfile kConnectX7_400G = connectX7_400G();

// Rejects a profile whose fields cannot describe hardware: a zero link or
// goodput, a goodput above the link, a zero MTU, a stage split that does not
// reconstruct the lumped fixed offset, or a derived profile that names no
// base.
void validateRnicHwProfile(const RnicHwProfile& profile);

// The wire rate a full-MTU packet must be serialized at for its payload to
// arrive at exactly `goodput_bps`. This is where header bytes are paid, so a
// packetizer that charges them separately would charge them twice.
std::uint64_t effectiveWireBps(const RnicHwProfile& profile);

// The number of MTU-sized packets one message of `payload_bytes` becomes. A
// zero-byte message is still one packet.
std::uint64_t packetsForMessage(
    const RnicHwProfile& profile,
    std::uint64_t payload_bytes);

// The canonical schema-tagged JSON whose exact bytes are hashed. Field order
// is fixed, integers are decimal, and there is no floating point.
std::string renderRnicHwProfileJson(const RnicHwProfile& profile);
std::string rnicHwProfileSha256(const RnicHwProfile& profile);

struct RnicHwProfileRecord {
    std::uint32_t version{kRnicHwProfileVersion};
    std::string schema{kRnicHwProfileSchema};
    std::string name;
    std::string profile_json;
    std::string sha256;
};

RnicHwProfileRecord makeRnicHwProfileRecord(const RnicHwProfile& profile);

// Looks a preset up by its stable name, so a C caller or a study can name a
// profile without duplicating its constants.
bool findRnicHwProfile(const std::string& name, RnicHwProfile* out);

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_HW_PROFILE_H
