#ifndef SIMLLM_RNIC_RNIC_TX_PIPELINE_H
#define SIMLLM_RNIC_RNIC_TX_PIPELINE_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <vector>

#include "simllm/rnic/network_port.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicTxPipelineConfigVersion = 1;

// The transmit pipeline sits between the work queue and the network port. The
// queue keeps submitting one flow extent per WQE, exactly as it does on ABI
// v1; the pipeline segments that extent into MTU-sized packets, bounds the
// outstanding work, paces the issue of each packet, and translates the port's
// per-packet events back into the packet-attempt and flow-extent events the
// queue's timeline is built from.
//
// Downstream packet-port contract: the port returns one token per accepted
// attempt and later reports, for that token, `PacketTxFinished`,
// `PacketRxArrived` and one terminal (`Delivered` or `Dropped`) with ABI v2
// packet-attempt scope. The pipeline is the transmit authority, so it stamps
// the TX start itself at the paced issue instant; a port must not emit one.
struct RnicTxPipelineConfig {
    std::uint32_t version{kRnicTxPipelineConfigVersion};
    bool enabled{false};

    // Packetizer.
    std::uint64_t mtu_bytes{4096};
    std::uint64_t wire_header_bytes{64};
    std::uint32_t initial_psn{0};

    // Outstanding-work window. Zero disables that bound. A WQE is in flight
    // from the issue of its first packet to the terminal of its last, so the
    // window is what is on the wire, not what the send queue holds.
    std::uint64_t max_inflight_wqes{0};
    std::uint64_t max_inflight_bytes{0};
    std::uint64_t max_inflight_packets{0};

    // Pacer. Zero disables that ceiling. The bit rate is charged on wire
    // bytes, so it must be the effective wire rate at which a full-MTU packet
    // delivers the profile's goodput.
    std::uint64_t wire_bps_per_qp{0};
    std::uint64_t wire_bps_per_nic{0};
    // The measured small-message ceiling is a host-bound message rate: it is
    // charged once per work request, not once per wire packet. At or below
    // the MTU the two readings coincide, which is where it was measured.
    std::uint64_t message_rate_per_qp{0};
    std::uint64_t message_rate_per_nic{0};
};

struct RnicTxPipelineCounters {
    std::uint64_t extents_accepted{0};
    std::uint64_t extents_completed{0};
    std::uint64_t packets_issued{0};
    std::uint64_t packets_delivered{0};
    std::uint64_t packets_dropped{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes{0};
    std::uint64_t window_stalls{0};
    std::uint64_t pacer_stalls{0};
    std::uint64_t downstream_busy{0};
    std::uint64_t downstream_rejected{0};
    // A release the caller forced later than the instant the pipeline had
    // already announced through nextEventTime(). It stays zero for a caller
    // that steps to the times it is given, and a study treats a nonzero value
    // as a voided run rather than a measurement.
    std::uint64_t late_releases{0};
    std::uint64_t inflight_wqes{0};
    std::uint64_t inflight_bytes{0};
    std::uint64_t inflight_packets{0};
};

void validateRnicTxPipelineConfig(const RnicTxPipelineConfig& config);

class RnicTxPipeline final : public NetworkPort {
public:
    RnicTxPipeline(RnicTxPipelineConfig config, NetworkPort& downstream);

    RnicTxPipeline(const RnicTxPipeline&) = delete;
    RnicTxPipeline& operator=(const RnicTxPipeline&) = delete;

    NetworkPortCapabilities capabilities() const noexcept override;

    // Accepts one flow extent per WQE. The pipeline never returns Busy: the
    // outstanding-work window gates packet issue, not admission, so an
    // accepted extent is one the scheduler has handed to the transmit path
    // and its first packet issue is a separate, paced instant.
    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override;

    // Issues every packet whose paced instant has arrived and returns the
    // upstream events the work queue must observe, in order.
    std::vector<NetworkEvent> releaseDue(Picoseconds now_ps);

    // Translates one downstream packet event into the upstream events for its
    // attempt, plus the extent terminal when the last packet retires.
    std::vector<NetworkEvent> onDownstreamEvent(const NetworkEvent& event);

    std::optional<Picoseconds> nextEventTime() const;
    bool hasPendingWork() const noexcept;

    const RnicTxPipelineConfig& config() const noexcept;
    const RnicTxPipelineCounters& counters() const noexcept;
    void validateInvariants() const;

private:
    // An exact rational rate gate. The remainder carries the division's
    // fractional picoseconds forward, so a long run has bounded error instead
    // of accumulating one truncation per packet.
    struct RateGate {
        std::uint64_t rate{0};
        Picoseconds free_at_ps{0};
        std::uint64_t remainder{0};

        Picoseconds delayFor(std::uint64_t units);
    };

    struct Packet {
        NetworkToken attempt_token{0};
        std::uint64_t packet_index{0};
        std::uint64_t payload_offset_bytes{0};
        std::uint64_t payload_bytes{0};
        std::uint64_t wire_bytes{0};
        std::uint32_t psn{0};
        bool issued{false};
        bool terminal{false};
        NetworkEvent started;
    };

    struct Extent {
        NetworkToken extent_token{0};
        WqeId wqe_id{0};
        NetworkTxDescriptor descriptor;
        std::vector<Packet> packets;
        std::size_t issued_count{0};
        std::size_t terminal_count{0};
        bool in_flight{false};
        bool dropped{false};
        DropLocation drop_location{DropLocation::None};
        DropReason drop_reason{DropReason::None};
        DropEvidenceProvenance drop_evidence{DropEvidenceProvenance::None};
        std::uint64_t drop_resource_id{0};
    };

    struct QueueEntry {
        NetworkToken extent_token{0};
        std::size_t packet_index{0};
        Picoseconds queued_at_ps{0};
    };

    Picoseconds eligibleAt(const QueueEntry& entry) const;
    bool windowAllows(const QueueEntry& entry) const;
    void retirePacket(Extent& extent, Packet& packet);

    RnicTxPipelineConfig config_;
    NetworkPort& downstream_;
    NetworkToken next_token_{1};
    std::uint32_t next_psn_{0};
    Picoseconds last_now_ps_{0};
    Picoseconds window_open_ps_{0};
    std::optional<Picoseconds> downstream_retry_at_ps_;
    RateGate qp_bits_;
    RateGate nic_bits_;
    RateGate qp_messages_;
    RateGate nic_messages_;
    std::map<NetworkToken, Extent> extents_;
    std::map<NetworkToken, std::pair<NetworkToken, std::size_t>>
        downstream_tokens_;
    std::deque<QueueEntry> queue_;
    RnicTxPipelineCounters counters_;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_TX_PIPELINE_H
