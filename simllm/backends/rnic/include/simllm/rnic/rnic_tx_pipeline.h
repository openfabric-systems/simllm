#ifndef SIMLLM_RNIC_RNIC_TX_PIPELINE_H
#define SIMLLM_RNIC_RNIC_TX_PIPELINE_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "simllm/rnic/network_port.h"
#include "simllm/rnic/rnic_cc.h"
#include "simllm/rnic/rnic_nic_counters.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicTxPipelineConfigVersion = 1;

// One inbound transport packet at the requester. The responder's ACK is
// carried by the downstream port's own delivery terminal, so only the
// negative acknowledgement needs its own path in.
struct RnicTransportPacket {
    NetworkPacketKind kind{NetworkPacketKind::Nak};
    std::uint32_t qpn{0};
    // For a NAK this is the sequence number the responder is waiting for, so
    // go-back-N restarts there. For an ACK it is the accepted number.
    std::uint32_t psn{0};
};

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

    // Requester transport. Off is the slice-B path exactly: a packet the port
    // reports dropped ends its extent with a transport error and nothing is
    // ever resent. On, the pipeline keeps per-QP sequence and acknowledgement
    // state and recovers by go-back-N, so a lost packet is replayed together
    // with every packet the responder threw away behind it.
    bool transport_enabled{false};
    // Zero disables the timer, which leaves a silently lost packet with no
    // recovery path at all. That is a test configuration, not a hardware one.
    Picoseconds rto_ps{0};
    // Firmware 16.31 counts a timeout-driven recovery on
    // `local_ack_timeout_err`; firmware 16.32 counts zero for the same
    // stimulus. False selects 16.32, which is the campaign's default node.
    bool counts_local_ack_timeout{false};

    // The congestion-control reaction point. Disabled is the identity default:
    // the pacer keeps its fixed ceilings, a congestion notification is refused
    // rather than silently absorbed, and every accepted slice-B and slice-C
    // row is unchanged. Enabled, the reaction point owns one more rate gate in
    // front of the pacer, and its state lives here rather than on a work
    // request, which is what makes it persist across them.
    RnicCcReactionConfig reaction;
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
    // Requester transport. All stay zero without it.
    std::uint64_t naks_received{0};
    std::uint64_t recovery_episodes{0};
    std::uint64_t packets_retransmitted{0};
    std::uint64_t timeouts{0};
    // Terminals that arrived for an attempt the transport had already
    // replayed. They close the work queue's lifecycle and change nothing
    // else, so they are the honest measure of go-back-N waste on the wire.
    std::uint64_t stale_terminals{0};
    // Reaction point. All stay zero without it, except the rate, which reads
    // zero because there is no rate gate rather than because it is stopped.
    std::uint64_t cnps_handled{0};
    std::uint64_t cnps_ignored{0};
    std::uint64_t rate_cuts{0};
    std::uint64_t rate_increases{0};
    std::uint64_t current_rate_bps{0};
    std::uint64_t min_rate_bps{0};
    std::uint64_t alpha_ppm{0};
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

    // Delivers one inbound transport packet to the requester. A NAK opens one
    // recovery episode at its sequence number: every attempt at or above it
    // that is still on the wire is closed as dropped and requeued, in
    // sequence order, ahead of anything newer. A second NAK inside the same
    // episode is absorbed, because the responder sends one per epoch and a
    // second recovery would replay the replay.
    std::vector<NetworkEvent> onTransportPacket(
        const RnicTransportPacket& packet,
        Picoseconds now_ps);

    // One congestion notification for this endpoint's queue pair. It cuts the
    // reaction point's rate multiplicatively and is never ignored: an endpoint
    // with no reaction point refuses the notification instead, so a caller
    // cannot read silence as a modelled reaction.
    void onCongestionNotification(Picoseconds now_ps);
    bool hasReactionPoint() const noexcept;
    // The rate the reaction point currently holds, or zero without one. A
    // study reads this across a work-request boundary to see that the state
    // persists.
    std::uint64_t reactionRateBps() const noexcept;

    std::optional<Picoseconds> nextEventTime() const;
    bool hasPendingWork() const noexcept;

    const RnicTxPipelineConfig& config() const noexcept;
    const RnicTxPipelineCounters& counters() const noexcept;
    const RnicNicCounters& nicCounters() const noexcept;
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
        std::uint32_t transmission_attempt{0};
        // The port token of the attempt currently on the wire, so a recovery
        // can retire it without searching every live binding.
        NetworkToken downstream_token{0};
        bool issued{false};
        bool terminal{false};
        Picoseconds issued_at_ps{0};
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

    // The issue queue is keyed by sequence number, not by arrival order. A
    // packetizer hands packets over in sequence order, so for a first
    // transmission the two are the same thing; a go-back-N replay is the case
    // where they are not, and putting a replayed number back where it belongs
    // is what keeps the responder in sequence. A deque with the replays
    // pushed onto the front would send a higher number before a lower one
    // that was still waiting, which the responder reads as a fresh loss.

    Picoseconds eligibleAt(const QueueEntry& entry) const;
    bool windowAllows(const QueueEntry& entry) const;
    void refreshReactionCounters();
    void retirePacket(Extent& extent, Packet& packet);
    // Closes every live attempt whose sequence number is at or above `psn`
    // and requeues them in sequence order ahead of anything newer. Returns
    // the upstream drop terminals those closures produce.
    std::vector<NetworkEvent> goBackN(std::uint32_t psn, Picoseconds now_ps);
    std::optional<Picoseconds> earliestTimeout() const;

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
    // The reaction point's own gate. Its rate is refreshed from the reaction
    // point at each issue, so a cut takes effect on the next packet rather
    // than retroactively on one already on the wire.
    RateGate cc_bits_;
    // Null unless the reaction point is configured.
    std::unique_ptr<RnicCcReactionPoint> reaction_;
    std::map<NetworkToken, Extent> extents_;
    struct DownstreamBinding {
        NetworkToken extent_token{0};
        std::size_t packet_index{0};
        std::uint32_t transmission_attempt{0};
    };
    std::map<NetworkToken, DownstreamBinding> downstream_tokens_;
    std::map<std::uint32_t, QueueEntry> queue_;
    // Live attempts by sequence number, so go-back-N can find everything at
    // or above a NAK's number without walking every extent.
    std::map<std::uint32_t, std::pair<NetworkToken, std::size_t>> live_psns_;
    bool in_recovery_{false};
    std::uint32_t recovery_psn_{0};
    RnicTxPipelineCounters counters_;
    RnicNicCounters nic_counters_;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_TX_PIPELINE_H
