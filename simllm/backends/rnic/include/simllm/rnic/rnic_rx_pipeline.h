#ifndef SIMLLM_RNIC_RNIC_RX_PIPELINE_H
#define SIMLLM_RNIC_RNIC_RX_PIPELINE_H

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "simllm/rnic/network_port.h"
#include "simllm/rnic/rnic_nic_counters.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicRxPipelineConfigVersion = 1;

enum class RnicTransportService : std::uint8_t {
    ReliableConnected,
    Unreliable,
};

// The receive pipeline is two blocks in series behind one entry point.
//
// The ingress meter is the finite receive buffer at the port. Wire bytes are
// admitted into it and drained at `drain_bps`; a packet that does not fit is
// discarded at the PHY and counted on `rx_discards_phy` with no transport
// signal of any kind, which is what makes the measured loss silent.
//
// The receive processor then applies the packet-rate ceilings and, for a
// reliable connection, the responder's sequence check. The sequence check runs
// at line rate on arrival, not at the drain instant: on real silicon the
// transport parser sits in the receive path and the buffer stages payload
// toward the host, so buffer occupancy delays delivery and not the ACK or the
// NAK. Modelling it the other way round would make the NAK late by the
// standing queue depth and collapse go-back-N at any loss rate at all.
struct RnicRxPipelineConfig {
    std::uint32_t version{kRnicRxPipelineConfigVersion};
    bool enabled{false};

    // Ingress meter. Zero disables that bound.
    std::uint64_t ingress_bytes{0};
    std::uint64_t drain_bps{0};

    // Receive processor ceilings, packets per second. Zero disables one.
    std::uint64_t rc_pps_per_qp{0};
    std::uint64_t ud_pps_per_qp{0};
    std::uint64_t pps_per_nic{0};

    // One pause frame is emitted per `pause_discard_interval` discards while
    // the buffer is over its pause threshold. The campaign measured a
    // receiver emitting pause frames under overload that no peer ever
    // received, so the frames are counted and never delivered.
    std::uint64_t pause_discard_interval{64};
};

// One inbound wire packet as the port presents it.
struct RnicRxPacket {
    std::uint32_t qpn{0};
    std::uint32_t source{0};
    std::uint32_t psn{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes{0};
    RnicTransportService service{RnicTransportService::ReliableConnected};
    NetworkPacketKind kind{NetworkPacketKind::Data};
    bool ecn_marked{false};
    // Set on the last packet of a message so the responder can count
    // `rx_write_requests` the way silicon counts completed requests.
    bool last_of_message{false};
};

enum class RnicRxOutcome : std::uint8_t {
    // Admitted, in sequence, delivered to the host. The responder replies.
    Delivered,
    // Discarded at the PHY by the ingress meter or a packet-rate ceiling.
    // Nothing is sent back: this is the silent loss the campaign measured.
    DiscardedSilently,
    // Admitted but out of sequence at a reliable responder. The packet is
    // dropped and, on the first such packet of a recovery epoch, one NAK
    // naming the expected sequence number is emitted.
    DiscardedOutOfSequence,
    // A sequence number the responder has already accepted. Dropped, counted
    // on `duplicate_request`, and re-acknowledged.
    DiscardedDuplicate,
};

struct RnicRxResult {
    RnicRxOutcome outcome{RnicRxOutcome::Delivered};
    // The responder's reply, if it made one. An ACK names the accepted
    // sequence number; a NAK names the sequence number it is still waiting
    // for.
    bool has_reply{false};
    NetworkPacketKind reply_kind{NetworkPacketKind::Ack};
    std::uint32_t reply_psn{0};
    std::uint64_t reply_wire_bytes{0};
    // Occupancy after this packet, so a study can see the meter working.
    std::uint64_t ingress_occupancy_bytes{0};
};

struct RnicRxPipelineCounters {
    std::uint64_t packets_offered{0};
    std::uint64_t packets_admitted{0};
    std::uint64_t packets_delivered{0};
    std::uint64_t packets_discarded_meter{0};
    std::uint64_t packets_discarded_rate{0};
    std::uint64_t packets_discarded_sequence{0};
    std::uint64_t packets_discarded_duplicate{0};
    std::uint64_t payload_bytes_delivered{0};
    std::uint64_t wire_bytes_offered{0};
    std::uint64_t wire_bytes_admitted{0};
    std::uint64_t acks{0};
    std::uint64_t naks{0};
    std::uint64_t ingress_occupancy_bytes{0};
    std::uint64_t ingress_high_watermark_bytes{0};
};

void validateRnicRxPipelineConfig(const RnicRxPipelineConfig& config);

class RnicRxPipeline final {
public:
    explicit RnicRxPipeline(RnicRxPipelineConfig config);

    RnicRxPipeline(const RnicRxPipeline&) = delete;
    RnicRxPipeline& operator=(const RnicRxPipeline&) = delete;

    // Presents one wire packet at `now_ps`. Time must not regress.
    RnicRxResult onPacket(const RnicRxPacket& packet, Picoseconds now_ps);

    // Drains the meter forward with no arrival. A caller that steps the clock
    // without traffic keeps the occupancy honest by calling this.
    void progress(Picoseconds now_ps);

    const RnicRxPipelineConfig& config() const noexcept;
    const RnicRxPipelineCounters& counters() const noexcept;
    const RnicNicCounters& nicCounters() const noexcept;
    std::uint64_t ingressOccupancyBytes() const noexcept;
    void validateInvariants() const;

private:
    // The same exact rational rate gate the transmit pacer uses: the
    // remainder carries the division's fractional picoseconds forward so a
    // long run has bounded error instead of one truncation per packet.
    struct RateGate {
        std::uint64_t rate{0};
        Picoseconds free_at_ps{0};
        std::uint64_t remainder{0};

        bool admits(Picoseconds now_ps) const noexcept;
        void charge(Picoseconds now_ps, std::uint64_t units);
    };

    struct QpState {
        RnicTransportService service{RnicTransportService::ReliableConnected};
        std::uint32_t expected_psn{0};
        bool in_recovery{false};
        RateGate rate;
    };

    void drainTo(Picoseconds now_ps);
    QpState& qpState(const RnicRxPacket& packet);
    void notePause();

    RnicRxPipelineConfig config_;
    Picoseconds last_now_ps_{0};
    std::uint64_t occupancy_bytes_{0};
    // Fractional bytes the drain has earned but not yet spent, kept as a
    // numerator over `drain_bps` so the meter is exact over a long run.
    std::uint64_t drain_remainder_{0};
    std::uint64_t discards_since_pause_{0};
    RateGate nic_rate_;
    std::map<std::pair<std::uint32_t, std::uint32_t>, QpState> qps_;
    RnicRxPipelineCounters counters_;
    RnicNicCounters nic_counters_;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_RX_PIPELINE_H
