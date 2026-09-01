#include "simllm/rnic/rnic_rx_pipeline.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace simllm::rnic {
namespace {

constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;
constexpr std::uint64_t kAckWireBytes = 64;

}  // namespace

bool rnicNicCountersMonotone(
    const RnicNicCounters& earlier,
    const RnicNicCounters& later) noexcept {
    return later.packet_seq_err >= earlier.packet_seq_err
        && later.roce_adp_retrans >= earlier.roce_adp_retrans
        && later.roce_slow_restart_cnps >= earlier.roce_slow_restart_cnps
        && later.local_ack_timeout_err >= earlier.local_ack_timeout_err
        && later.rp_cnp_handled >= earlier.rp_cnp_handled
        && later.rp_cnp_ignored >= earlier.rp_cnp_ignored
        && later.out_of_sequence >= earlier.out_of_sequence
        && later.duplicate_request >= earlier.duplicate_request
        && later.rx_discards_phy >= earlier.rx_discards_phy
        && later.rx_prio0_discards >= earlier.rx_prio0_discards
        && later.tx_pause_ctrl_phy >= earlier.tx_pause_ctrl_phy
        && later.tx_global_pause >= earlier.tx_global_pause
        && later.np_cnp_sent >= earlier.np_cnp_sent
        && later.rx_write_requests >= earlier.rx_write_requests
        && later.rx_packets_phy >= earlier.rx_packets_phy
        && later.rx_bytes_phy >= earlier.rx_bytes_phy
        && later.tx_packets_phy >= earlier.tx_packets_phy
        && later.tx_bytes_phy >= earlier.tx_bytes_phy
        && later.np_ecn_marked_roce_packets
            >= earlier.np_ecn_marked_roce_packets
        && later.rx_pause_ctrl_phy >= earlier.rx_pause_ctrl_phy
        && later.rx_global_pause >= earlier.rx_global_pause
        && later.rx_out_of_buffer >= earlier.rx_out_of_buffer
        && later.outbound_pci_stalled_rd >= earlier.outbound_pci_stalled_rd
        && later.outbound_pci_stalled_wr >= earlier.outbound_pci_stalled_wr;
}

void validateRnicRxPipelineConfig(const RnicRxPipelineConfig& config) {
    if (config.version != kRnicRxPipelineConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC receive pipeline config version");
    }
    if (!config.enabled) {
        throw std::invalid_argument(
            "RNIC receive pipeline config is not enabled");
    }
    if (config.ingress_bytes != 0 && config.drain_bps == 0) {
        throw std::invalid_argument(
            "a bounded RNIC ingress buffer needs a positive drain rate");
    }
    if (config.pause_discard_interval == 0) {
        throw std::invalid_argument(
            "RNIC ingress pause interval must be positive");
    }
}

bool RnicRxPipeline::RateGate::admits(Picoseconds now_ps) const noexcept {
    return rate == 0 || now_ps >= free_at_ps;
}

void RnicRxPipeline::RateGate::charge(
    Picoseconds now_ps,
    std::uint64_t units) {
    if (rate == 0 || units == 0) {
        return;
    }
    if (units > std::numeric_limits<std::uint64_t>::max()
            / kPicosecondsPerSecond) {
        throw std::overflow_error("RNIC receive rate gate unit overflow");
    }
    const std::uint64_t scaled = units * kPicosecondsPerSecond;
    if (scaled > std::numeric_limits<std::uint64_t>::max() - remainder) {
        throw std::overflow_error("RNIC receive rate gate remainder overflow");
    }
    const std::uint64_t numerator = scaled + remainder;
    remainder = numerator % rate;
    const Picoseconds delay = numerator / rate;
    // A virtual clock, not a dead time. The gate charges from where it last
    // was, so an offer that does not divide the ceiling still averages out to
    // the ceiling instead of quantizing down to the next whole divisor. One
    // packet of credit is the most it may carry back from an idle stretch, so
    // a long silence cannot be spent as a burst.
    const Picoseconds floor_ps = now_ps > delay ? now_ps - delay : 0;
    Picoseconds base = std::max(free_at_ps, floor_ps);
    if (base > std::numeric_limits<Picoseconds>::max() - delay) {
        throw std::overflow_error("RNIC receive rate gate timestamp overflow");
    }
    free_at_ps = base + delay;
}

RnicRxPipeline::RnicRxPipeline(RnicRxPipelineConfig config)
    : config_(std::move(config)) {
    validateRnicRxPipelineConfig(config_);
    nic_rate_.rate = config_.pps_per_nic;
}

void RnicRxPipeline::drainTo(Picoseconds now_ps) {
    if (now_ps < last_now_ps_) {
        throw std::logic_error("RNIC receive pipeline time regressed");
    }
    const Picoseconds elapsed = now_ps - last_now_ps_;
    last_now_ps_ = now_ps;
    if (config_.drain_bps == 0 || occupancy_bytes_ == 0) {
        // With nothing queued the remainder must not carry credit forward: an
        // idle meter does not bank drain capacity it never used.
        drain_remainder_ = 0;
        return;
    }
    if (elapsed > std::numeric_limits<std::uint64_t>::max()
            / config_.drain_bps) {
        occupancy_bytes_ = 0;
        drain_remainder_ = 0;
        return;
    }
    const std::uint64_t bits_numerator =
        elapsed * config_.drain_bps + drain_remainder_;
    const std::uint64_t drained_bits = bits_numerator / kPicosecondsPerSecond;
    drain_remainder_ = bits_numerator % kPicosecondsPerSecond;
    const std::uint64_t drained_bytes = drained_bits / 8;
    // The bits that did not make a whole byte stay owed to the meter.
    drain_remainder_ += (drained_bits % 8) * kPicosecondsPerSecond;
    if (drained_bytes >= occupancy_bytes_) {
        occupancy_bytes_ = 0;
        drain_remainder_ = 0;
    } else {
        occupancy_bytes_ -= drained_bytes;
    }
}

void RnicRxPipeline::progress(Picoseconds now_ps) {
    drainTo(now_ps);
    counters_.ingress_occupancy_bytes = occupancy_bytes_;
}

RnicRxPipeline::QpState& RnicRxPipeline::qpState(const RnicRxPacket& packet) {
    const auto key = std::make_pair(packet.source, packet.qpn);
    const auto found = qps_.find(key);
    if (found != qps_.end()) {
        return found->second;
    }
    QpState state;
    state.service = packet.service;
    state.expected_psn = packet.psn;
    state.rate.rate = packet.service == RnicTransportService::Unreliable
        ? config_.ud_pps_per_qp
        : config_.rc_pps_per_qp;
    return qps_.emplace(key, state).first->second;
}

void RnicRxPipeline::notePause() {
    ++discards_since_pause_;
    if (discards_since_pause_ < config_.pause_discard_interval) {
        return;
    }
    discards_since_pause_ = 0;
    // A receiver under overload emits a global pause frame. The campaign
    // measured that no peer ever received one, so it is counted here and
    // never handed to a port.
    ++nic_counters_.tx_pause_ctrl_phy;
    ++nic_counters_.tx_global_pause;
}

RnicRxResult RnicRxPipeline::onPacket(
    const RnicRxPacket& packet,
    Picoseconds now_ps) {
    drainTo(now_ps);
    if (packet.wire_bytes == 0 || packet.payload_bytes > packet.wire_bytes) {
        throw std::invalid_argument("RNIC receive packet has no wire envelope");
    }

    ++counters_.packets_offered;
    counters_.wire_bytes_offered += packet.wire_bytes;
    ++nic_counters_.rx_packets_phy;
    nic_counters_.rx_bytes_phy += packet.wire_bytes;

    RnicRxResult result;
    result.ingress_occupancy_bytes = occupancy_bytes_;

    // Block one: the ingress meter. An overflow is a PHY discard with no
    // transport signal at all, which is what makes the loss silent.
    const bool bounded = config_.ingress_bytes != 0;
    if (bounded && occupancy_bytes_ + packet.wire_bytes > config_.ingress_bytes) {
        ++counters_.packets_discarded_meter;
        ++nic_counters_.rx_discards_phy;
        ++nic_counters_.rx_prio0_discards;
        notePause();
        result.outcome = RnicRxOutcome::DiscardedSilently;
        result.ingress_occupancy_bytes = occupancy_bytes_;
        return result;
    }

    QpState& qp = qpState(packet);
    // Block two, first half: the packet-rate ceilings. A ceiling that is not
    // free discards at the PHY exactly as an overflow does, because the
    // packet never reaches the transport.
    if (!qp.rate.admits(now_ps) || !nic_rate_.admits(now_ps)) {
        ++counters_.packets_discarded_rate;
        ++nic_counters_.rx_discards_phy;
        ++nic_counters_.rx_prio0_discards;
        notePause();
        result.outcome = RnicRxOutcome::DiscardedSilently;
        result.ingress_occupancy_bytes = occupancy_bytes_;
        return result;
    }
    qp.rate.charge(now_ps, 1);
    nic_rate_.charge(now_ps, 1);

    occupancy_bytes_ += packet.wire_bytes;
    if (occupancy_bytes_ > counters_.ingress_high_watermark_bytes) {
        counters_.ingress_high_watermark_bytes = occupancy_bytes_;
    }
    counters_.ingress_occupancy_bytes = occupancy_bytes_;
    result.ingress_occupancy_bytes = occupancy_bytes_;
    ++counters_.packets_admitted;
    counters_.wire_bytes_admitted += packet.wire_bytes;

    // Block two, second half: the responder's sequence check. Unreliable
    // datagrams have none, which is precisely why their loss is invisible.
    if (packet.service == RnicTransportService::Unreliable) {
        ++counters_.packets_delivered;
        counters_.payload_bytes_delivered += packet.payload_bytes;
        if (packet.last_of_message) {
            ++nic_counters_.rx_write_requests;
        }
        result.outcome = RnicRxOutcome::Delivered;
        return result;
    }

    if (packet.psn == qp.expected_psn) {
        ++qp.expected_psn;
        qp.in_recovery = false;
        ++counters_.packets_delivered;
        counters_.payload_bytes_delivered += packet.payload_bytes;
        if (packet.last_of_message) {
            ++nic_counters_.rx_write_requests;
        }
        ++counters_.acks;
        ++nic_counters_.tx_packets_phy;
        nic_counters_.tx_bytes_phy += kAckWireBytes;
        result.outcome = RnicRxOutcome::Delivered;
        result.has_reply = true;
        result.reply_kind = NetworkPacketKind::Ack;
        result.reply_psn = packet.psn;
        result.reply_wire_bytes = kAckWireBytes;
        return result;
    }

    // The bytes stay charged. A packet the responder throws away was still
    // received, parsed and sequence-checked, so it consumed the ingress
    // service its bytes were metered for. Refunding it would make go-back-N
    // free at the receiver and pin the equilibrium goodput to the drain rate,
    // which is the one thing the measured equilibrium says it is not.
    if (packet.psn < qp.expected_psn) {
        ++counters_.packets_discarded_duplicate;
        ++nic_counters_.duplicate_request;
        ++counters_.acks;
        ++nic_counters_.tx_packets_phy;
        nic_counters_.tx_bytes_phy += kAckWireBytes;
        result.outcome = RnicRxOutcome::DiscardedDuplicate;
        result.has_reply = true;
        result.reply_kind = NetworkPacketKind::Ack;
        result.reply_psn = qp.expected_psn == 0 ? 0 : qp.expected_psn - 1;
        result.reply_wire_bytes = kAckWireBytes;
        return result;
    }

    ++counters_.packets_discarded_sequence;
    result.outcome = RnicRxOutcome::DiscardedOutOfSequence;
    // One NAK per recovery epoch, which is why the responder's out-of-sequence
    // count and the requester's sequence-error count track each other one for
    // one across a run. A retransmission that is still out of sequence opens a
    // new epoch: it says the replay itself did not survive, and without a
    // second NAK the connection would sit on the requester's timer instead.
    if (!qp.in_recovery
        || packet.kind == NetworkPacketKind::Retransmission) {
        qp.in_recovery = true;
        ++nic_counters_.out_of_sequence;
        ++counters_.naks;
        ++nic_counters_.tx_packets_phy;
        nic_counters_.tx_bytes_phy += kAckWireBytes;
        result.has_reply = true;
        result.reply_kind = NetworkPacketKind::Nak;
        result.reply_psn = qp.expected_psn;
        result.reply_wire_bytes = kAckWireBytes;
    }
    return result;
}

const RnicRxPipelineConfig& RnicRxPipeline::config() const noexcept {
    return config_;
}

const RnicRxPipelineCounters& RnicRxPipeline::counters() const noexcept {
    return counters_;
}

const RnicNicCounters& RnicRxPipeline::nicCounters() const noexcept {
    return nic_counters_;
}

std::uint64_t RnicRxPipeline::ingressOccupancyBytes() const noexcept {
    return occupancy_bytes_;
}

void RnicRxPipeline::validateInvariants() const {
    const std::uint64_t accounted = counters_.packets_delivered
        + counters_.packets_discarded_meter
        + counters_.packets_discarded_rate
        + counters_.packets_discarded_sequence
        + counters_.packets_discarded_duplicate;
    if (accounted != counters_.packets_offered) {
        throw std::logic_error(
            "RNIC receive pipeline lost track of an offered packet");
    }
    if (counters_.packets_admitted
        != counters_.packets_delivered
            + counters_.packets_discarded_sequence
            + counters_.packets_discarded_duplicate) {
        throw std::logic_error(
            "RNIC ingress admissions disagree with the receive processor");
    }
    if (config_.ingress_bytes != 0
        && occupancy_bytes_ > config_.ingress_bytes) {
        throw std::logic_error("RNIC ingress buffer overfilled");
    }
    if (nic_counters_.np_ecn_marked_roce_packets != 0
        || nic_counters_.rx_pause_ctrl_phy != 0
        || nic_counters_.rx_global_pause != 0
        || nic_counters_.rx_out_of_buffer != 0
        || nic_counters_.outbound_pci_stalled_rd != 0
        || nic_counters_.outbound_pci_stalled_wr != 0) {
        throw std::logic_error(
            "an RNIC counter the campaign measured inert has moved");
    }
}

}  // namespace simllm::rnic
