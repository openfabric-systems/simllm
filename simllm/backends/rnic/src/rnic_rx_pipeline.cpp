#include "simllm/rnic/rnic_rx_pipeline.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace simllm::rnic {
namespace {

constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;
constexpr std::uint64_t kAckWireBytes = 64;
constexpr std::uint64_t kByteCreditDenominator = 8 * kPicosecondsPerSecond;

std::uint64_t ceilDivide(std::uint64_t numerator, std::uint64_t denominator) {
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}
// A RoCEv2 congestion notification packet is one BTH plus its padding on the
// wire, which is the same envelope an acknowledgement occupies.
constexpr std::uint64_t kCnpWireBytes = 64;

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
    const RnicRxServiceConfig& service = config.service;
    if (service.version != kRnicRxServiceConfigVersion
        || service.ready != RnicRxReadySemantics::ContinuouslyReady) {
        throw std::invalid_argument("unsupported RNIC receive service configuration");
    }
    if (service.mode == RnicRxServiceMode::Fluid) {
        if (service.beat_bytes != 0 || service.period_ps != 0
            || service.phase_ps != 0 || service.token_capacity_bytes != 0) {
            throw std::invalid_argument("fluid receive service has no beat geometry");
        }
    } else if (service.mode == RnicRxServiceMode::SerializedPacketBeats) {
        if (service.beat_bytes == 0 || service.period_ps == 0
            || service.phase_ps >= service.period_ps || config.drain_bps == 0
            || service.token_capacity_bytes < service.beat_bytes) {
            throw std::invalid_argument("invalid RNIC serialized receive geometry");
        }
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        if (service.token_capacity_bytes > maximum / kByteCreditDenominator) {
            throw std::overflow_error("RNIC receive token capacity overflow");
        }
        const auto cap = service.token_capacity_bytes * kByteCreditDenominator;
        if (config.drain_bps > (maximum - cap) / service.period_ps) {
            throw std::overflow_error("RNIC receive clock refill overflow");
        }
    } else {
        throw std::invalid_argument("unknown RNIC receive service mode");
    }
    if (config.pause_discard_interval == 0) {
        throw std::invalid_argument(
            "RNIC ingress pause interval must be positive");
    }
    if (config.notification.enabled) {
        validateRnicCcNotificationConfig(config.notification);
        if (config.ingress_bytes != 0
            && config.notification.threshold_bytes > config.ingress_bytes) {
            throw std::invalid_argument(
                "an RNIC notification threshold above the ingress buffer can "
                "never be reached");
        }
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
    if (config_.service.mode == RnicRxServiceMode::SerializedPacketBeats) {
        next_tick_ps_ = config_.service.phase_ps;
        service_cap_ = config_.service.token_capacity_bytes * kByteCreditDenominator;
        service_refill_ = config_.drain_bps * config_.service.period_ps;
    }
    if (config_.notification.enabled) {
        notification_ = std::make_unique<RnicCcNotificationPoint>(
            config_.notification);
    }
}

void RnicRxPipeline::drainTo(Picoseconds now_ps) {
    if (now_ps < last_now_ps_) {
        throw std::logic_error("RNIC receive pipeline time regressed");
    }
    if (config_.service.mode == RnicRxServiceMode::SerializedPacketBeats) {
        drainSerializedTo(now_ps);
        last_now_ps_ = now_ps;
        return;
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

void RnicRxPipeline::clockAfter(Picoseconds now_ps) noexcept {
    const auto phase = config_.service.phase_ps;
    const auto period = config_.service.period_ps;
    if (now_ps < phase) {
        next_tick_ps_ = phase;
        return;
    }
    const auto distance = period - (now_ps - phase) % period;
    if (distance > std::numeric_limits<Picoseconds>::max() - now_ps) {
        next_tick_ps_.reset();
    } else {
        next_tick_ps_ = now_ps + distance;
    }
}

void RnicRxPipeline::accrueServiceCredit(std::uint64_t ticks) {
    const auto room = service_cap_ - service_credit_;
    // Compare before multiplying, including a leap over a long credit wait.
    if (ticks >= ceilDivide(room, service_refill_)) {
        service_credit_ = service_cap_;
    } else {
        service_credit_ += ticks * service_refill_;
    }
}

std::optional<Picoseconds> RnicRxPipeline::nextServiceTime() const {
    if (!hasPendingService()) {
        return std::nullopt;
    }
    if (!next_tick_ps_.has_value()) {
        throw std::overflow_error("RNIC receive service timestamp horizon exhausted");
    }
    const auto cost = std::min(config_.service.beat_bytes, service_records_.front())
                      * kByteCreditDenominator;
    const auto owed = cost > service_credit_ ? cost - service_credit_ : 0;
    const auto ticks = std::max<std::uint64_t>(1, ceilDivide(owed, service_refill_));
    if (ticks - 1 > (std::numeric_limits<Picoseconds>::max() - *next_tick_ps_)
                   / config_.service.period_ps) {
        throw std::overflow_error("RNIC receive service timestamp horizon exhausted");
    }
    return *next_tick_ps_ + (ticks - 1) * config_.service.period_ps;
}

bool RnicRxPipeline::hasPendingService() const noexcept {
    return config_.service.mode == RnicRxServiceMode::SerializedPacketBeats
           && !service_records_.empty();
}

void RnicRxPipeline::drainSerializedTo(Picoseconds now_ps) {
    if (same_time_credit_.has_value()
        && now_ps > same_time_credit_->timestamp_ps) {
        // Even an off-grid positive empty interval ends the earned epoch.
        same_time_credit_.reset();
    }
    while (hasPendingService() && next_tick_ps_.has_value()
           && *next_tick_ps_ <= now_ps) {
        const auto next = nextServiceTime();
        if (!next.has_value() || *next > now_ps) {
            // Finite backlog and a tiny rate must not loop once per idle clock.
            const auto ticks = (now_ps - *next_tick_ps_) / config_.service.period_ps + 1;
            accrueServiceCredit(ticks);
            clockAfter(now_ps);
            return;
        }
        const auto ticks = (*next - *next_tick_ps_) / config_.service.period_ps + 1;
        accrueServiceCredit(ticks);
        const auto bytes = std::min(config_.service.beat_bytes, service_records_.front());
        service_credit_ -= bytes * kByteCreditDenominator;
        occupancy_bytes_ -= bytes;
        // A legal debit remains observable even if the next clock overflows.
        counters_.ingress_occupancy_bytes = occupancy_bytes_;
        service_bytes_ += bytes;
        service_records_.front() -= bytes;
        if (service_records_.front() == 0) {
            service_records_.pop_front();
        }
        if (!hasPendingService() && *next == now_ps && service_credit_ != 0) {
            same_time_credit_ = SameTimeCredit{now_ps, service_credit_};
        }
        clockAfter(*next);
    }
    if (hasPendingService() && !next_tick_ps_.has_value()) {
        throw std::overflow_error("RNIC receive service timestamp horizon exhausted");
    }
    if (!hasPendingService()) {
        // The private exact-time candidate is not live credit or pending work.
        service_credit_ = 0;
        clockAfter(now_ps);
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

void RnicRxPipeline::observeCongestion(
    const RnicRxPacket& packet,
    std::uint64_t observed_occupancy_bytes,
    Picoseconds now_ps,
    RnicRxResult& result) {
    if (notification_ == nullptr) {
        return;
    }
    const bool raised = notification_->observe(
        packet.source, packet.qpn, observed_occupancy_bytes, now_ps);
    const RnicCcNotificationCounters& counters = notification_->counters();
    counters_.cnps_sent = counters.sent;
    counters_.cnps_suppressed = counters.suppressed;
    counters_.congestion_observations = counters.observed;
    if (!raised) {
        return;
    }
    // The notification leaves as a wire packet, so it costs the receiver a
    // transmit exactly as an acknowledgement does. It is deliberately not
    // recorded on the marking counter: that one stays inert on silicon while
    // notifications are generated, and reproducing the pair is the point.
    ++nic_counters_.np_cnp_sent;
    ++nic_counters_.tx_packets_phy;
    nic_counters_.tx_bytes_phy += kCnpWireBytes;
    result.has_cnp = true;
    result.cnp_source = packet.source;
    result.cnp_qpn = packet.qpn;
}

RnicRxResult RnicRxPipeline::onPacket(
    const RnicRxPacket& packet,
    Picoseconds now_ps) {
    drainTo(now_ps);
    if (packet.wire_bytes == 0 || packet.payload_bytes > packet.wire_bytes) {
        throw std::invalid_argument("RNIC receive packet has no wire envelope");
    }

    if (config_.service.mode == RnicRxServiceMode::SerializedPacketBeats) {
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        if (!next_tick_ps_.has_value()
            || packet.wire_bytes > maximum - occupancy_bytes_
            || packet.wire_bytes > maximum - counters_.wire_bytes_offered) {
            throw std::overflow_error("RNIC serialized receive packet arithmetic overflow");
        }
    }
    ++counters_.packets_offered;
    counters_.wire_bytes_offered += packet.wire_bytes;
    ++nic_counters_.rx_packets_phy;
    nic_counters_.rx_bytes_phy += packet.wire_bytes;

    RnicRxResult result;
    result.ingress_occupancy_bytes = occupancy_bytes_;

    // The notification point runs before the meter's verdict and on the
    // occupancy the packet would leave behind, so an arrival into a buffer
    // that is already too full to keep it is the loudest congestion signal
    // rather than a silent one. Nothing here changes what the meter then does.
    observeCongestion(
        packet, occupancy_bytes_ + packet.wire_bytes, now_ps, result);

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

    if (config_.service.mode == RnicRxServiceMode::SerializedPacketBeats) {
        service_records_.push_back(packet.wire_bytes);
        // Every admission gate and the potentially throwing FIFO push has
        // succeeded. A transport sequence rejection still consumes ingress.
        if (same_time_credit_.has_value()
            && same_time_credit_->timestamp_ps == now_ps) {
            service_credit_ = same_time_credit_->numerator;
            same_time_credit_.reset();
        }
    }
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
    if (config_.service.mode == RnicRxServiceMode::SerializedPacketBeats) {
        if (same_time_credit_.has_value()) {
            const auto& candidate = *same_time_credit_;
            if (hasPendingService() || service_credit_ != 0
                || candidate.numerator == 0 || candidate.numerator > service_cap_
                || candidate.timestamp_ps != last_now_ps_
                || candidate.timestamp_ps < config_.service.phase_ps
                || (candidate.timestamp_ps - config_.service.phase_ps)
                    % config_.service.period_ps != 0) {
                throw std::logic_error("RNIC same-time receive credit isolation failed");
            }
        }
        std::uint64_t queued = 0;
        for (const auto remaining : service_records_) {
            if (remaining == 0 || remaining > occupancy_bytes_ - queued) {
                throw std::logic_error("RNIC serialized receive FIFO accounting failed");
            }
            queued += remaining;
        }
        if (queued != occupancy_bytes_ || service_credit_ > service_cap_
            || service_bytes_ != counters_.wire_bytes_admitted - occupancy_bytes_
            || (!hasPendingService() && service_credit_ != 0)) {
            throw std::logic_error("RNIC serialized receive conservation failed");
        }
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
