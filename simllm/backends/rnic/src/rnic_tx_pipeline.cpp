#include "simllm/rnic/rnic_tx_pipeline.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace simllm::rnic {
namespace {

constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;

Picoseconds checkedAdd(Picoseconds lhs, Picoseconds rhs) {
    if (lhs > std::numeric_limits<Picoseconds>::max() - rhs) {
        throw std::overflow_error("RNIC transmit pipeline timestamp overflow");
    }
    return lhs + rhs;
}

}  // namespace

void validateRnicTxPipelineConfig(const RnicTxPipelineConfig& config) {
    if (config.version != kRnicTxPipelineConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC transmit pipeline config version");
    }
    if (!config.enabled) {
        throw std::invalid_argument(
            "RNIC transmit pipeline config is not enabled");
    }
    if (config.mtu_bytes == 0 || config.wire_header_bytes == 0) {
        throw std::invalid_argument(
            "RNIC transmit pipeline needs a positive MTU and wire header");
    }
    if (config.mtu_bytes
        > std::numeric_limits<std::uint64_t>::max() / 8
            - config.wire_header_bytes) {
        throw std::out_of_range("RNIC transmit pipeline MTU overflows");
    }
    if (!config.transport_enabled
        && (config.rto_ps != 0 || config.counts_local_ack_timeout)) {
        throw std::invalid_argument(
            "RNIC requester transport fields need the transport enabled");
    }
}

Picoseconds RnicTxPipeline::RateGate::delayFor(std::uint64_t units) {
    if (rate == 0 || units == 0) {
        return 0;
    }
    if (units > std::numeric_limits<std::uint64_t>::max()
            / kPicosecondsPerSecond) {
        throw std::overflow_error("RNIC rate gate unit overflow");
    }
    const std::uint64_t scaled = units * kPicosecondsPerSecond;
    if (scaled > std::numeric_limits<std::uint64_t>::max() - remainder) {
        throw std::overflow_error("RNIC rate gate remainder overflow");
    }
    const std::uint64_t numerator = scaled + remainder;
    remainder = numerator % rate;
    return numerator / rate;
}

RnicTxPipeline::RnicTxPipeline(
    RnicTxPipelineConfig config,
    NetworkPort& downstream)
    : config_(std::move(config)), downstream_(downstream) {
    validateRnicTxPipelineConfig(config_);
    const NetworkPortCapabilities downstream_caps = downstream_.capabilities();
    if (downstream_caps.abi_version != kNetworkPortAbiVersionV2
        || !downstream_caps.packet_attempt_events) {
        throw std::invalid_argument(
            "RNIC transmit pipeline requires an ABI v2 packet-attempt port");
    }
    next_psn_ = config_.initial_psn;
    qp_bits_.rate = config_.wire_bps_per_qp;
    nic_bits_.rate = config_.wire_bps_per_nic;
    qp_messages_.rate = config_.message_rate_per_qp;
    nic_messages_.rate = config_.message_rate_per_nic;
}

NetworkPortCapabilities RnicTxPipeline::capabilities() const noexcept {
    NetworkPortCapabilities caps;
    caps.abi_version = kNetworkPortAbiVersionV2;
    caps.packet_attempt_events = true;
    return caps;
}

NetworkSubmitResult RnicTxPipeline::trySubmit(
    const NetworkTxDescriptor& descriptor,
    Picoseconds now_ps) {
    if (descriptor.abi_version != kNetworkPortAbiVersionV2
        || descriptor.extent_count != 1 || descriptor.extent_index != 0) {
        return NetworkSubmitResult::rejected(
            DropLocation::TxPort, DropReason::PolicyRejected);
    }
    if (now_ps < last_now_ps_) {
        return NetworkSubmitResult::rejected(
            DropLocation::TxPort, DropReason::PolicyRejected);
    }
    last_now_ps_ = now_ps;

    const std::uint64_t mtu = config_.mtu_bytes;
    const std::uint64_t payload = descriptor.payload_bytes;
    const std::uint64_t packet_count =
        payload == 0 ? 1 : (payload + mtu - 1) / mtu;
    if (packet_count > std::numeric_limits<std::uint32_t>::max()) {
        return NetworkSubmitResult::rejected(
            DropLocation::TxPort, DropReason::PolicyRejected);
    }

    Extent extent;
    extent.extent_token = next_token_++;
    extent.wqe_id = descriptor.wqe_id;
    extent.descriptor = descriptor;
    extent.packets.reserve(static_cast<std::size_t>(packet_count));
    std::uint64_t offset = 0;
    for (std::uint64_t index = 0; index < packet_count; ++index) {
        Packet packet;
        packet.attempt_token = next_token_++;
        packet.packet_index = index;
        packet.payload_offset_bytes = offset;
        packet.payload_bytes = std::min(mtu, payload - offset);
        packet.wire_bytes = packet.payload_bytes + config_.wire_header_bytes;
        packet.psn = next_psn_++;
        offset += packet.payload_bytes;
        extent.packets.push_back(packet);
    }
    if (offset != payload) {
        throw std::logic_error("RNIC packetizer lost payload bytes");
    }

    const NetworkToken token = extent.extent_token;
    const auto inserted = extents_.emplace(token, std::move(extent));
    if (!inserted.second) {
        throw std::logic_error("duplicate RNIC transmit extent token");
    }
    for (std::size_t index = 0;
         index < inserted.first->second.packets.size();
         ++index) {
        queue_.emplace(
            inserted.first->second.packets[index].psn,
            QueueEntry{token, index, now_ps});
    }
    ++counters_.extents_accepted;
    return NetworkSubmitResult::accepted(token);
}

Picoseconds RnicTxPipeline::eligibleAt(const QueueEntry& entry) const {
    const Extent& extent = extents_.at(entry.extent_token);
    const Packet& packet = extent.packets[entry.packet_index];
    Picoseconds eligible = entry.queued_at_ps;
    eligible = std::max(eligible, window_open_ps_);
    eligible = std::max(eligible, qp_bits_.free_at_ps);
    eligible = std::max(eligible, nic_bits_.free_at_ps);
    if (packet.packet_index == 0) {
        eligible = std::max(eligible, qp_messages_.free_at_ps);
        eligible = std::max(eligible, nic_messages_.free_at_ps);
    }
    if (downstream_retry_at_ps_.has_value()) {
        eligible = std::max(eligible, *downstream_retry_at_ps_);
    }
    return eligible;
}

bool RnicTxPipeline::windowAllows(const QueueEntry& entry) const {
    const Extent& extent = extents_.at(entry.extent_token);
    const Packet& packet = extent.packets[entry.packet_index];
    if (!extent.in_flight && config_.max_inflight_wqes != 0
        && counters_.inflight_wqes >= config_.max_inflight_wqes) {
        return false;
    }
    // A bound smaller than one packet would deadlock the queue, so an empty
    // window always admits its head.
    if (config_.max_inflight_bytes != 0 && counters_.inflight_bytes != 0
        && counters_.inflight_bytes + packet.payload_bytes
            > config_.max_inflight_bytes) {
        return false;
    }
    if (config_.max_inflight_packets != 0 && counters_.inflight_packets != 0
        && counters_.inflight_packets >= config_.max_inflight_packets) {
        return false;
    }
    return true;
}

std::vector<NetworkEvent> RnicTxPipeline::goBackN(
    std::uint32_t psn,
    Picoseconds now_ps) {
    std::vector<NetworkEvent> events;
    for (auto live = live_psns_.lower_bound(psn); live != live_psns_.end();) {
        Extent& extent = extents_.at(live->second.first);
        Packet& packet = extent.packets[live->second.second];

        // Close the attempt the responder threw away, so the work queue sees
        // one clean lifecycle per attempt instead of a token that vanishes.
        NetworkEvent dropped = packet.started;
        dropped.kind = NetworkEventKind::Dropped;
        dropped.event_time_ps = now_ps;
        dropped.drop_location = DropLocation::RxPort;
        dropped.drop_reason = DropReason::QueueOverflow;
        dropped.drop_evidence = DropEvidenceProvenance::Inferred;
        dropped.drop_resource_id = packet.attempt_token;
        events.push_back(dropped);

        // Its terminal is still coming. Dropping the binding makes the
        // arriving terminal stale, which is what it is: the attempt it names
        // has already been replaced.
        if (packet.downstream_token != 0) {
            downstream_tokens_.erase(packet.downstream_token);
            packet.downstream_token = 0;
        }

        counters_.inflight_packets -= 1;
        counters_.inflight_bytes -= packet.payload_bytes;
        ++counters_.packets_dropped;
        packet.issued = false;
        packet.attempt_token = next_token_++;
        ++packet.transmission_attempt;
        ++counters_.packets_retransmitted;
        ++nic_counters_.roce_adp_retrans;
        ++nic_counters_.roce_slow_restart_cnps;
        queue_.emplace(
            packet.psn,
            QueueEntry{extent.extent_token, live->second.second, now_ps});
        live = live_psns_.erase(live);
    }
    return events;
}

std::optional<Picoseconds> RnicTxPipeline::earliestTimeout() const {
    if (!config_.transport_enabled || config_.rto_ps == 0
        || live_psns_.empty()) {
        return std::nullopt;
    }
    // Sequence numbers are issued in order and a recovery replays them in
    // order, so the lowest live sequence number is always the oldest attempt
    // and its deadline is the earliest. That keeps this O(1) on a path the
    // event loop walks once per step.
    const auto& oldest = *live_psns_.begin();
    const Extent& extent = extents_.at(oldest.second.first);
    const Packet& packet = extent.packets[oldest.second.second];
    return checkedAdd(packet.issued_at_ps, config_.rto_ps);
}

std::vector<NetworkEvent> RnicTxPipeline::onTransportPacket(
    const RnicTransportPacket& packet,
    Picoseconds now_ps) {
    if (!config_.transport_enabled) {
        throw std::logic_error(
            "RNIC transmit pipeline has no requester transport");
    }
    if (now_ps < last_now_ps_) {
        throw std::logic_error("RNIC requester transport time regressed");
    }
    last_now_ps_ = now_ps;
    if (packet.kind != NetworkPacketKind::Nak) {
        // An ACK arrives as the downstream port's delivery terminal, so a
        // duplicate one here carries no new information.
        return {};
    }
    ++counters_.naks_received;
    if (in_recovery_ && packet.psn >= recovery_psn_) {
        // The responder emits one NAK per epoch; a second inside the same
        // episode would replay the replay.
        return {};
    }
    in_recovery_ = true;
    recovery_psn_ = packet.psn;
    ++counters_.recovery_episodes;
    ++nic_counters_.packet_seq_err;
    return goBackN(packet.psn, now_ps);
}

std::vector<NetworkEvent> RnicTxPipeline::releaseDue(Picoseconds now_ps) {
    if (now_ps < last_now_ps_) {
        throw std::logic_error("RNIC transmit pipeline time regressed");
    }
    last_now_ps_ = now_ps;
    std::vector<NetworkEvent> events;
    if (config_.transport_enabled && config_.rto_ps != 0) {
        // A packet the responder never saw draws no NAK, so the only way out
        // is the timer. One expiry recovers the whole window from it.
        for (;;) {
            const auto deadline = earliestTimeout();
            if (!deadline.has_value() || *deadline > now_ps) {
                break;
            }
            const std::uint32_t psn = live_psns_.begin()->first;
            ++counters_.timeouts;
            ++counters_.recovery_episodes;
            if (config_.counts_local_ack_timeout) {
                ++nic_counters_.local_ack_timeout_err;
            }
            in_recovery_ = true;
            recovery_psn_ = psn;
            const std::vector<NetworkEvent> closed = goBackN(psn, now_ps);
            events.insert(events.end(), closed.begin(), closed.end());
        }
    }
    while (!queue_.empty()) {
        const QueueEntry entry = queue_.begin()->second;
        if (!windowAllows(entry)) {
            ++counters_.window_stalls;
            break;
        }
        const Picoseconds eligible = eligibleAt(entry);
        if (eligible > now_ps) {
            ++counters_.pacer_stalls;
            break;
        }
        if (eligible < now_ps) {
            ++counters_.late_releases;
        }

        Extent& extent = extents_.at(entry.extent_token);
        Packet& packet = extent.packets[entry.packet_index];

        NetworkTxDescriptor descriptor = extent.descriptor;
        descriptor.abi_version = kNetworkPortAbiVersionV2;
        descriptor.payload_bytes = packet.payload_bytes;
        descriptor.extent_index =
            static_cast<std::uint32_t>(packet.packet_index);
        descriptor.extent_count =
            static_cast<std::uint32_t>(extent.packets.size());
        descriptor.eligible_at_ps = now_ps;
        descriptor.psn = packet.psn;
        descriptor.transmission_attempt = packet.transmission_attempt;
        const NetworkSubmitResult result =
            downstream_.trySubmit(descriptor, now_ps);
        if (result.status == NetworkSubmitStatus::Busy) {
            if (!result.has_retry_time || result.retry_at_ps <= now_ps) {
                throw std::logic_error(
                    "busy RNIC packet port must provide a future retry time");
            }
            downstream_retry_at_ps_ = result.retry_at_ps;
            ++counters_.downstream_busy;
            break;
        }
        downstream_retry_at_ps_.reset();

        NetworkEvent started;
        started.abi_version = kNetworkPortAbiVersionV2;
        started.scope = NetworkEventScope::PacketAttempt;
        started.kind = NetworkEventKind::PacketTxStarted;
        started.token = packet.attempt_token;
        started.parent_token = extent.extent_token;
        started.wqe_id = extent.wqe_id;
        started.event_time_ps = now_ps;
        started.extent_index = 0;
        started.packet_index = packet.packet_index;
        started.transmission_attempt = packet.transmission_attempt;
        started.payload_offset_bytes = packet.payload_offset_bytes;
        started.payload_bytes = packet.payload_bytes;
        started.wire_bytes = packet.wire_bytes;
        started.packet_kind = packet.transmission_attempt == 0
            ? NetworkPacketKind::Data
            : NetworkPacketKind::Retransmission;
        packet.started = started;
        packet.issued = true;
        packet.issued_at_ps = now_ps;

        if (result.status == NetworkSubmitStatus::Rejected) {
            // A refused packet is a controlled drop at the port. The queue
            // still needs the started event first, so the attempt has a
            // lifecycle rather than a terminal out of nowhere.
            ++counters_.downstream_rejected;
            events.push_back(started);
            NetworkEvent dropped = started;
            dropped.kind = NetworkEventKind::Dropped;
            dropped.drop_location = result.rejection_location;
            dropped.drop_reason = result.rejection_reason;
            dropped.drop_evidence = DropEvidenceProvenance::Controlled;
            dropped.drop_resource_id = packet.attempt_token;
            events.push_back(dropped);
            queue_.erase(queue_.begin());
            ++counters_.packets_issued;
            ++counters_.packets_dropped;
            if (!extent.in_flight) {
                extent.in_flight = true;
                ++counters_.inflight_wqes;
            }
            ++extent.issued_count;
            extent.dropped = true;
            extent.drop_location = dropped.drop_location;
            extent.drop_reason = dropped.drop_reason;
            extent.drop_evidence = dropped.drop_evidence;
            extent.drop_resource_id = dropped.drop_resource_id;
            retirePacket(extent, packet);
            if (extent.terminal_count == extent.packets.size()) {
                NetworkEvent terminal;
                terminal.abi_version = kNetworkPortAbiVersionV2;
                terminal.scope = NetworkEventScope::FlowExtent;
                terminal.kind = NetworkEventKind::Dropped;
                terminal.token = extent.extent_token;
                terminal.wqe_id = extent.wqe_id;
                terminal.event_time_ps = now_ps;
                terminal.drop_location = extent.drop_location;
                terminal.drop_reason = extent.drop_reason;
                terminal.drop_evidence = extent.drop_evidence;
                terminal.drop_resource_id = extent.drop_resource_id;
                events.push_back(terminal);
                counters_.inflight_wqes -= 1;
                window_open_ps_ = now_ps;
                ++counters_.extents_completed;
                extents_.erase(entry.extent_token);
            }
            continue;
        }

        if (result.token == 0) {
            throw std::logic_error(
                "RNIC packet port returned an invalid acceptance");
        }
        if (!downstream_tokens_
                 .emplace(
                     result.token,
                     DownstreamBinding{
                         entry.extent_token,
                         entry.packet_index,
                         packet.transmission_attempt})
                 .second) {
            throw std::logic_error("duplicate RNIC packet port token");
        }
        packet.downstream_token = result.token;
        if (config_.transport_enabled) {
            live_psns_[packet.psn] =
                std::make_pair(entry.extent_token, entry.packet_index);
            if (in_recovery_ && packet.psn == recovery_psn_) {
                // The replay this episode was opened for is on the wire, so
                // the episode is closed. A later NAK at the same number is
                // then a new loss, not an echo of this one, and absorbing it
                // would leave the connection waiting on the timer.
                in_recovery_ = false;
            }
        }

        qp_bits_.free_at_ps = checkedAdd(
            now_ps, qp_bits_.delayFor(packet.wire_bytes * 8));
        nic_bits_.free_at_ps = checkedAdd(
            now_ps, nic_bits_.delayFor(packet.wire_bytes * 8));
        if (packet.packet_index == 0) {
            qp_messages_.free_at_ps =
                checkedAdd(now_ps, qp_messages_.delayFor(1));
            nic_messages_.free_at_ps =
                checkedAdd(now_ps, nic_messages_.delayFor(1));
        }

        if (!extent.in_flight) {
            extent.in_flight = true;
            ++counters_.inflight_wqes;
        }
        ++counters_.inflight_packets;
        counters_.inflight_bytes += packet.payload_bytes;
        ++extent.issued_count;
        ++counters_.packets_issued;
        counters_.payload_bytes += packet.payload_bytes;
        counters_.wire_bytes += packet.wire_bytes;
        events.push_back(started);
        queue_.erase(queue_.begin());
    }
    return events;
}

void RnicTxPipeline::retirePacket(Extent& extent, Packet& packet) {
    if (packet.terminal) {
        throw std::logic_error("RNIC packet attempt retired twice");
    }
    packet.terminal = true;
    ++extent.terminal_count;
}

std::vector<NetworkEvent> RnicTxPipeline::onDownstreamEvent(
    const NetworkEvent& event) {
    if (event.event_time_ps < last_now_ps_) {
        throw std::logic_error("RNIC transmit pipeline event time regressed");
    }
    last_now_ps_ = event.event_time_ps;
    const auto mapping = downstream_tokens_.find(event.token);
    if (mapping == downstream_tokens_.end()) {
        if (config_.transport_enabled) {
            // A terminal for an attempt go-back-N already replaced. The work
            // queue closed that attempt when the recovery episode opened, so
            // there is nothing left to say about it.
            ++counters_.stale_terminals;
            return {};
        }
        throw std::logic_error("unknown RNIC packet port token");
    }
    Extent& extent = extents_.at(mapping->second.extent_token);
    Packet& packet = extent.packets[mapping->second.packet_index];

    std::vector<NetworkEvent> events;
    NetworkEvent upstream = packet.started;
    upstream.event_time_ps = event.event_time_ps;
    upstream.ecn_marked = event.ecn_marked;
    bool terminal = false;
    switch (event.kind) {
    case NetworkEventKind::PacketTxFinished:
        upstream.kind = NetworkEventKind::PacketTxFinished;
        break;
    case NetworkEventKind::PacketRxArrived:
        upstream.kind = NetworkEventKind::PacketRxArrived;
        break;
    case NetworkEventKind::Delivered:
        upstream.kind = NetworkEventKind::Delivered;
        terminal = true;
        ++counters_.packets_delivered;
        if (config_.transport_enabled) {
            live_psns_.erase(packet.psn);
            packet.downstream_token = 0;
            if (in_recovery_ && packet.psn >= recovery_psn_) {
                in_recovery_ = false;
            }
        }
        break;
    case NetworkEventKind::Dropped:
        if (config_.transport_enabled) {
            // A port that reports the loss itself carries the same
            // information a NAK would, so it opens the same episode. The
            // attempt's own drop terminal comes out of go-back-N, not from
            // here, so the lifecycle is closed exactly once.
            downstream_tokens_.erase(mapping);
            if (in_recovery_ && packet.psn >= recovery_psn_) {
                ++counters_.stale_terminals;
                return {};
            }
            in_recovery_ = true;
            recovery_psn_ = packet.psn;
            ++counters_.recovery_episodes;
            ++nic_counters_.packet_seq_err;
            return goBackN(packet.psn, event.event_time_ps);
        }
        upstream.kind = NetworkEventKind::Dropped;
        upstream.drop_location = event.drop_location;
        upstream.drop_reason = event.drop_reason;
        upstream.drop_evidence = event.drop_evidence
                == DropEvidenceProvenance::None
            ? DropEvidenceProvenance::Observed
            : event.drop_evidence;
        upstream.drop_resource_id = event.drop_resource_id == 0
            ? packet.attempt_token
            : event.drop_resource_id;
        terminal = true;
        ++counters_.packets_dropped;
        extent.dropped = true;
        extent.drop_location = upstream.drop_location;
        extent.drop_reason = upstream.drop_reason;
        extent.drop_evidence = upstream.drop_evidence;
        extent.drop_resource_id = upstream.drop_resource_id;
        break;
    default:
        throw std::invalid_argument(
            "RNIC packet port emitted an unsupported event kind");
    }
    events.push_back(upstream);

    if (!terminal) {
        return events;
    }

    downstream_tokens_.erase(mapping);
    retirePacket(extent, packet);
    counters_.inflight_packets -= 1;
    counters_.inflight_bytes -= packet.payload_bytes;
    if (extent.terminal_count != extent.packets.size()) {
        return events;
    }

    NetworkEvent extent_event;
    extent_event.abi_version = kNetworkPortAbiVersionV2;
    extent_event.scope = NetworkEventScope::FlowExtent;
    extent_event.token = extent.extent_token;
    extent_event.wqe_id = extent.wqe_id;
    extent_event.event_time_ps = event.event_time_ps;
    if (extent.dropped) {
        extent_event.kind = NetworkEventKind::Dropped;
        extent_event.drop_location = extent.drop_location;
        extent_event.drop_reason = extent.drop_reason;
        extent_event.drop_evidence = extent.drop_evidence;
        extent_event.drop_resource_id = extent.drop_resource_id;
    } else {
        extent_event.kind = NetworkEventKind::Delivered;
    }
    events.push_back(extent_event);
    counters_.inflight_wqes -= 1;
    ++counters_.extents_completed;
    window_open_ps_ = event.event_time_ps;
    extents_.erase(extent.extent_token);
    return events;
}

std::optional<Picoseconds> RnicTxPipeline::nextEventTime() const {
    const std::optional<Picoseconds> timeout = earliestTimeout();
    std::optional<Picoseconds> issue;
    if (!queue_.empty()) {
        const QueueEntry& entry = queue_.begin()->second;
        if (windowAllows(entry)) {
            issue = std::max(eligibleAt(entry), last_now_ps_);
        }
    }
    if (!timeout.has_value()) {
        return issue;
    }
    const Picoseconds due = std::max(*timeout, last_now_ps_);
    return issue.has_value() ? std::min(*issue, due) : due;
}

bool RnicTxPipeline::hasPendingWork() const noexcept {
    return !queue_.empty() || !extents_.empty();
}

const RnicTxPipelineConfig& RnicTxPipeline::config() const noexcept {
    return config_;
}

const RnicTxPipelineCounters& RnicTxPipeline::counters() const noexcept {
    return counters_;
}

const RnicNicCounters& RnicTxPipeline::nicCounters() const noexcept {
    return nic_counters_;
}

void RnicTxPipeline::validateInvariants() const {
    std::uint64_t inflight_wqes = 0;
    std::uint64_t inflight_packets = 0;
    std::uint64_t inflight_bytes = 0;
    for (const auto& token_and_extent : extents_) {
        const Extent& extent = token_and_extent.second;
        if (extent.packets.empty()) {
            throw std::logic_error("RNIC transmit extent has no packets");
        }
        if (extent.terminal_count > extent.issued_count
            || extent.issued_count > extent.packets.size()) {
            throw std::logic_error("RNIC transmit extent counts are impossible");
        }
        if (extent.in_flight) {
            ++inflight_wqes;
        }
        std::uint64_t payload = 0;
        for (const Packet& packet : extent.packets) {
            payload += packet.payload_bytes;
            if (packet.terminal && !packet.issued) {
                throw std::logic_error(
                    "RNIC transmit packet retired before it was issued");
            }
            if (packet.issued && !packet.terminal) {
                ++inflight_packets;
                inflight_bytes += packet.payload_bytes;
            }
        }
        if (payload != extent.descriptor.payload_bytes) {
            throw std::logic_error(
                "RNIC packetizer does not conserve payload bytes");
        }
    }
    if (inflight_wqes != counters_.inflight_wqes
        || inflight_packets != counters_.inflight_packets
        || inflight_bytes != counters_.inflight_bytes) {
        throw std::logic_error(
            "RNIC transmit window counters disagree with its extents");
    }
    if (counters_.packets_delivered + counters_.packets_dropped
        > counters_.packets_issued) {
        throw std::logic_error(
            "RNIC transmit pipeline retired more packets than it issued");
    }
}

}  // namespace simllm::rnic
