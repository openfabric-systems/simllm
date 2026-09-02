#ifndef SIMLLM_RNIC_TESTS_FAKE_NETWORK_H
#define SIMLLM_RNIC_TESTS_FAKE_NETWORK_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "simllm/rnic/network_port.h"
#include "simllm/rnic/rnic_rx_pipeline.h"
#include "simllm/rnic/rnic_tx_pipeline.h"

namespace simllm::rnic::testing {

struct FakeSubmission {
    NetworkToken token{0};
    NetworkTxDescriptor descriptor;
    Picoseconds submitted_at_ps{0};
    Picoseconds completion_at_ps{0};
};

struct FakeNetworkPortState {
    std::size_t capacity{0};
    Picoseconds latency_ps{0};
    NetworkToken next_token{0};
    bool reject_next{false};
    std::optional<Picoseconds> forced_busy_until_ps;
    std::map<NetworkToken, FakeSubmission> inflight;
    std::vector<FakeSubmission> history;
};

class FakeNetworkPort final : public NetworkPort {
public:
    FakeNetworkPort(std::size_t capacity, Picoseconds latency_ps)
        : capacity_(capacity), latency_ps_(latency_ps) {
        if (capacity_ == 0) {
            throw std::invalid_argument("fake network capacity must be positive");
        }
    }

    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        if (descriptor.abi_version != kNetworkPortAbiVersion) {
            return NetworkSubmitResult::rejected();
        }
        if (forced_busy_until_ps_.has_value()) {
            if (now_ps < *forced_busy_until_ps_) {
                return NetworkSubmitResult::busy(*forced_busy_until_ps_);
            }
            forced_busy_until_ps_.reset();
        }
        if (reject_next_) {
            reject_next_ = false;
            return NetworkSubmitResult::rejected(
                DropLocation::TxPort, DropReason::PolicyRejected);
        }
        if (inflight_.size() == capacity_) {
            const Picoseconds retry_at = nextCompletionTime().value();
            if (retry_at <= now_ps) {
                throw std::logic_error(
                    "fake network does not support zero-latency "
                    "oversubscription; dispatch due completions first");
            }
            return NetworkSubmitResult::busy(retry_at);
        }
        if (latency_ps_ > std::numeric_limits<Picoseconds>::max() - now_ps) {
            throw std::overflow_error("fake network timestamp overflow");
        }
        const NetworkToken token = next_token_++;
        const Picoseconds completion_at = now_ps + latency_ps_;
        FakeSubmission submission{token, descriptor, now_ps, completion_at};
        inflight_.emplace(token, submission);
        history_.push_back(submission);
        return NetworkSubmitResult::accepted(token);
    }

    void rejectNext() noexcept { reject_next_ = true; }
    void forceBusyUntil(Picoseconds retry_at_ps) {
        forced_busy_until_ps_ = retry_at_ps;
    }

    std::optional<Picoseconds> nextCompletionTime() const {
        std::optional<Picoseconds> result;
        for (const auto& item : inflight_) {
            if (!result.has_value()
                || item.second.completion_at_ps < *result) {
                result = item.second.completion_at_ps;
            }
        }
        return result;
    }

    std::vector<NetworkEvent> takeDue(Picoseconds now_ps) {
        std::vector<std::pair<Picoseconds, NetworkToken>> due;
        for (const auto& item : inflight_) {
            if (item.second.completion_at_ps <= now_ps) {
                due.emplace_back(item.second.completion_at_ps, item.first);
            }
        }
        std::sort(due.begin(), due.end());
        std::vector<NetworkEvent> events;
        events.reserve(due.size());
        for (const auto& time_and_token : due) {
            events.push_back(take(
                time_and_token.second,
                NetworkEventKind::Delivered,
                time_and_token.first));
        }
        return events;
    }

    NetworkEvent take(
        NetworkToken token,
        NetworkEventKind kind,
        Picoseconds event_time_ps,
        DropLocation location = DropLocation::None,
        DropReason reason = DropReason::None) {
        const auto item = inflight_.find(token);
        if (item == inflight_.end()) {
            throw std::out_of_range("unknown fake network token");
        }
        NetworkEvent event;
        event.kind = kind;
        event.token = token;
        event.wqe_id = item->second.descriptor.wqe_id;
        event.event_time_ps = event_time_ps;
        event.drop_location = location;
        event.drop_reason = reason;
        inflight_.erase(item);
        return event;
    }

    NetworkToken tokenForWqe(WqeId wqe_id) const {
        for (const auto& item : inflight_) {
            if (item.second.descriptor.wqe_id == wqe_id) {
                return item.first;
            }
        }
        throw std::out_of_range("WQE has no fake network token");
    }

    std::size_t inflightCount() const noexcept { return inflight_.size(); }
    const std::vector<FakeSubmission>& history() const noexcept {
        return history_;
    }
    FakeNetworkPortState state() const {
        return FakeNetworkPortState{
            capacity_,
            latency_ps_,
            next_token_,
            reject_next_,
            forced_busy_until_ps_,
            inflight_,
            history_,
        };
    }

private:
    std::size_t capacity_;
    Picoseconds latency_ps_;
    NetworkToken next_token_{1};
    bool reject_next_{false};
    std::optional<Picoseconds> forced_busy_until_ps_;
    std::map<NetworkToken, FakeSubmission> inflight_;
    std::vector<FakeSubmission> history_;
};

struct FakeV2NetworkConfig {
    // The wire the packets are put on. Serialization is exact rational
    // arithmetic, so a long run does not accumulate one truncation per packet.
    std::uint64_t link_bps{100000000000ULL};
    Picoseconds one_way_latency_ps{0};
    std::size_t capacity{4096};
};

// The ABI v2 counterpart of FakeNetworkPort: it serializes each packet
// attempt at a link rate, adds a fixed one-way latency, and acknowledges per
// packet. It follows the downstream packet-port contract that the transmit
// pipeline defines: it returns one token per attempt and reports TX finish,
// RX arrival and one terminal for it. It never stamps a TX start, because the
// endpoint's packetizer owns the issue instant.
class FakeV2NetworkPort final : public NetworkPort {
public:
    explicit FakeV2NetworkPort(FakeV2NetworkConfig config)
        : config_(config) {
        if (config_.link_bps == 0 || config_.capacity == 0) {
            throw std::invalid_argument(
                "fake v2 network needs a positive rate and capacity");
        }
    }

    NetworkPortCapabilities capabilities() const noexcept override {
        NetworkPortCapabilities caps;
        caps.abi_version = kNetworkPortAbiVersionV2;
        caps.packet_attempt_events = true;
        return caps;
    }

    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        if (descriptor.abi_version != kNetworkPortAbiVersionV2) {
            return NetworkSubmitResult::rejected();
        }
        if (inflight_ >= config_.capacity) {
            const auto retry = nextEventTime();
            if (!retry.has_value() || *retry <= now_ps) {
                throw std::logic_error(
                    "fake v2 network is full with nothing in flight to retire");
            }
            return NetworkSubmitResult::busy(*retry);
        }
        const NetworkToken token = next_token_++;
        const std::uint64_t wire_bytes =
            descriptor.payload_bytes + wire_header_bytes_;
        const Picoseconds start = std::max(now_ps, link_free_at_ps_);
        const Picoseconds finish = start + serialize(wire_bytes * 8);
        link_free_at_ps_ = finish;
        const Picoseconds arrival = finish + config_.one_way_latency_ps;
        const Picoseconds acknowledged = arrival + config_.one_way_latency_ps;

        const bool drop = drop_next_;
        drop_next_ = false;
        schedule(finish, token, descriptor.wqe_id,
                 NetworkEventKind::PacketTxFinished, false);
        if (!drop) {
            schedule(arrival, token, descriptor.wqe_id,
                     NetworkEventKind::PacketRxArrived, false);
            schedule(acknowledged, token, descriptor.wqe_id,
                     NetworkEventKind::Delivered, false);
        } else {
            schedule(arrival, token, descriptor.wqe_id,
                     NetworkEventKind::Dropped, true);
        }
        ++inflight_;
        ++accepted_;
        return NetworkSubmitResult::accepted(token);
    }

    void setWireHeaderBytes(std::uint64_t bytes) { wire_header_bytes_ = bytes; }
    void dropNext() noexcept { drop_next_ = true; }

    std::optional<Picoseconds> nextEventTime() const {
        if (pending_.empty()) {
            return std::nullopt;
        }
        return pending_.begin()->first.first;
    }

    std::vector<NetworkEvent> takeDue(Picoseconds now_ps) {
        std::vector<NetworkEvent> events;
        while (!pending_.empty() && pending_.begin()->first.first <= now_ps) {
            const NetworkEvent event = pending_.begin()->second;
            pending_.erase(pending_.begin());
            if (event.kind == NetworkEventKind::Delivered
                || event.kind == NetworkEventKind::Dropped) {
                --inflight_;
            }
            events.push_back(event);
        }
        return events;
    }

    std::size_t inflightCount() const noexcept { return inflight_; }
    std::uint64_t acceptedCount() const noexcept { return accepted_; }

private:
    Picoseconds serialize(std::uint64_t bits) {
        constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;
        const std::uint64_t numerator =
            bits * kPicosecondsPerSecond + remainder_;
        remainder_ = numerator % config_.link_bps;
        return numerator / config_.link_bps;
    }

    void schedule(
        Picoseconds when,
        NetworkToken token,
        WqeId wqe_id,
        NetworkEventKind kind,
        bool dropped) {
        NetworkEvent event;
        event.abi_version = kNetworkPortAbiVersionV2;
        event.scope = NetworkEventScope::PacketAttempt;
        event.kind = kind;
        event.token = token;
        event.wqe_id = wqe_id;
        event.event_time_ps = when;
        if (dropped) {
            event.drop_location = DropLocation::Fabric;
            event.drop_reason = DropReason::QueueOverflow;
            event.drop_evidence = DropEvidenceProvenance::Controlled;
            event.drop_resource_id = token;
        }
        pending_.emplace(std::make_pair(when, next_sequence_++), event);
    }

    FakeV2NetworkConfig config_;
    std::uint64_t wire_header_bytes_{64};
    NetworkToken next_token_{1};
    std::uint64_t next_sequence_{1};
    std::size_t inflight_{0};
    std::uint64_t accepted_{0};
    bool drop_next_{false};
    Picoseconds link_free_at_ps_{0};
    std::uint64_t remainder_{0};
    std::map<std::pair<Picoseconds, std::uint64_t>, NetworkEvent> pending_;
};

// How the wire loses a packet. Deterministic is a fixed one in every `period`
// attempts, which is what a replay guard needs; Bernoulli is a reproducible
// pseudo-random stream at a fixed rate in parts per million, which is what an
// incast needs. Neither is an endpoint behaviour: both stamp
// `DropLocation::Fabric` with `DropEvidenceProvenance::Controlled` so a reader
// can always separate an injected loss from a modelled one.
enum class FakeLossMode {
    None,
    Deterministic,
    Bernoulli,
};

struct FakeLossConfig {
    FakeLossMode mode{FakeLossMode::None};
    // Deterministic: one packet in every `period` is lost.
    std::uint64_t period{0};
    // Bernoulli: loss rate in parts per million.
    std::uint64_t rate_ppm{0};
    std::uint64_t seed{1};
};

// A reproducible loss generator. The Bernoulli stream is a 64-bit xorshift, so
// the same seed and the same packet order always lose the same packets and a
// study can replay a cell exactly.
class FakeLossSource {
public:
    explicit FakeLossSource(FakeLossConfig config)
        : config_(config), state_(config.seed == 0 ? 1 : config.seed) {}

    bool losesNext() {
        ++offered_;
        switch (config_.mode) {
        case FakeLossMode::None:
            return false;
        case FakeLossMode::Deterministic:
            return config_.period != 0 && offered_ % config_.period == 0;
        case FakeLossMode::Bernoulli:
            state_ ^= state_ << 13;
            state_ ^= state_ >> 7;
            state_ ^= state_ << 17;
            return (state_ % 1000000ULL) < config_.rate_ppm;
        }
        return false;
    }

    std::uint64_t offered() const noexcept { return offered_; }

private:
    FakeLossConfig config_;
    std::uint64_t state_;
    std::uint64_t offered_{0};
};

struct FakeDuplexLinkConfig {
    std::uint64_t link_bps{100000000000ULL};
    Picoseconds one_way_latency_ps{0};
    // Bytes the link may hold before it refuses an attempt with a busy
    // retry time. Zero leaves the link unbounded, which is the slice-B
    // behaviour of the single-ended fake.
    std::uint64_t queue_bytes{0};
};

// One direction of wire between two endpoints. It serializes with the same
// exact rational arithmetic the single-ended fake uses, so a long run does not
// accumulate one truncation per packet, and it reports its own standing queue
// so a test can see the delay that sets a go-back-N replay depth.
class FakeDuplexLink {
public:
    explicit FakeDuplexLink(FakeDuplexLinkConfig config) : config_(config) {
        if (config_.link_bps == 0) {
            throw std::invalid_argument("fake duplex link needs a rate");
        }
    }

    bool accepts(Picoseconds now_ps, std::uint64_t wire_bytes) const {
        if (config_.queue_bytes == 0) {
            return true;
        }
        return queuedBytesAt(now_ps) + wire_bytes <= config_.queue_bytes;
    }

    Picoseconds retryAt(Picoseconds now_ps) const {
        return free_at_ps_ > now_ps ? free_at_ps_ : now_ps + 1;
    }

    // Returns the instant the last bit leaves the wire.
    Picoseconds serialize(Picoseconds now_ps, std::uint64_t wire_bytes) {
        const Picoseconds start = std::max(now_ps, free_at_ps_);
        const std::uint64_t bits = wire_bytes * 8;
        constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;
        const std::uint64_t numerator =
            bits * kPicosecondsPerSecond + remainder_;
        remainder_ = numerator % config_.link_bps;
        free_at_ps_ = start + numerator / config_.link_bps;
        return free_at_ps_;
    }

    std::uint64_t queuedBytesAt(Picoseconds now_ps) const {
        if (free_at_ps_ <= now_ps) {
            return 0;
        }
        constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;
        const Picoseconds busy = free_at_ps_ - now_ps;
        return busy / 8 * config_.link_bps / kPicosecondsPerSecond;
    }

    Picoseconds oneWayLatencyPs() const noexcept {
        return config_.one_way_latency_ps;
    }

private:
    FakeDuplexLinkConfig config_;
    Picoseconds free_at_ps_{0};
    std::uint64_t remainder_{0};
};

struct FakeEgressQueueConfig {
    // The port the queue drains onto.
    std::uint64_t link_bps{100000000000ULL};
    // The tail-drop threshold in bytes. Zero leaves the queue unbounded, which
    // is the slice-C behaviour of a fabric that never drops on its own.
    std::uint64_t capacity_bytes{0};
};

// One switch egress port, as the measured leaf has it: a finite buffer that
// drains at the port rate and drops what does not fit, with no marking, no
// pause and no notification of any kind. That absence is the point. The
// campaign found zero congestion-experienced marks in 670 M packets with this
// buffer full and dropping, so a fabric that tail-drops in silence is the
// normal case here and the endpoint is the only thing left that can notice.
//
// The queue is a serializer with a bound: the instant it becomes free is the
// instant the last queued byte leaves, so the bytes it still owes are exactly
// that interval at the link rate, and an arrival that would push the total
// past the bound is dropped rather than delayed.
class FakeEgressQueue {
public:
    explicit FakeEgressQueue(FakeEgressQueueConfig config) : config_(config) {
        if (config_.link_bps == 0) {
            throw std::invalid_argument("fake egress queue needs a rate");
        }
    }

    std::uint64_t occupancyBytes(Picoseconds now_ps) const {
        if (free_at_ps_ <= now_ps) {
            return 0;
        }
        constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;
        return (free_at_ps_ - now_ps) / 8 * config_.link_bps
            / kPicosecondsPerSecond;
    }

    // Offers one packet. Returns the instant its last bit leaves the port, or
    // nothing when the buffer tail-drops it.
    std::optional<Picoseconds> offer(
        Picoseconds now_ps,
        std::uint64_t wire_bytes) {
        ++offered_;
        offered_bytes_ += wire_bytes;
        if (config_.capacity_bytes != 0
            && occupancyBytes(now_ps) + wire_bytes > config_.capacity_bytes) {
            ++dropped_;
            dropped_bytes_ += wire_bytes;
            return std::nullopt;
        }
        ++admitted_;
        admitted_bytes_ += wire_bytes;
        constexpr std::uint64_t kPicosecondsPerSecond = 1000000000000ULL;
        const Picoseconds start = std::max(now_ps, free_at_ps_);
        const std::uint64_t numerator =
            wire_bytes * 8 * kPicosecondsPerSecond + remainder_;
        remainder_ = numerator % config_.link_bps;
        free_at_ps_ = start + numerator / config_.link_bps;
        const std::uint64_t depth = occupancyBytes(now_ps);
        high_watermark_bytes_ = std::max(high_watermark_bytes_, depth);
        return free_at_ps_;
    }

    std::uint64_t offeredCount() const noexcept { return offered_; }
    std::uint64_t admittedCount() const noexcept { return admitted_; }
    std::uint64_t droppedCount() const noexcept { return dropped_; }
    std::uint64_t offeredBytes() const noexcept { return offered_bytes_; }
    std::uint64_t admittedBytes() const noexcept { return admitted_bytes_; }
    std::uint64_t droppedBytes() const noexcept { return dropped_bytes_; }
    std::uint64_t highWatermarkBytes() const noexcept {
        return high_watermark_bytes_;
    }
    Picoseconds freeAtPs() const noexcept { return free_at_ps_; }

private:
    FakeEgressQueueConfig config_;
    Picoseconds free_at_ps_{0};
    std::uint64_t remainder_{0};
    std::uint64_t offered_{0};
    std::uint64_t admitted_{0};
    std::uint64_t dropped_{0};
    std::uint64_t offered_bytes_{0};
    std::uint64_t admitted_bytes_{0};
    std::uint64_t dropped_bytes_{0};
    std::uint64_t high_watermark_bytes_{0};
};

struct FakeV2FabricConfig {
    FakeDuplexLinkConfig forward;
    FakeDuplexLinkConfig reverse;
    FakeLossConfig loss;
    std::uint64_t wire_header_bytes{64};
};

// The two-endpoint fake. A requester submits packet attempts on the forward
// link; each one either reaches the peer's receive pipeline or is lost in the
// fabric, and the peer's ACK or NAK comes back on the reverse link. The two
// links are separate objects, so a bidirectional test loads each direction
// independently instead of sharing one serializer.
//
// The requester's own port face is `port()`. Nothing about the peer leaks
// through it: the pipeline sees the same four-event lifecycle it sees from the
// single-ended fake.
class FakeV2Fabric final : public NetworkPort {
public:
    struct Arrival {
        Picoseconds when_ps{0};
        RnicRxPacket packet;
        NetworkToken token{0};
        bool lost{false};
    };

    struct Reply {
        Picoseconds when_ps{0};
        RnicTransportPacket packet;
    };

    explicit FakeV2Fabric(FakeV2FabricConfig config)
        : config_(config),
          forward_(config.forward),
          reverse_(config.reverse),
          loss_(config.loss) {}

    NetworkPortCapabilities capabilities() const noexcept override {
        NetworkPortCapabilities caps;
        caps.abi_version = kNetworkPortAbiVersionV2;
        caps.packet_attempt_events = true;
        return caps;
    }

    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        if (descriptor.abi_version != kNetworkPortAbiVersionV2) {
            return NetworkSubmitResult::rejected();
        }
        const std::uint64_t wire_bytes =
            descriptor.payload_bytes + config_.wire_header_bytes;
        if (!forward_.accepts(now_ps, wire_bytes)) {
            return NetworkSubmitResult::busy(forward_.retryAt(now_ps));
        }
        const NetworkToken token = next_token_++;
        const Picoseconds finish = forward_.serialize(now_ps, wire_bytes);
        schedule(finish, token, descriptor.wqe_id,
                 NetworkEventKind::PacketTxFinished);

        Arrival arrival;
        arrival.when_ps = finish + forward_.oneWayLatencyPs();
        arrival.token = token;
        arrival.lost = loss_.losesNext();
        arrival.packet.qpn = descriptor.qpn;
        arrival.packet.source = descriptor.source;
        arrival.packet.psn = descriptor.psn;
        arrival.packet.payload_bytes = descriptor.payload_bytes;
        arrival.packet.wire_bytes = wire_bytes;
        arrival.packet.service = RnicTransportService::ReliableConnected;
        arrival.packet.kind = descriptor.transmission_attempt == 0
            ? NetworkPacketKind::Data
            : NetworkPacketKind::Retransmission;
        arrival.packet.last_of_message =
            descriptor.extent_index + 1 == descriptor.extent_count;
        arrivals_.emplace(
            std::make_pair(arrival.when_ps, next_sequence_++), arrival);
        if (arrival.lost) {
            ++losses_;
        }
        ++accepted_;
        return NetworkSubmitResult::accepted(token);
    }

    // Hands every arrival due at `now_ps` to `responder` and records the ACK
    // or NAK it makes. A lost packet is never presented, which is exactly what
    // makes the responder learn of it only from the next one.
    void deliverDue(RnicRxPipeline& responder, Picoseconds now_ps) {
        while (!arrivals_.empty()
               && arrivals_.begin()->first.first <= now_ps) {
            const Arrival arrival = arrivals_.begin()->second;
            arrivals_.erase(arrivals_.begin());
            if (arrival.lost) {
                continue;
            }
            // The packet reached the receiving NIC. Whether the ingress meter
            // then keeps it is the next question, and a separate one.
            schedule(arrival.when_ps, arrival.token, 0,
                     NetworkEventKind::PacketRxArrived);
            const RnicRxResult result =
                responder.onPacket(arrival.packet, arrival.when_ps);
            // Any acknowledged packet retires its attempt, and the
            // responder acknowledges a duplicate as well as a delivery.
            if (result.has_reply
                && result.reply_kind == NetworkPacketKind::Ack) {
                delivered_tokens_.push_back(std::make_pair(
                    arrival.when_ps + reverse_.oneWayLatencyPs(),
                    arrival.token));
            }
            if (!result.has_reply
                || result.reply_kind != NetworkPacketKind::Nak) {
                continue;
            }
            Reply reply;
            const Picoseconds finish =
                reverse_.serialize(arrival.when_ps, result.reply_wire_bytes);
            reply.when_ps = finish + reverse_.oneWayLatencyPs();
            reply.packet.kind = NetworkPacketKind::Nak;
            reply.packet.qpn = arrival.packet.qpn;
            reply.packet.psn = result.reply_psn;
            replies_.emplace(
                std::make_pair(reply.when_ps, next_sequence_++), reply);
        }
    }

    std::vector<NetworkEvent> takeDue(Picoseconds now_ps) {
        std::vector<NetworkEvent> events;
        while (!pending_.empty() && pending_.begin()->first.first <= now_ps) {
            events.push_back(pending_.begin()->second);
            pending_.erase(pending_.begin());
        }
        for (auto entry = delivered_tokens_.begin();
             entry != delivered_tokens_.end();) {
            if (entry->first > now_ps) {
                ++entry;
                continue;
            }
            NetworkEvent event;
            event.abi_version = kNetworkPortAbiVersionV2;
            event.scope = NetworkEventScope::PacketAttempt;
            event.kind = NetworkEventKind::Delivered;
            event.token = entry->second;
            event.event_time_ps = entry->first;
            events.push_back(event);
            entry = delivered_tokens_.erase(entry);
        }
        return events;
    }

    std::vector<RnicTransportPacket> takeRepliesDue(Picoseconds now_ps) {
        std::vector<RnicTransportPacket> out;
        while (!replies_.empty() && replies_.begin()->first.first <= now_ps) {
            out.push_back(replies_.begin()->second.packet);
            replies_.erase(replies_.begin());
        }
        return out;
    }

    std::optional<Picoseconds> nextEventTime() const {
        std::optional<Picoseconds> next;
        const auto consider = [&next](Picoseconds when) {
            if (!next.has_value() || when < *next) {
                next = when;
            }
        };
        if (!pending_.empty()) {
            consider(pending_.begin()->first.first);
        }
        if (!arrivals_.empty()) {
            consider(arrivals_.begin()->first.first);
        }
        if (!replies_.empty()) {
            consider(replies_.begin()->first.first);
        }
        for (const auto& entry : delivered_tokens_) {
            consider(entry.first);
        }
        return next;
    }

    std::uint64_t acceptedCount() const noexcept { return accepted_; }
    std::uint64_t lossCount() const noexcept { return losses_; }
    bool idle() const noexcept {
        return pending_.empty() && arrivals_.empty() && replies_.empty()
            && delivered_tokens_.empty();
    }

private:
    void schedule(
        Picoseconds when,
        NetworkToken token,
        WqeId wqe_id,
        NetworkEventKind kind) {
        NetworkEvent event;
        event.abi_version = kNetworkPortAbiVersionV2;
        event.scope = NetworkEventScope::PacketAttempt;
        event.kind = kind;
        event.token = token;
        event.wqe_id = wqe_id;
        event.event_time_ps = when;
        pending_.emplace(std::make_pair(when, next_sequence_++), event);
    }

    FakeV2FabricConfig config_;
    FakeDuplexLink forward_;
    FakeDuplexLink reverse_;
    FakeLossSource loss_;
    NetworkToken next_token_{1};
    std::uint64_t next_sequence_{1};
    std::uint64_t accepted_{0};
    std::uint64_t losses_{0};
    std::map<std::pair<Picoseconds, std::uint64_t>, NetworkEvent> pending_;
    std::map<std::pair<Picoseconds, std::uint64_t>, Arrival> arrivals_;
    std::map<std::pair<Picoseconds, std::uint64_t>, Reply> replies_;
    std::vector<std::pair<Picoseconds, NetworkToken>> delivered_tokens_;
};

}  // namespace simllm::rnic::testing

#endif  // SIMLLM_RNIC_TESTS_FAKE_NETWORK_H
