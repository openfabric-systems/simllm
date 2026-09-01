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

}  // namespace simllm::rnic::testing

#endif  // SIMLLM_RNIC_TESTS_FAKE_NETWORK_H
