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

private:
    std::size_t capacity_;
    Picoseconds latency_ps_;
    NetworkToken next_token_{1};
    bool reject_next_{false};
    std::optional<Picoseconds> forced_busy_until_ps_;
    std::map<NetworkToken, FakeSubmission> inflight_;
    std::vector<FakeSubmission> history_;
};

}  // namespace simllm::rnic::testing

#endif  // SIMLLM_RNIC_TESTS_FAKE_NETWORK_H
