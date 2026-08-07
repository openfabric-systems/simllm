#include "simllm/rnic/work_queue.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace simllm::rnic {
namespace {

Picoseconds checkedAdd(Picoseconds lhs, Picoseconds rhs) {
    if (rhs > std::numeric_limits<Picoseconds>::max() - lhs) {
        throw std::overflow_error("RNIC timestamp overflow");
    }
    return lhs + rhs;
}

void validateServiceTime(Picoseconds value, const char* field_name) {
    constexpr Picoseconds max_service_time =
        static_cast<Picoseconds>(std::numeric_limits<std::int64_t>::max());
    if (value > max_service_time) {
        throw std::invalid_argument(
            std::string("RNIC ") + field_name
            + " must be between 0 and INT64_MAX ps");
    }
}

bool validDropLocation(DropLocation location) {
    switch (location) {
    case DropLocation::None:
    case DropLocation::TxPort:
    case DropLocation::Fabric:
    case DropLocation::RxPort:
        return true;
    default:
        return false;
    }
}

bool validDropReason(DropReason reason) {
    switch (reason) {
    case DropReason::None:
    case DropReason::Injected:
    case DropReason::QueueOverflow:
    case DropReason::LinkDown:
    case DropReason::PolicyRejected:
        return true;
    default:
        return false;
    }
}

}  // namespace

class WorkQueue::Impl {
public:
    Impl(WorkQueueConfig config, NetworkPort& network_port)
        : config_(std::move(config)), network_port_(network_port) {
        if (config_.version != kWorkQueueConfigVersion) {
            throw std::invalid_argument("unsupported RNIC work-queue config version");
        }
        if (config_.sq_depth == 0 || config_.cq_depth == 0) {
            throw std::invalid_argument("RNIC SQ and CQ depths must be positive");
        }
        if (config_.qpn == 0) {
            throw std::invalid_argument("RNIC QPN must be nonzero");
        }
        if (config_.policy_context_token == 0) {
            throw std::invalid_argument(
                "RNIC policy-context token must be nonzero");
        }
        validateServiceTime(
            config_.doorbell_service_ps, "doorbell_service_ps");
        validateServiceTime(
            config_.wqe_fetch_service_ps, "wqe_fetch_service_ps");
        validateServiceTime(
            config_.qpc_lookup_service_ps, "qpc_lookup_service_ps");
        validateServiceTime(
            config_.scheduler_service_ps, "scheduler_service_ps");
        validateServiceTime(
            config_.cqe_write_service_ps, "cqe_write_service_ps");
    }

    PostResult postSend(const WorkRequest& request, Picoseconds now_ps) {
        observeTime(now_ps);
        if (fatal_) {
            return PostResult{PostStatus::Fatal, 0, 0};
        }
        if (occupied_sq_entries_ == config_.sq_depth) {
            ++counters_.sq_full_rejections;
            evidence_.push_back(EvidenceEvent{
                EvidenceTier::Controlled,
                EvidenceKind::SqFull,
                0,
                request.wr_id,
                now_ps,
                DropLocation::None,
                DropReason::QueueOverflow,
            });
            return PostResult{PostStatus::SqFull, 0, 0};
        }

        const WqeId wqe_id = next_wqe_id_++;
        const std::uint64_t sq_sequence = next_sq_sequence_++;
        WqeRecord record;
        record.wqe_id = wqe_id;
        record.sq_sequence = sq_sequence;
        record.request = request;
        record.timeline.posted_at_ps = now_ps;
        records_.push_back(record);
        unpublished_.push_back(wqe_id);

        ++occupied_sq_entries_;
        ++counters_.posted_wqes;
        counters_.sq_high_watermark =
            std::max(counters_.sq_high_watermark, occupied_sq_entries_);
        return PostResult{PostStatus::Accepted, wqe_id, sq_sequence};
    }

    PostBatchResult postSendBatch(
        const std::vector<WorkRequest>& requests,
        Picoseconds now_ps) {
        PostBatchResult batch;
        batch.accepted.reserve(requests.size());
        for (std::size_t index = 0; index < requests.size(); ++index) {
            PostResult result = postSend(requests[index], now_ps);
            if (result.status != PostStatus::Accepted) {
                batch.status = result.status;
                batch.bad_wr_index = index;
                break;
            }
            batch.accepted.push_back(result);
        }
        return batch;
    }

    DoorbellBatch ringDoorbell(Picoseconds now_ps) {
        validateTime(now_ps);
        if (unpublished_.empty()) {
            last_observed_time_ps_ = now_ps;
            return DoorbellBatch{0, 0, now_ps, now_ps};
        }
        if (fatal_) {
            throw std::logic_error("cannot ring RNIC doorbell after fatal error");
        }

        if (next_batch_id_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("RNIC doorbell batch ID overflow");
        }
        const std::uint64_t batch_id = next_batch_id_;
        const Picoseconds doorbell_base = std::max(now_ps, doorbell_cursor_ps_);
        const Picoseconds observed_at =
            checkedAdd(doorbell_base, config_.doorbell_service_ps);
        const std::size_t count = unpublished_.size();

        struct PlannedDoorbellWqe {
            WqeId wqe_id{0};
            Picoseconds fetch_begin{0};
            Picoseconds fetch_end{0};
            Picoseconds qpc_ready{0};
            Picoseconds admitted{0};
        };
        std::vector<PlannedDoorbellWqe> plan;
        plan.reserve(count);
        std::deque<WqeId> planned_ready = ready_;
        Picoseconds planned_fetch_cursor = wqe_fetch_cursor_ps_;
        Picoseconds planned_scheduler_cursor = scheduler_cursor_ps_;
        for (const WqeId wqe_id : unpublished_) {
            const WqeRecord& record = wqe(wqe_id);
            if (record.state != WqeState::Posted) {
                throw std::logic_error(
                    "unpublished RNIC WQE is not in Posted state");
            }
            const Picoseconds fetch_begin =
                std::max(observed_at, planned_fetch_cursor);
            const Picoseconds fetch_end =
                checkedAdd(fetch_begin, config_.wqe_fetch_service_ps);
            planned_fetch_cursor = fetch_end;
            const Picoseconds qpc_ready =
                checkedAdd(fetch_end, config_.qpc_lookup_service_ps);
            const Picoseconds scheduler_begin =
                std::max(qpc_ready, planned_scheduler_cursor);
            const Picoseconds admitted =
                checkedAdd(scheduler_begin, config_.scheduler_service_ps);
            planned_scheduler_cursor = admitted;
            plan.push_back(PlannedDoorbellWqe{
                wqe_id, fetch_begin, fetch_end, qpc_ready, admitted});
            planned_ready.push_back(wqe_id);
        }

        for (const PlannedDoorbellWqe& item : plan) {
            WqeRecord& record = mutableWqe(item.wqe_id);
            record.doorbell_batch_id = batch_id;
            record.timeline.doorbelled_at_ps = now_ps;
            record.timeline.doorbell_seen_at_ps = observed_at;
            record.timeline.wqe_fetch_begin_at_ps = item.fetch_begin;
            record.timeline.wqe_fetch_end_at_ps = item.fetch_end;
            record.timeline.qpc_ready_at_ps = item.qpc_ready;
            record.timeline.admitted_at_ps = item.admitted;
            record.state = WqeState::Doorbelled;
        }
        ready_.swap(planned_ready);
        unpublished_.clear();
        ++next_batch_id_;
        last_observed_time_ps_ = now_ps;
        doorbell_cursor_ps_ = observed_at;
        wqe_fetch_cursor_ps_ = planned_fetch_cursor;
        scheduler_cursor_ps_ = planned_scheduler_cursor;
        ++counters_.doorbells;
        counters_.doorbelled_wqes += count;
        return DoorbellBatch{batch_id, count, now_ps, observed_at};
    }

    std::size_t progress(Picoseconds now_ps) {
        observeTime(now_ps);
        if (fatal_) {
            return 0;
        }
        std::size_t changes = publishDueCqes(now_ps);
        if (fatal_) {
            return changes;
        }

        if (network_blocked_) {
            if (!network_retry_at_ps_.has_value()
                || now_ps < *network_retry_at_ps_) {
                return changes;
            }
            network_blocked_ = false;
            network_retry_at_ps_.reset();
        }

        while (!ready_.empty()) {
            WqeRecord& record = mutableWqe(ready_.front());
            if (!record.timeline.admitted_at_ps.has_value()) {
                throw std::logic_error("doorbelled WQE has no admission time");
            }
            if (*record.timeline.admitted_at_ps > now_ps) {
                break;
            }

            NetworkTxDescriptor descriptor;
            descriptor.wqe_id = record.wqe_id;
            descriptor.wr_id = record.request.wr_id;
            descriptor.flow_id = record.request.flow_id;
            descriptor.flow_tag = record.request.flow_tag;
            descriptor.policy_context_token = config_.policy_context_token;
            descriptor.source = config_.source;
            descriptor.destination = record.request.destination;
            descriptor.qpn = config_.qpn;
            descriptor.traffic_class = record.request.traffic_class;
            descriptor.payload_bytes = record.request.payload_bytes;
            descriptor.eligible_at_ps = *record.timeline.admitted_at_ps;

            ++counters_.network_submit_attempts;
            const NetworkSubmitResult result =
                network_port_.trySubmit(descriptor, now_ps);
            switch (result.status) {
            case NetworkSubmitStatus::Busy:
                if (!result.has_retry_time || result.retry_at_ps <= now_ps) {
                    throw std::logic_error(
                        "busy RNIC network port must provide a future retry time");
                }
                if (result.token != 0
                    || result.rejection_location != DropLocation::None
                    || result.rejection_reason != DropReason::None) {
                    throw std::logic_error(
                        "busy RNIC network result carries contradictory fields");
                }
                ++counters_.network_busy;
                network_blocked_ = true;
                network_retry_at_ps_ = result.retry_at_ps;
                break;
            case NetworkSubmitStatus::Accepted:
                if (result.has_retry_time
                    || result.rejection_location != DropLocation::None
                    || result.rejection_reason != DropReason::None) {
                    throw std::logic_error(
                        "accepted RNIC network result carries contradictory fields");
                }
                if (result.token == 0 || inflight_.count(result.token) != 0) {
                    throw std::logic_error(
                        "RNIC network port returned an invalid or duplicate token");
                }
                ready_.pop_front();
                record.network_token = result.token;
                record.timeline.network_accepted_at_ps = now_ps;
                record.state = WqeState::InFlight;
                inflight_.emplace(result.token, record.wqe_id);
                ++counters_.network_accepted;
                ++changes;
                continue;
            case NetworkSubmitStatus::Rejected: {
                if (result.token != 0 || result.has_retry_time
                    || !validDropLocation(result.rejection_location)
                    || !validDropReason(result.rejection_reason)
                    || result.rejection_location == DropLocation::None
                    || result.rejection_reason == DropReason::None) {
                    throw std::logic_error(
                        "rejected RNIC network result lacks controlled evidence");
                }
                NetworkEvent event;
                event.kind = NetworkEventKind::Dropped;
                event.wqe_id = record.wqe_id;
                event.event_time_ps = now_ps;
                event.drop_location = result.rejection_location;
                event.drop_reason = result.rejection_reason;
                const CandidateOutcome candidate{
                    record.wqe_id,
                    PendingOutcome{
                        event, CompletionStatus::NetworkRejected}};
                RetirementPlan retirement_plan =
                    planRetirements(&candidate);
                evidence_.reserve(evidence_.size() + 1);
                const auto inserted = pending_outcomes_.emplace(
                    candidate.wqe_id, candidate.outcome);
                if (!inserted.second) {
                    throw std::logic_error(
                        "duplicate RNIC WQE network outcome");
                }

                ready_.pop_front();
                ++counters_.network_rejected;
                evidence_.push_back(EvidenceEvent{
                    EvidenceTier::Controlled,
                    EvidenceKind::NetworkRejected,
                    record.wqe_id,
                    record.request.wr_id,
                    now_ps,
                    result.rejection_location,
                    result.rejection_reason,
                });
                record.timeline.network_outcome_at_ps = now_ps;
                record.state = WqeState::AwaitingOrderedRetirement;
                commitRetirements(std::move(retirement_plan));
                ++changes;
                continue;
            }
            default:
                throw std::logic_error("invalid RNIC network submit status");
            }
            // Busy retains the SQ head and ends this progress pass.
            break;
        }

        changes += publishDueCqes(now_ps);
        return changes;
    }

    void onNetworkEvent(const NetworkEvent& event) {
        if (event.abi_version != kNetworkPortAbiVersion) {
            throw std::invalid_argument("unsupported RNIC network event ABI");
        }
        if (!validDropLocation(event.drop_location)
            || !validDropReason(event.drop_reason)) {
            throw std::invalid_argument(
                "RNIC network event carries an invalid drop enum");
        }
        switch (event.kind) {
        case NetworkEventKind::Delivered:
            if (event.drop_location != DropLocation::None
                || event.drop_reason != DropReason::None) {
                throw std::invalid_argument(
                    "delivered RNIC network event carries drop evidence");
            }
            break;
        case NetworkEventKind::Dropped:
            if (event.drop_location == DropLocation::None
                || event.drop_reason == DropReason::None) {
                throw std::invalid_argument(
                    "dropped RNIC network event lacks controlled evidence");
            }
            break;
        default:
            throw std::invalid_argument("invalid RNIC network event kind");
        }
        validateTime(event.event_time_ps);
        const auto inflight_it = inflight_.find(event.token);
        if (inflight_it == inflight_.end()) {
            throw std::logic_error("unknown or duplicate RNIC network token");
        }
        if (event.wqe_id != inflight_it->second) {
            throw std::logic_error("RNIC network token/WQE mismatch");
        }

        WqeRecord& record = mutableWqe(event.wqe_id);
        if (!record.timeline.network_accepted_at_ps.has_value()
            || event.event_time_ps
                < *record.timeline.network_accepted_at_ps) {
            throw std::logic_error("RNIC network event predates acceptance");
        }
        CompletionStatus status = CompletionStatus::Success;
        if (event.kind == NetworkEventKind::Dropped) {
            status = CompletionStatus::TransportError;
        }
        const CandidateOutcome candidate{
            record.wqe_id, PendingOutcome{event, status}};
        RetirementPlan retirement_plan = planRetirements(&candidate);
        if (event.kind == NetworkEventKind::Dropped) {
            evidence_.reserve(evidence_.size() + 1);
        }
        const auto inserted = pending_outcomes_.emplace(
            candidate.wqe_id, candidate.outcome);
        if (!inserted.second) {
            throw std::logic_error("duplicate RNIC WQE network outcome");
        }
        last_observed_time_ps_ = event.event_time_ps;
        inflight_.erase(inflight_it);
        record.timeline.network_outcome_at_ps = event.event_time_ps;
        record.state = WqeState::AwaitingOrderedRetirement;
        record.ecn_marked = record.ecn_marked || event.ecn_marked;
        if (event.kind == NetworkEventKind::Delivered) {
            ++counters_.network_delivered;
        } else {
            ++counters_.network_dropped;
            evidence_.push_back(EvidenceEvent{
                EvidenceTier::Controlled,
                EvidenceKind::NetworkDrop,
                record.wqe_id,
                record.request.wr_id,
                event.event_time_ps,
                event.drop_location,
                event.drop_reason,
            });
        }
        commitRetirements(std::move(retirement_plan));
    }

    std::vector<CompletionEntry> pollCompletionQueue(
        std::size_t max_entries,
        Picoseconds now_ps) {
        observeTime(now_ps);
        if (!fatal_) {
            publishCqesBefore(now_ps);
        }
        std::vector<CompletionEntry> result;
        result.reserve(std::min(max_entries, cq_.size()));
        while (result.size() < max_entries && !cq_.empty()) {
            CompletionEntry entry = cq_.front();
            cq_.pop_front();
            entry.polled_at_ps = now_ps;
            WqeRecord& record = mutableWqe(entry.wqe_id);
            record.timeline.polled_at_ps = now_ps;
            record.state = WqeState::Completed;
            reclaimThrough(entry.sq_sequence, now_ps);
            ++counters_.cqes_polled;
            result.push_back(entry);
        }
        return result;
    }

    std::optional<Picoseconds> nextEventTime() const {
        if (fatal_) {
            return std::nullopt;
        }
        std::optional<Picoseconds> next;
        if (!pending_cqes_.empty()) {
            next = std::max(
                pending_cqes_.begin()->first.first,
                last_observed_time_ps_);
        }
        if (!ready_.empty()) {
            const WqeRecord& record = wqe(ready_.front());
            if (!record.timeline.admitted_at_ps.has_value()) {
                throw std::logic_error("ready RNIC WQE has no admission time");
            }
            Picoseconds ready_at = std::max(
                *record.timeline.admitted_at_ps,
                last_observed_time_ps_);
            if (network_blocked_) {
                if (!network_retry_at_ps_.has_value()) {
                    return next;
                }
                ready_at = std::max(ready_at, *network_retry_at_ps_);
            }
            if (!next.has_value() || ready_at < *next) {
                next = ready_at;
            }
        }
        return next;
    }

    bool hasPendingPhysicalWork() const noexcept {
        return fatal_ || !unpublished_.empty() || !ready_.empty()
            || !inflight_.empty() || !pending_outcomes_.empty()
            || !pending_cqes_.empty() || !cq_.empty();
    }

    bool fatal() const noexcept { return fatal_; }
    std::size_t occupiedSqEntries() const noexcept {
        return occupied_sq_entries_;
    }
    std::size_t completionQueueDepth() const noexcept { return cq_.size(); }
    std::size_t unpublishedWqeCount() const noexcept {
        return unpublished_.size();
    }
    const WorkQueueConfig& config() const noexcept { return config_; }
    const WorkQueueCounters& counters() const noexcept { return counters_; }
    const std::vector<WqeRecord>& records() const noexcept { return records_; }
    const std::vector<EvidenceEvent>& evidence() const noexcept {
        return evidence_;
    }

    const WqeRecord& wqe(WqeId wqe_id) const {
        if (wqe_id == 0 || wqe_id > records_.size()) {
            throw std::out_of_range("unknown RNIC WQE ID");
        }
        const WqeRecord& record = records_[static_cast<std::size_t>(wqe_id - 1)];
        if (record.wqe_id != wqe_id) {
            throw std::logic_error("RNIC WQE index invariant failed");
        }
        return record;
    }

    void validateInvariants() const {
        if (occupied_sq_entries_ > config_.sq_depth) {
            throw std::logic_error("RNIC SQ occupancy exceeds depth");
        }
        if (cq_.size() > config_.cq_depth) {
            throw std::logic_error("RNIC CQ occupancy exceeds depth");
        }
        if (counters_.posted_wqes != records_.size()) {
            throw std::logic_error("RNIC posted counter disagrees with records");
        }

        std::vector<std::size_t> owner_counts(records_.size(), 0);
        const auto mark_owned = [this, &owner_counts](
                                    WqeId wqe_id,
                                    WqeState expected_state) {
            const WqeRecord& record = wqe(wqe_id);
            if (record.state != expected_state) {
                throw std::logic_error(
                    "RNIC WQE state disagrees with its owner container");
            }
            std::size_t& count = owner_counts[static_cast<std::size_t>(
                wqe_id - 1)];
            ++count;
            if (count != 1) {
                throw std::logic_error(
                    "RNIC WQE appears in multiple owner containers");
            }
        };
        for (const WqeId wqe_id : unpublished_) {
            mark_owned(wqe_id, WqeState::Posted);
        }
        for (const WqeId wqe_id : ready_) {
            mark_owned(wqe_id, WqeState::Doorbelled);
        }
        for (const auto& token_and_wqe : inflight_) {
            const WqeRecord& record = wqe(token_and_wqe.second);
            if (!record.network_token.has_value()
                || *record.network_token != token_and_wqe.first) {
                throw std::logic_error(
                    "RNIC in-flight token accounting mismatch");
            }
            mark_owned(token_and_wqe.second, WqeState::InFlight);
        }
        for (const auto& wqe_and_outcome : pending_outcomes_) {
            if (wqe_and_outcome.first
                != wqe_and_outcome.second.event.wqe_id) {
                throw std::logic_error(
                    "RNIC pending outcome/WQE identity mismatch");
            }
            mark_owned(
                wqe_and_outcome.first,
                WqeState::AwaitingOrderedRetirement);
        }
        for (const auto& key_and_entry : pending_cqes_) {
            const CompletionEntry& entry = key_and_entry.second;
            if (key_and_entry.first.first != entry.visible_at_ps
                || key_and_entry.first.second != entry.cqe_sequence) {
                throw std::logic_error(
                    "RNIC pending CQE key disagrees with its entry");
            }
            mark_owned(entry.wqe_id, WqeState::CompletionPending);
        }
        for (const CompletionEntry& entry : cq_) {
            mark_owned(entry.wqe_id, WqeState::CqeVisible);
        }

        std::size_t unreclaimed = 0;
        std::size_t reclaimed = 0;
        for (std::size_t index = 0; index < records_.size(); ++index) {
            const WqeRecord& record = records_[index];
            if (record.wqe_id != index + 1 || record.sq_sequence != index + 1) {
                throw std::logic_error("RNIC WQE identity/sequence is not monotonic");
            }
            if (record.timeline.sq_reclaimed_at_ps.has_value()) {
                ++reclaimed;
                if (!record.timeline.transport_retired_at_ps.has_value()) {
                    throw std::logic_error("RNIC reclaimed an unretired WQE");
                }
            } else {
                ++unreclaimed;
            }
            bool needs_owner = false;
            switch (record.state) {
            case WqeState::Posted:
            case WqeState::Doorbelled:
            case WqeState::InFlight:
            case WqeState::AwaitingOrderedRetirement:
            case WqeState::CompletionPending:
            case WqeState::CqeVisible:
                needs_owner = true;
                break;
            case WqeState::RetiredUnsignaled:
            case WqeState::Reclaimed:
            case WqeState::Completed:
            case WqeState::Error:
                break;
            default:
                throw std::logic_error("invalid RNIC WQE state");
            }
            if (owner_counts[index] != (needs_owner ? 1U : 0U)) {
                throw std::logic_error(
                    "RNIC WQE owner-container conservation failed");
            }
        }
        if (unreclaimed != occupied_sq_entries_
            || reclaimed != counters_.sq_reclaimed_wqes) {
            throw std::logic_error("RNIC SQ reclaim accounting mismatch");
        }
        if (counters_.cqes_visible < counters_.cqes_polled
            || counters_.cqes_visible - counters_.cqes_polled != cq_.size()) {
            throw std::logic_error("RNIC CQ producer/consumer accounting mismatch");
        }
        if (fatal_ != fatal_lost_sq_sequence_.has_value()) {
            throw std::logic_error("RNIC fatal CQ-overrun accounting mismatch");
        }
        if (fatal_lost_sq_sequence_.has_value()) {
            const WqeRecord& lost =
                wqe(static_cast<WqeId>(*fatal_lost_sq_sequence_));
            if (lost.state != WqeState::Error
                || lost.timeline.sq_reclaimed_at_ps.has_value()) {
                throw std::logic_error("RNIC reclaimed its first lost CQE");
            }
        }
    }

private:
    struct PendingOutcome {
        NetworkEvent event;
        CompletionStatus status{CompletionStatus::Success};
    };

    using PendingCqeKey = std::pair<Picoseconds, std::uint64_t>;

    struct CandidateOutcome {
        WqeId wqe_id{0};
        PendingOutcome outcome;
    };

    struct PlannedRetirement {
        WqeId wqe_id{0};
        Picoseconds retired_at_ps{0};
        CompletionStatus status{CompletionStatus::Success};
        bool completion_pending{false};
    };

    struct RetirementPlan {
        std::vector<PlannedRetirement> retirements;
        std::map<PendingCqeKey, CompletionEntry> pending_cqes;
        Picoseconds retirement_cursor_ps{0};
        Picoseconds cqe_write_cursor_ps{0};
        std::uint64_t next_cqe_sequence{1};
        std::uint64_t next_retire_sequence{1};
    };

    void validateTime(Picoseconds now_ps) const {
        if (now_ps < last_observed_time_ps_) {
            throw std::logic_error("RNIC model time regressed");
        }
    }

    void observeTime(Picoseconds now_ps) {
        validateTime(now_ps);
        last_observed_time_ps_ = now_ps;
    }

    WqeRecord& mutableWqe(WqeId wqe_id) {
        return const_cast<WqeRecord&>(wqe(wqe_id));
    }

    CompletionEntry makeCqe(
        const WqeRecord& record,
        CompletionStatus status,
        Picoseconds retired_at_ps,
        Picoseconds cqe_write_cursor_ps,
        std::uint64_t cqe_sequence) const {
        CompletionEntry entry;
        entry.cqe_sequence = cqe_sequence;
        entry.wr_id = record.request.wr_id;
        entry.wqe_id = record.wqe_id;
        entry.sq_sequence = record.sq_sequence;
        entry.qpn = config_.qpn;
        entry.opcode = record.request.opcode;
        entry.status = status;
        entry.valid_fields = kCqeValidWrId | kCqeValidQpn
            | kCqeValidOpcode | kCqeValidStatus | kCqeValidVendorSyndrome;
        // SEND byte count is not a valid raw CQE field. Keep it zero and do
        // not set kCqeValidByteCount; payload size remains in the WQE record.
        entry.byte_count = 0;
        if (status == CompletionStatus::TransportError) {
            entry.vendor_syndrome = 1;
        } else if (status == CompletionStatus::NetworkRejected) {
            entry.vendor_syndrome = 2;
        }
        const Picoseconds write_begin =
            std::max(retired_at_ps, cqe_write_cursor_ps);
        entry.visible_at_ps =
            checkedAdd(write_begin, config_.cqe_write_service_ps);
        return entry;
    }

    RetirementPlan planRetirements(
        const CandidateOutcome* candidate = nullptr) const {
        RetirementPlan plan;
        plan.retirement_cursor_ps = retirement_cursor_ps_;
        plan.cqe_write_cursor_ps = cqe_write_cursor_ps_;
        plan.next_cqe_sequence = next_cqe_sequence_;
        plan.next_retire_sequence = next_retire_sequence_;
        if (next_retire_sequence_ <= records_.size()) {
            plan.retirements.reserve(
                records_.size()
                - static_cast<std::size_t>(next_retire_sequence_) + 1);
        }

        while (plan.next_retire_sequence <= records_.size()) {
            const WqeRecord& record =
                records_[plan.next_retire_sequence - 1];
            const PendingOutcome* outcome = nullptr;
            const bool is_candidate =
                candidate != nullptr && candidate->wqe_id == record.wqe_id;
            if (is_candidate) {
                if (pending_outcomes_.count(record.wqe_id) != 0) {
                    throw std::logic_error(
                        "duplicate RNIC WQE network outcome");
                }
                outcome = &candidate->outcome;
            } else {
                const auto outcome_it =
                    pending_outcomes_.find(record.wqe_id);
                if (outcome_it == pending_outcomes_.end()) {
                    break;
                }
                outcome = &outcome_it->second;
            }

            if (is_candidate) {
                if (record.timeline.network_outcome_at_ps.has_value()) {
                    throw std::logic_error(
                        "candidate RNIC retirement already has an outcome");
                }
            } else if (record.state
                           != WqeState::AwaitingOrderedRetirement
                       || !record.timeline.network_outcome_at_ps.has_value()
                       || *record.timeline.network_outcome_at_ps
                           != outcome->event.event_time_ps) {
                throw std::logic_error(
                    "RNIC retirement lacks its recorded network outcome");
            }

            const Picoseconds retired_at = std::max(
                outcome->event.event_time_ps,
                plan.retirement_cursor_ps);
            const bool completion_pending = record.request.signaled
                || outcome->status != CompletionStatus::Success;
            if (completion_pending) {
                if (plan.next_cqe_sequence
                    == std::numeric_limits<std::uint64_t>::max()) {
                    throw std::overflow_error(
                        "RNIC CQE sequence overflow");
                }
                CompletionEntry entry = makeCqe(
                    record,
                    outcome->status,
                    retired_at,
                    plan.cqe_write_cursor_ps,
                    plan.next_cqe_sequence);
                const PendingCqeKey key{
                    entry.visible_at_ps, entry.cqe_sequence};
                if (pending_cqes_.count(key) != 0
                    || !plan.pending_cqes.emplace(key, entry).second) {
                    throw std::logic_error(
                        "duplicate RNIC pending CQE key");
                }
                plan.cqe_write_cursor_ps = entry.visible_at_ps;
                ++plan.next_cqe_sequence;
            }
            plan.retirement_cursor_ps = retired_at;
            plan.retirements.push_back(PlannedRetirement{
                record.wqe_id,
                retired_at,
                outcome->status,
                completion_pending,
            });
            ++plan.next_retire_sequence;
        }
        return plan;
    }

    void commitRetirements(RetirementPlan&& plan) {
        pending_cqes_.merge(plan.pending_cqes);
        for (const PlannedRetirement& item : plan.retirements) {
            pending_outcomes_.erase(item.wqe_id);
            WqeRecord& record = mutableWqe(item.wqe_id);
            record.timeline.transport_retired_at_ps = item.retired_at_ps;
            record.completion_status = item.status;
            record.state = item.completion_pending
                ? WqeState::CompletionPending
                : WqeState::RetiredUnsignaled;
        }
        retirement_cursor_ps_ = plan.retirement_cursor_ps;
        cqe_write_cursor_ps_ = plan.cqe_write_cursor_ps;
        next_cqe_sequence_ = plan.next_cqe_sequence;
        next_retire_sequence_ = plan.next_retire_sequence;
    }

    std::size_t publishCqes(
        Picoseconds boundary_ps,
        bool include_boundary) {
        std::size_t published = 0;
        while (!pending_cqes_.empty()) {
            const Picoseconds visible_at =
                pending_cqes_.begin()->first.first;
            if (visible_at > boundary_ps
                || (!include_boundary && visible_at == boundary_ps)) {
                break;
            }
            CompletionEntry entry = pending_cqes_.begin()->second;
            pending_cqes_.erase(pending_cqes_.begin());
            WqeRecord& record = mutableWqe(entry.wqe_id);
            if (cq_.size() == config_.cq_depth) {
                fatal_ = true;
                fatal_lost_sq_sequence_ = entry.sq_sequence;
                ++counters_.cq_overruns;
                evidence_.push_back(EvidenceEvent{
                    EvidenceTier::Controlled,
                    EvidenceKind::CqOverrun,
                    record.wqe_id,
                    record.request.wr_id,
                    entry.visible_at_ps,
                    DropLocation::None,
                    DropReason::QueueOverflow,
                });
                record.state = WqeState::Error;
                break;
            }

            entry.cq_producer_index = counters_.cqes_visible;
            entry.cq_slot = static_cast<std::size_t>(
                entry.cq_producer_index % config_.cq_depth);
            entry.owner_generation = static_cast<std::uint8_t>(
                (entry.cq_producer_index / config_.cq_depth) & 1U);
            record.timeline.cqe_visible_at_ps = entry.visible_at_ps;
            record.state = WqeState::CqeVisible;
            cq_.push_back(entry);
            ++counters_.cqes_visible;
            counters_.cq_high_watermark =
                std::max(counters_.cq_high_watermark, cq_.size());
            ++published;
        }
        return published;
    }

    std::size_t publishDueCqes(Picoseconds now_ps) {
        return publishCqes(now_ps, true);
    }

    std::size_t publishCqesBefore(Picoseconds now_ps) {
        return publishCqes(now_ps, false);
    }

    void reclaimThrough(std::uint64_t sq_sequence, Picoseconds now_ps) {
        if (sq_sequence < sq_reclaimed_through_) {
            throw std::logic_error("RNIC SQ reclaim sequence regressed");
        }
        if (fatal_lost_sq_sequence_.has_value()
            && sq_sequence >= *fatal_lost_sq_sequence_) {
            throw std::logic_error(
                "RNIC CQ poll cannot reclaim through a lost CQE");
        }
        while (sq_reclaimed_through_ < sq_sequence) {
            WqeRecord& record = records_[sq_reclaimed_through_];
            if (!record.timeline.transport_retired_at_ps.has_value()) {
                throw std::logic_error("RNIC CQE attempted to reclaim live WQE");
            }
            if (!record.timeline.sq_reclaimed_at_ps.has_value()) {
                record.timeline.sq_reclaimed_at_ps = now_ps;
                if (record.state != WqeState::Completed) {
                    record.state = WqeState::Reclaimed;
                }
                --occupied_sq_entries_;
                ++counters_.sq_reclaimed_wqes;
            }
            ++sq_reclaimed_through_;
        }
    }

    WorkQueueConfig config_;
    NetworkPort& network_port_;
    WorkQueueCounters counters_;
    std::vector<WqeRecord> records_;
    std::vector<EvidenceEvent> evidence_;
    std::deque<WqeId> unpublished_;
    std::deque<WqeId> ready_;
    std::map<NetworkToken, WqeId> inflight_;
    std::map<WqeId, PendingOutcome> pending_outcomes_;
    std::map<PendingCqeKey, CompletionEntry> pending_cqes_;
    std::deque<CompletionEntry> cq_;

    WqeId next_wqe_id_{1};
    std::uint64_t next_sq_sequence_{1};
    std::uint64_t next_batch_id_{1};
    std::uint64_t next_cqe_sequence_{1};
    std::uint64_t next_retire_sequence_{1};
    std::uint64_t sq_reclaimed_through_{0};
    std::size_t occupied_sq_entries_{0};

    Picoseconds last_observed_time_ps_{0};
    Picoseconds doorbell_cursor_ps_{0};
    Picoseconds wqe_fetch_cursor_ps_{0};
    Picoseconds scheduler_cursor_ps_{0};
    Picoseconds retirement_cursor_ps_{0};
    Picoseconds cqe_write_cursor_ps_{0};
    bool network_blocked_{false};
    std::optional<Picoseconds> network_retry_at_ps_;
    bool fatal_{false};
    std::optional<std::uint64_t> fatal_lost_sq_sequence_;
};

WorkQueue::WorkQueue(WorkQueueConfig config, NetworkPort& network_port)
    : impl_(std::make_unique<Impl>(std::move(config), network_port)) {}

WorkQueue::~WorkQueue() = default;
WorkQueue::WorkQueue(WorkQueue&&) noexcept = default;
WorkQueue& WorkQueue::operator=(WorkQueue&&) noexcept = default;

PostResult WorkQueue::postSend(
    const WorkRequest& request,
    Picoseconds now_ps) {
    return impl_->postSend(request, now_ps);
}

PostBatchResult WorkQueue::postSendBatch(
    const std::vector<WorkRequest>& requests,
    Picoseconds now_ps) {
    return impl_->postSendBatch(requests, now_ps);
}

DoorbellBatch WorkQueue::ringDoorbell(Picoseconds now_ps) {
    return impl_->ringDoorbell(now_ps);
}

std::size_t WorkQueue::progress(Picoseconds now_ps) {
    return impl_->progress(now_ps);
}

void WorkQueue::onNetworkEvent(const NetworkEvent& event) {
    impl_->onNetworkEvent(event);
}

std::vector<CompletionEntry> WorkQueue::pollCompletionQueue(
    std::size_t max_entries,
    Picoseconds now_ps) {
    return impl_->pollCompletionQueue(max_entries, now_ps);
}

std::optional<Picoseconds> WorkQueue::nextEventTime() const {
    return impl_->nextEventTime();
}

bool WorkQueue::hasPendingPhysicalWork() const noexcept {
    return impl_->hasPendingPhysicalWork();
}

bool WorkQueue::fatal() const noexcept { return impl_->fatal(); }

std::size_t WorkQueue::occupiedSqEntries() const noexcept {
    return impl_->occupiedSqEntries();
}

std::size_t WorkQueue::completionQueueDepth() const noexcept {
    return impl_->completionQueueDepth();
}

std::size_t WorkQueue::unpublishedWqeCount() const noexcept {
    return impl_->unpublishedWqeCount();
}

const WorkQueueConfig& WorkQueue::config() const noexcept {
    return impl_->config();
}

const WorkQueueCounters& WorkQueue::counters() const noexcept {
    return impl_->counters();
}

const std::vector<WqeRecord>& WorkQueue::records() const noexcept {
    return impl_->records();
}

const std::vector<EvidenceEvent>& WorkQueue::evidence() const noexcept {
    return impl_->evidence();
}

const WqeRecord& WorkQueue::wqe(WqeId wqe_id) const {
    return impl_->wqe(wqe_id);
}

void WorkQueue::validateInvariants() const {
    impl_->validateInvariants();
}

}  // namespace simllm::rnic
