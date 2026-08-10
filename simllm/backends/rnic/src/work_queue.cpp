#include "simllm/rnic/work_queue.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "simllm/rnic/pcie_fabric.h"

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

std::uint32_t ringByteOffset(
    std::uint32_t base_offset,
    std::uint64_t sequence,
    std::size_t depth,
    std::uint64_t entry_bytes) {
    if (sequence == 0 || depth == 0) {
        throw std::logic_error("invalid RNIC ring-offset input");
    }
    const std::uint64_t slot = (sequence - 1) % depth;
    const std::uint64_t offset = (
        base_offset
        + (slot % 4096) * (entry_bytes % 4096))
        % 4096;
    return static_cast<std::uint32_t>(offset);
}

std::uint64_t ringAllocationOffset(
    std::uint64_t sequence,
    std::size_t depth,
    std::uint64_t entry_bytes) {
    if (sequence == 0 || depth == 0 || entry_bytes == 0) {
        throw std::logic_error("invalid RNIC ring-allocation offset input");
    }
    const std::uint64_t slot = (sequence - 1) % depth;
    if (slot > std::numeric_limits<std::uint64_t>::max() / entry_bytes) {
        throw std::overflow_error("RNIC ring-allocation offset overflow");
    }
    return slot * entry_bytes;
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
    Impl(
        WorkQueueConfig config,
        NetworkPort& network_port,
        PcieFabric* pcie_fabric,
        std::optional<WorkQueuePcieBinding> pcie_binding,
        bool qpc_lookup_enabled,
        VirtualHostMemory* host_memory,
        std::optional<WorkQueueHostMemoryBinding> host_memory_binding)
        : config_(std::move(config)),
          network_port_(network_port),
          pcie_fabric_(pcie_fabric),
          pcie_binding_(std::move(pcie_binding)),
          qpc_lookup_enabled_(qpc_lookup_enabled),
          host_memory_(host_memory),
          host_memory_binding_(std::move(host_memory_binding)) {
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
        if (!qpc_lookup_enabled_ && config_.qpc_lookup_service_ps != 0) {
            throw std::invalid_argument(
                "RNIC disabled QPC cannot charge scalar lookup service");
        }
        if (pcie_fabric_ != nullptr) {
            if (!pcie_binding_.has_value()
                || pcie_binding_->version
                    != kWorkQueuePcieBindingVersion) {
                throw std::invalid_argument(
                    "unsupported RNIC WQ PCIe binding version");
            }
            if (config_.doorbell_service_ps != 0
                || config_.wqe_fetch_service_ps != 0
                || config_.cqe_write_service_ps != 0) {
                throw std::invalid_argument(
                    "RNIC PCIe mode cannot also charge doorbell, WQE-fetch "
                    "or CQE-write scalar service");
            }
            if (config_.sq_id == 0 || config_.cq_id == 0
                || config_.sq_id
                    > (std::numeric_limits<std::uint64_t>::max() >> 1)
                || config_.cq_id
                    > (std::numeric_limits<std::uint64_t>::max() >> 1)) {
                throw std::invalid_argument(
                    "RNIC PCIe mode requires nonzero 63-bit SQ and CQ IDs");
            }
            if (pcie_binding_->pcie_submission_ordering_domain == 0) {
                pcie_binding_->pcie_submission_ordering_domain =
                    (config_.sq_id << 1) | 1;
            }
            if (pcie_binding_->pcie_completion_ordering_domain == 0) {
                pcie_binding_->pcie_completion_ordering_domain =
                    config_.cq_id << 1;
            }
            if (pcie_binding_->pcie_doorbell_record_bytes == 0
                || pcie_binding_->pcie_uar_doorbell_bytes == 0
                || pcie_binding_->pcie_wqe_bytes == 0
                || pcie_binding_->pcie_cqe_bytes == 0) {
                throw std::invalid_argument(
                    "RNIC PCIe WQ transfer sizes must be positive");
            }
            if (pcie_binding_->pcie_uar_first_byte_offset >= 4096
                || pcie_binding_->pcie_doorbell_record_first_byte_offset
                    >= 4096
                || pcie_binding_->pcie_sq_first_byte_offset >= 4096
                || pcie_binding_->pcie_cq_first_byte_offset >= 4096) {
                throw std::invalid_argument(
                    "RNIC PCIe WQ byte offsets must be below 4096");
            }
            const PcieFabricConfig pcie_config = pcie_fabric_->config();
            const auto find_enabled_path = [&pcie_config](
                                               std::uint32_t path_id) {
                const auto path = std::find_if(
                    pcie_config.paths.begin(),
                    pcie_config.paths.end(),
                    [path_id](const PciePathConfig& path) {
                        return path.path_id == path_id && path.enabled;
                    });
                return path == pcie_config.paths.end() ? nullptr : &*path;
            };
            const PciePathConfig* uar_path = find_enabled_path(
                pcie_binding_->pcie_uar_path_id);
            const PciePathConfig* db_path = find_enabled_path(
                pcie_binding_->pcie_doorbell_record_path_id);
            const PciePathConfig* sq_path = find_enabled_path(
                pcie_binding_->pcie_sq_memory_path_id);
            const PciePathConfig* cq_path = find_enabled_path(
                pcie_binding_->pcie_cq_memory_path_id);
            if (uar_path == nullptr || db_path == nullptr
                || sq_path == nullptr || cq_path == nullptr) {
                throw std::invalid_argument(
                    "RNIC PCIe WQ references an unknown or disabled path");
            }
            if (uar_path->endpoint != PcieEndpointKind::MmioBar
                || db_path->endpoint != PcieEndpointKind::HostPinnedMemory
                || sq_path->endpoint != PcieEndpointKind::HostPinnedMemory
                || cq_path->endpoint != PcieEndpointKind::HostPinnedMemory) {
                throw std::invalid_argument(
                    "RNIC PCIe WQ path endpoint kind is incompatible");
            }
        }
        if (host_memory_ != nullptr) {
            if (pcie_fabric_ == nullptr || !pcie_binding_.has_value()
                || !host_memory_binding_.has_value()
                || host_memory_binding_->version
                    != kWorkQueueHostMemoryBindingVersion) {
                throw std::invalid_argument(
                    "RNIC host-memory WQ requires PCIe and a valid binding");
            }
            if (!qpc_lookup_enabled_ || config_.qpc_lookup_service_ps != 0) {
                throw std::invalid_argument(
                    "RNIC host-memory QPC requires enabled non-scalar lookup");
            }
            const auto& binding = *host_memory_binding_;
            if (binding.qpc_icm_allocation_id == 0
                || binding.sq_ring_allocation_id == 0
                || binding.rq_ring_allocation_id == 0
                || binding.cq_ring_allocation_id == 0
                || binding.doorbell_record_allocation_id == 0
                || binding.qpc_context_bytes == 0) {
                throw std::invalid_argument(
                    "RNIC host-memory WQ allocation identities must be positive");
            }
        } else if (host_memory_binding_.has_value()) {
            throw std::invalid_argument(
                "RNIC host-memory binding requires an attached registry");
        }
    }

    PostResult postSend(const WorkRequest& request, Picoseconds now_ps) {
        validateRequestMemory(request);
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
        const std::size_t count = unpublished_.size();

        std::optional<PcieFabric::Plan> pcie_plan;
        std::vector<HostMemoryAccessRecord> planned_memory_accesses;
        Picoseconds observed_at = 0;
        if (pcie_fabric_ != nullptr) {
            pcie_plan.emplace(pcie_fabric_->beginPlan());
            PcieTransactionResult db_result;
            if (host_memory_ != nullptr) {
                HostMemoryAccessRequest db_record;
                db_record.allocation_id =
                    host_memory_binding_->doorbell_record_allocation_id;
                db_record.client_id = config_.qpn;
                db_record.client_token = batch_id;
                db_record.service_class = PcieServiceClass::DoorbellRecord;
                db_record.operation = PcieOperation::HostStore;
                db_record.request_direction = PcieDirection::HostToDevice;
                db_record.ordering = PcieOrdering::VisibilityDependency;
                db_record.ordering_domain =
                    pcie_binding_->pcie_submission_ordering_domain;
                db_record.useful_bytes =
                    pcie_binding_->pcie_doorbell_record_bytes;
                db_record.transfer_bytes =
                    pcie_binding_->pcie_doorbell_record_bytes;
                db_record.submitted_at_ps = now_ps;
                HostMemoryAccessResult result = host_memory_->scheduleAccess(
                    *pcie_fabric_, *pcie_plan, db_record);
                db_result = result.access_transaction;
                planned_memory_accesses.push_back(std::move(result.record));
            } else {
                PcieTransactionRequest db_record;
                db_record.client_id = config_.qpn;
                db_record.client_token = batch_id;
                db_record.service_class = PcieServiceClass::DoorbellRecord;
                db_record.operation = PcieOperation::HostStore;
                db_record.request_direction = PcieDirection::HostToDevice;
                db_record.ordering = PcieOrdering::VisibilityDependency;
                db_record.path_id =
                    pcie_binding_->pcie_doorbell_record_path_id;
                db_record.ordering_domain =
                    pcie_binding_->pcie_submission_ordering_domain;
                db_record.useful_bytes =
                    pcie_binding_->pcie_doorbell_record_bytes;
                db_record.transfer_bytes =
                    pcie_binding_->pcie_doorbell_record_bytes;
                db_record.first_byte_offset =
                    pcie_binding_->pcie_doorbell_record_first_byte_offset;
                db_record.submitted_at_ps = now_ps;
                db_result = pcie_fabric_->schedule(*pcie_plan, db_record);
            }

            PcieTransactionRequest uar;
            uar.client_id = config_.qpn;
            uar.client_token = batch_id;
            uar.service_class = PcieServiceClass::UarDoorbell;
            uar.operation = PcieOperation::PostedWrite;
            uar.request_direction = PcieDirection::HostToDevice;
            uar.ordering = PcieOrdering::VisibilityDependency;
            uar.path_id = pcie_binding_->pcie_uar_path_id;
            uar.ordering_domain =
                pcie_binding_->pcie_submission_ordering_domain;
            uar.useful_bytes = pcie_binding_->pcie_uar_doorbell_bytes;
            uar.transfer_bytes = pcie_binding_->pcie_uar_doorbell_bytes;
            uar.first_byte_offset =
                pcie_binding_->pcie_uar_first_byte_offset;
            uar.submitted_at_ps = db_result.completed_at_ps;
            observed_at = pcie_fabric_->schedule(
                *pcie_plan, uar).completed_at_ps;
        } else {
            const Picoseconds doorbell_base = std::max(
                now_ps, doorbell_cursor_ps_);
            observed_at = checkedAdd(
                doorbell_base, config_.doorbell_service_ps);
        }

        struct PlannedDoorbellWqe {
            WqeId wqe_id{0};
            Picoseconds fetch_begin{0};
            Picoseconds fetch_end{0};
            std::optional<Picoseconds> qpc_ready;
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
            Picoseconds fetch_begin = 0;
            Picoseconds fetch_end = 0;
            if (pcie_fabric_ != nullptr) {
                PcieTransactionResult fetch;
                if (host_memory_ != nullptr) {
                    HostMemoryAccessRequest wqe_read;
                    wqe_read.allocation_id =
                        host_memory_binding_->sq_ring_allocation_id;
                    wqe_read.client_id = config_.qpn;
                    wqe_read.client_token = wqe_id;
                    wqe_read.service_class = PcieServiceClass::WqeRead;
                    wqe_read.operation = PcieOperation::NonPostedRead;
                    wqe_read.request_direction = PcieDirection::DeviceToHost;
                    wqe_read.ordering = PcieOrdering::Independent;
                    wqe_read.ordering_domain =
                        pcie_binding_->pcie_submission_ordering_domain;
                    wqe_read.allocation_offset_bytes = ringAllocationOffset(
                        record.sq_sequence,
                        config_.sq_depth,
                        pcie_binding_->pcie_wqe_bytes);
                    wqe_read.useful_bytes = pcie_binding_->pcie_wqe_bytes;
                    wqe_read.transfer_bytes = pcie_binding_->pcie_wqe_bytes;
                    wqe_read.submitted_at_ps = observed_at;
                    HostMemoryAccessResult result =
                        host_memory_->scheduleAccess(
                            *pcie_fabric_, *pcie_plan, wqe_read);
                    fetch = result.access_transaction;
                    planned_memory_accesses.push_back(
                        std::move(result.record));
                } else {
                    PcieTransactionRequest wqe_read;
                    wqe_read.client_id = config_.qpn;
                    wqe_read.client_token = wqe_id;
                    wqe_read.service_class = PcieServiceClass::WqeRead;
                    wqe_read.operation = PcieOperation::NonPostedRead;
                    wqe_read.request_direction = PcieDirection::DeviceToHost;
                    wqe_read.ordering = PcieOrdering::Independent;
                    wqe_read.path_id =
                        pcie_binding_->pcie_sq_memory_path_id;
                    wqe_read.ordering_domain =
                        pcie_binding_->pcie_submission_ordering_domain;
                    wqe_read.useful_bytes = pcie_binding_->pcie_wqe_bytes;
                    wqe_read.transfer_bytes = pcie_binding_->pcie_wqe_bytes;
                    wqe_read.first_byte_offset = ringByteOffset(
                        pcie_binding_->pcie_sq_first_byte_offset,
                        record.sq_sequence,
                        config_.sq_depth,
                        pcie_binding_->pcie_wqe_bytes);
                    wqe_read.submitted_at_ps = observed_at;
                    fetch = pcie_fabric_->schedule(*pcie_plan, wqe_read);
                }
                fetch_begin = fetch.first_issue_at_ps;
                fetch_end = fetch.completed_at_ps;
                planned_fetch_cursor = std::max(
                    planned_fetch_cursor, fetch_end);
            } else {
                fetch_begin = std::max(
                    observed_at, planned_fetch_cursor);
                fetch_end = checkedAdd(
                    fetch_begin, config_.wqe_fetch_service_ps);
                planned_fetch_cursor = fetch_end;
            }
            std::optional<Picoseconds> qpc_ready;
            Picoseconds data_ready = fetch_end;
            if (qpc_lookup_enabled_) {
                if (host_memory_ != nullptr) {
                    HostMemoryAccessRequest qpc_read;
                    qpc_read.allocation_id =
                        host_memory_binding_->qpc_icm_allocation_id;
                    qpc_read.client_id = config_.qpn;
                    qpc_read.client_token = wqe_id;
                    qpc_read.service_class = PcieServiceClass::QpcIcm;
                    qpc_read.operation = PcieOperation::NonPostedRead;
                    qpc_read.request_direction = PcieDirection::DeviceToHost;
                    qpc_read.ordering = PcieOrdering::Independent;
                    qpc_read.ordering_domain =
                        pcie_binding_->pcie_submission_ordering_domain;
                    qpc_read.useful_bytes =
                        host_memory_binding_->qpc_context_bytes;
                    qpc_read.transfer_bytes =
                        host_memory_binding_->qpc_context_bytes;
                    qpc_read.submitted_at_ps = fetch_end;
                    HostMemoryAccessResult result =
                        host_memory_->scheduleAccess(
                            *pcie_fabric_, *pcie_plan, qpc_read);
                    qpc_ready = result.access_transaction.completed_at_ps;
                    planned_memory_accesses.push_back(
                        std::move(result.record));
                } else {
                    qpc_ready = checkedAdd(
                        fetch_end, config_.qpc_lookup_service_ps);
                }
                data_ready = *qpc_ready;
            }
            if (host_memory_ != nullptr && record.request.payload_bytes != 0) {
                const WorkRequestDataMemory& data =
                    *record.request.data_memory;
                HostMemoryAccessRequest payload_read;
                payload_read.allocation_id = data.allocation_id;
                payload_read.mkey = data.mkey;
                payload_read.client_id = config_.qpn;
                payload_read.client_token = wqe_id;
                payload_read.service_class = PcieServiceClass::PayloadRead;
                payload_read.operation = PcieOperation::NonPostedRead;
                payload_read.request_direction = PcieDirection::DeviceToHost;
                payload_read.ordering = PcieOrdering::Independent;
                payload_read.ordering_domain =
                    pcie_binding_->pcie_submission_ordering_domain;
                payload_read.allocation_offset_bytes =
                    data.allocation_offset_bytes;
                payload_read.useful_bytes = record.request.payload_bytes;
                payload_read.transfer_bytes = record.request.payload_bytes;
                payload_read.submitted_at_ps = data_ready;
                HostMemoryAccessResult result = host_memory_->scheduleAccess(
                    *pcie_fabric_, *pcie_plan, payload_read);
                data_ready = result.access_transaction.completed_at_ps;
                planned_memory_accesses.push_back(std::move(result.record));
            }
            const Picoseconds scheduler_begin =
                std::max(
                    data_ready,
                    planned_scheduler_cursor);
            const Picoseconds admitted =
                checkedAdd(scheduler_begin, config_.scheduler_service_ps);
            planned_scheduler_cursor = admitted;
            plan.push_back(PlannedDoorbellWqe{
                wqe_id, fetch_begin, fetch_end, qpc_ready, admitted});
            planned_ready.push_back(wqe_id);
        }

        if (pcie_plan.has_value()) {
            memory_accesses_.reserve(
                memory_accesses_.size() + planned_memory_accesses.size());
            pcie_fabric_->commit(std::move(*pcie_plan));
            for (HostMemoryAccessRecord& access : planned_memory_accesses) {
                memory_accesses_.push_back(std::move(access));
            }
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
                try {
                    commitRetirementPcie(retirement_plan);
                } catch (...) {
                    pending_outcomes_.erase(candidate.wqe_id);
                    throw;
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
        try {
            commitRetirementPcie(retirement_plan);
        } catch (...) {
            pending_outcomes_.erase(candidate.wqe_id);
            throw;
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
    std::optional<WorkQueuePcieBinding> pcieBinding() const {
        return pcie_binding_;
    }
    const WorkQueueCounters& counters() const noexcept { return counters_; }
    const std::vector<WqeRecord>& records() const noexcept { return records_; }
    const std::vector<EvidenceEvent>& evidence() const noexcept {
        return evidence_;
    }
    const std::vector<HostMemoryAccessRecord>& memoryAccesses() const noexcept {
        return memory_accesses_;
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
        if (host_memory_ == nullptr && !memory_accesses_.empty()) {
            throw std::logic_error(
                "RNIC compatibility WQ retained host-memory accesses");
        }
        for (const HostMemoryAccessRecord& access : memory_accesses_) {
            if (access.version != kHostMemoryAccessRecordVersion
                || access.allocation_id == 0
                || access.access_transaction_id == 0
                || access.completed_at_ps < access.submitted_at_ps) {
                throw std::logic_error(
                    "RNIC host-memory access record is invalid");
            }
            if (access.translation_transaction_ids.size()
                > access.translation_stages.size()) {
                throw std::logic_error(
                    "RNIC host-memory translation projection is invalid");
            }
            if (access.object_kind == HostMemoryObjectKind::QpcIcm
                && (!access.translation_stages.empty()
                    || !access.translation_transaction_ids.empty())) {
                throw std::logic_error(
                    "RNIC QPC access consumed a translation event");
            }
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
        std::optional<PcieFabric::Plan> pcie_plan;
        std::vector<HostMemoryAccessRecord> memory_accesses;
        Picoseconds retirement_cursor_ps{0};
        Picoseconds cqe_write_cursor_ps{0};
        std::uint64_t next_cqe_sequence{1};
        std::uint64_t next_retire_sequence{1};
    };

    void validateRequestMemory(const WorkRequest& request) const {
        if (host_memory_ == nullptr) {
            if (request.data_memory.has_value()) {
                throw std::invalid_argument(
                    "RNIC data-memory descriptor requires host memory");
            }
            return;
        }
        if (request.payload_bytes == 0) {
            if (request.data_memory.has_value()) {
                throw std::invalid_argument(
                    "zero-byte RNIC WQE cannot carry data memory");
            }
            return;
        }
        if (!request.data_memory.has_value()
            || request.data_memory->version != kWorkRequestDataMemoryVersion
            || request.data_memory->allocation_id == 0
            || request.data_memory->mkey == 0) {
            throw std::invalid_argument(
                "RNIC host-memory WQE requires a valid data descriptor");
        }
        const HostMemoryAllocation& allocation = host_memory_->allocation(
            request.data_memory->allocation_id);
        if (allocation.object_kind != HostMemoryObjectKind::DataRegion
            || allocation.mkey != request.data_memory->mkey) {
            throw std::invalid_argument(
                "RNIC WQE data descriptor does not match its allocation");
        }
        if (request.data_memory->allocation_offset_bytes
                > allocation.length_bytes
            || request.payload_bytes
                > allocation.length_bytes
                    - request.data_memory->allocation_offset_bytes) {
            throw std::out_of_range(
                "RNIC WQE data descriptor exceeds its allocation");
        }
    }

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
        std::uint64_t cqe_sequence,
        RetirementPlan& retirement_plan) const {
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
        if (pcie_fabric_ != nullptr) {
            if (!retirement_plan.pcie_plan.has_value()) {
                retirement_plan.pcie_plan.emplace(
                    pcie_fabric_->beginPlan());
            }
            if (host_memory_ != nullptr) {
                HostMemoryAccessRequest cqe_write;
                cqe_write.allocation_id =
                    host_memory_binding_->cq_ring_allocation_id;
                cqe_write.client_id = config_.qpn;
                cqe_write.client_token = record.wqe_id;
                cqe_write.service_class = PcieServiceClass::CqeWrite;
                cqe_write.operation = PcieOperation::PostedWrite;
                cqe_write.request_direction = PcieDirection::DeviceToHost;
                cqe_write.ordering = PcieOrdering::VisibilityDependency;
                cqe_write.ordering_domain =
                    pcie_binding_->pcie_completion_ordering_domain;
                cqe_write.allocation_offset_bytes = ringAllocationOffset(
                    cqe_sequence,
                    config_.cq_depth,
                    pcie_binding_->pcie_cqe_bytes);
                cqe_write.useful_bytes = pcie_binding_->pcie_cqe_bytes;
                cqe_write.transfer_bytes = pcie_binding_->pcie_cqe_bytes;
                cqe_write.submitted_at_ps = retired_at_ps;
                HostMemoryAccessResult result = host_memory_->scheduleAccess(
                    *pcie_fabric_, *retirement_plan.pcie_plan, cqe_write);
                entry.visible_at_ps =
                    result.access_transaction.completed_at_ps;
                retirement_plan.memory_accesses.push_back(
                    std::move(result.record));
            } else {
                PcieTransactionRequest cqe_write;
                cqe_write.client_id = config_.qpn;
                cqe_write.client_token = record.wqe_id;
                cqe_write.service_class = PcieServiceClass::CqeWrite;
                cqe_write.operation = PcieOperation::PostedWrite;
                cqe_write.request_direction = PcieDirection::DeviceToHost;
                cqe_write.ordering = PcieOrdering::VisibilityDependency;
                cqe_write.path_id =
                    pcie_binding_->pcie_cq_memory_path_id;
                cqe_write.ordering_domain =
                    pcie_binding_->pcie_completion_ordering_domain;
                cqe_write.useful_bytes = pcie_binding_->pcie_cqe_bytes;
                cqe_write.transfer_bytes = pcie_binding_->pcie_cqe_bytes;
                cqe_write.first_byte_offset = ringByteOffset(
                    pcie_binding_->pcie_cq_first_byte_offset,
                    cqe_sequence,
                    config_.cq_depth,
                    pcie_binding_->pcie_cqe_bytes);
                cqe_write.submitted_at_ps = retired_at_ps;
                entry.visible_at_ps = pcie_fabric_->schedule(
                    *retirement_plan.pcie_plan,
                    cqe_write).completed_at_ps;
            }
        } else {
            const Picoseconds write_begin =
                std::max(retired_at_ps, cqe_write_cursor_ps);
            entry.visible_at_ps =
                checkedAdd(write_begin, config_.cqe_write_service_ps);
        }
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
                    plan.next_cqe_sequence,
                    plan);
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

    void commitRetirementPcie(RetirementPlan& plan) {
        if (plan.pcie_plan.has_value()) {
            memory_accesses_.reserve(
                memory_accesses_.size() + plan.memory_accesses.size());
            pcie_fabric_->commit(std::move(*plan.pcie_plan));
            plan.pcie_plan.reset();
            for (HostMemoryAccessRecord& access : plan.memory_accesses) {
                memory_accesses_.push_back(std::move(access));
            }
            plan.memory_accesses.clear();
        }
    }

    void commitRetirements(RetirementPlan&& plan) {
        if (plan.pcie_plan.has_value()) {
            throw std::logic_error(
                "RNIC retirement PCIe plan was not committed first");
        }
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
    PcieFabric* pcie_fabric_{nullptr};
    std::optional<WorkQueuePcieBinding> pcie_binding_;
    bool qpc_lookup_enabled_{true};
    VirtualHostMemory* host_memory_{nullptr};
    std::optional<WorkQueueHostMemoryBinding> host_memory_binding_;
    WorkQueueCounters counters_;
    std::vector<WqeRecord> records_;
    std::vector<EvidenceEvent> evidence_;
    std::vector<HostMemoryAccessRecord> memory_accesses_;
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
    : WorkQueue(
          std::move(config),
          network_port,
          nullptr,
          std::nullopt,
          true,
          nullptr,
          std::nullopt) {}

WorkQueue::WorkQueue(
    WorkQueueConfig config,
    NetworkPort& network_port,
    PcieFabric& pcie_fabric,
    WorkQueuePcieBinding pcie_binding)
    : WorkQueue(
          std::move(config),
          network_port,
          &pcie_fabric,
          std::move(pcie_binding),
          true,
          nullptr,
          std::nullopt) {}

WorkQueue::WorkQueue(
    WorkQueueConfig config,
    NetworkPort& network_port,
    PcieFabric* pcie_fabric,
    std::optional<WorkQueuePcieBinding> pcie_binding,
    bool qpc_lookup_enabled,
    VirtualHostMemory* host_memory,
    std::optional<WorkQueueHostMemoryBinding> host_memory_binding)
    : impl_(std::make_unique<Impl>(
          std::move(config),
          network_port,
          pcie_fabric,
          std::move(pcie_binding),
          qpc_lookup_enabled,
          host_memory,
          std::move(host_memory_binding))) {}

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

std::optional<WorkQueuePcieBinding> WorkQueue::pcieBinding() const {
    return impl_->pcieBinding();
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

const std::vector<HostMemoryAccessRecord>&
WorkQueue::memoryAccesses() const noexcept {
    return impl_->memoryAccesses();
}

const WqeRecord& WorkQueue::wqe(WqeId wqe_id) const {
    return impl_->wqe(wqe_id);
}

void WorkQueue::validateInvariants() const {
    impl_->validateInvariants();
}

}  // namespace simllm::rnic
