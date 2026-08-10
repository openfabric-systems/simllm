#include "simllm/rnic/rnic_device.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace simllm::rnic {
namespace {

void validateAnalyticalProfileVersion(
    const PcieAnalyticalDelayProfile& profile) {
    if (profile.version != kPcieAnalyticalDelayProfileVersion) {
        throw std::invalid_argument(
            "unsupported RNIC PCIe analytical profile version");
    }
}

void validatePcieSubconfigVersions(const PcieFabricConfig& config) {
    if (config.version != kPcieFabricConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC PCIe fabric config version");
    }
    for (const PciePathConfig& path : config.paths) {
        const PciePathPenaltyProfiles& profiles =
            path.analytical_penalties;
        validateAnalyticalProfileVersion(profiles.numa);
        validateAnalyticalProfileVersion(profiles.iommu);
        validateAnalyticalProfileVersion(profiles.acs);
        validateAnalyticalProfileVersion(profiles.switch_path);
        validateAnalyticalProfileVersion(profiles.ddio_miss);
        validateAnalyticalProfileVersion(profiles.gpu_direct);
    }
}

std::optional<Picoseconds> earlier(
    std::optional<Picoseconds> lhs,
    std::optional<Picoseconds> rhs) {
    if (!lhs.has_value()) {
        return rhs;
    }
    if (!rhs.has_value()) {
        return lhs;
    }
    return std::min(*lhs, *rhs);
}

}  // namespace

class RnicDevice::InertNetworkPort final : public NetworkPort {
public:
    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        if (descriptor.abi_version != kNetworkPortAbiVersion) {
            return NetworkSubmitResult::rejected();
        }
        if (next_token_ == 0
            || next_token_ == std::numeric_limits<NetworkToken>::max()) {
            throw std::overflow_error("RNIC inert network token overflow");
        }
        const NetworkToken token = next_token_++;
        const auto inserted = pending_.emplace(
            token, Pending{descriptor.wqe_id, now_ps});
        if (!inserted.second) {
            throw std::logic_error("duplicate RNIC inert network token");
        }
        return NetworkSubmitResult::accepted(token);
    }

    std::optional<Picoseconds> nextEventTime() const {
        std::optional<Picoseconds> next;
        for (const auto& token_and_pending : pending_) {
            const Picoseconds event_time = token_and_pending.second.event_time;
            if (!next.has_value() || event_time < *next) {
                next = event_time;
            }
        }
        return next;
    }

    std::optional<NetworkEvent> nextDue(Picoseconds now_ps) const {
        for (const auto& token_and_pending : pending_) {
            if (token_and_pending.second.event_time > now_ps) {
                continue;
            }
            NetworkEvent event;
            event.token = token_and_pending.first;
            event.wqe_id = token_and_pending.second.wqe_id;
            event.event_time_ps = token_and_pending.second.event_time;
            return event;
        }
        return std::nullopt;
    }

    void consume(NetworkToken token) {
        if (pending_.erase(token) != 1) {
            throw std::logic_error("unknown RNIC inert network token");
        }
    }

private:
    struct Pending {
        WqeId wqe_id{0};
        Picoseconds event_time{0};
    };

    NetworkToken next_token_{1};
    std::map<NetworkToken, Pending> pending_;
};

RnicDevice::RnicDevice(
    RnicDeviceConfig config,
    RnicDeviceAttachments attachments)
    : config_(std::move(config)) {
    if (config_.version != kRnicDeviceConfigVersion) {
        throw std::invalid_argument("unsupported RNIC device config version");
    }
    if (config_.identity.version != kRnicDeviceIdentityVersion) {
        throw std::invalid_argument("unsupported RNIC device identity version");
    }
    if (config_.qpc.version != kRnicQpcConfigVersion) {
        throw std::invalid_argument("unsupported RNIC QPC config version");
    }
    if (config_.dma.version != kRnicDmaConfigVersion) {
        throw std::invalid_argument("unsupported RNIC DMA config version");
    }
    if (config_.network.version != kRnicNetworkConfigVersion) {
        throw std::invalid_argument("unsupported RNIC network config version");
    }
    if (config_.work_queue.version != kWorkQueueConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC work-queue config version");
    }
    if (config_.dma.work_queue.version
        != kWorkQueuePcieBindingVersion) {
        throw std::invalid_argument(
            "unsupported RNIC WQ PCIe binding version");
    }
    validatePcieSubconfigVersions(config_.dma.fabric);

    if (config_.identity.qpn == 0
        || config_.identity.policy_context_token == 0) {
        throw std::invalid_argument(
            "RNIC device identity fields must be nonzero");
    }
    if (config_.identity.qpn != config_.work_queue.qpn
        || config_.identity.policy_context_token
            != config_.work_queue.policy_context_token) {
        throw std::invalid_argument(
            "RNIC work-queue identity must match the device identity");
    }
    if (!config_.qpc.enabled
        && config_.work_queue.qpc_lookup_service_ps != 0) {
        throw std::invalid_argument(
            "RNIC disabled QPC cannot charge scalar lookup service");
    }
    if (config_.dma.enabled
        && (config_.work_queue.doorbell_service_ps != 0
            || config_.work_queue.wqe_fetch_service_ps != 0
            || config_.work_queue.cqe_write_service_ps != 0)) {
        throw std::invalid_argument(
            "RNIC DMA mode cannot also charge doorbell, WQE-fetch or "
            "CQE-write scalar service");
    }
    if (!config_.dma.enabled && attachments.shared_pcie_fabric) {
        throw std::invalid_argument(
            "RNIC DMA-disabled device cannot attach a PCIe fabric");
    }
    if (config_.network.enabled && attachments.network_port == nullptr) {
        throw std::invalid_argument(
            "RNIC network-enabled device requires an external port");
    }
    if (!config_.network.enabled && attachments.network_port != nullptr) {
        throw std::invalid_argument(
            "RNIC network-disabled device rejects an external port");
    }

    if (config_.qpc.enabled) {
        stage_report_.qpc_lookup = RnicStageApplicability::Applicable;
    } else {
        stage_report_.qpc_lookup = RnicStageApplicability::NotApplicable;
    }
    if (config_.dma.enabled) {
        stage_report_.scalar_doorbell_service =
            RnicStageApplicability::NotApplicable;
        stage_report_.scalar_wqe_fetch_service =
            RnicStageApplicability::NotApplicable;
        stage_report_.scalar_cqe_write_service =
            RnicStageApplicability::NotApplicable;
        stage_report_.pcie_doorbell_record =
            RnicStageApplicability::Applicable;
        stage_report_.pcie_uar_doorbell =
            RnicStageApplicability::Applicable;
        stage_report_.pcie_wqe_read =
            RnicStageApplicability::Applicable;
        stage_report_.pcie_cqe_write =
            RnicStageApplicability::Applicable;
    }
    if (config_.network.enabled) {
        stage_report_.external_network = RnicStageApplicability::Applicable;
        stage_report_.inert_network = RnicStageApplicability::NotApplicable;
        network_port_ = attachments.network_port;
    } else {
        inert_network_port_ = std::make_unique<InertNetworkPort>();
        network_port_ = inert_network_port_.get();
    }

    std::optional<WorkQueuePcieBinding> pcie_binding;
    if (config_.dma.enabled) {
        pcie_binding = config_.dma.work_queue;
        if (attachments.shared_pcie_fabric) {
            pcie_fabric_ = std::move(attachments.shared_pcie_fabric);
            const bool needs_namespace =
                pcie_binding->pcie_submission_ordering_domain == 0
                || pcie_binding->pcie_completion_ordering_domain == 0;
            if (needs_namespace) {
                const std::uint64_t name_space =
                    config_.dma.shared_ordering_domain_namespace;
                if (name_space == 0
                    || name_space
                        > (std::numeric_limits<std::uint64_t>::max() >> 1)) {
                    throw std::invalid_argument(
                        "shared RNIC PCIe fabric requires a nonzero 63-bit "
                        "ordering-domain namespace");
                }
                if (pcie_binding->pcie_submission_ordering_domain == 0) {
                    pcie_binding->pcie_submission_ordering_domain =
                        (name_space << 1) | 1;
                }
                if (pcie_binding->pcie_completion_ordering_domain == 0) {
                    pcie_binding->pcie_completion_ordering_domain =
                        name_space << 1;
                }
            } else if (config_.dma.shared_ordering_domain_namespace != 0) {
                throw std::invalid_argument(
                    "explicit shared RNIC PCIe domains reject a namespace");
            }
            pcie_fabric_->claimOrderingDomains(
                pcie_binding->pcie_submission_ordering_domain,
                pcie_binding->pcie_completion_ordering_domain);
            claimed_ordering_domains_ = true;
            claimed_submission_domain_ =
                pcie_binding->pcie_submission_ordering_domain;
            claimed_completion_domain_ =
                pcie_binding->pcie_completion_ordering_domain;
        } else {
            if (config_.dma.shared_ordering_domain_namespace != 0) {
                throw std::invalid_argument(
                    "owned RNIC PCIe fabric rejects a shared namespace");
            }
            pcie_fabric_ = std::make_shared<PcieFabric>(config_.dma.fabric);
        }
    }

    try {
        work_queue_.reset(new WorkQueue(
            config_.work_queue,
            *network_port_,
            pcie_fabric_.get(),
            std::move(pcie_binding),
            config_.qpc.enabled));
    } catch (...) {
        if (claimed_ordering_domains_) {
            pcie_fabric_->releaseOrderingDomains(
                claimed_submission_domain_,
                claimed_completion_domain_);
            claimed_ordering_domains_ = false;
        }
        throw;
    }
}

RnicDevice::~RnicDevice() {
    work_queue_.reset();
    if (claimed_ordering_domains_) {
        pcie_fabric_->releaseOrderingDomains(
            claimed_submission_domain_,
            claimed_completion_domain_);
    }
}

PostResult RnicDevice::postSend(
    const WorkRequest& request,
    Picoseconds now_ps) {
    observeCallerTime(now_ps);
    return work_queue_->postSend(request, now_ps);
}

PostBatchResult RnicDevice::postSendBatch(
    const std::vector<WorkRequest>& requests,
    Picoseconds now_ps) {
    observeCallerTime(now_ps);
    return work_queue_->postSendBatch(requests, now_ps);
}

DoorbellBatch RnicDevice::ringDoorbell(Picoseconds now_ps) {
    observeCallerTime(now_ps);
    return work_queue_->ringDoorbell(now_ps);
}

void RnicDevice::onNetworkEvent(const NetworkEvent& event) {
    if (!config_.network.enabled) {
        throw std::logic_error(
            "external RNIC network event supplied to the inert port");
    }
    observeCallerTime(event.event_time_ps);
    work_queue_->onNetworkEvent(event);
}

std::size_t RnicDevice::progress(Picoseconds now_ps) {
    observeCallerTime(now_ps);
    if (config_.network.enabled) {
        return work_queue_->progress(now_ps);
    }

    std::size_t changes = 0;
    for (;;) {
        while (const auto event = inert_network_port_->nextDue(now_ps)) {
            work_queue_->onNetworkEvent(*event);
            inert_network_port_->consume(event->token);
            ++changes;
        }
        changes += work_queue_->progress(now_ps);
        const std::optional<Picoseconds> next_network_event =
            inert_network_port_->nextEventTime();
        if (!next_network_event.has_value()
            || *next_network_event > now_ps) {
            break;
        }
    }
    return changes;
}

std::vector<CompletionEntry> RnicDevice::pollCompletionQueue(
    std::size_t max_entries,
    Picoseconds now_ps) {
    observeCallerTime(now_ps);
    return work_queue_->pollCompletionQueue(max_entries, now_ps);
}

PcieTransactionResult RnicDevice::submitPcie(
    const PcieTransactionRequest& request) {
    if (!config_.dma.enabled || !pcie_fabric_) {
        throw std::logic_error(
            "RNIC device has no enabled PCIe fabric");
    }
    observeCallerTime(request.submitted_at_ps);
    return pcie_fabric_->submit(request);
}

std::optional<Picoseconds> RnicDevice::nextEventTime() const {
    const std::optional<Picoseconds> queue_time =
        work_queue_->nextEventTime();
    if (config_.network.enabled) {
        return queue_time;
    }
    return earlier(queue_time, inert_network_port_->nextEventTime());
}

bool RnicDevice::hasPendingPhysicalWork() const noexcept {
    return work_queue_->hasPendingPhysicalWork();
}

bool RnicDevice::fatal() const noexcept {
    return work_queue_->fatal();
}

std::size_t RnicDevice::occupiedSqEntries() const noexcept {
    return work_queue_->occupiedSqEntries();
}

std::size_t RnicDevice::completionQueueDepth() const noexcept {
    return work_queue_->completionQueueDepth();
}

std::size_t RnicDevice::unpublishedWqeCount() const noexcept {
    return work_queue_->unpublishedWqeCount();
}

const RnicDeviceConfig& RnicDevice::config() const noexcept {
    return config_;
}

const RnicDeviceStageReport& RnicDevice::stageReport() const noexcept {
    return stage_report_;
}

std::optional<WorkQueuePcieBinding> RnicDevice::pcieBinding() const {
    return work_queue_->pcieBinding();
}

const PcieFabric* RnicDevice::pcieFabric() const noexcept {
    return pcie_fabric_.get();
}

const WorkQueueConfig& RnicDevice::workQueueConfig() const noexcept {
    return work_queue_->config();
}

const WorkQueueCounters& RnicDevice::counters() const noexcept {
    return work_queue_->counters();
}

const std::vector<WqeRecord>& RnicDevice::records() const noexcept {
    return work_queue_->records();
}

const std::vector<EvidenceEvent>& RnicDevice::evidence() const noexcept {
    return work_queue_->evidence();
}

const WqeRecord& RnicDevice::wqe(WqeId wqe_id) const {
    return work_queue_->wqe(wqe_id);
}

void RnicDevice::validateInvariants() const {
    work_queue_->validateInvariants();
    if (pcie_fabric_) {
        pcie_fabric_->validateInvariants();
    }
    if (!config_.qpc.enabled) {
        for (const WqeRecord& record : records()) {
            if (record.timeline.qpc_ready_at_ps.has_value()) {
                throw std::logic_error(
                    "disabled RNIC QPC produced an applicable timestamp");
            }
        }
    }
    if (config_.dma.enabled) {
        if (!pcieBinding().has_value()) {
            throw std::logic_error("enabled RNIC DMA has no WQ binding");
        }
    } else if (pcie_fabric_ || pcieBinding().has_value()) {
        throw std::logic_error("disabled RNIC DMA retained fabric state");
    }
}

void RnicDevice::observeCallerTime(Picoseconds now_ps) {
    if (now_ps < last_caller_time_ps_) {
        throw std::logic_error("RNIC device caller time regressed");
    }
    last_caller_time_ps_ = now_ps;
}

const char* toString(RnicStageApplicability applicability) noexcept {
    switch (applicability) {
    case RnicStageApplicability::Applicable:
        return "applicable";
    case RnicStageApplicability::NotApplicable:
        return "not_applicable";
    default:
        return "invalid";
    }
}

}  // namespace simllm::rnic
