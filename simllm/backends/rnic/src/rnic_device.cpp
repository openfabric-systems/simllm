#include "simllm/rnic/rnic_device.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
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

std::uint64_t checkedMultiply(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* detail) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string("RNIC ") + detail);
    }
    return lhs * rhs;
}

const PciePathConfig& enabledPath(
    const PcieFabricConfig& config,
    std::uint32_t path_id) {
    const auto found = std::find_if(
        config.paths.begin(),
        config.paths.end(),
        [path_id](const PciePathConfig& path) {
            return path.path_id == path_id && path.enabled;
        });
    if (found == config.paths.end()) {
        throw std::invalid_argument(
            "RNIC host-memory layout references an unknown PCIe path");
    }
    return *found;
}

void validateAllocationPath(
    const HostMemoryAllocation& allocation,
    const PcieFabricConfig& fabric) {
    if (enabledPath(fabric, allocation.path_id).endpoint
        != allocation.endpoint) {
        throw std::invalid_argument(
            "RNIC host-memory allocation endpoint disagrees with its path");
    }
}

void validateHostMemoryLayout(
    const RnicDeviceConfig& config,
    const RnicSubmissionProfile& submission) {
    const RnicHostMemoryConfig& memory = config.host_memory;
    const WorkQueueHostMemoryBinding& binding = memory.work_queue;
    const WorkQueuePcieBinding& pcie = config.dma.work_queue;
    if (memory.allocations.size() < 6) {
        throw std::invalid_argument(
            "RNIC host-memory layout requires all queue and data objects");
    }
    std::map<HostMemoryAllocationId, const HostMemoryAllocation*> by_id;
    std::map<HostMemoryObjectKind, std::size_t> kind_counts;
    for (const HostMemoryAllocation& allocation : memory.allocations) {
        if (allocation.device_owner_id != memory.device_owner_id
            || !by_id.emplace(allocation.allocation_id, &allocation).second) {
            throw std::invalid_argument(
                "RNIC host-memory layout has a foreign or duplicate allocation");
        }
        ++kind_counts[allocation.object_kind];
        validateAllocationPath(allocation, config.dma.fabric);
    }
    for (const HostMemoryObjectKind kind : {
             HostMemoryObjectKind::QpcIcm,
             HostMemoryObjectKind::SqRing,
             HostMemoryObjectKind::RqRing,
             HostMemoryObjectKind::CqRing,
             HostMemoryObjectKind::DoorbellRecord}) {
        if (kind_counts[kind] != 1) {
            throw std::invalid_argument(
                "RNIC host-memory layout requires one allocation per queue object");
        }
    }
    const bool cpu_proxy = submission.producer_shape
        == RnicProducerShape::CpuProxy;
    const auto descriptor_item = kind_counts.find(
        HostMemoryObjectKind::DescriptorQueue);
    const std::size_t descriptor_count = descriptor_item == kind_counts.end()
        ? 0U
        : descriptor_item->second;
    if (kind_counts[HostMemoryObjectKind::DataRegion] == 0
        || descriptor_count != (cpu_proxy ? 1U : 0U)
        || kind_counts.size() != (cpu_proxy ? 7U : 6U)) {
        throw std::invalid_argument(
            "RNIC host-memory layout has an incompatible data or descriptor "
            "allocation set");
    }
    const auto require = [&by_id](
                             HostMemoryAllocationId allocation_id,
                             HostMemoryObjectKind kind) {
        const auto found = by_id.find(allocation_id);
        if (found == by_id.end() || found->second->object_kind != kind) {
            throw std::invalid_argument(
                "RNIC host-memory binding has a missing or mistyped object");
        }
        return found->second;
    };
    const HostMemoryAllocation* qpc = require(
        binding.qpc_icm_allocation_id, HostMemoryObjectKind::QpcIcm);
    const HostMemoryAllocation* sq = require(
        binding.sq_ring_allocation_id, HostMemoryObjectKind::SqRing);
    static_cast<void>(require(
        binding.rq_ring_allocation_id, HostMemoryObjectKind::RqRing));
    const HostMemoryAllocation* cq = require(
        binding.cq_ring_allocation_id, HostMemoryObjectKind::CqRing);
    const HostMemoryAllocation* doorbell = require(
        binding.doorbell_record_allocation_id,
        HostMemoryObjectKind::DoorbellRecord);
    if (qpc->owner_id != config.identity.qpn
        || sq->owner_id != config.work_queue.sq_id
        || cq->owner_id != config.work_queue.cq_id
        || doorbell->owner_id != config.work_queue.sq_id) {
        throw std::invalid_argument(
            "RNIC host-memory object owner disagrees with the device");
    }
    if (qpc->endpoint != PcieEndpointKind::HostPinnedMemory
        || sq->endpoint != submission.queue_endpoint
        || cq->endpoint != submission.queue_endpoint
        || doorbell->endpoint != submission.queue_endpoint) {
        throw std::invalid_argument(
            "RNIC host-memory queue endpoint disagrees with the submission "
            "shape");
    }
    if (cpu_proxy) {
        const HostMemoryAllocation* descriptor = require(
            submission.descriptor_queue_allocation_id,
            HostMemoryObjectKind::DescriptorQueue);
        if (!submission.descriptor_writer.has_value()
            || descriptor->owner_id != submission.descriptor_writer->id
            || descriptor->endpoint != PcieEndpointKind::HostPinnedMemory) {
            throw std::invalid_argument(
                "RNIC proxy descriptor queue has incompatible ownership or "
                "placement");
        }
    }
    for (const HostMemoryAllocation& allocation : memory.allocations) {
        if (allocation.object_kind == HostMemoryObjectKind::DataRegion
            && (!allocation.mkey.has_value()
                || allocation.owner_id != *allocation.mkey)) {
            throw std::invalid_argument(
                "RNIC data-region owner must equal its MKey");
        }
    }
    if (sq->path_id != pcie.pcie_sq_memory_path_id
        || cq->path_id != pcie.pcie_cq_memory_path_id
        || doorbell->path_id != pcie.pcie_doorbell_record_path_id
        || sq->virtual_address % 4096 != pcie.pcie_sq_first_byte_offset
        || cq->virtual_address % 4096 != pcie.pcie_cq_first_byte_offset
        || doorbell->virtual_address % 4096
            != pcie.pcie_doorbell_record_first_byte_offset) {
        throw std::invalid_argument(
            "RNIC host-memory queue location disagrees with the PCIe binding");
    }
    if (qpc->length_bytes < binding.qpc_context_bytes
        || sq->length_bytes < checkedMultiply(
               config.work_queue.sq_depth,
               pcie.pcie_wqe_bytes,
               "SQ allocation size overflow")
        || cq->length_bytes < checkedMultiply(
               config.work_queue.cq_depth,
               pcie.pcie_cqe_bytes,
               "CQ allocation size overflow")
        || doorbell->length_bytes < pcie.pcie_doorbell_record_bytes) {
        throw std::invalid_argument(
            "RNIC host-memory queue allocation is too small");
    }
    const PciePathConfig& translation_path = enabledPath(
        config.dma.fabric, memory.registry.translation_path_id);
    if (translation_path.endpoint != PcieEndpointKind::HostPinnedMemory) {
        throw std::invalid_argument(
            "RNIC host-memory translation path must be host-pinned");
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
    if (config_.host_memory.version != kRnicHostMemoryConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC host-memory config version");
    }
    if (config_.submission.version != kRnicSubmissionConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC submission config version");
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
    std::optional<RnicSubmissionProfile> submission_profile;
    if (config_.dma.enabled) {
        submission_profile = resolveRnicSubmissionProfile(
            config_.submission, config_.identity.qpn);
    } else if (!isDefaultRnicSubmissionConfig(config_.submission)) {
        throw std::invalid_argument(
            "RNIC DMA-disabled device rejects active submission fields");
    }
    if (submission_profile.has_value()
        && submission_profile->producer_shape
            != RnicProducerShape::HostCpuDriver
        && !config_.host_memory.enabled) {
        throw std::invalid_argument(
            "RNIC non-host producer requires registered host memory");
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
    if (!config_.host_memory.enabled && attachments.shared_host_memory) {
        throw std::invalid_argument(
            "RNIC host-memory-disabled device cannot attach a registry");
    }
    if (config_.host_memory.enabled) {
        if (!config_.dma.enabled || !config_.qpc.enabled
            || config_.work_queue.qpc_lookup_service_ps != 0
            || config_.host_memory.device_owner_id == 0
            || config_.host_memory.work_queue.version
                != kWorkQueueHostMemoryBindingVersion) {
            throw std::invalid_argument(
                "RNIC host memory requires DMA, QPC and non-scalar bindings");
        }
        validateHostMemoryLayout(config_, *submission_profile);
    }
    if (config_.network.enabled && attachments.network_port == nullptr) {
        throw std::invalid_argument(
            "RNIC network-enabled device requires an external port");
    }
    if (!config_.network.enabled && attachments.network_port != nullptr) {
        throw std::invalid_argument(
            "RNIC network-disabled device rejects an external port");
    }
    if (config_.dma.enabled && attachments.shared_pcie_fabric
        && !samePcieFabricConfig(
            config_.dma.fabric,
            attachments.shared_pcie_fabric->config())) {
        throw std::invalid_argument(
            "shared RNIC PCIe fabric config must match the device config");
    }
    if (config_.dma.fabric_endpoint_id != 0) {
        if (!config_.dma.enabled) {
            throw std::invalid_argument(
                "RNIC fabric endpoint identity requires enabled DMA");
        }
        if (config_.dma.fabric.host_endpoint_id == 0
            || config_.dma.fabric_endpoint_id
                == config_.dma.fabric.host_endpoint_id) {
            throw std::invalid_argument(
                "RNIC fabric endpoint identity must be nonzero, distinct from "
                "the fabric host endpoint, and the host endpoint must be "
                "named");
        }
        // The UAR mapping owner is not yet a fabric endpoint identity, so a
        // GPU-owned mapping fails closed rather than being charged to the host.
        if (submission_profile.has_value()
            && submission_profile->uar_mapping_owner
                != RnicUarMappingOwner::HostCpu) {
            throw std::invalid_argument(
                "RNIC fabric endpoint identity cannot yet attribute a "
                "GPU-owned UAR mapping");
        }
    }
    if (!config_.host_memory.enabled
        && !config_.host_memory.peer_read_grants.empty()) {
        throw std::invalid_argument(
            "RNIC host-memory-disabled device rejects peer read grants");
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
    if (config_.host_memory.enabled) {
        stage_report_.pcie_qpc_icm = RnicStageApplicability::Applicable;
        stage_report_.pcie_mtt_mpt = RnicStageApplicability::Applicable;
        stage_report_.pcie_payload_read = RnicStageApplicability::Applicable;
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
                this,
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
        if (config_.dma.fabric_endpoint_id != 0) {
            pcie_fabric_->claimEndpoint(
                this, config_.dma.fabric_endpoint_id, PcieDeviceKind::Rnic);
            claimed_fabric_endpoint_ = true;
        }
        std::optional<VirtualHostMemory::RegistrationPlan> memory_plan;
        if (config_.host_memory.enabled) {
            if (attachments.shared_host_memory) {
                if (!sameVirtualHostMemoryConfig(
                        config_.host_memory.registry,
                        attachments.shared_host_memory->config())) {
                    throw std::invalid_argument(
                        "shared RNIC host-memory config must match the "
                        "device config");
                }
                host_memory_ = std::move(attachments.shared_host_memory);
            } else {
                host_memory_ = std::make_shared<VirtualHostMemory>(
                    config_.host_memory.registry);
            }
            host_memory_->claimDeviceOwner(
                this,
                config_.host_memory.device_owner_id,
                config_.dma.fabric_endpoint_id,
                PcieDeviceKind::Rnic,
                config_.host_memory.peer_read_grants);
            claimed_host_memory_owner_ = true;
            memory_plan.emplace(host_memory_->planClaimedRegistrations(
                this, config_.host_memory.allocations, 0));
        }

        work_queue_.reset(new WorkQueue(
            config_.work_queue,
            *network_port_,
            pcie_fabric_.get(),
            std::move(pcie_binding),
            config_.qpc.enabled,
            host_memory_.get(),
            config_.host_memory.enabled
                ? std::optional<WorkQueueHostMemoryBinding>(
                      config_.host_memory.work_queue)
                : std::nullopt,
            config_.host_memory.enabled
                ? config_.host_memory.device_owner_id
                : 0,
            config_.dma.fabric_endpoint_id,
            std::move(submission_profile)));
        if (memory_plan.has_value()) {
            host_memory_->commit(std::move(*memory_plan));
            host_memory_registered_ = true;
        }
    } catch (...) {
        work_queue_.reset();
        if (claimed_host_memory_owner_) {
            host_memory_->releaseDeviceOwner(
                this, config_.host_memory.device_owner_id);
            claimed_host_memory_owner_ = false;
        }
        if (claimed_fabric_endpoint_) {
            pcie_fabric_->releaseEndpoint(
                this, config_.dma.fabric_endpoint_id);
            claimed_fabric_endpoint_ = false;
        }
        if (claimed_ordering_domains_) {
            pcie_fabric_->releaseOrderingDomains(
                this,
                claimed_submission_domain_,
                claimed_completion_domain_);
            claimed_ordering_domains_ = false;
        }
        throw;
    }
}

RnicDevice::~RnicDevice() {
    work_queue_.reset();
    if (host_memory_registered_) {
        try {
            host_memory_->teardownClaimedOwner(
                this,
                config_.host_memory.device_owner_id,
                last_caller_time_ps_);
            host_memory_registered_ = false;
        } catch (...) {
            std::terminate();
        }
    }
    if (claimed_host_memory_owner_) {
        host_memory_->releaseDeviceOwner(
            this, config_.host_memory.device_owner_id);
        claimed_host_memory_owner_ = false;
    }
    if (claimed_fabric_endpoint_) {
        pcie_fabric_->releaseEndpoint(this, config_.dma.fabric_endpoint_id);
        claimed_fabric_endpoint_ = false;
    }
    if (claimed_ordering_domains_) {
        pcie_fabric_->releaseOrderingDomains(
            this,
            claimed_submission_domain_,
            claimed_completion_domain_);
    }
}

PostResult RnicDevice::postSend(
    const WorkRequest& request,
    Picoseconds now_ps) {
    requireHostMemoryLive();
    validateCallerTime(now_ps);
    PostResult result = work_queue_->postSend(request, now_ps);
    last_caller_time_ps_ = now_ps;
    return result;
}

PostBatchResult RnicDevice::postSendBatch(
    const std::vector<WorkRequest>& requests,
    Picoseconds now_ps) {
    requireHostMemoryLive();
    validateCallerTime(now_ps);
    PostBatchResult result = work_queue_->postSendBatch(requests, now_ps);
    last_caller_time_ps_ = now_ps;
    return result;
}

DoorbellBatch RnicDevice::ringDoorbell(Picoseconds now_ps) {
    requireHostMemoryLive();
    validateCallerTime(now_ps);
    DoorbellBatch result = work_queue_->ringDoorbell(now_ps);
    last_caller_time_ps_ = now_ps;
    return result;
}

DoorbellBatch RnicDevice::ringDoorbell(
    Picoseconds now_ps,
    const RnicProducerTaskLink& producer_task) {
    requireHostMemoryLive();
    validateCallerTime(now_ps);
    DoorbellBatch result = work_queue_->ringDoorbell(
        now_ps, producer_task);
    last_caller_time_ps_ = now_ps;
    return result;
}

void RnicDevice::onNetworkEvent(const NetworkEvent& event) {
    requireHostMemoryLive();
    if (!config_.network.enabled) {
        throw std::logic_error(
            "external RNIC network event supplied to the inert port");
    }
    validateCallerTime(event.event_time_ps);
    work_queue_->onNetworkEvent(event);
    last_caller_time_ps_ = event.event_time_ps;
}

std::size_t RnicDevice::progress(Picoseconds now_ps) {
    requireHostMemoryLive();
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
    requireHostMemoryLive();
    validateCallerTime(now_ps);
    std::vector<CompletionEntry> result =
        work_queue_->pollCompletionQueue(max_entries, now_ps);
    last_caller_time_ps_ = now_ps;
    return result;
}

void RnicDevice::teardownHostMemory(Picoseconds now_ps) {
    if (!config_.host_memory.enabled || !host_memory_
        || !host_memory_registered_) {
        throw std::logic_error("RNIC device has no live host memory");
    }
    validateCallerTime(now_ps);
    if (work_queue_->hasPendingPhysicalWork()
        || work_queue_->occupiedSqEntries() != 0
        || work_queue_->completionQueueDepth() != 0
        || work_queue_->unpublishedWqeCount() != 0) {
        throw std::logic_error(
            "RNIC host memory cannot be torn down with live queue state");
    }
    host_memory_->teardownClaimedOwner(
        this, config_.host_memory.device_owner_id, now_ps);
    host_memory_registered_ = false;
    host_memory_->releaseDeviceOwner(
        this, config_.host_memory.device_owner_id);
    claimed_host_memory_owner_ = false;
    last_caller_time_ps_ = now_ps;
}

PcieTransactionResult RnicDevice::submitPcie(
    const PcieTransactionRequest& request) {
    return submitPcie(request, PcieEndpointAttribution{});
}

PcieTransactionResult RnicDevice::submitPcie(
    const PcieTransactionRequest& request,
    const PcieEndpointAttribution& attribution) {
    requireHostMemoryLive();
    if (!config_.dma.enabled || !pcie_fabric_) {
        throw std::logic_error(
            "RNIC device has no enabled PCIe fabric");
    }
    validateCallerTime(request.submitted_at_ps);
    if (pcie_fabric_->orderingDomainClaimedByOther(
            this, request.ordering_domain)) {
        throw std::invalid_argument(
            "RNIC PCIe ordering domain is claimed by another device");
    }
    if (attribution.requester_endpoint_id != 0
        && !pcie_fabric_->endpointClaimedBy(
               this, attribution.requester_endpoint_id)) {
        throw std::invalid_argument(
            "RNIC PCIe requester endpoint is not claimed by this device");
    }
    PcieTransactionResult result = pcie_fabric_->submit(request, attribution);
    last_caller_time_ps_ = request.submitted_at_ps;
    return result;
}

std::optional<Picoseconds> RnicDevice::nextEventTime() const {
    requireHostMemoryLive();
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

bool RnicDevice::usesSharedPcieFabric() const noexcept {
    return claimed_ordering_domains_;
}

std::optional<WorkQueuePcieBinding> RnicDevice::pcieBinding() const {
    return work_queue_->pcieBinding();
}

const PcieFabric* RnicDevice::pcieFabric() const noexcept {
    return pcie_fabric_.get();
}

const VirtualHostMemory* RnicDevice::hostMemory() const noexcept {
    return host_memory_.get();
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

const std::vector<HostMemoryAccessRecord>&
RnicDevice::memoryAccesses() const noexcept {
    return work_queue_->memoryAccesses();
}

const std::optional<RnicSubmissionProfile>&
RnicDevice::submissionProfile() const noexcept {
    return work_queue_->submissionProfile();
}

const std::vector<RnicSubmissionRecord>&
RnicDevice::submissionRecords() const noexcept {
    return work_queue_->submissionRecords();
}

const std::vector<RnicCqConsumptionRecord>&
RnicDevice::cqConsumptionRecords() const noexcept {
    return work_queue_->cqConsumptionRecords();
}

const WqeRecord& RnicDevice::wqe(WqeId wqe_id) const {
    return work_queue_->wqe(wqe_id);
}

void RnicDevice::validateInvariants() const {
    work_queue_->validateInvariants();
    if (pcie_fabric_) {
        pcie_fabric_->validateInvariants();
    }
    if (config_.host_memory.enabled) {
        if (!host_memory_) {
            throw std::logic_error(
                "enabled RNIC host memory has no registry");
        }
        host_memory_->validateInvariants();
        const std::size_t expected_live = host_memory_registered_
            ? config_.host_memory.allocations.size()
            : 0;
        if (claimed_host_memory_owner_ != host_memory_registered_
            || host_memory_->deviceOwnerClaimedBy(
                   this, config_.host_memory.device_owner_id)
                != claimed_host_memory_owner_) {
            throw std::logic_error(
                "RNIC host-memory device-owner claim mismatch");
        }
        if (host_memory_->liveAllocationCount(
                config_.host_memory.device_owner_id)
            != expected_live) {
            throw std::logic_error(
                "RNIC host-memory live allocation count mismatch");
        }
    } else if (host_memory_ || claimed_host_memory_owner_
               || host_memory_registered_
               || !memoryAccesses().empty()) {
        throw std::logic_error(
            "disabled RNIC host memory retained structural state");
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
        if (!pcieBinding().has_value()
            || !submissionProfile().has_value()) {
            throw std::logic_error(
                "enabled RNIC DMA has no WQ binding or submission profile");
        }
        if (claimed_fabric_endpoint_
                != (config_.dma.fabric_endpoint_id != 0)
            || (claimed_fabric_endpoint_
                && !pcie_fabric_->endpointClaimedBy(
                       this, config_.dma.fabric_endpoint_id))) {
            throw std::logic_error(
                "RNIC fabric endpoint claim mismatch");
        }
    } else if (pcie_fabric_ || pcieBinding().has_value()
               || submissionProfile().has_value()
               || !submissionRecords().empty()
               || !cqConsumptionRecords().empty()) {
        throw std::logic_error(
            "disabled RNIC DMA retained fabric or submission state");
    }
}

void RnicDevice::validateCallerTime(Picoseconds now_ps) const {
    if (now_ps < last_caller_time_ps_) {
        throw std::logic_error("RNIC device caller time regressed");
    }
}

void RnicDevice::observeCallerTime(Picoseconds now_ps) {
    validateCallerTime(now_ps);
    last_caller_time_ps_ = now_ps;
}

void RnicDevice::requireHostMemoryLive() const {
    if (config_.host_memory.enabled && !host_memory_registered_) {
        throw std::logic_error("RNIC host memory was torn down");
    }
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
