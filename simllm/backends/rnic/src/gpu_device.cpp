#include "simllm/rnic/gpu_device.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace simllm::rnic {
namespace {

std::uint64_t checkedAdd(std::uint64_t lhs, std::uint64_t rhs) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error("GPU device byte accounting overflow");
    }
    return lhs + rhs;
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
            "GPU region references an unknown PCIe path");
    }
    return *found;
}

bool deviceLocalEndpoint(PcieEndpointKind endpoint) noexcept {
    return endpoint == PcieEndpointKind::GpuMemory
        || endpoint == PcieEndpointKind::DeviceMemory;
}

void validateIdentity(const GpuDeviceIdentity& identity, bool fabric_enabled) {
    if (identity.version != kGpuDeviceIdentityVersion) {
        throw std::invalid_argument("unsupported GPU device identity version");
    }
    if (identity.gpu_id == 0) {
        throw std::invalid_argument("GPU device number must be nonzero");
    }
    if (fabric_enabled && identity.requester_id == 0) {
        throw std::invalid_argument(
            "fabric-attached GPU requires a nonzero PCIe requester identity");
    }
    if (!fabric_enabled && identity.requester_id != 0) {
        throw std::invalid_argument(
            "fabric-disabled GPU rejects a PCIe requester identity");
    }
}

void validateFabricConfig(const GpuFabricConfig& fabric) {
    if (fabric.version != kGpuFabricConfigVersion) {
        throw std::invalid_argument("unsupported GPU fabric config version");
    }
    if (!fabric.enabled) {
        if (fabric.endpoint_id != 0 || fabric.ordering_domain != 0) {
            throw std::invalid_argument(
                "fabric-disabled GPU rejects active fabric fields");
        }
        return;
    }
    if (fabric.endpoint_id == 0 || fabric.ordering_domain == 0) {
        throw std::invalid_argument(
            "fabric-attached GPU requires a nonzero endpoint identity and "
            "ordering domain");
    }
    if (fabric.fabric.host_endpoint_id == 0
        || fabric.endpoint_id == fabric.fabric.host_endpoint_id) {
        throw std::invalid_argument(
            "GPU endpoint identity must differ from the named fabric host "
            "endpoint, and that host endpoint must be named");
    }
}

void validateMemoryConfig(const GpuDeviceConfig& config) {
    const GpuMemoryConfig& memory = config.memory;
    if (memory.version != kGpuMemoryConfigVersion) {
        throw std::invalid_argument("unsupported GPU memory config version");
    }
    if (!memory.enabled) {
        if (memory.device_owner_id != 0 || !memory.allocations.empty()
            || !memory.peer_read_grants.empty()) {
            throw std::invalid_argument(
                "memory-disabled GPU rejects active memory fields");
        }
        return;
    }
    if (!config.fabric.enabled) {
        throw std::invalid_argument(
            "GPU regions require an attached PCIe fabric");
    }
    if (memory.device_owner_id == 0 || memory.allocations.empty()) {
        throw std::invalid_argument(
            "GPU memory requires a nonzero device owner and at least one "
            "region");
    }
    std::set<HostMemoryAllocationId> allocation_ids;
    for (const HostMemoryAllocation& allocation : memory.allocations) {
        if (allocation.device_owner_id != memory.device_owner_id
            || !allocation_ids.insert(allocation.allocation_id).second) {
            throw std::invalid_argument(
                "GPU memory layout has a foreign or duplicate region");
        }
        if (allocation.object_kind != HostMemoryObjectKind::DataRegion
            || allocation.owner_kind != HostMemoryOwnerKind::MemoryRegion) {
            throw std::invalid_argument(
                "GPU memory layout accepts data regions only");
        }
        if (!allocation.mkey.has_value()
            || allocation.owner_id != *allocation.mkey) {
            throw std::invalid_argument(
                "GPU data-region owner must equal its MKey");
        }
        if (enabledPath(config.fabric.fabric, allocation.path_id).endpoint
            != allocation.endpoint) {
            throw std::invalid_argument(
                "GPU region endpoint disagrees with its PCIe path");
        }
    }
}

void validateTransferShape(const GpuFabricTransfer& request) {
    if (request.version != kGpuFabricTransferVersion) {
        throw std::invalid_argument("unsupported GPU fabric transfer version");
    }
    if (request.allocation_id == 0 || request.transfer_bytes == 0) {
        throw std::invalid_argument(
            "GPU fabric transfer identity and byte count must be positive");
    }
    const bool write = request.service_class == PcieServiceClass::PayloadWrite
        && request.operation == PcieOperation::PostedWrite;
    const bool read = request.service_class == PcieServiceClass::PayloadRead
        && request.operation == PcieOperation::NonPostedRead;
    if (!write && !read) {
        throw std::invalid_argument(
            "GPU fabric transfer must be a payload posted write or a payload "
            "non-posted read");
    }
}

}  // namespace

GpuDevice::GpuDevice(
    GpuDeviceConfig config,
    GpuDeviceAttachments attachments)
    : config_(std::move(config)) {
    if (config_.version != kGpuDeviceConfigVersion) {
        throw std::invalid_argument("unsupported GPU device config version");
    }
    validateIdentity(config_.identity, config_.fabric.enabled);
    validateFabricConfig(config_.fabric);
    validateMemoryConfig(config_);
    if (!config_.fabric.enabled && attachments.shared_pcie_fabric) {
        throw std::invalid_argument(
            "fabric-disabled GPU cannot attach a PCIe fabric");
    }
    if (!config_.memory.enabled && attachments.shared_memory) {
        throw std::invalid_argument(
            "memory-disabled GPU cannot attach a registry");
    }
    if (config_.fabric.enabled && attachments.shared_pcie_fabric
        && !samePcieFabricConfig(
            config_.fabric.fabric,
            attachments.shared_pcie_fabric->config())) {
        throw std::invalid_argument(
            "shared GPU PCIe fabric config must match the device config");
    }
    if (!config_.fabric.enabled) {
        return;
    }

    if (attachments.shared_pcie_fabric) {
        pcie_fabric_ = std::move(attachments.shared_pcie_fabric);
        shared_pcie_fabric_ = true;
    } else {
        pcie_fabric_ = std::make_shared<PcieFabric>(config_.fabric.fabric);
    }

    try {
        pcie_fabric_->claimOrderingDomain(
            this, config_.fabric.ordering_domain);
        claimed_ordering_domain_ = true;
        pcie_fabric_->claimEndpoint(
            this, config_.fabric.endpoint_id, PcieDeviceKind::Gpu);
        claimed_fabric_endpoint_ = true;
        if (config_.memory.enabled) {
            if (attachments.shared_memory) {
                if (!sameVirtualHostMemoryConfig(
                        config_.memory.registry,
                        attachments.shared_memory->config())) {
                    throw std::invalid_argument(
                        "shared GPU registry config must match the device "
                        "config");
                }
                memory_ = std::move(attachments.shared_memory);
            } else {
                memory_ = std::make_shared<VirtualHostMemory>(
                    config_.memory.registry);
            }
            memory_->claimDeviceOwner(
                this,
                config_.memory.device_owner_id,
                config_.fabric.endpoint_id,
                PcieDeviceKind::Gpu,
                config_.memory.peer_read_grants);
            claimed_memory_owner_ = true;
            VirtualHostMemory::RegistrationPlan plan =
                memory_->planClaimedRegistrations(
                    this, config_.memory.allocations, 0);
            memory_->commit(std::move(plan));
            memory_registered_ = true;
        }
    } catch (...) {
        if (claimed_memory_owner_) {
            memory_->releaseDeviceOwner(
                this, config_.memory.device_owner_id);
            claimed_memory_owner_ = false;
        }
        if (claimed_fabric_endpoint_) {
            pcie_fabric_->releaseEndpoint(this, config_.fabric.endpoint_id);
            claimed_fabric_endpoint_ = false;
        }
        if (claimed_ordering_domain_) {
            pcie_fabric_->releaseOrderingDomain(
                this, config_.fabric.ordering_domain);
            claimed_ordering_domain_ = false;
        }
        throw;
    }
}

GpuDevice::~GpuDevice() {
    if (memory_registered_) {
        try {
            memory_->teardownClaimedOwner(
                this, config_.memory.device_owner_id, last_transfer_time_ps_);
            memory_registered_ = false;
        } catch (...) {
            std::terminate();
        }
    }
    if (claimed_memory_owner_) {
        memory_->releaseDeviceOwner(this, config_.memory.device_owner_id);
        claimed_memory_owner_ = false;
    }
    if (claimed_fabric_endpoint_) {
        pcie_fabric_->releaseEndpoint(this, config_.fabric.endpoint_id);
        claimed_fabric_endpoint_ = false;
    }
    if (claimed_ordering_domain_) {
        pcie_fabric_->releaseOrderingDomain(
            this, config_.fabric.ordering_domain);
        claimed_ordering_domain_ = false;
    }
}

GpuFabricTransferRecord GpuDevice::transfer(
    const GpuFabricTransfer& request) {
    if (!config_.fabric.enabled || !pcie_fabric_) {
        throw std::logic_error("GPU device has no enabled PCIe fabric");
    }
    if (!memory_registered_) {
        throw std::logic_error("GPU device has no live regions");
    }
    validateTransferShape(request);
    if (request.submitted_at_ps < last_transfer_time_ps_) {
        throw std::logic_error("GPU device caller time regressed");
    }

    // Ownership first: a region this device neither owns nor was granted is
    // refused before the fabric is touched at all.
    const HostMemoryAllocation& target = memory_->allocation(
        request.allocation_id);
    const HostMemoryDeviceOwnerId peer = request.peer_device_owner_id;
    if (peer != 0 && peer == config_.memory.device_owner_id) {
        throw std::invalid_argument(
            "GPU peer transfer names its own device owner");
    }
    const HostMemoryDeviceOwnerId expected_owner =
        peer == 0 ? config_.memory.device_owner_id : peer;
    if (target.object_kind != HostMemoryObjectKind::DataRegion
        || target.device_owner_id != expected_owner) {
        throw std::invalid_argument(
            "GPU fabric transfer does not match its named region owner");
    }
    if (peer != 0
        && !memory_->peerReadGranted(peer, config_.memory.device_owner_id)) {
        throw std::invalid_argument(
            "GPU fabric transfer names an ungranted peer region");
    }
    const std::uint64_t access_end = checkedAdd(
        request.allocation_offset_bytes, request.transfer_bytes);
    if (access_end > target.length_bytes) {
        throw std::out_of_range("GPU fabric transfer exceeds its region");
    }
    const PcieFabricConfig fabric_config = pcie_fabric_->config();
    if (enabledPath(fabric_config, target.path_id).endpoint
        != target.endpoint) {
        throw std::invalid_argument(
            "GPU region endpoint disagrees with its PCIe path");
    }
    const std::optional<PcieEndpointId> owner_endpoint =
        memory_->deviceOwnerEndpointId(target.device_owner_id);
    const PcieEndpointId completer = deviceLocalEndpoint(target.endpoint)
        ? owner_endpoint.value_or(0)
        : fabric_config.host_endpoint_id;
    if (completer == 0) {
        throw std::invalid_argument(
            "GPU fabric transfer completer has no fabric endpoint identity");
    }
    if (pcie_fabric_->orderingDomainClaimedByOther(
            this, config_.fabric.ordering_domain)) {
        throw std::invalid_argument(
            "GPU PCIe ordering domain is claimed by another device");
    }
    if (transfer_records_.size() == transfer_records_.max_size()
        || next_transfer_sequence_
            == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("GPU fabric transfer capacity overflow");
    }

    // The GPU copy engine addresses the region directly, so no MKey, MPT or
    // MTT read is emitted here. Translation belongs to the RNIC's registered
    // memory path, not to a device writing its own or a granted peer's region.
    PcieTransactionRequest access;
    access.client_id = config_.identity.requester_id;
    access.client_token = request.client_token;
    access.service_class = request.service_class;
    access.operation = request.operation;
    access.request_direction = PcieDirection::DeviceToHost;
    access.ordering = request.ordering;
    access.path_id = target.path_id;
    access.ordering_domain = config_.fabric.ordering_domain;
    access.useful_bytes = request.transfer_bytes;
    access.transfer_bytes = request.transfer_bytes;
    access.first_byte_offset = static_cast<std::uint32_t>(
        (target.virtual_address + request.allocation_offset_bytes) % 4096);
    access.submitted_at_ps = request.submitted_at_ps;
    const PcieEndpointAttribution attribution{
        config_.fabric.endpoint_id, completer};
    const PcieTransactionResult result = pcie_fabric_->submit(
        access, attribution);

    GpuFabricTransferRecord record;
    record.sequence = next_transfer_sequence_;
    record.allocation_id = target.allocation_id;
    record.service_class = request.service_class;
    record.requester_endpoint_id = config_.fabric.endpoint_id;
    record.completer_endpoint_id = completer;
    record.completer_kind =
        pcie_fabric_->attachedEndpointKind(completer).value_or(
            PcieDeviceKind::Host);
    record.transfer_bytes = request.transfer_bytes;
    record.transaction_id = result.transaction_id;
    record.modeled_link_bytes = checkedAdd(
        result.host_to_device.modeled_link_bytes,
        result.device_to_host.modeled_link_bytes);
    record.credit_wait_ps = result.waits.credit_ps;
    record.link_queue_wait_ps = result.waits.link_queue_ps;
    record.submitted_at_ps = request.submitted_at_ps;
    record.completed_at_ps = result.completed_at_ps;
    transfer_records_.push_back(record);
    ++next_transfer_sequence_;
    transferred_bytes_ = checkedAdd(
        transferred_bytes_, request.transfer_bytes);
    last_transfer_time_ps_ = request.submitted_at_ps;
    return record;
}

void GpuDevice::teardownMemory(Picoseconds now_ps) {
    if (!config_.memory.enabled || !memory_ || !memory_registered_) {
        throw std::logic_error("GPU device has no live regions");
    }
    if (now_ps < last_transfer_time_ps_) {
        throw std::logic_error("GPU device caller time regressed");
    }
    memory_->teardownClaimedOwner(
        this, config_.memory.device_owner_id, now_ps);
    memory_registered_ = false;
    memory_->releaseDeviceOwner(this, config_.memory.device_owner_id);
    claimed_memory_owner_ = false;
    last_transfer_time_ps_ = now_ps;
}

const GpuDeviceConfig& GpuDevice::config() const noexcept {
    return config_;
}

const std::vector<GpuFabricTransferRecord>&
GpuDevice::transferRecords() const noexcept {
    return transfer_records_;
}

const PcieFabric* GpuDevice::pcieFabric() const noexcept {
    return pcie_fabric_.get();
}

const VirtualHostMemory* GpuDevice::memory() const noexcept {
    return memory_.get();
}

bool GpuDevice::usesSharedPcieFabric() const noexcept {
    return shared_pcie_fabric_;
}

std::uint64_t GpuDevice::transferredBytes() const noexcept {
    return transferred_bytes_;
}

void GpuDevice::validateInvariants() const {
    if (!config_.fabric.enabled) {
        if (pcie_fabric_ || memory_ || claimed_fabric_endpoint_
            || claimed_ordering_domain_ || claimed_memory_owner_
            || memory_registered_ || !transfer_records_.empty()) {
            throw std::logic_error(
                "fabric-disabled GPU retained structural state");
        }
        return;
    }
    if (!pcie_fabric_ || !claimed_fabric_endpoint_
        || !claimed_ordering_domain_
        || !pcie_fabric_->endpointClaimedBy(
               this, config_.fabric.endpoint_id)) {
        throw std::logic_error("GPU fabric endpoint claim mismatch");
    }
    pcie_fabric_->validateInvariants();
    if (config_.memory.enabled) {
        if (!memory_) {
            throw std::logic_error("enabled GPU memory has no registry");
        }
        memory_->validateInvariants();
        if (claimed_memory_owner_ != memory_registered_
            || memory_->deviceOwnerClaimedBy(
                   this, config_.memory.device_owner_id)
                != claimed_memory_owner_) {
            throw std::logic_error("GPU memory device-owner claim mismatch");
        }
        const std::size_t expected_live = memory_registered_
            ? config_.memory.allocations.size()
            : 0;
        if (memory_->liveAllocationCount(config_.memory.device_owner_id)
            != expected_live) {
            throw std::logic_error(
                "GPU memory live allocation count mismatch");
        }
    } else if (memory_ || claimed_memory_owner_ || memory_registered_) {
        throw std::logic_error("disabled GPU memory retained state");
    }
    std::uint64_t sequence = 1;
    std::uint64_t bytes = 0;
    Picoseconds previous_submitted_ps = 0;
    for (const GpuFabricTransferRecord& record : transfer_records_) {
        if (record.version != kGpuFabricTransferRecordVersion
            || record.sequence != sequence++
            || record.requester_endpoint_id != config_.fabric.endpoint_id
            || record.completer_endpoint_id == 0
            || record.transfer_bytes == 0
            || record.submitted_at_ps < previous_submitted_ps
            || record.completed_at_ps < record.submitted_at_ps) {
            throw std::logic_error("GPU fabric transfer record invariant failed");
        }
        previous_submitted_ps = record.submitted_at_ps;
        bytes = checkedAdd(bytes, record.transfer_bytes);
    }
    if (sequence != next_transfer_sequence_ || bytes != transferred_bytes_) {
        throw std::logic_error("GPU fabric transfer conservation failed");
    }
}

}  // namespace simllm::rnic
