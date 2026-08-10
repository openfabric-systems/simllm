#include "simllm/rnic/host_memory.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace simllm::rnic {
namespace {

bool isPowerOfTwo(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1U)) == 0;
}

std::uint64_t checkedAdd(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* detail) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(std::string("RNIC host memory ") + detail);
    }
    return lhs + rhs;
}

std::uint64_t checkedMultiply(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* detail) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error(std::string("RNIC host memory ") + detail);
    }
    return lhs * rhs;
}

bool validEndpoint(PcieEndpointKind endpoint) noexcept {
    return endpoint == PcieEndpointKind::HostPinnedMemory
        || endpoint == PcieEndpointKind::GpuMemory;
}

bool isRing(HostMemoryObjectKind kind) noexcept {
    return kind == HostMemoryObjectKind::SqRing
        || kind == HostMemoryObjectKind::RqRing
        || kind == HostMemoryObjectKind::CqRing;
}

HostMemoryOwnerKind requiredOwnerKind(HostMemoryObjectKind kind) {
    switch (kind) {
    case HostMemoryObjectKind::QpcIcm:
        return HostMemoryOwnerKind::QueuePair;
    case HostMemoryObjectKind::SqRing:
    case HostMemoryObjectKind::DoorbellRecord:
        return HostMemoryOwnerKind::SendQueue;
    case HostMemoryObjectKind::RqRing:
        return HostMemoryOwnerKind::ReceiveQueue;
    case HostMemoryObjectKind::CqRing:
        return HostMemoryOwnerKind::CompletionQueue;
    case HostMemoryObjectKind::DataRegion:
        return HostMemoryOwnerKind::MemoryRegion;
    default:
        throw std::invalid_argument("invalid RNIC host-memory object kind");
    }
}

void validateConfig(const VirtualHostMemoryConfig& config) {
    if (config.version != kVirtualHostMemoryConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC virtual host-memory config version");
    }
    if (config.translation_path_id == 0
        || config.queue_page_list_entry_bytes == 0
        || config.mpt_entry_bytes == 0
        || config.mtt_entry_bytes == 0) {
        throw std::invalid_argument(
            "RNIC host-memory translation fields must be positive");
    }
    if (config.queue_page_list_first_byte_offset >= 4096
        || config.mpt_first_byte_offset >= 4096
        || config.mtt_first_byte_offset >= 4096) {
        throw std::invalid_argument(
            "RNIC host-memory translation offsets must be below 4096");
    }
}

void validateAllocation(const HostMemoryAllocation& allocation) {
    if (allocation.version != kHostMemoryAllocationVersion
        || allocation.pages.version != kHostMemoryPageGeometryVersion) {
        throw std::invalid_argument(
            "unsupported RNIC host-memory allocation version");
    }
    if (allocation.allocation_id == 0 || allocation.device_owner_id == 0
        || allocation.owner_id == 0 || allocation.path_id == 0
        || allocation.length_bytes == 0) {
        throw std::invalid_argument(
            "RNIC host-memory allocation identity and extent must be positive");
    }
    if (!validEndpoint(allocation.endpoint)) {
        throw std::invalid_argument(
            "RNIC host-memory allocation endpoint must be host or GPU memory");
    }
    if (allocation.owner_kind != requiredOwnerKind(allocation.object_kind)) {
        throw std::invalid_argument(
            "RNIC host-memory allocation owner kind is incompatible");
    }
    if (allocation.object_kind == HostMemoryObjectKind::QpcIcm
        && allocation.endpoint != PcieEndpointKind::HostPinnedMemory) {
        throw std::invalid_argument(
            "RNIC QPC ICM allocation must use host-pinned memory");
    }
    if (allocation.object_kind == HostMemoryObjectKind::DataRegion) {
        if (!allocation.mkey.has_value() || *allocation.mkey == 0) {
            throw std::invalid_argument(
                "RNIC data allocation requires a nonzero MKey");
        }
    } else if (allocation.mkey.has_value()) {
        throw std::invalid_argument(
            "RNIC non-data allocation cannot carry an MKey");
    }
    if (!isPowerOfTwo(allocation.pages.page_size_bytes)
        || allocation.pages.page_size_bytes < 4096
        || allocation.pages.physical_page_addresses.empty()) {
        throw std::invalid_argument(
            "RNIC host-memory page geometry is invalid");
    }
    std::set<std::uint64_t> physical_pages;
    for (const std::uint64_t address
         : allocation.pages.physical_page_addresses) {
        if (address % allocation.pages.page_size_bytes != 0
            || !physical_pages.insert(address).second) {
            throw std::invalid_argument(
                "RNIC host-memory physical pages must be aligned and unique");
        }
    }
    const std::uint64_t capacity = checkedMultiply(
        allocation.pages.page_size_bytes,
        allocation.pages.physical_page_addresses.size(),
        "page capacity overflow");
    const std::uint64_t first_page_offset =
        allocation.virtual_address % allocation.pages.page_size_bytes;
    const std::uint64_t covered = checkedAdd(
        first_page_offset,
        allocation.length_bytes,
        "allocation coverage overflow");
    if (covered > capacity) {
        throw std::invalid_argument(
            "RNIC host-memory pages do not cover the allocation");
    }
    static_cast<void>(checkedAdd(
        allocation.virtual_address,
        allocation.length_bytes,
        "virtual-address extent overflow"));
}

bool extentsOverlap(
    const HostMemoryAllocation& lhs,
    const HostMemoryAllocation& rhs) {
    if (lhs.endpoint != rhs.endpoint) {
        return false;
    }
    const std::uint64_t lhs_end = lhs.virtual_address + lhs.length_bytes;
    const std::uint64_t rhs_end = rhs.virtual_address + rhs.length_bytes;
    return lhs.virtual_address < rhs_end && rhs.virtual_address < lhs_end;
}

const PciePathConfig& findPath(
    const PcieFabricConfig& config,
    std::uint32_t path_id) {
    const auto path = std::find_if(
        config.paths.begin(),
        config.paths.end(),
        [path_id](const PciePathConfig& candidate) {
            return candidate.path_id == path_id && candidate.enabled;
        });
    if (path == config.paths.end()) {
        throw std::invalid_argument(
            "RNIC host-memory access references an unknown PCIe path");
    }
    return *path;
}

std::uint32_t metadataOffset(
    std::uint32_t base,
    std::uint64_t index,
    std::uint64_t entry_bytes) {
    const std::uint64_t offset =
        (base + (index % 4096) * (entry_bytes % 4096)) % 4096;
    return static_cast<std::uint32_t>(offset);
}

void validateAccessShape(
    const HostMemoryAllocation& allocation,
    const HostMemoryAccessRequest& request) {
    switch (allocation.object_kind) {
    case HostMemoryObjectKind::QpcIcm:
        if (request.service_class != PcieServiceClass::QpcIcm
            || request.operation != PcieOperation::NonPostedRead
            || request.request_direction != PcieDirection::DeviceToHost
            || request.mkey.has_value()) {
            throw std::invalid_argument(
                "RNIC QPC access shape is incompatible");
        }
        break;
    case HostMemoryObjectKind::SqRing:
    case HostMemoryObjectKind::RqRing:
        if (request.service_class != PcieServiceClass::WqeRead
            || request.operation != PcieOperation::NonPostedRead
            || request.request_direction != PcieDirection::DeviceToHost
            || request.mkey.has_value()) {
            throw std::invalid_argument(
                "RNIC work-queue ring access shape is incompatible");
        }
        break;
    case HostMemoryObjectKind::CqRing:
        if (request.service_class != PcieServiceClass::CqeWrite
            || request.operation != PcieOperation::PostedWrite
            || request.request_direction != PcieDirection::DeviceToHost
            || request.mkey.has_value()) {
            throw std::invalid_argument(
                "RNIC completion-ring access shape is incompatible");
        }
        break;
    case HostMemoryObjectKind::DoorbellRecord:
        if (request.service_class != PcieServiceClass::DoorbellRecord
            || request.operation != PcieOperation::HostStore
            || request.request_direction != PcieDirection::HostToDevice
            || request.mkey.has_value()) {
            throw std::invalid_argument(
                "RNIC doorbell-record access shape is incompatible");
        }
        break;
    case HostMemoryObjectKind::DataRegion:
        if ((request.service_class != PcieServiceClass::PayloadRead
             && request.service_class != PcieServiceClass::PayloadWrite)
            || (request.service_class == PcieServiceClass::PayloadRead
                && (request.operation != PcieOperation::NonPostedRead
                    || request.request_direction
                        != PcieDirection::DeviceToHost))
            || (request.service_class == PcieServiceClass::PayloadWrite
                && (request.operation != PcieOperation::PostedWrite
                    || request.request_direction
                        != PcieDirection::DeviceToHost))
            || request.mkey != allocation.mkey) {
            throw std::invalid_argument(
                "RNIC data-region access shape or MKey is incompatible");
        }
        break;
    default:
        throw std::invalid_argument("invalid RNIC host-memory object kind");
    }
}

PcieTransactionRequest metadataRequest(
    const VirtualHostMemoryConfig& config,
    const HostMemoryAccessRequest& access,
    std::uint64_t bytes,
    std::uint32_t first_byte_offset,
    Picoseconds submitted_at_ps) {
    PcieTransactionRequest request;
    request.client_id = access.client_id;
    request.client_token = access.client_token;
    request.service_class = PcieServiceClass::MttMpt;
    request.operation = PcieOperation::NonPostedRead;
    request.request_direction = PcieDirection::DeviceToHost;
    request.ordering = PcieOrdering::Independent;
    request.path_id = config.translation_path_id;
    request.ordering_domain = access.ordering_domain;
    request.useful_bytes = bytes;
    request.transfer_bytes = bytes;
    request.first_byte_offset = first_byte_offset;
    request.submitted_at_ps = submitted_at_ps;
    return request;
}

}  // namespace

class VirtualHostMemory::Impl {
public:
    explicit Impl(VirtualHostMemoryConfig config_value)
        : config(std::move(config_value)) {
        validateConfig(config);
    }

    VirtualHostMemoryConfig config;
    std::map<HostMemoryAllocationId, HostMemoryAllocation> allocations;
    std::vector<HostMemoryLifecycleEvent> lifecycle_events;
    std::map<HostMemoryDeviceOwnerId, Picoseconds> last_owner_event_time;
    std::uint64_t generation{0};
    std::uint64_t next_event_sequence{1};
};

class VirtualHostMemory::RegistrationPlan::Impl {
public:
    Impl(std::unique_ptr<VirtualHostMemory::Impl> candidate_value,
         std::uint64_t base_generation_value)
        : candidate(std::move(candidate_value)),
          base_generation(base_generation_value) {}

    std::unique_ptr<VirtualHostMemory::Impl> candidate;
    std::uint64_t base_generation{0};
};

VirtualHostMemory::RegistrationPlan::RegistrationPlan(
    std::unique_ptr<RegistrationPlan::Impl> impl)
    : impl_(std::move(impl)) {}

VirtualHostMemory::RegistrationPlan::~RegistrationPlan() = default;
VirtualHostMemory::RegistrationPlan::RegistrationPlan(
    RegistrationPlan&&) noexcept = default;
VirtualHostMemory::RegistrationPlan&
VirtualHostMemory::RegistrationPlan::operator=(
    RegistrationPlan&&) noexcept = default;

VirtualHostMemory::VirtualHostMemory(VirtualHostMemoryConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

VirtualHostMemory::~VirtualHostMemory() = default;

VirtualHostMemory::RegistrationPlan VirtualHostMemory::planRegistrations(
    const std::vector<HostMemoryAllocation>& allocations,
    Picoseconds registered_at_ps) const {
    if (allocations.empty()) {
        throw std::invalid_argument(
            "RNIC host-memory registration batch must not be empty");
    }
    auto candidate = std::make_unique<Impl>(impl_->config);
    candidate->allocations = impl_->allocations;
    candidate->lifecycle_events = impl_->lifecycle_events;
    candidate->last_owner_event_time = impl_->last_owner_event_time;
    candidate->generation = impl_->generation;
    candidate->next_event_sequence = impl_->next_event_sequence;
    const std::size_t maximum_size =
        std::numeric_limits<std::size_t>::max();
    if (candidate->lifecycle_events.size()
            > maximum_size - candidate->allocations.size()
        || allocations.size()
            > (maximum_size - candidate->lifecycle_events.size()
               - candidate->allocations.size())
                / 2) {
        throw std::overflow_error(
            "RNIC host-memory lifecycle capacity overflow");
    }
    const std::size_t future_event_capacity =
        candidate->lifecycle_events.size()
        + candidate->allocations.size() + 2 * allocations.size();
    candidate->lifecycle_events.reserve(future_event_capacity);

    std::set<std::pair<HostMemoryDeviceOwnerId, HostMemoryMkey>> mkeys;
    for (const auto& item : candidate->allocations) {
        if (item.second.mkey.has_value()) {
            mkeys.emplace(item.second.device_owner_id, *item.second.mkey);
        }
    }

    for (const HostMemoryAllocation& allocation : allocations) {
        validateAllocation(allocation);
        const auto last_time = candidate->last_owner_event_time.find(
            allocation.device_owner_id);
        if (last_time != candidate->last_owner_event_time.end()
            && registered_at_ps < last_time->second) {
            throw std::logic_error(
                "RNIC host-memory owner time regressed at registration");
        }
        if (candidate->allocations.count(allocation.allocation_id) != 0) {
            throw std::invalid_argument(
                "duplicate RNIC host-memory allocation identity");
        }
        for (const auto& existing : candidate->allocations) {
            if (extentsOverlap(allocation, existing.second)) {
                throw std::invalid_argument(
                    "overlapping RNIC host-memory virtual allocations");
            }
            if (allocation.endpoint == existing.second.endpoint) {
                for (const std::uint64_t page
                     : allocation.pages.physical_page_addresses) {
                    if (std::find(
                            existing.second.pages.physical_page_addresses.begin(),
                            existing.second.pages.physical_page_addresses.end(),
                            page)
                        != existing.second.pages.physical_page_addresses.end()) {
                        throw std::invalid_argument(
                            "overlapping RNIC host-memory physical pages");
                    }
                }
            }
        }
        if (allocation.mkey.has_value()
            && !mkeys.emplace(
                    allocation.device_owner_id, *allocation.mkey).second) {
            throw std::invalid_argument(
                "duplicate RNIC host-memory MKey for one device");
        }
        candidate->allocations.emplace(
            allocation.allocation_id, allocation);
        if (candidate->next_event_sequence
            == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "RNIC host-memory lifecycle sequence overflow");
        }
        candidate->lifecycle_events.push_back(HostMemoryLifecycleEvent{
            kHostMemoryLifecycleEventVersion,
            candidate->next_event_sequence++,
            HostMemoryLifecycleKind::Registration,
            allocation.allocation_id,
            allocation.device_owner_id,
            allocation.object_kind,
            registered_at_ps,
        });
        candidate->last_owner_event_time[allocation.device_owner_id] =
            registered_at_ps;
    }
    if (candidate->generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("RNIC host-memory generation overflow");
    }
    ++candidate->generation;
    return RegistrationPlan(std::make_unique<RegistrationPlan::Impl>(
        std::move(candidate), impl_->generation));
}

void VirtualHostMemory::commit(RegistrationPlan&& plan) {
    if (!plan.impl_ || !plan.impl_->candidate) {
        throw std::invalid_argument(
            "RNIC host-memory registration plan is empty");
    }
    if (plan.impl_->base_generation != impl_->generation) {
        throw std::logic_error(
            "RNIC host-memory registration plan is stale");
    }
    impl_.swap(plan.impl_->candidate);
    plan.impl_.reset();
}

void VirtualHostMemory::teardownOwner(
    HostMemoryDeviceOwnerId device_owner_id,
    Picoseconds teardown_at_ps) {
    if (device_owner_id == 0) {
        throw std::invalid_argument(
            "RNIC host-memory teardown owner must be nonzero");
    }
    const auto last_time = impl_->last_owner_event_time.find(device_owner_id);
    if (last_time == impl_->last_owner_event_time.end()) {
        throw std::invalid_argument(
            "RNIC host-memory teardown owner has no live allocation");
    }
    if (teardown_at_ps < last_time->second) {
        throw std::logic_error(
            "RNIC host-memory owner time regressed at teardown");
    }
    const std::size_t owner_count = liveAllocationCount(device_owner_id);
    if (owner_count == 0) {
        throw std::invalid_argument(
            "RNIC host-memory teardown owner has no live allocation");
    }
    if (owner_count
        > std::numeric_limits<std::uint64_t>::max()
            - impl_->next_event_sequence) {
        throw std::overflow_error(
            "RNIC host-memory lifecycle sequence overflow");
    }
    if (impl_->lifecycle_events.capacity() - impl_->lifecycle_events.size()
        < owner_count) {
        throw std::logic_error(
            "RNIC host-memory teardown event capacity invariant failed");
    }
    if (impl_->generation == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("RNIC host-memory generation overflow");
    }

    for (auto item = impl_->allocations.begin();
         item != impl_->allocations.end();) {
        if (item->second.device_owner_id != device_owner_id) {
            ++item;
            continue;
        }
        impl_->lifecycle_events.push_back(HostMemoryLifecycleEvent{
            kHostMemoryLifecycleEventVersion,
            impl_->next_event_sequence++,
            HostMemoryLifecycleKind::Teardown,
            item->second.allocation_id,
            item->second.device_owner_id,
            item->second.object_kind,
            teardown_at_ps,
        });
        item = impl_->allocations.erase(item);
    }
    impl_->last_owner_event_time[device_owner_id] = teardown_at_ps;
    ++impl_->generation;
}

HostMemoryAccessResult VirtualHostMemory::scheduleAccess(
    PcieFabric& fabric,
    PcieFabric::Plan& fabric_plan,
    const HostMemoryAccessRequest& request) const {
    if (request.version != kHostMemoryAccessRequestVersion) {
        throw std::invalid_argument(
            "unsupported RNIC host-memory access version");
    }
    if (request.allocation_id == 0 || request.client_id == 0
        || request.useful_bytes == 0 || request.transfer_bytes == 0
        || request.useful_bytes > request.transfer_bytes) {
        throw std::invalid_argument(
            "RNIC host-memory access identity and bytes are invalid");
    }
    const HostMemoryAllocation& target = allocation(request.allocation_id);
    validateAccessShape(target, request);
    const std::uint64_t access_end = checkedAdd(
        request.allocation_offset_bytes,
        request.transfer_bytes,
        "access extent overflow");
    if (access_end > target.length_bytes) {
        throw std::out_of_range(
            "RNIC host-memory access exceeds its allocation");
    }
    const PcieFabricConfig fabric_config = fabric.config();
    const PciePathConfig& target_path = findPath(
        fabric_config, target.path_id);
    if (target_path.endpoint != target.endpoint) {
        throw std::invalid_argument(
            "RNIC host-memory allocation and PCIe endpoint disagree");
    }
    const PciePathConfig& translation_path = findPath(
        fabric_config, impl_->config.translation_path_id);
    if (translation_path.endpoint != PcieEndpointKind::HostPinnedMemory) {
        throw std::invalid_argument(
            "RNIC host-memory translation metadata must be host-pinned");
    }

    const std::uint64_t absolute_offset = checkedAdd(
        target.virtual_address % target.pages.page_size_bytes,
        request.allocation_offset_bytes,
        "page-offset overflow");
    const std::uint64_t page_index =
        absolute_offset / target.pages.page_size_bytes;
    if (page_index >= target.pages.physical_page_addresses.size()) {
        throw std::logic_error(
            "RNIC host-memory access resolved outside its page list");
    }

    HostMemoryAccessResult result;
    result.record.allocation_id = target.allocation_id;
    result.record.object_kind = target.object_kind;
    result.record.page_index = page_index;
    result.record.submitted_at_ps = request.submitted_at_ps;
    Picoseconds access_ready_at_ps = request.submitted_at_ps;

    const auto schedule_metadata = [&](HostMemoryTranslationStage stage,
                                       std::uint64_t bytes,
                                       std::uint32_t first_byte_offset) {
        const PcieTransactionResult transaction = fabric.schedule(
            fabric_plan,
            metadataRequest(
                impl_->config,
                request,
                bytes,
                first_byte_offset,
                access_ready_at_ps));
        result.record.translation_stages.push_back(stage);
        result.record.translation_transaction_ids.push_back(
            transaction.transaction_id);
        result.translation_transactions.push_back(transaction);
        access_ready_at_ps = transaction.completed_at_ps;
    };

    if (isRing(target.object_kind)) {
        schedule_metadata(
            HostMemoryTranslationStage::QueuePageList,
            impl_->config.queue_page_list_entry_bytes,
            metadataOffset(
                impl_->config.queue_page_list_first_byte_offset,
                page_index,
                impl_->config.queue_page_list_entry_bytes));
    } else if (target.object_kind == HostMemoryObjectKind::DataRegion) {
        result.record.translation_stages.push_back(
            HostMemoryTranslationStage::Mkey);
        schedule_metadata(
            HostMemoryTranslationStage::Mpt,
            impl_->config.mpt_entry_bytes,
            metadataOffset(
                impl_->config.mpt_first_byte_offset,
                *target.mkey,
                impl_->config.mpt_entry_bytes));
        schedule_metadata(
            HostMemoryTranslationStage::Mtt,
            impl_->config.mtt_entry_bytes,
            metadataOffset(
                impl_->config.mtt_first_byte_offset,
                page_index,
                impl_->config.mtt_entry_bytes));
    }

    PcieTransactionRequest access;
    access.client_id = request.client_id;
    access.client_token = request.client_token;
    access.service_class = request.service_class;
    access.operation = request.operation;
    access.request_direction = request.request_direction;
    access.ordering = request.ordering;
    access.path_id = target.path_id;
    access.ordering_domain = request.ordering_domain;
    access.useful_bytes = request.useful_bytes;
    access.transfer_bytes = request.transfer_bytes;
    access.first_byte_offset = static_cast<std::uint32_t>(
        (target.virtual_address + request.allocation_offset_bytes) % 4096);
    access.submitted_at_ps = access_ready_at_ps;
    result.access_transaction = fabric.schedule(fabric_plan, access);
    result.record.access_transaction_id =
        result.access_transaction.transaction_id;
    result.record.completed_at_ps =
        result.access_transaction.completed_at_ps;
    return result;
}

const VirtualHostMemoryConfig& VirtualHostMemory::config() const noexcept {
    return impl_->config;
}

std::uint64_t VirtualHostMemory::generation() const noexcept {
    return impl_->generation;
}

std::size_t VirtualHostMemory::liveAllocationCount() const noexcept {
    return impl_->allocations.size();
}

std::size_t VirtualHostMemory::liveAllocationCount(
    HostMemoryDeviceOwnerId device_owner_id) const noexcept {
    return static_cast<std::size_t>(std::count_if(
        impl_->allocations.begin(),
        impl_->allocations.end(),
        [device_owner_id](const auto& item) {
            return item.second.device_owner_id == device_owner_id;
        }));
}

const HostMemoryAllocation& VirtualHostMemory::allocation(
    HostMemoryAllocationId allocation_id) const {
    const auto found = impl_->allocations.find(allocation_id);
    if (found == impl_->allocations.end()) {
        throw std::out_of_range(
            "unknown or torn-down RNIC host-memory allocation");
    }
    return found->second;
}

const std::vector<HostMemoryLifecycleEvent>&
VirtualHostMemory::lifecycleEvents() const noexcept {
    return impl_->lifecycle_events;
}

void VirtualHostMemory::validateInvariants() const {
    validateConfig(impl_->config);
    std::set<std::pair<HostMemoryDeviceOwnerId, HostMemoryMkey>> mkeys;
    std::set<std::pair<PcieEndpointKind, std::uint64_t>> physical_pages;
    for (auto lhs = impl_->allocations.begin();
         lhs != impl_->allocations.end(); ++lhs) {
        validateAllocation(lhs->second);
        if (lhs->first != lhs->second.allocation_id) {
            throw std::logic_error(
                "RNIC host-memory allocation key mismatch");
        }
        if (lhs->second.mkey.has_value()
            && !mkeys.emplace(
                    lhs->second.device_owner_id,
                    *lhs->second.mkey).second) {
            throw std::logic_error("RNIC host-memory MKey duplication");
        }
        for (const std::uint64_t page
             : lhs->second.pages.physical_page_addresses) {
            if (!physical_pages.emplace(lhs->second.endpoint, page).second) {
                throw std::logic_error(
                    "RNIC host-memory physical-page overlap");
            }
        }
        for (auto rhs = std::next(lhs); rhs != impl_->allocations.end(); ++rhs) {
            if (extentsOverlap(lhs->second, rhs->second)) {
                throw std::logic_error(
                    "RNIC host-memory virtual allocation overlap");
            }
        }
    }
    std::uint64_t expected_sequence = 1;
    std::map<HostMemoryAllocationId, HostMemoryDeviceOwnerId> registered;
    for (const HostMemoryLifecycleEvent& event : impl_->lifecycle_events) {
        if (event.version != kHostMemoryLifecycleEventVersion
            || event.sequence != expected_sequence++) {
            throw std::logic_error(
                "RNIC host-memory lifecycle sequence mismatch");
        }
        if (event.kind == HostMemoryLifecycleKind::Registration) {
            if (!registered.emplace(
                    event.allocation_id, event.device_owner_id).second) {
                throw std::logic_error(
                    "RNIC host-memory allocation registered twice");
            }
        } else if (event.kind == HostMemoryLifecycleKind::Teardown) {
            const auto live = registered.find(event.allocation_id);
            if (live == registered.end()
                || live->second != event.device_owner_id) {
                throw std::logic_error(
                    "RNIC host-memory teardown lacks registration");
            }
            registered.erase(live);
        } else {
            throw std::logic_error(
                "invalid RNIC host-memory lifecycle kind");
        }
    }
    if (expected_sequence != impl_->next_event_sequence
        || registered.size() != impl_->allocations.size()) {
        throw std::logic_error(
            "RNIC host-memory lifecycle conservation failed");
    }
    for (const auto& item : registered) {
        if (impl_->allocations.count(item.first) != 1) {
            throw std::logic_error(
                "RNIC host-memory live allocation projection mismatch");
        }
    }
}

bool sameVirtualHostMemoryConfig(
    const VirtualHostMemoryConfig& lhs,
    const VirtualHostMemoryConfig& rhs) noexcept {
    return lhs.version == rhs.version
        && lhs.translation_path_id == rhs.translation_path_id
        && lhs.queue_page_list_entry_bytes
            == rhs.queue_page_list_entry_bytes
        && lhs.mpt_entry_bytes == rhs.mpt_entry_bytes
        && lhs.mtt_entry_bytes == rhs.mtt_entry_bytes
        && lhs.queue_page_list_first_byte_offset
            == rhs.queue_page_list_first_byte_offset
        && lhs.mpt_first_byte_offset == rhs.mpt_first_byte_offset
        && lhs.mtt_first_byte_offset == rhs.mtt_first_byte_offset;
}

const char* toString(HostMemoryObjectKind kind) noexcept {
    switch (kind) {
    case HostMemoryObjectKind::QpcIcm:
        return "qpc_icm";
    case HostMemoryObjectKind::SqRing:
        return "sq_ring";
    case HostMemoryObjectKind::RqRing:
        return "rq_ring";
    case HostMemoryObjectKind::CqRing:
        return "cq_ring";
    case HostMemoryObjectKind::DoorbellRecord:
        return "doorbell_record";
    case HostMemoryObjectKind::DataRegion:
        return "data_region";
    default:
        return "invalid";
    }
}

const char* toString(HostMemoryOwnerKind kind) noexcept {
    switch (kind) {
    case HostMemoryOwnerKind::QueuePair:
        return "queue_pair";
    case HostMemoryOwnerKind::SendQueue:
        return "send_queue";
    case HostMemoryOwnerKind::ReceiveQueue:
        return "receive_queue";
    case HostMemoryOwnerKind::CompletionQueue:
        return "completion_queue";
    case HostMemoryOwnerKind::MemoryRegion:
        return "memory_region";
    default:
        return "invalid";
    }
}

const char* toString(HostMemoryLifecycleKind kind) noexcept {
    switch (kind) {
    case HostMemoryLifecycleKind::Registration:
        return "registration";
    case HostMemoryLifecycleKind::Teardown:
        return "teardown";
    default:
        return "invalid";
    }
}

const char* toString(HostMemoryTranslationStage stage) noexcept {
    switch (stage) {
    case HostMemoryTranslationStage::QueuePageList:
        return "queue_page_list";
    case HostMemoryTranslationStage::Mkey:
        return "mkey";
    case HostMemoryTranslationStage::Mpt:
        return "mpt";
    case HostMemoryTranslationStage::Mtt:
        return "mtt";
    default:
        return "invalid";
    }
}

}  // namespace simllm::rnic
