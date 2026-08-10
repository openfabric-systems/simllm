#ifndef SIMLLM_RNIC_HOST_MEMORY_H
#define SIMLLM_RNIC_HOST_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "simllm/rnic/pcie_fabric.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kVirtualHostMemoryConfigVersion = 1;
inline constexpr std::uint32_t kHostMemoryAllocationVersion = 1;
inline constexpr std::uint32_t kHostMemoryPageGeometryVersion = 1;
inline constexpr std::uint32_t kHostMemoryLifecycleEventVersion = 1;
inline constexpr std::uint32_t kHostMemoryAccessRequestVersion = 1;
inline constexpr std::uint32_t kHostMemoryAccessRecordVersion = 2;

using HostMemoryAllocationId = std::uint64_t;
using HostMemoryOwnerId = std::uint64_t;
using HostMemoryDeviceOwnerId = std::uint64_t;
using HostMemoryMkey = std::uint32_t;

enum class HostMemoryObjectKind : std::uint8_t {
    QpcIcm,
    SqRing,
    RqRing,
    CqRing,
    DoorbellRecord,
    DataRegion,
    DescriptorQueue,
};

enum class HostMemoryOwnerKind : std::uint8_t {
    QueuePair,
    SendQueue,
    ReceiveQueue,
    CompletionQueue,
    MemoryRegion,
    SubmissionProducer,
};

enum class HostMemoryLifecycleKind : std::uint8_t {
    Registration,
    Teardown,
};

enum class HostMemoryTranslationStage : std::uint8_t {
    QueuePageList,
    Mkey,
    Mpt,
    Mtt,
};

struct VirtualHostMemoryConfig {
    std::uint32_t version{kVirtualHostMemoryConfigVersion};
    std::uint32_t translation_path_id{2};
    std::uint64_t queue_page_list_entry_bytes{8};
    std::uint64_t mpt_entry_bytes{64};
    std::uint64_t mtt_entry_bytes{8};
    std::uint32_t queue_page_list_first_byte_offset{0};
    std::uint32_t mpt_first_byte_offset{0};
    std::uint32_t mtt_first_byte_offset{0};
};

bool sameVirtualHostMemoryConfig(
    const VirtualHostMemoryConfig& lhs,
    const VirtualHostMemoryConfig& rhs) noexcept;

struct HostMemoryPageGeometry {
    std::uint32_t version{kHostMemoryPageGeometryVersion};
    std::uint64_t page_size_bytes{4096};
    std::vector<std::uint64_t> physical_page_addresses;
};

struct HostMemoryAllocation {
    std::uint32_t version{kHostMemoryAllocationVersion};
    HostMemoryAllocationId allocation_id{0};
    HostMemoryDeviceOwnerId device_owner_id{0};
    HostMemoryOwnerKind owner_kind{HostMemoryOwnerKind::QueuePair};
    HostMemoryOwnerId owner_id{0};
    HostMemoryObjectKind object_kind{HostMemoryObjectKind::QpcIcm};
    PcieEndpointKind endpoint{PcieEndpointKind::HostPinnedMemory};
    std::uint32_t path_id{0};
    std::uint64_t virtual_address{0};
    std::uint64_t length_bytes{0};
    HostMemoryPageGeometry pages;
    std::optional<HostMemoryMkey> mkey;
};

struct HostMemoryLifecycleEvent {
    std::uint32_t version{kHostMemoryLifecycleEventVersion};
    std::uint64_t sequence{0};
    HostMemoryLifecycleKind kind{HostMemoryLifecycleKind::Registration};
    HostMemoryAllocationId allocation_id{0};
    HostMemoryDeviceOwnerId device_owner_id{0};
    HostMemoryObjectKind object_kind{HostMemoryObjectKind::QpcIcm};
    Picoseconds occurred_at_ps{0};
};

struct HostMemoryAccessRequest {
    std::uint32_t version{kHostMemoryAccessRequestVersion};
    HostMemoryAllocationId allocation_id{0};
    std::optional<HostMemoryMkey> mkey;
    std::uint32_t client_id{0};
    std::uint64_t client_token{0};
    PcieServiceClass service_class{PcieServiceClass::Count};
    PcieOperation operation{PcieOperation::PostedWrite};
    PcieDirection request_direction{PcieDirection::HostToDevice};
    PcieOrdering ordering{PcieOrdering::VisibilityDependency};
    std::uint64_t ordering_domain{0};
    std::uint64_t allocation_offset_bytes{0};
    std::uint64_t useful_bytes{0};
    std::uint64_t transfer_bytes{0};
    Picoseconds submitted_at_ps{0};
};

struct HostMemoryAccessRecord {
    std::uint32_t version{kHostMemoryAccessRecordVersion};
    HostMemoryAllocationId allocation_id{0};
    HostMemoryObjectKind object_kind{HostMemoryObjectKind::QpcIcm};
    std::uint32_t client_id{0};
    std::uint64_t client_token{0};
    std::uint64_t page_index{0};
    std::vector<HostMemoryTranslationStage> translation_stages;
    std::vector<std::uint64_t> translation_transaction_ids;
    std::uint64_t access_transaction_id{0};
    Picoseconds submitted_at_ps{0};
    Picoseconds completed_at_ps{0};
};

struct HostMemoryAccessResult {
    HostMemoryAccessRecord record;
    std::vector<PcieTransactionResult> translation_transactions;
    PcieTransactionResult access_transaction;
};

class VirtualHostMemory {
public:
    class RegistrationPlan {
    public:
        ~RegistrationPlan();
        RegistrationPlan(RegistrationPlan&&) noexcept;
        RegistrationPlan& operator=(RegistrationPlan&&) noexcept;

        RegistrationPlan(const RegistrationPlan&) = delete;
        RegistrationPlan& operator=(const RegistrationPlan&) = delete;

    private:
        class Impl;
        explicit RegistrationPlan(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
        friend class VirtualHostMemory;
    };

    explicit VirtualHostMemory(VirtualHostMemoryConfig config = {});
    ~VirtualHostMemory();

    VirtualHostMemory(const VirtualHostMemory&) = delete;
    VirtualHostMemory& operator=(const VirtualHostMemory&) = delete;
    VirtualHostMemory(VirtualHostMemory&&) = delete;
    VirtualHostMemory& operator=(VirtualHostMemory&&) = delete;

    RegistrationPlan planRegistrations(
        const std::vector<HostMemoryAllocation>& allocations,
        Picoseconds registered_at_ps) const;
    void commit(RegistrationPlan&& plan);
    void teardownOwner(
        HostMemoryDeviceOwnerId device_owner_id,
        Picoseconds teardown_at_ps);

    HostMemoryAccessResult scheduleAccess(
        PcieFabric& fabric,
        PcieFabric::Plan& fabric_plan,
        const HostMemoryAccessRequest& request) const;

    const VirtualHostMemoryConfig& config() const noexcept;
    std::uint64_t generation() const noexcept;
    std::size_t liveAllocationCount() const noexcept;
    std::size_t liveAllocationCount(
        HostMemoryDeviceOwnerId device_owner_id) const noexcept;
    const HostMemoryAllocation& allocation(
        HostMemoryAllocationId allocation_id) const;
    const std::vector<HostMemoryLifecycleEvent>& lifecycleEvents() const noexcept;
    void validateInvariants() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

const char* toString(HostMemoryObjectKind kind) noexcept;
const char* toString(HostMemoryOwnerKind kind) noexcept;
const char* toString(HostMemoryLifecycleKind kind) noexcept;
const char* toString(HostMemoryTranslationStage stage) noexcept;

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_HOST_MEMORY_H
