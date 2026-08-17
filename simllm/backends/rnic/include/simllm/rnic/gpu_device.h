#ifndef SIMLLM_RNIC_GPU_DEVICE_H
#define SIMLLM_RNIC_GPU_DEVICE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "simllm/rnic/host_memory.h"
#include "simllm/rnic/pcie_fabric.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kGpuDeviceConfigVersion = 1;
inline constexpr std::uint32_t kGpuDeviceIdentityVersion = 1;
inline constexpr std::uint32_t kGpuFabricConfigVersion = 1;
inline constexpr std::uint32_t kGpuMemoryConfigVersion = 1;
inline constexpr std::uint32_t kGpuFabricTransferVersion = 1;
inline constexpr std::uint32_t kGpuFabricTransferRecordVersion = 1;

struct GpuDeviceIdentity {
    std::uint32_t version{kGpuDeviceIdentityVersion};
    std::uint64_t gpu_id{1};
    // PCIe requester identity carried on this device's transactions. It is
    // separate from the GPU number and from every RNIC identity.
    std::uint32_t requester_id{0};
};

struct GpuFabricConfig {
    std::uint32_t version{kGpuFabricConfigVersion};
    bool enabled{false};
    PcieFabricConfig fabric{defaultPcieFabricConfig()};
    // Enabled requires a nonzero identity distinct from the fabric's host
    // endpoint, and requires the fabric to name that host endpoint. There is
    // no unattributed GPU shape to preserve, so identity fails closed.
    PcieEndpointId endpoint_id{0};
    // The dependency domain this device claims for its own transfers. It is a
    // raw value in the same flat namespace an RNIC claims into, where a shared
    // RNIC derives the pair (2 * namespace, 2 * namespace + 1) from
    // RnicDmaConfig::shared_ordering_domain_namespace, so a GPU domain has to
    // avoid both halves of every RNIC namespace on the same fabric. This device
    // claims one domain rather than a pair, so its reads and its writes share
    // one posted-visibility and non-posted-completion horizon.
    std::uint64_t ordering_domain{0};
};

struct GpuMemoryConfig {
    std::uint32_t version{kGpuMemoryConfigVersion};
    bool enabled{false};
    HostMemoryDeviceOwnerId device_owner_id{0};
    VirtualHostMemoryConfig registry;
    // Data regions only. Queue objects belong to an RNIC and are rejected
    // here, so this device cannot quietly become a second queue authority.
    std::vector<HostMemoryAllocation> allocations;
    // Device owners this device grants read access to its data regions. This
    // is the owner half of the cross-device contract; the reader's descriptor
    // supplies the other half.
    std::vector<HostMemoryDeviceOwnerId> peer_read_grants;
};

struct GpuDeviceConfig {
    std::uint32_t version{kGpuDeviceConfigVersion};
    GpuDeviceIdentity identity;
    GpuFabricConfig fabric;
    GpuMemoryConfig memory;
};

struct GpuDeviceAttachments {
    // Shared ownership makes the external fabric's stable-address lifetime
    // explicit. A null pointer asks the device to heap-own its configured
    // fabric when its fabric module is enabled.
    std::shared_ptr<PcieFabric> shared_pcie_fabric;
    // A null pointer asks an enabled device to own its registry. Sharing one
    // registry with an RNIC is what makes a cross-device region claim
    // expressible at all.
    std::shared_ptr<VirtualHostMemory> shared_memory;
};

// One payload movement this device drives across the fabric: staging a chunk
// into a host bounce buffer is a posted write, reading a host-pinned region is
// a non-posted read. Traffic inside any device's own memory never appears here
// and is rejected, because it does not cross the fabric: this device's own GPU
// memory sits behind its own port, and a peer's device-local memory is the
// unmodeled peer-to-peer leg. This device has no service model of its own; the
// fabric charges the transfer and the GPU's internal copy engine and peer ports
// stay with the compute-side tasks that own them.
//
// A transfer bypasses VirtualHostMemory::scheduleAccess on purpose: a copy
// engine addresses the region directly, so no MKey, MPT or MTT read is emitted
// and the transfer never appears in an RNIC's memoryAccesses(). Its only
// records are transferRecords() here and the fabric's endpoint ledger.
struct GpuFabricTransfer {
    std::uint32_t version{kGpuFabricTransferVersion};
    HostMemoryAllocationId allocation_id{0};
    std::uint64_t client_token{0};
    PcieServiceClass service_class{PcieServiceClass::PayloadWrite};
    PcieOperation operation{PcieOperation::PostedWrite};
    PcieOrdering ordering{PcieOrdering::Independent};
    std::uint64_t allocation_offset_bytes{0};
    std::uint64_t transfer_bytes{0};
    // Zero for a region this device owns. Nonzero names the peer owner whose
    // grant makes the access legal.
    HostMemoryDeviceOwnerId peer_device_owner_id{0};
    Picoseconds submitted_at_ps{0};
};

struct GpuFabricTransferRecord {
    std::uint32_t version{kGpuFabricTransferRecordVersion};
    std::uint64_t sequence{0};
    HostMemoryAllocationId allocation_id{0};
    PcieServiceClass service_class{PcieServiceClass::PayloadWrite};
    PcieEndpointId requester_endpoint_id{0};
    PcieEndpointId completer_endpoint_id{0};
    PcieDeviceKind completer_kind{PcieDeviceKind::Host};
    std::uint64_t transfer_bytes{0};
    std::uint64_t transaction_id{0};
    std::uint64_t modeled_link_bytes{0};
    Picoseconds credit_wait_ps{0};
    Picoseconds link_queue_wait_ps{0};
    Picoseconds submitted_at_ps{0};
    Picoseconds completed_at_ps{0};
};

class GpuDevice {
public:
    explicit GpuDevice(
        GpuDeviceConfig config,
        GpuDeviceAttachments attachments = {});
    ~GpuDevice();

    GpuDevice(const GpuDevice&) = delete;
    GpuDevice& operator=(const GpuDevice&) = delete;
    // The fabric and registry observe this object's address as the owner of its
    // claims. Keeping the composer non-movable makes that lifetime rule
    // explicit, exactly as it is for the RNIC composer.
    GpuDevice(GpuDevice&&) = delete;
    GpuDevice& operator=(GpuDevice&&) = delete;

    // Rejects an illegal region claim before any fabric or registry state
    // moves, so a refused transfer leaves both exactly as they were.
    GpuFabricTransferRecord transfer(const GpuFabricTransfer& request);
    void teardownMemory(Picoseconds now_ps);

    const GpuDeviceConfig& config() const noexcept;
    const std::vector<GpuFabricTransferRecord>& transferRecords()
        const noexcept;
    const PcieFabric* pcieFabric() const noexcept;
    const VirtualHostMemory* memory() const noexcept;
    bool usesSharedPcieFabric() const noexcept;
    std::uint64_t transferredBytes() const noexcept;

    void validateInvariants() const;

private:
    GpuDeviceConfig config_;
    std::shared_ptr<PcieFabric> pcie_fabric_;
    std::shared_ptr<VirtualHostMemory> memory_;
    bool shared_pcie_fabric_{false};
    bool claimed_ordering_domain_{false};
    bool claimed_fabric_endpoint_{false};
    bool claimed_memory_owner_{false};
    bool memory_registered_{false};
    std::uint64_t next_transfer_sequence_{1};
    std::uint64_t transferred_bytes_{0};
    Picoseconds last_transfer_time_ps_{0};
    std::vector<GpuFabricTransferRecord> transfer_records_;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_GPU_DEVICE_H
