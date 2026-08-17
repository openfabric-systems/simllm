#ifndef SIMLLM_RNIC_RNIC_DEVICE_H
#define SIMLLM_RNIC_RNIC_DEVICE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "simllm/rnic/host_memory.h"
#include "simllm/rnic/pcie_fabric.h"
#include "simllm/rnic/work_queue.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicDeviceConfigVersion = 2;
inline constexpr std::uint32_t kRnicDeviceIdentityVersion = 1;
inline constexpr std::uint32_t kRnicQpcConfigVersion = 1;
inline constexpr std::uint32_t kRnicDmaConfigVersion = 1;
inline constexpr std::uint32_t kRnicNetworkConfigVersion = 1;
inline constexpr std::uint32_t kRnicHostMemoryConfigVersion = 1;

struct RnicDeviceIdentity {
    std::uint32_t version{kRnicDeviceIdentityVersion};
    std::uint32_t qpn{1};
    PolicyContextToken policy_context_token{1};
};

struct RnicQpcConfig {
    std::uint32_t version{kRnicQpcConfigVersion};
    // V1 uses WorkQueueConfig::qpc_lookup_service_ps as the scalar QPC
    // compatibility module. Disabling it makes qpc_ready_at_ps not
    // applicable while retaining device identity.
    bool enabled{true};
};

struct RnicDmaConfig {
    std::uint32_t version{kRnicDmaConfigVersion};
    bool enabled{false};
    PcieFabricConfig fabric{defaultPcieFabricConfig()};
    WorkQueuePcieBinding work_queue;
    // DMA off leaves this field inert. Owned DMA requires zero. Shared DMA
    // uses a nonzero value only when either ordering domain must be derived;
    // an explicit pair requires zero. The resolved pair is 2 * namespace
    // plus the submission/completion bit.
    std::uint64_t shared_ordering_domain_namespace{0};
    // Zero leaves the device unattributed on the fabric: its transactions are
    // charged only to the service-class ledger, which is the shape every
    // accepted BACK-10, BACK-19 and BACK-20 artifact was produced with. A
    // nonzero identity is claimed on the fabric for this device's lifetime and
    // requires the fabric to name a host endpoint identity.
    PcieEndpointId fabric_endpoint_id{0};
};

struct RnicNetworkConfig {
    std::uint32_t version{kRnicNetworkConfigVersion};
    // Enabled requires an injected external NetworkPort. Disabled selects
    // the owned inert port and rejects an external pointer.
    bool enabled{false};
};

struct RnicHostMemoryConfig {
    std::uint32_t version{kRnicHostMemoryConfigVersion};
    bool enabled{false};
    HostMemoryDeviceOwnerId device_owner_id{0};
    VirtualHostMemoryConfig registry;
    WorkQueueHostMemoryBinding work_queue;
    std::vector<HostMemoryAllocation> allocations;
    // Device owners this device grants read access to its own data regions.
    // Empty is the closed default: no other device may name them.
    std::vector<HostMemoryDeviceOwnerId> peer_read_grants;
};

struct RnicDeviceConfig {
    std::uint32_t version{kRnicDeviceConfigVersion};
    RnicDeviceIdentity identity;
    WorkQueueConfig work_queue;
    RnicQpcConfig qpc;
    RnicDmaConfig dma;
    RnicNetworkConfig network;
    RnicHostMemoryConfig host_memory;
    RnicSubmissionConfig submission;
};

struct RnicDeviceAttachments {
    // Shared ownership makes the external fabric's stable-address lifetime
    // explicit. A null pointer asks the device to heap-own its configured
    // fabric when DMA is enabled.
    std::shared_ptr<PcieFabric> shared_pcie_fabric;
    // The attached registry retains lifecycle evidence after device teardown.
    // A null pointer asks an enabled device to own its registry.
    std::shared_ptr<VirtualHostMemory> shared_host_memory;
    // The caller owns an injected port and must outlive this device.
    NetworkPort* network_port{nullptr};
};

enum class RnicStageApplicability : std::uint8_t {
    Applicable,
    NotApplicable,
};

struct RnicDeviceStageReport {
    RnicStageApplicability scalar_doorbell_service{
        RnicStageApplicability::Applicable};
    RnicStageApplicability scalar_wqe_fetch_service{
        RnicStageApplicability::Applicable};
    RnicStageApplicability qpc_lookup{
        RnicStageApplicability::Applicable};
    RnicStageApplicability scalar_cqe_write_service{
        RnicStageApplicability::Applicable};
    RnicStageApplicability pcie_doorbell_record{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability pcie_uar_doorbell{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability pcie_wqe_read{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability pcie_qpc_icm{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability pcie_mtt_mpt{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability pcie_payload_read{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability pcie_cqe_write{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability external_network{
        RnicStageApplicability::NotApplicable};
    RnicStageApplicability inert_network{
        RnicStageApplicability::Applicable};
};

const char* toString(RnicStageApplicability applicability) noexcept;

class RnicDevice {
public:
    explicit RnicDevice(
        RnicDeviceConfig config,
        RnicDeviceAttachments attachments = {});
    ~RnicDevice();

    RnicDevice(const RnicDevice&) = delete;
    RnicDevice& operator=(const RnicDevice&) = delete;
    // The bound WorkQueue observes stable port and fabric addresses. Keeping
    // the composer non-movable makes that lifetime rule explicit.
    RnicDevice(RnicDevice&&) = delete;
    RnicDevice& operator=(RnicDevice&&) = delete;

    PostResult postSend(const WorkRequest& request, Picoseconds now_ps);
    PostBatchResult postSendBatch(
        const std::vector<WorkRequest>& requests,
        Picoseconds now_ps);
    DoorbellBatch ringDoorbell(Picoseconds now_ps);
    DoorbellBatch ringDoorbell(
        Picoseconds now_ps,
        const RnicProducerTaskLink& producer_task);

    // External events must be delivered before progress() at the same
    // timestamp. The inert owned port is pumped internally in that order.
    void onNetworkEvent(const NetworkEvent& event);
    std::size_t progress(Picoseconds now_ps);
    std::vector<CompletionEntry> pollCompletionQueue(
        std::size_t max_entries,
        Picoseconds now_ps);
    void teardownHostMemory(Picoseconds now_ps);

    // Standalone module probes submit through the device so the same caller
    // clock covers direct fabric traffic and queue traffic. Rejected fabric
    // requests do not advance that clock.
    PcieTransactionResult submitPcie(
        const PcieTransactionRequest& request);
    // Rejects an attribution whose requester identity this device does not
    // hold, so a probe cannot charge another device's endpoint.
    PcieTransactionResult submitPcie(
        const PcieTransactionRequest& request,
        const PcieEndpointAttribution& attribution);

    std::optional<Picoseconds> nextEventTime() const;
    bool hasPendingPhysicalWork() const noexcept;
    bool fatal() const noexcept;
    std::size_t occupiedSqEntries() const noexcept;
    std::size_t completionQueueDepth() const noexcept;
    std::size_t unpublishedWqeCount() const noexcept;

    const RnicDeviceConfig& config() const noexcept;
    const RnicDeviceStageReport& stageReport() const noexcept;
    bool usesSharedPcieFabric() const noexcept;
    std::optional<WorkQueuePcieBinding> pcieBinding() const;
    const PcieFabric* pcieFabric() const noexcept;
    const VirtualHostMemory* hostMemory() const noexcept;
    const WorkQueueConfig& workQueueConfig() const noexcept;
    const WorkQueueCounters& counters() const noexcept;
    const std::vector<WqeRecord>& records() const noexcept;
    const std::vector<EvidenceEvent>& evidence() const noexcept;
    const std::vector<HostMemoryAccessRecord>& memoryAccesses() const noexcept;
    const std::optional<RnicSubmissionProfile>& submissionProfile()
        const noexcept;
    const std::vector<RnicSubmissionRecord>& submissionRecords()
        const noexcept;
    const std::vector<RnicCqConsumptionRecord>& cqConsumptionRecords()
        const noexcept;
    const WqeRecord& wqe(WqeId wqe_id) const;

    void validateInvariants() const;

private:
    class InertNetworkPort;
    void validateCallerTime(Picoseconds now_ps) const;
    void observeCallerTime(Picoseconds now_ps);
    void requireHostMemoryLive() const;

    RnicDeviceConfig config_;
    RnicDeviceStageReport stage_report_;
    std::shared_ptr<PcieFabric> pcie_fabric_;
    std::shared_ptr<VirtualHostMemory> host_memory_;
    std::unique_ptr<InertNetworkPort> inert_network_port_;
    NetworkPort* network_port_{nullptr};
    std::unique_ptr<WorkQueue> work_queue_;
    bool claimed_ordering_domains_{false};
    std::uint64_t claimed_submission_domain_{0};
    std::uint64_t claimed_completion_domain_{0};
    bool claimed_fabric_endpoint_{false};
    bool claimed_host_memory_owner_{false};
    bool host_memory_registered_{false};
    Picoseconds last_caller_time_ps_{0};
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_DEVICE_H
