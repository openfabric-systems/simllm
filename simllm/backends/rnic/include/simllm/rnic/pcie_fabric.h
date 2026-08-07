#ifndef SIMLLM_RNIC_PCIE_FABRIC_H
#define SIMLLM_RNIC_PCIE_FABRIC_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "simllm/rnic/network_port.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kPcieFabricConfigVersion = 1;
inline constexpr std::uint32_t kPcieTransactionAbiVersion = 1;
inline constexpr std::uint32_t kPcieTransactionResultVersion = 1;

enum class PcieServiceClass : std::uint8_t {
    UarDoorbell,
    BlueFlame,
    DoorbellRecord,
    WqeRead,
    QpcIcm,
    MttMpt,
    PayloadRead,
    PayloadWrite,
    CqeWrite,
    Command,
    Interrupt,
    OdpIommuFault,
    Count,
};

inline constexpr std::size_t kPcieServiceClassCount =
    static_cast<std::size_t>(PcieServiceClass::Count);

enum class PcieOperation : std::uint8_t {
    HostStore,
    PostedWrite,
    NonPostedRead,
};

enum class PcieDirection : std::uint8_t {
    HostToDevice,
    DeviceToHost,
};

enum class PcieOrdering : std::uint8_t {
    // This is an explicit simulator dependency on modeled visibility. It is
    // not the PCIe Relaxed Ordering, IDO, TC or VC ordering matrix.
    VisibilityDependency,
    Independent,
};

enum class PcieGeneration : std::uint8_t {
    Gen1 = 1,
    Gen2 = 2,
    Gen3 = 3,
    Gen4 = 4,
    Gen5 = 5,
};

enum class PcieEndpointKind : std::uint8_t {
    MmioBar,
    HostPinnedMemory,
    GpuMemory,
    DeviceMemory,
};

struct PcieLatencyProfile {
    // V1 accepts exactly one fixed sample. Chronological arbitration for a
    // multi-sample replay profile remains open BACK-10 work.
    std::vector<Picoseconds> samples_ps{0};
};

struct PciePathConfig {
    std::uint32_t path_id{0};
    PcieEndpointKind endpoint{PcieEndpointKind::HostPinnedMemory};
    bool enabled{true};
    Picoseconds base_latency_ps{0};
    Picoseconds numa_penalty_ps{0};
    Picoseconds iommu_penalty_ps{0};
    Picoseconds acs_penalty_ps{0};
    Picoseconds switch_penalty_ps{0};
    Picoseconds ddio_miss_penalty_ps{0};
    Picoseconds gpu_direct_penalty_ps{0};
};

struct PcieCreditConfig {
    std::uint32_t posted_header_credits{64};
    std::uint32_t posted_data_credits{4096};
    std::uint32_t nonposted_header_credits{64};
    // Memory Read requests carry no data and do not debit this standard pool.
    std::uint32_t nonposted_data_credits{1};
    std::uint32_t completion_header_credits{64};
    std::uint32_t completion_data_credits{4096};
};

struct PcieFabricConfig {
    std::uint32_t version{kPcieFabricConfigVersion};
    PcieGeneration generation{PcieGeneration::Gen5};
    std::uint16_t lane_count{16};
    std::uint32_t max_payload_size_bytes{256};
    std::uint32_t max_read_request_size_bytes{512};
    std::uint32_t read_completion_boundary_bytes{64};
    std::uint32_t posted_write_overhead_bytes{24};
    std::uint32_t read_request_overhead_bytes{24};
    std::uint32_t completion_overhead_bytes{20};
    std::uint32_t data_credit_unit_bytes{16};
    PcieCreditConfig host_to_device_credits;
    PcieCreditConfig device_to_host_credits;
    std::uint32_t max_outstanding_read_requests{64};
    std::uint64_t completion_buffer_bytes{64 * 1024};
    std::uint64_t max_tlps_per_transaction{1'000'000};
    Picoseconds credit_return_latency_ps{0};
    Picoseconds completion_buffer_release_latency_ps{0};
    PcieLatencyProfile host_store_latency_ps;
    PcieLatencyProfile posted_write_visibility_latency_ps;
    PcieLatencyProfile read_completion_latency_ps;
    std::vector<PciePathConfig> paths;
};

// Returns an explicit synthetic baseline, not an asserted ConnectX-7 profile.
// Path 1 is the MMIO BAR and path 2 is local pinned host memory.
PcieFabricConfig defaultPcieFabricConfig();

struct PcieTransactionRequest {
    std::uint32_t abi_version{kPcieTransactionAbiVersion};
    std::uint32_t client_id{0};
    std::uint64_t client_token{0};
    PcieServiceClass service_class{PcieServiceClass::Count};
    PcieOperation operation{PcieOperation::PostedWrite};
    PcieDirection request_direction{PcieDirection::HostToDevice};
    PcieOrdering ordering{PcieOrdering::VisibilityDependency};
    std::uint32_t path_id{0};
    // Domain zero is a valid global conservative dependency domain. Clients
    // should use nonzero namespaces when unrelated queues may share a fabric.
    std::uint64_t ordering_domain{0};
    std::uint64_t useful_bytes{0};
    std::uint64_t transfer_bytes{0};
    // Only low address bits are needed for DWORD, RCB and 4 KiB splitting.
    std::uint32_t first_byte_offset{0};
    Picoseconds submitted_at_ps{0};
};

struct PcieDirectionAccounting {
    std::uint64_t tlps{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t overhead_bytes{0};
    std::uint64_t modeled_link_bytes{0};
};

struct PcieWaitAccounting {
    Picoseconds ordering_ps{0};
    Picoseconds outstanding_read_ps{0};
    Picoseconds completion_buffer_ps{0};
    Picoseconds credit_ps{0};
    Picoseconds link_queue_ps{0};
};

struct PcieServiceDelayAccounting {
    Picoseconds host_store_ps{0};
    Picoseconds posted_write_visibility_ps{0};
    Picoseconds read_completion_ps{0};
};

struct PciePathDelayAccounting {
    Picoseconds base_ps{0};
    Picoseconds numa_ps{0};
    Picoseconds iommu_ps{0};
    Picoseconds acs_ps{0};
    Picoseconds switch_ps{0};
    Picoseconds ddio_miss_ps{0};
    Picoseconds gpu_direct_ps{0};
};

struct PcieTransactionResult {
    std::uint32_t version{kPcieTransactionResultVersion};
    std::uint64_t transaction_id{0};
    std::uint32_t client_id{0};
    std::uint64_t client_token{0};
    PcieServiceClass service_class{PcieServiceClass::Count};
    PcieOperation operation{PcieOperation::PostedWrite};
    PcieDirection request_direction{PcieDirection::HostToDevice};
    PcieOrdering ordering{PcieOrdering::VisibilityDependency};
    std::uint32_t path_id{0};
    std::uint64_t ordering_domain{0};
    std::uint64_t useful_bytes{0};
    std::uint64_t transferred_bytes{0};
    std::uint64_t host_store_bytes{0};
    std::uint64_t request_tlps{0};
    std::uint64_t completion_tlps{0};
    PcieDirectionAccounting host_to_device;
    PcieDirectionAccounting device_to_host;
    PcieWaitAccounting waits;
    PcieServiceDelayAccounting service_delay;
    PciePathDelayAccounting path_delay;
    std::uint64_t latency_samples_used{0};
    Picoseconds submitted_at_ps{0};
    Picoseconds first_issue_at_ps{0};
    Picoseconds request_finished_at_ps{0};
    // Present only for a non-posted read response. For a transfer split into
    // multiple MRds this is the first response-ready timestamp. Posted writes
    // have no PCIe transaction-layer completion; completed_at_ps is modeled
    // visibility.
    std::optional<Picoseconds> completion_ready_at_ps;
    Picoseconds completed_at_ps{0};
};

struct PcieClassAccounting {
    std::uint64_t transactions{0};
    std::uint64_t useful_bytes{0};
    std::uint64_t transferred_bytes{0};
    std::uint64_t host_store_bytes{0};
    PcieDirectionAccounting host_to_device;
    PcieDirectionAccounting device_to_host;
    PcieWaitAccounting waits;
    PcieServiceDelayAccounting service_delay;
    PciePathDelayAccounting path_delay;
    std::uint64_t latency_samples_used{0};
};

class PcieFabric {
public:
    class Plan {
    public:
        ~Plan();
        Plan(Plan&&) noexcept;
        Plan& operator=(Plan&&) noexcept;

        Plan(const Plan&) = delete;
        Plan& operator=(const Plan&) = delete;

    private:
        class Impl;
        explicit Plan(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
        friend class PcieFabric;
    };

    explicit PcieFabric(PcieFabricConfig config);
    ~PcieFabric();

    PcieFabric(const PcieFabric&) = delete;
    PcieFabric& operator=(const PcieFabric&) = delete;
    // Plans and bound WorkQueues use this object's stable address. The fabric
    // must outlive them and cannot be moved in v1.
    PcieFabric(PcieFabric&&) = delete;
    PcieFabric& operator=(PcieFabric&&) = delete;

    // A plan owns a complete private fabric snapshot. Scheduling failures or
    // discarding a plan cannot mutate the shared fabric. Commit rejects a
    // stale snapshot before swapping any state.
    Plan beginPlan() const;
    PcieTransactionResult schedule(
        Plan& plan,
        const PcieTransactionRequest& request);
    void commit(Plan&& plan);
    PcieTransactionResult submit(const PcieTransactionRequest& request);

    PcieFabricConfig config() const;
    std::uint64_t generation() const noexcept;
    PcieClassAccounting accounting(PcieServiceClass service_class) const;
    PcieClassAccounting totalAccounting() const;
    void validateInvariants() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

const char* toString(PcieServiceClass service_class) noexcept;
const char* toString(PcieOperation operation) noexcept;
const char* toString(PcieDirection direction) noexcept;

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_PCIE_FABRIC_H
