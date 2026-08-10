#ifndef SIMLLM_RNIC_WORK_QUEUE_H
#define SIMLLM_RNIC_WORK_QUEUE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "simllm/rnic/host_memory.h"
#include "simllm/rnic/network_port.h"
#include "simllm/rnic/submission.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kWorkQueueConfigVersion = 1;
inline constexpr std::uint32_t kWorkQueuePcieBindingVersion = 1;
inline constexpr std::uint32_t kWorkQueueHostMemoryBindingVersion = 1;
inline constexpr std::uint32_t kWorkRequestDataMemoryVersion = 1;

class PcieFabric;
class RnicDevice;

enum class WqeOpcode {
    Send,
};

enum class WqeState {
    Posted,
    Doorbelled,
    InFlight,
    AwaitingOrderedRetirement,
    RetiredUnsignaled,
    CompletionPending,
    CqeVisible,
    Reclaimed,
    Completed,
    Error,
};

enum class CompletionStatus {
    Success,
    TransportError,
    NetworkRejected,
};

enum class PostStatus {
    Accepted,
    SqFull,
    Fatal,
};

inline constexpr std::uint32_t kCqeValidWrId = 1U << 0;
inline constexpr std::uint32_t kCqeValidQpn = 1U << 1;
inline constexpr std::uint32_t kCqeValidOpcode = 1U << 2;
inline constexpr std::uint32_t kCqeValidStatus = 1U << 3;
inline constexpr std::uint32_t kCqeValidVendorSyndrome = 1U << 4;
inline constexpr std::uint32_t kCqeValidByteCount = 1U << 5;

enum class EvidenceTier {
    Controlled,
};

enum class EvidenceKind {
    SqFull,
    NetworkRejected,
    NetworkDrop,
    CqOverrun,
};

struct WorkQueueConfig {
    std::uint32_t version{kWorkQueueConfigVersion};
    std::uint64_t sq_id{1};
    std::uint64_t cq_id{1};
    std::uint32_t source{0};
    std::uint32_t qpn{1};
    PolicyContextToken policy_context_token{1};
    std::size_t sq_depth{64};
    std::size_t cq_depth{64};
    // Service durations use the nonnegative signed-64-bit domain. Keeping the
    // storage unsigned preserves timestamp arithmetic, while construction
    // rejects values above INT64_MAX so a negative signed input cannot wrap
    // into a multi-day service time.
    Picoseconds doorbell_service_ps{0};
    Picoseconds wqe_fetch_service_ps{0};
    Picoseconds qpc_lookup_service_ps{0};
    Picoseconds scheduler_service_ps{0};
    Picoseconds cqe_write_service_ps{0};
};

struct WorkQueuePcieBinding {
    std::uint32_t version{kWorkQueuePcieBindingVersion};
    // Path IDs refer to the shared fabric config. Byte offsets are low address
    // bits used for DWORD, RCB and 4 KiB segmentation.
    std::uint32_t pcie_uar_path_id{1};
    std::uint32_t pcie_doorbell_record_path_id{2};
    std::uint32_t pcie_sq_memory_path_id{2};
    std::uint32_t pcie_cq_memory_path_id{2};
    // A zero domain derives a nonzero namespace from SQ/CQ identity. Explicit
    // values support intentional sharing across WorkQueue objects.
    std::uint64_t pcie_submission_ordering_domain{0};
    std::uint64_t pcie_completion_ordering_domain{0};
    std::uint64_t pcie_doorbell_record_bytes{4};
    std::uint64_t pcie_uar_doorbell_bytes{8};
    std::uint64_t pcie_wqe_bytes{64};
    std::uint64_t pcie_cqe_bytes{64};
    std::uint32_t pcie_uar_first_byte_offset{0};
    std::uint32_t pcie_doorbell_record_first_byte_offset{0};
    std::uint32_t pcie_sq_first_byte_offset{0};
    std::uint32_t pcie_cq_first_byte_offset{0};
};

struct WorkQueueHostMemoryBinding {
    std::uint32_t version{kWorkQueueHostMemoryBindingVersion};
    HostMemoryAllocationId qpc_icm_allocation_id{0};
    HostMemoryAllocationId sq_ring_allocation_id{0};
    HostMemoryAllocationId rq_ring_allocation_id{0};
    HostMemoryAllocationId cq_ring_allocation_id{0};
    HostMemoryAllocationId doorbell_record_allocation_id{0};
    std::uint64_t qpc_context_bytes{256};
};

struct WorkRequestDataMemory {
    std::uint32_t version{kWorkRequestDataMemoryVersion};
    HostMemoryAllocationId allocation_id{0};
    HostMemoryMkey mkey{0};
    std::uint64_t allocation_offset_bytes{0};
};

struct WorkRequest {
    std::uint64_t wr_id{0};
    FlowId flow_id{0};
    std::uint32_t flow_tag{0};
    std::uint32_t destination{0};
    std::uint64_t payload_bytes{0};
    std::uint8_t traffic_class{0};
    WqeOpcode opcode{WqeOpcode::Send};
    bool signaled{true};
    std::optional<WorkRequestDataMemory> data_memory;
};

struct WqeTimeline {
    Picoseconds posted_at_ps{0};
    std::optional<Picoseconds> doorbelled_at_ps;
    std::optional<Picoseconds> doorbell_seen_at_ps;
    std::optional<Picoseconds> wqe_fetch_begin_at_ps;
    std::optional<Picoseconds> wqe_fetch_end_at_ps;
    std::optional<Picoseconds> qpc_ready_at_ps;
    std::optional<Picoseconds> admitted_at_ps;
    std::optional<Picoseconds> network_accepted_at_ps;
    std::optional<Picoseconds> network_outcome_at_ps;
    // These remain unset in the flow-extent v1 port. A packetized adapter
    // must populate them from explicit TX issue events, not acceptance or
    // end-to-end delivery timestamps.
    std::optional<Picoseconds> first_packet_at_ps;
    std::optional<Picoseconds> last_packet_at_ps;
    std::optional<Picoseconds> transport_retired_at_ps;
    std::optional<Picoseconds> cqe_visible_at_ps;
    std::optional<Picoseconds> polled_at_ps;
    std::optional<Picoseconds> sq_reclaimed_at_ps;
};

struct WqeRecord {
    WqeId wqe_id{0};
    std::uint64_t sq_sequence{0};
    std::uint64_t doorbell_batch_id{0};
    WorkRequest request;
    WqeTimeline timeline;
    WqeState state{WqeState::Posted};
    std::optional<NetworkToken> network_token;
    std::optional<CompletionStatus> completion_status;
    bool ecn_marked{false};
};

struct PostResult {
    PostStatus status{PostStatus::Fatal};
    WqeId wqe_id{0};
    std::uint64_t sq_sequence{0};
};

struct PostBatchResult {
    PostStatus status{PostStatus::Accepted};
    std::vector<PostResult> accepted;
    std::optional<std::size_t> bad_wr_index;
};

struct DoorbellBatch {
    std::uint64_t batch_id{0};
    std::size_t wqe_count{0};
    Picoseconds rung_at_ps{0};
    Picoseconds observed_at_ps{0};
};

struct CompletionEntry {
    std::uint64_t cqe_sequence{0};
    std::uint64_t cq_producer_index{0};
    std::size_t cq_slot{0};
    std::uint8_t owner_generation{0};
    std::uint64_t wr_id{0};
    WqeId wqe_id{0};
    std::uint64_t sq_sequence{0};
    std::uint32_t qpn{0};
    WqeOpcode opcode{WqeOpcode::Send};
    CompletionStatus status{CompletionStatus::Success};
    std::uint32_t valid_fields{0};
    std::uint64_t byte_count{0};
    std::uint32_t vendor_syndrome{0};
    Picoseconds visible_at_ps{0};
    Picoseconds polled_at_ps{0};
};

struct EvidenceEvent {
    EvidenceTier tier{EvidenceTier::Controlled};
    EvidenceKind kind{EvidenceKind::SqFull};
    WqeId wqe_id{0};
    std::uint64_t wr_id{0};
    Picoseconds observed_at_ps{0};
    DropLocation drop_location{DropLocation::None};
    DropReason drop_reason{DropReason::None};
};

struct WorkQueueCounters {
    std::uint64_t posted_wqes{0};
    std::uint64_t sq_full_rejections{0};
    std::uint64_t doorbells{0};
    std::uint64_t doorbelled_wqes{0};
    std::uint64_t network_submit_attempts{0};
    std::uint64_t network_accepted{0};
    std::uint64_t network_busy{0};
    std::uint64_t network_rejected{0};
    std::uint64_t network_delivered{0};
    std::uint64_t network_dropped{0};
    std::uint64_t cqes_visible{0};
    std::uint64_t cqes_polled{0};
    std::uint64_t cq_overruns{0};
    std::uint64_t sq_reclaimed_wqes{0};
    std::size_t sq_high_watermark{0};
    std::size_t cq_high_watermark{0};
};

class WorkQueue {
public:
    WorkQueue(WorkQueueConfig config, NetworkPort& network_port);
    WorkQueue(
        WorkQueueConfig config,
        NetworkPort& network_port,
        PcieFabric& pcie_fabric,
        WorkQueuePcieBinding pcie_binding = {});
    ~WorkQueue();

    WorkQueue(const WorkQueue&) = delete;
    WorkQueue& operator=(const WorkQueue&) = delete;
    WorkQueue(WorkQueue&&) noexcept;
    WorkQueue& operator=(WorkQueue&&) noexcept;

    PostResult postSend(const WorkRequest& request, Picoseconds now_ps);
    PostBatchResult postSendBatch(
        const std::vector<WorkRequest>& requests,
        Picoseconds now_ps);
    DoorbellBatch ringDoorbell(Picoseconds now_ps);
    // Deliver same-timestamp NetworkEvents before progress() so a completed
    // network credit is visible when its advertised retry deadline arrives.
    std::size_t progress(Picoseconds now_ps);
    void onNetworkEvent(const NetworkEvent& event);

    // progress() and pollCompletionQueue() are separate event actions. A poll
    // first publishes any CQE strictly older than t. For CQEs due exactly at
    // t, calling progress first gives device-publication priority while
    // polling first gives host-consumer priority.
    std::vector<CompletionEntry> pollCompletionQueue(
        std::size_t max_entries,
        Picoseconds now_ps);

    // Fatal queues remain physically non-quiescent but have no schedulable
    // internal event. Event loops must check fatal() when this returns empty.
    std::optional<Picoseconds> nextEventTime() const;
    bool hasPendingPhysicalWork() const noexcept;
    bool fatal() const noexcept;
    std::size_t occupiedSqEntries() const noexcept;
    std::size_t completionQueueDepth() const noexcept;
    std::size_t unpublishedWqeCount() const noexcept;

    const WorkQueueConfig& config() const noexcept;
    std::optional<WorkQueuePcieBinding> pcieBinding() const;
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
    WorkQueue(
        WorkQueueConfig config,
        NetworkPort& network_port,
        PcieFabric* pcie_fabric,
        std::optional<WorkQueuePcieBinding> pcie_binding,
        bool qpc_lookup_enabled,
        VirtualHostMemory* host_memory,
        std::optional<WorkQueueHostMemoryBinding> host_memory_binding,
        HostMemoryDeviceOwnerId host_memory_device_owner_id,
        std::optional<RnicSubmissionProfile> submission_profile);

    class Impl;
    std::unique_ptr<Impl> impl_;
    friend class RnicDevice;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_WORK_QUEUE_H
