#ifndef SIMLLM_RNIC_SUBMISSION_H
#define SIMLLM_RNIC_SUBMISSION_H

#include <cstdint>
#include <optional>
#include <string>

#include "simllm/rnic/pcie_fabric.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicSubmissionConfigVersion = 1;
inline constexpr std::uint32_t kRnicSubmissionProfileVersion = 1;
inline constexpr std::uint32_t kRnicSubmissionAgentVersion = 1;
inline constexpr std::uint32_t kRnicProducerTaskLinkVersion = 1;
inline constexpr std::uint32_t kRnicSubmissionRecordVersion = 2;
inline constexpr std::uint32_t kRnicCqConsumptionRecordVersion = 1;

enum class RnicProducerShape : std::uint8_t {
    HostCpuDriver,
    CpuProxy,
    GpuInitiated,
};

enum class RnicSubmissionAgentKind : std::uint8_t {
    None,
    HostCpuDriver,
    CpuProxy,
    Gpu,
};

enum class RnicUarMappingOwner : std::uint8_t {
    HostCpu,
    Gpu,
};

struct RnicSubmissionAgent {
    std::uint32_t version{kRnicSubmissionAgentVersion};
    RnicSubmissionAgentKind kind{RnicSubmissionAgentKind::None};
    std::uint32_t id{0};
};

struct RnicSubmissionConfig {
    std::uint32_t version{kRnicSubmissionConfigVersion};
    RnicProducerShape producer_shape{RnicProducerShape::HostCpuDriver};
    // Zero identities are compatibility defaults only for HostCpuDriver.
    // They resolve to the QP number so existing PCIe requester bytes remain
    // unchanged. Nondefault shapes require explicit identities.
    std::uint32_t producer_id{0};
    std::uint32_t descriptor_writer_id{0};
    std::uint64_t descriptor_queue_allocation_id{0};
    std::uint32_t cq_consumer_id{0};
    std::uint32_t rnic_requester_id{0};
};

struct RnicSubmissionProfile {
    std::uint32_t version{kRnicSubmissionProfileVersion};
    RnicProducerShape producer_shape{RnicProducerShape::HostCpuDriver};
    RnicSubmissionAgent producer;
    std::optional<RnicSubmissionAgent> descriptor_writer;
    std::uint64_t descriptor_queue_allocation_id{0};
    std::optional<PcieEndpointKind> descriptor_queue_endpoint;
    PcieEndpointKind queue_endpoint{PcieEndpointKind::HostPinnedMemory};
    RnicUarMappingOwner uar_mapping_owner{RnicUarMappingOwner::HostCpu};
    RnicSubmissionAgent cq_consumer;
    std::uint32_t rnic_requester_id{0};
};

struct RnicProducerTaskLink {
    std::uint32_t version{kRnicProducerTaskLinkVersion};
    std::string task_id;
    RnicProducerShape producer_shape{RnicProducerShape::HostCpuDriver};
    RnicSubmissionAgent task_owner;
    Picoseconds submitted_at_ps{0};
    Picoseconds eligible_at_ps{0};
    Picoseconds started_at_ps{0};
    Picoseconds finished_at_ps{0};
    Picoseconds completed_at_ps{0};
};

struct RnicSubmissionRecord {
    std::uint32_t version{kRnicSubmissionRecordVersion};
    std::uint64_t sequence{0};
    WqeId wqe_id{0};
    std::uint64_t sq_id{0};
    std::uint32_t qpn{0};
    std::uint64_t doorbell_batch_id{0};
    RnicSubmissionAgent producer;
    std::optional<RnicSubmissionAgent> descriptor_writer;
    std::uint64_t descriptor_queue_allocation_id{0};
    std::optional<PcieEndpointKind> descriptor_queue_endpoint;
    PcieEndpointKind sq_endpoint{PcieEndpointKind::HostPinnedMemory};
    PcieEndpointKind doorbell_endpoint{PcieEndpointKind::HostPinnedMemory};
    RnicUarMappingOwner uar_mapping_owner{RnicUarMappingOwner::HostCpu};
    std::optional<RnicProducerTaskLink> producer_task;
    Picoseconds posted_at_ps{0};
    Picoseconds submitted_at_ps{0};
    Picoseconds visible_to_rnic_at_ps{0};
};

struct RnicCqConsumptionRecord {
    std::uint32_t version{kRnicCqConsumptionRecordVersion};
    std::uint64_t sequence{0};
    std::uint64_t cqe_sequence{0};
    WqeId wqe_id{0};
    std::uint64_t cq_id{0};
    std::uint32_t qpn{0};
    RnicSubmissionAgent consumer;
    PcieEndpointKind cq_endpoint{PcieEndpointKind::HostPinnedMemory};
    Picoseconds visible_at_ps{0};
    Picoseconds consumed_at_ps{0};
};

bool isDefaultRnicSubmissionConfig(
    const RnicSubmissionConfig& config) noexcept;
RnicSubmissionProfile resolveRnicSubmissionProfile(
    const RnicSubmissionConfig& config,
    std::uint32_t qpn);
void validateRnicProducerTaskLink(
    const RnicSubmissionProfile& profile,
    const RnicProducerTaskLink& link,
    Picoseconds record_submitted_at_ps);

const char* toString(RnicProducerShape shape) noexcept;
const char* toString(RnicSubmissionAgentKind kind) noexcept;
const char* toString(RnicUarMappingOwner owner) noexcept;

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_SUBMISSION_H
