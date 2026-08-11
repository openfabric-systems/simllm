#include "simllm/rnic/submission.h"

#include <stdexcept>

namespace simllm::rnic {
namespace {

RnicSubmissionAgent agent(
    RnicSubmissionAgentKind kind,
    std::uint32_t id) {
    if (kind == RnicSubmissionAgentKind::None || id == 0) {
        throw std::invalid_argument(
            "RNIC submission agent kind and identity must be present");
    }
    return RnicSubmissionAgent{kRnicSubmissionAgentVersion, kind, id};
}

std::uint32_t compatibilityIdentity(
    std::uint32_t configured,
    std::uint32_t qpn) noexcept {
    return configured == 0 ? qpn : configured;
}

bool sameAgent(
    const RnicSubmissionAgent& lhs,
    const RnicSubmissionAgent& rhs) noexcept {
    return lhs.version == rhs.version && lhs.kind == rhs.kind
        && lhs.id == rhs.id;
}

}  // namespace

bool isDefaultRnicSubmissionConfig(
    const RnicSubmissionConfig& config) noexcept {
    return config.version == kRnicSubmissionConfigVersion
        && config.producer_shape == RnicProducerShape::HostCpuDriver
        && config.producer_id == 0
        && config.descriptor_writer_id == 0
        && config.descriptor_queue_allocation_id == 0
        && config.cq_consumer_id == 0
        && config.rnic_requester_id == 0;
}

RnicSubmissionProfile resolveRnicSubmissionProfile(
    const RnicSubmissionConfig& config,
    std::uint32_t qpn) {
    if (config.version != kRnicSubmissionConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC submission config version");
    }
    if (qpn == 0) {
        throw std::invalid_argument(
            "RNIC submission profile requires a nonzero QP number");
    }

    RnicSubmissionProfile profile;
    profile.producer_shape = config.producer_shape;
    switch (config.producer_shape) {
    case RnicProducerShape::HostCpuDriver:
        if (config.descriptor_writer_id != 0
            || config.descriptor_queue_allocation_id != 0) {
            throw std::invalid_argument(
                "RNIC host CPU producer cannot name a descriptor queue");
        }
        profile.producer = agent(
            RnicSubmissionAgentKind::HostCpuDriver,
            compatibilityIdentity(config.producer_id, qpn));
        profile.cq_consumer = agent(
            RnicSubmissionAgentKind::HostCpuDriver,
            compatibilityIdentity(config.cq_consumer_id, qpn));
        profile.rnic_requester_id = compatibilityIdentity(
            config.rnic_requester_id, qpn);
        break;
    case RnicProducerShape::CpuProxy:
        if (config.producer_id == 0 || config.descriptor_writer_id == 0
            || config.descriptor_queue_allocation_id == 0
            || config.cq_consumer_id == 0
            || config.rnic_requester_id == 0) {
            throw std::invalid_argument(
                "RNIC CPU proxy requires explicit producer, descriptor, "
                "consumer and requester identities");
        }
        profile.producer = agent(
            RnicSubmissionAgentKind::CpuProxy, config.producer_id);
        profile.descriptor_writer = agent(
            RnicSubmissionAgentKind::Gpu, config.descriptor_writer_id);
        profile.descriptor_queue_allocation_id =
            config.descriptor_queue_allocation_id;
        profile.descriptor_queue_endpoint =
            PcieEndpointKind::HostPinnedMemory;
        profile.cq_consumer = agent(
            RnicSubmissionAgentKind::CpuProxy, config.cq_consumer_id);
        profile.rnic_requester_id = config.rnic_requester_id;
        break;
    case RnicProducerShape::GpuInitiated:
        if (config.producer_id == 0 || config.cq_consumer_id == 0
            || config.rnic_requester_id == 0) {
            throw std::invalid_argument(
                "RNIC GPU producer requires explicit producer, consumer "
                "and requester identities");
        }
        if (config.descriptor_writer_id != 0
            || config.descriptor_queue_allocation_id != 0) {
            throw std::invalid_argument(
                "RNIC GPU producer writes WQEs directly and cannot name a "
                "proxy descriptor queue");
        }
        profile.producer = agent(
            RnicSubmissionAgentKind::Gpu, config.producer_id);
        profile.queue_endpoint = PcieEndpointKind::GpuMemory;
        profile.uar_mapping_owner = RnicUarMappingOwner::Gpu;
        profile.cq_consumer = agent(
            RnicSubmissionAgentKind::Gpu, config.cq_consumer_id);
        profile.rnic_requester_id = config.rnic_requester_id;
        break;
    default:
        throw std::invalid_argument("invalid RNIC producer shape");
    }
    return profile;
}

void validateRnicProducerTaskLink(
    const RnicSubmissionProfile& profile,
    const RnicProducerTaskLink& link,
    Picoseconds record_submitted_at_ps) {
    if (link.version != kRnicProducerTaskLinkVersion) {
        throw std::invalid_argument(
            "unsupported RNIC producer task-link version");
    }
    if (link.task_id.empty()) {
        throw std::invalid_argument(
            "RNIC producer task link requires a task identity");
    }
    if (link.producer_shape != profile.producer_shape) {
        throw std::invalid_argument(
            "RNIC producer task link disagrees with the producer shape");
    }

    const RnicSubmissionAgent* expected_owner = nullptr;
    switch (profile.producer_shape) {
    case RnicProducerShape::HostCpuDriver:
        throw std::invalid_argument(
            "RNIC host CPU submission cannot carry a GPU task link");
    case RnicProducerShape::CpuProxy:
        if (!profile.descriptor_writer.has_value()) {
            throw std::invalid_argument(
                "RNIC CPU proxy has no GPU descriptor writer");
        }
        expected_owner = &*profile.descriptor_writer;
        break;
    case RnicProducerShape::GpuInitiated:
        expected_owner = &profile.producer;
        break;
    default:
        throw std::invalid_argument("invalid RNIC producer task-link shape");
    }
    if (!sameAgent(link.task_owner, *expected_owner)
        || link.task_owner.kind != RnicSubmissionAgentKind::Gpu) {
        throw std::invalid_argument(
            "RNIC producer task link names the wrong GPU owner");
    }
    if (link.eligible_at_ps < link.submitted_at_ps
        || link.started_at_ps < link.eligible_at_ps
        || link.finished_at_ps < link.started_at_ps
        || link.completed_at_ps < link.finished_at_ps
        || record_submitted_at_ps < link.completed_at_ps) {
        throw std::invalid_argument(
            "RNIC producer task-link timestamps are not monotonic");
    }
}

const char* toString(RnicProducerShape shape) noexcept {
    switch (shape) {
    case RnicProducerShape::HostCpuDriver:
        return "host_cpu_driver";
    case RnicProducerShape::CpuProxy:
        return "cpu_proxy";
    case RnicProducerShape::GpuInitiated:
        return "gpu_initiated";
    default:
        return "invalid";
    }
}

const char* toString(RnicSubmissionAgentKind kind) noexcept {
    switch (kind) {
    case RnicSubmissionAgentKind::None:
        return "none";
    case RnicSubmissionAgentKind::HostCpuDriver:
        return "host_cpu_driver";
    case RnicSubmissionAgentKind::CpuProxy:
        return "cpu_proxy";
    case RnicSubmissionAgentKind::Gpu:
        return "gpu";
    default:
        return "invalid";
    }
}

const char* toString(RnicUarMappingOwner owner) noexcept {
    switch (owner) {
    case RnicUarMappingOwner::HostCpu:
        return "host_cpu";
    case RnicUarMappingOwner::Gpu:
        return "gpu";
    default:
        return "invalid";
    }
}

}  // namespace simllm::rnic
