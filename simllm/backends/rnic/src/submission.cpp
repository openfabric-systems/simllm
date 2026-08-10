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
