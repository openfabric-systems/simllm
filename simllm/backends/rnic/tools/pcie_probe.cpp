#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "simllm/rnic/pcie_fabric.h"

namespace {

using simllm::rnic::PcieDirection;
using simllm::rnic::PcieAnalyticalDelayKind;
using simllm::rnic::PcieAnalyticalDelayProfile;
using simllm::rnic::PcieEndpointKind;
using simllm::rnic::PcieFabric;
using simllm::rnic::PcieGeneration;
using simllm::rnic::PcieOperation;
using simllm::rnic::PcieOrdering;
using simllm::rnic::PcieServiceClass;
using simllm::rnic::PcieTransactionRequest;
using simllm::rnic::Picoseconds;
using simllm::rnic::defaultPcieFabricConfig;
using simllm::rnic::toString;

struct Options {
    PcieOperation operation{PcieOperation::PostedWrite};
    PcieDirection direction{PcieDirection::DeviceToHost};
    std::uint64_t transactions{1};
    std::uint64_t bytes{4096};
    std::uint64_t useful_bytes{0};
    std::uint64_t offset{0};
    std::uint64_t generation{5};
    std::uint64_t lanes{16};
    std::uint64_t mps{256};
    std::uint64_t mrrs{512};
    std::uint64_t rcb{64};
    std::uint64_t outstanding_reads{64};
    std::uint64_t completion_buffer_bytes{64 * 1024};
    std::uint64_t credit_return_ps{0};
    std::uint64_t host_store_latency_ps{0};
    std::uint64_t posted_visibility_ps{0};
    std::uint64_t read_completion_ps{0};
    std::uint64_t base_path_ps{0};
    std::uint64_t numa_mean_ps{0};
    std::uint64_t iommu_ps{0};
    std::uint64_t acs_ps{0};
    std::uint64_t switch_ps{0};
    std::uint64_t ddio_ps{0};
    std::uint64_t gpu_direct_ps{0};
    std::string numa_profile{"auto"};
    bool numa_profile_set{false};
    std::optional<std::uint64_t> numa_incidence_ppm;
    std::uint64_t numa_standard_deviation_ps{0};
    std::uint64_t numa_tail_probability_ppm{0};
    std::uint64_t numa_tail_mean_ps{0};
    std::uint64_t numa_tail_standard_deviation_ps{0};
    std::uint64_t analytical_seed{0};
    std::uint64_t visibility_dependency{0};
};

std::uint64_t parseUnsigned(const std::string& text, const char* option) {
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value, 10);
    if (parsed.ec == std::errc::result_out_of_range) {
        throw std::out_of_range(
            std::string(option) + " is outside the uint64 range");
    }
    if (text.empty() || parsed.ec != std::errc{}
        || parsed.ptr != text.data() + text.size()) {
        throw std::invalid_argument(
            std::string(option) + " must be an unsigned integer");
    }
    return value;
}

PcieOperation parseOperation(const std::string& text) {
    if (text == "host_store") {
        return PcieOperation::HostStore;
    }
    if (text == "posted_write") {
        return PcieOperation::PostedWrite;
    }
    if (text == "nonposted_read") {
        return PcieOperation::NonPostedRead;
    }
    throw std::invalid_argument("unknown RNIC PCIe operation: " + text);
}

PcieDirection parseDirection(const std::string& text) {
    if (text == "host_to_device") {
        return PcieDirection::HostToDevice;
    }
    if (text == "device_to_host") {
        return PcieDirection::DeviceToHost;
    }
    throw std::invalid_argument("unknown RNIC PCIe direction: " + text);
}

PcieAnalyticalDelayKind parseAnalyticalKind(const std::string& text) {
    if (text == "disabled") {
        return PcieAnalyticalDelayKind::Disabled;
    }
    if (text == "fixed") {
        return PcieAnalyticalDelayKind::Fixed;
    }
    if (text == "gaussian") {
        return PcieAnalyticalDelayKind::Gaussian;
    }
    if (text == "gaussian_tail_mixture") {
        return PcieAnalyticalDelayKind::GaussianTailMixture;
    }
    throw std::invalid_argument(
        "unknown RNIC PCIe analytical profile: " + text);
}

const char* endpointName(PcieEndpointKind endpoint) {
    switch (endpoint) {
    case PcieEndpointKind::MmioBar:
        return "mmio_bar";
    case PcieEndpointKind::HostPinnedMemory:
        return "host_pinned_memory";
    case PcieEndpointKind::GpuMemory:
        return "gpu_memory";
    case PcieEndpointKind::DeviceMemory:
        return "device_memory";
    default:
        return "invalid";
    }
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument(
                "every RNIC PCIe probe option needs a value");
        }
        const std::string option = argv[index];
        const std::string text = argv[index + 1];
        if (option == "--operation") {
            options.operation = parseOperation(text);
            continue;
        }
        if (option == "--direction") {
            options.direction = parseDirection(text);
            continue;
        }
        if (option == "--numa-profile") {
            (void)parseAnalyticalKind(text);
            options.numa_profile = text;
            options.numa_profile_set = true;
            continue;
        }
        const std::uint64_t value = parseUnsigned(text, argv[index]);
        if (option == "--transactions") {
            options.transactions = value;
        } else if (option == "--bytes") {
            options.bytes = value;
        } else if (option == "--useful-bytes") {
            options.useful_bytes = value;
        } else if (option == "--offset") {
            options.offset = value;
        } else if (option == "--generation") {
            options.generation = value;
        } else if (option == "--lanes") {
            options.lanes = value;
        } else if (option == "--mps") {
            options.mps = value;
        } else if (option == "--mrrs") {
            options.mrrs = value;
        } else if (option == "--rcb") {
            options.rcb = value;
        } else if (option == "--outstanding-reads") {
            options.outstanding_reads = value;
        } else if (option == "--completion-buffer-bytes") {
            options.completion_buffer_bytes = value;
        } else if (option == "--credit-return-ps") {
            options.credit_return_ps = value;
        } else if (option == "--host-store-latency-ps") {
            options.host_store_latency_ps = value;
        } else if (option == "--posted-visibility-ps") {
            options.posted_visibility_ps = value;
        } else if (option == "--read-completion-ps") {
            options.read_completion_ps = value;
        } else if (option == "--base-path-ps") {
            options.base_path_ps = value;
        } else if (option == "--numa-ps"
                   || option == "--numa-mean-ps") {
            options.numa_mean_ps = value;
        } else if (option == "--iommu-ps") {
            options.iommu_ps = value;
        } else if (option == "--acs-ps") {
            options.acs_ps = value;
        } else if (option == "--switch-ps") {
            options.switch_ps = value;
        } else if (option == "--ddio-ps") {
            options.ddio_ps = value;
        } else if (option == "--gpu-direct-ps") {
            options.gpu_direct_ps = value;
        } else if (option == "--numa-incidence-ppm") {
            options.numa_incidence_ppm = value;
        } else if (option == "--numa-standard-deviation-ps") {
            options.numa_standard_deviation_ps = value;
        } else if (option == "--numa-tail-probability-ppm") {
            options.numa_tail_probability_ppm = value;
        } else if (option == "--numa-tail-mean-ps") {
            options.numa_tail_mean_ps = value;
        } else if (option == "--numa-tail-standard-deviation-ps") {
            options.numa_tail_standard_deviation_ps = value;
        } else if (option == "--analytical-seed") {
            options.analytical_seed = value;
        } else if (option == "--visibility-dependency") {
            options.visibility_dependency = value;
        } else {
            throw std::invalid_argument(
                "unknown RNIC PCIe probe option: " + option);
        }
    }
    if (options.transactions == 0 || options.bytes == 0) {
        throw std::invalid_argument(
            "RNIC PCIe probe transactions and bytes must be positive");
    }
    if (options.useful_bytes == 0) {
        options.useful_bytes = options.bytes;
    }
    if (options.useful_bytes > options.bytes) {
        throw std::invalid_argument(
            "RNIC PCIe probe useful bytes exceed transfer bytes");
    }
    if (options.offset >= 4096 || options.generation < 1
        || options.generation > 5
        || options.lanes > std::numeric_limits<std::uint16_t>::max()
        || options.mps > std::numeric_limits<std::uint32_t>::max()
        || options.mrrs > std::numeric_limits<std::uint32_t>::max()
        || options.rcb > std::numeric_limits<std::uint32_t>::max()
        || options.outstanding_reads
            > std::numeric_limits<std::uint32_t>::max()
        || (options.numa_incidence_ppm.has_value()
            && *options.numa_incidence_ppm
                > simllm::rnic::kPcieProbabilityScalePpm)
        || options.numa_tail_probability_ppm
            > simllm::rnic::kPcieProbabilityScalePpm
        || options.visibility_dependency > 1) {
        throw std::out_of_range(
            "RNIC PCIe probe option exceeds its modeled range");
    }
    return options;
}

PcieAnalyticalDelayProfile fixedPenalty(std::uint64_t delay_ps) {
    PcieAnalyticalDelayProfile profile;
    if (delay_ps == 0) {
        return profile;
    }
    profile.kind = PcieAnalyticalDelayKind::Fixed;
    profile.incidence_probability_ppm =
        simllm::rnic::kPcieProbabilityScalePpm;
    profile.mean_ps = delay_ps;
    return profile;
}

PcieGeneration generation(std::uint64_t value) {
    return static_cast<PcieGeneration>(value);
}

int run(const Options& options) {
    auto config = defaultPcieFabricConfig();
    config.generation = generation(options.generation);
    config.lane_count = static_cast<std::uint16_t>(options.lanes);
    config.max_payload_size_bytes = static_cast<std::uint32_t>(options.mps);
    config.max_read_request_size_bytes =
        static_cast<std::uint32_t>(options.mrrs);
    config.read_completion_boundary_bytes =
        static_cast<std::uint32_t>(options.rcb);
    config.max_outstanding_read_requests =
        static_cast<std::uint32_t>(options.outstanding_reads);
    config.completion_buffer_bytes = options.completion_buffer_bytes;
    config.credit_return_latency_ps = options.credit_return_ps;
    config.analytical_seed = options.analytical_seed;
    config.host_store_latency_ps.samples_ps = {
        options.host_store_latency_ps};
    config.posted_write_visibility_latency_ps.samples_ps = {
        options.posted_visibility_ps};
    config.read_completion_latency_ps.samples_ps = {
        options.read_completion_ps};
    config.paths[1].base_latency_ps = options.base_path_ps;
    PcieAnalyticalDelayProfile numa = fixedPenalty(options.numa_mean_ps);
    if (options.numa_profile_set) {
        numa.kind = parseAnalyticalKind(options.numa_profile);
        numa.incidence_probability_ppm = static_cast<std::uint32_t>(
            options.numa_incidence_ppm.value_or(
                numa.kind == PcieAnalyticalDelayKind::Disabled
                    ? 0
                    : simllm::rnic::kPcieProbabilityScalePpm));
        numa.mean_ps = options.numa_mean_ps;
        numa.standard_deviation_ps = options.numa_standard_deviation_ps;
        numa.tail_probability_ppm = static_cast<std::uint32_t>(
            options.numa_tail_probability_ppm);
        numa.tail_mean_ps = options.numa_tail_mean_ps;
        numa.tail_standard_deviation_ps =
            options.numa_tail_standard_deviation_ps;
    } else if (options.numa_incidence_ppm.has_value()
               || options.numa_standard_deviation_ps != 0
               || options.numa_tail_probability_ppm != 0
               || options.numa_tail_mean_ps != 0
               || options.numa_tail_standard_deviation_ps != 0) {
        throw std::invalid_argument(
            "NUMA analytical fields require --numa-profile");
    }
    auto& penalties = config.paths[1].analytical_penalties;
    penalties.numa = numa;
    penalties.iommu = fixedPenalty(options.iommu_ps);
    penalties.acs = fixedPenalty(options.acs_ps);
    penalties.switch_path = fixedPenalty(options.switch_ps);
    penalties.ddio_miss = fixedPenalty(options.ddio_ps);
    penalties.gpu_direct = fixedPenalty(options.gpu_direct_ps);
    PcieFabric fabric(config);

    const PcieServiceClass service_class =
        options.operation == PcieOperation::NonPostedRead
        ? PcieServiceClass::PayloadRead
        : (options.operation == PcieOperation::PostedWrite
               ? PcieServiceClass::PayloadWrite
               : PcieServiceClass::DoorbellRecord);

    Picoseconds first_issue_at_ps = 0;
    Picoseconds jct_ps = 0;
    Picoseconds minimum_completion_ps =
        std::numeric_limits<Picoseconds>::max();
    std::uint64_t request_tlps = 0;
    std::uint64_t completion_tlps = 0;
    for (std::uint64_t index = 0; index < options.transactions; ++index) {
        PcieTransactionRequest request;
        request.client_token = index + 1;
        request.service_class = service_class;
        request.operation = options.operation;
        request.request_direction = options.direction;
        request.ordering = options.visibility_dependency != 0
            ? PcieOrdering::VisibilityDependency
            : PcieOrdering::Independent;
        request.path_id = 2;
        request.ordering_domain = 1;
        request.useful_bytes = options.useful_bytes;
        request.transfer_bytes = options.bytes;
        request.first_byte_offset = static_cast<std::uint32_t>(options.offset);
        request.submitted_at_ps = 0;
        const auto result = fabric.submit(request);
        if (index == 0) {
            first_issue_at_ps = result.first_issue_at_ps;
        }
        request_tlps += result.request_tlps;
        completion_tlps += result.completion_tlps;
        jct_ps = std::max(jct_ps, result.completed_at_ps);
        minimum_completion_ps = std::min(
            minimum_completion_ps, result.completed_at_ps);
    }
    fabric.validateInvariants();
    const auto accounting = fabric.accounting(service_class);
    const auto& selected_path = config.paths[1];

    std::cout
        << "operation,direction,service_class,config_version,"
           "transaction_abi_version,result_version,requested_transactions,"
           "accounted_transactions,bytes,useful_bytes,offset,submitted_at_ps,"
           "path_id,path_endpoint,ordering_domain,visibility_dependency,"
           "generation,lanes,mps,mrrs,rcb,posted_write_overhead_bytes,"
           "read_request_overhead_bytes,completion_overhead_bytes,"
           "data_credit_unit_bytes,max_tlps_per_transaction,"
           "h2d_posted_header_credits,h2d_posted_data_credits,"
           "h2d_nonposted_header_credits,h2d_nonposted_data_credits,"
           "h2d_completion_header_credits,h2d_completion_data_credits,"
           "d2h_posted_header_credits,d2h_posted_data_credits,"
           "d2h_nonposted_header_credits,d2h_nonposted_data_credits,"
           "d2h_completion_header_credits,d2h_completion_data_credits,"
           "outstanding_reads,"
           "completion_buffer_bytes,credit_return_ps,"
           "completion_buffer_release_ps,"
           "host_store_latency_ps,posted_visibility_ps,"
           "read_completion_ps,base_path_ps,numa_mean_ps,iommu_ps,acs_ps,"
           "switch_ps,ddio_ps,gpu_direct_ps,analytical_seed,"
           "analytical_profile_version,numa_profile,"
           "numa_incidence_ppm,numa_standard_deviation_ps,"
           "numa_tail_probability_ppm,numa_tail_mean_ps,"
           "numa_tail_standard_deviation_ps,"
           "request_tlps,completion_tlps,accounted_useful_bytes,"
           "accounted_transferred_bytes,host_store_bytes,"
           "h2d_tlps,h2d_payload_bytes,h2d_overhead_bytes,h2d_link_bytes,"
           "d2h_tlps,d2h_payload_bytes,d2h_overhead_bytes,d2h_link_bytes,"
           "ordering_wait_ps,outstanding_wait_ps,completion_buffer_wait_ps,"
           "credit_wait_ps,link_queue_wait_ps,base_path_attributed_ps,"
           "numa_attributed_ps,iommu_attributed_ps,acs_attributed_ps,"
           "switch_attributed_ps,ddio_attributed_ps,"
           "gpu_direct_attributed_ps,numa_profile_draws,"
           "numa_profile_occurrences,numa_profile_tail_draws,"
           "iommu_profile_draws,iommu_profile_occurrences,"
           "iommu_profile_tail_draws,acs_profile_draws,"
           "acs_profile_occurrences,acs_profile_tail_draws,"
           "switch_profile_draws,switch_profile_occurrences,"
           "switch_profile_tail_draws,ddio_profile_draws,"
           "ddio_profile_occurrences,ddio_profile_tail_draws,"
           "gpu_direct_profile_draws,gpu_direct_profile_occurrences,"
           "gpu_direct_profile_tail_draws,host_store_service_ps,"
           "posted_visibility_service_ps,read_completion_service_ps,"
           "latency_samples,first_issue_ps,minimum_completion_ps,jct_ps\n"
        << toString(options.operation) << ','
        << toString(options.direction) << ','
        << toString(service_class) << ','
        << config.version << ','
        << simllm::rnic::kPcieTransactionAbiVersion << ','
        << simllm::rnic::kPcieTransactionResultVersion << ','
        << options.transactions << ','
        << accounting.transactions << ','
        << options.bytes << ','
        << options.useful_bytes << ','
        << options.offset << ','
        << 0 << ','
        << selected_path.path_id << ','
        << endpointName(selected_path.endpoint) << ','
        << 1 << ','
        << options.visibility_dependency << ','
        << options.generation << ','
        << options.lanes << ','
        << options.mps << ','
        << options.mrrs << ','
        << options.rcb << ','
        << config.posted_write_overhead_bytes << ','
        << config.read_request_overhead_bytes << ','
        << config.completion_overhead_bytes << ','
        << config.data_credit_unit_bytes << ','
        << config.max_tlps_per_transaction << ','
        << config.host_to_device_credits.posted_header_credits << ','
        << config.host_to_device_credits.posted_data_credits << ','
        << config.host_to_device_credits.nonposted_header_credits << ','
        << config.host_to_device_credits.nonposted_data_credits << ','
        << config.host_to_device_credits.completion_header_credits << ','
        << config.host_to_device_credits.completion_data_credits << ','
        << config.device_to_host_credits.posted_header_credits << ','
        << config.device_to_host_credits.posted_data_credits << ','
        << config.device_to_host_credits.nonposted_header_credits << ','
        << config.device_to_host_credits.nonposted_data_credits << ','
        << config.device_to_host_credits.completion_header_credits << ','
        << config.device_to_host_credits.completion_data_credits << ','
        << options.outstanding_reads << ','
        << options.completion_buffer_bytes << ','
        << options.credit_return_ps << ','
        << config.completion_buffer_release_latency_ps << ','
        << options.host_store_latency_ps << ','
        << options.posted_visibility_ps << ','
        << options.read_completion_ps << ','
        << options.base_path_ps << ','
        << options.numa_mean_ps << ','
        << options.iommu_ps << ','
        << options.acs_ps << ','
        << options.switch_ps << ','
        << options.ddio_ps << ','
        << options.gpu_direct_ps << ','
        << options.analytical_seed << ','
        << simllm::rnic::kPcieAnalyticalDelayProfileVersion << ','
        << toString(numa.kind) << ','
        << numa.incidence_probability_ppm << ','
        << numa.standard_deviation_ps << ','
        << numa.tail_probability_ppm << ','
        << numa.tail_mean_ps << ','
        << numa.tail_standard_deviation_ps << ','
        << request_tlps << ','
        << completion_tlps << ','
        << accounting.useful_bytes << ','
        << accounting.transferred_bytes << ','
        << accounting.host_store_bytes << ','
        << accounting.host_to_device.tlps << ','
        << accounting.host_to_device.payload_bytes << ','
        << accounting.host_to_device.overhead_bytes << ','
        << accounting.host_to_device.modeled_link_bytes << ','
        << accounting.device_to_host.tlps << ','
        << accounting.device_to_host.payload_bytes << ','
        << accounting.device_to_host.overhead_bytes << ','
        << accounting.device_to_host.modeled_link_bytes << ','
        << accounting.waits.ordering_ps << ','
        << accounting.waits.outstanding_read_ps << ','
        << accounting.waits.completion_buffer_ps << ','
        << accounting.waits.credit_ps << ','
        << accounting.waits.link_queue_ps << ','
        << accounting.path_delay.base_ps << ','
        << accounting.path_delay.numa_ps << ','
        << accounting.path_delay.iommu_ps << ','
        << accounting.path_delay.acs_ps << ','
        << accounting.path_delay.switch_ps << ','
        << accounting.path_delay.ddio_miss_ps << ','
        << accounting.path_delay.gpu_direct_ps << ','
        << accounting.path_profiles.numa.draws << ','
        << accounting.path_profiles.numa.occurrences << ','
        << accounting.path_profiles.numa.tail_draws << ','
        << accounting.path_profiles.iommu.draws << ','
        << accounting.path_profiles.iommu.occurrences << ','
        << accounting.path_profiles.iommu.tail_draws << ','
        << accounting.path_profiles.acs.draws << ','
        << accounting.path_profiles.acs.occurrences << ','
        << accounting.path_profiles.acs.tail_draws << ','
        << accounting.path_profiles.switch_path.draws << ','
        << accounting.path_profiles.switch_path.occurrences << ','
        << accounting.path_profiles.switch_path.tail_draws << ','
        << accounting.path_profiles.ddio_miss.draws << ','
        << accounting.path_profiles.ddio_miss.occurrences << ','
        << accounting.path_profiles.ddio_miss.tail_draws << ','
        << accounting.path_profiles.gpu_direct.draws << ','
        << accounting.path_profiles.gpu_direct.occurrences << ','
        << accounting.path_profiles.gpu_direct.tail_draws << ','
        << accounting.service_delay.host_store_ps << ','
        << accounting.service_delay.posted_write_visibility_ps << ','
        << accounting.service_delay.read_completion_ps << ','
        << accounting.latency_samples_used << ','
        << first_issue_at_ps << ','
        << minimum_completion_ps << ','
        << jct_ps << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "simllm_rnic_pcie_probe: " << error.what() << '\n';
        return 2;
    }
}
