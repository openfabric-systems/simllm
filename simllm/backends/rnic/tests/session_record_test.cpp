#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fake_network.h"
#include "simllm/rnic/session_record.h"

namespace {

using simllm::rnic::CompletionEntry;
using simllm::rnic::CompletionStatus;
using simllm::rnic::PcieAnalyticalDelayKind;
using simllm::rnic::PcieAnalyticalDelayProfile;
using simllm::rnic::PcieEndpointKind;
using simllm::rnic::PcieFabric;
using simllm::rnic::PcieGeneration;
using simllm::rnic::PciePathConfig;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicAuthorityAudit;
using simllm::rnic::RnicAuthoritySelection;
using simllm::rnic::RnicCompletionCsvRow;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceAttachments;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::RnicHardwareMode;
using simllm::rnic::RnicSessionConfigRecord;
using simllm::rnic::RnicSessionResultRecord;
using simllm::rnic::RnicWqeProjectionRecord;
using simllm::rnic::WqeId;
using simllm::rnic::WorkRequest;
using simllm::rnic::defaultPcieFabricConfig;
using simllm::rnic::effectiveHardwareConfigSha256;
using simllm::rnic::makeBypassSessionConfigRecord;
using simllm::rnic::makeBypassSessionResultRecord;
using simllm::rnic::makeStructuralSessionConfigRecord;
using simllm::rnic::projectStructuralSessionResult;
using simllm::rnic::renderRnicBookkeepingProjectionJson;
using simllm::rnic::renderRnicCompletionCsv;
using simllm::rnic::renderRnicSessionConfigJson;
using simllm::rnic::renderRnicSessionResultJson;
using simllm::rnic::rnicSha256Hex;
using simllm::rnic::testing::FakeNetworkPort;
using Mutation = std::function<void(RnicDeviceConfig&)>;

class TestRunner {
public:
    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template <typename Expected, typename Callable>
    void expectThrowAs(Callable&& callable, const std::string& message) {
        try {
            callable();
            check(false, message);
        } catch (const Expected&) {
            check(true, message);
        } catch (const std::exception& error) {
            check(
                false,
                message + "; wrong exception type: " + error.what());
        } catch (...) {
            check(false, message + "; wrong non-standard exception type");
        }
    }

    int failures() const noexcept { return failures_; }

private:
    int failures_{0};
};

struct SensitivityCase {
    SensitivityCase(
        std::string group_value,
        std::string field_value,
        Mutation mutate_value,
        Mutation prepare_value = {},
        bool baseline_shared_value = false,
        bool changed_shared_value = false)
        : group(std::move(group_value)),
          field(std::move(field_value)),
          mutate(std::move(mutate_value)),
          prepare(std::move(prepare_value)),
          baseline_shared(baseline_shared_value),
          changed_shared(changed_shared_value) {}

    std::string group;
    std::string field;
    Mutation mutate;
    Mutation prepare;
    bool baseline_shared{false};
    bool changed_shared{false};
};

struct CompletedFixture {
    RnicSessionConfigRecord config;
    RnicSessionResultRecord result;
    std::string config_json;
    std::string result_json;
    std::string bookkeeping_json;
    std::string completion_csv;
};

#ifndef SIMLLM_RNIC_SESSION_RECORD_EMBEDDED
std::string jsonString(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u00";
                constexpr char digits[] = "0123456789abcdef";
                output << digits[(character >> 4U) & 0xfU]
                       << digits[character & 0xfU];
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}
#endif

RnicDeviceConfig scalarConfig(
    std::size_t sq_depth = 64,
    Picoseconds doorbell_service_ps = 0) {
    RnicDeviceConfig config;
    config.identity.qpn = 17;
    config.identity.policy_context_token = 9001;
    config.work_queue.sq_id = 41;
    config.work_queue.cq_id = 43;
    config.work_queue.source = 3;
    config.work_queue.qpn = config.identity.qpn;
    config.work_queue.policy_context_token =
        config.identity.policy_context_token;
    config.work_queue.sq_depth = sq_depth;
    config.work_queue.cq_depth = 64;
    config.work_queue.doorbell_service_ps = doorbell_service_ps;
    config.work_queue.wqe_fetch_service_ps = 11;
    config.work_queue.qpc_lookup_service_ps = 5;
    config.work_queue.scheduler_service_ps = 7;
    config.work_queue.cqe_write_service_ps = 13;
    return config;
}

RnicDeviceConfig dmaConfig() {
    RnicDeviceConfig config = scalarConfig();
    config.work_queue.doorbell_service_ps = 0;
    config.work_queue.wqe_fetch_service_ps = 0;
    config.work_queue.cqe_write_service_ps = 0;
    config.dma.enabled = true;
    config.dma.fabric = defaultPcieFabricConfig();
    PciePathConfig extra_mmio = config.dma.fabric.paths[0];
    extra_mmio.path_id = 3;
    PciePathConfig extra_host = config.dma.fabric.paths[1];
    extra_host.path_id = 4;
    config.dma.fabric.paths.push_back(extra_mmio);
    config.dma.fabric.paths.push_back(extra_host);
    return config;
}

std::string hashConfig(
    const RnicDeviceConfig& config,
    bool shared_fabric = false) {
    FakeNetworkPort network(8, 0);
    RnicDeviceAttachments attachments;
    if (config.network.enabled) {
        attachments.network_port = &network;
    }
    if (shared_fabric) {
        attachments.shared_pcie_fabric =
            std::make_shared<PcieFabric>(config.dma.fabric);
    }
    RnicDevice device(config, attachments);
    return effectiveHardwareConfigSha256(device);
}

void configureFixed(PcieAnalyticalDelayProfile& profile) {
    profile.kind = PcieAnalyticalDelayKind::Fixed;
    profile.incidence_probability_ppm = 500'000;
    profile.mean_ps = 20;
}

void configureGaussian(PcieAnalyticalDelayProfile& profile) {
    profile.kind = PcieAnalyticalDelayKind::Gaussian;
    profile.incidence_probability_ppm = 500'000;
    profile.mean_ps = 20;
    profile.standard_deviation_ps = 2;
}

void configureGaussianTail(PcieAnalyticalDelayProfile& profile) {
    profile.kind = PcieAnalyticalDelayKind::GaussianTailMixture;
    profile.incidence_probability_ppm = 500'000;
    profile.mean_ps = 20;
    profile.standard_deviation_ps = 2;
    profile.tail_probability_ppm = 100'000;
    profile.tail_mean_ps = 40;
    profile.tail_standard_deviation_ps = 4;
}

#ifndef SIMLLM_RNIC_SESSION_RECORD_EMBEDDED
std::string configJson(
    const RnicDeviceConfig& config,
    const std::string& session_id,
    const std::string& policy) {
    FakeNetworkPort network(8, 0);
    RnicDeviceAttachments attachments;
    if (config.network.enabled) {
        attachments.network_port = &network;
    }
    RnicDevice device(config, attachments);
    return renderRnicSessionConfigJson(
        makeStructuralSessionConfigRecord(
            session_id, policy, device));
}
#endif

std::vector<SensitivityCase> sensitivityCases() {
    return {
        {"scalar", "work_queue.sq_depth", [](auto& c) { c.work_queue.sq_depth = 32; }},
        {"scalar", "work_queue.cq_depth", [](auto& c) { c.work_queue.cq_depth = 32; }},
        {"scalar", "work_queue.doorbell_service_ps", [](auto& c) { c.work_queue.doorbell_service_ps += 1; }},
        {"scalar", "work_queue.wqe_fetch_service_ps", [](auto& c) { c.work_queue.wqe_fetch_service_ps += 1; }},
        {"scalar", "work_queue.qpc_lookup_service_ps", [](auto& c) { c.work_queue.qpc_lookup_service_ps += 1; }},
        {"scalar", "work_queue.scheduler_service_ps", [](auto& c) { c.work_queue.scheduler_service_ps += 1; }},
        {"scalar", "work_queue.cqe_write_service_ps", [](auto& c) { c.work_queue.cqe_write_service_ps += 1; }},
        {"scalar", "qpc.enabled", [](auto& c) { c.qpc.enabled = false; c.work_queue.qpc_lookup_service_ps = 0; }},
        {"scalar", "network.enabled", [](auto& c) { c.network.enabled = true; }},
        {"scalar", "dma.enabled.activation", [](auto& c) {
             c.dma.enabled = true;
             c.work_queue.doorbell_service_ps = 0;
             c.work_queue.wqe_fetch_service_ps = 0;
             c.work_queue.cqe_write_service_ps = 0;
         }},
        {"dma_binding", "pcie_uar_path_id", [](auto& c) { c.dma.work_queue.pcie_uar_path_id = 3; }},
        {"dma_binding", "pcie_doorbell_record_path_id", [](auto& c) { c.dma.work_queue.pcie_doorbell_record_path_id = 4; }},
        {"dma_binding", "pcie_sq_memory_path_id", [](auto& c) { c.dma.work_queue.pcie_sq_memory_path_id = 4; }},
        {"dma_binding", "pcie_cq_memory_path_id", [](auto& c) { c.dma.work_queue.pcie_cq_memory_path_id = 4; }},
        {"dma_binding", "pcie_submission_ordering_domain", [](auto& c) { c.dma.work_queue.pcie_submission_ordering_domain = 101; }},
        {"dma_binding", "pcie_completion_ordering_domain", [](auto& c) { c.dma.work_queue.pcie_completion_ordering_domain = 100; }},
        {"dma_binding", "pcie_doorbell_record_bytes", [](auto& c) { c.dma.work_queue.pcie_doorbell_record_bytes += 4; }},
        {"dma_binding", "pcie_uar_doorbell_bytes", [](auto& c) { c.dma.work_queue.pcie_uar_doorbell_bytes += 4; }},
        {"dma_binding", "pcie_wqe_bytes", [](auto& c) { c.dma.work_queue.pcie_wqe_bytes += 4; }},
        {"dma_binding", "pcie_cqe_bytes", [](auto& c) { c.dma.work_queue.pcie_cqe_bytes += 4; }},
        {"dma_binding", "pcie_uar_first_byte_offset", [](auto& c) { c.dma.work_queue.pcie_uar_first_byte_offset = 4; }},
        {"dma_binding", "pcie_doorbell_record_first_byte_offset", [](auto& c) { c.dma.work_queue.pcie_doorbell_record_first_byte_offset = 4; }},
        {"dma_binding", "pcie_sq_first_byte_offset", [](auto& c) { c.dma.work_queue.pcie_sq_first_byte_offset = 4; }},
        {"dma_binding", "pcie_cq_first_byte_offset", [](auto& c) { c.dma.work_queue.pcie_cq_first_byte_offset = 4; }},
        {"dma_binding", "shared_ordering_domain_namespace", [](auto& c) {
             c.dma.shared_ordering_domain_namespace = 18;
         }, [](auto& c) {
             c.dma.shared_ordering_domain_namespace = 17;
         }, true, true},
        {"dma_binding", "fabric_scope", [](auto&) {}, [](auto& c) {
             c.dma.work_queue.pcie_submission_ordering_domain = 35;
             c.dma.work_queue.pcie_completion_ordering_domain = 34;
         }, false, true},
        {"pcie_fabric", "generation", [](auto& c) { c.dma.fabric.generation = PcieGeneration::Gen4; }},
        {"pcie_fabric", "lane_count", [](auto& c) { c.dma.fabric.lane_count = 8; }},
        {"pcie_fabric", "max_payload_size_bytes", [](auto& c) { c.dma.fabric.max_payload_size_bytes = 128; }},
        {"pcie_fabric", "max_read_request_size_bytes", [](auto& c) { c.dma.fabric.max_read_request_size_bytes = 256; }},
        {"pcie_fabric", "read_completion_boundary_bytes", [](auto& c) { c.dma.fabric.read_completion_boundary_bytes = 128; }},
        {"pcie_fabric", "posted_write_overhead_bytes", [](auto& c) { ++c.dma.fabric.posted_write_overhead_bytes; }},
        {"pcie_fabric", "read_request_overhead_bytes", [](auto& c) { ++c.dma.fabric.read_request_overhead_bytes; }},
        {"pcie_fabric", "completion_overhead_bytes", [](auto& c) { ++c.dma.fabric.completion_overhead_bytes; }},
        {"pcie_fabric", "data_credit_unit_bytes", [](auto& c) { c.dma.fabric.data_credit_unit_bytes = 32; }},
        {"pcie_fabric", "host_to_device_credits.posted_header_credits", [](auto& c) { ++c.dma.fabric.host_to_device_credits.posted_header_credits; }},
        {"pcie_fabric", "host_to_device_credits.posted_data_credits", [](auto& c) { ++c.dma.fabric.host_to_device_credits.posted_data_credits; }},
        {"pcie_fabric", "host_to_device_credits.nonposted_header_credits", [](auto& c) { ++c.dma.fabric.host_to_device_credits.nonposted_header_credits; }},
        {"pcie_fabric", "host_to_device_credits.completion_header_credits", [](auto& c) { ++c.dma.fabric.host_to_device_credits.completion_header_credits; }},
        {"pcie_fabric", "host_to_device_credits.completion_data_credits", [](auto& c) { ++c.dma.fabric.host_to_device_credits.completion_data_credits; }},
        {"pcie_fabric", "device_to_host_credits.posted_header_credits", [](auto& c) { ++c.dma.fabric.device_to_host_credits.posted_header_credits; }},
        {"pcie_fabric", "device_to_host_credits.posted_data_credits", [](auto& c) { ++c.dma.fabric.device_to_host_credits.posted_data_credits; }},
        {"pcie_fabric", "device_to_host_credits.nonposted_header_credits", [](auto& c) { ++c.dma.fabric.device_to_host_credits.nonposted_header_credits; }},
        {"pcie_fabric", "device_to_host_credits.completion_header_credits", [](auto& c) { ++c.dma.fabric.device_to_host_credits.completion_header_credits; }},
        {"pcie_fabric", "device_to_host_credits.completion_data_credits", [](auto& c) { ++c.dma.fabric.device_to_host_credits.completion_data_credits; }},
        {"pcie_fabric", "max_outstanding_read_requests", [](auto& c) { --c.dma.fabric.max_outstanding_read_requests; }},
        {"pcie_fabric", "completion_buffer_bytes", [](auto& c) { c.dma.fabric.completion_buffer_bytes /= 2; }},
        {"pcie_fabric", "max_tlps_per_transaction", [](auto& c) { --c.dma.fabric.max_tlps_per_transaction; }},
        {"pcie_fabric", "credit_return_latency_ps", [](auto& c) { c.dma.fabric.credit_return_latency_ps = 1; }},
        {"pcie_fabric", "completion_buffer_release_latency_ps", [](auto& c) { c.dma.fabric.completion_buffer_release_latency_ps = 1; }},
        {"pcie_fabric", "analytical_seed", [](auto& c) { c.dma.fabric.analytical_seed = 1; }},
        {"pcie_fabric", "host_store_latency_ps", [](auto& c) { c.dma.fabric.host_store_latency_ps.samples_ps[0] = 1; }},
        {"pcie_fabric", "posted_write_visibility_latency_ps", [](auto& c) { c.dma.fabric.posted_write_visibility_latency_ps.samples_ps[0] = 1; }},
        {"pcie_fabric", "read_completion_latency_ps", [](auto& c) { c.dma.fabric.read_completion_latency_ps.samples_ps[0] = 1; }},
        {"pcie_path", "path_id", [](auto& c) { c.dma.fabric.paths[3].path_id = 5; }},
        {"pcie_path", "endpoint", [](auto& c) { c.dma.fabric.paths[3].endpoint = PcieEndpointKind::DeviceMemory; }},
        {"pcie_path", "enabled", [](auto& c) { c.dma.fabric.paths[3].enabled = false; }},
        {"pcie_path", "base_latency_ps", [](auto& c) { c.dma.fabric.paths[3].base_latency_ps = 1; }},
        {"analytical", "component.numa.activation", [](auto& c) { configureFixed(c.dma.fabric.paths[3].analytical_penalties.numa); }},
        {"analytical", "component.iommu.activation", [](auto& c) { configureFixed(c.dma.fabric.paths[3].analytical_penalties.iommu); }},
        {"analytical", "component.acs.activation", [](auto& c) { configureFixed(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "component.switch_path.activation", [](auto& c) { configureFixed(c.dma.fabric.paths[3].analytical_penalties.switch_path); }},
        {"analytical", "component.ddio_miss.activation", [](auto& c) { configureFixed(c.dma.fabric.paths[3].analytical_penalties.ddio_miss); }},
        {"analytical", "component.gpu_direct.activation", [](auto& c) { configureFixed(c.dma.fabric.paths[3].analytical_penalties.gpu_direct); }},
        {"analytical", "profile.gaussian.activation", [](auto& c) { configureGaussian(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "profile.gaussian_tail_mixture.activation", [](auto& c) { configureGaussianTail(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "profile.incidence_probability_ppm", [](auto& c) { ++c.dma.fabric.paths[3].analytical_penalties.acs.incidence_probability_ppm; }, [](auto& c) { configureGaussianTail(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "profile.mean_ps", [](auto& c) { ++c.dma.fabric.paths[3].analytical_penalties.acs.mean_ps; }, [](auto& c) { configureGaussianTail(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "profile.standard_deviation_ps", [](auto& c) { ++c.dma.fabric.paths[3].analytical_penalties.acs.standard_deviation_ps; }, [](auto& c) { configureGaussianTail(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "profile.tail_probability_ppm", [](auto& c) { ++c.dma.fabric.paths[3].analytical_penalties.acs.tail_probability_ppm; }, [](auto& c) { configureGaussianTail(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "profile.tail_mean_ps", [](auto& c) { ++c.dma.fabric.paths[3].analytical_penalties.acs.tail_mean_ps; }, [](auto& c) { configureGaussianTail(c.dma.fabric.paths[3].analytical_penalties.acs); }},
        {"analytical", "profile.tail_standard_deviation_ps", [](auto& c) { ++c.dma.fabric.paths[3].analytical_penalties.acs.tail_standard_deviation_ps; }, [](auto& c) { configureGaussianTail(c.dma.fabric.paths[3].analytical_penalties.acs); }},
    };
}

CompletedFixture structuralFixture() {
    RnicDeviceConfig config = scalarConfig(8, 37);
    RnicDevice device(config);
    RnicAuthorityAudit audit(
        RnicHardwareMode::Structural, {true, false});
    for (std::uint64_t index = 0; index < 2; ++index) {
        WorkRequest request;
        request.wr_id = 100 + index;
        request.flow_id = 200 + index;
        request.flow_tag = static_cast<std::uint32_t>(7 + index);
        request.destination = 4;
        request.payload_bytes = 4096 * (index + 1);
        const auto posted = device.postSend(request, 0);
        if (posted.status != PostStatus::Accepted) {
            throw std::logic_error("structural fixture post was rejected");
        }
        audit.noteNativePost();
    }
    device.ringDoorbell(0);
    std::vector<CompletionEntry> completions;
    Picoseconds now_ps = 0;
    std::size_t iterations = 0;
    while (device.hasPendingPhysicalWork()) {
        const std::optional<Picoseconds> next = device.nextEventTime();
        if (!next.has_value() || *next < now_ps || ++iterations > 100) {
            throw std::logic_error("structural fixture lost event progress");
        }
        now_ps = *next;
        device.progress(now_ps);
        std::vector<CompletionEntry> polled =
            device.pollCompletionQueue(8, now_ps);
        completions.insert(
            completions.end(), polled.begin(), polled.end());
    }
    std::vector<CompletionEntry> final_poll =
        device.pollCompletionQueue(8, now_ps);
    completions.insert(
        completions.end(), final_poll.begin(), final_poll.end());
    RnicSessionConfigRecord record =
        makeStructuralSessionConfigRecord(
            "session-structural", "rnic-nn", device);
    RnicSessionResultRecord result = projectStructuralSessionResult(
        record, device, completions, audit);
    return {
        record,
        result,
        renderRnicSessionConfigJson(record),
        renderRnicSessionResultJson(result),
        renderRnicBookkeepingProjectionJson(result),
        renderRnicCompletionCsv(result),
    };
}

CompletedFixture bypassFixture(const CompletedFixture& structural) {
    RnicSessionConfigRecord config = makeBypassSessionConfigRecord(
        "session-bypass", "rnic-nn-fluid");
    RnicAuthorityAudit audit(
        RnicHardwareMode::Bypass, {false, true});
    audit.noteLegacyMutation(4);
    std::vector<RnicWqeProjectionRecord> wqes = structural.result.wqes;
    std::vector<RnicCompletionCsvRow> rows =
        structural.result.completion_rows;
    for (RnicWqeProjectionRecord& wqe : wqes) {
        wqe.key.session_id = config.session_id;
    }
    for (RnicCompletionCsvRow& row : rows) {
        row.profile = config.transport_policy;
        row.rq_id = 47;
    }
    RnicSessionResultRecord result = makeBypassSessionResultRecord(
        config, audit, std::move(wqes), std::move(rows), true);
    return {
        config,
        result,
        renderRnicSessionConfigJson(config),
        renderRnicSessionResultJson(result),
        renderRnicBookkeepingProjectionJson(result),
        renderRnicCompletionCsv(result),
    };
}

void testSha256(TestRunner& test) {
    test.check(
        rnicSha256Hex("")
            == "e3b0c44298fc1c149afbf4c8996fb924"
               "27ae41e4649b934ca495991b7852b855",
        "SHA-256 empty-string oracle matches");
    test.check(
        rnicSha256Hex("abc")
            == "ba7816bf8f01cfea414140de5dae2223"
               "b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc oracle matches");
}

void testHashContract(TestRunner& test) {
    for (std::size_t sq_depth : {32U, 64U}) {
        for (Picoseconds doorbell : {0ULL, 1000ULL}) {
            RnicDeviceConfig config = scalarConfig(sq_depth, doorbell);
            config.network.enabled = true;
            FakeNetworkPort network(8, 0);
            RnicDeviceAttachments attachments;
            attachments.network_port = &network;
            RnicDevice device(config, attachments);
            const std::string expected =
                effectiveHardwareConfigSha256(device);
            for (const std::string policy : {
                     "rnic-nn", "rnic-cn", "dcqcn"}) {
                const auto record = makeStructuralSessionConfigRecord(
                    "session-" + policy, policy, device);
                test.check(
                    record.hardware_config_sha256 == expected,
                    "transport policy is excluded from hardware hash");
            }
        }
    }
    const std::string scalar_baseline = hashConfig(scalarConfig());
    RnicDeviceConfig identity = scalarConfig();
    identity.identity.qpn = 19;
    identity.work_queue.qpn = 19;
    identity.identity.policy_context_token = 9003;
    identity.work_queue.policy_context_token = 9003;
    identity.work_queue.sq_id = 51;
    identity.work_queue.cq_id = 53;
    identity.work_queue.source = 7;
    test.check(
        hashConfig(identity) == scalar_baseline,
        "correlation and queue identities are excluded from scalar hash");
    RnicDeviceConfig inert = scalarConfig();
    inert.dma.fabric.lane_count = 0;
    inert.dma.fabric.paths.clear();
    inert.dma.work_queue.pcie_wqe_bytes = 0;
    inert.dma.shared_ordering_domain_namespace = 99;
    test.check(
        hashConfig(inert) == scalar_baseline,
        "disabled DMA payload is excluded from scalar hash");

    const auto cases = sensitivityCases();
    test.check(cases.size() == 72, "sensitivity census has audited breadth");
    for (const SensitivityCase& item : cases) {
        RnicDeviceConfig baseline =
            item.group == "scalar" ? scalarConfig() : dmaConfig();
        if (item.prepare) {
            item.prepare(baseline);
        }
        RnicDeviceConfig changed = baseline;
        item.mutate(changed);
        test.check(
            hashConfig(baseline, item.baseline_shared)
                != hashConfig(changed, item.changed_shared),
            "effective mutation changes hash: " + item.field);
    }

    RnicDeviceConfig reordered = dmaConfig();
    std::swap(reordered.dma.fabric.paths[2], reordered.dma.fabric.paths[3]);
    test.check(
        hashConfig(reordered) == hashConfig(dmaConfig()),
        "path declaration order is excluded from hardware identity");
    RnicDeviceConfig disabled_path = dmaConfig();
    disabled_path.dma.fabric.paths[3].enabled = false;
    const std::string disabled_path_hash = hashConfig(disabled_path);
    disabled_path.dma.fabric.paths[3].endpoint =
        PcieEndpointKind::DeviceMemory;
    disabled_path.dma.fabric.paths[3].base_latency_ps = 99;
    configureFixed(
        disabled_path.dma.fabric.paths[3].analytical_penalties.numa);
    test.check(
        hashConfig(disabled_path) == disabled_path_hash,
        "disabled path payload is excluded from hardware identity");
    RnicDeviceConfig inert_nonposted_data = dmaConfig();
    ++inert_nonposted_data.dma.fabric.host_to_device_credits
         .nonposted_data_credits;
    ++inert_nonposted_data.dma.fabric.device_to_host_credits
         .nonposted_data_credits;
    test.check(
        hashConfig(inert_nonposted_data) == hashConfig(dmaConfig()),
        "unused nonposted-data credit pools are excluded from identity");
}

void testAuthorityExclusivity(TestRunner& test) {
    bool mutation_reached = false;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            RnicAuthorityAudit invalid(
                RnicHardwareMode::Structural, {true, true});
            mutation_reached = true;
            invalid.noteNativePost();
        },
        "both authorities fail before audit mutation");
    test.check(
        !mutation_reached,
        "both-authority negative control did not reach mutation");
    test.expectThrowAs<std::invalid_argument>(
        []() {
            RnicAuthorityAudit invalid(
                RnicHardwareMode::Bypass, {false, false});
            (void)invalid;
        },
        "neither authority is rejected");
    RnicAuthorityAudit structural(
        RnicHardwareMode::Structural, {true, false});
    test.expectThrowAs<std::logic_error>(
        [&]() { structural.noteLegacyMutation(); },
        "structural mode rejects legacy mutation");
    test.check(
        structural.counters().legacy_mutations == 0
            && structural.counters().native_posts == 0,
        "failed structural misuse preserves audit counters");
    RnicAuthorityAudit bypass(
        RnicHardwareMode::Bypass, {false, true});
    test.expectThrowAs<std::logic_error>(
        [&]() { bypass.noteNativePost(); },
        "bypass mode rejects native post");
    test.check(
        bypass.counters().legacy_mutations == 0
            && bypass.counters().native_posts == 0,
        "failed bypass misuse preserves audit counters");
}

void testProjection(TestRunner& test) {
    const CompletedFixture structural = structuralFixture();
    const CompletedFixture bypass = bypassFixture(structural);
    test.check(
        structural.result.authority_counters.native_session_constructed == 1
            && structural.result.authority_counters.legacy_ledger_constructed == 0
            && structural.result.authority_counters.native_posts == 2
            && structural.result.authority_counters.legacy_mutations == 0,
        "structural result reports the frozen authority counter tuple");
    test.check(
        bypass.result.authority_counters.native_session_constructed == 0
            && bypass.result.authority_counters.legacy_ledger_constructed == 1
            && bypass.result.authority_counters.native_posts == 0
            && bypass.result.authority_counters.legacy_mutations == 4,
        "bypass result reports the frozen authority counter tuple");
    test.check(
        std::all_of(
            structural.result.completion_rows.begin(),
            structural.result.completion_rows.end(),
            [](const RnicCompletionCsvRow& row) {
                return !row.rq_id.has_value();
            }),
        "structural send projection does not invent an RQ");
    test.check(
        structural.completion_csv.find("\r") == std::string::npos
            && structural.completion_csv.rfind(
                "profile,flow_id,source,destination,tag,payload_bytes,",
                0)
                == 0,
        "completion projection uses pinned header and LF bytes");
    test.check(
        structural.config_json.find(
            "\"schema\":\"simllm-rnic-session-config-v1\"")
                != std::string::npos
            && structural.result_json.find(
                "\"schema\":\"simllm-rnic-session-result-v1\"")
                != std::string::npos
            && structural.bookkeeping_json.find(
                "\"schema\":\"simllm-rnic-bookkeeping-v1\"")
                != std::string::npos,
        "all native record surfaces are schema tagged");
    test.check(
        bypass.config_json.find(
            "\"authority\":\"AtlahsWqeLedger\"")
                != std::string::npos
            && bypass.config_json.find(
                "\"hardware_mode\":\"bypass\"")
                != std::string::npos
            && bypass.config_json.find(
                "\"hardware_config_sha256\":null")
            != std::string::npos,
        "bypass configuration is explicit and hashless");

    RnicSessionConfigRecord malformed_config = structural.config;
    malformed_config.effective_hardware_json = "null";
    malformed_config.hardware_config_sha256 = rnicSha256Hex("null");
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionConfigRecord(
                malformed_config);
        },
        "structural configuration rejects a non-object hash payload");
    const std::string null_modules =
        "{\"dma\":null,\"network\":null,\"qpc\":null,"
        "\"schema\":\"simllm-rnic-effective-hardware-v1\","
        "\"work_queue\":null}";
    malformed_config.effective_hardware_json = null_modules;
    malformed_config.hardware_config_sha256 =
        rnicSha256Hex(null_modules);
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionConfigRecord(
                malformed_config);
        },
        "structural configuration rejects null schema members");
    std::string zero_depth = *structural.config.effective_hardware_json;
    const std::string depth = "\"sq_depth\":8";
    const std::size_t depth_position = zero_depth.find(depth);
    if (depth_position == std::string::npos) {
        test.check(false, "structural fixture exposes its SQ-depth field");
    } else {
        zero_depth.replace(
            depth_position, depth.size(), "\"sq_depth\":0");
        malformed_config.effective_hardware_json = zero_depth;
        malformed_config.hardware_config_sha256 =
            rnicSha256Hex(zero_depth);
        test.expectThrowAs<std::invalid_argument>(
            [&]() {
                simllm::rnic::validateRnicSessionConfigRecord(
                    malformed_config);
            },
            "structural configuration rejects impossible queue depth");
    }

    RnicSessionResultRecord timestamp_drift = structural.result;
    ++timestamp_drift.completion_rows[0].completion_time_ps;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(
                timestamp_drift);
        },
        "completion timestamp drift is rejected");
    RnicSessionResultRecord duplicate = structural.result;
    duplicate.wqes[1].key = duplicate.wqes[0].key;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(duplicate);
        },
        "duplicate stable key is rejected");
    RnicSessionResultRecord invented_rq = structural.result;
    invented_rq.completion_rows[0].rq_id = 99;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(invented_rq);
        },
        "structural send RQ fabrication is rejected");
    RnicSessionResultRecord wrong_bypass_count = bypass.result;
    wrong_bypass_count.authority_counters.legacy_mutations = 0;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(
                wrong_bypass_count);
        },
        "bypass mutations reconcile with its WQE count");
    RnicSessionResultRecord nonmonotonic = structural.result;
    nonmonotonic.wqes[0].timeline.qpc_ready_at_ps = 0;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(nonmonotonic);
        },
        "nonmonotonic WQE projection is rejected");
    RnicSessionResultRecord half_packet_extent = structural.result;
    half_packet_extent.wqes[0].timeline.first_packet_at_ps =
        half_packet_extent.wqes[0].timeline.network_accepted_at_ps;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(
                half_packet_extent);
        },
        "packet timestamps must appear as a pair");
    RnicSessionResultRecord packet_without_acceptance = structural.result;
    packet_without_acceptance.wqes[0].timeline.network_accepted_at_ps.reset();
    packet_without_acceptance.wqes[0].timeline.first_packet_at_ps =
        packet_without_acceptance.wqes[0].timeline.network_outcome_at_ps;
    packet_without_acceptance.wqes[0].timeline.last_packet_at_ps =
        packet_without_acceptance.wqes[0].timeline.network_outcome_at_ps;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(
                packet_without_acceptance);
        },
        "packet timestamps require network acceptance");
    RnicSessionResultRecord impossible_state = structural.result;
    impossible_state.wqes[0].state = simllm::rnic::WqeState::Posted;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(
                impossible_state);
        },
        "quiescent projection requires a terminal WQE state");
    RnicSessionResultRecord false_unsignaled = structural.result;
    false_unsignaled.wqes[0].signaled = false;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(
                false_unsignaled);
        },
        "unsignaled projection cannot carry CQ sequences");

    RnicSessionResultRecord separate_cqs = bypass.result;
    RnicWqeProjectionRecord& second_wqe = separate_cqs.wqes[1];
    second_wqe.cq_id = separate_cqs.wqes[0].cq_id + 1;
    second_wqe.cqe_sequence = separate_cqs.wqes[0].cqe_sequence;
    second_wqe.cq_producer_index =
        separate_cqs.wqes[0].cq_producer_index;
    second_wqe.cq_consume_sequence =
        separate_cqs.wqes[0].cq_consume_sequence;
    for (RnicCompletionCsvRow& row : separate_cqs.completion_rows) {
        if (row.wqe_id == std::optional<WqeId>(second_wqe.wqe_id)) {
            row.cq_id = second_wqe.cq_id;
            row.cq_post_sequence = second_wqe.cqe_sequence;
            row.cq_consume_sequence = second_wqe.cq_consume_sequence;
        }
    }
    try {
        simllm::rnic::validateRnicSessionResultRecord(separate_cqs);
        test.check(true, "CQE sequences are scoped by CQ identity");
    } catch (const std::exception& error) {
        test.check(
            false,
            std::string("CQ-scoped sequence was rejected: ")
                + error.what());
    }
    RnicSessionResultRecord duplicate_cq_sequence = separate_cqs;
    duplicate_cq_sequence.wqes[1].cq_id =
        duplicate_cq_sequence.wqes[0].cq_id;
    for (RnicCompletionCsvRow& row :
         duplicate_cq_sequence.completion_rows) {
        if (row.wqe_id
            == std::optional<WqeId>(
                duplicate_cq_sequence.wqes[1].wqe_id)) {
            row.cq_id = duplicate_cq_sequence.wqes[1].cq_id;
        }
    }
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(
                duplicate_cq_sequence);
        },
        "duplicate CQE sequence is rejected within one CQ");
    RnicSessionResultRecord zero_cqe = bypass.result;
    zero_cqe.wqes[0].cqe_sequence = 0;
    zero_cqe.wqes[0].cq_consume_sequence = 0;
    zero_cqe.completion_rows[0].cq_post_sequence = 0;
    zero_cqe.completion_rows[0].cq_consume_sequence = 0;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            simllm::rnic::validateRnicSessionResultRecord(zero_cqe);
        },
        "CQE sequences are one based");
}

void testUnsignaledErrorProjection(TestRunner& test) {
    RnicDeviceConfig config = scalarConfig(2, 0);
    config.qpc.enabled = false;
    config.work_queue.qpc_lookup_service_ps = 0;
    config.work_queue.wqe_fetch_service_ps = 0;
    config.work_queue.scheduler_service_ps = 0;
    config.work_queue.cqe_write_service_ps = 0;
    config.network.enabled = true;
    FakeNetworkPort network(1, 0);
    network.rejectNext();
    RnicDeviceAttachments attachments;
    attachments.network_port = &network;
    RnicDevice device(config, attachments);
    RnicAuthorityAudit audit(
        RnicHardwareMode::Structural, {true, false});
    WorkRequest request;
    request.wr_id = 901;
    request.flow_id = 902;
    request.flow_tag = 9;
    request.destination = 4;
    request.payload_bytes = 128;
    request.signaled = false;
    const auto posted = device.postSend(request, 0);
    audit.noteNativePost();
    device.ringDoorbell(0);
    device.progress(0);
    std::vector<CompletionEntry> completions =
        device.pollCompletionQueue(1, 0);
    const RnicSessionConfigRecord session =
        makeStructuralSessionConfigRecord(
            "session-unsignaled-error", "rnic-cn", device);
    try {
        const RnicSessionResultRecord result =
            projectStructuralSessionResult(
                session, device, completions, audit);
        test.check(
            posted.status == PostStatus::Accepted
                && completions.size() == 1 && result.wqes.size() == 1
                && !result.wqes[0].signaled
                && result.wqes[0].completion_status
                    == std::optional<CompletionStatus>(
                        CompletionStatus::NetworkRejected)
                && result.wqes[0].cqe_sequence.has_value()
                && result.wqes[0].state
                    == simllm::rnic::WqeState::Completed,
            "unsignaled failure projects its mandatory error CQE");
    } catch (const std::exception& error) {
        test.check(
            false,
            std::string("unsignaled failure projection was rejected: ")
                + error.what());
    }
}

#ifndef SIMLLM_RNIC_SESSION_RECORD_EMBEDDED
std::string boolObject(
    const std::vector<std::pair<std::string, bool>>& values) {
    std::ostringstream output;
    output << '{';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << jsonString(values[index].first) << ':'
               << (values[index].second ? "true" : "false");
    }
    output << '}';
    return output.str();
}

std::string studyJson() {
    const CompletedFixture structural = structuralFixture();
    const CompletedFixture bypass = bypassFixture(structural);
    std::ostringstream output;
    output << '{';
    output << "\"schema\":\"simllm-rnic-session-record-study-v1\",";
    output << "\"hash_rows\":[";
    bool first = true;
    for (std::size_t sq_depth : {32U, 64U}) {
        for (Picoseconds doorbell : {0ULL, 1000ULL}) {
            for (const std::string policy : {
                     "rnic-nn", "rnic-cn", "dcqcn"}) {
                if (!first) {
                    output << ',';
                }
                first = false;
                RnicDeviceConfig config =
                    scalarConfig(sq_depth, doorbell);
                config.network.enabled = true;
                output << "{\"config\":"
                       << configJson(
                              config,
                              "hash-" + std::to_string(sq_depth) + "-"
                                  + std::to_string(doorbell) + "-" + policy,
                              policy)
                       << ",\"doorbell_service_ps\":" << doorbell
                       << ",\"policy\":" << jsonString(policy)
                       << ",\"sq_depth\":" << sq_depth << '}';
            }
        }
    }
    output << "],\"sensitivity_rows\":[";
    const auto cases = sensitivityCases();
    for (std::size_t index = 0; index < cases.size(); ++index) {
        const SensitivityCase& item = cases[index];
        RnicDeviceConfig baseline =
            item.group == "scalar" ? scalarConfig() : dmaConfig();
        if (item.prepare) {
            item.prepare(baseline);
        }
        RnicDeviceConfig changed = baseline;
        item.mutate(changed);
        if (index != 0) {
            output << ',';
        }
        output << "{\"after_hash\":"
               << jsonString(hashConfig(changed, item.changed_shared))
               << ",\"before_hash\":"
               << jsonString(hashConfig(baseline, item.baseline_shared))
               << ",\"field\":" << jsonString(item.field)
               << ",\"group\":" << jsonString(item.group) << '}';
    }
    output << "],\"inactive_hash_guards\":";
    const std::string baseline = hashConfig(scalarConfig());
    RnicDeviceConfig identities = scalarConfig();
    identities.identity.qpn = 19;
    identities.work_queue.qpn = 19;
    identities.identity.policy_context_token = 9003;
    identities.work_queue.policy_context_token = 9003;
    identities.work_queue.sq_id = 51;
    identities.work_queue.cq_id = 53;
    identities.work_queue.source = 7;
    RnicDeviceConfig inert = scalarConfig();
    inert.dma.fabric.lane_count = 0;
    inert.dma.fabric.paths.clear();
    RnicDevice hash_identity_device(scalarConfig());
    const auto session_a = makeStructuralSessionConfigRecord(
        "hash-session-a", "rnic-nn", hash_identity_device);
    const auto session_b = makeStructuralSessionConfigRecord(
        "hash-session-b", "rnic-nn", hash_identity_device);
    const auto policy_b = makeStructuralSessionConfigRecord(
        "hash-session-a", "dcqcn", hash_identity_device);
    RnicDeviceConfig reordered = dmaConfig();
    std::swap(reordered.dma.fabric.paths[2], reordered.dma.fabric.paths[3]);
    RnicDeviceConfig disabled_path = dmaConfig();
    disabled_path.dma.fabric.paths[3].enabled = false;
    const std::string disabled_path_hash = hashConfig(disabled_path);
    disabled_path.dma.fabric.paths[3].endpoint =
        PcieEndpointKind::DeviceMemory;
    disabled_path.dma.fabric.paths[3].base_latency_ps = 99;
    configureFixed(
        disabled_path.dma.fabric.paths[3].analytical_penalties.numa);
    RnicDeviceConfig inert_nonposted_data = dmaConfig();
    ++inert_nonposted_data.dma.fabric.host_to_device_credits
         .nonposted_data_credits;
    ++inert_nonposted_data.dma.fabric.device_to_host_credits
         .nonposted_data_credits;
    output << boolObject({
        {"policy_permutation", session_a.hardware_config_sha256
            == policy_b.hardware_config_sha256},
        {"session_id", session_a.hardware_config_sha256
            == session_b.hardware_config_sha256},
        {"correlation_identity", hashConfig(identities) == baseline},
        {"disabled_dma_payload", hashConfig(inert) == baseline},
        {"path_declaration_order", hashConfig(reordered)
            == hashConfig(dmaConfig())},
        {"disabled_path_payload", hashConfig(disabled_path)
            == disabled_path_hash},
        {"unused_nonposted_data_credits",
         hashConfig(inert_nonposted_data) == hashConfig(dmaConfig())},
    });
    output << ",\"dma_config\":"
           << configJson(dmaConfig(), "session-dma-reader", "rnic-nn");
    output << ",\"structural_config\":" << structural.config_json;
    output << ",\"structural_result\":" << structural.result_json;
    output << ",\"structural_bookkeeping\":"
           << structural.bookkeeping_json;
    output << ",\"structural_completion_csv\":"
           << jsonString(structural.completion_csv);
    output << ",\"bypass_config\":" << bypass.config_json;
    output << ",\"bypass_result\":" << bypass.result_json;
    output << ",\"bypass_bookkeeping\":" << bypass.bookkeeping_json;
    output << ",\"projection_checks\":";
    output << boolObject({
        {"one_to_one", structural.result.wqes.size() == 2
            && structural.result.completion_rows.size() == 2},
        {"stable_keys", structural.result.wqes[0].key.post_sequence == 1
            && structural.result.wqes[1].key.post_sequence == 2},
        {"no_structural_rq", std::none_of(
            structural.result.completion_rows.begin(),
            structural.result.completion_rows.end(),
            [](const auto& row) { return row.rq_id.has_value(); })},
        {"native_timestamps", std::all_of(
            structural.result.wqes.begin(),
            structural.result.wqes.end(),
            [](const auto& wqe) {
                return wqe.timeline.network_outcome_at_ps.has_value()
                    && wqe.timeline.cqe_visible_at_ps.has_value()
                    && wqe.timeline.polled_at_ps.has_value();
            })},
    });
    output << ",\"authority_negative_controls\":";
    bool both_rejected = false;
    bool neither_rejected = false;
    bool wrong_native_rejected = false;
    bool wrong_legacy_rejected = false;
    try {
        RnicAuthorityAudit invalid(
            RnicHardwareMode::Structural, {true, true});
        (void)invalid;
    } catch (const std::invalid_argument&) {
        both_rejected = true;
    }
    try {
        RnicAuthorityAudit invalid(
            RnicHardwareMode::Bypass, {false, false});
        (void)invalid;
    } catch (const std::invalid_argument&) {
        neither_rejected = true;
    }
    RnicAuthorityAudit structural_audit(
        RnicHardwareMode::Structural, {true, false});
    try {
        structural_audit.noteLegacyMutation();
    } catch (const std::logic_error&) {
        wrong_legacy_rejected = true;
    }
    RnicAuthorityAudit bypass_audit(
        RnicHardwareMode::Bypass, {false, true});
    try {
        bypass_audit.noteNativePost();
    } catch (const std::logic_error&) {
        wrong_native_rejected = true;
    }
    output << boolObject({
        {"both_rejected", both_rejected},
        {"neither_rejected", neither_rejected},
        {"wrong_native_rejected", wrong_native_rejected},
        {"wrong_legacy_rejected", wrong_legacy_rejected},
        {"failed_counters_unchanged",
         structural_audit.counters().legacy_mutations == 0
             && bypass_audit.counters().native_posts == 0},
    });
    output << '}';
    return output.str();
}
#endif

}  // namespace

int runRnicSessionRecordChecks() {
    TestRunner test;
    testSha256(test);
    testHashContract(test);
    testAuthorityExclusivity(test);
    testProjection(test);
    testUnsignaledErrorProjection(test);
    return test.failures();
}

#ifndef SIMLLM_RNIC_SESSION_RECORD_EMBEDDED
int main(int argc, char** argv) {
    if (argc == 4
        && std::string(argv[1]) == "--validate-effective-hardware") {
        try {
            std::ifstream input(argv[2], std::ios::binary);
            if (!input) {
                std::cerr << "cannot open effective-hardware input\n";
                return 2;
            }
            std::ostringstream bytes;
            bytes << input.rdbuf();
            if (!input.good() && !input.eof()) {
                std::cerr << "cannot read effective-hardware input\n";
                return 2;
            }
            RnicSessionConfigRecord record;
            record.session_id = "effective-hardware-probe";
            record.hardware_mode = RnicHardwareMode::Structural;
            record.authority =
                simllm::rnic::RnicWqeAuthority::SimllmNativeRnicSession;
            record.transport_policy = "rnic-nn";
            record.effective_hardware_json = bytes.str();
            record.hardware_config_sha256 = argv[3];
            simllm::rnic::validateRnicSessionConfigRecord(record);
            std::cout << "accepted\n";
            return 0;
        } catch (const std::invalid_argument& error) {
            std::cerr << "rejected: " << error.what() << '\n';
            return 1;
        } catch (const std::exception& error) {
            std::cerr << "effective-hardware probe failed: "
                      << error.what() << '\n';
            return 2;
        }
    }
    if (argc == 2 && std::string(argv[1]) == "--study-json") {
        try {
            std::cout << studyJson() << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "session record study failed: "
                      << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 1) {
        std::cerr << "usage: " << argv[0]
                  << " [--study-json | --validate-effective-hardware FILE SHA256]\n";
        return 2;
    }
    const int failures = runRnicSessionRecordChecks();
    if (failures != 0) {
        std::cerr << failures
                  << " RNIC session-record checks failed\n";
        return 1;
    }
    std::cout << "RNIC session-record checks passed\n";
    return 0;
}
#endif
