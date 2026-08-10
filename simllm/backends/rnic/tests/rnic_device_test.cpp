#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "fake_network.h"
#include "simllm/rnic/rnic_device.h"

int runRnicSessionRecordChecks();

namespace {

using simllm::rnic::CompletionEntry;
using simllm::rnic::DoorbellBatch;
using simllm::rnic::EvidenceEvent;
using simllm::rnic::NetworkEvent;
using simllm::rnic::NetworkPort;
using simllm::rnic::NetworkSubmitResult;
using simllm::rnic::NetworkToken;
using simllm::rnic::PcieAnalyticalDelayAccounting;
using simllm::rnic::PcieClassAccounting;
using simllm::rnic::PcieFabric;
using simllm::rnic::PcieServiceClass;
using simllm::rnic::PcieTransactionRequest;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostResult;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceAttachments;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::RnicDmaConfig;
using simllm::rnic::RnicStageApplicability;
using simllm::rnic::WorkQueue;
using simllm::rnic::WorkQueueConfig;
using simllm::rnic::WorkQueueCounters;
using simllm::rnic::WorkQueuePcieBinding;
using simllm::rnic::WorkRequest;
using simllm::rnic::WqeId;
using simllm::rnic::WqeRecord;
using simllm::rnic::defaultPcieFabricConfig;
using simllm::rnic::testing::FakeNetworkPortState;
using simllm::rnic::testing::FakeNetworkPort;
using simllm::rnic::testing::FakeSubmission;

static_assert(!std::is_copy_constructible_v<RnicDevice>);
static_assert(!std::is_copy_assignable_v<RnicDevice>);
static_assert(!std::is_move_constructible_v<RnicDevice>);
static_assert(!std::is_move_assignable_v<RnicDevice>);

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

class ReferenceInertPort final : public NetworkPort {
public:
    NetworkSubmitResult trySubmit(
        const simllm::rnic::NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        const NetworkToken token = next_token_++;
        pending_.emplace(token, Pending{descriptor.wqe_id, now_ps});
        return NetworkSubmitResult::accepted(token);
    }

    std::optional<Picoseconds> nextEventTime() const {
        std::optional<Picoseconds> next;
        for (const auto& token_and_pending : pending_) {
            if (!next.has_value()
                || token_and_pending.second.event_time < *next) {
                next = token_and_pending.second.event_time;
            }
        }
        return next;
    }

    std::vector<NetworkEvent> takeDue(Picoseconds now_ps) {
        std::vector<NetworkEvent> events;
        for (auto item = pending_.begin(); item != pending_.end();) {
            if (item->second.event_time > now_ps) {
                ++item;
                continue;
            }
            NetworkEvent event;
            event.token = item->first;
            event.wqe_id = item->second.wqe_id;
            event.event_time_ps = item->second.event_time;
            events.push_back(event);
            item = pending_.erase(item);
        }
        return events;
    }

private:
    struct Pending {
        WqeId wqe_id{0};
        Picoseconds event_time{0};
    };

    NetworkToken next_token_{1};
    std::map<NetworkToken, Pending> pending_;
};

class CapturingPort final : public NetworkPort {
public:
    NetworkSubmitResult trySubmit(
        const simllm::rnic::NetworkTxDescriptor& descriptor,
        Picoseconds) override {
        descriptors.push_back(descriptor);
        return NetworkSubmitResult::accepted(next_token_++);
    }

    std::vector<simllm::rnic::NetworkTxDescriptor> descriptors;

private:
    NetworkToken next_token_{1};
};

template <typename Value>
void appendValue(std::ostringstream& output, const Value& value) {
    if constexpr (std::is_enum_v<Value>) {
        output << static_cast<std::underlying_type_t<Value>>(value);
    } else {
        output << value;
    }
    output << '|';
}

template <typename Value>
void appendOptional(
    std::ostringstream& output,
    const std::optional<Value>& value) {
    appendValue(output, value.has_value());
    if (value.has_value()) {
        appendValue(output, *value);
    }
}

void appendWorkQueueConfig(
    std::ostringstream& output,
    const WorkQueueConfig& config) {
    appendValue(output, config.version);
    appendValue(output, config.sq_id);
    appendValue(output, config.cq_id);
    appendValue(output, config.source);
    appendValue(output, config.qpn);
    appendValue(output, config.policy_context_token);
    appendValue(output, config.sq_depth);
    appendValue(output, config.cq_depth);
    appendValue(output, config.doorbell_service_ps);
    appendValue(output, config.wqe_fetch_service_ps);
    appendValue(output, config.qpc_lookup_service_ps);
    appendValue(output, config.scheduler_service_ps);
    appendValue(output, config.cqe_write_service_ps);
}

void appendPcieBinding(
    std::ostringstream& output,
    const std::optional<WorkQueuePcieBinding>& binding) {
    appendValue(output, binding.has_value());
    if (!binding.has_value()) {
        return;
    }
    appendValue(output, binding->version);
    appendValue(output, binding->pcie_uar_path_id);
    appendValue(output, binding->pcie_doorbell_record_path_id);
    appendValue(output, binding->pcie_sq_memory_path_id);
    appendValue(output, binding->pcie_cq_memory_path_id);
    appendValue(output, binding->pcie_submission_ordering_domain);
    appendValue(output, binding->pcie_completion_ordering_domain);
    appendValue(output, binding->pcie_doorbell_record_bytes);
    appendValue(output, binding->pcie_uar_doorbell_bytes);
    appendValue(output, binding->pcie_wqe_bytes);
    appendValue(output, binding->pcie_cqe_bytes);
    appendValue(output, binding->pcie_uar_first_byte_offset);
    appendValue(output, binding->pcie_doorbell_record_first_byte_offset);
    appendValue(output, binding->pcie_sq_first_byte_offset);
    appendValue(output, binding->pcie_cq_first_byte_offset);
}

void appendRequest(std::ostringstream& output, const WorkRequest& request) {
    appendValue(output, request.wr_id);
    appendValue(output, request.flow_id);
    appendValue(output, request.flow_tag);
    appendValue(output, request.destination);
    appendValue(output, request.payload_bytes);
    appendValue(output, request.traffic_class);
    appendValue(output, request.opcode);
    appendValue(output, request.signaled);
}

void appendNetworkDescriptor(
    std::ostringstream& output,
    const simllm::rnic::NetworkTxDescriptor& descriptor) {
    appendValue(output, descriptor.abi_version);
    appendValue(output, descriptor.wqe_id);
    appendValue(output, descriptor.wr_id);
    appendValue(output, descriptor.flow_id);
    appendValue(output, descriptor.flow_tag);
    appendValue(output, descriptor.policy_context_token);
    appendValue(output, descriptor.source);
    appendValue(output, descriptor.destination);
    appendValue(output, descriptor.qpn);
    appendValue(output, descriptor.traffic_class);
    appendValue(output, descriptor.payload_bytes);
    appendValue(output, descriptor.extent_index);
    appendValue(output, descriptor.extent_count);
    appendValue(output, descriptor.eligible_at_ps);
}

void appendFakeSubmission(
    std::ostringstream& output,
    const FakeSubmission& submission) {
    appendValue(output, submission.token);
    appendNetworkDescriptor(output, submission.descriptor);
    appendValue(output, submission.submitted_at_ps);
    appendValue(output, submission.completion_at_ps);
}

std::string canonicalPortState(const FakeNetworkPort& port) {
    const FakeNetworkPortState state = port.state();
    std::ostringstream output;
    appendValue(output, state.capacity);
    appendValue(output, state.latency_ps);
    appendValue(output, state.next_token);
    appendValue(output, state.reject_next);
    appendOptional(output, state.forced_busy_until_ps);
    appendValue(output, state.inflight.size());
    for (const auto& token_and_submission : state.inflight) {
        appendValue(output, token_and_submission.first);
        appendFakeSubmission(output, token_and_submission.second);
    }
    appendValue(output, state.history.size());
    for (const FakeSubmission& submission : state.history) {
        appendFakeSubmission(output, submission);
    }
    return output.str();
}

void appendRecord(std::ostringstream& output, const WqeRecord& record) {
    appendValue(output, record.wqe_id);
    appendValue(output, record.sq_sequence);
    appendValue(output, record.doorbell_batch_id);
    appendRequest(output, record.request);
    appendValue(output, record.timeline.posted_at_ps);
    appendOptional(output, record.timeline.doorbelled_at_ps);
    appendOptional(output, record.timeline.doorbell_seen_at_ps);
    appendOptional(output, record.timeline.wqe_fetch_begin_at_ps);
    appendOptional(output, record.timeline.wqe_fetch_end_at_ps);
    appendOptional(output, record.timeline.qpc_ready_at_ps);
    appendOptional(output, record.timeline.admitted_at_ps);
    appendOptional(output, record.timeline.network_accepted_at_ps);
    appendOptional(output, record.timeline.network_outcome_at_ps);
    appendOptional(output, record.timeline.first_packet_at_ps);
    appendOptional(output, record.timeline.last_packet_at_ps);
    appendOptional(output, record.timeline.transport_retired_at_ps);
    appendOptional(output, record.timeline.cqe_visible_at_ps);
    appendOptional(output, record.timeline.polled_at_ps);
    appendOptional(output, record.timeline.sq_reclaimed_at_ps);
    appendValue(output, record.state);
    appendOptional(output, record.network_token);
    appendOptional(output, record.completion_status);
    appendValue(output, record.ecn_marked);
}

void appendCompletion(
    std::ostringstream& output,
    const CompletionEntry& completion) {
    appendValue(output, completion.cqe_sequence);
    appendValue(output, completion.cq_producer_index);
    appendValue(output, completion.cq_slot);
    appendValue(output, completion.owner_generation);
    appendValue(output, completion.wr_id);
    appendValue(output, completion.wqe_id);
    appendValue(output, completion.sq_sequence);
    appendValue(output, completion.qpn);
    appendValue(output, completion.opcode);
    appendValue(output, completion.status);
    appendValue(output, completion.valid_fields);
    appendValue(output, completion.byte_count);
    appendValue(output, completion.vendor_syndrome);
    appendValue(output, completion.visible_at_ps);
    appendValue(output, completion.polled_at_ps);
}

void appendEvidence(std::ostringstream& output, const EvidenceEvent& event) {
    appendValue(output, event.tier);
    appendValue(output, event.kind);
    appendValue(output, event.wqe_id);
    appendValue(output, event.wr_id);
    appendValue(output, event.observed_at_ps);
    appendValue(output, event.drop_location);
    appendValue(output, event.drop_reason);
}

void appendCounters(
    std::ostringstream& output,
    const WorkQueueCounters& counters) {
    appendValue(output, counters.posted_wqes);
    appendValue(output, counters.sq_full_rejections);
    appendValue(output, counters.doorbells);
    appendValue(output, counters.doorbelled_wqes);
    appendValue(output, counters.network_submit_attempts);
    appendValue(output, counters.network_accepted);
    appendValue(output, counters.network_busy);
    appendValue(output, counters.network_rejected);
    appendValue(output, counters.network_delivered);
    appendValue(output, counters.network_dropped);
    appendValue(output, counters.cqes_visible);
    appendValue(output, counters.cqes_polled);
    appendValue(output, counters.cq_overruns);
    appendValue(output, counters.sq_reclaimed_wqes);
    appendValue(output, counters.sq_high_watermark);
    appendValue(output, counters.cq_high_watermark);
}

struct ScenarioResult {
    std::vector<PostResult> posts;
    std::vector<DoorbellBatch> doorbells;
    std::vector<CompletionEntry> completions;
    WorkQueueConfig config;
    std::optional<WorkQueuePcieBinding> pcie_binding;
    WorkQueueCounters counters;
    std::vector<WqeRecord> records;
    std::vector<EvidenceEvent> evidence;
    Picoseconds jct_ps{0};
    bool fatal{false};
    std::size_t occupied_sq_entries{0};
    std::size_t completion_queue_depth{0};
    std::size_t unpublished_wqes{0};
};

std::string canonical(const ScenarioResult& result) {
    std::ostringstream output;
    appendWorkQueueConfig(output, result.config);
    appendPcieBinding(output, result.pcie_binding);
    appendValue(output, result.posts.size());
    for (const PostResult& post : result.posts) {
        appendValue(output, post.status);
        appendValue(output, post.wqe_id);
        appendValue(output, post.sq_sequence);
    }
    appendValue(output, result.doorbells.size());
    for (const DoorbellBatch& doorbell : result.doorbells) {
        appendValue(output, doorbell.batch_id);
        appendValue(output, doorbell.wqe_count);
        appendValue(output, doorbell.rung_at_ps);
        appendValue(output, doorbell.observed_at_ps);
    }
    appendValue(output, result.completions.size());
    for (const CompletionEntry& completion : result.completions) {
        appendCompletion(output, completion);
    }
    appendCounters(output, result.counters);
    appendValue(output, result.records.size());
    for (const WqeRecord& record : result.records) {
        appendRecord(output, record);
    }
    appendValue(output, result.evidence.size());
    for (const EvidenceEvent& event : result.evidence) {
        appendEvidence(output, event);
    }
    appendValue(output, result.jct_ps);
    appendValue(output, result.fatal);
    appendValue(output, result.occupied_sq_entries);
    appendValue(output, result.completion_queue_depth);
    appendValue(output, result.unpublished_wqes);
    return output.str();
}

template <typename Queue>
void postScenario(
    Queue& queue,
    std::size_t wqe_count,
    std::size_t doorbell_batch,
    ScenarioResult& result) {
    for (std::size_t index = 1; index <= wqe_count; ++index) {
        WorkRequest request;
        request.wr_id = index;
        request.flow_id = 1000 + index;
        request.flow_tag = 7;
        request.destination = 1;
        request.payload_bytes = 4096;
        request.traffic_class = 3;
        request.signaled = true;
        result.posts.push_back(queue.postSend(request, 0));
        if (index % doorbell_batch == 0 || index == wqe_count) {
            result.doorbells.push_back(queue.ringDoorbell(0));
        }
    }
}

template <typename Queue>
void collectCompletions(
    Queue& queue,
    Picoseconds now_ps,
    ScenarioResult& result) {
    const auto completions = queue.pollCompletionQueue(
        std::numeric_limits<std::size_t>::max(), now_ps);
    if (!completions.empty()) {
        result.jct_ps = now_ps;
        result.completions.insert(
            result.completions.end(),
            completions.begin(),
            completions.end());
    }
}

ScenarioResult runDirectInert(
    WorkQueueConfig config,
    std::size_t wqe_count,
    std::size_t doorbell_batch) {
    ReferenceInertPort network;
    WorkQueue queue(config, network);
    ScenarioResult result;
    postScenario(queue, wqe_count, doorbell_batch, result);
    Picoseconds now_ps = 0;
    while (queue.hasPendingPhysicalWork()) {
        const auto next = [&queue, &network]() {
            const auto queue_time = queue.nextEventTime();
            const auto network_time = network.nextEventTime();
            if (!queue_time.has_value()) {
                return network_time;
            }
            if (!network_time.has_value()) {
                return queue_time;
            }
            return std::optional<Picoseconds>(
                std::min(*queue_time, *network_time));
        }();
        if (!next.has_value() || *next < now_ps) {
            throw std::logic_error("direct inert scenario lost event order");
        }
        now_ps = *next;
        for (const NetworkEvent& event : network.takeDue(now_ps)) {
            queue.onNetworkEvent(event);
        }
        queue.progress(now_ps);
        collectCompletions(queue, now_ps, result);
    }
    queue.validateInvariants();
    result.config = queue.config();
    result.pcie_binding = queue.pcieBinding();
    result.counters = queue.counters();
    result.records = queue.records();
    result.evidence = queue.evidence();
    result.fatal = queue.fatal();
    result.occupied_sq_entries = queue.occupiedSqEntries();
    result.completion_queue_depth = queue.completionQueueDepth();
    result.unpublished_wqes = queue.unpublishedWqeCount();
    return result;
}

ScenarioResult runComposedDevice(
    RnicDevice& device,
    std::size_t wqe_count,
    std::size_t doorbell_batch) {
    ScenarioResult result;
    postScenario(device, wqe_count, doorbell_batch, result);
    Picoseconds now_ps = 0;
    while (device.hasPendingPhysicalWork()) {
        const auto next = device.nextEventTime();
        if (!next.has_value() || *next < now_ps) {
            throw std::logic_error("composed inert scenario lost event order");
        }
        now_ps = *next;
        device.progress(now_ps);
        collectCompletions(device, now_ps, result);
    }
    device.validateInvariants();
    result.config = device.workQueueConfig();
    result.pcie_binding = device.pcieBinding();
    result.counters = device.counters();
    result.records = device.records();
    result.evidence = device.evidence();
    result.fatal = device.fatal();
    result.occupied_sq_entries = device.occupiedSqEntries();
    result.completion_queue_depth = device.completionQueueDepth();
    result.unpublished_wqes = device.unpublishedWqeCount();
    return result;
}

ScenarioResult runComposedInert(
    WorkQueueConfig config,
    std::size_t wqe_count,
    std::size_t doorbell_batch,
    RnicDmaConfig dma = {}) {
    RnicDeviceConfig device_config;
    device_config.identity.qpn = config.qpn;
    device_config.identity.policy_context_token =
        config.policy_context_token;
    device_config.work_queue = config;
    device_config.dma = std::move(dma);
    RnicDevice device(device_config);
    return runComposedDevice(device, wqe_count, doorbell_batch);
}

template <typename Queue>
ScenarioResult runExternal(
    Queue& queue,
    FakeNetworkPort& network,
    const WorkQueueConfig& config,
    std::size_t wqe_count,
    std::size_t doorbell_batch) {
    ScenarioResult result;
    postScenario(queue, wqe_count, doorbell_batch, result);
    Picoseconds now_ps = 0;
    while (queue.hasPendingPhysicalWork()) {
        const auto queue_time = queue.nextEventTime();
        const auto network_time = network.nextCompletionTime();
        std::optional<Picoseconds> next;
        if (!queue_time.has_value()) {
            next = network_time;
        } else if (!network_time.has_value()) {
            next = queue_time;
        } else {
            next = std::min(*queue_time, *network_time);
        }
        if (!next.has_value() || *next < now_ps) {
            throw std::logic_error("external scenario lost event order");
        }
        now_ps = *next;
        for (const NetworkEvent& event : network.takeDue(now_ps)) {
            queue.onNetworkEvent(event);
        }
        queue.progress(now_ps);
        collectCompletions(queue, now_ps, result);
    }
    queue.validateInvariants();
    result.config = config;
    result.pcie_binding = queue.pcieBinding();
    result.counters = queue.counters();
    result.records = queue.records();
    result.evidence = queue.evidence();
    result.fatal = queue.fatal();
    result.occupied_sq_entries = queue.occupiedSqEntries();
    result.completion_queue_depth = queue.completionQueueDepth();
    result.unpublished_wqes = queue.unpublishedWqeCount();
    return result;
}

bool sameAnalyticalAccounting(
    const PcieAnalyticalDelayAccounting& lhs,
    const PcieAnalyticalDelayAccounting& rhs) {
    return lhs.draws == rhs.draws
        && lhs.occurrences == rhs.occurrences
        && lhs.tail_draws == rhs.tail_draws;
}

bool samePcieAccounting(
    const PcieClassAccounting& lhs,
    const PcieClassAccounting& rhs) {
    return lhs.transactions == rhs.transactions
        && lhs.useful_bytes == rhs.useful_bytes
        && lhs.transferred_bytes == rhs.transferred_bytes
        && lhs.host_store_bytes == rhs.host_store_bytes
        && lhs.host_to_device.tlps == rhs.host_to_device.tlps
        && lhs.host_to_device.payload_bytes
            == rhs.host_to_device.payload_bytes
        && lhs.host_to_device.overhead_bytes
            == rhs.host_to_device.overhead_bytes
        && lhs.host_to_device.modeled_link_bytes
            == rhs.host_to_device.modeled_link_bytes
        && lhs.device_to_host.tlps == rhs.device_to_host.tlps
        && lhs.device_to_host.payload_bytes
            == rhs.device_to_host.payload_bytes
        && lhs.device_to_host.overhead_bytes
            == rhs.device_to_host.overhead_bytes
        && lhs.device_to_host.modeled_link_bytes
            == rhs.device_to_host.modeled_link_bytes
        && lhs.waits.ordering_ps == rhs.waits.ordering_ps
        && lhs.waits.outstanding_read_ps == rhs.waits.outstanding_read_ps
        && lhs.waits.completion_buffer_ps == rhs.waits.completion_buffer_ps
        && lhs.waits.credit_ps == rhs.waits.credit_ps
        && lhs.waits.link_queue_ps == rhs.waits.link_queue_ps
        && lhs.service_delay.host_store_ps
            == rhs.service_delay.host_store_ps
        && lhs.service_delay.posted_write_visibility_ps
            == rhs.service_delay.posted_write_visibility_ps
        && lhs.service_delay.read_completion_ps
            == rhs.service_delay.read_completion_ps
        && lhs.path_delay.base_ps == rhs.path_delay.base_ps
        && lhs.path_delay.numa_ps == rhs.path_delay.numa_ps
        && lhs.path_delay.iommu_ps == rhs.path_delay.iommu_ps
        && lhs.path_delay.acs_ps == rhs.path_delay.acs_ps
        && lhs.path_delay.switch_ps == rhs.path_delay.switch_ps
        && lhs.path_delay.ddio_miss_ps == rhs.path_delay.ddio_miss_ps
        && lhs.path_delay.gpu_direct_ps == rhs.path_delay.gpu_direct_ps
        && sameAnalyticalAccounting(
            lhs.path_profiles.numa, rhs.path_profiles.numa)
        && sameAnalyticalAccounting(
            lhs.path_profiles.iommu, rhs.path_profiles.iommu)
        && sameAnalyticalAccounting(
            lhs.path_profiles.acs, rhs.path_profiles.acs)
        && sameAnalyticalAccounting(
            lhs.path_profiles.switch_path, rhs.path_profiles.switch_path)
        && sameAnalyticalAccounting(
            lhs.path_profiles.ddio_miss, rhs.path_profiles.ddio_miss)
        && sameAnalyticalAccounting(
            lhs.path_profiles.gpu_direct, rhs.path_profiles.gpu_direct)
        && lhs.latency_samples_used == rhs.latency_samples_used;
}

WorkQueueConfig scalarConfig(Picoseconds doorbell_service_ps) {
    WorkQueueConfig config;
    config.sq_depth = 64;
    config.cq_depth = 64;
    config.source = 3;
    config.qpn = 17;
    config.policy_context_token = 9001;
    config.doorbell_service_ps = doorbell_service_ps;
    config.wqe_fetch_service_ps = 10;
    return config;
}

RnicDeviceConfig deviceConfig(WorkQueueConfig work_queue) {
    RnicDeviceConfig config;
    config.identity.qpn = work_queue.qpn;
    config.identity.policy_context_token =
        work_queue.policy_context_token;
    config.work_queue = std::move(work_queue);
    return config;
}

struct DeviceAuditSnapshot {
    std::string wqe_records;
    std::string counters;
    std::string evidence;
    std::string port_ledger;
    std::string physical_state;
    std::string pcie_state;
};

std::string canonicalWqeRecords(const RnicDevice& device) {
    std::ostringstream output;
    appendValue(output, device.records().size());
    for (const WqeRecord& record : device.records()) {
        appendRecord(output, record);
    }
    return output.str();
}

std::string canonicalCounters(const RnicDevice& device) {
    std::ostringstream output;
    appendCounters(output, device.counters());
    return output.str();
}

std::string canonicalEvidence(const RnicDevice& device) {
    std::ostringstream output;
    appendValue(output, device.evidence().size());
    for (const EvidenceEvent& event : device.evidence()) {
        appendEvidence(output, event);
    }
    return output.str();
}

std::string canonicalPhysicalState(const RnicDevice& device) {
    std::ostringstream output;
    const RnicDeviceConfig& config = device.config();
    appendValue(output, config.version);
    appendValue(output, config.identity.version);
    appendValue(output, config.identity.qpn);
    appendValue(output, config.identity.policy_context_token);
    appendWorkQueueConfig(output, device.workQueueConfig());
    appendValue(output, config.qpc.version);
    appendValue(output, config.qpc.enabled);
    appendValue(output, config.dma.version);
    appendValue(output, config.dma.enabled);
    appendValue(output, config.dma.shared_ordering_domain_namespace);
    appendValue(output, config.network.version);
    appendValue(output, config.network.enabled);
    appendValue(output, device.stageReport().scalar_doorbell_service);
    appendValue(output, device.stageReport().scalar_wqe_fetch_service);
    appendValue(output, device.stageReport().qpc_lookup);
    appendValue(output, device.stageReport().scalar_cqe_write_service);
    appendValue(output, device.stageReport().pcie_doorbell_record);
    appendValue(output, device.stageReport().pcie_uar_doorbell);
    appendValue(output, device.stageReport().pcie_wqe_read);
    appendValue(output, device.stageReport().pcie_cqe_write);
    appendValue(output, device.stageReport().external_network);
    appendValue(output, device.stageReport().inert_network);
    appendValue(output, device.usesSharedPcieFabric());
    appendPcieBinding(output, device.pcieBinding());
    appendOptional(output, device.nextEventTime());
    appendValue(output, device.hasPendingPhysicalWork());
    appendValue(output, device.fatal());
    appendValue(output, device.occupiedSqEntries());
    appendValue(output, device.completionQueueDepth());
    appendValue(output, device.unpublishedWqeCount());
    return output.str();
}

std::string canonicalPcieState(const RnicDevice& device) {
    std::ostringstream output;
    const PcieFabric* fabric = device.pcieFabric();
    appendValue(output, fabric != nullptr);
    if (fabric != nullptr) {
        appendValue(output, fabric->generation());
    }
    return output.str();
}

DeviceAuditSnapshot captureDeviceSnapshot(
    const RnicDevice& device,
    const FakeNetworkPort& port) {
    return DeviceAuditSnapshot{
        canonicalWqeRecords(device),
        canonicalCounters(device),
        canonicalEvidence(device),
        canonicalPortState(port),
        canonicalPhysicalState(device),
        canonicalPcieState(device),
    };
}

bool sameDeviceSnapshot(
    const DeviceAuditSnapshot& lhs,
    const DeviceAuditSnapshot& rhs) {
    return lhs.wqe_records == rhs.wqe_records
        && lhs.counters == rhs.counters
        && lhs.evidence == rhs.evidence
        && lhs.port_ledger == rhs.port_ledger
        && lhs.physical_state == rhs.physical_state
        && lhs.pcie_state == rhs.pcie_state;
}

struct ExceptionObservation {
    bool threw{false};
    bool exact_logic_error{false};
    std::string type{"none"};
    std::string message;
};

template <typename Callable>
ExceptionObservation captureLogicError(Callable&& callable) {
    try {
        callable();
        return {};
    } catch (const std::logic_error& error) {
        const bool exact = typeid(error) == typeid(std::logic_error);
        return ExceptionObservation{
            true,
            exact,
            exact ? "std::logic_error" : "derived_std::logic_error",
            error.what(),
        };
    } catch (const std::exception& error) {
        return ExceptionObservation{
            true, false, "other_std::exception", error.what()};
    } catch (...) {
        return ExceptionObservation{
            true, false, "non_standard_exception", {}};
    }
}

struct ActionObservation {
    bool succeeded{false};
    std::size_t changes{0};
};

template <typename Callable>
ActionObservation captureAction(Callable&& callable) {
    try {
        return ActionObservation{true, callable()};
    } catch (...) {
        return {};
    }
}

enum class InvalidTerminalForm {
    UnknownToken,
    DuplicateToken,
    CrossWqe,
};

const char* invalidTerminalName(InvalidTerminalForm form) {
    switch (form) {
    case InvalidTerminalForm::UnknownToken:
        return "unknown_token";
    case InvalidTerminalForm::DuplicateToken:
        return "duplicate_token";
    case InvalidTerminalForm::CrossWqe:
        return "cross_wqe";
    default:
        return "invalid";
    }
}

const char* invalidTerminalMessage(InvalidTerminalForm form) {
    if (form == InvalidTerminalForm::CrossWqe) {
        return "RNIC network token/WQE mismatch";
    }
    return "unknown or duplicate RNIC network token";
}

RnicDeviceConfig back24Config() {
    WorkQueueConfig work_queue;
    work_queue.sq_depth = 8;
    work_queue.cq_depth = 8;
    work_queue.qpn = 24;
    work_queue.policy_context_token = 2400;
    RnicDeviceConfig config = deviceConfig(work_queue);
    config.network.enabled = true;
    return config;
}

RnicDeviceAttachments externalNetworkAttachments(FakeNetworkPort& port) {
    RnicDeviceAttachments attachments;
    attachments.network_port = &port;
    return attachments;
}

struct Back24Fixture {
    explicit Back24Fixture(
        InvalidTerminalForm form,
        Picoseconds future_event_time_ps)
        : port(8, 0),
          device(back24Config(), externalNetworkAttachments(port)) {
        const std::size_t wqe_count =
            form == InvalidTerminalForm::UnknownToken ? 1 : 2;
        for (std::size_t index = 0; index < wqe_count; ++index) {
            WorkRequest request;
            request.wr_id = index + 1;
            request.flow_id = 2400 + index;
            request.destination = 2;
            request.payload_bytes = 1024;
            posts.push_back(device.postSend(request, 10));
        }
        device.ringDoorbell(10);
        (void)device.progress(10);

        if (form == InvalidTerminalForm::DuplicateToken) {
            NetworkEvent first = port.take(
                *device.wqe(posts[0].wqe_id).network_token,
                simllm::rnic::NetworkEventKind::Delivered,
                10);
            device.onNetworkEvent(first);
            (void)device.progress(10);
            invalid = first;
            invalid.event_time_ps = future_event_time_ps;
            continuation = port.take(
                *device.wqe(posts[1].wqe_id).network_token,
                simllm::rnic::NetworkEventKind::Delivered,
                20);
            continuation_wqe_id = posts[1].wqe_id;
            return;
        }

        continuation = port.take(
            *device.wqe(posts[0].wqe_id).network_token,
            simllm::rnic::NetworkEventKind::Delivered,
            20);
        continuation_wqe_id = posts[0].wqe_id;
        invalid = continuation;
        invalid.event_time_ps = future_event_time_ps;
        if (form == InvalidTerminalForm::UnknownToken) {
            invalid.token = 1000000 + future_event_time_ps;
        } else {
            invalid.wqe_id = posts[1].wqe_id;
        }
    }

    FakeNetworkPort port;
    RnicDevice device;
    std::vector<PostResult> posts;
    NetworkEvent invalid;
    NetworkEvent continuation;
    WqeId continuation_wqe_id{0};
};

std::optional<std::int64_t> signedTimestampDelta(
    const std::optional<Picoseconds>& actual,
    const std::optional<Picoseconds>& control) {
    if (!actual.has_value() || !control.has_value()) {
        return std::nullopt;
    }
    if (*actual >= *control) {
        return static_cast<std::int64_t>(*actual - *control);
    }
    return -static_cast<std::int64_t>(*control - *actual);
}

void appendSignedDelta(
    std::ostream& output,
    const std::optional<std::int64_t>& delta) {
    if (delta.has_value()) {
        output << *delta;
    } else {
        output << "missing";
    }
}

void testRejectedNetworkTerminalsAreTransactional(
    TestRunner& test,
    std::ostream* study_rows = nullptr) {
    if (study_rows != nullptr) {
        *study_rows
            << "invalid_terminal,future_event_time_ps,exception_type,"
               "exception_message,exception_identity,pre_probe_zero,"
               "wqe_records_equal,counters_equal,evidence_equal,"
               "port_ledger_equal,physical_state_equal,pcie_state_equal,"
               "post_probe_zero,continuation_succeeded,"
               "continuation_surface_equal,terminal_delta_ps,"
               "cqe_delta_ps,invariants_valid\n";
    }
    for (const InvalidTerminalForm form : {
             InvalidTerminalForm::UnknownToken,
             InvalidTerminalForm::DuplicateToken,
             InvalidTerminalForm::CrossWqe,
         }) {
        for (const Picoseconds future_event_time_ps : {110ULL, 1010ULL}) {
            Back24Fixture subject(form, future_event_time_ps);
            Back24Fixture control(form, future_event_time_ps);

            const ActionObservation subject_pre_probe = captureAction(
                [&subject]() { return subject.device.progress(10); });
            const ActionObservation control_pre_probe = captureAction(
                [&control]() { return control.device.progress(10); });
            const bool pre_probe_zero = subject_pre_probe.succeeded
                && subject_pre_probe.changes == 0
                && control_pre_probe.succeeded
                && control_pre_probe.changes == 0;
            const DeviceAuditSnapshot before = captureDeviceSnapshot(
                subject.device, subject.port);
            const ExceptionObservation exception = captureLogicError(
                [&subject]() {
                    subject.device.onNetworkEvent(subject.invalid);
                });
            const DeviceAuditSnapshot after = captureDeviceSnapshot(
                subject.device, subject.port);

            const bool exception_identity = exception.threw
                && exception.exact_logic_error
                && exception.message == invalidTerminalMessage(form);
            const bool wqe_records_equal =
                before.wqe_records == after.wqe_records;
            const bool counters_equal = before.counters == after.counters;
            const bool evidence_equal = before.evidence == after.evidence;
            const bool port_ledger_equal =
                before.port_ledger == after.port_ledger;
            const bool physical_state_equal =
                before.physical_state == after.physical_state;
            const bool pcie_state_equal =
                before.pcie_state == after.pcie_state;

            bool immediate_invariants = true;
            try {
                subject.device.validateInvariants();
                control.device.validateInvariants();
            } catch (...) {
                immediate_invariants = false;
            }

            const ActionObservation post_probe = captureAction(
                [&subject]() { return subject.device.progress(10); });
            const bool post_probe_zero =
                post_probe.succeeded && post_probe.changes == 0;

            const ActionObservation subject_continuation = captureAction(
                [&subject]() {
                    subject.device.onNetworkEvent(subject.continuation);
                    return subject.device.progress(20);
                });
            const ActionObservation control_continuation = captureAction(
                [&control]() {
                    control.device.onNetworkEvent(control.continuation);
                    return control.device.progress(20);
                });
            const bool continuation_succeeded =
                subject_continuation.succeeded
                && control_continuation.succeeded;
            const DeviceAuditSnapshot subject_final = captureDeviceSnapshot(
                subject.device, subject.port);
            const DeviceAuditSnapshot control_final = captureDeviceSnapshot(
                control.device, control.port);
            const bool continuation_surface_equal = sameDeviceSnapshot(
                subject_final, control_final);

            const WqeRecord& subject_record =
                subject.device.wqe(subject.continuation_wqe_id);
            const WqeRecord& control_record =
                control.device.wqe(control.continuation_wqe_id);
            const auto terminal_delta = signedTimestampDelta(
                subject_record.timeline.network_outcome_at_ps,
                control_record.timeline.network_outcome_at_ps);
            const auto cqe_delta = signedTimestampDelta(
                subject_record.timeline.cqe_visible_at_ps,
                control_record.timeline.cqe_visible_at_ps);
            bool final_invariants = true;
            try {
                subject.device.validateInvariants();
                control.device.validateInvariants();
            } catch (...) {
                final_invariants = false;
            }
            const bool invariants_valid =
                immediate_invariants && final_invariants;

            const std::string cell = std::string(invalidTerminalName(form))
                + " at " + std::to_string(future_event_time_ps) + " ps";
            test.check(exception_identity, cell + " keeps exception identity");
            test.check(pre_probe_zero, cell + " has an inert pre-probe");
            test.check(wqe_records_equal, cell + " preserves WQE records");
            test.check(counters_equal, cell + " preserves counters");
            test.check(evidence_equal, cell + " preserves evidence");
            test.check(port_ledger_equal, cell + " preserves the port ledger");
            test.check(
                physical_state_equal, cell + " preserves physical state");
            test.check(pcie_state_equal, cell + " preserves PCIe state");
            test.check(post_probe_zero, cell + " preserves caller-clock behavior");
            test.check(
                continuation_succeeded, cell + " accepts the valid continuation");
            test.check(
                continuation_surface_equal,
                cell + " matches the paired-control continuation");
            test.check(
                terminal_delta.has_value() && *terminal_delta == 0,
                cell + " preserves the native terminal timestamp");
            test.check(
                cqe_delta.has_value() && *cqe_delta == 0,
                cell + " preserves the completion-visible timestamp");
            test.check(invariants_valid, cell + " preserves invariants");

            if (study_rows != nullptr) {
                *study_rows << invalidTerminalName(form) << ','
                            << future_event_time_ps << ','
                            << exception.type << ','
                            << exception.message << ','
                            << (exception_identity ? 1 : 0) << ','
                            << (pre_probe_zero ? 1 : 0) << ','
                            << (wqe_records_equal ? 1 : 0) << ','
                            << (counters_equal ? 1 : 0) << ','
                            << (evidence_equal ? 1 : 0) << ','
                            << (port_ledger_equal ? 1 : 0) << ','
                            << (physical_state_equal ? 1 : 0) << ','
                            << (pcie_state_equal ? 1 : 0) << ','
                            << (post_probe_zero ? 1 : 0) << ','
                            << (continuation_succeeded ? 1 : 0) << ','
                            << (continuation_surface_equal ? 1 : 0) << ',';
                appendSignedDelta(*study_rows, terminal_delta);
                *study_rows << ',';
                appendSignedDelta(*study_rows, cqe_delta);
                *study_rows << ',' << (invariants_valid ? 1 : 0) << '\n';
            }
        }
    }
}

void testScalarSweepExactEquivalence(
    TestRunner& test,
    std::ostream* study_rows = nullptr) {
    if (study_rows != nullptr) {
        *study_rows
            << "doorbell_batch,doorbell_service_ps,direct_jct_ps,"
               "composed_jct_ps,direct_doorbells,composed_doorbells,"
               "direct_cqes,composed_cqes,exact_surface_equal,"
               "closed_form_match,structural_invariants\n";
    }
    for (const Picoseconds doorbell_service_ps : {0ULL, 1000ULL}) {
        for (const std::size_t batch : {1U, 4U, 16U}) {
            const WorkQueueConfig config =
                scalarConfig(doorbell_service_ps);
            const ScenarioResult direct =
                runDirectInert(config, 32, batch);
            const ScenarioResult composed =
                runComposedInert(config, 32, batch);
            const bool exact_equal =
                canonical(direct) == canonical(composed);
            test.check(
                exact_equal,
                "scalar direct/composed surfaces are exactly equal for B="
                    + std::to_string(batch) + ", D="
                    + std::to_string(doorbell_service_ps));
            const Picoseconds expected_jct = doorbell_service_ps == 0
                ? 320
                : (32 / batch) * doorbell_service_ps + batch * 10;
            const bool closed_form_match =
                direct.jct_ps == expected_jct
                && composed.jct_ps == expected_jct
                && direct.counters.doorbells == 32 / batch
                && composed.counters.doorbells == 32 / batch
                && direct.completions.size() == 32
                && composed.completions.size() == 32;
            test.check(
                closed_form_match,
                "scalar composed row matches the frozen closed form");
            const bool structural_invariants =
                composed.counters.posted_wqes == 32
                    && composed.counters.network_delivered == 32
                    && composed.counters.sq_reclaimed_wqes == 32
                    && composed.counters.network_busy == 0
                    && composed.counters.network_rejected == 0
                    && composed.counters.network_dropped == 0
                    && composed.counters.sq_full_rejections == 0
                    && composed.counters.cq_overruns == 0
                    && composed.evidence.empty();
            test.check(
                structural_invariants,
                "scalar conservation and forced-zero invariants hold");
            if (study_rows != nullptr) {
                *study_rows
                    << batch << ',' << doorbell_service_ps << ','
                    << direct.jct_ps << ',' << composed.jct_ps << ','
                    << direct.counters.doorbells << ','
                    << composed.counters.doorbells << ','
                    << direct.completions.size() << ','
                    << composed.completions.size() << ','
                    << (exact_equal ? 1 : 0) << ','
                    << (closed_form_match ? 1 : 0) << ','
                    << (structural_invariants ? 1 : 0) << '\n';
            }
        }
    }
}

void testPcieExactEquivalence(TestRunner& test) {
    auto fabric_config = defaultPcieFabricConfig();
    fabric_config.max_payload_size_bytes = 128;
    fabric_config.max_read_request_size_bytes = 128;
    fabric_config.host_store_latency_ps.samples_ps = {10};
    fabric_config.posted_write_visibility_latency_ps.samples_ps = {20};
    fabric_config.read_completion_latency_ps.samples_ps = {30};
    fabric_config.paths[0].base_latency_ps = 5;
    fabric_config.paths[1].base_latency_ps = 7;

    WorkQueueConfig work_queue;
    work_queue.sq_depth = 8;
    work_queue.cq_depth = 8;
    work_queue.qpn = 17;
    work_queue.policy_context_token = 9001;
    work_queue.qpc_lookup_service_ps = 11;
    work_queue.scheduler_service_ps = 13;

    PcieFabric direct_fabric(fabric_config);
    FakeNetworkPort direct_network(4, 100);
    WorkQueue direct_queue(work_queue, direct_network, direct_fabric);
    const ScenarioResult direct = runExternal(
        direct_queue, direct_network, work_queue, 4, 4);

    FakeNetworkPort composed_network(4, 100);
    RnicDeviceConfig composed_config = deviceConfig(work_queue);
    composed_config.dma.enabled = true;
    composed_config.dma.fabric = fabric_config;
    composed_config.network.enabled = true;
    RnicDeviceAttachments attachments;
    attachments.network_port = &composed_network;
    RnicDevice composed_device(composed_config, attachments);
    const ScenarioResult composed = runExternal(
        composed_device, composed_network, work_queue, 4, 4);

    test.check(
        canonical(direct) == canonical(composed),
        "PCIe direct/composed queue surfaces are exactly equal");
    test.check(
        direct_fabric.generation()
                == composed_device.pcieFabric()->generation()
            && samePcieAccounting(
                direct_fabric.totalAccounting(),
                composed_device.pcieFabric()->totalAccounting()),
        "PCIe direct/composed aggregate accounting is exactly equal");
    for (std::size_t index = 0;
         index < simllm::rnic::kPcieServiceClassCount;
         ++index) {
        const auto service_class = static_cast<PcieServiceClass>(index);
        test.check(
            samePcieAccounting(
                direct_fabric.accounting(service_class),
                composed_device.pcieFabric()->accounting(service_class)),
            "PCIe direct/composed class accounting is exactly equal");
    }
    const auto& stages = composed_device.stageReport();
    test.check(
        stages.scalar_doorbell_service
                == RnicStageApplicability::NotApplicable
            && stages.scalar_wqe_fetch_service
                == RnicStageApplicability::NotApplicable
            && stages.scalar_cqe_write_service
                == RnicStageApplicability::NotApplicable
            && stages.pcie_doorbell_record
                == RnicStageApplicability::Applicable
            && stages.pcie_uar_doorbell
                == RnicStageApplicability::Applicable
            && stages.pcie_wqe_read
                == RnicStageApplicability::Applicable
            && stages.pcie_cqe_write
                == RnicStageApplicability::Applicable,
        "DMA module selects fabric stages exclusively");
}

void testInertNetworkExactEquivalence(TestRunner& test) {
    WorkQueueConfig config = scalarConfig(37);
    config.wqe_fetch_service_ps = 11;
    config.qpc_lookup_service_ps = 5;
    config.scheduler_service_ps = 7;
    config.cqe_write_service_ps = 13;
    const ScenarioResult direct = runDirectInert(config, 3, 2);
    const ScenarioResult composed = runComposedInert(config, 3, 2);
    test.check(
        canonical(direct) == canonical(composed),
        "owned inert network exactly matches the direct reference port");
    bool tokens_are_fresh = composed.records.size() == 3;
    for (std::size_t index = 0; index < composed.records.size(); ++index) {
        const WqeRecord& record = composed.records[index];
        tokens_are_fresh = tokens_are_fresh
            && record.network_token.has_value()
            && *record.network_token == index + 1
            && record.timeline.network_accepted_at_ps.has_value()
            && record.timeline.network_outcome_at_ps.has_value()
            && *record.timeline.network_accepted_at_ps
                == *record.timeline.network_outcome_at_ps
            && !record.timeline.first_packet_at_ps.has_value()
            && !record.timeline.last_packet_at_ps.has_value();
    }
    test.check(
        tokens_are_fresh,
        "inert progress returns fresh tokens without packet timestamps");

    RnicDeviceConfig device_config = deviceConfig(config);
    RnicDevice device(device_config);
    test.check(
        device.stageReport().external_network
                == RnicStageApplicability::NotApplicable
            && device.stageReport().inert_network
                == RnicStageApplicability::Applicable
            && std::string(simllm::rnic::toString(
                device.stageReport().external_network))
                == "not_applicable",
        "absent external network reports only the inert stage applicable");
}

void testDisabledModulesAndIdentity(TestRunner& test) {
    const WorkQueueConfig baseline_config = scalarConfig(1000);
    const ScenarioResult baseline =
        runComposedInert(baseline_config, 4, 2);
    simllm::rnic::RnicDmaConfig inert_dma;
    inert_dma.fabric.lane_count = 0;
    inert_dma.fabric.paths.clear();
    inert_dma.work_queue.pcie_wqe_bytes = 0;
    inert_dma.work_queue.pcie_submission_ordering_domain = 777;
    inert_dma.shared_ordering_domain_namespace = 99;
    const ScenarioResult with_inert_parameters =
        runComposedInert(baseline_config, 4, 2, inert_dma);
    test.check(
        canonical(baseline) == canonical(with_inert_parameters),
        "disabled DMA parameters are inert on the scalar path");

    WorkQueueConfig qpc_off_queue = scalarConfig(0);
    qpc_off_queue.wqe_fetch_service_ps = 0;
    RnicDeviceConfig qpc_off = deviceConfig(qpc_off_queue);
    qpc_off.qpc.enabled = false;
    qpc_off.network.enabled = true;
    CapturingPort capture;
    RnicDeviceAttachments attachments;
    attachments.network_port = &capture;
    RnicDevice device(qpc_off, attachments);
    WorkRequest request;
    request.wr_id = 1;
    request.flow_id = 88;
    request.flow_tag = 9;
    request.destination = 4;
    request.payload_bytes = 512;
    const PostResult posted = device.postSend(request, 0);
    device.ringDoorbell(0);
    device.progress(0);
    test.check(
        posted.status == PostStatus::Accepted
            && !device.wqe(posted.wqe_id)
                    .timeline.qpc_ready_at_ps.has_value()
            && device.stageReport().qpc_lookup
                == RnicStageApplicability::NotApplicable,
        "disabled QPC reports its lookup stage as not applicable");
    test.check(
        capture.descriptors.size() == 1
            && capture.descriptors.front().wqe_id == posted.wqe_id
            && capture.descriptors.front().wr_id == request.wr_id
            && capture.descriptors.front().flow_id == request.flow_id
            && capture.descriptors.front().flow_tag == request.flow_tag
            && capture.descriptors.front().qpn
                == qpc_off.identity.qpn
            && capture.descriptors.front().policy_context_token
                == qpc_off.identity.policy_context_token
            && capture.descriptors.front().source
                == qpc_off.work_queue.source
            && capture.descriptors.front().destination
                == request.destination
            && capture.descriptors.front().traffic_class
                == request.traffic_class
            && capture.descriptors.front().payload_bytes
                == request.payload_bytes
            && capture.descriptors.front().extent_index == 0
            && capture.descriptors.front().extent_count == 1
            && capture.descriptors.front().eligible_at_ps == 0
            && capture.descriptors.front().abi_version
                == simllm::rnic::kNetworkPortAbiVersion,
        "device identity and network ABI survive a disabled QPC");
    test.check(
        !device.wqe(posted.wqe_id).timeline.first_packet_at_ps.has_value()
            && !device.wqe(posted.wqe_id)
                    .timeline.last_packet_at_ps.has_value(),
        "flow admission does not fabricate packet-issue timestamps");

    NetworkEvent invalid_event;
    invalid_event.abi_version = 0;
    invalid_event.token = *device.wqe(posted.wqe_id).network_token;
    invalid_event.wqe_id = posted.wqe_id;
    invalid_event.event_time_ps = 0;
    test.expectThrowAs<std::invalid_argument>(
        [&device, &invalid_event]() { device.onNetworkEvent(invalid_event); },
        "device hard-fails a mismatched network event ABI");
    test.check(
        device.counters().network_delivered == 0
            && !device.wqe(posted.wqe_id)
                    .timeline.network_outcome_at_ps.has_value(),
        "rejected network event version mutates no WQE state");
    invalid_event.abi_version = simllm::rnic::kNetworkPortAbiVersion;
    device.onNetworkEvent(invalid_event);
    device.progress(0);
    (void)device.pollCompletionQueue(1, 0);
    device.validateInvariants();
}

void testCallerClockIsDeviceWide(TestRunner& test) {
    RnicDevice device(deviceConfig(scalarConfig(0)));
    device.progress(10);
    WorkRequest request;
    request.wr_id = 1;
    test.expectThrowAs<std::logic_error>(
        [&device, &request]() { (void)device.postSend(request, 9); },
        "device rejects a caller-clock regression across entry points");

    RnicDeviceConfig dma_config = deviceConfig(WorkQueueConfig{});
    dma_config.qpc.enabled = false;
    dma_config.dma.enabled = true;
    RnicDevice dma_device(dma_config);
    dma_device.progress(10);
    simllm::rnic::PcieTransactionRequest transaction;
    transaction.service_class = PcieServiceClass::DoorbellRecord;
    transaction.operation = simllm::rnic::PcieOperation::HostStore;
    transaction.path_id = 2;
    transaction.useful_bytes = 4;
    transaction.transfer_bytes = 4;
    transaction.submitted_at_ps = 9;
    test.expectThrowAs<std::logic_error>(
        [&dma_device, &transaction]() {
            (void)dma_device.submitPcie(transaction);
        },
        "device clock also covers standalone fabric submissions");
    test.check(
        dma_device.pcieFabric()->generation() == 0,
        "rejected regressive fabric submission mutates no fabric state");

    RnicDevice validation_device(dma_config);
    PcieTransactionRequest invalid = transaction;
    invalid.path_id = 999;
    invalid.submitted_at_ps = 100;
    test.expectThrowAs<std::invalid_argument>(
        [&validation_device, &invalid]() {
            (void)validation_device.submitPcie(invalid);
        },
        "failed fabric validation does not commit caller time");
    PcieTransactionRequest valid = invalid;
    valid.path_id = 2;
    valid.submitted_at_ps = 50;
    const auto accepted = validation_device.submitPcie(valid);
    test.check(
        accepted.submitted_at_ps == 50
            && validation_device.pcieFabric()->generation() == 1,
        "an earlier valid submission follows a later rejected request");
}

void testModuleAttachmentRejection(TestRunner& test) {
    RnicDeviceConfig config = deviceConfig(scalarConfig(0));
    auto shared_fabric =
        std::make_shared<PcieFabric>(defaultPcieFabricConfig());
    RnicDeviceAttachments fabric_attachment;
    fabric_attachment.shared_pcie_fabric = shared_fabric;
    test.expectThrowAs<std::invalid_argument>(
        [&config, &fabric_attachment]() {
            RnicDevice device(config, fabric_attachment);
            (void)device;
        },
        "DMA-disabled device rejects an external fabric");

    FakeNetworkPort network(1, 0);
    RnicDeviceAttachments network_attachment;
    network_attachment.network_port = &network;
    test.expectThrowAs<std::invalid_argument>(
        [&config, &network_attachment]() {
            RnicDevice device(config, network_attachment);
            (void)device;
        },
        "network-disabled device rejects an external port");

    config.network.enabled = true;
    test.expectThrowAs<std::invalid_argument>(
        [&config]() {
            RnicDevice device(config);
            (void)device;
        },
        "network-enabled device requires an external port");

    config.network.enabled = false;
    config.qpc.enabled = false;
    config.work_queue.qpc_lookup_service_ps = 1;
    test.expectThrowAs<std::invalid_argument>(
        [&config]() {
            RnicDevice device(config);
            (void)device;
        },
        "QPC-disabled device rejects scalar QPC service");

    RnicDeviceConfig dma_base = deviceConfig(scalarConfig(0));
    dma_base.work_queue.wqe_fetch_service_ps = 0;
    dma_base.qpc.enabled = false;
    dma_base.dma.enabled = true;
    dma_base.dma.shared_ordering_domain_namespace = 10;
    const std::uint64_t generation_before = shared_fabric->generation();
    const PcieClassAccounting accounting_before =
        shared_fabric->totalAccounting();
    RnicDeviceConfig invalid_doorbell = dma_base;
    invalid_doorbell.work_queue.doorbell_service_ps = 1;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid_doorbell, &fabric_attachment]() {
            RnicDevice device(invalid_doorbell, fabric_attachment);
            (void)device;
        },
        "DMA-enabled device rejects scalar doorbell service");
    RnicDeviceConfig invalid_fetch = dma_base;
    invalid_fetch.work_queue.wqe_fetch_service_ps = 1;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid_fetch, &fabric_attachment]() {
            RnicDevice device(invalid_fetch, fabric_attachment);
            (void)device;
        },
        "DMA-enabled device rejects scalar WQE-fetch service");
    RnicDeviceConfig invalid_cqe = dma_base;
    invalid_cqe.work_queue.cqe_write_service_ps = 1;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid_cqe, &fabric_attachment]() {
            RnicDevice device(invalid_cqe, fabric_attachment);
            (void)device;
        },
        "DMA-enabled device rejects scalar CQE-write service");
    test.check(
        shared_fabric->generation() == generation_before
            && samePcieAccounting(
                shared_fabric->totalAccounting(), accounting_before),
        "all scalar double-charge failures leave shared fabric unchanged");
}

void expectConfigVersionFailure(
    TestRunner& test,
    RnicDeviceConfig config,
    const std::string& message) {
    test.expectThrowAs<std::invalid_argument>(
        [config = std::move(config)]() mutable {
            RnicDevice device(std::move(config));
            (void)device;
        },
        message);
}

void testVersionClosure(TestRunner& test) {
    const RnicDeviceConfig valid = deviceConfig(scalarConfig(0));
    {
        RnicDevice device(valid);
        test.check(
            device.config().version == simllm::rnic::kRnicDeviceConfigVersion
                && device.config().identity.version
                    == simllm::rnic::kRnicDeviceIdentityVersion
                && device.config().qpc.version
                    == simllm::rnic::kRnicQpcConfigVersion
                && device.config().dma.version
                    == simllm::rnic::kRnicDmaConfigVersion
                && device.config().network.version
                    == simllm::rnic::kRnicNetworkConfigVersion
                && device.workQueueConfig().version
                    == simllm::rnic::kWorkQueueConfigVersion
                && device.config().dma.work_queue.version
                    == simllm::rnic::kWorkQueuePcieBindingVersion
                && device.config().dma.fabric.version
                    == simllm::rnic::kPcieFabricConfigVersion,
            "supported sub-config versions pass through unchanged");
    }

    RnicDeviceConfig invalid = valid;
    invalid.version = 0;
    expectConfigVersionFailure(test, invalid, "device version mismatch fails");
    invalid = valid;
    invalid.identity.version = 0;
    expectConfigVersionFailure(test, invalid, "identity version mismatch fails");
    invalid = valid;
    invalid.qpc.version = 0;
    expectConfigVersionFailure(test, invalid, "QPC version mismatch fails");
    invalid = valid;
    invalid.dma.version = 0;
    expectConfigVersionFailure(test, invalid, "DMA version mismatch fails");
    invalid = valid;
    invalid.network.version = 0;
    expectConfigVersionFailure(test, invalid, "network version mismatch fails");
    invalid = valid;
    invalid.work_queue.version = 0;
    expectConfigVersionFailure(
        test, invalid, "work-queue version mismatch fails");
    invalid = valid;
    invalid.dma.work_queue.version = 0;
    expectConfigVersionFailure(
        test, invalid, "disabled DMA binding version mismatch fails");
    invalid = valid;
    invalid.dma.fabric.version = 0;
    expectConfigVersionFailure(
        test, invalid, "disabled fabric version mismatch fails");
    invalid = valid;
    invalid.dma.fabric.paths.front()
        .analytical_penalties.numa.version = 0;
    expectConfigVersionFailure(
        test,
        invalid,
        "disabled analytical profile version mismatch fails");
}

RnicDeviceConfig sharedDeviceConfig(std::uint64_t name_space) {
    WorkQueueConfig work_queue;
    work_queue.qpn = static_cast<std::uint32_t>(100 + name_space);
    work_queue.policy_context_token = 1000 + name_space;
    RnicDeviceConfig config = deviceConfig(work_queue);
    config.qpc.enabled = false;
    config.dma.enabled = true;
    config.dma.shared_ordering_domain_namespace = name_space;
    return config;
}

PcieTransactionRequest hostStoreRequest(
    std::uint64_t ordering_domain,
    Picoseconds submitted_at_ps) {
    PcieTransactionRequest request;
    request.client_token = 1;
    request.service_class = PcieServiceClass::DoorbellRecord;
    request.operation = simllm::rnic::PcieOperation::HostStore;
    request.ordering = simllm::rnic::PcieOrdering::VisibilityDependency;
    request.path_id = 2;
    request.ordering_domain = ordering_domain;
    request.useful_bytes = 4;
    request.transfer_bytes = 4;
    request.submitted_at_ps = submitted_at_ps;
    return request;
}

void testSharedFabricConfigAuthority(TestRunner& test) {
    auto fabric_config = defaultPcieFabricConfig();
    fabric_config.lane_count = 2;
    fabric_config.host_store_latency_ps.samples_ps = {17};
    fabric_config.paths[1].base_latency_ps = 23;
    auto shared_fabric = std::make_shared<PcieFabric>(fabric_config);
    RnicDeviceAttachments attachments;
    attachments.shared_pcie_fabric = shared_fabric;

    RnicDeviceConfig mismatched = sharedDeviceConfig(31);
    mismatched.dma.fabric = fabric_config;
    mismatched.dma.fabric.lane_count = 0;
    mismatched.dma.fabric.paths.clear();
    const std::uint64_t generation_before = shared_fabric->generation();
    const PcieClassAccounting accounting_before =
        shared_fabric->totalAccounting();
    test.expectThrowAs<std::invalid_argument>(
        [&mismatched, &attachments]() {
            RnicDevice device(mismatched, attachments);
            (void)device;
        },
        "shared fabric rejects a mismatched embedded config");
    test.check(
        shared_fabric->generation() == generation_before
            && samePcieAccounting(
                shared_fabric->totalAccounting(), accounting_before),
        "shared config rejection mutates no fabric state");

    RnicDeviceConfig matching = sharedDeviceConfig(31);
    matching.dma.fabric = fabric_config;
    RnicDevice device(matching, attachments);
    test.check(
        device.pcieFabric() == shared_fabric.get()
            && device.config().dma.fabric.lane_count == 2
            && device.config().dma.fabric.paths.size() == 2
            && device.config().dma.fabric
                    .host_store_latency_ps.samples_ps
                == std::vector<Picoseconds>{17}
            && device.config().dma.fabric.paths[1].base_latency_ps == 23,
        "matching non-default shared config reports the effective fabric");
}

void testSharedOrderingDomainsAndLifetime(TestRunner& test) {
    auto shared_fabric =
        std::make_shared<PcieFabric>(defaultPcieFabricConfig());
    std::weak_ptr<PcieFabric> weak_fabric = shared_fabric;
    RnicDeviceAttachments attachments;
    attachments.shared_pcie_fabric = shared_fabric;
    {
        RnicDevice first(sharedDeviceConfig(10), attachments);
        RnicDevice second(sharedDeviceConfig(11), attachments);
        const auto first_binding = first.pcieBinding();
        const auto second_binding = second.pcieBinding();
        test.check(
            first_binding.has_value() && second_binding.has_value()
                && first_binding->pcie_submission_ordering_domain == 21
                && first_binding->pcie_completion_ordering_domain == 20
                && second_binding->pcie_submission_ordering_domain == 23
                && second_binding->pcie_completion_ordering_domain == 22,
            "shared devices derive distinct namespaced ordering domains");

        PcieTransactionRequest foreign_request = hostStoreRequest(
            first_binding->pcie_submission_ordering_domain, 100);
        const std::uint64_t generation_before = shared_fabric->generation();
        const PcieClassAccounting accounting_before =
            shared_fabric->totalAccounting();
        test.expectThrowAs<std::invalid_argument>(
            [&second, &foreign_request]() {
                (void)second.submitPcie(foreign_request);
            },
            "device rejects another device's claimed submission domain");
        PcieTransactionRequest foreign_completion_request = hostStoreRequest(
            first_binding->pcie_completion_ordering_domain, 110);
        test.expectThrowAs<std::invalid_argument>(
            [&second, &foreign_completion_request]() {
                (void)second.submitPcie(foreign_completion_request);
            },
            "device rejects another device's claimed completion domain");
        test.check(
            shared_fabric->generation() == generation_before
                && samePcieAccounting(
                    shared_fabric->totalAccounting(), accounting_before),
            "foreign-domain rejection mutates no fabric state");

        PcieTransactionRequest own_request = hostStoreRequest(
            second_binding->pcie_submission_ordering_domain, 50);
        const auto own_result = second.submitPcie(own_request);
        PcieTransactionRequest own_completion_request = hostStoreRequest(
            second_binding->pcie_completion_ordering_domain, 60);
        const auto own_completion_result =
            second.submitPcie(own_completion_request);
        test.check(
            own_result.ordering_domain
                    == second_binding->pcie_submission_ordering_domain
                && own_result.submitted_at_ps == 50
                && own_completion_result.ordering_domain
                    == second_binding->pcie_completion_ordering_domain
                && own_completion_result.submitted_at_ps == 60
                && shared_fabric->generation() == generation_before + 2,
            "device accepts both of its own claimed ordering domains");

        RnicDeviceConfig collision = sharedDeviceConfig(12);
        collision.dma.shared_ordering_domain_namespace = 0;
        collision.dma.work_queue.pcie_submission_ordering_domain = 21;
        collision.dma.work_queue.pcie_completion_ordering_domain = 20;
        test.expectThrowAs<std::invalid_argument>(
            [&collision, &attachments]() {
                RnicDevice device(collision, attachments);
                (void)device;
            },
            "shared fabric rejects an explicitly colliding domain pair");
        shared_fabric.reset();
        test.check(
            !weak_fabric.expired(),
            "device shared ownership keeps the external fabric alive");
    }
    attachments.shared_pcie_fabric.reset();
    test.check(
        weak_fabric.expired(),
        "external fabric lifetime ends after every device releases it");

    auto repeated_fabric =
        std::make_shared<PcieFabric>(defaultPcieFabricConfig());
    RnicDeviceAttachments repeated_attachments;
    repeated_attachments.shared_pcie_fabric = repeated_fabric;
    RnicDevice repeated(sharedDeviceConfig(10), repeated_attachments);
    test.check(
        repeated.pcieBinding()->pcie_submission_ordering_domain == 21
            && repeated.pcieBinding()->pcie_completion_ordering_domain == 20,
        "shared namespace derivation is deterministic on a fresh fabric");

    auto explicit_fabric =
        std::make_shared<PcieFabric>(defaultPcieFabricConfig());
    RnicDeviceAttachments explicit_attachments;
    explicit_attachments.shared_pcie_fabric = explicit_fabric;
    RnicDeviceConfig explicit_config = sharedDeviceConfig(12);
    explicit_config.dma.shared_ordering_domain_namespace = 0;
    explicit_config.dma.work_queue.pcie_submission_ordering_domain = 101;
    explicit_config.dma.work_queue.pcie_completion_ordering_domain = 100;
    RnicDevice explicit_device(explicit_config, explicit_attachments);
    test.check(
        explicit_device.pcieBinding()->pcie_submission_ordering_domain == 101
            && explicit_device.pcieBinding()
                    ->pcie_completion_ordering_domain == 100,
        "explicit shared ordering domains pass through unchanged");

    WorkQueueConfig timing_queue;
    timing_queue.qpn = 177;
    timing_queue.policy_context_token = 7177;
    RnicDeviceConfig owned_timing_config = deviceConfig(timing_queue);
    owned_timing_config.qpc.enabled = false;
    owned_timing_config.dma.enabled = true;
    RnicDevice owned_timing_device(owned_timing_config);
    ScenarioResult owned_timing = runComposedDevice(
        owned_timing_device, 1, 1);

    auto timing_fabric =
        std::make_shared<PcieFabric>(defaultPcieFabricConfig());
    RnicDeviceConfig shared_timing_config = owned_timing_config;
    shared_timing_config.dma.shared_ordering_domain_namespace = 77;
    RnicDeviceAttachments timing_attachments;
    timing_attachments.shared_pcie_fabric = timing_fabric;
    RnicDevice shared_timing_device(
        shared_timing_config, timing_attachments);
    ScenarioResult shared_timing = runComposedDevice(
        shared_timing_device, 1, 1);
    owned_timing.pcie_binding->pcie_submission_ordering_domain = 0;
    owned_timing.pcie_binding->pcie_completion_ordering_domain = 0;
    shared_timing.pcie_binding->pcie_submission_ordering_domain = 0;
    shared_timing.pcie_binding->pcie_completion_ordering_domain = 0;
    test.check(
        canonical(owned_timing) == canonical(shared_timing)
            && samePcieAccounting(
                owned_timing_device.pcieFabric()->totalAccounting(),
                shared_timing_device.pcieFabric()->totalAccounting()),
        "shared namespacing changes domains only without contention");

    RnicDeviceConfig owned_config = sharedDeviceConfig(10);
    owned_config.dma.shared_ordering_domain_namespace = 0;
    owned_config.work_queue.qpn = owned_config.identity.qpn;
    RnicDevice owned(owned_config);
    const auto owned_binding = owned.pcieBinding();
    test.check(
        owned_binding.has_value()
            && owned_binding->pcie_submission_ordering_domain
                == ((owned_config.work_queue.sq_id << 1) | 1)
            && owned_binding->pcie_completion_ordering_domain
                == (owned_config.work_queue.cq_id << 1),
        "owned fabric preserves SQ/CQ-derived baseline domains");
}

void testComposedPcieOperationAtomicity(TestRunner& test) {
    auto fabric_config = defaultPcieFabricConfig();
    fabric_config.host_store_latency_ps.samples_ps = {
        static_cast<Picoseconds>(std::numeric_limits<std::int64_t>::max())};
    fabric_config.paths[1].base_latency_ps =
        static_cast<Picoseconds>(std::numeric_limits<std::int64_t>::max());
    auto fabric = std::make_shared<PcieFabric>(fabric_config);
    RnicDeviceConfig config = sharedDeviceConfig(55);
    config.dma.fabric = fabric_config;
    RnicDeviceAttachments attachments;
    attachments.shared_pcie_fabric = fabric;
    RnicDevice device(config, attachments);
    WorkRequest request;
    request.wr_id = 1;
    const PostResult posted = device.postSend(request, 0);
    const std::uint64_t generation_before = fabric->generation();
    const PcieClassAccounting accounting_before = fabric->totalAccounting();
    test.expectThrowAs<std::overflow_error>(
        [&device]() { (void)device.ringDoorbell(0); },
        "composed PCIe doorbell failure preserves plan atomicity");
    test.check(
        posted.status == PostStatus::Accepted
            && device.unpublishedWqeCount() == 1
            && device.occupiedSqEntries() == 1
            && device.wqe(posted.wqe_id).state
                == simllm::rnic::WqeState::Posted
            && device.counters().doorbells == 0
            && fabric->generation() == generation_before
            && samePcieAccounting(
                fabric->totalAccounting(), accounting_before),
        "failed composed plan mutates neither queue nor fabric");
    device.validateInvariants();
}

void testInertDeliveryFailureRetainsEvent(TestRunner& test) {
    WorkQueueConfig work_queue = scalarConfig(0);
    work_queue.wqe_fetch_service_ps = 0;
    work_queue.cqe_write_service_ps = 1;
    RnicDevice device(deviceConfig(work_queue));
    WorkRequest request;
    request.wr_id = 1;
    const PostResult posted = device.postSend(request, 0);
    device.ringDoorbell(0);
    const Picoseconds near_overflow =
        std::numeric_limits<Picoseconds>::max();
    test.expectThrowAs<std::overflow_error>(
        [&device, near_overflow]() { device.progress(near_overflow); },
        "failed inert delivery keeps its event pending transactionally");
    test.check(
        posted.status == PostStatus::Accepted
            && device.wqe(posted.wqe_id).state
                == simllm::rnic::WqeState::InFlight
            && !device.wqe(posted.wqe_id)
                    .timeline.network_outcome_at_ps.has_value()
            && device.counters().network_accepted == 1
            && device.counters().network_delivered == 0
            && device.nextEventTime() == near_overflow,
        "failed inert completion is neither consumed nor dark-dropped");
    device.validateInvariants();
}

void testFailedSharedConstructionReleasesClaim(TestRunner& test) {
    auto shared_fabric =
        std::make_shared<PcieFabric>(defaultPcieFabricConfig());
    RnicDeviceAttachments attachments;
    attachments.shared_pcie_fabric = shared_fabric;
    RnicDeviceConfig invalid = sharedDeviceConfig(44);
    invalid.dma.work_queue.pcie_uar_path_id = 2;
    const std::uint64_t generation_before = shared_fabric->generation();
    const PcieClassAccounting accounting_before =
        shared_fabric->totalAccounting();
    test.expectThrowAs<std::invalid_argument>(
        [&invalid, &attachments]() {
            RnicDevice device(invalid, attachments);
            (void)device;
        },
        "invalid shared binding fails after transactional domain claim");
    test.check(
        shared_fabric->generation() == generation_before
            && samePcieAccounting(
                shared_fabric->totalAccounting(), accounting_before),
        "failed shared construction mutates no fabric transaction state");

    RnicDevice valid(sharedDeviceConfig(44), attachments);
    test.check(
        valid.pcieBinding()->pcie_submission_ordering_domain == 89
            && valid.pcieBinding()->pcie_completion_ordering_domain == 88,
        "failed construction releases its domain claim for retry");
}

}  // namespace

int main(int argc, char** argv) {
    TestRunner test;
    if (argc == 2 && std::string(argv[1]) == "--scalar-csv") {
        testScalarSweepExactEquivalence(test, &std::cout);
        return test.failures() == 0 ? 0 : 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--back24-csv") {
        testRejectedNetworkTerminalsAreTransactional(test, &std::cout);
        return test.failures() == 0 ? 0 : 1;
    }
    if (argc != 1) {
        std::cerr
            << "usage: simllm_rnic_device_test "
               "[--scalar-csv|--back24-csv]\n";
        return 2;
    }
    testScalarSweepExactEquivalence(test);
    testPcieExactEquivalence(test);
    testInertNetworkExactEquivalence(test);
    testDisabledModulesAndIdentity(test);
    testCallerClockIsDeviceWide(test);
    testModuleAttachmentRejection(test);
    testVersionClosure(test);
    testSharedOrderingDomainsAndLifetime(test);
    testSharedFabricConfigAuthority(test);
    testFailedSharedConstructionReleasesClaim(test);
    testComposedPcieOperationAtomicity(test);
    testInertDeliveryFailureRetainsEvent(test);
    testRejectedNetworkTerminalsAreTransactional(test);
    const int session_record_failures =
        runRnicSessionRecordChecks();
    test.check(
        session_record_failures == 0,
        "embedded RNIC session-record checks pass");
    if (test.failures() != 0) {
        std::cerr << test.failures() << " RNIC device check(s) failed\n";
        return 1;
    }
    std::cout << "all RNIC device composition checks passed\n";
    return 0;
}
