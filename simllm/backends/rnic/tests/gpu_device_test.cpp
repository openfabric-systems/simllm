#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "simllm/rnic/gpu_device.h"
#include "simllm/rnic/rnic_device.h"

namespace {

using simllm::rnic::CompletionEntry;
using simllm::rnic::GpuDevice;
using simllm::rnic::GpuDeviceAttachments;
using simllm::rnic::GpuDeviceConfig;
using simllm::rnic::GpuFabricTransfer;
using simllm::rnic::GpuFabricTransferRecord;
using simllm::rnic::HostMemoryAccessRecord;
using simllm::rnic::HostMemoryAllocation;
using simllm::rnic::HostMemoryObjectKind;
using simllm::rnic::HostMemoryOwnerKind;
using simllm::rnic::PcieClassAccounting;
using simllm::rnic::PcieDeviceKind;
using simllm::rnic::PcieDirection;
using simllm::rnic::PcieDirectionAccounting;
using simllm::rnic::PcieEndpointAccounting;
using simllm::rnic::PcieEndpointAttribution;
using simllm::rnic::PcieEndpointId;
using simllm::rnic::PcieEndpointLifecycle;
using simllm::rnic::PcieEndpointKind;
using simllm::rnic::PcieFabric;
using simllm::rnic::PcieFabricConfig;
using simllm::rnic::PcieOperation;
using simllm::rnic::PcieOrdering;
using simllm::rnic::PciePathConfig;
using simllm::rnic::PcieServiceClass;
using simllm::rnic::PcieTransactionRequest;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceAttachments;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::VirtualHostMemory;
using simllm::rnic::VirtualHostMemoryConfig;
using simllm::rnic::WorkRequest;
using simllm::rnic::WorkRequestDataMemory;
using simllm::rnic::WqeOpcode;
using simllm::rnic::WqeTimeline;
using simllm::rnic::kPcieServiceClassCount;
using simllm::rnic::toString;

constexpr PcieEndpointId kHostEndpoint = 4000;
constexpr PcieEndpointId kNicEndpoint = 4001;
constexpr PcieEndpointId kGpuEndpoint = 4002;
constexpr std::uint64_t kNicOwner = 920;
constexpr std::uint64_t kGpuOwner = 930;
constexpr std::uint64_t kGpuOrderingDomain = 31;
constexpr std::uint64_t kSharedNamespace = 10;
constexpr std::uint32_t kQpn = 19;
constexpr std::uint64_t kSqId = 201;
constexpr std::uint64_t kRqId = 202;
constexpr std::uint64_t kCqId = 203;
constexpr std::uint32_t kNicMkey = 177;
constexpr std::uint32_t kPeerMkey = 511;
constexpr std::uint64_t kQpcAllocation = 21;
constexpr std::uint64_t kSqAllocation = 22;
constexpr std::uint64_t kRqAllocation = 23;
constexpr std::uint64_t kCqAllocation = 24;
constexpr std::uint64_t kDoorbellAllocation = 25;
constexpr std::uint64_t kNicDataAllocation = 26;
constexpr std::uint64_t kPeerAllocation = 40;
constexpr PcieEndpointId kPeerGpuEndpoint = 4003;
constexpr std::uint64_t kPeerGpuOwner = 940;
constexpr std::uint64_t kPeerGpuAllocation = 41;
constexpr std::uint32_t kRnicRequester = 9100;
constexpr std::uint32_t kGpuRequester = 9200;
constexpr std::uint64_t kPeerRegionBytes = 65536;
constexpr std::uint32_t kGpuPathId = 3;
constexpr std::uint32_t kHostPathId = 2;

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
            check(false, message + "; wrong exception: " + error.what());
        } catch (...) {
            check(false, message + "; wrong non-standard exception");
        }
    }

    int failures() const noexcept { return failures_; }

private:
    int failures_{0};
};

enum class Arm {
    HostBounce,
    GpuDirect,
};

const char* armName(Arm arm) {
    return arm == Arm::HostBounce ? "host_bounce" : "gpu_direct";
}

PcieFabricConfig fabricConfig(
    std::uint16_t lane_count,
    PcieEndpointId host_endpoint_id) {
    PcieFabricConfig config = simllm::rnic::defaultPcieFabricConfig();
    config.lane_count = lane_count;
    config.host_endpoint_id = host_endpoint_id;
    PciePathConfig gpu_path;
    gpu_path.path_id = kGpuPathId;
    gpu_path.endpoint = PcieEndpointKind::GpuMemory;
    config.paths.push_back(gpu_path);
    return config;
}

HostMemoryAllocation makeAllocation(
    std::uint64_t allocation_id,
    std::uint64_t device_owner_id,
    HostMemoryObjectKind object_kind,
    HostMemoryOwnerKind owner_kind,
    std::uint64_t owner_id,
    PcieEndpointKind endpoint,
    std::uint32_t path_id,
    std::uint64_t virtual_base,
    std::uint64_t physical_base,
    std::uint64_t slot,
    std::size_t page_count,
    std::uint64_t length_bytes,
    std::optional<std::uint32_t> mkey = std::nullopt) {
    constexpr std::uint64_t page_size = 4096;
    HostMemoryAllocation allocation;
    allocation.allocation_id = allocation_id;
    allocation.device_owner_id = device_owner_id;
    allocation.object_kind = object_kind;
    allocation.owner_kind = owner_kind;
    allocation.owner_id = owner_id;
    allocation.endpoint = endpoint;
    allocation.path_id = path_id;
    allocation.virtual_address = virtual_base + slot * 16 * page_size;
    allocation.length_bytes = length_bytes;
    allocation.pages.page_size_bytes = page_size;
    for (std::size_t page = 0; page < page_count; ++page) {
        allocation.pages.physical_page_addresses.push_back(
            physical_base + (slot * 16 + page) * page_size);
    }
    allocation.mkey = mkey;
    return allocation;
}

RnicDeviceConfig nicConfig(
    std::uint16_t lane_count,
    PcieEndpointId host_endpoint_id,
    PcieEndpointId nic_endpoint_id) {
    constexpr std::uint64_t virtual_base = UINT64_C(0x300000000);
    constexpr std::uint64_t physical_base = UINT64_C(0x400000000);
    RnicDeviceConfig config;
    config.identity.qpn = kQpn;
    config.identity.policy_context_token = 2900;
    config.work_queue.sq_id = kSqId;
    config.work_queue.cq_id = kCqId;
    config.work_queue.source = 7;
    config.work_queue.qpn = kQpn;
    config.work_queue.policy_context_token = 2900;
    config.work_queue.sq_depth = 16;
    config.work_queue.cq_depth = 16;
    config.dma.enabled = true;
    config.dma.fabric = fabricConfig(lane_count, host_endpoint_id);
    config.dma.shared_ordering_domain_namespace = kSharedNamespace;
    config.dma.fabric_endpoint_id = nic_endpoint_id;
    config.host_memory.enabled = true;
    config.host_memory.device_owner_id = kNicOwner;
    config.host_memory.work_queue.qpc_icm_allocation_id = kQpcAllocation;
    config.host_memory.work_queue.sq_ring_allocation_id = kSqAllocation;
    config.host_memory.work_queue.rq_ring_allocation_id = kRqAllocation;
    config.host_memory.work_queue.cq_ring_allocation_id = kCqAllocation;
    config.host_memory.work_queue.doorbell_record_allocation_id =
        kDoorbellAllocation;
    config.host_memory.allocations = {
        makeAllocation(
            kQpcAllocation,
            kNicOwner,
            HostMemoryObjectKind::QpcIcm,
            HostMemoryOwnerKind::QueuePair,
            kQpn,
            PcieEndpointKind::HostPinnedMemory,
            kHostPathId,
            virtual_base,
            physical_base,
            1,
            1,
            256),
        makeAllocation(
            kSqAllocation,
            kNicOwner,
            HostMemoryObjectKind::SqRing,
            HostMemoryOwnerKind::SendQueue,
            kSqId,
            PcieEndpointKind::HostPinnedMemory,
            kHostPathId,
            virtual_base,
            physical_base,
            2,
            1,
            16 * 64),
        makeAllocation(
            kRqAllocation,
            kNicOwner,
            HostMemoryObjectKind::RqRing,
            HostMemoryOwnerKind::ReceiveQueue,
            kRqId,
            PcieEndpointKind::HostPinnedMemory,
            kHostPathId,
            virtual_base,
            physical_base,
            3,
            1,
            16 * 64),
        makeAllocation(
            kCqAllocation,
            kNicOwner,
            HostMemoryObjectKind::CqRing,
            HostMemoryOwnerKind::CompletionQueue,
            kCqId,
            PcieEndpointKind::HostPinnedMemory,
            kHostPathId,
            virtual_base,
            physical_base,
            4,
            1,
            16 * 64),
        makeAllocation(
            kDoorbellAllocation,
            kNicOwner,
            HostMemoryObjectKind::DoorbellRecord,
            HostMemoryOwnerKind::SendQueue,
            kSqId,
            PcieEndpointKind::HostPinnedMemory,
            kHostPathId,
            virtual_base,
            physical_base,
            5,
            1,
            4),
        makeAllocation(
            kNicDataAllocation,
            kNicOwner,
            HostMemoryObjectKind::DataRegion,
            HostMemoryOwnerKind::MemoryRegion,
            kNicMkey,
            PcieEndpointKind::HostPinnedMemory,
            kHostPathId,
            virtual_base,
            physical_base,
            6,
            16,
            kPeerRegionBytes,
            kNicMkey),
    };
    config.submission.producer_id = 7101;
    config.submission.cq_consumer_id = 8101;
    config.submission.rnic_requester_id = kRnicRequester;
    return config;
}

GpuDeviceConfig gpuConfig(
    std::uint16_t lane_count,
    Arm arm,
    bool grant_nic) {
    constexpr std::uint64_t virtual_base = UINT64_C(0x500000000);
    constexpr std::uint64_t physical_base = UINT64_C(0x600000000);
    const bool direct = arm == Arm::GpuDirect;
    GpuDeviceConfig config;
    config.identity.gpu_id = 1;
    config.identity.requester_id = kGpuRequester;
    config.fabric.enabled = true;
    config.fabric.fabric = fabricConfig(lane_count, kHostEndpoint);
    config.fabric.endpoint_id = kGpuEndpoint;
    config.fabric.ordering_domain = kGpuOrderingDomain;
    config.memory.enabled = true;
    config.memory.device_owner_id = kGpuOwner;
    config.memory.allocations = {
        makeAllocation(
            kPeerAllocation,
            kGpuOwner,
            HostMemoryObjectKind::DataRegion,
            HostMemoryOwnerKind::MemoryRegion,
            kPeerMkey,
            direct ? PcieEndpointKind::GpuMemory
                   : PcieEndpointKind::HostPinnedMemory,
            direct ? kGpuPathId : kHostPathId,
            virtual_base,
            physical_base,
            1,
            16,
            kPeerRegionBytes,
            kPeerMkey),
    };
    if (grant_nic) {
        config.memory.peer_read_grants = {kNicOwner};
    }
    return config;
}

WorkRequest peerWorkRequest(std::uint64_t payload_bytes) {
    WorkRequest request;
    request.wr_id = 7001;
    request.flow_id = 11;
    request.flow_tag = 3;
    request.destination = 1;
    request.payload_bytes = payload_bytes;
    request.opcode = WqeOpcode::Send;
    request.signaled = true;
    WorkRequestDataMemory data;
    data.allocation_id = kPeerAllocation;
    data.mkey = kPeerMkey;
    data.allocation_offset_bytes = 0;
    data.peer_device_owner_id = kGpuOwner;
    request.data_memory = data;
    return request;
}

WorkRequest ownWorkRequest(std::uint64_t payload_bytes) {
    WorkRequest request = peerWorkRequest(payload_bytes);
    WorkRequestDataMemory data;
    data.allocation_id = kNicDataAllocation;
    data.mkey = kNicMkey;
    data.allocation_offset_bytes = 0;
    request.data_memory = data;
    return request;
}

GpuFabricTransfer stagingTransfer(std::uint64_t payload_bytes) {
    GpuFabricTransfer transfer;
    transfer.allocation_id = kPeerAllocation;
    transfer.client_token = 91;
    transfer.service_class = PcieServiceClass::PayloadWrite;
    transfer.operation = PcieOperation::PostedWrite;
    transfer.ordering = PcieOrdering::Independent;
    transfer.allocation_offset_bytes = 0;
    transfer.transfer_bytes = payload_bytes;
    transfer.submitted_at_ps = 0;
    return transfer;
}

bool sameDirection(
    const PcieDirectionAccounting& lhs,
    const PcieDirectionAccounting& rhs) {
    return lhs.tlps == rhs.tlps && lhs.payload_bytes == rhs.payload_bytes
        && lhs.overhead_bytes == rhs.overhead_bytes
        && lhs.modeled_link_bytes == rhs.modeled_link_bytes;
}

bool sameClassAccounting(
    const PcieClassAccounting& lhs,
    const PcieClassAccounting& rhs) {
    return lhs.transactions == rhs.transactions
        && lhs.useful_bytes == rhs.useful_bytes
        && lhs.transferred_bytes == rhs.transferred_bytes
        && lhs.host_store_bytes == rhs.host_store_bytes
        && sameDirection(lhs.host_to_device, rhs.host_to_device)
        && sameDirection(lhs.device_to_host, rhs.device_to_host)
        && lhs.waits.ordering_ps == rhs.waits.ordering_ps
        && lhs.waits.outstanding_read_ps == rhs.waits.outstanding_read_ps
        && lhs.waits.completion_buffer_ps == rhs.waits.completion_buffer_ps
        && lhs.waits.credit_ps == rhs.waits.credit_ps
        && lhs.waits.link_queue_ps == rhs.waits.link_queue_ps
        && lhs.service_delay.host_store_ps == rhs.service_delay.host_store_ps
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
        && lhs.latency_samples_used == rhs.latency_samples_used;
}

bool sameTimeline(const WqeTimeline& lhs, const WqeTimeline& rhs) {
    return lhs.posted_at_ps == rhs.posted_at_ps
        && lhs.doorbelled_at_ps == rhs.doorbelled_at_ps
        && lhs.doorbell_seen_at_ps == rhs.doorbell_seen_at_ps
        && lhs.wqe_fetch_begin_at_ps == rhs.wqe_fetch_begin_at_ps
        && lhs.wqe_fetch_end_at_ps == rhs.wqe_fetch_end_at_ps
        && lhs.qpc_ready_at_ps == rhs.qpc_ready_at_ps
        && lhs.admitted_at_ps == rhs.admitted_at_ps
        && lhs.network_accepted_at_ps == rhs.network_accepted_at_ps
        && lhs.network_outcome_at_ps == rhs.network_outcome_at_ps
        && lhs.transport_retired_at_ps == rhs.transport_retired_at_ps
        && lhs.cqe_visible_at_ps == rhs.cqe_visible_at_ps
        && lhs.polled_at_ps == rhs.polled_at_ps
        && lhs.sq_reclaimed_at_ps == rhs.sq_reclaimed_at_ps;
}

struct FabricLedger {
    std::vector<PcieClassAccounting> classes;
    PcieClassAccounting total;
    std::uint64_t generation{0};
};

FabricLedger readLedger(const PcieFabric& fabric) {
    FabricLedger ledger;
    ledger.classes.reserve(kPcieServiceClassCount);
    for (std::size_t index = 0; index < kPcieServiceClassCount; ++index) {
        ledger.classes.push_back(
            fabric.accounting(static_cast<PcieServiceClass>(index)));
    }
    ledger.total = fabric.totalAccounting();
    ledger.generation = fabric.generation();
    return ledger;
}

bool sameLedger(const FabricLedger& lhs, const FabricLedger& rhs) {
    if (lhs.classes.size() != rhs.classes.size()
        || !sameClassAccounting(lhs.total, rhs.total)) {
        return false;
    }
    for (std::size_t index = 0; index < lhs.classes.size(); ++index) {
        if (!sameClassAccounting(lhs.classes[index], rhs.classes[index])) {
            return false;
        }
    }
    return true;
}

// Drives one device to quiescence from the caller clock it was posted on.
Picoseconds driveToQuiescence(RnicDevice& device, Picoseconds now_ps) {
    Picoseconds now = now_ps;
    for (std::size_t guard = 0; guard < 256; ++guard) {
        device.progress(now);
        const std::optional<Picoseconds> next = device.nextEventTime();
        if (!next.has_value()) {
            return now;
        }
        now = std::max(now, *next);
    }
    throw std::logic_error("RNIC device did not quiesce");
}

struct BaselineCell {
    WqeTimeline timeline;
    FabricLedger ledger;
    std::uint64_t attributed_requester{0};
    std::uint64_t attributed_completer{0};
};

// One NIC-only cell over its own host-pinned region. The three switches are
// exactly the ones that must not move a timestamp: naming the fabric host
// endpoint, giving the NIC an endpoint identity, and attaching an idle GPU.
BaselineCell runBaselineCell(
    PcieEndpointId host_endpoint_id,
    PcieEndpointId nic_endpoint_id,
    bool attach_idle_gpu) {
    const std::uint16_t lane_count = 16;
    auto fabric = std::make_shared<PcieFabric>(
        fabricConfig(lane_count, host_endpoint_id));
    auto registry = std::make_shared<VirtualHostMemory>(
        VirtualHostMemoryConfig{});
    std::optional<GpuDevice> gpu;
    if (attach_idle_gpu) {
        GpuDeviceAttachments gpu_attachments;
        gpu_attachments.shared_pcie_fabric = fabric;
        gpu_attachments.shared_memory = registry;
        gpu.emplace(
            gpuConfig(lane_count, Arm::GpuDirect, true), gpu_attachments);
    }
    RnicDeviceAttachments nic_attachments;
    nic_attachments.shared_pcie_fabric = fabric;
    nic_attachments.shared_host_memory = registry;
    RnicDevice nic(
        nicConfig(lane_count, host_endpoint_id, nic_endpoint_id),
        nic_attachments);
    nic.postSend(ownWorkRequest(4096), 0);
    nic.ringDoorbell(0);
    const Picoseconds quiesced = driveToQuiescence(nic, 0);
    const std::vector<CompletionEntry> polled =
        nic.pollCompletionQueue(4, quiesced);
    if (polled.size() != 1) {
        throw std::logic_error("baseline cell did not complete one WQE");
    }
    nic.validateInvariants();
    if (gpu.has_value()) {
        gpu->validateInvariants();
    }
    BaselineCell cell;
    cell.timeline = nic.wqe(polled.front().wqe_id).timeline;
    cell.ledger = readLedger(*fabric);
    cell.attributed_requester = fabric->attributedRequesterTransactions();
    cell.attributed_completer = fabric->attributedCompleterTransactions();
    return cell;
}

struct StudyCell {
    Arm arm{Arm::HostBounce};
    std::uint64_t payload_bytes{0};
    std::uint16_t lane_count{0};
    PcieEndpointKind data_endpoint{PcieEndpointKind::HostPinnedMemory};
    PcieEndpointId payload_completer{0};
    PcieDeviceKind payload_completer_kind{PcieDeviceKind::Host};
    std::size_t staging_transfers{0};
    Picoseconds staging_completed_ps{0};
    std::uint64_t staging_link_bytes{0};
    Picoseconds staging_credit_wait_ps{0};
    Picoseconds staging_link_queue_wait_ps{0};
    Picoseconds wqe_admitted_ps{0};
    Picoseconds wqe_cqe_visible_ps{0};
    std::uint64_t payload_read_transactions{0};
    PcieEndpointAccounting gpu;
    PcieEndpointAccounting nic;
    PcieEndpointAccounting host;
    std::uint64_t attributed_requester{0};
    std::uint64_t attributed_completer{0};
    bool invariants_valid{false};
};

// The GPU-attached leg. In the host-bounce arm the GPU first stages the payload
// into its own host-pinned region across the fabric, and the WQE cannot be
// posted before that data has arrived. In the GPU-direct arm the payload never
// crosses the fabric before the NIC reads it out of GPU memory.
StudyCell runStudyCell(
    Arm arm,
    std::uint64_t payload_bytes,
    std::uint16_t lane_count) {
    auto fabric = std::make_shared<PcieFabric>(
        fabricConfig(lane_count, kHostEndpoint));
    auto registry = std::make_shared<VirtualHostMemory>(
        VirtualHostMemoryConfig{});
    GpuDeviceAttachments gpu_attachments;
    gpu_attachments.shared_pcie_fabric = fabric;
    gpu_attachments.shared_memory = registry;
    GpuDevice gpu(gpuConfig(lane_count, arm, true), gpu_attachments);
    RnicDeviceAttachments nic_attachments;
    nic_attachments.shared_pcie_fabric = fabric;
    nic_attachments.shared_host_memory = registry;
    RnicDevice nic(
        nicConfig(lane_count, kHostEndpoint, kNicEndpoint), nic_attachments);

    StudyCell cell;
    cell.arm = arm;
    cell.payload_bytes = payload_bytes;
    cell.lane_count = lane_count;
    cell.data_endpoint = gpu.config().memory.allocations.front().endpoint;

    Picoseconds post_at_ps = 0;
    if (arm == Arm::HostBounce) {
        const GpuFabricTransferRecord staged =
            gpu.transfer(stagingTransfer(payload_bytes));
        cell.staging_transfers = 1;
        cell.staging_completed_ps = staged.completed_at_ps;
        cell.staging_link_bytes = staged.modeled_link_bytes;
        cell.staging_credit_wait_ps = staged.credit_wait_ps;
        cell.staging_link_queue_wait_ps = staged.link_queue_wait_ps;
        post_at_ps = staged.completed_at_ps;
    }

    nic.postSend(peerWorkRequest(payload_bytes), post_at_ps);
    nic.ringDoorbell(post_at_ps);
    const Picoseconds quiesced = driveToQuiescence(nic, post_at_ps);
    const std::vector<CompletionEntry> polled =
        nic.pollCompletionQueue(4, quiesced);
    if (polled.size() != 1) {
        throw std::logic_error("study cell did not complete one WQE");
    }
    const WqeTimeline& timeline = nic.wqe(polled.front().wqe_id).timeline;
    cell.wqe_admitted_ps = timeline.admitted_at_ps.value_or(0);
    cell.wqe_cqe_visible_ps = timeline.cqe_visible_at_ps.value_or(0);
    for (const HostMemoryAccessRecord& access : nic.memoryAccesses()) {
        if (access.object_kind == HostMemoryObjectKind::DataRegion) {
            cell.payload_completer = access.completer_endpoint_id;
        }
    }
    cell.payload_completer_kind =
        fabric->attachedEndpointKind(cell.payload_completer)
            .value_or(PcieDeviceKind::Host);
    cell.payload_read_transactions =
        fabric->accounting(PcieServiceClass::PayloadRead).transactions;
    cell.gpu = fabric->endpointAccounting(kGpuEndpoint);
    cell.nic = fabric->endpointAccounting(kNicEndpoint);
    cell.host = fabric->endpointAccounting(kHostEndpoint);
    cell.attributed_requester = fabric->attributedRequesterTransactions();
    cell.attributed_completer = fabric->attributedCompleterTransactions();

    nic.validateInvariants();
    gpu.validateInvariants();
    fabric->validateInvariants();
    registry->validateInvariants();
    cell.invariants_valid = true;
    return cell;
}

const std::vector<std::uint64_t>& studyPayloads() {
    static const std::vector<std::uint64_t> payloads{4096, 16384};
    return payloads;
}

const std::vector<std::uint16_t>& studyLaneCounts() {
    static const std::vector<std::uint16_t> lanes{8, 16};
    return lanes;
}

void testOffPathEquivalence(TestRunner& test) {
    const BaselineCell unnamed_host = runBaselineCell(0, 0, false);
    const BaselineCell named_host = runBaselineCell(kHostEndpoint, 0, false);
    const BaselineCell attributed = runBaselineCell(
        kHostEndpoint, kNicEndpoint, false);
    const BaselineCell idle_gpu = runBaselineCell(
        kHostEndpoint, 0, true);

    test.check(
        sameTimeline(unnamed_host.timeline, named_host.timeline)
            && sameLedger(unnamed_host.ledger, named_host.ledger)
            && named_host.attributed_requester == 0
            && named_host.attributed_completer == 0,
        "naming a fabric host endpoint alone moves no timestamp or charge");
    test.check(
        sameTimeline(unnamed_host.timeline, attributed.timeline)
            && sameLedger(unnamed_host.ledger, attributed.ledger),
        "endpoint attribution preserves every timestamp and class charge");
    // Eight transactions cross the link in this fixture: the WQE read and its
    // queue-page-list read, the QPC read, the payload read with its MPT and MTT
    // reads, and the CQE write with its queue-page-list read. The doorbell
    // record and the UAR doorbell are host stores and stay unattributed.
    test.check(
        attributed.attributed_requester == 8
            && attributed.attributed_completer == 8,
        "attribution charges exactly the eight link-crossing transactions");
    test.check(
        sameTimeline(unnamed_host.timeline, idle_gpu.timeline)
            && sameLedger(unnamed_host.ledger, idle_gpu.ledger)
            && idle_gpu.attributed_requester == 0,
        "an idle second device on the shared fabric changes nothing");
}

void testFabricEndpointClaims(TestRunner& test) {
    auto fabric = std::make_shared<PcieFabric>(fabricConfig(16, kHostEndpoint));
    auto registry = std::make_shared<VirtualHostMemory>(
        VirtualHostMemoryConfig{});
    GpuDeviceAttachments gpu_attachments;
    gpu_attachments.shared_pcie_fabric = fabric;
    gpu_attachments.shared_memory = registry;
    GpuDevice gpu(gpuConfig(16, Arm::GpuDirect, true), gpu_attachments);

    test.check(
        fabric->attachedEndpoints()
                == std::vector<PcieEndpointId>{kHostEndpoint, kGpuEndpoint}
            && fabric->attachedEndpointKind(kHostEndpoint)
                == PcieDeviceKind::Host
            && fabric->attachedEndpointKind(kGpuEndpoint)
                == PcieDeviceKind::Gpu
            && !fabric->attachedEndpointKind(kNicEndpoint).has_value(),
        "the fabric reports its host and GPU endpoints with their kinds");
    test.expectThrowAs<std::invalid_argument>(
        [&fabric]() {
            (void)fabric->endpointAccounting(kNicEndpoint);
        },
        "endpoint accounting rejects an unattached identity");

    // Reached through the public fabric surface so both resolve branches are
    // covered: a device call would stop at its own requester-ownership check
    // before the fabric ever resolved the pair.
    PcieTransactionRequest direct;
    direct.client_id = kGpuRequester;
    direct.service_class = PcieServiceClass::PayloadWrite;
    direct.operation = PcieOperation::PostedWrite;
    direct.request_direction = PcieDirection::DeviceToHost;
    direct.ordering = PcieOrdering::Independent;
    direct.path_id = kHostPathId;
    direct.ordering_domain = kGpuOrderingDomain;
    direct.useful_bytes = 64;
    direct.transfer_bytes = 64;
    const std::uint64_t generation_before = fabric->generation();
    test.expectThrowAs<std::invalid_argument>(
        [&fabric, &direct]() {
            (void)fabric->submit(
                direct, PcieEndpointAttribution{kNicEndpoint, kHostEndpoint});
        },
        "the fabric rejects an unattached requester endpoint");
    test.expectThrowAs<std::invalid_argument>(
        [&fabric, &direct]() {
            (void)fabric->submit(
                direct, PcieEndpointAttribution{kGpuEndpoint, kGpuEndpoint});
        },
        "the fabric rejects one identity named as both ends");
    test.check(
        fabric->generation() == generation_before
            && fabric->attributedRequesterTransactions() == 0,
        "a rejected attribution charges nothing at the fabric surface");

    test.expectThrowAs<std::invalid_argument>(
        [&gpu_attachments]() {
            GpuDeviceConfig collision = gpuConfig(16, Arm::GpuDirect, true);
            collision.memory.device_owner_id = kGpuOwner;
            collision.fabric.endpoint_id = kGpuEndpoint;
            collision.fabric.ordering_domain = kGpuOrderingDomain + 1;
            GpuDevice second(collision, gpu_attachments);
            (void)second;
        },
        "a second device cannot claim a claimed endpoint identity");
    test.expectThrowAs<std::invalid_argument>(
        [&gpu_attachments]() {
            GpuDeviceConfig collision = gpuConfig(16, Arm::GpuDirect, true);
            collision.fabric.endpoint_id = kGpuEndpoint + 1;
            collision.fabric.ordering_domain = kGpuOrderingDomain;
            collision.memory.allocations.front().allocation_id =
                kPeerAllocation + 1;
            GpuDevice second(collision, gpu_attachments);
            (void)second;
        },
        "a second device cannot claim a claimed ordering domain");
    test.expectThrowAs<std::invalid_argument>(
        [&gpu_attachments]() {
            GpuDeviceConfig host_collision =
                gpuConfig(16, Arm::GpuDirect, true);
            host_collision.fabric.endpoint_id = kHostEndpoint;
            GpuDevice second(host_collision, gpu_attachments);
            (void)second;
        },
        "a device cannot claim the fabric host endpoint identity");
    test.expectThrowAs<std::invalid_argument>(
        []() {
            GpuDeviceConfig unnamed_host = gpuConfig(16, Arm::GpuDirect, true);
            unnamed_host.fabric.fabric.host_endpoint_id = 0;
            GpuDevice alone(unnamed_host);
            (void)alone;
        },
        "an attached GPU requires the fabric to name its host endpoint");
    test.expectThrowAs<std::invalid_argument>(
        []() {
            GpuDeviceConfig queue_object =
                gpuConfig(16, Arm::GpuDirect, true);
            queue_object.memory.allocations.front().object_kind =
                HostMemoryObjectKind::SqRing;
            GpuDevice alone(queue_object);
            (void)alone;
        },
        "a GPU cannot register a queue object");
    gpu.validateInvariants();
}

void testCrossDeviceRejection(TestRunner& test) {
    auto fabric = std::make_shared<PcieFabric>(fabricConfig(16, kHostEndpoint));
    auto registry = std::make_shared<VirtualHostMemory>(
        VirtualHostMemoryConfig{});
    GpuDeviceAttachments gpu_attachments;
    gpu_attachments.shared_pcie_fabric = fabric;
    gpu_attachments.shared_memory = registry;
    GpuDevice gpu(gpuConfig(16, Arm::HostBounce, false), gpu_attachments);
    RnicDeviceAttachments nic_attachments;
    nic_attachments.shared_pcie_fabric = fabric;
    nic_attachments.shared_host_memory = registry;
    RnicDevice nic(
        nicConfig(16, kHostEndpoint, kNicEndpoint), nic_attachments);

    const FabricLedger before = readLedger(*fabric);
    const std::uint64_t registry_generation = registry->generation();
    const std::size_t live_allocations = registry->liveAllocationCount();
    const std::size_t occupied = nic.occupiedSqEntries();
    const std::uint64_t posted = nic.counters().posted_wqes;

    test.expectThrowAs<std::invalid_argument>(
        [&nic]() {
            (void)nic.postSend(peerWorkRequest(4096), 0);
        },
        "a WQE naming an ungranted peer region is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&nic]() {
            WorkRequest request = peerWorkRequest(4096);
            request.data_memory->peer_device_owner_id = kNicOwner;
            (void)nic.postSend(request, 0);
        },
        "a WQE naming its own device as the peer owner is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&nic]() {
            WorkRequest request = peerWorkRequest(4096);
            request.data_memory->peer_device_owner_id = kGpuOwner + 77;
            (void)nic.postSend(request, 0);
        },
        "a WQE whose named peer disagrees with the region owner is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&gpu]() {
            GpuFabricTransfer foreign = stagingTransfer(4096);
            foreign.allocation_id = kNicDataAllocation;
            foreign.peer_device_owner_id = kNicOwner;
            (void)gpu.transfer(foreign);
        },
        "a GPU transfer into an ungranted peer region is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&gpu]() {
            GpuFabricTransfer own = stagingTransfer(4096);
            own.peer_device_owner_id = kGpuOwner;
            (void)gpu.transfer(own);
        },
        "a GPU transfer naming its own owner as a peer is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&nic]() {
            PcieTransactionRequest request;
            request.client_id = kRnicRequester;
            request.service_class = PcieServiceClass::PayloadRead;
            request.operation = PcieOperation::NonPostedRead;
            request.request_direction = PcieDirection::DeviceToHost;
            request.ordering = PcieOrdering::Independent;
            request.path_id = kHostPathId;
            request.ordering_domain =
                nic.pcieBinding()->pcie_submission_ordering_domain;
            request.useful_bytes = 64;
            request.transfer_bytes = 64;
            (void)nic.submitPcie(
                request,
                PcieEndpointAttribution{kGpuEndpoint, kHostEndpoint});
        },
        "a device cannot charge another device's requester endpoint");
    test.expectThrowAs<std::invalid_argument>(
        [&nic]() {
            PcieTransactionRequest request;
            request.client_id = kRnicRequester;
            request.service_class = PcieServiceClass::UarDoorbell;
            request.operation = PcieOperation::HostStore;
            request.request_direction = PcieDirection::HostToDevice;
            request.ordering = PcieOrdering::VisibilityDependency;
            request.path_id = 1;
            request.ordering_domain =
                nic.pcieBinding()->pcie_submission_ordering_domain;
            request.useful_bytes = 8;
            request.transfer_bytes = 8;
            (void)nic.submitPcie(
                request,
                PcieEndpointAttribution{kNicEndpoint, kHostEndpoint});
        },
        "an attributed host store is rejected because it crosses no link");
    test.expectThrowAs<std::invalid_argument>(
        [&nic]() {
            PcieTransactionRequest request;
            request.client_id = kRnicRequester;
            request.service_class = PcieServiceClass::PayloadRead;
            request.operation = PcieOperation::NonPostedRead;
            request.request_direction = PcieDirection::DeviceToHost;
            request.ordering = PcieOrdering::Independent;
            request.path_id = kHostPathId;
            request.ordering_domain =
                nic.pcieBinding()->pcie_submission_ordering_domain;
            request.useful_bytes = 64;
            request.transfer_bytes = 64;
            (void)nic.submitPcie(
                request, PcieEndpointAttribution{kNicEndpoint, 0});
        },
        "a half-named endpoint pair is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&nic]() {
            PcieTransactionRequest request;
            request.client_id = kRnicRequester;
            request.service_class = PcieServiceClass::PayloadRead;
            request.operation = PcieOperation::NonPostedRead;
            request.request_direction = PcieDirection::DeviceToHost;
            request.ordering = PcieOrdering::Independent;
            request.path_id = kHostPathId;
            request.ordering_domain =
                nic.pcieBinding()->pcie_submission_ordering_domain;
            request.useful_bytes = 64;
            request.transfer_bytes = 64;
            (void)nic.submitPcie(
                request,
                PcieEndpointAttribution{kNicEndpoint, kGpuEndpoint + 999});
        },
        "a transaction naming an unattached completer endpoint is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&nic_attachments]() {
            RnicDeviceConfig duplicate =
                nicConfig(16, kHostEndpoint, kNicEndpoint + 1);
            RnicDevice second(duplicate, nic_attachments);
            (void)second;
        },
        "a second device cannot claim a claimed host-memory owner");
    test.expectThrowAs<std::invalid_argument>(
        [&gpu_attachments]() {
            GpuDeviceConfig collision = gpuConfig(16, Arm::HostBounce, false);
            collision.memory.device_owner_id = kGpuOwner + 1;
            collision.memory.allocations.front().device_owner_id =
                kGpuOwner + 1;
            collision.memory.allocations.front().allocation_id =
                kPeerAllocation + 1;
            collision.fabric.ordering_domain = kGpuOrderingDomain + 1;
            GpuDevice second(collision, gpu_attachments);
            (void)second;
        },
        "a second device cannot claim a claimed endpoint identity");
    test.expectThrowAs<std::invalid_argument>(
        [&nic_attachments]() {
            RnicDeviceConfig device_local =
                nicConfig(16, kHostEndpoint, kNicEndpoint);
            device_local.host_memory.allocations.back().endpoint =
                PcieEndpointKind::GpuMemory;
            device_local.host_memory.allocations.back().path_id = kGpuPathId;
            RnicDevice second(device_local, nic_attachments);
            (void)second;
        },
        "an endpoint-attributed RNIC cannot own device-local memory");

    test.check(
        sameLedger(before, readLedger(*fabric))
            && fabric->generation() == before.generation
            && registry->generation() == registry_generation
            && registry->liveAllocationCount() == live_allocations
            && nic.occupiedSqEntries() == occupied
            && nic.counters().posted_wqes == posted
            && fabric->attributedRequesterTransactions() == 0
            && fabric->attributedCompleterTransactions() == 0
            && fabric->attributedUsefulBytes() == 0
            && fabric->attributedTransferredBytes() == 0
            && gpu.transferRecords().empty(),
        "every rejected cross-device claim left the fabric and registry "
        "unchanged");
    nic.validateInvariants();
    gpu.validateInvariants();
    fabric->validateInvariants();
    registry->validateInvariants();

    // The shared caller clock is the last cross-device guard: one device may not
    // schedule fabric work behind another device's clock and silently inherit
    // the backlog that peer left on the link.
    nic.postSend(ownWorkRequest(4096), 2'000'000);
    const FabricLedger after_post = readLedger(*fabric);
    test.expectThrowAs<std::logic_error>(
        [&gpu]() {
            (void)gpu.transfer(stagingTransfer(4096));
        },
        "a GPU transfer behind the shared fabric clock is rejected");
    test.expectThrowAs<std::logic_error>(
        [&nic]() {
            (void)nic.postSend(ownWorkRequest(4096), 1'000'000);
        },
        "an RNIC post behind the shared fabric clock is rejected");
    test.check(
        sameLedger(after_post, readLedger(*fabric))
            && gpu.transferRecords().empty(),
        "a rejected shared-clock operation charges nothing");
    nic.validateInvariants();
    gpu.validateInvariants();
    fabric->validateInvariants();
}

// A transfer must actually reach the link. Device-local memory does not: this
// device's own GPU memory sits behind its own port, and a peer's device-local
// memory is the peer-to-peer leg BACK-51 registers. Charging either on the host
// link would invent a traversal, exactly what the host-store guard refuses.
void testDeviceLocalTransfersFailClosed(TestRunner& test) {
    auto fabric = std::make_shared<PcieFabric>(fabricConfig(16, kHostEndpoint));
    auto registry = std::make_shared<VirtualHostMemory>(
        VirtualHostMemoryConfig{});
    GpuDeviceAttachments attachments;
    attachments.shared_pcie_fabric = fabric;
    attachments.shared_memory = registry;
    GpuDeviceConfig owner_config = gpuConfig(16, Arm::GpuDirect, false);
    owner_config.memory.peer_read_grants = {kPeerGpuOwner};
    GpuDevice owner(owner_config, attachments);

    GpuDeviceConfig peer_config = gpuConfig(16, Arm::GpuDirect, false);
    peer_config.identity.gpu_id = 2;
    peer_config.identity.requester_id = kGpuRequester + 1;
    peer_config.fabric.endpoint_id = kPeerGpuEndpoint;
    peer_config.fabric.ordering_domain = kGpuOrderingDomain + 2;
    peer_config.memory.device_owner_id = kPeerGpuOwner;
    peer_config.memory.allocations = {
        makeAllocation(
            kPeerGpuAllocation,
            kPeerGpuOwner,
            HostMemoryObjectKind::DataRegion,
            HostMemoryOwnerKind::MemoryRegion,
            kPeerMkey,
            PcieEndpointKind::HostPinnedMemory,
            kHostPathId,
            UINT64_C(0x700000000),
            UINT64_C(0x800000000),
            1,
            16,
            kPeerRegionBytes,
            kPeerMkey),
    };
    GpuDevice peer(peer_config, attachments);

    const FabricLedger before = readLedger(*fabric);
    test.expectThrowAs<std::invalid_argument>(
        [&owner]() {
            (void)owner.transfer(stagingTransfer(4096));
        },
        "a GPU transfer into its own device-local memory is rejected");
    test.expectThrowAs<std::invalid_argument>(
        [&peer]() {
            GpuFabricTransfer peer_to_peer = stagingTransfer(4096);
            peer_to_peer.allocation_id = kPeerAllocation;
            peer_to_peer.peer_device_owner_id = kGpuOwner;
            (void)peer.transfer(peer_to_peer);
        },
        "a granted GPU-to-GPU device-local transfer is rejected as unmodeled");
    test.check(
        sameLedger(before, readLedger(*fabric))
            && fabric->generation() == before.generation
            && fabric->attributedRequesterTransactions() == 0
            && owner.transferRecords().empty()
            && peer.transferRecords().empty(),
        "a rejected device-local transfer charges no link bytes");

    // The peer's own host-pinned region still transfers, so the rejection is
    // about the traversal and not about peer ownership.
    GpuFabricTransfer host_staged = stagingTransfer(4096);
    host_staged.allocation_id = kPeerGpuAllocation;
    const GpuFabricTransferRecord record = peer.transfer(host_staged);
    test.check(
        record.completer_endpoint_id == kHostEndpoint
            && record.completer_kind == PcieDeviceKind::Host
            && record.requester_endpoint_id == kPeerGpuEndpoint,
        "a host-pinned staging transfer still crosses to the host endpoint");
    owner.validateInvariants();
    peer.validateInvariants();
    fabric->validateInvariants();
    registry->validateInvariants();
}

// A released identity keeps its charges and can never be handed to a second
// device, so one row means one device for the fabric's whole lifetime, and the
// conservation identity stays checkable through the public surface after
// teardown.
void testEndpointIdentityLifecycle(TestRunner& test) {
    auto fabric = std::make_shared<PcieFabric>(fabricConfig(16, kHostEndpoint));
    auto registry = std::make_shared<VirtualHostMemory>(
        VirtualHostMemoryConfig{});
    GpuDeviceAttachments gpu_attachments;
    gpu_attachments.shared_pcie_fabric = fabric;
    gpu_attachments.shared_memory = registry;
    {
        GpuDevice gpu(gpuConfig(16, Arm::HostBounce, true), gpu_attachments);
        const GpuFabricTransferRecord staged =
            gpu.transfer(stagingTransfer(4096));
        test.check(
            staged.transfer_bytes == 4096
                && fabric->endpointLifecycle(kGpuEndpoint)
                    == PcieEndpointLifecycle::Attached,
            "an attached GPU endpoint reports itself attached");
        gpu.validateInvariants();
    }

    const PcieEndpointAccounting released =
        fabric->endpointAccounting(kGpuEndpoint);
    test.check(
        released.lifecycle == PcieEndpointLifecycle::Released
            && released.device_kind == PcieDeviceKind::Gpu
            && released.requester.transactions == 1
            && released.requester.useful_bytes == 4096,
        "a released endpoint keeps its charges under a released label");
    test.check(
        fabric->attachedEndpoints()
                == std::vector<PcieEndpointId>{kHostEndpoint}
            && fabric->knownEndpoints()
                == std::vector<PcieEndpointId>{kHostEndpoint, kGpuEndpoint}
            && !fabric->attachedEndpointKind(kGpuEndpoint).has_value(),
        "a released endpoint leaves the attached set but stays known");

    std::uint64_t requester_transactions = 0;
    std::uint64_t completer_transactions = 0;
    std::uint64_t requester_useful_bytes = 0;
    for (const PcieEndpointId endpoint_id : fabric->knownEndpoints()) {
        const PcieEndpointAccounting row =
            fabric->endpointAccounting(endpoint_id);
        requester_transactions += row.requester.transactions;
        completer_transactions += row.completer.transactions;
        requester_useful_bytes += row.requester.useful_bytes;
    }
    test.check(
        requester_transactions == fabric->attributedRequesterTransactions()
            && completer_transactions
                == fabric->attributedCompleterTransactions()
            && requester_useful_bytes == fabric->attributedUsefulBytes(),
        "the conservation identity is reproducible after teardown");

    test.expectThrowAs<std::invalid_argument>(
        [&gpu_attachments]() {
            GpuDevice reused(
                gpuConfig(16, Arm::HostBounce, true), gpu_attachments);
            (void)reused;
        },
        "a released endpoint identity cannot be reclaimed by its own kind");
    RnicDeviceAttachments nic_attachments;
    nic_attachments.shared_pcie_fabric = fabric;
    nic_attachments.shared_host_memory = registry;
    test.expectThrowAs<std::invalid_argument>(
        [&nic_attachments]() {
            RnicDevice cross_kind(
                nicConfig(16, kHostEndpoint, kGpuEndpoint), nic_attachments);
            (void)cross_kind;
        },
        "a released endpoint identity is refused across device kinds too");
    test.expectThrowAs<std::invalid_argument>(
        [&fabric]() {
            (void)fabric->endpointAccounting(kGpuEndpoint + 500);
        },
        "endpoint accounting rejects an identity never handed out");
    fabric->validateInvariants();
}

void testGpuAttachedLeg(TestRunner& test) {
    for (const std::uint64_t payload : studyPayloads()) {
        for (const std::uint16_t lanes : studyLaneCounts()) {
            const StudyCell bounce = runStudyCell(
                Arm::HostBounce, payload, lanes);
            const StudyCell direct = runStudyCell(
                Arm::GpuDirect, payload, lanes);
            const std::string cell = "P=" + std::to_string(payload)
                + ",L=" + std::to_string(lanes);
            test.check(
                direct.payload_completer == kGpuEndpoint
                    && direct.payload_completer_kind == PcieDeviceKind::Gpu
                    && bounce.payload_completer == kHostEndpoint
                    && bounce.payload_completer_kind == PcieDeviceKind::Host,
                cell + ": the payload read names the owning endpoint");
            test.check(
                direct.gpu.completer.transactions == 1
                    && direct.gpu.completer.useful_bytes == payload
                    && direct.gpu.requester.transactions == 0
                    && bounce.gpu.completer.transactions == 0
                    && bounce.gpu.requester.transactions == 1
                    && bounce.gpu.requester.useful_bytes == payload,
                cell + ": the GPU endpoint is charged in the right role");
            test.check(
                direct.gpu.device_kind == PcieDeviceKind::Gpu
                    && direct.nic.device_kind == PcieDeviceKind::Rnic
                    && direct.host.device_kind == PcieDeviceKind::Host,
                cell + ": endpoint rows carry their device kind");
            test.check(
                bounce.attributed_requester == bounce.attributed_completer
                    && direct.attributed_requester
                        == direct.attributed_completer
                    && bounce.attributed_requester
                        == direct.attributed_requester + 1,
                cell + ": the host bounce adds exactly one charged crossing");
            test.check(
                bounce.wqe_cqe_visible_ps > direct.wqe_cqe_visible_ps,
                cell + ": the host bounce completes later");
            test.check(
                bounce.wqe_cqe_visible_ps - direct.wqe_cqe_visible_ps
                    == bounce.staging_completed_ps,
                cell + ": the bounce penalty equals the staged serialization");
            test.check(
                direct.staging_transfers == 0
                    && direct.staging_completed_ps == 0
                    && bounce.staging_transfers == 1,
                cell + ": only the host bounce stages across the fabric");
            test.check(
                bounce.invariants_valid && direct.invariants_valid,
                cell + ": both arms validate their invariants");
        }
    }
}

void testStagingSerialization(TestRunner& test) {
    // Modeled link bytes are the payload plus one 24-byte posted-write header
    // per maximum-payload TLP, and picoseconds per byte are exact rational
    // generation-5 arithmetic: 8 * 130 * 10^6 / (32000 * lanes * 128).
    for (const std::uint64_t payload : studyPayloads()) {
        const std::uint64_t expected_link_bytes =
            payload + 24 * (payload / 256);
        for (const std::uint16_t lanes : studyLaneCounts()) {
            const StudyCell bounce = runStudyCell(
                Arm::HostBounce, payload, lanes);
            const std::uint64_t denominator =
                UINT64_C(32000) * lanes * UINT64_C(128);
            const std::uint64_t numerator =
                expected_link_bytes * UINT64_C(8) * UINT64_C(130)
                * UINT64_C(1000000);
            const Picoseconds expected_ps =
                (numerator + denominator - 1) / denominator;
            const std::string cell = "P=" + std::to_string(payload)
                + ",L=" + std::to_string(lanes);
            test.check(
                bounce.staging_link_bytes == expected_link_bytes,
                cell + ": staged link bytes match the TLP closed form");
            test.check(
                bounce.staging_completed_ps == expected_ps,
                cell + ": staged completion matches the serialization "
                       "closed form");
            test.check(
                bounce.staging_credit_wait_ps == 0
                    && bounce.staging_link_queue_wait_ps == 0,
                cell + ": the staged transfer stalls on no fabric resource");
        }
    }
}

void testGpuTransferRejectsBadShapes(TestRunner& test) {
    auto fabric = std::make_shared<PcieFabric>(fabricConfig(16, kHostEndpoint));
    auto registry = std::make_shared<VirtualHostMemory>(
        VirtualHostMemoryConfig{});
    GpuDeviceAttachments attachments;
    attachments.shared_pcie_fabric = fabric;
    attachments.shared_memory = registry;
    GpuDevice gpu(gpuConfig(16, Arm::HostBounce, true), attachments);

    test.expectThrowAs<std::invalid_argument>(
        [&gpu]() {
            GpuFabricTransfer store = stagingTransfer(4096);
            store.operation = PcieOperation::HostStore;
            (void)gpu.transfer(store);
        },
        "a GPU transfer cannot be a host store");
    test.expectThrowAs<std::invalid_argument>(
        [&gpu]() {
            GpuFabricTransfer mismatch = stagingTransfer(4096);
            mismatch.service_class = PcieServiceClass::CqeWrite;
            (void)gpu.transfer(mismatch);
        },
        "a GPU transfer cannot carry a queue service class");
    test.expectThrowAs<std::out_of_range>(
        [&gpu]() {
            GpuFabricTransfer oversized = stagingTransfer(kPeerRegionBytes + 1);
            (void)gpu.transfer(oversized);
        },
        "a GPU transfer cannot exceed its region");
    test.expectThrowAs<std::out_of_range>(
        [&gpu]() {
            GpuFabricTransfer unknown = stagingTransfer(4096);
            unknown.allocation_id = kPeerAllocation + 99;
            (void)gpu.transfer(unknown);
        },
        "a GPU transfer cannot name an unknown region");
    test.check(
        gpu.transferRecords().empty() && gpu.transferredBytes() == 0
            && fabric->generation() == 0,
        "every rejected GPU transfer left the device and fabric unchanged");

    const GpuFabricTransferRecord staged = gpu.transfer(stagingTransfer(4096));
    test.check(
        staged.sequence == 1 && staged.transfer_bytes == 4096
            && staged.requester_endpoint_id == kGpuEndpoint
            && staged.completer_endpoint_id == kHostEndpoint
            && staged.completer_kind == PcieDeviceKind::Host
            && staged.completed_at_ps > 0,
        "an accepted GPU transfer records its endpoint pair and completion");
    GpuFabricTransfer later = stagingTransfer(4096);
    later.submitted_at_ps = 1'000'000;
    const GpuFabricTransferRecord second = gpu.transfer(later);
    test.check(
        second.sequence == 2 && gpu.transferredBytes() == 8192,
        "a second accepted GPU transfer extends the ledger");
    test.expectThrowAs<std::logic_error>(
        [&gpu]() {
            GpuFabricTransfer backward = stagingTransfer(4096);
            backward.submitted_at_ps = 999'999;
            (void)gpu.transfer(backward);
        },
        "a GPU transfer cannot regress the caller clock");
    gpu.validateInvariants();
}

void writeStudyCsv(std::ostream& out) {
    out << "arm,payload_bytes,lane_count,data_endpoint,"
        << "payload_completer_endpoint,payload_completer_kind,"
        << "gpu_completer_transactions,gpu_completer_useful_bytes,"
        << "gpu_requester_transactions,gpu_requester_useful_bytes,"
        << "nic_requester_transactions,nic_requester_useful_bytes,"
        << "host_completer_transactions,host_completer_useful_bytes,"
        << "staging_transfers,staging_completed_ps,staging_link_bytes,"
        << "staging_credit_wait_ps,staging_link_queue_wait_ps,"
        << "wqe_admitted_ps,wqe_cqe_visible_ps,payload_read_transactions,"
        << "attributed_requester_transactions,"
        << "attributed_completer_transactions,invariants_valid\n";
    for (const Arm arm : {Arm::HostBounce, Arm::GpuDirect}) {
        for (const std::uint64_t payload : studyPayloads()) {
            for (const std::uint16_t lanes : studyLaneCounts()) {
                const StudyCell cell = runStudyCell(arm, payload, lanes);
                out << armName(cell.arm) << ',' << cell.payload_bytes << ','
                    << cell.lane_count << ','
                    << (cell.data_endpoint == PcieEndpointKind::GpuMemory
                            ? "gpu_memory"
                            : "host_pinned_memory")
                    << ',' << cell.payload_completer << ','
                    << toString(cell.payload_completer_kind) << ','
                    << cell.gpu.completer.transactions << ','
                    << cell.gpu.completer.useful_bytes << ','
                    << cell.gpu.requester.transactions << ','
                    << cell.gpu.requester.useful_bytes << ','
                    << cell.nic.requester.transactions << ','
                    << cell.nic.requester.useful_bytes << ','
                    << cell.host.completer.transactions << ','
                    << cell.host.completer.useful_bytes << ','
                    << cell.staging_transfers << ','
                    << cell.staging_completed_ps << ','
                    << cell.staging_link_bytes << ','
                    << cell.staging_credit_wait_ps << ','
                    << cell.staging_link_queue_wait_ps << ','
                    << cell.wqe_admitted_ps << ','
                    << cell.wqe_cqe_visible_ps << ','
                    << cell.payload_read_transactions << ','
                    << cell.attributed_requester << ','
                    << cell.attributed_completer << ','
                    << (cell.invariants_valid ? 1 : 0) << '\n';
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--study-csv") {
            writeStudyCsv(std::cout);
            return 0;
        }
        if (argc != 1) {
            std::cerr << "usage: simllm_rnic_gpu_device_test [--study-csv]\n";
            return 2;
        }
        TestRunner test;
        testOffPathEquivalence(test);
        testFabricEndpointClaims(test);
        testCrossDeviceRejection(test);
        testDeviceLocalTransfersFailClosed(test);
        testEndpointIdentityLifecycle(test);
        testGpuTransferRejectsBadShapes(test);
        testGpuAttachedLeg(test);
        testStagingSerialization(test);
        if (test.failures() != 0) {
            std::cerr << test.failures() << " GPU device check(s) failed\n";
            return 1;
        }
        std::cout << "all GPU device checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "unexpected non-standard exception\n";
        return 1;
    }
}
