#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "simllm/rnic/rnic_device.h"
#include "simllm/rnic/session_record.h"

namespace {

using simllm::rnic::CompletionEntry;
using simllm::rnic::HostMemoryAccessRecord;
using simllm::rnic::HostMemoryAllocation;
using simllm::rnic::HostMemoryObjectKind;
using simllm::rnic::HostMemoryOwnerKind;
using simllm::rnic::HostMemoryTranslationStage;
using simllm::rnic::PcieEndpointKind;
using simllm::rnic::PciePathConfig;
using simllm::rnic::PcieServiceClass;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicCqConsumptionRecord;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::RnicProducerShape;
using simllm::rnic::RnicSubmissionAgent;
using simllm::rnic::RnicSubmissionAgentKind;
using simllm::rnic::RnicSubmissionProfile;
using simllm::rnic::RnicSubmissionRecord;
using simllm::rnic::RnicUarMappingOwner;
using simllm::rnic::WorkRequest;
using simllm::rnic::WorkRequestDataMemory;
using simllm::rnic::makeStructuralSessionConfigRecord;
using simllm::rnic::rnicSha256Hex;
using simllm::rnic::toString;
using simllm::rnic::validateRnicSessionConfigRecord;

constexpr std::uint64_t kDeviceOwner = 920;
constexpr std::uint32_t kQpn = 19;
constexpr std::uint64_t kSqId = 201;
constexpr std::uint64_t kRqId = 202;
constexpr std::uint64_t kCqId = 203;
constexpr std::uint32_t kMkey = 177;
constexpr std::uint64_t kQpcAllocation = 21;
constexpr std::uint64_t kSqAllocation = 22;
constexpr std::uint64_t kRqAllocation = 23;
constexpr std::uint64_t kCqAllocation = 24;
constexpr std::uint64_t kDoorbellAllocation = 25;
constexpr std::uint64_t kDataAllocation = 26;
constexpr std::uint64_t kDescriptorAllocation = 27;
constexpr std::uint32_t kRnicRequester = 9100;

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

const char* endpointName(PcieEndpointKind endpoint) {
    switch (endpoint) {
    case PcieEndpointKind::HostPinnedMemory:
        return "host_pinned_memory";
    case PcieEndpointKind::GpuMemory:
        return "gpu_memory";
    case PcieEndpointKind::MmioBar:
        return "mmio_bar";
    case PcieEndpointKind::DeviceMemory:
        return "device_memory";
    default:
        return "invalid";
    }
}

bool sameAgent(
    const RnicSubmissionAgent& lhs,
    const RnicSubmissionAgent& rhs) {
    return lhs.version == rhs.version && lhs.kind == rhs.kind
        && lhs.id == rhs.id;
}

HostMemoryAllocation makeAllocation(
    std::uint64_t allocation_id,
    HostMemoryObjectKind object_kind,
    HostMemoryOwnerKind owner_kind,
    std::uint64_t owner_id,
    PcieEndpointKind endpoint,
    std::uint32_t path_id,
    std::uint64_t slot,
    std::size_t page_count,
    std::uint64_t length_bytes,
    std::optional<std::uint32_t> mkey = std::nullopt) {
    constexpr std::uint64_t page_size = 4096;
    HostMemoryAllocation allocation;
    allocation.allocation_id = allocation_id;
    allocation.device_owner_id = kDeviceOwner;
    allocation.object_kind = object_kind;
    allocation.owner_kind = owner_kind;
    allocation.owner_id = owner_id;
    allocation.endpoint = endpoint;
    allocation.path_id = path_id;
    allocation.virtual_address =
        UINT64_C(0x300000000) + slot * 4 * page_size;
    allocation.length_bytes = length_bytes;
    allocation.pages.page_size_bytes = page_size;
    for (std::size_t page = 0; page < page_count; ++page) {
        allocation.pages.physical_page_addresses.push_back(
            UINT64_C(0x400000000) + (slot * 4 + page) * page_size);
    }
    allocation.mkey = mkey;
    return allocation;
}

std::uint32_t producerId(RnicProducerShape shape) {
    switch (shape) {
    case RnicProducerShape::HostCpuDriver:
        return 7101;
    case RnicProducerShape::CpuProxy:
        return 7102;
    case RnicProducerShape::GpuInitiated:
        return 7103;
    default:
        throw std::logic_error("invalid fixture producer shape");
    }
}

std::uint32_t consumerId(RnicProducerShape shape) {
    switch (shape) {
    case RnicProducerShape::HostCpuDriver:
        return 8101;
    case RnicProducerShape::CpuProxy:
        return 8102;
    case RnicProducerShape::GpuInitiated:
        return 8103;
    default:
        throw std::logic_error("invalid fixture producer shape");
    }
}

RnicDeviceConfig submissionConfig(RnicProducerShape shape) {
    const bool gpu_initiated = shape == RnicProducerShape::GpuInitiated;
    const bool cpu_proxy = shape == RnicProducerShape::CpuProxy;
    const PcieEndpointKind ring_endpoint = gpu_initiated
        ? PcieEndpointKind::GpuMemory
        : PcieEndpointKind::HostPinnedMemory;
    const std::uint32_t ring_path = gpu_initiated ? 3 : 2;
    const PcieEndpointKind data_endpoint =
        shape == RnicProducerShape::HostCpuDriver
        ? PcieEndpointKind::HostPinnedMemory
        : PcieEndpointKind::GpuMemory;
    const std::uint32_t data_path =
        data_endpoint == PcieEndpointKind::GpuMemory ? 3 : 2;

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
    PciePathConfig gpu_path;
    gpu_path.path_id = 3;
    gpu_path.endpoint = PcieEndpointKind::GpuMemory;
    config.dma.fabric.paths.push_back(gpu_path);
    config.dma.work_queue.pcie_doorbell_record_path_id = ring_path;
    config.dma.work_queue.pcie_sq_memory_path_id = ring_path;
    config.dma.work_queue.pcie_cq_memory_path_id = ring_path;
    config.host_memory.enabled = true;
    config.host_memory.device_owner_id = kDeviceOwner;
    config.host_memory.work_queue.qpc_icm_allocation_id = kQpcAllocation;
    config.host_memory.work_queue.sq_ring_allocation_id = kSqAllocation;
    config.host_memory.work_queue.rq_ring_allocation_id = kRqAllocation;
    config.host_memory.work_queue.cq_ring_allocation_id = kCqAllocation;
    config.host_memory.work_queue.doorbell_record_allocation_id =
        kDoorbellAllocation;
    config.host_memory.allocations = {
        makeAllocation(
            kQpcAllocation,
            HostMemoryObjectKind::QpcIcm,
            HostMemoryOwnerKind::QueuePair,
            kQpn,
            PcieEndpointKind::HostPinnedMemory,
            2,
            1,
            1,
            256),
        makeAllocation(
            kSqAllocation,
            HostMemoryObjectKind::SqRing,
            HostMemoryOwnerKind::SendQueue,
            kSqId,
            ring_endpoint,
            ring_path,
            2,
            1,
            16 * 64),
        makeAllocation(
            kRqAllocation,
            HostMemoryObjectKind::RqRing,
            HostMemoryOwnerKind::ReceiveQueue,
            kRqId,
            PcieEndpointKind::HostPinnedMemory,
            2,
            3,
            1,
            16 * 64),
        makeAllocation(
            kCqAllocation,
            HostMemoryObjectKind::CqRing,
            HostMemoryOwnerKind::CompletionQueue,
            kCqId,
            ring_endpoint,
            ring_path,
            4,
            1,
            16 * 64),
        makeAllocation(
            kDoorbellAllocation,
            HostMemoryObjectKind::DoorbellRecord,
            HostMemoryOwnerKind::SendQueue,
            kSqId,
            ring_endpoint,
            ring_path,
            5,
            1,
            4),
        makeAllocation(
            kDataAllocation,
            HostMemoryObjectKind::DataRegion,
            HostMemoryOwnerKind::MemoryRegion,
            kMkey,
            data_endpoint,
            data_path,
            6,
            2,
            8192,
            kMkey),
    };
    config.submission.producer_shape = shape;
    config.submission.producer_id = producerId(shape);
    config.submission.cq_consumer_id = consumerId(shape);
    config.submission.rnic_requester_id = kRnicRequester;
    if (cpu_proxy) {
        config.submission.descriptor_writer_id = 7202;
        config.submission.descriptor_queue_allocation_id =
            kDescriptorAllocation;
        config.host_memory.allocations.push_back(makeAllocation(
            kDescriptorAllocation,
            HostMemoryObjectKind::DescriptorQueue,
            HostMemoryOwnerKind::SubmissionProducer,
            7202,
            PcieEndpointKind::HostPinnedMemory,
            2,
            7,
            1,
            4096));
    }
    return config;
}

std::size_t countObject(
    const std::vector<HostMemoryAccessRecord>& accesses,
    HostMemoryObjectKind kind) {
    return static_cast<std::size_t>(std::count_if(
        accesses.begin(), accesses.end(), [kind](const auto& access) {
            return access.object_kind == kind;
        }));
}

std::size_t countStage(
    const std::vector<HostMemoryAccessRecord>& accesses,
    HostMemoryObjectKind kind,
    HostMemoryTranslationStage stage) {
    std::size_t count = 0;
    for (const HostMemoryAccessRecord& access : accesses) {
        if (access.object_kind == kind) {
            count += static_cast<std::size_t>(std::count(
                access.translation_stages.begin(),
                access.translation_stages.end(),
                stage));
        }
    }
    return count;
}

struct StudyRow {
    RnicProducerShape producer_shape{RnicProducerShape::HostCpuDriver};
    std::size_t batch_size{0};
    RnicSubmissionProfile profile;
    PcieEndpointKind data_endpoint{PcieEndpointKind::HostPinnedMemory};
    std::size_t submission_records{0};
    std::size_t cq_consumption_records{0};
    std::size_t completed_wqes{0};
    std::size_t qpc_fetches{0};
    std::uint64_t qpc_icm_transactions{0};
    std::size_t qpc_mkey_events{0};
    std::size_t qpc_mpt_events{0};
    std::size_t qpc_mtt_events{0};
    std::size_t data_mkey_events{0};
    std::size_t data_mpt_events{0};
    std::size_t data_mtt_events{0};
    bool qpc_stays_host_icm{false};
    bool exactly_one_cq_consumer{false};
    bool identities_separate_from_qpn{false};
    bool invariants_valid{false};
};

StudyRow runFixture(RnicProducerShape shape, std::size_t batch_size) {
    RnicDevice device(submissionConfig(shape));
    if (!device.submissionProfile().has_value()) {
        throw std::logic_error("submission fixture lacks a profile");
    }
    for (std::size_t index = 0; index < batch_size; ++index) {
        WorkRequest request;
        request.wr_id = index + 1;
        request.flow_id = 2000 + index;
        request.destination = 2;
        request.payload_bytes = 64;
        request.signaled = true;
        WorkRequestDataMemory data;
        data.allocation_id = kDataAllocation;
        data.mkey = kMkey;
        data.allocation_offset_bytes = 4096;
        request.data_memory = data;
        if (device.postSend(request, 0).status != PostStatus::Accepted) {
            throw std::logic_error("submission fixture post failed");
        }
    }
    if (device.ringDoorbell(0).wqe_count != batch_size) {
        throw std::logic_error("submission fixture doorbell failed");
    }

    std::vector<CompletionEntry> completions;
    Picoseconds now_ps = 0;
    std::size_t iterations = 0;
    while (device.hasPendingPhysicalWork()) {
        const std::optional<Picoseconds> next = device.nextEventTime();
        if (!next.has_value() || *next < now_ps || ++iterations > 1000) {
            throw std::logic_error("submission fixture lost event progress");
        }
        now_ps = *next;
        static_cast<void>(device.progress(now_ps));
        std::vector<CompletionEntry> polled = device.pollCompletionQueue(
            std::numeric_limits<std::size_t>::max(), now_ps);
        completions.insert(
            completions.end(), polled.begin(), polled.end());
    }
    device.validateInvariants();

    const RnicSubmissionProfile& profile = *device.submissionProfile();
    const auto& accesses = device.memoryAccesses();
    const auto& submissions = device.submissionRecords();
    const auto& consumptions = device.cqConsumptionRecords();
    const HostMemoryAllocation& qpc =
        device.hostMemory()->allocation(kQpcAllocation);
    const HostMemoryAllocation& data =
        device.hostMemory()->allocation(kDataAllocation);
    const bool sole_consumer = consumptions.size() == batch_size
        && std::all_of(
            consumptions.begin(),
            consumptions.end(),
            [&profile](const RnicCqConsumptionRecord& record) {
                return sameAgent(record.consumer, profile.cq_consumer);
            });
    // Doorbell producer identity depends on the selected shape, so check it
    // separately from all device-initiated accesses.
    const bool clients_match = std::all_of(
        accesses.begin(), accesses.end(), [shape](const auto& access) {
            return access.client_id
                == (access.object_kind == HostMemoryObjectKind::DoorbellRecord
                        ? producerId(shape)
                        : kRnicRequester);
        });

    StudyRow row;
    row.producer_shape = shape;
    row.batch_size = batch_size;
    row.profile = profile;
    row.data_endpoint = data.endpoint;
    row.submission_records = submissions.size();
    row.cq_consumption_records = consumptions.size();
    row.completed_wqes = completions.size();
    row.qpc_fetches = countObject(accesses, HostMemoryObjectKind::QpcIcm);
    row.qpc_icm_transactions =
        device.pcieFabric()->accounting(PcieServiceClass::QpcIcm).transactions;
    row.qpc_mkey_events = countStage(
        accesses,
        HostMemoryObjectKind::QpcIcm,
        HostMemoryTranslationStage::Mkey);
    row.qpc_mpt_events = countStage(
        accesses,
        HostMemoryObjectKind::QpcIcm,
        HostMemoryTranslationStage::Mpt);
    row.qpc_mtt_events = countStage(
        accesses,
        HostMemoryObjectKind::QpcIcm,
        HostMemoryTranslationStage::Mtt);
    row.data_mkey_events = countStage(
        accesses,
        HostMemoryObjectKind::DataRegion,
        HostMemoryTranslationStage::Mkey);
    row.data_mpt_events = countStage(
        accesses,
        HostMemoryObjectKind::DataRegion,
        HostMemoryTranslationStage::Mpt);
    row.data_mtt_events = countStage(
        accesses,
        HostMemoryObjectKind::DataRegion,
        HostMemoryTranslationStage::Mtt);
    row.qpc_stays_host_icm =
        qpc.endpoint == PcieEndpointKind::HostPinnedMemory
        && row.qpc_fetches == batch_size && row.qpc_mkey_events == 0
        && row.qpc_mpt_events == 0 && row.qpc_mtt_events == 0;
    row.exactly_one_cq_consumer = sole_consumer;
    row.identities_separate_from_qpn = profile.producer.id != kQpn
        && profile.cq_consumer.id != kQpn
        && profile.rnic_requester_id != kQpn
        && (!profile.descriptor_writer.has_value()
            || profile.descriptor_writer->id != kQpn)
        && clients_match;
    row.invariants_valid = true;
    return row;
}

void checkStudyRow(TestRunner& test, const StudyRow& row) {
    const std::size_t batch = row.batch_size;
    test.check(row.submission_records == batch, "submission record count");
    test.check(
        row.cq_consumption_records == batch,
        "CQ-consumption record count");
    test.check(row.completed_wqes == batch, "completed WQE count");
    test.check(row.qpc_fetches == batch, "QPC fetch count");
    test.check(row.qpc_icm_transactions == batch, "QpcIcm count");
    test.check(row.qpc_mkey_events == 0, "QPC has no MKey event");
    test.check(row.qpc_mpt_events == 0, "QPC has no MPT event");
    test.check(row.qpc_mtt_events == 0, "QPC has no MTT event");
    test.check(row.data_mkey_events == batch, "data MKey count");
    test.check(row.data_mpt_events == batch, "data MPT count");
    test.check(row.data_mtt_events == batch, "data MTT count");
    test.check(row.qpc_stays_host_icm, "QPC remains host ICM");
    test.check(row.exactly_one_cq_consumer, "CQ has one consumer");
    test.check(
        row.identities_separate_from_qpn,
        "agent identities are separate from QPN");
    test.check(row.invariants_valid, "fixture invariants valid");
}

void testValidationAndAtomicity(TestRunner& test) {
    RnicDeviceConfig gpu_mismatch = submissionConfig(
        RnicProducerShape::GpuInitiated);
    gpu_mismatch.host_memory.allocations[1].endpoint =
        PcieEndpointKind::HostPinnedMemory;
    gpu_mismatch.host_memory.allocations[1].path_id = 2;
    test.expectThrowAs<std::invalid_argument>(
        [&]() { RnicDevice device(gpu_mismatch); },
        "GPU producer rejects a host SQ");

    RnicDeviceConfig qpc_gpu = submissionConfig(
        RnicProducerShape::GpuInitiated);
    qpc_gpu.host_memory.allocations[0].endpoint = PcieEndpointKind::GpuMemory;
    qpc_gpu.host_memory.allocations[0].path_id = 3;
    test.expectThrowAs<std::invalid_argument>(
        [&]() { RnicDevice device(qpc_gpu); },
        "every producer rejects GPU-resident QPC ICM");

    RnicDeviceConfig proxy_missing = submissionConfig(
        RnicProducerShape::CpuProxy);
    proxy_missing.submission.descriptor_queue_allocation_id = 0;
    test.expectThrowAs<std::invalid_argument>(
        [&]() { RnicDevice device(proxy_missing); },
        "CPU proxy requires its descriptor queue");

    RnicDeviceConfig scalar;
    scalar.submission.producer_shape = RnicProducerShape::GpuInitiated;
    scalar.submission.producer_id = 1;
    scalar.submission.cq_consumer_id = 2;
    scalar.submission.rnic_requester_id = 3;
    test.expectThrowAs<std::invalid_argument>(
        [&]() { RnicDevice device(scalar); },
        "DMA-off device rejects active submission fields");

    RnicDevice device(submissionConfig(RnicProducerShape::GpuInitiated));
    const std::uint64_t generation = device.pcieFabric()->generation();
    WorkRequest invalid;
    invalid.wr_id = 1;
    invalid.flow_id = 1;
    invalid.destination = 2;
    invalid.payload_bytes = 64;
    invalid.data_memory = WorkRequestDataMemory{};
    invalid.data_memory->allocation_id = kDataAllocation;
    invalid.data_memory->mkey = kMkey + 1;
    invalid.data_memory->allocation_offset_bytes = 4096;
    test.expectThrowAs<std::invalid_argument>(
        [&]() { static_cast<void>(device.postSend(invalid, 0)); },
        "bad GPU data descriptor rejects");
    test.check(
        device.records().empty() && device.submissionRecords().empty()
            && device.cqConsumptionRecords().empty()
            && device.pcieFabric()->generation() == generation,
        "failed GPU post preserves all ledgers and fabric state");
    test.check(
        device.pollCompletionQueue(1, 0).empty()
            && device.cqConsumptionRecords().empty(),
        "empty CQ poll creates no consumption record");
}

void testDefaultAndSessionSchema(TestRunner& test) {
    RnicDevice scalar{RnicDeviceConfig{}};
    test.check(
        !scalar.submissionProfile().has_value()
            && scalar.submissionRecords().empty()
            && scalar.cqConsumptionRecords().empty(),
        "DMA-off default retains no submission state");

    RnicDeviceConfig dma;
    dma.dma.enabled = true;
    RnicDevice default_dma(dma);
    test.check(
        default_dma.submissionProfile().has_value()
            && default_dma.submissionProfile()->producer_shape
                == RnicProducerShape::HostCpuDriver
            && default_dma.submissionProfile()->producer.id == dma.identity.qpn
            && default_dma.submissionProfile()->rnic_requester_id
                == dma.identity.qpn,
        "DMA default resolves to the byte-compatible host CPU identity");

    RnicDevice host(submissionConfig(RnicProducerShape::HostCpuDriver));
    auto v2_record = makeStructuralSessionConfigRecord(
        "submission-v2", "rnic-nn", host);
    const std::string v3_schema = "simllm-rnic-effective-hardware-v3";
    const std::string v2_schema = "simllm-rnic-effective-hardware-v2";
    std::size_t schema_position =
        v2_record.effective_hardware_json->find(v3_schema);
    const std::string submission_field = ",\"submission\":{";
    std::size_t submission_position =
        v2_record.effective_hardware_json->find(submission_field);
    std::size_t submission_end = submission_position == std::string::npos
        ? std::string::npos
        : v2_record.effective_hardware_json->find(
              '}', submission_position + submission_field.size());
    if (schema_position == std::string::npos
        || submission_position == std::string::npos
        || submission_end == std::string::npos) {
        test.check(false, "v2 compatibility fixture found v3 fields");
    } else {
        v2_record.effective_hardware_json->replace(
            schema_position, v3_schema.size(), v2_schema);
        v2_record.effective_hardware_json->erase(
            submission_position, submission_end - submission_position + 1U);
        v2_record.hardware_config_sha256 = rnicSha256Hex(
            *v2_record.effective_hardware_json);
        try {
            validateRnicSessionConfigRecord(v2_record);
            test.check(true, "native parser retains strict v2 compatibility");
        } catch (const std::exception& error) {
            test.check(
                false,
                std::string("native parser retains strict v2 compatibility; ")
                    + error.what());
        }
    }

    RnicDevice proxy(submissionConfig(RnicProducerShape::CpuProxy));
    auto record = makeStructuralSessionConfigRecord(
        "submission", "rnic-nn", proxy);
    test.check(
        record.effective_hardware_json.has_value()
            && record.effective_hardware_json->find(
                   "simllm-rnic-effective-hardware-v3")
                != std::string::npos
            && record.effective_hardware_json->find(
                   "\"producer_shape\":\"cpu_proxy\"")
                != std::string::npos
            && record.effective_hardware_json->find(
                   "\"descriptor_queue_allocation_id\":27")
                != std::string::npos,
        "effective hardware v3 records the resolved proxy shape");

    const std::string needle =
        "\"queue_endpoint\":\"host_pinned_memory\"";
    const std::string replacement =
        "\"queue_endpoint\":\"gpu_memory\"";
    const std::size_t position = record.effective_hardware_json->find(needle);
    if (position == std::string::npos) {
        test.check(false, "strict-schema mutation fixture found its field");
        return;
    }
    record.effective_hardware_json->replace(
        position, needle.size(), replacement);
    record.hardware_config_sha256 = rnicSha256Hex(
        *record.effective_hardware_json);
    test.expectThrowAs<std::invalid_argument>(
        [&]() { validateRnicSessionConfigRecord(record); },
        "strict v3 parser rejects shape and endpoint disagreement");
}

std::vector<StudyRow> studyRows() {
    std::vector<StudyRow> rows;
    for (const RnicProducerShape shape : {
             RnicProducerShape::HostCpuDriver,
             RnicProducerShape::CpuProxy,
             RnicProducerShape::GpuInitiated}) {
        for (const std::size_t batch : {std::size_t{1}, std::size_t{4}}) {
            rows.push_back(runFixture(shape, batch));
        }
    }
    return rows;
}

void printStudyCsv(const std::vector<StudyRow>& rows) {
    std::cout
        << "producer_shape,batch_size,producer_kind,producer_id,"
           "descriptor_writer_kind,descriptor_writer_id,"
           "descriptor_queue_endpoint,sq_endpoint,cq_endpoint,"
           "doorbell_endpoint,data_endpoint,uar_mapping_owner,"
           "cq_consumer_kind,cq_consumer_id,rnic_requester_id,qpn,"
           "submission_records,cq_consumption_records,completed_wqes,"
           "qpc_fetches,qpc_icm_transactions,qpc_mkey_events,"
           "qpc_mpt_events,qpc_mtt_events,data_mkey_events,"
           "data_mpt_events,data_mtt_events,qpc_stays_host_icm,"
           "exactly_one_cq_consumer,identities_separate_from_qpn,"
           "invariants_valid\n";
    for (const StudyRow& row : rows) {
        const RnicSubmissionProfile& profile = row.profile;
        std::cout << toString(row.producer_shape) << ',' << row.batch_size
                  << ',' << toString(profile.producer_shape) << ','
                  << profile.producer.id << ','
                  << (profile.descriptor_writer.has_value()
                          ? toString(profile.descriptor_writer->kind)
                          : "none")
                  << ','
                  << (profile.descriptor_writer.has_value()
                          ? profile.descriptor_writer->id
                          : 0)
                  << ','
                  << (profile.descriptor_queue_endpoint.has_value()
                          ? endpointName(*profile.descriptor_queue_endpoint)
                          : "none")
                  << ',' << endpointName(profile.queue_endpoint) << ','
                  << endpointName(profile.queue_endpoint) << ','
                  << endpointName(profile.queue_endpoint) << ','
                  << endpointName(row.data_endpoint) << ','
                  << toString(profile.uar_mapping_owner) << ','
                  << toString(profile.cq_consumer.kind) << ','
                  << profile.cq_consumer.id << ','
                  << profile.rnic_requester_id << ',' << kQpn << ','
                  << row.submission_records << ','
                  << row.cq_consumption_records << ',' << row.completed_wqes
                  << ',' << row.qpc_fetches << ','
                  << row.qpc_icm_transactions << ',' << row.qpc_mkey_events
                  << ',' << row.qpc_mpt_events << ',' << row.qpc_mtt_events
                  << ',' << row.data_mkey_events << ',' << row.data_mpt_events
                  << ',' << row.data_mtt_events << ','
                  << static_cast<int>(row.qpc_stays_host_icm) << ','
                  << static_cast<int>(row.exactly_one_cq_consumer) << ','
                  << static_cast<int>(row.identities_separate_from_qpn) << ','
                  << static_cast<int>(row.invariants_valid) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<StudyRow> rows = studyRows();
        if (argc == 2 && std::string(argv[1]) == "--study-csv") {
            printStudyCsv(rows);
            return 0;
        }
        if (argc != 1) {
            std::cerr << "usage: simllm_rnic_submission_test [--study-csv]\n";
            return 2;
        }

        TestRunner test;
        for (const StudyRow& row : rows) {
            checkStudyRow(test, row);
        }
        testValidationAndAtomicity(test);
        testDefaultAndSessionSchema(test);
        if (test.failures() != 0) {
            std::cerr << test.failures() << " submission checks failed\n";
            return 1;
        }
        std::cout << "RNIC submission checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unexpected submission failure: " << error.what() << '\n';
        return 1;
    }
}
