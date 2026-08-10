#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "simllm/rnic/host_memory.h"
#include "simllm/rnic/rnic_device.h"
#include "simllm/rnic/session_record.h"

namespace {

using simllm::rnic::CompletionEntry;
using simllm::rnic::HostMemoryAccessRecord;
using simllm::rnic::HostMemoryAllocation;
using simllm::rnic::HostMemoryDeviceOwnerId;
using simllm::rnic::HostMemoryLifecycleKind;
using simllm::rnic::HostMemoryObjectKind;
using simllm::rnic::HostMemoryOwnerKind;
using simllm::rnic::HostMemoryTranslationStage;
using simllm::rnic::PcieEndpointKind;
using simllm::rnic::PcieServiceClass;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceAttachments;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::RnicStageApplicability;
using simllm::rnic::effectiveHardwareConfigSha256;
using simllm::rnic::makeStructuralSessionConfigRecord;
using simllm::rnic::VirtualHostMemory;
using simllm::rnic::WorkRequest;
using simllm::rnic::WorkRequestDataMemory;

constexpr std::uint64_t kDeviceOwner = 900;
constexpr std::uint32_t kQpn = 19;
constexpr std::uint64_t kSqId = 101;
constexpr std::uint64_t kRqId = 102;
constexpr std::uint64_t kCqId = 103;
constexpr std::uint32_t kMkey = 77;
constexpr std::uint64_t kQpcAllocation = 11;
constexpr std::uint64_t kSqAllocation = 12;
constexpr std::uint64_t kRqAllocation = 13;
constexpr std::uint64_t kCqAllocation = 14;
constexpr std::uint64_t kDoorbellAllocation = 15;
constexpr std::uint64_t kDataAllocation = 16;

static_assert(!std::is_copy_constructible_v<VirtualHostMemory>);
static_assert(!std::is_move_constructible_v<VirtualHostMemory>);

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

HostMemoryAllocation makeAllocation(
    std::uint64_t allocation_id,
    HostMemoryObjectKind object_kind,
    HostMemoryOwnerKind owner_kind,
    std::uint64_t owner_id,
    std::uint64_t page_size_bytes,
    std::uint64_t slot,
    std::size_t page_count,
    std::uint64_t length_bytes,
    std::optional<std::uint32_t> mkey = std::nullopt) {
    HostMemoryAllocation allocation;
    allocation.allocation_id = allocation_id;
    allocation.device_owner_id = kDeviceOwner;
    allocation.object_kind = object_kind;
    allocation.owner_kind = owner_kind;
    allocation.owner_id = owner_id;
    allocation.endpoint = PcieEndpointKind::HostPinnedMemory;
    allocation.path_id = 2;
    allocation.virtual_address =
        UINT64_C(0x100000000) + slot * 4 * page_size_bytes;
    allocation.length_bytes = length_bytes;
    allocation.pages.page_size_bytes = page_size_bytes;
    for (std::size_t page = 0; page < page_count; ++page) {
        allocation.pages.physical_page_addresses.push_back(
            UINT64_C(0x200000000)
            + (slot * 4 + page) * page_size_bytes);
    }
    allocation.mkey = mkey;
    return allocation;
}

std::vector<HostMemoryAllocation> makeAllocations(
    std::uint64_t page_size_bytes) {
    return {
        makeAllocation(
            kQpcAllocation,
            HostMemoryObjectKind::QpcIcm,
            HostMemoryOwnerKind::QueuePair,
            kQpn,
            page_size_bytes,
            1,
            1,
            256),
        makeAllocation(
            kSqAllocation,
            HostMemoryObjectKind::SqRing,
            HostMemoryOwnerKind::SendQueue,
            kSqId,
            page_size_bytes,
            2,
            1,
            16 * 64),
        makeAllocation(
            kRqAllocation,
            HostMemoryObjectKind::RqRing,
            HostMemoryOwnerKind::ReceiveQueue,
            kRqId,
            page_size_bytes,
            3,
            1,
            16 * 64),
        makeAllocation(
            kCqAllocation,
            HostMemoryObjectKind::CqRing,
            HostMemoryOwnerKind::CompletionQueue,
            kCqId,
            page_size_bytes,
            4,
            1,
            16 * 64),
        makeAllocation(
            kDoorbellAllocation,
            HostMemoryObjectKind::DoorbellRecord,
            HostMemoryOwnerKind::SendQueue,
            kSqId,
            page_size_bytes,
            5,
            1,
            4),
        makeAllocation(
            kDataAllocation,
            HostMemoryObjectKind::DataRegion,
            HostMemoryOwnerKind::MemoryRegion,
            kMkey,
            page_size_bytes,
            6,
            2,
            2 * page_size_bytes,
            kMkey),
    };
}

RnicDeviceConfig hostMemoryConfig(std::uint64_t page_size_bytes) {
    RnicDeviceConfig config;
    config.identity.qpn = kQpn;
    config.identity.policy_context_token = 1900;
    config.work_queue.sq_id = kSqId;
    config.work_queue.cq_id = kCqId;
    config.work_queue.source = 7;
    config.work_queue.qpn = kQpn;
    config.work_queue.policy_context_token = 1900;
    config.work_queue.sq_depth = 16;
    config.work_queue.cq_depth = 16;
    config.dma.enabled = true;
    config.host_memory.enabled = true;
    config.host_memory.device_owner_id = kDeviceOwner;
    config.host_memory.work_queue.qpc_icm_allocation_id = kQpcAllocation;
    config.host_memory.work_queue.sq_ring_allocation_id = kSqAllocation;
    config.host_memory.work_queue.rq_ring_allocation_id = kRqAllocation;
    config.host_memory.work_queue.cq_ring_allocation_id = kCqAllocation;
    config.host_memory.work_queue.doorbell_record_allocation_id =
        kDoorbellAllocation;
    config.host_memory.allocations = makeAllocations(page_size_bytes);
    return config;
}

RnicDeviceConfig secondHostMemoryConfig(
    HostMemoryDeviceOwnerId device_owner_id) {
    constexpr std::uint64_t allocation_delta = 100;
    constexpr std::uint64_t address_delta = UINT64_C(0x1000000000);
    constexpr std::uint32_t qpn = 29;
    constexpr std::uint64_t sq_id = 301;
    constexpr std::uint64_t rq_id = 302;
    constexpr std::uint64_t cq_id = 303;

    RnicDeviceConfig config = hostMemoryConfig(4096);
    config.identity.qpn = qpn;
    config.identity.policy_context_token = 2900;
    config.work_queue.sq_id = sq_id;
    config.work_queue.cq_id = cq_id;
    config.work_queue.qpn = qpn;
    config.work_queue.policy_context_token = 2900;
    config.host_memory.device_owner_id = device_owner_id;
    config.host_memory.work_queue.qpc_icm_allocation_id += allocation_delta;
    config.host_memory.work_queue.sq_ring_allocation_id += allocation_delta;
    config.host_memory.work_queue.rq_ring_allocation_id += allocation_delta;
    config.host_memory.work_queue.cq_ring_allocation_id += allocation_delta;
    config.host_memory.work_queue.doorbell_record_allocation_id +=
        allocation_delta;
    for (HostMemoryAllocation& allocation : config.host_memory.allocations) {
        allocation.allocation_id += allocation_delta;
        allocation.device_owner_id = device_owner_id;
        allocation.virtual_address += address_delta;
        for (std::uint64_t& page : allocation.pages.physical_page_addresses) {
            page += address_delta;
        }
        switch (allocation.object_kind) {
        case HostMemoryObjectKind::QpcIcm:
            allocation.owner_id = qpn;
            break;
        case HostMemoryObjectKind::SqRing:
        case HostMemoryObjectKind::DoorbellRecord:
            allocation.owner_id = sq_id;
            break;
        case HostMemoryObjectKind::RqRing:
            allocation.owner_id = rq_id;
            break;
        case HostMemoryObjectKind::CqRing:
            allocation.owner_id = cq_id;
            break;
        case HostMemoryObjectKind::DataRegion:
            break;
        default:
            throw std::logic_error(
                "unexpected alternate host-memory allocation kind");
        }
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
    std::uint64_t page_size_bytes{0};
    std::size_t batch_size{0};
    std::size_t qpc_fetches{0};
    std::uint64_t qpc_icm_transactions{0};
    std::size_t qpc_mkey_events{0};
    std::size_t qpc_mpt_events{0};
    std::size_t qpc_mtt_events{0};
    std::size_t sq_page_list_events{0};
    std::size_t data_mkey_events{0};
    std::size_t data_mpt_events{0};
    std::size_t data_mtt_events{0};
    std::size_t cq_page_list_events{0};
    std::uint64_t mtt_mpt_transactions{0};
    std::uint64_t wqe_read_transactions{0};
    std::uint64_t payload_read_transactions{0};
    std::uint64_t cqe_write_transactions{0};
    std::uint64_t doorbell_record_transactions{0};
    std::uint64_t uar_transactions{0};
    std::size_t registration_events{0};
    std::size_t teardown_events{0};
    bool second_data_page_selected{false};
    bool invariants_valid{false};
};

StudyRow runFixture(std::uint64_t page_size_bytes, std::size_t batch_size) {
    RnicDevice device(hostMemoryConfig(page_size_bytes));
    const VirtualHostMemory* memory = device.hostMemory();
    if (memory == nullptr || memory->liveAllocationCount(kDeviceOwner) != 6) {
        throw std::logic_error("host-memory fixture registration failed");
    }

    for (std::size_t index = 0; index < batch_size; ++index) {
        WorkRequest request;
        request.wr_id = index + 1;
        request.flow_id = 1000 + index;
        request.destination = 2;
        request.payload_bytes = 64;
        request.signaled = true;
        WorkRequestDataMemory data;
        data.allocation_id = kDataAllocation;
        data.mkey = kMkey;
        data.allocation_offset_bytes = page_size_bytes;
        request.data_memory = data;
        if (device.postSend(request, 0).status != PostStatus::Accepted) {
            throw std::logic_error("host-memory fixture post failed");
        }
    }
    const auto doorbell = device.ringDoorbell(0);
    if (doorbell.wqe_count != batch_size) {
        throw std::logic_error("host-memory fixture doorbell count failed");
    }

    std::vector<CompletionEntry> completions;
    Picoseconds now_ps = 0;
    std::size_t iterations = 0;
    while (device.hasPendingPhysicalWork()) {
        const std::optional<Picoseconds> next = device.nextEventTime();
        if (!next.has_value() || *next < now_ps || ++iterations > 1000) {
            throw std::logic_error("host-memory fixture lost event progress");
        }
        now_ps = *next;
        static_cast<void>(device.progress(now_ps));
        std::vector<CompletionEntry> polled = device.pollCompletionQueue(
            std::numeric_limits<std::size_t>::max(), now_ps);
        completions.insert(
            completions.end(), polled.begin(), polled.end());
    }
    if (completions.size() != batch_size) {
        throw std::logic_error("host-memory fixture completion count failed");
    }

    device.validateInvariants();
    const auto& accesses = device.memoryAccesses();
    const auto* fabric = device.pcieFabric();
    if (fabric == nullptr) {
        throw std::logic_error("host-memory fixture lacks PCIe fabric");
    }

    StudyRow row;
    row.page_size_bytes = page_size_bytes;
    row.batch_size = batch_size;
    row.qpc_fetches = countObject(accesses, HostMemoryObjectKind::QpcIcm);
    row.qpc_icm_transactions =
        fabric->accounting(PcieServiceClass::QpcIcm).transactions;
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
    row.sq_page_list_events = countStage(
        accesses,
        HostMemoryObjectKind::SqRing,
        HostMemoryTranslationStage::QueuePageList);
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
    row.cq_page_list_events = countStage(
        accesses,
        HostMemoryObjectKind::CqRing,
        HostMemoryTranslationStage::QueuePageList);
    row.mtt_mpt_transactions =
        fabric->accounting(PcieServiceClass::MttMpt).transactions;
    row.wqe_read_transactions =
        fabric->accounting(PcieServiceClass::WqeRead).transactions;
    row.payload_read_transactions =
        fabric->accounting(PcieServiceClass::PayloadRead).transactions;
    row.cqe_write_transactions =
        fabric->accounting(PcieServiceClass::CqeWrite).transactions;
    row.doorbell_record_transactions =
        fabric->accounting(PcieServiceClass::DoorbellRecord).transactions;
    row.uar_transactions =
        fabric->accounting(PcieServiceClass::UarDoorbell).transactions;
    row.registration_events = static_cast<std::size_t>(std::count_if(
        memory->lifecycleEvents().begin(),
        memory->lifecycleEvents().end(),
        [](const auto& event) {
            return event.kind == HostMemoryLifecycleKind::Registration;
        }));
    row.second_data_page_selected = std::any_of(
        accesses.begin(), accesses.end(), [](const auto& access) {
            return access.object_kind == HostMemoryObjectKind::DataRegion
                && access.page_index == 1;
        });

    device.teardownHostMemory(now_ps);
    device.validateInvariants();
    row.teardown_events = static_cast<std::size_t>(std::count_if(
        memory->lifecycleEvents().begin(),
        memory->lifecycleEvents().end(),
        [](const auto& event) {
            return event.kind == HostMemoryLifecycleKind::Teardown;
        }));
    bool rejected_after_teardown = false;
    try {
        static_cast<void>(device.progress(now_ps));
    } catch (const std::logic_error&) {
        rejected_after_teardown = true;
    }
    if (!rejected_after_teardown) {
        throw std::logic_error("host-memory fixture accepted use after teardown");
    }
    row.invariants_valid = true;
    return row;
}

void checkStudyRow(TestRunner& test, const StudyRow& row) {
    const std::size_t batch = row.batch_size;
    test.check(row.qpc_fetches == batch, "QPC fetch count");
    test.check(row.qpc_icm_transactions == batch, "QpcIcm count");
    test.check(row.qpc_mkey_events == 0, "QPC has no MKey event");
    test.check(row.qpc_mpt_events == 0, "QPC has no MPT event");
    test.check(row.qpc_mtt_events == 0, "QPC has no MTT event");
    test.check(row.sq_page_list_events == batch, "SQ page-list count");
    test.check(row.data_mkey_events == batch, "data MKey count");
    test.check(row.data_mpt_events == batch, "data MPT count");
    test.check(row.data_mtt_events == batch, "data MTT count");
    test.check(row.cq_page_list_events == batch, "CQ page-list count");
    test.check(row.mtt_mpt_transactions == 4 * batch, "MttMpt count");
    test.check(row.wqe_read_transactions == batch, "WqeRead count");
    test.check(row.payload_read_transactions == batch, "PayloadRead count");
    test.check(row.cqe_write_transactions == batch, "CqeWrite count");
    test.check(row.doorbell_record_transactions == 1, "doorbell count");
    test.check(row.uar_transactions == 1, "UAR count");
    test.check(row.registration_events == 6, "registration event count");
    test.check(row.teardown_events == 6, "teardown event count");
    test.check(row.second_data_page_selected, "second data page selected");
    test.check(row.invariants_valid, "fixture invariants valid");
}

void testRegistryTransactions(TestRunner& test) {
    VirtualHostMemory memory;
    std::vector<HostMemoryAllocation> allocations = makeAllocations(4096);
    auto plan = memory.planRegistrations(allocations, 3);
    test.check(memory.liveAllocationCount() == 0, "plan is non-mutating");
    memory.commit(std::move(plan));
    memory.validateInvariants();
    test.check(memory.liveAllocationCount() == 6, "registration committed");
    test.check(memory.lifecycleEvents().size() == 6, "registration ledger");

    const std::uint64_t generation = memory.generation();
    const std::size_t event_count = memory.lifecycleEvents().size();
    std::vector<HostMemoryAllocation> invalid;
    invalid.push_back(makeAllocation(
        100,
        HostMemoryObjectKind::QpcIcm,
        HostMemoryOwnerKind::QueuePair,
        kQpn,
        4096,
        20,
        1,
        256));
    invalid.push_back(allocations.front());
    test.expectThrowAs<std::invalid_argument>(
        [&]() { static_cast<void>(memory.planRegistrations(invalid, 4)); },
        "invalid registration batch rejects atomically");
    test.check(
        memory.generation() == generation
            && memory.liveAllocationCount() == 6
            && memory.lifecycleEvents().size() == event_count,
        "failed registration preserves registry state");

    test.expectThrowAs<std::invalid_argument>(
        [&]() { memory.teardownOwner(123456, 4); },
        "foreign-owner teardown rejects");
    test.check(
        memory.generation() == generation
            && memory.lifecycleEvents().size() == event_count,
        "failed teardown preserves registry state");

    memory.teardownOwner(kDeviceOwner, 5);
    memory.validateInvariants();
    test.check(memory.liveAllocationCount() == 0, "teardown removes owner");
    test.check(memory.lifecycleEvents().size() == 12, "teardown ledger");
    test.expectThrowAs<std::out_of_range>(
        [&]() { static_cast<void>(memory.allocation(kQpcAllocation)); },
        "torn-down allocation rejects lookup");
}

void testStaleRegistrationPlan(TestRunner& test) {
    VirtualHostMemory memory;
    const HostMemoryAllocation first = makeAllocation(
        201,
        HostMemoryObjectKind::QpcIcm,
        HostMemoryOwnerKind::QueuePair,
        kQpn,
        4096,
        30,
        1,
        256);
    const HostMemoryAllocation second = makeAllocation(
        202,
        HostMemoryObjectKind::SqRing,
        HostMemoryOwnerKind::SendQueue,
        kSqId,
        4096,
        31,
        1,
        1024);
    auto first_plan = memory.planRegistrations({first}, 0);
    auto stale_plan = memory.planRegistrations({second}, 0);
    memory.commit(std::move(first_plan));
    test.expectThrowAs<std::logic_error>(
        [&]() { memory.commit(std::move(stale_plan)); },
        "stale registration plan rejects");
    test.check(
        memory.liveAllocationCount() == 1 && memory.generation() == 1,
        "stale plan preserves committed generation");
}

void testMultiBatchTeardownCapacity(TestRunner& test) {
    VirtualHostMemory memory;
    const HostMemoryAllocation first = makeAllocation(
        301,
        HostMemoryObjectKind::QpcIcm,
        HostMemoryOwnerKind::QueuePair,
        kQpn,
        4096,
        40,
        1,
        256);
    const HostMemoryAllocation second = makeAllocation(
        302,
        HostMemoryObjectKind::SqRing,
        HostMemoryOwnerKind::SendQueue,
        kSqId,
        4096,
        41,
        1,
        1024);
    auto first_plan = memory.planRegistrations({first}, 0);
    memory.commit(std::move(first_plan));
    auto second_plan = memory.planRegistrations({second}, 1);
    memory.commit(std::move(second_plan));
    memory.teardownOwner(kDeviceOwner, 2);
    memory.validateInvariants();
    test.check(
        memory.liveAllocationCount() == 0
            && memory.lifecycleEvents().size() == 4,
        "multi-batch registration reserves atomic teardown evidence");
}

void testConstructionRollback(TestRunner& test) {
    auto shared = std::make_shared<VirtualHostMemory>();
    HostMemoryAllocation conflict = makeAllocations(4096).front();
    conflict.allocation_id = 999;
    conflict.device_owner_id = 901;
    auto seed = shared->planRegistrations({conflict}, 0);
    shared->commit(std::move(seed));
    const std::uint64_t generation = shared->generation();
    const std::size_t events = shared->lifecycleEvents().size();

    RnicDeviceAttachments attachments;
    attachments.shared_host_memory = shared;
    test.expectThrowAs<std::invalid_argument>(
        [&]() {
            RnicDevice device(hostMemoryConfig(4096), attachments);
        },
        "device registration conflict rejects");
    test.check(
        shared->generation() == generation
            && shared->liveAllocationCount() == 1
            && shared->lifecycleEvents().size() == events,
        "failed device construction preserves shared registry");
    shared->teardownOwner(901, 0);
    {
        RnicDeviceAttachments retry_attachments;
        retry_attachments.shared_host_memory = shared;
        RnicDevice retry(hostMemoryConfig(4096), retry_attachments);
        test.check(
            shared->liveAllocationCount(kDeviceOwner) == 6,
            "failed construction releases its device-owner claim");
    }
}

void testDeviceOwnerClaims(TestRunner& test) {
    auto shared = std::make_shared<VirtualHostMemory>();
    {
        RnicDeviceAttachments first_attachments;
        first_attachments.shared_host_memory = shared;
        RnicDevice first(hostMemoryConfig(4096), first_attachments);
        const std::uint64_t generation = shared->generation();
        const std::size_t events = shared->lifecycleEvents().size();

        RnicDeviceAttachments duplicate_attachments;
        duplicate_attachments.shared_host_memory = shared;
        test.expectThrowAs<std::invalid_argument>(
            [&]() {
                RnicDevice duplicate(
                    secondHostMemoryConfig(kDeviceOwner),
                    duplicate_attachments);
            },
            "duplicate device-owner claim rejects before registration");
        test.check(
            shared->generation() == generation
                && shared->liveAllocationCount() == 6
                && shared->liveAllocationCount(kDeviceOwner) == 6
                && shared->lifecycleEvents().size() == events,
            "duplicate owner claim preserves every live allocation");

        test.expectThrowAs<std::invalid_argument>(
            [&]() { shared->teardownOwner(kDeviceOwner, 0); },
            "foreign teardown of a claimed owner rejects");
        test.check(
            shared->generation() == generation
                && shared->liveAllocationCount() == 6
                && shared->lifecycleEvents().size() == events,
            "foreign claimed-owner teardown is atomic");

        first.teardownHostMemory(0);
        test.check(
            shared->liveAllocationCount() == 0,
            "owning device teardown removes only its allocations");
    }

    {
        RnicDeviceAttachments replacement_attachments;
        replacement_attachments.shared_host_memory = shared;
        RnicDevice replacement(
            hostMemoryConfig(4096), replacement_attachments);
        test.check(
            shared->liveAllocationCount(kDeviceOwner) == 6,
            "explicit teardown releases the owner identity for reuse");
    }
    test.check(
        shared->liveAllocationCount() == 0,
        "teardown followed by both destructors remains safe");
}

void testCrossDeviceMkeyIsolation(TestRunner& test) {
    auto shared = std::make_shared<VirtualHostMemory>();
    RnicDeviceAttachments first_attachments;
    first_attachments.shared_host_memory = shared;
    RnicDevice first(hostMemoryConfig(4096), first_attachments);

    constexpr HostMemoryDeviceOwnerId second_owner = 901;
    RnicDeviceConfig second_config = secondHostMemoryConfig(second_owner);
    RnicDeviceAttachments second_attachments;
    second_attachments.shared_host_memory = shared;
    RnicDevice second(second_config, second_attachments);
    const std::uint64_t generation = shared->generation();

    WorkRequest foreign;
    foreign.wr_id = 1;
    foreign.flow_id = 1;
    foreign.destination = 2;
    foreign.payload_bytes = 64;
    foreign.data_memory = WorkRequestDataMemory{};
    foreign.data_memory->allocation_id = kDataAllocation;
    foreign.data_memory->mkey = kMkey;
    foreign.data_memory->allocation_offset_bytes = 4096;
    test.expectThrowAs<std::invalid_argument>(
        [&]() { static_cast<void>(second.postSend(foreign, 0)); },
        "WQE rejects another device's allocation in the same MKey namespace");
    test.check(
        shared->generation() == generation
            && shared->liveAllocationCount() == 12
            && first.records().empty() && second.records().empty(),
        "cross-device MKey rejection preserves both devices");
    first.validateInvariants();
    second.validateInvariants();
    first.teardownHostMemory(0);
    test.check(
        shared->liveAllocationCount() == 6
            && shared->liveAllocationCount(kDeviceOwner) == 0
            && shared->liveAllocationCount(second_owner) == 6,
        "one owner teardown preserves the other device's allocations");
    second.validateInvariants();
}

void testRequestRejectionAtomicity(TestRunner& test) {
    RnicDevice device(hostMemoryConfig(4096));
    const std::uint64_t fabric_generation = device.pcieFabric()->generation();
    const std::size_t accesses = device.memoryAccesses().size();
    const std::size_t lifecycle = device.hostMemory()->lifecycleEvents().size();

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
        [&]() { static_cast<void>(device.postSend(invalid, 10)); },
        "bad data descriptor rejects");
    test.check(
        device.records().empty()
            && device.pcieFabric()->generation() == fabric_generation
            && device.memoryAccesses().size() == accesses
            && device.hostMemory()->lifecycleEvents().size() == lifecycle,
        "bad descriptor preserves queue, fabric and registry");

    WorkRequest valid = invalid;
    valid.data_memory->mkey = kMkey;
    test.check(
        device.postSend(valid, 5).status == PostStatus::Accepted,
        "failed post does not advance caller time");
}

void testDisabledIdentityMode(TestRunner& test) {
    RnicDevice device(RnicDeviceConfig{});
    test.check(device.hostMemory() == nullptr, "default registry is absent");
    test.check(device.memoryAccesses().empty(), "default access ledger empty");
    test.check(
        device.stageReport().pcie_qpc_icm
                == RnicStageApplicability::NotApplicable
            && device.stageReport().pcie_mtt_mpt
                == RnicStageApplicability::NotApplicable
            && device.stageReport().pcie_payload_read
                == RnicStageApplicability::NotApplicable,
        "default host-memory stages are not applicable");
    device.validateInvariants();
}

void testEffectiveHardwareRecord(TestRunner& test) {
    RnicDevice small_pages(hostMemoryConfig(4096));
    RnicDevice huge_pages(hostMemoryConfig(2097152));
    const auto record = makeStructuralSessionConfigRecord(
        "host-memory", "rnic-nn", small_pages);
    test.check(
        record.effective_hardware_json.has_value()
            && record.effective_hardware_json->find(
                   "simllm-rnic-effective-hardware-v3")
                != std::string::npos
            && record.effective_hardware_json->find("\"host_memory\"")
                != std::string::npos
            && record.effective_hardware_json->find("\"submission\"")
                != std::string::npos,
        "enabled host memory uses the strict effective-hardware v3 record");
    test.check(
        effectiveHardwareConfigSha256(small_pages)
            != effectiveHardwareConfigSha256(huge_pages),
        "page geometry contributes to effective hardware identity");
}

std::vector<StudyRow> studyRows() {
    std::vector<StudyRow> rows;
    for (const std::uint64_t page_size : {UINT64_C(4096), UINT64_C(2097152)}) {
        for (const std::size_t batch_size : {std::size_t{1}, std::size_t{4}}) {
            rows.push_back(runFixture(page_size, batch_size));
        }
    }
    return rows;
}

void printStudyCsv(const std::vector<StudyRow>& rows) {
    std::cout
        << "page_size_bytes,batch_size,qpc_fetches,qpc_icm_transactions,"
           "qpc_mkey_events,qpc_mpt_events,qpc_mtt_events,"
           "sq_page_list_events,data_mkey_events,data_mpt_events,"
           "data_mtt_events,cq_page_list_events,mtt_mpt_transactions,"
           "wqe_read_transactions,payload_read_transactions,"
           "cqe_write_transactions,doorbell_record_transactions,"
           "uar_transactions,registration_events,teardown_events,"
           "second_data_page_selected,invariants_valid\n";
    for (const StudyRow& row : rows) {
        std::cout << row.page_size_bytes << ',' << row.batch_size << ','
                  << row.qpc_fetches << ',' << row.qpc_icm_transactions << ','
                  << row.qpc_mkey_events << ',' << row.qpc_mpt_events << ','
                  << row.qpc_mtt_events << ',' << row.sq_page_list_events << ','
                  << row.data_mkey_events << ',' << row.data_mpt_events << ','
                  << row.data_mtt_events << ',' << row.cq_page_list_events << ','
                  << row.mtt_mpt_transactions << ','
                  << row.wqe_read_transactions << ','
                  << row.payload_read_transactions << ','
                  << row.cqe_write_transactions << ','
                  << row.doorbell_record_transactions << ','
                  << row.uar_transactions << ',' << row.registration_events
                  << ',' << row.teardown_events << ','
                  << static_cast<int>(row.second_data_page_selected) << ','
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
            std::cerr << "usage: simllm_rnic_host_memory_test [--study-csv]\n";
            return 2;
        }

        TestRunner test;
        for (const StudyRow& row : rows) {
            checkStudyRow(test, row);
        }
        testRegistryTransactions(test);
        testStaleRegistrationPlan(test);
        testMultiBatchTeardownCapacity(test);
        testConstructionRollback(test);
        testDeviceOwnerClaims(test);
        testCrossDeviceMkeyIsolation(test);
        testRequestRejectionAtomicity(test);
        testDisabledIdentityMode(test);
        testEffectiveHardwareRecord(test);
        if (test.failures() != 0) {
            std::cerr << test.failures() << " host-memory checks failed\n";
            return 1;
        }
        std::cout << "RNIC host-memory checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unexpected host-memory failure: " << error.what() << '\n';
        return 1;
    }
}
