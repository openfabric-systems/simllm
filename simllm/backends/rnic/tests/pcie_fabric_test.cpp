#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

#include "simllm/rnic/pcie_fabric.h"

namespace {

using simllm::rnic::PcieDirection;
using simllm::rnic::PcieClassAccounting;
using simllm::rnic::PcieFabric;
using simllm::rnic::PcieFabricConfig;
using simllm::rnic::PcieGeneration;
using simllm::rnic::PcieOperation;
using simllm::rnic::PcieOrdering;
using simllm::rnic::PcieServiceClass;
using simllm::rnic::PcieTransactionRequest;
using simllm::rnic::PcieTransactionResult;
using simllm::rnic::Picoseconds;
using simllm::rnic::defaultPcieFabricConfig;
using simllm::rnic::kPcieServiceClassCount;
using simllm::rnic::kPcieTransactionAbiVersion;

static_assert(!std::is_copy_constructible_v<PcieFabric>);
static_assert(!std::is_copy_assignable_v<PcieFabric>);
static_assert(!std::is_move_constructible_v<PcieFabric>);
static_assert(!std::is_move_assignable_v<PcieFabric>);

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

PcieTransactionRequest transaction(
    PcieOperation operation,
    std::uint64_t bytes,
    PcieDirection direction = PcieDirection::HostToDevice,
    PcieServiceClass service_class = PcieServiceClass::PayloadRead) {
    PcieTransactionRequest request;
    if (service_class == PcieServiceClass::PayloadRead
        && operation == PcieOperation::PostedWrite) {
        request.service_class = PcieServiceClass::PayloadWrite;
    } else if (service_class == PcieServiceClass::PayloadRead
               && operation == PcieOperation::HostStore) {
        request.service_class = PcieServiceClass::DoorbellRecord;
    } else {
        request.service_class = service_class;
    }
    request.operation = operation;
    request.request_direction = direction;
    request.ordering = PcieOrdering::Independent;
    request.path_id = 2;
    request.useful_bytes = bytes;
    request.transfer_bytes = bytes;
    return request;
}

std::uint64_t modeledBytes(const PcieTransactionResult& result) {
    return result.host_to_device.modeled_link_bytes
        + result.device_to_host.modeled_link_bytes;
}

void addAccounting(
    PcieClassAccounting& destination,
    const PcieClassAccounting& source) {
    destination.transactions += source.transactions;
    destination.useful_bytes += source.useful_bytes;
    destination.transferred_bytes += source.transferred_bytes;
    destination.host_store_bytes += source.host_store_bytes;
    destination.host_to_device.tlps += source.host_to_device.tlps;
    destination.host_to_device.payload_bytes +=
        source.host_to_device.payload_bytes;
    destination.host_to_device.overhead_bytes +=
        source.host_to_device.overhead_bytes;
    destination.host_to_device.modeled_link_bytes +=
        source.host_to_device.modeled_link_bytes;
    destination.device_to_host.tlps += source.device_to_host.tlps;
    destination.device_to_host.payload_bytes +=
        source.device_to_host.payload_bytes;
    destination.device_to_host.overhead_bytes +=
        source.device_to_host.overhead_bytes;
    destination.device_to_host.modeled_link_bytes +=
        source.device_to_host.modeled_link_bytes;
    destination.waits.ordering_ps += source.waits.ordering_ps;
    destination.waits.outstanding_read_ps +=
        source.waits.outstanding_read_ps;
    destination.waits.completion_buffer_ps +=
        source.waits.completion_buffer_ps;
    destination.waits.credit_ps += source.waits.credit_ps;
    destination.waits.link_queue_ps += source.waits.link_queue_ps;
    destination.service_delay.host_store_ps +=
        source.service_delay.host_store_ps;
    destination.service_delay.posted_write_visibility_ps +=
        source.service_delay.posted_write_visibility_ps;
    destination.service_delay.read_completion_ps +=
        source.service_delay.read_completion_ps;
    destination.path_delay.base_ps += source.path_delay.base_ps;
    destination.path_delay.numa_ps += source.path_delay.numa_ps;
    destination.path_delay.iommu_ps += source.path_delay.iommu_ps;
    destination.path_delay.acs_ps += source.path_delay.acs_ps;
    destination.path_delay.switch_ps += source.path_delay.switch_ps;
    destination.path_delay.ddio_miss_ps += source.path_delay.ddio_miss_ps;
    destination.path_delay.gpu_direct_ps += source.path_delay.gpu_direct_ps;
    destination.latency_samples_used += source.latency_samples_used;
}

bool sameAccounting(
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
        && lhs.latency_samples_used == rhs.latency_samples_used;
}

void testConfigValidation(TestRunner& test) {
    PcieFabricConfig invalid = defaultPcieFabricConfig();
    invalid.version = 2;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "config version is validated");

    invalid = defaultPcieFabricConfig();
    invalid.max_payload_size_bytes = 192;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "MPS must be a supported power of two");

    invalid = defaultPcieFabricConfig();
    invalid.generation = static_cast<PcieGeneration>(0);
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "PCIe generation is validated");

    invalid = defaultPcieFabricConfig();
    invalid.lane_count = 3;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "PCIe lane width is validated");

    invalid = defaultPcieFabricConfig();
    invalid.max_read_request_size_bytes = 64;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "MRRS is validated independently from MPS");

    invalid = defaultPcieFabricConfig();
    invalid.read_completion_boundary_bytes = 32;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "read completion boundary is validated");

    invalid = defaultPcieFabricConfig();
    invalid.paths.push_back(invalid.paths.front());
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "path IDs are unique");

    invalid = defaultPcieFabricConfig();
    invalid.read_completion_latency_ps.samples_ps = {10, 20};
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric fabric(invalid); },
        "v1 rejects variable latency before FIFO reservation can create HOL");

    PcieFabric fabric(defaultPcieFabricConfig());
    PcieTransactionRequest mismatched = transaction(
        PcieOperation::PostedWrite, 8);
    mismatched.service_class = PcieServiceClass::PayloadRead;
    test.expectThrowAs<std::invalid_argument>(
        [&fabric, &mismatched]() { (void)fabric.submit(mismatched); },
        "payload read/write labels cannot silently cross operations");

    invalid = defaultPcieFabricConfig();
    invalid.host_to_device_credits.posted_data_credits = 1;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric invalid_fabric(invalid); },
        "config rejects a data-credit pool smaller than one MPS TLP");

    invalid = defaultPcieFabricConfig();
    const Picoseconds maximum_duration =
        static_cast<Picoseconds>(std::numeric_limits<std::int64_t>::max());
    invalid.paths[1].base_latency_ps = maximum_duration;
    invalid.paths[1].numa_penalty_ps = maximum_duration;
    invalid.paths[1].iommu_penalty_ps = maximum_duration;
    test.expectThrowAs<std::invalid_argument>(
        [&invalid]() { PcieFabric invalid_fabric(invalid); },
        "aggregate path-delay overflow is rejected at construction");

    PcieFabric transaction_validation(defaultPcieFabricConfig());
    PcieTransactionRequest invalid_request = transaction(
        PcieOperation::PostedWrite, 8);
    invalid_request.abi_version = kPcieTransactionAbiVersion + 1;
    test.expectThrowAs<std::invalid_argument>(
        [&transaction_validation, &invalid_request]() {
            (void)transaction_validation.submit(invalid_request);
        },
        "transaction ABI version is validated");
    invalid_request = transaction(PcieOperation::PostedWrite, 8);
    invalid_request.useful_bytes = 0;
    invalid_request.transfer_bytes = 0;
    test.expectThrowAs<std::invalid_argument>(
        [&transaction_validation, &invalid_request]() {
            (void)transaction_validation.submit(invalid_request);
        },
        "zero transaction sizes are rejected");
    invalid_request = transaction(
        PcieOperation::HostStore,
        std::numeric_limits<std::uint64_t>::max());
    invalid_request.first_byte_offset = 4095;
    test.expectThrowAs<std::overflow_error>(
        [&transaction_validation, &invalid_request]() {
            (void)transaction_validation.submit(invalid_request);
        },
        "first address plus transfer span overflow is rejected");
    test.check(
        transaction_validation.totalAccounting().transactions == 0,
        "invalid transactions leave the fabric ledger unchanged");

    invalid = defaultPcieFabricConfig();
    invalid.paths[1].enabled = false;
    PcieFabric disabled_path(invalid);
    invalid_request = transaction(PcieOperation::PostedWrite, 8);
    test.expectThrowAs<std::invalid_argument>(
        [&disabled_path, &invalid_request]() {
            (void)disabled_path.submit(invalid_request);
        },
        "disabled transaction path is rejected");
    test.check(
        disabled_path.totalAccounting().transactions == 0,
        "disabled-path rejection leaves the ledger unchanged");
}

void testPostedWriteSegmentationAndSerialization(TestRunner& test) {
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.generation = PcieGeneration::Gen5;
    config.lane_count = 16;
    config.max_payload_size_bytes = 256;
    PcieFabric fabric(config);
    const PcieTransactionResult result = fabric.submit(transaction(
        PcieOperation::PostedWrite,
        4096,
        PcieDirection::DeviceToHost));
    test.check(
        result.request_tlps == 16
            && result.device_to_host.payload_bytes == 4096
            && result.device_to_host.overhead_bytes == 16 * 24
            && modeledBytes(result) == 4480,
        "aligned 4 KiB MWr has exact MPS and overhead accounting");
    test.check(
        result.completed_at_ps == 71094,
        "Gen5 x16 stream uses one rational final rounding");
    test.check(
        !result.completion_ready_at_ps.has_value(),
        "posted write exposes modeled visibility without a fake completion");
    fabric.validateInvariants();

    config = defaultPcieFabricConfig();
    config.max_payload_size_bytes = 128;
    PcieFabric unaligned(config);
    PcieTransactionRequest crossing = transaction(
        PcieOperation::PostedWrite, 2);
    crossing.first_byte_offset = 4095;
    crossing.useful_bytes = 2;
    const PcieTransactionResult split = unaligned.submit(crossing);
    test.check(
        split.request_tlps == 2
            && split.host_to_device.payload_bytes == 8
            && split.useful_bytes == 2
            && split.transferred_bytes == 2,
        "4 KiB crossing and DWORD byte enables are both charged");

    PcieFabric dword(defaultPcieFabricConfig());
    PcieTransactionRequest offset = transaction(
        PcieOperation::PostedWrite, 3);
    offset.first_byte_offset = 3;
    const PcieTransactionResult padded = dword.submit(offset);
    test.check(
        padded.request_tlps == 1
            && padded.host_to_device.payload_bytes == 8
            && padded.useful_bytes == 3,
        "unaligned three-byte write carries two DWORDs");
}

void testReadByteMatrixAndDirection(TestRunner& test) {
    constexpr std::array<std::uint32_t, 3> sizes{128, 256, 512};
    constexpr std::array<std::array<std::uint64_t, 3>, 3> expected{{
        {{688, 688, 688}},
        {{640, 600, 600}},
        {{616, 576, 556}},
    }};
    for (std::size_t mrrs_index = 0; mrrs_index < sizes.size(); ++mrrs_index) {
        for (std::size_t mps_index = 0; mps_index < sizes.size(); ++mps_index) {
            PcieFabricConfig config = defaultPcieFabricConfig();
            config.max_payload_size_bytes = sizes[mps_index];
            config.max_read_request_size_bytes = sizes[mrrs_index];
            PcieFabric fabric(config);
            const PcieTransactionResult result = fabric.submit(transaction(
                PcieOperation::NonPostedRead,
                512,
                PcieDirection::DeviceToHost));
            test.check(
                modeledBytes(result) == expected[mrrs_index][mps_index],
                "MPS/MRRS matrix has exact modeled-link bytes");
            test.check(
                result.device_to_host.payload_bytes == 0
                    && result.device_to_host.tlps
                        == result.request_tlps
                    && result.host_to_device.payload_bytes == 512
                    && result.host_to_device.tlps
                        == result.completion_tlps,
                "DMA read keeps request headers and CplD data directional");
            test.check(
                result.completion_ready_at_ps.has_value(),
                "non-posted read exposes its first response-ready timestamp");
            fabric.validateInvariants();
        }
    }

    PcieFabricConfig rcb_config = defaultPcieFabricConfig();
    rcb_config.max_payload_size_bytes = 128;
    rcb_config.max_read_request_size_bytes = 128;
    rcb_config.read_completion_boundary_bytes = 64;
    PcieFabric rcb_fabric(rcb_config);
    PcieTransactionRequest rcb = transaction(
        PcieOperation::NonPostedRead,
        8,
        PcieDirection::DeviceToHost);
    rcb.first_byte_offset = 60;
    const PcieTransactionResult rcb_result = rcb_fabric.submit(rcb);
    test.check(
        rcb_result.request_tlps == 1
            && rcb_result.completion_tlps == 2
            && rcb_result.host_to_device.payload_bytes == 8,
        "read completion boundary splits one MRd response");
}

void testHostStoreAndClassLedgers(TestRunner& test) {
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.host_store_latency_ps.samples_ps = {10};
    config.paths[1].base_latency_ps = 5;
    PcieFabric fabric(config);
    for (std::size_t index = 0; index < kPcieServiceClassCount; ++index) {
        const auto service_class = static_cast<PcieServiceClass>(index);
        PcieOperation operation = PcieOperation::HostStore;
        if (service_class == PcieServiceClass::PayloadRead) {
            operation = PcieOperation::NonPostedRead;
        } else if (service_class == PcieServiceClass::PayloadWrite) {
            operation = PcieOperation::PostedWrite;
        }
        const PcieDirection direction =
            service_class == PcieServiceClass::PayloadRead
            ? PcieDirection::DeviceToHost
            : PcieDirection::HostToDevice;
        PcieTransactionRequest request = transaction(
            operation,
            8,
            direction,
            service_class);
        request.ordering_domain = index;
        const PcieTransactionResult result = fabric.submit(request);
        if (operation == PcieOperation::HostStore) {
            test.check(
                result.host_store_bytes == 8
                    && modeledBytes(result) == 0
                    && result.completed_at_ps == 15
                    && !result.completion_ready_at_ps.has_value(),
                "host store has host bytes, latency and zero PCIe bytes");
        }
    }
    const auto total = fabric.totalAccounting();
    const std::uint64_t host_store_classes = kPcieServiceClassCount - 2;
    test.check(
        total.transactions == kPcieServiceClassCount
            && total.useful_bytes == 8 * kPcieServiceClassCount
            && total.transferred_bytes == 8 * kPcieServiceClassCount
            && total.host_store_bytes == 8 * host_store_classes
            && total.host_to_device.tlps == 2
            && total.host_to_device.payload_bytes == 16
            && total.host_to_device.overhead_bytes == 44
            && total.host_to_device.modeled_link_bytes == 60
            && total.device_to_host.tlps == 1
            && total.device_to_host.payload_bytes == 0
            && total.device_to_host.overhead_bytes == 24
            && total.device_to_host.modeled_link_bytes == 24
            && total.service_delay.host_store_ps
                == 10 * host_store_classes
            && total.service_delay.posted_write_visibility_ps == 0
            && total.service_delay.read_completion_ps == 0
            && total.path_delay.base_ps == 5 * kPcieServiceClassCount
            && total.path_delay.numa_ps == 0
            && total.path_delay.iommu_ps == 0
            && total.path_delay.acs_ps == 0
            && total.path_delay.switch_ps == 0
            && total.path_delay.ddio_miss_ps == 0
            && total.path_delay.gpu_direct_ps == 0
            && total.latency_samples_used == kPcieServiceClassCount,
        "all semantic classes retain expected non-wait accounting");
    PcieClassAccounting manual_total;
    for (std::size_t index = 0; index < kPcieServiceClassCount; ++index) {
        const auto item = fabric.accounting(
            static_cast<PcieServiceClass>(index));
        test.check(
            item.transactions == 1,
            "each semantic class owns one independent ledger row");
        addAccounting(manual_total, item);
    }
    test.check(
        sameAccounting(total, manual_total),
        "per-class totals reconcile every global accounting field");
    fabric.validateInvariants();
}

void testPlanAtomicityAndStaleness(TestRunner& test) {
    PcieFabric fabric(defaultPcieFabricConfig());
    {
        auto discarded = fabric.beginPlan();
        (void)fabric.schedule(
            discarded, transaction(PcieOperation::HostStore, 8));
    }
    test.check(
        fabric.generation() == 0
            && fabric.totalAccounting().transactions == 0,
        "discarded plan mutates neither state nor transaction IDs");

    auto plan = fabric.beginPlan();
    const PcieTransactionResult first = fabric.schedule(
        plan, transaction(PcieOperation::HostStore, 8));
    PcieTransactionRequest invalid = transaction(
        PcieOperation::PostedWrite, 8);
    invalid.path_id = 999;
    test.expectThrowAs<std::invalid_argument>(
        [&fabric, &plan, &invalid]() {
            (void)fabric.schedule(plan, invalid);
        },
        "failed schedule preserves an existing private plan");
    fabric.commit(std::move(plan));
    test.check(
        first.transaction_id == 1
            && fabric.totalAccounting().transactions == 1,
        "valid work before a failed schedule still commits exactly once");

    auto winner = fabric.beginPlan();
    auto stale = fabric.beginPlan();
    (void)fabric.schedule(
        winner, transaction(PcieOperation::HostStore, 8));
    (void)fabric.schedule(
        stale, transaction(PcieOperation::HostStore, 8));
    fabric.commit(std::move(winner));
    test.expectThrowAs<std::logic_error>(
        [&fabric, &stale]() { fabric.commit(std::move(stale)); },
        "stale plan is rejected before shared-state mutation");
    test.check(
        fabric.totalAccounting().transactions == 2,
        "stale commit does not duplicate its transaction");
    fabric.validateInvariants();
}

void testFiniteCreditsAndReadResources(TestRunner& test) {
    PcieFabricConfig credit_config = defaultPcieFabricConfig();
    credit_config.host_to_device_credits.posted_header_credits = 1;
    credit_config.host_to_device_credits.posted_data_credits = 64;
    credit_config.credit_return_latency_ps = 1000;
    PcieFabric credit_fabric(credit_config);
    const PcieTransactionResult first = credit_fabric.submit(transaction(
        PcieOperation::PostedWrite, 64));
    const PcieTransactionResult second = credit_fabric.submit(transaction(
        PcieOperation::PostedWrite, 64));
    test.check(
        first.waits.credit_ps == 0 && second.waits.credit_ps == 1000,
        "posted-header exhaustion waits exactly for configured credit return");
    credit_fabric.validateInvariants();

    PcieFabricConfig posted_data_config = defaultPcieFabricConfig();
    posted_data_config.max_payload_size_bytes = 128;
    posted_data_config.host_to_device_credits.posted_data_credits = 8;
    posted_data_config.credit_return_latency_ps = 1000;
    PcieFabric posted_data_fabric(posted_data_config);
    (void)posted_data_fabric.submit(transaction(
        PcieOperation::PostedWrite, 128));
    const auto posted_data_second = posted_data_fabric.submit(transaction(
        PcieOperation::PostedWrite, 128));
    test.check(
        posted_data_second.waits.credit_ps == 1000,
        "posted-data exhaustion waits for its independent credit return");

    PcieFabricConfig nonposted_config = defaultPcieFabricConfig();
    nonposted_config.device_to_host_credits.nonposted_header_credits = 1;
    nonposted_config.credit_return_latency_ps = 10'000;
    PcieFabric nonposted_fabric(nonposted_config);
    (void)nonposted_fabric.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto nonposted_second = nonposted_fabric.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    test.check(
        nonposted_second.waits.credit_ps == 10'000,
        "non-posted-header exhaustion waits exactly for credit return");

    const auto completion_credit_wait = [](bool limit_headers) {
        PcieFabricConfig config = defaultPcieFabricConfig();
        config.max_payload_size_bytes = 128;
        config.max_read_request_size_bytes = 512;
        config.credit_return_latency_ps = 10'000;
        if (limit_headers) {
            config.host_to_device_credits.completion_header_credits = 1;
        } else {
            config.host_to_device_credits.completion_data_credits = 8;
        }
        PcieFabric fabric(config);
        return fabric.submit(transaction(
            PcieOperation::NonPostedRead,
            512,
            PcieDirection::DeviceToHost)).waits.credit_ps;
    };
    test.check(
        completion_credit_wait(true) == 30'000
            && completion_credit_wait(false) == 30'000,
        "completion header and data credits wait exactly and independently");

    PcieFabricConfig too_small = defaultPcieFabricConfig();
    too_small.completion_buffer_bytes = 256;
    PcieFabric small_buffer(too_small);
    auto buffer_plan = small_buffer.beginPlan();
    test.expectThrowAs<std::invalid_argument>(
        [&small_buffer, &buffer_plan]() {
            (void)small_buffer.schedule(
                buffer_plan,
                transaction(
                    PcieOperation::NonPostedRead,
                    512,
                    PcieDirection::DeviceToHost));
        },
        "undersized completion buffer rejects rather than deadlocking");
    test.check(
        small_buffer.totalAccounting().transactions == 0,
        "resource rejection has no shared accounting side effects");

    PcieFabricConfig contended_buffer_config = defaultPcieFabricConfig();
    contended_buffer_config.max_outstanding_read_requests = 4;
    contended_buffer_config.completion_buffer_bytes = 512;
    contended_buffer_config.read_completion_latency_ps.samples_ps = {
        1'000'000};
    PcieFabric contended_buffer(contended_buffer_config);
    (void)contended_buffer.submit(transaction(
        PcieOperation::NonPostedRead,
        512,
        PcieDirection::DeviceToHost));
    const auto buffer_wait = contended_buffer.submit(transaction(
        PcieOperation::NonPostedRead,
        512,
        PcieDirection::DeviceToHost));
    test.check(
        buffer_wait.waits.completion_buffer_ps > 0,
        "completion buffer releases a full MRd response span at final CplD");

    const auto run_reads = [](std::uint32_t slots, std::uint32_t mps) {
        PcieFabricConfig config = defaultPcieFabricConfig();
        config.max_outstanding_read_requests = slots;
        config.max_payload_size_bytes = mps;
        config.max_read_request_size_bytes = 512;
        config.read_completion_latency_ps.samples_ps = {1'000'000};
        PcieFabric fabric(config);
        PcieTransactionResult last;
        for (std::uint64_t index = 0; index < 16; ++index) {
            PcieTransactionRequest request = transaction(
                PcieOperation::NonPostedRead,
                512,
                PcieDirection::DeviceToHost);
            request.client_token = index + 1;
            last = fabric.submit(request);
        }
        fabric.validateInvariants();
        return std::array<std::uint64_t, 2>{
            last.completed_at_ps,
            fabric.accounting(PcieServiceClass::PayloadRead)
                .waits.outstanding_read_ps,
        };
    };
    for (std::uint32_t mps : {128U, 512U}) {
        const auto one = run_reads(1, mps);
        const auto four = run_reads(4, mps);
        test.check(
            four[0] < one[0] && four[1] < one[1],
            "four read slots reduce JCT and read-window wait");
    }
}

void testOrderingPathAndOverflow(TestRunner& test) {
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.host_store_latency_ps.samples_ps = {100};
    PcieFabric fabric(config);
    PcieTransactionRequest strict = transaction(
        PcieOperation::HostStore, 8);
    strict.ordering = PcieOrdering::VisibilityDependency;
    strict.ordering_domain = 7;
    const auto first = fabric.submit(strict);
    const auto second = fabric.submit(strict);
    PcieTransactionRequest relaxed = strict;
    relaxed.ordering = PcieOrdering::Independent;
    const auto bypass = fabric.submit(relaxed);
    test.check(
        first.completed_at_ps == 100
            && second.waits.ordering_ps == 100
            && second.completed_at_ps == 200
            && bypass.completed_at_ps == 100,
        "visibility dependencies serialize while independent work may bypass");

    const auto run_path = [](Picoseconds numa, Picoseconds iommu) {
        PcieFabricConfig path_config = defaultPcieFabricConfig();
        path_config.paths[1].numa_penalty_ps = numa;
        path_config.paths[1].iommu_penalty_ps = iommu;
        PcieFabric path_fabric(path_config);
        return path_fabric.submit(transaction(
            PcieOperation::NonPostedRead,
            512,
            PcieDirection::DeviceToHost));
    };
    const auto local = run_path(0, 0);
    const auto remote_iommu = run_path(100'000, 200'000);
    test.check(
        remote_iommu.completed_at_ps - local.completed_at_ps == 300'000
            && remote_iommu.path_delay.numa_ps == 100'000
            && remote_iommu.path_delay.iommu_ps == 200'000
            && modeledBytes(remote_iommu) == modeledBytes(local),
        "NUMA and IOMMU delay is exact, separate and byte-neutral");

    PcieFabric overflow(defaultPcieFabricConfig());
    auto overflow_plan = overflow.beginPlan();
    PcieTransactionRequest near_max = transaction(
        PcieOperation::PostedWrite, 8);
    near_max.submitted_at_ps = std::numeric_limits<std::uint64_t>::max();
    test.expectThrowAs<std::overflow_error>(
        [&overflow, &overflow_plan, &near_max]() {
            (void)overflow.schedule(overflow_plan, near_max);
        },
        "serializer timestamp overflow has the asserted error type");
    const auto recovered = overflow.schedule(
        overflow_plan, transaction(PcieOperation::HostStore, 8));
    overflow.commit(std::move(overflow_plan));
    test.check(
        recovered.transaction_id == 1
            && overflow.totalAccounting().transactions == 1,
        "overflow consumes no transaction ID or private-plan state");
    overflow.validateInvariants();
}

void testCrossClassAccountingOverflowIsAtomic(TestRunner& test) {
    constexpr std::uint64_t half_range = std::uint64_t{1} << 63;

    PcieFabric class_overflow(defaultPcieFabricConfig());
    PcieTransactionRequest maximum = transaction(
        PcieOperation::HostStore,
        std::numeric_limits<std::uint64_t>::max(),
        PcieDirection::HostToDevice,
        PcieServiceClass::DoorbellRecord);
    (void)class_overflow.submit(maximum);
    PcieTransactionRequest one_more = maximum;
    one_more.useful_bytes = 1;
    one_more.transfer_bytes = 1;
    test.expectThrowAs<std::overflow_error>(
        [&class_overflow, &one_more]() {
            (void)class_overflow.submit(one_more);
        },
        "per-class byte overflow is rejected before plan commit");
    test.check(
        class_overflow.generation() == 1
            && class_overflow.totalAccounting().transactions == 1
            && class_overflow.accounting(PcieServiceClass::DoorbellRecord)
                    .useful_bytes
                == std::numeric_limits<std::uint64_t>::max(),
        "per-class byte overflow leaves shared ledgers unchanged");
    class_overflow.validateInvariants();

    PcieFabric byte_overflow(defaultPcieFabricConfig());
    PcieTransactionRequest first = transaction(
        PcieOperation::HostStore,
        half_range,
        PcieDirection::HostToDevice,
        PcieServiceClass::DoorbellRecord);
    (void)byte_overflow.submit(first);
    PcieTransactionRequest second = first;
    second.service_class = PcieServiceClass::UarDoorbell;
    test.expectThrowAs<std::overflow_error>(
        [&byte_overflow, &second]() {
            (void)byte_overflow.submit(second);
        },
        "cross-class byte overflow is rejected before plan commit");
    const auto byte_total = byte_overflow.totalAccounting();
    test.check(
        byte_overflow.generation() == 1
            && byte_total.transactions == 1
            && byte_total.useful_bytes == half_range
            && byte_overflow.accounting(PcieServiceClass::UarDoorbell)
                    .transactions
                == 0,
        "cross-class byte overflow leaves shared ledgers unchanged");
    byte_overflow.validateInvariants();

    PcieFabricConfig path_config = defaultPcieFabricConfig();
    const Picoseconds maximum_configured_delay =
        static_cast<Picoseconds>(std::numeric_limits<std::int64_t>::max());
    path_config.paths[1].base_latency_ps = maximum_configured_delay;
    PcieFabric path_overflow(path_config);
    first.useful_bytes = 1;
    first.transfer_bytes = 1;
    second = first;
    second.service_class = PcieServiceClass::UarDoorbell;
    (void)path_overflow.submit(first);
    (void)path_overflow.submit(second);
    test.expectThrowAs<std::overflow_error>(
        [&path_overflow, &second]() {
            (void)path_overflow.submit(second);
        },
        "cross-class path-delay overflow is rejected before plan commit");
    const auto path_total = path_overflow.totalAccounting();
    test.check(
        path_overflow.generation() == 2
            && path_total.transactions == 2
            && path_total.path_delay.base_ps
                == 2 * maximum_configured_delay
            && path_overflow.accounting(PcieServiceClass::UarDoorbell)
                    .transactions
                == 1,
        "cross-class path-delay overflow leaves shared ledgers unchanged");
    path_overflow.validateInvariants();
}

}  // namespace

int main() {
    TestRunner test;
    testConfigValidation(test);
    testPostedWriteSegmentationAndSerialization(test);
    testReadByteMatrixAndDirection(test);
    testHostStoreAndClassLedgers(test);
    testPlanAtomicityAndStaleness(test);
    testFiniteCreditsAndReadResources(test);
    testOrderingPathAndOverflow(test);
    testCrossClassAccountingOverflowIsAtomic(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " PCIe test(s) failed\n";
        return 1;
    }
    std::cout << "all RNIC PCIe fabric tests passed\n";
    return 0;
}
