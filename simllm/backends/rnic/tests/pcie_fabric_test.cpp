#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "simllm/rnic/pcie_fabric.h"

namespace {

using simllm::rnic::PcieDirection;
using simllm::rnic::PcieClassAccounting;
using simllm::rnic::PcieAnalyticalDelayAccounting;
using simllm::rnic::PcieAnalyticalDelayKind;
using simllm::rnic::PcieAnalyticalDelayProfile;
using simllm::rnic::PcieFabric;
using simllm::rnic::PcieFabricConfig;
using simllm::rnic::PcieGeneration;
using simllm::rnic::PcieOperation;
using simllm::rnic::PcieOrdering;
using simllm::rnic::PciePathDelayAccounting;
using simllm::rnic::PciePathPenaltyProfiles;
using simllm::rnic::PciePathProfileAccounting;
using simllm::rnic::PcieServiceClass;
using simllm::rnic::PcieTransactionRequest;
using simllm::rnic::PcieTransactionResult;
using simllm::rnic::Picoseconds;
using simllm::rnic::defaultPcieFabricConfig;
using simllm::rnic::kPcieAnalyticalDelayProfileVersion;
using simllm::rnic::kPcieProbabilityScalePpm;
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

using PenaltyProfileMember =
    PcieAnalyticalDelayProfile PciePathPenaltyProfiles::*;
using PenaltyAccountingMember =
    PcieAnalyticalDelayAccounting PciePathProfileAccounting::*;
using PenaltyDelayMember = Picoseconds PciePathDelayAccounting::*;

constexpr std::array<PenaltyProfileMember, 6> kPenaltyProfileMembers{
    &PciePathPenaltyProfiles::numa,
    &PciePathPenaltyProfiles::iommu,
    &PciePathPenaltyProfiles::acs,
    &PciePathPenaltyProfiles::switch_path,
    &PciePathPenaltyProfiles::ddio_miss,
    &PciePathPenaltyProfiles::gpu_direct,
};

constexpr std::array<PenaltyAccountingMember, 6>
    kPenaltyAccountingMembers{
        &PciePathProfileAccounting::numa,
        &PciePathProfileAccounting::iommu,
        &PciePathProfileAccounting::acs,
        &PciePathProfileAccounting::switch_path,
        &PciePathProfileAccounting::ddio_miss,
        &PciePathProfileAccounting::gpu_direct,
    };

constexpr std::array<PenaltyDelayMember, 6> kPenaltyDelayMembers{
    &PciePathDelayAccounting::numa_ps,
    &PciePathDelayAccounting::iommu_ps,
    &PciePathDelayAccounting::acs_ps,
    &PciePathDelayAccounting::switch_ps,
    &PciePathDelayAccounting::ddio_miss_ps,
    &PciePathDelayAccounting::gpu_direct_ps,
};

PcieAnalyticalDelayProfile fixedProfile(
    Picoseconds mean_ps,
    std::uint32_t incidence_probability_ppm =
        kPcieProbabilityScalePpm) {
    PcieAnalyticalDelayProfile profile;
    profile.kind = PcieAnalyticalDelayKind::Fixed;
    profile.incidence_probability_ppm = incidence_probability_ppm;
    profile.mean_ps = mean_ps;
    return profile;
}

PcieAnalyticalDelayProfile gaussianProfile(
    Picoseconds mean_ps,
    Picoseconds standard_deviation_ps,
    std::uint32_t incidence_probability_ppm =
        kPcieProbabilityScalePpm) {
    PcieAnalyticalDelayProfile profile;
    profile.kind = PcieAnalyticalDelayKind::Gaussian;
    profile.incidence_probability_ppm = incidence_probability_ppm;
    profile.mean_ps = mean_ps;
    profile.standard_deviation_ps = standard_deviation_ps;
    return profile;
}

PcieAnalyticalDelayProfile gaussianTailProfile(
    Picoseconds mean_ps,
    Picoseconds standard_deviation_ps,
    std::uint32_t tail_probability_ppm,
    Picoseconds tail_mean_ps,
    Picoseconds tail_standard_deviation_ps,
    std::uint32_t incidence_probability_ppm =
        kPcieProbabilityScalePpm) {
    PcieAnalyticalDelayProfile profile = gaussianProfile(
        mean_ps, standard_deviation_ps, incidence_probability_ppm);
    profile.kind = PcieAnalyticalDelayKind::GaussianTailMixture;
    profile.tail_probability_ppm = tail_probability_ppm;
    profile.tail_mean_ps = tail_mean_ps;
    profile.tail_standard_deviation_ps = tail_standard_deviation_ps;
    return profile;
}

void addAnalyticalAccounting(
    PcieAnalyticalDelayAccounting& destination,
    const PcieAnalyticalDelayAccounting& source) {
    destination.draws += source.draws;
    destination.occurrences += source.occurrences;
    destination.tail_draws += source.tail_draws;
}

void addPathProfileAccounting(
    PciePathProfileAccounting& destination,
    const PciePathProfileAccounting& source) {
    for (PenaltyAccountingMember member : kPenaltyAccountingMembers) {
        addAnalyticalAccounting(destination.*member, source.*member);
    }
}

bool sameAnalyticalAccounting(
    const PcieAnalyticalDelayAccounting& lhs,
    const PcieAnalyticalDelayAccounting& rhs) {
    return lhs.draws == rhs.draws
        && lhs.occurrences == rhs.occurrences
        && lhs.tail_draws == rhs.tail_draws;
}

bool samePathProfileAccounting(
    const PciePathProfileAccounting& lhs,
    const PciePathProfileAccounting& rhs) {
    for (PenaltyAccountingMember member : kPenaltyAccountingMembers) {
        if (!sameAnalyticalAccounting(lhs.*member, rhs.*member)) {
            return false;
        }
    }
    return true;
}

bool hasAnalyticalAccounting(
    const PcieAnalyticalDelayAccounting& accounting,
    std::uint64_t draws,
    std::uint64_t occurrences,
    std::uint64_t tail_draws) {
    return accounting.draws == draws
        && accounting.occurrences == occurrences
        && accounting.tail_draws == tail_draws;
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
    addPathProfileAccounting(
        destination.path_profiles, source.path_profiles);
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
        && samePathProfileAccounting(lhs.path_profiles, rhs.path_profiles)
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
    invalid.paths[1].analytical_penalties.numa = fixedProfile(
        maximum_duration);
    invalid.paths[1].analytical_penalties.iommu = fixedProfile(
        maximum_duration);
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

void testAnalyticalProfileValidation(TestRunner& test) {
    const auto reject_profile = [&test](
                                    const PcieAnalyticalDelayProfile& profile,
                                    const std::string& message) {
        PcieFabricConfig config = defaultPcieFabricConfig();
        config.paths[1].analytical_penalties.numa = profile;
        test.expectThrowAs<std::invalid_argument>(
            [&config]() { PcieFabric fabric(config); }, message);
    };

    PcieAnalyticalDelayProfile profile = fixedProfile(1);
    profile.version = kPcieAnalyticalDelayProfileVersion + 1;
    reject_profile(profile, "analytical profile version is validated");

    profile = fixedProfile(1);
    profile.kind = static_cast<PcieAnalyticalDelayKind>(255);
    reject_profile(profile, "analytical profile kind is validated");

    profile = fixedProfile(1, kPcieProbabilityScalePpm + 1);
    reject_profile(profile, "incidence probability cannot exceed one");

    profile = PcieAnalyticalDelayProfile{};
    profile.mean_ps = 1;
    reject_profile(profile, "disabled profile rejects active parameters");

    profile = fixedProfile(1, 0);
    reject_profile(
        profile, "active profile cannot silently use zero incidence");

    profile = gaussianTailProfile(10, 1, 500'000, 20, 1);
    profile.tail_probability_ppm = kPcieProbabilityScalePpm + 1;
    reject_profile(profile, "tail probability cannot exceed one");

    profile = fixedProfile(1);
    profile.standard_deviation_ps = 1;
    reject_profile(profile, "fixed profile rejects an active sigma field");

    profile = fixedProfile(1);
    profile.tail_probability_ppm = 1;
    profile.tail_mean_ps = 2;
    profile.tail_standard_deviation_ps = 1;
    reject_profile(profile, "fixed profile rejects active tail fields");

    profile = gaussianProfile(10, 0);
    reject_profile(profile, "Gaussian profile requires positive sigma");

    profile = gaussianProfile(10, 1);
    profile.tail_mean_ps = 20;
    reject_profile(profile, "Gaussian profile rejects inactive tail fields");

    profile = gaussianTailProfile(10, 0, 500'000, 20, 1);
    reject_profile(
        profile, "Gaussian-tail profile requires positive base sigma");

    profile = gaussianTailProfile(10, 1, 500'000, 20, 0);
    reject_profile(
        profile, "Gaussian-tail profile requires positive tail sigma");

    profile = gaussianTailProfile(10, 1, 0, 20, 1);
    reject_profile(
        profile, "Gaussian-tail profile rejects an empty tail mixture");

    profile = gaussianTailProfile(
        10, 1, kPcieProbabilityScalePpm, 20, 1);
    reject_profile(
        profile, "Gaussian-tail profile rejects an all-tail mixture");

    profile = gaussianTailProfile(10, 1, 500'000, 10, 1);
    reject_profile(
        profile, "Gaussian-tail profile requires a larger tail mean");

    const Picoseconds maximum_duration =
        static_cast<Picoseconds>(std::numeric_limits<std::int64_t>::max());
    reject_profile(
        fixedProfile(maximum_duration + 1),
        "analytical duration cannot exceed INT64_MAX");
    reject_profile(
        gaussianProfile(maximum_duration, 1),
        "maximum Gaussian sample cannot overflow the timestamp range");

    PcieFabricConfig aggregate = defaultPcieFabricConfig();
    aggregate.paths[1].base_latency_ps = maximum_duration - 2;
    aggregate.paths[1].analytical_penalties.numa = fixedProfile(2);
    aggregate.paths[1].analytical_penalties.iommu = fixedProfile(1);
    test.expectThrowAs<std::invalid_argument>(
        [&aggregate]() { PcieFabric fabric(aggregate); },
        "aggregate maxima across path components cannot overflow");
}

void testAnalyticalSamplingGoldens(TestRunner& test) {
    constexpr std::uint64_t analytical_seed = 0x123456789abcdef0ULL;
    constexpr std::array<Picoseconds, 8> expected_gaussian{
        64'767,
        98'824,
        86'014,
        109'337,
        79'137,
        127'324,
        102'750,
        135'233,
    };
    PcieFabricConfig gaussian_config = defaultPcieFabricConfig();
    gaussian_config.analytical_seed = analytical_seed;
    gaussian_config.paths[1].analytical_penalties.numa = gaussianProfile(
        100'000, 20'000);
    PcieFabric gaussian(gaussian_config);
    for (std::size_t index = 0; index < expected_gaussian.size(); ++index) {
        const auto result = gaussian.submit(transaction(
            PcieOperation::HostStore, 8));
        test.check(
            result.path_delay.numa_ps == expected_gaussian[index]
                && result.completed_at_ps == expected_gaussian[index]
                && hasAnalyticalAccounting(
                    result.path_profiles.numa, 1, 1, 0),
            "Gaussian counter sample matches its frozen golden at draw "
                + std::to_string(index));
        for (std::size_t component = 1;
             component < kPenaltyAccountingMembers.size();
             ++component) {
            test.check(
                hasAnalyticalAccounting(
                    result.path_profiles.*kPenaltyAccountingMembers[
                        component],
                    1,
                    0,
                    0),
                "inactive profile still records one draw per sample");
        }
    }
    const auto gaussian_total = gaussian.totalAccounting();
    test.check(
        gaussian_total.path_delay.numa_ps == 803'386
            && hasAnalyticalAccounting(
                gaussian_total.path_profiles.numa, 8, 8, 0),
        "Gaussian realized delay and counters accumulate exactly");
    gaussian.validateInvariants();

    constexpr std::array<Picoseconds, 12> expected_tail_mixture{
        268'331,
        227'736,
        93'007,
        248'039,
        89'568,
        113'662,
        0,
        117'617,
        269'558,
        0,
        104'668,
        97'829,
    };
    constexpr std::array<bool, 12> expected_occurrence{
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
        false,
        true,
        true,
    };
    constexpr std::array<bool, 12> expected_tail_draw{
        true,
        true,
        false,
        true,
        false,
        false,
        false,
        false,
        true,
        false,
        false,
        false,
    };
    PcieFabricConfig tail_config = defaultPcieFabricConfig();
    tail_config.analytical_seed = analytical_seed;
    tail_config.paths[1].analytical_penalties.numa = gaussianTailProfile(
        100'000, 10'000, 500'000, 250'000, 20'000, 700'000);
    PcieFabric tail_mixture(tail_config);
    for (std::size_t index = 0;
         index < expected_tail_mixture.size();
         ++index) {
        const auto result = tail_mixture.submit(transaction(
            PcieOperation::HostStore, 8));
        test.check(
            result.path_delay.numa_ps == expected_tail_mixture[index]
                && result.completed_at_ps == expected_tail_mixture[index]
                && hasAnalyticalAccounting(
                    result.path_profiles.numa,
                    1,
                    expected_occurrence[index] ? 1 : 0,
                    expected_tail_draw[index] ? 1 : 0),
            "Gaussian-tail counter sample matches its frozen golden at draw "
                + std::to_string(index));
    }
    const auto tail_total = tail_mixture.totalAccounting();
    const auto tail_class = tail_mixture.accounting(
        PcieServiceClass::DoorbellRecord);
    test.check(
        tail_total.path_delay.numa_ps == 1'630'015
            && hasAnalyticalAccounting(
                tail_total.path_profiles.numa, 12, 10, 4)
            && sameAccounting(tail_total, tail_class),
        "Gaussian-tail realized delay and counters accumulate exactly");
    tail_mixture.validateInvariants();
}

void testEveryComponentSupportsEveryProfileKind(TestRunner& test) {
    constexpr std::array<PcieAnalyticalDelayKind, 3> kinds{
        PcieAnalyticalDelayKind::Fixed,
        PcieAnalyticalDelayKind::Gaussian,
        PcieAnalyticalDelayKind::GaussianTailMixture,
    };
    for (PcieAnalyticalDelayKind kind : kinds) {
        PcieAnalyticalDelayProfile profile;
        switch (kind) {
        case PcieAnalyticalDelayKind::Fixed:
            profile = fixedProfile(100'000);
            break;
        case PcieAnalyticalDelayKind::Gaussian:
            profile = gaussianProfile(100'000, 10'000);
            break;
        case PcieAnalyticalDelayKind::GaussianTailMixture:
            profile = gaussianTailProfile(
                100'000, 10'000, 500'000, 250'000, 20'000);
            break;
        default:
            test.check(false, "test profile kind is valid");
            continue;
        }

        PcieFabricConfig config = defaultPcieFabricConfig();
        config.analytical_seed = 0x42ULL;
        for (PenaltyProfileMember member : kPenaltyProfileMembers) {
            config.paths[1].analytical_penalties.*member = profile;
        }
        PcieFabric fabric(config);
        const auto result = fabric.submit(transaction(
            PcieOperation::HostStore, 8));
        for (std::size_t component = 0;
             component < kPenaltyProfileMembers.size();
             ++component) {
            const Picoseconds delay =
                result.path_delay.*kPenaltyDelayMembers[component];
            const auto& accounting =
                result.path_profiles.*kPenaltyAccountingMembers[component];
            test.check(
                delay > 0
                    && hasAnalyticalAccounting(
                        accounting,
                        1,
                        1,
                        accounting.tail_draws),
                "path component supports profile kind "
                    + std::to_string(static_cast<unsigned>(kind))
                    + " at component " + std::to_string(component));
            test.check(
                accounting.tail_draws
                    <= (kind
                            == PcieAnalyticalDelayKind::GaussianTailMixture
                        ? 1U
                        : 0U),
                "only the mixture kind can report a tail draw");
            if (kind == PcieAnalyticalDelayKind::Fixed) {
                test.check(
                    delay == 100'000,
                    "fixed component profile realizes its exact mean");
            }
        }
        fabric.validateInvariants();
    }
}

void testFragmentSamplingAndAccounting(TestRunner& test) {
    constexpr std::array<Picoseconds, 6> component_delays{
        11, 22, 33, 44, 55, 66};
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.max_payload_size_bytes = 128;
    config.max_read_request_size_bytes = 256;
    for (std::size_t component = 0;
         component < kPenaltyProfileMembers.size();
         ++component) {
        config.paths[1].analytical_penalties.*kPenaltyProfileMembers[
            component] = fixedProfile(component_delays[component]);
    }
    PcieFabric fabric(config);

    const auto posted = fabric.submit(transaction(
        PcieOperation::PostedWrite, 512));
    const auto read = fabric.submit(transaction(
        PcieOperation::NonPostedRead,
        512,
        PcieDirection::DeviceToHost));
    test.check(
        posted.request_tlps == 4 && posted.latency_samples_used == 4,
        "posted write consumes one profile draw per MWr fragment");
    test.check(
        read.request_tlps == 2
            && read.completion_tlps == 4
            && read.latency_samples_used == 2,
        "read consumes profile draws per MRd and not per CplD");

    const auto posted_class = fabric.accounting(
        PcieServiceClass::PayloadWrite);
    const auto read_class = fabric.accounting(PcieServiceClass::PayloadRead);
    const auto total = fabric.totalAccounting();
    for (std::size_t component = 0;
         component < kPenaltyProfileMembers.size();
         ++component) {
        const PenaltyDelayMember delay_member =
            kPenaltyDelayMembers[component];
        const PenaltyAccountingMember accounting_member =
            kPenaltyAccountingMembers[component];
        const Picoseconds unit = component_delays[component];
        test.check(
            posted.path_delay.*delay_member == 4 * unit
                && hasAnalyticalAccounting(
                    posted.path_profiles.*accounting_member, 4, 4, 0),
            "posted transaction reports exact realized component delay");
        test.check(
            read.path_delay.*delay_member == 2 * unit
                && hasAnalyticalAccounting(
                    read.path_profiles.*accounting_member, 2, 2, 0),
            "read transaction reports exact realized component delay");
        test.check(
            posted_class.path_delay.*delay_member == 4 * unit
                && hasAnalyticalAccounting(
                    posted_class.path_profiles.*accounting_member,
                    4,
                    4,
                    0),
            "posted class ledger preserves component delay and counters");
        test.check(
            read_class.path_delay.*delay_member == 2 * unit
                && hasAnalyticalAccounting(
                    read_class.path_profiles.*accounting_member, 2, 2, 0),
            "read class ledger preserves component delay and counters");
        test.check(
            total.path_delay.*delay_member == 6 * unit
                && hasAnalyticalAccounting(
                    total.path_profiles.*accounting_member, 6, 6, 0),
            "global ledger sums component delay and counters exactly");
    }
    fabric.validateInvariants();

    PcieFabricConfig readiness_config = defaultPcieFabricConfig();
    readiness_config.max_payload_size_bytes = 128;
    readiness_config.max_read_request_size_bytes = 128;
    readiness_config.paths[1].analytical_penalties.numa = gaussianProfile(
        100'000, 100'000);
    PcieFabric readiness(readiness_config);
    const auto variable_read = readiness.submit(transaction(
        PcieOperation::NonPostedRead,
        256,
        PcieDirection::DeviceToHost));
    test.check(
        variable_read.request_tlps == 2
            && variable_read.path_delay.numa_ps == 156'544
            && variable_read.completion_ready_at_ps == 75'023,
        "read reports the earliest response-ready MRd under variable delay");
    readiness.validateInvariants();
}

void testLinkQueueCountsOnlyExternalContention(TestRunner& test) {
    PcieFabricConfig multi_mwr_config = defaultPcieFabricConfig();
    multi_mwr_config.max_payload_size_bytes = 128;
    PcieFabric multi_mwr(multi_mwr_config);
    const auto mwr = multi_mwr.submit(transaction(
        PcieOperation::PostedWrite, 512));
    test.check(
        mwr.request_tlps == 4
            && mwr.completion_tlps == 0
            && mwr.waits.link_queue_ps == 0,
        "one uncontended multi-MWr transaction has zero link-queue wait");
    multi_mwr.validateInvariants();

    PcieFabricConfig credit_chain_config = multi_mwr_config;
    credit_chain_config.host_to_device_credits.posted_header_credits = 1;
    credit_chain_config.credit_return_latency_ps = 1'000;
    PcieFabric credit_chain(credit_chain_config);
    const auto credit_stalled = credit_chain.submit(transaction(
        PcieOperation::PostedWrite, 512));
    test.check(
        credit_stalled.request_tlps == 4
            && credit_stalled.waits.credit_ps == 3'000
            && credit_stalled.waits.link_queue_ps == 0,
        "intra-transaction credit stalls are not relabeled as link queueing");
    credit_chain.validateInvariants();

    PcieFabricConfig multi_mrd_config = defaultPcieFabricConfig();
    multi_mrd_config.max_payload_size_bytes = 512;
    multi_mrd_config.max_read_request_size_bytes = 128;
    PcieFabric multi_mrd(multi_mrd_config);
    const auto mrd = multi_mrd.submit(transaction(
        PcieOperation::NonPostedRead,
        512,
        PcieDirection::DeviceToHost));
    test.check(
        mrd.request_tlps == 4
            && mrd.completion_tlps == 4
            && mrd.waits.link_queue_ps == 0,
        "one uncontended multi-MRd transaction has zero link-queue wait");
    multi_mrd.validateInvariants();

    PcieFabricConfig multi_cpld_config = defaultPcieFabricConfig();
    multi_cpld_config.max_payload_size_bytes = 128;
    multi_cpld_config.max_read_request_size_bytes = 512;
    PcieFabric multi_cpld(multi_cpld_config);
    const auto cpld = multi_cpld.submit(transaction(
        PcieOperation::NonPostedRead,
        512,
        PcieDirection::DeviceToHost));
    test.check(
        cpld.request_tlps == 1
            && cpld.completion_tlps == 4
            && cpld.waits.link_queue_ps == 0,
        "one uncontended multi-CplD transaction has zero link-queue wait");
    multi_cpld.validateInvariants();

    PcieFabricConfig backlog_config = defaultPcieFabricConfig();
    backlog_config.max_payload_size_bytes = 128;
    PcieFabric backlog(backlog_config);
    const auto external = backlog.submit(transaction(
        PcieOperation::PostedWrite, 64));
    const auto behind_external = backlog.submit(transaction(
        PcieOperation::PostedWrite, 512));
    test.check(
        external.waits.link_queue_ps == 0
            && external.completed_at_ps == 1'397
            && behind_external.request_tlps == 4
            && behind_external.first_issue_at_ps == 1'397
            && behind_external.completed_at_ps == 11'045
            && behind_external.waits.link_queue_ps == 1'397,
        "a 64-byte external write backlog is charged exactly once");
    backlog.validateInvariants();

    PcieFabricConfig completion_gap_config = defaultPcieFabricConfig();
    completion_gap_config.max_payload_size_bytes = 128;
    completion_gap_config.max_read_request_size_bytes = 128;
    completion_gap_config.read_completion_latency_ps.samples_ps = {10'000};
    PcieFabric completion_gap(completion_gap_config);
    const auto read_before_gap = completion_gap.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::HostToDevice));
    const auto posted_around_completion = completion_gap.submit(transaction(
        PcieOperation::PostedWrite,
        1'024,
        PcieDirection::DeviceToHost));
    test.check(
        read_before_gap.completion_ready_at_ps == 10'381
            && read_before_gap.completed_at_ps == 12'730
            && posted_around_completion.request_tlps == 8
            && posted_around_completion.first_issue_at_ps == 0
            && posted_around_completion.completed_at_ps == 22'379
            && posted_around_completion.waits.link_queue_ps == 3'081,
        "legal posted wait behind a future completion is charged exactly");
    completion_gap.validateInvariants();
}

void testAnalyticalPlanRollbackAndStreamIndependence(TestRunner& test) {
    constexpr Picoseconds first_sample = 64'767;
    constexpr Picoseconds second_sample = 98'824;
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.analytical_seed = 0x123456789abcdef0ULL;
    config.paths[1].analytical_penalties.numa = gaussianProfile(
        100'000, 20'000);

    PcieFabric discarded(config);
    {
        auto plan = discarded.beginPlan();
        const auto private_result = discarded.schedule(
            plan, transaction(PcieOperation::HostStore, 8));
        test.check(
            private_result.path_delay.numa_ps == first_sample,
            "discarded plan observes the first private profile sample");
    }
    const auto after_discard = discarded.submit(transaction(
        PcieOperation::HostStore, 8));
    const auto after_discard_second = discarded.submit(transaction(
        PcieOperation::HostStore, 8));
    test.check(
        after_discard.path_delay.numa_ps == first_sample
            && after_discard_second.path_delay.numa_ps == second_sample,
        "discarding a plan rolls back its analytical draw cursor");

    PcieFabric stale_fabric(config);
    auto winner = stale_fabric.beginPlan();
    auto stale = stale_fabric.beginPlan();
    const auto winner_result = stale_fabric.schedule(
        winner, transaction(PcieOperation::HostStore, 8));
    const auto stale_result = stale_fabric.schedule(
        stale, transaction(PcieOperation::HostStore, 8));
    stale_fabric.commit(std::move(winner));
    test.expectThrowAs<std::logic_error>(
        [&stale_fabric, &stale]() {
            stale_fabric.commit(std::move(stale));
        },
        "stale sampled plan is rejected");
    const auto after_stale = stale_fabric.submit(transaction(
        PcieOperation::HostStore, 8));
    test.check(
        winner_result.path_delay.numa_ps == first_sample
            && stale_result.path_delay.numa_ps == first_sample
            && after_stale.path_delay.numa_ps == second_sample,
        "stale-plan rejection preserves the committed draw stream");

    PcieFabric failed_fabric(config);
    auto failed_plan = failed_fabric.beginPlan();
    PcieTransactionRequest overflowing = transaction(
        PcieOperation::HostStore, 8);
    overflowing.submitted_at_ps =
        std::numeric_limits<std::uint64_t>::max();
    test.expectThrowAs<std::overflow_error>(
        [&failed_fabric, &failed_plan, &overflowing]() {
            (void)failed_fabric.schedule(failed_plan, overflowing);
        },
        "failed schedule throws after sampling its candidate state");
    const auto recovered = failed_fabric.schedule(
        failed_plan, transaction(PcieOperation::HostStore, 8));
    failed_fabric.commit(std::move(failed_plan));
    const auto after_failed = failed_fabric.submit(transaction(
        PcieOperation::HostStore, 8));
    test.check(
        recovered.path_delay.numa_ps == first_sample
            && after_failed.path_delay.numa_ps == second_sample,
        "failed schedule rolls back its analytical draw cursor");

    PcieFabricConfig isolated_config = defaultPcieFabricConfig();
    isolated_config.analytical_seed = 0x123456789abcdef0ULL;
    isolated_config.paths[1].analytical_penalties.iommu = gaussianProfile(
        100'000, 20'000);
    PcieFabricConfig extra_component_config = isolated_config;
    extra_component_config.paths[1].analytical_penalties.numa =
        gaussianTailProfile(
            100'000, 10'000, 500'000, 250'000, 20'000);
    PcieFabric isolated(isolated_config);
    PcieFabric with_extra_component(extra_component_config);
    for (std::size_t draw = 0; draw < 8; ++draw) {
        const auto isolated_result = isolated.submit(transaction(
            PcieOperation::HostStore, 8));
        const auto extra_result = with_extra_component.submit(transaction(
            PcieOperation::HostStore, 8));
        test.check(
            isolated_result.path_delay.iommu_ps
                    == extra_result.path_delay.iommu_ps
                && sameAnalyticalAccounting(
                    isolated_result.path_profiles.iommu,
                    extra_result.path_profiles.iommu),
            "unrelated component activity cannot perturb an IOMMU stream at draw "
                + std::to_string(draw));
    }
    discarded.validateInvariants();
    stale_fabric.validateInvariants();
    failed_fabric.validateInvariants();
    isolated.validateInvariants();
    with_extra_component.validateInvariants();
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

void testPostedBypassesSameDomainReadCompletion(TestRunner& test) {
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.max_payload_size_bytes = 128;
    config.max_read_request_size_bytes = 128;
    config.read_completion_latency_ps.samples_ps = {1'000'000};
    PcieFabric fabric(config);

    PcieTransactionRequest read = transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost);
    read.ordering = PcieOrdering::VisibilityDependency;
    read.ordering_domain = 6;
    const auto read_result = fabric.submit(read);

    PcieTransactionRequest posted = transaction(
        PcieOperation::PostedWrite,
        4,
        PcieDirection::DeviceToHost);
    posted.ordering = PcieOrdering::VisibilityDependency;
    posted.ordering_domain = 6;
    const auto posted_result = fabric.submit(posted);

    test.check(
        read_result.transaction_id == 1
            && read_result.waits.ordering_ps == 0
            && read_result.request_finished_at_ps == 381
            && read_result.completion_ready_at_ps == 1'000'381
            && read_result.completed_at_ps == 1'002'730,
        "long same-domain MRd advances its completion horizon exactly");
    test.check(
        posted_result.transaction_id == 2
            && posted_result.waits.ordering_ps == 0
            && posted_result.waits.link_queue_ps == 381
            && posted_result.first_issue_at_ps == 381
            && posted_result.request_finished_at_ps == 826
            && posted_result.completed_at_ps == 826
            && posted_result.completed_at_ps < read_result.completed_at_ps,
        "same-domain posted write bypasses the MRd completion horizon");
    fabric.validateInvariants();
}

void testSplitOrderingHorizons(TestRunner& test) {
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.max_payload_size_bytes = 128;
    config.max_read_request_size_bytes = 128;
    PcieFabric fabric(config);

    PcieTransactionRequest posted = transaction(
        PcieOperation::PostedWrite,
        4,
        PcieDirection::DeviceToHost);
    posted.ordering = PcieOrdering::VisibilityDependency;
    posted.ordering_domain = 9;
    const auto posted_result = fabric.submit(posted);

    PcieTransactionRequest read = transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost);
    read.ordering = PcieOrdering::VisibilityDependency;
    read.ordering_domain = 9;
    const auto first_read = fabric.submit(read);
    const auto second_read = fabric.submit(read);

    PcieTransactionRequest independent = transaction(
        PcieOperation::HostStore, 8);
    independent.ordering = PcieOrdering::Independent;
    independent.ordering_domain = 9;
    const auto bypass = fabric.submit(independent);

    test.check(
        posted_result.completed_at_ps == 445
            && first_read.waits.ordering_ps == 445
            && first_read.first_issue_at_ps == 445
            && first_read.request_finished_at_ps == 826
            && first_read.completed_at_ps == 3'175,
        "non-posted read waits for the same-domain posted horizon exactly");
    test.check(
        second_read.waits.ordering_ps == 3'175
            && second_read.first_issue_at_ps == 3'175
            && second_read.request_finished_at_ps == 3'556
            && second_read.completed_at_ps == 5'905,
        "non-posted read waits for the prior read completion horizon exactly");
    test.check(
        bypass.waits.ordering_ps == 0
            && bypass.first_issue_at_ps == 0
            && bypass.completed_at_ps == 0,
        "independent work bypasses both same-domain dependency horizons");
    fabric.validateInvariants();
}

void testPostedFillsBlockedReadGapTransactionally(TestRunner& test) {
    PcieFabricConfig config = defaultPcieFabricConfig();
    config.max_payload_size_bytes = 128;
    config.max_read_request_size_bytes = 128;
    config.max_outstanding_read_requests = 1;
    config.completion_buffer_bytes = 128;
    config.read_completion_latency_ps.samples_ps = {1'000'000};

    PcieFabric gap_fabric(config);
    const auto first_read = gap_fabric.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto blocked_read = gap_fabric.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto posted = gap_fabric.submit(transaction(
        PcieOperation::PostedWrite,
        4,
        PcieDirection::DeviceToHost));
    test.check(
        first_read.request_finished_at_ps == 381
            && first_read.completed_at_ps == 1'002'730
            && blocked_read.first_issue_at_ps == 1'002'730
            && blocked_read.waits.outstanding_read_ps == 1'002'349,
        "one read slot creates the frozen resource-blocked MRd gap");
    test.check(
        posted.first_issue_at_ps == 381
            && posted.request_finished_at_ps == 826
            && posted.completed_at_ps == 826
            && posted.waits.link_queue_ps == 381
            && posted.completed_at_ps < blocked_read.first_issue_at_ps,
        "ready posted write fills the blocked-MRd serializer gap exactly");
    gap_fabric.validateInvariants();

    PcieFabric failure_fabric(config);
    (void)failure_fabric.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    (void)failure_fabric.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto before_failure = failure_fabric.totalAccounting();
    PcieTransactionRequest too_large = transaction(
        PcieOperation::PostedWrite,
        64 * 1024ULL,
        PcieDirection::DeviceToHost);
    test.expectThrowAs<std::logic_error>(
        [&failure_fabric, &too_large]() {
            (void)failure_fabric.submit(too_large);
        },
        "posted transaction too large for a blocked-MRd gap is rejected");
    test.check(
        failure_fabric.generation() == 2
            && sameAccounting(
                before_failure, failure_fabric.totalAccounting())
            && failure_fabric.accounting(PcieServiceClass::PayloadWrite)
                    .transactions
                == 0,
        "oversized gap insertion leaves IDs, reservations and ledgers unchanged");

    const auto recovered = failure_fabric.submit(transaction(
        PcieOperation::PostedWrite,
        4,
        PcieDirection::DeviceToHost));
    test.check(
        recovered.transaction_id == 3
            && recovered.first_issue_at_ps == 381
            && recovered.completed_at_ps == 826
            && recovered.waits.link_queue_ps == 381,
        "small posted gap insertion succeeds exactly after rejected work");
    failure_fabric.validateInvariants();
}

void testPostedCreditAwareArbitration(TestRunner& test) {
    PcieFabricConfig post_credit_config = defaultPcieFabricConfig();
    post_credit_config.max_payload_size_bytes = 128;
    post_credit_config.max_read_request_size_bytes = 128;
    post_credit_config.max_outstanding_read_requests = 1;
    post_credit_config.completion_buffer_bytes = 128;
    post_credit_config.credit_return_latency_ps = 1'003'000;
    post_credit_config.read_completion_latency_ps.samples_ps = {1'000'000};
    post_credit_config.device_to_host_credits.posted_header_credits = 1;
    PcieFabric post_credit(post_credit_config);

    const auto seed_posted = post_credit.submit(transaction(
        PcieOperation::PostedWrite,
        4,
        PcieDirection::DeviceToHost));
    const auto first_read = post_credit.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto blocked_read = post_credit.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto posted_after_credit = post_credit.submit(transaction(
        PcieOperation::PostedWrite,
        4,
        PcieDirection::DeviceToHost));

    test.check(
        seed_posted.completed_at_ps == 445
            && first_read.request_finished_at_ps == 826
            && first_read.completed_at_ps == 1'003'175
            && blocked_read.first_issue_at_ps == 1'003'175
            && blocked_read.request_finished_at_ps == 1'003'556,
        "credit-aware queue case creates the frozen future MRd interval");
    test.check(
        posted_after_credit.waits.credit_ps == 1'002'619
            && posted_after_credit.waits.link_queue_ps == 937
            && posted_after_credit.first_issue_at_ps == 1'003'556
            && posted_after_credit.completed_at_ps == 1'004'001,
        "link wait reached after posted credit return is charged exactly");
    post_credit.validateInvariants();

    PcieFabricConfig unavailable_config = defaultPcieFabricConfig();
    unavailable_config.max_payload_size_bytes = 128;
    unavailable_config.max_read_request_size_bytes = 128;
    unavailable_config.max_outstanding_read_requests = 1;
    unavailable_config.completion_buffer_bytes = 128;
    unavailable_config.credit_return_latency_ps = 3'500;
    unavailable_config.device_to_host_credits.posted_header_credits = 1;
    PcieFabric unavailable(unavailable_config);

    (void)unavailable.submit(transaction(
        PcieOperation::PostedWrite,
        4,
        PcieDirection::DeviceToHost));
    (void)unavailable.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto short_gap_read = unavailable.submit(transaction(
        PcieOperation::NonPostedRead,
        128,
        PcieDirection::DeviceToHost));
    const auto unavailable_until_after_gap = unavailable.submit(transaction(
        PcieOperation::PostedWrite,
        128,
        PcieDirection::DeviceToHost));
    test.check(
        short_gap_read.first_issue_at_ps == 3'175
            && short_gap_read.request_finished_at_ps == 3'556
            && unavailable_until_after_gap.waits.link_queue_ps == 826
            && unavailable_until_after_gap.waits.credit_ps == 3'119
            && unavailable_until_after_gap.first_issue_at_ps == 3'945
            && unavailable_until_after_gap.completed_at_ps == 6'358,
        "posted arbitration waits for credit before enforcing displacement");
    unavailable.validateInvariants();
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
        path_config.paths[1].analytical_penalties.numa = fixedProfile(numa);
        path_config.paths[1].analytical_penalties.iommu = fixedProfile(iommu);
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
    testAnalyticalProfileValidation(test);
    testAnalyticalSamplingGoldens(test);
    testEveryComponentSupportsEveryProfileKind(test);
    testFragmentSamplingAndAccounting(test);
    testLinkQueueCountsOnlyExternalContention(test);
    testAnalyticalPlanRollbackAndStreamIndependence(test);
    testPostedWriteSegmentationAndSerialization(test);
    testReadByteMatrixAndDirection(test);
    testHostStoreAndClassLedgers(test);
    testPlanAtomicityAndStaleness(test);
    testFiniteCreditsAndReadResources(test);
    testPostedBypassesSameDomainReadCompletion(test);
    testSplitOrderingHorizons(test);
    testPostedFillsBlockedReadGapTransactionally(test);
    testPostedCreditAwareArbitration(test);
    testOrderingPathAndOverflow(test);
    testCrossClassAccountingOverflowIsAtomic(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " PCIe test(s) failed\n";
        return 1;
    }
    std::cout << "all RNIC PCIe fabric tests passed\n";
    return 0;
}
