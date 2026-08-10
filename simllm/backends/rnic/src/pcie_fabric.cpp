#include "simllm/rnic/pcie_fabric.h"

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simllm::rnic {
namespace {

constexpr std::int64_t kGaussianScaleQ20 = 1 << 20;
// Rounded midpoint quantiles Phi^-1((i + 0.5) / 64) in signed Q20. Keeping
// this table and all sampling arithmetic integral makes replay independent of
// the host C++ library's floating-point normal-distribution implementation.
constexpr std::array<std::int32_t, 64> kGaussianQuantilesQ20{
    -2534994, -2083969, -1847245, -1678779, -1545043, -1432569,
    -1334521, -1246929, -1167269, -1093831, -1025400, -961079,
    -900186, -842187, -786658, -733252, -681684, -631714,
    -583140, -535786, -489502, -444152, -399618, -355794,
    -312583, -269897, -227653, -185776, -144193, -102836,
    -61638, -20536, 20536, 61638, 102836, 144193, 185776,
    227653, 269897, 312583, 355794, 399618, 444152, 489502,
    535786, 583140, 631714, 681684, 733252, 786658, 842187,
    900186, 961079, 1025400, 1093831, 1167269, 1246929,
    1334521, 1432569, 1545043, 1678779, 1847245, 2083969,
    2534994,
};

enum class PathComponent : std::uint8_t {
    Numa,
    Iommu,
    Acs,
    Switch,
    DdioMiss,
    GpuDirect,
    Count,
};

constexpr std::size_t kPathComponentCount =
    static_cast<std::size_t>(PathComponent::Count);

std::uint64_t checkedAdd(std::uint64_t lhs, std::uint64_t rhs) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error("RNIC PCIe unsigned addition overflow");
    }
    return lhs + rhs;
}

std::uint64_t checkedMultiply(std::uint64_t lhs, std::uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::overflow_error("RNIC PCIe unsigned multiplication overflow");
    }
    return lhs * rhs;
}

std::uint64_t ceilDivide(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::logic_error("RNIC PCIe division by zero");
    }
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

void checkedAccumulate(std::uint64_t& destination, std::uint64_t value) {
    destination = checkedAdd(destination, value);
}

bool isPowerOfTwo(std::uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

void validateDuration(Picoseconds value, const char* field_name) {
    constexpr Picoseconds maximum = static_cast<Picoseconds>(
        std::numeric_limits<std::int64_t>::max());
    if (value > maximum) {
        throw std::invalid_argument(
            std::string("RNIC PCIe ") + field_name
            + " must be between 0 and INT64_MAX ps");
    }
}

std::size_t classIndex(PcieServiceClass service_class) {
    const std::size_t index = static_cast<std::size_t>(service_class);
    if (index >= kPcieServiceClassCount) {
        throw std::invalid_argument("invalid RNIC PCIe service class");
    }
    return index;
}

std::size_t directionIndex(PcieDirection direction) {
    switch (direction) {
    case PcieDirection::HostToDevice:
        return 0;
    case PcieDirection::DeviceToHost:
        return 1;
    default:
        throw std::invalid_argument("invalid RNIC PCIe direction");
    }
}

PcieDirection opposite(PcieDirection direction) {
    switch (direction) {
    case PcieDirection::HostToDevice:
        return PcieDirection::DeviceToHost;
    case PcieDirection::DeviceToHost:
        return PcieDirection::HostToDevice;
    default:
        throw std::invalid_argument("invalid RNIC PCIe direction");
    }
}

void validateOperation(PcieOperation operation) {
    switch (operation) {
    case PcieOperation::HostStore:
    case PcieOperation::PostedWrite:
    case PcieOperation::NonPostedRead:
        return;
    default:
        throw std::invalid_argument("invalid RNIC PCIe operation");
    }
}

void validateOrdering(PcieOrdering ordering) {
    switch (ordering) {
    case PcieOrdering::VisibilityDependency:
    case PcieOrdering::Independent:
        return;
    default:
        throw std::invalid_argument("invalid RNIC PCIe ordering mode");
    }
}

void validateEndpoint(PcieEndpointKind endpoint) {
    switch (endpoint) {
    case PcieEndpointKind::MmioBar:
    case PcieEndpointKind::HostPinnedMemory:
    case PcieEndpointKind::GpuMemory:
    case PcieEndpointKind::DeviceMemory:
        return;
    default:
        throw std::invalid_argument("invalid RNIC PCIe endpoint kind");
    }
}

const PcieAnalyticalDelayProfile& penaltyProfile(
    const PciePathPenaltyProfiles& profiles,
    PathComponent component) {
    switch (component) {
    case PathComponent::Numa:
        return profiles.numa;
    case PathComponent::Iommu:
        return profiles.iommu;
    case PathComponent::Acs:
        return profiles.acs;
    case PathComponent::Switch:
        return profiles.switch_path;
    case PathComponent::DdioMiss:
        return profiles.ddio_miss;
    case PathComponent::GpuDirect:
        return profiles.gpu_direct;
    default:
        throw std::logic_error("invalid RNIC PCIe path component");
    }
}

Picoseconds& penaltyDelay(
    PciePathDelayAccounting& accounting,
    PathComponent component) {
    switch (component) {
    case PathComponent::Numa:
        return accounting.numa_ps;
    case PathComponent::Iommu:
        return accounting.iommu_ps;
    case PathComponent::Acs:
        return accounting.acs_ps;
    case PathComponent::Switch:
        return accounting.switch_ps;
    case PathComponent::DdioMiss:
        return accounting.ddio_miss_ps;
    case PathComponent::GpuDirect:
        return accounting.gpu_direct_ps;
    default:
        throw std::logic_error("invalid RNIC PCIe path component");
    }
}

PcieAnalyticalDelayAccounting& penaltyAccounting(
    PciePathProfileAccounting& accounting,
    PathComponent component) {
    switch (component) {
    case PathComponent::Numa:
        return accounting.numa;
    case PathComponent::Iommu:
        return accounting.iommu;
    case PathComponent::Acs:
        return accounting.acs;
    case PathComponent::Switch:
        return accounting.switch_path;
    case PathComponent::DdioMiss:
        return accounting.ddio_miss;
    case PathComponent::GpuDirect:
        return accounting.gpu_direct;
    default:
        throw std::logic_error("invalid RNIC PCIe path component");
    }
}

const PcieAnalyticalDelayAccounting& penaltyAccounting(
    const PciePathProfileAccounting& accounting,
    PathComponent component) {
    switch (component) {
    case PathComponent::Numa:
        return accounting.numa;
    case PathComponent::Iommu:
        return accounting.iommu;
    case PathComponent::Acs:
        return accounting.acs;
    case PathComponent::Switch:
        return accounting.switch_path;
    case PathComponent::DdioMiss:
        return accounting.ddio_miss;
    case PathComponent::GpuDirect:
        return accounting.gpu_direct;
    default:
        throw std::logic_error("invalid RNIC PCIe path component");
    }
}

std::uint64_t splitMix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::uint64_t analyticalWord(
    std::uint64_t seed,
    std::uint32_t path_id,
    PathComponent component,
    std::uint64_t draw_index,
    std::uint64_t stream) noexcept {
    std::uint64_t key = seed;
    key ^= 0x9e3779b97f4a7c15ULL
        * (static_cast<std::uint64_t>(path_id) + 1);
    key ^= 0xd1b54a32d192ed03ULL
        * (static_cast<std::uint64_t>(component) + 1);
    key ^= 0x94d049bb133111ebULL * (draw_index + 1);
    key ^= 0xbf58476d1ce4e5b9ULL * (stream + 1);
    return splitMix64(key);
}

std::uint32_t probabilityBucket(std::uint64_t word) noexcept {
    const std::uint64_t high = word >> 32;
    return static_cast<std::uint32_t>(
        (high * kPcieProbabilityScalePpm) >> 32);
}

bool probabilityEvent(
    std::uint32_t probability_ppm,
    std::uint64_t word) noexcept {
    if (probability_ppm == 0) {
        return false;
    }
    if (probability_ppm == kPcieProbabilityScalePpm) {
        return true;
    }
    return probabilityBucket(word) < probability_ppm;
}

Picoseconds scaledGaussianMagnitude(
    Picoseconds standard_deviation_ps,
    std::uint32_t absolute_quantile_q20) {
    const std::uint64_t whole = standard_deviation_ps
        / static_cast<std::uint64_t>(kGaussianScaleQ20);
    const std::uint64_t remainder = standard_deviation_ps
        % static_cast<std::uint64_t>(kGaussianScaleQ20);
    const std::uint64_t whole_product = checkedMultiply(
        whole, absolute_quantile_q20);
    const std::uint64_t remainder_product = checkedMultiply(
        remainder, absolute_quantile_q20);
    const std::uint64_t rounded_remainder = checkedAdd(
        remainder_product,
        static_cast<std::uint64_t>(kGaussianScaleQ20 / 2));
    return checkedAdd(
        whole_product,
        rounded_remainder
            / static_cast<std::uint64_t>(kGaussianScaleQ20));
}

Picoseconds gaussianSample(
    Picoseconds mean_ps,
    Picoseconds standard_deviation_ps,
    std::uint64_t word) {
    const std::int32_t quantile = kGaussianQuantilesQ20[
        static_cast<std::size_t>(word & 63ULL)];
    const std::uint32_t magnitude_q20 = static_cast<std::uint32_t>(
        quantile < 0 ? -static_cast<std::int64_t>(quantile) : quantile);
    const Picoseconds magnitude = scaledGaussianMagnitude(
        standard_deviation_ps, magnitude_q20);
    if (quantile < 0) {
        return magnitude >= mean_ps ? 0 : mean_ps - magnitude;
    }
    return checkedAdd(mean_ps, magnitude);
}

Picoseconds maximumGaussianSample(
    Picoseconds mean_ps,
    Picoseconds standard_deviation_ps) {
    return checkedAdd(
        mean_ps,
        scaledGaussianMagnitude(
            standard_deviation_ps,
            static_cast<std::uint32_t>(kGaussianQuantilesQ20.back())));
}

void validateAnalyticalProfile(
    const PcieAnalyticalDelayProfile& profile,
    const char* name) {
    if (profile.version != kPcieAnalyticalDelayProfileVersion) {
        throw std::invalid_argument(
            std::string("unsupported RNIC PCIe ") + name
            + " analytical profile version");
    }
    if (profile.incidence_probability_ppm > kPcieProbabilityScalePpm
        || profile.tail_probability_ppm > kPcieProbabilityScalePpm) {
        throw std::invalid_argument(
            std::string("RNIC PCIe ") + name
            + " analytical probability exceeds one million ppm");
    }
    validateDuration(profile.mean_ps, name);
    validateDuration(profile.standard_deviation_ps, name);
    validateDuration(profile.tail_mean_ps, name);
    validateDuration(profile.tail_standard_deviation_ps, name);
    switch (profile.kind) {
    case PcieAnalyticalDelayKind::Disabled:
        if (profile.incidence_probability_ppm != 0
            || profile.mean_ps != 0
            || profile.standard_deviation_ps != 0
            || profile.tail_probability_ppm != 0
            || profile.tail_mean_ps != 0
            || profile.tail_standard_deviation_ps != 0) {
            throw std::invalid_argument(
                std::string("RNIC PCIe disabled ") + name
                + " profile has active fields");
        }
        return;
    case PcieAnalyticalDelayKind::Fixed:
        if (profile.incidence_probability_ppm == 0) {
            throw std::invalid_argument(
                std::string("RNIC PCIe active ") + name
                + " profile has zero incidence; use Disabled explicitly");
        }
        if (profile.standard_deviation_ps != 0
            || profile.tail_probability_ppm != 0
            || profile.tail_mean_ps != 0
            || profile.tail_standard_deviation_ps != 0) {
            throw std::invalid_argument(
                std::string("RNIC PCIe fixed ") + name
                + " profile has active Gaussian fields");
        }
        return;
    case PcieAnalyticalDelayKind::Gaussian:
        if (profile.incidence_probability_ppm == 0
            || profile.standard_deviation_ps == 0
            || profile.tail_probability_ppm != 0
            || profile.tail_mean_ps != 0
            || profile.tail_standard_deviation_ps != 0) {
            throw std::invalid_argument(
                std::string("RNIC PCIe Gaussian ") + name
                + " profile fields are inconsistent");
        }
        break;
    case PcieAnalyticalDelayKind::GaussianTailMixture:
        if (profile.incidence_probability_ppm == 0
            || profile.standard_deviation_ps == 0
            || profile.tail_probability_ppm == 0
            || profile.tail_probability_ppm
                == kPcieProbabilityScalePpm
            || profile.tail_mean_ps <= profile.mean_ps
            || profile.tail_standard_deviation_ps == 0) {
            throw std::invalid_argument(
                std::string("RNIC PCIe Gaussian-tail ") + name
                + " profile fields are inconsistent");
        }
        break;
    default:
        throw std::invalid_argument(
            std::string("invalid RNIC PCIe ") + name
            + " analytical profile kind");
    }
    try {
        (void)maximumGaussianSample(
            profile.mean_ps, profile.standard_deviation_ps);
        if (profile.kind == PcieAnalyticalDelayKind::GaussianTailMixture) {
            (void)maximumGaussianSample(
                profile.tail_mean_ps,
                profile.tail_standard_deviation_ps);
        }
    } catch (const std::overflow_error&) {
        throw std::invalid_argument(
            std::string("RNIC PCIe ") + name
            + " analytical profile exceeds the timestamp range");
    }
}

Picoseconds maximumAnalyticalProfileDelay(
    const PcieAnalyticalDelayProfile& profile) {
    switch (profile.kind) {
    case PcieAnalyticalDelayKind::Disabled:
        return 0;
    case PcieAnalyticalDelayKind::Fixed:
        return profile.mean_ps;
    case PcieAnalyticalDelayKind::Gaussian:
        return maximumGaussianSample(
            profile.mean_ps, profile.standard_deviation_ps);
    case PcieAnalyticalDelayKind::GaussianTailMixture:
        return std::max(
            maximumGaussianSample(
                profile.mean_ps, profile.standard_deviation_ps),
            maximumGaussianSample(
                profile.tail_mean_ps,
                profile.tail_standard_deviation_ps));
    default:
        throw std::logic_error(
            "invalid validated RNIC PCIe analytical profile kind");
    }
}

struct LinkRate {
    std::uint64_t transfers_per_second_millions{0};
    std::uint64_t encoding_numerator{0};
    std::uint64_t encoding_denominator{0};
};

LinkRate linkRate(PcieGeneration generation) {
    switch (generation) {
    case PcieGeneration::Gen1:
        return LinkRate{2500, 8, 10};
    case PcieGeneration::Gen2:
        return LinkRate{5000, 8, 10};
    case PcieGeneration::Gen3:
        return LinkRate{8000, 128, 130};
    case PcieGeneration::Gen4:
        return LinkRate{16000, 128, 130};
    case PcieGeneration::Gen5:
        return LinkRate{32000, 128, 130};
    default:
        throw std::invalid_argument("unsupported RNIC PCIe generation");
    }
}

struct LinkReservation {
    Picoseconds started_at_ps{0};
    Picoseconds finished_at_ps{0};
};

enum class LinkReservationKind : std::uint8_t {
    Posted,
    NonPosted,
    Completion,
};

class RationalLink {
public:
    RationalLink() = default;

    RationalLink(PcieGeneration generation, std::uint16_t lane_count) {
        const LinkRate rate = linkRate(generation);
        denominator_ = checkedMultiply(
            checkedMultiply(
                rate.transfers_per_second_millions,
                lane_count),
            rate.encoding_numerator);
        numerator_per_byte_ = checkedMultiply(
            checkedMultiply(8, rate.encoding_denominator),
            1'000'000);
    }

    Picoseconds readyAt(Picoseconds desired_at_ps) const {
        return ceiling(maxPoint(
            RationalPoint{desired_at_ps, 0}, cursor_));
    }

    Picoseconds postedOpportunityAt(Picoseconds desired_at_ps) const {
        // Credit availability decides whether a posted TLP is ready to claim
        // a gap. Return the first idle point here; reservePosted checks the
        // complete TLP fit after credits have been evaluated.
        RationalPoint candidate{desired_at_ps, 0};
        if (has_posted_reservation_) {
            candidate = maxPoint(candidate, posted_cursor_);
        }
        for (const Interval& interval : reservations_) {
            if (lessOrEqual(interval.finish, candidate)) {
                continue;
            }
            if (lessOrEqual(interval.start, candidate)) {
                candidate = interval.finish;
                continue;
            }
            break;
        }
        return ceiling(candidate);
    }

    LinkReservation reserveFifo(
        Picoseconds desired_at_ps,
        std::uint64_t modeled_link_bytes,
        LinkReservationKind kind) {
        if (modeled_link_bytes == 0) {
            throw std::logic_error("RNIC PCIe cannot serialize zero bytes");
        }
        if (kind == LinkReservationKind::Posted) {
            throw std::logic_error(
                "RNIC PCIe posted traffic requires posted arbitration");
        }
        const RationalPoint start = maxPoint(
            RationalPoint{desired_at_ps, 0}, cursor_);
        const RationalPoint finish = addBytes(start, modeled_link_bytes);
        reservations_.push_back(Interval{start, finish, kind});
        cursor_ = finish;
        return reported(start, finish);
    }

    LinkReservation reservePosted(
        Picoseconds desired_at_ps,
        std::uint64_t modeled_link_bytes) {
        if (modeled_link_bytes == 0) {
            throw std::logic_error("RNIC PCIe cannot serialize zero bytes");
        }
        const RationalPoint start = findPostedStart(
            desired_at_ps, modeled_link_bytes);
        const RationalPoint finish = addBytes(start, modeled_link_bytes);
        const auto insertion = std::lower_bound(
            reservations_.begin(),
            reservations_.end(),
            start,
            [](const Interval& interval, const RationalPoint& point) {
                return less(interval.start, point);
            });
        reservations_.insert(
            insertion,
            Interval{start, finish, LinkReservationKind::Posted});
        if (less(cursor_, finish)) {
            cursor_ = finish;
        }
        posted_cursor_ = finish;
        has_posted_reservation_ = true;
        return reported(start, finish);
    }

    void validate() const {
        if (denominator_ == 0 || numerator_per_byte_ == 0) {
            throw std::logic_error("RNIC PCIe link-rate invariant failed");
        }
        if (cursor_.remainder >= denominator_
            || posted_cursor_.remainder >= denominator_) {
            throw std::logic_error(
                "RNIC PCIe serializer remainder invariant failed");
        }
        RationalPoint previous_end{};
        RationalPoint latest_posted{};
        bool first = true;
        bool found_posted = false;
        for (const Interval& interval : reservations_) {
            if (interval.start.remainder >= denominator_
                || interval.finish.remainder >= denominator_
                || !less(interval.start, interval.finish)
                || (!first && less(interval.start, previous_end))) {
                throw std::logic_error(
                    "RNIC PCIe serializer calendar invariant failed");
            }
            previous_end = interval.finish;
            first = false;
            if (interval.kind == LinkReservationKind::Posted) {
                latest_posted = interval.finish;
                found_posted = true;
            }
        }
        const RationalPoint expected_cursor = reservations_.empty()
            ? RationalPoint{}
            : reservations_.back().finish;
        if (!equal(cursor_, expected_cursor)
            || found_posted != has_posted_reservation_
            || (found_posted && !equal(posted_cursor_, latest_posted))) {
            throw std::logic_error(
                "RNIC PCIe serializer cursor invariant failed");
        }
    }

private:
    struct RationalPoint {
        Picoseconds whole_ps{0};
        std::uint64_t remainder{0};
    };

    struct Interval {
        RationalPoint start;
        RationalPoint finish;
        LinkReservationKind kind{LinkReservationKind::NonPosted};
    };

    static bool less(
        const RationalPoint& lhs,
        const RationalPoint& rhs) noexcept {
        return lhs.whole_ps < rhs.whole_ps
            || (lhs.whole_ps == rhs.whole_ps
                && lhs.remainder < rhs.remainder);
    }

    static bool equal(
        const RationalPoint& lhs,
        const RationalPoint& rhs) noexcept {
        return lhs.whole_ps == rhs.whole_ps
            && lhs.remainder == rhs.remainder;
    }

    static bool lessOrEqual(
        const RationalPoint& lhs,
        const RationalPoint& rhs) noexcept {
        return !less(rhs, lhs);
    }

    static RationalPoint maxPoint(
        const RationalPoint& lhs,
        const RationalPoint& rhs) noexcept {
        return less(lhs, rhs) ? rhs : lhs;
    }

    static Picoseconds ceiling(const RationalPoint& point) {
        return point.remainder == 0
            ? point.whole_ps
            : checkedAdd(point.whole_ps, 1);
    }

    LinkReservation reported(
        const RationalPoint& start,
        const RationalPoint& finish) const {
        return LinkReservation{ceiling(start), ceiling(finish)};
    }

    RationalPoint addBytes(
        const RationalPoint& start,
        std::uint64_t modeled_link_bytes) const {
        const std::uint64_t increment = checkedMultiply(
            modeled_link_bytes, numerator_per_byte_);
        const std::uint64_t increment_whole = increment / denominator_;
        const std::uint64_t increment_remainder = increment % denominator_;
        RationalPoint finish{
            checkedAdd(start.whole_ps, increment_whole),
            start.remainder,
        };
        const std::uint64_t remainder_sum = checkedAdd(
            finish.remainder, increment_remainder);
        finish.remainder = remainder_sum;
        if (finish.remainder >= denominator_) {
            finish.remainder -= denominator_;
            finish.whole_ps = checkedAdd(finish.whole_ps, 1);
        }
        return finish;
    }

    RationalPoint findPostedStart(
        Picoseconds desired_at_ps,
        std::uint64_t modeled_link_bytes) const {
        if (modeled_link_bytes == 0) {
            throw std::logic_error("RNIC PCIe cannot serialize zero bytes");
        }
        RationalPoint candidate{desired_at_ps, 0};
        if (has_posted_reservation_) {
            candidate = maxPoint(candidate, posted_cursor_);
        }
        for (const Interval& interval : reservations_) {
            if (lessOrEqual(interval.finish, candidate)) {
                continue;
            }
            if (less(interval.start, candidate)
                || equal(interval.start, candidate)) {
                candidate = interval.finish;
                continue;
            }
            const RationalPoint finish = addBytes(
                candidate, modeled_link_bytes);
            if (lessOrEqual(finish, interval.start)) {
                return candidate;
            }
            if (interval.kind == LinkReservationKind::NonPosted) {
                throw std::logic_error(
                    "RNIC PCIe posted request would have to displace a "
                    "finalized non-posted reservation");
            }
            candidate = interval.finish;
        }
        return candidate;
    }

    std::uint64_t denominator_{1};
    std::uint64_t numerator_per_byte_{1};
    RationalPoint cursor_{};
    RationalPoint posted_cursor_{};
    bool has_posted_reservation_{false};
    std::vector<Interval> reservations_;
};

class CapacityPool {
public:
    CapacityPool() = default;
    explicit CapacityPool(std::uint64_t capacity) : capacity_(capacity) {}

    Picoseconds earliest(
        std::uint64_t required,
        Picoseconds desired_at_ps) const {
        if (required > capacity_) {
            throw std::invalid_argument(
                "RNIC PCIe request exceeds finite resource capacity");
        }
        if (desired_at_ps < last_acquire_at_ps_) {
            throw std::logic_error(
                "RNIC PCIe resource acquisition time regressed");
        }
        std::uint64_t used = in_use_;
        std::size_t release_index = 0;
        Picoseconds candidate = desired_at_ps;
        const auto release_through = [&]() {
            while (release_index < releases_.size()
                   && releases_[release_index].at_ps <= candidate) {
                if (releases_[release_index].amount > used) {
                    throw std::logic_error(
                        "RNIC PCIe resource release underflow");
                }
                used -= releases_[release_index].amount;
                ++release_index;
            }
        };
        release_through();
        while (required > capacity_ - used) {
            if (release_index == releases_.size()) {
                throw std::logic_error(
                    "RNIC PCIe resource has no scheduled release");
            }
            candidate = std::max(
                candidate, releases_[release_index].at_ps);
            release_through();
        }
        return candidate;
    }

    void acquireAt(std::uint64_t amount, Picoseconds at_ps) {
        if (at_ps < last_acquire_at_ps_) {
            throw std::logic_error(
                "RNIC PCIe resource acquisition time regressed");
        }
        releaseThrough(at_ps);
        if (amount > capacity_ - in_use_) {
            throw std::logic_error(
                "RNIC PCIe resource acquired before it was available");
        }
        in_use_ = checkedAdd(in_use_, amount);
        high_watermark_ = std::max(high_watermark_, in_use_);
        last_acquire_at_ps_ = at_ps;
    }

    void releaseAt(std::uint64_t amount, Picoseconds at_ps) {
        if (amount == 0) {
            return;
        }
        const auto insertion = std::upper_bound(
            releases_.begin(),
            releases_.end(),
            at_ps,
            [](Picoseconds time, const Release& release) {
                return time < release.at_ps;
            });
        releases_.insert(insertion, Release{at_ps, amount});
    }

    void validate() const {
        if (capacity_ == 0 || in_use_ > capacity_
            || high_watermark_ > capacity_) {
            throw std::logic_error("RNIC PCIe capacity invariant failed");
        }
        std::uint64_t scheduled = 0;
        Picoseconds previous = 0;
        bool first = true;
        for (const Release& release : releases_) {
            if (!first && release.at_ps < previous) {
                throw std::logic_error(
                    "RNIC PCIe resource releases are not ordered");
            }
            scheduled = checkedAdd(scheduled, release.amount);
            previous = release.at_ps;
            first = false;
        }
        if (scheduled != in_use_) {
            throw std::logic_error(
                "RNIC PCIe resource reservation conservation failed");
        }
    }

private:
    struct Release {
        Picoseconds at_ps{0};
        std::uint64_t amount{0};
    };

    void releaseThrough(Picoseconds at_ps) {
        while (!releases_.empty() && releases_.front().at_ps <= at_ps) {
            if (releases_.front().amount > in_use_) {
                throw std::logic_error(
                    "RNIC PCIe resource release underflow");
            }
            in_use_ -= releases_.front().amount;
            releases_.pop_front();
        }
    }

    std::uint64_t capacity_{1};
    std::uint64_t in_use_{0};
    std::uint64_t high_watermark_{0};
    Picoseconds last_acquire_at_ps_{0};
    std::deque<Release> releases_;
};

struct DirectionResources {
    RationalLink link;
    CapacityPool posted_headers;
    CapacityPool posted_data;
    CapacityPool nonposted_headers;
    CapacityPool nonposted_data;
    CapacityPool completion_headers;
    CapacityPool completion_data;

    DirectionResources() = default;

    DirectionResources(
        const PcieFabricConfig& config,
        const PcieCreditConfig& credits)
        : link(config.generation, config.lane_count),
          posted_headers(credits.posted_header_credits),
          posted_data(credits.posted_data_credits),
          nonposted_headers(credits.nonposted_header_credits),
          nonposted_data(std::max<std::uint32_t>(
              1, credits.nonposted_data_credits)),
          completion_headers(credits.completion_header_credits),
          completion_data(credits.completion_data_credits) {}

    void validate() const {
        link.validate();
        posted_headers.validate();
        posted_data.validate();
        nonposted_headers.validate();
        nonposted_data.validate();
        completion_headers.validate();
        completion_data.validate();
    }
};

struct PathDelay {
    PciePathDelayAccounting fields;
    PciePathProfileAccounting profiles;
    Picoseconds total_ps{0};
};

struct AnalyticalSample {
    Picoseconds delay_ps{0};
    bool occurrence{false};
    bool tail{false};
};

AnalyticalSample sampleAnalyticalProfile(
    const PcieAnalyticalDelayProfile& profile,
    std::uint64_t seed,
    std::uint32_t path_id,
    PathComponent component,
    std::uint64_t draw_index) {
    AnalyticalSample sample;
    sample.occurrence = probabilityEvent(
        profile.incidence_probability_ppm,
        analyticalWord(seed, path_id, component, draw_index, 0));
    if (!sample.occurrence) {
        return sample;
    }
    switch (profile.kind) {
    case PcieAnalyticalDelayKind::Disabled:
        return sample;
    case PcieAnalyticalDelayKind::Fixed:
        sample.delay_ps = profile.mean_ps;
        return sample;
    case PcieAnalyticalDelayKind::Gaussian:
        sample.delay_ps = gaussianSample(
            profile.mean_ps,
            profile.standard_deviation_ps,
            analyticalWord(seed, path_id, component, draw_index, 2));
        return sample;
    case PcieAnalyticalDelayKind::GaussianTailMixture:
        sample.tail = probabilityEvent(
            profile.tail_probability_ppm,
            analyticalWord(seed, path_id, component, draw_index, 1));
        sample.delay_ps = sample.tail
            ? gaussianSample(
                  profile.tail_mean_ps,
                  profile.tail_standard_deviation_ps,
                  analyticalWord(seed, path_id, component, draw_index, 3))
            : gaussianSample(
                  profile.mean_ps,
                  profile.standard_deviation_ps,
                  analyticalWord(seed, path_id, component, draw_index, 2));
        return sample;
    default:
        throw std::logic_error(
            "invalid validated RNIC PCIe analytical profile kind");
    }
}

using PathDrawKey = std::pair<std::uint32_t, PathComponent>;

PathDelay samplePathDelay(
    const PciePathConfig& path,
    std::uint64_t seed,
    std::map<PathDrawKey, std::uint64_t>& draw_cursors) {
    PathDelay result;
    result.fields.base_ps = path.base_latency_ps;
    result.total_ps = checkedAdd(result.total_ps, result.fields.base_ps);
    for (std::size_t index = 0; index < kPathComponentCount; ++index) {
        const PathComponent component = static_cast<PathComponent>(index);
        const PathDrawKey key{path.path_id, component};
        const auto cursor = draw_cursors.find(key);
        const std::uint64_t draw_index = cursor == draw_cursors.end()
            ? 0
            : cursor->second;
        if (draw_index == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error(
                "RNIC PCIe analytical draw index overflow");
        }
        const AnalyticalSample sample = sampleAnalyticalProfile(
            penaltyProfile(path.analytical_penalties, component),
            seed,
            path.path_id,
            component,
            draw_index);
        Picoseconds& field = penaltyDelay(result.fields, component);
        field = sample.delay_ps;
        PcieAnalyticalDelayAccounting& accounting = penaltyAccounting(
            result.profiles, component);
        accounting.draws = 1;
        accounting.occurrences = sample.occurrence ? 1 : 0;
        accounting.tail_draws = sample.tail ? 1 : 0;
        result.total_ps = checkedAdd(result.total_ps, sample.delay_ps);
        draw_cursors[key] = draw_index + 1;
    }
    return result;
}

void accumulateDirection(
    PcieDirectionAccounting& destination,
    const PcieDirectionAccounting& source) {
    checkedAccumulate(destination.tlps, source.tlps);
    checkedAccumulate(destination.payload_bytes, source.payload_bytes);
    checkedAccumulate(destination.overhead_bytes, source.overhead_bytes);
    checkedAccumulate(
        destination.modeled_link_bytes, source.modeled_link_bytes);
}

void accumulateWait(
    PcieWaitAccounting& destination,
    const PcieWaitAccounting& source) {
    checkedAccumulate(destination.ordering_ps, source.ordering_ps);
    checkedAccumulate(
        destination.outstanding_read_ps,
        source.outstanding_read_ps);
    checkedAccumulate(
        destination.completion_buffer_ps,
        source.completion_buffer_ps);
    checkedAccumulate(destination.credit_ps, source.credit_ps);
    checkedAccumulate(destination.link_queue_ps, source.link_queue_ps);
}

void accumulateServiceDelay(
    PcieServiceDelayAccounting& destination,
    const PcieServiceDelayAccounting& source) {
    checkedAccumulate(destination.host_store_ps, source.host_store_ps);
    checkedAccumulate(
        destination.posted_write_visibility_ps,
        source.posted_write_visibility_ps);
    checkedAccumulate(
        destination.read_completion_ps, source.read_completion_ps);
}

void accumulatePath(
    PciePathDelayAccounting& destination,
    const PciePathDelayAccounting& source) {
    checkedAccumulate(destination.base_ps, source.base_ps);
    checkedAccumulate(destination.numa_ps, source.numa_ps);
    checkedAccumulate(destination.iommu_ps, source.iommu_ps);
    checkedAccumulate(destination.acs_ps, source.acs_ps);
    checkedAccumulate(destination.switch_ps, source.switch_ps);
    checkedAccumulate(destination.ddio_miss_ps, source.ddio_miss_ps);
    checkedAccumulate(destination.gpu_direct_ps, source.gpu_direct_ps);
}

void accumulateAnalyticalAccounting(
    PcieAnalyticalDelayAccounting& destination,
    const PcieAnalyticalDelayAccounting& source) {
    checkedAccumulate(destination.draws, source.draws);
    checkedAccumulate(destination.occurrences, source.occurrences);
    checkedAccumulate(destination.tail_draws, source.tail_draws);
}

void accumulatePathProfiles(
    PciePathProfileAccounting& destination,
    const PciePathProfileAccounting& source) {
    for (std::size_t index = 0; index < kPathComponentCount; ++index) {
        const PathComponent component = static_cast<PathComponent>(index);
        accumulateAnalyticalAccounting(
            penaltyAccounting(destination, component),
            penaltyAccounting(source, component));
    }
}

void validatePathProfiles(const PciePathProfileAccounting& accounting) {
    for (std::size_t index = 0; index < kPathComponentCount; ++index) {
        const auto& component = penaltyAccounting(
            accounting, static_cast<PathComponent>(index));
        if (component.occurrences > component.draws
            || component.tail_draws > component.occurrences) {
            throw std::logic_error(
                "RNIC PCIe analytical profile accounting invariant failed");
        }
    }
}

void accumulatePathRepeated(
    PciePathDelayAccounting& destination,
    const PciePathDelayAccounting& source) {
    accumulatePath(destination, source);
}

void accountTlp(
    PcieTransactionResult& result,
    PcieDirection direction,
    std::uint64_t payload_bytes,
    std::uint64_t overhead_bytes) {
    PcieDirectionAccounting& accounting =
        direction == PcieDirection::HostToDevice
        ? result.host_to_device
        : result.device_to_host;
    checkedAccumulate(accounting.tlps, 1);
    checkedAccumulate(accounting.payload_bytes, payload_bytes);
    checkedAccumulate(accounting.overhead_bytes, overhead_bytes);
    checkedAccumulate(
        accounting.modeled_link_bytes,
        checkedAdd(payload_bytes, overhead_bytes));
}

void validateDirectionAccounting(const PcieDirectionAccounting& accounting) {
    if (checkedAdd(accounting.payload_bytes, accounting.overhead_bytes)
        != accounting.modeled_link_bytes) {
        throw std::logic_error(
            "RNIC PCIe direction byte conservation failed");
    }
}

struct TransactionFragment {
    std::uint64_t logical_bytes{0};
    std::uint64_t wire_payload_bytes{0};
    std::uint64_t aligned_address{0};
    std::vector<std::uint64_t> completion_payloads;
};

struct TransactionShape {
    std::vector<TransactionFragment> fragments;
    std::uint64_t total_tlps{0};
};

std::uint64_t dwordSpan(
    std::uint64_t address,
    std::uint64_t logical_bytes) {
    const std::uint64_t with_leading = checkedAdd(
        address % 4, logical_bytes);
    return checkedMultiply(ceilDivide(with_leading, 4), 4);
}

TransactionShape buildShape(
    const PcieFabricConfig& config,
    const PcieTransactionRequest& request) {
    if (request.abi_version != kPcieTransactionAbiVersion) {
        throw std::invalid_argument(
            "unsupported RNIC PCIe transaction ABI version");
    }
    (void)classIndex(request.service_class);
    validateOperation(request.operation);
    if ((request.service_class == PcieServiceClass::PayloadRead
         && request.operation != PcieOperation::NonPostedRead)
        || (request.service_class == PcieServiceClass::PayloadWrite
            && request.operation != PcieOperation::PostedWrite)) {
        throw std::invalid_argument(
            "RNIC PCIe payload class disagrees with its operation");
    }
    (void)directionIndex(request.request_direction);
    validateOrdering(request.ordering);
    if (request.useful_bytes == 0 || request.transfer_bytes == 0) {
        throw std::invalid_argument(
            "RNIC PCIe transaction byte counts must be positive");
    }
    if (request.useful_bytes > request.transfer_bytes) {
        throw std::invalid_argument(
            "RNIC PCIe useful bytes exceed transferred bytes");
    }
    if (request.first_byte_offset >= 4096) {
        throw std::invalid_argument(
            "RNIC PCIe first-byte offset must be below 4096");
    }
    (void)checkedAdd(request.first_byte_offset, request.transfer_bytes);

    TransactionShape shape;
    if (request.operation == PcieOperation::HostStore) {
        return shape;
    }

    const std::uint64_t protocol_limit =
        request.operation == PcieOperation::PostedWrite
        ? config.max_payload_size_bytes
        : config.max_read_request_size_bytes;
    std::uint64_t remaining = request.transfer_bytes;
    std::uint64_t address = request.first_byte_offset;
    while (remaining != 0) {
        const std::uint64_t protocol_capacity =
            protocol_limit - address % 4;
        const std::uint64_t page_capacity = 4096 - address % 4096;
        const std::uint64_t logical = std::min(
            remaining, std::min(protocol_capacity, page_capacity));
        if (logical == 0) {
            throw std::logic_error("RNIC PCIe made no segmentation progress");
        }
        TransactionFragment fragment;
        fragment.logical_bytes = logical;
        fragment.wire_payload_bytes = dwordSpan(address, logical);
        fragment.aligned_address = address - address % 4;

        shape.total_tlps = checkedAdd(shape.total_tlps, 1);
        if (request.operation == PcieOperation::NonPostedRead) {
            std::uint64_t completion_address = fragment.aligned_address;
            std::uint64_t completion_remaining =
                fragment.wire_payload_bytes;
            while (completion_remaining != 0) {
                std::uint64_t completion_capacity =
                    config.max_payload_size_bytes;
                const std::uint64_t rcb_offset = completion_address
                    % config.read_completion_boundary_bytes;
                if (rcb_offset != 0) {
                    completion_capacity = std::min(
                        completion_capacity,
                        static_cast<std::uint64_t>(
                            config.read_completion_boundary_bytes)
                            - rcb_offset);
                }
                const std::uint64_t payload = std::min(
                    completion_remaining, completion_capacity);
                fragment.completion_payloads.push_back(payload);
                shape.total_tlps = checkedAdd(shape.total_tlps, 1);
                completion_remaining -= payload;
                completion_address = checkedAdd(
                    completion_address, payload);
            }
        }
        if (shape.total_tlps > config.max_tlps_per_transaction) {
            throw std::invalid_argument(
                "RNIC PCIe transaction exceeds the configured TLP limit");
        }
        shape.fragments.push_back(std::move(fragment));
        remaining -= logical;
        address = checkedAdd(address, logical);
    }
    return shape;
}

void validateLatencyProfile(
    const PcieLatencyProfile& profile,
    const char* name) {
    if (profile.samples_ps.size() != 1) {
        throw std::invalid_argument(
            std::string("RNIC PCIe ") + name
            + " profile must contain exactly one fixed v1 sample");
    }
    for (Picoseconds sample : profile.samples_ps) {
        validateDuration(sample, name);
    }
}

void validateCreditConfig(
    const PcieCreditConfig& credits,
    const char* direction_name) {
    if (credits.posted_header_credits == 0
        || credits.posted_data_credits == 0
        || credits.nonposted_header_credits == 0
        || credits.completion_header_credits == 0
        || credits.completion_data_credits == 0) {
        throw std::invalid_argument(
            std::string("RNIC PCIe ") + direction_name
            + " active credit pools must be positive");
    }
}

void validateConfig(const PcieFabricConfig& config) {
    if (config.version != kPcieFabricConfigVersion) {
        throw std::invalid_argument(
            "unsupported RNIC PCIe fabric config version");
    }
    (void)linkRate(config.generation);
    if (!isPowerOfTwo(config.lane_count) || config.lane_count > 32) {
        throw std::invalid_argument(
            "RNIC PCIe lane count must be a power of two up to x32");
    }
    const auto valid_transaction_size = [](std::uint32_t size) {
        return isPowerOfTwo(size) && size >= 128 && size <= 4096;
    };
    if (!valid_transaction_size(config.max_payload_size_bytes)
        || !valid_transaction_size(config.max_read_request_size_bytes)) {
        throw std::invalid_argument(
            "RNIC PCIe MPS and MRRS must be powers of two from 128 to 4096");
    }
    if (config.read_completion_boundary_bytes != 64
        && config.read_completion_boundary_bytes != 128) {
        throw std::invalid_argument(
            "RNIC PCIe read completion boundary must be 64 or 128 bytes");
    }
    if (config.posted_write_overhead_bytes == 0
        || config.read_request_overhead_bytes == 0
        || config.completion_overhead_bytes == 0) {
        throw std::invalid_argument(
            "RNIC PCIe TLP overheads must be positive");
    }
    if (config.data_credit_unit_bytes == 0
        || !isPowerOfTwo(config.data_credit_unit_bytes)) {
        throw std::invalid_argument(
            "RNIC PCIe data-credit unit must be a positive power of two");
    }
    validateCreditConfig(
        config.host_to_device_credits, "host-to-device");
    validateCreditConfig(
        config.device_to_host_credits, "device-to-host");
    const auto validate_data_capacity = [&config](
                                            const PcieCreditConfig& credits,
                                            const char* direction_name) {
        const std::uint64_t posted_capacity = checkedMultiply(
            credits.posted_data_credits, config.data_credit_unit_bytes);
        const std::uint64_t completion_capacity = checkedMultiply(
            credits.completion_data_credits,
            config.data_credit_unit_bytes);
        if (posted_capacity < config.max_payload_size_bytes
            || completion_capacity < config.max_payload_size_bytes) {
            throw std::invalid_argument(
                std::string("RNIC PCIe ") + direction_name
                + " data credits cannot hold one MPS payload");
        }
    };
    validate_data_capacity(
        config.host_to_device_credits, "host-to-device");
    validate_data_capacity(
        config.device_to_host_credits, "device-to-host");
    if (config.max_outstanding_read_requests == 0
        || config.completion_buffer_bytes == 0
        || config.max_tlps_per_transaction == 0) {
        throw std::invalid_argument(
            "RNIC PCIe finite read resources and TLP limit must be positive");
    }
    validateDuration(
        config.credit_return_latency_ps, "credit-return latency");
    validateDuration(
        config.completion_buffer_release_latency_ps,
        "completion-buffer release latency");
    validateLatencyProfile(
        config.host_store_latency_ps, "host-store latency");
    validateLatencyProfile(
        config.posted_write_visibility_latency_ps,
        "posted-write visibility latency");
    validateLatencyProfile(
        config.read_completion_latency_ps,
        "read-completion latency");
    if (config.paths.empty()) {
        throw std::invalid_argument("RNIC PCIe config has no path profiles");
    }
    std::vector<std::uint32_t> path_ids;
    path_ids.reserve(config.paths.size());
    for (const PciePathConfig& path : config.paths) {
        if (path.path_id == 0) {
            throw std::invalid_argument(
                "RNIC PCIe path ID must be nonzero");
        }
        validateEndpoint(path.endpoint);
        validateDuration(path.base_latency_ps, "path base latency");
        Picoseconds aggregate_path_delay = path.base_latency_ps;
        constexpr std::array<const char*, kPathComponentCount> names{
            "NUMA penalty",
            "IOMMU penalty",
            "ACS penalty",
            "switch penalty",
            "DDIO-miss penalty",
            "GPU Direct penalty",
        };
        for (std::size_t index = 0; index < kPathComponentCount; ++index) {
            const auto& profile = penaltyProfile(
                path.analytical_penalties,
                static_cast<PathComponent>(index));
            validateAnalyticalProfile(profile, names[index]);
            const Picoseconds maximum = maximumAnalyticalProfileDelay(
                profile);
            if (maximum
                > static_cast<Picoseconds>(
                      std::numeric_limits<std::int64_t>::max())
                    - aggregate_path_delay) {
                throw std::invalid_argument(
                    "RNIC PCIe aggregate analytical path delay overflows");
            }
            aggregate_path_delay += maximum;
        }
        if (std::find(path_ids.begin(), path_ids.end(), path.path_id)
            != path_ids.end()) {
            throw std::invalid_argument("duplicate RNIC PCIe path ID");
        }
        path_ids.push_back(path.path_id);
    }
}

}  // namespace

PcieFabricConfig defaultPcieFabricConfig() {
    PcieFabricConfig config;
    PciePathConfig mmio;
    mmio.path_id = 1;
    mmio.endpoint = PcieEndpointKind::MmioBar;
    PciePathConfig host_memory;
    host_memory.path_id = 2;
    host_memory.endpoint = PcieEndpointKind::HostPinnedMemory;
    config.paths = {mmio, host_memory};
    return config;
}

class PcieFabric::Impl {
public:
    explicit Impl(PcieFabricConfig config)
        : config_(std::move(config)),
          directions_{
              DirectionResources(config_, config_.host_to_device_credits),
              DirectionResources(config_, config_.device_to_host_credits)},
          outstanding_reads_{
              CapacityPool(config_.max_outstanding_read_requests),
              CapacityPool(config_.max_outstanding_read_requests)},
          completion_buffers_{
              CapacityPool(config_.completion_buffer_bytes),
              CapacityPool(config_.completion_buffer_bytes)} {
        validateConfig(config_);
    }

    std::uint64_t generation() const noexcept { return generation_; }

    void setGeneration(std::uint64_t generation) noexcept {
        generation_ = generation;
    }

    const PcieFabricConfig& config() const noexcept { return config_; }

    const PcieClassAccounting& accounting(
        PcieServiceClass service_class) const {
        return accounting_[classIndex(service_class)];
    }

    PcieClassAccounting totalAccounting() const {
        PcieClassAccounting total;
        for (const PcieClassAccounting& item : accounting_) {
            accumulateClass(total, item);
        }
        return total;
    }

    PcieTransactionResult scheduleTransaction(
        const PcieTransactionRequest& request) {
        const PciePathConfig& path = findPath(request.path_id);
        const TransactionShape shape = buildShape(config_, request);
        if (next_transaction_id_ == std::numeric_limits<std::uint64_t>::max()) {
            throw std::overflow_error("RNIC PCIe transaction ID overflow");
        }

        PcieTransactionResult result;
        result.transaction_id = next_transaction_id_;
        result.client_id = request.client_id;
        result.client_token = request.client_token;
        result.service_class = request.service_class;
        result.operation = request.operation;
        result.request_direction = request.request_direction;
        result.ordering = request.ordering;
        result.path_id = request.path_id;
        result.ordering_domain = request.ordering_domain;
        result.useful_bytes = request.useful_bytes;
        result.transferred_bytes = request.transfer_bytes;
        result.submitted_at_ps = request.submitted_at_ps;

        Picoseconds eligible_at_ps = request.submitted_at_ps;
        if (request.ordering == PcieOrdering::VisibilityDependency) {
            const auto ordering = ordering_cursors_.find(
                request.ordering_domain);
            Picoseconds ordering_horizon = 0;
            if (ordering != ordering_cursors_.end()) {
                ordering_horizon = ordering->second.posted_visibility_ps;
                if (request.operation == PcieOperation::NonPostedRead) {
                    ordering_horizon = std::max(
                        ordering_horizon,
                        ordering->second.nonposted_completion_ps);
                }
            }
            if (ordering_horizon > eligible_at_ps) {
                result.waits.ordering_ps =
                    ordering_horizon - eligible_at_ps;
                eligible_at_ps = ordering_horizon;
            }
        }

        switch (request.operation) {
        case PcieOperation::HostStore:
            scheduleHostStore(
                request, path, eligible_at_ps, result);
            break;
        case PcieOperation::PostedWrite:
            schedulePostedWrite(
                request,
                shape,
                path,
                eligible_at_ps,
                result);
            break;
        case PcieOperation::NonPostedRead:
            scheduleRead(
                request,
                shape,
                path,
                eligible_at_ps,
                result);
            break;
        default:
            throw std::logic_error("invalid validated RNIC PCIe operation");
        }

        if (request.ordering == PcieOrdering::VisibilityDependency) {
            OrderingDomainCursors& ordering =
                ordering_cursors_[request.ordering_domain];
            if (request.operation == PcieOperation::NonPostedRead) {
                ordering.nonposted_completion_ps = result.completed_at_ps;
            } else {
                ordering.posted_visibility_ps = result.completed_at_ps;
            }
        }
        validateDirectionAccounting(result.host_to_device);
        validateDirectionAccounting(result.device_to_host);
        validatePathProfiles(result.path_profiles);
        accumulateResult(result);
        // The public ledger is the checked sum of every class row. Validate
        // that sum while this Impl is still a private plan candidate so a
        // cross-class overflow cannot become committed shared state.
        (void)totalAccounting();
        ++next_transaction_id_;
        return result;
    }

    void validateInvariants() const {
        validateConfig(config_);
        for (const DirectionResources& direction : directions_) {
            direction.validate();
        }
        for (const CapacityPool& pool : outstanding_reads_) {
            pool.validate();
        }
        for (const CapacityPool& pool : completion_buffers_) {
            pool.validate();
        }
        std::uint64_t transactions = 0;
        for (const PcieClassAccounting& item : accounting_) {
            validateDirectionAccounting(item.host_to_device);
            validateDirectionAccounting(item.device_to_host);
            validatePathProfiles(item.path_profiles);
            transactions = checkedAdd(transactions, item.transactions);
        }
        const PcieClassAccounting total = totalAccounting();
        for (const auto& item : path_draw_cursors_) {
            const bool known_path = std::any_of(
                config_.paths.begin(),
                config_.paths.end(),
                [&item](const PciePathConfig& path) {
                    return path.path_id == item.first.first;
                });
            if (!known_path
                || static_cast<std::size_t>(item.first.second)
                    >= kPathComponentCount) {
                throw std::logic_error(
                    "RNIC PCIe analytical draw-cursor key invariant failed");
            }
        }
        for (std::size_t index = 0; index < kPathComponentCount; ++index) {
            const PathComponent component = static_cast<PathComponent>(index);
            std::uint64_t cursor_draws = 0;
            for (const auto& item : path_draw_cursors_) {
                if (item.first.second == component) {
                    cursor_draws = checkedAdd(cursor_draws, item.second);
                }
            }
            if (cursor_draws
                != penaltyAccounting(total.path_profiles, component).draws) {
                throw std::logic_error(
                    "RNIC PCIe analytical draw-cursor invariant failed");
            }
        }
        const std::uint64_t service_samples = checkedAdd(
            checkedAdd(
                host_store_latency_cursor_,
                posted_write_latency_cursor_),
            read_completion_latency_cursor_);
        if (service_samples != total.latency_samples_used) {
            throw std::logic_error(
                "RNIC PCIe service-latency cursor invariant failed");
        }
        if (checkedAdd(transactions, 1) != next_transaction_id_) {
            throw std::logic_error(
                "RNIC PCIe transaction-ID accounting mismatch");
        }
    }

private:
    static void accumulateClass(
        PcieClassAccounting& destination,
        const PcieClassAccounting& source) {
        checkedAccumulate(destination.transactions, source.transactions);
        checkedAccumulate(destination.useful_bytes, source.useful_bytes);
        checkedAccumulate(
            destination.transferred_bytes, source.transferred_bytes);
        checkedAccumulate(
            destination.host_store_bytes, source.host_store_bytes);
        accumulateDirection(
            destination.host_to_device, source.host_to_device);
        accumulateDirection(
            destination.device_to_host, source.device_to_host);
        accumulateWait(destination.waits, source.waits);
        accumulateServiceDelay(
            destination.service_delay, source.service_delay);
        accumulatePath(destination.path_delay, source.path_delay);
        accumulatePathProfiles(
            destination.path_profiles, source.path_profiles);
        checkedAccumulate(
            destination.latency_samples_used,
            source.latency_samples_used);
    }

    const PciePathConfig& findPath(std::uint32_t path_id) const {
        const auto path = std::find_if(
            config_.paths.begin(),
            config_.paths.end(),
            [path_id](const PciePathConfig& candidate) {
                return candidate.path_id == path_id;
            });
        if (path == config_.paths.end()) {
            throw std::invalid_argument("unknown RNIC PCIe path ID");
        }
        if (!path->enabled) {
            throw std::invalid_argument("RNIC PCIe path is disabled");
        }
        return *path;
    }

    Picoseconds sample(
        const PcieLatencyProfile& profile,
        std::uint64_t& cursor) {
        const std::size_t index = static_cast<std::size_t>(
            cursor % profile.samples_ps.size());
        const Picoseconds value = profile.samples_ps[index];
        cursor = checkedAdd(cursor, 1);
        return value;
    }

    static void addPathDelay(
        PcieTransactionResult& result,
        const PathDelay& delay) {
        accumulatePathRepeated(result.path_delay, delay.fields);
        accumulatePathProfiles(result.path_profiles, delay.profiles);
    }

    static void noteFirstIssue(
        PcieTransactionResult& result,
        Picoseconds issue_at_ps) {
        if (result.request_tlps == 0 && result.completion_tlps == 0
            && result.host_store_bytes == 0) {
            result.first_issue_at_ps = issue_at_ps;
        }
    }

    static Picoseconds addDelay(
        Picoseconds base,
        Picoseconds protocol_delay,
        Picoseconds path_delay) {
        return checkedAdd(checkedAdd(base, protocol_delay), path_delay);
    }

    void scheduleHostStore(
        const PcieTransactionRequest& request,
        const PciePathConfig& path,
        Picoseconds eligible_at_ps,
        PcieTransactionResult& result) {
        const PathDelay path_delay = samplePathDelay(
            path, config_.analytical_seed, path_draw_cursors_);
        noteFirstIssue(result, eligible_at_ps);
        result.host_store_bytes = request.transfer_bytes;
        const Picoseconds latency = sample(
            config_.host_store_latency_ps, host_store_latency_cursor_);
        result.service_delay.host_store_ps = latency;
        result.latency_samples_used = 1;
        addPathDelay(result, path_delay);
        result.request_finished_at_ps = addDelay(
            eligible_at_ps, latency, path_delay.total_ps);
        result.completed_at_ps = result.request_finished_at_ps;
    }

    void schedulePostedWrite(
        const PcieTransactionRequest& request,
        const TransactionShape& shape,
        const PciePathConfig& path,
        Picoseconds eligible_at_ps,
        PcieTransactionResult& result) {
        DirectionResources& direction =
            directions_[directionIndex(request.request_direction)];
        std::optional<Picoseconds> preceding_fragment_end;
        Picoseconds latest_visibility = eligible_at_ps;
        bool first = true;
        for (const TransactionFragment& fragment : shape.fragments) {
            const PathDelay path_delay = samplePathDelay(
                path, config_.analytical_seed, path_draw_cursors_);
            const std::uint64_t data_credits = ceilDivide(
                fragment.wire_payload_bytes,
                config_.data_credit_unit_bytes);
            const std::uint64_t link_bytes = checkedAdd(
                fragment.wire_payload_bytes,
                config_.posted_write_overhead_bytes);
            const Picoseconds link_not_before = eligible_at_ps;
            const Picoseconds accounting_eligible = preceding_fragment_end
                ? std::max(link_not_before, *preceding_fragment_end)
                : link_not_before;
            Picoseconds issue_at = direction.link.postedOpportunityAt(
                link_not_before);
            checkedAccumulate(
                result.waits.link_queue_ps, issue_at - accounting_eligible);
            Picoseconds credit_ready = direction.posted_headers.earliest(
                1, issue_at);
            credit_ready = direction.posted_data.earliest(
                data_credits, credit_ready);
            checkedAccumulate(
                result.waits.credit_ps, credit_ready - issue_at);
            const bool credit_stalled = credit_ready > issue_at;
            issue_at = credit_ready;
            direction.posted_headers.acquireAt(1, issue_at);
            direction.posted_data.acquireAt(data_credits, issue_at);

            const LinkReservation reservation = direction.link.reservePosted(
                credit_stalled ? issue_at : link_not_before,
                link_bytes);
            checkedAccumulate(
                result.waits.link_queue_ps,
                reservation.started_at_ps - issue_at);
            if (first) {
                noteFirstIssue(result, reservation.started_at_ps);
                first = false;
            }
            const Picoseconds credit_release = checkedAdd(
                reservation.finished_at_ps,
                config_.credit_return_latency_ps);
            direction.posted_headers.releaseAt(1, credit_release);
            direction.posted_data.releaseAt(
                data_credits, credit_release);

            const Picoseconds latency = sample(
                config_.posted_write_visibility_latency_ps,
                posted_write_latency_cursor_);
            checkedAccumulate(
                result.service_delay.posted_write_visibility_ps,
                latency);
            result.latency_samples_used = checkedAdd(
                result.latency_samples_used, 1);
            addPathDelay(result, path_delay);
            const Picoseconds visible_at = addDelay(
                reservation.finished_at_ps,
                latency,
                path_delay.total_ps);
            latest_visibility = std::max(latest_visibility, visible_at);
            result.request_finished_at_ps = reservation.finished_at_ps;
            ++result.request_tlps;
            accountTlp(
                result,
                request.request_direction,
                fragment.wire_payload_bytes,
                config_.posted_write_overhead_bytes);
            preceding_fragment_end = reservation.finished_at_ps;
        }
        result.completed_at_ps = latest_visibility;
    }

    void scheduleRead(
        const PcieTransactionRequest& request,
        const TransactionShape& shape,
        const PciePathConfig& path,
        Picoseconds eligible_at_ps,
        PcieTransactionResult& result) {
        const std::size_t request_index = directionIndex(
            request.request_direction);
        const PcieDirection completion_direction = opposite(
            request.request_direction);
        DirectionResources& request_resources = directions_[request_index];
        DirectionResources& completion_resources =
            directions_[directionIndex(completion_direction)];
        CapacityPool& outstanding = outstanding_reads_[request_index];
        CapacityPool& completion_buffer =
            completion_buffers_[request_index];

        std::optional<Picoseconds> preceding_request_end;
        std::optional<Picoseconds> preceding_completion_end;
        Picoseconds latest_completion = eligible_at_ps;
        bool first_request = true;
        bool first_response = true;
        for (const TransactionFragment& fragment : shape.fragments) {
            const PathDelay path_delay = samplePathDelay(
                path, config_.analytical_seed, path_draw_cursors_);
            const Picoseconds link_not_before = eligible_at_ps;
            const Picoseconds request_accounting_eligible =
                preceding_request_end
                ? std::max(link_not_before, *preceding_request_end)
                : link_not_before;
            Picoseconds issue_at = request_resources.link.readyAt(
                link_not_before);
            checkedAccumulate(
                result.waits.link_queue_ps,
                issue_at - request_accounting_eligible);
            Picoseconds outstanding_ready = outstanding.earliest(1, issue_at);
            checkedAccumulate(
                result.waits.outstanding_read_ps,
                outstanding_ready - issue_at);
            issue_at = outstanding_ready;
            Picoseconds buffer_ready = completion_buffer.earliest(
                fragment.wire_payload_bytes, issue_at);
            checkedAccumulate(
                result.waits.completion_buffer_ps,
                buffer_ready - issue_at);
            issue_at = buffer_ready;
            const Picoseconds credit_ready =
                request_resources.nonposted_headers.earliest(1, issue_at);
            checkedAccumulate(
                result.waits.credit_ps, credit_ready - issue_at);
            const bool request_resource_stalled =
                credit_ready > request_resources.link.readyAt(link_not_before);
            issue_at = credit_ready;

            outstanding.acquireAt(1, issue_at);
            completion_buffer.acquireAt(
                fragment.wire_payload_bytes, issue_at);
            request_resources.nonposted_headers.acquireAt(1, issue_at);
            const LinkReservation request_reservation =
                request_resources.link.reserveFifo(
                    request_resource_stalled ? issue_at : link_not_before,
                    config_.read_request_overhead_bytes,
                    LinkReservationKind::NonPosted);
            if (first_request) {
                noteFirstIssue(result, request_reservation.started_at_ps);
                first_request = false;
            }
            request_resources.nonposted_headers.releaseAt(
                1,
                checkedAdd(
                    request_reservation.finished_at_ps,
                    config_.credit_return_latency_ps));
            ++result.request_tlps;
            accountTlp(
                result,
                request.request_direction,
                0,
                config_.read_request_overhead_bytes);
            result.request_finished_at_ps =
                request_reservation.finished_at_ps;
            preceding_request_end = request_reservation.finished_at_ps;

            const Picoseconds response_latency = sample(
                config_.read_completion_latency_ps,
                read_completion_latency_cursor_);
            checkedAccumulate(
                result.service_delay.read_completion_ps,
                response_latency);
            result.latency_samples_used = checkedAdd(
                result.latency_samples_used, 1);
            addPathDelay(result, path_delay);
            const Picoseconds response_ready = addDelay(
                request_reservation.finished_at_ps,
                response_latency,
                path_delay.total_ps);
            if (first_response
                || response_ready < *result.completion_ready_at_ps) {
                result.completion_ready_at_ps = response_ready;
                first_response = false;
            }

            const Picoseconds completion_root_ready = response_ready;
            Picoseconds final_completion = response_ready;
            for (std::uint64_t payload : fragment.completion_payloads) {
                const std::uint64_t data_credits = ceilDivide(
                    payload, config_.data_credit_unit_bytes);
                const Picoseconds completion_link_not_before =
                    completion_root_ready;
                const Picoseconds completion_accounting_eligible =
                    preceding_completion_end
                    ? std::max(
                        completion_link_not_before,
                        *preceding_completion_end)
                    : completion_link_not_before;
                Picoseconds completion_issue =
                    completion_resources.link.readyAt(
                        completion_link_not_before);
                checkedAccumulate(
                    result.waits.link_queue_ps,
                    completion_issue - completion_accounting_eligible);
                Picoseconds completion_credit_ready =
                    completion_resources.completion_headers.earliest(
                        1, completion_issue);
                completion_credit_ready =
                    completion_resources.completion_data.earliest(
                        data_credits, completion_credit_ready);
                checkedAccumulate(
                    result.waits.credit_ps,
                    completion_credit_ready - completion_issue);
                const bool completion_credit_stalled =
                    completion_credit_ready > completion_issue;
                completion_issue = completion_credit_ready;
                completion_resources.completion_headers.acquireAt(
                    1, completion_issue);
                completion_resources.completion_data.acquireAt(
                    data_credits, completion_issue);
                const std::uint64_t link_bytes = checkedAdd(
                    payload, config_.completion_overhead_bytes);
                const LinkReservation completion_reservation =
                    completion_resources.link.reserveFifo(
                        completion_credit_stalled
                            ? completion_issue
                            : completion_link_not_before,
                        link_bytes,
                        LinkReservationKind::Completion);
                const Picoseconds credit_release = checkedAdd(
                    completion_reservation.finished_at_ps,
                    config_.credit_return_latency_ps);
                completion_resources.completion_headers.releaseAt(
                    1, credit_release);
                completion_resources.completion_data.releaseAt(
                    data_credits, credit_release);
                ++result.completion_tlps;
                accountTlp(
                    result,
                    completion_direction,
                    payload,
                    config_.completion_overhead_bytes);
                final_completion = completion_reservation.finished_at_ps;
                preceding_completion_end =
                    completion_reservation.finished_at_ps;
            }
            outstanding.releaseAt(1, final_completion);
            completion_buffer.releaseAt(
                fragment.wire_payload_bytes,
                checkedAdd(
                    final_completion,
                    config_.completion_buffer_release_latency_ps));
            latest_completion = std::max(
                latest_completion, final_completion);
        }
        result.completed_at_ps = latest_completion;
    }

    void accumulateResult(const PcieTransactionResult& result) {
        PcieClassAccounting& destination =
            accounting_[classIndex(result.service_class)];
        checkedAccumulate(destination.transactions, 1);
        checkedAccumulate(destination.useful_bytes, result.useful_bytes);
        checkedAccumulate(
            destination.transferred_bytes, result.transferred_bytes);
        checkedAccumulate(
            destination.host_store_bytes, result.host_store_bytes);
        accumulateDirection(
            destination.host_to_device, result.host_to_device);
        accumulateDirection(
            destination.device_to_host, result.device_to_host);
        accumulateWait(destination.waits, result.waits);
        accumulateServiceDelay(
            destination.service_delay, result.service_delay);
        accumulatePath(destination.path_delay, result.path_delay);
        accumulatePathProfiles(
            destination.path_profiles, result.path_profiles);
        checkedAccumulate(
            destination.latency_samples_used,
            result.latency_samples_used);
    }

    PcieFabricConfig config_;
    std::array<DirectionResources, 2> directions_;
    std::array<CapacityPool, 2> outstanding_reads_;
    std::array<CapacityPool, 2> completion_buffers_;
    struct OrderingDomainCursors {
        Picoseconds posted_visibility_ps{0};
        Picoseconds nonposted_completion_ps{0};
    };

    std::map<std::uint64_t, OrderingDomainCursors> ordering_cursors_;
    std::map<PathDrawKey, std::uint64_t> path_draw_cursors_;
    std::array<PcieClassAccounting, kPcieServiceClassCount> accounting_{};
    std::uint64_t generation_{0};
    std::uint64_t next_transaction_id_{1};
    std::uint64_t host_store_latency_cursor_{0};
    std::uint64_t posted_write_latency_cursor_{0};
    std::uint64_t read_completion_latency_cursor_{0};
};

class PcieFabric::Plan::Impl {
public:
    Impl(
        const PcieFabric* owner,
        std::uint64_t base_generation,
        std::unique_ptr<PcieFabric::Impl> state)
        : owner_(owner),
          base_generation_(base_generation),
          state_(std::move(state)) {}

private:
    const PcieFabric* owner_{nullptr};
    std::uint64_t base_generation_{0};
    std::unique_ptr<PcieFabric::Impl> state_;
    friend class PcieFabric;
};

class PcieFabric::OrderingDomainClaims {
public:
    void claim(
        const RnicDevice* owner,
        std::uint64_t submission_domain,
        std::uint64_t completion_domain) {
        if (owner == nullptr
            || submission_domain == 0
            || completion_domain == 0
            || submission_domain == completion_domain) {
            throw std::invalid_argument(
                "shared RNIC PCIe ordering domains must be distinct and "
                "nonzero");
        }
        if (domains_.count(submission_domain) != 0
            || domains_.count(completion_domain) != 0) {
            throw std::invalid_argument(
                "shared RNIC PCIe ordering domain is already claimed");
        }
        std::map<std::uint64_t, const RnicDevice*> candidate = domains_;
        candidate.emplace(submission_domain, owner);
        candidate.emplace(completion_domain, owner);
        domains_.swap(candidate);
    }

    void release(
        const RnicDevice* owner,
        std::uint64_t submission_domain,
        std::uint64_t completion_domain) noexcept {
        releaseOne(owner, submission_domain);
        releaseOne(owner, completion_domain);
    }

    bool claimedByOther(
        const RnicDevice* owner,
        std::uint64_t ordering_domain) const noexcept {
        const auto found = domains_.find(ordering_domain);
        return found != domains_.end() && found->second != owner;
    }

private:
    void releaseOne(
        const RnicDevice* owner,
        std::uint64_t ordering_domain) noexcept {
        const auto found = domains_.find(ordering_domain);
        if (found != domains_.end() && found->second == owner) {
            domains_.erase(found);
        }
    }

    std::map<std::uint64_t, const RnicDevice*> domains_;
};

PcieFabric::Plan::Plan(std::unique_ptr<Plan::Impl> impl)
    : impl_(std::move(impl)) {}

PcieFabric::Plan::~Plan() = default;
PcieFabric::Plan::Plan(Plan&&) noexcept = default;
PcieFabric::Plan& PcieFabric::Plan::operator=(Plan&&) noexcept = default;

PcieFabric::PcieFabric(PcieFabricConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))),
      ordering_domain_claims_(std::make_unique<OrderingDomainClaims>()) {}

PcieFabric::~PcieFabric() = default;

PcieFabric::Plan PcieFabric::beginPlan() const {
    return Plan(std::make_unique<Plan::Impl>(
        this,
        impl_->generation(),
        std::make_unique<Impl>(*impl_)));
}

PcieTransactionResult PcieFabric::schedule(
    Plan& plan,
    const PcieTransactionRequest& request) {
    if (!plan.impl_ || plan.impl_->owner_ != this || !plan.impl_->state_) {
        throw std::invalid_argument(
            "RNIC PCIe plan does not belong to this fabric");
    }
    auto candidate = std::make_unique<Impl>(*plan.impl_->state_);
    PcieTransactionResult result = candidate->scheduleTransaction(request);
    plan.impl_->state_.swap(candidate);
    return result;
}

void PcieFabric::commit(Plan&& plan) {
    if (!plan.impl_ || plan.impl_->owner_ != this || !plan.impl_->state_) {
        throw std::invalid_argument(
            "RNIC PCIe plan does not belong to this fabric");
    }
    if (plan.impl_->base_generation_ != impl_->generation()) {
        throw std::logic_error("cannot commit stale RNIC PCIe plan");
    }
    if (impl_->generation() == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("RNIC PCIe plan generation overflow");
    }
    plan.impl_->state_->setGeneration(impl_->generation() + 1);
    impl_.swap(plan.impl_->state_);
    plan.impl_.reset();
}

PcieTransactionResult PcieFabric::submit(
    const PcieTransactionRequest& request) {
    Plan plan = beginPlan();
    PcieTransactionResult result = schedule(plan, request);
    commit(std::move(plan));
    return result;
}

PcieFabricConfig PcieFabric::config() const {
    return impl_->config();
}

std::uint64_t PcieFabric::generation() const noexcept {
    return impl_->generation();
}

PcieClassAccounting PcieFabric::accounting(
    PcieServiceClass service_class) const {
    return impl_->accounting(service_class);
}

PcieClassAccounting PcieFabric::totalAccounting() const {
    return impl_->totalAccounting();
}

void PcieFabric::validateInvariants() const {
    impl_->validateInvariants();
}

void PcieFabric::claimOrderingDomains(
    const RnicDevice* owner,
    std::uint64_t submission_domain,
    std::uint64_t completion_domain) {
    ordering_domain_claims_->claim(
        owner, submission_domain, completion_domain);
}

void PcieFabric::releaseOrderingDomains(
    const RnicDevice* owner,
    std::uint64_t submission_domain,
    std::uint64_t completion_domain) noexcept {
    ordering_domain_claims_->release(
        owner, submission_domain, completion_domain);
}

bool PcieFabric::orderingDomainClaimedByOther(
    const RnicDevice* owner,
    std::uint64_t ordering_domain) const noexcept {
    return ordering_domain_claims_->claimedByOther(owner, ordering_domain);
}

const char* toString(PcieServiceClass service_class) noexcept {
    switch (service_class) {
    case PcieServiceClass::UarDoorbell:
        return "uar_doorbell";
    case PcieServiceClass::BlueFlame:
        return "blue_flame";
    case PcieServiceClass::DoorbellRecord:
        return "doorbell_record";
    case PcieServiceClass::WqeRead:
        return "wqe_read";
    case PcieServiceClass::QpcIcm:
        return "qpc_icm";
    case PcieServiceClass::MttMpt:
        return "mtt_mpt";
    case PcieServiceClass::PayloadRead:
        return "payload_read";
    case PcieServiceClass::PayloadWrite:
        return "payload_write";
    case PcieServiceClass::CqeWrite:
        return "cqe_write";
    case PcieServiceClass::Command:
        return "command";
    case PcieServiceClass::Interrupt:
        return "interrupt";
    case PcieServiceClass::OdpIommuFault:
        return "odp_iommu_fault";
    case PcieServiceClass::Count:
        return "count";
    default:
        return "invalid";
    }
}

const char* toString(PcieOperation operation) noexcept {
    switch (operation) {
    case PcieOperation::HostStore:
        return "host_store";
    case PcieOperation::PostedWrite:
        return "posted_write";
    case PcieOperation::NonPostedRead:
        return "nonposted_read";
    default:
        return "invalid";
    }
}

const char* toString(PcieDirection direction) noexcept {
    switch (direction) {
    case PcieDirection::HostToDevice:
        return "host_to_device";
    case PcieDirection::DeviceToHost:
        return "device_to_host";
    default:
        return "invalid";
    }
}

const char* toString(PcieAnalyticalDelayKind kind) noexcept {
    switch (kind) {
    case PcieAnalyticalDelayKind::Disabled:
        return "disabled";
    case PcieAnalyticalDelayKind::Fixed:
        return "fixed";
    case PcieAnalyticalDelayKind::Gaussian:
        return "gaussian";
    case PcieAnalyticalDelayKind::GaussianTailMixture:
        return "gaussian_tail_mixture";
    default:
        return "invalid";
    }
}

}  // namespace simllm::rnic
