#include "fake_network.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "simllm/rnic/rnic_device.h"

namespace {

using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceAttachments;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::WorkQueueConfig;
using simllm::rnic::WorkRequest;
using simllm::rnic::testing::FakeNetworkPort;

struct Options {
    std::uint64_t wqes{32};
    std::uint64_t doorbell_batch{1};
    std::uint64_t signal_every{1};
    std::uint64_t network_capacity{32};
    std::uint64_t network_latency_ps{0};
    std::uint64_t doorbell_service_ps{1000};
    std::uint64_t fetch_service_ps{10};
    std::uint64_t sq_depth{64};
    std::uint64_t cq_depth{64};
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

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("every RNIC probe option needs a value");
        }
        const std::string option = argv[index];
        const std::uint64_t value = parseUnsigned(argv[index + 1], argv[index]);
        if (option == "--wqes") {
            options.wqes = value;
        } else if (option == "--doorbell-batch") {
            options.doorbell_batch = value;
        } else if (option == "--signal-every") {
            options.signal_every = value;
        } else if (option == "--network-capacity") {
            options.network_capacity = value;
        } else if (option == "--network-latency-ps") {
            options.network_latency_ps = value;
        } else if (option == "--doorbell-service-ps") {
            options.doorbell_service_ps = value;
        } else if (option == "--fetch-service-ps") {
            options.fetch_service_ps = value;
        } else if (option == "--sq-depth") {
            options.sq_depth = value;
        } else if (option == "--cq-depth") {
            options.cq_depth = value;
        } else {
            throw std::invalid_argument("unknown RNIC probe option: " + option);
        }
    }
    if (options.wqes == 0 || options.doorbell_batch == 0
        || options.signal_every == 0 || options.network_capacity == 0
        || options.sq_depth == 0 || options.cq_depth == 0) {
        throw std::invalid_argument("RNIC probe sizes and capacities must be positive");
    }
    if (options.wqes > options.sq_depth) {
        throw std::invalid_argument("RNIC probe requires SQ depth >= WQE count");
    }
    constexpr std::uint64_t max_size =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (options.wqes > max_size || options.network_capacity > max_size
        || options.sq_depth > max_size || options.cq_depth > max_size) {
        throw std::out_of_range(
            "RNIC probe size exceeds the platform size_t range");
    }
    return options;
}

std::optional<Picoseconds> earlier(
    std::optional<Picoseconds> lhs,
    std::optional<Picoseconds> rhs) {
    if (!lhs.has_value()) {
        return rhs;
    }
    if (!rhs.has_value()) {
        return lhs;
    }
    return std::min(*lhs, *rhs);
}

int run(const Options& options) {
    FakeNetworkPort network(
        static_cast<std::size_t>(options.network_capacity),
        options.network_latency_ps);
    WorkQueueConfig config;
    config.sq_depth = static_cast<std::size_t>(options.sq_depth);
    config.cq_depth = static_cast<std::size_t>(options.cq_depth);
    config.source = 0;
    config.qpn = 1;
    config.doorbell_service_ps = options.doorbell_service_ps;
    config.wqe_fetch_service_ps = options.fetch_service_ps;
    RnicDeviceConfig device_config;
    device_config.work_queue = config;
    device_config.network.enabled = true;
    RnicDeviceAttachments attachments;
    attachments.network_port = &network;
    RnicDevice queue(device_config, attachments);

    for (std::uint64_t index = 1; index <= options.wqes; ++index) {
        WorkRequest request;
        request.wr_id = index;
        request.destination = 1;
        request.payload_bytes = 4096;
        request.signaled = index % options.signal_every == 0
            || index == options.wqes;
        const auto posted = queue.postSend(request, 0);
        if (posted.status != PostStatus::Accepted) {
            throw std::runtime_error("RNIC probe unexpectedly rejected a WQE");
        }
        if (index % options.doorbell_batch == 0 || index == options.wqes) {
            queue.ringDoorbell(0);
        }
    }

    Picoseconds now_ps = 0;
    Picoseconds jct_ps = 0;
    std::uint64_t polled_cqes = 0;
    std::uint64_t iterations = 0;
    while (queue.hasPendingPhysicalWork()) {
        if (++iterations > 1'000'000) {
            throw std::runtime_error("RNIC probe event loop did not converge");
        }
        const auto next = earlier(queue.nextEventTime(), network.nextCompletionTime());
        if (!next.has_value()) {
            throw std::runtime_error("RNIC probe has pending work without an event");
        }
        if (*next < now_ps) {
            throw std::runtime_error("RNIC probe event time regressed");
        }
        now_ps = *next;
        for (const auto& event : network.takeDue(now_ps)) {
            queue.onNetworkEvent(event);
        }
        queue.progress(now_ps);
        const auto cqes = queue.pollCompletionQueue(
            std::numeric_limits<std::size_t>::max(), now_ps);
        polled_cqes += cqes.size();
        if (!cqes.empty()) {
            jct_ps = now_ps;
        }
    }
    queue.validateInvariants();
    if (queue.occupiedSqEntries() != 0 || queue.fatal()
        || !queue.evidence().empty()) {
        throw std::runtime_error("RNIC probe did not drain cleanly");
    }

    const auto& counters = queue.counters();
    std::cout
        << "wqes,doorbell_batch,signal_every,network_capacity,"
           "network_latency_ps,doorbell_service_ps,fetch_service_ps,"
           "doorbells,cqes,network_busy,jct_ps,posted,delivered,reclaimed,"
           "sq_high_watermark,cq_high_watermark,evidence_count\n"
        << options.wqes << ','
        << options.doorbell_batch << ','
        << options.signal_every << ','
        << options.network_capacity << ','
        << options.network_latency_ps << ','
        << options.doorbell_service_ps << ','
        << options.fetch_service_ps << ','
        << counters.doorbells << ','
        << polled_cqes << ','
        << counters.network_busy << ','
        << jct_ps << ','
        << counters.posted_wqes << ','
        << counters.network_delivered << ','
        << counters.sq_reclaimed_wqes << ','
        << counters.sq_high_watermark << ','
        << counters.cq_high_watermark << ','
        << queue.evidence().size() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "simllm_rnic_wq_probe: " << error.what() << '\n';
        return 2;
    }
}
