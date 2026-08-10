#include "tier_a_port_factory.h"

#include <memory>
#include <limits>
#include <set>
#include <stdexcept>

#include "fake_network.h"

namespace simllm::rnic::tier_a {
namespace {

class FakeDrivenPort final : public DrivenPort {
public:
    explicit FakeDrivenPort(const PortConfig& config)
        : capacity_(config.capacity),
          link_rate_gbps_(config.link_rate_gbps),
          drop_first_(config.drop_first) {
        if (capacity_ == 0 || link_rate_gbps_ == 0) {
            throw std::invalid_argument(
                "fake Tier A capacity and link rate must be positive");
        }
        if (config.data_header_bytes != 0
            || config.propagation_delay_ps != 0 || config.control_frames
            || config.congestion) {
            throw std::invalid_argument(
                "fake Tier A factory requires the frozen zero-overhead fixture");
        }
    }

    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        const Picoseconds service_ps = serviceTime(descriptor.payload_bytes);
        if (!inner_) {
            fixed_service_ps_ = service_ps;
            inner_ = std::make_unique<testing::FakeNetworkPort>(
                capacity_, service_ps);
        } else if (service_ps != fixed_service_ps_) {
            throw std::invalid_argument(
                "one fake Tier A port cannot mix serialization times");
        }
        NetworkSubmitResult result = inner_->trySubmit(descriptor, now_ps);
        if (result.status == NetworkSubmitStatus::Accepted) {
            if (!live_tokens_.insert(result.token).second) {
                throw std::logic_error("fake factory recycled a live token");
            }
            issued_.push_back(IssuedToken{
                result.token,
                descriptor.wqe_id,
                now_ps,
                now_ps,
                descriptor.payload_bytes,
            });
        }
        return result;
    }

    std::optional<Picoseconds> nextEventTime() const override {
        if (!inner_) {
            return std::nullopt;
        }
        return inner_->nextCompletionTime();
    }

    std::vector<NetworkEvent> takeDue(Picoseconds now_ps) override {
        if (!inner_) {
            return {};
        }
        std::vector<NetworkEvent> events = inner_->takeDue(now_ps);
        for (NetworkEvent& event : events) {
            if (drop_first_ && !drop_emitted_) {
                event.kind = NetworkEventKind::Dropped;
                event.drop_location = DropLocation::Fabric;
                event.drop_reason = DropReason::Injected;
                drop_emitted_ = true;
            }
            if (live_tokens_.erase(event.token) != 1) {
                throw std::logic_error(
                    "fake factory terminated a token that was not live");
            }
            terminals_.push_back(TerminalToken{
                event.token,
                event.wqe_id,
                event.kind,
                event.event_time_ps,
            });
        }
        return events;
    }

    const std::vector<IssuedToken>& issued() const noexcept override {
        return issued_;
    }

    const std::vector<TerminalToken>& terminals() const noexcept override {
        return terminals_;
    }

    std::vector<NetworkToken> liveTokens() const override {
        return {live_tokens_.begin(), live_tokens_.end()};
    }

private:
    Picoseconds serviceTime(std::uint64_t payload_bytes) const {
        if (payload_bytes
            > std::numeric_limits<std::uint64_t>::max() / 8000ULL) {
            throw std::overflow_error(
                "fake Tier A serialization fixture overflows");
        }
        const std::uint64_t numerator = payload_bytes * 8000ULL;
        if (numerator % link_rate_gbps_ != 0) {
            throw std::invalid_argument(
                "fake Tier A serialization time is not an integer");
        }
        return numerator / link_rate_gbps_;
    }

    std::size_t capacity_{0};
    std::uint64_t link_rate_gbps_{0};
    std::unique_ptr<testing::FakeNetworkPort> inner_;
    Picoseconds fixed_service_ps_{0};
    bool drop_first_{false};
    bool drop_emitted_{false};
    std::set<NetworkToken> live_tokens_;
    std::vector<IssuedToken> issued_;
    std::vector<TerminalToken> terminals_;
};

class FakePortFactory final : public PortFactory {
public:
    const char* name() const noexcept override { return "fake"; }

    std::unique_ptr<DrivenPort> create(const PortConfig& config) override {
        return std::make_unique<FakeDrivenPort>(config);
    }
};

}  // namespace

std::unique_ptr<PortFactory> makePortFactory(const std::string& name) {
    if (name == "fake") {
        return std::make_unique<FakePortFactory>();
    }
    if (name == "htsim") {
        throw std::runtime_error(
            "htsim Tier A port factory is unavailable in this repository; "
            "HTSIM-9 supplies a replacement factory without changing runTierA");
    }
    throw std::invalid_argument("--factory must be either fake or htsim");
}

}  // namespace simllm::rnic::tier_a
