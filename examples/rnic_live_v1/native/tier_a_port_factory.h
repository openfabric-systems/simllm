#ifndef SIMLLM_RNIC_LIVE_V1_TIER_A_PORT_FACTORY_H
#define SIMLLM_RNIC_LIVE_V1_TIER_A_PORT_FACTORY_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "simllm/rnic/network_port.h"

namespace simllm::rnic::tier_a {

struct PortConfig {
    std::size_t capacity{1};
    std::uint64_t link_rate_gbps{400};
    std::uint64_t data_header_bytes{0};
    Picoseconds propagation_delay_ps{0};
    bool control_frames{false};
    bool congestion{false};
    bool drop_first{false};
};

struct IssuedToken {
    NetworkToken token{0};
    WqeId wqe_id{0};
    Picoseconds accepted_at_ps{0};
    Picoseconds port_tx_at_ps{0};
    std::uint64_t payload_bytes{0};
};

struct TerminalToken {
    NetworkToken token{0};
    WqeId wqe_id{0};
    NetworkEventKind kind{NetworkEventKind::Delivered};
    Picoseconds at_ps{0};
};

class DrivenPort : public NetworkPort {
public:
    ~DrivenPort() override = default;

    virtual std::optional<Picoseconds> nextEventTime() const = 0;
    virtual std::vector<NetworkEvent> takeDue(Picoseconds now_ps) = 0;
    virtual const std::vector<IssuedToken>& issued() const noexcept = 0;
    virtual const std::vector<TerminalToken>& terminals() const noexcept = 0;
    virtual std::vector<NetworkToken> liveTokens() const = 0;
};

class PortFactory {
public:
    virtual ~PortFactory() = default;

    virtual const char* name() const noexcept = 0;
    virtual std::unique_ptr<DrivenPort> create(const PortConfig& config) = 0;
};

std::unique_ptr<PortFactory> makePortFactory(const std::string& name);

struct ProducerOptions {
    std::string factory;
    std::string expectations_path;
    std::string observations_path;
};

int runTierA(const ProducerOptions& options, PortFactory& factory);

}  // namespace simllm::rnic::tier_a

#endif  // SIMLLM_RNIC_LIVE_V1_TIER_A_PORT_FACTORY_H
