#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "simllm_atlahs_flow_runtime.h"
#include "simllm_htsim_network_port.h"
#include "simllm/rnic/session_record.h"

int main() {
    using namespace htsim::simllm_rnic;
    std::string hardware;
    for (const std::uint32_t nodes : {2U, 8U}) {
        for (const std::uint64_t rate : {UINT64_C(200000000000),
                                        UINT64_C(400000000000)}) {
            HtsimNetworkPortConfig port_config;
            port_config.endpoint_count = nodes;
            port_config.link_rate_bps = rate;
            port_config.traffic_class = 3;
            HtsimNetworkPort port(port_config);
            const auto caps = port.capabilities();
            if (caps.abi_version != simllm::rnic::kNetworkPortAbiVersionV1
                || caps.packet_attempt_events || caps.ecn_cnp_events
                || caps.policy_update_events || caps.pfc_events
                || caps.dynamic_link_events) {
                throw std::logic_error("session identity port is not ABI v1");
            }
            auto config = defaultSimllmAtlahsDeviceConfig();
            config.work_queue.source = nodes - 1;
            simllm::rnic::RnicDeviceAttachments attachments;
            attachments.network_port = &port;
            simllm::rnic::RnicDevice device(config, attachments);
            // No EventList, runtime, topology, post, or progress is created.
            const std::string value =
                simllm::rnic::renderEffectiveHardwareConfigJson(device);
            if (!hardware.empty() && value != hardware) {
                throw std::logic_error("session hardware identity changed");
            }
            hardware = value;
        }
    }
    std::cout << simllm::rnic::rnicSha256Hex(hardware) << '\n'
              << hardware << '\n';
}
