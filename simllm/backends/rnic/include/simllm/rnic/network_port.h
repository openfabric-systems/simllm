#ifndef SIMLLM_RNIC_NETWORK_PORT_H
#define SIMLLM_RNIC_NETWORK_PORT_H

#include <cstdint>

namespace simllm::rnic {

using Picoseconds = std::uint64_t;
using WqeId = std::uint64_t;
using FlowId = std::uint64_t;
using PolicyContextToken = std::uint64_t;
using NetworkToken = std::uint64_t;

inline constexpr std::uint32_t kNetworkPortAbiVersion = 1;

enum class NetworkSubmitStatus {
    Accepted,
    Busy,
    Rejected,
};

enum class NetworkEventKind {
    Delivered,
    Dropped,
};

enum class DropLocation {
    None,
    TxPort,
    Fabric,
    RxPort,
};

enum class DropReason {
    None,
    Injected,
    QueueOverflow,
    LinkDown,
    PolicyRejected,
};

// Version 1 carries one admitted WQE extent. Packetization may emit several
// extents later without exposing SQ, CQ, QP or QPC objects to the network.
struct NetworkTxDescriptor {
    std::uint32_t abi_version{kNetworkPortAbiVersion};
    // Correlation IDs are opaque to NetworkPort implementations. They do not
    // grant ownership of the corresponding RNIC or application objects.
    WqeId wqe_id{0};
    std::uint64_t wr_id{0};
    FlowId flow_id{0};
    std::uint32_t flow_tag{0};
    PolicyContextToken policy_context_token{0};
    std::uint32_t source{0};
    std::uint32_t destination{0};
    std::uint32_t qpn{0};
    std::uint8_t traffic_class{0};
    std::uint64_t payload_bytes{0};
    std::uint32_t extent_index{0};
    std::uint32_t extent_count{1};
    Picoseconds eligible_at_ps{0};
};

struct NetworkSubmitResult {
    NetworkSubmitStatus status{NetworkSubmitStatus::Rejected};
    NetworkToken token{0};
    bool has_retry_time{false};
    Picoseconds retry_at_ps{0};
    DropLocation rejection_location{DropLocation::TxPort};
    DropReason rejection_reason{DropReason::PolicyRejected};

    static NetworkSubmitResult accepted(NetworkToken accepted_token) {
        NetworkSubmitResult result;
        result.status = NetworkSubmitStatus::Accepted;
        result.token = accepted_token;
        result.rejection_location = DropLocation::None;
        result.rejection_reason = DropReason::None;
        return result;
    }

    static NetworkSubmitResult busy(Picoseconds retry_time_ps) {
        NetworkSubmitResult result;
        result.status = NetworkSubmitStatus::Busy;
        result.has_retry_time = true;
        result.retry_at_ps = retry_time_ps;
        result.rejection_location = DropLocation::None;
        result.rejection_reason = DropReason::None;
        return result;
    }

    static NetworkSubmitResult rejected(
        DropLocation location = DropLocation::TxPort,
        DropReason reason = DropReason::PolicyRejected) {
        NetworkSubmitResult result;
        result.status = NetworkSubmitStatus::Rejected;
        result.rejection_location = location;
        result.rejection_reason = reason;
        return result;
    }
};

struct NetworkEvent {
    std::uint32_t abi_version{kNetworkPortAbiVersion};
    NetworkEventKind kind{NetworkEventKind::Delivered};
    NetworkToken token{0};
    WqeId wqe_id{0};
    Picoseconds event_time_ps{0};
    bool ecn_marked{false};
    DropLocation drop_location{DropLocation::None};
    DropReason drop_reason{DropReason::None};
};

class NetworkPort {
public:
    virtual ~NetworkPort() = default;

    // Busy must carry a strictly future retry time. Accepted tokens must be
    // nonzero and unique until their NetworkEvent is returned. A completion
    // for another token does not revoke an advertised retry time.
    virtual NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) = 0;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_NETWORK_PORT_H
