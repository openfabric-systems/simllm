#ifndef SIMLLM_RNIC_NETWORK_PORT_H
#define SIMLLM_RNIC_NETWORK_PORT_H

#include <cstdint>

namespace simllm::rnic {

using Picoseconds = std::uint64_t;
using WqeId = std::uint64_t;
using FlowId = std::uint64_t;
using PolicyContextToken = std::uint64_t;
using NetworkToken = std::uint64_t;

inline constexpr std::uint32_t kNetworkPortAbiVersionV1 = 1;
inline constexpr std::uint32_t kNetworkPortAbiVersionV2 = 2;
// ABI v1 remains the source-compatible default. A v2 port advertises its
// capabilities before the first descriptor is constructed.
inline constexpr std::uint32_t kNetworkPortAbiVersion =
    kNetworkPortAbiVersionV1;

enum class NetworkSubmitStatus {
    Accepted,
    Busy,
    Rejected,
};

enum class NetworkEventKind {
    Delivered,
    Dropped,
    PacketTxStarted,
    PacketTxFinished,
    PacketRxArrived,
    EcnMarked,
    CnpReceived,
    EligibilityUpdated,
    RateUpdated,
    PfcFrameSubmitted,
    PfcPaused,
    PfcResumed,
    LinkStateChanged,
};

enum class NetworkEventScope {
    FlowExtent,
    PacketAttempt,
    TransportControl,
};

enum class NetworkPacketKind {
    Data,
    Retransmission,
    Ack,
    Nak,
    Cnp,
    Pfc,
    OtherControl,
};

enum class DropEvidenceProvenance {
    None,
    Controlled,
    Asserted,
    Observed,
    Inferred,
};

enum class NetworkLinkState {
    Unknown,
    Up,
    Down,
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

struct NetworkPortCapabilities {
    std::uint32_t abi_version{kNetworkPortAbiVersionV1};
    bool packet_attempt_events{false};
    bool ecn_cnp_events{false};
    bool policy_update_events{false};
    bool pfc_events{false};
    bool dynamic_link_events{false};
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
    NetworkEventScope scope{NetworkEventScope::FlowExtent};
    // Flow-extent events use token directly. Packet-attempt events use a
    // separate session-unique token and retain the admitted extent token in
    // parent_token. Intermediate packet observations never consume a token.
    NetworkToken token{0};
    NetworkToken parent_token{0};
    WqeId wqe_id{0};
    Picoseconds event_time_ps{0};
    std::uint32_t extent_index{0};
    std::uint64_t packet_index{0};
    std::uint32_t transmission_attempt{0};
    std::uint64_t payload_offset_bytes{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes{0};
    NetworkPacketKind packet_kind{NetworkPacketKind::Data};
    bool ecn_marked{false};
    DropLocation drop_location{DropLocation::None};
    DropReason drop_reason{DropReason::None};
    std::uint64_t drop_resource_id{0};
    DropEvidenceProvenance drop_evidence{
        DropEvidenceProvenance::None};

    // Transport-control payload. The event kind selects the applicable
    // subset, and disabled capabilities must leave every field at its default.
    PolicyContextToken policy_context_token{0};
    std::uint32_t source{0};
    std::uint32_t destination{0};
    std::uint64_t link_id{0};
    std::uint8_t priority{0};
    std::uint32_t pause_quanta{0};
    bool has_pause_duration{false};
    Picoseconds pause_duration_ps{0};
    Picoseconds effective_at_ps{0};
    bool has_effective_rate{false};
    std::uint64_t effective_rate_bps{0};
    NetworkLinkState link_state{NetworkLinkState::Unknown};
};

class NetworkPort {
public:
    virtual ~NetworkPort() = default;

    virtual NetworkPortCapabilities capabilities() const noexcept {
        return {};
    }

    // Busy must carry a strictly future retry time. Accepted tokens must be
    // nonzero. ABI v1 tokens remain unique while live; ABI v2 extent and
    // packet-attempt tokens are unique for the session. A completion for
    // another token does not revoke an advertised retry time.
    virtual NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) = 0;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_NETWORK_PORT_H
