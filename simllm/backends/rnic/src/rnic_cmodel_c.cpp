#include "simllm/rnic/rnic_cmodel_c.h"

#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "simllm/rnic/rnic_device.h"
#include "simllm/rnic/rnic_hw_profile.h"

namespace {

using simllm::rnic::CompletionEntry;
using simllm::rnic::CompletionStatus;
using simllm::rnic::DropLocation;
using simllm::rnic::DropReason;
using simllm::rnic::EvidenceClass;
using simllm::rnic::FlowId;
using simllm::rnic::kNetworkPortAbiVersionV1;
using simllm::rnic::kNetworkPortAbiVersionV2;
using simllm::rnic::NetworkEvent;
using simllm::rnic::NetworkEventKind;
using simllm::rnic::NetworkEventScope;
using simllm::rnic::NetworkPacketKind;
using simllm::rnic::NetworkPort;
using simllm::rnic::NetworkPortCapabilities;
using simllm::rnic::NetworkSubmitResult;
using simllm::rnic::NetworkToken;
using simllm::rnic::NetworkTxDescriptor;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceAttachments;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::RnicEcnStamp;
using simllm::rnic::RnicFirmwareCounterVariant;
using simllm::rnic::RnicHwProfile;
using simllm::rnic::RnicRecoveryMode;
using simllm::rnic::WorkRequest;
using simllm::rnic::WqeId;

// The endpoint-owned port the facade hands to the device. It records every
// accepted transmit so `rnic_cm_tx_next` can drain it, and turns a caller
// event back into the native event for the token it issued. It never
// synthesizes a timestamp of its own.
//
// Downstream packet-port contract, used here and by the test fake: the port
// returns one token per accepted attempt and the caller reports, for that
// token, TX finish, RX arrival and one terminal. The transmit pipeline is the
// transmit authority and stamps the TX start itself.
class CapturePort final : public NetworkPort {
public:
    struct Attempt {
        WqeId wqe_id{0};
        std::uint64_t wr_id{0};
        std::uint32_t qpn{0};
        std::uint32_t destination{0};
        std::uint32_t psn{0};
        std::uint32_t packet_index{0};
        std::uint32_t packet_count{1};
        std::uint32_t extent_index{0};
        std::uint64_t payload_offset_bytes{0};
        std::uint64_t payload_bytes{0};
        std::uint64_t wire_bytes{0};
        Picoseconds issued_at_ps{0};
        std::uint8_t traffic_class{0};
    };

    CapturePort(
        bool packetized,
        std::uint64_t mtu_bytes,
        std::uint64_t wire_header_bytes)
        : packetized_(packetized),
          mtu_bytes_(mtu_bytes),
          wire_header_bytes_(wire_header_bytes) {}

    NetworkPortCapabilities capabilities() const noexcept override {
        NetworkPortCapabilities caps;
        caps.abi_version = packetized_ ? kNetworkPortAbiVersionV2
                                       : kNetworkPortAbiVersionV1;
        caps.packet_attempt_events = packetized_;
        return caps;
    }

    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        const std::uint32_t expected = packetized_ ? kNetworkPortAbiVersionV2
                                                   : kNetworkPortAbiVersionV1;
        if (descriptor.abi_version != expected) {
            return NetworkSubmitResult::rejected();
        }
        const NetworkToken token = next_token_++;
        Attempt attempt;
        attempt.wqe_id = descriptor.wqe_id;
        attempt.wr_id = descriptor.wr_id;
        attempt.qpn = descriptor.qpn;
        attempt.destination = descriptor.destination;
        attempt.psn = next_psn_++;
        attempt.packet_index = descriptor.extent_index;
        attempt.packet_count = descriptor.extent_count;
        attempt.extent_index = descriptor.extent_index;
        // The packetizer segments at the MTU, so the packet index and the MTU
        // are the offset. A flow extent starts at zero by construction.
        attempt.payload_offset_bytes = packetized_
            ? static_cast<std::uint64_t>(descriptor.extent_index) * mtu_bytes_
            : 0;
        attempt.payload_bytes = descriptor.payload_bytes;
        attempt.wire_bytes = descriptor.payload_bytes + wire_header_bytes_;
        attempt.issued_at_ps = now_ps;
        attempt.traffic_class = descriptor.traffic_class;
        live_.emplace(token, attempt);
        emitted_.push_back(std::make_pair(token, attempt));
        return NetworkSubmitResult::accepted(token);
    }

    bool known(NetworkToken token) const {
        return live_.find(token) != live_.end();
    }

    const Attempt& attempt(NetworkToken token) const {
        return live_.at(token);
    }

    void retire(NetworkToken token) { live_.erase(token); }

    std::deque<std::pair<NetworkToken, Attempt>>& emitted() { return emitted_; }

private:
    bool packetized_;
    std::uint64_t mtu_bytes_;
    std::uint64_t wire_header_bytes_;
    NetworkToken next_token_{1};
    std::uint32_t next_psn_{0};
    std::map<NetworkToken, Attempt> live_;
    std::deque<std::pair<NetworkToken, Attempt>> emitted_;
};

void copyName(char* out, std::size_t bytes, const char* value) {
    std::memset(out, 0, bytes);
    if (value == nullptr) {
        return;
    }
    const std::size_t length = std::strlen(value);
    const std::size_t copied = length < bytes - 1 ? length : bytes - 1;
    std::memcpy(out, value, copied);
}

void toCProfile(const RnicHwProfile& profile, rnic_cm_profile* out) {
    std::memset(out, 0, sizeof(*out));
    out->version = profile.version;
    out->derived_link_factor = profile.derived_link_factor;
    copyName(out->name, SIMLLM_RNIC_CM_NAME_BYTES, profile.name);
    copyName(
        out->derived_from, SIMLLM_RNIC_CM_NAME_BYTES, profile.derived_from);
    out->link_bps = profile.link_bps;
    out->goodput_bps = profile.goodput_bps;
    out->mtu_bytes = profile.mtu_bytes;
    out->wire_header_bytes = profile.wire_header_bytes;
    out->t_eff_ps = profile.t_eff_ps;
    out->wire_round_trip_floor_ps = profile.wire_round_trip_floor_ps;
    out->doorbell_service_ps = profile.doorbell_service_ps;
    out->wqe_fetch_service_ps = profile.wqe_fetch_service_ps;
    out->qpc_lookup_service_ps = profile.qpc_lookup_service_ps;
    out->scheduler_service_ps = profile.scheduler_service_ps;
    out->cqe_write_service_ps = profile.cqe_write_service_ps;
    out->sq_depth = profile.sq_depth;
    out->max_inflight_bytes = profile.max_inflight_bytes;
    out->max_inflight_packets = profile.max_inflight_packets;
    out->tx_pps_per_qp = profile.tx_pps_per_qp;
    out->tx_pps_per_nic = profile.tx_pps_per_nic;
    out->rx_pps_per_qp_rc = profile.rx_pps_per_qp_rc;
    out->rx_pps_per_qp_ud = profile.rx_pps_per_qp_ud;
    out->rx_pps_per_nic = profile.rx_pps_per_nic;
    out->rx_ingress_bytes = profile.rx_ingress_bytes;
    out->rx_drain_bps = profile.rx_drain_bps;
    out->internal_budget_bps = profile.internal_budget_bps;
    out->rto_ps = profile.rto_ps;
    out->cnp_min_interval_ps = profile.cnp_min_interval_ps;
    out->dcqcn_alpha_update_ps = profile.dcqcn_alpha_update_ps;
    out->dcqcn_rate_reduce_ps = profile.dcqcn_rate_reduce_ps;
    out->dcqcn_byte_reset = profile.dcqcn_byte_reset;
    out->dcqcn_rate_step_bps = profile.dcqcn_rate_step_bps;
    out->selective_repeat_window = profile.selective_repeat_window;
    out->loopback_priority = profile.loopback_priority ? 1u : 0u;
    out->recovery = profile.recovery == RnicRecoveryMode::GoBackN ? 0u : 1u;
    out->ack_coalescing = profile.ack_coalescing ? 1u : 0u;
    out->dcqcn_enabled = profile.dcqcn_enabled ? 1u : 0u;
    out->ecn_stamp = static_cast<std::uint8_t>(profile.ecn_stamp);
    out->pfc_enabled = profile.pfc_enabled ? 1u : 0u;
    out->global_pause_tx = profile.global_pause_tx ? 1u : 0u;
    out->pause_propagates = profile.pause_propagates ? 1u : 0u;
    out->firmware_counter_variant =
        static_cast<std::uint8_t>(profile.firmware_counter_variant);
}

// Compares the hardware values of two profiles while ignoring identity and
// evidence. Rendering both with the same name and the default evidence block
// reuses the canonical renderer instead of growing a second field list that
// could drift from it.
bool sameProfileValues(RnicHwProfile lhs, RnicHwProfile rhs) {
    lhs.name = "fingerprint";
    rhs.name = "fingerprint";
    lhs.derived_from = "base";
    rhs.derived_from = "base";
    lhs.evidence = simllm::rnic::RnicHwProfileEvidence{};
    rhs.evidence = simllm::rnic::RnicHwProfileEvidence{};
    try {
        return simllm::rnic::renderRnicHwProfileJson(lhs)
            == simllm::rnic::renderRnicHwProfileJson(rhs);
    } catch (const std::exception&) {
        return false;
    }
}

std::string boundedName(const char* value, std::size_t bytes) {
    std::size_t length = 0;
    while (length < bytes && value[length] != '\0') {
        ++length;
    }
    return std::string(value, length);
}

RnicHwProfile valuesFromCProfile(const rnic_cm_profile& source);

// A caller-supplied parameter set that matches a preset exactly recovers the
// preset's evidence classes. Anything else is what it is: declared. Caller
// names are not carried into the native profile, whose identity strings are
// static storage.
RnicHwProfile fromCProfile(const rnic_cm_profile& source) {
    const std::string name =
        boundedName(source.name, SIMLLM_RNIC_CM_NAME_BYTES);
    RnicHwProfile preset;
    if (simllm::rnic::findRnicHwProfile(name, &preset)
        && sameProfileValues(preset, valuesFromCProfile(source))) {
        return preset;
    }
    return valuesFromCProfile(source);
}

RnicHwProfile valuesFromCProfile(const rnic_cm_profile& source) {
    RnicHwProfile profile;
    profile.version = source.version;
    profile.name = "";
    // Caller-supplied identity strings are not carried into the native
    // profile, whose identity is static storage. A derived parameter set
    // keeps its factor and names an unnamed base rather than failing
    // validation for a missing string.
    profile.derived_from = source.derived_link_factor != 0 ? "unnamed" : "";
    profile.derived_link_factor = source.derived_link_factor;
    profile.link_bps = source.link_bps;
    profile.goodput_bps = source.goodput_bps;
    profile.mtu_bytes = source.mtu_bytes;
    profile.wire_header_bytes = source.wire_header_bytes;
    profile.t_eff_ps = source.t_eff_ps;
    profile.wire_round_trip_floor_ps = source.wire_round_trip_floor_ps;
    profile.doorbell_service_ps = source.doorbell_service_ps;
    profile.wqe_fetch_service_ps = source.wqe_fetch_service_ps;
    profile.qpc_lookup_service_ps = source.qpc_lookup_service_ps;
    profile.scheduler_service_ps = source.scheduler_service_ps;
    profile.cqe_write_service_ps = source.cqe_write_service_ps;
    profile.sq_depth = source.sq_depth;
    profile.max_inflight_bytes = source.max_inflight_bytes;
    profile.max_inflight_packets = source.max_inflight_packets;
    profile.tx_pps_per_qp = source.tx_pps_per_qp;
    profile.tx_pps_per_nic = source.tx_pps_per_nic;
    profile.rx_pps_per_qp_rc = source.rx_pps_per_qp_rc;
    profile.rx_pps_per_qp_ud = source.rx_pps_per_qp_ud;
    profile.rx_pps_per_nic = source.rx_pps_per_nic;
    profile.rx_ingress_bytes = source.rx_ingress_bytes;
    profile.rx_drain_bps = source.rx_drain_bps;
    profile.internal_budget_bps = source.internal_budget_bps;
    profile.loopback_priority = source.loopback_priority != 0;
    profile.recovery = source.recovery == 0 ? RnicRecoveryMode::GoBackN
                                            : RnicRecoveryMode::SelectiveRepeat;
    profile.selective_repeat_window = source.selective_repeat_window;
    profile.rto_ps = source.rto_ps;
    profile.ack_coalescing = source.ack_coalescing != 0;
    profile.dcqcn_enabled = source.dcqcn_enabled != 0;
    profile.ecn_stamp = source.ecn_stamp == 0
        ? RnicEcnStamp::NotEct
        : (source.ecn_stamp == 1 ? RnicEcnStamp::Ect0 : RnicEcnStamp::Ect1);
    profile.cnp_min_interval_ps = source.cnp_min_interval_ps;
    profile.dcqcn_alpha_update_ps = source.dcqcn_alpha_update_ps;
    profile.dcqcn_rate_reduce_ps = source.dcqcn_rate_reduce_ps;
    profile.dcqcn_byte_reset = source.dcqcn_byte_reset;
    profile.dcqcn_rate_step_bps = source.dcqcn_rate_step_bps;
    profile.pfc_enabled = source.pfc_enabled != 0;
    profile.global_pause_tx = source.global_pause_tx != 0;
    profile.pause_propagates = source.pause_propagates != 0;
    profile.firmware_counter_variant = source.firmware_counter_variant == 0
        ? RnicFirmwareCounterVariant::Fw1632
        : RnicFirmwareCounterVariant::Fw1631;
    return profile;
}

std::uint32_t completionStatusCode(CompletionStatus status) noexcept {
    switch (status) {
    case CompletionStatus::Success:
        return RNIC_CM_COMPLETION_SUCCESS;
    case CompletionStatus::TransportError:
        return RNIC_CM_COMPLETION_TRANSPORT_ERROR;
    case CompletionStatus::NetworkRejected:
        return RNIC_CM_COMPLETION_NETWORK_REJECTED;
    }
    return RNIC_CM_COMPLETION_TRANSPORT_ERROR;
}

bool dropLocationFromCode(std::uint32_t code, DropLocation* out) noexcept {
    switch (code) {
    case RNIC_CM_DROP_LOCATION_TX_PORT:
        *out = DropLocation::TxPort;
        return true;
    case RNIC_CM_DROP_LOCATION_FABRIC:
        *out = DropLocation::Fabric;
        return true;
    case RNIC_CM_DROP_LOCATION_RX_PORT:
        *out = DropLocation::RxPort;
        return true;
    default:
        return false;
    }
}

bool dropReasonFromCode(std::uint32_t code, DropReason* out) noexcept {
    switch (code) {
    case RNIC_CM_DROP_REASON_INJECTED:
        *out = DropReason::Injected;
        return true;
    case RNIC_CM_DROP_REASON_QUEUE_OVERFLOW:
        *out = DropReason::QueueOverflow;
        return true;
    case RNIC_CM_DROP_REASON_LINK_DOWN:
        *out = DropReason::LinkDown;
        return true;
    case RNIC_CM_DROP_REASON_POLICY_REJECTED:
        *out = DropReason::PolicyRejected;
        return true;
    default:
        return false;
    }
}

}  // namespace

// The handle. It owns the port and the device and holds no timing state of
// its own beyond the trace and the emitted-packet queue.
struct rnic_cm_device {
    RnicHwProfile profile;
    std::string profile_name;
    std::string profile_sha256;
    bool packetized{false};
    bool trace_enabled{false};
    std::unique_ptr<CapturePort> port;
    std::unique_ptr<RnicDevice> device;
    std::vector<std::string> trace;
    Picoseconds last_time_ps{0};

    void note(Picoseconds now_ps, const std::string& line) {
        if (!trace_enabled) {
            return;
        }
        trace.push_back(std::to_string(now_ps) + " " + line);
    }
};

namespace {

int guard(rnic_cm_device* device) noexcept {
    if (device == nullptr || device->device == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    return RNIC_CM_OK;
}

std::string keyValue(const char* key, std::uint64_t value) {
    return std::string(key) + "=" + std::to_string(value);
}

}  // namespace

extern "C" {

int rnic_cm_profile_preset(const char* name, rnic_cm_profile* out) {
    if (name == nullptr || out == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    try {
        RnicHwProfile profile;
        if (!simllm::rnic::findRnicHwProfile(std::string(name), &profile)) {
            return RNIC_CM_ERROR_ARGUMENT;
        }
        toCProfile(profile, out);
        return RNIC_CM_OK;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_INTERNAL;
    }
}

int rnic_cm_profile_sha256(
    const rnic_cm_profile* profile,
    char* out,
    size_t bytes) {
    if (profile == nullptr || out == nullptr || bytes < 65) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    try {
        RnicHwProfile native = fromCProfile(*profile);
        if (native.name == nullptr || native.name[0] == '\0') {
            native.name = "caller";
        }
        const std::string digest = simllm::rnic::rnicHwProfileSha256(native);
        std::memcpy(out, digest.c_str(), digest.size() + 1);
        return RNIC_CM_OK;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
}

rnic_cm_device* rnic_cm_create(
    const rnic_cm_profile* profile,
    const rnic_cm_config* config) {
    if (profile == nullptr || config == nullptr) {
        return nullptr;
    }
    if (config->version != SIMLLM_RNIC_CM_ABI_VERSION) {
        return nullptr;
    }
    try {
        auto handle = std::make_unique<rnic_cm_device>();
        handle->profile = fromCProfile(*profile);
        if (handle->profile.name == nullptr
            || handle->profile.name[0] == '\0') {
            handle->profile.name = "caller";
        }
        simllm::rnic::validateRnicHwProfile(handle->profile);
        handle->profile_name = handle->profile.name;
        handle->profile_sha256 =
            simllm::rnic::rnicHwProfileSha256(handle->profile);
        handle->packetized = config->packetization != 0;
        handle->trace_enabled = config->trace_enabled != 0;

        if (config->sq_depth == 0 || config->cq_depth == 0
            || config->qpn == 0 || config->policy_context_token == 0) {
            return nullptr;
        }

        RnicDeviceConfig device_config;
        device_config.identity.qpn = config->qpn;
        device_config.identity.policy_context_token =
            config->policy_context_token;
        device_config.work_queue.qpn = config->qpn;
        device_config.work_queue.source = config->source;
        device_config.work_queue.policy_context_token =
            config->policy_context_token;
        device_config.work_queue.sq_depth =
            static_cast<std::size_t>(config->sq_depth);
        device_config.work_queue.cq_depth =
            static_cast<std::size_t>(config->cq_depth);
        device_config.work_queue.doorbell_service_ps =
            handle->profile.doorbell_service_ps;
        device_config.work_queue.wqe_fetch_service_ps =
            handle->profile.wqe_fetch_service_ps;
        device_config.work_queue.qpc_lookup_service_ps =
            handle->profile.qpc_lookup_service_ps;
        device_config.work_queue.scheduler_service_ps =
            handle->profile.scheduler_service_ps;
        device_config.work_queue.cqe_write_service_ps =
            handle->profile.cqe_write_service_ps;
        device_config.network.enabled = true;

        const std::uint64_t operating_mtu = config->mtu_bytes != 0
            ? config->mtu_bytes
            : handle->profile.mtu_bytes;
        if (handle->packetized) {
            simllm::rnic::RnicTxPipelineConfig pipeline;
            pipeline.enabled = true;
            pipeline.mtu_bytes = operating_mtu;
            pipeline.wire_header_bytes = handle->profile.wire_header_bytes;
            pipeline.max_inflight_wqes = config->max_inflight_wqes != 0
                ? config->max_inflight_wqes
                : config->sq_depth;
            pipeline.max_inflight_bytes = config->max_inflight_bytes;
            pipeline.max_inflight_packets =
                handle->profile.max_inflight_packets;
            // The pacer runs at the rate that makes a full calibration-MTU
            // packet deliver exactly the profile's goodput, so the header
            // bytes are paid once, here.
            const std::uint64_t wire_bps =
                simllm::rnic::effectiveWireBps(handle->profile);
            pipeline.wire_bps_per_qp = wire_bps;
            pipeline.wire_bps_per_nic = wire_bps;
            pipeline.message_rate_per_qp = handle->profile.tx_pps_per_qp;
            pipeline.message_rate_per_nic = handle->profile.tx_pps_per_nic;
            device_config.network.abi_version = kNetworkPortAbiVersionV2;
            device_config.network.packetization = pipeline;
        }

        handle->port = std::make_unique<CapturePort>(
            handle->packetized,
            operating_mtu,
            handle->profile.wire_header_bytes);
        RnicDeviceAttachments attachments;
        attachments.network_port = handle->port.get();
        handle->device = std::make_unique<RnicDevice>(
            device_config, attachments);

        handle->note(
            0,
            "create profile=" + handle->profile_name + " sha256="
                + handle->profile_sha256 + " "
                + keyValue("sq_depth", config->sq_depth) + " "
                + keyValue("cq_depth", config->cq_depth) + " "
                + keyValue("qpn", config->qpn) + " "
                + keyValue("packetization", handle->packetized ? 1 : 0));
        return handle.release();
    } catch (const std::exception&) {
        return nullptr;
    }
}

int rnic_cm_post(
    rnic_cm_device* device,
    const rnic_cm_wqe* wqe,
    uint64_t now_ps,
    uint64_t* out_wqe_id) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if (wqe == nullptr || wqe->opcode != 0) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    try {
        WorkRequest request;
        request.wr_id = wqe->wr_id;
        request.flow_id = static_cast<FlowId>(wqe->flow_id);
        request.flow_tag = wqe->flow_tag;
        request.destination = wqe->destination;
        request.payload_bytes = wqe->payload_bytes;
        request.traffic_class = wqe->traffic_class;
        request.signaled = wqe->signaled != 0;
        const auto result = device->device->postSend(request, now_ps);
        device->last_time_ps = now_ps;
        if (out_wqe_id != nullptr) {
            *out_wqe_id = result.wqe_id;
        }
        const char* status = result.status == PostStatus::Accepted
            ? "accepted"
            : (result.status == PostStatus::SqFull ? "sq_full" : "fatal");
        device->note(
            now_ps,
            "post " + keyValue("wr_id", wqe->wr_id) + " "
                + keyValue("bytes", wqe->payload_bytes) + " "
                + keyValue("sge", wqe->sge_count) + " "
                + keyValue("signaled", wqe->signaled != 0 ? 1 : 0) + " "
                + keyValue("dest", wqe->destination) + " status="
                + status + " " + keyValue("wqe", result.wqe_id) + " "
                + keyValue("seq", result.sq_sequence));
        if (result.status == PostStatus::Accepted) {
            return RNIC_CM_OK;
        }
        return result.status == PostStatus::SqFull ? RNIC_CM_ERROR_SQ_FULL
                                                   : RNIC_CM_ERROR_STATE;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_STATE;
    }
}

int rnic_cm_doorbell(
    rnic_cm_device* device,
    uint64_t now_ps,
    rnic_cm_doorbell_batch* out_batch) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    try {
        const auto batch = device->device->ringDoorbell(now_ps);
        device->last_time_ps = now_ps;
        if (out_batch != nullptr) {
            out_batch->batch_id = batch.batch_id;
            out_batch->wqe_count = batch.wqe_count;
            out_batch->rung_at_ps = batch.rung_at_ps;
            out_batch->observed_at_ps = batch.observed_at_ps;
        }
        device->note(
            now_ps,
            "doorbell " + keyValue("batch", batch.batch_id) + " "
                + keyValue("wqes", batch.wqe_count) + " "
                + keyValue("observed_at_ps", batch.observed_at_ps));
        return RNIC_CM_OK;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_STATE;
    }
}

int rnic_cm_rx_packet(
    rnic_cm_device* device,
    const rnic_cm_packet* packet,
    uint64_t now_ps) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if (packet == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    device->note(
        now_ps,
        "rx_packet " + keyValue("psn", packet->psn) + " status=unsupported");
    // The receive pipeline is BACK-56. Failing closed keeps a testbench from
    // reading silence as a modelled receive path.
    return RNIC_CM_ERROR_UNSUPPORTED;
}

int rnic_cm_event(
    rnic_cm_device* device,
    const rnic_cm_event_info* event,
    uint64_t now_ps) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if (event == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    if (event->kind >= RNIC_CM_EVENT_ECN_MARKED) {
        device->note(
            now_ps,
            "event " + keyValue("kind", event->kind) + " "
                + keyValue("token", event->token) + " status=unsupported");
        return RNIC_CM_ERROR_UNSUPPORTED;
    }
    if (!device->port->known(event->token)) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    const CapturePort::Attempt attempt = device->port->attempt(event->token);

    NetworkEvent native;
    native.abi_version = device->packetized ? kNetworkPortAbiVersionV2
                                            : kNetworkPortAbiVersionV1;
    native.token = event->token;
    native.wqe_id = attempt.wqe_id;
    native.event_time_ps = now_ps;
    native.extent_index = attempt.extent_index;
    native.packet_index = attempt.packet_index;
    native.payload_offset_bytes = attempt.payload_offset_bytes;
    native.payload_bytes = attempt.payload_bytes;
    native.wire_bytes = attempt.wire_bytes;
    native.packet_kind = NetworkPacketKind::Data;
    native.ecn_marked = event->ecn_marked != 0;

    bool terminal = false;
    switch (event->kind) {
    case RNIC_CM_EVENT_EXTENT_DELIVERED:
        if (device->packetized) {
            return RNIC_CM_ERROR_ARGUMENT;
        }
        native.scope = NetworkEventScope::FlowExtent;
        native.kind = NetworkEventKind::Delivered;
        terminal = true;
        break;
    case RNIC_CM_EVENT_EXTENT_DROPPED:
        if (device->packetized) {
            return RNIC_CM_ERROR_ARGUMENT;
        }
        native.scope = NetworkEventScope::FlowExtent;
        native.kind = NetworkEventKind::Dropped;
        terminal = true;
        break;
    case RNIC_CM_EVENT_PACKET_TX_FINISHED:
    case RNIC_CM_EVENT_PACKET_RX_ARRIVED:
    case RNIC_CM_EVENT_PACKET_DELIVERED:
    case RNIC_CM_EVENT_PACKET_DROPPED:
        if (!device->packetized) {
            return RNIC_CM_ERROR_ARGUMENT;
        }
        native.scope = NetworkEventScope::PacketAttempt;
        native.kind = event->kind == RNIC_CM_EVENT_PACKET_TX_FINISHED
            ? NetworkEventKind::PacketTxFinished
            : (event->kind == RNIC_CM_EVENT_PACKET_RX_ARRIVED
                   ? NetworkEventKind::PacketRxArrived
                   : (event->kind == RNIC_CM_EVENT_PACKET_DELIVERED
                          ? NetworkEventKind::Delivered
                          : NetworkEventKind::Dropped));
        terminal = event->kind == RNIC_CM_EVENT_PACKET_DELIVERED
            || event->kind == RNIC_CM_EVENT_PACKET_DROPPED;
        break;
    default:
        return RNIC_CM_ERROR_ARGUMENT;
    }

    const bool dropped = event->kind == RNIC_CM_EVENT_EXTENT_DROPPED
        || event->kind == RNIC_CM_EVENT_PACKET_DROPPED;
    if (dropped) {
        DropLocation location = DropLocation::None;
        DropReason reason = DropReason::None;
        if (!dropLocationFromCode(event->drop_location, &location)
            || !dropReasonFromCode(event->drop_reason, &reason)) {
            return RNIC_CM_ERROR_ARGUMENT;
        }
        native.drop_location = location;
        native.drop_reason = reason;
        native.drop_evidence =
            simllm::rnic::DropEvidenceProvenance::Controlled;
        native.drop_resource_id = event->token;
    }

    try {
        device->device->onNetworkEvent(native);
        device->last_time_ps = now_ps;
        if (terminal) {
            device->port->retire(event->token);
        }
        device->note(
            now_ps,
            "event " + keyValue("kind", event->kind) + " "
                + keyValue("token", event->token) + " "
                + keyValue("wqe", attempt.wqe_id) + " status=ok");
        return RNIC_CM_OK;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_STATE;
    }
}

int rnic_cm_progress(
    rnic_cm_device* device,
    uint64_t now_ps,
    uint64_t* out_changes) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    try {
        const std::size_t changes = device->device->progress(now_ps);
        device->last_time_ps = now_ps;
        if (out_changes != nullptr) {
            *out_changes = static_cast<std::uint64_t>(changes);
        }
        device->note(now_ps, "progress " + keyValue("changes", changes));
        return RNIC_CM_OK;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_STATE;
    }
}

int rnic_cm_next_event_ps(rnic_cm_device* device, uint64_t* out_now_ps) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if (out_now_ps == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    try {
        const auto next = device->device->nextEventTime();
        if (!next.has_value()) {
            return RNIC_CM_NO_EVENT;
        }
        *out_now_ps = *next;
        return RNIC_CM_OK;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_STATE;
    }
}

int rnic_cm_poll(
    rnic_cm_device* device,
    rnic_cm_cqe* out,
    size_t max_entries,
    uint64_t now_ps,
    size_t* out_count) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if ((out == nullptr && max_entries != 0) || out_count == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    try {
        const std::vector<CompletionEntry> entries =
            device->device->pollCompletionQueue(max_entries, now_ps);
        device->last_time_ps = now_ps;
        *out_count = entries.size();
        for (std::size_t index = 0; index < entries.size(); ++index) {
            const CompletionEntry& entry = entries[index];
            rnic_cm_cqe& slot = out[index];
            std::memset(&slot, 0, sizeof(slot));
            slot.cqe_sequence = entry.cqe_sequence;
            slot.wr_id = entry.wr_id;
            slot.wqe_id = entry.wqe_id;
            slot.sq_sequence = entry.sq_sequence;
            slot.byte_count = entry.byte_count;
            slot.visible_at_ps = entry.visible_at_ps;
            slot.polled_at_ps = entry.polled_at_ps;
            slot.qpn = entry.qpn;
            slot.status = completionStatusCode(entry.status);
            slot.opcode = 0;
            slot.valid_fields = entry.valid_fields;
            device->note(
                now_ps,
                "cqe " + keyValue("seq", entry.cqe_sequence) + " "
                    + keyValue("wr_id", entry.wr_id) + " "
                    + keyValue("wqe", entry.wqe_id) + " "
                    + keyValue("status", slot.status) + " "
                    + keyValue("bytes", entry.byte_count) + " "
                    + keyValue("visible_at_ps", entry.visible_at_ps));
        }
        device->note(now_ps, "poll " + keyValue("count", entries.size()));
        return RNIC_CM_OK;
    } catch (const std::exception&) {
        return RNIC_CM_ERROR_STATE;
    }
}

int rnic_cm_tx_next(
    rnic_cm_device* device,
    rnic_cm_packet* out,
    size_t max_packets,
    size_t* out_count) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if ((out == nullptr && max_packets != 0) || out_count == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    std::size_t count = 0;
    auto& emitted = device->port->emitted();
    while (count < max_packets && !emitted.empty()) {
        const auto entry = emitted.front();
        emitted.pop_front();
        rnic_cm_packet& slot = out[count];
        std::memset(&slot, 0, sizeof(slot));
        slot.token = entry.first;
        slot.wqe_id = entry.second.wqe_id;
        slot.wr_id = entry.second.wr_id;
        slot.payload_offset_bytes = entry.second.payload_offset_bytes;
        slot.payload_bytes = entry.second.payload_bytes;
        slot.wire_bytes = entry.second.wire_bytes;
        slot.issued_at_ps = entry.second.issued_at_ps;
        slot.qpn = entry.second.qpn;
        slot.destination = entry.second.destination;
        slot.psn = entry.second.psn;
        slot.packet_index = entry.second.packet_index;
        slot.packet_count = entry.second.packet_count;
        slot.kind = RNIC_CM_PACKET_DATA;
        slot.traffic_class = entry.second.traffic_class;
        device->note(
            entry.second.issued_at_ps,
            "packet " + keyValue("token", slot.token) + " "
                + keyValue("wqe", slot.wqe_id) + " "
                + keyValue("psn", slot.psn) + " "
                + keyValue("index", slot.packet_index) + " "
                + keyValue("count", slot.packet_count) + " "
                + keyValue("offset", slot.payload_offset_bytes) + " "
                + keyValue("bytes", slot.payload_bytes) + " "
                + keyValue("wire", slot.wire_bytes));
        ++count;
    }
    *out_count = count;
    return RNIC_CM_OK;
}

int rnic_cm_counters(rnic_cm_device* device, rnic_cm_counter_set* out) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if (out == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    const auto& counters = device->device->counters();
    std::memset(out, 0, sizeof(*out));
    out->version = SIMLLM_RNIC_CM_ABI_VERSION;
    out->posted_wqes = counters.posted_wqes;
    out->sq_full_rejections = counters.sq_full_rejections;
    out->doorbells = counters.doorbells;
    out->doorbelled_wqes = counters.doorbelled_wqes;
    out->network_submit_attempts = counters.network_submit_attempts;
    out->network_accepted = counters.network_accepted;
    out->network_busy = counters.network_busy;
    out->network_rejected = counters.network_rejected;
    out->network_delivered = counters.network_delivered;
    out->network_dropped = counters.network_dropped;
    out->cqes_visible = counters.cqes_visible;
    out->cqes_polled = counters.cqes_polled;
    out->cq_overruns = counters.cq_overruns;
    out->sq_reclaimed_wqes = counters.sq_reclaimed_wqes;
    out->sq_high_watermark =
        static_cast<std::uint64_t>(counters.sq_high_watermark);
    out->cq_high_watermark =
        static_cast<std::uint64_t>(counters.cq_high_watermark);
    const simllm::rnic::RnicTxPipeline* pipeline =
        device->device->txPipeline();
    if (pipeline != nullptr) {
        const auto& tx = pipeline->counters();
        out->tx_packets = tx.packets_issued;
        out->tx_payload_bytes = tx.payload_bytes;
        out->tx_wire_bytes = tx.wire_bytes;
        out->tx_window_stalls = tx.window_stalls;
        out->tx_pacer_stalls = tx.pacer_stalls;
        out->tx_inflight_wqes = tx.inflight_wqes;
        out->tx_inflight_bytes = tx.inflight_bytes;
        out->tx_late_releases = tx.late_releases;
        out->tx_packets_dropped = tx.packets_dropped;
    }
    return RNIC_CM_OK;
}

int rnic_cm_trace(rnic_cm_device* device, const char* path) {
    const int ready = guard(device);
    if (ready != RNIC_CM_OK) {
        return ready;
    }
    if (path == nullptr) {
        return RNIC_CM_ERROR_ARGUMENT;
    }
    if (!device->trace_enabled) {
        return RNIC_CM_ERROR_STATE;
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.is_open()) {
        return RNIC_CM_ERROR_STATE;
    }
    for (const std::string& line : device->trace) {
        stream << line << '\n';
    }
    stream.flush();
    return stream.good() ? RNIC_CM_OK : RNIC_CM_ERROR_INTERNAL;
}

void rnic_cm_destroy(rnic_cm_device* device) {
    if (device == nullptr) {
        return;
    }
    delete device;
}

}  // extern "C"
