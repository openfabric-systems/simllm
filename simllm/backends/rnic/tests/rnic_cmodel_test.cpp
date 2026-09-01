#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fake_network.h"

#include "simllm/rnic/rnic_anomaly_table.h"
#include "simllm/rnic/rnic_cmodel_c.h"
#include "simllm/rnic/rnic_device.h"
#include "simllm/rnic/rnic_hw_profile.h"
#include "simllm/rnic/rnic_tx_pipeline.h"
#include "simllm/rnic/session_record.h"

namespace {

using simllm::rnic::AnomalyKind;
using simllm::rnic::CompletionEntry;
using simllm::rnic::EvidenceClass;
using simllm::rnic::kConnectX5_100G;
using simllm::rnic::kConnectX7_400G;
using simllm::rnic::kNetworkPortAbiVersionV1;
using simllm::rnic::kRnicAnomalyTable;
using simllm::rnic::kRnicAnomalyRowCount;
using simllm::rnic::kRnicHwProfileSchema;
using simllm::rnic::NetworkEvent;
using simllm::rnic::NetworkEventKind;
using simllm::rnic::NetworkPort;
using simllm::rnic::NetworkPortCapabilities;
using simllm::rnic::NetworkSubmitResult;
using simllm::rnic::NetworkToken;
using simllm::rnic::NetworkTxDescriptor;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::RnicAnomalyRow;
using simllm::rnic::RnicDevice;
using simllm::rnic::RnicDeviceAttachments;
using simllm::rnic::RnicDeviceConfig;
using simllm::rnic::RnicHwProfile;
using simllm::rnic::WorkRequest;

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
            check(false, message + "; wrong exception: " + error.what());
        } catch (...) {
            check(false, message + "; wrong non-standard exception");
        }
    }

    std::size_t failures() const noexcept { return failures_; }

private:
    std::size_t failures_{0};
};

constexpr std::uint32_t kQpn = 7;
constexpr std::uint64_t kPolicyToken = 31;
constexpr std::uint64_t kWqeCount = 6;
constexpr std::uint64_t kPayloadBytes = 8192;
constexpr Picoseconds kWireLatencyPs = 1050000;

// Accepts every extent with the same token sequence the facade's own port
// uses, so the oracle device and the facade device see one identical wire.
class OraclePort final : public NetworkPort {
public:
    struct Emitted {
        NetworkToken token{0};
        simllm::rnic::WqeId wqe_id{0};
        Picoseconds issued_at_ps{0};
    };

    NetworkPortCapabilities capabilities() const noexcept override {
        NetworkPortCapabilities caps;
        caps.abi_version = kNetworkPortAbiVersionV1;
        return caps;
    }

    NetworkSubmitResult trySubmit(
        const NetworkTxDescriptor& descriptor,
        Picoseconds now_ps) override {
        const NetworkToken token = next_token_++;
        emitted_.push_back(Emitted{token, descriptor.wqe_id, now_ps});
        return NetworkSubmitResult::accepted(token);
    }

    std::vector<Emitted> drain() {
        std::vector<Emitted> out;
        out.swap(emitted_);
        return out;
    }

private:
    NetworkToken next_token_{1};
    std::vector<Emitted> emitted_;
};

struct CompletionRow {
    std::uint64_t cqe_sequence{0};
    std::uint64_t wr_id{0};
    std::uint64_t wqe_id{0};
    std::uint64_t sq_sequence{0};
    std::uint64_t byte_count{0};
    std::uint64_t visible_at_ps{0};
    std::uint64_t polled_at_ps{0};
    std::uint32_t status{0};
};

bool sameRow(const CompletionRow& lhs, const CompletionRow& rhs) {
    return lhs.cqe_sequence == rhs.cqe_sequence && lhs.wr_id == rhs.wr_id
        && lhs.wqe_id == rhs.wqe_id && lhs.sq_sequence == rhs.sq_sequence
        && lhs.byte_count == rhs.byte_count
        && lhs.visible_at_ps == rhs.visible_at_ps
        && lhs.polled_at_ps == rhs.polled_at_ps && lhs.status == rhs.status;
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

// Runs the reference stimulus through the C++ device directly.
std::vector<CompletionRow> runOracle(const RnicHwProfile& profile) {
    OraclePort port;
    RnicDeviceConfig config;
    config.identity.qpn = kQpn;
    config.identity.policy_context_token = kPolicyToken;
    config.work_queue.qpn = kQpn;
    config.work_queue.policy_context_token = kPolicyToken;
    config.work_queue.sq_depth = 4;
    config.work_queue.cq_depth = 8;
    config.work_queue.doorbell_service_ps = profile.doorbell_service_ps;
    config.work_queue.wqe_fetch_service_ps = profile.wqe_fetch_service_ps;
    config.work_queue.qpc_lookup_service_ps = profile.qpc_lookup_service_ps;
    config.work_queue.scheduler_service_ps = profile.scheduler_service_ps;
    config.work_queue.cqe_write_service_ps = profile.cqe_write_service_ps;
    config.network.enabled = true;
    RnicDeviceAttachments attachments;
    attachments.network_port = &port;
    RnicDevice device(config, attachments);

    std::map<Picoseconds, std::vector<std::pair<NetworkToken, std::uint64_t>>>
        wire;
    std::vector<CompletionRow> rows;
    std::uint64_t next_wr = 1;
    Picoseconds now_ps = 0;
    std::uint64_t completed = 0;
    std::size_t guard = 0;

    while (completed < kWqeCount) {
        if (++guard > 100000) {
            throw std::runtime_error("oracle loop did not converge");
        }
        bool posted = false;
        while (next_wr <= kWqeCount && device.occupiedSqEntries() < 4) {
            WorkRequest request;
            request.wr_id = next_wr;
            request.destination = 3;
            request.payload_bytes = kPayloadBytes;
            request.signaled = true;
            if (device.postSend(request, now_ps).status
                != PostStatus::Accepted) {
                break;
            }
            ++next_wr;
            posted = true;
        }
        if (posted) {
            device.ringDoorbell(now_ps);
        }
        for (const auto& emitted : port.drain()) {
            wire[emitted.issued_at_ps + kWireLatencyPs].push_back(
                std::make_pair(emitted.token, emitted.wqe_id));
        }
        device.progress(now_ps);
        for (const auto& emitted : port.drain()) {
            wire[emitted.issued_at_ps + kWireLatencyPs].push_back(
                std::make_pair(emitted.token, emitted.wqe_id));
        }
        for (const CompletionEntry& entry : device.pollCompletionQueue(
                 std::numeric_limits<std::size_t>::max(), now_ps)) {
            rows.push_back(CompletionRow{
                entry.cqe_sequence,
                entry.wr_id,
                entry.wqe_id,
                entry.sq_sequence,
                entry.byte_count,
                entry.visible_at_ps,
                entry.polled_at_ps,
                0});
            ++completed;
        }

        std::optional<Picoseconds> wire_time;
        if (!wire.empty()) {
            wire_time = wire.begin()->first;
        }
        const auto next = earlier(device.nextEventTime(), wire_time);
        if (!next.has_value()) {
            if (next_wr > kWqeCount && completed >= kWqeCount) {
                break;
            }
            throw std::runtime_error("oracle stalled with pending work");
        }
        now_ps = std::max(now_ps, *next);
        while (!wire.empty() && wire.begin()->first <= now_ps) {
            const Picoseconds due = wire.begin()->first;
            for (const auto& token_and_wqe : wire.begin()->second) {
                NetworkEvent event;
                event.kind = NetworkEventKind::Delivered;
                event.token = token_and_wqe.first;
                event.wqe_id = token_and_wqe.second;
                event.event_time_ps = due;
                device.onNetworkEvent(event);
            }
            wire.erase(wire.begin());
        }
    }
    device.validateInvariants();
    return rows;
}

// Runs the same stimulus through the C facade.
std::vector<CompletionRow> runFacade(
    const rnic_cm_profile& profile,
    bool trace_enabled,
    const std::string& trace_path,
    TestRunner& test) {
    rnic_cm_config config;
    std::memset(&config, 0, sizeof(config));
    config.version = SIMLLM_RNIC_CM_ABI_VERSION;
    config.qpn = kQpn;
    config.source = 0;
    config.policy_context_token = kPolicyToken;
    config.sq_depth = 4;
    config.cq_depth = 8;
    config.packetization = 0;
    config.trace_enabled = trace_enabled ? 1u : 0u;

    rnic_cm_device* device = rnic_cm_create(&profile, &config);
    if (device == nullptr) {
        throw std::runtime_error("facade construction failed");
    }

    std::map<Picoseconds, std::vector<std::uint64_t>> wire;
    std::vector<CompletionRow> rows;
    std::uint64_t next_wr = 1;
    Picoseconds now_ps = 0;
    std::uint64_t completed = 0;
    std::size_t guard = 0;
    std::vector<rnic_cm_packet> packets(16);
    std::vector<rnic_cm_cqe> cqes(16);

    const auto drain_tx = [&]() {
        std::size_t count = 0;
        do {
            test.check(
                rnic_cm_tx_next(device, packets.data(), packets.size(), &count)
                    == RNIC_CM_OK,
                "facade tx_next succeeds");
            for (std::size_t index = 0; index < count; ++index) {
                wire[packets[index].issued_at_ps + kWireLatencyPs].push_back(
                    packets[index].token);
            }
        } while (count == packets.size());
    };

    while (completed < kWqeCount) {
        if (++guard > 100000) {
            rnic_cm_destroy(device);
            throw std::runtime_error("facade loop did not converge");
        }
        bool posted = false;
        while (next_wr <= kWqeCount) {
            rnic_cm_wqe request;
            std::memset(&request, 0, sizeof(request));
            request.wr_id = next_wr;
            request.destination = 3;
            request.payload_bytes = kPayloadBytes;
            request.sge_count = 1;
            request.signaled = 1;
            std::uint64_t wqe_id = 0;
            const int status = rnic_cm_post(device, &request, now_ps, &wqe_id);
            if (status == RNIC_CM_ERROR_SQ_FULL) {
                break;
            }
            if (status != RNIC_CM_OK) {
                rnic_cm_destroy(device);
                throw std::runtime_error("facade post failed");
            }
            ++next_wr;
            posted = true;
        }
        if (posted) {
            rnic_cm_doorbell_batch batch;
            test.check(
                rnic_cm_doorbell(device, now_ps, &batch) == RNIC_CM_OK,
                "facade doorbell succeeds");
        }
        drain_tx();
        std::uint64_t changes = 0;
        test.check(
            rnic_cm_progress(device, now_ps, &changes) == RNIC_CM_OK,
            "facade progress succeeds");
        drain_tx();

        std::size_t polled = 0;
        test.check(
            rnic_cm_poll(device, cqes.data(), cqes.size(), now_ps, &polled)
                == RNIC_CM_OK,
            "facade poll succeeds");
        for (std::size_t index = 0; index < polled; ++index) {
            rows.push_back(CompletionRow{
                cqes[index].cqe_sequence,
                cqes[index].wr_id,
                cqes[index].wqe_id,
                cqes[index].sq_sequence,
                cqes[index].byte_count,
                cqes[index].visible_at_ps,
                cqes[index].polled_at_ps,
                cqes[index].status});
            ++completed;
        }

        std::uint64_t device_next = 0;
        const int has_next = rnic_cm_next_event_ps(device, &device_next);
        std::optional<Picoseconds> next;
        if (has_next == RNIC_CM_OK) {
            next = device_next;
        }
        if (!wire.empty()) {
            next = earlier(next, wire.begin()->first);
        }
        if (!next.has_value()) {
            if (next_wr > kWqeCount && completed >= kWqeCount) {
                break;
            }
            rnic_cm_destroy(device);
            throw std::runtime_error("facade stalled with pending work");
        }
        now_ps = std::max(now_ps, *next);
        while (!wire.empty() && wire.begin()->first <= now_ps) {
            const Picoseconds due = wire.begin()->first;
            for (const std::uint64_t token : wire.begin()->second) {
                rnic_cm_event_info event;
                std::memset(&event, 0, sizeof(event));
                event.kind = RNIC_CM_EVENT_EXTENT_DELIVERED;
                event.token = token;
                test.check(
                    rnic_cm_event(device, &event, due) == RNIC_CM_OK,
                    "facade delivery event accepted");
            }
            wire.erase(wire.begin());
        }
    }

    if (trace_enabled) {
        test.check(
            rnic_cm_trace(device, trace_path.c_str()) == RNIC_CM_OK,
            "facade trace written");
    }
    rnic_cm_destroy(device);
    return rows;
}

// Built from a separator character rather than a literal so no tracked source
// carries something that reads as an absolute path.
std::string joinPath(const std::string& directory, const std::string& name) {
    if (directory.empty()) {
        return name;
    }
    return directory + '/' + name;
}

std::string readFile(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void testProfile(TestRunner& test) {
    simllm::rnic::validateRnicHwProfile(kConnectX5_100G);
    simllm::rnic::validateRnicHwProfile(kConnectX7_400G);

    test.check(
        std::string(kConnectX5_100G.name) == "cx5_100g",
        "cx5 profile is named");
    test.check(
        kConnectX5_100G.link_bps == 100000000000ULL
            && kConnectX5_100G.goodput_bps == 97100000000ULL
            && kConnectX5_100G.mtu_bytes == 4096
            && kConnectX5_100G.wire_header_bytes == 64,
        "cx5 link group carries the measured constants");
    test.check(
        kConnectX5_100G.t_eff_ps == 4480000
            && kConnectX5_100G.doorbell_service_ps
                    + kConnectX5_100G.wqe_fetch_service_ps
                    + kConnectX5_100G.qpc_lookup_service_ps
                    + kConnectX5_100G.scheduler_service_ps
                    + kConnectX5_100G.cqe_write_service_ps
                    + kConnectX5_100G.wire_round_trip_floor_ps
                == kConnectX5_100G.t_eff_ps,
        "cx5 stage split reconstructs the lumped fixed offset");
    test.check(
        kConnectX5_100G.tx_pps_per_qp == 3870000
            && kConnectX5_100G.rx_pps_per_qp_ud == 3070000
            && kConnectX5_100G.sq_depth == 1024,
        "cx5 packet-rate and queue constants are the measured ones");
    test.check(
        kConnectX5_100G.evidence.goodput_bps
                == EvidenceClass::CalibratedOpaque
            && kConnectX5_100G.evidence.mtu_bytes == EvidenceClass::Documented
            && kConnectX5_100G.evidence.qpc_lookup_service_ps
                == EvidenceClass::Declared
            && kConnectX5_100G.evidence.cnp_min_interval_ps
                == EvidenceClass::Declared,
        "cx5 evidence classes match how each value was established");

    test.check(
        kConnectX7_400G.link_bps == 400000000000ULL
            && kConnectX7_400G.goodput_bps == 388400000000ULL
            && kConnectX7_400G.tx_pps_per_qp == 15480000
            && kConnectX7_400G.rx_pps_per_qp_ud == 12280000
            && kConnectX7_400G.internal_budget_bps == 788000000000ULL,
        "cx7 scales the link, goodput, packet-rate and threshold fields");
    test.check(
        kConnectX7_400G.t_eff_ps == kConnectX5_100G.t_eff_ps
            && kConnectX7_400G.mtu_bytes == kConnectX5_100G.mtu_bytes
            && kConnectX7_400G.wire_header_bytes
                == kConnectX5_100G.wire_header_bytes
            && kConnectX7_400G.sq_depth == kConnectX5_100G.sq_depth
            && kConnectX7_400G.rto_ps == kConnectX5_100G.rto_ps
            && kConnectX7_400G.pfc_enabled == kConnectX5_100G.pfc_enabled,
        "cx7 keeps the initiation, MTU, header, transport and flow-control "
        "fields");
    test.check(
        kConnectX7_400G.evidence.link_bps == EvidenceClass::Declared
            && kConnectX7_400G.evidence.goodput_bps == EvidenceClass::Declared
            && kConnectX7_400G.evidence.tx_pps_per_qp == EvidenceClass::Declared
            && kConnectX7_400G.evidence.mtu_bytes == EvidenceClass::Documented
            && kConnectX7_400G.evidence.t_eff_ps
                == EvidenceClass::CalibratedOpaque,
        "every scaled cx7 field is declared and every kept field is not");
    test.check(
        std::string(kConnectX7_400G.derived_from) == "cx5_100g"
            && kConnectX7_400G.derived_link_factor == 4,
        "cx7 names its base profile and factor");

    test.check(
        simllm::rnic::effectiveWireBps(kConnectX5_100G) == 98617187500ULL,
        "the effective wire rate delivers exactly the goodput at full MTU");
    test.check(
        simllm::rnic::packetsForMessage(kConnectX5_100G, 0) == 1
            && simllm::rnic::packetsForMessage(kConnectX5_100G, 1) == 1
            && simllm::rnic::packetsForMessage(kConnectX5_100G, 4096) == 1
            && simllm::rnic::packetsForMessage(kConnectX5_100G, 4097) == 2
            && simllm::rnic::packetsForMessage(kConnectX5_100G, 1048576)
                == 256,
        "MTU segmentation counts packets by ceiling");

    const std::string json =
        simllm::rnic::renderRnicHwProfileJson(kConnectX5_100G);
    test.check(
        json.find(kRnicHwProfileSchema) != std::string::npos
            && json.find("\"calibrated-opaque\"") != std::string::npos
            && json.find('.') == std::string::npos,
        "the profile record is schema tagged and has no floating point");
    test.check(
        simllm::rnic::rnicHwProfileSha256(kConnectX5_100G)
            == simllm::rnic::rnicHwProfileSha256(kConnectX5_100G),
        "the profile hash is reproducible");
    test.check(
        simllm::rnic::rnicHwProfileSha256(kConnectX5_100G)
            != simllm::rnic::rnicHwProfileSha256(kConnectX7_400G),
        "the two profiles hash differently");

    const auto record = simllm::rnic::makeRnicHwProfileRecord(kConnectX5_100G);
    test.check(
        record.schema == kRnicHwProfileSchema && record.name == "cx5_100g"
            && record.sha256.size() == 64,
        "the profile record carries its own schema, name and digest");

    // The profile record must not be the effective-hardware record: a policy
    // comparison identity and a hardware calibration identity are different
    // objects and must hash separately.
    OraclePort port;
    RnicDeviceConfig device_config;
    device_config.network.enabled = true;
    RnicDeviceAttachments attachments;
    attachments.network_port = &port;
    RnicDevice device(device_config, attachments);
    const std::string effective =
        simllm::rnic::renderEffectiveHardwareConfigJson(device);
    test.check(
        effective.find(kRnicHwProfileSchema) == std::string::npos
            && simllm::rnic::effectiveHardwareConfigSha256(device)
                != record.sha256,
        "the profile hash is separate from the effective-hardware hash");

    RnicHwProfile found;
    test.check(
        simllm::rnic::findRnicHwProfile("cx7_400g", &found)
            && found.link_bps == 400000000000ULL
            && !simllm::rnic::findRnicHwProfile("nonexistent", &found),
        "profiles are found by their stable names");

    RnicHwProfile broken = kConnectX5_100G;
    broken.qpc_lookup_service_ps += 1;
    test.expectThrowAs<std::invalid_argument>(
        [&broken]() { simllm::rnic::validateRnicHwProfile(broken); },
        "a stage split that does not reconstruct t_eff is rejected");
    RnicHwProfile overspeed = kConnectX5_100G;
    overspeed.goodput_bps = overspeed.link_bps + 1;
    test.expectThrowAs<std::invalid_argument>(
        [&overspeed]() { simllm::rnic::validateRnicHwProfile(overspeed); },
        "a goodput above the link rate is rejected");
}

void testAnomalyTable(
    TestRunner& test,
    const std::string& projection_path,
    const std::string& design_path) {
    test.check(
        kRnicAnomalyTable.size() == kRnicAnomalyRowCount
            && kRnicAnomalyRowCount == 15,
        "the anomaly table has the registered row count");

    std::vector<std::string> ids;
    for (const RnicAnomalyRow& row : kRnicAnomalyTable) {
        ids.push_back(row.id);
        const std::string kind_text = row.kind_text;
        const std::string kind = simllm::rnic::toString(row.kind);
        test.check(
            kind_text.rfind(kind, 0) == 0,
            std::string("row ") + row.id + " kind text starts with its kind");
        test.check(
            row.magnitude != nullptr && row.magnitude[0] != '\0',
            std::string("row ") + row.id + " carries a magnitude handle");
        test.check(
            row.evidence != nullptr && row.evidence[0] != '\0',
            std::string("row ") + row.id + " names its evidence");
    }
    std::vector<std::string> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    test.check(
        std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()
            && ids == sorted,
        "anomaly ids are unique and in order");

    const std::string rendered =
        simllm::rnic::renderRnicAnomalyTableMarkdown();
    if (!projection_path.empty()) {
        const std::string committed = readFile(projection_path);
        test.check(
            rendered == committed,
            "the rendered anomaly table equals the committed projection byte "
            "for byte");
    }
    if (!design_path.empty()) {
        const std::string design = readFile(design_path);
        for (const RnicAnomalyRow& row : kRnicAnomalyTable) {
            const std::string line = simllm::rnic::renderRnicAnomalyRow(row);
            test.check(
                design.find(line) != std::string::npos,
                std::string("the design document still carries ") + row.id);
        }
    }
}

struct PipelineRun {
    Picoseconds last_completion_ps{0};
    std::uint64_t completions{0};
    std::uint64_t errors{0};
    std::uint64_t packets{0};
    std::uint64_t packets_dropped{0};
    std::uint64_t payload_bytes{0};
    std::uint64_t wire_bytes{0};
    std::uint64_t late_releases{0};
    Picoseconds first_packet_ps{0};
    Picoseconds last_packet_ps{0};
};

// Drives one closed-loop cell through the device with the transmit pipeline
// enabled and the v2 fake wire behind it.
PipelineRun runPipeline(
    const RnicHwProfile& profile,
    std::uint64_t message_bytes,
    std::size_t depth,
    std::uint64_t mtu_bytes,
    std::uint64_t messages,
    bool drop_first_packet) {
    simllm::rnic::testing::FakeV2NetworkConfig wire_config;
    wire_config.link_bps = profile.link_bps;
    wire_config.one_way_latency_ps = profile.wire_round_trip_floor_ps / 2;
    wire_config.capacity = 1 << 20;
    simllm::rnic::testing::FakeV2NetworkPort wire(wire_config);
    wire.setWireHeaderBytes(profile.wire_header_bytes);
    if (drop_first_packet) {
        wire.dropNext();
    }

    RnicDeviceConfig config;
    config.identity.qpn = kQpn;
    config.identity.policy_context_token = kPolicyToken;
    config.work_queue.qpn = kQpn;
    config.work_queue.policy_context_token = kPolicyToken;
    config.work_queue.sq_depth = depth;
    config.work_queue.cq_depth = depth * 2;
    config.work_queue.doorbell_service_ps = profile.doorbell_service_ps;
    config.work_queue.wqe_fetch_service_ps = profile.wqe_fetch_service_ps;
    config.work_queue.qpc_lookup_service_ps = profile.qpc_lookup_service_ps;
    config.work_queue.scheduler_service_ps = profile.scheduler_service_ps;
    config.work_queue.cqe_write_service_ps = profile.cqe_write_service_ps;
    config.network.enabled = true;
    config.network.abi_version = simllm::rnic::kNetworkPortAbiVersionV2;
    config.network.packetization.enabled = true;
    config.network.packetization.mtu_bytes = mtu_bytes;
    config.network.packetization.wire_header_bytes = profile.wire_header_bytes;
    config.network.packetization.max_inflight_wqes = depth;
    const std::uint64_t wire_bps = simllm::rnic::effectiveWireBps(profile);
    config.network.packetization.wire_bps_per_qp = wire_bps;
    config.network.packetization.wire_bps_per_nic = wire_bps;
    config.network.packetization.message_rate_per_qp = profile.tx_pps_per_qp;
    config.network.packetization.message_rate_per_nic = profile.tx_pps_per_nic;
    RnicDeviceAttachments attachments;
    attachments.network_port = &wire;
    RnicDevice device(config, attachments);

    PipelineRun run;
    std::uint64_t next_wr = 1;
    Picoseconds now_ps = 0;
    std::size_t guard = 0;
    while (run.completions + run.errors < messages) {
        if (++guard > 4000000) {
            throw std::runtime_error("pipeline loop did not converge");
        }
        for (const NetworkEvent& event : wire.takeDue(now_ps)) {
            device.onNetworkEvent(event);
        }
        bool posted = false;
        while (next_wr <= messages && device.occupiedSqEntries() < depth) {
            WorkRequest request;
            request.wr_id = next_wr;
            request.destination = 3;
            request.payload_bytes = message_bytes;
            request.signaled = true;
            if (device.postSend(request, now_ps).status
                != PostStatus::Accepted) {
                break;
            }
            ++next_wr;
            posted = true;
        }
        if (posted) {
            device.ringDoorbell(now_ps);
        }
        device.progress(now_ps);
        for (const CompletionEntry& entry : device.pollCompletionQueue(
                 std::numeric_limits<std::size_t>::max(), now_ps)) {
            if (entry.status == simllm::rnic::CompletionStatus::Success) {
                ++run.completions;
            } else {
                ++run.errors;
            }
            run.last_completion_ps = entry.polled_at_ps;
        }
        const auto next = earlier(device.nextEventTime(), wire.nextEventTime());
        if (!next.has_value()) {
            // The endpoint is idle. If work is still waiting to be posted,
            // the next stimulus is the post itself, at this same timestamp.
            if (next_wr <= messages) {
                continue;
            }
            break;
        }
        now_ps = std::max(now_ps, *next);
    }
    device.validateInvariants();

    const simllm::rnic::RnicTxPipeline* pipeline = device.txPipeline();
    run.packets = pipeline->counters().packets_issued;
    run.packets_dropped = pipeline->counters().packets_dropped;
    run.payload_bytes = pipeline->counters().payload_bytes;
    run.wire_bytes = pipeline->counters().wire_bytes;
    run.late_releases = pipeline->counters().late_releases;
    for (const simllm::rnic::WqeRecord& record : device.records()) {
        if (record.timeline.first_packet_at_ps.has_value()) {
            if (run.first_packet_ps == 0) {
                run.first_packet_ps = *record.timeline.first_packet_at_ps;
            }
            run.last_packet_ps = *record.timeline.last_packet_at_ps;
        }
    }
    return run;
}

void testTxPipeline(TestRunner& test) {
    const RnicHwProfile& profile = kConnectX5_100G;

    RnicDeviceConfig defaults;
    test.check(
        defaults.network.abi_version
                == simllm::rnic::kNetworkPortAbiVersionV1
            && !defaults.network.packetization.enabled,
        "the default network configuration is the unchanged v1 path");

    const PipelineRun one = runPipeline(profile, 1048576, 1, 4096, 1, false);
    test.check(
        one.completions == 1 && one.errors == 0 && one.packets == 256
            && one.packets_dropped == 0 && one.payload_bytes == 1048576
            && one.wire_bytes == 1048576 + 256 * 64,
        "one 1 MiB message segments into 256 MTU packets with header bytes");
    test.check(
        one.late_releases == 0,
        "an event-stepping caller never forces a late packet release");
    test.check(
        one.first_packet_ps != 0 && one.last_packet_ps > one.first_packet_ps,
        "first and last packet issue come from real TX-start events");

    // Depth-1 closed form: the lumped fixed offset plus the paced
    // serialization of the message. The wire serializes the last packet at the
    // link rate rather than the pacer rate, which is the only residual.
    const double serialization = static_cast<double>(one.wire_bytes) * 8.0
        / static_cast<double>(simllm::rnic::effectiveWireBps(profile));
    const double expected = static_cast<double>(profile.t_eff_ps) / 1e12
        + serialization;
    const double measured = static_cast<double>(one.last_completion_ps) / 1e12;
    test.check(
        measured > expected * 0.99 && measured < expected * 1.01,
        "a depth-1 message costs the lumped offset plus its paced wire time");

    const PipelineRun ragged = runPipeline(profile, 5000, 1, 4096, 1, false);
    test.check(
        ragged.packets == 2 && ragged.payload_bytes == 5000
            && ragged.wire_bytes == 5000 + 2 * 64,
        "a ragged message segments into a full packet and a short one");

    const PipelineRun empty = runPipeline(profile, 0, 1, 4096, 1, false);
    test.check(
        empty.packets == 1 && empty.payload_bytes == 0
            && empty.wire_bytes == 64,
        "a zero-byte message is still one packet of header bytes");

    const PipelineRun dropped = runPipeline(profile, 8192, 1, 4096, 1, true);
    test.check(
        dropped.errors == 1 && dropped.completions == 0
            && dropped.packets_dropped == 1,
        "a dropped packet retires its WQE with a transport error");

    // Depth raises throughput because the window, not the fixed offset, is
    // what a deep queue removes. Small messages then run into the pacer.
    const PipelineRun shallow = runPipeline(profile, 8192, 1, 4096, 64, false);
    const PipelineRun deep = runPipeline(profile, 8192, 64, 4096, 64, false);
    test.check(
        deep.last_completion_ps < shallow.last_completion_ps,
        "a deeper outstanding-work window finishes the same bytes sooner");
    const double deep_bps = 64.0 * 8192.0 * 8.0
        / (static_cast<double>(deep.last_completion_ps) / 1e12);
    test.check(
        deep_bps < static_cast<double>(profile.goodput_bps) * 1.001,
        "a saturated pipeline never exceeds the profile goodput ceiling");

    const PipelineRun paced = runPipeline(profile, 1024, 64, 4096, 512, false);
    const double packet_rate = 512.0
        / (static_cast<double>(paced.last_completion_ps) / 1e12);
    test.check(
        packet_rate < static_cast<double>(profile.tx_pps_per_qp) * 1.02,
        "the per-QP message-rate ceiling caps small messages");

    // Configuration must fail closed rather than resolve a contradiction.
    const auto expectRejected = [&test](
                                    RnicDeviceConfig config,
                                    NetworkPort* port,
                                    const std::string& message) {
        RnicDeviceAttachments attachments;
        attachments.network_port = port;
        test.expectThrowAs<std::invalid_argument>(
            [&config, &attachments]() {
                RnicDevice device(config, attachments);
            },
            message);
    };
    simllm::rnic::testing::FakeV2NetworkPort wire(
        simllm::rnic::testing::FakeV2NetworkConfig{});
    RnicDeviceConfig mismatched;
    mismatched.network.enabled = true;
    mismatched.network.abi_version = simllm::rnic::kNetworkPortAbiVersionV2;
    expectRejected(mismatched, &wire, "ABI v2 without a packetizer is refused");
    RnicDeviceConfig orphan;
    orphan.network.enabled = true;
    orphan.network.packetization.enabled = true;
    expectRejected(
        orphan, &wire, "a packetizer without ABI v2 is refused");
    RnicDeviceConfig detached;
    detached.network.enabled = false;
    detached.network.abi_version = simllm::rnic::kNetworkPortAbiVersionV2;
    detached.network.packetization.enabled = true;
    expectRejected(
        detached, nullptr, "a packetizer without an external port is refused");

    simllm::rnic::testing::FakeNetworkPort flow_extent_port(4, 0);
    RnicDeviceConfig v1_only;
    v1_only.network.enabled = true;
    v1_only.network.abi_version = simllm::rnic::kNetworkPortAbiVersionV2;
    v1_only.network.packetization.enabled = true;
    expectRejected(
        v1_only,
        &flow_extent_port,
        "a packetizer over a flow-extent port is refused");
}

void testFacade(TestRunner& test, const std::string& scratch_dir) {
    rnic_cm_profile profile;
    test.check(
        rnic_cm_profile_preset("cx5_100g", &profile) == RNIC_CM_OK,
        "the facade exposes the measured preset");
    test.check(
        rnic_cm_profile_preset("nope", &profile) == RNIC_CM_ERROR_ARGUMENT
            && rnic_cm_profile_preset("cx5_100g", &profile) == RNIC_CM_OK,
        "an unknown preset name is refused");

    char digest[65];
    test.check(
        rnic_cm_profile_sha256(&profile, digest, sizeof(digest)) == RNIC_CM_OK
            && std::string(digest)
                == simllm::rnic::rnicHwProfileSha256(kConnectX5_100G),
        "the facade digest equals the native profile digest");
    test.check(
        rnic_cm_profile_sha256(&profile, digest, 8) == RNIC_CM_ERROR_ARGUMENT,
        "a short digest buffer is refused");

    rnic_cm_profile derived;
    test.check(
        rnic_cm_profile_preset("cx7_400g", &derived) == RNIC_CM_OK
            && rnic_cm_profile_sha256(&derived, digest, sizeof(digest))
                == RNIC_CM_OK
            && std::string(digest)
                == simllm::rnic::rnicHwProfileSha256(kConnectX7_400G),
        "the derived preset round-trips through the facade with its identity");

    const std::vector<CompletionRow> oracle = runOracle(kConnectX5_100G);
    const std::vector<CompletionRow> facade =
        runFacade(profile, false, std::string(), test);
    test.check(
        oracle.size() == kWqeCount && facade.size() == oracle.size(),
        "both drivers complete every work request");
    bool identical = oracle.size() == facade.size();
    for (std::size_t index = 0; identical && index < oracle.size(); ++index) {
        identical = sameRow(oracle[index], facade[index]);
    }
    test.check(
        identical,
        "the facade reproduces the C++ device timestamps exactly");

    const std::string prefix =
        scratch_dir.empty() ? std::string(".") : scratch_dir;
    const std::string trace_a = joinPath(prefix, "rnic_cmodel_trace_a.txt");
    const std::string trace_b = joinPath(prefix, "rnic_cmodel_trace_b.txt");
    const std::vector<CompletionRow> replay_a =
        runFacade(profile, true, trace_a, test);
    const std::vector<CompletionRow> replay_b =
        runFacade(profile, true, trace_b, test);
    bool replay_identical = replay_a.size() == replay_b.size();
    for (std::size_t index = 0; replay_identical && index < replay_a.size();
         ++index) {
        replay_identical = sameRow(replay_a[index], replay_b[index]);
    }
    test.check(replay_identical, "replayed completions are identical");
    const std::string bytes_a = readFile(trace_a);
    const std::string bytes_b = readFile(trace_b);
    test.check(
        !bytes_a.empty() && bytes_a == bytes_b,
        "two identical stimulus sequences produce byte-identical traces");
    test.check(
        bytes_a.find("create profile=cx5_100g") != std::string::npos
            && bytes_a.find(" packet ") != std::string::npos
            && bytes_a.find(" cqe ") != std::string::npos,
        "the trace records stimulus and observed transitions");

    rnic_cm_config config;
    std::memset(&config, 0, sizeof(config));
    config.version = SIMLLM_RNIC_CM_ABI_VERSION;
    config.qpn = kQpn;
    config.policy_context_token = kPolicyToken;
    config.sq_depth = 4;
    config.cq_depth = 8;
    config.version = SIMLLM_RNIC_CM_ABI_VERSION + 1;
    test.check(
        rnic_cm_create(&profile, &config) == nullptr,
        "an unsupported facade config version is refused");
    config.version = SIMLLM_RNIC_CM_ABI_VERSION;
    config.sq_depth = 0;
    test.check(
        rnic_cm_create(&profile, &config) == nullptr,
        "a zero send-queue depth is refused");

    config.sq_depth = 4;
    rnic_cm_device* device = rnic_cm_create(&profile, &config);
    test.check(device != nullptr, "the facade constructs with a valid config");
    if (device != nullptr) {
        rnic_cm_packet packet;
        std::memset(&packet, 0, sizeof(packet));
        test.check(
            rnic_cm_rx_packet(device, &packet, 0) == RNIC_CM_ERROR_UNSUPPORTED,
            "the receive entry point fails closed until its pipeline lands");
        rnic_cm_event_info event;
        std::memset(&event, 0, sizeof(event));
        event.kind = RNIC_CM_EVENT_EXTENT_DELIVERED;
        event.token = 99;
        test.check(
            rnic_cm_event(device, &event, 0) == RNIC_CM_ERROR_ARGUMENT,
            "an unknown token is refused");
        event.kind = RNIC_CM_EVENT_ECN_MARKED;
        test.check(
            rnic_cm_event(device, &event, 0) == RNIC_CM_ERROR_UNSUPPORTED,
            "a control event is refused until rate control lands");
        rnic_cm_counter_set counters;
        test.check(
            rnic_cm_counters(device, &counters) == RNIC_CM_OK
                && counters.posted_wqes == 0 && counters.tx_packets == 0,
            "counters read back zeroed on a fresh device");
        test.check(
            rnic_cm_trace(device, joinPath(prefix, "unused.txt").c_str())
                == RNIC_CM_ERROR_STATE,
            "a trace request without tracing enabled is refused");
        rnic_cm_destroy(device);
    }
    test.check(
        rnic_cm_post(nullptr, nullptr, 0, nullptr) == RNIC_CM_ERROR_ARGUMENT,
        "a null handle is refused rather than dereferenced");
    rnic_cm_destroy(nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string projection_path;
        std::string design_path;
        std::string scratch_dir;
        if (argc == 2
            && std::string(argv[1]) == "--render-anomaly-table") {
            std::cout << simllm::rnic::renderRnicAnomalyTableMarkdown();
            return 0;
        }
        if (argc >= 2) {
            projection_path = argv[1];
        }
        if (argc >= 3) {
            design_path = argv[2];
        }
        if (argc >= 4) {
            scratch_dir = argv[3];
        }
        if (argc > 4) {
            std::cerr
                << "usage: simllm_rnic_cmodel_test "
                   "[projection.md [design.md [scratch-dir]]]\n";
            return 2;
        }
        TestRunner test;
        testProfile(test);
        testAnomalyTable(test, projection_path, design_path);
        testTxPipeline(test);
        testFacade(test, scratch_dir);
        if (test.failures() != 0) {
            std::cerr << test.failures() << " golden-model checks failed\n";
            return 1;
        }
        std::cout << "RNIC golden-model checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unexpected golden-model failure: " << error.what()
                  << '\n';
        return 1;
    }
}
