#include "tier_a_port_factory.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "simllm/rnic/rnic_device.h"

namespace simllm::rnic::tier_a {
namespace {

class Json {
public:
    enum class Kind { Null, Boolean, Integer, String, Array, Object };
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;

    Json() = default;
    explicit Json(bool value) : kind_(Kind::Boolean), boolean_(value) {}
    explicit Json(std::uint64_t value)
        : kind_(Kind::Integer), integer_(value) {}
    explicit Json(std::string value)
        : kind_(Kind::String), string_(std::move(value)) {}
    explicit Json(const char* value) : Json(std::string(value)) {}
    explicit Json(Array value) : kind_(Kind::Array), array_(std::move(value)) {}
    explicit Json(Object value)
        : kind_(Kind::Object), object_(std::move(value)) {}

    std::string dump(unsigned indent = 0) const {
        std::ostringstream output;
        write(output, indent, 0);
        return output.str();
    }

private:
    static std::string escaped(const std::string& input) {
        std::ostringstream output;
        for (const unsigned char byte : input) {
            switch (byte) {
                case '\"':
                    output << "\\\"";
                    break;
                case '\\':
                    output << "\\\\";
                    break;
                case '\b':
                    output << "\\b";
                    break;
                case '\f':
                    output << "\\f";
                    break;
                case '\n':
                    output << "\\n";
                    break;
                case '\r':
                    output << "\\r";
                    break;
                case '\t':
                    output << "\\t";
                    break;
                default:
                    if (byte < 0x20U) {
                        static constexpr char digits[] = "0123456789abcdef";
                        output << "\\u00" << digits[(byte >> 4U) & 0xFU]
                               << digits[byte & 0xFU];
                    } else {
                        output << static_cast<char>(byte);
                    }
                    break;
            }
        }
        return output.str();
    }

    static void pad(std::ostream& output, unsigned count) {
        for (unsigned index = 0; index < count; ++index) {
            output.put(' ');
        }
    }

    void write(std::ostream& output, unsigned indent, unsigned depth) const {
        switch (kind_) {
            case Kind::Null:
                output << "null";
                return;
            case Kind::Boolean:
                output << (boolean_ ? "true" : "false");
                return;
            case Kind::Integer:
                output << integer_;
                return;
            case Kind::String:
                output << '\"' << escaped(string_) << '\"';
                return;
            case Kind::Array:
                writeArray(output, indent, depth);
                return;
            case Kind::Object:
                writeObject(output, indent, depth);
                return;
        }
        throw std::logic_error("unknown JSON kind");
    }

    void writeArray(
        std::ostream& output, unsigned indent, unsigned depth) const {
        output << '[';
        for (std::size_t index = 0; index < array_.size(); ++index) {
            if (index != 0) {
                output << ',';
            }
            if (indent != 0) {
                output << '\n';
                pad(output, (depth + 1) * indent);
            }
            array_[index].write(output, indent, depth + 1);
        }
        if (indent != 0 && !array_.empty()) {
            output << '\n';
            pad(output, depth * indent);
        }
        output << ']';
    }

    void writeObject(
        std::ostream& output, unsigned indent, unsigned depth) const {
        output << '{';
        std::size_t index = 0;
        for (const auto& item : object_) {
            if (index++ != 0) {
                output << ',';
            }
            if (indent != 0) {
                output << '\n';
                pad(output, (depth + 1) * indent);
            }
            output << '\"' << escaped(item.first) << '\"' << ':';
            if (indent != 0) {
                output << ' ';
            }
            item.second.write(output, indent, depth + 1);
        }
        if (indent != 0 && !object_.empty()) {
            output << '\n';
            pad(output, depth * indent);
        }
        output << '}';
    }

    Kind kind_{Kind::Null};
    bool boolean_{false};
    std::uint64_t integer_{0};
    std::string string_;
    Array array_;
    Object object_;
};

Json object(std::initializer_list<std::pair<const std::string, Json>> values) {
    Json::Object result;
    for (const auto& value : values) {
        result.emplace(value.first, value.second);
    }
    return Json(std::move(result));
}

Json array(Json::Array values) { return Json(std::move(values)); }

template <typename T>
T required(const std::optional<T>& value, const char* name) {
    if (!value.has_value()) {
        throw std::logic_error(std::string("missing native field: ") + name);
    }
    return *value;
}

const char* terminalKind(NetworkEventKind kind) {
    switch (kind) {
        case NetworkEventKind::Delivered:
            return "delivered";
        case NetworkEventKind::Dropped:
            return "dropped";
    }
    throw std::logic_error("unknown network terminal kind");
}

const char* completionStatus(CompletionStatus status) {
    switch (status) {
        case CompletionStatus::Success:
            return "success";
        case CompletionStatus::TransportError:
            return "transport_error";
        case CompletionStatus::NetworkRejected:
            return "network_rejected";
    }
    throw std::logic_error("unknown completion status");
}

const char* wqeState(WqeState state) {
    switch (state) {
        case WqeState::Posted:
            return "posted";
        case WqeState::Doorbelled:
            return "doorbelled";
        case WqeState::InFlight:
            return "in_flight";
        case WqeState::AwaitingOrderedRetirement:
            return "awaiting_ordered_retirement";
        case WqeState::RetiredUnsignaled:
            return "retired_unsignaled";
        case WqeState::CompletionPending:
            return "completion_pending";
        case WqeState::CqeVisible:
            return "cqe_visible";
        case WqeState::Reclaimed:
            return "reclaimed";
        case WqeState::Completed:
            return "completed";
        case WqeState::Error:
            return "error";
    }
    throw std::logic_error("unknown WQE state");
}

Picoseconds serviceTime(
    std::uint64_t payload_bytes, std::uint64_t link_rate_gbps) {
    if (link_rate_gbps == 0
        || payload_bytes
            > std::numeric_limits<std::uint64_t>::max() / 8000ULL) {
        throw std::invalid_argument("invalid Tier A serialization fixture");
    }
    const std::uint64_t numerator = payload_bytes * 8000ULL;
    if (numerator % link_rate_gbps != 0) {
        throw std::invalid_argument(
            "Tier A serialization time is not an integer");
    }
    return numerator / link_rate_gbps;
}

struct AuthorityState {
    std::uint64_t native_session_constructed{0};
    std::uint64_t native_posts{0};
    std::uint64_t legacy_ledger_constructed{0};
    std::uint64_t legacy_posts{0};
    std::uint64_t legacy_mutations{0};
};

Json authoritySnapshot(const AuthorityState& state) {
    return object({
        {"native_session_constructed",
         Json(state.native_session_constructed)},
        {"native_posts", Json(state.native_posts)},
        {"legacy_ledger_constructed",
         Json(state.legacy_ledger_constructed)},
        {"legacy_posts", Json(state.legacy_posts)},
        {"legacy_mutations", Json(state.legacy_mutations)},
    });
}

class CompositionAuthority {
public:
    CompositionAuthority(bool structural, bool bypass) {
        if (structural && bypass) {
            throw std::invalid_argument(
                "structural and bypass authorities are mutually exclusive");
        }
        if (!structural && !bypass) {
            throw std::invalid_argument("one composition authority is required");
        }
        structural_ = structural;
        if (structural_) {
            state_.native_session_constructed = 1;
        } else {
            state_.legacy_ledger_constructed = 1;
        }
    }

    void notePost() {
        if (structural_) {
            ++state_.native_posts;
        } else {
            ++state_.legacy_posts;
            ++state_.legacy_mutations;
        }
    }

    const AuthorityState& state() const noexcept { return state_; }

private:
    bool structural_{false};
    AuthorityState state_;
};

class ValidationLedger {
public:
    std::uint64_t noteConstructed() {
        const std::uint64_t session_id = ++constructed_sessions_;
        open_sessions_.insert(session_id);
        return session_id;
    }

    void noteFinalValidated(std::uint64_t session_id) {
        if (open_sessions_.erase(session_id) != 1
            || !validated_sessions_.insert(session_id).second) {
            throw std::logic_error(
                "Tier A session validation ledger is inconsistent");
        }
    }

    void requirePublicationReady() const {
        if (!open_sessions_.empty()
            || validated_sessions_.size() != constructed_sessions_) {
            throw std::logic_error(
                "not every Tier A session passed final invariant validation");
        }
    }

private:
    std::uint64_t constructed_sessions_{0};
    std::set<std::uint64_t> open_sessions_;
    std::set<std::uint64_t> validated_sessions_;
};

Json cellAuthority(const AuthorityState& state) {
    return object({
        {"mode", Json("structural")},
        {"native_session_constructed",
         Json(state.native_session_constructed)},
        {"native_posts", Json(state.native_posts)},
        {"legacy_ledger_constructed",
         Json(state.legacy_ledger_constructed)},
        {"legacy_posts", Json(state.legacy_posts)},
        {"legacy_mutations", Json(state.legacy_mutations)},
    });
}

class TerminalIngress {
public:
    TerminalIngress(RnicDevice& device, const DrivenPort& port)
        : device_(device), port_(port) {}

    void onNetworkEvent(const NetworkEvent& event) {
        refreshIssued();
        const auto owner = token_owner_.find(event.token);
        if (owner == token_owner_.end()) {
            throw std::invalid_argument("terminal token was never issued");
        }
        if (owner->second != event.wqe_id) {
            throw std::invalid_argument("terminal token does not belong to WQE");
        }
        if (consumed_.count(event.token) != 0) {
            throw std::invalid_argument("terminal token already consumed");
        }
        device_.onNetworkEvent(event);
        consumed_.insert(event.token);
        caller_time_ps_ = event.event_time_ps;
    }

    std::size_t progress(Picoseconds now_ps) {
        const std::size_t changes = device_.progress(now_ps);
        caller_time_ps_ = now_ps;
        return changes;
    }

    void refreshIssued() {
        for (const IssuedToken& issued : port_.issued()) {
            const auto result = token_owner_.emplace(issued.token, issued.wqe_id);
            if (!result.second && result.first->second != issued.wqe_id) {
                throw std::logic_error("network token changed WQE ownership");
            }
        }
    }

    const std::map<NetworkToken, WqeId>& owners() const noexcept {
        return token_owner_;
    }

    const std::set<NetworkToken>& consumed() const noexcept {
        return consumed_;
    }

    Picoseconds callerTime() const noexcept { return caller_time_ps_; }

private:
    RnicDevice& device_;
    const DrivenPort& port_;
    std::map<NetworkToken, WqeId> token_owner_;
    std::set<NetworkToken> consumed_;
    Picoseconds caller_time_ps_{0};
};

struct SessionOutput {
    std::vector<CompletionEntry> completions;
    Picoseconds jct_ps{0};
};

RnicDeviceConfig deviceConfig(
    Picoseconds doorbell_service_ps, std::size_t wqe_count) {
    RnicDeviceConfig config;
    config.identity.qpn = 17;
    config.identity.policy_context_token = 9001;
    config.work_queue.sq_id = 1;
    config.work_queue.cq_id = 1;
    config.work_queue.source = 3;
    config.work_queue.qpn = config.identity.qpn;
    config.work_queue.policy_context_token =
        config.identity.policy_context_token;
    config.work_queue.sq_depth = std::max<std::size_t>(8, wqe_count);
    config.work_queue.cq_depth = std::max<std::size_t>(8, wqe_count);
    config.work_queue.doorbell_service_ps = doorbell_service_ps;
    config.work_queue.wqe_fetch_service_ps = 0;
    config.work_queue.qpc_lookup_service_ps = 0;
    config.work_queue.scheduler_service_ps = 0;
    config.work_queue.cqe_write_service_ps = 0;
    config.qpc.enabled = false;
    config.dma.enabled = false;
    config.network.enabled = true;
    return config;
}

WorkRequest requestFor(
    std::size_t ordinal, std::uint64_t payload_bytes, bool signaled) {
    WorkRequest request;
    request.wr_id = ordinal + 1;
    request.flow_id = 1000 + ordinal + 1;
    request.flow_tag = 7;
    request.destination = 1;
    request.payload_bytes = payload_bytes;
    request.traffic_class = 3;
    request.signaled = signaled;
    return request;
}

std::optional<Picoseconds> nextTime(
    const RnicDevice& device, const DrivenPort& port) {
    const std::optional<Picoseconds> device_time = device.nextEventTime();
    const std::optional<Picoseconds> port_time = port.nextEventTime();
    if (!device_time.has_value()) {
        return port_time;
    }
    if (!port_time.has_value()) {
        return device_time;
    }
    return std::min(*device_time, *port_time);
}

SessionOutput driveToQuiescence(
    RnicDevice& device, DrivenPort& port, TerminalIngress& ingress) {
    SessionOutput output;
    Picoseconds now_ps = 0;
    std::size_t iterations = 0;
    while (device.hasPendingPhysicalWork()) {
        if (++iterations > 10000) {
            throw std::logic_error("Tier A event loop did not quiesce");
        }
        if (device.fatal()) {
            throw std::logic_error("Tier A device became fatal");
        }
        const std::optional<Picoseconds> next = nextTime(device, port);
        if (!next.has_value() || *next < now_ps) {
            throw std::logic_error("Tier A event loop lost causal order");
        }
        now_ps = *next;
        for (const NetworkEvent& event : port.takeDue(now_ps)) {
            ingress.onNetworkEvent(event);
        }
        ingress.progress(now_ps);
        ingress.refreshIssued();
        std::vector<CompletionEntry> completions =
            device.pollCompletionQueue(
                std::numeric_limits<std::size_t>::max(), now_ps);
        if (!completions.empty()) {
            output.jct_ps = now_ps;
            output.completions.insert(
                output.completions.end(),
                completions.begin(),
                completions.end());
        }
    }
    return output;
}

void validatePortLedger(
    const DrivenPort& port,
    const RnicDevice& device,
    const TerminalIngress& ingress) {
    std::set<NetworkToken> issued;
    std::set<NetworkToken> terminal;
    std::set<WqeId> issued_wqes;
    std::set<WqeId> terminal_wqes;
    for (const IssuedToken& row : port.issued()) {
        if (row.token == 0 || !issued.insert(row.token).second) {
            throw std::logic_error("Tier A issue ledger is not unique");
        }
        issued_wqes.insert(row.wqe_id);
    }
    for (const TerminalToken& row : port.terminals()) {
        if (!terminal.insert(row.token).second || issued.count(row.token) == 0) {
            throw std::logic_error("Tier A terminal ledger is not conserved");
        }
        terminal_wqes.insert(row.wqe_id);
    }
    if (!port.liveTokens().empty() || issued != terminal
        || issued != ingress.consumed()
        || port.issued().size() != device.records().size()
        || port.terminals().size() != device.records().size()
        || issued_wqes.size() != device.records().size()
        || terminal_wqes.size() != device.records().size()) {
        throw std::logic_error(
            "Tier A token conservation failed at quiescence");
    }
}

Json issuedJson(const std::vector<IssuedToken>& issued) {
    Json::Array rows;
    for (const IssuedToken& row : issued) {
        rows.push_back(object({
            {"token", Json(row.token)},
            {"wqe_id", Json(row.wqe_id)},
            {"accepted_at_ps", Json(row.accepted_at_ps)},
            {"port_tx_at_ps", Json(row.port_tx_at_ps)},
            {"payload_bytes", Json(row.payload_bytes)},
        }));
    }
    return array(std::move(rows));
}

Json terminalsJson(const std::vector<TerminalToken>& terminals) {
    Json::Array rows;
    for (const TerminalToken& row : terminals) {
        rows.push_back(object({
            {"token", Json(row.token)},
            {"wqe_id", Json(row.wqe_id)},
            {"kind", Json(terminalKind(row.kind))},
            {"at_ps", Json(row.at_ps)},
        }));
    }
    return array(std::move(rows));
}

Json liveTokensJson(const std::vector<NetworkToken>& tokens) {
    Json::Array rows;
    for (NetworkToken token : tokens) {
        rows.emplace_back(token);
    }
    return array(std::move(rows));
}

Json portJson(const DrivenPort& port) {
    return object({
        {"issued", issuedJson(port.issued())},
        {"terminals", terminalsJson(port.terminals())},
        {"live_tokens", liveTokensJson(port.liveTokens())},
    });
}

Json countersJson(const WorkQueueCounters& counters) {
    return object({
        {"posted_wqes", Json(counters.posted_wqes)},
        {"network_accepted", Json(counters.network_accepted)},
        {"network_delivered", Json(counters.network_delivered)},
        {"network_dropped", Json(counters.network_dropped)},
    });
}

Json deviceJson(const RnicDevice& device) {
    return object({
        {"counters", countersJson(device.counters())},
        {"has_pending_physical_work",
         Json(device.hasPendingPhysicalWork())},
        {"occupied_sq_entries",
         Json(static_cast<std::uint64_t>(device.occupiedSqEntries()))},
        {"completion_queue_depth",
         Json(static_cast<std::uint64_t>(device.completionQueueDepth()))},
        {"unpublished_wqes",
         Json(static_cast<std::uint64_t>(device.unpublishedWqeCount()))},
        {"fatal", Json(device.fatal())},
    });
}

const TerminalToken& terminalFor(
    const DrivenPort& port, WqeId wqe_id) {
    const auto match = std::find_if(
        port.terminals().begin(),
        port.terminals().end(),
        [wqe_id](const TerminalToken& item) {
            return item.wqe_id == wqe_id;
        });
    if (match == port.terminals().end()) {
        throw std::logic_error("missing terminal projection for WQE");
    }
    return *match;
}

const IssuedToken& issueFor(const DrivenPort& port, WqeId wqe_id) {
    const auto match = std::find_if(
        port.issued().begin(),
        port.issued().end(),
        [wqe_id](const IssuedToken& item) {
            return item.wqe_id == wqe_id;
        });
    if (match == port.issued().end()) {
        throw std::logic_error("missing issue projection for WQE");
    }
    return *match;
}

Json wqesJson(const RnicDevice& device, const DrivenPort& port) {
    Json::Array rows;
    const std::vector<WqeRecord>& records = device.records();
    for (std::size_t ordinal = 0; ordinal < records.size(); ++ordinal) {
        const WqeRecord& record = records[ordinal];
        const IssuedToken& issued = issueFor(port, record.wqe_id);
        const TerminalToken& terminal = terminalFor(port, record.wqe_id);
        rows.push_back(object({
            {"ordinal", Json(static_cast<std::uint64_t>(ordinal))},
            {"wqe_id", Json(record.wqe_id)},
            {"eligible_at_ps",
             Json(required(record.timeline.admitted_at_ps, "admitted_at_ps"))},
            {"network_accepted_at_ps",
             Json(required(
                 record.timeline.network_accepted_at_ps,
                 "network_accepted_at_ps"))},
            {"port_tx_at_ps", Json(issued.port_tx_at_ps)},
            {"terminal_kind", Json(terminalKind(terminal.kind))},
            {"terminal_at_ps",
             Json(required(
                 record.timeline.network_outcome_at_ps,
                 "network_outcome_at_ps"))},
            {"cqe_status",
             Json(completionStatus(required(
                 record.completion_status,
                 "completion_status")))},
            {"cqe_visible_at_ps",
             Json(required(
                 record.timeline.cqe_visible_at_ps,
                 "cqe_visible_at_ps"))},
            {"polled_at_ps",
             Json(required(record.timeline.polled_at_ps, "polled_at_ps"))},
        }));
    }
    return array(std::move(rows));
}

Json cqeOrderJson(const std::vector<CompletionEntry>& completions) {
    Json::Array values;
    for (const CompletionEntry& completion : completions) {
        values.emplace_back(completion.wqe_id);
    }
    return array(std::move(values));
}

Json completionStatusesJson(
    const std::vector<CompletionEntry>& completions) {
    Json::Array values;
    for (const CompletionEntry& completion : completions) {
        values.emplace_back(completionStatus(completion.status));
    }
    return array(std::move(values));
}

Json evidenceJson(const RnicDevice& device) {
    Json::Array values;
    for (const EvidenceEvent& evidence : device.evidence()) {
        if (evidence.kind != EvidenceKind::NetworkDrop
            || evidence.drop_location != DropLocation::Fabric
            || evidence.drop_reason != DropReason::Injected) {
            throw std::logic_error("unexpected Tier A evidence event");
        }
        values.push_back(object({
            {"kind", Json("network_drop")},
            {"drop_location", Json("fabric")},
            {"drop_reason", Json("injected")},
            {"wqe_id", Json(evidence.wqe_id)},
        }));
    }
    return array(std::move(values));
}

Json runCell(
    PortFactory& factory,
    ValidationLedger& validation_ledger,
    std::uint64_t payload_bytes,
    std::uint64_t link_rate_gbps,
    Picoseconds reported_doorbell_service_ps,
    Picoseconds actual_doorbell_service_ps,
    std::size_t wqe_count,
    std::size_t port_capacity,
    bool signaled,
    bool controlled_drop) {
    CompositionAuthority authority(true, false);
    const PortConfig port_config{
        port_capacity,
        link_rate_gbps,
        0,
        0,
        false,
        false,
        controlled_drop,
    };
    std::unique_ptr<DrivenPort> port = factory.create(port_config);
    RnicDeviceAttachments attachments;
    attachments.network_port = port.get();
    RnicDevice device(
        deviceConfig(actual_doorbell_service_ps, wqe_count), attachments);
    const std::uint64_t validation_session =
        validation_ledger.noteConstructed();

    for (std::size_t ordinal = 0; ordinal < wqe_count; ++ordinal) {
        const PostResult posted = device.postSend(
            requestFor(ordinal, payload_bytes, signaled), 0);
        if (posted.status != PostStatus::Accepted) {
            throw std::logic_error("Tier A native post was not accepted");
        }
        authority.notePost();
    }
    const DoorbellBatch doorbell = device.ringDoorbell(0);
    if (doorbell.wqe_count != wqe_count) {
        throw std::logic_error("Tier A doorbell did not publish every WQE");
    }

    TerminalIngress ingress(device, *port);
    const SessionOutput session = driveToQuiescence(device, *port, ingress);
    validatePortLedger(*port, device, ingress);
    device.validateInvariants();
    validation_ledger.noteFinalValidated(validation_session);

    Json::Object cell;
    cell.emplace("payload_bytes", Json(payload_bytes));
    cell.emplace("link_rate_gbps", Json(link_rate_gbps));
    cell.emplace(
        "doorbell_service_ps", Json(reported_doorbell_service_ps));
    cell.emplace("authority", cellAuthority(authority.state()));
    cell.emplace("device", deviceJson(device));
    cell.emplace("port", portJson(*port));
    cell.emplace("wqes", wqesJson(device, *port));
    cell.emplace("cqe_order", cqeOrderJson(session.completions));
    cell.emplace("jct_ps", Json(session.jct_ps));
    if (controlled_drop) {
        cell.emplace("signaled", Json(signaled));
        cell.emplace(
            "all_cqe_statuses",
            completionStatusesJson(session.completions));
        cell.emplace("evidence", evidenceJson(device));
    } else if (!device.evidence().empty()) {
        throw std::logic_error("success cell produced controlled evidence");
    }
    return Json(std::move(cell));
}

Json authorityCases() {
    CompositionAuthority structural(true, false);
    structural.notePost();
    CompositionAuthority bypass(false, true);
    bypass.notePost();

    const AuthorityState before;
    std::string exception_type;
    std::string exception_message;
    try {
        CompositionAuthority invalid(true, true);
        (void)invalid;
    } catch (const std::invalid_argument& error) {
        exception_type = "std::invalid_argument";
        exception_message = error.what();
    }
    const AuthorityState after;
    return object({
        {"structural", authoritySnapshot(structural.state())},
        {"bypass", authoritySnapshot(bypass.state())},
        {"dual_attempt",
         object({
             {"exception_type", Json(exception_type)},
             {"exception_message", Json(exception_message)},
             {"before", authoritySnapshot(before)},
             {"after", authoritySnapshot(after)},
         })},
    });
}

Json terminalRecordJson(const RnicDevice& device) {
    Json::Array values;
    for (const WqeRecord& record : device.records()) {
        values.push_back(object({
            {"wqe_id", Json(record.wqe_id)},
            {"state", Json(wqeState(record.state))},
            {"network_token",
             Json(required(record.network_token, "network_token"))},
            {"network_accepted_at_ps",
             Json(required(
                 record.timeline.network_accepted_at_ps,
                 "network_accepted_at_ps"))},
            {"network_outcome_at_ps",
             Json(required(
                 record.timeline.network_outcome_at_ps,
                 "network_outcome_at_ps"))},
            {"completion_status",
             Json(completionStatus(required(
                 record.completion_status,
                 "completion_status")))},
        }));
    }
    return array(std::move(values));
}

Json emptyEvidence(const RnicDevice& device) {
    if (!device.evidence().empty()) {
        throw std::logic_error("terminal control produced evidence");
    }
    return Json(Json::Array{});
}

Json terminalSnapshot(
    const RnicDevice& device,
    const DrivenPort& port,
    const TerminalIngress& ingress) {
    return object({
        {"caller_time_ps", Json(ingress.callerTime())},
        {"device_records", terminalRecordJson(device)},
        {"device_counters", countersJson(device.counters())},
        {"device_evidence", emptyEvidence(device)},
        {"port_issued", issuedJson(port.issued())},
        {"port_terminals", terminalsJson(port.terminals())},
        {"port_live_tokens", liveTokensJson(port.liveTokens())},
        {"occupied_sq_entries",
         Json(static_cast<std::uint64_t>(device.occupiedSqEntries()))},
        {"completion_queue_depth",
         Json(static_cast<std::uint64_t>(device.completionQueueDepth()))},
        {"unpublished_wqes",
         Json(static_cast<std::uint64_t>(device.unpublishedWqeCount()))},
        {"has_pending_physical_work",
         Json(device.hasPendingPhysicalWork())},
    });
}

NetworkEvent invalidTerminalEvent(
    const std::string& kind,
    const TerminalIngress& ingress,
    Picoseconds invalid_time_ps) {
    if (ingress.owners().size() != 2) {
        throw std::logic_error("terminal control did not issue two tokens");
    }
    const auto first = ingress.owners().begin();
    const auto second = std::next(first);
    NetworkEvent event;
    event.kind = NetworkEventKind::Delivered;
    event.event_time_ps = invalid_time_ps;
    if (kind == "duplicate") {
        event.token = first->first;
        event.wqe_id = first->second;
    } else if (kind == "unknown") {
        event.token = ingress.owners().rbegin()->first + 1000;
        event.wqe_id = first->second;
    } else if (kind == "cross_wqe") {
        event.token = first->first;
        event.wqe_id = second->second;
    } else {
        throw std::invalid_argument("unknown terminal control kind");
    }
    return event;
}

Json runTerminalControl(
    PortFactory& factory,
    ValidationLedger& validation_ledger,
    const std::string& kind) {
    constexpr std::uint64_t payload_bytes = 4096;
    constexpr std::uint64_t link_rate_gbps = 400;
    constexpr Picoseconds invalid_time_ps = 200000;
    constexpr Picoseconds clock_probe_time_ps = 150000;
    const Picoseconds service_ps = serviceTime(payload_bytes, link_rate_gbps);

    CompositionAuthority authority(true, false);
    std::unique_ptr<DrivenPort> port = factory.create(PortConfig{
        2,
        link_rate_gbps,
        0,
        0,
        false,
        false,
        false,
    });
    RnicDeviceAttachments attachments;
    attachments.network_port = port.get();
    RnicDevice device(deviceConfig(0, 2), attachments);
    const std::uint64_t validation_session =
        validation_ledger.noteConstructed();
    for (std::size_t ordinal = 0; ordinal < 2; ++ordinal) {
        const PostResult post =
            device.postSend(requestFor(ordinal, payload_bytes, true), 0);
        if (post.status != PostStatus::Accepted) {
            throw std::logic_error("terminal control post failed");
        }
        authority.notePost();
    }
    device.ringDoorbell(0);
    TerminalIngress ingress(device, *port);
    const SessionOutput session = driveToQuiescence(device, *port, ingress);
    if (session.jct_ps != service_ps || session.completions.size() != 2) {
        throw std::logic_error("terminal control fixture did not complete");
    }
    validatePortLedger(*port, device, ingress);
    const Json before = terminalSnapshot(device, *port, ingress);

    const NetworkEvent invalid =
        invalidTerminalEvent(kind, ingress, invalid_time_ps);
    std::string exception_type;
    std::string exception_message;
    try {
        ingress.onNetworkEvent(invalid);
    } catch (const std::invalid_argument& error) {
        exception_type = "std::invalid_argument";
        exception_message = error.what();
    }

    const Json after = terminalSnapshot(device, *port, ingress);

    std::string clock_probe_exception_type;
    std::uint64_t clock_probe_changes = 0;
    try {
        clock_probe_changes = ingress.progress(clock_probe_time_ps);
    } catch (const std::exception&) {
        clock_probe_exception_type = "std::exception";
    }
    if (before.dump() != after.dump()) {
        throw std::logic_error(
            "terminal wrapper rejection mutated the raw snapshot");
    }
    validatePortLedger(*port, device, ingress);
    device.validateInvariants();
    validation_ledger.noteFinalValidated(validation_session);

    return object({
        {"kind", Json(kind)},
        {"invalid_event_time_ps", Json(invalid_time_ps)},
        {"exception_type", Json(exception_type)},
        {"exception_message", Json(exception_message)},
        {"clock_probe_time_ps", Json(clock_probe_time_ps)},
        {"clock_probe_exception_type", Json(clock_probe_exception_type)},
        {"clock_probe_changes", Json(clock_probe_changes)},
        {"before", before},
        {"after", after},
    });
}

void verifyExpectationsFile(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument("cannot open expectations file: " + path);
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("cannot read expectations file: " + path);
    }
    const std::string text = contents.str();
    if (text.find("simllm-rnic-tier-a-expectations-v1")
            == std::string::npos
        || text.find("simllm-rnic-tier-a-observations-v1")
            == std::string::npos) {
        throw std::invalid_argument(
            "expectations file has the wrong Tier A schema");
    }
}

void publishAtomically(const std::string& path, const std::string& contents) {
    const std::filesystem::path destination(path);
    if (destination.empty() || destination.filename().empty()) {
        throw std::invalid_argument("observations path must name a file");
    }
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path());
    }
    if (std::filesystem::exists(destination)) {
        throw std::invalid_argument("observations file already exists: " + path);
    }
    const std::filesystem::path temporary = destination.string() + ".tmp";
    if (std::filesystem::exists(temporary)) {
        throw std::invalid_argument(
            "temporary observations file already exists: "
            + temporary.string());
    }
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot create temporary observations file: "
                + temporary.string());
        }
        output << contents;
        output.flush();
        if (!output) {
            throw std::runtime_error(
                "cannot write temporary observations file: "
                + temporary.string());
        }
        output.close();
        if (!output) {
            throw std::runtime_error(
                "cannot close temporary observations file: "
                + temporary.string());
        }
        std::filesystem::rename(temporary, destination);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

ProducerOptions parseArguments(int argc, char** argv) {
    ProducerOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--factory" || argument == "--expectations"
            || argument == "--observations") {
            if (index + 1 >= argc) {
                throw std::invalid_argument(argument + " requires a value");
            }
            const std::string value = argv[++index];
            if (argument == "--factory") {
                options.factory = value;
            } else if (argument == "--expectations") {
                options.expectations_path = value;
            } else {
                options.observations_path = value;
            }
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.factory.empty() || options.expectations_path.empty()
        || options.observations_path.empty()) {
        throw std::invalid_argument(
            "usage: simllm_rnic_tier_a --factory fake|htsim "
            "--expectations PATH --observations PATH");
    }
    return options;
}

}  // namespace

int runTierA(const ProducerOptions& options, PortFactory& factory) {
    if (options.factory != factory.name()) {
        throw std::invalid_argument("selected port factory does not match CLI");
    }
    verifyExpectationsFile(options.expectations_path);
    ValidationLedger validation_ledger;

    Json::Array single_rows;
    for (std::uint64_t payload_bytes : {4096ULL, 1048576ULL}) {
        for (std::uint64_t link_rate_gbps : {200ULL, 400ULL}) {
            for (Picoseconds doorbell_service_ps : {0ULL, 1000ULL}) {
                single_rows.push_back(runCell(
                    factory,
                    validation_ledger,
                    payload_bytes,
                    link_rate_gbps,
                    doorbell_service_ps,
                    doorbell_service_ps,
                    1,
                    1,
                    true,
                    false));
            }
        }
    }

    Json::Array fifo_rows;
    for (std::uint64_t link_rate_gbps : {200ULL, 400ULL}) {
        for (Picoseconds doorbell_service_ps : {0ULL, 1000ULL}) {
            fifo_rows.push_back(runCell(
                factory,
                validation_ledger,
                4096,
                link_rate_gbps,
                doorbell_service_ps,
                doorbell_service_ps,
                2,
                1,
                true,
                false));
        }
    }

    Json::Array bypass_rows;
    for (Picoseconds reported_doorbell_service_ps : {0ULL, 1000ULL}) {
        bypass_rows.push_back(runCell(
            factory,
            validation_ledger,
            4096,
            400,
            reported_doorbell_service_ps,
            0,
            1,
            1,
            true,
            false));
    }

    Json::Array terminal_controls;
    for (const std::string kind : {"duplicate", "unknown", "cross_wqe"}) {
        terminal_controls.push_back(
            runTerminalControl(factory, validation_ledger, kind));
    }

    Json observations = object({
        {"schema", Json("simllm-rnic-tier-a-observations-v1")},
        {"factory", Json(options.factory)},
        {"single_wqe", array(std::move(single_rows))},
        {"fifo", array(std::move(fifo_rows))},
        {"wrapper_bypass_control", array(std::move(bypass_rows))},
        {"authority_cases", authorityCases()},
        {"terminal_controls", array(std::move(terminal_controls))},
        {"controlled_drop",
         runCell(
             factory,
             validation_ledger,
             4096,
             400,
             0,
             0,
             1,
             1,
             false,
             true)},
    });

    const std::string serialized = observations.dump(2) + "\n";
    validation_ledger.requirePublicationReady();
    publishAtomically(options.observations_path, serialized);
    return 0;
}

}  // namespace simllm::rnic::tier_a

int main(int argc, char** argv) {
    try {
        const simllm::rnic::tier_a::ProducerOptions options =
            simllm::rnic::tier_a::parseArguments(argc, argv);
        std::unique_ptr<simllm::rnic::tier_a::PortFactory> factory =
            simllm::rnic::tier_a::makePortFactory(options.factory);
        return simllm::rnic::tier_a::runTierA(options, *factory);
    } catch (const std::exception& error) {
        std::cerr << "Tier A producer error: " << error.what() << '\n';
        return 2;
    }
}
