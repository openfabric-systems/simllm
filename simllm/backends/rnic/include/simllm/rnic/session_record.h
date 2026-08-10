#ifndef SIMLLM_RNIC_SESSION_RECORD_H
#define SIMLLM_RNIC_SESSION_RECORD_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "simllm/rnic/rnic_device.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicSessionConfigRecordVersion = 1;
inline constexpr std::uint32_t kRnicSessionResultRecordVersion = 1;
inline constexpr std::uint32_t kRnicBookkeepingProjectionVersion = 1;
inline constexpr const char* kRnicEffectiveHardwareSchema =
    "simllm-rnic-effective-hardware-v1";
inline constexpr const char* kRnicEffectiveHardwareHostMemorySchema =
    "simllm-rnic-effective-hardware-v2";
inline constexpr const char* kRnicSessionConfigSchema =
    "simllm-rnic-session-config-v1";
inline constexpr const char* kRnicSessionResultSchema =
    "simllm-rnic-session-result-v1";
inline constexpr const char* kRnicBookkeepingProjectionSchema =
    "simllm-rnic-bookkeeping-v1";

// This schema is the public structural WQE projection. It is separate from
// the older request-bookkeeping-v1 compatibility ledger, whose WQE shape
// requires a receive-queue parent even for a local send. The projection is
// immutable and CORE-4/5 later join request and execution scope without
// advancing the native WQE lifecycle.

enum class RnicHardwareMode : std::uint8_t {
    Structural,
    Bypass,
};

enum class RnicWqeAuthority : std::uint8_t {
    SimllmNativeRnicSession,
    AtlahsWqeLedger,
};

enum class RnicWqKind : std::uint8_t {
    Send,
    Receive,
    SharedReceive,
};

const char* toString(RnicHardwareMode mode) noexcept;
const char* toString(RnicWqeAuthority authority) noexcept;
const char* toString(RnicWqKind kind) noexcept;

struct RnicSessionConfigRecord {
    std::uint32_t version{kRnicSessionConfigRecordVersion};
    std::string session_id;
    RnicHardwareMode hardware_mode{RnicHardwareMode::Structural};
    RnicWqeAuthority authority{
        RnicWqeAuthority::SimllmNativeRnicSession};
    std::string transport_policy;
    std::optional<std::string> hardware_config_sha256;
    // This is the canonical schema-tagged JSON snapshot whose exact bytes are
    // hashed. It is absent in bypass mode and never contains session, policy,
    // workload or per-WQE identity.
    std::optional<std::string> effective_hardware_json;
};

// Canonical hardware identity is derived from a successfully constructed
// device. This captures resolved PCIe domains and the attached fabric config,
// while excluding transport-policy and run identity.
std::string renderEffectiveHardwareConfigJson(const RnicDevice& device);
std::string effectiveHardwareConfigSha256(const RnicDevice& device);

RnicSessionConfigRecord makeStructuralSessionConfigRecord(
    std::string session_id,
    std::string transport_policy,
    const RnicDevice& device);
RnicSessionConfigRecord makeBypassSessionConfigRecord(
    std::string session_id,
    std::string transport_policy);
void validateRnicSessionConfigRecord(const RnicSessionConfigRecord& record);
std::string renderRnicSessionConfigJson(
    const RnicSessionConfigRecord& record);

struct RnicAuthoritySelection {
    bool native_session_enabled{false};
    bool legacy_ledger_enabled{false};
};

struct RnicAuthorityCounters {
    std::uint64_t native_session_constructed{0};
    std::uint64_t legacy_ledger_constructed{0};
    std::uint64_t native_posts{0};
    std::uint64_t legacy_mutations{0};
};

// This audit object counts calls made by the selected authority. It neither
// owns nor advances a WQE lifecycle. Construction validates exclusivity before
// any counter is exposed or incremented.
class RnicAuthorityAudit {
public:
    RnicAuthorityAudit(
        RnicHardwareMode mode,
        RnicAuthoritySelection selection);

    void noteNativePost(std::uint64_t count = 1);
    void noteLegacyMutation(std::uint64_t count = 1);

    RnicHardwareMode hardwareMode() const noexcept;
    RnicWqeAuthority authority() const noexcept;
    const RnicAuthorityCounters& counters() const noexcept;

private:
    RnicHardwareMode hardware_mode_;
    RnicWqeAuthority authority_;
    RnicAuthorityCounters counters_;
};

struct RnicWqeProjectionKey {
    std::string session_id;
    std::uint32_t endpoint{0};
    RnicWqKind wq_kind{RnicWqKind::Send};
    std::uint64_t wq_id{0};
    std::uint64_t post_sequence{0};
};

struct RnicTransportProjection {
    WqeId wqe_id{0};
    std::string transport_kind{"none"};
    std::uint64_t transport_object_id{0};
};

struct RnicWqeProjectionRecord {
    RnicWqeProjectionKey key;
    WqeId wqe_id{0};
    std::optional<std::uint64_t> wr_id;
    FlowId flow_id{0};
    std::uint32_t flow_tag{0};
    std::uint32_t source{0};
    std::uint32_t destination{0};
    std::uint64_t payload_bytes{0};
    std::optional<std::uint32_t> qpn;
    bool signaled{true};
    WqeOpcode opcode{WqeOpcode::Send};
    WqeTimeline timeline;
    WqeState state{WqeState::Posted};
    std::optional<CompletionStatus> completion_status;
    std::uint64_t cq_id{0};
    std::optional<std::uint64_t> cqe_sequence;
    std::optional<std::uint64_t> cq_producer_index;
    std::optional<std::uint64_t> cq_consume_sequence;
    std::string transport_kind{"none"};
    std::uint64_t transport_object_id{0};
};

struct RnicCompletionCsvRow {
    std::string profile;
    FlowId flow_id{0};
    std::uint32_t source{0};
    std::uint32_t destination{0};
    std::uint32_t tag{0};
    std::uint64_t payload_bytes{0};
    Picoseconds start_time_ps{0};
    Picoseconds completion_time_ps{0};
    Picoseconds fct_ps{0};
    std::optional<WqeId> wqe_id;
    std::optional<std::uint64_t> sq_id;
    std::optional<std::uint64_t> rq_id;
    std::optional<std::uint64_t> cq_id;
    std::optional<std::uint64_t> sq_post_sequence;
    std::optional<std::uint64_t> sq_dispatch_sequence;
    std::optional<std::uint64_t> cq_post_sequence;
    std::optional<std::uint64_t> cq_consume_sequence;
    std::optional<std::string> transport_kind;
    std::optional<std::uint64_t> transport_object_id;
};

struct RnicSessionResultRecord {
    std::uint32_t version{kRnicSessionResultRecordVersion};
    std::string session_id;
    RnicHardwareMode hardware_mode{RnicHardwareMode::Structural};
    RnicWqeAuthority authority{
        RnicWqeAuthority::SimllmNativeRnicSession};
    std::string transport_policy;
    std::optional<std::string> hardware_config_sha256;
    RnicAuthorityCounters authority_counters;
    bool quiescent{false};
    std::vector<RnicWqeProjectionRecord> wqes;
    std::vector<RnicCompletionCsvRow> completion_rows;
};

RnicSessionResultRecord projectStructuralSessionResult(
    const RnicSessionConfigRecord& config_record,
    const RnicDevice& device,
    const std::vector<CompletionEntry>& polled_completions,
    const RnicAuthorityAudit& authority_audit,
    const std::vector<RnicTransportProjection>& transport = {});

RnicSessionResultRecord makeBypassSessionResultRecord(
    const RnicSessionConfigRecord& config_record,
    const RnicAuthorityAudit& authority_audit,
    std::vector<RnicWqeProjectionRecord> wqes,
    std::vector<RnicCompletionCsvRow> completion_rows,
    bool quiescent);

void validateRnicSessionResultRecord(
    const RnicSessionResultRecord& record);
std::string renderRnicSessionResultJson(
    const RnicSessionResultRecord& record);
std::string renderRnicBookkeepingProjectionJson(
    const RnicSessionResultRecord& record);
std::string renderRnicCompletionCsv(
    const RnicSessionResultRecord& record);

// Exposed for independent hash-oracle tests and schema readers. Production
// configuration hashes use effectiveHardwareConfigSha256 above.
std::string rnicSha256Hex(std::string_view bytes);

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_SESSION_RECORD_H
