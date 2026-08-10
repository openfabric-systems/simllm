#include "simllm/rnic/session_record.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace simllm::rnic {
namespace {

using JsonFields = std::vector<std::pair<std::string, std::string>>;

std::string jsonString(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
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
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0')
                       << static_cast<unsigned int>(character)
                       << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}

std::string jsonObject(JsonFields fields) {
    std::sort(
        fields.begin(),
        fields.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });
    for (std::size_t index = 1; index < fields.size(); ++index) {
        if (fields[index - 1].first == fields[index].first) {
            throw std::logic_error("duplicate RNIC JSON field");
        }
    }
    std::ostringstream output;
    output << '{';
    for (std::size_t index = 0; index < fields.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << jsonString(fields[index].first) << ':'
               << fields[index].second;
    }
    output << '}';
    return output.str();
}

std::string jsonArray(const std::vector<std::string>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
    return output.str();
}

template <typename Integer>
std::string jsonInteger(Integer value) {
    return std::to_string(value);
}

std::string jsonBoolean(bool value) {
    return value ? "true" : "false";
}

template <typename Integer>
std::string jsonOptionalInteger(const std::optional<Integer>& value) {
    return value.has_value() ? jsonInteger(*value) : "null";
}

std::string jsonOptionalString(const std::optional<std::string>& value) {
    return value.has_value() ? jsonString(*value) : "null";
}

void requireNonblank(const std::string& value, const char* field) {
    if (value.empty()
        || std::all_of(
            value.begin(),
            value.end(),
            [](unsigned char character) {
                return std::isspace(character) != 0;
            })) {
        throw std::invalid_argument(
            std::string("RNIC session ") + field + " must be nonblank");
    }
}

bool isSha256(const std::string& value) {
    return value.size() == 64
        && std::all_of(
            value.begin(),
            value.end(),
            [](unsigned char character) {
                return std::isdigit(character) != 0
                    || (character >= 'a' && character <= 'f');
            });
}

struct EffectiveJsonValue {
    enum class Kind : std::uint8_t {
        Boolean,
        Integer,
        String,
        Array,
        Object,
    };

    Kind kind{Kind::Object};
    bool boolean{false};
    std::uint64_t integer{0};
    std::string string;
    std::vector<EffectiveJsonValue> array;
    std::map<std::string, EffectiveJsonValue> object;
};

class EffectiveJsonParser {
public:
    explicit EffectiveJsonParser(std::string_view input) : input_(input) {
        constexpr std::size_t kMaximumEffectiveHardwareBytes = 1U << 20U;
        if (input_.empty()
            || input_.size() > kMaximumEffectiveHardwareBytes) {
            fail("has an invalid byte length");
        }
    }

    EffectiveJsonValue parse() {
        EffectiveJsonValue result = parseValue(0);
        if (position_ != input_.size()) {
            fail("has trailing bytes");
        }
        return result;
    }

private:
    [[noreturn]] void fail(const char* detail) const {
        throw std::invalid_argument(
            std::string("RNIC effective hardware JSON ") + detail);
    }

    bool consume(char expected) {
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void require(char expected) {
        if (!consume(expected)) {
            fail("is not canonical JSON");
        }
    }

    std::string parseString() {
        require('"');
        const std::size_t begin = position_;
        while (position_ < input_.size()) {
            const unsigned char character =
                static_cast<unsigned char>(input_[position_]);
            if (character == '"') {
                const std::string result(
                    input_.substr(begin, position_ - begin));
                ++position_;
                return result;
            }
            // Every string in this schema is an ASCII enum or field name.
            // Rejecting escapes and non-ASCII bytes makes the accepted bytes
            // exactly the bytes produced by jsonString above.
            if (character < 0x20U || character > 0x7eU
                || character == '\\') {
                fail("contains a noncanonical string");
            }
            ++position_;
        }
        fail("has an unterminated string");
    }

    std::uint64_t parseInteger() {
        if (position_ == input_.size()
            || input_[position_] < '0' || input_[position_] > '9') {
            fail("requires an unsigned integer");
        }
        if (input_[position_] == '0'
            && position_ + 1 < input_.size()
            && input_[position_ + 1] >= '0'
            && input_[position_ + 1] <= '9') {
            fail("contains a leading-zero integer");
        }
        std::uint64_t value = 0;
        do {
            const std::uint64_t digit = static_cast<std::uint64_t>(
                input_[position_] - '0');
            if (value
                > (std::numeric_limits<std::uint64_t>::max() - digit)
                    / 10U) {
                fail("contains an integer outside uint64");
            }
            value = value * 10U + digit;
            ++position_;
        } while (position_ < input_.size()
                 && input_[position_] >= '0'
                 && input_[position_] <= '9');
        return value;
    }

    EffectiveJsonValue parseArray(std::size_t depth) {
        EffectiveJsonValue result;
        result.kind = EffectiveJsonValue::Kind::Array;
        require('[');
        if (consume(']')) {
            return result;
        }
        while (true) {
            result.array.push_back(parseValue(depth));
            if (consume(']')) {
                return result;
            }
            require(',');
        }
    }

    EffectiveJsonValue parseObject(std::size_t depth) {
        EffectiveJsonValue result;
        result.kind = EffectiveJsonValue::Kind::Object;
        require('{');
        if (consume('}')) {
            return result;
        }
        std::optional<std::string> previous;
        while (true) {
            std::string name = parseString();
            if (previous.has_value() && name <= *previous) {
                fail("object fields are not unique and sorted");
            }
            previous = name;
            require(':');
            result.object.emplace(std::move(name), parseValue(depth));
            if (consume('}')) {
                return result;
            }
            require(',');
        }
    }

    EffectiveJsonValue parseValue(std::size_t depth) {
        constexpr std::size_t kMaximumDepth = 32;
        if (depth >= kMaximumDepth || position_ == input_.size()) {
            fail("is truncated or too deeply nested");
        }
        if (input_[position_] == '{') {
            return parseObject(depth + 1U);
        }
        if (input_[position_] == '[') {
            return parseArray(depth + 1U);
        }
        if (input_[position_] == '"') {
            EffectiveJsonValue result;
            result.kind = EffectiveJsonValue::Kind::String;
            result.string = parseString();
            return result;
        }
        if (input_.substr(position_, 4) == "true") {
            position_ += 4;
            EffectiveJsonValue result;
            result.kind = EffectiveJsonValue::Kind::Boolean;
            result.boolean = true;
            return result;
        }
        if (input_.substr(position_, 5) == "false") {
            position_ += 5;
            EffectiveJsonValue result;
            result.kind = EffectiveJsonValue::Kind::Boolean;
            return result;
        }
        EffectiveJsonValue result;
        result.kind = EffectiveJsonValue::Kind::Integer;
        result.integer = parseInteger();
        return result;
    }

    std::string_view input_;
    std::size_t position_{0};
};

using EffectiveJsonObject =
    std::map<std::string, EffectiveJsonValue>;

[[noreturn]] void invalidEffectiveHardware(const std::string& detail) {
    throw std::invalid_argument(
        "RNIC effective hardware " + detail);
}

const EffectiveJsonObject& effectiveObject(
    const EffectiveJsonValue& value,
    const char* field) {
    if (value.kind != EffectiveJsonValue::Kind::Object) {
        invalidEffectiveHardware(std::string(field) + " must be an object");
    }
    return value.object;
}

const std::vector<EffectiveJsonValue>& effectiveArray(
    const EffectiveJsonValue& value,
    const char* field) {
    if (value.kind != EffectiveJsonValue::Kind::Array) {
        invalidEffectiveHardware(std::string(field) + " must be an array");
    }
    return value.array;
}

bool effectiveBoolean(
    const EffectiveJsonValue& value,
    const char* field) {
    if (value.kind != EffectiveJsonValue::Kind::Boolean) {
        invalidEffectiveHardware(std::string(field) + " must be boolean");
    }
    return value.boolean;
}

std::uint64_t effectiveInteger(
    const EffectiveJsonValue& value,
    const char* field) {
    if (value.kind != EffectiveJsonValue::Kind::Integer) {
        invalidEffectiveHardware(
            std::string(field) + " must be an unsigned integer");
    }
    return value.integer;
}

const std::string& effectiveString(
    const EffectiveJsonValue& value,
    const char* field) {
    if (value.kind != EffectiveJsonValue::Kind::String) {
        invalidEffectiveHardware(std::string(field) + " must be a string");
    }
    return value.string;
}

const EffectiveJsonValue& effectiveField(
    const EffectiveJsonObject& object,
    const char* field) {
    const auto item = object.find(field);
    if (item == object.end()) {
        invalidEffectiveHardware(std::string("is missing field ") + field);
    }
    return item->second;
}

void requireEffectiveFields(
    const EffectiveJsonObject& object,
    std::initializer_list<const char*> fields) {
    if (object.size() != fields.size()) {
        invalidEffectiveHardware("has an unexpected field set");
    }
    for (const char* field : fields) {
        static_cast<void>(effectiveField(object, field));
    }
}

void requirePositive(std::uint64_t value, const char* field) {
    if (value == 0) {
        invalidEffectiveHardware(std::string(field) + " must be positive");
    }
}

void requireTimestamp(std::uint64_t value, const char* field) {
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())) {
        invalidEffectiveHardware(
            std::string(field) + " exceeds the signed timestamp domain");
    }
}

bool isPowerOfTwo(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1U)) == 0;
}

void validateAnalyticalProfile(
    const EffectiveJsonValue& value) {
    const EffectiveJsonObject& profile =
        effectiveObject(value, "analytical profile");
    const std::string& kind = effectiveString(
        effectiveField(profile, "kind"), "analytical kind");
    if (kind == "disabled") {
        requireEffectiveFields(profile, {"kind"});
        return;
    }
    if (kind == "fixed") {
        requireEffectiveFields(
            profile, {"incidence_probability_ppm", "kind", "mean_ps"});
    } else if (kind == "gaussian") {
        requireEffectiveFields(
            profile,
            {"incidence_probability_ppm", "kind", "mean_ps",
             "standard_deviation_ps"});
    } else if (kind == "gaussian_tail_mixture") {
        requireEffectiveFields(
            profile,
            {"incidence_probability_ppm", "kind", "mean_ps",
             "standard_deviation_ps", "tail_mean_ps",
             "tail_probability_ppm", "tail_standard_deviation_ps"});
    } else {
        invalidEffectiveHardware("has an unknown analytical kind");
    }
    const std::uint64_t incidence = effectiveInteger(
        effectiveField(profile, "incidence_probability_ppm"),
        "incidence_probability_ppm");
    if (incidence == 0 || incidence > kPcieProbabilityScalePpm) {
        invalidEffectiveHardware(
            "active analytical incidence is outside one million ppm");
    }
    const std::uint64_t mean = effectiveInteger(
        effectiveField(profile, "mean_ps"), "mean_ps");
    requireTimestamp(mean, "mean_ps");
    if (kind == "fixed") {
        return;
    }
    const std::uint64_t deviation = effectiveInteger(
        effectiveField(profile, "standard_deviation_ps"),
        "standard_deviation_ps");
    requireTimestamp(deviation, "standard_deviation_ps");
    requirePositive(deviation, "standard_deviation_ps");
    if (kind == "gaussian") {
        return;
    }
    const std::uint64_t tail_probability = effectiveInteger(
        effectiveField(profile, "tail_probability_ppm"),
        "tail_probability_ppm");
    const std::uint64_t tail_mean = effectiveInteger(
        effectiveField(profile, "tail_mean_ps"), "tail_mean_ps");
    const std::uint64_t tail_deviation = effectiveInteger(
        effectiveField(profile, "tail_standard_deviation_ps"),
        "tail_standard_deviation_ps");
    if (tail_probability == 0
        || tail_probability >= kPcieProbabilityScalePpm
        || tail_mean <= mean || tail_deviation == 0) {
        invalidEffectiveHardware("has an invalid analytical tail");
    }
    requireTimestamp(tail_mean, "tail_mean_ps");
    requireTimestamp(tail_deviation, "tail_standard_deviation_ps");
}

void validatePenalties(const EffectiveJsonValue& value) {
    const EffectiveJsonObject& penalties =
        effectiveObject(value, "analytical penalties");
    requireEffectiveFields(
        penalties,
        {"acs", "ddio_miss", "gpu_direct", "iommu", "numa",
         "switch_path"});
    for (const char* field : {
             "acs", "ddio_miss", "gpu_direct", "iommu", "numa",
             "switch_path"}) {
        validateAnalyticalProfile(effectiveField(penalties, field));
    }
}

struct ValidatedEffectivePath {
    bool enabled{false};
    std::string endpoint;
};

ValidatedEffectivePath validatePath(const EffectiveJsonValue& value) {
    const EffectiveJsonObject& path = effectiveObject(value, "PCIe path");
    const bool enabled = effectiveBoolean(
        effectiveField(path, "enabled"), "path enabled");
    const std::uint64_t path_id = effectiveInteger(
        effectiveField(path, "path_id"), "path_id");
    requirePositive(path_id, "path_id");
    if (path_id > std::numeric_limits<std::uint32_t>::max()) {
        invalidEffectiveHardware("path_id exceeds uint32");
    }
    if (!enabled) {
        requireEffectiveFields(path, {"enabled", "path_id"});
        return {false, {}};
    }
    requireEffectiveFields(
        path,
        {"analytical_penalties", "base_latency_ps", "enabled", "endpoint",
         "path_id"});
    validatePenalties(effectiveField(path, "analytical_penalties"));
    requireTimestamp(
        effectiveInteger(
            effectiveField(path, "base_latency_ps"), "base_latency_ps"),
        "base_latency_ps");
    const std::string& endpoint = effectiveString(
        effectiveField(path, "endpoint"), "path endpoint");
    if (endpoint != "mmio_bar" && endpoint != "host_pinned_memory"
        && endpoint != "gpu_memory" && endpoint != "device_memory") {
        invalidEffectiveHardware("has an invalid PCIe endpoint");
    }
    return {true, endpoint};
}

void validateCredits(
    const EffectiveJsonValue& value,
    std::uint64_t data_credit_unit,
    std::uint64_t max_payload) {
    const EffectiveJsonObject& credits =
        effectiveObject(value, "PCIe credits");
    requireEffectiveFields(
        credits,
        {"completion_data_credits", "completion_header_credits",
         "nonposted_header_credits", "posted_data_credits",
         "posted_header_credits"});
    for (const char* field : {
             "completion_data_credits", "completion_header_credits",
             "nonposted_header_credits", "posted_data_credits",
             "posted_header_credits"}) {
        const std::uint64_t count = effectiveInteger(
            effectiveField(credits, field), field);
        requirePositive(count, field);
        if (count > std::numeric_limits<std::uint32_t>::max()) {
            invalidEffectiveHardware(
                std::string(field) + " exceeds uint32");
        }
        if ((std::string_view(field) == "completion_data_credits"
             || std::string_view(field) == "posted_data_credits")
            && count < (max_payload - 1U) / data_credit_unit + 1U) {
            invalidEffectiveHardware(
                std::string(field) + " cannot hold one MPS payload");
        }
    }
}

void validateLatencyProfile(
    const EffectiveJsonValue& value,
    const char* field) {
    const std::vector<EffectiveJsonValue>& samples =
        effectiveArray(value, field);
    if (samples.size() != 1) {
        invalidEffectiveHardware(
            std::string(field) + " requires one fixed sample");
    }
    requireTimestamp(effectiveInteger(samples.front(), field), field);
}

std::map<std::uint32_t, ValidatedEffectivePath> validateFabric(
    const EffectiveJsonValue& value) {
    const EffectiveJsonObject& fabric =
        effectiveObject(value, "PCIe fabric");
    requireEffectiveFields(
        fabric,
        {"analytical_seed", "completion_buffer_bytes",
         "completion_buffer_release_latency_ps", "completion_overhead_bytes",
         "credit_return_latency_ps", "data_credit_unit_bytes",
         "device_to_host_credits", "generation", "host_store_latency_ps",
         "host_to_device_credits", "lane_count",
         "max_outstanding_read_requests", "max_payload_size_bytes",
         "max_read_request_size_bytes", "max_tlps_per_transaction", "paths",
         "posted_write_overhead_bytes", "posted_write_visibility_latency_ps",
         "read_completion_boundary_bytes", "read_completion_latency_ps",
         "read_request_overhead_bytes"});
    static_cast<void>(effectiveInteger(
        effectiveField(fabric, "analytical_seed"), "analytical_seed"));
    const std::uint64_t generation = effectiveInteger(
        effectiveField(fabric, "generation"), "generation");
    if (generation < 1 || generation > 5) {
        invalidEffectiveHardware("generation is outside [1, 5]");
    }
    const std::uint64_t lanes = effectiveInteger(
        effectiveField(fabric, "lane_count"), "lane_count");
    if (!isPowerOfTwo(lanes) || lanes > 32) {
        invalidEffectiveHardware("lane_count must be a power of two up to 32");
    }
    const std::uint64_t max_payload = effectiveInteger(
        effectiveField(fabric, "max_payload_size_bytes"),
        "max_payload_size_bytes");
    const std::uint64_t max_read = effectiveInteger(
        effectiveField(fabric, "max_read_request_size_bytes"),
        "max_read_request_size_bytes");
    for (const auto& size : {
             std::pair<std::uint64_t, const char*>{
                 max_payload, "max_payload_size_bytes"},
             {max_read, "max_read_request_size_bytes"}}) {
        if (!isPowerOfTwo(size.first) || size.first < 128
            || size.first > 4096) {
            invalidEffectiveHardware(
                std::string(size.second)
                + " must be a power of two in [128, 4096]");
        }
    }
    const std::uint64_t boundary = effectiveInteger(
        effectiveField(fabric, "read_completion_boundary_bytes"),
        "read_completion_boundary_bytes");
    if (boundary != 64 && boundary != 128) {
        invalidEffectiveHardware(
            "read_completion_boundary_bytes must be 64 or 128");
    }
    for (const char* field : {
             "posted_write_overhead_bytes", "read_request_overhead_bytes",
             "completion_overhead_bytes", "max_outstanding_read_requests",
             "completion_buffer_bytes", "max_tlps_per_transaction"}) {
        requirePositive(
            effectiveInteger(effectiveField(fabric, field), field), field);
    }
    const std::uint64_t data_credit_unit = effectiveInteger(
        effectiveField(fabric, "data_credit_unit_bytes"),
        "data_credit_unit_bytes");
    if (!isPowerOfTwo(data_credit_unit)
        || data_credit_unit > std::numeric_limits<std::uint32_t>::max()) {
        invalidEffectiveHardware(
            "data_credit_unit_bytes must be a power of two");
    }
    for (const char* field : {
             "credit_return_latency_ps",
             "completion_buffer_release_latency_ps"}) {
        requireTimestamp(
            effectiveInteger(effectiveField(fabric, field), field), field);
    }
    validateLatencyProfile(
        effectiveField(fabric, "host_store_latency_ps"),
        "host_store_latency_ps");
    validateLatencyProfile(
        effectiveField(fabric, "posted_write_visibility_latency_ps"),
        "posted_write_visibility_latency_ps");
    validateLatencyProfile(
        effectiveField(fabric, "read_completion_latency_ps"),
        "read_completion_latency_ps");
    validateCredits(
        effectiveField(fabric, "host_to_device_credits"),
        data_credit_unit,
        max_payload);
    validateCredits(
        effectiveField(fabric, "device_to_host_credits"),
        data_credit_unit,
        max_payload);

    const std::vector<EffectiveJsonValue>& paths = effectiveArray(
        effectiveField(fabric, "paths"), "paths");
    if (paths.empty()) {
        invalidEffectiveHardware("paths must not be empty");
    }
    std::map<std::uint32_t, ValidatedEffectivePath> by_id;
    std::uint64_t previous_id = 0;
    for (const EffectiveJsonValue& path_value : paths) {
        const EffectiveJsonObject& path =
            effectiveObject(path_value, "PCIe path");
        const std::uint64_t path_id = effectiveInteger(
            effectiveField(path, "path_id"), "path_id");
        if (path_id <= previous_id) {
            invalidEffectiveHardware(
                "paths require unique ascending path IDs");
        }
        previous_id = path_id;
        by_id.emplace(
            static_cast<std::uint32_t>(path_id),
            validatePath(path_value));
    }
    return by_id;
}

void validateBinding(
    const EffectiveJsonValue& value,
    const std::map<std::uint32_t, ValidatedEffectivePath>& paths,
    std::string_view queue_endpoint) {
    const EffectiveJsonObject& binding =
        effectiveObject(value, "work-queue PCIe binding");
    requireEffectiveFields(
        binding,
        {"pcie_completion_ordering_domain", "pcie_cq_first_byte_offset",
         "pcie_cq_memory_path_id", "pcie_cqe_bytes",
         "pcie_doorbell_record_bytes",
         "pcie_doorbell_record_first_byte_offset",
         "pcie_doorbell_record_path_id", "pcie_sq_first_byte_offset",
         "pcie_sq_memory_path_id", "pcie_submission_ordering_domain",
         "pcie_uar_doorbell_bytes", "pcie_uar_first_byte_offset",
         "pcie_uar_path_id", "pcie_wqe_bytes"});
    for (const char* field : {
             "pcie_completion_ordering_domain", "pcie_cq_memory_path_id",
             "pcie_cqe_bytes", "pcie_doorbell_record_bytes",
             "pcie_doorbell_record_path_id", "pcie_sq_memory_path_id",
             "pcie_submission_ordering_domain", "pcie_uar_doorbell_bytes",
             "pcie_uar_path_id", "pcie_wqe_bytes"}) {
        requirePositive(
            effectiveInteger(effectiveField(binding, field), field), field);
    }
    for (const char* field : {
             "pcie_cq_first_byte_offset",
             "pcie_doorbell_record_first_byte_offset",
             "pcie_sq_first_byte_offset", "pcie_uar_first_byte_offset"}) {
        if (effectiveInteger(effectiveField(binding, field), field) >= 4096) {
            invalidEffectiveHardware(
                std::string(field) + " must be below 4096");
        }
    }
    for (const char* field : {
             "pcie_uar_path_id", "pcie_doorbell_record_path_id",
             "pcie_sq_memory_path_id", "pcie_cq_memory_path_id"}) {
        const std::string_view expected =
            std::string_view(field) == "pcie_uar_path_id"
            ? std::string_view("mmio_bar")
            : queue_endpoint;
        const std::uint64_t path_id = effectiveInteger(
            effectiveField(binding, field), field);
        if (path_id > std::numeric_limits<std::uint32_t>::max()) {
            invalidEffectiveHardware(
                std::string(field) + " exceeds uint32");
        }
        const auto path = paths.find(static_cast<std::uint32_t>(path_id));
        if (path == paths.end() || !path->second.enabled
            || std::string_view(path->second.endpoint) != expected) {
            invalidEffectiveHardware(
                std::string(field)
                + " references an incompatible path");
        }
    }
}

void validateModule(
    const EffectiveJsonValue& value,
    const char* field) {
    const EffectiveJsonObject& module = effectiveObject(value, field);
    requireEffectiveFields(module, {"enabled"});
    static_cast<void>(effectiveBoolean(
        effectiveField(module, "enabled"), field));
}

struct ValidatedSubmission {
    std::string producer_shape;
    std::string queue_endpoint;
    std::uint64_t descriptor_writer_id{0};
    std::uint64_t descriptor_queue_allocation_id{0};
};

ValidatedSubmission validateSubmission(const EffectiveJsonValue& value) {
    const EffectiveJsonObject& submission =
        effectiveObject(value, "submission");
    requireEffectiveFields(
        submission,
        {"cq_consumer_id", "cq_consumer_kind",
         "descriptor_queue_allocation_id", "descriptor_queue_endpoint",
         "descriptor_writer_id", "descriptor_writer_kind", "producer_id",
         "producer_kind", "producer_shape", "queue_endpoint",
         "rnic_requester_id", "uar_mapping_owner"});
    for (const char* field : {
             "producer_id", "cq_consumer_id", "rnic_requester_id"}) {
        const std::uint64_t identity = effectiveInteger(
            effectiveField(submission, field), field);
        if (identity == 0
            || identity > std::numeric_limits<std::uint32_t>::max()) {
            invalidEffectiveHardware(
                std::string(field) + " must be a nonzero uint32");
        }
    }

    ValidatedSubmission result;
    result.producer_shape = effectiveString(
        effectiveField(submission, "producer_shape"), "producer_shape");
    result.queue_endpoint = effectiveString(
        effectiveField(submission, "queue_endpoint"), "queue_endpoint");
    result.descriptor_writer_id = effectiveInteger(
        effectiveField(submission, "descriptor_writer_id"),
        "descriptor_writer_id");
    result.descriptor_queue_allocation_id = effectiveInteger(
        effectiveField(submission, "descriptor_queue_allocation_id"),
        "descriptor_queue_allocation_id");
    const std::string& producer_kind = effectiveString(
        effectiveField(submission, "producer_kind"), "producer_kind");
    const std::string& writer_kind = effectiveString(
        effectiveField(submission, "descriptor_writer_kind"),
        "descriptor_writer_kind");
    const std::string& descriptor_endpoint = effectiveString(
        effectiveField(submission, "descriptor_queue_endpoint"),
        "descriptor_queue_endpoint");
    const std::string& consumer_kind = effectiveString(
        effectiveField(submission, "cq_consumer_kind"),
        "cq_consumer_kind");
    const std::string& uar_owner = effectiveString(
        effectiveField(submission, "uar_mapping_owner"),
        "uar_mapping_owner");

    if (result.producer_shape == "host_cpu_driver") {
        if (producer_kind != "host_cpu_driver" || writer_kind != "none"
            || result.descriptor_writer_id != 0
            || result.descriptor_queue_allocation_id != 0
            || descriptor_endpoint != "none"
            || result.queue_endpoint != "host_pinned_memory"
            || consumer_kind != "host_cpu_driver"
            || uar_owner != "host_cpu") {
            invalidEffectiveHardware(
                "host CPU submission fields are inconsistent");
        }
    } else if (result.producer_shape == "cpu_proxy") {
        if (producer_kind != "cpu_proxy" || writer_kind != "gpu"
            || result.descriptor_writer_id == 0
            || result.descriptor_writer_id
                > std::numeric_limits<std::uint32_t>::max()
            || result.descriptor_queue_allocation_id == 0
            || descriptor_endpoint != "host_pinned_memory"
            || result.queue_endpoint != "host_pinned_memory"
            || consumer_kind != "cpu_proxy" || uar_owner != "host_cpu") {
            invalidEffectiveHardware(
                "CPU-proxy submission fields are inconsistent");
        }
    } else if (result.producer_shape == "gpu_initiated") {
        if (producer_kind != "gpu" || writer_kind != "none"
            || result.descriptor_writer_id != 0
            || result.descriptor_queue_allocation_id != 0
            || descriptor_endpoint != "none"
            || result.queue_endpoint != "gpu_memory"
            || consumer_kind != "gpu" || uar_owner != "gpu") {
            invalidEffectiveHardware(
                "GPU-initiated submission fields are inconsistent");
        }
    } else {
        invalidEffectiveHardware("has an unknown producer shape");
    }
    return result;
}

void validateHostMemory(
    const EffectiveJsonValue& value,
    const std::map<std::uint32_t, ValidatedEffectivePath>& paths,
    const ValidatedSubmission* submission) {
    const EffectiveJsonObject& memory =
        effectiveObject(value, "host memory");
    requireEffectiveFields(
        memory,
        {"allocations", "device_owner_id", "enabled", "registry",
         "work_queue"});
    if (!effectiveBoolean(
            effectiveField(memory, "enabled"), "host memory enabled")) {
        invalidEffectiveHardware("host memory schema must be enabled");
    }
    requirePositive(
        effectiveInteger(
            effectiveField(memory, "device_owner_id"), "device_owner_id"),
        "device_owner_id");

    const EffectiveJsonObject& registry = effectiveObject(
        effectiveField(memory, "registry"), "host-memory registry");
    requireEffectiveFields(
        registry,
        {"mpt_entry_bytes", "mpt_first_byte_offset", "mtt_entry_bytes",
         "mtt_first_byte_offset", "queue_page_list_entry_bytes",
         "queue_page_list_first_byte_offset", "translation_path_id"});
    for (const char* field : {
             "mpt_entry_bytes", "mtt_entry_bytes",
             "queue_page_list_entry_bytes", "translation_path_id"}) {
        requirePositive(
            effectiveInteger(effectiveField(registry, field), field), field);
    }
    for (const char* field : {
             "mpt_first_byte_offset", "mtt_first_byte_offset",
             "queue_page_list_first_byte_offset"}) {
        if (effectiveInteger(effectiveField(registry, field), field) >= 4096) {
            invalidEffectiveHardware(
                std::string(field) + " must be below 4096");
        }
    }
    const std::uint64_t translation_path_id = effectiveInteger(
        effectiveField(registry, "translation_path_id"),
        "translation_path_id");
    if (translation_path_id > std::numeric_limits<std::uint32_t>::max()) {
        invalidEffectiveHardware("translation_path_id exceeds uint32");
    }
    const auto translation_path = paths.find(
        static_cast<std::uint32_t>(translation_path_id));
    if (translation_path == paths.end() || !translation_path->second.enabled
        || translation_path->second.endpoint != "host_pinned_memory") {
        invalidEffectiveHardware(
            "host-memory translation path must be host-pinned");
    }

    const EffectiveJsonObject& binding = effectiveObject(
        effectiveField(memory, "work_queue"), "host-memory WQ binding");
    requireEffectiveFields(
        binding,
        {"cq_ring_allocation_id", "doorbell_record_allocation_id",
         "qpc_context_bytes", "qpc_icm_allocation_id",
         "rq_ring_allocation_id", "sq_ring_allocation_id"});
    for (const auto& item : binding) {
        requirePositive(
            effectiveInteger(item.second, item.first.c_str()),
            item.first.c_str());
    }

    const std::vector<EffectiveJsonValue>& allocations = effectiveArray(
        effectiveField(memory, "allocations"), "host-memory allocations");
    if (allocations.empty()) {
        invalidEffectiveHardware(
            "host-memory allocations must not be empty");
    }
    std::map<std::uint64_t, std::string> object_by_id;
    std::map<std::uint64_t, std::uint64_t> owner_by_id;
    std::map<std::uint64_t, std::string> endpoint_by_id;
    std::size_t descriptor_queue_count = 0;
    std::uint64_t previous_id = 0;
    for (const EffectiveJsonValue& value : allocations) {
        const EffectiveJsonObject& allocation = effectiveObject(
            value, "host-memory allocation");
        const std::string& object_kind = effectiveString(
            effectiveField(allocation, "object_kind"), "object_kind");
        const bool data_region = object_kind == "data_region";
        if (data_region) {
            requireEffectiveFields(
                allocation,
                {"allocation_id", "device_owner_id", "endpoint",
                 "length_bytes", "mkey", "object_kind", "owner_id",
                 "owner_kind", "pages", "path_id", "virtual_address"});
        } else {
            requireEffectiveFields(
                allocation,
                {"allocation_id", "device_owner_id", "endpoint",
                 "length_bytes", "object_kind", "owner_id", "owner_kind",
                 "pages", "path_id", "virtual_address"});
        }
        const std::uint64_t allocation_id = effectiveInteger(
            effectiveField(allocation, "allocation_id"), "allocation_id");
        if (allocation_id == 0 || allocation_id <= previous_id) {
            invalidEffectiveHardware(
                "host-memory allocation IDs must be positive and ascending");
        }
        previous_id = allocation_id;
        object_by_id.emplace(allocation_id, object_kind);
        for (const char* field : {
                 "device_owner_id", "length_bytes", "owner_id", "path_id"}) {
            requirePositive(
                effectiveInteger(effectiveField(allocation, field), field),
                field);
        }
        if (data_region) {
            requirePositive(
                effectiveInteger(effectiveField(allocation, "mkey"), "mkey"),
                "mkey");
        }
        const std::string& owner_kind = effectiveString(
            effectiveField(allocation, "owner_kind"), "owner_kind");
        std::map<std::string, std::string> expected_owners{
            {"qpc_icm", "queue_pair"},
            {"sq_ring", "send_queue"},
            {"rq_ring", "receive_queue"},
            {"cq_ring", "completion_queue"},
            {"doorbell_record", "send_queue"},
            {"data_region", "memory_region"},
        };
        if (submission != nullptr) {
            expected_owners.emplace(
                "descriptor_queue", "submission_producer");
        }
        const auto expected_owner = expected_owners.find(object_kind);
        if (expected_owner == expected_owners.end()
            || owner_kind != expected_owner->second) {
            invalidEffectiveHardware(
                "host-memory object and owner kinds are incompatible");
        }
        const std::string& endpoint = effectiveString(
            effectiveField(allocation, "endpoint"), "endpoint");
        if (endpoint != "host_pinned_memory" && endpoint != "gpu_memory") {
            invalidEffectiveHardware(
                "host-memory allocation endpoint is incompatible");
        }
        if (object_kind == "qpc_icm" && endpoint != "host_pinned_memory") {
            invalidEffectiveHardware("QPC ICM must be host-pinned");
        }
        const std::uint64_t owner_id = effectiveInteger(
            effectiveField(allocation, "owner_id"), "owner_id");
        owner_by_id.emplace(allocation_id, owner_id);
        endpoint_by_id.emplace(allocation_id, endpoint);
        descriptor_queue_count += static_cast<std::size_t>(
            object_kind == "descriptor_queue");
        const std::uint64_t path_id = effectiveInteger(
            effectiveField(allocation, "path_id"), "path_id");
        if (path_id > std::numeric_limits<std::uint32_t>::max()) {
            invalidEffectiveHardware("host-memory path_id exceeds uint32");
        }
        const auto path = paths.find(static_cast<std::uint32_t>(path_id));
        if (path == paths.end() || !path->second.enabled
            || path->second.endpoint != endpoint) {
            invalidEffectiveHardware(
                "host-memory allocation path is incompatible");
        }

        const EffectiveJsonObject& pages = effectiveObject(
            effectiveField(allocation, "pages"), "page geometry");
        requireEffectiveFields(
            pages, {"page_size_bytes", "physical_page_addresses"});
        const std::uint64_t page_size = effectiveInteger(
            effectiveField(pages, "page_size_bytes"), "page_size_bytes");
        if (!isPowerOfTwo(page_size) || page_size < 4096) {
            invalidEffectiveHardware("page_size_bytes is invalid");
        }
        const auto& physical_pages = effectiveArray(
            effectiveField(pages, "physical_page_addresses"),
            "physical_page_addresses");
        if (physical_pages.empty()) {
            invalidEffectiveHardware(
                "physical_page_addresses must not be empty");
        }
        std::set<std::uint64_t> unique_pages;
        for (const EffectiveJsonValue& page : physical_pages) {
            const std::uint64_t address = effectiveInteger(
                page, "physical page address");
            if (address % page_size != 0
                || !unique_pages.insert(address).second) {
                invalidEffectiveHardware(
                    "physical pages must be aligned and unique");
            }
        }
    }
    for (const auto& field_and_kind : {
             std::pair<const char*, const char*>{
                 "qpc_icm_allocation_id", "qpc_icm"},
             {"sq_ring_allocation_id", "sq_ring"},
             {"rq_ring_allocation_id", "rq_ring"},
             {"cq_ring_allocation_id", "cq_ring"},
             {"doorbell_record_allocation_id", "doorbell_record"}}) {
        const std::uint64_t allocation_id = effectiveInteger(
            effectiveField(binding, field_and_kind.first),
            field_and_kind.first);
        const auto found = object_by_id.find(allocation_id);
        if (found == object_by_id.end()
            || found->second != field_and_kind.second) {
            invalidEffectiveHardware(
                std::string(field_and_kind.first)
                + " has a missing or mistyped allocation");
        }
        const std::string_view bound_kind(field_and_kind.second);
        if (submission != nullptr && bound_kind != "qpc_icm"
            && bound_kind != "rq_ring"
            && endpoint_by_id.at(allocation_id)
                != submission->queue_endpoint) {
            invalidEffectiveHardware(
                std::string(field_and_kind.first)
                + " disagrees with the submission endpoint");
        }
    }
    if (submission != nullptr
        && submission->producer_shape == "cpu_proxy") {
        const auto object = object_by_id.find(
            submission->descriptor_queue_allocation_id);
        if (descriptor_queue_count != 1 || object == object_by_id.end()
            || object->second != "descriptor_queue"
            || owner_by_id.at(object->first)
                != submission->descriptor_writer_id
            || endpoint_by_id.at(object->first) != "host_pinned_memory") {
            invalidEffectiveHardware(
                "CPU-proxy descriptor allocation is incompatible");
        }
    } else if (submission != nullptr && descriptor_queue_count != 0) {
        invalidEffectiveHardware(
            "non-proxy submission cannot carry a descriptor queue");
    }
}

void validateEffectiveHardwareJson(std::string_view bytes) {
    const EffectiveJsonValue root = EffectiveJsonParser(bytes).parse();
    const EffectiveJsonObject& hardware =
        effectiveObject(root, "root");
    const std::string& schema = effectiveString(
        effectiveField(hardware, "schema"), "schema");
    const bool submission_schema =
        schema == kRnicEffectiveHardwareSubmissionSchema;
    const bool host_memory_schema = submission_schema
        || schema == kRnicEffectiveHardwareHostMemorySchema;
    if (schema != kRnicEffectiveHardwareSchema && !host_memory_schema) {
        invalidEffectiveHardware("has an unsupported schema");
    }
    if (submission_schema) {
        requireEffectiveFields(
            hardware,
            {"dma", "host_memory", "network", "qpc", "schema",
             "submission", "work_queue"});
    } else if (host_memory_schema) {
        requireEffectiveFields(
            hardware,
            {"dma", "host_memory", "network", "qpc", "schema",
             "work_queue"});
    } else {
        requireEffectiveFields(
            hardware,
            {"dma", "network", "qpc", "schema", "work_queue"});
    }
    validateModule(effectiveField(hardware, "network"), "network");
    validateModule(effectiveField(hardware, "qpc"), "qpc");
    const EffectiveJsonObject& qpc = effectiveObject(
        effectiveField(hardware, "qpc"), "qpc");
    const bool qpc_enabled = effectiveBoolean(
        effectiveField(qpc, "enabled"), "qpc enabled");

    std::optional<ValidatedSubmission> submission;
    if (submission_schema) {
        submission = validateSubmission(
            effectiveField(hardware, "submission"));
    }

    const EffectiveJsonObject& dma = effectiveObject(
        effectiveField(hardware, "dma"), "dma");
    const bool dma_enabled = effectiveBoolean(
        effectiveField(dma, "enabled"), "DMA enabled");
    std::map<std::uint32_t, ValidatedEffectivePath> paths;
    if (dma_enabled) {
        requireEffectiveFields(
            dma, {"enabled", "fabric", "fabric_scope", "work_queue"});
        const std::string& scope = effectiveString(
            effectiveField(dma, "fabric_scope"), "fabric_scope");
        if (scope != "owned" && scope != "shared") {
            invalidEffectiveHardware("fabric_scope must be owned or shared");
        }
        paths = validateFabric(effectiveField(dma, "fabric"));
        validateBinding(
            effectiveField(dma, "work_queue"),
            paths,
            submission.has_value()
                ? std::string_view(submission->queue_endpoint)
                : std::string_view("host_pinned_memory"));
    } else {
        requireEffectiveFields(dma, {"enabled"});
    }
    if (host_memory_schema) {
        if (!dma_enabled || !qpc_enabled) {
            invalidEffectiveHardware("host memory requires DMA and QPC");
        }
        validateHostMemory(
            effectiveField(hardware, "host_memory"),
            paths,
            submission.has_value() ? &*submission : nullptr);
    }

    const EffectiveJsonObject& queue = effectiveObject(
        effectiveField(hardware, "work_queue"), "work_queue");
    if (dma_enabled && qpc_enabled) {
        requireEffectiveFields(
            queue,
            {"cq_depth", "qpc_lookup_service_ps", "scheduler_service_ps",
             "sq_depth"});
    } else if (dma_enabled) {
        requireEffectiveFields(
            queue, {"cq_depth", "scheduler_service_ps", "sq_depth"});
    } else if (qpc_enabled) {
        requireEffectiveFields(
            queue,
            {"cq_depth", "cqe_write_service_ps", "doorbell_service_ps",
             "qpc_lookup_service_ps", "scheduler_service_ps", "sq_depth",
             "wqe_fetch_service_ps"});
    } else {
        requireEffectiveFields(
            queue,
            {"cq_depth", "cqe_write_service_ps", "doorbell_service_ps",
             "scheduler_service_ps", "sq_depth", "wqe_fetch_service_ps"});
    }
    requirePositive(
        effectiveInteger(effectiveField(queue, "sq_depth"), "sq_depth"),
        "sq_depth");
    requirePositive(
        effectiveInteger(effectiveField(queue, "cq_depth"), "cq_depth"),
        "cq_depth");
    for (const auto& field_and_value : queue) {
        if (field_and_value.first.size() >= 3
            && field_and_value.first.compare(
                field_and_value.first.size() - 3, 3, "_ps") == 0) {
            requireTimestamp(
                effectiveInteger(
                    field_and_value.second, field_and_value.first.c_str()),
                field_and_value.first.c_str());
        }
    }
}

std::uint64_t checkedAdd(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const char* message) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::overflow_error(message);
    }
    return lhs + rhs;
}

const char* endpointName(PcieEndpointKind endpoint) noexcept {
    switch (endpoint) {
    case PcieEndpointKind::MmioBar:
        return "mmio_bar";
    case PcieEndpointKind::HostPinnedMemory:
        return "host_pinned_memory";
    case PcieEndpointKind::GpuMemory:
        return "gpu_memory";
    case PcieEndpointKind::DeviceMemory:
        return "device_memory";
    default:
        return "invalid";
    }
}

const char* wqeOpcodeName(WqeOpcode opcode) noexcept {
    switch (opcode) {
    case WqeOpcode::Send:
        return "send";
    default:
        return "invalid";
    }
}

const char* wqeStateName(WqeState state) noexcept {
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
    default:
        return "invalid";
    }
}

const char* completionStatusName(CompletionStatus status) noexcept {
    switch (status) {
    case CompletionStatus::Success:
        return "success";
    case CompletionStatus::TransportError:
        return "transport_error";
    case CompletionStatus::NetworkRejected:
        return "network_rejected";
    default:
        return "invalid";
    }
}

std::string analyticalProfileJson(
    const PcieAnalyticalDelayProfile& profile) {
    JsonFields fields{{"kind", jsonString(toString(profile.kind))}};
    switch (profile.kind) {
    case PcieAnalyticalDelayKind::Disabled:
        break;
    case PcieAnalyticalDelayKind::Fixed:
        fields.emplace_back(
            "incidence_probability_ppm",
            jsonInteger(profile.incidence_probability_ppm));
        fields.emplace_back("mean_ps", jsonInteger(profile.mean_ps));
        break;
    case PcieAnalyticalDelayKind::Gaussian:
        fields.emplace_back(
            "incidence_probability_ppm",
            jsonInteger(profile.incidence_probability_ppm));
        fields.emplace_back("mean_ps", jsonInteger(profile.mean_ps));
        fields.emplace_back(
            "standard_deviation_ps",
            jsonInteger(profile.standard_deviation_ps));
        break;
    case PcieAnalyticalDelayKind::GaussianTailMixture:
        fields.emplace_back(
            "incidence_probability_ppm",
            jsonInteger(profile.incidence_probability_ppm));
        fields.emplace_back("mean_ps", jsonInteger(profile.mean_ps));
        fields.emplace_back(
            "standard_deviation_ps",
            jsonInteger(profile.standard_deviation_ps));
        fields.emplace_back(
            "tail_mean_ps",
            jsonInteger(profile.tail_mean_ps));
        fields.emplace_back(
            "tail_probability_ppm",
            jsonInteger(profile.tail_probability_ppm));
        fields.emplace_back(
            "tail_standard_deviation_ps",
            jsonInteger(profile.tail_standard_deviation_ps));
        break;
    default:
        throw std::logic_error(
            "invalid analytical profile in constructed RNIC device");
    }
    return jsonObject(std::move(fields));
}

std::string pathPenaltiesJson(const PciePathPenaltyProfiles& profiles) {
    return jsonObject({
        {"acs", analyticalProfileJson(profiles.acs)},
        {"ddio_miss", analyticalProfileJson(profiles.ddio_miss)},
        {"gpu_direct", analyticalProfileJson(profiles.gpu_direct)},
        {"iommu", analyticalProfileJson(profiles.iommu)},
        {"numa", analyticalProfileJson(profiles.numa)},
        {"switch_path", analyticalProfileJson(profiles.switch_path)},
    });
}

std::string pathConfigJson(const PciePathConfig& path) {
    if (!path.enabled) {
        return jsonObject({
            {"enabled", jsonBoolean(false)},
            {"path_id", jsonInteger(path.path_id)},
        });
    }
    return jsonObject({
        {"analytical_penalties", pathPenaltiesJson(path.analytical_penalties)},
        {"base_latency_ps", jsonInteger(path.base_latency_ps)},
        {"enabled", jsonBoolean(path.enabled)},
        {"endpoint", jsonString(endpointName(path.endpoint))},
        {"path_id", jsonInteger(path.path_id)},
    });
}

std::string creditConfigJson(const PcieCreditConfig& credits) {
    return jsonObject({
        {"completion_data_credits", jsonInteger(credits.completion_data_credits)},
        {"completion_header_credits", jsonInteger(credits.completion_header_credits)},
        {"nonposted_header_credits", jsonInteger(credits.nonposted_header_credits)},
        {"posted_data_credits", jsonInteger(credits.posted_data_credits)},
        {"posted_header_credits", jsonInteger(credits.posted_header_credits)},
    });
}

std::string latencyProfileJson(const PcieLatencyProfile& profile) {
    std::vector<std::string> samples;
    samples.reserve(profile.samples_ps.size());
    for (Picoseconds sample : profile.samples_ps) {
        samples.push_back(jsonInteger(sample));
    }
    return jsonArray(samples);
}

std::string fabricConfigJson(const PcieFabricConfig& config) {
    std::vector<PciePathConfig> ordered_paths = config.paths;
    std::sort(
        ordered_paths.begin(),
        ordered_paths.end(),
        [](const PciePathConfig& lhs, const PciePathConfig& rhs) {
            return lhs.path_id < rhs.path_id;
        });
    std::vector<std::string> paths;
    paths.reserve(ordered_paths.size());
    for (const PciePathConfig& path : ordered_paths) {
        paths.push_back(pathConfigJson(path));
    }
    return jsonObject({
        {"analytical_seed", jsonInteger(config.analytical_seed)},
        {"completion_buffer_bytes", jsonInteger(config.completion_buffer_bytes)},
        {"completion_buffer_release_latency_ps", jsonInteger(config.completion_buffer_release_latency_ps)},
        {"completion_overhead_bytes", jsonInteger(config.completion_overhead_bytes)},
        {"credit_return_latency_ps", jsonInteger(config.credit_return_latency_ps)},
        {"data_credit_unit_bytes", jsonInteger(config.data_credit_unit_bytes)},
        {"device_to_host_credits", creditConfigJson(config.device_to_host_credits)},
        {"generation", jsonInteger(static_cast<std::uint8_t>(config.generation))},
        {"host_store_latency_ps", latencyProfileJson(config.host_store_latency_ps)},
        {"host_to_device_credits", creditConfigJson(config.host_to_device_credits)},
        {"lane_count", jsonInteger(config.lane_count)},
        {"max_outstanding_read_requests", jsonInteger(config.max_outstanding_read_requests)},
        {"max_payload_size_bytes", jsonInteger(config.max_payload_size_bytes)},
        {"max_read_request_size_bytes", jsonInteger(config.max_read_request_size_bytes)},
        {"max_tlps_per_transaction", jsonInteger(config.max_tlps_per_transaction)},
        {"paths", jsonArray(paths)},
        {"posted_write_overhead_bytes", jsonInteger(config.posted_write_overhead_bytes)},
        {"posted_write_visibility_latency_ps", latencyProfileJson(config.posted_write_visibility_latency_ps)},
        {"read_completion_boundary_bytes", jsonInteger(config.read_completion_boundary_bytes)},
        {"read_completion_latency_ps", latencyProfileJson(config.read_completion_latency_ps)},
        {"read_request_overhead_bytes", jsonInteger(config.read_request_overhead_bytes)},
    });
}

std::string pcieBindingJson(const WorkQueuePcieBinding& binding) {
    return jsonObject({
        {"pcie_completion_ordering_domain", jsonInteger(binding.pcie_completion_ordering_domain)},
        {"pcie_cq_first_byte_offset", jsonInteger(binding.pcie_cq_first_byte_offset)},
        {"pcie_cq_memory_path_id", jsonInteger(binding.pcie_cq_memory_path_id)},
        {"pcie_cqe_bytes", jsonInteger(binding.pcie_cqe_bytes)},
        {"pcie_doorbell_record_bytes", jsonInteger(binding.pcie_doorbell_record_bytes)},
        {"pcie_doorbell_record_first_byte_offset", jsonInteger(binding.pcie_doorbell_record_first_byte_offset)},
        {"pcie_doorbell_record_path_id", jsonInteger(binding.pcie_doorbell_record_path_id)},
        {"pcie_sq_first_byte_offset", jsonInteger(binding.pcie_sq_first_byte_offset)},
        {"pcie_sq_memory_path_id", jsonInteger(binding.pcie_sq_memory_path_id)},
        {"pcie_submission_ordering_domain", jsonInteger(binding.pcie_submission_ordering_domain)},
        {"pcie_uar_doorbell_bytes", jsonInteger(binding.pcie_uar_doorbell_bytes)},
        {"pcie_uar_first_byte_offset", jsonInteger(binding.pcie_uar_first_byte_offset)},
        {"pcie_uar_path_id", jsonInteger(binding.pcie_uar_path_id)},
        {"pcie_wqe_bytes", jsonInteger(binding.pcie_wqe_bytes)},
    });
}

std::string hostMemoryPagesJson(const HostMemoryPageGeometry& pages) {
    std::vector<std::string> physical_pages;
    physical_pages.reserve(pages.physical_page_addresses.size());
    for (const std::uint64_t address : pages.physical_page_addresses) {
        physical_pages.push_back(jsonInteger(address));
    }
    return jsonObject({
        {"page_size_bytes", jsonInteger(pages.page_size_bytes)},
        {"physical_page_addresses", jsonArray(physical_pages)},
    });
}

std::string hostMemoryAllocationJson(
    const HostMemoryAllocation& allocation) {
    JsonFields fields{
        {"allocation_id", jsonInteger(allocation.allocation_id)},
        {"device_owner_id", jsonInteger(allocation.device_owner_id)},
        {"endpoint", jsonString(endpointName(allocation.endpoint))},
        {"length_bytes", jsonInteger(allocation.length_bytes)},
        {"object_kind", jsonString(toString(allocation.object_kind))},
        {"owner_id", jsonInteger(allocation.owner_id)},
        {"owner_kind", jsonString(toString(allocation.owner_kind))},
        {"pages", hostMemoryPagesJson(allocation.pages)},
        {"path_id", jsonInteger(allocation.path_id)},
        {"virtual_address", jsonInteger(allocation.virtual_address)},
    };
    if (allocation.mkey.has_value()) {
        fields.emplace_back("mkey", jsonInteger(*allocation.mkey));
    }
    return jsonObject(std::move(fields));
}

std::string hostMemoryConfigJson(const RnicHostMemoryConfig& memory) {
    std::vector<HostMemoryAllocation> ordered = memory.allocations;
    std::sort(
        ordered.begin(),
        ordered.end(),
        [](const HostMemoryAllocation& lhs,
           const HostMemoryAllocation& rhs) {
            return lhs.allocation_id < rhs.allocation_id;
        });
    std::vector<std::string> allocations;
    allocations.reserve(ordered.size());
    for (const HostMemoryAllocation& allocation : ordered) {
        allocations.push_back(hostMemoryAllocationJson(allocation));
    }
    return jsonObject({
        {"allocations", jsonArray(allocations)},
        {"device_owner_id", jsonInteger(memory.device_owner_id)},
        {"enabled", jsonBoolean(memory.enabled)},
        {"registry", jsonObject({
            {"mpt_entry_bytes", jsonInteger(memory.registry.mpt_entry_bytes)},
            {"mpt_first_byte_offset", jsonInteger(memory.registry.mpt_first_byte_offset)},
            {"mtt_entry_bytes", jsonInteger(memory.registry.mtt_entry_bytes)},
            {"mtt_first_byte_offset", jsonInteger(memory.registry.mtt_first_byte_offset)},
            {"queue_page_list_entry_bytes", jsonInteger(memory.registry.queue_page_list_entry_bytes)},
            {"queue_page_list_first_byte_offset", jsonInteger(memory.registry.queue_page_list_first_byte_offset)},
            {"translation_path_id", jsonInteger(memory.registry.translation_path_id)},
        })},
        {"work_queue", jsonObject({
            {"cq_ring_allocation_id", jsonInteger(memory.work_queue.cq_ring_allocation_id)},
            {"doorbell_record_allocation_id", jsonInteger(memory.work_queue.doorbell_record_allocation_id)},
            {"qpc_context_bytes", jsonInteger(memory.work_queue.qpc_context_bytes)},
            {"qpc_icm_allocation_id", jsonInteger(memory.work_queue.qpc_icm_allocation_id)},
            {"rq_ring_allocation_id", jsonInteger(memory.work_queue.rq_ring_allocation_id)},
            {"sq_ring_allocation_id", jsonInteger(memory.work_queue.sq_ring_allocation_id)},
        })},
    });
}

std::string submissionProfileJson(const RnicSubmissionProfile& profile) {
    if (profile.version != kRnicSubmissionProfileVersion
        || profile.producer.version != kRnicSubmissionAgentVersion
        || profile.cq_consumer.version != kRnicSubmissionAgentVersion) {
        throw std::logic_error(
            "constructed RNIC device has an invalid submission profile");
    }
    return jsonObject({
        {"cq_consumer_id", jsonInteger(profile.cq_consumer.id)},
        {"cq_consumer_kind", jsonString(toString(profile.cq_consumer.kind))},
        {"descriptor_queue_allocation_id",
         jsonInteger(profile.descriptor_queue_allocation_id)},
        {"descriptor_queue_endpoint",
         jsonString(
             profile.descriptor_queue_endpoint.has_value()
                 ? endpointName(*profile.descriptor_queue_endpoint)
                 : "none")},
        {"descriptor_writer_id",
         jsonInteger(
             profile.descriptor_writer.has_value()
                 ? profile.descriptor_writer->id
                 : 0)},
        {"descriptor_writer_kind",
         jsonString(
             profile.descriptor_writer.has_value()
                 ? toString(profile.descriptor_writer->kind)
                 : "none")},
        {"producer_id", jsonInteger(profile.producer.id)},
        {"producer_kind", jsonString(toString(profile.producer.kind))},
        {"producer_shape", jsonString(toString(profile.producer_shape))},
        {"queue_endpoint", jsonString(endpointName(profile.queue_endpoint))},
        {"rnic_requester_id", jsonInteger(profile.rnic_requester_id)},
        {"uar_mapping_owner",
         jsonString(toString(profile.uar_mapping_owner))},
    });
}

std::string workQueueHardwareJson(
    const RnicDeviceConfig& config) {
    const WorkQueueConfig& queue = config.work_queue;
    JsonFields fields{
        {"cq_depth", jsonInteger(queue.cq_depth)},
        {"scheduler_service_ps", jsonInteger(queue.scheduler_service_ps)},
        {"sq_depth", jsonInteger(queue.sq_depth)},
    };
    if (config.qpc.enabled) {
        fields.emplace_back(
            "qpc_lookup_service_ps",
            jsonInteger(queue.qpc_lookup_service_ps));
    }
    if (!config.dma.enabled) {
        fields.emplace_back(
            "cqe_write_service_ps",
            jsonInteger(queue.cqe_write_service_ps));
        fields.emplace_back(
            "doorbell_service_ps",
            jsonInteger(queue.doorbell_service_ps));
        fields.emplace_back(
            "wqe_fetch_service_ps",
            jsonInteger(queue.wqe_fetch_service_ps));
    }
    return jsonObject(std::move(fields));
}

std::string timelineJson(const WqeTimeline& timeline) {
    return jsonObject({
        {"admitted_at_ps", jsonOptionalInteger(timeline.admitted_at_ps)},
        {"cqe_visible_at_ps", jsonOptionalInteger(timeline.cqe_visible_at_ps)},
        {"doorbell_seen_at_ps", jsonOptionalInteger(timeline.doorbell_seen_at_ps)},
        {"doorbelled_at_ps", jsonOptionalInteger(timeline.doorbelled_at_ps)},
        {"first_packet_at_ps", jsonOptionalInteger(timeline.first_packet_at_ps)},
        {"last_packet_at_ps", jsonOptionalInteger(timeline.last_packet_at_ps)},
        {"network_accepted_at_ps", jsonOptionalInteger(timeline.network_accepted_at_ps)},
        {"network_outcome_at_ps", jsonOptionalInteger(timeline.network_outcome_at_ps)},
        {"polled_at_ps", jsonOptionalInteger(timeline.polled_at_ps)},
        {"posted_at_ps", jsonInteger(timeline.posted_at_ps)},
        {"qpc_ready_at_ps", jsonOptionalInteger(timeline.qpc_ready_at_ps)},
        {"sq_reclaimed_at_ps", jsonOptionalInteger(timeline.sq_reclaimed_at_ps)},
        {"transport_retired_at_ps", jsonOptionalInteger(timeline.transport_retired_at_ps)},
        {"wqe_fetch_begin_at_ps", jsonOptionalInteger(timeline.wqe_fetch_begin_at_ps)},
        {"wqe_fetch_end_at_ps", jsonOptionalInteger(timeline.wqe_fetch_end_at_ps)},
    });
}

std::string projectionKeyJson(const RnicWqeProjectionKey& key) {
    return jsonObject({
        {"endpoint", jsonInteger(key.endpoint)},
        {"post_sequence", jsonInteger(key.post_sequence)},
        {"session_id", jsonString(key.session_id)},
        {"wq_id", jsonInteger(key.wq_id)},
        {"wq_kind", jsonString(toString(key.wq_kind))},
    });
}

std::string wqeProjectionJson(const RnicWqeProjectionRecord& record) {
    const std::optional<std::string> completion_status =
        record.completion_status.has_value()
        ? std::optional<std::string>(
            completionStatusName(*record.completion_status))
        : std::nullopt;
    return jsonObject({
        {"completion_status", jsonOptionalString(completion_status)},
        {"cq_consume_sequence", jsonOptionalInteger(record.cq_consume_sequence)},
        {"cq_id", jsonInteger(record.cq_id)},
        {"cq_producer_index", jsonOptionalInteger(record.cq_producer_index)},
        {"cqe_sequence", jsonOptionalInteger(record.cqe_sequence)},
        {"destination", jsonInteger(record.destination)},
        {"flow_id", jsonInteger(record.flow_id)},
        {"flow_tag", jsonInteger(record.flow_tag)},
        {"key", projectionKeyJson(record.key)},
        {"opcode", jsonString(wqeOpcodeName(record.opcode))},
        {"payload_bytes", jsonInteger(record.payload_bytes)},
        {"qpn", jsonOptionalInteger(record.qpn)},
        {"signaled", jsonBoolean(record.signaled)},
        {"source", jsonInteger(record.source)},
        {"state", jsonString(wqeStateName(record.state))},
        {"timeline", timelineJson(record.timeline)},
        {"transport_kind", jsonString(record.transport_kind)},
        {"transport_object_id", jsonInteger(record.transport_object_id)},
        {"wqe_id", jsonInteger(record.wqe_id)},
        {"wr_id", jsonOptionalInteger(record.wr_id)},
    });
}

std::string completionRowJson(const RnicCompletionCsvRow& row) {
    return jsonObject({
        {"completion_time_ps", jsonInteger(row.completion_time_ps)},
        {"cq_consume_sequence", jsonOptionalInteger(row.cq_consume_sequence)},
        {"cq_id", jsonOptionalInteger(row.cq_id)},
        {"cq_post_sequence", jsonOptionalInteger(row.cq_post_sequence)},
        {"destination", jsonInteger(row.destination)},
        {"fct_ps", jsonInteger(row.fct_ps)},
        {"flow_id", jsonInteger(row.flow_id)},
        {"payload_bytes", jsonInteger(row.payload_bytes)},
        {"profile", jsonString(row.profile)},
        {"rq_id", jsonOptionalInteger(row.rq_id)},
        {"source", jsonInteger(row.source)},
        {"sq_dispatch_sequence", jsonOptionalInteger(row.sq_dispatch_sequence)},
        {"sq_id", jsonOptionalInteger(row.sq_id)},
        {"sq_post_sequence", jsonOptionalInteger(row.sq_post_sequence)},
        {"start_time_ps", jsonInteger(row.start_time_ps)},
        {"tag", jsonInteger(row.tag)},
        {"transport_kind", jsonOptionalString(row.transport_kind)},
        {"transport_object_id", jsonOptionalInteger(row.transport_object_id)},
        {"wqe_id", jsonOptionalInteger(row.wqe_id)},
    });
}

std::string countersJson(const RnicAuthorityCounters& counters) {
    return jsonObject({
        {"legacy_ledger_constructed", jsonInteger(counters.legacy_ledger_constructed)},
        {"legacy_mutations", jsonInteger(counters.legacy_mutations)},
        {"native_posts", jsonInteger(counters.native_posts)},
        {"native_session_constructed", jsonInteger(counters.native_session_constructed)},
    });
}

void validateTransportProjection(const RnicTransportProjection& projection) {
    if (projection.wqe_id == 0) {
        throw std::invalid_argument(
            "RNIC transport projection requires a nonzero WQE ID");
    }
    requireNonblank(projection.transport_kind, "transport kind");
    if ((projection.transport_kind == "none")
        != (projection.transport_object_id == 0)) {
        throw std::invalid_argument(
            "RNIC transport kind none must match object ID zero");
    }
}

void validateProjectionTimeline(const WqeTimeline& timeline) {
    std::optional<Picoseconds> previous = timeline.posted_at_ps;
    const std::array<std::optional<Picoseconds>, 14> ordered{
        timeline.doorbelled_at_ps,
        timeline.doorbell_seen_at_ps,
        timeline.wqe_fetch_begin_at_ps,
        timeline.wqe_fetch_end_at_ps,
        timeline.qpc_ready_at_ps,
        timeline.admitted_at_ps,
        timeline.network_accepted_at_ps,
        timeline.first_packet_at_ps,
        timeline.last_packet_at_ps,
        timeline.network_outcome_at_ps,
        timeline.transport_retired_at_ps,
        timeline.cqe_visible_at_ps,
        timeline.polled_at_ps,
        timeline.sq_reclaimed_at_ps,
    };
    for (const std::optional<Picoseconds>& value : ordered) {
        if (!value.has_value()) {
            continue;
        }
        if (previous.has_value() && *value < *previous) {
            throw std::invalid_argument(
                "RNIC WQE projection timeline is not monotonic");
        }
        previous = value;
    }
    if (timeline.first_packet_at_ps.has_value()
            != timeline.last_packet_at_ps.has_value()
        || (timeline.first_packet_at_ps.has_value()
            && !timeline.network_accepted_at_ps.has_value())) {
        throw std::invalid_argument(
            "RNIC WQE packet timestamps have inconsistent applicability");
    }
}

bool isValidWqeState(WqeState state) noexcept {
    return std::string_view(wqeStateName(state)) != "invalid";
}

bool isValidWqeOpcode(WqeOpcode opcode) noexcept {
    return std::string_view(wqeOpcodeName(opcode)) != "invalid";
}

bool isValidCompletionStatus(CompletionStatus status) noexcept {
    return std::string_view(completionStatusName(status)) != "invalid";
}

void validateAuthorityCounters(
    RnicHardwareMode mode,
    const RnicAuthorityCounters& counters) {
    if (mode == RnicHardwareMode::Structural) {
        if (counters.native_session_constructed != 1
            || counters.legacy_ledger_constructed != 0
            || counters.legacy_mutations != 0) {
            throw std::invalid_argument(
                "structural RNIC result has invalid authority counters");
        }
        return;
    }
    if (mode == RnicHardwareMode::Bypass) {
        if (counters.native_session_constructed != 0
            || counters.legacy_ledger_constructed != 1
            || counters.native_posts != 0) {
            throw std::invalid_argument(
                "bypass RNIC result has invalid authority counters");
        }
        return;
    }
    throw std::invalid_argument("invalid RNIC hardware mode");
}

std::string csvField(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

template <typename Integer>
std::string csvOptionalInteger(const std::optional<Integer>& value) {
    return value.has_value() ? std::to_string(*value) : std::string{};
}

std::uint32_t rotateRight(std::uint32_t value, std::uint32_t count) {
    return (value >> count) | (value << (32U - count));
}

}  // namespace

std::string rnicSha256Hex(std::string_view bytes) {
    static constexpr std::array<std::uint32_t, 64> round_constants{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };
    if (bytes.size() > std::numeric_limits<std::uint64_t>::max() / 8U) {
        throw std::length_error("RNIC SHA-256 input is too large");
    }
    std::vector<std::uint8_t> message(bytes.begin(), bytes.end());
    const std::uint64_t bit_length =
        static_cast<std::uint64_t>(bytes.size()) * 8U;
    message.push_back(0x80U);
    while ((message.size() % 64U) != 56U) {
        message.push_back(0U);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        message.push_back(
            static_cast<std::uint8_t>(bit_length >> shift));
    }

    std::array<std::uint32_t, 8> state{
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    for (std::size_t offset = 0; offset < message.size(); offset += 64U) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const std::size_t base = offset + index * 4U;
            words[index] =
                (static_cast<std::uint32_t>(message[base]) << 24U)
                | (static_cast<std::uint32_t>(message[base + 1U]) << 16U)
                | (static_cast<std::uint32_t>(message[base + 2U]) << 8U)
                | static_cast<std::uint32_t>(message[base + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const std::uint32_t s0 =
                rotateRight(words[index - 15U], 7U)
                ^ rotateRight(words[index - 15U], 18U)
                ^ (words[index - 15U] >> 3U);
            const std::uint32_t s1 =
                rotateRight(words[index - 2U], 17U)
                ^ rotateRight(words[index - 2U], 19U)
                ^ (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + s0
                + words[index - 7U] + s1;
        }

        std::uint32_t a = state[0];
        std::uint32_t b = state[1];
        std::uint32_t c = state[2];
        std::uint32_t d = state[3];
        std::uint32_t e = state[4];
        std::uint32_t f = state[5];
        std::uint32_t g = state[6];
        std::uint32_t h = state[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const std::uint32_t sum1 = rotateRight(e, 6U)
                ^ rotateRight(e, 11U) ^ rotateRight(e, 25U);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 = h + sum1 + choose
                + round_constants[index] + words[index];
            const std::uint32_t sum0 = rotateRight(a, 2U)
                ^ rotateRight(a, 13U) ^ rotateRight(a, 22U);
            const std::uint32_t majority =
                (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::ostringstream digest;
    digest << std::hex << std::setfill('0');
    for (std::uint32_t word : state) {
        digest << std::setw(8) << word;
    }
    return digest.str();
}

std::string renderEffectiveHardwareConfigJson(const RnicDevice& device) {
    const RnicDeviceConfig& config = device.config();
    JsonFields dma{{"enabled", jsonBoolean(config.dma.enabled)}};
    if (config.dma.enabled) {
        const PcieFabric* const fabric = device.pcieFabric();
        const std::optional<WorkQueuePcieBinding> binding =
            device.pcieBinding();
        if (fabric == nullptr || !binding.has_value()) {
            throw std::logic_error(
                "constructed DMA-enabled RNIC device lost its fabric");
        }
        dma.emplace_back("fabric", fabricConfigJson(fabric->config()));
        dma.emplace_back(
            "fabric_scope",
            jsonString(
                device.usesSharedPcieFabric() ? "shared" : "owned"));
        dma.emplace_back("work_queue", pcieBindingJson(*binding));
    }
    JsonFields hardware{
        {"dma", jsonObject(std::move(dma))},
        {"network", jsonObject({{"enabled", jsonBoolean(config.network.enabled)}})},
        {"qpc", jsonObject({{"enabled", jsonBoolean(config.qpc.enabled)}})},
        {"work_queue", workQueueHardwareJson(config)},
    };
    if (config.host_memory.enabled) {
        if (!device.submissionProfile().has_value()) {
            throw std::logic_error(
                "constructed host-memory RNIC lost its submission profile");
        }
        hardware.emplace_back(
            "host_memory", hostMemoryConfigJson(config.host_memory));
        hardware.emplace_back(
            "schema", jsonString(kRnicEffectiveHardwareSubmissionSchema));
        hardware.emplace_back(
            "submission",
            submissionProfileJson(*device.submissionProfile()));
    } else {
        hardware.emplace_back(
            "schema", jsonString(kRnicEffectiveHardwareSchema));
    }
    return jsonObject(std::move(hardware));
}

std::string effectiveHardwareConfigSha256(const RnicDevice& device) {
    return rnicSha256Hex(renderEffectiveHardwareConfigJson(device));
}

RnicSessionConfigRecord makeStructuralSessionConfigRecord(
    std::string session_id,
    std::string transport_policy,
    const RnicDevice& device) {
    RnicSessionConfigRecord record;
    record.session_id = std::move(session_id);
    record.transport_policy = std::move(transport_policy);
    record.hardware_mode = RnicHardwareMode::Structural;
    record.authority = RnicWqeAuthority::SimllmNativeRnicSession;
    record.effective_hardware_json =
        renderEffectiveHardwareConfigJson(device);
    record.hardware_config_sha256 =
        rnicSha256Hex(*record.effective_hardware_json);
    validateRnicSessionConfigRecord(record);
    return record;
}

RnicSessionConfigRecord makeBypassSessionConfigRecord(
    std::string session_id,
    std::string transport_policy) {
    RnicSessionConfigRecord record;
    record.session_id = std::move(session_id);
    record.transport_policy = std::move(transport_policy);
    record.hardware_mode = RnicHardwareMode::Bypass;
    record.authority = RnicWqeAuthority::AtlahsWqeLedger;
    validateRnicSessionConfigRecord(record);
    return record;
}

void validateRnicSessionConfigRecord(
    const RnicSessionConfigRecord& record) {
    if (record.version != kRnicSessionConfigRecordVersion) {
        throw std::invalid_argument(
            "unsupported RNIC session config record version");
    }
    requireNonblank(record.session_id, "ID");
    requireNonblank(record.transport_policy, "transport policy");
    if (record.hardware_mode == RnicHardwareMode::Structural) {
        if (record.authority
                != RnicWqeAuthority::SimllmNativeRnicSession
            || !record.effective_hardware_json.has_value()
            || !record.hardware_config_sha256.has_value()) {
            throw std::invalid_argument(
                "structural RNIC config record has inconsistent authority");
        }
        if (!isSha256(*record.hardware_config_sha256)) {
            throw std::invalid_argument(
                "RNIC hardware hash is not lowercase SHA-256");
        }
        validateEffectiveHardwareJson(*record.effective_hardware_json);
        if (*record.hardware_config_sha256
            != rnicSha256Hex(*record.effective_hardware_json)) {
            throw std::invalid_argument(
                "RNIC effective hardware hash does not match its bytes");
        }
        return;
    }
    if (record.hardware_mode == RnicHardwareMode::Bypass) {
        if (record.authority != RnicWqeAuthority::AtlahsWqeLedger
            || record.effective_hardware_json.has_value()
            || record.hardware_config_sha256.has_value()) {
            throw std::invalid_argument(
                "bypass RNIC config record has inconsistent authority");
        }
        return;
    }
    throw std::invalid_argument("invalid RNIC hardware mode");
}

std::string renderRnicSessionConfigJson(
    const RnicSessionConfigRecord& record) {
    validateRnicSessionConfigRecord(record);
    return jsonObject({
        {"authority", jsonString(toString(record.authority))},
        {"effective_hardware", record.effective_hardware_json.value_or("null")},
        {"hardware_config_sha256", jsonOptionalString(record.hardware_config_sha256)},
        {"hardware_mode", jsonString(toString(record.hardware_mode))},
        {"schema", jsonString(kRnicSessionConfigSchema)},
        {"session_id", jsonString(record.session_id)},
        {"transport_policy", jsonString(record.transport_policy)},
    });
}

RnicAuthorityAudit::RnicAuthorityAudit(
    RnicHardwareMode mode,
    RnicAuthoritySelection selection)
    : hardware_mode_(mode),
      authority_(RnicWqeAuthority::SimllmNativeRnicSession) {
    if (mode == RnicHardwareMode::Structural) {
        if (!selection.native_session_enabled
            || selection.legacy_ledger_enabled) {
            throw std::invalid_argument(
                "structural RNIC mode requires only the native authority");
        }
        authority_ = RnicWqeAuthority::SimllmNativeRnicSession;
        counters_.native_session_constructed = 1;
        return;
    }
    if (mode == RnicHardwareMode::Bypass) {
        if (selection.native_session_enabled
            || !selection.legacy_ledger_enabled) {
            throw std::invalid_argument(
                "bypass RNIC mode requires only AtlahsWqeLedger");
        }
        authority_ = RnicWqeAuthority::AtlahsWqeLedger;
        counters_.legacy_ledger_constructed = 1;
        return;
    }
    throw std::invalid_argument("invalid RNIC hardware mode");
}

void RnicAuthorityAudit::noteNativePost(std::uint64_t count) {
    if (hardware_mode_ != RnicHardwareMode::Structural) {
        throw std::logic_error(
            "native RNIC post recorded while native authority is disabled");
    }
    counters_.native_posts = checkedAdd(
        counters_.native_posts,
        count,
        "RNIC native-post audit counter overflow");
}

void RnicAuthorityAudit::noteLegacyMutation(std::uint64_t count) {
    if (hardware_mode_ != RnicHardwareMode::Bypass) {
        throw std::logic_error(
            "legacy RNIC mutation recorded while ledger is disabled");
    }
    counters_.legacy_mutations = checkedAdd(
        counters_.legacy_mutations,
        count,
        "RNIC legacy-mutation audit counter overflow");
}

RnicHardwareMode RnicAuthorityAudit::hardwareMode() const noexcept {
    return hardware_mode_;
}

RnicWqeAuthority RnicAuthorityAudit::authority() const noexcept {
    return authority_;
}

const RnicAuthorityCounters& RnicAuthorityAudit::counters() const noexcept {
    return counters_;
}

RnicSessionResultRecord projectStructuralSessionResult(
    const RnicSessionConfigRecord& config_record,
    const RnicDevice& device,
    const std::vector<CompletionEntry>& polled_completions,
    const RnicAuthorityAudit& authority_audit,
    const std::vector<RnicTransportProjection>& transport) {
    validateRnicSessionConfigRecord(config_record);
    if (config_record.hardware_mode != RnicHardwareMode::Structural
        || authority_audit.hardwareMode()
            != RnicHardwareMode::Structural
        || authority_audit.authority()
            != RnicWqeAuthority::SimllmNativeRnicSession) {
        throw std::invalid_argument(
            "native RNIC projection requires structural authority");
    }
    if (config_record.hardware_config_sha256
        != std::optional<std::string>(
            effectiveHardwareConfigSha256(device))) {
        throw std::invalid_argument(
            "native RNIC projection hardware hash disagrees with device");
    }
    device.validateInvariants();
    if (device.fatal() || device.hasPendingPhysicalWork()
        || device.occupiedSqEntries() != 0
        || device.completionQueueDepth() != 0
        || device.unpublishedWqeCount() != 0) {
        throw std::logic_error(
            "native RNIC projection requires nonfatal quiescence");
    }
    if (authority_audit.counters().native_posts
            != device.counters().posted_wqes
        || device.counters().posted_wqes != device.records().size()) {
        throw std::logic_error(
            "native RNIC post audit disagrees with the authority");
    }
    if (polled_completions.size()
            != device.counters().cqes_polled) {
        throw std::logic_error(
            "native RNIC projection lost returned CQ entries");
    }

    std::map<WqeId, const CompletionEntry*> completion_by_wqe;
    for (const CompletionEntry& completion : polled_completions) {
        if (completion.wqe_id == 0
            || !completion_by_wqe.emplace(
                    completion.wqe_id, &completion).second) {
            throw std::invalid_argument(
                "native RNIC projection has a duplicate CQ WQE ID");
        }
    }
    std::map<WqeId, RnicTransportProjection> transport_by_wqe;
    for (const RnicTransportProjection& projection : transport) {
        validateTransportProjection(projection);
        if (!transport_by_wqe.emplace(
                projection.wqe_id, projection).second) {
            throw std::invalid_argument(
                "native RNIC projection has duplicate transport metadata");
        }
    }

    RnicSessionResultRecord result;
    result.session_id = config_record.session_id;
    result.hardware_mode = config_record.hardware_mode;
    result.authority = config_record.authority;
    result.transport_policy = config_record.transport_policy;
    result.hardware_config_sha256 =
        config_record.hardware_config_sha256;
    result.authority_counters = authority_audit.counters();
    result.quiescent = true;
    result.wqes.reserve(device.records().size());
    result.completion_rows.reserve(device.records().size());

    const RnicDeviceConfig& device_config = device.config();
    std::set<WqeId> consumed_transport;
    for (const WqeRecord& native : device.records()) {
        if (!native.timeline.network_outcome_at_ps.has_value()
            || !native.timeline.transport_retired_at_ps.has_value()) {
            throw std::logic_error(
                "quiescent native RNIC WQE lacks a terminal timeline");
        }
        const auto completion_item = completion_by_wqe.find(native.wqe_id);
        const CompletionEntry* completion = completion_item
            == completion_by_wqe.end()
            ? nullptr
            : completion_item->second;
        const bool requires_cqe = native.request.signaled
            || native.completion_status != CompletionStatus::Success;
        if (requires_cqe != (completion != nullptr)) {
            throw std::logic_error(
                "native RNIC CQ cardinality disagrees with signaling and status");
        }
        if (completion != nullptr) {
            if (completion->wr_id != native.request.wr_id
                || completion->sq_sequence != native.sq_sequence
                || completion->qpn != device_config.identity.qpn
                || completion->opcode != native.request.opcode
                || completion->status != native.completion_status
                || native.timeline.cqe_visible_at_ps
                    != std::optional<Picoseconds>(
                        completion->visible_at_ps)
                || native.timeline.polled_at_ps
                    != std::optional<Picoseconds>(
                        completion->polled_at_ps)) {
                throw std::logic_error(
                    "native RNIC CQ entry disagrees with its WQE");
            }
        }

        RnicTransportProjection transport_projection;
        transport_projection.wqe_id = native.wqe_id;
        const auto transport_item = transport_by_wqe.find(native.wqe_id);
        if (transport_item != transport_by_wqe.end()) {
            transport_projection = transport_item->second;
            consumed_transport.insert(native.wqe_id);
        }

        RnicWqeProjectionRecord projected;
        projected.key = RnicWqeProjectionKey{
            config_record.session_id,
            device_config.work_queue.source,
            RnicWqKind::Send,
            device_config.work_queue.sq_id,
            native.sq_sequence,
        };
        projected.wqe_id = native.wqe_id;
        projected.wr_id = native.request.wr_id;
        projected.flow_id = native.request.flow_id;
        projected.flow_tag = native.request.flow_tag;
        projected.source = device_config.work_queue.source;
        projected.destination = native.request.destination;
        projected.payload_bytes = native.request.payload_bytes;
        projected.qpn = device_config.identity.qpn;
        projected.signaled = native.request.signaled;
        projected.opcode = native.request.opcode;
        projected.timeline = native.timeline;
        projected.state = native.state;
        projected.completion_status = native.completion_status;
        projected.cq_id = device_config.work_queue.cq_id;
        projected.transport_kind =
            transport_projection.transport_kind;
        projected.transport_object_id =
            transport_projection.transport_object_id;
        if (completion != nullptr) {
            projected.cqe_sequence = completion->cqe_sequence;
            projected.cq_producer_index =
                completion->cq_producer_index;
            projected.cq_consume_sequence =
                completion->cqe_sequence;
        }
        result.wqes.push_back(projected);

        const Picoseconds start = native.timeline.posted_at_ps;
        const Picoseconds finished =
            *native.timeline.network_outcome_at_ps;
        if (finished < start) {
            throw std::logic_error(
                "native RNIC flow completion predates its post");
        }
        RnicCompletionCsvRow row;
        row.profile = config_record.transport_policy;
        row.flow_id = native.request.flow_id;
        row.source = device_config.work_queue.source;
        row.destination = native.request.destination;
        row.tag = native.request.flow_tag;
        row.payload_bytes = native.request.payload_bytes;
        row.start_time_ps = start;
        row.completion_time_ps = finished;
        row.fct_ps = finished - start;
        row.wqe_id = native.wqe_id;
        row.sq_id = device_config.work_queue.sq_id;
        row.cq_id = device_config.work_queue.cq_id;
        row.sq_post_sequence = native.sq_sequence;
        row.sq_dispatch_sequence = native.sq_sequence;
        if (completion != nullptr) {
            row.cq_post_sequence = completion->cqe_sequence;
            row.cq_consume_sequence = completion->cqe_sequence;
        }
        row.transport_kind = transport_projection.transport_kind;
        row.transport_object_id =
            transport_projection.transport_object_id;
        result.completion_rows.push_back(std::move(row));
    }
    if (completion_by_wqe.size() != polled_completions.size()
        || consumed_transport.size() != transport_by_wqe.size()) {
        throw std::invalid_argument(
            "native RNIC projection references an unknown WQE");
    }
    std::sort(
        result.completion_rows.begin(),
        result.completion_rows.end(),
        [](const RnicCompletionCsvRow& lhs,
           const RnicCompletionCsvRow& rhs) {
            return std::tie(lhs.flow_id, lhs.wqe_id)
                < std::tie(rhs.flow_id, rhs.wqe_id);
        });
    validateRnicSessionResultRecord(result);
    return result;
}

RnicSessionResultRecord makeBypassSessionResultRecord(
    const RnicSessionConfigRecord& config_record,
    const RnicAuthorityAudit& authority_audit,
    std::vector<RnicWqeProjectionRecord> wqes,
    std::vector<RnicCompletionCsvRow> completion_rows,
    bool quiescent) {
    validateRnicSessionConfigRecord(config_record);
    if (config_record.hardware_mode != RnicHardwareMode::Bypass
        || authority_audit.hardwareMode() != RnicHardwareMode::Bypass
        || authority_audit.authority()
            != RnicWqeAuthority::AtlahsWqeLedger) {
        throw std::invalid_argument(
            "bypass RNIC result requires AtlahsWqeLedger authority");
    }
    RnicSessionResultRecord result;
    result.session_id = config_record.session_id;
    result.hardware_mode = config_record.hardware_mode;
    result.authority = config_record.authority;
    result.transport_policy = config_record.transport_policy;
    result.authority_counters = authority_audit.counters();
    result.quiescent = quiescent;
    result.wqes = std::move(wqes);
    result.completion_rows = std::move(completion_rows);
    std::sort(
        result.completion_rows.begin(),
        result.completion_rows.end(),
        [](const RnicCompletionCsvRow& lhs,
           const RnicCompletionCsvRow& rhs) {
            return std::tie(lhs.flow_id, lhs.wqe_id)
                < std::tie(rhs.flow_id, rhs.wqe_id);
        });
    validateRnicSessionResultRecord(result);
    return result;
}

void validateRnicSessionResultRecord(
    const RnicSessionResultRecord& record) {
    if (record.version != kRnicSessionResultRecordVersion) {
        throw std::invalid_argument(
            "unsupported RNIC session result record version");
    }
    requireNonblank(record.session_id, "ID");
    requireNonblank(record.transport_policy, "transport policy");
    validateAuthorityCounters(
        record.hardware_mode, record.authority_counters);
    if (record.hardware_mode == RnicHardwareMode::Structural) {
        if (record.authority
                != RnicWqeAuthority::SimllmNativeRnicSession
            || !record.hardware_config_sha256.has_value()
            || !isSha256(*record.hardware_config_sha256)
            || record.authority_counters.native_posts
                != record.wqes.size()) {
            throw std::invalid_argument(
                "structural RNIC result metadata is inconsistent");
        }
    } else if (record.hardware_mode == RnicHardwareMode::Bypass) {
        if (record.authority != RnicWqeAuthority::AtlahsWqeLedger
            || record.hardware_config_sha256.has_value()
            || record.wqes.size()
                > std::numeric_limits<std::uint64_t>::max() / 2U
            || record.authority_counters.legacy_mutations
                != static_cast<std::uint64_t>(record.wqes.size()) * 2U) {
            throw std::invalid_argument(
                "bypass RNIC result metadata is inconsistent");
        }
    } else {
        throw std::invalid_argument("invalid RNIC hardware mode");
    }

    using Key = std::tuple<
        std::string,
        std::uint32_t,
        RnicWqKind,
        std::uint64_t,
        std::uint64_t>;
    std::set<Key> keys;
    std::map<WqeId, const RnicWqeProjectionRecord*> wqe_by_id;
    std::set<std::pair<std::uint64_t, std::uint64_t>> cqe_sequences;
    for (const RnicWqeProjectionRecord& wqe : record.wqes) {
        if (wqe.key.session_id != record.session_id
            || wqe.key.endpoint != wqe.source
            || wqe.key.wq_kind != RnicWqKind::Send
            || wqe.key.wq_id == 0 || wqe.key.post_sequence == 0
            || wqe.wqe_id == 0 || wqe.cq_id == 0
            || (wqe.qpn.has_value() && *wqe.qpn == 0)
            || !isValidWqeOpcode(wqe.opcode)
            || !isValidWqeState(wqe.state)
            || !wqe.completion_status.has_value()
            || !isValidCompletionStatus(*wqe.completion_status)
            || !wqe.timeline.network_outcome_at_ps.has_value()
            || !wqe.timeline.transport_retired_at_ps.has_value()) {
            throw std::invalid_argument(
                "RNIC WQE projection has an invalid stable key");
        }
        validateProjectionTimeline(wqe.timeline);
        const Key key{
            wqe.key.session_id,
            wqe.key.endpoint,
            wqe.key.wq_kind,
            wqe.key.wq_id,
            wqe.key.post_sequence,
        };
        if (!keys.insert(key).second
            || !wqe_by_id.emplace(wqe.wqe_id, &wqe).second) {
            throw std::invalid_argument(
                "RNIC WQE projection has duplicate identity");
        }
        requireNonblank(wqe.transport_kind, "transport kind");
        const bool has_cqe = wqe.cqe_sequence.has_value();
        const bool requires_cqe = wqe.signaled
            || *wqe.completion_status != CompletionStatus::Success;
        if ((wqe.transport_kind == "none")
                != (wqe.transport_object_id == 0)
            || (wqe.cqe_sequence.has_value()
                != wqe.cq_producer_index.has_value())
            || (wqe.cqe_sequence.has_value()
                != wqe.cq_consume_sequence.has_value())
            || has_cqe != requires_cqe
            || wqe.timeline.cqe_visible_at_ps.has_value()
                != requires_cqe
            || wqe.timeline.polled_at_ps.has_value()
                != requires_cqe
            || (has_cqe
                && *wqe.cqe_sequence != *wqe.cq_consume_sequence)
            || (has_cqe && *wqe.cqe_sequence == 0)
            || (has_cqe
                && !cqe_sequences.insert(
                        {wqe.cq_id, *wqe.cqe_sequence}).second)
            || (record.hardware_mode == RnicHardwareMode::Structural
                && !wqe.qpn.has_value())) {
            throw std::invalid_argument(
                "RNIC WQE projection has inconsistent optional fields");
        }
        if ((requires_cqe && wqe.state != WqeState::Completed)
            || (!requires_cqe
                && wqe.state != WqeState::Reclaimed)) {
            throw std::invalid_argument(
                "RNIC result has a nonterminal WQE state");
        }
    }

    std::set<WqeId> completion_wqes;
    std::optional<std::pair<FlowId, WqeId>> prior_sort_key;
    for (const RnicCompletionCsvRow& row : record.completion_rows) {
        requireNonblank(row.profile, "completion profile");
        if (row.profile != record.transport_policy
            || row.completion_time_ps < row.start_time_ps
            || row.fct_ps
                != row.completion_time_ps - row.start_time_ps
            || !row.wqe_id.has_value()) {
            throw std::invalid_argument(
                "RNIC completion projection has invalid boundaries");
        }
        const std::pair<FlowId, WqeId> sort_key{
            row.flow_id, *row.wqe_id};
        if (prior_sort_key.has_value() && sort_key < *prior_sort_key) {
            throw std::invalid_argument(
                "RNIC completion projection is not sorted by flow ID");
        }
        prior_sort_key = sort_key;
        if (!completion_wqes.insert(*row.wqe_id).second) {
            throw std::invalid_argument(
                "RNIC completion projection duplicates a WQE");
        }
        const auto wqe_item = wqe_by_id.find(*row.wqe_id);
        if (wqe_item == wqe_by_id.end()) {
            throw std::invalid_argument(
                "RNIC completion projection references an unknown WQE");
        }
        const RnicWqeProjectionRecord& wqe = *wqe_item->second;
        if (row.flow_id != wqe.flow_id || row.source != wqe.source
            || row.destination != wqe.destination
            || row.tag != wqe.flow_tag
            || row.payload_bytes != wqe.payload_bytes
            || row.start_time_ps != wqe.timeline.posted_at_ps
            || row.completion_time_ps
                != wqe.timeline.network_outcome_at_ps
            || row.sq_id
                != std::optional<std::uint64_t>(wqe.key.wq_id)
            || row.cq_id
                != std::optional<std::uint64_t>(wqe.cq_id)
            || row.sq_post_sequence
                != std::optional<std::uint64_t>(
                    wqe.key.post_sequence)
            || row.sq_dispatch_sequence != row.sq_post_sequence
            || row.cq_post_sequence != wqe.cqe_sequence
            || row.cq_consume_sequence
                != wqe.cq_consume_sequence
            || row.transport_kind
                != std::optional<std::string>(wqe.transport_kind)
            || row.transport_object_id
                != std::optional<std::uint64_t>(
                    wqe.transport_object_id)) {
            throw std::invalid_argument(
                "RNIC completion row disagrees with WQE projection");
        }
        if (record.hardware_mode == RnicHardwareMode::Structural
            && row.rq_id.has_value()) {
            throw std::invalid_argument(
                "structural send projection cannot invent a receive WQ");
        }
    }
    if (completion_wqes.size() != record.wqes.size()) {
        throw std::invalid_argument(
            "RNIC result lost or duplicated a completion projection");
    }
}

std::string renderRnicSessionResultJson(
    const RnicSessionResultRecord& record) {
    validateRnicSessionResultRecord(record);
    std::vector<std::string> wqes;
    wqes.reserve(record.wqes.size());
    for (const RnicWqeProjectionRecord& wqe : record.wqes) {
        wqes.push_back(wqeProjectionJson(wqe));
    }
    std::vector<std::string> completion_rows;
    completion_rows.reserve(record.completion_rows.size());
    for (const RnicCompletionCsvRow& row : record.completion_rows) {
        completion_rows.push_back(completionRowJson(row));
    }
    return jsonObject({
        {"authority", jsonString(toString(record.authority))},
        {"authority_counters", countersJson(record.authority_counters)},
        {"completion_rows", jsonArray(completion_rows)},
        {"hardware_config_sha256", jsonOptionalString(record.hardware_config_sha256)},
        {"hardware_mode", jsonString(toString(record.hardware_mode))},
        {"quiescent", jsonBoolean(record.quiescent)},
        {"schema", jsonString(kRnicSessionResultSchema)},
        {"session_id", jsonString(record.session_id)},
        {"transport_policy", jsonString(record.transport_policy)},
        {"wqes", jsonArray(wqes)},
    });
}

std::string renderRnicBookkeepingProjectionJson(
    const RnicSessionResultRecord& record) {
    validateRnicSessionResultRecord(record);
    std::vector<std::string> wqes;
    wqes.reserve(record.wqes.size());
    for (const RnicWqeProjectionRecord& wqe : record.wqes) {
        wqes.push_back(wqeProjectionJson(wqe));
    }
    return jsonObject({
        {"authority", jsonString(toString(record.authority))},
        {"hardware_config_sha256", jsonOptionalString(record.hardware_config_sha256)},
        {"hardware_mode", jsonString(toString(record.hardware_mode))},
        {"schema", jsonString(kRnicBookkeepingProjectionSchema)},
        {"session_id", jsonString(record.session_id)},
        {"wqes", jsonArray(wqes)},
    });
}

std::string renderRnicCompletionCsv(
    const RnicSessionResultRecord& record) {
    validateRnicSessionResultRecord(record);
    std::ostringstream output;
    output << "profile,flow_id,source,destination,tag,payload_bytes,"
              "start_time_ps,completion_time_ps,fct_ps,"
              "wqe_id,sq_id,rq_id,cq_id,sq_post_sequence,"
              "sq_dispatch_sequence,cq_post_sequence,cq_consume_sequence,"
              "transport_kind,transport_object_id\n";
    for (const RnicCompletionCsvRow& row : record.completion_rows) {
        output << csvField(row.profile) << ',' << row.flow_id << ','
               << row.source << ',' << row.destination << ',' << row.tag
               << ',' << row.payload_bytes << ',' << row.start_time_ps
               << ',' << row.completion_time_ps << ',' << row.fct_ps
               << ',' << csvOptionalInteger(row.wqe_id)
               << ',' << csvOptionalInteger(row.sq_id)
               << ',' << csvOptionalInteger(row.rq_id)
               << ',' << csvOptionalInteger(row.cq_id)
               << ',' << csvOptionalInteger(row.sq_post_sequence)
               << ',' << csvOptionalInteger(row.sq_dispatch_sequence)
               << ',' << csvOptionalInteger(row.cq_post_sequence)
               << ',' << csvOptionalInteger(row.cq_consume_sequence)
               << ','
               << (row.transport_kind.has_value()
                       ? csvField(*row.transport_kind)
                       : std::string{})
               << ',' << csvOptionalInteger(row.transport_object_id)
               << '\n';
    }
    return output.str();
}

const char* toString(RnicHardwareMode mode) noexcept {
    switch (mode) {
    case RnicHardwareMode::Structural:
        return "structural";
    case RnicHardwareMode::Bypass:
        return "bypass";
    default:
        return "invalid";
    }
}

const char* toString(RnicWqeAuthority authority) noexcept {
    switch (authority) {
    case RnicWqeAuthority::SimllmNativeRnicSession:
        return "SimllmNativeRnicSession";
    case RnicWqeAuthority::AtlahsWqeLedger:
        return "AtlahsWqeLedger";
    default:
        return "invalid";
    }
}

const char* toString(RnicWqKind kind) noexcept {
    switch (kind) {
    case RnicWqKind::Send:
        return "send";
    case RnicWqKind::Receive:
        return "receive";
    case RnicWqKind::SharedReceive:
        return "shared_receive";
    default:
        return "invalid";
    }
}

}  // namespace simllm::rnic
