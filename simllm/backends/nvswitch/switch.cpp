#include "switch.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct simllm_nvs_context {
    std::vector<int32_t> chips;
    std::vector<uint64_t> inputs, outputs;
    uint64_t now = 0;
    size_t cursor = 0;
    uint32_t policy = 0;
    std::string error;
};

namespace {
uint64_t serialize(uint64_t bytes, uint64_t rate, uint64_t scale) {
    if (rate == 0 || bytes > std::numeric_limits<uint64_t>::max() / scale) {
        throw std::invalid_argument("invalid rate or serialization overflow");
    }
    const auto numerator = bytes * scale;
    return numerator / rate + (numerator % rate != 0);
}
}

extern "C" {
uint32_t simllm_nvs_abi_version(void) { return 1; }

simllm_nvs_context* simllm_nvs_create(
    const int32_t* port_chips, size_t port_count, uint32_t policy) {
    if (!port_chips || !port_count || policy > 1) return nullptr;
    try {
        for (size_t i = 0; i < port_count; ++i) {
            if (port_chips[i] < -1) return nullptr;
        }
        auto* context = new simllm_nvs_context;
        try {
            context->chips.assign(port_chips, port_chips + port_count);
            context->inputs.resize(port_count, 0);
            context->outputs.resize(port_count, 0);
            context->policy = policy;
        } catch (...) {
            delete context;
            throw;
        }
        return context;
    } catch (...) { return nullptr; }
}

void simllm_nvs_destroy(simllm_nvs_context* context) { delete context; }

const char* simllm_nvs_error(const simllm_nvs_context* context) {
    return context ? context->error.c_str() : "null NVSwitch context";
}

int simllm_nvs_plan(
    simllm_nvs_context* context, uint64_t now_ps, uint64_t crossbar_bytes_per_second,
    const simllm_nvs_head* heads, size_t head_count,
    const uint64_t* available_bytes, size_t pool_count,
    simllm_nvs_grant* grants, size_t grant_capacity, size_t* grant_count) {
    if (!context) return -1;
    try {
        if (!grant_count || (head_count && (!heads || !grants)) ||
            (pool_count && !available_bytes) || grant_capacity < head_count ||
            !crossbar_bytes_per_second || now_ps < context->now) {
            throw std::invalid_argument("invalid grant call or time rewind");
        }
        std::unordered_set<uint64_t> tokens;
        std::vector<uint64_t> finishes(head_count);
        std::vector<size_t> ordered;
        ordered.reserve(head_count);
        // Validate the entire call before selecting or committing any grant.
        for (size_t i = 0; i < head_count; ++i) {
            const auto& h = heads[i];
            if (h.input_port >= context->chips.size() ||
                h.output_port >= context->chips.size() || h.input_port == h.output_port ||
                context->chips[h.input_port] < 0 ||
                context->chips[h.input_port] != context->chips[h.output_port] ||
                h.pool >= pool_count || !h.wire_bytes || !tokens.insert(h.token).second) {
                throw std::invalid_argument("invalid or duplicate physical switch head");
            }
            const auto duration = std::max(
                serialize(h.wire_bytes, crossbar_bytes_per_second, 1000000000000ULL),
                serialize(h.wire_bytes, h.link_rate_bps, 8000000000000ULL));
            if (duration > std::numeric_limits<uint64_t>::max() - now_ps) {
                throw std::invalid_argument("switch completion overflows time horizon");
            }
            finishes[i] = now_ps + duration;
            if (context->inputs[h.input_port] <= now_ps &&
                context->outputs[h.output_port] <= now_ps &&
                available_bytes[h.pool] >= h.wire_bytes) ordered.push_back(i);
        }
        std::sort(ordered.begin(), ordered.end(), [&](size_t a, size_t b) {
            return heads[a].token < heads[b].token;
        });
        const size_t offset = ordered.empty() || context->policy == 0 ? 0 :
                              context->cursor % ordered.size();
        if (!ordered.empty()) std::rotate(ordered.begin(), ordered.begin() + offset, ordered.end());
        std::vector<uint64_t> remaining;
        if (pool_count) remaining.assign(available_bytes, available_bytes + pool_count);
        std::unordered_set<uint64_t> used_inputs, used_outputs;
        std::vector<size_t> selected;
        for (auto i : ordered) {
            const auto& h = heads[i];
            if (used_inputs.count(h.input_port) || used_outputs.count(h.output_port) ||
                remaining[h.pool] < h.wire_bytes) continue;
            remaining[h.pool] -= h.wire_bytes;
            used_inputs.insert(h.input_port);
            used_outputs.insert(h.output_port);
            selected.push_back(i);
        }
        // All allocating and throwing operations precede this commit point.
        for (size_t i = 0; i < selected.size(); ++i) {
            const auto index = selected[i];
            const auto& h = heads[index];
            context->inputs[h.input_port] = finishes[index];
            context->outputs[h.output_port] = finishes[index];
            grants[i] = {h.token, finishes[index]};
        }
        if (context->policy && !ordered.empty()) {
            context->cursor = (offset + std::max(size_t{1}, selected.size())) % ordered.size();
        }
        context->now = now_ps;
        *grant_count = selected.size();
        context->error.clear();
        return 0;
    } catch (const std::exception& exception) {
        context->error = exception.what();
        return -1;
    } catch (...) {
        context->error = "unknown native switch error";
        return -1;
    }
}
}
