#ifndef SIMLLM_NVSWITCH_C_H
#define SIMLLM_NVSWITCH_C_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define SIMLLM_NVS_API __declspec(dllexport)
#else
#define SIMLLM_NVS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The caller owns VOQs, bytes, credits and the event calendar. This context
 * alone owns crossbar port occupancy and the selected arbitration cursor.
 * Ports on different chips cannot be joined. -1 denotes a non-switch port.
 * Policy 0 is deterministic identity; policy 1 rotates the candidate order.
 * Neither policy is a claim about NVIDIA's deployed arbitration algorithm. */
typedef struct simllm_nvs_context simllm_nvs_context;
typedef struct {
    uint64_t token, input_port, output_port, pool, wire_bytes, link_rate_bps;
} simllm_nvs_head;
typedef struct {
    uint64_t token, finished_at_ps;
} simllm_nvs_grant;

SIMLLM_NVS_API uint32_t simllm_nvs_abi_version(void);
SIMLLM_NVS_API simllm_nvs_context* simllm_nvs_create(
    const int32_t* port_chips, size_t port_count, uint32_t policy);
SIMLLM_NVS_API void simllm_nvs_destroy(simllm_nvs_context* context);
SIMLLM_NVS_API const char* simllm_nvs_error(const simllm_nvs_context* context);
/* One atomic grant interval. Inputs are read-only snapshots of legal VOQ
 * heads and available shared receiver bytes. Success commits all grants.
 * Error returns -1 with every timing/cursor state and caller output unchanged.
 * Output storage must have capacity >= head_count. Times are integer ps. */
SIMLLM_NVS_API int simllm_nvs_plan(
    simllm_nvs_context* context, uint64_t now_ps, uint64_t crossbar_bytes_per_second,
    const simllm_nvs_head* heads, size_t head_count,
    const uint64_t* available_bytes, size_t pool_count,
    simllm_nvs_grant* grants, size_t grant_capacity, size_t* grant_count);

#ifdef __cplusplus
}
#endif
#endif
