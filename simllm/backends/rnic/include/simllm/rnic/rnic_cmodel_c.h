#ifndef SIMLLM_RNIC_RNIC_CMODEL_C_H
#define SIMLLM_RNIC_RNIC_CMODEL_C_H

/*
 * C-linkage facade over the native RNIC golden model.
 *
 * This is the boundary an RTL testbench drives through DPI-C. Everything it
 * exposes is a plain struct of fixed-width integers with picosecond
 * timestamps, no exception ever crosses the boundary, and the caller owns the
 * clock exactly as the C++ device requires: deliver external events, then
 * progress, then poll.
 *
 * Determinism is a contract. The same stimulus sequence against the same
 * profile and configuration produces byte-identical traces, so a trace
 * recorded from a simulation run is the expected-result file for the
 * testbench and a divergence localizes to the first differing line.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIMLLM_RNIC_CM_ABI_VERSION 1u
#define SIMLLM_RNIC_CM_NAME_BYTES 32u

/* Status codes. Every entry point reports failure instead of throwing. */
enum {
    RNIC_CM_OK = 0,
    RNIC_CM_ERROR_ARGUMENT = 1,
    RNIC_CM_ERROR_STATE = 2,
    RNIC_CM_ERROR_SQ_FULL = 3,
    RNIC_CM_ERROR_UNSUPPORTED = 4,
    RNIC_CM_ERROR_INTERNAL = 5,
    RNIC_CM_NO_EVENT = 6
};

/* Completion status, mirroring the native CompletionStatus. */
enum {
    RNIC_CM_COMPLETION_SUCCESS = 0,
    RNIC_CM_COMPLETION_TRANSPORT_ERROR = 1,
    RNIC_CM_COMPLETION_NETWORK_REJECTED = 2
};

/* Packet kinds, mirroring the native NetworkPacketKind subset the transmit
 * path can emit. */
enum {
    RNIC_CM_PACKET_DATA = 0,
    RNIC_CM_PACKET_RETRANSMISSION = 1
};

/* Event kinds the caller may deliver. The packet kinds require an ABI v2
 * configuration; the extent kinds require an ABI v1 one. The control kinds
 * are declared here so the vocabulary is stable, and are refused with
 * RNIC_CM_ERROR_UNSUPPORTED until the rate-control block lands. */
enum {
    RNIC_CM_EVENT_EXTENT_DELIVERED = 1,
    RNIC_CM_EVENT_EXTENT_DROPPED = 2,
    RNIC_CM_EVENT_PACKET_TX_FINISHED = 3,
    RNIC_CM_EVENT_PACKET_RX_ARRIVED = 4,
    RNIC_CM_EVENT_PACKET_DELIVERED = 5,
    RNIC_CM_EVENT_PACKET_DROPPED = 6,
    RNIC_CM_EVENT_ECN_MARKED = 7,
    RNIC_CM_EVENT_CNP_RECEIVED = 8,
    RNIC_CM_EVENT_PFC_PAUSED = 9,
    RNIC_CM_EVENT_PFC_RESUMED = 10,
    RNIC_CM_EVENT_RATE_UPDATED = 11
};

/* Drop evidence, mirroring the native DropLocation and DropReason. A dropped
 * event must carry both. */
enum {
    RNIC_CM_DROP_LOCATION_NONE = 0,
    RNIC_CM_DROP_LOCATION_TX_PORT = 1,
    RNIC_CM_DROP_LOCATION_FABRIC = 2,
    RNIC_CM_DROP_LOCATION_RX_PORT = 3
};

enum {
    RNIC_CM_DROP_REASON_NONE = 0,
    RNIC_CM_DROP_REASON_INJECTED = 1,
    RNIC_CM_DROP_REASON_QUEUE_OVERFLOW = 2,
    RNIC_CM_DROP_REASON_LINK_DOWN = 3,
    RNIC_CM_DROP_REASON_POLICY_REJECTED = 4
};

typedef struct rnic_cm_device rnic_cm_device;

/*
 * The hardware parameter set. This is the value view of the native
 * RnicHwProfile: the evidence class of each field lives on the native record,
 * which the facade recovers when the supplied values match a preset exactly.
 * A parameter set that does not match a preset is adopted as declared, which
 * is what a caller-chosen value is.
 */
typedef struct rnic_cm_profile {
    uint32_t version;
    uint32_t derived_link_factor;
    char name[SIMLLM_RNIC_CM_NAME_BYTES];
    char derived_from[SIMLLM_RNIC_CM_NAME_BYTES];

    uint64_t link_bps;
    uint64_t goodput_bps;
    uint64_t mtu_bytes;
    uint64_t wire_header_bytes;

    uint64_t t_eff_ps;
    uint64_t wire_round_trip_floor_ps;
    uint64_t doorbell_service_ps;
    uint64_t wqe_fetch_service_ps;
    uint64_t qpc_lookup_service_ps;
    uint64_t scheduler_service_ps;
    uint64_t cqe_write_service_ps;

    uint64_t sq_depth;
    uint64_t max_inflight_bytes;
    uint64_t max_inflight_packets;

    uint64_t tx_pps_per_qp;
    uint64_t tx_pps_per_nic;
    uint64_t rx_pps_per_qp_rc;
    uint64_t rx_pps_per_qp_ud;
    uint64_t rx_pps_per_nic;

    uint64_t rx_ingress_bytes;
    uint64_t rx_drain_bps;
    uint64_t internal_budget_bps;
    uint64_t rto_ps;
    uint64_t cnp_min_interval_ps;
    uint64_t dcqcn_alpha_update_ps;
    uint64_t dcqcn_rate_reduce_ps;
    uint64_t dcqcn_byte_reset;
    uint64_t dcqcn_rate_step_bps;
    uint64_t selective_repeat_window;

    uint8_t loopback_priority;
    uint8_t recovery;      /* 0 go-back-N, 1 selective repeat */
    uint8_t ack_coalescing;
    uint8_t dcqcn_enabled;
    uint8_t ecn_stamp;     /* 0 not-ECT, 1 ECT(0), 2 ECT(1) */
    uint8_t pfc_enabled;
    uint8_t global_pause_tx;
    uint8_t pause_propagates;
    uint8_t firmware_counter_variant; /* 0 fw 16.32, 1 fw 16.31 */
    uint8_t reserved[7];
} rnic_cm_profile;

typedef struct rnic_cm_config {
    uint32_t version;
    uint32_t qpn;
    uint32_t source;
    uint32_t reserved0;
    uint64_t policy_context_token;
    uint64_t sq_depth;
    uint64_t cq_depth;
    /* 0 selects the flow-extent port (network ABI v1). 1 selects the
     * transmit pipeline and its per-packet port (network ABI v2). */
    uint8_t packetization;
    uint8_t trace_enabled;
    uint8_t reserved1[6];
    /* 0 means the send queue itself is the bound. */
    uint64_t max_inflight_wqes;
    /* 0 means no byte bound. */
    uint64_t max_inflight_bytes;
} rnic_cm_config;

typedef struct rnic_cm_wqe {
    uint64_t wr_id;
    uint64_t flow_id;
    uint64_t payload_bytes;
    uint32_t flow_tag;
    uint32_t destination;
    uint32_t sge_count;
    uint8_t traffic_class;
    uint8_t opcode;   /* 0 send */
    uint8_t signaled;
    uint8_t reserved[5];
} rnic_cm_wqe;

typedef struct rnic_cm_doorbell_batch {
    uint64_t batch_id;
    uint64_t wqe_count;
    uint64_t rung_at_ps;
    uint64_t observed_at_ps;
} rnic_cm_doorbell_batch;

typedef struct rnic_cm_packet {
    uint64_t token;
    uint64_t wqe_id;
    uint64_t wr_id;
    uint64_t payload_offset_bytes;
    uint64_t payload_bytes;
    uint64_t wire_bytes;
    uint64_t issued_at_ps;
    uint32_t qpn;
    uint32_t destination;
    uint32_t psn;
    uint32_t packet_index;
    uint32_t packet_count;
    uint8_t kind;
    uint8_t traffic_class;
    uint8_t reserved[2];
} rnic_cm_packet;

typedef struct rnic_cm_event_info {
    uint32_t kind;
    uint32_t drop_location;
    uint32_t drop_reason;
    uint32_t ecn_marked;
    uint64_t token;
} rnic_cm_event_info;

typedef struct rnic_cm_cqe {
    uint64_t cqe_sequence;
    uint64_t wr_id;
    uint64_t wqe_id;
    uint64_t sq_sequence;
    uint64_t byte_count;
    uint64_t visible_at_ps;
    uint64_t polled_at_ps;
    uint32_t qpn;
    uint32_t status;
    uint32_t opcode;
    uint32_t valid_fields;
} rnic_cm_cqe;

typedef struct rnic_cm_counter_set {
    uint32_t version;
    uint32_t reserved0;
    uint64_t posted_wqes;
    uint64_t sq_full_rejections;
    uint64_t doorbells;
    uint64_t doorbelled_wqes;
    uint64_t network_submit_attempts;
    uint64_t network_accepted;
    uint64_t network_busy;
    uint64_t network_rejected;
    uint64_t network_delivered;
    uint64_t network_dropped;
    uint64_t cqes_visible;
    uint64_t cqes_polled;
    uint64_t cq_overruns;
    uint64_t sq_reclaimed_wqes;
    uint64_t sq_high_watermark;
    uint64_t cq_high_watermark;
    /* Transmit-pipeline counters. They stay zero without packetization. */
    uint64_t tx_packets;
    uint64_t tx_payload_bytes;
    uint64_t tx_wire_bytes;
    uint64_t tx_window_stalls;
    uint64_t tx_pacer_stalls;
    uint64_t tx_inflight_wqes;
    uint64_t tx_inflight_bytes;
} rnic_cm_counter_set;

/* Fills `out` with a named preset ("cx5_100g" or "cx7_400g"). */
int rnic_cm_profile_preset(const char* name, rnic_cm_profile* out);

/* Writes the profile's canonical record hash as a NUL-terminated lowercase
 * hex string. `bytes` must be at least 65. */
int rnic_cm_profile_sha256(
    const rnic_cm_profile* profile,
    char* out,
    size_t bytes);

/* Constructs one endpoint. Returns NULL on an invalid profile or config. */
rnic_cm_device* rnic_cm_create(
    const rnic_cm_profile* profile,
    const rnic_cm_config* config);

int rnic_cm_post(
    rnic_cm_device* device,
    const rnic_cm_wqe* wqe,
    uint64_t now_ps,
    uint64_t* out_wqe_id);

int rnic_cm_doorbell(
    rnic_cm_device* device,
    uint64_t now_ps,
    rnic_cm_doorbell_batch* out_batch);

/* Delivers one wire packet to the receive side. The receive pipeline is not
 * landed, so this is refused with RNIC_CM_ERROR_UNSUPPORTED rather than
 * silently ignored. */
int rnic_cm_rx_packet(
    rnic_cm_device* device,
    const rnic_cm_packet* packet,
    uint64_t now_ps);

int rnic_cm_event(
    rnic_cm_device* device,
    const rnic_cm_event_info* event,
    uint64_t now_ps);

int rnic_cm_progress(
    rnic_cm_device* device,
    uint64_t now_ps,
    uint64_t* out_changes);

/* Reports the next internally scheduled time so a testbench can step exactly.
 * Returns RNIC_CM_NO_EVENT when the endpoint has nothing scheduled. */
int rnic_cm_next_event_ps(rnic_cm_device* device, uint64_t* out_now_ps);

int rnic_cm_poll(
    rnic_cm_device* device,
    rnic_cm_cqe* out,
    size_t max_entries,
    uint64_t now_ps,
    size_t* out_count);

/* Drains the packets the endpoint has emitted since the last call. Without
 * packetization one admitted flow extent is reported as one attempt. */
int rnic_cm_tx_next(
    rnic_cm_device* device,
    rnic_cm_packet* out,
    size_t max_packets,
    size_t* out_count);

int rnic_cm_counters(rnic_cm_device* device, rnic_cm_counter_set* out);

/* Writes the transaction trace: one line per stimulus and per observed
 * transition, each stamped in picoseconds. */
int rnic_cm_trace(rnic_cm_device* device, const char* path);

void rnic_cm_destroy(rnic_cm_device* device);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SIMLLM_RNIC_RNIC_CMODEL_C_H */
