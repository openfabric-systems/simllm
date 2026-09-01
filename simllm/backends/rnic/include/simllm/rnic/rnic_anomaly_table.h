#ifndef SIMLLM_RNIC_RNIC_ANOMALY_TABLE_H
#define SIMLLM_RNIC_RNIC_ANOMALY_TABLE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicAnomalyTableVersion = 1;

// How a row reaches the model.
//
// `Emergent` falls out of a modelled mechanism and is validated against its
// band. `Injected` is applied by rule because the mechanism is not public.
// `Fabric` is a property of the switch or link, reproduced by the packet
// simulator rather than by the endpoint. `Counter` is a facade behaviour with
// no datapath effect.
enum class AnomalyKind : std::uint8_t {
    Emergent,
    Injected,
    Fabric,
    Counter,
};

const char* toString(AnomalyKind kind) noexcept;

struct RnicAnomalyRow {
    // Stable identity, cited by tests and by the design document.
    const char* id;
    const char* name;
    const char* trigger;
    // The rendered "effect and magnitude" cell, verbatim.
    const char* effect;
    // The short machine-readable band handle a per-row test asserts against.
    // It is deliberately not rendered: the prose cell above is the document.
    const char* magnitude;
    AnomalyKind kind;
    // The rendered "kind" cell, verbatim. It always begins with the spelling
    // of `kind` and may add the mechanism that reproduces the row.
    const char* kind_text;
    const char* evidence;
};

inline constexpr std::size_t kRnicAnomalyRowCount = 15;

inline constexpr std::array<RnicAnomalyRow, kRnicAnomalyRowCount>
    kRnicAnomalyTable{{
        {"ANOM-01",
         "single UD QP receive cap",
         "one UD QP receiving above 3.07 Mpps",
         "excess dropped at the PHY, no sender-visible signal, 47.5 percent "
         "at 5.85 Mpps offered",
         "47.5 percent lost at 5.85 Mpps offered",
         AnomalyKind::Emergent,
         "emergent (receive processor)",
         "P3 seed 1"},
        {"ANOM-02",
         "two-SGE SEND sequence-error storm",
         "RC SEND with two gather entries at 512 B each, 32 QPs",
         "packet_seq_err of 68 k to 400 k per 30 s, 1-SGE control zero, "
         "goodput within 3 percent",
         "68000 to 400000 packet_seq_err per 30 s",
         AnomalyKind::Injected,
         "injected (per-packet drop rule keyed on SGE count and size)",
         "P3 seed 5/6"},
        {"ANOM-03",
         "saturated deep-pipeline loss equilibrium",
         "one RC QP, queue depth 1024, no inter-burst gap, above about "
         "92 Gb/s",
         "responder PHY discards plus go-back-N tail; goodput settles at 78 "
         "to 92 Gb/s",
         "78 to 92 Gb/s equilibrium goodput",
         AnomalyKind::Emergent,
         "emergent (ingress meter)",
         "P2, P3, P4"},
        {"ANOM-04",
         "drain window",
         "inter-burst gap of at least 4 us at 8 KiB, 4 to 100 us at 64 KiB",
         "discards go to zero; goodput follows bytes over (burst plus gap) "
         "within 0.1 percent; a 4 us gap raises 8 KiB goodput 13.8 percent",
         "13.8 percent gain at a 4 us gap and 8 KiB",
         AnomalyKind::Emergent,
         "emergent (ingress meter)",
         "P4"},
        {"ANOM-05",
         "in-NIC loopback starvation",
         "loopback and wire ingress together above 197 Gb/s",
         "wire keeps 99 percent, loopback drops to 51 percent, no PCIe stall",
         "99 percent wire share, 51 percent loopback share",
         AnomalyKind::Emergent,
         "emergent (internal arbiter)",
         "P3 seed 13"},
        {"ANOM-06",
         "ECT(0) stamping",
         "any RoCEv2 transmit",
         "ECN bits forced to ECT(0) regardless of requested ToS; DSCP "
         "honoured",
         "ECT(0) on every transmit",
         AnomalyKind::Injected,
         "injected (rate control stamp)",
         "P5b"},
        {"ANOM-07",
         "inert marking counter",
         "any traffic",
         "np_ecn_marked_roce_packets stays zero while CNPs are generated",
         "zero marks with nonzero CNPs",
         AnomalyKind::Counter,
         "counter",
         "P3, P5a, P5b"},
        {"ANOM-08",
         "one-hop pause",
         "receiver overload",
         "receiver emits global pause, no peer ever receives it",
         "zero pause frames received at any peer",
         AnomalyKind::Fabric,
         "fabric",
         "P3, audit"},
        {"ANOM-09",
         "firmware counter variant",
         "retransmission",
         "fw 16.31 counts local_ack_timeout_err, fw 16.32 counts zero",
         "fw 16.31 nonzero, fw 16.32 zero",
         AnomalyKind::Counter,
         "counter",
         "P5a"},
        {"ANOM-10",
         "bidirectional cleaner than unidirectional",
         "full duplex at 91.8 Gb/s per direction versus one direction at 93.4",
         "duplex counter-clean, unidirectional dirty",
         "91.8 Gb/s clean versus 93.4 Gb/s dirty",
         AnomalyKind::Emergent,
         "emergent (ingress meter)",
         "P4"},
        {"ANOM-11",
         "incast amplification",
         "N senders into one receiver, 1 MiB messages",
         "wire full, goodput tax equals loss rate times a go-back-N "
         "amplification factor (16x at 1.65 percent loss over 1 MiB messages, "
         "a 26.9 percent tax)",
         "16x amplification, 26.9 percent tax at 1.65 percent loss",
         AnomalyKind::Emergent,
         "emergent (packetizer plus go-back-N) with fabric loss from htsim",
         "P5a"},
        {"ANOM-12",
         "UD one-over-N delivery",
         "N UD senders into one receiver",
         "delivery exactly 1/N, no endpoint counter moves",
         "1/N delivery, 0.500 plus or minus 0.000 at N of 2",
         AnomalyKind::Fabric,
         "fabric",
         "P5a"},
        {"ANOM-13",
         "MTU tax",
         "MTU 1024 versus 4096",
         "5.6 percent goodput",
         "5.6 percent goodput tax",
         AnomalyKind::Emergent,
         "emergent (packetizer)",
         "P2"},
        {"ANOM-14",
         "host-bound message rate",
         "small messages, single process",
         "3.87 Mpps single QP at 1 KiB, 16.7 Mmsg/s per sender at 512 B",
         "3.87 Mpps per QP at 1 KiB",
         AnomalyKind::Emergent,
         "emergent (transmit pacer)",
         "P2, P5a"},
        {"ANOM-15",
         "memory-region and gather insensitivity",
         "12 000 MRs, 1024 MRs of 64 KiB, gathers except ANOM-02",
         "no throughput effect",
         "within 1.2 percent of the control",
         AnomalyKind::Emergent,
         "emergent by absence (no MR cache modelled; documented)",
         "P3 seeds 7/8"},
    }};

// Renders the table as the committed Markdown projection, including the
// generation banner and the kind legend. The bytes are the file, so a
// difference is a real drift rather than a formatting accident.
std::string renderRnicAnomalyTableMarkdown();

// The single table row, rendered exactly as it appears in both the projection
// and the design document. Exposed so a test can prove the design document
// still carries every row.
std::string renderRnicAnomalyRow(const RnicAnomalyRow& row);

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_ANOMALY_TABLE_H
