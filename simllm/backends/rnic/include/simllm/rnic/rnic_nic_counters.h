#ifndef SIMLLM_RNIC_RNIC_NIC_COUNTERS_H
#define SIMLLM_RNIC_RNIC_NIC_COUNTERS_H

#include <cstdint>

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicNicCountersVersion = 1;

// The observable-state facade. Every field is spelled the way the real NIC
// spells it in the driver's per-port hardware-counter directory and in the
// ethtool port statistics, because detection tooling reads those names and a
// golden model that renames them is not a reference for that tooling.
//
// Three groups of fields are deliberately inert, and each one reproduces a
// measured silicon behaviour rather than an omission:
//
//   * `np_ecn_marked_roce_packets` stays zero on both campaign firmwares while
//     CNPs are generated (ANOM-07). It is wired to nothing on purpose.
//   * `rx_pause_ctrl_phy` and `rx_global_pause` stay zero everywhere, over the
//     full lifetime of every campaign node: the switch never pauses a host
//     (ANOM-08). A receiver's own pause frames are counted, and no peer ever
//     sees them.
//   * `rx_out_of_buffer` and the two `outbound_pci_stalled_*` counters were
//     zero for the entire campaign, including under saturation. Receive
//     overflow lands on `rx_discards_phy`, not on a buffer or PCIe stall, and
//     these three are the negative controls that say so.
//
// `local_ack_timeout_err` is firmware dependent: firmware 16.31 counts a
// timeout-driven recovery and firmware 16.32 counts zero for the same
// stimulus (ANOM-09). The profile's `firmware_counter_variant` selects which.
struct RnicNicCounters {
    std::uint32_t version{kRnicNicCountersVersion};

    // Requester side.
    std::uint64_t packet_seq_err{0};
    std::uint64_t roce_adp_retrans{0};
    std::uint64_t roce_slow_restart_cnps{0};
    std::uint64_t local_ack_timeout_err{0};
    std::uint64_t rp_cnp_handled{0};
    std::uint64_t rp_cnp_ignored{0};

    // Responder side.
    std::uint64_t out_of_sequence{0};
    std::uint64_t duplicate_request{0};
    std::uint64_t rx_discards_phy{0};
    std::uint64_t rx_prio0_discards{0};
    std::uint64_t tx_pause_ctrl_phy{0};
    std::uint64_t tx_global_pause{0};
    std::uint64_t np_cnp_sent{0};
    std::uint64_t rx_write_requests{0};

    // Wire volume, both directions.
    std::uint64_t rx_packets_phy{0};
    std::uint64_t rx_bytes_phy{0};
    std::uint64_t tx_packets_phy{0};
    std::uint64_t tx_bytes_phy{0};

    // Inert by measurement, not by omission. See the comment above.
    std::uint64_t np_ecn_marked_roce_packets{0};
    std::uint64_t rx_pause_ctrl_phy{0};
    std::uint64_t rx_global_pause{0};
    std::uint64_t rx_out_of_buffer{0};
    std::uint64_t outbound_pci_stalled_rd{0};
    std::uint64_t outbound_pci_stalled_wr{0};
};

// True when no field of `later` is below the same field of `earlier`. A NIC
// counter never counts down, so a study treats a false here as a voided run.
bool rnicNicCountersMonotone(
    const RnicNicCounters& earlier,
    const RnicNicCounters& later) noexcept;

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_NIC_COUNTERS_H
