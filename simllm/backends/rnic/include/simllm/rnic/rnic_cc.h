#ifndef SIMLLM_RNIC_RNIC_CC_H
#define SIMLLM_RNIC_RNIC_CC_H

#include <cstdint>
#include <map>
#include <optional>
#include <utility>

#include "simllm/rnic/network_port.h"

namespace simllm::rnic {

inline constexpr std::uint32_t kRnicCcConfigVersion = 1;

// Alpha and the gain are carried as parts per million, so the whole reaction
// point is integer arithmetic and two runs of the same stimulus cut a rate to
// the same bit.
inline constexpr std::uint64_t kRnicCcAlphaScale = 1000000;

// The congestion-control block is two halves that never meet inside one
// endpoint: a notification point on the receive side and a reaction point on
// the transmit side. They meet on the wire, because the notification the
// receiver raises is addressed to the sender of the queue pair whose packet
// observed the congestion.
//
// The measured fabric marks nothing at all: zero congestion-experienced
// packets in 670 M, at two traffic classes, with the switch's egress buffer
// full and dropping. What does rise during reliable fan-in is the receiving
// NIC's own notification counter. So the notification point here sits at the
// endpoint's own ingress meter and a fabric that never marks is the normal
// case rather than a degenerate one.
struct RnicCcNotificationConfig {
    // Off is the identity default: the receive pipeline behaves exactly as it
    // did before this block existed and raises nothing.
    bool enabled{false};
    // Ingress occupancy, in bytes, at or above which an arriving packet is
    // treated as having observed congestion. Zero with the block enabled means
    // every packet observes congestion, which is a test configuration.
    std::uint64_t threshold_bytes{0};
    // Shortest gap between two notifications for one queue pair. Zero leaves
    // the notification point unlimited, which no hardware is.
    Picoseconds cnp_min_interval_ps{0};
};

// The per-queue-pair reaction point. Its state persists across work requests
// by construction: it belongs to the transmit pipeline, which outlives every
// individual request, and nothing in a request boundary touches it.
struct RnicCcReactionConfig {
    // Off is the identity default: the pacer keeps the fixed ceilings it had
    // and a notification is refused rather than silently dropped.
    bool enabled{false};
    // The rate a queue pair starts at and may never exceed.
    std::uint64_t ceiling_bps{0};
    // The rate a cut may never go below. Zero would stop the queue pair dead.
    std::uint64_t floor_bps{0};
    // Alpha at the first notification, and the gain of its recursion.
    std::uint64_t alpha_init_ppm{0};
    std::uint64_t alpha_gain_ppm{0};
    // The interval alpha decays over when no notification arrived in it.
    Picoseconds alpha_update_ps{0};
    // The additive increase: one step of bits per second per interval. The
    // campaign measured a recovery that is linear in time rather than the
    // fast-recovery-then-additive shape the vendor default implies, so this is
    // one step and one interval and not a schedule.
    std::uint64_t increase_step_bps{0};
    Picoseconds increase_interval_ps{0};
};

struct RnicCcConfig {
    std::uint32_t version{kRnicCcConfigVersion};
    RnicCcNotificationConfig notification;
    RnicCcReactionConfig reaction;
};

void validateRnicCcNotificationConfig(const RnicCcNotificationConfig& config);
void validateRnicCcReactionConfig(const RnicCcReactionConfig& config);

struct RnicCcNotificationCounters {
    // Notifications raised.
    std::uint64_t sent{0};
    // Packets that observed congestion while their queue pair's limiter was
    // still closed. This is the difference between what the meter saw and what
    // the wire carried, and it is why the measured notification rate is a
    // property of the limiter and not of the arrival rate.
    std::uint64_t suppressed{0};
    // Packets that observed congestion at all, whether or not one was raised.
    std::uint64_t observed{0};
};

// One notification point, shared by every queue pair the receiver terminates.
// The per-queue-pair limiter state is keyed by the sending endpoint and its
// queue pair, which is the pair a notification is addressed to.
class RnicCcNotificationPoint {
public:
    explicit RnicCcNotificationPoint(RnicCcNotificationConfig config);

    // Reports whether this arrival raises a notification for its queue pair.
    // `occupancy_bytes` is the ingress occupancy the packet observed, which is
    // the occupancy it would leave behind whether or not the meter keeps it: a
    // packet the meter discards is the strongest congestion evidence there is,
    // so it observes the same number an admitted one does.
    bool observe(
        std::uint32_t source,
        std::uint32_t qpn,
        std::uint64_t occupancy_bytes,
        Picoseconds now_ps);

    const RnicCcNotificationConfig& config() const noexcept;
    const RnicCcNotificationCounters& counters() const noexcept;

private:
    RnicCcNotificationConfig config_;
    // The instant each queue pair's limiter reopens.
    std::map<std::pair<std::uint32_t, std::uint32_t>, Picoseconds> reopen_ps_;
    RnicCcNotificationCounters counters_;
};

struct RnicCcReactionCounters {
    // Notifications the reaction point acted on. Every one it is handed is
    // acted on, which is why the ignored count below stays zero.
    std::uint64_t cnps_handled{0};
    // Notifications the reaction point refused. It refuses none: a reaction
    // point with no rate to cut is a configuration error and is rejected at
    // construction, not silently at run time. The counter exists because the
    // NIC exposes it and a detection tool reads it.
    std::uint64_t cnps_ignored{0};
    std::uint64_t rate_cuts{0};
    std::uint64_t rate_increases{0};
    std::uint64_t alpha_updates{0};
    // The lowest rate the queue pair ever held, so a study can see the depth
    // of a transient it only sampled the outside of.
    std::uint64_t min_rate_bps{0};
};

class RnicCcReactionPoint {
public:
    explicit RnicCcReactionPoint(RnicCcReactionConfig config);

    // Advances the alpha and increase timers to `now_ps`. Time must not
    // regress. A caller that steps without traffic keeps the rate honest by
    // calling this, and the transmit pipeline calls it on every release.
    void progress(Picoseconds now_ps);

    // One congestion notification for this queue pair. It advances the timers
    // first, so a notification and a timer tick at the same instant are
    // ordered the way the hardware orders them: the tick, then the cut.
    void onNotification(Picoseconds now_ps);

    // The next instant a timer fires. A caller that steps to it sees every
    // alpha update and every increase.
    std::optional<Picoseconds> nextEventTime() const;

    std::uint64_t rateBps() const noexcept;
    std::uint64_t alphaPpm() const noexcept;
    const RnicCcReactionConfig& config() const noexcept;
    const RnicCcReactionCounters& counters() const noexcept;
    void validateInvariants() const;

private:
    void decayAlpha();

    RnicCcReactionConfig config_;
    std::uint64_t rate_bps_{0};
    std::uint64_t alpha_ppm_{0};
    Picoseconds last_now_ps_{0};
    Picoseconds next_alpha_ps_{0};
    Picoseconds next_increase_ps_{0};
    // Whether a notification arrived inside the alpha interval now closing. An
    // interval that saw one has already raised alpha and must not also decay
    // it.
    bool notified_in_interval_{false};
    RnicCcReactionCounters counters_;
};

}  // namespace simllm::rnic

#endif  // SIMLLM_RNIC_RNIC_CC_H
