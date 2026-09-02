#include "simllm/rnic/rnic_cc.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace simllm::rnic {
namespace {

// A decay catch-up longer than this is treated as a complete decay. With any
// gain the candidate grid carries, alpha after this many intervals is already
// below one part per million, so the cap is a bound on work and not on
// behaviour.
constexpr std::uint64_t kMaxAlphaCatchUp = 65536;

}  // namespace

void validateRnicCcNotificationConfig(
    const RnicCcNotificationConfig& config) {
    if (!config.enabled) {
        throw std::invalid_argument(
            "RNIC congestion notification config is not enabled");
    }
}

void validateRnicCcReactionConfig(const RnicCcReactionConfig& config) {
    if (!config.enabled) {
        throw std::invalid_argument(
            "RNIC congestion reaction config is not enabled");
    }
    if (config.ceiling_bps == 0) {
        throw std::invalid_argument(
            "RNIC congestion reaction point needs a rate ceiling");
    }
    if (config.floor_bps == 0 || config.floor_bps > config.ceiling_bps) {
        throw std::invalid_argument(
            "RNIC congestion reaction floor must be positive and at or below "
            "the ceiling");
    }
    if (config.alpha_init_ppm > kRnicCcAlphaScale
        || config.alpha_gain_ppm == 0
        || config.alpha_gain_ppm > kRnicCcAlphaScale) {
        throw std::invalid_argument(
            "RNIC congestion reaction alpha must be a fraction and its gain a "
            "positive one");
    }
    if (config.alpha_update_ps == 0) {
        throw std::invalid_argument(
            "RNIC congestion reaction point needs an alpha update interval");
    }
    if (config.increase_interval_ps == 0 || config.increase_step_bps == 0) {
        throw std::invalid_argument(
            "RNIC congestion reaction point needs an additive increase");
    }
}

RnicCcNotificationPoint::RnicCcNotificationPoint(
    RnicCcNotificationConfig config)
    : config_(std::move(config)) {
    validateRnicCcNotificationConfig(config_);
}

bool RnicCcNotificationPoint::observe(
    std::uint32_t source,
    std::uint32_t qpn,
    std::uint64_t occupancy_bytes,
    Picoseconds now_ps) {
    if (occupancy_bytes < config_.threshold_bytes) {
        return false;
    }
    ++counters_.observed;
    const auto key = std::make_pair(source, qpn);
    const auto found = reopen_ps_.find(key);
    if (found != reopen_ps_.end() && now_ps < found->second) {
        ++counters_.suppressed;
        return false;
    }
    if (config_.cnp_min_interval_ps
        > std::numeric_limits<Picoseconds>::max() - now_ps) {
        throw std::overflow_error(
            "RNIC congestion notification timestamp overflow");
    }
    reopen_ps_[key] = now_ps + config_.cnp_min_interval_ps;
    ++counters_.sent;
    return true;
}

const RnicCcNotificationConfig&
RnicCcNotificationPoint::config() const noexcept {
    return config_;
}

const RnicCcNotificationCounters&
RnicCcNotificationPoint::counters() const noexcept {
    return counters_;
}

RnicCcReactionPoint::RnicCcReactionPoint(RnicCcReactionConfig config)
    : config_(std::move(config)) {
    validateRnicCcReactionConfig(config_);
    rate_bps_ = config_.ceiling_bps;
    alpha_ppm_ = config_.alpha_init_ppm;
    next_alpha_ps_ = config_.alpha_update_ps;
    next_increase_ps_ = config_.increase_interval_ps;
    counters_.min_rate_bps = rate_bps_;
}

void RnicCcReactionPoint::decayAlpha() {
    ++counters_.alpha_updates;
    const std::uint64_t step =
        alpha_ppm_ * config_.alpha_gain_ppm / kRnicCcAlphaScale;
    if (step == 0) {
        // Integer truncation would park alpha on a small nonzero floor and a
        // queue pair would keep cutting forever at a rate no measurement
        // supports. One part per million per interval is what carries it the
        // rest of the way down.
        alpha_ppm_ = alpha_ppm_ == 0 ? 0 : alpha_ppm_ - 1;
        return;
    }
    alpha_ppm_ -= step;
}

void RnicCcReactionPoint::progress(Picoseconds now_ps) {
    if (now_ps < last_now_ps_) {
        throw std::logic_error("RNIC congestion reaction point time regressed");
    }
    last_now_ps_ = now_ps;

    if (next_alpha_ps_ <= now_ps) {
        const std::uint64_t elapsed = now_ps - next_alpha_ps_;
        const std::uint64_t ticks = elapsed / config_.alpha_update_ps + 1;
        // The interval that is closing now is the one the notification flag
        // belongs to; every later interval is silent by definition.
        std::uint64_t decays = ticks;
        if (notified_in_interval_) {
            notified_in_interval_ = false;
            --decays;
            ++counters_.alpha_updates;
        }
        if (decays > kMaxAlphaCatchUp) {
            counters_.alpha_updates += decays - kMaxAlphaCatchUp;
            alpha_ppm_ = 0;
            decays = kMaxAlphaCatchUp;
        }
        for (std::uint64_t step = 0; step < decays; ++step) {
            decayAlpha();
        }
        next_alpha_ps_ += ticks * config_.alpha_update_ps;
    }

    if (next_increase_ps_ <= now_ps) {
        const std::uint64_t elapsed = now_ps - next_increase_ps_;
        const std::uint64_t ticks = elapsed / config_.increase_interval_ps + 1;
        counters_.rate_increases += ticks;
        const std::uint64_t headroom = config_.ceiling_bps - rate_bps_;
        const std::uint64_t gain =
            ticks > headroom / config_.increase_step_bps
            ? headroom
            : ticks * config_.increase_step_bps;
        rate_bps_ += std::min(gain, headroom);
        next_increase_ps_ += ticks * config_.increase_interval_ps;
    }
}

void RnicCcReactionPoint::onNotification(Picoseconds now_ps) {
    progress(now_ps);
    ++counters_.cnps_handled;
    notified_in_interval_ = true;
    // alpha <- (1 - g) alpha + g, in parts per million.
    alpha_ppm_ += (kRnicCcAlphaScale - alpha_ppm_) * config_.alpha_gain_ppm
        / kRnicCcAlphaScale;
    const std::uint64_t cut = rate_bps_ * alpha_ppm_ / (2 * kRnicCcAlphaScale);
    rate_bps_ = cut >= rate_bps_ ? config_.floor_bps
                                 : std::max(rate_bps_ - cut, config_.floor_bps);
    ++counters_.rate_cuts;
    counters_.min_rate_bps = std::min(counters_.min_rate_bps, rate_bps_);
}

std::optional<Picoseconds> RnicCcReactionPoint::nextEventTime() const {
    return std::min(next_alpha_ps_, next_increase_ps_);
}

std::uint64_t RnicCcReactionPoint::rateBps() const noexcept {
    return rate_bps_;
}

std::uint64_t RnicCcReactionPoint::alphaPpm() const noexcept {
    return alpha_ppm_;
}

const RnicCcReactionConfig& RnicCcReactionPoint::config() const noexcept {
    return config_;
}

const RnicCcReactionCounters&
RnicCcReactionPoint::counters() const noexcept {
    return counters_;
}

void RnicCcReactionPoint::validateInvariants() const {
    if (rate_bps_ < config_.floor_bps || rate_bps_ > config_.ceiling_bps) {
        throw std::logic_error(
            "RNIC congestion reaction rate left its configured interval");
    }
    if (alpha_ppm_ > kRnicCcAlphaScale) {
        throw std::logic_error("RNIC congestion reaction alpha left its range");
    }
    if (counters_.cnps_ignored != 0) {
        throw std::logic_error(
            "RNIC congestion reaction point ignored a notification");
    }
}

}  // namespace simllm::rnic
