// Native checks for the congestion-control block: the notification point at
// the receiving endpoint's ingress meter, the per-queue-pair reaction point in
// front of the transmit pacer, and the tail-drop egress queue the test fabric
// needs before either of them can be exercised.
//
// Everything here is a mechanism check. The measured bands live in the slice-D
// study, because a band is a claim about silicon and a test is a claim about
// the model.

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fake_network.h"
#include "simllm/rnic/rnic_cc.h"
#include "simllm/rnic/rnic_cmodel_c.h"
#include "simllm/rnic/rnic_hw_profile.h"
#include "simllm/rnic/rnic_rx_pipeline.h"

namespace {

using simllm::rnic::Picoseconds;
using simllm::rnic::RnicCcNotificationConfig;
using simllm::rnic::RnicCcNotificationPoint;
using simllm::rnic::RnicCcReactionConfig;
using simllm::rnic::RnicCcReactionPoint;
using simllm::rnic::RnicRxPacket;
using simllm::rnic::RnicRxPipeline;
using simllm::rnic::RnicRxPipelineConfig;
using simllm::rnic::RnicRxResult;
using simllm::rnic::kRnicCcAlphaScale;
using simllm::rnic::testing::FakeEgressQueue;
using simllm::rnic::testing::FakeEgressQueueConfig;

class TestRunner {
public:
    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template <typename Expected, typename Callable>
    void expectThrowAs(Callable&& callable, const std::string& message) {
        try {
            callable();
            check(false, message);
        } catch (const Expected&) {
            check(true, message);
        } catch (const std::exception& error) {
            check(false, message + "; wrong exception: " + error.what());
        } catch (...) {
            check(false, message + "; wrong non-standard exception");
        }
    }

    int failures() const noexcept { return failures_; }

private:
    int failures_{0};
};

RnicCcReactionConfig reactionConfig() {
    RnicCcReactionConfig config;
    config.enabled = true;
    config.ceiling_bps = 100000000000ULL;
    config.floor_bps = 1000000000ULL;
    config.alpha_init_ppm = 400000;
    config.alpha_gain_ppm = 62500;
    config.alpha_update_ps = 1000000000ULL;
    config.increase_step_bps = 100000000ULL;
    config.increase_interval_ps = 1000000000ULL;
    return config;
}

void testNotificationPoint(TestRunner& test) {
    RnicCcNotificationConfig config;
    config.enabled = true;
    config.threshold_bytes = 1000;
    config.cnp_min_interval_ps = 1000000;

    RnicCcNotificationPoint point(config);
    test.check(
        !point.observe(2, 7, 999, 0),
        "an occupancy below the threshold observes no congestion");
    test.check(
        point.counters().observed == 0,
        "an occupancy below the threshold is not even an observation");
    test.check(
        point.observe(2, 7, 1000, 0),
        "an occupancy at the threshold raises one notification");
    test.check(
        !point.observe(2, 7, 5000, 999999),
        "the per-queue-pair limiter suppresses a second notification inside "
        "its interval");
    test.check(
        point.counters().suppressed == 1,
        "a suppressed observation is counted rather than lost");
    test.check(
        point.observe(2, 7, 5000, 1000000),
        "the limiter reopens exactly at its interval");
    test.check(
        point.observe(3, 7, 5000, 1000001),
        "the limiter is per sending endpoint, not per receiver");
    test.check(
        point.observe(2, 8, 5000, 1000001),
        "the limiter is per queue pair, not per endpoint");
    test.check(
        point.counters().sent == 4 && point.counters().observed == 5,
        "the notification point counts what it saw and what it raised");

    RnicCcNotificationConfig off;
    test.expectThrowAs<std::invalid_argument>(
        [&off]() { RnicCcNotificationPoint disabled(off); },
        "a disabled notification config is refused rather than adopted");
}

void testReactionPointCut(TestRunner& test) {
    RnicCcReactionPoint point(reactionConfig());
    test.check(
        point.rateBps() == 100000000000ULL,
        "a reaction point starts at its ceiling");

    // alpha <- (1 - g) alpha + g with alpha 0.4 and g 0.0625 is 0.4375, so the
    // first cut is 21.875 percent of the ceiling.
    point.onNotification(1000);
    const std::uint64_t expected_alpha = 400000 + (kRnicCcAlphaScale - 400000)
        * 62500 / kRnicCcAlphaScale;
    test.check(
        point.alphaPpm() == expected_alpha,
        "the alpha recursion runs on the notification");
    const std::uint64_t expected_rate = 100000000000ULL
        - 100000000000ULL * expected_alpha / (2 * kRnicCcAlphaScale);
    test.check(
        point.rateBps() == expected_rate,
        "the cut is alpha over two of the standing rate");
    test.check(
        point.counters().cnps_handled == 1
            && point.counters().cnps_ignored == 0,
        "every notification is handled and none is ignored");

    // Enough notifications to drive the rate into the floor, which it must not
    // cross.
    for (int index = 0; index < 4000; ++index) {
        point.onNotification(1000);
    }
    test.check(
        point.rateBps() == 1000000000ULL,
        "the rate stops at the configured floor");
    point.validateInvariants();
}

void testReactionPointRecovery(TestRunner& test) {
    RnicCcReactionConfig config = reactionConfig();
    RnicCcReactionPoint point(config);
    for (int index = 0; index < 4000; ++index) {
        point.onNotification(1000);
    }
    test.check(
        point.rateBps() == config.floor_bps,
        "the transient drove the rate to the floor");

    // Ten increase intervals with no notification: purely additive, so the
    // rate is the floor plus ten steps and alpha has decayed.
    const std::uint64_t alpha_after_transient = point.alphaPpm();
    point.progress(1000 + 10 * config.increase_interval_ps);
    test.check(
        point.rateBps()
            == config.floor_bps + 10 * config.increase_step_bps,
        "recovery is additive, one step per interval");
    test.check(
        point.alphaPpm() < alpha_after_transient,
        "alpha decays across intervals with no notification");

    // Far enough forward that the ceiling clamps rather than overshoots.
    point.progress(1000 + 100000 * config.increase_interval_ps);
    test.check(
        point.rateBps() == config.ceiling_bps,
        "recovery stops at the ceiling");
    point.validateInvariants();

    RnicCcReactionPoint monotone(config);
    monotone.progress(config.increase_interval_ps);
    test.expectThrowAs<std::logic_error>(
        [&monotone]() { monotone.progress(0); },
        "a reaction point refuses a clock that regressed");
}

void testEgressQueue(TestRunner& test) {
    FakeEgressQueueConfig config;
    config.link_bps = 100000000000ULL;
    config.capacity_bytes = 8192;
    FakeEgressQueue queue(config);

    test.check(
        queue.offer(0, 4096).has_value(),
        "an empty tail-drop queue admits a packet");
    test.check(
        queue.offer(0, 4096).has_value(),
        "the queue admits up to its capacity");
    test.check(
        !queue.offer(0, 4096).has_value(),
        "the queue tail-drops what does not fit, with no signal of any kind");
    test.check(
        queue.droppedCount() == 1 && queue.admittedCount() == 2,
        "the queue counts the drop separately from the admission");
    test.check(
        queue.offeredBytes()
            == queue.admittedBytes() + queue.droppedBytes(),
        "the egress queue conserves bytes");

    // 8192 bytes at 100 Gb/s is 655.36 ns, so the queue is empty again after
    // it and admits the next packet immediately.
    test.check(
        queue.occupancyBytes(1000000) == 0,
        "the queue drains at the link rate");
    test.check(
        queue.offer(1000000, 4096).has_value(),
        "a drained queue admits again");

    FakeEgressQueueConfig unbounded;
    unbounded.capacity_bytes = 0;
    FakeEgressQueue open(unbounded);
    for (int index = 0; index < 1000; ++index) {
        test.check(
            open.offer(0, 4096).has_value(),
            "an unbounded queue never drops");
    }
    test.check(
        open.droppedCount() == 0,
        "an unbounded queue is the slice-C fabric, which drops nothing");
}

RnicRxPipelineConfig receiveConfig(bool notify) {
    RnicRxPipelineConfig config;
    config.enabled = true;
    config.ingress_bytes = 262016;
    config.drain_bps = 96600000000ULL;
    if (notify) {
        config.notification.enabled = true;
        config.notification.threshold_bytes = 131008;
        config.notification.cnp_min_interval_ps = 1000000;
    }
    return config;
}

RnicRxPacket dataPacket(std::uint32_t psn) {
    RnicRxPacket packet;
    packet.qpn = 5;
    packet.source = 2;
    packet.psn = psn;
    packet.payload_bytes = 4096;
    packet.wire_bytes = 4160;
    return packet;
}

void testReceiveIntegration(TestRunner& test) {
    // Arrivals with no drain time between them, so the meter fills and the
    // notification point starts observing.
    RnicRxPipeline notifying(receiveConfig(true));
    RnicRxPipeline quiet(receiveConfig(false));
    std::uint64_t notifications = 0;
    for (std::uint32_t psn = 0; psn < 200; ++psn) {
        const RnicRxResult noisy = notifying.onPacket(dataPacket(psn), 0);
        const RnicRxResult silent = quiet.onPacket(dataPacket(psn), 0);
        if (noisy.has_cnp) {
            ++notifications;
            test.check(
                noisy.cnp_source == 2 && noisy.cnp_qpn == 5,
                "a notification names the queue pair whose packet observed "
                "the congestion");
        }
        test.check(
            noisy.outcome == silent.outcome,
            "the notification point does not change what the meter does");
    }
    test.check(
        notifications > 0,
        "a meter past its threshold raises notifications");
    test.check(
        notifying.nicCounters().np_cnp_sent == notifications,
        "the NIC-named notification counter tracks what was raised");
    test.check(
        notifying.nicCounters().np_ecn_marked_roce_packets == 0,
        "the marking counter stays inert while notifications are generated");
    test.check(
        quiet.nicCounters().np_cnp_sent == 0,
        "a pipeline with no notification point raises nothing");
    test.check(
        notifying.counters().packets_discarded_meter
            == quiet.counters().packets_discarded_meter,
        "the identity-off receive path discards exactly what it did before");

    // With the limiter open every observing packet notifies, which is how a
    // packet the meter is about to throw away gets to say so.
    RnicRxPipelineConfig unlimited = receiveConfig(true);
    unlimited.notification.cnp_min_interval_ps = 0;
    RnicRxPipeline loud(unlimited);
    bool loud_notified_on_discard = false;
    for (std::uint32_t psn = 0; psn < 200; ++psn) {
        const RnicRxResult verdict = loud.onPacket(dataPacket(psn), 0);
        if (verdict.has_cnp
            && verdict.outcome
                == simllm::rnic::RnicRxOutcome::DiscardedSilently) {
            loud_notified_on_discard = true;
        }
    }
    test.check(
        loud_notified_on_discard,
        "a packet the meter throws away still observes the congestion it "
        "arrived into");

    RnicRxPipeline empty(receiveConfig(true));
    const RnicRxResult first = empty.onPacket(dataPacket(0), 0);
    test.check(
        !first.has_cnp,
        "an empty meter observes no congestion, so an uncontended flow "
        "generates nothing at all");

    RnicRxPipelineConfig impossible = receiveConfig(true);
    impossible.notification.threshold_bytes = impossible.ingress_bytes + 1;
    test.expectThrowAs<std::invalid_argument>(
        [&impossible]() { RnicRxPipeline refused(impossible); },
        "a threshold above the ingress buffer is refused rather than never "
        "reached");
}

rnic_cm_device* makeEndpoint(bool congestion_control) {
    rnic_cm_profile profile;
    if (rnic_cm_profile_preset("cx5_100g", &profile) != RNIC_CM_OK) {
        throw std::runtime_error("no cx5_100g preset");
    }
    rnic_cm_config config;
    std::memset(&config, 0, sizeof(config));
    config.version = SIMLLM_RNIC_CM_ABI_VERSION;
    config.qpn = 4;
    config.source = 4;
    config.policy_context_token = 4;
    config.sq_depth = 32;
    config.cq_depth = 64;
    config.packetization = 1;
    config.receive = 1;
    config.congestion_control = congestion_control ? 1u : 0u;
    config.max_inflight_wqes = 32;
    return rnic_cm_create(&profile, &config);
}

void testFacade(TestRunner& test) {
    rnic_cm_profile profile;
    test.check(
        rnic_cm_profile_preset("cx5_100g", &profile) == RNIC_CM_OK
            && profile.np_cnp_threshold_bytes != 0
            && profile.dcqcn_alpha_gain_ppm != 0
            && profile.dcqcn_rate_increase_step_bps != 0
            && profile.dcqcn_rate_floor_bps != 0,
        "the preset carries the congestion-control parameters");

    rnic_cm_device* controlled = makeEndpoint(true);
    test.check(controlled != nullptr, "an endpoint with rate control builds");
    rnic_cm_device* plain = makeEndpoint(false);
    test.check(plain != nullptr, "an endpoint without rate control builds");
    if (controlled == nullptr || plain == nullptr) {
        rnic_cm_destroy(controlled);
        rnic_cm_destroy(plain);
        return;
    }

    rnic_cm_event_info notice;
    std::memset(&notice, 0, sizeof(notice));
    notice.kind = RNIC_CM_EVENT_CNP_RECEIVED;
    test.check(
        rnic_cm_event(plain, &notice, 0) == RNIC_CM_ERROR_UNSUPPORTED,
        "an endpoint without a reaction point refuses a notification rather "
        "than absorbing it");

    rnic_cm_nic_counter_set before;
    rnic_cm_nic_counter_set after;
    test.check(
        rnic_cm_nic_counters(controlled, &before) == RNIC_CM_OK,
        "the counter facade reads");
    test.check(
        rnic_cm_event(controlled, &notice, 1000) == RNIC_CM_OK,
        "an endpoint with a reaction point accepts a notification");
    test.check(
        rnic_cm_nic_counters(controlled, &after) == RNIC_CM_OK,
        "the counter facade reads after the notification");
    test.check(
        after.rp_cnp_handled == before.rp_cnp_handled + 1
            && after.rp_cnp_ignored == 0,
        "the facade reports the notification handled and none ignored");
    test.check(
        after.rp_current_rate_bps < before.rp_current_rate_bps
            && after.rp_rate_cuts == 1,
        "the notification cut the reaction point's rate");
    test.check(
        after.np_ecn_marked_roce_packets == 0,
        "the marking counter stays inert at the facade too");

    // A marking event stays refused: the measured fabric never marks, and a
    // model that accepted a mark would let a study invent one.
    rnic_cm_event_info mark;
    std::memset(&mark, 0, sizeof(mark));
    mark.kind = RNIC_CM_EVENT_ECN_MARKED;
    test.check(
        rnic_cm_event(controlled, &mark, 2000) == RNIC_CM_ERROR_UNSUPPORTED,
        "a switch mark is still refused, because no measured switch makes one");

    rnic_cm_destroy(controlled);
    rnic_cm_destroy(plain);

    // Half a loop is a configuration error, not a degraded mode.
    rnic_cm_config half;
    std::memset(&half, 0, sizeof(half));
    half.version = SIMLLM_RNIC_CM_ABI_VERSION;
    half.qpn = 4;
    half.sq_depth = 32;
    half.cq_depth = 64;
    half.packetization = 1;
    half.receive = 0;
    half.congestion_control = 1;
    test.check(
        rnic_cm_create(&profile, &half) == nullptr,
        "rate control without a receive side is refused");
}

// The reaction point's state lives on the transmit pipeline, so a work request
// beginning or ending cannot touch it. This drives the facade the way the
// probe does and shows the rate crossing a completion unchanged.
void testPersistenceAcrossWorkRequests(TestRunner& test) {
    rnic_cm_device* endpoint = makeEndpoint(true);
    if (endpoint == nullptr) {
        test.check(false, "endpoint for the persistence check builds");
        return;
    }
    rnic_cm_event_info notice;
    std::memset(&notice, 0, sizeof(notice));
    notice.kind = RNIC_CM_EVENT_CNP_RECEIVED;
    for (int index = 0; index < 8; ++index) {
        rnic_cm_event(endpoint, &notice, 1000);
    }
    rnic_cm_nic_counter_set cut;
    rnic_cm_nic_counters(endpoint, &cut);

    rnic_cm_wqe request;
    std::memset(&request, 0, sizeof(request));
    request.wr_id = 1;
    request.destination = 1;
    request.payload_bytes = 4096;
    request.sge_count = 1;
    request.signaled = 1;
    std::uint64_t wqe_id = 0;
    test.check(
        rnic_cm_post(endpoint, &request, 1000, &wqe_id) == RNIC_CM_OK,
        "a work request posts after the cut");
    rnic_cm_doorbell_batch batch;
    rnic_cm_doorbell(endpoint, 1000, &batch);

    rnic_cm_nic_counter_set posted;
    rnic_cm_nic_counters(endpoint, &posted);
    test.check(
        posted.rp_current_rate_bps == cut.rp_current_rate_bps,
        "posting a work request does not touch the reaction point");

    rnic_cm_destroy(endpoint);
}

}  // namespace

int main() {
    try {
        TestRunner test;
        testNotificationPoint(test);
        testReactionPointCut(test);
        testReactionPointRecovery(test);
        testEgressQueue(test);
        testReceiveIntegration(test);
        testFacade(test);
        testPersistenceAcrossWorkRequests(test);
        if (test.failures() != 0) {
            std::cerr << test.failures()
                      << " congestion-control checks failed\n";
            return 1;
        }
        std::cout << "RNIC congestion-control checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "unexpected congestion-control failure: " << error.what()
                  << '\n';
        return 1;
    }
}
