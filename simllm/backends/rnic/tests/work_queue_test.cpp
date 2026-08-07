#include "fake_network.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "simllm/rnic/work_queue.h"

namespace {

using simllm::rnic::CompletionStatus;
using simllm::rnic::DropLocation;
using simllm::rnic::DropReason;
using simllm::rnic::EvidenceKind;
using simllm::rnic::NetworkEventKind;
using simllm::rnic::Picoseconds;
using simllm::rnic::PostStatus;
using simllm::rnic::WorkQueue;
using simllm::rnic::WorkQueueConfig;
using simllm::rnic::WorkRequest;
using simllm::rnic::WqeState;
using simllm::rnic::testing::FakeNetworkPort;

class TestRunner {
public:
    void check(bool condition, const std::string& message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template <typename Callable>
    void expectThrow(Callable&& callable, const std::string& message) {
        try {
            callable();
            check(false, message);
        } catch (const std::exception&) {
            check(true, message);
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
            check(
                false,
                message + "; wrong exception type: " + error.what());
        } catch (...) {
            check(false, message + "; wrong non-standard exception type");
        }
    }

    int failures() const noexcept { return failures_; }

private:
    int failures_{0};
};

WorkRequest request(
    std::uint64_t wr_id,
    bool signaled,
    std::uint64_t payload_bytes = 4096) {
    WorkRequest result;
    result.wr_id = wr_id;
    result.flow_id = wr_id + 1000;
    result.flow_tag = 7;
    result.destination = 1;
    result.payload_bytes = payload_bytes;
    result.signaled = signaled;
    return result;
}

WorkQueueConfig config(std::size_t sq_depth, std::size_t cq_depth) {
    WorkQueueConfig result;
    result.sq_depth = sq_depth;
    result.cq_depth = cq_depth;
    result.source = 0;
    result.qpn = 17;
    result.policy_context_token = 9001;
    return result;
}

void testServiceTimeValidation(TestRunner& test) {
    FakeNetworkPort network(1, 0);
    struct ServiceField {
        Picoseconds WorkQueueConfig::* member;
        const char* name;
    };
    const ServiceField fields[]{
        {&WorkQueueConfig::doorbell_service_ps, "doorbell service"},
        {&WorkQueueConfig::wqe_fetch_service_ps, "WQE-fetch service"},
        {&WorkQueueConfig::qpc_lookup_service_ps, "QPC-lookup service"},
        {&WorkQueueConfig::scheduler_service_ps, "scheduler service"},
        {&WorkQueueConfig::cqe_write_service_ps, "CQE-write service"},
    };
    const Picoseconds wrapped_negative = static_cast<Picoseconds>(-1);
    for (const ServiceField& field : fields) {
        WorkQueueConfig invalid = config(1, 1);
        invalid.*field.member = wrapped_negative;
        test.expectThrowAs<std::invalid_argument>(
            [&network, invalid]() {
                WorkQueue queue(invalid, network);
                (void)queue;
            },
            std::string(field.name)
                + " rejects a signed negative wrapped into uint64");
    }

    WorkQueueConfig boundary = config(1, 1);
    const Picoseconds max_valid = static_cast<Picoseconds>(
        std::numeric_limits<std::int64_t>::max());
    boundary.doorbell_service_ps = max_valid;
    boundary.wqe_fetch_service_ps = max_valid;
    boundary.qpc_lookup_service_ps = max_valid;
    boundary.scheduler_service_ps = max_valid;
    boundary.cqe_write_service_ps = max_valid;
    WorkQueue boundary_queue(boundary, network);
    test.check(
        boundary_queue.config().cqe_write_service_ps == max_valid,
        "INT64_MAX remains a valid configured service duration");
}

void testDoorbellTimestampOverflowIsAtomic(TestRunner& test) {
    FakeNetworkPort network(1, 0);
    WorkQueueConfig cfg = config(1, 1);
    cfg.doorbell_service_ps = 1;
    WorkQueue queue(cfg, network);
    const auto posted = queue.postSend(request(91, true), 0);
    const Picoseconds max_time = std::numeric_limits<Picoseconds>::max();

    test.expectThrowAs<std::overflow_error>(
        [&queue, max_time]() { queue.ringDoorbell(max_time); },
        "doorbell timestamp overflow has the exact asserted error type");
    const auto& record = queue.wqe(posted.wqe_id);
    test.check(
        queue.unpublishedWqeCount() == 1
            && queue.occupiedSqEntries() == 1
            && record.state == WqeState::Posted
            && record.doorbell_batch_id == 0
            && !record.timeline.doorbelled_at_ps.has_value()
            && !record.timeline.wqe_fetch_begin_at_ps.has_value()
            && queue.counters().doorbells == 0
            && queue.counters().doorbelled_wqes == 0,
        "failed doorbell leaves its WQE, timeline and counters untouched");
    queue.validateInvariants();

    const auto retry = queue.ringDoorbell(0);
    test.check(
        retry.batch_id == 1 && retry.wqe_count == 1
            && retry.observed_at_ps == 1,
        "failed doorbell consumes neither model time nor batch identity");
    queue.validateInvariants();
}

void testMidBatchFetchOverflowIsAtomic(TestRunner& test) {
    FakeNetworkPort network(2, 0);
    WorkQueueConfig cfg = config(2, 2);
    cfg.wqe_fetch_service_ps = 1;
    WorkQueue queue(cfg, network);
    const Picoseconds near_max =
        std::numeric_limits<Picoseconds>::max() - 1;
    queue.postSendBatch(
        {request(92, true), request(93, true)}, near_max);

    test.expectThrowAs<std::overflow_error>(
        [&queue, near_max]() { queue.ringDoorbell(near_max); },
        "second-WQE fetch overflow aborts the complete doorbell batch");
    test.check(
        queue.unpublishedWqeCount() == 2
            && queue.occupiedSqEntries() == 2
            && queue.counters().doorbells == 0
            && queue.counters().doorbelled_wqes == 0
            && !queue.nextEventTime().has_value(),
        "mid-batch overflow publishes no WQE or service event");
    for (const auto& record : queue.records()) {
        test.check(
            record.state == WqeState::Posted
                && record.doorbell_batch_id == 0
                && !record.timeline.doorbelled_at_ps.has_value()
                && !record.timeline.wqe_fetch_end_at_ps.has_value()
                && !record.timeline.admitted_at_ps.has_value(),
            "mid-batch overflow preserves each Posted WQE exactly");
    }
    queue.validateInvariants();
}

void testCqeTimestampOverflowIsAtomic(TestRunner& test) {
    FakeNetworkPort network(1, 0);
    WorkQueueConfig cfg = config(1, 1);
    cfg.cqe_write_service_ps = 1;
    WorkQueue queue(cfg, network);
    const auto posted = queue.postSend(request(94, true), 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    const auto token = network.tokenForWqe(posted.wqe_id);

    simllm::rnic::NetworkEvent overflow_event;
    overflow_event.kind = NetworkEventKind::Delivered;
    overflow_event.token = token;
    overflow_event.wqe_id = posted.wqe_id;
    overflow_event.event_time_ps =
        std::numeric_limits<Picoseconds>::max();
    test.expectThrowAs<std::overflow_error>(
        [&queue, &overflow_event]() {
            queue.onNetworkEvent(overflow_event);
        },
        "CQE timestamp overflow has the exact asserted error type");

    const auto& failed = queue.wqe(posted.wqe_id);
    test.check(
        failed.state == WqeState::InFlight
            && failed.network_token == token
            && !failed.timeline.network_outcome_at_ps.has_value()
            && !failed.timeline.transport_retired_at_ps.has_value()
            && !failed.completion_status.has_value()
            && queue.counters().network_delivered == 0
            && queue.counters().cqes_visible == 0
            && queue.completionQueueDepth() == 0,
        "failed CQE plan consumes no token, outcome, counter or timestamp");
    queue.validateInvariants();

    queue.onNetworkEvent(network.take(
        token, NetworkEventKind::Delivered, 0));
    queue.progress(1);
    const auto cqes = queue.pollCompletionQueue(1, 1);
    test.check(
        cqes.size() == 1 && cqes[0].wqe_id == posted.wqe_id
            && cqes[0].cqe_sequence == 1
            && queue.occupiedSqEntries() == 0,
        "CQE-overflow rollback preserves a successful retry and CQE identity");
    queue.validateInvariants();
}

void testRetirementPrefixOverflowIsAtomic(TestRunner& test) {
    FakeNetworkPort network(2, 0);
    WorkQueueConfig cfg = config(2, 2);
    cfg.cqe_write_service_ps = 1;
    WorkQueue queue(cfg, network);
    queue.postSendBatch({request(95, true), request(96, true)}, 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    const auto first = network.tokenForWqe(1);
    const auto second = network.tokenForWqe(2);
    const Picoseconds near_max =
        std::numeric_limits<Picoseconds>::max() - 1;
    queue.onNetworkEvent(network.take(
        second, NetworkEventKind::Delivered, near_max));

    simllm::rnic::NetworkEvent first_event;
    first_event.kind = NetworkEventKind::Delivered;
    first_event.token = first;
    first_event.wqe_id = 1;
    first_event.event_time_ps = near_max;
    test.expectThrowAs<std::overflow_error>(
        [&queue, &first_event]() { queue.onNetworkEvent(first_event); },
        "overflow on CQE two aborts the entire newly unlocked prefix");
    test.check(
        queue.wqe(1).state == WqeState::InFlight
            && !queue.wqe(1).timeline.network_outcome_at_ps.has_value()
            && queue.wqe(2).state
                == WqeState::AwaitingOrderedRetirement
            && !queue.wqe(2).timeline.transport_retired_at_ps.has_value()
            && queue.counters().network_delivered == 1
            && queue.counters().cqes_visible == 0
            && queue.completionQueueDepth() == 0,
        "prefix overflow commits neither the head callback nor any retirement");
    queue.validateInvariants();
}

void testAcceptedPrefixAndDoorbell(TestRunner& test) {
    FakeNetworkPort network(2, 100);
    WorkQueueConfig cfg = config(2, 2);
    cfg.doorbell_service_ps = 10;
    cfg.wqe_fetch_service_ps = 3;
    WorkQueue queue(cfg, network);

    const std::vector<WorkRequest> requests{
        request(101, true), request(102, true), request(103, true)};
    const auto posted = queue.postSendBatch(requests, 0);
    test.check(posted.status == PostStatus::SqFull,
               "three-WR batch reports SQ full");
    test.check(posted.accepted.size() == 2,
               "accepted prefix contains exactly two WQEs");
    test.check(posted.bad_wr_index.has_value() && *posted.bad_wr_index == 2,
               "bad WR index identifies the first unaccepted WR");
    test.check(queue.counters().sq_full_rejections == 1,
               "SQ full has one controlled rejection counter");
    test.check(queue.evidence().size() == 1
                   && queue.evidence()[0].kind == EvidenceKind::SqFull
                   && queue.evidence()[0].wr_id == 103,
               "SQ-full evidence identifies the rejected WR");

    const auto doorbell = queue.ringDoorbell(0);
    test.check(doorbell.wqe_count == 2 && doorbell.observed_at_ps == 10,
               "one doorbell publishes the accepted prefix");
    test.check(queue.wqe(1).timeline.admitted_at_ps == 13,
               "first WQE records fetch/admission timing");
    test.check(queue.wqe(2).timeline.admitted_at_ps == 16,
               "second WQE serializes behind the first fetch");
    queue.validateInvariants();
}

void testUnsignaledOrderedRetirement(TestRunner& test) {
    FakeNetworkPort network(2, 100);
    WorkQueueConfig cfg = config(4, 4);
    cfg.doorbell_service_ps = 10;
    WorkQueue queue(cfg, network);

    const auto posted = queue.postSendBatch(
        {request(201, false), request(202, true)}, 0);
    queue.ringDoorbell(0);
    queue.progress(10);
    test.check(posted.accepted.size() == 2 && network.inflightCount() == 2,
               "two WQEs enter the fake network");
    test.check(network.history()[0].descriptor.flow_id == 1201
                   && network.history()[0].descriptor.flow_tag == 7
                   && network.history()[0].descriptor.policy_context_token
                       == 9001,
               "network boundary preserves flow and opaque policy identity");

    const auto second_token = network.tokenForWqe(2);
    queue.onNetworkEvent(network.take(
        second_token, NetworkEventKind::Delivered, 50));
    test.check(queue.wqe(2).timeline.network_outcome_at_ps == 50
                   && !queue.wqe(2)
                           .timeline.transport_retired_at_ps.has_value(),
               "network outcome is observable before ordered retirement");
    queue.progress(50);
    test.check(queue.completionQueueDepth() == 0,
               "later network callback waits for SQ retirement order");

    const auto first_token = network.tokenForWqe(1);
    queue.onNetworkEvent(network.take(
        first_token, NetworkEventKind::Delivered, 60));
    queue.progress(60);
    test.check(queue.completionQueueDepth() == 1,
               "one signaled CQE covers the ordered prefix");
    test.check(queue.occupiedSqEntries() == 2,
               "transport retirement alone does not reclaim SQ slots");

    const auto cqes = queue.pollCompletionQueue(8, 60);
    test.check(cqes.size() == 1 && cqes[0].wqe_id == 2,
               "only the signaled WQE produces a CQE");
    test.check(cqes[0].byte_count == 0
                   && (cqes[0].valid_fields
                       & simllm::rnic::kCqeValidByteCount) == 0,
               "SEND CQE does not invent a valid byte-count field");
    test.check(queue.occupiedSqEntries() == 0,
               "polling the signaled CQE reclaims both SQ slots");
    test.check(queue.wqe(1).timeline.transport_retired_at_ps == 60
                   && queue.wqe(2).timeline.transport_retired_at_ps == 60,
               "out-of-order callbacks retire in SQ order");
    queue.validateInvariants();
}

void testAllUnsignaledExhaustsSq(TestRunner& test) {
    FakeNetworkPort network(2, 10);
    WorkQueue queue(config(2, 2), network);
    queue.postSend(request(301, false), 0);
    queue.postSend(request(302, false), 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    for (const auto& event : network.takeDue(10)) {
        queue.onNetworkEvent(event);
    }
    queue.progress(10);

    test.check(!queue.hasPendingPhysicalWork(),
               "retired unsignaled WQEs leave no physical work");
    test.check(queue.occupiedSqEntries() == 2,
               "retired unsignaled WQEs remain unreclaimed");
    test.check(queue.postSend(request(303, true), 10).status == PostStatus::SqFull,
               "all-unsignaled stream reaches the SQ capacity boundary");
    queue.validateInvariants();
}

void testNetworkBusyPreservesHeadOrder(TestRunner& test) {
    FakeNetworkPort network(1, 100);
    WorkQueueConfig cfg = config(4, 4);
    cfg.wqe_fetch_service_ps = 10;
    WorkQueue queue(cfg, network);
    queue.postSendBatch({request(401, true), request(402, true)}, 0);
    queue.ringDoorbell(0);

    queue.progress(10);
    queue.progress(20);
    test.check(queue.counters().network_busy == 1,
               "second WQE records one network-busy stall");
    test.check(network.history().size() == 1
                   && network.history()[0].descriptor.wqe_id == 1,
               "busy port does not bypass the SQ head");

    for (const auto& event : network.takeDue(110)) {
        queue.onNetworkEvent(event);
    }
    queue.progress(110);
    test.check(network.history().size() == 2
                   && network.history()[1].descriptor.wqe_id == 2
                   && network.history()[1].submitted_at_ps == 110,
               "blocked SQ head retries at the advertised time");
    for (const auto& event : network.takeDue(210)) {
        queue.onNetworkEvent(event);
    }
    queue.progress(210);
    const auto cqes = queue.pollCompletionQueue(8, 210);
    test.check(cqes.size() == 2 && cqes[0].wqe_id == 1
                   && cqes[1].wqe_id == 2,
               "busy/retry path preserves completion order");
    queue.validateInvariants();
}

void testBusyRetrySurvivesUnrelatedCompletion(TestRunner& test) {
    FakeNetworkPort network(1, 100);
    WorkQueueConfig cfg = config(4, 4);
    cfg.wqe_fetch_service_ps = 10;
    WorkQueue queue(cfg, network);
    queue.postSendBatch({request(451, true), request(452, true)}, 0);
    queue.ringDoorbell(0);

    queue.progress(10);
    network.forceBusyUntil(1000);
    queue.progress(20);
    test.check(queue.nextEventTime() == 1000,
               "busy result publishes its advertised retry time");

    const auto first = network.tokenForWqe(1);
    queue.onNetworkEvent(network.take(
        first, NetworkEventKind::Delivered, 110));
    queue.progress(110);
    test.check(network.history().size() == 1
                   && queue.nextEventTime() == 1000,
               "unrelated completion does not revoke a policy retry gate");

    queue.progress(1000);
    test.check(network.history().size() == 2
                   && network.history()[1].submitted_at_ps == 1000,
               "blocked WQE retries exactly at the advertised time");
    queue.validateInvariants();
}

void testDropProducesControlledErrorCqe(TestRunner& test) {
    FakeNetworkPort network(1, 100);
    WorkQueue queue(config(2, 2), network);
    const auto posted = queue.postSend(request(501, false), 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    const auto token = network.tokenForWqe(posted.wqe_id);
    queue.onNetworkEvent(network.take(
        token,
        NetworkEventKind::Dropped,
        10,
        DropLocation::RxPort,
        DropReason::Injected));
    queue.progress(10);
    const auto cqes = queue.pollCompletionQueue(1, 10);

    test.check(cqes.size() == 1
                   && cqes[0].status == CompletionStatus::TransportError,
               "unsignaled dropped WQE produces an error CQE");
    test.check(queue.evidence().size() == 1
                   && queue.evidence()[0].kind == EvidenceKind::NetworkDrop
                   && queue.evidence()[0].drop_location == DropLocation::RxPort
                   && queue.evidence()[0].drop_reason == DropReason::Injected,
               "drop evidence preserves controlled location and reason");
    test.check(queue.occupiedSqEntries() == 0,
               "polling error CQE reclaims the failed WQE");
    test.check(queue.wqe(posted.wqe_id).timeline.network_accepted_at_ps == 0
                   && queue.wqe(posted.wqe_id).timeline.network_outcome_at_ps
                       == 10
                   && !queue.wqe(posted.wqe_id)
                           .timeline.first_packet_at_ps.has_value()
                   && !queue.wqe(posted.wqe_id)
                           .timeline.last_packet_at_ps.has_value(),
               "flow port does not fabricate packet issue timestamps");
    queue.validateInvariants();
}

void testCqOverrunIsFatalAndAccounted(TestRunner& test) {
    FakeNetworkPort network(3, 100);
    WorkQueue queue(config(4, 1), network);
    queue.postSendBatch(
        {request(601, true), request(602, true), request(603, true)}, 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    const auto first = network.tokenForWqe(1);
    const auto second = network.tokenForWqe(2);
    const auto third = network.tokenForWqe(3);
    queue.onNetworkEvent(network.take(
        first, NetworkEventKind::Delivered, 10));
    queue.onNetworkEvent(network.take(
        second, NetworkEventKind::Delivered, 10));
    queue.progress(10);

    test.check(queue.fatal() && queue.counters().cq_overruns == 1,
               "second CQE causes one fatal CQ overrun");
    test.check(queue.evidence().size() == 1
                   && queue.evidence()[0].kind == EvidenceKind::CqOverrun
                   && queue.evidence()[0].wqe_id == 2,
               "CQ-overrun evidence identifies the first lost CQE");
    queue.onNetworkEvent(network.take(
        third, NetworkEventKind::Delivered, 20));
    queue.progress(20);
    const auto cqes = queue.pollCompletionQueue(2, 20);
    test.check(cqes.size() == 1 && cqes[0].wqe_id == 1,
               "CQ overrun never overwrites the owned entry");
    test.check(queue.occupiedSqEntries() == 2
                   && queue.wqe(2).state == simllm::rnic::WqeState::Error
                   && !queue.wqe(2).timeline.sq_reclaimed_at_ps.has_value()
                   && queue.wqe(3).state
                       == simllm::rnic::WqeState::CompletionPending,
               "fatal overrun freezes later CQ publication and reclamation");
    test.check(queue.counters().cqes_visible == 1
                   && queue.counters().cq_overruns == 1,
               "fatal overrun remains the unique first failure");
    test.check(queue.hasPendingPhysicalWork()
                   && !queue.nextEventTime().has_value(),
               "fatal queue reports non-quiescence without a schedulable event");
    queue.validateInvariants();
}

void testSerializedCqeWritesAndExplicitEventPriority(TestRunner& test) {
    FakeNetworkPort network(2, 0);
    WorkQueueConfig cfg = config(2, 1);
    cfg.cqe_write_service_ps = 10;
    WorkQueue queue(cfg, network);
    queue.postSendBatch({request(651, true), request(652, true)}, 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    const auto first = network.tokenForWqe(1);
    const auto second = network.tokenForWqe(2);
    queue.onNetworkEvent(network.take(
        first, NetworkEventKind::Delivered, 0));
    queue.onNetworkEvent(network.take(
        second, NetworkEventKind::Delivered, 0));

    queue.progress(10);
    test.check(queue.completionQueueDepth() == 1
                   && queue.wqe(1).timeline.cqe_visible_at_ps == 10
                   && !queue.wqe(2).timeline.cqe_visible_at_ps.has_value(),
               "CQE writes serialize on the configured service resource");

    const auto first_cqe = queue.pollCompletionQueue(1, 20);
    queue.progress(20);
    const auto second_cqe = queue.pollCompletionQueue(1, 20);
    test.check(first_cqe.size() == 1 && second_cqe.size() == 1
                   && second_cqe[0].visible_at_ps == 20 && !queue.fatal(),
               "host-first polling at a timestamp frees space before progress");
    queue.validateInvariants();
}

void testPollPublishesStrictlyOlderCqe(TestRunner& test) {
    FakeNetworkPort network(1, 0);
    WorkQueueConfig cfg = config(1, 1);
    cfg.cqe_write_service_ps = 10;
    WorkQueue queue(cfg, network);
    queue.postSend(request(656, true), 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    queue.onNetworkEvent(network.take(
        network.tokenForWqe(1), NetworkEventKind::Delivered, 0));

    const auto cqes = queue.pollCompletionQueue(1, 20);
    test.check(cqes.size() == 1 && cqes[0].visible_at_ps == 10
                   && cqes[0].polled_at_ps == 20,
               "late host poll cannot skip a strictly older CQE event");
    queue.validateInvariants();
}

void testRejectedAndMalformedNetworkOutcomes(TestRunner& test) {
    FakeNetworkPort rejecting(1, 0);
    WorkQueue rejected_queue(config(1, 1), rejecting);
    rejecting.rejectNext();
    const auto rejected = rejected_queue.postSend(request(661, false), 0);
    rejected_queue.ringDoorbell(0);
    rejected_queue.progress(0);
    const auto rejected_cqe = rejected_queue.pollCompletionQueue(1, 0);
    test.check(rejected_cqe.size() == 1
                   && rejected_cqe[0].status
                       == CompletionStatus::NetworkRejected
                   && !rejected_queue.wqe(rejected.wqe_id)
                           .timeline.network_accepted_at_ps.has_value()
                   && rejected_queue.wqe(rejected.wqe_id)
                           .timeline.network_outcome_at_ps == 0,
               "immediate rejection has an outcome but no packet acceptance");
    rejected_queue.validateInvariants();

    FakeNetworkPort network(1, 0);
    WorkQueue queue(config(1, 1), network);
    const auto posted = queue.postSend(request(662, true), 0);
    queue.ringDoorbell(0);
    queue.progress(0);
    auto malformed = network.take(
        network.tokenForWqe(posted.wqe_id),
        NetworkEventKind::Delivered,
        1,
        DropLocation::RxPort,
        DropReason::Injected);
    malformed.kind = NetworkEventKind::Dropped;
    malformed.drop_location = static_cast<DropLocation>(99);
    test.expectThrow(
        [&queue, &malformed]() { queue.onNetworkEvent(malformed); },
        "unknown drop-location enum is not controlled evidence");
    malformed.drop_location = DropLocation::RxPort;
    malformed.drop_reason = static_cast<DropReason>(99);
    test.expectThrow(
        [&queue, &malformed]() { queue.onNetworkEvent(malformed); },
        "unknown drop-reason enum is not controlled evidence");
    malformed.kind = NetworkEventKind::Delivered;
    malformed.drop_reason = DropReason::Injected;
    test.expectThrow(
        [&queue, &malformed]() { queue.onNetworkEvent(malformed); },
        "delivered event with drop evidence is rejected before accounting");
    test.check(queue.counters().network_delivered == 0,
               "malformed event does not mutate delivery counters");
    malformed.drop_location = DropLocation::None;
    malformed.drop_reason = DropReason::None;
    queue.onNetworkEvent(malformed);
    queue.progress(1);
    queue.pollCompletionQueue(1, 1);
    queue.validateInvariants();
}

void testZeroLatencyOversubscriptionIsExplicitlyUnsupported(
    TestRunner& test) {
    FakeNetworkPort network(1, 0);
    WorkQueue queue(config(2, 2), network);
    queue.postSendBatch({request(666, true), request(667, true)}, 0);
    queue.ringDoorbell(0);
    test.expectThrow(
        [&queue]() { queue.progress(0); },
        "fake port rejects zero-latency oversubscription explicitly");
}

void testFakeNetworkTimestampOverflow(TestRunner& test) {
    FakeNetworkPort network(
        1, std::numeric_limits<simllm::rnic::Picoseconds>::max());
    WorkQueue queue(config(1, 1), network);
    queue.postSend(request(671, true), 1);
    queue.ringDoorbell(1);
    test.expectThrow(
        [&queue]() { queue.progress(1); },
        "fake network rejects timestamp overflow instead of wrapping");
}

void testCqWrapOwnerGeneration(TestRunner& test) {
    FakeNetworkPort network(2, 0);
    WorkQueue queue(config(2, 2), network);
    std::vector<simllm::rnic::CompletionEntry> all;

    for (std::uint64_t wave = 0; wave < 2; ++wave) {
        queue.postSendBatch(
            {request(700 + wave * 2, true),
             request(701 + wave * 2, true)},
            wave * 10);
        queue.ringDoorbell(wave * 10);
        queue.progress(wave * 10);
        const auto first = network.tokenForWqe(wave * 2 + 1);
        const auto second = network.tokenForWqe(wave * 2 + 2);
        const auto completion_time = wave * 10 + 1;
        queue.onNetworkEvent(network.take(
            first, NetworkEventKind::Delivered, completion_time));
        queue.onNetworkEvent(network.take(
            second, NetworkEventKind::Delivered, completion_time));
        queue.progress(completion_time);
        auto cqes = queue.pollCompletionQueue(2, completion_time);
        all.insert(all.end(), cqes.begin(), cqes.end());
    }

    test.check(all.size() == 4,
               "two CQ waves return four completions");
    test.check(all[0].cq_slot == 0 && all[1].cq_slot == 1
                   && all[2].cq_slot == 0 && all[3].cq_slot == 1,
               "CQ slots wrap modulo depth");
    test.check(all[0].owner_generation == 0 && all[1].owner_generation == 0
                   && all[2].owner_generation == 1
                   && all[3].owner_generation == 1,
               "CQ owner generation toggles on wrap");
    queue.validateInvariants();
}

void testUnknownNetworkTokenThrows(TestRunner& test) {
    FakeNetworkPort network(1, 0);
    WorkQueue queue(config(1, 1), network);
    simllm::rnic::NetworkEvent event;
    event.token = 999;
    event.wqe_id = 1;
    event.event_time_ps = 0;
    test.expectThrow(
        [&queue, &event]() { queue.onNetworkEvent(event); },
        "unknown network token is an asserted programming failure");
}

}  // namespace

int main() {
    TestRunner test;
    testServiceTimeValidation(test);
    testDoorbellTimestampOverflowIsAtomic(test);
    testMidBatchFetchOverflowIsAtomic(test);
    testCqeTimestampOverflowIsAtomic(test);
    testRetirementPrefixOverflowIsAtomic(test);
    testAcceptedPrefixAndDoorbell(test);
    testUnsignaledOrderedRetirement(test);
    testAllUnsignaledExhaustsSq(test);
    testNetworkBusyPreservesHeadOrder(test);
    testBusyRetrySurvivesUnrelatedCompletion(test);
    testDropProducesControlledErrorCqe(test);
    testCqOverrunIsFatalAndAccounted(test);
    testSerializedCqeWritesAndExplicitEventPriority(test);
    testPollPublishesStrictlyOlderCqe(test);
    testRejectedAndMalformedNetworkOutcomes(test);
    testZeroLatencyOversubscriptionIsExplicitlyUnsupported(test);
    testFakeNetworkTimestampOverflow(test);
    testCqWrapOwnerGeneration(test);
    testUnknownNetworkTokenThrows(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " RNIC WQ checks failed\n";
        return 1;
    }
    std::cout << "all RNIC WQ checks passed\n";
    return 0;
}
