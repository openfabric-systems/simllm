import pytest

from simllm.goal import GoalMessage, GoalTrace


def test_two_rank_ping():
    trace = GoalTrace(2)
    r0 = trace.rank(0)
    c = r0.calc(5000)
    s = r0.send(8192, to=1, tag=7)
    r0.requires(s, c)
    trace.rank(1).recv(8192, source=0, tag=7)

    text = trace.render()
    assert text.splitlines()[0] == "num_ranks 2"
    assert "r0op0: calc 5000" in text
    assert "r0op1: send 8192b to 1 tag 7" in text
    assert "r0op1 requires r0op0" in text
    assert "r1op0: recv 8192b from 0 tag 7" in text
    assert text.count("rank ") == 2


def test_write(tmp_path):
    trace = GoalTrace(1)
    trace.rank(0).calc(1)
    out = trace.write(tmp_path / "t.goal")
    assert out.read_text().startswith("num_ranks 1")


def test_request_partition_is_structured_metadata_not_goal_text():
    trace = GoalTrace(2)
    send = trace.rank(0).send(12, to=1, tag=9)
    receive = trace.rank(1).recv(12, source=0, tag=9)
    before = trace.render()

    message = GoalMessage(
        operation_id="step-0:layer-0:ep-dispatch",
        source_rank=0,
        destination_rank=1,
        payload_bytes=12,
        tag=9,
        send_label=send,
        receive_label=receive,
        request_payload_bytes=(("alpha", 4), ("beta", 8)),
    )
    trace.record_message(message)

    assert trace.messages == (message,)
    assert trace.render() == before
    assert "alpha" not in trace.render()


@pytest.mark.parametrize(
    ("partition", "match"),
    [
        ((("", 12),), "nonblank"),
        ((("alpha", 0),), "at least 1"),
        ((("alpha", 4), ("alpha", 8)), "duplicate"),
        ((("beta", 8), ("alpha", 4)), "request-major"),
        ((("alpha", 11),), "partition sums"),
    ],
)
def test_goal_message_rejects_malformed_request_partition(partition, match):
    with pytest.raises((TypeError, ValueError), match=match):
        GoalMessage(
            operation_id="a2av",
            source_rank=0,
            destination_rank=1,
            payload_bytes=12,
            tag=9,
            send_label="send",
            receive_label="receive",
            request_payload_bytes=partition,
        )
