import queue
import threading
from collections import deque

import pytest

from simllm.adapters.sglang import (
    PreparedSglangSubmission,
    SglangHttpSubmitter,
    SglangOpenLoopDriver,
    sglang_generate_payload,
    token_completion_times_from_sglang_chunks,
)
from simllm.workload import GenerationRequest, TransportRequestObservation


def request(request_id="r0", *, ordinal=0, arrived_at_ps=0, output_token_count=3):
    return GenerationRequest(
        ordinal=ordinal,
        request_id=request_id,
        arrived_at_ps=arrived_at_ps,
        prompt_token_ids=(1, 2, 3),
        output_token_count=output_token_count,
    )


def test_sglang_generate_payload_is_native_streaming_shape():
    assert sglang_generate_payload(request()) == {
        "rid": "r0",
        "input_ids": [1, 2, 3],
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 3,
            "ignore_eos": True,
        },
        "stream": True,
        "return_logprob": False,
    }


def test_cumulative_sglang_chunks_preserve_observed_completion_frontier():
    chunks = (
        ({"meta_info": {"completion_tokens": 1}}, 1_000),
        ({"meta_info": {"completion_tokens": 3}}, 1_500),
    )
    assert token_completion_times_from_sglang_chunks(chunks, expected_count=3) == (
        1_000,
        1_500,
        1_500,
    )


def test_cumulative_sglang_chunks_validates_identity_and_length_finish():
    chunks = (
        (
            {
                "meta_info": {
                    "id": "r0",
                    "completion_tokens": 3,
                    "finish_reason": {"type": "length", "length": 3},
                }
            },
            1_000,
        ),
    )
    assert token_completion_times_from_sglang_chunks(
        chunks, expected_count=3, expected_request_id="r0"
    ) == (1_000, 1_000, 1_000)
    with pytest.raises(ValueError, match="does not match"):
        token_completion_times_from_sglang_chunks(
            chunks, expected_count=3, expected_request_id="wrong"
        )


@pytest.mark.parametrize(
    ("meta", "match"),
    [
        ({"id": "r0", "finish_reason": {"type": "length"}}, "completion_tokens"),
        (
            {
                "id": "r0",
                "completion_tokens": 3,
                "finish_reason": {"type": "abort", "message": "stopped"},
            },
            "requested length",
        ),
    ],
)
def test_cumulative_sglang_chunks_rejects_schema_drift_or_abort(meta, match):
    with pytest.raises((ValueError, RuntimeError), match=match):
        token_completion_times_from_sglang_chunks(
            (({"meta_info": meta, "output_ids": [1]}, 1_000),),
            expected_count=3,
            expected_request_id="r0",
        )
@pytest.mark.parametrize(
    ("chunks", "expected", "match"),
    [
        ((), 1, "ended with 0"),
        ((({"meta_info": {"completion_tokens": 2}}, 1),), 1, "at most"),
        (
            (
                ({"meta_info": {"completion_tokens": 1}}, 2),
                ({"meta_info": {"completion_tokens": 0}}, 3),
            ),
            1,
            "decreased",
        ),
        (
            (
                ({"meta_info": {"completion_tokens": 1}}, 2),
                ({"meta_info": {"completion_tokens": 1}}, 1),
            ),
            1,
            "nonnegative and nondecreasing",
        ),
    ],
)
def test_cumulative_sglang_chunks_reject_invalid_streams(chunks, expected, match):
    with pytest.raises((TypeError, ValueError), match=match):
        token_completion_times_from_sglang_chunks(chunks, expected_count=expected)


def test_cumulative_sglang_chunks_rejects_error_even_after_expected_tokens():
    with pytest.raises(RuntimeError, match="invalid request"):
        token_completion_times_from_sglang_chunks(
            (
                ({"meta_info": {"completion_tokens": 1}}, 1),
                ({"error": {"message": "invalid request"}}, 2),
            ),
            expected_count=1,
        )


@pytest.mark.parametrize(
    "chunks",
    [
        (
            (
                {
                    "meta_info": {
                        "id": "r0",
                        "completion_tokens": 3,
                        "finish_reason": {"type": "length", "length": 3},
                    }
                },
                100,
            ),
            ({"meta_info": {"id": "r0", "completion_tokens": 3}}, 200),
        ),
        (
            (
                {
                    "meta_info": {
                        "id": "r0",
                        "completion_tokens": 3,
                        "finish_reason": {"type": "length", "length": 2},
                    }
                },
                100,
            ),
        ),
    ],
)
def test_cumulative_sglang_chunks_requires_one_consistent_terminal_frame(chunks):
    with pytest.raises(RuntimeError, match="length|terminal"):
        token_completion_times_from_sglang_chunks(
            chunks,
            expected_count=3,
            expected_request_id="r0",
        )


class FakeHttpResponse:
    status = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)

    def close(self):
        pass


class FakeHttpConnection:
    def __init__(self, lines):
        self.response = FakeHttpResponse(lines)

    def request(self, method, target, body, headers):
        assert method == "POST"
        assert target == "/generate"
        assert body
        assert headers["Content-Type"] == "application/json"

    def getresponse(self):
        return self.response

    def close(self):
        pass


def test_http_submitter_requires_done_and_rejects_stream_error(monkeypatch):
    clock = lambda: 1_000
    submitter = SglangHttpSubmitter("http://example.invalid")

    monkeypatch.setattr(
        "simllm.adapters.sglang.client._http_connection",
        lambda *args, **kwargs: (
            FakeHttpConnection(
                [
                    (
                        b'data: {"meta_info":{"id":"r0","completion_tokens":3,'
                        b'"finish_reason":{"type":"length","length":3}}}\n'
                    ),
                    b"data: [DONE]\n",
                ]
            ),
            "/generate",
        ),
    )
    observed = submitter(request(), submitted_at_ps=100, now_ps=clock)
    assert observed.token_completed_at_ps == (1_000, 1_000, 1_000)

    monkeypatch.setattr(
        "simllm.adapters.sglang.client._http_connection",
        lambda *args, **kwargs: (
            FakeHttpConnection(
                [
                    (
                        b'data: {"meta_info":{"id":"r0","completion_tokens":3,'
                        b'"finish_reason":{"type":"length","length":3}}}\n'
                    )
                ]
            ),
            "/generate",
        ),
    )
    with pytest.raises(RuntimeError, match=r"without a \[DONE\]"):
        submitter(request(), submitted_at_ps=100, now_ps=clock)

    monkeypatch.setattr(
        "simllm.adapters.sglang.client._http_connection",
        lambda *args, **kwargs: (
            FakeHttpConnection(
                [
                    (
                        b'data: {"meta_info":{"id":"r0","completion_tokens":1}}\n'
                    ),
                    b'data: {"error":{"message":"invalid request"}}\n',
                    b"data: [DONE]\n",
                ]
            ),
            "/generate",
        ),
    )
    with pytest.raises(RuntimeError, match="invalid request"):
        submitter(request(), submitted_at_ps=100, now_ps=clock)


class FakeClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        value = self.now_ns
        self.now_ns += 1
        return value

    def sleep(self, seconds):
        self.now_ns += round(seconds * 1_000_000_000)


def test_open_loop_driver_paces_arrivals_without_waiting_for_completion():
    clock = FakeClock()
    submitted = deque()

    class Submitter:
        def start(self, item, now_ps):
            submitted_at_ps = now_ps()
            submitted.append((item.request_id, submitted_at_ps))

            def await_observation():
                return TransportRequestObservation(
                    item.request_id,
                    submitted_at_ps,
                    tuple(now_ps() + index for index in range(item.output_token_count)),
                )

            return PreparedSglangSubmission(
                item.request_id, submitted_at_ps, await_observation
            )

    plan = (
        request("r0", ordinal=0, arrived_at_ps=0),
        request("r1", ordinal=1, arrived_at_ps=2_000, output_token_count=1),
        request("r2", ordinal=2, arrived_at_ps=2_000, output_token_count=2),
    )
    observations = SglangOpenLoopDriver(
        submit=Submitter(),
        monotonic_ns=clock,
        sleep=clock.sleep,
        max_workers=3,
    ).run_open_loop(plan)
    assert tuple(item.request_id for item in observations) == ("r0", "r1", "r2")
    assert tuple(item[0] for item in submitted) == ("r0", "r1", "r2")
    assert submitted[0][1] >= 0
    assert submitted[1][1] >= 2_000
    assert submitted[2][1] >= 2_000


def test_open_loop_driver_starts_equal_time_requests_in_order_while_all_are_in_flight():
    started: queue.Queue[str] = queue.Queue()
    release = threading.Event()

    class Submitter:
        def start(self, item, now_ps):
            submitted_at_ps = now_ps()
            started.put(item.request_id)

            def await_observation():
                assert release.wait(timeout=2)
                return TransportRequestObservation(
                    item.request_id,
                    submitted_at_ps,
                    (submitted_at_ps + 1,) * item.output_token_count,
                )

            return PreparedSglangSubmission(
                item.request_id, submitted_at_ps, await_observation
            )

    plan = tuple(request(f"r{index}", ordinal=index) for index in range(3))
    result = []
    failure = []

    def run():
        try:
            result.extend(
                SglangOpenLoopDriver(submit=Submitter(), max_workers=3).run_open_loop(plan)
            )
        except BaseException as exc:  # noqa: BLE001 - surface thread failure to test
            failure.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert [started.get(timeout=2) for _ in plan] == ["r0", "r1", "r2"]
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not failure
    assert [item.request_id for item in result] == ["r0", "r1", "r2"]


def test_open_loop_driver_rejects_a_pool_that_would_gate_arrivals():
    class Submitter:
        def start(self, item, now_ps):
            submitted = now_ps()
            return PreparedSglangSubmission(
                item.request_id,
                submitted,
                lambda: TransportRequestObservation(
                    item.request_id, submitted, (now_ps(),) * item.output_token_count
                ),
            )

    with pytest.raises(ValueError, match="exceeds max_workers"):
        SglangOpenLoopDriver(
            submit=Submitter(),
            max_workers=1,
        ).run_open_loop((request("r0"), request("r1", ordinal=1)))


def test_open_loop_driver_rejects_unsorted_or_misattributed_plan():
    class Submitter:
        def start(self, item, now_ps):
            submitted = now_ps()
            return PreparedSglangSubmission(
                "wrong",
                submitted,
                lambda: TransportRequestObservation(
                    "wrong", submitted, (now_ps(),) * item.output_token_count
                ),
            )

    driver = SglangOpenLoopDriver(submit=Submitter())
    with pytest.raises(ValueError, match="ordered"):
        driver.run_open_loop(
            (
                request("late", ordinal=0, arrived_at_ps=2),
                request("early", ordinal=1, arrived_at_ps=1),
            )
        )
    with pytest.raises(ValueError, match="for 'r0'"):
        driver.run_open_loop((request(),))


def test_open_loop_driver_rejects_disagreeing_submission_projection():
    class Submitter:
        def start(self, item, now_ps):
            del now_ps
            return PreparedSglangSubmission(
                item.request_id,
                100,
                lambda: TransportRequestObservation(
                    item.request_id,
                    900,
                    (1_000,) * item.output_token_count,
                ),
            )

    with pytest.raises(ValueError, match="changed.*timestamp"):
        SglangOpenLoopDriver(submit=Submitter()).run_open_loop((request(),))
