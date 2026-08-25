from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from simllm.adapters.vllm.pd_connector import (
    PD_KV_PARAMS_SCHEMA,
    SimPdConnector,
    _params_for_consumer,
)


def _decode_request(**changes):
    values = {
        "request_id": "request-0",
        "num_prompt_tokens": 9,
        "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 512],
        "kv_transfer_params": {
            "schema": PD_KV_PARAMS_SCHEMA,
            "do_remote_prefill": True,
            "remote_request_id": "request-0",
            "session_request_id": "request-0",
            "remote_num_tokens": 8,
            "bootstrap_token_id": 512,
        },
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _connector(kv_role):
    connector = object.__new__(SimPdConnector)
    connector._kv_transfer_config = SimpleNamespace(
        kv_role=kv_role,
        engine_id=f"engine-{kv_role}",
    )
    connector._scheduled_request_ids = set()
    return connector


def test_consumer_reports_only_original_prompt_coverage():
    connector = _connector("kv_consumer")
    request = _decode_request()

    assert _params_for_consumer(request) is request.kv_transfer_params
    assert connector.get_num_new_matched_tokens(request, 0) == (8, False)
    assert connector.get_num_new_matched_tokens(request, 4) == (4, False)


def test_consumer_refuses_request_or_bootstrap_identity_loss():
    with pytest.raises(ValueError, match="identity"):
        _params_for_consumer(
            _decode_request(
                kv_transfer_params={
                    **_decode_request().kv_transfer_params,
                    "remote_request_id": "other",
                }
            )
        )
    with pytest.raises(ValueError, match="bootstrap"):
        _params_for_consumer(_decode_request(prompt_token_ids=[1] * 9))


def test_producer_returns_scheduler_visible_params_without_tensor_claim():
    connector = _connector("kv_producer")
    request = SimpleNamespace(
        request_id="request-0",
        num_prompt_tokens=8,
        output_token_ids=[512],
        kv_transfer_params={
            "schema": PD_KV_PARAMS_SCHEMA,
            "session_request_id": "request-0",
        },
    )

    delay_free, params = connector.request_finished(request, [1])

    assert delay_free is False
    assert params == {
        "schema": PD_KV_PARAMS_SCHEMA,
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_engine_id": "engine-kv_producer",
        "remote_request_id": "request-0",
        "session_request_id": "request-0",
        "remote_num_tokens": 8,
        "bootstrap_token_id": 512,
        "worker_tensor_transfer": False,
        "timing_authority": "simllm-declared-kv-handoff-v1",
    }
