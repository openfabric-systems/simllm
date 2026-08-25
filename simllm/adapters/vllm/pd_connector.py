"""vLLM KV-connector seam for a simulated prefill/decode session.

The real vLLM scheduler calls this connector. SimExecutor has no paged KV
tensor, so worker-side load and save methods are deliberately empty. The
producer returns the exact prompt coverage and bootstrap token through
``kv_transfer_params``; the consumer reports that coverage to its scheduler.
Transfer timing belongs to the session's core KV-handoff event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

PD_KV_PARAMS_SCHEMA = "simllm-pd-kv-params-v1"
PD_CONNECTOR_NAME = "SimPdConnector"
PD_CONNECTOR_MODULE = "simllm.adapters.vllm.pd_connector"


@dataclass(frozen=True)
class SimPdConnectorMetadata(KVConnectorMetadata):
    """Opaque no-tensor metadata accepted by SimExecutor."""

    request_ids: tuple[str, ...] = ()


def _params_for_consumer(request: Request) -> dict[str, Any] | None:
    params = request.kv_transfer_params
    if not params or not params.get("do_remote_prefill"):
        return None
    if params.get("schema") != PD_KV_PARAMS_SCHEMA:
        raise ValueError("decode request has unsupported SimLLM KV parameters")
    session_request_id = params.get("session_request_id")
    if not isinstance(session_request_id, str) or not session_request_id:
        raise ValueError("decode request is missing its session request identity")
    if params.get("remote_request_id") != session_request_id:
        raise ValueError("decode request identity disagrees with remote KV parameters")
    remote_num_tokens = params.get("remote_num_tokens")
    if isinstance(remote_num_tokens, bool) or type(remote_num_tokens) is not int:
        raise TypeError("remote_num_tokens must be an integer")
    if not 0 < remote_num_tokens < request.num_prompt_tokens:
        raise ValueError(
            "remote_num_tokens must cover the original prompt and leave the "
            "decode bootstrap token local"
        )
    bootstrap = params.get("bootstrap_token_id")
    if isinstance(bootstrap, bool) or type(bootstrap) is not int or bootstrap < 0:
        raise ValueError("bootstrap_token_id must be a nonnegative integer")
    if request.prompt_token_ids is None or request.prompt_token_ids[-1] != bootstrap:
        raise ValueError("decode prompt does not end with the producer bootstrap token")
    return params


class SimPdConnector(KVConnectorBase_V1):
    """Scheduler-active, tensor-free connector for SimExecutor."""

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        kv_role = self._kv_transfer_config.kv_role
        if kv_role not in ("kv_producer", "kv_consumer"):
            raise ValueError(
                "SimPdConnector requires an exact kv_producer or kv_consumer role"
            )
        self._scheduled_request_ids: set[str] = set()

    @property
    def declared_pool_role(self) -> str:
        return (
            "prefill"
            if self._kv_transfer_config.kv_role == "kv_producer"
            else "decode"
        )

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        return None

    def wait_for_save(self) -> None:
        return None

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self._kv_transfer_config.kv_role != "kv_consumer":
            return 0, False
        params = _params_for_consumer(request)
        if params is None:
            return 0, False
        count = int(params["remote_num_tokens"]) - int(num_computed_tokens)
        return max(count, 0), False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens > 0:
            if self._kv_transfer_config.kv_role != "kv_consumer":
                raise RuntimeError("a producer cannot admit external KV tokens")
            _params_for_consumer(request)
            self._scheduled_request_ids.add(request.request_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimPdConnectorMetadata:
        request_ids = tuple(sorted(self._scheduled_request_ids))
        self._scheduled_request_ids.clear()
        return SimPdConnectorMetadata(request_ids=request_ids)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._kv_transfer_config.kv_role != "kv_producer":
            return False, None
        output_token_ids = tuple(request.output_token_ids)
        if len(output_token_ids) != 1:
            raise ValueError(
                "the simulated prefill producer must finish with one bootstrap token"
            )
        input_params = request.kv_transfer_params
        if not isinstance(input_params, dict):
            raise TypeError("prefill request is missing its session parameters")
        if input_params.get("schema") != PD_KV_PARAMS_SCHEMA:
            raise ValueError("prefill request has unsupported session parameters")
        session_request_id = input_params.get("session_request_id")
        if not isinstance(session_request_id, str) or not session_request_id:
            raise ValueError("prefill request is missing its session request identity")
        return False, {
            "schema": PD_KV_PARAMS_SCHEMA,
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": self._kv_transfer_config.engine_id,
            "remote_request_id": session_request_id,
            "session_request_id": session_request_id,
            "remote_num_tokens": request.num_prompt_tokens,
            "bootstrap_token_id": int(output_token_ids[0]),
            "worker_tensor_transfer": False,
            "timing_authority": "simllm-declared-kv-handoff-v1",
        }


__all__ = [
    "PD_CONNECTOR_MODULE",
    "PD_CONNECTOR_NAME",
    "PD_KV_PARAMS_SCHEMA",
    "SimPdConnector",
    "SimPdConnectorMetadata",
]
