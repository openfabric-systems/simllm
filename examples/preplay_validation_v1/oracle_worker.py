"""Validation-only instrumentation around vLLM's stock CPU worker."""

from __future__ import annotations

import re
from typing import Any

import torch
from vllm.v1.worker.cpu_worker import CPUWorker

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_LATEST_WORKER: ValidationCPUWorker | None = None
CONSTRUCTION_STARTED = False


def latest_validation_worker() -> ValidationCPUWorker | None:
    """Return the most recently completed validation-worker construction."""

    return _LATEST_WORKER


class ValidationCPUWorker(CPUWorker):
    """Delegate execution to CPUWorker while observing logits and gates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global CONSTRUCTION_STARTED
        CONSTRUCTION_STARTED = True
        self.validation_steps: list[dict[str, Any]] = []
        self._validation_active = False
        self._validation_scheduler_output: Any | None = None
        self._validation_gate_rows: dict[int, list[dict[str, Any]]] = {}
        self._validation_token_rows: list[dict[str, Any]] = []
        self._validation_handles: list[Any] = []
        super().__init__(*args, **kwargs)
        global _LATEST_WORKER
        _LATEST_WORKER = self

    @staticmethod
    def _boundary_rows(logits: torch.Tensor, selected_count: int) -> list[dict[str, Any]]:
        needed = selected_count + 1
        values, indices = torch.topk(logits.detach().float(), needed, dim=-1)
        rows = []
        for row_values, row_indices in zip(
            values.cpu().tolist(), indices.cpu().tolist(), strict=True
        ):
            rows.append(
                {
                    "selected_ids": row_indices[:selected_count],
                    "boundary_ids": row_indices[selected_count - 1 : selected_count + 1],
                    "margin": row_values[selected_count - 1] - row_values[selected_count],
                }
            )
        return rows

    def _attach_validation_hooks(self) -> None:
        model = self.model_runner.get_model()
        layers = []
        for name, module in model.named_modules():
            match = _LAYER_RE.search(name)
            if not name.endswith(".block_sparse_moe.gate") or match is None:
                continue
            layer_index = int(match.group(1))
            layers.append(layer_index)

            def capture_gate(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                layer: int = layer_index,
            ) -> None:
                if not self._validation_active:
                    return
                logits = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(f"vLLM gate layer {layer} returned no tensor")
                if layer in self._validation_gate_rows:
                    raise RuntimeError(f"vLLM gate layer {layer} ran twice in one step")
                self._validation_gate_rows[layer] = self._boundary_rows(logits, 8)

            self._validation_handles.append(module.register_forward_hook(capture_gate))

        if tuple(sorted(layers)) != tuple(range(24)):
            raise RuntimeError(f"expected vLLM Granite gate layers 0..23, found {layers}")

        def capture_sampler(
            _module: Any,
            inputs: tuple[Any, ...],
        ) -> None:
            if not self._validation_active:
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("vLLM sampler hook received no logits tensor")
            logits = inputs[0]
            values, indices = torch.topk(logits.detach().float(), 2, dim=-1)
            self._validation_token_rows = [
                {
                    "boundary_ids": row_indices,
                    "margin": row_values[0] - row_values[1],
                }
                for row_values, row_indices in zip(
                    values.cpu().tolist(), indices.cpu().tolist(), strict=True
                )
            ]

        self._validation_handles.append(
            self.model_runner.sampler.register_forward_pre_hook(capture_sampler)
        )

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        result = super().load_model(*args, **kwargs)
        self._attach_validation_hooks()
        return result

    def execute_model(self, scheduler_output: Any) -> Any:
        if self._validation_active:
            raise RuntimeError("validation step overlap")
        self._validation_active = True
        self._validation_scheduler_output = scheduler_output
        self._validation_gate_rows = {}
        self._validation_token_rows = []
        try:
            result = super().execute_model(scheduler_output)
        except BaseException:
            self._validation_active = False
            raise
        if result is not None:
            self._finish_validation_step(result)
        return result

    def sample_tokens(self, grammar_output: Any) -> Any:
        try:
            result = super().sample_tokens(grammar_output)
            self._finish_validation_step(result)
            return result
        except BaseException:
            self._validation_active = False
            raise

    def _finish_validation_step(self, result: Any) -> None:
        scheduler_output = self._validation_scheduler_output
        if scheduler_output is None:
            raise RuntimeError("validation step has no scheduler output")
        req_ids = list(getattr(result, "req_ids", ()) or ())
        sampled = getattr(result, "sampled_token_ids", ())
        if isinstance(sampled, torch.Tensor):
            sampled_rows = [
                [int(value) for value in values] for values in sampled.detach().cpu().tolist()
            ]
        else:
            sampled_rows = [[int(value) for value in values] for values in (sampled or ())]
        token_rows = []
        for index, request_id in enumerate(req_ids):
            values = sampled_rows[index] if index < len(sampled_rows) else []
            boundary = (
                self._validation_token_rows[index]
                if index < len(self._validation_token_rows)
                else None
            )
            token_rows.append(
                {
                    "request_id": request_id,
                    "sampled_token_ids": values,
                    "boundary": boundary,
                }
            )
        self.validation_steps.append(
            {
                "num_scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
                "gate_rows": {
                    str(layer): rows for layer, rows in sorted(self._validation_gate_rows.items())
                },
                "token_rows": token_rows,
            }
        )
        self._validation_active = False
        self._validation_scheduler_output = None
