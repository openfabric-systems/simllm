"""Real pinned vLLM batch queue with model-side pipeline service.

Run in a vLLM 0.27.1 CPU environment with VLLM_ENABLE_V1_MULTIPROCESSING=0.
No checkpoint weights, GPU execution or hardware-calibration claim is involved.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
from pathlib import Path

FREEZE = "575f94c7"


def exact_pipeline_oracles():
    """Independent source-hold recurrence, separate from engine relations."""
    from simllm.adapters.vllm.pipeline import dense_pipeline_boundaries
    from simllm.compute import ModelDims
    from simllm.core import (
        CoarseDeviceProfile,
        CoarseDeviceRuntime,
        ComputeWork,
        ExecutionGraph,
        ExecutionOperation,
        RequestPhase,
        ScheduledRequest,
        StepRecord,
    )
    from simllm.traffic.pipeline import compose_pipeline_graph

    rows = []
    geometry = ModelDims(num_layers=4, hidden_size=128, intermediate_size=256,
                         num_heads=8, num_kv_heads=8, head_size=16, vocab_size=1024)
    compute_ps = 10_000
    for pp in (2, 4):
        for batches in (2, 4):
            for rate in (100_000_000_000, 200_000_000_000):
                runtime = CoarseDeviceRuntime(CoarseDeviceProfile(rnic_rate_bps=rate))
                stages = tuple((stage * 8,) for stage in range(pp))
                previous_release = [0] * pp
                send_ps = 2 * (256 * 8 * 10**12 // rate)
                expected = []
                actual = []
                for batch in range(batches):
                    arrival = 0
                    for stage in range(pp):
                        completed = max(arrival, previous_release[stage]) + compute_ps
                        previous_release[stage] = completed + (send_ps if stage < pp - 1 else 0)
                        arrival = previous_release[stage]
                    expected.append(completed)
                    record = StepRecord(batch, 0, (
                        ScheduledRequest(f"request-{batch}", RequestPhase.DECODE, 1, 0, 49),
                    ), num_sampled=1)
                    graphs = tuple(ExecutionGraph(f"oracle-{batch}", batch, 0, (
                        ExecutionOperation("compute", ranks[0], "compute", ComputeWork(
                            "declared-fixed", nominal_duration_ps=compute_ps)),
                    )) for ranks in stages)
                    graph = compose_pipeline_graph(
                        record, geometry, pp, stages, graphs,
                        boundary_graphs=dense_pipeline_boundaries(record, stages, 128, 2),
                    )
                    actual.append(runtime.execute(graph).completed_at_ps)
                assert actual == expected, (pp, batches, rate, actual, expected)
                causal_floor = pp * compute_ps + (pp - 1) * send_ps
                serial_ceiling = batches * causal_floor
                assert causal_floor <= actual[-1] <= serial_ceiling
                rows.append({"pp": pp, "batches": batches, "rate_bps": rate,
                             "compute_ps": compute_ps, "send_ps": send_ps,
                             "actual_completion_ps": actual, "expected_completion_ps": expected,
                             "causal_floor_ps": causal_floor, "serial_ceiling_ps": serial_ceiling})
    return rows


def run_case(model, pp, tp, rate):
    from vllm import EngineArgs, LLMEngine, SamplingParams

    from simllm.adapters.vllm import (
        SimExecutor,
        SimExecutorConfig,
        configure,
        latest_executor,
        reset_configuration,
    )
    from simllm.backends.pipeline_step_sink import PipelineRuntimeStepSink
    from simllm.compute import GpuSpec, RooflineProvider
    from simllm.core import CoarseDeviceProfile, CoarseDeviceRuntime, RequestPhase

    sink = PipelineRuntimeStepSink(
        runtime=CoarseDeviceRuntime(CoarseDeviceProfile(rnic_rate_bps=rate)),
        # One TP group per coarse eight-slot node. Unused slots are idle.
        rank_map=tuple(stage * 8 + lane for stage in range(pp) for lane in range(tp)),
    )
    reset_configuration()
    configure(step_sink=sink, gpu=GpuSpec("declared-study", 100e12, 1e12),
              compute_provider=RooflineProvider(efficiency=1),
              config=SimExecutorConfig(mode="virtual", token_id=512,
                                       emit_sampled_request_ids=True))
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(model), skip_tokenizer_init=True,
        distributed_executor_backend=SimExecutor,
        tensor_parallel_size=tp, pipeline_parallel_size=pp,
        enforce_eager=True, dtype="bfloat16", max_model_len=256,
        num_gpu_blocks_override=128, max_num_batched_tokens=16, max_num_seqs=8,
        enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, disable_log_stats=True,
    ))
    executor = latest_executor()
    assert executor is not None and executor._pipeline is not None
    assert engine.engine_core.engine_core.batch_queue is not None
    assigned = []
    times = {f"request-{index}": [] for index in range(4)}
    token_ids = {request_id: [] for request_id in times}
    try:
        for index, request_id in enumerate(times):
            assigned.append(engine.add_request(
                request_id, {"prompt_token_ids": [10 + index] * 48},
                SamplingParams(max_tokens=8, temperature=0, ignore_eos=True, detokenize=False),
            ))
        iterations = 0
        while engine.has_unfinished_requests():
            iterations += 1
            assert iterations < 1000, "batch queue failed to drain"
            for output in engine.step():
                current = list(output.outputs[0].token_ids)
                previous = token_ids[output.request_id]
                assert current[:len(previous)] == previous, "output history changed"
                times[output.request_id].extend(
                    [executor.clock.now_ps] * (len(current) - len(previous)))
                token_ids[output.request_id] = current
        assert all(tokens == [512] * 8 for tokens in token_ids.values())
        assert all(len(values) == 8 for values in times.values())
        assert executor.step_results
        assert any(any(request.phase is RequestPhase.PREFILL
                       and request.context_length < 48 for request in record.scheduled)
                   for record in executor.step_records), "no chunked prefill"
        assert all(len(outcome.stages) == pp for outcome in sink.outcomes)
        assert sum(end - start for start, end in executor._pipeline.layer_ranges) == 5
        # Decode timing is reported after each request's first token. TTFT is
        # retained as an unscored initialization diagnostic, not a PD study.
        metrics = [{"request_id": name, "setup_ttft_ps": values[0],
                    "decode_tpot_ps": (values[-1] - values[0]) / (len(values) - 1),
                    "token_completion_ps": values}
                   for name, values in times.items()]
        decode_steps = [(record, result) for record, result in zip(
            executor.step_records, executor.step_results)
            if record.scheduled and all(request.phase is RequestPhase.DECODE
                                        for request in record.scheduled)]
        assert decode_steps, "no pure decode step"
        # A second group implements the decode-only experiment: prefill all
        # requests, retain their KV allocation, then open a common window.
        scheduler = engine.engine_core.engine_core.scheduler
        resident = [engine.add_request(
            f"resident-{index}", {"prompt_token_ids": [20 + index] * 48},
            SamplingParams(max_tokens=128, temperature=0, ignore_eos=True, detokenize=False),
        ) for index in range(4)]
        resident_steps = 0
        while not all(scheduler.requests[name].num_output_tokens >= 1 for name in resident):
            engine.step()
            resident_steps += 1
            assert resident_steps < 100
        initial_counts = [scheduler.requests[name].num_output_tokens for name in resident]
        initial_contexts = [scheduler.requests[name].num_computed_tokens for name in resident]
        assert all(context >= 48 for context in initial_contexts)
        assert all(all(scheduler.kv_cache_manager.get_block_ids(name)) for name in resident)
        window_start = executor.clock.now_ps
        first_record = len(executor.step_records)
        while not all(scheduler.requests[name].num_output_tokens >= initial + 8
                      for name, initial in zip(resident, initial_counts)):
            engine.step()
            resident_steps += 1
            assert resident_steps < 200
        window_ps = executor.clock.now_ps - window_start
        increments = [scheduler.requests[name].num_output_tokens - initial
                      for name, initial in zip(resident, initial_counts)]
        assert all(scheduler.requests[name].num_preemptions == 0 for name in resident)
        window_records = executor.step_records[first_record:]
        assert all(request.phase is RequestPhase.DECODE
                   for record in window_records for request in record.scheduled)
        resident_metrics = {
            "window_ps": window_ps, "initial_computed_tokens": initial_contexts,
            "decoded_tokens_per_request": increments,
            "tokens_per_simulated_second": sum(increments) * 1e12 / window_ps,
            "mean_request_token_interval_ps": window_ps / statistics.mean(increments),
            "no_prefill_or_preemption_in_window": True,
        }
        return {
            "pp": pp, "tp": tp, "rnic_rate_bps": rate,
            "layer_ranges": executor._pipeline.layer_ranges,
            "rank_map": sink.rank_map, "request_metrics": metrics,
            "median_decode_tpot_ps": statistics.median(row["decode_tpot_ps"] for row in metrics),
            "completion_ps": executor.clock.now_ps,
            "step_count": len(executor.step_results),
            "pure_decode_steps": len(decode_steps),
            "resident_decode": resident_metrics,
            "max_queued_steps": max(sum(
                other.record.virtual_time_ps <= outcome.record.virtual_time_ps
                < other.step_result.completed_at_ps for other in sink.outcomes)
                for outcome in sink.outcomes),
            "stage_outcomes": [
                {"step_index": outcome.record.step_index,
                 "release_ps": outcome.record.virtual_time_ps,
                 "completion_ps": outcome.step_result.completed_at_ps,
                 "stages": [vars(stage) for stage in outcome.stages]}
                for outcome in sink.outcomes],
            "assigned_ids": assigned,
            "fatal_guards": "pass",
        }
    finally:
        engine.engine_core.shutdown()
        reset_configuration()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = args.output_dir / "tiny-llama"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 5,
        "num_attention_heads": 8, "num_key_value_heads": 8,
        "head_dim": 16, "vocab_size": 1024, "max_position_embeddings": 256,
        "torch_dtype": "bfloat16", "rms_norm_eps": 1e-5,
        "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
    }) + "\n")
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
    version = importlib.metadata.version("vllm")
    assert version.split("+")[0] == "0.27.1", version
    report = {
        "schema": "simllm-vllm-pipeline-study-v1", "expectations_commit": FREEZE,
        "vllm_version": version, "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "evidence_class": "real-engine declared-runtime; no hardware calibration",
        "exact_oracles": exact_pipeline_oracles(),
        "cases": [],
    }
    axes = [(2, 1, 100_000_000_000)] if args.single else [
        (pp, tp, rate) for pp in (2, 4) for tp in (1, 2)
        for rate in (100_000_000_000, 200_000_000_000)] + [(2, 4, 100_000_000_000)]
    for pp, tp, rate in axes:
        row = run_case(model, pp, tp, rate)
        report["cases"].append(row)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in (
            "pp", "tp", "rnic_rate_bps", "median_decode_tpot_ps", "completion_ps")}), flush=True)
    if not args.single:
        for pp in (2, 4):
            for tp in (1, 2):
                slow, fast = [row for row in report["cases"] if row["pp"] == pp and row["tp"] == tp]
                assert slow["completion_ps"] > fast["completion_ps"]
                assert slow["median_decode_tpot_ps"] > fast["median_decode_tpot_ps"]
                assert slow["resident_decode"]["window_ps"] > fast["resident_decode"]["window_ps"]
        report["bandwidth_relation_instances"] = 4
    report["status"] = "PASS"
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PIPELINE_STUDY_PASS", flush=True)


if __name__ == "__main__":
    main()
