"""Exercise protocol uncertainty through sampled-token metrics on an original graph.

The fixed compute service isolates the collective term. These token metrics
are integration evidence, not a calibrated inference performance forecast.
"""

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simllm.backends import HtsimRequestMetricReducer, HtsimStepSink, HtsimStepSinkConfig
from simllm.compute import ComputeProvider, DurationEstimate, ModelDims
from simllm.core import RequestPhase, ScheduledRequest, StepRecord
from simllm.placement import PlacementManifest, RankPlacement
from simllm.traffic import CollectiveLatencyProfile, NcclRingProtocolModel


class ControlledCompute(ComputeProvider):
    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=200_000_000, bound="compute")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    models = [NcclRingProtocolModel.from_json(v) for v in json.loads(args.models.read_text())]
    dims = ModelDims(
        num_layers=1,
        hidden_size=4096,
        intermediate_size=8192,
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        vocab_size=256,
        dtype_bytes=4,
    )
    rows = []
    for model in models:
        for tokens in (32, 128):
            for phase in (RequestPhase.PREFILL, RequestPhase.DECODE):
                scheduled = (
                    [ScheduledRequest("r0", phase, num_new_tokens=tokens, context_length=tokens)]
                    if phase is RequestPhase.PREFILL
                    else [
                        ScheduledRequest(f"r{i}", phase, num_new_tokens=1, context_length=32)
                        for i in range(tokens)
                    ]
                )
                item = StepRecord(
                    step_index=0, virtual_time_ps=0, scheduled=scheduled, num_sampled=len(scheduled)
                )
                results = {}
                for arm in ("lower", "central", "upper"):
                    profile = CollectiveLatencyProfile(
                        profile_id=model.model_id + "-" + arm,
                        bandwidth_bytes_per_second=model.endpoint_rate_bytes_per_second,
                        participant_latency_ps=((model.width, model.startup_ps),),
                        source_payload_bytes_min=model.payload_min_bytes,
                        source_payload_bytes_max=model.payload_max_bytes,
                        propagation_reference_ps=0,
                        protocol_models=((model.width, model),),
                        protocol_arm=arm,
                    )
                    workdir = args.output / model.model_id / str(tokens) / phase.value / arm
                    instance = HtsimStepSink(
                        HtsimStepSinkConfig(
                            profile="rnic-nn-fluid",
                            tp_ranks=tuple(range(model.width)),
                            dims=dims,
                            workdir=workdir,
                            provider=ControlledCompute(),
                            collective_latency_profile=profile,
                            placement_manifest=PlacementManifest(
                                ranks=[
                                    RankPlacement(global_rank=r, hostname="node", local_rank=r)
                                    for r in range(model.width)
                                ]
                            ),
                        ),
                        request_metric_reducer=HtsimRequestMetricReducer(
                            {r.request_id: 0 for r in scheduled}
                        ),
                    )
                    first = instance(item)
                    result = (
                        first
                        if phase is RequestPhase.PREFILL
                        else instance(
                            replace(item, step_index=1, virtual_time_ps=first.completed_at_ps)
                        )
                    )
                    metric = result.request_metrics[0]
                    results[arm] = metric
                    estimate = model.predict(tokens * 4096 * 4)
                    rows.append(
                        {
                            "architecture": model.architecture,
                            "width": model.width,
                            "phase": phase.value,
                            "payload_bytes": tokens * 4096 * 4,
                            "arm": arm,
                            "collective_us": getattr(estimate, arm + "_ps") / 1e6,
                            "ttft_us": metric.ttft_ps / 1e6,
                            "tpot_us": ""
                            if metric.tpot_ps is None
                            else float(metric.tpot_ps / 1e6),
                            "step_us": result.step_latency_ps / 1e6,
                        }
                    )
                estimate = model.predict(tokens * 4096 * 4)
                for arm in ("lower", "upper"):
                    expected = 2 * (getattr(estimate, arm + "_ps") - estimate.central_ps)
                    assert results[arm].ttft_ps - results["central"].ttft_ps == expected
                    if phase is RequestPhase.DECODE:
                        assert results[arm].tpot_ps - results["central"].tpot_ps == expected
    with (args.output / "live_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "simllm-nccl-protocol-live-v1",
        "integrity": "valid",
        "compute_service_ps_per_kernel": 200000000,
        "metric_rows": len(rows),
        "relation": "two collectives per graph; TTFT and TPOT arm differences equal twice the modeled collective difference",
        "scope": "integration experiment with fixed compute; not inference accuracy evidence",
    }
    (args.output / "live_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
