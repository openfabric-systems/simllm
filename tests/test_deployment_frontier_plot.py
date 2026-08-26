import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.deployment_frontier_v1.frontier import load_expectations, operating_point
from examples.deployment_frontier_v1.plot_study import prepare_plot


def _synthetic_result():
    frozen = load_expectations()
    points = []
    for configuration in frozen["configurations"]:
        for batch in frozen["batch_per_gpu_sweep"]:
            analytical = 10_000_000_000 + batch
            simulated = analytical + 100
            points.append(
                {
                    "configuration_id": configuration["id"],
                    "configuration_label": configuration["label"],
                    "batch_per_gpu": batch,
                    "analytical_operating_point": operating_point(
                        batch_per_gpu=batch,
                        step_time_ps=analytical,
                    ),
                    "simulated_operating_point": operating_point(
                        batch_per_gpu=batch,
                        step_time_ps=simulated,
                    ),
                    "accounting": {
                        "inter_node_attributed_ps": 40,
                        "intra_node_attributed_ps": 60,
                    },
                    "bottleneck": {"classification": "intra-node"},
                }
            )
    return {
        "schema": "simllm-deployment-frontier-result-v1",
        "status": "PASS",
        "plot_contract": frozen["plot_contract"],
        "published_context": frozen["published_context"],
        "intra_node_candidate_disclosure": frozen["network_inputs"]["intra_node"][
            "cross_architecture_use"
        ],
        "points": points,
    }


def test_plot_description_keeps_lines_and_dots_separate():
    plot = prepare_plot(_synthetic_result())

    assert len(plot["curves"]) == 3
    assert all(len(curve["points"]) == 6 for curve in plot["curves"])
    for curve in plot["curves"]:
        for point in curve["points"]:
            assert point["analytical_x"] > point["simulated_x"] > 0
            assert point["analytical_y"] > point["simulated_y"] > 0


def test_published_marker_and_y_only_anchor_retain_their_contract():
    plot = prepare_plot(_synthetic_result())

    assert plot["paired_marker"] == {
        "label": "Published SGLang H100 EP72 standard decode",
        "x": 22_282 / 256,
        "y": 22_282 / 8,
    }
    assert plot["y_only_anchor"]["y"] == 14_800 / 8
