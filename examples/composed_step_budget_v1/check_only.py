"""Validate every frozen input and arithmetic fact of the composed-step study.

This entry point produces nothing. It exists so the freeze can be exercised
before the measurement harness exists, as the validation discipline requires:
it re-derives every frozen literal from the installed calibrations, proves that
the additive and overlapped intervals are disjoint, and refuses a missing model
snapshot, a wrong vLLM version, an absent backend binary or an absent GOAL
converter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / MODEL_REVISION
)
REQUIRED_VLLM_VERSION = "0.26.0"


def load_expectations() -> dict[str, Any]:
    """Return the frozen contract as plain data."""

    return json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))


def _quantize_ps(value_ps: int) -> int:
    """Return the whole-nanosecond GOAL enclosure of a picosecond duration."""

    return -(-value_ps // 1000) * 1000


def check_arithmetic(frozen: dict[str, Any]) -> dict[str, Any]:
    """Re-derive every frozen number and prove the intervals discriminate."""

    from simllm.compute import HostInitiationModel
    from simllm.traffic import B200_NCCL_2_27_LOCAL_PROFILE

    constants = frozen["constants_ps"]
    collectives = frozen["collectives_per_step"]

    floor_width8 = B200_NCCL_2_27_LOCAL_PROFILE.base_latency_ps(8)
    if floor_width8 != constants["collective_floor_width8"]:
        raise SystemExit("installed width-8 collective floor drifted from the freeze")
    floor_total = collectives * floor_width8
    if floor_total != constants["collective_floor_total"]:
        raise SystemExit("frozen collective floor total is not 48 base charges")

    envelope = B200_NCCL_2_27_LOCAL_PROFILE.endpoint_byte_bounds(8)
    if list(envelope) != frozen["endpoint_envelope_bytes"]:
        raise SystemExit("installed width-8 endpoint envelope drifted from the freeze")

    graph = HostInitiationModel.turing_cuda_graph(440)
    eager = HostInitiationModel.turing_eager_host(567)
    if graph.point_ps_per_launch != constants["graph_point_per_launch"]:
        raise SystemExit("installed graph per-launch point drifted from the freeze")
    if eager.point_ps_per_launch != constants["eager_point_per_launch"]:
        raise SystemExit("installed eager per-launch point drifted from the freeze")

    compute_ps = constants["accepted_decode_compute"]
    quantized = {
        "on-graph440-400g": _quantize_ps(max(compute_ps, graph.launch_floor_ps)),
        "on-graph440-200g": _quantize_ps(max(compute_ps, graph.launch_floor_ps)),
        "on-eager567-400g": _quantize_ps(max(compute_ps, eager.launch_floor_ps)),
    }
    if quantized["on-graph440-400g"] != constants["quantized_graph440"]:
        raise SystemExit("frozen graph quantized service is not the derived value")
    if quantized["on-eager567-400g"] != constants["quantized_eager567"]:
        raise SystemExit("frozen eager quantized service is not the derived value")
    if (
        constants["quantized_eager567"] - constants["quantized_graph440"]
        != constants["quantized_delta"]
    ):
        raise SystemExit("frozen host-profile separation is not the quantized delta")

    intervals = frozen["intervals_ps"]
    ideal_lo, ideal_hi = intervals["ideal_compute"]
    network = {
        "on-graph440-400g": intervals["network_400g"],
        "on-eager567-400g": intervals["network_400g"],
        "on-graph440-200g": intervals["network_200g"],
    }
    for label, (net_lo, net_hi) in network.items():
        additive = [
            quantized[label] + floor_total + net_lo,
            quantized[label] + floor_total + net_hi,
        ]
        overlapped = [ideal_lo + floor_total + net_lo, ideal_hi + floor_total + net_hi]
        if additive != intervals["additive"][label]:
            raise SystemExit(f"frozen additive interval for {label} is not derived")
        if overlapped != intervals["overlapped"][label]:
            raise SystemExit(f"frozen overlapped interval for {label} is not derived")
        if additive[0] <= overlapped[1]:
            raise SystemExit(
                f"additive and overlapped intervals for {label} overlap, so the "
                "flagship relation cannot discriminate the two compositions"
            )

    graph_interval = intervals["additive"]["on-graph440-400g"]
    eager_interval = intervals["additive"]["on-eager567-400g"]
    if eager_interval[0] <= graph_interval[1]:
        raise SystemExit(
            "the two 400 Gbit/s additive intervals overlap, so a single "
            "overlapped answer could satisfy both host profiles"
        )

    published = frozen["published_projections_ps"]
    overlapped_point = (
        compute_ps + constants["accepted_decode_network_400g"] + floor_total
    )
    if overlapped_point != published["overlapped"]:
        raise SystemExit("published overlapped projection is not reproducible")
    additive_points = {
        "additive_graph440": constants["quantized_graph440"]
        + constants["accepted_decode_network_400g"]
        + floor_total,
        "additive_eager567": constants["quantized_eager567"]
        + constants["accepted_decode_network_400g"]
        + floor_total,
    }
    for name, value in additive_points.items():
        if value != published[name]:
            raise SystemExit(f"published {name} projection is not reproducible")
    factor = additive_points["additive_eager567"] / overlapped_point
    if not math.isclose(factor, published["disagreement_factor"], rel_tol=1e-9):
        raise SystemExit("published disagreement factor is not reproducible")

    coverage = frozen["traffic_coverage"]
    added = coverage["layers"] * coverage["all_reduces_per_layer"] * floor_width8
    if added != coverage["added_floor_ps"]:
        raise SystemExit("tensor-parallel all-reduce addition is not derived")
    if coverage["overlap_counterfactual_hidden_ps"] * 2 != floor_total:
        raise SystemExit("overlap counterfactual is not half the collective floor")
    share_lo, share_hi = coverage["share_band"]
    for point in additive_points.values():
        share = added / point
        if not share_lo <= share <= share_hi:
            raise SystemExit("traffic-coverage share band excludes its own projection")

    return {
        "additive_points_ps": additive_points,
        "collective_floor_total_ps": floor_total,
        "disagreement_factor": factor,
        "endpoint_envelope_bytes": list(envelope),
        "overlapped_point_ps": overlapped_point,
        "quantized_service_ps": quantized,
        "traffic_coverage_added_ps": added,
    }


def check_inputs(args: argparse.Namespace) -> None:
    """Refuse a missing snapshot, engine, backend binary or converter."""

    import importlib.metadata

    model = args.cache_dir / MODEL_RELATIVE_PATH
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"pinned model snapshot is missing: {model}")
    if importlib.metadata.version("vllm") != REQUIRED_VLLM_VERSION:
        raise SystemExit(f"this study requires vLLM {REQUIRED_VLLM_VERSION}")
    if not args.htsim_rnic.is_file() or not os.access(args.htsim_rnic, os.X_OK):
        raise SystemExit(f"htsim_rnic is missing or not executable: {args.htsim_rnic}")
    converter = os.environ.get("SIMLLM_TXT2BIN")
    if converter is not None:
        path = Path(converter)
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"SIMLLM_TXT2BIN is not executable: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--htsim-rnic", type=Path, required=True)
    parser.add_argument(
        "--arithmetic-only",
        action="store_true",
        help="skip the model, engine and backend checks",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    frozen = load_expectations()
    derived = check_arithmetic(frozen)
    if not args.arithmetic_only:
        check_inputs(args)
    print(json.dumps(derived, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
