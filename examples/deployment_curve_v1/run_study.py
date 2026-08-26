"""Run the CORE-54 deployment-curve scaffold."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any

from curve_tools import (
    fit_tunable_constants,
    fraction_json,
    load_json,
    propagate_curve_interval,
    score_held_out_predictions,
    sha256,
    validate_anchor_freeze,
    validate_constant_declarations,
    write_json,
)

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
ANCHOR_PATH = STUDY_DIR / "expectations.json"
DEFAULT_CONFIG_PATH = STUDY_DIR / "dry_run_config.json"
ANCHOR_FREEZE_COMMIT = "629fc7bdd93750e3426ac4d46cf9c5bbdf9d9175"
ANCHOR_FREEZE_SHA256 = "b1a918ed02329a242d033943fb18b93fd9be8fdaa18093477e6abb8298540df5"
CONFIG_SCHEMA = "simllm-deployment-curve-study-config-v1"
RESULT_SCHEMA = "simllm-deployment-curve-study-result-v1"
CURVE_SCHEMA = "simllm-deployment-curve-v1"
POINT_SCHEMA = "simllm-deployment-curve-point-v1"
PS_PER_SECOND = 1_000_000_000_000
RUN_ROOT_ENV = "SIMLLM_CORE54_RUN_ROOT"


def render_cli_path(path: PurePath) -> str:
    """Render executed paths with POSIX separators on every host."""

    return path.as_posix()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        raise SystemExit("the deployment-curve run requires a clean tracked worktree")


def _require_anchor_commit() -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ANCHOR_FREEZE_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit("the anchor freeze commit is not an ancestor of HEAD")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _validate_arrangement(
    arrangement: dict[str, Any],
    pool: dict[str, Any],
    index: int,
) -> None:
    expected = {
        "tensor_parallel",
        "expert_parallel",
        "data_parallel",
        "pipeline_parallel",
    }
    for role in ("prefill", "decode"):
        values = arrangement.get(role)
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(
                f"configurations[{index}].arrangement.{role} must declare {sorted(expected)}"
            )
        for name, value in values.items():
            _positive_int(value, f"configurations[{index}].arrangement.{role}.{name}")
        if values["tensor_parallel"] != pool["gpus_per_node"]:
            raise ValueError(f"{role} tensor parallelism must equal GPUs per node")


def validate_study_config(config: dict[str, Any]) -> None:
    """Validate complete declared configurations without importing a frontend."""

    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("study configuration schema disagrees")
    study = config.get("study")
    if not isinstance(study, dict):
        raise TypeError("study metadata must be an object")
    if study.get("classification") not in {"dry-run", "scored"}:
        raise ValueError("study classification must be dry-run or scored")
    if (study["classification"] == "scored") != (study.get("scored_flagship") is True):
        raise ValueError("scored_flagship must agree with the classification")
    prompt_fixture = Path(_nonblank(config.get("prompt_fixture"), "prompt_fixture"))
    if prompt_fixture.is_absolute() or ".." in prompt_fixture.parts:
        raise ValueError("prompt_fixture must be a repository-relative path")
    validate_constant_declarations(config.get("constants", []))

    configurations = config.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("at least one deployment configuration is required")
    ids = []
    for index, deployment in enumerate(configurations):
        if not isinstance(deployment, dict):
            raise TypeError(f"configurations[{index}] must be an object")
        ids.append(_nonblank(deployment.get("configuration_id"), "configuration_id"))
        _nonblank(deployment.get("legend_label"), "legend_label")
        if deployment.get("framework") != "vllm":
            raise ValueError("the scaffold currently supports framework='vllm'")
        pool = deployment.get("pool")
        if not isinstance(pool, dict) or set(pool) != {
            "prefill_nodes",
            "decode_nodes",
            "gpus_per_node",
        }:
            raise ValueError("pool must declare prefill, decode and GPU node counts")
        for name, value in pool.items():
            _positive_int(value, f"configurations[{index}].pool.{name}")
        arrangement = deployment.get("arrangement")
        if not isinstance(arrangement, dict):
            raise TypeError("arrangement must be an object")
        _validate_arrangement(arrangement, pool, index)

        model = deployment.get("model")
        if not isinstance(model, dict):
            raise TypeError("model must be an object")
        for key in ("id", "revision", "column"):
            _nonblank(model.get(key), f"configurations[{index}].model.{key}")
        dims = model.get("dims")
        if not isinstance(dims, dict) or not dims:
            raise ValueError("model dims must be a nonempty object")
        for name, value in dims.items():
            _positive_int(value, f"configurations[{index}].model.dims.{name}")
        geometry = model.get("kv_geometry")
        if not isinstance(geometry, dict) or set(geometry) != {
            "num_layers",
            "num_kv_heads",
            "head_size",
            "element_bytes",
        }:
            raise ValueError("model kv_geometry is incomplete")
        for name, value in geometry.items():
            _positive_int(value, f"configurations[{index}].kv_geometry.{name}")

        pricing = deployment.get("pricing")
        if not isinstance(pricing, dict) or pricing.get("mode") != "bootstrap":
            raise ValueError("the scaffold currently supports pricing mode 'bootstrap'")
        if pricing.get("collective_fixed_cost_arm") != "lower":
            raise ValueError("the granite dry run requires the lower collective arm")
        uncertainty = pricing.get("record_uncertainty_relative")
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, int | float):
            raise TypeError("record uncertainty must be numeric")
        if not math_is_finite_nonnegative(float(uncertainty)) or uncertainty >= 1:
            raise ValueError("record uncertainty must be finite and in [0, 1)")

        requests = deployment.get("requests")
        if not isinstance(requests, dict):
            raise TypeError("requests must be an object")
        loads = requests.get("offered_load_requests_per_second")
        if not isinstance(loads, list) or not loads:
            raise ValueError("offered load sweep must be a nonempty list")
        if any(type(load) is not int or load <= 0 for load in loads):
            raise ValueError("offered loads must be positive integers")
        if loads != sorted(loads) or len(loads) != len(set(loads)):
            raise ValueError("offered loads must be unique and increasing")
        if any(PS_PER_SECOND % load for load in loads):
            raise ValueError("offered loads must map to exact picosecond interarrivals")
        for name in (
            "prompt_tokens",
            "requests_per_point",
            "decode_output_tokens_per_request",
            "max_model_len",
            "max_num_seqs",
            "num_gpu_blocks_override",
            "token_id",
        ):
            _positive_int(requests.get(name), f"configurations[{index}].requests.{name}")
    if len(ids) != len(set(ids)):
        raise ValueError("configuration IDs must be unique")


def math_is_finite_nonnegative(value: float) -> bool:
    """Keep numeric validation import-light for check-only mode."""

    return math.isfinite(value) and value >= 0


def check_registry(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the freeze and configuration without importing vLLM."""

    freeze = load_json(ANCHOR_PATH)
    config = load_json(config_path)
    validate_anchor_freeze(freeze)
    validate_study_config(config)
    if sha256(ANCHOR_PATH) != ANCHOR_FREEZE_SHA256:
        raise SystemExit("anchor freeze digest disagrees")
    _require_anchor_commit()
    return freeze, config


def _prompt_tokens(path: Path, required: int) -> tuple[int, ...]:
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("row_type") == "request":
            tokens = tuple(row["input_token_ids"])
            if len(tokens) < required:
                raise RuntimeError("prompt fixture is shorter than the declared prompt")
            return tokens
    raise RuntimeError("prompt fixture contains no request row")


def _model_dims(values: dict[str, Any]) -> Any:
    from simllm.compute import ModelDims

    return ModelDims(**values)


def _session_config(
    deployment: dict[str, Any],
    workdir: Path,
    constants: dict[str, dict[str, Any]],
) -> Any:
    from simllm.adapters.vllm.pd_session import VllmPdSessionConfig
    from simllm.compute import GPU_ENVELOPES, RooflineProvider
    from simllm.core import DeclaredKvHandoffPolicy, KvHandoffGeometry

    pool = deployment["pool"]
    model = deployment["model"]
    pricing = deployment["pricing"]
    requests = deployment["requests"]
    constant = constants[pricing["kv_handoff_constant_id"]]
    duration_ps = constant["selected"]
    if type(duration_ps) is not int:
        raise ValueError("the live KV handoff constant must select an integer ps value")
    return VllmPdSessionConfig(
        model=model["id"],
        model_revision=model["revision"],
        workdir=workdir,
        dims=_model_dims(model["dims"]),
        handoff_geometry=KvHandoffGeometry(**model["kv_geometry"]),
        handoff_policy=DeclaredKvHandoffPolicy(duration_ps),
        prefill_engines=pool["prefill_nodes"],
        decode_engines=pool["decode_nodes"],
        tensor_parallel_size=pool["gpus_per_node"],
        max_model_len=requests["max_model_len"],
        num_gpu_blocks_override=requests["num_gpu_blocks_override"],
        max_num_seqs=requests["max_num_seqs"],
        token_id=requests["token_id"],
        provider=RooflineProvider(efficiency=pricing["roofline_efficiency"]),
        gpu=GPU_ENVELOPES[pricing["gpu"]],
    )


def _request_id(configuration_id: str, load: int, index: int) -> str:
    return f"{configuration_id}-load{load}-request{index}"


def _record_bounds(point: dict[str, Any], relative: float) -> dict[str, Any]:
    from curve_tools import as_fraction

    width = Fraction(str(relative))
    throughput = as_fraction(
        point["aggregated_output_throughput_tokens_per_second"],
        "curve.throughput",
    )
    delay = as_fraction(point["per_token_request_delay_ps"], "curve.delay")
    return {
        "source_id": "roofline-bootstrap-record-bound",
        "aggregated_output_throughput_tokens_per_second": {
            "lower": fraction_json(throughput * (1 - width)),
            "upper": fraction_json(throughput * (1 + width)),
        },
        "per_token_request_delay_ps": {
            "lower": fraction_json(delay * (1 - width)),
            "upper": fraction_json(delay * (1 + width)),
        },
    }


def _run_configuration(
    deployment: dict[str, Any],
    run_dir: Path,
    prompt: tuple[int, ...],
    declarations: list[dict[str, Any]],
) -> dict[str, Any]:
    from simllm.adapters.vllm.pd_session import (
        VllmDisaggregatedSession,
        VllmPdCurveRecord,
        VllmPdRequest,
    )

    constants = validate_constant_declarations(declarations)
    requests_config = deployment["requests"]
    prompt_length = requests_config["prompt_tokens"]
    output_tokens = requests_config["decode_output_tokens_per_request"]
    points = []
    batch_observations = []
    construction = []

    def observe_engine(engine: Any) -> None:
        construction.append(
            {
                "engine_id": engine.engine_id,
                "role": engine.role.value,
                "simulated_worker_count": engine.simulated_worker_count,
                "construction_seconds": engine.construction_seconds,
            }
        )

    configuration_id = deployment["configuration_id"]
    with VllmDisaggregatedSession(
        _session_config(deployment, run_dir / configuration_id, constants),
        construction_observer=observe_engine,
    ) as session:
        for load in requests_config["offered_load_requests_per_second"]:
            interarrival_ps = PS_PER_SECOND // load
            admitted_start = session.clock.now_ps
            requests = tuple(
                VllmPdRequest(
                    request_id=_request_id(configuration_id, load, index),
                    prompt_token_ids=prompt[:prompt_length],
                    decode_output_tokens=output_tokens,
                    admitted_at_ps=admitted_start + index * interarrival_ps,
                )
                for index in range(requests_config["requests_per_point"])
            )
            result = session.run_requests(requests)
            point = result.curve_point(Fraction(load))
            point_json = point.to_json()
            uncertainty = {
                "record_bounds": _record_bounds(
                    point_json,
                    deployment["pricing"]["record_uncertainty_relative"],
                ),
                "distribution_spreads": deployment["uncertainty"]["distribution_spreads"],
                "tuned_constant_envelopes": deployment["uncertainty"]["tuned_constant_envelopes"],
            }
            point_json["uncertainty"] = propagate_curve_interval(
                point_json,
                uncertainty,
                declarations,
            )
            points.append((point, point_json))
            batch_observations.append(
                {
                    "offered_load_requests_per_second": load,
                    "admissions": len(requests),
                    "terminals": len(result.requests),
                    "terminal_output_tokens": sum(
                        len(row.decode_token_ids) for row in result.requests
                    ),
                    "maximum_prefill_batch_size": result.maximum_prefill_batch_size,
                    "maximum_decode_batch_size": result.maximum_decode_batch_size,
                }
            )

    record = VllmPdCurveRecord(
        configuration_id=configuration_id,
        prefill_engines=deployment["pool"]["prefill_nodes"],
        decode_engines=deployment["pool"]["decode_nodes"],
        prompt_tokens=prompt_length,
        points=tuple(point for point, _ in points),
    ).to_json()
    if record["schema"] != CURVE_SCHEMA or any(
        point["schema"] != POINT_SCHEMA for point in record["points"]
    ):
        raise RuntimeError("session curve schema disagrees with the established contract")
    record["points"] = [point_json for _, point_json in points]
    record.update(
        {
            "configuration_label": deployment["legend_label"],
            "framework": deployment["framework"],
            "pool": deployment["pool"],
            "arrangement": deployment["arrangement"],
            "model_column": deployment["model"]["column"],
            "pricing_mode": deployment["pricing"]["mode"],
            "evidence_label": "DRY RUN, not the CORE-54 flagship result",
            "construction": construction,
            "batch_observations": batch_observations,
        }
    )
    return record


def run_observation(
    freeze: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Drive every declared deployment configuration through one session."""

    prompt_path = REPOSITORY_ROOT / config["prompt_fixture"]
    required = max(
        deployment["requests"]["prompt_tokens"] for deployment in config["configurations"]
    )
    prompt = _prompt_tokens(prompt_path, required)
    curves = [
        _run_configuration(deployment, run_dir, prompt, config["constants"])
        for deployment in config["configurations"]
    ]
    fit_rows = config["calibration_fit_rows"]
    if fit_rows:
        fit = fit_tunable_constants(freeze, config["constants"], fit_rows)
    else:
        fit = {
            "status": "NOT_RUN",
            "reason": "the granite bootstrap dry run does not calibrate constants",
            "accessed_anchor_ids": [],
        }
    predictions = config["held_out_predictions"]
    if predictions:
        score = score_held_out_predictions(freeze, predictions)
    else:
        score = {
            "status": "NOT_SCORED",
            "reason": "the granite bootstrap dry run is not the DeepSeek flagship",
            "accessed_anchor_ids": [],
        }
    return {"curves": curves, "constant_fit": fit, "held_out_score": score}


def _validate_run_dir(run_dir: Path) -> None:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise SystemExit(f"{RUN_ROOT_ENV} must name the external run root")
    root = Path(configured).resolve()
    try:
        run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"run directory must remain under {RUN_ROOT_ENV}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze, config = check_registry(args.config)
    if args.check_only:
        print(
            f"validated {len(freeze['anchors'])} anchors, "
            f"{len(config['constants'])} bounded constants and "
            f"{len(config['configurations'])} deployment configuration(s); "
            "no frontend imported and no artifact produced"
        )
        return
    if args.run_dir is None:
        raise SystemExit("--run-dir is required unless --check-only is selected")
    _require_clean_worktree()
    _validate_run_dir(args.run_dir)
    if args.run_dir.exists():
        raise SystemExit(f"selected run directory already exists: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    observation = run_observation(freeze, config, args.run_dir)
    result = {
        "schema": RESULT_SCHEMA,
        "classification": config["study"]["classification"],
        "label": config["study"]["label"],
        "scored_flagship": config["study"]["scored_flagship"],
        "provenance": {
            "run_head": _git_head(),
            "anchor_freeze_commit": ANCHOR_FREEZE_COMMIT,
            "anchor_freeze_sha256": ANCHOR_FREEZE_SHA256,
            "configuration_sha256": sha256(args.config),
            "prompt_fixture": config["prompt_fixture"],
            "prompt_fixture_sha256": sha256(REPOSITORY_ROOT / config["prompt_fixture"]),
        },
        "declared_configurations": config["configurations"],
        "constant_declarations": config["constants"],
        **observation,
    }
    write_json(args.run_dir / "result.json", result)
    print(
        f"DRY RUN wrote {len(result['curves'])} curve record(s) to "
        f"{render_cli_path(args.run_dir / 'result.json')}; not scored"
    )


if __name__ == "__main__":
    main()
