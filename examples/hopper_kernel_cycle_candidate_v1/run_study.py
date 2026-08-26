"""Compile the local Hopper candidate from byte-verified retained evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simllm.calibration.canonical import canonical_sha256, sha256_bytes
from simllm.calibration.kernel_cycle_lut import (
    KERNEL_CYCLE_LUT_SCHEMA,
    compile_device_service_entries,
    compile_profile_table,
    validate_kernel_cycle_lut,
)
from simllm.compute.provider import ProfileTableProvider

PS_PER_SECOND = 1_000_000_000_000
INPUT_PATH = Path(__file__).with_name("retained_evidence.json")
FREEZE_PATH = Path(__file__).with_name("expectations.json")
GRANITE_SM_HZ = 1_980_000_000
GH200_MEMORY_HZ = 2_619_000_000
A100_REPORT_SHA256 = "95c56afd0cf5f974d748a9789129b2594a765cdbcc4170a76e3e9ba3b75e95f8"
GH200_REPORT_SHA256 = "8c4f21f0fdb99fd6b007b21fa3a631523294be616f4078b08d2a1934507cb798"
PROJECTION_SHA256 = "ee154ed5f07c104269df9cf60d8730b8c6dced0ccf619fb7ff7146ec2ddfd5a2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_paths(
    inputs: dict[str, Any],
    evidence_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    definitions: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for source in inputs["sources"]:
        name = str(source["name"])
        root = evidence_root if source["root"] == "evidence" else REPO_ROOT
        path = root / str(source["path"])
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if len(data) != int(source["bytes"]):
            raise ValueError(f"{name}: expected {source['bytes']} bytes, found {len(data)}")
        if digest != source["sha256"]:
            raise ValueError(f"{name}: expected sha256 {source['sha256']}, found {digest}")
        definitions[name] = source
        paths[name] = path
    if list(definitions) != sorted(definitions):
        raise ValueError("retained source names must be sorted")
    return definitions, paths


def _csv_rows(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row[key]: row for row in rows}


def _round_decimal(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _verify_granite_cells(
    inputs: dict[str, Any],
    source_paths: dict[str, Path],
) -> None:
    row_cache: dict[str, dict[str, dict[str, str]]] = {}
    for cell in inputs["granite_cells"]:
        cell_id = str(cell[0])
        repeat_source = str(cell[9])
        if repeat_source not in row_cache:
            row_cache[repeat_source] = _csv_rows(source_paths[repeat_source], "cell")
        row = row_cache[repeat_source][f"baseline_{cell_id.replace('-', '_')}"]
        service_ps = _round_decimal(
            Decimal(row["all_clock_median_nsys_service_ms"]) * Decimal(1_000_000_000)
        )
        cv_ppm = _round_decimal(Decimal(row["all_clock_trimmed_cv"]) * Decimal(1_000_000))
        if service_ps != int(cell[5]):
            raise ValueError(f"{cell_id}: retained service changed to {service_ps} ps")
        if cv_ppm != int(cell[6]):
            raise ValueError(f"{cell_id}: retained CV changed to {cv_ppm} ppm")
        if int(row["observed_repetitions"]) != int(cell[7]):
            raise ValueError(f"{cell_id}: retained replay count changed")


def _verify_deepseek_cells(
    inputs: dict[str, Any],
    source_paths: dict[str, Path],
) -> None:
    prefill = _csv_rows(source_paths["deepseek-prefill-cell-summary"], "cell")
    decode = _csv_rows(source_paths["deepseek-decode-summary"], "cell")
    for cell in inputs["deepseek_cells"]:
        cell_id = str(cell[0])
        csv_cell = str(cell[7])
        if cell[1] == "prefill":
            service_ps = int(prefill[csv_cell]["compute_ns"]) * 1_000
        else:
            service_ps = _round_decimal(
                Decimal(decode[csv_cell]["compute_rank_step_ns"]) * Decimal(1_000)
            )
        if service_ps != int(cell[4]):
            raise ValueError(f"{cell_id}: retained service changed to {service_ps} ps")


def _verify_projection(path: Path) -> None:
    projection = _read_json(path)
    if projection["model"]["geometry"]["layers"] != 61:
        raise ValueError("DeepSeek projection no longer declares 61 layers")
    case_ids = {
        case["case_id"]
        for unit in projection["units"]
        for case in unit["case_projections"]
    }
    required = {
        "sglang-prefill-ep32-r16-t16384",
        "sglang-prefill-ep32-r8-t16384",
        "sglang-prefill-ep32-r4-t16384",
        "sglang-decode-ep72-b32-c2000",
        "sglang-decode-ep72-mtp-b16-c4000",
    }
    if not required <= case_ids:
        raise ValueError(f"DeepSeek projection missing cases {sorted(required - case_ids)}")


def _lookup_shape(pool: str, primary: int, secondary: int) -> dict[str, Any]:
    if pool == "decode":
        return {
            "batch_size": primary,
            "per_request_kv_lengths": [secondary] * primary,
        }
    return {
        "computed_new_tokens": primary,
        "existing_context_tokens": secondary,
    }


def _evidence(
    *,
    service_class: str,
    split: str,
    sources: list[str],
    source_definitions: dict[str, dict[str, Any]],
    derivation: str | None,
) -> dict[str, Any]:
    return {
        "service_class": service_class,
        "component_class": "DISCLOSED",
        "split": split,
        "source_sha256s": sorted(
            {str(source_definitions[name]["sha256"]) for name in sources}
        ),
        "derivation": derivation,
    }


def _aggregate_kernel(
    service_ps: int,
    sm_hz: int,
    memory_hz: int,
    service_class: str,
) -> dict[str, Any]:
    elapsed_cycles = service_ps * sm_hz // PS_PER_SECOND
    compute_ps = -(-(elapsed_cycles * PS_PER_SECOND) // sm_hz)
    method = (
        "Retained Nsys additive noncollective service encoded as elapsed SM-clock cycles; "
        "component attribution remains a disclosed A100 bound because GH200 counters were denied"
    )
    if service_class == "DECLARED":
        method = (
            "Declared 61 / 4 depth-scaled Nsys service encoded as elapsed SM-clock cycles; "
            "component attribution remains a disclosed A100 bound"
        )
    return {
        "kernel_id": "aggregate_noncollective_step_service",
        "name": "aggregate retained Nsys noncollective kernel stream",
        "implementation_class": "unknown",
        "launch_count": 1,
        "measured_elapsed_ps": service_ps,
        "measured_elapsed_sm_cycles": elapsed_cycles,
        "nsys_median_elapsed_ps": service_ps,
        "cross_instrument_ratio_ppm": 1_000_000,
        "components": {
            "compute_sm_cycles": elapsed_cycles,
            "memory": {
                "weight_bytes": None,
                "kv_bytes": None,
                "other_bytes": None,
                "achieved_bandwidth_bytes_per_second": None,
                "observed_memory_clock_hz": memory_hz,
                "service_ps": 0,
            },
            "fixed_overhead_ps": service_ps - compute_ps,
            "method": method,
        },
        "code_object": {
            "ptx_sha256": None,
            "sass_sha256": None,
            "compile_configuration_sha256": None,
        },
    }


def _entry(
    *,
    implementation_id: str,
    key: dict[str, Any],
    service_ps: int,
    sm_hz: int,
    memory_hz: int,
    replay_count: int,
    cv_ppm: int,
    verdict: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "key": key,
        "implementation_id": implementation_id,
        "coverage": "complete-kernel-stream",
        "measured_service_ps": service_ps,
        "observed_clocks": {
            "sm_hz": {"min": sm_hz, "median": sm_hz, "max": sm_hz},
            "memory_hz": {"min": memory_hz, "median": memory_hz, "max": memory_hz},
        },
        "distribution": {
            "replay_count": replay_count,
            "peak_count": 1,
            "peak_centers_ps": [service_ps],
            "trimmed_coefficient_of_variation_ppm": cv_ppm,
            "clock_correlation_ppm": None,
            "verdict": verdict,
        },
        "kernels": [_aggregate_kernel(service_ps, sm_hz, memory_hz, evidence["service_class"])],
        "evidence": evidence,
    }


def _base_key(
    inputs: dict[str, Any],
    model_name: str,
    pool: str,
    launch_mode: str,
    parallelism: dict[str, int],
    shape: dict[str, Any],
) -> dict[str, Any]:
    model = inputs["models"][model_name]
    key = {
        "framework_identity": inputs["framework"],
        "model_identity": model,
        "pool": pool,
        "launch_mode": launch_mode,
        "parallelism": parallelism,
        "shape": shape,
        "input_dependency": (
            "dense-content-independent" if model["family"] == "dense" else "moe-routing-dependent"
        ),
    }
    return key


def _build_granite_entries(
    inputs: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for cell in inputs["granite_cells"]:
        cell_id, mode, pool = str(cell[0]), str(cell[1]), str(cell[2])
        primary, secondary = int(cell[3]), int(cell[4])
        service_ps, cv_ppm, replay_count = int(cell[5]), int(cell[6]), int(cell[7])
        split, repeat_source, kernel_source = str(cell[8]), str(cell[9]), str(cell[10])
        key = _base_key(
            inputs,
            "granite",
            pool,
            mode,
            {
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "data_parallel": 1,
                "expert_parallel": 1,
            },
            _lookup_shape(pool, primary, secondary),
        )
        evidence = _evidence(
            service_class="MEASURED",
            split=split,
            sources=[
                "a100-attribution-report",
                "gh200-lane-report",
                "granite-a100-vs-gh200",
                repeat_source,
                kernel_source,
            ],
            source_definitions=sources,
            derivation=None,
        )
        result.append(
            _entry(
                implementation_id=f"granite-3.0-8b-instruct-vllm-{cell_id}",
                key=key,
                service_ps=service_ps,
                sm_hz=GRANITE_SM_HZ,
                memory_hz=GH200_MEMORY_HZ,
                replay_count=replay_count,
                cv_ppm=cv_ppm,
                verdict=(
                    "tight-single-peak"
                    if mode == "cuda-graph" and pool == "decode"
                    else "insufficient-replays"
                ),
                evidence=evidence,
            )
        )
    return result


def _deepseek_key(
    inputs: dict[str, Any],
    model_name: str,
    cell_id: str,
    pool: str,
    primary: int,
    secondary: int,
    target_parallelism: bool,
) -> dict[str, Any]:
    if pool == "prefill":
        shape = _lookup_shape(pool, 16_384, 0)
        data_parallel = expert_parallel = 32 if target_parallelism else 1
    else:
        shape = _lookup_shape(pool, primary, secondary)
        data_parallel = expert_parallel = 72 if target_parallelism else 1
    key = _base_key(
        inputs,
        model_name,
        pool,
        "cuda-graph",
        {
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "data_parallel": data_parallel,
            "expert_parallel": expert_parallel,
        },
        shape,
    )
    key["routing"] = {
        "availability": "not-captured",
        "expert_loads": None,
        "evidence_sha256": canonical_sha256(
            {
                "cell_id": cell_id,
                "request_count": primary,
                "tokens_per_request_or_context": secondary,
                "token_pattern_seed": 17,
            }
        ),
    }
    return key


def _build_deepseek_entries(
    inputs: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    scale = inputs["depth_projection"]
    for cell in inputs["deepseek_cells"]:
        cell_id, pool = str(cell[0]), str(cell[1])
        primary, secondary, measured_ps, sm_hz = map(int, cell[2:6])
        split = str(cell[6])
        source_names = [
            "a100-attribution-report",
            "deepseek-deployment-projection",
            "gh200-lane-report",
        ]
        if pool == "prefill":
            source_names += ["deepseek-prefill-cell-summary", "deepseek-prefill-kernel-summary"]
        else:
            source_names += ["deepseek-decode-summary", "deepseek-decode-kernel-summary"]
        measured_evidence = _evidence(
            service_class="MEASURED",
            split=split,
            sources=source_names,
            source_definitions=sources,
            derivation=None,
        )
        result.append(
            _entry(
                implementation_id=f"deepseek-v3-reduced4-vllm-{cell_id}",
                key=_deepseek_key(
                    inputs, "deepseek_reduced", cell_id, pool, primary, secondary, False
                ),
                service_ps=measured_ps,
                sm_hz=sm_hz,
                memory_hz=GH200_MEMORY_HZ,
                replay_count=1,
                cv_ppm=0,
                verdict="insufficient-replays",
                evidence=measured_evidence,
            )
        )
        declared_ps = measured_ps * int(scale["target_layers"]) // int(scale["source_layers"])
        declared_evidence = _evidence(
            service_class="DECLARED",
            split=split,
            sources=source_names,
            source_definitions=sources,
            derivation=str(scale["rule"]),
        )
        result.append(
            _entry(
                implementation_id=f"deepseek-v3-full61-vllm-{cell_id}",
                key=_deepseek_key(
                    inputs, "deepseek_full", cell_id, pool, primary, secondary, True
                ),
                service_ps=declared_ps,
                sm_hz=sm_hz,
                memory_hz=GH200_MEMORY_HZ,
                replay_count=1,
                cv_ppm=0,
                verdict="insufficient-replays",
                evidence=declared_evidence,
            )
        )
    return result


def _build_record(
    inputs: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entries = _build_granite_entries(inputs, sources) + _build_deepseek_entries(inputs, sources)
    entries.sort(key=lambda entry: entry["implementation_id"])
    return {
        "schema": KERNEL_CYCLE_LUT_SCHEMA,
        "acceptance_status": "candidate",
        "campaign_id": inputs["campaign_id"],
        "created": inputs["created"],
        "device": inputs["device"],
        "capture_protocol": {
            "graph_minimum_replays": 256,
            "eager_minimum_replays": 30,
            "program_counter_sampling": "denied",
            "compile_graph_inference": "not-captured",
            "code_object_double_harvest": "not-captured",
        },
        "sources": [
            {
                "name": name,
                "fixture_sha256": source["sha256"],
                "fixture_bytes": source["bytes"],
                "retained_source_name": source["path"],
                "retained_source_sha256": source["sha256"],
            }
            for name, source in sources.items()
        ],
        "entries": entries,
    }


def _score(record: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    entries = record["entries"]
    ledger: dict[str, dict[str, int]] = {}
    for entry in entries:
        family = "deepseek_v3" if entry["implementation_id"].startswith("deepseek") else "granite"
        service_class = entry["evidence"]["service_class"]
        ledger.setdefault(family, {}).setdefault(service_class, 0)
        ledger[family][service_class] += 1
        if entry["evidence"]["component_class"] != "DISCLOSED":
            raise ValueError("every emitted component bound must remain DISCLOSED")
    expected = freeze["ledger_expectations"]
    for family in ("granite", "deepseek_v3"):
        normalized = {
            evidence_class: ledger[family].get(evidence_class, 0)
            for evidence_class in ("MEASURED", "DECLARED")
        }
        if normalized != expected[family]["service_entries"]:
            raise ValueError(f"{family}: ledger mismatch {ledger[family]}")

    floors = {128: 199_000_000, 512: 383_000_000, 2048: 1_531_000_000}
    for entry in entries:
        if not entry["implementation_id"].startswith("granite"):
            continue
        key = entry["key"]
        if key["pool"] == "decode":
            floor_ps = 199_000_000
        else:
            floor_ps = floors[key["shape"]["computed_new_tokens"]]
        if entry["measured_service_ps"] < floor_ps:
            raise ValueError(f"{entry['implementation_id']}: below physical floor")

    missing = [
        "granite registered grid: 1212 ABSENT cells",
        "DeepSeek EP72 MTP B16 KV4000: ABSENT",
        "GH200 counter-pass attribution",
        "full-depth DeepSeek silicon captures",
        "uncaptured tensor-parallel widths",
    ]
    return {
        "verdict": "CANDIDATE_COMPILED",
        "scored_after_freeze": True,
        "entry_count": len(entries),
        "evidence_ledger": ledger,
        "component_overlay_ledger": {"granite": {"DISCLOSED": 12}, "deepseek_v3": {"DISCLOSED": 8}},
        "absent": missing,
        "held_out_policy": "compiled by the frozen identity transform only; no fitted parameter exists",
    }


def _profile_payload(path: Path, record: dict[str, Any], record_sha256: str) -> dict[str, Any]:
    provider = compile_profile_table(record)
    provider.save(path)
    payload = _read_json(path)
    evidence_by_id = {
        entry["implementation_id"]: entry["evidence"] for entry in record["entries"]
    }
    for entry in payload["entries"]:
        entry["evidence"] = evidence_by_id[entry["kernel"]]
    payload["acceptance_status"] = "candidate"
    payload["lookup_record_sha256"] = record_sha256
    _write_json(path, payload)
    ProfileTableProvider.load(path)
    return payload


def _device_payload(record: dict[str, Any], record_sha256: str) -> dict[str, Any]:
    compiled = compile_device_service_entries(record)
    evidence_by_id = {
        f"kernel-cycle-{canonical_sha256(entry)}": entry["evidence"]
        for entry in record["entries"]
    }
    return {
        "schema": "simllm-kernel-cycle-device-service-compilation-v1",
        "acceptance_status": compiled.acceptance_status,
        "lookup_record_sha256": record_sha256,
        "resource_registry": compiled.resource_registry.to_obj(),
        "resource_registry_sha256": compiled.resource_registry_sha256,
        "shape_schemas": [schema.to_obj() for schema in compiled.shape_schemas],
        "service_entries": [
            {
                "evidence": evidence_by_id[item.service_entry_id],
                "service_entry": item.to_obj(),
            }
            for item in compiled.service_entries
        ],
    }


def _deepseek_consumption(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for entry in record["entries"]:
        implementation_id = entry["implementation_id"]
        if not implementation_id.startswith("deepseek-v3-"):
            continue
        result.append(
            {
                "implementation_id": implementation_id,
                "evidence_class": entry["evidence"]["service_class"],
                "rank_service_ps": entry["measured_service_ps"],
                "per_layer_ps": (
                    entry["measured_service_ps"] // 61
                    if "full61" in implementation_id
                    else entry["measured_service_ps"] // 4
                ),
                "split": entry["evidence"]["split"],
            }
        )
    return result


def _artifact_manifest(output_dir: Path, names: list[str]) -> dict[str, Any]:
    return {
        "schema": "simllm-hopper-candidate-artifact-manifest-v1",
        "artifacts": [
            {
                "name": name,
                "bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_bytes((output_dir / name).read_bytes()),
            }
            for name in sorted(names)
        ],
    }


def compile_study(evidence_root: Path, output_dir: Path) -> dict[str, Any]:
    inputs = _read_json(INPUT_PATH)
    freeze = _read_json(FREEZE_PATH)
    sources, paths = _source_paths(inputs, evidence_root)
    _verify_granite_cells(inputs, paths)
    _verify_deepseek_cells(inputs, paths)
    _verify_projection(paths["deepseek-deployment-projection"])

    record_value = _build_record(inputs, sources)
    record = validate_kernel_cycle_lut(record_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "candidate-record.json"
    record_path.write_bytes(record.canonical)

    profile_path = output_dir / "profile-table.json"
    _profile_payload(profile_path, record_value, record.record_id)
    device_path = output_dir / "device-service-entries.json"
    _write_json(device_path, _device_payload(record_value, record.record_id))
    result = {
        "schema": "simllm-hopper-kernel-cycle-candidate-result-v1",
        "lookup_record_sha256": record.record_id,
        "acceptance_status": "candidate",
        "freeze_sha256": sha256_bytes(FREEZE_PATH.read_bytes()),
        "input_sha256": sha256_bytes(INPUT_PATH.read_bytes()),
        "score": _score(record_value, freeze),
        "deepseek_per_rank_per_layer": _deepseek_consumption(record_value),
    }
    result_path = output_dir / "result.json"
    _write_json(result_path, result)
    manifest = _artifact_manifest(
        output_dir,
        [record_path.name, profile_path.name, device_path.name, result_path.name],
    )
    _write_json(output_dir / "artifact-manifest.json", manifest)
    result["artifact_manifest_sha256"] = sha256_bytes(
        (output_dir / "artifact-manifest.json").read_bytes()
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="retained simllm-kernelprobe root; defaults to SIMLLM_KERNELPROBE_ROOT",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    evidence_root = args.evidence_root
    if evidence_root is None:
        configured = os.environ.get("SIMLLM_KERNELPROBE_ROOT")
        if not configured:
            raise ValueError("set SIMLLM_KERNELPROBE_ROOT or pass --evidence-root")
        evidence_root = Path(configured)
    result = compile_study(evidence_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
