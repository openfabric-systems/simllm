"""Qualify packed metadata admission without weights, frameworks or GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simllm.calibration.canonical import canonical_bytes, canonical_sha256
from simllm.calibration.extraction import (
    ModelExtractionError,
    _expected_dims,
    load_extraction_suite,
)

HERE = Path(__file__).resolve().parent
FREEZE = "2bec2fd"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def independent_floor(parameters):
    groups, remainder = divmod(parameters, 32)
    tail_values = len(range(0, remainder, 2))
    return 17 * groups + tail_values + int(remainder > 0)


def synthetic_suite(template, parameters, byte_count, quantization):
    suite = json.loads(json.dumps(template))
    suite["suite"] = "synthetic-packed-storage-envelope"
    model = suite["reference_model"]
    model["name"] = "synthetic/packed-storage-envelope"
    model["quantization"] = quantization
    model["parameter_count"] = parameters
    model["weight_bytes"] = byte_count
    model["weight_shards"] = [
        {"name": "model-00001-of-00001.safetensors", "sha256": "1" * 64, "bytes": byte_count}
    ]
    model["weight_sha256"] = canonical_sha256(model["weight_shards"])
    return suite


def run_study(output_root):
    sources = ("simllm/calibration/extraction.py", str(Path(__file__).resolve().relative_to(ROOT)))
    for path in sources:
        if (ROOT / path).read_bytes() != git("show", f"HEAD:{path}"):
            raise ValueError(f"commit the study source before execution: {path}")
    git("merge-base", "--is-ancestor", FREEZE, "HEAD")
    for name in ("expectations.md", "expectations.json"):
        relative = str((HERE / name).relative_to(ROOT))
        if (HERE / name).read_bytes() != git("show", f"{FREEZE}:{relative}"):
            raise ValueError("packed storage expectations changed after the freeze")
    frozen = json.loads((HERE / "expectations.json").read_bytes())
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    template = json.loads((ROOT / frozen["identity_template"]).read_bytes())
    rows, oracles, guards, relations = [], [], [], []
    legacy = []
    for path, expected_digest in frozen["preserved_sha256"].items():
        raw = (ROOT / path).read_bytes()
        if digest(raw) != expected_digest:
            guards.append(f"legacy suite changed: {path}")
        suite, identity = load_extraction_suite(raw)
        original = suite["reference_model"]
        projection = {key: original[key] for key in identity.to_obj()}
        expected = canonical_bytes(projection)
        observed = canonical_bytes(identity.to_obj())
        if expected != observed:
            guards.append(f"legacy identity changed: {path}")
        legacy.append(
            {"suite": path, "identity_sha256": digest(observed), "unchanged": expected == observed}
        )
    for parameters in frozen["parameter_counts"]:
        floor = independent_floor(parameters)
        payload = (parameters * frozen["value_bits"] + 7) // 8
        scales = (parameters + frozen["group_size"] - 1) // frozen["group_size"]
        oracles.append(
            {
                "parameters": parameters,
                "value_bytes": payload,
                "scale_bytes": scales,
                "floor_bytes": floor,
                "residual_bytes": payload + scales - floor,
            }
        )
        if floor != payload + scales:
            guards.append(f"independent byte floor disagrees for {parameters}")
        for extra in [-1, *frozen["container_overhead_bytes"]]:
            byte_count = floor + extra
            suite = synthetic_suite(template, parameters, byte_count, frozen["quantization"])
            row = {
                "parameters": parameters,
                "container_extra_bytes": extra,
                "floor_bytes": floor,
                "file_bytes": byte_count,
                "expected_acceptance": extra >= 0,
                "synthetic_input": True,
            }
            try:
                _, identity = load_extraction_suite(canonical_bytes(suite))
            except ModelExtractionError as error:
                row.update(accepted=False, rejection=str(error))
            else:
                row.update(
                    accepted=True, identity_sha256=digest(canonical_bytes(identity.to_obj()))
                )
                if identity.weight_bytes != byte_count:
                    guards.append("identity changed the supplied container bytes")
                try:
                    _expected_dims(identity)
                except ModelExtractionError:
                    row["generic_compute_rejected"] = True
                else:
                    row["generic_compute_rejected"] = False
                    guards.append("packed identity incorrectly enabled generic compute")
            if row["accepted"] != row["expected_acceptance"]:
                guards.append(f"incorrect metadata admission for {parameters}/{extra}")
            rows.append(row)
            (output_root / f"p{parameters}-extra{extra}.json").write_bytes(canonical_bytes(row))
        relation = {
            "family": "container_overhead",
            "parameters": parameters,
            "observed_bytes": rows[-1]["file_bytes"] - rows[-2]["file_bytes"],
            "expected_bytes": 48,
        }
        relations.append(relation)
    by_parameters = {row["parameters"]: row for row in oracles}
    for component in ("value_bytes", "scale_bytes"):
        relations.append(
            {
                "family": "complete_group_scaling",
                "component": component,
                "observed_bytes": by_parameters[1024][component],
                "expected_bytes": 2 * by_parameters[512][component],
            }
        )
    for relation in relations:
        relation["verdict"] = (
            "PASS" if relation["observed_bytes"] == relation["expected_bytes"] else "REFUTED"
        )
        if relation["verdict"] != "PASS":
            guards.append("exact storage scaling relation failed")
    result = {
        "schema": "simllm-packed-weight-identity-result-v1",
        "verdict": "VOID"
        if guards
        else "PASS"
        if all(row["verdict"] == "PASS" for row in relations)
        else "REFUTED",
        "evidence_class": "synthetic_metadata_identity",
        "hardware_measurement": False,
        "weights_downloaded_or_verified": False,
        "expectations_commit": git("rev-parse", FREEZE).decode().strip(),
        "implementation_commit": git("rev-parse", "HEAD").decode().strip(),
        "source_sha256": {path: digest((ROOT / path).read_bytes()) for path in sources},
        "configuration_count": len(rows),
        "configurations": rows,
        "exact_oracle_count": len(oracles),
        "exact_oracles": oracles,
        "storage_relation_family_count": len({row["family"] for row in relations}),
        "storage_relation_instance_count": len(relations),
        "storage_relations": relations,
        "behavioral_family_count": 0,
        "behavioral_instance_count": 0,
        "behavioral_score": None,
        "evidence_classification": "Storage arithmetic, admission, rejection and identity controls are unscored; this study has no timing-behavior denominator.",
        "fatal_findings": guards,
        "legacy_controls": legacy,
        "closure_scope": "COMP-91 packed metadata admission only",
        "remaining_tasks": frozen["remaining_tasks"],
    }
    (output_root / "summary.json").write_bytes(canonical_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.output_root)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "verdict",
                    "configuration_count",
                    "exact_oracle_count",
                    "behavioral_instance_count",
                    "fatal_findings",
                )
            }
        )
    )
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
