"""Phase-independent attention pair successor, with the v1 record immutable."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from simllm.calibration.extraction import case_records_from_suite
from simllm.calibration.model_inventory import ModelKernelInventory
from simllm.compute import ModelDims, step_kernel, step_shape
from simllm.core import RequestPhase, ScheduledRequest, StepRecord

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
REGISTRY = ROOT / "offline/calibration/model-inventories"
SUITE = ROOT / "offline/calibration/suites/qwen3-32b-fp8-text-v1-frameworks-2026-08-28/suite.json"
INVENTORIES = {
    "vllm": "b3ec33edcab6a81868c00efdc2147df4d41b633bd932cb6c4831ea282a1d79fe",
    "sglang": "d25471742a987e31d872dc6fbf8c60ee8746ba17eb25e95a245689064ed3a8d7",
}
EXPECTATIONS_COMMIT = "37676c4f443026fc20944126f643c77eb8705b90"
EXPECTATIONS_SHA256 = "ead11b885885a01905e2af32ddc9e9d3c8aa90c29188148424e9f9faee21e688"
PREDECESSORS_SHA256 = "726dab82a29e01624d6d9c3d841037d7f83145c4df0022b6a1755323e8ef54a0"
PROMPTS = (128, 512, 2048, 3500, 4096)
CONTEXTS = (128, 2048, 4250, 8192)


def lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def digest_matches(path: Path, expected: str) -> bool:
    raw = path.read_bytes()
    variants = (raw,) if path.suffix in {".pdf", ".png"} else (raw, lf_bytes(path))
    return any(hashlib.sha256(value).hexdigest() == expected for value in variants)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def reference_pricer():
    """Reuse frozen candidate/pricing logic in an isolated module with new provenance."""
    name = "_attention_pair_successor_reference"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "examples/frontier_comparison_v1/run_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.INVENTORY_SHA256 = INVENTORIES["vllm"]
    return module


def load_inventory(record_id: str) -> ModelKernelInventory:
    path = REGISTRY / f"{record_id}.json"
    inventory = ModelKernelInventory.from_obj(json.loads(lf_bytes(path)))
    if not digest_matches(path, record_id) or inventory.record.canonical != lf_bytes(path):
        raise ValueError("inventory is not canonical at its content-addressed filename")
    return inventory


def neutral(inventory: ModelKernelInventory) -> dict:
    value = inventory.to_obj()
    value.pop("framework")
    value["implementation_identity"].pop("join_tasks")
    return value


def dims_for(inventory: ModelKernelInventory) -> ModelDims:
    g = inventory.model.geometry
    return ModelDims(
        num_layers=g.layers,
        hidden_size=g.hidden_size,
        intermediate_size=g.intermediate_size,
        num_heads=g.num_heads,
        num_kv_heads=g.num_kv_heads,
        head_size=g.head_size,
        vocab_size=g.vocab_size,
        dtype_bytes=2,
        weight_dtype_bytes=1,
        kv_dtype_bytes=2,
    )


def audit_inventory(inventory: ModelKernelInventory, suite: dict) -> list[dict]:
    """Compare pair axes and all family totals with the live step model."""
    rows = []
    for case, record in zip(inventory.cases, case_records_from_suite(suite), strict=True):
        score = next(p for p in case.kernel_projections if p.family_id == "attn_score")
        fused = step_kernel(dims_for(inventory), record, record.num_sampled or 0)
        rows.append(
            {
                "case_id": case.case_id,
                "phase": case.phase,
                "attention_pairs": score.attention_pairs,
                "flops_per_pair": score.attention_flops_per_pair,
                "pair_axis_exact": score.attention_pairs == step_shape(record)[2],
                "coefficient_exact": score.attention_flops_per_pair == 2_097_152,
                "family_flops_exact": sum(p.aggregate_flops for p in case.kernel_projections)
                == fused.flops,
                "family_bytes_exact": sum(p.aggregate_hbm_bytes for p in case.kernel_projections)
                == fused.bytes_moved,
            }
        )
    return rows


@dataclass(frozen=True)
class PairWork:
    static_parameter_bytes: int
    decode_total_flops_per_batch_item: int
    decode_kv_bytes: int
    prefill_total_flops_per_request: int
    prefill_kv_bytes: int
    attention_flops_per_pair: int
    decode_attention_pairs_per_item: int
    prefill_attention_pairs_per_request: int
    decode_attention_flops_per_item: int
    prefill_attention_flops_per_request: int


def exact_divide(value: int, denominator: int) -> int:
    quotient, remainder = divmod(value, denominator)
    if remainder:
        raise ValueError("inventory coefficient is not exactly divisible")
    return quotient


def derive_work(
    inventory: ModelKernelInventory, prompt: int = 3500, context: int = 4250
) -> PairWork:
    """Project one inventory authority into request work, then price per TP rank."""
    cases = {
        case.case_id: {p.family_id: p for p in case.kernel_projections} for case in inventory.cases
    }
    decode, prefill = cases["db-train-b1-c2048"], cases["cp-train-r4-t128"]
    coefficient = decode["attn_score"].attention_flops_per_pair
    if coefficient is None or coefficient != prefill["attn_score"].attention_flops_per_pair:
        raise ValueError("decode and prefill must identify the same attention pair coefficient")
    counts = []
    for phase, n, c in ((RequestPhase.DECODE, 1, context), (RequestPhase.PREFILL, prompt, prompt)):
        counts.append(
            step_shape(
                StepRecord(
                    step_index=0,
                    virtual_time_ps=0,
                    scheduled=[
                        ScheduledRequest("projection", phase, num_new_tokens=n, context_length=c),
                    ],
                )
            )[2]
        )
    decode_pairs, prefill_pairs = counts
    kv_per_token = exact_divide(
        decode["kv_read"].aggregate_hbm_bytes, decode["kv_read"].shape_vector.values[0]
    )
    linear = exact_divide(
        sum(prefill[f].aggregate_flops for f in ("attn_gemm", "mlp_gemm")),
        prefill["attn_gemm"].shape_vector.values[0],
    )
    head = exact_divide(
        prefill["lm_head"].aggregate_flops, prefill["lm_head"].shape_vector.values[0]
    )
    return PairWork(
        static_parameter_bytes=sum(
            p.aggregate_hbm_bytes for f, p in decode.items() if f != "kv_read"
        ),
        decode_total_flops_per_batch_item=sum(
            p.aggregate_flops for f, p in decode.items() if f != "attn_score"
        )
        + coefficient * decode_pairs,
        decode_kv_bytes=kv_per_token * context,
        prefill_total_flops_per_request=linear * prompt + coefficient * prefill_pairs + head,
        prefill_kv_bytes=kv_per_token * prompt,
        attention_flops_per_pair=coefficient,
        decode_attention_pairs_per_item=decode_pairs,
        prefill_attention_pairs_per_request=prefill_pairs,
        decode_attention_flops_per_item=coefficient * decode_pairs,
        prefill_attention_flops_per_request=coefficient * prefill_pairs,
    )


def physical_bounds(work: PairWork, phase: str, tp: int) -> dict:
    batch = 64 if phase == "decode" else 1
    flops = (
        (
            work.decode_total_flops_per_batch_item
            if phase == "decode"
            else work.prefill_total_flops_per_request
        )
        * batch
        // tp
    )
    moved = (
        work.static_parameter_bytes
        + batch * (work.decode_kv_bytes if phase == "decode" else work.prefill_kv_bytes)
    ) // tp
    compute_ps = Fraction(flops * 10**12, 1_979_000_000_000_000)
    memory_ps = Fraction(moved * 10**12, 4_800_000_000_000)
    floor = max(compute_ps, memory_ps)
    return {
        "flops_per_rank": flops,
        "bytes_per_rank": moved,
        "compute_floor_ps": float(compute_ps),
        "memory_floor_ps": float(memory_ps),
        "floor_ps": math.ceil(floor),
        "e04_ceiling_ps": math.ceil(floor * Fraction(5, 2)),
    }


def sweep(pricer, inventory: ModelKernelInventory) -> tuple[list[dict], dict]:
    rows = []
    bounds_hold = True
    for prompt in PROMPTS:
        for context in CONTEXTS:
            work = derive_work(inventory, prompt, context)
            for tp in pricer.TP_WIDTHS:
                candidate = pricer.make_candidate(
                    prefill_tp=tp,
                    prefill_workers=1,
                    decode_tp=tp,
                    decode_workers=1,
                    decode_batch=64,
                )
                for efficiency in pricer.EFFICIENCY_ARMS:
                    point = pricer._price_candidate(
                        candidate, decode_batch=64, efficiency=efficiency, derivation=work
                    )
                    row = {
                        "prompt_tokens": prompt,
                        "context_tokens": context,
                        "tensor_parallel": tp,
                        "efficiency": str(efficiency),
                        "decode_flops_per_item": work.decode_total_flops_per_batch_item,
                        "prefill_flops_per_request": work.prefill_total_flops_per_request,
                        "decode_ps": point["decode_step_ps"],
                        "prefill_ps": point["prefill_request_ps"],
                    }
                    for phase in ("decode", "prefill"):
                        floor = physical_bounds(work, phase, tp)["floor_ps"]
                        bounds_hold &= (
                            floor - 1
                            <= row[f"{phase}_ps"]
                            <= math.ceil(floor / float(efficiency)) + 1
                        )
                    rows.append(row)
    keyed = {
        (r["prompt_tokens"], r["context_tokens"], r["tensor_parallel"], r["efficiency"]): r
        for r in rows
    }
    monotone = True
    for r in rows:
        for tp in pricer.TP_WIDTHS:
            for e in pricer.EFFICIENCY_ARMS:
                if tp >= r["tensor_parallel"] and e >= Decimal(r["efficiency"]):
                    other = keyed[(r["prompt_tokens"], r["context_tokens"], tp, str(e))]
                    monotone &= all(
                        other[f"{phase}_ps"] <= r[f"{phase}_ps"] + 1
                        for phase in ("decode", "prefill")
                    )
    # Independent analytical directions, including the linear intercept and quadratic coefficient.
    directions = all(
        r["decode_flops_per_item"] == 63_967_068_160 + 2_097_152 * (r["context_tokens"] - 1)
        and r["prefill_flops_per_request"]
        == 62_411_243_520 * r["prompt_tokens"]
        + 2_097_152 * (r["prompt_tokens"] ** 2 // 2)
        + 1_555_824_640
        for r in rows
    )
    return rows, {
        "floor_and_ceiling": bounds_hold,
        "tp_and_efficiency_monotone": monotone,
        "affine_decode_and_quadratic_prefill": directions,
    }


def finish(guards: dict[str, bool], result: dict) -> dict:
    """Fatal failure voids all scoring, while retaining clearly named diagnostics."""
    nonvoid = all(guards.values())
    passed = nonvoid and (
        result["x2"]["passed"] == result["x2"]["denominator"]
        and all(result["behavioral"].values())
        and all(
            result["frontier"][family]["passed"] >= result["frontier"][family].get(
                "acceptance_minimum", result["frontier"][family]["denominator"]
            )
            for family in ("X3a", "X3b", "X3c")
        )
    )
    if not nonvoid:
        result = dict(result)
        result["unscored_diagnostics"] = {
            key: result.pop(key) for key in ("x2", "behavioral", "frontier") if key in result
        }
    return {
        **result,
        "fatal_guards": guards,
        "nonvoid": nonvoid,
        "verdict": ("PASS" if passed else "MIXED") if nonvoid else "VOID",
        "scores": {
            "x2": result["x2"],
            "behavioral": result["behavioral"],
            "frontier": {k: result["frontier"][k] for k in ("X3a", "X3b", "X3c")},
        }
        if nonvoid
        else None,
    }


def run_study() -> tuple[dict, float]:
    predecessors = json.loads(lf_bytes(STUDY / "predecessors.json"))
    receipt = json.loads(lf_bytes(STUDY / "extraction.json"))
    suite = json.loads(lf_bytes(SUITE))
    inventories = {name: load_inventory(i) for name, i in INVENTORIES.items()}
    audits = {name: audit_inventory(value, suite) for name, value in inventories.items()}
    guards = {
        "repeat_extractions": all(
            receipt["frameworks"][name]["inventory_sha256"] == [identity, identity]
            and len(set(receipt["frameworks"][name]["step_records_sha256"])) == 1
            and receipt["config_sha256"] == inventories[name].model.config_sha256
            for name, identity in INVENTORIES.items()
        ),
        "chronology": subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTATIONS_COMMIT, "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0,
        "expectations_bytes": digest_matches(STUDY / "expectations.md", EXPECTATIONS_SHA256),
        "predecessor_manifest": digest_matches(STUDY / "predecessors.json", PREDECESSORS_SHA256),
        "immutable_predecessors": all(digest_matches(ROOT / p, d) for p, d in predecessors.items()),
        "framework_neutral_agreement": neutral(inventories["vllm"])
        == neutral(inventories["sglang"]),
        "pair_and_conservation": all(
            all(
                row[k]
                for k in (
                    "pair_axis_exact",
                    "coefficient_exact",
                    "family_flops_exact",
                    "family_bytes_exact",
                )
            )
            for rows in audits.values()
            for row in rows
        ),
        "identity_and_15_cases": all(
            value.model.revision == "aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df"
            and len(value.cases) == 15
            and value.suite.suite_sha256 == hashlib.sha256(lf_bytes(SUITE)).hexdigest()
            for value in inventories.values()
        ),
    }
    result = {
        "schema": "simllm-frontier-comparison-study-v2",
        "expectations_commit": EXPECTATIONS_COMMIT,
        "expectations_sha256": EXPECTATIONS_SHA256,
        "inventory_sha256": INVENTORIES,
        "predecessors": predecessors,
        "extraction": {
            "mode": "offline-local-config-with-retained-framework-bindings",
            "fresh_native_framework_inspection": False,
            "gpu_execution": False,
            "weight_loading": False,
            "repeat_evidence": receipt,
        },
        "structural_oracles": audits,
        "evidence_classes": {
            "pricing": "ESTIMATE",
            "device_envelope": "DECLARED",
            "external": "MEASURED-EXTERNAL operation-database estimates",
        },
        "e_star_installed": False,
    }
    if not all(guards.values()):
        return finish(guards, result), 0.0
    pricer = reference_pricer()
    external = pricer._external_rows()
    guards["external_evidence_class"] = all(
        r["evidence_class"] == "MEASURED-EXTERNAL" for r in external
    )
    work = derive_work(inventories["vllm"])
    with pricer.ProcessGuard() as guard:
        x2 = pricer._x2(work)
        rows, behavioral = sweep(pricer, inventories["vllm"])
    frontier, attempts, _ = pricer._x3(work, external)
    guards["no_pricing_subprocess"] = attempts == 0 and not guard.attempts
    old = json.loads(lf_bytes(ROOT / "examples/frontier_comparison_v1/results.json"))
    behavioral["matched_throughput_nonincreasing"] = all(
        row[bound] <= previous[bound]
        for row, previous in zip(
            frontier["X3c"]["rows"], old["families"]["X3"]["X3c"]["rows"], strict=True
        )
        for bound in ("low_y", "high_y")
    )
    elapsed = frontier.pop("elapsed_seconds")
    result.update(
        {
            "work": asdict(work),
            "work_at_2048": asdict(derive_work(inventories["vllm"], 2048)),
            "physical_bounds": {p: physical_bounds(work, p, 4) for p in ("decode", "prefill")},
            "old_x2": old["families"]["X2"],
            "x2": x2,
            "sweep": rows,
            "behavioral": behavioral,
            "frontier": frontier,
        }
    )
    return finish(guards, result), elapsed


def plot(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old = json.loads(lf_bytes(ROOT / "examples/frontier_comparison_v1/results.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for color, arm in enumerate(("1.0", "0.8", "0.6")):
        for label, source, style in [
            ("Old", old["families"]["X3"], "--"),
            ("Successor", result["frontier"], "-"),
        ]:
            points = source["arms"][arm]["frontier"]
            axes[0].plot(
                [p["x_tokens_per_second_per_user"]["decimal"] for p in points],
                [p["y_tokens_per_second_per_gpu"]["decimal"] for p in points],
                style,
                color=f"C{color}",
                label=f"{label} e={arm}",
            )
    external = result["frontier"]["external_rows"]
    axes[0].scatter(
        [float(p["x_tokens_per_second_per_user"]) for p in external],
        [float(p["y_tokens_per_second_per_gpu"]) for p in external],
        marker="x",
        color="black",
        label="External",
    )
    axes[0].set(
        xlabel="Per-user speed (tokens/s)",
        ylabel="Throughput (tokens/s/GPU)",
        xscale="log",
        yscale="log",
    )
    axes[0].legend(fontsize=7)
    axes[0].set_title(
        "X3 frontier: old (dashed) and successor (solid)\n"
        "pair contract 2,097,152 FLOPs; prefill +4.346% at 3,500 tokens, decode +0.003%",
        fontsize=8,
    )
    e_star = {}
    for label, key, marker, size in [("Old", "old_x2", "o", 9), ("Successor", "x2", "x", 7)]:
        values = [result[key][f"{p}_e_star"]["decimal"] for p in ("decode", "prefill")]
        e_star[label] = values
        axes[1].plot(
            ["Decode", "Prefill"],
            values,
            marker,
            markersize=size,
            markerfacecolor="none" if marker == "o" else None,
            label=label,
        )
    axes[1].axhline(0.4, color="gray", linestyle=":", label="Frozen lower bound 0.4")
    axes[1].annotate(
        f"old = successor = {e_star['Old'][0]:.4f}",
        xy=(0, e_star["Old"][0]),
        xytext=(0.12, e_star["Old"][0] - 0.06),
        fontsize=7,
        ha="left",
        va="top",
    )
    axes[1].annotate(
        f"old {e_star['Old'][1]:.4f}\nsuccessor {e_star['Successor'][1]:.4f}\n(both below the band)",
        xy=(1, e_star["Successor"][1]),
        xytext=(0.62, e_star["Successor"][1] + 0.12),
        fontsize=7,
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.6},
    )
    axes[1].set(ylabel="Implied efficiency e-star (dimensionless)", ylim=(0, 0.7))
    axes[1].set_title("X2c implied efficiency, frozen band floor", fontsize=9)
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].margins(x=0.25)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            output / f"attention-pair-successor.{ext}",
            dpi=160,
            metadata={"CreationDate": None} if ext == "pdf" else None,
        )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.resolve().is_relative_to(ROOT):
        parser.error("output-root must be outside the repository under SIMLLM_DATA_ROOT")
    args.output_root.mkdir(parents=True, exist_ok=False)
    try:
        result, elapsed = run_study()
    except Exception as error:
        (args.output_root / "error.json").write_bytes(
            json_bytes({"verdict": "VOID", "scores": None, "error": str(error)})
        )
        raise
    (args.output_root / "results.json").write_bytes(json_bytes(result))
    (args.output_root / "timing.json").write_bytes(
        json_bytes({"frontier_elapsed_seconds": elapsed})
    )
    if result["nonvoid"]:
        plot(result, args.output_root / "figures")
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "fatal_guards": result["fatal_guards"],
                "work": result.get("work"),
                "elapsed_seconds": elapsed,
            },
            indent=2,
        )
    )
    return 0 if result["nonvoid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
