"""Publish split and combined VLLM-42 result records."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
FREEZE_COMMIT = "375639c147f39fe4f01ea212855ef9e8efb5d7fa"
EXPECTATIONS_SHA256 = "95a5921d2075136073189ead7ca7fdc9eca4c8fcb6482cffda7e04eee35989da"
SPLIT_RESULT_SCHEMA = "simllm-pd-session-batching-service-split-result-v1"
PUBLICATION_SCHEMA = "simllm-pd-session-batching-service-publication-v1"
COMBINED_SCHEMA = "simllm-pd-session-batching-service-result-v1"
SPLIT_FILES = {
    "non-held-out": ("non_held_out_results.json", "NON_HELD_OUT_RESULTS.md"),
    "held-out": ("held_out_results.json", "HELD_OUT_RESULTS.md"),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def _microseconds(value: dict[str, int]) -> str:
    return f"{float(_fraction(value) / 1_000_000):.6f}"


def publish_split(result: dict[str, Any], raw_sha256: str) -> dict[str, Any]:
    """Reduce one raw external result to its tracked scored projection."""

    if result["schema"] != SPLIT_RESULT_SCHEMA:
        raise ValueError("raw split result schema disagrees")
    provenance = result["provenance"]
    analysis = result["analysis"]
    observation = result["observation"]
    split = analysis["split"]
    if split not in SPLIT_FILES or observation["split"] != split:
        raise ValueError("raw split identity disagrees")
    if provenance["freeze_commit"] != FREEZE_COMMIT:
        raise ValueError("raw result uses another freeze")
    if provenance["expectations_sha256"] != EXPECTATIONS_SHA256:
        raise ValueError("raw result uses another expectation record")
    expected_cells = 48 if split == "non-held-out" else 30
    if analysis["service_band_summary"]["evaluated"] != expected_cells:
        raise ValueError("raw result has an incomplete split score")
    if analysis["conservation"]["cells"] != expected_cells:
        raise ValueError("raw result has an incomplete split conservation ledger")
    expectations = _load_json(EXPECTATIONS_PATH)
    rows = analysis["service_band_verdicts"]
    observed = [
        _fraction(row["observed_batching_service_per_token_ps"]) for row in rows
    ]
    errors = [
        abs(
            _fraction(row["observed_batching_service_per_token_ps"])
            - _fraction(row["predicted_batching_service_per_token_ps"])
        )
        for row in rows
    ]
    floor = expectations["physical_bounds"]["floor_service_per_token_ps"]
    ceiling = expectations["physical_bounds"]["ceiling_service_per_token_ps"]
    return {
        "schema": PUBLICATION_SCHEMA,
        "status": analysis["status"],
        "split": split,
        "provenance": {
            **provenance,
            "raw_result_sha256": raw_sha256,
            "raw_result_location": (
                f"{split}/result.json under SIMLLM_VLLM42_RUN_ROOT"
            ),
        },
        "runtime": observation["runtime"],
        "fatal_guards": analysis["fatal_guards"],
        "conservation": analysis["conservation"],
        "service_band_verdicts": analysis["service_band_verdicts"],
        "service_band_summary": analysis["service_band_summary"],
        "physical_sanity": {
            "floor_service_per_token_ps": floor,
            "ceiling_service_per_token_ps": ceiling,
            "observed_minimum_service_per_token_ps": {
                "numerator": min(observed).numerator,
                "denominator": min(observed).denominator,
            },
            "observed_maximum_service_per_token_ps": {
                "numerator": max(observed).numerator,
                "denominator": max(observed).denominator,
            },
            "maximum_absolute_prediction_error_ps": {
                "numerator": max(errors).numerator,
                "denominator": max(errors).denominator,
            },
            "all_observations_inside_physical_bounds": all(
                _fraction(floor) <= value <= _fraction(ceiling)
                for value in observed
            ),
            "scored": False,
        },
        "separate_fields": {
            "arrival_to_prefill_published": analysis[
                "arrival_to_prefill_published"
            ],
            "handoff_to_decode_published": analysis[
                "handoff_to_decode_published"
            ],
            "batching_service_published": analysis["batching_service_published"],
        },
        "settled_claims": {
            "onset": analysis["onset_claim"],
            "monotonic_250_to_8000": analysis[
                "monotonic_250_to_8000_claim"
            ],
        },
    }


def render_split_markdown(publication: dict[str, Any]) -> str:
    """Render one disclosure-stage result with every cell visible."""

    summary = publication["service_band_summary"]
    conservation = publication["conservation"]
    sanity = publication["physical_sanity"]
    rows = [
        f"# VLLM-42 {publication['split']} batching-service result",
        "",
        f"Status: `{publication['status']}`. {summary['held']} of {summary['evaluated']} frozen service bands held; {summary['missed']} missed.",
        "",
        f"Conservation held for {conservation['admissions']} admissions, {conservation['handoffs']} handoffs, {conservation['terminals']} terminals, and {conservation['terminal_decode_tokens']} decode tokens. The maximum time-to-first-token decomposition residual was {conservation['maximum_ttft_residual_ps']} picoseconds.",
        "",
        f"The unscored physical check places the observed service range at {_microseconds(sanity['observed_minimum_service_per_token_ps'])} to {_microseconds(sanity['observed_maximum_service_per_token_ps'])} microseconds per request-token, inside the frozen {_microseconds(sanity['floor_service_per_token_ps'])} to {_microseconds(sanity['ceiling_service_per_token_ps'])} microsecond surface bounds. The largest absolute prediction error is {_microseconds(sanity['maximum_absolute_prediction_error_ps'])} microseconds.",
        "",
        "Arrival-to-prefill wait, handoff-to-decode admission wait, and batching",
        "service remain separate below. Wait fields are diagnostics and are not scored.",
        "The common 210 to 220 requests per second onset and the 250 to 8,000",
        "requests per second monotonic direction are preserved and not rescored.",
        "",
        "| Prefill | Decode | Prompt | Load | Predicted us | Band us | Observed us | Arrival to prefill us | Handoff to decode us | Verdict |",
        "|---:|---:|---:|---:|---:|:---|---:|---:|---:|:---|",
    ]
    for row in publication["service_band_verdicts"]:
        prefill, decode, prompt, load = row["cell"]
        band = row["batching_service_per_token_band_ps"]
        rows.append(
            "| "
            f"{prefill} | {decode} | {prompt} | {load} | "
            f"{_microseconds(row['predicted_batching_service_per_token_ps'])} | "
            f"{_microseconds(band['lower'])} to {_microseconds(band['upper'])} | "
            f"{_microseconds(row['observed_batching_service_per_token_ps'])} | "
            f"{_microseconds(row['arrival_to_prefill_wait_ps'])} | "
            f"{_microseconds(row['handoff_to_decode_admission_wait_ps'])} | "
            f"{'HELD' if row['held'] else 'MISSED'} |"
        )
    return "\n".join(rows) + "\n"


def combine_publications(
    non_held_out: dict[str, Any],
    held_out: dict[str, Any],
) -> dict[str, Any]:
    """Combine the already disclosed split publications without rescoring."""

    for expected_split, publication in (
        ("non-held-out", non_held_out),
        ("held-out", held_out),
    ):
        if publication["schema"] != PUBLICATION_SCHEMA:
            raise ValueError("split publication schema disagrees")
        if publication["split"] != expected_split:
            raise ValueError("split publication identity disagrees")
        if publication["provenance"]["freeze_commit"] != FREEZE_COMMIT:
            raise ValueError("split publication uses another freeze")
    release = held_out["provenance"].get("non_held_out_publication")
    if not release or not release.get("commit"):
        raise ValueError("held-out evidence lacks committed disclosure provenance")
    rows = sorted(
        [
            *non_held_out["service_band_verdicts"],
            *held_out["service_band_verdicts"],
        ],
        key=lambda row: tuple(row["cell"]),
    )
    if len(rows) != 78 or len({tuple(row["cell"]) for row in rows}) != 78:
        raise ValueError("combined result does not contain 78 unique cells")
    fatal = [
        *non_held_out["fatal_guards"]["findings"],
        *held_out["fatal_guards"]["findings"],
    ]
    if fatal:
        status = "VOID"
    elif all(row["held"] for row in rows):
        status = "PASS"
    else:
        status = "REFUTED"
    return {
        "schema": COMBINED_SCHEMA,
        "status": status,
        "task": "VLLM-42",
        "provenance": {
            "freeze_commit": FREEZE_COMMIT,
            "expectations_sha256": EXPECTATIONS_SHA256,
            "non_held_out_publication_commit": release["commit"],
            "non_held_out_publication_sha256": release["sha256"],
            "held_out_run_head": held_out["provenance"]["run_head"],
        },
        "fatal_guards": {
            "status": "HELD" if not fatal else "VIOLATED",
            "findings": fatal,
        },
        "conservation": {
            key: (
                max(
                    non_held_out["conservation"][key],
                    held_out["conservation"][key],
                )
                if key == "maximum_ttft_residual_ps"
                else non_held_out["conservation"][key]
                + held_out["conservation"][key]
            )
            for key in non_held_out["conservation"]
        },
        "service_band_verdicts": rows,
        "service_band_summary": {
            "held": sum(row["held"] for row in rows),
            "missed": sum(not row["held"] for row in rows),
            "evaluated": len(rows),
            "non_held_out_held": non_held_out["service_band_summary"]["held"],
            "held_out_held": held_out["service_band_summary"]["held"],
        },
        "settled_claims": {
            "onset": "PRESERVED_NOT_RESCORED",
            "monotonic_250_to_8000": "PRESERVED_NOT_RESCORED",
        },
        "closure": {
            "VLLM-42": "CLOSED" if status == "PASS" else "OPEN",
            "VLLM-50": "UNUSED" if status == "PASS" else "REGISTER_RESIDUAL",
        },
    }


def render_combined_markdown(result: dict[str, Any]) -> str:
    """Render the combined verdict and complete per-cell table."""

    summary = result["service_band_summary"]
    conservation = result["conservation"]
    rows = [
        "# VLLM-42 batching-service result",
        "",
        f"Status: `{result['status']}`. {summary['held']} of {summary['evaluated']} frozen service bands held; {summary['missed']} missed.",
        "",
        "The phase-complete predictor advances the shared clock through prompt service",
        "and handoff before decode eligibility, then prices only predicted decode batch",
        "membership with the independent measured service surface. It has no observed",
        "curve inputs and no fitted parameters.",
        "",
        f"Conservation held for {conservation['admissions']} admissions, {conservation['handoffs']} handoffs, {conservation['terminals']} terminals, and {conservation['terminal_decode_tokens']} decode tokens. The maximum time-to-first-token residual was {conservation['maximum_ttft_residual_ps']} picoseconds.",
        "",
        f"VLLM-42 is `{result['closure']['VLLM-42']}`. VLLM-50 is `{result['closure']['VLLM-50']}`. The settled onset and high-load monotonic claims remain preserved and unscored.",
        "",
        "| Prefill | Decode | Prompt | Load | Split | Predicted us | Band us | Observed us | Arrival to prefill us | Handoff to decode us | Verdict |",
        "|---:|---:|---:|---:|:---|---:|:---|---:|---:|---:|:---|",
    ]
    for row in result["service_band_verdicts"]:
        prefill, decode, prompt, load = row["cell"]
        band = row["batching_service_per_token_band_ps"]
        rows.append(
            "| "
            f"{prefill} | {decode} | {prompt} | {load} | {row['split']} | "
            f"{_microseconds(row['predicted_batching_service_per_token_ps'])} | "
            f"{_microseconds(band['lower'])} to {_microseconds(band['upper'])} | "
            f"{_microseconds(row['observed_batching_service_per_token_ps'])} | "
            f"{_microseconds(row['arrival_to_prefill_wait_ps'])} | "
            f"{_microseconds(row['handoff_to_decode_admission_wait_ps'])} | "
            f"{'HELD' if row['held'] else 'MISSED'} |"
        )
    return "\n".join(rows) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--raw-result", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--non-held-out", type=Path, required=True)
    combine.add_argument("--held-out", type=Path, required=True)
    combine.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "split":
        raw = _load_json(args.raw_result)
        publication = publish_split(raw, _sha256(args.raw_result))
        json_name, markdown_name = SPLIT_FILES[publication["split"]]
        _write_json(args.output_dir / json_name, publication)
        (args.output_dir / markdown_name).write_text(
            render_split_markdown(publication),
            encoding="utf-8",
            newline="\n",
        )
        return 0
    combined = combine_publications(
        _load_json(args.non_held_out),
        _load_json(args.held_out),
    )
    _write_json(args.output_dir / "results.json", combined)
    (args.output_dir / "RESULTS.md").write_text(
        render_combined_markdown(combined),
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
