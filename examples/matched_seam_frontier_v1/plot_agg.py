#!/usr/bin/env python3
"""Render the additive matched-seam aggregate-arm figure pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

STUDY_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = STUDY_DIR.parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from examples.matched_seam_frontier_v1 import plot_publication, plot_study

BASE_RECORD_PATH = STUDY_DIR / "record.json"
SCHEMA = "simllm-matched-seam-aggregate-record-v1"

SERIES_LABELS = {
    "external-agg": "AIC agg: co-located mix",
    "external-disagg": "AIC disagg: split P/D",
    "simllm-disagg-unpriced": "SimLLM disagg: unpriced P/D",
    "simllm-disagg-packet": "SimLLM disagg: packet P/D",
    "simllm-agg-unpriced": "SimLLM agg: unpriced zero-byte P/D",
    "simllm-agg-packet": "SimLLM agg: packet zero-byte P/D",
}

AGGREGATE_STYLES = (
    {
        "id": "simllm-agg-unpriced",
        "label": SERIES_LABELS["simllm-agg-unpriced"],
        "evidence_class": "MEASURED-EXTERNAL",
        "color": "#8e44ad",
        "marker": "P",
        "markerfacecolor": "white",
        "linestyle": (0, (5.0, 2.0)),
        "linewidth": 1.45,
        "zorder": 6,
    },
    {
        "id": "simllm-agg-packet",
        "label": SERIES_LABELS["simllm-agg-packet"],
        "evidence_class": "MEASURED-EXTERNAL",
        "color": "#cc79a7",
        "marker": "X",
        "markerfacecolor": "#cc79a7",
        "linestyle": "None",
        "linewidth": 1.15,
        "zorder": 7,
    },
)

CAPTION = (
    "Every SimLLM compute duration comes from the same imported measured "
    "database. Aggregate differences are in-flight composition, not kernel "
    "timing. The aggregate pool co-locates prefill and decode, so both of its "
    "P/D handoff arms carry zero bytes and coincide exactly."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_base_record(record: dict[str, Any]) -> dict[str, Any]:
    expected_hash = str(record["base_record_sha256"])
    if _sha256(BASE_RECORD_PATH) != expected_hash:
        raise ValueError("protected base record hash mismatch")
    value = json.loads(BASE_RECORD_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("protected base record must be a JSON object")
    return value


def _aggregate_points(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("schema") != SCHEMA:
        raise ValueError(f"unexpected aggregate record schema {record.get('schema')!r}")
    points = []
    for projection in record["families"]["AR"]["baseline_projection"]:
        point = projection["point"]
        points.append(
            {
                "x": float.fromhex(point["tokens_per_second_per_user"]),
                "y": float.fromhex(point["tokens_per_second_per_gpu"]),
                "row": int(projection["row"]),
                "candidate_id": point["configuration_id"],
            }
        )
    return points


def prepare_study_data(
    record: dict[str, Any],
    base_record: dict[str, Any],
) -> dict[str, Any]:
    """Add both aggregate traffic identities to the protected study projection."""

    plot = plot_study.prepare_plot_data(base_record)
    relabel = {
        "external-agg": "external-agg",
        "external-disagg": "external-disagg",
        "simllm-ideal": "simllm-disagg-unpriced",
        "simllm-packet": "simllm-disagg-packet",
    }
    for series in plot["series"]:
        series["label"] = SERIES_LABELS[relabel[series["id"]]]
    aggregate_points = _aggregate_points(record)
    plot["series"].extend(
        {**style, "points": aggregate_points}
        for style in AGGREGATE_STYLES
    )
    all_x = [point["x"] for series in plot["series"] for point in series["points"]]
    all_y = [point["y"] for series in plot["series"] for point in series["points"]]
    plot["axes"]["x"]["limits"] = [min(all_x) * 0.86, max(all_x) * 1.13]
    plot["axes"]["y"]["limits"] = [min(all_y) * 0.82, max(all_y) * 1.18]
    plot["caption"] = CAPTION
    return plot


def prepare_publication_data(
    record: dict[str, Any],
    base_record: dict[str, Any],
) -> dict[str, Any]:
    """Add both aggregate traffic identities to the protected publication view."""

    plot = plot_publication.prepare_publication_data(base_record)
    for series, label_id in zip(
        plot["series"],
        (
            "external-agg",
            "external-disagg",
            "simllm-disagg-unpriced",
            "simllm-disagg-packet",
        ),
        strict=True,
    ):
        series["label"] = SERIES_LABELS[label_id]
    aggregate_tuples = [
        (point["x"], point["y"])
        for point in _aggregate_points(record)
    ]
    plot["series"].extend(
        {
            key: value
            for key, value in {**style, "points": aggregate_tuples}.items()
            if key != "evidence_class"
        }
        for style in AGGREGATE_STYLES
    )
    plot["caption"] = (
        "Same imported measured DB; aggregate is co-located and its P/D handoff "
        "is zero bytes."
    )
    return plot


def _render_with_projection(
    module: Any,
    *,
    base_record: dict[str, Any],
    projection: dict[str, Any],
    pdf_path: Path,
    png_path: Path,
) -> dict[str, Any]:
    original = module.prepare_plot_data if module is plot_study else module.prepare_publication_data
    replacement_name = (
        "prepare_plot_data" if module is plot_study else "prepare_publication_data"
    )
    setattr(module, replacement_name, lambda _: projection)
    try:
        return module.render(base_record, pdf_path=pdf_path, png_path=png_path)
    finally:
        setattr(module, replacement_name, original)


def render_all(
    record: dict[str, Any],
    *,
    study_pdf: Path,
    study_png: Path,
    publication_pdf: Path,
    publication_png: Path,
) -> dict[str, Any]:
    """Render both additive PDF/PNG pairs from the aggregate record."""

    base_record = _load_base_record(record)
    study_projection = prepare_study_data(record, base_record)
    publication_projection = prepare_publication_data(record, base_record)
    _render_with_projection(
        plot_study,
        base_record=base_record,
        projection=study_projection,
        pdf_path=study_pdf,
        png_path=study_png,
    )
    _render_with_projection(
        plot_publication,
        base_record=base_record,
        projection=publication_projection,
        pdf_path=publication_pdf,
        png_path=publication_png,
    )
    return {
        "study": study_projection,
        "publication": publication_projection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--study-pdf", type=Path, required=True)
    parser.add_argument("--study-png", type=Path, required=True)
    parser.add_argument("--publication-pdf", type=Path, required=True)
    parser.add_argument("--publication-png", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    render_all(
        record,
        study_pdf=args.study_pdf,
        study_png=args.study_png,
        publication_pdf=args.publication_pdf,
        publication_png=args.publication_png,
    )


if __name__ == "__main__":
    main()
