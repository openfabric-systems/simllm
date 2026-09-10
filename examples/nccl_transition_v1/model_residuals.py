"""Post-specified descriptive errors of the unchanged, previously fitted curve."""

import json
import math
from bisect import bisect_right


def model_time_us(model, payload):
    endpoint = payload * 2 * (model["width"] - 1) / model["width"]
    anchors = model["anchors"]
    index = bisect_right([p[0] for p in anchors], endpoint)
    if index == 0:
        bandwidth = anchors[0][1]
    elif index == len(anchors):
        bandwidth = anchors[-1][1]
    else:
        (x0, y0), (x1, y1) = anchors[index - 1 : index + 1]
        fraction = math.log(endpoint / x0) / math.log(x1 / x0)
        bandwidth = math.exp(math.log(y0) + fraction * math.log(y1 / y0))
    return model["floor_us"] + endpoint / bandwidth * 1e6


def residuals(points, repo):
    models = json.loads((repo / "examples/collective_regime_curve_v1/validation.json").read_text())[
        "curves"
    ]
    rows, summaries = [], []
    for model in models:
        # Check the plotting projection against the existing retained predictions.
        for original in model["held_out"]:
            if abs(model_time_us(model, original["payload_bytes"]) - original["curve_us"]) > 1e-7:
                raise ValueError("Unchanged-model projection disagrees with retained predictions")
        for lane in ("legacy", "timing"):
            selected = [
                r
                for r in points
                if r["architecture"] == model["machine"]
                and r["width"] == model["width"]
                and r["arm"] == "auto"
                and r["lane"] == lane
            ]
            if not selected:
                continue
            curve_rows = []
            for row in selected:
                predicted = model_time_us(model, row["bytes"])
                curve_rows.append(
                    {
                        "architecture": row["architecture"],
                        "width": row["width"],
                        "lane": lane,
                        "arm": "auto",
                        "bytes": row["bytes"],
                        "model_us": predicted,
                        "measured_us": row["median_us"],
                        "error_pct": 100 * (predicted / row["median_us"] - 1),
                    }
                )
            worst = max(curve_rows, key=lambda r: abs(r["error_pct"]))
            summaries.append(
                {
                    "architecture": model["machine"],
                    "width": model["width"],
                    "lane": lane,
                    "worst_signed_error_pct": worst["error_pct"],
                    "worst_payload_bytes": worst["bytes"],
                    "shapes_outside_15_percent": sum(abs(r["error_pct"]) > 15 for r in curve_rows),
                    "shapes": len(curve_rows),
                }
            )
            rows.extend(curve_rows)
    return rows, {
        "chronology": "Post-specified descriptive comparison after capture and visual inspection; not a new H1-H5 acceptance family.",
        "model_unchanged": True,
        "curves": summaries,
    }
