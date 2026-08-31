#!/usr/bin/env python3
"""Score the one-shot CORE-61 depth-8 measurement against its frozen prediction."""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any


def _percent(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def score(freeze: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
    if freeze.get("status") != "EXPECTATIONS_ONLY_PRE_SCORING_AMENDMENT":
        raise ValueError("CORE-61 retry freeze has the wrong status")
    if measurement.get("status") != "DIGEST_READY_MEASUREMENT":
        raise ValueError("CORE-61 measurement is not digest ready")
    if measurement.get("evidence_class") != "MEASURED":
        raise ValueError("CORE-61 score requires measured evidence")
    expected_shape = {
        "depth_layers": 8,
        "batch_size": 32,
        "remote_kv_tokens_per_request": 2000,
    }
    if measurement.get("shape") != expected_shape:
        raise ValueError("CORE-61 measured shape changed")
    acceptance = freeze["acceptance"]
    prediction = int(acceptance["preregistered_prediction_ps"])
    measured = int(measurement["measured_service_ps"])
    if measured <= 0:
        raise ValueError("CORE-61 measured service must be positive")
    residual = measured - prediction
    signed_percent = Decimal(residual) * Decimal(100) / Decimal(measured)
    absolute_percent = abs(signed_percent)
    tolerance = Decimal(acceptance["held_out_tolerance_percent"])
    passed = absolute_percent <= tolerance
    verdict = (
        "VALIDATED_LINEAR_DEPTH_SCALING"
        if passed
        else "QUANTIFIED_DEPTH_NONLINEARITY"
    )
    interpretation = acceptance["on_pass"] if passed else acceptance["on_miss"]
    return {
        "schema": "simllm-deployment-curve-core61-depth-retry-result-v1",
        "task": "CORE-61",
        "status": "SCORED",
        "measurement": measurement,
        "score": {
            "preregistered_prediction_ps": prediction,
            "measured_service_ps": measured,
            "signed_residual_ps": residual,
            "signed_residual_percent": _percent(signed_percent),
            "absolute_residual_percent": _percent(absolute_percent),
            "held_out_tolerance_percent": str(tolerance),
            "prediction_within_tolerance": passed,
            "linearity_verdict": verdict,
        },
        "interpretation": interpretation,
        "signed_residual_ledger": [
            {
                "owner": "CORE-61",
                "term": "held-out depth scaling",
                "signed_ps": residual,
                "signed_percent": _percent(signed_percent),
            },
            {
                "owner": "TRAF-66",
                "term": "finite compute and communication overlap",
                "signed_ps": None,
                "state": "UNCHANGED_NOT_RECOMPUTED",
            },
        ],
        "registry": {
            "core61": "CLOSE_LINEARITY_VALIDATED" if passed else "OPEN_REFUTED",
            "core63": "NOT_REGISTERED" if passed else "REGISTER_IF_RESIDUAL_REMAINS",
            "comp76": "UNCHANGED",
        },
        "preservation": freeze["preservation"],
    }


def markdown(result: dict[str, Any], freeze_commit: str) -> str:
    scored = result["score"]
    direction = "above" if scored["signed_residual_ps"] >= 0 else "below"
    verdict = scored["linearity_verdict"].replace("_", " ").lower()
    return (
        "# CORE-61 depth-8 retry result\n\n"
        "## Signed depth residual and verdict\n\n"
        f"The signed depth residual is **{scored['signed_residual_ps']:,} ps** "
        f"({scored['signed_residual_percent']} percent, measured minus predicted), "
        f"with the measurement {direction} the preregistered "
        f"{scored['preregistered_prediction_ps']:,} ps prediction. The frozen "
        f"five-percent comparison is **{verdict}**.\n\n"
        f"{result['interpretation']}\n\n"
        "## Frozen comparison\n\n"
        f"The pre-scoring harness amendment was frozen at `{freeze_commit}`. "
        "It changed startup scaffolding only. The prediction, sign convention, "
        "tolerance and exact batch-32, remote-KV-2000 boundary remained unchanged.\n\n"
        "## Residual separation\n\n"
        "The signed value above belongs only to CORE-61 depth scaling. TRAF-66's "
        "finite compute and communication overlap was not recomputed and remains "
        "a separate ledger term. COMP-76 is unchanged.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    measurement = json.loads(args.measurement.read_text(encoding="utf-8"))
    result = score(freeze, measurement)
    args.json_out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.markdown_out.write_text(
        markdown(result, args.freeze_commit),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["score"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
