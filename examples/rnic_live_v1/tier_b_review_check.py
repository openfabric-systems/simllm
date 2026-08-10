"""Freeze-time validation for the Tier B review supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("tier_b_review_expectations.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--producer", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.check_only:
        raise RuntimeError("freeze-time harness supports check-only mode")
    if not arguments.out.is_absolute() or not arguments.producer.is_absolute():
        raise ValueError("Tier B output and producer paths must be absolute")
    if not arguments.producer.resolve(strict=False).is_relative_to(
        arguments.out.resolve(strict=False)
    ):
        raise ValueError("Tier B producer must reside under its output directory")
    data = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    if data["schema"] != "simllm-rnic-tier-b-expectations-v1":
        raise AssertionError("Tier B expectations schema drifted")
    if data["factory"] != "htsim":
        raise AssertionError("Tier B factory drifted")
    if data["simllm_base_commit"] != "fc282efc91573638de7dcfae2befee1cf022011b":
        raise AssertionError("Tier B base commit drifted")
    if data["producer_argument_names"] != [
        "--factory",
        "--expectations",
        "--observations",
    ]:
        raise AssertionError("Tier B producer invocation drifted")
    if data["retained_bypass_profiles"] != [
        "rnic-nn-fluid",
        "rnic-nn",
        "rnic-cn",
        "dcqcn",
    ]:
        raise AssertionError("Tier B bypass set drifted")
    if data["behavioral_family_instances"]["two_wqe_fifo"] != 4:
        raise AssertionError("Tier B FIFO family drifted")
    if set(data["doorbell_owner_mappings"]) != {"queue_owner", "nic_owner"}:
        raise AssertionError("Tier B doorbell mappings drifted")
    for relative in (
        "examples/rnic_live_v1/tier_b_expectations.md",
        "examples/rnic_live_v1/tier_b_review_supplement.md",
        "examples/core5_reduction/expectations.md",
        "examples/core5_reduction/review_expectations.md",
    ):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    print("Tier B review supplement registry check passed; no artifacts produced")


if __name__ == "__main__":
    main()
