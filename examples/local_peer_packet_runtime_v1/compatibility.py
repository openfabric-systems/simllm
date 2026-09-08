"""Reproduce complete accepted bypass artifacts under controlled observer inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

BASELINE = "4b7042681a8b5d942fc21f96dd9786f887203a46"
EXCLUDED_PROVENANCE = {
    "observed_simllm_revision", "observed_htsim_gitlink", "simllm_revision",
    "htsim_gitlink", "python", "platform", "htsim_rnic_sha256", "txt2bin_sha256",
}
FAMILIES = ("nvlink_locality_v1", "mixed_attribution_v1", "rnic_live_v1", "rnic_packet_v2")


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load_study(root, family):
    path = root / "examples" / family / "run_study.py"
    spec = importlib.util.spec_from_file_location("_peer_compat_" + family, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def produce(args):
    root, output = args.source_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(SIMLLM_HTSIM_RNIC=str(args.htsim.resolve()), SIMLLM_TXT2BIN=str(args.txt2bin.resolve()),
                      SIMLLM_MIXED_ATTRIBUTION_RUN_ROOT=str(output))
    locality = load_study(root, "nvlink_locality_v1")
    locality.run(SimpleNamespace(out=output / "nvlink_locality_v1"))
    mixed = load_study(root, "mixed_attribution_v1")
    # Only this study module's host stopwatch is controlled. The simulator,
    # subprocess timeouts and all virtual time authorities retain real inputs.
    mixed.time = SimpleNamespace(time=lambda: 0.0)
    status = mixed.run(SimpleNamespace(run_dir=output / "mixed_attribution_v1", htsim_rnic=args.htsim))
    if status:
        raise RuntimeError("accepted mixed attribution study failed")
    expectations = root / "examples/rnic_live_v1/tier_a_expectations.json"
    for abi, family in ((1, "rnic_live_v1"), (2, "rnic_packet_v2")):
        directory = output / family
        directory.mkdir()
        result = subprocess.run([str(args.native_producer.resolve()), "--factory", "htsim", "--network-abi-version", str(abi),
                                 "--expectations", str(expectations), "--observations", str(directory / "observations.json")],
                                check=True, capture_output=True)
        (directory / "producer.stdout").write_bytes(result.stdout)
        (directory / "producer.stderr").write_bytes(result.stderr)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    (output / "source.json").write_bytes(canonical({"source_commit": revision,
        "native_producer_sha256": sha(args.native_producer.read_bytes()),
        "htsim_sha256": sha(args.htsim.read_bytes()), "txt2bin_sha256": sha(args.txt2bin.read_bytes()),
        "mixed_host_stopwatch_input": "constant_zero_unscored", "hardware_measurements": 0}))


def normalized(path):
    raw = path.read_bytes()
    excluded = []
    if path.name == "summary.json":
        value = json.loads(raw)
        provenance = value.get("provenance", {})
        for key in sorted(EXCLUDED_PROVENANCE & provenance.keys()):
            excluded.append("/provenance/" + key)
            del provenance[key]
        raw = canonical(value)
    return raw, excluded


def inventory(root):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                  if path.is_file() and (path.suffix in (".goal", ".csv", ".bin")
                                        or path.name in ("summary.json", "observations.json")))


def compare(args):
    before, after, output = args.before.resolve(), args.after.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before_identity = json.loads((before / "source.json").read_bytes())
    after_identity = json.loads((after / "source.json").read_bytes())
    if before_identity["source_commit"] != BASELINE:
        raise ValueError("compatibility baseline does not match the prospective freeze")
    result = {"schema": "simllm-peer-compatibility-v1", "baseline_commit": BASELINE,
              "current_commit": after_identity["source_commit"], "before_identity": before_identity,
              "after_identity": after_identity, "families": {}}
    for family in FAMILIES:
        left, right = inventory(before / family), inventory(after / family)
        if left != right or not left:
            raise ValueError("compatibility artifact inventory changed: " + family)
        rows, before_bundle, after_bundle = [], bytearray(), bytearray()
        for name in left:
            old, old_excluded = normalized(before / family / name)
            new, new_excluded = normalized(after / family / name)
            if old_excluded != new_excluded:
                raise ValueError("compatibility provenance shape changed")
            rows.append({"artifact": name, "before_sha256": sha(old), "after_sha256": sha(new),
                         "equal": old == new, "excluded_json_pointers": old_excluded,
                         "before_raw_sha256": sha((before / family / name).read_bytes()),
                         "after_raw_sha256": sha((after / family / name).read_bytes())})
            for bundle, payload in ((before_bundle, old), (after_bundle, new)):
                encoded = name.encode()
                bundle.extend(len(encoded).to_bytes(8, "big") + encoded + len(payload).to_bytes(8, "big") + payload)
        (output / (family + "-before.bin")).write_bytes(before_bundle)
        (output / (family + "-after.bin")).write_bytes(after_bundle)
        result["families"][family] = {"before_sha256": sha(before_bundle), "after_sha256": sha(after_bundle),
                                     "equal": before_bundle == after_bundle, "artifacts": rows}
    (output / "comparison.json").write_bytes(canonical(result))
    print(json.dumps({family: value["equal"] for family, value in result["families"].items()}, indent=2))
    return 0 if all(value["equal"] for value in result["families"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    producer = commands.add_parser("produce")
    for name in ("source-root", "output", "htsim", "txt2bin", "native-producer"):
        producer.add_argument("--" + name, type=Path, required=True)
    comparison = commands.add_parser("compare")
    for name in ("before", "after", "output"):
        comparison.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "produce":
        produce(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
