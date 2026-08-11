"""Run or freeze-check the registered Tier C packet live-chain study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTATIONS = Path(__file__).with_name("tier_c_expectations.json")
SIMLLM_AUDIT_COMMIT = "90ada43070adb3b1e624b6819aff34d8620e8571"
HTSIM_AUDIT_COMMIT = "4885c647eecdfdf81479d1df052223c016ad086b"
BRANCH_OUTPUT = Path("codex") / "htsim9_packet_closure"
FREEZE_COMMIT = "2bd61cdfe7b6d545c05ea17db6894bb50eb14735"
TIER_A_ACCEPTANCE = Path(__file__).with_name("tier_a_acceptance.py")
TIER_B_LAUNCHER = Path(__file__).with_name("tier_b_producer_launcher.sh")
TIER_C_LAUNCHER = Path(__file__).with_name("tier_c_producer_launcher.sh")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _revision(source: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wave5_root() -> Path:
    configured = os.environ.get("SIMLLM_WAVE5_RUN_ROOT")
    if not configured:
        raise RuntimeError(
            "SIMLLM_WAVE5_RUN_ROOT must name the external wave-5 run root"
        )
    return Path(configured).resolve()


def _validate_registry(
    htsim_source: Path,
    reference_rnic: Path,
    reference_dcqcn: Path,
    out: Path,
) -> None:
    expectations = _load_json(EXPECTATIONS)
    if expectations.get("schema") != "simllm-rnic-tier-c-expectations-v1":
        raise AssertionError("Tier C expectation schema drifted")
    if expectations.get("simllm_audit_commit") != SIMLLM_AUDIT_COMMIT:
        raise AssertionError("Tier C SimLLM audit commit drifted")
    if expectations.get("htsim_audit_commit") != HTSIM_AUDIT_COMMIT:
        raise AssertionError("Tier C htsim audit commit drifted")
    if _revision(htsim_source) != HTSIM_AUDIT_COMMIT:
        raise AssertionError("Tier C htsim source is not the audited pin")
    subprocess.run(
        ["git", "cat-file", "-e", f"{SIMLLM_AUDIT_COMMIT}^{{commit}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    if not all(path.is_absolute() for path in (reference_rnic, reference_dcqcn)):
        raise ValueError("Tier C reference binary paths must be absolute")
    expected_out = (_wave5_root() / BRANCH_OUTPUT).resolve()
    if out.resolve() != expected_out:
        raise ValueError(f"Tier C output must resolve to {expected_out}")
    single = expectations.get("single_wqe", {})
    fifo = expectations.get("fifo", {})
    single_count = (
        len(single.get("payload_bytes", []))
        * len(single.get("link_rate_gbps", []))
        * len(single.get("doorbell_service_ps", []))
    )
    fifo_count = (
        len(fifo.get("link_rate_gbps", []))
        * len(fifo.get("doorbell_service_ps", []))
    )
    if single_count != 8 or fifo_count != 4:
        raise AssertionError("Tier C matrix cardinality drifted")
    if expectations.get("behavioral_family_instances") != {
        "doorbell_packet_to_live_chain": 4,
        "link_rate_packet_to_live_chain": 4,
    }:
        raise AssertionError("Tier C scored family registry drifted")
    doorbell = expectations.get("doorbell_live_relations", [])
    if len(doorbell) != 4 or any(
        row.get("packet_offset_delta_ps") != 1000
        or row.get("live_metric_delta_ps") != 1000
        or row.get("absolute_completion_delta_ps") != [1000, 2000, 3000]
        for row in doorbell
    ):
        raise AssertionError("Tier C doorbell relation drifted")
    link = expectations.get("link_rate_live_relations", [])
    expected_link = {
        (4096, 0): (0, 81920),
        (4096, 1000): (0, 81920),
        (1048576, 0): (20889600, 20971520),
        (1048576, 1000): (20889600, 20971520),
    }
    observed_link = {
        (row.get("payload_bytes"), row.get("doorbell_service_ps")): (
            row.get("last_packet_offset_delta_ps"),
            row.get("live_metric_delta_ps"),
        )
        for row in link
        if row.get("first_packet_offset_delta_ps") == 0
    }
    if observed_link != expected_link:
        raise AssertionError("Tier C link-rate relation drifted")
    if expectations.get("negative_controls") != [
        "acceptance_surrogate",
        "producer_constant",
        "missing_tx_start",
    ]:
        raise AssertionError("Tier C negative-control registry drifted")
    if set(expectations.get("accepted_abi_v1_sha256", {})) != {
        "tier_a_raw_observations.json",
        "tier_a_summary.json",
        "tier_b_raw_observations.json",
        "tier_b_results.json",
    }:
        raise AssertionError("Tier C ABI-v1 digest registry drifted")
    if expectations.get("tier_c_producer_argument_names") != [
        "--factory",
        "--expectations",
        "--observations",
    ]:
        raise AssertionError("Tier C producer invocation drifted")


def _parse_ctest(output: str) -> dict[str, int]:
    match = re.search(
        r"100% tests passed, (\d+) tests failed out of (\d+)",
        output,
    )
    if match is None:
        raise RuntimeError("could not parse CTest summary")
    failed = int(match.group(1))
    total = int(match.group(2))
    return {"passed": total - failed, "failed": failed, "total": total}


def _cmake_gate(
    source: Path,
    build: Path,
    configure_options: list[str],
) -> dict[str, int]:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            *configure_options,
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "4"],
        check=True,
    )
    completed = subprocess.run(
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    return _parse_ctest(completed.stdout)


def _build_native_suites(htsim_source: Path, out: Path) -> dict[str, Any]:
    htsim_build = out / "htsim-build"
    htsim_ctest = _cmake_gate(
        htsim_source / "htsim" / "sim",
        htsim_build,
        [
            "-DENABLE_TESTS=ON",
            "-DHTSIM_ENABLE_SIMLLM_RNIC=ON",
            f"-DSIMLLM_REPOSITORY_ROOT={REPO_ROOT}",
            "-DHTSIM_CREATE_SOURCE_SYMLINKS=OFF",
        ],
    )
    native_build = out / "simllm-native-build"
    native_ctest = _cmake_gate(
        REPO_ROOT / "simllm" / "backends" / "rnic",
        native_build,
        [
            "-DSIMLLM_RNIC_BUILD_TESTS=ON",
            "-DSIMLLM_RNIC_BUILD_TOOLS=ON",
            "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON",
        ],
    )
    return {
        "htsim_build": htsim_build,
        "htsim_ctest": htsim_ctest,
        "simllm_native_build": native_build,
        "simllm_native_ctest": native_ctest,
    }


def _executable(build: Path, *relative_candidates: str) -> Path:
    for relative in relative_candidates:
        candidate = build / relative
        if candidate.is_file():
            return candidate.resolve()
        windows = candidate.with_suffix(".exe")
        if windows.is_file():
            return windows.resolve()
    raise RuntimeError(
        f"build did not produce any of: {', '.join(relative_candidates)}"
    )


def _copy_executable(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | 0o111)
    return destination.resolve()


def _run_tier_a_v1(producer: Path, out: Path) -> dict[str, str]:
    local = _copy_executable(producer, out / "build" / producer.name)
    with _environment({"SIMLLM_TIER_A_RUN_ROOT": str(out)}):
        subprocess.run(
            [
                sys.executable,
                str(TIER_A_ACCEPTANCE),
                "--factory",
                "htsim",
                "--producer",
                str(local),
                "--run-dir",
                str(out),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    return {
        "tier_a_raw_observations.json": _digest(out / "raw_observations.json"),
        "tier_a_summary.json": _digest(out / "summary.json"),
    }


@contextmanager
def _environment(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _binary_environment(
    *,
    tier_a_producer: Path,
    rnic: Path,
    dcqcn: Path,
    txt2bin: Path,
    reference_rnic: Path,
    reference_dcqcn: Path,
) -> dict[str, str]:
    return {
        "SIMLLM_RNIC_TIER_A_PRODUCER": str(tier_a_producer),
        "SIMLLM_HTSIM_RNIC": str(rnic),
        "SIMLLM_TIER_B_REFERENCE_RNIC": str(reference_rnic),
        "SIMLLM_TIER_B_REFERENCE_DCQCN": str(reference_dcqcn),
        "SIMLLM_TIER_B_BYPASS_RNIC": str(rnic),
        "SIMLLM_TIER_B_BYPASS_DCQCN": str(dcqcn),
        "SIMLLM_TXT2BIN": str(txt2bin),
    }


def _run_tier_b_v1(out: Path, environment: dict[str, str]) -> dict[str, str]:
    launcher = _copy_executable(
        TIER_B_LAUNCHER,
        out / "build" / TIER_B_LAUNCHER.name,
    )
    with _environment(environment):
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "examples" / "core5_reduction" / "run_study.py"),
                "--out",
                str(out),
                "--tier-b-only",
                "--tier-b-producer",
                str(launcher),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    return {
        "tier_b_raw_observations.json": _digest(out / "raw_observations.json"),
        "tier_b_results.json": _digest(out / "results.json"),
    }


def _run_tier_c(
    out: Path,
    environment: dict[str, str],
    v1_observations: Path,
) -> dict[str, Any]:
    from examples.rnic_live_v1.tier_c_acceptance import run_acceptance

    launcher = _copy_executable(
        TIER_C_LAUNCHER,
        out / "build" / TIER_C_LAUNCHER.name,
    )
    with _environment(environment):
        return run_acceptance(out, launcher, v1_observations)


def _python_gate(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return {
        "command": command,
        "tail": lines[-1] if lines else "",
        "passed": True,
    }


def _run(
    htsim_source: Path,
    reference_rnic: Path,
    reference_dcqcn: Path,
    out: Path,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"Tier C output already exists: {out}")
    for reference in (reference_rnic, reference_dcqcn):
        if not reference.is_file():
            raise FileNotFoundError(f"Tier C reference binary is absent: {reference}")

    python_gates = {
        "ruff": _python_gate([str(REPO_ROOT / ".venv" / "bin" / "ruff"), "check", "."]),
        "pytest": _python_gate(
            [str(REPO_ROOT / ".venv" / "bin" / "pytest"), "-q"]
        ),
    }
    out.mkdir(parents=True)
    native = _build_native_suites(htsim_source, out)
    htsim_build = native["htsim_build"]
    tier_a_producer = _executable(htsim_build, "htsim_rnic_tier_a")
    rnic = _executable(htsim_build, "datacenter/htsim_rnic", "htsim_rnic")
    dcqcn = _executable(
        htsim_build,
        "datacenter/htsim_dcqcn_atlahs",
        "htsim_dcqcn_atlahs",
    )
    txt2bin = _executable(htsim_build, "txt2bin")
    environment = _binary_environment(
        tier_a_producer=tier_a_producer,
        rnic=rnic,
        dcqcn=dcqcn,
        txt2bin=txt2bin,
        reference_rnic=reference_rnic,
        reference_dcqcn=reference_dcqcn,
    )

    tier_a_dir = out / "abi-v1-tier-a"
    tier_b_dir = out / "abi-v1-tier-b"
    actual_digests = {
        **_run_tier_a_v1(tier_a_producer, tier_a_dir),
        **_run_tier_b_v1(tier_b_dir, environment),
    }
    expectations = _load_json(EXPECTATIONS)
    expected_digests = expectations["accepted_abi_v1_sha256"]
    if actual_digests != expected_digests:
        raise AssertionError(
            f"ABI-v1 off-path artifacts changed: {actual_digests}"
        )

    tier_c = _run_tier_c(
        out / "tier-c",
        environment,
        tier_b_dir / "raw_observations.json",
    )
    report = {
        "schema": "simllm-rnic-tier-c-run-v1",
        "expectation_commit": FREEZE_COMMIT,
        "simllm_revision": _revision(REPO_ROOT),
        "htsim_revision": _revision(htsim_source),
        "python_gates": python_gates,
        "native_gates": {
            "htsim_ctest": native["htsim_ctest"],
            "simllm_native_ctest": native["simllm_native_ctest"],
        },
        "abi_v1_artifact_identity": {
            "classification": "fatal_unscored_off_path",
            "actual_sha256": actual_digests,
            "expected_sha256": expected_digests,
            "passed": True,
        },
        "tier_c": tier_c,
    }
    (out / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--htsim-source", type=Path, required=True)
    parser.add_argument("--tier-b-reference-rnic", type=Path, required=True)
    parser.add_argument("--tier-b-reference-dcqcn", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    _validate_registry(
        arguments.htsim_source.resolve(),
        arguments.tier_b_reference_rnic.resolve(),
        arguments.tier_b_reference_dcqcn.resolve(),
        arguments.out.resolve(),
    )
    if not arguments.check_only:
        _run(
            arguments.htsim_source.resolve(strict=True),
            arguments.tier_b_reference_rnic.resolve(strict=True),
            arguments.tier_b_reference_dcqcn.resolve(strict=True),
            arguments.out.resolve(strict=False),
        )
        return
    print("Tier C registry check passed; no artifacts were produced")


if __name__ == "__main__":
    main()
