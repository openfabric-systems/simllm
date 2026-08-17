"""Run the frozen BACK-46 GPU fabric-endpoint study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "simllm" / "backends" / "rnic"
RESULTS = Path(__file__).with_name("results.csv")
RUN_ROOT_ENV = "SIMLLM_WAVE17_RUN_ROOT"

ARMS = ("host_bounce", "gpu_direct")
PAYLOAD_SIZES = (4096, 16384)
LANE_COUNTS = (8, 16)

HOST_ENDPOINT = 4000
NIC_ENDPOINT = 4001
GPU_ENDPOINT = 4002

#: PCIe generation 5 is 32 GT/s per lane under 128b/130b encoding.
GEN5_TRANSFERS_PER_SECOND_MILLIONS = 32_000
GEN5_ENCODING_NUMERATOR = 128
GEN5_ENCODING_DENOMINATOR = 130
MAX_PAYLOAD_SIZE_BYTES = 256
POSTED_WRITE_OVERHEAD_BYTES = 24

#: The frozen closed-form staging completions from the expectation record.
FROZEN_STAGING_PS = {
    (4096, 16): 71_094,
    (4096, 8): 142_188,
    (16384, 16): 284_375,
    (16384, 8): 568_750,
}

#: Accepted artifacts whose bytes this change must not move.
FROZEN_ARTIFACT_DIGESTS = {
    "examples/rnic_wq_v1/results.csv": (
        "598f0e10ca4e5a83a9dfb8ed8289e25cdc4c80fc24f92f2f70db967724be5682"
    ),
    "examples/rnic_pcie_v1/results.csv": (
        "464b92fd5327287db6b5e71a5449add5b893285bc3c4bcdf6a4950355339a5e2"
    ),
    "examples/rnic_device_v1/results.csv": (
        "7a0b8423d0a99de9538047f307bb7fd2f20c8d19bd408ef90fe02199da868934"
    ),
    "examples/rnic_device_v1/native_tests.csv": (
        "969963477314bfb723770556a02e4f038c7220820d522ae60dfa8c80744a202d"
    ),
    "examples/rnic_session_records_v1/results.json": (
        "d83575d1c873d3375bc24819c4d6eca0b85ea3a414fe8578f30262268a39fdf6"
    ),
    "examples/rnic_hostmem_v1/results.csv": (
        "1bc7bcc8e72b7aef9fda1ed7e6ca2078d60c48a00377cbf8dfded75ff4d2fa53"
    ),
    "examples/rnic_submission_v1/results.csv": (
        "8f74c6fd92d012f2c70c1c2b09d6f49a4d99bcc35fd418a239f7b577777edbc7"
    ),
}

#: Accepted artifacts this study regenerates from source rather than hashing.
REGENERATED_ARTIFACTS = {
    "examples/rnic_hostmem_v1/results.csv": "simllm_rnic_host_memory_test",
    "examples/rnic_submission_v1/results.csv": "simllm_rnic_submission_test",
}

ROW_FIELDS = (
    "arm",
    "payload_bytes",
    "lane_count",
    "data_endpoint",
    "payload_completer_endpoint",
    "payload_completer_kind",
    "gpu_completer_transactions",
    "gpu_completer_useful_bytes",
    "gpu_requester_transactions",
    "gpu_requester_useful_bytes",
    "nic_requester_transactions",
    "nic_requester_useful_bytes",
    "host_completer_transactions",
    "host_completer_useful_bytes",
    "staging_transfers",
    "staging_completed_ps",
    "staging_link_bytes",
    "staging_credit_wait_ps",
    "staging_link_queue_wait_ps",
    "wqe_admitted_ps",
    "wqe_cqe_visible_ps",
    "payload_read_transactions",
    "attributed_requester_transactions",
    "attributed_completer_transactions",
    "invariants_valid",
)

INTEGER_FIELDS = tuple(
    field
    for field in ROW_FIELDS
    if field not in {"arm", "data_endpoint", "payload_completer_kind"}
)


def _picoseconds_per_byte(lane_count: int) -> Fraction:
    """Exact serialization cost of one byte on the modeled generation-5 link."""
    return Fraction(
        8 * GEN5_ENCODING_DENOMINATOR * 1_000_000,
        GEN5_TRANSFERS_PER_SECOND_MILLIONS
        * lane_count
        * GEN5_ENCODING_NUMERATOR,
    )


def _staging_link_bytes(payload_bytes: int) -> int:
    if payload_bytes % MAX_PAYLOAD_SIZE_BYTES != 0:
        raise AssertionError("frozen payloads must be maximum-payload aligned")
    tlps = payload_bytes // MAX_PAYLOAD_SIZE_BYTES
    return payload_bytes + POSTED_WRITE_OVERHEAD_BYTES * tlps


def _ceil_picoseconds(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _expected_staging_ps(payload_bytes: int, lane_count: int) -> int:
    rate = _picoseconds_per_byte(lane_count)
    return _ceil_picoseconds(_staging_link_bytes(payload_bytes) * rate)


def _payload_floor_ps(payload_bytes: int, lane_count: int) -> int:
    """No transfer of these bytes can beat its own link serialization."""
    return _ceil_picoseconds(payload_bytes * _picoseconds_per_byte(lane_count))


def _run_root() -> Path:
    configured = os.environ.get(RUN_ROOT_ENV)
    if not configured:
        raise RuntimeError(
            f"{RUN_ROOT_ENV} must name the external wave-17 run root"
        )
    return Path(configured).resolve()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_registry(out: Path) -> None:
    cells = {
        (arm, payload, lanes)
        for arm in ARMS
        for payload in PAYLOAD_SIZES
        for lanes in LANE_COUNTS
    }
    if len(cells) != 8:
        raise AssertionError("BACK-46 registry must contain eight unique cells")
    if len(ROW_FIELDS) != len(set(ROW_FIELDS)):
        raise AssertionError("BACK-46 row fields must be unique")
    if set(FROZEN_STAGING_PS) != {
        (payload, lanes) for payload in PAYLOAD_SIZES for lanes in LANE_COUNTS
    }:
        raise AssertionError("BACK-46 staging registry is incomplete")
    for (payload, lanes), frozen in FROZEN_STAGING_PS.items():
        derived = _expected_staging_ps(payload, lanes)
        if derived != frozen:
            raise AssertionError(
                "BACK-46 frozen staging literal disagrees with the closed "
                f"form for payload {payload} at {lanes} lanes: "
                f"{frozen} against {derived}"
            )
        if derived <= _payload_floor_ps(payload, lanes):
            raise AssertionError(
                "BACK-46 staging closed form sits below its physical floor"
            )
    for relative, expected in FROZEN_ARTIFACT_DIGESTS.items():
        actual = _digest(REPO_ROOT / relative)
        if actual != expected:
            raise AssertionError(
                f"frozen artifact digest drifted for {relative}: {actual}"
            )
    try:
        out.resolve().relative_to(_run_root())
    except ValueError as error:
        raise ValueError(
            f"BACK-46 output must remain under {RUN_ROOT_ENV}"
        ) from error


def _native_executable(build_dir: Path, name: str) -> Path:
    candidates = (
        build_dir / name,
        build_dir / f"{name}.exe",
        build_dir / "Release" / name,
        build_dir / "Release" / f"{name}.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"native executable not found; checked {joined}")


def _build(build_dir: Path) -> tuple[Path, dict[str, int]]:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSIMLLM_RNIC_BUILD_TESTS=ON",
            "-DSIMLLM_RNIC_BUILD_TOOLS=ON",
            "-DSIMLLM_RNIC_WARNINGS_AS_ERRORS=ON",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
        ],
        check=True,
    )
    completed = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "-C",
            "Release",
            "--output-on-failure",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    match = re.search(
        r"100% tests passed, (\d+) tests failed out of (\d+)",
        completed.stdout,
    )
    if match is None:
        raise RuntimeError("could not parse the native CTest summary")
    failed = int(match.group(1))
    total = int(match.group(2))
    return build_dir, {
        "passed": total - failed,
        "failed": failed,
        "total": total,
    }


def _study_csv(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "--study-csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _regenerate_off_path(build_dir: Path, out: Path) -> dict[str, bool]:
    """Re-derives every accepted artifact this change could have perturbed.

    Hashing the tracked file only proves the file did not move. Rebuilding the
    native library and re-emitting the accepted rows is what makes the off-path
    lock sensitive to a C++ change.
    """
    identity: dict[str, bool] = {}
    for relative, executable_name in REGENERATED_ARTIFACTS.items():
        rendered = _study_csv(
            _native_executable(build_dir, executable_name)
        ).encode("utf-8")
        (out / Path(relative).parent.name).with_suffix(".regenerated.csv").write_bytes(
            rendered
        )
        identity[relative] = (REPO_ROOT / relative).read_bytes() == rendered
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples" / "rnic_pcie_v1" / "run_rnic_pcie_v1.py"),
            "--check",
            "--build-dir",
            str(out / "build_pcie_check"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    print(completed.stdout, end="")
    identity["examples/rnic_pcie_v1/results.csv"] = completed.returncode == 0
    if completed.returncode != 0:
        print(completed.stderr, end="")
    return identity


def _require_integer(row: dict[str, str], field: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise AssertionError(f"BACK-46 field {field} must be an integer") from error


def _validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    expected_cells = {
        (arm, str(payload), str(lanes))
        for arm in ARMS
        for payload in PAYLOAD_SIZES
        for lanes in LANE_COUNTS
    }
    actual_cells = {
        (row["arm"], row["payload_bytes"], row["lane_count"]) for row in rows
    }
    if len(rows) != 8 or actual_cells != expected_cells:
        raise AssertionError("BACK-46 native rows differ from the frozen grid")

    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    structural: list[str] = []
    for row in rows:
        values = {field: _require_integer(row, field) for field in INTEGER_FIELDS}
        key = (row["arm"], values["payload_bytes"], values["lane_count"])
        indexed[key] = {**row, **values}
        cell = f"A={key[0]},P={key[1]},L={key[2]}"
        bounce = row["arm"] == "host_bounce"
        expected_endpoint = HOST_ENDPOINT if bounce else GPU_ENDPOINT
        expected_kind = "host" if bounce else "gpu"
        expected_data = "host_pinned_memory" if bounce else "gpu_memory"
        if (
            row["data_endpoint"] != expected_data
            or row["payload_completer_kind"] != expected_kind
            or values["payload_completer_endpoint"] != expected_endpoint
            or values["nic_requester_transactions"] != 8
            or values["payload_read_transactions"] != 1
            or values["invariants_valid"] != 1
            or values["attributed_requester_transactions"]
            != values["attributed_completer_transactions"]
        ):
            structural.append(f"{cell}: endpoint or ledger labels disagree")
        if bounce:
            if (
                values["staging_transfers"] != 1
                or values["gpu_completer_transactions"] != 0
                or values["gpu_requester_transactions"] != 1
                or values["gpu_requester_useful_bytes"] != values["payload_bytes"]
                or values["staging_credit_wait_ps"] != 0
                or values["staging_link_queue_wait_ps"] != 0
            ):
                structural.append(f"{cell}: host-bounce staging shape disagrees")
        elif (
            values["staging_transfers"] != 0
            or values["staging_completed_ps"] != 0
            or values["gpu_requester_transactions"] != 0
            or values["gpu_completer_transactions"] != 1
        ):
            structural.append(f"{cell}: GPU-direct shape disagrees")
    if structural:
        raise AssertionError(
            "BACK-46 fatal structural failures: " + "; ".join(structural)
        )

    families: dict[str, list[dict[str, Any]]] = {
        "arm_ordering": [],
        "bounce_penalty_equals_staging": [],
        "staging_closed_form": [],
        "gpu_completer_charge": [],
    }
    for payload in PAYLOAD_SIZES:
        for lanes in LANE_COUNTS:
            bounce = indexed[("host_bounce", payload, lanes)]
            direct = indexed[("gpu_direct", payload, lanes)]
            cell = f"P={payload},L={lanes}"
            families["arm_ordering"].append(
                {
                    "cell": cell,
                    "bounce_ps": bounce["wqe_cqe_visible_ps"],
                    "direct_ps": direct["wqe_cqe_visible_ps"],
                    "passed": bounce["wqe_cqe_visible_ps"]
                    > direct["wqe_cqe_visible_ps"],
                }
            )
            difference = (
                bounce["wqe_cqe_visible_ps"] - direct["wqe_cqe_visible_ps"]
            )
            families["bounce_penalty_equals_staging"].append(
                {
                    "cell": cell,
                    "difference_ps": difference,
                    "staging_ps": bounce["staging_completed_ps"],
                    "passed": difference == bounce["staging_completed_ps"],
                }
            )
            expected_ps = _expected_staging_ps(payload, lanes)
            floor_ps = _payload_floor_ps(payload, lanes)
            families["staging_closed_form"].append(
                {
                    "cell": cell,
                    "measured_ps": bounce["staging_completed_ps"],
                    "expected_ps": expected_ps,
                    "frozen_ps": FROZEN_STAGING_PS[(payload, lanes)],
                    "floor_ps": floor_ps,
                    "link_bytes": bounce["staging_link_bytes"],
                    "expected_link_bytes": _staging_link_bytes(payload),
                    "passed": bounce["staging_completed_ps"] == expected_ps
                    and bounce["staging_link_bytes"]
                    == _staging_link_bytes(payload)
                    and bounce["staging_completed_ps"] > floor_ps,
                }
            )
            charge = (
                direct["gpu_completer_useful_bytes"]
                - bounce["gpu_completer_useful_bytes"]
            )
            families["gpu_completer_charge"].append(
                {
                    "cell": cell,
                    "charge_bytes": charge,
                    "payload_bytes": payload,
                    "passed": charge == payload,
                }
            )
    scored_passed = sum(
        1
        for instances in families.values()
        for instance in instances
        if instance["passed"]
    )
    scored_total = sum(len(instances) for instances in families.values())
    return {
        "rows": list(indexed.values()),
        "families": families,
        "scored_passed": scored_passed,
        "scored_total": scored_total,
    }


def _run(out: Path) -> dict[str, Any]:
    before = {
        relative: _digest(REPO_ROOT / relative)
        for relative in FROZEN_ARTIFACT_DIGESTS
    }
    out.mkdir(parents=True, exist_ok=True)
    build_dir, ctest = _build(out / "build")
    raw_csv = _study_csv(
        _native_executable(build_dir, "simllm_rnic_gpu_device_test")
    )
    reader = csv.DictReader(io.StringIO(raw_csv))
    if tuple(reader.fieldnames or ()) != ROW_FIELDS:
        raise RuntimeError("BACK-46 native row schema drifted")
    checked = _validate_rows(list(reader))
    regenerated = _regenerate_off_path(build_dir, out)
    (out / "raw_results.csv").write_bytes(raw_csv.encode("utf-8"))
    RESULTS.write_bytes(raw_csv.encode("utf-8"))
    after = {
        relative: _digest(REPO_ROOT / relative)
        for relative in FROZEN_ARTIFACT_DIGESTS
    }
    identity = {
        relative: before[relative] == after[relative] == expected
        for relative, expected in FROZEN_ARTIFACT_DIGESTS.items()
    }
    if not all(identity.values()):
        raise AssertionError("BACK-46 accepted artifact byte identity failed")
    if not all(regenerated.values()):
        raise AssertionError(
            "BACK-46 regenerated off-path bytes differ from the accepted "
            "artifacts"
        )
    return {
        **checked,
        "artifact_identity": identity,
        "artifact_identity_total": len(identity),
        "regenerated_identity": regenerated,
        "ctest": ctest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the frozen registry without creating outputs",
    )
    arguments = parser.parse_args()
    _validate_registry(arguments.out)
    if arguments.check_only:
        print(
            "RNIC GPU fabric-endpoint registry check passed; "
            "no artifacts were produced"
        )
        return
    summary = _run(arguments.out.resolve())
    print(
        "BACK-46 scored "
        f"{summary['scored_passed']}/{summary['scored_total']} relation "
        f"instances; native CTest {summary['ctest']['passed']}/"
        f"{summary['ctest']['total']} passed"
    )
    for name, instances in summary["families"].items():
        passed = sum(1 for instance in instances if instance["passed"])
        print(f"  {name}: {passed}/{len(instances)}")


if __name__ == "__main__":
    main()
