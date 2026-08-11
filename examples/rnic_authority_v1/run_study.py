"""Build and run the frozen CORE-21 and BACK-31 authority study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simllm._native import cmake_binary_candidates, is_runnable_file
from simllm.backends.rnic_records import (
    BypassArtifactComparison,
    BypassArtifacts,
    compare_bypass_artifacts,
)

STUDY_DIR = Path(__file__).resolve().parent
EXPECTATIONS_PATH = STUDY_DIR / "expectations.json"
PRODUCER_PATH = STUDY_DIR / "produce_observations.py"
CHECKER_PATH = STUDY_DIR / "check_results.py"
FROZEN_EXPECTATIONS_COMMIT = "1871cfcc2e343205f9e78693c0aa87b7b34942e9"
FROZEN_SIMLLM_BASE = "90ada43070adb3b1e624b6819aff34d8620e8571"
FROZEN_HTSIM_SOURCE = "4885c647eecdfdf81479d1df052223c016ad086b"
CONFIGURATION = "Release"
POSITIVE_TARGETS = ("htsim_rnic_tier_a", "htsim_rnic", "txt2bin")
NEGATIVE_TARGETS = ("htsim_rnic", "txt2bin")
BINARY_LAYOUT = {
    "tier_a_producer": ("htsim_rnic_tier_a", None),
    "htsim_rnic": ("htsim_rnic", "datacenter"),
    "txt2bin": ("txt2bin", None),
}
ARTIFACT_BYTE_FIELDS = (
    "goal_text",
    "goal_binary",
    "topology",
    "completion_csv",
    "canonical_completion",
    "step_results",
    "replay_summary",
)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=check,
        capture_output=True,
        text=text,
    )


def _git_head(repository: Path) -> str:
    return _git(repository, ("rev-parse", "HEAD")).stdout.strip()


def _git_status(repository: Path) -> str:
    return _git(
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    ).stdout


def _require_known_commit(repository: Path, commit: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise AssertionError(f"{name} must be a full lowercase commit hash")
    result = _git(
        repository,
        ("cat-file", "-e", f"{commit}^{{commit}}"),
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{name} is not available in {repository}")


def _require_ancestor(repository: Path, ancestor: str, name: str) -> None:
    result = _git(
        repository,
        ("merge-base", "--is-ancestor", ancestor, "HEAD"),
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"{name} is not an ancestor of the current checkout")


def _require_frozen_file(relative: str) -> None:
    committed = _git(
        ROOT,
        ("show", f"{FROZEN_EXPECTATIONS_COMMIT}:{relative}"),
        text=False,
    ).stdout
    if (ROOT / relative).read_bytes() != committed:
        raise AssertionError(f"frozen expectation file drifted: {relative}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _path_absent(path: Path) -> bool:
    return not path.exists() and not path.is_symlink()


def _require_under(path: Path, root: Path, name: str) -> None:
    if not _is_within(path, root):
        raise ValueError(f"{name} must remain under {root}")


def _binary_candidates(build_root: Path, label: str) -> list[Path]:
    binary_name, subdirectory = BINARY_LAYOUT[label]
    return cmake_binary_candidates(
        build_root,
        binary_name,
        subdirectory=subdirectory,
    )


def _planned_binary(build_root: Path, label: str) -> Path:
    candidates = _binary_candidates(build_root, label)
    if not candidates:
        raise AssertionError(f"no CMake candidates registered for {label}")
    return candidates[0]


def _configure_command(
    htsim_source: Path,
    build_root: Path,
    link_mode: str,
) -> list[str]:
    return [
        "cmake",
        "-S",
        str(htsim_source / "htsim" / "sim"),
        "-B",
        str(build_root),
        f"-DCMAKE_BUILD_TYPE={CONFIGURATION}",
        f"-DHTSIM_ENABLE_SIMLLM_RNIC={link_mode}",
        "-DHTSIM_CREATE_SOURCE_SYMLINKS=OFF",
        f"-DSIMLLM_REPOSITORY_ROOT={ROOT}",
    ]


def _build_command(build_root: Path, targets: Sequence[str]) -> list[str]:
    return [
        "cmake",
        "--build",
        str(build_root),
        "--config",
        CONFIGURATION,
        "--parallel",
        "4",
        "--target",
        *targets,
    ]


def _producer_command(
    observations: Path,
    tier_a_producer: Path,
    htsim_rnic: Path,
    txt2bin: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PRODUCER_PATH),
        "--expectations",
        str(EXPECTATIONS_PATH),
        "--observations",
        str(observations),
        "--tier-a-producer",
        str(tier_a_producer),
        "--htsim-rnic",
        str(htsim_rnic),
        "--txt2bin",
        str(txt2bin),
    ]


def _checker_command(
    observations: Path,
    negative_evidence: Path,
    results: Path,
) -> list[str]:
    return [
        sys.executable,
        str(CHECKER_PATH),
        "--expectations",
        str(EXPECTATIONS_PATH),
        "--observations",
        str(observations),
        "--negative-evidence",
        str(negative_evidence),
        "--results",
        str(results),
    ]


def _option_names(command: Sequence[str]) -> list[str]:
    return [argument for argument in command[2:] if argument.startswith("--")]


def _validate_expectations(expectations: Mapping[str, Any]) -> None:
    if expectations.get("schema") != "simllm-rnic-authority-expectations-v1":
        raise AssertionError("authority expectation schema drifted")
    if expectations.get("simllm_base_commit") != FROZEN_SIMLLM_BASE:
        raise AssertionError("SimLLM source anchor drifted")
    if expectations.get("htsim_source_commit") != FROZEN_HTSIM_SOURCE:
        raise AssertionError("htsim source anchor drifted")
    if expectations.get("sweep") != {
        "link_rate_gbps": [200, 400],
        "doorbell_service_ps": [0, 1000],
    }:
        raise AssertionError("authority parameter matrix drifted")

    graph = expectations.get("fixed_graph")
    if not isinstance(graph, dict):
        raise TypeError("fixed_graph must be an object")
    if graph.get("operation_ids") != ["core21-contended-send"]:
        raise AssertionError("fixed graph operation inventory drifted")
    if graph.get("destination_ranks") != [8, 16]:
        raise AssertionError("fixed graph destinations drifted")
    if graph.get("payload_bytes") != 4096 or graph.get("wqe_count") != 2:
        raise AssertionError("fixed graph contention shape drifted")

    wire_service = {
        rate: graph["payload_bytes"] * 8 * 1000 // rate
        for rate in expectations["sweep"]["link_rate_gbps"]
    }
    if wire_service != {200: 163_840, 400: 81_920}:
        raise AssertionError("wire-service closed form drifted")
    relation = expectations.get("signed_authority_relation")
    if not isinstance(relation, dict):
        raise TypeError("signed_authority_relation must be an object")
    if (
        relation.get("direction") != "positive"
        or relation.get("lower_bound_ps") != 1000
        or relation.get("upper_bound_ps") != 1000
        or relation.get("instances") != 6
    ):
        raise AssertionError("signed authority relation drifted")
    inverse = expectations.get("inverse_rate_relation")
    if not isinstance(inverse, dict):
        raise TypeError("inverse_rate_relation must be an object")
    if (
        inverse.get("lower_bound_ps") != 163_840
        or inverse.get("upper_bound_ps") != 163_840
        or inverse.get("instances") != 12
    ):
        raise AssertionError("inverse-rate relation drifted")

    builds = expectations.get("builds")
    if builds != {
        "positive": {
            "HTSIM_ENABLE_SIMLLM_RNIC": "ON",
            "HTSIM_CREATE_SOURCE_SYMLINKS": "OFF",
            "targets": list(POSITIVE_TARGETS),
        },
        "negative": {
            "HTSIM_ENABLE_SIMLLM_RNIC": "OFF",
            "HTSIM_CREATE_SOURCE_SYMLINKS": "OFF",
            "targets": list(NEGATIVE_TARGETS),
            "absent_target": "htsim_rnic_tier_a",
        },
    }:
        raise AssertionError("authority build registry drifted")
    if expectations.get("producer_argument_names") != [
        "--expectations",
        "--observations",
        "--tier-a-producer",
        "--htsim-rnic",
        "--txt2bin",
    ]:
        raise AssertionError("producer argument registry drifted")
    if expectations.get("checker_argument_names") != [
        "--expectations",
        "--observations",
        "--negative-evidence",
        "--results",
    ]:
        raise AssertionError("checker argument registry drifted")
    negative = expectations.get("negative_control")
    if not isinstance(negative, dict):
        raise TypeError("negative_control must be an object")
    if (
        negative.get("expected_producer_exit") != 2
        or negative.get("expected_checker_exit") != 2
        or negative.get("forbidden_outputs")
        != [
            "raw_observations.json",
            "raw_observations.json.tmp",
            "results.json",
            "results.json.tmp",
        ]
    ):
        raise AssertionError("negative control registry drifted")


def _validate_command_plan(
    expectations: Mapping[str, Any],
    out: Path,
    htsim_source: Path,
) -> None:
    positive_build = out / "positive" / "build"
    negative_build = out / "negative" / "build"
    positive_observations = out / "positive" / "raw_observations.json"
    negative_observations = out / "negative" / "raw_observations.json"
    negative_evidence = out / "negative" / "evidence.json"

    positive_configure = _configure_command(htsim_source, positive_build, "ON")
    negative_configure = _configure_command(htsim_source, negative_build, "OFF")
    for command, mode in (
        (positive_configure, "ON"),
        (negative_configure, "OFF"),
    ):
        if f"-DHTSIM_ENABLE_SIMLLM_RNIC={mode}" not in command:
            raise AssertionError(f"{mode} configure command lost its link mode")
        if "-DHTSIM_CREATE_SOURCE_SYMLINKS=OFF" not in command:
            raise AssertionError("configure command could mutate the source tree")
        if f"-DSIMLLM_REPOSITORY_ROOT={ROOT}" not in command:
            raise AssertionError("configure command lost the common SimLLM root")
    if _build_command(positive_build, POSITIVE_TARGETS)[-3:] != list(POSITIVE_TARGETS):
        raise AssertionError("positive target plan drifted")
    if _build_command(negative_build, NEGATIVE_TARGETS)[-2:] != list(NEGATIVE_TARGETS):
        raise AssertionError("negative target plan drifted")

    positive_producer = _producer_command(
        positive_observations,
        _planned_binary(positive_build, "tier_a_producer"),
        _planned_binary(positive_build, "htsim_rnic"),
        _planned_binary(positive_build, "txt2bin"),
    )
    negative_producer = _producer_command(
        negative_observations,
        _planned_binary(negative_build, "tier_a_producer"),
        _planned_binary(negative_build, "htsim_rnic"),
        _planned_binary(negative_build, "txt2bin"),
    )
    positive_checker = _checker_command(
        positive_observations,
        negative_evidence,
        out / "positive" / "results.json",
    )
    negative_checker = _checker_command(
        negative_observations,
        negative_evidence,
        out / "negative" / "results.json",
    )
    for command in (positive_producer, negative_producer):
        if _option_names(command) != expectations["producer_argument_names"]:
            raise AssertionError("producer command plan drifted")
    for command in (positive_checker, negative_checker):
        if _option_names(command) != expectations["checker_argument_names"]:
            raise AssertionError("checker command plan drifted")
    for path, name in (
        (positive_build, "positive build"),
        (negative_build, "negative build"),
        (positive_observations, "positive observations"),
        (negative_observations, "negative observations"),
        (negative_evidence, "negative evidence"),
    ):
        _require_under(path, out, name)
    for build_root, labels in (
        (positive_build, BINARY_LAYOUT),
        (negative_build, BINARY_LAYOUT),
    ):
        for label in labels:
            for candidate in _binary_candidates(build_root, label):
                _require_under(candidate, build_root, f"{label} candidate")


def _validate_registry(out: Path, htsim_source: Path) -> dict[str, Any]:
    if not out.is_absolute():
        raise ValueError("--out must be an absolute path")
    if not htsim_source.is_absolute():
        raise ValueError("--htsim-source must be an absolute path")
    if not _path_absent(out):
        raise FileExistsError("fresh authority output root already exists")
    resolved_out = out.resolve(strict=False)
    resolved_source = htsim_source.resolve(strict=True)
    resolved_root = ROOT.resolve(strict=True)
    if _paths_overlap(resolved_out, resolved_root):
        raise ValueError("authority output must remain outside the repository")
    if _paths_overlap(resolved_out, resolved_source):
        raise ValueError("authority output and htsim source must be disjoint")
    top_level = Path(
        _git(resolved_source, ("rev-parse", "--show-toplevel")).stdout.strip()
    ).resolve(strict=True)
    if top_level != resolved_source:
        raise ValueError("--htsim-source must name the htsim repository root")
    if not (resolved_source / "htsim" / "sim" / "CMakeLists.txt").is_file():
        raise FileNotFoundError("htsim source is missing htsim/sim/CMakeLists.txt")
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake must be available on PATH")
    if not Path(sys.executable).is_file():
        raise RuntimeError("the current Python executable does not exist")
    for path in (EXPECTATIONS_PATH, PRODUCER_PATH, CHECKER_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"registered study path is missing: {path}")

    expectations = _load_json(EXPECTATIONS_PATH, "authority expectations")
    _validate_expectations(expectations)
    _require_known_commit(ROOT, FROZEN_SIMLLM_BASE, "SimLLM source anchor")
    _require_known_commit(
        ROOT,
        FROZEN_EXPECTATIONS_COMMIT,
        "authority expectation commit",
    )
    _require_ancestor(ROOT, FROZEN_SIMLLM_BASE, "SimLLM source anchor")
    _require_ancestor(
        ROOT,
        FROZEN_EXPECTATIONS_COMMIT,
        "authority expectation commit",
    )
    for relative in (
        "examples/rnic_authority_v1/expectations.json",
        "examples/rnic_authority_v1/expectations.md",
    ):
        _require_frozen_file(relative)
    if _git_head(resolved_source) != FROZEN_HTSIM_SOURCE:
        raise AssertionError("htsim source HEAD differs from the frozen commit")
    if _git_status(resolved_source):
        raise RuntimeError("htsim source must be clean before the study")
    _validate_command_plan(expectations, resolved_out, resolved_source)
    return expectations


def _cache_value(build_root: Path, name: str) -> str:
    cache = build_root / "CMakeCache.txt"
    values = []
    for line in cache.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith(f"{name}:") and "=" in line:
            values.append(line.split("=", 1)[1])
    if len(values) != 1:
        raise AssertionError(f"CMake cache must contain one {name} entry")
    return values[0]


def _validate_cache(
    build_root: Path,
    htsim_source: Path,
    link_mode: str,
) -> None:
    expected = {
        "HTSIM_ENABLE_SIMLLM_RNIC": link_mode,
        "HTSIM_CREATE_SOURCE_SYMLINKS": "OFF",
    }
    for name, value in expected.items():
        if _cache_value(build_root, name) != value:
            raise AssertionError(f"CMake cache {name} is not {value}")
    configured_root = Path(_cache_value(build_root, "SIMLLM_REPOSITORY_ROOT")).resolve(strict=False)
    if configured_root != ROOT.resolve(strict=True):
        raise AssertionError("CMake cache uses a different SimLLM root")
    configured_source = Path(_cache_value(build_root, "CMAKE_HOME_DIRECTORY")).resolve(strict=True)
    if configured_source != (htsim_source / "htsim" / "sim").resolve(strict=True):
        raise AssertionError("CMake cache uses a different htsim source")


def _find_binary(build_root: Path, label: str) -> Path:
    candidates = _binary_candidates(build_root, label)
    matches = [candidate.resolve() for candidate in candidates if is_runnable_file(candidate)]
    if len(matches) != 1:
        rendered = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            f"expected one runnable {label} binary, found {len(matches)}; checked {rendered}"
        )
    return matches[0]


def _build(
    htsim_source: Path,
    build_root: Path,
    link_mode: str,
    targets: Sequence[str],
) -> dict[str, Path]:
    subprocess.run(
        _configure_command(htsim_source, build_root, link_mode),
        check=True,
    )
    subprocess.run(_build_command(build_root, targets), check=True)
    _validate_cache(build_root, htsim_source, link_mode)
    labels = (
        ("tier_a_producer", "htsim_rnic", "txt2bin")
        if link_mode == "ON"
        else ("htsim_rnic", "txt2bin")
    )
    return {label: _find_binary(build_root, label) for label in labels}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _accepted_bypass_artifacts(observations_path: Path) -> BypassArtifacts:
    from examples.rnic_authority_v1.produce_observations import (
        bypass_artifacts_from_mode,
    )

    observations = _load_json(observations_path, "positive observations")
    cells = observations.get("cells")
    if not isinstance(cells, list):
        raise TypeError("positive observations cells must be an array")
    selected = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and cell.get("link_rate_gbps") == 400
        and cell.get("doorbell_service_ps") == 1000
    ]
    if len(selected) != 1:
        raise AssertionError("positive observations need one 400G, D1000 cell")
    bypass = selected[0].get("bypass")
    if not isinstance(bypass, dict):
        raise TypeError("selected positive cell omitted its bypass mode")
    return bypass_artifacts_from_mode(bypass)


def compare_bypass_replay(
    reference: BypassArtifacts,
    replay_mode: Mapping[str, Any],
) -> BypassArtifactComparison:
    """Compare a replayed mode with the accepted standard bypass bundle."""

    from examples.rnic_authority_v1.produce_observations import (
        bypass_artifacts_from_mode,
    )

    return compare_bypass_artifacts(
        reference,
        bypass_artifacts_from_mode(replay_mode),
    )


def _positive_assets(
    binaries: Mapping[str, Path],
    observations_path: Path,
    native_observations_path: Path,
    bypass: BypassArtifacts,
) -> dict[str, str]:
    assets = {
        "tier_a_producer_sha256": _sha256(binaries["tier_a_producer"]),
        "htsim_rnic_sha256": _sha256(binaries["htsim_rnic"]),
        "txt2bin_sha256": _sha256(binaries["txt2bin"]),
        "raw_observations_sha256": _sha256(observations_path),
        "native_observations_sha256": _sha256(native_observations_path),
    }
    for field in ARTIFACT_BYTE_FIELDS:
        assets[f"bypass_{field}_sha256"] = _sha256_bytes(getattr(bypass, field))
    return assets


def _expected_failure(
    command: Sequence[str], expected_returncode: int, diagnostic: str
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != expected_returncode:
        raise RuntimeError(
            f"negative command exited {completed.returncode}, expected "
            f"{expected_returncode}: {' '.join(command)}\n{completed.stderr}"
        )
    if completed.stdout:
        raise RuntimeError("negative command unexpectedly wrote to stdout")
    if completed.stderr.strip() != diagnostic:
        raise RuntimeError(
            "negative command diagnostic drifted: "
            f"expected {diagnostic!r}, got {completed.stderr.strip()!r}"
        )
    return completed


def _forbidden_outputs(negative_root: Path, expectations: Mapping[str, Any]) -> dict[str, bool]:
    names = expectations["negative_control"]["forbidden_outputs"]
    result = {name: _path_absent(negative_root / name) for name in names}
    if not all(result.values()):
        present = [name for name, absent in result.items() if not absent]
        raise AssertionError(f"negative control published forbidden outputs: {present}")
    return result


def _publish_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if not _path_absent(path) or not _path_absent(temporary):
        raise FileExistsError("negative evidence or temporary file already exists")
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _source_output_candidates(htsim_source: Path) -> tuple[Path, ...]:
    sim = htsim_source / "htsim" / "sim"
    names = ("htsim_rnic", "htsim_rnic.exe")
    return tuple(sim / "datacenter" / name for name in names)


def _require_source_unchanged(htsim_source: Path) -> None:
    if _git_head(htsim_source) != FROZEN_HTSIM_SOURCE:
        raise AssertionError("htsim source commit changed during the study")
    if _git_status(htsim_source):
        raise AssertionError("htsim source tree changed during the study")
    if any(path.exists() or path.is_symlink() for path in _source_output_candidates(htsim_source)):
        raise AssertionError("htsim build created a source-tree executable link")


def _require_simllm_unchanged(expected_head: str) -> None:
    if _git_head(ROOT) != expected_head:
        raise AssertionError("SimLLM source commit changed during the study")
    if _git_status(ROOT):
        raise AssertionError("SimLLM source tree changed during the study")


def _run(
    expectations: dict[str, Any],
    out: Path,
    htsim_source: Path,
) -> None:
    if _git_status(ROOT):
        raise RuntimeError("SimLLM checkout must be clean before producing results")
    simllm_head = _git_head(ROOT)
    _require_source_unchanged(htsim_source)
    out.mkdir(parents=True, exist_ok=False)
    positive_root = out / "positive"
    negative_root = out / "negative"
    positive_root.mkdir()
    negative_root.mkdir()

    positive_build = positive_root / "build"
    positive_binaries = _build(
        htsim_source,
        positive_build,
        "ON",
        POSITIVE_TARGETS,
    )
    _require_source_unchanged(htsim_source)
    positive_observations = positive_root / "raw_observations.json"
    positive_native = positive_root / "native_observations.json"
    subprocess.run(
        _producer_command(
            positive_observations,
            positive_binaries["tier_a_producer"],
            positive_binaries["htsim_rnic"],
            positive_binaries["txt2bin"],
        ),
        check=True,
    )
    if not positive_observations.is_file() or not positive_native.is_file():
        raise AssertionError("positive producer omitted an accepted observation artifact")
    positive_raw = _load_json(positive_observations, "positive observations")
    if positive_raw.get("simllm_source_commit") != simllm_head:
        raise AssertionError("positive observations do not identify the clean implementation HEAD")
    accepted_bypass = _accepted_bypass_artifacts(positive_observations)
    positive_assets_before = _positive_assets(
        positive_binaries,
        positive_observations,
        positive_native,
        accepted_bypass,
    )

    negative_build = negative_root / "build"
    negative_binaries = _build(
        htsim_source,
        negative_build,
        "OFF",
        NEGATIVE_TARGETS,
    )
    _require_source_unchanged(htsim_source)
    negative_tier_a_candidates = _binary_candidates(negative_build, "tier_a_producer")
    negative_tier_a_target_absent = not any(
        candidate.exists() or candidate.is_symlink() for candidate in negative_tier_a_candidates
    )
    if not negative_tier_a_target_absent:
        raise AssertionError("link-disabled build produced htsim_rnic_tier_a")
    negative_main_sha256 = _sha256(negative_binaries["htsim_rnic"])

    negative_observations = negative_root / "raw_observations.json"
    negative_results = negative_root / "results.json"
    negative_evidence_path = negative_root / "evidence.json"
    negative_control = expectations["negative_control"]
    producer = _expected_failure(
        _producer_command(
            negative_observations,
            negative_tier_a_candidates[0],
            negative_binaries["htsim_rnic"],
            negative_binaries["txt2bin"],
        ),
        negative_control["expected_producer_exit"],
        negative_control["producer_diagnostic"],
    )
    _forbidden_outputs(negative_root, expectations)
    negative_native = negative_root / "native_observations.json"
    if not _path_absent(negative_native) or not _path_absent(Path(f"{negative_native}.tmp")):
        raise AssertionError("negative producer published native observations")
    checker = _expected_failure(
        _checker_command(
            negative_observations,
            negative_evidence_path,
            negative_results,
        ),
        negative_control["expected_checker_exit"],
        negative_control["checker_diagnostic"],
    )
    forbidden_outputs_absent = _forbidden_outputs(negative_root, expectations)
    if not _path_absent(negative_native) or not _path_absent(Path(f"{negative_native}.tmp")):
        raise AssertionError("negative checker published native observations")
    if _sha256(negative_binaries["htsim_rnic"]) != negative_main_sha256:
        raise AssertionError("negative composed binary changed while it ran")

    accepted_bypass_after = _accepted_bypass_artifacts(positive_observations)
    from examples.rnic_authority_v1.produce_observations import (
        bypass_artifacts_from_mode,
        run_bypass_replay,
    )

    replay_mode, replay_artifacts = run_bypass_replay(expectations, rate_gbps=400)
    comparison = compare_bypass_replay(accepted_bypass_after, replay_mode)
    if replay_artifacts != bypass_artifacts_from_mode(replay_mode):
        raise AssertionError("bypass replay row and artifact bundle disagree")
    if not comparison.equivalent:
        raise AssertionError(
            "post-negative bypass replay changed: "
            f"inputs={list(comparison.changed_inputs)}, "
            f"artifacts={list(comparison.changed_artifacts)}"
        )
    positive_assets_after = _positive_assets(
        positive_binaries,
        positive_observations,
        positive_native,
        accepted_bypass_after,
    )
    positive_assets_preserved = positive_assets_before == positive_assets_after
    if not positive_assets_preserved:
        raise AssertionError("positive binaries or artifacts changed in the negative run")

    evidence = {
        "schema": "simllm-rnic-authority-negative-evidence-v1",
        "htsim_source_commit": FROZEN_HTSIM_SOURCE,
        "cache_value": _cache_value(negative_build, "HTSIM_ENABLE_SIMLLM_RNIC"),
        "negative_main_sha256": negative_main_sha256,
        "positive_assets_before": positive_assets_before,
        "positive_assets_after": positive_assets_after,
        "positive_assets_preserved": positive_assets_preserved,
        "bypass_comparison": {
            "input_matches": [list(item) for item in comparison.input_matches],
            "behavioral_matches": [list(item) for item in comparison.behavioral_matches],
            "equivalent": comparison.equivalent,
        },
        "negative_tier_a_target_absent": negative_tier_a_target_absent,
        "producer_returncode": producer.returncode,
        "producer_stderr": producer.stderr,
        "checker_returncode": checker.returncode,
        "checker_stderr": checker.stderr,
        "forbidden_outputs_absent": forbidden_outputs_absent,
    }
    _publish_json(negative_evidence_path, evidence)
    _require_source_unchanged(htsim_source)

    positive_results = positive_root / "results.json"
    subprocess.run(
        _checker_command(
            positive_observations,
            negative_evidence_path,
            positive_results,
        ),
        check=True,
    )
    if not positive_results.is_file():
        raise AssertionError("positive checker did not publish results")
    _require_source_unchanged(htsim_source)
    _require_simllm_unchanged(simllm_head)
    print(f"wrote authority results to {positive_results}")
    print(f"wrote link-disabled evidence to {negative_evidence_path}")


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--htsim-source", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    out = arguments.out.resolve(strict=False) if arguments.out.is_absolute() else arguments.out
    htsim_source = (
        arguments.htsim_source.resolve(strict=True)
        if arguments.htsim_source.is_absolute()
        else arguments.htsim_source
    )
    expectations = _validate_registry(out, htsim_source)
    if arguments.check_only:
        print("RNIC authority study registry check passed; no artifacts produced")
        return 0
    _run(expectations, out, htsim_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
