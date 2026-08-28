# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0
"""Offline import and exact resolution of the pinned H200 NCCL table."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import lzma
import os
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[2]))

from simllm.calibration.external_db import (
    EXPECTED_AICONFIGURATOR_VERSION,
    EXPECTED_APACHE_LICENSE_HASH,
    EXPECTED_CORE_VERSION,
    EXTERNAL_EVIDENCE_CLASS,
    EXTERNAL_VENV_ENV,
    ExternalDatabaseError,
    ExternalDatabaseGapError,
    ExternalDatabaseIdentityError,
    ExternalLatency,
    ExternalSourceIdentity,
)

EXTERNAL_NCCL_SCHEMA = "simllm-external-nccl-database-v1"
EXTERNAL_NCCL_ROW_SCHEMA = "simllm-external-nccl-row-v1"
EXTERNAL_NCCL_CONVERTER_SCHEMA = "simllm-external-nccl-converter-v1"

EXPECTED_NCCL_SYSTEM = "h200_sxm"
EXPECTED_NCCL_COLLECTION_VERSION = "2.26.2"
EXPECTED_NCCL_ROW_VERSION = "2.29.2"
EXPECTED_NCCL_SOURCE_HASH = "e432db694195110aa39c1e1eccf1accda012e69ef68e95210d049809bb93f015"
EXPECTED_NCCL_PARQUET_HASH = "85bc8eeed2e20da0c74d035a9f1172ef9196fc729a49956c78ac19c659d101c2"
EXPECTED_NCCL_METADATA_HASH = "6b40bd84085192ec4f7cb2780635dfe2eb00857e30987dd6c1a90c3b4a63cd8a"
EXPECTED_NCCL_ROW_COUNT = 1_008

NCCL_INTRA_NODE_BANDWIDTH_BYTES_PER_SECOND = 450_000_000_000
NCCL_INTER_NODE_BANDWIDTH_BYTES_PER_SECOND = 50_000_000_000

NCCL_ARTIFACT_DIRECTORY_NAME = EXPECTED_NCCL_SOURCE_HASH
NCCL_ARTIFACT_RELATIVE_ROOT = Path("offline/calibration/external-databases")

NVIDIA_NCCL_COPYRIGHT = (
    "SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _package_root(package: str) -> Path:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ExternalDatabaseIdentityError(f"installed package {package!r} is unavailable")
    return Path(spec.origin).resolve().parent


def _installed_python(venv_root: Path) -> Path:
    for candidate in (venv_root / "bin/python", venv_root / "Scripts/python.exe"):
        if candidate.is_file():
            return candidate
    raise ExternalDatabaseIdentityError(
        f"{EXTERNAL_VENV_ENV} does not name a Python virtual environment with a usable interpreter"
    )


def _verify_installed_identity() -> tuple[Path, dict[str, str]]:
    versions = {
        "aiconfigurator": importlib.metadata.version("aiconfigurator"),
        "aiconfigurator-core": importlib.metadata.version("aiconfigurator-core"),
        "numpy": importlib.metadata.version("numpy"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }
    expected = {
        "aiconfigurator": EXPECTED_AICONFIGURATOR_VERSION,
        "aiconfigurator-core": EXPECTED_CORE_VERSION,
    }
    mismatches = [
        f"{name}={versions[name]} (expected {version})"
        for name, version in expected.items()
        if versions[name] != version
    ]
    if mismatches:
        raise ExternalDatabaseIdentityError(
            "installed package identity mismatch: " + ", ".join(mismatches)
        )
    return _package_root("aiconfigurator_core"), versions


def _source_manifest(collection_root: Path) -> bytes:
    paths = ("collection_meta.yaml", "nccl_perf.parquet")
    lines = []
    for relative in paths:
        path = collection_root / relative
        if not path.is_file():
            raise ExternalDatabaseIdentityError(f"missing frozen NCCL source file {relative}")
        lines.append(f"{_sha256_file(path)}  {relative}\n")
    return "".join(lines).encode("utf-8")


def _conversion_recipe(*, pyarrow_version: str) -> dict[str, Any]:
    return {
        "converter_schema": EXTERNAL_NCCL_CONVERTER_SCHEMA,
        "pyarrow": pyarrow_version,
        "row_ordering": "rows retain PyArrow to_pylist source order",
        "duplicate_resolution": "the first source row at each lookup coordinate wins",
        "float_encoding": "Python float.hex() encodes source IEEE-754 binary64 values",
        "json_lines": {
            "record": "one seven-element JSON array per source row",
            "separators": [",", ":"],
            "ensure_ascii": True,
            "encoding": "ASCII",
            "line_termination": "LF (0x0a) after every record",
        },
        "xz": {
            "format": "FORMAT_XZ",
            "check": "CHECK_CRC64",
            "preset": 9,
            "extreme": True,
            "preset_expression": "9 | PRESET_EXTREME",
        },
    }


def _artifact_notices(payload_name: str) -> tuple[str, str]:
    notice = f"""Third-party converted NCCL performance table

{NVIDIA_NCCL_COPYRIGHT}
SPDX-License-Identifier: Apache-2.0

Source packages: aiconfigurator 0.11.0 and aiconfigurator-core 0.11.0.
The copyright and license lines come from the NCCL collection metadata.
"""
    modified = f"""Modified-file statement

SimLLM converted the source NCCL Parquet table and collection metadata into
deterministic compressed JSON Lines plus a portable identity manifest. Every
source latency is retained as Python float.hex. The converted files are not
original NVIDIA distribution files.

Artifact derivations:

- LICENSE: byte-identical Apache License 2.0 text from the installed package.
- MODIFIED: this file-by-file conversion and modification statement.
- THIRD_PARTY_NOTICE: copyright and license lines retained from collection_meta.yaml.
- manifest.json: source identity, conversion recipe, row inventory and hashes.
- {payload_name}: row-preserving conversion of nccl_perf.parquet.
- source-files.sha256: sorted SHA-256 manifest of the two source files.
"""
    return notice, modified


def default_external_nccl_artifact_dir() -> Path:
    """Return the tracked auxiliary NCCL artifact directory."""

    return _repo_root() / NCCL_ARTIFACT_RELATIVE_ROOT / NCCL_ARTIFACT_DIRECTORY_NAME


def import_external_nccl_database(
    *,
    venv_root: str | os.PathLike[str] | None = None,
    output_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Import the pinned H200 NCCL table without adding a runtime PyArrow dependency."""

    raw_venv = venv_root if venv_root is not None else os.environ.get(EXTERNAL_VENV_ENV)
    if raw_venv is None:
        raise ExternalDatabaseIdentityError(
            f"set {EXTERNAL_VENV_ENV} to the pinned aiconfigurator virtual environment"
        )
    python = _installed_python(Path(raw_venv).expanduser())
    destination = (
        Path(output_root)
        if output_root is not None
        else _repo_root() / NCCL_ARTIFACT_RELATIVE_ROOT
    )
    completed = subprocess.run(
        [
            os.fspath(python),
            os.fspath(Path(__file__).resolve()),
            "--worker-import",
            "--output-root",
            os.fspath(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "NCCL converter failed without diagnostics"
        )
        raise ExternalDatabaseError(f"external NCCL import failed: {detail}")
    artifact = destination / NCCL_ARTIFACT_DIRECTORY_NAME
    if not artifact.is_dir():
        raise ExternalDatabaseError(
            "NCCL converter reported success without creating the expected artifact"
        )
    return artifact


def _write_worker_artifact(output_root: Path) -> Path:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ExternalDatabaseIdentityError(
            "the NCCL import worker must run inside the pinned environment with PyArrow"
        ) from error

    package_root, versions = _verify_installed_identity()
    collection_root = (
        package_root
        / "systems/data"
        / EXPECTED_NCCL_SYSTEM
        / "comm/nccl"
        / EXPECTED_NCCL_COLLECTION_VERSION
    )
    source_manifest = _source_manifest(collection_root)
    source_hash = _sha256_bytes(source_manifest)
    if source_hash != EXPECTED_NCCL_SOURCE_HASH:
        raise ExternalDatabaseIdentityError(
            f"NCCL source hash mismatch: expected {EXPECTED_NCCL_SOURCE_HASH}, "
            f"found {source_hash}"
        )
    parquet_path = collection_root / "nccl_perf.parquet"
    metadata_path = collection_root / "collection_meta.yaml"
    if _sha256_file(parquet_path) != EXPECTED_NCCL_PARQUET_HASH:
        raise ExternalDatabaseIdentityError("NCCL Parquet identity mismatch")
    if _sha256_file(metadata_path) != EXPECTED_NCCL_METADATA_HASH:
        raise ExternalDatabaseIdentityError("NCCL collection metadata identity mismatch")

    arrow = pq.read_table(parquet_path)
    if arrow.num_rows != EXPECTED_NCCL_ROW_COUNT:
        raise ExternalDatabaseIdentityError(
            f"NCCL row count mismatch: expected {EXPECTED_NCCL_ROW_COUNT}, "
            f"found {arrow.num_rows}"
        )
    records = []
    row_versions: set[str] = set()
    for row in arrow.to_pylist():
        row_version = str(row["version"])
        row_versions.add(row_version)
        record = [
            EXTERNAL_NCCL_ROW_SCHEMA,
            row_version,
            str(row["nccl_dtype"]),
            str(row["op_name"]),
            int(row["num_gpus"]),
            int(row["message_size"]),
            float(row["latency"]).hex(),
        ]
        records.append(
            json.dumps(record, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            + b"\n"
        )
    if row_versions != {EXPECTED_NCCL_ROW_VERSION}:
        raise ExternalDatabaseIdentityError(
            f"NCCL source rows contain versions {sorted(row_versions)!r}"
        )

    payload_raw = b"".join(records)
    payload = lzma.compress(
        payload_raw,
        format=lzma.FORMAT_XZ,
        check=lzma.CHECK_CRC64,
        preset=9 | lzma.PRESET_EXTREME,
    )
    payload_hash = _sha256_bytes(payload)
    payload_name = f"rows-{payload_hash}.jsonl.xz"
    manifest = {
        "schema": EXTERNAL_NCCL_SCHEMA,
        "source": {
            "tool": "NVIDIA AIConfigurator",
            "aiconfigurator_version": versions["aiconfigurator"],
            "aiconfigurator_core_version": versions["aiconfigurator-core"],
            "system": EXPECTED_NCCL_SYSTEM,
            "backend": "nccl",
            "database_version": EXPECTED_NCCL_COLLECTION_VERSION,
            "database_mode": "SILICON",
            "shared_layer": False,
            "estimator_surface": "python",
            "data_slice_sha256": source_hash,
            "source_path": (
                "systems/data/h200_sxm/comm/nccl/2.26.2/nccl_perf.parquet"
            ),
            "parquet_sha256": EXPECTED_NCCL_PARQUET_HASH,
            "collection_metadata_sha256": EXPECTED_NCCL_METADATA_HASH,
            "row_versions": sorted(row_versions),
        },
        "conversion": {
            "format": "XZ-compressed JSON Lines with binary64 hex values",
            "modified": True,
            "row_schema": EXTERNAL_NCCL_ROW_SCHEMA,
            "rows": len(records),
            "payload": payload_name,
            "payload_sha256": payload_hash,
            "payload_bytes": len(payload),
            "uncompressed_bytes": len(payload_raw),
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "numpy": versions["numpy"],
            "pyarrow": versions["pyarrow"],
            "recipe": _conversion_recipe(pyarrow_version=versions["pyarrow"]),
        },
        "table": {
            "source_path": "nccl_perf.parquet",
            "source_sha256": EXPECTED_NCCL_PARQUET_HASH,
            "rows": len(records),
            "measured_ranks": [2, 4, 8],
            "operations": ["all_gather", "all_reduce", "alltoall", "reduce_scatter"],
            "dtypes": ["half", "int8"],
        },
        "source_hash_manifest": "source-files.sha256",
        "resolver_sources": [
            {
                "path": "aiconfigurator_core/sdk/operations/communication.py",
                "line": 470,
                "semantic": "effective rank selection and NCCL table lookup",
            },
            {
                "path": "aiconfigurator_core/sdk/operations/communication.py",
                "line": 480,
                "semantic": "raw linear message-size interpolation",
            },
            {
                "path": "aiconfigurator_core/sdk/operations/communication.py",
                "line": 491,
                "semantic": "rank extrapolation above the measured maximum",
            },
            {
                "path": "aiconfigurator_core/sdk/operations/communication.py",
                "line": 720,
                "semantic": "first source row wins when a coordinate repeats",
            },
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    artifact = output_root / NCCL_ARTIFACT_DIRECTORY_NAME
    if artifact.exists():
        raise ExternalDatabaseError(f"refusing to overwrite existing artifact {artifact}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{NCCL_ARTIFACT_DIRECTORY_NAME}.", dir=output_root)
    )
    try:
        (temporary / payload_name).write_bytes(payload)
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        (temporary / "source-files.sha256").write_bytes(source_manifest)
        notice, modified = _artifact_notices(payload_name)
        (temporary / "THIRD_PARTY_NOTICE").write_text(
            notice, encoding="utf-8", newline="\n"
        )
        (temporary / "MODIFIED").write_text(modified, encoding="utf-8", newline="\n")
        license_candidates = sorted(
            package_root.parent.glob("aiconfigurator_core-*.dist-info/licenses/LICENSE")
        )
        if not license_candidates:
            license_candidates = sorted(
                package_root.parent.glob("aiconfigurator-*.dist-info/licenses/LICENSE")
            )
        if not license_candidates:
            raise ExternalDatabaseIdentityError(
                "installed Apache 2.0 license text is unavailable"
            )
        if _sha256_file(license_candidates[0]) != EXPECTED_APACHE_LICENSE_HASH:
            raise ExternalDatabaseIdentityError(
                "installed Apache 2.0 license text differs from the frozen upstream bytes"
            )
        shutil.copyfile(license_candidates[0], temporary / "LICENSE")
        os.replace(temporary, artifact)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return artifact


def external_nccl_artifact_licensing_findings(
    artifact_dir: str | os.PathLike[str],
) -> tuple[str, ...]:
    """Return exact licensing findings for the auxiliary NCCL artifact."""

    artifact = Path(artifact_dir)
    findings = []
    license_path = artifact / "LICENSE"
    if (
        not license_path.is_file()
        or _sha256_file(license_path) != EXPECTED_APACHE_LICENSE_HASH
    ):
        findings.append("LICENSE is not byte-identical to the frozen upstream Apache text")
    try:
        notice = (artifact / "THIRD_PARTY_NOTICE").read_text(encoding="utf-8")
    except FileNotFoundError:
        notice = ""
    for line in (NVIDIA_NCCL_COPYRIGHT, "SPDX-License-Identifier: Apache-2.0"):
        if line not in notice.splitlines():
            findings.append(f"THIRD_PARTY_NOTICE is missing exact line {line!r}")
    try:
        modified = (artifact / "MODIFIED").read_text(encoding="utf-8")
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        payload = str(manifest["conversion"]["payload"])
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        modified = ""
        payload = "missing-payload"
    for filename in (
        "LICENSE",
        "MODIFIED",
        "THIRD_PARTY_NOTICE",
        "manifest.json",
        payload,
        "source-files.sha256",
    ):
        if f"- {filename}:" not in modified:
            findings.append(f"MODIFIED does not enumerate {filename}")
    return tuple(findings)


@dataclass(frozen=True)
class _NcclRow:
    row_version: str
    dtype: str
    operation: str
    ranks: int
    message_size: int
    latency_ms: float


class ExternalNcclDatabase:
    """Read-only resolver for the imported H200 NCCL 2.26.2 collection."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        manifest: Mapping[str, Any],
        rows: Sequence[_NcclRow],
    ) -> None:
        self.artifact_dir = artifact_dir
        self.manifest = dict(manifest)
        self.source = ExternalSourceIdentity.from_manifest(manifest)
        self._rows = tuple(rows)
        self._curves: dict[str, dict[str, dict[int, dict[int, float]]]] = {}
        for row in self._rows:
            curve = (
                self._curves.setdefault(row.dtype, {})
                .setdefault(row.operation, {})
                .setdefault(row.ranks, {})
            )
            if row.message_size not in curve:
                curve[row.message_size] = row.latency_ms

    @classmethod
    def load(
        cls, artifact_dir: str | os.PathLike[str] | None = None
    ) -> ExternalNcclDatabase:
        artifact = (
            Path(artifact_dir)
            if artifact_dir is not None
            else default_external_nccl_artifact_dir()
        )
        try:
            manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ExternalDatabaseIdentityError(
                "external NCCL manifest is missing or invalid"
            ) from error
        if manifest.get("schema") != EXTERNAL_NCCL_SCHEMA:
            raise ExternalDatabaseIdentityError(
                f"unsupported external NCCL schema {manifest.get('schema')!r}"
            )
        source = ExternalSourceIdentity.from_manifest(manifest)
        expected_source = ExternalSourceIdentity(
            tool="NVIDIA AIConfigurator",
            aiconfigurator_version=EXPECTED_AICONFIGURATOR_VERSION,
            core_version=EXPECTED_CORE_VERSION,
            system=EXPECTED_NCCL_SYSTEM,
            backend="nccl",
            database_version=EXPECTED_NCCL_COLLECTION_VERSION,
            slice_hash=EXPECTED_NCCL_SOURCE_HASH,
        )
        if source != expected_source:
            raise ExternalDatabaseIdentityError(
                f"NCCL artifact source identity mismatch: {source.as_dict()}"
            )
        source_manifest = (artifact / "source-files.sha256").read_bytes()
        if _sha256_bytes(source_manifest) != EXPECTED_NCCL_SOURCE_HASH:
            raise ExternalDatabaseIdentityError("stored NCCL source manifest hash mismatch")
        conversion = manifest.get("conversion")
        if not isinstance(conversion, Mapping):
            raise ExternalDatabaseIdentityError("external NCCL conversion must be an object")
        recipe = conversion.get("recipe")
        pyarrow_version = conversion.get("pyarrow")
        if not isinstance(pyarrow_version, str) or recipe != _conversion_recipe(
            pyarrow_version=pyarrow_version
        ):
            raise ExternalDatabaseIdentityError("external NCCL conversion recipe mismatch")
        payload_hash = conversion.get("payload_sha256")
        payload_name = conversion.get("payload")
        if not isinstance(payload_hash, str) or payload_name != (
            f"rows-{payload_hash}.jsonl.xz"
        ):
            raise ExternalDatabaseIdentityError(
                "NCCL payload filename is not content-addressed by its declared hash"
            )
        payload_path = artifact / payload_name
        if _sha256_file(payload_path) != payload_hash:
            raise ExternalDatabaseIdentityError("external NCCL payload hash mismatch")

        rows = []
        with lzma.open(payload_path, "rt", encoding="ascii", newline="\n") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                    schema, version, dtype, operation, ranks, message_size, latency_hex = record
                    latency = float.fromhex(latency_hex)
                except (TypeError, ValueError) as error:
                    raise ExternalDatabaseIdentityError(
                        f"invalid external NCCL row at payload line {line_number}"
                    ) from error
                if schema != EXTERNAL_NCCL_ROW_SCHEMA:
                    raise ExternalDatabaseIdentityError(
                        f"invalid NCCL row schema at payload line {line_number}"
                    )
                if version != EXPECTED_NCCL_ROW_VERSION:
                    raise ExternalDatabaseIdentityError(
                        f"unexpected NCCL row version at payload line {line_number}: {version!r}"
                    )
                rows.append(
                    _NcclRow(
                        row_version=str(version),
                        dtype=str(dtype),
                        operation=str(operation),
                        ranks=int(ranks),
                        message_size=int(message_size),
                        latency_ms=latency,
                    )
                )
        if conversion.get("rows") != EXPECTED_NCCL_ROW_COUNT or len(rows) != (
            EXPECTED_NCCL_ROW_COUNT
        ):
            raise ExternalDatabaseIdentityError("external NCCL row count mismatch")
        return cls(artifact_dir=artifact, manifest=manifest, rows=rows)

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def payload_sha256(self) -> str:
        return str(self.manifest["conversion"]["payload_sha256"])

    @staticmethod
    def _bandwidth(ranks: int) -> int:
        return (
            NCCL_INTRA_NODE_BANDWIDTH_BYTES_PER_SECOND
            if ranks <= 8
            else NCCL_INTER_NODE_BANDWIDTH_BYTES_PER_SECOND
        )

    @staticmethod
    def _linear(curve: Mapping[int, float], message_size: int) -> float:
        points = sorted(curve)
        index = bisect_left(points, message_size)
        if index < len(points) and points[index] == message_size:
            return float(curve[points[index]])
        if index == 0 or index == len(points):
            raise ExternalDatabaseGapError(
                f"NCCL message size {message_size} is outside the measured interpolation range"
            )
        low = points[index - 1]
        high = points[index]
        weight = (message_size - low) / (high - low)
        return float(curve[low]) + (float(curve[high]) - float(curve[low])) * weight

    def query(
        self,
        *,
        dtype: str,
        operation: str,
        ranks: int,
        message_size: int,
    ) -> ExternalLatency:
        """Resolve one NCCL value with the source rank extrapolation."""

        if type(ranks) is not int or ranks < 2:
            raise ValueError("ranks must be an integer of at least two")
        if type(message_size) is not int or message_size <= 0:
            raise ValueError("message_size must be a positive integer")
        try:
            by_rank = self._curves[dtype][operation]
        except KeyError as error:
            raise ExternalDatabaseGapError(
                f"external NCCL table gap for dtype={dtype!r}, operation={operation!r}"
            ) from error
        effective_ranks = min(ranks, max(by_rank))
        latency = self._linear(by_rank[effective_ranks], message_size)
        if ranks > effective_ranks:
            latency *= (
                (ranks - 1)
                / ranks
                * effective_ranks
                / (effective_ranks - 1)
                * self._bandwidth(effective_ranks)
                / self._bandwidth(ranks)
            )
        return ExternalLatency(
            latency_ms=latency,
            source=self.source,
            operation=f"nccl_{operation}",
            rule=(
                "raw-linear-message-size;effective-ranks=min(requested,8);"
                "h200-rank-bandwidth-extrapolation"
            ),
            evidence_class=EXTERNAL_EVIDENCE_CLASS,
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-external", action="store_true")
    parser.add_argument("--worker-import", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    if args.worker_import:
        if args.output_root is None:
            raise SystemExit("--worker-import requires --output-root")
        print(_write_worker_artifact(args.output_root))
        return 0
    if args.import_external:
        print(
            import_external_nccl_database(
                venv_root=args.venv,
                output_root=args.output_root,
            )
        )
        return 0
    raise SystemExit("select --import-external")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_NCCL_COLLECTION_VERSION",
    "EXPECTED_NCCL_SOURCE_HASH",
    "ExternalNcclDatabase",
    "default_external_nccl_artifact_dir",
    "external_nccl_artifact_licensing_findings",
    "import_external_nccl_database",
]
