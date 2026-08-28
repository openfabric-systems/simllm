# SPDX-FileCopyrightText: Copyright (c) 2026 Yifeng Wang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from simllm.calibration.external_db import (
    ExternalDatabaseIdentityError,
)
from simllm.calibration.external_nccl import (
    EXPECTED_NCCL_COLLECTION_VERSION,
    EXPECTED_NCCL_SOURCE_HASH,
    ExternalNcclDatabase,
    default_external_nccl_artifact_dir,
    external_nccl_artifact_licensing_findings,
    import_external_nccl_database,
)


@pytest.fixture(scope="module")
def database() -> ExternalNcclDatabase:
    return ExternalNcclDatabase.load()


def test_nccl_artifact_identity_inventory_and_license(
    database: ExternalNcclDatabase,
) -> None:
    assert database.artifact_dir == default_external_nccl_artifact_dir()
    assert database.source.slice_hash == EXPECTED_NCCL_SOURCE_HASH
    assert database.source.database_version == EXPECTED_NCCL_COLLECTION_VERSION
    assert database.source.database_mode == "SILICON"
    assert database.source.shared_layer is False
    assert database.row_count == 1008
    assert database.payload_sha256 == (
        "12ed4c1dc12b3d9f0f04ffecf025d0dea5599946fa36944bcb60117035d70efb"
    )
    assert database.manifest["source"]["row_versions"] == ["2.29.2"]
    assert database.manifest["table"]["measured_ranks"] == [2, 4, 8]
    assert external_nccl_artifact_licensing_findings(database.artifact_dir) == ()


@pytest.mark.parametrize(
    ("ranks", "message_size", "ag_hex", "rs_hex"),
    [
        (8, 98304, "0x1.0cb295e9e1b09p-6", "0x1.af8df7a4e7ab8p-7"),
        (32, 393216, "0x1.4705173aec83dp-3", "0x1.29882fa76eca8p-3"),
        (128, 1572864, "0x1.3addfad3cc5dfp-2", "0x1.088c8a23f97e4p-2"),
        (256, 3145728, "0x1.a68ad33c8ff20p-2", "0x1.83223b197426bp-2"),
    ],
)
def test_nccl_queries_reproduce_executed_interpolation_and_rank_extrapolation(
    database: ExternalNcclDatabase,
    ranks: int,
    message_size: int,
    ag_hex: str,
    rs_hex: str,
) -> None:
    all_gather = database.query(
        dtype="half",
        operation="all_gather",
        ranks=ranks,
        message_size=message_size,
    )
    reduce_scatter = database.query(
        dtype="half",
        operation="reduce_scatter",
        ranks=ranks,
        message_size=message_size,
    )
    assert all_gather.hex == ag_hex
    assert reduce_scatter.hex == rs_hex


def test_nccl_import_requires_explicit_pinned_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIMLLM_EXTERNAL_AIC_VENV", raising=False)
    with pytest.raises(ExternalDatabaseIdentityError, match="SIMLLM_EXTERNAL_AIC_VENV"):
        import_external_nccl_database()


def test_nccl_payload_hash_is_checked_before_decompression(tmp_path: Path) -> None:
    artifact = default_external_nccl_artifact_dir()
    tampered = tmp_path / "tampered"
    shutil.copytree(artifact, tampered)
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["conversion"]["payload_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExternalDatabaseIdentityError, match="content-addressed"):
        ExternalNcclDatabase.load(tampered)
