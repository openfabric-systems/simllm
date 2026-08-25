from __future__ import annotations

import os
from pathlib import Path

import pytest

from simllm.calibration.canonical import CanonicalError, canonical_bytes, sha256_bytes
from simllm.calibration.manifests import (
    ManifestError,
    ObjectManifest,
    ObjectReference,
    validate_manifest_closure,
)
from simllm.calibration.record_types import RecordIntent, RecordObject
from simllm.calibration.store import ObjectStore, ObjectStoreError


def _record(label: str, *, schema: str = "test-object-v1") -> RecordObject:
    return RecordObject.from_value({"schema": schema, "label": label})


def _reference(
    record: RecordObject,
    intent: RecordIntent,
    *references: RecordObject,
    schema: str | None = None,
) -> ObjectReference:
    return ObjectReference(
        record_id=record.record_id,
        schema=record.schema if schema is None else schema,
        intent=intent,
        references=tuple(sorted(item.record_id for item in references)),
    )


def test_store_writes_digest_named_canonical_objects(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = store.write({"label": "alpha", "schema": "test-object-v1"})
    expected_path = store.root / f"{record.record_id}.json"
    assert store.path_for(record.record_id) == expected_path
    assert expected_path.read_bytes() == record.canonical
    assert store.contains(record.record_id)
    assert store.read(record.record_id) == record


def test_store_reuses_an_identical_existing_object(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    first = store.write(_record("same"))
    before = store.path_for(first.record_id).stat()
    second = store.write(_record("same"))
    after = store.path_for(first.record_id).stat()
    assert second == first
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns


def test_store_accepts_only_canonical_record_bytes(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    canonical = b'{"label":"alpha","schema":"test-object-v1"}'
    assert store.write_bytes(canonical).canonical == canonical
    with pytest.raises(CanonicalError, match="not canonical"):
        store.write_bytes(b'{ "schema": "test-object-v1", "label": "alpha" }')


@pytest.mark.parametrize(
    "record_id",
    [
        "../" + "0" * 61,
        "/" + "0" * 63,
        "A" * 64,
        "0" * 63,
        "0" * 65,
    ],
)
def test_store_rejects_unsafe_or_noncanonical_object_ids(
    tmp_path: Path,
    record_id: str,
) -> None:
    store = ObjectStore(tmp_path / "objects")
    with pytest.raises(CanonicalError, match="64 lowercase hexadecimal"):
        store.path_for(record_id)
    with pytest.raises(CanonicalError, match="64 lowercase hexadecimal"):
        store.read(record_id)


def test_store_rejects_hash_mismatch_at_a_valid_digest_path(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    expected = _record("expected")
    wrong = _record("wrong")
    path = store.path_for(expected.record_id)
    path.parent.mkdir(parents=True)
    path.write_bytes(wrong.canonical)
    with pytest.raises(ObjectStoreError, match="hash mismatch"):
        store.read(expected.record_id)


def test_store_rejects_noncanonical_bytes_before_hash_acceptance(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = _record("alpha")
    noncanonical = b'{ "schema":"test-object-v1","label":"alpha"}'
    path = store.path_for(record.record_id)
    path.parent.mkdir(parents=True)
    path.write_bytes(noncanonical)
    with pytest.raises(ObjectStoreError, match="not canonical"):
        store.read(record.record_id)


def test_store_checks_the_declared_schema_on_read(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = store.write(_record("alpha"))
    assert store.read(record.record_id, expected_schema="test-object-v1") == record
    with pytest.raises(ObjectStoreError, match="expected 'other-v1'"):
        store.read(record.record_id, expected_schema="other-v1")


def test_store_enforces_the_object_size_limit_on_write_and_read(tmp_path: Path) -> None:
    source = _record("a-long-enough-label")
    small = ObjectStore(tmp_path / "small", max_object_bytes=len(source.canonical) - 1)
    with pytest.raises(ObjectStoreError, match="limit"):
        small.write(source)

    large = ObjectStore(tmp_path / "large")
    large.write(source)
    strict_reader = ObjectStore(large.root, max_object_bytes=len(source.canonical) - 1)
    with pytest.raises(ObjectStoreError, match="limit"):
        strict_reader.read(source.record_id)


def test_store_rejects_a_file_where_a_root_or_object_is_required(tmp_path: Path) -> None:
    root_file = tmp_path / "root-file"
    root_file.write_text("not a directory")
    with pytest.raises(ObjectStoreError, match="not a directory"):
        ObjectStore(root_file)

    store = ObjectStore(tmp_path / "objects")
    record = _record("alpha")
    path = store.path_for(record.record_id)
    path.parent.mkdir(parents=True)
    path.mkdir()
    with pytest.raises(ObjectStoreError, match="not a regular file"):
        store.read(record.record_id)


def _symlink_or_skip(target: Path, link: Path, *, target_is_directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")


def test_store_rejects_symlink_root_and_parent_components(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root_link = tmp_path / "root-link"
    _symlink_or_skip(target, root_link, target_is_directory=True)
    with pytest.raises(ObjectStoreError, match="symlink"):
        ObjectStore(root_link)

    parent_link = tmp_path / "parent-link"
    _symlink_or_skip(target, parent_link, target_is_directory=True)
    with pytest.raises(ObjectStoreError, match="symlink"):
        ObjectStore(parent_link / "objects")


def test_store_rejects_symlink_objects(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = _record("alpha")
    store.root.mkdir(parents=True)
    target_file = tmp_path / "target.json"
    target_file.write_bytes(record.canonical)
    object_path = store.path_for(record.record_id)
    _symlink_or_skip(target_file, object_path, target_is_directory=False)
    with pytest.raises(ObjectStoreError, match="symlink"):
        store.read(record.record_id)


def _valid_release_manifest(store: ObjectStore) -> tuple[ObjectManifest, tuple[RecordObject, ...]]:
    evidence = store.write(_record("evidence", schema="evidence-v1"))
    fit = store.write(_record("fit", schema="fit-v1"))
    release = store.write(_record("release", schema="release-v1"))
    manifest = ObjectManifest.create(
        intent=RecordIntent.RELEASE,
        roots=(release.record_id,),
        objects=(
            _reference(evidence, RecordIntent.EVIDENCE),
            _reference(fit, RecordIntent.FIT, evidence),
            _reference(release, RecordIntent.RELEASE, evidence, fit),
        ),
    )
    return manifest, (evidence, fit, release)


def test_manifest_validates_a_complete_dependency_first_release_closure(
    tmp_path: Path,
) -> None:
    store = ObjectStore(tmp_path / "objects")
    manifest, (evidence, fit, release) = _valid_release_manifest(store)
    closure = validate_manifest_closure(store, manifest)
    assert set(closure.records) == {evidence.record_id, fit.record_id, release.record_id}
    assert closure.dependency_order[-1] == release.record_id
    assert closure.dependency_order.index(evidence.record_id) < closure.dependency_order.index(
        fit.record_id
    )
    assert closure.record(fit.record_id) == fit


def test_manifest_rejects_a_missing_root_metadata_entry(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = store.write(_record("evidence"))
    missing = "f" * 64
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(missing,),
        objects=(_reference(record, RecordIntent.EVIDENCE),),
    )
    with pytest.raises(ManifestError, match="roots are missing"):
        validate_manifest_closure(store, manifest)


def test_manifest_rejects_a_missing_referenced_metadata_entry(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = store.write(_record("evidence"))
    missing = "f" * 64
    reference = ObjectReference(
        record_id=record.record_id,
        schema=record.schema,
        intent=RecordIntent.EVIDENCE,
        references=(missing,),
    )
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(record.record_id,),
        objects=(reference,),
    )
    with pytest.raises(ManifestError, match="references missing object"):
        validate_manifest_closure(store, manifest)


def test_manifest_rejects_an_object_missing_from_the_store(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = _record("evidence")
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(record.record_id,),
        objects=(_reference(record, RecordIntent.EVIDENCE),),
    )
    with pytest.raises(ManifestError, match="missing object"):
        validate_manifest_closure(store, manifest)


def test_manifest_rejects_hash_mismatch_in_reachable_storage(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = _record("evidence")
    path = store.path_for(record.record_id)
    path.parent.mkdir(parents=True)
    path.write_bytes(_record("different").canonical)
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(record.record_id,),
        objects=(_reference(record, RecordIntent.EVIDENCE),),
    )
    with pytest.raises(ManifestError, match="hash mismatch"):
        validate_manifest_closure(store, manifest)


def test_manifest_rejects_cycles(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    first = store.write(_record("first"))
    second = store.write(_record("second"))
    first_ref = _reference(first, RecordIntent.EVIDENCE, second)
    second_ref = _reference(second, RecordIntent.EVIDENCE, first)
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(first.record_id,),
        objects=(first_ref, second_ref),
    )
    with pytest.raises(ManifestError, match="cycle"):
        validate_manifest_closure(store, manifest)


def test_manifest_rejects_unreachable_extra_objects(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    root = store.write(_record("root"))
    extra = store.write(_record("extra"))
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(root.record_id,),
        objects=(
            _reference(root, RecordIntent.EVIDENCE),
            _reference(extra, RecordIntent.EVIDENCE),
        ),
    )
    with pytest.raises(ManifestError, match="unreachable extra"):
        validate_manifest_closure(store, manifest)


def test_manifest_rejects_declared_schema_splicing(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    root = store.write(_record("root", schema="actual-v1"))
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(root.record_id,),
        objects=(_reference(root, RecordIntent.EVIDENCE, schema="claimed-v1"),),
    )
    with pytest.raises(ManifestError, match="expected 'claimed-v1'"):
        validate_manifest_closure(store, manifest)


def test_manifest_rejects_a_reference_to_a_later_intent(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    evidence = store.write(_record("evidence"))
    release = store.write(_record("release"))
    manifest = ObjectManifest.create(
        intent=RecordIntent.EVIDENCE,
        roots=(evidence.record_id,),
        objects=(
            _reference(evidence, RecordIntent.EVIDENCE, release),
            _reference(release, RecordIntent.RELEASE),
        ),
    )
    with pytest.raises(ManifestError, match="cannot reference later intent"):
        validate_manifest_closure(store, manifest)


def test_manifest_root_intent_must_match_manifest_intent(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    root = store.write(_record("fit"))
    manifest = ObjectManifest.create(
        intent=RecordIntent.RELEASE,
        roots=(root.record_id,),
        objects=(_reference(root, RecordIntent.FIT),),
    )
    with pytest.raises(ManifestError, match="expected 'release'"):
        validate_manifest_closure(store, manifest)


def test_strict_manifest_constructor_rejects_noncanonical_order_and_duplicates() -> None:
    first = _record("first")
    second = _record("second")
    low, high = sorted((first, second), key=lambda item: item.record_id)
    low_ref = _reference(low, RecordIntent.EVIDENCE)
    high_ref = _reference(high, RecordIntent.EVIDENCE)

    with pytest.raises(ManifestError, match="canonical digest order"):
        ObjectManifest(
            intent=RecordIntent.EVIDENCE,
            roots=(high.record_id, low.record_id),
            objects=(low_ref, high_ref),
        )
    with pytest.raises(ManifestError, match="duplicate object identity"):
        ObjectManifest(
            intent=RecordIntent.EVIDENCE,
            roots=(low.record_id,),
            objects=(low_ref, low_ref),
        )
    with pytest.raises(ManifestError, match="duplicate object identity"):
        ObjectReference(
            record_id=low.record_id,
            schema=low.schema,
            intent=RecordIntent.EVIDENCE,
            references=(high.record_id, high.record_id),
        )


def test_manifest_rejects_invalid_intent_and_metadata_types() -> None:
    record = _record("root")
    with pytest.raises(ManifestError, match="RecordIntent"):
        ObjectReference(
            record_id=record.record_id,
            schema=record.schema,
            intent="evidence",  # type: ignore[arg-type]
        )
    with pytest.raises(ManifestError, match="expected a tuple"):
        ObjectReference(
            record_id=record.record_id,
            schema=record.schema,
            intent=RecordIntent.EVIDENCE,
            references=[],  # type: ignore[arg-type]
        )


def test_validated_closure_mapping_is_immutable(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    manifest, records = _valid_release_manifest(store)
    closure = validate_manifest_closure(store, manifest)
    with pytest.raises(TypeError):
        closure.records[records[0].record_id] = records[0]  # type: ignore[index]
    with pytest.raises(ManifestError, match="outside"):
        closure.record("0" * 64)


def test_content_digest_is_over_exact_record_bytes_not_the_path(tmp_path: Path) -> None:
    store = ObjectStore(tmp_path / "objects")
    record = store.write(_record("alpha"))
    assert record.record_id == sha256_bytes(record.canonical)
    assert record.record_id == sha256_bytes(canonical_bytes(dict(record.value)))
    assert os.fspath(store.path_for(record.record_id)).endswith(f"{record.record_id}.json")
