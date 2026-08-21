"""In-memory closure manifests for content-addressed calibration objects.

The Wave 0 contract deliberately does not define one generic manifest wire
record.  These utilities therefore validate typed-record closures without
inventing a public JSON schema.  Later evidence, fit and release schemas can
project their exact references into this narrow representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .canonical import CanonicalError, validate_sha256
from .record_types import RecordIntent, RecordObject, validate_schema_id
from .store import ObjectStore, ObjectStoreError


class ManifestError(ValueError):
    """A manifest is incomplete, cyclic, inconsistent or not closed."""


_INTENT_RANK = {
    RecordIntent.EVIDENCE: 0,
    RecordIntent.FIT: 1,
    RecordIntent.RELEASE: 2,
}


@dataclass(frozen=True, slots=True)
class ObjectReference:
    """External type and dependency metadata for one record object."""

    record_id: str
    schema: str
    intent: RecordIntent
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            validate_sha256(self.record_id, "object.record_id")
            validate_schema_id(self.schema, "object.schema")
        except CanonicalError as error:
            raise ManifestError(str(error)) from error
        if type(self.intent) is not RecordIntent:
            raise ManifestError("object.intent: expected a RecordIntent")
        if not isinstance(self.references, tuple):
            raise ManifestError("object.references: expected a tuple")
        validated: list[str] = []
        for index, reference in enumerate(self.references):
            try:
                validated.append(
                    validate_sha256(reference, f"object.references[{index}]")
                )
            except CanonicalError as error:
                raise ManifestError(str(error)) from error
        if len(set(validated)) != len(validated):
            raise ManifestError("object.references: duplicate object identity")
        if tuple(sorted(validated)) != self.references:
            raise ManifestError("object.references: expected canonical digest order")


@dataclass(frozen=True, slots=True)
class ObjectManifest:
    """One closed set of object metadata rooted at an evidence, fit or release."""

    intent: RecordIntent
    roots: tuple[str, ...]
    objects: tuple[ObjectReference, ...]

    def __post_init__(self) -> None:
        if type(self.intent) is not RecordIntent:
            raise ManifestError("manifest.intent: expected a RecordIntent")
        if not isinstance(self.roots, tuple) or not self.roots:
            raise ManifestError("manifest.roots: expected a nonempty tuple")
        if not isinstance(self.objects, tuple) or not self.objects:
            raise ManifestError("manifest.objects: expected a nonempty tuple")
        roots: list[str] = []
        for index, root in enumerate(self.roots):
            try:
                roots.append(validate_sha256(root, f"manifest.roots[{index}]"))
            except CanonicalError as error:
                raise ManifestError(str(error)) from error
        if len(set(roots)) != len(roots):
            raise ManifestError("manifest.roots: duplicate object identity")
        if tuple(sorted(roots)) != self.roots:
            raise ManifestError("manifest.roots: expected canonical digest order")
        if any(not isinstance(item, ObjectReference) for item in self.objects):
            raise ManifestError("manifest.objects: expected ObjectReference members")
        identities = tuple(item.record_id for item in self.objects)
        if len(set(identities)) != len(identities):
            raise ManifestError("manifest.objects: duplicate object identity")
        if tuple(sorted(identities)) != identities:
            raise ManifestError("manifest.objects: expected canonical digest order")

    @classmethod
    def create(
        cls,
        *,
        intent: RecordIntent,
        roots: tuple[str, ...],
        objects: tuple[ObjectReference, ...],
    ) -> ObjectManifest:
        """Sort caller-supplied sets before constructing a strict manifest."""

        return cls(
            intent=intent,
            roots=tuple(sorted(roots)),
            objects=tuple(sorted(objects, key=lambda item: item.record_id)),
        )


@dataclass(frozen=True, slots=True)
class ManifestClosure:
    """A validated immutable closure in dependency-first order."""

    manifest: ObjectManifest
    records: Mapping[str, RecordObject]
    dependency_order: tuple[str, ...]

    def record(self, record_id: str) -> RecordObject:
        """Return one member after validating its external identity syntax."""

        digest = validate_sha256(record_id, "record_id")
        try:
            return self.records[digest]
        except KeyError as error:
            raise ManifestError(f"object {digest} is outside the validated closure") from error


def _metadata_by_id(manifest: ObjectManifest) -> dict[str, ObjectReference]:
    return {item.record_id: item for item in manifest.objects}


def _validate_missing_and_intents(
    manifest: ObjectManifest,
    metadata: dict[str, ObjectReference],
) -> None:
    missing_roots = sorted(set(manifest.roots) - metadata.keys())
    if missing_roots:
        raise ManifestError(f"manifest roots are missing objects: {missing_roots}")
    for root in manifest.roots:
        root_intent = metadata[root].intent
        if root_intent is not manifest.intent:
            raise ManifestError(
                f"manifest root {root} has intent {root_intent.value!r}; "
                f"expected {manifest.intent.value!r}"
            )
    for item in manifest.objects:
        for child_id in item.references:
            child = metadata.get(child_id)
            if child is None:
                raise ManifestError(
                    f"object {item.record_id} references missing object {child_id}"
                )
            if _INTENT_RANK[child.intent] > _INTENT_RANK[item.intent]:
                raise ManifestError(
                    f"object {item.record_id} with intent {item.intent.value!r} "
                    f"cannot reference later intent {child.intent.value!r}"
                )


def _dependency_order(
    manifest: ObjectManifest,
    metadata: dict[str, ObjectReference],
) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    ordered: list[str] = []

    def visit(record_id: str, trail: tuple[str, ...]) -> None:
        if record_id in visiting:
            start = trail.index(record_id)
            cycle = trail[start:] + (record_id,)
            raise ManifestError(f"manifest object cycle: {' -> '.join(cycle)}")
        if record_id in visited:
            return
        visiting.add(record_id)
        for child_id in metadata[record_id].references:
            visit(child_id, trail + (record_id,))
        visiting.remove(record_id)
        visited.add(record_id)
        ordered.append(record_id)

    for root in manifest.roots:
        visit(root, ())
    extras = sorted(metadata.keys() - visited)
    if extras:
        raise ManifestError(f"manifest contains unreachable extra objects: {extras}")
    return tuple(ordered)


def validate_manifest_closure(store: ObjectStore, manifest: ObjectManifest) -> ManifestClosure:
    """Validate structure, reachability and exact stored bytes for a manifest."""

    if not isinstance(store, ObjectStore):
        raise TypeError("store must be an ObjectStore")
    if not isinstance(manifest, ObjectManifest):
        raise TypeError("manifest must be an ObjectManifest")
    metadata = _metadata_by_id(manifest)
    _validate_missing_and_intents(manifest, metadata)
    dependency_order = _dependency_order(manifest, metadata)
    records: dict[str, RecordObject] = {}
    for item in manifest.objects:
        try:
            records[item.record_id] = store.read(
                item.record_id,
                expected_schema=item.schema,
            )
        except ObjectStoreError as error:
            raise ManifestError(f"manifest object {item.record_id} failed validation: {error}") from error
    return ManifestClosure(
        manifest=manifest,
        records=MappingProxyType(records),
        dependency_order=dependency_order,
    )


__all__ = [
    "ManifestClosure",
    "ManifestError",
    "ObjectManifest",
    "ObjectReference",
    "validate_manifest_closure",
]
