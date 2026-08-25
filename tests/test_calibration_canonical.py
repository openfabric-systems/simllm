from __future__ import annotations

import hashlib
import math
import sys
import unicodedata

import pytest

from simllm.calibration.canonical import (
    CANONICAL_PYTHON,
    CANONICAL_UNICODE_DATABASE,
    CanonicalError,
    assert_canonical_runtime,
    canonical_bytes,
    canonical_loads,
    canonical_sha256,
    normalize_json,
    sha256_bytes,
    strict_json_loads,
    validate_sha256,
)
from simllm.calibration.record_types import (
    RecordIntent,
    RecordObject,
    parse_record_intent,
    record_object,
    validate_schema_id,
)

_FULL_UNICODE_AUTHORITY = (
    sys.implementation.name == "cpython"
    and sys.version_info[:2] == CANONICAL_PYTHON
    and unicodedata.unidata_version == CANONICAL_UNICODE_DATABASE
)
requires_full_unicode_authority = pytest.mark.skipif(
    not _FULL_UNICODE_AUTHORITY,
    reason="full-Unicode calibration authority is frozen to CPython 3.10 / UCD 13.0.0",
)


def test_runtime_matches_the_frozen_full_unicode_authority() -> None:
    assert CANONICAL_PYTHON == (3, 10)
    assert CANONICAL_UNICODE_DATABASE == "13.0.0"
    if _FULL_UNICODE_AUTHORITY:
        assert assert_canonical_runtime() is None
    else:
        with pytest.raises(RuntimeError, match="requires"):
            assert_canonical_runtime()


@requires_full_unicode_authority
def test_canonical_bytes_normalize_and_sort_unicode_keys() -> None:
    value = {
        "z": "e\u0301",
        "e\u0301": "A\u030a",
        "a": [3, True, None, -9],
    }
    assert canonical_bytes(value) == (
        '{"a":[3,true,null,-9],"z":"é","é":"Å"}'.encode()
    )


def test_canonical_strings_use_the_exact_ascii_escape_contract() -> None:
    value = '"\\\b\t\n\f\r\u0000\u0001\u001f/a'
    assert canonical_bytes(value) == (
        b'"\\"\\\\\\b\\t\\n\\f\\r\\u0000\\u0001\\u001f/a"'
    )


def test_arrays_preserve_order_and_objects_do_not() -> None:
    first = {"b": [2, 1], "a": 0}
    second = {"a": 0, "b": [2, 1]}
    reordered_array = {"a": 0, "b": [1, 2]}
    assert canonical_bytes(first) == canonical_bytes(second)
    assert canonical_bytes(first) != canonical_bytes(reordered_array)


@pytest.mark.parametrize(
    "integer",
    [
        -(1 << 4096),
        -(1 << 127) - 1,
        -1,
        0,
        1,
        (1 << 127) - 1,
        1 << 4096,
    ],
)
def test_arbitrary_precision_integers_round_trip_without_float(integer: int) -> None:
    encoded = canonical_bytes({"integer": integer})
    decoded = canonical_loads(encoded)
    assert decoded == {"integer": integer}
    assert type(decoded["integer"]) is int


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1.5,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_writer_rejects_every_float(value: float) -> None:
    with pytest.raises(CanonicalError, match="floating-point"):
        canonical_bytes({"value": value})


@pytest.mark.parametrize(
    "raw",
    [
        b"0.0",
        b"-0.0",
        b"1e2",
        b"1E+2",
        b"NaN",
        b"Infinity",
        b"-Infinity",
    ],
)
def test_reader_rejects_float_and_nonfinite_tokens(raw: bytes) -> None:
    with pytest.raises(CanonicalError, match="floating-point|nonfinite"):
        strict_json_loads(raw)


@pytest.mark.parametrize("raw", [b"-0", b"01", b"+1"])
def test_reader_rejects_nonminimal_integer_spellings(raw: bytes) -> None:
    with pytest.raises(CanonicalError, match="negative-zero|invalid JSON"):
        strict_json_loads(raw)


def test_reader_rejects_duplicate_keys_before_object_construction() -> None:
    with pytest.raises(CanonicalError, match="duplicate object key 'nested'"):
        strict_json_loads(b'{"outer":{"nested":1,"nested":2}}')


@requires_full_unicode_authority
def test_reader_rejects_keys_that_collide_after_nfc_normalization() -> None:
    raw = '{"e\\u0301":1,"é":2}'.encode()
    with pytest.raises(CanonicalError, match="collide after NFC normalization"):
        strict_json_loads(raw)


@requires_full_unicode_authority
def test_writer_rejects_keys_that_collide_after_nfc_normalization() -> None:
    with pytest.raises(CanonicalError, match="collide after NFC normalization"):
        canonical_bytes({"e\u0301": 1, "é": 2})


@pytest.mark.parametrize(
    "raw",
        [
            b'"\\ud800"',
            b'"\\udfff"',
        ],
)
def test_reader_rejects_surrogate_code_units_even_when_paired(raw: bytes) -> None:
    with pytest.raises(CanonicalError, match="surrogate"):
        strict_json_loads(raw)


@requires_full_unicode_authority
def test_reader_accepts_a_valid_surrogate_pair_as_one_unicode_scalar() -> None:
    assert strict_json_loads(b'"\\ud800\\udc00"') == "\U00010000"
    assert canonical_bytes("\U00010000") == b'"\xf0\x90\x80\x80"'


def test_writer_rejects_a_literal_surrogate() -> None:
    with pytest.raises(CanonicalError, match="surrogate"):
        canonical_bytes("\ud800")


@pytest.mark.parametrize(
    "raw, message",
    [
        (b"\xef\xbb\xbf{}", "byte-order mark"),
        (b"\xff", "strict UTF-8"),
        (b'"raw\x01control"', "invalid JSON"),
    ],
)
def test_reader_rejects_bom_invalid_utf8_and_raw_controls(raw: bytes, message: str) -> None:
    with pytest.raises(CanonicalError, match=message):
        strict_json_loads(raw)


@requires_full_unicode_authority
def test_strict_reader_normalizes_but_canonical_reader_requires_exact_bytes() -> None:
    noncanonical = b'{ "z" : "e\\u0301", "a" : 1 }\n'
    assert strict_json_loads(noncanonical) == {"a": 1, "z": "é"}
    with pytest.raises(CanonicalError, match="not canonical"):
        canonical_loads(noncanonical)
    assert canonical_loads(b'{"a":1,"z":"\xc3\xa9"}') == {"a": 1, "z": "é"}


def test_terminal_newline_is_not_part_of_calibration_canonical_bytes() -> None:
    encoded = canonical_bytes({"schema": "example-v1"})
    assert not encoded.endswith(b"\n")
    with pytest.raises(CanonicalError, match="not canonical"):
        canonical_loads(encoded + b"\n")


def test_hash_helpers_cover_exact_bytes() -> None:
    value = {"schema": "example-v1", "wide": 1 << 200}
    encoded = canonical_bytes(value)
    expected = hashlib.sha256(encoded).hexdigest()
    assert expected == "5ddbb1a21a653ac7db5bbd8e2ea8148b53791c8613b10688d6af56d7832ee933"
    assert sha256_bytes(encoded) == expected
    assert canonical_sha256(value) == expected
    assert validate_sha256(expected) == expected


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "0" * 63,
        "0" * 65,
        "A" * 64,
        "g" * 64,
        "../" + "0" * 61,
        0,
    ],
)
def test_sha256_validation_is_exact_and_path_safe(digest: object) -> None:
    with pytest.raises(CanonicalError, match="64 lowercase hexadecimal"):
        validate_sha256(digest)


def test_programmatic_values_reject_non_json_types_and_nonstring_keys() -> None:
    with pytest.raises(CanonicalError, match="outside the canonical JSON domain"):
        canonical_bytes({"bad": b"bytes"})
    with pytest.raises(CanonicalError, match="object key"):
        canonical_bytes({1: "bad"})
    with pytest.raises(CanonicalError, match="outside the canonical JSON domain"):
        canonical_bytes({"bad": {1, 2}})


def test_programmatic_cycles_are_rejected_but_shared_subtrees_are_allowed() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(CanonicalError, match="cyclic array"):
        canonical_bytes(cycle)

    shared = [1, 2]
    assert canonical_bytes([shared, shared]) == b"[[1,2],[1,2]]"


@requires_full_unicode_authority
def test_normalize_json_returns_plain_normalized_structures() -> None:
    normalized = normalize_json({"e\u0301": ("A\u030a",)})
    assert normalized == {"é": ["Å"]}
    assert type(normalized) is dict
    assert type(normalized["é"]) is list


def test_record_object_requires_schema_and_keeps_identity_external() -> None:
    with pytest.raises(CanonicalError, match="missing required field"):
        RecordObject.from_value({"value": 1})
    with pytest.raises(CanonicalError, match="external"):
        RecordObject.from_value({"schema": "example-v1", "record_id": "0" * 64})

    record = RecordObject.from_value({"schema": "example-v1", "value": 1})
    assert record.schema == "example-v1"
    assert record.record_id == sha256_bytes(record.canonical)
    assert "record_id" not in record.value


@pytest.mark.parametrize(
    "schema",
    ["", " ", " example-v1", "example-v1 ", "example\n-v1", 1],
)
def test_record_schema_is_a_strict_nonblank_identifier(schema: object) -> None:
    with pytest.raises(CanonicalError, match="schema"):
        RecordObject.from_value({"schema": schema})


def test_record_object_requires_canonical_input_bytes_and_exact_schema() -> None:
    canonical = b'{"schema":"example-v1","value":1}'
    record = RecordObject.from_bytes(canonical, expected_schema="example-v1")
    assert record.canonical == canonical
    assert record.require_schema("example-v1") is record
    with pytest.raises(CanonicalError, match="expected 'other-v1'"):
        record.require_schema("other-v1")
    with pytest.raises(CanonicalError, match="not canonical"):
        RecordObject.from_bytes(b'{"value":1,"schema":"example-v1"}')


def test_record_value_is_recursively_immutable() -> None:
    record = RecordObject.from_value(
        {"schema": "example-v1", "nested": {"array": [1, {"value": 2}]}}
    )
    with pytest.raises(TypeError):
        record.value["new"] = 3  # type: ignore[index]
    nested = record.value["nested"]
    with pytest.raises(TypeError):
        nested["array"] = ()  # type: ignore[index]
    assert nested["array"] == (1, {"value": 2})


def test_record_factory_accepts_mapping_or_canonical_bytes() -> None:
    mapping_record = record_object({"schema": "example-v1", "value": 7})
    bytes_record = record_object(mapping_record.canonical)
    assert mapping_record == bytes_record


@pytest.mark.parametrize("intent", list(RecordIntent))
def test_record_intent_is_closed_and_round_trips(intent: RecordIntent) -> None:
    assert parse_record_intent(intent.value) is intent


@pytest.mark.parametrize("intent", ["", "measurement", "Evidence", 1, None])
def test_unknown_record_intent_is_rejected(intent: object) -> None:
    with pytest.raises(CanonicalError, match="intent"):
        parse_record_intent(intent)


@requires_full_unicode_authority
def test_schema_validator_rejects_non_nfc_text() -> None:
    with pytest.raises(CanonicalError, match="NFC"):
        validate_schema_id("e\u0301-v1")


def test_nan_payload_never_reaches_hashing() -> None:
    assert math.isnan(float("nan"))
    with pytest.raises(CanonicalError, match="floating-point"):
        canonical_sha256({"schema": "example-v1", "value": float("nan")})


def test_nonascii_work_requires_the_frozen_runtime_authority() -> None:
    if _FULL_UNICODE_AUTHORITY:
        assert canonical_bytes("é") == b'"\xc3\xa9"'
    else:
        with pytest.raises(RuntimeError, match="requires"):
            canonical_bytes("é")
