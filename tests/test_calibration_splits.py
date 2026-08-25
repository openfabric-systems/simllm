"""Immutable split-isolation checks for offline calibration evidence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from simllm.calibration.splits import (
    CalibrationSplit,
    ImmutableEvidenceSplit,
    SplitMember,
)


def _digest(number: int) -> str:
    return f"{number:064x}"


def _members() -> tuple[SplitMember, ...]:
    return (
        SplitMember(_digest(1), _digest(101), CalibrationSplit.TRAIN),
        SplitMember(_digest(2), _digest(101), CalibrationSplit.TRAIN),
        SplitMember(_digest(3), _digest(102), CalibrationSplit.VALIDATION),
        SplitMember(_digest(4), _digest(103), CalibrationSplit.TEST),
    )


def test_immutable_split_requires_all_three_partitions_and_keeps_groups_together() -> None:
    split = ImmutableEvidenceSplit(_members())
    assert split.records(CalibrationSplit.TRAIN) == (_digest(1), _digest(2))
    assert split.records(CalibrationSplit.VALIDATION) == (_digest(3),)
    assert split.records(CalibrationSplit.TEST) == (_digest(4),)


def test_split_factory_canonicalizes_only_record_order() -> None:
    split = ImmutableEvidenceSplit.create(tuple(reversed(_members())))
    assert split.members == _members()


def test_split_rejects_shape_or_graph_group_leakage() -> None:
    members = list(_members())
    members[2] = replace(members[2], split_group_sha256=_digest(101))
    with pytest.raises(ValueError, match="split-group leakage"):
        ImmutableEvidenceSplit(tuple(members))


@pytest.mark.parametrize(
    "members,message",
    (
        ((), "must not be empty"),
        (_members()[:3], "every split must be nonempty"),
        (_members()[2:], "every split must be nonempty"),
        (_members() + (_members()[0],), "sorted|duplicate"),
        (tuple(reversed(_members())), "must be sorted"),
    ),
)
def test_split_rejects_incomplete_duplicate_or_noncanonical_members(
    members: tuple[SplitMember, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ImmutableEvidenceSplit(members)


def test_split_member_rejects_invalid_identity_and_enum() -> None:
    with pytest.raises(ValueError, match="64 lowercase"):
        SplitMember("bad", _digest(1), CalibrationSplit.TRAIN)
    with pytest.raises(TypeError, match="CalibrationSplit"):
        SplitMember(_digest(1), _digest(2), "train")  # type: ignore[arg-type]


def test_split_records_requires_closed_enum() -> None:
    with pytest.raises(TypeError, match="CalibrationSplit"):
        ImmutableEvidenceSplit(_members()).records("train")  # type: ignore[arg-type]


def test_split_objects_are_immutable() -> None:
    split = ImmutableEvidenceSplit(_members())
    with pytest.raises(FrozenInstanceError):
        split.members = ()  # type: ignore[misc]
