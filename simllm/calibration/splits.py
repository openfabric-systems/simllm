"""Immutable train, validation, and untouched-test partition rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .canonical import CanonicalError, validate_sha256


class CalibrationSplit(str, Enum):
    """Closed evidence partitions used by the offline compiler."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitMember:
    """One evidence record and the identity group that must stay together."""

    record_sha256: str
    split_group_sha256: str
    split: CalibrationSplit

    def __post_init__(self) -> None:
        try:
            validate_sha256(self.record_sha256, "SplitMember.record_sha256")
            validate_sha256(
                self.split_group_sha256,
                "SplitMember.split_group_sha256",
            )
        except CanonicalError as error:
            raise ValueError(str(error)) from error
        if not isinstance(self.split, CalibrationSplit):
            raise TypeError("SplitMember.split: expected CalibrationSplit")


@dataclass(frozen=True, slots=True)
class ImmutableEvidenceSplit:
    """A total, leakage-free partition over content-addressed evidence."""

    members: tuple[SplitMember, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple):
            raise TypeError("ImmutableEvidenceSplit.members: expected a tuple")
        if not self.members:
            raise ValueError("ImmutableEvidenceSplit.members: must not be empty")
        for index, member in enumerate(self.members):
            if not isinstance(member, SplitMember):
                raise TypeError(
                    f"ImmutableEvidenceSplit.members[{index}]: expected SplitMember"
                )
        identities = tuple(member.record_sha256 for member in self.members)
        if identities != tuple(sorted(identities)):
            raise ValueError(
                "ImmutableEvidenceSplit.members: must be sorted by record SHA-256"
            )
        if len(identities) != len(set(identities)):
            raise ValueError(
                "ImmutableEvidenceSplit.members: duplicate evidence record"
            )

        present = {member.split for member in self.members}
        missing = tuple(split.value for split in CalibrationSplit if split not in present)
        if missing:
            raise ValueError(
                "ImmutableEvidenceSplit.members: every split must be nonempty; "
                f"missing {missing}"
            )

        group_splits: dict[str, CalibrationSplit] = {}
        for member in self.members:
            previous = group_splits.setdefault(member.split_group_sha256, member.split)
            if previous is not member.split:
                raise ValueError(
                    "ImmutableEvidenceSplit.members: split-group leakage between "
                    f"{previous.value!r} and {member.split.value!r}"
                )

    @classmethod
    def create(cls, members: tuple[SplitMember, ...]) -> ImmutableEvidenceSplit:
        """Sort caller-supplied members into the canonical in-memory order."""

        return cls(tuple(sorted(members, key=lambda member: member.record_sha256)))

    def records(self, split: CalibrationSplit) -> tuple[str, ...]:
        """Return record identities in one frozen partition."""

        if not isinstance(split, CalibrationSplit):
            raise TypeError("split must be a CalibrationSplit")
        return tuple(
            member.record_sha256 for member in self.members if member.split is split
        )


__all__ = [
    "CalibrationSplit",
    "ImmutableEvidenceSplit",
    "SplitMember",
]
