#!/usr/bin/env python3
"""Structural linter for the module docs under `docs/modules/`.

The contract this script enforces is written down in `docs/modules/FORMAT.md`;
this file is the executable half of that document. `tests/test_docs_format.py`
runs it, so `pytest -q` and CI fail on a violation.

Usage:

    python scripts/check_docs_format.py                 # lint docs/modules
    python scripts/check_docs_format.py path/to/doc.md  # lint specific files
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOC_DIR = REPO_ROOT / "docs" / "modules"

#: Files in the module doc directory that are not module docs.
EXCLUDED_FILES = frozenset({"FORMAT.md", "README.md"})

#: Required second-level sections, in the order they must appear.
REQUIRED_SECTIONS = ("Interface", "Status", "Open tasks")

#: Third-level task buckets inside a registry section, in the order they must
#: appear. "Uncategorized" holds legacy entries that predate the category rule;
#: AGENTS.md migrates them when a task is next edited, not in bulk.
CATEGORY_SECTIONS = ("Precision", "Completeness", "Uncategorized")

#: Stable task-ID prefixes owned by the module registries (AGENTS.md).
TASK_PREFIXES = (
    "ATLAHS",
    "BACK",
    "BRIDGE",
    "COMP",
    "CORE",
    "GOAL",
    "HTSIM",
    "PLACE",
    "PLAY",
    "SGL",
    "TRAF",
    "VLLM",
    "WORK",
)

_PREFIX_ALT = "|".join(TASK_PREFIXES)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
TASK_RE = re.compile(rf"^-\s+(?P<id>(?:{_PREFIX_ALT})-\d+)\b")
TAG_RE = re.compile(
    rf"^-\s+(?:{_PREFIX_ALT})-\d+\s+"
    r"\((?P<category>Precision|Completeness);\s*"
    r"(?P<priority>P[0-2]);\s*(?P<difficulty>[SML])\)"
)
#: A second-level section that carries task entries.
REGISTRY_TITLE_RE = re.compile(r"^(?:Open tasks\b|.*\bfollow-ups\b)", re.IGNORECASE)

#: The banned punctuation dash, spelled by codepoint so this file stays clean.
EM_DASH = "\u2014"


@dataclass(frozen=True)
class Heading:
    level: int
    title: str
    line: int


@dataclass(frozen=True)
class Task:
    task_id: str
    line: int
    category: str | None
    section: str | None
    subsection: str | None


@dataclass
class Document:
    path: Path
    headings: list[Heading]
    tasks: list[Task]
    first_content_line: int
    em_dash_lines: list[int]
    body_lines: list[int]

    @property
    def sections(self) -> list[Heading]:
        return [h for h in self.headings if h.level == 2]

    def children(self, section: Heading) -> list[Heading]:
        """Third-level headings owned by `section`."""
        out: list[Heading] = []
        for heading in self.headings:
            if heading.line <= section.line:
                continue
            if heading.level <= 2:
                break
            if heading.level == 3:
                out.append(heading)
        return out


def is_registry(title: str) -> bool:
    return bool(REGISTRY_TITLE_RE.match(title))


def parse(path: Path) -> Document:
    headings: list[Heading] = []
    tasks: list[Task] = []
    em_dash_lines: list[int] = []
    body_lines: list[int] = []
    first_content_line = 0
    in_fence = False
    current_section: str | None = None
    current_subsection: str | None = None

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if EM_DASH in raw:
            em_dash_lines.append(number)
        if raw.strip() and not first_content_line:
            first_content_line = number

        if FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        heading = HEADING_RE.match(raw)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            headings.append(Heading(level, title, number))
            if level <= 2:
                current_section = title if level == 2 else None
                current_subsection = None
            elif level == 3:
                current_subsection = title
            continue

        if raw.strip():
            body_lines.append(number)

        task = TASK_RE.match(raw)
        if task:
            tag = TAG_RE.match(raw)
            tasks.append(
                Task(
                    task_id=task.group("id"),
                    line=number,
                    category=tag.group("category") if tag else None,
                    section=current_section,
                    subsection=current_subsection,
                )
            )

    return Document(
        path=path,
        headings=headings,
        tasks=tasks,
        first_content_line=first_content_line,
        em_dash_lines=em_dash_lines,
        body_lines=body_lines,
    )


def _check_title(doc: Document, errors: list[tuple[int, str]]) -> None:
    titles = [h for h in doc.headings if h.level == 1]
    if not titles:
        errors.append((1, "missing the level-1 title heading"))
        return
    if titles[0].line != doc.first_content_line:
        errors.append((titles[0].line, "the level-1 title must be the first non-blank line"))
    for extra in titles[1:]:
        errors.append((extra.line, f"only one level-1 heading is allowed, found {extra.title!r}"))

    sections = doc.sections
    summary_end = sections[0].line if sections else 10**9
    has_summary = any(titles[0].line < line < summary_end for line in doc.body_lines)
    if not has_summary:
        errors.append(
            (
                titles[0].line,
                "the title must be followed by a summary paragraph before the first section",
            )
        )


def _check_required_sections(doc: Document, errors: list[tuple[int, str]]) -> None:
    sections = doc.sections
    titles = [h.title for h in sections]
    positions: dict[str, int] = {}
    for required in REQUIRED_SECTIONS:
        found = [i for i, title in enumerate(titles) if title == required]
        if not found:
            errors.append((1, f"missing the required section '## {required}'"))
            continue
        for duplicate in found[1:]:
            errors.append(
                (sections[duplicate].line, f"section '## {required}' appears more than once")
            )
        positions[required] = found[0]

    ordered = [positions[name] for name in REQUIRED_SECTIONS if name in positions]
    if ordered != sorted(ordered):
        errors.append(
            (
                sections[ordered[0]].line if ordered else 1,
                "required sections must appear in the order "
                + " then ".join(f"'## {name}'" for name in REQUIRED_SECTIONS),
            )
        )

    if "Status" in positions and "Open tasks" in positions:
        status_index = positions["Status"]
        if titles[status_index + 1 : status_index + 2] != ["Open tasks"]:
            offending = sections[status_index + 1] if status_index + 1 < len(sections) else None
            errors.append(
                (
                    offending.line if offending else sections[status_index].line,
                    (
                        "'## Open tasks' must directly follow '## Status'; "
                        "move design detail above '## Status'"
                    ),
                )
            )


def _check_registry_sections(doc: Document, errors: list[tuple[int, str]]) -> None:
    for section in doc.sections:
        children = doc.children(section)
        if not is_registry(section.title):
            for child in children:
                if child.title in CATEGORY_SECTIONS:
                    errors.append(
                        (
                            child.line,
                            (
                                f"'### {child.title}' belongs in a task registry "
                                f"section, not '## {section.title}'"
                            ),
                        )
                    )
            continue

        seen: list[str] = []
        for child in children:
            if child.title not in CATEGORY_SECTIONS:
                errors.append(
                    (
                        child.line,
                        f"'### {child.title}' is not a task bucket; allowed buckets are "
                        + ", ".join(f"'### {name}'" for name in CATEGORY_SECTIONS),
                    )
                )
                continue
            if child.title in seen:
                errors.append(
                    (child.line, f"'### {child.title}' appears more than once in this registry")
                )
                continue
            seen.append(child.title)

        canonical = [name for name in CATEGORY_SECTIONS if name in seen]
        if seen != canonical:
            errors.append(
                (
                    children[0].line if children else section.line,
                    "task buckets must appear in the order "
                    + " then ".join(f"'### {name}'" for name in CATEGORY_SECTIONS)
                    + f", found {' then '.join(seen)}",
                )
            )


def _check_tasks(doc: Document, errors: list[tuple[int, str]]) -> None:
    seen_ids: dict[str, int] = {}
    for task in doc.tasks:
        if task.task_id in seen_ids:
            first = seen_ids[task.task_id]
            errors.append(
                (task.line, f"duplicate task id {task.task_id} (first seen on line {first})")
            )
        else:
            seen_ids[task.task_id] = task.line

        if task.section is None or not is_registry(task.section):
            where = f"'## {task.section}'" if task.section else "the preamble"
            errors.append(
                (task.line, f"task {task.task_id} sits in {where}; move it into a registry section")
            )
            continue

        if task.subsection is None:
            errors.append(
                (
                    task.line,
                    (
                        f"task {task.task_id} sits directly under '## {task.section}'; "
                        "every entry belongs in a task bucket"
                    ),
                )
            )
            continue

        expected = task.category or "Uncategorized"
        if task.subsection not in CATEGORY_SECTIONS:
            continue  # already reported by the registry check
        if task.subsection != expected:
            if task.category is None:
                detail = (
                    f"task {task.task_id} carries no '(Category; P<n>; S|M|L)' tag "
                    f"but sits under '### {task.subsection}'"
                )
            else:
                detail = (
                    f"task {task.task_id} is tagged {task.category} "
                    f"but sits under '### {task.subsection}'"
                )
            errors.append((task.line, detail))


def _check_style(doc: Document, errors: list[tuple[int, str]]) -> None:
    for line in doc.em_dash_lines:
        errors.append(
            (line, "em dash is not allowed in this repo; use a comma, colon or parentheses")
        )


def check_file(path: Path) -> list[str]:
    """Return one message per violation found in `path`."""
    doc = parse(path)
    errors: list[tuple[int, str]] = []
    _check_title(doc, errors)
    _check_required_sections(doc, errors)
    _check_registry_sections(doc, errors)
    _check_tasks(doc, errors)
    _check_style(doc, errors)

    try:
        display = path.relative_to(REPO_ROOT)
    except ValueError:
        display = path
    return [f"{display.as_posix()}:{line}: {message}" for line, message in sorted(errors)]


def collect_docs(targets: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        if target.is_dir():
            paths.extend(
                p for p in sorted(target.glob("*.md")) if p.name not in EXCLUDED_FILES
            )
        else:
            paths.append(target)
    return paths


def check_all(targets: list[Path] | None = None) -> list[str]:
    """Return every violation across `targets` (default: `docs/modules/`)."""
    paths = collect_docs(targets or [DEFAULT_DOC_DIR])
    messages: list[str] = []
    for path in paths:
        messages.extend(check_file(path))
    return messages


def uncategorized_debt(targets: list[Path] | None = None) -> dict[str, int]:
    """Count the untagged legacy entries left per file, for reporting only."""
    debt: dict[str, int] = {}
    for path in collect_docs(targets or [DEFAULT_DOC_DIR]):
        count = sum(1 for task in parse(path).tasks if task.category is None)
        if count:
            debt[path.name] = count
    return debt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to check (default: docs/modules)",
    )
    args = parser.parse_args(argv)
    targets = args.paths or [DEFAULT_DOC_DIR]

    messages = check_all(targets)
    for message in messages:
        print(message)

    debt = uncategorized_debt(targets)
    if debt:
        total = sum(debt.values())
        detail = ", ".join(f"{name} {count}" for name, count in sorted(debt.items()))
        print(f"note: {total} untagged legacy entries pending category migration ({detail})")

    if messages:
        print(f"FAIL: {len(messages)} format violation(s); see docs/modules/FORMAT.md")
        return 1
    print(f"OK: {len(collect_docs(targets))} module doc(s) match docs/modules/FORMAT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
