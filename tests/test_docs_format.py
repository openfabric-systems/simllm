"""The module docs under docs/modules/ must match docs/modules/FORMAT.md."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_docs_format.py"
DOC_DIR = REPO_ROOT / "docs" / "modules"


def _load_checker():
    spec = importlib.util.spec_from_file_location("simllm_check_docs_format", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_module_docs_match_the_format_contract():
    messages = checker.check_all([DOC_DIR])
    assert not messages, "docs/modules/FORMAT.md violations:\n" + "\n".join(messages)


def test_every_module_doc_is_checked():
    checked = {path.name for path in checker.collect_docs([DOC_DIR])}
    on_disk = {path.name for path in DOC_DIR.glob("*.md")}
    assert checked == on_disk - checker.EXCLUDED_FILES
    assert len(checked) > 5


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            "## Open tasks\n\n### Completeness\n\n- CORE-1 (Completeness; P1; S): x.\n\n"
            "### Precision\n\n- CORE-2 (Precision; P1; S): y.\n",
            "task buckets must appear in the order",
            id="bucket-order",
        ),
        pytest.param(
            "## Open tasks\n\n### Precision\n\n- CORE-1 (Completeness; P1; S): x.\n",
            "is tagged Completeness but sits under",
            id="tag-mismatch",
        ),
        pytest.param(
            "## Open tasks\n\n- CORE-1 (Precision; P1; S): x.\n",
            "every entry belongs in a task bucket",
            id="loose-entry",
        ),
        pytest.param(
            "## Open tasks\n\n### Uncategorized\n\n- CORE-1 (Precision; P1; S): x.\n",
            "is tagged Precision but sits under",
            id="tagged-in-uncategorized",
        ),
        pytest.param(
            "## Open tasks\n\n### Precision\n\n- CORE-1: x.\n",
            "carries no '(Category; P<n>; S|M|L)' tag",
            id="untagged-in-precision",
        ),
        pytest.param(
            "## Open tasks\n\n### Precision\n\n- CORE-1 (Precision; P1; S): x.\n"
            "- CORE-1 (Precision; P1; S): y.\n",
            "duplicate task id CORE-1",
            id="duplicate-id",
        ),
        pytest.param(
            "## Open tasks\n\n### Precision\n\n- CORE-1 (Precision; P1; S): a \u2014 b.\n",
            "em dash is not allowed",
            id="em-dash",
        ),
    ],
)
def test_rejects_malformed_registry(tmp_path, body, expected):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "# simllm.sample\n\nSummary line.\n\n## Interface\n\nStuff.\n\n"
        "## Status\n\nStuff.\n\n" + body,
        encoding="utf-8",
    )
    messages = checker.check_file(doc)
    assert any(expected in message for message in messages), messages


def test_accepts_the_canonical_skeleton(tmp_path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "# simllm.sample\n\nSummary line.\n\n"
        "## Why\n\nContext.\n\n"
        "## Interface\n\nStuff.\n\n"
        "## Detail\n\nStuff.\n\n"
        "## Status\n\nStuff.\n\n"
        "## Open tasks\n\nLegend.\n\n"
        "### Precision\n\n- CORE-1 (Precision; P1; S): x.\n\n"
        "### Completeness\n\n- CORE-2 (Completeness; P2; M): y.\n\n"
        "### Uncategorized\n\n- CORE-3: z.\n\n"
        "## Backend-repo follow-ups\n\n"
        "### Precision\n\n- HTSIM-1 (Precision; P0; L): w.\n",
        encoding="utf-8",
    )
    assert checker.check_file(doc) == []


def test_rejects_detail_between_status_and_open_tasks(tmp_path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "# simllm.sample\n\nSummary line.\n\n## Interface\n\nStuff.\n\n"
        "## Status\n\nStuff.\n\n## Detail\n\nStuff.\n\n## Open tasks\n\nNone currently.\n",
        encoding="utf-8",
    )
    messages = checker.check_file(doc)
    assert any("must directly follow '## Status'" in message for message in messages), messages


def test_rejects_missing_required_section(tmp_path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "# simllm.sample\n\nSummary line.\n\n## Status\n\nStuff.\n\n"
        "## Open tasks\n\nNone currently.\n",
        encoding="utf-8",
    )
    messages = checker.check_file(doc)
    assert any("missing the required section '## Interface'" in m for m in messages), messages


def test_rejects_missing_summary(tmp_path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "# simllm.sample\n\n## Interface\n\nStuff.\n\n## Status\n\nStuff.\n\n"
        "## Open tasks\n\nNone currently.\n",
        encoding="utf-8",
    )
    messages = checker.check_file(doc)
    assert any("summary paragraph" in message for message in messages), messages


def test_script_runs_as_a_command():
    assert checker.main([str(DOC_DIR)]) == 0
