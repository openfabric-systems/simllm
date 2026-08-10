from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER_PATH = Path("tests/test_path_portability.py")
ARCHITECTURE_PATH = Path("docs/architecture.md")
ARCHITECTURE_STORAGE_LITERAL = "/" + "data3/yifeng/"
ARCHITECTURE_STORAGE_CONTEXT = (
    "Bulk raw traces stay outside Git under\n"
    f"`{ARCHITECTURE_STORAGE_LITERAL}`."
)
PORTABLE_SUFFIXES = {
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".ipp",
    ".markdown",
    ".md",
    ".tpp",
}
SCRIPT_SUFFIXES = {".bash", ".bat", ".cmd", ".ps1", ".py", ".sh", ".zsh"}

POSIX_ABSOLUTE_PATH = re.compile(
    r"(?<![)/:A-Za-z0-9_.<>{}])/(?!/|\.\.?/)"
    r"[A-Za-z0-9_.@+~-]+(?:/[A-Za-z0-9_.@+~-]+)*/?"
)
WINDOWS_DRIVE_PATH = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:[^\s`'\"<>|]+[\\/])*[^\s`'\"<>|]*"
)
WINDOWS_UNC_PATH = re.compile(r"(?<![\\])\\\\[^\s\\]+\\[^\s`'\"<>|]+")
HOME_SHORTCUT_PATH = re.compile(r"(?<![A-Za-z0-9_])~[\\/]")
MARKDOWN_ROOT_URL = re.compile(r"\]\(/(?!/)[^)\s]+\)")
HTML_ROOT_URL = re.compile(
    r"\b(?:href|src)\s*=\s*([\"'])/(?!/).*?\1",
    flags=re.IGNORECASE,
)
PERSONAL_SCRIPT_PATH = re.compile(
    r"(?:/(?:home|Users|data[0-9]*|scratch|gpfs|nfs)/[^/\s`'\"]+|"
    r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s`'\"]+|"
    r"(?<![\\])\\\\[^\s\\]+\\[^\s`'\"<>|]+|"
    r"(?<![A-Za-z0-9_])~[\\/])"
)


def _tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [Path(raw.decode()) for raw in completed.stdout.split(b"\0") if raw]


def _line_matches(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    return [
        (line_number, match.group(0))
        for line_number, line in enumerate(text.splitlines(), start=1)
        for match in pattern.finditer(line)
    ]


def _mask_document_urls(text: str) -> str:
    return "\n".join(
        HTML_ROOT_URL.sub("", MARKDOWN_ROOT_URL.sub("", line))
        for line in text.splitlines()
    )


def test_markdown_and_cpp_paths_are_portable() -> None:
    violations: list[str] = []
    patterns = (
        POSIX_ABSOLUTE_PATH,
        WINDOWS_DRIVE_PATH,
        WINDOWS_UNC_PATH,
        HOME_SHORTCUT_PATH,
    )

    for relative_path in _tracked_files():
        if relative_path.suffix.lower() not in PORTABLE_SUFFIXES:
            continue
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        if relative_path == ARCHITECTURE_PATH:
            assert text.count(ARCHITECTURE_STORAGE_CONTEXT) == 1
            text = text.replace(ARCHITECTURE_STORAGE_LITERAL, "", 1)
        if relative_path.suffix.lower() in {".markdown", ".md"}:
            text = _mask_document_urls(text)
        for pattern in patterns:
            for line_number, match in _line_matches(text, pattern):
                violations.append(f"{relative_path}:{line_number}: {match}")

    assert not violations, "non-portable filesystem paths:\n" + "\n".join(violations)


def test_scripts_have_no_personal_path_defaults() -> None:
    violations: list[str] = []
    for relative_path in _tracked_files():
        # This scanner contains intentional matcher fixtures tested below.
        if relative_path == SCANNER_PATH:
            continue
        if relative_path.suffix.lower() not in SCRIPT_SUFFIXES:
            continue
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, match in _line_matches(text, PERSONAL_SCRIPT_PATH):
            violations.append(f"{relative_path}:{line_number}: {match}")

    assert not violations, "personal script filesystem paths:\n" + "\n".join(violations)


def test_absolute_path_matchers_cover_supported_forms() -> None:
    posix_paths = (
        "/" + "usr",
        "/" + "etc/service/config",
        "/" + "var/tmp",
        "/" + "home/alice/project",
        "/" + "CMakeLists.txt",
    )
    windows_paths = (
        "C:" + r"\Users\alice\project",
        "D:" + "/Users/alice/project",
    )
    unc_path = "\\" + r"\server\share\project"

    assert all(POSIX_ABSOLUTE_PATH.fullmatch(path) for path in posix_paths)
    assert all(WINDOWS_DRIVE_PATH.fullmatch(path) for path in windows_paths)
    assert WINDOWS_UNC_PATH.fullmatch(unc_path)
    assert HOME_SHORTCUT_PATH.match("~" + "/project")


def test_absolute_path_matcher_ignores_portable_path_forms() -> None:
    portable_paths = (
        "https:" + "//example.com/project/file",
        "../examples/study",
        "./scripts/check.sh",
        "${SIMLLM_DATA_ROOT}" + "/study/run",
        "<configured-root>" + "/study/run",
    )

    assert not any(POSIX_ABSOLUTE_PATH.search(path) for path in portable_paths)

    root_relative_urls = (
        "[Guide](" + "/docs/guide)",
        '<a href="' + '/docs/guide">Guide</a>',
        '<img src="' + '/assets/plot.png">',
    )
    assert not any(
        POSIX_ABSOLUTE_PATH.search(_mask_document_urls(url))
        for url in root_relative_urls
    )


def test_architecture_exception_does_not_allow_a_deeper_path() -> None:
    deeper_path = ARCHITECTURE_STORAGE_LITERAL + "private/run"
    mutated_context = ARCHITECTURE_STORAGE_CONTEXT.replace(
        ARCHITECTURE_STORAGE_LITERAL,
        deeper_path,
    )

    assert ARCHITECTURE_STORAGE_CONTEXT not in mutated_context
    assert POSIX_ABSOLUTE_PATH.search(deeper_path)


def test_personal_script_matcher_covers_user_roots() -> None:
    personal_paths = (
        "/" + "home/alice/project",
        "/" + "scratch/alice/project",
        "C:" + r"\Users\alice\project",
        "D:" + "/Users/alice/project",
        "\\" + r"\server\share\project",
        "~" + "/project",
    )

    assert all(PERSONAL_SCRIPT_PATH.search(path) for path in personal_paths)
