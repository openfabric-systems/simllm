from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE_PATH = Path("docs/architecture.md")
IMMUTABLE_MERLIN_FREEZE_PATH = Path(
    "examples/merlin_collective_capture_v1/expectations.md"
)
ARCHITECTURE_STORAGE_LITERAL = "/" + "data3/yifeng/"
ARCHITECTURE_STORAGE_CONTEXT = (
    "Bulk raw traces stay outside Git under\n"
    f"`{ARCHITECTURE_STORAGE_LITERAL}`."
)
IMMUTABLE_MERLIN_FREEZE_LITERALS = (
    "~" + "/simllm-data/",
    "/" + "data3/yifeng/simllm-dev/planmode-runs/traf77-t2/",
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
MARKDOWN_API_ROUTE = re.compile(r"`/(?:generate|v1/[A-Za-z0-9_.{}-]+)`")
PERSONAL_SCRIPT_PATH = re.compile(
    r"(?:/(?:home|Users|data[0-9]*|scratch|gpfs|nfs)/[^/\s`'\"]+|"
    r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s`'\"]+|"
    r"(?<![\\])\\\\[^\s\\]+\\[^\s`'\"<>|]+|"
    r"(?<![A-Za-z0-9_])~[\\/])"
)
PATH_SPLIT_QUOTE = re.compile(r"(?<=/)['\"]|(?<=[A-Za-z]:\\)['\"]")


def _tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [Path(raw.decode()) for raw in completed.stdout.split(b"\0") if raw]


_BARE_RATIONAL = re.compile(r"\A/\d+(?:\.\d+)?\Z")
_JSON_ESCAPE = re.compile(r'(?:\\)+(?:u[0-9A-Fa-f]{4}|["\\/bfnrt])')


def _is_bare_rational(match: str) -> bool:
    """A slash followed only by a number is a denominator, not a path.

    Exact rational metrics render as `1/3` and `20/1`, and paired
    measurements render as `0.510%/1.768%`, so prose and result records
    legitimately contain a slash before a number. No machine-local path is
    ever spelled `/7` or `/1.768`, which keeps this exemption narrow.
    """

    return bool(_BARE_RATIONAL.match(match))


def _is_json_escape_vector(match: str) -> bool:
    """Ignore a UNC-shaped run made entirely from JSON escape spellings.

    Canonical-JSON conformance vectors contain source-level backslashes before
    control escapes and surrogate pairs. The UNC matcher can begin at two of
    those backslashes, but a real UNC or personal path retains an unconsumed
    separator or path component after the JSON escapes are removed.
    """

    if not re.search(r"(?:\\)+u[0-9A-Fa-f]{4}", match):
        return False
    return "\\" not in _JSON_ESCAPE.sub("", match)


def _line_matches(text: str, pattern: re.Pattern[str]) -> list[tuple[int, str]]:
    return [
        (line_number, match.group(0))
        for line_number, line in enumerate(text.splitlines(), start=1)
        for match in pattern.finditer(line)
    ]


def _mask_document_urls(text: str) -> str:
    return "\n".join(
        MARKDOWN_API_ROUTE.sub(
            "", HTML_ROOT_URL.sub("", MARKDOWN_ROOT_URL.sub("", line))
        )
        for line in text.splitlines()
    )


def _collapse_path_split_quotes(text: str) -> str:
    """Expose a path whose first component is split after its root marker."""

    return PATH_SPLIT_QUOTE.sub("", text)


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
        if relative_path == IMMUTABLE_MERLIN_FREEZE_PATH:
            # TRAF77-T2A is explicitly forbidden from changing its committed
            # expectations-only freeze. Mask only the two binding paths that
            # freeze already carried, and keep every other path check active.
            for literal in IMMUTABLE_MERLIN_FREEZE_LITERALS:
                assert text.count(literal) == 1
                text = text.replace(literal, "", 1)
        if relative_path.suffix.lower() in {".markdown", ".md"}:
            text = _mask_document_urls(text)
        text = _collapse_path_split_quotes(text)
        for pattern in patterns:
            for line_number, match in _line_matches(text, pattern):
                if _is_bare_rational(match):
                    continue
                violations.append(f"{relative_path}:{line_number}: {match}")

    assert not violations, "non-portable filesystem paths:\n" + "\n".join(violations)


def test_scripts_have_no_personal_path_defaults() -> None:
    violations: list[str] = []
    # This source necessarily contains the matcher text and synthetic fixtures.
    scanner_path = Path(__file__).resolve().relative_to(REPO_ROOT)
    for relative_path in _tracked_files():
        if (
            relative_path == scanner_path
            or relative_path.suffix.lower() not in SCRIPT_SUFFIXES
        ):
            continue
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, match in _line_matches(text, PERSONAL_SCRIPT_PATH):
            if _is_json_escape_vector(match):
                continue
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


def test_absolute_path_matcher_cannot_be_bypassed_by_split_string_quotes() -> None:
    split_paths = (
        "/\"data3/account/project",
        "/'home/account/project",
        "C:\\\"Users\\account\\project",
    )

    collapsed = tuple(_collapse_path_split_quotes(path) for path in split_paths)
    assert POSIX_ABSOLUTE_PATH.fullmatch(collapsed[0])
    assert POSIX_ABSOLUTE_PATH.fullmatch(collapsed[1])
    assert WINDOWS_DRIVE_PATH.fullmatch(collapsed[2])


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
        "`" + "/generate" + "`",
    )
    assert not any(
        POSIX_ABSOLUTE_PATH.search(_mask_document_urls(url))
        for url in root_relative_urls
    )
    assert POSIX_ABSOLUTE_PATH.search(_mask_document_urls("`" + "/usr" + "`"))


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


def test_json_escape_exemption_is_narrower_than_personal_paths() -> None:
    escape_vectors = (
        r'\\u0000\\u0001\\u001f',
        r'\\ud800\\udc00',
        r'\\","controls":"\b\f\n\r\t\u0001',
    )
    personal_paths = (
        r"\\server\share\u0001",
        r"\\server\u0001\project",
        r"C:\Users\u0001\project",
        "/home/u0001/project",
    )

    assert all(_is_json_escape_vector(value) for value in escape_vectors)
    assert not any(_is_json_escape_vector(value) for value in personal_paths)
