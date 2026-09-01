"""History-wide identity gate.

Every commit in this repository is authored under the maintainer
identity, and no commit message may carry an agent co-author trailer:
GitHub counts co-author trailers on the default branch as contributors,
so a single trailer puts the agent on the public contributor list. The
Aug 31 2026 quad-wave close let three such trailers through and main
had to be rewritten to remove them; this test keeps any pull request
that reintroduces one red in CI, where full history is available
because the checkout uses fetch-depth 0.
"""

from __future__ import annotations

import subprocess

FORBIDDEN_FRAGMENTS = (
    "anthropic.com",
    "co-authored-by: claude",
)


def _history_bodies() -> str:
    completed = subprocess.run(
        ["git", "log", "--format=%b", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.lower()


def test_no_agent_coauthor_trailer_anywhere_in_history() -> None:
    bodies = _history_bodies()
    for fragment in FORBIDDEN_FRAGMENTS:
        assert fragment not in bodies, (
            f"a commit message contains {fragment!r}; commits must use the "
            "maintainer identity with no agent co-author trailer"
        )
