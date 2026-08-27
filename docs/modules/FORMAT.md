# Module doc format

Every module doc under `docs/modules/` follows one skeleton, so a reader can
move between modules without relearning the layout and so the open-task
registry stays machine-checkable. `scripts/check_docs_format.py` is the
executable half of this document, and `tests/test_docs_format.py` runs it, so
`pytest -q` and CI fail on a violation.

This file and any `README.md` in the same directory are not module docs and
are excluded from the check.

## Skeleton

```
# <module identity>                     level-1 title, first non-blank line

<summary paragraph>                     what the module is, in prose

## <optional context sections>          e.g. "## Why" for a design-only module
## Interface                            required: the surface other code uses
## <optional design sections>           mechanism detail, contracts, tables
## Status                               required: what is landed and validated
## Open tasks                           required: the registry, see below
## <optional trailing registry>         e.g. backend-repo follow-ups
```

Rules the linter enforces:

- Exactly one level-1 heading, and it is the first non-blank line.
- A non-empty summary paragraph sits between the title and the first section.
- `## Interface`, `## Status` and `## Open tasks` each appear exactly once,
  in that relative order.
- `## Open tasks` directly follows `## Status`. Design detail belongs above
  `## Status`, not between the two. A doc that wants a long plan for one task
  (see [compute.md](compute.md)) puts that section in the design block.
- Optional sections are free in title and count. Context sections may precede
  `## Interface`; detail sections sit between `## Interface` and `## Status`.

## Task registry sections

A registry section carries task entries. `## Open tasks` is always one; a
section whose title contains "follow-ups" is also one, which is how
[backends.md](backends.md) tracks work executed in the backend repos. Task
entries are not allowed anywhere else.

A registry section is laid out as:

```
## Open tasks

<optional preamble: the tag legend, closure and retraction notes>

### Precision
- <ID> (Precision; P<n>; S|M|L): <what the current approximation is and what
  observable identifies the replacement>

### Completeness
- <ID> (Completeness; P<n>; S|M|L): <the unavailable path and the off path
  that must be preserved>

### Uncategorized
- <ID>: <legacy entry that predates the category rule>
```

Rules the linter enforces:

- The only third-level headings inside a registry section are
  `### Precision`, `### Completeness` and `### Uncategorized`, each at most
  once, in that order. Precision comes first because active-path precision
  normally precedes P2 completeness (AGENTS.md).
- A bucket is present only when it has entries. A registry with no open work
  says so in prose (see [goal.md](goal.md)) and carries no buckets.
- Every task entry sits inside a bucket, never directly under the section
  heading and never in the preamble.
- An entry tagged `(Precision; ...)` sits under `### Precision`, an entry
  tagged `(Completeness; ...)` under `### Completeness`, and an entry with no
  category tag under `### Uncategorized`.
- Task ids are unique within a file.
- Closure and retraction notes go in the preamble, above the buckets, so the
  buckets hold open work only.

### Task entry grammar

```
- <PREFIX>-<n> (<Category>; P<0-2>; <S|M|L>)[ (<qualifier>)]: <description>
```

`<PREFIX>` is the module's stable prefix (`CORE-`, `WORK-`, `COMP-`,
`PLACE-`, `TRAF-`, `GOAL-`, `PLAY-`, `BACK-`, `VLLM-`, `SGL-`, `BRIDGE-`,
`DEPLOY-`, and `HTSIM-`/`ATLAHS-` for backend-repo follow-ups). The category
tag is the first parenthetical after the id; an optional second parenthetical
carries a qualifier such as `(remaining half)`. Categories, priorities and
difficulties are defined in AGENTS.md, which is the authority on what each
level means.

### The Uncategorized bucket

AGENTS.md migrates existing entries to a category when the task is next
changed, not through unrelated bulk churn. `### Uncategorized` is where those
entries wait: it makes the remaining migration debt countable without forcing
anyone to invent a priority and difficulty they cannot defend. The linter
prints the per-file count as a note. Moving an entry out of this bucket means
assigning a real `(Category; P<n>; S|M|L)` tag, so it is a content change and
belongs with the change that touches the task.

## Writing style

The repo-wide rules in AGENTS.md apply here in full. The one the linter adds
to the structural rules above is the em dash ban: the em dash character
(U+2014) appears nowhere in these docs, in prose or in code blocks. Use
"i.e.", "e.g.", commas, colons, semicolons or parentheses instead. Markdown
table separators (`|---|`) are fine.

The rules the linter does not check still hold, in particular the filesystem
path portability rule: no absolute paths and no path components derived from a
person, account, home directory or machine in any tracked Markdown file.

## Running the check

```bash
python scripts/check_docs_format.py
```

It exits non-zero and prints `path:line: message` for every violation. Pass
explicit paths to check a subset:

```bash
python scripts/check_docs_format.py docs/modules/backends.md
```
