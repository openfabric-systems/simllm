from __future__ import annotations

import importlib.util
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "examples" / "htsim_uec_bounds_v1" / "run_study.py"
SPEC = importlib.util.spec_from_file_location("htsim_uec_bounds_study", STUDY_PATH)
assert SPEC is not None and SPEC.loader is not None
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    datacenter = tmp_path / "data center"
    binary = tmp_path / "bin dir" / "htsim'uec"
    log = tmp_path / "run dir" / "plan.log"
    return datacenter, binary, log


def _project(source: bytes, tmp_path: Path):
    datacenter, binary, log = _paths(tmp_path)
    projected, ledger, _ = STUDY._project_plan_bytes(
        source=source,
        binary=binary,
        log_path=log,
        datacenter=datacenter,
    )
    audited = STUDY._audit_projection(
        source=source,
        projected=projected,
        binary=binary,
        log_path=log,
        datacenter=datacenter,
    )
    STUDY._require_matching_projection_ledgers(ledger, audited)
    return projected, ledger


def test_projection_preserves_missing_final_lf(tmp_path: Path):
    source = (
        b"matrix.cm\n"
        b"!Experiment exact EOF\n"
        b"!Binary ./old\n"
        b"!Param -topo topo.topo\n"
        b"!tailFCT 18"
    )
    datacenter, binary, log = _paths(tmp_path)
    projected, ledger = _project(source, tmp_path)
    expected = b"".join(
        (
            str((datacenter / "matrix.cm").resolve()).encode() + b"\n",
            b"!Experiment exact EOF\n",
            f"!Binary {shlex.quote(str(binary))}\n".encode(),
            f"!Param -o {shlex.quote(str(log))}\n".encode(),
            f"!Param -topo {shlex.quote(str((datacenter / 'topo.topo').resolve()))}\n".encode(),
            b"!tailFCT 18",
        )
    )
    assert projected == expected
    assert not ledger["source_final_lf"]
    assert not ledger["projected_final_lf"]


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n"])
def test_projection_preserves_uniform_terminators(
    tmp_path: Path, terminator: bytes
):
    source = terminator.join(
        (
            b"matrix.cm",
            b"!Experiment final newline",
            b"!Binary ./old",
            b"!tailFCT 18",
            b"",
        )
    )
    projected, ledger = _project(source, tmp_path)
    assert projected.endswith(terminator)
    assert ledger["source_final_lf"]
    assert ledger["projected_final_lf"]
    if terminator == b"\r\n":
        assert projected.replace(b"\r\n", b"").find(b"\n") == -1


def test_projection_preserves_mixed_terminators_and_unchanged_rows(tmp_path: Path):
    source = (
        b"# comment\r\n"
        b"\n"
        b"matrix.cm\r\n"
        b"!Experiment  exact spacing  \n"
        b"!Binary ./old\r\n"
        b"!Param -end 1000\n"
        b"!tailFCT 18"
    )
    projected, ledger = _project(source, tmp_path)
    assert b"# comment\r\n\n" in projected
    assert b"!Experiment  exact spacing  \n" in projected
    assert b"!Param -end 1000\n" in projected
    assert not ledger["projected_final_lf"]


def test_projection_handles_two_experiments_deterministically(tmp_path: Path):
    source = (
        b"one.cm\n!Experiment one\n!Binary ./old\n!tailFCT 18\n"
        b"two.cm\n!Experiment two\n!Binary ./old\n!Param -topo topo\n!tailFCT 47\n"
    )
    original = bytes(source)
    first, ledger = _project(source, tmp_path)
    second, _ = _project(source, tmp_path)
    assert first == second
    assert source == original
    assert ledger["matrix_replacements"] == 2
    assert ledger["binary_replacements"] == 2
    assert ledger["log_insertions"] == 2
    assert ledger["topology_replacements"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.replace(b"!Experiment exact", b"!Experiment changed"),
        lambda data: data.replace(b"!Param -o ", b"!Param -x ", 1),
        lambda data: data.replace(b"!Param -o ", b"!Param -o duplicate\n!Param -o ", 1),
        lambda data: data + b"\n",
        lambda data: data.replace(b"\n", b"\r\n", 1),
        lambda data: data[:-1],
    ],
)
def test_independent_audit_rejects_one_byte_or_row_drift(
    tmp_path: Path, mutation
):
    source = b"matrix.cm\n!Experiment exact\n!Binary ./old\n!tailFCT 18"
    datacenter, binary, log = _paths(tmp_path)
    projected, _, _ = STUDY._project_plan_bytes(
        source=source,
        binary=binary,
        log_path=log,
        datacenter=datacenter,
    )
    with pytest.raises(ValueError):
        STUDY._audit_projection(
            source=source,
            projected=mutation(projected),
            binary=binary,
            log_path=log,
            datacenter=datacenter,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.replace(b"matrix.cm", b"matrix.zz", 1),
        lambda data: data.replace(b"!Binary ", b"!Binary /wrong/", 1),
        lambda data: data.replace(b"topo.topo", b"wrong.topo", 1),
        lambda data: data.replace(b"plan.log", b"wrong.log", 1),
        lambda data: b"".join(
            row
            for row in data.splitlines(keepends=True)
            if not row.startswith(b"!Param -o ")
        ),
        lambda data: b"".join(
            (
                data.splitlines(keepends=True)[0],
                data.splitlines(keepends=True)[1],
                data.splitlines(keepends=True)[3],
                data.splitlines(keepends=True)[2],
                *data.splitlines(keepends=True)[4:],
            )
        ),
        lambda data: data.replace(b"!tailFCT 18", b"!tailFCT\r18", 1),
    ],
)
def test_independent_audit_rejects_authorized_field_or_structure_drift(
    tmp_path: Path, mutation
):
    source = (
        b"matrix.cm\n"
        b"!Experiment exact\n"
        b"!Binary ./old\n"
        b"!Param -topo topo.topo\n"
        b"!tailFCT 18"
    )
    datacenter, binary, log = _paths(tmp_path)
    projected, _, _ = STUDY._project_plan_bytes(
        source=source,
        binary=binary,
        log_path=log,
        datacenter=datacenter,
    )
    with pytest.raises(ValueError):
        STUDY._audit_projection(
            source=source,
            projected=mutation(projected),
            binary=binary,
            log_path=log,
            datacenter=datacenter,
        )


def test_snapshot_rejects_concurrent_input_mutation(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    snapshot = STUDY._snapshot((first, second))

    second.write_bytes(b"changed\n")

    with pytest.raises(ValueError, match="input changed during the run"):
        STUDY._assert_snapshot(snapshot, boundary="fixture")


@pytest.mark.parametrize(
    "source",
    [
        b"matrix.cm\n!Experiment log\n!Binary ./old\n!Param -o old.log\n!tailFCT 18\n",
        b"matrix.cm\n!Experiment unterminated\n!Binary ./old",
        b"matrix.cm\n!Experiment duplicate\n!Binary ./one\n!Binary ./two\n!tailFCT 18\n",
        b"!Experiment no matrix\n!Binary ./old\n!tailFCT 18\n",
        b"matrix.cm\n!Experiment malformed\n!Binary\n!tailFCT 18\n",
        b"matrix.cm\n!Experiment malformed topo\n!Binary ./old\n!Param -topo\n!tailFCT 18\n",
        b"matrix.cm\n!Experiment bad utf8\n!Binary ./old\n!tailFCT \xff\n",
        b"matrix.cm\n!Experiment bare carriage return\r!Binary ./old\n!tailFCT 18\n",
    ],
)
def test_projector_rejects_ambiguous_or_malformed_sources(
    tmp_path: Path, source: bytes
):
    datacenter, binary, log = _paths(tmp_path)
    with pytest.raises(ValueError):
        STUDY._project_plan_bytes(
            source=source,
            binary=binary,
            log_path=log,
            datacenter=datacenter,
        )
