"""Harness self-checks for the TRAF-51 load-bearing recalibration.

The freeze (examples/merlin_ss_fabric_loadbearing_v1/expectations.md)
promises that the late-arrival descriptor derivation, applied to the x4
cell, reproduces the frozen think table byte-exactly, that the frozen
fail-closed clauses are machine-enforced, and that the runner's frozen
cell matrix carries exactly the frozen think times, offered rates, starts
and durations. These tests enforce those promises in CI, with
mutation-sensitive negative controls so the checks cannot pass vacuously.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import pathlib
from fractions import Fraction

import pytest

STUDY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "merlin_ss_fabric_loadbearing_v1"
)
DATASET = (
    pathlib.Path(__file__).resolve().parents[1]
    / "examples"
    / "merlin_fabric_flow_capture_v1"
    / "dataset"
)

FROZEN_THINK_RATE_PS = [4_642_143_756, 2_255_636_718, 2_263_071_395, 2_495_255_483]
FROZEN_THINK_P50_PS = [4_798_387_720, 1_783_214_720, 1_835_331_720, 1_794_560_720]
FROZEN_OFFERED_BPS = [13_468_749_005, 25_850_334_413, 25_776_514_662, 23_665_940_890]
F_PS = 340_417_280


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, STUDY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load("analyze_recalibration")
runner = _load("run_cells")


def test_late_derivation_reproduces_the_frozen_x4_think_table():
    descriptor = analyzer.derive_late_cell(DATASET, "x4-incast", [0, 1, 2, 3])
    assert [f["think_ps"] for f in descriptor["flows"]] == FROZEN_THINK_RATE_PS
    assert [f["src"] for f in descriptor["flows"]] == [4, 5, 6, 7]
    assert all(f["dst"] == 19 for f in descriptor["flows"])
    assert [f["start_ps"] for f in descriptor["flows"]] == [
        0, 400_000_000, 800_000_000, 1_200_000_000]
    assert [f["measured_count"] for f in descriptor["flows"]] == [
        4014, 7704, 7682, 7053]
    assert descriptor["instance"] == "x4buf32"
    assert descriptor["duration_ps"] == 6_000_000_000_000
    assert descriptor["chunk_payload_bytes"] == 8_388_608
    assert descriptor["measured_group_aggregate_bps"] == pytest.approx(
        11_095_192_371.2)


def test_frozen_runner_matrix_matches_the_freeze():
    cells = {str(cell["name"]): cell for cell in runner.CELLS}
    assert sorted(cells) == sorted([
        "ctrl-v1", "ctrl-b32", "ctrl-b64", "mj2x-v1", "x4p-v1",
        "x4r-b32", "x4r-b64", "x4r-v1", "x4p50-b32", "sat-v1", "sat-b32"])
    assert cells["x4r-v1"]["expected_exit"] == 2
    assert all(cells[name]["expected_exit"] == 0
               for name in cells if name != "x4r-v1")
    for name in ("x4r-b32", "x4r-b64", "x4r-v1"):
        specs = [a for a in cells[name]["args"] if str(a).startswith("src=")]
        assert [int(s.split("think_ps=")[1].split(",")[0]) for s in specs] == \
            FROZEN_THINK_RATE_PS
    p50_specs = [a for a in cells["x4p50-b32"]["args"] if str(a).startswith("src=")]
    assert [int(s.split("think_ps=")[1].split(",")[0]) for s in p50_specs] == \
        FROZEN_THINK_P50_PS
    paced_specs = [a for a in cells["x4p-v1"]["args"] if str(a).startswith("src=")]
    assert [int(s.split("offered_bps=")[1].split(",")[0]) for s in paced_specs] == \
        FROZEN_OFFERED_BPS
    for name, duration in (("ctrl-v1", "200000000000"), ("mj2x-v1", "300000000000"),
                           ("x4r-b32", "6000000000000"), ("sat-v1", "5000000000")):
        args = [str(a) for a in cells[name]["args"]]
        assert args[args.index("-duration_ps") + 1] == duration
    assert runner.EXPECTED_BINARY_SHA256 == analyzer.FROZEN_BINARY_SHA256
    assert runner.EXPECTED_PIN == analyzer.FROZEN_PIN
    assert analyzer.FROZEN_THINK_RATE_PS == FROZEN_THINK_RATE_PS
    assert analyzer.FROZEN_THINK_P50_PS == FROZEN_THINK_P50_PS


def _synthetic_cell(root: pathlib.Path, name: str, counts: list[int],
                    chunk_bytes: int = 8_388_608) -> None:
    cell_dir = root / name
    cell_dir.mkdir(parents=True)
    (cell_dir / "nccl_interfaces.txt").write_text("stub\n", encoding="utf-8")
    (cell_dir / "port_tx_deltas.json").write_text("{}\n", encoding="utf-8")
    for flow, count in enumerate(counts):
        rows = ["cell,flow,side,chunk_idx,chunk_bytes,t_ns_since_t0"]
        for index in range(count):
            t_ns = 160_000_000_000 + index * (20_000_000_000 // (count + 1))
            rows.append(f"{name},{flow},dest,{index},{chunk_bytes},{t_ns}")
        with gzip.open(cell_dir / f"{name}_flow{flow}_dest.csv.gz", "wt",
                       encoding="utf-8", newline="") as handle:
            handle.write("\n".join(rows) + "\n")


def test_late_derivation_is_mutation_sensitive(tmp_path):
    _synthetic_cell(tmp_path, "synth", [30, 40])
    descriptor = analyzer.derive_late_cell(tmp_path, "synth", [0, 1])
    expected = [int(round(Fraction(20 * 10**12, n)) - F_PS) for n in (30, 40)]
    assert [f["think_ps"] for f in descriptor["flows"]] == expected
    assert descriptor["flows"][0]["think_ps"] not in FROZEN_THINK_RATE_PS


def test_late_derivation_fails_closed(tmp_path):
    _synthetic_cell(tmp_path, "synth", [30, 40])
    with pytest.raises(SystemExit, match="fail closed"):
        analyzer.derive_late_cell(tmp_path, "synth", [0])
    with pytest.raises(SystemExit, match="fail closed"):
        analyzer.derive_late_cell(tmp_path, "synth", [0, 1, 2, 3, 4])
    (tmp_path / "synth" / "nccl_interfaces.txt").unlink()
    with pytest.raises(SystemExit, match="fail closed"):
        analyzer.derive_late_cell(tmp_path, "synth", [0, 1])
    _synthetic_cell(tmp_path, "wrongchunk", [30, 40], chunk_bytes=4_194_304)
    with pytest.raises(SystemExit, match="fail closed"):
        analyzer.derive_late_cell(tmp_path, "wrongchunk", [0, 1])


def test_frozen_measured_inputs_still_derive_from_the_dataset():
    derived = analyzer.derive_x4_inputs(DATASET)
    assert derived["counts"] == [4014, 7704, 7682, 7053]
    assert derived["think_rate"] == FROZEN_THINK_RATE_PS
    assert derived["think_p50"] == FROZEN_THINK_P50_PS


def test_stdout_masking_is_stable():
    text = ("[ss-dragonfly manifest] a=1 topology=/some/where/x.topo b=2\n"
            "[ss-dragonfly manifest] p=20\n")
    other = ("[ss-dragonfly manifest] a=1 topology=/else/y.topo b=2\n"
             "[ss-dragonfly manifest] p=20\n")
    changed = ("[ss-dragonfly manifest] a=1 topology=/else/y.topo b=3\n"
               "[ss-dragonfly manifest] p=20\n")
    assert analyzer.masked_stdout_sha256(text) == analyzer.masked_stdout_sha256(other)
    assert analyzer.masked_stdout_sha256(text) != analyzer.masked_stdout_sha256(changed)


def test_descriptor_round_trips_into_a_runnable_cell():
    descriptor = analyzer.derive_late_cell(DATASET, "x4-incast", [0, 1, 2, 3])
    cell = runner.late_cell_to_cell(descriptor)
    assert cell["name"] == "late-x4-incast"
    assert cell["topo"] == "x4buf32"
    specs = [a for a in cell["args"] if str(a).startswith("src=")]
    assert [int(s.split("think_ps=")[1].split(",")[0]) for s in specs] == \
        FROZEN_THINK_RATE_PS


def test_dataset_manifest_pin_matches_the_freeze():
    import hashlib

    digest = hashlib.sha256((DATASET / "MANIFEST.json").read_bytes()).hexdigest()
    assert digest == analyzer.FROZEN_DATASET_MANIFEST_SHA256


def test_x4_stats_agree_with_the_frozen_counts():
    stats = json.loads((DATASET / "stats" / "x4-incast_stats.json").read_text())
    steady = stats["stages"][0]["steady_per_flow_gbps"]
    for flow, count in enumerate([4014, 7704, 7682, 7053]):
        assert steady[str(flow)] == pytest.approx(count * 8_388_608 / 20 / 1e9)


def test_series_reader_rejects_foreign_chunk_sizes(tmp_path):
    root = tmp_path / "ds"
    _synthetic_cell(root, "x4-incast", [10, 10, 10, 10], chunk_bytes=1)
    with pytest.raises(SystemExit, match="fail closed"):
        analyzer.derive_x4_inputs(root)


def _read_series_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_synthetic_series_shape_matches_the_real_schema(tmp_path):
    """The negative controls only bind if the synthetic schema is the real one."""
    _synthetic_cell(tmp_path, "synth", [3])
    synthetic = _read_series_rows(tmp_path / "synth" / "synth_flow0_dest.csv.gz")
    real = _read_series_rows(
        DATASET / "x4-incast" / "x4-incast_flow0_dest.csv.gz")[:1]
    assert set(synthetic[0]) == set(real[0])
