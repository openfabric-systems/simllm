#!/usr/bin/env python3
"""Score the TRAF-51 load-bearing recalibration against its freeze.

Consumes the bulk run tree written by run_cells.py plus the byte-locked
capture dataset, re-derives every frozen measured input from the tracked
bytes, evaluates the fatal guards and the frozen rows exactly as
expectations.md registers them, and writes the results summaries. Also
implements the frozen late-arrival mechanism: --derive-late-cell emits a
closed-loop descriptor for a shared-egress group of a landed dataset cell
by the frozen formulas (the x4 rate arm is the self-check and must
reproduce the frozen think table byte-exactly), and --score-late scores a
late run against the frozen BE-1 band with no code change.

    python analyze_recalibration.py --run-root DIR --out DIR
        [--dataset-root DIR]
    python analyze_recalibration.py --derive-late-cell CELL \
        --shared-flows 0,1,2,3 [--emit PATH] [--dataset-root DIR]
    python analyze_recalibration.py --score-late DESCRIPTOR --run-root DIR \
        --out DIR [--dataset-root DIR]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import statistics
from fractions import Fraction
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[1]
DEFAULT_DATASET = REPO_ROOT / "examples" / "merlin_fabric_flow_capture_v1" / "dataset"

# Frozen identities (expectations.md).
FROZEN_BINARY_SHA256 = "662416918457542c4fcab18aebd99f43f00cd80b5518ed78947a65e65620ce92"
FROZEN_PIN = "1dcbfec36a33753bf978cf6323bade1a6645fe4f"
FROZEN_DATASET_MANIFEST_SHA256 = (
    "67f898a04dd0e6787d4d50d6139f99f3c507963c6c937a9505434cf2d9dca002")
FROZEN_V1_TOPO_SHA256 = (
    "c0a0050b074385b7daa8b9fa83619dfd49ec1bcbf21fd82447b6029d6cff8f18")

# Archived wave-20 evaluation-of-record artifacts (reproduction targets).
ARCHIVED_DISC_A_BINS_SHA256 = (
    "af1d1e043685227e502411a7f9835290055f98bdda2157b1268d14d8917099f5")
ARCHIVED_CTRL_A_BINS_SHA256 = (
    "7eb66e67969286438bb25290e469a15dff5cc945a6866f17d907fdd64331cb4e")
ARCHIVED_CTRL_A_CHUNKS_SHA256 = (
    "1072cd8f86f589fdd21f15c1df284c8786e00dfcb96c27abda7e4cd75c6ebc96")
ARCHIVED_DISC_A_STDOUT_MASKED_SHA256 = (
    "4bdf0f070d1711f66e6598acea75a162961e90ab3f0ae24b0b9c195f6b7a7127")
ARCHIVED_CTRL_A_STDOUT_MASKED_SHA256 = (
    "4a7ce04689fa17e823265a892f3f8be4ad609e4f029c73550a623a3483e438ab")
ARCHIVED_DISC_A_MANIFEST = {
    "injected": 17980, "delivered": 14295, "delivered_payload_bytes": 127911660,
    "dropped": 3685, "first_drop_ps": 560255151, "per_flow_injected": [8990, 8990],
}
ARCHIVED_CTRL_A_MANIFEST = {
    "injected": 203546, "delivered": 203546, "dropped": 0,
    "chunks": [91, 126],
}

# Frozen instance constants.
SER_PS = 361_520
N_PKT = 938
PAYLOAD_B = 8_948
CHUNK_B = 8_388_608
F_PS = 340_417_280
BIN_BOUND_B = 2_487_544
CHUNK_WIRE_B = N_PKT * 9038
X4BUF32_CAPACITY_B = 33_554_432

# Frozen x4 measured inputs (re-derived and enforced by FG-6).
FROZEN_X4_COUNTS = [4014, 7704, 7682, 7053]
FROZEN_X4_P50_NS = [5_138_805, 2_123_632, 2_175_749, 2_134_978]
FROZEN_THINK_RATE_PS = [4_642_143_756, 2_255_636_718, 2_263_071_395, 2_495_255_483]
FROZEN_THINK_P50_PS = [4_798_387_720, 1_783_214_720, 1_835_331_720, 1_794_560_720]
MEASURED_X4_AGG = Fraction(sum(FROZEN_X4_COUNTS) * CHUNK_B, 20)  # bytes per second

# Frozen windows and bands.
WINDOW_LO_PS = 500_000_000_000
WINDOW_HI_PS = 6_000_000_000_000
WINDOW_S = Fraction(11, 2)
BAND_BE1 = (Fraction(90, 100), Fraction(1001, 1000))
BAND_BE2 = (Fraction(105, 100), Fraction(121, 100))
BAND_BE4 = (4_474_000_000, 4_476_000_000)
BAND_BE5 = (430, 442)
BAND_D1 = (3243, 3255)
FAULT_MESSAGE = "closed-loop cell dropped packets"

# Cell inventory: distinct destination ports (FG-3) and seam declarations
# (FG-7): per flow (think_ps or None, offered_bps or None, start_ps).
CELL_PORTS = {"ctrl-v1": 2, "ctrl-b32": 2, "ctrl-b64": 2, "mj2x-v1": 2,
              "x4p-v1": 1, "x4r-b32": 1, "x4r-b64": 1, "x4p50-b32": 1,
              "sat-v1": 1, "sat-b32": 1}
CTRL_SEAM = [(1_858_227_000, None, 0), (1_256_438_000, None, 0)]
X4_STARTS = [0, 400_000_000, 800_000_000, 1_200_000_000]
FROZEN_X4_OFFERED_BPS = [13_468_749_005, 25_850_334_413, 25_776_514_662, 23_665_940_890]
CELL_SEAM = {
    "ctrl-v1": CTRL_SEAM, "ctrl-b32": CTRL_SEAM, "ctrl-b64": CTRL_SEAM,
    "mj2x-v1": [(1_858_227_000, None, 0), (1_256_438_000, None, 100_000_000_000)],
    "x4p-v1": [(None, bps, 0) for bps in FROZEN_X4_OFFERED_BPS],
    "x4r-b32": list(zip(FROZEN_THINK_RATE_PS, [None] * 4, X4_STARTS)),
    "x4r-b64": list(zip(FROZEN_THINK_RATE_PS, [None] * 4, X4_STARTS)),
    "x4p50-b32": list(zip(FROZEN_THINK_P50_PS, [None] * 4, X4_STARTS)),
    "sat-v1": [(None, 130_000_000_000, 0), (None, 130_000_000_000, 278_092)],
    "sat-b32": [(None, 130_000_000_000, 0), (None, 130_000_000_000, 278_092)],
}
# mj2x consistency formulas (CN-2).
MJ2X_PERIODS = [F_PS + 1_858_227_000, F_PS + 1_256_438_000]
MJ2X_STARTS = [0, 100_000_000_000]
MJ2X_CHUNKS = [137, 126]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_stdout_sha256(text: str) -> str:
    tokens = ["topology=MASKED" if t.startswith("topology=") else t
              for t in text.split()]
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


def parse_stdout(text: str) -> dict[str, object]:
    """Parse the harness manifest lines into scalars plus per-flow records."""
    out: dict[str, object] = {"flow_records": []}
    for line in text.splitlines():
        if line.startswith(("[ss-dragonfly manifest]", "[ss-dragonfly load]")):
            for token in line.split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    out[key] = value
        elif line.startswith("[ss-dragonfly flow]"):
            flow: dict[str, str] = {}
            for token in line.split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    flow[key] = value
            out["flow_records"].append(flow)  # type: ignore[union-attr]
    return out


def read_bins(path: Path) -> list[dict[str, int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({k: int(v) for k, v in row.items()})
    return rows


def read_chunks(path: Path) -> dict[int, list[tuple[int, int, int]]]:
    """flow_id -> [(chunk_index, first_injection_ps, completion_ps)]."""
    flows: dict[int, list[tuple[int, int, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            flows.setdefault(int(row["flow_id"]), []).append(
                (int(row["chunk_index"]), int(row["first_injection_ps"]),
                 int(row["completion_ps"])))
    return flows


def derive_x4_inputs(dataset_root: Path) -> dict[str, object]:
    """Re-derive the frozen x4 measured inputs from the tracked series bytes."""
    counts: list[int] = []
    p50_ns: list[float] = []
    for flow in range(4):
        series = dataset_root / "x4-incast" / f"x4-incast_flow{flow}_dest.csv.gz"
        ts: list[int] = []
        with gzip.open(series, "rt", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["chunk_bytes"]) != CHUNK_B:
                    raise SystemExit(f"fail closed: {series} chunk size is not 8 MiB")
                ts.append(int(row["t_ns_since_t0"]))
        ts.sort()
        counts.append(sum(1 for t in ts if 160_000_000_000 <= t < 180_000_000_000))
        p50_ns.append(statistics.median(b - a for a, b in itertools.pairwise(ts)))
    think_rate = [round(Fraction(20 * 10**12, n)) - F_PS for n in counts]
    think_p50 = [int(p) * 1000 - F_PS for p in p50_ns]
    return {"counts": counts, "p50_ns": p50_ns,
            "think_rate": think_rate, "think_p50": think_p50}


def verify_dataset(dataset_root: Path, fatal: list[str]) -> None:
    # The freeze's claim is the CONSUMED manifest version, not the living
    # dataset tip: the capture study's committed fold-in protocol advances
    # MANIFEST.json, preserving each superseded version byte-exact and
    # self-verifying under dataset/manifest_versions/<its sha256>.json.
    # Accept the living manifest only while it still IS the frozen version;
    # otherwise require the preserved snapshot and fail closed on absence
    # or self-hash mismatch. The guard's recorded basis is the frozen
    # digest either way (post-published amendment, disclosed in RESULTS
    # correction 10; published outputs proven byte-identical).
    manifest_path = dataset_root / "MANIFEST.json"
    if sha256_file(manifest_path) != FROZEN_DATASET_MANIFEST_SHA256:
        snapshot = (dataset_root / "manifest_versions"
                    / f"{FROZEN_DATASET_MANIFEST_SHA256}.json")
        if not snapshot.is_file():
            fatal.append("FG-1: frozen dataset manifest version absent "
                         "(neither the living manifest nor a preserved "
                         "snapshot matches the freeze)")
            return
        if sha256_file(snapshot) != FROZEN_DATASET_MANIFEST_SHA256:
            fatal.append("FG-1: manifest snapshot fails its self-hash")
            return
        manifest_path = snapshot
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    consumed = ["stats/x4-incast_stats.json"] + [
        f"x4-incast/x4-incast_flow{i}_dest.csv.gz" for i in range(4)]
    for rel in consumed:
        meta = manifest.get(rel)
        if meta is None:
            fatal.append(f"FG-1: {rel} not in dataset manifest")
            continue
        data = (dataset_root / rel).read_bytes()
        if len(data) != meta["bytes"] or \
                hashlib.sha256(data).hexdigest() != meta["sha256"]:
            fatal.append(f"FG-1: dataset file {rel} does not verify")


def check_frozen_inputs(dataset_root: Path, fatal: list[str]) -> None:
    derived = derive_x4_inputs(dataset_root)
    if derived["counts"] != FROZEN_X4_COUNTS:
        fatal.append(f"FG-6: window counts {derived['counts']} != frozen")
    if [float(v) for v in derived["p50_ns"]] != [float(v) for v in FROZEN_X4_P50_NS]:
        fatal.append(f"FG-6: series p50s {derived['p50_ns']} != frozen")
    if derived["think_rate"] != FROZEN_THINK_RATE_PS:
        fatal.append("FG-6: rate-arm think table does not recompute")
    if derived["think_p50"] != FROZEN_THINK_P50_PS:
        fatal.append("FG-6: p50-arm think table does not recompute")
    stats = json.loads(
        (dataset_root / "stats" / "x4-incast_stats.json").read_text(encoding="utf-8"))
    steady = stats["stages"][0]["steady_per_flow_gbps"]
    for flow, n in enumerate(FROZEN_X4_COUNTS):
        expected = Fraction(n * CHUNK_B, 20) / 10**9
        if abs(float(expected) - steady[str(flow)]) > 1e-9:
            fatal.append(f"FG-6: stats steady rate flow {flow} mismatch")


class Cell:
    def __init__(self, run_root: Path, name: str, record: dict[str, object]):
        self.name = name
        self.record = record
        self.paths = {
            rep: {
                "csv": run_root / f"{name}_{rep}.csv",
                "chunks": run_root / f"{name}_{rep}.chunks.csv",
                "stdout": run_root / f"{name}_{rep}.stdout",
                "stderr": run_root / f"{name}_{rep}.stderr",
            } for rep in ("r1", "r2")
        }
        self.exit = (record["exit_r1"], record["exit_r2"])
        self.stdout = self.paths["r1"]["stdout"].read_text(encoding="utf-8")
        self.stderr = self.paths["r1"]["stderr"].read_text(encoding="utf-8")
        self.manifest = parse_stdout(self.stdout)

    def repeats_identical(self) -> bool:
        for kind in ("csv", "chunks", "stdout", "stderr"):
            a, b = self.paths["r1"][kind], self.paths["r2"][kind]
            if a.exists() != b.exists():
                return False
            if a.exists() and a.read_bytes() != b.read_bytes():
                return False
        return self.exit[0] == self.exit[1]


def check_cell_guards(cell: Cell, fatal: list[str]) -> None:
    name = cell.name
    if not cell.repeats_identical():
        fatal.append(f"FG-2: {name} repeats differ")
    expected_exit = int(cell.record["expected_exit"])  # type: ignore[arg-type]
    if name == "x4r-v1":
        pass  # BE-3 scores the outcome; FG-4 exempts this cell by the freeze.
    elif cell.exit[0] != 0:
        fatal.append(f"FG-4: {name} exit {cell.exit[0]} (expected {expected_exit})")
    if cell.exit[0] != 0:
        return
    manifest = cell.manifest
    injected = int(manifest["injected"])  # type: ignore[arg-type]
    delivered = int(manifest["delivered"])  # type: ignore[arg-type]
    dropped = int(manifest["dropped"])  # type: ignore[arg-type]
    payload = int(manifest["delivered_payload_bytes"])  # type: ignore[arg-type]
    if injected != delivered + dropped:
        fatal.append(f"FG-3: {name} injected != delivered + dropped")
    if payload != delivered * PAYLOAD_B:
        fatal.append(f"FG-3: {name} payload arithmetic")
    bins = read_bins(cell.paths["r1"]["csv"])
    per_bin: dict[int, int] = {}
    for row in bins:
        if row["delivered_payload_bytes"] > BIN_BOUND_B:
            fatal.append(f"FG-3: {name} per-flow bin over quantization bound")
            break
        per_bin[row["bin_start_ps"]] = (
            per_bin.get(row["bin_start_ps"], 0) + row["delivered_payload_bytes"])
    bound = BIN_BOUND_B * CELL_PORTS.get(name, 1)
    if any(total > bound for total in per_bin.values()):
        fatal.append(f"FG-3: {name} aggregate bin over {bound}")
    seam = CELL_SEAM.get(name)
    if seam is not None:
        flows = cell.manifest["flow_records"]  # type: ignore[index]
        for index, (think, offered, start) in enumerate(seam):
            flow = flows[index]  # type: ignore[index]
            want_think = "na" if think is None else str(think)
            want_offered = "line" if offered is None else str(offered)
            if flow.get("think_ps") != want_think or \
                    flow.get("offered_bps") != want_offered or \
                    flow.get("start_ps") != str(start):
                fatal.append(f"FG-7: {name} flow {index} seam echo mismatch")
    if cell.paths["r1"]["chunks"].exists():
        chunks = read_chunks(cell.paths["r1"]["chunks"])
        flows = cell.manifest["flow_records"]  # type: ignore[index]
        for flow_record in flows:  # type: ignore[union-attr]
            flow_id = int(flow_record["flow"])
            declared = int(flow_record["chunks_completed"])
            have = chunks.get(flow_id, [])
            if len(have) != declared:
                fatal.append(f"FG-5: {name} flow {flow_id} chunk rows != manifest")
            comps = [c for (_, _, c) in have]
            if comps != sorted(comps) or len(set(comps)) != len(comps):
                fatal.append(f"FG-5: {name} flow {flow_id} completions not increasing")
        if name in ("x4r-b32", "x4p50-b32"):
            for flow_id, rows in chunks.items():
                in_window = sum(
                    1 for (_, _, c) in rows if WINDOW_LO_PS <= c < WINDOW_HI_PS)
                if in_window < 500:
                    fatal.append(
                        f"FG-5: {name} flow {flow_id} only {in_window} in window")


def window_aggregate(cell: Cell) -> tuple[Fraction, list[int]]:
    chunks = read_chunks(cell.paths["r1"]["chunks"])
    per_flow = [
        sum(1 for (_, _, c) in chunks.get(flow, [])
            if WINDOW_LO_PS <= c < WINDOW_HI_PS)
        for flow in range(4)
    ]
    aggregate = Fraction(sum(per_flow) * CHUNK_B) / WINDOW_S
    return aggregate, per_flow


def row(rows: list[dict[str, object]], row_id: str, klass: str, verdict: bool,
        detail: str) -> None:
    rows.append({"id": row_id, "class": klass,
                 "verdict": "PASS" if verdict else "FAIL", "detail": detail})
    print(f"{row_id} [{klass}]: {'PASS' if verdict else 'FAIL'} {detail}")


def note(records: list[dict[str, object]], row_id: str, ok: bool, detail: str) -> None:
    records.append({"id": row_id, "holds": bool(ok), "detail": detail})
    print(f"{row_id} [consistency, unscored]: {'holds' if ok else 'VIOLATED'} {detail}")


def check_run_identity(manifest: dict[str, object], fatal: list[str]) -> None:
    """FG-1's run-identity half. Absence of a field fails closed: a manifest
    that never recorded the pin cannot prove it was the run of record
    (review finding F5; the run of record carries both fields)."""
    if manifest.get("binary_sha256") != FROZEN_BINARY_SHA256:
        fatal.append("FG-1: binary sha missing or mismatched")
    if manifest.get("submodule_head") != FROZEN_PIN:
        fatal.append("FG-1: submodule pin missing or mismatched")


def evaluate(run_root: Path, dataset_root: Path, out_dir: Path) -> int:
    fatal: list[str] = []
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    check_run_identity(manifest, fatal)
    tracked = {
        "v1": REPO_ROOT / "examples" / "merlin_ss_fabric_calibration_v1" /
        "merlin_a100_singleswitch_v1.topo",
        "x4buf32": STUDY_DIR / "merlin_a100_singleswitch_v1_x4buf32.topo",
        "x4buf64": STUDY_DIR / "merlin_a100_singleswitch_v1_x4buf64.topo",
    }
    for key, path in tracked.items():
        digest = sha256_file(path)
        if manifest["topologies"].get(key) != digest:
            fatal.append(f"FG-1: topology {key} run hash != tracked bytes")
    if sha256_file(tracked["v1"]) != FROZEN_V1_TOPO_SHA256:
        fatal.append("FG-1: v1 instance drifted from the freeze")
    verify_dataset(dataset_root, fatal)
    check_frozen_inputs(dataset_root, fatal)

    cells = {name: Cell(run_root, name, record)
             for name, record in manifest["cells"].items()}
    for cell in cells.values():
        check_cell_guards(cell, fatal)

    rows: list[dict[str, object]] = []
    consistency: list[dict[str, object]] = []
    derived: list[dict[str, object]] = []
    observed: dict[str, object] = {}

    # EX-1: sat-v1 reproduces the wave-20 discriminate-A record.
    sat_v1 = cells["sat-v1"]
    ex1_bins = sha256_file(sat_v1.paths["r1"]["csv"])
    ex1_masked = masked_stdout_sha256(sat_v1.stdout)
    ex1_ok = (ex1_bins == ARCHIVED_DISC_A_BINS_SHA256 and
              ex1_masked == ARCHIVED_DISC_A_STDOUT_MASKED_SHA256)
    ints = {key: int(sat_v1.manifest.get(key, -1))  # type: ignore[arg-type]
            for key in ("injected", "delivered", "delivered_payload_bytes",
                        "dropped", "first_drop_ps")}
    per_flow_injected = [int(f["injected_packets"])
                        for f in sat_v1.manifest["flow_records"]]  # type: ignore[index]
    ex1_ok = ex1_ok and all(
        ints[k] == ARCHIVED_DISC_A_MANIFEST[k] for k in ints) and \
        per_flow_injected == ARCHIVED_DISC_A_MANIFEST["per_flow_injected"]
    row(rows, "EX-1", "exact", ex1_ok,
        f"bins {ex1_bins[:8]}.. masked {ex1_masked[:8]}.. manifest {ints}")
    observed["sat_v1"] = ints

    # EX-2: ctrl-v1 reproduces the wave-20 control-A record.
    ctrl_v1 = cells["ctrl-v1"]
    ex2_bins = sha256_file(ctrl_v1.paths["r1"]["csv"])
    ex2_chunks = sha256_file(ctrl_v1.paths["r1"]["chunks"])
    ex2_masked = masked_stdout_sha256(ctrl_v1.stdout)
    ex2_ok = (ex2_bins == ARCHIVED_CTRL_A_BINS_SHA256 and
              ex2_chunks == ARCHIVED_CTRL_A_CHUNKS_SHA256 and
              ex2_masked == ARCHIVED_CTRL_A_STDOUT_MASKED_SHA256)
    row(rows, "EX-2", "exact", ex2_ok,
        f"bins {ex2_bins[:8]}.. chunks {ex2_chunks[:8]}.. masked {ex2_masked[:8]}..")

    # BE-1: the load-bearing x4 aggregate, rate arm.
    agg_rate, per_flow_rate = window_aggregate(cells["x4r-b32"])
    ratio_rate = agg_rate / MEASURED_X4_AGG
    be1_ok = BAND_BE1[0] <= ratio_rate <= BAND_BE1[1]
    row(rows, "BE-1", "behavioral", be1_ok,
        f"aggregate {float(agg_rate) / 1e9:.6f} GB/s ratio {float(ratio_rate):.5f} "
        f"band [0.90, 1.001] fluid point 0.954; per-flow {per_flow_rate}")
    observed["x4_rate"] = {"per_flow_chunks": per_flow_rate,
                           "aggregate_bps": float(agg_rate),
                           "ratio": float(ratio_rate)}

    # BE-2: the skew-diagnosis arm.
    agg_p50, per_flow_p50 = window_aggregate(cells["x4p50-b32"])
    ratio_p50 = agg_p50 / MEASURED_X4_AGG
    be2_ok = BAND_BE2[0] <= ratio_p50 <= BAND_BE2[1]
    row(rows, "BE-2", "behavioral", be2_ok,
        f"aggregate {float(agg_p50) / 1e9:.6f} GB/s ratio {float(ratio_p50):.5f} "
        f"band [1.05, 1.21] fluid point 1.127; per-flow {per_flow_p50}")
    observed["x4_p50"] = {"per_flow_chunks": per_flow_p50,
                          "aggregate_bps": float(agg_p50),
                          "ratio": float(ratio_p50)}

    # BE-3: the composed-level discrimination row.
    fault = cells["x4r-v1"]
    be3_ok = fault.exit == (2, 2) and FAULT_MESSAGE in fault.stderr
    row(rows, "BE-3", "behavioral", be3_ok,
        f"x4r-v1 exits {fault.exit}, registered message "
        f"{'present' if FAULT_MESSAGE in fault.stderr else 'ABSENT'}")

    # BE-4 and BE-5: sat-b32 buffer-shifted drop timing.
    sat_b32 = cells["sat-b32"]
    first_drop = int(sat_b32.manifest.get("first_drop_ps", -1))  # type: ignore[arg-type]
    dropped_b32 = int(sat_b32.manifest.get("dropped", -1))  # type: ignore[arg-type]
    be4_ok = BAND_BE4[0] <= first_drop <= BAND_BE4[1]
    row(rows, "BE-4", "behavioral", be4_ok,
        f"first_drop_ps {first_drop} band {BAND_BE4} point 4474938884")
    be5_ok = BAND_BE5[0] <= dropped_b32 <= BAND_BE5[1]
    row(rows, "BE-5", "behavioral", be5_ok,
        f"dropped {dropped_b32} band {BAND_BE5} point 436")
    observed["sat_b32"] = {"first_drop_ps": first_drop, "dropped": dropped_b32}

    # ST-1: buffer insensitivity above the closed-loop occupancy bound.
    b32, b64 = cells["x4r-b32"], cells["x4r-b64"]
    st1_ok = (b32.paths["r1"]["csv"].read_bytes() == b64.paths["r1"]["csv"].read_bytes()
              and b32.paths["r1"]["chunks"].read_bytes() ==
              b64.paths["r1"]["chunks"].read_bytes()
              and masked_stdout_sha256(b32.stdout) == masked_stdout_sha256(b64.stdout))
    row(rows, "ST-1", "structural", st1_ok,
        "x4r-b32 and x4r-b64 byte-identical (bins, chunks; stdout masked)")

    # Consistency-only rows, recorded and unscored.
    ctrl_ident = all(
        cells[name].paths["r1"]["csv"].read_bytes() ==
        ctrl_v1.paths["r1"]["csv"].read_bytes() and
        cells[name].paths["r1"]["chunks"].read_bytes() ==
        ctrl_v1.paths["r1"]["chunks"].read_bytes() and
        masked_stdout_sha256(cells[name].stdout) == ex2_masked
        for name in ("ctrl-b32", "ctrl-b64"))
    note(consistency, "CN-1", ctrl_ident,
         "control cell byte-identical across v1, x4buf32, x4buf64")

    mj2x = cells["mj2x-v1"]
    mj2x_chunks = read_chunks(mj2x.paths["r1"]["chunks"])
    cn2_ok = True
    for flow in range(2):
        want = [MJ2X_STARTS[flow] + F_PS + c * MJ2X_PERIODS[flow]
                for c in range(MJ2X_CHUNKS[flow])]
        have = [c for (_, _, c) in mj2x_chunks.get(flow, [])]
        cn2_ok = cn2_ok and have == want
    cn2_ok = cn2_ok and int(mj2x.manifest["injected"]) == 246_694  # type: ignore[arg-type]
    note(consistency, "CN-2", cn2_ok,
         "staggered-join mirror cadences equal the solo formulas "
         f"(chunks {MJ2X_CHUNKS}, injected 246694)")

    x4p = cells["x4p-v1"]
    x4p_bins = read_bins(x4p.paths["r1"]["csv"])
    per_flow_payload: dict[int, int] = {}
    for entry in x4p_bins:
        per_flow_payload[entry["flow_id"]] = (
            per_flow_payload.get(entry["flow_id"], 0) + entry["delivered_payload_bytes"])
    x4p_flows = x4p.manifest["flow_records"]  # type: ignore[index]
    cn3_ok = int(x4p.manifest["dropped"]) == 0 and all(  # type: ignore[arg-type]
        per_flow_payload.get(int(f["flow"]), -1) ==
        int(f["injected_packets"]) * PAYLOAD_B for f in x4p_flows)
    note(consistency, "CN-3", cn3_ok,
         "paced mirror delivers exactly what it injects at the measured rates")

    # Derived rows, reported and never scored.
    dropped_v1 = ints["dropped"]
    d1 = dropped_v1 - dropped_b32
    derived.append({"id": "D-1", "value": d1,
                    "band": list(BAND_D1), "inside": BAND_D1[0] <= d1 <= BAND_D1[1]})
    derived.append({"id": "D-2", "value": f"{ints['first_drop_ps']} < {first_drop}",
                    "holds": ints["first_drop_ps"] < first_drop})
    derived.append({"id": "D-3", "fluid_residual_rate": float(ratio_rate) - 0.95387,
                    "fluid_residual_p50": float(ratio_p50) - 1.12680})
    rates = [Fraction(n * CHUNK_B) / WINDOW_S for n in per_flow_rate]
    total = sum(rates)
    jain = float((total * total) / (4 * sum(r * r for r in rates))) if total else 0.0
    derived.append({"id": "D-4", "sim_jain": jain,
                    "per_flow_gbps": [float(r) / 1e9 for r in rates]})
    for entry in derived:
        print(f"{entry['id']} [derived, unscored]: "
              f"{ {k: v for k, v in entry.items() if k != 'id'} }")

    void = bool(fatal)
    summary = {
        "study": "merlin_ss_fabric_loadbearing_v1",
        "binary_sha256": manifest["binary_sha256"],
        "submodule_head": manifest.get("submodule_head"),
        "dataset_manifest_sha256": FROZEN_DATASET_MANIFEST_SHA256,
        "void": void,
        "fatal_failures": fatal,
        "rows": rows,
        "consistency": consistency,
        "derived": derived,
        "observed": observed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recalibration_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    run_manifest = json.loads(
        (run_root / "run_manifest.json").read_text(encoding="utf-8"))
    (out_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'recalibration_summary.json'}")
    if fatal:
        for failure in fatal:
            print(f"FATAL {failure}")
        return 1
    scored_fail = [r for r in rows if r["verdict"] == "FAIL"]
    print(f"scored: {len(rows) - len(scored_fail)} PASS, {len(scored_fail)} FAIL, "
          f"guards clean")
    return 0


def verify_shared_egress_evidence(dataset_root: Path, cell: str,
                                  group_payload_bytes: int) -> dict[str, str]:
    """Machine check of the freeze's shared-egress mapping rule (review F6).

    The frozen rule classifies a group as shared-egress when its flows share
    one recorded sender hsn port or one destination hsn device. The recorded
    counters can prove that directly when the cell's evidence has per-node
    hsn tx and rx deltas: a single sender interface carrying at least 90
    percent of its node's hsn tx and at least 99 percent of the group's
    payload proves a shared sender port, and a single receiver interface
    carrying at least 90 percent of its node's hsn rx while that rx total is
    at least 1 percent of the payload (the cxi_ss1 rx counter undercounts
    bulk receive by roughly 10x) proves a shared destination device. When
    the evidence is readable and refutes both, the derivation fails closed;
    when its shape cannot be evaluated (missing or empty per-node counters),
    the descriptor says so rather than silently trusting the operator's
    --shared-flows input, and RESULTS must repeat that label.
    """
    path = dataset_root / cell / "port_tx_deltas.json"
    try:
        deltas = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "not-machine-checkable",
                "detail": "port_tx_deltas.json unreadable"}
    nodes: dict[str, dict[str, dict[str, int]]] = {}
    for node, interfaces in deltas.items():
        if not isinstance(interfaces, dict):
            continue
        hsn = {name: counters for name, counters in interfaces.items()
               if name.startswith("hsn") and isinstance(counters, dict)}
        if hsn:
            nodes[node] = hsn
    if len(nodes) < 2:
        return {"status": "not-machine-checkable",
                "detail": "fewer than two nodes carry hsn counters"}
    sender = None
    receiver = None
    for node, hsn in nodes.items():
        tx = {name: int(c.get("tx_delta_bytes", 0)) for name, c in hsn.items()}
        rx = {name: int(c.get("rx_delta_bytes", 0)) for name, c in hsn.items()}
        tx_total, rx_total = sum(tx.values()), sum(rx.values())
        if tx_total > 0:
            top_name, top = max(tx.items(), key=lambda item: item[1])
            if top * 10 >= 9 * tx_total and top * 100 >= 99 * group_payload_bytes:
                sender = (node, top_name)
        if rx_total > 0:
            top_name, top = max(rx.items(), key=lambda item: item[1])
            if top * 10 >= 9 * rx_total and rx_total * 100 >= group_payload_bytes:
                receiver = (node, top_name)
    if sender and receiver and sender[0] != receiver[0]:
        return {"status": "machine-verified-both",
                "detail": f"sender {sender[0]}/{sender[1]}, "
                          f"destination {receiver[0]}/{receiver[1]}"}
    if sender:
        return {"status": "machine-verified-shared-sender-port",
                "detail": f"sender {sender[0]}/{sender[1]}"}
    if receiver:
        return {"status": "machine-verified-shared-destination-device",
                "detail": f"destination {receiver[0]}/{receiver[1]}"}
    raise SystemExit(
        f"fail closed: {cell} port evidence shows no shared sender port and "
        "no shared destination device for the group")


def derive_late_cell(dataset_root: Path, cell: str,
                     shared_flows: list[int]) -> dict[str, object]:
    """Emit a frozen-formula closed-loop descriptor for a shared-egress group.

    Frozen shape: one model source host per stack starting at host 4
    (gpu102's ports on the declared instance; host identity is immaterial at
    the single-switch shape), the shared model destination host 19, starts
    staggered 400 us apart, duration 6e12 ps on the x4buf32 instance,
    think_i = round(20e12 / n_i) - F from the group's final-20-second
    completion counts. Fails closed on missing evidence, a foreign chunk
    size, an occupancy bound above the x4buf32 capacity, or port evidence
    that refutes the shared-egress classification; where the evidence shape
    cannot be machine-checked the descriptor's mapping_check says so.
    """
    if len(shared_flows) < 2:
        raise SystemExit("fail closed: a shared-egress group needs >= 2 flows")
    for evidence in ("nccl_interfaces.txt", "port_tx_deltas.json"):
        if not (dataset_root / cell / evidence).exists():
            raise SystemExit(f"fail closed: {cell} lacks {evidence}")
    if (len(shared_flows) - 1) * CHUNK_WIRE_B >= X4BUF32_CAPACITY_B:
        raise SystemExit("fail closed: occupancy bound exceeds the x4buf32 capacity")
    flows = []
    group_chunks_total = 0
    for index, flow in enumerate(shared_flows):
        series = dataset_root / cell / f"{cell}_flow{flow}_dest.csv.gz"
        if not series.exists():
            raise SystemExit(f"fail closed: {series} missing")
        ts: list[int] = []
        with gzip.open(series, "rt", encoding="utf-8") as handle:
            for entry in csv.DictReader(handle):
                if int(entry["chunk_bytes"]) != CHUNK_B:
                    raise SystemExit(f"fail closed: {cell} chunk size is not 8 MiB")
                ts.append(int(entry["t_ns_since_t0"]))
        count = sum(1 for t in ts if 160_000_000_000 <= t < 180_000_000_000)
        if count == 0:
            raise SystemExit(f"fail closed: {cell} flow {flow} empty window")
        group_chunks_total += len(ts)
        flows.append({
            "src": 4 + index, "dst": 19,
            "think_ps": int(round(Fraction(20 * 10**12, count)) - F_PS),
            "start_ps": 400_000_000 * index,
            "measured_count": count,
        })
    mapping_check = verify_shared_egress_evidence(
        dataset_root, cell, group_chunks_total * CHUNK_B)
    return {
        "cell": f"late-{cell}", "instance": "x4buf32",
        "chunk_payload_bytes": CHUNK_B, "duration_ps": 6_000_000_000_000,
        "window_ps": [WINDOW_LO_PS, WINDOW_HI_PS],
        "band": [str(BAND_BE1[0]), str(BAND_BE1[1])],
        "flows": flows,
        "mapping_check": mapping_check,
        "measured_group_aggregate_bps":
            float(Fraction(sum(f["measured_count"] for f in flows) * CHUNK_B, 20)),
    }


def late_fatal_guards(run_root: Path, descriptor: dict[str, object]) -> list[str]:
    """The applicable fatal-guard set for a late cell (review F5): run
    identity, repeat determinism, clean execution, conservation, seam echo
    and the completion floor, evaluated before any band verdict, because
    BE-1's meaning rests on exactly these preconditions."""
    fatal: list[str] = []
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        return ["FG-1: late run manifest missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    check_run_identity(manifest, fatal)
    name = str(descriptor["cell"])
    record = manifest.get("cells", {}).get(name)
    if record is None:
        fatal.append(f"FG-1: late run manifest lacks cell {name}")
        return fatal
    cell = Cell(run_root, name, record)
    if not cell.repeats_identical():
        fatal.append(f"FG-2: {name} repeats differ")
    if cell.exit[0] != 0:
        fatal.append(f"FG-4: {name} exit {cell.exit[0]}")
        return fatal
    m = cell.manifest
    injected = int(m["injected"])  # type: ignore[arg-type]
    delivered = int(m["delivered"])  # type: ignore[arg-type]
    dropped = int(m["dropped"])  # type: ignore[arg-type]
    payload = int(m["delivered_payload_bytes"])  # type: ignore[arg-type]
    if dropped != 0:
        fatal.append(f"FG-4: {name} closed-loop cell dropped packets")
    if injected != delivered + dropped:
        fatal.append(f"FG-3: {name} injected != delivered + dropped")
    if payload != delivered * PAYLOAD_B:
        fatal.append(f"FG-3: {name} payload arithmetic")
    per_bin: dict[int, int] = {}
    for entry in read_bins(cell.paths["r1"]["csv"]):
        if entry["delivered_payload_bytes"] > BIN_BOUND_B:
            fatal.append(f"FG-3: {name} per-flow bin over quantization bound")
            break
        per_bin[entry["bin_start_ps"]] = (
            per_bin.get(entry["bin_start_ps"], 0) + entry["delivered_payload_bytes"])
    if any(total > BIN_BOUND_B for total in per_bin.values()):
        fatal.append(f"FG-3: {name} aggregate bin over the single-port bound")
    records = m["flow_records"]  # type: ignore[index]
    declared = descriptor["flows"]  # type: ignore[index]
    if len(records) != len(declared):  # type: ignore[arg-type]
        fatal.append(f"FG-7: {name} flow count mismatch")
        return fatal
    chunks = read_chunks(cell.paths["r1"]["chunks"])
    for index, want in enumerate(declared):  # type: ignore[arg-type]
        have = records[index]  # type: ignore[index]
        if have.get("think_ps") != str(want["think_ps"]) or \
                have.get("start_ps") != str(want["start_ps"]) or \
                have.get("offered_bps") != "line":
            fatal.append(f"FG-7: {name} flow {index} seam echo mismatch")
        rows = chunks.get(index, [])
        if len(rows) != int(have.get("chunks_completed", -1)):
            fatal.append(f"FG-5: {name} flow {index} chunk rows != manifest")
        comps = [c for (_, _, c) in rows]
        if comps != sorted(comps) or len(set(comps)) != len(comps):
            fatal.append(f"FG-5: {name} flow {index} completions not increasing")
        in_window = sum(1 for c in comps if WINDOW_LO_PS <= c < WINDOW_HI_PS)
        if in_window < 500:
            fatal.append(f"FG-5: {name} flow {index} only {in_window} in window")
    return fatal


def score_late(descriptor_path: Path, run_root: Path, out_dir: Path) -> int:
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    name = descriptor["cell"]
    fatal = late_fatal_guards(run_root, descriptor)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"late_{name}_score.json"
    if fatal:
        result: dict[str, object] = {
            "cell": name, "verdict": "VOID", "fatal_failures": fatal,
            "mapping_check": descriptor.get("mapping_check"),
        }
        out_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(json.dumps(result, indent=1, sort_keys=True))
        return 1
    chunks = read_chunks(run_root / f"{name}_r1.chunks.csv")
    per_flow = [
        sum(1 for (_, _, c) in chunks.get(i, [])
            if WINDOW_LO_PS <= c < WINDOW_HI_PS)
        for i in range(len(descriptor["flows"]))
    ]
    aggregate = Fraction(sum(per_flow) * CHUNK_B) / WINDOW_S
    measured = Fraction(
        sum(f["measured_count"] for f in descriptor["flows"]) * CHUNK_B, 20)
    ratio = aggregate / measured
    ok = BAND_BE1[0] <= ratio <= BAND_BE1[1]
    result = {
        "cell": name, "per_flow_chunks": per_flow,
        "aggregate_bps": float(aggregate), "ratio": float(ratio),
        "band": [str(BAND_BE1[0]), str(BAND_BE1[1])],
        "fatal_failures": [],
        "mapping_check": descriptor.get("mapping_check"),
        "verdict": "PASS" if ok else "FAIL",
    }
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", default="")
    parser.add_argument("--derive-late-cell", default="")
    parser.add_argument("--shared-flows", default="")
    parser.add_argument("--emit", default="")
    parser.add_argument("--score-late", default="")
    options = parser.parse_args()
    dataset_root = Path(options.dataset_root)

    if options.derive_late_cell:
        if not options.shared_flows:
            parser.error("--derive-late-cell requires --shared-flows")
        flows = [int(v) for v in options.shared_flows.split(",")]
        descriptor = derive_late_cell(dataset_root, options.derive_late_cell, flows)
        text = json.dumps(descriptor, indent=1, sort_keys=True) + "\n"
        if options.emit:
            Path(options.emit).write_text(text, encoding="utf-8")
            print(f"wrote {options.emit}")
        else:
            print(text, end="")
        return 0
    if options.score_late:
        if not options.run_root or not options.out:
            parser.error("--score-late requires --run-root and --out")
        return score_late(Path(options.score_late), Path(options.run_root),
                          Path(options.out))
    if not options.run_root or not options.out:
        parser.error("--run-root and --out are required")
    return evaluate(Path(options.run_root), dataset_root, Path(options.out))


if __name__ == "__main__":
    raise SystemExit(main())
