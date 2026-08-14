"""Qualify the tensor-parallel allreduce site inventory (TRAF-33).

Pure lowering arithmetic: no backend binary, no network model and no timing.
The study renders one decode step through three independent consumers of
``step_tp_allreduces`` and compares each against the closed forms frozen in
expectations.md before the behavior was implemented.

Run it from the repository root:

    python examples/moe_tp_sites_v1/run_study.py --out <run directory>

Without ``--out`` the raw rows go to ``$SIMLLM_DATA_ROOT/moe_tp_sites_v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from simllm.backends.step_lowerer import SerialStepLowerer, SerialStepLowererConfig
from simllm.compute import ModelDims
from simllm.core import CollectiveWork, RequestPhase, ScheduledRequest, StepRecord
from simllm.traffic import (
    EXPERT_PARALLEL_TP_ALLREDUCE_SITES,
    TP_ALLREDUCE_SITES,
    layer_tp_allreduce_sites,
    render_step_goal,
    step_communication_phases,
    step_moe_alltoalls,
    step_tp_allreduces,
)

BASE_TAG = 1000
HIDDEN_SIZE = 4096
DTYPE_BYTES = 2
EP_RANKS = tuple(range(8))

TP_WIDTHS = (1, 2, 8)
LAYER_COUNTS = (1, 4, 24)
TOKEN_COUNTS = (3, 12)

_BASE = ModelDims(
    num_layers=1,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=1024,
    num_heads=32,
    num_kv_heads=8,
    head_size=128,
    vocab_size=32000,
    dtype_bytes=DTYPE_BYTES,
)
_ROUTED = replace(
    _BASE,
    num_experts=32,
    top_k=2,
    moe_intermediate_size=512,
)

#: kind -> (dims template, expert-parallel group, frozen sites per layer)
KINDS: dict[str, tuple[ModelDims, tuple[int, ...] | None, int]] = {
    "dense": (_BASE, None, 2),
    "routed-tp": (replace(_ROUTED, local_num_experts=32), None, 2),
    "routed-ep": (replace(_ROUTED, local_num_experts=4), EP_RANKS, 1),
}


@dataclass(frozen=True)
class Cell:
    kind: str
    tp_width: int
    num_layers: int
    tokens: int

    @property
    def dims(self) -> ModelDims:
        template, _, _ = KINDS[self.kind]
        return replace(template, num_layers=self.num_layers)

    @property
    def ep_ranks(self) -> tuple[int, ...] | None:
        return KINDS[self.kind][1]

    @property
    def sites_per_layer(self) -> int:
        return KINDS[self.kind][2]

    @property
    def tp_ranks(self) -> tuple[int, ...]:
        return tuple(range(self.tp_width))

    @property
    def record(self) -> StepRecord:
        return StepRecord(
            step_index=0,
            virtual_time_ps=0,
            scheduled=[
                ScheduledRequest(
                    "r0",
                    RequestPhase.DECODE,
                    num_new_tokens=self.tokens,
                    context_length=64 + self.tokens,
                )
            ],
        )

    @property
    def payload_bytes(self) -> int:
        return self.tokens * HIDDEN_SIZE * DTYPE_BYTES

    @property
    def n_sites(self) -> int:
        return self.num_layers * self.sites_per_layer

    @property
    def tag_stride(self) -> int:
        return 2 * (self.tp_width - 1)

    @property
    def chunk_bytes(self) -> int:
        return max(1, self.payload_bytes // self.tp_width)

    @property
    def tp_messages(self) -> int:
        if self.tp_width < 2:
            return 0
        return self.n_sites * self.tag_stride * self.tp_width

    @property
    def tp_bytes(self) -> int:
        return self.tp_messages * self.chunk_bytes

    @property
    def moe_operations(self) -> int:
        return 2 * self.num_layers if self.ep_ranks is not None else 0

    @property
    def label(self) -> str:
        return f"{self.kind} W={self.tp_width} L={self.num_layers} T={self.tokens}"


def cells() -> list[Cell]:
    return [
        Cell(kind, width, layers, tokens)
        for kind in KINDS
        for width in TP_WIDTHS
        for layers in LAYER_COUNTS
        for tokens in TOKEN_COUNTS
    ]


class Ledger:
    """Separate registers for scored instances and fatal guards."""

    def __init__(self) -> None:
        self.scored: dict[str, list[tuple[str, bool, str]]] = {}
        self.fatal: list[tuple[str, str, bool, str]] = []

    def score(self, family: str, instance: str, ok: bool, detail: str = "") -> None:
        self.scored.setdefault(family, []).append((instance, ok, detail))

    def guard(self, group: str, instance: str, ok: bool, detail: str = "") -> None:
        self.fatal.append((group, instance, ok, detail))

    @property
    def violated_guards(self) -> list[tuple[str, str, bool, str]]:
        return [row for row in self.fatal if not row[2]]

    @property
    def failed_instances(self) -> list[tuple[str, str, str]]:
        return [
            (family, instance, detail)
            for family, rows in self.scored.items()
            for instance, ok, detail in rows
            if not ok
        ]


def _render(cell: Cell):
    return render_step_goal(
        cell.record,
        cell.dims,
        cell.tp_ranks,
        per_layer_calc_ns=5,
        ep_ranks=cell.ep_ranks,
    )


def _tp_messages(trace) -> list:
    return [
        message
        for message in trace.messages
        if message.operation_id is not None and ":tp-" in message.operation_id
    ]


def _moe_messages(trace) -> list:
    return [
        message
        for message in trace.messages
        if message.operation_id is not None and ":ep-" in message.operation_id
    ]


def measure(cell: Cell) -> dict:
    """One cell's raw observations from the three independent consumers."""

    record = cell.record
    dims = cell.dims
    row: dict = {
        "kind": cell.kind,
        "tp_width": cell.tp_width,
        "num_layers": cell.num_layers,
        "tokens": cell.tokens,
        "expected_sites_per_layer": cell.sites_per_layer,
        "expected_tp_messages": cell.tp_messages,
        "expected_tp_bytes": cell.tp_bytes,
        "expected_moe_operations": cell.moe_operations,
    }

    planned = step_tp_allreduces(record, dims, cell.tp_ranks)
    row["planned_sites"] = [operation.site for operation in planned]
    row["planned_operations"] = len(planned)
    row["planned_payload_bytes"] = sorted({op.payload_bytes for op in planned})

    moe = step_moe_alltoalls(
        record, dims, cell.ep_ranks if cell.ep_ranks is not None else ()
    )
    row["moe_operations"] = len(moe)
    row["moe_pair_bytes"] = sum(
        size for op in moe for _, _, size in op.pair_payload_bytes
    )

    renderable = bool(planned) or bool(moe)
    row["renderable"] = renderable
    if not renderable:
        try:
            _render(cell)
        except ValueError as error:
            row["refusal"] = str(error)
        else:
            row["refusal"] = None
        return row

    trace = _render(cell)
    text = trace.render()
    row["goal_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    tp_messages = _tp_messages(trace)
    row["render_tp_messages"] = len(tp_messages)
    row["render_tp_bytes"] = sum(message.payload_bytes for message in tp_messages)
    row["render_tp_operations"] = len(
        {message.operation_id for message in tp_messages}
    )

    tags_by_operation: dict[str, set[int]] = {}
    for message in trace.messages:
        tags_by_operation.setdefault(message.operation_id, set()).add(message.tag)
    shared = 0
    seen: set[int] = set()
    for tags in tags_by_operation.values():
        shared += len(tags & seen)
        seen |= tags
    row["shared_tags"] = shared
    moe_tags = {
        tag
        for name, tags in tags_by_operation.items()
        if ":ep-" in name
        for tag in tags
    }
    row["moe_tag_base"] = min(moe_tags) if moe_tags else None
    row["expected_moe_tag_base"] = (
        BASE_TAG + cell.n_sites * cell.tag_stride if moe_tags else None
    )
    row["render_moe_messages"] = len(_moe_messages(trace))

    phases = step_communication_phases(
        record, dims, cell.tp_ranks, ep_ranks=cell.ep_ranks
    )
    ring_phases = [phase for phase in phases if ":tp-" in phase.phase_id]
    row["phase_tp_rounds"] = len(ring_phases)
    row["phase_tp_bytes"] = sum(
        segment.payload_bytes for phase in ring_phases for segment in phase.segments
    )
    row["phase_tp_sites"] = len(
        {phase.phase_id.split(":")[1] for phase in ring_phases}
    )

    graph = SerialStepLowerer(
        SerialStepLowererConfig(dims, cell.tp_ranks, ep_ranks=cell.ep_ranks)
    ).lower(record)
    all_reduce_ids = {
        operation.operation_id
        for operation in graph.operations
        if isinstance(operation.work, CollectiveWork)
        and operation.work.collective == "all-reduce"
    }
    row["graph_tp_operations"] = len(all_reduce_ids)
    row["graph_tp_bytes"] = sum(
        extent.payload_bytes
        for plan in graph.collective_plans
        if plan.operation_id in all_reduce_ids
        for extent in plan.extents
    )
    row["graph_moe_operations"] = sum(
        1
        for operation in graph.operations
        if isinstance(operation.work, CollectiveWork)
        and operation.work.collective == "all-to-allv"
    )
    return row


def evaluate(rows: list[dict], ledger: Ledger) -> None:
    by_key = {
        (row["kind"], row["tp_width"], row["num_layers"], row["tokens"]): row
        for row in rows
    }
    for cell in cells():
        row = by_key[(cell.kind, cell.tp_width, cell.num_layers, cell.tokens)]
        label = cell.label

        # F1: the site rule itself, by construction of the code under test.
        expected_sites = (
            EXPERT_PARALLEL_TP_ALLREDUCE_SITES
            if cell.sites_per_layer == 1
            else TP_ALLREDUCE_SITES
        )
        ledger.guard(
            "F1 site rule",
            label,
            layer_tp_allreduce_sites(cell.dims) == expected_sites,
            f"sites={layer_tp_allreduce_sites(cell.dims)}",
        )

        # F3: the routed all-to-all inventory is untouched.
        graph_moe = row.get("graph_moe_operations", 0)
        ledger.guard(
            "F3 moe inventory",
            label,
            row["moe_operations"] == cell.moe_operations
            and graph_moe == cell.moe_operations,
            f"planner={row['moe_operations']} graph={graph_moe}",
        )

        if cell.tp_width < 2:
            # F5: a step with no collective at all is still refused.
            if cell.ep_ranks is None:
                ledger.guard(
                    "F5 refusal",
                    label,
                    row.get("refusal") is not None
                    and "no tensor-parallel collectives" in row["refusal"],
                    str(row.get("refusal")),
                )
            ledger.guard(
                "F2 no width dependence",
                label,
                row["planned_operations"] == 0,
                f"operations={row['planned_operations']}",
            )
            continue

        # F2: the site inventory is the frozen one at every group width.
        ledger.guard(
            "F2 no width dependence",
            label,
            row["planned_sites"]
            == list(expected_sites) * cell.num_layers,
            f"sites={row['planned_sites'][:4]}",
        )

        # S1: the GOAL renderer against the frozen closed form.
        ledger.score(
            "S1 renderer",
            label,
            row["render_tp_messages"] == cell.tp_messages
            and row["render_tp_bytes"] == cell.tp_bytes,
            f"messages={row['render_tp_messages']}/{cell.tp_messages} "
            f"bytes={row['render_tp_bytes']}/{cell.tp_bytes}",
        )

        # S2: the communication-phase planner against the same closed form.
        ledger.score(
            "S2 phases",
            label,
            row["phase_tp_rounds"] == cell.n_sites * cell.tag_stride
            and row["phase_tp_bytes"] == cell.tp_bytes,
            f"rounds={row['phase_tp_rounds']}/{cell.n_sites * cell.tag_stride} "
            f"bytes={row['phase_tp_bytes']}/{cell.tp_bytes}",
        )

        # S3: the graph lowerer and its collective plan.
        ledger.score(
            "S3 graph",
            label,
            row["graph_tp_operations"] == cell.n_sites
            and row["graph_tp_bytes"] == cell.tp_bytes,
            f"operations={row['graph_tp_operations']}/{cell.n_sites} "
            f"bytes={row['graph_tp_bytes']}/{cell.tp_bytes}",
        )

        # S4: tags, only where both collective families are rendered.
        if cell.moe_operations:
            ledger.score(
                "S4 tags",
                label,
                row["shared_tags"] == 0
                and row["moe_tag_base"] == row["expected_moe_tag_base"],
                f"shared={row['shared_tags']} "
                f"moe_base={row['moe_tag_base']}/{row['expected_moe_tag_base']}",
            )

    # F4: the two unchanged arms share one rendered GOAL identity per shape.
    for width in (2, 8):
        for layers in LAYER_COUNTS:
            for tokens in TOKEN_COUNTS:
                dense = by_key[("dense", width, layers, tokens)]["goal_sha256"]
                sharded = by_key[("routed-tp", width, layers, tokens)]["goal_sha256"]
                ledger.guard(
                    "F4 unchanged arms",
                    f"W={width} L={layers} T={tokens}",
                    dense == sharded,
                    f"{dense[:12]} vs {sharded[:12]}",
                )


def corollaries(rows: list[dict]) -> list[tuple[str, str]]:
    """Entailed readings, reported and never scored."""

    by_key = {
        (row["kind"], row["tp_width"], row["num_layers"], row["tokens"]): row
        for row in rows
    }
    reported: list[tuple[str, str]] = []
    for kind in KINDS:
        for layers in LAYER_COUNTS:
            small = by_key[(kind, 8, layers, 3)]["render_tp_bytes"]
            large = by_key[(kind, 8, layers, 12)]["render_tp_bytes"]
            narrow = by_key[(kind, 2, layers, 3)]["render_tp_bytes"]
            reported.append(
                (
                    f"{kind} L={layers}",
                    (
                        f"token ratio {large / small:.6f}, width ratio "
                        f"{small / narrow:.6f}"
                    ),
                )
            )
    for width in (2, 8):
        for layers in LAYER_COUNTS:
            dense = by_key[("dense", width, layers, 3)]["render_tp_bytes"]
            routed = by_key[("routed-ep", width, layers, 3)]["render_tp_bytes"]
            reported.append(
                (
                    f"routed-ep over dense W={width} L={layers}",
                    f"{routed / dense:.6f}",
                )
            )
    headline = by_key[("routed-ep", 8, 24, 3)]
    reported.append(
        (
            "headline inventory",
            (
                f"{headline['render_tp_operations']} allreduces plus "
                f"{headline['moe_operations']} all-to-alls, "
                f"{headline['render_tp_operations'] + headline['moe_operations']}"
                " collectives"
            ),
        )
    )
    return reported


def resolve_output(argument: str | None) -> Path:
    if argument:
        return Path(argument)
    root = os.environ.get("SIMLLM_DATA_ROOT")
    if not root:
        raise SystemExit(
            "pass --out, or set SIMLLM_DATA_ROOT to the external run root"
        )
    return Path(root) / "moe_tp_sites_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="directory for the raw rows")
    arguments = parser.parse_args()
    output = resolve_output(arguments.out)
    output.mkdir(parents=True, exist_ok=True)

    rows = [measure(cell) for cell in cells()]
    ledger = Ledger()
    evaluate(rows, ledger)

    (output / "cells.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"cells: {len(rows)}")
    print()
    print("fatal guards (void, never scored)")
    groups: dict[str, list[bool]] = {}
    for group, _, ok, _ in ledger.fatal:
        groups.setdefault(group, []).append(ok)
    for group, results in groups.items():
        state = "held" if all(results) else "VIOLATED"
        print(f"  {group}: {state} over {len(results)} checks")
    for group, instance, _, detail in ledger.violated_guards:
        print(f"    violation {group} at {instance}: {detail}")
    print()
    print("scored families (genuine risk)")
    for family, results in ledger.scored.items():
        passed = sum(1 for _, ok, _ in results if ok)
        print(f"  {family}: {passed}/{len(results)}")
    for family, instance, detail in ledger.failed_instances:
        print(f"    failure {family} at {instance}: {detail}")
    print()
    print("entailed corollaries (reported, not scored)")
    for name, value in corollaries(rows):
        print(f"  {name}: {value}")
    print()
    print(f"raw rows: {output / 'cells.json'}")

    if ledger.violated_guards:
        print("VERDICT: void, a fatal guard was violated")
        return 2
    total = sum(len(results) for results in ledger.scored.values())
    passed = total - len(ledger.failed_instances)
    print(f"VERDICT: {passed}/{total} scored instances passed, no guard violated")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
