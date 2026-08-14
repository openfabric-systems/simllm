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

#: kind -> (dims template, expert-parallel group, sites per layer)
KINDS: dict[str, tuple[ModelDims, tuple[int, ...] | None, int]] = {
    "dense": (_BASE, None, 2),
    "routed-tp": (replace(_ROUTED, local_num_experts=32), None, 2),
    "routed-ep": (replace(_ROUTED, local_num_experts=4), EP_RANKS, 1),
    # post-specified arm, added after the 2026-08-14 review: expert-parallel
    # dims with no declared all-to-all group, i.e. naive expert parallelism
    # (pinned vLLM 0.26.0 fused_moe/config.py:1052-1055 needs dp, pcp or
    # sequence parallelism for all-to-all kernels, and
    # fused_moe/runner/moe_runner.py:436-465 then all-reduces the fused
    # output over the TP group). It must render two sites and no all-to-all.
    "routed-naive-ep": (replace(_ROUTED, local_num_experts=4), None, 2),
}

#: the kinds the pre-registered freeze enumerated; the rest are post-specified
FROZEN_KINDS = ("dense", "routed-tp", "routed-ep")


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
    def moe_pair_bytes(self) -> int:
        """The uniform routed byte total, independent of the site rule."""

        if self.ep_ranks is None or len(self.ep_ranks) < 2 or self.tokens <= 0:
            return 0
        dims = self.dims
        per_pair = (
            self.tokens * dims.top_k * HIDDEN_SIZE * DTYPE_BYTES
        ) // len(self.ep_ranks)
        return 2 * self.num_layers * (len(self.ep_ranks) - 1) * per_pair

    @property
    def frozen(self) -> bool:
        return self.kind in FROZEN_KINDS

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
    """Separate registers per evidence class; the classes are never summed.

    ``scored`` holds the pre-registered families over the frozen cells.
    ``post_scored`` holds the same families evaluated on cells the freeze did
    not enumerate, which are post-specified and carry their own denominator.
    ``fatal`` holds void-not-scored guards and is never reported as a
    fraction.
    """

    def __init__(self) -> None:
        self.scored: dict[str, list[tuple[str, bool, str]]] = {}
        self.post_scored: dict[str, list[tuple[str, bool, str]]] = {}
        self.fatal: list[tuple[str, str, bool, str]] = []

    def score(
        self,
        family: str,
        instance: str,
        ok: bool,
        detail: str = "",
        *,
        frozen: bool = True,
    ) -> None:
        register = self.scored if frozen else self.post_scored
        register.setdefault(family, []).append((instance, ok, detail))

    def guard(self, group: str, instance: str, ok: bool, detail: str = "") -> None:
        self.fatal.append((group, instance, ok, detail))

    @property
    def violated_guards(self) -> list[tuple[str, str, bool, str]]:
        return [row for row in self.fatal if not row[2]]

    def failures(
        self, register: dict[str, list[tuple[str, bool, str]]]
    ) -> list[tuple[str, str, str]]:
        return [
            (family, instance, detail)
            for family, rows in register.items()
            for instance, ok, detail in rows
            if not ok
        ]

    @property
    def failed_instances(self) -> list[tuple[str, str, str]]:
        return self.failures(self.scored) + self.failures(self.post_scored)


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
        "expected_moe_pair_bytes": cell.moe_pair_bytes,
        "frozen": cell.frozen,
    }

    planned = step_tp_allreduces(
        record, dims, cell.tp_ranks, ep_ranks=cell.ep_ranks
    )
    row["planned_sites"] = [operation.site for operation in planned]
    row["planned_operations"] = len(planned)
    row["planned_payload_bytes"] = sorted({op.payload_bytes for op in planned})

    # F2, registered clauses executed after the 2026-08-14 review: the site
    # tuple may not move under a disjoint tensor-parallel rank relabeling, nor
    # under a change of expert-parallel group width, nor under a change of the
    # resident-expert count the rule no longer reads.
    relabelled = step_tp_allreduces(
        record,
        dims,
        tuple(rank + 64 for rank in cell.tp_ranks),
        ep_ranks=cell.ep_ranks,
    )
    row["relabelled_sites"] = [operation.site for operation in relabelled]
    narrow_group = None if cell.ep_ranks is None else cell.ep_ranks[:2]
    narrowed = step_tp_allreduces(
        record, dims, cell.tp_ranks, ep_ranks=narrow_group
    )
    row["narrow_group_sites"] = [operation.site for operation in narrowed]
    row["narrow_group_width"] = None if narrow_group is None else len(narrow_group)
    resident_swept = step_tp_allreduces(
        record,
        replace(dims, local_num_experts=dims.num_experts)
        if dims.num_experts > 0
        else dims,
        cell.tp_ranks,
        ep_ranks=cell.ep_ranks,
    )
    row["resident_swept_sites"] = [operation.site for operation in resident_swept]

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
        observed_sites = layer_tp_allreduce_sites(cell.dims, ep_ranks=cell.ep_ranks)
        ledger.guard(
            "F1 site rule",
            label,
            observed_sites == expected_sites,
            f"sites={observed_sites}",
        )

        # F3: the routed all-to-all inventory is untouched, in operations AND
        # in bytes. The byte oracle is the uniform closed form, which does not
        # read the site rule and never went through the changed code.
        graph_moe = row.get("graph_moe_operations", 0)
        ledger.guard(
            "F3 moe inventory",
            label,
            row["moe_operations"] == cell.moe_operations
            and graph_moe == cell.moe_operations
            and row["moe_pair_bytes"] == cell.moe_pair_bytes,
            f"planner={row['moe_operations']} graph={graph_moe} "
            f"bytes={row['moe_pair_bytes']}/{cell.moe_pair_bytes}",
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
            # A one-rank tensor-parallel world emits no allreduce at all, so
            # there is no site tuple to compare. This cell substitutes the
            # weaker "emitted nothing" predicate, and the substitution is
            # stated rather than left implicit.
            ledger.guard(
                "F2 invariance, W=1 substitution",
                label,
                row["planned_operations"] == 0,
                f"operations={row['planned_operations']}",
            )
            continue

        # F2: the site inventory does not move with the tensor-parallel width,
        # a disjoint rank relabeling, the expert-parallel group width, or the
        # resident-expert count the rule no longer reads. The last three
        # clauses were registered in the freeze and executed late.
        expected_list = list(expected_sites) * cell.num_layers
        ledger.guard(
            "F2 invariance",
            label,
            row["planned_sites"] == expected_list
            and row["relabelled_sites"] == expected_list
            and row["narrow_group_sites"] == expected_list
            and row["resident_swept_sites"] == expected_list,
            f"declared={row['planned_sites'][:2]} "
            f"relabelled={row['relabelled_sites'][:2]} "
            f"narrow={row['narrow_group_sites'][:2]} "
            f"resident={row['resident_swept_sites'][:2]}",
        )

        # S1: the GOAL renderer against the frozen closed form.
        ledger.score(
            "S1 renderer",
            label,
            row["render_tp_messages"] == cell.tp_messages
            and row["render_tp_bytes"] == cell.tp_bytes,
            f"messages={row['render_tp_messages']}/{cell.tp_messages} "
            f"bytes={row['render_tp_bytes']}/{cell.tp_bytes}",
            frozen=cell.frozen,
        )

        # S2: the communication-phase planner against the same closed form.
        ledger.score(
            "S2 phases",
            label,
            row["phase_tp_rounds"] == cell.n_sites * cell.tag_stride
            and row["phase_tp_bytes"] == cell.tp_bytes,
            f"rounds={row['phase_tp_rounds']}/{cell.n_sites * cell.tag_stride} "
            f"bytes={row['phase_tp_bytes']}/{cell.tp_bytes}",
            frozen=cell.frozen,
        )

        # S3: the graph lowerer and its collective plan.
        ledger.score(
            "S3 graph",
            label,
            row["graph_tp_operations"] == cell.n_sites
            and row["graph_tp_bytes"] == cell.tp_bytes,
            f"operations={row['graph_tp_operations']}/{cell.n_sites} "
            f"bytes={row['graph_tp_bytes']}/{cell.tp_bytes}",
            frozen=cell.frozen,
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
                frozen=cell.frozen,
            )

    # F4: every arm that renders no all-to-all shares one GOAL identity per
    # shape, which is the pre-change renderer's output. The naive expert
    # parallel arm joins them, which is the whole point of the corrected rule.
    for width in (2, 8):
        for layers in LAYER_COUNTS:
            for tokens in TOKEN_COUNTS:
                digests = {
                    kind: by_key[(kind, width, layers, tokens)]["goal_sha256"]
                    for kind in ("dense", "routed-tp", "routed-naive-ep")
                }
                ledger.guard(
                    "F4 unchanged arms",
                    f"W={width} L={layers} T={tokens}",
                    len(set(digests.values())) == 1,
                    ", ".join(
                        f"{kind}={digest[:12]}" for kind, digest in digests.items()
                    ),
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
            "headline inventory, all-to-all expert parallelism",
            (
                f"{headline['render_tp_operations']} allreduces plus "
                f"{headline['moe_operations']} all-to-alls, "
                f"{headline['render_tp_operations'] + headline['moe_operations']}"
                " collectives"
            ),
        )
    )
    naive = by_key[("routed-naive-ep", 8, 24, 3)]
    reported.append(
        (
            "same dims under naive expert parallelism",
            (
                f"{naive['render_tp_operations']} allreduces plus "
                f"{naive['moe_operations']} all-to-alls, "
                f"{naive['render_tp_operations'] + naive['moe_operations']}"
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

    frozen = sum(1 for row in rows if row["frozen"])
    print(f"cells: {len(rows)} ({frozen} frozen, {len(rows) - frozen} post-specified)")
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

    def report(title: str, register: dict[str, list[tuple[str, bool, str]]]) -> None:
        print()
        print(title)
        for family, results in register.items():
            passed = sum(1 for _, ok, _ in results if ok)
            changed = sum(
                1 for instance, _, _ in results if instance.startswith("routed-ep")
            )
            print(
                f"  {family}: {passed}/{len(results)} "
                f"({changed} on the changed path)"
            )
        for family, instance, detail in ledger.failures(register):
            print(f"    failure {family} at {instance}: {detail}")

    report("pre-registered scored families (genuine risk)", ledger.scored)
    report(
        "post-specified scored families, naive expert parallelism",
        ledger.post_scored,
    )
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
    passed = total - len(ledger.failures(ledger.scored))
    post_total = sum(len(results) for results in ledger.post_scored.values())
    post_passed = post_total - len(ledger.failures(ledger.post_scored))
    print(
        f"VERDICT: {passed}/{total} pre-registered and {post_passed}/{post_total} "
        "post-specified scored instances passed, no guard violated. The two "
        "denominators are separate evidence classes and are not summed."
    )
    return 0 if (passed, post_passed) == (total, post_total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
