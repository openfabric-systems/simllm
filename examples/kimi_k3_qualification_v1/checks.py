"""Complete evidence identities and causal controls for K3 qualification."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace

from examples.kimi_k3_structure_v1.run_study import Evidence as HistoricalEvidence
from examples.kimi_k3_structure_v1.run_study import digest
from simllm.calibration.canonical import canonical_bytes

KINDS = {"guard": "guards", "oracle": "oracles", "relation": "relations"}
MLA_LAYERS = (*range(3, 92, 4), 92)


def successor_ids(frozen):
    result = {}
    for stage in frozen["successor_stage_ids"]:
        category = stage.split(":")[0]
        for pattern in frozen["successor_check_ids"][category]:
            if ":LAYER" in pattern:
                stem = pattern.split(":LAYER")[0]
                domain = MLA_LAYERS if pattern.startswith("cache-ancestry") else range(1, 93)
                checks = [f"{stem}:{layer}" for layer in domain]
            elif pattern.startswith("file:NAME"):
                framework = stage.split(":")[1]
                checks = ["file:" + n for n in frozen["source_files"]
                          if n.startswith(framework + "-")]
            elif pattern.startswith("origin:MODULE"):
                checks = ["origin:" + n for n in frozen["required_import_origins"][
                    stage.split(":")[1]
                ]]
            else:
                checks = [pattern]
            for check in checks:
                name = f"successor:{stage}:{check}"
                if name in result:
                    raise ValueError("duplicate frozen successor evidence identity")
                result[name] = "oracle" if check == "completion-oracle" else "guard"
    return result


def inherited_ids(frozen, historical):
    result = {}
    moved = set(frozen["coverage"]["unscored_inherited_relations"])
    for directory in ("frozen-v1", "retained-audit-v1"):
        for kind, filename in KINDS.items():
            relative = f"{directory}/{filename}.json"
            if relative not in frozen["historical_inputs"]:
                continue
            raw = (historical / relative).read_bytes()
            if digest(raw) != frozen["historical_inputs"][relative]:
                raise ValueError("historical identity manifest hash differs: " + relative)
            for row in json.loads(raw):
                name = row["name"]
                selected_kind = kind
                if directory == "retained-audit-v1":
                    if not name.startswith(("vllm:", "sglang:", "vllm-p", "sglang-p",
                                            "reject-logical:")):
                        continue
                    if name.endswith(":post-specified-shared-addition"):
                        continue
                    name = "audit:" + name
                elif name in moved:
                    selected_kind = "guard"
                if name in result:
                    raise ValueError("duplicate historical evidence identity")
                result[name] = selected_kind
    return result


def inherited_families(frozen, historical):
    raw = (historical / "frozen-v1/relations.json").read_bytes()
    if digest(raw) != frozen["historical_inputs"]["frozen-v1/relations.json"]:
        raise ValueError("historical relation identity manifest changed")
    moved = set(frozen["coverage"]["unscored_inherited_relations"])
    return {row["name"]: row["family"] for row in json.loads(raw) if row["name"] not in moved}


class Evidence(HistoricalEvidence):
    def __init__(self, expected, stages):
        super().__init__()
        self.expected = expected
        self.stage_ids = set(stages)
        self.seen = {}
        self.finished_stages = set()
        self.admission_findings = []

    def check(self, name, actual, expected, *, kind="guard", family=None):
        if name.endswith(":decode-prompt-independence"):
            kind, family = "guard", None
        if name in self.seen or self.expected.get(name) != kind:
            self.admission_findings.append("duplicate, unknown or mistyped evidence: " + name)
        self.seen[name] = kind
        super().check(name, actual, expected, kind=kind, family=family)

    def finish_stage(self, stage):
        if stage not in self.stage_ids or stage in self.finished_stages:
            self.admission_findings.append("duplicate or unknown stage: " + stage)
        self.finished_stages.add(stage)

    def completeness(self):
        findings = list(self.admission_findings)
        for label, expected, actual in (
            ("evidence", set(self.expected), set(self.seen)),
            ("stage", self.stage_ids, self.finished_stages),
        ):
            missing, extra = expected - actual, actual - expected
            if missing or extra:
                findings.append({"domain": label, "missing_count": len(missing),
                                 "extra_count": len(extra), "missing": sorted(missing),
                                 "extra": sorted(extra)})
        return findings


class Scoped:
    def __init__(self, evidence, prefix):
        self.evidence, self.prefix = evidence, prefix

    def check(self, name, actual, expected, *, kind="guard", family=None):
        self.evidence.check(self.prefix + name, actual, expected, kind=kind, family=family)


def scoped(evidence, stage):
    return Scoped(evidence, "successor:" + stage + ":")


def ancestors(graph, target):
    parents = {op.operation_id: op.depends_on for op in graph.operations}
    pending, result = list(parents[target]), set()
    while pending:
        name = pending.pop()
        if name not in parents:
            raise ValueError("dependency names an absent operation")
        if name not in result:
            result.add(name)
            pending.extend(parents[name])
    if target in result:
        raise ValueError("dependency cycle")
    return result


def operation_map(graph):
    return {(op.correlation.layer, op.work.kernel.removeprefix("kimi_k3.")): op
            for op in graph.operations}


def causal_guards(graph, phase, evidence):
    ops = operation_map(graph)
    head = ops[None, "lm_head"].operation_id
    head_parents = ancestors(graph, head)
    successors = {}
    for op in graph.operations:
        for parent in op.depends_on:
            successors.setdefault(parent, []).append(op.operation_id)
    expected_ids = [f"{graph.execution_id}:"
                    + ("outer" if op.correlation.layer is None
                       else f"layer-{op.correlation.layer:03d}")
                    + ":" + op.work.kernel.removeprefix("kimi_k3.") for op in graph.operations]
    actual_ids = [op.operation_id for op in graph.operations]
    evidence.check("operation-identity-set",
                   actual_ids == expected_ids and len(set(actual_ids)) == len(actual_ids), True)
    evidence.check("logical-demand-boundary", all(
        op.work.hbm_bytes is None and op.work.nominal_duration_ps is None
        and op.work.scope == "logical-operator" for op in graph.operations
    ), True)
    evidence.check("completion-frontier", graph.completion_operation_ids,
                   (head, *(ops[layer, "mla.compressed_cache_commit"].operation_id
                            for layer in MLA_LAYERS)))
    for layer in range(1, 93):
        get = lambda name, layer=layer: ops[layer, name]
        join = get("moe.shared_routed_add")
        routed, shared = get("moe.latent_up_projection"), get("moe.shared_down_projection")
        gate, up, active = (get("moe.shared_" + name)
                            for name in ("gate_projection", "up_projection", "situ"))
        post = next(op for (index, name), op in ops.items()
                    if index == layer and name.startswith("residual.mlp_sources_"))
        evidence.check(f"join-parents:{layer}", sorted(join.depends_on),
                       sorted((routed.operation_id, shared.operation_id)))
        evidence.check(f"shared-fork:{layer}", (gate.depends_on, up.depends_on),
                       ((post.operation_id,), (post.operation_id,)))
        evidence.check(f"shared-activation:{layer}", sorted(active.depends_on),
                       sorted((gate.operation_id, up.operation_id)))
        evidence.check(f"shared-down:{layer}", shared.depends_on, (active.operation_id,))
        evidence.check(f"join-successor:{layer}", successors.get(join.operation_id, []),
                       [get("residual.ffn_delta_prefix_add").operation_id])
        evidence.check(f"join-head-ancestor:{layer}", join.operation_id in head_parents, True)
    for layer in MLA_LAYERS:
        cache = ops[layer, "mla.compressed_cache_commit"].operation_id
        attention = ops[layer, "mla.query_key_product"].operation_id
        evidence.check(f"cache-ancestry:{layer}", cache in ancestors(graph, attention),
                       phase == "decode")


def dependency_times(graph, ordinary, shared, addition):
    times, services = {}, {}
    for op in graph.operations:
        name = op.work.kernel
        service = (addition if name == "kimi_k3.moe.shared_routed_add" else
                   shared if name.startswith("kimi_k3.moe.shared_") else ordinary)
        services[op.operation_id] = service
        times[op.operation_id] = service + max(
            (times[parent] for parent in op.depends_on), default=0
        )
    return times, services


def causal_cell(graph, phase, shared, addition, frozen, evidence):
    ordinary = frozen["causal_grid"]["ordinary_region_ps"]
    times, services = dependency_times(graph, ordinary, shared, addition)
    actual = max(times[name] for name in graph.completion_operation_ids)
    attention = 8 if phase == "prefill" else 10
    expected = ordinary * (4 + 3 * 93 + 6 * 69 + attention * 24 + 3) + 92 * (
        max(9 * ordinary, 3 * shared) + addition
    )
    evidence.check("completion-oracle", actual, expected, kind="oracle")
    depth = 1812 if phase == "prefill" else 1860
    evidence.check("logical-floor", actual >= depth * min(ordinary, shared, addition), True)
    evidence.check("serial-ceiling", actual <= sum(services.values()), True)
    ops = operation_map(graph)
    for layer in range(1, 93):
        join = ops[layer, "moe.shared_routed_add"]
        parents = [ops[layer, name].operation_id for name in
                   ("moe.latent_up_projection", "moe.shared_down_projection")]
        evidence.check(f"join-equation:{layer}", times[join.operation_id],
                       max(times[parent] for parent in parents) + addition)
    return {"phase": phase, "shared_ps": shared, "final_add_ps": addition,
            "completion_ps": actual, "expected_ps": expected,
            "logical_floor_ps": depth * min(ordinary, shared, addition),
            "serial_ceiling_ps": sum(services.values())}


def corrupt_graph(graph, mode):
    ops = operation_map(graph)
    join = ops[1, "moe.shared_routed_add"]
    target, parents, guard = None, (), ""
    if mode in {"remove-routed-parent", "remove-shared-parent"}:
        target = join
        remove = ops[1, "moe.latent_up_projection" if mode == "remove-routed-parent"
                     else "moe.shared_down_projection"].operation_id
        parents = tuple(p for p in join.depends_on if p != remove)
        guard = "join-parents:1"
    elif mode == "serialize-shared-projections":
        target = ops[1, "moe.shared_up_projection"]
        parents = (*target.depends_on, ops[1, "moe.shared_gate_projection"].operation_id)
        guard = "shared-fork:1"
    elif mode == "bypass-final-add":
        target = ops[1, "residual.ffn_delta_prefix_add"]
        parents = tuple(ops[1, "moe.latent_up_projection"].operation_id
                        if p == join.operation_id else p for p in target.depends_on)
        guard = "join-successor:1"
    elif mode == "prefill-hidden-cache-wait":
        target = ops[3, "mla.kv_b_projection"]
        parents = (*target.depends_on, ops[3, "mla.compressed_cache_commit"].operation_id)
        guard = "cache-ancestry:3"
    elif mode == "decode-remove-cache-wait":
        target = ops[3, "mla.query_key_product"]
        parents = tuple(p for p in target.depends_on
                        if p != ops[3, "mla.compressed_cache_commit"].operation_id)
        guard = "cache-ancestry:3"
    elif mode == "omit-cache-frontier":
        return replace(graph, completion_operation_ids=graph.completion_operation_ids[:-1]), (
            "completion-frontier"
        )
    else:
        raise ValueError("unknown corruption control")
    return replace(graph, operations=tuple(
        replace(op, depends_on=parents) if op is target else op for op in graph.operations
    )), guard


def discriminate(graph, phase, mode, evidence):
    changed, guard = corrupt_graph(graph, mode)
    control = HistoricalEvidence()
    causal_guards(changed, phase, control)
    evidence.check("operation-count-preserved", len(changed.operations), len(graph.operations))
    evidence.check("owning-guard-discriminates",
                   [r["passed"] for r in control.guards if r["name"] == guard], [False])


def _equal_values(row):
    try:
        return canonical_bytes(row.get("actual")) == canonical_bytes(row.get("expected"))
    except (TypeError, ValueError):
        return False


def validate_evidence(rows, expected, stages, finished, families):
    findings, found = [], {}
    if (not isinstance(finished, list) or not all(isinstance(s, str) for s in finished)
            or Counter(finished) != Counter(stages)):
        findings.append("stage identities differ")
    for kind, key in KINDS.items():
        values = rows.get(key)
        if not isinstance(values, list):
            findings.append("missing evidence class: " + key)
            continue
        for row in values:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                findings.append("malformed evidence row")
                continue
            name = row["name"]
            if name in found or expected.get(name) != kind:
                findings.append("duplicate, unknown or mistyped evidence: " + name)
            found[name] = kind
            if (type(row.get("passed")) is not bool or row["passed"] is not True
                    or not _equal_values(row)
                    or set(row) != ({"name", "actual", "expected", "passed", "family"}
                                    if kind == "relation" else
                                    {"name", "actual", "expected", "passed"})):
                findings.append("failed or malformed assertion: " + name)
            if kind == "relation" and row.get("family") != families.get(name):
                findings.append("behavioral family identity differs: " + name)
            if kind != "relation" and "family" in row:
                findings.append("unscored row has behavioral family: " + name)
    relations = rows.get("relations")
    observed = {r.get("family") for r in relations if isinstance(r, dict)
                and isinstance(r.get("family"), str)} if isinstance(relations, list) else set()
    if observed != set(families.values()):
        findings.append("complete behavioral family set differs")
    if set(found) != set(expected):
        findings.append("complete evidence identity set differs")
    return findings
