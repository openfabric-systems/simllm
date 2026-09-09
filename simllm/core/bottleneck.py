"""Strict, timing-neutral rankings of disjoint selected critical-path intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, fields
from fractions import Fraction

from simllm.core._wire import (
    _array,
    _fields,
    _integer,
    _number,
    _object,
    _scalar,
    _string,
)

SCHEMA = "simllm-bottleneck-report-v1"
CLASSES = frozenset({
    "hbm-bound", "compute-bound", "unclassified-kernel", "device-queue",
    "intra-node-collective", "fabric", "co-critical-collective", "host-launch",
    "batching-queue", "external-dependency", "kv", "dma", "control",
})


@dataclass(frozen=True)
class KernelEvidence:
    """Provider kernel/config identity with an optional explicitly joined cell.

    Kernel identity is the same name/config pair used by profile calibration.
    A fused analytical kernel is not an invented join to a measured CUDA kernel.
    """

    kernel_id: str
    config: tuple[tuple[str, str | int | float | bool], ...]
    flops: float
    bytes_moved: float
    peak_flops_s: float
    hbm_bytes_s: float
    gpu_name: str
    ledger_cell_id: str | None = None
    measured_time_ps: int | None = None
    evidence_class: str = "absent-by-design"
    source_state: str = "absent-by-design"

    def __post_init__(self):
        _string(self.kernel_id, "kernel_id")
        _string(self.gpu_name, "gpu_name")
        if not isinstance(self.config, tuple):
            raise TypeError("kernel config must be a tuple")
        for item in self.config:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("kernel config requires key/value tuples")
            _string(item[0], "kernel.config.key")
            _scalar(item[1], "kernel.config.value")
        if len({k for k, _ in self.config}) != len(self.config):
            raise ValueError("duplicate kernel config key")
        for name in ("flops", "bytes_moved", "peak_flops_s", "hbm_bytes_s"):
            _number(getattr(self, name), name, nonnegative=True)
        if self.peak_flops_s <= 0 or self.hbm_bytes_s <= 0:
            raise ValueError("kernel roofs must be positive")
        if self.ledger_cell_id is None:
            if (self.measured_time_ps is not None or self.evidence_class != "absent-by-design"
                    or self.source_state != "absent-by-design"):
                raise ValueError("absent measurement must be absent-by-design")
        else:
            _string(self.ledger_cell_id, "ledger_cell_id")
            _integer(self.measured_time_ps, "measured_time_ps", minimum=1)
            _string(self.evidence_class, "evidence_class")
            if self.evidence_class == "absent-by-design" or self.source_state not in {"void", "nonvoid"}:
                raise ValueError("measured cell requires evidence and void state")

    @property
    def bound_class(self):
        if not self.flops and not self.bytes_moved:
            return "unclassified-kernel"
        return ("hbm-bound" if Fraction(self.flops) * Fraction(self.hbm_bytes_s)
                < Fraction(self.bytes_moved) * Fraction(self.peak_flops_s) else "compute-bound")

    @property
    def achieved_fraction(self):
        if self.measured_time_ps is None or self.bound_class == "unclassified-kernel":
            return None
        return float(max(Fraction(self.flops) / Fraction(self.peak_flops_s),
                         Fraction(self.bytes_moved) / Fraction(self.hbm_bytes_s))
                     * 10**12 / self.measured_time_ps)

    @classmethod
    def from_kernel(cls, kernel, gpu):
        return cls(kernel.name, kernel.config, kernel.flops, kernel.bytes_moved,
                   gpu.peak_flops, gpu.mem_bandwidth, gpu.name)


def join_kernel_cell(kernel, cells):
    """Use the calibration table's exact (kernel name, config, GPU name) key."""
    if cells is None:
        return kernel
    if not isinstance(cells, Mapping):
        raise TypeError("kernel cells must be a mapping")
    key = kernel.kernel_id, kernel.config, kernel.gpu_name
    candidate = cells.get(key)
    if candidate is None:
        return kernel
    if not isinstance(candidate, KernelEvidence):
        raise TypeError("kernel cell must be KernelEvidence")
    candidate.__post_init__()
    if (candidate.kernel_id, candidate.config, candidate.gpu_name) != key:
        raise ValueError("kernel ledger join identity mismatch")
    if candidate.flops != kernel.flops or candidate.bytes_moved != kernel.bytes_moved:
        raise ValueError("kernel ledger join work mismatch")
    return candidate


@dataclass(frozen=True)
class FlowTail:
    """Within-artifact FCT spread, diagnostic only and never additive wall time."""

    execution_id: str
    operation_id: str
    flow_count: int
    min_fct_ps: int
    max_fct_ps: int

    def __post_init__(self):
        _string(self.execution_id, "execution_id")
        _string(self.operation_id, "operation_id")
        _integer(self.flow_count, "flow_count", minimum=1)
        _integer(self.min_fct_ps, "min_fct_ps", nonnegative=True)
        _integer(self.max_fct_ps, "max_fct_ps", minimum=max(1, self.min_fct_ps))

    @property
    def tail_share(self):
        return Fraction(self.max_fct_ps - self.min_fct_ps, self.max_fct_ps)


@dataclass(frozen=True)
class ClassShare:
    class_name: str
    participant_width: int | None
    span_ps: int

    def __post_init__(self):
        if self.class_name not in CLASSES:
            raise ValueError("unknown bottleneck class")
        _integer(self.span_ps, "class.span_ps", minimum=1)
        if self.participant_width is not None:
            _integer(self.participant_width, "participant_width", minimum=1)
        if ("collective" in self.class_name or self.class_name == "fabric"):
            if self.participant_width is None:
                raise ValueError("communication class requires participant width")
        elif self.participant_width is not None:
            raise ValueError("noncommunication class must omit participant width")

    @property
    def order(self):
        return (-self.span_ps, self.class_name, self.participant_width or 0)


def _kernel_key(kernel):
    # Strict canonical wire bytes retain scalar types and configuration order.
    import json
    return json.dumps(_kernel_json(kernel), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class BottleneckRanking:
    span_ps: int
    divisor: int = 1
    classes: tuple[ClassShare, ...] = ()
    kernels: tuple[KernelEvidence, ...] = ()
    flow_tails: tuple[FlowTail, ...] = ()

    def __post_init__(self):
        _integer(self.span_ps, "ranking.span_ps", nonnegative=True)
        _integer(self.divisor, "ranking.divisor", minimum=1)
        for name, cls in (("classes", ClassShare), ("kernels", KernelEvidence), ("flow_tails", FlowTail)):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(v, cls) for v in value):
                raise ValueError(f"ranking.{name} requires typed tuples")
            for item in value:
                item.__post_init__()
        if sum(c.span_ps for c in self.classes) != self.span_ps:
            raise ValueError("bottleneck classes do not conserve selected latency")
        if self.classes != tuple(sorted(self.classes, key=lambda c: c.order)):
            raise ValueError("bottleneck classes are not ranked")
        keys = [(c.class_name, c.participant_width) for c in self.classes]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate bottleneck class")
        bound_classes = {c.class_name for c in self.classes} & {"hbm-bound", "compute-bound"}
        if not bound_classes <= {k.bound_class for k in self.kernels}:
            raise ValueError("bound kernel class lacks work and roof evidence")
        identities = [(k.kernel_id, k.config, k.gpu_name) for k in self.kernels]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate kernel identity")
        keys = [_kernel_key(k) for k in self.kernels]
        if keys != sorted(set(keys)):
            raise ValueError("kernel evidence must be unique and canonical")
        keys = [(f.execution_id, f.operation_id) for f in self.flow_tails]
        if keys != sorted(set(keys)):
            raise ValueError("flow tail identities must be unique and canonical")

    @property
    def latency_ps(self):
        return Fraction(self.span_ps, self.divisor)

    def share(self, class_name):
        return (Fraction(sum(c.span_ps for c in self.classes if c.class_name == class_name), self.span_ps)
                if self.span_ps else Fraction(0))


def ranking(parts=(), *, kernels=(), flow_tails=(), divisor=1):
    totals = defaultdict(int)
    for name, width, duration in parts:
        _integer(duration, "selected duration", nonnegative=True)
        totals[name, width] += duration
    classes = tuple(sorted((ClassShare(n, w, t) for (n, w), t in totals.items() if t),
                           key=lambda c: c.order))
    kernel_map = {}
    kernel_identities = {}
    for kernel in kernels:
        identity = kernel.kernel_id, kernel.config, kernel.gpu_name
        if identity in kernel_identities and kernel_identities[identity] != kernel:
            raise ValueError("inconsistent kernel evidence for one calibration identity")
        kernel_identities[identity] = kernel
        kernel_map[_kernel_key(kernel)] = kernel
    tail_map = {}
    for tail in flow_tails:
        key = (tail.execution_id, tail.operation_id)
        if key in tail_map and tail_map[key] != tail:
            raise ValueError("inconsistent flow tail projection")
        tail_map[key] = tail
    return BottleneckRanking(sum(c.span_ps for c in classes), divisor, classes,
                             tuple(kernel_map[k] for k in sorted(kernel_map)),
                             tuple(tail_map[k] for k in sorted(tail_map)))


def combine(*values, divisor=1):
    if any(v.divisor != 1 for v in values):
        raise ValueError("combine requires undivided disjoint spans")
    return ranking(((c.class_name, c.participant_width, c.span_ps) for v in values for c in v.classes),
                   kernels=(k for v in values for k in v.kernels),
                   flow_tails=(f for v in values for f in v.flow_tails), divisor=divisor)


@dataclass(frozen=True)
class RequestBottleneck:
    request_id: str
    ttft: BottleneckRanking
    tpot: BottleneckRanking | None = None

    def __post_init__(self):
        _string(self.request_id, "request_id")
        if not isinstance(self.ttft, BottleneckRanking) or self.ttft.divisor != 1:
            raise ValueError("TTFT requires an undivided ranking")
        self.ttft.__post_init__()
        if self.tpot is not None:
            if not isinstance(self.tpot, BottleneckRanking):
                raise ValueError("TPOT requires a ranking")
            self.tpot.__post_init__()


@dataclass(frozen=True)
class BottleneckReport:
    step_index: int
    step: BottleneckRanking
    requests: tuple[RequestBottleneck, ...] = ()

    def __post_init__(self):
        _integer(self.step_index, "step_index", nonnegative=True)
        if not isinstance(self.step, BottleneckRanking) or self.step.divisor != 1:
            raise ValueError("step requires an undivided ranking")
        self.step.__post_init__()
        if not isinstance(self.requests, tuple) or any(not isinstance(r, RequestBottleneck) for r in self.requests):
            raise ValueError("requests must be a typed tuple")
        for request in self.requests:
            request.__post_init__()
        ids = [r.request_id for r in self.requests]
        if ids != sorted(set(ids)):
            raise ValueError("request identities must be unique and canonical")

    def validate_result(self, result):
        self.__post_init__()
        if self.step_index != result.step_index or self.step.span_ps != result.step_latency_ps:
            raise ValueError("bottleneck report disagrees with StepResult")
        by_id = {r.request_id: r for r in self.requests}
        for metric in result.request_metrics:
            if metric.request_id not in by_id:
                raise ValueError("missing request bottleneck")
            row = by_id[metric.request_id]
            if ((row.tpot is None) != (metric.token_index == 1)
                    or (row.tpot is not None and row.tpot.divisor != metric.token_index - 1)):
                raise ValueError("TPOT divisor disagrees with completed token intervals")
            if (row.ttft.latency_ps != metric.ttft_ps
                    or (None if row.tpot is None else row.tpot.latency_ps) != metric.tpot_ps):
                raise ValueError("request bottleneck disagrees with metric")


def classify_segment(segment, visits, *, kernel=None, width=1, owner=None, flow_tail=None):
    """Clip only the authority's selected visits to one causal segment.

    Packet GPU service can be an external-dependency timing component while
    still having a kernel owner. Visit ownership resolves that cross projection.
    """
    from simllm.core.execution import ResourceKind as R

    parts = []
    cursor = segment.started_at_ps
    kernel_used = fabric_used = False
    for visit in visits:
        if visit.operation_id != segment.operation_id:
            raise ValueError("selected visit belongs to another operation")
        if visit.completed_at_ps > segment.completed_at_ps:
            raise ValueError("selected visit exceeds its segment")
        start = max(cursor, visit.eligible_at_ps)
        parts.append(("external-dependency", None, start - cursor))
        kind = visit.resource.kind
        communication = kind in {R.NCCL_CHANNEL, R.NVLINK, R.NIC, R.NIC_SEND_QUEUE,
                                 R.NIC_RECEIVE_QUEUE, R.COMPLETION_QUEUE}
        if kind == R.HOST_LAUNCH_QUEUE:
            name = "host-launch"
        elif communication:
            if kind in {R.NIC, R.NIC_SEND_QUEUE, R.NIC_RECEIVE_QUEUE, R.COMPLETION_QUEUE}:
                name = "fabric"
            elif visit.stage == "co_critical_ps":
                name = "co-critical-collective"
            else:
                name = "intra-node-collective"
        elif kind in {R.GPU_WORK_QUEUE, R.GPU_SCHEDULER} or (kind == R.HBM_QUEUE and owner == "kernel"):
            name = kernel.bound_class if kernel else "unclassified-kernel"
            kernel_used = True
        elif kind == R.COPY_ENGINE or owner == "dma":
            name = "dma"
        elif kind == R.HBM_QUEUE and owner == "kv":
            name = "kv"
        else:
            name = "control"
        wait = max(0, visit.started_at_ps - start)
        queue_name = "device-queue" if name in {"hbm-bound", "compute-bound", "unclassified-kernel"} else name
        parts.append((queue_name, width if communication else None, wait))
        duration = max(0, visit.completed_at_ps - max(start, visit.started_at_ps))
        parts.append((name, width if communication else None, duration))
        fabric_used |= name == "fabric" and wait + duration > 0
        cursor = max(cursor, visit.completed_at_ps)
    parts.append(("external-dependency", None, segment.completed_at_ps - cursor))
    result = ranking(parts, kernels=(kernel,) if kernel_used and kernel else (),
                     flow_tails=(flow_tail,) if fabric_used and flow_tail else ())
    if result.span_ps != segment.breakdown.operation_latency_ps:
        raise ValueError("classified segment does not conserve critical breakdown")
    return result


def classify_runtime(graph, report, selected_paths, gpu, *, kernel_cells=None):
    """Return rankings keyed by the runtime's exact participant segment identity."""
    from simllm.compute import KernelSpec
    from simllm.core.execution import CollectiveWork, ComputeWork, DmaWork, KvCacheWork
    from simllm.core.runtime import CoarseDeviceRuntime

    if report.execution_id != graph.execution_id:
        raise ValueError("runtime report belongs to another graph")
    operations = {o.operation_id: o for o in graph.operations}
    result = {}
    for record in report.operations:
        work = operations[record.operation_id].work
        kernel = None
        if isinstance(work, ComputeWork):
            work.require_executable()
            if work.scope != "kernel-region":
                raise ValueError("synthetic operator work has no physical bottleneck classification")
            kernel = KernelEvidence.from_kernel(KernelSpec(work.kernel, work.flops, work.hbm_bytes, work.config), gpu)
            kernel = join_kernel_cell(kernel, kernel_cells)
        owner = ("kernel" if isinstance(work, ComputeWork) else "kv" if isinstance(work, KvCacheWork)
                 else "dma" if isinstance(work, DmaWork) else None)
        width = len(work.ranks) if isinstance(work, CollectiveWork) else 1
        for segment in record.critical_segments:
            key = segment.operation_id, segment.participant_rank
            if key not in selected_paths:
                raise ValueError("authority did not retain selected visits")
            visits = selected_paths[key]
            if any(v.execution_id != graph.execution_id for v in visits):
                raise ValueError("selected visit belongs to another execution")
            breakdown = CoarseDeviceRuntime._critical_path_breakdown(
                segment.operation_id, visits, segment.started_at_ps, segment.completed_at_ps,
            )
            if breakdown != segment.breakdown:
                raise ValueError("selected visits disagree with segment breakdown")
            result[key] = classify_segment(segment, visits, kernel=kernel, width=width, owner=owner)
    return result


class BottleneckHistory:
    """Read-only metric history of disjoint intervals, with atomic publication."""

    def __init__(self):
        self._states = {}
        self.reports = []

    def project(self, record, result, step, intervals, origins):
        """Validate the next projection without advancing request history."""
        from simllm.core.completion import sampled_request_ids

        states = dict(self._states)
        sampled = sampled_request_ids(record)
        requests = []
        for scheduled in record.scheduled:
            rid = scheduled.request_id
            accounted, pending, ttft, decode, count = states.get(rid, (origins[rid], ranking(), None, ranking(), 0))
            gap = record.virtual_time_ps - accounted
            if gap < 0:
                raise ValueError("bottleneck history moved backward")
            interval = combine(pending, ranking((("batching-queue", None, gap),)), intervals[rid])
            accounted = record.virtual_time_ps + intervals[rid].span_ps
            if rid not in sampled:
                states[rid] = accounted, interval, ttft, decode, count
                continue
            if ttft is None:
                ttft = interval
            else:
                decode = combine(decode, interval)
                count += 1
            requests.append(RequestBottleneck(rid, ttft, combine(decode, divisor=count) if count else None))
            states[rid] = accounted, ranking(), ttft, decode, count
        report = BottleneckReport(record.step_index, step, tuple(sorted(requests, key=lambda r: r.request_id)))
        report.validate_result(result)
        return report, states

    def commit(self, projection):
        report, states = projection
        self._states = states
        self.reports.append(report)
        return report

    def consume(self, record, result, step, intervals, origins):
        return self.commit(self.project(record, result, step, intervals, origins))


def _fraction_json(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _kernel_json(k):
    payload = {f.name: getattr(k, f.name) for f in fields(k)}
    payload["config"] = [[a, b] for a, b in k.config]
    payload["bound_class"] = k.bound_class
    payload["achieved_fraction"] = k.achieved_fraction
    return payload


def _ranking_json(r):
    return {"span_ps": r.span_ps, "divisor": r.divisor,
            "classes": [{"class_name": c.class_name, "participant_width": c.participant_width,
                         "span_ps": c.span_ps, "share": _fraction_json(Fraction(c.span_ps, r.span_ps))} for c in r.classes],
            "kernels": [_kernel_json(k) for k in r.kernels],
            "flow_tails": [{**{f.name: getattr(t, f.name) for f in fields(t)},
                            "tail_share": _fraction_json(t.tail_share)} for t in r.flow_tails]}


def bottleneck_report_to_json(report):
    report.__post_init__()
    return {"schema": SCHEMA, "step_index": report.step_index, "step": _ranking_json(report.step),
            "requests": [{"request_id": r.request_id, "ttft": _ranking_json(r.ttft),
                          "tpot": None if r.tpot is None else _ranking_json(r.tpot)} for r in report.requests]}


def _record(value, path, names):
    value = _object(value, path)
    _fields(value, path, required=set(names))
    return value


def _fraction_read(value, path):
    value = _record(value, path, ("numerator", "denominator"))
    n = _integer(value["numerator"], path + ".numerator", nonnegative=True)
    d = _integer(value["denominator"], path + ".denominator", minimum=1)
    result = Fraction(n, d)
    if result.numerator != n or result.denominator != d:
        raise ValueError("fraction must be reduced")
    return result


def _kernel_read(value, path):
    row = dict(_record(value, path, [f.name for f in fields(KernelEvidence)] + ["bound_class", "achieved_fraction"]))
    bound, achieved = row.pop("bound_class"), row.pop("achieved_fraction")
    config = []
    for item in _array(row["config"], path + ".config"):
        pair = _array(item, path + ".config[]")
        if len(pair) != 2:
            raise ValueError("config entry must have two elements")
        config.append((_string(pair[0], path + ".key"), _scalar(pair[1], path + ".value")))
    row["config"] = tuple(config)
    kernel = KernelEvidence(**row)
    _string(bound, path + ".bound_class")
    if achieved is not None:
        _number(achieved, path + ".achieved_fraction", nonnegative=True)
    if bound != kernel.bound_class or achieved != kernel.achieved_fraction:
        raise ValueError("declared kernel classification disagrees with work and roof")
    return kernel


def _ranking_read(value, path):
    row = _record(value, path, ("span_ps", "divisor", "classes", "kernels", "flow_tails"))
    span = _integer(row["span_ps"], path + ".span_ps", nonnegative=True)
    classes = []
    for item in _array(row["classes"], path + ".classes"):
        item = dict(_record(item, path + ".class", ("class_name", "participant_width", "span_ps", "share")))
        share = _fraction_read(item.pop("share"), path + ".share")
        entry = ClassShare(**item)
        if not span or share != Fraction(entry.span_ps, span):
            raise ValueError("class share disagrees with span")
        classes.append(entry)
    tails = []
    for item in _array(row["flow_tails"], path + ".flow_tails"):
        item = dict(_record(item, path + ".tail", [f.name for f in fields(FlowTail)] + ["tail_share"]))
        share = _fraction_read(item.pop("tail_share"), path + ".tail_share")
        tail = FlowTail(**item)
        if share != tail.tail_share:
            raise ValueError("flow tail share disagrees with FCT range")
        tails.append(tail)
    return BottleneckRanking(span, _integer(row["divisor"], path + ".divisor", minimum=1), tuple(classes),
                             tuple(_kernel_read(k, path + ".kernel") for k in _array(row["kernels"], path + ".kernels")), tuple(tails))


def bottleneck_report_from_json(value):
    row = _record(value, "bottleneck", ("schema", "step_index", "step", "requests"))
    if _string(row["schema"], "bottleneck.schema") != SCHEMA:
        raise ValueError("unsupported bottleneck report schema")
    requests = []
    for item in _array(row["requests"], "bottleneck.requests"):
        item = _record(item, "bottleneck.request", ("request_id", "ttft", "tpot"))
        requests.append(RequestBottleneck(_string(item["request_id"], "request_id"),
                                         _ranking_read(item["ttft"], "ttft"),
                                         None if item["tpot"] is None else _ranking_read(item["tpot"], "tpot")))
    return BottleneckReport(_integer(row["step_index"], "step_index", nonnegative=True),
                            _ranking_read(row["step"], "step"), tuple(requests))


def bottleneck_step_to_json(result, report=None):
    from simllm.core.step_io import step_result_to_json
    payload = step_result_to_json(result)
    if report is not None:
        report.validate_result(result)
        payload["bottleneck_report"] = bottleneck_report_to_json(report)
    return payload


def bottleneck_step_from_json(value):
    from simllm.core.step_io import step_result_from_json
    payload = dict(_object(value, "step"))
    present = "bottleneck_report" in payload
    report = bottleneck_report_from_json(payload.pop("bottleneck_report")) if present else None
    result = step_result_from_json(payload)
    if report is not None:
        report.validate_result(result)
    return result, report
