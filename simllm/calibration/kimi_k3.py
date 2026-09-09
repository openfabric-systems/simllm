"""Kimi K3 native contract checks and inventories projected from one graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from simllm.backends.kimi_k3_lowerer import KimiK3Lowerer, KimiK3LowererConfig
from simllm.compute.device_model import ShapeAxis, ShapeSchema, ShapeVector
from simllm.compute.kimi_k3 import MXFP4_GROUP32, KimiK3Spec
from simllm.core.execution import ComputeWork, ExecutionGraph, ExecutionOperation
from simllm.core.step import StepRecord, step_record_to_json

from .canonical import canonical_sha256, sha256_bytes
from .graph_identity import execution_graph_template_record, unbound_execution_graph_record
from .model_inventory import (
    AbsentPhysicalIdentity,
    FrameworkIdentity,
    ImplementationIdentityEnvelope,
    InventoryCase,
    InventorySuiteIdentity,
    KernelFamilyDefinition,
    KernelProjection,
    ModelCheckpointIdentity,
    ModelKernelInventory,
    PhaseLaunchCount,
)


def exact_config_ratio(value: object) -> str:
    if type(value) not in (int, float):
        raise TypeError("native numeric configuration requires a finite number")
    return str(Fraction(str(value)))


def validate_kimi_k3_native_contract(
    text: Any, native_layers: tuple[str, ...], *, quant_method: str
) -> str:
    """Check the supported native policy without constructing its geometry."""
    if text.model_type != "kimi_linear" or tuple(text.architectures or ()) != ("KimiLinearForCausalLM",):
        raise ValueError("native K3 text architecture changed")
    if text.num_nextn_predict_layers != 0 or text.moe_layer_freq != 1:
        raise ValueError("K3 speculative depth or expert cadence is undeclared")
    if text.use_grouped_topk is not True:
        raise ValueError("K3 requires the declared grouped expert selection")
    linear = text.linear_attn_config
    if not isinstance(linear, dict):
        raise TypeError("native K3 linear attention configuration must be an object")
    declared = []
    for key, kind in (("kda_layers", "kda"), ("full_attn_layers", "mla")):
        values = linear[key]
        if not isinstance(values, list) or any(type(i) is not int for i in values):
            raise ValueError("native K3 layer lists require original one-based integer indices")
        if len(values) != len(set(values)) or values != sorted(values):
            raise ValueError("native K3 layer lists must be sorted without duplicates")
        declared.extend((i - 1, kind) for i in values)
    if [i for i, _ in sorted(declared)] != list(range(text.num_hidden_layers)):
        raise ValueError("native K3 layer partition is not total and disjoint")
    if tuple(kind for _, kind in sorted(declared)) != native_layers:
        raise ValueError("native K3 layer dispatch disagrees with its declared partition")
    quantization = text.quantization_config
    if not isinstance(quantization, dict):
        raise TypeError("native K3 quantization must be an object")
    if quantization.get("quant_method") != quant_method or quantization.get("format") != "mxfp4-pack-quantized":
        raise ValueError("native K3 checkpoint encoding is undeclared")
    groups = quantization.get("config_groups")
    if not isinstance(groups, dict) or set(groups) != {"group_0"}:
        raise ValueError("native K3 requires one declared packed weight group")
    group = groups["group_0"]
    if group.get("targets") != ["Linear"] or group.get("format") != "mxfp4-pack-quantized":
        raise ValueError("native K3 packed weight target changed")
    weights = group.get("weights", {})
    expected = {"num_bits": 4, "group_size": 32, "type": "float", "scale_dtype": "torch.uint8",
                "strategy": "group", "symmetric": True, "dynamic": False}
    if any(weights.get(key) != value or type(weights.get(key)) is not type(value)
           for key, value in expected.items()):
        raise ValueError("native K3 packed value or scale geometry changed")
    ignored = [r"re:.*self_attn.*", r"re:.*shared_experts.*",
               r"re:.*mlp\.(gate|up|gate_up|down)_proj.*", r"re:.*lm_head.*",
               r"re:.*vision_tower.*", r"re:.*mm_projector.*"]
    if quantization.get("ignore") != ignored:
        raise ValueError("native K3 high-precision weight exceptions changed")
    return MXFP4_GROUP32



def validate_kimi_k3_suite(suite: Mapping[str, Any]) -> None:
    """Reject unsupported K3 modes before publishing any derived records."""
    if set(suite) != {"schema", "suite", "state", "reference_model", "frameworks",
                      "phase_scope", "base_parallelism", "graph_cells"}:
        raise ValueError("K3 suite has an undeclared configuration surface")
    if suite["base_parallelism"] != {"tensor": 1, "pipeline": 1, "data": 1, "expert": 1} or any(
        type(value) is not int for value in suite["base_parallelism"].values()
    ):
        raise ValueError("K3 distributed rank projection is outside the declared model")
    scope = {"mode": "text-only", "included_phases": ["prefill", "decode"],
             "optional_components": [],
             "excluded_components": ["vision", "projector", "multi-token-prediction",
                                     "prefix-prefill", "mixed-phase-batch", "distributed-rank-projection"],
             "unsupported_mechanism_policy": "reject-undeclared-dispatch"}
    if suite["phase_scope"] != scope:
        raise ValueError("K3 phase or multimodal dispatch is undeclared")
    model = suite["reference_model"]
    if (model.get("architecture"), model.get("model_type"), model["dtype"], model["quantization"]) != (
        "KimiK3ForConditionalGeneration", "kimi_k3", "bfloat16", MXFP4_GROUP32
    ):
        raise ValueError("K3 suite architecture or encoding does not match its geometry")
    for cell in suite["graph_cells"]:
        if cell["family"] not in {"compute-prefill", "memory-decode", "dense-batch-decode"} or any(
            name in cell for name in ("parallelism_override", "expert_participants")
        ):
            raise ValueError("K3 cell requests an undeclared distributed rank projection")


def operator_family(operation: ExecutionOperation) -> str:
    work = operation.work
    if not isinstance(work, ComputeWork) or work.scope != "logical-operator":
        raise ValueError("K3 inventory requires exclusively logical compute regions")
    name = work.kernel
    config = dict(work.config)
    # A snapshot and a prefix reset change invocation work. Keep them in
    # separate shape families instead of averaging unlike regions together.
    for key in ("snapshot", "reset_prefix", "variant"):
        if key in config:
            name += f"/{key}={int(config[key]) if type(config[key]) is bool else config[key]}"
    return name


def operator_axes(operation: ExecutionOperation) -> tuple[tuple[str, int], ...]:
    work = operation.work
    if not isinstance(work, ComputeWork):
        raise TypeError("operator axes require compute work")
    return tuple((name, int(value)) for name, value in work.config if type(value) in (int, bool))


def project_kimi_k3_graphs(records: tuple[StepRecord, ...], spec: KimiK3Spec,
                          framework: str) -> tuple[ExecutionGraph, ...]:
    lowerer = KimiK3Lowerer(KimiK3LowererConfig(spec, framework))
    return tuple(lowerer.lower(record) for record in records)


def build_kimi_k3_inventory(*, suite_raw: bytes, suite: Mapping[str, Any],
                           model: ModelCheckpointIdentity, framework: FrameworkIdentity,
                           records: tuple[StepRecord, ...]) -> ModelKernelInventory:
    if not isinstance(model.geometry, KimiK3Spec):
        raise TypeError("K3 inventory requires heterogeneous checkpoint geometry")
    graphs = project_kimi_k3_graphs(records, model.geometry, framework.framework_id)
    grouped: list[dict[str, list[ExecutionOperation]]] = []
    family_axes: dict[str, tuple[str, ...]] = {}
    for graph in graphs:
        by_family: dict[str, list[ExecutionOperation]] = defaultdict(list)
        for operation in graph.operations:
            family = operator_family(operation)
            by_family[family].append(operation)
            axes = operator_axes(operation)
            names = tuple(name for name, _ in axes)
            if family in family_axes and family_axes[family] != names:
                raise ValueError("a K3 family changed its invocation-axis vocabulary")
            family_axes[family] = names
        for family, operations in by_family.items():
            shapes = {operator_axes(operation) for operation in operations}
            if len(shapes) != 1:
                raise ValueError(f"K3 family {family!r} mixes unlike invocation shapes")
        grouped.append(dict(by_family))
    families = tuple(sorted(family_axes))
    schemas = []
    definitions = []
    for family in families:
        names = ("logical_visits", *family_axes[family])
        vectors = []
        for by_family in grouped:
            operations = by_family.get(family, ())
            vectors.append((len(operations), *(value for _, value in operator_axes(operations[0])))
                           if operations else (0,) * len(names))
        schema_id = f"simllm-{family}-logical-shape-v1"
        schemas.append(ShapeSchema(schema_id, tuple(
            ShapeAxis(name, "logical-visits" if name == "logical_visits" else
                      "tokens" if "tokens" in name else "pairs" if name == "pairs" else "elements",
                      min(vector[i] for vector in vectors), max(vector[i] for vector in vectors))
            for i, name in enumerate(names))))
        definitions.append(KernelFamilyDefinition(family, schema_id,
                           (PhaseLaunchCount("prefill", 1), PhaseLaunchCount("decode", 1)),
                           launch_scale_axis="logical_visits"))
    cases = []
    cells = suite["graph_cells"]
    for record, graph, by_family, cell in zip(records, graphs, grouped, cells, strict=True):
        projections = []
        for family, schema in zip(families, schemas, strict=True):
            operations = by_family.get(family, ())
            values = ((len(operations), *(value for _, value in operator_axes(operations[0])))
                      if operations else (0,) * len(schema.axes))
            flops = [op.work.flops for op in operations]
            aggregate = None if any(value is None for value in flops) else sum(flops)
            projections.append(KernelProjection(
                family, ShapeVector(schema.shape_schema_id, values), len(operations), aggregate,
                None if operations else 0, scope="logical-operator"))
        if sum(projection.logical_launch_count for projection in projections) != len(graph.operations):
            raise ValueError("K3 inventory lost or duplicated logical graph visits")
        cases.append(InventoryCase(
            case_id=cell["id"], family=cell["family"], phase=cell["phase"], split=cell["split"],
            suite_case_sha256=canonical_sha256(cell),
            step_record_sha256=canonical_sha256(step_record_to_json(record)),
            instance_graph_sha256=unbound_execution_graph_record(graph).record_id,
            template_graph_sha256=execution_graph_template_record(graph).record_id,
            kernel_projections=tuple(projections),
        ))
    inventory = ModelKernelInventory(
        suite=InventorySuiteIdentity(suite["suite"], sha256_bytes(suite_raw), len(cases)),
        framework=framework, model=model, shape_schemas=tuple(schemas), kernel_families=tuple(definitions),
        cases=tuple(cases), implementation_identity=ImplementationIdentityEnvelope(
            code_object_hashes=AbsentPhysicalIdentity(), observed_launches=AbsentPhysicalIdentity(),
            join_tasks=tuple(sorted(("COMP-6", "COMP-59", "VLLM-12" if framework.framework_id == "vllm" else "SGL-10"))),
        ),
    )
    ModelKernelInventory.from_obj(inventory.to_obj())
    return inventory
