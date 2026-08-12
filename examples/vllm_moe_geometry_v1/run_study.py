"""Run the frozen VLLM-6 MoE geometry and expert group qualification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

EVIDENCE_AUTHORED_AGAINST = "aeb40ac95cdd8163942297335948c94df0376e04"
VLLM_AUTHORED_AGAINST = "568afb3a13806beb53bb2e6bd518269357b237c0"

GRANITE_INTERMEDIATE_SIZE = 512
GRANITE_TOP_K = 8
GRANITE_EXPERTS = 32

#: cell -> (dp, pcp, tp, enable_expert_parallel, rank, num_local_experts,
#:          num_redundant_experts, pp)
GEOMETRY_INPUTS = {
    "g-dense-world1": (1, 1, 1, False, 0, 32, 0, 1),
    "g-ep-off-tp2": (1, 1, 2, False, 1, 32, 0, 1),
    "g-ep-off-dp2-tp2": (2, 1, 2, False, 3, 32, 0, 1),
    "g-ep-flag-world1": (1, 1, 1, True, 0, 32, 0, 1),
    "g-ep-dp8": (8, 1, 1, True, 5, 32, 0, 1),
    "g-ep-dp2-tp2": (2, 1, 2, True, 3, 32, 0, 1),
    "g-ep-dp2-pcp2-tp2": (2, 2, 2, True, 7, 32, 0, 1),
    "g-ep-dp8-uneven": (8, 1, 1, True, 0, 30, 0, 1),
    "g-ep-dp8-uneven-hi": (8, 1, 1, True, 7, 30, 0, 1),
    "g-ep-dp2-eplb": (2, 1, 1, True, 1, 32, 2, 1),
    "llama-dense": (1, 1, 1, False, 0, None, 0, 1),
    "llama-dense-tp2": (1, 1, 2, False, 1, None, 0, 1),
}

#: cell -> (num_experts, top_k, moe_intermediate_size, local_num_experts)
EXPECTED_GEOMETRY = {
    "g-dense-world1": (32, 8, 512, 32),
    "g-ep-off-tp2": (32, 8, 256, 32),
    "g-ep-off-dp2-tp2": (32, 8, 128, 32),
    "g-ep-flag-world1": (32, 8, 512, 32),
    "g-ep-dp8": (32, 8, 512, 4),
    "g-ep-dp2-tp2": (32, 8, 512, 8),
    "g-ep-dp2-pcp2-tp2": (32, 8, 512, 4),
    "g-ep-dp8-uneven": (30, 8, 512, 4),
    "g-ep-dp8-uneven-hi": (30, 8, 512, 3),
    "g-ep-dp2-eplb": (34, 8, 512, 17),
    "llama-dense": (0, 0, None, 0),
    "llama-dense-tp2": (0, 0, None, 0),
}

#: cell -> expected EP group of the cell's rank, None when the model is dense
EXPECTED_GEOMETRY_EP_RANKS = {
    "g-dense-world1": (0,),
    "g-ep-off-tp2": (0, 1),
    "g-ep-off-dp2-tp2": (0, 1, 2, 3),
    "g-ep-flag-world1": (0,),
    "g-ep-dp8": tuple(range(8)),
    "g-ep-dp2-tp2": (0, 1, 2, 3),
    "g-ep-dp2-pcp2-tp2": tuple(range(8)),
    "g-ep-dp8-uneven": tuple(range(8)),
    "g-ep-dp8-uneven-hi": tuple(range(8)),
    "g-ep-dp2-eplb": (0, 1),
    "llama-dense": None,
    "llama-dense-tp2": None,
}

#: cell -> (external_dp, dp, pp, pcp, tp, rank)
LAYOUT_INPUTS = {
    "layout-dp2-tp2": (1, 2, 1, 1, 2, 3),
    "layout-dp2-pp2-tp2": (1, 2, 2, 1, 2, 6),
    "layout-dp2-pp2-tp2-lo": (1, 2, 2, 1, 2, 1),
    "layout-dp2-pcp2-tp2": (1, 2, 1, 2, 2, 5),
    "layout-extdp2-dp2-tp2": (2, 2, 1, 1, 2, 5),
}

#: cell -> (ep_ranks, ep_rank)
EXPECTED_LAYOUT = {
    "layout-dp2-tp2": ((0, 1, 2, 3), 3),
    "layout-dp2-pp2-tp2": ((2, 3, 6, 7), 2),
    "layout-dp2-pp2-tp2-lo": ((0, 1, 4, 5), 1),
    "layout-dp2-pcp2-tp2": (tuple(range(8)), 5),
    "layout-extdp2-dp2-tp2": ((4, 5, 6, 7), 1),
}

#: cell -> (field, pre-change value, post-change value)
EXPECTED_DIRECTIONS = (
    ("g-ep-dp2-tp2", "moe_intermediate_size", 256, 512),
    ("g-ep-dp2-tp2", "local_num_experts", 16, 8),
    ("g-ep-off-dp2-tp2", "moe_intermediate_size", 256, 128),
)

EXPECTED_SCORED_INSTANCES = 22
EXPECTED_FATAL_GUARDS = ("default-and-dense-behavior", "vllm22-schedule-identity")


def _reference_ep_group(
    external_dp: int, dp: int, pp: int, pcp: int, tp: int, rank: int
) -> tuple[tuple[int, ...], int]:
    """The frozen closed form for vLLM's ExternalDP x DP x PP x PCP x TP layout."""

    stride = pcp * tp
    block = dp * pp * stride
    base = (rank // block) * block
    within = rank % block
    pipeline_stage = (within // stride) % pp
    members = tuple(
        base + ((index * pp + pipeline_stage) * pcp + context) * tp + tensor
        for index in range(dp)
        for context in range(pcp)
        for tensor in range(tp)
    )
    return members, members.index(rank)


def _reference_geometry(
    dp: int,
    pcp: int,
    tp: int,
    enable_expert_parallel: bool,
    rank: int,
    num_local_experts: int | None,
    num_redundant_experts: int,
    pp: int,
) -> tuple[int, int, int | None, int]:
    """The frozen closed form for the pinned vLLM v0.26.0 MoE geometry."""

    if not num_local_experts:
        return 0, 0, None, 0
    num_experts = num_local_experts + num_redundant_experts
    flatten_tp = dp * pcp * tp
    use_ep = enable_expert_parallel and flatten_tp > 1
    ep_size = flatten_tp if use_ep else 1
    moe_tp_size = 1 if use_ep else flatten_tp
    moe_intermediate = GRANITE_INTERMEDIATE_SIZE // moe_tp_size
    _, ep_rank = _reference_ep_group(1, dp, pp, pcp, tp, rank)
    base, remainder = divmod(num_experts, ep_size)
    local = base + 1 if ep_rank < remainder else base
    return num_experts, GRANITE_TOP_K, moe_intermediate, local


def _check_frozen_registry() -> None:
    if set(GEOMETRY_INPUTS) != set(EXPECTED_GEOMETRY):
        raise AssertionError("geometry cell registry is incomplete")
    if set(GEOMETRY_INPUTS) != set(EXPECTED_GEOMETRY_EP_RANKS):
        raise AssertionError("geometry EP group registry is incomplete")
    if set(LAYOUT_INPUTS) != set(EXPECTED_LAYOUT):
        raise AssertionError("layout cell registry is incomplete")

    for cell, inputs in GEOMETRY_INPUTS.items():
        derived = _reference_geometry(*inputs)
        if derived != EXPECTED_GEOMETRY[cell]:
            raise AssertionError(
                f"frozen geometry row {cell} disagrees with its own closed form: "
                f"{EXPECTED_GEOMETRY[cell]} against {derived}"
            )
        num_experts, _, _, local = derived
        expected_group = EXPECTED_GEOMETRY_EP_RANKS[cell]
        if num_experts == 0:
            if expected_group is not None:
                raise AssertionError(f"dense cell {cell} must have no EP group")
            continue
        if expected_group is None:
            raise AssertionError(f"MoE cell {cell} must have an EP group")
        dp, pcp, tp = inputs[0], inputs[1], inputs[2]
        pp, rank = inputs[7], inputs[4]
        members, _ = _reference_ep_group(1, dp, pp, pcp, tp, rank)
        if members != expected_group:
            raise AssertionError(f"frozen EP group row {cell} drifted")
        if not 1 <= local <= num_experts:
            raise AssertionError(f"cell {cell} violates the local expert floor")

    # sum over EP ranks of local_num_experts == num_experts, wherever expert
    # parallelism is actually in use; without it every rank holds every expert
    # and the experts are tensor-sharded instead
    for cell, inputs in GEOMETRY_INPUTS.items():
        num_experts = EXPECTED_GEOMETRY[cell][0]
        dp, pcp, tp = inputs[0], inputs[1], inputs[2]
        pp = inputs[7]
        if num_experts == 0 or not (inputs[3] and dp * pcp * tp > 1):
            continue
        members, _ = _reference_ep_group(1, dp, pp, pcp, tp, inputs[4])
        total = sum(
            _reference_geometry(
                dp, pcp, tp, inputs[3], member, inputs[5], inputs[6], pp
            )[3]
            for member in members
        )
        if total != num_experts:
            raise AssertionError(
                f"cell {cell} loses experts across its EP group: {total} of "
                f"{num_experts}"
            )

    for cell, inputs in LAYOUT_INPUTS.items():
        members, index = _reference_ep_group(*inputs)
        if (members, index) != EXPECTED_LAYOUT[cell]:
            raise AssertionError(
                f"frozen layout row {cell} disagrees with its own closed form: "
                f"{EXPECTED_LAYOUT[cell]} against {(members, index)}"
            )
        if len(members) != inputs[1] * inputs[3] * inputs[4]:
            raise AssertionError(f"layout row {cell} has the wrong EP world size")
        if len(set(members)) != len(members):
            raise AssertionError(f"layout row {cell} repeats a rank")

    for cell, field, before, after in EXPECTED_DIRECTIONS:
        index = {"moe_intermediate_size": 2, "local_num_experts": 3}[field]
        if EXPECTED_GEOMETRY[cell][index] != after:
            raise AssertionError(f"direction row {cell}/{field} disagrees with F1")
        if before == after:
            raise AssertionError(f"direction row {cell}/{field} claims no change")

    if EXPECTED_SCORED_INSTANCES != (
        len(GEOMETRY_INPUTS) + len(LAYOUT_INPUTS) + len(EXPECTED_DIRECTIONS) + 2
    ):
        raise AssertionError("scored instance arithmetic drifted")
    if len(EXPECTED_FATAL_GUARDS) != 2:
        raise AssertionError("fatal guard registry drifted")
    # the weight-byte companion stated in the freeze
    if 512 * 8 != 256 * 16:
        raise AssertionError("weight-byte companion arithmetic drifted")


def check_only(args: argparse.Namespace) -> None:
    _check_frozen_registry()
    print(
        f"check-only out={args.out}; validated {len(GEOMETRY_INPUTS)} geometry "
        f"cells, {len(LAYOUT_INPUTS)} layout cells and "
        f"{EXPECTED_SCORED_INSTANCES} scored instances, and produced no artifacts"
    )



# --- production run ---------------------------------------------------------
#
# Everything above this line is the frozen expectation. Everything below reads
# the implementation and reports what it observed. F5 runs first, because a
# defaulting or dense regression voids the run rather than scoring as a cell
# miss, and only then are F1, F2, F3 and F4 scored.


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _observed_provenance() -> dict:
    import subprocess

    def _git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=_repository_root(),
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception:  # noqa: BLE001 - provenance is best effort
            return None
        return completed.stdout.strip() or None

    return {
        "repository_commit": _git("rev-parse", "HEAD"),
        "repository_dirty": bool(_git("status", "--porcelain")),
        "evidence_authored_against": EVIDENCE_AUTHORED_AGAINST,
        "vllm_authored_against": VLLM_AUTHORED_AGAINST,
    }


class _TextConfig(SimpleNamespace):
    pass


def _model_config(num_local_experts: int | None, layers: int = 24):
    text = _TextConfig(
        model_type="granitemoe" if num_local_experts else "llama",
        architectures=(
            ["GraniteMoeForCausalLM"] if num_local_experts else ["LlamaForCausalLM"]
        ),
        intermediate_size=GRANITE_INTERMEDIATE_SIZE,
    )
    if num_local_experts is not None:
        text.num_local_experts = num_local_experts
        text.num_experts_per_tok = GRANITE_TOP_K

    class _ModelConfig:
        dtype = SimpleNamespace(itemsize=2)
        hf_text_config = text
        max_model_len = 4_096
        is_moe = bool(num_local_experts)

        @staticmethod
        def get_hidden_size() -> int:
            return 1_024

        @staticmethod
        def get_num_layers(parallel_config) -> int:
            return layers

        @staticmethod
        def get_num_attention_heads(parallel_config) -> int:
            return 16

        @staticmethod
        def get_num_kv_heads(parallel_config) -> int:
            return 8

        @staticmethod
        def get_head_size() -> int:
            return 64

        @staticmethod
        def get_vocab_size() -> int:
            return 49_155

    return _ModelConfig()


def _vllm_config(inputs: tuple):
    dp, pcp, tp, enable_ep, rank, experts, redundant, pp = inputs
    return SimpleNamespace(
        model_config=_model_config(experts),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            rank=rank,
            enable_expert_parallel=enable_ep,
            eplb_config=SimpleNamespace(num_redundant_experts=redundant),
        ),
        quant_config=None,
    )


def _layout_vllm_config(inputs: tuple):
    external_dp, dp, pp, pcp, tp, rank = inputs
    del external_dp
    return SimpleNamespace(
        model_config=_model_config(GRANITE_EXPERTS),
        cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            rank=rank,
            enable_expert_parallel=True,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        quant_config=None,
    )


def _pre_change_reader(inputs: tuple) -> tuple[int | None, int]:
    """The mapping this branch replaces, reproduced for the direction claim.

    It divided the per-expert intermediate size by ``tensor_parallel_size``
    whatever the expert-parallel state, and sized the expert-parallel world
    from ``data_parallel_size`` alone.
    """

    dp, _, tp, enable_ep, _, experts, _, _ = inputs
    if not experts:
        return None, 0
    ep_size = max(dp, 1) if enable_ep else 1
    return max(GRANITE_INTERMEDIATE_SIZE // tp, 1), experts // ep_size


class _RecordingSink:
    """An expert-group-capable sink that records what the executor bound."""

    def __init__(self) -> None:
        self.bound: list[tuple[int, ...]] = []

    def bind_clock(self, clock) -> None:
        del clock

    def bind_expert_group(self, ep_ranks) -> None:
        self.bound.append(tuple(ep_ranks))

    def __call__(self, record, observations):
        raise AssertionError("this study never executes a step")


def _fatal_guards() -> list[dict]:
    from simllm.adapters.vllm.executor import (
        expert_group_ranks,
        model_dims_from_vllm_config,
    )

    findings: list[dict] = []

    dense = _vllm_config(GEOMETRY_INPUTS["llama-dense"])
    dims = model_dims_from_vllm_config(dense)
    findings.append(
        {
            "guard": "default-and-dense-behavior",
            "case": "dense config yields no MoE geometry and no EP group",
            "held": (
                dims.num_experts == 0
                and dims.top_k == 0
                and dims.moe_intermediate_size is None
                and dims.local_num_experts == 0
                and expert_group_ranks(dense) is None
            ),
        }
    )

    broken = _vllm_config(GEOMETRY_INPUTS["g-ep-dp8"])
    broken.model_config.hf_text_config = _TextConfig(
        model_type="granitemoe",
        architectures=["GraniteMoeForCausalLM"],
        num_local_experts=32,
        num_experts_per_tok=8,
    )
    broken_dims = model_dims_from_vllm_config(broken)
    findings.append(
        {
            "guard": "default-and-dense-behavior",
            "case": "a missing accessor is stamped rather than raised",
            "held": "intermediate_size" in broken_dims.defaulted_fields,
        }
    )
    return findings


def _schedule_identity() -> list[dict]:
    """F6: the accepted VLLM-22 Granite schedule and serial-off identities."""

    from simllm.adapters.vllm import (
        VllmBatchSlice,
        build_granite_execution_observations,
    )
    from simllm.compute import GPU_ENVELOPES, HostInitiationModel, RooflineProvider
    from simllm.core import CollectiveWork, RequestPhase, ScheduledRequest, StepRecord

    dims = _granite_model_dims()
    record = StepRecord(
        step_index=0,
        virtual_time_ps=0,
        scheduled=[
            ScheduledRequest(f"r{index}", RequestPhase.DECODE, 1, context_length=8)
            for index in range(3)
        ],
        num_sampled=3,
    )
    observations, _ = build_granite_execution_observations(
        record,
        dims,
        tuple(range(8)),
        (VllmBatchSlice(None, ("r0", "r1", "r2"), 3),),
        RooflineProvider(),
        GPU_ENVELOPES["b100"],
        HostInitiationModel(initiation_delay_ps=0),
    )
    collectives = [
        operation
        for operation in observations.operations
        if isinstance(operation.work, CollectiveWork)
    ]
    return [
        {
            "guard": "vllm22-schedule-identity",
            "case": "24 layers x 2 phases of semantic MoE sites",
            "held": len(collectives) == 48,
        },
        {
            "guard": "vllm22-schedule-identity",
            "case": "every routed site is a zero-byte semantic marker",
            "held": all(
                work.payload_bytes == 0 and not work.pair_payload_bytes
                for work in (operation.work for operation in collectives)
            ),
        },
        {
            "guard": "vllm22-schedule-identity",
            "case": "one visibility frontier for the unbatched slice",
            "held": observations.completion_operation_ids
            == ("step-0:ubatch-0:requests-visible",),
        },
    ]


def _granite_model_dims():
    from simllm.compute import ModelDims

    return ModelDims(
        num_layers=24,
        hidden_size=1_024,
        intermediate_size=512,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        vocab_size=49_155,
        dtype_bytes=2,
        num_experts=GRANITE_EXPERTS,
        top_k=GRANITE_TOP_K,
        moe_intermediate_size=512,
        local_num_experts=4,
    )


def _score_geometry() -> list[dict]:
    from simllm.adapters.vllm.executor import (
        expert_group_ranks,
        model_dims_from_vllm_config,
    )

    rows: list[dict] = []
    for cell, inputs in GEOMETRY_INPUTS.items():
        config = _vllm_config(inputs)
        dims = model_dims_from_vllm_config(config)
        observed = (
            dims.num_experts,
            dims.top_k,
            dims.moe_intermediate_size,
            dims.local_num_experts,
        )
        group = expert_group_ranks(config)
        rows.append(
            {
                "family": "F1",
                "cell": cell,
                "expected": list(EXPECTED_GEOMETRY[cell]),
                "observed": list(observed),
                "expected_ep_ranks": (
                    None
                    if EXPECTED_GEOMETRY_EP_RANKS[cell] is None
                    else list(EXPECTED_GEOMETRY_EP_RANKS[cell])
                ),
                "observed_ep_ranks": None if group is None else list(group),
                "passed": (
                    observed == EXPECTED_GEOMETRY[cell]
                    and (group if group is None else tuple(group))
                    == EXPECTED_GEOMETRY_EP_RANKS[cell]
                ),
            }
        )
    return rows


def _score_layout() -> list[dict]:
    from simllm.adapters.vllm.executor import expert_parallel_geometry

    rows: list[dict] = []
    for cell, inputs in LAYOUT_INPUTS.items():
        geometry = expert_parallel_geometry(_layout_vllm_config(inputs))
        observed = (geometry.ep_ranks, geometry.ep_rank)
        rows.append(
            {
                "family": "F2",
                "cell": cell,
                "expected": [list(EXPECTED_LAYOUT[cell][0]), EXPECTED_LAYOUT[cell][1]],
                "observed": [list(observed[0]), observed[1]],
                "passed": observed == EXPECTED_LAYOUT[cell],
            }
        )
    return rows


def _score_directions() -> list[dict]:
    from simllm.adapters.vllm.executor import model_dims_from_vllm_config

    rows: list[dict] = []
    for cell, field, frozen_before, frozen_after in EXPECTED_DIRECTIONS:
        inputs = GEOMETRY_INPUTS[cell]
        before_intermediate, before_local = _pre_change_reader(inputs)
        before = (
            before_intermediate if field == "moe_intermediate_size" else before_local
        )
        dims = model_dims_from_vllm_config(_vllm_config(inputs))
        after = getattr(dims, field)
        rows.append(
            {
                "family": "F3",
                "cell": f"{cell}/{field}",
                "expected": [frozen_before, frozen_after],
                "observed": [before, after],
                "passed": before == frozen_before and after == frozen_after,
            }
        )
    return rows


def _score_binding() -> list[dict]:
    from simllm.adapters.vllm.executor import (
        SimExecutor,
        expert_group_ranks,
        expert_parallel_geometry,
    )

    rows: list[dict] = []
    for cell, expect_bound in (("g-ep-dp8", True), ("g-ep-flag-world1", False)):
        config = _vllm_config(GEOMETRY_INPUTS[cell])
        sink = _RecordingSink()
        executor = SimExecutor.__new__(SimExecutor)
        executor.vllm_config = config
        executor.step_sink = sink
        executor.expert_parallel = expert_parallel_geometry(config)
        executor.ep_ranks = expert_group_ranks(config)
        executor._bind_expert_group()
        observed = tuple(sink.bound)
        expected = (
            (tuple(EXPECTED_GEOMETRY_EP_RANKS[cell]),) if expect_bound else ()
        )
        rows.append(
            {
                "family": "F4",
                "cell": cell,
                "expected": [list(entry) for entry in expected],
                "observed": [list(entry) for entry in observed],
                "passed": observed == expected,
            }
        )
    return rows


def _pinned_parallel_arm(interpreter: str) -> dict:
    """Re-derive the parallel-side cells against the real pinned ParallelConfig."""

    import json
    import subprocess

    payload = json.dumps(
        {
            "geometry": {cell: list(inputs) for cell, inputs in GEOMETRY_INPUTS.items()},
            "layout": {cell: list(inputs) for cell, inputs in LAYOUT_INPUTS.items()},
        }
    )
    script = _repository_root() / "examples/vllm_moe_geometry_v1/pinned_probe.py"
    completed = subprocess.run(
        [interpreter, str(script)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
        cwd=_repository_root(),
        env={
            "PYTHONPATH": str(_repository_root()),
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "VLLM_LOGGING_LEVEL": "ERROR",
        },
    )
    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip().splitlines()[-3:],
        }
    return json.loads(completed.stdout)


def production(args: argparse.Namespace) -> None:
    import json

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fatal = _fatal_guards() + _schedule_identity()
    void = [finding for finding in fatal if not finding["held"]]

    scored = (
        _score_geometry() + _score_layout() + _score_directions() + _score_binding()
    )
    passed = sum(1 for row in scored if row["passed"])

    pinned = (
        _pinned_parallel_arm(args.vllm_python)
        if args.vllm_python
        else {"available": False, "error": ["no --vllm-python given"]}
    )

    report = {
        "study": "vllm_moe_geometry_v1",
        "task": "VLLM-6",
        "provenance": _observed_provenance(),
        "fatal_guards": fatal,
        "void": bool(void),
        "scored_rows": scored,
        "scored": None if void else f"{passed}/{len(scored)}",
        "scored_expected_denominator": EXPECTED_SCORED_INSTANCES,
        "pinned_parallel_arm": pinned,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if void:
        print("VOID: fatal guard violated")
        for finding in void:
            print(f"  {finding['guard']}: {finding['case']}")
    else:
        print(f"scored {passed}/{len(scored)}")
    for row in scored:
        if not row["passed"]:
            print(f"  MISS {row['family']} {row['cell']}: {row}")
    print(f"report written to {out / 'report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="vllm_moe_geometry_v1")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--vllm-python",
        default=None,
        help="interpreter with the pinned vLLM v0.26.0 installed",
    )
    args = parser.parse_args()
    if args.check_only:
        check_only(args)
        return
    _check_frozen_registry()
    production(args)


if __name__ == "__main__":
    main()
