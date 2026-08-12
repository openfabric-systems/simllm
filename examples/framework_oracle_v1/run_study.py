"""Run the frozen serving-framework CPU oracle study."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).parents[2]

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445"
MODEL_RELATIVE_PATH = (
    Path("hub")
    / "models--ibm-granite--granite-3.0-1b-a400m-instruct"
    / "snapshots"
    / MODEL_REVISION
)
VLLM_VERSION = "0.26.0"
VLLM_AUTHORED_AGAINST = "568afb3a13806beb53bb2e6bd518269357b237c0"
VLLM_SOURCE_ARCHIVE_SHA256 = (
    "c1ded7e5af7e7fd27c6abcce1ecfaf795bc927a0f747f10f1e5a7a13f2c4a6a9"
)
SGLANG_AUTHORED_AGAINST = "8f2a3ad6d7d68c58ae65b61a75bb2115449addca"
SEED = 173
VECTOR_BYTES = 2_048
MATERIAL_CHANGED_BYTES = 2 * VECTOR_BYTES
PRESSURE_CAPACITIES = (64, 256)
EXPECTED_SCORED = 8

COMPARISON_REQUESTS = (
    {
        "request_id": "eos-brief",
        "prompt": "Reply with exactly one word: OK",
        "max_new_tokens": 16,
        "stop_strings": (),
        "source_rank": 0,
    },
    {
        "request_id": "length-cap",
        "prompt": "Continue this sequence with ten more integers: 1 2 3",
        "max_new_tokens": 1,
        "stop_strings": (),
        "source_rank": 1,
    },
    {
        "request_id": "stop-string",
        "prompt": "Reply with exactly SIMLLM_STOP and no other text",
        "max_new_tokens": 16,
        "stop_strings": ("SIMLLM_STOP",),
        "source_rank": 0,
    },
)
PREFIX_PROMPT = "Summarize in one word: cached framework oracle prefix."
PRESSURE_REQUESTS = tuple(
    {
        "request_id": f"pressure-{letter.lower()}",
        "prompt": f"Count upward slowly from zero for pressure request {letter}:",
        "max_new_tokens": 16,
        "stop_strings": (),
    }
    for letter in "ABCD"
)

VLLM_SOURCE_HASHES = {
    "cmake/cpu_extension.cmake": (
        "5b78bee8804b17ef997876fcc109e21d5e246cfebd48ad029d912462127e3383"
    ),
    "docs/getting_started/installation/cpu.x86.inc.md": (
        "c5b646e00e86b9c903970e38df6ca014d28a05ef73540a493a12578f3710eda5"
    ),
    "vllm/model_executor/layers/fused_moe/cpu_fused_moe.py": (
        "d1a87df2bad22bbedbf7d94ad294a9217674bc121c435306f61839efbeb891f1"
    ),
    "vllm/v1/core/block_pool.py": (
        "202a13cb129174849d798019aaedc04c59775ec2a4b9dfcc7c1e3c563a43a661"
    ),
    "vllm/v1/core/kv_cache_manager.py": (
        "3f4af8d247f3fe9570b0132818b832b66ae6a2ac12942588828f899f6ff77ccf"
    ),
    "vllm/v1/core/sched/scheduler.py": (
        "2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941"
    ),
}
SGLANG_SOURCE_HASHES = {
    "python/sglang/srt/layers/moe/topk.py": (
        "45f61df269e2b17f411cf6a2bede0188cd980a2c5f110bafc653c170f2e639c7"
    ),
    "python/sglang/srt/managers/schedule_batch.py": (
        "7bdd988f3274bb11b3b5fd699b985ae27c28ff9e2e7f601a6320e182efcf1595"
    ),
    "python/sglang/srt/managers/tp_worker.py": (
        "88a352303c7871c1d1c5a54018b951dc960da138a37776a03a55bc18d5804deb"
    ),
    "python/sglang/srt/mem_cache/allocation.py": (
        "940a05f4031c81794f3a8ac89fce60da1da6eebaf4c3fc25228ced77fa042690"
    ),
    "python/sglang/srt/mem_cache/radix_cache.py": (
        "08f04dce327949a48b2f6296cf7e10df59c914eae19df7614e3158242d62f4a6"
    ),
    "python/sglang/srt/models/granitemoe.py": (
        "f1a465f47fa472801ba0c8cefb9227254ea6309273e5ba6394054d329ad03910"
    ),
    "python/sglang/srt/state_capturer/routed_experts.py": (
        "49ef692f7d468f297d9aeef5a30657908d4a353bd8a229be8ebc40266b64c669"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hashes(root: Path, expected: dict[str, str], label: str) -> None:
    for relative, digest in expected.items():
        source = root / relative
        if not source.is_file():
            raise SystemExit(f"{label} source is missing: {relative}")
        if not digest:
            raise SystemExit(f"{label} source hash is not frozen: {relative}")
        observed = _sha256(source)
        if observed != digest:
            raise SystemExit(
                f"{label} source changed: {relative}; "
                f"expected {digest}, observed {observed}"
            )


def _model_source(cache_dir: Path) -> Path:
    direct = cache_dir / MODEL_RELATIVE_PATH
    if direct.is_dir():
        return direct
    alternate = cache_dir / MODEL_RELATIVE_PATH.relative_to("hub")
    if alternate.is_dir():
        return alternate
    raise SystemExit(
        f"pinned model snapshot is missing below configured cache: {MODEL_ID}"
    )


def _python_version(executable: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [str(executable), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(int(part) for part in result.stdout.strip().split("."))


def _gcc_version() -> str:
    result = subprocess.run(
        ["gcc", "-dumpfullversion"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def check_only(args: argparse.Namespace) -> None:
    if args.run_dir.resolve() == REPOSITORY_ROOT.resolve():
        raise SystemExit("run directory must be outside the repository")
    if not args.run_dir.is_absolute():
        raise SystemExit("run directory must be an explicit absolute path")
    _model_source(args.cache_dir)
    _validate_hashes(args.vllm_source, VLLM_SOURCE_HASHES, "vLLM")
    _validate_hashes(args.sglang_source, SGLANG_SOURCE_HASHES, "SGLang")
    if not args.sglang_python.is_file() or not os.access(args.sglang_python, os.X_OK):
        raise SystemExit(f"SGLang Python is missing or not executable: {args.sglang_python}")
    python_version = _python_version(args.sglang_python)
    if python_version < (3, 10):
        raise SystemExit(f"SGLang Python is too old: {python_version}")
    if MATERIAL_CHANGED_BYTES != 4_096:
        raise AssertionError("one dispatch plus combine byte quantum changed")
    if PRESSURE_CAPACITIES != (64, 256):
        raise AssertionError("KV capacity family changed")
    if EXPECTED_SCORED != 3 + 3 + 1 + 1:
        raise AssertionError("behavioral evidence denominator changed")
    if SGLANG_AUTHORED_AGAINST == "":
        raise AssertionError("SGLang authored-against provenance is missing")
    if re.fullmatch(r"[0-9a-f]{40}", SGLANG_AUTHORED_AGAINST) is None:
        raise AssertionError("SGLang authored-against provenance is malformed")
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit("the frozen CPU build ladder is x86-64 only")
    summary = {
        "artifacts_written": 0,
        "gcc": _gcc_version(),
        "material_changed_bytes": MATERIAL_CHANGED_BYTES,
        "model_revision": MODEL_REVISION,
        "python": ".".join(str(part) for part in python_version),
        "run_dir": str(args.run_dir),
        "seed": SEED,
        "vllm_version": VLLM_VERSION,
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))


def _git_head(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", observed) is None:
        raise RuntimeError(f"source returned malformed commit identity: {observed!r}")
    return observed


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _framework_request_from_json(value: dict[str, Any]) -> Any:
    from simllm.preplay import FrameworkOracleRequest, PromptFormat

    return FrameworkOracleRequest(
        request_id=str(value["request_id"]),
        prompt_sha256=str(value["prompt_sha256"]),
        prompt_format=PromptFormat(str(value["prompt_format"])),
        input_token_ids=tuple(int(token_id) for token_id in value["input_token_ids"]),
        max_new_tokens=int(value["max_new_tokens"]),
        stop_strings=tuple(str(item) for item in value["stop_strings"]),
    )


def _run_sglang_child(args: argparse.Namespace) -> None:
    if args.sglang_child_spec is None or args.sglang_child_capacity is None:
        raise SystemExit("SGLang child mode requires a spec and token capacity")
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from simllm.preplay import SglangCpuRunner

    specification = json.loads(args.sglang_child_spec.read_text(encoding="utf-8"))
    groups = tuple(
        tuple(_framework_request_from_json(request) for request in group)
        for group in specification["groups"]
    )
    capacity = args.sglang_child_capacity
    cell_dir = args.run_dir / f"sglang-{capacity}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    runner = SglangCpuRunner(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_path=_model_source(args.cache_dir),
        tokenizer_sha256=str(specification["tokenizer_sha256"]),
        observed_source=_git_head(args.sglang_source),
        authored_against_source=SGLANG_AUTHORED_AGAINST,
        dtype="float32",
        torch_num_threads=8,
        engine_seed=SEED,
    )
    runner.capture_groups(
        groups,
        cell_dir / "framework-trace-v2.jsonl",
        max_total_tokens=capacity,
        context_length=64,
        max_running_requests=4,
        page_size=1,
        overwrite=True,
        observation_path=cell_dir / "sglang-observations.jsonl",
        raw_response_path=cell_dir / "sglang-responses.json",
    )
    _write_json(
        cell_dir / "child-status.json",
        {
            "capacity": capacity,
            "groups": len(groups),
            "request_count": sum(len(group) for group in groups),
            "status": "PASS",
        },
    )


def _run_vllm_child(args: argparse.Namespace) -> None:
    if args.vllm_child_spec is None or args.vllm_child_capacity is None:
        raise SystemExit("vLLM child mode requires a spec and token capacity")
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from simllm.preplay import VllmCpuRunner

    specification = json.loads(args.vllm_child_spec.read_text(encoding="utf-8"))
    groups = tuple(
        tuple(_framework_request_from_json(request) for request in group)
        for group in specification["groups"]
    )
    build = _vllm_build_observation(args)
    capacity = args.vllm_child_capacity
    cell_dir = args.run_dir / f"vllm-{capacity}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    runner = VllmCpuRunner(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_path=_model_source(args.cache_dir),
        tokenizer_sha256=str(specification["tokenizer_sha256"]),
        observed_source=str(build["observed_source"]),
        authored_against_source=VLLM_AUTHORED_AGAINST,
        dtype="float32",
        torch_num_threads=8,
        engine_seed=SEED,
    )
    runner.capture_groups(
        groups,
        cell_dir / "framework-trace-v2.jsonl",
        kv_token_capacity=capacity,
        context_length=64,
        max_running_requests=4,
        block_size=16,
        overwrite=True,
        observation_path=cell_dir / "vllm-observations.jsonl",
        raw_response_path=cell_dir / "vllm-responses.json",
    )
    _write_json(
        cell_dir / "child-status.json",
        {
            "capacity": capacity,
            "groups": len(groups),
            "request_count": sum(len(group) for group in groups),
            "status": "PASS",
        },
    )


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise RuntimeError(f"{path.name} line {line_number} is blank")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path.name} line {line_number} is not an object")
        rows.append(row)
    return rows


def _request_spec(
    *,
    request_id: str,
    prompt: str,
    input_token_ids: list[int],
    max_new_tokens: int,
    stop_strings: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_format": "chat",
        "input_token_ids": input_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop_strings": list(stop_strings),
    }


def _capture_transformers(args: argparse.Namespace) -> tuple[Path, Path]:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from simllm.preplay import (
        PreplayRequest,
        PromptFormat,
        SamplingConfig,
        TransformersCpuRunner,
    )

    trace_path = args.run_dir / "transformers" / "preplay-trace-v1.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    importlib.import_module("torch").manual_seed(SEED)
    runner = TransformersCpuRunner(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        dtype="float32",
        torch_num_threads=8,
    )
    comparison = tuple(
        PreplayRequest(
            request_id=str(row["request_id"]),
            prompt=str(row["prompt"]),
            max_new_tokens=int(row["max_new_tokens"]),
            stop_strings=tuple(str(value) for value in row["stop_strings"]),
            prompt_format=PromptFormat.CHAT,
        )
        for row in COMPARISON_REQUESTS
    )
    runner.capture(
        comparison,
        trace_path,
        sampling=SamplingConfig.greedy(),
        overwrite=True,
    )
    raw_rows = _read_jsonl_rows(trace_path)
    comparison_rows = {
        str(row["request_id"]): row
        for row in raw_rows
        if row.get("row_type") == "request"
    }
    if set(comparison_rows) != {
        str(request["request_id"]) for request in COMPARISON_REQUESTS
    }:
        raise RuntimeError("Transformers raw request identities differ from the freeze")

    groups: list[list[dict[str, Any]]] = []
    groups.append(
        [
            {
                "request_id": row["request_id"],
                "prompt_sha256": comparison_rows[str(row["request_id"])][
                    "prompt_sha256"
                ],
                "prompt_format": comparison_rows[str(row["request_id"])][
                    "prompt_format"
                ],
                "input_token_ids": comparison_rows[str(row["request_id"])][
                    "input_token_ids"
                ],
                "max_new_tokens": row["max_new_tokens"],
                "stop_strings": list(row["stop_strings"]),
            }
            for row in COMPARISON_REQUESTS
        ]
    )

    def encode(prompt: str, request_id: str, max_new_tokens: int) -> dict[str, Any]:
        request = PreplayRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_format=PromptFormat.CHAT,
        )
        input_ids = runner._encode_request(request)
        return _request_spec(
            request_id=request_id,
            prompt=prompt,
            input_token_ids=[int(value) for value in input_ids[0].tolist()],
            max_new_tokens=max_new_tokens,
            stop_strings=(),
        )

    groups.append([encode(PREFIX_PROMPT, "prefix-cold", 1)])
    groups.append([encode(PREFIX_PROMPT, "prefix-warm", 1)])
    groups.append(
        [
            encode(
                str(row["prompt"]),
                str(row["request_id"]),
                int(row["max_new_tokens"]),
            )
            for row in PRESSURE_REQUESTS
        ]
    )
    max_request_tokens = max(
        len(request["input_token_ids"]) + int(request["max_new_tokens"])
        for group in groups
        for request in group
    )
    if max_request_tokens > 64:
        raise RuntimeError(
            f"frozen request needs {max_request_tokens} tokens, above context length 64"
        )
    specification = {
        "groups": groups,
        "tokenizer_sha256": runner.tokenizer_sha256,
    }
    spec_path = args.run_dir / "framework-request-spec.json"
    _write_json(spec_path, specification)
    return trace_path, spec_path


def _run_sglang_cell(
    args: argparse.Namespace,
    *,
    capacity: int,
    spec_path: Path,
) -> None:
    command = [
        str(args.sglang_python),
        str(Path(__file__).resolve()),
        "--cache-dir",
        str(args.cache_dir),
        "--vllm-source",
        str(args.vllm_source),
        "--sglang-source",
        str(args.sglang_source),
        "--sglang-python",
        str(args.sglang_python),
        "--run-dir",
        str(args.run_dir),
        "--sglang-child-capacity",
        str(capacity),
        "--sglang-child-spec",
        str(spec_path),
    ]
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not prior
        else str(REPOSITORY_ROOT) + os.pathsep + prior
    )
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, env=env
    )
    cell_dir = args.run_dir / f"sglang-{capacity}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "child-stdout.log").write_text(result.stdout, encoding="utf-8")
    (cell_dir / "child-stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        _write_json(
            cell_dir / "child-status.json",
            {
                "capacity": capacity,
                "returncode": result.returncode,
                "status": "FAIL",
            },
        )
        failure = result.stderr.strip() or result.stdout.strip() or "no child output"
        raise RuntimeError(
            f"SGLang {capacity}-token cell failed with {result.returncode}: {failure}"
        )


def _run_vllm_cell(
    args: argparse.Namespace,
    *,
    build: dict[str, Any],
    capacity: int,
    spec_path: Path,
) -> None:
    executable = Path(str(build["python"]))
    command = [
        str(executable),
        str(Path(__file__).resolve()),
        "--cache-dir",
        str(args.cache_dir),
        "--vllm-source",
        str(args.vllm_source),
        "--sglang-source",
        str(args.sglang_source),
        "--sglang-python",
        str(args.sglang_python),
        "--run-dir",
        str(args.run_dir),
        "--vllm-child-capacity",
        str(capacity),
        "--vllm-child-spec",
        str(spec_path),
    ]
    env = os.environ.copy()
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not prior
        else str(REPOSITORY_ROOT) + os.pathsep + prior
    )
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, env=env
    )
    cell_dir = args.run_dir / f"vllm-{capacity}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "child-stdout.log").write_text(result.stdout, encoding="utf-8")
    (cell_dir / "child-stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        _write_json(
            cell_dir / "child-status.json",
            {
                "capacity": capacity,
                "returncode": result.returncode,
                "status": "FAIL",
            },
        )
        failure = result.stderr.strip() or result.stdout.strip() or "no child output"
        raise RuntimeError(
            f"vLLM {capacity}-token cell failed with {result.returncode}: {failure}"
        )


def _sglang_responses(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("requests")
    if not isinstance(rows, list):
        raise TypeError("raw SGLang response artifact has no request list")
    responses: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("response"), dict):
            raise TypeError("raw SGLang response row is malformed")
        request_id = str(row.get("request_id"))
        if request_id in responses:
            raise RuntimeError(f"duplicate raw SGLang request {request_id!r}")
        responses[request_id] = row["response"]
    return responses


def _vllm_responses(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("requests")
    if not isinstance(rows, list):
        raise TypeError("raw vLLM response artifact has no request list")
    responses: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("raw vLLM response row is malformed")
        request_id = str(row.get("request_id"))
        if request_id in responses:
            raise RuntimeError(f"duplicate raw vLLM request {request_id!r}")
        responses[request_id] = row
    return responses


def _raw_sglang_stop(response: dict[str, Any], *, eos_token_id: int) -> str:
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise TypeError("raw SGLang response has no meta_info")
    finish = meta_info.get("finish_reason")
    if not isinstance(finish, dict):
        raise TypeError("raw SGLang response has no structured finish reason")
    kind = finish.get("type")
    if kind == "length":
        return "length-cap"
    if kind != "stop":
        raise RuntimeError(f"unsupported raw SGLang finish reason {finish!r}")
    matched = finish.get("matched")
    matched_ids = matched if isinstance(matched, list) else [matched]
    return "eos" if eos_token_id in matched_ids else "stop-string"


def _raw_vllm_stop(response: dict[str, Any], *, eos_token_id: int) -> str:
    finish = response.get("finish_reason")
    if finish == "length":
        return "length-cap"
    if finish != "stop":
        raise RuntimeError(f"unsupported raw vLLM finish reason {finish!r}")
    matched = response.get("stop_reason")
    if isinstance(matched, str):
        return "stop-string"
    if matched not in (None, eos_token_id):
        raise RuntimeError(f"unsupported raw vLLM stop token {matched!r}")
    return "eos"


def _decode_sglang_routing(
    response: dict[str, Any],
    *,
    token_count: int,
) -> tuple[int, ...]:
    meta_info = response.get("meta_info")
    encoded = None if not isinstance(meta_info, dict) else meta_info.get("routed_experts")
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("raw SGLang response has no routed experts")
    values = array("i")
    values.frombytes(base64.b64decode(encoded, validate=True))
    if sys.byteorder != "little":
        values.byteswap()
    expected = token_count * 24 * 8
    if len(values) != expected:
        raise RuntimeError(
            f"raw SGLang route count {len(values)} differs from expected {expected}"
        )
    return tuple(int(value) for value in values)


def _raw_transformers_projection(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str, int, int], tuple[int, tuple[int, ...]]],
    dict[str, Any],
]:
    headers = [row for row in rows if row.get("row_type") == "header"]
    if len(headers) != 1 or not isinstance(headers[0].get("provenance"), dict):
        raise RuntimeError("raw Transformers trace has no unique header")
    requests = {
        str(row["request_id"]): row
        for row in rows
        if row.get("row_type") == "request"
    }
    routes: dict[tuple[str, str, int, int], tuple[int, tuple[int, ...]]] = {}
    for row in rows:
        if row.get("row_type") != "forward-token":
            continue
        for layer in row["routing"]:
            key = (
                str(row["request_id"]),
                str(row["phase"]),
                int(row["token_index"]),
                int(layer["layer_index"]),
            )
            if key in routes:
                raise RuntimeError(f"duplicate raw Transformers routing key {key!r}")
            routes[key] = (
                int(row["token_id"]),
                tuple(int(value) for value in layer["expert_ids"]),
            )
    return requests, routes, headers[0]["provenance"]


def _raw_sglang_projection(
    responses: dict[str, dict[str, Any]],
    request_specs: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, int, int], tuple[int, tuple[int, ...]]]:
    routes: dict[tuple[str, str, int, int], tuple[int, tuple[int, ...]]] = {}
    for request_id, response in responses.items():
        if request_id not in request_specs:
            raise RuntimeError(f"SGLang returned unknown request {request_id!r}")
        specification = request_specs[request_id]
        input_ids = [int(value) for value in specification["input_token_ids"]]
        output_ids = [int(value) for value in response["output_ids"]]
        forwarded = input_ids + output_ids[:-1]
        flattened = _decode_sglang_routing(response, token_count=len(forwarded))
        offset = 0
        for flat_index, token_id in enumerate(forwarded):
            if flat_index < len(input_ids):
                phase = "prefill"
                token_index = flat_index
            else:
                phase = "decode"
                token_index = flat_index - len(input_ids)
            for layer_index in range(24):
                experts = flattened[offset : offset + 8]
                offset += 8
                key = (request_id, phase, token_index, layer_index)
                routes[key] = (token_id, experts)
    return routes


def _raw_vllm_projection(
    responses: dict[str, dict[str, Any]],
    request_specs: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, int, int], tuple[int, tuple[int, ...]]]:
    routes: dict[tuple[str, str, int, int], tuple[int, tuple[int, ...]]] = {}
    for request_id, response in responses.items():
        if request_id not in request_specs:
            raise RuntimeError(f"vLLM returned unknown request {request_id!r}")
        specification = request_specs[request_id]
        input_ids = [int(value) for value in specification["input_token_ids"]]
        output_ids = [int(value) for value in response["output_token_ids"]]
        forwarded = input_ids + output_ids[:-1]
        shape = tuple(int(value) for value in response["routed_experts_shape"])
        expected_shape = (len(forwarded), 24, 8)
        if shape != expected_shape:
            raise RuntimeError(
                f"raw vLLM route shape {shape} differs from {expected_shape}"
            )
        flattened = tuple(int(value) for value in response["routed_experts"])
        if len(flattened) != shape[0] * shape[1] * shape[2]:
            raise RuntimeError("raw vLLM route payload disagrees with its shape")
        offset = 0
        for flat_index, token_id in enumerate(forwarded):
            if flat_index < len(input_ids):
                phase = "prefill"
                token_index = flat_index
            else:
                phase = "decode"
                token_index = flat_index - len(input_ids)
            for layer_index in range(24):
                experts = flattened[offset : offset + 8]
                offset += 8
                routes[(request_id, phase, token_index, layer_index)] = (
                    token_id,
                    experts,
                )
    return routes


def _traffic_bytes(experts: tuple[int, ...], source_rank: int) -> int:
    remote_rank = 1 - source_rank
    has_remote_destination = any(expert_id // 16 == remote_rank for expert_id in experts)
    return MATERIAL_CHANGED_BYTES if has_remote_destination else 0


def _evaluate_comparisons(
    *,
    transformer_rows: list[dict[str, Any]],
    framework_responses: dict[str, dict[str, Any]],
    framework: str,
    specification: dict[str, Any],
) -> dict[str, Any]:
    transformer_requests, transformer_routes, provenance = (
        _raw_transformers_projection(transformer_rows)
    )
    request_specs = {
        str(request["request_id"]): request
        for group in specification["groups"]
        for request in group
    }
    framework_routes = (
        _raw_vllm_projection(framework_responses, request_specs)
        if framework == "vllm"
        else _raw_sglang_projection(framework_responses, request_specs)
    )
    outcome_rows = []
    routing_rows = []
    taxonomy_total: Counter[str] = Counter()
    exact_routing_rows = 0
    compared_routing_rows = 0
    eos_token_id = int(provenance["eos_token_id"])

    for frozen in COMPARISON_REQUESTS:
        request_id = str(frozen["request_id"])
        baseline = transformer_requests[request_id]
        response = framework_responses[request_id]
        baseline_output = tuple(int(value) for value in baseline["output_token_ids"])
        output_field = "output_token_ids" if framework == "vllm" else "output_ids"
        framework_output = tuple(int(value) for value in response[output_field])
        baseline_stop = str(baseline["stop_reason"])
        framework_stop = (
            _raw_vllm_stop(response, eos_token_id=eos_token_id)
            if framework == "vllm"
            else _raw_sglang_stop(response, eos_token_id=eos_token_id)
        )
        outcome_pass = (
            framework_output == baseline_output and framework_stop == baseline_stop
        )
        outcome_rows.append(
            {
                "request_id": request_id,
                "transformers_output_token_ids": list(baseline_output),
                "framework_output_token_ids": list(framework_output),
                "transformers_length": len(baseline_output),
                "framework_length": len(framework_output),
                "transformers_stop_reason": baseline_stop,
                "framework_stop_reason": framework_stop,
                "status": "PASS" if outcome_pass else "FAIL",
            }
        )

        relevant_keys = sorted(
            {
                key[1:]
                for key in set(transformer_routes) | set(framework_routes)
                if key[0] == request_id
            },
            key=lambda value: (value[0] != "prefill", value[1], value[2]),
        )
        taxonomy: Counter[str] = Counter()
        changed_bytes = 0
        alignment_failed = False
        causal_context_changed = False
        for suffix in relevant_keys:
            key = (request_id, *suffix)
            baseline_route = transformer_routes.get(key)
            framework_route = framework_routes.get(key)
            phase, token_index, _layer_index = suffix
            causal_difference = (
                phase == "decode"
                and baseline_output[: token_index + 1]
                != framework_output[: token_index + 1]
            )
            compared_routing_rows += 1
            if baseline_route is None or framework_route is None:
                classification = (
                    "output-cascade" if causal_difference else "unaligned"
                )
                causal_context_changed = (
                    causal_context_changed or causal_difference
                )
                alignment_failed = alignment_failed or not causal_difference
            else:
                baseline_token, baseline_experts = baseline_route
                framework_token, framework_experts = framework_route
                if baseline_token != framework_token:
                    classification = (
                        "output-cascade" if causal_difference else "unaligned"
                    )
                    causal_context_changed = causal_context_changed or causal_difference
                    alignment_failed = alignment_failed or not causal_difference
                elif baseline_experts == framework_experts:
                    exact_routing_rows += 1
                    continue
                elif causal_difference:
                    classification = "output-cascade"
                    causal_context_changed = True
                elif set(baseline_experts) == set(framework_experts):
                    classification = "order-only"
                else:
                    baseline_bytes = _traffic_bytes(
                        baseline_experts, int(frozen["source_rank"])
                    )
                    framework_bytes = _traffic_bytes(
                        framework_experts, int(frozen["source_rank"])
                    )
                    if baseline_bytes == framework_bytes:
                        classification = "expert-id-only"
                    else:
                        classification = "byte-changing"
                        changed_bytes += abs(framework_bytes - baseline_bytes)
            taxonomy[classification] += 1
            taxonomy_total[classification] += 1
        routing_pass = (
            changed_bytes == 0
            and not alignment_failed
            and not causal_context_changed
        )
        routing_rows.append(
            {
                "request_id": request_id,
                "changed_all_to_all_bytes": changed_bytes,
                "material": changed_bytes >= MATERIAL_CHANGED_BYTES,
                "status": "PASS" if routing_pass else "FAIL",
                "taxonomy": dict(sorted(taxonomy.items())),
            }
        )

    classified = sum(taxonomy_total.values())
    if exact_routing_rows + classified != compared_routing_rows:
        raise RuntimeError("routing taxonomy did not cover every compared row")
    return {
        "outcomes": outcome_rows,
        "routing": routing_rows,
        "routing_diagnostics": {
            "classified_difference_rows": classified,
            "compared_rows": compared_routing_rows,
            "exact_rows": exact_routing_rows,
            "exact_fraction": (
                0.0 if compared_routing_rows == 0 else exact_routing_rows / compared_routing_rows
            ),
            "taxonomy": dict(sorted(taxonomy_total.items())),
        },
    }


def _captured_sidecar_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl_rows(path)
    markers = [index for index, row in enumerate(rows) if row.get("kind") == "capture-start"]
    if len(markers) != 1:
        raise RuntimeError(f"{path.name} does not contain one capture-start marker")
    return rows[markers[0] + 1 :]


def _event_count_by_request(
    rows: list[dict[str, Any]],
    kind: str,
    internal_to_logical: dict[str, str] | None = None,
) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        if row.get("kind") != kind or row.get("request_id") is None:
            continue
        observed = str(row["request_id"])
        request_id = (
            observed
            if internal_to_logical is None
            else internal_to_logical.get(observed)
        )
        if request_id is None:
            raise RuntimeError(f"unmapped framework request ID {observed!r}")
        result[request_id] += 1
    return result


def _prefix_tokens_by_request(
    rows: list[dict[str, Any]],
    internal_to_logical: dict[str, str] | None = None,
) -> dict[str, int]:
    matches: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "prefix-hit" or row.get("request_id") is None:
            continue
        observed = str(row["request_id"])
        request_id = (
            observed
            if internal_to_logical is None
            else internal_to_logical.get(observed)
        )
        if request_id is None:
            raise RuntimeError(f"unmapped framework request ID {observed!r}")
        matches[request_id].append(int(row["token_count"]))
    return {
        request_id: (values[0] if internal_to_logical is not None else max(values))
        for request_id, values in matches.items()
    }


def _prefix_observations_by_request(
    rows: list[dict[str, Any]],
    internal_to_logical: dict[str, str] | None = None,
) -> dict[str, list[int]]:
    result: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get("kind") != "prefix-hit" or row.get("request_id") is None:
            continue
        observed = str(row["request_id"])
        request_id = (
            observed
            if internal_to_logical is None
            else internal_to_logical.get(observed)
        )
        if request_id is None:
            raise RuntimeError(f"unmapped framework request ID {observed!r}")
        result[request_id].append(int(row["token_count"]))
    return dict(result)


def _vllm_request_mapping(rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        if row.get("kind") != "request-mapping":
            continue
        for mapping in row.get("mappings", []):
            internal = str(mapping["internal_request_id"])
            if internal in result:
                raise RuntimeError(f"duplicate vLLM request mapping {internal!r}")
            result[internal] = str(mapping["request_id"])
    if not result:
        raise RuntimeError("vLLM sidecar has no request mappings")
    return result


def _kv_cell_observation(
    args: argparse.Namespace, capacity: int, framework: str
) -> dict[str, Any]:
    cell_dir = args.run_dir / f"{framework}-{capacity}"
    sidecar = cell_dir / f"{framework}-observations.jsonl"
    all_rows = _read_jsonl_rows(sidecar)
    rows = _captured_sidecar_rows(sidecar)
    internal_to_logical = _vllm_request_mapping(rows) if framework == "vllm" else None
    pressure_ids = [str(row["request_id"]) for row in PRESSURE_REQUESTS]
    pressure_markers = [
        index
        for index, row in enumerate(rows)
        if row.get("kind") == "submission-group-start"
        and row.get("request_ids") == pressure_ids
    ]
    if len(pressure_markers) != 1:
        raise RuntimeError(
            f"{cell_dir.name} does not contain one pressure submission marker"
        )
    pressure_rows = rows[pressure_markers[0] + 1 :]
    responses = (
        _vllm_responses(cell_dir / "vllm-responses.json")
        if framework == "vllm"
        else _sglang_responses(cell_dir / "sglang-responses.json")
    )
    prefix_tokens = _prefix_tokens_by_request(rows, internal_to_logical)
    prefix_observations = _prefix_observations_by_request(
        rows, internal_to_logical
    )
    preemptions = _event_count_by_request(
        rows, "preemption", internal_to_logical
    )
    pressure_preemptions = _event_count_by_request(
        pressure_rows, "preemption", internal_to_logical
    )
    evictions = [row for row in pressure_rows if row.get("kind") == "eviction"]
    final_preemptions: dict[str, int] = {}
    if framework == "vllm":
        assert internal_to_logical is not None
        for row in rows:
            if row.get("kind") != "request-final-counters":
                continue
            observed = str(row["request_id"])
            request_id = internal_to_logical.get(observed)
            if request_id is None:
                raise RuntimeError(f"unmapped final counter {observed!r}")
            final_preemptions[request_id] = int(row["num_preemptions"])
        manager_rows = [
            row for row in all_rows if row.get("kind") == "kv-manager-qualified"
        ]
        if len(manager_rows) != 1:
            raise RuntimeError("vLLM sidecar has no unique KV manager row")
        page_size = int(manager_rows[0]["block_size"])
        observed_capacity = int(manager_rows[0]["token_capacity"])
    else:
        page_size = 1
        observed_capacity = capacity
    reconciliation = []
    for request_id, response in sorted(responses.items()):
        if framework == "vllm":
            cached_tokens = int(response["num_cached_tokens"])
            framework_preemptions = final_preemptions[request_id]
        else:
            meta_info = response.get("meta_info")
            if not isinstance(meta_info, dict):
                raise TypeError(f"response {request_id!r} has no meta_info")
            cached_tokens = int(meta_info["cached_tokens"])
            framework_preemptions = int(meta_info["num_retractions"])
        observed_cached_tokens = prefix_tokens.get(request_id, 0)
        observed_preemptions = preemptions.get(request_id, 0)
        reconciliation.append(
            {
                "request_id": request_id,
                "response_cached_tokens": cached_tokens,
                "event_cached_tokens": observed_cached_tokens,
                "response_preemptions": framework_preemptions,
                "event_preemptions": observed_preemptions,
                "status": (
                    "PASS"
                    if cached_tokens == observed_cached_tokens
                    and framework_preemptions == observed_preemptions
                    else "FAIL"
                ),
            }
        )
    return {
        "capacity": capacity,
        "observed_capacity": observed_capacity,
        "page_size": page_size,
        "prefix_tokens": prefix_tokens,
        "prefix_observations": prefix_observations,
        "eviction_events": len(evictions),
        "evicted_tokens": sum(int(row["token_count"]) for row in evictions),
        "preemption_events": sum(pressure_preemptions.values()),
        "preemptions_by_request": dict(sorted(pressure_preemptions.items())),
        "reconciliation": reconciliation,
    }


def _evaluate_kv(
    args: argparse.Namespace,
    framework: str,
    specification: dict[str, Any],
) -> dict[str, Any]:
    cells = {
        capacity: _kv_cell_observation(args, capacity, framework)
        for capacity in PRESSURE_CAPACITIES
    }
    low = cells[64]
    high = cells[256]
    warm_tokens = int(high["prefix_tokens"].get("prefix-warm", 0))
    cold_tokens = int(high["prefix_tokens"].get("prefix-cold", 0))
    requests = {
        str(request["request_id"]): request
        for group in specification["groups"]
        for request in group
    }
    prompt_tokens = len(requests["prefix-warm"]["input_token_ids"])
    page_size = int(high["page_size"])
    minimum_warm_tokens = ((prompt_tokens - 1) // page_size) * page_size
    prefix_pass = (
        cold_tokens == 0
        and warm_tokens > cold_tokens
        and warm_tokens >= minimum_warm_tokens
    )
    pressure_pass = (
        int(low["eviction_events"]) >= 1
        and int(low["preemption_events"]) >= 1
        and int(high["preemption_events"]) == 0
        and int(high["eviction_events"]) <= int(low["eviction_events"])
    )
    reconciliation_pass = all(
        row["status"] == "PASS"
        for cell in cells.values()
        for row in cell["reconciliation"]
    )
    return {
        "cells": {str(capacity): cell for capacity, cell in cells.items()},
        "prefix_relation": {
            "cold_tokens": cold_tokens,
            "minimum_warm_tokens": minimum_warm_tokens,
            "prompt_tokens": prompt_tokens,
            "warm_tokens": warm_tokens,
            "status": "PASS" if prefix_pass else "FAIL",
        },
        "pressure_relation": {
            "low_eviction_events": low["eviction_events"],
            "low_preemption_events": low["preemption_events"],
            "high_eviction_events": high["eviction_events"],
            "high_preemption_events": high["preemption_events"],
            "status": "PASS" if pressure_pass else "FAIL",
        },
        "event_counter_reconciliation": (
            "PASS" if reconciliation_pass else "FAIL"
        ),
    }


def _strict_guards(
    args: argparse.Namespace,
    *,
    framework: str,
    transformer_trace_path: Path,
    kv_result: dict[str, Any],
) -> dict[str, Any]:
    from simllm.preplay import (
        read_framework_preplay_trace,
        read_preplay_trace,
        write_framework_preplay_trace,
    )

    _validate_hashes(args.vllm_source, VLLM_SOURCE_HASHES, "vLLM")
    _validate_hashes(args.sglang_source, SGLANG_SOURCE_HASHES, "SGLang")
    transformer_trace = read_preplay_trace(transformer_trace_path)
    canonical_cells = {}
    for capacity in PRESSURE_CAPACITIES:
        source = args.run_dir / f"{framework}-{capacity}" / "framework-trace-v2.jsonl"
        trace = read_framework_preplay_trace(source)
        copy = args.run_dir / f"{framework}-{capacity}" / "canonical-roundtrip.jsonl"
        write_framework_preplay_trace(copy, trace, overwrite=True)
        canonical_cells[str(capacity)] = {
            "byte_identical": copy.read_bytes() == source.read_bytes(),
            "event_count": len(trace.kv_events),
            "kv_token_capacity": trace.provenance.kv_token_capacity,
            "request_count": len(trace.requests),
            "runner": trace.provenance.runner,
            "framework_version": trace.provenance.framework_version,
            "observed_source": trace.provenance.observed_source,
            "authored_against_source": trace.provenance.authored_against_source,
        }
    fixture = REPOSITORY_ROOT / "examples/preplay_trace_v1/writer_golden.jsonl"
    frozen_fixture = subprocess.run(
        ["git", "show", f"2bba046:{fixture.relative_to(REPOSITORY_ROOT)}"],
        check=True,
        capture_output=True,
    ).stdout
    fixture_preserved = fixture.read_bytes() == frozen_fixture
    canonical_pass = all(
        bool(cell["byte_identical"])
        and int(cell["kv_token_capacity"]) == int(capacity)
        and cell["runner"] == f"{framework}-cpu"
        for capacity, cell in canonical_cells.items()
    )
    reconciliation_pass = kv_result["event_counter_reconciliation"] == "PASS"
    framework_qualification = _framework_qualification_guard(args, framework)
    qualification_pass = framework_qualification["status"] == "PASS"
    return {
        "source_hashes": "PASS",
        "framework_qualification": framework_qualification,
        "strict_v1_read": {
            "request_count": len(transformer_trace.requests),
            "status": "PASS",
        },
        "strict_v2_reads_and_canonical_roundtrips": {
            "cells": canonical_cells,
            "status": "PASS" if canonical_pass else "FAIL",
        },
        "v1_frozen_fixture": {
            "bytes": len(frozen_fixture),
            "status": "PASS" if fixture_preserved else "FAIL",
        },
        "event_counter_reconciliation": (
            "PASS" if reconciliation_pass else "FAIL"
        ),
        "status": (
            "PASS"
            if canonical_pass
            and fixture_preserved
            and reconciliation_pass
            and qualification_pass
            else "FAIL"
        ),
    }


def _framework_qualification_guard(
    args: argparse.Namespace, framework: str
) -> dict[str, Any]:
    cells = {}
    for capacity in PRESSURE_CAPACITIES:
        sidecar = (
            args.run_dir
            / f"{framework}-{capacity}"
            / f"{framework}-observations.jsonl"
        )
        rows = _read_jsonl_rows(sidecar)
        worker_rows = [row for row in rows if row.get("kind") == "worker-qualified"]
        if len(worker_rows) != 1:
            raise RuntimeError(
                f"{framework}-{capacity} has no unique worker qualification"
            )
        worker = worker_rows[0]
        if framework == "vllm":
            before = worker.get("cuda_memory_allocated_before")
            after = worker.get("cuda_memory_allocated_after")
            cell_pass = (
                worker.get("worker_class") == "CPUWorker"
                and worker.get("model_runner_class") == "CPUModelRunner"
                and worker.get("model_class") == "GraniteMoeForCausalLM"
                and worker.get("parameter_devices") == ["cpu"]
                and type(worker.get("cuda_available_before")) is bool
                and type(worker.get("cuda_available_after")) is bool
                and worker.get("cuda_available_after")
                == worker.get("cuda_available_before")
                and type(before) is int
                and type(after) is int
                and before >= 0
                and after == before
            )
            cells[str(capacity)] = {
                "cuda_available_after": worker.get("cuda_available_after"),
                "cuda_available_before": worker.get("cuda_available_before"),
                "cuda_memory_allocated_after": after,
                "cuda_memory_allocated_before": before,
                "model_class": worker.get("model_class"),
                "model_runner_class": worker.get("model_runner_class"),
                "parameter_devices": worker.get("parameter_devices"),
                "status": "PASS" if cell_pass else "FAIL",
                "worker_class": worker.get("worker_class"),
            }
        else:
            storage_rows = [
                row
                for row in rows
                if row.get("kind") == "capture-storage-qualified"
            ]
            cell_pass = (
                worker.get("worker_class") == "TpModelWorker"
                and worker.get("model_runner_class") == "ModelRunner"
                and worker.get("model_class") == "GraniteMoeForCausalLM"
                and worker.get("parameter_devices") == ["cpu"]
                and bool(storage_rows)
                and all(
                    row.get("device") == "cpu" and row.get("pinned") is False
                    for row in storage_rows
                )
            )
            cells[str(capacity)] = {
                "capture_storage_rows": len(storage_rows),
                "model_class": worker.get("model_class"),
                "model_runner_class": worker.get("model_runner_class"),
                "parameter_devices": worker.get("parameter_devices"),
                "status": "PASS" if cell_pass else "FAIL",
                "worker_class": worker.get("worker_class"),
            }

    build = None
    build_pass = True
    if framework == "vllm":
        build = _vllm_build_observation(args)
        import_qualification = build.get("import_qualification", {})
        runtime_qualification = build.get("runtime_qualification", {})
        build_pass = (
            build.get("status") == "qualified"
            and import_qualification.get("platform") == "CpuPlatform"
            and import_qualification.get("device") == "cpu"
            and import_qualification.get("operator_present_after_cpu_platform_selection")
            is True
            and runtime_qualification.get("worker_class") == "CPUWorker"
            and runtime_qualification.get("model_runner_class") == "CPUModelRunner"
            and runtime_qualification.get("model_class")
            == "GraniteMoeForCausalLM"
            and runtime_qualification.get("device") == "cpu"
        )
    cells_pass = all(cell["status"] == "PASS" for cell in cells.values())
    return {
        "build": (
            None
            if build is None
            else {
                "distribution_version": build.get("distribution_version"),
                "operator_present": build["import_qualification"].get(
                    "operator_present_after_cpu_platform_selection"
                ),
                "platform": build["import_qualification"].get("platform"),
                "status": "PASS" if build_pass else "FAIL",
                "target_device": build.get("target_device"),
            }
        ),
        "cells": cells,
        "status": "PASS" if build_pass and cells_pass else "FAIL",
    }


def _vllm_build_observation(args: argparse.Namespace) -> dict[str, Any]:
    path = args.run_dir / "vllm-build-observation.json"
    if not path.is_file():
        raise RuntimeError(
            "vLLM build observation is missing; finish the isolated source-build ladder"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") not in {"blocked", "qualified"}:
        raise RuntimeError("vLLM build observation has no valid status")
    if value["status"] == "qualified":
        executable = Path(str(value.get("python", "")))
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError("qualified vLLM observation names no executable Python")
        try:
            executable.absolute().relative_to(args.run_dir.absolute())
        except ValueError as exc:
            raise RuntimeError(
                "qualified vLLM Python is not inside the isolated run directory"
            ) from exc
        if value.get("vllm_version") != VLLM_VERSION:
            raise RuntimeError("qualified vLLM observation has the wrong version")
        if value.get("distribution_version") != f"{VLLM_VERSION}+cpu":
            raise RuntimeError(
                "qualified vLLM observation is not the expected CPU distribution"
            )
        if value.get("target_device") != "cpu":
            raise RuntimeError("qualified vLLM observation is not a CPU build")
        if value.get("source_archive_sha256") != VLLM_SOURCE_ARCHIVE_SHA256:
            raise RuntimeError("qualified vLLM observation has the wrong source archive")
        if value.get("authored_against_source") != VLLM_AUTHORED_AGAINST:
            raise RuntimeError(
                "qualified vLLM observation has the wrong authored-against source"
            )
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(value.get("wheel_sha256", "")))
            is None
        ):
            raise RuntimeError("qualified vLLM observation has no wheel checksum")
        import_qualification = value.get("import_qualification")
        if not isinstance(import_qualification, dict) or (
            import_qualification.get("platform") != "CpuPlatform"
            or import_qualification.get("device") != "cpu"
            or import_qualification.get(
                "operator_present_after_cpu_platform_selection"
            )
            is not True
        ):
            raise RuntimeError("qualified vLLM import observation is incomplete")
        for field in ("observed_source", "authored_against_source"):
            if re.fullmatch(r"[0-9a-f]{40}", str(value.get(field, ""))) is None:
                raise RuntimeError(f"qualified vLLM observation has invalid {field}")
    return value


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir.resolve() == REPOSITORY_ROOT.resolve():
        raise SystemExit("run directory must be outside the repository")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    vllm_build = _vllm_build_observation(args)
    framework = "vllm" if vllm_build["status"] == "qualified" else "sglang"
    transformer_trace_path, spec_path = _capture_transformers(args)
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    for capacity in PRESSURE_CAPACITIES:
        if framework == "vllm":
            _run_vllm_cell(
                args,
                build=vllm_build,
                capacity=capacity,
                spec_path=spec_path,
            )
        else:
            _run_sglang_cell(args, capacity=capacity, spec_path=spec_path)

    transformer_rows = _read_jsonl_rows(transformer_trace_path)
    response_path = (
        args.run_dir
        / f"{framework}-256"
        / f"{framework}-responses.json"
    )
    framework_responses = (
        _vllm_responses(response_path)
        if framework == "vllm"
        else _sglang_responses(response_path)
    )

    # Frozen scored relations are evaluated from the raw artifacts first.
    comparisons = _evaluate_comparisons(
        transformer_rows=transformer_rows,
        framework_responses=framework_responses,
        framework=framework,
        specification=specification,
    )
    kv = _evaluate_kv(args, framework, specification)
    scored_rows = [*comparisons["outcomes"], *comparisons["routing"]]
    scored_rows.extend([kv["prefix_relation"], kv["pressure_relation"]])
    if len(scored_rows) != EXPECTED_SCORED:
        raise RuntimeError(
            f"scored denominator changed: {len(scored_rows)} != {EXPECTED_SCORED}"
        )
    scored_passes = sum(row["status"] == "PASS" for row in scored_rows)

    # Fatal source, schema, conservation, and canonicality gates follow scoring.
    guards = _strict_guards(
        args,
        framework=framework,
        transformer_trace_path=transformer_trace_path,
        kv_result=kv,
    )
    result = {
        "chronology": {
            "expectation_commit": "2bba046",
            "raw_relations_evaluated_before_fatal_guards": True,
            "post_run_unscored_guard_correction": (
                "vLLM cached-token reconciliation uses the first prefill, "
                "matching RequestOutput semantics; later post-preemption "
                "prefix observations remain in the event ledger"
            ),
        },
        "configuration": {
            "capacities": list(PRESSURE_CAPACITIES),
            "material_changed_bytes": MATERIAL_CHANGED_BYTES,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": SEED,
        },
        "framework": {
            "selected": framework,
            "vllm_build": vllm_build,
            **(
                {
                    "vllm_observed_source": vllm_build["observed_source"],
                    "vllm_authored_against_source": VLLM_AUTHORED_AGAINST,
                }
                if framework == "vllm"
                else {
                    "sglang_observed_source": _git_head(args.sglang_source),
                    "sglang_authored_against_source": SGLANG_AUTHORED_AGAINST,
                }
            ),
        },
        "scored": {
            "executed": EXPECTED_SCORED,
            "passed": scored_passes,
            "status": "PASS" if scored_passes == EXPECTED_SCORED else "FAIL",
        },
        "comparisons": comparisons,
        "kv": kv,
        "fatal_unscored_guards": guards,
        "native_executables": {},
    }
    _write_json(args.run_dir / "summary.json", result)
    print(json.dumps(result["scored"], sort_keys=True, separators=(",", ":")))
    if guards["status"] != "PASS":
        raise RuntimeError("one or more fatal unscored guards failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, required=True)
    parser.add_argument("--sglang-source", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--sglang-child-capacity", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sglang-child-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--vllm-child-capacity", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--vllm-child-spec", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.vllm_child_capacity is not None or args.vllm_child_spec is not None:
        _run_vllm_child(args)
        return
    if args.sglang_child_capacity is not None or args.sglang_child_spec is not None:
        _run_sglang_child(args)
        return
    if args.check_only:
        check_only(args)
        return
    run_study(args)


if __name__ == "__main__":
    main()
