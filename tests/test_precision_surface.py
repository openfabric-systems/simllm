"""The unified precision surface: validation, refusal, stamp and compatibility.

Byte compatibility is the load-bearing property here, so it is pinned by
comparing the artifacts a legacy configuration renders against the artifacts an
explicitly configured one renders, not by asserting that they agree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simllm.backends import HtsimStepSink, HtsimStepSinkConfig
from simllm.backends.htsim_rnic import HtsimRnicConfig, build_htsim_rnic_command
from simllm.compute import (
    ComputeProvider,
    DurationEstimate,
    ModelDims,
    ProfileTableProvider,
    RooflineProvider,
)
from simllm.core import (
    PRECISION_CONFIG_SCHEMA,
    PRECISION_SEAMS,
    RUN_PROVENANCE_SCHEMA,
    STEP_SCHEMA,
    CoarseDeviceRuntime,
    ComputeLevel,
    DependencyLevel,
    ExecutionObservations,
    FrameworkLevel,
    LocalityLevel,
    NetworkLevel,
    PrecisionConfig,
    RequestOutcomeLevel,
    RequestPhase,
    RnicAuthorityMode,
    RnicHardwareLevel,
    RunProvenance,
    ScheduledRequest,
    StepRecord,
    WorkloadLevel,
    check_precision_selection,
    compute_level_for_provider,
    dependency_level_for_observations,
    locality_level_for_placement,
    network_level_for_profile,
    precision_config_from_json,
    precision_config_sha256,
    precision_config_to_bytes,
    precision_config_to_json,
    read_run_provenance,
    resolve_precision_config,
    rnic_hardware_level_for_authority_mode,
    run_provenance_from_bytes,
    run_provenance_from_json,
    run_provenance_to_bytes,
    run_provenance_to_json,
    step_record_to_json,
    write_run_provenance,
)
from simllm.placement import PlacementManifest, RankPlacement

# The frozen expectations of examples/precision_surface_v1. These literals are
# the study's registered oracle and are duplicated here so a regression fails
# in the fast test suite as well as in the study.
REFUSAL_DIAGNOSTIC = (
    "precision.rnic_hardware='composed-native' is incompatible with "
    "precision.network='rnic-nn-fluid'; select "
    "rnic_hardware='timing-neutral-bypass' or network='packet-level'"
)
LEGAL_PRECISION_JSON = (
    '{"compute":"roofline","dependency":"serial","framework":"recorded-steps",'
    '"locality":"all-remote","network":"packet-level",'
    '"request_outcome":"fabricated","rnic_hardware":"composed-native",'
    '"schema":"simllm-precision-config-v1","workload":"fixed-trace"}'
)
LEGAL_PRECISION_SHA256 = (
    "8e65df0c5296334800755254cb73c4c4f9cb2c090a2b8805a6409bdc3fbe7d45"
)
SOURCE_BYTES_LENGTH = 143
SOURCE_SHA256 = "499a5aee695b8269b1ffb5263f62fee6a00207416f7d62d1b0af64f543a68dca"
LEGAL_PROVENANCE_BYTES = 515
LEGAL_PROVENANCE_SHA256 = (
    "9eea24bf89de06325ee492cba345a22c0245c3a806bdd14da0fdbbd77871978d"
)

_DIMS = ModelDims(
    num_layers=2,
    hidden_size=64,
    intermediate_size=128,
    num_heads=4,
    num_kv_heads=4,
    head_size=16,
    vocab_size=256,
    dtype_bytes=2,
)


class UndeclaredProvider(ComputeProvider):
    """A caller-defined provider that names no precision level."""

    def estimate(self, kernel, gpu):
        return DurationEstimate(duration_ps=2_000_000, bound="compute")


def _legal_precision() -> PrecisionConfig:
    return PrecisionConfig(
        workload=WorkloadLevel.FIXED_TRACE,
        request_outcome=RequestOutcomeLevel.FABRICATED,
        framework=FrameworkLevel.RECORDED_STEPS,
        compute=ComputeLevel.ROOFLINE,
        dependency=DependencyLevel.SERIAL,
        locality=LocalityLevel.ALL_REMOTE,
        network=NetworkLevel.PACKET_LEVEL,
        rnic_hardware=RnicHardwareLevel.COMPOSED_NATIVE,
    )


def _canonical_source_bytes() -> bytes:
    record = StepRecord(step_index=0, virtual_time_ps=0, scheduled=[])
    payload = json.dumps(
        step_record_to_json(record),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload.encode("utf-8") + b"\n"


def _record(step_index: int = 4) -> StepRecord:
    return StepRecord(
        step_index=step_index,
        virtual_time_ps=step_index * 1_000_000,
        scheduled=[
            ScheduledRequest(
                f"request-{step_index}",
                RequestPhase.DECODE,
                num_new_tokens=1,
                context_length=128,
            )
        ],
    )


def _manifest() -> PlacementManifest:
    return PlacementManifest(
        ranks=[
            RankPlacement(0, "node-a", 0),
            RankPlacement(1, "node-b", 0),
        ]
    )


def _goal_artifacts(workdir: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (path.name, path.read_bytes()) for path in sorted(workdir.glob("step-*.goal"))
    )


# --- vocabulary and strict construction -----------------------------------


def test_the_surface_names_the_eight_fidelity_matrix_seams():
    assert PRECISION_SEAMS == (
        "workload",
        "request_outcome",
        "framework",
        "compute",
        "dependency",
        "locality",
        "network",
        "rnic_hardware",
    )
    payload = precision_config_to_json(PrecisionConfig.compatibility())
    assert set(payload) == {"schema", *PRECISION_SEAMS}
    assert payload["schema"] == PRECISION_CONFIG_SCHEMA


def test_the_compatibility_configuration_is_every_seams_baseline_level():
    assert precision_config_to_json(PrecisionConfig.compatibility()) == {
        "schema": PRECISION_CONFIG_SCHEMA,
        "workload": "fixed-trace",
        "request_outcome": "fabricated",
        "framework": "recorded-steps",
        "compute": "fixed",
        "dependency": "serial",
        "locality": "all-remote",
        "network": "rnic-nn-fluid",
        "rnic_hardware": "timing-neutral-bypass",
    }


def test_a_precision_configuration_rejects_a_level_of_the_wrong_seam():
    with pytest.raises(TypeError, match="precision.network"):
        PrecisionConfig(
            workload=WorkloadLevel.FIXED_TRACE,
            request_outcome=RequestOutcomeLevel.FABRICATED,
            framework=FrameworkLevel.RECORDED_STEPS,
            compute=ComputeLevel.ROOFLINE,
            dependency=DependencyLevel.SERIAL,
            locality=LocalityLevel.ALL_REMOTE,
            network=ComputeLevel.ROOFLINE,  # type: ignore[arg-type]
            rnic_hardware=RnicHardwareLevel.TIMING_NEUTRAL_BYPASS,
        )


@pytest.mark.parametrize("seam", PRECISION_SEAMS)
def test_a_parsed_configuration_requires_every_seam(seam):
    payload = precision_config_to_json(_legal_precision())
    del payload[seam]
    with pytest.raises(ValueError, match="precision"):
        precision_config_from_json(payload)


def test_a_parsed_configuration_rejects_unknown_fields_values_and_schemas():
    payload = precision_config_to_json(_legal_precision())
    with pytest.raises(ValueError, match="precision"):
        precision_config_from_json({**payload, "transport": "roce-v2"})
    with pytest.raises(ValueError, match="precision.network"):
        precision_config_from_json({**payload, "network": "rnic-cn"})
    with pytest.raises(ValueError, match="unsupported schema"):
        precision_config_from_json({**payload, "schema": "simllm-precision-config-v2"})


# --- the refused combination (the study's scored family) -------------------


def test_fluid_network_with_composed_native_hardware_is_refused_exactly():
    with pytest.raises(ValueError) as error:
        PrecisionConfig(
            workload=WorkloadLevel.FIXED_TRACE,
            request_outcome=RequestOutcomeLevel.FABRICATED,
            framework=FrameworkLevel.RECORDED_STEPS,
            compute=ComputeLevel.ROOFLINE,
            dependency=DependencyLevel.SERIAL,
            locality=LocalityLevel.ALL_REMOTE,
            network=NetworkLevel.RNIC_NN_FLUID,
            rnic_hardware=RnicHardwareLevel.COMPOSED_NATIVE,
        )
    assert str(error.value) == REFUSAL_DIAGNOSTIC


def test_the_refusal_survives_the_parser_and_never_degrades_silently():
    payload = precision_config_to_json(_legal_precision())
    payload["network"] = "rnic-nn-fluid"
    with pytest.raises(ValueError) as error:
        precision_config_from_json(payload)
    assert str(error.value) == REFUSAL_DIAGNOSTIC


@pytest.mark.parametrize(
    ("network", "rnic_hardware"),
    [
        (NetworkLevel.RNIC_NN_FLUID, RnicHardwareLevel.TIMING_NEUTRAL_BYPASS),
        (NetworkLevel.PACKET_LEVEL, RnicHardwareLevel.TIMING_NEUTRAL_BYPASS),
        (NetworkLevel.PACKET_LEVEL, RnicHardwareLevel.COMPOSED_NATIVE),
    ],
)
def test_the_three_legal_network_hardware_cells_are_accepted(network, rnic_hardware):
    config = PrecisionConfig(
        workload=WorkloadLevel.FIXED_TRACE,
        request_outcome=RequestOutcomeLevel.FABRICATED,
        framework=FrameworkLevel.RECORDED_STEPS,
        compute=ComputeLevel.ROOFLINE,
        dependency=DependencyLevel.SERIAL,
        locality=LocalityLevel.ALL_REMOTE,
        network=network,
        rnic_hardware=rnic_hardware,
    )
    assert config.network is network
    assert config.rnic_hardware is rnic_hardware


# --- the provenance stamp --------------------------------------------------


def test_the_legal_stamp_matches_its_frozen_bytes_and_digests():
    precision = _legal_precision()
    assert precision_config_to_bytes(precision).decode("utf-8") == LEGAL_PRECISION_JSON
    assert precision_config_sha256(precision) == LEGAL_PRECISION_SHA256

    source = _canonical_source_bytes()
    assert len(source) == SOURCE_BYTES_LENGTH
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256

    provenance = RunProvenance.from_source_bytes(
        source_schema=STEP_SCHEMA,
        source_bytes=source,
        precision=precision,
    )
    payload = run_provenance_to_bytes(provenance)
    assert len(payload) == LEGAL_PROVENANCE_BYTES
    assert hashlib.sha256(payload).hexdigest() == LEGAL_PROVENANCE_SHA256
    assert run_provenance_to_json(provenance)["precision_sha256"] == (
        LEGAL_PRECISION_SHA256
    )


def test_the_stamp_round_trips_through_parse_and_serialization_exactly():
    provenance = _legal_precision().stamp(
        source_schema=STEP_SCHEMA,
        source_sha256=SOURCE_SHA256,
    )
    payload = run_provenance_to_bytes(provenance)
    assert run_provenance_to_bytes(run_provenance_from_bytes(payload)) == payload
    parsed = run_provenance_from_json(json.loads(payload.decode("utf-8")))
    assert parsed == provenance
    assert parsed.precision == _legal_precision()


def test_the_stamp_round_trips_through_a_written_file(tmp_path):
    provenance = _legal_precision().stamp(
        source_schema=STEP_SCHEMA,
        source_sha256=SOURCE_SHA256,
    )
    target = write_run_provenance(provenance, tmp_path / "nested" / "provenance.json")
    assert target.read_bytes() == run_provenance_to_bytes(provenance)
    assert read_run_provenance(target) == provenance
    with pytest.raises(FileExistsError):
        write_run_provenance(provenance, target)


def test_a_stamp_reader_rejects_every_corrupted_field():
    provenance = _legal_precision().stamp(
        source_schema=STEP_SCHEMA,
        source_sha256=SOURCE_SHA256,
    )
    payload = run_provenance_to_json(provenance)
    with pytest.raises(ValueError, match="run_provenance"):
        run_provenance_from_json({**payload, "extra": 1})
    with pytest.raises(ValueError, match="run_provenance"):
        run_provenance_from_json(
            {name: value for name, value in payload.items() if name != "source_schema"}
        )
    with pytest.raises(ValueError, match="unsupported schema"):
        run_provenance_from_json({**payload, "schema": "simllm-run-provenance-v2"})
    with pytest.raises(ValueError, match="hexadecimal"):
        run_provenance_from_json({**payload, "source_sha256": "not-a-digest"})
    with pytest.raises(ValueError, match="digest mismatch"):
        run_provenance_from_json({**payload, "precision_sha256": "0" * 64})
    corrupted = {**payload, "precision": {**payload["precision"], "network": "rnic-cn"}}
    with pytest.raises(ValueError, match="precision.network"):
        run_provenance_from_json(corrupted)
    with pytest.raises(ValueError, match="invalid UTF-8 JSON"):
        run_provenance_from_bytes(b"{not json")


def test_a_stamp_requires_the_schema_and_hash_it_claims():
    with pytest.raises(ValueError, match="hexadecimal"):
        _legal_precision().stamp(source_schema=STEP_SCHEMA, source_sha256="abc")
    with pytest.raises(ValueError, match="run_provenance.source_schema"):
        _legal_precision().stamp(source_schema="   ", source_sha256=SOURCE_SHA256)
    with pytest.raises(TypeError, match="source_bytes"):
        RunProvenance.from_source_bytes(
            source_schema=STEP_SCHEMA,
            source_bytes="not bytes",  # type: ignore[arg-type]
            precision=_legal_precision(),
        )


# --- resolving the existing per-seam spellings -----------------------------


@pytest.mark.parametrize(
    ("profile", "level"),
    [
        ("rnic-nn-fluid", NetworkLevel.RNIC_NN_FLUID),
        ("rnic-nn", NetworkLevel.PACKET_LEVEL),
        ("rnic-cn", NetworkLevel.PACKET_LEVEL),
        ("dcqcn", NetworkLevel.PACKET_LEVEL),
    ],
)
def test_the_profile_spelling_selects_its_network_level(profile, level):
    assert network_level_for_profile(profile) is level


def test_an_unknown_profile_names_no_network_level():
    with pytest.raises(ValueError, match="selects no known network level"):
        network_level_for_profile("rnic-quantum")


def test_the_authority_mode_spelling_selects_its_rnic_hardware_level():
    assert (
        rnic_hardware_level_for_authority_mode(RnicAuthorityMode.BYPASS)
        is RnicHardwareLevel.TIMING_NEUTRAL_BYPASS
    )
    assert (
        rnic_hardware_level_for_authority_mode(RnicAuthorityMode.STRUCTURAL)
        is RnicHardwareLevel.COMPOSED_NATIVE
    )
    with pytest.raises(ValueError, match="selects no known RNIC hardware level"):
        rnic_hardware_level_for_authority_mode("emulated")


def test_placement_and_observation_presence_select_their_levels():
    assert locality_level_for_placement(None) is LocalityLevel.ALL_REMOTE
    assert locality_level_for_placement(_manifest()) is LocalityLevel.ANALYTIC_NVLINK
    assert dependency_level_for_observations(None) is DependencyLevel.SERIAL
    observations = ExecutionObservations(operations=(), completion_operation_ids=())
    assert (
        dependency_level_for_observations(observations)
        is DependencyLevel.OBSERVED_FRAMEWORK_SCHEDULE
    )


def test_a_provider_resolves_only_the_level_it_declares():
    assert compute_level_for_provider(RooflineProvider()) is ComputeLevel.ROOFLINE
    table = ProfileTableProvider({})
    assert compute_level_for_provider(table) is ComputeLevel.PROFILE_TABLE
    assert compute_level_for_provider(UndeclaredProvider()) is None


# --- partial views never claim or refuse a seam they cannot see ------------


def test_a_component_reports_only_the_seams_it_observes():
    observed = check_precision_selection(
        network=NetworkLevel.PACKET_LEVEL,
        selection_source="probe",
    )
    assert observed == {"network": NetworkLevel.PACKET_LEVEL}


def test_a_single_seam_view_is_never_refused_on_an_unobserved_seam():
    """A structural runtime selects no network level, so none may be assumed."""

    observed = check_precision_selection(
        rnic_hardware=RnicHardwareLevel.COMPOSED_NATIVE,
        selection_source="probe",
    )
    assert observed == {"rnic_hardware": RnicHardwareLevel.COMPOSED_NATIVE}


def test_the_run_level_composer_completes_and_validates_as_a_whole():
    composed = resolve_precision_config(
        network=NetworkLevel.PACKET_LEVEL,
        rnic_hardware=RnicHardwareLevel.COMPOSED_NATIVE,
    )
    assert composed.compute is ComputeLevel.FIXED
    assert composed.workload is WorkloadLevel.FIXED_TRACE
    with pytest.raises(ValueError) as error:
        resolve_precision_config(rnic_hardware=RnicHardwareLevel.COMPOSED_NATIVE)
    assert str(error.value) == REFUSAL_DIAGNOSTIC
    explicit = _legal_precision()
    assert resolve_precision_config(explicit) is explicit


def test_an_explicit_disagreement_names_the_seam_the_level_and_the_source():
    with pytest.raises(ValueError) as error:
        check_precision_selection(
            _legal_precision(),
            network=NetworkLevel.RNIC_NN_FLUID,
            selection_source="HtsimStepSinkConfig",
        )
    assert str(error.value) == (
        "precision.network='packet-level' conflicts with HtsimStepSinkConfig, "
        "which selects 'rnic-nn-fluid'"
    )


def test_a_selection_rejects_a_level_from_the_wrong_seam():
    with pytest.raises(TypeError, match="probe.network"):
        check_precision_selection(
            network=RnicHardwareLevel.COMPOSED_NATIVE,  # type: ignore[arg-type]
            selection_source="probe",
        )


# --- routing the existing spellings through the components -----------------


def test_the_runtime_reports_its_authority_mode_as_the_rnic_hardware_level():
    bypass = CoarseDeviceRuntime()
    assert bypass.precision is None
    assert bypass.selected_precision_levels == {
        "rnic_hardware": RnicHardwareLevel.TIMING_NEUTRAL_BYPASS
    }


def test_the_runtime_refuses_a_precision_that_contradicts_its_authority_mode():
    precision = _legal_precision()
    assert precision.rnic_hardware is RnicHardwareLevel.COMPOSED_NATIVE
    with pytest.raises(ValueError) as error:
        CoarseDeviceRuntime(authority_mode=RnicAuthorityMode.BYPASS, precision=precision)
    assert str(error.value) == (
        "precision.rnic_hardware='composed-native' conflicts with "
        "CoarseDeviceRuntime, which selects 'timing-neutral-bypass'"
    )


def test_the_htsim_config_reports_and_checks_its_profile_spelling(tmp_path):
    config = HtsimRnicConfig(
        goal_bin=tmp_path / "step.bin",
        profile="rnic-nn-fluid",
        linkspeed_bps=400_000_000_000,
    )
    assert config.selected_precision_levels == {"network": NetworkLevel.RNIC_NN_FLUID}
    with pytest.raises(ValueError, match="conflicts with HtsimRnicConfig"):
        HtsimRnicConfig(
            goal_bin=tmp_path / "step.bin",
            profile="rnic-nn-fluid",
            linkspeed_bps=400_000_000_000,
            precision=_legal_precision(),
        )


def test_the_step_sink_config_reports_every_seam_it_selects(tmp_path):
    config = HtsimStepSinkConfig(
        profile="rnic-nn",
        tp_ranks=(0, 1),
        dims=_DIMS,
        workdir=tmp_path / "legacy",
    )
    assert config.selected_precision_levels == {
        "compute": ComputeLevel.ROOFLINE,
        "dependency": DependencyLevel.SERIAL,
        "locality": LocalityLevel.ALL_REMOTE,
        "network": NetworkLevel.PACKET_LEVEL,
    }
    placed = HtsimStepSinkConfig(
        profile="rnic-nn-fluid",
        tp_ranks=(0, 1),
        dims=_DIMS,
        workdir=tmp_path / "placed",
        placement_manifest=_manifest(),
        provider=UndeclaredProvider(),
    )
    assert placed.selected_precision_levels == {
        "dependency": DependencyLevel.SERIAL,
        "locality": LocalityLevel.ANALYTIC_NVLINK,
        "network": NetworkLevel.RNIC_NN_FLUID,
    }


@pytest.mark.parametrize(
    ("overrides", "seam", "configured", "selected"),
    [
        ({"profile": "rnic-nn-fluid"}, "network", "packet-level", "rnic-nn-fluid"),
        ({"provider": ProfileTableProvider({})}, "compute", "roofline", "profile-table"),
    ],
)
def test_the_step_sink_refuses_a_contradicting_surface_before_any_output(
    tmp_path, overrides, seam, configured, selected
):
    workdir = tmp_path / "refused"
    values = {
        "profile": "rnic-nn",
        "tp_ranks": (0, 1),
        "dims": _DIMS,
        "workdir": workdir,
        "precision": _legal_precision(),
    }
    values.update(overrides)
    with pytest.raises(ValueError) as error:
        HtsimStepSink(HtsimStepSinkConfig(**values))
    assert str(error.value) == (
        f"precision.{seam}={configured!r} conflicts with HtsimStepSinkConfig, "
        f"which selects {selected!r}"
    )
    assert not workdir.exists()


def test_a_locality_disagreement_is_refused_before_any_output(tmp_path):
    workdir = tmp_path / "refused-locality"
    with pytest.raises(ValueError, match="precision.locality='all-remote'"):
        HtsimStepSink(
            HtsimStepSinkConfig(
                profile="rnic-nn",
                tp_ranks=(0, 1),
                dims=_DIMS,
                workdir=workdir,
                placement_manifest=_manifest(),
                precision=_legal_precision(),
            )
        )
    assert not workdir.exists()


# --- byte compatibility of every migrated spelling -------------------------


@pytest.mark.parametrize("profile", ["rnic-nn", "rnic-nn-fluid"])
def test_an_agreeing_surface_renders_byte_identical_step_artifacts(tmp_path, profile):
    """The explicit surface is metadata: it may not move a single byte."""

    record = _record()
    precision = resolve_precision_config(
        compute=ComputeLevel.ROOFLINE,
        dependency=DependencyLevel.SERIAL,
        locality=LocalityLevel.ALL_REMOTE,
        network=network_level_for_profile(profile),
    )

    def _plan(workdir: Path, explicit: PrecisionConfig | None):
        sink = HtsimStepSink(
            HtsimStepSinkConfig(
                profile=profile,
                tp_ranks=(0, 1),
                dims=_DIMS,
                workdir=workdir,
                precision=explicit,
            )
        )
        return sink._plan_step(record)

    legacy_dir = tmp_path / "legacy"
    explicit_dir = tmp_path / "explicit"
    legacy_plan = _plan(legacy_dir, None)
    explicit_plan = _plan(explicit_dir, precision)
    assert legacy_plan is not None and explicit_plan is not None

    legacy_goals = _goal_artifacts(legacy_dir)
    explicit_goals = _goal_artifacts(explicit_dir)
    assert legacy_goals
    assert [name for name, _ in legacy_goals] == [name for name, _ in explicit_goals]
    assert [body for _, body in legacy_goals] == [body for _, body in explicit_goals]

    assert legacy_plan.compute_service_ps == explicit_plan.compute_service_ps
    assert legacy_plan.compute_estimate_ps == explicit_plan.compute_estimate_ps
    assert legacy_plan.profile == explicit_plan.profile
    assert legacy_plan.linkspeed_bps == explicit_plan.linkspeed_bps
    assert legacy_plan.graph_artifact_count == explicit_plan.graph_artifact_count
    assert legacy_plan.effective_dependency_edge_count == (
        explicit_plan.effective_dependency_edge_count
    )
    assert legacy_plan.locality == explicit_plan.locality


@pytest.mark.parametrize("profile", ["rnic-nn", "rnic-nn-fluid", "rnic-cn"])
def test_an_agreeing_surface_leaves_the_backend_command_unchanged(tmp_path, profile):
    precision = resolve_precision_config(network=network_level_for_profile(profile))

    def _argv(explicit: PrecisionConfig | None) -> list[str]:
        return build_htsim_rnic_command(
            Path("htsim_rnic"),
            HtsimRnicConfig(
                goal_bin=tmp_path / "step.bin",
                profile=profile,
                linkspeed_bps=400_000_000_000,
                completion_csv=tmp_path / "completions.csv",
                precision=explicit,
            ),
        )

    assert _argv(None) == _argv(precision)
    assert "-precision" not in _argv(precision)


def test_an_agreeing_surface_leaves_the_runtime_authority_unchanged():
    legacy = CoarseDeviceRuntime()
    explicit = CoarseDeviceRuntime(
        precision=resolve_precision_config(
            rnic_hardware=RnicHardwareLevel.TIMING_NEUTRAL_BYPASS
        )
    )
    assert legacy.authority_name == explicit.authority_name
    assert legacy.bypass_ledger is not None
    assert explicit.bypass_ledger is not None


def test_the_vllm_executor_config_reports_the_seams_its_env_spellings_select():
    from simllm.adapters.vllm.executor import SimExecutorConfig

    assert SimExecutorConfig().selected_precision_levels() == {
        "request_outcome": RequestOutcomeLevel.FABRICATED,
        "dependency": DependencyLevel.SERIAL,
    }
    replay = SimExecutorConfig(
        replay_run_path="run.json",
        observed_schedule="granite-dbo",
    )
    assert replay.selected_precision_levels() == {
        "request_outcome": RequestOutcomeLevel.PREPLAY_ORACLE,
        "dependency": DependencyLevel.OBSERVED_FRAMEWORK_SCHEDULE,
    }
    with pytest.raises(ValueError) as error:
        replay.selected_precision_levels(_legal_precision())
    assert str(error.value) == (
        "precision.request_outcome='fabricated' conflicts with "
        "SimExecutorConfig, which selects 'preplay-oracle'"
    )
    assert SimExecutorConfig().selected_precision_levels(_legal_precision()) == {
        "request_outcome": RequestOutcomeLevel.FABRICATED,
        "dependency": DependencyLevel.SERIAL,
    }


def test_the_precision_field_stays_out_of_the_step_record_wire_format():
    """Old readers keep parsing new runs: the stamp lives beside them."""

    payload = step_record_to_json(_record())
    assert "precision" not in payload
    assert payload["schema"] == STEP_SCHEMA
    assert run_provenance_to_json(
        _legal_precision().stamp(
            source_schema=STEP_SCHEMA,
            source_sha256=SOURCE_SHA256,
        )
    )["schema"] == RUN_PROVENANCE_SCHEMA
