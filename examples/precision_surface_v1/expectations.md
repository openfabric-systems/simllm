# Precision surface v1 expectations

Date: 2026-08-12

This is the expectations-only record for CORE-36. It freezes the configuration
vocabulary, current-selector audit, incompatibility matrix, provenance stamp,
compatibility evidence and closure boundary before the precision surface is
implemented or any result-producing run is made.

## Claim boundary and source audit

The fidelity matrix in `docs/README_PRO.md:278-303` is the authority for the
eight seams. The new surface names those seams exactly once as `workload`,
`request_outcome`, `framework`, `compute`, `dependency`, `locality`, `network`
and `rnic_hardware`. Diagnostic cross-checks, transport policies, calibration
identities and seeds remain details of their owning levels rather than new
seams.

The evidence was authored against SimLLM commit
`76223875557a552deb5aa2c2c529a07f000135ba`. No backend revision is an input to
the frozen relation, and no frozen literal is required to equal a future live
submodule pin.

Before designing the surface, the current selectors were audited as follows:

| Seam | Current selection point |
|---|---|
| Workload | Callers construct `PoissonArrivals` or `TraceArrivals` in `simllm/workload/arrivals.py:16-46` and `FixedLengths`, `LogNormalLengths` or `TraceLengths` in `simllm/workload/lengths.py:15-55`. There is no combined selector. |
| Request outcome | vLLM selects fabricated output or pre-play replay by the absence or presence of `SimExecutorConfig.replay_run_path` and `SIMLLM_VLLM_REPLAY_RUN` in `simllm/adapters/vllm/executor.py:247-301`, then constructs `ReplayTokenSource` or `None` at `simllm/adapters/vllm/executor.py:1093-1100`. Framework CPU capture uses `SglangCpuRunner` and `VllmCpuRunner` at `simllm/preplay/framework_runner.py:405-453` and `simllm/preplay/framework_runner.py:967-1015`. |
| Framework seam | Recorded mode reads `StepRecord` JSONL in `simllm/core/step.py:363-378`. vLLM selects executor RPC with the `SimExecutor` dotted backend at `simllm/adapters/vllm/executor.py:1045-1051`, or the model-runner skeleton through `SIMLLM_VLLM_WORKER_MODE=skeleton` at `simllm/adapters/vllm/worker.py:609-625`. SGLang selects its model-runner replacement with `SIMLLM_SGLANG_ENABLE=1` or direct `install()` at `simllm/adapters/sglang/plugin.py:60-105`. |
| Compute | Callers inject a `ComputeProvider`; `SerialStepLowererConfig.provider` and `HtsimStepSinkConfig.provider` default to `RooflineProvider` at `simllm/backends/step_lowerer.py:50-64` and `simllm/backends/step_sink.py:129-138`. Implemented providers are defined at `simllm/compute/provider.py:58-188` and `simllm/compute/provider.py:256-318`. |
| Dependency | Presence of `ExecutionObservations` selects observed lowering while `None` preserves the serial path at `simllm/backends/step_lowerer.py:380-408`. The vLLM environment spelling is `SIMLLM_VLLM_OBSERVED_SCHEDULE` at `simllm/adapters/vllm/executor.py:265-300`. |
| Locality | `HtsimStepSinkConfig.placement_manifest` is absent for all-remote compatibility or present for the analytic NVLink split at `simllm/backends/step_sink.py:147-171`; the classification is applied at `simllm/traffic/locality.py:259-335`. |
| Network | `HtsimStepSinkConfig.profile` and `HtsimRnicConfig.profile` select `rnic-nn`, `rnic-nn-fluid` or `rnic-cn` at `simllm/backends/step_sink.py:129-160` and `simllm/backends/htsim_rnic.py:27-80`. DCQCN uses the separate `HtsimDcqcnConfig` entry point at `simllm/backends/htsim_dcqcn.py:37-66`. |
| RNIC hardware | `CoarseDeviceRuntime.authority_mode` selects `bypass` or `structural` and validates native-session presence at `simllm/core/runtime.py:882-931`. Build-time native availability is separate from this per-run choice. |

The repository audit is the only source audit needed for this configuration
task. No external timing, hardware or framework source determines the frozen
decision.

## Frozen level vocabulary

The v1 surface supports the levels that are executable now. Registered future
levels remain rejected until their owning tasks land evidence and extend a
later schema.

| Seam | Accepted v1 levels |
|---|---|
| `workload` | `fixed-trace`, `poisson-arrivals` |
| `request_outcome` | `fabricated`, `preplay-oracle`, `framework-cpu-oracle` |
| `framework` | `recorded-steps`, `executor-rpc`, `model-runner` |
| `compute` | `fixed`, `roofline`, `profile-table` |
| `dependency` | `serial`, `observed-framework-schedule` |
| `locality` | `all-remote`, `analytic-nvlink` |
| `network` | `rnic-nn-fluid`, `packet-level` |
| `rnic_hardware` | `timing-neutral-bypass`, `composed-native` |

The canonical compatibility configuration is the first value in every row.
An explicit configuration is strict: every field is required, unknown fields
and values are rejected, and strings are never silently coerced to a nearby
level.

## Decision-relevant two-seam sweep

The matrix crosses network and RNIC hardware precision while holding the
other six seams at their compatibility levels:

| Network | RNIC hardware | Frozen outcome |
|---|---|---|
| `rnic-nn-fluid` | `timing-neutral-bypass` | accept |
| `packet-level` | `timing-neutral-bypass` | accept |
| `packet-level` | `composed-native` | accept |
| `rnic-nn-fluid` | `composed-native` | refuse |

The refusal follows the existing architecture. The fluid closed form is the
explicit nonstructural bypass anchor in `docs/modules/backends.md:157-163`,
while the structural link-on path supports packet-level `rnic-nn` and
`rnic-cn` in `docs/modules/backends.md:700-702`.

The refused cell must raise `ValueError` with exactly:

```text
precision.rnic_hardware='composed-native' is incompatible with precision.network='rnic-nn-fluid'; select rnic_hardware='timing-neutral-bypass' or network='packet-level'
```

The runner records the raw accepted value or exception type and message for
all four cells before checking the expected matrix, schema, hash or exact
stamp. Only the refused cell is scored. It is one genuine-risk family with
one parameterized instance because a permissive validator can accept it and
reach the observation point. The three legal acceptances are fatal, unscored
preconditions for interpreting the surface.

## Frozen legal provenance stamp

The legal round-trip cell uses fixed trace, fabricated outcome, recorded
steps, roofline compute, serial dependencies, all-remote locality,
packet-level network and composed-native RNIC hardware. Its canonical
precision JSON, before the enclosing newline, is:

```json
{"compute":"roofline","dependency":"serial","framework":"recorded-steps","locality":"all-remote","network":"packet-level","request_outcome":"fabricated","rnic_hardware":"composed-native","schema":"simllm-precision-config-v1","workload":"fixed-trace"}
```

Its SHA-256 is
`8e65df0c5296334800755254cb73c4c4f9cb2c090a2b8805a6409bdc3fbe7d45`.
The source is one canonical empty `atlahs-closed-loop-step-v1` record whose
143 bytes have SHA-256
`499a5aee695b8269b1ffb5263f62fee6a00207416f7d62d1b0af64f543a68dca`.
The complete newline-terminated `simllm-run-provenance-v1` stamp is 515 bytes
with SHA-256
`9eea24bf89de06325ee492cba345a22c0245c3a806bdd14da0fdbbd77871978d`.

The run provenance object must carry `schema`, `source_schema`,
`source_sha256`, `precision` and `precision_sha256`. Strict parse followed by
canonical serialization must reproduce all 515 bytes exactly. Missing or
unknown fields, a malformed source hash, an unsupported schema, an invalid
precision combination and a mismatched precision hash are rejected. These
are exact-oracle and structural guards, not scored evidence.

## Existing-selector routing and byte compatibility

Explicit precision is metadata and validation, never a second mechanism. A
legacy provider object, profile string, placement-manifest presence,
dependency observation and authority mode remains the actual operational
selector. The unified surface resolves the level selected by that spelling
and rejects an explicit disagreement before a backend process, output file or
adapter runtime is created.

The compatibility study covers every current spelling named in the audit. It
compares legacy and explicitly configured command arguments, step-record
bytes, step-result JSON and bypass artifacts. The bypass comparison must use
the existing `BypassArtifacts` and `assert_bypass_artifact_identity` contract
from `simllm/backends/rnic_records.py:337-387` and
`simllm/backends/rnic_records.py:2005-2037`; no parallel comparator is added.
That contract intentionally excludes run provenance while locking GOAL text
and binary, topology, profile, seed, baseline parameters, completion CSV,
canonical completion rows, `StepResult` bytes and replay summary.

All compatibility comparisons are fatal and unscored. Some are
by-construction guards because the new field is omitted from old wire
schemas. They demonstrate the required identity but do not increase the
behavioral denominator.

## Entailment and evidence classes

The incompatible refusal is evaluated from the constructor's raw observation
before the exact diagnostic and expected-matrix checks. It can fail even if
the later legal provenance round trip and compatibility comparisons pass, so
no earlier oracle entails it. The exact stamp, canonical hash, strict-reader
rejections, legal acceptance cells, legacy identity checks and output-path
guard are fatal-unscored. Focused tests, full repository tests and lint are
separate executable evidence classes.

If any fatal guard fails, the run is void for closure. The retained report
names the finding without a behavioral fraction, and CORE-36 remains open.
Only a valid run reports the refusal family as a genuine-risk fraction.

## Physical sanity

The change selects and records existing levels without changing a duration,
byte count, rate, queue or modeled result. The study reads no TTFT, TPOT, JCT
or FCT value. First-principles timing floors and ceilings are therefore not
applicable. If implementation causes any modeled timing to differ, that is a
compatibility failure rather than a timing relation to rationalize.

## CORE-36 closure scope

Closure requires evidence for every registered clause:

1. "one configuration naming the level of every seam" requires the strict
   eight-field schema, selector audit and legal complete stamp;
2. "validate it up front" requires strict construction and legacy-conflict
   rejection before observable adapter or backend side effects;
3. "refuse incompatible combinations explicitly rather than silently
   degrading" requires the raw fluid plus composed-native refusal with the
   frozen diagnostic;
4. "stamp the resolved selection into the run provenance next to the
   existing schema and hash fields" requires the exact legal provenance
   payload and byte round trip; and
5. "The current per-seam spellings remain supported and byte-identical"
   requires the audited legacy-resolution tests and the accepted bypass
   comparator.

Any clause not demonstrated moves to CORE-44 or CORE-45 with a category,
priority and difficulty. Neither ID is registered speculatively.

## Registered command and pre-freeze dry run

The result-producing command is:

```bash
.venv/bin/python examples/precision_surface_v1/run_study.py \
  --out "$SIMLLM_CORE36_RUN_ROOT"
```

Before this expectations commit, that exact CLI is run with `--check-only`.
Check-only parses the production argument and validates only frozen literal
shapes, canonical JSON and hash arithmetic, the two-seam matrix, exact
diagnostic and evidence counts. It imports no SimLLM implementation, reads no
input, invokes no native binary, creates no output directory and writes no
result.
