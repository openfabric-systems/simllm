# Precision surface v1 results

## Outcome

The run is **valid**. The one registered genuine-risk family passed its one
parameterized instance: **1 of 1**. No fatal guard was violated, so the scored
fraction is interpretable.

The refused cell is the only scored evidence. A configuration naming
`network='rnic-nn-fluid'` together with `rnic_hardware='composed-native'`
raised `ValueError` with exactly the frozen diagnostic:

```text
precision.rnic_hardware='composed-native' is incompatible with precision.network='rnic-nn-fluid'; select rnic_hardware='timing-neutral-bypass' or network='packet-level'
```

The three legal cells of the same two-seam sweep were accepted. Those
acceptances, the exact stamp, the canonical hashes, the strict-reader
rejections, the legacy-resolution rows, the bypass identity comparisons and
the output-path guard are fatal and unscored: they are the preconditions
under which the scored refusal means what it claims, not additional points.

## Chronology and provenance

| Event | Commit |
|---|---|
| Evidence authored against | `76223875557a552deb5aa2c2c529a07f000135ba` |
| Expectations-only freeze | `6db6fd95705ab99124eeaa0617cfa6a195010e9d` |
| Implementation | `772c822cd6b79211db2b5ff1c9db2c045f3a1ca2` |

The registered command passed `--check-only` before the freeze commit, reading
no source artifact, importing no SimLLM implementation and creating no output
directory. Every SimLLM import in the result path is local to a function, so
check-only remains free of the implementation it validates.

| Provenance field | Observed value |
|---|---|
| htsim gitlink observed by this run | `fc4400e4ca619223481536632074045cb6af2756` |
| Observation SHA-256 | `0c386761703e6c507d04c4323d796e78688f646dad9e38527b7c7a01def9094e` |
| Result SHA-256 | `71e26b8dfb3d14f1b0de005af7c05a2a19e24381d180d8f95a9fdc9cb5801857` |
| Run provenance SHA-256 | `54f05ec2b0062e8cda5c58e70b88324ccf6fc323ace72f2f5377d5a3ce76d0b2` |

The htsim gitlink is recorded as observed. No frozen literal asserts it, so the
pin can move without editing this study. The submodule is not checked out in
this worktree and no native binary exists, which is why the backend is stubbed
(see the harness limitation below).

The run stamps itself: `run-provenance.json` names the observation file's
schema and hash next to the resolved eight-seam configuration, so the result
carries the precision that produced it.

## The seam inventory and where each level is selected today

The audit was frozen before the design. The surface resolves a legacy spelling
into a level wherever a component can observe one:

| Seam | Spelling observed today | Resolves to |
|---|---|---|
| Workload | none; callers construct arrivals and lengths separately | not routed |
| Request outcome | `SimExecutorConfig.replay_run_path` absent / present | `fabricated` / `preplay-oracle` |
| Framework seam | which entry point a deployment starts | not routed |
| Compute | `provider` declaring `precision_compute_level` | `roofline`, `profile-table`, or unresolved |
| Dependency | `ExecutionObservations` absent / present, `SIMLLM_VLLM_OBSERVED_SCHEDULE` | `serial` / `observed-framework-schedule` |
| Locality | `placement_manifest` absent / present | `all-remote` / `analytic-nvlink` |
| Network | `profile` string | `rnic-nn-fluid` / `packet-level` |
| RNIC hardware | `RnicAuthorityMode` | `timing-neutral-bypass` / `composed-native` |

A provider that declares no level resolves to nothing rather than being
assumed to be `fixed`. That is deliberate: a caller-defined provider is the
common case in this repository's studies, and guessing its level would put a
false claim into a published stamp.

## Partial views never claim or refuse a seam they cannot see

The first wiring attempt completed every component's partial view with the
compatibility level and then validated the whole. That broke eighteen existing
tests, and the break was correct behavior of a wrong design rather than a bug
in the frozen relation. A `CoarseDeviceRuntime` in structural mode selects
`composed-native` and selects no network level at all; filling the network
seam with its compatibility default manufactured `rnic-nn-fluid` and then
refused the run on a pair the runtime never chose.

The surface therefore has two entry points with different contracts:

- `check_precision_selection` reports only the seams a component observes and
  refuses an explicit disagreement on those. It never invents a level, so a
  partial view cannot be refused on a seam it does not own.
- `resolve_precision_config` composes a complete run configuration, because
  its caller is asserting a whole run. Completing `composed-native` with the
  default fluid network is a genuine incoherence at run level and is refused
  with the same diagnostic.

## Entailment analysis

The scored refusal is evaluated from the constructor's raw observation before
any expected-matrix, schema, hash or stamp check runs. It is not entailed by
an earlier fatal oracle, and the claim was tested rather than argued: a mutant
that disables the incompatibility rule (`if False:` in place of the pair test)
leaves every fatal guard passing and the run valid, while the scored family
drops to **0 of 1**. A permissive validator therefore reaches the observation
point exactly as the freeze predicted.

## Compatibility evidence and one harness correction

Every routed spelling was run twice, once with the legacy spelling alone and
once with an agreeing explicit surface, and compared through the existing
`BypassArtifacts` / `assert_bypass_artifact_identity` contract. No parallel
comparator was added. All six cells matched on every input and behavioral byte
class.

| Cell | GOAL bytes | Backend argv bytes | `StepResult` bytes |
|---|---|---|---|
| `profile=rnic-nn` | 1464 | 781 | 149 |
| `profile=rnic-nn-fluid` | 1464 | 829 | 149 |
| `provider=RooflineProvider` | 1464 | 781 | 149 |
| `provider=ProfileTableProvider` | 1464 | 781 | 151 |
| `placement_manifest=absent` | 9600 | 781 | 149 |
| `placement_manifest=present` | 4320 | 4681 | 149 |

The cells differ from one another, which is what makes the comparison
informative: the profile-table provider produces a two-byte-different result
payload, and the analytic NVLink split renders 4320 GOAL bytes against the
all-remote 9600 for the same four ranks.

The first version of this harness was weaker than it looked and the weakness
was found by mutation, not by inspection. A probe that doubled the link rate
on the explicit path only should have voided the run; it did not, because the
stub backend ignored `linkspeed_bps` and the artifacts recorded a constant
instead of the arguments actually issued. The stub now derives each flow's
completion from the payload serialization at the configured rate and records
every backend argument vector into the input byte class. The same probe now
voids the run with all six identity findings, and the reported comparison is
sensitive to any backend-visible configuration difference.

## Harness limitation, stated plainly

`third_party/htsim` is not checked out in this worktree and no `htsim_rnic`
binary exists, so the backend is an in-process stub. Every byte under
comparison is rendered by SimLLM itself: GOAL text, the source step record,
the backend argument vector, the locality projection and the `StepResult`. The
stub supplies only flow completions, and this study reads no duration as a
modeled result. The comparison is therefore valid for the compatibility claim
and carries no fidelity claim.

## Physical sanity

The change selects and records existing levels. It changes no duration, byte
count, rate, queue or modeled result, and the study reads no TTFT, TPOT, JCT
or FCT value. First-principles timing floors and ceilings are not applicable,
exactly as the freeze stated. The byte-identity comparison is the operative
check that no modeled timing moved: had one moved, the `StepResult` and
locality byte classes would have differed and the run would be void.

## Closure scope

| Registered clause | Evidence | Verdict |
|---|---|---|
| 1. one configuration naming the level of every seam | strict eight-field schema, frozen selector audit, complete legal stamp | demonstrated |
| 2. validate it up front | strict construction, and a contradicting surface refused before the workdir, any GOAL artifact, any backend process or either WQE authority exists | demonstrated |
| 3. refuse incompatible combinations explicitly | raw fluid plus composed-native refusal carrying the frozen diagnostic, scored 1 of 1 | demonstrated |
| 4. stamp the resolved selection into run provenance | exact 515-byte payload, three frozen digests, byte round trip through parse, file and reader | demonstrated |
| 5. current per-seam spellings stay supported and byte-identical | audited legacy-resolution rows plus six accepted bypass comparisons | demonstrated for every spelling the surface routes |

CORE-36 closes on those five clauses. Two residuals move to their own IDs
rather than being narrated as done:

- **CORE-44** covers the workload and framework seams. Neither has a spelling
  the surface can observe: there is no combined workload selector anywhere,
  and the framework level is chosen by which entry point a deployment starts
  rather than by any record. A run therefore still cannot derive those two
  levels from its own components, and must name them explicitly.
- **CORE-45** covers live emission. A sink observes four seams and cannot
  compose a complete configuration by itself, and the source artifact identity
  is not known when it is constructed, so no live run writes its own stamp
  today. This study writes its own because it knows both.

## Contradiction sweep

Swept after closure:

| File | Hit | Action |
|---|---|---|
| `README.md` | none; it names no per-seam selector | no edit needed |
| `docs/README_PRO.md:345` | "CORE-36 owns a single selection surface" | rewritten; the fidelity section is in this task's scope |
| `docs/architecture.md:499` | "CORE-36 owns making that one validated surface rather than the current per-seam mixture" | reported, not edited; out of scope for this task and now stale |
| `docs/modules/traffic.md:51,191` | CORE-36 named as the future selector | dangling closed-ID pointer, repointed at `PrecisionConfig` |
| `docs/modules/backends.md:90` | CORE-36 named as the future selector | dangling closed-ID pointer, repointed at `PrecisionConfig` |
| `docs/modules/core.md:131,495` | CORE-36 named as the future owner | repointed at the landed surface |

Frozen expectation files and past `RESULTS.md` records in
`examples/dispatch_sequence_v1` and `examples/dependency_authority_v1` also
name CORE-36 as future work. They are historical record and are deliberately
left unmodified.
