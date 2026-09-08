# Exact completion boundaries and retained step execution

The complete 56-execution successor study passes after a disclosed correction
to its cross-profile message checker. Retaining the native queue makes an
8 KiB successor take **42.7648 microseconds instead of 2.4992 microseconds**
at 200 Gbit/s. The checked physical path also carries its network completion
times into time to first token (TTFT) and time per output token (TPOT).
BACK-38 and HTSIM-28 close for persistent execution across one checked step's
ordered artifacts. BRIDGE-2's cross-step framed client, BACK-72's optional
compositions and TRAF-8's captured pipeline serving remain open. This study
does not calibrate hardware or rescore the older congestion-chain experiment.

## Mechanism and chronology

A dependent message must start when its predecessor completes, while the
network retains unfinished packets and queued work. The native session now
stops immediately after the callback that exposes that completion. A boundary
token permits dependent injections at the exact timestamp, preserving other
callbacks already pending at that time. Existing session verbs keep their
original response bytes.

The installed `HtsimStepSinkConfig.flow_session` option retains one native
child through every artifact of a checked step. Immutable GOAL snapshots
preserve the original graph's messages and dependencies. Native lifecycle
observations feed canonical completion events, the execution result, the step
result and the existing request reducer. Native queue state has one authority;
its Python records are projections. The sink drains only after the final
artifact. Each following step uses a new child, so this is not a claim of
retained transport state between steps.

The [behavioral expectations](expectations.md) were frozen in the final
expectations-only commit `4d75849189df4a4b9113ee36101b246559cbf47b`, before
implementation and the first execution. The chronology is preserved:

1. The first complete execution used SimLLM `7526d88` and native backend
   `6efec16`. It is **VOID**, with no behavioral score. Twelve fatal message
   comparisons incorrectly treated runtime-local accepted sequence numbers as
   cross-profile identities. Both profiles actually contain the same 192
   unique declared messages; their schedulers accept twelve of them in a
   different order. This finding does not make the void run scoreable.
2. Expectations-only commit `5b19d24c56d34353ac5a1a539da3bb0e0eb2117f`
   froze the [post-specified comparison checks](post_run_checks.md) before
   changing the checker or repeating the population. These are explicitly
   post-specified regression checks, not pre-registration of an unseen
   outcome. The behavioral freeze, inputs, bounds and simulator are unchanged.
3. Checker commit `dd0463d40a3a31b116b7f4c8ef2b2de9cf047d45` joins by
   declared message identity and preserves both native sequences. The new
   complete execution passes. All 1,926 framed requests and responses across
   48 sessions, original native flow rows, live step results and compatibility
   controls are identical between the two executions. The original summary
   remains unchanged and void.

The [machine-readable result](results.json) retains every repeat verdict,
oracle, relation, native flow metric, step result and compatibility control.
It also carries completion and quiescence boundaries, source and executable
identities, evidence hashes and the first run's fatal findings. Raw frames,
logs, graphs and generated backend artifacts remain in external evidence
storage. Each run's manifest covers 5,141 files.

## Native continuation and retained queues

Floor: a flow needs its payload bytes divided by link rate, plus the ideal
profile's two microseconds of propagation.

Ceiling: the unloaded fixed-packet path is exactly
`F=(B/4096+1)*q+2,000,000` ps, where `q=4160*8*10^12/R` ps.
The extra frame accounts for destination serialization. A chain of K
nonoverlapping dependent flows completes at exactly `K*F`.

The eight chain configurations vary payload between 4 and 8 KiB, chain length
between two and four, and link rate between 200 and 400 Gbit/s. All 24
per-flow oracle rows and eight chain-boundary rows have **0 ps residual**.
Doubling rate halves the serialization term after removing propagation;
doubling payload adds exactly `K*q`. Both four-instance relation families
hold without a fitted constant.

The separate retention experiment starts a 4 KiB trigger and an opposite
1 MiB background transfer together, then releases a successor at the trigger's
exact completion. The comparison resets the native session and releases only
that successor at the same absolute time. The reset arm is a diagnostic,
not a supported production physical mode.

Floor: the background alone needs 1 MiB divided by rate plus propagation;
the successor also retains its own payload and propagation floor.

Ceiling: serializing every finite frame at both endpoints plus four propagation
allowances bounds quiescence by `2*(1+256+B/4096)*q+8,000,000` ps.

| Rate | Successor | Retained FCT | Fresh FCT | Retained quiescence | Quiescence ceiling |
|---:|---:|---:|---:|---:|---:|
| 200 Gbit/s | 8 KiB | 42.7648 us | 2.4992 us | 45.0976 us | 94.1952 us |
| 200 Gbit/s | 16 KiB | 43.0976 us | 2.8320 us | 45.4304 us | 94.8608 us |
| 400 Gbit/s | 8 KiB | 21.3824 us | 2.2496 us | 23.5488 us | 51.0976 us |
| 400 Gbit/s | 16 KiB | 21.5488 us | 2.4160 us | 23.7152 us | 51.4304 us |

Flow completion time (FCT) is measured from logical submission. Every retained
successor is slower and its source send-queue high-water mark is two, compared
with one in the fresh arm. Each session constructs exactly one native authority
with no legacy posts or mutations. This distinguishes retained work from an
empty simulator that merely retains an absolute clock.

## Checked request metrics

The live grid uses two active tensor-parallel ranks on separate leaves of an
eight-endpoint topology, two dense layers, batch sizes 16 and 32, and both
rates. Each configuration runs one prefill and two decode steps. Each step
contains six ordered artifacts, eight original graph operations, sixteen
messages and eight network rounds. Compute is explicitly declared at
32 microseconds per step with ideal host launch cost. No measured GPU service
or real deployment throughput is inferred from this synthetic network study.

Floor: physical completion requires the declared 32 microseconds of compute
plus eight rounds of payload serialization and four one-microsecond link pipes,
`L_CN >= 32,000,000+8*(B*8*10^12/R+4,000,000)` ps.

Ceiling: each step must finish within one millisecond and native work must
quiesce within ten milliseconds after release. These are frozen engineering
guards, not predictions of controller behavior.

The independent ideal oracle is `L_NN=32,000,000+8*F` ps. Its twelve step
rows have zero residual. Removing compute and propagation gives exact
inverse-rate scaling, and doubling batch adds `8*q`; the six instances in
each family hold. Physical timing is compared with each cell's additive
floor, without assuming a congestion-controller monotonicity law.

| Rate | Batch | Ideal TTFT and TPOT | Physical TTFT | Physical TPOT | Physical step floor |
|---:|---:|---:|---:|---:|---:|
| 200 Gbit/s | 16 | 50.6624 us | 90.7424 us | 90.7520 us | 65.31072 us |
| 200 Gbit/s | 32 | 51.9936 us | 92.0448 us | 92.0320 us | 66.62144 us |
| 400 Gbit/s | 16 | 49.3312 us | 77.5072 us | 77.5200 us | 64.65536 us |
| 400 Gbit/s | 32 | 49.9968 us | 78.3264 us | 78.3200 us | 65.31072 us |

Both decode intervals agree within each cell. The reported TPOT is the
existing exact request-history reduction. All twelve physical step excesses
respect their own positive floor; the observed range is 28.1760 to
40.0896 microseconds above the matched ideal path. All 384 native flow rows
retain their raw FCT, and the 192 physical rows also retain the ratio to their
uniquely matched ideal message.

The physical 200 Gbit/s, batch-16 prefill completes logically at 90.7424
microseconds, while native and graph quiescence occur at 94.58624 microseconds.
The remaining 3.84384 microseconds is not added to TTFT. Native quiescence and
graph quiescence are recorded separately even when they coincide. No sum of
queue visits is substituted for a request's critical-path latency.

## Evidence classes and integration gates

Every fatal frame, ownership, identity, dependency, physical-floor, budget,
quiescence and metric-conservation guard is clear in the repeat. The following
counts describe separate evidence classes and are not added together:

- 56 configuration executions, including both sides of compatibility pairs.
- 44 exact-oracle rows: 24 native flows, eight native chains and twelve ideal
  live steps, all with zero residual.
- Six behavioral families with 36 parameterized instances, all satisfied:
  native bandwidth and payload, retained state, live bandwidth and payload,
  and physical-versus-ideal request excess.
- Sixteen unscored compatibility controls: four old native transcripts,
  eight stateless ideal scenario pairs and four physical refusal pairs.
  Accepted GOAL text, binaries, CSVs, serialized step results and artifact
  inventories remain byte-identical. Refusals still precede child creation.
- Twenty-four published step results with 192 matched physical/ideal flows.
- The complete composed native suite passes 587 test cases; the standalone
  native interface suite passes eight executables. The legacy gate passes
  eight plans containing 95 experiments. These are integration gates, not
  behavioral study samples.

The native tests include a callback already queued at the exposed completion
timestamp, multiple completions in one callback, multiple boundary injections,
token invalidation, validation before mutation and zero-time boundaries.
Python checks cover strict framing, child cleanup, unchanged disabled behavior,
earlier local work, late receives, immutable graph projections and publication
only after complete evidence. Twenty-five study-checker tests include reversed
accepted order and fatal missing, duplicated or changed message identities.

## Reproduction and remaining scope

Keep source worktrees, builds and raw evidence separate. Configure the paired
backend with `HTSIM_ENABLE_SIMLLM_RNIC=ON`, `ENABLE_TESTS=ON` and
`SIMLLM_REPOSITORY_ROOT` pointing to the candidate SimLLM checkout. Build
`htsim_rnic` and `txt2bin`. Link `hardware_identity.cpp` against that build's
`htsim_dc`, `htsim` and `simllm_rnic` static libraries, using their source
include directories. The helper constructs idle devices only; the runner
records source and executable identities before invoking it or advancing any
native event.

With paths configured through local environment variables:

```bash
python examples/completion_boundary_v1/run_study.py \
  --candidate-code "$SIMLLM_CANDIDATE" --baseline-code "$SIMLLM_BASELINE" \
  --candidate-backend "$HTSIM_CANDIDATE" \
  --candidate-binary "$HTSIM_CANDIDATE_BINARY" \
  --baseline-binary "$HTSIM_BASELINE_BINARY" \
  --candidate-txt2bin "$TXT2BIN_CANDIDATE" \
  --baseline-txt2bin "$TXT2BIN_BASELINE" \
  --hardware-helper "$SESSION_HARDWARE_HELPER" \
  --out "$COMPLETION_STUDY_OUTPUT"
```

The runner requires a new output directory, clean tracked sources, unchanged
freeze files and the independently retained baseline executable. The public
result records its binary hash and the tree-equivalent baseline checkout.
It refuses to overwrite a prior execution or silently substitute a rebuilt
baseline. `--plan-only` records identities and inputs without native execution.

HTSIM-28's exact native continuation and BACK-38's retained-artifact metric
path are literal deliverables, advancing M4's checked physical execution
foundation. M4 remains in progress. BRIDGE-2 is unblocked at the native continuation
boundary, but still owns cross-step state, online graph frames and bookkeeping
cursors. BACK-72 owns deliberately rejected compositions and additional
platform support. TRAF-8 and TRAF-88 retain captured pipeline serving and the
calendar-aware arrival qualification. The first void execution and the older
blocked congestion-chain study remain as recorded.
