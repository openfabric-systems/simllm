# Single-pass pending publication snapshots

This expectations-only contract precedes the reader refactor and its first
source-paired experiment. CORE-68 owns the host-work refinement. Its retained
native timeout remains VOID, and this local experiment cannot close native
independent-engine qualification.

## Mechanism and preserved authority

The sink's private reader currently converts its selected configuration and
complete publication history into typed immutable tuples. The core then
converts those tuples again while snapshotting the reader's enclosing value.
The second encoding visits the type labels and tuple structure produced by
the first. This is a source observation, not measured timeout attribution.

Introduce raw state and publication reader helpers. Keep the existing
`_deferred_state` and `_deferred_publications` wrappers as one `value_snapshot`
of those raw values, preserving the snapshots used by `price.validate`.
Capture the two raw unbound helper implementations in the private core reader.
Include both helpers in the existing owner, function and code binding ledger.
Return raw state first, raw publications second, then the existing identities,
input, simulation and binding identities. The core takes one full snapshot.

Keep every validation call, the complete ten publication collections, all
configuration fields including `compare=False` fields, provider type and
values, mapper and registration state, and separate object identities. No
cache, history truncation, length-only check, digest substitution or trusted
already-snapshotted marker is introduced. The core private representation
changes; the published model values and lifecycle authority do not.

State and publication values are traversed before simulation values, retaining
the explicitly tested simultaneous-fault precedence below. Universal exception
precedence for arbitrary malformed objects is not claimed. For example,
constructing a raw tuple can encounter a missing attribute before traversing
an earlier nonfinite member. Any such error must still reject publication and
poison the runtime.

## Source pair and finite work law

Run two fresh local worker processes, before then after. The exact before
commit and source hashes are in JSON. All installed package files stay
identical except `simllm/backends/step_sink.py`. The typed snapshot visitor
itself remains byte-identical. Use the same committed study worker for both
roots, verify actual module origins and retain complete package-file hashes.
Each process has a 600-second wall limit and an 8-GiB sampled resident-memory
limit. Sources must be committed and clean and must remain so through final
receipt reread. Raw outputs stay outside the repositories.

Use the small two-rank, one-layer sink configuration in JSON. Populate its
first three publication lists with synthetic frozen dataclass rows; the other
seven lists are empty. Each row has one field `payload`, a tuple containing
integers `0` through `q-1`. Cross history lengths `h={1,4,16}` with row widths
`q={1,4,16}`, plus one empty-history control `h=0,q=1`. These are synthetic
reader mechanism inputs, not claimed model outcomes or hardware traces.

Let `V(x)` be the existing typed snapshot and `N(x)` the number of recursive
visitor calls for input `x`. For the complete raw publication dictionary `P`:

```
N(P)    = 21 + 3*h*(2+q)
N(V(P)) = 94 + 3*h*(10+4*q)
```

Build the pending price and reader before profiling. The measured operation
is exactly one `value_snapshot(binding.read_values(binding.payload))`, as used
by the core. Profile the real functions without replacing any callable. Read
back the full binary profile and require complete identity with its exported
rows, including source path, definition line and function name. Require that
the profile hook is restored.

Let `S` be the raw selected state. Freeze its complete typed snapshot and an
independently counted `N(V(S))` before the measured call in each process.
Its values and count are identical across the source pair; the configured
unused sink work directory is the same external path. The sink creates that
directory, but it must remain empty in this declared local mode.
An independent structural walker counts the full raw reader input before
profiling. The exact relation is:

```
before_visits - after_visits = N(V(S)) + N(V(P)) - 8
```

The eight added visits are the two new source-binding entries, each one tuple
and three identity integers. Successful top-level snapshot calls decrease
from three to one. The before visitor count equals the outer raw-reader count
plus `N(S)+N(P)`; the after count equals its outer raw-reader count. All counts
must be positive and at least the corresponding raw state and publication
input work. The predicted reduction is strictly positive. For fixed `q`, each
additional history row increases the reduction by `3*(10+4*q)`; for fixed
positive `h`, each added payload integer increases it by `12*h`.

These are finite operation-count relations. They do not establish whole-runtime
complexity or elapsed speedup. Complete history checks remain and their total
work can still grow quadratically across a sequence of growing histories.

## Actual supported sink jobs and physical bounds

Separately cross model layer counts `{1,2}` with prior completed steps
`{0,1,4}` using the JSON dimensions, two local ranks, B100 envelope, roofline
efficiency 0.7, ideal host initiation and lower local collective envelope.
Every step has one new decode token, context length 16, no cached token or
finished notification, one sampled token and a distinct stable request ID.

Execute each history through the ordinary sink. Price the next step without
publication at the history's final completion. Initialize the actual
`EngineStepRuntime` clock to that same timestamp and release the next record
there. Bind the price, advance to one
picosecond before completion, then to its exact completion. Compare this with
an independent ordinary-sink job of the same complete input sequence. Retain
every record, result, event, visit, clock observation and all ten full sink
collections. No publication occurs early; exactly one owned result appears
at its due time. Across sources and between ordinary and deferred paths,
complete results and publications are identical.

The dimensions are already per rank. At least one 64-by-128 two-byte dense
matrix of 16,384 bytes is read per layer and rank. Dividing by the declared
8-TB/s memory ceiling gives a 2,048-ps per-layer floor. A deliberately loose
model ceiling is one millisecond
per layer: the finite toy step has less than one million arithmetic operations
and one MiB of tensor traffic per rank, with ideal host initiation and only
the selected local collective costs. This is a bound on the selected model,
not a calibrated hardware latency. Record these bounds before running a job.

Within the bounds, each repeated identical-shape step has constant service
`C(layers)` and the serial job completes at `(history+1)*C(layers)`.
Two layers must cost more than one and no more than twice one layer plus one
nanosecond of final rounding. These unchanged service relations and ordinary
off-path identity are unscored guards. This experiment changes host work only.

## Integrity controls and retention

For each source, apply all fourteen common corruptions named in JSON to fresh
actual pending sink work. The fixture has one ordinary completed one-layer
step and a second positive-service step bound to a runtime whose clock starts
at the first completion. Mutate only after binding. Unless the table says
otherwise, advance to the second step's exact due time and call `complete_due`.
The shorthand `private mismatch` means exactly
`ValueError("pending engine prepared publication values changed")`.

| Corruption | Exact change or action | Expected failure |
|---|---|---|
| prior-row | Replace the first outcome's `host_profile` with `"changed"`, preserving list length | private mismatch |
| provider-values | Change existing provider efficiency from 0.7 to 0.5 | private mismatch |
| config-evidence | Set `resolved_collective_evidence_class` to `"changed"` | private mismatch |
| config-replacement | Replace the config with a shallow copy, preserving its provider | private mismatch |
| provider-replacement | Replace the provider with a new efficiency-0.7 instance | private mismatch |
| prepared-payload-forgery | Replace the prepared outcome with its makespan increased by 1 ps; overwrite the same price's `simulation` and matching `simulation_state` | private mismatch |
| typed-config | Change `emit_packet_breakdown` from `False` to integer `0` | private mismatch |
| nonfinite-state-before-simulation | Set `resolved_collective_evidence_class` to NaN and the prepared simulation's `outcome` to a new plain object | `ValueError("deferred values must be finite")` |
| nonfinite-publication-before-simulation | Append NaN to outcomes and set the prepared simulation's `outcome` to a new plain object | `ValueError("deferred values must be finite")` |
| cycle-publication | Append a list containing itself to outcomes | `ValueError("deferred value state contains a cycle")` |
| unsupported-publication | Append a new plain object to outcomes | `TypeError("unsupported deferred value type: builtins.object")` |
| binding-forgery | Shadow the instance `_deferred_bindings` to return `price.bindings`; shadow `_publish` with a forwarding function around its original bound method | private mismatch |
| early-publication | Call `publish_deferred` while still at the first step's completion, outside retirement | `RuntimeError("publication is outside its owned engine retirement")` |
| duplicate-publication | In the due callback, call `publish_deferred` twice for the same positive-service price | `RuntimeError("deferred sink published or changed outcomes before completion")` |

Record each actual field delta or callable replacement before the trigger,
and preserve the exception class and message for this finite common set.
The two simultaneous faults require nonfinite state or publication values
to fail before the unsupported simulation member.

The six new-helper controls additionally shadow class or instance readers or
forge the binding helper while changing a raw reader. Fix the helper names as
`_deferred_state_values` and `_deferred_publication_values`. Each of the four
class or instance shadows returns `None` instead of the raw values. The two
forgery cases shadow the corresponding instance helper with a deep copy of its
pre-mutation raw values and shadow `_deferred_bindings` with `price.bindings`.
The state forgery also changes the configuration evidence class to `"changed"`;
the publication forgery changes the first outcome's host profile to that value.
All six trigger due completion and must fail with `private mismatch`. The
captured true readers must reject despite forged self-reports. Restore all
field and callable changes before checking that poison prevents further use.

Every corruption poisons its runtime. Repairing the mutated field cannot
resume it. No rejected completion publishes a result or completed visit; the
duplicate-publication case may retain exactly its first authorized sink
publication: outcomes, locality outcomes and collective timing outcomes each
grow from one row to two, while the other seven collections stay empty. It
cannot create a second publication or a core completion. All other mutations
add no sink publication beyond their explicitly injected corrupt data.
Existing tie, zero-service drain and native
bridge regression tests also remain required software gates.

Lock first raw receipts immediately after every process exit, including fatal
exits, before parsing or admission. Re-read full profile files, process records,
case records, module origins, all source files and exact raw domains at the end.
The six evidence-corruption controls reject changed source, profile count,
count-law input, job result, mutation inventory and disk-only raw bytes.
Any failed guard, relation, missing evidence, timeout or failed process makes
the whole study VOID with a null score. Never rescore a failed attempt.

Keep twenty top-level-call exact oracle rows separate from ten paired visitor
reduction instances in one behavioral family. Structural bounds, source and
publication identity, integrity controls and supported-path compatibility are
fatal and unscored. Software test counts are separate. A passing study
qualifies this reader refactor only. CORE-68 still requires its complete fresh
native grid with the original resource limits; CORE-69 and CORE-70 remain
explicitly excluded cancellation and shared-resource paths.
