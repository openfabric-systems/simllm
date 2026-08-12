# HTSIM-8 backend validation gate expectations

This is the expectations-only record for HTSIM-8. It precedes every backend
implementation change and every result-producing command in this study. The
study repairs validation infrastructure, not a modeled timing mechanism, so it
does not claim a TTFT or TPOT relation. Its decision-relevant outcome is
whether the backend gate can remain admissible release evidence.

## External-source audit and baseline decision

The audit used HTSIM commit
`fc4400e4ca619223481536632074045cb6af2756`, the backend commit against which
this evidence was authored. The result report records the source commit that
was actually observed. It does not assert that a live SimLLM submodule pin must
equal this literal.

- HTSIM `htsim/sim/datacenter/commit_check.sh:1-75` runs eight validation
  plans. Lines 67, 70 and 75 invoke the validator, a baseline extraction and a
  regression comparison without propagating every failure. Lines 43-49 also
  fetch a remote even though a local gate does not need network state.
- `htsim/sim/datacenter/validate.py:108-154` initializes both FCT extrema to
  zero, leaves them there when no completion row is parsed, divides the two
  zeros, and dereferences the last output row even when none exists. Lines
  156-158 print a child-process error without returning failure.
- `htsim/sim/datacenter/check_regressions.py:19-31` exits successfully when a
  baseline file is absent. Its later comparison distinguishes only a previous
  pass becoming a new failure.
- `git ls-tree -r fc4400e4 -- htsim/sim/datacenter/validate_outputs` is empty.
  The repository `.gitignore:10` also ignores `*.out`. There is therefore no
  checked-in baseline authority to compare with.

The repair removes the historical output comparison outright. Adding
baselines would require inventing outputs after the failure was discovered,
force-tracking ignored files, and defining a separate refresh authority. Such
artifacts could not establish the provenance that the current gate implies.
The repaired gate instead uses the already-authored absolute FCT and
completion-count bounds in each validation plan. Any validator failure is a
gate failure.

If either scored rejection relation fails, the design decision changes: the
script cannot be cited as a release gate and HTSIM-8 remains open. If the
default eight-plan gate cannot pass after both negative controls reject, the
repair is also incomplete.

## Fixed fixtures

The focused study creates three temporary validation plans outside Git and
passes each plan explicitly to the real `commit_check.sh` entry point:

1. `passing-control` declares one connection. Its executable emits one
   source-authored UEC completion row at 10 us plus a packet summary, and both
   its per-flow and tail limits are 20 us.
2. `child-exit-defect` declares one connection. Its executable exits with
   status 23 and emits no completion.
3. `zero-completion-defect` declares one connection. Its executable exits zero
   but emits no completion row.

The harness removes the three plan, matrix and executable fixtures after the
gate invocations. Captured stdout, stderr and return codes remain under the
external study output. No deliberate defect remains in the backend checkout.

The exact source-authored completion spelling is audited from HTSIM
`htsim/sim/uec.cpp:867-918`: it starts with `Flow <name>`, contains
`finished at <microseconds>`, and may carry `total messages <count>`. The
validator may parse this vocabulary without depending on token positions.

## F1: passing control, fatal and unscored

The passing control must make the real gate exit zero and report one completed
connection. This is an author-defined fixture and a necessary positive
control, so it is fatal-unscored. If it fails, the rejection results are void:
a gate that rejects everything has not demonstrated useful fail-fast behavior.

The default gate must later run all eight tracked validation plans and exit
zero. That native acceptance run is a separate support class, not a scored
relation instance.

## R1: raw gate rejection family

The harness invokes the real gate separately for the two defects and records
each raw gate return code before parsing its diagnostic text.

- `child-exit-defect` passes only when the gate return code is nonzero and the
  child status 23 is named in the captured diagnostic.
- `zero-completion-defect` passes only when the gate return code is nonzero,
  the diagnostic explicitly says that zero flows completed while one was
  expected, and neither `ZeroDivisionError` nor a traceback appears.

R1 has two scored instances. Both are genuine risk. A validator that prints an
error but exits zero, a shell loop that retains only its final command status,
or the original zero-flow division can reach and fail either predicate.
Planned genuine-risk fraction: `2/2`.

### Entailment analysis

The study evaluates each raw gate return code before it checks explanatory
diagnostic text. No earlier exact oracle pins those return codes. The positive
control constrains a different plan and does not entail either rejection. The
deliberate inputs are fixed, but propagation through the repaired gate is live
behavior and was the defect under test. R1 is therefore genuine-risk evidence.

## Closure scope

HTSIM-8 registers three clauses:

| registered acceptance clause | frozen evidence |
|---|---|
| "Add checked-in baselines or remove that compare" | Source audit plus removal of the absent-baseline comparison and remote fetch. |
| "fix zero-flow diagnostics" | The zero-completion raw gate instance must reject with a stable empty-case diagnostic and no exception. |
| "make every failed command fail the gate" | The child-status-23 raw gate instance, the positive control and the complete eight-plan native gate. |

Any clause not demonstrated moves to HTSIM-25 or HTSIM-26 with a categorized
priority and difficulty tag. A fatal-control failure voids closure rather than
reducing a score.

## Registered command and check-only dry run

Bulk outputs remain outside Git. The registered focused command is:

```bash
HTSIM_SOURCE_ROOT="${HTSIM_SOURCE_ROOT:?configure the backend source}" \
.venv/bin/python examples/htsim_commit_gate_v1/run_study.py \
  --out "${SIMLLM_DATA_ROOT:?configure the data root}/htsim_commit_gate_v1"
```

The same command with `--check-only` must run before this expectations commit.
Check-only validates the backend source path, all three fixture registries,
the exact gate invocation and external output placement. It prints the plan
and creates no artifacts. The untracked harness present at freeze time encodes
only these frozen literals, orchestration and check-only validation. It
contains no backend repair or observed outcome.
