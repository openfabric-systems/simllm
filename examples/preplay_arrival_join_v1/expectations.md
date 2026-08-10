# Pre-play arrival join v1 expectations

This document freezes the PLAY-2 validation contract before the arrival join
implementation and before any study execution. PLAY-2 is component-only. Its
specific live-reachability successor is PLAY-3, which consumes the joined
request records in the vLLM adapter and changes scheduler-visible completion,
`StepRecord`, `StepResult`, TTFT and TPOT.

## Frozen inputs

The study uses the tracked
`examples/preplay_trace_v1/granite_length_cap.jsonl` trace as its external
artifact fixture. Its request identity is `length-cap`, its output token IDs
are `(38,)`, its stop reason is `length-cap`, and its frozen SHA-256 is
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.

The study also builds a dependency-free two-request trace through the public
PLAY-1 schema. Request `alpha` has output token IDs `(101, 102, 0)` and stop
reason `eos`. Request `beta` has output token ID `(201,)` and stop reason
`length-cap`. Both requests use one prompt token, one MoE layer, one selected
expert, and valid routing weights. This synthetic trace is an input fixture,
not behavioral evidence.

Two independent parameter families vary:

1. request cardinality and identity: the tracked one-request trace and the
   synthetic two-request trace;
2. arrival realization: zero-origin arrivals and the same arrivals shifted by
   exactly 7,000 ps. The synthetic unshifted arrivals are `alpha=1,000 ps` and
   `beta=4,000 ps`; the shifted arrivals are `alpha=8,000 ps` and
   `beta=11,000 ps`.

Arrival timestamps are integer picoseconds at the framework-entry boundary.
No floating-point seconds conversion is part of this contract.

## Scored behavioral relations

### B1: exact request projection

For every request in all four cardinality and arrival cells, the joined
request projection and its `RequestBookkeeper` framework-request object must
agree exactly on request identity, arrival timestamp, output length, stop
reason, output token IDs, and routing reference. Output length must equal the
number of output token IDs. The number of new bookkeeping objects must equal
the number of arrivals, with no loss or duplication.

Shifting every arrival by 7,000 ps must shift every projected
`created_at_ps` and arrival metadata value by exactly 7,000 ps. It must change
no request identity, output field, routing reference, trace identity, object
order, or non-arrival metadata byte.

### B2: trace authority survives the join

For each cell, the versioned run record must name the exact input trace path
and SHA-256. Every request's versioned routing reference must name the same
trace SHA-256 and the request row it projects. For the tracked fixture, both
the run record and the `length-cap` routing reference must carry exactly
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.

This relation is decision-relevant. If trace identity cannot survive exactly
from input artifact through the run record and every bookkeeping projection,
the adapter-side metadata design is rejected. PLAY-2 would be redesigned
around a first-class core bookkeeping fact before PLAY-3 or PLAY-4 consumes
it.

### B3: cardinality scaling

Moving from the tracked one-request cell to the synthetic two-request cell
must change the number of joined request projections, framework-request
objects, and routing references from one to two exactly. It must not create a
second mutable request authority: each returned projection corresponds to
exactly one immutable bookkeeping object and both share the same stable
object identity.

## Exact-oracle relation

### E1: canonical run-record round trip

Writing a joined run record, reading it strictly, and writing it again must
reproduce identical UTF-8 JSON bytes. The parser must reject unknown fields,
an unsupported schema, a duplicate request identity, a request whose declared
output length disagrees with its token IDs, and a routing reference whose
trace SHA-256 disagrees with the run record. E1 is reported separately from
B1 through B3.

## Fatal unscored structural guards

These checks fail the study but never increase the scored denominator:

- an arrival request missing from the trace is rejected before the
  `RequestBookkeeper` changes;
- duplicate arrival identities, negative timestamps, booleans passed as
  timestamps, and an empty arrival realization are rejected atomically;
- a pre-existing bookkeeping object collision is rejected atomically;
- the trace file is read and hashed before any bookkeeping mutation;
- every joined object is a framework-owned `FRAMEWORK_REQUEST` correlated to
  exactly its own request ID;
- all schema tags are exact and every output token ID remains an integer.

These are structural or configuration-forced properties. They are fatal and
unscored under the repository evidence rules.

## External-source audit

PLAY-2 mirrors no external runtime, hardware specification, or framework
source. Its only external artifact is the already tracked PLAY-1 trace named
above, whose bytes and hash are frozen here. No external-source line audit is
therefore applicable to this slice.

## Registered commands and pre-freeze dry run

The registered study command is:

```text
.venv/bin/python examples/preplay_arrival_join_v1/run_study.py --check-only
```

Before this freeze, that exact command was executed against an argument-parser
skeleton. It exited zero after resolving the tracked fixture and the required
output-root argument default, and it produced no result rows or output files.
The parser skeleton was then removed, so this expectations-only commit
contains no implementation or study harness.

The eventual scored invocation replaces `--check-only` with `--run-dir`
under `/data3/yifeng/simllm-dev/wave2-runs/codex_play23_arrival_replay/`.
Gate commands are the repository-standard ruff and pytest invocations and are
not study commands.

## Interpretation

A pass establishes a strict, atomic and provenance-preserving projection of
arrival and oracle facts into existing bookkeeping. It does not establish a
live metric effect. PLAY-3 must prove that the adapter consumes these facts
and changes scheduler-visible completion while its bypass preserves the
accepted baseline exactly.
