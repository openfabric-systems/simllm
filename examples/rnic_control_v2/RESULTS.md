# Physical transport-control producer results

## Scope and provenance

This study closes HTSIM-15 and HTSIM-16 at the clauses registered in
`docs/modules/backends.md`. Expectations were frozen before implementation in
SimLLM commit `51af85937d6b1d3c36f6d841c6445d98ef84c2d3`, against SimLLM base
`90ada43070adb3b1e624b6819aff34d8620e8571` and htsim base
`4885c647eecdfdf81479d1df052223c016ad086b`. The check-only form of the
registered command passed before that freeze and created no output directory.

The result-producing run used SimLLM implementation commit
`89dbb4b0d275fd893799792491411e4bd2c24a2d` and htsim implementation commit
`d716067acc3f7ef89807e6ab310197dfbdea99f7`. Bulk build and raw observations
are under
`${SIMLLM_DATA_ROOT}/wave5-runs/codex/htsim1516_control_producers/control-v2`.
The tracked [`results.json`](results.json) is byte-identical to that run's
summary and has SHA-256
`73c20588366ca0cdef61843170b17d9cf4f582f9520ac99e9de6ea88a8502f17`.

An earlier invocation stopped after the native gate because its inherited
Tier A run root was not forwarded. That external directory is retained as
`control-v2-failed-tier-a-root`. A later successful run made before the two
implementation commits is retained as `control-v2-precommit-diagnostic`; it
is diagnostic only and is not the evidence cited here.

## Registered acceptance clauses

HTSIM-15 requires: "add a timestamped dynamic-link transition producer to an
htsim runtime and advertise the ABI-v2 capability only for that enabled
path." The DCQCN endpoint serializer now applies scheduled down and up events
from one physical event source. The two registered cells emit exact down and
up rows at 1,000 ps and at 201,000 or 401,000 ps. Only their enabled variants
advertise `dynamic_link_events`; disabled and no-transition variants emit no
control rows.

HTSIM-15 also requires: "The disabled path must preserve the ABI-v1 and
ABI-v2 no-transition baselines exactly." Both accepted ABI-v1 artifacts keep
their frozen SHA-256 values. The six ABI-v2 physical cells pass the repository
`BypassArtifacts` comparison with no changed input or behavioral artifact
when control observation is disabled. The paired BACK-34 run independently
keeps the accepted full-quantum ABI-v2 projection exact. Empty transition
schedules retain the existing static runtime path and advertise no dynamic
capability.

HTSIM-16 requires: "Emit packet-keyed ECN and CNP plus effective rate updates
from the DCQCN policy, PFC submission, pause and resume from the lossless
fabric, and link-state observations from the HTSIM-15 transition source."
The four-flow and eight-flow ECN cells carry physical packet-attempt identity
through ECN and CNP. Their first real CNP changes the effective rate from
400,000,000,000 to 200,000,000,000 bit/s. The 64 KiB and 128 KiB lossless
cells each carry real PFC submission, arrival, eligibility-to-zero and resume
observations. The dynamic cells use the transition source described above.

HTSIM-16 requires: "Advertise each capability only when its real producer is
active." Every enabled condition has its condition-specific capability set.
Every disabled condition retains packet attempts but advertises no congestion,
policy-update, PFC or dynamic-link capability. The full native suite also
passes the directed unsupported-capability rejection cases.

HTSIM-16 acceptance further requires the actual policy and fabric, late CNP
correlation, and exact compatibility. The registered probe constructs the
physical `DcqcnAtlahsRuntime` and ns-tm3 policy rather than a test runtime.
Late CNP correlation after packet delivery is observed 4, 8, 8 and 8 times in
the two ECN and two PFC cells while each parent extent remains live. The six
enabled-versus-disabled comparisons preserve completion timestamps, issued
and terminal token order, packet events, authority counters and physical
counters. The observer paths consume no random sample, and the seed-9
physical projections remain identical. ABI-v1 raw observations and summary
remain byte-identical to their accepted references.

All registered clauses are demonstrated. No residual HTSIM-21, HTSIM-22 or
HTSIM-23 entry is created.

## Scored relations

The registered genuine-risk result is 15 of 15:

- CNP rate delta: 2 of 2. Both four-flow and eight-flow conditions move by
  `-200,000,000,000` bit/s at 8,334,080 ps, within the frozen
  `[-300,000,000,000, -1]` band.
- PFC pause duration: 2 of 2. Both payload conditions pause for 81,920 ps,
  within the frozen `[1, 1,000,000,000]` ps band.
- Dynamic-link completion delta: 2 of 2. The short hold moves completion by
  +119,080 ps and the long hold by +319,080 ps, within their frozen bands.
- Dynamic-link duration spacing: 1 of 1. The long-minus-short completion
  response is +200,000 ps, within the frozen `[190,000, 210,000]` ps band.
- ABI-v2 control-disabled physical identity: 6 of 6 conditions have no
  changed `BypassArtifacts` input or behavioral field.
- ABI-v1 byte identity: 2 of 2 artifacts match exactly.

The accepted ABI-v1 SHA-256 values remain
`37a4e9cf88a1b60094409150dfad25599eb77cbf268b3d08bfacf527e493a26a`
for raw observations and
`00ef7e4f5bdbd38f4eabe9ba42dc75f56de528c8751b93e6eef4a3089fa61004`
for the summary.

## Entailment and unscored evidence

Every scored family is evaluated directly from generated raw observations
before capability, token, geometry, schedule, counter or exact-row checks.
The CNP can arrive without producing the signed rate change. PFC can emit a
zero or out-of-band interval. Exact transition rows do not force either
completion response. Installing an observer can still perturb timing, token
order, counters or packet projections. Semantic acceptance does not force
byte identity. Each scored instance can therefore fail in a run that reaches
it and is not entailed by an earlier fatal oracle.

Exact packet geometry, event order, token correlation, capability matrices,
balanced PFC sequences, exact link rows, counters and quiescence are fatal
unscored. Reference digests checked at entry are by-construction change-set
guards and are also unscored.

## Gates and deliberate omissions

The Release build's complete htsim CTest suite passed 374 of 374. The Tier A
ABI-v1 gate passed and retained both accepted artifacts exactly. The paired
BACK-34 study proves the accepted full-quantum ABI-v2 projection remains
exact.

This closure does not claim persistent DCQCN state across WQEs, algorithm
calibration, Tier B packet reachability, TTFT or TPOT. HTSIM-5 and HTSIM-9
retain those separate scopes. No submodule pin, push, branch rewrite or bulk
artifact commit is part of this worker change.
