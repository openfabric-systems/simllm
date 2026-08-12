# Arrival admission v1 expectations

This document freezes the WORK-3, CORE-31 and PLAY-12 validation contract
before arrival gating is implemented and before the first admission study run.
The task is live-reachable through a real vLLM scheduler, adapter-produced
`StepRecord`, core `ExecutionGraph`, `CoarseDeviceRuntime`, `StepResult`, and
per-request TTFT.

## Design decision under test

The arrival gate belongs in the in-process replay harness. It reads logical
arrival timestamps from framework-request creation facts, withholds a request
from `LLMEngine.add_request` until the shared virtual clock reaches that
timestamp, and records the successful handoff. It never selects a batch.
After handoff, the framework scheduler remains the only batching authority.

The decision-relevant relation is the per-request movement of queue time when
one fixed follower burst crosses a scheduler-step boundary. If the live
scheduler path does not move TTFT by the exact registered queue-time delta, or
if it changes a request's captured tokens or stop reason, a step-boundary
in-process gate is not an admissible arrival model. The design must then move
to an event-interleavable engine coordinator rather than patching metrics
after execution.

Server mode has a different boundary. An HTTP server advances independently
in wall time and does not expose the same caller-owned `has_unfinished` and
`step` loop. This slice therefore makes no server-mode claim. WORK-4 owns a
future server ingress coordinator that maps external injection to simulated
time without changing framework scheduling policy.

## Frozen external-source audit

The live runtime is vLLM 0.26.0. The source was audited before this freeze.
The relevant package-relative files and complete SHA-256 values are:

- `vllm/v1/engine/llm_engine.py`,
  `17e5edfc625c77e9663368c7d69136e5e5935ee81608a65be3996411d502225e`;
- `vllm/v1/engine/core.py`,
  `3ae1381a6af841e21058c825702382dc66faae45c950ac5acb8495d2d3d05aad`;
- `vllm/v1/core/sched/scheduler.py`,
  `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`;
- `vllm/v1/engine/input_processor.py`,
  `c5673988c0f7cfec268220e3f044e718702c015a4f236c020937cfd40a793f15`;
- `vllm/v1/request.py`,
  `92124fbad28cda49bd06fa12c2c4fd5f53fc9381ddb4dc35f275c5ccfbd27378`.

The audited contracts are:

1. `LLMEngine.add_request` processes one input, creates output state, and only
   then calls `EngineCore.add_request` at
   `vllm/v1/engine/llm_engine.py:218-277`. Holding this call back is therefore
   the framework-entry gate.
2. `EngineCore.add_request` forwards the request to the scheduler at
   `vllm/v1/engine/core.py:431-471`.
3. The scheduler places a new request in its waiting queue at
   `vllm/v1/core/sched/scheduler.py:2120-2142`.
4. Waiting requests are admitted only while the configured running-request
   capacity is available at
   `vllm/v1/core/sched/scheduler.py:665-679`. The study fixes this capacity at
   two so a three-request burst contains real framework queueing.
5. One engine step schedules, executes and updates only requests already held
   by the scheduler at `vllm/v1/engine/core.py:576-606`. `LLMEngine.step`
   returns the processed request outputs at
   `vllm/v1/engine/llm_engine.py:296-334`.
6. The input processor accepts the explicit arrival value that becomes
   request metadata at `vllm/v1/engine/input_processor.py:248-289`. The gate
   passes the logical arrival converted from picoseconds to seconds, but the
   timing authority remains the SimLLM virtual clock and the delayed call.

These citations support the external seam and capacity relation. No exact
author-written call sequence is scored.

## Frozen capture and workload

The tracked Granite length-cap fixture has SHA-256
`36334f3aaa767c46d5f9c8498e02f6c2805a46e5000a57aea2747e17dd5d1341`.
The study derives one deterministic replay trace from its prompt and routing
rows before engine construction:

- anchor `r0` arrives at 0 ps and serves `(38, 39, 40, 41)` with stop reason
  `length-cap`;
- followers `r1`, `r2` and `r3` serve `(61,)`, `(62,)` and `(63,)`, each with
  stop reason `length-cap`;
- every request has two prompt tokens, greedy sampling, no stop string,
  `max_model_len=64`, no chunked prefill, no asynchronous scheduling and no
  prefix caching;
- vLLM has `max_num_seqs=2` and remains the sole authority over which waiting
  request enters each batch.

The offered-load sweep uses a single-follower cell (`r0`, `r1`) and a
three-follower burst cell (`r0`, `r1`, `r2`, `r3`). All followers in one cell
share one arrival timestamp. That timestamp is swept over 750,000 ps and
1,250,000 ps. The first value falls inside the anchor's first service step;
the second falls inside its second service step.

Every nonempty engine-produced step becomes one core graph with one shared
compute operation correlated to the exact scheduled request IDs. Its service
is exactly 1,000,000 ps. The persistent core runtime and the adapter share one
`VirtualClock`. This synthetic service is an exact experiment input, not a
claim about Granite compute time.

## Raw observations and metric definitions

For each request, the study records the input arrival, first adapter
`StepRecord.virtual_time_ps`, first returned-token completion timestamp, exact
returned token IDs, and framework finish reason before applying any oracle.
Define:

```text
queue_ps = first_step_release_ps - arrived_at_ps
service_ps = first_token_completed_at_ps - first_step_release_ps
TTFT_ps = first_token_completed_at_ps - arrived_at_ps
```

The first-token `RequestMetric` must independently report the same TTFT and a
`LatencyAttribution` whose `queue_ps` equals raw queue time and whose remaining
components sum to raw service. This exact decomposition is fatal and unscored.
It is evaluated only after the scored relations below so it cannot entail a
scored pass.

## Scored behavioral relations

### A1: arrival-stagger movement

For every follower present at both arrival offsets, increasing the offset
from 750,000 ps to 1,250,000 ps must increase its raw and reported queue time
by exactly 500,000 ps. Its TTFT must increase by exactly 500,000 ps, its
service must change by exactly 0 ps, and its returned token tuple and finish
reason must remain identical.

The exact first-token values are:

| Load | Offset (ps) | Request | Queue (ps) | Service (ps) | TTFT (ps) |
|---|---:|---|---:|---:|---:|
| one follower | 750,000 | r1 | 250,000 | 1,000,000 | 1,250,000 |
| one follower | 1,250,000 | r1 | 750,000 | 1,000,000 | 1,750,000 |
| three followers | 750,000 | r1 | 250,000 | 1,000,000 | 1,250,000 |
| three followers | 750,000 | r2 | 1,250,000 | 1,000,000 | 2,250,000 |
| three followers | 750,000 | r3 | 2,250,000 | 1,000,000 | 3,250,000 |
| three followers | 1,250,000 | r1 | 750,000 | 1,000,000 | 1,750,000 |
| three followers | 1,250,000 | r2 | 1,750,000 | 1,000,000 | 2,750,000 |
| three followers | 1,250,000 | r3 | 2,750,000 | 1,000,000 | 3,750,000 |

This family has four paired instances: `r1` at load one, plus `r1`, `r2` and
`r3` at load three. These are live-runtime relations and are scored.

### A2: offered-load queue slope

At the three-follower load, the scheduler capacity admits only one follower
beside the still-running anchor. At each arrival offset, `r2` queue time minus
`r1` queue time must be exactly 1,000,000 ps, and `r3` minus `r2` must also be
exactly 1,000,000 ps. All four differences must be strictly positive.

This family is not implied by A1. A1 compares the same request across arrival
offsets; A2 compares different requests at one offset and tests genuine
framework queueing under the offered burst. It has four scored instances.

### Anchor control

The anchor's first-token queue time is 0 ps, service is 1,000,000 ps and TTFT
is 1,000,000 ps in every gated cell. This is a configuration-forced control
and is fatal but unscored.

## Identity off path

`all-at-once` is the default and identity off mode. For both offered loads,
the study runs the old direct loop and the disabled gate from fresh engines
with identical inputs. Their streamed `StepRecord` bytes, returned token and
finish tables, final virtual clock, replay snapshot and bookkeeping ledger
must be byte-identical. The disabled gate must not advance the clock or append
admission facts before the first engine step.

The existing no-replay adapter byte lock remains an additional fatal guard:
`tests/fixtures/vllm/no_replay_r1_p4_steps.jsonl` has SHA-256
`71862c9a49814bef3fc830f647f1b439d9c4d6ad0ef9707be6597528adb1808a`.
Identity checks are by-construction or change-set guards and are unscored.

## Other fatal unscored guards

- the gate consumes `CreatedObjectRecord.created_at_ps`, not the duplicated
  `preplay_arrived_at_ps` metadata value;
- equal-time arrivals retain bookkeeping sequence order;
- no gated request reaches `add_request` before its arrival;
- a successful gated handoff appends exactly one scheduler-entry fact at the
  current virtual time, while a failed callback appends nothing;
- a request scheduled before its bookkeeping arrival is rejected without
  advancing the reducer clock or metric history;
- all exact TTFT decompositions and request token counts pass;
- exact batch shapes, config echoes, source hashes and schema checks remain
  structural and never enter a behavioral denominator.

## Entailment and genuine-risk forecast

The harness captures raw engine observations first and evaluates A1 and A2
before the exact decomposition oracle, batch-shape guards or bookkeeping
checks. Therefore no earlier fatal oracle pins either scored result. The later
decomposition rows are deliberately unscored because they exactly constrain
the same queue, service and TTFT quantities.

All eight scored instances are expected to carry genuine risk. A competent
implementation can release requests too early, advance an idle clock while
work remains, seed TTFT at first schedule instead of arrival, reorder a tied
burst, or bypass the framework waiting queue. Any of those defects can pass
token replay while failing A1 or A2.

## Registered command and pre-freeze dry run

The portable command is:

```text
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" examples/arrival_admission_v1/run_study.py --check-only --run-dir "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/arrival_admission_v1"
"${SIMLLM_VLLM_PYTHON:?configure SIMLLM_VLLM_PYTHON}" examples/arrival_admission_v1/run_study.py --run-dir "${SIMLLM_DATA_ROOT:?configure SIMLLM_DATA_ROOT}/arrival_admission_v1"
```

Before this freeze, the first command was executed with a parser-only harness
using the same basename, arguments, source hashes, tracked fixture and model
cache checks. It exited zero and produced no artifacts, model construction or
results. The parser harness was removed before the freeze, so the freeze
contains no task implementation or study runner.

## Deliberate omissions

This task does not add bursty or MMPP arrivals; WORK-2 remains open. It does
not model framework admission control, rejection, rate limiting, priority or
scheduling policy; CORE-32 owns that optional path. It does not make the
in-process gate a server-mode traffic injector; WORK-4 owns that path. SGLang
joined replay remains PLAY-7, so this study uses the already supported vLLM
replay adapter only.
