# First shared-handoff campaign: VOID

The first campaign is VOID because its compatibility checker charges another
engine's service to the request being checked. CORE-71 stays open. The failed
execution and its first receipts are immutable; correcting the checker cannot
turn this execution into a qualified result.

Eleven fresh native serving processes ran under execution commit
`a09cbf27d54d8da8ff9bf6abdb82a64c713bb9bf`. The original freeze
`2ad72f72d947e636ea384bfa40f365da27e1ec5a`, deadline amendment
`0e0e71aa586b2ccd2d786534d6b686d888e300b0` and identity amendment
`4c0cf5f4494d4534f4cdd481543827bbb3528032` precede implementation and execution.
The native frontend is vLLM 0.27.1 with virtual workers, modeled compute and
the pinned ideal packet endpoint. No GPU kernels, weights or physical network
were measured.

All eight primary packet processes and both persistent-session processes
completed admission, covering 24 requests. The next process,
`before-serialized-off`, completed its two requests but failed the fatal
`native-step-vector:decode` guard. The campaign stopped there. Its behavioral
score is null; neither these requests nor the earlier packet observations
close a task.

## What the failed guard exposed

The serialized compatibility clock runs one selected engine step at a time.
With two ready decode engines, it alternates between them. Each request waits
while the other engine takes its turn.

The frozen service model supplies a 77.952-microsecond decode step. The first
request's second step starts at 346.752 microseconds and finishes at 424.704
microseconds. Its previous token completed at 268.800 microseconds. Thus the
155.904-microsecond interval between tokens contains 77.952 microseconds of
waiting plus 77.952 microseconds of service. The failed guard incorrectly
expects that entire interval to be one step's service. The native record and
the already frozen reference scheduler keep these quantities separate.

The physical checks precede this observation. The inherited resident-weight
model gives a conditional 40.108032-microsecond memory floor per rank; the
77.952-microsecond decode service is above it. For two serialized requests,
two 95.424-microsecond prefill steps plus eight 77.952-microsecond decode steps
give an exact 814.464-microsecond makespan when the declared transfer is off.
The captured makespan and complete request-time vector meet that relation.
The later per-step guard fails because it assigns the intervening engine's
work to the wrong interval. This diagnoses the reader, not physical hardware
accuracy.

## Evidence and project effect

The [portable receipt](void-frozen-v1.json) records the first raw summary
digest and failed guard. First and final inventories match for all 1,283 raw
files and all 22 root receipts. No reporting or receipt-integrity failure
replaced the original error.

CORE-71 requires a prospective correction of the compatibility service-vector
interpretation and a fresh complete 40-request campaign. The checker must use
the frozen reference's individual service starts and finishes, while retaining
the full source-pair comparisons, request times and packet bounds. No tolerance,
workload, timeout or memory limit is relaxed.

This finding does not change CORE-68's qualified independent-engine result.
It does not close CORE-71, qualify the unexecuted compatibility pairs, or
establish physical fabric, retained tensor-buffer or large-model latency
accuracy. Those limits remain with CORE-70, TRAF-64 and the deployment tasks.
