"""Closed-loop bridge schemas.

The frontend adapter and the simulator exchange versioned JSON manifests per
scheduler step (or window of steps):

- step manifest, schema ``atlahs-closed-loop-step-v1``: what the scheduler
  ran (request ids, phases, token counts, cache hits) and the virtual time.
- result manifest, schema ``atlahs-closed-loop-result-v1``: the simulated
  time (``simulated_time_us``) and per-flow completions.

Per-step subprocess invocation is the diagnostic mode; a persistent
co-simulator process is planned.
"""

STEP_SCHEMA = "atlahs-closed-loop-step-v1"
RESULT_SCHEMA = "atlahs-closed-loop-result-v1"

__all__ = ["RESULT_SCHEMA", "STEP_SCHEMA"]
