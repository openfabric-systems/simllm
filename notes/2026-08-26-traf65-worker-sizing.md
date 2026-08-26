# TRAF-65 worker sizing

Date: 2026-08-26

Size: large.

TRAF-65 spans an expectations-only freeze with 80 stable cases, a GPU-less
local validation arm, a CUDA hardware harness with three producer classes, a
resumable four-A100 Slurm campaign, and an htsim-side candidate NVLink packet
service. The work crosses the SimLLM repository and its pinned htsim submodule,
requires an explicit compatibility path, and must leave all existing accepted
study artifacts byte-identical.

The planned commit boundaries are:

1. Freeze the five-corner catalog, per-case expected bands, fatal guards, and
   decision rules before adding the harness or running a timed cell.
2. Add the local mock and compile-check path, hardware producers, paced Slurm
   runner, candidate htsim service, conformance fixtures, and tests.
3. Record the local result, the still-open hardware remainder, the maintenance
   window, exact resume commands, and final validation evidence.

The hardware portion cannot run before the Merlin maintenance reservation ends
at 2026-08-28T06:30. It remains a digest-pinned, resumable remainder. Candidate
parameters and local model output are not hardware measurements; only the
already published A100 envelope rows may be described as measured in this arm.
