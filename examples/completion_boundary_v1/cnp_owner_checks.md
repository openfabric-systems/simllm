# External congestion-notification ownership regression checks

BACK-71 fixes a validation-gate regression identified while integrating the
completion-boundary work. These checks precede its repair. The existing
physical composition test has already failed with a missing native reaction
point; that observation is diagnostic history, not a newly registered study.

The external data-center congestion-control model owns its sender-rate
reaction. Its congestion notification packet (CNP) observation passes through
the native work queue's existing token, extent and capability checks. When the
native transmit pipeline owns the flow, its enabled reaction point instead
handles the notification once. Neither path enables a second controller.

Before changing routing, require:

- An external-owner notification reaches the existing validated observation
  path without requiring a native transmit pipeline. Unknown tokens, mismatched
  extents and absent notification capability still fail closed.
- A native transmit pipeline with its reaction point disabled still rejects a
  notification. With the reaction point enabled, the notification changes its
  rate exactly once and retains the existing pacing assertions.
- The existing real external-control composition test retains all its packet,
  marking, rate-reduction, pause/resume and quiescence assertions.
- A physical composition sanity matrix varies sender count in {4,8} and
  payload per sender in {65,536;131,072} bytes, with the existing 64-endpoint
  400 Gbit/s Clos, seed nine, buffer and control settings. Each cell runs with
  packet/control observation absent and present. Completion identities and
  timestamps are exactly equal between those modes. Observations cannot
  change the external controller's packet schedule.
- Floor: completing the shared receiver's payload requires at least total
  payload bytes times 20 ps per byte plus the shortest two-link propagation
  path. Ceiling: each finite fixture finishes and quiesces within one
  millisecond and 100,000 native callbacks. This is an engineering guard,
  not a physical theorem. Report actual completion boundaries, with no
  assumed monotone congestion-control law across the four cells.
- The complete composed native suite and the existing native reaction-point
  and work-queue suites pass. No policy constants or accepted packet timings
  change to make the observation path work.

This repairs control ownership and its gate. It does not calibrate congestion
control, qualify serving latency or close BACK-38's persistent-session study.
Retain the failing gate log and all new matrix evidence separately from the
completion-boundary population. Any violated fatal guard voids these checks.
