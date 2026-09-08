# Cross-profile message identity checks

These are post-specified regression checks for a study-checker correction.
The first execution has already occurred and remains void with its original
evidence. The original behavioral expectations in commit
`4d75849189df4a4b9113ee36101b246559cbf47b` stay unchanged. This correction
changes no simulator, GOAL input, native sequence, timestamp, parameter,
physical bound or acceptance relation.

Native accepted sequence numbers identify the order in which one runtime
accepts work. Different transport schedulers can make opposite directions
ready in a different order. That sequence is therefore a local authority
cursor, not a cross-profile message identity.

Before changing the checker or repeating the frozen population, require:

- Pair ideal and physical flows using batch, link rate, step, original graph
  operation ID, logical GOAL flow ID, source, destination, tag and payload.
  Each key occurs exactly once in each profile, and the complete key sets
  agree. The existing population has 192 flows per profile, derived from four
  configurations, three steps and sixteen messages.
- Preserve each runtime's contiguous sequence, native flow and work-queue
  identity checks. No row or timestamp is rewritten. Record both local
  sequences in the normalized diagnostic, without requiring equality.
- Derive each normalized flow completion time from the exact physical and
  ideal rows joined by that key. A permutation of local accepted order alone
  preserves the mapping. A missing or duplicated key, or a changed logical
  message identity, is fatal and voids the run.
- Keep all original physical floors, graph joins, timing oracles, behavioral
  relations and sixteen compatibility controls. Run the identical complete
  population in a new evidence directory after the correction, retaining
  both runs and their actual source chronology. The first result is never
  rescored or replaced by the repeat.

Synthetic tests cover reversed accepted order, missing and duplicated rows,
and mutation of each message identity field. These checks support a corrected
comparison method; they are not public pre-registration of an unobserved
outcome.
