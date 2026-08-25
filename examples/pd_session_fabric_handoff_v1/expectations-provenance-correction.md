# Expectations provenance correction

The expectations-only commit `7536e08` recorded the correct short pre-freeze
revision, `dc350b6`, but transcribed an invalid full commit suffix in
`expectations.json`. The actual full revision is
`dc350b6996215adf69384c23335b496440042fe7`.

This correction lands after the concurrent-session implementation and before
the packet-handoff implementation and every scored run. It changes only the git
object name used to read the already frozen source bytes. It changes no source
hash, sweep cell, relation, bound, evidence class, fatal guard or acceptance
bar. The original chronology remains visible and this correction is not
presented as part of the expectations-only commit.
