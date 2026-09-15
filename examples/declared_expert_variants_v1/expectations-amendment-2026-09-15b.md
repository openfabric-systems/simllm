# Declared expert variant amendment, 2026-09-15 (cell V13)

This amendment is post-specified to an observation made while implementing the
builder, and it is frozen before the study harness exists and before any run.
The original freeze of 2026-09-15 stays in place unchanged; this file states
what it got wrong, what replaces it, and what is not affected. The baseline
amendment of the same date is untouched and remains baseline-only.

## Refuted assumption

Cell V13 stated that over the all-owner manifest of cell V11,
`ExpertPlacementSnapshot.from_manifest(manifest, ep_ranks)` carries
`8 * 24 * 32 = 6,144` owner entries and that `owner_map()` collapses them to
768 keys, so seven of every eight entries are lost to a projection that assumes
one owner per `(layer, expert)` pair.

That is not what the code does. `simllm/traffic/routed_moe.py` validates the
snapshot on construction: `validate_expert_placement_snapshot` collects the
`(layer, expert)` keys and raises
`ValueError("snapshot.expert_owners: expert has multiple owners")` when they
are not unique. Construction therefore fails before `owner_map()` is ever
reached, and nothing is silently lost. The freeze described a lossy projection
where the tree has a refusal.

The claim was written from the shape of `owner_map()`, which does collapse
duplicate keys, without checking the validator that runs first. That is the
error, and it is the kind the freeze exists to catch.

## Replacement rules

- The all-owner manifest of cell V11 carries exactly
  `8 * 24 * 32 = 6,144` ownership entries over exactly 768 distinct
  `(layer, expert)` pairs. Those two counts stand as frozen and are what the
  study asserts.
- `ExpertPlacementSnapshot.from_manifest` over the whole eight-rank expert
  parallel group of that manifest is refused with
  `ValueError("snapshot.expert_owners: expert has multiple owners")`. The
  study asserts the refusal, not a collapse, and the `owner_map()` row is
  withdrawn.
- A single-rank slice of the same manifest, for example `ep_ranks = (3,)`,
  builds and carries 768 entries. The projection can see one owner at a time
  and cannot represent replication at all. This row is new.
- The expert-parallel-on twin of the same world builds and carries 768
  entries, unchanged, which is the accepted PLACE-3 cell C2 count. This row is
  new and is the control that proves the refusal is about replication rather
  than about the manifest.
- The claim about the vLLM step schedule is unaffected: it still refuses the
  geometry, because `dims.local_num_experts * len(ranks)` is `32 * 8` and not
  32.

## What this changes for the conclusion

It strengthens it. The freeze argued that the enabled PLACE-7 path is not
live-reachable today and that no live cell is warranted. A consumer that fails
closed is a better reason for that than a consumer that loses rows quietly, so
the conclusion, the zero scored denominator and the decision to run no backend
all stand and are better supported than when they were written. The residual
work stays with PLACE-8, which owns the uneven and all-owner consumer path,
and PLACE-8's entry gains this finding in the closing change.

## What does not change

Every other cell keeps its literals: V1 through V12 are untouched, including
the 6,144 and 768 counts that V13 shares with V11. The evidence classes are
unchanged, V13 remains a recorded finding rather than a structural guard or a
scored row, and the two fatal study `--check` identities and the five reference
manifest digests are unaffected. Neither task's closure condition moves:
PLACE-7 and PLACE-11 close on the same cells as before, since V13 was never
among them.
