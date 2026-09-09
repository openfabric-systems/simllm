# Call-local dependency lookup

This expectations-only contract precedes implementation and the first run.
CORE-52 owns this bounded host-cost change. Its target-scale feasibility task
stays open until a separately frozen native successor completes. The retained
host diagnostic and its earlier VOID attempts remain unchanged.

## Mechanism and authority

The host checks that every modeled dependency appears exactly once in the
generated communication schedule. For every serialized dependency, its current
checker scans the entire ordered effective-edge tuple until the first matching
edge. That repeated search creates temporary edge objects for candidates it
does not use. A call-local dictionary can retain the same first matching edge
while the existing occurrence Counter is built.

The key contains predecessor, target, scope, origin and participant rank.
Retain the first effective occurrence with `setdefault`; keep the Counter,
validation order, full multiplicity checks and final canonical-order checks.
Only this lookup changes. No cross-call cache, disabled validation, identity
alias, timer, modeled service or emitted byte changes. The graph stays the
sole dependency authority and projections remain read-only.

## Source-paired protocol

Use two fresh worker processes, before then after, with explicit repository
roots. The before root is the exact commit recorded in `expectations.json`.
The after root contains this freeze as an ancestor and a committed, clean
implementation. The only changed installed package file is
`simllm/traffic/execution_goal.py`. All other package files and the existing
Granite configuration helper remain identical. Hash every package file and
the study sources before and after execution; verify actual imported module
origins. Never replace the old verifier with a reconstruction of its source.

The same committed worker script runs against each package root through its
explicit import path. Record its own independent source receipt. Each process
has a 900-second host limit and an 8-GiB sampled resident-memory limit. Bulk
files live outside both repositories. A timeout, process failure, changed
source, failed guard or relation makes the campaign VOID with a null score.
Retain logs and initial raw receipts; do not rescore a failed attempt.

## Direct-verifier sweep

Build a valid graph and projection before profiling one direct call to
`verify_execution_goal_projection`. Exclude projection construction from the
count. Use depth `d` in `{1,2,4,8}` crossed with width `w` in `{2,4,8}`, in
that order, for twelve configurations.

Start with ring R0. For each layer j, append w compute operations C(j,r),
ordered by rank, each locally dependent on R(j-1). Append ring Rj, locally
dependent on all those compute operations in rank order. All rings share one
logical queue. Every compute operation has a distinct queue. Rings carry
64 bytes per rank, use the ring algorithm, and span ranks 0 through w-1.
Compute service is 1,000 ps. The final ring is the completion operation.

Let E be the number of effective-edge occurrences, B the boundary occurrences,
S the serialized occurrences, and p_s the one-based position of the first
matching effective edge for serialized occurrence s. The exact formulas are:

```
artifacts = 2*d + 1
E = d*(2*w + 1)
B = d
S = 2*d*w
sum(p_s) = w*(2*w + 1)*d*d
before_calls = E + B + S + sum(p_s) = (2*w + 1)*d*(2 + w*d)
after_calls = E + B + S = 2*(2*w + 1)*d
before_calls / after_calls = 1 + w*d/2
```

Count actual `_goal_edge` invocations with the standard deterministic call
profiler. Retain the complete profile and exported call records. Independently
read the binary profile and join function identity to the captured source.
No callable replacement supplies a count. Re-read profiles after admission.
The canonical artifact subgraphs have no effective inter-operation edges,
so their rerenders add no `_goal_edge` invocations in this finite family.

The floor is E+B+S conversions under the retained three conversion sites.
The existing linear search cannot inspect more than E candidates per
serialized occurrence, giving the ceiling E+B+S+E*S. Both observed counts
must lie within these bounds before exact matching. This is operation-count
evidence. It neither states whole-verifier complexity nor predicts elapsed
host-time speedup. At fixed width, doubling depth doubles the new count;
the old count follows its quadratic-plus-linear formula exactly.

## Projection and rejection identity

For every sweep cell retain the full graph, effective-edge inventory and
projection: all artifact operation IDs, rank counts, GOAL text, operations,
messages and dependency provenance, plus ordered boundaries and serialized
edges. Before and after values must match without normalization. Both calls
must leave their graph and projection unchanged.

Two additional valid graphs exercise keys sharing endpoints: a rank-local
compute dependency that is both explicit and implicit queue order, and a
distributed dependency expanded into separate participant-rank edges. These
are different full keys, not duplicate full-key effective edges. Their
projections and successful outcomes must remain exact. Current valid graph
validation excludes identical full-key effective duplicates; repeated
projection occurrences still must fail.

Apply the following fourteen corruptions to fresh projection copies in each
of the twelve sweep cells. Compare exact exception class and message between
sources; accepting any corrupted projection is fatal. Rejection comparisons
are unscored guards, separate from successful-call count relations.

1. Change the first serialized predecessor to an absent ID.
2. Change its target to an absent ID.
3. Change its scope to whole-operation and clear its participant rank.
4. Change its origin to logical-queue-fifo.
5. Change its participant rank to another valid rank of the graph.
6. Replace it with a non-edge string.
7. Delete the first serialized occurrence.
8. Duplicate the first serialized occurrence.
9. Reverse the serialized inventory.
10. Move the first required boundary edge into the serialized inventory.
11. Also register the first serialized edge as a boundary with its correct
    artifact indexes.
12. Change the first compute artifact's rank-zero calculation from 1 ns to
    2 ns without changing its owner or dependency provenance.
13. Combine an incorrect first boundary index with corruption 6. The boundary
    index error must remain the first failure.
14. Combine corruption 12 with corruption 7. The edge occurrence mismatch
    must remain the first failure, ahead of the deferred canonical-text check.

Construct mutated edge values legally before invoking the verifier; an error
from the mutation constructor cannot count as a verifier rejection. Snapshot
each corrupted input before and after the rejected call and require identity.

## Supported sink jobs

Use the actual `HtsimStepSink` with the accepted Granite 24-layer dimensions,
B100 envelope, eight local ranks, one node, roofline efficiency 0.7, ideal host
initiation and the lower `intra-node-fixed-cost-v1` collective envelope. No
packet backend, GPU, native serving frontend or weights are used in this slice.

Cross prompt tokens `{8,16}` with serial request counts `{1,4}`. For each
request submit one prefill step and four decode steps, using deterministic
request IDs and dense step indexes. The first decode step carries one new
token, the prompt as cached tokens, phase prefill and context prompt+1; later
decode steps carry one new token, no cached tokens, phase decode and increasing
context. Each next step starts at the prior `StepResult.completed_at_ps`.
The prompt prefill context equals its prompt length. There are no preempted
or finished notifications in this fixed sink workload, and `num_sampled=1`.

Retain all full inputs, results and all ten sink publication collections.
Require exact before/after bytes, source-bound selection and consecutive
step/completion intervals. Record job completion time, equal to the final
completion timestamp of this serial five-step-per-request job. The known
accepted service values in JSON are regression oracles, not new measurements.
Completion must equal request_count*(prefill_service+4*decode_service), so
four serial requests take exactly four times one request. This unchanged
compatibility relation is unscored.

As a conditional check of the existing resident-streaming surrogate, the
320,864,256 per-rank weight bytes over 8 TB/s imply a 40,108,032-ps step floor.
The sum of the declared step services is the job ceiling and exact serial
oracle. This surrogate is not a routed-expert hardware floor; COMP-7 owns that
precision. These sink jobs qualify completion compatibility, not serving
time to first token, time per output token or a deployment frontier.

## Evidence classes and closure

The campaign requires two source processes, twelve direct-verifier paired
configurations, two valid key controls per process, 168 rejected projections
per source, and four paired sink jobs totaling fifty steps per source.
Keep 24 successful-call exact oracles separate from twelve paired count
reduction relations in one family. Source identity, physical bounds,
compatibility, rejection fidelity and receipt guards are unscored. Software
test counts never enter either evidence denominator.

The required stages are protocol, before capture and admission, after capture
and admission, paired relations, corruption controls and final receipts.
Corruption controls must reject a changed source receipt, a modified profile
count, a missing serialized occurrence in the retained positive projection,
a changed sink result, and a disk-only raw-byte change. Lock raw files before
parsing and require exact domains and bytes at final reread. Any violated
fatal guard voids the result, regardless of earlier successful relations.

A passing result makes this lookup optimization defensible and unblocks the
next CORE-52 target freeze. It closes no target-scale or model-fidelity task.
Fresh native target feasibility, independent-engine timing, GPU calibration
and end-to-end deployment frontiers keep their existing owning tasks.
