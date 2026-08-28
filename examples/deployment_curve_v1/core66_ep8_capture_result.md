# CORE-66 EP8 capture result

## Frozen capture and deviation ledger

The EP8 capture did not receive a hardware allocation. The frozen cell was two
GH200 nodes with four GPUs per node, eight ranks, four routed experts resident
per rank and 32 experts total. Each rank was configured for batch 32 and
key-value cache length 2,000. Multi-token prediction was disabled. The model
used dummy weights, data-parallel attention, the data-parallel language-model
head, DeepEP, three dense layers, one mixture of experts layer and one measured
decode iteration. SGLang was pinned at
`bfeae4e79a8dc4600e006f1a5fbc85321a01c1a3`.

The scheduler preflight rejected that exact eight-GPU request before creating
a job. Partition `gh-hourly` declares `gpu_hourly` as its default quality of
service, and the quality-of-service definition permits eight GPUs per user and
eight per job. The requesting user association permits only `gpu_general`, so
the explicit frozen request failed with
`allocation failure: Invalid qos specification`. No real submission followed.
The deciding number is therefore zero allocated GPUs and zero GPU-hours.

The signed deviation ledger remained frozen before that refusal:

- Eight rather than 72 expert-parallel peers biases dispatch and combine
  service downward.
- Thirty-two rather than 256 unique routed experts raises uniform-routing
  locality and biases remote traffic downward. Grouped-kernel occupancy is
  indeterminate.
- Thirty-two unique slots omit the registered 288-slot population and its
  three-plus-one-redundant cohort. Locality and duplicate-residency effects are
  indeterminate.
- Four rather than 61 transformer layers lowers raw step service by
  construction. Only separately identified per-layer services may use the
  frozen multipliers.
- Two rather than nine nodes lowers participant count and fabric contention,
  biasing aggregate communication service downward.
- Four rather than eight GPUs per node raises the cross-node peer fraction and
  can increase fabric service per byte.
- Four routed expert slots per rank match the registered residency, so no
  routed-weight residency bias is expected.
- Eager semantic instrumentation raises host launch overhead. Deterministic
  kernel and DeepEP service remain identifiable, but raw eager step time cannot
  become registered graph-mode step service.
- Any fallback from DeepEP invalidates communication pricing and forces null
  signed movement.
- Dummy weights preserve tensor shapes and byte demand but do not provide
  production routing statistics.

None of these differences promotes an EP8 duration to measured EP72 service.

## Physical identities and DeepEP services

No physical SGLang launch ran, so zero of the 37 semantically classified but
physically unbound rows received a binding. Attention, mixture of experts and
data-parallel language-model-head backend identities remain unavailable.
DeepEP dispatch and combine peers, payload bytes and durations are unavailable.
The high-bandwidth memory counter permission was not reached, so per-kernel and
per-step reads and writes are unavailable. Routed expert IDs, assignment counts
and local physical slots are also unavailable. The `1/64` count-and-weight and
`1/9` assignment candidates remain physically unchecked.

## Signed movement

The calibration-only movement of the standard-decode anchor remains null, not
zero. Neither required correction direction was priced: there is no nonzero
DeepEP dispatch/combine service and no rank-preserving high-bandwidth memory
counter result. The downward DeepEP correction was not published alone. The
common `61/4`, dense `1`, mixture of experts `58`, step `1` and output `1`
multipliers were not applied to a measurement, and no constant was fitted.

## Project disposition

CORE-66 stays open and is blocked on adding `gpu_hourly` to the requesting
user's `gmerlin7` `merlin` association. Once that external policy change is
made, the already frozen EP8 cell can be submitted once. No milestone moved,
no fifth scored run occurred and the registered EP72 cell remains impossible
on this project cluster.

This result does not bind the 37 launch identities, price DeepEP, decide the
high-bandwidth memory candidate, check either routing scale, move the
standard-decode anchor or claim EP72 service.

## Guard and disclosure

The expectations-only commits `92e0ebe` and `991ac0b` preceded staging and
scheduler preflight. No held-out value was used in arithmetic, prediction,
fitting or publication, and no incidental held-out exposure occurred. Six
bounded registry and hardware-remainder reads produced twelve ordered reader
events; every paired forbidden ledger is empty. Pytest, ruff, documentation
checkers and git plumbing remain automated-process exemptions.
