# CORE-66 feasible capture result

## Achieved capture and deviation ledger

No hardware capture was submitted. The registered EP72 cell remains blocked
because this project has three four-GPU GH200 nodes rather than nine eight-GPU
nodes. The feasible cell was frozen as EP12 over all twelve GPUs, with four
logical experts per rank and 48 total, batch 32, KV length 2,000, four layers,
MTP disabled, dummy weights, data-parallel attention, the data-parallel LM head
and DeepEP. It would have measured one decode iteration.

The frozen cell differs from the registered capture in every scale-sensitive
topology dimension. It has 12 rather than 72 peers, three rather than nine
nodes, four rather than eight GPUs per node, four rather than 61 layers and 48
unique experts in 48 slots rather than 256 unique experts in 288 slots. Fewer
peers and nodes are expected to bias dispatch and combine service downward.
The smaller expert population raises the uniform-routing local share, also
biasing remote traffic downward, while omission of the registered
three-plus-one-redundant rank cohort has an indeterminate routing and grouped
kernel effect. Dummy weights preserve shapes and byte demand but do not make
the routed IDs representative of production routing.

None of these differences would have promoted an EP12 duration to a measured
EP72 service. The capture was scoped only to physical identities and physics.

## Physical identities and DeepEP services

No physical SGLang launch identity was obtained, so zero of CORE-65's 37
semantically classified but physically unbound rows are newly bound. Attention,
MoE and data-parallel LM-head backend identities remain unavailable. DeepEP
dispatch and combine peers, payload bytes and durations are unavailable. HBM
read and write counters and per-layer routed expert IDs, assignment counts and
local slot IDs are also unavailable. Consequently neither the `1/64`
count-and-weight candidate nor the `1/9` assignment scale received a physical
check.

## Signed movement

The calibration-only signed movement is null, not zero. Both required
correction directions are missing: no nonzero DeepEP dispatch/combine service
was measured, and no rank-preserving HBM counter pass was captured. Publishing
the downward DeepEP correction alone is forbidden.

## Protocol disposition

The worktree is protocol void before hardware allocation. Before the required
first reader commit, a case-sensitive repository search listed and grepped
result artifacts, and the full pytest collection was run twice. Those test
processes opened record and result artifacts without contemporaneous CORE-66
reader rows. No held-out MTP number appeared in command output or entered
CORE-66 arithmetic, but it cannot be certified that the broad test processes
did not decode one. The literal empty forbidden-access requirement is therefore
not met and cannot be repaired in this worktree.

The later committed reader behaved correctly: its four per-tranche forbidden
ledgers are empty and every allowed access has a BEGIN event before source open.
That later evidence does not erase the earlier incidents. No fifth scored run
was performed, no constant was fitted and the shared GH200 partition consumed
zero GPU-hours.
