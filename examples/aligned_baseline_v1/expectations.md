# Aligned baseline: frozen expectations

BACK-68 separates a scheduling explanation from impossible byte service on
pinned htsim `617ce20`. This expectations-only commit precedes implementation
and all runs. No simulator results inform these predictions. The worktree
root and its ancestors contain no AGENTS.md; the supplied wave contract is
binding. Results cite this commit's full hash.

## Matrix and identical phases

Run 40 primary cells plus 16 isolated controls, all on 64 GOAL endpoints.
Every message starts at zero with no dependencies. Compare identical GOAL
bytes across profiles `rnic-nn` and `rnic-cn`, at 400 and 200 Gbit/s.
The physical reference is `examples/m1/topologies/clos_64_400g.topo`;
the 200G copy changes both tier rates, preserving 1,000 ns per link.
Use the binaries under SIMLLM_HTSIM_BUILD and retain executable hashes.

Two-flow cells: sizes S and 4S, S in {262144, 1048576} bytes. Destination
is rank 0. The two placements put both sources on the receiver leaf
(ranks 1 and 2), or both on one remote leaf (ranks 8 and 9). The lower
source carries S and the higher source carries 4S. This controls receiver
contention while varying path length. There are 16 cells.

Many-to-one cells: F in {8, 16, 32} distinct senders, B in {65536, 1048576}
bytes per sender, destination 0. Take the first F ranks from rail-major
order 0, 8, ..., 56, 1, 9, ..., 63 after removing rank 0. Thus F counts
actual senders, unlike the width study's participating-rank count. There
are 24 cells. Include local-leaf senders; these are real RNIC endpoints,
not the GPU-local bypass in the supported step path.

Isolated controls: F=1, B in {65536, 1048576}, source 1 (local) or 8
(remote), each rate and profile. These 16 cells test whether any ratio
below one occurs without receiver sharing. All tags are 1000, keyed with
source and destination. GOAL, raw completion CSV, manifest, command argv,
per-flow normalization and every completion-prefix floor row go to a fresh
external directory under SIMLLM_DATA_ROOT. Never reuse old bulk evidence.

## Physical sanity and predictions before measurement

At 400G, serialization is 20 ps/byte; at 200G, 40 ps/byte. Ideal propagation
P is 2,000,000 ps. Physical path propagation is 2,000,000 ps for same-leaf
and 4,000,000 ps for remote-leaf flows. An individual flow's floor is
its own bytes times ps/byte plus its path propagation. Print this floor
beside every p50. Physical ceilings are unbounded: link capacity gives no
finite delay ceiling under arbitration, queueing and flow control.

For continuous, ideal max-min service, the two-flow small completion is
2S/R + P and the large completion is 5S/R + P, converting bytes to bits.
These analytical reference values, in ps, are fixed before running:

| S (bytes) | Rate (Gbit/s) | Small fluid FCT (ps) | Large fluid FCT (ps) |
|---|---|---|---|
| 262144 | 400 | 12485760 | 28214400 |
| 262144 | 200 | 22971520 | 54428800 |
| 1048576 | 400 | 43943040 | 106857600 |
| 1048576 | 200 | 85886080 | 211715200 |

These are exact fluid formulas, not zero-tolerance predictions of the
packetized `rnic-nn` profile. The latter accounts for 4096-byte payloads,
64-byte headers and a destination packet slot. Let q=4160*ps_per_byte,
N=total_payload/4096. Its conditional phase envelope is between
payload*ps_per_byte+P and (N+2)*q+P for this one-receiver matrix, assuming
work-conserving serialization. Record measured packetized baselines and
fluid residuals separately. Do not silently redefine the requested ideal.

For F equal senders, the fluid ideal completion is F*B*ps_per_byte+P.
At 400G and B=65536, these are 12485760, 22971520, 43943040 ps for
F=8,16,32. At B=1048576 they are 169772160, 337544320, 673088640 ps.
At 200G, subtract P, double serialization, then restore P.
Payload-only lower floors and packetized conditional ceilings are derived
from these formulas for every row before executing its backend.

Predicted shapes: at fixed F and placement, quadrupling payload quadruples
the payload service term; halving capacity doubles it, leaving propagation
unchanged. Equal-sender ideal makespan grows linearly with F*B, plus bounded
packetization. Larger payload and fan-in should increase physical phase
makespan; halved rate should not reduce it. Physical per-flow order and
ratios need not be monotone because arbitration can change service order.
A below-one small-flow ratio with a large-flow ratio at least one, and a
pair makespan above its floor, is direct scheduling evidence. Absence of
that event in the two-flow matrix is not proof against H-sched.

## Deciding floors and evidence classes

For each phase, sort completions by (completion time, source, destination,
tag, flow ID). For every k, let C_k be the sum of payload bytes of the k
completed flows. Check elapsed_k >= C_k*ps_per_byte+P_min, where elapsed
is measured from the first phase start and P_min is the minimum path
propagation among its members. This conservative common propagation remains
valid for mixed local and remote senders. Also retain each flow's stronger
individual path floor. The receiver has one link, so headers and partial
service to other flows can only strengthen these payload lower bounds.

Floor (ii): physical first-start-to-last-completion phase makespan must be
at least the identical packetized ideal phase makespan. Also check the
independent total-byte-plus-propagation floor, keeping the fluid analytical
reference distinct from the measured packet baseline. Every floor is fatal;
a below-one aligned per-flow ratio is a scored observation, never a guard
when receiver service is shared. Identity, multiplicity, payload, aligned
zero starts, consistent timestamps and physical quiescence are fatal too.
No fatal in this controlled study is survivable. A failed fatal voids the
behavioral score, not the retained exact rows and failure diagnostics.

If all floors hold, isolated controls have no sub-one ratios, and shared
receiver cells do, the tested H-sched prediction is supported and the
per-flow lower-bound convention is refuted for shared bottlenecks. These
finite completion checks cannot prove absence of every hidden credit bug.
If either deciding floor fails, report the exact cell, prefix/phase rows
and command as H-credit evidence under the task's decision rule; do not
modify third_party/htsim. Keep BACK-68 P0 with the exact residual scope.
If no distinguishing sub-one ratio appears, report inconclusive evidence.

## Conditional follow-through

Only after scheduling evidence, add reusable phase normalization and
completion-prefix byte floors to simllm/backends/fct.py, without changing
normalized_fct behavior or existing tests. The identical phase's GOAL and
flow population must match; first-start-to-last-completion uses each run's
own starts. Aligned per-flow normalization is a lower bound only without
another aligned flow sharing its bottleneck, otherwise diagnostic.

Before the width-tail rerun, make a separate expectations-only amendment:
replace the shared per-flow fatal with the two floors and declare only the
width-64 physical all-to-all HTSIM-40 control-loss exit survivable. Its
failed cells stay void and unscored, while the rest of the matrix remains
interpretable. Change no workload, point prediction or behavioral band.
Rerun all 64 attempts fresh, preserving original ideal measurements and
standalone oracle rows exactly, including the unchanged TRAF-89 residual.

Small tracked JSON is the numerical authority; RESULTS.md and the plain
matplotlib PNG/PDF are projections. Plot per-flow minimum and phase ratios
against fan-in, and completion-prefix elapsed/floor ratios against k for
the most discriminating shared cell. Axes identify counts and dimensionless
ratios. Offline tests use synthetic flows, require no native simulator,
network, GPU or personal paths. Emit LF bytes, accept raw or LF-normalized
tracked-input digests, and set text eol=lf attributes. Required final gates:
ruff check ., full pytest -q, and check_docs_format.py for the module edit.
