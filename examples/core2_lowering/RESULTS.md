# CORE-2 lowering and WQE bookkeeping results

Date: 2026-08-05

The study passed every pre-registered check in
[expectations.md](expectations.md). The tested HTSIM backend commit is
`d778326` (`Add WQE queue bookkeeping`). Raw GOAL, graph JSON, completion CSV
and summary artifacts are under
`/data3/yifeng/simllm-dev/core2-lowering-runs`; they are not Git content.

## Reproduction

```bash
SIMLLM_HTSIM_RNIC=/path/to/htsim_rnic \
uv run --isolated --no-project --offline \
python -m examples.core2_lowering.run_core2 \
  --out /data3/yifeng/simllm-dev/core2-lowering-runs
```

The runner lowers each step, round-trips only the execution graph through
JSON, renders GOAL from that parsed graph, then compares the result with both
the legacy step path and the frozen closed form.

## Exact results

| case | width | rate Gbit/s | legacy JCT ps | graph JCT ps | frozen JCT ps | flows/WQEs | all residuals ps |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 2 | 400 | 82,003,040 | 82,003,040 | 82,003,040 | 16 | 0 |
| dense | 2 | 200 | 123,946,080 | 123,946,080 | 123,946,080 | 16 | 0 |
| dense | 4 | 400 | 134,974,560 | 134,974,560 | 134,974,560 | 96 | 0 |
| dense | 4 | 200 | 197,889,120 | 197,889,120 | 197,889,120 | 96 | 0 |
| MoE sentinel | 4 | 400 | 25,811,524 | 25,811,524 | 25,811,524 | 48 | 0 |

The compared row key includes the flow ID and every flow, queue, WQE and
transport field together. Consequently, equality proves the exact
flow-to-WQE mapping, not two independently sorted multisets. Every dense and
MoE row had:

- exact legacy, graph and closed-form JCT equality;
- exact combined completion-ledger equality;
- unique flow and WQE IDs with a one-to-one mapping;
- SQ post sequence equal to SQ dispatch sequence;
- CQ post sequence equal to immediate CQ consume sequence;
- transport kind `none` and transport object ID 0 for the fluid null profile;
- verified physical quiescence and zero final queue depth.

After subtracting 24,060,000 ps of represented compute and the fixed
propagation terms, serialization-only time was:

| width | Q at 400 Gbit/s ps | Q at 200 Gbit/s ps | exact relation |
|---:|---:|---:|---|
| 2 | 41,943,040 | 83,886,080 | `Q(200) = 2 * Q(400)` |
| 4 | 62,914,560 | 125,829,120 | `Q(200) = 2 * Q(400)` |

At either rate, width-4 serialization was exactly 1.5 times width-2
serialization. Total JCT did not halve with doubled bandwidth because compute
and propagation remained fixed, as registered.

## WQE seam validation

The focused HTSIM matrix crossed WQE count `{2, 4}` with transport kind
`{dcqcn-qp, rnic-cn-link-pair}`. Each cell had exactly N SQ posts and FIFO
dispatches, N CQ posts and immediate consumes, zero RQ activity, zero final
queue depth, same-direction transport-identity reuse, and a distinct reverse
direction identity.

Two live driver smokes closed the physical-profile integration seam with one
4,096-byte WQE:

| driver | transport field | SQ/CQ sequences | RQ | physical quiescence |
|---|---|---|---|---|
| `htsim_rnic` `rnic-cn` | `rnic-cn-link-pair`, nonzero object ID | all 1 | identity only, zero activity | verified |
| `htsim_dcqcn_atlahs` | `dcqcn-qp`, nonzero object ID | all 1 | identity only, zero activity | verified |

The manual smokes used a 64-rank GOAL with one 4,096-byte rank-0 to rank-1
send/receive at tag 77 and a one-nanosecond `calc` on every other rank. It was
generated with:

```python
from pathlib import Path
from simllm.goal import GoalTrace, to_binary

trace = GoalTrace(64)
trace.rank(0).send(4096, 1, 77)
trace.rank(1).recv(4096, 0, 77)
for rank in range(2, 64):
    trace.rank(rank).calc(1)
to_binary(trace.write(Path("physical-smoke.goal")), Path("physical-smoke.bin"))
```

Both drivers used the committed 64-node 400G Clos topology and default model
flags:

```bash
HTSIM_BUILD=/path/to/htsim-build/datacenter
HTSIM_SRC=/path/to/htsim
TOPOLOGY="$HTSIM_SRC/experiments/rnic_multibaseline/topologies/clos_64_400g.topo"

"$HTSIM_BUILD/htsim_rnic" \
  -goal physical-smoke.bin -linkspeed_bps 400000000000 \
  -rnic_profile rnic-cn -topo "$TOPOLOGY" \
  -completion_csv rnic-cn.csv

"$HTSIM_BUILD/htsim_dcqcn_atlahs" \
  -goal physical-smoke.bin -topology "$TOPOLOGY" -link_bps 400000000000 \
  -completion_csv dcqcn.csv
```

The complete HTSIM CTest suite passed 344 of 344 tests, including the frozen
WQE count by transport-kind matrix. The two physical-driver commands above
were separate manual smokes, not CTests. The legacy backend `commit_check.sh`
is not a valid gate on current `main`: its expected baseline files are absent
and `validate.py` divides by zero before comparison. Repair is tracked as
HTSIM-8 in [backends.md](../../docs/modules/backends.md).

After the final contract review, the SimLLM repository passed Ruff 0.16.1 and
the complete Python suite with 188 passed and 3 environment-dependent tests
skipped. The frozen five-row matrix above was then rerun against the revised
per-edge dependency wire form with every value unchanged.

## Scope boundary

CORE-2 validates the core ledger contract and HTSIM completion-CSV
bookkeeping separately. A concrete `DeviceRuntime` still has to correlate
graph operation IDs and rendered flow tags with NCCL commands and backend WQE
rows; that wiring is CORE-4. The backend SQ is currently a timing-neutral
identity and lifecycle queue. Fixed WQE initiation latency, persistent
per-QP DCQCN rate state and behavioral pipelining remain HTSIM-5. The
`rnic-cn` established-pair fast path and one-RTT SQ lookahead remain HTSIM-6.
