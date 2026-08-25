# Offline model extraction v1 expectations

## Freeze scope and chronology

This expectations-only freeze follows the isolated path-scanner repair at
`7196435081841c77c6271f41c16923f070d32023` and precedes the model inventory
schema, extraction drivers, command, generated inventories and first study
run. Immediately before these freeze files were authored, `git status
--porcelain=v1` produced no rows. The worktree was clean.

The frozen input is the unchanged `transformer-dag-v1` suite. Its exact bytes
hash to `5ec3296dd34ef42c65bc3677916aedc284585ec5b6b11ea2ecd5873a3e5d2266`.
No observed hash, inventory identifier, measured value or outcome-dependent
bar appears in this freeze.

## Extraction boundary

One inventory is emitted per exact `(framework, model)` pair. The framework
driver obtains model geometry through the framework's CPU-safe configuration
surface, then joins it to the suite's authored shape grid at the StepRecord
boundary. The vLLM path additionally requires the existing exact
`SIMLLM_VLLM_WORKER_MODE=skeleton` gate. The SGLang path uses its CPU engine
configuration boundary. Both paths serialize and reload the complete ordered
StepRecord set, project `step_kernels()` in its existing family order, and
lower each record through the existing execution-graph path.

This join uses each existing interface for one purpose:

- the framework config supplies model geometry and proves support for the
  pinned checkpoint;
- the authored suite supplies phase and shape cases;
- StepRecord supplies the framework-neutral scheduled work;
- `step_kernels()` supplies the ordered aggregate family work and typed shape
  coordinates;
- execution-graph lowering supplies the graph instance and normalized
  template identities.

The driver rejects before writing when any input or projection is incomplete.
It does not treat the CPU engine as physical capture. Code-object hashes and
observed launch rows remain explicit `absent-by-design` values with null
payloads. VLLM-12, SGL-10 and COMP-6 own their later physical joins.

## Frozen identities

The checkpoint is
`ibm-granite/granite-3.0-1b-a400m-instruct` at revision
`ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445`, with the exact config hash,
weight hash, weight byte count and 24-layer geometry in
`expectations.json`. The framework rows are vLLM 0.26.0 at source commit
`568afb3a13806beb53bb2e6bd518269357b237c0` and SGLang
`0.0.0.dev1+g8f2a3ad6d` at source commit
`8f2a3ad6d7d68c58ae65b61a75bb2115449addca` and tree
`5be26db1f559064c0f9e724e78c1a8f619754867`.

## Parameter sweep

The existing 15 graph cells provide three independently varied parameters:

| Family | Varied parameter | Frozen values | Fixed input |
|---|---|---|---|
| compute prefill | prompt tokens per request | 32, 128, 192, 256, 512 | 4 requests |
| memory decode | context tokens | 128, 512, 1024, 2048, 8192 | batch 4 |
| MoE communication decode | batch | 1, 4, 8, 16, 64 | context 2048 |

Cases retain exact suite array order. Reordering, omission or duplication is
fatal because the inventory defines column denominators.

## Expected relations

R1, exact family projection: every case has exactly the ordered families
`attn_gemm`, `attn_score`, `mlp_gemm`, `lm_head`, `kv_read`. The stored shape
vector, aggregate integer FLOPs and aggregate integer HBM bytes for each entry
equal the corresponding `step_kernels()` projection, and the family sums equal
the fused step exactly. Changing any family, shape or work value can fail this
relation.

R2, layer launch scaling: `attn_gemm`, `attn_score`, `mlp_gemm` and `kv_read`
have one logical projected invocation per layer. `lm_head` has one per sampled
step. At layer count L the total is exactly `4L + 1`; at the frozen 24 layers
it is 97. A wrong layer count or family multiplicity can fail this relation.
These are structural column denominators, not observed physical launch rows.

R3, shape-axis sensitivity: prefill new-token and KV-token axes equal the
authored total prompt tokens. Decode new-token and sampled axes equal batch,
while KV tokens equal batch times context. Each of the three parameter sweeps
must therefore change its declared vector exactly. A constant, swapped or
partially propagated axis can fail this relation.

R4, template equivalence classes: the ten single-rank compute-prefill and
memory-decode cases share one normalized template because instance shape and
request identity are excluded from the frozen template schema. The five
four-rank MoE decode cases share a second template, and the two template hashes
differ because rank and collective structure are retained. A missing
collective, retained shape, or lost rank can fail this relation.

R5, byte determinism: extracting the same framework and suite twice produces
identical canonical bytes. The vLLM and SGLang records differ in framework
identity even when every structural denominator agrees.

## Fatal guards and negative controls

Any suite-byte, checkpoint-identity, framework-identity or totality mismatch
voids the run. Nonintegral or negative projected work, a physical field that
claims a known value, or ordinary `simllm` import loading either framework also
voids it. Fatal means the behavioral score is not interpreted.

Unsupported models, incomplete case sets, unknown families and an unflagged
vLLM skeleton must reject without an output object. These are structural
negative controls and do not add to the behavioral pass denominator.

## Bypass and task effect

The hand-authored transformer-dag-v1 suite is the explicit bypass. Its bytes
must retain the frozen SHA-256 above. This slice does not close COMP-54 unless
both framework records and their published column denominators are complete.
It never closes COMP-6, VLLM-12 or SGL-10 because it performs no physical
capture.
