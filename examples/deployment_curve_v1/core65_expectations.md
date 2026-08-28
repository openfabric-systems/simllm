# CORE-65 physical-binding expectations

Status: **EXPECTATIONS ONLY, PROTOCOL VOID ALREADY RECORDED**. These physical
identity rules and signed directions are frozen before any retained service
arithmetic.

## Candidate 1: layer-type composition

The captured four-layer stack is expected to be the first four model layers
if the retained `num_hidden_layers=4` override is literal. The model contract
places dense MLPs in layers 0 through 2 and MoE in every layer from layer 3,
so that hypothesis predicts a three-dense plus one-MoE capture, not a
homogeneous four-layer basis. The record must decide this; the hypothesis is
not treated as evidence.

Signed directions are frozen per component before timing arithmetic:

- common per-layer work stays at the same 61-over-4 scale;
- dense-only work decreases because three captured dense layers already equal
  the three production dense layers and must not be multiplied by 61 over 4;
- MoE-only non-routed work increases because one captured MoE layer represents
  58 production MoE layers, not 61 over 4;
- routed-expert work combines the MoE-layer coefficient with only the physical
  compute or byte scale justified for its kernel;
- the net direction is structurally indeterminate before the opposing service
  components are inventoried.

An all-dense four-layer basis extrapolated over the real MoE region would
overprice dense service and move the corrected step down. An all-MoE basis
would omit the three dense layers and underrepresent the 58-layer MoE count,
so its separate missing terms move the corrected step up. Neither homogeneous
case is assumed.

## Candidate 2: expert population

The captured TP1 model is expected to have 256 resident logical routed experts
per MoE layer. A real EP72 rank has four physical expert slots. Directions and
scales are frozen separately:

- assignment-tracking compute retains the exact inherited `1/9` scale and
  moves no further unless dispatch evidence refutes that tracking variable;
- expert-count or resident-weight-byte work uses `4/256 = 1/64` relative to a
  256-expert captured shard and therefore moves service down;
- per-active-unique-expert weight reads require the actual routing identities
  and cannot inherit either scale without evidence;
- common attention, router, shared-expert, normalization, and output work is
  never expert-population scaled.

## Candidate 3: weight-read volume

The deployment projection's standard-decode EP72 physical static bytes per
rank will be compared with a capture-side per-step read volume, not merely a
resident allocation. If captured reads are higher, the signed step movement
is down; if equal, it is null; if lower, it is up. Resident bytes, assignment
count, active unique experts, and measured HBM bytes are distinct quantities.
No conversion from one to another is allowed without a physical identity or
counter record.

## Candidate 4: other and missing kernels

Every retained kernel must receive exactly one disposition: EP72 counterpart,
capture-only, EP72-required-but-absent, or undecidable. Removing a capture-only
kernel moves the step down. Adding an EP72-required missing operation moves it
up. Counterpart kernels remain unchanged unless Candidates 1 through 3 supply
an explicit component scale. CUDA graph bookkeeping and fixed service remain
separate from repeatable layer work.

## Arithmetic and publication locks

The inherited anchor is 22,282 tokens/s/node and the inherited CORE-64 result
is the sole standard-decode calibration basis. No constant may be fitted and
no fifth-run or MTP value may enter a computation or comparison. An
undecidable or null result must publish literally and register the precise
EP72-shaped hardware capture that resolves it.

The minimum preservation class is 154 files: CORE-64's inherited 134-file
class plus the 20 entries in `core65_prior_git_blobs.txt`. No protected value
may be decoded during hash verification.
