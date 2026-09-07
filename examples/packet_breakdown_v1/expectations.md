# Packet step breakdown: frozen expectations

BACK-69 adds an optional read-only projection of executed packet artifacts.
This registration precedes implementation and all runs. RESULTS.md must cite
this expectations-only commit. The supplied wave contract applies; the root
AGENTS.md was absent when work began.

## Cells and independent axes

Run the collective_width_tail_v1 ideal step cells for pattern {ring,
all-to-all}, width {8, 64}, and link rate {400, 200} Gbit/s on rnic-nn.
Reuse its declared 64-rank placement, rail-major ring order, single-engine
expert dispatch/combine, dimensions and fixed 100,000,000 ps compute input.
There are eight width cells. Run the breakdown study's eight-step request
(2048-token prefill, seven decodes), TP {2, 8}, 400G, profiles
{rnic-nn-fluid, rnic-nn}, with its original 32-layer dimensions. Replay both
complete m4 fixtures at TP=8, 400G, rnic-nn-fluid with m4 dimensions. For
request reduction, release replayed records sequentially at the preceding
simulated completion, retaining each record's scheduler decision.

Every cell compares omitted selection, explicit disabled selection, enabled
selection and the persistent sink's prepared batches. The off paths must
retain the exact StepResult, GOAL, binary GOAL, completion CSV, locality and
other accepted outcome projections. Preparation must publish no outcome
before consumption. Small retained fixture evidence must permit deterministic
replay without native tools, network, GPU or machine-local paths.

## Exact relations and ownership

At 0 ps tolerance, the five CriticalPathBreakdown segments sum to each
completed step's latency: launch queue, device queue, service, completion
delivery, external dependency. Selected-path queue time and additive visit
waits remain separately named reductions. Each artifact has a queue visit
with submitted <= eligible <= started <= finished <= completed. Eligibility
is the ordered predecessor boundary. Fabric start/finish come from first
start/last completion rows, offset by the artifact's authoritative boundary;
analytic artifacts use their declared interval. Visibility extends to the
ordered artifact boundary, including backend completion quantization.

Registration is a launch visit. The exposed host floor already included in
compute is partitioned into launch time, never added twice; its unexposed
floor remains an input bound rather than invented elapsed service. Semantic
base and aggregate floor retain their existing names. Compute between
communication artifacts is external dependency in the selected communication
path, while its explicit GPU visit preserves kernel attribution. With ideal
host, external dependency equals the compute service charged by the sink.
With exposed host, their sum equals that charged compute service.

Reproject the existing ttft_attribution, decode_attribution and medium shares
from the conserved segments/visits at 0 ps tolerance, including pending
unsampled intervals and scheduler gaps. Preserve co-critical ownership and
masked work separately. No summation of concurrent media into elapsed time.
CompletionEvent rows and a strict versioned JSON projection accompany the
enabled outcome. Absent optional fields read as disabled and write no new
keys. Invalid schema, extra fields, booleans as integers, negative or
nonmonotonic timestamps, and nonconservation must fail closed.

## Physical sanity before reading measurements

400G costs 20 ps/payload byte; 200G costs 40. For every fabric service
interval, bytes on the critical endpoint divided by rate plus propagation is
a lower bound. A ring has 2(W-1) dependent rounds of S/W bytes, so its floor
is 2(W-1)*(S/W*8000/rate_Gbps + 2,000,000) ps. The ideal packet ring has
4,096-byte payloads plus 64-byte headers, giving the exact point
2(W-1)*((S/(4096W)+1)*4160*8000/rate_Gbps + 2,000,000).
An expert dispatch or combine has D=7W/8 remote peers and 65,536 bytes per
peer, with floor D*65,536*8000/rate_Gbps + 2,000,000. The conservative
ceiling serializes every packet globally with one extra endpoint slot:
(W*D*16+1)*4160*8000/rate_Gbps + 2,000,000. Twice these communication bounds
plus the fixed compute input encloses each width step. Local service has
floor ceil(critical endpoint bytes/rate), propagation zero, and ceiling equal
to the declared analytic serializer. Aggregate fitted service, when tested,
is a model input and not a first-principles wire claim.

For breakdown and m4 ring artifacts use their token-derived activation
payload S = new_tokens*hidden_size*dtype_bytes and the same per-round bound.
For partial packets round each payload up only for the header-count term;
an extra full packet slot per round is a conservative ideal ceiling. Charged
compute has equal floor and ceiling at its declared quantized duration.
Every time segment lies in [0, step latency]; every share lies in [0, 1].
Physical queued networks have no finite first-principles upper bound.

Fabric service share should rise strictly from width 8 to 64 for each
pattern and rate. Halving rate doubles serialization and leaves propagation
unchanged, not total latency. For rings subtract 4(W-1)*2,000,000 ps per
two-collective step before checking exact doubling. For expert steps use
two propagation traversals and allow separately reported boundary delivery;
the service serialization relation is exact at 0 ps. Report a failed
directional or exact relation as a miss, without modifying this registration.

## Evidence and artifacts

Fatal guards are void-not-score: missing or invalid completion rows,
nonquiescence, timestamp or byte identity failures, projection mismatch and
conservation failure invalidate that cell's attribution. Such failure is
survivable for the study runner: retain its error and raw artifacts, leave
its shares null, and keep independently valid cells interpretable. The
existing stateful multi-artifact rnic-cn guard remains active and is not a
scored cell or a reason to substitute standalone timings.

Write bulk GOALs, binaries, CSVs and logs under SIMLLM_DATA_ROOT, selectable
with --out. Track only the runner, compact results, deterministic fixture
evidence, this registration, RESULTS.md, and one plain matplotlib PNG/PDF.
The plot uses width as x and the five segment durations in milliseconds as
y, with clearly labeled pattern/rate panels. Inspect the PNG visually.
Digests accept raw or LF-normalized tracked text, generated text is written
as LF bytes, and .gitattributes declares text eol=lf for new text artifacts.
