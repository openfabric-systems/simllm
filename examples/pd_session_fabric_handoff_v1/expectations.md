# Fabric-rendered KV handoff expectations

This is the expectations-only freeze for TRAF-62. It predates the packet
handoff implementation, its harness, generated GOAL and backend artifacts, and
every scored run. The accepted constant and off arms remain unchanged.

## Question

Can the disaggregated session render its key-value cache handoff through the
existing flow, Group Operation Assembly Language (GOAL) and htsim machinery,
charge a declared PCIe submission term before packet service, and expose the
last required arrival as the sole handoff completion that moves TTFT but not
decode TPOT?

## Frozen runtime and source identity

The packet arm uses the htsim gitlink and executable hashes frozen in the JSON
registry, the `rnic-nn` packetized null-network profile and the repository's
existing GOAL converter. The executable and converter are explicit local
inputs. No download is permitted. The Python run uses the repository's Python
3.10 environment.

The JSON registry freezes the pre-change core handoff, adapter, placement,
GOAL, transfer-pattern and backend-reader source hashes. It also freezes every
accepted file under `examples/pd_session_v1`. A disagreement in any input or
baseline digest is fatal.

## Transfer geometry and endpoints

The same Granite authority as the constant arm gives 49,152 KV bytes per
original prompt token. The 8-token cell carries 393,216 aggregate bytes and
the 16-token cell carries 786,432.

One prefill tensor-parallel engine occupies GOAL ranks 0 through 7. One decode
engine occupies ranks 8 through 15. Prefill local rank `i` sends the complete
local KV shard to decode local rank `i`, so the pairs are `(0, 8)` through
`(7, 15)`. This is the repository's direct pairwise transfer convention. Each
request therefore renders eight positive messages, one per pair, without a
second traffic authority.

The 8-token cell sends 49,152 bytes per pair. The 16-token cell sends 98,304
bytes per pair. The eight chunks must sum to the aggregate geometry bytes.
Every declared source and destination appears exactly once, no other endpoint
appears, and every backend completion row joins to one rendered message by
source, destination, tag and payload.

## Timing contract

The declared PCIe submission term is 20,000,000 ps per request. It is charged
once before any of that request's eight sends may start. Parallel local-rank
submissions are a projection of the one request-level term and do not multiply
TTFT by eight.

The handoff event uses the queue-visit vocabulary directly:

- `submitted_at` is the producer completion;
- `eligible_at` is producer completion plus the PCIe submission term;
- `started_at` is the first packet-service start and equals `eligible_at` in
  this no-queue study;
- `finished_at` is the last required receive arrival;
- `completed_at` equals `finished_at` because there is no visibility tail.

The packet service term is `finished_at - started_at`. The complete handoff
duration used by TTFT is `completed_at - submitted_at`, so the PCIe term cannot
disappear from the decomposition or be folded into a packet timestamp.

## Frozen sweep

The packet arm crosses prompt lengths 8 and 16 with endpoint link rates 200
and 400 Gbit/s, producing four exact-oracle rows. The constant comparator uses
the accepted 100,000,000 ps handoff. The off arm remains a zero-duration
identity control.

Only the explicit packet arm may write GOAL text, GOAL binary, completion CSV
or backend manifest artifacts. Constant and off construction and application
must write none. Their accepted session files, timestamps and compact result
stay byte-identical.

## Exact metric movement

For each context and bandwidth, define `P` as the packet handoff's complete
duration, including PCIe submission. Let `C` be the 100,000,000 ps constant.
The exact signed relation is:

```text
packet TTFT - constant TTFT = P - C
```

The two sides must agree to zero picoseconds. The complete decode token sequence
and decode TPOT are identical between arms. A packet arm that changes only an
isolated flow-completion metric without reaching the session TTFT fails this
study.

The accepted constant controls are 273,376,000 ps TTFT and 77,952,000 ps TPOT
at 8 tokens, and 292,912,000 ps TTFT and 77,976,000 ps TPOT at 16 tokens.

## Physical bounds before modeled values

Eight independent tensor-parallel ranks use eight endpoint links. The fastest
possible handoff serializes one eighth of the aggregate bytes on each link.
At 400 Gbit/s the packet-service floors are 983,040 ps for 8 tokens and
1,966,080 ps for 16. At 200 Gbit/s they are 1,966,080 and 3,932,160 ps.

The deliberately broad packet-service ceiling serializes the aggregate bytes
on one link and adds 50,000,000 ps fixed overhead. The four ceilings are
57,864,320, 65,728,640, 65,728,640 and 81,457,280 ps as recorded in the JSON
matrix. The declared 20,000,000 ps PCIe term is then added outside packet
service. These bounds distinguish an eight-rail transfer from an impossible
faster-than-wire result while tolerating packet and propagation overhead. They
do not calibrate a physical fabric.

Halving bandwidth must not reduce packet service. Doubling context must double
bytes exactly and must not reduce service. Strict twofold timing is not frozen,
because packet framing and fixed propagation remain visible.

## Fatal guards and evidence accounting

A source, binary, gitlink, baseline, geometry, endpoint, chunk, ordering,
completion, quiescence, arm-isolation, metric or physical-bound disagreement
voids the run. Conservation and authority checks remain fatal-unscored. The
four exact metric rows and the three behavioral families remain separate
classes and are never summed into one headline fraction.

## PLACE-5 dependency and scope

The frozen live cell uses the existing one-plus-one role-aware endpoint
projection. TRAF-62's registry also declares PLACE-5 as a dependency for the
complete fixed target topology. The result may close TRAF-62 only if that
dependency is literal at reporting time. If the packet mechanism and all
one-plus-one acceptance rows pass while PLACE-5 remains open, TRAF-62 stays
open and TRAF-64 records only the remaining topology-qualified closure work.

A valid result does not calibrate the KV transfer, validate the 448-rank
physical target, add new backend transport behavior or change the constant
arm. It demonstrates the packet mechanism and its live TTFT effect on the
bounded session configuration.
