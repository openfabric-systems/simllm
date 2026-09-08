# Live peer packet runtime expectations

This is an expectations-only freeze for BACK-48, COMP-40 and TRAF-45. It
contains no implementation, measured values or generated outcomes. The study
uses declared physical resources and synthetic request routing. No GPU runs,
weights or hardware-calibrated validity are implied.

## Authority and supported boundary

The original step record lowers to one checked execution graph. Its locality
classifier keeps the accepted analytic endpoint ledger and byte partition. An
explicit packet selection binds those same local segments to declared GPU ports
and physical links in the fabric manifest. One retained causal calendar per
physical domain owns packet admission, grants, reservations, transmission,
arrival, ingress and visibility across graph phases and later steps.

The graph coordinator owns external eligibility and phase order. Packet
attempts use the port-neutral version 2 application binary interface (ABI)
vocabulary, including the existing wire aliases. GPU port reports project the
packet authority without independently advancing a transfer. Unsupported
capabilities reject before events, counters, clocks or ownership mutate.
The native wire grammar and existing version 1 and version 2 serialized
artifacts stay exact. Extent identity, attempt identity, parent identity and
byte offsets stay distinct. A packet terminal does not consume its parent.

Only extent and operation completion events cross into the execution result.
An extent becomes visible after every required payload packet becomes visible.
The event-derived execution result must agree exactly with the returned step
result before request metrics consume it. A selected local packet phase never
also charges its analytic scalar or an aggregate calibrated collective floor.
An absent packet selection runs the unchanged analytic branch.

The first supported graph path is the existing serial phase projection in the
ordinary step sink. The retained physical remote-session/locality composition
stays explicitly unavailable under BACK-72. Parallel speculative preparation
cannot share this mutable packet session. All overlapping packet visits remain
a complete resource ledger; their sum is never called a critical-path wait or
added wholesale to token latency. TRAF-54 owns richer critical-path reporting.

## Physical route model

A direct route names the physical source and destination ports and a directed
link. Opposite directions have separate resources when declared full duplex.
A switched route names a GPU input attachment, finite shared switch input
storage, matched switch input/output resources, an output attachment and a
finite destination buffer. Every destination sharing a source attachment uses
one input-link calendar. Every source sharing a destination attachment uses
one output-link calendar. Virtual output queues do not multiply input capacity.

Source feed and its link occupy max(feed service, link service). The switch
input/output and output attachment occupy max(crossbar service, output-link
service), then the second hop propagates. This declares store-and-forward
packet admission with overlapped output feeding, without a third full packet
serialization stage. Each downstream capacity is reserved before transmission.
Physical link release and downstream arrival are separated by propagation. A
link acknowledgement cannot precede the forward and reverse propagation path.
No value here identifies a confidential NVSwitch buffer or arbitration field.

## Frozen live matrix and independent arithmetic

The exact parameters live in [expectations.json](expectations.json). For
K in {1,3}, rank0 is the request home and rank i in1..K owns expert i-1 at
layer0. There are K routed experts and top-k equals K. Each of the three tokens
selects every expert once. The real lowering emits a 1024-byte dispatch to
each donor and a 1024-byte combine from each donor, with increasing donor-rank
extent order. Tensor parallel width is one. All ranks occupy one declared node.
There is one prefill and two decode steps, each sampling one output. Step0
releases at0; each later step releases at the preceding returned completion.
A deterministic provider contributes exactly37000ps per step; host launch is
the existing ideal off mode. This toy compute constant is a control, not a
prediction for a real model.

Each extent is four packets of256 payload and272 wire bytes. Define N=4,
L=10880ps, P=1000ps and R=10880 or21760ps. Buffers and credits are finite
but nonblocking for the live matrix. The frozen source order transmits one
dispatch extent before its next extent. The exact phase expectations are:

| Route | Dispatch | Converging combine |
|---|---|---|
| Direct | (K-1) N L + L + P + N R | L + P + K N R |
| Switched | (K-1) N L + 2 L + 2 P + N R | 2 L + 2 P + K N R |

The combined phase sums are frozen as integer literals in the JSON. A physical
floor is K N272/rx_rate plus the first packet's transport time at the common
receiver. A conservative ceiling serializes every packet through every declared
stage, including propagation and bounded returns. Both are stated and computed
before reading any result. Passing an internally exact oracle outside those
bounds voids the run. The physical checks use three angles: link bytes and
propagation, finite input/destination capacity and ingress service, and the
end-to-end request delta with identical compute and launch work.

Halving receiver rate adds (K+1) N10880ps to total communication. Changing K
from1 to3 adds2NL+2NR. A switched route adds2(L+P) to the two-phase sum.
Analytic communication is6000ps at K=1 and14000ps at K=3. Therefore packet
minus analytic time to first token and every subsequent time per output token
is exactly packet communication minus that analytic sum. Job completion for
three consecutive steps is three times the per-step latency. No calibration
coefficient may be fitted to these relations.

## Retention, attachment and negative controls

A one-packet direct transfer consumes one credit and the whole272-byte receive
buffer. Submit an identical next transfer at its logical visibility boundary.
For return delay D in {0,100000}ps, the next transmission starts at that
boundary plus D; its completion adds D+L+P+R. A200000ps acknowledgement
tail must remain pending without delaying consumer visibility or resetting the
calendar. Drain only after the final logical step and retain its separate time.

For two queued fan-out extents, source feed is100GB/s, link rates are12.5 or
25GB/s, and total physical input capacity is272 or544bytes. The source's two
destinations must share exactly one attachment calendar and the declared total
buffer budget. Aggregate input serialization cannot beat total wire bytes over
that link rate. Reducing capacity cannot admit more simultaneously owned bytes.
Reversing extent submission order under identity swaps which independent donor
gets early service while preserving geometry, bytes and the symmetric makespan.
A class-label permutation preserves every event and timestamp under identity.
These are fatal conformance controls, not extra scored timing families.

The common consumer must reject malformed attempts, duplicate or missing
terminals, changing byte geometry, invalid timestamp order, unsupported event
kinds, reused session tokens and a parent completed before its packets. Rejected
capability negotiation and malformed phase preflight preserve the prior state
exactly. Retain raw observations before validation. Any fatal finding voids
the entire run and sets its behavioral score to null; never count guards as
passed behavior or score the unaffected remainder.

## Evidence classes and exact bypass

Report configurations, exact phase/step/request oracles, four timing relation
families with their parameterized instances, structural guards, compatibility
controls and native executables separately. Hardware coverage is zero. Record
the final expectations-only commit preceding implementation and first execution,
the executed source identity, frozen-input digests and immutable raw outputs.
Preserve failed runs and their chronology.

At the recorded baseline commit, reproduce the accepted locality and mixed
attribution examples and compare all published timestamps, bytes, component
attribution and canonical serialized outputs with packet selection absent.
Likewise preserve accepted native wire ABI1/ABI2 artifact bytes. Path-dependent
provenance may be reported alongside a comparison, but excluding any field from
byte comparison must be named explicitly before execution; the permitted
exclusions are only runtime output path, source revision, source digest and
platform/build identity, never a modeled value or semantic identifier.

Completion of this study establishes a supported declared packet path into
request metrics. It does not qualify any real NVSwitch deployment, compute
kernel timing, DeepSeek-V3 deployment frontier or full Kimi K3 structure.
TRAF-65/73/86 retain hardware identification, BACK-72 retains mixed physical
sessions, and COMP-54 retains full K3 architecture.
