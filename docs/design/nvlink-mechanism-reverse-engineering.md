# NVLink mechanism reverse-engineered from public documents

This document reconstructs the NVLink mechanism from public protocol and
architecture evidence before the repository changes its three-module model.
Every external claim has evidence class **PUBLIC_DOCUMENT**. No value in this
document is a new hardware measurement, and no public-document claim promotes
a parameter in the current A100 profile by itself.

The central correction is basic but consequential: the documented NVLink
transfer unit is a **128-bit flit, which is 16 bytes**, not a 128-byte flit.
The public Pascal protocol has variable packets from one to eighteen flits.
The current model's `256 + 16 = 272` byte packet is one 17-flit case, not a
universal packet or credit unit. NVIDIA's protocol description and replay
patent agree on the 128-bit unit
([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf),
[PUB-03](https://patents.google.com/patent/US20170111144A1/en)).

## Scope and evidence rules

The reconstruction uses vendor architecture papers and manuals, NVIDIA patent
disclosures, and academic papers that expose their source chain. It does not
use serving-throughput publications. Pages that crossed that boundary were
closed and recorded only in the external scratch ledger.

The three reconciliation verdicts mean:

- **CONFIRMED**: public evidence supports the exact value or structural choice
  at the stated generation and topology.
- **CONTRADICTED**: public evidence rules out the current choice as a complete
  representation of the mechanism. The choice can still be a compatibility
  surrogate until the follow-on lands.
- **UNDOCUMENTED**: the accepted public record does not identify the value or
  product-specific choice. A patent embodiment or an academic simulator input
  is not enough to bind it to A100, H100, or GH200.

Generation scope matters. NVIDIA publishes a detailed packet diagram for the
Pascal implementation, then publishes rates, lane counts, topology, and
features for later generations without republishing the full bit layout.
Accordingly, the Pascal format is the documented protocol-family anchor. A
field is not silently assumed unchanged in NVLink 3 or NVLink 4.

## Reconstructed mechanism

A peer memory operation passes through five distinct mechanisms:

```text
memory operation and ordering domain
    -> transaction packetization into 16-byte flits
    -> per-link flow control, serialization, CRC, acknowledgement, replay
    -> optional NVSwitch route, queue, crossbar grant, and output service
    -> endpoint reorder, memory visibility, and transaction completion
```

Bonding can place packets from one ordered operation on several physical
links. Reliability is link-local, while transaction ordering and final memory
visibility are higher-level responsibilities. A model that combines all of
these into one packet cursor can reproduce an endpoint rate while assigning
the delay to the wrong mechanism.

## Link layer

### Flit and packet structure

The public Pascal implementation transfers 128-bit flits between the physical
layer and the controller. Packets contain one to eighteen flits. NVIDIA gives
a simple read request as a one-flit example and a 256-byte write with an
address extension as an eighteen-flit example
([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)).

The first flit is a 128-bit header. NVIDIA's Hot Chips diagram divides it into
a 25-bit cyclic redundancy check (CRC), an 83-bit transaction-layer header,
and a 20-bit data-link header. An optional address-extension flit carries
information such as upper address bits, an optional command-specific
byte-enable flit carries byte enables, and zero to sixteen flits carry as much
as 256 bytes of payload
([PUB-02](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf)).

| Documented packet part | Size | Role |
|---|---:|---|
| Header flit | 16 bytes | 25-bit CRC, 83-bit transaction header, 20-bit data-link header |
| Address extension | 0 or 16 bytes | Optional address and other relatively stable command information |
| Byte enable | 0 or 16 bytes | Optional command-specific byte enables |
| Data | 0 to 256 bytes | Zero to sixteen 16-byte payload flits |
| Complete packet | 16 to 288 bytes | One to eighteen flits in the published Pascal implementation |

The one-to-eighteen limit and the optional fields mean that `payload + 16`
is valid only for commands that require neither optional flit. It is not a
general wire-size equation. The sources do not publish the rules that select
every optional-field combination, so the follow-on must represent the fields
without inventing when each appears.

The 256-byte maximum data payload is documented for Pascal. The accepted
sources do not expose a later-generation packet diagram that binds the same
maximum and fields to A100 NVLink 3 or H100 NVLink 4. That continuity remains
a generation-specific hypothesis rather than a confirmed A100 constant.

### CRC, acknowledgement, and replay

The data-link layer protects each packet with the 25-bit CRC. The CRC covers
the current header and the previous payload, which lets the receiver learn
the next packet length early. NVIDIA states that this CRC detects as many as
five random bit errors or a burst as long as 25 bits on one lane
([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf),
[PUB-02](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf)).

The transmitter retains unacknowledged packets in a replay buffer. A receiver
with a good CRC returns a positive acknowledgement. On a CRC error it withholds
the acknowledgement and prepares for retransmission; the transmitter times
out and replays from stored data. A packet leaves the replay buffer only after
positive acknowledgement
([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)).
NVIDIA's replay patent further discloses 128-bit flits with implicit sequence
identifiers, replay initiation packets, and storage of recently transmitted
packets that are not yet confirmed
([PUB-03](https://patents.google.com/patent/US20170111144A1/en)).

This is a separate service from receive credits. A credit proves that a
downstream buffer can accept traffic. An acknowledgement proves that a packet
crossed the link without a detected error. One counter or fixed delay must not
stand in for both.

The public record establishes replay-buffer existence but not its A100 or H100
depth, acknowledgement timer, retry latency, or interaction with a depleted
credit pool. Those quantities are undocumented.

### Framing, scrambling, and lane behavior

The physical layer deskews lanes, finds packet boundaries, scrambles and
descrambles the stream for transition density and clock recovery, and handles
polarity inversion and lane reversal. The Pascal link uses an embedded clock,
non-return-to-zero signaling, direct-current coupling, and eight differential
pairs in each direction
([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)).

Framing and scrambling operate as part of the physical stream. The source does
not publish an extra per-packet service delay for them. A rate model should
therefore keep them in the physical link contract unless a later source or
measurement identifies a separate term.

## Flow control

### What a credit protects

NVIDIA assigns link flow control and virtual channels to the transaction
layer, above physical framing and data-link replay
([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)).
That establishes a lossless receiver-capacity gate, but the vendor paper does
not publish the numeric credit pools or bit encoding.

For a GPU switch design, NVIDIA discloses destination-per-virtual-channel
credit counters. A request becomes eligible for switch dispatch only when the
destination has enough buffer capacity. Each virtual channel has its own
destination buffer and independent flow control, and a destination returns a
credit when that buffer frees
([PUB-09](https://patents.google.com/patent/US20230070690A1/en)).
This patent proves a public NVIDIA switch mechanism, not that every NVSwitch
generation uses the exact embodiment.

The causal contract that follows is strong enough for modeling:

```text
receiver advertises capacity for a link and virtual channel
    -> sender consumes the applicable credit before transmission
    -> packet or flit occupies receiver buffering
    -> downstream progress frees that buffering
    -> receiver returns capacity to the same credit domain
```

A fixed sender-side timer is not this contract. Credit reuse is caused by
receiver buffer release, even when an implementation communicates the return
after an additional transport delay.

### Credit granularity and return transport

The word *flit* means flow-control digit, and the public NVLink implementation
uses 16-byte flits as its transfer unit. The accepted primary NVLink sources,
however, do not state that one A100 link credit equals one flit, one packet, or
a fixed byte count. The NVIDIA switch patent describes dispatch in cells or
packets rather than exposing an NVLink 3 wire-credit encoding
([PUB-09](https://patents.google.com/patent/US20230070690A1/en)).

An academic protocol synthesis says that the 83-bit transaction header carries
flow-control credits
([PUB-14](https://people.cs.uchicago.edu/~aachien/lssg/research/10x10/Jiya_Su_MS.pdf)).
That is consistent with an in-band or piggybacked return, but the primary
vendor diagram only labels the field as a transaction header. No accepted
primary source publishes the return field, aggregation rule, standalone
credit packet, or idle-link behavior. Product-specific credit piggybacking is
therefore **UNDOCUMENTED**, not confirmed.

The present evidence boundary is:

| Credit question | Public conclusion |
|---|---|
| Is NVLink flow-controlled? | Confirmed |
| Are there multiple virtual channels? | Confirmed for the published protocol family |
| Are switch credits scoped by destination and virtual channel? | Disclosed NVIDIA switch embodiment |
| When does switch capacity become returnable? | When the destination buffer frees |
| Is A100 credit granularity a 16-byte flit? | Undocumented |
| Is A100 credit granularity one variable packet? | Undocumented |
| Is one credit always 272 bytes? | Contradicted as a universal packet-occupancy representation |
| Are returns piggybacked on reverse traffic? | Consistent with a secondary interpretation, not confirmed |
| How many credits exist per A100 link and virtual channel? | Undocumented |
| What are A100 and H100 receive-buffer depths? | Undocumented |

An academic NVLink simulator assigns a 40 KiB merge table, eight virtual
channels, 256 entries per channel, and round-robin arbitration
([PUB-15](https://chenzhangsjtu.github.io/files/2026-HPCA-CAIS.pdf)).
Those are the paper's modeling inputs. The source does not turn them into
product disclosures, so none of those numbers is imported as hardware truth.

## Virtual channels, traffic classes, and ordering

The Pascal paper explicitly assigns virtual channels to the transaction
layer, but it gives no count or class map
([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)).
The NVIDIA switch-arbitration patent uses multiple virtual channels with
independent destination buffers and credits, but its two-VC drawings are
examples rather than product specifications
([PUB-09](https://patents.google.com/patent/US20230070690A1/en)).

The transaction classes are clearer than their virtual-channel assignment.
NVIDIA describes most writes as posted transactions: a source can issue them
without waiting for an acknowledgement or completion. Reads are non-posted:
the source sends a command and address and expects an acknowledgement or read
data. Posted atomics also exist. A flush after posted writes provides the
ordering point that makes preceding writes visible before dependent work
continues
([PUB-08](https://patents.google.com/patent/US10789194B2/en)).

Thus the current write-request and read-request/read-response direction split
matches the documented transaction direction. What remains undocumented is
whether posted requests, non-posted requests, data responses, acknowledgements,
and control traffic occupy distinct virtual channels on A100 or H100.

NVIDIA separately discloses ordering domains, ordered multi-packet transfers,
target-side reorder buffers, and response reordering for multipath networks.
Packets in one ordering domain become visible in the required sequence even
when physical paths do not preserve arrival order
([PUB-13](https://patents.google.com/patent/US20200374593A1/en)).
The model therefore needs both packet sequence identity and an explicit
reorder or visibility boundary. A single arrival-order FIFO is not a complete
substitute.

## Switch mechanism

### Port and crossbar architecture

The first-generation NVSwitch has eighteen NVLink ports and an internal
18 by 18 fully connected crossbar. Each port carries 25 GB/s in each direction,
and the crossbar is described as nonblocking so every port can communicate at
full NVLink rate. Same-baseboard traffic crosses one switch; traffic between
the two DGX-2 baseboards crosses two
([PUB-07](https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf)).

The first-generation switch protects datapaths, routing, and state with
error-correcting codes. Fabric Manager controls indexed route tables, while
the switch checks final-hop address fidelity and buffer overflow and underflow
([PUB-07](https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf)).
NVIDIA's transaction-tracking patent adds per-port classification, routing,
error/statistics logic, transaction tracking, and packet transformation around
an 18 by 18 crossbar embodiment
([PUB-08](https://patents.google.com/patent/US10789194B2/en)).

The public Fabric Manager guide exposes the consequence in software: separate
request and response hardware route-table entries are programmed, and the
audit utility reconstructs pairwise reachability by decoding those internal
tables
([PUB-12](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/)).

### Queues and arbitration

The product overview does not publish a cycle-level arbitration algorithm or
queue depth. NVIDIA's later switch patent does disclose an input-queued design
with virtual output queues (VOQs). Each input and virtual channel has queues
separated by destination, which prevents one blocked destination at the head
of a flat input FIFO from hiding traffic for a free destination
([PUB-09](https://patents.google.com/patent/US20230070690A1/en)).

That disclosed design applies destination-per-virtual-channel credit masks
before arbitration. A two-dimensional arbiter first lets each destination
choose a source and virtual channel, then lets each source choose among
destinations. The patent discusses round-robin selection, least-recently-used
selection that favors the least recently granted request, and a rolling
round-robin construction that freezes a blocked virtual channel's pointer to
avoid starvation
([PUB-09](https://patents.google.com/patent/US20230070690A1/en)).

This evidence supports the following structural requirements for an NVSwitch
model:

- input-port identity and output-port identity remain explicit;
- virtual channels and destination VOQs remain explicit;
- a packet is eligible only after routing legality and output credit checks;
- the crossbar grants at most one cell from an input and at most one cell to an
  output in a grant interval in the disclosed embodiment;
- arbitration state persists across grants, including fairness or age state;
- the deployed arbitration algorithm and all numeric depths remain
  undocumented unless a product-specific source or measurement identifies
  them.

A single FIFO cursor, even with a `head_of_line_blocking` Boolean, cannot
represent this mechanism. It lacks destination VOQs, per-VC eligibility,
two-sided crossbar matching, and persistent grant state.

### Multicast and reductions

H100 and later DGX/HGX NVSwitch systems support multicast traffic and
reduction operations. Multicast slots are finite resources. The Fabric Manager
guide says H100 partitions receive dedicated sets, while later B200/B300
partitions share a bounded pool
([PUB-12](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/)).

NVIDIA identifies third-generation NVSwitch hardware acceleration for write
broadcast, all-gather, reduce-scatter, and broadcast atomics
([PUB-10](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)).
These engines matter for a scale-up collective model because one multicast or
reduction operation does not necessarily become independent replicated
unicast packets. The exact pipeline rate, slot count, reduction width, and
arbitration interaction are not published in the accepted sources.

### Direct NV4 board versus NVSwitch board

The four-GPU HGX A100 board connects the GPUs directly with NVLink and does
not contain NVSwitch. The eight-GPU HGX A100 board uses NVSwitch
([PUB-05](https://docs.nvidia.com/datacenter/tesla/hgx-software-guide/index.html)).
On the direct board, every GPU can reach every other GPU and the published
peer bandwidth is 200 GB/s bidirectional
([PUB-06](https://developer.nvidia.com/blog/introducing-hgx-a100-most-powerful-accelerated-server-platform-for-ai-hpc)).

For the repository's four-A100 `NV4` topology, switch pass-through is therefore
confirmed. It must add exactly zero switch bytes, time, or reorder state. An
NVSwitch profile is a different topology and must instantiate port, queue,
route, arbitration, and output services rather than turning on a delay inside
the direct-board identity box.

## Addressing, routing, and bonded links

Regular NVLink lets GPUs share an address space and routes requests using GPU
physical addresses. Hopper's NVLink Network adds a distinct network address
space, address translation, isolation between endpoint address spaces, and
explicit connection establishment
([PUB-10](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)).

A peer write therefore follows this physical story on the four-A100 board:

1. The source resolves the peer memory address and creates posted write
   transactions.
2. Packet headers carry the transaction and destination information.
3. The transaction layer selects an eligible physical link in the bonded
   peer connection, subject to that link's flow control.
4. Link serialization, CRC checking, acknowledgement, and possible replay
   complete independently on each used link.
5. The target preserves the operation's ordering domain and makes data visible
   to memory consumers under the required ordering rule.

On an NVSwitch board, route-table lookup and crossbar services sit between
steps 4 and 5. Posted-write groups may use several source ports, which is why
NVIDIA's flush mechanism tracks the relevant paths before returning the flush
response
([PUB-08](https://patents.google.com/patent/US10789194B2/en)).

NVIDIA states that ganged NVLinks spray data across links
([PUB-02](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf)).
The public record does not identify whether A100 chooses links by round robin,
earliest availability, address bits, packet class, or another deterministic
rule. Bonding is confirmed; `earliest_available_packet_striping` is not.

## Physical constants and bounds

Published bandwidths in this table are directional unless the total column
says otherwise. Decimal GB/s is used because the vendor documents use that
convention.

| Product and path | Interface | Differential pairs per direction per link | Pair rate | Links at endpoint | Rate per link per direction | Endpoint rate per direction | Bidirectional endpoint total |
|---|---|---:|---:|---:|---:|---:|---:|
| A100 SXM | NVLink 3 | 4 | 50 Gbit/s | 12 | 25 GB/s | 300 GB/s | 600 GB/s |
| H100 SXM | NVLink 4 | 2 | 100 Gbit/s, implied by pair count and effective link rate | 18 | 25 GB/s | 450 GB/s | 900 GB/s |
| GH200 Hopper GPU to external scale-up fabric | NVLink 4 | 2 | 100 Gbit/s, implied by pair count and effective link rate | 18 | 25 GB/s | 450 GB/s | 900 GB/s |
| GH200 Grace CPU to local Hopper GPU | NVLink-C2C | Not published in accepted sources | Not published | One coherent chip-to-chip interface | Not decomposed into external links | 450 GB/s | 900 GB/s |

The A100 constants come from NVIDIA's Ampere architecture paper
([PUB-04](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)).
The H100 lane structure, per-link rate, and endpoint total come from NVIDIA's
Hopper architecture description. Dividing the documented 25 GB/s effective
link rate by two pairs gives the table's 100 Gbit/s per-pair rate
([PUB-10](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)).
The GH200 C2C directionality and the distinction between C2C and the Hopper
GPU's separate NVLink 4 ports come from NVIDIA's Grace Hopper description
([PUB-11](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/)).

The four-A100 direct board publishes 200 GB/s bidirectional per peer. Dividing
its 100 GB/s one-way peer rate by 25 GB/s per link confirms four links per
directed peer pair
([PUB-06](https://developer.nvidia.com/blog/introducing-hgx-a100-most-powerful-accelerated-server-platform-for-ai-hpc),
[PUB-04](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)).

### First-principles bounds for the current A100 packet envelope

For a large aligned transfer over four A100 peer links, the physical wire
ceiling is:

```text
4 links * 25 GB/s/link = 100 GB/s in one direction
```

If every 256-byte payload uses only a 16-byte header, payload efficiency is
`256 / 272`, so the payload ceiling is 94.117647 GB/s. This is the exact
physical explanation behind the current 17-flit formula, independent of a
measured plateau.

If the command requires the documented address-extension flit, the packet is
288 bytes. Payload efficiency becomes `256 / 288`, the payload ceiling becomes
88.888889 GB/s, serialization time rises by 5.882 percent, and payload
throughput falls by 5.556 percent relative to the 17-flit case. The document
does not say that every aligned bulk write needs that flit, so this is a
conditional bound, not a replacement calibration.

At the endpoint, twelve A100 links provide a 300 GB/s one-way physical ceiling.
An effective source or destination plateau must fall at or below that value.
The public rate does not identify the GPU's internal copy-engine, cache,
memory, or endpoint-queue bottleneck, so a lower effective rate remains a
measured parameter rather than a protocol constant.

## Reconciliation with the current three-module model

The table covers every profile parameter plus the structural choices that
determine packet service in `simllm.backends.htsim_nvlink`. Verdicts compare
the current A100 model to public documents. They do not modify the profile.

| Module or seam | Current choice | Verdict | Documentary reconciliation | Required follow-on |
|---|---|---|---|---|
| Domain | Fixed `TX -> switch -> RX` composition | **CONFIRMED** | The public stack separates transaction/link work, optional switch traversal, and endpoint visibility. | Keep the narrow three-module interface, deepen the services behind it. |
| Transfer | Four endpoints by default for NV4 | **CONFIRMED** | HGX A100 4-GPU is a direct, fully connected four-GPU topology ([PUB-05](https://docs.nvidia.com/datacenter/tesla/hgx-software-guide/index.html)). | Keep topology identity explicit in the profile. |
| TX | `max_payload_bytes = 256` | **UNDOCUMENTED** | 256 bytes is confirmed for Pascal, but no accepted source republishes the A100 NVLink 3 packet layout ([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)). | Make packet format generation-scoped; retain 256 only as a documented-family hypothesis until A100 evidence binds it. |
| TX | `header_bytes = 16` | **UNDOCUMENTED** | The 16-byte Pascal header is documented, but its exact NVLink 3 continuity is not ([PUB-02](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf)). | Represent a generation-scoped header flit instead of an unqualified byte constant. |
| TX | `wire_bytes = payload + header` for every packet | **CONTRADICTED** | Optional address-extension and byte-enable flits make total wire occupancy command-dependent ([PUB-02](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf)). | Carry header, AE, BE, and payload flit counts separately. |
| TX | Write payload is request; read control is request; read payload is response | **CONFIRMED** | NVIDIA documents posted writes and non-posted reads with returning acknowledgement or data ([PUB-08](https://patents.google.com/patent/US10789194B2/en)). | Preserve direction fields and add posted/non-posted/control class. |
| TX | Strict per-extent packet sequence | **CONFIRMED** | Ordered transfers and ordering domains preserve required request order across multipath ([PUB-13](https://patents.google.com/patent/US20200374593A1/en)). | Preserve stable sequence identity, but let a reorder stage, not arrival order alone, enforce visibility. |
| TX | `links_per_peer = 4` | **CONFIRMED** | Four direct links follow from 100 GB/s one-way peer bandwidth and 25 GB/s per link on the four-A100 board ([PUB-04](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf), [PUB-06](https://developer.nvidia.com/blog/introducing-hgx-a100-most-powerful-accelerated-server-platform-for-ai-hpc)). | Keep for the NV4 direct profile only. |
| TX | `per_link_rate = 25,000,000,000 B/s` | **CONFIRMED** | NVLink 3 publishes 25 GB/s in each direction per link ([PUB-04](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)). | Keep the physical serializer rate. |
| TX | Effective endpoint egress `160,795,737,454 B/s` | **UNDOCUMENTED** | Public documents give the 300 GB/s physical ceiling, not this internal effective plateau. | Keep its measured evidence class and do not relabel it PUBLIC_DOCUMENT. |
| TX | One source-wide endpoint egress cursor | **UNDOCUMENTED** | No accepted source exposes the A100 internal egress queue or copy-engine arbitration scope. | Leave queue scope to the registered TRAF-73 controls or a later targeted study. |
| TX | `earliest_available_packet_striping` with low-link tie break | **UNDOCUMENTED** | Data spraying across ganged links is documented; the link selector is not ([PUB-02](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf)). | Keep striping behind a policy and let an ordering/link-counter discriminator choose it. |
| TX | Independent directional cursor for each physical link | **CONFIRMED** | Each NVLink is a bidirectional link built from independent directional signaling and link-local reliability ([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf)). | Preserve per-link service and replay identity. |
| TX | `credits_per_destination = 256`, instantiated per physical link | **UNDOCUMENTED** | Public sources do not give A100 numeric link-credit pools. An academic simulator's 256 depth is not a product disclosure ([PUB-15](https://chenzhangsjtu.github.io/files/2026-HPCA-CAIS.pdf)). | Retain only as a candidate until the TRAF-73 knee identifies an effective window. |
| TX | `credit_unit_bytes = 272` as one maximum-packet slot | **CONTRADICTED** | Packets occupy one to eighteen flits and optional control changes occupancy; 272 bytes is only one 17-flit packet case ([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf), [PUB-02](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf)). | Track flit occupancy; keep the undisclosed credit accounting quantum separate. |
| TX/RX | One implicit virtual channel and one shared request/response credit domain | **CONTRADICTED** | The protocol has multiple virtual channels, and NVIDIA discloses independent per-VC switch flow control ([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf), [PUB-09](https://patents.google.com/patent/US20230070690A1/en)). | Add explicit VC and traffic-class identities; leave their count and map configurable and undocumented. |
| TX/RX | Credit slot reusable at TX finish plus 200,000 ps | **CONTRADICTED** | Capacity becomes returnable when the downstream destination buffer frees, not at a fixed offset from sender serialization ([PUB-09](https://patents.google.com/patent/US20230070690A1/en)). | Drive credit release from the owning receive or switch buffer, then add an independently parameterized return transport delay. |
| Link | No CRC, acknowledgement, or replay service | **CONTRADICTED** | CRC, ACK, retained unacknowledged packets, timeout, and replay are explicit NVLink mechanisms ([PUB-01](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf), [PUB-03](https://patents.google.com/patent/US20170111144A1/en)). | Add error-free identity behavior plus explicit error/replay events and replay-buffer occupancy. |
| Switch | A100 NV4 `pass_through` | **CONFIRMED** | The four-GPU HGX A100 board has direct NVLinks and no NVSwitch ([PUB-05](https://docs.nvidia.com/datacenter/tesla/hgx-software-guide/index.html)). | Preserve exact tuple, byte, timestamp, and random-draw identity. |
| Switch | Queued profiles choose input, output, or shared FIFO placement | **UNDOCUMENTED** | NVIDIA discloses input VOQs for a GPU switch design, but no accepted source binds a queue layout to a named NVSwitch generation ([PUB-09](https://patents.google.com/patent/US20230070690A1/en)). | Replace flat placement with ports plus VOQs, while keeping product layout configurable. |
| Switch | FIFO-only arbitration and Boolean head-of-line behavior | **CONTRADICTED** | The disclosed NVIDIA design separates destination/VC VOQs and uses stateful two-dimensional round-robin or least-recently-used arbitration ([PUB-09](https://patents.google.com/patent/US20230070690A1/en)). | Implement legality and credit filters, VOQs, crossbar matching, and a policy seam with identity off mode. |
| Switch | Future `service_rate` and byte `buffer_capacity` have no shipped values | **UNDOCUMENTED** | Port rates are public, but internal queue rate and depth are not ([PUB-07](https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf)). | Bound service by port physics and identify depths separately; do not borrow academic simulator values. |
| RX | Effective ingress `207,101,921,876 B/s` | **UNDOCUMENTED** | Public documents give link ceilings, not the GPU's effective destination acceptance plateau. | Preserve its current measured class pending structural realignment and revalidation. |
| RX | `buffer_capacity_bytes = 1,048,576` | **UNDOCUMENTED** | No accepted source gives the A100 destination merge-buffer depth. | Keep as a candidate and require an effective loss-free knee before promotion. |
| RX | `reassembly_policy = extent_sequence` | **CONFIRMED** | Ordering domains and receiver reorder state enforce required sequence visibility across multipath ([PUB-13](https://patents.google.com/patent/US20200374593A1/en)). | Keep the semantic rule and expose reorder occupancy and release. |
| RX | `delivery_order = per_extent` and visibility at RX serializer finish | **UNDOCUMENTED** | Per-operation ordering is supported, but the exact memory-visibility boundary and its latency are not public. | Split ingress completion, reorder completion, and consumer visibility. |
| Transfer | Offered-rate shaping changes packet release by prior modeled wire bytes | **UNDOCUMENTED** | This is a workload generator policy, not a published NVLink mechanism. | Keep it outside hardware service and label it as offered-load construction. |
| Domain | Profile-absent analytic bypass returns the input object by identity | **CONFIRMED** | This is a repository compatibility invariant rather than a hardware claim. It does not conflict with the physical mechanism. | Preserve exact bypass behavior through the alignment. |

Across these 28 choices, the documentary verdict is 10 confirmed, 6
contradicted, and 12 undocumented. The deciding result is not the count by
itself. It is that every remaining numeric credit, buffer, internal endpoint,
and product-arbitration value stays undocumented, so TRAF-73 tunes only those
unknowns after the six structural contradictions are corrected.

## Implied model changes and signed effects

No change below is implemented by this task.

| Implied change | Expected signed effect on existing envelope validations |
|---|---|
| Replace constant `payload + 16` accounting with header, AE, BE, and payload flits | Zero for a command that needs no optional flit. One extra flit raises 256-byte-packet serialization by 5.882 percent and lowers its four-link payload ceiling by 5.556 percent. |
| Separate flit occupancy from the unknown credit-counting quantum | No forced change when credits do not bind. Under depletion, variable occupancy can move the knee in either direction relative to one 272-byte slot, so the signed effect is a TRAF-73 discriminator rather than an assumed correction. |
| Return credits from receive or switch buffer release, then transport the return | With the same 200 ns transport delay, moving the causal start from TX finish to later buffer release adds a nonnegative wait. The large-transfer envelope is unchanged only when the effective window still covers the full round trip. |
| Add explicit virtual channels and request/response/control classes | With one active class, identity policy gives zero change. With mixed classes, separation reduces cross-class head-of-line delay; partitioned credit pools can increase delay for a class that exhausts its own pool. Both effects require class-specific cells. |
| Add CRC acknowledgement and replay-buffer state | Exactly zero time and bytes on the error-free identity path if acknowledgements do not gate a full replay buffer. Injected errors add nonnegative retransmission bytes and completion delay. |
| Add receiver reorder and a separate visibility event | Zero when arrivals are already ordered. Out-of-order multipath arrivals gain a nonnegative visibility hold while correctness improves. |
| Replace future flat switch FIFO with per-port VOQs and crossbar arbitration | Zero on direct NV4 pass-through. On NVSwitch profiles, VOQs reduce avoidable head-of-line delay, while real port grants and output contention add nonnegative service relative to an impossible zero-delay switch. |
| Keep A100's 25 GB/s link rate and four-link peer bundle | No signed change. These are the physical floor and ceiling anchors against which the realigned model is checked. |
| Keep effective endpoint egress and ingress rates as measured parameters | No immediate signed change. Public documents bound them but do not identify their internal mechanism, so replacing them from documents would be unjustified. |

The first post-alignment sanity check has two branches fixed before execution:

- A 17-flit aligned 256-byte write cannot exceed 94.117647 GB/s of payload on
  four A100 links.
- The corresponding 18-flit write with one optional control flit cannot exceed
  88.888889 GB/s.

Any result above its applicable ceiling is void. A result below the ceiling is
not automatically correct; its credit, endpoint, buffer, and ordering terms
must still scale with the registered interventions.

## What public documents cannot decide

The following remain candidates for TRAF-73 or later targeted evidence:

- the A100 credit quantum and numeric pool depth;
- the A100 virtual-channel count and traffic-class mapping;
- whether and how A100 credit returns piggyback, aggregate, or use control
  packets;
- link-credit and replay-buffer interaction;
- A100 replay-buffer and receive-buffer depths;
- the exact A100 bonded-link selection rule;
- the scope of the effective source egress and destination ingress services;
- the deployed NVSwitch queue layout, queue depths, grant interval,
  round-robin/aging rule, and multicast/reduction service rates;
- the exact NVLink 3 and NVLink 4 packet-field continuity from Pascal.

These unknowns are not defects in the document. They are the boundary that
prevents measured plateaus and simulator-friendly constants from being
mistaken for publicly established mechanism.

## Citation inventory

All entries have evidence class **PUBLIC_DOCUMENT**.

| ID | Public source | Kind | Claims used and limitation |
|---|---|---|---|
| PUB-01 | [NVIDIA GP100 Pascal Whitepaper](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper-v1.2.pdf) | Vendor architecture paper | Pascal flits, packet range, physical layer, CRC/replay, flow control, VCs, aggregation. Does not specify later-generation field continuity or credit numbers. |
| PUB-02 | [Foley and Danskin, Hot Chips 28](https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf) | Vendor conference presentation | Header split, AE/BE/data flits, replay sequence, ganged-link spraying, transaction operations. Pascal scope. |
| PUB-03 | [NVIDIA patent US20170111144A1](https://patents.google.com/patent/US20170111144A1/en) | Primary-assignee patent | NVLink 128-bit flits, implicit sequence IDs, unacknowledged-packet storage, replay initiation. Patent embodiments are not later-product specifications. |
| PUB-04 | [NVIDIA A100 Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf) | Vendor architecture paper | NVLink 3 signal pairs, signal rate, per-link direction rate, twelve-link endpoint total, error recovery. Does not expose packet or buffer internals. |
| PUB-05 | [NVIDIA HGX A100 Software User Guide](https://docs.nvidia.com/datacenter/tesla/hgx-software-guide/index.html) | Vendor product guide | Four-GPU direct NVLink versus eight-GPU NVSwitch topology. |
| PUB-06 | [NVIDIA HGX A100 Architecture](https://developer.nvidia.com/blog/introducing-hgx-a100-most-powerful-accelerated-server-platform-for-ai-hpc) | Vendor architecture description | Fully connected direct four-GPU board and 200 GB/s bidirectional peer rate. Does not expose bond scheduling. |
| PUB-07 | [NVIDIA NVSwitch Technical Overview](https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf) | Vendor technical overview | First-generation ports, crossbar, rate, traversal, route tables, and protection. Does not publish arbitration or buffer depth. |
| PUB-08 | [NVIDIA patent US10789194B2](https://patents.google.com/patent/US10789194B2/en) | Primary-assignee patent | Posted/non-posted transactions, multiport write groups, flush ordering, per-port tracking, and crossbar embodiment. |
| PUB-09 | [NVIDIA patent US20230070690A1](https://patents.google.com/patent/US20230070690A1/en) | Primary-assignee patent | Input queues, VOQs, per-destination/per-VC credits, credit-at-dispatch, round robin, least-recently-used aging, and two-dimensional arbitration. Not bound to a named NVSwitch product. |
| PUB-10 | [NVIDIA Hopper Architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | Vendor architecture description | NVLink 4 pairs, links and rates; physical versus network addressing; third-generation NVSwitch multicast and reductions. |
| PUB-11 | [NVIDIA Grace Hopper Architecture](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/) | Vendor architecture description | NVLink-C2C coherence and 450 GB/s per direction; separate external Hopper NVLink 4 scale-up path. |
| PUB-12 | [NVIDIA Fabric Manager User Guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/) | Vendor product manual | Request/response routes, link maps, finite multicast slots, H100-and-later multicast and reduction support. |
| PUB-13 | [NVIDIA patent US20200374593A1](https://patents.google.com/patent/US20200374593A1/en) | Primary-assignee patent | Ordered transfers, ordering domains, target reorder, acknowledgements, and multipath operation. Not an A100 packet-field specification. |
| PUB-14 | [Su, UpDown network requirements](https://people.cs.uchicago.edu/~aachien/lssg/research/10x10/Jiya_Su_MS.pdf) | Academic secondary synthesis | Interprets transaction-header content, including flow-control credits. Used only as corroboration because its packet source is PUB-02. |
| PUB-15 | [Zhang et al., CAIS](https://chenzhangsjtu.github.io/files/2026-HPCA-CAIS.pdf) | Academic simulator paper | Shows that numeric queue, VC, and arbitration values in the literature can be model inputs rather than disclosures. Its product-unverified values are explicitly not imported. |

## Registry consequence

TRAF-79 is literal with this cited reconstruction and reconciliation. TRAF-80
owns the model alignment: implement generation-scoped flit accounting,
receiver-driven credit release, explicit virtual-channel and traffic-class
identity, link replay, receiver reorder, and an NVSwitch port/VOQ/arbitration
structure while preserving the exact analytic and direct-mesh identity paths.
TRAF-73 remains the gate that identifies only the numeric and policy choices
the public record leaves undocumented.

This document changes no runtime profile, timestamp, hardware artifact, or
reported time to first token (TTFT) or time per output token (TPOT). It changes
the order of work: mechanism alignment precedes further numeric fine-tuning.
