# Merlin multi-node collective capture result

## Outcome

What ran: four A100 Slurm jobs captured all-gather, reduce-scatter,
all-reduce and pairwise all-to-allv over the frozen 8-byte through 128-MiB
ladder at widths 2 and 8 under declared one-port and four-port routing.

What came out: the campaign is **NONVOID** because the only campaign-voiding
guard, FG-4, held. FG-2 applies to cells. Separate TX and RX evaluation finds
317 contradicted cells, 35 cells below the frozen 1 MiB signal minimum, and
zero proven cells. Family R therefore has 129 VOID rows, 21 UNEVALUABLE rows,
and zero scoreable rows. Independent Family C passes all three anchors, Family
L publishes all 352 cells without a duplicate, and Family W passes.

What it changes: TRAF-77 narrows to the literal achieved evidence: phase
timing at widths 2 and 8 with anchors held; endpoint proxies captured;
concentration control refuted pending a working pinning mechanism; switch
occupancy remains unobservable. TRAF-77 stays open. TRAF-87 now owns working
TX-side routing control, the unresolved TX and RX counter semantics, and the
two capture-harness portability defects.

What it does not change: this campaign supplies no routing-direction score,
fabric-only width-8 attribution, fitted transport calibration, H200 evidence,
switch occupancy, receiver occupancy, queue-wait, buffer high-water, time to
first token (TTFT), or time per output token (TPOT) acceptance. It closes no
task and does not widen a frozen band.

## Chronology and identity

The expectations-only commit is `9c9a42e`. The harness followed at `4bdf437`,
and the hierarchical-topology warning was frozen in addendum commit `d49679b`
before capture. The integrator reported a pre-capture outage; the fetched
evidence does not independently timestamp it, so the outage remains unscored
chronology. The recorded flagship ping and reply cleared the A100 submission
fence.

The submitted tree was `f6bc59cee65019b876b29d642ae16790192c1162`.
That commit fixed compute-node interpreter selection. The submitted-script
hashes match on the local side, the remote side before submission, the remote
side after the run, and inside every attempt. Both normalized files are
byte-identical at SHA-256
`80a7852b42ad756493b1bdc1d91f314f766483d9f937823b30f64d219334d6aa`.
The scored record hashes all 179 fetched evidence files in place and stores
only evidence-root-relative paths.

Three runbook deviations are retained:

- The wheel finder was architecture blind and selected the GH200 AArch64
  wheel first. The integrator explicitly pinned the x86-64 NCCL 2.31.2 wheel.
- A fixed scratch path replaced `mktemp` because the integrator shell did not
  retain variables between commands. Hashes proved the staged contents
  identical.
- The login-node `python3` was 3.6.15, below the hash helper's requirement.
  Login-side commands selected the site's Python 3.11 interpreter explicitly.

Jobs `202415`, `202416`, `202417` and `202418` all completed with exit code
`0:0`. Their Slurm elapsed times were 54, 47, 78 and 73 seconds. The attempt
records span 2,435 seconds from the first start to the last finish; the four
Slurm elapsed times sum to 252 seconds.

## Physical sanity before the scores

Coverage arithmetic gives
`2 widths * 2 declarations * 4 operations * 22 payloads = 352 cells`.
The capture contains exactly those 352 cell identifiers and no duplicate.
Each published median is no faster than its payload divided by the declared
25 GB/s per-port rate. The logs provide no finite progress ceiling, so the
honest upper physical bound is unbounded.

The job-level RX snapshot also agrees with independent per-cell arithmetic.
Each cell records 25 measured repeats after five warmups. On width 8,
one-port gpu101, the 352-cell subset for that attempt sums to
21,268,107,469 RX bytes. Multiplying by `30 / 25` predicts
25,521,728,962.8 bytes; the snapshot measured 25,527,859,035 bytes, a
quotient of 1.000240190. On gpu105 the same calculation predicts
25,579,657,987.2 bytes and measures 25,595,714,529 bytes, a quotient of
1.000627707. The other six attempt-node readings are within 1.52 percent of
the same independent projection. This check supports the RX accounting before
any concentration or asymmetry claim is read.

## Frozen family outcomes

### Family C: PASS

Physical sanity first: an 8-byte payload over four declared 25 GB/s ports has
a 0.08 ns serialization floor. All three observed completions are tens of
microseconds, above that floor, and the source provides no finite upper bound.

| Frozen anchor cell | Observed median | Observed / anchor | Frozen band | Outcome |
|---|---:|---:|---:|---|
| width 2, four-port, all-reduce, 8 B | 60.403 us | 1.504778218291071 | [0.5, 2.0] | PASS |
| width 8, four-port, all-reduce, 8 B | 71.664 us | 1.410986414648553 | [0.5, 2.0] | PASS |
| width 8, four-port, all-to-allv, 8 B | 126.407 us | 1.4075719614720783 | [0.5, 2.0] | PASS |

### Family R: UNEVALUABLE

Physical sanity first: the frozen routing proof requires at least 95 percent
on hsn0 for one-port and at least 5 percent on every hsn port for four-port,
after at least 1 MiB moved in the cell's node-level TX-plus-RX counter window.
The separate TX and RX predicate finds 317 contradicted cells, 35
insufficient-signal cells and zero proven cells. The six width-8 four-port
cells that passed the old pooled-fraction calculation have balanced RX but
more than 99 percent of TX on one port, so none survives directional testing.

Family R publishes 129 **VOID** rows because at least one consumed cell is
contradicted and 21 **UNEVALUABLE** rows because at least one consumed cell is
below the signal minimum and none is contradicted. No row consumes only proven
cells, so the behavioral denominator is zero. E1 contributes 75 VOID and 13
UNEVALUABLE rows; E2 contributes 42 VOID and 2 UNEVALUABLE rows; E3 contributes
12 VOID rows; and all 6 E4 rows are UNEVALUABLE. The concentration-control
refutation is cell-scoped and undiminished: contradicted cells remain void and
are never relabeled.

### Family L: PASS

Physical sanity first: the frozen Cartesian product has 352 cells. The ledger
has 352 distinct cells, zero missing cells and zero unexpected cells. Each row
publishes its median, p95 and serialization-floor quotient in
[results.csv](results.csv).

Family L passes 352 of 352 completeness rows. This is a completeness score,
not permission to interpret concentration direction.

### Family W: PASS

Physical sanity first: the frozen ceiling is 600 seconds for analysis. Two
fresh scorer builds completed inside that ceiling and reproduced
[record.json](record.json) and [results.csv](results.csv) byte for byte.

Family W passes. Cluster execution time is disclosed separately and is not
mixed into the analysis wall score.

## Achieved concentration

The table records `ip -s -j link` `stats64` byte deltas. Each port entry is
`TX / RX` bytes. These are the achieved observations, not relabeled routing
conditions.

| Attempt and declaration | Node | hsn0 TX / RX | hsn1 TX / RX | hsn2 TX / RX | hsn3 TX / RX |
|---|---|---:|---:|---:|---:|
| w2 four-port, job 202416 | gpu101 | 29,366,721 / 3,124,366,386 | 24,131,805 / 3,124,227,264 | 66,445,127,755 / 37,133,635 | 204,600 / 31,116 |
| w2 four-port, job 202416 | gpu102 | 21,460,664 / 3,124,390,872 | 66,452,882,175 / 41,949,441 | 9,087,028 / 67,438 | 18,437,977 / 3,128,797,974 |
| w2 one-port, job 202415 | gpu101 | 27,237,497 / 3,124,346,448 | 24,128,399 / 3,124,224,340 | 66,445,093,992 / 37,124,246 | 204,014 / 30,682 |
| w2 one-port, job 202415 | gpu102 | 21,460,240 / 3,124,386,353 | 66,452,864,314 / 41,944,760 | 9,086,118 / 66,644 | 18,433,443 / 3,128,794,070 |
| w8 four-port, job 202418 | gpu101 | 43,744,933 / 6,204,334,782 | 40,081,030 / 6,209,959,362 | 255,825,983,054 / 6,374,018,674 | 35,335,429 / 6,209,161,044 |
| w8 four-port, job 202418 | gpu105 | 255,992,640,848 / 6,361,513,885 | 30,496,495 / 6,194,094,606 | 49,920,233 / 6,189,797,310 | 40,005,916 / 6,192,657,512 |
| w8 one-port, job 202417 | gpu101 | 167,822,819 / 25,370,948,372 | 2,243,225 / 57,842 | 262,604,484,054 / 156,787,647 | 2,217,916 / 65,174 |
| w8 one-port, job 202417 | gpu105 | 262,091,603,087 / 25,591,799,681 | 3,945,594 / 2,336,482 | 3,268,554 / 387,564 | 1,654,127 / 1,190,802 |

The width-8 four-port jobs received approximately 6.2 GB on each hsn port,
but each node transmitted more than 99.95 percent of its counted TX bytes on
one port. The width-8 one-port job received more than 99.38 percent on hsn0
at both nodes. gpu105 also transmitted on hsn0, while gpu101 transmitted more
than 99.93 percent on hsn2. At width 2 the one-port and four-port snapshots are
nearly identical: each node received approximately 3.12 GB on two ports and
transmitted more than 99.9 percent on one non-hsn0 port.

### Per-direction concentration verdicts

The job-level snapshots have ample signal in both directions. Applying the
declared concentration to TX and RX separately gives:

| Attempt and declaration | Node | TX verdict | RX verdict |
|---|---|---|---|
| w2 four-port, job 202416 | gpu101 | CONTRADICTED | CONTRADICTED |
| w2 four-port, job 202416 | gpu102 | CONTRADICTED | CONTRADICTED |
| w2 one-port, job 202415 | gpu101 | CONTRADICTED | CONTRADICTED |
| w2 one-port, job 202415 | gpu102 | CONTRADICTED | CONTRADICTED |
| w8 four-port, job 202418 | gpu101 | CONTRADICTED | PROVEN |
| w8 four-port, job 202418 | gpu105 | CONTRADICTED | PROVEN |
| w8 one-port, job 202417 | gpu101 | CONTRADICTED | PROVEN |
| w8 one-port, job 202417 | gpu105 | PROVEN | PROVEN |

## Transport mechanism

The hardware moved bytes from GPU buffers through host memory and ordinary
kernel sockets. Every rank reported GDR 0, so GPU Direct Remote Direct Memory
Access was inactive. NCCL could not load `libnccl-net.so`, `NET/IB` found no
device, and every rank selected NCCL's built-in `Socket` transport. No Open
Fabrics Interfaces (OFI) or Cassini CXI plugin was selected.

The width-2 jobs used RING with LL and SIMPLE protocols. Each attempt reported
two collective channels and had eight logical network connections: four
collective and four shared peer-to-peer connections. Four-port connectors used
socket devices 2 and 4 from a map containing management interface nmn0 plus
hsn0 through hsn3. One-port connectors used only socket device 0, mapped to
hsn0.

The width-8 jobs used RING and TREE with LL and SIMPLE. The one-port job had
two collective channels and 72 logical network connections: eight collective
and 64 shared peer-to-peer. The four-port job had eight collective channels
and 96 logical connections: 32 collective and 64 shared peer-to-peer. Its
connectors used all four hsn socket devices.

In the one-port logs, `NCCL_SOCKET_IFNAME` is `=hsn0`, NCCL exposes only hsn0
as socket device 0, and its proxy listeners use the hsn0 addresses. The gpu101
TX counter nevertheless lands on hsn2. The evidence therefore localizes the
divergence below NCCL's logical socket-device selection. The strongest
supported inference is that `NCCL_SOCKET_IFNAME` constrained address discovery
and the listener, while an unobserved Linux route, provider layer, or Cassini
driver accounting decision selected or charged the physical TX interface.
The capture lacks `ip route`, `ip rule`, route-get, socket-binding and provider
state, so it cannot decide which of those lower layers made the choice.

## Why TX bytes are about ten times RX bytes

Physical sanity first: width-8 one-port gpu101 has 262,776,768,014 counted TX
bytes against 25,527,859,035 RX bytes, a quotient of 10.2937. Yet it has
33,649,046 TX packets and 33,508,230 RX packets, and the opposite node has
33,229,717 TX packets and 33,844,821 RX packets. Every captured error and drop
delta is zero. Those Linux packet and error fields cannot exclude
retransmission below the offload accounting boundary.

The Linux TX and RX byte fields are not symmetric wire-byte authorities.
Across the attempts, TX averages roughly 7.6 to 7.9 KiB per counted packet
while RX averages roughly 0.73 to 0.75 KiB per counted packet. The independently
reconstructed RX total matches the payload ladder and warmup arithmetic, but
the capture contains no authoritative Cassini hardware counter that identifies
the wire-byte relation. The exact TX and RX field semantics remain unexplained
and are assigned to TRAF-87.

## Width-8 interpretation limit

The width-8 logs contain both `P2P/CUMEM/read` intra-node connectors and
`NET/Socket` cross-node connectors. A completion therefore combines an
intra-node NVLink stage with the fabric stage. The capture does not time those
stages separately. In accordance with the frozen pre-capture addendum, no
width-8 one-port to four-port ratio is interpreted as a fabric-only
concentration result. The phase timings remain useful achieved observations,
but they cannot identify switch service or dilute a refutation into a pass.

## Reproduction

Set `SIMLLM_TRAF77_EVIDENCE_ROOT` to the fetched append-only evidence root, or
pass `--evidence-root`, then run:

```bash
env -u PYTHONPATH python examples/merlin_collective_capture_v1/run_study.py --check
```

The runner reads evidence in place, verifies the normalized and submitted
hashes, recomputes counters and mechanism facts from raw snapshots and NCCL
logs, evaluates the frozen families, and checks the tracked JSON and CSV byte
for byte. The test suite retains a named skip when the external evidence root
is unavailable.
