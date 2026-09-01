# Matched-seam LogGOPSim third-arm result

## Outcome

What ran: append-only `attempt-0001` reran the complete corrected matched-seam
evaluation twice in fresh processes, executed the pinned LogGOPSim binary on
the exact TP4-to-TP2, TP4, and TP8 packet GOAL binaries, exercised the explicit
zero-service bypass, and published all ten three-arm decompositions.

What came out: the run is nonvoid. All 15 fatal guards hold, the two complete
evaluation payloads are byte-identical, and the original S, R, F, M, and W
tallies remain 13 of 13, 10 of 10, 12 of 13, 2 of 2, and 1 of 1. The deciding
new number is a 1.021230716258 frontier penalty beyond the priced LogGOP terms
on TP2 rows 1 through 3. Its network-leg residual is 2.365525200 ms.

What it changes for the project: DEPLOY-12 closes. The LogGOP arm prices
2.295758 ms of latency and sender serialization on TP2, while the packet arm
costs 4.661283 ms. The 2.365525 ms remainder survives at the frontier and is
specific to the TP4-to-TP2 schedule in this grid. The TP2 receiver must accept
458,752,000 bytes through two 400 Gbit/s ingress endpoints, whose 4.587520 ms
floor explains 2.291762 ms of the packet-over-LogGOP remainder before the
packet leg's final 73.763 microseconds above that floor. No residual task ID is
needed or consumed.

What it does not change: the first publication remains void, the corrected
two-arm publication remains byte-identical, and its 1.042715399805
packet-to-unpriced observation remains valid under its original name. The
result does not repair F-2-09, widen any band, validate either planner against
hardware, establish a general contention coefficient, close a calibration
task, or move DEPLOY-13 and the breadth or silicon-precision tasks.

## Chronology and frozen inputs

The expectations-only freeze is commit
`2db8595ab869f500d6da2b0690d977dd11093ff6`, SHA-256
`ed784f7514fe766c509b02ed591391370129b84c63cc51552e278f5fcee44812`.
The implementation is commit
`d736ec6bbbf7a246a032dbe88b74b6b3070df836`, which is also the run commit.
The freeze therefore precedes both implementation and execution.

The native identities are:

| Tool | SHA-256 |
|---|---|
| LogGOPSim | `7e0f13ee3c87a20e9d2e94dbbd74c46075fd03df2f1b04d1ed9739c43ee0a2bf` |
| `htsim_rnic` | `cfb5014a663791f7619fe33309114a74e82878de860c14fc8a723713501f027d` |
| `txt2bin` | `f3745f34ad86febe9c9eebef10aee5fae00b8865cb29943344fb75b0f142495b` |

Every LogGOPSim cell used `L=2000` ns, `o=0`, `g=0`, exact
`G="0.02"` ns/byte, `O=0`, `S=9223372036854775807`, and network type
`LogGP`. The recorded argument vectors use portable GOAL paths and the exact
decimal `G` spelling.

## Physical sanity before the measured values

The freeze states each bound before any third-arm execution. Passing a bound is
necessary and is not treated as proof.

### Sender and receiver physics

Each of four senders owns one quarter of the 458,752,000-byte payload. Bytes
over one 400 Gbit/s sender link set a 2.293760 ms floor. The three LogGOP
services are 2.295758, 2.295756, and 2.295752 ms at TP2, TP4, and TP8. Each is
about 2 microseconds above the sender floor and far below its conservative
9.191040 to 9.239040 ms serial ceiling.

TP2 changes the receiving bound. Two destinations each accept half the payload,
so the receiver floor is 4.587520 ms. The packet service is 4.661283200 ms,
73.763200 microseconds above that bound. The LogGOP model has no receiver
per-byte gap and remains sender-limited at 2.295758 ms. Thus
`4.661283200 - 2.295758000 = 2.365525200` ms survives beyond the priced terms,
and `4.587520000 - 2.295758000 = 2.291762000` ms of it is required by receiver
serialization alone.

### Width scaling

Doubling destination width halves the per-flow payload, while total bytes per
sender stay fixed. The LogGOP services remain nearly constant, moving only
from 2.295758 ms at TP2 to 2.295752 ms at TP8 as the independently floored
message terms change. Packet-minus-LogGOP residuals are 2.365525 ms at TP2,
0.035927 ms at TP4, and 0.036014 ms at TP8. The TP2 residual is about 65.7
times the TP4 or TP8 baseline and appears only where receiver bandwidth is
tighter than sender bandwidth.

### End-to-end capacity

Rows 1 and 3 are prefill-limited in all three arms. Their unpriced,
LogGOP-priced, and packet coordinates are respectively 773.201649,
757.270176, and 741.527026 tokens/s/GPU on row 1, and 644.334708,
631.058480, and 617.939188 tokens/s/GPU on row 3. Exact rational arithmetic
gives:

```text
1.042715399805 = 1.021038031078 * 1.021230716258
total packet penalty = priced LogGOP penalty * residual penalty
```

The composition therefore keeps the measured network effect live at the
reported capacity boundary. It does not infer a residual from an unreachable
component probe.

## Three-arm decomposition

Network services repeat by TP cell. Coordinates and capacity limiting are
row-specific. `Priced` is unpriced divided by LogGOP-priced capacity,
`Residual` is LogGOP-priced divided by packet capacity, and `Total` is unpriced
divided by packet capacity.

| Row | TP | Network ms, unpriced / LogGOP / packet | tokens/s/GPU, unpriced / LogGOP / packet | Priced | Residual | Total | Frontier residual |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 0 / 2.295758 / 4.661283 | 773.201649 / 757.270176 / 741.527026 | 1.021038031078 | 1.021230716258 | 1.042715399805 | yes |
| 2 | 2 | 0 / 2.295758 / 4.661283 | 764.091736 / 757.270176 / 741.527026 | 1.009008093396 | 1.021230716258 | 1.030430057929 | yes |
| 3 | 2 | 0 / 2.295758 / 4.661283 | 644.334708 / 631.058480 / 617.939188 | 1.021038031078 | 1.021230716258 | 1.042715399805 | yes |
| 4 | 4 | 0 / 2.295756 / 2.331683 | 601.373518 / 601.373518 / 601.373518 | 1.000000000000 | 1.000000000000 | 1.000000000000 | hidden by decode |
| 5 | 4 | 0 / 2.295756 / 2.331683 | 540.507462 / 540.507462 / 540.507462 | 1.000000000000 | 1.000000000000 | 1.000000000000 | hidden by decode |
| 6 | 8 | 0 / 2.295752 / 2.331766 | 440.267555 / 440.267555 / 440.267555 | 1.000000000000 | 1.000000000000 | 1.000000000000 | hidden by decode |
| 7 | 4 | 0 / 2.295756 / 2.331683 | 386.600825 / 378.635095 / 378.513044 | 1.021038012750 | 1.000322448564 | 1.021367244991 | yes |
| 8 | 4 | 0 / 2.295756 / 2.331683 | 343.645178 / 336.564529 / 336.456039 | 1.021038012750 | 1.000322448564 | 1.021367244991 | yes |
| 9 | 8 | 0 / 2.295752 / 2.331766 | 257.800548 / 257.800548 / 257.800548 | 1.000000000000 | 1.000000000000 | 1.000000000000 | hidden by decode |
| 10 | 8 | 0 / 2.295752 / 2.331766 | 156.933025 / 156.933025 / 156.933025 | 1.000000000000 | 1.000000000000 | 1.000000000000 | hidden by decode |

At TP2, the priced terms account for 49.2516 percent of the packet network
service and the positive residual accounts for 50.7484 percent. Rows 1 and 3
retain the protected total quotient 1.042715399805. Row 2 starts decode-limited
in the unpriced arm and becomes prefill-limited when either network arm is
enabled. TP4 has a smaller frontier-visible residual on rows 7 and 8. The TP4
and TP8 network-leg residuals remain published even where decode capacity hides
them from the frontier.

## Fatal guards and determinism

All inherited corrected guards FG-1a, FG-1b, FG-1c, FG-2, FG-3, FG-4, and FG-5
hold. All new guards FG-A through FG-H hold:

- the prior void and corrected publication set is byte-identical before and
  after both evaluations;
- tool identities and exact argument vectors match the freeze;
- every LogGOP cell consumes the packet cell's exact GOAL text and binary;
- inherited scored families and bands remain unchanged;
- the freeze precedes implementation and execution;
- the two complete fresh-process payloads are byte-identical;
- the explicit bypass starts zero LogGOPSim processes and reproduces the
  corrected unpriced points and frontier byte for byte;
- all three native LogGOP values lie inside their frozen physical bounds.

The two complete evaluation files both have SHA-256
`18d29e03fb3fc7a48bdf160c8e75129b0fe72f7f4a994666eac2a23156c880bd`.
Only `elapsed_seconds` and `W-1` are excluded, by name. The coordinator took
256.431386 seconds against the unchanged 600 second W ceiling.

Fatal guards, original behavioral families, wall time, and the new unscored
decomposition remain separate evidence classes. No denominator combines them.

## Records and reproduction

- [deploy12_record.json](deploy12_record.json) is the strict portable record.
- [deploy12_results.csv](deploy12_results.csv) is the LF-only flattened ledger.
- [expectations_deploy12.md](expectations_deploy12.md) is the immutable freeze.
- [record.json](record.json), [results.csv](results.csv), and both existing
  figure pairs remain the protected corrected publication.

Bulk GOAL texts, binaries, native completions, fresh-process JSON, and stderr
remain outside Git under the configured append-only root.

```bash
.venv/bin/python examples/matched_seam_frontier_v1/run_deploy12_arm.py \
  --bulk-root "${SIMLLM_DEPLOY12_BULK_ROOT:?configure an external run root}" \
  --external-venv "${SIMLLM_EXTERNAL_AIC_VENV:?configure the external venv}" \
  --htsim-rnic "${SIMLLM_HTSIM_RNIC:?configure htsim_rnic}" \
  --txt2bin "${SIMLLM_TXT2BIN:?configure txt2bin}" \
  --loggopsim "${SIMLLM_LOGGOPSIM:?configure LogGOPSim}" \
  --write-tracked
```
