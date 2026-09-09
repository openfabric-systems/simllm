# DGX NVLink topology and native switch result

The first native switch grant/service kernel agrees exactly with the Python
reference in 192 component configurations and 36 live request configurations.
No fatal guard was violated in the completed run. Explicit A100/H100 wiring
now drives tensor-parallel and expert-parallel requests through the existing
physical packet calendar.

## What ran

The expectations-only commit is `24f41b8`, preceding implementation and the
first run. `run_study.py` swept both eight-GPU products, two directional link
rates, two payloads, two buffer capacities and both declared grant policies.
Packet patterns include isolated, bidirectional, disjoint and 1/3/7-source
fan-in. Live cases use widths 2/4/8, tensor-parallel all-reduce and balanced or
concentrated expert dispatch/combine, with one prefill and two decode steps.

These are synthetic model-conformance experiments, with zero hardware
captures. The fixed compute input is 37,000 ps per step, deliberately exposing
transport. It is not a measured transformer kernel or a real LLM throughput
prediction. Concentrating expert ownership can shorten this fixture's
communication schedule; the fixed provider does not model the corresponding
change in expert compute load.

## Evidence

| Evidence class | Result |
|---|---|
| Component configuration rows | 192, each run with Python and native allocation |
| Exact component conformance | 192 comparisons have zero residual across grants, packet timestamps and buffer/credit records |
| Independent one-packet equation | 16 instances agree exactly |
| Isolated/disjoint link-rate direction | 32 instances satisfy the frozen direction |
| Delayed-credit backpressure | 2 product instances show additional waiting and recovery |
| Live configurations | 36, each run with Python and native allocation |
| Exact live conformance | 36 comparisons have zero residual in request metrics and final packet observations |
| Reachable transport effect | All 18 live rate pairs change completion time; compute service stays identical |
| Fatal physical/state guards | No violations in the completed run |
| Hardware calibration | No captures; no hardware precision task closes |

Counts in different evidence classes are not added. The compact
[results](results.json) retain every configuration and live metric. Raw packet
and request observations remain in the external evidence directory supplied
through `--output`.

For the A100 width-8 tensor-parallel fixture, doubling each attachment from
12.5 to 25 GB/s reduces time to first token and time per output token from
751,560 to 429,000 ps. The three-step completion falls from 2,254,680 to
1,287,000 ps. This is not exactly a factor of two because compute, propagation
and receiver service remain fixed. The packet-level byte/rate floors and the
independent one-packet equation hold before conformance is accepted.

The first attempt was void because its result writer could not serialize a
`RequestPhase` enum. Its `VOID.json` is retained externally. The fix encodes
enums by their public value; no model behavior, parameter cell or frozen
acceptance relation changed. The second attempt completed under the original
freeze and is the reported result. A final reproduction with `--check`
produces the same committed summary and retains the selected native library
digest in the raw observation records.

## Effect and remaining work

PLACE-6's nominal product presets and BACK-74's native crossbar service are
implemented and live-reachable. Their remaining work is captured hardware
port binding and extending native ownership beyond crossbar service. TRAF-92
still owns product calibration; TRAF-54 owns collective protocol fidelity and
BACK-73 owns detailed critical-path attribution. These P1 tasks remain open.
No M4/M5 hardware-precision milestone closes from software agreement alone.

The default analytic path and existing Python selection retain their behavior.
The native model is selected by an explicit library argument and fails rather
than silently falling back when that library is unavailable. H100 reduction
offload, optional reference RTL (BACK-75), Merlin calibration and NVL72 remain separate
work; no result here identifies undocumented vendor queues or arbitration.

A read-only Merlin GPU-cluster inventory on 2026-09-09 lists four-A100-SXM4
and four-GH200 nodes. No eight-GPU switched allocation appears in that
inventory. This establishes the available calibration topology classes, not
new link measurements or calibration results.
