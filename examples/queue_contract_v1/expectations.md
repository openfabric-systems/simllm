# Queue service contract v1 expectations

## Freeze status

This file is the expectations-only record for the first shared queue-service
contract. It is written before the contract implementation or any run of its
conformance study. Existing component outputs may be used as regression
baselines, but no result from the new study is included here. The results must
cite both the original freeze commit and the final pre-run expectations commit.

## Scope

The study will define one semantic queue visit shared by the core, GPU and
RNIC models. Python and C++ may use different implementations, but they must
emit the same fields and pass the same golden fixtures.

Each non-preemptive visit has these ordered timestamps:

1. `submitted_at`: the request enters the logical queue.
2. `eligible_at`: dependencies and ordering gates outside this resource have
   cleared.
3. `started_at`: the arbitration policy grants the resource and service begins.
4. `finished_at`: this resource releases the request.
5. `completed_at`: downstream visibility or completion occurs.

For one visit, queue wait is `started_at - eligible_at`, service is
`finished_at - started_at`, and downstream response is
`completed_at - finished_at`. Upstream dependency delay is not queue wait.
The sum of visit waits is a work total, not a critical-path delay. Only a
separately reduced critical-path queue delay may be added to TTFT or TPOT.

The explicit off policy is identity. It ignores class and priority and
preserves each mechanism's deterministic baseline order after protocol
legality is applied. In the reference serial-resource fixture below, that
baseline is FIFO by `(eligible_at, enqueue_sequence)`. The shared contract does
not impose global FIFO on GPU schedulers or protocol resources.

## Exact conformance cases

All timestamps are integer ticks in the fixture; a component may map a tick to
cycles or picoseconds without changing the relations.

1. **Isolated service.** One request submitted and eligible at 0 with service
   10 starts at 0, finishes at 10 and has queue wait 0.
2. **FIFO contention.** Two requests submitted and eligible at 0 with service
   10 start at 0 and 10, finish at 10 and 20, and report waits 0 and 10. The
   total visit wait is 10, not a repeated charge from the transaction origin.
3. **Upstream delay.** A request submitted at 0 but eligible at 100 on an idle
   resource starts at 100 and reports queue wait 0, not 100.
4. **Fragment chain.** Three fragments with service 10 and no external
   contention use the preceding fragment finish as the next fragment's
   eligibility. Every fragment reports queue wait 0. A competing request that
   occupies `[10, 20)` delays only the fragment that is eligible at 10, and the
   delay is charged once.
5. **Finite capacity.** A request blocked only by a capacity resource becomes
   eligible for service at the recorded release time. Capacity delay and
   serializer queue wait are distinct and their intervals do not overlap.
6. **PCIe posted progress.** A non-posted read waiting on a finite read or
   completion resource is not a legal ready candidate. A posted write eligible
   during that interval uses an idle link opportunity before the read. The
   identity policy never reverses this mandatory legality rule.
7. **Identity labels.** Changing only class and priority labels under identity
   changes no selected order, timestamp, wait, byte count, counter or random
   draw.
8. **Overflow rollback.** Timestamp overflow fails before any queue cursor,
   reservation, capacity counter, random-draw cursor or owner container is
   mutated.

The Python and C++ fixture outputs must be byte-identical after normalizing the
time-unit label. Existing PCIe and Work Queue accepted rows must remain
byte-identical during extraction of their private mechanisms into the shared
semantic contract. The GPU task-mix result rows must also remain byte-identical
when HBM and NVLink visits are exposed.

## Evidence classes

The report will keep these counts separate:

- exact fixture rows;
- behavioral relation families and their parameterized instances;
- cross-language conformance rows;
- structural invariants;
- unchanged component-regression rows.

Inactive fields and by-construction zeros are structural invariants. They do
not increase a behavioral pass denominator.
