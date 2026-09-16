# Simple buffered-read mechanism results, version 1

TRAF-93's retained-calendar implementation passes the frozen synthetic
mechanism study. This is causal and accounting evidence, not an H100/A100
calibration result: every rate and GPU service cost used here comes from the
versioned fixtures in this directory.

## What was exercised

- Isolated peer reads preserve request timing while an externally controlled
  source-memory delay shifts response start and final visibility by exactly the
  same amount.
- Source reads consume the buffer owner's GPU-memory cursor even though the
  requesting block resides on the receiving GPU.
- Simple `buffered_read` uses request packets, owner-memory service, and
  response packets; Simple `buffered` keeps the existing receiver-side write.
- LL and LL128 remain byte-for-byte and visit-for-visit identical under a
  read-capable communicator because the placement switch applies only to
  Simple.
- Empty Simple slices retain their control operations without inventing a
  zero-byte payload read.
- Both placements flow through the original graph-to-request metric path with
  the declared compute duration unchanged.

## Reproduction

From the repository root:

```bash
python examples/nccl_simple_read_v1/run_study.py --output /tmp/nccl-simple-read-v1
```

The 2026-09-11 run completed with all fatal guards valid and produced 9
isolated-read rows, 6 remote-memory rows, 288 collective rows, and 32 graph
metric rows. The output CSV files are intentionally not committed because they
are reproducible synthetic expansions; `summary.json` in the selected output
directory records the row counts and evidence class.

## Interpretation boundary

Passing this study says that the simulator represents the required order:

```text
request visible -> source-owner memory service -> response eligible
                 -> consumer work -> returned head
```

It does not determine the real memory, polling, synchronization, or NVSwitch
costs. TRAF-94 owns those measurements, and H100 results require a separately
frozen, capability-qualified inventory rather than relabeling the existing
A100/GH200 design.
