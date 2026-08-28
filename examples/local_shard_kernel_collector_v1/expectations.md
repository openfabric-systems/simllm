# Local-shard kernel collector expectations

This freeze owns the first executable local `run` slice of COMP-50. It is
committed before the implementation and before the conformance study runs.

The collector measures one declared physical shard with synthetic inputs. The
request keeps the logical tensor, pipeline, data and expert parallel sizes
separate from the physical shard coordinate. A framework target may compile
and execute the rank-local kernels only when it can prove that mapping. It
rejects an unavailable mapping rather than claiming that one GPU executed the
distributed system.

The target must report the exact framework, model revision, device ISA,
numeric format and shard it executed. Any disagreement with the request voids
the cell. In particular, an A100 can produce SM80 evidence only; it cannot
stand in for an SM90 or AMD implementation.

The no-hardware conformance study sweeps tensor parallel sizes 1 and 4 and
batch sizes 1 and 8. All four fixture cells preserve their exact request
identity. Synthetic input rows equal the batch size, the local GEMM output
width scales inversely with tensor parallel size, repeated output order is
stable, and either swept axis changes the content-addressed result.

Kernel raw outputs use a caller-supplied untracked root. The tracked
`offline/calibration/kernel` and `offline/calibration/network` namespaces state
the separate review boundaries; a kernel run never writes network evidence.

Passing this study makes the local run slice of COMP-50 literal. It does not
close COMP-50, install a GPU or network constant, promote a calibration
record, or move time to first token (TTFT) or time per output token (TPOT).
