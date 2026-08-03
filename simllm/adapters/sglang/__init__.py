"""SGLang adapter (milestone M3).

The seam is the TP worker: ``SimTpModelWorker`` implements SGLang's
``BaseTpWorker`` interface, and its ``forward_batch_generation(batch)``
returns a ``GenerationBatchResult`` with fabricated ``next_token_ids`` and
simulated timing. It is selected at the scheduler's worker-construction
point; the first iteration targets ``--disable-overlap-schedule``.

RadixCache prefix matching, eviction and the token/request pool accounting
are scheduler-side and stay real, so radix hit rates and vRAM pressure
respond to the workload exactly as in production.

This package intentionally has no import-time SGLang dependency.
"""
