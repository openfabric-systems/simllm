"""Host-side initiation model: what happens before bytes hit the wire.

A rank-to-rank handoff involves a control-plane "doorbell": the small packet
(or flag write) that tells the next rank its input is ready. The full real
path is GPU → (CPU proxy thread or GPU-initiated networking) → PCIe → RNIC →
wire, each stage with its own latency and jitter.

SimLLM's default deliberately collapses the host side to zero:

- **no host-side jitter** (no CPU scheduling, no proxy-thread wakeup),
- **zero RNIC doorbell-to-wire time**, **no PCIe transfer time or jitter**,
- the doorbell itself IS modeled — as a small in-band control message on the
  fabric (the RNIC endpoint models already carry high-priority ~64 B control
  packets), so it competes for wire time and experiences *network-side*
  jitter, which is this project's focus.

Host-path effects (GPU-initiated networking vs a CPU proxy thread) differ by
roughly constant microseconds per operation and are orthogonal to fabric
behavior; folding them in by default would confound network attribution.
They matter only when per-message launch overhead rivals transfer time
(small-message MoE all-to-all), so the knob exists but defaults to zero:
``initiation_delay_ps`` is a per-endpoint constant added before a send
becomes wire-visible. Calibrated constants (e.g. sub-µs for GPU-initiated,
a few µs for CPU-proxy paths) can be swapped in to study the launch path
without any architectural change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HostInitiationModel:
    """Constant per-operation delay between "ready" and "on the wire"."""

    initiation_delay_ps: int = 0
    #: label for provenance ("ideal", "gin", "cpu-proxy", ...)
    profile: str = "ideal"

    def delay_ps(self) -> int:
        return self.initiation_delay_ps
