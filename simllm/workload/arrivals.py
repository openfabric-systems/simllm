"""Request arrival processes.

Arrival times are produced in seconds on the virtual clock. Length
distributions and shared-prefix structure will live in sibling modules; an
arrival process only decides *when* requests enter the system.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


class PoissonArrivals:
    """Open-loop Poisson arrivals at ``rate_rps`` requests per second."""

    def __init__(self, rate_rps: float, seed: int = 0):
        if rate_rps <= 0:
            raise ValueError("rate_rps must be positive")
        self.rate_rps = rate_rps
        self._rng = np.random.default_rng(seed)

    def times(self, n: int) -> np.ndarray:
        """Return ``n`` absolute arrival times in seconds, starting at 0."""
        gaps = self._rng.exponential(1.0 / self.rate_rps, size=n)
        return np.cumsum(gaps)

    def __iter__(self) -> Iterator[float]:
        t = 0.0
        while True:
            t += float(self._rng.exponential(1.0 / self.rate_rps))
            yield t


class TraceArrivals:
    """Replay absolute arrival times (seconds) from a one-column text file."""

    def __init__(self, path: str | Path):
        self.times_s = np.loadtxt(Path(path), ndmin=1)
        if np.any(np.diff(self.times_s) < 0):
            raise ValueError("trace arrival times must be non-decreasing")

    def times(self, n: int | None = None) -> np.ndarray:
        return self.times_s if n is None else self.times_s[:n]
