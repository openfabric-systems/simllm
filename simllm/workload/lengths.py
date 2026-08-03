"""Request length distributions.

Lengths are token counts (or byte counts, the sampler does not care); the
caller maps them onto compute cost and transfer sizes. All samplers are
reproducible by seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FixedLengths:
    """Every request has the same length."""

    def __init__(self, length: int):
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length

    def sample(self, n: int) -> np.ndarray:
        return np.full(n, self.length, dtype=np.int64)


class LogNormalLengths:
    """Heavy-tailed lengths, parameterized by their actual mean and sigma."""

    def __init__(self, mean: float, sigma: float, seed: int = 0, minimum: int = 1):
        if mean <= 0 or sigma <= 0:
            raise ValueError("mean and sigma must be positive")
        # numpy parameterizes by the underlying normal; solve mu so that the
        # sampled distribution has the requested arithmetic mean.
        self._mu = np.log(mean) - 0.5 * sigma**2
        self._sigma = sigma
        self._minimum = minimum
        self._rng = np.random.default_rng(seed)

    def sample(self, n: int) -> np.ndarray:
        raw = self._rng.lognormal(self._mu, self._sigma, size=n)
        return np.maximum(raw.astype(np.int64), self._minimum)


class TraceLengths:
    """Replay lengths from a one-column text file, cycling if exhausted."""

    def __init__(self, path: str | Path):
        self.lengths = np.loadtxt(Path(path), dtype=np.int64, ndmin=1)
        if np.any(self.lengths <= 0):
            raise ValueError("trace lengths must be positive")

    def sample(self, n: int) -> np.ndarray:
        reps = -(-n // len(self.lengths))
        return np.tile(self.lengths, reps)[:n]
