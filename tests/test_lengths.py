import numpy as np
import pytest

from simllm.workload import FixedLengths, LogNormalLengths, TraceLengths


def test_fixed_lengths():
    assert FixedLengths(128).sample(3).tolist() == [128, 128, 128]
    with pytest.raises(ValueError):
        FixedLengths(0)


def test_lognormal_mean_and_reproducibility():
    a = LogNormalLengths(mean=1000.0, sigma=1.0, seed=7).sample(20000)
    b = LogNormalLengths(mean=1000.0, sigma=1.0, seed=7).sample(20000)
    assert np.array_equal(a, b)
    assert np.all(a >= 1)
    # parameterized by the arithmetic mean, so the sample mean must sit near it
    assert 0.95 < a.mean() / 1000.0 < 1.05


def test_trace_lengths_cycle(tmp_path):
    p = tmp_path / "lengths.txt"
    p.write_text("10\n20\n30\n")
    t = TraceLengths(p)
    assert t.sample(5).tolist() == [10, 20, 30, 10, 20]
    bad = tmp_path / "bad.txt"
    bad.write_text("10\n-5\n")
    with pytest.raises(ValueError):
        TraceLengths(bad)
