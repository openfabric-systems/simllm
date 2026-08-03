import numpy as np
import pytest

from simllm.workload import PoissonArrivals, TraceArrivals


def test_poisson_reproducible():
    a = PoissonArrivals(rate_rps=100.0, seed=42).times(1000)
    b = PoissonArrivals(rate_rps=100.0, seed=42).times(1000)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)
    # mean inter-arrival ~ 10 ms
    assert 0.008 < np.diff(a).mean() < 0.012


def test_poisson_rejects_bad_rate():
    with pytest.raises(ValueError):
        PoissonArrivals(rate_rps=0.0)


def test_trace_arrivals(tmp_path):
    p = tmp_path / "arrivals.txt"
    p.write_text("0.0\n0.5\n1.5\n")
    t = TraceArrivals(p)
    assert t.times().tolist() == [0.0, 0.5, 1.5]
    assert t.times(2).tolist() == [0.0, 0.5]
