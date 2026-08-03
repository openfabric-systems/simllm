import pytest

from simllm.core import VirtualClock


def test_orders_events_and_advances():
    clock = VirtualClock()
    clock.schedule(300, "c")
    clock.schedule(100, "a")
    clock.schedule(200, "b")
    assert clock.peek_next_time() == 100
    assert clock.pop_next() == (100, "a")
    assert clock.pop_next() == (200, "b")
    assert clock.now_ps == 200
    assert len(clock) == 1


def test_ties_keep_insertion_order():
    clock = VirtualClock()
    clock.schedule(50, "first")
    clock.schedule(50, "second")
    assert clock.pop_next()[1] == "first"
    assert clock.pop_next()[1] == "second"


def test_rejects_time_travel():
    clock = VirtualClock(start_ps=1000)
    with pytest.raises(ValueError):
        clock.schedule(999, "late")
    clock.advance_to(2000)
    with pytest.raises(ValueError):
        clock.advance_to(1500)
    with pytest.raises(IndexError):
        clock.pop_next()
