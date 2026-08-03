from simllm.goal import GoalTrace


def test_two_rank_ping():
    trace = GoalTrace(2)
    r0 = trace.rank(0)
    c = r0.calc(5000)
    s = r0.send(8192, to=1, tag=7)
    r0.requires(s, c)
    trace.rank(1).recv(8192, source=0, tag=7)

    text = trace.render()
    assert text.splitlines()[0] == "num_ranks 2"
    assert "r0op0: calc 5000" in text
    assert "r0op1: send 8192b to 1 tag 7" in text
    assert "r0op1 requires r0op0" in text
    assert "r1op0: recv 8192b from 0 tag 7" in text
    assert text.count("rank ") == 2


def test_write(tmp_path):
    trace = GoalTrace(1)
    trace.rank(0).calc(1)
    out = trace.write(tmp_path / "t.goal")
    assert out.read_text().startswith("num_ranks 1")
