import pytest

from simllm.backends import FlowCompletion, normalized_fct


def flow(profile, src, dst, tag, fct, start=0, size=65536):
    return FlowCompletion(profile, 0, src, dst, tag, size, start, start + fct, fct)


def test_normalized_fct_matches_by_key():
    cn = [flow("rnic-cn", 0, 1, 1, 200), flow("rnic-cn", 0, 2, 1, 300)]
    nn = [flow("rnic-nn", 0, 2, 1, 150), flow("rnic-nn", 0, 1, 1, 100)]
    norm = normalized_fct(cn, nn)
    by_dst = {n.destination: n.slowdown for n in norm}
    assert by_dst == {1: 2.0, 2: 2.0}


def test_normalized_fct_rejects_mismatched_flow_sets():
    with pytest.raises(ValueError, match="flow sets differ"):
        normalized_fct([flow("rnic-cn", 0, 1, 1, 200)], [flow("rnic-nn", 0, 9, 1, 100)])


def test_normalized_fct_rejects_payload_drift():
    with pytest.raises(ValueError, match="payload differs"):
        normalized_fct(
            [flow("rnic-cn", 0, 1, 1, 200, size=100)],
            [flow("rnic-nn", 0, 1, 1, 100, size=999)],
        )
