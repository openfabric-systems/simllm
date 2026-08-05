from pathlib import Path

import pytest

from simllm.backends import (
    FlowCompletion,
    HtsimRnicConfig,
    RnicRunResult,
    build_htsim_rnic_command,
    find_htsim_rnic,
    parse_completion_csv,
    run_htsim_rnic,
)
from simllm.backends.htsim_rnic import _parse_goal_completion_time_ps
from simllm.goal import GoalTrace, find_txt2bin, to_binary
from simllm.traffic import gather, scatter

CSV_SAMPLE = """profile,flow_id,source,destination,tag,payload_bytes,start_time_ps,completion_time_ps,fct_ps
rnic-cn,21474836482,5,6,1,65536,178513600,210652800,32139200
rnic-cn,25769803778,6,7,1,65536,215652800,243539200,27886400
"""

WQE_CSV_SAMPLE = """profile,flow_id,source,destination,tag,payload_bytes,start_time_ps,completion_time_ps,fct_ps,wqe_id,sq_id,rq_id,cq_id,sq_post_sequence,sq_dispatch_sequence,cq_post_sequence,cq_consume_sequence,transport_kind,transport_object_id
rnic-cn,7,1,2,9,4096,100,300,200,7,5,10,7,3,3,2,2,rnic-cn-link-pair,4294967298
"""


def test_command_builder_rejects_unknown_profile():
    with pytest.raises(ValueError):
        HtsimRnicConfig(goal_bin=Path("x.bin"), profile="rnic-ss", linkspeed_bps=1)


def test_command_builder_layout():
    cfg = HtsimRnicConfig(
        goal_bin=Path("t.bin"), profile="rnic-cn", linkspeed_bps=400_000_000_000,
        completion_csv=Path("out.csv"), extra_flags={"-rnic_cn_margin_ppm": "900000"},
    )
    argv = build_htsim_rnic_command(Path("htsim_rnic"), cfg)
    assert argv[argv.index("-rnic_profile") + 1] == "rnic-cn"
    assert argv[argv.index("-linkspeed_bps") + 1] == "400000000000"
    assert argv[argv.index("-rnic_cn_margin_ppm") + 1] == "900000"


def test_parse_completion_csv(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(CSV_SAMPLE)
    flows = parse_completion_csv(path)
    assert len(flows) == 2
    assert flows[0] == FlowCompletion(
        "rnic-cn", 21474836482, 5, 6, 1, 65536, 178513600, 210652800, 32139200
    )
    assert flows[1].fct_ps == flows[1].completion_time_ps - flows[1].start_time_ps


def test_parse_completion_csv_preserves_appended_wqe_fields(tmp_path):
    path = tmp_path / "wqe.csv"
    path.write_text(WQE_CSV_SAMPLE)
    flow = parse_completion_csv(path)[0]
    assert flow.wqe_id == flow.flow_id == 7
    assert (flow.sq_id, flow.rq_id, flow.cq_id) == (5, 10, 7)
    assert (flow.sq_post_sequence, flow.sq_dispatch_sequence) == (3, 3)
    assert (flow.cq_post_sequence, flow.cq_consume_sequence) == (2, 2)
    assert flow.transport_kind == "rnic-cn-link-pair"
    assert flow.transport_object_id == 4_294_967_298


def test_compute_only_goal_uses_the_driver_completion_summary():
    stdout = "noise\nMaximum finishing time at host 0: 24060 (2.406e-05 s)\n"
    completion_ps = _parse_goal_completion_time_ps(stdout)
    assert completion_ps == 24_060_000
    assert RnicRunResult([], [], True, completion_ps).job_completion_time_ps() == completion_ps


def test_flow_followed_by_trailing_compute_uses_schedule_completion():
    flow = FlowCompletion(
        "rnic-nn",
        1,
        0,
        1,
        7,
        4096,
        0,
        10_000,
        10_000,
    )

    assert RnicRunResult([flow], [], True, 15_000).job_completion_time_ps() == 15_000
    assert RnicRunResult([flow], [], True, 9_000).job_completion_time_ps() == 10_000


@pytest.mark.skipif(
    find_txt2bin() is None or find_htsim_rnic() is None,
    reason="backend toolchain not available (submodule not built)",
)
def test_end_to_end_scatter_gather(tmp_path):
    """Emit a 4-rank scatter/compute/gather GOAL and run it on rnic-nn-fluid."""
    trace = GoalTrace(4)
    sdone = scatter(trace, root=0, workers=[1, 2, 3], size_bytes=65536, tag=1)
    cdone = {}
    for w in [1, 2, 3]:
        c = trace.rank(w).calc(5000)
        trace.rank(w).requires(c, sdone[w])
        cdone[w] = c
    gather(trace, root=0, workers=[1, 2, 3], size_bytes=65536, tag=2, after=cdone)

    goal_bin = to_binary(trace.write(tmp_path / "sg.goal"))
    result = run_htsim_rnic(HtsimRnicConfig(
        goal_bin=goal_bin, profile="rnic-nn-fluid",
        linkspeed_bps=400_000_000_000, completion_csv=tmp_path / "sg.csv",
    ))
    assert result.quiescent
    assert len(result.flows) == 6
    assert result.job_completion_time_ps() > 0
    # gather flows complete after their scatter counterparts
    by_tag = {tag: max(f.completion_time_ps for f in result.flows if f.tag == tag)
              for tag in (1, 2)}
    assert by_tag[2] > by_tag[1]
