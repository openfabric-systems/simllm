from pathlib import Path

import pytest

from simllm.backends import (
    FlowCompletion,
    HtsimRnicConfig,
    build_htsim_rnic_command,
    find_htsim_rnic,
    parse_completion_csv,
    run_htsim_rnic,
)
from simllm.goal import GoalTrace, find_txt2bin, to_binary
from simllm.traffic import gather, scatter

CSV_SAMPLE = """profile,flow_id,source,destination,tag,payload_bytes,start_time_ps,completion_time_ps,fct_ps
rnic-cn,21474836482,5,6,1,65536,178513600,210652800,32139200
rnic-cn,25769803778,6,7,1,65536,215652800,243539200,27886400
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
