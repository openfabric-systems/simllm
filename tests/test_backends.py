from pathlib import Path

from simllm.backends import HtsimUecConfig, build_htsim_uec_command


def test_htsim_uec_command():
    cfg = HtsimUecConfig(
        goal_bin=Path("trace.bin"),
        topology=Path("leaf_spine_8_1os.topo"),
        nodes=8,
        linkspeed_mbps=400_000,
    )
    argv = build_htsim_uec_command(Path("htsim_uec"), cfg)
    assert argv[0] == "htsim_uec"
    assert "-goal" in argv and argv[argv.index("-goal") + 1] == "trace.bin"
    assert argv[argv.index("-nodes") + 1] == "8"
    assert argv[argv.index("-linkspeed") + 1] == "400000"
    assert "-lgs_flow_stats" in argv
