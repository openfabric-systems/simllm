import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTATIONS = (
    ROOT / "examples" / "disaggregated_target_topology_v1" / "expectations.json"
)


def _expectations():
    return json.loads(EXPECTATIONS.read_text(encoding="utf-8"))


def test_place5_freeze_pins_declared_reference_constants():
    frozen = _expectations()
    topology = frozen["topology"]

    assert frozen["schema"] == (
        "simllm-disaggregated-target-topology-expectations-v1"
    )
    assert topology == {
        "endpoint_links_per_leaf": 8,
        "evidence_class": "declared",
        "goal_rank_mapping": "gpu-rank",
        "leaf_spine_fanout": 8,
        "link_propagation_delay_ps": 1_000_000,
        "link_rate_bps": 400_000_000_000,
        "name": "simllm-disaggregated-448-clos-declared-v1",
        "spine_switches": 8,
        "switch_latency_ps": 0,
        "tiers": 2,
    }


def test_place5_freeze_conserves_both_declared_scales():
    frozen = _expectations()
    cells = {cell["label"]: cell for cell in frozen["cells"]}

    assert set(cells) == {"one_plus_one", "target"}
    for cell in cells.values():
        ranks = cell["rank_count"]
        leaves = cell["leaf_switches"]
        spines = cell["spine_switches"]
        assert ranks == cell["gpu_count"] == cell["nic_count"]
        assert ranks == cell["prefill_ranks"] + cell["decode_ranks"]
        assert ranks == cell["goal_messages"] == cell["endpoint_links"]
        assert leaves == ranks // 8
        assert cell["leaf_spine_links"] == leaves * spines
        assert cell["link_count"] == ranks + leaves * spines
        assert cell["switch_count"] == leaves + spines
        assert cell["switch_port_count"] == 2 * cell["link_count"] - ranks

    assert cells["target"]["rank_count"] == 28 * cells["one_plus_one"][
        "rank_count"
    ]
    assert cells["target"]["leaf_switches"] == 28 * cells["one_plus_one"][
        "leaf_switches"
    ]


def test_place5_freeze_pins_goal_path_and_physical_bounds():
    frozen = _expectations()
    witness = frozen["goal_witness"]
    sanity = frozen["physical_sanity"]

    serialization_ps = (
        witness["payload_bytes"]
        * 8
        * 10**12
        // frozen["topology"]["link_rate_bps"]
    )
    propagation_ps = (
        witness["expected_path_links"]
        * frozen["topology"]["link_propagation_delay_ps"]
    )
    assert serialization_ps == witness["serialization_ps_per_link"]
    assert propagation_ps == witness["expected_path_propagation_ps"]
    assert sanity["cross_leaf_cut_through_floor_ps"] == (
        propagation_ps + serialization_ps
    )
    assert sanity["cross_leaf_store_and_forward_ceiling_ps"] == (
        propagation_ps + witness["expected_path_links"] * serialization_ps
    )


def test_place5_freeze_pins_disabled_placement_identity_and_closure_scope():
    frozen = _expectations()
    records = frozen["baseline"]["placement_records"]

    assert records == {
        "one_plus_one": {
            "bytes": 15_772,
            "sha256": (
                "019f818e02252407e560b37415da12151c8f6ca0ff01bcb7ff8aabfead47f286"
            ),
        },
        "target": {
            "bytes": 2_639_042,
            "sha256": (
                "48029d871293762007ab33082d59a7b5a4efb22583394e718c97e733717fd709"
            ),
        },
    }
    assert frozen["closure"]["closes_if_literal"] == ["PLACE-5"]
    assert frozen["closure"]["does_not_close"] == [
        "PLACE-1",
        "PLACE-2",
        "TRAF-62",
        "TRAF-64",
    ]
    assert frozen["evidence"]["scored_behavioral_families"] == 0
    assert frozen["evidence"]["violation_status"] == "VOID"
