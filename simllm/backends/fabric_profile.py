"""Measured fabric profiles and their rendering onto the DCQCN packet path.

A fabric profile is the carrier for one switching fabric's measured or
declared constants, separate from the endpoints attached to it and separate
from the transport that runs across it. :data:`HACC_LEAF_4X100G` holds the
measured HACC leaf: one non-blocking switch with four 100 G ports, a four-pipe
path at 515 ns per pipe, a 5.2 MB per-port tail-drop egress buffer, no ECN
marking at all, PFC off, pause emitted by the hosts and ignored by the switch,
and no path diversity.

Every model field carries one evidence class. The fabric campaign read nothing
off the switch itself, so `inferred` is what its constants are: endpoint
counters and per-packet logging bracket each one. `documented` is available for
a field read from a device that states it, and `declared` for a field that is
asserted rather than evidenced, which is what a sensitivity arm of a study
produces.

:func:`render_dcqcn` renders a NIC profile and a fabric profile together into
the topology file and the flag dict the RoCEv2 DCQCN comparator accepts. The
fabric owns the link rate, the switch buffers and the marking policy; the NIC
half comes from :func:`simllm.backends.nic_profile.dcqcn_flags`. The one thing
the flag vector cannot say honestly is "this switch does not mark", and
:func:`fabric_gap_fields` returns exactly that.

The module is deliberately framework-agnostic: it imports nothing from the
serving adapters and nothing from the simulator wrappers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from simllm.backends.nic_profile import NicProfile, dcqcn_flags

#: The evidence classes a fabric profile field may carry. `inferred` sits
#: between the other two: the constant is bracketed by measurement, but not on
#: the device it describes.
EVIDENCE_CLASSES = ("documented", "inferred", "declared")

#: Fields that describe the modeled fabric. `name`, `evidence` and
#: `provenance` describe the record itself and are excluded.
MODEL_FIELDS = (
    "switch_count",
    "host_ports",
    "port_bps",
    "pipe_latency_ps",
    "pipes_per_path",
    "egress_buffer_bytes",
    "ecn",
    "pfc_enabled",
    "pause_honoured",
    "ecmp_paths",
)

#: The value of `ecn` for a fabric that tail-drops without ever marking.
ECN_NONE = "none"

#: Keys a marking fabric's `ecn` mapping must carry, and only those.
ECN_KEYS = frozenset({"kmin_bytes", "kmax_bytes", "pmax_ppm"})

#: Fabric gap field to the registry task that owns closing it. A drop-only
#: switch has no flag: the runtime's configuration guard requires
#: `0 <= Kmin < Kmax < egress buffer` and a nonzero Pmax, so marking can be
#: parked out of reach but not switched off.
FABRIC_GAP_TASKS: Mapping[str, str] = MappingProxyType({"ecn": "HTSIM-38"})

_BPS_PER_GBPS = 1_000_000_000
_PS_PER_NS = 1_000
_PARTS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class FabricProfile:
    """One fabric's constants with an evidence class per field.

    Rates are bits per second, times picoseconds, sizes bytes.
    `egress_buffer_bytes` is the per-port tail-drop threshold;
    `pipes_per_path` counts the pipe traversals one message plus its
    acknowledgement pays end to end, which is what sets the latency floor.
    `ecn` is either :data:`ECN_NONE` or a mapping with exactly
    :data:`ECN_KEYS`. `pause_honoured` records whether the switch acts on a
    pause frame it receives, which is a separate question from whether
    link-level flow control is configured at all.
    """

    name: str
    switch_count: int
    host_ports: int
    port_bps: int
    pipe_latency_ps: int
    pipes_per_path: int
    egress_buffer_bytes: int
    ecn: str | Mapping[str, int]
    pfc_enabled: bool
    pause_honoured: bool
    ecmp_paths: int
    evidence: Mapping[str, str]
    provenance: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FabricProfile.name must be nonempty")
        if not self.provenance:
            raise ValueError(f"{self.name}: provenance must be nonempty")
        positive = (
            "switch_count",
            "host_ports",
            "port_bps",
            "pipe_latency_ps",
            "pipes_per_path",
            "egress_buffer_bytes",
            "ecmp_paths",
        )
        for field_name in positive:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{self.name}: {field_name} must be a positive integer")
        if self.pfc_enabled and not self.pause_honoured:
            raise ValueError(
                f"{self.name}: pfc_enabled with pause_honoured false is not a fabric; "
                "a switch that runs PFC and ignores pause has no model"
            )
        object.__setattr__(self, "ecn", self._validated_ecn())
        missing = [name for name in MODEL_FIELDS if name not in self.evidence]
        if missing:
            raise ValueError(f"{self.name}: no evidence class for {sorted(missing)}")
        extra = [name for name in self.evidence if name not in MODEL_FIELDS]
        if extra:
            raise ValueError(f"{self.name}: evidence names non-model fields {sorted(extra)}")
        bad = {
            name: value
            for name, value in self.evidence.items()
            if value not in EVIDENCE_CLASSES
        }
        if bad:
            raise ValueError(f"{self.name}: unknown evidence classes {bad}")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def _validated_ecn(self) -> str | Mapping[str, int]:
        if self.ecn == ECN_NONE:
            return ECN_NONE
        if not isinstance(self.ecn, Mapping) or set(self.ecn) != ECN_KEYS:
            raise ValueError(
                f"{self.name}: ecn must be {ECN_NONE!r} or a mapping with exactly the keys "
                f"{sorted(ECN_KEYS)}, got {self.ecn!r}"
            )
        for key in sorted(ECN_KEYS):
            value = self.ecn[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{self.name}: ecn[{key!r}] must be a nonnegative integer")
        if self.ecn["kmin_bytes"] >= self.ecn["kmax_bytes"]:
            raise ValueError(f"{self.name}: ecn kmin_bytes must be below kmax_bytes")
        if self.ecn["kmax_bytes"] >= self.egress_buffer_bytes:
            raise ValueError(
                f"{self.name}: ecn kmax_bytes must be below egress_buffer_bytes; the "
                "runtime refuses a marking threshold at or above the buffer"
            )
        if not 0 < self.ecn["pmax_ppm"] <= _PARTS_PER_MILLION:
            raise ValueError(f"{self.name}: ecn pmax_ppm must be in 1 to {_PARTS_PER_MILLION}")
        return MappingProxyType(dict(self.ecn))

    @property
    def marks_ecn(self) -> bool:
        """Whether the fabric marks at all."""
        return self.ecn != ECN_NONE

    @property
    def latency_floor_ps(self) -> int:
        """Propagation part of the smallest message's round trip."""
        return self.pipes_per_path * self.pipe_latency_ps

    @property
    def shared_buffer_bytes(self) -> int:
        """Switch-wide pool, sized so the per-port limit is the binding one.

        No campaign has read a shared pool off a switch, so this is the sum of
        the measured per-port pools rather than a measurement. A study checks
        that the shared-pool drop counter stays at zero, which is what makes
        the choice safe rather than merely conservative.
        """
        return self.host_ports * self.egress_buffer_bytes


def fabric_gap_fields(fabric: FabricProfile) -> tuple[str, ...]:
    """Model fields of `fabric` that no comparator flag can carry.

    Only one field can be in this ledger. A drop-only fabric is inexpressible:
    the runtime requires `0 <= Kmin < Kmax < egress buffer` and a nonzero
    Pmax, so :func:`render_dcqcn` parks the marking band in the last two bytes
    of the buffer at one part per million and a study has to check the
    marked-packet counter rather than rely on the configuration. A marking
    fabric has no gap, because its three thresholds render directly.
    """
    return () if fabric.marks_ecn else ("ecn",)


def dcqcn_port_gbps(fabric: FabricProfile) -> int:
    """The whole-Gb/s port rate the topology file and `-link_bps` share.

    The fat-tree loader parses `Downlink_speed_Gbps` as a whole number and the
    runtime rejects a topology whose rate differs from `-link_bps`, so a
    fabric whose port rate is not a whole Gb/s cannot be rendered at all
    rather than being silently rounded.
    """
    if fabric.port_bps % _BPS_PER_GBPS:
        raise ValueError(
            f"{fabric.name}: port_bps {fabric.port_bps} is not a whole Gb/s, which the "
            "topology loader cannot express"
        )
    return fabric.port_bps // _BPS_PER_GBPS


def render_topology(fabric: FabricProfile) -> str:
    """Render `fabric` as the ns-tm3 Clos topology file the runtime loads.

    The runtime accepts exactly two tiers, so a single-switch fabric is
    expressed as the degenerate Clos whose whole node set hangs off one leaf:
    pod size equals the host count, the leaf's down and up radices are equal
    (oversubscription 1), and each spine takes one uplink and carries no
    traffic. Every host pair is then same-leaf, which is the measured one-hop
    path, and the geometry also fixes `ecmp_paths` at 1.
    """
    if fabric.switch_count != 1:
        raise ValueError(
            f"{fabric.name}: only a single-switch fabric renders onto the two-tier "
            f"Clos the runtime builds, got switch_count {fabric.switch_count}"
        )
    if fabric.ecmp_paths != 1:
        raise ValueError(
            f"{fabric.name}: a one-leaf geometry has one path, so ecmp_paths must be 1, "
            f"got {fabric.ecmp_paths}"
        )
    gbps = dcqcn_port_gbps(fabric)
    if fabric.pipe_latency_ps % _PS_PER_NS:
        raise ValueError(
            f"{fabric.name}: pipe_latency_ps {fabric.pipe_latency_ps} is not a whole "
            "nanosecond, which the topology loader cannot express"
        )
    latency_ns = fabric.pipe_latency_ps // _PS_PER_NS
    hosts = fabric.host_ports
    return (
        f"Nodes {hosts}\n"
        "Tiers 2\n"
        f"Podsize {hosts}\n"
        "\n"
        f"# Generated from the {fabric.name} fabric profile. One leaf switch with\n"
        f"# {hosts} directly attached hosts and one uplink to each of {hosts} spine\n"
        "# switches. Every host pair is same-leaf, so the spines carry no traffic:\n"
        "# the measured fabric is a single switch hop and a same-leaf pair\n"
        "# reproduces it exactly.\n"
        "#\n"
        f"# {latency_ns} ns per pipe over {fabric.pipes_per_path} pipe traversals puts the\n"
        "# smallest message's round trip at the measured latency floor once the\n"
        "# store-and-forward serializations are added.\n"
        "Tier 0\n"
        f"Downlink_speed_Gbps {gbps}\n"
        f"Radix_Down {hosts}\n"
        f"Radix_Up {hosts}\n"
        f"Downlink_Latency_ns {latency_ns}\n"
        "Switch_Latency_ns 0\n"
        "\n"
        f"# {hosts} spine switches, each with one link to the leaf.\n"
        "Tier 1\n"
        f"Downlink_speed_Gbps {gbps}\n"
        "Radix_Down 1\n"
        f"Downlink_Latency_ns {latency_ns}\n"
        "Switch_Latency_ns 0\n"
    )


def _no_marking_thresholds(fabric: FabricProfile) -> dict[str, str]:
    # The runtime has no "marking off" switch. Its configuration guard refuses
    # a zero Pmax and refuses Kmax at or above the egress buffer, and its RED
    # predicate returns false whenever the egress occupancy is at or below
    # Kmin. So the only legal spelling of a drop-only fabric is the highest
    # legal threshold pair with the lowest legal probability: the marking band
    # becomes the last two bytes before the tail-drop limit, at one part per
    # million. That is not zero by construction, so a study using this
    # rendering must check the marked-packet counter. FABRIC_GAP_TASKS points
    # at the task that closes it.
    buffer_bytes = fabric.egress_buffer_bytes
    if buffer_bytes < 3:
        raise ValueError(
            f"{fabric.name}: egress_buffer_bytes {buffer_bytes} leaves no room for the "
            "no-marking threshold pair"
        )
    return {
        "-ecn_kmin_bytes": str(buffer_bytes - 2),
        "-ecn_kmax_bytes": str(buffer_bytes - 1),
        "-ecn_pmax_ppm": "1",
    }


def render_dcqcn(nic: NicProfile, fabric: FabricProfile) -> tuple[str, dict[str, str]]:
    """Render one NIC on one fabric into a topology file and comparator flags.

    Returns the topology file's text and the flag dict, which together are the
    whole configuration except for the GOAL binary, the completion CSV and the
    seeds a study owns.

    The split of authority is the point. The NIC half comes from
    :func:`simllm.backends.nic_profile.dcqcn_flags` and covers packetization,
    recovery, the loss response and the timeout. The fabric then overrides the
    rate, the buffers and the marking policy, because those are properties of
    the switch and the link rather than of the endpoint. In particular
    `-link_bps` renders the fabric's port rate, not the NIC's goodput
    asymptote: the comparator has one rate, and the asymptote is exactly the
    `link_bps` gap field the NIC profile already registers.

    A NIC whose PFC state disagrees with the fabric's is refused rather than
    silently resolved, because `-pfc` is one flag for both ends.
    """
    if nic.pfc_enabled != fabric.pfc_enabled:
        raise ValueError(
            f"{nic.name} on {fabric.name}: the comparator has one -pfc flag for both "
            f"ends, and the NIC says {nic.pfc_enabled} while the fabric says "
            f"{fabric.pfc_enabled}"
        )
    flags = dcqcn_flags(nic)
    flags["-link_bps"] = str(fabric.port_bps)
    flags["-egress_buffer_bytes"] = str(fabric.egress_buffer_bytes)
    flags["-shared_buffer_bytes"] = str(fabric.shared_buffer_bytes)
    if fabric.marks_ecn:
        flags["-ecn_kmin_bytes"] = str(fabric.ecn["kmin_bytes"])
        flags["-ecn_kmax_bytes"] = str(fabric.ecn["kmax_bytes"])
        flags["-ecn_pmax_ppm"] = str(fabric.ecn["pmax_ppm"])
    else:
        flags.update(_no_marking_thresholds(fabric))
    return render_topology(fabric), flags


_HACC_PROVENANCE = (
    "HACC leaf fabric measured 2026-09-01 and 2026-09-02 by endpoint probing from four "
    "directly attached ConnectX-5 Ex 100 GbE hosts, RoCEv2 GID 3, active MTU 4096. "
    "Campaign records RESULTS-p6-fabric.md, its freeze expectations-p6-fabric.md and "
    "FINDINGS-cx5.md section D, with raw tables under data/p6 (buffer.csv, ecn_fit.txt, "
    "dcqcn_summary.csv, loneflow.csv, udcap.csv, latency_matrix.csv, uplink.csv). "
    "Evidence class throughout: inferred. No switch counter was readable from the "
    "endpoints, so endpoint counters and per-packet logging bracket every constant, and "
    "the cluster-wide Clos beyond this leaf is out of reach and undescribed."
)

#: The measured HACC leaf: one non-blocking switch, four 100 G ports.
HACC_LEAF_4X100G = FabricProfile(
    name="hacc_leaf_4x100g",
    # All six host pairs fall in one latency class, range 0.13 us over 12
    # directions, so no pair of these hosts crosses a second switch.
    switch_count=1,
    # The four hosts that were measured. A directed ring of four ran
    # 391.94 Gb/s aggregate with every port at 97.98, so the leaf is
    # non-blocking across them.
    host_ports=4,
    # 100000 Mb/s on every port; no uplink is reachable from these hosts.
    port_bps=100_000_000_000,
    # The 2.08 us 2 B WRITE floor over four pipe traversals. Derived from the
    # route the comparator builds and confirmed against the measurement.
    pipe_latency_ps=515_000,
    # Host queue, host to leaf, leaf ingress, leaf to host, and the mirror of
    # that for the acknowledgement.
    pipes_per_path=4,
    # Per-port tail drop. t_drop times excess gives 5.39, 5.04 and 5.05 MB at
    # excess 4.76, 9.74 and 19.68 Gb/s, spread 6.8 percent over 12 runs; the
    # independent drain-tail estimate is 5.76 MB.
    egress_buffer_bytes=5_200_000,
    # Zero CE-marked packets in 670 M, at two DSCP classes, with every packet
    # ECT(0) and the buffer full and dropping. Kmin, Kmax and Pmax are
    # undefined rather than small.
    ecn=ECN_NONE,
    pfc_enabled=False,
    # The hosts emit 802.3x pause under load, about 760 frames per run, and
    # the switch has never paused a host over the whole node lifetime.
    pause_honoured=False,
    # 16 fresh 5-tuples unimodal, range 0.10 us: no path diversity is
    # reachable from these hosts.
    ecmp_paths=1,
    evidence={field_name: "inferred" for field_name in MODEL_FIELDS},
    provenance=_HACC_PROVENANCE,
)

#: Profiles by name, for study command lines.
FABRICS: Mapping[str, FabricProfile] = MappingProxyType(
    {profile.name: profile for profile in (HACC_LEAF_4X100G,)}
)
