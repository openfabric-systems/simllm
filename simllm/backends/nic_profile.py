"""Measured RNIC hardware profiles and their rendering into backend flags.

A profile is the carrier for one NIC's measured or declared hardware
constants, separate from the transport and congestion-control policy that
runs on top of it. `CX5_100G` holds the ConnectX-5 Ex 100 GbE campaign
values; `CX7_400G` is the same architecture at four times the rate, derived
from `CX5_100G` by :func:`scale_profile` alone.

Every model field carries one evidence class, per the truth and observability
contract in `docs/papers/rnic-hardware-calibration.md`: `documented`,
`driver-inferred` or `calibrated-opaque` for a field with evidence on the
device it describes, and `declared` for a field that is asserted rather than
evidenced. Every field of a scaled profile is `declared`, because no
ConnectX-7 silicon was measured: scaling is an architectural assertion, not a
measurement, and the distinction between the fields the factor multiplied and
the fields it carried across is kept in :data:`SCALED_FIELDS` and in the
profile's provenance string rather than smuggled into the evidence class.

:func:`dcqcn_flags` renders the subset the RoCEv2 DCQCN comparator's command
line can express. :func:`gap_fields` names the rest, which is exactly the
model's gap ledger: each of those fields is a hardware property the packet
path has no place to put, and :data:`GAP_TASKS` maps each one to the registry
task in `docs/modules/backends.md` that owns closing it.

The module is deliberately framework-agnostic: it imports nothing from the
serving adapters and nothing from the simulator wrappers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

#: The evidence classes a profile field may carry.
EVIDENCE_CLASSES = ("documented", "driver-inferred", "calibrated-opaque", "declared")

#: The loss-recovery modes the backend transport implements.
RECOVERY_MODES = ("gbn", "sr")

#: Fields that describe the modeled hardware. `name`, `evidence` and
#: `provenance` describe the record itself and are excluded.
MODEL_FIELDS = (
    "link_bps",
    "goodput_bps",
    "mtu_bytes",
    "header_bytes",
    "t_eff_ps",
    "sq_depth",
    "pps_ceiling_per_qp",
    "pps_ceiling_per_nic",
    "rx_ingress_meter_bytes",
    "pfc_enabled",
    "recovery",
    "loss_rate_cut",
    "rto_ps",
    "ecn_kmin_bytes",
    "ecn_kmax_bytes",
    "ecn_pmax_ppm",
)

#: Fields :func:`scale_profile` multiplies by the link factor. Everything else
#: in :data:`MODEL_FIELDS` is carried across unchanged, because it is either a
#: rate-independent time, a packetization constant or a mode.
SCALED_FIELDS = (
    "link_bps",
    "goodput_bps",
    "pps_ceiling_per_qp",
    "pps_ceiling_per_nic",
    "ecn_kmin_bytes",
    "ecn_kmax_bytes",
)

#: Fields the DCQCN comparator's command line can carry.
_CLI_BACKED_FIELDS = frozenset(
    {
        "goodput_bps",
        "mtu_bytes",
        "header_bytes",
        "pfc_enabled",
        "recovery",
        "loss_rate_cut",
        "rto_ps",
        "ecn_kmin_bytes",
        "ecn_kmax_bytes",
        "ecn_pmax_ppm",
    }
)

#: Gap field to the registry task that owns closing it.
GAP_TASKS: Mapping[str, str] = MappingProxyType(
    {
        "link_bps": "BACK-54",
        "t_eff_ps": "BACK-54",
        "sq_depth": "HTSIM-34",
        "pps_ceiling_per_qp": "HTSIM-36",
        "pps_ceiling_per_nic": "HTSIM-36",
        "rx_ingress_meter_bytes": "HTSIM-35",
    }
)

_BPS_PER_GBPS = 1_000_000_000
_PS_PER_US = 1_000_000


@dataclass(frozen=True)
class NicProfile:
    """One NIC's hardware constants with an evidence class per field.

    Rates are bits per second, times picoseconds, sizes bytes. `goodput_bps`
    is the payload asymptote a saturated flow reaches, which is below
    `link_bps` by the wire framing the device actually pays;
    `rx_ingress_meter_bytes` is the responder-side ingress pool, `None` when
    the device does not expose one.
    """

    name: str
    link_bps: int
    goodput_bps: int
    mtu_bytes: int
    header_bytes: int
    t_eff_ps: int
    sq_depth: int
    pps_ceiling_per_qp: int
    pps_ceiling_per_nic: int
    rx_ingress_meter_bytes: int | None
    pfc_enabled: bool
    recovery: str
    loss_rate_cut: bool
    rto_ps: int
    ecn_kmin_bytes: int
    ecn_kmax_bytes: int
    ecn_pmax_ppm: int
    evidence: Mapping[str, str]
    provenance: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NicProfile.name must be nonempty")
        if not self.provenance:
            raise ValueError(f"{self.name}: provenance must be nonempty")
        if self.recovery not in RECOVERY_MODES:
            raise ValueError(
                f"{self.name}: recovery must be one of {RECOVERY_MODES}, got {self.recovery!r}"
            )
        positive = (
            "link_bps",
            "goodput_bps",
            "mtu_bytes",
            "header_bytes",
            "t_eff_ps",
            "sq_depth",
            "pps_ceiling_per_qp",
            "pps_ceiling_per_nic",
            "rto_ps",
            "ecn_kmin_bytes",
            "ecn_kmax_bytes",
            "ecn_pmax_ppm",
        )
        for field_name in positive:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{self.name}: {field_name} must be a positive integer")
        if self.rx_ingress_meter_bytes is not None and self.rx_ingress_meter_bytes <= 0:
            raise ValueError(f"{self.name}: rx_ingress_meter_bytes must be positive or None")
        if self.header_bytes >= self.mtu_bytes:
            raise ValueError(f"{self.name}: header_bytes must be smaller than mtu_bytes")
        if self.goodput_bps > self.link_bps:
            raise ValueError(f"{self.name}: goodput_bps must not exceed link_bps")
        if self.ecn_kmin_bytes > self.ecn_kmax_bytes:
            raise ValueError(f"{self.name}: ecn_kmin_bytes must not exceed ecn_kmax_bytes")
        if self.ecn_pmax_ppm > 1_000_000:
            raise ValueError(f"{self.name}: ecn_pmax_ppm must not exceed 1000000")
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

    @property
    def mss_bytes(self) -> int:
        """Payload bytes per full wire packet."""
        return self.mtu_bytes - self.header_bytes


def _scaled_int(value: int, factor: float, name: str, field_name: str) -> int:
    scaled = value * factor
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-6:
        raise ValueError(
            f"{name}: scaling {field_name} by {factor} does not give a whole number"
        )
    return int(rounded)


def scale_profile(base: NicProfile, link_factor: float, name: str | None = None) -> NicProfile:
    """Return `base` at `link_factor` times its rate, every field `declared`.

    The factor multiplies the rate-carrying fields and the ECN byte
    thresholds (:data:`SCALED_FIELDS`); the message offset, packetization,
    queue depth, recovery mode, timeout and ingress pool are carried across
    unchanged. Scaling asserts that the two devices are the same architecture
    at different rates, which is a claim about the target device rather than a
    measurement of it, so the result carries `declared` on every model field.
    """
    if link_factor <= 0:
        raise ValueError("link_factor must be positive")
    scaled = {
        field_name: _scaled_int(getattr(base, field_name), link_factor, base.name, field_name)
        for field_name in SCALED_FIELDS
    }
    carried = tuple(
        field_name for field_name in MODEL_FIELDS if field_name not in SCALED_FIELDS
    )
    values = {field.name: getattr(base, field.name) for field in fields(base)}
    values.update(scaled)
    values["name"] = name or f"{base.name}_x{link_factor:g}"
    values["evidence"] = {field_name: "declared" for field_name in MODEL_FIELDS}
    values["provenance"] = (
        f"declared: {base.name} scaled by link factor {link_factor:g}; no silicon of this "
        f"device was measured. Scaled: {', '.join(SCALED_FIELDS)}. "
        f"Carried unchanged: {', '.join(carried)}. Base provenance: {base.provenance}"
    )
    return NicProfile(**values)


def dcqcn_link_bps(profile: NicProfile) -> int:
    """The whole-Gb/s rate the comparator's `-link_bps` and topology share.

    The backend's fat-tree loader parses `Downlink_speed_Gbps` as a whole
    number and the DCQCN runtime rejects any topology whose link rate differs
    from `-link_bps`, so a profile rate that is not a whole Gb/s is rounded
    here, once, and the same value must be written into the topology file.
    """
    return round(profile.goodput_bps / _BPS_PER_GBPS) * _BPS_PER_GBPS


def dcqcn_flags(profile: NicProfile) -> dict[str, str]:
    """Render `profile` into the DCQCN comparator flags that can carry it.

    The rate is rendered from `goodput_bps`, not `link_bps`: the packet path
    has a single rate rather than a wire rate and a payload rate, and the
    measured goodput asymptote is the one a study compares against. The
    caller owns where `-link_bps` goes, because the run wrapper passes it as a
    named argument rather than as an extra flag.

    Flags outside a profile's scope stay with the study: switch buffer sizes,
    the ECMP and RED seeds, the selective-repeat window and the DCQCN rate
    floor are fabric or policy parameters, not NIC properties.
    """
    return {
        "-link_bps": str(dcqcn_link_bps(profile)),
        "-max_wire_packet_bytes": str(profile.mtu_bytes),
        "-data_header_bytes": str(profile.header_bytes),
        "-pfc": "on" if profile.pfc_enabled else "off",
        "-recovery": profile.recovery,
        "-loss_rate_cut": "on" if profile.loss_rate_cut else "off",
        "-silent_rto_us": str(profile.rto_ps // _PS_PER_US),
        "-ecn_kmin_bytes": str(profile.ecn_kmin_bytes),
        "-ecn_kmax_bytes": str(profile.ecn_kmax_bytes),
        "-ecn_pmax_ppm": str(profile.ecn_pmax_ppm),
    }


def gap_fields(profile: NicProfile) -> tuple[str, ...]:
    """Model fields of `profile` that no DCQCN comparator flag can carry.

    This is the gap ledger. Every name it returns is a key of
    :data:`GAP_TASKS`, which points at the registry task that owns the missing
    mechanism.
    """
    return tuple(
        field.name
        for field in fields(profile)
        if field.name in MODEL_FIELDS and field.name not in _CLI_BACKED_FIELDS
    )


_CX5_PROVENANCE = (
    "ConnectX-5 Ex 100 GbE (MT4121) measured 2026-09-01 on the inbox mlx5_core driver, "
    "firmware 16.32.2004 and 16.31.2006, RoCEv2 GID 3, active MTU 4096, PFC off, PCIe Gen4 x16. "
    "Campaign records RESULTS-p2-msgsize.md, RESULTS-p4-kernels.md, RESULTS-p5a-incast.md, "
    "FINDINGS-cx5.md and data/p5a/congestion_control_config.md. Loss ledger tier: inferred to "
    "asserted (counters bracket the events; no injector named a transaction)."
)

#: The measured ConnectX-5 Ex 100 GbE profile.
CX5_100G = NicProfile(
    name="cx5_100g",
    # ethtool reports 100000 Mb/s on the port.
    link_bps=100_000_000_000,
    # Two-parameter refit of the true depth-1 WRITE curve, residuals at or
    # below 5.1 percent; the multi-QP fabric ceiling is 97.7 Gb/s.
    goodput_bps=97_100_000_000,
    mtu_bytes=4096,
    # RoCEv2 header stack plus FCS is about 58 B on the wire; 64 B is the
    # backend's own per-packet accounting and the value the study renders.
    header_bytes=64,
    # Lumped message offset from the same refit. Its split across doorbell,
    # WQE fetch, context lookup, admission and completion is not measured.
    t_eff_ps=4_480_000,
    # The depth at which the measured deep-pipeline equilibrium was observed;
    # the engine's default send queue depth, not a device maximum.
    sq_depth=1024,
    # A single UD receive queue pair discards silently beyond this rate.
    pps_ceiling_per_qp=3_070_000,
    # Highest counter-clean receive packet rate observed (512 B RC WRITE under
    # 2 to 1 fan-in, 20.5 Mmsg/s aggregate); a lower bound on the per-NIC
    # ceiling, not a located knee.
    pps_ceiling_per_nic=20_500_000,
    # One lossy pool, zero PFC headroom, read from the DCB buffer interface.
    rx_ingress_meter_bytes=262_016,
    pfc_enabled=False,
    # The device does limited selective repeat in hardware; go-back-N is the
    # modeling choice that reproduces the observed retransmit amplification.
    recovery="gbn",
    loss_rate_cut=True,
    # Local ACK timeout 14, i.e. 4.096 us x 2^14 = 67.109 ms.
    rto_ps=67_108_864_000,
    # The deployment's endpoint ECN state is unreadable on the inbox driver,
    # so these come from the 100 G vendor row of the message-size paper.
    ecn_kmin_bytes=102_400,
    ecn_kmax_bytes=409_600,
    ecn_pmax_ppm=250_000,
    evidence={
        "link_bps": "documented",
        "goodput_bps": "calibrated-opaque",
        "mtu_bytes": "documented",
        "header_bytes": "driver-inferred",
        "t_eff_ps": "calibrated-opaque",
        "sq_depth": "driver-inferred",
        "pps_ceiling_per_qp": "calibrated-opaque",
        "pps_ceiling_per_nic": "calibrated-opaque",
        "rx_ingress_meter_bytes": "documented",
        "pfc_enabled": "documented",
        "recovery": "calibrated-opaque",
        "loss_rate_cut": "calibrated-opaque",
        "rto_ps": "documented",
        "ecn_kmin_bytes": "declared",
        "ecn_kmax_bytes": "declared",
        "ecn_pmax_ppm": "declared",
    },
    provenance=_CX5_PROVENANCE,
)

#: ConnectX-7 400 G, declared as the ConnectX-5 architecture at four times the
#: rate. Scaling the ECN byte thresholds by four reproduces the 400 G row of
#: `docs/papers/msg-size-vs-bandwidth.md` exactly, which is a consistency
#: check on the scaling rule rather than independent evidence.
CX7_400G = scale_profile(CX5_100G, link_factor=4.0, name="cx7_400g")

#: Profiles by name, for study command lines.
PROFILES: Mapping[str, NicProfile] = MappingProxyType(
    {profile.name: profile for profile in (CX5_100G, CX7_400G)}
)
