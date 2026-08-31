# Collective floor calibration, second freeze: the corrected axis

The first attempt is VOID exactly as the first freeze demanded: its
axis pin said BYTES, the evidence says ELEMENTS, and the run stopped
before any regime choice, fit or modification. This second freeze
records that finding and corrects the pin. Nothing else changes: every
guard, family, band and closure rule of expectations.md carries forward
unchanged, and expectations.md itself stays immutable.

## The finding

The external table's message_size coordinate is an ELEMENT COUNT, not
bytes. The pinned sdk constructs tokens times elements per token and
passes that count directly to query_nccl
(aiconfigurator_core/sdk/operations/communication.py:516); its analytic
path multiplies the same count by the dtype width separately
(communication.py:394); half is two bytes and int8 one byte
(common.py:1107); and the table itself confirms it physically, the half
ladder starting at 256 elements where the int8 ladder starts at 512,
both spanning 512 bytes. The sdk's message_bytes interpolation label is
mislabeled; simllm's importer carries the raw coordinate unconverted
(simllm/calibration/external_nccl.py:281).

Two consequences are recorded so later readers are not misled:

- The probe's fixed-overhead decomposition tables listed the raw
  coordinate under a byte heading; for half-precision rows the true
  byte size is twice the listed value. Intercepts (zero-size floors)
  are unaffected; slopes and effective bandwidths must be re-derived
  over true bytes by the study and the probe's values treated as
  element-axis artifacts.
- The merged MiniMax reproduction is NOT impugned: its Family E cells
  were bit-equal to the live sdk because both sides used the same
  element semantics end to end. The mislabel lived in the axis name
  and in the first freeze's pin, not in any served value.

## The corrected pin

- The table coordinate is ELEMENTS. The calibration's physical axis is
  BYTES, computed as elements times the dtype width, two for half and
  one for int8, with that conversion cited to the sources above in the
  study configuration.
- Every fitted slope is expressed in picoseconds per BYTE and every
  effective bandwidth in bytes per second over the true byte axis.
  Regime boundaries are chosen and frozen on the true byte axis.
- The 63/63 membership rule of the first freeze is unchanged; size
  indices refer to the same 21-step ladders regardless of axis, so the
  membership is identical.
- Callers of the calibrated authority supply bytes; the authority owns
  the conversion at the import boundary and a test proves a half and
  an int8 query at equal BYTE size resolve to their distinct measured
  cells.

All bands, including the Family H 10 percent bar and the Family D8
[0.90, 1.10] quotient, carry forward exactly as first frozen.
