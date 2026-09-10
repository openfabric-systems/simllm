# Relative-error calibration, frozen before the next fit

The third development candidate is retained with its failures. Its absolute
microsecond least-squares objective gives the largest forced-protocol messages
the greatest influence, although the acceptance criterion is relative error.
Use squared fractional residuals for the same frozen calibration rows and
nonnegative mechanism coefficients. The row weight is the reciprocal of its
calibration median. This changes no payload partition, source geometry, timing
parameter count, uncertainty rule or acceptance threshold. Validation medians
never determine weights or fitted values.

The observed protocol boundary is a bracket, not a known exact byte. Within
the open interval between the final observed old-protocol payload and the first
observed new-protocol payload, the uncertainty band is the union of both
protocol estimates. Keep the central prediction at the old protocol until the
recorded first-new-protocol size. Report the ambiguity explicitly. This is
selection uncertainty from the frozen 16-KiB sampling resolution, distinct from
the software overhead envelope. Do not infer a sharper boundary from fresh
validation and then rescore the same run.
