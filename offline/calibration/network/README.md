# Network calibration namespace

This namespace is separate from kernel calibration. It holds reviewed
transport and fabric calibration configuration when the owning traffic or
backend task publishes it. Raw packet captures and large simulator outputs
remain outside Git and are joined by content identity.

The local-shard kernel collector never writes here and never claims a network
duration. DCQCN remains the expected-fail comparator and needs no generic
calibration bundle for that role. Slingshot and InfiniBand calibration enter
through their traffic and backend authorities when their registered campaigns
publish validated evidence; they do not reuse kernel records or kernel sample
blobs.
