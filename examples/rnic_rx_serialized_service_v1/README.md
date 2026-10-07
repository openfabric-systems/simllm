# Declared RNIC receive service

This native component study compares an explicit packet-aligned receive service
against independent scheduling calculations. The original measured CX5 fluid
meter remains the default. [RESULTS.md](RESULTS.md) records the qualified scope
and [expectations.json](expectations.json) freezes the relations before code and
measurements.

From the repository root:

```bash
python examples/rnic_rx_serialized_service_v1/run_native.py \
  --output "${SIMLLM_DATA_ROOT}/rnic_rx_serialized_service_v1" --mutants
```

The output must be outside the checkout. The runner builds the native RNIC with
warnings as errors, runs its complete tests, writes the 36-row grid and exact
source/expectation hashes, then compiles three separately mutated source copies.
A negative copy must reach a fixture failure rather than merely fail to compile.
Generated builds, traces and logs remain under the external data root.

`rnic_cm_create_with_rx_service` accepts a version-1 service option with
`RNIC_CM_RX_SERVICE_SERIALIZED_PACKET_BEATS`,
`RNIC_CM_RX_READY_CONTINUOUS`, beat width, period, phase and token capacity.
Receive and packetization must be enabled. The configuration is immutable;
original `rnic_cm_create` retains fluid service. Invalid or unsupported options
fail explicitly.

The reported metric is time to drain accepted wire bytes from the selected
receive queue. It is separate from ACK generation, payload delivery, CQE
visibility and request TTFT/TPOT. Framework selection and request propagation
are registered as BACK-76 in [the backend module](../../docs/modules/backends.md).
