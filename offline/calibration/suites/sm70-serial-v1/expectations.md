# SM70 serial kernel integration expectations

This freeze precedes implementation and behavioral tests. It covers an explicit
FP16 Tesla V100 PCIe peak envelope, an optional offline simulator adapter, and
a compilable CUDA probe that executes real matrix and reduction kernels on one
stream. It changes no default GPU, provider or serving path.

The oracle grid varies kernel count and core clock, independently. Serial
service is the sum of per-kernel cycles. Doubling clock halves time within one
picosecond of integer rounding; repeating an identical kernel scales service
linearly. Fake subprocess outputs used by unit tests are diagnostic fixtures,
never captured traces or measured simulator results. CUDA compilation proves
only that a target executable can be built. Real replay requires captured SM70
inputs, immutable hashes, the declared official simulator pins, complete output
and one active kernel at a time. No candidate promotes itself to silicon evidence.

The V100 PCIe peak ceiling is 112 TFLOP/s for FP16 tensor arithmetic and
900 GB/s for HBM. These are nameplate limits, not achieved service rates.
The source is the [NVIDIA V100 data sheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf).
The pinned [official simulator configuration](https://github.com/accel-sim/accel-sim-framework/blob/3016c658f810bdae9a14bf4534ee99e9945eedae/util/job_launching/configs/define-standard-cfgs.yml)
names QV100-SASS and SM7_QV100. That profile is an explicit simulator target,
not an automatically calibrated Tesla V100 PCIe substitute.
