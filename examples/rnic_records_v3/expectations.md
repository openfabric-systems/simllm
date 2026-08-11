# BACK-28 effective-hardware reader expectations

## Freeze status and scope

This expectations-only record precedes Python effective-hardware v2/v3
ingestion and every result-producing reader-parity run. The companion
untracked dry-run harness contains the frozen native-emitted fixture values,
hashes, mutation registry, source pins and check-only validation. Its
check-only path does not import or exercise the future Python reader changes.

BACK-28 is a component completeness task. Its decision metric is strict
acceptance parity between the existing native reader and
`simllm.backends.rnic_records` for canonical effective-hardware objects. It
does not change a simulator runtime and therefore makes no TTFT or TPOT claim.
CORE-21 is the named live-chain successor that will consume accepted
structural records while comparing native and bypass authority on one fixed
contended graph.

## Native-source audit before freeze

The audit used SimLLM commit
`9923c9f0add6b6f23a0019382962931e1792bc47` before this freeze. No Python
v2/v3 implementation or result-producing rejection run existed.

- `simllm/backends/rnic/include/simllm/rnic/session_record.h:17-22` fixes the
  effective-hardware v1, v2 and v3 schema names.
- `simllm/backends/rnic/src/session_record.cpp:158-325` parses the native
  canonical JSON byte grammar with unsigned 64-bit integers, sorted unique
  fields and bounded depth. The Python API begins from an already decoded
  object, so byte-grammar mutations are outside this object-parity study.
- `simllm/backends/rnic/src/session_record.cpp:335-417` owns strict object,
  array, scalar, field-set, positivity, signed-timestamp and power-of-two
  checks used by all versions.
- `simllm/backends/rnic/src/session_record.cpp:420-749` validates analytical
  profiles, PCIe paths, credits, fabric geometry and the work-queue binding.
- `simllm/backends/rnic/src/session_record.cpp:760-850` validates the exact
  submission field set, nonzero bounded producer, requester and sole CQ
  consumer identities, and all three producer-shape ownership combinations.
- `simllm/backends/rnic/src/session_record.cpp:853-1085` validates host-memory
  registry and allocation field sets, typed ownership, endpoints, paths, page
  geometry, queue allocation bindings, submission endpoint agreement and the
  CPU-proxy descriptor allocation.
- `simllm/backends/rnic/src/session_record.cpp:1088-1200` dispatches schema
  v1/v2/v3, requires DMA and QPC for host memory and validates the effective
  work-queue shape.
- `simllm/backends/rnic/src/session_record.cpp:1425-1529,1879-1924` emits the
  canonical host-memory, submission and complete effective-hardware objects,
  then hashes their exact canonical bytes.
- `simllm/backends/rnic/src/session_record.cpp:1955-1980` validates lowercase
  SHA-256 syntax, calls the effective-hardware reader and compares the hash
  with the effective object bytes.
- `simllm/backends/rnic/tests/submission_test.cpp:596-684` proves native v2
  compatibility, v3 CPU-proxy emission and shape-versus-endpoint rejection.

An audit-only program instantiated the same native submission fixtures and
called `renderEffectiveHardwareConfigJson`; it did not modify production
sources. The four frozen effective-object SHA-256 values are:

| Fixture | Schema and shape | SHA-256 |
|---|---|---|
| `v2` | v2 host memory, no submission object | `4a94c6ec23c0af9a18524d33dbb3127dd1d4cde4dcfced7e972fdb1dda5dfebf` |
| `v3_host` | v3 `host_cpu_driver` | `4ffabebbb9c6f5aace706f241af030b95c286c607a6e4f9e39a146e5065dfa17` |
| `v3_proxy` | v3 `cpu_proxy` | `a9cc2fa1df269f75d6c8d48ba27bff81dd5c65f086ea9c4fe3f3eaadb264cee3` |
| `v3_gpu` | v3 `gpu_initiated` | `cd4da0c3635006ce3a02f1a19b1ecd1700ca92fa6758ee87a77bccca6f15c4ad` |

All semantic mutations except N099 and N100 are rendered back to sorted,
separator-free ASCII JSON and receive a freshly recomputed hash. They
therefore reach the native content check instead of failing an unrelated stale
hash. N099 and N100 deliberately leave the valid v2 object unchanged and
mutate only its supplied hash.

## Frozen native rejection corpus

The following 100 mutations are one representative for every distinct native
rejection branch relevant to strict v2/v3 object ingestion. Array indexes refer
to the native-emitted order. Values `UINT32+1`, `UINT64+1` and `INT64+1` mean
`2^32`, `2^64` and `2^63`, respectively.

### Root, modules and host-memory enablement

These derive from `session_record.cpp:751-758,1088-1159`.

- N001 (`v2`): set `schema` to `simllm-rnic-effective-hardware-v4`.
- N002 (`v2`): remove `host_memory`.
- N003 (`v2`): add the `v3_host.submission` object.
- N004 (`v3_host`): remove `submission`.
- N005 (`v2`): add root field `unexpected: true`.
- N006 (`v2`): add `network.unexpected: true`.
- N007 (`v2`): set `qpc.enabled` to integer 1.
- N008 (`v2`): replace `dma` with `{"enabled": false}`.
- N009 (`v2`): set `qpc.enabled` false.

### DMA fabric and path geometry

These derive from `session_record.cpp:420-749,1129-1150`.

- N010 (`v2`): add `dma.unexpected: true`.
- N011 (`v2`): set `dma.fabric_scope` to `borrowed`.
- N012 (`v2`): remove `dma.fabric.analytical_seed`.
- N013 (`v2`): set `dma.fabric.generation` to 6.
- N014 (`v2`): set `dma.fabric.lane_count` to 3.
- N015 (`v2`): set `dma.fabric.max_payload_size_bytes` to 64.
- N016 (`v2`): set `dma.fabric.read_completion_boundary_bytes` to 32.
- N017 (`v2`): set `dma.fabric.completion_overhead_bytes` to zero.
- N018 (`v2`): set `dma.fabric.data_credit_unit_bytes` to 3.
- N019 (`v2`): set host-to-device `posted_header_credits` to zero.
- N020 (`v2`): set host-to-device `completion_header_credits` to `UINT32+1`.
- N021 (`v2`): set host-to-device `posted_data_credits` to 1.
- N022 (`v2`): set `host_store_latency_ps` to `[0, 0]`.
- N023 (`v2`): empty `dma.fabric.paths`.
- N024 (`v2`): set `paths[1].path_id` to 1, duplicating its predecessor.
- N025 (`v2`): set `paths[0].path_id` to zero.
- N026 (`v2`): set `paths[2].path_id` to `UINT32+1`.
- N027 (`v2`): remove `paths[1].endpoint` while it remains enabled.
- N028 (`v2`): set `paths[1].endpoint` to `system_memory`.
- N029 (`v2`): set `paths[1].base_latency_ps` to `INT64+1`.
- N030 (`v2`): set `paths[2].enabled` false while retaining active fields.
- N031 (`v2`): remove `dma.work_queue.pcie_wqe_bytes`.
- N032 (`v2`): set `dma.work_queue.pcie_wqe_bytes` to zero.
- N033 (`v2`): set `dma.work_queue.pcie_cq_first_byte_offset` to 4096.
- N034 (`v2`): set `dma.work_queue.pcie_cq_memory_path_id` to `UINT32+1`.
- N035 (`v2`): set `dma.work_queue.pcie_cq_memory_path_id` to missing path 999.
- N036 (`v2`): set `dma.work_queue.pcie_cq_memory_path_id` to MMIO path 1.

### Submission shape, identities and sole CQ consumer

These derive from `session_record.cpp:760-850`.

- N037 (`v3_host`): add extra field `submission.cq_consumer_ids: [8101]`.
- N038 (`v3_host`): set `submission.producer_id` to zero.
- N039 (`v3_host`): set `submission.cq_consumer_id` to `UINT32+1`.
- N040 (`v3_host`): set `submission.rnic_requester_id` to zero.
- N041 (`v3_host`): set `submission.producer_shape` to `fpga_proxy`.
- N042 (`v3_host`): set `submission.producer_kind` to `gpu`.
- N043 (`v3_host`): set `submission.descriptor_writer_id` to 1.
- N044 (`v3_proxy`): set `submission.descriptor_writer_kind` to
  `host_cpu_driver`.
- N045 (`v3_proxy`): set `submission.descriptor_writer_id` to zero.
- N046 (`v3_proxy`): set `submission.descriptor_queue_allocation_id` to zero.
- N047 (`v3_proxy`): set `submission.descriptor_queue_endpoint` to `none`.
- N048 (`v3_gpu`): set `submission.producer_kind` to `cpu_proxy`.
- N049 (`v3_gpu`): set `submission.queue_endpoint` to `host_pinned_memory`.
- N050 (`v3_gpu`): set `submission.cq_consumer_kind` to `cpu_proxy`.
- N051 (`v3_gpu`): set `submission.uar_mapping_owner` to `host_cpu`.

### Host-memory registry and allocation scalars

These derive from `session_record.cpp:853-1014`.

- N052 (`v2`): add `host_memory.unexpected: true`.
- N053 (`v2`): set `host_memory.enabled` false.
- N054 (`v2`): set `host_memory.device_owner_id` to zero.
- N055 (`v2`): remove `host_memory.registry.mpt_entry_bytes`.
- N056 (`v2`): set `host_memory.registry.mpt_entry_bytes` to zero.
- N057 (`v2`): set `host_memory.registry.mpt_first_byte_offset` to 4096.
- N058 (`v2`): set `host_memory.registry.translation_path_id` to
  `UINT32+1`.
- N059 (`v2`): set `host_memory.registry.translation_path_id` to GPU path 3.
- N060 (`v2`): remove `host_memory.work_queue.qpc_context_bytes`.
- N061 (`v2`): set `host_memory.work_queue.qpc_context_bytes` to zero.
- N062 (`v2`): empty `host_memory.allocations`.
- N063 (`v2`): remove `allocations[0].owner_id`.
- N064 (`v2`): add `mkey: 1` to non-data `allocations[0]`.
- N065 (`v2`): set `allocations[0].allocation_id` to zero.
- N066 (`v2`): swap `allocations[0]` and `allocations[1]`.
- N067 (`v2`): set `allocations[0].device_owner_id` to zero.
- N068 (`v2`): set `allocations[0].length_bytes` to zero.
- N069 (`v2`): set `allocations[0].owner_id` to zero.
- N070 (`v2`): set `allocations[0].path_id` to zero.
- N071 (`v2`): set data-region `allocations[5].mkey` to zero.
- N072 (`v2`): set `allocations[0].owner_kind` to `memory_region`.
- N073 (`v2`): set `allocations[0].object_kind` to `unknown_object`.
- N074 (`v2`): set `allocations[5].endpoint` to `device_memory`.
- N075 (`v2`): set QPC `allocations[0]` to GPU endpoint and path 3.
- N076 (`v2`): set `allocations[0].path_id` to `UINT32+1`.
- N077 (`v2`): set `allocations[0].path_id` to GPU path 3 while retaining
  its host-pinned endpoint.

### Page geometry, WQ allocation binding and descriptor ownership

These derive from `session_record.cpp:1016-1085`.

- N078 (`v2`): remove `allocations[0].pages.physical_page_addresses`.
- N079 (`v2`): set `allocations[0].pages.page_size_bytes` to 2048.
- N080 (`v2`): set `allocations[0].pages.page_size_bytes` to 6144.
- N081 (`v2`): empty `allocations[0].pages.physical_page_addresses`.
- N082 (`v2`): add one byte to the first physical page address.
- N083 (`v2`): duplicate the first data-region physical page address.
- N084 (`v2`): set `host_memory.work_queue.qpc_icm_allocation_id` to 999.
- N085 (`v2`): bind `sq_ring_allocation_id` to CQ allocation 24.
- N086 (`v3_gpu`): move SQ allocation 22 to host endpoint and path 2 while
  submission remains GPU-initiated.
- N087 (`v3_gpu`): move CQ allocation 24 to host endpoint and path 2.
- N088 (`v3_gpu`): move doorbell allocation 25 to host endpoint and path 2.
- N089 (`v3_proxy`): remove descriptor allocation 27.
- N090 (`v3_proxy`): append a second valid descriptor allocation with ID 28.
- N091 (`v3_proxy`): change allocation 27 to an SQ ring owned by a send queue.
- N092 (`v3_proxy`): set allocation 27 owner ID to 7203.
- N093 (`v3_proxy`): move allocation 27 to GPU endpoint and path 3.
- N094 (`v3_host`): append the valid proxy descriptor allocation 27.

### Effective WQ and canonical hash

These derive from `session_record.cpp:1162-1198,1955-1980`.

- N095 (`v2`): add `work_queue.cqe_write_service_ps: 0`.
- N096 (`v2`): set `work_queue.sq_depth` to zero.
- N097 (`v2`): set `work_queue.cq_depth` to zero.
- N098 (`v2`): set `work_queue.scheduler_service_ps` to `INT64+1`.
- N099 (`v2`): supply 64 uppercase `A` characters as the hardware hash.
- N100 (`v2`): supply 64 lowercase zeroes as the hardware hash.

## Scored relation and entailment

For mutation N, let `A_native(N)` and `A_python(N)` equal one when that reader
accepts the session configuration and zero when it rejects. The exact scored
relation for each of N001 through N100 is:

```text
A_native(N) = 0
A_python(N) = 0
A_python(N) - A_native(N) = 0
```

The band is exact. The two varied axes are schema/producer base and rejection
category. Every pair is genuine-risk evidence because the Python reader must
first accept all valid v2/v3 bases, then independently reach the nested check;
an early blanket non-v1 rejection cannot satisfy the acceptance controls.

Entailment is evaluated from raw observations. For every mutation, the harness
records the native exit status and diagnostic plus the Python accept/reject
status and diagnostic before evaluating the relation. It does not first assert
that the native reader rejected, and no fatal oracle pins the Python result.
Valid-base acceptance, frozen hashes and off-path identity are separate fatal
unscored evidence. They do not entail any mutated-object result. Thus every
scored instance can fail in a run that reaches it.

## Fatal unscored controls

- The native probe is a thin CLI over the repository's existing
  `RnicSessionConfigRecord`, `rnicSha256Hex` and
  `validateRnicSessionConfigRecord`. The harness does not implement a second
  native comparator.
- Native and Python readers must both accept all four unmodified v2/v3 bases.
  Their recursively frozen Python projections must retain every field and
  array value; attempted nested mapping or tuple mutation must fail.
- The v1 effective object remains
  `a9732c130d2ed0075668c7ee1f77c742492ca059f1c50b1ca35c078799deaa9c`.
  The complete v1 structural config remains
  `69f20997fede3a9a00b386a5a0412f948dba5bc1b6eb0c7e93d6d6dd85e01d0c`,
  and bypass remains
  `c750be3ba90023987478e6ecd111ee70ad90c02f669470e547ef252e047afc2b`.
  Parsing must yield the same record values as the existing v1 and bypass
  fixtures. These SHA guards are change-set guards; parsed-record identity is
  the fatal off-path evidence.
- Every accepted effective object must recompute to its supplied canonical
  hash. Every semantic mutation except N099/N100 uses a recomputed hash so the
  intended content rejection is observable.
- Native CMake build and CTest, Python 3.10-compatible unit tests, and the full
  repository lint/test gates must pass. No test requires `third_party/`.

## Registered command and pre-freeze dry run

Local configuration sets `SIMLLM_WAVE5_RUN_ROOT` to this branch's external run
root. The registered command is:

```bash
.venv/bin/python examples/rnic_records_v3/run_study.py \
  --out "$SIMLLM_WAVE5_RUN_ROOT/back28"
```

Before this freeze, the command was run with `--check-only`. It reconstructed
the four native-emitted bases from frozen literals, matched all four hashes,
matched the v1 and bypass hashes, validated the contiguous N001 through N100
registry, checked every source-audit input and enforced the external-output
root. It created no output directory and imported no future reader behavior.
