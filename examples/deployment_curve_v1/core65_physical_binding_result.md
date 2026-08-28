# CORE-65 retained kernel inventory and EP72 physical binding

Status: **PROTOCOL_VOID_NULL_MOVEMENT_EXACT_HARDWARE_REMAINDER**. The retained stream is enumerated totally, but
it cannot be bound totally to SGLang EP72 physical launches. CORE-65 therefore
publishes a null calibration movement and registers the literal EP72 capture as
CORE-66.

## Total retained kernel inventory and EP72 comparison

All **46 of 46**
retained rows are named below. Their services sum exactly to
**1,875,680,000 ps**. No row is unmapped.

| Order | Retained physical kernel | Family | Count/step | Count/layer basis | Service share | EP72 binding |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 0 | <code>void at::native::vectorized_elementwise_kernel&lt;(int)4, at::native::FillFunctor&lt;int&gt;, std::array&lt;char *, (unsigned long)1&gt;&gt;(int, T2, T3)</code> | step_setup | 2.0 | 2/1 | 0.081890% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 1 | <code>void at::native::index_elementwise_kernel&lt;(int)128, (int)4, void at::native::gpu_index_kernel&lt;void at::native::index_kernel_impl&lt;at::native::OpaqueType&lt;(int)4&gt;&gt;(at::TensorIteratorBase &amp;, c10::ArrayRef&lt;long&gt;, c10::ArrayRef&lt;long&gt;)::[lambda(char *, const char *, long) (instance 1)]&gt;(at::TensorIteratorBase &amp;, c10::ArrayRef&lt;long&gt;, c10::ArrayRef&lt;long&gt;, const T1 &amp;, bool)::[lambda(int) (instance 1)]&gt;(long, T3)</code> | step_setup | 1.0 | 1/1 | 0.080184% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 2 | <code>void at::native::unrolled_elementwise_kernel&lt;at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &amp;)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(long) (instance 1)], std::array&lt;char *, (unsigned long)2&gt;, (int)4, TrivialOffsetCalculator&lt;(int)1, unsigned int&gt;, TrivialOffsetCalculator&lt;(int)1, unsigned int&gt;, at::native::memory::LoadWithCast&lt;(int)1&gt;, at::native::memory::StoreWithCast&lt;(int)1&gt;&gt;(int, T1, T2, T4, T5, T6, T7)</code> | step_setup | 2.0 | 2/1 | 0.155250% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 3 | <code>void at::native::vectorized_elementwise_kernel&lt;(int)2, at::native::CUDAFunctor_add&lt;long&gt;, std::array&lt;char *, (unsigned long)3&gt;&gt;(int, T2, T3)</code> | step_setup | 1.0 | 1/1 | 0.056300% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 4 | <code>void at::native::vectorized_elementwise_kernel&lt;(int)4, at::native::CUDAFunctor_add&lt;int&gt;, std::array&lt;char *, (unsigned long)3&gt;&gt;(int, T2, T3)</code> | step_setup | 1.0 | 1/1 | 0.054594% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 6 | <code>_compute_slot_mapping_kernel</code> | kv_slot_mapping | 1.0 | 1/1 | 0.527169% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 7 | <code>void at::native::unrolled_elementwise_kernel&lt;at::native::CUDAFunctorOnSelf_add&lt;int&gt;, std::array&lt;char *, (unsigned long)2&gt;, (int)4, TrivialOffsetCalculator&lt;(int)1, unsigned int&gt;, TrivialOffsetCalculator&lt;(int)1, unsigned int&gt;, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast&gt;(int, T1, T2, T4, T5, T6, T7)</code> | step_setup | 1.0 | 1/1 | 0.046063% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 8 | <code>void flash::prepare_varlen_num_blocks_kernel&lt;(int)2, (bool)0&gt;(int, int, int, const int *, const int *, const int *, const int *, const int *, const int *, int, int, int, int, int, cutlass::FastDivmod, cutlass::FastDivmod, int *, int *, int *, int *, int *, bool, bool, bool, int)</code> | mla_attention_setup | 1.0 | 1/1 | 0.095539% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 9 | <code>void at::native::unrolled_elementwise_kernel&lt;at::native::FillFunctor&lt;int&gt;, std::array&lt;char *, (unsigned long)1&gt;, (int)4, TrivialOffsetCalculator&lt;(int)0, unsigned int&gt;, TrivialOffsetCalculator&lt;(int)1, unsigned int&gt;, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast&gt;(int, T1, T2, T4, T5, T6, T7)</code> | step_setup | 1.0 | 1/1 | 0.037533% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 10 | <code>triton_red_fused__to_copy_embedding_rms_norm_0</code> | embedding_and_input_norm | 1.0 | 1/1 | 0.184253% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 11 | <code>void per_token_group_quant_8bit_kernel&lt;c10::BFloat16, c10::Float8_e4m3fn, (bool)1, (bool)1, float&gt;(const T1 *, void *, T5 *, int, int, int, float, float, float, int, int)</code> | mixed_projection_quantization | 20.0 | 5/1 | 3.325087% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 12 | <code>void deep_gemm::sm90_fp8_gemm_1d2d_impl&lt;(cute::UMMA::Major)0, (unsigned int)0, (unsigned int)2112, (unsigned int)7168, (unsigned int)1, (unsigned int)32, (unsigned int)16, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)32, (unsigned int)16, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)132, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity&gt;(float *, int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)</code> | mla_q_and_kv_compression | 4.0 | 1/1 | 2.265632% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 13 | <code>triton_poi_fused_1</code> | first_layer_mla_transform | 1.0 | 1/1 | 0.088714% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 14 | <code>triton_red_fused_2</code> | first_layer_mla_transform | 1.0 | 1/1 | 0.180841% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 16 | <code>void deep_gemm::sm90_fp8_gemm_1d2d_impl&lt;(cute::UMMA::Major)0, (unsigned int)0, (unsigned int)24576, (unsigned int)1536, (unsigned int)1, (unsigned int)32, (unsigned int)96, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)64, (unsigned int)13, (unsigned int)128, (unsigned int)128, (unsigned int)2, (bool)0, (unsigned int)132, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity&gt;(float *, int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)</code> | mla_q_decompression | 4.0 | 1/1 | 3.942677% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 17 | <code>triton_poi_fused_add_clone_copy_expand_index_mul_neg_slice_split_stack_unsqueeze_view_3</code> | first_layer_rotary_transform | 1.0 | 1/1 | 0.185959% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 18 | <code>void vllm::concat_and_cache_mla_kernel&lt;__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0&gt;(const T1 *, const T1 *, T2 *, const long *, int, int, int, int, int, int, int, const float *)</code> | mla_kv_cache_write | 4.0 | 1/1 | 0.452103% | BACKEND_DEPENDENT_SEMANTIC_COUNTERPART_ONLY |
| 19 | <code>nvjet_tst_256x32_64x5_2x1_v_bz_TNT</code> | mla_attention_projection | 4.0 | 1/1 | 1.871535% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 20 | <code>void cutlass::device_kernel&lt;flash::enable_sm90_or_later&lt;flash::FlashAttnFwdSm90&lt;flash::CollectiveMainloopFwdSm90&lt;(int)2, cute::tuple&lt;cute::C&lt;(int)1&gt;, cute::C&lt;(int)1&gt;, cute::C&lt;(int)1&gt;&gt;, cute::tuple&lt;cute::C&lt;(int)64&gt;, cute::C&lt;(int)64&gt;, cute::C&lt;(int)64&gt;&gt;, (int)512, cutlass::bfloat16_t, float, cutlass::arch::Sm90, (bool)0, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, (bool)1, (bool)0, (bool)0, (bool)1, (bool)1, (bool)0, cutlass::bfloat16_t, (int)1&gt;, flash::CollectiveEpilogueFwd&lt;cute::tuple&lt;cute::C&lt;(int)64&gt;, cute::C&lt;(int)512&gt;, cute::C&lt;(int)64&gt;&gt;, cute::tuple&lt;cute::C&lt;(int)1&gt;, cute::C&lt;(int)1&gt;, cute::C&lt;(int)1&gt;&gt;, cutlass::bfloat16_t, cutlass::arch::Sm90, (int)256, (bool)1, (bool)1, (bool)1, (bool)0, (int)1&gt;, flash::VarlenDynamicPersistentTileScheduler&lt;(int)64, (int)64, (int)256, (int)128, (bool)1, (bool)1, (bool)1, (bool)0, (bool)0, (bool)1&gt;&gt;&gt;&gt;(T1::Params)</code> | mla_attention | 4.0 | 1/1 | 9.975262% | BACKEND_DEPENDENT_SEMANTIC_COUNTERPART_ONLY |
| 21 | <code>void cutlass::device_kernel&lt;flash::FlashAttnFwdCombine&lt;cute::tuple&lt;cute::C&lt;(int)8&gt;, cute::C&lt;(int)128&gt;&gt;, (int)5, (int)256, (int)1, (bool)0, (bool)1, cutlass::bfloat16_t, float, cutlass::arch::Sm90&gt;&gt;(T1::Params)</code> | mla_attention_combine | 4.0 | 1/1 | 3.529813% | BACKEND_DEPENDENT_SEMANTIC_COUNTERPART_ONLY |
| 22 | <code>nvjet_tst_128x32_64x10_1x1_h_bz_NNT</code> | mla_attention_projection | 4.0 | 1/1 | 1.810117% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 24 | <code>void deep_gemm::sm90_fp8_gemm_1d2d_impl&lt;(cute::UMMA::Major)0, (unsigned int)0, (unsigned int)7168, (unsigned int)16384, (unsigned int)1, (unsigned int)32, (unsigned int)64, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)16, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)132, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity&gt;(float *, int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)</code> | mla_output_projection | 4.0 | 1/1 | 9.308198% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 25 | <code>triton_red_fused_fused_add_rms_norm_0</code> | dense_input_norm | 3.0 | 1/1 | 0.474281% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 27 | <code>void deep_gemm::sm90_fp8_gemm_1d2d_impl&lt;(cute::UMMA::Major)0, (unsigned int)0, (unsigned int)36864, (unsigned int)7168, (unsigned int)1, (unsigned int)32, (unsigned int)144, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)32, (unsigned int)9, (unsigned int)128, (unsigned int)128, (unsigned int)2, (bool)0, (unsigned int)132, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity&gt;(float *, int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)</code> | dense_gate_up_projection | 3.0 | 1/1 | 12.919901% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 28 | <code>triton_poi_fused_mul_silu_slice_1</code> | dense_activation | 3.0 | 1/1 | 0.363388% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 30 | <code>void deep_gemm::sm90_fp8_gemm_1d2d_impl&lt;(cute::UMMA::Major)0, (unsigned int)0, (unsigned int)7168, (unsigned int)18432, (unsigned int)1, (unsigned int)32, (unsigned int)64, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)16, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)132, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity&gt;(float *, int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)</code> | dense_down_projection | 3.0 | 1/1 | 6.924849% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 31 | <code>triton_red_fused_fused_add_rms_norm_2</code> | dense_residual_and_norm | 3.0 | 1/1 | 0.520345% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 34 | <code>triton_poi_fused_3</code> | later_layer_mla_transform | 3.0 | 1/1 | 0.255907% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 35 | <code>triton_red_fused_4</code> | later_layer_mla_transform | 3.0 | 1/1 | 0.533993% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 38 | <code>triton_poi_fused_add_clone_copy_expand_index_mul_neg_slice_split_stack_unsqueeze_view_5</code> | later_layer_rotary_transform | 3.0 | 1/1 | 0.556172% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 88 | <code>triton_red_fused_fused_add_rms_norm_moe_forward_shared_0</code> | moe_input_norm | 1.0 | 1/1 | 0.192783% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 90 | <code>nvjet_tst_64x32_64x16_4x1_v_bz_splitK_TNT</code> | moe_router | 1.0 | 1/1 | 0.460633% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 91 | <code>void deep_gemm::sm90_fp8_gemm_1d2d_impl&lt;(cute::UMMA::Major)0, (unsigned int)0, (unsigned int)4096, (unsigned int)7168, (unsigned int)1, (unsigned int)32, (unsigned int)32, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)64, (unsigned int)16, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)132, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity&gt;(float *, int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)</code> | moe_shared_expert_gate_up | 1.0 | 1/1 | 0.904205% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 92 | <code>void cublasLt::splitKreduce_kernel&lt;(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv_bfloat16, (bool)1, (bool)0, (bool)0, (bool)0&gt;(cublasLt::cublasSplitKParams&lt;T6&gt;, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)</code> | moe_router_reduction | 1.0 | 1/1 | 0.235435% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 93 | <code>void vllm::moe::grouped_topk_fused_small_expert_count_kernel&lt;__nv_bfloat16, float, int, (vllm::moe::ScoringFunc)1, (int)256, (bool)1, (int)8&gt;(T1 *, float *, T3 *, const T2 *, long, long, long, long, long, long, bool, double)</code> | moe_topk | 1.0 | 1/1 | 0.213256% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 94 | <code>void per_token_group_quant_8bit_kernel&lt;c10::BFloat16, c10::Float8_e4m3fn, (bool)0, (bool)1, float&gt;(const T1 *, void *, T5 *, int, int, int, float, float, float, int, int)</code> | moe_mixed_quantization | 2.0 | 2/1 | 0.407745% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 95 | <code>triton_poi_fused_mul_silu_slice_0</code> | moe_shared_expert_activation | 1.0 | 1/1 | 0.129660% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 96 | <code>void vllm::moe::moe_align_block_size_kernel&lt;int&gt;(const T1 *, int *, int *, int *, int *, int, int, int, int, unsigned long, int *, int, int, bool)</code> | vllm_routed_expert_alignment | 1.0 | 1/1 | 0.180841% | NO_ONE_TO_ONE_COUNTERPART |
| 98 | <code>void vllm::moe::count_and_sort_expert_tokens_kernel&lt;int&gt;(const T1 *, int *, int *, int *, unsigned long, int, int, int, bool)</code> | vllm_routed_expert_sort | 1.0 | 1/1 | 0.081890% | NO_ONE_TO_ONE_COUNTERPART |
| 99 | <code>void deep_gemm::sm90_fp8_gemm_1d2d_impl&lt;(cute::UMMA::Major)0, (unsigned int)0, (unsigned int)7168, (unsigned int)2048, (unsigned int)1, (unsigned int)32, (unsigned int)64, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)128, (unsigned int)16, (unsigned int)128, (unsigned int)128, (unsigned int)1, (bool)0, (unsigned int)132, (deep_gemm::GemmType)0, cutlass::bfloat16_t, deep_gemm::epilogue::transform::EpilogueIdentity&gt;(float *, int *, unsigned int, unsigned int, unsigned int, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)</code> | moe_shared_expert_down | 1.0 | 1/1 | 0.423100% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 100 | <code>fused_moe_kernel</code> | vllm_routed_expert_compute | 2.0 | 2/1 | 7.011857% | NO_ONE_TO_ONE_COUNTERPART |
| 101 | <code>void vllm::act_and_mul_kernel&lt;c10::BFloat16, __nv_bfloat162, &amp;vllm::silu_kernel&lt;c10::BFloat16&gt;, &amp;vllm::packed_silu_kernel&lt;__nv_bfloat162&gt;, (bool)1, (bool)1, (bool)0, (bool)0&gt;(T1 *, const T1 *, int, float, float, float)</code> | vllm_routed_expert_activation | 1.0 | 1/1 | 0.127954% | NO_ONE_TO_ONE_COUNTERPART |
| 104 | <code>void vllm::moe::moe_sum_vec_kernel&lt;c10::BFloat16, int, (int)8, (bool)0&gt;(T1 *, const T1 *, long, int, long, long, const T2 *, const int *, long, long)</code> | vllm_routed_expert_sum | 1.0 | 1/1 | 0.146720% | NO_ONE_TO_ONE_COUNTERPART |
| 105 | <code>triton_red_fused_add_fused_add_rms_norm_mul_1</code> | moe_residual_and_norm | 1.0 | 1/1 | 0.339504% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 107 | <code>void at::native::vectorized_gather_kernel&lt;(int)16, long&gt;(char *, char *, T2 *, int, long, long, long, long, bool)</code> | output_token_gather | 1.0 | 1/1 | 0.087008% | SEMANTIC_COUNTERPART_PHYSICAL_IDENTITY_UNBOUND |
| 108 | <code>nvjet_tst_512x32_64x3_2x1_v_bz_TNT</code> | lm_head | 1.0 | 1/1 | 28.253860% | SEMANTIC_COUNTERPART_WITH_DP_LM_HEAD_REQUIRED |

The stream is vLLM TP1/DP1/EP1/PP1. The real target is SGLang DP72/EP72 with
DeepEP. Orders 96, 98, 100, 101 and 104 are vLLM EP1 routed-expert scheduling,
compute, activation and sum rows; they have no one-to-one physical SGLang
DeepEP counterpart. Conversely, real EP72 requires DeepEP dispatch A/B and
combine A/B launches that are absent from this noncollective stream. Attention
and LM-head semantic work exists in both, but their SGLang physical identities
remain backend-dependent and uncaptured.

## Candidate verdicts

1. **Layer-type composition: decided.** The effective capture has four layers,
   and the pinned vLLM rule with `first_k_dense_replace=3` and frequency one
   makes them exactly three dense layers followed by one MoE layer. Common work
   legitimately scales by `61/4`; dense-only work scales by `3/3 = 1`, not
   `61/4`; MoE-only work scales by `58/1 = 58`, not `61/4`; step/output work
   stays once. Thus naive depth scaling overprices dense and step/output work,
   underprices MoE work, and has no preregisterable net sign without a valid
   component service split.
2. **Expert population: partly decided, service movement undecidable.** The
   retained MoE layer is resident over 256 logical experts, while an EP72 rank
   has four physical slots. A per-layer expert-count or resident-weight term
   scales by `4/256 = 1/64`; after replacing one captured MoE layer by 58 real
   MoE layers, its full-model resident routed-weight ratio is `58*4/256 =
   29/32`. The inherited `1/9` remains only on the previously classified
   assignment-tracked `fused_moe_kernel` service. It is not silently reused as
   an expert-count or weight-byte scale. Routing identities are absent, so the
   unique active expert weights actually read are undecidable.
3. **Weight-read volume: undecidable.** The reduced TP1 model's static resident
   inventory is 15,116,101,504 bytes; naive
   `61/4` scaling gives
   230,520,547,936 bytes. The declared
   EP72 per-rank inventory is 27,446,643,040
   bytes: 17,226,824,032 common bytes plus
   10,219,819,008 routed bytes. Static
   residency is not a per-step read count. The retained record has no HBM
   counter attribution and no routing trace, so neither side's actual bytes
   read per step is known.
4. **Other counterparts: identity mismatch decided, service undecidable.** The
   five vLLM routed rows above are capture-only physical identities and point
   downward if removed. Missing DeepEP dispatch/combine points upward. Exact
   services and overlap are absent, so neither is priced.

## Conditional arithmetic, rejected for calibration

For audit only, a frequency-only regrouping gives
**50794696000/3 ps**
(16,931,565,333 ps), predicting
**15119.688875 tokens/s/node**, a
movement of **5575.031079
tokens/s/node**. It is rejected because row 11 mixes attention, dense and
shared-expert quantization without per-launch durations; the vLLM launches are
not physically bound to SGLang EP72; DeepEP communication is missing; and
routing plus actual HBM reads were not captured. It is not the publication.

## Published signed movement

```text
CORE-65 prediction movement = 0.000000 tokens/s/node
final standard-decode prediction = 9544.657796 tokens/s/node
calibration anchor = 22282 tokens/s/node
signed difference = -12737.342204 tokens/s/node
signed residual movement = 0.000000 percentage points
final signed residual = -57.164268 percent
```

No parameter was fitted, no overlap term was added, and no held-out MTP value
was used or compared.

## Protocol and registry disposition

The committed reader made eight partial, contemporaneously logged accesses;
the optional original profile was logged unavailable without reading bytes.
However, two pre-reader incidents make the forbidden-access ledger nonempty:
one held-out numeric exposure, redacted and unused, and one unlogged broad
registry inspection. Literal CORE-65 closure is therefore impossible in this
worker even independently of the missing physical evidence.

CORE-65 remains open. CORE-66 is the exact hardware remainder: run the pinned
SGLang EP72 `b32/c2000`, MTP-disabled cell across all 72 ranks, capturing both
expert-residency rank classes, every launch and semantic correlation, routing
and physical slot identities, DeepEP payloads, and per-kernel HBM read/write
bytes. The exact configuration and command are in
`core66_hardware_remainder.json`.
