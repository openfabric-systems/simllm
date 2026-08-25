# DeepSeek-V3 current-framework extraction suite

This authored-input suite pins `deepseek-ai/DeepSeek-V3` at revision
`e815299b0bcbac849fa540c768ef21845365c9eb`. The original V3 checkpoint is
the target because the DeepSeek production disclosure names that model and
both pinned framework registries bind `DeepseekV3ForCausalLM`. The
V3-0324 configuration has identical model and quantization fields. Its only
configuration difference is the recorded Transformers version. DeepSeek-R1
is outside this suite and is not substituted for the deployment checkpoint.

The weight identity comes only from Hugging Face API metadata. The suite
records all 163 shard names, per-shard SHA-256 digests and byte counts. Their
canonical manifest SHA-256 is
`ec8b878368c5fdb9f3288bd3a36a723a1637ec76464135a3f5b2e9aeff4072b4`,
and their total is 688,586,727,753 bytes. No weight shard is downloaded or
opened.

The first 15 text-only cells preserve the established prefill and decode
grid. Five additional cells carry the SGLang disclosure shapes:

- three prefill cells each process 16,384 tokens per GPU at prompt lengths
  1,024, 2,048 and 4,096 under expert parallelism 32;
- standard decode projects the disclosed per-node batch 256 to batch 32 per
  GPU with context 2,000 under expert parallelism 72;
- simulated multi-token prediction projects per-node batch 128 to batch 16
  per GPU with context 4,000 and sets `mtp_enabled`.

The four deployment projections name the SGLang prefill and decode units and
the DeepSeek production prefill expert-parallelism-32 and decode
expert-parallelism-144 units. Every unit has 256 unique experts and 32
redundant physical slots. Redundant slots replicate residency but never
create new logical work. DeepSeek did not disclose batch or context shapes
for its production units, so those projections reuse authored phase cells
for static shape accounting and do not invent a throughput workload.

The suite contains no extracted record, inventory, duration, measured value,
physical code object or observed launch. Its file SHA-256 is
`88f718c94ad35bb0c74314811680b5ebff7e5df70759096dc0b640f84f47bd69`.
