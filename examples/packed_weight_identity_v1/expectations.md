# Packed weight identity expectations

COMP-91 repairs the metadata validator's assumption that every quantized
parameter occupies at least one byte. The explicitly named format
`mxfp4-e2m1-group32-e8m0` packs two four-bit values per byte and supplies one
eight-bit scale per 32 values. This statement is frozen before implementation
and the synthetic study. Earlier diagnosis is post-specified, not this study.

The [pinned Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)
declares that packing under `text_config.quantization_config`, with selected
modules excluded. The format is also distinguished from NVFP4 in NVIDIA's
[format comparison](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/).
No checkpoint weights are downloaded or claimed byte-verified.

## Physical bound and sweep

For P positive weight values, packed payload needs at least ceil(P/2) bytes
and at least ceil(P/32) scale bytes. The floor is their sum. The bound permits
a compact partial final group; actual block padding, tensor boundaries,
unquantized parameters and file metadata can increase storage. Each replacement
of a packed parameter with a value of at least two bytes preserves this lower
bound. Parameter count excludes the additional scale values themselves.

Floor: no envelope for this declared format can contain fewer than the packed
value bytes plus the minimum scale bytes. Ceiling: the synthetic envelope in
this study is exactly that floor plus its declared container overhead, not a
ceiling on a real checkpoint.

Sweep P over 31, 32, 33, 512 and 1024 and container overhead over 0 and 48 bytes.
Use explicit whole groups and a partial tail as an independent byte oracle.
At each P, an envelope one byte below the floor rejects; the exact floor and
the floor plus the declared overhead accept. Doubling a complete-group P
doubles its payload and scale terms exactly. Adding container bytes changes
the recorded file total by exactly that amount without changing the payload
floor. No timing or high-bandwidth-memory traffic is inferred from these bytes.

## Interface and controls

Exercise `load_extraction_suite`, its metadata manifest verification and
`ModelCheckpointIdentity` canonical projection with synthetic identity records.
Retain all existing shard name, digest, size-total, ordering and metadata-only
disclosures. A checksum mismatch, malformed count or incorrect local-verification
claim still rejects. Existing BF16 and FP8 suites and inventory bytes remain
exact; unnamed or unsupported formats retain their existing conservative
one-byte floor. Only the precise new format declaration enables packed admission.

The generic compute projection still rejects this packed format. A parser's
valid metadata identity is not a complete K3 execution graph. COMP-54 owns its
mixed layer, format, latent expert and residual-bank structure; COMP-59 owns
the separate reduced-depth physical envelope. Neither closes here.

Report configurations, exact byte oracles, positive scaling relations and fatal
guards separately. Rejection, metadata preservation and off-path behavior are
unscored guards. Any violated guard voids the study and removes its behavioral
score. Preserve its evidence. Close only COMP-91's false-rejection defect.
