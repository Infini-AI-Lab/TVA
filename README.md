# TVA

TVA is a Lean 4 library for proving that Triton 3.8.0 kernels do what a fixed claim says, on an
NVIDIA H200 (sm_90).

A claim is written by hand once. It transcribes the kernel into TVA's `tl.*` vocabulary (the
*model*), says which launches it covers, and states what every covered launch leaves in memory. A
prover then proves `theorem main : claim.Holds`, and Lean checks the proof. What a reader of a
proved claim trusts is listed in one place, the module docstring of `lean/TVA/Claim.lean`: Lean and
Mathlib, the transcription, the compiler, and the device facts (each measured in `evidence/`).

## The parts

| Path | What |
|---|---|
| `lean/` | the library. `import TVA` is the trusted part: values, tensors, memory, the `tl.*` operations, launches, the device facts and what a claim means. `import TVA.Proof` is the proof kit, which is proved, not trusted. |
| `claims/` | 41 fixed claims, one folder per kernel, each with its oracle proof (`oracle/`, kept from provers); [claims/README.md](claims/README.md) says what a folder holds |
| `sandbox/` | the container a prover works in, and the tool that grades a proof; see [sandbox/README.md](sandbox/README.md) |
| `evidence/` | TVA's own measurements on the H200 behind the device facts |
| `tools/` | `make_index.py`, which generates `docs/index/` from the built library |
| `docs/` | [DESIGN.md](docs/DESIGN.md) (the contract), [SEMANTICS.md](docs/SEMANTICS.md) (what a model means), [PROVING.md](docs/PROVING.md) (how to prove a claim), [LIBRARY.md](docs/LIBRARY.md) (a map of `lean/TVA/`), [EXCLUSIONS.md](docs/EXCLUSIONS.md) (what no claim covers), [INDEX.md](docs/INDEX.md) (every definition and lemma, searchable) |
| `third_party/` | the licenses of the projects whose kernel sources the claims copy, and which claim comes from which; see [third_party/README.md](third_party/README.md) |

## Building

Lean 4 `v4.35.0` with Mathlib `v4.35.0` (`lean/lean-toolchain`, `lean/lakefile.toml`):

```bash
cd lean && lake build
```

When several agents share the machine, serialise builds as [DESIGN.md](docs/DESIGN.md) says.

## Checking one claim

A claim file must elaborate against the built library, including its `claim_satisfiable` proof:

```bash
cd lean && lake env lean ../claims/<kernel>/Claim.lean
```

A proof of a claim is checked by `sandbox/claim/tva-check` (the claim file unchanged, the proof
builds, `main : claim.Holds` has exactly that type, and only the standard axioms are used).

## The claims

| Kernel | Title |
|---|---|
| `attn_bwd_f16` | Attention backward, tutorial 06 (`_attn_bwd` with `_attn_bwd_dkdv` and `_attn_bwd_dq`, `float16`) |
| `attn_bwd_pre_f16` | Attention backward preprocess, tutorial 06 (`_attn_bwd_preprocess`, `float16` inputs, `float32` output) |
| `attn_fwd_f16` | Fused attention, tutorial 06 (`_attn_fwd`, float16, host tensor descriptors) |
| `block_scaled_matmul_mxfp8` | Block-scaled matmul, Triton tutorial 10 (`block_scaled_matmul_kernel`, mxfp8 and mixed: e4m3 by e4m3 or by e2m1 with e8m0 scales, float16 output) |
| `chunk_bwd_dv_local_bf16` | Intra-chunk value gradient of chunked (gated) linear attention (`chunk_bwd_kernel_dv_local`, bfloat16), fla |
| `chunk_cumsum_f32` | Chunk-local cumulative sum (`chunk_local_cumsum_scalar_kernel`, float32), fla |
| `chunk_delta_h_bf16` | Gated delta rule: the chunked state recurrence, forward (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, bfloat16), fla |
| `chunk_fwd_o_bf16` | Chunked linear-attention output (`chunk_fwd_kernel_o`, bfloat16, BK = BV of 32, 64 or 128), fla |
| `count_expert_tokens_i32` | MoE expert token counts, through an expert map (`_count_expert_num_tokens`, int32), vLLM |
| `extend_attn_bf16` | Extend attention over a paged KV cache, causal or with a target verification's tree mask, with or without a sliding window (`_fwd_kernel`, bfloat16), SGLang |
| `fused_moe_bf16` | Fused MoE expert GEMMs (both GEMMs of `fused_experts`, tokens sorted by expert or the naive decode assignment, the H200's tuned tiles or `get_default_config`'s; `fused_moe_kernel`, bfloat16), vLLM |
| `gdn_gating_bf16` | Gated DeltaNet gating, softplus decay and sigmoid (`fused_gdn_gating_kernel`, bfloat16), SGLang |
| `kv_cache_write_bf16` | KV cache write, flash layout (`reshape_and_cache_kernel_flash`, a bfloat16 or float8e4nv cache), vLLM |
| `kv_indices_i32` | KV indices for FlashInfer, gathered from `req_to_token`, with or without per-request start offsets (`create_flashinfer_kv_indices_triton`, int32), SGLang |
| `l2norm_fwd_bf16` | L2 normalisation, forward (`l2norm_fwd_kernel`, bfloat16 input, bfloat16 or float32 output), fla |
| `layer_norm_fwd_bf16` | LayerNorm and GroupNorm forward, with or without a residual (`layer_norm_fwd_kernel`, bfloat16), fla |
| `layer_norm_fwd_f16` | Layer normalization forward, tutorial 05 (`_layer_norm_fwd_fused`, float16) |
| `lora_shrink_bf16` | Multi-LoRA shrink, tokens grouped by adapter (`_lora_shrink_kernel` with `do_shrink_kernel`, bfloat16, split-K with `float32` atomics, one to several slices), vLLM |
| `matmul_e5m2` | Matrix multiplication, Triton tutorial 03 (`matmul_kernel`, float8e5 operands, float16 output) |
| `matmul_f16` | Matrix multiplication, Triton tutorial 03 (`matmul_kernel`, float16) |
| `merge_attn_states_bf16` | Merge of two attention states by their lse, with or without the merged lse (`merge_attn_states_kernel`, bfloat16), vLLM |
| `moe_router_bf16` | MoE router, soft cap and top-k by argmax (`fused_moe_router_cudacore_kernel`, bfloat16), SGLang |
| `moe_sum_reduce_bf16` | MoE sum-reduce, the top-k combine (`_moe_sum_reduce_kernel`, bfloat16), SGLang |
| `paged_attention_f16` | Paged attention, decode (`paged_attention_kernel`, float16) |
| `persistent_matmul_f16` | Persistent matrix multiplication, Triton tutorial 09 (`matmul_kernel_persistent`, float16) |
| `prefill_attn_varlen_bf16` | Prefill attention over packed variable-length sequences, causal or not (`_fwd_kernel`, bfloat16), SGLang |
| `recompute_wu_bf16` | WY representation of the gated DeltaNet's chunks, `w` and `u` (`recompute_w_u_fwd_kernel`, bfloat16), fla |
| `rejection_greedy_i32` | Greedy rejection sampling for speculative decoding, all-greedy or mixed batches (`rejection_greedy_sample_kernel`, int32), vLLM |
| `rotary_varlen_bf16` | Rotary position embedding, interleaved or not, over variable-length batches (`rotary_embedding_kernel`, bfloat16), fla |
| `route_histogram_i32` | LoRA route histogram, pairs per adapter and group (`_route_histogram_kernel`, `tl.histogram` and atomics, int32), SGLang |
| `scaled_mm_int8` | Scaled int8 GEMM, int32 accumulation, per-token/per-tensor and per-channel/per-tensor float32 scales, optional bias, bfloat16 or float16 output (`scaled_mm_kernel`), vLLM |
| `seeded_dropout_f32` | Seeded dropout, Triton tutorial 04 (`_seeded_dropout`, float32) |
| `softmax_f32` | Fused softmax, Triton tutorial 02 (`softmax_kernel`, float32) |
| `ssd_bmm_chunk_bf16` | Mamba-2 SSD per-chunk C·Bᵀ with sequence masking (`_bmm_chunk_fwd_kernel`, bfloat16 into float32), SGLang |
| `ssd_chunk_scan_bf16` | Mamba-2 SSD chunk scan of a packed prefill batch, with or without initial states (`_chunk_scan_fwd_kernel`, bfloat16), SGLang |
| `ssd_chunk_state_bf16` | Mamba-2 SSD chunk states of a packed prefill batch (`_chunk_state_fwd_kernel`, bfloat16 x and B, float32 states), SGLang |
| `ssd_state_passing_f32` | Mamba-2 SSD state passing across chunks of a packed prefill batch, with or without initial states (`_state_passing_fwd_kernel`, float32), SGLang |
| `streamk_matmul_f16` | Stream-K matrix multiplication (`streamk_matmul_kernel`, float16 into float32 with atomic adds) |
| `tma_persistent_matmul_f16` | Persistent matrix multiplication with host TMA descriptors, Triton tutorial 09 (`matmul_kernel_tma_persistent`, float16) |
| `vec_add_f32` | Vector addition, Triton tutorial 01 (`add_kernel`, float32) |
| `vocab_embedding_bf16` | Vocabulary-parallel embedding lookup (`_vocab_parallel_embedding_kernel`, bfloat16), SGLang |

## License

TVA is released under the [MIT License](LICENSE). The kernel sources copied into `claims/<k>/source/` keep the
licenses of the projects they come from: Triton and flash-linear-attention (MIT), SGLang and vLLM (Apache-2.0).
[third_party/](third_party/README.md) lists each claim's project and holds the license texts.
