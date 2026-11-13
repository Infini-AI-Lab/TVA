# Third-party code

39 of the 41 claims copy their kernel's source file from another project into `claims/<k>/source/`.
The files are unchanged, except `extend_attn_bf16`'s, whose docstring lists each change. Each claim's
`SETUP.md` gives the upstream path and line numbers. These files keep their projects' licenses, copied here:

| Project | Version | License | Claims |
|---|---|---|---|
| [Triton](https://github.com/triton-lang/triton) | 3.8.0, `python/tutorials/` | MIT, [LICENSE-triton](LICENSE-triton) | `attn_bwd_f16`, `attn_bwd_pre_f16`, `attn_fwd_f16`, `block_scaled_matmul_mxfp8`, `layer_norm_fwd_f16`, `matmul_e5m2`, `matmul_f16`, `persistent_matmul_f16`, `seeded_dropout_f32`, `softmax_f32`, `tma_persistent_matmul_f16`, `vec_add_f32` |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | commit `413ad2c655` | MIT, [LICENSE-flash-linear-attention](LICENSE-flash-linear-attention) | `chunk_bwd_dv_local_bf16`, `chunk_cumsum_f32`, `chunk_delta_h_bf16`, `chunk_fwd_o_bf16`, `l2norm_fwd_bf16`, `layer_norm_fwd_bf16`, `recompute_wu_bf16`, `rotary_varlen_bf16` |
| [SGLang](https://github.com/sgl-project/sglang) | v0.5.23, commit `297e54fd10` | Apache-2.0, [LICENSE-sglang](LICENSE-sglang) | `extend_attn_bf16`, `gdn_gating_bf16`, `kv_indices_i32`, `moe_router_bf16`, `moe_sum_reduce_bf16`, `prefill_attn_varlen_bf16`, `route_histogram_i32`, `ssd_bmm_chunk_bf16`, `ssd_chunk_scan_bf16`, `ssd_chunk_state_bf16`, `ssd_state_passing_f32`, `vocab_embedding_bf16` |
| [vLLM](https://github.com/vllm-project/vllm) | commit `4ba787ec99` | Apache-2.0, [LICENSE-vllm](LICENSE-vllm) | `count_expert_tokens_i32`, `fused_moe_bf16`, `kv_cache_write_bf16`, `lora_shrink_bf16`, `merge_attn_states_bf16`, `rejection_greedy_i32`, `scaled_mm_int8` |

`paged_attention_f16` and `streamk_matmul_f16` were written for TVA and fall under its [MIT License](../LICENSE).
