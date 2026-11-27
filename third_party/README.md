# Third-party code

58 of the 60 claims copy their kernel's source file from another project into `claims/<k>/source/`.
The kernels are unchanged, except `extend_attn_bf16`'s; where a host wrapper, an autotuner, an import
or a comment was adapted, the source's module docstring and the claim's `SETUP.md` list each change. Each claim's
`SETUP.md` gives the upstream path and line numbers. These files keep their projects' licenses, copied here:

| Project | Version | License | Claims |
|---|---|---|---|
| [Triton](https://github.com/triton-lang/triton) | 3.8.0, `python/tutorials/` | MIT, [LICENSE-triton](LICENSE-triton) | `attn_bwd_f16`, `attn_bwd_pre_f16`, `attn_fwd_f16`, `block_scaled_matmul_mxfp8`, `layer_norm_fwd_f16`, `matmul_e5m2`, `matmul_f16`, `persistent_matmul_f16`, `seeded_dropout_f32`, `softmax_f32`, `tma_persistent_matmul_f16`, `vec_add_f32` |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | commit `413ad2c655` | MIT, [LICENSE-flash-linear-attention](LICENSE-flash-linear-attention) | `chunk_bwd_dv_local_bf16`, `chunk_cumsum_f32`, `chunk_delta_h_bf16`, `chunk_fwd_o_bf16`, `chunk_global_cumsum_f32`, `l2norm_bwd_bf16`, `l2norm_fwd_bf16`, `layer_norm_fwd_bf16`, `layer_norm_gated_fwd_bf16`, `logsigmoid_fwd_bf16`, `logsumexp_fwd_f32`, `mean_pooling_fwd_bf16`, `recompute_wu_bf16`, `rotary_varlen_bf16`, `swiglu_fwd_bf16` |
| [SGLang](https://github.com/sgl-project/sglang) | v0.5.23, commit `297e54fd10` | Apache-2.0, [LICENSE-sglang](LICENSE-sglang) | `compute_position_i64`, `ep_src2dst_i32`, `extend_attn_bf16`, `gdn_gating_bf16`, `kv_indices_i32`, `moe_router_bf16`, `moe_sum_reduce_bf16`, `num_kv_splits_i32`, `prefill_attn_varlen_bf16`, `route_histogram_i32`, `ssd_bmm_chunk_bf16`, `ssd_chunk_scan_bf16`, `ssd_chunk_state_bf16`, `ssd_state_passing_f32`, `token_bitmask_f32`, `vocab_embedding_bf16`, `write_req_to_token_pool_i32` |
| [vLLM](https://github.com/vllm-project/vllm) | commit `4ba787ec99` | Apache-2.0, [LICENSE-vllm](LICENSE-vllm) | `combine_draft_tokens_i32`, `count_expert_tokens_i32`, `fused_moe_bf16`, `gelu_and_mul_sparse_bf16`, `kv_cache_write_bf16`, `logprob_ranks_i64`, `lora_shrink_bf16`, `merge_attn_states_bf16`, `prepare_pos_seq_lens_i32`, `rejection_expand_f32`, `rejection_greedy_i32`, `scaled_mm_int8`, `slot_mappings_i64`, `topk_log_softmax_f32` |

`paged_attention_f16` and `streamk_matmul_f16` were written for TVA and fall under its [MIT License](../LICENSE).
