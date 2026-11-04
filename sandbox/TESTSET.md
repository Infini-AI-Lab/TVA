# The sandbox's kernel test set

Twenty-five Triton kernels from three serving and training code bases, each to be proved and reviewed
in the sandbox ([README.md](README.md)) as one task: `make_task.py <kernel>`, one prover, two
reviewers, the same time limit. The twelve kernels in `kernels/` come from Triton's tutorials; this
set asks what the protocol does on the kernels that real systems run, and how a model's result
depends on what a kernel does.

Sources, pinned:

| Code base | Version | Commit |
|---|---|---|
| vLLM | v0.33.0 | `4ba787ec99` |
| SGLang | v0.5.23 | `297e54fd10` |
| flash-linear-attention (fla) | v0.6.0 | `413ad2c655` |

## How the kernels were chosen

1. **One functionality each.** The set covers what a serving step does, end to end: activations,
   norms, positions, losses, quantisation, the KV cache and its indices, attention (prefill, decode,
   merge), MoE routing and expert GEMMs, LoRA, sampling and grammar masks, and the linear-attention
   and state-space layers. No two kernels do the same job; none repeats a tutorial kernel's.
2. **In TVA's language today.** Every kernel uses only forms [SEMANTICS.md](../docs/SEMANTICS.md)
   gives a meaning, so its task needs no library change. Kernels were screened out for:
   - updating an input in place, or a program writing and reading back the same bytes (excluded,
     [exclusions.md](../docs/exclusions.md)): lightning attention's `_fwd_none_diag_kernel` and
     `_fwd_kv_reduce`, vLLM's `_penalties_kernel` and `_triton_mrope_forward`, SGLang's
     `apply_token_bitmask_inplace_kernel` (it writes `-inf` into the caller's logits) and
     `_moe_sorting_small_kernel_sorted` (which also uses `tl.gather`, not in the library). Every kernel
     of the set writes only buffers it does not read, and its host allocates them fresh, except the
     KV-cache write (11), which stores into the caller's cache without reading it;
   - `tl.inline_asm_elementwise`, and libdevice functions the library does not name (vLLM's
     `_per_token_quant_int8` rounds with `libdevice.round`);
   - functions and forms the library gives no meaning, on the path launched: `tl.clamp` and
     `tl.ceil` (every fp8 and int8 quantisation kernel of the three code bases clamps, so none is
     in the set: vLLM's `_per_token_group_quant_fp8` waits for the library to give `tl.clamp` a
     meaning), a dtype's `is_fp8()` (vLLM's decode kernels test it; SGLang's copy is taken), and
     jit options other than `do_not_specialize` (vLLM's `_state_passing_fwd_kernel` has
     `do_not_specialize_on_alignment`; SGLang's copy is taken);
   - work counters, spin locks and flags passed between programs.
3. **Four levels**, L0 (very easy) to L3 (difficult), by the proof expected, not by line count. They
   cover the tutorial kernels too ([Levels](#levels)), so a result on the set reads against kernels
   already certified. A model's result is then a curve, not one number.
4. **Known findings as controls.** Some kernels carry a finding verified on a GPU in an earlier
   check of these code bases. There, the honest outcome is a claim whose conditions exclude the
   failure, stated as a narrowing in README.md, or a write-up that names it. A claim that covers it
   is a soundness failure of the run, whatever the certificate says. Which kernels these are, and
   the findings, are kept with the run records outside TVA: a task's image holds TVA's tree, this
   file included, and neither the prover nor the reviewers may know them.

## The set

Kernel names follow `kernels/`: the operation, then the data type of the configuration proved.
Source paths are relative to the package: `vllm/`, `python/sglang/`, `fla/`.

| # | Kernel (TVA name) | Source | Lines | What it does | Level |
|---|---|---|---|---|---|
| 1 | `l2norm_fwd_bf16` | fla `modules/norm/l2norm.py:69` `l2norm_fwd_kernel` | 16 | one row's L2 normalisation (`1 / sqrt`) | L1 |
| 2 | `act_and_mul_bf16` | SGLang `kernels/ops/moe/fused_moe_triton_kernels.py:1141` `act_and_mul_kernel` | 49 | SiLU or GELU of the gate times the up projection, the MoE activation | L0 |
| 3 | `layer_norm_fwd_bf16` | fla `modules/norm/layernorm.py:48` `layer_norm_fwd_kernel` | 64 | LayerNorm or RMSNorm with a residual added and written out, and the mean and `rstd` saved | L1 |
| 4 | `vocab_embedding_bf16` | SGLang `kernels/ops/embeddings/vocab_parallel_embedding.py:9` `_vocab_parallel_embedding_kernel` | 45 | a tensor-parallel shard's embedding lookup: rows outside the shard's vocabulary are zero | L0 |
| 5 | `moe_sum_reduce_bf16` | SGLang `kernels/ops/moe/fused_moe_triton_kernels.py:1238` `_moe_sum_reduce_kernel` | 52 | the MoE combine: each token's top-k expert outputs summed in `float32` and scaled | L1 |
| 6 | `route_histogram_i32` | SGLang `kernels/ops/lora/common/routing.py:105` `_route_histogram_kernel` | 35 | counts per adapter or expert: `tl.histogram` in a program, `tl.atomic_add` across programs | L1 |
| 7 | `count_expert_tokens_i32` | vLLM `model_executor/layers/fused_moe/utils.py:171` `_count_expert_num_tokens` | 29 | tokens routed to each expert, through an expert map (expert parallelism) | L0 |
| 8 | `gdn_gating_bf16` | SGLang `kernels/ops/attention/linear/fused_gdn_gating.py:11` `fused_gdn_gating_kernel` | 31 | gated DeltaNet's gates: the decay `−exp(A_log)·softplus(a + dt_bias)` (with softplus's threshold) and `β = sigmoid(b)` | L0 |
| 9 | `rotary_varlen_bf16` | fla `modules/rotary/ops.py:31` `rotary_embedding_kernel` | 91 | rotary positions, interleaved or not, with sequence offsets and packed variable-length batches | L1 |
| 10 | `cross_entropy_fwd_bf16` | fla `modules/fused_cross_entropy/ops.py:27` `cross_entropy_fwd_kernel` | 57 | the loss over a row split in blocks: the block's logsumexp, label smoothing, `ignore_index` | L2 |
| 11 | `kv_cache_write_bf16` | vLLM `v1/attention/ops/triton_reshape_and_cache_flash.py:33` `reshape_and_cache_kernel_flash` | 122 | new keys and values scattered into the paged cache by slot (`-1` is padding), either layout | L0 |
| 12 | `kv_indices_i32` | SGLang `kernels/ops/attention/kv_indices.py:59` `create_flashinfer_kv_indices_triton` | 81 | a request's KV token indices gathered from `req_to_token` into a flat list, by page | L2 |
| 13 | `merge_attn_states_bf16` | vLLM `v1/attention/ops/triton_merge_attn_states.py:73` `merge_attn_states_kernel` | 151 | two partial attention outputs merged by their log-sum-exps (prefix and suffix), with `inf` cases | L2 |
| 14 | `rejection_greedy_i32` | vLLM `v1/sample/rejection_sampler.py:793` `rejection_greedy_sample_kernel` | 55 | speculative decoding: draft tokens accepted while they match the target's argmax, then the bonus token | L2 |
| 15 | `chunk_cumsum_f32` | fla `ops/utils/cumsum.py:34` `chunk_local_cumsum_scalar_kernel` | 40 | the cumulative sum within each chunk, forward or reverse, optionally scaled, over variable-length sequences | L2 |
| 16 | `causal_conv1d_fwd_bf16` | fla `modules/causal_conv1d/ops.py:51` `causal_conv1d_fwd_kernel` | 97 | depthwise causal convolution with initial states, bias and SiLU, over variable-length sequences | L2 |
| 17 | `moe_router_bf16` | SGLang `kernels/ops/moe/router.py:13` `fused_moe_router_cudacore_kernel` | 100 | the router: hidden state times the gate weights, softmax, top-k by repeated argmax, renormalised weights | L2 |
| 18 | `block_fp8_matmul` | vLLM `model_executor/layers/quantization/utils/fp8_utils.py:841` `_w8a8_triton_block_scaled_mm` | 81 | `e4m3` GEMM with per-token-group scales for A and 128×128 block scales for B (DeepSeek's format) | L2 |
| 19 | `fused_moe_bf16` | vLLM `model_executor/layers/fused_moe/fused_moe.py:298` `fused_moe_kernel` | 311 | the expert GEMM over tokens sorted by expert, padded blocks skipped, top-k weights applied | L3 |
| 20 | `lora_shrink_bf16` | vLLM `lora/ops/triton_ops/lora_shrink_op.py:102` `_lora_shrink_kernel` (with `kernel_utils.do_shrink_kernel`) | 115 | multi-LoRA: each token's rows times its adapter's A, grouped by adapter (split-K adds with relaxed atomics) | L3 |
| 21 | `prefill_attn_varlen_bf16` | SGLang `kernels/ops/attention/prefill_attention.py:35` `_fwd_kernel` | 133 | causal attention over packed variable-length prompts (`b_start_loc`, `b_seq_len`) | L3 |
| 22 | `decode_attn_gqa_bf16` | SGLang `kernels/ops/attention/decode_attention.py:530` `_fwd_grouped_kernel_stage1` | 200 | decode attention for grouped queries over a paged cache, split along the KV length: partial outputs and their lse, logit cap by `tanh` | L3 |
| 23 | `chunk_kkt_bf16` | fla `ops/common/chunk_scaled_dot_kkt.py:33` `chunk_scaled_dot_kkt_fwd_kernel` | 48 | within each chunk, `β·K·Kᵀ` strictly lower triangular, gated by `exp(g_i − g_j)` (DeltaNet's WY form) | L2 |
| 24 | `gated_delta_rule_recurrent_bf16` | fla `ops/gated_delta_rule/fused_recurrent.py:30` `fused_recurrent_gated_delta_rule_fwd_kernel` | 152 | the gated delta rule token by token: the state decays, is corrected by `β(v − Sᵀk)k`, and read out; the final state saved | L3 |
| 25 | `ssd_state_passing_f32` | SGLang `kernels/ops/mamba/triton_ops/ssd_state_passing.py:34` `_state_passing_fwd_kernel` | 95 | Mamba-2: chunk states carried across chunks with `exp(dA)` decay, reset at sequence boundaries | L3 |

By code base: vLLM 7, SGLang 10, fla 8. By level (L0 / L1 / L2 / L3): 5 / 5 / 9 / 6.

## Levels

Every kernel in `kernels/` has a level, by the proof expected: what the control flow asks
(one pass, a loop, several programs or stages) and what the numbers ask (exact, one bound, a bound
carried through a loop or a chain of `tl.dot`s and rescalings). Where a kernel has two of a level's
marks, or marks of two levels, it takes the higher level.

| Level | What marks it | Anchor (sandbox pilots, certified proof) |
|---|---|---|
| L0, very easy | One pass, elementwise or a gather or scatter; the result exact (copies, integers) or a few float operations per element; no reduction | `vec_add_f32`: certified in round 1 (pilot 1); 321 lines of Lean |
| L1, easy | One reduction per row or block (sum, max, `tl.histogram`), integer atomics whose order does not matter, or elementwise work behind nontrivial index logic (offsets, two layouts) | `softmax_f32`: certified in round 1 (pilot 2); 1 553 lines |
| L2, medium | A loop that carries a float bound (a `tl.dot` over K, a running cumulative sum or convolution), indirection with packed variable lengths, case splits a claim must state (`inf`, fully masked rows, ignored labels), sequential integer logic, top-k by repeated argmax | `matmul_f16`: 1 464 lines, 3 rounds |
| L3, difficult | An online softmax, two or more `tl.dot`s chained through a rescaling, a recurrence over time or chunks, a GEMM behind indirection (sorted tokens, adapters), float atomics across programs | `paged_attention_f16`: no submission in six pilots; `attn_fwd_e5m2`: 7 646 lines, 7 rounds |

| Level | Tutorial kernels (certified) | Test set (number in [The set](#the-set)) |
|---|---|---|
| L0 | `vec_add_f32` | `act_and_mul_bf16` (2), `vocab_embedding_bf16` (4), `count_expert_tokens_i32` (7), `gdn_gating_bf16` (8), `kv_cache_write_bf16` (11) |
| L1 | `softmax_f32`, `attn_bwd_pre_f16` | `l2norm_fwd_bf16` (1), `layer_norm_fwd_bf16` (3), `moe_sum_reduce_bf16` (5), `route_histogram_i32` (6), `rotary_varlen_bf16` (9) |
| L2 | `matmul_f16`, `matmul_e5m2`, `block_scaled_matmul_mxfp8`, `grouped_matmul_tma_f16` | `cross_entropy_fwd_bf16` (10), `kv_indices_i32` (12), `merge_attn_states_bf16` (13), `rejection_greedy_i32` (14), `chunk_cumsum_f32` (15), `causal_conv1d_fwd_bf16` (16), `moe_router_bf16` (17), `block_fp8_matmul` (18), `chunk_kkt_bf16` (23) |
| L3 | `streamk_matmul_f16`, `attn_fwd_f16`, `attn_fwd_e5m2`, `attn_bwd_f16`, `paged_attention_f16` | `fused_moe_bf16` (19), `lora_shrink_bf16` (20), `prefill_attn_varlen_bf16` (21), `decode_attn_gqa_bf16` (22), `gated_delta_rule_recurrent_bf16` (24), `ssd_state_passing_f32` (25) |

All kernels: 6 / 7 / 13 / 11. The borderline calls, which a run's results may move:

- `kv_cache_write_bf16` is L0 though it has two layouts and scatters by slot: every value is a copy,
  and that programs write distinct slots is a condition of the claim, not a proof.
- `gdn_gating_bf16` and `act_and_mul_bf16` are L0 though their math goes beyond `exp`: the bound is
  per element, through a fixed chain of functions.
- `route_histogram_i32` is L1, not L0, for its atomics across programs, whose sum is exact in any order.
- `moe_router_bf16` is L2, not L3: its `tl.dot` is one GEMV per token, and the hard part, the top-k
  by repeated argmax with its ties, is integer logic.
- `chunk_kkt_bf16` is L2: one `tl.dot` within a chunk, then a gate and a mask, with no carry between
  chunks.
- `lora_shrink_bf16` and `fused_moe_bf16` are L3: each is an L2 GEMM behind indirection (adapters,
  tokens sorted by expert, padded blocks skipped), which the claim must describe.
- `ssd_state_passing_f32` is L3 for its recurrence across chunks, reset at sequence boundaries,
  whose bound grows with the number of chunks.

## What the set exercises that the tutorials do not

| Feature | Kernels |
|---|---|
| packed variable-length batches (`cu_seqlens`, start offsets) | 9, 12, 15, 16, 21, 24, 25 |
| indirection: a load's address comes from another load (slots, page tables, token ids, adapter ids) | 4, 7, 11, 12, 19, 20, 22 |
| atomics across programs | 6, 20 (split-K, not in the configuration proved) |
| `tl.histogram`, `tl.cumsum`, `tl.argmax` | 6; 15; 17 |
| fp8 inputs and scales | 18 (and the fp8 paths of 11 and 19, not proved) |
| infinite inputs and fully masked rows a claim must state | 13, 22 |
| recurrences over time or chunks | 16, 24, 25 |
| math beyond `exp` (`sqrt`, `sigmoid` and the `tanh` built from it, `log`, softplus) | 1, 2, 3, 8, 10, 22 |
| integer-only results (indices, counts, accepted tokens) | 6, 7, 12, 14 |

## Preparing a kernel

Each kernel becomes a folder `kernels/<name>/` with no proof yet, as `make_task.py` needs, holding:

- `source/<file>.py`: the kernel verbatim. Next to it go:
  - the `@triton.jit` helpers it calls, copied from their modules (fla's `exp`, `softplus`,
    `exp2`; vLLM's `do_shrink_kernel`, `fast_exp`), since TVA inlines only helpers defined in the
    kernel's file;
  - a host function taken from the code base's wrapper.

  Each change from the source is listed in a comment at the top, with the source's
  `file:line`. Every kernel keeps its own lines and line numbers inside the file.
- **Autotuning and heuristics removed.** Remove `@triton.autotune`, fla's `fla_cache_autotune` and
  `@triton.heuristics`: TVA verifies none of them. The host passes one configuration from the
  source's list as explicit launch arguments, and sets the flags the heuristics computed. This
  changes no line of the kernel's body.
- **Parameter defaults and module values removed.** TVA's subset has no default for a kernel's
  parameter and no value at the top level of the module. The kernel's signature loses its defaults
  (the host passes every argument, so no launch changes), and the module's constants (`_is_hip`,
  dtype tables, `tl.constexpr` globals the host reads) move into the host. Code in a compile-time
  branch the launch never takes has no meaning and stays, even where it names a function the file
  no longer imports (fla's `tanh` from libdevice under softcapping).
- **One configuration proved.** A kernel with `constexpr` paths (dtypes, fp8 or not, split-K, softcap)
  keeps them all in its source, and `SETUP.md` names the configuration the task proves: the one the
  code base runs most (bf16 activations, the default flags). Kernel 20 is proved with `SPLIT_K = 1`
  unless its relaxed atomics are shown ordered, and kernel 10 with softcapping off, whose `tanh` is
  libdevice's. A narrower claim is then the prover's to state, as for any kernel.
- `SETUP.md`: the shapes, dtypes and launch the code base uses (from its tests and model configs),
  as in `kernels/paged_attention_f16/SETUP.md`.
- An entry in `make_task.py`'s `EXAMPLES`: the three closest certified kernels.

  | Kernels | Examples |
  |---|---|
  | elementwise and row kernels | `vec_add_f32`, `softmax_f32` |
  | GEMMs | `matmul_f16`, `block_scaled_matmul_mxfp8`, `grouped_matmul_tma_f16` |
  | attention | `attn_fwd_f16`, `paged_attention_f16` |

- **The kernel's commit.** Commit the folders, since a task's image holds TVA at a commit. The kernels
  index (`kernels/README.md`) lists them as having no claim yet.

The controls' findings stay out of TVA: they are kept with the run records, outside the tasks, and
read only when a run's result is judged.

## Running the set

- **One task per kernel**, with the pilots' settings: DeepSeek V4.1 Flash 1017 for the prover and
  both reviewers, 8 h, a 650K window, metered with a cap per run.
- **By level, in waves of five.** Wave 1 is the five L0 kernels: a run that fails there points at
  the harness or the model, not at the kernel. The machine ran two pilots beside each other without
  contention.
  Lean builds dominate the CPU, so five at once is the first try, and the first wave's build times
  (in the transcripts) tell whether to go higher.
- **Cost.** The paged pilots cost $6–7 each, and softmax and vec_add $1–3. Expect about $2 per
  L0 or L1 kernel, $4 per L2 and $8 per L3, about $105 per sweep of one model. A cap of
  $15 per run bounds it at $375.
- **What is recorded per run:**
  - reward and certificate;
  - time to the first submission;
  - review rounds and findings by kind;
  - the claim's narrowings;
  - for the controls, whether the claim covers the known failure;
  - spend (`meter.py report`).

  Each row goes in a results table next to the run records.
