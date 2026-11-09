# Prepared claims

One folder per kernel, `claims/<kernel>/`, holding the claim an agent is asked to prove: the model, the
conditions, the specification and the assumptions, fixed in advance. The agent proves the claim as it is
stated; it may not narrow the conditions or weaken the specification. Each folder is self-contained: it holds
the kernel's source and setup as well as the claim.

These are the claims for kernels believed correct on every launch their `SETUP.md` asks to cover.

## The claims

41 kernels: Triton 3.8.0's tutorials, kernels of vLLM v0.33.0, SGLang v0.5.23 and fla v0.6.0, and two written
for TVA (`streamk_matmul_f16`, `paged_attention_f16`), each with the launches its `SETUP.md` names.

| Claim | Kernel |
|---|---|
| [`attn_bwd_f16`](attn_bwd_f16/README.md) | Attention backward, tutorial 06 (`_attn_bwd` with `_attn_bwd_dkdv` and `_attn_bwd_dq`, `float16`) |
| [`attn_bwd_pre_f16`](attn_bwd_pre_f16/README.md) | Attention backward preprocess, tutorial 06 (`_attn_bwd_preprocess`, `float16` inputs, `float32` output) |
| [`attn_fwd_f16`](attn_fwd_f16/README.md) | Fused attention, tutorial 06 (`_attn_fwd`, float16, host tensor descriptors) |
| [`block_scaled_matmul_mxfp8`](block_scaled_matmul_mxfp8/README.md) | Block-scaled matmul, Triton tutorial 10 (`block_scaled_matmul_kernel`, mxfp8 and mixed: e4m3 by e4m3 or by e2m1 with e8m0 scales, float16 output) |
| [`chunk_bwd_dv_local_bf16`](chunk_bwd_dv_local_bf16/README.md) | Intra-chunk value gradient of chunked (gated) linear attention (`chunk_bwd_kernel_dv_local`, bfloat16), fla |
| [`chunk_cumsum_f32`](chunk_cumsum_f32/README.md) | Chunk-local cumulative sum (`chunk_local_cumsum_scalar_kernel`, float32), fla |
| [`chunk_delta_h_bf16`](chunk_delta_h_bf16/README.md) | Gated delta rule: the chunked state recurrence, forward (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, bfloat16), fla |
| [`chunk_fwd_o_bf16`](chunk_fwd_o_bf16/README.md) | Chunked linear-attention output (`chunk_fwd_kernel_o`, bfloat16, BK = BV of 32, 64 or 128), fla |
| [`count_expert_tokens_i32`](count_expert_tokens_i32/README.md) | MoE expert token counts, through an expert map (`_count_expert_num_tokens`, int32), vLLM |
| [`extend_attn_bf16`](extend_attn_bf16/README.md) | Extend attention over a paged KV cache, causal or with a target verification's tree mask, with or without a sliding window (`_fwd_kernel`, bfloat16), SGLang |
| [`fused_moe_bf16`](fused_moe_bf16/README.md) | Fused MoE expert GEMMs (both GEMMs of `fused_experts`, tokens sorted by expert or the naive decode assignment, the H200's tuned tiles or `get_default_config`'s; `fused_moe_kernel`, bfloat16), vLLM |
| [`gdn_gating_bf16`](gdn_gating_bf16/README.md) | Gated DeltaNet gating, softplus decay and sigmoid (`fused_gdn_gating_kernel`, bfloat16), SGLang |
| [`kv_cache_write_bf16`](kv_cache_write_bf16/README.md) | KV cache write, flash layout (`reshape_and_cache_kernel_flash`, a bfloat16 or float8e4nv cache), vLLM |
| [`kv_indices_i32`](kv_indices_i32/README.md) | KV indices for FlashInfer, gathered from `req_to_token`, with or without per-request start offsets (`create_flashinfer_kv_indices_triton`, int32), SGLang |
| [`l2norm_fwd_bf16`](l2norm_fwd_bf16/README.md) | L2 normalisation, forward (`l2norm_fwd_kernel`, bfloat16 input, bfloat16 or float32 output), fla |
| [`layer_norm_fwd_bf16`](layer_norm_fwd_bf16/README.md) | LayerNorm and GroupNorm forward, with or without a residual (`layer_norm_fwd_kernel`, bfloat16), fla |
| [`layer_norm_fwd_f16`](layer_norm_fwd_f16/README.md) | Layer normalization forward, tutorial 05 (`_layer_norm_fwd_fused`, float16) |
| [`lora_shrink_bf16`](lora_shrink_bf16/README.md) | Multi-LoRA shrink, tokens grouped by adapter (`_lora_shrink_kernel` with `do_shrink_kernel`, bfloat16, split-K with `float32` atomics, one to several slices), vLLM |
| [`matmul_e5m2`](matmul_e5m2/README.md) | Matrix multiplication, Triton tutorial 03 (`matmul_kernel`, float8e5 operands, float16 output) |
| [`matmul_f16`](matmul_f16/README.md) | Matrix multiplication, Triton tutorial 03 (`matmul_kernel`, float16) |
| [`merge_attn_states_bf16`](merge_attn_states_bf16/README.md) | Merge of two attention states by their lse, with or without the merged lse (`merge_attn_states_kernel`, bfloat16), vLLM |
| [`moe_router_bf16`](moe_router_bf16/README.md) | MoE router, soft cap and top-k by argmax (`fused_moe_router_cudacore_kernel`, bfloat16), SGLang |
| [`moe_sum_reduce_bf16`](moe_sum_reduce_bf16/README.md) | MoE sum-reduce, the top-k combine (`_moe_sum_reduce_kernel`, bfloat16), SGLang |
| [`paged_attention_f16`](paged_attention_f16/README.md) | Paged attention, decode (`paged_attention_kernel`, float16) |
| [`persistent_matmul_f16`](persistent_matmul_f16/README.md) | Persistent matrix multiplication, Triton tutorial 09 (`matmul_kernel_persistent`, float16) |
| [`prefill_attn_varlen_bf16`](prefill_attn_varlen_bf16/README.md) | Prefill attention over packed variable-length sequences, causal or not (`_fwd_kernel`, bfloat16), SGLang |
| [`recompute_wu_bf16`](recompute_wu_bf16/README.md) | WY representation of the gated DeltaNet's chunks, `w` and `u` (`recompute_w_u_fwd_kernel`, bfloat16), fla |
| [`rejection_greedy_i32`](rejection_greedy_i32/README.md) | Greedy rejection sampling for speculative decoding, all-greedy or mixed batches (`rejection_greedy_sample_kernel`, int32), vLLM |
| [`rotary_varlen_bf16`](rotary_varlen_bf16/README.md) | Rotary position embedding, interleaved or not, over variable-length batches (`rotary_embedding_kernel`, bfloat16), fla |
| [`route_histogram_i32`](route_histogram_i32/README.md) | LoRA route histogram, pairs per adapter and group (`_route_histogram_kernel`, `tl.histogram` and atomics, int32), SGLang |
| [`scaled_mm_int8`](scaled_mm_int8/README.md) | Scaled int8 GEMM, int32 accumulation, per-token/per-tensor and per-channel/per-tensor float32 scales, optional bias, bfloat16 or float16 output (`scaled_mm_kernel`), vLLM |
| [`seeded_dropout_f32`](seeded_dropout_f32/README.md) | Seeded dropout, Triton tutorial 04 (`_seeded_dropout`, float32) |
| [`softmax_f32`](softmax_f32/README.md) | Fused softmax, Triton tutorial 02 (`softmax_kernel`, float32) |
| [`ssd_bmm_chunk_bf16`](ssd_bmm_chunk_bf16/README.md) | Mamba-2 SSD per-chunk C·Bᵀ with sequence masking (`_bmm_chunk_fwd_kernel`, bfloat16 into float32), SGLang |
| [`ssd_chunk_scan_bf16`](ssd_chunk_scan_bf16/README.md) | Mamba-2 SSD chunk scan of a packed prefill batch, with or without initial states (`_chunk_scan_fwd_kernel`, bfloat16), SGLang |
| [`ssd_chunk_state_bf16`](ssd_chunk_state_bf16/README.md) | Mamba-2 SSD chunk states of a packed prefill batch (`_chunk_state_fwd_kernel`, bfloat16 x and B, float32 states), SGLang |
| [`ssd_state_passing_f32`](ssd_state_passing_f32/README.md) | Mamba-2 SSD state passing across chunks of a packed prefill batch, with or without initial states (`_state_passing_fwd_kernel`, float32), SGLang |
| [`streamk_matmul_f16`](streamk_matmul_f16/README.md) | Stream-K matrix multiplication (`streamk_matmul_kernel`, float16 into float32 with atomic adds) |
| [`tma_persistent_matmul_f16`](tma_persistent_matmul_f16/README.md) | Persistent matrix multiplication with host TMA descriptors, Triton tutorial 09 (`matmul_kernel_tma_persistent`, float16) |
| [`vec_add_f32`](vec_add_f32/README.md) | Vector addition, Triton tutorial 01 (`add_kernel`, float32) |
| [`vocab_embedding_bf16`](vocab_embedding_bf16/README.md) | Vocabulary-parallel embedding lookup (`_vocab_parallel_embedding_kernel`, bfloat16), SGLang |

## What a folder holds

| File | What it is |
|---|---|
| `claim.json` | the manifest: `kernel`, `title`, `setup`, `source` (`file`, `sha256`, kernel line `ranges`, `host` line ranges), `lean`, `namespace`, `statement` (`<namespace>.Statement`), `model_map`, `readme`, `check` |
| `SETUP.md` | the setup, in four sections: `## The source` (what `source/` holds, where it comes from, what changed from upstream), `## The config` (the configuration fixed for the launch: removed autotuning, meta-parameters, launch options), `## The launches` (every launch the claim must cover), `## Checked on the H200` (the host's own test on the device) |
| `source/` | the kernel and its host, the file `claim.json` names (with its `sha256`) |
| `Claim.lean` | one Lean file, namespace `Claims.<Root>`: the model (the kernel in the `tl.*` vocabulary, `docs/TRANSCRIPTION.md`), `Inst`, `Inst.ok`, `model`, `launch`, `Pre`, `Post`, `Spec`, `Assumptions`, and `def Statement : Prop := ∀ K P, Assumptions K P → Spec K`, the claim to prove. It proves only `consistent`, `satisfiable` and the coverage facts below; no `sorry`, no `axiom`. |
| `model-map.json` | `source` (as in `claim.json`), `lean`, `namespace`, and `entries`: every code line of the kernel mapped to the model lines that mean it, or the reason it is dropped (`docs/PROVER_GUIDE.md`) |
| `README.md` | Specification, Conditions, Assumptions (as in a kernel write-up), then: Coverage (each launch of `SETUP.md` and where `Inst.ok` covers it), Tolerances (where each bound comes from: the library facts it composes, and why it is provable), Exclusions (anything `SETUP.md` allows that `Inst.ok`/`Pre` leaves out, with the reason, labelled a kernel limit, a library gap, outside real use, or out of scope (a form the library excludes, stated in the claim's own words)), Checks (the results of the checks below) |
| `check_gpu.py` | the GPU check below; it imports the host from `source/` |

## How a claim is checked (no proof of `Statement`)

1. **It builds.** From `lean/`: `lake env lean ../claims/<kernel>/Claim.lean` (the library must be built), with no
   errors, no `sorry`, no `axiom`.
2. **It is not vacuous.** `theorem consistent : Claim.Consistent Assumptions` and
   `theorem satisfiable : Claim.Satisfiable Inst.ok Pre`, as in a kernel bundle.
3. **It covers the setup.** For each concrete launch of `SETUP.md` (the host's test cases and the configurations it
   names), an instance `setup_<case> : Inst` and a proof `setup_<case>_ok : (setup_<case>).ok`.
4. **It holds on the GPU.** `check_gpu.py` runs the kernel's own host (`source/`) on the setup's
   launches and on launches at the edges of `Inst.ok`, computes `Post`'s reference and tolerance from the inputs in
   `float64` exactly as `Post` states them, and reports, for each output, the largest error / tolerance (below 1,
   with its margin) and that every exactness and frame clause holds.
5. **Independent review** of the model, the conditions against the setup, and the specification (its reference
   computed from the host's inputs, never from what the kernel reads; each tolerance meaningful at the covered
   sizes).
6. **Every covered launch is valid.** `faithful` says nothing about a launch that is not `Launch.Valid` or has
   `debug = true`, so for each claim `∀ d, d.ok → (launch d).Valid ∧ (launch d).debug = false` is proved in Lean
   (a scratch copy of the claim with the theorem appended; the theorem is not part of the claim).
7. **A provability audit** (best effort, by reading, not a proof): for every instance and memory the claim covers,
   the model's launch is well behaved (addresses in bounds, no `int32`/`int64` wrap that leaves a buffer, no race,
   no excluded atomic, no undefined operation, loops that end), and each step of the README's Tolerances meets the
   preconditions of the library fact or lemma it cites. A claim is narrowed (or its tolerance or guard weakened)
   where an obstacle is found, keeping the real launches covered.

Each claim's README opens with **What you prove**: the goal unfolded, each part of `Post`, the assumptions, the
kernel as the model runs it, and the obligations a proof must discharge, with `Claim.lean` line numbers ("line N";
a line of the kernel's source is "source line N").

## Library requests

Facts the claims would use if the library stated them. Each claim states only what the library supports
today; its README's Library requests section gives the details.

**Tighter error facts** (the tolerances are correct but loose):

- The depth of `tl.cumsum`'s combine tree (the library states only that each lane is some tree over its
  leaves): `chunk_cumsum_f32` (about 40× the measured error at `BT = 128`).
- A sharper fp8 `tl.dot` fact for a fed-back accumulator: `matmul_e5m2` (covered up to `K = 2048`, where the
  tolerance is about `0.44` of a typical output and about 2.2× the device's own loss on coherent data).
- A tighter fact for an accumulated `bfloat16` `tl.dot`'s share of `|acc|`: `extend_attn_bf16` (with no
  cached prefix its bound grows to `0.047` at 128K new tokens; `DotBF16Chunk` does not help, since each
  `tl.dot` there is a single 64-wide chunk).
- One fused product per quotient (a broadcast quotient's fused product is one value on the device):
  `layer_norm_fwd_f16` (it would keep rows with `cond > 1/2` in `Pre`).

**Facts the library states that a claim could use** (optional; each claim is true and provable as it is):

- `DotF16Chunk` / `DotBF16Chunk` (64 consecutive `k` per chunk): `streamk_matmul_f16`, `ssd_bmm_chunk_bf16`
  would get tighter tolerances.
- `ExpF32Overflow` and `DivF32InfDivisor`: `gdn_gating_bf16` could cover every `b`.
- `AddRel` with one exact operand: `l2norm_fwd_bf16` could cover `eps ∈ [2^-126, 2^-125)`.
- `sum_bcast_mul_depth_err`: a proof of `attn_bwd_pre_f16` can cite it for its depth bound.
