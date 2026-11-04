# Setup of `decode_attn_gqa_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (`sandbox/TESTSET.md`, kernel
22), from SGLang v0.5.23 (commit `297e54fd10`),
`python/sglang/kernels/ops/attention/decode_attention.py:530` `_fwd_grouped_kernel_stage1`.

## The source

`source/decode_attention.py` holds:
- `_extract_kv_strides` (lines 84-97), a host helper, verbatim from the source's lines 212-225;
- `tanh` (lines 100-103), the `@triton.jit` helper the kernel calls under a logit cap, verbatim
  from the source's lines 228-231;
- `_fwd_grouped_kernel_stage1` (lines 106-370), the `@triton.jit` kernel, verbatim from the
  source's lines 529-793 except for its parameters' defaults: one program per (request, block of
  query heads sharing a KV head, KV split) runs an online softmax over the split's tokens, read
  through the request's slots in `kv_indices`, and stores the split's normalised partial output
  and its log-sum-exp;
- `decode_attention_fwd_grouped_stage1`, the host, from the stage-1 launcher
  `_decode_grouped_att_m_fwd` (source lines 796-931) with the entry points' checks, scale and
  dispatch (`decode_attention_fwd`, source lines 1191-1316, and `decode_attention_fwd_grouped`,
  lines 1129-1188);
- `reference`, each split's output and lse in `float64`, and the test under
  `if __name__ == "__main__":`.

What changed from the code base's source (the module docstring lists each with its line): the
SGLang imports, the logger and `typing` are removed; the module-level `_is_hip`, `_is_gfx1250`
(both `False` on an H200), `_MIN_BLOCK_KV = 32` and `_GROUPED_BLOCK_H = 16` are locals of the host,
and `_grouped_head_tiles` is inlined, so the file has no module-level assignments; the defaults
of the kernel's parameters are removed (`HAS_MLA: tl.constexpr = False` → `HAS_MLA:
tl.constexpr`, `Aux0=None` → `Aux0`, and the same for the other eleven), and the host passes
every parameter; the host allocates `att_out` and `att_lse` with `torch.empty` (as SGLang's
Triton backend does, `python/sglang/srt/layers/attention/triton_backend.py:953-970`) and returns
them, where the source's launcher takes them from the caller; the host passes the off values of
the paths the task does not take (no MLA, PDL, score modifier, aux tensor, forced splits, batch
rotation or gfx1250) and asserts `kv_group_num != 1`, since the entry point sends that case to the
MHA kernel. The kernel's body is unchanged. Stage 2 (`_fwd_kernel_stage2`) is not part of this
task.

## The config

No autotuning and no heuristics: the launcher picks the launch itself, and on the CUDA branch
(source lines 818-867):
- `BLOCK_DMODEL = next_power_of_2(Lk)`, `BLOCK_DPE = 0` (for `Lk` other than 576 and 288),
  `BLOCK_DV = next_power_of_2(Lv)`;
- `BLOCK_N = 32`, `BLOCK_H = 16`, `MIN_BLOCK_KV = 32`;
- grid `(batch, cdiv(num_heads, min(16, kv_group_num)), max_kv_splits)`;
- `num_warps = 4`, `num_stages = 2`.

The parameters' defaults removed: TVA's subset has no kernel parameter defaults; the host passes
each. The flags of the proved path, as the host passes them: `HAS_MLA = False`, `USE_PDL = False`,
`IS_GFX1250 = False`, `SCORE_MOD = None` with `Aux0 = None` and `aux0_stride_t = aux0_stride_h =
aux0_len = 0`, `forced_kv_splits = 0` with `USE_FORCED = False`, `uneven_batch = None` with
`ROTATE_BATCH = False`, `logit_cap = 0.0` (the entry points' default: no cap, so `tanh` is not
called), `xai_temperature_len = -1` (the default: off), `PAGE_SIZE` the pool's page size,
`sm_scale_withk = sm_scale * k_scale` with `k_scale = 1.0` (a `bfloat16` cache). SGLang's Triton
backend sets `use_pdl = is_arch_support_pdl()` on CUDA (true from capability 9), which only adds a
`tl.extra.cuda.gdc_launch_dependents()` at the kernel's end; the task proves `USE_PDL = False`,
the launcher's default.

## The launches

The claim should cover the host's launches with, as variables:
- `q`: `bfloat16`, `[batch, num_heads, Lk]`, last dimension contiguous;
- `k_buffer`, `v_buffer`: `bfloat16`, `[num_slots, num_kv_heads, Lk]` and `[num_slots,
  num_kv_heads, Lv]` (SGLang's token pool, 3-D), last dimension contiguous, slot and head strides
  as given; `num_heads` a multiple of `num_kv_heads`, `kv_group_num = num_heads // num_kv_heads > 1`;
- `kv_indptr`: `int32`, `[batch + 1]`, non-decreasing from 0; `kv_indices`: `int64`, at least
  `kv_indptr[batch]` slots, each a slot of the buffers (the backend's `kv_indices` are `int64`,
  `triton_backend.py:499`);
- `num_kv_splits`: `int32`, `[batch]`, each in `[1, max_kv_splits]` (the backend's
  `get_num_kv_splits`); `max_kv_splits` (`triton_attention_num_kv_splits`, default 8);
- `sm_scale` a float, `page_size` 1 or more (`PAGE_SIZE`);
- the code base's shapes (`test/registered/attention/test_triton_attention_kernels.py:855-868`):
  batch 2, lengths 5, 100, 128 and 500, heads 16/1, 64/1, 128/1 and 128/2 (and 16/16, which the
  host's `kv_group_num != 1` excludes), head dims 64, 13, 80 and 512 (and 576/512, MLA-shaped),
  4 splits of at most 8; GQA models run 4 to 128 query heads per rank over 1 to 8 KV heads
  (Llama-3-8B 32/8, Qwen2.5-7B 28/4, Llama-3-70B 64/8), head dim 128.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes on 8 configurations
(the code base's 16/1, 64/1, 128/1 and 128/2 with head dims 64, 13, 80 and 512; 32/8, 28/4, 64/8 and
16/2 with head dims 128 and 96; lengths 1 to 4000, per-request split counts 1 to 16, pages of 1 and
16, empty splits included): on every non-empty split, the partial output's largest absolute error
against the `float64` reference is 3.2e-3, and the lse's 1.6e-6.
