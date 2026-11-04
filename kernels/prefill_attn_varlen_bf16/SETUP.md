# Setup of `prefill_attn_varlen_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (`sandbox/TESTSET.md`, kernel
21), from SGLang v0.5.23 (commit `297e54fd10`),
`python/sglang/kernels/ops/attention/prefill_attention.py:35` `_fwd_kernel`.

## The source

`source/prefill_attention.py` holds:
- `_fwd_kernel` (lines 51-184), the `@triton.jit` kernel, verbatim from the source's lines
  34-167: one program per (sequence, query head, block of `BLOCK_M` queries) runs an online
  softmax over the sequence's keys in blocks of `BLOCK_N`, with `tl.dot`, causal or not
  (`IS_CAUSAL`);
- `context_attention_fwd`, the host wrapper, from the source's lines 170-219;
- `reference`, attention in `float64` sequence by sequence, and the test under
  `if __name__ == "__main__":`.

What changed from the code base's source (the module docstring lists each with its line): the
module's device globals are locals of the host with their values on an H200 (`_is_cuda = True`,
`_is_hip = False`, `CUDA_CAPABILITY = (9, 0)`), so the file has no module-level assignments; the host allocates the output `o` with `torch.empty_like(q)` and
returns it, where the source's wrapper takes `o` from the caller (the vision encoder's caller
allocates it the same way, `python/sglang/srt/layers/attention/vision.py:508`). The kernel's body
is unchanged.

## The config

No autotuning and no heuristics: the wrapper picks the launch itself (source lines 180-217), and
on an H200 (capability 9 > 8):
- `BLOCK_M = BLOCK_N = 128`;
- `BLOCK_DMODEL = next_power_of_2(Lk)`, `Lk = k.shape[-1]`;
- `num_warps = 4` when `Lk <= 64`, else `8`; `num_stages = 1`;
- grid `(batch, num_heads, cdiv(max_input_len, 128))`;
- `kv_group_num = num_heads // num_kv_heads`, `sm_scale = 1 / sqrt(Lq)` by default.

The kernel declares no parameter defaults; the host passes each parameter.

The proved path: `IS_CAUSAL = True` (the wrapper's default, as the extend-attention reference
`python/sglang/test/kernels/extend_attention.py:34` calls it), `bfloat16` `q`, `k`, `v` and
output. (SGLang's vision encoder calls it with `is_causal=False`; that path stays in the source.)

## The launches

The claim should cover the wrapper's causal launches with, as variables:
- `q`: `bfloat16`, `[total_tokens, num_heads, head_dim]`; `k`, `v`: `bfloat16`,
  `[total_tokens, num_kv_heads, head_dim]`; last dimension contiguous, token and head strides
  as the tensors give them; `num_heads` a multiple of `num_kv_heads`;
- `b_start_loc`, `b_seq_len`: `int32`, `[batch]` (the vision caller passes `cu_seqlens` and
  `int32` lengths; the code base's unit test passes `int64` tensors); the sequences packed:
  sequence `b`'s tokens `b_start_loc[b] … b_start_loc[b] + b_seq_len[b] - 1` inside the tensors;
- `max_input_len` at least every `b_seq_len[b]`;
- the code base's shapes: the unit test
  (`test/registered/attention/test_triton_attention_kernels.py:640-680`) runs 4 heads, head dims
  128, 96, 80 and 13, sequence lengths 8 and 12; models run head dims 64 to 128 (vision encoders
  80 and 64, language models 128), 1 to 64 heads per rank, grouped KV heads, and prompts of 1 to
  thousands of tokens.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes on 8 configurations
(heads 4/4 with head dims 128, 96, 80 and 13 on lengths 8 and 12; 32/8 and 28/4 with head dim 128,
16/16 with 80, 8/1 with 64; lengths 1 to 2048, multiples of the block and not): the largest
absolute error against the `float64` reference is 1.3e-2, from 8.6e-3 to 1.3e-2 per
configuration (the output and the probabilities fed to the second `tl.dot` are `bfloat16`).
