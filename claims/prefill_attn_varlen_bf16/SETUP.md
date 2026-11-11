# Setup of `prefill_attn_varlen_bf16`

From SGLang v0.5.23 (commit `297e54fd10`),
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

The changes from the code base's source (the module docstring lists each with its line): the
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

The covered paths: `IS_CAUSAL = True` (the wrapper's default, as the extend-attention reference
`python/sglang/test/kernels/extend_attention.py:34` calls it) and `IS_CAUSAL = False` (the vision
encoder's call, `python/sglang/srt/layers/attention/vision.py:510`), `bfloat16` `q`, `k`, `v` and
output.

## The launches

The claim covers the wrapper's launches, causal or not (`is_causal`), with, as variables:
- `q`: `bfloat16`, `[total_tokens, num_heads, head_dim]`; `k`, `v`: `bfloat16`,
  `[total_tokens, num_kv_heads, head_dim]`; last dimension contiguous, token and head strides
  as the tensors give them (the vision encoder passes the views of its fused `qkv` projection,
  `vision.py:1425-1432`, with no copy: `STRIDED_QKV_BACKENDS`, `vision.py:1066`); `num_heads` a
  multiple of `num_kv_heads`;
- `b_start_loc`, `b_seq_len`: `int32`, `[batch]` (the vision caller passes `cu_seqlens` and
  `int32` lengths; the code base's unit test passes `int64` tensors, which are not part of the
  setup); the sequences packed:
  sequence `b`'s tokens `b_start_loc[b] … b_start_loc[b] + b_seq_len[b] - 1` inside the tensors;
- `max_input_len` at least every `b_seq_len[b]`;
- `sm_scale` the default `1/sqrt(head_dim)` or a value the caller passes (the vision towers
  pass their own, `None` by default; Gemma 4's passes `1.0`,
  `python/sglang/srt/models/gemma4_vision.py:194`);
- the code base's shapes, causal: the unit test
  (`test/registered/attention/test_triton_attention_kernels.py:640-680`) runs 4 heads, head dims
  128, 96, 80 and 13, sequence lengths 8 and 12; language models run head dim 128, 1 to 64 heads
  per rank, grouped KV heads, and prompts of 1 to thousands of tokens;
- the code base's shapes, not causal (the vision encoders, `is_causal=False`, `b_start_loc` the
  `int32` `cu_seqlens`, `b_seq_len` its differences, one sequence per image, video frame pair or
  attention window): Qwen2.5-VL's encoder (16 heads of 80, full-attention layers over each image
  and window layers over windows of at most 64 patches,
  `python/sglang/srt/models/qwen2_5_vl.py:433-470`), Qwen3-VL's (16 heads of 72), MiMo-VL's (32
  heads of 64 on 8 KV heads, `python/sglang/srt/models/mimo_vl.py:37-56`), heads split over up to
  8 ranks; images of up to `SGLANG_IMAGE_MAX_PIXELS = 16384·28·28` pixels, `65536` patches of 14
  (`python/sglang/srt/environ.py:1599`), video frames of up to `768·28·28` pixels
  (`python/sglang/srt/multimodal/processors/qwen_vl.py:74`).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs the host's 8 test cases (heads 4/4 with head dims 128, 96,
80 and 13; 32/8 and 28/4 with 128; 16/16 with 80; 8/1 with 64; lengths 1 to 2048), causal and
not; a causal fused-`qkv` strided launch and the largest causal model shape (64/8 heads, head dim
128, prompts of 8192 and 4096); the vision encoders' non-causal launches on fused-`qkv` views
(16 heads of 80 on one image of 5476 patches, on its 100 windows of at most 64 patches, on 16
video frame pairs of 3072 patches, 2 heads of 80 on one image of 65536 patches, 16 heads of 72,
32 heads of 64 on 8 KV heads) and `sm_scale = 1.0` on unit-RMS inputs; `Inst.ok` edges (head
dims 9 and 256, 64 heads on one KV head, gaps and empty sequences, `sm_scale = 0.5`, one prompt
of `2^17` tokens) and `Pre` edges (large scores, `|v| ≈ 10^30`, subnormal `v`, uniform
attention), each causal and not. The worst error / tolerance is `0.434` (the uniform causal
case; `0.40` on the host's own cases, `0.38` on the non-causal launches); every written element is finite, every input is
unchanged, and every element of `o` outside the sequences keeps its bits. Wrong outputs (all
zero, the other mode's output, a key block or a key left out, the wrong key/value head) are
rejected. Head dims 1 and 8 are rejected on the host (`CompilationError`).
