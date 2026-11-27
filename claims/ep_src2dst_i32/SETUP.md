# Setup of `ep_src2dst_i32`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/moe/ep_moe_kernels.py:147`,
the kernel `compute_src2dst_triton_kernel`, launched by `cutlass_w4_run_moe_ep_preproess`
(`ep_moe_kernels.py:205`).

## The source

`source/ep_src2dst.py` holds:
- `compute_src2dst_triton_kernel`, the `@triton.jit` kernel, verbatim (`ep_moe_kernels.py:146-168`):
  every program finds the first non-negative sorted id (a binary search when `sorted_topk_ids[0]`
  is negative), then one program per 512 sorted positions stores each position's packed row (or
  `-1`) at its slot `reorder_ids[i]` of `src2dst`;
- `cutlass_w4_run_moe_ep_preproess`, the host, verbatim (`ep_moe_kernels.py:205-217`);
- `reference`, a test helper (the map from its definition, on the CPU);
- the test under `if __name__ == "__main__":`.

Changes from the code base: none to the kernel or the host. The source module's other kernels and
hosts, its logging, its `sglang` imports and its device checks are dropped (neither copied function
uses them); the module docstring lists this.

## The config

No autotuning and no heuristics in the source. The host's own choice: grid
`(triton.cdiv(numel, 512),)`, `num_toks = topk_ids.numel()` (a runtime `int32`), `BLOCK_SIZE = 512`
(a `tl.constexpr`); no launch options, so `num_warps=4`, `num_stages=3`, Triton's defaults. The host
launches nothing when `numel = 0`.

## The launches

The caller in the code base is `cutlass_w4a8_moe` (`srt/layers/moe/cutlass_w4a8_moe.py:166`, the
W4A8 MoE of `srt/layers/quantization/w4afp8.py:315` and of the compressed-tensors W4A8-FP8 scheme),
on its path without the CUDA op `cutlass_prepare_moe_input`. The claim covers the host's launches
with shapes as variables:
- `topk_ids`: `[num_tokens, topk]`, made contiguous `int32` by the caller
  (`cutlass_w4a8_moe.py:131-132`); an entry is a local expert id, or negative for a slot routed to
  no local expert (expert parallelism, padded tokens), which the kernel gives `-1`;
- `sorted_topk_ids`, `reorder_ids`: `torch.sort(topk_ids.view(-1), stable=True)`, `int32` ascending
  and the `int64` permutation of `[0, numel)`;
- `src2dst`: `[numel]`, `int32`, a fresh `torch.empty`;
- sizes: the host's test (1 to 4096 tokens, top-1 to top-8, 8 to 160 experts, from none to all of
  the slots negative); serving: decode (one token, top-8, DeepSeek-V3's routing), prefill chunks of
  16384 tokens, batches up to `2^22` tokens (`2^25` slots of top-8).

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11) passes on 23 launches: the host's six test
cases, three serving shapes (decode `1×8`, a `16384×8` prefill on 8-way expert parallelism with
7/8 of the slots negative, `2^22` tokens of top-8: `2^25` slots), six host edges (a single negative
slot; 511, 512, 513, 1024, 1025 slots), and eight guarded direct launches at the edges of `Pre`
(`n = 1`, the compile-time `num_toks`, with a negative and a non-negative id; ids that are not
sorted after the negative prefix, with values down to `-2^31 + 1`; `reorder_ids` an injection into a
longer output; all ids negative; the prefix ending at a block boundary; `2^24 + 3` slots, half
negative). All 50,585,906 written elements are exact as `Post` states; every other output element,
every input and every guard is unchanged; 66 wrong outputs (the negative prefix not removed, the
prefix off by one, `-1` not written, the row stored at the sorted position) all fail `Post`. The
kernel's limit `n = 2^31 - 1` is not run (its buffers exceed 4 GB).
