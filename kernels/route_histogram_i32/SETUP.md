# Setup of `route_histogram_i32`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/lora/common/routing.py:105`
`_route_histogram_kernel`.

## The source

`source/route_histogram.py` holds:
- `route_bucket_ids` and `_add_counts`, the `@triton.jit` helpers the kernel calls, copied
  verbatim from the same file (`routing.py:40, 88`): a pair's bucket from its token's adapter
  slot and its group (`-1` when either is invalid), and the counting, by `tl.histogram` and one
  `tl.atomic_add` per nonzero bin (`BINS > 0`) or one `tl.atomic_add` per pair (`BINS == 0`);
- `_route_histogram_kernel`, the `@triton.jit` kernel, copied verbatim: one program per `BLOCK`
  pairs;
- `_histogram_bins`, the source's choice of `BINS` (`routing.py:364`), with two module constants
  inlined;
- `route_histogram`, the host function, adapted from the histogram launch of
  `_build_large_route` (`routing.py:416`, launch at `:455-469`);
- `reference`, a test helper (the buckets counted with `torch.bincount`);
- the test under `if __name__ == "__main__":`.

The host allocates `counts` with `torch.zeros`, as the source does on first allocation
(`_aligned_route_scratch`, `routing.py:393-406`), launches the histogram kernel only (not the
scan and place kernels that follow it in `_build_large_route`) and returns `counts`; the module
docstring lists the changes with their source lines.

## The config

No decorator was removed: the kernel has no `@triton.autotune` or `@triton.heuristics`. The
configuration is the source's own: `BLOCK = 512` and `num_warps = 8` (`_HIST_BLOCK`,
`_HIST_WARPS`, `routing.py:26-27`), grid `(cdiv(max(num_pairs, 1), 512),)`,
`NUM_BUCKETS = GROUPS_PER_SLOT * MAX_LORAS + 1`, `WIDTH = group_ids.shape[1]` (1 without
groups), `HAS_GROUPS = group_ids is not None`, and `BINS = _histogram_bins(num_buckets,
num_pairs)`: `0` when `num_buckets ≥ 512` or `num_pairs < 16384`, otherwise
`1 << num_buckets.bit_length()`, the power of two above `num_buckets`. The flag: `USE_PDL=False`
and no `launch_pdl`; the source passes `USE_PDL=is_arch_support_pdl()` (`routing.py:453, 467`),
and programmatic dependent launch is outside TVA's launches. The launch option `num_stages=3`
is Triton 3.8.0's default, which the source's launch gets.

## The launches

The claim should cover the source's launches on the `tl.histogram` path, with shapes as
variables:
- `int32` `token_slots` and `group_ids` (`srt/lora/backend/triton_v2_backend.py:99`, and MoE's
  `topk_ids`), contiguous; `int32` `counts`, zero at the launch;
- `BINS > 0`: `num_pairs ≥ 16384` and `NUM_BUCKETS < 512`, `BINS = 1 << NUM_BUCKETS.bit_length()`;
  `USE_PDL=False`. The path `BINS == 0` (one atomic per pair) stays in the source and is not part
  of the claim;
- the routes `build_route` builds with the large builder (`routing.py:709`, which takes it for
  `num_pairs ≥ 16384`, `routing.py:21, 753-767`):
  - dense LoRA, no groups: `token_slots` of `num_tokens ≥ 16384` tokens, `GROUPS_PER_SLOT = 1`,
    `NUM_BUCKETS = max_loras + 1` (`srt/lora/dense/runner.py:142-153`);
  - dense LoRA per (token, head) pair: `group_ids` the heads, `[num_tokens · num_heads, 1]`,
    `GROUPS_PER_SLOT = num_heads` (`srt/lora/dense/runner.py:162-196`);
  - MoE LoRA: `group_ids = topk_ids`, `[num_tokens, topk]`, `GROUPS_PER_SLOT` the number of local
    experts, or 1 for the shared-outer route (`srt/lora/moe/routing.py:75-97`);
- `max_loras` from 1 to tens of adapters, `num_tokens · WIDTH` from 16384 up, the number of pairs
  not a multiple of `BLOCK`;
- any `int32` values in `token_slots` and `group_ids` (slots of `-1` or past `MAX_LORAS`, and
  groups of `-1` or past `GROUPS_PER_SLOT`, are invalid and counted in the last bucket).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes, the counts equal to the
`torch.bincount` reference in every case, and summing to the number of pairs. On the
`tl.histogram` path: dense routes of 16384, 16385, 100000 and 32768 tokens with 1, 4, 8 and 16
adapters (`BINS` 4, 8, 16, 32); MoE per-expert routes of 2048×8 pairs with 64 experts and 4
adapters (257 buckets, `BINS` 512), 8193×2 with 8 experts and 8 adapters (`BINS` 128), 16384×8
with 32 experts and 8 adapters (`BINS` 512); a shared-outer route of 3000×6 pairs (`BINS` 16);
a per-head route of 1024 tokens × 32 heads with 4 adapters (`BINS` 256). Each with random slots
in `[-1, max_loras]` and groups in `[-1, num_groups)`, invalid ones included. Also run, on the
path `BINS == 0`: 1000 dense pairs, and 4096×8 MoE pairs with 513 buckets.
