# Setup of `route_histogram_i32`

From SGLang v0.5.23 (commit `297e54fd10`), `python/sglang/kernels/ops/lora/common/routing.py:105`
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

The kernel has no `@triton.autotune` or `@triton.heuristics`. The
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

The claim covers the source's launches on both paths, with shapes as variables:
- `int32` `token_slots` and `group_ids` (`srt/lora/backend/triton_v2_backend.py:99`, and MoE's
  `topk_ids`), contiguous; `int32` `counts`, zero at the launch (any initial value is covered);
- `BINS` as the host's `_histogram_bins` chooses it, `USE_PDL=False`:
  - `BINS > 0` (the `tl.histogram` path): `num_pairs ≥ 16384` and `NUM_BUCKETS < 512`,
    `BINS = 1 << NUM_BUCKETS.bit_length()`;
  - `BINS == 0` (one `tl.atomic_add` per pair): `NUM_BUCKETS ≥ 512`, or `num_pairs < 16384`;
- the routes `build_route` builds with the large builder (`routing.py:709`), which it takes when
  `num_pairs ≥ 16384`, when the route has at least 8192 buckets (`groups_per_slot · max_loras`), or
  when a capacity is given (`routing.py:19-20, 753-767`):
  - dense LoRA, no groups: `token_slots` of `num_tokens ≥ 16384` tokens, `GROUPS_PER_SLOT = 1`,
    `NUM_BUCKETS = max_loras + 1` (`srt/lora/dense/runner.py:142-153`);
  - dense LoRA per (token, head) pair: `group_ids` the heads, `[num_tokens · num_heads, 1]`,
    `GROUPS_PER_SLOT = num_heads` (`srt/lora/dense/runner.py:162-196`);
  - MoE LoRA: `group_ids = topk_ids`, `[num_tokens, topk]`, `GROUPS_PER_SLOT` the number of local
    experts, or 1 for the shared-outer route (`srt/lora/moe/routing.py:75-97`); with 128 local
    experts and 4 adapters (513 buckets) or more, every such route takes `BINS == 0`, and a route
    of 384 experts × 32 adapters (12289 buckets) takes the large builder at any pair count (the
    upstream test `test/registered/kernels/ops/lora/test_moe_lora_routing.py:208`, 768 pairs);
- `max_loras` from 1 to tens of adapters, `GROUPS_PER_SLOT · max_loras + 1 < 2^31` (the host's
  check), `num_pairs` from 0 up (below `2^31`), not a multiple of `BLOCK`;
- any `int32` values in `token_slots` and `group_ids` (slots of `-1` or past `MAX_LORAS`, and
  groups of `-1` or past `GROUPS_PER_SLOT`, are invalid and counted in the last bucket).

`int64` `token_slots` or `group_ids` (which the upstream routing test also passes) are not part of
the setup.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0) runs 30 launches: the host's 11 test launches (on the `tl.histogram`
path: dense routes of 16384, 16385, 100000 and 32768 tokens with `BINS` 4, 8, 16, 32; MoE
per-expert routes of 2048×8, 8193×2 and 16384×8 pairs with `BINS` 512, 128, 512; a shared-outer
route of 3000×6 pairs, `BINS` 16; a per-head route of 1024 tokens × 32 heads, `BINS` 256; with
`BINS == 0`: 1000 dense tokens, and 4096×8 pairs with 513 buckets), four real routes with
`BINS == 0` (16384×8 pairs with 128 experts × 4 adapters, 513 buckets; 65536×8 pairs with
256 × 8, 2049 buckets; 96×8 pairs with 384 × 32, 12289 buckets; a shared-outer route of 2000×6
pairs) and 15 edges (arbitrary `int32` slots and groups on both paths, 511 buckets with
`BINS = 512`, one group per slot with arbitrary groups, `2^26` pairs, `BINS > 0` below 16384 pairs,
`BINS` above the host's, `BINS == 0` at `2^25` pairs and 9 buckets, `BLOCK` 128 and 2048, one pair
and no pair on both paths). On every launch the counts are exact, a guarded relaunch with random
initial counts (some near `2^31 − 1`, so the sum wraps) gives exactly `init + count`, the guards
and inputs are unchanged, and four wrong outputs (nothing added, one pair in the wrong bucket, the
last program's pairs missing, the invalid pairs not counted) fail `Post`.
