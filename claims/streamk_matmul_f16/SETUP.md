# Setup of `streamk_matmul_f16`

A classic kernel, Stream-K GEMM (Osama et al., "Stream-K: Work-centric Parallel Decomposition for
Dense Matrix-Matrix Multiplication on the GPU", 2023), written in Triton 3.8.0 for TVA:
`source/streamk_matmul.py`, kernel `streamk_matmul_kernel`, host `streamk_matmul`.

## The source

`source/streamk_matmul.py` holds the `@triton.jit` kernel `streamk_matmul_kernel` (lines 26–64), the
block sizes (line 67), the host wrapper `streamk_matmul` (lines 70–93) and the test under
`if __name__ == "__main__":` (lines 96–114). It has no upstream file to differ from.

`C = A @ B` with `float16` `A`, `B` and a `float32` `C`. The work is `num_tiles · iters_per_tile`
iterations of the K loop, split into equal contiguous ranges, one per program. For each tile its
range touches, a program computes the partial product over the K iterations it owns; it stores the
tile if it owns all of the tile's iterations, and otherwise adds the partial product into `C` with
`tl.atomic_add`; the wrapper allocates `C` with `torch.zeros`.

## The config

No autotuning and no heuristics. The wrapper launches with `BLOCK_M = BLOCK_N = 128`,
`BLOCK_K = 64`, `num_warps=8`, `num_stages=3`, grid `(num_programs,)`, where `num_programs` is the
device's SM count unless the caller passes one, `total_iters = cdiv(M, 128)·cdiv(N, 128)·cdiv(K, 64)`
and `iters_per_program = cdiv(total_iters, num_programs)`.

## The launches

The claim must cover the wrapper's launches as the test makes them, with shapes as variables:
- `float16` `A` (`M × K`) and `B` (`K × N`), contiguous (`torch.randn`), a fresh zero-initialised
  contiguous `float32` `C`, and positive `M`, `N`, `K`;
- the test's shapes and program counts: `512³`, `1000³`, `3072 × 4096 × 1024` and
  `8192 × 8192 × 4096` with the SM count (132 on the H200), `257 × 129 × 4097` with 7 programs, and
  `64³` with 3 programs (more programs than iterations, so some programs do no work);
- input values: any `float16` values (the test uses `torch.randn`).

Both paths run: at `1000³` with 132 programs every tile is split between two programs (accumulated
with atomics); at `8192 × 8192 × 4096` some tiles are stored by one program and others are split.

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, one H200): on the six test launches, every element of `C` is finite
and within `Post`'s tolerance, with worst error / tolerance `0.0024`–`0.0100` per shape. On the edges
of `Inst.ok` (`1 × 1 × 1`, shapes not multiples of the blocks, `K = 2^13`, large and subnormal
operands, NaN and infinities in some rows and columns, all-positive inputs, inputs whose mass is in
the last chunk, transposed `A` and `B`, `1` to `2048` programs, `C` padded inside a sentinel
buffer) the worst is `0.036`; `A` and `B` are unchanged and nothing outside the matrix `C` is
written. Wrong results fail `Post` at every tested `K` from `64` to `8192`: on `randn` inputs the
result rounded to `bfloat16`, one 64-chunk left out, and a `float16` accumulation chunk by chunk;
with the mass in the last chunk, the result rounded to `float16`. 44 of 44 cases pass.
