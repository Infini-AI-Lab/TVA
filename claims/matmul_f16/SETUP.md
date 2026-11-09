# Setup of `matmul_f16`

From Triton 3.8.0, `python/tutorials/03-matrix-multiplication.py`: the kernel `matmul_kernel` and its
host `matmul`, on the tutorial's `float16` run.

## The source

`source/03-matrix-multiplication.py` is the tutorial's file with two changes, and nothing else (line numbers here and in the
folder are those of `source/`):

1. The `@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])` decorator of
   `matmul_kernel` is removed: `matmul_kernel` (line 228 here) is a plain `@triton.jit` function,
   launched directly.
2. The launch in `matmul` (lines 339-349) passes the config's meta-parameters and launch options
   (below), before `ACTIVATION=activation`.

The functions that build the config lists (`get_cuda_autotune_config`, `get_hip_autotune_config`,
`get_autotune_config`) stay in the file; nothing calls them. The `@triton.jit` helper `leaky_relu`
stays too; no launch uses it (`ACTIVATION` is always `""`). The file runs its test (lines 359-388)
and benchmark (lines 400-440) at module level.

## The config

The first config of `get_cuda_autotune_config()` (lines 166-167), with no `pre_hook`:
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`;
- `num_stages = 3`, `num_warps = 8`;
- grid `(cdiv(M, 128) · cdiv(N, 256),)`; `ACTIVATION = ""` (`matmul`'s default; the file never passes
  another).

## The launches

The claim covers every launch of `matmul_kernel` that the file's `float16` code makes through
`matmul(a, b)`:
- `a` `float16` `[M, K]`, contiguous (`matmul` asserts it); `b` `float16` `[K, N]`, contiguous; `c` the
  fresh contiguous `float16` `[M, N]` that `matmul` allocates;
- the test (lines 359-370): `M = N = K = 512`, `a`, `b` drawn as `torch.rand(...) - 0.5`;
- the benchmark's `fp16` line (lines 402-440, `fp8_inputs = False`, provider `triton`):
  `M = N = K = 128·i` for `i = 2, …, 32` (`256, 384, …, 4096`), `a`, `b` drawn as `torch.randn`.

The file's `float8e5` test (lines 372-388) and the benchmark's `fp8` line launch the same kernel
with `float8e5` operands: that is a different instantiation of the kernel (other pointer dtypes and
another `tl.dot`), and those launches are not part of this setup.

## Checked on the H200

`check_gpu.py` (H200 NVL, Triton 3.8.0) runs `matmul` on the test and on all 31 benchmark sizes with
the file's draws, plus edges (`K = 8192`, `K = 4095`, `M = N = 1`, `K = 1`, `K = 0`, `M = 0`, inputs
scaled by 8 and by `2^-12` (subnormal operands), NaN and infinities in the inputs) and two direct
launches (a padded and a column-major `c` inside a guarded buffer, `b` with strides `(1, K)`). Every
element of `c` meets both clauses of `Post`: the largest error / tolerance is 0.977 (subnormal
operands, where the output's rounding is all the error); 0.75 on the test and 0.83 down to 0.48 on
the benchmark. Three wrong results (rounded to `bfloat16`, accumulated in `float16` step by step, the
last step's products left out) each break `Post` on the test's and the benchmark's data. The inputs
are unchanged and no byte outside the matrix `c` is written.
