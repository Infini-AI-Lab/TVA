# Setup of `matmul_e5m2`

From Triton 3.8.0, `python/tutorials/03-matrix-multiplication.py`: the kernel `matmul_kernel` and its
host `matmul`, on the tutorial's `float8e5` run.

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

The claim covers every launch of `matmul_kernel` that the file's `float8e5` code makes through
`matmul(a, b)`:
- `a` `float8e5` `[M, K]`, contiguous (`torch.randn(…, dtype=torch.float16)` cast with
  `.to(torch.float8_e5m2)`); `b` `float8e5` `[K, N]`, the transpose of a contiguous matrix, strides
  `(1, K)` (`b = b.T` before the cast, which keeps the strides); `c` the fresh contiguous `float16`
  `[M, N]` that `matmul` allocates;
- the test (lines 372-388): `M = N = K = 512`;
- the benchmark's `fp8` line (lines 402-440, `fp8_inputs = True`, provider `triton`) at
  `M = N = K = 128·i` for `i = 2, …, 16` (`256, 384, …, 2048`).

The benchmark's `fp8` launches at `M = N = K = 128·i` for `i = 17, …, 32` (`2176, …, 4096`) are not
part of this setup, and the claim does not cover them (`K ≤ 2048`), a library gap: the library's fp8
`tl.dot` fact cannot give a meaningful tolerance there. One bound must hold on coherent data too,
where the kernel's fp8 accumulation loses `30%` of the output at `K = 4096` (on the file's own data
its error is about `0.06%`), and that bound no longer rejects typical wrong results (`0.9` times the
exact result passes from `K = 3072` on; README, Exclusions).

The file's `float16` run (lines 359-370 and the benchmark's `fp16` line) launches the same kernel with
`float16` operands; it is not part of this setup.

## Checked on the H200

`check_gpu.py` (H200 NVL, Triton 3.8.0) runs `matmul` on the test and on the 15 benchmark sizes
`256 … 2048` with the file's draws, plus edges (`K = 2047`, `K = 2000`, `M = 4096` with `K = 2048`,
`M = N = 1`, `K = 1`, inputs scaled by 16), two direct launches (a padded and a column-major `c`
inside a guarded buffer, a row-major `b`) and coherent data at `K = 512, 1024, 2048`. Every element
of `c` is within `Post`'s bound: on the file's draws the largest error / tolerance is `0.045`
(`K = 256`), `0.023` on the test and `0.010` at `K = 2048`; on coherent data `0.458`, where the
device's own error is `5–18%` of the output. The tolerance itself is loose next to the outputs on
the file's data: its median over `|S|` is `0.095` at `K = 512` and `0.44` at `K = 2048` (README,
Tolerances). Wrong results (all zero, a step or a product left out, `0.9` times the exact result,
another column, `b` shifted along `K`) each break `Post` at `K = 512, 1024, 2048`, at `2048` some
on only about `0.01–0.02%` of the elements. The inputs are unchanged and no byte outside the matrix `c` is
written.
