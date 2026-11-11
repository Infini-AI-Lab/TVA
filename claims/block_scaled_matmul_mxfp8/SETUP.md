# Setup of `block_scaled_matmul_mxfp8`

From Triton 3.8.0, `python/tutorials/10-block-scaled-matmul.py`: `block_scaled_matmul_kernel`
(lines 174–240) and its host `block_scaled_matmul` (lines 243–281), in the `mxfp8` and `mixed`
configurations.

## The source

`source/10-block-scaled-matmul.py` is the tutorial's file as it is, with no change. It holds:
- `block_scaled_matmul_kernel` (`@triton.jit(launch_metadata=_matmul_launch_metadata)`, lines
  174–240), a tiled GEMM through host TMA descriptors: program `p` computes the `BLOCK_M × BLOCK_N`
  tile at row block `p % cdiv(M, BLOCK_M)` and column block `p // cdiv(M, BLOCK_M)`, looping over
  `K` in steps of `BLOCK_K`: each step loads a block of `a`, a block of `b` and the matching 5-D
  blocks of their scales, rearranges the scales to `[BLOCK_M, BLOCK_K // VEC_SIZE]` and
  `[BLOCK_N, BLOCK_K // VEC_SIZE]`, and accumulates `tl.dot_scaled(…)` into a `float32`
  accumulator (`"e4m3"` by `"e2m1"` when `MIXED_PREC`, line 230; `"e2m1"` by `"e2m1"` when both
  operands are packed fp4, line 232; `"e4m3"` by `"e4m3"` otherwise, line 234); the tile is
  converted to the output dtype and stored;
- the host: `initialize_block_scaled` (lines 330–449: the inputs, the scales in the 5-D
  preshuffled layout, the descriptors, the config, and optionally the reference), `block_scaled_matmul`
  (the output, its descriptor, the launch), the test `validate_block_scaled` (lines 452–469), the
  benchmark `bench_block_scaled` (lines 472–502), the cuBLAS baseline and `__main__` (lines
  730–765);
- the AMD kernel `block_scaled_matmul_kernel_cdna4` and its host functions (lines 515–727), not
  part of this setup.

## The config

The kernel is not autotuned and has no `pre_hook`. `initialize_block_scaled(M, N, K, fmt)` fixes
the config on the H200 (`is_rubin()` false) for `fmt = "mxfp8"` or `"mixed"` and
`block_scaled_matmul` passes it to the launch, with no launch option:
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 128` (`"fp4"` is not in either
  name, line 333), `VEC_SIZE = 32`, `ELEM_PER_BYTE_A = 1`, `rep_m = 1`, `rep_n = 2`, `rep_k = 1`;
- `ELEM_PER_BYTE_B = 1` for `"mxfp8"` and `2` for `"mixed"` (line 337): `b` is `float8e4nv`
  `[N, K]` with the block `[256, 128]`, or `uint8` `[N, K // 2]` holding two `e2m1` elements per
  byte along `K` (`to_packed_tensor(dim=1)`, line 363, the low nibble first) with the block
  `[256, 64]` (line 368);
- `NUM_STAGES = 4`, `disallow_acc_multi_buffer = True` (kernel arguments, not launch options), the
  default `num_warps`;
- `output_type = 1` (`float16`, the only output dtype the file asks for);
- grid `(cdiv(M, 128) · cdiv(N, 256), 1)`;
- `MIXED_PREC` (line 216) is `False` for `"mxfp8"` and `True` for `"mixed"`: the step is
  `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)` (line 234) or
  `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)` (line 230), which Triton
  emulates on sm_90 (`docs/rules/num.md`, "`tl.dot_scaled` on sm_90").

## The launches

The claim covers the `mxfp8` and `mixed` launches of the file's host functions (`--format mxfp8`
and `--format mixed`), as if `supports_block_scaling()` passed (on the H200 it is false, and the
script's `__main__` prints that the example needs block-scaled support and stops before the test),
with shapes as variables:
- `a` `float8e4nv` `[M, K]`, contiguous; `b` `float8e4nv` `[N, K]` (`mxfp8`) or `uint8`
  `[N, K // 2]` of packed `e2m1` (`mixed`), contiguous; their `e8m0` scales
  (`MXScaleTensor(…).data`, `uint8`) `[M // 128, K // 128, 32, 16]` and `[N // 128, K // 128, 32, 16]`
  viewed as `[1, M // 128, K // 128, 2, 256]`; `c` a fresh `float16` `[M, N]`; the five host
  `TensorDescriptor`s of `initialize_block_scaled` and `block_scaled_matmul`;
- the test, `validate_block_scaled(8192, 8192, 8192, fmt)` (line 749), for both formats;
- the benchmark's sizes, `bench_block_scaled(K, block_scale_type=fmt)` (lines 472–502):
  `M = N = 8192` and `K` a positive multiple of `128` (asserted, line 473) up to `2^17`, for both
  formats. The file's `K`s are the default `-K 512` and, for `mxfp8`, the module docstring's
  `--K_range 8192 16384 --K_step 2048` (`K = 8192, 10240, 12288, 14336, 16384`); `K` above `2^17`
  is outside real use and not part of the setup;
- the host's values: elements `MXFP4Tensor(…).random()` (a sign and one of `0, 0.5, 1, 1.5, 2, 3,
  4, 6`), cast exactly to `e4m3` (`a`, and `b` for `mxfp8`) or packed as `e2m1` (`b` for `mixed`);
  scale bytes `floor(log2(torch.rand + 1e-8)) + 127`, in `100 … 126`.

The `mxfp4` and `nvfp4` configurations (rejected on the H200 at the tutorial's config: out of
shared memory), the other output dtypes and the AMD kernel are not part of the setup.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0, at most 2.9 GiB of device memory) runs the host's
`block_scaled_matmul` for both formats on the test's launch (`8192³`, inputs drawn from the test's
distribution, not its seed-42 bytes) and the benchmark's `K = 512` (inputs from the host's own
`initialize_block_scaled`), for `mxfp8` also at `K = 16384`, and on edges of the claim's conditions
and inputs, and checks every element of `c` against the claim's float64 reference and tolerance.
At `8192³` and `K = 16384` the inputs are built by the host builder's steps in slices of rows (same
distributions, layout and descriptors), since the host's builder alone needs more than 4 GB there.
Worst error / bound: `0.220` (test), `0.692` (`K = 512`), `0.135` (`K = 16384`); `0.809` over all
cases (`M = N = K = 128`). On the host's data the two formats give the same values (the
host draws the same `e2m1` values for `b` and casts them exactly to `e4m3` for `mxfp8`), and the
results are the same. The claim's float64 reference matches the host's own reference
(`compute_reference=True`) to `5·10^-8` for both formats. Wrong outputs (all zero, a group of 32
products left out, one group's scale doubled, a wrong scale layout, the nibbles of `mixed`'s `b`
swapped, the neighbouring column) each break `Post` on `77%` to `100%` of the elements at
`1024 × 1024 × 512` and `× 8192`. Every input stays unchanged byte for byte, every element of `c`
is written, and the bytes around `c` keep their values.
