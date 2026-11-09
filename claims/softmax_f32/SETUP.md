# Setup of `softmax_f32`

Triton 3.8.0 tutorial 02, "Fused Softmax" (`python/tutorials/02-fused-softmax.py`), kernel
`softmax_kernel` and its host `softmax`.

## The source

`source/02-fused-softmax.py` is the tutorial file, unchanged: the `@triton.jit` kernel
`softmax_kernel` (lines 84–109), the device properties (lines 115–121), the host `softmax` (lines
124–175), `naive_softmax`, the tutorial's test (lines 186–190) and its benchmark (`benchmark`,
lines 203–226, run at line 229). The kernel has no `@triton.autotune` and no `@triton.heuristics`.

## The config

Launched directly, through a precompiled kernel: `softmax` sets `BLOCK_SIZE =
triton.next_power_of_2(n_cols)`, `num_warps = 8`, `num_stages = 4 if SIZE_SMEM > 200000 else 2`
(`4` on the H200), allocates `y = torch.empty_like(x)`, compiles with
`softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=…,
num_stages=…, num_warps=8, grid=(1,))`, and launches that `CompiledKernel` with the same arguments
on the grid `(num_programs, 1, 1)`, `num_programs = min(NUM_SM · occupancy, n_rows)` (occupancy
from the compiled kernel's registers and shared memory). No other launch option.

## The launches

The claim must cover every launch the file makes, through `softmax(x)`:
- `x` a contiguous `float32` matrix `n_rows × n_cols`, `y` fresh and contiguous;
- the test: `1823 × 781` (`torch.randn`, seed 0), `BLOCK_SIZE = 1024`;
- the benchmark: `4096 × N` for `N = 256, 384, …, 12672` (`128·i`, `i = 2 … 99`, `torch.randn`),
  `BLOCK_SIZE = next_power_of_2(N)` from `256` to `16384`;
- `num_programs` as the host computes it; on the H200 (132 SMs) it is `1056` for
  `BLOCK_SIZE ≤ 2048`, `528` for `4096`, `264` for `8192`, `132` for `16384` (the claim covers every
  value from `1` to `n_rows`);
- input values: any `float32` values (the file uses `torch.randn`).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, one H200): on the test and on all 98 benchmark shapes, every output
is finite and `Post` holds on every row, with worst error / tolerance `0.26` (test) and `0.25`–`0.30`
(benchmark). On the edges of `Inst.ok` (rows with `-inf`, rows near the `float32` range, gaps up
to `1000`, subnormal inputs, rows that are not `RowOk`, `1`, `2`, `100` and `16384` columns,
`BLOCK_SIZE` above `next_power_of_2(n_cols)`, from `1` program to `n_rows`, strided rows) the worst
over every `RowOk` row is `0.42`, the input is unchanged and no element outside the output elements is written. The host's
output rounded to `float16` or `bfloat16` fails `Post`, and so
does the exact softmax with a denominator that drops the row's last column. 112 of 112 cases pass.
