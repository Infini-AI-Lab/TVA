# Setup of `vec_add_f32`

Triton 3.8.0 tutorial 01, "Vector Addition" (`python/tutorials/01-vector-add.py`), kernel
`add_kernel` and its host `add`.

## The source

`source/01-vector-add.py` is the tutorial file, unchanged: the `@triton.jit` kernel `add_kernel`
(lines 29–54), the host `add` (lines 62–78), the tutorial's test (lines 84–93) and its benchmark
(`benchmark`, lines 107–129, run at line 135). The kernel has no `@triton.autotune` and no
`@triton.heuristics`.

## The config

Launched directly: `add` launches `add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)` on
the grid `(cdiv(n_elements, 1024),)`, with no launch option (Triton's defaults: `num_warps = 4`,
`num_stages = 3`, `enable_fp_fusion = True`, `debug = False`).

## The launches

The claim must cover every launch the file makes, through `add(x, y)`:
- `x`, `y` contiguous `float32` vectors of one size `n`, `output = torch.empty_like(x)` (fresh), and
  `n_elements = n`, `BLOCK_SIZE = 1024`;
- the test: `n = 98432` (`torch.rand`, seed 0);
- the benchmark: `n = 2^12, 2^13, …, 2^27` (`torch.rand(size, dtype=torch.float32)`);
- input values: any `float32` values (the file uses `torch.rand`).

## Checked on the H200

`check_gpu.py` (Triton 3.8.0, one H200): on the test and on all sixteen benchmark sizes, `add`'s
output equals the IEEE `float32` sum `x[j] + y[j]` bit for bit (both NaN counted equal) on every
element; on the edges of `Inst.ok` (sizes `1, 2, 1023, 1025, 4097, 98433` with random bit patterns,
NaN, infinities, signed zeros, subnormals, overflow and ties; `add(x, x)`; direct launches with
`BLOCK_SIZE` `1, 16, 64, 128, 1024, 32768`) every element matches, the inputs are unchanged and no
element outside the first `n` of a sentinel-guarded output is written: 26 of 26 cases pass.
