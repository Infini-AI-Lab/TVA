# Setup of `persistent_matmul_f16`

From Triton 3.8.0, `python/tutorials/09-persistent-matmul.py`: the pointer kernel
`matmul_kernel_persistent` (upstream lines 287–343), its `@triton.jit` helper `_compute_pid`
(upstream lines 273–280) and its host `matmul_persistent` (upstream lines 346–366), on the tutorial's
`float16` run (`--prec fp16`, the default).

## The source

`source/09-persistent-matmul.py` is the tutorial's file with a 14-line comment at the top listing
the changes, and two changes; line `N` of the tutorial is line `N + 14` of `source/` (line numbers
here and in the folder are those of `source/`):

1. The `@triton.autotune(configs=matmul_get_configs(), key=["M", "N", "K"])` decorator of
   `matmul_kernel_persistent` (upstream lines 283–286) is replaced by four comment lines
   (lines 297–300): `matmul_kernel_persistent` (lines 301–357) is a `@triton.jit` function, still
   `@triton.jit(launch_metadata=_matmul_launch_metadata)`, launched directly.
2. The launch in `matmul_persistent` (lines 372–379) passes the config's meta-parameters and launch
   options (below) on the line of `NUM_SMS=NUM_SMS` (upstream line 364, line 378 here).

Nothing else changes. `matmul_kernel_persistent` reads no module-level value but `triton`, `tl`
and the helper `_compute_pid` (lines 287–294); it has no parameter defaults. The file's other
kernels (`matmul_kernel`, `matmul_kernel_tma`, `matmul_kernel_tma_persistent`,
`matmul_kernel_descriptor_persistent`) and their hosts keep their autotuning: they are not part of
this setup (`matmul_kernel`, through `matmul`, is the naive reference of the file's test). The
file's test (`validate`, lines 706–730) and benchmark (`bench`, lines 668–692) run from its
`__main__` block (lines 745–772); importing the file runs only its module-level definitions (and,
on CUDA, allocates the cuBLAS workspace, lines 57–66).

## The config

One config of `matmul_get_configs()` (lines 104–113), the one the tutorial's autotuner picks on the
H200 for the file's `float16` launches (`M = N = 8192`, `K = 128, …, 1024`: at every `K = 512` run
and 12 of 16 runs in all; the other picks differ only in `num_stages = 4`), with no `pre_hook`:
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`;
- `num_stages = 3`, `num_warps = 8`;
- `NUM_SMS` as the host computes it, `torch.cuda.get_device_properties("cuda").multi_processor_count`
  (line 364): `132` on the H200 (NVL and SXM). The claim takes `NUM_SMS` as a variable: any
  positive value below `2^31`, with the host's grid formula;
- grid `(min(NUM_SMS, cdiv(M, 128) · cdiv(N, 256)),)` (line 371): program `p` computes the tiles
  `p, p + NUM_SMS, p + 2·NUM_SMS, …` (`tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)`,
  line 326), each tile mapped to a row block and a column block by `_compute_pid`'s grouping of
  `GROUP_SIZE_M = 8` row blocks;
- the compile-time condition `c_ptr.dtype.element_ty == tl.float8e4nv` (line 353) is false
  (`float16` output).

## The launches

The claim covers every launch of `matmul_kernel_persistent` that the file's `float16` code makes
through `matmul_persistent(a, b.T)` on whole tiles:
- `a` `float16` `[M, K]`, contiguous (strides `(K, 1)`); `b.T` the transpose of the contiguous
  `float16` `[N, K]` that `b.T.contiguous()` makes (strides `(1, K)`); `c` the fresh contiguous
  `float16` `[M, N]` that `matmul_persistent` allocates; `a`, `b` drawn as
  `torch.randn(…, dtype=torch.float16)`;
- the test's second call, `validate(8192, 8192, K_range[0])` (line 765, the call of line 715), and
  the benchmark's `bench(K)` for each `K` of `range(K_range[0], K_range[1] + 1, K_step)`
  (lines 769–770, the call of line 682): `M = N = 8192`, with `K = 512` by default (`-K 512`,
  line 747) and `K = 128, 256, …, 1024` with the module docstring's
  `--K_range 128 1024 --K_step 128` (line 31). The claim covers any `K ≤ 2^13`.

Not part of the setup: the test's first call, `validate(32, 32, 32)` (line 764), whose `M = N = 32`
are partial tiles, where the kernel's own hints of lines 334–335 are false (a kernel limit:
undefined behaviour under the library's semantics; README.md, Exclusions); the file's `--prec fp8` run (a different
instantiation, `float8e4nv` operands and output); the file's other kernels.

## Checked on the H200

`check_gpu.py` (H200 NVL, Triton 3.8.0, `NUM_SMS = 132`, under 1 GB of device memory) runs
`matmul_persistent(a, b.T)` on the eight setup sizes (`M = N = 8192`, `K = 128, 256, …, 1024`, the
file's draws and layout), plus edges (`K = 8192`; `K = 4095` with `M = 1408`, `N = 768`; one tile
with `K = 65`; `K = 1`; `K = 0`, where `c = 0` exactly; `M = 0`; inputs scaled by 8 and by `2^-12`
(subnormal operands); NaN and infinities in the inputs) and four direct launches with
`NUM_SMS = 1, 7, 132, 1000` (one program looping over 9 tiles; 33 tiles over 7 programs; 220 tiles
over 132; a grid of 40 for 40 tiles), a padded or column-major `c` inside a guarded buffer and a
transposed `a`. Every element of `c` meets both clauses of `Post`: the largest error / tolerance is
0.977 (subnormal operands, where the output's rounding is all the error); 0.92 at `K = 128` down
to 0.74 at `K = 1024` on the setup's sizes. Three wrong results (rounded to `bfloat16`, accumulated
in `float16` step by step, the last step's products left out) each break `Post` at `K = 128, 512,
1024`. The inputs are unchanged and no byte outside the matrix `c` is written.
