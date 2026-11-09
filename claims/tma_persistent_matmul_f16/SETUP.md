# Setup of `tma_persistent_matmul_f16`

From Triton 3.8.0, `python/tutorials/09-persistent-matmul.py`: the kernel
`matmul_kernel_tma_persistent` (upstream lines 389–445), its `@triton.jit` helper `_compute_pid`
(upstream lines 273–280) and its host `matmul_tma_persistent` (upstream lines 448–483), which passes
host `TensorDescriptor`s, on the tutorial's `float16` run (`--prec fp16`, the default).

## The source

`source/09-persistent-matmul.py` is the tutorial's file with an 18-line comment at the top listing
the changes, and three changes; line `N` of the tutorial is line `N + 18` of `source/` up to
upstream line 475, and line `N + 20` from upstream line 476 on (line numbers here and in the folder
are those of `source/`):

1. The `@triton.autotune(configs=matmul_tma_persistent_get_configs(pre_hook=matmul_tma_set_block_size_hook),
   key=["M", "N", "K", "WARP_SPECIALIZE"])` decorator of `matmul_kernel_tma_persistent` (upstream
   lines 385–388) is replaced by four comment lines (lines 403–406): `matmul_kernel_tma_persistent`
   (lines 407–463) is a `@triton.jit` function, still
   `@triton.jit(launch_metadata=_matmul_launch_metadata)`, launched directly.
2. In `matmul_tma_persistent`, two lines are inserted before the launch (lines 494–495): a comment,
   and a call of the config's `pre_hook`, `matmul_tma_set_block_size_hook` (lines 201–211), with the
   three descriptors and the config's `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` and
   `EPILOGUE_SUBTILE`, as the autotuner runs it before its launch. It sets the host descriptors'
   block shapes (below).
3. The launch in `matmul_tma_persistent` (lines 496–502) passes the config's meta-parameters and
   launch options (below) on the line of `WARP_SPECIALIZE=warp_specialize` (upstream line 481,
   line 501 here).

Nothing else changes. `matmul_kernel_tma_persistent` reads no module-level value but `triton`,
`tl` and the helper `_compute_pid` (lines 291–298); it has no parameter defaults. The file's other
kernels (`matmul_kernel`, `matmul_kernel_tma`, `matmul_kernel_persistent`,
`matmul_kernel_descriptor_persistent`) and their hosts keep their autotuning: they are not part of
this setup (`matmul_kernel`, through `matmul`, is the naive reference of the file's test). The
file's test (`validate`, lines 712–736) and benchmark (`bench`, lines 674–698) run from its
`__main__` block (lines 751–778); importing the file runs only its module-level definitions (and,
on CUDA, allocates the cuBLAS workspace, lines 61–64).

## The config

One config of `matmul_tma_persistent_get_configs()` (lines 387–400): the most frequent pick of the
tutorial's autotuner on the H200 at the file's default `float16` launch (`M = N = 8192`, `K = 512`,
`-K 512`), a near tie with other configs on a shared GPU (other `K` pick other tiles, such as
`128 × 128 × 128` at `K = 128, 256`). The config:
- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 64`, `GROUP_SIZE_M = 8`,
  `EPILOGUE_SUBTILE = True`;
- `num_stages = 4`, `num_warps = 8`;
- its `pre_hook` (lines 201–211), done by the host before the launch (line 495): the block shapes of
  the host descriptors `a_desc = [128, 64]`, `b_desc = [256, 64]` and, with
  `EPILOGUE_SUBTILE = True`, `c_desc = [128, 128]` (`[BLOCK_SIZE_M, BLOCK_SIZE_N // 2]`);
- `NUM_SMS` as the host computes it, `torch.cuda.get_device_properties("cuda").multi_processor_count`
  (line 477): `132` on the H200 (NVL and SXM). The claim takes `NUM_SMS` as a variable: any
  positive value below `2^31`, with the host's grid formula;
- grid `(min(NUM_SMS, cdiv(M, 128) · cdiv(N, 256)),)` (lines 485–492): program `p` computes the
  tiles `p, p + NUM_SMS, p + 2·NUM_SMS, …` (`tl.range(start_pid, num_tiles, NUM_SMS, flatten=True,
  warp_specialize=WARP_SPECIALIZE)`, line 432), each tile mapped to a row block and a column block
  by `_compute_pid`'s grouping of `GROUP_SIZE_M = 8` row blocks; `tile_id_c` (lines 426, 444)
  follows `tile_id` for the epilogue, which stores the tile as two `128 × 128` halves (lines
  454–460);
- the flags: `FP8_OUTPUT = False` (`dtype == torch.float8_e4m3fn` with `float16` matrices,
  line 499); `WARP_SPECIALIZE = False`: on the H200 `supports_ws()` is true, but the file calls
  `matmul_tma_persistent` with `warp_specialize = True` on no Hopper device (the test skips it,
  line 733, and so does the benchmark, line 693), so every covered launch passes `False`;
  `flatten=True` is written in the kernel (line 432; the tutorial's `FLATTEN` flag belongs to
  `matmul_kernel_descriptor_persistent`, not to this kernel). `flatten` and `warp_specialize` are
  `tl.range` scheduling keywords, which the model does not represent.

## The launches

The claim covers every launch of `matmul_kernel_tma_persistent` that the file's `float16` code makes
through `matmul_tma_persistent(a, b, False)`:
- `a` `float16` `[M, K]`, contiguous (row stride `K`); `b` the contiguous `float16` `[N, K]` that
  `b.T.contiguous()` makes (row stride `K`); `c` the fresh contiguous `float16` `[M, N]` that
  `matmul_tma_persistent` allocates (line 475); the three host descriptors
  `TensorDescriptor.from_tensor(x, [1, 1])` (lines 481–483, the tensor's shape and strides, the
  default padding `"zero"`), with the block shapes of the pre-hook; `a`, `b` drawn as
  `torch.randn(…, dtype=torch.float16)`;
- the test's first call, `validate(32, 32, 32)` (line 770, the call of line 735): `M = N = K = 32`,
  one partial tile in every dimension (the descriptors' loads pad with zeros and their stores drop
  the lanes outside `c`);
- the test's second call, `validate(8192, 8192, K_range[0])` (line 771), and the benchmark's
  `bench(K)` for each `K` of `range(K_range[0], K_range[1] + 1, K_step)` (lines 775–776, the call of
  line 694): `M = N = 8192`, with `K = 512` by default (`-K 512`, line 753) and
  `K = 128, 256, …, 1024` with the module docstring's `--K_range 128 1024 --K_step 128` (line 35).
  The claim covers any `M, N ≥ 1`, `1 ≤ K ≤ 2^13`, partial tiles included, and row strides that are
  multiples of `8` elements.

Not part of the setup: `matmul_tma_persistent(a, b, True)` (warp specialization, which the file
does not run on the H200); the file's `--prec fp8` run (`float8e4nv` operands and output, a
different instantiation, outside this `float16` claim's scope); the file's other kernels.

## Checked on the H200

`check_gpu.py` (H200 NVL, Triton 3.8.0, `NUM_SMS = 132`, under 2 GB of device memory) runs
`matmul_tma_persistent(a, b, False)` on the nine setup sizes (`M = N = K = 32`; `M = N = 8192`,
`K = 128, 256, …, 1024`, the file's draws and layout), plus edges (`K = 8192`; `M = 1`, `N = K = 8`;
partial tiles `M = N = K = 1000`; `N = 96`, where each tile's second `128`-column half lies wholly
past `N`; `M = 129`, `N = 136`, `K = 72`; inputs scaled by 8 and by `2^-12`, the last with
subnormal operands; NaN and infinities in the inputs) and four direct launches with
`NUM_SMS = 1, 7, 132, 1000` (one program looping over 9 tiles; 33 tiles over 7 programs; 220 tiles
over 132; a grid of 40 for 40 tiles), with `a` and `c` in buffers with padded rows (row strides
above `K` and `N`, `c` inside a guarded buffer). Every element of `c` meets both clauses of `Post`:
the largest error / tolerance is 0.977 (subnormal operands, where the output's rounding is all the
error); 0.89 at `M = N = K = 32`, and 0.92 at `K = 128` down to 0.73 at `K = 1024` on the setup's
`8192` sizes. Four wrong results each break `Post` at `K = 32, 128, 512, 1024` and on the partial
tiles of `M = N = K = 1000`: the exact result rounded to `bfloat16` (ratio 16 to 121), the last
`K` block of 64 products left out (ratio 2950 and more), and, from the kernel's own output, two row
blocks of 128 swapped and the two `128`-column halves of each tile swapped (ratio 13000 and more;
these two need `M`, `N > 128`, so they are not run at `32`). The inputs are unchanged and no byte
outside the matrix `c` is written. The host's own check, `validate`'s `torch.allclose` against the
naive `matmul` with `atol = 1` (lines 701–709), passes at `M = N = K = 32` and at `M = N = 8192`,
`K = 128, 512, 1024` (largest difference `0`).
