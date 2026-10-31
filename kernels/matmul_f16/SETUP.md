# Setup of `matmul_f16`

Set up by the coordinator on 2026-10-31, before any prover started (protocol §3, "Setting up a
kernel folder", library issue L116).

## The config

Tutorial 03 (`tutorials/03-matrix-multiplication.py`, Triton 3.8.0) autotunes `matmul_kernel` over
`get_autotune_config()`, which on CUDA is `get_cuda_autotune_config()` (lines 164-199). The config
chosen is the first of that list (lines 166-167):

```python
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
              num_warps=8),
```

It has no `pre_hook`, so the host does nothing in its place.

## What changed from the tutorial's source

`source/03-matrix-multiplication.py` is the tutorial's file with two changes, and nothing else:

1. The `@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])` decorator of
   `matmul_kernel` (the tutorial's lines 228-231) is removed. `matmul_kernel` is now a plain
   `@triton.jit` function, launched directly.
2. The launch in `matmul` passes the config's meta-parameters and launch options:
   `BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8` and
   `num_stages=3, num_warps=8`, before `ACTIVATION=activation`.

The functions that build the config lists (`get_cuda_autotune_config`, `get_hip_autotune_config`,
`get_autotune_config`) are left in the file. Nothing calls them now.

## The folder

`matmul_f16` is for the tutorial's `float16` run: `a`, `b` and `c` are `float16`. The tutorial's
`float8e5` run uses the same source and would be a folder of its own.
