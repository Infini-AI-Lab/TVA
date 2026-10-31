# Setup of `grouped_matmul_tma_f16`

Set up by the coordinator on 2026-10-31, before any prover started (protocol §3, "Setting up a
kernel folder", library issue L116).

## The config

Tutorial 08 (`tutorials/08-grouped-gemm.py`, Triton 3.8.0) autotunes `grouped_matmul_tma_kernel`
over `tma_configs` (lines 216-223), keyed by `group_size`, with the decorator at lines 226-229. The
config chosen is the first of that list (`BM = 128`, `BN = 128`, `BK = 64`, `s = 3`, `w = 4`):

```python
triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4)
```

It has no `pre_hook`, so the host does nothing in its place. `NUM_SM` is not part of the config.
The launches pass `NUM_SM=num_sms()` (the device's multiprocessor count) themselves, as in the
tutorial.

## What changed from the tutorial's source

`source/08-grouped-gemm.py` is the tutorial's file with these changes, and nothing else:

1. The `@triton.autotune(tma_configs, key=['group_size'])` decorator of `grouped_matmul_tma_kernel`
   (the tutorial's lines 226-229) is removed. `grouped_matmul_tma_kernel` is now a plain
   `@triton.jit` function, launched directly.
2. Its two launches, in `group_gemm_tma_fn` (lines 367-368) and `triton_tma_perf_fn` (lines
   418-419), pass the config's meta-parameters and launch options:
   `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64` and `num_stages=3, num_warps=4`.
3. The file's other kernel, `grouped_matmul_kernel`, is out of scope (`docs/exclusions.md`; the
   user's decision of 2026-10-28, library issue L37) and is not this folder's. Its own
   `@triton.autotune` decorator (lines 52-92) is removed too, so that the source has none. Its two
   launches, in `group_gemm_fn` (lines 204-211) and `triton_perf_fn` (lines 406-413), pass the
   first config of its list (lines 54-59): `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
   NUM_SM=84`, with the default launch options. For this folder's claim those launches are host
   code. The tutorial's module-level code runs one of them (line 392) before the TMA kernel's
   launch (line 398), and with this change the file still runs as a script.

`tma_configs` is left in the file. Nothing uses it now.

## The folder

`grouped_matmul_tma_f16` is for the tutorial's runs of `grouped_matmul_tma_kernel`, all with
`FP8=False`: `group_A[0].dtype` is `float16` in every call the tutorial makes. So `a`, `b` and `c`
are `float16`. The `FP8=True` path (`float8e4nv`) is never launched by the tutorial and is not this
folder's.
