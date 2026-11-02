# Setup of `streamk_matmul_f16`

Set up by the coordinator on 2026-11-01, before any prover started (protocol §3, "Setting up a
kernel folder"). This kernel is not from a tutorial: the user asked for new kernels ("please write
and prove new kernels. [...] 2. stream-k gemm (atomic op). in triton."), and the coordinator wrote
`source/streamk_matmul.py` for this folder.

## The source

`source/streamk_matmul.py` holds:
- `streamk_matmul_kernel`, the `@triton.jit` kernel;
- `streamk_matmul`, the host wrapper that launches it;
- the test under `if __name__ == "__main__":`.

`C = A @ B` with `float16` `A`, `B` and a `float32` `C`. The work is `num_tiles · iters_per_tile`
iterations of the K loop, split into equal contiguous ranges, one per program. For each tile its
range touches, a program computes the partial product over the K iterations it owns. It stores the
tile if it owns all of the tile's iterations, and otherwise adds the partial product into `C` with
`tl.atomic_add`; the wrapper allocates `C` with `torch.zeros`.

## The config

No autotuning. The wrapper launches with `BLOCK_M = BLOCK_N = 128`, `BLOCK_K = 64`, `num_warps=8`,
`num_stages=3`, grid `(num_programs,)`, where `num_programs` is the device's SM count unless the
caller passes one. It also passes `total_iters` and `iters_per_program = cdiv(total_iters,
num_programs)`.

## The launches

The claim should cover the wrapper's launches as the test makes them, with shapes as variables:
- `float16` `A` and `B`, a zero-initialised `float32` `C`, and positive `M`, `N`, `K`;
- the test's shapes and program counts: 512³, 1000³, 3072×4096×1024 and 8192×8192×4096 with the
  SM count (132 on the H200), 257×129×4097 with 7 programs, and 64³ with 3 programs (more programs
  than iterations, so some programs do no work).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), every test shape passes: max error relative to
`|A| @ |B|` between 6.2e-8 and 9.3e-7. Both paths run. At 1000³ every tile is split between two
programs, so every tile is accumulated with atomics. At 8192×8192×4096 some tiles are stored by
one program and others are split.
