# In-kernel tensor descriptors and the launch's scratch memory

**Question.** A kernel that calls `tl.make_tensor_descriptor` builds each descriptor in global
memory that is not one of its arguments. Where does that memory come from, how much of it is
there, which bytes does the launch write, and what happens without an allocator or with a bad one?
Library issue L5 (round 8): the semantics has no such memory, so "every byte the claim does not
name is unchanged" needs a statement about it.

**Sources** (Triton 3.8.0).
- `primitives/triton-3.8.0-TMALowering.cpp` lines 166-184 (`TMACreateDescLowering`):
  each `tt.make_tensor_descriptor` becomes a `ttg.global_scratch_alloc` of `TMA_SIZE_BYTES` bytes
  aligned to `TMA_ALIGN`, the descriptor is written there (`createTMADesc`), then fenced
  (`TensormapFenceproxyAcquireOp`); `primitives/triton-3.8.0-TMAUtilities.h`
  lines 11-12: `TMA_SIZE_BYTES = 128`, `TMA_ALIGN = 128`.
- `primitives/triton-3.8.0-GlobalScratchMemoryAllocation.cpp` lines 71-104: every scratch
  allocation of the kernel gets its own offset (no reuse); the sum is the module's
  `ttg.global_scratch_memory_size`, reported as `metadata.global_scratch_size`
  (`primitives/triton-3.8.0-nvidia-compiler.py` lines 471-472).
- `primitives/triton-3.8.0-TritonGPUToLLVM-Utility.cpp` lines 1490-1548 (`getScratchPtrImpl`): a program's
  scratch starts at `base + linear program id x global_scratch_size` (times `num_ctas`, plus the CTA
  id, with clusters), so the programs' regions are disjoint.
- `primitives/triton-3.8.0-nvidia-driver.py` lines 297-333 (`allocate_scratch` 301-307, called at 316) (`CudaLauncher.__call__`): when
  `global_scratch_size > 0`, each launch calls the allocator once with
  `grid_size x num_ctas x global_scratch_size` bytes, `global_scratch_align`, and the launch's stream,
  and passes the returned buffer's `data_ptr()` as an extra kernel argument
  (`primitives/triton-3.8.0-nvidia-driver.c` lines 1432, 1441, 1502).
- `primitives/triton-3.8.0-allocation.py` lines 17-29: the default allocator raises `RuntimeError`;
  `triton.set_allocator` sets a `ContextVar`.

**Method.** `run.py` compiles kernels that copy a 128 x 128 `float32` matrix plus one through
32 x 32 blocks, with one, two or three descriptors built in the kernel, and launches them on GPU 0,
each case in its own process:

- D1: no allocator set (the default).
- D2: an allocator that records each request and returns `torch.empty(size, int8)`; grids (1, 1),
  (4, 4), (2, 3), (4, 4) again (the compiled kernel reused); a side stream; `num_ctas=2`; a
  `warmup` (compile only).
- D3: the scratch is the middle of a larger buffer filled with a sentinel; the input and the
  outputs carry 4096-byte guard regions filled with the same sentinel; run with sentinels `0xA5`
  and `0x5A` and take the union of changed bytes.
- D4: adversarial allocator that returns the first bytes of the input `x` as the scratch.
- D5: the same copy with descriptors built on the host (`triton.tools.tensor_descriptor.TensorDescriptor`
  arguments) and no allocator.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-scratch/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`).

- D1: the launch raises `RuntimeError: Kernel requires a runtime memory allocation, but no
  allocator was set. Use triton.set_allocator to specify an allocator.` on the host; no output byte
  changed (no program ran).
- D2: exactly one request per launch, also when the compiled kernel is reused; never during
  `warmup`. `global_scratch_size` is 128 bytes per descriptor built (128, 256, 384 for one, two,
  three), `global_scratch_align` 128; the size asked is programs x scratch size (128 x 1, 2048 for
  16 programs, 768 for 6, ...), and with `num_ctas=2` programs x 2 x scratch size (8192 for 16
  programs, two descriptors). The stream passed is the launch's (also on a side stream). All
  results correct.
- D3: in every program's region all 128 bytes of each descriptor are written, and nothing else:
  the guards around the scratch, the input with its guards, and the guards around the outputs are
  unchanged; the outputs are correct.
- D4: with the scratch inside `x`, the launch overwrites the first 4096 bytes of `x` (1024
  elements: 16 programs x 256 bytes) and about as many output elements are wrong (1023 or 1024,
  depending on timing). No fault.
- D5: host-built descriptors need no scratch (`global_scratch_size = 0`) and no allocator; the
  result is correct.

**Conclusions.**

1. A launch of a kernel that builds descriptors in the kernel writes, besides what its operations
   write, the scratch buffer its allocator returned for that launch: each program writes its own
   `global_scratch_size` bytes (128 per descriptor built), and no other memory.
2. Without an allocator the launch is rejected on the host with `RuntimeError` before any program
   runs (no outcome; L3's rejected launches).
3. The library cannot see the allocator's buffer: if it overlaps a byte the launch reads or writes,
   or something else writes it during the launch, the results are wrong (D4). A claim covering such
   a launch must require that the scratch holds no byte of the memory the claim is about (`init`'s
   buffers) and that nothing else writes it while the launch runs; then the scratch's contents are
   not observable through the claim.
4. Descriptors built on the host and passed as arguments use no scratch.
