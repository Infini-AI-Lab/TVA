# The memory a launch writes besides its buffers

**Question.** A claim's outcome `ok fin` holds only the bytes `init` gives the launch, so no claim
could say that the launch leaves other memory alone, which a host that composes launches needs.
Trusting the compiler implies it (the model writes no other byte), if the launcher and the compiled
code write no memory of their own besides the descriptor scratch that scope 5 of
`Compiler.Faithful` already names. Do they? Library issue L73 (round 11; raised by lr-10b).

**Source.** `primitives/triton-3.8.0-nvidia-driver.py` lines 280-333 (`CudaLauncher`): the launcher
passes the kernel, besides its arguments, a global scratch buffer (`global_scratch_size` bytes per
program, from the allocator `triton.set_allocator` sets; only descriptors built in the kernel need
one, `evidence/desc-scratch`) and a profile scratch buffer (`profile_scratch_size`, for proton
instrumentation, `instrumentation_mode`, which scopes 2 and 4 exclude), and, with
`instrumentation_mode` `gsan`, a sanitizer state (excluded likewise).

**Method.** `run.py` carves every buffer of a launch (inputs, outputs, and for K6 the allocator's
scratch) out of one 4 MiB pool filled with a sentinel byte, with 4096-byte guards between them, runs
the kernel, and counts the pool's bytes that changed during the launch outside the buffers the
kernel's program writes (and the scratch, for K6). Each kernel runs with sentinels `0xA5` and
`0x5A`, checks its results against the host, and prints `global_scratch_size` and
`profile_scratch_size` from the compiled kernel's metadata; each runs in its own process, on GPU 0.

- K1: vector add, `n = 1000`, `BLOCK = 128` (a masked partial tile), 8 programs.
- K2: a row softmax (`tl.max`, `tl.exp`, `tl.sum`), 64 rows of 100 in masked blocks of 128.
- K3: `tl.dot` of `float16` `64 x 64` blocks, `float32` out, a `K` loop with
  `tl.range(num_stages=3)` (pipelined).
- K4: integer and float `tl.atomic_add` into 16 counters from 64 programs.
- K5: `tl.sort`, `tl.cumsum` and `tl.histogram` of a block of 128 `int32`.
- K6: a copy through two descriptors built in the kernel, 16 programs.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/launch-frame/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`). Every kernel was correct with both sentinels, and no byte of the pool
outside the written buffers (and K6's scratch) changed. `global_scratch_size` was `0` for K1-K5 and
`256` (two descriptors) for K6, whose one allocator request was `4096` bytes (16 programs);
`profile_scratch_size` was `0` for all.

**Conclusions**, used by `Rules.Compiler.Faithful` ("Memory"), its rule page, `TVA.Claim` ("Other
memory") and `docs/SEMANTICS.md` ("Launches"):

1. A launch within the scope writes no global memory but what its program writes and, when it
   builds descriptors, its descriptor scratch: the launcher's only other buffer, the profile
   scratch, is empty without instrumentation.
2. So the rule may state the frame its trust in the compiler implies: a valid, well-behaved launch
   in scope writes no byte other than those `init` gives, the descriptor scratch aside. The
   evidence tests kernels of the shapes kernels use; the statement itself rests on the trust in the
   compiler, as the rest of `Compiler.Faithful` does.
