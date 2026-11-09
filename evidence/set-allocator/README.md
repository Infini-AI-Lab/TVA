# `triton.set_allocator` on a launch that builds no descriptor (library issue L146)

**Question.** Round 32 excluded descriptors built in the kernel (`Compiler.Faithful`, scope 5), the one source of a
launch's global scratch, and dropped `triton.set_allocator` from scope 10(c)'s list of allowed uses of Triton's API.
A host may still set an allocator: tutorial 06's host calls `triton.set_allocator(alloc_fn)` before its
host-descriptor launch (`claims/attn_fwd_f16`, source line 548). Does setting an allocator change anything for a
launch within scope 5?

**Source.** `triton.set_allocator` sets a context variable and nothing else
(`primitives/triton-3.8.0-allocation.py:26-34`). The launcher reads it only to allocate the global scratch, and calls
it only when the kernel's `global_scratch_size` is positive (`primitives/triton-3.8.0-nvidia-driver.py:301-321`); the
profile scratch has its own allocator, used only for a positive `profile_scratch_size` (instrumentation, scopes 2
and 4). The compiler gives a kernel global scratch only for a `GlobalScratchAllocOp`, which only
`tl.make_tensor_descriptor` creates (`primitives/triton-3.8.0-TMALowering.cpp:166-184`,
`primitives/triton-3.8.0-PipeliningUtility.cpp:596-612`).

**Method** (`run.py`). On an H200 NVL (sm_90), Triton 3.8.0 (torch 2.11.0+cu130), GPU 2, 2026-11-08: the in-scope
kernels of round 32's library review (lr-32a's survey of scratch sizes: host descriptors in a pipelined and a
warp-specialised loop, pointer loads with `tl.dot` in a pipelined loop, block pointers, gather, sort, cumsum,
histogram, atomics and inline asm, `tl.dot_scaled`, a 3-D `tl.dot`), each launched twice on the same inputs: first
with Triton's null allocator (a launch that needed global scratch would raise), then after `triton.set_allocator`
installs an allocator that records its calls and returns zero-filled buffers. The control builds a descriptor in
the kernel.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/set-allocator/run.py` from TVA's root.

**Result** (`output.txt`). Every in-scope kernel has global and profile scratch size `0`, runs with the null
allocator, calls the allocator set by `triton.set_allocator` `0` times, and writes the same bytes as without it.
The control has global scratch `128`, raises with the null allocator, and calls the set allocator once
(`128` bytes, alignment `128`).

**Conclusion.** Within scope 5 the launcher never calls the allocator `triton.set_allocator` sets, so setting one
changes neither the code nor the launch: it is one of the uses of Triton's API that scope 10(c) allows.
