# Setup of `block_scaled_matmul_mxfp8`

Set up by the coordinator on 2026-11-01, before any prover started (protocol §3, "Setting up a
kernel folder", library issue L116). The user's request: "add proof of tutorial 10 mxfp8 kernel."

## The config

Tutorial 10 (`tutorials/10-block-scaled-matmul.py`, Triton 3.8.0) does not autotune
`block_scaled_matmul_kernel`. The host function `initialize_block_scaled` (lines 330–449) fixes the
config, and `block_scaled_matmul` (lines 243–281) passes it to the launch. With
`block_scale_type = "mxfp8"` on the H200 (not Rubin, `is_rubin()` false), the values are:

- `BLOCK_SIZE_M = 128`, `BLOCK_SIZE_N = 256`, `BLOCK_SIZE_K = 128`;
- `VEC_SIZE = 32`, `ELEM_PER_BYTE_A = ELEM_PER_BYTE_B = 1`;
- `num_stages = 4`, `disallow_acc_multi_buffer = True`;
- `rep_m = 1`, `rep_n = 2`, `rep_k = 1`;
- the default `num_warps` (4).

## What changed from the tutorial's source

Nothing. `source/10-block-scaled-matmul.py` is the tutorial's file as it is. It has no
`@triton.autotune`, and no `pre_hook` to replay.

## The folder

`block_scaled_matmul_mxfp8` is for the tutorial's `mxfp8` runs of `block_scaled_matmul_kernel`:
- `a` and `b` are `float8e4nv` (`torch.float8_e4m3fn`), with `e8m0` scales (`MXScaleTensor`) in groups
  of 32 along `K`;
- `tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)` (line 234);
- the output is `float16`, which every call of `block_scaled_matmul` in the file asks for.

Its other formats are not this folder's:
- `mxfp4` and `nvfp4` are rejected on the host on the H200, where they need more shared memory
  than the device has;
- `mixed` (`e4m3` by `e2m1`) would be a folder of its own.

The file's AMD kernel, `block_scaled_matmul_kernel_cdna4`, is out of scope (`docs/exclusions.md`,
targets other than the H200).

**The launches covered.** The script's `__main__` runs nothing on the H200: `supports_block_scaling()`
(line 139) is false on sm_90, so it prints that the example needs block-scaled support and stops.
`docs/exclusions.md` records that the kernel is still covered on the H200, where `tl.dot_scaled`
runs through Triton's emulation. The claim is for the launches the file's host functions make for
`"mxfp8"`, as if that check passed:
- `validate_block_scaled(8192, 8192, 8192, "mxfp8")`, the test (line 749);
- the benchmark's sizes, as `bench_block_scaled(K, block_scale_type="mxfp8")` makes them
  (`M = N = 8192`, `K` a multiple of 128 from the command line; the module docstring's `mxfp8`
  command uses `K` from 8192 to 16384 in steps of 2048, and the default is `K = 512`).

**The benchmark's own launches are not covered** (found by kernel reviewers r-mx8 and r-mx8b,
round 1). The script calls `proton.start(…, hook="triton")` (line 756) before
every `bench_block_scaled` call. That registers Triton launch-enter and launch-exit hooks and a
kernel-load hook (`knobs.runtime.*_hook`). `proton.deactivate()` pauses the profiling but leaves
them installed, and only `proton.finalize()` (line 764) removes them. So the benchmark's warm-up
and timed launches run with a hook installed, which `Compiler.Faithful`'s scope leaves out (point 3,
part (B); `docs/exclusions.md`, "a hook"). The claim covers launches of the benchmark's sizes made
without a hook, and the test's launch, which runs before line 756.

The coordinator ran these host functions on the H200 (GPU 0) at `M = N = K = 512`, `1024` and
`8192`. The kernel compiled, and each result passed the test's
`torch.testing.assert_close(…, atol=1e-3, rtol=1e-3)`.
