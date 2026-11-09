# Tutorial 10's formats on the H200 (library issue L153)

**Question.** Which `--format`s of tutorial 10's `block_scaled_matmul_kernel` run on the H200, and
which are rejected on the host? `docs/exclusions.md` said that its fp4 launches are rejected there;
the fixed-claims recheck of 2026-11-08 reported that `--format mixed` (`e4m3` by `e2m1`, the kernel's
`MIXED_PREC` branch) runs.

**Method.** `run.py` imports the pinned tutorial (`tutorials/10-block-scaled-matmul.py`, not edited)
and, for each format, builds the tutorial's own inputs, scales and host descriptors with its
`initialize_block_scaled(M, N, K, format, compute_reference=True)` and launches its
`block_scaled_matmul` (the kernel, the tutorial's config, `torch.float16` output) at
`M = N = K = 1024`, and `mixed` also at the tutorial's own validation size `8192³`. It prints whether
the launch ran or was rejected on the host (the exception), and for a launch that ran the largest
absolute error against the tutorial's reference and whether the tutorial's own check
(`torch.testing.assert_close`, `atol = rtol = 1e-3`) passes.

**Setup.** Triton 3.8.0, PyTorch 2.11.0+cu130, an NVIDIA H200 NVL (sm_90), 2026-11-08.
`CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/tut10-formats/run.py`; output in `out.txt`.

**Result** (`out.txt`).
- `mxfp4` (`e2m1` by `e2m1`) and `nvfp4` (`e2m1` by `e2m1` with `float8e4nv` scales, groups of 16)
  are rejected on the host before any program runs: `OutOfResources`, shared memory (287864 and
  297080 bytes required, 232448 available) at the tutorial's config (`BLOCK_M/N/K = 128/256/128`).
  Such a launch has no outcome.
- `mxfp8` (`e4m3` by `e4m3`) runs: largest error `0.0625` at `1024³`, the tutorial's check passes.
- `mixed` (`e4m3` by `e2m1`, the `MIXED_PREC` branch, `e8m0` scales, groups of 32) **runs**: largest
  error `0.0625` at `1024³` and `0.2495` at `8192³`, the tutorial's check passes, every output finite.
  Its `tl.dot_scaled` is `"e4m3"` by `"e2m1"`, for which the library states `DotScaledE4m3E2m1`
  (`docs/rules/num.md`).

**Conclusion.** Of tutorial 10's formats on the H200, `mxfp4` and `nvfp4` are rejected on the host at
the tutorial's config, and `mxfp8` and `mixed` run; `docs/exclusions.md` is corrected to say so.
