# Which launches end with an error on the host, and which return without one

Library evidence (not a kernel's), for library issues L3 and L9. Triton 3.8.0
(`~/fpv/tva/.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27.
Script: [run.py](run.py); output: [out.txt](out.txt).

**Questions.**
1. When Triton or the driver refuses a launch, does the host get an error, before any program
   runs, with memory unchanged? (L3: such a launch has no outcome in `TVA.Claim.Behaviour`.)
2. Does every launch that returns without an error run its grid? In particular, what does the
   launcher do with a program count of `2^31` or more? (L9: `Launch.Valid`.)

**Method.** Each case launches a kernel in which every program adds 1 to one of 4096 `int64`
counters (`pid_0 % 4096`), so the counters' sum after the launch is the number of programs that
ran. For each case the script prints the exception the host got, the number of programs that ran,
and whether the device still runs a control launch afterwards.

**The launcher.** Triton 3.8.0's CUDA launcher reads the grid with `PyArg_ParseTuple` format
`"iii"`, so each dimension is a C `int`, and then launches only
`if (gridX * gridY * gridZ > 0)`, a product computed in C `int`
(`primitives/triton-3.8.0-nvidia-driver.c:968-972`, `:1423`, `:1437`; the installed
`triton/backends/nvidia/driver.c` has the same sha256). It returns normally, with no error, when
the guard fails.

**Result.**

| Case | Host | Programs run |
|---|---|---|
| A1 shared memory over the limit (`OutOfResources`) | error | 0 |
| A2 2048 threads per program (`OutOfResources`) | error | 0 |
| A3 `tl.dot` with `K = 8` (`CompilationError`) | error | 0 |
| A4 `float8e4b8` on CUDA (`CompilationError`) | error | 0 |
| A5 a failing `tl.static_assert` (`CompileTimeAssertionFailure`) | error | 0 |
| A6, A7 grid `y` or `z` = `65536` (`RuntimeError: invalid argument`, from the driver) | error | 0 |
| A8 grid `x` = `2^31` (`OverflowError`, from the argument parser) | error | 0 |
| B1, B2 grids with a zero dimension | no error | 0 |
| C1 `(2^31 − 1, 1, 1)`, C2 `(65535, 2^15, 1)`, C3 `(2^14, 2^14, 2)` | no error | all |
| C4 `(2^30, 2, 1)`, C5 `(2^15, 2^15, 2)`: `2^31` programs | **no error** | **0** |
| C6 `(2^16 + 1, 2^15, 1)`: `2^31 + 2^15` programs | **no error** | **0** |
| C7 `(2^16, 2^15, 2)`: `2^32` programs | **no error** | **0** |
| C8 `(6700417, 641, 1)`: `2^32 + 1` programs (the `int` product is `1`) | no error | all `2^32 + 1` |

After every case the device ran the control launch.

**Conclusions.**
1. Each refusal tried is reported to the host with an exception before any program runs, and
   leaves memory unchanged and the device usable (A1–A8). Such a launch has no outcome.
2. A launch with a zero grid dimension returns without an error and runs no program (B1, B2).
   That is the semantics' normal end with memory unchanged (`Config.Done` holds at once).
3. A launch whose program count is in `[2^31, 2^32]` returns without an error and runs no
   program (C4–C7), although each grid dimension is within CUDA's limits. Above `2^32` the guard
   depends on the product modulo `2^32` (C8 runs). Below `2^31` the whole grid runs (C1–C3).
   So `Launch.Valid` requires `L.size < 2^31`: with it, a valid launch that returns without an
   error ran its whole grid.
