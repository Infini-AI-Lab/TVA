# Programmatic dependent launch and the grid-dependency-control calls

Library evidence (not a kernel's), for library issue L17 (rounds 3 and 4). Triton 3.8.0
(`~/fpv/tva/.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27 (rerun with part
D in round 4). Script: [run.py](run.py); output:
[out.txt](out.txt).

**Questions.**
1. With `launch_pdl` at its default (`False`), do `gdc_wait()` and `gdc_launch_dependents()`
   change any result?
2. Does the launcher accept `launch_pdl=True`, and can a kernel launched with it read memory
   before the previous kernel on the stream has finished?
3. (Round 4.) Can a kernel launched *without* it, the kind a claim covers, see its memory change
   while it runs, when it calls `gdc_launch_dependents()` and the next launch on the stream has
   `launch_pdl=True`?

**What the library reads.**
- `launch_pdl` is an option of the NVIDIA backend, default `False`
  (`primitives/triton-3.8.0-nvidia-compiler.py:124`); the launcher turns a nonzero value into the
  launch attribute `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`
  (`primitives/triton-3.8.0-nvidia-driver.c:1004-1010`, the attribute at `:1006`).
- `gdc_wait` and `gdc_launch_dependents` (`primitives/triton-3.8.0-gdc.py:12-42`) emit
  `griddepcontrol.wait` and `griddepcontrol.launch_dependents` as inline assembly; their
  docstrings say both are safe to execute when programmatic dependent launch is disabled.

**Method.**
- Part A: a kernel `y = 2x + 1` (inputs with NaNs) with and without both calls, at the default
  `launch_pdl`: count the `griddepcontrol` instructions in the PTX, and compare the two outputs bit
  for bit, every element, for `n` = 1, 7, 1023, 1024, 1025, 65536, 1000003 (block 1024).
- Part B: the same kernels launched with `launch_pdl=True` after nothing else.
- Part C (adversarial): a producer calls `gdc_launch_dependents()` first, spins, then writes
  `buf[i] = i + 1` (`2^22` elements, `buf` zero before); the consumer, launched next on the same
  stream, copies `buf` to `out`. 20 runs per variant; an element of `out` that differs from `buf`
  afterwards was read before the producer wrote it.
- Part D (round 4): the kernel under claim is launched with the default `launch_pdl`. It calls
  `gdc_launch_dependents()` first (or not), spins, then reads `x` and writes `y = x + 1`; the next
  launch writes `x` (never `y`), with `launch_pdl` False or True. Run alone, the claimed kernel
  ends with `y = x_init + 1`. 20 runs of `2^22` elements per variant.

**Result.**

| Case | Result |
|---|---|
| A: PTX of the variant with the calls | 1 `griddepcontrol.wait`, 1 `griddepcontrol.launch_dependents`; none without the calls; compiled `launch_pdl` is `False` |
| A: outputs with and without the calls, default `launch_pdl` | 0 differing elements in every size |
| B: `launch_pdl=True` | accepted by the launcher (the compiled kernel's `launch_pdl` is `True`); output right when nothing precedes it |
| C: consumer without `gdc_wait`, `launch_pdl=False` | 0 stale elements in 20 runs |
| C: consumer without `gdc_wait`, `launch_pdl=True` | stale reads in 19 of 20 runs, about 34% of all elements (28327936 of 83886080; 29812736 in round 3's run) |
| C: consumer with `gdc_wait`, `launch_pdl=True` | 0 stale elements in 20 runs |
| D: the claimed kernel without the call, next launch with or without `launch_pdl` | `y = x_init + 1` everywhere in all 20 runs of each |
| D: the claimed kernel with the call, next launch without `launch_pdl` | `y = x_init + 1` everywhere in 20 runs |
| D: the claimed kernel with the call, next launch with `launch_pdl=True` | **wrong `y` in 20 of 20 runs**, 42% of all elements (35225600 of 83886080): the next kernel overwrote `x` while the claimed kernel still read it |

**Conclusions (what the library relies on).**
1. With `launch_pdl=False` (the default), the kernel starts after the previous kernel on the stream
   has finished, and `gdc_wait()` changes no result. `gdc_launch_dependents()` changes no result of
   the kernel itself either, as long as nothing else runs while it does; but it lets the *next*
   launch on the stream, if that launch has `launch_pdl=True`, start early, and then that launch
   may write the memory the kernel still reads (part D). So the rule `Compiler.Faithful` covers a
   launch only while it runs alone, and the model drops
   `gdc_launch_dependents()` only when the next launch on the stream does not use `launch_pdl`
   (library issue L17, round 4).
2. With `launch_pdl=True` the launcher accepts the launch and the kernel may run before the
   previous kernel's writes are done, so its programs may read memory other than the claim's
   initial memory. `Launch.Valid` therefore requires `pdl = false`, and such launches are out of
   scope (`docs/exclusions.md`).
