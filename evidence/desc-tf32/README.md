# A host `TensorDescriptor` with `round_f32_to_tf32=True`

**Question.** Triton 3.8.0's host `TensorDescriptor` (`triton.tools.tensor_descriptor`) has a field
`round_f32_to_tf32`, `False` by default and accepted only for `float32` tensors. With it set, does
`desc.load` return the stored values, as the library's `tl.desc_load` says? Library issue L76
(round 15; reported by the round-11 author).

**Sources** (Triton 3.8.0). `primitives/triton-3.8.0-nvidia-driver.py` lines 231-232: with the
field set, the launcher builds the TMA descriptor with the element type `TMA_TF32` instead of
`float32`. `primitives/triton-3.8.0-core.py` lines 2364-2367 (`tl.dot`'s documentation) advises
the field to round `float32` inputs to tf32 for a `tl.dot`.

**Method.** `run.py` loads a `16 x 16` `float32` block through a host descriptor and stores it back
through a pointer, once with the field `False` (T1) and once `True` (T2). The block holds values
with low mantissa bits set (`1 + 2^-23`, `1 + 2^-13 + 2^-14`, `1 + 2^-11`, `1 + 2^-10 + 2^-11`,
`-3.0000002`, `0.1`), `1.0` and the subnormal `2^-130`. Each case runs in its own process, with its
own Triton cache.

Run: `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -I evidence/desc-tf32/run.py` from TVA's root.

**Triton version.** 3.8.0 (torch 2.11.0+cu130).
**GPU.** NVIDIA H200 NVL (sm_90), GPU 0.
**Date.** 2026-10-28.

**Result** (`output.txt`). T1 returned every stored value. T2 returned the six values with low
mantissa bits rounded to 10 mantissa bits, to nearest (`0x3f800001` → `0x3f800000`, `0x3f803000` →
`0x3f804000`, `0x3dcccccd` → `0x3dccc000`), and `1.0` and the subnormal unchanged.

**Conclusion**, used by `docs/exclusions.md` and `Rules.Compiler.Faithful` (scope 8): with
`round_f32_to_tf32=True`, a load through the host descriptor returns other values than memory holds,
which `tl.desc_load` does not model. No pinned tutorial sets the field. The library excludes it: the
rule covers a launch only when no host descriptor it is given has the field set.
