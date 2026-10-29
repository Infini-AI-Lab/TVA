# Triton's environment: `TRITON_F32_DEFAULT` and `TRITON_DEFAULT_FP_FUSION`

Library evidence (not a kernel's), for library issue L18. Triton 3.8.0 (`~/fpv/tva/.venv`, torch
2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script: [run.py](run.py); output:
[out.txt](out.txt).

**Question.** `Rules.Compiler.Faithful` covers only launches with Triton's environment at its
defaults (`docs/rules/compiler-faithful.md`). Do two of the variables it requires unset change
what is compiled, and is the unset behaviour what the model assumes: a `tl.dot` without
`input_precision` uses `tf32` (`DotCfg`), and `enable_fp_fusion` defaults to `true`
(`Launch.fusion`)?

**What the library reads.**
- `TRITON_F32_DEFAULT` is `knobs.language.fp32_default`, `TRITON_DEFAULT_FP_FUSION` is
  `knobs.language.default_fp_fusion`, default `True` (`primitives/triton-3.8.0-knobs.py:495-497`).
- `tl.dot` without `input_precision` takes `knobs.language.fp32_default or ("tf32" if
  (supports_tf32 and (allow_tf32 or allow_tf32 is None)) else "ieee")` (`primitives/triton-3.8.0-core.py:2385-2388`).
- The NVIDIA backend sets `enable_fp_fusion` to `knobs.language.default_fp_fusion` when neither the
  source nor the launch gives it (`primitives/triton-3.8.0-nvidia-compiler.py:211-212`).
- In TTIR, `tt.dot` prints `inputPrecision = …` only when it is not the attribute's default `IEEE`
  (`primitives/triton-3.8.0-TritonOps.td:654`, `:663`).

**Method.** Every case runs in a fresh subprocess with all `TRITON_*` variables removed, a new
empty `TRITON_CACHE_DIR`, and only the variable under test set.
- `TRITON_F32_DEFAULT` unset, `ieee`, `tf32`, `tf32x3`, `bogus`: a float32 16×16 `tl.dot(a, b)`
  without `input_precision`, and as controls with `allow_tf32=False` and with
  `input_precision="ieee"`. Every element of `a` is `1 + 2^-12` (not a tf32 value) and `b` is all
  ones, so the exact result is `16 + 2^-8`; tf32 inputs give `16`.
- `TRITON_DEFAULT_FP_FUSION` unset, `0`, `1`: `y = a * b + c` in float32, at the default
  `enable_fp_fusion` and with `enable_fp_fusion=True` at the launch, with `a = b = 1 + 2^-13`,
  `c = -(1 + 2^-12)`: fused `2^-26`, unfused `0`. Counted: `fma.rn.f32`, `mul.rn.f32`,
  `add.rn.f32` in the PTX.

**Result.**

| Environment | `tl.dot(a, b)` | `allow_tf32=False` | `input_precision="ieee"` |
|---|---|---|---|
| `TRITON_F32_DEFAULT` unset | `tf32`, result `16` | `ieee`, `16 + 2^-8` | `ieee`, `16 + 2^-8` |
| `= ieee` | `ieee`, `16 + 2^-8` | `ieee` | `ieee` |
| `= tf32` | `tf32`, `16` | **`tf32`**, `16` | `ieee` |
| `= tf32x3` | `tf32x3`, `16 + 2^-8` | **`tf32x3`** | `ieee` |
| `= bogus` | rejected on the host (`CompilationError`) | rejected | `ieee` |

| Environment | default `enable_fp_fusion` | `enable_fp_fusion=True` at the launch |
|---|---|---|
| `TRITON_DEFAULT_FP_FUSION` unset | `True`: 1 `fma.rn.f32`, `y = 2^-26` | `True`, fused |
| `= 0` | **`False`**: 1 `mul.rn.f32` + 1 `add.rn.f32`, no `fma`, `y = 0` | `True`, fused |
| `= 1` | `True`, fused | `True`, fused |

**Conclusions.**
1. With both variables unset, a `tl.dot` without `input_precision` on float32 operands uses `tf32`
   (`ieee` with `allow_tf32=False`), and `enable_fp_fusion` defaults to `True`: what the model
   (`DotCfg`, `Launch.fusion`) assumes.
2. Either variable, when set, changes what a source without the corresponding argument compiles to:
   `TRITON_F32_DEFAULT` even overrides `allow_tf32=False`, and `TRITON_DEFAULT_FP_FUSION=0` turns
   fusion off. An explicit `input_precision` or launch-time `enable_fp_fusion` is not overridden.
   So the model's reading of a source depends on these variables, and `Compiler.Faithful` requires
   them unset.
