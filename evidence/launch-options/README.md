# Knobs, hooks and launch options that change what Triton compiles

Library evidence (not a kernel's), for library issues L18 and L19. Triton 3.8.0
(`~/fpv/tva/.venv`, torch 2.11.0+cu130), NVIDIA H200 NVL (sm_90), GPU 0, 2026-10-27. Script:
[run.py](run.py); output: [out.txt](out.txt).

**Question.** Besides the environment variables of [env-knobs](../env-knobs/README.md), can the
host change what a launch computes from Python alone: by assigning a `triton.knobs` knob, by
installing a hook, or by a launch keyword that `TVA.Launch` does not record? And do the options
`Rules.Compiler.Faithful` lets vary (`num_warps`, `num_stages`, `maxnreg`,
`launch_cooperative_grid`) change any exact result?

**What the library reads.**
- Every field of `CUDAOptions` is a launch keyword (`primitives/triton-3.8.0-nvidia-compiler.py:108-135`;
  `primitives/triton-3.8.0-jit.py:713-715` rejects only keywords that are neither options nor
  parameters).
- `knobs` reads each environment variable only as the default of a knob that Python can assign
  (`primitives/triton-3.8.0-knobs.py:68-100`: an assigned value replaces the variable; `setenv`, `:33-40`, also sets the variable unless `propagate_env` is `False`, `:30`); `knobs.runtime.debug` is
  a plain attribute read once at import (`:474`), and `run` turns `debug` on when it is set
  (`jit.py:727`).
- Hooks: `knobs.runtime.*_hook` and the listeners (`knobs.py:470-492`, `:372`, `:385`); `run` calls
  the pre-run hooks (`jit.py:735-736`), and returns without compiling or launching when
  `jit_cache_hook` returns true (`jit.py:880-882`, `:757-759`).

**Method.** Each case runs in a fresh subprocess with every `TRITON_*` variable removed and an empty
cache; it prints the `TRITON_*` variables set when the kernel compiles, and the result.
- `tl.dot(a, b)` on float32 without `input_precision` (`a = 1 + 2^-12`, `b = 1`; tf32 gives 16,
  ieee `16 + 2^-8`): D0 control; D1 `allowed_dot_input_precisions=("ieee",)` at the launch; D2
  `default_dot_input_precision="ieee"` at the launch; D3 `knobs.propagate_env = False`, then
  `knobs.language.fp32_default = "ieee"`; D4 the same knob with the default `propagate_env`.
- `y = x + 1.0`: A0 control; A1 `ir_override=` the kernel's own PTX with `1.0` replaced by `2.0`;
  A2 `knobs.runtime.add_stages_inspection_hook`, rewriting the PTX stage; A3
  `knobs.runtime.jit_cache_hook` returning true.
- `y = a * b + c` (fused `2^-26`, unfused `0`): F0 control; F1 `knobs.propagate_env = False`, then
  `knobs.language.default_fp_fusion = False`.
- `tl.math.exp2` at `-130, -126, -149, 0`: E0 control; E1 `enable_reflect_ftz=False` at the launch.
- `tl.device_assert(x > 0)` with `x = -1`: S0 control; S1 `knobs.runtime.debug = True`.
- B: one kernel with exact results (an `int8` `tl.dot` into `int32`, an integer `tl.sum`, a float32
  `x * 2 + 1` with fusion, inputs with NaNs; 64 programs), under `num_warps` 1, 2, 8,
  `num_stages` 1, 2, 4, `maxnreg` 64, 128 and `launch_cooperative_grid=True`, compared bit for bit
  with the defaults.

**Result** (`out.txt`).

| Case | Result | Changed? |
|---|---|---|
| D0 control | tf32 (`16`) | |
| D1 `allowed_dot_input_precisions=("ieee",)` | ieee (`16 + 2^-8`) | yes |
| D2 `default_dot_input_precision="ieee"` | tf32 | no (the frontend reads the knob, not this option) |
| D3 knob `fp32_default`, `propagate_env = False` | ieee, with no `TRITON_*` variable set | yes |
| D4 knob `fp32_default`, default `propagate_env` | ieee; the assignment set `TRITON_F32_DEFAULT` | yes |
| A1 `ir_override` | `y = x + 2` | yes |
| A2 `add_stages_inspection_hook` | `y = x - 1` | yes |
| A3 `jit_cache_hook` returning true | the call returns, nothing runs, `y` keeps its initial value | yes |
| F1 knob `default_fp_fusion = False`, `propagate_env = False` | `0`, no `fma.rn.f32` (control: `2^-26`, one `fma`) | yes |
| E1 `enable_reflect_ftz=False` | `ex2.approx.f32`, `exp2(-130) = 2^-130` (control: `ex2.approx.ftz.f32`, `0`) | yes |
| S1 knob `runtime.debug = True` | the device assert traps (control: returns) | yes |
| B `num_warps`, `num_stages`, `maxnreg`, `launch_cooperative_grid` | 0 differing elements in each of 9 variants | no |

**Conclusions.**
1. Triton's knobs change what is compiled however they are set: from the environment
   (env-knobs), or from Python with no environment variable at all (D3, F1, S1). So
   `Compiler.Faithful` asks for the knobs at their defaults, not only the environment.
2. Hooks can rewrite the compiled code (A2) or skip the launch with no error (A3); a launch
   keyword outside `TVA.Launch` can replace the code (A1), the `tl.dot` precision (D1) or libdevice's
   flush-to-zero (E1). So `Compiler.Faithful` asks for no hook and for no compile or launch option
   except those `TVA.Launch` records and the four below. D2 changes nothing here, but the rule
   excludes every option it does not list, without testing each.
3. `num_warps`, `num_stages`, `maxnreg` and `launch_cooperative_grid` changed no exact result in
   B. They change how the work is laid out, not the meaning the semantics gives (a reduction's tree
   is one Triton chooses in any case), so the rule lets them vary.
