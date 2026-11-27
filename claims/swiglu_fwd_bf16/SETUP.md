# Setup of `swiglu_fwd_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`),
`fla/modules/activations/ops.py:533` (the CUDA file; `triton_ascend.py` beside it is the Ascend
backend), the kernel `swiglu_fwd_kernel`, launched by the host `swiglu_fwd`, which fla's `swiglu`
(`SwiGLUFunction`) and `swiglu_linear` (`SwiGLULinearFunction`) call.

## The source

`source/swiglu.py` holds:
- `prepare_input` and `alloc_output` (`ops.py:44-74`), verbatim: an input keeps its layout when its
  last dimension has unit stride and its leading dimensions one row stride, and is copied to
  contiguous otherwise; the output is a fresh contiguous tensor;
- `swiglu_fwd_kernel`, the `@triton.jit` kernel (lines 92-102), verbatim from `ops.py:532-542`
  but for the removed autotuner: `z = x · sigmoid(x) · y` in `float32`, one program per `B` flat
  elements;
- `swiglu_fwd`, the host, adapted from `ops.py:587-605`;
- `SwiGLUFunction`, `swiglu`, `SwiGLULinearFunction`, `swiglu_linear`, adapted from
  `ops.py:647-713` (forward passes only);
- `reference`, `x · sigmoid(x) · y` in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla's source (each listed with its `file:line` in the module docstring): only the
imports the file uses; the autotuner `@triton.autotune(...)` removed from the kernel; the host
drops fla's `@dispatch` (its backend registry, which falls back to this function on NVIDIA GPUs)
and passes one configuration, `B` and `num_warps` (new keyword parameters, defaults `4096` and
`8`), in place of the autotuner's choice; the autograd functions drop `@input_guard` (with
`no_guard_contiguous=True` it only selects the inputs' device) and the autocast decorators, and
their backward passes, which launch another kernel (`swiglu_fwdbwd_kernel`), raise.
`reference` and the test are new.

## The config

Removed: `@triton.autotune(configs=_activation_autotune_configs(), key=['D'],
**autotune_cache_kwargs)` (`ops.py:531`): on NVIDIA GPUs, `B` in `[512, 1024, 2048, 4096, 8192]`
times `num_warps` in `[1, 2, 4, 8, 16, 32]` (`ops.py:28-41`).

Chosen: `B = 4096`, `num_warps = 8` by default; the claim covers every power-of-two `B` up to
`2^20`, so every `B` of the list (`num_warps` is performance only). `num_stages` is Triton's
default. The wrapper's own values: grid `(triton.cdiv(T, B),)`, `T = x.numel()`, `D =
x.shape[-1]` (a constexpr), the row strides from `prepare_input`, `stride_z_row = D`; `T` is in
`do_not_specialize`. No heuristics.

## The launches

The claim should cover the host's launches as fla's modules make them, with shapes as variables:
- `swiglu(gate, y)` and `swiglu_linear(gate, y, W, b)` from `GatedMLP`
  (`fla/modules/mlp.py:71-78`: `gate, y = self.gate_proj(x), self.up_proj(x)`; `fuse_swiglu=True`,
  the default, takes `swiglu_linear`, which calls `swiglu_fwd(x, y, output_contiguous=True)`), from
  BitNet's MLP (`fla/models/bitnet/modeling_bitnet.py:75-76`), and as an output gate in the HGRN,
  ABC and YOCO layers (`fla/layers/hgrn.py:139`, `fla/layers/abc.py:234`,
  `fla/layers/yoco.py:228`);
- `x`, `y`, `z` `bfloat16` (fla's models in `bfloat16`); `x`, `y` of one shape `[..., D]`,
  contiguous (the projections' outputs), or views that `prepare_input` keeps (one half of
  `proj(x).chunk(2, dim=-1)`, row stride `2·D`; broadcast rows, stride `0`), or copied to
  contiguous (any other layout);
- `D` the intermediate size, `256·⌈⌊hidden·ratio·2/3⌋ / 256⌉` in `GatedMLP`
  (`fla/modules/mlp.py:44-50`): `5632` for hidden size 2048 (fla's default configs), `11008` for
  4096; or a projection width for the gates; `N` the tokens (batch × sequence), from one decoding
  token to training batches of `2^15` tokens and more (`T` beyond `2^31` makes the argument an
  `int64`);
- `B` any of the autotune list; any finite input values.

## Checked on the H200

`check_gpu.py` on an H200 NVL (Triton 3.8.0, torch 2.11; each case under 4 GB) passes: the
source's four test shapes and its `chunk` of a `(4, 1024, 1376)` projection read in place;
`GatedMLP` shapes `(8, 2048, 5632)`, `(1, 1, 5632)` and `(4, 4096, 11008)`, and `SwiGLULinear`'s
call; every `B` of the autotune list with `num_warps` 1, 4 and 32; every finite `bfloat16` `x`
against eight values of `y` from `2^-133` to `-2^127`, and every finite `y` against eleven values of
`x` around the overflow of `tl.exp(-x)` (`-88`, `-88.5`, `-89`, `-90`, `-1e30`) and at `0`,
`±2^-133`, `1e-3`, `88.5`, `1e30`; exact results up to `|zRef| = 2^127`; `D = 1`, a 1-D input, broadcast rows,
a transposed `y`, a 3-D `chunk`, an empty input; direct launches with `B` from 1 to `2^15`, row
strides `0` and larger than `D`, into a sentinel-filled buffer; and `T = 2^31 + 6144` (the `int64`
class) on the first two programs. On every element inside `Pre`, `Post` holds: the worst error /
tolerance is `0.966` (margin `1.04×`, the `bfloat16` rounding itself), every output for
`x ≤ -89` is a zero, the inputs are unchanged and every sentinel is kept. The negative tests
(`sigmoid(x)·y`, `x·y`, rounding toward zero, `x·sigmoid(x)` rounded to `bfloat16` first, the
sigmoid with `e^x`, `z = 0`) exceed the tolerance, by `1.6×` and more.
