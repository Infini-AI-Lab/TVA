# Setup of `logsigmoid_fwd_bf16`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`),
`fla/modules/activations/ops.py:250`, the kernel `logsigmoid_fwd_kernel`, launched by the
wrapper `logsigmoid_fwd` (`ops.py:289`).

## The source

`source/logsigmoid.py` holds:
- the `@triton.jit` helpers `exp` and `log` (lines 53-56), verbatim from
  `fla/ops/utils/op.py:44-49`: the branch taken when the environment variable `FLA_USE_FAST_OPS` is
  not `'1'` (its default), `tl.exp` and `tl.log` of the argument converted to `float32`;
- `prepare_input` and `alloc_output` (lines 59-89), verbatim from
  `fla/modules/activations/ops.py:44-74`: the input's row stride (the input itself when its rows are
  contiguous and evenly strided, a contiguous copy otherwise) and the contiguous output;
- `logsigmoid_fwd_kernel`, the kernel (lines 92-103), verbatim from
  `fla/modules/activations/ops.py:249-260`: one program per `B` consecutive elements of the
  flattened tensor computes `(min(0, x) - log(1 + exp(-|x|))) / temperature` in `float32` and
  stores it in the dtype of `y`;
- `logsigmoid_fwd`, the host (lines 106-121), adapted from `fla/modules/activations/ops.py:288-302`;
- `reference` (`logsigmoid(x) / temperature` in `float64`) and the test under
  `if __name__ == "__main__":`.

Changes from fla's source (each listed with its `file:line` in the module docstring): the
autotuner `@triton.autotune(configs=_activation_autotune_configs(), key=['D'], ...)` is removed
from the kernel; the host drops fla's `@dispatch` (its backend registry, which falls back to this
function on NVIDIA GPUs) and passes one configuration explicitly; `exp2` of `op.py`, which the
kernel does not call, is left out. `reference` and the test are new.

## The config

Removed: `@triton.autotune(configs=_activation_autotune_configs(), key=['D'],
**autotune_cache_kwargs)` (`fla/modules/activations/ops.py:248`): on NVIDIA GPUs the
configurations `triton.Config({'B': bs}, num_warps=nw)` for `bs` in `[512, 1024, 2048, 4096, 8192]`
and `nw` in `[1, 2, 4, 8, 16, 32]` (`ops.py:28-41`).

Chosen: `B = 1024`, `num_warps = 4`; `num_stages` is Triton's default. The claim covers every
power of two `B ≤ 2^20`, so every configuration of the list. The wrapper's own values: grid
`(triton.cdiv(T, B),)`, `T = x.numel()`, `D = x.shape[-1]` (a constexpr, and the runtime
`stride_y_row`), `stride_x_row` from `prepare_input`, `temperature` the Python `float` passed
through. No heuristics, no flags. `FLA_USE_FAST_OPS=1`, which makes `exp` and `log` libdevice's
`fast_expf` and `fast_logf`, is another kernel, not covered.

## The launches

The claim should cover the host's launches as fla's callers make them, with shapes as variables:
- dtypes: `x` `bfloat16` (fla's layers run in `bfloat16`; its activation table `ACT2FN` maps
  `'logsigmoid'` to this function, `ops.py:1111-1120`, used as the gate function or feature map of
  the GLA, SimpleGLA, RWKV6 and Raven layers, `fla/layers/gla.py:108, 177`), `y` the
  dtype of `x`; a `float32` `x` is another compiled kernel, not covered;
- `temperature`: a Python `float`, `1.0` by default; fla's tests use `0.5`, `1.0`, `2.0`
  (`tests/modules/test_activation.py:104-116`); GLA-style layers divide the log-sigmoid of their gate
  logits by `gate_logit_normalizer = 16` (`fla/layers/gla.py:92, 241`, there with PyTorch's
  `logsigmoid`); any positive `float` in real use;
- `x` of any shape `[..., D]`: contiguous, or a view whose rows are contiguous and evenly strided
  (one half of `x.chunk(2, dim=-1)`, row stride `2·D`; a broadcast row, stride `0`); other layouts
  are copied to contiguous by `prepare_input` first; `T = x.numel()` from 0 up (an empty tensor
  launches an empty grid);
- shapes: fla's tests `(1, 1, 64)`, `(2, 500, 128)`, `(2, 512, 128)`, `(3, 2048, 1200)`, contiguous
  and as `chunk` halves, and `(63,)` (its strided `[::3]` form is copied first); GLA's gate logits
  `4 × 2048` tokens × key dimension `1024` (hidden size `2048`, `expand_k = 0.5`) and its output
  gate `4 × 2048 × 2048`; a long sequence `65536 × 4096`;
- any finite input values.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0, torch 2.11) runs the host on fla's nine test launches, the
three model-sized ones (up to `65536 × 4096` elements), every finite `bfloat16` input against 19
temperatures from `±2^-126` to `±2^127` (including `0.3`, `1/3`, which `float32` rounds), empty and
one-column shapes, a broadcast row and an evenly strided view; direct launches into a
sentinel-filled output at `B` from `1` to `2^16` with row strides `0`, `D`, `D + 7`, `2·D`; and the
`int64` classes of `stride_x_row` (`2^31 + 5`) and `T` (`2^31 + 7`, the first two programs). Over
310,336,597 cells inside `Pre`, every output is a finite `bfloat16` within `yTol`; the worst error
is `0.995` of it (the `bfloat16` rounding, `2^-8` relative, against `2^-8 + 2^-19`); `x` is
unchanged and every sentinel intact. Of the 192,032 cells outside `Pre` (`|x| + 1 > 2^125·|t|`),
59,607 violate `Post`. Wrong outputs (`y = 0`, no temperature, no log term, the log term's sign
flipped, the neighbouring element, one `bfloat16` ulp off) exceed the tolerance.
