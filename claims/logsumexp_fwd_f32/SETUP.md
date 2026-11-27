# Setup of `logsumexp_fwd_f32`

From flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`),
`fla/modules/fused_linear_cross_entropy/ops.py:50`, the function `logsumexp_fwd_kernel`, and its
host `logsumexp_fwd` (lines 94–107). On NVIDIA GPUs the host's only caller in fla is its test
(`tests/modules/test_cross_entropy.py:415-421`); fla's fused linear cross entropy on Ascend NPUs
uses a counterpart of this kernel (`triton_ascend.py`) for the log-sum-exp of its logits.

## The source

`source/ops.py` holds:
- `exp`, `log`, `tanh` (lines 55–64): the `@triton.jit` helpers of `fla/ops/utils/op.py:44-53`
  in the default branch (`FLA_USE_FAST_OPS` not `'1'`): `tl.exp`, `tl.log` and libdevice `tanh`
  of the argument converted to `float32`;
- `logsumexp_fwd_kernel` (lines 67–91), the kernel's signature and body verbatim from
  `ops.py:49-73`;
- `logsumexp_fwd` (lines 94–108), the host, adapted from `ops.py:94-107`;
- `reference`, the log-sum-exp in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla's source (each listed with its `file:line` in the module docstring): the
decorator `@triton.heuristics` (`HAS_SCALE`, `HAS_SOFTCAPPING`) is removed and the host passes the
two flags it computed; the host drops fla's `@dispatch` (its backend registry, which calls this
function on NVIDIA GPUs); the helpers of `op.py` are copied in (the fast-math branch is not).
`reference` and the test are new.

## The config

No autotuning. Removed: `@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None,
'HAS_SOFTCAPPING': lambda args: args['softcapping'] is not None})`. The host's values: grid
`(N, ND)`, `BD = min(triton.next_power_of_2(D), 64 * 1024)`, `ND = triton.cdiv(D, BD)`, `D` and
`BD` constexprs, `scale` and `softcapping` runtime scalars (or `None`); `num_warps` and
`num_stages` are Triton's defaults.

## The launches

The claim should cover the host's launches with shapes as variables:
- `x` `float32`, contiguous, any shape `[..., D]` viewed as `[N, D]` (logits of a chunk of
  tokens, `D` the vocabulary size; fla's test passes a `float32` `2 × 1025`); `z` fresh,
  `float32`, `[N, ND]`;
- `scale = None` or a Python `float` (a logit scale; fla's test passes `1.0`); `softcapping =
  None` (fla's test passes `0.5`; the claim does not cover softcapping);
- `D` from 1 up; fla's test uses `2 × 1025`, and common vocabularies give `D = 32000, 50257, 128256, 151936, 262144` (`ND` from 1 to 4);
- `N` from 0 up (the tokens of a chunk, thousands).

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host on 16 launches: the host test's nine shapes
from `1 × 1` to `8 × 262144` (scales `None`, `0`, `0.5`, `1`, `30`), logits chunks of
`2048 × 32000`, `1024 × 128256`, `512 × 151936` and `256 × 262144`, fla's test input (zeros,
`2 × 1025`), a 3-D input and `N = 0`. It runs 46 direct launches into sentinel-filled buffers:
`(D, BD)` from `(1, 1)` to `(2^20, 2^16)` with and without a scale, `2^20` rows, `|x| = 2^126`
without a scale, `|y| ≈ 2^20` with scales `1`, `−1`, `2^-10`, scales `2^-126`, `2^126`, `−2^126`,
`2^-140` (subnormal in `float32`) and `10^-300` (`0` in `float32`), subnormal inputs, constant
rows, one dominant column and gaps up to `2000`. Every block is finite and within the tolerance:
the worst error / tolerance is `0.051` on the host's launches and `0.069` on direct launches;
`x` is unchanged and every sentinel intact. 21 wrong outputs (without `+ m`, the block's largest
column left out, the whole row instead of the block, base 2, without the scale, rounded to
`bfloat16` or `float16`, the next block's value) exceed the tolerance, the closest by `152×`. On
blocks that are not `BlockOk` the kernel stores NaN for a NaN, `+inf` or all-`−inf` row, `2^127`
for a row of `2^127`, and accurate values near `1.7·10^8` for `|y| ≈ 2^26` with a scale.
