# Setup of `cross_entropy_fwd_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`),
`fla/modules/fused_cross_entropy/ops.py:27` `cross_entropy_fwd_kernel`.

## The source

`source/cross_entropy.py` holds:
- `cross_entropy_fwd_kernel`, the `@triton.jit` kernel (lines 67-124), verbatim from
  `fla/modules/fused_cross_entropy/ops.py:26-83`: program `(n, j)` takes block `j` of `BV`
  classes of row `n` of the logits and computes the block's log-sum-exp and its part of the loss
  (the target's logit, label smoothing, `ignore_index`), adding the log-sum-exp and the z-loss
  when the row is one block. All its compile-time paths are kept (logit scale, soft-capping, label
  smoothing, z-loss, split rows);
- its helpers `exp` and `log` (lines 59-64), verbatim from `fla/ops/utils/op.py:44-45, 48-49`
  (fla's default definitions, `FLA_USE_FAST_OPS` unset). fla's module also imports `tanh` from
  `triton.language.extra.libdevice` (`ops.py:16`), used only under soft-capping: the import is
  removed, since the host never turns soft-capping on;
- `cross_entropy_fwd`, the host, adapted from fla's wrapper `cross_entropy_fwd`
  (`ops.py:230-295`);
- `reference`, each block's loss and log-sum-exp in `float64`, and the test under
  `if __name__ == "__main__":`.

Changed from fla's source (each listed with its `file:line` in the module docstring): the kernel
has no autotuning, so it is unchanged. The host drops fla's `@dispatch` (its backend registry,
which falls back to this function on NVIDIA GPUs unless the Gluon backend is enabled, off by
default) and the argument `process_group` (one rank: `world_size = 1`, `rank = 0`), and does not
call `cross_entropy_reduce`, which combines a split row's blocks in a second kernel
(`cross_entropy_fwd_kernel_reduce`): for `NV > 1` it returns the kernel's per-block outputs.

## The config

No decorator is removed: the kernel is `@triton.jit(do_not_specialize=['N'])` in fla, with no
autotuning or heuristics. The wrapper's own choices: `BV = min(triton.next_power_of_2(V),
64 * 1024)` (`8 * 1024` when soft-capping with `V > BV`), `num_warps = 4 if BV < 2048 else
(8 if BV < 8192 else 16)`, `NV = triton.cdiv(V, BV)`, grid `(N, NV)`, `SPLIT = NV > 1` (with one
rank), `V_TOTAL = V`, `V_START = 0`, `s_logits = logits.stride(0)`; `num_stages` is Triton's
default. The compile-time scalars are the wrapper's arguments: `label_smoothing`, `logit_scale`,
`lse_square_scale`, `logit_softcapping`, `ignore_index`.

## The launches

The claim should cover the host's launches from fla's `FusedCrossEntropyLoss` with its defaults
(`fla/modules/fused_cross_entropy/ops.py:462-473`; fla's models use it with `fuse_cross_entropy`,
`fla/models/modeling_utils.py:125-126`), with shapes as variables:
- `logits` `bfloat16` (the `lm_head` output of a bf16 model), `[N, V]`, contiguous in the last
  dimension (`stride(-1) = 1`; the wrapper copies other layouts), row stride `s_logits >= V`;
  `target` `int64`, each value in `[0, V)` or `ignore_index`; `loss`, `lse` and `z_loss`
  `float32`;
- `logit_softcapping = None` (soft-capping off: its `tanh` is libdevice's), `logit_scale = 1.0`,
  `lse_square_scale = 0.0`, `ignore_index = -100` (the defaults);
- `label_smoothing`: `0.0` (the default) and values in `(0, 1)` (fla's tests use `0.1`,
  `tests/modules/test_cross_entropy.py:28-145`);
- `V` from 1 up: one block (`NV = 1`) up to `V = 65536`, split rows (`NV > 1`, `SPLIT`) beyond;
  fla's model configs default to a vocabulary of 32000, and the tests use `V` from 1 to 131071
  (1, 4103, 8193, 32000, 65536, 65537, 131071); common vocabularies of 50277, 128256 and 151936
  give `NV` = 1, 2 and 3;
- `N` (the tokens) from 1 up: the tests use up to 2047 rows, a training batch tens of thousands.

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes on 13 launches from `(N, V) =
(1, 1)` to `(1024, 151936)`, with label smoothing 0 and 0.1, two with a padded row stride, and
four with split rows (`NV` = 2 and 3); about a third of the targets are `-100`. Max abs error
against the `float64` reference: loss 1.3e-6, lse 8.0e-7; `z_loss` (written when `NV = 1`) exactly 0.
