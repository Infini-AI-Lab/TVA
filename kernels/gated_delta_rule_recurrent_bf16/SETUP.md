# Setup of `gated_delta_rule_recurrent_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
flash-linear-attention (fla) v0.6.0 (commit `413ad2c655`),
`fla/ops/gated_delta_rule/fused_recurrent.py:30` `fused_recurrent_gated_delta_rule_fwd_kernel`.

## The source

`source/fused_recurrent_gdn.py` holds:
- `fused_recurrent_gated_delta_rule_fwd_kernel`, the `@triton.jit` kernel, verbatim: the gated
  delta rule token by token; one program per (block of `BV` value channels, sequence, value head)
  carries a `float32` state block in registers through the sequence, writes each token's output,
  and stores the final state;
- the `@triton.jit` helper it calls on the proved path, copied verbatim: `exp` from
  `fla/ops/utils/op.py:44-45` (the branch taken without `FLA_USE_FAST_OPS`, fla's default, on an
  H200 too: `tl.exp` in `float32`). fla's `softplus` (`fla/ops/utils/softplus.py`), which the
  kernel names only under `USE_GATE_IN_KERNEL`, is not copied (below);
- the host: `fused_recurrent_gated_delta_rule` (fla's entry point, `fused_recurrent.py:309-475`,
  with its defaults: `scale = K^-0.5` when not given, `beta = torch.ones_like(q[..., 0])` when not
  given, no initial state and no final state by default) and `fused_recurrent_gated_delta_rule_fwd`
  (the launch, `fused_recurrent.py:184-250`, which allocates the output and, with
  `output_final_state`, a fresh `float32` final state);
- `reference`, the recurrence in `float64`, and the test under `if __name__ == "__main__":`.

Changes from fla, each listed with its `file:line` in the module docstring: the kernel's decorator
`@triton.heuristics` removed (`@triton.jit(do_not_specialize=['T'])` kept); the helper `softplus`
not copied (it is reached only in a compile-time branch the proved launches do not take); the launch passes the
flags the heuristics computed; `FusedRecurrentFunction` (an autograd function whose backward
raises) is not kept, and its `@input_guard` is replaced by `.contiguous()` calls on the tensor
arguments before the launch; the entry point's docstring is shortened, and the
`DeprecationWarning` it issues for the old keyword `transpose_state_layout` dropped.

## The config

- Removed: `@triton.heuristics({'USE_G': …, 'USE_GK': …, 'USE_GV': …, 'USE_INITIAL_STATE': …,
  'STORE_FINAL_STATE': …, 'IS_VARLEN': …, 'USE_GATE_IN_KERNEL': …, 'HAS_DT_BIAS': …})`
  (`fused_recurrent.py:19-28`; each is `args[name] is not None` for `g`, `gk`, `gv`, `h0`, `ht`,
  `cu_seqlens`, `A_log`, `dt_bias`). The kernel has no autotuning.
- Not copied: fla's `softplus` with what it uses (`fla/ops/utils/softplus.py:17-35, 57-65, 68-81,
  111-118`; on NVIDIA its body is inline PTX built from module-level `tl.constexpr` values). The
  kernel calls it only in `if USE_GATE_IN_KERNEL:` (`fused_recurrent.py:132-137`), false on the
  proved path (`A_log` not given), where Triton does not resolve the name.
- Chosen: the wrapper's own launch options, `num_warps=1`, `num_stages=3`, and its block sizes
  `BK = next_power_of_2(K)`, `BV = min(8, next_power_of_2(V))` (`next_power_of_2(V)` with `gv`);
  grid `(NV * N * HV,)`, `NV = cdiv(V, BV)`.
- Flags set by the host for the removed heuristics: `USE_G`, `USE_GK`, `USE_GV`,
  `USE_INITIAL_STATE`, `STORE_FINAL_STATE`, `IS_VARLEN`, `USE_GATE_IN_KERNEL`, `HAS_DT_BIAS`. The
  wrapper itself passes `IS_BETA_HEADWISE = beta.ndim != v.ndim`, `USE_QK_L2NORM_IN_KERNEL`,
  `APPLY_BETA_SIGMOID`, `ALLOW_NEG_EIGVAL` and `STATE_V_FIRST` from its arguments.

## The launches

The claim should cover the entry point's launches as fla's gated DeltaProduct layer makes them in
inference (`fla/layers/gated_deltaproduct.py:263-273`; fla's test, `tests/ops/test_gdn.py:69-124`
`test_fused_recurrent`, calls it the same way): a log-space gate `g`, the in-kernel L2 norm of `q`
and `k`, an initial state, the final state saved, every other argument at the wrapper's default:
- `bfloat16` `q`, `k`, `v` and `o`; `float32` `g` (the log decay, `≤ 0`); `beta` given as
  `bfloat16` `[B, T, HV]` or not given (the wrapper's default); `float32` `initial_state` and
  `final_state`; `scale` given or defaulted to `K^-0.5`;
- `USE_G`, `USE_QK_L2NORM_IN_KERNEL`, `IS_BETA_HEADWISE`, `USE_INITIAL_STATE`,
  `STORE_FINAL_STATE` true; `USE_GK`, `USE_GV`, `STATE_V_FIRST`, `IS_VARLEN`,
  `USE_GATE_IN_KERNEL`, `HAS_DT_BIAS`, `APPLY_BETA_SIGMOID`, `ALLOW_NEG_EIGVAL` false;
- `q`, `k` `[B, T, H, K]`, `v` `[B, T, HV, V]`, `g` `[B, T, HV]`, `initial_state` `[B, HV, K, V]`,
  contiguous;
- shapes as variables: `H` key heads and `HV` value heads with `HV` a multiple of `H` (the entry
  point's docstring: grouped value attention when `HV > H`; fla's tests: `H` from 1 to 4, `HV` up
  to 8; Qwen3-Next-style models 16 and 32), `K` and `V` (fla's tests: 60, 64, 128; gated DeltaNet's
  defaults `K = 256`, `V = 512`, `fla/models/gated_deltanet/configuration_gated_deltanet.py:26-33`),
  any batch size `B`, and `T ≥ 1` (fla's layers use this kernel in inference for `T ≤ 64`,
  `fla/layers/gated_deltanet.py:226-231`; decoding has `T = 1`; fla's tests run `T` up to 2048).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0, torch 2.11), the test passes for fla's test shapes
and larger ones (`H`/`HV` from 1/1 to 16/32, `K`, `V` from 60 to 256 and 512, `T` from 1 to 500,
`B` up to 64), `beta` given and defaulted, fixed-length and
variable-length, with and without the in-kernel L2 norm: the largest errors against the `float64`
reference, relative to `1 + |ref|`, were 6.7e-4 for `o` and 1.9e-7 for the final state.
