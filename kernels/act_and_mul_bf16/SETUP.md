# Setup of `act_and_mul_bf16`

Set up by the coordinator on 2026-11-04 for the sandbox's test set (sandbox/TESTSET.md), from
SGLang v0.5.23 (commit `297e54fd10`),
`python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py:1141` `act_and_mul_kernel`.

## The source

`source/act_and_mul.py` holds:
- `tanh` and `_apply_activation`, the `@triton.jit` helpers the kernel calls, copied verbatim from
  the same file (`fused_moe_triton_kernels.py:1114, 1119`);
- `act_and_mul_kernel`, the `@triton.jit` kernel, copied verbatim but for its parameters'
  defaults (`:1141`): one program per row of
  the gate-up GEMM's output; a row whose expert is `-1` returns at once, the others store
  `act(gate) * up` block by block;
- `act_and_mul`, the host function, adapted from `act_and_mul_triton` (`:1192`);
- `reference` and `live_rows`, test helpers (the activation in `float64`, and the rows the kernel
  writes);
- the test under `if __name__ == "__main__":`.

The kernel keeps both activations (SiLU, and GELU in tanh form with `tanh` built from
`tl.sigmoid`), the optional clamp (`HAS_SWIGLU_LIMIT`) and the optional expert filter
(`HAS_EXPERT_FILTER`). The host differs from the wrapper only in allocating the output itself
(`torch.empty((rows, hidden_size // 2))` in the input's dtype, as its caller `fused_moe.py:713`
allocates it), returning it, and passing the launch options explicitly; the module docstring lists
the changes with their source lines.

## The config

No decorator was removed: the kernel has no `@triton.autotune` or `@triton.heuristics`. The
parameters' defaults are removed (`SWIGLU_LIMIT = 0.0`, `HAS_SWIGLU_LIMIT = False`,
`HAS_EXPERT_FILTER = True`, `fused_moe_triton_kernels.py:1149-1151`): TVA's subset has no kernel
parameter defaults; the host passes each. The configuration is the wrapper's own: grid `(rows,)`, `BLOCK_SIZE=512`, `expert_step` 1 with
`topk_ids` (or `config["BLOCK_SIZE_M"]` with the sorted TMA layout's `expert_ids`),
`SWIGLU_LIMIT=float(swiglu_limit)` and `HAS_SWIGLU_LIMIT=True` when a limit is given,
`HAS_EXPERT_FILTER` when `topk_ids` or `expert_ids` is given. The launch options are
`num_warps=4, num_stages=3`: the wrapper passes none, and these are Triton 3.8.0's defaults.

## The launches

The claim should cover the launches of the configuration SGLang runs, with shapes as variables:
- `bfloat16` `gateup_output` and output; `int32` `topk_ids`;
- `ACTIVATION_TYPE="silu"`, `HAS_SWIGLU_LIMIT=True` with `SWIGLU_LIMIT=10.0`,
  `HAS_EXPERT_FILTER=True` with `topk_ids` (`expert_step = 1`, `down_moe_use_tma=False`). Both
  of the code base's calls of `act_and_mul_triton` pass a limit: the Triton MoE runner, for
  DeepSeek V4's clamp, with `topk_ids` when the experts are filtered (expert parallelism), and it
  asserts `swiglu_limit == 10` (`srt/layers/moe/moe_runner/triton_utils/fused_moe.py:768, 799`);
  and the Humming runner, without a filter (`srt/layers/moe/moe_runner/humming.py:384-395`). The
  GELU path, the path without a limit and the TMA layout stay in the source and are not part of
  the claim;
- `rows = num_tokens * topk` (`fused_moe.py:610`; no TMA padding without the TMA layout), with
  `num_tokens` from 1 to the batch's token count, which only the server's prefill settings bound
  (thousands to tens of thousands of tokens), and `topk` 6 to 8;
  `hidden_size = 2 * I`, `I` the expert's intermediate size divided by the tensor-parallel size
  (DeepSeek-V3's 2048, Qwen3-MoE's 768 and 1536, DeepSeek-VL2's 1407; TP 1 to 8), not
  necessarily a multiple of `BLOCK_SIZE`;
- `topk_ids` values in `[-1, num_experts)`, `-1` marking an expert not on this rank; the rows
  with `-1` are left unwritten (the output is a fresh `torch.empty`).

## Checked on the H200

On the coordinator's H200 run (Triton 3.8.0), the test passes. For the proved configuration
(SiLU, limit 10, `topk_ids` filter), with `(num_tokens, topk, I)` = (1, 8, 256), (7, 8, 2048),
(33, 6, 1408), (512, 8, 768) and (8192, 8, 256) and some experts off the rank, the live rows match
the `float64` reference (the activation rounded to `bfloat16` before the product) within a
relative error of 3.9e-3 (max absolute 0.25, on products near 100), within one `bfloat16`
rounding. The other paths, also run: the TMA layout (3.9e-3), no filter (3.9e-3), SiLU without a
limit (3.9e-3) and GELU (6.7e-3).
