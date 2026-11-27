# Setup of `gelu_and_mul_sparse_bf16`

From vLLM (commit `4ba787ec99`), `vllm/kernels/triton/activation.py:17`, the kernel
`_gelu_and_mul_sparse_kernel`, launched by the host `_gelu_and_mul_sparse_triton_op`.

## The source

`source/activation.py` holds:
- `_gelu_and_mul_sparse_kernel`, the `@triton.jit` kernel (lines 56-92), verbatim from
  `vllm/kernels/triton/activation.py:16-52`: one program per row of `x = [gate | up]`; it loads
  the row's `gate` and `up` (`d` elements each, `BLOCK_SIZE` lanes), computes the mean and the
  biased variance of `gate` with two `tl.sum`s, the standard deviation with `tl.sqrt`, the cutoff
  `mean + std * std_multiplier` (rounding the mean, the standard deviation, the scaled standard
  deviation and the cutoff to the input dtype, as PyTorch's native implementation does), the
  sparse gate `max(gate - cutoff, 0)` (rounded to the input dtype), the tanh-approximate GELU of
  it through `2 * tl.sigmoid(2 * inner) - 1`, rounded to the input dtype, and stores its product
  with `up`;
- `_supports_gelu_and_mul_sparse` (lines 95-110) and its constants `_MAX_INTERMEDIATE_SIZE`,
  `_SUPPORTED_DTYPES` (lines 52-53), verbatim (`activation.py:12-13`, `55-70`);
- `_gelu_and_mul_sparse_triton_op`, the host (lines 113-134), and `gelu_and_mul_sparse_triton`
  (lines 137-141), adapted from `activation.py:73-111`;
- `reference`, PyTorch's native `gelu_and_mul_sparse` in `float64`, and the test under
  `if __name__ == "__main__":`.

Changes from vLLM's source (each listed with its `file:line` in the module docstring): `triton`
and `triton.language` are imported directly, not through `vllm.triton_utils`, and
`torch.library`, `vllm.ir` and `vllm.platforms` are not imported; the host drops its
`@triton_op("vllm::gelu_and_mul_sparse_triton", mutates_args=())` decorator (a `torch.library`
registration) and launches `_gelu_and_mul_sparse_kernel[grid](...)` instead of
`wrap_triton(_gelu_and_mul_sparse_kernel)[grid](...)` (which makes the launch traceable by
`torch.compile`; eagerly it is this launch); `gelu_and_mul_sparse_triton` drops its
`@ir.ops.gelu_and_mul_sparse.register_impl(...)` decorator (vLLM's dispatch registry). The
kernel's body and the host's body are otherwise verbatim. `reference` and the test under
`__main__` are new.

## The config

The kernel has no `@triton.autotune` or `@triton.heuristics`, and its parameters have no
defaults. The configuration is the host's own: grid `(n_rows,)` with
`n_rows = x.numel() // x.shape[-1]`, `d = x.shape[-1] // 2` and
`BLOCK_SIZE = triton.next_power_of_2(d)` as `tl.constexpr`s, `num_warps =
min(max(BLOCK_SIZE // 256, 1), 8)`, `num_stages` Triton's default, and `std_multiplier` a runtime
Python `float` (a `float32` scalar in the kernel). The host returns before the launch when the
output is empty. No flags.

## The launches

The claim should cover the kernel's launches as vLLM makes them, with shapes as variables:
- **The caller.** `GeluAndMulSparse` (`vllm/model_executor/layers/activation.py:303-347`) is
  the MLP activation of Gemma 3n's sparse layers. Its constructor sets
  `std_multiplier = Normal(0, 1).icdf(torch.tensor(activation_sparsity, dtype=torch.float32)).item()`,
  the `float32` quantile as a Python `float`: `1.6448535919189453` for `activation_sparsity = 0.95`.
  `forward_cuda` calls `ir.ops.gelu_and_mul_sparse(x, std_multiplier, "tanh")`, which on CUDA
  dispatches to this kernel first (`vllm/platforms/cuda.py:733`, priority `["triton", "native"]`)
  when `_supports_gelu_and_mul_sparse` accepts the arguments (a contiguous CUDA tensor of
  `float16`, `bfloat16` or `float32`, an even last dimension, `d ≤ 32768`), and otherwise to the
  native implementation (`vllm/ir/ops/activation.py:12-22`, which computes the same function and
  keeps the statistics local to each tensor-parallel shard).
- **The model.** `Gemma3nMLP` (`vllm/model_executor/models/gemma3n.py:229-273`) builds
  `gate_up_proj` as a `MergedColumnParallelLinear` of `[intermediate_size] * 2`, so each
  tensor-parallel rank holds `x = [gate | up]` of its own shard, `d = intermediate_size / tp`, and
  uses `GeluAndMulSparse(activation_sparsity, approximate="tanh")` only on layers whose
  `activation_sparsity_pattern` entry is positive (`gemma3n.py:261-267`, `:474-479`; the other
  layers use `GeluAndMul`). The configurations of `google/gemma-3n-E2B-it` and
  `google/gemma-3n-E4B-it` (not part of the vLLM tree) give `intermediate_size` 8192 (E2B) and
  16384 (E4B) on every layer and `activation_sparsity_pattern` 0.95 on the first ten layers.
- dtype `bfloat16` (Gemma 3n's dtype), for `x` and so for the output;
- `d = 16384 / tp` (E4B) or `8192 / tp` (E2B) for `tp` in 1, 2, 4, 8: `d` from 1024 to 16384;
  `BLOCK_SIZE = d`;
- the rows: the tokens of the batch, from 1 (a decoding step) to the prefill chunk (thousands);
  inputs of more than two dimensions are flattened to rows by the host;
- `std_multiplier = 1.6448535919189453`; vLLM's input generator for this operation and its tests
  use `1.6448536269514722` (`vllm/ir/ops/activation.py:30`);
- any finite activations.

vLLM's tests of this kernel (`tests/kernels/ir/test_activation.py`) launch it on `bfloat16`
(also `float16` and `float32`) inputs of 1, 8 and 64 tokens with `d` in 2048, 4096, 8192, 13824,
16384, 32768, of 5 tokens with `d` in 1, 31, 128, on a `[2, 3, 16384]` input, on rows whose gate
is the constant `3.0`, and on gates holding a NaN or an infinity.

Not part of the claim: `float16` and `float32` inputs (other compiled kernels, which the host
supports); rows whose gate is constant or nearly so (`RowOk`: vLLM's constant-row test and its
`d = 1` rows); non-finite inputs (vLLM's non-finite test); activations of magnitude above `2^32`;
multipliers with `|m| > 64`.

## Checked on the H200

`check_gpu.py` (H200, Triton 3.8.0) runs the host on 82 launches: every `setup_*` of `Claim.lean`
(Gemma 3n E4B and E2B shards, `d` from 1024 to 16384, one token and 8192 tokens; the source's
test shapes; 8192 × 16384 with `mGen`), vLLM's tests' `bfloat16` shapes, inputs of three
dimensions, and edges: gate and up scaled by `0.02`, `10`, `1000`, gate means `±100` and `3`,
multipliers `0`, `0.1`, `±1.6448…`, `8`, `-38.5`, `±64`, `d` = 2, 3, 7, 100, 257, 32767, 32768,
rows with a tenth of the gate within `2·cutTol` of the row's own cutoff, `|gate|, |up|` in
`[2^31, 2^32]` (multipliers `±64` and `1.6448…`), rows of one value repeated but for its
`bfloat16` neighbour (`d = 32768` and `4096`), and gates of scale `2^-40` and `2^-58`. On the 77
launches inside `Pre`, every output is finite, the worst error / tolerance is `0.342` (the
multiplier `0`, where half the gate is above the cutoff), the zero clause holds on every element
it covers, and the direct launches into guarded outputs leave the guard bands and `x` unchanged
and write the host's output bit for bit. The 5 launches outside `Pre` (constant gates, an
all-zero gate, `d = 1`, gates of scale `10^-30`) are run and reported, not checked. Wrong outputs
at the E4B shape and on rows near the cutoff (no sparsity, the cutoff without `m·std`, the cutoff
`mean − m·std`, gate and up swapped, the output scaled by `1 + 2^-3`, the output `0`) exceed their
tolerances by factors of 2.4 to 64. The exact (erf) GELU and the GELU without its cube term stay
within the tolerance (at most `0.30` of it): the claim does not tell these functions apart. Peak
device memory 1.75 GiB.
