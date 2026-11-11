# `scaled_mm_int8`

**The kernel.** vLLM's `scaled_mm_kernel` (`source/triton_scaled_mm.py`, vLLM v0.33.0's
`compressed_tensors/triton_scaled_mm.py` with the changes `SETUP.md` lists; kernel lines 55–206,
host `triton_scaled_mm` lines 211–302). A 1-D grid of `cdiv(M, BM)·cdiv(N, BN)` programs; each
computes one `BM × BN` tile of `C = (scale_a · scale_bᵀ) ⊙ (A·B) + bias`: an `int32` `tl.dot`
accumulator over `K` from `int8` `A` (`M × K`) and `B` (`K × N`), converted to `float32`, multiplied
by the per-token or per-tensor `scale_a` and the per-channel or per-tensor `scale_b`, plus an
optional bias, converted to the `bfloat16` or `float16` output. The claim is
`Claims.ScaledMmInt8.claim` in `Claim.lean`.

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and:
- every element `C[i, j]`, `i < M`, `j < N`, holds a value `c` of the output dtype;
- when row `i` of `A` and column `j` of `B` hold `int8` values, the scales `sa`, `sb` of the row and
  column are finite, the bias value `β` is finite (`0` without a bias), and there is room in
  `float32` (`Room32`): with `S = Σ_k A[i, k]·B[k, j]` exact and `R = sa·sb·S + β`, `c` is the
  rounding to nearest even (up to NaN payload) of a finite `float32` value within
  `tol = 2^-21·mag + 2^-21` of `R`, `mag = |sa·sb|·(|S| + 1) + |β|`; and when `|R| + tol` is at most
  the output format's largest value, `c` is finite and within `outTol = 2·outU·mag + 2^-20` of `R`
  (`outU = 2^-8` for `bfloat16`, `2^-11` for `float16`);
- every byte outside the matrix `C` is unchanged.

## Conditions

(`Inst.ok`, `Pre`.)
- the output (and bias) dtype is `bfloat16` or `float16`; the tile is the host's heuristic's or the
  default `(32, 32, 32)`;
- `M, N, K > 0`; `K` a multiple of `BLOCK_SIZE_K` (Exclusions), a multiple of `128`, below `2^17`;
  `M ≤ 2^22`;
- the `int32` arithmetic does not wrap (`M + BM`, `N + BN ≤ 2^31`; strides and `BK·stride_ak`,
  `BK·stride_bk` below `2^31`) and the grid has fewer than `2^31` programs;
- distinct elements of `C` are distinct storage elements; the buffers hold the matrices, the scales
  and the bias the kernel reads; the vector scales and the bias are contiguous; the bias's address
  is not `0`; the matrix `C` shares no byte with what the kernel reads;
- the buffers are valid in the initial memory. Any values: `int8` values, any `float32` scales and
  any bias values, NaN and infinities included (then only "holds a value" is claimed).

**The launches covered** (`SETUP.md`): `setup_llama_ok` (every `1 ≤ M ≤ 16384` on 87 Llama-class
`(K, N)` shapes, either dtype, every flag, either layout of `B`), `setup_llama128_ok` (4 shapes with
`K` a multiple of 128 only, `64 < M ≤ 16384`), `setup_vllm_test_ok` and `setup_vllm_plain_ok`
(vLLM's tests). vLLM's rounding test (`K = 32`) is not covered (`K_dim`).

**Device facts** (in `Device P`): `Device.DotI8` (the `int8 × int8 → int32` `tl.dot` is exact when
`|acc| + Σ|a·b| ≤ 2^31 − 1`) and the totality of `tl.dot`. The other operations are exact or IEEE.

## Tolerances

With `u = 2^-24`, `η = 2^-150`: `accumulator.to(tl.float32)` errs by `e1 = |S|·u + η`; the product
by `scale_a` by `e2 = |sa|·e1 + (|sa·S| + |sa|·e1)·u + η`; the product by `scale_b` by
`e3 = |sb|·e2 + (|sa·sb·S| + |sb|·e2)·u + η`; the contractible bias addition adds
`3u·(|sa·sb·S| + e3 + |β|) + 4η`. Expanded, this is at most `6.000001·u·|sa·sb·S| + 3u·|β| +
1.000001·|sa·sb|·η + 1.000001·|sb|·η + 5.000001·η ≤ 2^-21·mag + 2^-21 = tol` (`|sb| < 2^128`).
The output's rounding adds `(|R| + tol)·outU + η_o`, within `outTol`. `Room32` keeps each
intermediate in `float32`'s range. The `int32` accumulator is exact: `K·2^14 < 2^31`.

## Exclusions

- **`K` not a multiple of `BLOCK_SIZE_K`** (`K_div`), a kernel limit: the last trip's masked loads
  (lines 161, 164) have no `other`, so unspecified values enter every output's `tl.dot`. Real use
  reaches it (vLLM's test `(1, 256, 128)`, several Llama-2 down projections at TP ≥ 2).
- **`K ≥ 2^17`** (`K_lt`), a kernel limit: the `int32` accumulator can wrap.
- **A non-contiguous vector scale or bias** (`scales_contig`), a kernel limit: the kernel reads
  element `i` (or `j`) of the storage, with no stride.
- **`int32` limits** (`M_le`, `N_le`, `grid_lt`, `strides`, `steps`) and **a bias at address `0`**
  (`bias_nz`): outside real use.
- **Real shapes only**: `K` not a multiple of `128` (vLLM's rounding test), `M > 2^22`.
- Non-finite scales or bias, values without room in `float32`; other output dtypes; other explicit
  block sizes; the tensor-descriptor and `fp8` paths: outside the claim.

`check_gpu.py` (one H200) checks both clauses of `OutSpec` on 94 host launches and two direct ones
(largest error / `outTol` about `0.5`), and that four wrong results break `Post`.
