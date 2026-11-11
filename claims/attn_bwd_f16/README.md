# `attn_bwd_f16`

**The kernel.** `_attn_bwd` of Triton 3.8.0's tutorial 06, "Fused Attention"
(`source/06-fused-attention.py`, kernel lines 366–500 with its helpers `_attn_bwd_dkdv`, lines
266–315, and `_attn_bwd_dq`, lines 319–363; host `_attention.backward`, launch at lines 608–621), on
`float16` inputs. The grid is `(N_CTX/128, 1, BATCH·N_HEAD)`; each program computes `128` rows of
`dK` and `dV` of one head, looping over the query rows in steps of `32` (a masked diagonal part
with `CAUSAL`), then `128` rows of `dQ`, looping over the key rows. The claim is
`Claims.AttnBwd.claim` in `Claim.lean`.

## Specification

For head `b < BATCH·N_HEAD` write `Q`, `K`, `V`, `dO` for its `N_CTX × HEAD_DIM` blocks of `q`,
`arg_k`, `v`, `do`, and `L`, `Δ` for its rows of `M` and `delta`, as values in the initial memory
(`HeadV`). For a key row `n` and a query row `m`, `Allowed n m` is `CAUSAL = false ∨ n ≤ m`, and in
the reals:
- `P[n, m] = 2^(Σ_c K[n, c]·Q[m, c] − L[m])` when `Allowed`, else `0`;
- `dS[n, m] = P[n, m]·(Σ_c V[n, c]·dO[m, c] − Δ[m])`;
- `dV[n, j] = Σ_m P[n, m]·dO[m, j]`, `dK[n, j] = s·Σ_m dS[n, m]·Q[m, j]` (`s` the `float32`
  `sm_scale` the kernel receives), `dQ[m, j] = LN2·Σ_n dS[n, m]·K[n, j]` (`LN2` the kernel's
  `float32` constant `0x3f317218`).

**What is claimed.** Every covered launch ends normally (no fault, no hang, no race), and for every
head, row and column, each of `dv`, `dk`, `dq` holds a `float16` value `c`; when the head's values
are covered (`HeadV.Ok`), `c` is the `float16` rounding (nearest even, up to NaN payload) of a
finite `float32` value within `T` (`dVTol`, `dKTol`, `dQTol`) of the exact value `E`, and when
`|E| + T ≤ 65504`, `c` is finite and within `T + (|E| + T)·2^-11 + 2^-25` of `E`. Every byte outside
the three outputs is unchanged. The claim does not assume the host's relations between the inputs
(`M` the log-sum-exp, `delta = Σ O·dO`): it speaks of the values in memory.

`HeadV.Ok`: the inputs finite, `|L|, |Δ| ≤ 2^126`, and for every `Allowed` pair the exponent with
its error at most `15` and the bound on `|P·(dP − Δ)|` at most `2^15`. On the host's inputs `P ≤ 1`.

## Conditions

(`Inst.ok`, `Pre`.)
- `HEAD_DIM ∈ {16, 32, 64, 128, 256}`; `N_CTX` a multiple of `128`, at most `2^14`;
- `BATCH ≤ 8192`, `BATCH·N_HEAD < 2^16` (the grid's axis 2);
- `(BATCH·N_HEAD − 1)·N_CTX·HEAD_DIM < 2^31` (the kernel's `int32` offsets) and
  `N_HEAD·N_CTX·HEAD_DIM < 2^31` (`stride_z` an `int32` argument);
- `sm_scale`'s `float32` is finite, at most `2^16` in magnitude;
- the nine tensors contiguous, each storage holding its tensor; the outputs share no byte with the
  inputs or each other; the buffers valid in the initial memory. Any values.

**The launches covered** (`SETUP.md`): all 48 test launches (`setup_test_ok`), all 20 benchmark
launches (`setup_bench_ok`), `setup_test_large_ok`, `setup_bench_large_ok`, and `HEAD_DIM` 16, 32,
256 at the benchmark's largest shape (`setup_headdims_ok`); `canon_ok` covers the host's launch
from numeric facts.

**Device facts** the tolerances rest on (in `Device P`): `Device.DotF16` (`tl.dot` of `float16` into
`float32`: within `relDot16 K·L + absDot16 K`, `relDot16 K = (K + 2)·2^-22`,
`absDot16 K = (K + 2)·2^-126`, `L` counting operands by `Device.liftMag`), `Device.Exp2F32`
(`tl.math.exp2` on `float32`: within `2^-22·2^x + 2^-126`), and the totality of both.

## Tolerances

- Per pair, `eSub` bounds a computed `tl.dot(a, b) − x` (the dot's error, then the fusable
  subtraction: `3·2^-24` of the operands plus `4·2^-150`); `p16Err` adds `tl.math.exp2`'s error at
  the largest exponent the error allows and one `float16` rounding; `ds16Err` the same for `dP − Δ`,
  the product with `P` and the rounding of `dS`; `0` on masked pairs.
- `T` accumulation steps of `acc += tl.dot(a, b)` err by `accB T B = ((1 + ρ)^T − 1)·B +
  T·(1 + ρ)^T·α`, with `ρ = relDot16 32 + 3·2^-24·(1 + relDot16 32) ≈ 8.3·10^-6` and
  `α = absDot16 32·(1 + 3·2^-24) + 4·2^-150` (the dot over `32` products plus one `float32`
  addition), `B` the sum of the bounds on the operands' lifted magnitudes, `T` the program's steps for
  that row (`stepsKV`, `stepsQ`).
- For `dk` and `dq`, the product by `s` or `LN2` scales that, plus one `float32` rounding.

On `check_gpu.py`'s launches the kernel's error reaches about `0.73` of `T`. An output that cancels
to near `0` keeps the floor of the `float16` roundings of `P` and `dS`, so there the tolerance can
exceed the output.

## Exclusions

- **`BATCH > 8192`**: real shapes only.
- **`(BATCH·N_HEAD − 1)·N_CTX·HEAD_DIM ≥ 2^31`**: a kernel limit (`adj` and `bhid * N_CTX`, lines
  384–385, wrap in `int32`); the first host launch that hits it has 1025 heads at `16384 × 128`.
- **`N_HEAD·N_CTX·HEAD_DIM ≥ 2^31`**, **`N_CTX > 2^14`**, **`BATCH·N_HEAD ≥ 2^16`**, non-contiguous
  layouts, an `int` `sm_scale` or one above `2^16`: outside the file's launches and real use.
- **Values outside `HeadV.Ok`**: only that a `float16` value is stored, and the frame.
- **Other dtypes** (`float8e5`): the file never runs the backward pass in fp8.

`check_gpu.py` (one H200) checks `OutSpec` and the frame on the file's launches.
