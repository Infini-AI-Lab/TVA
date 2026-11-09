# Claim: `moe_sum_reduce_bf16`

SGLang's `_moe_sum_reduce_kernel` (`source/moe_sum_reduce.py`, kernel
lines 50–104, host `moe_sum_reduce` 107–146). `Claim.lean`, namespace `Claims.MoeSumReduce`; the
claim to prove is `Statement`.

## Specification

`Post d init fin`, for every output element `(t, j)`, `t < token_num`, `j < hidden_dim`:
- `output[t, j]` holds a `bfloat16` value `c`;
- if the element is `InRange` — its `topk_num` inputs `x_k = input[t, k, j]` finite, the scale
  finite (the double `σ` and its `float32` constant `σ₃₂`), `A = Σ|x_k| ≤ 2^126` and
  `|σ₃₂|·A ≤ 2^126` — then `c` is finite and `|c − σ·S| ≤ tol K S A σ σ₃₂` with `S = Σ_k x_k`
  (exact, from the host's input) and `σ` the host's double;
- frame: every byte outside the output's `token_num·hidden_dim` elements is unchanged.

The condition is per element: a NaN, an infinity or a near-overflow input constrains only its own
elements (and their value still exists).

## Conditions

`Inst`: `T`, `K`, `D`, `BLOCK_M`, `BLOCK_DIM`, the scale's double pattern, `input` (with contents
`inVal`), `output`. The model passes the host's contiguous strides `(K·D, D, 1)`, `(D, 1)`.

`Inst.ok`: `BLOCK_M`, `BLOCK_DIM` powers of two with `BLOCK_M·BLOCK_DIM ≤ 2^20` (any, not only
`1 × 2048`); `1 ≤ T, K, D < 2^31`; `K ≤ 2^20`; `K·D < 2^31`; `cdiv(D, BLOCK_DIM) < 2^16`; buffer lengths;
`output` shares no byte with `input`. No bound on the scale.

`Pre`: buffers valid, `input` holds `inVal`. Every `bfloat16` input value is covered.

The launch: grid `(cdiv(T, BLOCK_M), cdiv(D, BLOCK_DIM), 1)`, the five block shapes of the source,
`debug = false`, `USE_GDC = False`, no `launch_pdl`. Part (A) of `Compiler.Faithful`'s scope:
`num_warps=16`, `num_stages=3` only. The claim relies on part (B).

## Assumptions

One `[rule]` field, `faithful`. The float operations are IEEE `+`, `*` and `bf16 ↔ f32`
conversions (exact operations, no device fact). No `[kernel]` field.

## Coverage

`setup_*` / `setup_*_ok`: the host's nine test launches (`(1,8,7168,2.5)` … `(8192,8,7168,2.5)`,
`(1,1,1,1.0)` included) with `BLOCK_M = 1`, `BLOCK_DIM = 2048` and the scale's pattern, and
`(65536, 9, 7168, 2.5)` for the Marlin/LoRA runners' large batches.

## Tolerances

With `u = 2^-24`, `η = 2^-150` (`Format.binary32`), `u_bf = 2^-8`, `η_bf = 2^-134`
(`Format.bfloat16`; `u32_eq` … `etabf_eq` prove the constants are the library's):
- `E1 = Tree.errBound u η K K A = ((1+u)^K − 1)·A + η·K·(1+u)^K`: the `float32` accumulation from
  `+0` is a chain of `K` IEEE additions (depth and node count `K`) over leaves of magnitude sum `A`
  (`FloatV.add_err` per node, the `Tree` bound), so `|acc − S| ≤ E1`, `|acc| ≤ |S| + E1`;
- `Em = u·|σ₃₂|·(|S| + E1) + η`: `FloatV.mul_err` for `acc·σ₃₂` (in range: `|σ₃₂|·(A + E1) < 2^127`);
- `Q = |σ₃₂|·|S| + |σ₃₂|·E1 + Em ≥ |p|`, and `FloatV.cvt_err` into `bfloat16`: `|c − p| ≤ u_bf·Q +
  η_bf` (in range: `Q < 2^127 ≤ maxVal`);
- `tol = (u_bf·Q + η_bf) + Em + |σ₃₂|·E1 + |σ − σ₃₂|·|S|`, by the triangle inequality; the last term
  is the scale's own `float32` conversion, written exactly so that the reference is the host's `σ`.

Meaningful: the dominant term is `2^-8·|σ·S|`, one `bfloat16` rounding of the result itself (not of
`|σ|·Σ|x|`); the `float32` terms are about `(K+1)·2^-24·|σ|·A`. Measured worst error / tol: 0.9961
(the bound is the worst case of one rounding to nearest, reached near the bottom of a binade). A
truncating store (error up to `2^-7` relative) violates it.

## Exclusions

- **`cdiv(hidden_dim, BLOCK_DIM) ≥ 2^16`** — a limit of the launch (CUDA's grid-y limit): the launch
  is rejected. First launch: `hidden_dim = 2^16·2048 + 1`. Real use (hidden ≤ 7168) does not reach it.
- **Elements not `InRange`** — a limit of the kernel's formats, stated per element: a non-finite
  input or scale, or `Σ|x|`, `|σ₃₂|·Σ|x|` above `2^126` (the `float32` sum or the `bfloat16` result
  can overflow near `2^128`; e.g. two inputs of `2^127.9`). A scale beyond `float32`'s range
  (`|σ| ≳ 3.4·10^38`) makes `σ₃₂` infinite and no element in range. Real use (activations, scale
  0.3–2.5) is far inside.
- **`topk_num > 2^20`** — outside real use (`topk_num ≤ 9`); kept so that `(1+2^-24)^K < 2` and the
  accumulation bound `E1` leaves room in `float32` (the library's sum bound applies; it fails
  from `K ≈ 1.16·10^7`).
- **`topk_num·hidden_dim ≥ 2^31`** — outside real use (`≤ 9 × 7168`); `input.stride(0)` would be an
  `int64` argument (another specialisation).
- **`token_num`, `topk_num` or `hidden_dim` `= 0` or `≥ 2^31`** — outside real use (`SETUP.md`:
  from 1; the `int` annotation takes values modulo `2^32`).
- `USE_GDC = True` / `launch_pdl` — out of scope: the claim covers launches without programmatic dependent launch (`launch_pdl = False`, so `USE_GDC = False`).

## Modelling decisions

- `tl.cast(x, dtype=tl.int64)` (lines 68–70) is `semantic.cast(x, tl.int64)`, the conversion
  `tl.convert .i32 .i64` (SEMANTICS.md, "`x.to(d)`, `tl.cast(x, d)` and `x.cast(d)`"). At the value
  `1` a stride is a compile-time constant; `tl.cast` first makes it a scalar tensor and gives the
  `int64` `1`, the model's value.
- `routed_scaling_factor` is a constexpr `float`: the constant `FloatV.litF32` of the double (`+0`
  for `±0`).
- `NUM_STAGE` (a `tl.range` scheduling keyword) changes no outcome; lines 85–87 not visited.

## Library requests

None.

## Checks

- Builds (`lake env lean ../claims/moe_sum_reduce_bf16/Claim.lean` from `lean/`) with no errors or warnings, no `sorry`/`axiom`; `consistent`, `satisfiable`, `setup_*_ok`
  use only the three standard axioms.
- `check_gpu.py` (H200): 25 cases (the host's 9; cancellation, near `2^126`, `bfloat16` subnormals,
  scales `1e-40`, `±0`, `−7.75`, `1e30`, `1e39`, NaN/±inf in some elements, top-1, top-64, hidden 1
  and 2049, `2^15` tokens; relaunched with `BLOCK` `(4,256)`, `(2,1024)`, `(8,64)`, `(1,4096)`):
  worst error / tol 0.9961 over every in-range element, all finite; out-of-range elements counted
  (e.g. 9218 in the `scale 3` near-overflow case); inputs unchanged (bitwise) and guards untouched.
