# Claim: `moe_sum_reduce_bf16`

SGLang's `_moe_sum_reduce_kernel` (`source/moe_sum_reduce.py`, kernel
lines 50–104, host `moe_sum_reduce` 107–146). `Claim.lean`, namespace `Claims.MoeSumReduce`; the
claim to prove is `Statement`.

## What you prove

**The goal.** In `Proof.lean`, `theorem Claims.MoeSumReduce.main : Statement` (`Statement`:
`Claim.lean` line 269). Unfolded (`Claim.Meets`), it says:

```lean
∀ K P, Assumptions K P →
  ∀ d, d.ok → ∀ init, Pre d init → ∀ o, K d init o → ∃ fin, o = .ok fin ∧ Post d init fin
```

for every device behaviour `K` and platform `P` that satisfy `Assumptions`, every covered instance `d`
(`Inst.ok`, line 133), and every initial memory `init` that `Pre` accepts (line 173), every outcome of
the launch is a normal end (no fault, no hang) whose final memory `fin` satisfies `Post` (line 248).

**What `Post` asks** (line 248), two parts:
1. for every `t < T` and `j < D`: element `t·D + j` of `output` reads in `fin` as some `bfloat16`
   `c`. This holds for every element. Only when `InRange d t j` (line 222: the `K` inputs
   `xv d t k j` (line 211) finite, `sigma d` and `sigma32 d` (lines 126, 130) finite,
   `absSum d t j ≤ 2^126` and `|σ₃₂|·absSum d t j ≤ 2^126`) it also asks that `c` is finite and
   `|c.toRat - (sigma d).toRat * refSum d t j| ≤ tol d.K (refSum d t j) (absSum d t j) (sigma d).toRat (sigma32 d).toRat`.
   `refSum` (line 214) is the exact sum `Σ_k input[t, k, j]` in the rationals from `init`'s values,
   `absSum` (line 217) the sum of magnitudes, `σ` the host's double; `tol` is line 239 (with
   `accErr`, line 229, and the constants `u32`, `eta32`, `ubf`, `etabf`, lines 199–203);
2. the frame: every byte outside `d.output.elems {e | e < d.T * d.D}` is unchanged.

**What you are given** (`Assumptions`, line 260): one field, `faithful` (`Compiler.Faithful`): the
device's outcomes `K d` are outcomes of the model's launch, when the launch is valid, has
`debug = false`, and the model's launch is well behaved from `init`. No numeric fact: the float
operations are IEEE `+`, `*` and conversions.

**The kernel, as the model runs it** (`moe_sum_reduce_kernel` line 47, `model` line 178, `launch`
line 188): a 2-D grid of `cdiv(T, BLOCK_M) × cdiv(D, BLOCK_DIM)` programs. Program
`(pid0, pid1)` takes the token tile `pid0·BLOCK_M + [0, BLOCK_M)` and the column tile
`pid1·BLOCK_DIM + [0, BLOCK_DIM)`, masked by `T` and `D`. A `tl.forRange` loop over
`i ∈ [0, K)` (trip count `K`) loads the masked `[BLOCK_M, BLOCK_DIM]` tile of row `i` of `input`
(int64 offsets `t·(K·D) + i·D + j`), converts it to `float32` and adds it to a `float32`
accumulator that starts at `+0`. After the loop it multiplies by `sigma32 d`, rounds to
`bfloat16` and stores the masked tile of `output` (offsets `t·D + j`). No reduction, no dot, no
atomic. Programs write disjoint output tiles and only read `input`: the route is
`TVA.meets_independent`; the loop is `safe_forRange` with an invariant on the accumulator and
`Wt := fun _ => ∅` (the body writes nothing; `iterW_empty`) (PROVER_GUIDE §4).

**What the proof must establish.**
- each launch has `debug = false` and is `Launch.Valid`: the grid is 2-D, so `Launch.valid_grid1`
  does not apply; axis 0 below `2^31` from `T_lt`, axis 1 below `2^16` from `grid1_lt`, the total
  number of programs below `2^31` from `grid_size_lt`, and the five block shapes accepted (`bm_pow2`, `bd_pow2`,
  `block_le`);
- each program is `Safe`: the `int32` tile offsets do not wrap (`Block.fits_of_grid_pow2` per axis,
  from `T_lt`, `D_lt`), the strides `IntV.ofNat (K·D)`, `IntV.ofNat D` are their values (`stride_lt`,
  `D_lt`), the `int64` load offsets do not wrap; every enabled load lane is inside `input`
  (`input_len`) and every enabled store lane inside `output` (`output_len`), with distinct lanes
  (`safe_load_buf`, `safe_store_buf` or `safe_load`, `safe_store`); `safe_convert`, `safe_cvt`,
  `safe_add`, `safe_mul` for the other lines;
- the write footprints (the program's output elements `t·D + j`) are disjoint across programs, and
  no program reads what another writes (`disj`: `output` shares no byte with `input`);
- the value part of part 1: the bound comes from `FloatV.add_err` per loop step (a chain of `K`
  additions, `Tree.errBound`, `K_le` keeps `(1+u)^K < 2`), `FloatV.mul_err` for the scale and
  `FloatV.cvt_err` into `bfloat16`, under `InRange`; the input conversion is exact
  (`FloatV.toRat_cvt_of_wider`). `## Tolerances` gives the derivation;
- the bytes no program writes give part 2.

`D_dim` and `T_ctx` restrict the instances to real model shapes (`128 ∣ D ≤ 2^17`, `T ≤ 2^22`).
With a `BLOCK_DIM` that divides `128` every column tile is full; the host's `BLOCK_DIM = 2048` does
not divide every such `D` (`7168 = 3.5·2048`), so the masked last column tile remains.

**Already in `Claim.lean`** (do not prove again): `consistent`, `satisfiable`, `mk_ok` and the
`setup_*_ok` theorems; `u32_eq`, `eta32_eq`, `ubf_eq`, `etabf_eq` (lines 205–208) tie the
constants to `Format.binary32` and `Format.bfloat16`.

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
`1 × 2048`); `1 ≤ T, K, D < 2^31`; `K ≤ 2^20`; `K·D < 2^31`; `cdiv(D, BLOCK_DIM) < 2^16`;
`cdiv(T, BLOCK_M)·cdiv(D, BLOCK_DIM) < 2^31` (the number of programs); buffer lengths;
`output` shares no byte with `input`; real model shapes (Exclusions): `D % 128 = 0` and
`D ≤ 2^17` (`D_dim`: `hidden_dim` is a model dimension), `T ≤ 2^22` (`T_ctx`: `token_num` is a
context length). No bound on the scale.

`Pre`: buffers valid, `input` holds `inVal`. Every `bfloat16` input value is covered.

The launch: grid `(cdiv(T, BLOCK_M), cdiv(D, BLOCK_DIM), 1)`, the five block shapes of the source,
`debug = false`, `USE_GDC = False`, no `launch_pdl`. Part (A) of `Compiler.Faithful`'s scope:
`num_warps=16`, `num_stages=3` only. The claim relies on part (B).

## Assumptions

One `[rule]` field, `faithful`. The float operations are IEEE `+`, `*` and `bf16 ↔ f32`
conversions (exact operations, no device fact). No `[kernel]` field.

## Coverage

`setup_*` / `setup_*_ok`: the host's six test launches whose `hidden_dim` is a multiple of 128
(`(1,8,7168,2.5)`, `(7,9,4096,0.3)`, `(32,8,2048,1.0)`, `(33,6,5120,1.0)`, `(1024,9,4096,0.3)`,
`(8192,8,7168,2.5)`) with `BLOCK_M = 1`, `BLOCK_DIM = 2048` and the scale's pattern, and
`(65536, 9, 7168, 2.5)` for the Marlin/LoRA runners' large batches. The host's other three test
launches, `(17,8,2880,1.0)`, `(1,1,1,1.0)` and `(300,3,100,2.5)`, are outside `Inst.ok` (`hidden_dim`
not a multiple of 128; Exclusions). `satisfiable` uses `(1, 1, 128)` with the all-zero memory.

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

- **Out of scope**: dimensions that are not multiples of 128 or exceed 2^17, contexts longer than
  2^22 tokens, and batches of more than 8192 requests: the claim covers real model shapes only.
  Here: `hidden_dim` not a multiple of 128 or above `2^17` (`D_dim`; the host's tests with
  `hidden_dim` 2880, 1 and 100 among them), and `token_num > 2^22` (`T_ctx`). The claim has no
  batch-size field (`token_num` counts tokens).
- **`cdiv(hidden_dim, BLOCK_DIM) ≥ 2^16`** — a limit of the launch (CUDA's grid-y limit): the launch
  is rejected. First launch: `hidden_dim = 2^16·2048 + 1`. Real use (hidden ≤ 7168) does not reach it.
- **`cdiv(token_num, BLOCK_M)·cdiv(hidden_dim, BLOCK_DIM) ≥ 2^31`** — a kernel limit: a launch of
  `2^31` or more programs is not a valid CUDA launch. With the host's blocks the first one is
  `token_num = 2^30`, `hidden_dim = 2049` (`2^30 × 2` programs), an input of `2^30·2049·topk_num`
  `bfloat16` values (at least 4.4 TB): no device holds it, and real use (`2^16` tokens) is far below.
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
  the launches with `hidden_dim` not a multiple of 128 (three host tests and most edges) are outside
  `Inst.ok` (`D_dim`) and run as extras; worst error / tol 0.9961 over every in-range element, all
  finite; out-of-range elements counted
  (e.g. 9218 in the `scale 3` near-overflow case); inputs unchanged (bitwise) and guards untouched.
