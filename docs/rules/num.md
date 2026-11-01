# Rules `Num.*`: facts about the platform

Lean: `TVA.Rules.Num` (`lean/TVA/Rules/Num.lean`), the definitions; `TVA.Rules.Num.Sm90`
(`lean/TVA/Rules/Num/Sm90.lean`), the facts the library states for sm_90 and Triton 3.8.0;
`TVA.Rules.Num.Dot8Chunk` (`lean/TVA/Rules/Num/Dot8Chunk.lean`), the fp8 `tl.dot` chunk by chunk;
`TVA.Lemmas.Sm90` and `TVA.Lemmas.Dot8Chunk`, a platform satisfying them all.

The Triton documentation does not define these operations exactly:
- `tl.exp` and the other approximate math functions;
- `/` on floats (`tl.fdiv`, not IEEE: `div.full.f32` on fp32);
- `tl.dot` and `tl.dot_scaled`;
- conversions into fp8;
- float `atomic_add`.

The library gives them as relations in the platform `P`, and a claim quantifies over `P`. A
bundle that needs a property of one of them assumes it as a field of `Assumptions`, citing a
fact the library states for the operation by its name ("Facts the library states", below). The
constants are part of the fact, so the field never restates them:

```lean
  /-- [rule] `Num.DivTotal`: `/` on fp32 always has a result. -/
  div_total : Rules.Num.DivTotal P Format.binary32
  /-- [rule] `Num.DivFull`: `/` on fp32 is `div.full.f32`. -/
  div_full : Rules.Num.DivFull P
  /-- [rule] `Num.Exp2F32`: `tl.math.exp2` on fp32. -/
  exp2 : Rules.Num.Exp2F32 P
  /-- [rule] `Num.Exp2F32NegInf`: `tl.math.exp2(-inf) = +0` on fp32. -/
  exp2_ninf : Rules.Num.Exp2F32NegInf P
  /-- [rule] `Num.DotF16`: `tl.dot` of fp16 operands into fp32. -/
  dot : Rules.Num.DotF16 P
```

A proof may weaken a fact it cites (`MathAcc.mono`, `DotAcc.mono`, `DotScaledAcc.mono`,
`DotChunkAcc.mono`, `CvtRn.cvtAcc`, `CvtSat.cvtRn`, `DivRn.divAcc`); the field itself is the fact as
the library states it.

The definitions:

| Definition | Says | Typical use |
|---|---|---|
| `MathTotal P fn f` | `P.math fn f` has a result for every input | `safe_math` |
| `DivTotal P f`, `CvtTotal P f g rnd`, `DotTotal P cfg`, `AtomicAddTotal P f` | the same for the other operations | `safe_div`, `safe_cvt8`, `safe_dot`, atomics |
| `MathAcc P fn f dom rel abs` | finite inputs in `dom` and in `fn.natDom`, with room in the format (`(1+rel)·\|ref x\| + abs ≤ maxVal`), give finite results within `rel·\|ref x\| + abs` of the real function `fn.ref` | accuracy of softmax, GELU, … |
| `MathAt P fn f x y` | at the input `x`, the result is exactly `y` | `exp(-∞) = +0` in a masked softmax |
| `DivAcc P f qmax rel abs` | finite `x / y` (`y ≠ 0`, `\|x/y\| ≤ qmax`) is finite and within `rel·\|x/y\| + abs` | normalisation |
| `DivFull P` | `DivAcc P binary32 2^126 2^-22 2^-148`: fp32 `/` is `div.full.f32`, within `2^-22·\|x/y\| + 2^-148` for `\|x/y\| ≤ 2^126` | softmax, layer norm, any fp32 `/` |
| `DivRn P f` | `/` on `f` is IEEE division up to NaN payload (`DivRn.divAcc`: one rounding's bound, `u·\|x/y\| + η`) | fp64 `/` |
| `CvtAcc P f g` | with `.rn`, a finite value in `g`'s range converts to a finite value within `\|x\|·u + η` (one rounding's bound) | fp8 quantisation |
| `CvtRn P f g` | with `.rn`, a finite value in `g`'s range converts to its rounding to nearest even (`FloatV.cvtWith .rn`), up to NaN payload; `CvtRn.cvtAcc` gives `CvtAcc` | fp8 quantisation, bit for bit |
| `CvtSat P f g` | with `.rn`, every input converts to its saturating rounding to nearest even (`FloatV.cvtSat .rn`), up to NaN payload: in range as `CvtRn` (`CvtSat.cvtRn`), a finite value beyond `g`'s range and `±∞` to `±maxVal`, NaN to a NaN (`FloatV.cvtSat_rn_err`: one rounding of the clamped value; `cvtSat_of_gt`, `cvtSat_of_isInf`) | fp8 quantisation of unbounded values (accumulators) |
| `AtomicAddAcc P f rel abs` | a float `atomic_add` with an in-range exact sum is finite and within `rel·\|o+v\| + abs` | split-K accumulation |
| `DotAcc P cfg dom rel abs` | per output `(i, n)`: if row `i` of `a` and column `n` of `b` have values (`DType.val?`: finite floats, integers) in `dom` and the accumulator has one, the result has a value within `rel K·(\|acc\| + Σ\|a·b\|) + abs K` of `acc + Σ a·b`, when that leaves room below the accumulator's `DType.bound` | GEMM accuracy; with `rel = abs = 0`, exact int8 GEMM |
| `DotChunkAcc P cfg ρ σ α` | per output `(i, n)`, with values and room (`relDot8`, `absDot8` on `\|acc\| + Σ\|a·b\|`): the result has a value, and there are partial values from the accumulator's to the result's, one step per chunk of 32 consecutive `k` in increasing order, each step within `ρ·Mj + σ·\|s j + c_j\| + α` of the previous value plus the chunk's exact sum `c_j`, for every `Mj` bounding the previous value and the chunk's lifted products (`liftMag`) | fp8 GEMM and attention: a bound from the chunks' largest terms (`Dot8Chunk.DotChunkAcc.err`), or on the whole dot from lifted magnitudes (`Dot8Chunk.DotF8Chunk.liftAcc`) |
| `DotScaledTotal P cfg` | `P.dotScaled cfg` has a result for all operands | `safe_dot_scaled` |
| `DotScaledAcc P cfg dom sdom rel abs` | per output `(i, n)`: if row `i` of `lhs` and column `n` of `rhs` have finite scaled values in `dom`, scale bits in `sdom`, and the accumulator is finite, the result is finite and within `rel K·(\|acc\| + Σ\|a·b\|) + abs K` of `acc + Σ a·b` | MXFP4/MXFP8 matmuls |

The weakening lemmas (`TVA.Rules.Num`, section `Mono`): `MathAcc.mono` (a smaller domain and a
pointwise larger bound), `DotAcc.mono` and `DotScaledAcc.mono` (smaller domains, larger
constants at every `K`), and `DotChunkAcc.mono` (larger constants; `TVA.Rules.Num.Dot8Chunk`). They
derive weaker statements and state nothing new about the device.

## What a kernel reviewer checks

A kernel reviewer trusts this page ([REVIEW_PROTOCOL.md](../REVIEW_PROTOCOL.md) §1–§2): whether
a fact is true of the device, how Triton lowers the operation, and whether the constants are
right are settled in library review, below. For each `[rule]` field that cites a `Num.*`
definition, the kernel reviewer checks that it is **the library's fact for the operation the
model uses**:
- **The operation.** The field names the fact this table gives for the model's operation, format
  and configuration:

  | The model's operation | Facts it may cite |
  |---|---|
  | any of these operations, for having a result | `MathTotal P fn f`, `DivTotal P f`, `CvtTotal P f g rnd`, `DotTotal P cfg`, `AtomicAddTotal P f`, `DotScaledTotal P cfg` |
  | `tl.exp` on `float32` | `ExpF32 P`; at `-∞`, `ExpF32NegInf P` |
  | `tl.math.exp2` on `float32` | `Exp2F32 P`; at `-∞`, `Exp2F32NegInf P` |
  | `tl.log`, `tl.math.log2` on `float32` | `LogF32 P`, `Log2F32 P` |
  | `tl.sqrt`, `tl.math.rsqrt` on `float32` | `SqrtF32 P`, `RsqrtF32 P` |
  | `/` on `float32` (also `fp16`/`bf16` `/`, which the model converts to `float32`) | `DivFull P` |
  | `/` on `float64` | `DivRn P .binary64` |
  | `tl.cvt8 x .rn` (`x.to(tl.float8e4nv)`, `x.to(tl.float8e5)`, default rounding) from `float32`, `float16`, `bfloat16` | in range: `CvtF32E4m3 P`, `CvtF32E5m2 P`, `CvtF16E4m3 P`, `CvtF16E5m2 P`, `CvtBF16E4m3 P`, `CvtBF16E5m2 P`; every input (saturation): `CvtSatF32E4m3 P`, `CvtSatF32E5m2 P`, `CvtSatF16E4m3 P`, `CvtSatF16E5m2 P`, `CvtSatBF16E4m3 P`, `CvtSatBF16E5m2 P` |
  | `tl.dot` of `float16` or `bfloat16` operands into `float32`, `maxImpreciseAcc = 0` (the default), any `precision` | `DotF16 P`, `DotBF16 P` |
  | `tl.dot` of fp8 operands into `float32`, `maxImpreciseAcc = 2^30` (the default on sm_90), any `precision` | `DotE4m3Chunk P`, `DotE5m2Chunk P`, `DotE4m3E5m2Chunk P`, `DotE5m2E4m3Chunk P` |
  | `tl.dot` of `float32` operands, `precision = .ieee`, into `float32` | `DotF32Ieee P` |
  | `tl.dot_scaled` with `e8m0` scales, `group = 32`, `K`-packed operands: `"e4m3"` by `"e4m3"`; `"e4m3"` by `"e2m1"` | `DotScaledE4m3 P`, `DotScaledE4m3E2m1 P` |
  | `tl.atomic_add` (pointer or descriptor) on `float32` | `AtomicAddF32 P` |

- **The format and configuration** are the model's: the same `Format`, rounding mode, `DotCfg`
  or `ScaledCfg` as the operation in the model.
- **No constants of the kernel's own.** A field that states a `Num.*` definition with constants
  the kernel chose (`dom`, `rel`, `abs`, `qmax`, `x`, `y`, …), instead of naming a fact of the
  table, is not a library fact (`RULE_MISCITED`). A kernel that needs a fact the table lacks asks
  the library for it and waits (protocol §2).

A doubt about the facts themselves is a `LIBRARY_REPORT`.

## Facts the library states (sm_90, Triton 3.8.0)

All on an NVIDIA H200 NVL (sm_90), measured with Triton 3.8.0 on 2026-10-28 (the chunk facts
`DotE4m3Chunk` … on 2026-10-31); each evidence folder has its question, method, script and output.

| Fact (`TVA.Rules.Num`) | Operation | Says | What Triton runs | Evidence |
|---|---|---|---|---|
| `MathTotal`, `DivTotal`, `CvtTotal`, `DotTotal`, `AtomicAddTotal`, `DotScaledTotal`, for every function, format and configuration | every operation here | some result for every input | the device's instruction returns a value; a combination Triton cannot lower is rejected on the host, with no outcome ([compiler-faithful.md](compiler-faithful.md)) | — |
| `ExpF32` | `tl.exp`, `float32` | `MathAcc` on `[-X, X]` for every `X ≥ 0`: `rel = 2^-22 + 2^-23·X`, `abs = 2^-140` | `mul.f32` by `log2 e`, `ex2.approx.f32` | [num-math-f32](../../evidence/num-math-f32/README.md) |
| `Exp2F32` | `tl.math.exp2`, `float32` | `MathAcc` on every input: `2^-22`, `2^-126` | libdevice: `ex2.approx.ftz.f32` | num-math-f32 |
| `LogF32`, `Log2F32` | `tl.log`, `tl.math.log2`, `float32` | `MathAcc` on every input (`x > 0`): `2^-22`, `2^-149` | libdevice polynomial, `fma.rn.ftz.f32` | num-math-f32 |
| `SqrtF32` | `tl.sqrt`, `float32` | `MathAcc` on `sqrtDom` (`0` and `x ≥ 2^-126`): `2^-22`, `2^-149` | `sqrt.approx.ftz.f32` | num-math-f32 |
| `RsqrtF32` | `tl.math.rsqrt`, `float32` | `MathAcc` on `x ≥ 2^-126`: `2^-22`, `2^-149` | `rsqrt.approx.ftz.f32` | num-math-f32 |
| `ExpF32NegInf`, `Exp2F32NegInf` | `tl.exp(-∞)`, `tl.math.exp2(-∞)`, `float32` | `MathAt`: `+0` (`zeroF32`) at `negInfF32` | as above | num-math-f32 |
| `DivFull` | `/`, `float32` | `DivAcc` with `qmax = 2^126`: `2^-22`, `2^-148` | `div.full.f32` | [num-div-f32](../../evidence/num-div-f32/README.md) |
| `DivRn P .binary64` | `/`, `float64` | IEEE (`FloatV.divRn`), up to NaN payload | `div.rn.f64` | [num-div-f64](../../evidence/num-div-f64/README.md) |
| `CvtF32E4m3`, `CvtF32E5m2`, `CvtF16E4m3`, `CvtF16E5m2`, `CvtBF16E4m3`, `CvtBF16E5m2` | `.to(fp8)`, `"rtne"` | `CvtRn`: round to nearest even for finite `\|x\| ≤ maxVal` (448, 57344) | `cvt.rn.satfinite.{e4m3,e5m2}x2.{f32,f16x2}`; `bfloat16` widened to `float32` first | [num-cvt8](../../evidence/num-cvt8/README.md) |
| `CvtSatF32E4m3`, `CvtSatF32E5m2`, `CvtSatF16E4m3`, `CvtSatF16E5m2`, `CvtSatBF16E4m3`, `CvtSatBF16E5m2` | `.to(fp8)`, `"rtne"`, every input | `CvtSat`: `FloatV.cvtSat .rn x` up to NaN payload (beyond `maxVal` and `±∞`: `±maxVal`) | as above (`.satfinite`) | [float-layer](../../evidence/float-layer/README.md) (part A), num-cvt8 |
| `DotF16`, `DotBF16` | `tl.dot` into `float32` | `DotAcc` for all operand values: `relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126` | `mma.sync`/`wgmma` with `f32` accumulation | [num-dot](../../evidence/num-dot/README.md) |
| `DotE5m2Chunk` | `tl.dot` of `float8e5` by `float8e5` into `float32`, `maxImpreciseAcc = 2^30`, any `precision` | `DotChunkAcc` for all operand values: chunks of 32 consecutive `k` in increasing order, each step within `2^-7·Mj + 2^-12·\|s j + c_j\| + 2^-126`, `Mj` bounding the previous value and the chunk's products with a subnormal operand counted as `2^-14` | one `wgmma` (`k32`) per chunk: the accumulator and 32 products aligned to the largest exponent field, 13 bits kept below it (truncated toward zero), the sum cut to 14 bits; or `mma.sync` after an upcast to `float16`, two per chunk | [num-dot-chunk](../../evidence/num-dot-chunk/README.md) |
| `DotE4m3Chunk` | `tl.dot` of `float8e4nv` by `float8e4nv`, as `DotE5m2Chunk` | as `DotE5m2Chunk`, a subnormal operand counted as `2^-6` | as `DotE5m2Chunk` | num-dot-chunk |
| `DotE4m3E5m2Chunk` | `tl.dot` of `float8e4nv` by `float8e5`, as `DotE5m2Chunk` | as `DotE5m2Chunk`, subnormals counted as `2^-6` (`a`) and `2^-14` (`b`) | as `DotE5m2Chunk` | num-dot-chunk |
| `DotE5m2E4m3Chunk` | `tl.dot` of `float8e5` by `float8e4nv`, as `DotE5m2Chunk` | as `DotE5m2Chunk`, subnormals counted as `2^-14` (`a`) and `2^-6` (`b`) | as `DotE5m2Chunk` | num-dot-chunk |
| `DotF32Ieee` | `tl.dot`, `float32` operands, `"ieee"` | `DotAcc` for all operand values: `(K+1)·2^-23`, `(K+1)·2^-149` | `K` sequential `fma.rn.f32` | num-dot |
| `DotScaledE4m3`, `DotScaledE4m3E2m1` | `tl.dot_scaled`, `e8m0` scales, groups of 32, `K`-packed, either `fast_math` | `DotScaledAcc` on `scaledDomBF16` (`0` or a normal `bfloat16` magnitude), scale bits `e8m0Dom` (`1 … 254`): `relDot16`, `absDot16` | Triton's emulation: upcast to `bfloat16`, `mul.bf16` by the scale, `bfloat16` `wgmma` into `float32` | [num-dot-scaled](../../evidence/num-dot-scaled/README.md) |
| `AtomicAddF32` | `tl.atomic_add` (every `sem` and `scope`) and `desc.atomic_add`, `float32` | `AtomicAddAcc`: `2^-23`, `2^-124` | `atom.global.{cta,gpu,sys}.{relaxed,acquire,release,acq_rel}.add.f32` (`.v4`); `cp.reduce.async.bulk.tensor ….add` | [num-atomic-add](../../evidence/num-atomic-add/README.md) |

Every fact chooses its constants with margin over what the evidence measured (each README gives
the largest error as a fraction of the bound), except `Exp2F32`, whose `abs` is the flush
threshold: the flushed results approach the bound from below (`1.000` of it to three digits) and
cannot reach it. The conversion facts (`Cvt…`, `CvtSat…`) and `DivRn P .binary64` are exact (bit for
bit, up to NaN payload); `DivFull` is a bound with margin, like the others (no error above 0.651 of
it, [num-div-f32](../../evidence/num-div-f32/README.md)).

## The approximate functions

(`ExpF32` … `RsqrtF32`, `ExpF32NegInf`, `Exp2F32NegInf`; evidence `evidence/num-math-f32`, every
`float32` input.)
- **`tl.exp`** is not libdevice: Triton multiplies by `log2 e` with `mul.f32` and calls
  `ex2.approx.f32`, without `.ftz` (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:634-658`). The
  rounding of the product moves the exponent by up to `2^-24·|x·log2 e|`, so the relative error
  grows with `|x|` (to `2^-18` near the overflow). The fact is a family: for each `X ≥ 0`, on
  `[-X, X]`, `rel = 2^-22 + 2^-23·X`. A kernel instantiates `X` with a bound on its inputs.
- **`tl.math.exp2`** is libdevice's `__nv_exp2f`. With Triton's default `enable_reflect_ftz` it
  is `ex2.approx.ftz.f32`, which flushes results below `2^-126` to `0`; hence `abs = 2^-126`.
- **`tl.log`, `tl.math.log2`**: libdevice's polynomials (with `fma.rn.ftz`; a subnormal input is
  scaled by `2^23` first, so it is accurate too).
- **`tl.sqrt`, `tl.math.rsqrt`**: `sqrt.approx.ftz.f32`, `rsqrt.approx.ftz.f32`. A subnormal input
  is flushed (`sqrt` gives `0`, `rsqrt` gives `+∞`), so the domains exclude subnormals.
- **At `-∞`** both `tl.exp` and `tl.math.exp2` give `+0` (`0x00000000`). That is what a masked
  softmax or an online softmax with a `-inf` running maximum needs.

`tl.sqrt_rn` and `tl.div_rn` are IEEE in the library and need no fact ("Exact operations").

## Conversions into fp8

(`CvtF32E4m3` … `CvtBF16E5m2`, `CvtSatF32E4m3` … `CvtSatBF16E5m2`; evidence `evidence/num-cvt8`
and `evidence/float-layer` part A, every input of each source format.) On sm_89 and later Triton
converts with `cvt.rn.satfinite.e4m3x2` / `e5m2x2`, from `.f32` or `.f16x2`; `bfloat16` is widened
exactly to `float32` first (`primitives/triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:190-246,
414-444`). Every finite input with `|x| ≤ maxVal` gives its rounding to nearest even, bit for bit,
zeros keeping their sign: `CvtRn`, which is exact (`CvtRn.cvtAcc` gives `CvtAcc`, one rounding's
bound `|x|·u + η`).
Values beyond `maxVal` and `±∞` saturate to `±maxVal` (`.satfinite`), and NaN gives the NaN
`0x7f`: on every input the result is the library's saturating rounding `FloatV.cvtSat .rn x`
(`Format.cvt … .rn true`), bit for bit up to NaN payload, which the `CvtSat…` facts state
(library issue L64). A kernel that converts an unbounded value (an accumulator) into fp8 cites
the `CvtSat…` fact of the pair; `FloatV.cvtSat_rn_err` bounds the result against
`max (−maxVal) (min x maxVal)`, and `CvtSat.cvtRn` gives the `CvtRn` fact of the same pair.

These facts are about conversions at run time: `.to(fp8)` of a tensor, a stored value and an
`other=` converted to an fp8 pointer's dtype (a Python number there is first a `float32` tensor;
only a constant zero's conversion is folded, keeping its sign). A Python number built directly in
an fp8 dtype (`tl.full(s, c, d)`, a value of `tl.where` beside an fp8 tensor) is not one: Triton
truncates its `float32` constant with `arith.truncf`, which MLIR folds at compile time only when
the conversion is exact, and rejects the kernel on the host otherwise, so the constant is the exact
conversion, `FloatV.litF8`, with no saturation: `float('-inf')` in `float8e5` is `−∞` (`0xfc`),
where the saturating conversion of the same `float32` gives `−57344` (`0xfb`)
([SEMANTICS.md](../SEMANTICS.md), "Constants"; [evidence/fp8-constants](../../evidence/fp8-constants/README.md);
library issue L64). No fact is needed for it, and none of these facts may be cited for it.

## `tl.dot` (the chunk model)

(`DotF16`, `DotBF16`, `DotF32Ieee`; evidence `evidence/num-dot`; the fp8 facts are in the next
section.) On sm_90 Triton picks `wgmma` when the block has `M ≥ 64` and `num_warps ≥ 4`, and
`mma.sync` otherwise; each
instruction adds one chunk of products along `K` (16 for 16-bit operands, 32 for fp8) into the
accumulator. The evidence measures, for one output element, which bits of the chunk survive
(Part 1), and then checks the facts on adversarial inputs at every `K` that compiles, on both
paths (Part 2). The facts hold for every `K`; their constants follow from a per-chunk bound:
- **16-bit operands.** Each instruction keeps every term's bits down to at least `2^-23` of the
  largest term of the chunk, the accumulator included, truncating below (Part 1). A chunk with
  `j` nonzero terms is therefore within `j·2^-23` times their magnitudes' sum, plus a final
  rounding; over the chunks that is about `(K + ⌈K/16⌉)·2^-23·(|acc| + Σ|a·b|)`, below
  `relDot16 K = (K+2)·2^-22`. The measured error reaches `0.13` of the bound. The `abs` term covers products below `float32`'s normal range
  (`bfloat16` has `float32`'s exponent range). `input_precision` makes no difference for these
  operands (Part 3, bit for bit), so the facts hold for every `precision`.
- **fp8 operands, `maxImpreciseAcc = 2^30`.** Triton's default on sm_90 accumulates the whole
  `K` in the tensor core's reduced-precision accumulator. On `wgmma` each instruction keeps only
  `13` bits below the largest term, the accumulator included (Part 1: `acc = 1 + 2^-23` comes
  back as `1`). On `mma.sync` Triton upcasts fp8 to `float16`. Until round 24 the library derived
  from this the whole-dot facts `DotE4m3`, `DotE5m2`, `DotE4m3E5m2`, `DotE5m2E4m3`: a chunk within
  `33·2^-13 < 2^-7` times its magnitudes' sum, compounding to `relDot8 K = (1 + 2^-7)^⌈K/32⌉ − 1`
  of `|acc| + Σ|a·b|`. **A subnormal operand breaks this derivation**
  ([evidence/num-dot-chunk](../../evidence/num-dot-chunk/README.md), Parts A and F, 2026-10-31). The
  window is set by exponent fields, and a subnormal fp8 operand's field reads as the smallest
  normal's: the product `2^-9 · 2^8` of `float8e4nv` values sets the unit `2^-11`, though it is
  `2^-1`, and 31 products just below that unit are lost whole. On `wgmma` the error then reaches
  `3.65` times `relDot8 32·(|acc| + Σ|a·b|) + absDot8 32` (`1.79` times at `K = 64`; `1.73` times for
  `float8e5` at `K = 32`, whose subnormal `2^-16` reads as `2^-14`; the mixed pairs likewise). The
  four facts were false of the device on such inputs, which `randn` cast to fp8 contains, and are
  withdrawn (library issue L125, round 24); num-dot's Part 2 had no input where a subnormal operand
  sets the window. The fp8 facts are the chunk facts (next section), which count a subnormal
  operand at the smallest normal (`liftMag`) and hold on these inputs; a bound on a whole dot
  follows from them by a lemma.
- **`float32` operands, `"ieee"`.** No tensor cores: `K` sequential `fma.rn.f32` from the
  accumulator (Part 4, bit for bit against an exact sequential `fma`). The classical bound for
  `K` roundings is `γ_K = K·2^-24/(1 − K·2^-24) ≈ K·2^-24` (times `|acc| + Σ|a·b|`); the fact's
  `(K+1)·2^-23` is about twice it at every `K` that compiles, with `(K+1)·2^-149` for underflow.
  The measured error reaches `0.499` of the bound on an accumulator with all significand bits set
  and products just below its ulp: the first `fma` rounds up to a power of two and each later
  product is lost, an error of almost `(K−1)·2^-24·(|acc| + Σ|a·b|)`, so the ratio is
  `(K−1)/(2(K+1))` (evidence/num-dot, "The ratio of `DotF32Ieee`").

## `tl.dot` of fp8, chunk by chunk

(`DotE4m3Chunk`, `DotE5m2Chunk`, `DotE4m3E5m2Chunk`, `DotE5m2E4m3Chunk`; Lean
`TVA.Rules.Num.Dot8Chunk`, lemmas `TVA.Lemmas.Dot8Chunk`; evidence
[evidence/num-dot-chunk](../../evidence/num-dot-chunk/README.md); library issue L124, on the user's
request of 2026-10-31, "please tighten the fp8 kernel error bound"; the whole-dot facts `DotE4m3`, …
were withdrawn in the same round, library issue L125.) These are the library's fp8 `tl.dot` facts.
They state its accumulation, chunk by chunk, so that a kernel can bound a dot by the chunks' largest
terms; a bound on the whole dot follows by a lemma.

**What they say.** For an fp8 × fp8 `tl.dot` into `float32` with `maxImpreciseAcc = 2^30` (Triton's
default on sm_90) and any `precision`, per output element `(i, n)` whose row of `a` and column of `b`
have values `va`, `vb` and whose accumulator has the value `vc`, with room
(`mag + relDot8 K·mag + absDot8 K` at most the `float32` maximum, `mag = |vc| + Σ|va·vb|`,
`relDot8 K = (1 + 2^-7)^⌈K/32⌉ − 1`, `absDot8 K = ⌈K/32⌉·(1 + 2^-7)^⌈K/32⌉·2^-126`): the result
has a value `vr`, and there are partial values `s 0 = vc, s 1, …, s n = vr`, `n = ⌈K/32⌉`
(`chunks8 K`), such that for each chunk `j < n` and every `Mj` at least `|s j|` and at least
`liftMag a_k · liftMag b_k` for each `k` of the chunk,

`|s (j+1) − (s j + c_j)| ≤ 2^-7·Mj + 2^-12·|s j + c_j| + 2^-126`,

where chunk `j` is the `k` with `k / 32 = j` (`chunk8 K j`, 32 consecutive `k`), the chunks are taken in
increasing order of `j`, and `c_j` is the chunk's exact sum `Σ va k·vb k` (`chunkSum8`).
`liftMag d q` is `0` for `q = 0` and `max |q| 2^emin` otherwise, with `2^emin` the smallest normal
magnitude (`minNormal8`: `2^-14` for `float8e5`, `2^-6` for `float8e4nv`); for a normal value it is
`|q|`. So `Mj` may be the larger of `|s j|` and the chunk's largest product, a subnormal operand counted
at the smallest normal (the tensor core aligns it so, below). The partial values are the accumulator
between the instructions on the device; a kernel does not know them, and the lemmas bound them.

**How a kernel uses them** (`TVA.Lemmas.Dot8Chunk`). `Dot8Chunk.DotF8Chunk.err` (and
`DotChunkAcc.err`): for any bounds `m j` on each chunk's lifted products (`chunkLift8`, the largest of
them, will do), the result is within `errB` of `vc + Σ va·vb`, a bound computed from the exact inputs
alone: with the exact partial sums `S j = vc + Σ_{i<j} c_i` (`psum`),
`errB (j+1) = (1 + ρ + σ)·errB j + ρ·max |S j| (m j) + σ·|S (j+1)| + α`, `errB 0 = 0`
(`ρ = 2^-7`, `σ = 2^-12`, `α = 2^-126`), and `Dot8Chunk.errB_le` gives the closed form
`errB n ≤ (1 + ρ + σ)^n · Σ_{j<n} (ρ·max |S j| (m j) + σ·|S (j+1)| + α)`. For `qk = tl.dot(q, k)` from
a zero accumulator this bounds a score by its chunks' largest terms and partial sums. For
`acc = tl.dot(p, v, acc)` each chunk's error is relative to the larger of `|acc|` and the chunk's
largest product, plus `2^-12` of the new value: an accumulator fed back once per block still
compounds by up to about `1 + 2^-7 + 2^-12` per chunk, as the device's window allows (below), not
by slack in the fact. `Dot8Chunk.chain_err` composes any chain of such steps.

**A bound on the whole dot** (library issue L125), where the withdrawn facts gave one:
`Dot8Chunk.DotF8Chunk.liftAcc`, a theorem of the chunk facts (no device fact of its own), gives
`Dot8Chunk.LiftAcc`: per output element, with room on `L = |vc| + Σ liftMag a_k · liftMag b_k`
(`L + relLift8 K·L + absLift8 K` at most the `float32` maximum), the result has a value within
`relLift8 K·L + absLift8 K` of `vc + Σ va·vb`, with `relLift8 K = (1 + 2^-7 + 2^-12)^⌈K/32⌉ − 1` and
`absLift8 K = ⌈K/32⌉·(1 + 2^-7 + 2^-12)^⌈K/32⌉·2^-126`. Each chunk is taken with
`Mj = |s j| + Σ_chunk liftMag a_k · liftMag b_k`, and `DotLoop.accum_step` composes them
(`DotChunkAcc.liftErr`, for any constants). For zero and normal operands `L = |vc| + Σ|a·b|`
(`liftMag_zero`, `liftMag_of_le`); a subnormal operand counts at the smallest normal, and
`liftMag_le` (`liftMag d q ≤ |q| + minNormal8 d`) bounds it. A `K` loop with `tl.dot(a, b, acc)` per
step (tutorial 03) composes these per-step bounds as it would `DotAcc`'s.

**What Triton runs, and the window** (evidence Parts A, A2, B, E).
- **`wgmma`** (blocks with `M ≥ 64` and at least 4 warps): one `wgmma.mma_async….m64nNk32.f32` per
  chunk of 32, in increasing `k`, each from the accumulator the previous one left (Part E: every
  input whose outcome depends on the order agrees with increasing `k`). Each instruction, per output
  element, computes as follows (Part A; Parts B and D: the device equals this, bit for bit, on every
  element tested, about `2·10^8` results of every kind at every `K` that compiles, chains included).
  The terms are the accumulator `s` and the 32 exact products. Each nonzero term has an exponent
  field: `max(⌊log₂|s|⌋, −126)` for the accumulator, and `e(a) + e(b)` for a product, with
  `e(x) = max(⌊log₂|x|⌋, emin)` (a subnormal operand reads as the smallest normal). With `E` the
  largest field and `u = 2^(E−13)`, every term is truncated toward zero to a multiple of `u`, the
  truncated terms are summed exactly, and the sum is truncated toward zero to 14 significant bits.
- **`mma.sync`** (smaller blocks): Triton converts each fp8 operand to `float16` (exact) and runs two
  `mma.sync…m16n8k16.f32.f16.f16.f32` per chunk of 32; the two instructions' `k` interleave within
  the chunk (Part E, chunks of 16), and the chunks of 32 go in increasing order. Each instruction
  keeps every term's bits down to at least `2^-23` of its largest term and rounds into `float32`
  ([evidence/num-dot](../../evidence/num-dot/README.md) Part 1; Part A2), and an accumulator passes
  unchanged, subnormals included (Part A2).

**The constants.** On `wgmma` each term loses less than `u`, and no more than its own magnitude,
so the truncated sum is within `33u` of `s j + c_j`; the cut to 14 bits loses less than `2^-13` of
the truncated sum, which is within `|s j + c_j| + 33u`. Since `2^e(x) ≤ liftMag x` and
`2^max(⌊log₂|s|⌋, −126) ≤ max(|s|, 2^-126)`, `u ≤ 2^-13·Mj + 2^-139` (the `2^-139` only when a
subnormal accumulator sets `E`). So a chunk's error is below
`33·(1 + 2^-13)·2^-13·Mj + 2^-13·|s j + c_j| + 34·2^-139`. The facts keep `2^-7 = 64·2^-13` (a factor
`1.94` over `33·(1 + 2^-13)·2^-13`), `2^-12`
(a factor 2) and `2^-126`. On `mma.sync` a chunk of two instructions is within about
`360·2^-23·Mj ≈ 2^-14.5·Mj` (each within `17·2^-23` of its largest term plus `2^-23` of its result,
the second's accumulator below `17·Mj`), and no absolute term arises (products of fp8 values are at
least `2^-32`, and accumulators pass unchanged). The evidence (Part C) checks every element against the
interval the facts imply for it (the per-chunk bound composed over the chunks with the exact partial
sums and the worst case of the unknown partial values): none is outside, and the largest error is
`0.468` of the bound's half-width on `wgmma` (`float8e4nv`, a subnormal operand setting the window,
31 products just below `u` and an accumulator just below `u`), `0.0002` on `mma.sync`; each chunk of
the model's chain is within `0.468` of the per-chunk bound.

**What they do not state.** Nothing for a `maxImpreciseAcc` other than `2^30`, for an accumulator
other than `float32`, for infinite or NaN operands or accumulators, or beyond the room. `K` is in
practice a power of two of at least 32 (Triton rejects an fp8 `tl.dot` with `K = 16` on the host), so
every chunk is full; the statement covers partial chunks too, which no launch has.

## `tl.dot_scaled` on sm_90

(`DotScaledE4m3`, `DotScaledE4m3E2m1`; evidence `evidence/num-dot-scaled`.) sm_90 has no
microscaling instructions, so Triton emulates `tl.dot_scaled`
(`triton-3.8.0-DecomposeScaledBlocked.cpp:191-228`): each element is converted to `bfloat16`
(exact for `e4m3` and `e2m1`), its `e8m0` scale is decoded to `bfloat16` by a bit shift and
multiplied in with `mul.bf16`, and the products go to a `bfloat16` `tl.dot` into `float32`. The
facts therefore have `DotBF16`'s constants, on the domain where the multiplication by the scale
is exact: each scaled value `0` or a normal `bfloat16` magnitude (`scaledDomBF16`), each scale's
bits in `1 … 254` (`e8m0Dom`). The emulation reads the scale `0x00` as `0` (not `2^-127`) and
`0xFF` as NaN, so both are outside `e8m0Dom`. The evidence checks both `fast_math` values.

## Float `atomic_add`

(`AtomicAddF32`; evidence `evidence/num-atomic-add`.)
- A pointer `tl.atomic_add` on `float32` is `atom.global.<scope>.<sem>.add.f32` (Triton packs
  four lanes into `.v4.f32`); a descriptor's `atomic_add` is the TMA reduction
  `cp.reduce.async.bulk.tensor ….add`. `P.atomicAdd` covers both, so the fact holds for both, and
  for every `sem` and `scope` of the pointer atomic (all 12 gave bit-identical results:
  `evidence/num-atomic-add`, `sem_scope.py`; library issue L101).
- `atom.add.f32` rounds to nearest and flushes subnormal inputs and results to zero. Two
  operands just below `2^-126` sum to almost `2^-125` and give `0`, so `abs ≥ 2^-125`; with
  rounding, the tight bound is `2^-24·|o + v| + 2^-125`. The fact keeps a factor `2`:
  `rel = 2^-23`, `abs = 2^-124`.
- A scalar `tl.atomic_add(p, 0)` (or `±0.0`) with `relaxed` or `acquire` order is lowered to a
  strong load at the atomic's scope and order, with no store (`ld.global.gpu.relaxed.b32` in
  evidence/num-atomic-add; `triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:648-687, 762-806`), so
  memory keeps `o`. The fact admits `n = o` when `v = ±0`, as every `P.atomicAdd` fact must.
- `TVA.Lemmas.SharedAcc` turns the fact into a bound on a cell that several programs update in
  any order, against the sum of every program's operands (`accStep_atomicAdd`, `Paired`,
  `replay_err_shared`).

## What the library does not state

A kernel that needs one of these asks the library first (protocol §2):
- `tl.dot` with `float32` operands and `tf32`, `tf32x3`, `bf16x3` or `bf16x6` precision; any
  `tl.dot` into `float16`; an fp8 `tl.dot` with a `max_num_imprecise_acc` other than the default;
  integer `tl.dot` (exact, but not stated: `dotAcc_int` shows it is satisfiable);
- the fp8 conversions with `"rtz"` (only `float32` to `float8e5` compiles);
- `tl.sin`, `tl.cos`, `tl.erf`, other libdevice functions, and any approximate function on
  `float64` (on `float16` and `bfloat16` Triton rejects them on the host: `_check_dtype` allows
  only `fp32` and `fp64`, `primitives/triton-3.8.0-math.py:8-30, 95-232`);
- `tl.dot_scaled` with any other pair of formats: `"e2m1"` by `"e2m1"` (MXFP4, which compiles
  and runs on sm_90 at a small tile), `"e2m1"` by `"e4m3"`, and any pair with `e5m2`, `e2m3`,
  `e3m2`, `bf16` or `fp16` operands; other scale formats or groups, non-`K`-packed operands,
  scaled values outside `scaledDomBF16`; the native path of later GPUs;
- `atomic_add` on `float16`, `bfloat16` or `float64`;
- anything on a GPU other than sm_90.

## Device questions (library review)

These are for library review and the library author, who settle them with Triton 3.8.0's sources
(`primitives/`) and TVA's evidence on the target (`evidence/`). Kernel reviewers do not check
them. A fact for an operation must take into account:
- `tl.exp` on fp32 lowers to `ex2.approx` after a multiplication by `log2 e`, so its error is
  relative and grows with `|x|` (`ExpF32` is a family over `X`);
- `/` on fp32 is `div.full.f32`, so its fact is `DivFull`, not a tighter `DivAcc` ("Division",
  below);
- `/` on fp64 is `div.rn.f64` (IEEE), so `DivRn P .binary64` holds. `DivRn P .binary32` does
  not hold on NVIDIA;
- `atom.add.f32` flushes subnormals, so an `AtomicAddAcc` fact for fp32 needs `abs ≥ 2^-125`
  (`2^(emin+1)` in general);
- `P.atomicAdd` covers both `atom.add` and the TMA reduction of `desc.atomic_add`
  (`cp.reduce.async.bulk`); an `AtomicAddAcc` fact must hold for every instruction a kernel may
  use;
- every `P.atomicAdd` fact must admit `n = o` when `v = ±0` (the load above), even where an
  exact sum or a flush would give another value;
- `tl.dot_scaled` runs on native microscaling instructions (sm_100+) or on an emulation that
  upcasts to `bf16` or `fp16` (`core.py:2433-2438`). `DotScaledAcc` must hold for the path the
  target takes, and `fast_math` may skip NaN handling of scales. The emulation reads an `e8m0`
  scale `0x00` as `0` rather than `2^-127`, and with `fp16` compute it rounds scales to `fp16`
  (`triton-3.8.0-DecomposeScaledBlocked.cpp:58-96`). It multiplies each element by its scale in
  `bf16`/`fp16`, so one element can overflow to `inf`, and another underflow to `0` (giving
  `inf·0 = NaN`), while the exact sum is small. A fact for that path must choose `dom` to keep
  each scaled value in the compute type's range, and `sdom` to exclude scale `0x00`. `rel` and
  `abs` depend on `K`, since `float32` accumulation error grows with it;
- `P.cvt` is judged per source format, rounding mode and architecture. Before sm_90,
  `float32 → fp8` goes through a truncation to `float16` and stays within `CvtAcc`'s bound, but is
  not always the rounding to nearest. On sm_80, `bfloat16 → float8e5` clamps below `2^-15` to `0`, which exceeds
  the bound for tiny inputs. `CvtAcc` and `CvtRn` speak only of `.rn`;
- **`tl.dot` (`DotAcc`)**, judged for one `DotCfg` and target:
  - the input precision (`tf32` keeps 10 fraction bits of fp32 operands; `tf32x3`, `bf16x3` and
    `bf16x6` split them into several products);
  - `maxImpreciseAcc`: an fp8 dot on sm_90 sums that many products in the tensor core's
    reduced-precision accumulator before adding them into fp32;
  - the summation order and truncation the tensor cores use, which `rel K` must cover for every
    order;
  - the alignment: on sm_90's fp8 `wgmma` the window is set by exponent fields, so a subnormal
    operand counts as the smallest normal ([evidence/num-dot-chunk](../../evidence/num-dot-chunk/README.md)
    Part A), and a bound relative to `|a·b|` must cover that (the withdrawn `DotE4m3`, … did not,
    Part F; library issue L125).

  The evidence is a test at every `K` that compiles together with a per-chunk derivation that
  extends it to every `K` ("`tl.dot`", above).

## Satisfiability

A bundle proves its assumptions satisfiable together (`Claim.Consistent`, checked by the gate),
and every fact of the library must be satisfiable together with the others.

**The sm_90 facts.** `Platform.sm90ref` (`lean/TVA/Lemmas/Sm90.lean`) is `Platform.ref` with
one rounding's bound (`u·|y| + η`) as every function's bound and `exp(-∞) = exp2(-∞) = +0` in `float32`; its
conversions to nearest saturate, as sm_90's into fp8 do. Every fact of `TVA.Rules.Num.Sm90` holds
of it (`Platform.sm90ref.expF32`, `.exp2F32`, …, `.cvtF32E4m3`, …, `.cvtSatF32E4m3`, …,
`.dotF16`, …, `.dotScaledE4m3`, `.atomicAddF32`), and so do `DivFull`, `DivRn` and every
`…Total` rule, and the chunk facts of `TVA.Rules.Num.Dot8Chunk` (`Platform.sm90ref.dotE4m3Chunk`,
…, `.dotE5m2E4m3Chunk`, in `lean/TVA/Lemmas/Dot8Chunk.lean`). A bundle that cites any of these facts
proves `Claim.Consistent` with this one platform.

**In general.** `Platform.ref rel abs sp` (`lean/TVA/Lemmas/Consistency.lean`) is the most
accurate platform. Approximate functions return a float within the requested bound whenever one
exists, and `tl.dot`/`tl.dot_scaled` round the exact result once. It satisfies all of these
together, with `u = 2^-(m+1)` and `η = 2^(emin−m−1)` of the format:
- every `…Total` rule (`mathTotal`, `divTotal`, `cvtTotal`, `atomicAddTotal`, `dotTotal`,
  `dotScaledTotal`);
- `MathAcc` with `rel ≥ u`:
  - for `abs ≥ η` on any domain (`mathAcc_of_eta`);
  - for any `abs ≥ 0` where the real function's values are normal, `|ref x| ≥ 2^emin`
    (`mathAcc_of_normal`);
  - wherever the bundle proves that floats that accurate exist (`mathAcc`);
- `MathAt` at non-finite inputs, given by the table `sp` (`mathAt`);
- `DivRn`, `DivFull`;
- `CvtAcc`; `CvtSat` (`cvtSat`: its conversions to nearest saturate, `FloatV.cvtSat .rn`; toward
  zero they are `FloatV.cvtWith .rz`); `CvtRn` into a regular format (`cvtRn`, in
  `TVA.Lemmas.Sm90`);
- `AtomicAddAcc` for `rel ≥ u`, `abs ≥ η`;
- `DotAcc`:
  - with a float accumulator, for `rel K ≥ u` and `abs K ≥ η` when `K > 0` (`dotAcc`);
  - with integer operands and a signed accumulator, for any `rel`, `abs ≥ 0` (`dotAcc_int`).
    Triton 3.8.0's only integer `tl.dot` is `int8 × int8 → int32` (`semantic.py:1483-1486`), so
    no other integer case arises;
- `DotScaledAcc` for `rel K ≥ u` and `abs K ≥ η` of `float32` when `K > 0` (`dotScaledAcc`);
- `DotChunkAcc` with a float accumulator, for `σ ≥ u`, `α ≥ η` and any `ρ ≥ 0`
  (`Platform.ref.dotChunkAcc`, in `TVA.Lemmas.Dot8Chunk`: the partial values are the exact partial
  sums, and the last step is the one rounding).

At `K = 0` the bounds need only be nonnegative, because the result is the accumulator. So
constants that vanish there, such as `γ_K = K·u / (1 − K·u)`, are fine. The rounding argument is
`FloatV.exists_nearest`: the nearest float to a real is within one rounding's bound, `u·|y| + η`,
and `u·|y|` in the normal range (at least half an ulp of `y`, and up to about a whole one). A bound
of that form with smaller constants is false of any device that rounds.

## Division (`DivFull`, `DivRn`)

What Triton 3.8.0 compiles `/` to on NVIDIA (sources pinned in `primitives/`; settled as
[review-questions.md](../review-questions.md) S10):
- **Frontend.** `x / y` and `tl.fdiv(x, y)` (`ieee_rounding=False`) emit `arith.divf`
  (`triton-3.8.0-semantic.py:292-315, 331-341`). `fp16` and `bf16` operands are converted to
  `fp32` first (`semantic.py:89-102`), and the model writes that conversion. `tl.div_rn` emits
  `tt.precise_divf` (IEEE, `tl.div_rn` in the library).
- **MLIR.** `arith.divf` folds `x / 1.0` to `x`, folds two constants to the IEEE quotient, and
  rewrites `(-x) / (-y)` to `x / y` (`llvm-ArithOps.cpp:1451-1468`,
  `llvm-ArithCanonicalization.td:526-528`). Each gives a value within the card at the original
  operands.
- **Lowering.** `f32` becomes `llvm.nvvm.div.full` and `f64` becomes `llvm.nvvm.div.rn.d`
  (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:553-580`). LLVM selects the first as
  `div.full.f32`, without `.ftz` (`llvm-NVPTXIntrinsics.td:1536-1537`). It turns the second into
  an IEEE `fdiv` (`llvm-NVPTXTargetTransformInfo.cpp:298`). Two other selections exist but apply
  only to plain `fdiv` nodes, never to this intrinsic: `rcp.approx` for `1/x` and `div.approx`
  (`llvm-NVPTXInstrInfo.td:1189-1241`).
- **The instruction.** The PTX ISA: `div.full.f32` "implements a relatively fast, full-range
  approximation that scales operands to achieve better accuracy", with a maximum error of 2
  ulp. The card is not "2 ulp" but the relative bound `2^-22·|x/y| + 2^-148` for `|x/y| ≤ 2^126`:
  2 ulp of a normal quotient is at most `2^-22·|x/y|`, and the bound is up to twice that near the
  top of a binade.
- **TVA's measurement** ([evidence/num-div-f32](../../evidence/num-div-f32/README.md)): about
  `3.1·10^10` adversarial pairs (every divisor mantissa at seven divisor exponents against
  dividends placed in every binade of interest, every subnormal divisor and dividend, and `2^31`
  random pairs). None is outside the card; the largest error is `0.651` of it, and `2.11` ulp of
  the quotient's binade (above the ISA's 2 ulp, near the top of a binade, where the card allows
  almost 4).
- **`float64`** ([evidence/num-div-f64](../../evidence/num-div-f64/README.md)): `x / y` and
  `tl.fdiv(x, y)` on `float64` against the library's IEEE division (`FloatV.divRn`, the very
  definition, compiled to native code), on about `4.5·10^8` pairs: random patterns, subnormal
  dividends, quotients and divisors, divisors within `2^20` ulps of `±1` and in `[1/2, 2)`,
  powers of two, quotients near overflow, next to the exact value and next to midpoints, the
  special values, and constant divisors. Every result is equal, up to NaN payload:
  `DivRn P .binary64`.
- **Satisfiable.** `DivFull.of_divRn`: IEEE `float32` division meets the card.

What the rule does not promise:
- nothing for `|x/y| > 2^126`, a zero divisor, or an infinite or NaN operand (the result there is
  some value of `P.div`);
- not determinism: each element of `tl.div` is some value within the card at its operands. A
  constant quotient folded by MLIR (IEEE) and the same quotient computed by the unit may differ.

## Exact operations need no rule

`+`, `-`, `*`, `tl.fma`, `tl.div_rn`, `tl.sqrt_rn` and conversions between the standard float
formats are IEEE operations in the library, up to NaN payload, so they need no rule. Their meaning
is TVA's float layer (`lean/TVA/Math`, definitions in `TVA.Math.Defs`), and
[evidence/float-layer](../../evidence/float-layer/README.md) runs those definitions against the
device on every input pattern or pair where that is feasible (every `float16` and `bfloat16` pair
for `+ - *`, every `float32` input of the conversions and of `tl.sqrt_rn`) and on structured
samples elsewhere: no difference. Their error bounds are proved:
`FloatV.add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err` and `cvt_err` (`TVA.Lemmas.FloatErr`). Each gives
`|q| · u + η` for an in-range exact result `q`, with `u = 2^-(m+1)` and `η = 2^(emin−m−1)`.
`FloatV.sqrtRn_err` gives `tl.sqrt_rn` of a finite `x ≥ 0` the same bound, `√x · u + η` of the
real `√x` (in `ℝ`), and a finite, nonnegative result; it rests on `Format.sqrtMag_rn_spec`, which
shows that the layer's integer-square-root rounding (`Format.sqrtMag`) is within `√x·u + η` of
`√x`, the bound of one rounding to nearest (not the half-ulp bound itself), and does not overflow.
Reductions over them (`tl.sum`, `tl.sumc` with fusable additions, `tl.cumsumc`, the tree form of
`tl.sum_bcast_mul`) have proved bounds in `TVA.Lemmas.Sum` and `TVA.Lemmas.SumFused`.
