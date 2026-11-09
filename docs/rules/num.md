# Rules `Num.*`: facts about the platform

Lean: `TVA.Rules.Num` (`lean/TVA/Rules/Num.lean`), the definitions; `TVA.Rules.Num.Sm90`
(`lean/TVA/Rules/Num/Sm90.lean`), the facts the library states for sm_90 and Triton 3.8.0;
`TVA.Rules.Num.Dot8Chunk` (`lean/TVA/Rules/Num/Dot8Chunk.lean`), the fp8 `tl.dot` chunk by chunk;
`TVA.Rules.Num.Dot16Chunk` (`lean/TVA/Rules/Num/Dot16Chunk.lean`), the 16-bit `tl.dot` chunk by chunk;
`TVA.Lemmas.Sm90`, `TVA.Lemmas.Dot8Chunk`, `TVA.Lemmas.Dot16Chunk` and `TVA.Lemmas.DotLift`, a platform satisfying them all
and the lemmas that put the `tl.dot` facts to use.

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

A proof may weaken a fact it cites (`MathAcc.mono`, `DotAcc.mono`, `DotLiftAcc.mono`,
`DotScaledAcc.mono`, `DotChunkAcc.mono`, `DotChunkLoss.mono`, `DotChunkLoss.chunkAcc`,
`DotF8ChunkLoss.chunk`, `CvtRn.cvtAcc`, `CvtSat.cvtRn`, `DivRn.divAcc`,
`DotLift.DotLiftAcc.dotAcc`); the field itself is the fact as the library states it.

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
| `DotAcc P cfg dom rel abs` | per output `(i, n)`: if row `i` of `a` and column `n` of `b` have values (`DType.val?`: finite floats, integers) in `dom` and the accumulator has one, the result has a value within `rel K·(\|acc\| + Σ\|a·b\|) + abs K` of `acc + Σ a·b`, when that leaves room below the accumulator's `DType.bound` | GEMM accuracy (`float32` operands); with `rel = abs = 0`, exact int8 GEMM |
| `DotLiftAcc P cfg rel abs` | per output `(i, n)`, for every operand value: with `L = \|acc\| + Σ liftMag a·liftMag b` and room (`L + rel K·L + abs K` at most the accumulator's `DType.bound`), the result has a value within `rel K·L + abs K` of `acc + Σ a·b`; `liftMag d q` is `0` for `q = 0` and `max \|q\| (minNormal8 d)` otherwise, a subnormal operand counted at the smallest normal (`2^-14` for `float16` and `float8e5`, `2^-126` for `bfloat16`, `2^-6` for `float8e4nv`) | 16-bit GEMM and attention (`DotF16`, `DotBF16`; one step of a `K` loop: `DotLift.dotAdd_step`); a whole fp8 dot from the chunk facts (`Dot8Chunk.DotF8Chunk.liftAcc`) |
| `DotChunkAcc P cfg ρ σ α` | per output `(i, n)`, with values and room (`relDot8`, `absDot8` on `\|acc\| + Σ\|a·b\|`): the result has a value, and there are partial values from the accumulator's to the result's, one step per chunk of 32 consecutive `k` in increasing order, each step within `ρ·Mj + σ·\|s j + c_j\| + α` of the previous value plus the chunk's exact sum `c_j`, for every `Mj` bounding the previous value and the chunk's lifted products (`liftMag`) | fp8 GEMM and attention: a bound from the chunks' largest terms (`Dot8Chunk.DotChunkAcc.err`), or on the whole dot from lifted magnitudes (`Dot8Chunk.DotF8Chunk.liftAcc`) |
| `DotChunkLoss P cfg ρ σ α κ ρ'` | as `DotChunkAcc` (the same values, room and partial values), each step within `min (ρ·Mj) (κ·A_j + ρ'·Mj) + σ·\|s j + c_j\| + α`, `A_j` the chunk's exact `Σ\|a·b\|` (`chunkAbs8`, not lifted); it gives `DotChunkAcc` with `ρ σ α` (`DotChunkLoss.chunkAcc`) | fp8 attention with the accumulator fed back through many chunks (`Dot8Chunk.DotChunkLoss.err`, `Dot8Chunk.chain_errM`) |
| `DotScaledTotal P cfg` | `P.dotScaled cfg` has a result for all operands | `safe_dot_scaled` |
| `DotScaledAcc P cfg dom sdom rel abs` | per output `(i, n)`: if row `i` of `lhs` and column `n` of `rhs` have finite scaled values in `dom`, scale bits in `sdom`, and the accumulator is finite, the result is finite and within `rel K·(\|acc\| + Σ\|a·b\|) + abs K` of `acc + Σ a·b` | MXFP4/MXFP8 matmuls |

The weakening lemmas (`TVA.Rules.Num`, section `Mono`): `MathAcc.mono` (a smaller domain and a
pointwise larger bound), `DotAcc.mono` and `DotScaledAcc.mono` (smaller domains, larger
constants at every `K`), `DotLiftAcc.mono` (larger constants), `DotAcc.liftAcc` (a `DotAcc` fact on
every value gives `DotLiftAcc` with the same constants, since `|a·b| ≤ liftMag a·liftMag b`),
`DotChunkAcc.mono` and `DotChunkLoss.mono` (larger constants), `DotChunkLoss.chunkAcc` (its first
arm) and `DotF8ChunkLoss.chunk` (`TVA.Rules.Num.Dot8Chunk`); and `DotLift.DotLiftAcc.dotAcc`
(`TVA.Lemmas.DotLift`: on zero and normal values, where `liftMag` is `|q|`, a `DotLiftAcc` fact is
`DotAcc`). They derive weaker statements and state nothing new about the device.

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
  | `tl.exp` on `float32` | `ExpF32 P`; at `-∞`, `ExpF32NegInf P`; from `1453635/16384` and at `+∞`, `ExpF32Overflow P` |
  | `tl.math.exp2` on `float32` | `Exp2F32 P`; at `-∞`, `Exp2F32NegInf P` |
  | `tl.log`, `tl.math.log2` on `float32` | `LogF32 P`, `Log2F32 P` |
  | `tl.sqrt`, `tl.math.rsqrt` on `float32` | `SqrtF32 P`, `RsqrtF32 P` |
  | `/` on `float32` (also `fp16`/`bf16` `/`, which the model converts to `float32`) | `DivFull P`; by an infinite divisor, `DivF32InfDivisor P` |
  | `/` on `float64` | `DivRn P .binary64` |
  | `tl.cvt8 x .rn` (`x.to(tl.float8e4nv)`, `x.to(tl.float8e5)`, default rounding) from `float32`, `float16`, `bfloat16` | in range: `CvtF32E4m3 P`, `CvtF32E5m2 P`, `CvtF16E4m3 P`, `CvtF16E5m2 P`, `CvtBF16E4m3 P`, `CvtBF16E5m2 P`; every input (saturation): `CvtSatF32E4m3 P`, `CvtSatF32E5m2 P`, `CvtSatF16E4m3 P`, `CvtSatF16E5m2 P`, `CvtSatBF16E4m3 P`, `CvtSatBF16E5m2 P` |
  | `tl.dot` of `float16` or `bfloat16` operands into `float32`, `maxImpreciseAcc = 0` (the default), any `precision` | `DotF16 P`, `DotBF16 P`; or, chunk by chunk, `DotF16Chunk P`, `DotBF16Chunk P` |
  | `tl.dot` of fp8 operands into `float32`, `maxImpreciseAcc = 2^30` (the default on sm_90), any `precision` | `DotE4m3Chunk P`, `DotE5m2Chunk P`, `DotE4m3E5m2Chunk P`, `DotE5m2E4m3Chunk P`; or, each step the smaller of two bounds, `DotE4m3ChunkLoss P`, `DotE5m2ChunkLoss P`, `DotE4m3E5m2ChunkLoss P`, `DotE5m2E4m3ChunkLoss P` |
  | `tl.dot` of `float32` operands, `precision = .ieee`, into `float32` | `DotF32Ieee P` |
  | `tl.dot` of `int8` operands (into `int32`; a `uint32` accumulator bitcast to `int32`), `maxImpreciseAcc = 0` (the default), any `precision` | `DotI8 P` |
  | `tl.dot_scaled` with `e8m0` scales, `group = 32`, `K`-packed operands: `"e4m3"` by `"e4m3"`; `"e4m3"` by `"e2m1"` | `DotScaledE4m3 P`, `DotScaledE4m3E2m1 P` |
  | `tl.atomic_add` (pointer or descriptor) on `float32` | `AtomicAddF32 P` |
  | the depth of a float `tl.sum`'s tree (`sum`, `sum1`, `sum0`, `sumc`, `sum1c`, `sum0c`) | `SumDepth P` |

- **The format and configuration** are the model's: the same `Format`, rounding mode, `DotCfg`
  or `ScaledCfg` as the operation in the model.
- **No constants of the kernel's own.** A field that states a `Num.*` definition with constants
  the kernel chose (`dom`, `rel`, `abs`, `qmax`, `x`, `y`, …), instead of naming a fact of the
  table, is not a library fact (`RULE_MISCITED`). A kernel that needs a fact the table lacks asks
  the library for it and waits (protocol §2).

A doubt about the facts themselves is a `LIBRARY_REPORT`.

## Facts the library states (sm_90, Triton 3.8.0)

All on an NVIDIA H200 NVL (sm_90), measured with Triton 3.8.0 on 2026-10-28 (the chunk facts
`DotE4m3Chunk` … on 2026-10-31, `DotF16`, `DotBF16` as restated in round 25 on 2026-11-01, and the
loss facts `DotE4m3ChunkLoss` … of round 26 on 2026-11-01, `DotI8` on 2026-11-08, and the round-32 facts
`DotF16Chunk`, `DotBF16Chunk`, `DivF32InfDivisor`, `SumDepth` on 2026-11-08, and the round-33 fact
`ExpF32Overflow` and the round-33 restatement of `DotF16Chunk`, `DotBF16Chunk` on 2026-11-08); each evidence
folder has its question, method, script and output.

| Fact (`TVA.Rules.Num`) | Operation | Says | What Triton runs | Evidence |
|---|---|---|---|---|
| `MathTotal`, `DivTotal`, `CvtTotal`, `DotTotal`, `AtomicAddTotal`, `DotScaledTotal`, for every function, format and configuration | every operation here | some result for every input | the device's instruction returns a value; a combination Triton cannot lower is rejected on the host, with no outcome ([compiler-faithful.md](compiler-faithful.md)) | — |
| `ExpF32` | `tl.exp`, `float32` | `MathAcc` on `[-X, X]` for every `X ≥ 0`: `rel = 2^-22 + 2^-23·X`, `abs = 2^-140` | `mul.f32` by `log2 e`, `ex2.approx.f32` | [num-math-f32](../../evidence/num-math-f32/README.md) |
| `Exp2F32` | `tl.math.exp2`, `float32` | `MathAcc` on every input: `2^-22`, `2^-126` | libdevice: `ex2.approx.ftz.f32` | num-math-f32 |
| `LogF32`, `Log2F32` | `tl.log`, `tl.math.log2`, `float32` | `MathAcc` on every input (`x > 0`): `2^-22`, `2^-149` | libdevice polynomial, `fma.rn.ftz.f32` | num-math-f32 |
| `SqrtF32` | `tl.sqrt`, `float32` | `MathAcc` on `sqrtDom` (`0` and `x ≥ 2^-126`): `2^-22`, `2^-149` | `sqrt.approx.ftz.f32` | num-math-f32 |
| `RsqrtF32` | `tl.math.rsqrt`, `float32` | `MathAcc` on `x ≥ 2^-126`: `2^-22`, `2^-149` | `rsqrt.approx.ftz.f32` | num-math-f32 |
| `ExpF32NegInf`, `Exp2F32NegInf` | `tl.exp(-∞)`, `tl.math.exp2(-∞)`, `float32` | `MathAt`: `+0` (`zeroF32`) at `negInfF32` | as above | num-math-f32 |
| `ExpF32Overflow` | `tl.exp`, `float32`, from `1453635/16384` (`88.72283935546875`, pattern `0x42b17218`) and at `+∞` | every result is `+∞` (`posInfF32`) | `mul.f32` by `log2 e`, `ex2.approx.f32` (`+∞` once the product reaches `128`) | [num-special-f32](../../evidence/num-special-f32/README.md) E1 |
| `DivFull` | `/`, `float32` | `DivAcc` with `qmax = 2^126`: `2^-22`, `2^-148` | `div.full.f32` | [num-div-f32](../../evidence/num-div-f32/README.md) |
| `DivF32InfDivisor` | `/`, `float32`, a finite dividend by `±∞` | the zero of sign `sign x xor sign y` (`1 / ∞ = +0`) | `div.full.f32` | [num-special-f32](../../evidence/num-special-f32/README.md) |
| `DivRn P .binary64` | `/`, `float64` | IEEE (`FloatV.divRn`), up to NaN payload | `div.rn.f64` | [num-div-f64](../../evidence/num-div-f64/README.md) |
| `CvtF32E4m3`, `CvtF32E5m2`, `CvtF16E4m3`, `CvtF16E5m2`, `CvtBF16E4m3`, `CvtBF16E5m2` | `.to(fp8)`, `"rtne"` | `CvtRn`: round to nearest even for finite `\|x\| ≤ maxVal` (448, 57344) | `cvt.rn.satfinite.{e4m3,e5m2}x2.{f32,f16x2}`; `bfloat16` widened to `float32` first | [num-cvt8](../../evidence/num-cvt8/README.md) |
| `CvtSatF32E4m3`, `CvtSatF32E5m2`, `CvtSatF16E4m3`, `CvtSatF16E5m2`, `CvtSatBF16E4m3`, `CvtSatBF16E5m2` | `.to(fp8)`, `"rtne"`, every input | `CvtSat`: `FloatV.cvtSat .rn x` up to NaN payload (beyond `maxVal` and `±∞`: `±maxVal`) | as above (`.satfinite`) | [float-layer](../../evidence/float-layer/README.md) (part A), num-cvt8 |
| `DotF16`, `DotBF16` | `tl.dot` of `float16` (`bfloat16`) operands into `float32`, `maxImpreciseAcc = 0`, any `precision` | `DotLiftAcc` for all operand values: `relDot16 K = (K+2)·2^-22`, `absDot16 K = (K+2)·2^-126`, relative to `\|acc\| + Σ liftMag a·liftMag b`, a subnormal operand counted as `2^-14` (`float16`) or `2^-126` (`bfloat16`) | one `mma.sync`/`wgmma` (`k16`) per chunk of 16 in increasing `k`: the accumulator and 16 products aligned to the largest exponent field (a subnormal operand read as the smallest normal), 25 bits kept below it (truncated toward zero), the sum truncated toward zero to `float32` | [num-dot-sub16-round25](../../evidence/num-dot-sub16-round25/README.md), num-dot |
| `DotF16Chunk`, `DotBF16Chunk` | `tl.dot` of `float16` (`bfloat16`) operands into `float32`, `maxImpreciseAcc = 0`, any `precision` | partial values `s 0 = acc, …, s ⌈K/64⌉ = result`, chunks of 64 consecutive `k` (all `K` when `K < 64`) in increasing order, each step within `min(2^-13·Mj, 2·A_j + 2^-22·Mj) + 2^-20·\|s j + c_j\| + 2^-143`, `Mj` bounding `\|s j\|` and the chunk's lifted products (a subnormal operand counted as `2^-14` or `2^-126`), `A_j` the chunk's `Σ\|a·b\|`; room `mag + relChunk16 K·mag + absChunk16 K` | at most four `mma.sync…m16n8k16` or `wgmma…m64nNk16` per chunk, each adding 16 of its `k` (16 consecutive on `wgmma`; on `mma.sync`, as the operands' `kWidth` of 2, 4 or 8 lays them out): the accumulator and 16 products aligned to the largest exponent field, 25 bits kept below it (truncated toward zero), the sum truncated toward zero to `float32` | [num-dot-chunk16-round33](../../evidence/num-dot-chunk16-round33/README.md), num-dot-chunk16, num-dot-sub16-round25 |
| `DotE5m2Chunk` | `tl.dot` of `float8e5` by `float8e5` into `float32`, `maxImpreciseAcc = 2^30`, any `precision` | `DotChunkAcc` for all operand values: chunks of 32 consecutive `k` in increasing order, each step within `2^-7·Mj + 2^-12·\|s j + c_j\| + 2^-126`, `Mj` bounding the previous value and the chunk's products with a subnormal operand counted as `2^-14` | one `wgmma` (`k32`) per chunk: the accumulator and 32 products aligned to the largest exponent field, 13 bits kept below it (truncated toward zero), the sum cut to 14 bits; or `mma.sync` after an upcast to `float16`, two per chunk | [num-dot-chunk](../../evidence/num-dot-chunk/README.md) |
| `DotE4m3Chunk` | `tl.dot` of `float8e4nv` by `float8e4nv`, as `DotE5m2Chunk` | as `DotE5m2Chunk`, a subnormal operand counted as `2^-6` | as `DotE5m2Chunk` | num-dot-chunk |
| `DotE4m3E5m2Chunk` | `tl.dot` of `float8e4nv` by `float8e5`, as `DotE5m2Chunk` | as `DotE5m2Chunk`, subnormals counted as `2^-6` (`a`) and `2^-14` (`b`) | as `DotE5m2Chunk` | num-dot-chunk |
| `DotE5m2E4m3Chunk` | `tl.dot` of `float8e5` by `float8e4nv`, as `DotE5m2Chunk` | as `DotE5m2Chunk`, subnormals counted as `2^-14` (`a`) and `2^-6` (`b`) | as `DotE5m2Chunk` | num-dot-chunk |
| `DotE5m2ChunkLoss` | as `DotE5m2Chunk` | `DotChunkLoss`: as `DotE5m2Chunk`, each step within `min(2^-7·Mj, 2·A_j + 2^-12·Mj) + 2^-12·\|s j + c_j\| + 2^-126`, `A_j` the chunk's `Σ\|a·b\|`; gives `DotE5m2Chunk` | as `DotE5m2Chunk`: a product truncated toward zero loses at most its own magnitude | [num-dot-chunk-round26](../../evidence/num-dot-chunk-round26/README.md), num-dot-chunk |
| `DotE4m3ChunkLoss` | as `DotE4m3Chunk` | as `DotE5m2ChunkLoss`, a subnormal operand counted as `2^-6`; gives `DotE4m3Chunk` | as `DotE5m2Chunk` | num-dot-chunk-round26, num-dot-chunk |
| `DotE4m3E5m2ChunkLoss` | as `DotE4m3E5m2Chunk` | as `DotE5m2ChunkLoss`, subnormals counted as `2^-6` (`a`) and `2^-14` (`b`); gives `DotE4m3E5m2Chunk` | as `DotE5m2Chunk` | num-dot-chunk-round26, num-dot-chunk |
| `DotE5m2E4m3ChunkLoss` | as `DotE5m2E4m3Chunk` | as `DotE5m2ChunkLoss`, subnormals counted as `2^-14` (`a`) and `2^-6` (`b`); gives `DotE5m2E4m3Chunk` | as `DotE5m2Chunk` | num-dot-chunk-round26, num-dot-chunk |
| `DotF32Ieee` | `tl.dot`, `float32` operands, `"ieee"` | `DotAcc` for all operand values: `(K+1)·2^-23`, `(K+1)·2^-149` | `K` sequential `fma.rn.f32` | num-dot |
| `DotI8` | `tl.dot` of `int8` operands into `int32`, with or without an accumulator (`int32`, or `uint32` read as `int32`: `tl.bitcast .u32 .i32`), `maxImpreciseAcc = 0`, any `precision` | `DotAcc` for all operand values with `rel = abs = 0`: exact, `acc + Σ a·b`, when `\|acc\| + Σ\|a·b\| ≤ 2^31 - 1` | `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32` or `wgmma.mma_async.sync.aligned.m64nNk32.s32.s8.s8` along `K`, integer sums (the first saturates at the `int32` range, the second wraps) | [num-dot-int8](../../evidence/num-dot-int8/README.md) |
| `DotScaledE4m3`, `DotScaledE4m3E2m1` | `tl.dot_scaled`, `e8m0` scales, groups of 32, `K`-packed, either `fast_math` | `DotScaledAcc` on `scaledDomBF16` (`0` or a normal `bfloat16` magnitude), scale bits `e8m0Dom` (`1 … 254`): `relDot16`, `absDot16` | Triton's emulation: upcast to `bfloat16`, `mul.bf16` by the scale, `bfloat16` `wgmma` into `float32` | [num-dot-scaled](../../evidence/num-dot-scaled/README.md) |
| `SumDepth` | `tl.sum`, `tl.sumc` and their axis forms, float, along an axis of `n` positions | each tree has depth at most `⌈log₂ n⌉` (`log₂ n` for a Triton axis): `P.sumDepth n ≤ Nat.clog 2 n`, so `SumDepth.depth_le` turns `t.depth ≤ tl.sumBound P n` into `t.depth ≤ Nat.clog 2 n`; `sum_err_depth`, `sumc_err_depth` give the error with that depth; not stated for a reduction the loop rewrite reassociates (exclusions.md) | `ReduceOpToLLVM.cpp`: a balanced binary tree over each thread's elements of the axis, a `shfl.sync.bfly` butterfly over its lanes, and the same butterfly over its warps after a conversion through shared memory | [reduce-tree](../../evidence/reduce-tree/README.md) |
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
- **Past the overflow** (`ExpF32Overflow`, round 33, library issue L154): `tl.exp` gives `+∞` on
  every `float32` input from `1453635/16384` (`88.72283935546875`, pattern `0x42b17218`) up, and at
  `+∞`; the largest input with a finite result is `0x42b17217` (`88.72283172607422`, result
  `0x7f7fff4f`) ([evidence/num-special-f32](../../evidence/num-special-f32/README.md) E1, every
  pattern from `+64` to `+∞`). `ExpF32` says nothing there: its room fails. With
  `DivF32InfDivisor`, `1 / (1 + tl.exp(x))` is `+0` there.

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

(`DotF16`, `DotBF16`, `DotF32Ieee`; evidence `evidence/num-dot`, whose "Part 1" … "Part 4" are cited
by name below, and, for the 16-bit facts, `evidence/num-dot-sub16-round25`; the fp8 facts are in the
next section.) On sm_90 Triton picks `wgmma` when the block has `M ≥ 64` and `num_warps ≥ 4`, and
`mma.sync` otherwise. Each instruction adds one chunk of products along `K` into the accumulator:
16 products for 16-bit operands on both paths; for fp8 operands, 32 on `wgmma`, and on `mma.sync`,
where Triton converts fp8 to `float16`, 16 per instruction, two instructions per chunk of 32 whose
`k` interleave (next section). The evidence measures, for one output element, which bits of the
chunk survive, and then checks the facts on adversarial inputs at every `K` that compiles, on both
paths. The facts hold for every `K`; their constants follow from a per-chunk bound:
- **16-bit operands** (`DotF16`, `DotBF16`, as restated in round 25, library issue L126;
  [evidence/num-dot-sub16-round25](../../evidence/num-dot-sub16-round25/README.md)).
  - **What each instruction computes** (that evidence's Part A; in Parts B and D the device equals
    it on every element tested, on both paths, at every `K` that compiles, chains included). The
    terms are the accumulator `s` and the 16 exact products of the chunk. Each nonzero term has an
    exponent field: `e(a) + e(b)` for a product, with `e(x) = max(⌊log₂|x|⌋, emin)` (`emin = −14` for
    `float16`, `−126` for `bfloat16`: a subnormal operand reads as the smallest normal), and
    `max(⌊log₂|s|⌋, −126)` for the accumulator. With `E` the largest field and `u = 2^(E−25)`, every
    term is truncated toward zero to a multiple of `u`, the truncated terms are summed exactly, and
    the sum is truncated toward zero to `float32` (24 significant bits, and the grid `2^-149` below
    `2^-126`: subnormal results are kept). Each instruction adds 16 of the `k`, one instruction
    after another: 16 consecutive `k` in increasing order (Part E) on `wgmma` and on `mma.sync` with
    operands loaded in 16 bits or wider; on `mma.sync` with an operand converted in registers from an
    8-bit load, other sets of 16 `k` within a block of 32 or 64 ("Chunk by chunk" below). The
    derivation of `DotF16` and `DotBF16` uses only that each instruction adds 16 products to the
    accumulator the previous one left, so they hold whichever 16.
  - **The constants.** Each of the at most 17 terms loses less than `u`, and no more than its own
    magnitude; the cut to `float32` loses less than `2^-23·|S'| + 2^-149` of the truncated sum `S'`,
    and `|S'| ≤ |s| + Σ_chunk |a·b|`. Since `2^e(x) ≤ liftMag x` and the accumulator's field is at
    most `max(|s|, 2^-126)`, `u ≤ 2^-25·Mj + 2^-151`, with `Mj` the larger of `|s|` and the chunk's
    largest `liftMag a·liftMag b`, so `Mj ≤ |s| + A_j`, `A_j` the chunk's sum of `liftMag a·liftMag b`.
    So a chunk's error is below `(17·2^-25 + 2^-23)·(|s| + A_j) + 17·2^-151 + 2^-149`, that is
    `5.25·2^-23·(|s| + A_j) + 5.25·2^-149`. Truncation toward zero never increases a magnitude, so
    the accumulator before chunk `j` is at most `|acc| + Σ_{i<j} Σ_chunk |a·b|`, and
    `|s| + A_j ≤ L = |acc| + Σ liftMag a·liftMag b`. The result's error is the sum of the chunks'
    errors, below `⌈K/16⌉·(5.25·2^-23·L + 5.25·2^-149)`. As `⌈K/16⌉·5.25 ≤ (K+15)·5.25/16 ≤ 2·(K+2)`
    for every `K ≥ 1`, that is below `relDot16 K·L + absDot16 K` (a factor `6.9` at `K = 16`, about
    `6.1` for large `K`). The measured error reaches `0.111` of the bound (the window's worst case
    with normal operands, `K = 16`), and each chunk of the model's chain is within `0.902` of the
    per-chunk bound. The `abs` term covers products below `float32`'s normal range (`bfloat16` has
    `float32`'s exponent range).
  - **Why the lift.** Until round 25 the facts were `DotAcc` relative to `|acc| + Σ|a·b|`, derived
    from "every term's bits down to `2^-23` of the largest term" (num-dot Part 1, whose inputs had
    normal operands only). A product with a subnormal operand has the field of the smallest normal
    times its partner, so with a large partner it sets the window while far smaller, and the other
    products lose their bits below `u`. The error then reached `53` times the old bound for
    `float16` (`a = 2^-23`, `b = 2^14`, 15 products `(2 − 2^-10)·2^-24`) and `13` times for
    `bfloat16`, on both paths, wherever the product sits
    ([evidence/num-dot-sub16](../../evidence/num-dot-sub16/README.md), the coordinator's), and
    `113.7` times in the new evidence (`float16`, `K = 16`); num-dot's Part 2 had no such input.
    Relative to lifted magnitudes the same inputs are at `0.104` of the bound. For zero and normal
    operands `L = |acc| + Σ|a·b|`, the old bound (`DotLift.DotLiftAcc.dotAcc`).
  - **Using them.** A kernel with `acc = tl.dot(a, b, acc)` per `K` step (tutorial 03) takes each
    step from `DotLift.dotAdd_step`, as it took it from `DotLoop.dotAdd_step`, with
    `Σ liftMag a·liftMag b` in place of `Σ|a·b|`; `DotLift.sum_liftMag_mul_le` bounds that by
    `Σ|a·b| + m·Σ|a| + m·Σ|b| + K·m²` (`m = minNormal8`, `2^-14` for `float16`). Subnormal operands
    stay covered: tutorial 06's `p.to(float16)` makes them, and its inputs, `randn` cast to
    `float16`, contain them.
  - `input_precision` makes no difference for these operands (num-dot Part 3, and the new
    evidence's Part G1 on subnormal operands, bit for bit), so the facts hold for every `precision`.
  - **Chunk by chunk** (`DotF16Chunk`, `DotBF16Chunk`, rounds 32 and 33, library issue L150; Lean
    `TVA.Rules.Num.Dot16Chunk`, lemmas `TVA.Lemmas.Dot16Chunk`;
    [evidence/num-dot-chunk16-round33](../../evidence/num-dot-chunk16-round33/README.md),
    [evidence/num-dot-chunk16](../../evidence/num-dot-chunk16/README.md)). Beside `DotF16` and
    `DotBF16`, which stay, these state the same instructions one chunk at a time, as the fp8 loss
    facts do (below): per output element, with room
    (`mag + relChunk16 K·mag + absChunk16 K` at most the `float32` maximum, `mag = |acc| + Σ|a·b|`,
    `relChunk16 K = (1 + 2^-13)^⌈K/64⌉ − 1`, `absChunk16 K = ⌈K/64⌉·(1 + 2^-13)^⌈K/64⌉·2^-143`), the
    result has a value, and there are partial values `s 0 = acc, …, s ⌈K/64⌉ = result`, one step per
    chunk of 64 consecutive `k` (`chunk16 K j`, the `k` with `k / 64 = j`; all `K` when `K < 64`) in
    increasing order, each within `min(2^-13·Mj, 2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| + 2^-143` of
    `s j + c_j`, for every `Mj` bounding `|s j|` and the chunk's lifted products, `c_j` the chunk's
    exact sum and `A_j` its exact `Σ|a·b|` (not lifted).
    - **Why chunks of 64.** Which 16 `k` an instruction adds depends on the operands' layout on
      `mma.sync`, whose `kWidth` Triton sets from the narrowest dtype the operands were loaded in
      (`primitives/triton-3.8.0-AccelerateMatmul.cpp:385-427, 561-568`; `kWidth = max(32 / bitwidth,
      1)`, `primitives/triton-3.8.0-TritonGPUAttrDefs.td:1478-1488`, with `bitwidth` at most the
      operands' own 16; the evidence's Part W): `2` for operands loaded in 16 bits or wider (16 consecutive `k`), `4`
      for an operand converted in registers from an 8-bit load (`float8`, `int8`, `uint8`, a
      `bool` load: the `k ≡ 0, 1 (mod 4)` of each block of 32, then the `k ≡ 2, 3`), `8` for an
      8-bit load unpacked into twice as many `k` by `tl.join` (the `k ≡ 2i, 2i+1 (mod 8)` of each
      block of 64, `i = 0, …, 3`; at `K = 32`, two sets within the 32). On `wgmma` they are 16
      consecutive `k`. In every case measured each instruction's `k` lie in one aligned block of
      64, the blocks are taken in increasing order and each block's instructions one after the
      other (Part P, 104 configurations, probed on the device). Until round 33 the facts were per
      chunk of 16 consecutive `k` (`2^-20`, `2^-22`, `2^-145`, `2^-24`), which a `K = 32` launch
      with an operand converted from `float8e5` refutes: its device result `0` is reached by no
      partial values of 16-chunks (Part X, round 32's library review).
    - **The constants**, from the per-instruction model above. In a chunk the `i`-th instruction
      (`i ≤ 4`) starts from an accumulator `|t| ≤ |s j| + Σ_{earlier} |p| ≤ (16i − 15)·Mj` (every
      truncation is toward zero), so its unit is `u ≤ 2^-25·max(|t|, Mj) + 2^-151`. In the first
      arm each instruction's 17 terms lose less than `u`, and each cut to `float32` less than
      `2^-23` of a sum exceeding `|s j + c_j|` by at most the products still to come: a chunk is
      within `17·(1 + 17 + 33 + 49)·2^-25·Mj + 2^-23·(48 + 32 + 16)·Mj + 2^-21·|s j + c_j| +
      21·2^-149` (up to terms `2^-21` smaller), that is `2084·2^-25·Mj + …`, kept as `2^-13` (a
      factor `1.97`), `2^-20` (2) and `2^-143` (3.05). In the second a product loses at most
      `min(|p|, u)` and the accumulator less than `u`, with `u ≤ 2^-25·(Mj + Σ_{earlier} |p|) +
      2^-151`, so a chunk is within `(1 + 2^-21 + 2^-23)·A_j + (1 + 2^-21)·2^-23·Mj +
      (1 + 2^-21)·2^-21·|s j + c_j| + 5·2^-149`, kept as `2·A_j` and `2^-22·Mj`. An accumulator
      that dominates its chunk is charged about `(2^-22 + 2^-20)·|acc|` per 64 `k`, where `DotF16`
      charges `relDot16 K·|acc|` per `tl.dot`.
    - **Using them.** A kernel bounds a dot from the exact inputs with
      `Dot16Chunk.Dot16BitChunk.err` (the result within `errM`, the recursion and closed forms of
      the fp8 lemmas, `errM_le_errB`, `errM_le_errL`), or on the whole dot with
      `Dot16Chunk.Dot16BitChunk.liftAcc` (`DotLiftAcc` with `relLift16`, `absLift16`).
    - **Evidence.** [evidence/num-dot-chunk16-round33](../../evidence/num-dot-chunk16-round33/README.md):
      the per-instruction model with the instructions' `k` equals the device on every element
      result (`2605056` of them, `kWidth` 2, 4 and 8 on `mma.sync`, `wgmma`, `float16` and
      `bfloat16`, `a` or `b` converted from 8 bits, `K = 16` to 256), and every step of 62976
      elements checked in exact rationals is within the bound (largest ratio `0.38`, an
      accumulator dominating its chunk). [evidence/num-dot-chunk16](../../evidence/num-dot-chunk16/README.md)
      (round 32) checked the model on about `1.2·10^8` element results with operands loaded in 16
      bits (16 consecutive `k`).
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

## The integer `tl.dot`

(`DotI8`; library issue L141, round 31; [evidence/num-dot-int8](../../evidence/num-dot-int8/README.md).)
Triton 3.8.0's only integer `tl.dot` is `int8 × int8`, into an `int32` result whatever `out_dtype` says, with
`K ≥ 32` (`semantic.py:1429-1520`, `nvidia-compiler.py:19-31`; another integer operand and a smaller `K` are
rejected on the host, that evidence's I5). The accumulator is `int32` or `uint32`; any other accumulator dtype is
rejected on the host, and a `uint32` one computes as its bits read as `int32`, so `DotI8` covers it with the
accumulator `tl.bitcast .u32 .i32 acc`, the room on that reading (round 32, library issue L141;
[evidence/num-dot-int8-round32](../../evidence/num-dot-int8-round32/README.md) A1–A3: 4096 of 4096 results on
`mma.sync` and 65536 of 65536 on `wgmma` at each of three `K`, the edge of the room included). On sm_90 it is a chain along
`K` of integer tensor-core instructions of 32 products each: `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32`
when the block has `M < 64` or `num_warps < 4`, and `wgmma.mma_async.sync.aligned.m64nNk32.s32.s8.s8` otherwise
(I6). The first clamps an `int32` result that leaves the range (`.satfinite`), the second wraps it (I4: a final
sum past `2^31 - 1` gave `2^31 - 1` and its value modulo `2^32`; a partial sum past it, brought back in range by
the later chunks, gave a clamped and an exact result). So the fact needs room: while
`|acc| + Σ|a·b| ≤ 2^31 - 1`, every partial sum, in any order and any grouping, has magnitude at most
`|acc| + Σ|a·b|`, neither path clamps or wraps, and the result is `acc + Σ a·b` exactly, for every `K`. That is
`DotAcc` with `rel = abs = 0` on all operand values, the room being `DotAcc`'s own condition with the
accumulator's `DType.bound` (`2^31 - 1`). The evidence found it exact on 2142192 outputs with room (14 tile
configurations, both paths, `M` and `N` from 1 to 128, three `K` up to 2048, four spellings with and without
an accumulator), on 236992 more under every `input_precision` the front end accepts (Triton ignores it for
these operands) and `allow_tf32=False`, and at the edge of the room (`K = 131040`, results `±(2^31 - 1)`).
`DotI8` is stated for `max_num_imprecise_acc = 0`, the value a source that does not give it has; a given value
(`32` gave the exact result too, I5) is not stated.

## `tl.dot` of fp8, chunk by chunk

(`DotE4m3Chunk`, `DotE5m2Chunk`, `DotE4m3E5m2Chunk`, `DotE5m2E4m3Chunk`, and since round 26 the loss
facts `DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`, `DotE4m3E5m2ChunkLoss`, `DotE5m2E4m3ChunkLoss`; Lean
`TVA.Rules.Num.Dot8Chunk`, lemmas `TVA.Lemmas.Dot8Chunk`; evidence
[evidence/num-dot-chunk](../../evidence/num-dot-chunk/README.md) and
[evidence/num-dot-chunk-round26](../../evidence/num-dot-chunk-round26/README.md); library issue L124,
on the user's request of 2026-10-31, "please tighten the fp8 kernel error bound", sharpened in round
26 after the round 24/25 library review; the whole-dot facts `DotE4m3`, … were withdrawn in round
24, library issue L125.) These are the library's fp8 `tl.dot` facts. They state its accumulation,
chunk by chunk, so that a kernel can bound a dot by the chunks' largest terms; a bound on the whole
dot follows by a lemma.

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
largest product, plus `2^-12` of the new value, so a bound composed from these steps grows by up to
about `1 + 2^-7 + 2^-12` per chunk when the accumulator is fed back once per block.
`Dot8Chunk.chain_err` composes any chain of such steps. That compounding comes from the facts'
form, each step relative to `Mj ≥ |s j|`, and from their margin (`2^-7` is about twice the window's
worst chunk, below), not from the device. The device truncates every term and the sum toward zero,
so its partial values never exceed `|acc| + Σ_{i<j} Σ_chunk |a·b|` in magnitude, and a product it
truncates loses at most its own magnitude: its error grows at most linearly in the chunks. The loss
facts, next, state a second bound for each step from this.

**Each chunk's error, the smaller of two bounds** (the loss facts; round 26, library issue L124).
`DotE4m3ChunkLoss`, `DotE5m2ChunkLoss`, `DotE4m3E5m2ChunkLoss`, `DotE5m2E4m3ChunkLoss` (`DotF8ChunkLoss`,
the shape `DotChunkLoss`) say what the facts above say, with the same values, room and partial
values, and each step within

`|s (j+1) − (s j + c_j)| ≤ min(2^-7·Mj, 2·A_j + 2^-12·Mj) + 2^-12·|s j + c_j| + 2^-126`,

where `A_j = Σ_chunk |va k·vb k|` is the chunk's exact magnitude (`chunkAbs8`, the products' own
magnitudes, not lifted) and `Mj` is as above (`kappaChunk8 = 2`, `rhoLossChunk8 = 2^-12`). The first
arm is the bound above, so each loss fact gives the chunk fact of its pair (`DotF8ChunkLoss.chunk`).
The second charges an accumulator that dominates its chunk about `2^-11` of itself per chunk, instead
of `2^-7 + 2^-12`. A kernel cites either fact; the loss facts are the stronger. From the exact inputs
(`Dot8Chunk.DotF8ChunkLoss.err`, `DotChunkLoss.err`) the result is within `errM` of `vc + Σ va·vb`,
with `errM 0 = 0` and
`errM (j+1) = (1 + σ)·errM j + min(ρ·(X j + errM j), κ·A_j + ρ'·(X j + errM j)) + σ·|S (j+1)| + α`,
`X j = max |S j| (m j)` (`ρ = 2^-7`, `κ = 2`, `ρ' = 2^-12`, `σ = 2^-12`, `α = 2^-126`);
`Dot8Chunk.chain_errM` composes any chain of such steps. `errM` is never above `errB`
(`errM_le_errB`), nor above `errL`, the second arm composed alone (`errM_le_errL`), which grows by
`1 + 2^-11` per chunk: `errL n ≤ (1 + 2^-11)^n · Σ_{j<n} (2·A_j + 2^-12·max |S j| (m j) +
2^-12·|S (j+1)| + 2^-126)` (`errL_le`). On tutorial 06's softmax chains at `T = 256` the new composed
bound is `0.29` to `0.78` of the round-24 one (evidence/num-dot-chunk-round26, Part C).

**A bound on the whole dot** (library issue L125), where the withdrawn facts gave one:
`Dot8Chunk.DotF8Chunk.liftAcc`, a theorem of the chunk facts (no device fact of its own), gives
`DotLiftAcc` (round 24's `Dot8Chunk.LiftAcc`, the same statement, moved to `TVA.Rules.Num` in round
25 for the 16-bit facts): per output element, with room on `L = |vc| + Σ liftMag a_k · liftMag b_k`
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
  is a `float16` one ("`tl.dot` (the chunk model)", above): it keeps every term's bits down to
  `2^-25` of its largest exponent field, a subnormal `float16` operand (a subnormal `float8e5`
  value) read as `2^-14`, and truncates the sum toward zero into `float32`
  ([evidence/num-dot-sub16-round25](../../evidence/num-dot-sub16-round25/README.md) Parts A and G2);
  an accumulator passes unchanged, subnormals included (Part A2).

**The constants.** On `wgmma` each term loses less than `u`, and no more than its own magnitude,
so the truncated sum is within `33u` of `s j + c_j`; the cut to 14 bits loses less than `2^-13` of
the truncated sum, which is within `|s j + c_j| + 33u`. Since `2^e(x) ≤ liftMag x` and
`2^max(⌊log₂|s|⌋, −126) ≤ max(|s|, 2^-126)`, `u ≤ 2^-13·Mj + 2^-139` (the `2^-139` only when a
subnormal accumulator sets `E`). So a chunk's error is below
`33·(1 + 2^-13)·2^-13·Mj + 2^-13·|s j + c_j| + 34·2^-139`. The facts keep `2^-7 = 64·2^-13` (a factor
`1.94` over `33·(1 + 2^-13)·2^-13`), `2^-12`
(a factor 2) and `2^-126`. On `mma.sync` a chunk of two instructions is within about
`360·2^-23·Mj ≈ 2^-14.5·Mj` (each within `17·2^-23` of its largest lifted term plus `2^-23` of its
result, the second's accumulator below `17·Mj`: a `float16` operand's field is at most its fp8
`liftMag`, the same `2^-14` for a subnormal `float8e5` value, and every `float8e4nv` value is a normal
`float16`), and no absolute term arises (products of fp8 values are at least `2^-32`, and
accumulators pass unchanged). A subnormal `float8e5` operand with a large partner makes `float16`
products lose bits on `mma.sync` too, far inside the facts (evidence/num-dot-sub16-round25 Part G2,
below `10^-4` of the bound). The evidence (Part C) checks every element against the
interval the facts imply for it (the per-chunk bound composed over the chunks with the exact partial
sums and the worst case of the unknown partial values): none is outside, and the largest error is
`0.468` of the bound's half-width on `wgmma` (`float8e4nv`, a subnormal operand setting the window,
31 products just below `u` and an accumulator just below `u`), `0.0002` on `mma.sync`; each chunk of
the model's chain is within `0.468` of the per-chunk bound.

**The loss facts' constants** (round 26; evidence/num-dot-chunk-round26, "Derivation"). On `wgmma`
a product truncated toward zero to a multiple of `u` loses at most `min(|p|, u)` (all of it when
`|p| < u`), and the accumulator less than `u`, so the truncated sum `S'` is within `A_j + u` of
`s j + c_j`; the cut to 14 bits loses less than `2^-13·|S'|`, and `|S'| ≤ |s j + c_j| + A_j + u`.
With `u ≤ 2^-13·Mj + 2^-139` as above, a chunk's error is below
`(1 + 2^-13)·A_j + (1 + 2^-13)·2^-13·Mj + 2^-13·|s j + c_j| + (1 + 2^-13)·2^-139`. The facts keep
`2·A_j` and `2^-12·Mj` (each a factor `1.9998`), `2^-12·|s j + c_j|` (a factor 2) and `2^-126`. The
worst case is an accumulator that sets the window with every bit below `u` set and a magnitude just
above `2^E`, with 32 products of its sign just below `u`: an error of about `33u` against about
`68u`. On `mma.sync` the two `float16` instructions of a chunk are within
`(1 + 2^-25 + 2^-22)·A_j + (2^-24 + 2^-22)·Mj + 2·2^-151 + 2·2^-149` (each loses at most the
products' magnitudes, `2^-25` of its largest lifted term, at most `Mj + A_j` for the second, and
`2^-23` of its truncated sum plus `2^-149` in the cut to `float32`). The evidence checks single
chunks on the device directly (with one chunk the partial values are the accumulator and the
result), every `K` that compiles, and tutorial 06's chains, on both paths and all four pairs: about
`9.1·10^7` elements, none outside; the largest ratio is `0.487` on `wgmma` (a whole dot of the
worst case above at `K = 1024`; `0.484` per chunk) and `0.0005` on `mma.sync`, and no result exceeds
`|acc| + Σ|a·b|` in magnitude.

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
`0xFF` as NaN, so both are outside `e8m0Dom`. The evidence checks both `fast_math` values. On
`scaledDomBF16` no operand of the emulated `bfloat16` dot is subnormal (a subnormal `e4m3` element,
`2^-9`, is a normal `bfloat16` value), so its lifted magnitudes are `|a·b|`, and `DotBF16`'s bound,
relative to lifted magnitudes since round 25, is the bound these facts state relative to
`|acc| + Σ|a·b|` (library issue L126; evidence/num-dot-sub16-round25 Part G3).

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

## The depth of a float reduction (`SumDepth`)

(Library issue L149, round 32; [evidence/reduce-tree](../../evidence/reduce-tree/README.md);
`primitives/triton-3.8.0-ReduceOpToLLVM.cpp`, `-TargetInfoBase.h`.) The library's `tl.sum` and its forms
pick a tree of depth at most `tl.sumBound P n = max (P.sumDepth n) ⌈log₂ n⌉` (`TVA.Lang.Ops`), any tree
under the default `Platform.sumDepth n = n − 1`. On sm_90 a float reduction over the `2^k` positions of
its axis is a perfect binary tree of depth `k`, whatever the layout: Triton 3.8.0 reduces each thread's
elements of the axis along a balanced binary tree (`treeReduce`, arity 2 on NVIDIA; `float16` and
`bfloat16` packed two by two), then along a `shfl.sync.bfly` butterfly over the lanes that hold the axis,
then, after a conversion through shared memory, along the same butterfly over the warps. The layout
decides only which positions pair up. `SumDepth P : ∀ n, P.sumDepth n ≤ Nat.clog 2 n` states it, and
`sum_err_depth` (`TVA.Lemmas.Sum`) and `sumc_err_depth` (`TVA.Lemmas.SumFused`) bound the result with
depth `⌈log₂ n⌉` and `n − 1` nodes, where `sum_err_covers` used `n − 1` for both. The evidence ran 7960
configurations, each checked twice (a tuple `tl.reduce` that returns its tree's depth and leaf count:
`log₂ n` over exactly `n` leaves; and float sums of leaves `1 + 2^-p` and `1 − 2^-(p+1)`, exact only
along a perfect tree): 1-D axes of 1 to 16384 positions with 1 to 16 warps and contiguous and strided
loads, in `float32`, `float16`, `bfloat16`, `float64` and of products; 2-D tiles along both axes,
row- and column-major, `tl.trans` and `keep_dims`; `tl.dot` results on `mma.sync` and `wgmma`; and
reductions of reductions; 259 distinct layouts of the reduced operand, none failing. It is not stated
for a reduction that Triton's `OptimizeThreadLocality` pass rewrites across a loop's trips, a
miscompile that is excluded ([exclusions.md](../exclusions.md); [SEMANTICS.md](../SEMANTICS.md),
"Float `+` and `-`", states the excluded loops on the source, more broadly than the pass's own pattern), nor for a scan (`tl.cumsum`), which is sequential within
a thread and over the warps, so that its depth depends on the layout.

## What the library does not state

A kernel that needs one of these asks the library first (protocol §2):
- `tl.dot` with `float32` operands and `tf32`, `tf32x3`, `bf16x3` or `bf16x6` precision; any
  `tl.dot` into `float16`; an fp8 `tl.dot` with a `max_num_imprecise_acc` other than the default;
  an `int8` `tl.dot` with a `max_num_imprecise_acc` given, or whose `|acc| + Σ|a·b|` exceeds
  `2^31 - 1` (then `mma.sync` saturates and `wgmma` wraps, so the result depends on the path:
  [evidence/num-dot-int8](../../evidence/num-dot-int8/README.md) I4);
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
    Part F; library issue L125). The same holds for `float16` and `bfloat16` operands on both paths
    ([evidence/num-dot-sub16-round25](../../evidence/num-dot-sub16-round25/README.md) Part A):
    `DotF16` and `DotBF16` did not cover it until round 25 (library issue L126), and are now relative
    to lifted magnitudes (`DotLiftAcc`).

  The evidence is a test at every `K` that compiles together with a per-chunk derivation that
  extends it to every `K` ("`tl.dot`", above).

## Satisfiability

A bundle proves its assumptions satisfiable together (`Claim.Consistent`, checked by the gate),
and every fact of the library must be satisfiable together with the others.

**The sm_90 facts.** `Platform.sm90ref` (`lean/TVA/Lemmas/Sm90.lean`) is `Platform.ref` with
one rounding's bound (`u·|y| + η`) as every function's bound, `exp(-∞) = exp2(-∞) = +0` in `float32`,
and `exp x = +∞` in `float32` from `1453635/16384` (where no float is within that bound of `e^x`:
`128·log 2 + 2^-23` is below it, `Real.log_two_lt_d9`) and at `+∞`; its
conversions to nearest saturate, as sm_90's into fp8 do. Every fact of `TVA.Rules.Num.Sm90` holds
of it (`Platform.sm90ref.expF32`, `.exp2F32`, …, `.expF32Overflow`, `.cvtF32E4m3`, …, `.cvtSatF32E4m3`, …,
`.dotF16`, `.dotBF16` (one rounding of the exact result, `DotAcc.liftAcc`), `.dotF16Chunk`,
`.dotBF16Chunk` (`Platform.ref.dotChunk16Loss`, in `lean/TVA/Lemmas/Dot16Chunk.lean`), `.sumDepth`
(`Platform.ref`'s trees are balanced), …, `.dotI8`
(`Platform.ref.dotAcc_int`), `.dotScaledE4m3`,
`.atomicAddF32`), and so do `DivFull`, `DivRn` and every
`…Total` rule, and the chunk facts of `TVA.Rules.Num.Dot8Chunk` (`Platform.sm90ref.dotE4m3Chunk`,
…, `.dotE5m2E4m3Chunk`, and the loss facts `.dotE4m3ChunkLoss`, …, `.dotE5m2E4m3ChunkLoss`, in
`lean/TVA/Lemmas/Dot8Chunk.lean`). A bundle that cites any of these facts proves `Claim.Consistent`
with this one platform.

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
- `MathAt` at non-finite inputs, given by the table `sp` (`mathAt`), and at finite inputs where
  no float is within the requested bound (`mathAt_far`; round 33, library issue L154);
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
- `DotLiftAcc` with a float accumulator, for `rel K ≥ u` and `abs K ≥ η` when `K > 0`
  (`Platform.ref.dotLiftAcc`, in `TVA.Lemmas.DotLift`: `dotAcc` and `DotAcc.liftAcc`);
- `DotScaledAcc` for `rel K ≥ u` and `abs K ≥ η` of `float32` when `K > 0` (`dotScaledAcc`);
- `DotChunkAcc` with a float accumulator, for `σ ≥ u`, `α ≥ η` and any `ρ ≥ 0`
  (`Platform.ref.dotChunkAcc`, in `TVA.Lemmas.Dot8Chunk`: the partial values are the exact partial
  sums, and the last step is the one rounding);
- `DotChunkLoss` with a float accumulator, for `σ ≥ u`, `α ≥ η` and any `ρ, κ, ρ' ≥ 0`
  (`Platform.ref.dotChunkLoss`: `dotChunkAcc` with `ρ = 0`, whose step bound is below both arms).

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
  `tt.precise_divf` (IEEE, `tl.div_rn` in the library), and so does `tl.fdiv(x, y,
  ieee_rounding=True)` (`semantic.py:337-338`), which is therefore `tl.div_rn` too and needs no fact
  (library issue L166; [evidence/fdiv-ieee](../../evidence/fdiv-ieee/README.md)). `tl.fdiv` does
  not convert its operands (`semantic.py:336`): on `float16` or `bfloat16` operands, either form fails
  to compile, and with `ieee_rounding=True` so does `float64` (`tt.precise_divf` lowers to
  `llvm.nvvm.div.rn.f`, a `float32` intrinsic, `nvidia-ElementwiseOpToLLVM.cpp:886-887`).
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
- **A following addition** (library issue L163). `ptxas` expands `div.full.f32` into `MUFU.RCP`
  and a multiply by the (scaled) dividend, and with `enable_fp_fusion` fuses that multiply with a
  following `add.f32` into one `FFMA`. The card is the quotient's, and the addition is a fusable one
  on it (SEMANTICS.md, "Float `+` and `-`": a `float32` quotient is a product), which
  [evidence/div-fusion](../../evidence/div-fusion/README.md) checks against `AddRel` at the quotient
  the division alone gives. `tl.div_rn`, `tl.fdiv(x, y, ieee_rounding=True)` and `float64`
  division are not fused.
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
- **Satisfiable.** `DivFull.of_divRn`: IEEE `float32` division meets the card, and
  `Platform.sm90ref.divF32InfDivisor` the infinite-divisor fact.
- **An infinite divisor** (`DivF32InfDivisor`, round 32, library issue L154): a finite dividend by
  `±∞` gives the zero of sign `sign x xor sign y`, as IEEE division does, on every finite `float32`
  dividend, by `+∞` and by `-∞` (about `8.6·10^9` pairs,
  [evidence/num-special-f32](../../evidence/num-special-f32/README.md) D1, no exception).

What the rules do not promise:
- nothing for `|x/y| > 2^126` (but an infinite divisor, above), a zero divisor, or an infinite
  dividend or a NaN operand (the result there is some value of `P.div`);
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
