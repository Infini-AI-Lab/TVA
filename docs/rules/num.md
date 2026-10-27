# Rules `Num.*`: facts about the platform

Lean: `TVA.Rules.Num` (`lean/TVA/Rules/Num.lean`).

The Triton documentation does not define these operations exactly:
- `tl.exp` and the other approximate math functions;
- `/` on floats (`tl.fdiv`, not IEEE: `div.full.f32` on fp32);
- `tl.dot`;
- conversions into fp8;
- float `atomic_add`.

The library gives them as relations in the platform `P`, and a claim quantifies over `P`. A
bundle that needs a property of one of them assumes it as a field of `Assumptions`, citing a
fact the library states for the operation ("Facts the library states", below), with the
library's arguments:

```lean
  /-- [rule] `Num.DivTotal`: `/` on fp32 always has a result. -/
  div_total : Rules.Num.DivTotal P Format.binary32
  /-- [rule] `Num.DivFull`: `/` on fp32 is `div.full.f32`, within 2 ulp. -/
  div_full : Rules.Num.DivFull P
```

The definitions:

| Definition | Says | Typical use |
|---|---|---|
| `MathTotal P fn f` | `P.math fn f` has a result for every input | `safe_math` |
| `DivTotal P f`, `CvtTotal P f g rnd`, `DotTotal P cfg`, `AtomicAddTotal P f` | the same for the other operations | `safe_div`, `safe_cvt8`, `safe_dot`, atomics |
| `MathAcc P fn f dom rel abs` | finite inputs in `dom` and in `fn.natDom`, with room in the format (`(1+rel)·|ref x| + abs ≤ maxVal`), give finite results within `rel·|ref x| + abs` of the real function `fn.ref` | accuracy of softmax, GELU, … |
| `MathAt P fn f x y` | at the input `x`, the result is exactly `y` | `exp(-∞) = +0` in a masked softmax |
| `DivAcc P f qmax rel abs` | finite `x / y` (`y ≠ 0`, `|x/y| ≤ qmax`) is finite and within `rel·|x/y| + abs` | normalisation |
| `DivFull P` | `DivAcc P binary32 2^126 2^-22 2^-148`: fp32 `/` is `div.full.f32`, 2 ulp | softmax, layer norm, any fp32 `/` |
| `DivRn P f` | `/` on `f` is IEEE division up to NaN payload (`DivRn.divAcc`: half an ulp) | fp64 `/` |
| `CvtAcc P f g` | with `.rn`, a finite value in `g`'s range converts to a finite value within `|x|·u + η` (one rounding's bound) | fp8 quantisation |
| `AtomicAddAcc P f rel abs` | a float `atomic_add` with an in-range exact sum is finite and within `rel·|o+v| + abs` | split-K accumulation |
| `DotAcc P cfg dom rel abs` | per output `(i, n)`: if row `i` of `a` and column `n` of `b` have values (`DType.val?`: finite floats, integers) in `dom` and the accumulator has one, the result has a value within `rel K·(|acc| + Σ|a·b|) + abs K` of `acc + Σ a·b`, when that leaves room below the accumulator's `DType.bound` | GEMM accuracy; with `rel = abs = 0`, exact int8 GEMM |
| `DotScaledTotal P cfg` | `P.dotScaled cfg` has a result for all operands | `safe_dot_scaled` |
| `DotScaledAcc P cfg dom sdom rel abs` | per output `(i, n)`: if row `i` of `lhs` and column `n` of `rhs` have finite scaled values in `dom`, scale bits in `sdom`, and the accumulator is finite, the result is finite and within `rel K·(|acc| + Σ|a·b|) + abs K` of `acc + Σ a·b` | MXFP4/MXFP8 matmuls |

## What a kernel reviewer checks

A kernel reviewer trusts this page ([REVIEW_PROTOCOL.md](../REVIEW_PROTOCOL.md) §1–§2): whether
a fact is true of the device, how Triton lowers the operation, and whether the constants are
right are settled in library review, below. For each `[rule]` field that cites a `Num.*`
definition, the kernel reviewer checks that it is **the library's fact for the operation the
model uses**:
- **The operation.** The definition is the one for the model's operation: `MathTotal`,
  `MathAcc`, `MathAt` with the model's `MathFn` for `tl.exp`, `tl.exp2`, …; `DivTotal`, `DivFull`
  for `/` on `float32`; `DivTotal`, `DivRn` for `/` on `float64`; `DotTotal`, `DotAcc` with the
  model's `DotCfg`; `CvtTotal`, `CvtAcc` for a conversion into fp8; `AtomicAddTotal`,
  `AtomicAddAcc` for a float `atomic_add`; `DotScaledTotal`, `DotScaledAcc` with the model's
  `ScaledCfg`.
- **The format and configuration** are the model's: the same `Format`, rounding mode, `DotCfg`
  or `ScaledCfg` as the operation in the model.
- **The arguments are the library's.** The field is a fact of the table below, with the same
  arguments. A field whose constants (`dom`, `rel`, `abs`, `qmax`, `x`, `y`, …) the kernel chose
  is not a library fact (`RULE_MISCITED`). A kernel that needs a fact the table lacks asks the
  library for it and waits (protocol §2).

A doubt about the facts themselves is a `LIBRARY_REPORT`.

## Facts the library states (sm_90, Triton 3.8.0)

| Fact | Operation | Evidence |
|---|---|---|
| `MathTotal P fn f`, `DivTotal P f`, `CvtTotal P f g rnd`, `DotTotal P cfg`, `AtomicAddTotal P f`, `DotScaledTotal P cfg`, for every function, format and configuration | the operation gives some result for every input | the device's instruction returns a value; a combination Triton cannot lower is rejected on the host, with no outcome ([compiler-faithful.md](compiler-faithful.md)) |
| `DivFull P` | `/` on `float32` | the lowering and the PTX ISA's 2 ulp ("Division", below); TVA's own measurement is library issue L4 |
| `DivRn P .binary64` | `/` on `float64` | the lowering ("Division", below) |

The accuracy facts with constants (`MathAcc`, `MathAt`, `DotAcc`, `CvtAcc`, `AtomicAddAcc`,
`DotScaledAcc`) are not stated yet: library issue L4 covers `tl.math.exp2`, `tl.math.log2`, `/`
and `tl.dot` with `float16` and `bfloat16` operands. Until then a kernel cannot cite them.

## Device questions (library review)

These are for library review and the library author, who settle them with Triton 3.8.0's sources
(`primitives/`) and TVA's evidence on the target (`evidence/`). Kernel reviewers do not check
them. A fact for an operation must take into account:
- `tl.exp` on fp32 lowers to `ex2.approx` after a multiplication by `log2 e`, so its error is
  relative and grows with `|x|`;
- `/` on fp32 is `div.full.f32`, so its fact is `DivFull`, not a tighter `DivAcc` ("Division",
  below);
- `/` on fp64 is `div.rn.f64` (IEEE), so `DivRn P .binary64` holds. `DivRn P .binary32` does
  not hold on NVIDIA;
- `atom.add.f32` rounds to nearest but flushes subnormal inputs and results to zero. Two
  operands just below the smallest normal sum to almost `2^-125` and give `0`, so `abs` must be
  at least `2^-125` for fp32 (`2^(emin+1)` in general);
- `P.atomicAdd` covers both `atom.add` and the TMA reduction of `desc.atomic_add`
  (`cp.reduce.async.bulk`); an `AtomicAddAcc` fact must hold for every instruction a kernel may
  use;
- a scalar `atomic_add(p, 0)` (or `±0.0`) with `relaxed` or `acquire` order is lowered to a plain
  load (`triton-3.8.0-nvidia-LoadStoreOpToLLVM.cpp:648-687`), so memory keeps `o`. Every
  `P.atomicAdd` fact must therefore admit `n = o` when `v = ±0`, even where an exact sum or a
  flush would give another value;
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
  `float32 → fp8` goes through a truncation to `float16` and stays within `CvtAcc`'s bound, not
  within half an ulp. On sm_80, `bfloat16 → float8e5` clamps below `2^-15` to `0`, which exceeds
  the bound for tiny inputs. `CvtAcc` speaks only of `.rn`; an `.rz` conversion has no fact yet;
- **`tl.dot` (`DotAcc`)**, judged for one `DotCfg` and target:
  - the input precision (`tf32` keeps 10 fraction bits of fp32 operands; `tf32x3`, `bf16x3` and
    `bf16x6` split them into several products);
  - `maxImpreciseAcc`: an fp8 dot on sm_90 sums that many products in the tensor core's
    reduced-precision accumulator before adding them into fp32;
  - the summation order the tensor cores use, which `rel K` must cover for every order.

  The evidence is a test at the largest `K` the fact covers, or a derivation. A fact that derives
  `rel K` from per-precision facts (exact products, a summation bound over all orders) is
  planned.

## Satisfiability

A bundle proves its assumptions satisfiable together (`Claim.Consistent`, checked by the gate),
and every fact of the library must be satisfiable together with the others.
`Platform.ref rel abs sp` (`lean/TVA/Lemmas/Consistency.lean`) is the most accurate platform.
Approximate functions return a float within the requested bound whenever one exists, and
`tl.dot`/`tl.dot_scaled` round the exact result once. It satisfies all of these together, with
`u = 2^-(m+1)` and `η = 2^(emin−m−1)` of the format:
- every `…Total` rule (`mathTotal`, `divTotal`, `cvtTotal`, `atomicAddTotal`, `dotTotal`,
  `dotScaledTotal`);
- `MathAcc` with `rel ≥ u`:
  - for `abs ≥ η` on any domain (`mathAcc_of_eta`);
  - for any `abs ≥ 0` where the real function's values are normal, `|ref x| ≥ 2^emin`
    (`mathAcc_of_normal`);
  - wherever the bundle proves that floats that accurate exist (`mathAcc`);
- `MathAt` at non-finite inputs, given by the table `sp` (`mathAt`);
- `DivRn`, `DivFull`;
- `CvtAcc`;
- `AtomicAddAcc` for `rel ≥ u`, `abs ≥ η`;
- `DotAcc`:
  - with a float accumulator, for `rel K ≥ u` and `abs K ≥ η` when `K > 0` (`dotAcc`);
  - with integer operands and a signed accumulator, for any `rel`, `abs ≥ 0` (`dotAcc_int`).
    Triton 3.8.0's only integer `tl.dot` is `int8 × int8 → int32` (`semantic.py:1483-1486`), so
    no other integer case arises;
- `DotScaledAcc` for `rel K ≥ u` and `abs K ≥ η` of `float32` when `K > 0` (`dotScaledAcc`).

At `K = 0` the bounds need only be nonnegative, because the result is the accumulator. So
constants that vanish there, such as `γ_K = K·u / (1 − K·u)`, are fine. The rounding argument is
`FloatV.exists_nearest`: the nearest float to a real is within half an ulp, `u·|y| + η`, and
`u·|y|` in the normal range. A bound tighter than that is false of any device that rounds.

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
  ulp. For `|x/y| ≤ 2^126` that is at most `2^-22·|x/y| + 2^-148`. TVA's own measurement on the
  target is library issue L4 ([ISSUES.md](../library-review/ISSUES.md)).
- **Satisfiable.** `DivFull.of_divRn`: IEEE `float32` division meets the card.

What the rule does not promise:
- nothing for `|x/y| > 2^126`, a zero divisor, or an infinite or NaN operand (the result there is
  some value of `P.div`);
- not determinism: each element of `tl.div` is some value within the card at its operands. A
  constant quotient folded by MLIR (IEEE) and the same quotient computed by the unit may differ.

## Exact operations need no rule

`+`, `-`, `*`, `tl.fma`, `tl.div_rn`, `tl.sqrt_rn` and conversions between the standard float
formats are IEEE operations in the library, up to NaN payload, so they need no rule. Their error bounds are proved:
`FloatV.add_err`, `sub_err`, `mul_err`, `fma_err`, `divRn_err` and `cvt_err` (`TVA.Lemmas.Num`). Each gives
`|q| · u + η` for an in-range exact result `q`, with `u = 2^-(m+1)` and `η = 2^(emin−m−1)`.
