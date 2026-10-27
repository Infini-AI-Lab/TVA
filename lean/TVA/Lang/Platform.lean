import TVA.Lang.Mem

/-!
# The platform: what the documentation leaves to the device

Some Triton operations have no exact documented meaning. The device computes `tl.exp`,
`tl.sqrt` and `/` on `tl.float32` with approximate instructions. `tl.dot` uses tensor cores whose
accumulation order and precision are the hardware's. Conversions into fp8 saturate or not by
lowering. Float atomics round as the memory system does. `tl.dot_scaled` runs on native
microscaling instructions or on an emulation.

The library does not guess these. A `Platform` gives each such operation as a *relation*
between operands and results; nothing about it is defined. A claim quantifies over platforms,
and the accuracy facts a proof needs are separate, reviewable rules about `P`
(`TVA.Rules.Num.*`: for example "`P.math .exp2` is within 2 ulp"). A per-element relation also
lets two occurrences of the same operation return different results; nothing assumes one global
function.

`Env` is what a running program can depend on besides memory: the platform and the kernel's
`enable_fp_fusion` option (default `True`).
-/
namespace TVA

open TVA.Math.Float

/-- Triton's approximate elementwise math functions (`tl.exp`, `tl.math.exp2`, …). `tl.abs`,
`tl.fma`, `tl.div_rn` and `tl.sqrt_rn` are exact and are not here. -/
inductive MathFn where
  | exp | exp2 | log | log2 | sin | cos | sqrt | rsqrt | erf
  deriving DecidableEq, Repr

/-- `tl.dot`'s `input_precision`: how `float32` operands are fed to the tensor cores
(`allowed_dot_input_precisions`, `triton-3.8.0-nvidia-compiler.py:128`). -/
inductive InputPrecision where
  | tf32 | tf32x3 | ieee | bf16x3 | bf16x6
  deriving DecidableEq, Repr

/-- One `tl.dot` configuration: operand dtypes, accumulator (= `out_dtype`), precision mode, and
the resolved `max_num_imprecise_acc`: the source's value, or Triton's default, which is `2^30`
for an fp8 × fp8 dot on sm_90 (imprecise accumulation) and `0` otherwise
(`semantic.py:1516-1521`, `triton-3.8.0-nvidia-compiler.py:214`). An accuracy assumption about
`P.dot cfg` is judged for this value. -/
structure DotCfg where
  a : DType
  b : DType
  acc : DType
  precision : InputPrecision := .tf32
  maxImpreciseAcc : Nat := 0
  deriving DecidableEq, Repr

/-- The element formats of `tl.dot_scaled` operands (`lhs_format`, `rhs_format`). -/
inductive MXFmt where
  | e2m1 | e4m3 | e5m2 | bf16 | fp16
  deriving DecidableEq, Repr

/-- The format of `tl.dot_scaled` scales. The IR accepts `i8` scale tensors, read as `e8m0`
(`uint8` or `int8` in the source), and float scale tensors (`TritonOps.td:692-693`), such as
NVFP4's `float8e4nv` scales. -/
inductive ScaleFmt where
  | e8m0
  | float (d : DType)
  deriving DecidableEq, Repr

/-- One `tl.dot_scaled` configuration: operand formats, the format of each side's scales (the IR
types them independently, `triton-3.8.0-TritonOps.td:692-693`), elements per scale along `K`
(the scale factor, `16` or `32`, which Triton deduces from the scale shapes), packing of `e2m1`
operands (`lhs_k_pack`, `rhs_k_pack`) and `fast_math`. -/
structure ScaledCfg where
  lhs : MXFmt
  rhs : MXFmt
  lhsScale : ScaleFmt := .e8m0
  rhsScale : ScaleFmt := .e8m0
  group : Nat := 32
  lhsKPack : Bool := true
  rhsKPack : Bool := true
  fastMath : Bool := false
  deriving DecidableEq, Repr

/-- The platform's meaning of the operations Triton does not define exactly. Each field is a
relation from operands to the results the device may produce. -/
structure Platform where
  /-- `tl.exp(x)` etc. on one element of format `f`: `math fn f x r`. -/
  math : MathFn → (f : Format) → FloatV f → FloatV f → Prop
  /-- `x / y` (`tl.fdiv`, `ieee_rounding=False`) on one element: `div f x y r`. -/
  div : (f : Format) → FloatV f → FloatV f → FloatV f → Prop
  /-- `tl.dot(a, b, acc)`: `dot cfg a b acc r`. -/
  dot : (cfg : DotCfg) → {M N K : Nat} → Tensor [M, K] cfg.a.Val → Tensor [K, N] cfg.b.Val →
    Tensor [M, N] cfg.acc.Val → Tensor [M, N] cfg.acc.Val → Prop
  /-- A conversion into an fp8 format with rounding mode `rnd` (`.rn`, or `.rz` for
  `fp_downcast_rounding="rtz"`): `cvt f g rnd x r`. -/
  cvt : (f g : Format) → RoundMode → FloatV f → FloatV g → Prop
  /-- The new value of a float `tl.atomic_add` (and of a float descriptor `atomic_add`):
  `atomicAdd f old v new`. -/
  atomicAdd : (f : Format) → FloatV f → FloatV f → FloatV f → Prop
  /-- `tl.dot_scaled`: `dotScaled cfg a sa b sb acc r`. `a` (`[M, K]`) and `b` (`[K, N]`) are
  the operands' element bit patterns in their formats, unpacked. `sa`/`sb` give each element's
  scale bits in `cfg.lhsScale`/`cfg.rhsScale` (`none` without scales). The accumulator and result are
  `float32`. -/
  dotScaled : (cfg : ScaledCfg) → {M N K : Nat} → Tensor [M, K] Nat → Option (Tensor [M, K] Nat) →
    Tensor [K, N] Nat → Option (Tensor [K, N] Nat) → Tensor [M, N] F32 → Tensor [M, N] F32 → Prop

/-- What a running program may consult besides memory. -/
structure Env where
  P : Platform
  /-- `enable_fp_fusion` (Triton's default: `True`): the compiler may fuse a multiplication into
  a following addition (`fma`). -/
  fusion : Bool := true

end TVA
