import TVA.Tensor

/-!
# The platform: what Triton leaves to the device

Most Triton operations have an exact meaning (`TVA.Base`, `TVA.Tensor`). A few do not: the
compiler lowers them to device instructions whose results are documented only up to an accuracy,
or depend on choices the lowering makes.

* `tl.exp`, `tl.log`, `tl.sqrt`, … on `float32` use approximate instructions (`ex2.approx`,
  `lg2.approx`, `sqrt.approx`, …).
* `/` on floats (`tl.fdiv` with `ieee_rounding=False`, the default) uses `div.full.f32`.
* `tl.dot` runs on tensor cores, whose accumulation order and internal precision are the
  hardware's.
* Conversions into an fp8 format are done by instructions whose saturation and rounding are the
  lowering's.
* A float `tl.atomic_add` is performed by the memory system, which rounds as it does.
* `tl.dot_scaled` runs on native microscaling instructions or on an emulation.
* A float reduction combines its elements along a tree whose shape the lowering decides.

TVA does not guess these. A `Platform` gives each as a *relation* between operands and the
results the device may produce; nothing about it is defined here. A claim quantifies over every
platform that satisfies the device facts (`TVA.Device`: for example, that `P.math .exp2` on
`float32` is within a stated relative error of `2^x`), so a proof works for every result those
facts allow. A relation, rather
than a function, also lets two occurrences of one operation on the same operands return different
results, as the device may (another instruction selection, another fusion).

A program reads the platform through `Env` (`Prog.env`, `TVA.Program`), together with the
kernel's `enable_fp_fusion` option.

Used by `TVA.Triton.Arith`, `TVA.Triton.Convert`, `TVA.Triton.Memory` (float atomics),
`TVA.Triton.Dot`, `TVA.Triton.Reduce`, and `TVA.Device`.
-/
namespace TVA

/-- Triton's approximate elementwise math functions (`tl.exp`, `tl.math.exp2`, …). `tl.abs`,
`tl.fma`, `tl.div_rn` and `tl.sqrt_rn` are exact and are not here. -/
inductive MathFn where
  /-- `tl.exp`. -/
  | exp
  /-- `tl.exp2`. -/
  | exp2
  /-- `tl.log`. -/
  | log
  /-- `tl.log2`. -/
  | log2
  /-- `tl.sin`. -/
  | sin
  /-- `tl.cos`. -/
  | cos
  /-- `tl.sqrt` (not `tl.sqrt_rn`). -/
  | sqrt
  /-- `tl.rsqrt`. -/
  | rsqrt
  /-- `tl.erf`. -/
  | erf
  deriving DecidableEq, Repr

/-- `tl.dot`'s `input_precision`: how `float32` operands are fed to the tensor cores
(`allowed_dot_input_precisions`, `nvidia-compiler.py:128`). -/
inductive InputPrecision where
  /-- Operands rounded to TF32. -/
  | tf32
  /-- Three TF32 products, emulating full precision. -/
  | tf32x3
  /-- Full `float32` products (FMA on CUDA cores). -/
  | ieee
  /-- Three bf16 products. -/
  | bf16x3
  /-- Six bf16 products. -/
  | bf16x6
  deriving DecidableEq, Repr

/-- One `tl.dot` configuration: operand dtypes, accumulator dtype (`out_dtype`), precision mode,
and the resolved `max_num_imprecise_acc`. -/
structure DotCfg where
  /-- The dtype of the left operand. -/
  a : DType
  /-- The dtype of the right operand. -/
  b : DType
  /-- The dtype of the accumulator and result (`out_dtype`). -/
  acc : DType
  /-- The source's `input_precision`. When the source omits it, Triton uses
  `triton.knobs.language.fp32_default` if set, and otherwise `tf32` on sm_90 (`ieee` with
  `allow_tf32=False`; `core.py:2384-2388`). TVA assumes Triton's knobs at their defaults, so an
  omitted `input_precision` is `tf32`. -/
  precision : InputPrecision := .tf32
  /-- The resolved `max_num_imprecise_acc`: the source's value, or Triton's default, which is
  `2^30` for an fp8 × fp8 dot on sm_90 (imprecise accumulation) and `0` otherwise
  (`semantic.py:1516-1521`, `nvidia-compiler.py:214`). -/
  maxImpreciseAcc : Nat := 0
  deriving DecidableEq, Repr

/-- The element formats of `tl.dot_scaled` operands (`lhs_format`, `rhs_format`). -/
inductive MXFmt where
  /-- `"e2m1"` (FP4), packed two per byte. -/
  | e2m1
  /-- `"e4m3"`. -/
  | e4m3
  /-- `"e5m2"`. -/
  | e5m2
  /-- `"bf16"`. -/
  | bf16
  /-- `"fp16"`. -/
  | fp16
  deriving DecidableEq, Repr

/-- The format of `tl.dot_scaled` scales. The IR accepts `i8` scale tensors, read as `e8m0`
(`uint8` or `int8` in the source), and float scale tensors such as NVFP4's `float8e4nv` scales
(`TritonOps.td:692-693`). -/
inductive ScaleFmt where
  /-- A power of two, `2^(b - 127)` for the byte `b`. -/
  | e8m0
  /-- A float of dtype `d`. -/
  | float (d : DType)
  deriving DecidableEq, Repr

/-- One `tl.dot_scaled` configuration. -/
structure ScaledCfg where
  /-- The left operand's element format. -/
  lhs : MXFmt
  /-- The right operand's element format. -/
  rhs : MXFmt
  /-- The format of the left scales (the IR types each side's independently,
  `TritonOps.td:692-693`). -/
  lhsScale : ScaleFmt := .e8m0
  /-- The format of the right scales. -/
  rhsScale : ScaleFmt := .e8m0
  /-- Elements per scale along `K` (`16` or `32`; Triton deduces it from the scale shapes). -/
  group : Nat := 32
  /-- `lhs_k_pack`: `e2m1` left operands are packed along `K`. -/
  lhsKPack : Bool := true
  /-- `rhs_k_pack`: `e2m1` right operands are packed along `K`. -/
  rhsKPack : Bool := true
  /-- `fast_math`. -/
  fastMath : Bool := false
  deriving DecidableEq, Repr

/-- The device's meaning of the operations Triton does not define exactly. Each field relates
operands to every result the device may produce; a claim holds for every platform satisfying the
device facts. -/
structure Platform where
  /-- `math fn f x r`: `r` is a result the device may give for the approximate function `fn` on
  one element `x` of format `f` (`tl.exp(x)`, …). A relation because the instructions are
  specified only up to an accuracy. -/
  math : MathFn → (f : Format) → FloatV f → FloatV f → Prop
  /-- `div f x y r`: `r` is a result the device may give for `x / y` on one element (`tl.fdiv`
  with `ieee_rounding=False`, the default `/`). A relation because `div.full.f32` is not the
  IEEE quotient. -/
  div : (f : Format) → FloatV f → FloatV f → FloatV f → Prop
  /-- `dot cfg a b acc r`: `r` is a result the device may give for `tl.dot(a, b, acc)`. A
  relation because the tensor cores' accumulation order, internal precision and rounding are the
  hardware's. -/
  dot : (cfg : DotCfg) → {M N K : Nat} → Tensor [M, K] cfg.a.Val → Tensor [K, N] cfg.b.Val →
    Tensor [M, N] cfg.acc.Val → Tensor [M, N] cfg.acc.Val → Prop
  /-- `cvt f g rnd x r`: `r` is a result the device may give for the conversion of `x` into the
  fp8 format `g` with rounding mode `rnd` (`.rn`, or `.rz` for `fp_downcast_rounding="rtz"`).
  A relation because whether it saturates, and how it treats NaN, is the lowering's. -/
  cvt : (f g : Format) → RoundMode → FloatV f → FloatV g → Prop
  /-- `atomicAdd f old v new`: `new` is a value a float `tl.atomic_add` of `v` may leave in a
  cell that held `old` (the cell's old value is what the atomic returns). A relation because the
  memory system's float addition is not documented to be the IEEE sum. -/
  atomicAdd : (f : Format) → FloatV f → FloatV f → FloatV f → Prop
  /-- `dotScaled cfg a sa b sb acc r`: `r` is a result the device may give for `tl.dot_scaled`.
  `a` (`[M, K]`) and `b` (`[K, N]`) are the operands' element bit patterns in their formats,
  unpacked; `sa`/`sb` give each element's scale bits in `cfg.lhsScale`/`cfg.rhsScale` (`none`
  without scales). The accumulator and result are `float32`. A relation because the device's
  microscaling instructions (or Triton's emulation) accumulate as they do. -/
  dotScaled : (cfg : ScaledCfg) → {M N K : Nat} → Tensor [M, K] Nat → Option (Tensor [M, K] Nat) →
    Tensor [K, N] Nat → Option (Tensor [K, N] Nat) → Tensor [M, N] F32 → Tensor [M, N] F32 → Prop
  /-- How deep the tree of a float reduction over `n` positions may be (`tl.sum` and the other
  float reductions): the device combines along some tree of depth at most
  `max (sumDepth n) ⌈log₂ n⌉` (the `⌈log₂ n⌉` keeps a balanced tree always allowed). This is a
  bound rather than a relation because the reduction's result is already chosen through its tree
  (`Tree.Eval`); only the trees' shape is the device's. The default `n - 1` allows every tree
  over the positions. -/
  sumDepth : Nat → Nat := fun n => n - 1

/-- What a running program may read besides memory (`Prog.env`). -/
structure Env where
  /-- The platform. -/
  P : Platform
  /-- The kernel's `enable_fp_fusion` option (Triton's default: `True`): when set, the compiler
  may fuse a multiplication into a following addition (`fma`). -/
  fusion : Bool := true

end TVA
