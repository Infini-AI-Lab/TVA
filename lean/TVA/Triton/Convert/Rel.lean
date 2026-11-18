import TVA.Core

/-!
# The relations of conversions

What a conversion between integers and floats (`tl.ItofRel`, `FtoiRel`), and `semantic.cast` for
every pair of dtypes (`ConvRel`), may return (`TVA.Triton.Convert` gives the operations). They come
before the catalog of sites (`TVA.Site`), which uses them.
-/
namespace TVA

namespace tl

/-- The results an integer `z` may convert to in format `g`: rounded to nearest even once, or
twice through a wider standard format. Before sm_90 there is no direct integer-to-`bfloat16`
conversion, so it goes through `float32`; and MLIR folds a default-rounding narrowing of an
integer-to-float result into one conversion to the final format (`truncf(sitofp x) → sitofp x`,
`llvm-ArithCanonicalization.td:453-463`). So `x.to(float64).to(float32)`, and the implicit
narrowing of a converted integer by `tl.store`, the atomics and descriptor stores, are one
`tl.itof` to the final format. A narrowing with `fp_downcast_rounding="rtz"` is not folded: the
model writes `tl.itof` to the intermediate format, then `tl.cvt … .rz`. -/
def ItofRel (g : Format) (z : Int) (r : FloatV g) : Prop :=
  r = FloatV.ofRat (z : Rat) ∨
  ((g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32) ∧ r = FloatV.cvt (FloatV.ofRat (z : Rat) : F64)) ∨
  ((g = .binary16 ∨ g = .bfloat16) ∧ r = FloatV.cvt (FloatV.ofRat (z : Rat) : F32))

end tl

/-- A float `x` converted to the integer type `IntV sg w` (`fptosi`/`fptoui`): truncation toward
zero, from a standard format only; no result (undefined behaviour) when `x` is NaN, infinite or out
of the type's range (LLVM gives poison). -/
def FtoiRel {f : Format} (sg : Bool) (w : Nat) (x : FloatV f) (r : IntV sg w) : Prop :=
  f.Std ∧ x.isFinite = true ∧
    IntV.minInt sg w ≤ x.toRat.num.tdiv x.toRat.den ∧
    x.toRat.num.tdiv x.toRat.den ≤ IntV.maxInt sg w ∧
    r = IntV.ofInt (x.toRat.num.tdiv x.toRat.den)

/-- An integer value `z` (a source integer's value, or `0`/`1` for `int1`) converted to dtype
`d'` (`semantic.py:870-898`): `z ≠ 0` for `int1`; `z` modulo `2^w` for an integer (sign- or
zero-extension by the source's signedness, then truncation); `tl.ItofRel` for a standard float;
nothing into fp8 (rejected). -/
def ofIntRel (z : Int) : (d' : DType) → d'.Val → Prop
  | .i1 => fun r => r = decide (z ≠ 0)
  | .i8 => fun r => r = IntV.ofInt z | .i16 => fun r => r = IntV.ofInt z
  | .i32 => fun r => r = IntV.ofInt z | .i64 => fun r => r = IntV.ofInt z
  | .u8 => fun r => r = IntV.ofInt z | .u16 => fun r => r = IntV.ofInt z
  | .u32 => fun r => r = IntV.ofInt z | .u64 => fun r => r = IntV.ofInt z
  | .f16 => fun r => tl.ItofRel .binary16 z r | .bf16 => fun r => tl.ItofRel .bfloat16 z r
  | .f32 => fun r => tl.ItofRel .binary32 z r | .f64 => fun r => tl.ItofRel .binary64 z r
  | .f8e5 | .f8e4nv => fun _ => False

/-- A float `x` of format `f` converted to format `g` with the default rounding
(`semantic.py:838-868`): the same bits when `f = g`; between standard formats one rounding to
nearest even, up to NaN payload (`tl.cvt`); from `float16`, `bfloat16` or `float32` into fp8, the
platform's conversion (`tl.cvt8`); from fp8 into `float16`, `bfloat16` or `float32`, exact on finite
values (`tl.cvtFrom8`); no result for any other pair (fp8 to fp8, fp8 to or from `float64`). -/
def FloatRel (P : Platform) (f g : Format) (x : FloatV f) (r : FloatV g) : Prop :=
  if f = g then r.pat = x.pat
  else (f.Std ∧ g.Std ∧ FloatV.NaNEq (FloatV.cvt x) r) ∨
    ((f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 ∧ P.cvt f g .rn x r) ∨
    (f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32) ∧ (x.isFinite = true → r = FloatV.cvt x))

/-- A float `x` converted to dtype `d'`: `x != 0` for `int1` (unordered: a NaN gives `true`, `-0.0`
gives `false`; `semantic.py:884-887`), `FtoiRel` for an integer, `FloatRel` for a float. -/
def ofFloatRel (P : Platform) {f : Format} (x : FloatV f) : (d' : DType) → d'.Val → Prop
  | .i1 => fun r => r = FloatV.ne x (FloatV.ofPat 0)
  | .i8 => fun r => FtoiRel true 8 x r | .i16 => fun r => FtoiRel true 16 x r
  | .i32 => fun r => FtoiRel true 32 x r | .i64 => fun r => FtoiRel true 64 x r
  | .u8 => fun r => FtoiRel false 8 x r | .u16 => fun r => FtoiRel false 16 x r
  | .u32 => fun r => FtoiRel false 32 x r | .u64 => fun r => FtoiRel false 64 x r
  | .f16 => fun r => FloatRel P f .binary16 x r | .bf16 => fun r => FloatRel P f .bfloat16 x r
  | .f32 => fun r => FloatRel P f .binary32 x r | .f64 => fun r => FloatRel P f .binary64 x r
  | .f8e5 => fun r => FloatRel P f .e5m2 x r | .f8e4nv => fun r => FloatRel P f .e4m3 x r

/-- `semantic.cast(x, d')` with the default rounding, for one element of dtype `d`: what the
result `r` may be. No `r` exists for a conversion that does not compile or is undefined. -/
def ConvRel (P : Platform) : (d d' : DType) → d.Val → d'.Val → Prop
  | .i1, d' => fun x r => ofIntRel (if x then 1 else 0) d' r
  | .i8, d' => fun x r => ofIntRel x.toInt d' r | .i16, d' => fun x r => ofIntRel x.toInt d' r
  | .i32, d' => fun x r => ofIntRel x.toInt d' r | .i64, d' => fun x r => ofIntRel x.toInt d' r
  | .u8, d' => fun x r => ofIntRel x.toInt d' r | .u16, d' => fun x r => ofIntRel x.toInt d' r
  | .u32, d' => fun x r => ofIntRel x.toInt d' r | .u64, d' => fun x r => ofIntRel x.toInt d' r
  | .f16, d' => fun x r => ofFloatRel P x d' r | .bf16, d' => fun x r => ofFloatRel P x d' r
  | .f32, d' => fun x r => ofFloatRel P x d' r | .f64, d' => fun x r => ofFloatRel P x d' r
  | .f8e5, d' => fun x r => ofFloatRel P x d' r | .f8e4nv, d' => fun x r => ofFloatRel P x d' r

end TVA
