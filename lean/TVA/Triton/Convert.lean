import TVA.Triton.Arith

/-!
# Conversions: `x.to(d)`, `tl.cast`, bit casts

`x.to(d, fp_downcast_rounding=r, bitcast=b)`, `tl.cast(x, d, r, b)` and `x.cast(d, r, b)` are one
operation (`core.py:1130-1135, 2322-2345`). Its arguments are compile-time values. `tl.cast` first
makes a Python number a scalar tensor of its own dtype (`TVA.Triton.Promote`), then converts.

* To `x`'s own dtype it is `x` itself, whatever `r`.
* Without `bitcast`, it is `semantic.cast` (`semantic.py:793-916`): `tl.convert d d' x` with the
  default rounding, which covers every pair of dtypes, or one of its parts when the model wants a
  narrower operation: `tl.cvt` between the standard float formats (also with `r = "rtz"`),
  `tl.cvt8` into fp8, `tl.cvtFrom8` out of fp8, `tl.itof` from an integer, `tl.ftoi` to an integer,
  and `IntV.cast` / `Tensor.castInt` between integers (wrapping, or extending by the source's
  signedness). A `fp_downcast_rounding` on a conversion that is not from a float to a float with
  fewer bits, or any value but `"rtne"` and `"rtz"`, is rejected on the host
  (`semantic.py:824-832`).
* With `bitcast=True`, it is `tl.bitcast`: each element's bits read in the target dtype.
* `x.to(tl.pointer_type(d))` of an integer is `tl.intToPtr`.

Every implicit conversion of the source (the promotion of a mixed-dtype operator, the value of
`tl.store` and of the atomics, `other=` of `tl.load`) is one of these, written by the model with
the default rounding: into an fp8 pointer that is `tl.cvt8 x .rn`, the device's saturating
conversion.

Float conversions in short: between the standard formats one rounding (to nearest even, or toward
zero with `"rtz"`, from `float32` only), widening exact; into fp8 the platform's `Platform.cvt`;
out of fp8 exact on finite values; there is no fp8 ↔ `float64` and no fp8 ↔ fp8 conversion.
-/
namespace TVA

namespace tl

variable {s : Shape} {f : Format}

/-! ## Between float formats -/

/-- The rounding modes `x.to(dtype, fp_downcast_rounding=…)` allows from format `f` to `g`:
`"rtne"` (`.rn`, the default) always; `"rtz"` (`.rz`) only from `float32` to `float16`,
`bfloat16` (`cvt.rz`) or `float8e5`. Other pairs do not compile (`nvidia-ElementwiseOpToLLVM.cpp:
485-490`). -/
def RoundOk (f g : Format) (rnd : RoundMode) : Prop :=
  rnd = .rn ∨ (rnd = .rz ∧ f = .binary32 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .e5m2))

instance (f g : Format) (rnd : RoundMode) : Decidable (RoundOk f g rnd) := by
  unfold RoundOk; infer_instance

/-- `x.to(dtype, fp_downcast_rounding=rnd)` between standard float formats (`float16`,
`bfloat16`, `float32`, `float64`): one rounding (`.rn` to nearest even, the default; `.rz` toward
zero), up to NaN payload; widening is exact. Undefined behaviour for other formats, which use
`tl.cvt8` (into fp8) and `tl.cvtFrom8` (out of fp8). -/
def cvt {g : Format} (x : Tensor s (FloatV f)) (rnd : RoundMode := .rn) :
    Prog (Tensor s (FloatV g)) :=
  .pick (fun _ : Unit => f.Std ∧ g.Std ∧ RoundOk f g rnd) .guard fun _ =>
    upToNaN (x.map (FloatV.cvtWith rnd))

/-- `x.to(dtype, fp_downcast_rounding=rnd)` from `float16`, `bfloat16` or `float32` into an fp8
format: as the platform converts (`Platform.cvt f g rnd`; whether it saturates, and how it treats
NaN, is the lowering's). -/
def cvt8 {g : Format} (x : Tensor s (FloatV f)) (rnd : RoundMode := .rn) :
    Prog (Tensor s (FloatV g)) :=
  .pick (fun _ : Unit => (f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 ∧ RoundOk f g rnd)
    .guard fun _ => .plat (.cvt f g rnd x) .ret

/-- `x.to(dtype)` from fp8 (`float8e5`, `float8e4nv`) to `float16`, `bfloat16` or `float32`:
exact on finite elements. An infinity or NaN may come out as any value: Triton converts
`float8e5` to `bfloat16` by a bit trick that turns them into finite numbers (`Fp8E5M2_to_Bf16`,
`nvidia-ElementwiseOpToLLVM.cpp:68-121`). There is no fp8 → `float64` conversion. -/
def cvtFrom8 {g : Format} (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV g)) :=
  .pick (fun _ : Unit => f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32)) .guard fun _ =>
  .pick (fun r : Tensor s (FloatV g) =>
    ∀ i, (x.get i).isFinite = true → r.get i = FloatV.cvt (x.get i))
    (.ok (x.map FloatV.cvt) fun _ _ => rfl) .ret

/-! ## Between integers and floats -/

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

/-- `x.to(float dtype)` of an integer tensor (`sitofp`/`uitofp`) into a standard format: each
element as `ItofRel` allows. Into fp8 it does not compile (`semantic.py:916`). -/
def itof {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun _ : Unit => f.Std) .guard fun _ =>
  .pick (fun r : Tensor s (FloatV f) => ∀ i, ItofRel f (x.get i).toInt (r.get i))
    (.ok (x.map fun a => FloatV.ofRat (a.toInt : Rat)) fun _ => Or.inl rfl) .ret

/-- A float `x` converted to the integer type `IntV sg w` (`fptosi`/`fptoui`): truncation toward
zero, from a standard format only; no result (undefined behaviour) when `x` is NaN, infinite or out
of the type's range (LLVM gives poison). -/
def _root_.TVA.FtoiRel {f : Format} (sg : Bool) (w : Nat) (x : FloatV f) (r : IntV sg w) : Prop :=
  f.Std ∧ x.isFinite = true ∧
    IntV.minInt sg w ≤ x.toRat.num.tdiv x.toRat.den ∧
    x.toRat.num.tdiv x.toRat.den ≤ IntV.maxInt sg w ∧
    r = IntV.ofInt (x.toRat.num.tdiv x.toRat.den)

/-- The only candidate `FtoiRel` allows, decided (for the solvers of `ftoi` and `convert`). -/
def ftoiAns {f : Format} (sg : Bool) (w : Nat) (x : FloatV f) : Option (Ans (FtoiRel sg w x)) :=
  if h : f.Std ∧ x.isFinite = true ∧ IntV.minInt sg w ≤ x.toRat.num.tdiv x.toRat.den ∧
      x.toRat.num.tdiv x.toRat.den ≤ IntV.maxInt sg w then
    some (.ok _ ⟨h.1, h.2.1, h.2.2.1, h.2.2.2, rfl⟩)
  else some (.empty fun _ hr => h ⟨hr.1, hr.2.1, hr.2.2.1, hr.2.2.2.1⟩)

/-- `x.to(int dtype)` of a float tensor in a standard format: each element truncated toward zero
(`FtoiRel`); undefined behaviour when an element is NaN, infinite or out of range. From fp8 it does
not compile (`semantic.py:916`). -/
def ftoi {sg : Bool} {w : Nat} (x : Tensor s (FloatV f)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun _ : Unit => f.Std) .guard fun _ =>
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, FtoiRel sg w (x.get i) (r.get i))
    (.each fun i => tl.ftoiAns sg w (x.get i)) .ret

end tl

/-! ## `semantic.cast` for every pair of dtypes -/

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

/-- A dtype's width in bits, Triton's `primitive_bitwidth`: `1` for `int1`, eight bits per byte
for the others. -/
def DType.bitwidth (d : DType) : Nat := if d = .i1 then 1 else 8 * d.bytes

/-- Reading a value's bit pattern back in its own dtype gives the value. -/
theorem DType.ofBits_toBits : (d : DType) → (v : d.Val) → d.ofBits (d.toBits v) = v
  | .i1, b => by cases b <;> rfl
  | .i8, ⟨b⟩ | .i16, ⟨b⟩ | .i32, ⟨b⟩ | .i64, ⟨b⟩ | .u8, ⟨b⟩ | .u16, ⟨b⟩ | .u32, ⟨b⟩ | .u64, ⟨b⟩ => by
      simp [DType.ofBits, DType.toBits, IntV.ofNat]
  | .f16, ⟨b⟩ | .bf16, ⟨b⟩ | .f32, ⟨b⟩ | .f64, ⟨b⟩ | .f8e5, ⟨b⟩ | .f8e4nv, ⟨b⟩ => by
      simp [DType.ofBits, DType.toBits, FloatV.ofPat, FloatV.pat]

/-! ### Solving a conversion

What the solver of `tl.convert` computes for one element: the conversion's result, with its proof,
or "no result". It gives up (`none`) only on a conversion into fp8, which is the platform's. -/

/-- `ofIntRel z d'`, solved. -/
def ofIntAns (z : Int) : (d' : DType) → Option (Ans (ofIntRel z d'))
  | .i1 => some (.ok _ rfl)
  | .i8 => some (.ok _ rfl) | .i16 => some (.ok _ rfl) | .i32 => some (.ok _ rfl)
  | .i64 => some (.ok _ rfl)
  | .u8 => some (.ok _ rfl) | .u16 => some (.ok _ rfl) | .u32 => some (.ok _ rfl)
  | .u64 => some (.ok _ rfl)
  | .f16 => some (.ok _ (Or.inl rfl)) | .bf16 => some (.ok _ (Or.inl rfl))
  | .f32 => some (.ok _ (Or.inl rfl)) | .f64 => some (.ok _ (Or.inl rfl))
  | .f8e5 => some (.empty fun _ h => h) | .f8e4nv => some (.empty fun _ h => h)

/-- `FloatRel P f g x`, solved, except into fp8 from a format the platform converts. -/
def floatAns (P : Platform) (f g : Format) (x : FloatV f) : Option (Ans (FloatRel P f g x)) :=
  if h : f = g then some (.ok (h ▸ x) (by subst h; simp [FloatRel]))
  else if hs : f.Std ∧ g.Std then
    some (.ok (FloatV.cvt x) (by rw [FloatRel, ite_eq_right h]; exact Or.inl ⟨hs.1, hs.2, Or.inl rfl⟩))
  else if hp : (f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 then none
  else if h8 : f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32) then
    some (.ok (FloatV.cvt x) (by rw [FloatRel, ite_eq_right h]; exact Or.inr (Or.inr ⟨h8.1, h8.2, fun _ => rfl⟩)))
  else some (.empty fun r hr => by
    rw [FloatRel, ite_eq_right h] at hr
    rcases hr with ⟨h1, h2, _⟩ | ⟨h1, h2, _⟩ | ⟨h1, h2, _⟩
    · exact hs ⟨h1, h2⟩
    · exact hp ⟨h1, h2⟩
    · exact h8 ⟨h1, h2⟩)

/-- `ofFloatRel P x d'`, solved, except into fp8. -/
def ofFloatAns (P : Platform) {f : Format} (x : FloatV f) : (d' : DType) → Option (Ans (ofFloatRel P x d'))
  | .i1 => some (.ok _ rfl)
  | .i8 => tl.ftoiAns true 8 x | .i16 => tl.ftoiAns true 16 x | .i32 => tl.ftoiAns true 32 x
  | .i64 => tl.ftoiAns true 64 x
  | .u8 => tl.ftoiAns false 8 x | .u16 => tl.ftoiAns false 16 x | .u32 => tl.ftoiAns false 32 x
  | .u64 => tl.ftoiAns false 64 x
  | .f16 => floatAns P f _ x | .bf16 => floatAns P f _ x | .f32 => floatAns P f _ x
  | .f64 => floatAns P f _ x | .f8e5 => floatAns P f _ x | .f8e4nv => floatAns P f _ x

/-- `ConvRel P d d' x`, solved, except into fp8. -/
def convAns (P : Platform) : (d d' : DType) → (x : d.Val) → Option (Ans (ConvRel P d d' x))
  | .i1, d', x => ofIntAns (if x then 1 else 0) d'
  | .i8, d', x => ofIntAns x.toInt d' | .i16, d', x => ofIntAns x.toInt d'
  | .i32, d', x => ofIntAns x.toInt d' | .i64, d', x => ofIntAns x.toInt d'
  | .u8, d', x => ofIntAns x.toInt d' | .u16, d', x => ofIntAns x.toInt d'
  | .u32, d', x => ofIntAns x.toInt d' | .u64, d', x => ofIntAns x.toInt d'
  | .f16, d', x => ofFloatAns P x d' | .bf16, d', x => ofFloatAns P x d'
  | .f32, d', x => ofFloatAns P x d' | .f64, d', x => ofFloatAns P x d'
  | .f8e5, d', x => ofFloatAns P x d' | .f8e4nv, d', x => ofFloatAns P x d'

namespace tl

/-- `x.to(d')` with the default rounding, and every implicit conversion of an operand (the
promotion of a mixed-dtype operator, the condition of `tl.where` to `int1`): each element as
`ConvRel` allows; undefined behaviour if some element has no result. -/
def convert {s : Shape} (d d' : DType) (x : Tensor s d.Val) : Prog (Tensor s d'.Val) :=
  .env fun e => .pick (fun r : Tensor s d'.Val => ∀ i, ConvRel e.P d d' (x.get i) (r.get i))
    (.each fun i => convAns e.P d d' (x.get i)) .ret

/-- `x.to(d', bitcast=True)` of a tensor of dtype `d` (`semantic.py:793-809`): each element's bit
pattern read as a value of `d'`, exact, NaN payloads included. Triton requires the same width
(`DType.bitwidth`, the proof `h`, found by itself for literal dtypes) and rejects every other pair
on the host. A pointer operand or target is converted as without `bitcast`. -/
def bitcast {s : Shape} (d d' : DType) (x : Tensor s d.Val)
    (_h : d.bitwidth = d'.bitwidth := by decide) : Tensor s d'.Val :=
  x.map fun v => d'.ofBits (d.toBits v)

/-- A bit cast to the tensor's own dtype is the tensor itself, as Triton returns its input. -/
theorem bitcast_self {s : Shape} (d : DType) (x : Tensor s d.Val) (h : d.bitwidth = d.bitwidth) :
    bitcast d d x h = x := by
  ext i
  exact DType.ofBits_toBits d (x.get i)

/-- `x.to(tl.pointer_type(d))` of an integer scalar (`inttoptr`, `semantic.py:909-910`): the
pointer whose address is the integer's bits read unsigned (LLVM zero-extends a narrower integer to
the 64-bit pointer size). -/
def intToPtr {sg : Bool} {w : Nat} (d : DType) (x : IntV sg w) : Ptr d := ⟨(x.bits.toNat : Int)⟩

end tl

end TVA
