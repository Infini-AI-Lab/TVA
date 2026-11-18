import TVA.Triton.Arith
import TVA.Triton.Convert.Rel

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
  .pick (.guard (f.Std ∧ g.Std ∧ RoundOk f g rnd)) fun _ =>
    upToNaN (x.map (FloatV.cvtWith rnd))

/-- `x.to(dtype, fp_downcast_rounding=rnd)` from `float16`, `bfloat16` or `float32` into an fp8
format: as the platform converts (`Platform.cvt f g rnd`; whether it saturates, and how it treats
NaN, is the lowering's). -/
def cvt8 {g : Format} (x : Tensor s (FloatV f)) (rnd : RoundMode := .rn) :
    Prog (Tensor s (FloatV g)) :=
  .pick (.guard ((f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 ∧ RoundOk f g rnd))
    fun _ => .plat (.cvt f g rnd x) .ret

/-- `x.to(dtype)` from fp8 (`float8e5`, `float8e4nv`) to `float16`, `bfloat16` or `float32`:
exact on finite elements. An infinity or NaN may come out as any value: Triton converts
`float8e5` to `bfloat16` by a bit trick that turns them into finite numbers (`Fp8E5M2_to_Bf16`,
`nvidia-ElementwiseOpToLLVM.cpp:68-121`). There is no fp8 → `float64` conversion. -/
def cvtFrom8 {g : Format} (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV g)) :=
  .pick (.guard (f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32))) fun _ =>
  .pick (.fromF8 s f g x) .ret

/-! ## Between integers and floats

`tl.ItofRel` and `FtoiRel` are in `TVA.Triton.Convert.Rel`. -/

/-- `x.to(float dtype)` of an integer tensor (`sitofp`/`uitofp`) into a standard format: each
element as `ItofRel` allows. Into fp8 it does not compile (`semantic.py:916`). -/
def itof {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) : Prog (Tensor s (FloatV f)) :=
  .pick (.guard (f.Std)) fun _ =>
  .pick (.itof s f sg w x) .ret

/-- `x.to(int dtype)` of a float tensor in a standard format: each element truncated toward zero
(`FtoiRel`); undefined behaviour when an element is NaN, infinite or out of range. From fp8 it does
not compile (`semantic.py:916`). -/
def ftoi {sg : Bool} {w : Nat} (x : Tensor s (FloatV f)) : Prog (Tensor s (IntV sg w)) :=
  .pick (.guard (f.Std)) fun _ =>
  .pick (.ftoi s f sg w x) .ret

end tl

/-! ## `semantic.cast` for every pair of dtypes

`ConvRel` and the relations it is made of (`ofIntRel`, `FloatRel`, `ofFloatRel`) are in
`TVA.Triton.Convert.Rel`. -/

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

namespace tl

/-- `x.to(d')` with the default rounding, and every implicit conversion of an operand (the
promotion of a mixed-dtype operator, the condition of `tl.where` to `int1`): each element as
`ConvRel` allows; undefined behaviour if some element has no result. -/
def convert {s : Shape} (d d' : DType) (x : Tensor s d.Val) : Prog (Tensor s d'.Val) :=
  .env fun e => .pick (.convert e.P s d d' x) .ret

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
