import TVA.Triton.Core

/-!
# Arithmetic: floats, integers, math functions

The arithmetic operators of a Triton kernel on tensors, after the operands have been converted to
the dtype Triton computes in (`TVA.Triton.Promote`; the model writes the conversions).

**Floats.** Every float operation is a `Prog`: an exact one (`tl.add`, `tl.mul`, `tl.fma`,
`tl.div_rn`, `tl.neg`, …) returns its IEEE result (round to nearest even) up to NaN payload
(`upToNaN`); an approximate one (`/`, `tl.exp`, `tl.sqrt`, …) returns what the platform's relation
allows (`Platform.div`, `Platform.math`), which the device facts constrain.

**Fusion.** With `enable_fp_fusion` (the launch's `Env.fusion`, Triton's default `True`), the
backend may contract a product and an addition into one `fma`. A *product* is an operand computed
by `*`, `tl.dot`, `tl.rand`, or a float `/` whose quotient is `float32` (`div.full.f32` ends in a
multiply that `ptxas` may fuse), through any dataflow, inlined helpers and library functions
included (`tl.sigmoid` of a `float32` tensor is one). An addition or subtraction that a product may
feed is written in its fusable form, any other in its IEEE form:

| IEEE form | fusable form |
|---|---|
| `tl.add` | `tl.mul_add a b c` for `a * b + c`; `tl.addc` otherwise |
| `tl.sub` | `tl.subc` |

and likewise for reductions and scans (`TVA.Triton.Reduce`, `TVA.Triton.Scan`). The fusable forms
allow every result the IEEE forms allow, so writing an IEEE addition in fusable form is sound but
loose. Contraction is the only float rewrite the backend applies: no reassociation, no reciprocal
division, no flushing of subnormals.

**Integers.** `+ - *`, comparisons and bitwise operations wrap and are exact: they are the
operations of `TVA.Tensor` (`x + y`, `Tensor.lt`, `Tensor.le`, `Tensor.eq`; `x > y` is
`Tensor.lt y x`, `x != y` is `(Tensor.eq x y).not`). Float comparisons are `Tensor.flt`, `fle`,
`feq`, `fne` (ordered, so a comparison with a NaN is false, except `!=`). The operations here are
those that can be undefined: `//`, `%`, shifts.
-/
namespace TVA

/-! ## The relations of float results -/

/-- What `tl.maximum` (`isMax`) or `tl.minimum` may return on two elements: the IEEE
`maxNum`/`minNum` up to NaN payload, or either operand when both are zeros (LLVM `maxnum` leaves
the sign of a zero result open). -/
def FloatV.MaxRel {f : Format} (isMax : Bool) (x y r : FloatV f) : Prop :=
  FloatV.NaNEq (if isMax then FloatV.max x y else FloatV.min x y) r ∨
    (x.isZero = true ∧ y.isZero = true ∧ (r = x ∨ r = y))

/-- `q` is an exact value that rounds (to nearest even) to `x`: it rounds to a finite value
equal to `x`, or to the same infinity (a product that overflowed when rounded, but enters an
`fma` exactly). -/
def RoundsTo {f : Format} (q : Rat) (x : FloatV f) : Prop :=
  (f.isFinite (f.roundRat .rn false false q) = true ∧ x.isFinite = true ∧
      f.toRat (f.roundRat .rn false false q) = x.toRat) ∨
    (f.isInf (f.roundRat .rn false false q) = true ∧ f.roundRat .rn false false q = x.pat)

/-- The results an addition `x + y` may produce when either operand may be a product.

Without fusion, the IEEE sum up to NaN payload. With fusion, also what an `fma` gives: an `fma`
takes one product unrounded and its other operand as it is (sm_90 has no instruction that adds two
unrounded products). The model does not track which operand is the product, so this allows:
- the rounding of `p + q` for exact `p`, `q` that round to the operands (`RoundsTo`), one of them
  the operand's own value (`p = x.toRat` or `q = y.toRat`), with either sign for an exact zero;
- an infinite operand added to an exact other operand, which gives that infinity. -/
def AddRel {f : Format} (fusion : Bool) (x y r : FloatV f) : Prop :=
  FloatV.NaNEq (FloatV.add x y) r ∨
    (fusion = true ∧
      ((∃ p q : Rat, RoundsTo p x ∧ RoundsTo q y ∧ (p = x.toRat ∨ q = y.toRat) ∧ ∃ z : Bool,
          FloatV.NaNEq (FloatV.ofPat (f.roundRat .rn false z (p + q))) r) ∨
        (∃ p : Rat, RoundsTo p x ∧ y.isInf = true ∧ r = y) ∨
        (∃ q : Rat, RoundsTo q y ∧ x.isInf = true ∧ r = x)))

/-- The IEEE sum is always a result `AddRel` allows. -/
theorem AddRel.ieee {f : Format} (fusion : Bool) (x y : FloatV f) :
    AddRel fusion x y (FloatV.add x y) :=
  Or.inl (Or.inl rfl)

/-- Without fusion, `AddRel` is the IEEE sum up to NaN payload. -/
theorem AddRel.false_iff {f : Format} {x y r : FloatV f} :
    AddRel false x y r ↔ FloatV.NaNEq (FloatV.add x y) r :=
  ⟨fun h => h.resolve_right (fun h' => by cases h'.1), Or.inl⟩

namespace tl

variable {s : Shape}

/-! ## Float arithmetic -/

section Float
variable {f : Format}

/-- The results of an elementwise operation whose exact result is `v`: `v` up to NaN payload. -/
def upToNaN (v : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun r : Tensor s (FloatV f) => ∀ i, FloatV.NaNEq (v.get i) (r.get i))
    (.ok v fun _ => Or.inl rfl) .ret

/-- `x + y` on floats that no product feeds: the IEEE sum. -/
def add (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  upToNaN (Tensor.map2 FloatV.add x y)

/-- `x - y` on floats that no product feeds: the IEEE difference. -/
def sub (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  upToNaN (Tensor.map2 FloatV.sub x y)

/-- `x * y` on floats: the IEEE product (a following addition may fuse it). -/
def mul (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  upToNaN (Tensor.map2 FloatV.mul x y)

/-- A source `a * b + c`: with fusion, either the two IEEE operations or one `fma`; without it,
the two operations. -/
def mul_add (a b c : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor s (FloatV f) => ∀ i,
    FloatV.NaNEq (FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)) (r.get i) ∨
    (e.fusion = true ∧ FloatV.NaNEq (FloatV.fma (a.get i) (b.get i) (c.get i)) (r.get i)))
    (.ok ⟨fun i => FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)⟩ fun _ => Or.inl (Or.inl rfl))
    .ret

/-- A source `x + y` that a product may feed (whose product the model does not see, or with
products on both sides): any result `AddRel` allows. -/
def addc (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor s (FloatV f) =>
    ∀ i, AddRel e.fusion (x.get i) (y.get i) (r.get i))
    (.ok (Tensor.map2 FloatV.add x y) fun _ => AddRel.ieee _ _ _) .ret

/-- A source `x - y` that a product may feed: `x + (-y)` under `addc` (`a*b - c` and `c - a*b`
both contract to an `fma`). -/
def subc (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := addc x (y.map FloatV.neg)

/-- `tl.fma(x, y, z)`: `x * y + z` with one rounding. -/
def fma (x y z : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  upToNaN (Tensor.map3 FloatV.fma x y z)

/-- `x / y` on floats, and `tl.fdiv(x, y)` (`ieee_rounding=False`): the platform's division
(`Platform.div`; `div.full.f32` on `float32`, not the IEEE quotient). Triton divides `float16`
and `bfloat16` operands in `float32` (`semantic.py:93-102`): the model converts first. A `float32`
quotient is a product for a following addition. -/
def div (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .plat (.div f x y) .ret

/-- `tl.div_rn(x, y)`: the IEEE quotient (`div.rn`), not a product. -/
def div_rn (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  upToNaN (Tensor.map2 FloatV.divRn x y)

/-- `tl.fdiv(x, y, ieee_rounding)`: `tl.div` by default, `tl.div_rn` with `ieee_rounding=True`
(both compile to `tt.precise_divf`, `div.rn.f32`, which accepts `float32` operands only;
`semantic.py:331-341`). -/
def fdiv (x y : Tensor s (FloatV f)) (ieee_rounding : Bool := false) :
    Prog (Tensor s (FloatV f)) :=
  if ieee_rounding then div_rn x y else div x y

/-- `tl.sqrt_rn(x)`: the IEEE square root. -/
def sqrt_rn (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (x.map FloatV.sqrtRn)

/-- An approximate elementwise function, as the platform computes it (`Platform.math`). Triton
accepts these on `float32` and `float64` only (`math.py:94-98`); a model converts first. -/
def math (fn : MathFn) (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .plat (.math fn f x) .ret

/-- `tl.exp(x)`. -/
def exp (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .exp x
/-- `tl.exp2(x)` (also `tl.math.exp2`). -/
def exp2 (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .exp2 x
/-- `tl.log(x)`. -/
def log (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .log x
/-- `tl.log2(x)`. -/
def log2 (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .log2 x
/-- `tl.sin(x)`. -/
def sin (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .sin x
/-- `tl.cos(x)`. -/
def cos (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .cos x
/-- `tl.sqrt(x)` (approximate; `tl.sqrt_rn` is exact). -/
def sqrt (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .sqrt x
/-- `tl.rsqrt(x)`. -/
def rsqrt (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .rsqrt x
/-- `tl.erf(x)`. -/
def erf (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := math .erf x

/-- `-x` on floats: exact (a NaN gives some NaN). -/
def neg (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (x.map FloatV.neg)

/-- `tl.abs(x)` on floats: exact (a NaN gives some NaN). -/
def abs (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (x.map FloatV.abs)

/-- `tl.maximum(x, y)` on floats (default `propagate_nan=NONE`): `MaxRel`. Triton converts
`bfloat16` operands to `float32` first (`core.py:2927`): the model converts, and the result is
`float32`. -/
def maximum (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun r : Tensor s (FloatV f) => ∀ i, FloatV.MaxRel true (x.get i) (y.get i) (r.get i))
    (.ok (Tensor.map2 FloatV.max x y) fun _ => Or.inl (Or.inl rfl)) .ret

/-- `tl.minimum(x, y)` on floats: as `tl.maximum`, with the minimum. -/
def minimum (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun r : Tensor s (FloatV f) => ∀ i, FloatV.MaxRel false (x.get i) (y.get i) (r.get i))
    (.ok (Tensor.map2 FloatV.min x y) fun _ => Or.inl (Or.inl rfl)) .ret

/-- `tl.sigmoid(x)`: Triton's `@triton.jit` body `1 / (1 + math.exp(-x))` (`standard.py:44-50`),
inlined: the negation, `tl.exp`, the IEEE addition of the constant `1`, and the default float
division of `1` by the sum. `tl.exp` admits only `float32` and `float64`, so the model converts a
narrower tensor first. On `float32` its result is a product for a following addition. -/
def sigmoid (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := do
  let nx ← neg x
  let ex ← exp nx
  let den ← add (Tensor.full (FloatV.ofRat 1)) ex
  div (Tensor.full (FloatV.ofRat 1)) den

end Float

/-! ## Integer arithmetic that can be undefined

`//`, `%`, `<<` and `>>` on integer tensors. A lane is undefined when it divides by zero,
overflows (`INT_MIN // -1`) or shifts by the width or more, even a lane a later `tl.where`
discards (LLVM's `sdiv` by zero is immediate undefined behaviour). Tensor `//` and `%` truncate
toward zero (C semantics), unlike Python's on compile-time values (`tl.py`). -/

section Int
variable {sg : Bool} {w : Nat}

/-- `tl.maximum(x, y)` on integer tensors, and Python's `max(x, y)` in a kernel when an operand is a
tensor: exact, signed or unsigned by the dtype (`arith.maxsi`/`maxui`, `semantic.py:366-398`). -/
def imaximum (x y : Tensor s (IntV sg w)) : Tensor s (IntV sg w) := Tensor.map2 IntV.max x y

/-- `tl.minimum(x, y)` on integer tensors, and Python's `min(x, y)` when an operand is a tensor. -/
def iminimum (x y : Tensor s (IntV sg w)) : Tensor s (IntV sg w) := Tensor.map2 IntV.min x y

/-- `x // y` on integer tensors: truncating division (`IntV.div?`). -/
def idiv (x y : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, (x.get i).div? (y.get i) = some (r.get i))
    (.lanes fun i => (x.get i).div? (y.get i)) .ret

/-- `x % y` on integer tensors: the remainder of truncating division (`IntV.rem?`). -/
def irem (x y : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, (x.get i).rem? (y.get i) = some (r.get i))
    (.lanes fun i => (x.get i).rem? (y.get i)) .ret

/-- `tl.cdiv(x, div)` with a tensor operand: Triton's `@triton.jit` body `(x + (div - 1)) // div`
(`standard.py:32-43`), inlined, so the addition wraps and `//` truncates. It is not ceiling
division: with `int32`, `tl.cdiv(2^31 - 1, 128)` is `-16777215` and `tl.cdiv(-5, 4)` is `0`. On two
compile-time values it is Python's (`tl.py.cdiv`). -/
def cdiv (x div : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  idiv (x + (div - Tensor.full 1)) div

/-- `x << n`: both operands have the promoted type (`semantic.py:421-425`). Undefined when an
amount, read as unsigned, is at least the width (a negative amount is a huge one). -/
def shl (x n : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i,
    (x.get i).shl? (n.get i).bits.toNat = some (r.get i))
    (.lanes fun i => (x.get i).shl? (n.get i).bits.toNat) .ret

/-- `x >> n`, arithmetic. Triton picks `ashr` or `lshr` by the source dtype of the tensor operand
Python dispatches on, before promotion: the left one when it is a tensor (`__rshift__`), the right
one when the left is a Python number (`__rrshift__`, `core.py:1012-1024`). So `int32 >> uint32` is an
arithmetic shift of `uint32` values, and `0xF0 >> s` with `s : uint8` is a logical one. -/
def ashr (x n : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i,
    (x.get i).ashr? (n.get i).bits.toNat = some (r.get i))
    (.lanes fun i => (x.get i).ashr? (n.get i).bits.toNat) .ret

/-- `x >> n`, logical: the dispatching operand's source dtype is unsigned (see `ashr`). -/
def lshr (x n : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i,
    (x.get i).lshr? (n.get i).bits.toNat = some (r.get i))
    (.lanes fun i => (x.get i).lshr? (n.get i).bits.toNat) .ret

end Int

end tl

end TVA
