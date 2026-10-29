import TVA.Lang.Prog

/-!
# The Triton operations (`tl.*`)

A prover writes the model of a kernel with these definitions, named as in `triton.language`, so
that the model reads like the source. Each definition is the library's reading of the Triton
3.8.0 documentation and lowering; its docstring says which reading. The reviewer may challenge a
reading, and the library then changes the definition, not the bundles' proofs.

Conventions:
* Exact operations are pure Lean functions (`TVA.Lang.Tensor`: integer arithmetic,
  comparisons, `tl.where`, broadcasting). Operations that round, are approximate or
  nondeterministic, or touch memory, return a `Prog`.
* Implicit broadcasting and implicit dtype conversions of the source are written explicitly in
  the model (`bc`, `.col`, `tl.cvt`, …).
* Shapes are Lean values (`Nat`), just as `tl.constexpr` block sizes are compile-time values.
-/
namespace TVA

open TVA.Math.Float Prog

/-- What a program knows about itself: `tl.program_id(axis)` and `tl.num_programs(axis)`. -/
structure Ctx where
  pid : Fin 3 → Nat
  grid : Fin 3 → Nat

/-! ## Float results the device may produce

NaN payloads are unspecified on the device. PTX gives "an unspecified NaN" for `neg`, `abs` and
arithmetic on NaN inputs, and MLIR folds such as `x * 1.0 → x` keep an input's payload. So a
NaN result of an exact operation may be any NaN (`FloatV.NaNEq`). `tl.maximum` and `tl.minimum`
may return either zero for `(+0, -0)` (`MaxRel`: LLVM `maxnum`). -/

/-- Equal, or both NaN. -/
def FloatV.NaNEq {f : Format} (x y : FloatV f) : Prop := x = y ∨ (x.isNaN = true ∧ y.isNaN = true)

theorem FloatV.NaNEq.refl {f : Format} (x : FloatV f) : FloatV.NaNEq x x := Or.inl rfl

/-- A result that is not NaN is exact. -/
theorem FloatV.NaNEq.eq {f : Format} {x y : FloatV f} (h : FloatV.NaNEq x y) (hx : x.isNaN = false) :
    y = x := by
  rcases h with h | ⟨h, -⟩
  · exact h.symm
  · rw [hx] at h; cases h

/-- What `tl.maximum` (`isMax`) or `tl.minimum` may return: the library's `max`/`min` up to NaN
payload, or either operand when both are zeros. -/
def FloatV.MaxRel {f : Format} (isMax : Bool) (x y r : FloatV f) : Prop :=
  FloatV.NaNEq (if isMax then FloatV.max x y else FloatV.min x y) r ∨
    (x.isZero = true ∧ y.isZero = true ∧ (r = x ∨ r = y))

/-! ## Rounding of fused additions -/

/-- `q` is an exact value that rounds (to nearest even) to `x`. Either it rounds to a finite
value equal to `x`, or it rounds to the same infinity (a product that overflowed when rounded,
but enters an `fma` exactly). -/
def RoundsTo {f : Format} (q : Rat) (x : FloatV f) : Prop :=
  (f.isFinite (f.roundRat .rn false false q) = true ∧ x.isFinite = true ∧
      f.toRat (f.roundRat .rn false false q) = x.toRat) ∨
    (f.isInf (f.roundRat .rn false false q) = true ∧ f.roundRat .rn false false q = x.pat)

/-- The results an addition `x + y` may produce.

Without fusion, the result is the IEEE sum (up to NaN payload). With `enable_fp_fusion` (the
default), the compiler may compute `a * b + c` as `fma(a, b, c)`. Then an operand that is a
rounded product enters the addition unrounded, as any exact value that rounds to it (finite or
not). The model does not track which operands are products, so this allows:
- the rounding of `p + q` for exact `p`, `q` rounding to the operands;
- an infinite operand added to an exact other operand, which gives that infinity.

This soundly over-approximates every fusion. Error bounds lose at most one rounding's bound
(`u·|y| + η`) per operand, and an exact zero result may have either sign. -/
def AddRel {f : Format} (fusion : Bool) (x y r : FloatV f) : Prop :=
  FloatV.NaNEq (FloatV.add x y) r ∨
    (fusion = true ∧
      ((∃ p q : Rat, RoundsTo p x ∧ RoundsTo q y ∧ ∃ z : Bool,
          FloatV.NaNEq (FloatV.ofPat (f.roundRat .rn false z (p + q))) r) ∨
        (∃ p : Rat, RoundsTo p x ∧ y.isInf = true ∧ r = y) ∨
        (∃ q : Rat, RoundsTo q y ∧ x.isInf = true ∧ r = x)))

theorem AddRel.ieee {f : Format} (fusion : Bool) (x y : FloatV f) : AddRel fusion x y (FloatV.add x y) :=
  Or.inl (FloatV.NaNEq.refl _)

/-- Without fusion, `AddRel` is the IEEE sum up to NaN payload. -/
theorem AddRel.false_iff {f : Format} {x y r : FloatV f} :
    AddRel false x y r ↔ FloatV.NaNEq (FloatV.add x y) r :=
  ⟨fun h => h.resolve_right (fun h' => by cases h'.1), Or.inl⟩

/-- The addition of an accumulator type: `AddRel` for the standard float formats, wrapping `+`
for integers. -/
def AccAdd (fusion : Bool) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .f16, x, y, r => AddRel fusion x y r | .bf16, x, y, r => AddRel fusion x y r
  | .f32, x, y, r => AddRel fusion x y r | .f64, x, y, r => AddRel fusion x y r
  | .i8, x, y, r => r = x + y | .i16, x, y, r => r = x + y | .i32, x, y, r => r = x + y
  | .i64, x, y, r => r = x + y
  | .u8, x, y, r => r = x + y | .u16, x, y, r => r = x + y | .u32, x, y, r => r = x + y
  | .u64, x, y, r => r = x + y
  | _, _, _, _ => False

/-- `padding_option` of a block-pointer or descriptor load: what a lane outside the bounds
holds. `undef` is an unspecified value (block pointers' default `""`), `zero` the zero bit
pattern, `nan` a NaN (float dtypes only; Triton rejects it for integers). -/
inductive Padding where
  | undef | zero | nan
  deriving DecidableEq, Repr

/-- A lane outside the bounds holds what the padding says. -/
def Padding.Holds {d : DType} : Padding → d.Val → Prop
  | .undef, _ => True
  | .zero, x => x = d.ofBits 0
  | .nan, x => d.IsNaN x

/-- The integer a compiler hint speaks about: an integer's value, or a pointer's byte address.
`unit` is the step between consecutive elements (`1`, or the pointer's element size: Triton
counts a pointer's contiguity in elements and its divisibility in bytes). -/
class HintVal (β : Type) where
  val : β → Int
  unit : Int

instance {sg : Bool} {w : Nat} : HintVal (IntV sg w) := ⟨IntV.toInt, 1⟩
instance {d : DType} : HintVal (Ptr d) := ⟨Ptr.addr, d.bytes⟩

/-- Positions `i` and `j` lie in the same group of `C` along dimension `k`: they agree on every
other coordinate, and their `k`-th coordinates have the same quotient by `C`. -/
def Idx.SameGroup {s : Shape} (k C : Nat) (i j : Idx s) : Prop :=
  (∀ k', k' ≠ k → i.coord k' = j.coord k') ∧ i.coord k / C = j.coord k / C

/-- Along dimension `k`, each group of `C` positions (`Idx.SameGroup`) holds consecutive values,
`HintVal.unit β` apart. -/
def ContigAlong {s : Shape} {β : Type} [HintVal β] (x : Tensor s β) (k C : Nat) : Prop :=
  ∀ i j : Idx s, Idx.SameGroup k C i j →
    HintVal.val (x.get j) = HintVal.val (x.get i) + ((j.coord k : Int) - i.coord k) * HintVal.unit β

/-- `C` is `x`'s contiguity along dimension `k`: each group of `C` positions holds consecutive
values, and, unless `2·C` exceeds the dimension's size, the groups of `2·C` positions do not all do
so. Since a group of `2·C` consecutive values splits into two groups of `C`, `C` is then the
largest such group size. -/
def ContigIs {s : Shape} {β : Type} [HintVal β] (x : Tensor s β) (k C : Nat) : Prop :=
  ContigAlong x k C ∧ (2 * C ≤ s.dim k → ¬ ContigAlong x k (2 * C))

/-- `v` divides the first element of each group of `C` positions along dimension `k`: the elements
whose coordinate `k` is a multiple of `C`. With `C = 1`, every element. -/
def FirstsDiv {s : Shape} {β : Type} [HintVal β] (x : Tensor s β) (k C v : Nat) : Prop :=
  ∀ i : Idx s, i.coord k % C = 0 → (v : Int) ∣ HintVal.val (x.get i)

/-- Along dimension `k`, each group of `C` positions holds equal values. -/
def ConstAlong {s : Shape} {β : Type} (x : Tensor s β) (k C : Nat) : Prop :=
  ∀ i j : Idx s, Idx.SameGroup k C i j → x.get j = x.get i

namespace tl

variable {s : Shape} {α : Type}

/-! ## Program identity -/

/-- `tl.program_id(axis)` (`tl.int32`). -/
def program_id (c : Ctx) (axis : Fin 3) : I32 := IntV.ofNat (c.pid axis)
/-- `tl.num_programs(axis)` (`tl.int32`). -/
def num_programs (c : Ctx) (axis : Fin 3) : I32 := IntV.ofNat (c.grid axis)

/-! ## Creation -/

/-- `tl.arange(start, end)`. -/
def arange (start stop : Nat) : Tensor [stop - start] I32 := Tensor.arange start (stop - start)

/-- `tl.zeros(shape, dtype)`: the all-zero bit pattern (`+0.0`, `0`, `False`). -/
def zeros (s : Shape) (d : DType) : Tensor s d.Val := Tensor.full (d.ofBits 0)

/-- `tl.full(shape, value, dtype)`. The model writes the constant Triton builds: any `value`
equal to zero, `-0.0` included, becomes the null value `+0.0` (`scalar_constant`,
`semantic.py:601-613`). The same holds for a Python scalar operand (`semantic.py:205-206`), so
`x + (-0.0)` adds `+0.0`. -/
def full (s : Shape) {α : Type} (a : α) : Tensor s α := Tensor.full a

/-- `tl.where(cond, x, y)`. -/
def «where» (c : Tensor s Bool) (x y : Tensor s α) : Tensor s α := Tensor.where_ c x y

/-! ## Memory -/

/-- The address of each enabled lane, `none` for the lanes the mask disables. -/
def addrs {d : DType} (ptrs : Tensor s (Ptr d)) (mask : Option (Tensor s Bool)) : Idx s → Option Int :=
  fun i => match mask with
    | none => some (ptrs.get i).addr
    | some m => if m.get i then some (ptrs.get i).addr else none

/-- `tl.load(ptrs, mask, other)`. Disabled lanes are not accessed; they hold `other` when given
and an unspecified value otherwise (the documentation: "If `other` is `None`, the masked-out
value is undefined"). `other` here is the source's `other` converted to the pointer's dtype (the
front end casts it, `triton-3.8.0-semantic.py:1043-1045`), after a Python number has become a tensor
of its own type (`docs/SEMANTICS.md`, `tl.load`; library issue L86). `cache_modifier`, `eviction_policy` and `volatile` do not change the
result in a race-free execution, the only kind the semantics gives meaning to. -/
def load {d : DType} (ptrs : Tensor s (Ptr d)) (mask : Option (Tensor s Bool) := none)
    (other : Option (Tensor s d.Val) := none) : Prog (Tensor s d.Val) :=
  .load d s (addrs ptrs mask) fun v =>
    .ret (match mask, other with
      | some m, some o => Tensor.where_ m v o
      | _, _ => v)

/-- Pad the lanes outside the bounds (`inb i = false`) of a loaded block. -/
def pad {d : DType} (p : Padding) (inb : Idx s → Bool) (v : Tensor s d.Val) : Prog (Tensor s d.Val) :=
  .pick (fun r : Tensor s d.Val => ∀ i, if inb i then r.get i = v.get i else p.Holds (r.get i)) .ret

/-- `x.to(tl.pointer_type(d))` of an integer scalar (`inttoptr`, `semantic.py:909-910`): the
pointer whose address is the integer's bits read unsigned (LLVM zero-extends a narrower integer
to the 64-bit pointer size). Grouped GEMM loads its matrices' addresses as `int64` and casts
them this way. -/
def intToPtr {sg : Bool} {w : Nat} (d : DType) (x : IntV sg w) : Ptr d := ⟨(x.bits.toNat : Int)⟩

/-- `tl.load` of a scalar pointer. -/
def load1 {d : DType} (p : Ptr d) : Prog d.Val :=
  .load d [] (fun _ => some p.addr) fun v => .ret v.item

/-- `tl.store(ptrs, vals, mask)`. The model passes values already of the pointer's dtype; the
implicit conversion `vals.to(ptr.dtype.element_ty)` of the source is written explicitly. -/
def store {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) : Prog Unit :=
  .store d s (addrs ptrs mask) vals (.ret ())

/-- `tl.store` to a scalar pointer. -/
def store1 {d : DType} (p : Ptr d) (v : d.Val) : Prog Unit :=
  .store d [] (fun _ => some p.addr) (Tensor.scalar v) (.ret ())

/-! ## Atomics

A tensor atomic is one atomic read-modify-write per enabled lane, in an order the program does
not control, each returning the element's old value. Disabled lanes return unspecified values. -/

/-- The read-modify-writes of the lanes `order`, in that order, collecting old values. -/
def rmwSeq {d : DType} (ptrs : Tensor s (Ptr d)) (upd : Idx s → d.Val → d.Val → Prop)
    (sem : MemSem) (scope : Scope) : List (Idx s) → (Idx s → d.Val) → Prog (Idx s → d.Val)
  | [], acc => .ret acc
  | i :: rest, acc => .rmw d (ptrs.get i).addr (upd i) sem scope fun old =>
      rmwSeq ptrs upd sem scope rest (Function.update acc i old)

/-- A tensor atomic with per-lane update relation `upd`. -/
def atomicWith {d : DType} (ptrs : Tensor s (Ptr d)) (upd : Idx s → d.Val → d.Val → Prop)
    (mask : Option (Tensor s Bool)) (sem : MemSem) (scope : Scope) : Prog (Tensor s d.Val) :=
  let enabled := (Idx.all s).filter (fun i => (addrs ptrs mask i).isSome)
  .pick (fun (_ : Idx s → d.Val) => True) fun junk =>
  .pick (fun (order : List (Idx s)) => order.Perm enabled) fun order => do
    let old ← rmwSeq ptrs upd sem scope order junk
    return ⟨old⟩

/-- The new value of `atomic_add` (integers wrap; floats as the platform rounds). -/
def addUpd (e : Env) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = o + v | .i16, v, o, n => n = o + v | .i32, v, o, n => n = o + v
  | .i64, v, o, n => n = o + v
  | .u8, v, o, n => n = o + v | .u16, v, o, n => n = o + v | .u32, v, o, n => n = o + v
  | .u64, v, o, n => n = o + v
  | .f16, v, o, n => e.P.atomicAdd _ o v n | .bf16, v, o, n => e.P.atomicAdd _ o v n
  | .f32, v, o, n => e.P.atomicAdd _ o v n | .f64, v, o, n => e.P.atomicAdd _ o v n
  | _, _, _, _ => False

/-- `tl.atomic_add(ptrs, vals, mask, sem, scope)`. -/
def atomic_add {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  .env fun e => atomicWith ptrs (fun i => addUpd e d (vals.get i)) mask sem scope

/-- `tl.atomic_xchg`. -/
def atomic_xchg {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i _ n => n = vals.get i) mask sem scope

/-- `tl.atomic_cas(ptrs, cmp, vals)`: the new value is `val` when the old bit pattern equals
`cmp`'s, the old value otherwise. -/
def atomic_cas {d : DType} (ptrs : Tensor s (Ptr d)) (cmp vals : Tensor s d.Val)
    (sem : MemSem := .acqRel) (scope : Scope := .gpu) : Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i o n => n = if o = cmp.get i then vals.get i else o) none sem scope

/-- A float `atomic_max` (`isMax`) or `atomic_min` result: one of the two values and, when
neither is NaN, not below (above) either. Which zero wins between `-0.0` and `+0.0`, and the
result with a NaN operand, are left open (Triton implements float max/min with integer atomics
on the bit pattern; the documentation does not say). -/
def fBound {f : TVA.Math.Float.Format} (isMax : Bool) (o v n : FloatV f) : Prop :=
  (n = o ∨ n = v) ∧ (o.isNaN = false → v.isNaN = false →
    if isMax then o.le n = true ∧ v.le n = true else n.le o = true ∧ n.le v = true)

/-- The new value of `atomic_max` (`isMax`) or `atomic_min`: integers compare as their
signedness says, floats as `fBound`. -/
def maxUpd (isMax : Bool) : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = if isMax then o.max v else o.min v
  | .i16, v, o, n => n = if isMax then o.max v else o.min v
  | .i32, v, o, n => n = if isMax then o.max v else o.min v
  | .i64, v, o, n => n = if isMax then o.max v else o.min v
  | .u8, v, o, n => n = if isMax then o.max v else o.min v
  | .u16, v, o, n => n = if isMax then o.max v else o.min v
  | .u32, v, o, n => n = if isMax then o.max v else o.min v
  | .u64, v, o, n => n = if isMax then o.max v else o.min v
  | .f16, v, o, n => fBound isMax o v n | .bf16, v, o, n => fBound isMax o v n
  | .f32, v, o, n => fBound isMax o v n | .f64, v, o, n => fBound isMax o v n
  | _, _, _, _ => False

/-- `tl.atomic_max`. -/
def atomic_max {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => maxUpd true d (vals.get i)) mask sem scope

/-- `tl.atomic_min`. -/
def atomic_min {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => maxUpd false d (vals.get i)) mask sem scope

/-- The new value of a bitwise atomic on integers (`op` on the bit vectors). -/
def bitUpd (op : ∀ {sg : Bool} {w : Nat}, IntV sg w → IntV sg w → IntV sg w) :
    (d : DType) → d.Val → d.Val → d.Val → Prop
  | .i8, v, o, n => n = op o v | .i16, v, o, n => n = op o v | .i32, v, o, n => n = op o v
  | .i64, v, o, n => n = op o v
  | .u8, v, o, n => n = op o v | .u16, v, o, n => n = op o v | .u32, v, o, n => n = op o v
  | .u64, v, o, n => n = op o v
  | _, _, _, _ => False

/-- `tl.atomic_and`. -/
def atomic_and {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => bitUpd IntV.and d (vals.get i)) mask sem scope

/-- `tl.atomic_or`. -/
def atomic_or {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => bitUpd IntV.or d (vals.get i)) mask sem scope

/-- `tl.atomic_xor`. -/
def atomic_xor {d : DType} (ptrs : Tensor s (Ptr d)) (vals : Tensor s d.Val)
    (mask : Option (Tensor s Bool) := none) (sem : MemSem := .acqRel) (scope : Scope := .gpu) :
    Prog (Tensor s d.Val) :=
  atomicWith ptrs (fun i => bitUpd IntV.xor d (vals.get i)) mask sem scope

/-! ## Floating-point arithmetic -/

section Float
variable {f : Format}

/-- The results of an elementwise operation whose exact result is `v`: `v` up to NaN payload. -/
def upToNaN (v : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun r : Tensor s (FloatV f) => ∀ i, FloatV.NaNEq (v.get i) (r.get i)) .ret

/-- `x + y` on floats, IEEE (round to nearest even). This is the meaning of a source `+` whose
operands are not, in the compiled dataflow, results of a floating multiplication. An addition
that consumes a product may be fused (`enable_fp_fusion`), and the model must then write
`tl.mul_add` or `tl.addc`. Whether every such addition is written so is part of the model's
correspondence to the source, which the reviewer checks.

Operand dtypes are promoted first, as Triton does (`semantic.py` `computation_type_impl`): the
model writes the conversions (`tl.cvt`). -/
def add (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (Tensor.map2 FloatV.add x y)

/-- `x - y` on floats, IEEE (the same caveat as `tl.add`). -/
def sub (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (Tensor.map2 FloatV.sub x y)

/-- A source `a * b + c` (an addition consuming a product): with `enable_fp_fusion`, either
the two IEEE operations or one `fma`; without it, the two operations. -/
def mul_add (a b c : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor s (FloatV f) => ∀ i,
    FloatV.NaNEq (FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)) (r.get i) ∨
    (e.fusion = true ∧ FloatV.NaNEq (FloatV.fma (a.get i) (b.get i) (c.get i)) (r.get i))) .ret

/-- A contractible addition `x + y` whose operands may be products the compiler fuses: any
result `AddRel` allows. Use it when an operand's product is not visible to the model, or when
both are products. -/
def addc (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor s (FloatV f) => ∀ i, AddRel e.fusion (x.get i) (y.get i) (r.get i)) .ret

/-- A contractible subtraction: `x + (-y)` under `addc` (`a*b - c` and `c - a*b` both contract
to an `fma`). -/
def subc (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := addc x (y.map FloatV.neg)

/-- `x * y` on floats: the IEEE product (a following addition may fuse it: `AddRel`). -/
def mul (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (Tensor.map2 FloatV.mul x y)

/-- `tl.fma(x, y, z)`: one rounding. -/
def fma (x y z : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (Tensor.map3 FloatV.fma x y z)

/-- `x / y` on floats (`tl.fdiv` with `ieee_rounding=False`): the platform's division. Triton
computes `/` of `float16` and `bfloat16` in `float32` (`semantic.py:93-102`): the model converts
explicitly. Its accuracy is a rule: `Rules.Num.DivFull` for `float32` (`div.full.f32`, within
`2^-22·|x/y| + 2^-148` for `|x/y| ≤ 2^126`), `Rules.Num.DivRn` for `float64` (IEEE). -/
def div (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor s (FloatV f) => ∀ i, e.P.div f (x.get i) (y.get i) (r.get i)) .ret

/-- `tl.div_rn(x, y)`: IEEE division. -/
def div_rn (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (Tensor.map2 FloatV.divRn x y)

/-- `tl.sqrt_rn(x)`: IEEE square root. -/
def sqrt_rn (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (x.map FloatV.sqrtRn)

/-- An approximate elementwise function, as the platform computes it. -/
def math (fn : MathFn) (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .env fun e => .pick (fun r : Tensor s (FloatV f) => ∀ i, e.P.math fn f (x.get i) (r.get i)) .ret

/-- `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`, `tl.sin`, `tl.cos`, `tl.sqrt`, `tl.rsqrt`,
`tl.erf`. -/
def exp := @math s f .exp
def exp2 := @math s f .exp2
def log := @math s f .log
def log2 := @math s f .log2
def sin := @math s f .sin
def cos := @math s f .cos
def sqrt := @math s f .sqrt
def rsqrt := @math s f .rsqrt
def erf := @math s f .erf

/-- `-x` and `tl.abs(x)`: exact, a NaN gives some NaN. -/
def neg (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (x.map FloatV.neg)
def abs (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) := upToNaN (x.map FloatV.abs)

/-- `tl.maximum(x, y)` / `tl.minimum(x, y)` (default `propagate_nan=NONE`): `MaxRel`. Triton
converts `bfloat16` operands to `float32` first (`core.py:2927, 2949`): the model converts
explicitly, and the result is `float32`. -/
def maximum (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun r : Tensor s (FloatV f) => ∀ i, FloatV.MaxRel true (x.get i) (y.get i) (r.get i)) .ret
def minimum (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun r : Tensor s (FloatV f) => ∀ i, FloatV.MaxRel false (x.get i) (y.get i) (r.get i)) .ret

/-- `tl.maximum(x, y)` / `tl.minimum(x, y)` on integer tensors, and Python's `max(x, y)` /
`min(x, y)` in a kernel when an operand is a tensor (`core.py:3882-3929`, which folds
`tl.maximum`/`tl.minimum`): exact, signed or unsigned by the dtype (`arith.maxsi`/`maxui`,
`minsi`/`minui`, `semantic.py:366-398`; `IntV.max`, `IntV.min`). The operands have the promoted
dtype (`TVA.Lang.Promote`, `BinOp.typing?`). -/
def imaximum {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) : Tensor s (IntV sg w) :=
  Tensor.map2 IntV.max x y
def iminimum {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) : Tensor s (IntV sg w) :=
  Tensor.map2 IntV.min x y

/-- The rounding modes `x.to(dtype, fp_downcast_rounding=…)` allows from format `f` to `g`:
`"rtne"` (`.rn`, the default) always; `"rtz"` (`.rz`) only from `float32`, to `float16`,
`bfloat16` (`cvt.rz`) or `float8e5`. Other pairs do not compile: a non-`float32` source with
`rtz` into fp8 is a fatal error (`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:485-490`). -/
def RoundOk (f g : Format) (rnd : RoundMode) : Prop :=
  rnd = .rn ∨ (rnd = .rz ∧ f = .binary32 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .e5m2))

/-- `x.to(dtype, fp_downcast_rounding=rnd)` between standard float formats (`float16`,
`bfloat16`, `float32`, `float64`): one rounding (`.rn` to nearest even, the default; `.rz`
toward zero), up to NaN payload. Widening is exact. Undefined behaviour for other formats, which
use `tl.cvt8` (into fp8) and `tl.cvtFrom8` (out of fp8). -/
def cvt {g : Format} (x : Tensor s (FloatV f)) (rnd : RoundMode := .rn) : Prog (Tensor s (FloatV g)) :=
  .pick (fun _ : Unit => f.Std ∧ g.Std ∧ RoundOk f g rnd) fun _ => upToNaN (x.map (FloatV.cvtWith rnd))

/-- `x.to(dtype, fp_downcast_rounding=rnd)` from `float16`, `bfloat16` or `float32` into an fp8
format: as the platform converts (`P.cvt f g rnd`). -/
def cvt8 {g : Format} (x : Tensor s (FloatV f)) (rnd : RoundMode := .rn) : Prog (Tensor s (FloatV g)) :=
  .pick (fun _ : Unit => (f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 ∧ RoundOk f g rnd) fun _ =>
  .env fun e => .pick (fun r : Tensor s (FloatV g) => ∀ i, e.P.cvt f g rnd (x.get i) (r.get i)) .ret

/-- `x.to(dtype)` from fp8 (`float8e5`, `float8e4nv`) to `float16`, `bfloat16` or `float32`:
exact on finite elements. An infinity or NaN may come out as any value: Triton converts
`float8e5` to `bfloat16` by a bit trick that turns them into finite numbers (`Fp8E5M2_to_Bf16`,
`triton-3.8.0-nvidia-ElementwiseOpToLLVM.cpp:68-121, 433-434`). There is no fp8 → `float64`
conversion (`:421-457`). -/
def cvtFrom8 {g : Format} (x : Tensor s (FloatV f)) : Prog (Tensor s (FloatV g)) :=
  .pick (fun _ : Unit => f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32)) fun _ =>
  .pick (fun r : Tensor s (FloatV g) => ∀ i, (x.get i).isFinite = true → r.get i = FloatV.cvt (x.get i)) .ret

/-- The results an integer `z` may convert to in format `g`: rounded to nearest even once, or
twice through a wider standard format. Before sm_90 there is no direct integer-to-`bfloat16`
conversion, so it goes through `float32`. And MLIR folds a narrowing with the default rounding of
an integer-to-float result into one conversion to the final format (`truncf(sitofp x) → sitofp x`,
`llvm-ArithCanonicalization.td:453-463`, with no one-use condition, for a `truncf` without a
rounding mode). That covers an explicit chain `x.to(float64).to(float32)` and also the implicit
conversions of `tl.store`, the atomics and `desc.store`. So each such narrowed value is one
`tl.itof` to the final format (`docs/SEMANTICS.md`, `x.to(float)`). A narrowing with
`fp_downcast_rounding="rtz"` is `tt.fp_to_fp`, which no fold touches: the model writes `tl.itof` to
the intermediate format, then `tl.cvt … .rz` (library issue L24). -/
def ItofRel (g : Format) (z : Int) (r : FloatV g) : Prop :=
  r = FloatV.ofRat (z : Rat) ∨
  ((g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32) ∧ r = FloatV.cvt (FloatV.ofRat (z : Rat) : F64)) ∨
  ((g = .binary16 ∨ g = .bfloat16) ∧ r = FloatV.cvt (FloatV.ofRat (z : Rat) : F32))

/-- `x.to(float dtype)` of an integer tensor (`sitofp`/`uitofp`) into a standard format: each
element as `ItofRel` allows. Into fp8 it does not compile (`semantic.py:916`). -/
def itof {sg : Bool} {w : Nat} (x : Tensor s (IntV sg w)) : Prog (Tensor s (FloatV f)) :=
  .pick (fun _ : Unit => f.Std) fun _ =>
  .pick (fun r : Tensor s (FloatV f) => ∀ i, ItofRel f (x.get i).toInt (r.get i)) .ret

/-- `x.to(int dtype)` of a float tensor in a standard format: truncation toward zero;
undefined behaviour when an element is NaN, infinite, or out of the integer type's range (LLVM
`fptosi`/`fptoui`). From fp8 it does not compile (`semantic.py:916`). -/
def ftoi {sg : Bool} {w : Nat} (x : Tensor s (FloatV f)) : Prog (Tensor s (IntV sg w)) :=
  let inRange (a : FloatV f) : Prop :=
    a.isFinite = true ∧ (Format.intRange sg w).1 ≤ a.toRat.num.tdiv a.toRat.den ∧
      a.toRat.num.tdiv a.toRat.den ≤ (Format.intRange sg w).2
  .pick (fun _ : Unit => f.Std) fun _ =>
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, inRange (x.get i) ∧
    r.get i = IntV.ofInt ((x.get i).toRat.num.tdiv (x.get i).toRat.den)) .ret

/-! ## Floating-point reductions

A reduction combines its elements with rounding additions along a tree Triton chooses. As with
`tl.add` and `tl.addc`, the model says whether a product may feed the additions. `tl.sum`
(`sum1`, `sum0`) is for inputs that are not products: every node is an IEEE addition
(`AddRel false`). `tl.sumc` (`sum1c`, `sum0c`) is for a sum of products (`tl.sum(a * b)`), where
the compiler may fuse the first additions: `AddRel` with the launch's `enable_fp_fusion`. -/

/-- A whole rank-1 reduction along some tree, each node an `AddRel fusion` addition. -/
def sumWith (fusion : Bool) {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) :=
  .pick (fun r => ∃ t : Tree, t.Covers n ∧
    t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r) .ret

/-- Row reductions of a rank-2 tensor (`axis=1`). -/
def sum1With (fusion : Bool) {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  .pick (fun r : Tensor [m] (FloatV f) => ∀ i, ∃ t : Tree, t.Covers n ∧
    t.Eval (AddRel fusion) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) .ret

/-- Column reductions of a rank-2 tensor (`axis=0`). -/
def sum0With (fusion : Bool) {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  .pick (fun r : Tensor [n] (FloatV f) => ∀ j, ∃ t : Tree, t.Covers m ∧
    t.Eval (AddRel fusion) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) .ret

/-- `tl.sum(x)` of a rank-1 tensor that is not a product: IEEE additions along a tree, in the
input's own format (Triton promotes only integers narrower than 32 bits, `standard.py:270-292`). -/
def sum {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) := sumWith false x
/-- `tl.sum(x, axis=1)`, IEEE. -/
def sum1 {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) := sum1With false x
/-- `tl.sum(x, axis=0)`, IEEE. -/
def sum0 {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) := sum0With false x

/-- `tl.sum(x)` of a product: the additions may be fused. -/
def sumc {n : Nat} (x : Tensor [n] (FloatV f)) : Prog (FloatV f) := .env fun e => sumWith e.fusion x
/-- `tl.sum(x, axis=1)` of a product. -/
def sum1c {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  .env fun e => sum1With e.fusion x
/-- `tl.sum(x, axis=0)` of a product. -/
def sum0c {m n : Nat} (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  .env fun e => sum0With e.fusion x

/-- `tl.max(x, axis=1)` (`isMax`) or `tl.min(x, axis=1)` of a rank-2 float tensor: each row
combined with `MaxRel` along a tree Triton chooses. Triton first converts floats narrower than
32 bits to `float32` (`triton-3.8.0-standard.py:178-190`): the model converts explicitly.
Integer `tl.max`/`tl.min` are the exact folds of `TVA.Lang.Tensor` (`IntV.max`/`IntV.min`),
after Triton's widening of integers narrower than 32 bits to `int32`; with `return_indices` they
are `tl.argmax1`/`argmax0`/`argmaxAll`. -/
def max1 {m n : Nat} (isMax : Bool) (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [m] (FloatV f)) :=
  .pick (fun r : Tensor [m] (FloatV f) => ∀ i, ∃ t : Tree, t.Covers n ∧
    t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.get1 i)) .ret

/-- `tl.max(x)` (`isMax`) or `tl.min(x)` of a rank-1 float tensor (or `axis=0` of it): the
whole tensor combined with `MaxRel` along a tree, after the same widening as `max1` (integers:
see `max1`). -/
def maxAll {n : Nat} (isMax : Bool) (x : Tensor [n] (FloatV f)) : Prog (FloatV f) :=
  .pick (fun r => ∃ t : Tree, t.Covers n ∧
    t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r) .ret

/-- `tl.max(x, axis=0)` (`isMax`) or `tl.min(x, axis=0)` of a rank-2 float tensor: each column,
as `max1` (integers: see `max1`). -/
def max0 {m n : Nat} (isMax : Bool) (x : Tensor [m, n] (FloatV f)) : Prog (Tensor [n] (FloatV f)) :=
  .pick (fun r : Tensor [n] (FloatV f) => ∀ j, ∃ t : Tree, t.Covers m ∧
    t.Eval (FloatV.MaxRel isMax) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) .ret

end Float

end tl

/-! ## Index reductions

`tl.max(x, axis, return_indices=True)`, `tl.min(…, return_indices=True)`, `tl.argmax` and
`tl.argmin` reduce `(value, index)` pairs along an axis with Triton's combine functions
`_argmax_combine` and `_argmin_combine` (`triton-3.8.0-standard.py:146-155, 172-190, 199-208,
223-240`). The index is an `int32` `tl.arange` along the axis (`_reduce_with_indices`,
`triton-3.8.0-core.py:3093-3107`). As for the other reductions, Triton combines along a tree of
its choosing, so the model takes some tree covering the axis.

These do not widen: Triton converts only `bfloat16` to `float32` first (the model writes that
`tl.cvt`). Unlike `tl.max`/`tl.min` without `return_indices`, they keep `float16`, fp8 and
integers narrower than 32 bits in their own type. -/

/-- The comparisons Triton's index reductions use on an element type: `<` and `==` (`arith.cmpi`,
signed or unsigned as the dtype is, for integers; the ordered `arith.cmpf olt`/`oeq` for floats,
so any comparison with a NaN is false), and which elements are NaN (none for integers). -/
class CmpKey (α : Type) where
  lt : α → α → Bool
  eq : α → α → Bool
  nan : α → Bool

instance {sg : Bool} {w : Nat} : CmpKey (IntV sg w) := ⟨IntV.lt, IntV.eq, fun _ => false⟩
instance {f : Format} : CmpKey (FloatV f) := ⟨FloatV.lt, FloatV.eq, FloatV.isNaN⟩

namespace tl

variable {s : Shape} {α : Type}

/-- Triton's `_argmax_combine` (`isMax`) and `_argmin_combine` on two `(value, index)` pairs
(`triton-3.8.0-standard.py:146-155, 199-208`). With `tieLeft` (`tie_break_left=True`, the
default), equal values (`==`) keep the smaller index; without it, a tie keeps the second pair.
`a` wins when its value is strictly greater (`isMax`) or smaller than `b`'s, or on such a tie.
Indices are below `2^20`, so the `int32` comparison of indices is the comparison of naturals. -/
def argCombine [CmpKey α] (isMax tieLeft : Bool) (a b : α × Nat) : α × Nat :=
  let tie := tieLeft && CmpKey.eq a.1 b.1 && decide (a.2 < b.2)
  let win := (if isMax then CmpKey.lt b.1 a.1 else CmpKey.lt a.1 b.1) || tie
  if win then a else b

/-- What an index reduction over the elements `g 0, …, g (n-1)` may return: the pair
`(value, index)` that `argCombine` gives along some tree over the positions, each leaf `j` being
`(g j, j)`; or, when some element is NaN, a value that is one of the elements (up to NaN payload)
with an index in range, not necessarily from the same position. With a NaN in the slice, the
device's `float16` and `bfloat16` results (the latter reduced in `float32`) take the value and the
index from different positions (`evidence/argmax`, library issue L17), so the library assumes no
more than this for any float type. -/
def ArgRes [CmpKey α] (isMax tieLeft : Bool) (n : Nat) (g : Nat → α) (r : α × I32) : Prop :=
  (∃ p : α × Nat, ∃ t : Tree, t.Covers n ∧
    t.Eval (fun a b c => c = argCombine isMax tieLeft a b) (fun j => (g j, j)) p ∧
    r = (p.1, IntV.ofNat p.2)) ∨
  ((∃ j, j < n ∧ CmpKey.nan (g j) = true) ∧
    (∃ j, j < n ∧ (r.1 = g j ∨ (CmpKey.nan r.1 = true ∧ CmpKey.nan (g j) = true))) ∧
    ∃ j, j < n ∧ r.2 = IntV.ofNat j)

/-- `tl.max(x, 0, return_indices=True, return_indices_tie_break_left=tieLeft)` (`isMax`) or
`tl.min(…)` of a rank-1 tensor: a result `ArgRes` allows. `tl.argmax(x, 0, tie_break_left=tieLeft)`
(`tl.argmin`) is the index. No widening (see the section): `bfloat16` is converted to `float32`
by the model first. With `tieLeft = false`, a tie may give any of the tied indices, and with a
NaN the result depends on the tree, and the value and the index may come from different
positions. -/
def argmaxAll [CmpKey α] [Inhabited α] {n : Nat} (isMax tieLeft : Bool) (x : Tensor [n] α) :
    Prog (α × I32) :=
  .pick (fun r : α × I32 => ArgRes isMax tieLeft n
    (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r) .ret

/-- `tl.max(x, 1, return_indices=True)` (`isMax`) or `tl.min(x, 1, …)` of a rank-2 tensor: for
each row, a pair `ArgRes` allows; `tl.argmax(x, 1)` (`tl.argmin`) is the index tensor. -/
def argmax1 [CmpKey α] [Inhabited α] {m n : Nat} (isMax tieLeft : Bool) (x : Tensor [m, n] α) :
    Prog (Tensor [m] α × Tensor [m] I32) :=
  .pick (fun r : Tensor [m] α × Tensor [m] I32 => ∀ i, ArgRes isMax tieLeft n
    (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.1.get1 i, r.2.get1 i)) .ret

/-- `tl.max(x, 0, return_indices=True)` (`isMax`) or `tl.min(x, 0, …)` of a rank-2 tensor: for
each column, a pair `ArgRes` allows; `tl.argmax(x, 0)` (`tl.argmin`) is the index tensor. -/
def argmax0 [CmpKey α] [Inhabited α] {m n : Nat} (isMax tieLeft : Bool) (x : Tensor [m, n] α) :
    Prog (Tensor [n] α × Tensor [n] I32) :=
  .pick (fun r : Tensor [n] α × Tensor [n] I32 => ∀ j, ArgRes isMax tieLeft m
    (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.1.get1 j, r.2.get1 j)) .ret

/-! ## Matrix multiplication -/

/-- `tl.dot(a, b, acc)` with configuration `cfg` (operand dtypes, `out_dtype`,
`input_precision`): as the platform computes it. -/
def dot (cfg : DotCfg) {M N K : Nat} (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (acc : Tensor [M, N] cfg.acc.Val) : Prog (Tensor [M, N] cfg.acc.Val) :=
  .env fun e => .pick (e.P.dot cfg a b acc) .ret

/-- A source `tl.dot(a, b, acc=z) + c` (or `c + tl.dot(…)`) with a zero accumulator `z` (the
default, or any constant `±0`) and no other use of the product. Triton's combine pass may
rewrite it to `tl.dot(a, b, c)` whatever `enable_fp_fusion` says (`CombineDotAddFPattern`,
`CombineDotAddIPattern`, `Combine.cpp:255-298`; for float `tl.dot` only when
`max_num_imprecise_acc` is `0`). Otherwise it is the product followed by the (fusable)
addition. -/
def dot_add (cfg : DotCfg) {M N K : Nat} (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val)
    (c : Tensor [M, N] cfg.acc.Val) (z : Tensor [M, N] cfg.acc.Val := Tensor.full (cfg.acc.ofBits 0)) :
    Prog (Tensor [M, N] cfg.acc.Val) :=
  .env fun e => .pick (fun r => e.P.dot cfg a b c r ∨
    ∃ d, e.P.dot cfg a b z d ∧ ∀ i, AccAdd e.fusion cfg.acc (d.get i) (c.get i) (r.get i)) .ret

/-- The product of a float dtype: the IEEE product up to NaN payload (`False` for other
dtypes). -/
def MulRel : (d : DType) → d.Val → d.Val → d.Val → Prop
  | .f16, a, b, c => FloatV.NaNEq (FloatV.mul a b) c | .bf16, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | .f32, a, b, c => FloatV.NaNEq (FloatV.mul a b) c | .f64, a, b, c => FloatV.NaNEq (FloatV.mul a b) c
  | _, _, _, _ => False

/-- A float reduction on axis 1 of the products `x[i, k] * y[k, j]`, with `x` expanded on its last
axis and `y` on its first: `tl.sum(x[:, :, None] * y[None, :, :], axis=1)`, and every other
spelling of that computation (`tl.expand_dims`, `tl.broadcast_to`, a `tl.reduce` whose combine
returns the sum of its two arguments; library issue L26). A `tl.reduce` whose combine is another
single addition (`a + a`, `b + b`) is not this computation, though the pass rewrites it too; the
library gives it no meaning yet. Triton's combine pass matches the IR, not the
spelling: it rewrites such a reduction to `tl.dot(x, y)` with `input_precision="ieee"` when the
additions are at most 32 bits wide and `M, N ≥ 16` (`CombineBroadcastMulReducePattern`,
`Combine.cpp:112-200`), whatever `enable_fp_fusion` says. Otherwise each element is the products
summed along a tree, with fusable additions (as `tl.sumc`). Both are allowed here, so the
definition covers a spelling whether or not the pass rewrites it. -/
def sum_bcast_mul (d : DType) {M K N : Nat} (x : Tensor [M, K] d.Val) (y : Tensor [K, N] d.Val) :
    Prog (Tensor [M, N] d.Val) :=
  .env fun e => .pick (fun r : Tensor [M, N] d.Val =>
    (∀ i j, ∃ p : Nat → d.Val,
      (∀ k (hk : k < K), MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
      ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd e.fusion d) p (r.get2 i j)) ∨
    (d.bytes ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
      e.P.dot { a := d, b := d, acc := d, precision := .ieee } x y (Tensor.full (d.ofBits 0)) r)) .ret

/-! ## Integer division and shifts

`//`, `%`, `<<` and `>>` on integer tensors. Each lane is undefined when it divides by zero,
overflows (`INT_MIN // -1`) or shifts by the width or more. This holds even for a lane that a
later `tl.where` discards, because LLVM's `sdiv` by zero is immediate undefined behaviour. Tensor
`//` and `%` truncate toward zero (C semantics). `constexpr` arithmetic is Python's, with floor
division, but it is computed in Lean by the model. -/

/-- `x // y` on integer tensors. -/
def idiv {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, (x.get i).div? (y.get i) = some (r.get i)) .ret

/-- `x % y` on integer tensors. -/
def irem {sg : Bool} {w : Nat} (x y : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, (x.get i).rem? (y.get i) = some (r.get i)) .ret

/-- `x << n`. Both operands have the promoted type (`semantic.py:421-425`). Undefined when an
amount, read as unsigned, is at least the width: LLVM gives poison, and a negative amount is a
huge one. -/
def shl {sg : Bool} {w : Nat} (x n : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, (x.get i).shl? (n.get i).bits.toNat = some (r.get i)) .ret

/-- `x >> n`, arithmetic. Triton picks `ashr` or `lshr` by the source dtype of the tensor
operand Python dispatches on, before promotion, and then shifts the promoted values. That
operand is the left one when it is a tensor (`__rshift__`, `core.py:1012-1015`), and the right
one when the left is a Python scalar (`__rrshift__`, `core.py:1018-1024`). So `int32 >> uint32`
is an arithmetic shift of `uint32` values, and `0xF0 >> s` with `s : uint8` is a logical one.
The model chooses `ashr`/`lshr` that way, not from the promoted type. -/
def ashr {sg : Bool} {w : Nat} (x n : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, (x.get i).ashr? (n.get i).bits.toNat = some (r.get i)) .ret

/-- `x >> n`, logical: the left operand's source type is unsigned (see `ashr`). -/
def lshr {sg : Bool} {w : Nat} (x n : Tensor s (IntV sg w)) : Prog (Tensor s (IntV sg w)) :=
  .pick (fun r : Tensor s (IntV sg w) => ∀ i, (x.get i).lshr? (n.get i).bits.toNat = some (r.get i)) .ret

/-! ## Compiler hints

A hint is a promise to the compiler, which uses it to vectorise and align accesses and to
compute a value once per group of equal ones. If the promise is false the behaviour is
undefined, so the model keeps each hint as a check (`core.py:3307-3382`,
`semantic.py:1775-1791`). The hint is an attribute of the *value* (`set_attr` on its defining
operation, `ir.cc:411-425`), and the compiler uses it at every use of that value, including uses
before the call or on paths that never reach it. So the model places the check where the hinted
value is defined, before any use, even when the source writes the hint later or inside a branch
(`docs/SEMANTICS.md`, "Compiler hints").

**Which value a hint marks** (library issues L1 and L60, `evidence/hint-loops`,
`evidence/hint-multi`). A hint marks the operation that defined its argument's value when the
front end visits the call. A value no operation defines takes no mark: a kernel's or a helper's
parameter, a loop's induction variable, and the value a loop carries inside its body (block
arguments; the attribute the front end leaves on the loop is read by no pass). Before it emits a
`for` or `while` loop, the front end visits the body once with every name bound to its value before
the loop and the induction variable bound to a placeholder no code uses. So:
- a hint on a value the body defines checks each iteration's value;
- a hint on a name the loop carries, where it still holds the carried value, checks the value
  before the loop, once, when an operation defined that value, and promises nothing about the
  values of later iterations. With nested loops that all carry the name, that is the value before
  the outermost of them; a value an enclosing loop's body defines is checked on each iteration of
  that body;
- a hint on a parameter or an induction variable has no effect, and the model drops it;
- a hint on the value a call of a `@triton.jit` function returns, when Triton inlines the call,
  marks the call, and inlining removes it: the hint has no effect, and the model drops it
  (`evidence/hint-multi` M12, M13). Every call is inlined but that of a helper declared
  `@triton.jit(noinline=True)`; the calls include the `tl.*` functions Triton writes in Triton
  (`triton-3.8.0-standard.py`): `tl.argmax`, `argmin`, `bitonic_merge`, `cdiv`, `cumprod`,
  `cumsum`, `flip`, `interleave`, `max`, `min`, `ravel`, `reduce_or`, `sigmoid`, `softmax`, `sort`,
  `squeeze`, `sum`, `swizzle2d`, `topk`, `unsqueeze`, `xor_sum`, `zeros`, `zeros_like`, and their
  method forms;
- **several results.** When the operation that defined the value has several results, the hint
  may apply to every one of them: the compiler reads it off the operation for each result
  (`AxisInfo.cpp:1239-1242, 1318-1342`), and a true hint on one result made the loads through
  another vectorise, and they faulted (`evidence/hint-multi` M4-M6, M14). The operations are
  `tl.split` (`a, b = tl.split(z)`), `tl.reduce` and `tl.associative_scan` on a tuple, the call of
  a helper declared `noinline=True` that returns a tuple, and the `for` or `while` loop or runtime
  `if` whose results are the names it carries or merges (a hint after it on one of them). Such a
  hint promises the same about each result of that operation, and the model checks each, where the
  operation defines them. The library states that promise when every result has the hinted value's
  shape and, for `multiple_of` and `max_contiguous`, is an integer or pointer tensor (`HintVal`;
  `ConstAlong` holds of any element type). A hint on one result of an operation with a result of
  another shape, or with a float result under `multiple_of` or `max_contiguous`, is excluded
  (`docs/exclusions.md`; library issue L86): the readings are not defined for that result. A hint
  after a loop or a runtime `if` marks the loop or the `if`, whose mark the compiler does not read
  (`evidence/hint-multi` M7-M10), and in some other cases the mark is lost (M1): checking each result
  is then stronger than the device needs, and sound.

**What a hint promises** (library issues L2 and L58, `evidence/hint-rank2`, `evidence/hint-contig`),
along each dimension `k` of the hinted tensor `x`, for the source's per-dimension values (one value
for rank 1). A hint replaces one part of what the compiler's analysis believes of the value
(`AxisInfo.cpp:1244-1259`): `max_contiguous` the contiguity, `multiple_of` the divisibility of the
first element of each group, `max_constancy` the constancy.
- `tl.max_contiguous(x, C)` alone: `C` is `x`'s contiguity along `k` (`ContigIs`): groups of `C`
  positions hold consecutive values, and groups of `2·C` do not. "Groups of `C`" alone is not
  sound: with a smaller `C` than `x`'s own, the compiler keeps the divisibility its analysis
  inferred for its own, longer groups and applies it to the hinted ones (`evidence/hint-rank2` C1:
  a true "groups of 1" faults).
- `tl.max_constancy(x, C)`: groups of `C` positions hold equal values (`ConstAlong`).
- `tl.multiple_of(x, v)` alone: `v` divides the first element of each group of `C` positions
  (`FirstsDiv x k C v`), where the compiler pairs `v` with the contiguity `C` its analysis infers
  for `x`. The model may take for `C` any lower bound the library states for that contiguity
  (below), and `1` otherwise, which makes it every element (`AxisInfo.h:31-40, 92-101`). A smaller
  `C` is a stronger promise, so a lower bound is sound.
- `tl.max_contiguous(tl.multiple_of(x, v), C)` (both on the same value, in either order): the pair
  replaces the compiler's contiguity and divisibility, and promises `ContigAlong x k C` and
  `FirstsDiv x k C v` (`tl.max_contiguous_multiple_of`), `C` below `x`'s own contiguity included
  (`evidence/hint-contig` H17-H19).

**The contiguity the analysis infers** (library issue L58, `AxisInfo.cpp:138-147, 169-183,
242-279, 553-575, 607-668, 1232-1259`, `evidence/hint-contig` H1-H16, `evidence/hint-hinted-bound`
G1-G6, `evidence/hint-unread-round17` N0-N10), a lower bound per dimension `k`, for the value the
hinted operation defines:
- a value that carries a `max_contiguous` hint, alone or in the pair (its defining operation is
  marked, as above): the smaller of the hinted `C` along `k` (with two such hints, the smallest) and
  the bound the items below give it as if it carried no hint. When the compiler reads the mark, the
  hint replaces what the analysis infers and the contiguity is `C`; but it reads no mark on a loop's
  or a runtime `if`'s result, inlining removes the mark from a call's result, an operation with
  several results keeps it in some cases only, and the canonicalizer drops it with an operation it
  folds away (`r + 0`, `r * 1`), and then the analysis keeps its own contiguity, which the items
  below bound. Values built from the hinted value start from that bound (after
  `a = tl.max_contiguous(tl.multiple_of(tl.arange(0, 512), 4), 4)`, `a + s` has the bound `4`,
  not `512`: `evidence/hint-hinted-bound` G2, G5, G6; after the pair `16` on a loaded `r`, `r + s`
  has the bound `1`: `evidence/hint-unread-round17` N2-N5, N9, N10, where the loads were as wide as
  without the hint);
- for a value that carries none:
  - `tl.arange(a, b)`: `b - a`;
  - `x + y`, `y + x` and `x - y`, where `x` has the lower bound `C` along `k` (by this list) and
    `y` is constant along `k` (a scalar, a Python number, a `tl.constexpr`, or a tensor of size 1
    along `k` that the operation broadcasts): `C`; with pointers, `p + x` and `x + p` likewise (`C`
    in elements);
  - `x[:, None]`, `x[None, :]` and `tl.expand_dims`: the kept dimensions keep their bound, and the
    new one has `1`; a broadcast (implicit, or `tl.broadcast_to`) keeps the bound of each dimension
    it does not broadcast;
  - an integer conversion `x.to(…)` between integer dtypes: `x`'s bound.

Anything else has the bound `1`; a `multiple_of` alone on a sub-expression does not change its bound
(`evidence/hint-hinted-bound` G4). The library gives these readings for values that are powers of
two, at most the dimension's size for `max_contiguous` and `max_constancy`; other values are
excluded (`docs/exclusions.md`). A hint applies to an integer tensor or a pointer tensor
(`HintVal`: a pointer's divisibility is in bytes, its contiguity in elements). The 1-D forms take a
single value. The forms for any rank end in `N` and take a list, one entry per dimension, as the
source does. -/

/-- `tl.assume(cond)`. -/
def assume (c : Bool) : Prog Unit := .pick (fun _ : Unit => c = true) .ret

/-- `tl.max_contiguous(x, C)` (1-D), with no `tl.multiple_of` on the same value: `C` is `x`'s
contiguity (`ContigIs`): `x` is `n / C` runs of `C` consecutive values, and not `n / (2·C)` runs of
`2·C` (see the section). With a `tl.multiple_of` on the same value, the pair is
`tl.max_contiguous_multiple_of`. -/
def max_contiguous {n : Nat} {β : Type} [HintVal β] (x : Tensor [n] β) (C : Nat) :
    Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ ContigIs x 0 C) .ret

/-- `tl.max_constancy(x, C)` (1-D): `x` is `n / C` runs of `C` equal values. -/
def max_constancy {n : Nat} {β : Type} (x : Tensor [n] β) (C : Nat) : Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ ConstAlong x 0 C) .ret

/-- `tl.multiple_of(x, v)` (1-D), with no `tl.max_contiguous` on the same value: `v` divides the
first element of each group of `contig` positions of `x` (`FirstsDiv`). `contig` is a lower bound
the library states for the contiguity Triton's analysis infers for `x` ("The contiguity the analysis
infers", in the section: `B` for `s + tl.arange(0, B)`), and `1` (the default: every element is a
multiple of `v`) otherwise. With a `tl.max_contiguous` on the same value, the pair is
`tl.max_contiguous_multiple_of`. -/
def multiple_of {n : Nat} {β : Type} [HintVal β] (x : Tensor [n] β) (v : Nat)
    (contig : Nat := 1) : Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ FirstsDiv x 0 contig v) .ret

/-- `tl.max_contiguous(tl.multiple_of(x, v), C)` (1-D; both hints on the same value, in either
order): the pair replaces what the compiler believes of `x`'s contiguity and divisibility, so it
promises that groups of `C` positions hold consecutive values (`ContigAlong`) and that `v` divides the
first element of each group (`FirstsDiv`), also with `C` below `x`'s own contiguity
(`evidence/hint-contig` H17-H19; library issue L58). -/
def max_contiguous_multiple_of {n : Nat} {β : Type} [HintVal β] (x : Tensor [n] β) (C v : Nat) :
    Prog (Tensor [n] β) :=
  .pick (fun y => y = x ∧ ContigAlong x 0 C ∧ FirstsDiv x 0 C v) .ret

/-- `tl.multiple_of(x, v)` of a scalar. -/
def multiple_of1 {β : Type} [HintVal β] (x : β) (v : Nat) : Prog β :=
  .pick (fun y => y = x ∧ (v : Int) ∣ HintVal.val x) .ret

/-- `tl.max_contiguous(x, Cs)` for any rank, with no `tl.multiple_of` on the same value: along each
dimension `k`, `Cs[k]` is `x`'s contiguity (`ContigIs`). -/
def max_contiguousN {β : Type} [HintVal β] (x : Tensor s β) (Cs : List Nat) : Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length → ContigIs x k (Cs.getD k 1)) .ret

/-- `tl.max_constancy(x, Cs)` for any rank: along each dimension `k`, every group of `Cs[k]`
positions holds equal values. -/
def max_constancyN {β : Type} (x : Tensor s β) (Cs : List Nat) : Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length → ConstAlong x k (Cs.getD k 1)) .ret

/-- `tl.multiple_of(x, vs)` for any rank, with no `tl.max_contiguous` on the same value: along each
dimension `k`, `vs[k]` divides the first element of each group of `contig[k]` positions
(`FirstsDiv`). `contig[k]` is a lower bound the library states for the contiguity Triton's analysis
infers for `x` along `k` (see the section), and `1` where it states none (the default `[]`: `1`
along every dimension, so every element is a multiple of each `vs[k]`). Tutorial 08's
`tl.multiple_of(a_ptrs, [16, 16])` on a block of row pointers is therefore a promise that every
pointer of the block is 16-byte aligned (dimension 0 has contiguity 1, so its value applies to every
element), which no launch meets (`evidence/hint-rank2` R1, R2). -/
def multiple_ofN {β : Type} [HintVal β] (x : Tensor s β) (vs : List Nat) (contig : List Nat := []) :
    Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length → FirstsDiv x k (contig.getD k 1) (vs.getD k 1)) .ret

/-- `tl.max_contiguous(tl.multiple_of(x, vs), Cs)` for any rank (both hints on the same value): along
each dimension `k`, groups of `Cs[k]` positions hold consecutive values and `vs[k]` divides the first
element of each group (see `max_contiguous_multiple_of`). -/
def max_contiguous_multiple_ofN {β : Type} [HintVal β] (x : Tensor s β) (Cs vs : List Nat) :
    Prog (Tensor s β) :=
  .pick (fun y => y = x ∧ ∀ k, k < s.length →
    ContigAlong x k (Cs.getD k 1) ∧ FirstsDiv x k (Cs.getD k 1) (vs.getD k 1)) .ret

/-! ## Shape changes with reordering -/

/-- `tl.reshape(x, s', can_reorder=True)` (and the deprecated `tl.view`): the same elements, in
an order the compiler chooses. `Tensor.reshape` is the row-major `can_reorder=False` form. -/
def reshape_any (x : Tensor s α) (s' : Shape) : Prog (Tensor s' α) :=
  .pick (fun r : Tensor s' α => ∃ e : Idx s' ≃ Idx s, ∀ i, r.get i = x.get (e i)) .ret

/-! ## Loops -/

/-- Iterate `body` for `t = 0, …, n-1`. -/
def forN {σ : Type} : Nat → (Nat → σ → Prog σ) → σ → Prog σ
  | 0, _, init => .ret init
  | n + 1, body, init => (forN n body init).bind (body n)

/-- The number of iterations of `range(lo, hi, step)` for `step > 0`. -/
def tripCount (lo hi step : Int) : Nat := ((hi - lo + step - 1) / step).toNat

/-- The bounds `forRange` covers: a positive step, and `lo`, `step` and `hi + step − 1` within
the *signed* range of the width. The last increment must not overflow. `scf.for` compares
signed even for an unsigned induction variable (`code_generator.py:1267-1288` passes no
signedness), so unsigned bounds at or above `2^(w−1)` would run a different number of
times. -/
def RangeOk {sg : Bool} {w : Nat} (lo hi step : IntV sg w) : Prop :=
  0 < step.toInt ∧ hi.toInt + step.toInt - 1 ≤ (Format.intRange true w).2 ∧
    lo.toInt ≤ (Format.intRange true w).2 ∧ step.toInt ≤ (Format.intRange true w).2

instance {sg : Bool} {w : Nat} (lo hi step : IntV sg w) : Decidable (RangeOk lo hi step) := by
  unfold RangeOk; infer_instance

/-- `for i in range(lo, hi, step)` (and `tl.range`) with a loop-carried state. The induction
variable has the integer type Triton gives it, the promotion of the three bounds
(`code_generator.py:1264-1265`): `int32` usually, `int64` when a bound is. It takes the values
`lo, lo + step, …` below `hi`. Bounds outside `RangeOk` are undefined behaviour here. A
`tl.static_range` is unrolled by the frontend, with a Python integer as the variable
(`code_generator.py:1208-1216`): the model writes the unrolled body. -/
def forRange {σ : Type} {sg : Bool} {w : Nat} (lo hi step : IntV sg w) (init : σ)
    (body : IntV sg w → σ → Prog σ) : Prog σ :=
  if RangeOk lo hi step then
    forN (tripCount lo.toInt hi.toInt step.toInt)
      (fun t => body (IntV.ofInt (lo.toInt + t * step.toInt))) init
  else .ub

/-- `while cond(s): s = body(s)`. -/
def whileLoop {σ : Type} (init : σ) (cond : σ → Prog Bool) (body : σ → Prog σ) : Prog σ :=
  .loop init (fun st => (cond st).bind fun c =>
    if c then (body st).bind (fun st' => .ret (.inl st')) else .ret (.inr st)) .ret

end tl

end TVA
