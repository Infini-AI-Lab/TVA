import TVA.Core
import TVA.Triton.Arith.Rel
import TVA.Triton.Convert.Rel

/-!
# Triton's sites: what a Triton program may leave open

The semantics (`TVA.Launch`) is written for any signature `Sig` (`TVA.Program`): the kinds of
value a program cannot predict, and what each allows. This module gives Triton's: the `Sig`
instance that every `tl.*` operation, every model and every claim uses.

* `Site`: where a `tl.*` operation leaves a value to the device or the compiler. `Site.other P`
  is any value of type `β` satisfying `P`.
* `Upd d`: the update of one element of dtype `d` by an atomic. `Upd.other upd` is any new value
  `upd old` allows.

`Prog.plat` is an operation the platform defines (`PlatQ`: `tl.exp`, `/`, `tl.dot`, a conversion
into fp8): read the environment, then any result its relation allows on the platform.
-/
namespace TVA

/-- A site: a value a Triton program cannot predict, and what it may be. -/
inductive Site : Type 1 where
  /-- A guard: no value but `()`, allowed when `c` holds; undefined behaviour otherwise. The
  checks Triton or the lowering makes and the promises a kernel makes (`tl.assume`,
  `tl.static_assert`, the format checks of a conversion, a block pointer's or a descriptor's
  validity, `tl.dot_scaled`'s shapes and types). -/
  | guard (c : Prop) [dec : Decidable c]
  /-- A compiler hint on `x`: the value is `x` itself, allowed when the promise `c` holds
  (`tl.max_contiguous`, `tl.multiple_of`); undefined behaviour otherwise. -/
  | hint (β : Type) (x : β) (c : Prop) [dec : Decidable c]
  /-- A partial lanewise computation: the tensor whose lane `i` is `g i`; undefined behaviour
  when some lane has no value (`//` and `%` by zero or overflowing, a shift by at least the
  width). -/
  | lanes (s : Shape) (α : Type) (g : Idx s → Option α)
  /-- An exact float result `v`, up to NaN payload: every lane of `v` that is not NaN, and some
  NaN where `v` has one (`tl.add`, `tl.mul`, `tl.fma`, `tl.div_rn`, `tl.sqrt_rn`, negation,
  `tl.abs`, a conversion between standard formats). -/
  | upToNaN (s : Shape) (f : Format) (v : Tensor s (FloatV f))
  /-- An addition `x + y` that a product may feed (`AddRel`, with `fusion` the launch's
  `enable_fp_fusion`). -/
  | addc (s : Shape) (f : Format) (fusion : Bool) (x y : Tensor s (FloatV f))
  /-- A source `a * b + c`: the two IEEE operations, or with `fusion` one `fma`, up to NaN
  payload. -/
  | mulAdd (s : Shape) (f : Format) (fusion : Bool) (a b c : Tensor s (FloatV f))
  /-- `tl.maximum` (`isMax`) or `tl.minimum` of float tensors (`FloatV.MaxRel`). -/
  | maxMin (s : Shape) (f : Format) (isMax : Bool) (x y : Tensor s (FloatV f))
  /-- `x.to(float dtype)` of an integer tensor into format `f` (`tl.ItofRel`, lane by lane). -/
  | itof (s : Shape) (f : Format) (sg : Bool) (w : Nat) (x : Tensor s (IntV sg w))
  /-- `x.to(int dtype)` of a float tensor (`FtoiRel`, lane by lane): no result when a lane is NaN,
  infinite or out of range. -/
  | ftoi (s : Shape) (f : Format) (sg : Bool) (w : Nat) (x : Tensor s (FloatV f))
  /-- `x.to(dtype)` from fp8 into format `g`: exact on the finite lanes, any value on the others. -/
  | fromF8 (s : Shape) (f g : Format) (x : Tensor s (FloatV f))
  /-- `semantic.cast` from dtype `d` to `d'` on platform `P` (`ConvRel`, lane by lane). -/
  | convert (P : Platform) (s : Shape) (d d' : DType) (x : Tensor s d.Val)
  /-- Any value of type `β` satisfying `P`. -/
  | other (β : Type) (P : β → Prop)

/-- The type of a site's value. -/
@[reducible] def Site.Out : Site → Type
  | @guard _ _ => Unit
  | @hint β _ _ _ => β
  | .lanes s α _ => Tensor s α
  | .itof s f _ _ _ => Tensor s (FloatV f)
  | .ftoi s _ sg w _ => Tensor s (IntV sg w)
  | .fromF8 s _ g _ => Tensor s (FloatV g)
  | .convert _ s _ d' _ => Tensor s d'.Val
  | .upToNaN s f _ | .addc s f _ _ _ | .mulAdd s f _ _ _ _ | .maxMin s f _ _ _ => Tensor s (FloatV f)
  | .other β _ => β

/-- The values a site allows. -/
@[reducible] def Site.Allows : (s : Site) → s.Out → Prop
  | @guard c _ => fun _ => c
  | @hint _ x c _ => fun y => y = x ∧ c
  | .lanes _ _ g => fun r => ∀ i, g i = some (r.get i)
  | .upToNaN _ _ v => fun r => ∀ i, FloatV.NaNEq (v.get i) (r.get i)
  | .addc _ _ fusion x y => fun r => ∀ i, AddRel fusion (x.get i) (y.get i) (r.get i)
  | .mulAdd _ _ fusion a b c => fun r => ∀ i,
      FloatV.NaNEq (FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)) (r.get i) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (a.get i) (b.get i) (c.get i)) (r.get i))
  | .maxMin _ _ isMax x y => fun r => ∀ i, FloatV.MaxRel isMax (x.get i) (y.get i) (r.get i)
  | .itof _ f _ _ x => fun r => ∀ i, tl.ItofRel f (x.get i).toInt (r.get i)
  | .ftoi _ _ sg w x => fun r => ∀ i, FtoiRel sg w (x.get i) (r.get i)
  | .fromF8 _ _ _ x => fun r => ∀ i, (x.get i).isFinite = true → r.get i = FloatV.cvt (x.get i)
  | .convert P _ d d' x => fun r => ∀ i, ConvRel P d d' (x.get i) (r.get i)
  | .other _ P => P

/-- The update an atomic makes to one element of dtype `d`. -/
inductive Upd (d : DType) : Type 1 where
  /-- Any new value `upd old` allows. -/
  | other (upd : d.Val → d.Val → Prop)

/-- The new values an update allows, given the old one. -/
@[reducible] def Upd.Allows {d : DType} : Upd d → d.Val → d.Val → Prop
  | .other upd => upd

/-- **Triton's signature**: its sites and its atomics' updates. -/
instance Site.sig : Sig where
  Site := Site
  Out := Site.Out
  Allows := Site.Allows
  Upd := Upd
  UpdAllows := Upd.Allows

@[simp] theorem Site.allows_other {β : Type} {P : β → Prop} {b : β} :
    Sig.Allows (Site.other β P) b ↔ P b := Iff.rfl

/-- A guard allows `()` exactly when its condition holds: the predicate `fun _ : Unit => c` of
the `Site.other` it replaces. -/
@[simp] theorem Site.allows_guard {c : Prop} [Decidable c] {u : Unit} :
    Sig.Allows (Site.guard c) u ↔ c := Iff.rfl

/-- A hint allows `x` exactly when its promise holds: the predicate `fun y => y = x ∧ c` of the
`Site.other` it replaces. -/
@[simp] theorem Site.allows_hint {β : Type} {x : β} {c : Prop} [Decidable c] {y : β} :
    Sig.Allows (Site.hint β x c) y ↔ y = x ∧ c := Iff.rfl

/-- A lanewise site allows exactly the tensors of `g`'s values: the predicate
`fun r => ∀ i, g i = some (r.get i)` of the `Site.other` it replaces. -/
@[simp] theorem Site.allows_lanes {s : Shape} {α : Type} {g : Idx s → Option α} {r : Tensor s α} :
    Sig.Allows (Site.lanes s α g) r ↔ ∀ i, g i = some (r.get i) := Iff.rfl

/-- The float sites allow exactly the predicates of the `Site.other`s they replace. -/
@[simp] theorem Site.allows_upToNaN {s : Shape} {f : Format} {v r : Tensor s (FloatV f)} :
    Sig.Allows (Site.upToNaN s f v) r ↔ ∀ i, FloatV.NaNEq (v.get i) (r.get i) := Iff.rfl

@[simp] theorem Site.allows_addc {s : Shape} {f : Format} {fusion : Bool}
    {x y r : Tensor s (FloatV f)} :
    Sig.Allows (Site.addc s f fusion x y) r ↔ ∀ i, AddRel fusion (x.get i) (y.get i) (r.get i) :=
  Iff.rfl

@[simp] theorem Site.allows_mulAdd {s : Shape} {f : Format} {fusion : Bool}
    {a b c r : Tensor s (FloatV f)} :
    Sig.Allows (Site.mulAdd s f fusion a b c) r ↔ ∀ i,
      FloatV.NaNEq (FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)) (r.get i) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (a.get i) (b.get i) (c.get i)) (r.get i)) :=
  Iff.rfl

@[simp] theorem Site.allows_maxMin {s : Shape} {f : Format} {isMax : Bool}
    {x y r : Tensor s (FloatV f)} :
    Sig.Allows (Site.maxMin s f isMax x y) r ↔
      ∀ i, FloatV.MaxRel isMax (x.get i) (y.get i) (r.get i) := Iff.rfl

/-- The conversion sites allow exactly the predicates of the `Site.other`s they replace. -/
@[simp] theorem Site.allows_itof {s : Shape} {f : Format} {sg : Bool} {w : Nat}
    {x : Tensor s (IntV sg w)} {r : Tensor s (FloatV f)} :
    Sig.Allows (Site.itof s f sg w x) r ↔ ∀ i, tl.ItofRel f (x.get i).toInt (r.get i) := Iff.rfl

@[simp] theorem Site.allows_ftoi {s : Shape} {f : Format} {sg : Bool} {w : Nat}
    {x : Tensor s (FloatV f)} {r : Tensor s (IntV sg w)} :
    Sig.Allows (Site.ftoi s f sg w x) r ↔ ∀ i, FtoiRel sg w (x.get i) (r.get i) := Iff.rfl

@[simp] theorem Site.allows_fromF8 {s : Shape} {f g : Format} {x : Tensor s (FloatV f)}
    {r : Tensor s (FloatV g)} :
    Sig.Allows (Site.fromF8 s f g x) r ↔
      ∀ i, (x.get i).isFinite = true → r.get i = FloatV.cvt (x.get i) := Iff.rfl

@[simp] theorem Site.allows_convert {P : Platform} {s : Shape} {d d' : DType}
    {x : Tensor s d.Val} {r : Tensor s d'.Val} :
    Sig.Allows (Site.convert P s d d' x) r ↔ ∀ i, ConvRel P d d' (x.get i) (r.get i) := Iff.rfl

@[simp] theorem Upd.allows_other {d : DType} {upd : d.Val → d.Val → Prop} {old new : d.Val} :
    Sig.UpdAllows (Upd.other upd) old new ↔ upd old new := Iff.rfl

namespace Prog

/-- An operation the platform defines, on concrete operands (`PlatQ`: `tl.exp`, `/`, `tl.dot`, a
conversion into fp8): read the environment, then any result `q.Rel` allows on its platform;
undefined behaviour when there is none. -/
def plat {α : Type} (q : PlatQ) (k : q.Out → Prog α) : Prog α :=
  env fun e => pick (Site.other _ (q.Rel e.P)) k

@[simp] theorem plat_bind {α β : Type} (q : PlatQ) (k : q.Out → Prog α) (f : α → Prog β) :
    (plat q k).bind f = plat q (fun r => (k r).bind f) := rfl

end Prog

end TVA
