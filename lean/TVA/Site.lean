import TVA.Core
import TVA.Triton.Arith.Rel
import TVA.Triton.Convert.Rel
import TVA.Triton.Dot.Rel
import TVA.Triton.Reduce.Rel
import TVA.Triton.Memory.Rel

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
  /-- An operation the platform `P` defines, on concrete operands (`PlatQ`: `tl.exp`, `/`,
  `tl.dot`, a conversion into fp8): any result `q.Rel P` allows. -/
  | plat (P : Platform) (q : PlatQ)
  /-- A source `tl.dot(a, b, acc=z) + c` on platform `P`: `tl.dot(a, b, c)`, or the product with
  accumulator `z` followed by the addition of `c` (`AccAdd`, with `fusion`). One site with both
  branches, so that it is undefined behaviour only when neither has a result. -/
  | dotAdd (P : Platform) (fusion : Bool) (cfg : DotCfg) (M N K : Nat)
      (a : Tensor [M, K] cfg.a.Val) (b : Tensor [K, N] cfg.b.Val) (c z : Tensor [M, N] cfg.acc.Val)
  /-- A source `tl.dot_scaled(…, acc=z) + c` on platform `P`, on unpacked operands and scales:
  as `dotAdd`, with `Platform.dotScaled` and the `float32` addition. -/
  | dotScaledAdd (P : Platform) (fusion : Bool) (cfg : ScaledCfg) (M N K : Nat)
      (a : Tensor [M, K] Nat) (sa : Option (Tensor [M, K] Nat)) (b : Tensor [K, N] Nat)
      (sb : Option (Tensor [K, N] Nat)) (c z : Tensor [M, N] F32)
  /-- A float sum of a rank-1 tensor along some tree over its `n` positions of depth at most
  `depth`, each node an `AddRel fusion` addition (`tl.sum`, `tl.sumc`). -/
  | fsum (f : Format) (fusion : Bool) (depth n : Nat) (x : Tensor [n] (FloatV f))
  /-- The row sums of a rank-2 tensor (`axis=1`), each along its own tree, as `fsum`. -/
  | fsum1 (f : Format) (fusion : Bool) (depth m n : Nat) (x : Tensor [m, n] (FloatV f))
  /-- The column sums of a rank-2 tensor (`axis=0`), each along its own tree, as `fsum`. -/
  | fsum0 (f : Format) (fusion : Bool) (depth m n : Nat) (x : Tensor [m, n] (FloatV f))
  /-- `tl.max(x)` (`isMax`) or `tl.min(x)` of a rank-1 float tensor: `MaxRel` along some tree. -/
  | fmax (f : Format) (isMax : Bool) (n : Nat) (x : Tensor [n] (FloatV f))
  /-- `tl.max(x, axis=1)` or `tl.min`: each row along its own tree, as `fmax`. -/
  | fmax1 (f : Format) (isMax : Bool) (m n : Nat) (x : Tensor [m, n] (FloatV f))
  /-- `tl.max(x, axis=0)` or `tl.min`: each column along its own tree, as `fmax`. -/
  | fmax0 (f : Format) (isMax : Bool) (m n : Nat) (x : Tensor [m, n] (FloatV f))
  /-- An index reduction of a rank-1 tensor (`tl.argmax`, `tl.max(…, return_indices=True)`, and
  the minima): a pair `tl.ArgRes` allows. -/
  | arg (α : Type) [cmp : CmpKey α] [inh : Inhabited α] (isMax tieLeft : Bool) (n : Nat)
      (x : Tensor [n] α)
  /-- An index reduction of each row of a rank-2 tensor (`axis=1`), as `arg`. -/
  | arg1 (α : Type) [cmp : CmpKey α] [inh : Inhabited α] (isMax tieLeft : Bool) (m n : Nat)
      (x : Tensor [m, n] α)
  /-- An index reduction of each column of a rank-2 tensor (`axis=0`), as `arg`. -/
  | arg0 (α : Type) [cmp : CmpKey α] [inh : Inhabited α] (isMax tieLeft : Bool) (m n : Nat)
      (x : Tensor [m, n] α)
  /-- `tl.histogram(x, bins, mask)`: the counts of the enabled elements per bin; undefined
  behaviour when an enabled element is outside `[0, bins)`. -/
  | histogram (n bins : Nat) (sg : Bool) (w : Nat) (x : Tensor [n] (IntV sg w))
      (mask : Option (Tensor [n] Bool))
  /-- A loaded block `v` whose lanes outside the bounds (`inb i = false`) hold what the padding
  `p` says; the others keep their loaded value. -/
  | pad (s : Shape) (d : DType) (p : Padding) (inb : Idx s → Bool) (v : Tensor s d.Val)
  /-- Any values of dtype `d`, one per lane: what the disabled lanes of an atomic return. -/
  | anyVals (s : Shape) (d : DType)
  /-- An order of the lanes `enabled`: a permutation of them (the order an atomic's lanes are
  performed in). -/
  | order (s : Shape) (enabled : List (Idx s))
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
  | .plat _ q => q.Out
  | .dotAdd _ _ cfg M N _ _ _ _ _ => Tensor [M, N] cfg.acc.Val
  | .dotScaledAdd _ _ _ M N _ _ _ _ _ _ _ => Tensor [M, N] F32
  | .fsum f _ _ _ _ | .fmax f _ _ _ => FloatV f
  | .fsum1 f _ _ m _ _ | .fmax1 f _ m _ _ => Tensor [m] (FloatV f)
  | .fsum0 f _ _ _ n _ | .fmax0 f _ _ n _ => Tensor [n] (FloatV f)
  | @arg α _ _ _ _ _ _ => α × I32
  | @arg1 α _ _ _ _ m _ _ => Tensor [m] α × Tensor [m] I32
  | @arg0 α _ _ _ _ _ n _ => Tensor [n] α × Tensor [n] I32
  | .histogram _ bins _ _ _ _ => Tensor [bins] I32
  | .pad s d _ _ _ => Tensor s d.Val
  | .anyVals s d => Idx s → d.Val
  | .order s _ => List (Idx s)
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
  | .plat P q => q.Rel P
  | .pad _ _ p inb v => fun r => ∀ i, if inb i then r.get i = v.get i else p.Holds (r.get i)
  | .anyVals _ _ => fun _ => True
  | .order _ enabled => fun order => order.Perm enabled
  | .fsum _ fusion depth n x => fun r => ∃ t : Tree, t.Covers n ∧ t.depth ≤ depth ∧
      t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r
  | .fsum1 _ fusion depth m n x => fun r : Tensor [m] _ => ∀ i, ∃ t : Tree, t.Covers n ∧
      t.depth ≤ depth ∧
      t.Eval (AddRel fusion) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.get1 i)
  | .fsum0 _ fusion depth m n x => fun r : Tensor [n] _ => ∀ j, ∃ t : Tree, t.Covers m ∧
      t.depth ≤ depth ∧
      t.Eval (AddRel fusion) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.get1 j)
  | .fmax _ isMax n x => fun r => ∃ t : Tree, t.Covers n ∧
      t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r
  | .fmax1 _ isMax m n x => fun r : Tensor [m] _ => ∀ i, ∃ t : Tree, t.Covers n ∧
      t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default)
        (r.get1 i)
  | .fmax0 _ isMax m n x => fun r : Tensor [n] _ => ∀ j, ∃ t : Tree, t.Covers m ∧
      t.Eval (FloatV.MaxRel isMax) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default)
        (r.get1 j)
  | @arg _ _ _ isMax tieLeft n x => fun r => tl.ArgRes isMax tieLeft n
      (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r
  | @arg1 _ _ _ isMax tieLeft _ n x => fun r => ∀ i, tl.ArgRes isMax tieLeft n
      (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.1.get1 i, r.2.get1 i)
  | @arg0 _ _ _ isMax tieLeft m _ x => fun r => ∀ j, tl.ArgRes isMax tieLeft m
      (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.1.get1 j, r.2.get1 j)
  | .histogram n bins _ _ x mask => fun r =>
      (∀ j, tl.enabledBy mask j = true → 0 ≤ (x.get1 j).toInt ∧ (x.get1 j).toInt < bins) ∧
      ∀ b : Fin bins, r.get1 b =
        IntV.ofNat (Finset.univ.filter fun j : Fin n =>
          tl.enabledBy mask j = true ∧ (x.get1 j).toInt = b).card
  | .dotAdd P fusion cfg _ _ _ a b c z => fun r => P.dot cfg a b c r ∨
      ∃ d, P.dot cfg a b z d ∧ ∀ i, AccAdd fusion cfg.acc (d.get i) (c.get i) (r.get i)
  | .dotScaledAdd P fusion cfg _ _ _ a sa b sb c z => fun r => P.dotScaled cfg a sa b sb c r ∨
      ∃ d, P.dotScaled cfg a sa b sb z d ∧ ∀ i, AddRel fusion (d.get i) (c.get i) (r.get i)
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

/-- The platform sites allow exactly the predicates of the `Site.other`s they replace. -/
@[simp] theorem Site.allows_plat {P : Platform} {q : PlatQ} {r : q.Out} :
    Sig.Allows (Site.plat P q) r ↔ q.Rel P r := Iff.rfl

@[simp] theorem Site.allows_dotAdd {P : Platform} {fusion : Bool} {cfg : DotCfg} {M N K : Nat}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {c z r : Tensor [M, N] cfg.acc.Val} :
    Sig.Allows (Site.dotAdd P fusion cfg M N K a b c z) r ↔ (P.dot cfg a b c r ∨
      ∃ d, P.dot cfg a b z d ∧ ∀ i, AccAdd fusion cfg.acc (d.get i) (c.get i) (r.get i)) := Iff.rfl

@[simp] theorem Site.allows_dotScaledAdd {P : Platform} {fusion : Bool} {cfg : ScaledCfg}
    {M N K : Nat} {a : Tensor [M, K] Nat} {sa : Option (Tensor [M, K] Nat)} {b : Tensor [K, N] Nat}
    {sb : Option (Tensor [K, N] Nat)} {c z r : Tensor [M, N] F32} :
    Sig.Allows (Site.dotScaledAdd P fusion cfg M N K a sa b sb c z) r ↔
      (P.dotScaled cfg a sa b sb c r ∨
        ∃ d, P.dotScaled cfg a sa b sb z d ∧ ∀ i, AddRel fusion (d.get i) (c.get i) (r.get i)) :=
  Iff.rfl

@[simp] theorem Upd.allows_other {d : DType} {upd : d.Val → d.Val → Prop} {old new : d.Val} :
    Sig.UpdAllows (Upd.other upd) old new ↔ upd old new := Iff.rfl

namespace Prog

/-- An operation the platform defines, on concrete operands (`PlatQ`: `tl.exp`, `/`, `tl.dot`, a
conversion into fp8): read the environment, then any result `q.Rel` allows on its platform;
undefined behaviour when there is none. -/
def plat {α : Type} (q : PlatQ) (k : q.Out → Prog α) : Prog α :=
  env fun e => pick (Site.plat e.P q) k

@[simp] theorem plat_bind {α β : Type} (q : PlatQ) (k : q.Out → Prog α) (f : α → Prog β) :
    (plat q k).bind f = plat q (fun r => (k r).bind f) := rfl

end Prog

end TVA
