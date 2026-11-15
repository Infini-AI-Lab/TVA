import TVA

/-!
# The catalog preserves meaning

The catalog replaced every `Site.other P` of an operation by a constructor
of a semantic family (`TVA.Site`), and every atomic's relation by an `Upd` constructor. The
semantics sees a site only through `Sig.Allows` (and an update through `Sig.UpdAllows`), so an
operation's meaning is unchanged when the new site allows exactly what the old predicate did: the
same results, and no result (undefined behaviour) in the same cases.

Below, for each family, the old predicate as it was written in the operation (copied verbatim
but for the bound names the operation gave its operands) is equivalent to the new
`Allows`. Each is `Iff.rfl`: the new relation is the old one by definition. The families of
guards, hints, lanewise computations, float elementwise operations, conversions and platform
requests have the same theorems as simp lemmas next to the catalog (`Site.allows_guard`, …,
`Site.allows_dotScaledAdd`).

That the data of each new site depends on the same operands as the old predicate matters for
`UsesSharedAtomicResult` and `Step.defer`, which compare continuations with `=`; it holds by
inspection of each operation (the site's arguments are the operands its old predicate mentioned,
or numbers computed from them, such as `tl.sumBound e.P n`).
-/
namespace TVA.Validation.Preserve

open TVA

variable {f : Format}

/-! ## Reductions (`TVA.Triton.Reduce`) -/

theorem sumWith {fusion : Bool} {n : Nat} (P : Platform) (x : Tensor [n] (FloatV f)) (r : FloatV f) :
    Sig.Allows (Site.fsum f fusion (tl.sumBound P n) n x) r ↔
      ∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound P n ∧
        t.Eval (AddRel fusion) (fun i => if h : i < n then x.get1 ⟨i, h⟩ else default) r :=
  Iff.rfl

theorem sum1With {fusion : Bool} {m n : Nat} (P : Platform) (x : Tensor [m, n] (FloatV f))
    (r : Tensor [m] (FloatV f)) :
    Sig.Allows (Site.fsum1 f fusion (tl.sumBound P n) m n x) r ↔
      ∀ i, ∃ t : Tree, t.Covers n ∧ t.depth ≤ tl.sumBound P n ∧
        t.Eval (AddRel fusion) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default)
          (r.get1 i) :=
  Iff.rfl

theorem sum0With {fusion : Bool} {m n : Nat} (P : Platform) (x : Tensor [m, n] (FloatV f))
    (r : Tensor [n] (FloatV f)) :
    Sig.Allows (Site.fsum0 f fusion (tl.sumBound P m) m n x) r ↔
      ∀ j, ∃ t : Tree, t.Covers m ∧ t.depth ≤ tl.sumBound P m ∧
        t.Eval (AddRel fusion) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default)
          (r.get1 j) :=
  Iff.rfl

theorem maxAll {isMax : Bool} {n : Nat} (x : Tensor [n] (FloatV f)) (r : FloatV f) :
    Sig.Allows (Site.fmax f isMax n x) r ↔
      ∃ t : Tree, t.Covers n ∧
        t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r :=
  Iff.rfl

theorem max1 {isMax : Bool} {m n : Nat} (x : Tensor [m, n] (FloatV f))
    (r : Tensor [m] (FloatV f)) :
    Sig.Allows (Site.fmax1 f isMax m n x) r ↔
      ∀ i, ∃ t : Tree, t.Covers n ∧
        t.Eval (FloatV.MaxRel isMax) (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default)
          (r.get1 i) :=
  Iff.rfl

theorem max0 {isMax : Bool} {m n : Nat} (x : Tensor [m, n] (FloatV f))
    (r : Tensor [n] (FloatV f)) :
    Sig.Allows (Site.fmax0 f isMax m n x) r ↔
      ∀ j, ∃ t : Tree, t.Covers m ∧
        t.Eval (FloatV.MaxRel isMax) (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default)
          (r.get1 j) :=
  Iff.rfl

variable {α : Type} [CmpKey α] [Inhabited α]

theorem argmaxAll {isMax tieLeft : Bool} {n : Nat} (x : Tensor [n] α) (r : α × I32) :
    Sig.Allows (Site.arg α isMax tieLeft n x) r ↔
      tl.ArgRes isMax tieLeft n (fun j => if h : j < n then x.get1 ⟨j, h⟩ else default) r :=
  Iff.rfl

theorem argmax1 {isMax tieLeft : Bool} {m n : Nat} (x : Tensor [m, n] α)
    (r : Tensor [m] α × Tensor [m] I32) :
    Sig.Allows (Site.arg1 α isMax tieLeft m n x) r ↔
      ∀ i, tl.ArgRes isMax tieLeft n
        (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) (r.1.get1 i, r.2.get1 i) :=
  Iff.rfl

theorem argmax0 {isMax tieLeft : Bool} {m n : Nat} (x : Tensor [m, n] α)
    (r : Tensor [n] α × Tensor [n] I32) :
    Sig.Allows (Site.arg0 α isMax tieLeft m n x) r ↔
      ∀ j, tl.ArgRes isMax tieLeft m
        (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) (r.1.get1 j, r.2.get1 j) :=
  Iff.rfl

theorem histogram {n bins : Nat} {sg : Bool} {w : Nat} (x : Tensor [n] (IntV sg w))
    (mask : Option (Tensor [n] Bool)) (r : Tensor [bins] I32) :
    Sig.Allows (Site.histogram n bins sg w x mask) r ↔
      (∀ j, tl.enabledBy mask j = true → 0 ≤ (x.get1 j).toInt ∧ (x.get1 j).toInt < bins) ∧
      ∀ b : Fin bins, r.get1 b =
        IntV.ofNat (Finset.univ.filter fun j : Fin n =>
          tl.enabledBy mask j = true ∧ (x.get1 j).toInt = b).card :=
  Iff.rfl

/-! ## `sum_bcast_mul` (`TVA.Triton.Dot`) -/

theorem sum_bcast_mul (e : Env) (d : DType) {M K N : Nat} (x : Tensor [M, K] d.Val)
    (y : Tensor [K, N] d.Val) (r : Tensor [M, N] d.Val) :
    Sig.Allows (Site.sumBcastMul e.P e.fusion d M K N x y) r ↔
      ((∀ i j, ∃ p : Nat → d.Val,
        (∀ k (hk : k < K), tl.MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
        ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd e.fusion d) p (r.get2 i j)) ∨
      (d.bytes ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
        e.P.dot { a := d, b := d, acc := d, precision := .ieee } x y (Tensor.full (d.ofBits 0)) r)) :=
  Iff.rfl

/-! ## Loads and atomics (`TVA.Triton.Memory`) -/

theorem pad {s : Shape} {d : DType} (p : Padding) (inb : Idx s → Bool) (v r : Tensor s d.Val) :
    Sig.Allows (Site.pad s d p inb v) r ↔
      ∀ i, if inb i then r.get i = v.get i else p.Holds (r.get i) :=
  Iff.rfl

theorem atomic_junk {s : Shape} {d : DType} (junk : Idx s → d.Val) :
    Sig.Allows (Site.anyVals s d) junk ↔ True :=
  Iff.rfl

theorem atomic_order {s : Shape} (enabled order : List (Idx s)) :
    Sig.Allows (Site.order s enabled) order ↔ order.Perm enabled :=
  Iff.rfl

/-- The updates: each `tl.atomic_*` passed `atomicWith` a relation per lane; it now passes an
`Upd`, which allows the same new values. -/
theorem atomic_add (e : Env) {d : DType} (v o n : d.Val) :
    Sig.UpdAllows (Upd.add e v) o n ↔ tl.addUpd e d v o n := Iff.rfl

theorem atomic_xchg {d : DType} (v o n : d.Val) :
    Sig.UpdAllows (Upd.xchg v) o n ↔ n = v := Iff.rfl

theorem atomic_cas {d : DType} (cmp v o n : d.Val) :
    Sig.UpdAllows (Upd.cas cmp v) o n ↔ n = if o = cmp then v else o := Iff.rfl

theorem atomic_max {d : DType} (isMax : Bool) (v o n : d.Val) :
    Sig.UpdAllows (Upd.max isMax v) o n ↔ tl.maxUpd isMax d v o n := Iff.rfl

theorem atomic_bit {d : DType} (op : ∀ {sg : Bool} {w : Nat}, IntV sg w → IntV sg w → IntV sg w)
    (v o n : d.Val) :
    Sig.UpdAllows (Upd.bit op v) o n ↔ tl.bitUpd op d v o n := Iff.rfl

theorem desc_atomic (e : Env) (op : RedOp) {d : DType} (v o n : d.Val) :
    Sig.UpdAllows (Upd.red e op v) o n ↔ op.Upd e d v o n := Iff.rfl

end TVA.Validation.Preserve
