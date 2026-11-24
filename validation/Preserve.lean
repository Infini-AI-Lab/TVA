import TVA

/-!
# The catalog preserves meaning

The semantics sees a site only through `Sig.Allows` (an update through `Sig.UpdAllows`). For each
family of `TVA.Site`, the predicate the operation used before the catalog (copied verbatim up to
bound names) is equivalent to the new `Allows`, by `Iff.rfl`: the same results, and undefined
behaviour in the same cases.

So the meaning is unchanged: a kernel has the outcomes of its erasure (`Prog.erase`, each site
replaced by its relation), is well behaved and `Correct` exactly when its erasure is, and two
kernels with the same erasure mean the same (`TVA.Launch.Erase`). `Step.defer` and
`UsesSharedAtomicResult` compare continuations after erasure too, so data a site carries but its
relation ignores is not a dependence: `cas_self` below is the regression case.
-/
namespace TVA.Validation.Preserve

open TVA

variable {f : Format}

/-! ## Guards, hints, lanewise, float, conversion and platform sites -/

theorem other {β : Type} {P : β → Prop} {b : β} :
    Sig.Allows (Site.other β P) b ↔ P b := Iff.rfl

theorem guard {c : Prop} [Decidable c] {u : Unit} :
    Sig.Allows (Site.guard c) u ↔ c := Iff.rfl

theorem hint {β : Type} {x : β} {c : Prop} [Decidable c] {y : β} :
    Sig.Allows (Site.hint β x c) y ↔ y = x ∧ c := Iff.rfl

theorem lanes {s : Shape} {α : Type} {g : Idx s → Option α} {r : Tensor s α} :
    Sig.Allows (Site.lanes s α g) r ↔ ∀ i, g i = some (r.get i) := Iff.rfl

theorem upToNaN {s : Shape} {f : Format} {v r : Tensor s (FloatV f)} :
    Sig.Allows (Site.upToNaN s f v) r ↔ ∀ i, FloatV.NaNEq (v.get i) (r.get i) := Iff.rfl

theorem addc {s : Shape} {f : Format} {fusion : Bool}
    {x y r : Tensor s (FloatV f)} :
    Sig.Allows (Site.addc s f fusion x y) r ↔ ∀ i, AddRel fusion (x.get i) (y.get i) (r.get i) :=
  Iff.rfl

theorem mulAdd {s : Shape} {f : Format} {fusion : Bool}
    {a b c r : Tensor s (FloatV f)} :
    Sig.Allows (Site.mulAdd s f fusion a b c) r ↔ ∀ i,
      FloatV.NaNEq (FloatV.add (FloatV.mul (a.get i) (b.get i)) (c.get i)) (r.get i) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (a.get i) (b.get i) (c.get i)) (r.get i)) :=
  Iff.rfl

theorem maxMin {s : Shape} {f : Format} {isMax : Bool}
    {x y r : Tensor s (FloatV f)} :
    Sig.Allows (Site.maxMin s f isMax x y) r ↔
      ∀ i, FloatV.MaxRel isMax (x.get i) (y.get i) (r.get i) := Iff.rfl

theorem itof {s : Shape} {f : Format} {sg : Bool} {w : Nat}
    {x : Tensor s (IntV sg w)} {r : Tensor s (FloatV f)} :
    Sig.Allows (Site.itof s f sg w x) r ↔ ∀ i, tl.ItofRel f (x.get i).toInt (r.get i) := Iff.rfl

theorem ftoi {s : Shape} {f : Format} {sg : Bool} {w : Nat}
    {x : Tensor s (FloatV f)} {r : Tensor s (IntV sg w)} :
    Sig.Allows (Site.ftoi s f sg w x) r ↔ ∀ i, FtoiRel sg w (x.get i) (r.get i) := Iff.rfl

theorem fromF8 {s : Shape} {f g : Format} {x : Tensor s (FloatV f)}
    {r : Tensor s (FloatV g)} :
    Sig.Allows (Site.fromF8 s f g x) r ↔
      ∀ i, (x.get i).isFinite = true → r.get i = FloatV.cvt (x.get i) := Iff.rfl

theorem convert {P : Platform} {s : Shape} {d d' : DType}
    {x : Tensor s d.Val} {r : Tensor s d'.Val} :
    Sig.Allows (Site.convert P s d d' x) r ↔ ∀ i, ConvRel P d d' (x.get i) (r.get i) := Iff.rfl

theorem plat {P : Platform} {q : PlatQ} {r : q.Out} :
    Sig.Allows (Site.plat P q) r ↔ q.Rel P r := Iff.rfl

theorem dotAdd {P : Platform} {fusion : Bool} {cfg : DotCfg} {M N K : Nat}
    {a : Tensor [M, K] cfg.a.Val} {b : Tensor [K, N] cfg.b.Val} {c z r : Tensor [M, N] cfg.acc.Val} :
    Sig.Allows (Site.dotAdd P fusion cfg M N K a b c z) r ↔ (P.dot cfg a b c r ∨
      ∃ d, P.dot cfg a b z d ∧ ∀ i, AccAdd fusion cfg.acc (d.get i) (c.get i) (r.get i)) := Iff.rfl

theorem dotScaledAdd {P : Platform} {fusion : Bool} {cfg : ScaledCfg}
    {M N K : Nat} {a : Tensor [M, K] Nat} {sa : Option (Tensor [M, K] Nat)} {b : Tensor [K, N] Nat}
    {sb : Option (Tensor [K, N] Nat)} {c z r : Tensor [M, N] F32} :
    Sig.Allows (Site.dotScaledAdd P fusion cfg M N K a sa b sb c z) r ↔
      (P.dotScaled cfg a sa b sb c r ∨
        ∃ d, P.dotScaled cfg a sa b sb z d ∧ ∀ i, AddRel fusion (d.get i) (c.get i) (r.get i)) :=
  Iff.rfl

theorem max_constancy {n : Nat} {β : Type} [DecidableEq β] (x : Tensor [n] β) (C : Nat)
    (y : Tensor [n] β) :
    Sig.Allows (Site.hint _ x (ConstAlong x 0 C)) y ↔ y = x ∧ ConstAlong x 0 C := Iff.rfl

theorem max_constancyN {s : Shape} {β : Type} [DecidableEq β] (x : Tensor s β) (Cs : List Nat)
    (y : Tensor s β) :
    Sig.Allows (Site.hint _ x (∀ k, k < s.length → ConstAlong x k (Cs.getD k 1))) y ↔
      y = x ∧ ∀ k, k < s.length → ConstAlong x k (Cs.getD k 1) := Iff.rfl

/-! ## Scans (`TVA.Triton.Scan`)

`tl.associative_scan` was a `Site.other` with the predicate below and a relation `op`; it is now
`Site.scan` with a `Combine` whose `rel` is that relation. Its erasure is the old operation's
(`associative_scan_erase`, by `rfl`), and each scan of the library keeps its relation. -/

theorem scan {s : Shape} {α : Type} {axis : Nat} {reverse : Bool} {c : Combine α}
    {x r : Tensor s α} :
    Sig.Allows (Site.scan s α axis reverse c x) r ↔ ∀ i, ∃ t : Tree,
      ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
      t.Eval c.rel (fun j => x.get (i.set axis j)) (r.get i) := Iff.rfl

/-- The scan with a combine means what the relation-only scan (the old `tl.associative_scan`,
`Site.other`) means with the combine's relation: the same program after erasure. -/
theorem associative_scan_erase {s : Shape} {α : Type} (x : Tensor s α) (axis : Nat)
    (c : Combine α) (reverse : Bool) :
    (tl.associative_scan x axis c reverse).erase =
      (tl.associative_scanRel x axis c.rel reverse).erase := rfl

theorem cumsumWith_rel (fusion : Bool) : (Combine.addRel (f := f) fusion).rel = AddRel fusion :=
  rfl

theorem cumprod_rel :
    (Combine.mulRn (f := f)).rel = fun a b c => FloatV.NaNEq (FloatV.mul a b) c := rfl

theorem icumsum_rel {sg : Bool} {w : Nat} :
    (Combine.ofFn (· + ·) : Combine (IntV sg w)).rel = fun a b c => c = a + b := rfl

theorem icumprod_rel {sg : Bool} {w : Nat} :
    (Combine.ofFn (· * ·) : Combine (IntV sg w)).rel = fun a b c => c = a * b := rfl

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

/-! ## Erasure

`Upd.cas v v` allows only `new = old`, whatever `v`. A continuation that passes an atomic's old
value to it does not depend on the value: its erasure is the same for every old value (so
`Step.defer` may pass the atomic, and `UsesSharedAtomicResult` does not count it), although the
programs differ. -/

/-- A continuation that uses the old value only as `cas`'s compare and new value. -/
def casSelf (d : DType) (b : Int) (sem : MemSem) (sc : Scope) (old : d.Val) : Prog Unit :=
  .rmw d b (Upd.cas old old) sem sc fun _ => .ret ()

theorem cas_self (d : DType) (b : Int) (sem : MemSem) (sc : Scope) (v w : d.Val) :
    (casSelf d b sem sc v).erase = (casSelf d b sem sc w).erase := by
  simp only [casSelf, Prog.erase]
  congr 2
  funext o n
  show (n = if o = v then v else o) = (n = if o = w then w else o)
  split_ifs <;> simp_all

theorem cas_self_ne {d : DType} {b : Int} {sem : MemSem} {sc : Scope} {v w : d.Val} (h : v ≠ w) :
    casSelf d b sem sc v ≠ casSelf d b sem sc w := by
  intro he
  simp only [casSelf, Prog.rmw.injEq, heq_eq_eq] at he
  exact h (Upd.cas.inj he.2.2.1).1

/-- So a kernel whose atomic feeds its old value to `cas old old` means the same as one that
passes a constant. -/
theorem cas_self_correct {E : Env} {L : Launch} {init : Mem} {Q : Mem → Prop} (d : DType)
    (a b : Int) (sem sc : _) (sem' sc' : _) (c : d.Val) :
    Correct E ⟨fun _ => .rmw d a (Upd.xchg c) sem sc (casSelf d b sem' sc')⟩ L init Q ↔
      Correct E ⟨fun _ => .rmw d a (Upd.xchg c) sem sc fun _ => casSelf d b sem' sc' c⟩ L init Q :=
  Correct.congr_erase (fun _ => by
    simp only [Prog.erase]; congr 1; funext v; exact cas_self d b sem' sc' v c) Q

end TVA.Validation.Preserve
