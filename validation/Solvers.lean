import Solve

/-!
# The solvers of Triton's sites

Stage 2, step 3 of `validation/PICKS.md`: an instance `Solvers` for Triton's signature
(`TVA.Site.sig`), one solver per constructor of `TVA.Site` and `TVA.Upd`. Each solver computes a
value with its proof where it can, proves that there is none where it can (a predicted undefined
behaviour), asks the oracle (`TQuery`) for what it cannot compute (a reduction tree, a lane
order, the device's value at a platform-dependent site), and gives up (`Ans.none`) otherwise.

A supplied answer is checked against the request: a tree that does not cover the leaves or is too
deep, an order that is not a permutation of the enabled lanes, a device dot when the site's
conditions for it fail, are `Gap.badAnswer`. A default (the sequential or the balanced tree,
index order, zero) is used only when the oracle supplied nothing.

The sites at which a solver may return an unproved value (`sitePlatform`, `updPlatform`) are those
whose relation mentions the platform: `plat`, the mixed `dotAdd`/`dotScaledAdd`/`sumBcastMul`, a
conversion into fp8, a float `atomic_add` (directly or through `desc.atomic_add`).
-/
namespace TVA.Validation

open TVA

/-! ## The questions, and the platform-dependent sites -/

/-- What Triton's solvers may ask the oracle. Closed, so that an oracle can match on it. -/
inductive TQuery : Type → Type where
  /-- A reduction tree over `n` leaves. -/
  | tree (n : Nat) : TQuery Tree
  /-- An order of the lanes of a block of shape `s`. -/
  | order (s : Shape) : TQuery (List (Idx s))
  /-- An element of dtype `d` (an undefined padding value). -/
  | val (d : DType) : TQuery d.Val
  /-- The device's result of a platform operation on its operands. -/
  | plat (q : PlatQ) : TQuery q.Out
  /-- The device's conversion of an element `x` of dtype `d` into the fp8 dtype `d'`. -/
  | convert (d d' : DType) (x : d.Val) : TQuery d'.Val
  /-- The device's new value of a float `atomic_add` of `v` to a cell holding `old`. -/
  | atomicAdd (d : DType) (old v : d.Val) : TQuery d.Val

/-- A standard float dtype (`float16`, `bfloat16`, `float32`, `float64`). -/
def stdFloat (d : DType) : Prop := d = .f16 ∨ d = .bf16 ∨ d = .f32 ∨ d = .f64

/-- The sites whose relation mentions the platform: an unproved (device) value is allowed. -/
def sitePlatform : Site → Prop
  | .plat _ _ | .dotAdd .. | .dotScaledAdd .. | .sumBcastMul .. => True
  | .convert _ _ _ d' _ => d' = .f8e5 ∨ d' = .f8e4nv
  | _ => False

/-- The updates whose relation mentions the platform: a float `atomic_add`. -/
def updPlatform {d : DType} : Upd d → Prop
  | .add _ _ => stdFloat d
  | .red _ op _ => op = .add ∧ stdFloat d
  | _ => False

/-! ## Helpers -/

namespace Ans

variable {plat : Prop}

/-- Lane by lane over a tensor (`Ans.pi` over `Idx s`, read as a tensor). -/
def lanes {s : Shape} {α : Type} {R : Idx s → α → Prop} (f : (i : Idx s) → Ans (R i) plat) :
    Ans (fun r : Tensor s α => ∀ i, R i (r.get i)) plat :=
  (Ans.pi (Idx.all s) f).transport Tensor.mk Tensor.get (fun _ h => h) (fun _ h => h) id

/-- Row by row of a rank-1 tensor (`Ans.pi` over `Fin m`, read as a tensor). -/
def rows {m : Nat} {α : Type} {R : Fin m → α → Prop} (f : (i : Fin m) → Ans (R i) plat) :
    Ans (fun r : Tensor [m] α => ∀ i, R i (r.get1 i)) plat :=
  (Ans.pi (List.finRange m) f).transport Tensor.of1 Tensor.get1 (fun _ h => h) (fun _ h => h) id

/-- An answer at a site with no platform freedom, at any site. -/
def weaken {β : Type} {P : β → Prop} (a : Ans P False) : Ans P plat :=
  a.cast (fun _ => Iff.rfl) False.elim

end Ans

/-- The oracle's answer to `q`. -/
abbrev Hint := Hints TQuery

/-! ## Elementwise: lanes, conversions -/

/-- A lanewise partial function: lane `i` is `g i`; empty when some lane has no value. -/
def solveLanes {s : Shape} {α : Type} (g : Idx s → Option α) {plat : Prop} :
    Ans (fun r : Tensor s α => ∀ i, g i = some (r.get i)) plat :=
  Ans.lanes fun i => match h : g i with
    | some a => .ok a h
    | Option.none => .empty fun _ h' => by rw [h] at h'; cases h'

/-- The only candidate `FtoiRel` allows, decided. -/
def ftoiAns {f : Format} (sg : Bool) (w : Nat) (x : FloatV f) {plat : Prop} :
    Ans (FtoiRel sg w x) plat :=
  if h : f.Std ∧ x.isFinite = true ∧ IntV.minInt sg w ≤ x.toRat.num.tdiv x.toRat.den ∧
      x.toRat.num.tdiv x.toRat.den ≤ IntV.maxInt sg w then
    .ok _ ⟨h.1, h.2.1, h.2.2.1, h.2.2.2, rfl⟩
  else .empty fun _ hr => h ⟨hr.1, hr.2.1, hr.2.2.1, hr.2.2.2.1⟩

/-- `ofIntRel z d'`, solved. -/
def ofIntAns (z : Int) {plat : Prop} : (d' : DType) → Ans (ofIntRel z d') plat
  | .i1 => .ok _ rfl
  | .i8 => .ok _ rfl | .i16 => .ok _ rfl | .i32 => .ok _ rfl | .i64 => .ok _ rfl
  | .u8 => .ok _ rfl | .u16 => .ok _ rfl | .u32 => .ok _ rfl | .u64 => .ok _ rfl
  | .f16 => .ok _ (Or.inl rfl) | .bf16 => .ok _ (Or.inl rfl)
  | .f32 => .ok _ (Or.inl rfl) | .f64 => .ok _ (Or.inl rfl)
  | .f8e5 => .empty fun _ h => h | .f8e4nv => .empty fun _ h => h

/-- `FloatRel P f g x`, solved; into fp8 from a format the platform converts, the oracle's value
`dev` (`none` when it has none), assumed with `hp`. -/
def floatAns (P : Platform) (f g : Format) (x : FloatV f) {plat : Prop}
    (dev : g.F8 → Option (FloatV g)) (hp : g.F8 → plat) : Ans (FloatRel P f g x) plat :=
  if h : f = g then .ok (h ▸ x) (by subst h; simp [FloatRel])
  else if hs : f.Std ∧ g.Std then
    .ok (FloatV.cvt x) (by
      rw [FloatRel, ite_eq_right h]; exact Or.inl ⟨hs.1, hs.2, Or.inl rfl⟩)
  else if hp8 : (f = .binary16 ∨ f = .bfloat16 ∨ f = .binary32) ∧ g.F8 then
    match dev hp8.2 with
    | some r => .assumed r (hp hp8.2)
    | Option.none => .none (.noOracle "conversion into fp8")
  else if h8 : f.F8 ∧ (g = .binary16 ∨ g = .bfloat16 ∨ g = .binary32) then
    .ok (FloatV.cvt x) (by
      rw [FloatRel, ite_eq_right h]; exact Or.inr (Or.inr ⟨h8.1, h8.2, fun _ => rfl⟩))
  else .empty fun r hr => by
    rw [FloatRel, ite_eq_right h] at hr
    rcases hr with ⟨h1, h2, _⟩ | ⟨h1, h2, _⟩ | ⟨h1, h2, _⟩
    · exact hs ⟨h1, h2⟩
    · exact hp8 ⟨h1, h2⟩
    · exact h8 ⟨h1, h2⟩

/-- `ofFloatRel P x d'`, solved; the oracle answers a conversion into fp8. -/
def ofFloatAns (P : Platform) {d : DType} {f : Format} (x : FloatV f) (xd : d.Val) (H : Hint) :
    (d' : DType) → Ans (ofFloatRel P x d') (d' = .f8e5 ∨ d' = .f8e4nv)
  | .i1 => .ok _ rfl
  | .i8 => ftoiAns true 8 x | .i16 => ftoiAns true 16 x | .i32 => ftoiAns true 32 x
  | .i64 => ftoiAns true 64 x
  | .u8 => ftoiAns false 8 x | .u16 => ftoiAns false 16 x | .u32 => ftoiAns false 32 x
  | .u64 => ftoiAns false 64 x
  | .f16 => floatAns P f _ x (fun h => absurd h (by decide)) (fun h => absurd h (by decide))
  | .bf16 => floatAns P f _ x (fun h => absurd h (by decide)) (fun h => absurd h (by decide))
  | .f32 => floatAns P f _ x (fun h => absurd h (by decide)) (fun h => absurd h (by decide))
  | .f64 => floatAns P f _ x (fun h => absurd h (by decide)) (fun h => absurd h (by decide))
  | .f8e5 => floatAns P f _ x (fun _ => H (.convert d .f8e5 xd)) (fun _ => Or.inl rfl)
  | .f8e4nv => floatAns P f _ x (fun _ => H (.convert d .f8e4nv xd)) (fun _ => Or.inr rfl)

/-- `ConvRel P d d' x`, solved; the oracle answers a conversion into fp8. -/
def convAns (P : Platform) (H : Hint) :
    (d d' : DType) → (x : d.Val) → Ans (ConvRel P d d' x) (d' = .f8e5 ∨ d' = .f8e4nv)
  | .i1, d', x => ofIntAns (if x then 1 else 0) d'
  | .i8, d', x => ofIntAns x.toInt d' | .i16, d', x => ofIntAns x.toInt d'
  | .i32, d', x => ofIntAns x.toInt d' | .i64, d', x => ofIntAns x.toInt d'
  | .u8, d', x => ofIntAns x.toInt d' | .u16, d', x => ofIntAns x.toInt d'
  | .u32, d', x => ofIntAns x.toInt d' | .u64, d', x => ofIntAns x.toInt d'
  | .f16, d', x => ofFloatAns P (d := .f16) x x H d'
  | .bf16, d', x => ofFloatAns P (d := .bf16) x x H d'
  | .f32, d', x => ofFloatAns P (d := .f32) x x H d'
  | .f64, d', x => ofFloatAns P (d := .f64) x x H d'
  | .f8e5, d', x => ofFloatAns P (d := .f8e5) x x H d'
  | .f8e4nv, d', x => ofFloatAns P (d := .f8e4nv) x x H d'

/-! ## Reductions: trees -/

/-- Evaluate a tree with the function `op` at each node. -/
def treeEval {α : Type} (op : α → α → α) (x : Nat → α) : Tree → α
  | .leaf i => x i
  | .node l r => op (treeEval op x l) (treeEval op x r)

/-- A function that each node's relation allows gives a result the tree can produce. -/
theorem eval_treeEval {α : Type} {rel : α → α → α → Prop} {op : α → α → α}
    (h : ∀ a b, rel a b (op a b)) (x : Nat → α) : (t : Tree) → t.Eval rel x (treeEval op x t)
  | .leaf i => .leaf i
  | .node l r => .node (eval_treeEval h x l) (eval_treeEval h x r) (h _ _)

instance (n : Nat) (t : Tree) : Decidable (t.Covers n) := by unfold Tree.Covers; infer_instance

theorem leaves_ne_nil : (t : Tree) → t.leaves ≠ []
  | .leaf _ => by simp [Tree.leaves]
  | .node l _ => by simp [Tree.leaves, leaves_ne_nil l]

/-- No tree covers zero leaves: a reduction of nothing has no result. -/
theorem not_covers_zero (t : Tree) : ¬ t.Covers 0 := fun h =>
  leaves_ne_nil t (List.Perm.eq_nil (by simpa [Tree.Covers] using h))

/-- The balanced tree over `lo, …, lo + n - 1` (depth `⌈log₂ n⌉`): a default within the depth
bound of a float sum when the sequential tree is too deep. -/
def balanced (lo : Nat) : Nat → Tree
  | 0 | 1 => .leaf lo
  | n + 2 => .node (balanced lo ((n + 2) / 2)) (balanced (lo + (n + 2) / 2) ((n + 2) - (n + 2) / 2))
termination_by n => n
decreasing_by all_goals omega

/-- **A result along a tree** over `n` leaves satisfying `C`, each node evaluated with `op`
(which `rel` allows): along the oracle's tree if it supplied one (`badAnswer` if it does not cover
the leaves or fails `C`), else along the sequential tree or the balanced one, whichever is first to
satisfy `C`. Empty when `n = 0`. -/
def solveTree {α : Type} (n : Nat) (C : Tree → Prop) [DecidablePred C] {rel : α → α → α → Prop}
    (op : α → α → α) (h : ∀ a b, rel a b (op a b)) (x : Nat → α) (H : Hint) {plat : Prop} :
    Ans (fun r => ∃ t : Tree, t.Covers n ∧ C t ∧ t.Eval rel x r) plat :=
  if hn : n = 0 then .empty fun _ ⟨t, hc, _⟩ => not_covers_zero t (hn ▸ hc)
  else
    let try1 (t : Tree) :
        Option (Ans (fun r => ∃ t : Tree, t.Covers n ∧ C t ∧ t.Eval rel x r) plat) :=
      if hc : t.Covers n ∧ C t then some (.ok _ ⟨t, hc.1, hc.2, eval_treeEval h x t⟩)
      else Option.none
    match H (.tree n) with
    | some t => (try1 t).getD (.none (.badAnswer s!"the oracle's tree does not fit {n} leaves"))
    | Option.none => ((try1 (Tree.seq n)).orElse fun _ => try1 (balanced 0 n)).getD
        (.none (.searchFailed s!"no default tree over {n} leaves within the bound"))

/-- `solveTree` without a condition on the tree. -/
def solveTree' {α : Type} (n : Nat) {rel : α → α → α → Prop} (op : α → α → α)
    (h : ∀ a b, rel a b (op a b)) (x : Nat → α) (H : Hint) {plat : Prop} :
    Ans (fun r => ∃ t : Tree, t.Covers n ∧ t.Eval rel x r) plat :=
  (solveTree n (fun _ => True) op h x H).cast
    (fun _ => ⟨fun ⟨t, h1, _, h2⟩ => ⟨t, h1, h2⟩, fun ⟨t, h1, h2⟩ => ⟨t, h1, trivial, h2⟩⟩) id

/-- `FloatV.max` (`isMax`) or `FloatV.min`: a result `MaxRel` allows. -/
def maxOf {f : Format} (isMax : Bool) (a b : FloatV f) : FloatV f :=
  if isMax then FloatV.max a b else FloatV.min a b

theorem maxRel_maxOf {f : Format} (isMax : Bool) (a b : FloatV f) :
    FloatV.MaxRel isMax a b (maxOf isMax a b) := by
  cases isMax <;> exact Or.inl (Or.inl rfl)

/-- `ArgRes` along a tree (`solveTree'`, with `argCombine`). Empty when `n = 0`. -/
def argAns {α : Type} [CmpKey α] (isMax tieLeft : Bool) (n : Nat) (g : Nat → α) (H : Hint)
    {plat : Prop} : Ans (tl.ArgRes isMax tieLeft n g) plat :=
  if hn : n = 0 then .empty fun r hr => by
    rcases hr with ⟨_, t, hc, _⟩ | ⟨⟨j, hj, _⟩, _⟩
    · exact not_covers_zero t (hn ▸ hc)
    · omega
  else
    match solveTree' (plat := plat) n (rel := fun a b c => c = tl.argCombine isMax tieLeft a b)
        (tl.argCombine isMax tieLeft) (fun _ _ => rfl) (fun j => (g j, j)) H with
    | .ok p hp => .ok (p.1, IntV.ofNat p.2) (Or.inl (by
        obtain ⟨t, h1, h2⟩ := hp
        exact ⟨p, t, h1, h2, rfl⟩))
    | .empty _ => .none (.searchFailed "index reduction: no tree")
    | .none g => .none g
    | .assumed p hp => .assumed (p.1, IntV.ofNat p.2) hp

/-! ## Padding and histograms -/

/-- A NaN of dtype `d`: the all-ones pattern for a float dtype; none for the others. -/
def nanAns {plat : Prop} : (d : DType) → Ans (fun x : d.Val => d.IsNaN x) plat
  | .f16 => Ans.check (DType.f16.ofBits (2 ^ 16 - 1)) "NaN"
  | .bf16 => Ans.check (DType.bf16.ofBits (2 ^ 16 - 1)) "NaN"
  | .f32 => Ans.check (DType.f32.ofBits (2 ^ 32 - 1)) "NaN"
  | .f64 => Ans.check (DType.f64.ofBits (2 ^ 64 - 1)) "NaN"
  | .f8e5 => Ans.check (DType.f8e5.ofBits (2 ^ 8 - 1)) "NaN"
  | .f8e4nv => Ans.check (DType.f8e4nv.ofBits (2 ^ 8 - 1)) "NaN"
  | .i1 | .i8 | .i16 | .i32 | .i64 | .u8 | .u16 | .u32 | .u64 => .empty fun _ h => h

/-- A padding value: the oracle's element (`.undef`, else zero), zero, or a NaN. -/
def padAns {d : DType} (p : Padding) (H : Hint) {plat : Prop} : Ans (p.Holds (d := d)) plat :=
  match p with
  | .undef => .ok ((H (.val d)).getD (d.ofBits 0)) trivial
  | .zero => .ok _ rfl
  | .nan => nanAns d

/-- `tl.histogram`: the counts, or empty when an enabled element is out of range. -/
def histAns (n bins : Nat) {sg : Bool} {w : Nat} (x : Tensor [n] (IntV sg w))
    (mask : Option (Tensor [n] Bool)) {plat : Prop} :
    Ans (Site.Allows (.histogram n bins sg w x mask)) plat :=
  if h : ∀ j, tl.enabledBy mask j = true → 0 ≤ (x.get1 j).toInt ∧ (x.get1 j).toInt < bins then
    .ok (Tensor.of1 fun b => IntV.ofNat (Finset.univ.filter fun j : Fin n =>
      tl.enabledBy mask j = true ∧ (x.get1 j).toInt = b).card) ⟨h, fun _ => rfl⟩
  else .empty fun _ hr => h hr.1

/-! ## Sums of broadcast products -/

/-- A product and an addition of dtype `d` that `tl.MulRel d` and `AccAdd fusion d` allow: the
IEEE ones of a standard float dtype; none for the others (no product is allowed). -/
def ieeeOps (fusion : Bool) : (d : DType) →
    Option {ops : (d.Val → d.Val → d.Val) × (d.Val → d.Val → d.Val) //
      (∀ a b, tl.MulRel d a b (ops.1 a b)) ∧ ∀ a b, AccAdd fusion d a b (ops.2 a b)}
  | .f16 => some ⟨(FloatV.mul, FloatV.add), fun _ _ => Or.inl rfl, fun _ _ => AddRel.ieee _ _ _⟩
  | .bf16 => some ⟨(FloatV.mul, FloatV.add), fun _ _ => Or.inl rfl, fun _ _ => AddRel.ieee _ _ _⟩
  | .f32 => some ⟨(FloatV.mul, FloatV.add), fun _ _ => Or.inl rfl, fun _ _ => AddRel.ieee _ _ _⟩
  | .f64 => some ⟨(FloatV.mul, FloatV.add), fun _ _ => Or.inl rfl, fun _ _ => AddRel.ieee _ _ _⟩
  | _ => Option.none

/-- The products-and-tree branch of `sumBcastMul` along the tree `t` (the same for every
element), when it covers `K` and the dtype has IEEE operations. -/
def sbmTree (P : Platform) (fusion : Bool) (d : DType) (M K N : Nat) (x : Tensor [M, K] d.Val)
    (y : Tensor [K, N] d.Val) (t : Tree) :
    Option (Ans (Site.Allows (.sumBcastMul P fusion d M K N x y)) True) :=
  match ieeeOps fusion d with
  | Option.none => Option.none
  | some ⟨ops, hmul, hadd⟩ =>
    if hc : t.Covers K then
      let p (i : Fin M) (j : Fin N) (k : Nat) : d.Val :=
        if hk : k < K then ops.1 (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) else default
      some (.ok (Tensor.of2 fun i j => treeEval ops.2 (p i j) t) (Or.inl fun i j =>
        ⟨p i j, fun k hk => by simp only [p, dite_eq_left hk]; exact hmul _ _,
          t, hc, eval_treeEval hadd _ t⟩))
    else Option.none

/-- `sumBcastMul`: along the oracle's tree if it supplied one; else the oracle's `tl.dot` (input
precision `ieee`), assumed, if it supplied one and the site allows the dot branch; else along the
sequential tree. -/
def sbmAns (P : Platform) (fusion : Bool) (d : DType) (M K N : Nat) (x : Tensor [M, K] d.Val)
    (y : Tensor [K, N] d.Val) (H : Hint) :
    Ans (Site.Allows (.sumBcastMul P fusion d M K N x y)) True :=
  let cfg : DotCfg := { a := d, b := d, acc := d, precision := .ieee }
  match H (.tree K) with
  | some t => (sbmTree P fusion d M K N x y t).getD
      (.none (.badAnswer s!"the oracle's tree does not fit {K} leaves, or no IEEE products"))
  | Option.none =>
    match H (.plat (.dot cfg x y (Tensor.full (d.ofBits 0)))) with
    | some r =>
      if d.bytes ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N then .assumed r trivial
      else .none (.badAnswer
        "a device dot where the site allows none (dtype wider than 32 bits, or M or N below 16)")
    | Option.none => (sbmTree P fusion d M K N x y (Tree.seq K)).getD
        (.none (.searchFailed "sum of broadcast products: no IEEE products, or K = 0"))

/-! ## The updates of atomics -/

/-- `tl.addUpd e d v o`: integers computed; a float's the oracle's, assumed. -/
def addAns (H : Hint) (e : Env) : (d : DType) → (v o : d.Val) →
    Ans (tl.addUpd e d v o) (stdFloat d)
  | .i8, _, _ => .ok _ rfl | .i16, _, _ => .ok _ rfl | .i32, _, _ => .ok _ rfl
  | .i64, _, _ => .ok _ rfl
  | .u8, _, _ => .ok _ rfl | .u16, _, _ => .ok _ rfl | .u32, _, _ => .ok _ rfl
  | .u64, _, _ => .ok _ rfl
  | .f16, v, o => match H (.atomicAdd .f16 o v) with
    | some n => .assumed n (Or.inl rfl) | Option.none => .none (.noOracle "float atomic_add")
  | .bf16, v, o => match H (.atomicAdd .bf16 o v) with
    | some n => .assumed n (Or.inr (Or.inl rfl))
    | Option.none => .none (.noOracle "float atomic_add")
  | .f32, v, o => match H (.atomicAdd .f32 o v) with
    | some n => .assumed n (Or.inr (Or.inr (Or.inl rfl)))
    | Option.none => .none (.noOracle "float atomic_add")
  | .f64, v, o => match H (.atomicAdd .f64 o v) with
    | some n => .assumed n (Or.inr (Or.inr (Or.inr rfl)))
    | Option.none => .none (.noOracle "float atomic_add")
  | .i1, _, _ => .empty fun _ h => h
  | .f8e5, _, _ => .empty fun _ h => h | .f8e4nv, _, _ => .empty fun _ h => h

/-- `tl.fBound`, solved: the new value is one of the two. -/
def fBoundAns {f : Format} (isMax : Bool) (o v : FloatV f) {plat : Prop} :
    Ans (tl.fBound isMax o v) plat :=
  if h : tl.fBound isMax o v o then .ok o h
  else if h' : tl.fBound isMax o v v then .ok v h'
  else .empty fun n hn => by
    rcases hn.1 with rfl | rfl
    · exact h hn
    · exact h' hn

/-- `tl.maxUpd isMax d v o`, solved. -/
def maxAns (isMax : Bool) {plat : Prop} : (d : DType) → (v o : d.Val) →
    Ans (tl.maxUpd isMax d v o) plat
  | .i8, _, _ => .ok _ rfl | .i16, _, _ => .ok _ rfl | .i32, _, _ => .ok _ rfl
  | .i64, _, _ => .ok _ rfl
  | .u8, _, _ => .ok _ rfl | .u16, _, _ => .ok _ rfl | .u32, _, _ => .ok _ rfl
  | .u64, _, _ => .ok _ rfl
  | .f16, v, o => fBoundAns isMax o v | .bf16, v, o => fBoundAns isMax o v
  | .f32, v, o => fBoundAns isMax o v | .f64, v, o => fBoundAns isMax o v
  | .i1, _, _ => .empty fun _ h => h
  | .f8e5, _, _ => .empty fun _ h => h | .f8e4nv, _, _ => .empty fun _ h => h

/-- `tl.bitUpd op d v o`, solved. -/
def bitAns (op : ∀ {sg : Bool} {w : Nat}, IntV sg w → IntV sg w → IntV sg w) {plat : Prop} :
    (d : DType) → (v o : d.Val) → Ans (tl.bitUpd op d v o) plat
  | .i8, _, _ => .ok _ rfl | .i16, _, _ => .ok _ rfl | .i32, _, _ => .ok _ rfl
  | .i64, _, _ => .ok _ rfl
  | .u8, _, _ => .ok _ rfl | .u16, _, _ => .ok _ rfl | .u32, _, _ => .ok _ rfl
  | .u64, _, _ => .ok _ rfl
  | .i1, _, _ => .empty fun _ h => h
  | .f16, _, _ | .bf16, _, _ | .f32, _, _ | .f64, _, _ => .empty fun _ h => h
  | .f8e5, _, _ => .empty fun _ h => h | .f8e4nv, _, _ => .empty fun _ h => h

/-- `RedOp.Upd e op d v o` (`desc.atomic_*`), solved. -/
def redAns (H : Hint) (e : Env) (op : RedOp) (d : DType) (v o : d.Val) :
    Ans (RedOp.Upd e op d v o) (op = .add ∧ stdFloat d) :=
  match op, d with
  | .add, d => (addAns H e d v o).cast (fun _ => Iff.rfl) fun h => ⟨rfl, h⟩
  | .min, .f16 => .ok (FloatV.min o v) (Or.inl (Or.inl (Or.inl rfl)))
  | .min, .bf16 => .ok (FloatV.min o v) (Or.inl (Or.inl (Or.inl rfl)))
  | .max, .f16 => .ok (FloatV.max o v) (Or.inl (Or.inl (Or.inl rfl)))
  | .max, .bf16 => .ok (FloatV.max o v) (Or.inl (Or.inl (Or.inl rfl)))
  | .min, .i1 => maxAns false .i1 v o | .min, .i8 => maxAns false .i8 v o
  | .min, .i16 => maxAns false .i16 v o | .min, .i32 => maxAns false .i32 v o
  | .min, .i64 => maxAns false .i64 v o | .min, .u8 => maxAns false .u8 v o
  | .min, .u16 => maxAns false .u16 v o | .min, .u32 => maxAns false .u32 v o
  | .min, .u64 => maxAns false .u64 v o | .min, .f32 => maxAns false .f32 v o
  | .min, .f64 => maxAns false .f64 v o | .min, .f8e5 => maxAns false .f8e5 v o
  | .min, .f8e4nv => maxAns false .f8e4nv v o
  | .max, .i1 => maxAns true .i1 v o | .max, .i8 => maxAns true .i8 v o
  | .max, .i16 => maxAns true .i16 v o | .max, .i32 => maxAns true .i32 v o
  | .max, .i64 => maxAns true .i64 v o | .max, .u8 => maxAns true .u8 v o
  | .max, .u16 => maxAns true .u16 v o | .max, .u32 => maxAns true .u32 v o
  | .max, .u64 => maxAns true .u64 v o | .max, .f32 => maxAns true .f32 v o
  | .max, .f64 => maxAns true .f64 v o | .max, .f8e5 => maxAns true .f8e5 v o
  | .max, .f8e4nv => maxAns true .f8e4nv v o
  | .and, d => bitAns IntV.and d v o
  | .or, d => bitAns IntV.or d v o
  | .xor, d => bitAns IntV.xor d v o

/-! ## The solvers -/

/-- **The solver of a site.** -/
def solveSite (s : Site) (H : Hint) : Ans (Site.Allows s) (sitePlatform s) :=
  match s with
  | @Site.guard c _ => if h : c then .ok () h else .empty fun _ h' => h h'
  | @Site.hint _ x c _ => if h : c then .ok x ⟨rfl, h⟩ else .empty fun _ hy => h hy.2
  | .lanes _ _ g => solveLanes g
  | .upToNaN _ _ v => .ok v fun _ => Or.inl rfl
  | .addc _ _ fusion x y => .ok (Tensor.map2 FloatV.add x y) fun _ => AddRel.ieee fusion _ _
  | .mulAdd _ _ _ a b c =>
    .ok (Tensor.map3 (fun a b c => FloatV.add (FloatV.mul a b) c) a b c)
      fun _ => Or.inl (Or.inl rfl)
  | .maxMin _ _ isMax x y => .ok (Tensor.map2 (maxOf isMax) x y) fun _ => maxRel_maxOf isMax _ _
  | .itof _ _ _ _ x => .ok (x.map fun a => FloatV.ofRat (a.toInt : Rat)) fun _ => Or.inl rfl
  | .ftoi _ _ sg w x => Ans.lanes fun i => ftoiAns sg w (x.get i)
  | .fromF8 _ _ _ x => .ok (x.map FloatV.cvt) fun _ _ => rfl
  | .convert P _ d d' x => Ans.lanes fun i => convAns P H d d' (x.get i)
  | .plat _ q => match H (.plat q) with
    | some r => .assumed r trivial
    | Option.none => .none (.noOracle "platform operation")
  | .dotAdd _ _ cfg _ _ _ a b c _ => match H (.plat (.dot cfg a b c)) with
    | some r => .assumed r trivial
    | Option.none => .none (.noOracle "tl.dot(a, b, acc) + c")
  | .dotScaledAdd _ _ cfg _ _ _ a sa b sb c _ =>
    match H (.plat (.dotScaled cfg a sa b sb c)) with
    | some r => .assumed r trivial
    | Option.none => .none (.noOracle "tl.dot_scaled(…) + c")
  | .fsum _ fusion depth n _ =>
    solveTree n (fun t => t.depth ≤ depth) FloatV.add (AddRel.ieee fusion) _ H
  | .fsum1 _ fusion depth _ n _ => Ans.rows fun _ =>
    solveTree n (fun t => t.depth ≤ depth) FloatV.add (AddRel.ieee fusion) _ H
  | .fsum0 _ fusion depth m _ _ => Ans.rows fun _ =>
    solveTree m (fun t => t.depth ≤ depth) FloatV.add (AddRel.ieee fusion) _ H
  | .fmax _ isMax n _ => solveTree' n (maxOf isMax) (maxRel_maxOf isMax) _ H
  | .fmax1 _ isMax _ n _ => Ans.rows fun _ => solveTree' n (maxOf isMax) (maxRel_maxOf isMax) _ H
  | .fmax0 _ isMax m _ _ => Ans.rows fun _ => solveTree' m (maxOf isMax) (maxRel_maxOf isMax) _ H
  | @Site.arg _ _ _ isMax tieLeft n _ => argAns isMax tieLeft n _ H
  | @Site.arg1 _ _ _ isMax tieLeft m n x =>
    (Ans.rows (m := m) fun i => argAns isMax tieLeft n
      (fun j => if h : j < n then x.get2 i ⟨j, h⟩ else default) H).transport
      (fun r => (r.map Prod.fst, r.map Prod.snd)) (fun r => Tensor.map2 Prod.mk r.1 r.2)
      (fun _ h i => h i) (fun _ h i => h i) id
  | @Site.arg0 _ _ _ isMax tieLeft m n x =>
    (Ans.rows (m := n) fun j => argAns isMax tieLeft m
      (fun i => if h : i < m then x.get2 ⟨i, h⟩ j else default) H).transport
      (fun r => (r.map Prod.fst, r.map Prod.snd)) (fun r => Tensor.map2 Prod.mk r.1 r.2)
      (fun _ h i => h i) (fun _ h i => h i) id
  | .histogram n bins _ _ x mask => histAns n bins x mask
  | .pad _ d p inb v =>
    Ans.lanes (R := fun i b => if inb i = true then b = v.get i else p.Holds b) fun i =>
      if h : inb i = true then .ok (v.get i) (by rw [ite_eq_left h])
      else (padAns (d := d) p H).cast (fun _ => by rw [ite_eq_right h]) id
  | .anyVals _ d => .ok (fun _ => d.ofBits 0) trivial
  | .order s enabled => match H (.order s) with
    | some o => if h : o.Perm enabled then .ok o h
      else .none (.badAnswer "the oracle's lane order is not a permutation of the enabled lanes")
    | Option.none => .ok enabled (.refl _)
  | .sumBcastMul P fusion d M K N x y => sbmAns P fusion d M K N x y H
  | .other _ _ => .none (.unsupported "Site.other")

/-- **The solver of an update**, given the old value. -/
def solveUpd {d : DType} (u : Upd d) (old : d.Val) (H : Hint) :
    Ans (Upd.Allows u old) (updPlatform u) :=
  match u with
  | .add e v => addAns H e d v old
  | .xchg v => .ok v rfl
  | .cas _ _ => .ok _ rfl
  | .max isMax v => maxAns isMax d v old
  | .bit op v => bitAns op d v old
  | .red e op v => redAns H e op d v old
  | .other _ => .none (.unsupported "Upd.other")

/-- **Triton's solvers.** -/
instance Site.solvers : Solvers where
  Query := TQuery
  Platform := sitePlatform
  UpdPlatform := updPlatform
  solve := solveSite
  solveUpd := solveUpd

end TVA.Validation
