import TVA.Lang.Scan
import TVA.Lemmas.Sum
import TVA.Lemmas.Int
import TVA.Lemmas.Ops
import Mathlib.Data.Fin.Tuple.Sort

/-!
# Proof rules for scans, sorting and histograms

* `safe_associative_scan`: a scan with a total combine relation; each output position comes
  with its tree (`ScanLeaves`). `safe_cumsum`, `safe_cumsumc`, `safe_cumprod`, `safe_icumsum`,
  `safe_icumprod` specialise it.
* Exact combining: `Tree.eval_fun_assoc` (an associative function gives the left fold of the
  leaves) and `Tree.eval_add_sum` (a monoid addition gives the sum). `safe_icumsum` states the
  integer prefix sums.
* Float error: `sum_err_list` bounds a float sum along a tree whose leaves are any list without
  repeats, such as a scan position's (`ScanLeaves.perm`).
* `safe_sort`: a sorted result exists, for dtypes with a linear key (`SortKey.Lawful`: integers
  and floats).
* `safe_histogram`: the counts, when every enabled value is in range.
-/
namespace TVA

open TVA.Math.Float TVA.Math.Float.Format Tree

/-! ## Trees for scans -/

namespace Tree

/-- Renumber the leaves. -/
def mapLeaves (g : Nat → Nat) : Tree → Tree
  | leaf i => leaf (g i)
  | node l r => node (mapLeaves g l) (mapLeaves g r)

@[simp] theorem leaves_mapLeaves (g : Nat → Nat) : ∀ t : Tree, (t.mapLeaves g).leaves = t.leaves.map g
  | leaf _ => rfl
  | node l r => by simp [mapLeaves, leaves, leaves_mapLeaves g l, leaves_mapLeaves g r]

/-- The fold of a list with an associative function starting from a value. -/
theorem foldl_assoc {β : Type} {g : β → β → β} (hg : ∀ a b c, g (g a b) c = g a (g b c))
    (x : Nat → β) : ∀ (l : List Nat) (A B : β),
    l.foldl (fun acc i => g acc (x i)) (g A B) = g A (l.foldl (fun acc i => g acc (x i)) B)
  | [], _, _ => rfl
  | i :: l, A, B => by simp only [List.foldl_cons, hg]; exact foldl_assoc hg x l A (g B (x i))

/-- **An associative function** evaluates a tree to the left fold of its leaves. -/
theorem eval_fun_assoc {β : Type} {g : β → β → β} (hg : ∀ a b c, g (g a b) c = g a (g b c))
    {x : Nat → β} {t : Tree} {r : β} (h : t.Eval (fun a b c => c = g a b) x r) :
    ∃ a l, t.leaves = a :: l ∧ r = l.foldl (fun acc i => g acc (x i)) (x a) := by
  induction h with
  | leaf i => exact ⟨i, [], rfl, rfl⟩
  | node _ _ hc ihl ihr =>
    obtain ⟨a₁, l₁, hl₁, rfl⟩ := ihl
    obtain ⟨a₂, l₂, hl₂, rfl⟩ := ihr
    refine ⟨a₁, l₁ ++ a₂ :: l₂, by simp [leaves, hl₁, hl₂], ?_⟩
    rw [hc, List.foldl_append, List.foldl_cons, foldl_assoc hg x l₂]

/-- **A monoid addition** evaluates a tree to the sum of its leaves. -/
theorem eval_add_sum {β : Type} [AddMonoid β] {x : Nat → β} {t : Tree} {r : β}
    (h : t.Eval (fun a b c => c = a + b) x r) : r = (t.leaves.map x).sum := by
  induction h with
  | leaf i => simp [leaves]
  | node _ _ hc ihl ihr => rw [hc, ihl, ihr]; simp [leaves]

end Tree

/-- A scan position's leaves, as a list up to order. -/
theorem ScanLeaves.perm {reverse : Bool} {n j : Nat} {t : Tree} (h : ScanLeaves reverse n j t) :
    t.leaves.Perm (if reverse then List.range' j (n - j) else List.range (j + 1)) := by
  unfold ScanLeaves at h
  cases reverse with
  | false => simp only [Bool.false_eq_true, ite_false] at h ⊢; rw [h]
  | true =>
    simp only [ite_true] at h ⊢
    rw [h]
    exact List.reverse_perm _

/-- Every position of an axis has a scan tree. -/
theorem ScanLeaves.exists (reverse : Bool) {n j : Nat} (hj : j < n) : ∃ t, ScanLeaves reverse n j t := by
  cases reverse with
  | false => exact ⟨Tree.seq (j + 1), by simp [ScanLeaves, Tree.seq_leaves (j + 1) (Nat.succ_pos j)]⟩
  | true =>
    refine ⟨(Tree.seq (n - j)).mapLeaves (j + (n - j) - 1 - ·), ?_⟩
    simp only [ScanLeaves, ite_true]
    rw [Tree.leaves_mapLeaves, Tree.seq_leaves _ (by omega), List.reverse_range']

/-! ## Float sums along any list of leaves -/

variable {f : Format}

/-- **A float sum along a tree whose leaves are the list `L` up to order** (`L` without
repeats): a scan position, a reduction, a segment. The bound uses depth and node count
`|L| − 1`. -/
theorem sum_err_list (hf : f.Regular) (x : Nat → FloatV f) {t : Tree} {L : List Nat} {r : FloatV f}
    (hL : t.leaves.Perm L) (h : t.Eval (AddRel false) x r) (hx : ∀ i ∈ L, (x i).isFinite = true)
    (hM : (L.map fun i => |(x i).toRat|).sum +
      errBound f.u f.eta (L.length - 1) (L.length - 1) (L.map fun i => |(x i).toRat|).sum ≤ f.maxVal) :
    r.isFinite = true ∧
      |r.toRat - (L.map fun i => (x i).toRat).sum| ≤
        errBound f.u f.eta (L.length - 1) (L.length - 1) (L.map fun i => |(x i).toRat|).sum := by
  have hN : t.nodes + 1 = L.length := by rw [nodes_add_one, hL.length_eq]
  have hD : t.depth ≤ L.length - 1 := by have := depth_le_nodes t; omega
  have hsum : sumOf FloatV.toRat x t = (L.map fun i => (x i).toRat).sum := (hL.map _).sum_eq
  have habs : absSumOf FloatV.toRat x t = (L.map fun i => |(x i).toRat|).sum := (hL.map _).sum_eq
  have hmono := errBound_mono (δ := f.u) (η := f.eta) u_nonneg eta_nonneg hD
    (show t.nodes ≤ L.length - 1 by omega) (absSumOf_nonneg FloatV.toRat x t)
    (le_refl (absSumOf FloatV.toRat x t))
  rw [habs] at hmono
  obtain ⟨hfin, herr⟩ := sum_err hf x h (fun i hi => hx i (hL.subset hi)) (by rw [habs]; linarith)
  rw [hsum, habs] at herr
  exact ⟨hfin, herr.trans hmono⟩

/-! ## Scan rules -/

section Rules
variable {s : Shape} {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.associative_scan`** with a total combine relation. -/
theorem safe_associative_scan {β : Type} {x : Tensor s β} {axis : Nat} {op : β → β → β → Prop}
    {reverse : Bool} (haxis : axis < s.length) (htot : ∀ a b, ∃ c, op a b c)
    {k : Tensor s β → Prog α}
    (hk : ∀ r : Tensor s β, (∀ i, ∃ t : Tree, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
        t.Eval op (fun j => x.get (i.set axis j)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.associative_scan x axis op reverse >>= k) := by
  simp only [tl.associative_scan, Prog.bind_eq, Prog.bind]
  have hex : ∀ i : Idx s, ∃ v t, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
      t.Eval op (fun j => x.get (i.set axis j)) v := by
    intro i
    obtain ⟨t, ht⟩ := ScanLeaves.exists reverse (Idx.coord_lt i haxis)
    obtain ⟨v, hv⟩ := Tree.eval_exists htot (fun j => x.get (i.set axis j)) t
    exact ⟨v, t, ht, hv⟩
  exact .pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ hk

section Float
variable {f : Format}

theorem safe_cumsumWith (fusion : Bool) {x : Tensor s (FloatV f)} {axis : Nat} {reverse : Bool}
    (haxis : axis < s.length) {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, ∃ t : Tree, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
        t.Eval (AddRel fusion) (fun j => x.get (i.set axis j)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cumsumWith fusion x axis reverse >>= k) :=
  safe_associative_scan haxis (addRel_total fusion) hk

/-- **`tl.cumsum`** of floats that are not products: IEEE additions along each position's
tree. `sum_err_list` (with `ScanLeaves.perm`) bounds the error. -/
theorem safe_cumsum {x : Tensor s (FloatV f)} {axis : Nat} {reverse : Bool}
    (haxis : axis < s.length) {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, ∃ t : Tree, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
        t.Eval (AddRel false) (fun j => x.get (i.set axis j)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cumsum x axis reverse >>= k) :=
  safe_cumsumWith false haxis hk

theorem safe_cumsumc {x : Tensor s (FloatV f)} {axis : Nat} {reverse : Bool}
    (haxis : axis < s.length) {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, ∃ t : Tree, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
        t.Eval (AddRel E.fusion) (fun j => x.get (i.set axis j)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cumsumc x axis reverse >>= k) := by
  simp only [tl.cumsumc, Prog.bind_eq, Prog.bind]
  exact .env (safe_cumsumWith E.fusion haxis hk)

theorem safe_cumprod {x : Tensor s (FloatV f)} {axis : Nat} {reverse : Bool}
    (haxis : axis < s.length) {k : Tensor s (FloatV f) → Prog α}
    (hk : ∀ r : Tensor s (FloatV f), (∀ i, ∃ t : Tree, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
        t.Eval (fun a b c => FloatV.NaNEq (FloatV.mul a b) c) (fun j => x.get (i.set axis j)) (r.get i)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cumprod x axis reverse >>= k) :=
  safe_associative_scan haxis (fun _ _ => ⟨_, FloatV.NaNEq.refl _⟩) hk

end Float

/-- **`tl.cumsum`** of integers: exact (wrapping) prefix sums, `j, …, n-1` for `reverse`. -/
theorem safe_icumsum {sg : Bool} {w : Nat} {x : Tensor s (IntV sg w)} {axis : Nat} {reverse : Bool}
    (haxis : axis < s.length) {k : Tensor s (IntV sg w) → Prog α}
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, r.get i =
        ((if reverse then List.range' (i.coord axis) (Shape.dim s axis - i.coord axis)
          else List.range (i.coord axis + 1)).map fun j => x.get (i.set axis j)).sum) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.icumsum x axis reverse >>= k) := by
  unfold tl.icumsum
  refine safe_associative_scan (op := fun a b c => c = a + b) haxis (fun a b => ⟨a + b, rfl⟩) fun r hr => hk r fun i => ?_
  obtain ⟨t, hl, he⟩ := hr i
  rw [Tree.eval_add_sum (β := IntV sg w) he]
  exact (hl.perm.map _).sum_eq

/-- **`tl.cumprod`** of integers: each position is the (wrapping) product of its leaves in
order (`Tree.eval_fun_assoc`). -/
theorem safe_icumprod {sg : Bool} {w : Nat} {x : Tensor s (IntV sg w)} {axis : Nat} {reverse : Bool}
    (haxis : axis < s.length) {k : Tensor s (IntV sg w) → Prog α}
    (hk : ∀ r : Tensor s (IntV sg w), (∀ i, ∃ t : Tree, ScanLeaves reverse (Shape.dim s axis) (i.coord axis) t ∧
        t.Eval (fun a b c => c = a * b) (fun j => x.get (i.set axis j)) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.icumprod x axis reverse >>= k) := by
  unfold tl.icumprod
  exact safe_associative_scan (op := fun a b c => c = a * b) haxis (fun a b => ⟨a * b, rfl⟩) hk

end Rules

/-! ## Sorting -/

/-- A linear key that agrees with `SortKey.lt` on ordered elements. It shows that a sorted
permutation exists (`safe_sort`). -/
class SortKey.Lawful (β : Type) [SortKey β] where
  key : β → Int
  lt_iff : ∀ a b : β, SortKey.Ordered a → SortKey.Ordered b → (SortKey.lt a b = true ↔ key a < key b)
  same_refl : ∀ a : β, SortKey.Ordered a → SortKey.Same a a

instance {sg : Bool} {w : Nat} : SortKey.Lawful (IntV sg w) where
  key := IntV.toInt
  lt_iff a b _ _ := by
    show IntV.lt a b = true ↔ IntV.toInt a < IntV.toInt b
    cases sg <;> simp [IntV.lt, IntV.toInt, BitVec.slt, BitVec.ult]
  same_refl _ _ := rfl

/-- The order key of a float: the IEEE order, with both zeros at `0`. -/
def FloatV.sortKey {f : Format} (x : FloatV f) : Int :=
  if f.isZero x.pat then 0 else ieeeOrderKey f.e f.m x.pat

theorem ieeeOrderKey_of_isZero {f : Format} {x : Nat} (h : f.isZero x = true) :
    ieeeOrderKey f.e f.m x = 0 ∨ ieeeOrderKey f.e f.m x = -1 := by
  simp only [Format.isZero, Format.expField, Format.fracField, Bool.and_eq_true, beq_iff_eq] at h
  simp only [ieeeOrderKey, floatFields, h.1, h.2, Nat.zero_mul, Nat.add_zero]
  split <;> simp

theorem ieeeOrderKey_of_not_isZero {f : Format} {x : Nat} (h : f.isZero x = false) :
    1 ≤ ieeeOrderKey f.e f.m x ∨ ieeeOrderKey f.e f.m x ≤ -2 := by
  simp only [Format.isZero, Format.expField, Format.fracField, Bool.and_eq_false_iff, beq_eq_false_iff_ne,
    ne_eq] at h
  have hmag : 1 ≤ (x >>> f.m) % 2 ^ f.e * 2 ^ f.m + x % 2 ^ f.m := by
    rcases h with hE | hF
    · have h1 : 1 ≤ (x >>> f.m) % 2 ^ f.e := Nat.one_le_iff_ne_zero.mpr hE
      have h2 : 1 ≤ 2 ^ f.m := Nat.one_le_two_pow
      have := Nat.mul_le_mul h1 h2
      omega
    · omega
  simp only [ieeeOrderKey, floatFields]
  generalize (x >>> f.m % 2 ^ f.e * 2 ^ f.m + x % 2 ^ f.m) = M at hmag ⊢
  split_ifs <;> omega

instance {f : Format} : SortKey.Lawful (FloatV f) where
  key := FloatV.sortKey
  lt_iff a b ha hb := by
    show FloatV.lt a b = true ↔ FloatV.sortKey a < FloatV.sortKey b
    have ha' : f.isNaN a.pat = false := ha
    have hb' : f.isNaN b.pat = false := hb
    unfold FloatV.lt Format.compare FloatV.sortKey
    simp only [ha', hb', Bool.or_false, Bool.false_eq_true, ite_false]
    cases hza : f.isZero a.pat <;> cases hzb : f.isZero b.pat
    · simp [compare_lt_iff_lt]
    · have := ieeeOrderKey_of_not_isZero hza
      have := ieeeOrderKey_of_isZero hzb
      simp [compare_lt_iff_lt]
      omega
    · have := ieeeOrderKey_of_isZero hza
      have := ieeeOrderKey_of_not_isZero hzb
      simp [compare_lt_iff_lt]
      omega
    · simp
  same_refl a ha := by
    show FloatV.eq a a = true
    have ha' : f.isNaN a.pat = false := ha
    unfold FloatV.eq Format.compare
    simp only [ha', Bool.or_false, Bool.false_eq_true, ite_false]
    split_ifs <;> simp

/-- A sorted arrangement of a row exists. -/
theorem sorted_row_exists {β : Type} [SortKey β] [SortKey.Lawful β] {n : Nat} (row : Fin n → β)
    (descending : Bool) :
    ∃ σ : Equiv.Perm (Fin n), ∀ t₁ t₂ : Fin n, t₁ < t₂ → (∀ t, SortKey.Ordered (row t)) →
      SortKey.lt (row (σ (if descending then t₁ else t₂))) (row (σ (if descending then t₂ else t₁))) = false := by
  set τ := Tuple.sort (fun t => SortKey.Lawful.key (row t))
  have hmono := Tuple.monotone_sort (fun t => SortKey.Lawful.key (row t))
  have hle : ∀ u v : Fin n, u ≤ v → SortKey.Lawful.key (row (τ u)) ≤ SortKey.Lawful.key (row (τ v)) :=
    fun u v h => hmono h
  cases descending with
  | false =>
    refine ⟨τ, fun t₁ t₂ h hord => ?_⟩
    simp only [Bool.false_eq_true, ite_false]
    rw [Bool.eq_false_iff]
    intro hlt
    have := (SortKey.Lawful.lt_iff _ _ (hord _) (hord _)).mp hlt
    exact absurd (hle t₁ t₂ h.le) (not_le.mpr this)
  | true =>
    refine ⟨Fin.revPerm.trans τ, fun t₁ t₂ h hord => ?_⟩
    simp only [ite_true, Equiv.trans_apply, Fin.revPerm_apply]
    rw [Bool.eq_false_iff]
    intro hlt
    have := (SortKey.Lawful.lt_iff _ _ (hord _) (hord _)).mp hlt
    exact absurd (hle (Fin.rev t₂) (Fin.rev t₁) (Fin.rev_le_rev.mpr h.le)) (not_le.mpr this)

section SortRule
variable {s : Shape} {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.sort`**: every row without NaN comes out as a sorted permutation (up to `==`). -/
theorem safe_sort {β : Type} [SortKey β] [SortKey.Lawful β] {x : Tensor s β} {descending : Bool}
    {k : Tensor s β → Prog α}
    (hk : ∀ r : Tensor s β, (∀ i,
      (∀ t, t < Shape.dim s (s.length - 1) → SortKey.Ordered (x.get (i.set (s.length - 1) t))) →
        PermAlong (s.length - 1) (Shape.dim s (s.length - 1)) x r i ∧
        SortedAlong descending (s.length - 1) (Shape.dim s (s.length - 1)) r i) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sort x descending >>= k) := by
  simp only [tl.sort, Prog.bind_eq, Prog.bind]
  set a := s.length - 1
  set n := Shape.dim s a
  let rowOf : Idx s → Fin n → β := fun i t => x.get (i.set a t.val)
  have hrow : ∀ i t, t < n → rowOf (i.set a t) = rowOf i := fun i t ht => by
    funext u; simp only [rowOf, Idx.set_set i u.2]
  let σ : Idx s → Equiv.Perm (Fin n) := fun i => Classical.choose (sorted_row_exists (rowOf i) descending)
  have hσ := fun i => Classical.choose_spec (sorted_row_exists (rowOf i) descending)
  let r : Tensor s β := ⟨fun j => if h : j.coord a < n then rowOf j (σ j ⟨j.coord a, h⟩) else x.get j⟩
  have hr : ∀ i t (ht : t < n), r.get (i.set a t) = rowOf i (σ i ⟨t, ht⟩) := by
    intro i t ht
    have hc : (i.set a t).coord a = t := Idx.coord_set_self i ht
    show (if h : (i.set a t).coord a < n then rowOf (i.set a t) (σ (i.set a t) ⟨_, h⟩) else _) = _
    have ht' : (i.set a t).coord a < n := by rw [hc]; exact ht
    rw [dif_pos ht']
    have e1 : σ (i.set a t) = σ i := by simp only [σ, hrow i t ht]
    rw [e1, hrow i t ht]
    congr 2
    exact Fin.ext hc
  refine .pick ⟨r, fun i hord => ⟨?_, ?_⟩⟩ hk
  · refine ⟨fun t => if h : t < n then (σ i ⟨t, h⟩).val else t, fun t ht => by simp [ht], ?_, ?_⟩
    · intro t₁ t₂ h₁ h₂ he
      simp only [h₁, h₂, dif_pos] at he
      exact congrArg Fin.val ((σ i).injective (Fin.ext he))
    · intro t ht
      rw [hr i t ht]
      simp only [dif_pos ht]
      exact SortKey.Lawful.same_refl _ (hord _ (σ i ⟨t, ht⟩).2)
  · intro t₁ t₂ h12 h2
    have h1 : t₁ < n := lt_trans h12 h2
    have := (hσ i) ⟨t₁, h1⟩ ⟨t₂, h2⟩ h12 (fun u => hord u.val u.2)
    cases descending with
    | false =>
      simp only [Bool.false_eq_true, ite_false] at this ⊢
      rw [hr i t₂ h2, hr i t₁ h1]; exact this
    | true =>
      simp only [ite_true] at this ⊢
      rw [hr i t₁ h1, hr i t₂ h2]; exact this

end SortRule

/-! ## Histograms -/

/-- **`tl.histogram`**: when every enabled value lies in `[0, bins)`, bin `b` counts them. -/
theorem safe_histogram {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {n : Nat} {sg : Bool} {w : Nat} {x : Tensor [n] (IntV sg w)} {bins : Nat}
    {mask : Option (Tensor [n] Bool)} {k : Tensor [bins] I32 → Prog α}
    (hrange : ∀ j, tl.enabledBy mask j = true → 0 ≤ (x.get1 j).toInt ∧ (x.get1 j).toInt < bins)
    (hk : ∀ r : Tensor [bins] I32, (∀ b : Fin bins, r.get1 b =
        IntV.ofNat (Finset.univ.filter fun j : Fin n => tl.enabledBy mask j = true ∧ (x.get1 j).toInt = b).card) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.histogram x bins mask >>= k) := by
  simp only [tl.histogram, Prog.bind_eq, Prog.bind]
  exact .pick ⟨Tensor.of1 fun b => IntV.ofNat
    (Finset.univ.filter fun j : Fin n => tl.enabledBy mask j = true ∧ (x.get1 j).toInt = b).card,
    hrange, fun b => rfl⟩ fun r hr => hk r hr.2

end TVA
