import Proof.Defs

/-!
# `route_histogram_i32`: arithmetic and counting

`int32` sums of counts, and the two ways of counting the lanes of a program that are enabled and
in a given bucket: over the list `Idx.all` of the block shape (as the semantics does), and over
`Finset.range` (as `progCount` does).
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-! ## `int32` arithmetic -/

theorem ofNat_zero {s : Bool} {w : Nat} : (IntV.ofNat 0 : IntV s w) = 0 := rfl

/-- A `Finset` sum of constants `IntV.ofNat (f i)` is the constant of the `Nat` sum. -/
theorem ofNat_sum {s : Bool} {w : Nat} {ι : Type} (t : Finset ι) (f : ι → Nat) :
    (∑ i ∈ t, (IntV.ofNat (f i) : IntV s w)) = IntV.ofNat (∑ i ∈ t, f i) := by
  classical
  refine Finset.induction_on t ?_ ?_
  · rw [Finset.sum_empty, Finset.sum_empty, ofNat_zero]
  · intro a t ha ih
    rw [Finset.sum_insert ha, Finset.sum_insert ha, ih, IntV.ofNat_add_ofNat]

/-- Modular arithmetic behind `bv_add_sub`: `(a + b) - a ≡ b (mod M)` for `a, b < M`. -/
theorem nat_mod_add_sub (M a b : Nat) (ha : a < M) (hb : b < M) :
    ((a + b) % M + (M - a)) % M = b := by
  have haM : a ≤ M := le_of_lt ha
  rcases Nat.lt_or_ge (a + b) M with h | h
  · rw [Nat.mod_eq_of_lt h]
    have h2 : a + b + (M - a) = b + M := by omega
    rw [h2, Nat.add_mod_right, Nat.mod_eq_of_lt hb]
  · have hx : (a + b) % M = a + b - M := by
      rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt (by omega)]
    rw [hx]
    have h2 : a + b - M + (M - a) = b := by omega
    rw [h2, Nat.mod_eq_of_lt hb]

/-- Bit-vector form of `a + b - a = b`. -/
theorem bv_add_sub (w : Nat) (a b : BitVec w) : a + b - a = b := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_sub, BitVec.toNat_add]
  have := nat_mod_add_sub (2 ^ w) a.toNat b.toNat a.isLt b.isLt
  simpa [Nat.add_comm] using this

/-- `int32` addition of a value and taking it away again. -/
theorem add_sub_cancel {s : Bool} {w : Nat} (a b : IntV s w) : a + b - a = b := by
  apply IntV.ext
  show (a.bits + b.bits) - a.bits = b.bits
  exact bv_add_sub w a.bits b.bits

/-- A masked add of `1` per counted lane: the sum of `1` over the lanes a predicate keeps is the
count of those lanes, as an `int32`. -/
theorem sum_map_ite_one {w : Nat} {ι : Type} (l : List ι) (P : ι → Bool) :
    (l.map (fun i => if P i then (1 : IntV true w) else 0)).sum = IntV.ofNat (l.countP P) := by
  induction l with
  | nil => rw [List.map_nil, List.sum_nil, List.countP_nil, ofNat_zero]
  | cons a t ih =>
    rw [List.map_cons, List.sum_cons, ih]
    by_cases h : P a = true
    · rw [ite_eq_left h, List.countP_cons_of_pos h]
      change (IntV.ofNat 1 + IntV.ofNat (List.countP P t) : IntV true w)
        = IntV.ofNat (List.countP P t + 1)
      rw [IntV.ofNat_add_ofNat]
      congr 1
      omega
    · rw [ite_eq_right h, List.countP_cons_of_neg h, zero_add]

/-! ## Counting the lanes of a program

The semantics enumerates the lanes of a block shape as `Idx.all [B]` (a `List`), the arithmetic
as `Finset.range B`. -/

/-- The lanes of `[B]` are the indices below `B`: filtering the list and filtering the range count
the same. -/
theorem length_filter_idx (B : Nat) (c : Nat → Bool) :
    ((Idx.all [B]).filter (fun i => c (i.1 : Nat))).length
      = ((Finset.range B).filter (fun l => c l)).card := by
  have h1 : ((Idx.all [B]).filter (fun i => c (i.1 : Nat))).length
      = ((Finset.univ : Finset (Idx [B])).filter
          (fun i => c (i.1 : Nat))).card := rfl
  rw [h1]
  have hsub : Subsingleton (Idx []) := ⟨fun a b => by cases a; cases b; rfl⟩
  have h2 : ((Finset.univ : Finset (Idx [B])).filter (fun i => c (i.1 : Nat))).card
      = ((Finset.univ : Finset (Fin B)).filter (fun i : Fin B => c i)).card := by
    have := hsub
    refine Finset.card_nbij (fun i : Idx [B] => (i.1 : Fin B)) ?_ ?_ ?_
    · intro i hi
      simpa using hi
    · intro i _ j _ h
      exact Prod.ext h (Subsingleton.elim _ _)
    · intro l hl
      have hl' : c l = true := by simpa using hl
      refine ⟨(fun x : Idx [B] => x) (l, ()), ?_, ?_⟩
      · rw [Finset.mem_coe, Finset.mem_filter]
        exact ⟨Finset.mem_univ _, by simpa using hl'⟩
      · rfl
  rw [h2]
  refine Finset.card_nbij (fun i : Fin B => (i : Nat)) ?_ ?_ ?_
  · intro i hi
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range, Finset.mem_univ,
      true_and] at hi ⊢
    exact ⟨i.isLt, hi⟩
  · intro i _ j _ h
    exact Fin.val_injective h
  · intro l hl
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range] at hl
    exact ⟨⟨l, hl.1⟩, by
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_univ, true_and]
      exact hl.2, rfl⟩

/-- Program `p`'s enabled lanes in bucket `b` are `progCount d p b` many. -/
theorem card_enabled_bucket {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) (b : Nat) :
    (((Idx.all [d.block]).filter (fun i =>
        (mask p d.block d.n).get i &&
          decide (bucketOf d (p * d.block + (i.1 : Nat)) = b))).length)
      = progCount d p b := by
  have hfit : Fits p d.block := fits hd hp
  have hn : d.n < 2 ^ 31 := by have := n_max_lt hd; omega
  set c : Nat → Bool := fun l =>
    decide (p * d.block + l < d.n ∧ bucketOf d (p * d.block + l) = b) with hcdef
  have hcongr : (Idx.all [d.block]).filter (fun i =>
        (mask p d.block d.n).get i &&
          decide (bucketOf d (p * d.block + (i.1 : Nat)) = b))
      = (Idx.all [d.block]).filter (fun i => c (i.1 : Nat)) := by
    refine List.filter_congr ?_
    intro i _
    obtain ⟨l, u⟩ := i
    cases u
    have hm : (mask p d.block d.n).get (l, ()) = decide (p * d.block + l < d.n) :=
      mask_get hfit hn l
    simp only [hcdef, hm, Bool.decide_and]
  rw [hcongr, length_filter_idx d.block c]
  unfold progCount
  congr 1

/-- The programs' pair ranges partition `[0, n)`, so the pairs in a bucket are the sum over the
programs of their lanes in it. -/
theorem pairsIn_eq_sum_progCount {d : Inst} (hd : d.ok) (b : Nat) :
    pairsIn d b = ∑ p : Fin (launch d).size, progCount d p b := by
  have hB : 0 < d.block := block_pos hd
  have hG : (launch d).size = grid d := launch_size d
  have hsum : (∑ p : Fin (launch d).size, progCount d p b)
      = ∑ p ∈ Finset.range (grid d), progCount d p b := by
    rw [hG]
    exact Fin.sum_univ_eq_sum_range (fun p => progCount d p b) (grid d)
  rw [hsum]
  unfold pairsIn
  have hfib : ∀ p ∈ Finset.range (grid d),
      (((Finset.range d.n).filter (fun j => bucketOf d j = b)).filter
          (fun j => j / d.block = p)).card = progCount d p b := by
    intro p hp
    unfold progCount
    refine Finset.card_nbij (fun j => j - p * d.block) ?_ ?_ ?_
    · intro j hj
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range] at hj
      obtain ⟨⟨hjn, hjb⟩, hjdiv⟩ := hj
      have h1 : p * d.block ≤ j := by
        have := Nat.div_mul_le_self j d.block
        rw [hjdiv] at this
        exact this
      have h2 : j < p * d.block + d.block := by
        have h2' : j < (p + 1) * d.block := by
          rw [← Nat.div_lt_iff_lt_mul hB]
          rw [hjdiv]
          omega
        rwa [Nat.add_mul, Nat.one_mul] at h2'
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range]
      refine ⟨?_, ?_, ?_⟩
      · omega
      · omega
      · have he : p * d.block + (j - p * d.block) = j := by omega
        rw [he]
        exact hjb
    · intro j1 hj1 j2 hj2 h
      simp only [Finset.mem_coe, Finset.mem_filter] at hj1 hj2
      obtain ⟨-, hd1⟩ := hj1
      obtain ⟨-, hd2⟩ := hj2
      have h1 : p * d.block ≤ j1 := by
        have := Nat.div_mul_le_self j1 d.block
        rw [hd1] at this
        exact this
      have h2 : p * d.block ≤ j2 := by
        have := Nat.div_mul_le_self j2 d.block
        rw [hd2] at this
        exact this
      change j1 - p * d.block = j2 - p * d.block at h
      omega
    · intro l hl
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range] at hl
      obtain ⟨hlB, hln, hlb⟩ := hl
      refine ⟨p * d.block + l, ?_, ?_⟩
      · simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range]
        refine ⟨⟨hln, hlb⟩, ?_⟩
        rw [Nat.mul_comm p d.block, Nat.mul_add_div hB, Nat.div_eq_of_lt hlB,
          Nat.add_zero]
      · change p * d.block + l - p * d.block = l
        omega
  rw [Finset.card_eq_sum_card_fiberwise (t := Finset.range (grid d)) ?_]
  · exact Finset.sum_congr rfl hfib
  · intro j hj
    simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_range] at hj
    simp only [Finset.mem_coe, Finset.mem_range]
    exact div_lt_grid hd hj.1

end Claims.RouteHistogram
