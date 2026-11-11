import Proof.Defs

/-!
# Summing the per-lane counts

The kernel stores `Tensor.foldAll (· + ·) (0 : I32) acc`, a wrapping `int32` sum of the lanes.
This file proves that its `toInt` is the `Nat` sum of the lanes' counts, and that the lanes'
counts of expert `e` add up to `countOf d e`.
-/

namespace Claims.CountExpertTokens

open TVA

/-- A folding `int32` addition is the sum of the `toInt`s, when nothing overflows. -/
theorem foldl_add_toInt (l : List I32) (a : I32) (ha : 0 ≤ a.toInt)
    (hpos : ∀ x ∈ l, 0 ≤ x.toInt)
    (hbound : a.toInt + (l.map IntV.toInt).sum < 2 ^ 31) :
    (l.foldl (· + ·) a).toInt = a.toInt + (l.map IntV.toInt).sum := by
  induction l generalizing a with
  | nil => simp
  | cons x t ih =>
    have hx : 0 ≤ x.toInt := hpos x (List.mem_cons_self ..)
    have ht : ∀ y ∈ t, 0 ≤ y.toInt := fun y hy => hpos y (List.mem_cons_of_mem _ hy)
    have htsum : 0 ≤ (t.map IntV.toInt).sum := by
      apply List.sum_nonneg
      intro y hy
      obtain ⟨z, hz, rfl⟩ := List.mem_map.mp hy
      exact ht z hz
    simp only [List.foldl_cons, List.map_cons, List.sum_cons] at hbound ⊢
    have hxle : x.toInt + (t.map IntV.toInt).sum < 2 ^ 31 := by omega
    have hax : (a + x).toInt = a.toInt + x.toInt := by
      refine IntV.toInt_add rfl rfl ?_
      simp only [IntV.SFits]
      constructor <;> norm_num <;> omega
    rw [ih (a + x) (by rw [hax]; omega) ht (by rw [hax]; omega), hax]
    omega

/-- The lanes' counts of expert `e` over all positions add up to `countOf d e`. -/
theorem sum_cnt (d : Inst) (hd : d.ok) (e : Nat) :
    (∑ l ∈ Finset.range d.B, cnt d e (numTrips d) l) = countOf d e := by
  have hB := B_pos hd
  have hcard : (∑ l ∈ Finset.range d.B, cnt d e (numTrips d) l)
      = ((Finset.range d.B).sigma (fun l => (Finset.range (numTrips d)).filter
          (fun x => x * d.B + l < d.numel ∧ routed d (x * d.B + l) = (e : Int)))).card := by
    rw [Finset.card_sigma]
    rfl
  rw [hcard]
  refine Finset.card_bij' (fun p _ => p.2 * d.B + p.1)
    (fun q _ => (⟨q % d.B, q / d.B⟩ : Sigma (fun _ : Nat => Nat))) ?_ ?_ ?_ ?_
  · intro p hp
    rw [Finset.mem_sigma] at hp
    obtain ⟨-, hp2⟩ := hp
    rw [Finset.mem_filter] at hp2
    obtain ⟨-, hlt, hr⟩ := hp2
    exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hlt, hr⟩
  · intro q hq
    rw [Finset.mem_filter] at hq
    obtain ⟨hqmem, hr⟩ := hq
    rw [Finset.mem_range] at hqmem
    rw [Finset.mem_sigma]
    constructor
    · rw [Finset.mem_range]
      exact Nat.mod_lt _ hB
    · refine Finset.mem_filter.mpr ⟨Finset.mem_range.mpr ?_, ?_⟩
      · apply Nat.div_lt_of_lt_mul
        rw [Nat.mul_comm]
        exact lt_of_lt_of_le hqmem (numel_le_numTrips_mul hd)
      · show q / d.B * d.B + q % d.B < d.numel ∧
          routed d (q / d.B * d.B + q % d.B) = (e : Int)
        rw [Nat.div_add_mod' q d.B]
        exact ⟨hqmem, hr⟩
  · intro p hp
    obtain ⟨l, x⟩ := p
    rw [Finset.mem_sigma] at hp
    obtain ⟨hpl, -⟩ := hp
    rw [Finset.mem_range] at hpl
    have hmod : (x * d.B + l) % d.B = l := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hpl]
    have hdiv : (x * d.B + l) / d.B = x := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hpl, Nat.zero_add]
    rw [hmod, hdiv]
  · intro q _
    exact Nat.div_add_mod' q d.B

/-- The `int32` sum of the accumulator is the count of expert `e`. -/
theorem foldAll_toInt {d : Inst} (hd : d.ok) {e : Nat} {acc : Tensor [d.B] I32}
    (hacc : ∀ l : Fin d.B, (acc.get (l, ())).toInt = (cnt d e (numTrips d) l : Int)) :
    (Tensor.foldAll (· + ·) (0 : I32) acc).toInt = (countOf d e : Int) := by
  have hacc' : ∀ l : Fin d.B, (acc.get1 l).toInt = (cnt d e (numTrips d) l : Int) :=
    fun l => hacc l
  have hmap : (Idx.all [d.B]).map acc.get = (List.finRange d.B).map acc.get1 := by
    let g : Fin d.B → Idx [d.B] := fun i => (i, ())
    have hfun : (fun i => (Idx.all ([] : Shape)).map (fun j => ((i, j) : Idx [d.B])))
        = (fun i => ([g i] : List (Idx [d.B]))) := by
      funext i
      rfl
    have h2 : (List.finRange d.B).flatMap
          (fun i => (Idx.all ([] : Shape)).map (fun j => ((i, j) : Idx [d.B])))
        = (List.finRange d.B).flatMap (fun i => ([g i] : List (Idx [d.B]))) :=
      congrArg (fun f => List.flatMap f (List.finRange d.B)) hfun
    have h3 : (List.finRange d.B).flatMap (fun i => ([g i] : List (Idx [d.B])))
        = (List.finRange d.B).map g := List.map_eq_flatMap.symm
    have hidx : Idx.all [d.B] = (List.finRange d.B).map g :=
      Eq.trans (rfl : Idx.all [d.B] = (List.finRange d.B).flatMap
          (fun i => (Idx.all ([] : Shape)).map (fun j => ((i, j) : Idx [d.B])))) (Eq.trans h2 h3)
    exact Eq.trans (congrArg (fun l => l.map acc.get) hidx)
      (Eq.trans List.map_map
        (List.map_congr_left (fun i _ => (rfl : acc.get (g i) = acc.get1 i))))
  have hlist : Tensor.foldAll (· + ·) (0 : I32) acc
      = ((List.finRange d.B).map acc.get1).foldl (· + ·) 0 :=
    congrArg (fun l => l.foldl (· + ·) (0 : I32)) hmap
  have h0 : (0 : I32).toInt = 0 := by decide
  have hsum_eq : (((List.finRange d.B).map acc.get1).map IntV.toInt).sum
      = (countOf d e : Int) := by
    rw [List.map_map]
    simp only [Function.comp_def]
    rw [List.map_congr_left (fun i _ => hacc' i)]
    rw [show ((List.finRange d.B).map
          (fun i : Fin d.B => (cnt d e (numTrips d) (i : Nat) : Int))).sum
        = ∑ l ∈ Finset.range d.B, (cnt d e (numTrips d) l : Int) from by
      rw [Finset.sum_range]
      rw [← List.sum_toFinset _ (List.nodup_finRange d.B), List.toFinset_finRange]]
    rw [← Nat.cast_sum, sum_cnt d hd e]
  have hpos : ∀ x ∈ (List.finRange d.B).map acc.get1, 0 ≤ x.toInt := by
    intro x hx
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp hx
    rw [hacc' i]
    exact Int.natCast_nonneg _
  have hbound : (0 : I32).toInt
      + (((List.finRange d.B).map acc.get1).map IntV.toInt).sum < 2 ^ 31 := by
    rw [h0, hsum_eq]
    have h1 : countOf d e ≤ d.numel := by
      rw [countOf]
      exact (Finset.card_filter_le _ _).trans (le_of_eq (Finset.card_range d.numel))
    have h2 := numel_lt hd
    omega
  rw [hlist, foldl_add_toInt _ _ (by rw [h0]) hpos hbound, h0, hsum_eq]
  omega

end Claims.CountExpertTokens
