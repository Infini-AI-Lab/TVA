import Proof.Basic

/-!
# The walk `firstDiff`

`firstDiff` walks from `p` while the draft equals the (truncated) target and while the position
is below `nOf`, and stops at the first disagreement.  These lemmas say where it stops.
-/

namespace Claims.RejectionGreedy

open TVA

/-- The walk never goes below its starting point. -/
theorem firstDiff_ge (d : Inst) (m : Mem) (r : Nat) :
    ∀ (fuel : Nat) (p : Int), p ≤ firstDiff d m r fuel p := by
  intro fuel
  induction fuel with
  | zero =>
    intro p
    rw [firstDiff.eq_1]
  | succ f ih =>
    intro p
    rw [firstDiff.eq_2]
    by_cases h : p < nOf d m r
    · simp only [h, ite_true]
      by_cases he : rd32 d.draft m (startOf d m r + p) = rdt64 d.target m (startOf d m r + p)
      · simp only [he, ite_true]
        exact le_trans (Int.le_add_one (le_refl p)) (ih (p + 1))
      · simp only [he, ite_false, le_refl]
    · simp only [h, ite_false, le_refl]

/-- The walk goes at most `fuel` steps past its starting point. -/
theorem firstDiff_le (d : Inst) (m : Mem) (r : Nat) :
    ∀ (fuel : Nat) (p : Int), firstDiff d m r fuel p ≤ p + (fuel : Int) := by
  intro fuel
  induction fuel with
  | zero =>
    intro p
    rw [firstDiff.eq_1]
    simp
  | succ f ih =>
    intro p
    rw [firstDiff.eq_2]
    by_cases h : p < nOf d m r
    · simp only [h, ite_true]
      by_cases he : rd32 d.draft m (startOf d m r + p) = rdt64 d.target m (startOf d m r + p)
      · simp only [he, ite_true]
        have := ih (p + 1)
        push_cast
        omega
      · simp only [he, ite_false]
        push_cast
        omega
    · simp only [h, ite_false]
      push_cast
      omega

/-- Below the walk's stop, the draft equals the target. -/
theorem firstDiff_eq (d : Inst) (m : Mem) (r : Nat) :
    ∀ (fuel : Nat) (p t : Int), p ≤ t → t < firstDiff d m r fuel p →
      rd32 d.draft m (startOf d m r + t) = rdt64 d.target m (startOf d m r + t) := by
  intro fuel
  induction fuel with
  | zero =>
    intro p t hpt ht
    rw [firstDiff.eq_1] at ht
    omega
  | succ f ih =>
    intro p t hpt ht
    rw [firstDiff.eq_2] at ht
    by_cases h : p < nOf d m r
    · simp only [h, ite_true] at ht
      by_cases he : rd32 d.draft m (startOf d m r + p) = rdt64 d.target m (startOf d m r + p)
      · simp only [he, ite_true] at ht
        rcases eq_or_lt_of_le hpt with rfl | hlt
        · exact he
        · exact ih (p + 1) t (by omega) ht
      · simp only [he, ite_false] at ht
        omega
    · simp only [h, ite_false] at ht
      omega

/-- If every position from `p` to `t` matches and `t` is below `nOf`, the walk goes past `t`. -/
theorem lt_firstDiff (d : Inst) (m : Mem) (r : Nat) :
    ∀ (fuel : Nat) (p t : Int), p ≤ t → t < p + (fuel : Int) → t < nOf d m r →
      (∀ s, p ≤ s → s ≤ t → rd32 d.draft m (startOf d m r + s) = rdt64 d.target m (startOf d m r + s)) →
      t < firstDiff d m r fuel p := by
  intro fuel
  induction fuel with
  | zero =>
    intro p t hpt htf htn hall
    rw [firstDiff.eq_1]
    simpa using htf
  | succ f ih =>
    intro p t hpt htf htn hall
    rw [firstDiff.eq_2]
    have hpn : p < nOf d m r := by omega
    have hprimary : rd32 d.draft m (startOf d m r + p) = rdt64 d.target m (startOf d m r + p) :=
      hall p le_rfl hpt
    rw [ite_eq_left (hpn), ite_eq_left (hprimary)]
    rcases eq_or_lt_of_le hpt with rfl | hlt
    · have := firstDiff_ge d m r f (p + 1)
      omega
    · refine ih (p + 1) t (by omega) ?_ htn ?_
      · omega
      · intro s hs hst
        exact hall s (by omega) hst

/-- `matchLen` is at most `nOf`. -/
theorem matchLen_le_nOf (d : Inst) (m : Mem) (r : Nat) (hn : 0 ≤ nOf d m r) :
    matchLen d m r ≤ nOf d m r := by
  have h := firstDiff_le d m r (nOf d m r).toNat 0
  simp only [zero_add] at h
  rwa [Int.toNat_of_nonneg hn] at h

/-- `matchLen` is not negative. -/
theorem matchLen_nonneg (d : Inst) (m : Mem) (r : Nat) : 0 ≤ matchLen d m r :=
  firstDiff_ge d m r (nOf d m r).toNat 0

/-- Below `matchLen` (and below `nOf`), the draft equals the target. -/
theorem matchLen_eq_of_lt (d : Inst) (m : Mem) (r : Nat) {t : Int} (h0 : 0 ≤ t)
    (ht : t < matchLen d m r) :
    rd32 d.draft m (startOf d m r + t) = rdt64 d.target m (startOf d m r + t) :=
  firstDiff_eq d m r (nOf d m r).toNat 0 t h0 ht

/-- At `matchLen` (`≤` and not below), below `nOf`, the draft differs from the target. -/
theorem matchLen_ne_of_lt (d : Inst) (m : Mem) (r : Nat) {t : Int} (h0 : 0 ≤ t)
    (ht : t < nOf d m r) (hk : t ≤ matchLen d m r) (hnot : ¬ t < matchLen d m r) :
    rd32 d.draft m (startOf d m r + t) ≠ rdt64 d.target m (startOf d m r + t) := by
  intro heq
  have hlt : t < firstDiff d m r (nOf d m r).toNat 0 := by
    refine lt_firstDiff d m r (nOf d m r).toNat 0 t h0 ?_ ht ?_
    · rw [Int.toNat_of_nonneg (le_trans h0 (le_of_lt ht))]
      simpa using ht
    · intro s hs hst
      rcases eq_or_lt_of_le hst with rfl | hlt2
      · exact heq
      · exact matchLen_eq_of_lt d m r hs (lt_of_lt_of_le hlt2 hk)
  exact hnot hlt

end Claims.RejectionGreedy
