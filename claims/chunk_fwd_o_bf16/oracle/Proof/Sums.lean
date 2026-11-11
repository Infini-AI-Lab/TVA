import Claim
import TVA.Proof
import Proof.Defs

/-!
# The padded partial sums

Facts about `SQ`, `MQ`, `SA`, `MA`: a trip's slice is the difference of two partial sums, the sums
past `K` are the claim's `QS`, `PQS`, `Aqk`, `PA`, a partial sum is at most its lifted magnitude,
and a lifted magnitude only grows.
-/

namespace Claims.ChunkFwdO

open TVA

/-- Past `K`, `qPad` is zero. -/
theorem qPad_ge {d : Inst} {t j x : Nat} (h : d.K ≤ x) : qPad d t j x = 0 := by
  simp [qPad, Nat.not_lt.mpr h]

/-- Past `K`, `kPad` is zero. -/
theorem kPad_ge {d : Inst} {s j x : Nat} (h : d.K ≤ x) : kPad d s j x = 0 := by
  simp [kPad, Nat.not_lt.mpr h]

/-- Past `K`, `hPad` is zero. -/
theorem hPad_ge {d : Inst} {r j e x : Nat} (h : d.K ≤ x) : hPad d r j e x = 0 := by
  simp [hPad, Nat.not_lt.mpr h]

/-- Below `K`, `qPad` is `qAt`. -/
theorem qPad_lt {d : Inst} {t j x : Nat} (h : x < d.K) : qPad d t j x = qAt d t (qh d j) x := by
  simp [qPad, h]

/-- Below `K`, `kPad` is `kAt`. -/
theorem kPad_lt {d : Inst} {s j x : Nat} (h : x < d.K) : kPad d s j x = kAt d s (qh d j) x := by
  simp [kPad, h]

/-- Below `K`, `hPad` is `hAt`. -/
theorem hPad_lt {d : Inst} {r j e x : Nat} (h : x < d.K) : hPad d r j e x = hAt d r j x e := by
  simp [hPad, h]

/-! ### Splitting at a trip -/

theorem SQ_split (d : Inst) (r t j e n m : Nat) :
    SQ d r t j e (n + m) = SQ d r t j e n +
      ∑ k ∈ Finset.range m, qPad d t j (n + k) * hPad d r j e (n + k) := by
  unfold SQ; exact Finset.sum_range_add _ n m

/-- `MQ` over `n + m` terms is `MQ` over `n` plus the sum of the next `m` products. -/
theorem MQ_split (d : Inst) (r t j e n m : Nat) :
    MQ d r t j e (n + m) = MQ d r t j e n +
      ∑ k ∈ Finset.range m, L (qPad d t j (n + k)) * L (hPad d r j e (n + k)) := by
  unfold MQ; exact Finset.sum_range_add _ n m

/-- `SA` over `n + m` terms is `SA` over `n` plus the sum of the next `m` products. -/
theorem SA_split (d : Inst) (t s j n m : Nat) :
    SA d t s j (n + m) = SA d t s j n +
      ∑ k ∈ Finset.range m, qPad d t j (n + k) * kPad d s j (n + k) := by
  unfold SA; exact Finset.sum_range_add _ n m

/-- `MA` over `n + m` terms is `MA` over `n` plus the sum of the next `m` products. -/
theorem MA_split (d : Inst) (t s j n m : Nat) :
    MA d t s j (n + m) = MA d t s j n +
      ∑ k ∈ Finset.range m, L (qPad d t j (n + k)) * L (kPad d s j (n + k)) := by
  unfold MA; exact Finset.sum_range_add _ n m

/-! ### Past `K` -/

theorem SQ_eq_QS {d : Inst} (r t j e : Nat) {n : Nat} (h : d.K ≤ n) :
    SQ d r t j e n = QS d r t j e := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [SQ_split]
  have : ∑ k ∈ Finset.range m, qPad d t j (d.K + k) * hPad d r j e (d.K + k) = 0 :=
    Finset.sum_eq_zero fun k _ => by rw [qPad_ge (by omega)]; simp
  rw [this, add_zero]
  unfold SQ QS
  exact Finset.sum_congr rfl fun x hx => by
    have := Finset.mem_range.mp hx
    rw [qPad_lt this, hPad_lt this]

/-- Once `n ≥ K`, `MQ` over `n` terms equals `PQS`. -/
theorem MQ_eq_PQS {d : Inst} (r t j e : Nat) {n : Nat} (h : d.K ≤ n) :
    MQ d r t j e n = PQS d r t j e := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [MQ_split]
  have : ∑ k ∈ Finset.range m, L (qPad d t j (d.K + k)) * L (hPad d r j e (d.K + k)) = 0 :=
    Finset.sum_eq_zero fun k _ => by
      rw [qPad_ge (by omega)]; simp [L]
  rw [this, add_zero]
  unfold MQ PQS
  exact Finset.sum_congr rfl fun x hx => by
    have := Finset.mem_range.mp hx
    rw [qPad_lt this, hPad_lt this]

/-- Once `n ≥ K`, `SA` over `n` terms equals `Aqk`. -/
theorem SA_eq_Aqk {d : Inst} (t s j : Nat) {n : Nat} (h : d.K ≤ n) :
    SA d t s j n = Aqk d t s j := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [SA_split]
  have : ∑ k ∈ Finset.range m, qPad d t j (d.K + k) * kPad d s j (d.K + k) = 0 :=
    Finset.sum_eq_zero fun k _ => by rw [qPad_ge (by omega)]; simp
  rw [this, add_zero]
  unfold SA Aqk
  exact Finset.sum_congr rfl fun x hx => by
    have := Finset.mem_range.mp hx
    rw [qPad_lt this, kPad_lt this]

/-- Once `n ≥ K`, `MA` over `n` terms equals `PA`. -/
theorem MA_eq_PA {d : Inst} (t s j : Nat) {n : Nat} (h : d.K ≤ n) :
    MA d t s j n = PA d t s j := by
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [MA_split]
  have : ∑ k ∈ Finset.range m, L (qPad d t j (d.K + k)) * L (kPad d s j (d.K + k)) = 0 :=
    Finset.sum_eq_zero fun k _ => by
      rw [qPad_ge (by omega)]; simp [L]
  rw [this, add_zero]
  unfold MA PA
  exact Finset.sum_congr rfl fun x hx => by
    have := Finset.mem_range.mp hx
    rw [qPad_lt this, kPad_lt this]

/-! ### Magnitudes -/

theorem L_nonneg (x : ℚ) : 0 ≤ L x := Device.liftMag_nonneg _ _

/-- `|SQ|` is at most `MQ`. -/
theorem abs_SQ_le (d : Inst) (r t j e n : Nat) : |SQ d r t j e n| ≤ MQ d r t j e n :=
  (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun _ _ => Device.abs_mul_le_liftMag _ _ _ _)

/-- `|SA|` is at most `MA`. -/
theorem abs_SA_le (d : Inst) (t s j n : Nat) : |SA d t s j n| ≤ MA d t s j n :=
  (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun _ _ => Device.abs_mul_le_liftMag _ _ _ _)

/-- `MQ` is monotone in the number of terms. -/
theorem MQ_mono (d : Inst) (r t j e : Nat) {n m : Nat} (h : n ≤ m) :
    MQ d r t j e n ≤ MQ d r t j e m :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr h)
    fun _ _ _ => mul_nonneg (L_nonneg _) (L_nonneg _)

/-- `MA` is monotone in the number of terms. -/
theorem MA_mono (d : Inst) (t s j : Nat) {n m : Nat} (h : n ≤ m) :
    MA d t s j n ≤ MA d t s j m :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr h)
    fun _ _ _ => mul_nonneg (L_nonneg _) (L_nonneg _)

end Claims.ChunkFwdO
