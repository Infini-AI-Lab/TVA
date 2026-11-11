import Claim
import TVA.Proof
import Proof.Util
import Proof.NumEntry

/-!
# The exact quantities and the covered values

The partial sums of the exact `Σ_c k·q` and of its magnitude over the first `n` components
(`kqP`, `kqMagP`; `kq`, `kqMag` of the claim are the full sums), their elementary properties, and
the consequences of `ValsOk` the proof uses.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- `Σ_{c < n} k[p, h, c]·q[j, h, c]` for value head `hv`, exact. -/
noncomputable def kqP (d : Inst) (m : Mem) (p j hv n : Nat) : ℝ :=
  ∑ c ∈ Finset.range n, ((valB m d.k (qkIdx d p (hk d hv) c) : ℚ) : ℝ) *
    ((valB m d.q (qkIdx d j (hk d hv) c) : ℚ) : ℝ)

/-- Its magnitude `Σ_{c < n} liftMag k · liftMag q`. -/
noncomputable def kqMagP (d : Inst) (m : Mem) (p j hv n : Nat) : ℝ :=
  ∑ c ∈ Finset.range n,
    ((Device.liftMag .bf16 (valB m d.k (qkIdx d p (hk d hv) c)) *
      Device.liftMag .bf16 (valB m d.q (qkIdx d j (hk d hv) c)) : ℚ) : ℝ)

/-- The claim's `kq` is the partial sum over all `K` components. -/
theorem kq_eq (d : Inst) (m : Mem) (p j hv : Nat) : kq d m p j hv = kqP d m p j hv d.K := rfl

/-- The claim's `kqMag` is the partial magnitude over all `K` components. -/
theorem kqMag_eq (d : Inst) (m : Mem) (p j hv : Nat) :
    kqMag d m p j hv = kqMagP d m p j hv d.K := rfl

/-- A partial magnitude is nonnegative. -/
theorem kqMagP_nonneg (d : Inst) (m : Mem) (p j hv n : Nat) : 0 ≤ kqMagP d m p j hv n :=
  Finset.sum_nonneg fun c _ => by
    have := mul_nonneg (Device.liftMag_nonneg .bf16 (valB m d.k (qkIdx d p (hk d hv) c)))
      (Device.liftMag_nonneg .bf16 (valB m d.q (qkIdx d j (hk d hv) c)))
    exact_mod_cast this

/-- A partial magnitude grows with the number of components. -/
theorem kqMagP_mono (d : Inst) (m : Mem) (p j hv : Nat) {n n' : Nat} (h : n ≤ n') :
    kqMagP d m p j hv n ≤ kqMagP d m p j hv n' := by
  unfold kqMagP
  refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr h) ?_
  intro c _ _
  have := mul_nonneg (Device.liftMag_nonneg .bf16 (valB m d.k (qkIdx d p (hk d hv) c)))
    (Device.liftMag_nonneg .bf16 (valB m d.q (qkIdx d j (hk d hv) c)))
  exact_mod_cast this

/-- A partial sum is at most its partial magnitude. -/
theorem abs_kqP_le (d : Inst) (m : Mem) (p j hv n : Nat) :
    |kqP d m p j hv n| ≤ kqMagP d m p j hv n := by
  unfold kqP kqMagP
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun c _ => ?_)
  rw [← Rat.cast_mul, ← Rat.cast_abs]
  exact_mod_cast Device.abs_mul_le_liftMag .bf16 .bf16 _ _

/-- A partial sum over `min ((t+1)·BK) K` indices is the partial sum over `min (t·BK) K` plus the
turn's slab, with `g` standing for `f` shifted and padded by zeros. -/
theorem split_sum {K BK t : Nat} (ht : t * BK < K) (f : Nat → ℝ) (g : Nat → ℝ)
    (hg : ∀ kk, g kk = if t * BK + kk < K then f (t * BK + kk) else 0) :
    (∑ i ∈ Finset.range (min ((t + 1) * BK) K), f i) =
      (∑ i ∈ Finset.range (min (t * BK) K), f i) + ∑ kk ∈ Finset.range BK, g kk := by
  have h1 : t * BK ≤ K := le_of_lt ht
  have h2 : (t + 1) * BK = t * BK + BK := by rw [Nat.succ_mul]
  rw [min_eq_left h1]
  by_cases hle : (t + 1) * BK ≤ K
  · rw [min_eq_left hle, h2, Finset.sum_range_add]
    congr 1
    refine Finset.sum_congr rfl fun kk hkk => ?_
    have hlt : t * BK + kk < K := by
      have : kk < BK := Finset.mem_range.mp hkk
      omega
    rw [hg kk, ite_eq_left hlt]
  · rw [min_eq_right (le_of_lt (Nat.lt_of_not_le hle))]
    have hKM : K - t * BK ≤ BK := by
      have : K < t * BK + BK := by rw [← h2]; exact Nat.lt_of_not_le hle
      omega
    have hsplit : (∑ i ∈ Finset.range K, f i)
        = (∑ i ∈ Finset.range (t * BK), f i)
          + ∑ kk ∈ Finset.range (K - t * BK), f (t * BK + kk) := by
      have h := Finset.sum_range_add (f := f) (t * BK) (K - t * BK)
      rwa [Nat.add_sub_cancel' h1] at h
    rw [hsplit, add_right_inj]
    have hsub : (∑ kk ∈ Finset.range BK, g kk)
        = ∑ kk ∈ Finset.range (K - t * BK), g kk := by
      refine (Finset.sum_subset (Finset.range_subset_range.mpr hKM) ?_).symm
      intro kk hkkBK hkknot
      have hlt : K ≤ t * BK + kk := by
        have : ¬ kk < K - t * BK := fun hc => hkknot (Finset.mem_range.mpr hc)
        omega
      rw [hg kk, ite_eq_right (by omega)]
    rw [hsub]
    refine Finset.sum_congr rfl fun kk hkk => ?_
    have : kk < K - t * BK := Finset.mem_range.mp hkk
    rw [hg kk, ite_eq_left (by omega)]

end Claims.ChunkBwdDvLocal
