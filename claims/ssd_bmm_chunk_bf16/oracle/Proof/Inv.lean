import Claim
import TVA.Proof
import Proof.Peel

/-!
# The loop invariant

The kernel's main loop runs `steps d.K` times; turn `t` loads the `t`-th `BK`-wide slice of `K`
from `a` and `b` and accumulates it into the `[BM, BN]` tile. The invariant `inv` records

* the memory is the initial one (the loop's body only reads),
* lane `(mll, kk)` of the `a` tile is the pointer to element `aIdx d (rowA d i mll) (hOf d i)
  (t·BK + kk)` of `a`, and likewise for `b`, on the lanes whose row (resp. column) lies inside
  the chunk `[0, C)` — outside it the `int32` tile offset may wrap, and those lanes are always
  masked out,
* every accumulator lane whose row and column are inside the program's chunk is finite, within
  `DotLoop.bound` of the exact partial dot product, and below `2^121` in magnitude.

The numeric clause is restricted to the lanes inside the chunk because the `a` and `b` loads
mask out the rows, columns and `k`-lanes outside it, and the masked-out lanes load `+0`.
-/

open TVA
namespace Claims.SsdBmm

set_option maxHeartbeats 800000

/-- **The loop invariant** after `t` turns: the pointers of the two tiles, the accumulator's
error bound, and its magnitude. -/
def inv (d : Inst) (init : Mem) (i : Fin (launch d).size) : Nat → StT → Mem → Prop :=
  fun t st m =>
    m = init ∧
    (∀ (mll : Fin BM) (kk : Fin BK), mOf d i * BM + mll.val < d.C →
      st.1.get (mll, kk, ()) =
        d.a.ptr (aIdx d (rowA d i mll.val) (hOf d i) (t * BK + kk.val) : ℤ)) ∧
    (∀ (kk : Fin BK) (nn : Fin BN), nOf d i * BN + nn.val < d.C →
      st.2.1.get (kk, nn, ()) =
        d.b.ptr (bIdx d (colB d i nn.val) (hOf d i) (t * BK + kk.val) : ℤ)) ∧
    (∀ (mll : Fin BM) (nn : Fin BN), enM d i mll.val = true → enN d i nn.val = true →
      ∀ va vb : Nat → BF16,
        (∀ k, k < d.K →
          init.read .bf16 (d.a.ptr (aIdx d (rowA d i mll.val) (hOf d i) k : ℤ)).addr = some (va k) ∧
            (va k).isFinite = true) →
        (∀ k, k < d.K →
          init.read .bf16 (d.b.ptr (bIdx d (colB d i nn.val) (hOf d i) k : ℤ)).addr = some (vb k) ∧
            (vb k).isFinite = true) →
        dotMag va vb (min (t * BK) d.K) ≤ 2 ^ 120 →
          (st.2.2.get2 mll nn).isFinite = true ∧
          |(st.2.2.get2 mll nn).toRat - dotSum va vb (min (t * BK) d.K)| ≤
            DotLoop.bound rho alpha t (dotMag va vb (min (t * BK) d.K)) ∧
          |(st.2.2.get2 mll nn).toRat| ≤ 2 ^ 121)

/-! ## Monotonicity of the exact sums -/

/-- The empty dot product is `0`. -/
theorem dotSum_zero (va vb : Nat → BF16) : dotSum va vb 0 = 0 := by simp [dotSum]

/-- `dotMag` is nonnegative. -/
theorem dotMag_nonneg (va vb : Nat → BF16) (n : Nat) : 0 ≤ dotMag va vb n := by
  unfold dotMag
  exact Finset.sum_nonneg fun k _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)

/-- `dotMag` grows with the number of terms. -/
theorem dotMag_mono {va vb : Nat → BF16} {n m : Nat} (h : n ≤ m) :
    dotMag va vb n ≤ dotMag va vb m := by
  unfold dotMag
  refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.mpr h) ?_
  intro k _ _
  exact mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)

/-- The exact dot product is bounded by the sum of the lifted magnitudes. -/
theorem abs_dotSum_le_dotMag (va vb : Nat → BF16) (n : Nat) :
    |dotSum va vb n| ≤ dotMag va vb n := by
  unfold dotSum dotMag
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  exact Finset.sum_le_sum fun k _ => Device.abs_mul_le_liftMag _ _ _ _

/-- Under `dotMag ≤ 2^120` the exact dot product is at most `2^120`. -/
theorem abs_dotSum_le (va vb : Nat → BF16) {n : Nat} (h : dotMag va vb n ≤ 2 ^ 120) :
    |dotSum va vb n| ≤ 2 ^ 120 := le_trans (abs_dotSum_le_dotMag va vb n) h

/-! ## Splitting a partial sum at the step's boundary -/

/-- The `k`-values of turn `t`'s slice of a load: the value at `t·BK + kk`, or `+0` past `K`. -/
def kex (d : Inst) (t : Nat) (v : Nat → BF16) : Nat → BF16 :=
  fun kk => if t * BK + kk < d.K then v (t * BK + kk) else FloatV.ofPat 0

/-- Inside `K`, the extended sequence `kex` reads the value itself. -/
theorem kex_pos {d : Inst} {t : Nat} {v : Nat → BF16} {kk : Nat} (h : t * BK + kk < d.K) :
    (kex d t v kk).toRat = (v (t * BK + kk)).toRat := by
  simp only [kex, ite_eq_left h]

/-- Outside `K`, the extended sequence `kex` is `0` (a masked-out lane loads `0`). -/
theorem kex_neg {d : Inst} {t : Nat} {v : Nat → BF16} {kk : Nat} (h : ¬ t * BK + kk < d.K) :
    (kex d t v kk).toRat = 0 := by
  simp only [kex, ite_eq_right h]
  exact bf16_zero_toRat

/-- A partial sum of `f` over the first `min ((t+1)·BK) K` indices is the partial sum over the
first `t·BK` indices plus the turn's slice, with `g` standing for `f` shifted and padded by
zeros. -/
theorem sum_range_split {K t : Nat} (ht : t * BK < K) (f g : Nat → ℚ)
    (hg : ∀ kk, g kk = if t * BK + kk < K then f (t * BK + kk) else 0) :
    (∑ i ∈ Finset.range (min ((t + 1) * BK) K), f i)
      = (∑ i ∈ Finset.range (t * BK), f i) + ∑ kk ∈ Finset.range BK, g kk := by
  have h1 : t * BK ≤ K := le_of_lt ht
  have h2 : (t + 1) * BK = t * BK + BK := by rw [Nat.succ_mul]
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
    have hz : (∑ kk ∈ Finset.range (K - t * BK), f (t * BK + kk))
        = ∑ kk ∈ Finset.range BK, g kk := by
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
    exact hz

/-- The dot product of the first `min((t+1)·32, K)` terms is that of the first `t·32` plus the block `t`. -/
theorem dotSum_split {d : Inst} {t : Nat} (ht : t * BK < d.K) (va vb : Nat → BF16) :
    dotSum va vb (min ((t + 1) * BK) d.K)
      = dotSum va vb (t * BK) + dotSum (kex d t va) (kex d t vb) BK := by
  unfold dotSum
  refine sum_range_split ht (fun i => (va i).toRat * (vb i).toRat)
    (fun kk => (kex d t va kk).toRat * (kex d t vb kk).toRat) ?_
  intro kk
  by_cases h : t * BK + kk < d.K
  · rw [kex_pos h, kex_pos h, ite_eq_left h]
  · simp only [kex_neg h, ite_eq_right h, mul_zero]

/-- The same split for the lifted magnitudes. -/
theorem dotMag_split {d : Inst} {t : Nat} (ht : t * BK < d.K) (va vb : Nat → BF16) :
    dotMag va vb (min ((t + 1) * BK) d.K)
      = dotMag va vb (t * BK) + dotMag (kex d t va) (kex d t vb) BK := by
  unfold dotMag
  refine sum_range_split ht
    (fun i => Device.liftMag .bf16 (va i).toRat * Device.liftMag .bf16 (vb i).toRat)
    (fun kk => Device.liftMag .bf16 (kex d t va kk).toRat *
      Device.liftMag .bf16 (kex d t vb kk).toRat) ?_
  intro kk
  by_cases h : t * BK + kk < d.K
  · rw [kex_pos h, kex_pos h, ite_eq_left h]
  · simp only [kex_neg h, ite_eq_right h, Device.liftMag_zero, mul_zero]

/-- At the end of the loop the partial sums run to `K`. -/
theorem min_steps {d : Inst} : min (steps d.K * BK) d.K = d.K := by
  have hBK : BK = 32 := rfl
  have h : d.K ≤ steps d.K * BK := by
    unfold steps
    rw [hBK]
    have h1 := Nat.div_add_mod (d.K + 32 - 1) 32
    have h2 := Nat.mod_lt (d.K + 32 - 1) (by norm_num : 0 < 32)
    omega
  exact Nat.min_eq_right h

end Claims.SsdBmm



