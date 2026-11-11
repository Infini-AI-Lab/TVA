import Claim
import TVA.Proof
import Proof.KTrip
import Proof.NumDot
import Proof.NumKStep

/-!
# The dot of one trip of the `K` loop

The loaded tiles `b_k`, `b_q` of trip `t` hold the slab of `k` and `q` (`KvalK`, `KvalQ`), so the
dot at an enabled pair `a ≤ b` is, by `Device.DotBF16`, within `relDot16·slabMag + absDot16` of
the slab's exact sum `kqP(min((t+1)BK, K)) - kqP(min(t·BK, K))`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The slab sums: the dot product of row `a` of `b_k` and column `b` of `b_q`, and its lifted
magnitude, are the increments of `kqP` and `kqMagP`. -/
theorem kslab_sums {d : Inst} {init : Mem} {r bh t : Nat} (ht : t * d.BK < d.K)
    {bk : Tensor [d.BT, d.BK] BF16} {bq : Tensor [d.BK, d.BT] BF16}
    (hKv : KvalK d init r bh t bk) (hQv : KvalQ d init r bh t bq) {a b : Fin d.BT}
    (hae : en d r bh a = true) (hbe : en d r bh b = true) :
    ((∑ kk : Fin d.BK, ((bk.get2 a kk).toRat * (bq.get2 kk b).toRat : ℚ) : ℚ) : ℝ) =
        kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) -
          kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) ∧
      ((∑ kk : Fin d.BK, (Device.liftMag .bf16 (bk.get2 a kk).toRat *
          Device.liftMag .bf16 (bq.get2 kk b).toRat : ℚ) : ℚ) : ℝ) =
        kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) -
          kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) := by
  set f : Nat → ℝ := fun c =>
    ((valB init d.k (qkIdx d (pos d r bh a) (hk d (hvOf d bh)) c) : ℚ) : ℝ) *
      ((valB init d.q (qkIdx d (pos d r bh b) (hk d (hvOf d bh)) c) : ℚ) : ℝ) with hf
  set fm : Nat → ℝ := fun c =>
    ((Device.liftMag .bf16 (valB init d.k (qkIdx d (pos d r bh a) (hk d (hvOf d bh)) c)) *
      Device.liftMag .bf16 (valB init d.q (qkIdx d (pos d r bh b) (hk d (hvOf d bh)) c)) : ℚ) : ℝ)
    with hfm
  refine ⟨?_, ?_⟩
  · have hsplit := split_sum ht f (fun kk => if t * d.BK + kk < d.K then f (t * d.BK + kk) else 0)
      (fun kk => rfl)
    have : kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) =
        ∑ i ∈ Finset.range (min ((t + 1) * d.BK) d.K), f i := rfl
    have h2 : kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) =
        ∑ i ∈ Finset.range (min (t * d.BK) d.K), f i := rfl
    rw [this, h2, hsplit]
    rw [add_sub_cancel_left, ← Fin.sum_univ_eq_sum_range (fun kk => if t * d.BK + kk < d.K then f (t * d.BK + kk) else 0)]
    push_cast
    refine Finset.sum_congr rfl fun kk _ => ?_
    obtain ⟨-, h1⟩ := hKv a hae kk
    obtain ⟨-, h2⟩ := hQv b hbe kk
    rw [h1, h2]
    by_cases hkk : t * d.BK + kk.val < d.K
    · simp only [hkk, ite_true, hf]
    · simp [hkk]
  · have hsplit := split_sum ht fm (fun kk => if t * d.BK + kk < d.K then fm (t * d.BK + kk) else 0)
      (fun kk => rfl)
    have : kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) =
        ∑ i ∈ Finset.range (min ((t + 1) * d.BK) d.K), fm i := rfl
    have h2 : kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) =
        ∑ i ∈ Finset.range (min (t * d.BK) d.K), fm i := rfl
    rw [this, h2, hsplit]
    rw [add_sub_cancel_left, ← Fin.sum_univ_eq_sum_range (fun kk => if t * d.BK + kk < d.K then fm (t * d.BK + kk) else 0)]
    push_cast
    refine Finset.sum_congr rfl fun kk _ => ?_
    obtain ⟨-, h1⟩ := hKv a hae kk
    obtain ⟨-, h2⟩ := hQv b hbe kk
    rw [h1, h2]
    by_cases hkk : t * d.BK + kk.val < d.K
    · simp only [hkk, ite_true, hfm]; push_cast; rfl
    · simp [hkk]

end Claims.ChunkBwdDvLocal
