import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spec
import Proof.NumGate

/-!
# `ValsOk` at the lanes of a program

`ValsOk` is stated for tokens of a sequence; here it is read at the lanes of the program
`(r, bh)`: finite inputs at enabled lanes, the room bounds at enabled pairs `a ≤ b`, and the
output bound.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The host's scale as a real is in `[2^-32, 2^32]`. -/
theorem sig_bounds {d : Inst} (hd : d.ok) : 1 / 2 ^ 32 ≤ sig d ∧ sig d ≤ 2 ^ 32 := by
  obtain ⟨-, h1, h2⟩ := hd.scale
  unfold sig
  constructor
  · have := (Rat.cast_le (K := ℝ)).mpr h1
    push_cast at this
    exact this
  · have := (Rat.cast_le (K := ℝ)).mpr h2
    push_cast at this
    exact this

/-- The gate's upper bound `wUp` is nonnegative. -/
theorem wUp_nonneg (d : Inst) (m : Mem) (p j hv : Nat) : 0 ≤ wUp d m p j hv := by
  unfold wUp wt wErr
  by_cases hG : d.useG = true
  · simp only [hG, ite_true]
    have h1 : (0 : ℝ) ≤ (2 : ℝ) ^ dg d m p j hv := Real.rpow_nonneg (by norm_num) _
    have h2 := wErrR_ge (dg d m p j hv)
    unfold wErrR at h2
    linarith
  · simp [hG]

/-- `kqMag` is nonnegative and `|kq| ≤ kqMag`. -/
theorem kq_le_kqMag (d : Inst) (m : Mem) (p j hv : Nat) :
    0 ≤ kqMag d m p j hv ∧ |kq d m p j hv| ≤ kqMag d m p j hv :=
  ⟨by rw [kqMag_eq]; exact kqMagP_nonneg d m p j hv d.K,
   by rw [kq_eq, kqMag_eq]; exact abs_kqP_le d m p j hv d.K⟩

section
variable {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
  (hV : ValsOk d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
include hd hT hV hr hbh

/-- The inputs at an enabled lane are finite and in range. -/
theorem vals_tok {l : Nat} (hl : en d r bh l = true) :
    (∀ h c, h < d.H → c < d.K →
      finB init d.q (qkIdx d (pos d r bh l) h c) ∧ finB init d.k (qkIdx d (pos d r bh l) h c)) ∧
    (∀ hv e, hv < d.HV → e < d.V →
      finB init d.dO (oIdx d (pos d r bh l) hv e) ∧
        |valB init d.dO (oIdx d (pos d r bh l) hv e)| ≤ 2 ^ 100) ∧
    (d.useG = true → ∀ hv, hv < d.HV → finF init d.g (gIdx d (pos d r bh l) hv)) :=
  (hV (nOf d r bh) (nOf_lt hd hT hr hbh)).1 (pos d r bh l) (lo_le_pos d r bh l)
    (pos_lt_hi hT (nOf_lt hd hT hr hbh) hl)

omit hV in
/-- The token of an enabled lane `b` is below the chunk's end as seen from any lane `a`. -/
theorem pos_lt_chunkEnd {a b : Nat} (ha : a < d.BT) (hb : b < d.BT) (hbe : en d r bh b = true) :
    pos d r bh b < chunkEnd d (nOf d r bh) (pos d r bh a) := by
  unfold chunkEnd
  rw [chunkStart_pos hd r bh ha]
  refine lt_min ?_ (pos_lt_hi hT (nOf_lt hd hT hr hbh) hbe)
  unfold pos
  omega

/-- **The room at an enabled pair `a ≤ b`** (the second conjunct of `ValsOk`). -/
theorem vals_pair {a b : Nat} (ha : a < d.BT) (hb : b < d.BT) (hab : a ≤ b)
    (hbe : en d r bh b = true) {hv : Nat} (hhv : hv < d.HV) :
    (d.useG = true → dg d init (pos d r bh a) (pos d r bh b) hv ≤ 64) ∧
      kqMag d init (pos d r bh a) (pos d r bh b) hv ≤ 2 ^ 100 ∧
      sig d * kqMag d init (pos d r bh a) (pos d r bh b) hv *
        (1 + wUp d init (pos d r bh a) (pos d r bh b) hv) ≤ 2 ^ 100 :=
  ((hV (nOf d r bh) (nOf_lt hd hT hr hbh)).2.1 (pos d r bh a) (pos d r bh b) hv
    (lo_le_pos d r bh a) (by unfold pos; omega) (pos_lt_chunkEnd hd hT hr hbh ha hb hbe) hhv)

/-- **The output bound** (the third conjunct of `ValsOk`). -/
theorem vals_out {a : Nat} (hae : en d r bh a = true) {hv e : Nat} (hhv : hv < d.HV)
    (he : e < d.V) :
    ∑ j ∈ Finset.Ico (pos d r bh a) (chunkEnd d (nOf d r bh) (pos d r bh a)),
      sig d * kqMag d init (pos d r bh a) j hv * wUp d init (pos d r bh a) j hv *
        doMag d init j hv e ≤ 2 ^ 120 :=
  (hV (nOf d r bh) (nOf_lt hd hT hr hbh)).2.2 (pos d r bh a) hv e (lo_le_pos d r bh a)
    (pos_lt_hi hT (nOf_lt hd hT hr hbh) hae) hhv he

/-- **`σ·kqMag ≤ 2^100`** at an enabled pair. -/
theorem sig_kqMag_le {a b : Nat} (ha : a < d.BT) (hb : b < d.BT) (hab : a ≤ b)
    (hbe : en d r bh b = true) {hv : Nat} (hhv : hv < d.HV) :
    sig d * kqMag d init (pos d r bh a) (pos d r bh b) hv ≤ 2 ^ 100 := by
  obtain ⟨-, hMg, hroom⟩ := vals_pair hd hT hV hr hbh ha hb hab hbe hhv
  have hw := wUp_nonneg d init (pos d r bh a) (pos d r bh b) hv
  have hσ : 0 ≤ sig d := (sig_bounds hd).1.trans' (by positivity)
  have hM : 0 ≤ kqMag d init (pos d r bh a) (pos d r bh b) hv := by
    rw [kqMag_eq]; exact kqMagP_nonneg _ _ _ _ _ _
  have h1 : 0 ≤ sig d * kqMag d init (pos d r bh a) (pos d r bh b) hv := mul_nonneg hσ hM
  nlinarith [mul_nonneg h1 hw]

end

end Claims.ChunkBwdDvLocal
