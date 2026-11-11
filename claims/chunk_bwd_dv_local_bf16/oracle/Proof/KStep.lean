import Claim
import TVA.Proof
import Proof.KDot
import Proof.ValsLane

/-!
# The `K` loop's invariant

After `t` trips the accumulator is, at every enabled pair `a ≤ b`, within
`σ·((ρ_K + 5u + 4u·t)·kqMagP + t·2^-114)` of `σ·kqP`, the sums over the first `min(t·BK, K)`
columns. `kinv_step` is one trip (the dot's error `Device.DotBF16` for the slab, the fused
multiply-add's error, `kstep`); `kinv_zero`, `kinv_final` the start and the end of the loop.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- **The invariant after `t` trips.** -/
def KInv (d : Inst) (init : Mem) (r bh t : Nat) (B : Tensor [d.BT, d.BT] F32) : Prop :=
  ∀ a b : Fin d.BT, en d r bh a = true → en d r bh b = true → a ≤ b →
    (B.get2 a b).isFinite = true ∧
      |(B.get2 a b).toReal - sig d * kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh)
          (min (t * d.BK) d.K)| ≤
        sig d * ((((d.BK : ℝ) + 2) / 2 ^ 22 + 5 / 2 ^ 24 + 4 / 2 ^ 24 * (t : ℝ)) *
          kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) +
          (t : ℝ) / 2 ^ 114)

/-- The slab sum is at most the slab's lifted magnitude. -/
theorem slab_abs_le {n : Nat} (x y : Fin n → ℚ) :
    |∑ k, x k * y k| ≤ ∑ k, Device.liftMag .bf16 (x k) * Device.liftMag .bf16 (y k) :=
  (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun _ _ => Device.abs_mul_le_liftMag _ _ _ _)

/-- **One trip keeps the invariant.** The dot of the loaded slab `X` (every result the platform
allows) and the fused multiply-add into the accumulator `B` give `r'`. -/
theorem kinv_step {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d) {t : Nat}
    (htK : t * d.BK < d.K) (ht15 : t ≤ 15)
    {bk : Tensor [d.BT, d.BK] BF16} {bq : Tensor [d.BK, d.BT] BF16}
    (hKv : KvalK d init r bh t bk) (hQv : KvalQ d init r bh t bq)
    {X : Tensor [d.BT, d.BT] F32} (hX : P.dot cfgBF16 bk bq (tl.zeros [d.BT, d.BT] .f32) X)
    {B r' : Tensor [d.BT, d.BT] F32} (hB : KInv d init r bh t B) {fusion : Bool}
    (hr' : ∀ a b : Fin d.BT,
      FloatV.NaNEq (FloatV.add (FloatV.mul (X.get2 a b) (FloatV.hostF32 d.scale)) (B.get2 a b))
        (r'.get2 a b) ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma (X.get2 a b) (FloatV.hostF32 d.scale) (B.get2 a b))
        (r'.get2 a b))) :
    KInv d init r bh (t + 1) r' := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  intro a b hae hbe hab
  obtain ⟨hBf, hBe⟩ := hB a b hae hbe hab
  obtain ⟨hs1, hs2⟩ := sig_bounds hd
  have hMg := sig_kqMag_le hd hT hV hr hbh a.isLt b.isLt hab hbe (hvOf_lt hd bh)
  -- the slab sums
  obtain ⟨hsum, hmag⟩ := kslab_sums htK hKv hQv hae hbe
  -- the slab magnitude and its room
  set Lq : ℚ := ∑ kk : Fin d.BK, Device.liftMag .bf16 (bk.get2 a kk).toRat *
    Device.liftMag .bf16 (bq.get2 kk b).toRat with hLq
  set Sq : ℚ := ∑ kk : Fin d.BK, (bk.get2 a kk).toRat * (bq.get2 kk b).toRat with hSq
  have hLnn : 0 ≤ Lq := Finset.sum_nonneg fun kk _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)
  have hn' : min ((t + 1) * d.BK) d.K ≤ d.K := min_le_right _ _
  have hMagK : kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) ≤
      kqMag d init (pos d r bh a) (pos d r bh b) (hvOf d bh) := by
    rw [kqMag_eq]; exact kqMagP_mono _ _ _ _ _ hn'
  have hMg2 : kqMag d init (pos d r bh a) (pos d r bh b) (hvOf d bh) ≤ 2 ^ 100 :=
    (vals_pair hd hT hV hr hbh a.isLt b.isLt hab hbe (hvOf_lt hd bh)).2.1
  have hMp0 := kqMagP_nonneg d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K)
  have hLR : (Lq : ℝ) = kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) -
      kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) := hmag
  have hSR : (Sq : ℝ) = kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) -
      kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) := hsum
  have hL100 : Lq ≤ 2 ^ 100 := by
    have : (Lq : ℝ) ≤ 2 ^ 100 := by rw [hLR]; linarith
    exact_mod_cast this
  have hBK := hd.bk_le
  have hroomQ : (1 + Device.relDot16 d.BK) * Lq + Device.absDot16 d.BK ≤ Format.binary32.maxVal := by
    have h1 : Device.relDot16 d.BK ≤ 1 := by
      unfold Device.relDot16
      have : ((d.BK : ℚ) + 2) ≤ 130 := by
        have : (d.BK : ℚ) ≤ 128 := by exact_mod_cast hBK
        linarith
      calc ((d.BK : ℚ) + 2) * 2 ^ (-22 : ℤ) ≤ 130 * 2 ^ (-22 : ℤ) :=
            mul_le_mul_of_nonneg_right this (by norm_num)
        _ ≤ 1 := by norm_num
    have h2 : Device.absDot16 d.BK ≤ 1 := by
      unfold Device.absDot16
      have : ((d.BK : ℚ) + 2) ≤ 130 := by
        have : (d.BK : ℚ) ≤ 128 := by exact_mod_cast hBK
        linarith
      calc ((d.BK : ℚ) + 2) * 2 ^ (-126 : ℤ) ≤ 130 * 2 ^ (-126 : ℤ) :=
            mul_le_mul_of_nonneg_right this (by norm_num)
        _ ≤ 1 := by norm_num
    have h0 : 0 ≤ Device.relDot16 d.BK := by unfold Device.relDot16; positivity
    rw [Format.binary32_maxVal]
    have : (1 + Device.relDot16 d.BK) * Lq ≤ 2 * 2 ^ 100 :=
      mul_le_mul (by linarith) hL100 hLnn (by norm_num)
    have h3 : (2 * 2 ^ 100 + 1 : ℚ) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by norm_num
    linarith
  obtain ⟨hXf, hXe⟩ := dot_lane hP hX a b (fun kk => (hKv a hae kk).1) (fun kk => (hQv b hbe kk).1) hroomQ
  -- the scale
  obtain ⟨hsc, h1, h2⟩ := hd.scale
  obtain ⟨hsf, hsg⟩ := scale_f32 hsc h1 h2
  have hsig : sig d = (d.scale.toRat : ℝ) := rfl
  -- the dot error in the reals
  have hρ : (((Device.relDot16 d.BK : ℚ) : ℝ)) = ((d.BK : ℝ) + 2) / 2 ^ 22 := by
    unfold Device.relDot16; push_cast; norm_num; ring
  have hα : (((Device.absDot16 d.BK : ℚ) : ℝ)) = ((d.BK : ℝ) + 2) / 2 ^ 126 := by
    unfold Device.absDot16; push_cast; norm_num; ring
  have hXeR : |(X.get2 a b).toReal - (Sq : ℝ)| ≤ ((d.BK : ℝ) + 2) / 2 ^ 22 * (Lq : ℝ) +
      ((d.BK : ℝ) + 2) / 2 ^ 126 := by
    have := (by exact_mod_cast hXe : |((X.get2 a b).toRat : ℝ) - (Sq : ℝ)| ≤
      ((Device.relDot16 d.BK : ℚ) : ℝ) * (Lq : ℝ) + ((Device.absDot16 d.BK : ℚ) : ℝ))
    rw [hρ, hα] at this
    exact this
  have hBKR : (d.BK : ℝ) ≤ 128 := by exact_mod_cast hBK
  have hBK0 : (0 : ℝ) ≤ d.BK := Nat.cast_nonneg _
  have hSabs : |(Sq : ℝ)| ≤ (Lq : ℝ) := by
    have := slab_abs_le (fun kk : Fin d.BK => (bk.get2 a kk).toRat) (fun kk => (bq.get2 kk b).toRat)
    exact_mod_cast this
  have hSp := abs_kqP_le d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K)
  have hMM : sig d * (kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) +
      (Lq : ℝ)) ≤ 2 ^ 100 := by
    have h3 : kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) +
        (Lq : ℝ) ≤ kqMag d init (pos d r bh a) (pos d r bh b) (hvOf d bh) := by
      rw [hLR]; linarith
    have h4 := (sig_kqMag_le hd hT hV hr hbh a.isLt b.isLt hab hbe (hvOf_lt hd bh))
    have hσ0 : 0 ≤ sig d := (hs1.trans' (by positivity))
    nlinarith [mul_le_mul_of_nonneg_left h3 hσ0]
  obtain ⟨hrf, hre⟩ := kstep (fusion := fusion) (hr' a b) hXf hsf hBf hs1 hs2 hsg
    (by exact_mod_cast hLnn) hSabs hMp0 hSp hMM ht15
    (by positivity) (by
      rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith)
    (by positivity) (by
      rw [div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith)
    hXeR hBe
  refine ⟨hrf, ?_⟩
  have e1 : kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) =
      kqP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) + (Sq : ℝ) := by
    rw [hSR]; ring
  have e2 : kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min ((t + 1) * d.BK) d.K) =
      kqMagP d init (pos d r bh a) (pos d r bh b) (hvOf d bh) (min (t * d.BK) d.K) + (Lq : ℝ) := by
    rw [hLR]; ring
  rw [e1, e2]
  push_cast
  exact hre

end Claims.ChunkBwdDvLocal
