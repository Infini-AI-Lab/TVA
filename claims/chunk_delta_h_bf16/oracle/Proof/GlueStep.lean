import Proof.NumS
import Proof.NumV
import Proof.GlueTok
import Proof.GlueScale

/-!
# `chunk_delta_h_bf16`: one step of the state

`numS` read at the claim's quantities, for one value column: the new state is finite and within
`errStep` of the exact step.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The decay exponent of a token is at most `8` (the claim's `Vals`; `0` without the gate), so the
factor `Gam` is in `(0, 2^8]`. -/
theorem Gam_bounds {d : Inst} {init : Mem} (hV : Vals init d) {n hv t i : Nat} (hn : n < d.N)
    (hhv : hv < d.HV) (ht : t < nch d n) (hi : i < cl d n t) :
    0 < Gam init d n hv t i ∧ Gam init d n hv t i ≤ 2 ^ 8 := by
  have hd8 := (hV.2.2.2.2.2.1 n hv t hn hhv ht).2 i hi
  refine ⟨Real.rpow_pos_of_pos (by norm_num) _, ?_⟩
  have := two_rpow_le_zpow (x := dgv init d n hv t i) (n := 8) (by push_cast; exact hd8)
  simpa [Gam] using this

/-- **One step of the state, one column** (`numS`). The state entering the trip `Y` (finite on the
rows of `h[t]`, within `padB` on the padded rows), its stored rounding `xh`, the decayed state `S`
(`gt` the device's `exp2(g_last)` with the gate, `S = Y` without), the rounded gated values `vb`,
the `k` tile `kk` and the update `Yn` (`hupd`, from the dot) give the finiteness of `Yn` and
the step's bound. -/
theorem core_s {d : Inst} (hd : d.ok) {P : Platform} (hE : Device.Exp2F32 P) {init m5 : Mem}
    (hV : Vals init d) {n hv jv j t nb : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    (_hj : inBlk d jv j) (hSt : StOk m5 d n hv j t) (hnb : d.K ≤ 64 * nb) (hnb' : 64 * nb < d.K + 64)
    (Y Yn : ℕ → F32) (xh : ℕ → BF16)
    (hxh : ∀ c, c < d.K → rdB m5 d.h (hIdx d (cbase d n + t) hv c j) = some (xh c))
    (hxhc : ∀ c, c < d.K → FloatV.NaNEq (FloatV.cvt (Y c)) (xh c))
    (hYf : ∀ c, c < d.K → (Y c).isFinite = true)
    (hpad : ∀ c, d.K ≤ c → c < 64 * nb → (Y c).isFinite = true ∧ |(Y c).toReal| ≤ padB init d n hv t)
    (hpb : padB init d n hv t ≤ 2 ^ 100) (gt : F32) (S : ℕ → F32)
    (hgate : (d.useG = true ∧ ∃ bgl, rdF init d.g (gTok d n hv (BT * t + cl d n t - 1)) = some bgl ∧
        P.math .exp2 .binary32 bgl gt ∧ ∀ c, c < 64 * nb → FloatV.NaNEq (FloatV.mul (Y c) gt) (S c)) ∨
      (d.useG = false ∧ ∀ c, c < 64 * nb → S c = Y c))
    (vb : ℕ → BF16) (hvbf : ∀ i, i < 64 → (vb i).isFinite = true)
    (hvr : ∀ i, i < cl d n t → |vrefS init m5 d n hv j t i| ≤
      |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)
    (hZ : ∀ i, i < cl d n t → |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t ≤ 2 ^ 61)
    (hvb : ∀ i, i < cl d n t → |(vb i).toReal - Gam init d n hv t i * vrefS init m5 d n hv j t i| ≤
      2 ^ (-8 : ℤ) * (Gam init d n hv t i * vAbsA (vrefS init m5 d n hv j t i)
        (|uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)) + 2 ^ (-63 : ℤ))
    (kk : ℕ → ℕ → BF16)
    (hkr : ∀ i, i < cl d n t → ∀ c, c < d.K →
      rdB init d.k (kIdx d (rowOf d n) (startOf d n + (BT * t + i)) (kHead d hv) c) = some (kk i c))
    (hkz : ∀ i, i < 64 → ∀ c, c < 64 * nb → (cl d n t ≤ i ∨ d.K ≤ c) → ZB (kk i c))
    (ρ2 α2 : ℝ) (hρ2 : 0 ≤ ρ2 ∧ ρ2 ≤ 16 / 10 ^ 6) (hα2 : 0 ≤ α2 ∧ α2 ≤ 2 ^ (-110 : ℤ))
    (ρP αP : ℝ) (hρP : 0 ≤ ρP ∧ ρP ≤ rD + 3 * u32) (hαP : αP ≤ (1 + rD + 3 * u32) * aD + 4 * e32)
    (hupdP : ∀ c, d.K ≤ c → c < 64 * nb → (S c).isFinite = true →
      (∀ i, i < 64 → (vb i).isFinite = true) → |(S c).toReal| ≤ 2 ^ 120 →
      (Yn c).isFinite = true ∧ |(Yn c).toReal| ≤ (1 + ρP) * |(S c).toReal| + αP)
    (hupd : ∀ c, c < 64 * nb → (S c).isFinite = true →
      (∀ i, i < 64 → (kk i c).isFinite = true ∧ (vb i).isFinite = true) →
      |(S c).toReal| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i) ≤ 2 ^ 120 →
      (Yn c).isFinite = true ∧
        |(Yn c).toReal - ((S c).toReal + ∑ i ∈ Finset.range 64, (kk i c).toReal * (vb i).toReal)| ≤
          ρ2 * (|(S c).toReal| + ∑ i ∈ Finset.range 64, lmB (kk i c) * lmB (vb i)) + α2) :
    ((∀ c, c < 64 * nb → (Yn c).isFinite = true) ∧
      colNorm d.K (fun c => (Yn c).toReal -
        Step init m5 d n hv j t (fun c' => (Y c').toReal) c) ≤ errStep init m5 d n hv j t) ∧
    (∀ c, d.K ≤ c → c < 64 * nb → |(Yn c).toReal| ≤ padB init d n hv (t + 1)) := by
  have hK0 := hd.K_pos; have hK1 := hd.K_le
  obtain ⟨hStf, hStn⟩ := hSt
  have hcl : cl d n t ≤ 64 := by simp only [cl, BT]; omega
  have hxhf : ∀ c, c < d.K → (xh c).isFinite = true := by
    intro c hc
    obtain ⟨x, hx, hf⟩ := hStf c hc
    rw [hxh c hc] at hx; cases hx; exact hf
  have hHs : ∀ c, c < d.K → Hs m5 d n hv j t c = (xh c).toReal := by
    intro c hc; unfold Hs; rw [valB_of_rd (hxh c hc)]; rfl
  have hYb : ∀ c, c < d.K → |(Y c).toReal| ≤ 2 ^ 41 := by
    intro c hc
    refine Y_le (hYf c hc) (hxhc c hc) (hxhf c hc) ?_
    rw [← hHs c hc]
    exact (abs_le_colNorm (Hs m5 d n hv j t) hc).trans hStn
  have hY : ∀ c, c < d.K → (Y c).isFinite = true ∧
      |Hs m5 d n hv j t c - (Y c).toReal| ≤ 2 ^ (-8 : ℤ) * |(Y c).toReal| + 2 ^ (-134 : ℤ) := by
    intro c hc
    refine ⟨hYf c hc, ?_⟩
    rw [hHs c hc]
    exact cvt_bf16_err_of_finite (hYf c hc) (hxhc c hc) (hxhf c hc)
  have hYall : ∀ c, c < 64 * nb → (Y c).isFinite = true ∧ |(Y c).toReal| ≤ 2 ^ 100 := by
    intro c hc
    by_cases hcK : c < d.K
    · exact ⟨hYf c hcK, (hYb c hcK).trans (by norm_num)⟩
    · obtain ⟨h1, h2⟩ := hpad c (not_lt.mp hcK) hc
      exact ⟨h1, h2.trans hpb⟩
  -- the decay
  obtain ⟨gtR, hgam0, hgam8, herrE, hgt, hS⟩ : ∃ gtR : ℝ, 0 < gam init d n hv t ∧ gam init d n hv t ≤ 2 ^ 8 ∧
      (0 ≤ errE init d n hv t ∧ errE init d n hv t ≤ 2 ^ (-22 : ℤ) * gam init d n hv t + mu) ∧
      |gtR - gam init d n hv t| ≤ errE init d n hv t ∧
      (∀ c, c < 64 * nb → (S c).isFinite = true ∧
        |(S c).toReal - gtR * (Y c).toReal| ≤ u32 * |gtR * (Y c).toReal| + e32) := by
    rcases hgate with ⟨hG, bgl, hbgl, he, hmul⟩ | ⟨hG, hSY⟩
    · obtain ⟨gtf, g0, g8, hgt, e0, e1⟩ := gt_facts hd hE hV hn hhv ht hG hbgl he
      have hgtb : |gt.toReal| ≤ 2 ^ 9 := by
        have h1 := abs_sub_abs_le_abs_sub gt.toReal (gam init d n hv t)
        have hmu : mu ≤ 1 := by unfold mu; norm_num
        have h22 : (2 : ℝ) ^ (-22 : ℤ) * gam init d n hv t ≤ 1 := by
          calc (2 : ℝ) ^ (-22 : ℤ) * gam init d n hv t ≤ (2 : ℝ) ^ (-22 : ℤ) * 2 ^ 8 :=
                mul_le_mul_of_nonneg_left g8 (by positivity)
            _ ≤ 1 := by norm_num
        rw [abs_of_pos g0] at h1
        have h8' : gam init d n hv t ≤ 256 := by norm_num at g8; linarith
        have := abs_sub_abs_le_abs_sub gt.toReal (gam init d n hv t)
        linarith [abs_sub_comm gt.toReal (gam init d n hv t)]
      refine ⟨gt.toReal, g0, g8, ⟨e0, e1⟩, hgt, fun c hc => ?_⟩
      obtain ⟨hf, hb⟩ := hYall c hc
      refine scale_facts hf gtf ?_ (hmul c hc)
      rw [abs_mul]
      calc |(Y c).toReal| * |gt.toReal| ≤ 2 ^ 100 * 2 ^ 9 :=
            mul_le_mul hb hgtb (abs_nonneg _) (by positivity)
        _ ≤ 2 ^ 120 := by norm_num
    · have hg0 : glast init d n hv t = 0 := by unfold glast; simp [hG]
      have hgam1 : gam init d n hv t = 1 := by unfold gam; rw [hg0]; simp
      have herr0 : errE init d n hv t = 0 := by unfold errE; simp [hG]
      refine ⟨1, by rw [hgam1]; norm_num, by rw [hgam1]; norm_num, ⟨by rw [herr0], by
        rw [herr0, hgam1]; have : (0:ℝ) < mu := by unfold mu; positivity
        nlinarith [zpow_pos (by norm_num : (0:ℝ) < 2) (-22 : ℤ)]⟩, by rw [herr0, hgam1]; simp, fun c hc => ?_⟩
      rw [hSY c hc]
      refine ⟨(hYall c hc).1, ?_⟩
      simp only [one_mul, sub_self, abs_zero]
      unfold u32 e32; positivity
  have hkc : ∀ i, i < cl d n t → ∀ c, c < d.K → kc init d n hv t i c = (kk i c).toReal := by
    intro i hi c hc
    unfold kc kq
    rw [valB_of_rd (hkr i hi c hc)]; rfl
  have hk : ∀ i c, i < 64 → c < 64 * nb → (kk i c).isFinite = true ∧ |(kk i c).toReal| ≤ 2 ^ 16 := by
    intro i c hi hc
    by_cases h : i < cl d n t ∧ c < d.K
    · obtain ⟨x, hx, hf, hb⟩ := vals_k hd hV hn hhv (cl_lt h.1) h.2
      rw [hkr i h.1 c h.2] at hx; cases hx
      exact ⟨hf, by simpa [FloatV.toReal] using (show ((|(kk i c).toRat| : ℚ) : ℝ) ≤ ((2 ^ 16 : ℚ) : ℝ) from
        Rat.cast_le.2 hb)⟩
    · obtain ⟨h1, h2⟩ := hkz i hi c hc (by
        by_contra hh; push Not at hh; exact h ⟨hh.1, hh.2⟩)
      exact ⟨h1, by rw [h2]; norm_num⟩
  have hsK : ∀ x : ℕ → ℝ, colNorm d.K (fun c => ∑ i ∈ Finset.range (cl d n t), (kk i c).toReal * x i) ≤
      sKn init d n hv t * colNorm (cl d n t) x := by
    intro x
    have := specNorm_spec d.K (cl d n t) (fun a i => kc init d n hv t i a) x
    refine le_trans (le_of_eq ?_) this
    refine colNorm_congr fun c hc => Finset.sum_congr rfl fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [hkc i hi c hc]
  have hlm : ∀ i, i < cl d n t → ∀ c, c < d.K →
      (Device.liftMag .bf16 (kq init d n hv (BT * t + i) c) : ℝ) = lmB (kk i c) := by
    intro i hi c hc
    unfold kq; rw [valB_of_rd (hkr i hi c hc)]; rfl
  have hsKle : sKn init d n hv t ≤ kl init d n hv t := by
    unfold sKn kl
    refine (specNorm_le_frob _ _ _).trans ?_
    refine Real.sqrt_le_sqrt ?_
    rw [Finset.sum_comm]
    refine Finset.sum_le_sum fun i hi => Finset.sum_le_sum fun c hc => ?_
    rw [Finset.mem_range] at hi hc
    have h1 : |kc init d n hv t i c| ≤ (Device.liftMag .bf16 (kq init d n hv (BT * t + i) c) : ℝ) := by
      rw [hkc i hi c hc, hlm i hi c hc]; exact abs_toReal_le_lmB _
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) h1 2
  have hkl : kl init d n hv t = Real.sqrt (∑ i ∈ Finset.range (cl d n t), ∑ c ∈ Finset.range d.K,
      lmB (kk i c) ^ 2) := by
    unfold kl
    congr 1
    refine Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun c hc => ?_
    rw [Finset.mem_range] at hi hc
    rw [hlm i hi c hc]
  have hGN : colNorm (cl d n t) (fun i => Gam init d n hv t i * vAbs init m5 d n hv j t i) =
      colNorm (cl d n t) (fun i => Gam init d n hv t i * vAbsA (vrefS init m5 d n hv j t i)
        (|uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)) := rfl
  have key := numS hK0 hK1 hnb hnb' hcl Y (Hs m5 d n hv j t) hY hStn (padB init d n hv t) hpb hpad
    (gam init d n hv t) (errE init d n hv t) gtR hgam0 hgam8 herrE hgt S hS
    (fun i => Gam init d n hv t i) (fun i => vrefS init m5 d n hv j t i)
    (fun i => |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)
    (fun i hi => Gam_bounds hV hn hhv ht hi)
    (fun i hi => ⟨(abs_nonneg _).trans (hvr i hi), hZ i hi⟩) hvr vb hvbf hvb kk hk
    (fun i c hi hc h => (hkz i hi c hc h).2) ρ2 α2 hρ2 hα2 Yn (fun c hc hSf hroom => hupd c hc hSf (fun i hi => ⟨(hk i c hi hc).1, hvbf i hi⟩) hroom)
    (sKn init d n hv t) (kl init d n hv t)
    (colNorm (cl d n t) (fun i => Gam init d n hv t i * vAbsA (vrefS init m5 d n hv j t i)
        (|uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)))
    hsK hsKle hkl rfl
  refine ⟨⟨key.1, ?_⟩, ?_⟩
  swap
  · intro c hcK hc
    obtain ⟨hSf, hSb⟩ := hS c hc
    obtain ⟨hpf, hpbd⟩ := hpad c hcK hc
    have hgtb : |gtR| ≤ eB init d n hv t := by
      unfold eB
      have h1 := abs_sub_abs_le_abs_sub gtR (gam init d n hv t)
      rw [abs_of_pos hgam0] at h1
      linarith
    have hSbig : |(S c).toReal| ≤ 2 ^ 111 := by
      have hY100 : |(Y c).toReal| ≤ 2 ^ 100 := hpbd.trans hpb
      have hg : |gtR| ≤ 2 ^ 9 := by
        have hmu : mu ≤ 1 := by unfold mu; norm_num
        have h22 : (2 : ℝ) ^ (-22 : ℤ) * gam init d n hv t ≤ 1 := by
          calc (2 : ℝ) ^ (-22 : ℤ) * gam init d n hv t ≤ (2 : ℝ) ^ (-22 : ℤ) * 2 ^ 8 :=
                mul_le_mul_of_nonneg_left hgam8 (by positivity)
            _ ≤ 1 := by norm_num
        have h8' : gam init d n hv t ≤ 256 := by norm_num at hgam8; linarith
        unfold eB at hgtb
        linarith [herrE.2]
      have hu : u32 ≤ 1 := by unfold u32; norm_num
      have he : e32 ≤ 1 := by unfold e32; norm_num
      have hgy : |gtR * (Y c).toReal| ≤ 2 ^ 109 := by
        rw [abs_mul]
        calc |gtR| * |(Y c).toReal| ≤ 2 ^ 9 * 2 ^ 100 := mul_le_mul hg hY100 (abs_nonneg _) (by positivity)
          _ = 2 ^ 109 := by norm_num
      have := abs_sub_abs_le_abs_sub (S c).toReal (gtR * (Y c).toReal)
      have hu0 : 0 ≤ u32 := by unfold u32; positivity
      have h3 : u32 * |gtR * (Y c).toReal| ≤ |gtR * (Y c).toReal| :=
        mul_le_of_le_one_left (abs_nonneg _) hu
      have h4 : (1 : ℝ) ≤ 2 ^ 109 := by norm_num
      have h5 : (2 : ℝ) ^ 111 = 4 * 2 ^ 109 := by norm_num
      linarith
    obtain ⟨-, hYn⟩ := hupdP c hcK hc hSf hvbf (hSbig.trans (by norm_num))
    have := pad_arith hpbd hgtb hSb hYn hρP.1 hρP.2 hαP
    simpa [padB, eB] using this
  have e : colNorm d.K (fun c => (Yn c).toReal - Step init m5 d n hv j t (fun c' => (Y c').toReal) c) =
      colNorm d.K (fun c => (Yn c).toReal - (gam init d n hv t * (Y c).toReal +
        ∑ i ∈ Finset.range (cl d n t), (kk i c).toReal * Gam init d n hv t i *
          vrefS init m5 d n hv j t i)) := by
    refine colNorm_congr fun c hc => ?_
    unfold Step
    congr 2
    refine Finset.sum_congr rfl fun i hi => ?_
    rw [Finset.mem_range] at hi
    rw [hkc i hi c hc]
  rw [e]
  refine key.2.trans (le_of_eq ?_)
  unfold errStep Claims.ChunkDeltaH.hn
  rfl

end Claims.ChunkDeltaH
