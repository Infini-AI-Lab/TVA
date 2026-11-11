import Proof.NumV
import Proof.NumG
import Proof.GlueAux

/-!
# `chunk_delta_h_bf16`: the new value of one token

`numV` read at the claim's quantities: the stored `v_new` value of token `i` of chunk `t`, column
`j`, is finite and within `tolV` of `vrefS`, and the difference before the rounding is within
`0.6·2^-13·Z + 2^-110` of it, `Z = |u| + ‖lmB w‖·‖lmB h[t]‖`.
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- **One token's new value.** The token's `w` row `w c`, the stored state `hs c = h[t][c, j]`
(read from `m5`), the `u` element `bu`, the dot chain `r` over `nb` blocks, the fusable
difference `bv = uf - r nb` and its rounding `x`. -/
theorem tok_v {d : Inst} (hd : d.ok) {init m5 : Mem} (hV : Vals init d) {n hv j t i nb : Nat}
    (hn : n < d.N) (hhv : hv < d.HV) (hi : i < cl d n t) (hj : j < d.V)
    (hnb : d.K ≤ 64 * nb) (hnb' : 64 * nb < d.K + 64) {P : Platform} {fusion : Bool}
    (w hs : ℕ → BF16) (r : ℕ → F32) (bu x : BF16) (uf bv : F32)
    (hwr : ∀ c, c < d.K →
      rdB init d.w (wIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv c) = some (w c))
    (hwz : ∀ c, d.K ≤ c → c < 64 * nb → (w c).toReal = 0 ∧ (w c).isFinite = true)
    (hhs : ∀ c, c < d.K → rdB m5 d.h (hIdx d (cbase d n + t) hv c j) = some (hs c))
    (hhsf : ∀ c, c < 64 * nb → (hs c).isFinite = true)
    (hhn : colNorm d.K (Hs m5 d n hv j t) ≤ 2 ^ 40)
    (hbu : rdB init d.u (uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv j) = some bu)
    (huf : ConvRel P .bf16 .f32 bu uf)
    (hr0 : (r 0).isFinite = true ∧ (r 0).toReal = 0)
    (hstep : ∀ b, b < nb → (r b).isFinite = true →
      |(r b).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c) ≤ 2 ^ 120 →
      (r (b + 1)).isFinite = true ∧
        |(r (b + 1)).toReal - ((r b).toReal +
            ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), (w c).toReal * (hs c).toReal)| ≤
          rD * (|(r b).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (w c) * lmB (hs c)) + aD)
    (hsub : AddRel fusion uf (FloatV.neg (r nb)) bv) (hx : FloatV.NaNEq (FloatV.cvt bv) x) :
    |vrefS init m5 d n hv j t i| ≤ |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t ∧
    |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t ≤ 2 ^ 61 ∧
    bv.isFinite = true ∧
    |bv.toReal - vrefS init m5 d n hv j t i| ≤ (3 / 5) * 2 ^ (-13 : ℤ) *
      (|uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t) + 2 ^ (-110 : ℤ) ∧
    x.isFinite = true ∧
    |x.toReal - vrefS init m5 d n hv j t i| ≤ tolV init m5 d n hv j t i := by
  have hK0 := hd.K_pos; have hK1 := hd.K_le
  have htok : BT * t + i < lenOf d n := cl_lt hi
  -- the values read
  have hwr' : ∀ c, c < d.K → (w c).isFinite = true ∧ |(w c).toReal| ≤ 2 ^ 16 := by
    intro c hc
    obtain ⟨x, hx, hf, hb⟩ := vals_w hd hV hn hhv htok hc
    rw [hwr c hc] at hx
    cases hx
    exact ⟨hf, by simpa [FloatV.toReal] using (show ((|(w c).toRat| : ℚ) : ℝ) ≤ ((2 ^ 16 : ℚ) : ℝ) from
      Rat.cast_le.2 hb)⟩
  obtain ⟨bx, hbx, hbf, hbb⟩ := vals_u hd hV hn hhv htok hj
  rw [hbu] at hbx
  cases hbx
  obtain ⟨-, hufin, -, hufr⟩ := ConvRel.bf16_f32 huf hbf
  have hu : uf.isFinite = true ∧ |uf.toReal| ≤ 2 ^ 16 := by
    refine ⟨hufin, ?_⟩
    rw [hufr]
    simpa [FloatV.toReal] using (show ((|bu.toRat| : ℚ) : ℝ) ≤ ((2 ^ 16 : ℚ) : ℝ) from Rat.cast_le.2 hbb)
  have hHs : ∀ c, c < d.K → Hs m5 d n hv j t c = (hs c).toReal := by
    intro c hc
    unfold Hs
    rw [valB_of_rd (hhs c hc)]; rfl
  have hhn' : colNorm d.K (fun c => (hs c).toReal) ≤ 2 ^ 40 := by
    rw [← colNorm_congr (fun c hc => hHs c hc)]; exact hhn
  have key := numV hK0 hK1 hnb hnb' w hs (r := r) uf bv x (fusion := fusion)
    (fun c hc => by
      by_cases hcK : c < d.K
      · exact hwr' c hcK
      · obtain ⟨h0, h1⟩ := hwz c (not_lt.mp hcK) hc
        exact ⟨h1, by rw [h0]; norm_num⟩)
    (fun c hcK hc => (hwz c hcK hc).1) hhsf hhn' hu hr0 hstep hsub hx
  obtain ⟨k1, k2, k3, k4, k5, k6⟩ := key
  have hwq : ∀ c, c < d.K → wq init d n hv (BT * t + i) c = (w c).toRat := by
    intro c hc
    unfold wq
    exact valB_of_rd (hwr c hc)
  have huq : uq init d n hv (BT * t + i) j = bu.toRat := by
    unfold uq; exact valB_of_rd hbu
  have hvref : vrefS init m5 d n hv j t i =
      uf.toReal - ∑ c ∈ Finset.range d.K, (w c).toReal * (hs c).toReal := by
    unfold vrefS uc wc
    rw [huq, hufr]
    congr 1
    refine Finset.sum_congr rfl fun c hc => ?_
    rw [Finset.mem_range] at hc
    rw [hwq c hc, hHs c hc]; rfl
  have hZ : |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t =
      |uf.toReal| + colNorm d.K (fun c => lmB (w c)) * colNorm d.K (fun c => lmB (hs c)) := by
    have e1 : uc init d n hv t i j = uf.toReal := by
      unfold uc; rw [huq, hufr]; rfl
    have e2 : wl init d n hv t i = colNorm d.K (fun c => lmB (w c)) := by
      unfold wl
      refine colNorm_congr fun c hc => ?_
      rw [hwq c hc]; rfl
    have e3 : hl m5 d n hv j t = colNorm d.K (fun c => lmB (hs c)) := by
      unfold hl
      refine colNorm_congr fun c hc => ?_
      rw [valB_of_rd (hhs c hc)]; rfl
    rw [e1, e2, e3]
  rw [hvref, hZ]
  refine ⟨k1, k2, k3, k4, k5, ?_⟩
  unfold tolV vAbs
  rw [hvref, hZ]
  simpa [vAbsA] using k6


/-- **One token through the gate.** With the gate, the value stored by the dot is `bv` (within
`0.6·2^-13·Z + 2^-110` of `vref`), multiplied by the device's `exp2(g_last - g)` and rounded to
`bfloat16`: finite and within `2^-8·2^dg·vAbs + 2^-63` of `2^dg·vref` (`numG`). -/
theorem tok_g {d : Inst} (hd : d.ok) {P : Platform} (hP : Device.Exp2F32 P) {init m5 : Mem}
    (hV : Vals init d) {n hv j t i : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    (hi : i < cl d n t) (hG : d.useG = true) (bgl bg xs e p bv : F32) (vb : BF16)
    (hbgl : rdF init d.g (gTok d n hv (BT * t + cl d n t - 1)) = some bgl)
    (hbg : rdF init d.g (gTok d n hv (BT * t + i)) = some bg)
    (hvr : |vrefS init m5 d n hv j t i| ≤ |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)
    (hZ : |uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t ≤ 2 ^ 61)
    (hbv : bv.isFinite = true ∧ |bv.toReal - vrefS init m5 d n hv j t i| ≤ (3 / 5) * 2 ^ (-13 : ℤ) *
      (|uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t) + 2 ^ (-110 : ℤ))
    (hxs : FloatV.NaNEq (FloatV.sub bgl bg) xs) (he : P.math .exp2 .binary32 xs e)
    (hp : FloatV.NaNEq (FloatV.mul bv e) p) (hvb : FloatV.NaNEq (FloatV.cvt p) vb) :
    vb.isFinite = true ∧
      |vb.toReal - Gam init d n hv t i * vrefS init m5 d n hv j t i| ≤
        2 ^ (-8 : ℤ) * (Gam init d n hv t i * vAbsA (vrefS init m5 d n hv j t i)
          (|uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)) + 2 ^ (-63 : ℤ) := by
  have htok := cl_lt hi
  have hcl : 0 < cl d n t := by omega
  have hlast : BT * t + cl d n t - 1 < lenOf d n := by
    simp only [cl, BT] at *; omega
  obtain ⟨gl, hgl, hglf, hglb⟩ := vals_g hd hV hG hn hhv hlast
  obtain ⟨gi, hgi, hgif, hgib⟩ := vals_g hd hV hG hn hhv htok
  rw [hbgl] at hgl; rw [hbg] at hgi
  cases hgl; cases hgi
  have hd8 := (hV.2.2.2.2.2.1 n hv t hn hhv ht).2 i hi
  have e1 : valF init d.g (gIdx d (rowOf d n) (startOf d n + (BT * t + cl d n t - 1)) hv) = bgl.toRat :=
    valF_of_rd hbgl
  have e2 : valF init d.g (gIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv) = bg.toRat :=
    valF_of_rd hbg
  have hGam : Gam init d n hv t i = Real.rpow 2 (bgl.toReal - bg.toReal) := by
    unfold Gam dgv glast gq
    rw [ite_eq_left hG, ite_eq_left hG, e1, e2]
    rfl
  have hd8' : bgl.toReal - bg.toReal ≤ 8 := by
    unfold dgv glast gq at hd8
    rw [ite_eq_left hG, ite_eq_left hG, e1, e2] at hd8
    exact hd8
  rw [hGam]
  exact numG hP hZ hvr bv hbv bgl bg xs e p vb
    ⟨hglf, by simpa [FloatV.toReal] using (show ((|bgl.toRat| : ℚ) : ℝ) ≤ ((2 ^ 24 : ℚ) : ℝ) from Rat.cast_le.2 hglb)⟩
    ⟨hgif, by simpa [FloatV.toReal] using (show ((|bg.toRat| : ℚ) : ℝ) ≤ ((2 ^ 24 : ℚ) : ℝ) from Rat.cast_le.2 hgib)⟩
    hd8' hxs he hp hvb

end Claims.ChunkDeltaH
