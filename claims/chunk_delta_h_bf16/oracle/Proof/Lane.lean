import Proof.GlueStep
import Proof.GlueIdx
import Proof.Assembly
import Proof.Inv

/-!
# `chunk_delta_h_bf16`: one value column through a trip

`lane_core` is the numeric heart of a trip for one column `j`, over `ℕ`-indexed lane data (the
state rows `c`, the tokens `i` of the chunk): the new values of the chunk's tokens are within
`tolV`, the new state is finite and one step from the old one, and its padded rows stay within
`padB`. The two layouts of the state give the same data (`Trip*`).
-/

set_option maxHeartbeats 800000

namespace Claims.ChunkDeltaH

open TVA

/-- The `float32` constant `+0` is finite. -/
theorem zf32_isFinite : zf32.isFinite = true := zf_zf32.1

/-- The `float32` constant `+0` has value `0`. -/
theorem zf32_toReal : zf32.toReal = 0 := zf_zf32.2

/-- **The numeric core of a trip, one column.** -/
theorem lane_core {d : Inst} (hd : d.ok) {P : Platform} (hE : Device.Exp2F32 P) {fusion : Bool}
    {init m5 : Mem} {n hv jv j t nb : Nat} (hn : n < d.N) (hhv : hv < d.HV) (ht : t < nch d n)
    (hj : inBlk d jv j) (hV : Vals init d) (hSt : StOk m5 d n hv j t)
    (hnb : d.K ≤ 64 * nb) (hnb' : 64 * nb < d.K + 64)
    -- the state entering the trip, its stored rounding and the rounding for the dots
    (Y : ℕ → F32) (xh xs : ℕ → BF16)
    (hxh : ∀ c, c < d.K → rdB m5 d.h (hIdx d (cbase d n + t) hv c j) = some (xh c))
    (hxhc : ∀ c, c < d.K → FloatV.NaNEq (FloatV.cvt (Y c)) (xh c))
    (hxsc : ∀ c, c < 64 * nb → FloatV.NaNEq (FloatV.cvt (Y c)) (xs c))
    (hYf : ∀ c, c < d.K → (Y c).isFinite = true)
    (hpad : ∀ c, d.K ≤ c → c < 64 * nb → (Y c).isFinite = true ∧ |(Y c).toReal| ≤ padB init d n hv t)
    (hpb : padB init d n hv t ≤ 2 ^ 100)
    -- the `w` tile, the `u` element, the dot chain and the difference
    (wt : ℕ → ℕ → BF16) (bu xv : ℕ → BF16) (uf bv5 : ℕ → F32) (rr : ℕ → ℕ → F32)
    (hwr : ∀ i, i < cl d n t → ∀ c, c < d.K →
      rdB init d.w (wIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv c) = some (wt i c))
    (hwz : ∀ i, i < 64 → ∀ c, c < 64 * nb → (cl d n t ≤ i ∨ d.K ≤ c) → ZB (wt i c))
    (hbu : ∀ i, i < cl d n t →
      rdB init d.u (uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv j) = some (bu i))
    (huz : ∀ i, cl d n t ≤ i → i < 64 → ZB (bu i))
    (huf : ∀ i, i < 64 → ConvRel P .bf16 .f32 (bu i) (uf i))
    (hr0 : ∀ i, i < 64 → (rr 0 i).isFinite = true ∧ (rr 0 i).toReal = 0)
    (hdot : ∀ i, i < 64 → ∀ b, b < nb →
      (∀ k, k < 64 → (wt i (64 * b + k)).isFinite = true ∧ (xs (64 * b + k)).isFinite = true) →
      (rr b i).isFinite = true →
      |(rr b i).toReal| + ∑ k ∈ Finset.range 64, lmB (wt i (64 * b + k)) * lmB (xs (64 * b + k)) ≤ 2 ^ 120 →
      (rr (b + 1) i).isFinite = true ∧
        |(rr (b + 1) i).toReal - ((rr b i).toReal +
            ∑ k ∈ Finset.range 64, (wt i (64 * b + k)).toReal * (xs (64 * b + k)).toReal)| ≤
          rD * (|(rr b i).toReal| + ∑ k ∈ Finset.range 64, lmB (wt i (64 * b + k)) * lmB (xs (64 * b + k))) + aD)
    (hsub : ∀ i, i < 64 → AddRel fusion (uf i) (FloatV.neg (rr nb i)) (bv5 i))
    (hxvc : ∀ i, i < 64 → FloatV.NaNEq (FloatV.cvt (bv5 i)) (xv i))
    (_hxvr : ∀ i, i < cl d n t →
      rdB m5 d.vn (uIdx d (rowOf d n) (startOf d n + (BT * t + i)) hv j) = some (xv i))
    -- the gate, the rounded gated values and the decayed state
    (bv6 : ℕ → F32) (vb : ℕ → BF16) (gt : F32) (S : ℕ → F32)
    (hvbc : ∀ i, i < 64 → FloatV.NaNEq (FloatV.cvt (bv6 i)) (vb i))
    (hgate : (d.useG = true ∧ ∃ bgl, rdF init d.g (gTok d n hv (BT * t + cl d n t - 1)) = some bgl ∧
        P.math .exp2 .binary32 bgl gt ∧
        (∃ bg xsg e : ℕ → F32,
          (∀ i, i < cl d n t → rdF init d.g (gTok d n hv (BT * t + i)) = some (bg i)) ∧
          (∀ i, i < 64 → FloatV.NaNEq (FloatV.sub bgl (bg i)) (xsg i)) ∧
          (∀ i, i < 64 → P.math .exp2 .binary32 (xsg i) (e i)) ∧
          (∀ i, i < 64 → FloatV.NaNEq (FloatV.mul (bv5 i)
            (if i < cl d n t then e i else zf32)) (bv6 i))) ∧
        ∀ c, c < 64 * nb → FloatV.NaNEq (FloatV.mul (Y c) gt) (S c)) ∨
      (d.useG = false ∧ (∀ i, i < 64 → bv6 i = bv5 i) ∧ ∀ c, c < 64 * nb → S c = Y c))
    -- the `k` tile and the update
    (kk : ℕ → ℕ → BF16) (Yn : ℕ → F32)
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
    (∀ i, i < cl d n t →
      |(xv i).toReal - vrefS init m5 d n hv j t i| ≤ tolV init m5 d n hv j t i ∧
        (xv i).isFinite = true) ∧
    ((∀ c, c < d.K → (Y c).isFinite = true ∧ (Yn c).isFinite = true) ∧
      colNorm d.K (fun c => (Yn c).toReal - Step init m5 d n hv j t (fun c' => (Y c').toReal) c) ≤
        errStep init m5 d n hv j t) ∧
    (∀ c, d.K ≤ c → c < 64 * nb → (Yn c).isFinite = true ∧ |(Yn c).toReal| ≤ padB init d n hv (t + 1)) := by
  have hK0 := hd.K_pos
  have hcl : cl d n t ≤ 64 := by simp only [cl, BT]; omega
  have hxhf : ∀ c, c < d.K → (xh c).isFinite = true := by
    intro c hc
    obtain ⟨x, hx, hf⟩ := hSt.1 c hc
    rw [hxh c hc] at hx; cases hx; exact hf
  have hxs_eq : ∀ c, c < d.K → xs c = xh c := by
    intro c hc
    have h1 := hxsc c (by omega)
    have h2 := hxhc c hc
    have hf : (FloatV.cvt (Y c) : BF16).isFinite = true := by rw [h2.isFinite_eq]; exact hxhf c hc
    rw [h1.eq_of_isFinite hf, h2.eq_of_isFinite hf]
  have hxsf : ∀ c, c < 64 * nb → (xs c).isFinite = true := by
    intro c hc
    by_cases hcK : c < d.K
    · rw [hxs_eq c hcK]; exact hxhf c hcK
    · obtain ⟨h1, h2⟩ := hpad c (not_lt.mp hcK) hc
      exact (cvt_bf16 h1 (h2.trans hpb) (hxsc c hc)).1
  have hHs : ∀ c, c < d.K → Hs m5 d n hv j t c = (xs c).toReal := by
    intro c hc; unfold Hs; rw [valB_of_rd (hxh c hc), hxs_eq c hc]; rfl
  have hxhs : ∀ c, c < d.K → rdB m5 d.h (hIdx d (cbase d n + t) hv c j) = some (xs c) := by
    intro c hc; rw [hxs_eq c hc]; exact hxh c hc
  have hwfin : ∀ i, i < 64 → ∀ c, c < 64 * nb → (wt i c).isFinite = true := by
    intro i hi c hc
    by_cases h : i < cl d n t ∧ c < d.K
    · obtain ⟨x, hx, hf, -⟩ := vals_w hd hV hn hhv (cl_lt h.1) h.2
      rw [hwr i h.1 c h.2] at hx; cases hx; exact hf
    · exact (hwz i hi c hc (by by_contra hh; push Not at hh; exact h ⟨hh.1, hh.2⟩)).1
  have hstep : ∀ i, i < 64 → ∀ b, b < nb → (rr b i).isFinite = true →
      |(rr b i).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (wt i c) * lmB (xs c) ≤ 2 ^ 120 →
      (rr (b + 1) i).isFinite = true ∧
        |(rr (b + 1) i).toReal - ((rr b i).toReal +
            ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), (wt i c).toReal * (xs c).toReal)| ≤
          rD * (|(rr b i).toReal| + ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), lmB (wt i c) * lmB (xs c)) + aD := by
    intro i hi b hb hacc hroom
    have e1 : ∀ f : ℕ → ℝ, ∑ c ∈ Finset.Ico (64 * b) (64 * b + 64), f c = ∑ k ∈ Finset.range 64, f (64 * b + k) := by
      intro f; rw [Finset.sum_Ico_eq_sum_range]; simp
    rw [e1 (fun c => lmB (wt i c) * lmB (xs c))] at hroom ⊢
    rw [e1 (fun c => (wt i c).toReal * (xs c).toReal)]
    exact hdot i hi b hb (fun k hk => ⟨hwfin i hi _ (by omega), hxsf _ (by omega)⟩) hacc hroom
  -- the tokens of the chunk
  have tokv := fun i (hi : i < cl d n t) => tok_v hd hV hn hhv hi hj.2.2 hnb hnb' (P := P)
    (fusion := fusion) (wt i) xs (fun b => rr b i) (bu i) (xv i) (uf i) (bv5 i) (hwr i hi)
    (fun c h1 h2 => ⟨(hwz i (by omega) c h2 (Or.inr h1)).2, (hwz i (by omega) c h2 (Or.inr h1)).1⟩)
    hxhs hxsf hSt.2 (hbu i hi) (huf i (by omega)) (hr0 i (by omega)) (hstep i (by omega))
    (hsub i (by omega)) (hxvc i (by omega))
  -- the tokens past the sequence's end: zero rows, so the difference is a zero
  have hmask : ∀ i, cl d n t ≤ i → i < 64 → (bv5 i).isFinite = true ∧ |(bv5 i).toReal| ≤ 2 ^ (-110 : ℤ) := by
    intro i hi hi64
    have hwzi : ∀ c, c < 64 * nb → ZB (wt i c) := fun c hc => hwz i hi64 c hc (Or.inl hi)
    obtain ⟨hbf, hbz⟩ := huz i hi hi64
    obtain ⟨-, hufin, -, hufr⟩ := ConvRel.bf16_f32 (huf i hi64) hbf
    have hhn' : colNorm d.K (fun c => (xs c).toReal) ≤ 2 ^ 40 := by
      rw [← colNorm_congr (fun c hc => hHs c hc)]; exact hSt.2
    have key := numV hK0 hd.K_le hnb hnb' (wt i) xs (uf i) (fun b => rr b i) (bv5 i) (xv i) (fusion := fusion)
      (fun c hc => ⟨(hwzi c hc).1, by rw [(hwzi c hc).2]; norm_num⟩) (fun c _ hc => (hwzi c hc).2) hxsf hhn'
      ⟨hufin, by rw [hufr, hbz]; norm_num⟩ (hr0 i hi64) (hstep i hi64) (hsub i hi64) (hxvc i hi64)
    obtain ⟨-, -, k3, k4, -, -⟩ := key
    have hS0 : ∑ c ∈ Finset.range d.K, (wt i c).toReal * (xs c).toReal = 0 :=
      Finset.sum_eq_zero fun c hc => by
        rw [(hwzi c (by have := Finset.mem_range.1 hc; omega)).2, zero_mul]
    have hl0 : colNorm d.K (fun c => lmB (wt i c)) = 0 := by
      unfold colNorm
      rw [Finset.sum_eq_zero]
      · simp
      · intro c hc
        show lmB (wt i c) ^ 2 = 0
        rw [lmB_zero (hwzi c (by have := Finset.mem_range.1 hc; omega)).2]; simp
    rw [hS0, hl0, hufr, hbz] at k4
    refine ⟨k3, ?_⟩
    simp only [sub_zero, zero_mul, abs_zero, add_zero, mul_zero, zero_add] at k4
    simpa using k4
  -- the rounded gated values
  have hvbk : ∀ i, i < cl d n t → (vb i).isFinite = true ∧
      |(vb i).toReal - Gam init d n hv t i * vrefS init m5 d n hv j t i| ≤
        2 ^ (-8 : ℤ) * (Gam init d n hv t i * vAbsA (vrefS init m5 d n hv j t i)
          (|uc init d n hv t i j| + wl init d n hv t i * hl m5 d n hv j t)) + 2 ^ (-63 : ℤ) := by
    intro i hi
    obtain ⟨k1, k2, k3, k4, k5, k6⟩ := tokv i hi
    rcases hgate with ⟨hG, bgl, hbgl, he, ⟨bg, xsg, e, hbgr, hsubg, hexp, hmulg⟩, hS⟩ | ⟨hG, hbv6, hSY⟩
    · refine tok_g hd hE hV hn hhv ht hi hG bgl (bg i) (xsg i) (e i) (bv6 i) (bv5 i) (vb i) hbgl (hbgr i hi)
        k1 k2 ⟨k3, k4⟩ (hsubg i (by omega)) (hexp i (by omega)) ?_ (hvbc i (by omega))
      have := hmulg i (by omega)
      simpa [hi] using this
    · have hGam : Gam init d n hv t i = 1 := by unfold Gam dgv; simp [hG]
      have hxv : vb i = xv i := by
        have h1 := hxvc i (by omega)
        have h2 := hvbc i (by omega)
        rw [hbv6 i (by omega)] at h2
        have hf : (FloatV.cvt (bv5 i) : BF16).isFinite = true := by rw [h1.isFinite_eq]; exact k5
        rw [h2.eq_of_isFinite hf, h1.eq_of_isFinite hf]
      rw [hxv, hGam, one_mul]
      refine ⟨k5, k6.trans ?_⟩
      unfold tolV vAbs vAbsA
      have : (2 : ℝ) ^ (-100 : ℤ) ≤ 2 ^ (-63 : ℤ) := by norm_num
      linarith
  have hvbm : ∀ i, cl d n t ≤ i → i < 64 → (vb i).isFinite = true := by
    intro i hi hi64
    obtain ⟨hbf, hbb⟩ := hmask i hi hi64
    rcases hgate with ⟨hG, bgl, hbgl, he, ⟨bg, xsg, e, hbgr, hsubg, hexp, hmulg⟩, hS⟩ | ⟨hG, hbv6, hSY⟩
    · refine numG_masked (bv5 i) zf32 (bv6 i) (vb i) hbf ⟨zf32_isFinite, zf32_toReal⟩ ?_ (hvbc i hi64)
      have := hmulg i hi64
      simpa [Nat.not_lt.2 hi] using this
    · have h2 := hvbc i hi64
      rw [hbv6 i hi64] at h2
      exact (cvt_bf16 hbf (hbb.trans (by norm_num)) h2).1
  have hvbf : ∀ i, i < 64 → (vb i).isFinite = true := by
    intro i hi
    by_cases h : i < cl d n t
    · exact (hvbk i h).1
    · exact hvbm i (not_lt.1 h) hi
  have hgate' : (d.useG = true ∧ ∃ bgl, rdF init d.g (gTok d n hv (BT * t + cl d n t - 1)) = some bgl ∧
        P.math .exp2 .binary32 bgl gt ∧ ∀ c, c < 64 * nb → FloatV.NaNEq (FloatV.mul (Y c) gt) (S c)) ∨
      (d.useG = false ∧ ∀ c, c < 64 * nb → S c = Y c) := by
    rcases hgate with ⟨hG, bgl, hbgl, he, -, hS⟩ | ⟨hG, -, hSY⟩
    · exact Or.inl ⟨hG, bgl, hbgl, he, hS⟩
    · exact Or.inr ⟨hG, hSY⟩
  obtain ⟨⟨hfin, hstepb⟩, hpadb⟩ := core_s hd hE hV hn hhv ht hj hSt hnb hnb' Y Yn xh hxh hxhc hYf hpad hpb gt S
    hgate' vb hvbf (fun i hi => (tokv i hi).1) (fun i hi => (tokv i hi).2.1) (fun i hi => (hvbk i hi).2)
    kk hkr hkz ρ2 α2 hρ2 hα2 ρP αP hρP hαP hupdP hupd
  refine ⟨fun i hi => ?_, ⟨fun c hc => ⟨hYf c hc, hfin c (by omega)⟩, hstepb⟩, fun c hc1 hc2 =>
    ⟨hfin c hc2, hpadb c hc1 hc2⟩⟩
  obtain ⟨-, -, -, -, k5, k6⟩ := tokv i hi
  exact ⟨k6, k5⟩

end Claims.ChunkDeltaH
