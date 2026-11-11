import Proof.FloatExp

/-!
# The float operations of a row's trip

The row maximum along any tree, the sum along any tree, the fused `deno` update, and the rounding
of `p` to `bfloat16` with its factor, each as an inequality over `ℝ`.
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- `F32`'s infinities are not finite, and `−∞` is not NaN. -/
theorem negInf_not_finite : (FloatV.negInf : F32).isFinite = false := by
  have hs : Format.binary32.specials = .ieee := rfl
  cases h : (FloatV.negInf : F32).isFinite
  · rfl
  · have := (FloatV.isFinite_iff.mp h).2
    rw [FloatV.negInf_isInf hs] at this; cases this

/-- A finite `float32` is above `−∞` in value. -/
theorem negInf_lt_of_finite {x : F32} (hx : x.isFinite = true) :
    (FloatV.negInf : F32).toRat < x.toRat := by
  have hs : Format.binary32.specials = .ieee := rfl
  have hn := (FloatV.isFinite_iff.mp hx).1
  obtain ⟨hle, hne⟩ := FloatV.negInf_toRat_le Format.binary32_regular hs hn
  refine lt_of_le_of_ne hle fun h => ?_
  have := hne h
  rw [this] at hx
  rw [negInf_not_finite] at hx; cases hx

/-- **The row maximum along any tree** (`tl.max(qk, axis=1)`): lanes are finite (valid) or `−∞`
(masked). If a lane is valid, the result is finite, equal to a valid lane, and no valid lane is
above it; if none is, the result is `−∞`. -/
theorem row_max_spec {qk : Fin 64 → F32} {valid : Fin 64 → Bool} {t : Tree} {rmax : F32}
    (hq : ∀ n, (valid n = true → (qk n).isFinite = true) ∧
      (valid n = false → qk n = FloatV.negInf))
    (hc : t.Covers 64)
    (h : t.Eval (FloatV.MaxRel true) (fun j => if hj : j < 64 then qk ⟨j, hj⟩ else default) rmax) :
    ((∃ n, valid n = true) → rmax.isFinite = true ∧ (∃ n, valid n = true ∧ rmax = qk n) ∧
      ∀ n, valid n = true → (qk n).toReal ≤ rmax.toReal) ∧
    ((∀ n, valid n = false) → rmax = FloatV.negInf) := by
  have hs : Format.binary32.specials = .ieee := rfl
  have hleaf : ∀ i ∈ t.leaves, i < 64 := fun i hi => by
    have := hc.mem_iff.mp hi
    simpa using this
  have hmemL : ∀ n : Fin 64, (n : ℕ) ∈ t.leaves := fun n => hc.mem_iff.mpr (by simp)
  have hnan : ∀ i ∈ t.leaves, ((fun j => if hj : j < 64 then qk ⟨j, hj⟩ else default) i).isNaN = false := by
    intro i hi
    have hi64 := hleaf i hi
    simp only [hi64, dite_true]
    cases hv : valid ⟨i, hi64⟩
    · rw [(hq _).2 hv]; exact FloatV.negInf_isNaN hs
    · exact (FloatV.isFinite_iff.mp ((hq _).1 hv)).1
  obtain ⟨⟨i0, hi0, hr⟩, hle⟩ := Tree.eval_maxRel_toReal h hnan
  have hi064 := hleaf i0 hi0
  simp only [hi064, dite_true] at hr
  constructor
  · rintro ⟨n, hn⟩
    -- the result is a valid lane
    have hvalid0 : valid ⟨i0, hi064⟩ = true := by
      by_contra hcon
      have hf : valid ⟨i0, hi064⟩ = false := by simpa using hcon
      rw [(hq _).2 hf] at hr
      have h1 := hle n (hmemL n)
      simp only [n.isLt, dite_true] at h1
      have h2 := negInf_lt_of_finite ((hq n).1 hn)
      rw [hr] at h1
      unfold FloatV.toReal at h1
      have : (qk n).toRat ≤ (FloatV.negInf : F32).toRat := by exact_mod_cast h1
      linarith
    refine ⟨?_, ⟨⟨i0, hi064⟩, hvalid0, hr⟩, fun n' hn' => ?_⟩
    · rw [hr]; exact (hq _).1 hvalid0
    · have := hle n' (hmemL n')
      simpa [n'.isLt] using this
  · intro hall
    rw [hr, (hq _).2 (hall _)]

/-- Summing the mapped list `range n` equals the finset sum over `range n`. -/
theorem sumList_eq (f : ℕ → ℚ) (n : ℕ) : ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i := rfl

/-- **A sum along any tree** (`tl.sum(p, axis=1)`) of 64 finite `float32` lanes of magnitude at most
`2`: finite and within `expm1 lam0 · Σ|p| + eps0` of the exact sum. -/
theorem sum_row {P : Platform} (hP : Device P) {p : Fin 64 → F32} {t : Tree} {S : F32}
    (hf : ∀ n, (p n).isFinite = true) (hb : ∀ n, |(p n).toReal| ≤ 2) (hc : t.Covers 64)
    (hd : t.depth ≤ tl.sumBound P 64)
    (h : t.Eval (AddRel false) (fun j => if hj : j < 64 then p ⟨j, hj⟩ else default) S) :
    S.isFinite = true ∧
      |S.toReal - ∑ n, (p n).toReal| ≤ expm1 lam0 * ∑ n, |(p n).toReal| + eps0 := by
  have hD : tl.sumBound P 64 ≤ 6 := by
    unfold tl.sumBound
    have h1 := hP.sumDepth 64
    have h2 : Nat.clog 2 64 = 6 := by decide
    omega
  set x : ℕ → F32 := fun j => if hj : j < 64 then p ⟨j, hj⟩ else default with hx
  have hxf : ∀ i, i < 64 → (x i).isFinite = true := fun i hi => by simp [hx, hi, hf]
  have hsum_abs : ((List.range 64).map fun i => |(x i).toRat|).sum = ∑ j : Fin 64, |(p j).toRat| := by
    rw [sumList_eq, Finset.sum_range]
    exact Finset.sum_congr rfl fun j _ => by simp [hx]
  have hsum : ((List.range 64).map fun i => (x i).toRat).sum = ∑ j : Fin 64, (p j).toRat := by
    rw [sumList_eq, Finset.sum_range]
    exact Finset.sum_congr rfl fun j _ => by simp [hx]
  have hbigq : (∑ j : Fin 64, |(p j).toRat| : ℚ) ≤ 128 := by
    have : ∑ j : Fin 64, |(p j).toRat| ≤ ∑ _j : Fin 64, (2 : ℚ) := Finset.sum_le_sum fun j _ => by
      have := hb j
      unfold FloatV.toReal at this
      exact_mod_cast this
    calc _ ≤ ∑ _j : Fin 64, (2 : ℚ) := this
      _ = 128 := by simp; norm_num
  have h0 : (0 : ℚ) ≤ ∑ j : Fin 64, |(p j).toRat| := Finset.sum_nonneg fun j _ => abs_nonneg _
  obtain ⟨hfin, he⟩ := sum_err_depth Format.binary32_regular x hc (hd.trans hD) h hxf (by
    rw [hsum_abs]
    have : Tree.errBound Format.binary32.u Format.binary32.eta 6 (64 - 1) (∑ j : Fin 64, |(p j).toRat|) ≤ 2 ^ 20 := by
      unfold Tree.errBound
      rw [Format.binary32_u, Format.binary32_eta]
      have hc : ((1 + (2 : ℚ) ^ (-24 : ℤ)) ^ 6 - 1) ≤ 1 / 2 ^ 10 := by norm_num
      have h1 : ((1 + (2 : ℚ) ^ (-24 : ℤ)) ^ 6 - 1) * ∑ j : Fin 64, |(p j).toRat| ≤ 1 / 2 ^ 10 * 128 :=
        mul_le_mul hc hbigq h0 (by norm_num)
      have h2 : (2 : ℚ) ^ (-150 : ℤ) * (64 - 1 : ℕ) * (1 + 2 ^ (-24 : ℤ)) ^ 6 ≤ 1 := by norm_num
      norm_num at h1 h2 ⊢
      linarith
    rw [Format.binary32_maxVal]
    linarith)
  refine ⟨hfin, ?_⟩
  rw [hsum, hsum_abs] at he
  have he' := (Rat.cast_le (K := ℝ)).mpr he
  unfold Tree.errBound at he'
  rw [Format.binary32_u, Format.binary32_eta] at he'
  push_cast at he'
  have e1 : ((1 + (2 : ℝ) ^ (-24 : ℤ)) ^ 6 - 1) ≤ expm1 lam0 := by
    unfold expm1 lam0
    have h1 : (1 + (2 : ℝ) ^ (-24 : ℤ)) ≤ Real.exp (2 ^ (-24 : ℤ)) := by
      have := Real.add_one_le_exp ((2 : ℝ) ^ (-24 : ℤ)); linarith
    have h2 : (1 + (2 : ℝ) ^ (-24 : ℤ)) ^ 6 ≤ Real.exp (2 ^ (-24 : ℤ)) ^ 6 :=
      pow_le_pow_left₀ (by positivity) h1 6
    have h3 : Real.exp (2 ^ (-24 : ℤ)) ^ 6 = Real.exp (6 / 2 ^ 24) := by
      rw [← Real.exp_nat_mul]; congr 1
    exact sub_le_sub_right (h2.trans_eq h3) 1
  have e2 : (2 : ℝ) ^ (-150 : ℤ) * ((64 - 1 : ℕ) : ℝ) * (1 + (2 : ℝ) ^ (-24 : ℤ)) ^ 6 ≤ eps0 := by
    unfold eps0; norm_num
  have hA : (0 : ℝ) ≤ ∑ j : Fin 64, |((p j).toRat : ℝ)| :=
    Finset.sum_nonneg fun j _ => abs_nonneg _
  have e3 := mul_le_mul_of_nonneg_right e1 hA
  have e4 : (2 : ℝ) ^ (-150 : ℤ) * 63 * (1 + (2 : ℝ) ^ (-24 : ℤ)) ^ 6 ≤ eps0 := by
    unfold eps0; norm_num
  unfold FloatV.toReal
  linarith

/-- **A relative factor for a rounding**: if `|ε| ≤ ub·|x| + etab` then `ε = x·δ + ρ` with
`|δ| ≤ ub` and `|ρ| ≤ etab`. -/
theorem round_factor {x ε : ℝ} (h : |ε| ≤ ub * |x| + etab) :
    ∃ δ : ℝ, |δ| ≤ ub ∧ |ε - x * δ| ≤ etab := by
  have hub : 0 ≤ ub := by unfold ub; norm_num
  have het : 0 ≤ etab := by unfold etab; norm_num
  by_cases hx : x = 0
  · refine ⟨0, by simpa using hub, ?_⟩
    simpa [hx] using h
  · have hx0 : 0 < |x| := abs_pos.mpr hx
    set q := ε / x with hq
    have hε : ε = x * q := by rw [hq]; field_simp
    have hε' : |ε| = |x| * |q| := by rw [hε, abs_mul]
    by_cases hc : |q| ≤ ub
    · refine ⟨q, hc, ?_⟩
      rw [← hε]; simpa using het
    · have hqu : ub < |q| := not_le.mp hc
      rcases le_or_gt 0 q with hs | hs
      · refine ⟨ub, by rw [abs_of_nonneg hub], ?_⟩
        have hqa : |q| = q := abs_of_nonneg hs
        have e3 : |ε - x * ub| = |x| * (q - ub) := by
          rw [hε, show x * q - x * ub = x * (q - ub) by ring, abs_mul,
            abs_of_nonneg (by linarith : 0 ≤ q - ub)]
        rw [e3]
        nlinarith
      · refine ⟨-ub, by rw [abs_neg, abs_of_nonneg hub], ?_⟩
        have hqa : |q| = -q := abs_of_neg hs
        have e3 : |ε - x * (-ub)| = |x| * (-q - ub) := by
          rw [hε, show x * q - x * -ub = x * (q + ub) by ring, abs_mul,
            abs_of_nonpos (by linarith : q + ub ≤ 0)]
          ring
        rw [e3]
        nlinarith

/-- **`p.to(bfloat16)`** of a finite `float32` of magnitude at most `2`: finite, `p (1 + δ)` up to
`etab`, with `|δ| ≤ ub`. -/
theorem cvt_p16 {p : F32} {p16 : BF16} (hp : p.isFinite = true) (hb : |p.toReal| ≤ 2)
    (h : FloatV.NaNEq (FloatV.cvtWith .rn p) p16) :
    p16.isFinite = true ∧ ∃ δ : ℝ, |δ| ≤ ub ∧ |p16.toReal - p.toReal * (1 + δ)| ≤ etab := by
  have hm : |p.toReal| ≤ ((Format.bfloat16.maxVal : ℚ) : ℝ) := by
    rw [Format.bfloat16_maxVal_real]; refine hb.trans ?_; norm_num
  obtain ⟨h1, h2⟩ := FloatV.cvt_err_real (g := Format.bfloat16) Format.bfloat16_regular hp hm
  have hcv : FloatV.cvtWith .rn p = (FloatV.cvt p : BF16) := FloatV.cvtWith_rn p
  rw [hcv] at h
  obtain ⟨hf, -, hr⟩ := h.exact_of_isFinite h1
  refine ⟨hf, ?_⟩
  rw [hr]
  rw [Format.bfloat16_u_real, Format.bfloat16_eta_real] at h2
  have h3 : |(FloatV.cvt p : BF16).toReal - p.toReal| ≤ ub * |p.toReal| + etab := by
    unfold ub etab; norm_num at h2 ⊢; linarith
  obtain ⟨δ, hδ, hρ⟩ := round_factor h3
  refine ⟨δ, hδ, ?_⟩
  have e : (FloatV.cvt p : BF16).toReal - p.toReal * (1 + δ) =
      ((FloatV.cvt p : BF16).toReal - p.toReal) - p.toReal * δ := by ring
  rw [e]; exact hρ

/-- **The fused `deno` update** `deno * re_scale + sum` in the reals. -/
theorem deno_madd {fusion : Bool} {l r S l' : F32} (hl : l.isFinite = true) (hr : r.isFinite = true)
    (hS : S.isFinite = true) (hlb : |l.toReal| ≤ 2 ^ 20) (hrb : |r.toReal| ≤ 2)
    (hSb : |S.toReal| ≤ 2 ^ 20)
    (h : FloatV.NaNEq (FloatV.add (FloatV.mul l r) S) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma l r S) l')) :
    l'.isFinite = true ∧
      |l'.toReal - (r.toReal * l.toReal + S.toReal)| ≤
        3 * uf * (|r.toReal * l.toReal| + |S.toReal|) + eps0 := by
  have hmax : |l.toRat * r.toRat| + |S.toRat| ≤ Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h1 : |l.toReal * r.toReal| ≤ 2 ^ 21 := by
      rw [abs_mul]
      calc |l.toReal| * |r.toReal| ≤ 2 ^ 20 * 2 :=
            mul_le_mul hlb hrb (abs_nonneg _) (by positivity)
        _ = 2 ^ 21 := by norm_num
    have h2 : |l.toRat * r.toRat| + |S.toRat| ≤ ((2 ^ 22 : ℚ)) := by
      have : (|l.toReal * r.toReal| + |S.toReal| : ℝ) ≤ 2 ^ 22 := by
        have := hSb; nlinarith
      unfold FloatV.toReal at this
      exact_mod_cast (by simpa [abs_mul] using this : ((|l.toRat * r.toRat| + |S.toRat| : ℚ) : ℝ) ≤ ((2 ^ 22 : ℚ) : ℝ))
    refine h2.trans ?_
    rw [Format.binary32_maxVal, Format.binary32_eta]; norm_num
  obtain ⟨hf, he⟩ := FloatV.mulAdd_err Format.binary32_regular (by rw [Format.binary32_u]; norm_num)
    h hl hr hS hmax
  refine ⟨hf, ?_⟩
  rw [Format.binary32_u, Format.binary32_eta] at he
  have he' := (Rat.cast_le (K := ℝ)).mpr he
  push_cast at he'
  unfold FloatV.toReal
  have e4 : (4 : ℝ) * 2 ^ (-150 : ℤ) ≤ eps0 := by unfold eps0; norm_num
  have e3 : (3 : ℝ) * 2 ^ (-24 : ℤ) = 3 * uf := by unfold uf; norm_num
  have em : (l.toRat : ℝ) * r.toRat = r.toRat * l.toRat := mul_comm _ _
  rw [em, e3] at he'
  linarith

end

end Claims.ExtendAttn
