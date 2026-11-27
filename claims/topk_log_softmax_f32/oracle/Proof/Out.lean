import Proof.Num

/-!
# `lse` and the outputs

`lse_bound`: `lse = tl.log(se)` is within `lseTol vocab_size` of `log ZR`. `out_bound`: an output
`(x − mv) − lse` of a finite logit is within `2^-22·|L| + 2^-37` of `L − sh`, `sh = lse − log ZR`;
`ninf_out`: a `-∞` logit gives `-∞`. `row_num` puts them together in the claim's terms.
-/
namespace Claims.TopkLogSoftmax

open TVA

/-- `log V ≤ (66 / 5)` for `V ≤ 2^19`. -/
lemma log_le_of_le {Z : ℝ} (hZ0 : 0 < Z) (hZ : Z ≤ 2 ^ 19) : Real.log Z ≤ (66 / 5) := by
  have h1 : Real.log Z ≤ Real.log ((2 : ℝ) ^ 19) := Real.log_le_log hZ0 hZ
  have h3 : Real.log ((2 : ℝ) ^ 19) = 19 * Real.log 2 := by
    rw [Real.log_pow]; norm_num
  have h2 := Real.log_two_lt_d9
  rw [h3] at h1
  linarith

/-- **`lse`.** From `se` within `(T + 143)·u·Z` of `Z ∈ [1, V]`, `T·1024 < V + 1024`, the
platform's `log` of `se` is finite and within `lseTol V` of `log Z`. -/
theorem lse_bound {P : Platform} (hlog : Device.LogF32 P) {V T : Nat} (hV19 : V ≤ 2 ^ 19)
    (hTV : T * 1024 < V + 1024) {Z : ℝ} (hZ1 : 1 ≤ Z) (hZV : Z ≤ (V : ℝ)) {se lse : F32}
    (hsefin : se.isFinite = true) (hse : |se.toReal - Z| ≤ ((T : ℝ) + 143) * uR * Z)
    (hl : P.math .log .binary32 se lse) :
    lse.isFinite = true ∧ |lse.toReal - Real.log Z| ≤ (lseTol V : ℝ) := by
  have hu := uR_pos
  have hT : (T : ℝ) * 1024 < V + 1024 := by exact_mod_cast hTV
  have hV19r : (V : ℝ) ≤ 2 ^ 19 := by exact_mod_cast hV19
  have hT0 : (0 : ℝ) ≤ T := Nat.cast_nonneg T
  have hTle : (T : ℝ) ≤ 513 := by nlinarith
  set ρ := ((T : ℝ) + 143) * uR with hρ
  have hρ0 : 0 ≤ ρ := by positivity
  have hρ656 : ρ ≤ 656 * uR := by rw [hρ]; nlinarith
  have hρh : ρ ≤ 1 / 1000 := by
    have : (656 : ℝ) * uR ≤ 1 / 1000 := by unfold uR; norm_num
    linarith
  have hρsq : ρ ^ 2 ≤ uR / 8 := by
    have h1 : ρ ^ 2 ≤ (656 * uR) ^ 2 := pow_le_pow_left₀ hρ0 hρ656 2
    have h2 : ((656 : ℝ) * uR) ^ 2 ≤ uR / 8 := by unfold uR; norm_num
    linarith
  have hZ0 : 0 < Z := by linarith
  have hse_lo : (1 - ρ) * Z ≤ se.toReal := by
    have := (abs_le.mp hse).1; nlinarith
  have hse_hi : se.toReal ≤ (1 + ρ) * Z := by
    have := (abs_le.mp hse).2; nlinarith
  have hsepos : 0 < se.toReal := by nlinarith
  -- `log se` against `log Z`
  have hq : Real.log se.toReal - Real.log Z = Real.log (se.toReal / Z) := by
    rw [Real.log_div hsepos.ne' hZ0.ne']
  have hqpos : 0 < se.toReal / Z := div_pos hsepos hZ0
  have hq_hi : se.toReal / Z ≤ 1 + ρ := by rw [div_le_iff₀ hZ0]; linarith
  have hq_lo : 1 - ρ ≤ se.toReal / Z := by rw [le_div_iff₀ hZ0]; linarith
  have hlogq_hi : Real.log (se.toReal / Z) ≤ ρ := by
    have := Real.log_le_sub_one_of_pos hqpos; linarith
  have hlogq_lo : -(ρ + 2 * ρ ^ 2) ≤ Real.log (se.toReal / Z) := by
    have h1 := Real.one_sub_inv_le_log_of_pos hqpos
    have h2 : (se.toReal / Z)⁻¹ ≤ 1 + ρ + 2 * ρ ^ 2 := by
      have h3 : 0 < 1 - ρ := by linarith
      rw [inv_le_comm₀ hqpos (by positivity)]
      refine le_trans ?_ hq_lo
      rw [inv_le_iff_one_le_mul₀ (by positivity)]
      nlinarith
    linarith
  have hlogd : |Real.log se.toReal - Real.log Z| ≤ ρ + 2 * ρ ^ 2 := by
    rw [hq, abs_le]; constructor <;> nlinarith
  have hlogZ0 : 0 ≤ Real.log Z := Real.log_nonneg hZ1
  have hlogZ : Real.log Z ≤ (66 / 5) := log_le_of_le hZ0 (hZV.trans hV19r)
  have hlogse : |Real.log se.toReal| ≤ (66 / 5) + ρ + 2 * ρ ^ 2 := by
    have := abs_sub_abs_le_abs_sub (Real.log se.toReal) (Real.log Z)
    rw [abs_of_nonneg hlogZ0] at this
    linarith
  -- the device's `log`
  obtain ⟨g, hg, hacc⟩ := hlog
  have hgr : g = Real.log := by simpa only [MathFn.ref, Option.some.injEq] using hg.symm
  rw [hgr] at hacc
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.log se.toReal| + (2 : ℝ) ^ (-149 : ℤ) ≤
      (Format.binary32.maxVal : ℝ) := by
    have h1 : ((Format.binary32.maxVal : ℚ) : ℝ) = ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := b32_maxVal_cast
    have h2 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |Real.log se.toReal| ≤ 2 * 14 := by
      have : (1 + (2 : ℝ) ^ (-22 : ℤ)) ≤ 2 := by norm_num
      have h3 : |Real.log se.toReal| ≤ 14 := by
        have : uR / 8 ≤ 1 / 10 := by unfold uR; norm_num
        linarith
      nlinarith [abs_nonneg (Real.log se.toReal)]
    have h4 : (2 : ℝ) * 14 + (2 : ℝ) ^ (-149 : ℤ) ≤ ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := by norm_num
    rw [h1]; linarith
  obtain ⟨hlfin, hlerr⟩ := hacc se lse hl hsefin (Set.mem_univ _) (show se.toReal ∈ Set.Ioi 0 from hsepos)
    hroom
  refine ⟨hlfin, ?_⟩
  have h1 := abs_sub_le lse.toReal (Real.log se.toReal) (Real.log Z)
  have h2 : (2 : ℝ) ^ (-22 : ℤ) = 4 * uR := by unfold uR; norm_num
  rw [h2] at hlerr
  have hlt : (lseTol V : ℝ) = (V : ℝ) / 1024 * uR + 256 * uR := by
    unfold lseTol uR; push_cast; ring
  rw [hlt]
  have h3 : 4 * uR * |Real.log se.toReal| ≤ 4 * uR * ((66 / 5) + ρ + 2 * ρ ^ 2) :=
    mul_le_mul_of_nonneg_left hlogse (by positivity)
  have h5 : (2 : ℝ) ^ (-149 : ℤ) ≤ uR / 8 := by unfold uR; norm_num
  have h6 : (T : ℝ) ≤ (V : ℝ) / 1024 + 1 := by linarith
  have h7 : ρ ≤ (V : ℝ) / 1024 * uR + 144 * uR := by
    have := mul_le_mul_of_nonneg_right (show (T : ℝ) + 143 ≤ (V : ℝ) / 1024 + 144 by linarith) hu.le
    rw [hρ]; linarith [show ((V : ℝ) / 1024 + 144) * uR = (V : ℝ) / 1024 * uR + 144 * uR by ring]
  have h8 : 4 * uR * ρ ≤ uR / 8 := by
    have h9 : 4 * uR * ρ ≤ 4 * uR * (1 / 1000) := mul_le_mul_of_nonneg_left hρh (by positivity)
    linarith
  have h10 : 8 * uR * ρ ^ 2 ≤ uR / 8 := by
    have h11 : 8 * uR * ρ ^ 2 ≤ 8 * uR * (uR / 8) := mul_le_mul_of_nonneg_left hρsq (by positivity)
    have : 8 * uR * (uR / 8) ≤ uR / 8 := by unfold uR; norm_num
    linarith
  have h12 : 4 * uR * ((66 / 5) + ρ + 2 * ρ ^ 2) = (264 / 5) * uR + 4 * uR * ρ + 8 * uR * ρ ^ 2 := by ring
  generalize (2 : ℝ) ^ (-149 : ℤ) = p149 at hlerr h5
  have hA : |lse.toReal - Real.log se.toReal| ≤ (264 / 5) * uR + uR / 8 + uR / 8 + uR / 8 := by
    linarith
  have hB : |Real.log se.toReal - Real.log Z| ≤ (V : ℝ) / 1024 * uR + 144 * uR + uR / 4 := by
    linarith
  linarith

/-- **An output of a finite logit.** `d = x − mv` (one rounding) and `o = d − lse` (one rounding),
with `|x − mv| ≤ 2^127`, `x ≤ mv` and `lse = log Z + sh`: `o` is finite and within
`2^-22·|L| + 2^-37` of `L − sh`, `L = x − mv − log Z`. -/
theorem out_bound {mv lse x d o : F32} {Z sh : ℝ} (hmvfin : mv.isFinite = true)
    (hxfin : x.isFinite = true) (hxle : x.toRat ≤ mv.toRat)
    (hgap : |x.toReal - mv.toReal| ≤ 2 ^ 127) (hZ1 : 1 ≤ Z) (hZV : Z ≤ 2 ^ 19)
    (hlsefin : lse.isFinite = true) (hsh : sh = lse.toReal - Real.log Z)
    (hshb : |sh| ≤ 2 ^ (-14 : ℤ))
    (hd : FloatV.NaNEq (FloatV.sub x mv) d) (ho : FloatV.NaNEq (FloatV.sub d lse) o) :
    o.isFinite = true ∧
      |o.toReal - ((x.toReal - mv.toReal - Real.log Z) - sh)| ≤
        (elemTol : ℝ) * |x.toReal - mv.toReal - Real.log Z| + (floorTol : ℝ) := by
  have hu := uR_pos
  have he := etaR_pos
  have hmax : ((Format.binary32.maxVal : ℚ) : ℝ) = ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := b32_maxVal_cast
  have hm1 : (2 : ℝ) ^ 127 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by rw [hmax]; norm_num
  obtain ⟨hs1, hs2⟩ := FloatV.sub_err_real b32_reg hxfin hmvfin (hgap.trans hm1)
  rw [b32_u_cast, b32_eta_cast] at hs2
  have hdeq : d = FloatV.sub x mv := FloatV.NaNEq.eq hd (FloatV.isFinite_iff.mp hs1).1
  subst hdeq
  set a := x.toReal - mv.toReal with ha
  have ha0 : a ≤ 0 := by
    have : (x.toRat : ℝ) ≤ (mv.toRat : ℝ) := by exact_mod_cast hxle
    simp only [ha, FloatV.toReal]; linarith
  have hZ0 : 0 < Z := by linarith
  have hlogZ0 : 0 ≤ Real.log Z := Real.log_nonneg hZ1
  have hlogZ : Real.log Z ≤ (66 / 5) := log_le_of_le hZ0 hZV
  have hlse : lse.toReal = Real.log Z + sh := by rw [hsh]; ring
  have hshs : |sh| ≤ 1 := hshb.trans (by norm_num)
  have hlseb : |lse.toReal| ≤ Real.log Z + |sh| := by
    rw [hlse]
    have := abs_add_le (Real.log Z) sh
    rwa [abs_of_nonneg hlogZ0] at this
  have hdb : |(FloatV.sub x mv).toReal| ≤ (1 + uR) * |a| + etaR := by
    have := abs_sub_abs_le_abs_sub (FloatV.sub x mv).toReal a
    nlinarith [abs_nonneg a]
  have hdl : |(FloatV.sub x mv).toReal - lse.toReal| ≤ (1 + uR) * |a| + etaR + Real.log Z + |sh| := by
    have := abs_sub (FloatV.sub x mv).toReal lse.toReal
    linarith
  have hroom2 : |(FloatV.sub x mv).toReal - lse.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    have h1 : (1 + uR) * |a| ≤ (1 + uR) * 2 ^ 127 := mul_le_mul_of_nonneg_left hgap (by linarith)
    have h2 : (1 + uR) * 2 ^ 127 + etaR + (66 / 5) + 1 ≤ ((2 : ℝ) ^ 24 - 1) * 2 ^ 104 := by
      unfold uR etaR; norm_num
    rw [hmax]; linarith
  obtain ⟨ho1, ho2⟩ := FloatV.sub_err_real b32_reg hs1 hlsefin hroom2
  rw [b32_u_cast, b32_eta_cast] at ho2
  have hoeq : o = FloatV.sub (FloatV.sub x mv) lse :=
    FloatV.NaNEq.eq ho (FloatV.isFinite_iff.mp ho1).1
  subst hoeq
  refine ⟨ho1, ?_⟩
  have htarget : (a - Real.log Z) - sh = a - lse.toReal := by rw [hlse]; ring
  rw [htarget]
  have hL : |a - Real.log Z| = |a| + Real.log Z := by
    rw [abs_of_nonpos (by linarith), abs_of_nonpos ha0]; ring
  rw [hL]
  have h1 : |(FloatV.sub (FloatV.sub x mv) lse).toReal - (a - lse.toReal)| ≤
      |(FloatV.sub (FloatV.sub x mv) lse).toReal - ((FloatV.sub x mv).toReal - lse.toReal)| +
        |(FloatV.sub x mv).toReal - a| := by
    have := abs_add_le ((FloatV.sub (FloatV.sub x mv) lse).toReal -
      ((FloatV.sub x mv).toReal - lse.toReal)) ((FloatV.sub x mv).toReal - a)
    rw [show (FloatV.sub (FloatV.sub x mv) lse).toReal - ((FloatV.sub x mv).toReal - lse.toReal) +
      ((FloatV.sub x mv).toReal - a) = (FloatV.sub (FloatV.sub x mv) lse).toReal - (a - lse.toReal)
      by ring] at this
    exact this
  have h2 : |(FloatV.sub x mv).toReal - a| ≤ uR * |a| + etaR := by
    have := hs2; rw [mul_comm] at this; exact this
  have h3 : |(FloatV.sub (FloatV.sub x mv) lse).toReal - ((FloatV.sub x mv).toReal - lse.toReal)| ≤
      uR * ((1 + uR) * |a| + etaR + Real.log Z + |sh|) + etaR := by
    have h4 := ho2
    rw [mul_comm] at h4
    have h5 := mul_le_mul_of_nonneg_left hdl hu.le
    linarith
  have hel : (elemTol : ℝ) = 4 * uR := by unfold elemTol uR; push_cast; norm_num
  have hfl : (floorTol : ℝ) = 2 ^ (-37 : ℤ) := by unfold floorTol; push_cast; ring
  rw [hel, hfl]
  have h6 : uR * |sh| + uR * etaR + 2 * etaR ≤ (2 : ℝ) ^ (-37 : ℤ) := by
    have h7 : uR * |sh| ≤ uR * 2 ^ (-14 : ℤ) := mul_le_mul_of_nonneg_left hshb hu.le
    have h8 : uR * 2 ^ (-14 : ℤ) + uR * etaR + 2 * etaR ≤ (2 : ℝ) ^ (-37 : ℤ) := by
      unfold uR etaR; norm_num
    linarith
  have h9 : uR * uR * |a| ≤ uR * |a| := by
    have : uR ≤ 1 := by unfold uR; norm_num
    have h10 : 0 ≤ uR * |a| := by positivity
    nlinarith
  generalize (2 : ℝ) ^ (-37 : ℤ) = p37 at h6 ⊢
  have h11 : uR * ((1 + uR) * |a|) = uR * |a| + uR * uR * |a| := by ring
  have h12 : uR * ((1 + uR) * |a| + etaR + Real.log Z + |sh|) =
      uR * ((1 + uR) * |a|) + uR * etaR + uR * Real.log Z + uR * |sh| := by ring
  have h13 : 4 * uR * (|a| + Real.log Z) = 4 * uR * |a| + 4 * uR * Real.log Z := by ring
  have h14 : 0 ≤ uR * |a| := by positivity
  have h15 : 0 ≤ uR * Real.log Z := by positivity
  linarith

/-- **An output of a `-∞` logit** is `-∞`: `−∞ − mv = −∞` and `−∞ − lse = −∞` for finite `mv`,
`lse`. -/
theorem ninf_out {mv lse d o : F32} (hmvfin : mv.isFinite = true) (hlsefin : lse.isFinite = true)
    (hd : FloatV.NaNEq (FloatV.sub FloatV.negInf mv) d) (ho : FloatV.NaNEq (FloatV.sub d lse) o) :
    o = FloatV.negInf := by
  have hn : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN b32_ieee
  rw [FloatV.sub_negInf_left b32_ieee hmvfin] at hd
  have hdeq : d = FloatV.negInf := FloatV.NaNEq.eq hd hn
  subst hdeq
  rw [FloatV.sub_negInf_left b32_ieee hlsefin] at ho
  exact FloatV.NaNEq.eq ho hn

/-- The claim's log-softmax in terms of the row's max: `L = x − mv − log ZR`. -/
lemma logSoftmax_eq {xs : Nat → F32} {V j : Nat} {mv : F32} (hZ : 0 < ZR mv xs V) :
    logSoftmax xs V j = (xs j).toReal - mv.toReal - Real.log (ZR mv xs V) := by
  unfold logSoftmax
  rw [expIn_sum mv xs V, Real.log_mul (Real.exp_pos _).ne' hZ.ne', Real.log_exp]
  simp only [mR, FloatV.toReal]
  ring

/-- **The numerics of one row.** On a `RowOk` row with max `mv = xs i0`, after the sum loop
(`SumInv` after `T` trips covering the row) and `lse = tl.log(se)`: there is one shift `sh`, within
`lseTol`, such that every output `(x_j − mv) − lse` is `-∞` for a `-∞` logit and otherwise finite
and within the claim's tolerance of `logSoftmax − sh`. -/
theorem row_num {P : Platform} (hexp : Device.ExpF32 P) (hninf : Device.ExpF32NegInf P)
    (hlog : Device.LogF32 P) {V T : Nat} (hV19 : V ≤ 2 ^ 19) (hVT : V ≤ T * 1024)
    (hTV : T * 1024 < V + 1024) {xs : Nat → F32} (hrow : RowOk xs V) {mv : F32} {i0 : Nat}
    (hi0 : i0 < V) (hmv : mv = xs i0) (hmvfin : mv.isFinite = true)
    (hle : ∀ k, k < V → (xs k).toRat ≤ mv.toRat) {se lse : F32} (hs : SumInv P V xs mv T se)
    (hl : P.math .log .binary32 se lse) :
    lse.isFinite = true ∧ ∃ sh : ℝ, |sh| ≤ (lseTol V : ℝ) ∧
      ∀ j, j < V → ∀ d o : F32, FloatV.NaNEq (FloatV.sub (xs j) mv) d →
        FloatV.NaNEq (FloatV.sub d lse) o →
        (xs j = FloatV.negInf → o = FloatV.negInf) ∧
        (xs j ≠ FloatV.negInf → o.isFinite = true ∧
          |o.toReal - (logSoftmax xs V j - sh)| ≤
            (elemTol : ℝ) * |logSoftmax xs V j| + (floorTol : ℝ)) := by
  obtain ⟨hsefin, hse⟩ := sum_bound hexp hninf hV19 hVT hTV hi0 hmv hmvfin hrow.1 hle hs
  have hmvne : mv ≠ FloatV.negInf := finite_ne_negInf hmvfin
  have hE1 : ∀ k, k < V → ER mv (xs k) ≤ 1 := by
    intro k hk
    by_cases hneg : xs k = FloatV.negInf
    · rw [hneg, ER_eq_zero rfl]; norm_num
    · refine ER_le_one hneg ?_
      rw [mR]; exact_mod_cast hle k hk
  have hZ1 : 1 ≤ ZR mv xs V := one_le_ZR hi0 (by rw [← hmv]; exact ER_self hmvne)
  have hZV : ZR mv xs V ≤ (V : ℝ) := ZR_le_n hE1
  have hV19r : (V : ℝ) ≤ 2 ^ 19 := by exact_mod_cast hV19
  obtain ⟨hlfin, hlerr⟩ := lse_bound hlog hV19 hTV hZ1 hZV hsefin hse hl
  refine ⟨hlfin, lse.toReal - Real.log (ZR mv xs V), hlerr, fun j hj d o hd ho => ⟨?_, ?_⟩⟩
  · intro hneg
    rw [hneg] at hd
    exact ninf_out hmvfin hlfin hd ho
  · intro hne
    have hxfin : (xs j).isFinite = true := (hrow.1 j hj).resolve_right hne
    have hgap : |(xs j).toReal - mv.toReal| ≤ 2 ^ 127 := by
      rw [hmv]; exact hrow.2.2 j i0 hj hi0 hxfin (by rw [← hmv]; exact hmvfin)
    have hshb : |lse.toReal - Real.log (ZR mv xs V)| ≤ 2 ^ (-14 : ℤ) := by
      refine hlerr.trans ?_
      unfold lseTol
      have : ((2 ^ (-34 : ℤ) * (V : ℚ) + 2 ^ (-16 : ℤ) : ℚ) : ℝ) =
          (2 : ℝ) ^ (-34 : ℤ) * (V : ℝ) + 2 ^ (-16 : ℤ) := by push_cast; ring
      rw [this]
      have h2 : (2 : ℝ) ^ (-34 : ℤ) * 2 ^ 19 + 2 ^ (-16 : ℤ) ≤ 2 ^ (-14 : ℤ) := by norm_num
      have h3 : (0 : ℝ) ≤ 2 ^ (-34 : ℤ) := by positivity
      nlinarith
    rw [logSoftmax_eq (lt_of_lt_of_le one_pos hZ1)]
    exact out_bound hmvfin hxfin (hle j hj) hgap hZ1 (hZV.trans hV19r) hlfin rfl hshb hd ho

end Claims.TopkLogSoftmax
