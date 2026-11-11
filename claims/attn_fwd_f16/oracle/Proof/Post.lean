import Proof.Link
import Proof.Outer

/-!
# The post-condition of one row

From the state of a row after the last block (`StateOK`) and the lanes of the epilogue (`log2`,
the sum, the division, the conversion), the claim's statement for the row's elements.
-/

namespace Claims.AttnFwd

open TVA Finset

noncomputable section

variable {d : Inst}

/-- For valid instances, `sigma` is positive. -/
theorem sigma_pos (hd : d.ok) : 0 < sigma d := by
  have := hd.scale.2.1
  unfold sigma
  have h : (0 : ℚ) < 1 / 2 ^ 64 := by norm_num
  exact lt_of_lt_of_le h this

/-- Row `n` is among its own attended keys. -/
theorem keys_self {n : Nat} (hn : n < d.N_CTX) : n ∈ keys d n := by
  simp [keys, hn]

/-- The score magnitude is nonnegative. -/
theorem scoreMag_nonneg (init : Mem) (bh n j : Nat) : 0 ≤ scoreMag d init bh n j := by
  unfold scoreMag
  exact Finset.sum_nonneg fun e _ =>
    mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)

/-- The bound `X` on the scaled score magnitudes of a row is in range. -/
theorem par_of_X (hd : d.ok) {init : Mem} {bh n : Nat} (hn : n < d.N_CTX) {X : ℚ}
    (hX : X ≤ 2 ^ 10) (hk : ∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) :
    Par d.HEAD_DIM (X : ℝ) := by
  refine ⟨hd.head_dim, ?_, ?_⟩
  · have h1 := hk n (keys_self hn)
    have h2 : 0 ≤ sigma d * scoreMag d init bh n n :=
      mul_nonneg (sigma_pos hd).le (scoreMag_nonneg init bh n n)
    exact_mod_cast h2.trans h1
  · exact_mod_cast (hX.trans (by norm_num))

/-- `tl.math.log2` on a `float32` in `[1/2, 2^18]`: finite, within `q22·|log₂ x| + 2^-149`. -/
theorem log2_fact {P : Platform} (hP : Device P) {x w : F32} (hxw : P.math .log2 .binary32 x w)
    (hx : x.isFinite = true) (hlo : 1 / 2 ≤ x.toReal) (hhi : x.toReal ≤ 2 ^ 18) :
    w.isFinite = true ∧ |w.toReal - Real.logb 2 x.toReal| ≤ q22 * |Real.logb 2 x.toReal| + 1 / 2 ^ 149 := by
  obtain ⟨ref, href, h⟩ := hP.log2F32
  have hr : ref = Real.logb 2 := by
    simp only [MathFn.ref, Option.some.injEq] at href; exact href.symm
  subst hr
  have hpos : 0 < x.toReal := by linarith
  have hlb : |Real.logb 2 x.toReal| ≤ 18 := by
    rw [abs_le]
    constructor
    · have : Real.logb 2 (1 / 2) ≤ Real.logb 2 x.toReal :=
        Real.logb_le_logb_of_le (by norm_num) (by norm_num) hlo
      have e : Real.logb 2 (1 / 2 : ℝ) = -1 := by
        rw [one_div, Real.logb_inv, Real.logb_self_eq_one (by norm_num)]
      linarith
    · have : Real.logb 2 x.toReal ≤ Real.logb 2 (2 ^ 18) :=
        Real.logb_le_logb_of_le (by norm_num) hpos hhi
      have e : Real.logb 2 ((2 : ℝ) ^ 18) = 18 := by
        rw [show ((2 : ℝ) ^ 18) = (2 : ℝ) ^ ((18 : ℕ) : ℝ) by norm_num,
          Real.logb_rpow (by norm_num) (by norm_num)]
        norm_num
      linarith
  obtain ⟨hf, he⟩ := h x w hxw hx (Set.mem_univ _) (Set.mem_Ioi.mpr hpos) (by
    rw [Format.binary32_maxVal_real]
    have : (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) := by positivity
    have h1 : (2 : ℝ) ^ (-22 : ℤ) ≤ 1 := by norm_num
    have h2 : (2 : ℝ) ^ (-149 : ℤ) ≤ 1 := by norm_num
    nlinarith [abs_nonneg (Real.logb 2 x.toReal)])
  refine ⟨hf, ?_⟩
  have e22 : (2 : ℝ) ^ (-22 : ℤ) = q22 := by unfold q22; norm_num
  have e149 : (2 : ℝ) ^ (-149 : ℤ) = 1 / 2 ^ 149 := by norm_num
  rw [e22, e149] at he
  exact he

/-- The default `/` on `float32` (`div.full`): finite, within `q22·|x/y| + 2^-148`, for a quotient
of magnitude at most `2^126`. -/
theorem div_fact {P : Platform} (hP : Device P) {x y q : F32} (hxy : P.div .binary32 x y q)
    (hx : x.isFinite = true) (hy : y.isFinite = true) (hy0 : y.toReal ≠ 0)
    (hbig : |x.toReal / y.toReal| ≤ 2 ^ 126) :
    q.isFinite = true ∧ |q.toReal - x.toReal / y.toReal| ≤ q22 * |x.toReal / y.toReal| + 1 / 2 ^ 148 := by
  have h := hP.divFull x y q hxy hx hy (by
    intro h0; apply hy0; unfold FloatV.toReal; rw [h0]; simp) (by
    have : ((|x.toRat / y.toRat| : ℚ) : ℝ) ≤ ((2 ^ 126 : ℚ) : ℝ) := by
      push_cast
      simpa [FloatV.toReal, Rat.cast_div, abs_div] using hbig
    exact_mod_cast this)
  obtain ⟨hf, he⟩ := h
  refine ⟨hf, ?_⟩
  have he' : ((|q.toRat - x.toRat / y.toRat| : ℚ) : ℝ) ≤
      (((2 : ℚ) ^ (-22 : ℤ) * |x.toRat / y.toRat| + 2 ^ (-148 : ℤ) : ℚ) : ℝ) := by exact_mod_cast he
  push_cast at he'
  simpa [FloatV.toReal, q22, abs_div, div_eq_mul_inv, mul_comm] using he'

/-- The reference quantities agree with the row-context ones on the covered tile range. -/
theorem ref_eqs {init : Mem} {bh n T : Nat} (h : HeadOk d init bh) (hT : keys d n ⊆ Finset.range (T * 64))
    (e : Nat) (m : ℝ) :
    (rowCtx d init bh n h).As e m T / (rowCtx d init bh n h).Ls m T = attnRef d init bh n e ∧
    (rowCtx d init bh n h).Ms e m T / (rowCtx d init bh n h).Ls m T = attnAbs d init bh n e := by
  rw [rowCtx_As h hT, rowCtx_Ms h hT, rowCtx_Ls h hT]
  exact ⟨mul_div_mul_left _ _ (Real.exp_pos _).ne', mul_div_mul_left _ _ (Real.exp_pos _).ne'⟩

/-- `vAbsSum ≤ N · attnMag` (the plain mean of `|v|` is at most `attnMag`). -/
theorem vAbs_le {init : Mem} {bh n : Nat} (hn : n < d.N_CTX) (e : Nat) :
    (vAbsSum d init bh n e : ℝ) ≤ d.N_CTX * attnMag d init bh n e := by
  have hcard : 0 < ((keys d n).card : ℝ) := by
    exact_mod_cast Finset.card_pos.mpr ⟨n, keys_self hn⟩
  have hcN : ((keys d n).card : ℝ) ≤ d.N_CTX := by
    have : (keys d n).card ≤ d.N_CTX := by
      calc (keys d n).card ≤ (Finset.range d.N_CTX).card := Finset.card_le_card (Finset.filter_subset _ _)
        _ = d.N_CTX := Finset.card_range _
    exact_mod_cast this
  have h1 : (vAbsSum d init bh n e : ℝ) / ((keys d n).card : ℝ) ≤ attnMag d init bh n e :=
    le_max_right _ _
  have hV0 : 0 ≤ (vAbsSum d init bh n e : ℝ) := by
    unfold vAbsSum; push_cast
    exact Finset.sum_nonneg fun j _ => by positivity
  have hA0 : 0 ≤ attnMag d init bh n e :=
    (div_nonneg hV0 hcard.le).trans h1
  rw [div_le_iff₀ hcard] at h1
  nlinarith

/-- The state of a row after the last block, as `StateOK` gives it. -/
theorem state_row {init : Mem} {bh sm T : Nat} {mt lt : Tensor [128] F32}
    {acc : Tensor [128, d.HEAD_DIM] F32} (hst : StateOK d init bh sm T mt lt acc)
    (h : HeadOk d init bh) (r : Fin 128) (hT : 1 ≤ T) :
    (mt.get1 r).isFinite = true ∧ (lt.get1 r).isFinite = true ∧
      (∀ e : Fin d.HEAD_DIM, (acc.get2 r e).isFinite = true) ∧
      ∀ X : ℝ, Par d.HEAD_DIM X →
        (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
        ∃ mf : ℝ, InvM (rowCtx d init bh (sm * 128 + r) h) X T (mt.get1 r).toReal mf ∧
          InvL (rowCtx d init bh (sm * 128 + r) h) X T (mt.get1 r).toReal mf (lt.get1 r).toReal ∧
          InvA (rowCtx d init bh (sm * 128 + r) h) X T (mt.get1 r).toReal mf
            (fun e => if h : e < d.HEAD_DIM then (acc.get2 r ⟨e, h⟩).toReal else 0) := by
  rcases hst h r with ⟨h0, -⟩ | ⟨-, h1, h2, h3, h4⟩
  · omega
  · exact ⟨h1, h2, h3, h4⟩

/-- **`M` of a row**: `m + log₂ l` (device `log2`, then the sum) is within `epsM` of the exact
log-sum-exp. -/
theorem row_M {P : Platform} (hP : Device P) (hd : d.ok) {init : Mem} {bh sm T : Nat}
    (hsm : sm < G d) (hT2 : 2 ≤ T) (hT1 : T ≤ 1024) (_hTN : 64 * T ≤ d.N_CTX)
    (hTk : ∀ r : Fin 128, keys d (sm * 128 + r) ⊆ Finset.range (T * 64))
    {mt lt : Tensor [128] F32} {acc : Tensor [128, d.HEAD_DIM] F32}
    (hst : StateOK d init bh sm T mt lt acc) (r : Fin 128) {w Mh : F32}
    (hw : P.math .log2 .binary32 (lt.get1 r) w)
    (hM : FloatV.NaNEq (FloatV.add (mt.get1 r) w) Mh) (h : HeadOk d init bh) :
    Mh.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
      (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
      |Mh.toReal - lseRef d init bh (sm * 128 + r)| ≤ (epsM d.HEAD_DIM X : ℝ) := by
  have hn : sm * 128 + (r : Nat) < d.N_CTX := sm_row_lt hd hsm r.isLt
  obtain ⟨hmf, hlf, haf, hX⟩ := state_row hst h r (by omega)
  have hreg : Format.binary32.Regular := Format.Std.regular Format.binary32_std
  -- finiteness, with the largest covered bound
  have hpar0 : Par d.HEAD_DIM 1024 := ⟨hd.head_dim, by norm_num, by norm_num⟩
  have hk0 : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ (1024 : ℝ) := by
    intro j hj
    have hjN : j < d.N_CTX := by
      simp only [keys, Finset.mem_filter, Finset.mem_range] at hj; exact hj.1
    have := h.2.2 (sm * 128 + r) j hn hjN
    have h2 : ((sigma d * scoreMag d init bh (sm * 128 + r) j : ℚ) : ℝ) ≤ ((2 ^ 10 : ℚ) : ℝ) := by
      exact_mod_cast this
    push_cast at h2; linarith
  obtain ⟨mf0, hM0, hL0, hA0⟩ := hX 1024 hpar0 hk0
  have hX' := hX
  have hbox0 := Box.of_par hpar0 hT2 hT1
  have hDpos : 0 < d.HEAD_DIM := by rcases hd.head_dim with h | h | h <;> omega
  obtain ⟨-, hl12, hl2, -, -⟩ := ratio_facts hpar0 hT2 hT1 hM0 hL0 hA0 (e := 0) hDpos
  obtain ⟨hwf, hwe⟩ := log2_fact hP hw hlf hl12 hl2
  have hlb : |Real.logb 2 (lt.get1 r).toReal| ≤ 18 := by
    have hl0 : 0 < (lt.get1 r).toReal := by linarith
    rw [abs_le]
    constructor
    · have : Real.logb 2 (1 / 2) ≤ Real.logb 2 (lt.get1 r).toReal :=
        Real.logb_le_logb_of_le (by norm_num) (by norm_num) hl12
      have e : Real.logb 2 (1 / 2 : ℝ) = -1 := by
        rw [one_div, Real.logb_inv, Real.logb_self_eq_one (by norm_num)]
      linarith
    · have : Real.logb 2 (lt.get1 r).toReal ≤ Real.logb 2 (2 ^ 18) :=
        Real.logb_le_logb_of_le (by norm_num) hl0 hl2
      have e : Real.logb 2 ((2 : ℝ) ^ 18) = 18 := by
        rw [show ((2 : ℝ) ^ 18) = (2 : ℝ) ^ ((18 : ℕ) : ℝ) by norm_num,
          Real.logb_rpow (by norm_num) (by norm_num)]
        norm_num
      linarith
  have hq0 : 0 ≤ q22 := by unfold q22; norm_num
  have hwa : |w.toReal| ≤ 19 := by
    have := abs_sub_abs_le_abs_sub w.toReal (Real.logb 2 (lt.get1 r).toReal)
    have h1 : q22 * |Real.logb 2 (lt.get1 r).toReal| ≤ q22 * 18 := mul_le_mul_of_nonneg_left hlb hq0
    have : q22 * 18 + 1 / 2 ^ 149 ≤ 1 := by unfold q22; norm_num
    linarith
  have hmabs : |(mt.get1 r).toReal| ≤ Bm 1024 := hM0.hm
  have hBm : Bm 1024 ≤ 1500 := by unfold Bm; norm_num
  have hsumabs : |(mt.get1 r).toReal + w.toReal| ≤ 2000 := by
    refine (abs_add_le _ _).trans ?_; linarith
  obtain ⟨haddf, hadde⟩ := FloatV.add_err_real hreg hmf hwf (by
    rw [Format.binary32_maxVal_real]; linarith [hsumabs])
  have hMeq : Mh = FloatV.add (mt.get1 r) w := hM.eq_of_isFinite haddf
  have hMfin : Mh.isFinite = true := by rw [hMeq]; exact haddf
  have hadde' : |Mh.toReal - ((mt.get1 r).toReal + w.toReal)| ≤
      u32 * |(mt.get1 r).toReal + w.toReal| + 1 / 2 ^ 150 := by
    rw [hMeq]
    have := hadde
    rw [Format.binary32_u_real, Format.binary32_eta_real] at this
    have e24 : (2 : ℝ) ^ (-24 : ℤ) = u32 := by unfold u32; norm_num
    have e150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
    rw [e24, e150] at this
    linarith [mul_comm |(mt.get1 r).toReal + w.toReal| u32]
  refine ⟨hMfin, fun X hX hk => ?_⟩
  have hPar := par_of_X hd hn hX hk
  have hkR : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ (X : ℝ) := by
    intro j hj
    have := hk j hj
    exact_mod_cast this
  obtain ⟨mfX, hMX, hLX, hAX⟩ := hX' X hPar hkR
  have hboxX := Box.of_par hPar hT2 hT1
  obtain ⟨hrel, hl1', hl2', -, -⟩ := ratio_facts hPar hT2 hT1 hMX hLX hAX (e := 0) hDpos
  have hLc := hMX.hlow (by omega)
  have hlse := lse_bound hboxX hMX.hm hrel hLc hl1' hl2' hwe hadde'
  have hmrel := relax_m hboxX
  have hrl := rowCtx_lse h (hTk r) ⟨_, keys_self hn⟩ ((mt.get1 r).toReal) (T := T)
  rw [hrl]
  have hcast : (epsM d.HEAD_DIM X : ℝ) = epsM' d.HEAD_DIM (X : ℝ) := by
    unfold epsM epsM'; push_cast; ring
  rw [hcast]
  linarith

/-- **An output element of a row**: `acc / l` (device `/`, then the `float16` conversion) is within
`epsO·attnMag + 2^-24` of the exact attention. -/
theorem row_O {P : Platform} (hP : Device P) (hd : d.ok) {init : Mem} {bh sm T : Nat}
    (hsm : sm < G d) (hT2 : 2 ≤ T) (hT1 : T ≤ 1024) (hTN : 64 * T ≤ d.N_CTX)
    (hTk : ∀ r : Fin 128, keys d (sm * 128 + r) ⊆ Finset.range (T * 64))
    {mt lt : Tensor [128] F32} {acc : Tensor [128, d.HEAD_DIM] F32}
    (hst : StateOK d init bh sm T mt lt acc) (r : Fin 128) (e : Fin d.HEAD_DIM) {qh : F32} {o : F16}
    (hq : P.div .binary32 (acc.get2 r e) (lt.get1 r) qh)
    (ho : FloatV.NaNEq (FloatV.cvtWith .rn qh) o) (h : HeadOk d init bh) :
    o.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
      (∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ X) →
      |o.toReal - attnRef d init bh (sm * 128 + r) e| ≤
        (epsO d.HEAD_DIM d.N_CTX X : ℝ) * attnMag d init bh (sm * 128 + r) e + 1 / 2 ^ 24 := by
  have hn : sm * 128 + (r : Nat) < d.N_CTX := sm_row_lt hd hsm r.isLt
  obtain ⟨hmf, hlf, haf, hX⟩ := state_row hst h r (by omega)
  have hreg : Format.binary32.Regular := Format.Std.regular Format.binary32_std
  have hreg16 : Format.binary16.Regular := Format.Std.regular Format.binary16_std
  have hDpos : 0 < d.HEAD_DIM := by rcases hd.head_dim with h | h | h <;> omega
  have hpar0 : Par d.HEAD_DIM 1024 := ⟨hd.head_dim, by norm_num, by norm_num⟩
  have hk0 : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ (1024 : ℝ) := by
    intro j hj
    have hjN : j < d.N_CTX := by
      simp only [keys, Finset.mem_filter, Finset.mem_range] at hj; exact hj.1
    have := h.2.2 (sm * 128 + r) j hn hjN
    have h2 : ((sigma d * scoreMag d init bh (sm * 128 + r) j : ℚ) : ℝ) ≤ ((2 ^ 10 : ℚ) : ℝ) := by
      exact_mod_cast this
    push_cast at h2; linarith
  have hX' := hX
  obtain ⟨mf0, hM0, hL0, hA0⟩ := hX 1024 hpar0 hk0
  -- the quotient of the largest covered instance
  obtain ⟨-, hl12, hl2, -, hq50⟩ := ratio_facts hpar0 hT2 hT1 hM0 hL0 hA0 (e := e) (by exact e.isLt)
  simp only [dite_eq_left e.isLt, Fin.eta] at hq50
  have hl0 : (lt.get1 r).toReal ≠ 0 := by intro h0; rw [h0] at hl12; norm_num at hl12
  obtain ⟨hqf, hqe⟩ := div_fact hP hq (haf e) hlf hl0 (by
    refine hq50.trans ?_; norm_num)
  have hqabs : |qh.toReal| ≤ 65504 := by
    have h1 := abs_sub_abs_le_abs_sub qh.toReal ((acc.get2 r e).toReal / (lt.get1 r).toReal)
    have h2 : q22 * |(acc.get2 r e).toReal / (lt.get1 r).toReal| ≤ q22 * 50000 :=
      mul_le_mul_of_nonneg_left hq50 (by unfold q22; norm_num)
    have : q22 * 50000 + 1 / 2 ^ 148 ≤ 1 := by unfold q22; norm_num
    linarith
  obtain ⟨hcf, hce⟩ := FloatV.cvt_err_real (g := Format.binary16) hreg16 hqf (by
    rw [Format.binary16_maxVal_real]; exact hqabs)
  have hoeq : o = FloatV.cvt qh := by
    have := ho.eq_of_isFinite (by rw [FloatV.cvtWith_rn]; exact hcf)
    rw [FloatV.cvtWith_rn] at this
    exact this.symm ▸ rfl
  refine ⟨by rw [hoeq]; exact hcf, fun X hX2 hk => ?_⟩
  have hPar := par_of_X hd hn hX2 hk
  have hkR : ∀ j ∈ keys d (sm * 128 + r), sigma d * scoreMag d init bh (sm * 128 + r) j ≤ (X : ℝ) := by
    intro j hj
    have := hk j hj
    exact_mod_cast this
  obtain ⟨mfX, hMX, hLX, hAX⟩ := hX' X hPar hkR
  have hboxX := Box.of_par hPar hT2 hT1
  obtain ⟨-, hl1', hl2', hrat, -⟩ := ratio_facts hPar hT2 hT1 hMX hLX hAX (e := e) (by exact e.isLt)
  simp only [dite_eq_left e.isLt, Fin.eta] at hrat
  obtain ⟨hEq1, hEq2⟩ := ref_eqs h (hTk r) e (mt.get1 r).toReal
  rw [rowCtx_Vs h (hTk r) e] at hrat
  rw [hEq1, hEq2] at hrat
  have hLs0 : 0 < (rowCtx d init bh (sm * 128 + r) h).Ls (mt.get1 r).toReal T := by
    have h1 : wlow d.HEAD_DIM (X : ℝ) ≤ (rowCtx d init bh (sm * 128 + r) h).Ls (mt.get1 r).toReal T :=
      hMX.hlow (by omega)
    have h2 := hboxX.par.wlow_ge
    linarith
  have hAM := (rowCtx d init bh (sm * 128 + r) h).abs_As_le e (mt.get1 r).toReal T
  have hw0 : |attnRef d init bh (sm * 128 + r) e| ≤ attnAbs d init bh (sm * 128 + r) e := by
    rw [← hEq1, ← hEq2, abs_div, abs_of_pos hLs0]
    exact div_le_div_of_nonneg_right hAM hLs0.le
  have hMs0 := (rowCtx d init bh (sm * 128 + r) h).Ms_nonneg e (mt.get1 r).toReal T
  have hattn : 0 ≤ attnAbs d init bh (sm * 128 + r) e := by rw [← hEq2]; exact div_nonneg hMs0 hLs0.le
  have hV : 0 ≤ (vAbsSum d init bh (sm * 128 + r) e : ℝ) := by
    unfold vAbsSum; push_cast
    exact Finset.sum_nonneg fun j _ => by positivity
  have ho' : |o.toReal - qh.toReal| ≤ u16 * |qh.toReal| + e16 := by
    rw [hoeq]
    have := hce
    rw [Format.binary16_u_real, Format.binary16_eta_real] at this
    have e11 : (2 : ℝ) ^ (-11 : ℤ) = u16 := by unfold u16; norm_num
    have e25 : (2 : ℝ) ^ (-25 : ℤ) = e16 := by unfold e16; norm_num
    rw [e11, e25] at this
    linarith [mul_comm |qh.toReal| u16]
  have hq' := out_bound hboxX hattn hV hw0 hrat hqe ho'
  have hmag0 : 0 ≤ attnMag d init bh (sm * 128 + r) e := hattn.trans (le_max_left _ _)
  have h1 : attnAbs d init bh (sm * 128 + r) e ≤ attnMag d init bh (sm * 128 + r) e := le_max_left _ _
  have h2 := vAbs_le (init := init) (bh := bh) hn e
  have hto : 0 ≤ tolO2 d.HEAD_DIM T X := by
    unfold tolO2; have := hboxX.tolO_nonneg; have : 0 ≤ lastRel := by unfold lastRel u16 q22; norm_num
    positivity
  have hts : 0 ≤ tolS2 d.HEAD_DIM T X := by
    unfold tolS2; have := hboxX.tolS_nonneg; have : 0 ≤ lastRel := by unfold lastRel u16 q22; norm_num
    positivity
  have hN1 : 64 * (T : ℝ) ≤ d.N_CTX := by exact_mod_cast hTN
  have hN2 : (d.N_CTX : ℝ) ≤ 2 ^ 16 := by exact_mod_cast hd.n_le
  have hro := relax_o hboxX hN1 hN2
  have hrf := relax_f hboxX
  have hcast : (epsO d.HEAD_DIM d.N_CTX X : ℝ) = epsO' d.HEAD_DIM d.N_CTX (X : ℝ) := by
    unfold epsO epsO'; push_cast; ring
  rw [hcast]
  have e1 : tolO2 d.HEAD_DIM T X * attnAbs d init bh (sm * 128 + r) e ≤
      tolO2 d.HEAD_DIM T X * attnMag d init bh (sm * 128 + r) e :=
    mul_le_mul_of_nonneg_left h1 hto
  have e2 : tolS2 d.HEAD_DIM T X * (vAbsSum d init bh (sm * 128 + r) e : ℝ) ≤
      tolS2 d.HEAD_DIM T X * (d.N_CTX * attnMag d init bh (sm * 128 + r) e) :=
    mul_le_mul_of_nonneg_left h2 hts
  have e3 := mul_le_mul_of_nonneg_right hro hmag0
  have e4 : (1 : ℝ) / 2 ^ 24 = 1 / 2 ^ 24 := rfl
  have hfin : |o.toReal - attnRef d init bh (sm * 128 + r) e| ≤
      tolO2 d.HEAD_DIM T X * attnAbs d init bh (sm * 128 + r) e +
      tolS2 d.HEAD_DIM T X * (vAbsSum d init bh (sm * 128 + r) e : ℝ) + tolF2 d.HEAD_DIM T X := hq'
  have e5 : (tolO2 d.HEAD_DIM T X + d.N_CTX * tolS2 d.HEAD_DIM T X) * attnMag d init bh (sm * 128 + r) e =
      tolO2 d.HEAD_DIM T X * attnMag d init bh (sm * 128 + r) e +
        tolS2 d.HEAD_DIM T X * (d.N_CTX * attnMag d init bh (sm * 128 + r) e) := by ring
  linarith

end

end Claims.AttnFwd
