import Proof.Stage

/-!
# One page of the loop, in numbers

From the values the operations of a trip return (the hypotheses are what the walk collects),
the new state satisfies the invariant: `weights_stage` (the new maximum, the rescaling, the
weights), `lcol_stage` (the sum `l`), `acol_stage` (one component of `acc`).
-/
namespace Claims.PagedAttention

open TVA Prog

/-- The scores' error is small on the covered range. -/
theorem dS_le {X : ℝ} (_hX0 : 0 ≤ X) (hX1 : X ≤ 1024) : Num.dS X ≤ 2 / 1000 := by
  unfold Num.dS Num.u; norm_num; nlinarith

/-- **The new maximum, the rescaling and the weights of a page**: the facts `lcol_stage` and `acol_stage` need. -/
theorem weights_stage {d : Inst} (_hd : d.ok) {E : Env} (hP : Device E.P) {m : Mem} {s h t c : Nat}
    {X : ℝ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024)
    (hSb : ∀ k, k < seqLen d m s → |Sx d m s h k| ≤ X)
    (hc1 : 1 ≤ c) (hcP : c ≤ d.PAGE_SIZE)
    (hcv : ∀ r : Fin d.PAGE_SIZE, r.val < c → t * d.PAGE_SIZE + r.val < seqLen d m s)
    {sc sp xl pp : Tensor [d.PAGE_SIZE] F32} {m_i mx mn al xa : F32}
    (hscore : ∀ r : Fin d.PAGE_SIZE, r.val < c → (sc.get1 r).isFinite = true ∧
      |(sc.get1 r).toReal - Sx d m s h (t * d.PAGE_SIZE + r.val)| ≤ Num.dS X)
    (hsp : ∀ r : Fin d.PAGE_SIZE, sp.get1 r = if r.val < c then sc.get1 r else FloatV.negInf)
    (hmx : ∃ tr : Tree, tr.Covers d.PAGE_SIZE ∧ tr.Eval (FloatV.MaxRel true)
      (fun j => if hj : j < d.PAGE_SIZE then sp.get1 ⟨j, hj⟩ else default) mx)
    (hmi : m_i = FloatV.negInf ∨ m_i.isFinite = true)
    (hmn : FloatV.MaxRel true m_i mx mn)
    (hold : 0 < t → m_i.isFinite = true ∧ -(Num.Kx X) ≤ m_i.toReal ∧ m_i.toReal ≤ Num.Kx X ∧
      (∀ k, k < t * d.PAGE_SIZE → Sx d m s h k ≤ m_i.toReal + Num.dS X) ∧
      ∃ k, k < t * d.PAGE_SIZE ∧ m_i.toReal ≤ Sx d m s h k + Num.dS X)
    (hm0 : t = 0 → m_i = FloatV.negInf)
    (hxa : (FloatV.sub m_i mn).NaNEq xa) (hal : E.P.math .exp .binary32 xa al)
    (hxl : ∀ r : Fin d.PAGE_SIZE, (FloatV.sub (sp.get1 r) mn).NaNEq (xl.get1 r))
    (hpp : ∀ r : Fin d.PAGE_SIZE, E.P.math .exp .binary32 (xl.get1 r) (pp.get1 r)) :
    mn.isFinite = true ∧ -(Num.Kx X) ≤ mn.toReal ∧ mn.toReal ≤ Num.Kx X ∧
      (∀ k, k < t * d.PAGE_SIZE + c → Sx d m s h k ≤ mn.toReal + Num.dS X) ∧
      (∃ k, k < t * d.PAGE_SIZE + c ∧ mn.toReal ≤ Sx d m s h k + Num.dS X) ∧
      (0 < t → m_i.toReal ≤ mn.toReal ∧ al.isFinite = true ∧
        |al.toReal - Real.exp (m_i.toReal - mn.toReal)| ≤
          (Num.aA0 + Num.aA1 * (mn.toReal - m_i.toReal)) * Real.exp (m_i.toReal - mn.toReal) +
            Num.zeta0 ∧ |al.toReal| ≤ 2) ∧
      (t = 0 → al = FloatV.ofPat 0) ∧
      (∀ r : Fin d.PAGE_SIZE, r.val < c → (pp.get1 r).isFinite = true ∧ |(pp.get1 r).toReal| ≤ 2 ∧
        |(pp.get1 r).toReal - Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal)| ≤
          Num.rhoP X * Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal) + Num.etaP) ∧
      (∀ r : Fin d.PAGE_SIZE, c ≤ r.val → pp.get1 r = FloatV.ofPat 0) := by
  have hdS := dS_le hX0 hX1
  have hdS0 : 0 ≤ Num.dS X := by unfold Num.dS; have := Num.u_pos; positivity
  have hsp_lane : ∀ r : Fin d.PAGE_SIZE, r.val < c → sp.get1 r = sc.get1 r := by
    intro r hr; rw [hsp r, ite_eq_left hr]
  have hsp_neg : ∀ r : Fin d.PAGE_SIZE, c ≤ r.val → sp.get1 r = FloatV.negInf := by
    intro r hr; rw [hsp r, ite_eq_right (by omega)]
  obtain ⟨hmnf, hmi_le, hmn_ge, hmn_att⟩ := Num.max_new hc1 hcP sp
    (fun r hr => by rw [hsp_lane r hr]; exact (hscore r hr).1) hsp_neg m_i mx mn hmi hmx hmn
  -- scores of the page against the exact ones
  have hscl : ∀ r : Fin d.PAGE_SIZE, r.val < c →
      Sx d m s h (t * d.PAGE_SIZE + r.val) - Num.dS X ≤ (sp.get1 r).toReal ∧
        (sp.get1 r).toReal ≤ Sx d m s h (t * d.PAGE_SIZE + r.val) + Num.dS X := by
    intro r hr
    rw [hsp_lane r hr]
    have := (abs_le.mp (hscore r hr).2)
    constructor <;> linarith [this.1, this.2]
  have hSbr : ∀ r : Fin d.PAGE_SIZE, r.val < c → |Sx d m s h (t * d.PAGE_SIZE + r.val)| ≤ X :=
    fun r hr => hSb _ (hcv r hr)
  have hPpos : 0 < d.PAGE_SIZE := by omega
  have hK1 : Num.Kx X = X + 1 := rfl
  -- bounds on the new reference
  have hmn_K : -(Num.Kx X) ≤ mn.toReal ∧ mn.toReal ≤ Num.Kx X ∧
      (∃ k, k < t * d.PAGE_SIZE + c ∧ mn.toReal ≤ Sx d m s h k + Num.dS X) := by
    rcases hmn_att with ⟨r, hr, hreq⟩ | ⟨hmif, hmeq⟩
    · have h1 := hscl r hr
      have h2 := hSbr r hr
      have h3 := abs_le.mp h2
      rw [hreq]
      refine ⟨by linarith [h1.1, h3.1], by linarith [h1.2, h3.2], ⟨t * d.PAGE_SIZE + r.val, by omega, ?_⟩⟩
      linarith [h1.2]
    · have ht : 0 < t := by
        rcases Nat.eq_zero_or_pos t with h0 | h0
        · exfalso; have := hm0 h0; rw [this] at hmif
          exact absurd hmif (by rw [Num.negInf_notFinite]; simp)
        · exact h0
      obtain ⟨-, hK1', hK2', -, k, hk, hkle⟩ := hold ht
      rw [hmeq]
      exact ⟨hK1', hK2', ⟨k, by omega, hkle⟩⟩
  obtain ⟨hmK1, hmK2, hmatt⟩ := hmn_K
  have hall : ∀ k, k < t * d.PAGE_SIZE + c → Sx d m s h k ≤ mn.toReal + Num.dS X := by
    intro k hk
    by_cases hkt : k < t * d.PAGE_SIZE
    · have ht : 0 < t := by
        rcases Nat.eq_zero_or_pos t with h0 | h0
        · subst h0; simp at hkt
        · exact h0
      obtain ⟨hmif, -, -, hs1, -⟩ := hold ht
      have := hmi_le hmif
      have := hs1 k hkt
      linarith
    · have hr : k - t * d.PAGE_SIZE < c := by omega
      have hrP : k - t * d.PAGE_SIZE < d.PAGE_SIZE := by omega
      have h1 := hscl ⟨k - t * d.PAGE_SIZE, hrP⟩ hr
      have h2 := hmn_ge ⟨k - t * d.PAGE_SIZE, hrP⟩ hr
      have hk' : t * d.PAGE_SIZE + (k - t * d.PAGE_SIZE) = k := by omega
      simp only [hk'] at h1
      linarith [h1.1, h1.2, h2]
  -- alpha
  have hal_pos : 0 < t → m_i.toReal ≤ mn.toReal ∧ al.isFinite = true ∧
      |al.toReal - Real.exp (m_i.toReal - mn.toReal)| ≤
        (Num.aA0 + Num.aA1 * (mn.toReal - m_i.toReal)) * Real.exp (m_i.toReal - mn.toReal) + Num.zeta0 ∧
      |al.toReal| ≤ 2 := by
    intro ht
    obtain ⟨hmif, hk1, hk2, -⟩ := hold ht
    have hle := hmi_le hmif
    refine ⟨hle, ?_⟩
    obtain ⟨hf, he⟩ := Num.alpha_ok hP hmif hmnf rfl rfl hle (by unfold Num.Kx at *; linarith) hxa hal
    refine ⟨hf, he, ?_⟩
    have he1 : Real.exp (m_i.toReal - mn.toReal) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hd0 : 0 ≤ mn.toReal - m_i.toReal := by linarith
    have hrho : Num.aA0 + Num.aA1 * (mn.toReal - m_i.toReal) ≤ 1 / 100 := by
      have : mn.toReal - m_i.toReal ≤ 2050 := by unfold Num.Kx at *; linarith
      have h1 : Num.aA1 * (mn.toReal - m_i.toReal) ≤ Num.aA1 * 2050 :=
        mul_le_mul_of_nonneg_left this (by unfold Num.aA1 Num.u; norm_num)
      unfold Num.aA0 Num.aA1 Num.u at *; norm_num at *; linarith
    have hz : Num.zeta0 ≤ 1 / 100 := by unfold Num.zeta0; norm_num
    have hepos := Real.exp_pos (m_i.toReal - mn.toReal)
    have h2 : (Num.aA0 + Num.aA1 * (mn.toReal - m_i.toReal)) * Real.exp (m_i.toReal - mn.toReal) ≤ 1 / 100 := by
      calc _ ≤ (1 / 100) * 1 := mul_le_mul hrho he1 hepos.le (by norm_num)
        _ = 1 / 100 := by norm_num
    have := abs_sub_abs_le_abs_sub al.toReal (Real.exp (m_i.toReal - mn.toReal))
    rw [abs_of_pos hepos] at this
    linarith
  have hal0 : t = 0 → al = FloatV.ofPat 0 := by
    intro h0
    have := hm0 h0
    rw [this] at hxa
    exact Num.alpha_first hP hmnf hxa hal
  -- the weights
  have hlanes : ∀ r : Fin d.PAGE_SIZE, r.val < c → (pp.get1 r).isFinite = true ∧ |(pp.get1 r).toReal| ≤ 2 ∧
      |(pp.get1 r).toReal - Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal)| ≤
        Num.rhoP X * Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal) + Num.etaP := by
    intro r hr
    have hsr := hsp_lane r hr
    have h1 := (hscore r hr)
    have hle := hmn_ge r hr
    obtain ⟨hf, he⟩ := Num.lane_ok hP hX0 hX1 (sh := sp.get1 r) (mn := mn) (xh := xl.get1 r)
      (ph := pp.get1 r) (Sk := Sx d m s h (t * d.PAGE_SIZE + r.val)) (μ' := mn.toReal)
      (by rw [hsr]; exact h1.1) hmnf rfl (hSbr r hr) (by rw [hsr]; exact h1.2) hle
      (by rw [abs_le]; unfold Num.Kx at *; constructor <;> linarith) (hxl r) (hpp r)
    refine ⟨hf, ?_, he⟩
    -- magnitude
    have hSle : Sx d m s h (t * d.PAGE_SIZE + r.val) ≤ mn.toReal + Num.dS X := hall _ (by omega)
    have hex1 : Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal) ≤ 11 / 10 := by
      have h2 : Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal) ≤ Real.exp (2 / 1000) :=
        Real.exp_le_exp.mpr (by linarith)
      refine h2.trans ?_
      refine (Num.exp_le_inv (by norm_num)).trans ?_
      norm_num
    have hepos := Real.exp_pos (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal)
    have hrho : Num.rhoP X ≤ 1 / 100 := by unfold Num.rhoP Num.u; norm_num; nlinarith
    have hρ0 : 0 ≤ Num.rhoP X := by unfold Num.rhoP; have := Num.u_pos; positivity
    have hη : Num.etaP ≤ 1 / 100 := by unfold Num.etaP; norm_num
    have h3 : Num.rhoP X * Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal) ≤ (1 / 100) * (11 / 10) :=
      mul_le_mul hrho hex1 hepos.le (by norm_num)
    have := abs_sub_abs_le_abs_sub (pp.get1 r).toReal (Real.exp (Sx d m s h (t * d.PAGE_SIZE + r.val) - mn.toReal))
    rw [abs_of_pos hepos] at this
    linarith
  have hmasked : ∀ r : Fin d.PAGE_SIZE, c ≤ r.val → pp.get1 r = FloatV.ofPat 0 := by
    intro r hr
    have := hxl r
    rw [hsp_neg r hr] at this
    exact Num.lane_masked hP hmnf this (hpp r)
  exact ⟨hmnf, hmK1, hmK2, hall, hmatt, hal_pos, hal0, hlanes, hmasked⟩

/-- The rates of `l` meet `col_step`'s conditions. -/
theorem rates_l {X : ℝ} (_hX0 : 0 ≤ X) (hX1 : X ≤ 1024) :
    Num.aA0 * (1 + Num.rhoOL) + Num.rhoOL ≤ Num.eL ∧ Num.aA1 * (1 + Num.rhoOL) ≤ Num.gam ∧
    Num.rhoP X + Num.rhoSL * (1 + Num.rhoP X) +
      Num.rhoOL * (1 + (Num.rhoP X + Num.rhoSL * (1 + Num.rhoP X))) ≤ Num.eps2L X ∧
    Num.eps2L X ≤ 48 * Num.u * X + 40 * Num.u ∧ Num.eL ≤ Num.eA := by
  unfold Num.aA0 Num.aA1 Num.rhoOL Num.eL Num.gam Num.rhoP Num.rhoSL Num.eps2L Num.eA Num.u
  refine ⟨by norm_num, by norm_num, ?_, by nlinarith, by norm_num⟩
  nlinarith

/-- The rates of `acc` meet `col_step`'s conditions. -/
theorem rates_a {X : ℝ} (_hX0 : 0 ≤ X) (hX1 : X ≤ 1024) :
    Num.aA0 * (1 + Num.rhoOA) + Num.rhoOA ≤ Num.eA ∧ Num.aA1 * (1 + Num.rhoOA) ≤ Num.gam ∧
    Num.rhoP X + Num.rhoSA * (1 + Num.rhoP X) +
      Num.rhoOA * (1 + (Num.rhoP X + Num.rhoSA * (1 + Num.rhoP X))) ≤ Num.eps2A X ∧
    Num.eps2A X ≤ 48 * Num.u * X + 40 * Num.u := by
  unfold Num.aA0 Num.aA1 Num.rhoOA Num.eA Num.gam Num.rhoP Num.rhoSA Num.eps2A Num.u
  refine ⟨by norm_num, by norm_num, ?_, by nlinarith⟩
  nlinarith

/-- The state of a column is bounded by `2^39` when its tokens are. -/
theorem colInv_bound {X : ℝ} (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) {e1 ε2 : ℝ} (he1 : 0 ≤ e1)
    (he1' : e1 ≤ Num.eA) (hε2 : 0 ≤ ε2) (hε2' : ε2 ≤ 48 * Num.u * X + 40 * Num.u)
    {S V : ℕ → ℝ} {N t : ℕ} {μ x : ℝ} (ht : t ≤ 2 ^ 16) (hN : N ≤ 2 ^ 22)
    (hV : ∀ k, k < N → |V k| ≤ 2 ^ 15) (hS : ∀ k, k < N → S k ≤ μ + Num.dS X)
    (hμ : μ ≤ Num.Kx X) (hμ' : -(Num.Kx X) ≤ μ)
    (h : Num.ColInv e1 Num.gam ε2 (Num.Kx X) Num.Gabs S V N t μ x) : |x| ≤ 2 ^ 39 := by
  have hu := Num.u_pos
  have hdS0 : 0 ≤ Num.dS X := by unfold Num.dS; positivity
  have hdS := dS_le hX0 hX1
  have hexpdS : Real.exp (Num.dS X) ≤ 2 := by
    refine (Num.exp_le_inv (by linarith)).trans ?_
    rw [div_le_iff₀ (by linarith)]; linarith
  have hM : Num.mS S V μ N ≤ 2 ^ 38 := by
    have := Num.mS_le S V (Vm := 2 ^ 15) (N := N) hdS0 (by norm_num) hS hV
    have hNr : (N : ℝ) ≤ 2 ^ 22 := by exact_mod_cast hN
    calc Num.mS S V μ N ≤ N * Real.exp (Num.dS X) * 2 ^ 15 := this
      _ ≤ 2 ^ 22 * 2 * 2 ^ 15 :=
          mul_le_mul (mul_le_mul hNr hexpdS (Real.exp_pos _).le (by positivity)) le_rfl
            (by positivity) (by positivity)
      _ = 2 ^ 38 := by norm_num
  have hEx := Num.Ex_le hX0 hX1 he1 he1' ht hμ hμ'
  have hFc := Num.Fc_le hX0 hX1 he1 he1' hε2 hε2' ht hμ hμ'
  have hGt : Num.Gabs * t * Num.Ex e1 Num.gam (Num.Kx X) t μ ≤ 1 := by
    have : (t : ℝ) ≤ 2 ^ 16 := by exact_mod_cast ht
    have hG : Num.Gabs * t ≤ Num.Gabs * 2 ^ 16 := mul_le_mul_of_nonneg_left this (by unfold Num.Gabs; positivity)
    have : Num.Gabs * 2 ^ 16 ≤ 1 / 2 ^ 60 := by unfold Num.Gabs; norm_num
    calc Num.Gabs * t * Num.Ex e1 Num.gam (Num.Kx X) t μ ≤ (1 / 2 ^ 60) * (21 / 20) :=
          mul_le_mul (hG.trans this) hEx (by unfold Num.Ex; positivity) (by positivity)
      _ ≤ 1 := by norm_num
  have h1 := Num.colInv_abs_le h
  have h2 : (1 + Num.Fc e1 Num.gam ε2 (Num.Kx X) t μ) * Num.mS S V μ N ≤ (17 / 16) * 2 ^ 38 :=
    mul_le_mul (by linarith) hM (by unfold Num.mS; positivity) (by norm_num)
  linarith

/-- **The sum `l` after a page** satisfies the invariant `InvL`. -/
theorem lcol_stage {d : Inst} {E : Env} (hP : Device E.P) {m : Mem} {s h t c : Nat} {X : ℝ}
    (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (hPpos : 0 < d.PAGE_SIZE) (hP256 : d.PAGE_SIZE ≤ 256)
    (hc1 : 1 ≤ c) (hcP : c ≤ d.PAGE_SIZE) (ht : t + 1 ≤ 2 ^ 16) (hN : t * d.PAGE_SIZE + c ≤ 2 ^ 22)
    {l_i r l' al : F32} {pp : Tensor [d.PAGE_SIZE] F32} {μ' : ℝ}
    (hμ'1 : -(Num.Kx X) ≤ μ') (hμ'2 : μ' ≤ Num.Kx X)
    (hlanes : ∀ i : Fin d.PAGE_SIZE, i.val < c → (pp.get1 i).isFinite = true ∧ |(pp.get1 i).toReal| ≤ 2 ∧
      |(pp.get1 i).toReal - Real.exp (Sx d m s h (t * d.PAGE_SIZE + i.val) - μ')| ≤
        Num.rhoP X * Real.exp (Sx d m s h (t * d.PAGE_SIZE + i.val) - μ') + Num.etaP)
    (hmasked : ∀ i : Fin d.PAGE_SIZE, c ≤ i.val → pp.get1 i = FloatV.ofPat 0)
    (hr : ∃ tr : Tree, tr.Covers d.PAGE_SIZE ∧ tr.depth ≤ tl.sumBound E.P d.PAGE_SIZE ∧
      tr.Eval (AddRel false) (fun i => if hi : i < d.PAGE_SIZE then pp.get1 ⟨i, hi⟩ else default) r)
    (hl' : (FloatV.add (FloatV.mul l_i al) r).NaNEq l' ∨
      (E.fusion = true ∧ (FloatV.fma l_i al r).NaNEq l'))
    (hold : 0 < t → ∃ μ : ℝ, l_i.isFinite = true ∧
      Num.InvL X (Sx d m s h) (t * d.PAGE_SIZE) t μ l_i.toReal ∧ μ ≤ μ' ∧ al.isFinite = true ∧
      |al.toReal - Real.exp (μ - μ')| ≤ (Num.aA0 + Num.aA1 * (μ' - μ)) * Real.exp (μ - μ') + Num.zeta0 ∧
      |al.toReal| ≤ 2 ∧ (∀ k, k < t * d.PAGE_SIZE → Sx d m s h k ≤ μ + Num.dS X) ∧
      -(Num.Kx X) ≤ μ ∧ μ ≤ Num.Kx X)
    (h0 : t = 0 → l_i = FloatV.ofPat 0 ∧ al = FloatV.ofPat 0) :
    l'.isFinite = true ∧ Num.InvL X (Sx d m s h) (t * d.PAGE_SIZE + c) (t + 1) μ' l'.toReal := by
  obtain ⟨hrfin, hr1024, hrerr⟩ := Num.lsum_block hP hPpos hP256 hcP pp r (fun i hi => ⟨(hlanes i hi).1, (hlanes i hi).2.1⟩) hmasked hr
  have hp' : ∀ j, j < c → |Num.phr pp j - Real.exp (Sx d m s h (t * d.PAGE_SIZE + j) - μ')| ≤
      Num.rhoP X * Real.exp (Sx d m s h (t * d.PAGE_SIZE + j) - μ') + Num.etaP := by
    intro j hj
    have hjP : j < d.PAGE_SIZE := by omega
    have := (hlanes ⟨j, hjP⟩ hj).2.2
    simpa [Num.phr, hjP] using this
  have hb' : |r.toReal - ∑ j ∈ Finset.range c, Num.phr pp j * (fun _ => (1 : ℝ)) (t * d.PAGE_SIZE + j)| ≤
      Num.rhoSL * ∑ j ∈ Finset.range c, |Num.phr pp j * (fun _ => (1 : ℝ)) (t * d.PAGE_SIZE + j)| +
        Num.etaS := by simpa using hrerr
  have hr' : |r.toReal| ≤ 1024 := hr1024
  have hrates := rates_l hX0 hX1
  have hc256 : c ≤ 256 := by omega
  have hu := Num.u_pos
  rcases Nat.eq_zero_or_pos t with h00 | ht0
  · subst h00
    obtain ⟨hl0, hal0⟩ := (h0 rfl)
    have hz : (FloatV.ofPat 0 : F32).toReal = 0 := by unfold FloatV.toReal; rw [Num.zero32_rat]; simp
    obtain ⟨hl'f, hl'e⟩ := Num.mulAdd32 (E := E) (a := l_i) (b := al) (c := r) (by rw [hl0]; exact Num.zero32_fin)
      (by rw [hal0]; exact Num.zero32_fin) hrfin (by
        rw [hl0, hal0, hz]; norm_num; linarith) hl'
    rw [hl0, hal0, hz] at hl'e
    simp only [zero_mul, zero_add, abs_zero] at hl'e
    refine ⟨hl'f, ?_⟩
    have := Num.col_first hX0 hX1 (Sx d m s h) (fun _ => 1) (c := c) (μ' := μ') (x' := l'.toReal)
      (b := r.toReal) (ρo := Num.rhoOL) (ρs := Num.rhoSL) (e1 := Num.eL) (ε2 := Num.eps2L X) (Vm := 1)
      (Num.phr pp) hc256 (by unfold Num.rhoOL; positivity) (by unfold Num.rhoSL; positivity)
      (by unfold Num.eL; positivity) (by norm_num) (by norm_num)
      (by unfold Num.rhoOL Num.u; norm_num) (by unfold Num.rhoSL Num.u; norm_num) hrates.2.2.1 hμ'1
      (fun k _ => by norm_num) (by simpa using hp') (by simpa using hb') (by simpa using hl'e)
    simpa [Num.InvL] using this
  · obtain ⟨μ, hlf, hxinv, hμμ', half, hal_err, hal2, hSold, hμ1, hμ2⟩ := hold ht0
    have hlb : |l_i.toReal| ≤ 2 ^ 39 :=
      colInv_bound hX0 hX1 (e1 := Num.eL) (by unfold Num.eL; positivity) hrates.2.2.2.2
        (by unfold Num.eps2L; positivity) hrates.2.2.2.1 (S := Sx d m s h) (V := fun _ => 1)
        (by omega) (by omega) (fun k _ => by norm_num) hSold hμ2 hμ1 hxinv
    obtain ⟨hl'f, hl'e⟩ := Num.mulAdd32 (E := E) (a := l_i) (b := al) (c := r) hlf half hrfin (by
      have h1 : |l_i.toReal * al.toReal| ≤ 2 ^ 39 * 2 := by
        rw [abs_mul]; exact mul_le_mul hlb hal2 (abs_nonneg _) (by norm_num)
      have : (2 : ℝ) ^ 39 * 2 + 1024 ≤ 2 ^ 100 := by norm_num
      linarith) hl'
    refine ⟨hl'f, ?_⟩
    have := Num.col_step hX0 hX1 (Sx d m s h) (fun _ => 1) (t := t) (N := t * d.PAGE_SIZE) (c := c)
      (μ := μ) (μ' := μ') (α := Real.exp (μ - μ')) (α' := al.toReal) (x := l_i.toReal)
      (x' := l'.toReal) (b := r.toReal) (ρo := Num.rhoOL) (ρs := Num.rhoSL) (e1 := Num.eL)
      (ε2 := Num.eps2L X) (Vm := 1) (Num.phr pp) ht0 (by omega) hc256 hN
      (by unfold Num.rhoOL; positivity) (by unfold Num.rhoSL; positivity)
      (by unfold Num.eL; positivity) (by norm_num) (by norm_num) hrates.1 hrates.2.1 hrates.2.2.1
      hrates.2.2.2.1 hrates.2.2.2.2 hμ1 hμμ' hμ'2 hSold (fun k _ => by norm_num) rfl hal_err hp' hb'
      hxinv (by
        have : al.toReal * l_i.toReal = l_i.toReal * al.toReal := mul_comm _ _
        rw [this]; exact hl'e)
    exact this
/-- **One component of `acc` after a page** satisfies the invariant `InvA`. -/
theorem acol_stage {d : Inst} {E : Env} (_hP : Device E.P) {m : Mem} {s h t c e : Nat} {X : ℝ}
    (hX0 : 0 ≤ X) (hX1 : X ≤ 1024) (hc1 : 1 ≤ c) (hcP : c ≤ d.PAGE_SIZE)
    (hP256 : d.PAGE_SIZE ≤ 256) (ht : t + 1 ≤ 2 ^ 16)
    (hN : t * d.PAGE_SIZE + c ≤ 2 ^ 22)
    {x pv x' al : F32} {pp : Tensor [d.PAGE_SIZE] F32} {vv : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32}
    {μ' : ℝ} (hμ'1 : -(Num.Kx X) ≤ μ') (hμ'2 : μ' ≤ Num.Kx X)
    (hlanes : ∀ i : Fin d.PAGE_SIZE, i.val < c → (pp.get1 i).isFinite = true ∧ |(pp.get1 i).toReal| ≤ 2 ∧
      |(pp.get1 i).toReal - Real.exp (Sx d m s h (t * d.PAGE_SIZE + i.val) - μ')| ≤
        Num.rhoP X * Real.exp (Sx d m s h (t * d.PAGE_SIZE + i.val) - μ') + Num.etaP)
    (hVb : ∀ k, k < t * d.PAGE_SIZE + c → |Vx d m s h e k| ≤ 2 ^ 15)
    (hmr : ∀ j, j < c → Num.mr vv j e = Vx d m s h e (t * d.PAGE_SIZE + j))
    (hpv : pv.isFinite = true ∧ |pv.toReal| ≤ 2 ^ 25 ∧
      |pv.toReal - ∑ j ∈ Finset.range c, Num.phr pp j * Num.mr vv j e| ≤
        Num.rhoSA * ∑ j ∈ Finset.range c, |Num.phr pp j * Num.mr vv j e| + Num.etaS)
    (hupd : (d.PAGE_SIZE = 1 ∧ ∃ am : F32, (FloatV.mul x al).NaNEq am ∧ AddRel E.fusion am pv x') ∨
      (d.PAGE_SIZE ≠ 1 ∧ ((FloatV.add (FloatV.mul x al) pv).NaNEq x' ∨
        (E.fusion = true ∧ (FloatV.fma x al pv).NaNEq x'))))
    (hold : 0 < t → ∃ μ : ℝ, x.isFinite = true ∧
      Num.InvA X (Sx d m s h) (Vx d m s h e) (t * d.PAGE_SIZE) t μ x.toReal ∧ μ ≤ μ' ∧
      al.isFinite = true ∧
      |al.toReal - Real.exp (μ - μ')| ≤ (Num.aA0 + Num.aA1 * (μ' - μ)) * Real.exp (μ - μ') + Num.zeta0 ∧
      |al.toReal| ≤ 2 ∧ (∀ k, k < t * d.PAGE_SIZE → Sx d m s h k ≤ μ + Num.dS X) ∧
      -(Num.Kx X) ≤ μ ∧ μ ≤ Num.Kx X)
    (h0 : t = 0 → x = FloatV.ofPat 0 ∧ al = FloatV.ofPat 0) :
    x'.isFinite = true ∧
      Num.InvA X (Sx d m s h) (Vx d m s h e) (t * d.PAGE_SIZE + c) (t + 1) μ' x'.toReal := by
  obtain ⟨hpvf, hpv25, hpverr⟩ := hpv
  have hPpos : 0 < d.PAGE_SIZE := by omega
  have hp' : ∀ j, j < c → |Num.phr pp j - Real.exp (Sx d m s h (t * d.PAGE_SIZE + j) - μ')| ≤
      Num.rhoP X * Real.exp (Sx d m s h (t * d.PAGE_SIZE + j) - μ') + Num.etaP := by
    intro j hj
    have hjP : j < d.PAGE_SIZE := by omega
    have := (hlanes ⟨j, hjP⟩ hj).2.2
    simpa [Num.phr, hjP] using this
  have hb' : |pv.toReal - ∑ j ∈ Finset.range c, Num.phr pp j * Vx d m s h e (t * d.PAGE_SIZE + j)| ≤
      Num.rhoSA * ∑ j ∈ Finset.range c, |Num.phr pp j * Vx d m s h e (t * d.PAGE_SIZE + j)| + Num.etaS := by
    have e1 : ∑ j ∈ Finset.range c, Num.phr pp j * Num.mr vv j e =
        ∑ j ∈ Finset.range c, Num.phr pp j * Vx d m s h e (t * d.PAGE_SIZE + j) :=
      Finset.sum_congr rfl fun j hj => by rw [hmr j (Finset.mem_range.mp hj)]
    have e2 : ∑ j ∈ Finset.range c, |Num.phr pp j * Num.mr vv j e| =
        ∑ j ∈ Finset.range c, |Num.phr pp j * Vx d m s h e (t * d.PAGE_SIZE + j)| :=
      Finset.sum_congr rfl fun j hj => by rw [hmr j (Finset.mem_range.mp hj)]
    rw [e1, e2] at hpverr; exact hpverr
  have hrates := rates_a hX0 hX1
  have hc256 : c ≤ 256 := by omega
  have hu := Num.u_pos
  -- the update, as one bound
  have hupd' : ∀ (xr : ℝ), |x.toReal| ≤ 2 ^ 39 → |al.toReal| ≤ 2 → x.isFinite = true → al.isFinite = true →
      x'.isFinite = true ∧ |x'.toReal - (al.toReal * x.toReal + pv.toReal)| ≤
        Num.rhoOA * (|al.toReal * x.toReal| + |pv.toReal|) + Num.etaO := by
    intro _ hxb hab hxf haf
    have h1 : |x.toReal * al.toReal| ≤ 2 ^ 39 * 2 := by
      rw [abs_mul]; exact mul_le_mul hxb hab (abs_nonneg _) (by norm_num)
    have h2 : |x.toReal * al.toReal| + |pv.toReal| ≤ 2 ^ 100 := by
      have : (2 : ℝ) ^ 39 * 2 + 2 ^ 25 ≤ 2 ^ 100 := by norm_num
      linarith
    rcases hupd with ⟨-, am, hm, hadd⟩ | ⟨-, hma⟩
    · -- PAGE_SIZE = 1
      obtain ⟨hf, he⟩ := Num.addc32 (E := E) hxf haf hpvf h2 hm hadd
      exact ⟨hf, he⟩
    · obtain ⟨hf, he⟩ := Num.mulAdd32 (E := E) hxf haf hpvf h2 hma
      refine ⟨hf, ?_⟩
      have hle : Num.rhoOL ≤ Num.rhoOA := by unfold Num.rhoOL Num.rhoOA; nlinarith
      have e1 : |x.toReal * al.toReal| = |al.toReal * x.toReal| := by rw [mul_comm]
      have e2 : x.toReal * al.toReal + pv.toReal = al.toReal * x.toReal + pv.toReal := by rw [mul_comm]
      rw [e2] at he
      have h3 : Num.rhoOL * (|x.toReal * al.toReal| + |pv.toReal|) ≤ Num.rhoOA * (|x.toReal * al.toReal| + |pv.toReal|) :=
        mul_le_mul_of_nonneg_right hle (by positivity)
      rw [e1] at h3 he
      linarith
  have hz : (FloatV.ofPat 0 : F32).toReal = 0 := by unfold FloatV.toReal; rw [Num.zero32_rat]; simp
  have hsAbs : ∀ j, j < c → |Vx d m s h e (t * d.PAGE_SIZE + j)| ≤ 2 ^ 15 :=
    fun j hj => hVb _ (by omega)
  rcases Nat.eq_zero_or_pos t with h00 | ht0
  · subst h00
    obtain ⟨hx0, hal0⟩ := h0 rfl
    obtain ⟨hx'f, hx'e⟩ := hupd' 0 (by rw [hx0, hz]; norm_num) (by rw [hal0, hz]; norm_num)
      (by rw [hx0]; exact Num.zero32_fin) (by rw [hal0]; exact Num.zero32_fin)
    refine ⟨hx'f, ?_⟩
    rw [hx0, hal0, hz] at hx'e
    simp only [zero_mul, zero_add, abs_zero] at hx'e
    have := Num.col_first hX0 hX1 (Sx d m s h) (Vx d m s h e) (c := c) (μ' := μ') (x' := x'.toReal)
      (b := pv.toReal) (ρo := Num.rhoOA) (ρs := Num.rhoSA) (e1 := Num.eA) (ε2 := Num.eps2A X)
      (Vm := 2 ^ 15) (Num.phr pp) hc256 (by unfold Num.rhoOA; positivity)
      (by unfold Num.rhoSA; positivity) (by unfold Num.eA; positivity) (by norm_num) (by norm_num)
      (by unfold Num.rhoOA Num.u; norm_num) (by unfold Num.rhoSA Num.u; norm_num)
      hrates.2.2.1 hμ'1 (fun k hk => by simpa using hVb k (by simpa using hk))
      (by simpa using hp') (by simpa using hb') (by simpa using hx'e)
    simpa [Num.InvA] using this
  · obtain ⟨μ, hxf, hxinv, hμμ', half, hal_err, hal2, hSold, hμ1, hμ2⟩ := hold ht0
    have hxb : |x.toReal| ≤ 2 ^ 39 :=
      colInv_bound hX0 hX1 (e1 := Num.eA) (ε2 := Num.eps2A X) (by unfold Num.eA; positivity) le_rfl
        (by unfold Num.eps2A; positivity) hrates.2.2.2 (S := Sx d m s h) (V := Vx d m s h e)
        (by omega) (by omega) (fun k hk => hVb k (by omega)) hSold hμ2 hμ1 hxinv
    obtain ⟨hx'f, hx'e⟩ := hupd' 0 hxb hal2 hxf half
    refine ⟨hx'f, ?_⟩
    have := Num.col_step hX0 hX1 (Sx d m s h) (Vx d m s h e) (t := t) (N := t * d.PAGE_SIZE) (c := c)
      (μ := μ) (μ' := μ') (α := Real.exp (μ - μ')) (α' := al.toReal) (x := x.toReal)
      (x' := x'.toReal) (b := pv.toReal) (ρo := Num.rhoOA) (ρs := Num.rhoSA) (e1 := Num.eA)
      (ε2 := Num.eps2A X) (Vm := 2 ^ 15) (Num.phr pp) ht0 (by omega) hc256 hN
      (by unfold Num.rhoOA; positivity) (by unfold Num.rhoSA; positivity)
      (by unfold Num.eA; positivity) (by norm_num) (by norm_num) hrates.1 hrates.2.1 hrates.2.2.1
      hrates.2.2.2 le_rfl hμ1 hμμ' hμ'2 hSold (fun k hk => hVb k (by omega)) rfl hal_err hp' hb'
      hxinv hx'e
    exact this


/-- **The values of one page**: the `float32` conversion of the loaded `float16` values. A valid
row holds the cache's values (finite, at most `2^15`); every row is finite (masked rows hold
the converted `other = 0`). -/
theorem v_stage {d : Inst} {m : Mem} {s h t pgN : Nat} (hSV : SeqVals d m s h)
    {vraw oth : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F16} {vv : Tensor [d.PAGE_SIZE, d.HEAD_DIM] F32}
    (hvraw : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
      decide (t * d.PAGE_SIZE + i.1.val < seqLen d m s) = true →
        m.read .f16 (d.v_cache.ptr (kvIdx d pgN i.1.val (h / qpk d) i.2.1.val : Int)).addr =
          some (vraw.get i))
    (hvoth : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
      decide (t * d.PAGE_SIZE + i.1.val < seqLen d m s) = false →
        ∀ o : Tensor [d.PAGE_SIZE, d.HEAD_DIM] DType.f16.Val, some oth = some o → vraw.get i = o.get i)
    (hoth : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM], (oth.get i).isFinite = true ∧ (oth.get i).toReal = 0)
    (hTok : ∀ r e, r < d.PAGE_SIZE → tokElem d m s h (t * d.PAGE_SIZE + r) e =
      kvIdx d pgN r (h / qpk d) e)
    (hvv : ∀ i : Idx [d.PAGE_SIZE, d.HEAD_DIM],
      (FloatV.cvtWith RoundMode.rn (vraw.get i)).NaNEq (vv.get i)) :
    ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), (vv.get2 r e).isFinite = true ∧
      (t * d.PAGE_SIZE + r.val < seqLen d m s →
        (vv.get2 r e).toReal = Vx d m s h e.val (t * d.PAGE_SIZE + r.val) ∧
          |(vv.get2 r e).toReal| ≤ 2 ^ 15) := by
  intro r e
  by_cases hr : t * d.PAGE_SIZE + r.val < seqLen d m s
  · obtain ⟨-, ⟨y, hy, hyf⟩, hvb⟩ := hSV.2.2.1 (t * d.PAGE_SIZE + r.val) e.val hr e.isLt
    have h2 := (hTok r.val e.val r.isLt)
    rw [h2] at hy
    have hrd : decide (t * d.PAGE_SIZE + (r, e, ()).1.val < seqLen d m s) = true := decide_eq_true hr
    have h1 := hvraw (r, e, ()) hrd
    have hye : vraw.get (r, e, ()) = y := by rw [hy] at h1; exact (Option.some.inj h1).symm
    obtain ⟨hf, hv⟩ := Num.cvt16_exact (x := vraw.get (r, e, ())) (by rw [hye]; exact hyf) RoundMode.rn
      (hvv (r, e, ()))
    have hval : val16 m d.v_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e.val) = y.toRat := by
      rw [h2]; exact val16_eq hy
    refine ⟨hf, fun _ => ⟨?_, ?_⟩⟩
    · show FloatV.toReal (vv.get (r, e, ())) = _
      rw [hv, hye]
      unfold Vx; rw [hval]; rfl
    · show |FloatV.toReal (vv.get (r, e, ()))| ≤ _
      rw [hv, hye]
      have : |(val16 m d.v_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e.val) : ℝ)| ≤ 2 ^ 15 := by
        have := hvb
        have h3 : ((|val16 m d.v_cache (tokElem d m s h (t * d.PAGE_SIZE + r.val) e.val)| : ℚ) : ℝ) ≤ ((2 ^ 15 : ℚ) : ℝ) := by
          exact_mod_cast this
        push_cast at h3; exact h3
      rw [hval] at this
      exact this
  · refine ⟨?_, fun h' => absurd h' hr⟩
    have hrd : decide (t * d.PAGE_SIZE + (r, e, ()).1.val < seqLen d m s) = false := decide_eq_false hr
    have h1 := hvoth (r, e, ()) hrd oth rfl
    obtain ⟨hf, -⟩ := Num.cvt16_exact (x := vraw.get (r, e, ())) (by rw [h1]; exact (hoth _).1) RoundMode.rn
      (hvv (r, e, ()))
    exact hf

/-- The update of `acc` (line 68), whichever branch of the `PAGE_SIZE = 1` split: the rest is
safe for every result the update may give. -/
theorem safe_acc_update {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    (d : Inst) (acc pv : Tensor [d.HEAD_DIM] F32) (al : F32)
    {k : Tensor [d.HEAD_DIM] F32 → Prog α}
    (hk : ∀ acc' : Tensor [d.HEAD_DIM] F32,
      (∀ i : Idx [d.HEAD_DIM],
        (d.PAGE_SIZE = 1 ∧ ∃ am : F32, (FloatV.mul (acc.get i) al).NaNEq am ∧
            AddRel E.fusion am (pv.get i) (acc'.get i)) ∨
        (d.PAGE_SIZE ≠ 1 ∧ ((FloatV.add (FloatV.mul (acc.get i) al) (pv.get i)).NaNEq (acc'.get i) ∨
            (E.fusion = true ∧ (FloatV.fma (acc.get i) al (pv.get i)).NaNEq (acc'.get i))))) →
        Safe E R W Q m (k acc')) :
    Safe E R W Q m (if d.PAGE_SIZE = 1 then
        (tl.mul acc (bc al) >>= fun x => tl.addc x pv >>= k)
      else (tl.mul_add acc (bc al) pv >>= k)) := by
  by_cases hP1 : d.PAGE_SIZE = 1
  · simp only [hP1, ite_true]
    refine safe_mul fun accm haccm => ?_
    refine safe_addc fun acc' hacc' => hk acc' fun i => Or.inl ⟨hP1, accm.get i, haccm i, hacc' i⟩
  · simp only [hP1, ite_false]
    refine safe_mul_add fun acc' hacc' => hk acc' fun i => Or.inr ⟨hP1, hacc' i⟩

end Claims.PagedAttention
