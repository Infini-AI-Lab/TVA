import Proof.Block

/-!
# One block, from the lanes' values to the claim's tolerance

`final_noscale` and `final_scale` take the relations the walk collects for one block (the lanes
after `tl.where`, the max, the subtraction, `tl.exp`, the sum, `tl.log` and the last addition) and
give `Post`'s conclusion: the result `c` is finite and within `2^-20·M + 2^-16` of
`log Σ_{l<n} e^(y_l)` for every bound `M` of the `|y_l|`. Without a scale the lanes are the inputs
(`y_l = v_l`) and every operation is IEEE; with one, `|v_l − y_l| ≤ 3u·|y_l| + 5u` and the
subtraction and the last addition are fusable (`AddRel`).
-/

namespace Claims.LogsumexpFwd

open TVA

/-- `log (e^m · A) = m + log A`. -/
lemma log_sum_shift {n : Nat} (hn : 0 < n) (m : ℝ) (t : Nat → ℝ) :
    Real.log (∑ l ∈ Finset.range n, Real.exp (t l + m)) =
      Real.log (∑ l ∈ Finset.range n, Real.exp (t l)) + m := by
  have hpos : 0 < ∑ l ∈ Finset.range n, Real.exp (t l) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) (Finset.nonempty_range_iff.mpr (by omega))
  have e1 : ∑ l ∈ Finset.range n, Real.exp (t l + m) =
      Real.exp m * ∑ l ∈ Finset.range n, Real.exp (t l) := by
    rw [Finset.mul_sum]; refine Finset.sum_congr rfl fun l _ => ?_
    rw [Real.exp_add]; ring
  rw [e1, Real.log_mul (Real.exp_pos m).ne' hpos.ne', Real.log_exp]; ring

/-- The lanes of one block, after `tl.where` (`v`), the max (`mx`), the subtraction (`t`), the
platform's `exp` (`e`), the sum (`S`) and the log (`L`): the shared hypotheses. -/
structure BlockRel (P : Platform) (B n : Nat) (v t e : Tensor [B] F32) (mx S L : F32) : Prop where
  hn : 0 < n
  hnB : n ≤ B
  hB : B ≤ 2 ^ 16
  v_fin : ∀ (l : Nat) (hl : l < B), l < n → (v.get1 ⟨l, hl⟩).isFinite = true
  v_neg : ∀ (l : Nat) (hl : l < B), ¬ l < n → v.get1 ⟨l, hl⟩ = FloatV.negInf
  max : ∃ tr : Tree, tr.Covers B ∧
    tr.Eval (FloatV.MaxRel true) (fun j => if h : j < B then v.get1 ⟨j, h⟩ else default) mx
  exp : ∀ (l : Nat) (hl : l < B), P.math .exp .binary32 (t.get1 ⟨l, hl⟩) (e.get1 ⟨l, hl⟩)
  sum : ∃ tr : Tree, tr.Covers B ∧ tr.depth ≤ tl.sumBound P B ∧
    tr.Eval (AddRel false) (fun j => if h : j < B then e.get1 ⟨j, h⟩ else default) S
  log : P.math .log .binary32 S L

set_option maxHeartbeats 1000000 in
/-- **A block without a scale.** -/
theorem final_noscale {P : Platform} (hP : Device P) {B n : Nat} {v t e : Tensor [B] F32}
    {mx S L c : F32} (h : BlockRel P B n v t e mx S L)
    (hv1 : ∀ (l : Nat) (hl : l < B), l < n → |(v.get1 ⟨l, hl⟩).toReal| ≤ 2 ^ 126)
    (ht : ∀ (l : Nat) (hl : l < B), FloatV.NaNEq (FloatV.sub (v.get1 ⟨l, hl⟩) mx) (t.get1 ⟨l, hl⟩))
    (hc : FloatV.NaNEq (FloatV.add L mx) c) :
    c.isFinite = true ∧ ∀ M : ℝ, (∀ (l : Nat) (hl : l < B), l < n → |(v.get1 ⟨l, hl⟩).toReal| ≤ M) →
      |c.toReal - Real.log (∑ l ∈ Finset.range n,
        Real.exp (if hl : l < B then (v.get1 ⟨l, hl⟩).toReal else 0))| ≤
        (zTolM : ℝ) * M + (zTol0 : ℝ) := by
  have hu := uR_pos
  have heta := etaR_pos
  have hetau := etaR_le
  obtain ⟨i0, hi0B, hi0n, hmx, hmxf, hle⟩ := max_facts h.hn h.hnB h.v_fin h.v_neg h.max
  have hmx1 : |mx.toReal| ≤ 2 ^ 126 := by rw [hmx]; exact hv1 i0 hi0B hi0n
  have hleR : ∀ (l : Nat) (hl : l < B), l < n → (v.get1 ⟨l, hl⟩).toReal ≤ mx.toReal := by
    intro l hl hln
    have := hle l hl hln
    simp only [FloatV.toReal_eq]; exact_mod_cast this
  -- the subtraction of each valid lane
  have hsub : ∀ (l : Nat) (hl : l < B), l < n → (t.get1 ⟨l, hl⟩).isFinite = true ∧
      |(t.get1 ⟨l, hl⟩).toReal - ((v.get1 ⟨l, hl⟩).toReal - mx.toReal)| ≤
        uR * |(v.get1 ⟨l, hl⟩).toReal - mx.toReal| + etaR := by
    intro l hl hln
    refine sub_valid_ieee (h.v_fin l hl hln) hmxf ?_ (ht l hl)
    have h1 := hv1 l hl hln
    have h2 := abs_sub (v.get1 ⟨l, hl⟩).toReal mx.toReal
    have : (2 : ℝ) ^ 126 + 2 ^ 126 = 2 ^ 127 := by norm_num
    linarith
  -- the block's sum and log
  set tf : Nat → F32 := fun j => if h : j < B then t.get1 ⟨j, h⟩ else default with htf
  set ef : Nat → F32 := fun j => if h : j < B then e.get1 ⟨j, h⟩ else default with hef
  have htf' : ∀ (l : Nat) (hl : l < B), tf l = t.get1 ⟨l, hl⟩ := fun l hl => by
    rw [htf]; exact dite_eq_left hl
  have hef' : ∀ (l : Nat) (hl : l < B), ef l = e.get1 ⟨l, hl⟩ := fun l hl => by
    rw [hef]; exact dite_eq_left hl
  have hblk := block_log hP.expF32 hP.expF32NegInf hP.logF32 hP.sumDepth h.hn h.hnB h.hB
    (κ := etaR) heta.le (by unfold etaR; norm_num) (s := tf) (e := ef) (S := S) (L := L)
    (fun l hln => by
      have hl := lt_of_lt_of_le hln h.hnB
      rw [htf' l hl]
      obtain ⟨hf, herr⟩ := hsub l hl hln
      refine ⟨hf, ?_⟩
      have h1 := (abs_le.mp herr).2
      have h2 := hleR l hl hln
      have h3 : |(v.get1 ⟨l, hl⟩).toReal - mx.toReal| = mx.toReal - (v.get1 ⟨l, hl⟩).toReal := by
        rw [abs_sub_comm]; exact abs_of_nonneg (by linarith)
      rw [h3] at h1
      have h4 : uR * (mx.toReal - (v.get1 ⟨l, hl⟩).toReal) ≤ mx.toReal - (v.get1 ⟨l, hl⟩).toReal := by
        have := uR_le
        nlinarith
      linarith)
    ⟨i0, hi0n, by
      rw [htf' i0 hi0B]
      obtain ⟨-, herr⟩ := hsub i0 hi0B hi0n
      rw [← hmx, sub_self, abs_zero, mul_zero, zero_add, sub_zero] at herr
      linarith [(abs_le.mp herr).1]⟩
    (fun l hln hlB => by
      rw [htf' l hlB]
      left
      have := ht l hlB
      rw [h.v_neg l hlB (not_lt.mpr hln)] at this
      exact sub_masked_ieee hmxf this)
    (fun l hl => by rw [htf' l hl, hef' l hl]; exact h.exp l hl)
    (by
      obtain ⟨tr, h1, h2, h3⟩ := h.sum
      exact ⟨tr, h1, h2, by rw [hef]; exact h3⟩)
    h.log
  obtain ⟨hLf, hLerr, hL12⟩ := hblk
  -- the last addition
  have hroomc : |L.toReal + mx.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    have h1 := abs_add_le L.toReal mx.toReal
    have h2 := b32_maxVal_ge
    have : (12 : ℝ) + 2 ^ 126 ≤ 2 ^ 127 := by norm_num
    linarith
  obtain ⟨hcf0, hcerr0⟩ := FloatV.add_err_real b32_reg hLf hmxf hroomc
  rw [b32_u_cast, b32_eta_cast] at hcerr0
  obtain ⟨hcf, -, hcr⟩ := FloatV.NaNEq.exact_of_isFinite hc hcf0
  refine ⟨hcf, fun M hM => ?_⟩
  rw [hcr]
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) (hM i0 hi0B hi0n)
  have hmxM : |mx.toReal| ≤ M := by rw [hmx]; exact hM i0 hi0B hi0n
  -- the log-sum-exp moves by at most `2u·M + η`
  have hlip : |Real.log (∑ l ∈ Finset.range n, Real.exp ((tf l).toReal + mx.toReal)) -
      Real.log (∑ l ∈ Finset.range n,
        Real.exp (if hl : l < B then (v.get1 ⟨l, hl⟩).toReal else 0))| ≤ 2 * uR * M + etaR := by
    refine lse_lip (Finset.nonempty_range_iff.mpr (by have := h.hn; omega)) fun l hl' => ?_
    have hln := Finset.mem_range.mp hl'
    have hl := lt_of_lt_of_le hln h.hnB
    rw [htf' l hl, dite_eq_left hl]
    obtain ⟨-, herr⟩ := hsub l hl hln
    have h1 : |(v.get1 ⟨l, hl⟩).toReal - mx.toReal| ≤ 2 * M := by
      have := abs_sub (v.get1 ⟨l, hl⟩).toReal mx.toReal
      have := hM l hl hln
      linarith
    have h2 : uR * |(v.get1 ⟨l, hl⟩).toReal - mx.toReal| ≤ uR * (2 * M) :=
      mul_le_mul_of_nonneg_left h1 hu.le
    rw [show (t.get1 ⟨l, hl⟩).toReal + mx.toReal - (v.get1 ⟨l, hl⟩).toReal =
      (t.get1 ⟨l, hl⟩).toReal - ((v.get1 ⟨l, hl⟩).toReal - mx.toReal) by ring]
    linarith
  rw [log_sum_shift h.hn] at hlip
  have hcadd : |L.toReal + mx.toReal| * uR ≤ (12 + M) * uR := by
    refine mul_le_mul_of_nonneg_right ?_ hu.le
    have := abs_add_le L.toReal mx.toReal
    linarith
  have htot : |(FloatV.add L mx).toReal - Real.log (∑ l ∈ Finset.range n,
      Real.exp (if hl : l < B then (v.get1 ⟨l, hl⟩).toReal else 0))| ≤
      |(FloatV.add L mx).toReal - (L.toReal + mx.toReal)| +
        |L.toReal - Real.log (∑ i ∈ Finset.range n, Real.exp (tf i).toReal)| +
        |Real.log (∑ l ∈ Finset.range n, Real.exp (tf l).toReal) + mx.toReal -
          Real.log (∑ l ∈ Finset.range n,
            Real.exp (if hl : l < B then (v.get1 ⟨l, hl⟩).toReal else 0))| := by
    have e : (FloatV.add L mx).toReal - Real.log (∑ l ∈ Finset.range n,
        Real.exp (if hl : l < B then (v.get1 ⟨l, hl⟩).toReal else 0)) =
        ((FloatV.add L mx).toReal - (L.toReal + mx.toReal)) +
        (L.toReal - Real.log (∑ i ∈ Finset.range n, Real.exp (tf i).toReal)) +
        (Real.log (∑ l ∈ Finset.range n, Real.exp (tf l).toReal) + mx.toReal -
          Real.log (∑ l ∈ Finset.range n,
            Real.exp (if hl : l < B then (v.get1 ⟨l, hl⟩).toReal else 0))) := by ring
    rw [e]
    exact le_trans (abs_add_le _ _) (add_le_add (abs_add_le _ _) le_rfl)
  have hz : (zTolM : ℝ) = 16 * uR := by unfold zTolM uR; norm_num
  have hz0 : (zTol0 : ℝ) = 256 * uR := by unfold zTol0 uR; norm_num
  rw [hz, hz0]
  have hfin : (12 + M) * uR + etaR + 100 * uR + (2 * uR * M + etaR) ≤ 16 * uR * M + 256 * uR := by
    nlinarith
  linarith

/-- `x + y` that a product may feed (`AddRel`), finite operands with room. -/
lemma add_fused {b : Bool} {x y r : F32} (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hroom : |x.toReal| + |y.toReal| ≤ 2 ^ 126) (hr : AddRel b x y r) :
    r.isFinite = true ∧ |r.toReal - (x.toReal + y.toReal)| ≤
      3 * uR * (|x.toReal| + |y.toReal|) + 4 * etaR := by
  have hroomQ : |x.toRat| + |y.toRat| ≤
      Format.binary32.maxVal / 2 - 2 * Format.binary32.eta := by
    have h1 : ((|x.toRat| + |y.toRat| : ℚ) : ℝ) ≤ 2 ^ 126 := by
      push_cast; rw [← FloatV.toReal_eq, ← FloatV.toReal_eq]; exact hroom
    have h2 : |x.toRat| + |y.toRat| ≤ 2 ^ 126 := by exact_mod_cast h1
    rw [Format.binary32_maxVal, Format.binary32_eta]
    norm_num at h2 ⊢
    linarith
  obtain ⟨hf, he⟩ := FloatV.AddRel.err b32_reg b32_nf (by rw [Format.binary32_u]; norm_num)
    hr hx hy hroomQ
  refine ⟨hf, ?_⟩
  have he' : ((|r.toRat - (x.toRat + y.toRat)| : ℚ) : ℝ) ≤
      ((3 * Format.binary32.u * (|x.toRat| + |y.toRat|) + 4 * Format.binary32.eta : ℚ) : ℝ) := by
    exact_mod_cast he
  push_cast at he'
  rw [b32_u_cast, b32_eta_cast] at he'
  simp only [FloatV.toReal_eq]
  exact he'

set_option maxHeartbeats 1000000 in
/-- **A block with a scale.** The lanes `v_l` approximate `y_l` (`|y_l| ≤ 2^20`); the subtraction
and the last addition are fusable. -/
theorem final_scale {P : Platform} (hP : Device P) {b : Bool} {B n : Nat} {v t e : Tensor [B] F32}
    {mx S L c : F32} (h : BlockRel P B n v t e mx S L) {y : Nat → ℝ}
    (hy : ∀ l, l < n → |y l| ≤ 2 ^ 20)
    (hvy : ∀ (l : Nat) (hl : l < B), l < n →
      |(v.get1 ⟨l, hl⟩).toReal - y l| ≤ 3 * uR * |y l| + 5 * uR)
    (ht : ∀ (l : Nat) (hl : l < B), AddRel b (v.get1 ⟨l, hl⟩) (FloatV.neg mx) (t.get1 ⟨l, hl⟩))
    (hc : AddRel b L mx c) :
    c.isFinite = true ∧ ∀ M : ℝ, (∀ l, l < n → |y l| ≤ M) →
      |c.toReal - Real.log (∑ l ∈ Finset.range n, Real.exp (y l))| ≤
        (zTolM : ℝ) * M + (zTol0 : ℝ) := by
  have hu := uR_pos
  have heta := etaR_pos
  have hetau := etaR_le
  have hu1 := uR_le
  obtain ⟨i0, hi0B, hi0n, hmx, hmxf, hle⟩ := max_facts h.hn h.hnB h.v_fin h.v_neg h.max
  have hleR : ∀ (l : Nat) (hl : l < B), l < n → (v.get1 ⟨l, hl⟩).toReal ≤ mx.toReal := by
    intro l hl hln
    have := hle l hl hln
    simp only [FloatV.toReal_eq]; exact_mod_cast this
  -- the lanes' magnitudes
  set V : ℝ := 2 ^ 20 * (1 + 3 * uR) + 5 * uR with hVdef
  have hvV : ∀ (l : Nat) (hl : l < B), l < n → |(v.get1 ⟨l, hl⟩).toReal| ≤ V := by
    intro l hl hln
    have h1 := hvy l hl hln
    have h2 := hy l hln
    have h3 := abs_add_le (y l) ((v.get1 ⟨l, hl⟩).toReal - y l)
    rw [add_sub_cancel] at h3
    have h4 : 3 * uR * |y l| ≤ 3 * uR * 2 ^ 20 := mul_le_mul_of_nonneg_left h2 (by positivity)
    rw [hVdef]; nlinarith
  have hmxV : |mx.toReal| ≤ V := by rw [hmx]; exact hvV i0 hi0B hi0n
  have hV : V ≤ 2 ^ 20 + 1 := by rw [hVdef]; unfold uR; norm_num
  -- the subtraction of each valid lane
  have hsub : ∀ (l : Nat) (hl : l < B), l < n → (t.get1 ⟨l, hl⟩).isFinite = true ∧
      |(t.get1 ⟨l, hl⟩).toReal - ((v.get1 ⟨l, hl⟩).toReal - mx.toReal)| ≤
        3 * uR * (|(v.get1 ⟨l, hl⟩).toReal| + |mx.toReal|) + 4 * etaR := by
    intro l hl hln
    refine sub_valid_fused (h.v_fin l hl hln) hmxf ?_ (ht l hl)
    have h1 := hvV l hl hln
    have : ((2 : ℝ) ^ 20 + 1) + (2 ^ 20 + 1) ≤ 2 ^ 126 := by norm_num
    linarith
  set κ : ℝ := 3 * uR * (V + V) + 4 * etaR with hκdef
  have hκ0 : 0 ≤ κ := by rw [hκdef]; positivity
  have hκ : κ ≤ 1 / 2 := by
    rw [hκdef]
    have h1 : 3 * uR * (V + V) ≤ 3 * uR * ((2 ^ 20 + 1) + (2 ^ 20 + 1)) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h2 : 3 * uR * (((2 : ℝ) ^ 20 + 1) + (2 ^ 20 + 1)) + 4 * etaR ≤ 1 / 2 := by
      unfold uR etaR; norm_num
    linarith
  have hδκ : ∀ (l : Nat) (hl : l < B), l < n →
      3 * uR * (|(v.get1 ⟨l, hl⟩).toReal| + |mx.toReal|) + 4 * etaR ≤ κ := by
    intro l hl hln
    rw [hκdef]
    have h1 := hvV l hl hln
    have h2 : 3 * uR * (|(v.get1 ⟨l, hl⟩).toReal| + |mx.toReal|) ≤ 3 * uR * (V + V) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    linarith
  -- the block's sum and log
  set tf : Nat → F32 := fun j => if h : j < B then t.get1 ⟨j, h⟩ else default with htf
  set ef : Nat → F32 := fun j => if h : j < B then e.get1 ⟨j, h⟩ else default with hef
  have htf' : ∀ (l : Nat) (hl : l < B), tf l = t.get1 ⟨l, hl⟩ := fun l hl => by
    rw [htf]; exact dite_eq_left hl
  have hef' : ∀ (l : Nat) (hl : l < B), ef l = e.get1 ⟨l, hl⟩ := fun l hl => by
    rw [hef]; exact dite_eq_left hl
  have hblk := block_log hP.expF32 hP.expF32NegInf hP.logF32 hP.sumDepth h.hn h.hnB h.hB
    hκ0 hκ (s := tf) (e := ef) (S := S) (L := L)
    (fun l hln => by
      have hl := lt_of_lt_of_le hln h.hnB
      rw [htf' l hl]
      obtain ⟨hf, herr⟩ := hsub l hl hln
      refine ⟨hf, ?_⟩
      have h1 := (abs_le.mp herr).2
      have h2 := hleR l hl hln
      have h3 := hδκ l hl hln
      linarith)
    ⟨i0, hi0n, by
      rw [htf' i0 hi0B]
      obtain ⟨-, herr⟩ := hsub i0 hi0B hi0n
      rw [← hmx, sub_self] at herr
      have h1 := (abs_le.mp herr).1
      have h3 := hδκ i0 hi0B hi0n
      rw [← hmx] at h3
      linarith⟩
    (fun l hln hlB => by
      rw [htf' l hlB]
      have := ht l hlB
      rw [h.v_neg l hlB (not_lt.mpr hln)] at this
      exact sub_masked_fused hmxf (by linarith [show (2 : ℝ) ^ 20 + 1 ≤ 2 ^ 126 by norm_num]) this)
    (fun l hl => by rw [htf' l hl, hef' l hl]; exact h.exp l hl)
    (by
      obtain ⟨tr, h1, h2, h3⟩ := h.sum
      exact ⟨tr, h1, h2, by rw [hef]; exact h3⟩)
    h.log
  obtain ⟨hLf, hLerr, hL12⟩ := hblk
  -- the last addition
  obtain ⟨hcf, hcerr⟩ := add_fused hLf hmxf
    (by linarith [show (12 : ℝ) + (2 ^ 20 + 1) ≤ 2 ^ 126 by norm_num]) hc
  refine ⟨hcf, fun M hM => ?_⟩
  have hM0 : 0 ≤ M := le_trans (abs_nonneg _) (hM 0 h.hn)
  have hvM : ∀ (l : Nat) (hl : l < B), l < n → |(v.get1 ⟨l, hl⟩).toReal| ≤ (1 + 3 * uR) * M + 5 * uR := by
    intro l hl hln
    have h1 := hvy l hl hln
    have h2 := hM l hln
    have h3 := abs_add_le (y l) ((v.get1 ⟨l, hl⟩).toReal - y l)
    rw [add_sub_cancel] at h3
    have h4 : 3 * uR * |y l| ≤ 3 * uR * M := mul_le_mul_of_nonneg_left h2 (by positivity)
    nlinarith
  have hmxM : |mx.toReal| ≤ (1 + 3 * uR) * M + 5 * uR := by rw [hmx]; exact hvM i0 hi0B hi0n
  -- the log-sum-exp moves by at most the lanes' errors
  have hlip : |Real.log (∑ l ∈ Finset.range n, Real.exp ((tf l).toReal + mx.toReal)) -
      Real.log (∑ l ∈ Finset.range n, Real.exp (y l))| ≤
        3 * uR * (2 * ((1 + 3 * uR) * M + 5 * uR)) + 4 * etaR + (3 * uR * M + 5 * uR) := by
    refine lse_lip (Finset.nonempty_range_iff.mpr (by have := h.hn; omega)) fun l hl' => ?_
    have hln := Finset.mem_range.mp hl'
    have hl := lt_of_lt_of_le hln h.hnB
    rw [htf' l hl]
    obtain ⟨-, herr⟩ := hsub l hl hln
    have h1 := hvM l hl hln
    have h2 : 3 * uR * (|(v.get1 ⟨l, hl⟩).toReal| + |mx.toReal|) ≤
        3 * uR * (2 * ((1 + 3 * uR) * M + 5 * uR)) :=
      mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    have h3 := hvy l hl hln
    have h4 : 3 * uR * |y l| ≤ 3 * uR * M := mul_le_mul_of_nonneg_left (hM l hln) (by positivity)
    have e : (t.get1 ⟨l, hl⟩).toReal + mx.toReal - y l =
        ((t.get1 ⟨l, hl⟩).toReal - ((v.get1 ⟨l, hl⟩).toReal - mx.toReal)) +
          ((v.get1 ⟨l, hl⟩).toReal - y l) := by ring
    rw [e]
    have := abs_add_le ((t.get1 ⟨l, hl⟩).toReal - ((v.get1 ⟨l, hl⟩).toReal - mx.toReal))
      ((v.get1 ⟨l, hl⟩).toReal - y l)
    linarith
  rw [log_sum_shift h.hn] at hlip
  have htot : |c.toReal - Real.log (∑ l ∈ Finset.range n, Real.exp (y l))| ≤
      |c.toReal - (L.toReal + mx.toReal)| +
        |L.toReal - Real.log (∑ i ∈ Finset.range n, Real.exp (tf i).toReal)| +
        |Real.log (∑ l ∈ Finset.range n, Real.exp (tf l).toReal) + mx.toReal -
          Real.log (∑ l ∈ Finset.range n, Real.exp (y l))| := by
    have e : c.toReal - Real.log (∑ l ∈ Finset.range n, Real.exp (y l)) =
        (c.toReal - (L.toReal + mx.toReal)) +
        (L.toReal - Real.log (∑ i ∈ Finset.range n, Real.exp (tf i).toReal)) +
        (Real.log (∑ l ∈ Finset.range n, Real.exp (tf l).toReal) + mx.toReal -
          Real.log (∑ l ∈ Finset.range n, Real.exp (y l))) := by ring
    rw [e]
    exact le_trans (abs_add_le _ _) (add_le_add (abs_add_le _ _) le_rfl)
  have hz : (zTolM : ℝ) = 16 * uR := by unfold zTolM uR; norm_num
  have hz0 : (zTol0 : ℝ) = 256 * uR := by unfold zTol0 uR; norm_num
  rw [hz, hz0]
  have hc3 : 3 * uR * (|L.toReal| + |mx.toReal|) ≤ 3 * uR * (12 + ((1 + 3 * uR) * M + 5 * uR)) :=
    mul_le_mul_of_nonneg_left (by linarith) (by positivity)
  have huu : uR * uR ≤ uR / 2 ^ 20 := by nlinarith
  have huM : uR * uR * M ≤ uR / 2 ^ 20 * M := mul_le_mul_of_nonneg_right huu hM0
  have hfin : 3 * uR * (12 + ((1 + 3 * uR) * M + 5 * uR)) + 4 * etaR + 100 * uR +
      (3 * uR * (2 * ((1 + 3 * uR) * M + 5 * uR)) + 4 * etaR + (3 * uR * M + 5 * uR)) ≤
      16 * uR * M + 256 * uR := by
    nlinarith
  linarith

end Claims.LogsumexpFwd
