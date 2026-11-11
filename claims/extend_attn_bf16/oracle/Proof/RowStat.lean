import Proof.RowTrip

/-!
# One trip of one row: `TripStat` from the float facts
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

set_option maxHeartbeats 1600000 in
/-- **One trip of one row** (see the module docstring): the float facts of the trip's reductions,
exponentials, sum, `deno` update and rounding of `p`, for the row's valid lanes (positions
`base + n`), give the core's `TripStat` and the rounding facts of `p`. -/
theorem row_stat {P : Platform} (hP : Device P) {fusion : Bool} {sc : ℕ → ℝ} {dl Xp : ℝ} {t : ℕ}
    {S : Finset ℕ} {base : ℕ} {valid : Fin 64 → Bool} {m l : ℝ} {G : Ghost}
    {emax deno l' rmax nmx re Ssum : F32} {qk p : Fin 64 → F32} {p16 : Fin 64 → BF16}
    (_hXp : 0 ≤ Xp) (hXp2 : Xp ≤ 2 ^ 13) (_hdl : 0 ≤ dl) (ht : t + 1 ≤ tmax)
    (hcard : (S ∪ laneSet base valid).card ≤ nmax)
    (_hdisj : Disjoint S (laneSet base valid)) (hrow : RowInv sc dl Xp t S m l G)
    (hdF : deno.isFinite = true) (hdV : deno.toReal = l)
    (hemaxF : S.Nonempty → emax.isFinite = true ∧ emax.toReal = m)
    (hemaxE : ¬ S.Nonempty → emax = FloatV.negInf ∨ emax = neg1e20)
    (hq : ∀ n : Fin 64, (valid n = true → (qk n).isFinite = true ∧
      |(qk n).toReal - sc (base + (n : ℕ))| ≤ dl ∧ |(qk n).toReal| ≤ Xp) ∧
      (valid n = false → qk n = FloatV.negInf))
    (hmax : ∃ t' : Tree, t'.Covers 64 ∧ t'.Eval (FloatV.MaxRel true)
      (fun j => if hj : j < 64 then qk ⟨j, hj⟩ else default) rmax)
    (hnm : FloatV.MaxRel true (if FloatV.eq rmax ninf32 = true then neg1e20 else rmax) emax nmx)
    (hre : ∃ d1, FloatV.NaNEq (FloatV.sub emax nmx) d1 ∧ P.math .exp .binary32 d1 re)
    (hp : ∀ n : Fin 64, ∃ d2, FloatV.NaNEq (FloatV.sub (qk n) nmx) d2 ∧
      P.math .exp .binary32 d2 (p n))
    (hsum : ∃ t' : Tree, t'.Covers 64 ∧ t'.depth ≤ tl.sumBound P 64 ∧ t'.Eval (AddRel false)
      (fun j => if hj : j < 64 then p ⟨j, hj⟩ else default) Ssum)
    (hden : FloatV.NaNEq (FloatV.add (FloatV.mul deno re) Ssum) l' ∨
      (fusion = true ∧ FloatV.NaNEq (FloatV.fma deno re Ssum) l'))
    (hcvt : ∀ n : Fin 64, FloatV.NaNEq (FloatV.cvtWith .rn (p n)) (p16 n)) :
    ∃ (m' g : ℝ) (f δ' : ℕ → ℝ),
      TripStat sc dl Xp S (laneSet base valid) m l (ext64 base fun n => (qk n).toReal) δ' f
        (ext64 base fun n => (p n).toReal) m' g re.toReal Ssum.toReal l'.toReal ∧
      (∀ n : Fin 64, valid n = true →
        |(p16 n).toReal - (p n).toReal * (1 + δ' (base + (n : ℕ)))| ≤ etab) ∧
      l'.isFinite = true ∧ re.isFinite = true ∧ (∀ n : Fin 64, (p n).isFinite = true) ∧
      (∀ n : Fin 64, (p16 n).isFinite = true) ∧
      (∀ n : Fin 64, valid n = false → (p16 n).toReal = 0) ∧ nmx.isFinite = true ∧
      ((S ∪ laneSet base valid).Nonempty → nmx.toReal = m') ∧
      (¬ (S ∪ laneSet base valid).Nonempty → nmx = neg1e20) := by
  classical
  obtain ⟨tm, hcm, hevm⟩ := hmax
  have hs : Format.binary32.specials = .ieee := rfl
  have hnN : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN hs
  obtain ⟨hN0f, hN0le, hN0ge⟩ := neg1e20_spec
  have hN0n : neg1e20.isNaN = false := (FloatV.isFinite_iff.mp hN0f).1
  have hq0 : ∀ n : Fin 64, (valid n = true → (qk n).isFinite = true) ∧
      (valid n = false → qk n = FloatV.negInf) := fun n =>
    ⟨fun h => ((hq n).1 h).1, (hq n).2⟩
  have hq1 : ∀ n : Fin 64, valid n = true → (qk n).isFinite = true ∧ |(qk n).toReal| ≤ Xp :=
    fun n h => ⟨((hq n).1 h).1, ((hq n).1 h).2.2⟩
  obtain ⟨hrm1, hrm2⟩ := row_max_spec hq0 hcm hevm
  have hSe : S.Nonempty → emax.isFinite = true ∧ emax.toReal = m ∧ |m| ≤ Xp := fun hS =>
    ⟨(hemaxF hS).1, (hemaxF hS).2, hrow.m_bound hS⟩
  obtain ⟨hnF, hN1, hN2, hN3⟩ := nmax_spec hXp2 hq1 hrm1 hrm2 hSe hemaxE hnm
  have hVne : (laneSet base valid).Nonempty ↔ ∃ n : Fin 64, valid n = true := by
    constructor
    · rintro ⟨j, hj⟩; obtain ⟨n, hn, -⟩ := mem_laneSet.mp hj; exact ⟨n, hn⟩
    · rintro ⟨n, hn⟩; exact ⟨base + (n : ℕ), mem_laneSet.mpr ⟨n, hn, rfl⟩⟩
  -- `m' ≤ Xp`
  have hm'le : (S ∪ laneSet base valid).Nonempty → nmx.toReal ≤ Xp := by
    intro hne
    by_cases hV : ∃ n : Fin 64, valid n = true
    · rcases (hN1 hV).2.2 with ⟨hS, h⟩ | ⟨n, hn, h⟩
      · rw [h]; exact (le_abs_self _).trans (hrow.m_bound hS)
      · rw [← h]; exact (le_abs_self _).trans (hq1 n hn).2
    · have hnv : ∀ n : Fin 64, valid n = false := fun n => by simpa using fun h => hV ⟨n, h⟩
      have hS : S.Nonempty := by
        rcases hne with ⟨j, hj⟩
        rcases Finset.mem_union.mp hj with h | h
        · exact ⟨j, h⟩
        · obtain ⟨n, hn, -⟩ := mem_laneSet.mp h; exact absurd hn (by simp [hnv n])
      rw [hN2 hnv hS]
      rw [(hemaxF hS).2]; exact (le_abs_self _).trans (hrow.m_bound hS)
  -- `emax ≤ nmx`
  have hrfixN : (if FloatV.eq rmax ninf32 = true then neg1e20 else rmax).isNaN = false := by
    split_ifs
    · exact hN0n
    · by_cases hV : ∃ n : Fin 64, valid n = true
      · exact (FloatV.isFinite_iff.mp (hrm1 hV).1).1
      · have hnv : ∀ n : Fin 64, valid n = false := fun n => by simpa using fun h => hV ⟨n, h⟩
        rw [hrm2 hnv]; exact hnN
  have hemN : emax.isNaN = false := by
    by_cases hS : S.Nonempty
    · exact (FloatV.isFinite_iff.mp (hemaxF hS).1).1
    · rcases hemaxE hS with h | h <;> rw [h] <;> assumption
  have hmle : emax.toReal ≤ nmx.toReal := by
    rw [hnm.toReal_eq_max hrfixN hemN]; exact le_max_right _ _
  -- `nmx` is at most `2^13`
  have hnmxle : nmx.toReal ≤ 2 ^ 13 := by
    by_cases hne : (S ∪ laneSet base valid).Nonempty
    · exact (hm'le hne).trans hXp2
    · have hS : ¬ S.Nonempty := fun h => hne ⟨h.choose, Finset.mem_union_left _ h.choose_spec⟩
      have hnv : ∀ n : Fin 64, valid n = false := fun n => by
        by_contra hv
        exact hne ⟨base + (n : ℕ), Finset.mem_union_right _ (mem_laneSet.mpr ⟨n, by simpa using hv, rfl⟩)⟩
      rw [hN3 hnv hS]
      have : (2 : ℝ) ^ 60 > 2 ^ 13 := by norm_num
      linarith
  -- `re_scale`
  obtain ⟨d1, hd1, hrex⟩ := hre
  have hreF : re.isFinite = true ∧ |re.toReal| ≤ 1 + 1 / 2 ^ 20 := by
    by_cases hS : S.Nonempty
    · obtain ⟨hef, hev⟩ := hemaxF hS
      refine exp_sub_le hP hef hnF hmle ?_ hd1 hrex
      have hm := (abs_le.mp (hrow.m_bound hS)).1
      rw [hev]
      have : (2 : ℝ) ^ 100 ≥ 2 ^ 14 := by norm_num
      linarith
    · rcases hemaxE hS with h | h
      · rw [h] at hd1
        obtain ⟨hf, h0⟩ := exp_negInf hP hnF hd1 hrex
        exact ⟨hf, by rw [h0]; norm_num⟩
      · refine exp_sub_le hP (h ▸ hN0f) hnF (h ▸ hmle) ?_ hd1 hrex
        rw [h]
        have : (2 : ℝ) ^ 100 ≥ 2 ^ 13 + 2 ^ 70 := by norm_num
        linarith
  -- the scale factor
  have hscale : ∃ g : ℝ, S.Nonempty → |g - 1| ≤ expm1 (e1 + e2 * (nmx.toReal - m)) ∧
      |re.toReal - Real.exp (m - nmx.toReal) * g| ≤ eps0 := by
    by_cases hS : S.Nonempty
    · obtain ⟨hef, hev⟩ := hemaxF hS
      have hm := (abs_le.mp (hrow.m_bound hS)).1
      have hle : emax.toReal ≤ nmx.toReal := hmle
      have hgap : nmx.toReal - emax.toReal ≤ 2 ^ 15 := by
        rw [hev]
        have : (2 : ℝ) ^ 15 ≥ 2 ^ 13 + 2 ^ 13 := by norm_num
        linarith
      obtain ⟨-, g, hg, hre'⟩ := exp_ok hP hef hnF hle hgap hd1 hrex
      rw [hev] at hg hre'
      exact ⟨g, fun _ => ⟨hg, hre'⟩⟩
    · exact ⟨1, fun h => absurd h hS⟩
  obtain ⟨g, hg⟩ := hscale
  -- the lanes of `p`
  have hpl : ∀ n : Fin 64, ∃ fn : ℝ, (p n).isFinite = true ∧ |(p n).toReal| ≤ 1 + 1 / 2 ^ 20 ∧
      (valid n = true → |fn - 1| ≤ expm1 (e1 + e2 * (nmx.toReal - (qk n).toReal)) ∧
        |(p n).toReal - Real.exp ((qk n).toReal - nmx.toReal) * fn| ≤ eps0) ∧
      (valid n = false → p n = FloatV.ofPat 0) := by
    intro n
    obtain ⟨d2, hd2, hpe⟩ := hp n
    by_cases hv : valid n = true
    · obtain ⟨hqf, -, hqX⟩ := (hq n).1 hv
      have hle : (qk n).toReal ≤ nmx.toReal := (hN1 ((hVne.mpr ⟨n, hv⟩) |> fun _ => ⟨n, hv⟩)).1 n hv
      have hne : (S ∪ laneSet base valid).Nonempty :=
        ⟨base + (n : ℕ), Finset.mem_union_right _ (mem_laneSet.mpr ⟨n, hv, rfl⟩)⟩
      have hgap : nmx.toReal - (qk n).toReal ≤ 2 ^ 15 := by
        have := hm'le hne
        have := (abs_le.mp hqX).1
        have : (2 : ℝ) ^ 15 ≥ 2 ^ 13 + 2 ^ 13 := by norm_num
        linarith
      obtain ⟨hpf, f, hf, hpr⟩ := exp_ok hP hqf hnF hle hgap hd2 hpe
      refine ⟨f, hpf, (exp_sub_le hP hqf hnF hle (by linarith) hd2 hpe).2, fun _ => ⟨hf, hpr⟩,
        fun h => absurd hv (by simp [h])⟩
    · have hvf : valid n = false := by simpa using hv
      rw [(hq n).2 hvf] at hd2
      have h0 := exp_negInf_eq hP hnF hd2 hpe
      obtain ⟨hpf, h0r⟩ := exp_negInf hP hnF hd2 hpe
      exact ⟨0, hpf, by rw [h0r]; norm_num, fun h => absurd h hv, fun _ => h0⟩
  choose fl hplF hplB hplV hplZ using hpl
  -- the sum
  obtain ⟨ts, hcs, hds, hevs⟩ := hsum
  obtain ⟨hSF, hSe'⟩ := sum_row hP hplF (fun n => (hplB n).trans (by norm_num)) hcs hds hevs
  have hsum_eq : ∀ h : ℕ → ℝ, (∀ n : Fin 64, valid n = false → h (base + (n : ℕ)) = 0) →
      ∑ j ∈ laneSet base valid, h j = ∑ n : Fin 64, h (base + (n : ℕ)) := by
    intro h hz
    rw [sum_laneSet]
    refine Finset.sum_congr rfl fun n _ => ?_
    by_cases hv : valid n = true
    · simp [hv]
    · have hvf : valid n = false := by simpa using hv
      simp [hvf, hz n hvf]
  have hpbar : ∀ n : Fin 64, ext64 base (fun n => (p n).toReal) (base + (n : ℕ)) = (p n).toReal :=
    fun n => ext64_lane base _ n
  have hS1 : ∑ j ∈ laneSet base valid, ext64 base (fun n => (p n).toReal) j = ∑ n : Fin 64, (p n).toReal := by
    rw [hsum_eq (ext64 base fun n => (p n).toReal) (fun n hv => by rw [hpbar]; rw [hplZ n hv]; simp)]
    exact Finset.sum_congr rfl fun n _ => hpbar n
  have hS2 : ∑ j ∈ laneSet base valid, |ext64 base (fun n => (p n).toReal) j| = ∑ n : Fin 64, |(p n).toReal| := by
    rw [hsum_eq (fun j => |ext64 base (fun n => (p n).toReal) j|) (fun n hv => by
      show |ext64 base (fun n => (p n).toReal) (base + (n : ℕ))| = 0
      rw [hpbar, hplZ n hv]; simp)]
    exact Finset.sum_congr rfl fun n _ => by rw [hpbar]
  -- the `deno`
  have ht1 : t ≤ tmax := by omega
  have hSc : S.card ≤ nmax := (Finset.card_le_card Finset.subset_union_left).trans hcard
  have hlb : |deno.toReal| ≤ 2 ^ 20 := by rw [hdV]; exact hrow.l_bound hXp2 ht1 hSc
  have hSb : |Ssum.toReal| ≤ 2 ^ 20 := by
    have h1 : |∑ n : Fin 64, (p n).toReal| ≤ 128 := by
      refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
      calc ∑ n : Fin 64, |(p n).toReal| ≤ ∑ _n : Fin 64, (2 : ℝ) :=
            Finset.sum_le_sum fun n _ => (hplB n).trans (by norm_num)
        _ = 128 := by simp; norm_num
    have h2 : expm1 lam0 * ∑ n : Fin 64, |(p n).toReal| ≤ 1 * 128 := by
      have hL0 : expm1 lam0 ≤ 1 := by
        have := exp_le_quad (x := lam0) (by unfold lam0; norm_num) (by unfold lam0; norm_num)
        unfold expm1; unfold lam0 at *; norm_num at *; linarith
      have hA : ∑ n : Fin 64, |(p n).toReal| ≤ 128 := by
        calc ∑ n : Fin 64, |(p n).toReal| ≤ ∑ _n : Fin 64, (2 : ℝ) :=
              Finset.sum_le_sum fun n _ => (hplB n).trans (by norm_num)
          _ = 128 := by simp; norm_num
      exact mul_le_mul hL0 hA (Finset.sum_nonneg fun n _ => abs_nonneg _) (by norm_num)
    have h3 : |Ssum.toReal| ≤ |∑ n : Fin 64, (p n).toReal| + |Ssum.toReal - ∑ n : Fin 64, (p n).toReal| := by
      have := abs_add_le (∑ n : Fin 64, (p n).toReal) (Ssum.toReal - ∑ n : Fin 64, (p n).toReal)
      rw [add_sub_cancel] at this; linarith
    have : eps0 ≤ 1 := by unfold eps0; norm_num
    norm_num at h2 ⊢
    linarith
  obtain ⟨hl'F, hl'e⟩ := deno_madd hdF hreF.1 hSF hlb (hreF.2.trans (by norm_num)) hSb hden
  -- the rounding of `p`
  have hcv : ∀ n : Fin 64, (p16 n).isFinite = true ∧ ∃ δ : ℝ, |δ| ≤ ub ∧
      |(p16 n).toReal - (p n).toReal * (1 + δ)| ≤ etab := fun n =>
    cvt_p16 (hplF n) ((hplB n).trans (by norm_num)) (hcvt n)
  choose δl hδl using fun n => (hcv n).2
  -- assemble
  have hV : ∀ j ∈ laneSet base valid, ∃ n : Fin 64, valid n = true ∧ j = base + (n : ℕ) := fun j hj =>
    mem_laneSet.mp hj
  refine ⟨nmx.toReal, g, ext64 base fl, ext64 base δl, ⟨?_, ?_, ?_, hg, ?_, ?_, ?_, ?_⟩, ?_,
    hl'F, hreF.1, hplF, fun n => (hcv n).1, ?_, hnF, fun _ => rfl, ?_⟩
  · -- lanes
    intro j hj
    obtain ⟨n, hn, rfl⟩ := hV j hj
    rw [ext64_lane, ext64_lane]
    obtain ⟨-, h1, h2⟩ := (hq n).1 hn
    exact ⟨h1, h2, (hδl n).1⟩
  · -- ref_ge
    refine ⟨fun j hj => ?_, fun hS => ?_⟩
    · obtain ⟨n, hn, rfl⟩ := hV j hj
      rw [ext64_lane]
      exact (hN1 ⟨n, hn⟩).1 n hn
    · by_cases hVn : ∃ n : Fin 64, valid n = true
      · rw [← (hemaxF hS).2]; exact (hN1 hVn).2.1 hS |> fun h => by rw [(hemaxF hS).2]; exact h
      · have hnv : ∀ n : Fin 64, valid n = false := fun n => by simpa using fun h => hVn ⟨n, h⟩
        rw [hN2 hnv hS, (hemaxF hS).2]
  · -- ref_att
    intro hne
    by_cases hVn : ∃ n : Fin 64, valid n = true
    · rcases (hN1 hVn).2.2 with ⟨hS, h⟩ | ⟨n, hn, h⟩
      · left; exact ⟨hS, h⟩
      · right
        refine ⟨base + (n : ℕ), mem_laneSet.mpr ⟨n, hn, rfl⟩, ?_⟩
        rw [ext64_lane]; exact h
    · have hnv : ∀ n : Fin 64, valid n = false := fun n => by simpa using fun h => hVn ⟨n, h⟩
      have hS : S.Nonempty := by
        rcases hne with ⟨j, hj⟩
        rcases Finset.mem_union.mp hj with h | h
        · exact ⟨j, h⟩
        · obtain ⟨n, hn, -⟩ := mem_laneSet.mp h; exact absurd hn (by simp [hnv n])
      left
      refine ⟨hS, ?_⟩
      rw [hN2 hnv hS, (hemaxF hS).2]
  · exact hreF.2
  · -- weight
    intro j hj
    obtain ⟨n, hn, rfl⟩ := hV j hj
    rw [ext64_lane, ext64_lane, ext64_lane]
    exact hplV n hn
  · -- sum
    rw [hS1, hS2]; exact hSe'
  · -- deno
    rw [← hdV]; exact hl'e
  · -- the rounding of `p`
    intro n hn
    rw [ext64_lane]
    exact (hδl n).2
  · -- masked lanes round to `0`
    intro n hn
    have h1 := hcvt n
    rw [hplZ n hn, FloatV.cvtWith_ofPat_zero Format.binary32_regular] at h1
    have h2 : p16 n = FloatV.ofPat 0 := h1.eq (FloatV.isNaN_ofPat_zero Format.bfloat16_regular)
    rw [h2]; simp
  · -- an empty state
    intro hne
    have hS : ¬ S.Nonempty := fun h => hne ⟨h.choose, Finset.mem_union_left _ h.choose_spec⟩
    have hnv : ∀ n : Fin 64, valid n = false := fun n => by
      by_contra hv
      exact hne ⟨base + (n : ℕ), Finset.mem_union_right _ (mem_laneSet.mpr ⟨n, by simpa using hv, rfl⟩)⟩
    exact hN3 hnv hS

end

end Claims.ExtendAttn
