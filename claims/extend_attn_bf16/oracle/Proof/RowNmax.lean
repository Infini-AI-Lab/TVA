import Proof.Consts

/-!
# The new reference of a row (`n_e_max`)

The running maximum `n_e_max = tl.maximum (row_max_fixed, e_max)`: its value in each of the cases
(something seen or not, a valid lane or not).
-/

namespace Claims.ExtendAttn

open TVA

noncomputable section

/-- `FloatV.eq` against `−∞` on a non-NaN value is the equality with `−∞`. -/
theorem eq_negInf_iff {x : F32} (hx : x.isNaN = false) : FloatV.eq x FloatV.negInf = true ↔ x = FloatV.negInf := by
  have hs : Format.binary32.specials = .ieee := rfl
  have hn := FloatV.negInf_isNaN (f := Format.binary32) hs
  rw [FloatV.eq_iff hx hn]
  constructor
  · intro h
    exact (FloatV.negInf_toRat_le Format.binary32_regular hs hx).2 h.symm
  · rintro rfl; rfl

/-- **The new reference** `n_e_max = max (row_max_fixed, e_max)`, `row_max_fixed = row_max`
unless it is `−∞` (then `−1e20`). The lanes `qk` are finite (valid) or `−∞`; the old reference
`emax` is a finite value (something was seen) or `−∞` or `−1e20` (nothing). -/
theorem nmax_spec {valid : Fin 64 → Bool} {qk : Fin 64 → F32} {rmax emax nmax : F32} {m Xp : ℝ}
    {Sne : Prop}
    (hXp : Xp ≤ 2 ^ 13)
    (hq : ∀ n, valid n = true → (qk n).isFinite = true ∧ |(qk n).toReal| ≤ Xp)
    (hmax1 : (∃ n, valid n = true) → rmax.isFinite = true ∧ (∃ n, valid n = true ∧ rmax = qk n) ∧
      ∀ n, valid n = true → (qk n).toReal ≤ rmax.toReal)
    (hmax2 : (∀ n, valid n = false) → rmax = FloatV.negInf)
    (hSe : Sne → emax.isFinite = true ∧ emax.toReal = m ∧ |m| ≤ Xp)
    (hSn : ¬ Sne → emax = FloatV.negInf ∨ emax = neg1e20)
    (hnm : FloatV.MaxRel true (if FloatV.eq rmax ninf32 = true then neg1e20 else rmax) emax nmax) :
    nmax.isFinite = true ∧
      ((∃ n, valid n = true) → (∀ n, valid n = true → (qk n).toReal ≤ nmax.toReal) ∧
        (Sne → m ≤ nmax.toReal) ∧ ((Sne ∧ nmax.toReal = m) ∨ ∃ n, valid n = true ∧ (qk n).toReal = nmax.toReal)) ∧
      ((∀ n, valid n = false) → Sne → nmax = emax) ∧
      ((∀ n, valid n = false) → ¬ Sne → nmax = neg1e20) := by
  have hs : Format.binary32.specials = .ieee := rfl
  have hnN : (FloatV.negInf : F32).isNaN = false := FloatV.negInf_isNaN hs
  obtain ⟨hN0f, hN0le, -⟩ := neg1e20_spec
  have hN0n : neg1e20.isNaN = false := (FloatV.isFinite_iff.mp hN0f).1
  have hlt : ∀ x : F32, x.isFinite = true → (FloatV.negInf : F32).toReal < x.toReal := by
    intro x hx
    unfold FloatV.toReal
    exact_mod_cast negInf_lt_of_finite hx
  have hXp0 : ∀ n, valid n = true → -Xp ≤ (qk n).toReal := fun n hn => by
    have := (abs_le.mp (hq n hn).2).1; linarith
  have hXp1 : -(2 : ℝ) ^ 13 ≤ -Xp ∨ True := Or.inr trivial
  by_cases hV : ∃ n, valid n = true
  · obtain ⟨hrf, ⟨n0, hn0, hrq⟩, hrle⟩ := hmax1 hV
    have hrn : rmax.isNaN = false := (FloatV.isFinite_iff.mp hrf).1
    have heq : FloatV.eq rmax ninf32 = false := by
      rw [ninf32_eq]
      cases h : FloatV.eq rmax FloatV.negInf
      · rfl
      · exfalso
        have := (eq_negInf_iff hrn).mp h
        rw [this] at hrf; rw [negInf_not_finite] at hrf; cases hrf
    have hrm : FloatV.eq rmax ninf32 = false := heq
    simp only [hrm, Bool.false_eq_true, ite_false] at hnm
    have hr0 : -Xp ≤ rmax.toReal := by
      have := hrle n0 hn0; have := hXp0 n0 hn0; linarith
    by_cases hS : Sne
    · obtain ⟨hef, hev, hem⟩ := hSe hS
      have hen : emax.isNaN = false := (FloatV.isFinite_iff.mp hef).1
      have hmx := hnm.toReal_eq_max hrn hen
      have hmem := hnm.mem hrn hen
      have hnf : nmax.isFinite = true := by rcases hmem with h | h <;> rw [h] <;> assumption
      refine ⟨hnf, fun _ => ⟨fun n hn => ?_, fun _ => ?_, ?_⟩, fun h => absurd (h n0) (by simp [hn0]), fun _ h => absurd hS h⟩
      · rw [hmx]; exact le_trans (hrle n hn) (le_max_left _ _)
      · rw [hmx, hev]; exact le_max_right _ _
      · rcases le_total rmax.toReal emax.toReal with h | h
        · left; exact ⟨hS, by rw [hmx, hev] at *; rw [max_eq_right h]⟩
        · right; refine ⟨n0, hn0, ?_⟩
          rw [hmx, max_eq_left h, ← hrq]
    · have hSn' := hSn hS
      have hebound : emax.toReal < rmax.toReal := by
        rcases hSn' with h | h
        · rw [h]; exact hlt _ hrf
        · rw [h]
          have : (2 : ℝ) ^ 60 > 2 ^ 13 := by norm_num
          linarith
      have hen : emax.isNaN = false := by rcases hSn' with h | h <;> rw [h] <;> assumption
      have hmx := hnm.toReal_eq_max hrn hen
      have hmem := hnm.mem hrn hen
      have hnr : nmax = rmax := by
        rcases hmem with h | h
        · exact h
        · exfalso
          rw [h] at hmx
          rw [max_eq_left hebound.le] at hmx
          linarith
      subst hnr
      refine ⟨hrf, fun _ => ⟨fun n hn => hrle n hn, fun h => absurd h hS, ?_⟩, fun h => absurd (h n0) (by simp [hn0]),
        fun h => absurd (h n0) (by simp [hn0])⟩
      right; exact ⟨n0, hn0, by rw [hrq]⟩
  · have hnv : ∀ n, valid n = false := fun n => by simpa using fun h => hV ⟨n, h⟩
    have hrn : rmax = FloatV.negInf := hmax2 hnv
    have heq : FloatV.eq rmax ninf32 = true := by
      rw [ninf32_eq, hrn]; exact (eq_negInf_iff hnN).mpr rfl
    simp only [heq, ite_true] at hnm
    refine ⟨?_, fun h => absurd h hV, ?_, ?_⟩
    · by_cases hS : Sne
      · obtain ⟨hef, hev, hem⟩ := hSe hS
        have hen : emax.isNaN = false := (FloatV.isFinite_iff.mp hef).1
        have hmem := hnm.mem hN0n hen
        rcases hmem with h | h <;> rw [h] <;> assumption
      · rcases hSn hS with h | h
        · have hmx := hnm.toReal_eq_max hN0n (by rw [h]; exact hnN)
          have hmem := hnm.mem hN0n (by rw [h]; exact hnN)
          rcases hmem with h' | h'
          · rw [h']; exact hN0f
          · exfalso; rw [h] at h'; rw [h'] at hmx
            rw [h] at hmx
            have := hlt _ hN0f
            rw [max_eq_left this.le] at hmx
            linarith
        · rw [h] at hnm
          have hmem := hnm.mem hN0n hN0n
          rcases hmem with h' | h' <;> rw [h'] <;> exact hN0f
    · intro _ hS
      obtain ⟨hef, hev, hem⟩ := hSe hS
      have hen : emax.isNaN = false := (FloatV.isFinite_iff.mp hef).1
      have hmx := hnm.toReal_eq_max hN0n hen
      have hmem := hnm.mem hN0n hen
      rcases hmem with h | h
      · exfalso
        rw [h] at hmx
        have : neg1e20.toReal < emax.toReal := by
          rw [hev]; have := (abs_le.mp hem).1
          have : (2 : ℝ) ^ 60 > 2 ^ 13 := by norm_num
          linarith
        rw [max_eq_right this.le] at hmx
        linarith
      · exact h
    · intro _ hS
      rcases hSn hS with h | h
      · rw [h] at hnm
        have hmx := hnm.toReal_eq_max hN0n hnN
        have hmem := hnm.mem hN0n hnN
        rcases hmem with h' | h'
        · exact h'
        · exfalso
          rw [h'] at hmx
          have := hlt _ hN0f
          rw [max_eq_left this.le] at hmx
          linarith
      · rw [h] at hnm
        have hmem := hnm.mem hN0n hN0n
        rcases hmem with h' | h' <;> exact h'

end

end Claims.ExtendAttn
