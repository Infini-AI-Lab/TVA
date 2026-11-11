import Claim
import TVA.Proof
import Proof.NumDefs

/-!
# Numerics 1: the loop's bound and the gates' exponentials

Pure facts about reals (and the device's `exp2` fact), stated in the form the walk uses.
-/

namespace Claims.ChunkFwdO

open TVA

/-- **The loop's bound** for every covered configuration: `n` trips of a `bk`-wide dot, with
`n·bk ≤ 256` (`K ≤ 256`), stay within `6.6·10^-5` of the magnitude and `2^-116` absolutely. -/
theorem loop_to_cD {bk n : ℕ} (hbk : bk = 32 ∨ bk = 64 ∨ bk = 128) (hn : n * bk ≤ 256)
    {P : ℚ} (hP : 0 ≤ P) :
    DotLoop.bound (Device.relDot16 bk) (Device.absDot16 bk) n P ≤
      33 / 500000 * P + 2 ^ (-116 : ℤ) := by
  have hmono : ∀ (ρ α : ℚ) (N : ℕ), 0 ≤ ρ → 0 ≤ α → n ≤ N →
      DotLoop.bound ρ α n P ≤ DotLoop.bound ρ α N P := by
    intro ρ α N hρ hα hnN
    unfold DotLoop.bound
    have h1 : (1 + ρ) ^ n ≤ (1 + ρ) ^ N := pow_le_pow_right₀ (by linarith) hnN
    have h2 : (n : ℚ) ≤ N := by exact_mod_cast hnN
    have h3 : (1 : ℚ) ≤ (1 + ρ) ^ n := one_le_pow₀ (by linarith)
    have h4 : (n : ℚ) * (1 + ρ) ^ n ≤ N * (1 + ρ) ^ N :=
      mul_le_mul h2 h1 (by linarith) (by positivity)
    nlinarith [mul_le_mul_of_nonneg_right h4 hα]
  rcases hbk with rfl | rfl | rfl
  · have hn8 : n ≤ 8 := by omega
    refine (hmono _ _ 8 (by unfold Device.relDot16; positivity)
      (by unfold Device.absDot16; positivity) hn8).trans ?_
    unfold DotLoop.bound Device.relDot16 Device.absDot16
    norm_num
    nlinarith
  · have hn8 : n ≤ 4 := by omega
    refine (hmono _ _ 4 (by unfold Device.relDot16; positivity)
      (by unfold Device.absDot16; positivity) hn8).trans ?_
    unfold DotLoop.bound Device.relDot16 Device.absDot16
    norm_num
    nlinarith
  · have hn8 : n ≤ 2 := by omega
    refine (hmono _ _ 2 (by unfold Device.relDot16; positivity)
      (by unfold Device.absDot16; positivity) hn8).trans ?_
    unfold DotLoop.bound Device.relDot16 Device.absDot16
    norm_num
    nlinarith

/-- `2^z` is within `0.71·|z|` of `1` for small `z`. -/
theorem two_rpow_sub_one_le {z : ℝ} (hz : |z| ≤ 1 / 64) : |(2 : ℝ) ^ z - 1| ≤ 71 / 100 * |z| := by
  have hz' := hz
  rw [abs_le] at hz'
  have hl2 : Real.log 2 < 0.6931471808 := Real.log_two_lt_d9
  have hl2' : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hy : |z * Real.log 2| ≤ 1 / 64 * 0.6931471808 := by
    rw [abs_mul, abs_of_pos hl2']
    nlinarith [abs_nonneg z]
  have hy1 : |z * Real.log 2| ≤ 1 := by linarith
  have hexp := Real.abs_exp_sub_one_sub_id_le hy1
  rw [Real.rpow_def_of_pos (by norm_num : (0 : ℝ) < 2)]
  have e : Real.log 2 * z = z * Real.log 2 := mul_comm _ _
  rw [e]
  have habs : |Real.exp (z * Real.log 2) - 1| ≤ |z * Real.log 2| + (z * Real.log 2) ^ 2 := by
    have := abs_sub_abs_le_abs_sub (Real.exp (z * Real.log 2) - 1) (z * Real.log 2)
    have h2 : |Real.exp (z * Real.log 2) - 1 - z * Real.log 2| ≤ (z * Real.log 2) ^ 2 := hexp
    have h3 := abs_sub_abs_le_abs_sub (Real.exp (z * Real.log 2) - 1) (z * Real.log 2)
    linarith
  have hsq : (z * Real.log 2) ^ 2 = |z * Real.log 2| * |z * Real.log 2| := by
    rw [abs_mul_abs_self]; ring
  have hm : |z * Real.log 2| = |z| * Real.log 2 := by rw [abs_mul, abs_of_pos hl2']
  have h0 := abs_nonneg (z * Real.log 2)
  have h00 := abs_nonneg z
  rw [hsq] at habs
  rw [hm] at habs hy
  nlinarith [mul_nonneg h00 hl2'.le]

/-- **`tl.exp2` on a `float32` input at most `100`**: the result is finite and within
`2^-22·2^x + 2^-126` of `2^x` (`Device.Exp2F32`, the room being automatic). -/
theorem exp2_fact {P : Platform} (hP : Device P) {x r : F32} (hr : P.math .exp2 .binary32 x r)
    (hx : x.isFinite = true) (hle : x.toReal ≤ 100) :
    r.isFinite = true ∧
      |r.toReal - (2 : ℝ) ^ x.toReal| ≤ 2 ^ (-22 : ℤ) * (2 : ℝ) ^ x.toReal + 2 ^ (-126 : ℤ) := by
  obtain ⟨ref, href, h⟩ := hP.exp2F32
  have hr2 : ref = fun x => (2 : ℝ) ^ x := by
    simp only [MathFn.ref] at href
    exact (Option.some.inj href).symm
  subst hr2
  have hpos : (0 : ℝ) < 2 ^ x.toReal := Real.rpow_pos_of_pos (by norm_num) _
  have hle2 : (2 : ℝ) ^ x.toReal ≤ 2 ^ (100 : ℝ) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) hle
  have h100 : (2 : ℝ) ^ (100 : ℝ) = 2 ^ (100 : ℕ) := by
    rw [← Real.rpow_natCast]; norm_num
  have hroom : (1 + (2 : ℝ) ^ (-22 : ℤ)) * |(fun x : ℝ => (2 : ℝ) ^ x) x.toReal| +
      2 ^ (-126 : ℤ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    show (1 + (2 : ℝ) ^ (-22 : ℤ)) * |(2 : ℝ) ^ x.toReal| + 2 ^ (-126 : ℤ) ≤ _
    rw [Format.binary32_maxVal_real, abs_of_pos hpos]
    rw [h100] at hle2
    have : (1 + (2 : ℝ) ^ (-22 : ℤ)) * (2 : ℝ) ^ x.toReal ≤
        (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ (100 : ℕ) :=
      mul_le_mul_of_nonneg_left hle2 (by positivity)
    have h5 : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ (100 : ℕ) + 2 ^ (-126 : ℤ) ≤ (2 ^ 24 - 1) * 2 ^ 104 := by
      norm_num
    linarith
  have := h x r hr hx (Set.mem_univ _) (Set.mem_univ _) hroom
  simpa [abs_of_pos hpos] using this

/-- A perturbed exponent: `exp2` on `r` with `|r - x| ≤ K ≤ 1/64` gives `2^x` within `(2^-22 + 0.7101·K)·2^x + 2^-126`. -/
theorem exp_perturb {x r wh K : ℝ} (hK : K ≤ 1 / 64) (hz : |r - x| ≤ K)
    (hwh : |wh - (2 : ℝ) ^ r| ≤ 2 ^ (-22 : ℤ) * (2 : ℝ) ^ r + 2 ^ (-126 : ℤ)) :
    |wh - (2 : ℝ) ^ x| ≤ (2 ^ (-22 : ℤ) + 7101 / 10000 * K) * (2 : ℝ) ^ x + 2 ^ (-126 : ℤ) := by
  have h1 := two_rpow_sub_one_le (z := r - x) (hz.trans hK)
  have hpos : (0 : ℝ) < 2 ^ x := Real.rpow_pos_of_pos (by norm_num) _
  have hsplit : (2 : ℝ) ^ r = 2 ^ x * 2 ^ (r - x) := by
    rw [← Real.rpow_add (by norm_num)]; ring_nf
  have h2 : |(2 : ℝ) ^ r - 2 ^ x| ≤ 2 ^ x * (71 / 100 * K) := by
    have : (2 : ℝ) ^ r - 2 ^ x = 2 ^ x * (2 ^ (r - x) - 1) := by rw [hsplit]; ring
    rw [this, abs_mul, abs_of_pos hpos]
    exact mul_le_mul_of_nonneg_left (h1.trans (by linarith)) hpos.le
  have h3 : (2 : ℝ) ^ r ≤ 2 ^ x * (1 + 71 / 100 * K) := by
    have := (abs_le.mp h2).2
    nlinarith
  have h4 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
  have hK0 : 0 ≤ K := (abs_nonneg _).trans hz
  have h5 := abs_sub_le (wh) ((2 : ℝ) ^ r) ((2 : ℝ) ^ x)
  have h6 : (2 : ℝ) ^ (-22 : ℤ) * (2 : ℝ) ^ r ≤
      2 ^ (-22 : ℤ) * (2 ^ x * (1 + 71 / 100 * K)) :=
    mul_le_mul_of_nonneg_left h3 (by positivity)
  rw [h4] at h6 hwh ⊢
  nlinarith [mul_nonneg hpos.le hK0]

/-- **The weight with `g`** (`exp2(fl(g[t] - g[s]))`): for `g[t] ≤ 64`, `|g[s]| ≤ 2^64`,
`g[t] - g[s] ≤ 64`, a rounded difference `r` (`|r - (g[t]-g[s])| ≤ 2^-24·|g[t]-g[s]| + 2^-150`) and
a result `wh` of `exp2` on it (`|wh - 2^r| ≤ 2^-22·2^r + 2^-126`): `wh` is within
`2^-16·w + 2^-123` of `w = 2^(g[t]-g[s])`; and `r ≤ 65` (room for `exp2`). -/
theorem w_g_num {gt gs r wh : ℝ} (hgt : gt ≤ 64) (hgs : |gs| ≤ 2 ^ 64) (hd : gt - gs ≤ 64)
    (hr : |r - (gt - gs)| ≤ 2 ^ (-24 : ℤ) * |gt - gs| + 2 ^ (-150 : ℤ))
    (hwh : |wh - (2 : ℝ) ^ r| ≤ 2 ^ (-22 : ℤ) * (2 : ℝ) ^ r + 2 ^ (-126 : ℤ)) :
    r ≤ 65 ∧ |wh - (2 : ℝ) ^ (gt - gs)| ≤ 2 ^ (-16 : ℤ) * (2 : ℝ) ^ (gt - gs) + 2 ^ (-123 : ℤ) := by
  have _ := hgt
  have _ := hgs
  have hu : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have he : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  simp only [hu, he] at hr
  have hwpos : (0 : ℝ) < 2 ^ (gt - gs) := Real.rpow_pos_of_pos (by norm_num) _
  have hrb := abs_le.mp hr
  have hr65 : r ≤ 65 := by
    rcases le_or_gt 0 (gt - gs) with h0 | h0
    · rw [abs_of_nonneg h0] at hrb
      norm_num at hrb
      linarith [hrb.2]
    · rw [abs_of_neg h0] at hrb
      norm_num at hrb
      linarith [hrb.2]
  refine ⟨hr65, ?_⟩
  rcases le_or_gt (-126) (gt - gs) with h126 | h126
  · have hd' : |gt - gs| ≤ 126 := abs_le.mpr ⟨by linarith, by linarith⟩
    have hK : |r - (gt - gs)| ≤ 126 / 16777216 + 1 / 2 ^ 150 := by
      have : 1 / 16777216 * |gt - gs| ≤ 1 / 16777216 * 126 := by
        exact mul_le_mul_of_nonneg_left hd' (by norm_num)
      linarith
    have h1 := exp_perturb (K := 126 / 16777216 + 1 / 2 ^ 150) (by norm_num) hK hwh
    refine h1.trans ?_
    have hc : (2 : ℝ) ^ (-22 : ℤ) + 7101 / 10000 * (126 / 16777216 + 1 / 2 ^ 150) ≤
        (2 : ℝ) ^ (-16 : ℤ) := by norm_num
    have hc2 : (2 : ℝ) ^ (-126 : ℤ) ≤ 2 ^ (-123 : ℤ) := by norm_num
    nlinarith [mul_le_mul_of_nonneg_right hc hwpos.le]
  · have hrle : r ≤ -125 := by
      rw [abs_of_neg (by linarith : gt - gs < 0)] at hrb
      norm_num at hrb
      linarith [hrb.2]
    have h2r : (2 : ℝ) ^ r ≤ 1 / 2 ^ 125 := by
      have : (2 : ℝ) ^ r ≤ 2 ^ (-125 : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) hrle
      refine this.trans ?_
      rw [Real.rpow_neg (by norm_num)]
      norm_num
    have h2d : (2 : ℝ) ^ (gt - gs) ≤ 1 / 2 ^ 126 := by
      have : (2 : ℝ) ^ (gt - gs) ≤ 2 ^ (-126 : ℝ) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) h126.le
      refine this.trans ?_
      rw [Real.rpow_neg (by norm_num)]
      norm_num
    have h2rpos : (0 : ℝ) < 2 ^ r := Real.rpow_pos_of_pos (by norm_num) _
    have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    have h126' : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
    have h123 : (2 : ℝ) ^ (-123 : ℤ) = 1 / 2 ^ 123 := by norm_num
    have h16 : (2 : ℝ) ^ (-16 : ℤ) = 1 / 65536 := by norm_num
    simp only [h22, h126', h123, h16] at hwh ⊢
    have hwh' := abs_le.mp hwh
    rw [abs_le]
    constructor <;> nlinarith [mul_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 65536) hwpos.le]

/-- **The gate with `g_gamma`** (`exp2(fl(γ·(i+1)))`), `Γ = 2^-15·|γ|`, `1 ≤ n ≤ 64` (`n = i+1`):
for a rounded product `bg` and a result `Gh` of `exp2` on it: `bg ≤ 65` (room) and `Gh` is within
`(2^-21 + Γ/2)·2^(γn) + 2^-126` of `2^(γn)`. -/
theorem gam_G_num {γ n bg Gh : ℝ} (hγ : |γ| ≤ 64) (hγ1 : γ ≤ 1) (hn1 : 1 ≤ n) (hn : n ≤ 64)
    (hbg : |bg - γ * n| ≤ 2 ^ (-24 : ℤ) * |γ * n| + 2 ^ (-150 : ℤ))
    (hGh : |Gh - (2 : ℝ) ^ bg| ≤ 2 ^ (-22 : ℤ) * (2 : ℝ) ^ bg + 2 ^ (-126 : ℤ)) :
    bg ≤ 65 ∧ |Gh - (2 : ℝ) ^ (γ * n)| ≤
      (2 ^ (-21 : ℤ) + 2 ^ (-15 : ℤ) * |γ| / 2) * (2 : ℝ) ^ (γ * n) + 2 ^ (-126 : ℤ) := by
  have hu : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have he : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have hn0 : (0 : ℝ) < n := by linarith
  have habs : |γ * n| = |γ| * n := by rw [abs_mul, abs_of_pos hn0]
  rw [habs] at hbg
  simp only [hu, he] at hbg
  have ha0 := abs_nonneg γ
  have ha64 : |γ| * n ≤ 64 * |γ| := by nlinarith
  have hbgb := abs_le.mp hbg
  have hx64 : γ * n ≤ 64 := by nlinarith
  have hK : |bg - γ * n| ≤ |γ| / 262144 + 1 / 2 ^ 150 := by
    rw [abs_le]; constructor <;> norm_num at hbgb ⊢ <;> linarith [hbgb.1, hbgb.2]
  have hK1 : |γ| / 262144 + 1 / 2 ^ 150 ≤ 1 / 64 := by
    have : |γ| / 262144 ≤ 64 / 262144 := by linarith
    norm_num at this ⊢; linarith
  refine ⟨?_, ?_⟩
  · have := (abs_le.mp hK).2
    have h2 : |γ| / 262144 ≤ 64 / 262144 := by linarith
    norm_num at this h2 ⊢
    linarith
  · have h1 := exp_perturb (K := |γ| / 262144 + 1 / 2 ^ 150) hK1 hK hGh
    refine h1.trans ?_
    have hwpos : (0 : ℝ) < 2 ^ (γ * n) := Real.rpow_pos_of_pos (by norm_num) _
    have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    have h21 : (2 : ℝ) ^ (-21 : ℤ) = 1 / 2097152 := by norm_num
    have h15 : (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num
    simp only [h22, h21, h15]
    have hc : 1 / 4194304 + 7101 / 10000 * (|γ| / 262144 + 1 / 2 ^ 150) ≤
        1 / 2097152 + 1 / 32768 * |γ| / 2 := by
      norm_num; linarith
    nlinarith [mul_le_mul_of_nonneg_right hc hwpos.le]

/-- **The weight with `g_gamma`** (`exp2(b_g[i] - b_g[i'])`, the difference possibly fused),
`1 ≤ n' ≤ n ≤ 64`: for the rounded products `bi`, `bj` of `γ·n`, `γ·n'`, a difference `r` within
`3u(|bi|+|bj|) + 4η` of `bi - bj`, and a result `wh` of `exp2` on it: `r ≤ 65` and `wh` is
within `(2^-16 + 0.72·Γ)·w + 2^-123` of `w = 2^(γ(n-n'))`, `Γ = 2^-15·|γ|`. -/
theorem gam_w_num {γ n n' bi bj r wh : ℝ} (hγ : |γ| ≤ 64) (hγ1 : γ ≤ 1) (hn' : 1 ≤ n')
    (hnn : n' ≤ n) (hn : n ≤ 64)
    (hbi : |bi - γ * n| ≤ 2 ^ (-24 : ℤ) * |γ * n| + 2 ^ (-150 : ℤ))
    (hbj : |bj - γ * n'| ≤ 2 ^ (-24 : ℤ) * |γ * n'| + 2 ^ (-150 : ℤ))
    (hr : |r - (bi + -bj)| ≤ 3 * 2 ^ (-24 : ℤ) * (|bi| + |-bj|) + 4 * 2 ^ (-150 : ℤ))
    (hwh : |wh - (2 : ℝ) ^ r| ≤ 2 ^ (-22 : ℤ) * (2 : ℝ) ^ r + 2 ^ (-126 : ℤ)) :
    r ≤ 65 ∧ |wh - (2 : ℝ) ^ (γ * (n - n'))| ≤
      (2 ^ (-16 : ℤ) + 72 / 100 * (2 ^ (-15 : ℤ) * |γ|)) * (2 : ℝ) ^ (γ * (n - n')) +
        2 ^ (-123 : ℤ) := by
  have hu : (2 : ℝ) ^ (-24 : ℤ) = 1 / 16777216 := by norm_num
  have he : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have hn0 : (0 : ℝ) < n := by linarith
  have hn0' : (0 : ℝ) < n' := by linarith
  have habs : |γ * n| = |γ| * n := by rw [abs_mul, abs_of_pos hn0]
  have habs' : |γ * n'| = |γ| * n' := by rw [abs_mul, abs_of_pos hn0']
  rw [habs] at hbi
  rw [habs'] at hbj
  simp only [hu, he] at hbi hbj hr
  have ha0 := abs_nonneg γ
  have ha64 : |γ| * n ≤ 64 * |γ| := by nlinarith
  have ha64' : |γ| * n' ≤ 64 * |γ| := by nlinarith
  have hbib := abs_le.mp hbi
  have hbjb := abs_le.mp hbj
  have hrb := abs_le.mp hr
  have hbi1 := abs_sub_abs_le_abs_sub bi (γ * n)
  have hbj1 := abs_sub_abs_le_abs_sub bj (γ * n')
  rw [habs] at hbi1
  rw [habs'] at hbj1
  have hnb : |-bj| = |bj| := abs_neg bj
  rw [hnb] at hrb
  have hx63 : γ * (n - n') ≤ 63 := by nlinarith
  have hK : |r - γ * (n - n')| ≤ 513 / 16777216 * |γ| + 7 / 2 ^ 150 := by
    rw [abs_le]
    constructor <;> norm_num at hbib hbjb hrb hbi1 hbj1 ⊢ <;>
      nlinarith [hbib.1, hbib.2, hbjb.1, hbjb.2, hrb.1, hrb.2]
  have hK1 : 513 / 16777216 * |γ| + 7 / 2 ^ 150 ≤ 1 / 64 := by
    have : 513 / 16777216 * |γ| ≤ 513 / 16777216 * 64 := by linarith
    norm_num at this ⊢; linarith
  refine ⟨?_, ?_⟩
  · have := (abs_le.mp hK).2
    have h2 : 513 / 16777216 * |γ| ≤ 513 / 16777216 * 64 := by linarith
    norm_num at this h2 ⊢
    linarith
  · have h1 := exp_perturb (K := 513 / 16777216 * |γ| + 7 / 2 ^ 150) hK1 hK hwh
    refine h1.trans ?_
    have hwpos : (0 : ℝ) < 2 ^ (γ * (n - n')) := Real.rpow_pos_of_pos (by norm_num) _
    have h22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    have h16 : (2 : ℝ) ^ (-16 : ℤ) = 1 / 65536 := by norm_num
    have h15 : (2 : ℝ) ^ (-15 : ℤ) = 1 / 32768 := by norm_num
    have h126 : (2 : ℝ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
    have h123 : (2 : ℝ) ^ (-123 : ℤ) = 1 / 2 ^ 123 := by norm_num
    simp only [h22, h16, h15, h126, h123]
    have hc : 1 / 4194304 + 7101 / 10000 * (513 / 16777216 * |γ| + 7 / 2 ^ 150) ≤
        1 / 65536 + 72 / 100 * (1 / 32768 * |γ|) := by
      norm_num; linarith
    nlinarith [mul_le_mul_of_nonneg_right hc hwpos.le, (by norm_num : (1 : ℝ) / 2 ^ 126 ≤ 1 / 2 ^ 123)]

end Claims.ChunkFwdO
