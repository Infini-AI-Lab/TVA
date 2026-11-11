import Claim
import TVA.Proof
import Proof.Setup
import Proof.Addrs
import Proof.Cast
import Proof.Cap
import Proof.CapReal
import Proof.CapRatio
import Proof.Tanh
import Proof.Const

/-!
# The cap bound: the computed soft cap is within `capEps` of `capRef (μ)`

The program computes, per expert `e`,

  `l = logits[e]`,  `s = l / 30`,  `d2 = 2·s`,  `E = exp d2`,  `q = (E − 1)/(E + 1)`,
  `C = 30·q`,

while the reference is `capRef μ = 30·tanh(μ/30) = 30·(e^{2μ/30} − 1)/(e^{2μ/30} + 1)`.

Write `S = absS init d t e`, `μ = mu init d t e`, `τ = tolL d S / 15 + 2^-21·(|μ| + tolL d S)/15 +
2^-146`, so that `|d2 − 2μ/30| ≤ τ`: `l/30` (`DivFull`, `2^-22·|l/30| + 2^-148`), then `2·s`
(`u·|2s| + η`). The map `t ↦ 30·tanh(t/2)` is `15`-Lipschitz (`capRef_lipschitz`, scaled), so the
input error contributes `15·τ = (1 + 2^-21)·tolL + 2^-21·|μ| + 2^-145`.

For the formula itself at the computed `d2`:
* `|μ| ≤ S` and `μ + capTol ≤ 1320` give `d2 ≤ 88` (and `d2` is not too negative: `d2 ≥ −2^120/15`),
  so `exp` is finite and `ExpF32` applies with `X = |d2|`: `|Ẽ − e^{d2}| ≤ rel·e^{d2} + 2^-140`,
  `rel = 2^-22 + 2^-23·|d2|`;
* `g(y) = (y−1)/(y+1)` moves by `2|Ẽ − e^{d2}|/((Ẽ+1)(e^{d2}+1))` (`cap_abs_diff`), and
  `(2^-22 + 2^-23·|d2|)·2e^{d2}/(e^{d2}+1)^2 ≤ 2·(2^-22 + 2^-21·|d2|)·e^{|d2|}/(e^{|d2|}+1)^2`;
  the `·30` and the `30·` scale this to at most `φ_exp` (see `Cap.exp_slope`);
* `Ẽ ∓ 1` (`sub_err`, `add_err`), `/` (`DivFull`), `·30` (`mul_err`) add at most `φ_ops`.

`φ = φ_exp + φ_ops ≤ 1.71e-5 < 2^-14 = 6.1e-5`, so in total

  `|C − capRef μ| ≤ (1 + 2^-21)·tolL d S + 2^-21·|μ| + φ`.

Then `tolL d S ≤ 40.0002·u·S + a₀` (`a₀ < 2^-134`, from `(1+3u)^D − 1 ≤ 39.0001u`, `D = ⌈log₂
BLOCK⌉ ≤ 13`), and `2^-21·|μ| ≤ 2^-21·S`, so `15τ ≤ (40.0002u·(1+2^-21) + 2^-21)·S + a₀ ≤
(48u − 9u)·S`; with `φ + a₀ ≤ 2^-14` this is `≤ 2^-18·S + 2^-14 = capEps S` (`64u = 2^-18`,
margin `1.33` on `S`).

The very negative case is the same bound: `d2 ≤ −91` gives `e^{d2} ≤ 2^-130`
(`Tanh.one_div_exp_two_le`), `|Ẽ| ≤ 2^-130 + 2^-140`, `q ≈ −1`, `C ≈ −30`, `capRef μ ≈ −30`, and
`S ≥ |μ| ≥ 1365` makes `capEps S ≥ 5·10^-3 ≫ 3·10^-39`.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

/-! ## The magnitude of the exact logit -/

private theorem absS_nonneg (m : Mem) (d : Inst) (t e : Nat) : 0 ≤ absS m d t e := by
  unfold absS
  exact Finset.sum_nonneg (fun h _ => abs_nonneg _)

private theorem abs_mu_le_absS (m : Mem) (d : Inst) (t e : Nat) :
    |mu m d t e| ≤ absS m d t e := by
  unfold mu absS
  exact Finset.abs_sum_le_sum_abs _ _

/-! ## Bridges from the rational to the real statements of the operations -/

private theorem mul_real {x y r : FloatV Format.binary32}
    (h : (x.mul y).NaNEq r) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal * y.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ)) :
    r.isFinite = true ∧ |r.toReal - x.toReal * y.toReal| ≤ |x.toReal * y.toReal| * u32 + eta32 := by
  have hmaxq : |x.toRat * y.toRat| ≤ Format.binary32.maxVal := by
    apply Rat.cast_le (K := ℝ).mp
    have h2 : ((|x.toRat * y.toRat| : ℚ) : ℝ) = |x.toReal * y.toReal| := by
      push_cast
      rfl
    rw [h2]; exact hmax
  obtain ⟨hfin, herr⟩ := FloatV.mul_err (f := Format.binary32) bin32_regular hx hy hmaxq
  have hr : r = x.mul y := FloatV.NaNEq.eq h (Format.not_isNaN_of_isFinite hfin)
  refine ⟨?_, ?_⟩
  · rw [hr]; exact hfin
  · rw [hr]
    have hh := Rat.cast_le (K := ℝ).mpr herr
    push_cast at hh ⊢
    simpa only [u32_eq, eta32_eq, FloatV.toReal] using hh

private theorem add_real {x y r : FloatV Format.binary32}
    (h : (x.add y).NaNEq r) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal + y.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ)) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal + y.toReal)| ≤ |x.toReal + y.toReal| * u32 + eta32 := by
  have hmaxq : |x.toRat + y.toRat| ≤ Format.binary32.maxVal := by
    apply Rat.cast_le (K := ℝ).mp
    have h2 : ((|x.toRat + y.toRat| : ℚ) : ℝ) = |x.toReal + y.toReal| := by
      push_cast
      rfl
    rw [h2]; exact hmax
  obtain ⟨hfin, herr⟩ := FloatV.add_err (f := Format.binary32) bin32_regular hx hy hmaxq
  have hr : r = x.add y := FloatV.NaNEq.eq h (Format.not_isNaN_of_isFinite hfin)
  refine ⟨?_, ?_⟩
  · rw [hr]; exact hfin
  · rw [hr]
    have hh := Rat.cast_le (K := ℝ).mpr herr
    push_cast at hh ⊢
    simpa only [u32_eq, eta32_eq, FloatV.toReal] using hh

private theorem sub_real {x y r : FloatV Format.binary32}
    (h : (x.sub y).NaNEq r) (hx : x.isFinite = true) (hy : y.isFinite = true)
    (hmax : |x.toReal - y.toReal| ≤ ((Format.binary32.maxVal : ℚ) : ℝ)) :
    r.isFinite = true ∧
      |r.toReal - (x.toReal - y.toReal)| ≤ |x.toReal - y.toReal| * u32 + eta32 := by
  have hmaxq : |x.toRat - y.toRat| ≤ Format.binary32.maxVal := by
    apply Rat.cast_le (K := ℝ).mp
    have h2 : ((|x.toRat - y.toRat| : ℚ) : ℝ) = |x.toReal - y.toReal| := by
      push_cast
      rfl
    rw [h2]; exact hmax
  obtain ⟨hfin, herr⟩ := FloatV.sub_err (f := Format.binary32) bin32_regular hx hy hmaxq
  have hr : r = x.sub y := FloatV.NaNEq.eq h (Format.not_isNaN_of_isFinite hfin)
  refine ⟨?_, ?_⟩
  · rw [hr]; exact hfin
  · rw [hr]
    have hh := Rat.cast_le (K := ℝ).mpr herr
    push_cast at hh ⊢
    simpa only [u32_eq, eta32_eq, FloatV.toReal] using hh

private theorem div_full_real {P : Platform} (hdivfull : Device.DivFull P)
    {x y r : FloatV Format.binary32} (h : P.div Format.binary32 x y r)
    (hx : x.isFinite = true) (hy : y.isFinite = true) (hy0 : y.toReal ≠ 0)
    (hmax : |x.toReal / y.toReal| ≤ (2:ℝ)^126) :
    r.isFinite = true ∧ |r.toReal - x.toReal / y.toReal| ≤
      (2:ℝ)^(-22:ℤ) * |x.toReal / y.toReal| + (2:ℝ)^(-148:ℤ) := by
  have hyq : y.toRat ≠ 0 := by
    intro h0
    exact hy0 (by simp [FloatV.toReal, h0])
  have hmaxq : |x.toRat / y.toRat| ≤ (2:ℚ)^126 := by
    apply Rat.cast_le (K := ℝ).mp
    have h2 : ((|x.toRat / y.toRat| : ℚ) : ℝ) = |x.toReal / y.toReal| := by
      push_cast
      rfl
    have h3 : (((2:ℚ)^126 : ℚ) : ℝ) = (2:ℝ)^126 := by push_cast; norm_num
    rw [h2, h3]; exact hmax
  obtain ⟨hfin, herr⟩ := hdivfull x y r h hx hy hyq hmaxq
  refine ⟨hfin, ?_⟩
  have hh := Rat.cast_le (K := ℝ).mpr herr
  push_cast at hh ⊢
  simpa only [FloatV.toReal] using hh

private theorem cap_budget :
    30 * ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) + 30 * (5 * u32)
      + 30 * ((2 : ℝ) ^ (-22 : ℤ) * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ))
        + (2 : ℝ) ^ (-148 : ℤ))
      + (30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) * u32
        + 30 * (2 : ℝ) ^ (-148 : ℤ) * u32 + eta32)
      + (2 : ℝ) ^ (-20 : ℤ) ≤ (2 : ℝ) ^ (-14 : ℤ) := by
  rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl, show eta32 = (2 : ℝ) ^ (-150 : ℤ) from rfl]
  norm_num

private theorem cap_abs_budget :
    30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) * (1 + u32)
      + 30 * (2 : ℝ) ^ (-148 : ℤ) * (1 + u32) + eta32 ≤ 30 * (1 + (2 : ℝ) ^ (-20 : ℤ)) := by
  rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl, show eta32 = (2 : ℝ) ^ (-150 : ℤ) from rfl]
  norm_num

/-- **The cap bound.** Each computed capped logit is finite, bounded by `30(1+2^-20)`, and within `capEps` of the exact soft cap `capRef (μ_e)`. -/
theorem cap_bound {d : Inst} (hd : d.ok) {init : Mem} {P : Platform}
    (_hexp : Device.MathTotal P MathFn.exp Format.binary32)
    (hexpacc : Device.ExpF32 P)
    (_hdiv : Device.DivTotal P Format.binary32)
    (hdivfull : Device.DivFull P)
    (t : Nat) (e : Fin (8 - 0))
    (hcap : mu init d t e + capTol d (mu init d t e) (absS init d t e) ≤ 1320)
    (hS : absS init d t e ≤ (2 : ℝ) ^ (120 : ℕ))
    {logits lsc doubled exped top bottom q capped : Tensor [8 - 0] (FloatV Format.binary32)}
    (hlogit : (logits.get1 e).isFinite = true ∧
      |(logits.get1 e).toReal - mu init d t e| ≤ tolL d (absS init d t e))
    (hls : ∀ (j : Idx [8 - 0]),
      P.div Format.binary32 (logits.get j) ((bc (c32 0x403E000000000000)).get j) (lsc.get j))
    (hdoubled : ∀ (j : Idx [8 - 0]),
      (((bc (c32 0x4000000000000000)).get j).mul (lsc.get j)).NaNEq (doubled.get j))
    (hexped : ∀ (j : Idx [8 - 0]),
      P.math MathFn.exp Format.binary32 (doubled.get j) (exped.get j))
    (htop : ∀ (j : Idx [8 - 0]),
      ((exped.get j).sub ((bc (c32 0x3FF0000000000000)).get j)).NaNEq (top.get j))
    (hbottom : ∀ (j : Idx [8 - 0]),
      ((exped.get j).add ((bc (c32 0x3FF0000000000000)).get j)).NaNEq (bottom.get j))
    (hq : ∀ (j : Idx [8 - 0]),
      P.div Format.binary32 (top.get j) (bottom.get j) (q.get j))
    (hcapped : ∀ (j : Idx [8 - 0]),
      ((q.get j).mul ((bc (c32 0x403E000000000000)).get j)).NaNEq (capped.get j)) :
    (capped.get1 e).isFinite = true ∧
      |(capped.get1 e).toReal - capRef (mu init d t e)| ≤ capEps (absS init d t e) ∧
      |(capped.get1 e).toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)) := by
  -- ## The exact logit and the tolerances
  set M : ℝ := mu init d t e with hM
  set S : ℝ := absS init d t e with hSdef
  set L : ℝ := tolL d S with hLdef
  have hS0 : 0 ≤ S := by rw [hSdef]; exact absS_nonneg init d t e
  have hMS : |M| ≤ S := by rw [hM, hSdef]; exact abs_mu_le_absS init d t e
  have hS120 : S ≤ (2 : ℝ) ^ (120 : ℕ) := by rw [hSdef]; exact hS
  have hLle : L ≤ 40.0002 * u32 * S + (2 : ℝ) ^ (-127 : ℤ) := by
    rw [hLdef]; exact tolL_le d hd hS0
  have hL0 : 0 ≤ L := by rw [hLdef]; exact tolL_nonneg d S hS0
  have hcapML : (1 + (2 : ℝ) ^ (-21 : ℤ)) * (M + L) ≤ 1320 := by
    have hcap' : M + capTol d M S ≤ 1320 := by rw [hM, hSdef]; exact hcap
    have hcapT : capTol d M S =
        (1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M| + (2 : ℝ) ^ (-14 : ℤ) := by
      rw [hLdef]; unfold capTol; rfl
    have hb : M + ((1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M| +
        (2 : ℝ) ^ (-14 : ℤ)) ≤ 1320 := by
      rw [← hcapT]; exact hcap'
    have hb2 : M + ((1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M|) ≤ 1320 := by
      have h14 : (0 : ℝ) ≤ (2 : ℝ) ^ (-14 : ℤ) := by norm_num
      have hb' : (M + ((1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M|)) +
          (2 : ℝ) ^ (-14 : ℤ) ≤ 1320 := by
        have he : (M + ((1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M|)) +
            (2 : ℝ) ^ (-14 : ℤ)
            = M + ((1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M| +
              (2 : ℝ) ^ (-14 : ℤ)) := by ring
        rw [he]; exact hb
      exact le_trans (le_add_of_nonneg_right h14) hb'
    have hm : (2 : ℝ) ^ (-21 : ℤ) * M ≤ (2 : ℝ) ^ (-21 : ℤ) * |M| :=
      mul_le_mul_of_nonneg_left (le_abs_self M) (by norm_num)
    have hstep : (1 + (2 : ℝ) ^ (-21 : ℤ)) * (M + L) ≤
        M + ((1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M|) := by
      have heq : (1 + (2 : ℝ) ^ (-21 : ℤ)) * (M + L)
          = (M + (1 + (2 : ℝ) ^ (-21 : ℤ)) * L) + (2 : ℝ) ^ (-21 : ℤ) * M := by ring
      have heq2 : M + ((1 + (2 : ℝ) ^ (-21 : ℤ)) * L + (2 : ℝ) ^ (-21 : ℤ) * |M|)
          = (M + (1 + (2 : ℝ) ^ (-21 : ℤ)) * L) + (2 : ℝ) ^ (-21 : ℤ) * |M| := by ring
      rw [heq, heq2]
      exact add_le_add_right hm (M + (1 + (2 : ℝ) ^ (-21 : ℤ)) * L)
    exact le_trans hstep hb2
  -- ## The computed logit
  set l : ℝ := (logits.get1 e).toReal with hl
  have hlfin : (logits.get1 e).isFinite = true := hlogit.1
  have hlM : |l - M| ≤ L := by
    have h := hlogit.2
    simpa only [← hl, ← hM, ← hLdef, ← hSdef] using h
  have hlS : |l| ≤ S + L := by
    have h1 : l = M + (l - M) := by ring
    rw [h1]
    exact le_trans (abs_add_le M (l - M)) (add_le_add hMS hlM)
  have hlML : |l| ≤ |M| + L := by
    have h1 : l = M + (l - M) := by ring
    rw [h1]
    exact le_trans (abs_add_le M (l - M)) (add_le_add le_rfl hlM)
  have hlabs : |l| ≤ (2 : ℝ) ^ (122 : ℕ) := by
    have hL2 : L ≤ (2 : ℝ) ^ (121 : ℕ) := by
      have hc : 40.0002 * u32 ≤ 1 := by
        rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl]; norm_num
      have hm : 40.0002 * u32 * S ≤ 1 * S := mul_le_mul_of_nonneg_right hc hS0
      have h2 : 40.0002 * u32 * S ≤ (2 : ℝ) ^ (120 : ℕ) := by
        rw [one_mul] at hm
        exact le_trans hm hS120
      have h3 : (2 : ℝ) ^ (-127 : ℤ) ≤ (2 : ℝ) ^ (120 : ℕ) := by
        have h4 : (2 : ℝ) ^ (-127 : ℤ) ≤ 1 := by norm_num
        have h5 : (1 : ℝ) ≤ (2 : ℝ) ^ (120 : ℕ) := by norm_num
        exact le_trans h4 h5
      have h4 : 40.0002 * u32 * S + (2 : ℝ) ^ (-127 : ℤ) ≤ (2 : ℝ) ^ (121 : ℕ) := by
        have h5 : (2 : ℝ) ^ (120 : ℕ) + (2 : ℝ) ^ (120 : ℕ) ≤ (2 : ℝ) ^ (121 : ℕ) :=
          by norm_num
        exact le_trans (add_le_add h2 h3) h5
      exact le_trans hLle h4
    have h2 : S + L ≤ (2 : ℝ) ^ (120 : ℕ) + (2 : ℝ) ^ (121 : ℕ) := add_le_add hS120 hL2
    have h3 : (2 : ℝ) ^ (120 : ℕ) + (2 : ℝ) ^ (121 : ℕ) ≤ (2 : ℝ) ^ (122 : ℕ) := by norm_num
    exact le_trans hlS (le_trans h2 h3)
  have hl30 : |l / 30| ≤ (2 : ℝ) ^ (119 : ℕ) := by
    have h1 : |l / 30| = |l| / 30 := by rw [abs_div]; norm_num
    rw [h1]
    have h2 : |l| / 30 ≤ (2 : ℝ) ^ (122 : ℕ) / 30 :=
      div_le_div_of_nonneg_right hlabs (by norm_num)
    have h3 : (2 : ℝ) ^ (122 : ℕ) / 30 ≤ (2 : ℝ) ^ (119 : ℕ) := by norm_num
    exact le_trans h2 h3
  -- ## The scaling `l / 30`
  have hls1 : P.div Format.binary32 (logits.get1 e)
      ((bc (c32 0x403E000000000000)).get1 e) (lsc.get1 e) := hls (e, ())
  have h30fin : ((bc (c32 0x403E000000000000)).get1 e).isFinite = true := c30_isFinite
  have h30v : ((bc (c32 0x403E000000000000)).get1 e).toReal = 30 := c30_val
  have h30ne : ((bc (c32 0x403E000000000000)).get1 e).toReal ≠ 0 := by rw [h30v]; norm_num
  have hmax1 : |(logits.get1 e).toReal / ((bc (c32 0x403E000000000000)).get1 e).toReal| ≤
      (2 : ℝ) ^ (126 : ℕ) := by
    rw [h30v, ← hl]
    exact le_trans hl30 (by norm_num : (2 : ℝ) ^ (119 : ℕ) ≤ (2 : ℝ) ^ (126 : ℕ))
  obtain ⟨hlscfin, hlscerr⟩ := div_full_real hdivfull hls1 hlfin h30fin h30ne hmax1
  set s : ℝ := (lsc.get1 e).toReal with hs
  have hserr : |s - l / 30| ≤ (2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ) := by
    have h := hlscerr
    rw [← hl, h30v] at h
    exact h
  have hsabs : |s| ≤ (2 : ℝ) ^ (120 : ℕ) := by
    have h1 : |s| ≤ |l / 30| + ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ)) := by
      have h2 : s = l / 30 + (s - l / 30) := by ring
      rw [h2]
      exact le_trans (abs_add_le _ _) (add_le_add le_rfl hserr)
    have h5 : (2 : ℝ) ^ (-22 : ℤ) * |l / 30| ≤ (2 : ℝ) ^ (97 : ℕ) := by
      have hm := mul_le_mul_of_nonneg_left hl30
        (show (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) by norm_num)
      have he : (2 : ℝ) ^ (-22 : ℤ) * (2 : ℝ) ^ (119 : ℕ) = (2 : ℝ) ^ (97 : ℕ) := by norm_num
      rw [he] at hm
      exact hm
    have h3 : (2 : ℝ) ^ (-148 : ℤ) ≤ 1 := by norm_num
    have h10 : |l / 30| + ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ)) ≤
        (2 : ℝ) ^ (120 : ℕ) := by
      have h11 : |l / 30| + ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ))
          ≤ (2 : ℝ) ^ (119 : ℕ) + ((2 : ℝ) ^ (97 : ℕ) + 1) := add_le_add hl30 (add_le_add h5 h3)
      have h12 : (2 : ℝ) ^ (119 : ℕ) + ((2 : ℝ) ^ (97 : ℕ) + 1) ≤ (2 : ℝ) ^ (120 : ℕ) :=
        by norm_num
      exact le_trans h11 h12
    exact le_trans h1 h10
  -- ## The doubling `2 · lsc`
  have h2fin : ((bc (c32 0x4000000000000000)).get1 e).isFinite = true := c2_isFinite
  have h2v : ((bc (c32 0x4000000000000000)).get1 e).toReal = 2 := c2_val
  have hmax2 : |((bc (c32 0x4000000000000000)).get1 e).toReal * (lsc.get1 e).toReal| ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [h2v, ← hs]
    have h1 : |2 * s| = 2 * |s| := by rw [abs_mul]; norm_num
    rw [h1]
    have h2 : 2 * |s| ≤ 2 * (2 : ℝ) ^ (120 : ℕ) := mul_le_mul_of_nonneg_left hsabs (by norm_num)
    have h3 : 2 * (2 : ℝ) ^ (120 : ℕ) ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
      rw [maxVal32_eq]; norm_num
    exact le_trans h2 h3
  obtain ⟨hd2fin, hd2err0⟩ := mul_real (x := (bc (c32 0x4000000000000000)).get1 e)
    (y := (lsc.get1 e)) (r := (doubled.get1 e)) (hdoubled (e, ())) h2fin hlscfin hmax2
  set d2 : ℝ := (doubled.get1 e).toReal with hd2
  have hd2err : |d2 - 2 * s| ≤ |2 * s| * u32 + eta32 := by
    have h := hd2err0
    simpa only [h2v, ← hs, ← hd2] using h
  have h2sle : |2 * s| ≤ |l| / 15 * (1 + (2 : ℝ) ^ (-22 : ℤ)) + (2 : ℝ) ^ (-147 : ℤ) := by
    have h3 : |s| ≤ |l / 30| + ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ)) := by
      have h2 : s = l / 30 + (s - l / 30) := by ring
      rw [h2]
      exact le_trans (abs_add_le _ _) (add_le_add le_rfl hserr)
    have h4 : 2 * |s| ≤
        2 * (|l / 30| + ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ))) :=
      mul_le_mul_of_nonneg_left h3 (by norm_num)
    have h5 : (2 : ℝ) * (|l / 30| + ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ)))
        = |l| / 15 * (1 + (2 : ℝ) ^ (-22 : ℤ)) + (2 : ℝ) ^ (-147 : ℤ) := by
      have h6 : |l / 30| = |l| / 30 := by rw [abs_div]; norm_num
      rw [h6]
      have h7 : (2 : ℝ) ^ (-147 : ℤ) = 2 * (2 : ℝ) ^ (-148 : ℤ) := by norm_num
      rw [h7]; ring
    have h8 : |2 * s| = 2 * |s| := by rw [abs_mul]; norm_num
    rw [h8]
    exact le_trans h4 (le_of_eq h5)
  have hd2l : |d2 - l / 15| ≤ (2 : ℝ) ^ (-21 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-145 : ℤ) := by
    have hsplit : d2 - l / 15 = (d2 - 2 * s) + 2 * (s - l / 30) := by ring
    have h1 : |d2 - l / 15| ≤ |d2 - 2 * s| + |2 * (s - l / 30)| := by
      rw [hsplit]; exact abs_add_le _ _
    have hA1 : |d2 - 2 * s| ≤
        (|l| / 15 * (1 + (2 : ℝ) ^ (-22 : ℤ)) + (2 : ℝ) ^ (-147 : ℤ)) * u32 + eta32 := by
      have h := mul_le_mul_of_nonneg_right h2sle (show (0 : ℝ) ≤ u32 from u32_pos.le)
      exact le_trans hd2err (add_le_add h le_rfl)
    have hA2 : |2 * (s - l / 30)| ≤ (2 : ℝ) ^ (-22 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-147 : ℤ) := by
      have h2 : 2 * |s - l / 30| ≤
          2 * ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ)) :=
        mul_le_mul_of_nonneg_left hserr (by norm_num)
      have h3 : 2 * ((2 : ℝ) ^ (-22 : ℤ) * |l / 30| + (2 : ℝ) ^ (-148 : ℤ))
          = (2 : ℝ) ^ (-22 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-147 : ℤ) := by
        have h6 : |l / 30| = |l| / 30 := by rw [abs_div]; norm_num
        rw [h6]
        have h7 : (2 : ℝ) ^ (-147 : ℤ) = 2 * (2 : ℝ) ^ (-148 : ℤ) := by norm_num
        rw [h7]; ring
      have h4 : |2 * (s - l / 30)| = 2 * |s - l / 30| := by rw [abs_mul]; norm_num
      rw [h4]
      exact le_trans h2 (le_of_eq h3)
    have hsum : |d2 - l / 15| ≤
        ((|l| / 15 * (1 + (2 : ℝ) ^ (-22 : ℤ)) + (2 : ℝ) ^ (-147 : ℤ)) * u32 + eta32) +
          ((2 : ℝ) ^ (-22 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-147 : ℤ)) :=
      le_trans h1 (add_le_add hA1 hA2)
    have hfac : ((|l| / 15 * (1 + (2 : ℝ) ^ (-22 : ℤ)) + (2 : ℝ) ^ (-147 : ℤ)) * u32 + eta32) +
          ((2 : ℝ) ^ (-22 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-147 : ℤ))
        = |l| / 15 * ((1 + (2 : ℝ) ^ (-22 : ℤ)) * u32 + (2 : ℝ) ^ (-22 : ℤ)) +
          ((2 : ℝ) ^ (-147 : ℤ) * u32 + eta32 + (2 : ℝ) ^ (-147 : ℤ)) := by ring
    have hcoef : (1 + (2 : ℝ) ^ (-22 : ℤ)) * u32 + (2 : ℝ) ^ (-22 : ℤ) ≤
        (2 : ℝ) ^ (-21 : ℤ) := by
      rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl]; norm_num
    have hconst : (2 : ℝ) ^ (-147 : ℤ) * u32 + eta32 + (2 : ℝ) ^ (-147 : ℤ) ≤
        (2 : ℝ) ^ (-145 : ℤ) := by
      rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl,
        show eta32 = (2 : ℝ) ^ (-150 : ℤ) from rfl]; norm_num
    have hnonneg : (0 : ℝ) ≤ |l| / 15 := by positivity
    calc |d2 - l / 15|
        ≤ ((|l| / 15 * (1 + (2 : ℝ) ^ (-22 : ℤ)) + (2 : ℝ) ^ (-147 : ℤ)) * u32 + eta32) +
          ((2 : ℝ) ^ (-22 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-147 : ℤ)) := hsum
      _ = |l| / 15 * ((1 + (2 : ℝ) ^ (-22 : ℤ)) * u32 + (2 : ℝ) ^ (-22 : ℤ)) +
          ((2 : ℝ) ^ (-147 : ℤ) * u32 + eta32 + (2 : ℝ) ^ (-147 : ℤ)) := hfac
      _ ≤ |l| / 15 * (2 : ℝ) ^ (-21 : ℤ) + (2 : ℝ) ^ (-145 : ℤ) :=
          add_le_add (mul_le_mul_of_nonneg_left hcoef hnonneg) hconst
      _ = (2 : ℝ) ^ (-21 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-145 : ℤ) := by ring
  have hd2Msh : |d2 - M / 15| ≤ L / 15 + (2 : ℝ) ^ (-21 : ℤ) * ((|M| + L) / 15) +
      (2 : ℝ) ^ (-145 : ℤ) := by
    have hsplit : d2 - M / 15 = (d2 - l / 15) + (l - M) / 15 := by ring
    have h1 : |d2 - M / 15| ≤ |d2 - l / 15| + |(l - M) / 15| := by
      rw [hsplit]; exact abs_add_le _ _
    have h2 : |(l - M) / 15| ≤ L / 15 := by
      have h3 : |(l - M) / 15| = |l - M| / 15 := by rw [abs_div]; norm_num
      rw [h3]
      exact div_le_div_of_nonneg_right hlM (by norm_num)
    have h5 : |l| / 15 ≤ (|M| + L) / 15 := div_le_div_of_nonneg_right hlML (by norm_num)
    have h4 : (2 : ℝ) ^ (-21 : ℤ) * |l| / 15 ≤ (2 : ℝ) ^ (-21 : ℤ) * ((|M| + L) / 15) := by
      have h6 := mul_le_mul_of_nonneg_left h5 (show (0 : ℝ) ≤ (2 : ℝ) ^ (-21 : ℤ) by norm_num)
      have he : (2 : ℝ) ^ (-21 : ℤ) * |l| / 15 = (2 : ℝ) ^ (-21 : ℤ) * (|l| / 15) := by ring
      rw [he]; exact h6
    calc |d2 - M / 15| ≤ |d2 - l / 15| + |(l - M) / 15| := h1
      _ ≤ ((2 : ℝ) ^ (-21 : ℤ) * |l| / 15 + (2 : ℝ) ^ (-145 : ℤ)) + L / 15 :=
          add_le_add hd2l h2
      _ ≤ ((2 : ℝ) ^ (-21 : ℤ) * ((|M| + L) / 15) + (2 : ℝ) ^ (-145 : ℤ)) + L / 15 :=
          add_le_add (add_le_add h4 le_rfl) le_rfl
      _ = L / 15 + (2 : ℝ) ^ (-21 : ℤ) * ((|M| + L) / 15) + (2 : ℝ) ^ (-145 : ℤ) := by ring
  have hd2M : |d2 - M / 15| ≤ L / 15 + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * ((|M| + L) / 15) +
      (2 : ℝ) ^ (-140 : ℤ) := by
    have h1 : (0 : ℝ) ≤ (|M| + L) / 15 :=
      div_nonneg (add_nonneg (abs_nonneg M) hL0) (by norm_num)
    have h2 : (2 : ℝ) ^ (-21 : ℤ) * ((|M| + L) / 15) ≤
        (u32 + (2 : ℝ) ^ (-21 : ℤ)) * ((|M| + L) / 15) := by
      have h3 : (2 : ℝ) ^ (-21 : ℤ) ≤ u32 + (2 : ℝ) ^ (-21 : ℤ) := by
        have h4 : (0 : ℝ) ≤ u32 := u32_pos.le
        linarith only [h4]
      exact mul_le_mul_of_nonneg_right h3 h1
    have h5 : (2 : ℝ) ^ (-145 : ℤ) ≤ (2 : ℝ) ^ (-140 : ℤ) := by norm_num
    exact le_trans hd2Msh (add_le_add (add_le_add le_rfl h2) h5)
  have hd2le : d2 ≤ 88.001 :=
    db_le (Al := l) (M := M) (L := L) (D := d2) hlM hd2l hcapML
  have hM15 : 15 * |d2 - M / 15| ≤ (2 : ℝ) ^ (-18 : ℤ) * S + (2 : ℝ) ^ (-20 : ℤ) := by
    have htau := tau_bound (M := M) (L := L) (S := S) hS0 hMS hLle
    have h1 : 15 * |d2 - M / 15| ≤
        15 * (L / 15 + (u32 + (2 : ℝ) ^ (-21 : ℤ)) * ((|M| + L) / 15) +
          (2 : ℝ) ^ (-140 : ℤ)) :=
      mul_le_mul_of_nonneg_left hd2M (by norm_num)
    exact le_trans h1 htau
  -- ## The exponential `exped = exp doubled`
  obtain ⟨eref, heref, hacc⟩ := hexpacc |d2| (abs_nonneg d2)
  have href : eref = Real.exp := by
    have h1 : some eref = (some Real.exp : Option (ℝ → ℝ)) := by
      rw [← heref]; rfl
    exact Option.some.inj h1
  rw [href] at hacc
  have hmemi : (doubled.get1 e).toReal ∈ Set.Icc (-|d2|) |d2| := by
    refine ⟨?_, ?_⟩
    · rw [← hd2]; exact neg_abs_le d2
    · rw [← hd2]; exact le_abs_self d2
  have hroomi : (1 + ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d2|)) *
      |Real.exp (doubled.get1 e).toReal| + (2 : ℝ) ^ (-140 : ℤ) ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [← hd2, abs_of_pos (Real.exp_pos d2)]
    exact room_exp hd2le
  obtain ⟨hEtfin, hEerr0⟩ := hacc (doubled.get1 e) (exped.get1 e) (hexped (e, ()))
    hd2fin hmemi (Set.mem_univ _) hroomi
  set Et : ℝ := (exped.get1 e).toReal with hEt
  have hEerr : |Et - Real.exp d2| ≤
      ((2 : ℝ) ^ (-22 : ℤ) + (2 : ℝ) ^ (-23 : ℤ) * |d2|) * Real.exp d2 +
        (2 : ℝ) ^ (-140 : ℤ) := by
    have h := hEerr0
    rw [← hd2, abs_of_pos (Real.exp_pos d2)] at h
    exact h
  have halow : -((2 : ℝ) ^ (-139 : ℤ)) ≤ Et := exp_low_of_err hd2le hEerr
  have hEtabs : |Et| ≤ (2 : ℝ) ^ (127 : ℕ) := exp_err_abs_le hd2le hEerr
  -- ## `top = exped - 1`, `bottom = exped + 1`
  have hc1fin : ((bc (c32 0x3FF0000000000000)).get1 e).isFinite = true := c1_isFinite
  have hc1v : ((bc (c32 0x3FF0000000000000)).get1 e).toReal = 1 := c1_val
  have hc1g : FloatV.toReal (((bc (c32 0x3FF0000000000000) :
      Tensor [8 - 0] (FloatV Format.binary32))).get (e, ())) = 1 := c1_val
  have h30g : FloatV.toReal (((bc (c32 0x403E000000000000) :
      Tensor [8 - 0] (FloatV Format.binary32))).get (e, ())) = 30 := c30_val
  have hone1 : (2 : ℝ) ^ (127 : ℕ) + 1 ≤ ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [maxVal32_eq]; norm_num
  have hmaxT : |Et - ((bc (c32 0x3FF0000000000000)).get1 e).toReal| ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [hc1v]
    have h1 : |Et - 1| ≤ |Et| + 1 := by
      have h2 := abs_add_le Et (-1)
      rw [abs_neg, abs_one] at h2
      rw [sub_eq_add_neg]
      exact h2
    exact le_trans h1 (le_trans (add_le_add hEtabs le_rfl) hone1)
  obtain ⟨htopfin, hTerr0⟩ := sub_real (htop (e, ())) hEtfin hc1fin hmaxT
  set T : ℝ := (top.get1 e).toReal with hT
  have hTerr : |T - (Et - 1)| ≤ |Et - 1| * u32 + eta32 := by
    have h := hTerr0
    rw [hc1g] at h
    exact h
  have hmaxB : |Et + ((bc (c32 0x3FF0000000000000)).get1 e).toReal| ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [hc1v]
    have h1 : |Et + 1| ≤ |Et| + 1 := by
      have h2 := abs_add_le Et 1
      rw [abs_one] at h2
      exact h2
    exact le_trans h1 (le_trans (add_le_add hEtabs le_rfl) hone1)
  obtain ⟨hbotfin, hBerr0⟩ := add_real (hbottom (e, ())) hEtfin hc1fin hmaxB
  set B : ℝ := (bottom.get1 e).toReal with hB
  have hBerr : |B - (Et + 1)| ≤ |Et + 1| * u32 + eta32 := by
    have h := hBerr0
    rw [hc1g] at h
    exact h
  -- ## The ratio `top / bottom`
  have hratio : |T / B - (Et - 1) / (Et + 1)| ≤ 5 * u32 :=
    ratio_op_bound (a := Et) (T := T) (B := B) halow hTerr hBerr
  have hrabs : |T / B| ≤ 1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ) :=
    ratio_abs_bound (a := Et) (T := T) (B := B) halow hTerr hBerr
  have hBne : B ≠ 0 := by
    have h1 : B - (Et + 1) ≥ -(|Et + 1| * u32 + eta32) := (abs_le.mp hBerr).1
    have hEt1 : (1 / 2 : ℝ) ≤ Et + 1 := by
      have h3 : -(1 / 2 : ℝ) ≤ -((2 : ℝ) ^ (-139 : ℤ)) := by norm_num
      have hA : -(1 / 2 : ℝ) ≤ Et := le_trans h3 halow
      linarith only [hA]
    have hEt1pos : (0 : ℝ) < Et + 1 := by linarith only [hEt1]
    rw [abs_of_pos hEt1pos] at h1
    have hu1 : (0 : ℝ) ≤ 1 - u32 := by
      rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl]; norm_num
    have hstep : (1 / 2 : ℝ) * (1 - u32) ≤ (Et + 1) * (1 - u32) :=
      mul_le_mul_of_nonneg_right hEt1 hu1
    have hB1 : B ≥ (Et + 1) - ((Et + 1) * u32 + eta32) := by linarith only [h1]
    have heq : (Et + 1) - ((Et + 1) * u32 + eta32) = (Et + 1) * (1 - u32) - eta32 := by ring
    rw [heq] at hB1
    have hpos : (0 : ℝ) < (1 / 2 : ℝ) * (1 - u32) - eta32 := by
      rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl,
        show eta32 = (2 : ℝ) ^ (-150 : ℤ) from rfl]; norm_num
    have h2 : (0 : ℝ) < B := by linarith only [hB1, hstep, hpos]
    linarith only [h2]
  have hmaxQ : |T / B| ≤ (2 : ℝ) ^ (126 : ℕ) := by
    have h1 : 1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ) ≤ (2 : ℝ) ^ (126 : ℕ) := by
      rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl]; norm_num
    exact le_trans hrabs h1
  -- ## The quotient `q = top / bottom`
  obtain ⟨hqfin, hqerr0⟩ := div_full_real hdivfull (hq (e, ())) htopfin hbotfin hBne hmaxQ
  set Q : ℝ := (q.get1 e).toReal with hQ
  have hqerr : |Q - T / B| ≤ (2 : ℝ) ^ (-22 : ℤ) * |T / B| + (2 : ℝ) ^ (-148 : ℤ) :=
    hqerr0
  have hQb : |Q| ≤ (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) +
      (2 : ℝ) ^ (-148 : ℤ) := by
    have h1 : |Q| ≤ |T / B| + ((2 : ℝ) ^ (-22 : ℤ) * |T / B| + (2 : ℝ) ^ (-148 : ℤ)) := by
      have h2 : Q = T / B + (Q - T / B) := by ring
      rw [h2]
      exact le_trans (abs_add_le _ _) (add_le_add le_rfl hqerr)
    have h2 : |T / B| * (1 + (2 : ℝ) ^ (-22 : ℤ)) ≤
        (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) := by
      have h3 : (0 : ℝ) ≤ 1 + (2 : ℝ) ^ (-22 : ℤ) := by positivity
      exact mul_le_mul_of_nonneg_right hrabs h3
    calc |Q| ≤ |T / B| + ((2 : ℝ) ^ (-22 : ℤ) * |T / B| + (2 : ℝ) ^ (-148 : ℤ)) := h1
      _ = |T / B| * (1 + (2 : ℝ) ^ (-22 : ℤ)) + (2 : ℝ) ^ (-148 : ℤ) := by ring
      _ ≤ (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) +
          (2 : ℝ) ^ (-148 : ℤ) := add_le_add h2 le_rfl
  have h30Qb : |30 * Q| ≤ 30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) *
      (1 + (2 : ℝ) ^ (-22 : ℤ)) + 30 * (2 : ℝ) ^ (-148 : ℤ) := by
    rw [abs_mul, show |(30 : ℝ)| = 30 by norm_num]
    calc 30 * |Q| ≤ 30 * ((1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) +
        (2 : ℝ) ^ (-148 : ℤ)) := mul_le_mul_of_nonneg_left hQb (by norm_num)
      _ = 30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) +
          30 * (2 : ℝ) ^ (-148 : ℤ) := by ring
  -- ## The capped value `capped = 30 · q`
  have hmaxC : |Q * ((bc (c32 0x403E000000000000)).get1 e).toReal| ≤
      ((Format.binary32.maxVal : ℚ) : ℝ) := by
    rw [h30v]
    have hQle : |Q| ≤ (5 : ℝ) := by
      have h1 : |T / B| ≤ 2 := le_trans hrabs (by
        rw [show u32 = (2 : ℝ) ^ (-24 : ℤ) from rfl]; norm_num)
      have h2 : (2 : ℝ) ^ (-22 : ℤ) * |T / B| ≤ 1 := by
        have h3 := mul_le_mul_of_nonneg_left h1 (show (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) by norm_num)
        have h4 : (2 : ℝ) ^ (-22 : ℤ) * 2 ≤ 1 := by norm_num
        exact le_trans h3 h4
      have h5 : (2 : ℝ) ^ (-148 : ℤ) ≤ 1 := by norm_num
      have h6 : |Q| ≤ |T / B| + ((2 : ℝ) ^ (-22 : ℤ) * |T / B| + (2 : ℝ) ^ (-148 : ℤ)) := by
        have h7 : Q = T / B + (Q - T / B) := by ring
        rw [h7]
        exact le_trans (abs_add_le _ _) (add_le_add le_rfl hqerr)
      linarith only [h6, h1, h2, h5]
    have h7 : |Q * 30| ≤ 5 * 30 := by
      rw [abs_mul, show |(30 : ℝ)| = 30 by norm_num]
      exact mul_le_mul_of_nonneg_right hQle (by norm_num)
    refine le_trans h7 ?_
    rw [maxVal32_eq]; norm_num
  obtain ⟨hCfin, hCerr0⟩ := mul_real (hcapped (e, ())) hqfin h30fin hmaxC
  set C : ℝ := (capped.get1 e).toReal with hC
  have hCerr : |C - 30 * Q| ≤ |30 * Q| * u32 + eta32 := by
    have h := hCerr0
    rw [h30g, mul_comm] at h
    exact h
  -- ## The five error terms
  have hT1 : |C - 30 * Q| ≤
      30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) * u32
        + 30 * (2 : ℝ) ^ (-148 : ℤ) * u32 + eta32 := by
    have h := mul_le_mul_of_nonneg_right h30Qb (show (0 : ℝ) ≤ u32 from u32_pos.le)
    calc |C - 30 * Q| ≤ |30 * Q| * u32 + eta32 := hCerr
      _ ≤ (30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) +
            30 * (2 : ℝ) ^ (-148 : ℤ)) * u32 + eta32 := add_le_add h le_rfl
      _ = 30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) * u32
            + 30 * (2 : ℝ) ^ (-148 : ℤ) * u32 + eta32 := by ring
  have hT2 : |30 * Q - 30 * (T / B)| ≤
      30 * ((2 : ℝ) ^ (-22 : ℤ) * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) +
        (2 : ℝ) ^ (-148 : ℤ)) := by
    have h1 : |30 * Q - 30 * (T / B)| = 30 * |Q - T / B| := by
      rw [show 30 * Q - 30 * (T / B) = 30 * (Q - T / B) by ring, abs_mul]
      norm_num
    rw [h1]
    have h2 := mul_le_mul_of_nonneg_left hqerr (show (0 : ℝ) ≤ 30 by norm_num)
    refine le_trans h2 ?_
    have h3 : (2 : ℝ) ^ (-22 : ℤ) * |T / B| ≤
        (2 : ℝ) ^ (-22 : ℤ) * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) :=
      mul_le_mul_of_nonneg_left hrabs (show (0 : ℝ) ≤ (2 : ℝ) ^ (-22 : ℤ) by norm_num)
    have h4 : 30 * ((2 : ℝ) ^ (-22 : ℤ) * |T / B| + (2 : ℝ) ^ (-148 : ℤ)) ≤
        30 * ((2 : ℝ) ^ (-22 : ℤ) * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) +
          (2 : ℝ) ^ (-148 : ℤ)) :=
      mul_le_mul_of_nonneg_left (add_le_add h3 le_rfl) (by norm_num)
    exact h4
  have hT3 : |30 * (T / B) - 30 * ((Et - 1) / (Et + 1))| ≤ 30 * (5 * u32) := by
    have h1 : |30 * (T / B) - 30 * ((Et - 1) / (Et + 1))| =
        30 * |T / B - (Et - 1) / (Et + 1)| := by
      rw [show 30 * (T / B) - 30 * ((Et - 1) / (Et + 1)) =
        30 * (T / B - (Et - 1) / (Et + 1)) by ring, abs_mul]
      norm_num
    rw [h1]
    exact mul_le_mul_of_nonneg_left hratio (by norm_num)
  have hT4 : |30 * ((Et - 1) / (Et + 1)) -
      30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1))| ≤
      30 * ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-39 : ℤ)) := by
    have hratioe := exp_ratio_err (a := Et) (d := d2) halow hd2le hEerr
    have h1 : |30 * ((Et - 1) / (Et + 1)) -
        30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1))| =
        30 * |(Et - 1) / (Et + 1) - (Real.exp d2 - 1) / (Real.exp d2 + 1)| := by
      rw [show 30 * ((Et - 1) / (Et + 1)) -
        30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1)) =
        30 * ((Et - 1) / (Et + 1) - (Real.exp d2 - 1) / (Real.exp d2 + 1)) by ring, abs_mul]
      norm_num
    rw [h1]
    exact mul_le_mul_of_nonneg_left hratioe (by norm_num)
  have hT5 : |30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1)) - capRef M| ≤
      (2 : ℝ) ^ (-18 : ℤ) * S + (2 : ℝ) ^ (-20 : ℤ) := by
    have hratio_eq : (30 : ℝ) * ((Real.exp d2 - 1) / (Real.exp d2 + 1)) = capRef (15 * d2) := by
      have h2 : Real.tanh ((15 * d2) / 30) = (Real.exp d2 - 1) / (Real.exp d2 + 1) := by
        have h := tanh_eq_ratio ((15 * d2) / 30)
        rw [show 2 * ((15 * d2) / 30) = d2 by ring] at h
        exact h
      show 30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1)) = 30 * Real.tanh ((15 * d2) / 30)
      rw [h2]
    rw [hratio_eq]
    have h1 := capRef_lipschitz (15 * d2) M
    have h2 : |15 * d2 - M| = 15 * |d2 - M / 15| := by
      rw [show (15 : ℝ) * d2 - M = 15 * (d2 - M / 15) by ring, abs_mul]
      norm_num
    rw [h2] at h1
    exact le_trans h1 hM15
  -- ## Assembling
  have hsum := abs_sub_le_sum5 C (30 * Q) (30 * (T / B)) (30 * ((Et - 1) / (Et + 1)))
    (30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1))) (capRef M)
  refine ⟨hCfin, ?_, ?_⟩
  · show |C - capRef M| ≤ (2 : ℝ) ^ (-18 : ℤ) * S + (2 : ℝ) ^ (-14 : ℤ)
    have hmain : |C - capRef M| ≤
        30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) * u32
          + 30 * (2 : ℝ) ^ (-148 : ℤ) * u32 + eta32
          + (30 * ((2 : ℝ) ^ (-22 : ℤ) * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) +
            (2 : ℝ) ^ (-148 : ℤ)))
          + 30 * (5 * u32) + 30 * ((2 : ℝ) ^ (-20 : ℤ) + (2 : ℝ) ^ (-39 : ℤ))
          + ((2 : ℝ) ^ (-18 : ℤ) * S + (2 : ℝ) ^ (-20 : ℤ)) := by
      calc |C - capRef M| ≤ |C - 30 * Q| + |30 * Q - 30 * (T / B)| +
          |30 * (T / B) - 30 * ((Et - 1) / (Et + 1))| +
          |30 * ((Et - 1) / (Et + 1)) - 30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1))| +
          |30 * ((Real.exp d2 - 1) / (Real.exp d2 + 1)) - capRef M| := hsum
        _ ≤ _ := add_le_add (add_le_add (add_le_add (add_le_add hT1 hT2) hT3) hT4) hT5
    refine le_trans hmain ?_
    linarith only [cap_budget]
  · have h1 : |C| ≤ |C - 30 * Q| + |30 * Q| := by
      have h2 : C = (C - 30 * Q) + 30 * Q := by ring
      calc |C| = |(C - 30 * Q) + 30 * Q| := congrArg abs h2
        _ ≤ |C - 30 * Q| + |30 * Q| := abs_add_le _ _
    have h3 : |30 * Q| = 30 * |Q| := by rw [abs_mul]; norm_num
    calc |C| ≤ |C - 30 * Q| + |30 * Q| := h1
      _ ≤ (|30 * Q| * u32 + eta32) + 30 * |Q| := add_le_add hCerr (le_of_eq h3)
      _ ≤ ((30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) +
            30 * (2 : ℝ) ^ (-148 : ℤ)) * u32 + eta32) +
          30 * ((1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) +
            (2 : ℝ) ^ (-148 : ℤ)) :=
          add_le_add (add_le_add (mul_le_mul_of_nonneg_right h30Qb u32_pos.le) le_rfl)
            (mul_le_mul_of_nonneg_left hQb (by norm_num))
      _ = 30 * (1 + 5 * u32 + (2 : ℝ) ^ (-136 : ℤ)) * (1 + (2 : ℝ) ^ (-22 : ℤ)) * (1 + u32)
            + 30 * (2 : ℝ) ^ (-148 : ℤ) * (1 + u32) + eta32 := by ring
      _ ≤ 30 * (1 + (2 : ℝ) ^ (-20 : ℤ)) := cap_abs_budget

end Claims.MoeRouter
