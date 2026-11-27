import Claim
import TVA.Proof

/-!
# The error budget of one output element

An output element is the `bfloat16` rounding `c` of the platform quotient `q` of the `float32` sum
`s` (along a tree of depth `e` over `BT = 2^e` leaves) by the chunk's length `L`. With `S` the exact
sum and `A` the sum of the magnitudes, `mean_err` bounds `|c − S/L|` by the claim's `tol`;
`sum_room`, `div_room`, `cvt_room` show the intermediate values stay in range.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- `(1 + 2^-24)^e ≤ 1 + e·2^-24·(1 + 2^-10)` for `e ≤ 2^10`: the growth of the relative error
along a tree of depth `e`. -/
theorem pow_u32_le : ∀ e : Nat, e ≤ 2 ^ 10 →
    (1 + (2 : ℚ) ^ (-24 : ℤ)) ^ e ≤ 1 + (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ))
  | 0, _ => by norm_num
  | n + 1, h => by
    have ih := pow_u32_le n (by omega)
    have hn : (n : ℚ) ≤ 2 ^ 10 := by exact_mod_cast (by omega : n ≤ 2 ^ 10)
    have hn0 : (0 : ℚ) ≤ n := by positivity
    rw [pow_succ]
    calc (1 + (2 : ℚ) ^ (-24 : ℤ)) ^ n * (1 + 2 ^ (-24 : ℤ))
        ≤ (1 + (n : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ))) * (1 + 2 ^ (-24 : ℤ)) :=
          mul_le_mul_of_nonneg_right ih (by positivity)
      _ ≤ 1 + ((n + 1 : ℕ) : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) := by
          push_cast
          norm_num
          nlinarith

/-- `1 ≤ (1 + 2^-24)^e`. -/
theorem one_le_pow_u32 (e : Nat) : 1 ≤ (1 + (2 : ℚ) ^ (-24 : ℤ)) ^ e :=
  one_le_pow₀ (by norm_num)

/-- The tree's error bound, at depth `e ≤ 16` with fewer than `2^16` nodes: at most
`e·2^-24·(1 + 2^-10)·A + 2^-134·(1 + 2^-10)`. -/
theorem errBound_le {e N : Nat} (he : e ≤ 16) (hN : N < 2 ^ 16) {A : ℚ} (hA : 0 ≤ A) :
    Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) e N A ≤
      (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * A + 2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ)) := by
  unfold Tree.errBound
  have hP := pow_u32_le e (by omega)
  have hP1 := one_le_pow_u32 e
  have he' : (e : ℚ) ≤ 16 := by exact_mod_cast he
  have hN' : (N : ℚ) ≤ 2 ^ 16 := by exact_mod_cast hN.le
  have hN0 : (0 : ℚ) ≤ N := by positivity
  set P := (1 + (2 : ℚ) ^ (-24 : ℤ)) ^ e with hPdef
  have h1 : (P - 1) * A ≤ ((e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ))) * A :=
    mul_le_mul_of_nonneg_right (by linarith) hA
  have hPle : P ≤ 1 + 2 ^ (-10 : ℤ) := by
    have : (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) ≤ 2 ^ (-10 : ℤ) := by
      have h := mul_le_mul_of_nonneg_right he' (show (0 : ℚ) ≤ 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ))
        by positivity)
      norm_num at h ⊢; linarith
    linarith
  have h2 : (2 : ℚ) ^ (-150 : ℤ) * N * P ≤ 2 ^ (-150 : ℤ) * 2 ^ 16 * (1 + 2 ^ (-10 : ℤ)) := by
    have := mul_le_mul hN' hPle (by linarith) (by norm_num)
    nlinarith [show (0 : ℚ) < 2 ^ (-150 : ℤ) by positivity]
  have h3 : (2 : ℚ) ^ (-150 : ℤ) * 2 ^ 16 * (1 + 2 ^ (-10 : ℤ)) = 2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ)) := by
    norm_num
  linarith

/-- **The error budget**: the `bfloat16` rounding `c` of a quotient `q` of the sum `s` by `L ≥ 1`
is within `2^-8·|S/L| + (e + 5)·2^-24·(A/L) + 2^-132` of `S/L`, when `s` is within the tree bound
of `S` (depth `e ≤ 16`, fewer than `2^16` nodes), `q` within the division's bound of `s/L`, and `c`
within one `bfloat16` rounding of `q`. -/
theorem mean_err {S A s q c L : ℚ} {e N : ℕ} (he : e ≤ 16) (hN : N < 2 ^ 16) (hL : 1 ≤ L)
    (hSA : |S| ≤ A) (hs : |s - S| ≤ Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) e N A)
    (hq : |q - s / L| ≤ 2 ^ (-22 : ℤ) * |s / L| + 2 ^ (-148 : ℤ))
    (hc : |c - q| ≤ |q| * 2 ^ (-8 : ℤ) + 2 ^ (-134 : ℤ)) :
    |c - S / L| ≤ 2 ^ (-8 : ℤ) * |S / L| + ((e : ℚ) + 5) * 2 ^ (-24 : ℤ) * (A / L) +
      2 ^ (-132 : ℤ) := by
  have hL0 : 0 < L := by linarith
  have hA : 0 ≤ A := le_trans (abs_nonneg _) hSA
  have hE := errBound_le he hN hA
  set E := Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) e N A with hEdef
  set x := s / L with hx
  set X := S / L with hX
  set a := A / L with ha
  have ha0 : 0 ≤ a := div_nonneg hA hL0.le
  have hXa : |X| ≤ a := by
    rw [hX, ha, abs_div, abs_of_pos hL0]; exact div_le_div_of_nonneg_right hSA hL0.le
  -- `|x − X| = |s − S|/L ≤ E/L`, and `E/L ≤ e·u·c·a + 2^-134·c`
  have hxX : |x - X| ≤ (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * a +
      2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ)) := by
    have h1 : |x - X| = |s - S| / L := by
      rw [hx, hX, ← sub_div, abs_div, abs_of_pos hL0]
    rw [h1]
    have h2 : |s - S| / L ≤ E / L := div_le_div_of_nonneg_right hs hL0.le
    have h3 : E / L ≤ ((e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * A +
        2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ))) / L := div_le_div_of_nonneg_right hE hL0.le
    have h4 : ((e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * A +
        2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ))) / L =
        (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * a + 2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ)) / L := by
      rw [ha, add_div, mul_div_assoc]
    have h5 : 2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ)) / L ≤ 2 ^ (-134 : ℤ) * (1 + 2 ^ (-10 : ℤ)) :=
      div_le_self (by positivity) hL
    linarith
  have hea : (e : ℚ) * a ≤ 16 * a :=
    mul_le_mul_of_nonneg_right (by exact_mod_cast he) ha0
  have hea0 : 0 ≤ (e : ℚ) * a := mul_nonneg (by positivity) ha0
  have t1 : |c - X| ≤ |c - q| + |q - x| + |x - X| := by
    calc |c - X| = |(c - q) + (q - x) + (x - X)| := by ring_nf
      _ ≤ |(c - q) + (q - x)| + |x - X| := abs_add_le _ _
      _ ≤ |c - q| + |q - x| + |x - X| := by linarith [abs_add_le (c - q) (q - x)]
  have t2 : |q| ≤ |x| + |q - x| := by
    calc |q| = |x + (q - x)| := by ring_nf
      _ ≤ |x| + |q - x| := abs_add_le _ _
  have t3 : |x| ≤ |X| + |x - X| := by
    calc |x| = |X + (x - X)| := by ring_nf
      _ ≤ |X| + |x - X| := abs_add_le _ _
  have hxe : (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * a =
      2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * ((e : ℚ) * a) := by ring
  have hte : ((e : ℚ) + 5) * 2 ^ (-24 : ℤ) * a = 2 ^ (-24 : ℤ) * ((e : ℚ) * a) + 5 * 2 ^ (-24 : ℤ) * a := by
    ring
  rw [hxe] at hxX
  rw [hte]
  norm_num at hxX hq hc ⊢
  nlinarith [abs_nonneg (c - q), abs_nonneg (q - x), abs_nonneg (x - X), abs_nonneg q, abs_nonneg x,
    abs_nonneg X]

/-- The sum stays in `float32`'s range: `A + errBound ≤ maxVal` when `A ≤ 2^125`. -/
theorem sum_room {e N : Nat} (he : e ≤ 16) (hN : N < 2 ^ 16) {A : ℚ} (hA : 0 ≤ A) (hA' : A ≤ 2 ^ 125) :
    A + Tree.errBound Format.binary32.u Format.binary32.eta e N A ≤ Format.binary32.maxVal := by
  rw [Format.binary32_u, Format.binary32_eta, Format.binary32_maxVal]
  have h := errBound_le he hN hA
  have he' : (e : ℚ) ≤ 16 := by exact_mod_cast he
  have h1 : (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * A ≤ 16 * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * A :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right he'
      (by positivity)) (by positivity)) hA
  norm_num at h h1 ⊢
  nlinarith

/-- The quotient's operand stays in `Device.DivFull`'s domain: `|s/L| ≤ 2^126`. -/
theorem div_room {e N : Nat} (he : e ≤ 16) (hN : N < 2 ^ 16) {S A s L : ℚ} (hL : 1 ≤ L)
    (hSA : |S| ≤ A) (hA' : A ≤ 2 ^ 125)
    (hs : |s - S| ≤ Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) e N A) :
    |s / L| ≤ 2 ^ 126 ∧ |s| ≤ 2 ^ 126 := by
  have hA : 0 ≤ A := le_trans (abs_nonneg _) hSA
  have h := errBound_le he hN hA
  have he' : (e : ℚ) ≤ 16 := by exact_mod_cast he
  have h1 : (e : ℚ) * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * A ≤ 16 * 2 ^ (-24 : ℤ) * (1 + 2 ^ (-10 : ℤ)) * A :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right he'
      (by positivity)) (by positivity)) hA
  have hs' : |s| ≤ |S| + |s - S| := by
    calc |s| = |S + (s - S)| := by ring_nf
      _ ≤ |S| + |s - S| := abs_add_le _ _
  have hs2 : |s| ≤ 2 ^ 126 := by norm_num at h h1 hs ⊢; nlinarith
  refine ⟨?_, hs2⟩
  have hL0 : 0 < L := by linarith
  rw [abs_div, abs_of_pos hL0]
  calc |s| / L ≤ |s| := div_le_self (abs_nonneg _) hL
    _ ≤ 2 ^ 126 := hs2

/-- The quotient stays in `bfloat16`'s range: `|q| ≤ maxVal` when `|s/L| ≤ 2^126`. -/
theorem cvt_room {x q : ℚ} (hx : |x| ≤ 2 ^ 126)
    (hq : |q - x| ≤ 2 ^ (-22 : ℤ) * |x| + 2 ^ (-148 : ℤ)) : |q| ≤ Format.bfloat16.maxVal := by
  rw [Format.bfloat16_maxVal]
  have t : |q| ≤ |x| + |q - x| := by
    calc |q| = |x + (q - x)| := by ring_nf
      _ ≤ |x| + |q - x| := abs_add_le _ _
  norm_num at hq ⊢
  nlinarith [abs_nonneg x]

end Claims.MeanPoolingFwd
