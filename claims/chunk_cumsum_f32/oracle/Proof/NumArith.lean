import Claim
import TVA.Proof
import Proof.Bounds

/-!
# Relaxing `Tree.errBound` for `float32`

With `u = 2^-24` and `η = 2^-150`, and at most `2^20` leaves, the error bound of a reduction
tree is at most `(m + 1)·2^-23·A + 2^-128`: this is the tolerance the claim states.
-/

namespace Claims.ChunkCumsum

open TVA

/-- The relaxation of `Tree.errBound` used by the claim: `(1+u)^m - 1 ≤ m·2^-23` and the
subnormal term is below `2^-128`. -/
theorem errBound_relax (A : ℚ) (hA : 0 ≤ A) (m : Nat) (hm : m ≤ 2 ^ 20) :
    Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) m m A ≤
      ((m : ℚ) + 1) * 2 ^ (-23 : ℤ) * A + 2 ^ (-128 : ℤ) := by
  unfold Tree.errBound
  set u : ℚ := (2 : ℚ) ^ (-24 : ℤ) with hu
  set w : ℚ := (2 : ℚ) ^ (-23 : ℤ) with hw
  set e : ℚ := (2 : ℚ) ^ (-150 : ℤ) with he
  set f : ℚ := (2 : ℚ) ^ (-128 : ℤ) with hf
  have hmq : (m : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hm
  have h20 : (2 : ℚ) ^ 20 = 1048576 := by norm_num
  have hu0 : (0 : ℚ) ≤ u := by rw [hu]; positivity
  have hw0 : (0 : ℚ) ≤ w := by rw [hw]; positivity
  have he0 : (0 : ℚ) ≤ e := by rw [he]; positivity
  have hu24 : u = 1 / 16777216 := by rw [hu]; norm_num
  have hw23 : w = 2 * u := by rw [hu, hw]; norm_num
  have hmu : 2 * (m : ℚ) * u ≤ 1 := by
    rw [hu24]; nlinarith [hmq, h20]
  have hpow : (1 + u) ^ m ≤ 1 + 2 * (m : ℚ) * u := one_add_pow_le u hu0 m hmu
  have hmul : 2 * (m : ℚ) * u = (m : ℚ) * w := by
    calc 2 * (m : ℚ) * u = (m : ℚ) * (2 * u) := by ring
      _ = (m : ℚ) * w := by rw [hw23]
  have h1 : (1 + u) ^ m - 1 ≤ (m : ℚ) * w := by
    calc (1 + u) ^ m - 1 ≤ (1 + 2 * (m : ℚ) * u) - 1 := sub_le_sub_right hpow 1
      _ = 2 * (m : ℚ) * u := by ring
      _ = (m : ℚ) * w := hmul
  have hterm1 : ((1 + u) ^ m - 1) * A ≤ (m : ℚ) * w * A :=
    mul_le_mul_of_nonneg_right h1 hA
  have hmono : (m : ℚ) * w * A ≤ ((m : ℚ) + 1) * w * A := by
    apply mul_le_mul_of_nonneg_right _ hA
    exact mul_le_mul_of_nonneg_right (by linarith) hw0
  have hpow9 : (1 + u) ^ m ≤ 9 / 8 := by
    have h2mu : 2 * (m : ℚ) * u ≤ 1 / 8 := by
      rw [hu24]; nlinarith [hmq, h20]
    calc (1 + u) ^ m ≤ 1 + 2 * (m : ℚ) * u := hpow
      _ ≤ 1 + 1 / 8 := by linarith [h2mu]
      _ = 9 / 8 := by norm_num
  have hterm2 : e * (m : ℚ) * (1 + u) ^ m ≤ f := by
    have hpownn : (0 : ℚ) ≤ (1 + u) ^ m := by positivity
    have he20 : (0 : ℚ) ≤ e * (2 : ℚ) ^ 20 := mul_nonneg he0 (by positivity)
    have hs1 : e * (m : ℚ) * (1 + u) ^ m ≤ e * (2 : ℚ) ^ 20 * (1 + u) ^ m := by
      apply mul_le_mul_of_nonneg_right _ hpownn
      exact mul_le_mul_of_nonneg_left hmq he0
    have hs2 : e * (2 : ℚ) ^ 20 * (1 + u) ^ m ≤ e * (2 : ℚ) ^ 20 * (9 / 8) :=
      mul_le_mul_of_nonneg_left hpow9 he20
    have hfin : e * (2 : ℚ) ^ 20 * (9 / 8) ≤ f := by
      rw [he, hf]; norm_num
    exact le_trans hs1 (le_trans hs2 hfin)
  linarith [hterm1, hmono, hterm2]

/-- With one more rounding (the scale multiplication) the same relaxation holds. -/
theorem errBound_relax_scale (A : ℚ) (hA : 0 ≤ A) (m : Nat) (hm : m ≤ 2 ^ 20) :
    2 ^ (-24 : ℤ) * A +
        (1 + 2 ^ (-24 : ℤ)) * Tree.errBound (2 ^ (-24 : ℤ)) (2 ^ (-150 : ℤ)) m m A +
        2 ^ (-150 : ℤ) ≤ ((m : ℚ) + 1) * 2 ^ (-23 : ℤ) * A + 2 ^ (-128 : ℤ) := by
  unfold Tree.errBound
  set u : ℚ := (2 : ℚ) ^ (-24 : ℤ) with hu
  set w : ℚ := (2 : ℚ) ^ (-23 : ℤ) with hw
  set e : ℚ := (2 : ℚ) ^ (-150 : ℤ) with he
  set f : ℚ := (2 : ℚ) ^ (-128 : ℤ) with hf
  have hmq : (m : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hm
  have h20 : (2 : ℚ) ^ 20 = 1048576 := by norm_num
  have hu0 : (0 : ℚ) ≤ u := by rw [hu]; positivity
  have he0 : (0 : ℚ) ≤ e := by rw [he]; positivity
  have hu24 : u = 1 / 16777216 := by rw [hu]; norm_num
  have hw23 : w = 2 * u := by rw [hu, hw]; norm_num
  -- collapse `u·A + (1+u)·errBound` into one exact `((1+u)^(m+1) - 1)·A` plus a subnormal term
  have hexp : u * A + (1 + u) * (((1 + u) ^ m - 1) * A + e * (m : ℚ) * (1 + u) ^ m) + e
      = ((1 + u) ^ (m + 1) - 1) * A + e * ((1 + u) * (m : ℚ) * (1 + u) ^ m + 1) := by
    rw [pow_succ]
    ring
  rw [hexp]
  have hmu1 : 2 * ((m : ℚ) + 1) * u ≤ 1 := by
    rw [hu24]; nlinarith [hmq, h20]
  have hmu1' : 2 * ((m + 1 : Nat) : ℚ) * u ≤ 1 := by
    rw [Nat.cast_add, Nat.cast_one]; exact hmu1
  have hpow1 : (1 + u) ^ (m + 1) ≤ 1 + 2 * ((m + 1 : Nat) : ℚ) * u :=
    one_add_pow_le u hu0 (m + 1) hmu1'
  rw [Nat.cast_add, Nat.cast_one] at hpow1
  have hmul1 : 2 * ((m : ℚ) + 1) * u = ((m : ℚ) + 1) * w := by
    calc 2 * ((m : ℚ) + 1) * u = ((m : ℚ) + 1) * (2 * u) := by ring
      _ = ((m : ℚ) + 1) * w := by rw [hw23]
  have hA_bound : ((1 + u) ^ (m + 1) - 1) * A ≤ ((m : ℚ) + 1) * w * A := by
    apply mul_le_mul_of_nonneg_right _ hA
    linarith [hpow1, hmul1]
  have h9u : 1 + u ≤ 9 / 8 := by rw [hu]; norm_num
  have hpow9 : (1 + u) ^ m ≤ 9 / 8 := by
    have hmu : 2 * (m : ℚ) * u ≤ 1 := by
      rw [hu24]; nlinarith [hmq, h20]
    have hpow : (1 + u) ^ m ≤ 1 + 2 * (m : ℚ) * u := one_add_pow_le u hu0 m hmu
    have h2mu : 2 * (m : ℚ) * u ≤ 1 / 8 := by
      rw [hu24]; nlinarith [hmq, h20]
    calc (1 + u) ^ m ≤ 1 + 2 * (m : ℚ) * u := hpow
      _ ≤ 1 + 1 / 8 := by linarith [h2mu]
      _ = 9 / 8 := by norm_num
  have hprod : (1 + u) * (m : ℚ) * (1 + u) ^ m ≤ (9 / 8) * (2 : ℚ) ^ 20 * (9 / 8) := by
    have hpownn : (0 : ℚ) ≤ (1 + u) ^ m := by positivity
    have hstep : (1 + u) * (m : ℚ) ≤ (9 / 8) * (2 : ℚ) ^ 20 :=
      mul_le_mul h9u hmq (by positivity) (by norm_num)
    calc (1 + u) * (m : ℚ) * (1 + u) ^ m
        ≤ ((9 / 8) * (2 : ℚ) ^ 20) * (9 / 8) := mul_le_mul hstep hpow9 hpownn (by positivity)
      _ = (9 / 8) * (2 : ℚ) ^ 20 * (9 / 8) := by ring
  have heta_bound : e * ((1 + u) * (m : ℚ) * (1 + u) ^ m + 1) ≤ f := by
    have hstep : e * ((1 + u) * (m : ℚ) * (1 + u) ^ m + 1) ≤
        e * ((9 / 8) * (2 : ℚ) ^ 20 * (9 / 8) + 1) :=
      mul_le_mul_of_nonneg_left (by linarith [hprod]) he0
    have hfin : e * ((9 / 8) * (2 : ℚ) ^ 20 * (9 / 8) + 1) ≤ f := by
      rw [he, hf]; norm_num
    exact le_trans hstep hfin
  linarith [hA_bound, heta_bound]

end Claims.ChunkCumsum
