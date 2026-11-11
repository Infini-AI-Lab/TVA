import Proof.Setup

/-!
# `scaled_mm_int8`: bounds on the `int8` values and the exact dot

The `int8` operands are within `128` of `0`, so one product is within `16384`, and an exact dot
over `K < 2^17` terms is within `2^31`: the `int32` accumulator never wraps.
-/

namespace Claims.ScaledMmInt8
open TVA

/-- An `int8` value is at most `128` in magnitude. -/
theorem I8.abs_toInt_le (x : I8) : |x.toInt| ≤ 128 := by
  have h := IntV.sfits_toInt x
  simp only [IntV.SFits] at h
  rw [abs_le]
  constructor <;> omega

/-- A product of two `int8` values is at most `16384` in magnitude. -/
theorem I8.abs_mul_le (x y : I8) : |x.toInt * y.toInt| ≤ 16384 := by
  rw [abs_mul]
  calc |x.toInt| * |y.toInt| ≤ 128 * 128 :=
        mul_le_mul (I8.abs_toInt_le x) (I8.abs_toInt_le y) (abs_nonneg _) (by norm_num)
    _ = 16384 := by norm_num

/-- The exact dot of two rows of `int8` values is at most `2^31` in magnitude. -/
theorem dotSum_bound {d : Inst} (hd : d.ok) (va vb : Nat → I8) :
    |((dotSum va vb d.K : ℤ) : ℚ)| ≤ 2 ^ 31 := by
  have hterm : ∀ k ∈ Finset.range d.K, |(va k).toInt * (vb k).toInt| ≤ (16384 : ℤ) :=
    fun k _ => I8.abs_mul_le (va k) (vb k)
  have h1 : |dotSum va vb d.K| ≤ (d.K : ℤ) * 16384 := by
    rw [dotSum]
    calc |∑ k ∈ Finset.range d.K, (va k).toInt * (vb k).toInt|
        ≤ ∑ k ∈ Finset.range d.K, |(va k).toInt * (vb k).toInt| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ k ∈ Finset.range d.K, (16384 : ℤ) := Finset.sum_le_sum hterm
      _ = (d.K : ℤ) * 16384 := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  rw [← Int.cast_abs]
  have h2 : ((|dotSum va vb d.K| : ℤ) : ℚ) ≤ (((d.K : ℤ) * 16384 : ℤ) : ℚ) := by
    exact_mod_cast h1
  have hk : (d.K : ℚ) < 2 ^ 17 := by
    have := hd.K_lt
    exact_mod_cast this
  have h3 : (((d.K : ℤ) * 16384 : ℤ) : ℚ) ≤ 2 ^ 31 := by
    push_cast
    nlinarith
  linarith

/-- The value a read of `A` returns is `rA`'s. -/
theorem rA_eq {m : Mem} {d : Inst} {i k : Nat} {v : I8}
    (h : m.read .i8 (d.a.ptr ((i * d.stride_am + k * d.stride_ak : Nat) : ℤ)).addr = some v) :
    rA m d i k = v.toInt := by
  simp only [rA, rI8, Buf.ptr]
  simp only [Buf.ptr] at h
  rw [h]
  rfl

/-- The value a read of `B` returns is `rB`'s. -/
theorem rB_eq {m : Mem} {d : Inst} {k j : Nat} {v : I8}
    (h : m.read .i8 (d.b.ptr ((k * d.stride_bk + j * d.stride_bn : Nat) : ℤ)).addr = some v) :
    rB m d k j = v.toInt := by
  simp only [rB, rI8, Buf.ptr]
  simp only [Buf.ptr] at h
  rw [h]
  rfl

end Claims.ScaledMmInt8
