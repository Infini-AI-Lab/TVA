import TVA.Proof.Int.Basic

/-!
# Integer arithmetic without wrapping

`+ - *` on `IntV` wrap modulo `2^w`, and `//`, `%` truncate toward zero and are undefined for a
zero divisor (`TVA.Base.Int`). A proof about index arithmetic shows that no step wraps and then
computes with `Int` or `Nat`. Every lemma here is in the tracked style of `TVA.Proof.Int`: it takes
the operands' values as hypotheses (`ha : a.toInt = x`) and gives the result's value, so that a
proof can carry `x` and `y` from step to step. None needs `0 < w`.

* `toInt_add`, `toInt_sub`, `toInt_mul`: signed, integer values, the exact result `SFits`.
* `add_nat`, `sub_nat`, `mul_nat`: signed, natural values, one bound on the result (the usual
  index arithmetic); `add_ofNat`, `sub_ofNat`, `mul_ofNat`, `ofNat_mul` with a constant operand;
  `add_nat_unsigned`, `sub_nat_unsigned`, `mul_nat_unsigned` for unsigned types; `I32.add_nat`,
  `I32.sub_nat`, `I32.mul_nat` and the `I64` forms with the bound written out.
* `//` and `%` of a nonnegative value by a positive one (truncation agrees with floor division
  there): `div?_of_nonneg`, `rem?_of_nonneg` (defined, with the value), `toInt_of_div?`,
  `toInt_of_rem?` (the value of a given result, such as a lane of `tl.idiv`), and the natural forms
  `div?_nat`, `rem?_nat`, `toInt_of_div?_nat`, `toInt_of_rem?_nat`; of two constants,
  `div?_ofNat`, `rem?_ofNat` (the result is the constant `n / m`, `n % m`).
* `ofNat_toNat`: an integer rebuilt from its bit pattern read as a natural number is itself.

The lanes of tensors are `TVA.Proof.Int.Lanes`.
-/
namespace TVA

namespace IntV

variable {s : Bool} {w : Nat}

/-! ## `+ - *` of signed integers -/

/-- `a + b` of signed integers of values `x`, `y` has value `x + y` when that fits. -/
theorem toInt_add {a b : IntV true w} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : SFits w (x + y)) : (a + b).toInt = x + y := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · simp only [toInt_width_zero] at ha hb ⊢; omega
  · subst ha hb
    simp only [toInt, ite_true] at h ⊢
    show (a.bits + b.bits).toInt = _
    rw [BitVec.toInt_add, ← toInt_ofInt (w := w) hw h]
    simp [toInt, ofInt, BitVec.toInt_ofInt]

/-- `a - b` of signed integers of values `x`, `y` has value `x - y` when that fits. -/
theorem toInt_sub {a b : IntV true w} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : SFits w (x - y)) : (a - b).toInt = x - y := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · simp only [toInt_width_zero] at ha hb ⊢; omega
  · subst ha hb
    simp only [toInt, ite_true] at h ⊢
    show (a.bits - b.bits).toInt = _
    rw [BitVec.toInt_sub, ← toInt_ofInt (w := w) hw h]
    simp [toInt, ofInt, BitVec.toInt_ofInt]

/-- `a * b` of signed integers of values `x`, `y` has value `x * y` when that fits. -/
theorem toInt_mul {a b : IntV true w} {x y : Int} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : SFits w (x * y)) : (a * b).toInt = x * y := by
  rcases Nat.eq_zero_or_pos w with rfl | hw
  · simp only [toInt_width_zero] at ha hb ⊢; subst ha hb; simp
  · subst ha hb
    simp only [toInt, ite_true] at h ⊢
    show (a.bits * b.bits).toInt = _
    rw [BitVec.toInt_mul, ← toInt_ofInt (w := w) hw h]
    simp [toInt, ofInt, BitVec.toInt_ofInt]

/-! ## Natural values -/

/-- `a + b` of signed integers of natural values `x`, `y` has value `x + y` when
`x + y < 2^(w-1)`. -/
theorem add_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x + y < 2 ^ (w - 1)) : (a + b).toInt = ((x + y : Nat) : Int) := by
  rw [toInt_add ha hb (by exact_mod_cast sfits_natCast h)]; norm_cast

/-- `a * b` of signed integers of natural values `x`, `y` has value `x * y` when
`x * y < 2^(w-1)`. -/
theorem mul_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ (w - 1)) : (a * b).toInt = ((x * y : Nat) : Int) := by
  rw [toInt_mul ha hb (by exact_mod_cast sfits_natCast h)]; norm_cast

/-- `a - b` of signed integers of natural values `y ≤ x` has value `x - y` when `x < 2^(w-1)`. -/
theorem sub_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hxy : y ≤ x) (h : x < 2 ^ (w - 1)) : (a - b).toInt = ((x - y : Nat) : Int) := by
  have hs : SFits w ((x : Int) - y) := by
    have := sfits_natCast (w := w) (n := x - y) (by omega)
    push_cast [hxy] at this
    exact this
  rw [toInt_sub ha hb hs]; push_cast [hxy]; rfl

/-- `a + n` with a constant `n` (`IntV.ofNat n` or a numeral such as `(64 : I32)`) and `a` of
natural value `x` has value `x + n` when `x + n < 2^(w-1)`. -/
theorem add_ofNat {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (h : x + n < 2 ^ (w - 1)) :
    (a + IntV.ofNat n).toInt = ((x + n : Nat) : Int) :=
  add_nat ha (toInt_ofNat (by omega)) h

/-- `a - n` with a constant `n ≤ x` and `a` of natural value `x < 2^(w-1)` has value `x - n`. -/
theorem sub_ofNat {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (hnx : n ≤ x)
    (h : x < 2 ^ (w - 1)) : (a - IntV.ofNat n).toInt = ((x - n : Nat) : Int) :=
  sub_nat ha (toInt_ofNat (by omega)) hnx h

/-- `a * n` with a constant `n` and `a` of natural value `x` has value `x * n` when
`x * n < 2^(w-1)` (`n` itself need not fit when `x = 0`). -/
theorem mul_ofNat {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (h : x * n < 2 ^ (w - 1)) :
    (a * IntV.ofNat n).toInt = ((x * n : Nat) : Int) := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · rw [toInt_mul ha rfl
      (by rw [Nat.cast_zero, zero_mul]; exact sfits_of_nonneg le_rfl (by positivity))]
    simp
  · exact mul_nat ha (toInt_ofNat (lt_of_le_of_lt (Nat.le_mul_of_pos_left n hx) h)) h

/-- `n * a` with a constant `n` and `a` of natural value `x` has value `n * x` when
`n * x < 2^(w-1)`. -/
theorem ofNat_mul {a : IntV true w} {x n : Nat} (ha : a.toInt = x) (h : n * x < 2 ^ (w - 1)) :
    (IntV.ofNat n * a).toInt = ((n * x : Nat) : Int) := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · rw [toInt_mul rfl ha
      (by rw [Nat.cast_zero, mul_zero]; exact sfits_of_nonneg le_rfl (by positivity))]
    simp
  · exact mul_nat (toInt_ofNat (lt_of_le_of_lt (Nat.le_mul_of_pos_right n hx) h)) ha h

/-- `a + b` of unsigned integers of values `x`, `y` has value `x + y` when `x + y < 2^w`. -/
theorem add_nat_unsigned {a b : IntV false w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x + y < 2 ^ w) : (a + b).toInt = ((x + y : Nat) : Int) := by
  simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at ha hb ⊢
  show (a.bits + b.bits).toNat = _
  rw [BitVec.toNat_add, ha, hb, Nat.mod_eq_of_lt h]

/-- `a * b` of unsigned integers of values `x`, `y` has value `x * y` when `x * y < 2^w`. -/
theorem mul_nat_unsigned {a b : IntV false w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ w) : (a * b).toInt = ((x * y : Nat) : Int) := by
  simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at ha hb ⊢
  show (a.bits * b.bits).toNat = _
  rw [BitVec.toNat_mul, ha, hb, Nat.mod_eq_of_lt h]

/-- `a - b` of unsigned integers of values `y ≤ x` has value `x - y`. -/
theorem sub_nat_unsigned {a b : IntV false w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hxy : y ≤ x) : (a - b).toInt = ((x - y : Nat) : Int) := by
  simp only [toInt, Bool.false_eq_true, ite_false, Int.natCast_inj] at ha hb ⊢
  show (a.bits - b.bits).toNat = _
  have hx : x < 2 ^ w := ha ▸ a.bits.isLt
  rw [BitVec.toNat_sub, ha, hb,
    show 2 ^ w - y + x = (x - y) + 2 ^ w by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

/-- `int32` `a + b` of natural values `x`, `y` has value `x + y` when `x + y < 2^31`. -/
theorem I32.add_nat {a b : I32} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x + y < 2 ^ 31) : (a + b).toInt = ((x + y : Nat) : Int) := IntV.add_nat ha hb h

/-- `int32` `a * b` of natural values `x`, `y` has value `x * y` when `x * y < 2^31`. -/
theorem I32.mul_nat {a b : I32} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ 31) : (a * b).toInt = ((x * y : Nat) : Int) := IntV.mul_nat ha hb h

/-- `int32` `a - b` of natural values `y ≤ x < 2^31` has value `x - y`. -/
theorem I32.sub_nat {a b : I32} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y) (hxy : y ≤ x)
    (h : x < 2 ^ 31) : (a - b).toInt = ((x - y : Nat) : Int) := IntV.sub_nat ha hb hxy h

/-- `int64` `a + b` of natural values `x`, `y` has value `x + y` when `x + y < 2^63`. -/
theorem I64.add_nat {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x + y < 2 ^ 63) : (a + b).toInt = ((x + y : Nat) : Int) := IntV.add_nat ha hb h

/-- `int64` `a * b` of natural values `x`, `y` has value `x * y` when `x * y < 2^63`. -/
theorem I64.mul_nat {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (h : x * y < 2 ^ 63) : (a * b).toInt = ((x * y : Nat) : Int) := IntV.mul_nat ha hb h

/-- `int64` `a - b` of natural values `y ≤ x < 2^63` has value `x - y`. -/
theorem I64.sub_nat {a b : I64} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y) (hxy : y ≤ x)
    (h : x < 2 ^ 63) : (a - b).toInt = ((x - y : Nat) : Int) := IntV.sub_nat ha hb hxy h

/-! ## `//` and `%` -/

/-- A positive signed divisor is neither `0` nor `-1`, the two that make `//` undefined. -/
private theorem bits_ne_of_pos {b : IntV true w} (hb : 0 < b.toInt) :
    b.bits ≠ 0 ∧ b.bits ≠ -1 := by
  simp only [toInt, ite_true] at hb
  refine ⟨fun h => by rw [h] at hb; simp at hb, fun h => ?_⟩
  rw [h, show (-1 : BitVec w) = BitVec.allOnes w from BitVec.neg_one_eq_allOnes,
    BitVec.toInt_allOnes] at hb
  split_ifs at hb <;> omega

/-- `a // b` of a nonnegative `a` by a positive `b` is defined and has value
`a.toInt / b.toInt` (truncation and floor division agree there). -/
theorem div?_of_nonneg {a b : IntV true w} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt) :
    ∃ q, a.div? b = some q ∧ q.toInt = a.toInt / b.toInt := by
  obtain ⟨hb0, hb1⟩ := bits_ne_of_pos hb
  refine ⟨⟨a.bits.sdiv b.bits⟩, by simp only [div?, hb0, hb1, and_false, ↓reduceIte], ?_⟩
  simp only [toInt, ite_true] at ha ⊢
  rw [BitVec.toInt_sdiv_of_ne_or_ne _ _ (Or.inr hb1), Int.tdiv_eq_ediv_of_nonneg ha]

/-- `a % b` of a nonnegative `a` by a positive `b` is defined and has value
`a.toInt % b.toInt`. -/
theorem rem?_of_nonneg {a b : IntV true w} (ha : 0 ≤ a.toInt) (hb : 0 < b.toInt) :
    ∃ q, a.rem? b = some q ∧ q.toInt = a.toInt % b.toInt := by
  obtain ⟨hb0, hb1⟩ := bits_ne_of_pos hb
  refine ⟨⟨a.bits.srem b.bits⟩, by simp only [rem?, hb0, hb1, and_false, ↓reduceIte], ?_⟩
  simp only [toInt, ite_true] at ha ⊢
  rw [BitVec.toInt_srem, Int.tmod_eq_emod_of_nonneg ha]

/-- A given result `q` of `a // b` (`a.div? b = some q`, a lane of `tl.idiv`), `a ≥ 0`, `b > 0`,
has value `a.toInt / b.toInt`. -/
theorem toInt_of_div? {a b q : IntV true w} (h : a.div? b = some q) (ha : 0 ≤ a.toInt)
    (hb : 0 < b.toInt) : q.toInt = a.toInt / b.toInt := by
  obtain ⟨q', hq', hv⟩ := div?_of_nonneg ha hb
  rw [hq'] at h
  obtain rfl := Option.some.inj h
  exact hv

/-- A given result `q` of `a % b` (a lane of `tl.irem`), `a ≥ 0`, `b > 0`, has value
`a.toInt % b.toInt`. -/
theorem toInt_of_rem? {a b q : IntV true w} (h : a.rem? b = some q) (ha : 0 ≤ a.toInt)
    (hb : 0 < b.toInt) : q.toInt = a.toInt % b.toInt := by
  obtain ⟨q', hq', hv⟩ := rem?_of_nonneg ha hb
  rw [hq'] at h
  obtain rfl := Option.some.inj h
  exact hv

/-- `a // b` of natural values `x`, `y > 0` is defined and has value `x / y`. -/
theorem div?_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hy : 0 < y) : ∃ q, a.div? b = some q ∧ q.toInt = ((x / y : Nat) : Int) := by
  obtain ⟨q, hq, hv⟩ := div?_of_nonneg (a := a) (b := b) (by rw [ha]; positivity)
    (by rw [hb]; exact_mod_cast hy)
  exact ⟨q, hq, by rw [hv, ha, hb]; norm_cast⟩

/-- `a % b` of natural values `x`, `y > 0` is defined and has value `x % y`. -/
theorem rem?_nat {a b : IntV true w} {x y : Nat} (ha : a.toInt = x) (hb : b.toInt = y)
    (hy : 0 < y) : ∃ q, a.rem? b = some q ∧ q.toInt = ((x % y : Nat) : Int) := by
  obtain ⟨q, hq, hv⟩ := rem?_of_nonneg (a := a) (b := b) (by rw [ha]; positivity)
    (by rw [hb]; exact_mod_cast hy)
  exact ⟨q, hq, by rw [hv, ha, hb]; norm_cast⟩

/-- A given result `q` of `a // b` of natural values `x`, `y > 0` has value `x / y`. -/
theorem toInt_of_div?_nat {a b q : IntV true w} {x y : Nat} (h : a.div? b = some q)
    (ha : a.toInt = x) (hb : b.toInt = y) (hy : 0 < y) : q.toInt = ((x / y : Nat) : Int) := by
  rw [toInt_of_div? h (by rw [ha]; positivity) (by rw [hb]; exact_mod_cast hy), ha, hb]
  norm_cast

/-- A given result `q` of `a % b` of natural values `x`, `y > 0` has value `x % y`. -/
theorem toInt_of_rem?_nat {a b q : IntV true w} {x y : Nat} (h : a.rem? b = some q)
    (ha : a.toInt = x) (hb : b.toInt = y) (hy : 0 < y) : q.toInt = ((x % y : Nat) : Int) := by
  rw [toInt_of_rem? h (by rw [ha]; positivity) (by rw [hb]; exact_mod_cast hy), ha, hb]
  norm_cast

/-- Use for `//` of two constants (`IntV.ofNat n // IntV.ofNat m`, such as `N // BLOCK` with
constant shapes), `m > 0`, both in range: it is the constant `n / m`. -/
theorem div?_ofNat {n m : Nat} (hn : n < 2 ^ (w - 1)) (hm0 : 0 < m) (hm : m < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).div? (IntV.ofNat m) = some (IntV.ofNat (n / m)) := by
  obtain ⟨q, hq, hv⟩ := div?_nat (toInt_ofNat hn) (toInt_ofNat hm) hm0
  rw [hq, ext_toInt (hv.trans (toInt_ofNat (lt_of_le_of_lt (Nat.div_le_self n m) hn)).symm)]

/-- Use for `%` of two constants (`IntV.ofNat n % IntV.ofNat m`), `m > 0`, both in range: it is
the constant `n % m`. -/
theorem rem?_ofNat {n m : Nat} (hn : n < 2 ^ (w - 1)) (hm0 : 0 < m) (hm : m < 2 ^ (w - 1)) :
    (IntV.ofNat n : IntV true w).rem? (IntV.ofNat m) = some (IntV.ofNat (n % m)) := by
  obtain ⟨q, hq, hv⟩ := rem?_nat (toInt_ofNat hn) (toInt_ofNat hm) hm0
  rw [hq, ext_toInt (hv.trans (toInt_ofNat (lt_trans (Nat.mod_lt n hm0) hm)).symm)]

/-! ## Bit patterns -/

/-- An integer rebuilt from its bit pattern read as a natural number (`IntV.ofNat x.bits.toNat`,
as `DType.ofBits` does with the bytes of a load) is itself. -/
theorem ofNat_toNat (x : IntV s w) : (IntV.ofNat x.bits.toNat : IntV s w) = x := by
  cases x; simp [IntV.ofNat]

end IntV

end TVA
