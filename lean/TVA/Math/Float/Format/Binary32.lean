import TVA.Math.Float.Format.Round

/-!
# Bridge: `TVA.Math.Float.F32` is the binary32 instance of `TVA.Math.Float.Format`

For every `x : F32.Bits`, the fields, the classification, the value, the rounding and the
operations of `TVA.Math.Float.F32` equal those of `Format.binary32` on `x.toNat`
(`*_binary32`), with rounding mode `rn` and the canonical NaN `0x7fffffff`. The IEEE field
decoding and comparison of `TVA.Math.Float` (`f32IsNaN`, `f32Compare`, and the bf16 and f16
ones) are the corresponding `Format` functions of `binary32`, `bfloat16` and `binary16`.

Proof module (Mathlib); part of `TVA.Math.Float.Format.Bridge`.
-/
set_option autoImplicit false

namespace TVA.Math.Float

open TVA.Math.Float.Format

/-! ## Constants of the named formats -/

namespace Format

@[simp] theorem binary32_e : binary32.e = 8 := rfl
@[simp] theorem binary32_m : binary32.m = 23 := rfl
@[simp] theorem binary32_special : binary32.special = .ieee := rfl
@[simp] theorem binary32_bias : binary32.bias = 127 := rfl
@[simp] theorem binary32_expMax : binary32.expMax = 255 := rfl
@[simp] theorem binary32_emin : binary32.emin = -126 := rfl
@[simp] theorem binary32_emax : binary32.emax = 127 := rfl
@[simp] theorem binary32_signBit : binary32.signBit = 2 ^ 31 := rfl
@[simp] theorem binary32_infMag : binary32.infMag = 0x7f800000 := rfl
@[simp] theorem binary32_nanMag : binary32.nanMag = 0x7fffffff := rfl
@[simp] theorem binary32_negZero : binary32.negZero = 2 ^ 31 := rfl

theorem zeroSumNeg_rn (a b : Bool) : zeroSumNeg .rn a b = (a && b) := by cases a <;> cases b <;> rfl
theorem cancelNeg_rn : cancelNeg .rn = false := rfl

/-- Under `rn` the sign passed to `roundMag` is irrelevant. -/
theorem roundMag_rn_neg (f : Format) (sat neg : Bool) (q : ℚ) :
    f.roundMag .rn sat neg q = f.roundMag .rn sat false q := rfl

end Format

/-! ## IEEE field decoding and comparison -/

theorem f32IsNaN_binary32 (b : ℕ) : f32IsNaN b = binary32.isNaN b := rfl
theorem bf16IsNaN_bfloat16 (b : ℕ) : bf16IsNaN b = bfloat16.isNaN b := rfl
theorem f16IsNaN_binary16 (b : ℕ) : f16IsNaN b = binary16.isNaN b := rfl
theorem f32Compare_binary32 : f32Compare = binary32.compare := rfl
theorem bf16Compare_bfloat16 : bf16Compare = bfloat16.compare := rfl
theorem f16Compare_binary16 : f16Compare = binary16.compare := rfl

namespace F32

/-! ## Fields, classification and value -/

theorem expField_binary32 (x : Bits) : expField x = binary32.expField x.toNat := rfl
theorem mantField_binary32 (x : Bits) : mantField x = binary32.fracField x.toNat := rfl
theorem isNaN_binary32 (x : Bits) : isNaN x = binary32.isNaN x.toNat := rfl
theorem isInf_binary32 (x : Bits) : isInf x = binary32.isInf x.toNat := rfl
theorem isZero_binary32 (x : Bits) : isZero x = binary32.isZero x.toNat := rfl
theorem magRat_binary32 (x : Bits) : magRat x = binary32.magRat x.toNat := rfl
theorem log2Floor_format : log2Floor = Format.log2Floor := rfl

theorem sign_binary32 (x : Bits) : sign x = binary32.sign x.toNat := by
  have hx := x.isLt
  unfold sign Format.sign
  rw [Nat.testBit_eq_decide_div_mod_eq, decide_eq_decide]
  simp only [binary32_e, binary32_m, Nat.reduceAdd]
  omega

theorem isFinite_binary32 (x : Bits) : isFinite x = binary32.isFinite x.toNat := by
  unfold isFinite Format.isFinite Format.isNaN Format.isInf
  rw [expField_binary32]
  simp only [binary32_special]
  cases binary32.expField x.toNat == 255 <;> cases h : binary32.fracField x.toNat == 0 <;> simp_all [bne]

theorem toRat_binary32 (x : Bits) : toRat x = binary32.toRat x.toNat := by
  unfold toRat Format.toRat; rw [sign_binary32, magRat_binary32]

/-! ## Rounding -/

theorem withSign_binary32 (s : Bool) (n : ℕ) :
    withSign s (BitVec.ofNat 32 n) = BitVec.ofNat 32 (binary32.withSign s n) := by
  apply BitVec.eq_of_toNat_eq
  unfold withSign Format.withSign
  cases s
  · simp only [Bool.false_eq_true, ↓reduceIte, BitVec.toNat_and, BitVec.toNat_ofNat, binary32_signBit]
    rw [show 2147483647 % 2 ^ 32 = 2 ^ 31 - 1 by norm_num, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_mod_of_dvd _ (by norm_num), Nat.mod_eq_of_lt (lt_of_lt_of_le (Nat.mod_lt _ (by norm_num)) (by norm_num))]
  · simp only [↓reduceIte, BitVec.toNat_or, BitVec.toNat_ofNat, binary32_signBit]
    rw [show 2147483648 % 2 ^ 32 = 2 ^ 31 by norm_num]
    apply Nat.eq_of_testBit_eq
    intro i
    simp only [Nat.testBit_or, Nat.testBit_mod_two_pow, Nat.testBit_two_pow]
    by_cases hi : i < 32
    · simp [hi]
    · have : (31 = i) = False := by simp; omega
      simp [hi, this]

theorem over_binary32 (m : ℕ) (e' : ℤ) : binary32.over m e' = decide (e' > 127) := rfl
theorem overflow_binary32 : binary32.overflow .rn false false = 0x7f800000 := rfl

theorem encode_binary32 (m : ℕ) (e' : ℤ) :
    encode m e' = BitVec.ofNat 32 (binary32.encode .rn false false m e') := by
  unfold encode Format.encode
  rw [show (2 : ℕ) ^ (binary32.m + 1) = 2 ^ 24 from rfl]
  by_cases hc : m = 2 ^ 24
  · subst hc
    simp only [beq_self_eq_true, ↓reduceIte, over_binary32, overflow_binary32]
    by_cases he : e' + 1 > 127
    · simp only [he, decide_true, ↓reduceIte]; rfl
    · simp only [he, decide_false, Bool.false_eq_true, ↓reduceIte]; rfl
  · have hb : (m == 2 ^ 24) = false := by simpa using hc
    simp only [hb, Bool.false_eq_true, ↓reduceIte, over_binary32, overflow_binary32]
    by_cases he : e' > 127
    · simp only [he, decide_true, ↓reduceIte]; rfl
    · simp only [he, decide_false, Bool.false_eq_true, ↓reduceIte]
      by_cases hm : m < 2 ^ 23
      · simp only [hm, ↓reduceIte, binary32_m]
      · simp only [hm, ↓reduceIte, binary32_m]; rfl

theorem roundMag_binary32 (q : ℚ) : roundMag q = BitVec.ofNat 32 (binary32.roundMag .rn false false q) := by
  rw [roundMag_eq, Format.roundMag_eq]
  have hexp : binary32.expOf q = expOf q := rfl
  have hsc : binary32.scaled q = scaled q := by
    unfold Format.scaled scaled; rw [hexp]; rfl
  rw [hexp, hsc]
  split_ifs
  · rfl
  · rcases scaled q with ⟨num, den⟩
    exact encode_binary32 _ _

theorem roundRat_binary32 (q : ℚ) : roundRat q = BitVec.ofNat 32 (binary32.roundRat .rn false false q) := by
  unfold roundRat Format.roundRat
  by_cases hq : q == 0
  · simp only [hq, ↓reduceIte, Bool.false_eq_true]; rfl
  · simp only [hq, Bool.false_eq_true, ↓reduceIte]
    rw [roundMag_binary32, withSign_binary32, Format.roundMag_rn_neg _ _ (decide (q < 0))]

/-! ## Operations -/

theorem ofNat_toNat (x : Bits) : BitVec.ofNat 32 x.toNat = x := by
  apply BitVec.eq_of_toNat_eq; simp

theorem negate_toNat (x : Bits) : (negate x).toNat = binary32.negate x.toNat := by
  unfold negate Format.negate
  simp only [BitVec.toNat_xor, binary32_signBit]
  rfl

theorem add_binary32 (x y : Bits) :
    add x y = BitVec.ofNat 32 (binary32.add .rn 0x7fffffff x.toNat y.toNat) := by
  unfold add Format.add
  simp only [isNaN_binary32, isInf_binary32, isZero_binary32, sign_binary32, toRat_binary32, roundRat_binary32,
    Format.zeroSumNeg_rn, Format.cancelNeg_rn, apply_ite (BitVec.ofNat 32), ofNat_toNat]
  rfl

theorem sub_binary32 (x y : Bits) :
    sub x y = BitVec.ofNat 32 (binary32.sub .rn 0x7fffffff x.toNat y.toNat) := by
  unfold sub Format.sub
  rw [add_binary32, negate_toNat]

theorem mul_binary32 (x y : Bits) :
    mul x y = BitVec.ofNat 32 (binary32.mul .rn 0x7fffffff x.toNat y.toNat) := by
  unfold mul Format.mul
  simp only [isNaN_binary32, isInf_binary32, isZero_binary32, sign_binary32, magRat_binary32, roundMag_binary32,
    apply_ite (BitVec.ofNat 32), ← withSign_binary32]
  rfl

theorem fma_binary32 (x y z : Bits) :
    fma x y z = BitVec.ofNat 32 (binary32.fma .rn 0x7fffffff x.toNat y.toNat z.toNat) := by
  unfold fma Format.fma
  simp only [isNaN_binary32, isInf_binary32, isZero_binary32, sign_binary32, toRat_binary32, roundRat_binary32,
    Format.zeroSumNeg_rn, Format.cancelNeg_rn, apply_ite (BitVec.ofNat 32), ofNat_toNat, ← withSign_binary32]
  rfl

theorem rcp_binary32 (x : Bits) :
    rcp x = BitVec.ofNat 32 (binary32.rcp .rn 0x7fffffff x.toNat) := by
  unfold rcp Format.rcp
  simp only [isNaN_binary32, isInf_binary32, isZero_binary32, sign_binary32, magRat_binary32, roundMag_binary32,
    apply_ite (BitVec.ofNat 32), ← withSign_binary32]
  rfl

theorem div_binary32 (x y : Bits) :
    div x y = BitVec.ofNat 32 (binary32.div .rn 0x7fffffff x.toNat y.toNat) := by
  unfold div Format.div
  simp only [isNaN_binary32, isInf_binary32, isZero_binary32, sign_binary32, magRat_binary32, roundMag_binary32,
    apply_ite (BitVec.ofNat 32), ← withSign_binary32]
  rfl

theorem lt_binary32 (x y : Bits) : lt x y = binary32.lt x.toNat y.toNat := rfl

theorem fmax_binary32 (x y : Bits) :
    fmax x y = BitVec.ofNat 32 (binary32.fmax 0x7fffffff x.toNat y.toNat) := by
  unfold fmax Format.fmax
  simp only [isNaN_binary32, isZero_binary32, sign_binary32, lt_binary32, apply_ite (BitVec.ofNat 32), ofNat_toNat]
  rfl

theorem fmin_binary32 (x y : Bits) :
    fmin x y = BitVec.ofNat 32 (binary32.fmin 0x7fffffff x.toNat y.toNat) := by
  unfold fmin Format.fmin
  simp only [isNaN_binary32, isZero_binary32, sign_binary32, lt_binary32, apply_ite (BitVec.ofNat 32), ofNat_toNat]
  rfl

end F32

end TVA.Math.Float
