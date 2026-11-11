import Claim
import TVA.Proof

/-!
# The Philox key and counter of `seeded_dropout`

The claim's reference random number `uRef seed j` is built from the host's `seed` and the index
`j`; the kernel's `tl.rand seed offsets` builds it from the argument `seed` (an `int32`, `int64`
or `uint64` holding the same integer modulo `2^w`) and the lane's `int32` offset. This module
shows that the two agree: the seed's value modulo `2^64` gives the two key words `key0`, `key1`,
and the lane's offset gives the counter word `j`, so the first word of `Philox.randint4x` is
`rbits d.seed j`, and `Philox.toNonneg` turns it into `rint d.seed j`.
-/

namespace Claims.SeededDropout

open TVA
open TVA.Format
open Block


/-! ## The 64-bit view of the seed -/

/-- Truncating `BitVec.ofInt 64 s` to 32 bits is `BitVec.ofInt 32 s` (`key0`). -/
theorem ofInt64_setWidth (s : Int) : (BitVec.ofInt 64 s).setWidth 32 = BitVec.ofInt 32 s := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_setWidth, BitVec.toNat_ofInt, BitVec.toNat_ofInt]
  omega

/-- The high 32 bits of `BitVec.ofInt 64 s` are `key1 s`. -/
theorem ofInt64_hi (s : Int) :
    ((BitVec.ofInt 64 s) >>> 32).setWidth 32 = BitVec.ofNat 32 ((s % 2 ^ 64).toNat / 2 ^ 32) := by
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_setWidth, BitVec.toNat_ushiftRight, BitVec.toNat_ofInt, BitVec.toNat_ofNat]
  omega

/-! ## The seed cast to `uint64` -/

/-- An `int32` seed in range, cast to `uint64` (sign extension), is `s mod 2^64`. -/
theorem cast_u64_of_i32 (s : Int) (h1 : -(2 ^ 31) ≤ s) (h2 : s < 2 ^ 31) :
    ((IntV.ofInt s : I32).cast : U64).bits = BitVec.ofInt 64 s := by
  show BitVec.ofInt 64 (BitVec.ofInt 32 s).toInt = BitVec.ofInt 64 s
  rw [BitVec.toInt_ofInt, Int.bmod_eq_of_le (by omega) (by omega)]

/-- An `int64` seed, cast to `uint64`, keeps its bits. -/
theorem cast_u64_of_i64 (s : Int) :
    ((IntV.ofInt s : I64).cast : U64).bits = BitVec.ofInt 64 s := by
  show BitVec.ofInt 64 (BitVec.ofInt 64 s).toInt = BitVec.ofInt 64 s
  rw [BitVec.ofInt_toInt]

/-- A `uint64` seed, cast to `uint64`, keeps its bits. -/
theorem cast_u64_of_u64 (s : Int) :
    ((IntV.ofInt s : U64).cast : U64).bits = BitVec.ofInt 64 s := by
  show BitVec.zeroExtend 64 (BitVec.ofInt 64 s) = BitVec.ofInt 64 s
  exact BitVec.setWidth_eq _

/-! ## The lane's counter word -/

/-- The bits of lane `l`'s offset are the natural number `p * B + l`. -/
theorem lane_bits {p B : Nat} (hfit : Fits p B) (l : Fin (B - 0)) :
    ((offs p B).get1 l).bits = BitVec.ofNat 32 (p * B + l) := by
  have h := Block.lane_toInt hfit l
  have h2 : ((offs p B).get1 l).bits
      = BitVec.ofInt 32 (((offs p B).get1 l).toInt) := BitVec.ofInt_toInt.symm
  rw [h2, h]
  exact BitVec.ofInt_ofNat 32 (p * B + l)

/-! ## `Philox.randint4x` on a lane -/

/-- With the seed's two key words and the lane's counter word fixed, the first word of
`Philox.randint4x` is `rbits d.seed j`. -/
theorem randint4x_fst {sg : Bool} {w : Nat} {d : Inst} (seed : IntV sg w) (o : I32) (j : Nat)
    (ho : o.bits = BitVec.ofNat 32 j)
    (hk0 : (seed.cast : U64).bits.setWidth 32 = key0 d.seed)
    (hk1 : ((seed.cast : U64).bits >>> 32).setWidth 32 = key1 d.seed) :
    (Philox.randint4x seed o 10).1 = (⟨rbits d.seed j⟩ : U32) := by
  have hlo : (o.cast : U32).bits = BitVec.ofNat 32 j := by
    show BitVec.ofInt 32 o.bits.toInt = BitVec.ofNat 32 j
    rw [BitVec.ofInt_toInt, ho]
  simp only [Philox.randint4x, hlo, hk0, hk1, Nat.lt_irrefl, ite_false]
  rfl

/-- `Philox.toNonneg` of `rbits d.seed j` is the claim's non-negative integer `rint d.seed j`. -/
theorem toNonneg_rint (seed : Int) (j : Nat) :
    (Philox.toNonneg (⟨rbits seed j⟩ : U32)).toInt = (rint seed j : Int) := by
  have hr : (rbits seed j).toNat < 2 ^ 32 := (rbits seed j).isLt
  set r := (rbits seed j).toNat with hrr
  have hb : (⟨rbits seed j⟩ : I32).toInt
      = if 2 * r < 2 ^ 32 then (r : Int) else (r : Int) - 2 ^ 32 := rfl
  have h0 : IntV.toInt (0 : I32) = 0 := by decide +kernel
  have hz : IntV.lt (⟨rbits seed j⟩ : I32) 0 = decide (2 ^ 32 ≤ 2 * r) := by
    rw [IntV.lt_eq, hb, h0, decide_eq_decide]
    omega
  have hcond : IntV.lt (⟨rbits seed j⟩ : I32) 0 = true ↔ 2 ^ 32 ≤ 2 * r := by
    rw [hz, decide_eq_true_eq]
  have hbody : (Philox.toNonneg (⟨rbits seed j⟩ : U32)).toInt
      = (if IntV.lt (⟨rbits seed j⟩ : I32) 0 = true then IntV.not (⟨rbits seed j⟩ : I32)
          else (⟨rbits seed j⟩ : I32)).toInt := rfl
  rw [hbody, rint]
  by_cases h : 2 ^ 32 ≤ 2 * r
  · rw [ite_eq_left (hcond.mpr h), ite_eq_right (by omega : ¬ r < 2 ^ 31)]
    show (IntV.not (⟨rbits seed j⟩ : I32)).toInt = _
    rw [IntV.not, IntV.toInt,
      BitVec.toInt_eq_toNat_of_lt (x := ~~~rbits seed j) (by rw [BitVec.toNat_not]; omega),
      BitVec.toNat_not]
    simp
  · rw [ite_eq_right (fun hc => h (hcond.mp hc)), ite_eq_left (by omega : r < 2 ^ 31)]
    show (⟨rbits seed j⟩ : I32).toInt = _
    rw [hb, ite_eq_left (by omega : 2 * r < 2 ^ 32)]

end Claims.SeededDropout
