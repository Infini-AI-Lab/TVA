import Proof.Bounds
import Proof.Keys

/-!
# The random number of one lane

`safe_rand` describes each element of `tl.rand` through the lane's counter word: some `v` with
`ItofRel` from the non-negative conversion and `NaNEq` of the product with `Philox.scale32`.
With the seed's key words set to the claim's `key0`, `key1` and the lane's offset set to
`p * BLOCK_SIZE + l`, that element is the claim's `uRef`.
-/

namespace Claims.SeededDropout

open TVA
open TVA.Format
open Block


/-- A lane of `tl.rand`, as `safe_rand` describes it, is the claim's `uRef` of that lane's element. -/
theorem lane_rand {sg : Bool} {w : Nat} {d : Inst} (seed : IntV sg w)
    (hk0 : (seed.cast : U64).bits.setWidth 32 = key0 d.seed)
    (hk1 : ((seed.cast : U64).bits >>> 32).setWidth 32 = key1 d.seed)
    {p : Nat} (hfit : Fits p d.BLOCK_SIZE) (l : Fin (d.BLOCK_SIZE - 0)) {v r : F32}
    (hv : tl.ItofRel .binary32
      (Philox.toNonneg (Philox.randint4x seed ((offs p d.BLOCK_SIZE).get1 l) 10).1).toInt v)
    (hr : FloatV.NaNEq (FloatV.mul v Philox.scale32) r) :
    r = uRef d.seed (p * d.BLOCK_SIZE + l) := by
  have h1 : (Philox.randint4x seed ((offs p d.BLOCK_SIZE).get1 l) 10).1
      = (⟨rbits d.seed (p * d.BLOCK_SIZE + l)⟩ : U32) :=
    randint4x_fst seed _ _ (lane_bits hfit l) hk0 hk1
  have h2 : (Philox.toNonneg
        (Philox.randint4x seed ((offs p d.BLOCK_SIZE).get1 l) 10).1).toInt
      = (rint d.seed (p * d.BLOCK_SIZE + l) : Int) := by
    rw [h1]
    exact toNonneg_rint _ _
  have hz : |(rint d.seed (p * d.BLOCK_SIZE + l) : Int)| < 2 ^ 53 := by
    have h := rint_lt d.seed (p * d.BLOCK_SIZE + l)
    rw [abs_of_nonneg (Int.natCast_nonneg _)]
    have h' : (rint d.seed (p * d.BLOCK_SIZE + l) : Int) < 2 ^ 31 := by exact_mod_cast h
    linarith
  have hv' : v = FloatV.ofRat ((rint d.seed (p * d.BLOCK_SIZE + l) : Nat) : ℚ) := by
    rw [h2] at hv
    simpa using (tl.itofRel_binary32 hz (r := v)).mp hv
  have hr' : FloatV.NaNEq (uRef d.seed (p * d.BLOCK_SIZE + l)) r := by
    rw [hv'] at hr
    exact hr
  exact FloatV.NaNEq.eq hr' (uRef_isFinite d.seed _).1

end Claims.SeededDropout
