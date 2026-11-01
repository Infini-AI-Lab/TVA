import TVA

/-! lr-25a: the statements of the 16-bit facts restated in round 25 (L126), the shape `DotLiftAcc` and the lemmas
that use it, as Lean elaborates them, and their axioms; the arithmetic of the derivation of `relDot16`. Checked with `lake env lean` on this file only. -/

open TVA TVA.Rules.Num

-- the facts
#print TVA.Rules.Num.DotF16
#print TVA.Rules.Num.DotBF16
#print TVA.Rules.Num.DotLiftAcc
#print TVA.Rules.Num.DotChunkAcc
#print TVA.Rules.Num.DotF8Chunk
#print TVA.Rules.Num.minNormal8
#print TVA.Rules.Num.liftMag

-- the thresholds, and fp8's unchanged by round 25
example : minNormal8 .f16 = 2 ^ (-14 : ℤ) := rfl
example : minNormal8 .bf16 = 2 ^ (-126 : ℤ) := rfl
example : minNormal8 .f8e5 = 2 ^ (-14 : ℤ) := rfl
example : minNormal8 .f8e4nv = 2 ^ (-6 : ℤ) := rfl
example : minNormal8 .f32 = 0 := rfl

-- the lemmas and their axioms
#check @TVA.Dot8Chunk.DotF8Chunk.liftAcc
#check @TVA.Dot8Chunk.DotChunkAcc.liftErr
#check @TVA.Dot8Chunk.DotF8Chunk.err
#check @TVA.DotLift.dotAdd_step
#check @TVA.DotLift.DotLiftAcc.dotAcc
#check @TVA.Rules.Num.DotAcc.liftAcc
#print axioms TVA.Dot8Chunk.DotF8Chunk.liftAcc
#print axioms TVA.Dot8Chunk.DotF8Chunk.err
#print axioms TVA.Dot8Chunk.errB_le
#print axioms TVA.DotLift.dotAdd_step
#print axioms TVA.DotLift.DotLiftAcc.dotAcc
#print axioms TVA.Rules.Num.DotAcc.liftAcc
#print axioms TVA.Rules.Num.DotLiftAcc.mono
#print axioms TVA.Platform.sm90ref.dotF16
#print axioms TVA.Platform.sm90ref.dotBF16
#print axioms TVA.Platform.sm90ref.dotE5m2Chunk
#print axioms TVA.Platform.ref.dotLiftAcc

-- the derivation of relDot16 from the per-chunk bound: ceil(K/16) * 5.25 <= 2 (K + 2) for every K >= 1
example : ∀ K : ℕ, 1 ≤ K → (((K + 15) / 16 : ℕ) : ℚ) * (21 / 4) ≤ 2 * ((K : ℚ) + 2) := by
  intro K hK
  have h : (((K + 15) / 16 : ℕ) : ℚ) * 16 ≤ (K : ℚ) + 15 := by
    have := Nat.div_mul_le_self (K + 15) 16
    exact_mod_cast this
  have hK' : (1 : ℚ) ≤ K := by exact_mod_cast hK
  nlinarith

-- the 16-bit per-chunk constants: 17 * 2^-25 + 2^-23 = 5.25 * 2^-23 and 17 * 2^-151 + 2^-149 = 5.25 * 2^-149
example : (17 : ℚ) * 2 ^ (-25 : ℤ) + 2 ^ (-23 : ℤ) = 21 / 4 * 2 ^ (-23 : ℤ) := by norm_num
example : (17 : ℚ) * 2 ^ (-151 : ℤ) + 2 ^ (-149 : ℤ) = 21 / 4 * 2 ^ (-149 : ℤ) := by norm_num
