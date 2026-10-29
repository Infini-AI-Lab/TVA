import TVA
/-! lr-20b: spot checks of round 20's Lean (library issues L64, L101, L108).
Run from TVA's `lean/` folder: `lake env lean ../docs/library-review/round-20/lr-20b/work/Check.lean`. -/
open TVA TVA.Math.Float

-- L64: fp8 constants through `Operand.constIn?` (the value `tl.full`/`tl.where` builds), as patterns.
def pat8? {g : Format} (x : Option (FloatV g)) : Option Nat := x.map (·.pat)
#eval pat8? (Operand.constIn? (.float (FloatV.ofPat 0xfff0000000000000)) .f8e5)   -- -inf: some 0xfc = 252
#eval pat8? (Operand.constIn? (.float (FloatV.ofPat 0x7ff0000000000000)) .f8e4nv) -- +inf in e4m3: none
#eval pat8? (Operand.constIn? (.float (FloatV.ofPat 0x7ff8000000000000)) .f8e5)   -- nan: some 0x7e = 126
#eval pat8? (Operand.constIn? (.float (FloatV.ofPat 0x8000000000000000)) .f8e5)   -- -0.0: some 0
#eval pat8? (Operand.constIn? (.float (FloatV.ofRat (448 : Rat))) .f8e4nv)        -- 448: some 0x7e = 126
#eval pat8? (Operand.constIn? (.float (FloatV.ofRat (460 : Rat))) .f8e4nv)        -- 460: none
#eval pat8? (Operand.constIn? (.int (-3)) .f8e5)                                  -- -3: some 0xc2 = 194
#eval pat8? (Operand.constIn? (.bool true) .f8e4nv)                               -- True: some 0x38 = 56
#eval pat8? (Operand.constIn? (.float (FloatV.ofRat ((-1 : Rat) / 10^50))) .f8e5)  -- -1e-50: float32 -0, some 0x80 = 128

-- L101, L108: the new theorems and their axioms.
#check @Format.e4m3_u
#check @Format.e5m2_eta
#check @IntV.le_toInt
#print axioms Format.e4m3_u
#print axioms Format.e4m3_eta
#print axioms Format.e5m2_u
#print axioms Format.e5m2_eta
#print axioms IntV.le_toInt
