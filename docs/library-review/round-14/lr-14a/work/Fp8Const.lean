import TVA.Lang.Promote
/-! lr-14a: the library's model of `tl.full([B], float('inf'), tl.float8e5)` (SEMANTICS
"Constants": `FloatV.litF32`, then `tl.cvt8`) under `CvtSatF32E5m2` is `cvtSat .rn` of the float32
constant; the device stores the exact conversion `FloatV.cvt` (the folded `arith.truncf`). -/
open TVA

def infF64 : F64 := FloatV.ofPat 0x7ff0000000000000
def negInfF64 : F64 := FloatV.ofPat 0xfff0000000000000

-- what CvtSatF32E5m2 makes of the model's `tl.cvt8 (litF32 c)`: 0x7b, 0xfb (±57344)
#eval ((FloatV.cvtSat .rn (FloatV.litF32 infF64) : F8E5).pat, (FloatV.cvtSat .rn (FloatV.litF32 negInfF64) : F8E5).pat)
-- the exact conversion of the float32 constant, as the compiler folds it: 0x7c, 0xfc (±∞)
#eval ((FloatV.cvt (FloatV.litF32 infF64) : F8E5).pat, (FloatV.cvt (FloatV.litF32 negInfF64) : F8E5).pat)
