import TVA
/-! lr-20f, round 20: can a kernel prover evaluate the new and changed library constants as it needs?
Each `example` is one use: L64's `FloatV.litF8` and `Operand.constIn?` on the fp8 dtypes, L101's fp8 `u`
and `eta`, L108's `IntV.le_toInt`. -/
open TVA TVA.Math.Float

-- L64: `tl.full([16], float('-inf'), tl.float8e5)`: the double 0xfff0000000000000 gives 0xfc
example : (FloatV.litF8 (g := Format.e5m2) (FloatV.ofPat 0xfff0000000000000)).map (·.pat) = some 0xfc := by
  decide
-- the same through `Operand.constIn?`, as a model writes it
example : ((Operand.float (FloatV.ofPat 0xfff0000000000000)).constIn? .f8e5).map (fun x : F8E5 => x.pat) = some 0xfc := by
  decide
-- `0.1` in float8e4nv: rejected on the host (`none`)
example : (Operand.float (FloatV.ofPat 0x3fb999999999999a)).constIn? .f8e4nv = none := by decide +kernel
-- `448` (a Python int) in float8e4nv: 0x7e
example : ((Operand.int 448).constIn? .f8e4nv).map (fun x : F8E4 => x.pat) = some 0x7e := by decide +kernel
-- `float('nan')` in float8e5: 0x7e; in float8e4nv: rejected
example : ((Operand.float (FloatV.ofPat 0x7ff8000000000000)).constIn? .f8e5).map (fun x : F8E5 => x.pat) = some 0x7e := by decide +kernel
example : (Operand.float (FloatV.ofPat 0x7ff8000000000000)).constIn? .f8e4nv = none := by decide +kernel
-- L101: fp8 `u` and `eta` as numbers, for `FloatV.cvtSat_rn_err`'s `|c|·u + η`
example : Format.e4m3.u = 1 / 16 ∧ Format.e5m2.eta = 1 / 131072 := by
  rw [Format.e4m3_u, Format.e5m2_eta]; norm_num
-- L108: `<=` of unsigned integers is the comparison of their values
example (x y : IntV false 32) (h : x.toInt ≤ y.toInt) : x.le y = true := by
  rw [IntV.le_toInt]; exact decide_eq_true h
#print axioms IntV.le_toInt
#print axioms Format.e4m3_eta
