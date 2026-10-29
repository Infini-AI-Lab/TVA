import TVA
/-! lr-19a: every name that L99's and L100's new docstrings cite resolves to the library's own
declaration (not Mathlib's), and the new theorems use only the standard axioms. -/
open TVA TVA.Math.Float

-- L99: `TVA.Lemmas.FloatMax`'s module docstring
#check @FloatV.le_iff
#check @FloatV.lt_iff
#check @FloatV.eq_iff
#check @FloatV.le_refl'
#check @FloatV.le_trans'
#check @FloatV.ext_pat
#check @FloatV.max_mem
#check @FloatV.min_mem
#check @FloatV.toRat_max
#check @FloatV.toRat_min
#check @FloatV.isNaN_of_mem
#check @FloatV.MaxRel.mem
#check @FloatV.MaxRel.le
#check @FloatV.MaxRel.ge
#check @FloatV.negInf_toRat_le
#check @FloatV.max_negInf_left
#check @FloatV.MaxRel.negInf_left
#check @FloatV.sub_negInf_left
#check @FloatV.sub_negInf_right
#check @FloatV.sub_negInf_negInf
-- L99: new theorems
#check @Format.toRat_of_isInf
#check @Format.maxVal_lt_inf
#check @Format.cvt_rz_sat
#check @Format.sqrt_rint_err
#check @Format.one_le_maxVal
#check @Format.sqrtMag_rn_spec
#check @Format.sqrt_rn_err
#check @FloatV.sqrtRn_err
-- L100
#check @Buf.Addrs.accBytes
#check @Buf.addrs_of_get
#check @Buf.addrs_of_get_all
#check @SafeA.atomic1
#check @SafeA.atomic
#check @safe_atomic_buf
#check @safe_upToNaN
#check @safe_cvt
#check @safe_dot_add
#check @safe_sum_bcast_mul
#check @IntV.toInt_cast
#check @IntV.toInt_cast_unsigned
#check @IntV.lt_toInt
#check @Safe.env
#check @SafeA.env

#print axioms Format.toRat_of_isInf
#print axioms Format.maxVal_lt_inf
#print axioms Format.cvt_rz_sat
#print axioms Format.sqrtMag_rn_spec
#print axioms Format.sqrt_rn_err
#print axioms FloatV.sqrtRn_err
#print axioms IntV.toInt_cast
#print axioms IntV.toInt_cast_unsigned

-- L98: the NaN constants, evaluated (patterns in hex)
def hx (n : Nat) : String := "0x" ++ String.ofList (Nat.toDigits 16 n)
#eval [FloatV.litF32 (.ofPat 0x7ff8000000000000), FloatV.litF32 (.ofPat 0xfff8000000000000),
       FloatV.litF32 (.ofPat 0x7ff0000000000001), FloatV.litF32 (.ofPat 0xfff0000020000000)].map (hx ·.pat)
#eval [FloatV.litF16 (.ofPat 0x7ff8000000000000), FloatV.litF16 (.ofPat 0xfff8000000000000),
       FloatV.litF16 (.ofPat 0x7ff4000000000000), FloatV.litF16 (.ofPat 0x7fffffffffffffff)].map (hx ·.pat)
#eval [FloatV.litBF16 (.ofPat 0x7ff8000000000000), FloatV.litBF16 (.ofPat 0xfff8000000000000),
       FloatV.litBF16 (.ofPat 0x7fffffffffffffff)].map (hx ·.pat)
#eval [FloatV.litF64 (.ofPat 0x7ff0000000000001), FloatV.litF64 (.ofPat 0xfff8000000000000)].map (hx ·.pat)
-- `constIn?` on a NaN operand, and the canonical NaN of each format
#eval ((Operand.float (.ofPat 0xfff8000000000000)).constIn? .f32).map (hx ·.pat)
#eval ((Operand.float (.ofPat 0xfff8000000000000)).constIn? .bf16).map (hx ·.pat)
#eval [hx (FloatV.nan : F32).pat, hx (FloatV.nan : F16).pat, hx (FloatV.nan : BF16).pat,
       hx (FloatV.nan : F8E5).pat, hx (FloatV.nan : F8E4).pat, hx (FloatV.nan : F64).pat]
-- the value of `±∞` and `maxVal` (L99)
#eval (FloatV.ofPat 0x7f800000 : F32).toRat == 2 ^ 128
#eval (FloatV.ofPat 0x7c : F8E5).toRat == 2 ^ 16
#eval Format.e5m2.maxVal < 2 ^ 16
