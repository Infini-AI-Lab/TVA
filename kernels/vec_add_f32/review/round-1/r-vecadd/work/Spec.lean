import VecAdd

/-! Reviewer r-vecadd, round 1: `Post`'s `FloatV.add` on `float32` rejects plausible wrong
kernels (exact, no tolerance).

* `1 + 2^-10` is exact in `float32` (`0x3F802000`); a kernel that adds in `bfloat16` (7 fraction
  bits) would leave `1.0` (`0x3F800000`), which `Post` rejects.
* `1 + 2^-24` is a tie and rounds to even (`1.0`); `1 + 3·2^-25` rounds up (`0x3F800001`);
  `1 + 3·2^-24` is a tie and rounds to even (`0x3F800002`).
* `x - y` and `x` (a dropped term) differ from the sum on `1 + 2`. -/

open TVA

def f (n : Nat) : F32 := FloatV.ofPat n

#eval (FloatV.add (f 0x3F800000) (f 0x3A800000)).pat == 0x3F802000   -- 1 + 2^-10
#eval (FloatV.add (f 0x3F800000) (f 0x33800000)).pat == 0x3F800000   -- 1 + 2^-24 (tie, even)
#eval (FloatV.add (f 0x3F800000) (f 0x33C00000)).pat == 0x3F800001   -- 1 + 3·2^-25 (0.75 ulp, up)
#eval (FloatV.add (f 0x3F800000) (f 0x34400000)).pat == 0x3F800002   -- 1 + 3·2^-24 (1.5 ulp, tie, even)
#eval (FloatV.add (f 0x3F800000) (f 0x40000000)).pat == 0x40400000   -- 1 + 2 = 3
#eval (FloatV.add (f 0x7F800000) (f 0xFF800000)).isNaN                -- inf + -inf is NaN
#eval (FloatV.add (f 0x7F7FFFFF) (f 0x7F7FFFFF)).pat == 0x7F800000   -- overflow to +inf
#eval (FloatV.add (f 0x80000000) (f 0x80000000)).pat == 0x80000000   -- -0 + -0 = -0
#eval (FloatV.add (f 0x00000001) (f 0x00000001)).pat == 0x00000002   -- subnormals kept
