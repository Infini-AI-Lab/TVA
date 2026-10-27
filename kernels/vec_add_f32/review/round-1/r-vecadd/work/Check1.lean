import VecAdd
open TVA
set_option pp.fullNames true
set_option pp.explicit false
#print TVAExamples.VecAdd.add_kernel
#print TVAExamples.VecAdd.model
#print TVAExamples.VecAdd.launch
#print TVAExamples.VecAdd.Post
#print axioms TVAExamples.VecAdd.main
#print axioms TVAExamples.VecAdd.consistent
-- IEEE spot checks (binary32 patterns)
def f (n : Nat) : F32 := FloatV.ofPat n
#eval (FloatV.add (f 0x3f800000) (f 0x33800000)).pat  -- 1 + 2^-24: tie -> 1.0 (0x3f800000)
#eval (FloatV.add (f 0x3f800001) (f 0x33800000)).pat  -- tie -> even 0x3f800002
#eval (FloatV.add (f 0x00000001) (f 0x00000001)).pat  -- subnormal: 0x2
#eval (FloatV.add (f 0x80000000) (f 0x80000000)).pat  -- -0 + -0 = -0 0x80000000
#eval (FloatV.add (f 0x80000000) (f 0x00000000)).pat  -- -0 + +0 = +0
#eval (FloatV.add (f 0x3f800000) (f 0xbf800000)).pat  -- 1 + -1 = +0
#eval (FloatV.add (f 0x7f7fffff) (f 0x7f7fffff)).pat  -- max+max = inf 0x7f800000
#eval (FloatV.add (f 0x7f800000) (f 0xff800000)).pat  -- inf + -inf = NaN
#eval (FloatV.add (f 0x7f800000) (f 0x3f800000)).pat  -- inf
#eval (FloatV.add (f 0x007fffff) (f 0x00000001)).pat  -- largest subnormal + tiny = min normal 0x00800000
#eval (FloatV.add (f 0x7f7fffff) (f 0x73000000)).pat  -- max + 2^103 (half ulp) tie -> inf? ulp(max)=2^104; half=2^103 -> tie, max frac odd -> round up to inf
#eval (FloatV.add (f 0x7f7fffff) (f 0x72ffffff)).pat  -- just below half ulp -> max
#eval (FloatV.add (f 0x80000001) (f 0x00000001)).pat  -- cancel -> +0
#eval (FloatV.add (f 0xc0400000) (f 0x3f800000)).pat  -- -3 + 1 = -2 0xc0000000
#eval (FloatV.add (f 0x4b800000) (f 0x3f800000)).pat  -- 2^24 + 1 -> tie -> 2^24 0x4b800000
#eval (FloatV.add (f 0x4b800001) (f 0x3f800000)).pat  -- (2^24+2) + 1 -> tie -> even: 2^24+4 0x4b800002
