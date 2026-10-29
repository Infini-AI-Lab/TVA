import VecAdd

/-! r-vecadd2, round 2, at gate 990243026f1289fb: what `Post` asks of each output element,
evaluated on `float32` patterns with the library as it is now (its `Format.toRat` and
`Format.nanMag` changed since the last acceptance). `Post` asks `NaNEq (FloatV.add a b) c`, so these
are the values a correct launch must store, and each line shows a plausible wrong kernel that
`Post` rejects. -/

open TVA

def f (n : Nat) : F32 := FloatV.ofPat n
def addp (a b : Nat) : Nat := (FloatV.add (f a) (f b)).pat
def hex (n : Nat) : String := "0x" ++ String.ofList (Nat.toDigits 16 n)

-- 1 + 2^-23: exact, 0x3f800001 (a kernel adding in bfloat16 or tf32 would give 0x3f800000).
#eval hex (addp 0x3f800000 0x34000000)
-- 1 + 2^-24: a tie, to even, 0x3f800000 (round half up would give 0x3f800001).
#eval hex (addp 0x3f800000 0x33800000)
-- 1 + 3*2^-25: rounds up, 0x3f800001 (truncation would give 0x3f800000).
#eval hex (addp 0x3f800000 0x33c00000)
-- the smallest subnormal twice: 0x00000002 (flush to zero would give 0).
#eval hex (addp 0x00000001 0x00000001)
-- the largest finite twice: +inf 0x7f800000.
#eval hex (addp 0x7f7fffff 0x7f7fffff)
-- +inf + -inf: a NaN.
#eval (FloatV.add (f 0x7f800000) (f 0xff800000)).isNaN
-- -0 + -0 is -0 (0x80000000), +0 + -0 is +0, and x + (-x) is +0.
#eval (hex (addp 0x80000000 0x80000000), hex (addp 0x00000000 0x80000000), hex (addp 0x3f800000 0xbf800000))
-- a NaN input gives a NaN, whatever the other operand.
#eval ((FloatV.add (f 0x7fc00001) (f 0x3f800000)).isNaN, (FloatV.add (f 0x3f800000) (f 0xffc00000)).isNaN)
-- `NaNEq x y` is `x = y ∨ (x.isNaN ∧ y.isNaN)`: it tells +0 from -0, so a kernel that stored +0
-- for -0 + -0 fails `Post` (equal: false; both NaN: false) ...
#eval (FloatV.add (f 0x80000000) (f 0x80000000) == f 0,
  (FloatV.add (f 0x80000000) (f 0x80000000)).isNaN && (f 0).isNaN)
-- ... and it accepts any NaN for a NaN sum (payload only): both NaN.
#eval (FloatV.add (f 0x7f800000) (f 0xff800000)).isNaN && (f 0xffc12345).isNaN

/-! `Post`'s frame on a covered instance: a final memory that also wrote one byte past the first
`n` elements of `output` fails `Post`. Here `n = 2` and `output` has 3 elements; byte `base + 8`
(element 2) is outside `output.elems {j | j < 2}`. -/

def inst : Kernels.VecAdd.Inst where
  n := 2
  BLOCK_SIZE := 1024
  x := ⟨0, 2⟩
  y := ⟨64, 2⟩
  output := ⟨128, 3⟩

example (init fin : Mem) (h : fin 136 ≠ init 136) : ¬ Kernels.VecAdd.Post inst init fin := by
  intro hp
  apply h
  refine hp.2 136 ?_
  rintro ⟨j, hj, hx⟩
  simp only [Set.mem_ofPred_eq, inst] at hj
  simp only [Mem.span, Set.mem_ofPred_eq, Buf.ptr, inst, DType.bytes] at hx
  omega
