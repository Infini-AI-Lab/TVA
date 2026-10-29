import VecAdd

/-! r-vecadd, round 2 (second filing): the conditions on the host's instances and at the edge of
the size condition, and what `Post` asks of one element. -/

namespace Kernels.VecAdd

/-- The host's launch shape: `BLOCK_SIZE = B`, three fresh buffers of `n` elements, one after
another from `2^21` (4-byte aligned, disjoint). -/
def hostInst (n B : Nat) : Inst where
  n := n
  BLOCK_SIZE := B
  x := ⟨2097152, n⟩
  y := ⟨2097152 + 4 * n, n⟩
  output := ⟨2097152 + 8 * n, n⟩

theorem host_ok {n B e : Nat} (hB : B = 2 ^ e) (he : e ≤ 20) (hn : n < 2 ^ 31) :
    (hostInst n B).ok := by
  refine ⟨⟨e, hB⟩, ?_, hn, le_refl _, le_refl _, le_refl _, ?_, ?_⟩
  · show B ≤ 2 ^ 20
    rw [hB]; exact Nat.pow_le_pow_right (by norm_num) he
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by simp [hostInst]; omega)
  · exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by simp [hostInst]; omega)

-- the source's test size, the benchmark's largest, and the largest `int32` size, at `BLOCK_SIZE = 1024`
example : (hostInst 98432 1024).ok := host_ok (e := 10) (by norm_num) (by norm_num) (by norm_num)
example : (hostInst (2 ^ 27) 1024).ok := host_ok (e := 10) (by norm_num) (by norm_num) (by norm_num)
example : (hostInst (2 ^ 31 - 1) 1024).ok := host_ok (e := 10) (by norm_num) (by norm_num) (by norm_num)
-- the largest block at the largest size; the smallest block; `n = 0` and `n = 1`
example : (hostInst (2 ^ 31 - 1) (2 ^ 20)).ok := host_ok (e := 20) rfl (by norm_num) (by norm_num)
example : (hostInst (2 ^ 31 - 1) 1).ok := host_ok (e := 0) rfl (by norm_num) (by norm_num)
example : (hostInst 0 1024).ok := host_ok (e := 10) (by norm_num) (by norm_num) (by norm_num)
example : (hostInst 1 1024).ok := host_ok (e := 10) (by norm_num) (by norm_num) (by norm_num)

-- `n = 2^31`, where Triton passes `n_elements` as an `int64`, is outside the conditions
example : ¬ (hostInst (2 ^ 31) 1024).ok := fun h => absurd h.n_lt (by simp [hostInst])

-- `x` and `y` the same tensor (`add(x, x)`) is covered
def sameInst (n : Nat) : Inst where
  n := n
  BLOCK_SIZE := 1024
  x := ⟨2097152, n⟩
  y := ⟨2097152, n⟩
  output := ⟨2097152 + 4 * n, n⟩

example : (sameInst (2 ^ 27)).ok := by
  refine ⟨⟨10, rfl⟩, by norm_num [sameInst], by norm_num [sameInst], le_refl _, le_refl _, le_refl _,
    ?_, ?_⟩ <;>
  exact elems_disjoint_of_lt (le_refl _) (le_refl _) (by simp [sameInst])

-- in place (`output` is `x`) is not covered: element 0's bytes are in both
example : ¬ (⟨1024, 1024, ⟨2097152, 1024⟩, ⟨8388608, 1024⟩, ⟨2097152, 1024⟩⟩ : Inst).ok := by
  intro h
  have hx : (2097152 : Int) ∈ TVA.Buf.elems (⟨2097152, 1024⟩ : TVA.Buf .f32) {j | j < 1024} :=
    ⟨0, by simp, by simp [TVA.Mem.span, TVA.Buf.ptr, TVA.DType.bytes]⟩
  exact Set.disjoint_left.mp h.output_x hx hx

end Kernels.VecAdd

open TVA

/-! `FloatV.add` on `float32` patterns: what `Post` asks of each element. -/
def f (n : Nat) : F32 := FloatV.ofPat n

-- 1 + 2^-24 is a tie: to even, 1 (0x3f800000)
#eval (FloatV.add (f 0x3f800000) (f 0x33800000)).pat == 0x3f800000
-- 1 + 3·2^-24 rounds up to 1 + 2^-22 (0x3f800002)
#eval (FloatV.add (f 0x3f800000) (f 0x34400000)).pat == 0x3f800002
-- 1 + 2^-10 is exact in float32 (0x3f802000); a bfloat16 sum would give 1, which `Post` rejects
#eval (FloatV.add (f 0x3f800000) (f 0x3a800000)).pat == 0x3f802000
-- the largest finite value doubled overflows to +inf
#eval (FloatV.add (f 0x7f7fffff) (f 0x7f7fffff)).pat == 0x7f800000
-- inf + (-inf) is NaN
#eval (FloatV.add (f 0x7f800000) (f 0xff800000)).isNaN
-- -0 + -0 = -0, and -0 + +0 = +0
#eval (FloatV.add (f 0x80000000) (f 0x80000000)).pat == 0x80000000
#eval (FloatV.add (f 0x80000000) (f 0x00000000)).pat == 0x00000000
-- subnormals are kept: 2^-149 + 2^-149 = 2^-148
#eval (FloatV.add (f 1) (f 1)).pat == 2
-- a NaN input gives a NaN
#eval (FloatV.add (f 0x7fc00001) (f 0x3f800000)).isNaN
