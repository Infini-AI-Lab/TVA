import TVA
open TVA

/-! Usability probes: does a kernel's typical goal close with the round-39 lemmas as stated? -/

-- L175: literals by `simp`
example : (128 : I32).toInt = 128 := by simp
example : (0 : I64).toInt = 0 := by simp
example : (1 : I32).toInt = 1 := by simp
example : (-1 : I64).toInt = -1 := by simp
example : (IntV.ofNat 4096 : I64).toInt = 4096 := by simp
example : (255 : U8).toInt = 255 := by simp
example (a b : Nat) : (IntV.ofNat a + IntV.ofNat b : I32) = IntV.ofNat (a + b) := by simp
-- `ne` by `eq`
example (a b : I32) (h : a.toInt ≠ b.toInt) : (!a.eq b) = true := by simp [IntV.not_eq_eq, h]
-- `pid * BM + arange` in int32, then promoted to int64 via ConvRel
example (pid : I32) (p : Nat) (hp : pid.toInt = p) (hfit : p * 64 + 64 ≤ 2 ^ 31) (i : Fin (64 - 0)) :
    ((pid * (64 : I32) + tl.arange 0 64).get1 i).toInt = ((p * 64 + i : Nat) : Int) := by
  have h1 : (pid * IntV.ofNat 64).toInt = ((p * 64 : Nat) : Int) :=
    IntV.mul_ofNat hp (by omega)
  exact IntV.I32.toInt_sadd_arange h1 hfit i

-- L176: mask written as Triton writes it, `(rows < M)[:, None] & (cols < N)[None, :]`:
-- the comparison before the broadcast
example {BM BN r0 c0 M N : Nat} {rows : Tensor [BM] I32} {cols : Tensor [BN] I32} {mv nv : I32}
    (hr : ∀ i : Fin BM, (rows.get1 i).toInt = ((r0 + i : Nat) : Int))
    (hc : ∀ j : Fin BN, (cols.get1 j).toInt = ((c0 + j : Nat) : Int)) (hmv : mv.toInt = M)
    (hnv : nv.toInt = N) :
    Tile2.MaskOk (some (((rows.ltS mv).col BN).and ((cols.ltS nv).row BM))) r0 c0 M N :=
  Tile2.maskOk_and hr hc hmv hnv

-- L177: constants evaluate
example : ((Math.Float.Format.binary32.u : ℚ) : ℝ) ≤ 1 / 1000000 := by
  rw [Math.Float.Format.binary32_u_real]; norm_num

-- L178: exp2 kernels
example (x : ℝ) : (2 : ℝ) ^ x = Real.exp (x * Real.log 2) := OnlineSoftmax.two_rpow_eq_exp x
