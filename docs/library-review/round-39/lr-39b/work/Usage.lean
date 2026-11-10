import TVA
/-! Usability probes of the round-39 lemmas on the shapes kernels write. -/
open TVA TVA.Tile2

-- L175: numerals by `simp`, any width.
example : ((128 : I32)).toInt = 128 := by simp
example : ((128 : I64)).toInt = 128 := by simp
example : ((0 : I64)).toInt = 0 := by simp
example : ((-1 : I32)).toInt = -1 := by simp
example : ((1 : I64)).toInt = 1 := by simp
example : ((255 : U8)).toInt = 255 := by simp

-- L175: int64 index arithmetic carried by tracked values.
example (a : I64) (h : a.toInt = (5 : Nat)) :
    (a * IntV.ofNat 1024 + IntV.ofNat 3 : I64).toInt = ((5 * 1024 + 3 : Nat) : Int) := by
  exact IntV.add_ofNat (IntV.mul_ofNat h (by norm_num)) (by norm_num)

-- L175: equality as `decide` on values.
example (a b : I32) (h : a.toInt = b.toInt) : a.eq b = true := by
  rw [IntV.eq_eq]; simp [h]

-- L176: the pointers of a pid-offset tile in int32, row-major `s1 = 1`.
example {d : DType} (b : Buf d) (pm BM BN M N K : Nat) (ha : ((IntV.ofNat pm : I32) * IntV.ofNat BM).toInt = ((pm * BM : Nat) : Int))
    (hfit : pm * BM + BM ≤ 2 ^ 31) (hK : (pm * BM + BM - 1) * K + (0 + BN - 1) * 1 < 2 ^ 31)
    (hBN : BN ≤ 2 ^ 31) (hKv : ((IntV.ofNat K : I32)).toInt = K) (i : Fin BM) (j : Fin BN) :
    (b.ptr (0 : Nat) + ((IntV.ofNat pm : I32) * IntV.ofNat BM + tl.arange 0 BM).col BN * (IntV.ofNat K : I32)
        + (tl.arange 0 BN).row BM * (IntV.ofNat 1 : I32)).get2 i j =
      b.ptr ((0 + (pm * BM + i) * K + (0 + j) * 1 : Nat) : Int) := by
  have hr := fun i => start_arange_toInt (n := BM) ha (by omega) i
  have hc := fun j => arange_toInt (n := BN) hBN j
  have hk1 : ((IntV.ofNat 1 : I32)).toInt = (1 : Nat) := by simp
  exact ptr_lane_seq b rfl (by norm_num) (by norm_num) hr hc hKv hk1 (by simp only [Nat.sub_zero]; omega) (by omega) i j
