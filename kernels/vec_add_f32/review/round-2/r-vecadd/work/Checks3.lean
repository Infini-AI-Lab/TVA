import VecAdd

/-! r-vecadd, round 2, third filing: small facts behind the map's notes. -/

open TVA Kernels.VecAdd

/-- Line 44's note: the constexpr built in `int32` (`IntV.ofInt`, modulo `2^32`) is the model's
`IntV.ofNat BLOCK_SIZE`. -/
example (B : Nat) : (IntV.ofNat B : I32) = IntV.ofInt (B : Int) := by
  simp [IntV.ofNat, IntV.ofInt]

/-- Line 47 at `n = 1`: the constant `1`, made an `int32` tensor, is the model's `IntV.ofNat 1`. -/
example : (IntV.ofNat 1 : I32) = IntV.ofInt 1 := by decide

/-- `program_id` axis `0` of the launch's context is the program's number on the 1-D grid. -/
example (d : Inst) (i : Nat) (hi : i < (launch d).grid 0) : ((launch d).ctx i).pid 0 = i := by
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt (by simpa [launch, Launch.grid1] using hi)]

/-- The block shape `tl.arange 0 B` gives is the one `launch` lists. -/
example (d : Inst) : (launch d).blocks = [[d.BLOCK_SIZE - 0]] := rfl

/-- The signed `int32` comparison of line 47 on the covered offsets, which lie in `[0, 2^31)`:
`offsets < n` is the comparison of the numbers. -/
example (o n : Nat) (ho : o < 2 ^ 31) (hn : n < 2 ^ 31) :
    (IntV.ofNat o : I32).lt (IntV.ofNat n) = decide (o < n) := by
  simp only [IntV.lt, IntV.ofNat, ite_true]
  rw [BitVec.slt_eq_decide, BitVec.toInt_ofNat', BitVec.toInt_ofNat']
  have h1 : Int.bmod (o : Int) (2 ^ 32) = o := by
    rw [Int.bmod_def, Int.emod_eq_of_lt (by omega) (by omega)]; split <;> omega
  have h2 : Int.bmod (n : Int) (2 ^ 32) = n := by
    rw [Int.bmod_def, Int.emod_eq_of_lt (by omega) (by omega)]; split <;> omega
  rw [h1, h2]; simp
