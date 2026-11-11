import Claim
import TVA.Proof
import Proof.Basic

/-!
# The program's `int32` arithmetic

The grid indices, the divisions of the model's prologue, and the `int32` facts the walk of
`prog_safe` needs: the loop's range check, its trip count, and the counters inside it.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-! ## The program ids -/

theorem pid_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : (i : Nat) < 2 ^ 31 := by
  have hi : (i : Nat) < grid d := by
    simpa only [launch_size] using i.isLt
  exact lt_trans hi (grid_lt hd)

/-- Program `i` of the launch reads `program_id 0` as the `int32` `i`. -/
theorem pid_val (d : Inst) (i : Fin (launch d).size) :
    tl.program_id ((launch d).ctx i) 0 = (IntV.ofNat (i : Nat) : I32) := by
  simp only [tl.program_id]
  rw [launch_pid d i]

/-- Its value as an integer is `i`. -/
theorem pid_toInt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (tl.program_id ((launch d).ctx i) 0).toInt = (i : Nat) := by
  rw [pid_val]
  exact IntV.I32.toInt_ofNat (pid_lt hd i)

/-- The two coordinates of program `i`: `pid_m` its row, `pid_n` its column. -/
theorem gridM_lt {d : Inst} (hd : d.ok) : gridM d < 2 ^ 31 := by
  have h1 : 0 < gridN d := gridN_pos hd
  have hle : gridM d ≤ grid d := by
    unfold grid
    exact Nat.le_mul_of_pos_right (gridM d) h1
  exact lt_of_le_of_lt hle (grid_lt hd)

/-- The value `int32` of a natural number below `2^31`. -/
theorem ofNat_toInt {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = (n : Int) :=
  IntV.I32.toInt_ofNat h

/-- An integer in `int32` range keeps its value as an `int32`. -/
theorem ofInt_toInt {x : Int} (h : -(2 ^ 31 : Int) ≤ x ∧ x < 2 ^ 31) :
    (IntV.ofInt x : I32).toInt = x :=
  IntV.I32.toInt_ofInt h

/-! ## The prologue's divisions -/

theorem gridM_cdiv (d : Inst) : (d.M + (128 - 1)) / 128 = gridM d := rfl

/-- `cdiv(K, 128) = K / 128`, as `K` is a multiple of `128`. -/
theorem cdivK_eq {d : Inst} (hd : d.ok) : (d.K + (128 - 1)) / 128 = d.K / 128 := by
  conv => lhs; rw [K_div d hd]
  rw [Nat.mul_add_div (by norm_num : 0 < 128),
    Nat.div_eq_of_lt (by norm_num : 127 < 128), Nat.add_zero]

/-- The loop bound `cdiv(K, 128)` fits in `int32`. -/
theorem cdivK_lt {d : Inst} (hd : d.ok) : (d.K + (128 - 1)) / 128 < 2 ^ 31 := by
  rw [cdivK_eq hd]
  have h := hd.K_ok.2.2
  have h2 : (2 : Nat) ^ 17 = 131072 := by norm_num
  omega

/-- The loop runs at most `2^10` trips. -/
theorem divK_le {d : Inst} (hd : d.ok) : d.K / 128 ≤ 2 ^ 10 := by
  have h := hd.K_ok.2.2
  have h2 : (2 : Nat) ^ 17 = 131072 := by norm_num
  omega

/-! ## The loop -/

/-- The loop `for k in range(0, K / 128)` is inside `int32`. -/
theorem rangeOk {d : Inst} (hd : d.ok) :
    tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32)
      (IntV.ofInt 1 : I32) := by
  have hhi : (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32).toInt
      = (((d.K + (128 - 1)) / 128 : Nat) : Int) := ofNat_toInt (cdivK_lt hd)
  have hlo : (IntV.ofInt 0 : I32).toInt = 0 := ofInt_toInt (by norm_num)
  have hst : (IntV.ofInt 1 : I32).toInt = 1 := ofInt_toInt (by norm_num)
  have hb : (((d.K + (128 - 1)) / 128 : Nat) : Int) ≤ 2 ^ 31 - 1 := by
    have h1 := cdivK_lt hd
    omega
  have hmax : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
    norm_num [IntV.maxInt]
  unfold tl.RangeOk
  rw [hhi, hlo, hst, hmax]
  refine ⟨by norm_num, ?_, by norm_num, by norm_num⟩
  linarith

/-- The loop runs `K / 128` times. -/
theorem tripCount_eq {d : Inst} (hd : d.ok) :
    tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32).toInt
      (IntV.ofInt 1 : I32).toInt = d.K / 128 := by
  have hhi : (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32).toInt
      = (((d.K + (128 - 1)) / 128 : Nat) : Int) := ofNat_toInt (cdivK_lt hd)
  have hlo : (IntV.ofInt 0 : I32).toInt = 0 := ofInt_toInt (by norm_num)
  have hst : (IntV.ofInt 1 : I32).toInt = 1 := ofInt_toInt (by norm_num)
  rw [tl.tripCount, hhi, hlo, hst, cdivK_eq hd]
  have h : ((d.K / 128 : Nat) : Int) - 0 + 1 - 1 = ((d.K / 128 : Nat) : Int) := by ring
  rw [h, Int.ediv_one, Int.toNat_natCast]

/-- The induction variable of step `t`: `int32`, value `t`. -/
theorem loopIdx_toInt {d : Inst} (hd : d.ok) {t : Nat} (ht : t ≤ d.K / 128) :
    (IntV.ofInt (0 + (t : Int) * 1) : I32).toInt = (t : Int) := by
  have h : (0 : Int) + (t : Int) * 1 = (t : Int) := by ring
  rw [h]
  refine IntV.I32.toInt_ofInt ⟨by omega, ?_⟩
  have h1 := divK_le hd
  have h2 : (t : Int) ≤ 1024 := by exact_mod_cast le_trans ht h1
  linarith

/-! ## The counters -/

theorem mul128_toInt {a : I32} {x : Nat} (ha : a.toInt = (x : Int)) (h : x * 128 < 2 ^ 31) :
    (a * (IntV.ofNat 128 : I32)).toInt = ((x * 128 : Nat) : Int) := by
  refine IntV.I32.mul_nat ha ?_ h
  exact IntV.I32.toInt_ofNat (by norm_num)

/-- `a * 256` in `int32` does not wrap when the product is below `2^31`. -/
theorem mul256_toInt {a : I32} {x : Nat} (ha : a.toInt = (x : Int)) (h : x * 256 < 2 ^ 31) :
    (a * (IntV.ofNat 256 : I32)).toInt = ((x * 256 : Nat) : Int) := by
  refine IntV.I32.mul_nat ha ?_ h
  exact IntV.I32.toInt_ofNat (by norm_num)

/-- `a + 128` in `int32` does not wrap when the sum is below `2^31`. -/
theorem add128_toInt {a : I32} {x : Nat} (ha : a.toInt = (x : Int)) (h : x + 128 < 2 ^ 31) :
    (a + (IntV.ofNat 128 : I32)).toInt = ((x + 128 : Nat) : Int) := by
  refine IntV.I32.add_nat ha ?_ h
  exact IntV.I32.toInt_ofNat (by norm_num)

end Claims.BlockScaledMxfp8
