import Claim
import TVA.Proof
import Proof.Basic
import Proof.Inv
import Proof.Arith
import Proof.Off
import Proof.Foot
import Proof.Lane
import Proof.Tol
import Proof.Body
import Proof.Loop

/-!
# One program of the grid, from `init`

`prog_safe` walks the whole body of the kernel: the two `int32` divisions of the prologue (the
`tl.irem` and the `tl.idiv`), the offsets they give, and then the loop and the epilogue
(`loop_safe`). The program's write set is its own tile of `c` (`W d i`), which the epilogue's
store produces; the loop itself only reads.

The walk needs four facts about the `int32` products of the prologue: the row and column of the
program are below `2^31` and their products with `128` and `256` stay inside `int32`.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-! ## The prologue's products -/

/-- `(i % gridM d) * 128` is inside `int32`. -/
theorem offm128_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ((i : Nat) % gridM d) * 128 < 2 ^ 31 := by
  have h1 : (i : Nat) % gridM d < gridM d := Nat.mod_lt _ (gridM_pos hd)
  have h2 : gridM d * 128 ≤ d.M := by
    rw [gridM_eq hd]
    exact Nat.div_mul_le_self d.M 128
  have h3 : ((i : Nat) % gridM d) * 128 < gridM d * 128 :=
    Nat.mul_lt_mul_of_pos_right h1 (by norm_num)
  have h4 := hd.M_ok.2.2
  omega

/-- `(i / gridM d) * 256` is inside `int32`. -/
theorem offn256_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ((i : Nat) / gridM d) * 256 < 2 ^ 31 := by
  have hg : 0 < gridM d := gridM_pos hd
  have hi : (i : Nat) < grid d := by simpa only [launch_size] using i.isLt
  have h1 : (i : Nat) / gridM d < gridN d := by
    rw [Nat.div_lt_iff_lt_mul hg]
    show (i : Nat) < gridN d * gridM d
    rw [Nat.mul_comm]
    exact hi
  have h2 : gridN d * 256 ≤ d.N + 255 := by
    unfold gridN
    exact Nat.div_mul_le_self (d.N + 255) 256
  have h3 : ((i : Nat) / gridM d) * 256 < gridN d * 256 :=
    Nat.mul_lt_mul_of_pos_right h1 (by norm_num)
  have h4 := hd.N_ok.2.2
  omega

/-- The row of program `i` is below `2^31`. -/
theorem offm_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ((i : Nat) % gridM d) < 2 ^ 31 := by
  have h1 : (i : Nat) % gridM d < gridM d := Nat.mod_lt _ (gridM_pos hd)
  have h2 := gridM_lt hd
  omega

/-- The column of program `i` is below `2^31`. -/
theorem offn_lt {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    ((i : Nat) / gridM d) < 2 ^ 31 := by
  have h1 := offn256_lt hd i
  omega

/-- An `int32` times the `int32` `1`. -/
theorem mul1_toInt {a : I32} {x : Nat} (ha : a.toInt = (x : Int)) (h : x < 2 ^ 31) :
    (a * (IntV.ofNat 1 : I32)).toInt = (x : Int) := by
  have h' := IntV.I32.mul_nat (a := a) (b := (IntV.ofNat 1 : I32)) ha
    (IntV.I32.toInt_ofNat (by norm_num)) (by simpa using h)
  simpa using h'

/-- An `int32` times the `int32` `2`. -/
theorem mul2_toInt {a : I32} {x : Nat} (ha : a.toInt = (x : Int)) (h : x * 2 < 2 ^ 31) :
    (a * (IntV.ofNat 2 : I32)).toInt = ((2 * x : Nat) : Int) := by
  have h' := IntV.I32.mul_nat (a := a) (b := (IntV.ofNat 2 : I32)) ha
    (IntV.I32.toInt_ofNat (by norm_num)) h
  simpa [Nat.mul_comm] using h'

/-! ## The walk -/

/-- **One program.** Program `i` of the grid, alone from `init`, is safe, reads the inputs,
writes its tile of `c`, and leaves the rounded result there. -/
theorem prog_safe (P : Platform) {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size)
    (hdot : Device.DotScaledE4m3 P) (hdotm : Device.DotScaledE4m3E2m1 P)
    (htot : Device.DotScaledTotal P cfgMx) (htotm : Device.DotScaledTotal P cfgMixed) :
    Safe ((launch d).env P) (R d i) (W d i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hpid : (tl.program_id ((launch d).ctx i) 0).toInt = ((i : Nat) : Int) := pid_toInt hd i
  have hnum : (IntV.ofNat ((d.M + (128 - 1)) / 128) : I32).toInt = ((gridM d : Nat) : Int) := by
    show (IntV.ofNat (gridM d) : I32).toInt = ((gridM d : Nat) : Int)
    exact ofNat_toInt (gridM_lt hd)
  unfold model block_scaled_matmul_kernel
  refine safe_irem1 (x := tl.program_id ((launch d).ctx i) 0)
    (y := (IntV.ofNat ((d.M + (128 - 1)) / 128) : I32)) ?_ ?_ ?_
  · rw [hpid]; norm_num
  · rw [hnum]; exact_mod_cast gridM_pos hd
  · intro r1 hr1
    refine safe_idiv1 (x := tl.program_id ((launch d).ctx i) 0)
      (y := (IntV.ofNat ((d.M + (128 - 1)) / 128) : I32)) ?_ ?_ ?_
    · rw [hpid]; norm_num
    · rw [hnum]; exact_mod_cast gridM_pos hd
    · intro r2 hr2
      have hrm : r1.item.toInt = (((i : Nat) % gridM d : Nat) : Int) := by
        rw [hr1, hpid, hnum, Int.natCast_emod]
      have hrn : r2.item.toInt = (((i : Nat) / gridM d : Nat) : Int) := by
        rw [hr2, hpid, hnum, Int.natCast_ediv]
      refine loop_safe P hd hpre i hdot hdotm htot htotm (am := r1.item * IntV.ofNat 128)
        (bn := r2.item * IntV.ofNat 256) (sm := r1.item * IntV.ofNat 1)
        (sn := r2.item * IntV.ofNat 2) ?_ ?_ ?_ ?_
      · exact mul128_toInt hrm (offm128_lt hd i)
      · exact mul256_toInt hrn (offn256_lt hd i)
      · exact mul1_toInt hrm (offm_lt hd i)
      · exact mul2_toInt hrn (by have h := offn256_lt hd i; omega)

end Claims.BlockScaledMxfp8
