import Claim
import TVA.Proof
import Proof.Basic
import Proof.Tol
import Proof.Num

/-!
# One step of the loop

The accumulator's error after `t` steps of `tl.dot_scaled` bounds it after `t + 1`, the room
condition the library's rule asks for follows from `ValsOk.room`, and at the end of the loop the
accumulator is within `vTol` of the exact result.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-- Step `t`'s products, summed over `Fin 128`, are `chunkSum p t`. -/
theorem sum_chunk {d : Inst} {m : Mem} {i j : Nat} (t : Nat) :
    (∑ k : Fin 128, prods d m i j (128 * t + (k : Nat))) = chunkSum (prods d m i j) t :=
  (Fin.sum_univ_eq_sum_range (fun k => prods d m i j (128 * t + k)) 128).trans rfl

/-- Step `t`'s products' magnitudes, summed over `Fin 128`, are `chunkAbs p t`. -/
theorem sum_abs_chunk {d : Inst} {m : Mem} {i j : Nat} (t : Nat) :
    (∑ k : Fin 128, |prods d m i j (128 * t + (k : Nat))|) = chunkAbs (prods d m i j) t :=
  (Fin.sum_univ_eq_sum_range (fun k => |prods d m i j (128 * t + k)|) 128).trans rfl

/-- **One loop step**: if the accumulator is within `tol p t` of `partSum p t`, one
`tl.dot_scaled` step (with the library rule's error) leaves it within `tol p (t + 1)`. -/
theorem step_inv {p : Nat → ℚ} {t : Nat} {acc acc' cs : ℚ}
    (hinv : |acc - partSum p t| ≤ tol p t) (hcs : cs = chunkSum p t)
    (hstep : |acc' - (acc + cs)| ≤ rho * (|acc| + chunkAbs p t) + alpha) :
    |acc' - partSum p (t + 1)| ≤ tol p (t + 1) := by
  have hsucc : partSum p (t + 1) = partSum p t + chunkSum p t := partSum_succ p t
  have hsplit : acc' - partSum p (t + 1) = (acc' - (acc + cs)) + (acc - partSum p t) := by
    rw [hsucc, hcs]; ring
  have h1 : |acc' - partSum p (t + 1)| ≤ |acc' - (acc + cs)| + |acc - partSum p t| := by
    rw [hsplit]; exact abs_add_le _ _
  have ha : |acc| ≤ |partSum p t| + tol p t := abs_le_of_abs_sub_le hinv
  have hchunk : 0 ≤ chunkAbs p t := by
    simp only [chunkAbs]
    exact Finset.sum_nonneg (fun e _ => abs_nonneg _)
  have hρ : (0:ℚ) ≤ rho := le_of_lt rho_pos
  have hmul : rho * (|acc| + chunkAbs p t) ≤ rho * (|partSum p t| + tol p t + chunkAbs p t) :=
    mul_le_mul_of_nonneg_left (by linarith) hρ
  have htol : tol p (t + 1) =
      tol p t + rho * (|partSum p t| + tol p t + chunkAbs p t) + alpha := by
    simp only [tol]
  rw [htol]
  linarith

/-- **The room of one step**: an accumulator within `tol p t` of `partSum p t` leaves room in
`float32` for one more step of `128` products, when the output element has room (`ValsOk.room`). -/
theorem room_step {p : Nat → ℚ} {t K : Nat} {acc : ℚ}
    (hinv : |acc - partSum p t| ≤ tol p t) (htK : 128 * (t + 1) ≤ K) (htT : t ≤ K / 128)
    (hroom : dotAbs p K + tol p (K / 128) ≤ 2 ^ (127 : ℤ)) :
    |acc| + chunkAbs p t + (rho * (|acc| + chunkAbs p t) + alpha) ≤
      Format.binary32.maxVal := by
  have ha : |acc| ≤ |partSum p t| + tol p t := abs_le_of_abs_sub_le hinv
  have hmag : |partSum p t| + chunkAbs p t ≤ dotAbs p K := mag_le_dotAbs p htK
  have ht2 : tol p t ≤ tol p (K / 128) := tol_mono p htT
  have hchunk : 0 ≤ chunkAbs p t := by
    simp only [chunkAbs]
    exact Finset.sum_nonneg (fun e _ => abs_nonneg _)
  have hX : |acc| + chunkAbs p t ≤ 2 ^ (127 : ℤ) := by linarith
  exact room_le hX

/-- **The end of the loop**: an accumulator within `tol` of `partSum` after `K / 128` steps is
within `vTol` of the exact result. -/
theorem acc_ref_bound {d : Inst} (hd : d.ok) {m : Mem} {i j : Nat} {acc : ℚ}
    (hinv : |acc - partSum (prods d m i j) (d.K / 128)| ≤ tol (prods d m i j) (d.K / 128)) :
    |acc - ref d m i j| ≤ vTol d m i j := by
  have hK : d.K ≤ 2 ^ 17 := hd.K_ok.2.2
  have hT : d.K / 128 ≤ 2 ^ 10 := by
    have h : (2:Nat) ^ 17 = 131072 := by norm_num
    omega
  have href : partSum (prods d m i j) (d.K / 128) = ref d m i j := partSum_ref hd _
  have htol := tol_le_vTol (prods d m i j) hT
  simp only [vTol, magOut] at htol ⊢
  rw [href] at hinv
  linarith

end Claims.BlockScaledMxfp8
