import Proof.NumericA

/-!
# `lora_shrink_bf16`: the tolerances of the claim are nonnegative, and their closed forms

Signs of `errPart`, `partMag`, `vMag`, `atomTol`; the constants `rho`, `alpha` of the step-by-step
bound in both launch configurations; `(1 + x)^n - 1` for small `n·x`; the bound on the sum of the
parts' errors.
-/

namespace Claims.LoraShrink

open TVA

/-- The `bfloat16` lift of `0` is `0`, and lifted products are nonnegative. -/
theorem lift_prod_nonneg (x y : ℚ) :
    0 ≤ Device.liftMag .bf16 x * Device.liftMag .bf16 y :=
  mul_nonneg (Device.liftMag_nonneg _ _) (Device.liftMag_nonneg _ _)

/-- One chunk step of the error recursion is nonnegative. -/
theorem cStep_nonneg {e Pg mg Pg' : ℚ} (he : 0 ≤ e) : 0 ≤ cStep e Pg mg Pg' := by
  have h1 : 0 ≤ max |Pg| mg := (abs_nonneg _).trans (le_max_left _ _)
  have h2 := Dot16Chunk.rhoChunk16_nonneg
  have h3 := Dot16Chunk.sigmaChunk16_nonneg
  have h4 := Dot16Chunk.alphaChunk16_nonneg
  unfold cStep
  have : 0 ≤ (1 + Device.rhoChunk16 + Device.sigmaChunk16) * e := mul_nonneg (by linarith) he
  have : 0 ≤ Device.rhoChunk16 * max |Pg| mg := mul_nonneg h2 h1
  have : 0 ≤ Device.sigmaChunk16 * |Pg'| := mul_nonneg h3 (abs_nonneg _)
  linarith

/-- The chunk-step chain is nonnegative from a nonnegative start. -/
theorem cChain_nonneg (d : Inst) (va vb : Nat → BF16) (p g0 : Nat) (P : Nat → ℚ) {e : ℚ}
    (he : 0 ≤ e) : ∀ n, 0 ≤ cChain d va vb p g0 P e n
  | 0 => he
  | n + 1 => cStep_nonneg (cChain_nonneg d va vb p g0 P he n)

/-- The chunk-by-chunk error `accErr` is nonnegative. -/
theorem accErr_nonneg (d : Inst) (va vb : Nat → BF16) (p : Nat) : ∀ t, 0 ≤ accErr d va vb p t
  | 0 => le_rfl
  | t + 1 => by
    have := cChain_nonneg d va vb p (t * cpb d) (pSum d va vb p) (accErr_nonneg d va vb p t) (cpb d)
    exact this.trans (le_max_left _ _)

/-- The lifted magnitude of a part is nonnegative. -/
theorem partLift_nonneg (d : Inst) (va vb : Nat → BF16) (p : Nat) : 0 ≤ partLift d va vb p :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => lift_prod_nonneg _ _

/-- The step-by-step error bound is nonnegative. -/
theorem errStep_nonneg (d : Inst) (va vb : Nat → BF16) (p : Nat) : 0 ≤ errStep d va vb p :=
  DotLoop.bound_nonneg (rho_nonneg d) (alpha_nonneg d) _ (partLift_nonneg d va vb p)

/-- The bound on a part's accumulator is nonnegative. -/
theorem errPart_nonneg (d : Inst) (va vb : Nat → BF16) (p : Nat) : 0 ≤ errPart d va vb p := by
  unfold errPart errChunk
  exact le_min (accErr_nonneg d va vb p _) (errStep_nonneg d va vb p)

/-- `partMag` is nonnegative. -/
theorem partMag_nonneg (d : Inst) (va vb : Nat → BF16) : 0 ≤ partMag d va vb :=
  Finset.sum_nonneg fun p _ => add_nonneg (abs_nonneg _) (errPart_nonneg d va vb p)

/-- `vMag` is nonnegative. -/
theorem vMag_nonneg (d : Inst) (va vb : Nat → BF16) : 0 ≤ vMag d va vb := by
  have hu := Format.u_nonneg (f := Format.binary32)
  have he := Format.eta_nonneg (f := Format.binary32)
  have h := partMag_nonneg d va vb
  unfold vMag mulTol
  have : 0 ≤ |(scaling32 d).toRat| * partMag d va vb := mul_nonneg (abs_nonneg _) h
  have : 0 ≤ Format.binary32.u * |(scaling32 d).toRat| * partMag d va vb :=
    mul_nonneg (mul_nonneg hu (abs_nonneg _)) h
  have : 0 ≤ (sk d : ℚ) * Format.binary32.eta := mul_nonneg (Nat.cast_nonneg _) he
  linarith

/-- `atomTol` is nonnegative. -/
theorem atomTol_nonneg (d : Inst) (va vb : Nat → BF16) : 0 ≤ atomTol d va vb :=
  DotLoop.bound_nonneg (by positivity) (by positivity) _ (vMag_nonneg d va vb)

/-- `|partSum p| ≤ partLift p`. -/
theorem abs_partSum_le (d : Inst) (va vb : Nat → BF16) (p : Nat) :
    |partSum d va vb p| ≤ partLift d va vb p := by
  unfold partSum partLift cSum
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun q _ => ?_)
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ => ?_)
  exact Device.abs_mul_le_liftMag _ _ _ _

/-- The parts' lifted magnitudes add up to `dotMag`. -/
theorem sum_partLift (d : Inst) (hd : d.ok) (va vb : Nat → BF16) :
    ∑ p ∈ Finset.range (sk d), partLift d va vb p = dotMag d va vb :=
  partition_sum d hd.K_pos (fun k => Device.liftMag .bf16 (va k).toRat * Device.liftMag .bf16 (vb k).toRat)

/-- The parts add up to the dot product. -/
theorem dotSum_eq_parts (d : Inst) (hd : d.ok) (va vb : Nat → BF16) :
    dotSum d va vb = ∑ p ∈ Finset.range (sk d), partSum d va vb p :=
  (partition_sum d hd.K_pos (fun k => (va k).toRat * (vb k).toRat)).symm

/-- `|dotSum| ≤ dotMag`. -/
theorem abs_dotSum_le (d : Inst) (va vb : Nat → BF16) : |dotSum d va vb| ≤ dotMag d va vb := by
  unfold dotSum dotMag
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun k _ => Device.abs_mul_le_liftMag _ _ _ _)

/-- `dotMag` is nonnegative. -/
theorem dotMag_nonneg (d : Inst) (va vb : Nat → BF16) : 0 ≤ dotMag d va vb :=
  Finset.sum_nonneg fun _ _ => lift_prod_nonneg _ _

end Claims.LoraShrink
