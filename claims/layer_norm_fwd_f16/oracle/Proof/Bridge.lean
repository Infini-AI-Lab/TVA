import Claim
import TVA.Proof
import Proof.Row
import Proof.Lanes

/-!
# The claim's row as a `RowP`

`rowP` packs row `t` of a covered launch into the plain-real data of `Proof.Row`; the claim's
quantities (`meanRef`, `meanTol`, `rstdRef`, ...) are then those of `Proof.Row` (`rfl`).
-/

namespace Claims.LayerNormFwdF16

open TVA

/-- A finite `float16` value is at most `65504` in magnitude. -/
theorem abs_f16_le {x : F16} (hx : x.isFinite = true) : |x.toReal| ≤ 65504 := by
  have := FloatV.abs_toReal_le_maxVal Format.binary16_regular hx
  rwa [Format.binary16_maxVal_real] at this

/-- The block size is `2^depth`. -/
theorem Inst.ok.block_eq {d : Inst} (hd : d.ok) : d.BLOCK = 2 ^ depth d := by
  obtain ⟨e, he⟩ := hd.block_pow2
  rw [depth, he, Nat.clog_pow _ _ (by norm_num)]

/-- Row `t` of a covered launch, as plain real data. -/
noncomputable def rowP (d : Inst) (hd : d.ok) (init : Mem) (hpre : Pre d init) (t : Nat)
    (ht : t < d.M) : RowP where
  N := d.N
  B := d.BLOCK
  D := depth d
  block_eq := hd.block_eq
  x := xR d init t
  eps := epsR d
  n_ge := by have := hd.n_dim.1; have := hd.n_ge; omega
  n_le := hd.n_le
  n_le_block := hd.n_le_block
  block_lt := hd.block_lt
  x_le := fun j hj => abs_f16_le (hpre.2.2.2.2.2.2.1 t ht j hj)
  eps_ge := by
    have := hd.eps_ge
    unfold epsR FloatV.toReal
    have h := (Rat.cast_le (K := ℝ)).mpr this
    push_cast at h
    exact h
  eps_le := by
    have := hd.eps_le
    unfold epsR FloatV.toReal
    have h := (Rat.cast_le (K := ℝ)).mpr this
    push_cast at h
    norm_num at h ⊢
    exact h

end Claims.LayerNormFwdF16
