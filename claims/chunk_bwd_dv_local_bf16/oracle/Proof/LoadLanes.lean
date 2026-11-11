import Claim
import TVA.Proof
import Proof.Util

/-!
# Loads with `other = 0`

The `other` argument of every `tl.load` of the kernel is `tl.convert .f32 .bf16 (tl.full .. zf32)`,
the `bfloat16` `+0` in every lane (`safe_conv_zero`).
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The conversion of `+0` from `float32` to `bfloat16` is `+0`. -/
theorem bf16_cvt_zero : FloatV.cvt (g := .bfloat16) (FloatV.ofPat 0 : F32) = (FloatV.ofPat 0 : BF16) :=
  FloatV.cvt_ofPat_zero Format.binary32_regular

/-- Every `float32` converts to `bfloat16` on the platform (`tl.convert` allows the IEEE result). -/
theorem exists_conv_f32_bf16 {P : Platform} (x : F32) : ∃ r, ConvRel P .f32 .bf16 x r := by
  refine ⟨FloatV.cvt x, ?_⟩
  show FloatRel P .binary32 .bfloat16 x (FloatV.cvt x)
  unfold FloatRel
  rw [ite_eq_right (by decide)]
  exact Or.inl ⟨Format.binary32_std, Format.bfloat16_std, FloatV.NaNEq.refl _⟩

/-- `tl.convert .f32 .bf16` sends `+0` to `+0`. -/
theorem conv_bf16_zero {P : Platform} {r : BF16} (h : ConvRel P .f32 .bf16 (FloatV.ofPat 0) r) :
    r = FloatV.ofPat 0 := by
  have hF : FloatRel P .binary32 .bfloat16 (FloatV.ofPat 0) r := h
  unfold FloatRel at hF
  rw [ite_eq_right (by decide)] at hF
  rcases hF with ⟨-, -, h1⟩ | ⟨-, h2, -⟩ | ⟨h3, -⟩
  · exact (h1.eq (by rw [bf16_cvt_zero]; exact FloatV.isNaN_ofPat_zero Format.bfloat16_regular)).trans
      bf16_cvt_zero
  · exact absurd h2 (by intro hh; rcases hh with hh | hh <;> exact absurd hh (by decide))
  · exact absurd h3 (by intro hh; rcases hh with hh | hh <;> exact absurd hh (by decide))

/-- **The `other` argument**: `tl.convert .f32 .bf16 (tl.full s zf32)` is `+0` in every lane. -/
theorem safe_conv_zero {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
    {s : Shape} {k : Tensor s BF16 → Prog α}
    (hk : ∀ z : Tensor s BF16, (∀ i, z.get i = FloatV.ofPat 0) → Safe E R W Q m (k z)) :
    Safe E R W Q m (tl.convert .f32 .bf16 (tl.full s zf32) >>= k) := by
  refine safe_convert (fun i => ?_) fun z hz => hk z fun i => ?_
  · simpa only [tl.full, Tensor.get_full, zf32_eq] using exists_conv_f32_bf16 (P := E.P) (FloatV.ofPat 0 : F32)
  · refine conv_bf16_zero (P := E.P) ?_
    simpa only [tl.full, Tensor.get_full, zf32_eq] using hz i

end Claims.ChunkBwdDvLocal
