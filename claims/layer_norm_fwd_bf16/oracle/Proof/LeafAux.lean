import Proof.ValFacts

/-! Small generic facts the row's leaf needs: the fused addition relation weakened to the plain
one, a reduction tree's relation lifted along a pointwise implication, the value of the masked
lane's zero constant, and the magnitude bounds that let a rounded `float32` stay below `2^57`. -/

namespace TVA


noncomputable section

/-- **A fused addition is a plain one** (`AddRel`'s fused cases are guarded by `fusion = true`). -/
theorem AddRel.of_true {f : Format} {fusion : Bool} {x y r : FloatV f}
    (h : AddRel fusion x y r) : AddRel true x y r := by
  unfold AddRel at h ⊢
  rcases h with h | ⟨hf, h⟩
  · exact Or.inl h
  · obtain rfl := hf
    exact Or.inr ⟨rfl, h⟩

/-- **Lifting a reduction tree along a pointwise implication** of its combination relations. -/
theorem Tree.Eval.mono {α : Type} {op op' : α → α → α → Prop} {x : Nat → α}
    (h : ∀ a b c, op a b c → op' a b c) :
    ∀ {t : Tree} {r : α}, t.Eval op x r → t.Eval op' x r := by
  intro t r he
  induction he with
  | leaf i => exact .leaf i
  | node _ _ hc ih1 ih2 => exact .node ih1 ih2 (h _ _ _ hc)

end

end TVA

namespace Claims.LayerNormFwd

open TVA
open Format

noncomputable section

namespace FloatV

/-- The `float32` constant of the zero `float64` is the zero pattern. -/
theorem litF32_ofPat_zero :
    FloatV.litF32 (FloatV.ofPat (0 : ℕ) : F64) = (FloatV.ofPat (0 : ℕ) : F32) := by
  unfold FloatV.litF32
  rw [ite_eq_left]
  unfold FloatV.isZero
  rw [FloatV.pat_ofPat (Nat.two_pow_pos Format.binary64.width)]
  exact Format.isZero_zero

end FloatV

/-- **The `float32 → bfloat16` conversion relation is the conversion** (up to NaN payload): the
only applicable case of `FloatRel` for this pair of standard formats. -/
theorem ConvRel.f32_bf16_cvt {P : Platform} {x : F32} {r : BF16}
    (h : ConvRel P .f32 .bf16 x r) : FloatV.NaNEq (FloatV.cvt x) r := by
  have h' : FloatRel P Format.binary32 Format.bfloat16 x r := h
  unfold FloatRel at h'
  rw [ite_eq_right (by decide : ¬ (Format.binary32 = Format.bfloat16))] at h'
  rcases h' with ⟨-, -, h'⟩ | ⟨-, h', -⟩ | ⟨h', -⟩
  · exact h'
  · rcases h' with h' | h' <;> exact absurd h' (by decide)
  · rcases h' with h' | h' <;> exact absurd h' (by decide)

/-- **The masked lane's value is zero**: a converted zero constant of `bfloat16` is a finite
zero. -/
theorem ConvRel.f32_bf16_zero {P : Platform} {r : BF16}
    (h : ConvRel P .f32 .bf16 (FloatV.ofPat (0 : ℕ) : F32) r) :
    r.toReal = 0 ∧ r.isFinite = true := by
  have hav := ConvRel.f32_bf16_cvt h
  have hfin : (FloatV.cvt (FloatV.ofPat (0 : ℕ) : F32) : BF16).isFinite = true :=
    FloatV.isFinite_cvt_zero Format.binary32_regular Format.bfloat16_regular
  have heq : r = FloatV.cvt (FloatV.ofPat (0 : ℕ) : F32) :=
    FloatV.NaNEq.eq_of_isFinite hav hfin
  refine ⟨?_, ?_⟩
  · rw [heq]; exact FloatV.toReal_cvt_zero Format.binary32_regular
  · rw [heq]; exact hfin

/-- An addition of two finite zeros has value zero. -/
theorem FloatV.toRat_add_of_zero {f : Format} (_hf : f.Regular) {x y : FloatV f}
    (hxfin : x.isFinite = true) (hyfin : y.isFinite = true)
    (hx : x.toRat = 0) (hy : y.toRat = 0) : (FloatV.add x y).toRat = 0 := by
  unfold FloatV.add FloatV.toRat at *
  rw [Format.add_of_finite .rn hxfin hyfin, hx, hy, add_zero]
  rw [Format.roundRat_zero]
  rw [FloatV.pat_ofPat
    (Format.setSign_lt (f.sign x.pat && f.sign y.pat) 0)]
  exact Format.toRat_setSign_zero f _

/-- An addition of two finite zeros is zero. -/
theorem FloatV.toReal_add_of_zero {f : Format} (hf : f.Regular) {x y : FloatV f}
    (hxfin : x.isFinite = true) (hyfin : y.isFinite = true)
    (hx : x.toRat = 0) (hy : y.toRat = 0) : (FloatV.add x y).toReal = 0 := by
  unfold FloatV.toReal
  rw [FloatV.toRat_add_of_zero hf hxfin hyfin hx hy]
  norm_num

/-- **Reading through the model of a read**: the `rval` of a present word is that word. -/
theorem rval_of_read {m : Mem} {p : Buf .bf16} {i : Nat} {v : BF16}
    (h : m.read .bf16 (p.ptr (i : Int)).addr = some v) : rval m p i = v := by
  unfold rval
  rw [h]

/-- A buffer word's own address is in the set of its elements. -/
theorem addr_mem_elems {dty : DType} {b : Buf dty} {S : Set Nat} {j : Nat} (hj : j ∈ S) :
    (b.ptr (j : Int)).addr ∈ b.elems S := by
  refine Buf.span_sub_elems b hj ?_
  simpa using Mem.mem_span (d := dty) (a := (b.ptr (j : Int)).addr) (i := 0)
    (DType.bytes_pos dty)

/-- **A read that stays in one buffer is unaffected by a write to a disjoint buffer.** -/
theorem read_frame {dty : DType} {B : Buf dty} {S : Set Nat} {j : Nat} (hj : j ∈ S)
    {dty' : DType} {C : Buf dty'} {T : Set Nat}
    (hdj : Disjoint (B.elems S) (C.elems T))
    {m m' : Mem} (hframe : ∀ x ∉ C.elems T, m' x = m x) :
    m'.read dty (B.ptr (j : Int)).addr = m.read dty (B.ptr (j : Int)).addr := by
  refine Mem.read_congr (fun x hx => ?_)
  exact hframe x
    (not_mem_right hdj (Buf.span_sub_elems B hj) (subset_refl (C.elems T)) hx)

/-- **Element sets shrink with their index sets.** -/
theorem elems_disjoint_mono {dty dty' : DType} {B : Buf dty} {C : Buf dty'} {S T S' T' : Set Nat}
    (h : Disjoint (B.elems S) (C.elems T)) (hS : S' ⊆ S) (hT : T' ⊆ T) :
    Disjoint (B.elems S') (C.elems T') :=
  Disjoint.mono (Buf.elems_mono B hS) (Buf.elems_mono C hT) h

/-- The lane `j` of row `t` is in the block image of its program instance. -/
theorem mem_img_mat {d : Inst} {p t j : Nat} (hp : t / 32 = p) (ht : t < d.T)
    (hj : j < d.D) (hjBD : j < d.BD) :
    idx d t j ∈ Buf.img (fun ix : Idx [32, d.BD] => idx d (32 * p + (ix.1 : Nat)) (ix.2.1 : Nat))
      (fun ix : Idx [32, d.BD] => decide (32 * p + (ix.1 : Nat) < d.T ∧ (ix.2.1 : Nat) < d.D)) := by
  have hrmod : t % 32 < 32 := Nat.mod_lt _ (by norm_num)
  have h32 : 32 * p + t % 32 = t := by
    rw [← hp]
    exact Nat.div_add_mod t 32
  refine ⟨⟨⟨t % 32, hrmod⟩, ⟨j, hjBD⟩, ()⟩, ?_, ?_⟩
  · dsimp only
    rw [h32]
    simp [ht, hj]
  · dsimp only
    rw [h32]

/-- Row `t` is in the row image of its program instance. -/
theorem mem_img_row {d : Inst} {p t : Nat} (hp : t / 32 = p) (ht : t < d.T) :
    t ∈ Buf.img (fun ix : Idx [32] => 32 * p + (ix.1 : Nat))
      (fun ix : Idx [32] => decide (32 * p + (ix.1 : Nat) < d.T)) := by
  have hrmod : t % 32 < 32 := Nat.mod_lt _ (by norm_num)
  have h32 : 32 * p + t % 32 = t := by
    rw [← hp]
    exact Nat.div_add_mod t 32
  refine ⟨⟨⟨t % 32, hrmod⟩, ()⟩, ?_, ?_⟩
  · dsimp only
    rw [h32]
    exact decide_eq_true ht
  · dsimp only
    rw [h32]

/-- `2^57` is a finite `float32`. -/
theorem two_pow_57_le_binary32_maxVal : (2 : ℚ) ^ 57 ≤ Format.binary32.maxVal := by
  rw [Format.binary32_maxVal]; norm_num

/-- `2^57` lies above `float32`'s smallest exponent. -/
theorem binary32_emin_le_57 : Format.binary32.emin ≤ (57 : ℤ) := by
  decide

end

end Claims.LayerNormFwd
