import Claim
import TVA.Proof
import Proof.Setup
import Proof.Addrs
import Proof.Argmax
import Proof.Const

/-!
# What the block's two argmaxes return

The first `tl.argmaxAll true true` picks `i0` (the largest soft-capped logit, smallest index
among equals); the second, over the same row with `i0` replaced by `-inf`, picks `i1`. The
`tl.maxAll` gives the value `top1v` at `i0` exactly (`FloatV.MaxRel` returns one of the row's
entries).

The masked row is `tl.where (map (fun k => !k.eq pair1.2) (tl.arange 0 8)) capped (bc -inf)`:
`Tensor.get_where`, `Tensor.get_map`, `Tensor.get1_arange` and `IntV.eq` (`Scalar.lean:63`,
`a.bits == b.bits`) reduce it to `fun j => if j = i0 then -inf else capped j`, with `i0` from
`argRes_finite` and `-inf` the constant `c32 0xFFF0000000000000` (`Const.cninf_isNaN`,
`Const.cninf_sortKey_lt`). Then `argRes_masked` (`Proof.Argmax`) gives the order facts for `i1`.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

private theorem intv_eq_iff (a b : I32) : (IntV.eq a b = true ↔ a = b) := by
  unfold IntV.eq
  rw [beq_iff_eq]
  exact ⟨fun h => by cases a; cases b; simp_all, fun h => by rw [h]⟩

/-- `IntV.eq` on `int32` values built from natural numbers below `2^32` is equality. -/
private theorem intv_eq_ofNat {j k : Nat} (hj : j < 2 ^ 32) (hk : k < 2 ^ 32) :
    (IntV.eq (IntV.ofNat j : I32) (IntV.ofNat k) = true ↔ j = k) := by
  rw [intv_eq_iff]
  constructor
  · intro h
    have h2 : (BitVec.ofNat 32 j).toNat = (BitVec.ofNat 32 k).toNat :=
      congrArg (fun x : I32 => x.bits.toNat) h
    rwa [BitVec.toNat_ofNat, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hj, Nat.mod_eq_of_lt hk] at h2
  · intro h; rw [h]

/-- The `tl.arange` entry of the mask, `IntV.ofInt (0 + j)`, as `IntV.ofNat j`. -/
private theorem ofInt_zero_add {s : Bool} {w : Nat} (n : Nat) :
    (IntV.ofInt (((0 : Nat) : Int) + (n : Int)) : IntV s w) = IntV.ofNat n := by
  simp [IntV.ofInt, IntV.ofNat]

/-! `get1` of the tensors the mask is built from (the `.get` versions need `Idx []`, which the
`rewrite` and `simp` tactics do not unfold). -/

private theorem get1_where {n : Nat} {α : Type} (c : Tensor [n] Bool) (x y : Tensor [n] α)
    (i : Fin n) : (tl.where c x y).get1 i = if c.get1 i then x.get1 i else y.get1 i := rfl

private theorem get1_map {n : Nat} {α β : Type} (f : α → β) (x : Tensor [n] α) (i : Fin n) :
    (x.map f).get1 i = f (x.get1 i) := rfl

private theorem get1_tl_arange (start stop : Nat) (i : Fin (stop - start)) :
    (tl.arange start stop).get1 i = (IntV.ofInt ((start : Int) + (i.val : Int)) : I32) := rfl

private theorem get1_full' {n : Nat} {α : Type} (a : α) (i : Fin n) :
    (Tensor.full a : Tensor [n] α).get1 i = a := rfl

/-- The two argmax calls return the first largest capped logit and the first largest among the
others, and the second top value is that logit. -/
theorem ids
    {capped : Tensor [8 - 0] (FloatV Format.binary32)}
    (hfin : ∀ j : Fin (8 - 0), (capped.get1 j).isFinite = true)
    {pair1 : FloatV Format.binary32 × I32}
    (hpair1 : tl.ArgRes true true (8 - 0)
      (fun j => if h : j < 8 - 0 then capped.get1 ⟨j, h⟩ else default) pair1)
    {top1v : FloatV Format.binary32}
    (htop1v : ∃ tr, Tree.Covers (8 - 0) tr ∧
      Tree.Eval (FloatV.MaxRel true)
        (fun j => if h : j < 8 - 0 then capped.get1 ⟨j, h⟩ else default) tr top1v)
    {pair2 : FloatV Format.binary32 × I32}
    (hpair2 : tl.ArgRes true true (8 - 0)
      (fun j => if h : j < 8 - 0 then
        (tl.where (Tensor.map (fun k => !k.eq pair1.2) (tl.arange 0 8)) capped
          (bc (c32 18442240474082181120))).get1 ⟨j, h⟩ else default) pair2) :
    ∃ i0 i1 : Fin (8 - 0), IntV.ofNat i0.val = pair1.2 ∧ i1 ≠ i0 ∧
      IntV.ofNat i1.val = pair2.2 ∧
      (∀ j : Fin (8 - 0), (capped.get1 j).toReal ≤ (capped.get1 i0).toReal) ∧
      (∀ j : Fin (8 - 0), j.val < i0.val →
        (capped.get1 j).toReal < (capped.get1 i0).toReal) ∧
      (∀ j : Fin (8 - 0), j ≠ i0 →
        (capped.get1 j).toReal ≤ (capped.get1 i1).toReal) ∧
      (∀ j : Fin (8 - 0), j.val < i1.val → j ≠ i0 →
        (capped.get1 j).toReal < (capped.get1 i1).toReal) ∧
      top1v.isFinite = true ∧ top1v.toReal = (capped.get1 i0).toReal := by
  set g : Nat → F32 := fun j => if h : j < 8 - 0 then capped.get1 ⟨j, h⟩ else default with hg
  have hg1 : ∀ j, g j = (if h : j < 8 - 0 then capped.get1 ⟨j, h⟩ else default) := by
    intro j; rw [hg]
  have hg2 : ∀ (j : Nat) (h : j < 8 - 0), g j = capped.get1 ⟨j, h⟩ := by
    intro j h; rw [hg1 j]; simp only [dite_eq_left h]
  have hg3 : ∀ j, ¬ (j < 8 - 0) → g j = default := by
    intro j h; rw [hg1 j]; simp only [dite_eq_right h]
  have hfin' : ∀ j, j < 8 - 0 → (g j).isFinite = true := by
    intro j hj; rw [hg2 j hj]; exact hfin ⟨j, hj⟩
  -- The first argmax
  obtain ⟨i0', hi0', hk1, hle0, hlt0⟩ :=
    argRes_finite (n := 8 - 0) (by norm_num) hfin' hpair1
  -- The maximum value
  have hmax : ∃ tr, Tree.Covers (8 - 0) tr ∧
      Tree.Eval (FloatV.MaxRel true) g tr top1v := htop1v
  obtain ⟨tr, hcov, hev⟩ := htop1v
  have hmem : ∀ a, a ∈ tr.leaves ↔ a < 8 - 0 := fun a => by
    rw [hcov.mem_iff, List.mem_range]
  have hno : ∀ j ∈ tr.leaves, (g j).isNaN = false := fun j hj =>
    Format.not_isNaN_of_isFinite (hfin' j ((hmem j).mp hj))
  obtain ⟨⟨i', hi'mem, hm'⟩, _hall⟩ := Tree.eval_maxRel hev hno
  have hi'lt : i' < 8 - 0 := (hmem i').mp hi'mem
  have hfin_top : top1v.isFinite = true := by
    have h1 : top1v = g i' := hm'
    rw [h1]
    exact hfin' i' hi'lt
  have hval_top : top1v.toReal = (capped.get1 ⟨i0', hi0'⟩).toReal := by
    have h := maxAll_eq (n := 8 - 0) (by norm_num) hfin' hmax hi0' hle0
    rwa [hg2 i0' hi0'] at h
  -- The masked row
  have hrow : ∀ (j : Nat) (hj : j < 8 - 0),
      (tl.where (Tensor.map (fun k => !k.eq pair1.2) (tl.arange 0 8)) capped
        (bc (c32 18442240474082181120))).get1 ⟨j, hj⟩
      = if j = i0' then c32 18442240474082181120 else capped.get1 ⟨j, hj⟩ := by
    intro j hj
    rw [get1_where, get1_map, get1_tl_arange, get1_full', ofInt_zero_add, hk1]
    by_cases h : j = i0'
    · subst h
      rw [ite_eq_left rfl]
      have he : IntV.eq (IntV.ofNat j : I32) (IntV.ofNat j) = true :=
        (intv_eq_iff _ _).mpr rfl
      rw [he]
      simp
    · rw [ite_eq_right h]
      have hne : IntV.eq (IntV.ofNat j : I32) (IntV.ofNat i0') = false := by
        cases hb : IntV.eq (IntV.ofNat j : I32) (IntV.ofNat i0')
        · rfl
        · exact absurd ((intv_eq_ofNat (by omega) (by omega)).mp hb) h
      rw [hne]
      simp
  have hfun : (fun j => if h : j < 8 - 0 then
        (tl.where (Tensor.map (fun k => !k.eq pair1.2) (tl.arange 0 8)) capped
          (bc (c32 18442240474082181120))).get1 ⟨j, h⟩ else default)
      = (fun j => if j = i0' then c32 18442240474082181120 else g j) := by
    funext j
    by_cases hj : j < 8 - 0
    · rw [dite_eq_left hj, hrow j hj]
      by_cases h : j = i0'
      · rw [ite_eq_left h, ite_eq_left h]
      · rw [ite_eq_right h, ite_eq_right h]
        exact (hg2 j hj).symm
    · have h1 : ¬ (j = i0') := fun h => hj (h ▸ hi0')
      rw [dite_eq_right hj, ite_eq_right h1, hg3 j hj]
  have hpair2' : tl.ArgRes true true (8 - 0)
      (fun j => if j = i0' then c32 18442240474082181120 else g j) pair2 := by
    rw [← hfun]
    exact hpair2
  -- The second argmax
  obtain ⟨i1', hi1', hne1, hk2, hle1, hlt1⟩ :=
    argRes_masked (n := 8 - 0) (by norm_num) hfin'
      cninf_isNaN (fun j hj => cninf_sortKey_lt (hfin' j hj)) hi0' hpair2'
  set i0 : Fin (8 - 0) := ⟨i0', hi0'⟩ with hi0def
  set i1 : Fin (8 - 0) := ⟨i1', hi1'⟩ with hi1def
  have hval0 : i0.val = i0' := rfl
  have hval1 : i1.val = i1' := rfl
  refine ⟨i0, i1, hk1.symm, ?_, hk2.symm, ?_, ?_, ?_, ?_, hfin_top, hval_top⟩
  · intro h
    exact hne1 (by simpa [hval0, hval1] using congrArg Fin.val h)
  · intro j
    have h := hle0 j.val j.isLt
    rwa [hg2 j.val j.isLt, hg2 i0' hi0'] at h
  · intro j hj
    have hj' : j.val < i0' := by simpa [hval0] using hj
    have h := hlt0 j.val hj'
    rwa [hg2 j.val j.isLt, hg2 i0' hi0'] at h
  · intro j hjne
    have hjne' : j.val ≠ i0' := by
      intro h
      exact hjne (Fin.ext (by simpa [hval0] using h))
    have h := hle1 j.val j.isLt hjne'
    rwa [hg2 j.val j.isLt, hg2 i1' hi1'] at h
  · intro j hj hjne
    have hjne' : j.val ≠ i0' := by
      intro h
      exact hjne (Fin.ext (by simpa [hval0] using h))
    have hj' : j.val < i1' := by simpa [hval1] using hj
    have h := hlt1 j.val hj' hjne'
    rwa [hg2 j.val j.isLt, hg2 i1' hi1'] at h

end Claims.MoeRouter
