import TVA.Lang.Tensor
import Mathlib.Tactic.Ring

/-!
# Shape changes agree with Triton's definitions

`tl.interleave(a, b)` is documented as `tl.join(a, b).reshape(a.shape[:-1] + [2 * a.shape[-1]])`
(`standard.py:540-560`). The library defines it directly (`Tensor.interleave`), and
`interleave_eq_reshape_join` proves the two agree. `Idx.pos_snoc` is the row-major position of
an index with a last coordinate appended.
-/
namespace TVA

theorem Shape.prod_append_one (s : Shape) (n : Nat) : (s ++ [n]).prod = s.prod * n := by
  simp [List.prod_append]

theorem Idx.pos_snoc {n : Nat} : ∀ {s : Shape} (i : Idx s) (t : Fin n),
    Idx.pos (s ++ [n]) (Idx.snoc i t) = Idx.pos s i * n + t
  | [], (), t => by simp [Idx.pos, Idx.snoc]
  | m :: s, (i, j), t => by
      show i.val * (s ++ [n]).prod + Idx.pos (s ++ [n]) (Idx.snoc j t) = _
      rw [Idx.pos_snoc j t, Shape.prod_append_one]
      simp only [Idx.pos]
      ring

/-- `interleave` is `join` followed by the row-major reshape. -/
theorem Tensor.interleave_eq_reshape_join {s : Shape} {n : Nat} {α : Type} (a b : Tensor (s ++ [n]) α)
    (h : (s ++ [2 * n]).prod = ((s ++ [n]) ++ [2]).prod) :
    Tensor.interleave a b = (Tensor.join a b).reshape (s ++ [2 * n]) h := by
  ext idx
  obtain ⟨i, t, rfl⟩ : ∃ i t, idx = Idx.snoc i t := ⟨_, _, (Idx.snoc_unsnoc idx).symm⟩
  rw [Tensor.get_interleave_snoc, Tensor.get_reshape]
  have ht2 : t.val % 2 < 2 := Nat.mod_lt _ (by norm_num)
  have hpos : Idx.pos ((s ++ [n]) ++ [2])
      (Idx.snoc (Idx.snoc i ⟨t.val / 2, Nat.div_lt_of_lt_mul t.2⟩) ⟨t.val % 2, ht2⟩) =
      Idx.pos (s ++ [2 * n]) (Idx.snoc i t) := by
    rw [Idx.pos_snoc, Idx.pos_snoc, Idx.pos_snoc]
    simp only
    have h2 := Nat.div_add_mod t.val 2
    generalize t.val / 2 = q at h2 ⊢
    generalize t.val % 2 = r at h2 ⊢
    rw [← h2]
    ring
  rw [Idx.ofPos_congr hpos.symm _ (Idx.pos_lt _ _), Idx.ofPos_pos, Tensor.get_join_snoc]

end TVA
