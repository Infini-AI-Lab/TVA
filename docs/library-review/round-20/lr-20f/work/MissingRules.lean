import TVA
/-! lr-20f, round 20: the proof rules the kit lacks (the NEW gap), stated and proved on the library as it is:
`tl.convert` (no rule), `tl.sum0c` (`safe_sum1c` exists, not `safe_sum0c`), and `tl.store1`, which is
`tl.store` of the scalar tensors, so `safe_store` applies after a rewrite. -/
open TVA
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}

theorem safe_convert' {d d' : DType} {x : Tensor s d.Val} {k : Tensor s d'.Val → Prog α}
    (hex : ∀ i, ∃ r, ConvRel E.P d d' (x.get i) r)
    (hk : ∀ r : Tensor s d'.Val, (∀ i, ConvRel E.P d d' (x.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.convert d d' x >>= k) := by
  simp only [tl.convert, Prog.bind_eq, Prog.bind]
  refine .env (.pick ⟨⟨fun i => Classical.choose (hex i)⟩, fun i => Classical.choose_spec (hex i)⟩ ?_)
  intro r hr
  simpa [Prog.bind] using hk r hr

theorem safe_sum0c' {M N : Nat} (hm : 0 < M) {f : TVA.Math.Float.Format} {x : Tensor [M, N] (FloatV f)}
    {k : Tensor [N] (FloatV f) → Prog α}
    (hk : ∀ r : Tensor [N] (FloatV f), (∀ j, ∃ t : Tree, t.Covers M ∧
        t.Eval (AddRel E.fusion) (fun i => if h : i < M then x.get2 ⟨i, h⟩ j else default) (r.get1 j)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum0c x >>= k) := by
  simp only [tl.sum0c, Prog.bind_eq, Prog.bind]
  exact .env (by simpa [Prog.bind_eq] using safe_sum0With E.fusion hm hk)

example {d : DType} (p : Ptr d) (v : d.Val) : tl.store1 p v = tl.store (Tensor.scalar p) (Tensor.scalar v) none :=
  rfl
