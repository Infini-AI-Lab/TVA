import TVA.Proof.Rules.Reduce

/-!
# Rules: `tl.dot` and its relatives

The proof rules for `TVA.Triton.Dot`. The product is the platform's (`Platform.dot`,
`Platform.dotScaled`), so a rule needs only that a result exists (a device fact gives it) and
hands the rest of the program every result the platform allows; how close that result is to the
exact product is the device facts' and the `tl.dot` accumulation kit's business.

* `safe_dot`, `safe_dot_add` (`tl.dot(a, b) + c`: the folded or the separate form);
* `safe_sum_bcast_mul` (`tl.sum(x[:, :, None] * y[None, :, :], axis=1)`: products summed along
  trees, or Triton's rewrite to `tl.dot`), with `tl.MulRel.total` and `AccAdd.total`;
* `safe_dot_scaled`, `safe_dot_scaled_add` (microscaled operands).

How this fits: model → `Safe` per program (the rules of `TVA.Proof.Rules`) → `correct_of_safe`.
-/
namespace TVA

open Prog

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.dot(a, b, acc)`**: a platform result exists (`hex`); the rest must be safe for every
result the platform allows. -/
theorem safe_dot {cfg : DotCfg} {M N K : Nat} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {acc : Tensor [M, N] cfg.acc.Val}
    {k : Tensor [M, N] cfg.acc.Val → Prog α}
    (hex : ∃ r, E.P.dot cfg a b acc r)
    (hk : ∀ r, E.P.dot cfg a b acc r → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot cfg a b acc >>= k) := by
  simp only [tl.dot, bind_eq, Prog.bind]
  exact .env (.pick hex hk)

/-- **`tl.dot(a, b) + c`**: given that the `tl.dot` with accumulator `c` has a result (`hex`), the
rest must be safe for either form: the folded dot, or the dot with the zero accumulator `z` and
then the addition. -/
theorem safe_dot_add {cfg : DotCfg} {M N K : Nat} {a : Tensor [M, K] cfg.a.Val}
    {b : Tensor [K, N] cfg.b.Val} {c z : Tensor [M, N] cfg.acc.Val}
    {k : Tensor [M, N] cfg.acc.Val → Prog α}
    (hex : ∃ r, E.P.dot cfg a b c r)
    (hk : ∀ r, (E.P.dot cfg a b c r ∨ ∃ d, E.P.dot cfg a b z d ∧
        ∀ i, AccAdd E.fusion cfg.acc (d.get i) (c.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot_add cfg a b c z >>= k) := by
  simp only [tl.dot_add, bind_eq, Prog.bind]
  obtain ⟨r, hr⟩ := hex
  exact .env (.pick ⟨r, Or.inl hr⟩ hk)

/-- Every pair of elements of a standard float dtype has a product `MulRel` allows. -/
theorem tl.MulRel.total {d : DType} (hd : d = .f16 ∨ d = .bf16 ∨ d = .f32 ∨ d = .f64) :
    ∀ a b : d.Val, ∃ c, tl.MulRel d a b c := by
  rcases hd with rfl | rfl | rfl | rfl <;> exact fun _ _ => ⟨_, Or.inl rfl⟩

/-- Every pair of elements of a standard float dtype has a sum `AccAdd` allows. -/
theorem AccAdd.total {d : DType} (fusion : Bool) (hd : d = .f16 ∨ d = .bf16 ∨ d = .f32 ∨ d = .f64) :
    ∀ a b : d.Val, ∃ c, AccAdd fusion d a b c := by
  rcases hd with rfl | rfl | rfl | rfl <;> exact fun a b => ⟨_, AddRel.ieee fusion a b⟩

/-- **`tl.sum(x[:, :, None] * y[None, :, :], axis=1)`** of a standard float dtype (`K > 0`): the
rest must be safe for either whole-tensor form: each element a sum of its products along its own
tree, or Triton's `tl.dot` rewrite (the platform's dot with the zero accumulator). -/
theorem safe_sum_bcast_mul {d : DType} {M K N : Nat} {x : Tensor [M, K] d.Val}
    {y : Tensor [K, N] d.Val} (hd : d = .f16 ∨ d = .bf16 ∨ d = .f32 ∨ d = .f64) (hK : 0 < K)
    {k : Tensor [M, N] d.Val → Prog α}
    (hk : ∀ r : Tensor [M, N] d.Val,
      ((∀ i j, ∃ p : Nat → d.Val,
        (∀ k (hk : k < K), tl.MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
        ∃ t : Tree, t.Covers K ∧ t.Eval (AccAdd E.fusion d) p (r.get2 i j)) ∨
      (d.bytes ≤ 4 ∧ 16 ≤ M ∧ 16 ≤ N ∧
        E.P.dot { a := d, b := d, acc := d, precision := .ieee } x y
          (Tensor.full (d.ofBits 0)) r)) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.sum_bcast_mul d x y >>= k) := by
  simp only [tl.sum_bcast_mul, bind_eq, Prog.bind]
  have hp : ∀ (i : Fin M) (j : Fin N), ∃ p : Nat → d.Val,
      ∀ k (hk : k < K), tl.MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k) := by
    intro i j
    refine ⟨fun k => if hk : k < K then
      Classical.choose (tl.MulRel.total hd (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j))
      else default, fun k hk => ?_⟩
    simp only [dite_eq_left hk]
    exact Classical.choose_spec (tl.MulRel.total hd _ _)
  have hex : ∀ (i : Fin M) (j : Fin N), ∃ (v : d.Val) (p : Nat → d.Val) (t : Tree),
      (∀ k (hk : k < K), tl.MulRel d (x.get2 i ⟨k, hk⟩) (y.get2 ⟨k, hk⟩ j) (p k)) ∧
      t.Covers K ∧ t.Eval (AccAdd E.fusion d) p v := by
    intro i j
    obtain ⟨p, hp⟩ := hp i j
    obtain ⟨v, hv⟩ := Tree.eval_exists (AccAdd.total E.fusion hd) p (Tree.seq K)
    exact ⟨v, p, Tree.seq K, hp, Tree.seq_covers hK, hv⟩
  refine .env (.pick ⟨Tensor.of2 fun i j => Classical.choose (hex i j), Or.inl fun i j => ?_⟩ hk)
  obtain ⟨p, t, h1, h2, h3⟩ := Classical.choose_spec (hex i j)
  exact ⟨p, h1, t, h2, h3⟩

/-! ## Microscaled operands -/

section Scaled
variable {cfg : ScaledCfg} {M N K : Nat} {dl dr dsl dsr : DType} {Ra Ca Rb Cb G : Nat}
  {lhs : Tensor [Ra, Ca] dl.Val} {lhs_scale : Option (Tensor [M, G] dsl.Val)}
  {rhs : Tensor [Rb, Cb] dr.Val} {rhs_scale : Option (Tensor [N, G] dsr.Val)}

/-- **`tl.dot_scaled`**: the shapes and dtypes are ones Triton accepts, a platform result exists,
and the rest must be safe for every result the platform allows. -/
theorem safe_dot_scaled {acc : Tensor [M, N] F32} {k : Tensor [M, N] F32 → Prog α}
    (hshape : cfg.ShapeOk M N K Ra Ca Rb Cb G) (htypes : cfg.TypesOk dl dr dsl dsr)
    (hex : ∃ r, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) acc r)
    (hk : ∀ r, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) acc r → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot_scaled cfg M N K lhs lhs_scale rhs rhs_scale acc >>= k) := by
  simp only [tl.dot_scaled, bind_eq, Prog.bind]
  exact .pick ⟨(), hshape, htypes⟩ fun _ _ => .env (.pick hex hk)

/-- **`tl.dot_scaled(…) + c`**: given that the folded form has a result, the rest must be safe
for either form: `c` folded into the accumulator, or the product then the addition. -/
theorem safe_dot_scaled_add {c z : Tensor [M, N] F32} {k : Tensor [M, N] F32 → Prog α}
    (hshape : cfg.ShapeOk M N K Ra Ca Rb Cb G) (htypes : cfg.TypesOk dl dr dsl dsr)
    (hex : ∃ r, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) c r)
    (hk : ∀ r, (E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
        (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) c r ∨
      ∃ d, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
        (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) z d ∧
        ∀ i, AddRel E.fusion (d.get i) (c.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot_scaled_add cfg M N K lhs lhs_scale rhs rhs_scale c z >>= k) := by
  simp only [tl.dot_scaled_add, bind_eq, Prog.bind]
  obtain ⟨r, hr⟩ := hex
  exact .pick ⟨(), hshape, htypes⟩ fun _ _ => .env (.pick ⟨r, Or.inl hr⟩ hk)

end Scaled

end TVA
