import TVA.Lang.Scaled
import TVA.Lemmas.Ops

/-!
# Proof rule for `tl.dot_scaled`

The product is the platform's (`P.dotScaled`), so the rule only needs a result to exist
(`Rules.Num.DotScaledTotal`). Its accuracy is the separate rule `Rules.Num.DotScaledAcc`, stated
with the exact element values (`scaledValue`).
-/
namespace TVA

section Rules
variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **`tl.dot_scaled`**: the shapes and dtypes check, a platform result exists, and the
rest is safe for every result the platform may give. -/
theorem safe_dot_scaled {cfg : ScaledCfg} {M N K : Nat} {dl dr dsl dsr : DType} {Ra Ca Rb Cb G : Nat}
    {lhs : Tensor [Ra, Ca] dl.Val} {lhs_scale : Option (Tensor [M, G] dsl.Val)}
    {rhs : Tensor [Rb, Cb] dr.Val} {rhs_scale : Option (Tensor [N, G] dsr.Val)} {acc : Tensor [M, N] F32}
    {k : Tensor [M, N] F32 → Prog α}
    (hshape : cfg.ShapeOk M N K Ra Ca Rb Cb G) (htypes : cfg.TypesOk dl dr dsl dsr)
    (hex : ∃ r, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) acc r)
    (hk : ∀ r, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) acc r → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot_scaled cfg M N K lhs lhs_scale rhs rhs_scale acc >>= k) := by
  simp only [tl.dot_scaled, Prog.bind_eq, Prog.bind]
  exact .pick ⟨(), hshape, htypes⟩ fun _ _ => .env (.pick hex hk)

/-- **`tl.dot_scaled(…) + c`**: either the combined form or the product then the addition. -/
theorem safe_dot_scaled_add {cfg : ScaledCfg} {M N K : Nat} {dl dr dsl dsr : DType} {Ra Ca Rb Cb G : Nat}
    {lhs : Tensor [Ra, Ca] dl.Val} {lhs_scale : Option (Tensor [M, G] dsl.Val)}
    {rhs : Tensor [Rb, Cb] dr.Val} {rhs_scale : Option (Tensor [N, G] dsr.Val)} {c z : Tensor [M, N] F32}
    {k : Tensor [M, N] F32 → Prog α}
    (hshape : cfg.ShapeOk M N K Ra Ca Rb Cb G) (htypes : cfg.TypesOk dl dr dsl dsr)
    (hex : ∃ r, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
      (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) c r)
    (hk : ∀ r, (E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
        (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) c r ∨
      ∃ d, E.P.dotScaled cfg (cfg.unpackL M K lhs) (lhs_scale.map (cfg.scalesL M K))
        (cfg.unpackR K N rhs) (rhs_scale.map (cfg.scalesR K N)) z d ∧
        ∀ i, AddRel E.fusion (d.get i) (c.get i) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.dot_scaled_add cfg M N K lhs lhs_scale rhs rhs_scale c z >>= k) := by
  simp only [tl.dot_scaled_add, Prog.bind_eq, Prog.bind]
  obtain ⟨r, hr⟩ := hex
  exact .pick ⟨(), hshape, htypes⟩ fun _ _ => .env (.pick ⟨r, Or.inl hr⟩ hk)

end Rules

end TVA
