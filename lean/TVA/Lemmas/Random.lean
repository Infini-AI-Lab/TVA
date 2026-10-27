import TVA.Lang.Random
import TVA.Lemmas.Ops

/-!
# Proof rules for random numbers

`tl.randint` is an exact function (`Philox.randint4x`), so it needs no rule. `tl.rand` rounds
twice: the conversion of the non-negative integer to `float32` (`tl.ItofRel`) and the IEEE
product with `Philox.scale32`.
-/
namespace TVA

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {s : Shape}

/-- **`uint_to_uniform_float(x)`**: each element is the product of a conversion of
`Philox.toNonneg x` with `Philox.scale32`. -/
theorem safe_uint_to_uniform_float {x : Tensor s U32} {k : Tensor s F32 → Prog α}
    (hk : ∀ r : Tensor s F32, (∀ i, ∃ v, tl.ItofRel .binary32 (Philox.toNonneg (x.get i)).toInt v ∧
      FloatV.NaNEq (FloatV.mul v Philox.scale32) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.uint_to_uniform_float x >>= k) := by
  simp only [tl.uint_to_uniform_float, Prog.bind_eq, Prog.bind_assoc]
  refine safe_itof (Or.inr (Or.inr (Or.inl rfl))) fun xf hxf => safe_mul fun r hr =>
    hk r fun i => ⟨xf.get i, ?_, ?_⟩
  · simpa [Tensor.get_map] using hxf i
  · simpa [Tensor.get_full] using hr i

/-- **`tl.rand(seed, offset)`**: as `uint_to_uniform_float` of `tl.randint`. -/
theorem safe_rand {sg : Bool} {w : Nat} {sg' : Bool} {w' : Nat} {seed : IntV sg w}
    {offset : Tensor s (IntV sg' w')} {nRounds : Nat} {k : Tensor s F32 → Prog α}
    (hk : ∀ r : Tensor s F32, (∀ i, ∃ v,
      tl.ItofRel .binary32 (Philox.toNonneg (Philox.randint4x seed (offset.get i) nRounds).1).toInt v ∧
      FloatV.NaNEq (FloatV.mul v Philox.scale32) (r.get i)) → Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.rand seed offset nRounds >>= k) :=
  safe_uint_to_uniform_float fun r hr => hk r fun i => by simpa [tl.randint, Tensor.get_map] using hr i

end TVA
