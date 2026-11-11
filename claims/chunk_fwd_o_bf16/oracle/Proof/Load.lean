import Claim
import TVA.Proof
import Proof.Peel

/-!
# Masked tile loads with a `+0` padding

The kernel's loads all have the form `tl.load(ptrs, mask, other=0.0)` with the `other` converted
to `bfloat16` first. `safe_load_tileZ` turns that into the lane facts a dot needs: an enabled
lane holds the element, any other lane holds `+0`.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- Every `float32` converts to `bfloat16`. -/
theorem exists_conv_f32_bf16 {P : Platform} (x : F32) : ∃ r, ConvRel P .f32 .bf16 x r := by
  refine ⟨FloatV.cvt x, ?_⟩
  show FloatRel P .binary32 .bfloat16 x (FloatV.cvt x)
  unfold FloatRel
  simp only [show ¬ (Format.binary32 = Format.bfloat16) from by decide, ↓reduceIte]
  exact Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩

/-- `tl.convert .f32 .bf16 (tl.full s zf32)`, the `other=0.0` of a load: the tensor of `+0`. -/
theorem safe_other_zero {s : Shape} {k : Tensor s BF16 → Prog α}
    (hk : ∀ z : Tensor s BF16, (∀ i, z.get i = FloatV.ofPat 0) → Safe E R W Q m (k z)) :
    Safe E R W Q m (tl.convert .f32 .bf16 (tl.full s zf32) >>= k) := by
  refine safe_convert (fun i => exists_conv_f32_bf16 _) fun r hr => hk r fun i => ?_
  have := hr i
  simp only [tl.full, Tensor.get_full, zf32_eq] at this
  exact conv_bf16_zero this

/-- **A masked `bfloat16` tile load with `other=0.0`**: lane `(i, j)` with `ρ i < M` and
`κ j < N` holds element `e + ρ i · s0 + κ j · s1` of `b`, every other lane `+0`. -/
theorem safe_load_tileZ {BM BN M N e s0 s1 : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}
    (b : Buf .bf16) {ptrs : Tensor [BM, BN] (Ptr .bf16)} {mk : Tensor [BM, BN] Bool}
    (hp : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      ptrs.get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int))
    (hm : Tile2.MaskAt (some mk) ρ κ M N)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hv : b.Valid m) (hR : b.tileAt e s0 s1 ρ κ M N ⊆ R)
    {k : Tensor [BM, BN] BF16 → Prog α}
    (hk : ∀ v : Tensor [BM, BN] BF16,
      (∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
        m.read .bf16 (b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int)).addr = some (v.get2 i j)) →
      (∀ (i : Fin BM) (j : Fin BN), ¬ (ρ i < M ∧ κ j < N) → v.get2 i j = FloatV.ofPat 0) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.convert .f32 .bf16 (tl.full [BM, BN] zf32) >>=
      fun z => tl.load ptrs (some mk) (some z) >>= k) :=
  safe_other_zero fun z hz =>
    safe_load_tileAt b hp hm hlen hv hR fun v h1 h2 =>
      hk v h1 fun i j hij => by
        rw [h2 i j hij z rfl]; exact hz _

/-- **A masked `bfloat16` tile load with `other=0.0`, from the buffer's contents**: the tile holds
`val (e + ρ i · s0 + κ j · s1)` on its enabled lanes and `+0` on the others. -/
theorem safe_load_tileV {BM BN M N e s0 s1 : Nat} {ρ : Fin BM → Nat} {κ : Fin BN → Nat}
    (b : Buf .bf16) (val : Nat → BF16) {init : Mem} (hH : b.Holds init val) (hm : m = init)
    {ptrs : Tensor [BM, BN] (Ptr .bf16)} {mk : Tensor [BM, BN] Bool}
    (hp : ∀ (i : Fin BM) (j : Fin BN), ρ i < M → κ j < N →
      ptrs.get2 i j = b.ptr ((e + ρ i * s0 + κ j * s1 : Nat) : Int))
    (hmk : Tile2.MaskAt (some mk) ρ κ M N)
    (hlen : M = 0 ∨ N = 0 ∨ e + (M - 1) * s0 + (N - 1) * s1 < b.len)
    (hv : b.Valid m) (hR : b.tileAt e s0 s1 ρ κ M N ⊆ R)
    {k : Tensor [BM, BN] BF16 → Prog α}
    (hk : ∀ v : Tensor [BM, BN] BF16,
      (∀ (i : Fin BM) (j : Fin BN), v.get2 i j =
        if ρ i < M ∧ κ j < N then val (e + ρ i * s0 + κ j * s1) else FloatV.ofPat 0) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.convert .f32 .bf16 (tl.full [BM, BN] zf32) >>=
      fun z => tl.load ptrs (some mk) (some z) >>= k) :=
  safe_load_tileZ b hp hmk hlen hv hR fun v h1 h2 => hk v fun i j => by
    by_cases hc : ρ i < M ∧ κ j < N
    · simp only [hc, and_self, ↓reduceIte]
      have hlt := Tile2.matrix_lt hlen (x := e + ρ i * s0 + κ j * s1) ⟨ρ i, κ j, hc.1, hc.2, rfl⟩
      have := h1 i j hc.1 hc.2
      rw [hm, hH _ hlt] at this
      exact (Option.some.inj this).symm
    · simp only [hc, ↓reduceIte]; exact h2 i j hc

end Claims.ChunkFwdO
