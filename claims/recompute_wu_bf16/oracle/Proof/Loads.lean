import Claim
import TVA.Proof
import Proof.Lanes

/-!
# The loads

Rules for the three kinds of load the body makes, each from the input buffers: a `[64, 64]` tile
of `bfloat16` with a `0` default (`safe_tile_load`), a `[64]` vector with an `other` tensor
(`safe_vec_load`), and the conversion of `0.0` that supplies the `bfloat16` default
(`safe_conv_zero`). Memory `m` agrees with `init` on the inputs, so every element read is
`init`'s.
-/

namespace Claims.RecomputeWU

open TVA

variable {E : Env} {W : Set Int} {Q : Unit → Mem → Prop}

/-- An input element read in `m` is `init`'s. -/
theorem read_ag {d : Inst} {init m : Mem} (hag : Ag d init m) {dt : DType} (b : Buf dt)
    (hb : b.bytes ⊆ Rall d) {j : Nat} (hj : j < b.len) :
    m.read dt (b.ptr (j : Int)).addr = init.read dt (b.ptr (j : Int)).addr :=
  Mem.read_congr fun x hx => hag x (hb (b.span_sub hj hx))

/-- An input buffer valid in `init` is valid in `m`. -/
theorem valid_ag {d : Inst} {init m : Mem} (hag : Ag d init m) {dt : DType} {b : Buf dt}
    (hb : b.bytes ⊆ Rall d) (hv : b.Valid init) : b.Valid m :=
  Buf.valid_of_agree hv fun x hx => by rw [hag x (hb hx)]; exact hv.2.2.2 x hx

/-- **`tl.convert(0.0, bfloat16)`** (the default `other=0.0` of a `bfloat16` load) is a tile of
`+0`. -/
theorem safe_conv_zero {s : Shape} {R : Set Int} {m : Mem} {k : Tensor s BF16 → Prog Unit}
    (hk : ∀ z : Tensor s BF16, (∀ i, z.get i = FloatV.ofPat 0) → Safe E R W Q m (k z)) :
    Safe E R W Q m
      (tl.convert .f32 .bf16 (tl.full s (FloatV.litF32 (FloatV.ofPat 0))) >>= k) := by
  refine safe_convert (d := .f32) (d' := .bf16) (fun i => ⟨_, ConvRel.cvt_f32_bf16 _⟩) ?_
  intro z hz
  refine hk z fun i => ?_
  have h := (ConvRel.f32_bf16_iff.mp (hz i))
  have h' : (FloatV.cvt ((tl.full s (FloatV.litF32 (FloatV.ofPat 0)) : Tensor s F32).get i) : BF16)
      = FloatV.ofPat 0 := by
    show (FloatV.cvt (FloatV.litF32 (FloatV.ofPat 0) : F32) : BF16) = _
    rw [FloatV.litF32_ofPat_zero]
    exact FloatV.cvt_ofPat_zero Format.binary32_regular
  rw [h'] at h
  exact (h.eq (by decide +kernel))

/-- **A masked `[64, 64]` tile load from a `bfloat16` input.** Rows are `it·64 + i` below `len`,
columns `c0 + j` below `N`, the element of lane `(i, j)` is `e + (it·64 + i)·s + (c0 + j)`. -/
theorem safe_tile_load {d : Inst} {init m : Mem} (hag : Ag d init m) (b : Buf .bf16)
    (hbR : b.bytes ⊆ Rall d) (hv : b.Valid init) {it len c0 N e s : Nat}
    {ptrs : Tensor [64, 64] (Ptr .bf16)} {mk : Tensor [64, 64] Bool} {z : Tensor [64, 64] BF16}
    (hp : ∀ i j : Fin 64, it * 64 + i < len → c0 + j < N →
      ptrs.get2 i j = b.ptr ((e + (it * 64 + i) * s + (c0 + j) * 1 : Nat) : Int))
    (hmk : ∀ i j : Fin 64, mk.get2 i j = (decide (it * 64 + i < len) && decide (c0 + j < N)))
    (hlen : len = 0 ∨ N = 0 ∨ e + (len - 1) * s + (N - 1) * 1 < b.len)
    (hz : ∀ i j : Fin 64, z.get2 i j = FloatV.ofPat 0) {k : Tensor [64, 64] BF16 → Prog Unit}
    (hk : ∀ v : Tensor [64, 64] BF16,
      (∀ i j : Fin 64, it * 64 + i < len → c0 + j < N →
        v.get2 i j = bfV b init (e + (it * 64 + i) * s + (c0 + j) * 1)) →
      (∀ i j : Fin 64, ¬ (it * 64 + i < len ∧ c0 + j < N) → v.get2 i j = FloatV.ofPat 0) →
      Safe E (Rall d) W Q m (k v)) :
    Safe E (Rall d) W Q m (tl.load ptrs (some mk) (some z) >>= k) := by
  refine safe_load_tileAt (ρ := fun i : Fin 64 => it * 64 + i.val) (κ := fun j : Fin 64 => c0 + j.val)
    (M := len) (N := N) (e := e) (s0 := s) (s1 := 1) b hp (Tile2.maskAt_of_get hmk) hlen
    (valid_ag hag hbR hv) ((b.tileAt_sub_bytes hlen).trans hbR) ?_
  intro v h1 h2
  refine hk v (fun i j hi hj => ?_) (fun i j hij => ?_)
  · have hlt : e + (it * 64 + i) * s + (c0 + j) * 1 < b.len := by
      rcases hlen with h | h | h
      · omega
      · omega
      · have h1 : (it * 64 + (i : Nat)) * s ≤ (len - 1) * s := Nat.mul_le_mul_right s (by omega)
        have h2 : (c0 + (j : Nat)) * 1 ≤ (N - 1) * 1 := Nat.mul_le_mul_right 1 (by omega)
        omega
    have := h1 i j hi hj
    rw [read_ag hag b hbR hlt] at this
    unfold bfV bfAt
    rw [this]; rfl
  · exact h2 i j hij _ rfl |>.trans (hz i j)


/-- **A masked `[64]` load from an input**: lane `r` reads element `g r` when row `r` is inside the
sequence. -/
theorem safe_vec_load {d : Inst} {init m : Mem} (hag : Ag d init m) {dt : DType} (b : Buf dt)
    (hbR : b.bytes ⊆ Rall d) (hv : b.Valid init) {it len : Nat} {g : Fin 64 → Nat}
    {ptrs : Tensor [64] (Ptr dt)} {mk : Tensor [64] Bool} {other : Option (Tensor [64] dt.Val)}
    (hmk : ∀ r : Fin 64, mk.get1 r = rowEn it len r)
    (hp : ∀ r : Fin 64, rowEn it len r = true → ptrs.get1 r = b.ptr (g r))
    (hlt : ∀ r : Fin 64, rowEn it len r = true → g r < b.len)
    {k : Tensor [64] dt.Val → Prog Unit}
    (hk : ∀ v : Tensor [64] dt.Val,
      (∀ r : Fin 64, rowEn it len r = true → init.read dt (b.ptr (g r)).addr = some (v.get1 r)) →
      (∀ r : Fin 64, rowEn it len r = false → ∀ o, other = some o → v.get1 r = o.get1 r) →
      Safe E (Rall d) W Q m (k v)) :
    Safe E (Rall d) W Q m (tl.load ptrs (some mk) other >>= k) := by
  refine safe_load_buf b (g := fun i : Idx [64] => g i.1) (en := fun i => rowEn it len i.1)
    (Buf.addrs_of_get b (fun i => hmk i.1) (fun i hi => hp i.1 hi)) (fun i hi => hlt i.1 hi)
    (valid_ag hag hbR hv) ?_ ?_
  · refine (b.elems_sub_bytes ?_).trans hbR
    rintro j ⟨i, hi, rfl⟩
    exact hlt i.1 hi
  · intro v h1 h2
    refine hk v (fun r hr => ?_) (fun r hr o ho => h2 (r, ()) hr o ho)
    have := h1 (r, ()) hr
    rw [read_ag hag b hbR (hlt r hr)] at this
    exact this

end Claims.RecomputeWU
