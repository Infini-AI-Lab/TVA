import Proof.Loop

/-!
# `slot_mappings_i64`: the stores of a trip

Both loops end a trip with `tl.store(slot_mapping_ptr + offset, vals, mask=offset < lim)`, with
`slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride` and `offset = i +
tl.arange(0, 1024)`: lane `l` stores element `(g, X + l)` when `X + l < lim`. `safe_store_row`
is that step.
-/

namespace Claims.SlotMappings

open TVA

/-- The lanes of `i + tl.arange(0, 1024)` for `i` of value `X ≤ 2^22`. -/
theorem offset_lane {i : I32} {X : Nat} (hi : i.toInt = X) (hX : X ≤ 2 ^ 22) (l : Idx [1024]) :
    ((i + tl.arange 0 1024).get l).toInt = (X : Int) + (l.1 : Nat) := by
  obtain ⟨l, ⟨⟩⟩ := l
  have h := IntV.I32.toInt_sadd_arange (n := 1024) hi (by omega) l
  simp only [Tensor.get1] at h
  rw [h]; push_cast; ring

/-- The row pointer of group `g`. -/
theorem row_ptr {d : Inst} (hd : d.ok) {g : Nat} (hg : g < d.G) :
    d.slot_mappings.ptr 0 + (IntV.ofNat g : I32) * (IntV.ofNat d.S : I32) =
      d.slot_mappings.ptr ((g * d.S : Nat) : Int) := by
  have h1 := hd.GS_lt
  have h0 := hd.G_lt
  have h2 : g * d.S ≤ (d.G - 1) * d.S := Nat.mul_le_mul_right _ (by omega)
  have hm : ((IntV.ofNat g : I32) * (IntV.ofNat d.S : I32)).toInt = ((g * d.S : Nat) : Int) :=
    IntV.I32.mul_nat (toInt_ofNat32 (by omega)) (toInt_ofNat32 hd.S_lt) (lt_of_le_of_lt h2 h1)
  exact ptr_zero_add _ hm

/-- **A trip's store** `tl.store(row + offset, vals, mask=offset < lim)` at the tokens
`[X, min H (X + 1024))` of group `g`: afterwards element `(g, X + l)` holds lane `l`, and nothing
else changed. -/
theorem safe_store_row {d : Inst} (hd : d.ok) {g : Nat} (hg : g < d.G) {E : Env}
    {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {X : Nat} {H : Int}
    (hH : H ≤ d.N) {i lim : I32} (hi : i.toInt = X) (hlim : lim.toInt = H) (hXH : (X : Int) < H)
    (vals : Tensor [1024] I64) (hW : d.slot_mappings.elems (tok d g X (min H (X + 1024))) ⊆ W)
    (hv : d.slot_mappings.Valid m) {k : Unit → Prog α}
    (hk : ∀ m', (∀ l : Idx [1024], (X : Int) + (l.1 : Nat) < H →
        m'.read .i64 (d.slot_mappings.ptr ((g * d.S + (X + (l.1 : Nat)) : Nat) : Int)).addr =
          some (vals.get l)) →
      (∀ z, z ∉ d.slot_mappings.elems (tok d g X (min H (X + 1024))) → m' z = m z) →
      (∀ z, m z ≠ none → m' z ≠ none) →
      Safe E R (W \ d.slot_mappings.elems (tok d g X (min H (X + 1024)))) Q m' (k ())) :
    Safe E R W Q m (tl.store ((d.slot_mappings.ptr 0 + (IntV.ofNat g : I32) * (IntV.ofNat d.S : I32))
      + (i + tl.arange 0 1024)) vals (some ((i + tl.arange 0 1024).ltS lim)) >>= k) := by
  have hN := hd.N_le
  have hX : X ≤ 2 ^ 22 := by
    have : (X : Int) < 2 ^ 22 := by have : (d.N : Int) ≤ 2 ^ 22 := by exact_mod_cast hN
                                    omega
    omega
  rw [row_ptr hd hg]
  let gL : Idx [1024] → Nat := fun l => g * d.S + (X + (l.1 : Nat))
  let enL : Idx [1024] → Bool := fun l => decide ((X : Int) + (l.1 : Nat) < H)
  have hmask : ∀ l, ((i + tl.arange 0 1024).ltS lim).get l = enL l := by
    intro l
    rw [Tensor.get_ltS, IntV.lt_eq, offset_lane hi hX, hlim]
  have hptr : ∀ l, enL l = true →
      (d.slot_mappings.ptr ((g * d.S : Nat) : Int) + (i + tl.arange 0 1024)).get l =
        d.slot_mappings.ptr (gL l : Int) := by
    intro l _
    rw [Tensor.get_ptrAdd, Buf.ptr_offset, offset_lane hi hX]
    simp only [gL]; push_cast; ring_nf
  have hin : ∀ l, enL l = true → gL l < d.slot_mappings.len := by
    intro l he
    have := of_decide_eq_true he
    exact out_lt hd hg (by omega)
  have hinj : ∀ a b, enL a = true → enL b = true → gL a = gL b → a = b := by
    rintro ⟨a, ⟨⟩⟩ ⟨b, ⟨⟩⟩ _ _ h
    dsimp only [gL] at h
    have : a = b := Fin.ext (by omega)
    rw [this]
  have himg : Buf.img gL enL = tok d g X (min H (X + 1024)) := by
    ext j
    constructor
    · rintro ⟨⟨l, ⟨⟩⟩, he, rfl⟩
      have : (X : Int) + ((l : Nat) : Int) < H := of_decide_eq_true he
      refine ⟨X + l, by push_cast; omega, ?_, rfl⟩
      rw [lt_min_iff]; push_cast; omega
    · rintro ⟨x, h1, h2, rfl⟩
      rw [lt_min_iff] at h2
      have hl : x - X < 1024 := by omega
      refine ⟨(⟨x - X, hl⟩, ()), decide_eq_true (by push_cast; omega), ?_⟩
      simp only [gL]; congr 1; omega
  refine safe_store_buf d.slot_mappings (Buf.addrs_of_get _ hmask hptr) hin hinj hv
    (by rw [himg]; exact hW) fun m' hread hfr hpres => ?_
  rw [himg] at hfr ⊢
  exact hk m' (fun l hl => hread l (decide_eq_true hl)) hfr hpres

end Claims.SlotMappings
