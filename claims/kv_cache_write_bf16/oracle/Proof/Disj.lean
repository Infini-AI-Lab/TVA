import Proof.Idx

/-!
# `kv_cache_write_bf16`: the footprints do not overlap

The write footprints of two different programs share no byte, and no program's write footprint
shares a byte with any program's read footprint.

* `dstSet_kv`: an element of `dstSet` is a `kvIdx` of the cache's `[num_blocks, block_size,
  num_heads, head_size]` view, with every coordinate in range.
* `kc_elems_disjoint_vc`: the key cache's elements and the value cache's elements touched by a
  launch share no byte, although the two caches' storage ranges may overlap (`hd.disj_caches`).
* `img_disjoint`: different programs' enabled lanes write different elements (`gi_ne_cross`).
-/

namespace Claims.KvCacheWrite

open TVA

/-! ## The cache elements as view indices -/

/-- An element of `dstSet` is one of the cache's view elements, with its four coordinates in
range. -/
theorem dstSet_kv {d : Inst} {init : Mem}
    (hsl : ∀ t, t < d.num_tokens → slotAt d init t < (numSlots d : Int))
    {j : Nat} (hj : j ∈ dstSet d init) :
    ∃ (b r h e : Nat), b < d.num_blocks ∧ r < d.block_size ∧ h < d.num_heads ∧
      e < d.head_size ∧ j = kvIdx d b r h e := by
  obtain ⟨t, h, e, ht, hs0, hh, he, rfl⟩ := hj
  have hslt := hsl t ht
  have hmem : (slotAt d init t).toNat < numSlots d := by
    have h2 : ((slotAt d init t).toNat : Int) < ((numSlots d : Nat) : Int) := by
      rw [Int.toNat_of_nonneg hs0]
      exact hslt
    exact_mod_cast h2
  have hbs : 0 < d.block_size := by
    rcases Nat.eq_zero_or_pos d.block_size with h0 | h0
    · exfalso
      have hz : numSlots d = 0 := by simp only [numSlots, h0, Nat.mul_zero]
      omega
    · exact h0
  refine ⟨(slotAt d init t).toNat / d.block_size, (slotAt d init t).toNat % d.block_size,
    h, e, ?_, ?_, hh, he, rfl⟩
  · rw [Nat.div_lt_iff_lt_mul hbs]
    simpa only [numSlots] using hmem
  · exact Nat.mod_lt _ hbs

/-- The key cache's elements and the value cache's elements of two parts of `dstSet` share no
byte. -/
theorem kc_elems_disjoint_vc {d : Inst} (hd : d.ok) {init : Mem}
    (hsl : ∀ t, t < d.num_tokens → slotAt d init t < (numSlots d : Int))
    {S T : Set Nat} (hS : S ⊆ dstSet d init) (hT : T ⊆ dstSet d init) :
    Disjoint (d.key_cache.elems S) (d.value_cache.elems T) := by
  rw [Set.disjoint_left]
  rintro x ⟨a, ha, hx⟩ ⟨b, hb, hy⟩
  obtain ⟨ba, ra, haa, ea, hba, hra, hha, hea, rfl⟩ := dstSet_kv hsl (hS ha)
  obtain ⟨bb, rb, hbb, eb, hba', hrb, hhb, heb, rfl⟩ := dstSet_kv hsl (hT hb)
  exact Set.disjoint_left.mp
    (hd.disj_caches ba ra haa ea bb rb hbb eb hba hra hha hea hba' hrb hhb heb) hx hy

/-! ## The written elements of different programs -/

/-- Different programs' enabled lanes write different elements. -/
theorem img_disjoint {d : Inst} (hd : d.ok) {init : Mem} {p q : Nat}
    (hp : p < d.num_tokens * numTiles d) (hq : q < d.num_tokens * numTiles d)
    (hslp : slotAt d init (tk d p) < (numSlots d : Int))
    (hslq : slotAt d init (tk d q) < (numSlots d : Int))
    (hinj : ∀ t t', t < d.num_tokens → t' < d.num_tokens → 0 ≤ slotAt d init t →
      slotAt d init t = slotAt d init t' → t = t')
    (hpq : p ≠ q) :
    Disjoint (Buf.img (gi d init p) (en d init p)) (Buf.img (gi d init q) (en d init q)) := by
  rw [Set.disjoint_left]
  rintro j ⟨l, hl, rfl⟩ ⟨l', hl', hgl⟩
  exact (gi_ne_cross hd hp hq hslp hslq hinj hpq hl hl') (hgl.symm)

/-! ## `W_disjoint` -/

/-- Distinct programs write disjoint bytes. -/
theorem W_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i j : Fin (launch d).size} (hij : i ≠ j) :
    Disjoint (W d init (i : Nat)) (W d init (j : Nat)) := by
  have hp : (i : Nat) < d.num_tokens * numTiles d := by
    simpa [launch_size] using i.isLt
  have hq : (j : Nat) < d.num_tokens * numTiles d := by
    simpa [launch_size] using j.isLt
  have hsl : ∀ t, t < d.num_tokens → slotAt d init t < (numSlots d : Int) :=
    hpre.2.2.2.2.2.1
  have hinj := hpre.2.2.2.2.2.2.1
  have hslp : slotAt d init (tk d (i : Nat)) < (numSlots d : Int) := slot_lt hsl hp
  have hslq : slotAt d init (tk d (j : Nat)) < (numSlots d : Int) := slot_lt hsl hq
  have hpq : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
  have himg :
      Disjoint (Buf.img (gi d init (i : Nat)) (en d init (i : Nat)))
        (Buf.img (gi d init (j : Nat)) (en d init (j : Nat))) :=
    img_disjoint hd hp hq hslp hslq hinj hpq
  have hS : Buf.img (gi d init (i : Nat)) (en d init (i : Nat)) ⊆ dstSet d init :=
    img_subset_dstSet hd hp
  have hT : Buf.img (gi d init (j : Nat)) (en d init (j : Nat)) ⊆ dstSet d init :=
    img_subset_dstSet hd hq
  simp only [W]
  rw [Set.disjoint_union_left]
  refine ⟨?_, ?_⟩
  · rw [Set.disjoint_union_right]
    exact ⟨Buf.elems_disjoint d.key_cache himg,
      kc_elems_disjoint_vc hd hsl hS hT⟩
  · rw [Set.disjoint_union_right]
    exact ⟨(kc_elems_disjoint_vc hd hsl hT hS).symm,
      Buf.elems_disjoint d.value_cache himg⟩

/-! ## `R_disjoint_W` -/

/-- The write footprint is inside the two caches' bytes. -/
theorem W_sub_caches {d : Inst} (hd : d.ok) {init : Mem}
    (hsl : ∀ t, t < d.num_tokens → slotAt d init t < (numSlots d : Int))
    {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    W d init p ⊆ d.key_cache.bytes ∪ d.value_cache.bytes := by
  intro x hx
  simp only [W, Set.mem_union] at hx
  rcases hx with hx | hx
  · left
    refine Buf.elems_sub_bytes d.key_cache (S := Buf.img (gi d init p) (en d init p)) ?_ hx
    intro k hk
    obtain ⟨l, hl, rfl⟩ := hk
    exact gi_lt_kc hd hp (slot_lt hsl hp) hl
  · right
    refine Buf.elems_sub_bytes d.value_cache (S := Buf.img (gi d init p) (en d init p)) ?_ hx
    intro k hk
    obtain ⟨l, hl, rfl⟩ := hk
    exact gi_lt_vc hd hp (slot_lt hsl hp) hl

/-- The read footprint's parts share no byte with the two caches' bytes. -/
theorem R_disjoint_caches {d : Inst} (hd : d.ok) :
    Disjoint (R d) (d.key_cache.bytes ∪ d.value_cache.bytes) := by
  have hK : Disjoint d.key.bytes (d.key_cache.bytes ∪ d.value_cache.bytes) :=
    Set.disjoint_union_right.mpr ⟨hd.disj_kc_key.symm, hd.disj_vc_key.symm⟩
  have hV : Disjoint d.value.bytes (d.key_cache.bytes ∪ d.value_cache.bytes) :=
    Set.disjoint_union_right.mpr ⟨hd.disj_kc_value.symm, hd.disj_vc_value.symm⟩
  have hS : Disjoint d.slot_mapping.bytes (d.key_cache.bytes ∪ d.value_cache.bytes) :=
    Set.disjoint_union_right.mpr ⟨hd.disj_kc_slot.symm, hd.disj_vc_slot.symm⟩
  have hD : Disjoint (match d.mode with
      | .fp8 => d.k_scale.bytes ∪ d.v_scale.bytes
      | _ => (∅ : Set Int))
      (d.key_cache.bytes ∪ d.value_cache.bytes) := by
    split
    · rename_i hm
      obtain ⟨-, -, h1, h2, h3, h4⟩ := hd.scales hm
      exact Set.disjoint_union_left.mpr
        ⟨Set.disjoint_union_right.mpr ⟨h1.symm, h3.symm⟩,
         Set.disjoint_union_right.mpr ⟨h2.symm, h4.symm⟩⟩
    · simp
  simp only [R]
  exact Set.disjoint_union_left.mpr
    ⟨Set.disjoint_union_left.mpr ⟨Set.disjoint_union_left.mpr ⟨hK, hV⟩, hS⟩, hD⟩

/-- A program writes none of the bytes the programs read. -/
theorem R_disjoint_W {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) : Disjoint (R d) (W d init (i : Nat)) := by
  have hp : (i : Nat) < d.num_tokens * numTiles d := by
    simpa [launch_size] using i.isLt
  have hsl : ∀ t, t < d.num_tokens → slotAt d init t < (numSlots d : Int) :=
    hpre.2.2.2.2.2.1
  exact (R_disjoint_caches hd).mono_right (W_sub_caches hd hsl hp)

end Claims.KvCacheWrite
