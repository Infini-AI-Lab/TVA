import Proof.Lanes

/-!
# `kv_cache_write_bf16`: source and destination elements

The element of `key` and `value` that a lane reads, the cache element it writes, and the facts
about them: the source elements are inside the buffers, the destination elements are inside the
caches, distinct enabled lanes of one program write distinct elements, and elements written by
different programs with the same position are the same program's.
-/

namespace Claims.KvCacheWrite

open TVA

/-- The element of `key` that lane `l` of program `p` reads. -/
def giK (d : Inst) (p : Nat) (l : Idx [d.TILE_SIZE]) : Nat := keyIdx d (tk d p) (pos d p l)

/-- The element of `value` that lane `l` of program `p` reads. -/
def giV (d : Inst) (p : Nat) (l : Idx [d.TILE_SIZE]) : Nat := valueIdx d (tk d p) (pos d p l)

/-! ## The source elements -/

theorem giK_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d)
    {init : Mem} {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    giK d p l < d.key.len :=
  hd.key_len _ _ (tk_lt hp) (pos_lt_vals hen)

/-- An enabled lane's flat index into the value input is within its length. -/
theorem giV_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d)
    {init : Mem} {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    giV d p l < d.value.len :=
  hd.value_len _ _ (tk_lt hp) (pos_lt_vals hen)

/-! ## The destination elements -/

theorem pos_div_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d)
    {init : Mem} {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    pos d p l / d.head_size < d.num_heads := by
  have hhs : 0 < d.head_size := head_pos hd hp
  have h := pos_lt_vals hen
  simp only [numVals] at h
  rwa [Nat.div_lt_iff_lt_mul hhs]

/-- A lane's position modulo the head size is below the head size. -/
theorem pos_mod_lt' {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d)
    (l : Idx [d.TILE_SIZE]) : pos d p l % d.head_size < d.head_size :=
  Nat.mod_lt _ (head_pos hd hp)

/-- The block of the slot of a token: inside the cache's blocks. -/
theorem slot_div_lt {d : Inst} {init : Mem} {p : Nat}
    (hsl : slotAt d init (tk d p) < (numSlots d : Int)) {l : Idx [d.TILE_SIZE]}
    (hen : en d init p l = true) :
    (slotAt d init (tk d p)).toNat / d.block_size < d.num_blocks := by
  have h0 : 0 ≤ slotAt d init (tk d p) := en_slot hen
  have hbs : 0 < d.block_size := block_size_pos hsl hen
  have h1 : (slotAt d init (tk d p)).toNat < numSlots d := by
    have h2 : ((slotAt d init (tk d p)).toNat : Int) < ((numSlots d : Nat) : Int) := by
      rw [Int.toNat_of_nonneg h0]
      exact hsl
    exact_mod_cast h2
  rw [Nat.div_lt_iff_lt_mul hbs]
  simpa only [numSlots] using h1

/-- The row of the slot of a token: inside the cache's rows. -/
theorem slot_mod_lt {d : Inst} {init : Mem} {p : Nat}
    (hsl : slotAt d init (tk d p) < (numSlots d : Int)) {l : Idx [d.TILE_SIZE]}
    (hen : en d init p l = true) :
    (slotAt d init (tk d p)).toNat % d.block_size < d.block_size :=
  Nat.mod_lt _ (block_size_pos hsl hen)

/-- An enabled lane's destination index is within the key cache length. -/
theorem gi_lt_kc {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d)
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    gi d init p l < d.key_cache.len := by
  have hh := (hd.kv_len _ _ _ _ (slot_div_lt hsl hen) (slot_mod_lt hsl hen)
    (pos_div_lt hd hp hen) (pos_mod_lt' hd hp l)).1
  simpa only [gi, dstIdx] using hh

/-- An enabled lane's destination index is within the value cache length. -/
theorem gi_lt_vc {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d)
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) :
    gi d init p l < d.value_cache.len := by
  have hh := (hd.kv_len _ _ _ _ (slot_div_lt hsl hen) (slot_mod_lt hsl hen)
    (pos_div_lt hd hp hen) (pos_mod_lt' hd hp l)).2
  simpa only [gi, dstIdx] using hh

/-- An enabled lane's destination index belongs to the destination set. -/
theorem gi_mem_dstSet {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) {l : Idx [d.TILE_SIZE]}
    (hen : en d init p l = true) : gi d init p l ∈ dstSet d init :=
  ⟨tk d p, pos d p l / d.head_size, pos d p l % d.head_size, tk_lt hp, en_slot hen,
    pos_div_lt hd hp hen, pos_mod_lt' hd hp l, rfl⟩

/-- The image of the program's enabled destination indices lies inside the destination set. -/
theorem img_subset_dstSet {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d) :
    Buf.img (gi d init p) (en d init p) ⊆ dstSet d init := by
  intro k hk
  obtain ⟨l, hen, rfl⟩ := hk
  exact gi_mem_dstSet hd hp hen

/-! ## Injectivity of the destination elements -/

/-- Equal `kvIdx` values have equal positions. -/
theorem pos_eq_of_head {d : Inst} {p q : Nat} {i j : Idx [d.TILE_SIZE]}
    (hdiv : pos d p i / d.head_size = pos d q j / d.head_size)
    (hmod : pos d p i % d.head_size = pos d q j % d.head_size) :
    pos d p i = pos d q j := by
  have h1 := Nat.div_add_mod (pos d p i) d.head_size
  have h2 := Nat.div_add_mod (pos d q j) d.head_size
  rw [hdiv, hmod] at h1
  omega

/-- Equal positions of two lanes of one program are equal lanes. -/
theorem idx_eq_of_pos {d : Inst} {p : Nat}
    {i j : Idx [d.TILE_SIZE]} (h : pos d p i = pos d p j) : i = j := by
  obtain ⟨a, u⟩ := i
  obtain ⟨b, v⟩ := j
  have hv : a.val = b.val := by
    simp only [pos] at h
    omega
  have hu : u = v := by cases u; cases v; rfl
  exact Prod.ext (Fin.ext hv) hu

/-- Distinct enabled lanes of one program write distinct elements. -/
theorem gi_inj {d : Inst} (hd : d.ok) {init : Mem} {p : Nat}
    (hp : p < d.num_tokens * numTiles d)
    (hsl : slotAt d init (tk d p) < (numSlots d : Int))
    {i j : Idx [d.TILE_SIZE]} (hi : en d init p i = true) (hj : en d init p j = true)
    (h : gi d init p i = gi d init p j) : i = j := by
  simp only [gi, dstIdx, kvIdx] at h
  obtain ⟨-, -, hh, he⟩ := hd.kv_inj _ _ _ _ _ _ _ _ (slot_div_lt hsl hi) (slot_mod_lt hsl hi)
    (pos_div_lt hd hp hi) (pos_mod_lt' hd hp i) (slot_div_lt hsl hj) (slot_mod_lt hsl hj)
    (pos_div_lt hd hp hj) (pos_mod_lt' hd hp j) h
  exact idx_eq_of_pos (pos_eq_of_head hh he)

/-- Equal block-and-row of two nonnegative slots are equal slots. -/
theorem slot_eq_of_div_mod {d : Inst} {init : Mem} {p q : Nat}
    (hdiv : (slotAt d init (tk d p)).toNat / d.block_size =
      (slotAt d init (tk d q)).toNat / d.block_size)
    (hmod : (slotAt d init (tk d p)).toNat % d.block_size =
      (slotAt d init (tk d q)).toNat % d.block_size)
    (hp0 : 0 ≤ slotAt d init (tk d p)) (hq0 : 0 ≤ slotAt d init (tk d q)) :
    slotAt d init (tk d p) = slotAt d init (tk d q) := by
  have h1 := Nat.div_add_mod (slotAt d init (tk d p)).toNat d.block_size
  have h2 := Nat.div_add_mod (slotAt d init (tk d q)).toNat d.block_size
  rw [hdiv, hmod] at h1
  have h : (slotAt d init (tk d p)).toNat = (slotAt d init (tk d q)).toNat := by omega
  calc slotAt d init (tk d p) = ((slotAt d init (tk d p)).toNat : Int) :=
        (Int.toNat_of_nonneg hp0).symm
    _ = ((slotAt d init (tk d q)).toNat : Int) := by rw [h]
    _ = slotAt d init (tk d q) := Int.toNat_of_nonneg hq0

/-- The tile of a program is its quotient by the number of tokens. -/
theorem ti_eq_div {d : Inst} {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    ti d p = p / d.num_tokens := by
  have h : p / d.num_tokens < numTiles d := by
    rw [Nat.div_lt_iff_lt_mul (tokens_pos hp)]
    simpa only [Nat.mul_comm] using hp
  simp only [ti]
  exact Nat.mod_eq_of_lt h

/-- A program is the pair of its token and tile. -/
theorem prog_eq {d : Inst} {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    ti d p * d.num_tokens + tk d p = p := by
  rw [ti_eq_div hp]
  simp only [tk]
  rw [Nat.mul_comm]
  exact Nat.div_add_mod p d.num_tokens

/-- The tile of a lane's position. -/
theorem pos_div {d : Inst} (hT : 0 < d.TILE_SIZE) {p : Nat} (l : Idx [d.TILE_SIZE]) :
    pos d p l / d.TILE_SIZE = ti d p := by
  have h : pos d p l = d.TILE_SIZE * ti d p + l.1.val := by
    simp only [pos, Nat.mul_comm]
  rw [h, Nat.mul_add_div hT, Nat.div_eq_of_lt l.1.isLt, Nat.add_zero]

/-- The lane offset of a lane's position. -/
theorem pos_mod {d : Inst} {p : Nat} (l : Idx [d.TILE_SIZE]) :
    pos d p l % d.TILE_SIZE = l.1.val := by
  have h : pos d p l = d.TILE_SIZE * ti d p + l.1.val := by
    simp only [pos, Nat.mul_comm]
  rw [h, Nat.mul_add_mod, Nat.mod_eq_of_lt l.1.isLt]

/-- Equal positions of two lanes: equal tiles and equal lane offsets. -/
theorem pos_lin {d : Inst} (hT : 0 < d.TILE_SIZE) {p q : Nat} {i j : Idx [d.TILE_SIZE]}
    (h : pos d p i = pos d q j) :
    ti d p = ti d q ∧ i.1.val = j.1.val :=
  ⟨by rw [← pos_div hT i, h, pos_div hT j],
   by rw [← pos_mod i, h, pos_mod j]⟩

/-- Two programs whose enabled lanes write the same element are the same program. -/
theorem gi_inj_cross {d : Inst} (hd : d.ok) {init : Mem} {p q : Nat}
    (hp : p < d.num_tokens * numTiles d) (hq : q < d.num_tokens * numTiles d)
    (hslp : slotAt d init (tk d p) < (numSlots d : Int))
    (hslq : slotAt d init (tk d q) < (numSlots d : Int))
    (hinj : ∀ t t', t < d.num_tokens → t' < d.num_tokens → 0 ≤ slotAt d init t →
      slotAt d init t = slotAt d init t' → t = t')
    {i j : Idx [d.TILE_SIZE]} (hi : en d init p i = true) (hj : en d init q j = true)
    (h : gi d init p i = gi d init q j) : p = q := by
  have hbs : 0 < d.block_size := block_size_pos hslp hi
  simp only [gi, dstIdx, kvIdx] at h
  obtain ⟨hb, hr, hh, he⟩ := hd.kv_inj _ _ _ _ _ _ _ _ (slot_div_lt hslp hi) (slot_mod_lt hslp hi)
    (pos_div_lt hd hp hi) (pos_mod_lt' hd hp i) (slot_div_lt hslq hj) (slot_mod_lt hslq hj)
    (pos_div_lt hd hq hj) (pos_mod_lt' hd hq j) h
  have hslot := slot_eq_of_div_mod hb hr (en_slot hi) (en_slot hj)
  have htk : tk d p = tk d q :=
    hinj _ _ (tk_lt hp) (tk_lt hq) (en_slot hi) hslot
  have hti : ti d p = ti d q := (pos_lin (tile_pos' hd) (pos_eq_of_head hh he)).1
  rw [← prog_eq hp, ← prog_eq hq, hti, htk]

/-- Two different programs' enabled lanes never write the same element. -/
theorem gi_ne_cross {d : Inst} (hd : d.ok) {init : Mem} {p q : Nat}
    (hp : p < d.num_tokens * numTiles d) (hq : q < d.num_tokens * numTiles d)
    (hslp : slotAt d init (tk d p) < (numSlots d : Int))
    (hslq : slotAt d init (tk d q) < (numSlots d : Int))
    (hinj : ∀ t t', t < d.num_tokens → t' < d.num_tokens → 0 ≤ slotAt d init t →
      slotAt d init t = slotAt d init t' → t = t')
    (hpq : p ≠ q) {i j : Idx [d.TILE_SIZE]} (hi : en d init p i = true)
    (hj : en d init q j = true) : gi d init p i ≠ gi d init q j :=
  fun h => hpq (gi_inj_cross hd hp hq hslp hslq hinj hi hj h)

end Claims.KvCacheWrite
