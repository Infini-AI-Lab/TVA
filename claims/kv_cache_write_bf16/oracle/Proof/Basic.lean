import Claim
import TVA.Proof

/-!
# `kv_cache_write_bf16`: the model of one program

The kernel's body, the grid, the per-program token/tile/lane arithmetic, and the footprints.
-/

namespace Claims.KvCacheWrite

open TVA

/-- The kernel body, unfolded into its explicit sequence of tile operations. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c = (do
      let token_idx := tl.program_id c 0
      let slot_idx := (← tl.convert .i64 .i64
        (Tensor.scalar (← tl.load1 ((d.slot_mapping.ptr 0) + token_idx)))).item
      if IntV.lt slot_idx (IntV.ofInt 0 : I64) then
        return ()
      else do
        let block_idx :=
          (← tl.idiv (Tensor.scalar slot_idx) (Tensor.scalar (IntV.ofNat d.block_size : I64))).item
        let block_offset :=
          (← tl.irem (Tensor.scalar slot_idx) (Tensor.scalar (IntV.ofNat d.block_size : I64))).item
        let tile_i := tl.program_id c 1
        let tile_offs := tl.arange 0 d.TILE_SIZE
        let tile_pos : Tensor [d.TILE_SIZE] I32 :=
          tile_i * (IntV.ofNat d.TILE_SIZE : I32) + tile_offs
        let src_key_idx : I64 :=
          (← tl.convert .i32 .i64 (Tensor.scalar token_idx)).item * IntV.ofNat d.key_stride
        let src_value_idx : I64 :=
          (← tl.convert .i32 .i64 (Tensor.scalar token_idx)).item * IntV.ofNat d.value_stride
        let cur_head ← tl.idiv tile_pos (tl.full [d.TILE_SIZE] (IntV.ofNat d.head_size : I32))
        let cur_dim ← tl.irem tile_pos (tl.full [d.TILE_SIZE] (IntV.ofNat d.head_size : I32))
        let head64 ← tl.convert .i32 .i64 cur_head
        let dim64 ← tl.convert .i32 .i64 cur_dim
        let tgt_idx_k : Tensor [d.TILE_SIZE] I64 :=
          block_idx * (IntV.ofNat d.block_stride : I64) +
            block_offset * (IntV.ofNat d.page_stride : I64) +
            head64 * (IntV.ofNat d.head_stride : I64) + dim64
        let tgt_idx_v : Tensor [d.TILE_SIZE] I64 := tgt_idx_k
        let mask : Tensor [d.TILE_SIZE] Bool :=
          tile_pos.ltS (IntV.ofNat (d.num_heads * d.head_size) : I32)
        (match d.mode, d.key.ptr 0, d.value.ptr 0, d.key_cache.ptr 0, d.value_cache.ptr 0 with
        | .bf16, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr => do
          let key_load ← tl.load (key_ptr + src_key_idx + tile_pos) (some mask)
          let key_tile := key_load
          let value_load ← tl.load (value_ptr + src_value_idx + tile_pos) (some mask)
          let value_tile := value_load
          tl.store (key_cache_ptr + tgt_idx_k) key_tile (some mask)
          tl.store (value_cache_ptr + tgt_idx_v) value_tile (some mask)
          return ()
        | .fp8, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr => do
          let key_load ← tl.load (key_ptr + src_key_idx + tile_pos) (some mask)
          let key_tile ←
            tl.div (← tl.cvt (g := .binary32) key_load) (bc (← tl.load1 (d.k_scale.ptr 0)))
          let value_load ← tl.load (value_ptr + src_value_idx + tile_pos) (some mask)
          let value_tile ←
            tl.div (← tl.cvt (g := .binary32) value_load) (bc (← tl.load1 (d.v_scale.ptr 0)))
          tl.store (key_cache_ptr + tgt_idx_k) (← tl.cvt8 (g := .e4m3) key_tile .rn) (some mask)
          tl.store (value_cache_ptr + tgt_idx_v) (← tl.cvt8 (g := .e4m3) value_tile .rn) (some mask)
          return ()
        | .fp8in, key_ptr, value_ptr, key_cache_ptr, value_cache_ptr => do
          let key_load ← tl.load (key_ptr + src_key_idx + tile_pos) (some mask)
          let key_tile := key_load
          let value_load ← tl.load (value_ptr + src_value_idx + tile_pos) (some mask)
          let value_tile := value_load
          tl.store (key_cache_ptr + tgt_idx_k) key_tile (some mask)
          tl.store (value_cache_ptr + tgt_idx_v) value_tile (some mask)
          return ())) := by
  simp only [model, reshape_and_cache_kernel_flash]
  rfl

/-- The launch has num_tokens times numTiles programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.num_tokens * numTiles d := by
  simp [launch, Launch.size]

/-- Program id 0 of program i is its index modulo num_tokens (the token). -/
theorem launch_pid0 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) % d.num_tokens := by
  simp [Launch.ctx, launch]

/-- Program id 1 of program i is its index divided by num_tokens, modulo numTiles (the tile). -/
theorem launch_pid1 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 1 = (i : Nat) / d.num_tokens % numTiles d := by
  simp [Launch.ctx, launch]

/-- An instance satisfying the claim's conditions yields a valid launch. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine ⟨hd.tokens_lt, hd.tiles_lt, by simp [launch], ?_, rfl, rfl, rfl, ?_⟩
  · rw [launch_size]; exact hd.programs_lt
  · intro s hs
    simp only [launch, List.mem_singleton] at hs
    subst hs
    refine ⟨fun k hk => ?_, ?_⟩
    · simp only [List.mem_singleton] at hk
      subst hk; exact hd.tile_pow2
    · simpa using hd.tile_le

/-- The token of program `p`. -/
def tk (d : Inst) (p : Nat) : Nat := p % d.num_tokens

/-- The tile of program `p`. -/
def ti (d : Inst) (p : Nat) : Nat := p / d.num_tokens % numTiles d

/-- The position (in the token's row) of lane `l` of program `p`. -/
def pos (d : Inst) (p : Nat) (l : Idx [d.TILE_SIZE]) : Nat := ti d p * d.TILE_SIZE + l.1.val

/-- The lane is enabled: in range and its slot is not a padding slot. -/
def en (d : Inst) (init : Mem) (p : Nat) (l : Idx [d.TILE_SIZE]) : Bool :=
  decide (pos d p l < numVals d) && decide (0 ≤ slotAt d init (tk d p))

/-- The cache element lane `l` of program `p` writes. -/
def gi (d : Inst) (init : Mem) (p : Nat) (l : Idx [d.TILE_SIZE]) : Nat :=
  dstIdx d init (tk d p) (pos d p l / d.head_size) (pos d p l % d.head_size)

/-- The read footprint of program `p`. -/
def R (d : Inst) : Set Int :=
  d.key.bytes ∪ d.value.bytes ∪ d.slot_mapping.bytes ∪
    (match d.mode with
     | .fp8 => d.k_scale.bytes ∪ d.v_scale.bytes
     | _ => (∅ : Set Int))

/-- The write footprint of program `p`: its key and value cache elements. -/
def W (d : Inst) (init : Mem) (p : Nat) : Set Int :=
  d.key_cache.elems (Buf.img (gi d init p) (en d init p)) ∪
    d.value_cache.elems (Buf.img (gi d init p) (en d init p))

/-- What program `p` leaves in its own memory `m`. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ l : Idx [d.TILE_SIZE], en d init p l = true →
    (∃ x v, init.read d.mode.kd (d.key.ptr (keyIdx d (tk d p) (pos d p l))).addr = some x ∧
      m.read d.mode.cd (d.key_cache.ptr (gi d init p l)).addr = some v ∧
      Stored d.mode (scaleAt init d.k_scale) x v) ∧
    (∃ x v, init.read d.mode.kd (d.value.ptr (valueIdx d (tk d p) (pos d p l))).addr = some x ∧
      m.read d.mode.cd (d.value_cache.ptr (gi d init p l)).addr = some v ∧
      Stored d.mode (scaleAt init d.v_scale) x v)

/-- The tile size is positive. -/
theorem tile_pos' {d : Inst} (hd : d.ok) : 0 < d.TILE_SIZE := by
  obtain ⟨e, he⟩ := hd.tile_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- With at least one program, the number of values per token is positive. -/
theorem numVals_pos {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    0 < numVals d := by
  rcases Nat.eq_zero_or_pos (numVals d) with h | h
  · exfalso
    have hT := tile_pos' hd
    have : numTiles d = 0 := by
      simp only [numTiles, h, Nat.zero_add]
      exact Nat.div_eq_of_lt (by omega)
    rw [this, Nat.mul_zero] at hp
    omega
  · exact h

/-- `numTiles d * TILE_SIZE ≤ 2^31`: the tile positions of the whole grid fit in an `int32`. -/
theorem tiles_mul_le {d : Inst} (hd : d.ok) : numTiles d * d.TILE_SIZE ≤ 2 ^ 31 := by
  obtain ⟨e, he⟩ := hd.tile_pow2
  have he20 : e ≤ 20 := by
    have h := hd.tile_le; rw [he] at h
    exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h
  have hT : 0 < d.TILE_SIZE := tile_pos' hd
  have hsplit : 2 ^ 31 = 2 ^ e * 2 ^ (31 - e) := by
    rw [← Nat.pow_add, Nat.add_sub_cancel' (by omega : e ≤ 31)]
  have hmul : (2 ^ (31 - e) + 1) * d.TILE_SIZE = 2 ^ 31 + d.TILE_SIZE := by
    rw [Nat.add_mul, Nat.one_mul, he, Nat.mul_comm (2 ^ (31 - e)) (2 ^ e), ← hsplit]
  have key : numVals d + d.TILE_SIZE - 1 < (2 ^ (31 - e) + 1) * d.TILE_SIZE := by
    have hvals := hd.vals_lt
    rw [hmul]; omega
  have hnum : numTiles d ≤ 2 ^ (31 - e) := by
    have h := (Nat.div_lt_iff_lt_mul hT).mpr key
    simp only [numTiles] at h ⊢
    omega
  simp only [numTiles]
  calc (numVals d + d.TILE_SIZE - 1) / d.TILE_SIZE * d.TILE_SIZE
      ≤ 2 ^ (31 - e) * d.TILE_SIZE := Nat.mul_le_mul_right _ hnum
    _ = 2 ^ 31 := by rw [Nat.mul_comm, he, ← hsplit]

/-- With at least one program, the head size is positive. -/
theorem head_pos {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    0 < d.head_size := by
  have h := numVals_pos hd hp
  simp only [numVals] at h
  exact Nat.pos_of_mul_pos_left h

/-- With at least one program, the token count is positive. -/
theorem tokens_pos {d : Inst} {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    0 < d.num_tokens :=
  Nat.pos_of_mul_pos_right (lt_of_le_of_lt (Nat.zero_le p) hp)

/-- The token index of a program is below the token count. -/
theorem tk_lt {d : Inst} {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    tk d p < d.num_tokens :=
  Nat.mod_lt _ (tokens_pos hp)

/-- The tile index of a program is below the tile count. -/
theorem ti_lt {d : Inst} {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    ti d p < numTiles d := by
  have h : 0 < d.num_tokens := tokens_pos hp
  have h' : p / d.num_tokens < numTiles d := by
    rw [Nat.div_lt_iff_lt_mul h]
    calc p < d.num_tokens * numTiles d := hp
      _ = numTiles d * d.num_tokens := Nat.mul_comm _ _
  simp only [ti]
  rw [Nat.mod_eq_of_lt h']
  exact h'

/-- Every lane position of a program is below 2^31. -/
theorem pos_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < d.num_tokens * numTiles d)
    (l : Idx [d.TILE_SIZE]) : pos d p l < 2 ^ 31 := by
  have hl : l.1.val < d.TILE_SIZE := l.1.isLt
  have h1 : ti d p < numTiles d := ti_lt hp
  have hT : 0 < d.TILE_SIZE := tile_pos' hd
  have h3 : (ti d p + 1) * d.TILE_SIZE ≤ numTiles d * d.TILE_SIZE :=
    Nat.mul_le_mul_right _ (Nat.succ_le_of_lt h1)
  have h4 := tiles_mul_le hd
  have h5 : ti d p * d.TILE_SIZE + l.1.val < (ti d p + 1) * d.TILE_SIZE := by
    rw [Nat.succ_mul]; omega
  simp only [pos]
  omega

/-- An enabled lane's position lies below the number of values per token. -/
theorem pos_lt_vals {d : Inst} {init : Mem} {p : Nat}
    {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) : pos d p l < numVals d := by
  simp only [en, Bool.and_eq_true, decide_eq_true_eq] at hen
  exact hen.1

/-- An enabled lane implies its token's slot is nonnegative. -/
theorem en_slot {d : Inst} {init : Mem} {p : Nat}
    {l : Idx [d.TILE_SIZE]} (hen : en d init p l = true) : 0 ≤ slotAt d init (tk d p) := by
  simp only [en, Bool.and_eq_true, decide_eq_true_eq] at hen
  exact hen.2

/-- If all tokens' slots are below the slot count, so is the slot of the program's token. -/
theorem slot_lt {d : Inst} {init : Mem}
    (hsl : ∀ t, t < d.num_tokens → slotAt d init t < (numSlots d : Int))
    {p : Nat} (hp : p < d.num_tokens * numTiles d) :
    slotAt d init (tk d p) < (numSlots d : Int) :=
  hsl _ (tk_lt hp)

/-- An enabled lane with an in-range slot implies the block size is positive. -/
theorem block_size_pos {d : Inst} {init : Mem} {p : Nat} {l : Idx [d.TILE_SIZE]}
    (hsl : slotAt d init (tk d p) < (numSlots d : Int)) (hen : en d init p l = true) :
    0 < d.block_size := by
  have h0 : (0 : Int) ≤ slotAt d init (tk d p) := en_slot hen
  rcases Nat.eq_zero_or_pos d.block_size with h | h
  · exfalso
    have : numSlots d = 0 := by simp [numSlots, h]
    rw [this] at hsl
    omega
  · exact h

end Claims.KvCacheWrite
