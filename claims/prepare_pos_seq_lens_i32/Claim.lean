import TVA

/-!
# `prepare_pos_seq_lens_i32`: positions and sequence lengths of a batch, vLLM

The kernel is `_prepare_pos_seq_lens_kernel` of `source/prepare_pos_seq_lens.py` (lines 54-87; it
reads the module-level names `triton` and `tl`, lines 50-51). The host `prepare_pos_seq_lens`
(lines 90-106) launches it with `num_reqs + 1` programs: program `r < num_reqs` serves request
`r` (an `int64` slot `idx_mapping[r]` into the `int32` `num_computed_tokens`, and the `int32`
`query_start_loc[r]`, `query_start_loc[r + 1]`), and the last program sets the unused entries of
`seq_lens` to `0`. `max_num_reqs` is `seq_lens.shape[0]`. `SETUP.md` lists the launches.

What is claimed: every covered launch ends normally; for every request `r < num_reqs`, with
`c = num_computed_tokens[idx_mapping[r]]`, `s = query_start_loc[r]` and
`q = query_start_loc[r + 1] - s`, the `int32` `seq_lens[r]` is `c + q` and the `int64`
`pos[s + j]` is `c + j` for every `j < q`; `seq_lens[r]` is `0` for every
`num_reqs ≤ r < max_num_reqs`; every byte outside those elements is unchanged.

The conditions: fewer than `2^31 - 1` requests and `max_num_reqs ≤ 2^31 - 1024` (the kernel's
limits: the grid, and the `int32` bounds of the zeroing loop); the buffers hold the batch's
entries and are valid; the two outputs share no byte with each other or with an input;
`query_start_loc` is nondecreasing on the batch from a nonnegative start, its last entry at most
`pos`'s length and at most `2^22` (the batch's tokens); every request's slot lies in
`num_computed_tokens`, and its computed tokens are a context length (`0` to `2^22`). Any
`int64` slot values in range and any initial outputs are covered.

The specification is exact: no float operation, no device fact.
-/
namespace Claims.PreparePosSeqLens

open TVA

/-! ## The model -/

/-- `_prepare_pos_seq_lens_kernel` (source lines 54-87). The pointers are `pos_ptr : *i64`,
`seq_lens_ptr : *i32`, `idx_mapping_ptr : *i64`, `query_start_loc_ptr : *i32` and
`num_computed_tokens_ptr : *i32` (the host's buffers, `SETUP.md`). `max_num_reqs` is a runtime
`int32` (an integer argument in `[0, 2^31)`; at `1` Triton makes it the compile-time constant
`1`, whose uses, the bound of `tl.range` and the right operand of `<` against an `int32` block,
build the `int32` `1`: the same values, so the model is the same). `BLOCK_SIZE: tl.constexpr =
1024` (the host's `BLOCK_SIZE=1024`) is the literal `1024` below. -/
def prepare_pos_seq_lens_kernel (pos_ptr : Ptr .i64) (seq_lens_ptr : Ptr .i32)
    (idx_mapping_ptr : Ptr .i64) (query_start_loc_ptr : Ptr .i32)
    (num_computed_tokens_ptr : Ptr .i32) (max_num_reqs : I32) : Kernel where
  body ctx := do
    -- line 64: axis 0 of the 1-D grid, an `int32`
    let req_id := tl.program_id ctx 0
    -- line 65: `tl.num_programs(0)`, an `int32`, minus the Python `1` built in `int32`, wrapping
    let num_reqs := tl.num_programs ctx 0 - (1 : I32)
    -- line 66: `if req_id == num_reqs:`, a runtime `if` on the `int32` `==`; lines 67-71 end the
    -- program with `return`, so the programs that take the branch run only lines 67-70, the
    -- others only lines 73-87
    if req_id.eq num_reqs then
      -- line 67: `for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE)`: two runtime `int32`
      -- bounds and the constant step `1024`, an `int32` induction variable; nothing carried
      tl.forRange num_reqs max_num_reqs (IntV.ofNat 1024 : I32) () fun i _ => do
        -- line 68: the `int32` scalar plus the `int32` block, wrapping
        let block := i + tl.arange 0 1024
        -- line 69: the signed `int32` `<`
        let mask := block.ltS max_num_reqs
        -- line 70: the Python `0` becomes an `int32` scalar tensor, broadcast to the block and
        -- converted to the pointer's dtype `int32` (itself)
        tl.store (seq_lens_ptr + block) (tl.full [1024] (IntV.ofInt 0 : I32)) (some mask)
    else do
      -- line 73: one `int64` element
      let req_state_idx ← tl.load1 (idx_mapping_ptr + req_id)
      -- line 74: one `int32` element, at the `int64` offset
      let num_computed_tokens ← tl.load1 (num_computed_tokens_ptr + req_state_idx)
      -- line 76: one `int32` element
      let start ← tl.load1 (query_start_loc_ptr + req_id)
      -- line 77: `query_start_loc_ptr + req_id + 1` is `(ptr + req_id) + 1`, the Python `1` an
      -- `int32`; one `int32` element
      let end_ ← tl.load1 (query_start_loc_ptr + req_id + (1 : I32))
      -- line 78: the wrapping `int32` difference
      let query_len := end_ - start
      -- line 80: the wrapping `int32` sum
      let seq_len := num_computed_tokens + query_len
      -- line 81: `seq_len` is `int32`, the pointer's dtype
      tl.store1 (seq_lens_ptr + req_id) seq_len
      -- line 83: `for i in tl.range(0, query_len, BLOCK_SIZE)`: the bounds `0` and `1024` built
      -- in `int32` beside the runtime `int32` `query_len`; nothing carried
      tl.forRange (IntV.ofInt 0 : I32) query_len (IntV.ofNat 1024 : I32) () fun i _ => do
        -- line 84
        let block := i + tl.arange 0 1024
        -- line 85: the signed `int32` `<`
        let mask := block.ltS query_len
        -- line 86: the `int32` scalar plus the `int32` block, wrapping
        let pos := num_computed_tokens + block
        -- line 87: the address `(pos_ptr + start) + block`, left to right; `pos` converted to
        -- the pointer's dtype `int64` (`tl.convert`, a sign extension)
        tl.store (pos_ptr + start + block) (← tl.convert .i32 .i64 pos) (some mask)

/-! ## The conditions -/

/-- What the host chooses: one launch through `prepare_pos_seq_lens` (lines 90-106). The tensors
are given by their storage (`Buf`). -/
structure Inst where
  /-- `num_reqs = idx_mapping.shape[0]`: the grid is `num_reqs + 1`. -/
  N : Nat
  /-- `max_num_reqs`, the launch's `seq_lens.shape[0]`. -/
  M : Nat
  /-- The storage of the output `pos` (`int64`). -/
  pos : Buf .i64
  /-- The storage of the output `seq_lens` (`int32`). -/
  seq_lens : Buf .i32
  /-- The storage of `idx_mapping` (`int64`). -/
  idx_mapping : Buf .i64
  /-- The storage of `query_start_loc` (`int32`). -/
  query_start_loc : Buf .i32
  /-- The storage of `num_computed_tokens` (`int32`). -/
  num_computed_tokens : Buf .i32

/-- The `int32` value memory `m` holds at element `i` of `b`, as an `Int` (`0` when the bytes
are absent). -/
def rd32 (b : Buf .i32) (m : Mem) (i : Int) : Int :=
  match m.read .i32 (b.ptr i).addr with | some v => v.toInt | none => 0

/-- The `int64` value memory `m` holds at element `i` of `b`, as an `Int` (`0` when the bytes
are absent). -/
def rd64 (b : Buf .i64) (m : Mem) (i : Int) : Int :=
  match m.read .i64 (b.ptr i).addr with | some v => v.toInt | none => 0

/-- Request `r`'s state slot, `idx_mapping[r]`. -/
def slot (d : Inst) (m : Mem) (r : Nat) : Int := rd64 d.idx_mapping m r

/-- Request `r`'s computed tokens, `num_computed_tokens[idx_mapping[r]]`. -/
def computed (d : Inst) (m : Mem) (r : Nat) : Int := rd32 d.num_computed_tokens m (slot d m r)

/-- Entry `r` of `query_start_loc`: request `r`'s first token in the batch. -/
def qsl (d : Inst) (m : Mem) (r : Nat) : Int := rd32 d.query_start_loc m r

/-- Request `r`'s query length, `query_start_loc[r + 1] - query_start_loc[r]`. -/
def qlen (d : Inst) (m : Mem) (r : Nat) : Int := qsl d m (r + 1) - qsl d m r

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The grid `(num_reqs + 1, 1, 1)` has fewer than `2^31` programs (the kernel's limit). -/
  grid_lt : d.N + 1 < 2 ^ 31
  /-- `max_num_reqs` is an `int32` runtime argument with `max_num_reqs + 1023` within `int32`
  (the zeroing loop's bound `tl.RangeOk`, a limit of the kernel). -/
  M_le : d.M + 1024 ≤ 2 ^ 31
  /-- `idx_mapping` has the batch's entries, `query_start_loc` their `num_reqs + 1` offsets. -/
  idx_len : d.N ≤ d.idx_mapping.len
  qsl_len : d.N + 1 ≤ d.query_start_loc.len
  /-- `seq_lens` holds the batch's entries and the `max_num_reqs` the launch passes (the host
  passes its length). -/
  sl_N : d.N ≤ d.seq_lens.len
  sl_M : d.M ≤ d.seq_lens.len
  /-- The outputs share no byte with each other or with an input (vLLM's separate buffers). -/
  disj : Disjoint d.pos.bytes d.seq_lens.bytes ∧
    Disjoint d.pos.bytes d.idx_mapping.bytes ∧ Disjoint d.pos.bytes d.query_start_loc.bytes ∧
    Disjoint d.pos.bytes d.num_computed_tokens.bytes ∧
    Disjoint d.seq_lens.bytes d.idx_mapping.bytes ∧
    Disjoint d.seq_lens.bytes d.query_start_loc.bytes ∧
    Disjoint d.seq_lens.bytes d.num_computed_tokens.bytes

/-- The model on an instance: the kernel's arguments are the buffers' first elements and
`max_num_reqs` as an `int32`. -/
def model (d : Inst) : Kernel :=
  prepare_pos_seq_lens_kernel (d.pos.ptr 0) (d.seq_lens.ptr 0) (d.idx_mapping.ptr 0)
    (d.query_start_loc.ptr 0) (d.num_computed_tokens.ptr 0) (IntV.ofNat d.M)

/-- The host's launch (line 98): a 1-D grid of `num_reqs + 1` programs; the source's one block
shape `[1024]`; `enable_fp_fusion` is Triton's default (`true`); the default `num_warps = 4` and
`num_stages = 3` change no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 (d.N + 1) [[1024]] (fusion := true)

/-- The inputs covered: the five buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); `query_start_loc` is nondecreasing on the batch's `num_reqs + 1` entries from a
nonnegative start (the host's running sums from `0`), its last entry at most `pos`'s length and
at most `2^22` (the batch's tokens); every request's slot is an element of
`num_computed_tokens`, whose value there is a context length (`0` to `2^22`). The other slots,
the entries past the batch, and the initial outputs are arbitrary. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.pos.Valid init ∧ d.seq_lens.Valid init ∧ d.idx_mapping.Valid init ∧
    d.query_start_loc.Valid init ∧ d.num_computed_tokens.Valid init ∧
    0 ≤ qsl d init 0 ∧
    (∀ r, r < d.N → qsl d init r ≤ qsl d init (r + 1)) ∧
    qsl d init d.N ≤ d.pos.len ∧ qsl d init d.N ≤ 2 ^ 22 ∧
    (∀ r, r < d.N → 0 ≤ slot d init r ∧ slot d init r < d.num_computed_tokens.len) ∧
    (∀ r, r < d.N → 0 ≤ computed d init r ∧ computed d init r ≤ 2 ^ 22)

/-! ## The specification -/

/-- The elements of `pos` the launch writes: request `r`'s tokens
`query_start_loc[r] … query_start_loc[r + 1] - 1`, for every request `r` of the batch. -/
def posOut (d : Inst) (init : Mem) : Set Nat :=
  {k : Nat | ∃ r : Nat, r < d.N ∧ qsl d init r ≤ (k : Int) ∧ (k : Int) < qsl d init (r + 1)}

/-- What the launch leaves: for every request `r < num_reqs`, `seq_lens[r]` holds
`computed r + qlen r` and `pos[query_start_loc[r] + j]` holds `computed r + j` for every `j < qlen r`;
for every `num_reqs ≤ r < max_num_reqs`, `seq_lens[r]` holds `0`; every byte outside these
elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r : Nat, r < d.N → ∃ v : I32,
      fin.read .i32 (d.seq_lens.ptr (r : Int)).addr = some v ∧
      v.toInt = computed d init r + qlen d init r) ∧
    (∀ r : Nat, r < d.N → ∀ j : Nat, (j : Int) < qlen d init r → ∃ v : I64,
      fin.read .i64 (d.pos.ptr (qsl d init r + (j : Int))).addr = some v ∧
      v.toInt = computed d init r + (j : Int)) ∧
    (∀ r : Nat, d.N ≤ r → r < d.M → ∃ v : I32,
      fin.read .i32 (d.seq_lens.ptr (r : Int)).addr = some v ∧ v.toInt = 0) ∧
    ∀ z, z ∉ d.seq_lens.elems {r | r < d.N ∨ r < d.M} ∪ d.pos.elems (posOut d init) →
      fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_prepare_pos_seq_lens_kernel` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- Two buffers whose byte ranges are ordered do not overlap. -/
theorem bytes_disjoint_of_le {d d' : DType} {b : Buf d} {c : Buf d'}
    (h : b.base + b.len * d.bytes ≤ c.base) : Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hcx => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hcx
  have hb := d.bytes_pos
  linarith

/-- An instance with the host's layout: `N` requests, `max_num_reqs = M` (the length of
`seq_lens` and of `num_computed_tokens`), a `pos` of `T` elements and a `query_start_loc` of
`N + 1`; the storages one after another from `2^30` with `2^20`-byte gaps, the outputs first.
Each storage takes 8 bytes per element, so every base is a multiple of 8. -/
def mk (N M T : Nat) : Inst :=
  let p0 : Int := 2 ^ 30
  let s0 : Int := p0 + T * 8 + 2 ^ 20
  let i0 : Int := s0 + M * 8 + 2 ^ 20
  let q0 : Int := i0 + N * 8 + 2 ^ 20
  let c0 : Int := q0 + (N + 1) * 8 + 2 ^ 20
  { N := N, M := M, pos := ⟨p0, T⟩, seq_lens := ⟨s0, M⟩, idx_mapping := ⟨i0, N⟩,
    query_start_loc := ⟨q0, N + 1⟩, num_computed_tokens := ⟨c0, M⟩ }

theorem mk_ok (N M T : Nat) (h1 : N + 1 < 2 ^ 31) (h2 : M + 1024 ≤ 2 ^ 31) (h3 : N ≤ M) :
    (mk N M T).ok := by
  refine ⟨h1, h2, le_refl _, le_refl _, h3, le_refl _, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals
    refine bytes_disjoint_of_le ?_; simp only [mk, DType.bytes]; push_cast; nlinarith

/-- The memory every byte of which is present and `0`. -/
def zeroMem : Mem := fun _ => some 0

theorem readBytes_zeroMem : ∀ (n : Nat) (a : Int),
    zeroMem.readBytes a n = some (List.replicate n 0) := by
  intro n
  induction n with
  | zero => intro a; rfl
  | succ n ih =>
    intro a
    show (match zeroMem a, zeroMem.readBytes (a + 1) n with
      | some b, some bs => some (b :: bs)
      | _, _ => none) = _
    rw [ih (a + 1)]
    rfl

theorem zeroMem_read {d : DType} {a : Int} (h : Mem.Aligned d a) :
    zeroMem.read d a = some (DType.ofBits d 0) := by
  simp only [Mem.read, h, ↓reduceIte, readBytes_zeroMem, Option.map_some]
  have hlv : ∀ n : Nat, Bytes.leValue (List.replicate n 0) = 0 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih => simp only [List.replicate, Bytes.leValue, ih]; norm_num
  rw [hlv]

theorem rd32_zeroMem (b : Buf .i32) (i : Int) : rd32 b zeroMem i = 0 := by
  rw [rd32]
  by_cases h : Mem.Aligned .i32 (b.ptr i).addr
  · rw [zeroMem_read h]; decide
  · simp only [Mem.read, h, ↓reduceIte]

theorem rd64_zeroMem (b : Buf .i64) (i : Int) : rd64 b zeroMem i = 0 := by
  rw [rd64]
  by_cases h : Mem.Aligned .i64 (b.ptr i).addr
  · rw [zeroMem_read h]; decide
  · simp only [Mem.read, h, ↓reduceIte]

theorem valid_zeroMem {d : DType} (b : Buf d) (h0 : 0 ≤ b.base)
    (h1 : b.base + b.len * d.bytes ≤ 2 ^ 64) (h2 : b.base % d.bytes = 0) : b.Valid zeroMem :=
  ⟨h0, h1, h2, fun _ _ => by simp [zeroMem]⟩

/-- A covered instance and memory: one request in a batch of `max_num_reqs = 1`, 16 positions,
every buffer `0` (slot `0`, no computed token, an empty query). -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 1 1 16, zeroMem, mk_ok 1 1 16 (by norm_num) (by norm_num) (by norm_num), ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals try (apply valid_zeroMem <;> norm_num [mk, DType.bytes])
  all_goals simp only [qsl, slot, computed, rd32_zeroMem, rd64_zeroMem]
  all_goals norm_num [mk]

/-! ## The setup's launches

The host's test cases (source lines 142-165) and vLLM's serving sizes: `N` requests,
`max_num_reqs = M`, a `pos` of `max_num_tokens = T` elements (`Inst.ok` does not depend on the
lengths of the requests, which `Pre` bounds). -/

/-- Decode: 8 requests of one token, `max_num_reqs = 256`, 8192 positions. -/
def setup_decode : Inst := mk 8 256 8192
theorem setup_decode_ok : setup_decode.ok := mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Prefill: 7 requests of 1, 1023, 1024, 1025, 3000, 0 and 17 tokens, `max_num_reqs = 16`. -/
def setup_prefill : Inst := mk 7 16 8192
theorem setup_prefill_ok : setup_prefill.ok := mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- A full batch: 128 requests, `max_num_reqs = 128` (the zeroing loop runs no trip). -/
def setup_full_batch : Inst := mk 128 128 512
theorem setup_full_batch_ok : setup_full_batch.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- One request, `max_num_reqs = 1` (Triton's compile-time `1`). -/
def setup_one_request : Inst := mk 1 1 16
theorem setup_one_request_ok : setup_one_request.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- 3 requests, `max_num_reqs = 4096` (four trips of the zeroing loop). -/
def setup_long_padding : Inst := mk 3 4096 64
theorem setup_long_padding_ok : setup_long_padding.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Speculative decoding: 64 requests of 4 tokens, `max_num_reqs = 1024`. -/
def setup_spec_decode : Inst := mk 64 1024 8192
theorem setup_spec_decode_ok : setup_spec_decode.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Serving: 1024 requests (1000 decodes, 24 chunks of 299 tokens), `max_num_reqs = 1024`,
8192 positions. -/
def setup_serving : Inst := mk 1024 1024 8192
theorem setup_serving_ok : setup_serving.ok := mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- A long prefill: one request of `2^20` tokens after up to `2^21` computed ones,
`max_num_reqs = 256`. -/
def setup_long_prefill : Inst := mk 1 256 (2 ^ 20)
theorem setup_long_prefill_ok : setup_long_prefill.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)
/-- Large serving sizes: `max_num_seqs = 8192` requests, every one scheduled, and
`max_num_tokens = 2^22` positions. -/
def setup_serving_max : Inst := mk 8192 8192 (2 ^ 22)
theorem setup_serving_max_ok : setup_serving_max.ok :=
  mk_ok _ _ _ (by norm_num) (by norm_num) (by norm_num)

end Claims.PreparePosSeqLens
