import TVA

/-!
# `compute_position_i64`: positions of an extend batch, SGLang

The kernel is `compute_position_kernel` of `source/position.py` (lines 64-91; it reads the
module-level names `triton` and `tl`, lines 34-35). The host `compute_position_triton` (lines
38-61) launches it with one program per request on the `int32` lengths `extend_seq_lens` (`[bs]`)
and `extend_prefix_lens` (`[bs]` when `has_prefix`; any other length means no prefixes), the
`int64` output `positions` (`[extend_seq_lens_sum]`) and the `int32` output `extend_start_loc`
(`[bs]`), with `has_prefix = extend_prefix_lens.shape[0] == bs`. `SETUP.md` lists the launches.

What is claimed: every covered launch ends normally; for every request `b` and position
`j < extend_seq_lens[b]`, element `start(b) + j` of `positions` holds the `int64` value
`p(b) + j`, where `start(b) = extend_seq_lens[0] + … + extend_seq_lens[b - 1]` and `p(b)` is
`extend_prefix_lens[b]` (`0` without prefixes); element `b` of `extend_start_loc` holds the `int32`
value `start(b)`; every byte outside those elements is unchanged.

The conditions: `bs < 2^31`; the buffers hold the batch's entries and are valid; the outputs share
no byte with each other or with the inputs; every length is nonnegative with
`extend_seq_lens[b] + 511 < 2^31`, and with prefixes `extend_prefix_lens[b] + extend_seq_lens[b]
≤ 2^31` (the kernel's `int32` arithmetic does not wrap); every start is below `2^31` (it fits the
`int32` `extend_start_loc`); `positions` holds the sum of the lengths. Any `int32` prefixes within
that bound are covered.

The specification is exact: no float operation, no device fact.
-/
namespace Claims.ComputePosition

open TVA

/-! ## The model -/

/-- `compute_position_kernel` (source lines 64-91). `positions` is `*i64`; `extend_start_loc`,
`extend_prefix_lens` and `extend_seq_lens` are `*i32` (the covered launches); `has_prefix` is
the `tl.constexpr` flag, a Lean parameter. `BLOCK_SIZE: tl.constexpr = 512` (line 72) keeps the
Python value: it is the literal `512` below. -/
def compute_position_kernel (positions : Ptr .i64) (extend_start_loc : Ptr .i32)
    (extend_prefix_lens : Ptr .i32) (extend_seq_lens : Ptr .i32) (has_prefix : Bool) :
    Kernel where
  body ctx := do
    -- line 73: axis 0 of the 1-D grid, an `int32`, `.to(tl.int64)`
    let pid := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 75: `a if has_prefix else 0` with a compile-time condition evaluates one side: the
    -- `int32` element at `extend_prefix_lens + pid` (the pointer moved by the `int64` `pid`),
    -- or the Python number `0`, which the assignment makes an `int32` scalar tensor
    let prefix_len : I32 ← if has_prefix then tl.load1 (extend_prefix_lens + pid)
      else pure (IntV.ofInt 0)
    -- line 76: one `int32` element
    let seq_len ← tl.load1 (extend_seq_lens + pid)
    -- line 79: `tl.cast` makes the Python `0` an `int32` tensor, then converts it to `int64`
    let cumsum_start := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofInt 0 : I32))).item
    -- lines 80-81: `range(pid)` (a plain name, nothing evaluated twice): the bounds `0`, `pid`,
    -- `1` promote to `int64` (`pid` is), so `i` is `int64`; the loop carries `cumsum_start`.
    -- `cumsum_start += …`: the loaded `int32` element converted to `int64`, the wrapping
    -- `int64` sum
    let cumsum_start ← tl.forRange (IntV.ofInt 0 : I64) pid (IntV.ofInt 1 : I64) cumsum_start
      fun i cumsum_start => do
        let x ← tl.load1 (extend_seq_lens + i)
        let x64 := (← tl.convert .i32 .i64 (Tensor.scalar x)).item
        pure (cumsum_start + x64)
    -- line 83: `tl.cdiv(seq_len, BLOCK_SIZE)` is `(seq_len + 511) // 512` in `int32`: the
    -- wrapping `+` of the constant `511` (Python's `512 - 1`), then the truncating `//`
    let num_loop := (← tl.idiv (Tensor.scalar (seq_len + (IntV.ofNat 511 : I32)))
      (Tensor.scalar (IntV.ofNat 512 : I32))).item
    -- line 84: `range(num_loop)` (a plain name), the `int32` bounds `0`, `num_loop`, `1`; the
    -- loop carries nothing
    let _ ← tl.forRange (IntV.ofInt 0 : I32) num_loop (IntV.ofInt 1 : I32) () fun i _ => do
      -- line 85: the `int32` block plus the `int32` scalar `i * 512` (the constexpr built in
      -- `int32`), wrapping
      let offset := tl.arange 0 512 + i * (IntV.ofNat 512 : I32)
      -- lines 86-90: the address `(positions + cumsum_start) + offset`, left to right (the
      -- `int64` scalar, then the `int32` lanes, each taken at its value); the value
      -- `prefix_len + offset` (`int32`, wrapping) converted to the pointer's `int64`; the mask
      -- `offset < seq_len`, signed `int32`
      let vals ← tl.convert .i32 .i64 (prefix_len + offset)
      tl.store (positions + cumsum_start + offset) vals (some (offset.ltS seq_len))
    -- line 91: the `int64` `cumsum_start` converted to the pointer's `int32` (which wraps)
    let start_loc := (← tl.convert .i64 .i32 (Tensor.scalar cumsum_start)).item
    tl.store1 (extend_start_loc + pid) start_loc

/-! ## The conditions -/

/-- What the host chooses: one launch through `compute_position_triton` (lines 38-61), with the
tensors given by their storage (`Buf`). -/
structure Inst where
  /-- The batch size `extend_seq_lens.shape[0]`: the grid's `x`. -/
  bs : Nat
  /-- `has_prefix = extend_prefix_lens.shape[0] == bs`. -/
  hasPrefix : Bool
  /-- The storage of the output `positions` (`int64`). -/
  positions : Buf .i64
  /-- The storage of the output `extend_start_loc` (`int32`). -/
  extend_start_loc : Buf .i32
  /-- The storage of `extend_prefix_lens` (`int32`; never read without prefixes). -/
  extend_prefix_lens : Buf .i32
  /-- The storage of `extend_seq_lens` (`int32`). -/
  extend_seq_lens : Buf .i32

/-- The value element `i` of an `int32` buffer holds in `m` (`0` when absent). -/
def valOf (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- Request `b`'s new tokens, `extend_seq_lens[b]`. -/
def len (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.extend_seq_lens b

/-- Request `b`'s prefix, `extend_prefix_lens[b]` with prefixes, `0` without. -/
def pre (d : Inst) (m : Mem) (b : Nat) : Int :=
  if d.hasPrefix then valOf m d.extend_prefix_lens b else 0

/-- Request `b`'s first position in the flat output: the sum of the lengths of the requests
before it. -/
def start (d : Inst) (m : Mem) (b : Nat) : Int := ∑ i ∈ Finset.range b, len d m i

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The grid `(bs, 1, 1)` within CUDA's limits, fewer than `2^31` programs. -/
  grid_lt : d.bs < 2 ^ 31
  /-- `extend_seq_lens` holds the batch's lengths, `extend_start_loc` one start per request. -/
  lens_len : d.bs ≤ d.extend_seq_lens.len
  start_len : d.bs ≤ d.extend_start_loc.len
  /-- With prefixes, `extend_prefix_lens` holds one per request (`has_prefix` says so). -/
  prefix_len : d.hasPrefix = true → d.bs ≤ d.extend_prefix_lens.len
  /-- The outputs share no byte with each other or with the inputs (fresh `torch.empty`). -/
  disj : Disjoint d.positions.bytes d.extend_start_loc.bytes ∧
    Disjoint d.positions.bytes d.extend_seq_lens.bytes ∧
    Disjoint d.extend_start_loc.bytes d.extend_seq_lens.bytes ∧
    (d.hasPrefix = true → Disjoint d.positions.bytes d.extend_prefix_lens.bytes ∧
      Disjoint d.extend_start_loc.bytes d.extend_prefix_lens.bytes)

/-- Request `b`'s entries are in range: a nonnegative length whose `int32` ceiling
`(len + 511) // 512` does not wrap, with prefixes a prefix whose `int32` sum with each position
`j < len` does not wrap (`p(b) + len ≤ 2^31`), and a start below `2^31` (so the `int32`
`extend_start_loc` holds it). These are limits of the kernel (README, Exclusions). -/
def ReqOk (d : Inst) (m : Mem) (b : Nat) : Prop :=
  0 ≤ len d m b ∧ len d m b + 511 < 2 ^ 31 ∧ pre d m b + len d m b ≤ 2 ^ 31 ∧ start d m b < 2 ^ 31

/-- The model on an instance: the kernel's arguments are the buffers' first elements and the
flag. -/
def model (d : Inst) : Kernel :=
  compute_position_kernel (d.positions.ptr 0) (d.extend_start_loc.ptr 0)
    (d.extend_prefix_lens.ptr 0) (d.extend_seq_lens.ptr 0) d.hasPrefix

/-- The host's launch (line 53): a 1-D grid of `bs` programs; the source's one block shape
`[512]`; `enable_fp_fusion` is Triton's default (`true`); `num_warps = 4`, `num_stages = 3` change
no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 d.bs [[512]] (fusion := true)

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present;
`extend_prefix_lens` with prefixes), every request's entries are in range (`ReqOk`), and
`positions` holds the sum of the lengths (the host's `extend_seq_lens_sum`, or more). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.positions.Valid init ∧ d.extend_start_loc.Valid init ∧ d.extend_seq_lens.Valid init ∧
    (d.hasPrefix = true → d.extend_prefix_lens.Valid init) ∧
    (∀ b, b < d.bs → ReqOk d init b) ∧
    start d init d.bs ≤ (d.positions.len : Int)

/-! ## The specification -/

/-- The elements of `positions` the launch writes: the first `start(bs)`, the sum of all
lengths. -/
def posOut (d : Inst) (init : Mem) : Set Nat := {j : Nat | (j : Int) < start d init d.bs}

/-- What the launch leaves: for every request `b` and position `j < extend_seq_lens[b]`, element
`start(b) + j` of `positions` holds the `int64` value `p(b) + j`; element `b` of
`extend_start_loc` holds the `int32` value `start(b)`; every byte outside these elements is
unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b : Nat, b < d.bs → ∀ j : Nat, (j : Int) < len d init b → ∃ v : I64,
      fin.read .i64 (d.positions.ptr (start d init b + (j : Int))).addr = some v ∧
      v.toInt = pre d init b + (j : Int)) ∧
    (∀ b : Nat, b < d.bs → ∃ v : I32,
      fin.read .i32 (d.extend_start_loc.ptr (b : Int)).addr = some v ∧
      v.toInt = start d init b) ∧
    ∀ z, z ∉ d.positions.elems (posOut d init) →
      z ∉ d.extend_start_loc.elems {b | b < d.bs} → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `compute_position_kernel` ends normally with `Post`. -/
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

/-- An instance with the host's layout: `bs` requests, prefixes when `hp` (`extend_prefix_lens`
of `bs` entries; without prefixes it has none), a `positions` of `total` elements. The storages
one after another from `2^30` with `2^20`-byte gaps. -/
def mk (bs : Nat) (hp : Bool) (total : Nat) : Inst :=
  let s0 : Int := 2 ^ 30
  let p0 : Int := s0 + bs * 4 + 2 ^ 20
  let o0 : Int := p0 + bs * 4 + 2 ^ 20
  let e0 : Int := o0 + total * 8 + 2 ^ 20
  { bs := bs, hasPrefix := hp, positions := ⟨o0, total⟩, extend_start_loc := ⟨e0, bs⟩,
    extend_prefix_lens := ⟨p0, if hp then bs else 0⟩, extend_seq_lens := ⟨s0, bs⟩ }

theorem mk_ok (bs : Nat) (hp : Bool) (total : Nat) (h1 : bs < 2 ^ 31) : (mk bs hp total).ok := by
  refine ⟨h1, le_refl _, le_refl _, fun h => by simp only [mk] at h; simp [mk, h], ?_, ?_, ?_,
    fun _ => ⟨?_, ?_⟩⟩
  · refine bytes_disjoint_of_le ?_
    simp only [mk, DType.bytes]; push_cast; nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    simp only [mk, DType.bytes]; push_cast; nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    simp only [mk, DType.bytes]; push_cast; nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    cases hp <;> simp only [mk, DType.bytes, Bool.false_eq_true, ↓reduceIte] <;> push_cast <;>
      nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    cases hp <;> simp only [mk, DType.bytes, Bool.false_eq_true, ↓reduceIte] <;> push_cast <;>
      nlinarith

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

/-- Every element of an `int32` buffer reads `0` in the all-zero memory. -/
theorem valOf_zeroMem (b : Buf .i32) (i : Nat) : valOf zeroMem b i = 0 := by
  unfold valOf
  by_cases h : Mem.Aligned .i32 (b.ptr (i : Int)).addr
  · rw [zeroMem_read h]
    decide
  · simp only [Mem.read, h, ↓reduceIte]

/-- A covered instance and memory: 37 requests with prefixes, every entry `0`. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 37 true 0, zeroMem, mk_ok _ _ _ (by norm_num), ?_⟩
  have hl : ∀ b, len (mk 37 true 0) zeroMem b = 0 := fun b => valOf_zeroMem _ _
  have hs : ∀ b, start (mk 37 true 0) zeroMem b = 0 := fun b => by simp [start, hl]
  have hp : ∀ b, pre (mk 37 true 0) zeroMem b = 0 := fun b => by
    simp only [pre, mk, ↓reduceIte]; exact valOf_zeroMem _ _
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, fun _ => ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    fun b _ => ?_, ?_⟩
  rotate_left 12
  · simp only [ReqOk, hl, hs, hp]; norm_num
  · rw [hs]; simp
  all_goals (simp only [mk, DType.bytes]; norm_num)

/-! ## The setup's launches

The callers' launches (`SETUP.md`): the batch size, prefixes or not, and the length of
`positions` (the sum of the lengths). The entries are `Pre`'s, not `Inst.ok`'s. -/

/-- The code base's test, 1 request of 7 draft tokens. -/
def setup_test_bs1 : Inst := mk 1 true 7
theorem setup_test_bs1_ok : setup_test_bs1.ok := mk_ok _ _ _ (by norm_num)
/-- The code base's test, 3 requests of 4 draft tokens. -/
def setup_test_bs3 : Inst := mk 3 true 12
theorem setup_test_bs3_ok : setup_test_bs3.ok := mk_ok _ _ _ (by norm_num)
/-- The code base's test, 129 requests of 7 draft tokens. -/
def setup_test_bs129 : Inst := mk 129 true (129 * 7)
theorem setup_test_bs129_ok : setup_test_bs129.ok := mk_ok _ _ _ (by norm_num)
/-- A prefill batch (`ForwardBatch.init_new`): 64 requests in a chunked-prefill budget of 8192
tokens. -/
def setup_prefill_bs64 : Inst := mk 64 true 8192
theorem setup_prefill_bs64_ok : setup_prefill_bs64.ok := mk_ok _ _ _ (by norm_num)
/-- A single long prefill: one request of `2^20` tokens after a prefix. -/
def setup_prefill_long : Inst := mk 1 true (2 ^ 20)
theorem setup_prefill_long_ok : setup_prefill_long.ok := mk_ok _ _ _ (by norm_num)
/-- A large mixed batch: 4096 requests, 65536 tokens. -/
def setup_prefill_bs4096 : Inst := mk 4096 true 65536
theorem setup_prefill_bs4096_ok : setup_prefill_bs4096.ok := mk_ok _ _ _ (by norm_num)
/-- Speculative draft extend (`eagle_worker_common.py`, DFlash, DSpark): 256 requests of 8
tokens. -/
def setup_draft_extend_bs256 : Inst := mk 256 true 2048
theorem setup_draft_extend_bs256_ok : setup_draft_extend_bs256.ok := mk_ok _ _ _ (by norm_num)
/-- The host without prefixes (`extend_prefix_lens` of another length): 16 requests. -/
def setup_no_prefix_bs16 : Inst := mk 16 false 4096
theorem setup_no_prefix_bs16_ok : setup_no_prefix_bs16.ok := mk_ok _ _ _ (by norm_num)
/-- An empty batch: no program. -/
def setup_empty : Inst := mk 0 true 0
theorem setup_empty_ok : setup_empty.ok := mk_ok _ _ _ (by norm_num)

end Claims.ComputePosition
