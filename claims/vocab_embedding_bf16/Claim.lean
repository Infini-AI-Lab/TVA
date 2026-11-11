import TVA

/-!
# `vocab_embedding_bf16`: vocabulary-parallel embedding lookup, bfloat16, SGLang

The kernel is `_vocab_parallel_embedding_kernel` of `source/vocab_parallel_embedding.py` (lines
54-99; it reads the module-level names `triton` and `tl`, lines 50-51). The host
`vocab_parallel_embedding` (lines 102-143) launches it with one program per (token, block of
`BLOCK_H` columns), on `int64` (or `int32`) token ids, a `bfloat16` weight with unit column stride
and a fresh `bfloat16` output; it launches nothing for an empty input.

What is claimed: every covered launch ends normally; element `(r, col)` of the output holds, bit
for bit, the weight's element at column `col` of the shard's local row for token `r`'s id when the
id is in the shard's original range (local row `id - org_vocab_start_index`) or added range (local
row `id - added_offset`), and `+0.0` otherwise; every byte outside the output's
`n_tokens * hidden_dim` elements is unchanged.

The conditions: `int64` or `int32` ids; `BLOCK_H` a power of two at most `1024`; `hidden_dim` a
positive multiple of `128`, at most `2^17`; at most `2^22` ids; the grid within CUDA's limits; the
vocabulary values in `[0, 2^63)`; the output holds the rows and shares no byte with the inputs; an
id in a range names a local row inside the weight. Ids outside both ranges, and every weight value,
are covered.

The lookup is a copy, so the specification is exact and uses no device fact.
-/

namespace Claims.VocabEmbedding

open TVA

/-! ## The model -/

/-- `_vocab_parallel_embedding_kernel` (source lines 54-99). `input_ptr` is `*dI` (`int64` or
`int32` token ids), `weight_ptr` and `out_ptr` `*bfloat16`; the eight `tl.constexpr` scalars are
Lean parameters. The vocabulary values are `Int` (`added_offset`, line 85, may be negative). -/
def vocab_parallel_embedding_kernel {dI : DType} (input_ptr : Ptr dI) (weight_ptr : Ptr .bf16)
    (out_ptr : Ptr .bf16) (hidden_dim weight_stride0 : Nat)
    (org_vocab_start_index org_vocab_end_index num_org_vocab_padding
      added_vocab_start_index added_vocab_end_index : Int) (BLOCK_H : Nat) : Kernel where
  body ctx := do
    -- line 73: `row = tl.program_id(0).to(tl.int64)`
    let row := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id ctx 0))).item
    -- line 74
    let col_block := tl.program_id ctx 1
    -- line 75: the constexpr `BLOCK_H` is built in the product's dtype, `int32` (wrapping)
    let cols := col_block * (IntV.ofNat BLOCK_H : I32) + tl.arange 0 BLOCK_H
    -- line 76: a signed `int32` comparison with the constexpr `hidden_dim` (`< 2^31`, `Inst.ok`)
    let col_mask := cols.ltS (IntV.ofNat hidden_dim : I32)
    -- line 78: one token id, `.to(tl.int64)` (sign-extending an `int32` id; the identity on an
    -- `int64` one)
    let t0 ← tl.load1 (input_ptr + row)
    let token := (← tl.convert dI .i64 (Tensor.scalar t0)).item
    -- lines 79-82: each constexpr becomes a scalar tensor of its own type (`int32` or `int64`,
    -- a value in `[0, 2^63)`, `Inst.ok`) promoted to the comparison's dtype `int64` (exact): the
    -- model writes the promoted constants; `&` of two `int1` is `Tensor.and`
    let org_vocab_mask :=
      Tensor.and ((Tensor.scalar (IntV.ofInt org_vocab_start_index : I64)).le (Tensor.scalar token))
        ((Tensor.scalar token).lt (Tensor.scalar (IntV.ofInt org_vocab_end_index : I64)))
    let added_vocab_mask :=
      Tensor.and ((Tensor.scalar (IntV.ofInt added_vocab_start_index : I64)).le (Tensor.scalar token))
        ((Tensor.scalar token).lt (Tensor.scalar (IntV.ofInt added_vocab_end_index : I64)))
    -- line 83
    let valid := org_vocab_mask.or added_vocab_mask
    -- lines 85-89: every operand is a compile-time value, so Python's integer arithmetic; the
    -- name then holds a scalar tensor of the number's own type, promoted to `int64` at line 91
    -- (exact: `added_offset` is in `[-2^63, 2^63)`, `Inst.ok`), the constant the model builds
    let added_offset : Int :=
      added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) -
        num_org_vocab_padding
    -- lines 90-92: `int64` subtractions (wrapping) of the promoted constants, then `tl.where`
    let local_id :=
      tl.where org_vocab_mask (Tensor.scalar (token - (IntV.ofInt org_vocab_start_index : I64)))
        (Tensor.scalar (token - (IntV.ofInt added_offset : I64)))
    -- line 93: the Python `0` is built in `int64`, the other value's dtype
    let local_id := tl.where valid local_id (Tensor.scalar (IntV.ofInt 0 : I64))
    -- lines 94-98: `(weight_ptr + local_id * weight_stride0) + cols`, left to right: the
    -- constexpr stride is built in `int64` (the product's dtype), the scalar pointer is advanced by
    -- the `int64` product, then by each `int32` lane of `cols`; the mask `col_mask & valid`
    -- broadcasts the scalar `valid`; `other=0.0` is the `float32` `+0`, converted to the
    -- pointer's dtype
    let other ← tl.convert .f32 .bf16 (tl.full [BLOCK_H] (FloatV.litF32 (FloatV.ofPat 0)))
    let vals ← tl.load (weight_ptr + local_id.item * (IntV.ofNat weight_stride0 : I64) + cols)
      (some (Tensor.and col_mask (bc valid.item))) (some other)
    -- line 99: `(out_ptr + row * hidden_dim) + cols`: the `int64` product with the constexpr
    -- built in `int64`, then the `int32` lanes; `vals` already has the pointer's dtype
    tl.store (out_ptr + row * (IntV.ofNat hidden_dim : I64) + cols) vals (some col_mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of the kernel through `vocab_parallel_embedding`: the token
ids' dtype, the three tensors (by their storage), the compile-time values, and the shape. -/
structure Inst where
  /-- The token ids' dtype (`int64`, SGLang's `input_ids`; or `int32`). -/
  dI : DType
  /-- The number of token ids (`input_.numel()`, the grid's `x`). -/
  n_tokens : Nat
  /-- `hidden_dim` (the weight's second dimension; a `tl.constexpr`). -/
  hidden_dim : Nat
  /-- `BLOCK_H` (the host's `min(1024, next_power_of_2(hidden_dim))`). -/
  BLOCK_H : Nat
  /-- `weight.stride(0)`, in elements (a `tl.constexpr`). -/
  weight_stride0 : Nat
  /-- The shard's ranges and the original vocabulary's padding rows (`tl.constexpr`s). -/
  org_vocab_start_index : Int
  org_vocab_end_index : Int
  num_org_vocab_padding : Int
  added_vocab_start_index : Int
  added_vocab_end_index : Int
  /-- The token ids (contiguous, any shape: `n_tokens` elements). -/
  input_ : Buf dI
  /-- The weight `[num_embeddings_per_partition, hidden_dim]`, unit column stride, row stride
  `weight_stride0` (its storage from its first element). -/
  weight : Buf .bf16
  /-- The output `[*input_.shape, hidden_dim]`, contiguous, a fresh allocation. -/
  output : Buf .bf16

/-- `added_offset` (line 85). -/
def addedOffset (d : Inst) : Int :=
  d.added_vocab_start_index - (d.org_vocab_end_index - d.org_vocab_start_index) -
    d.num_org_vocab_padding

/-- The signed value of an `int64` or `int32` element (the id dtypes covered; `0` for any other
dtype, which no covered instance has). -/
def ivOf {dt : DType} (x : dt.Val) : Int :=
  match dt, x with
  | .i64, y => y.toInt
  | .i32, y => y.toInt
  | _, _ => 0

/-- The id at element `r` of `input_` in memory `m`, if present. -/
def tokAt (d : Inst) (m : Mem) (r : Nat) : Option Int :=
  (m.read d.dI (d.input_.ptr (r : Int)).addr).map ivOf

/-- The grid's `y`: `cdiv(hidden_dim, BLOCK_H)`. -/
def nColBlocks (d : Inst) : Nat := (d.hidden_dim + d.BLOCK_H - 1) / d.BLOCK_H

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The ids are `int64` or `int32` (the host asserts one of the two). -/
  dI_ok : d.dI = .i64 ∨ d.dI = .i32
  /-- `BLOCK_H` is a power of two of at most `1024` (the host's is). -/
  block : ∃ e, e ≤ 10 ∧ d.BLOCK_H = 2 ^ e
  /-- The `int32` columns `col_block * BLOCK_H + l` (below `hidden_dim + BLOCK_H`) do not wrap,
  and `hidden_dim` is an `int32` constant (a limit of the kernel, README "Exclusions"). -/
  hidden_lt : d.hidden_dim + d.BLOCK_H ≤ 2 ^ 31
  /-- The grid `(n_tokens, cdiv(hidden_dim, BLOCK_H))` is within CUDA's limits and has fewer
  than `2^31` programs (`Launch.Valid`). -/
  tokens_lt : d.n_tokens < 2 ^ 31
  cols_lt : nColBlocks d < 2 ^ 16
  grid_lt : d.n_tokens * nColBlocks d < 2 ^ 31
  /-- `hidden_dim` is a model dimension (the embedding width): positive, a multiple of 128, at
  most `2^17` (real model shapes, README, Exclusions). -/
  hidden_pos : 0 < d.hidden_dim
  hidden_dvd : d.hidden_dim % 128 = 0
  hidden_le : d.hidden_dim ≤ 2 ^ 17
  /-- `n_tokens` (the batch's total number of tokens) is a context length: at most `2^22` (real
  model shapes, README, Exclusions). -/
  tokens_le : d.n_tokens ≤ 2 ^ 22
  /-- Every token row has an id. -/
  input_len : d.n_tokens ≤ d.input_.len
  /-- The vocabulary values are a rank's indices and counts: in `[0, 2^63)`, so each is an
  `int32` or `int64` constant, exact in `int64`; `added_offset` is in `[-2^63, 2^63)` too (an
  `int32` or `int64` constant: at `2^63` or above it would be a `uint64` one). -/
  org_start : 0 ≤ d.org_vocab_start_index ∧ d.org_vocab_start_index < 2 ^ 63
  org_end : 0 ≤ d.org_vocab_end_index ∧ d.org_vocab_end_index < 2 ^ 63
  pad : 0 ≤ d.num_org_vocab_padding ∧ d.num_org_vocab_padding < 2 ^ 63
  add_start : 0 ≤ d.added_vocab_start_index ∧ d.added_vocab_start_index < 2 ^ 63
  add_end : 0 ≤ d.added_vocab_end_index ∧ d.added_vocab_end_index < 2 ^ 63
  offset : -(2 ^ 63 : Int) ≤ addedOffset d ∧ addedOffset d < 2 ^ 63
  /-- The row stride is an `int64` constant. -/
  stride_lt : d.weight_stride0 < 2 ^ 63
  /-- The output holds `n_tokens` rows of `hidden_dim`. -/
  out_len : d.n_tokens * d.hidden_dim ≤ d.output.len
  /-- The output shares no byte with the inputs (`torch.empty` allocates it fresh). -/
  out_input : Disjoint d.output.bytes d.input_.bytes
  out_weight : Disjoint d.output.bytes d.weight.bytes

/-- The condition on a token id `v`: an id in the original range names a local row whose
`hidden_dim` columns lie in the weight's storage (and below `2^63`); likewise for the added range,
against `added_offset`. Ids outside both ranges are unconstrained (`-1`, ids of other ranks, ids
past the vocabulary, the `int64` extremes). -/
def TokOk (d : Inst) (v : Int) : Prop :=
  ((d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index) →
      (v - d.org_vocab_start_index) * (d.weight_stride0 : Int) + (d.hidden_dim : Int) ≤
        (d.weight.len : Int) ∧
      (v - d.org_vocab_start_index) * (d.weight_stride0 : Int) + (d.hidden_dim : Int) < 2 ^ 63) ∧
  ((d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index) →
      0 ≤ v - addedOffset d ∧
      (v - addedOffset d) * (d.weight_stride0 : Int) + (d.hidden_dim : Int) ≤
        (d.weight.len : Int) ∧
      (v - addedOffset d) * (d.weight_stride0 : Int) + (d.hidden_dim : Int) < 2 ^ 63)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and every id the launch reads meets `TokOk` (an id of the shard names a row of the
weight, SETUP.md). Any `bfloat16` values in the weight and the output. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.input_.Valid init ∧ d.weight.Valid init ∧ d.output.Valid init ∧
    ∀ r, r < d.n_tokens → ∀ v, tokAt d init r = some v → TokOk d v

/-! ## The specification -/

/-- The output elements the launch writes. -/
def outElems (d : Inst) : Set Nat := {j | j < d.n_tokens * d.hidden_dim}

/-- What element `col` of the output's row for the id `v` holds: the weight's element at column
`col` of local row `v - org_vocab_start_index` (original range) or `v - added_offset` (added
range), as memory holds it at the launch, bit for bit; `+0` otherwise. The original range wins
when the two overlap (the kernel's `tl.where(org_vocab_mask, …)`). -/
def outVal (d : Inst) (init : Mem) (v : Int) (col : Nat) (c : BF16) : Prop :=
  if d.org_vocab_start_index ≤ v ∧ v < d.org_vocab_end_index then
    init.read .bf16
        (d.weight.ptr ((v - d.org_vocab_start_index) * (d.weight_stride0 : Int) + (col : Int))).addr =
      some c
  else if d.added_vocab_start_index ≤ v ∧ v < d.added_vocab_end_index then
    init.read .bf16
        (d.weight.ptr ((v - addedOffset d) * (d.weight_stride0 : Int) + (col : Int))).addr =
      some c
  else c = (FloatV.ofPat 0 : BF16)

/-- What the launch leaves: element `r * hidden_dim + col` of the output holds `outVal` of the id
at element `r` of `input_`; every byte outside the output's first `n_tokens * hidden_dim`
elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ r col, r < d.n_tokens → col < d.hidden_dim → ∃ v, tokAt d init r = some v ∧
      ∃ c : BF16,
        fin.read .bf16 (d.output.ptr (((r * d.hidden_dim + col : Nat) : Int))).addr = some c ∧
        outVal d init v col c) ∧
    ∀ z, z ∉ d.output.elems (outElems d) → fin z = init z

/-- The model on an instance: the kernel's arguments are the buffers' first elements and the
compile-time values. -/
def model (d : Inst) : Kernel :=
  vocab_parallel_embedding_kernel (d.input_.ptr 0) (d.weight.ptr 0) (d.output.ptr 0)
    d.hidden_dim d.weight_stride0 d.org_vocab_start_index d.org_vocab_end_index
    d.num_org_vocab_padding d.added_vocab_start_index d.added_vocab_end_index d.BLOCK_H

/-- The host's launch (lines 126-142): the grid `(n_tokens, cdiv(hidden_dim, BLOCK_H))`, the
source's block shape `[BLOCK_H]`, `enable_fp_fusion` Triton's default (`true`); `num_warps = 8`
and `num_stages = 3` change no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch :=
  { grid := ![d.n_tokens, nColBlocks d, 1], fusion := true, blocks := [[d.BLOCK_H]] }

/-! ## The claim -/

/-- **The claim**: every covered launch of `_vocab_parallel_embedding_kernel` ends normally with
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

/-- An instance with the host's layout: ids, weight and output one after another from `2^30`,
with `2^20`-byte gaps; the weight has `rows` rows of stride `stride`. -/
def mk (dI : DType) (n hidden BLOCK stride rows : Nat) (os oe pad as ae : Int) : Inst :=
  let i0 : Int := 2 ^ 30
  let w0 : Int := i0 + n * 8 + 2 ^ 20
  let o0 : Int := w0 + rows * stride * 2 + 2 ^ 20
  { dI := dI, n_tokens := n, hidden_dim := hidden, BLOCK_H := BLOCK, weight_stride0 := stride,
    org_vocab_start_index := os, org_vocab_end_index := oe, num_org_vocab_padding := pad,
    added_vocab_start_index := as, added_vocab_end_index := ae,
    input_ := ⟨i0, n⟩, weight := ⟨w0, rows * stride⟩, output := ⟨o0, n * hidden⟩ }

theorem mk_ok (dI : DType) (n hidden BLOCK stride rows : Nat) (os oe pad as ae : Int)
    (hd : dI = .i64 ∨ dI = .i32) (hb : ∃ e, e ≤ 10 ∧ BLOCK = 2 ^ e)
    (h1 : hidden + BLOCK ≤ 2 ^ 31) (h2 : n < 2 ^ 31) (h3 : (hidden + BLOCK - 1) / BLOCK < 2 ^ 16)
    (h4 : n * ((hidden + BLOCK - 1) / BLOCK) < 2 ^ 31)
    (h5 : 0 ≤ os ∧ os < 2 ^ 63) (h6 : 0 ≤ oe ∧ oe < 2 ^ 63) (h7 : 0 ≤ pad ∧ pad < 2 ^ 63)
    (h8 : 0 ≤ as ∧ as < 2 ^ 63) (h9 : 0 ≤ ae ∧ ae < 2 ^ 63)
    (h10 : -(2 ^ 63 : Int) ≤ as - (oe - os) - pad ∧ as - (oe - os) - pad < 2 ^ 63) (h11 : stride < 2 ^ 63)
    (h12 : 0 < hidden := by norm_num) (h13 : hidden % 128 = 0 := by norm_num)
    (h14 : hidden ≤ 2 ^ 17 := by norm_num) (h15 : n ≤ 2 ^ 22 := by norm_num) :
    (mk dI n hidden BLOCK stride rows os oe pad as ae).ok := by
  refine ⟨hd, hb, h1, h2, h3, h4, h12, h13, h14, h15, le_refl _, h5, h6, h7, h8, h9, h10, h11, le_refl _, ?_, ?_⟩
  · refine (bytes_disjoint_of_le ?_).symm
    rcases hd with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    simp only [mk, DType.bytes]; push_cast; nlinarith

/-- A covered instance and memory: one `int64` id (`0`, in the original range `[0, 512)`),
`hidden_dim = 128`, on the all-zero memory. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk .i64 1 128 128 128 512 0 512 0 1000 1000, zeroMem,
    mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inl rfl) ⟨7, by norm_num, rfl⟩ (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num), ?_⟩
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ?_⟩
  rotate_left 9
  · intro r hr v hv
    have hr0 : r = 0 := by simp [mk] at hr; omega
    subst hr0
    have hal : Mem.Aligned .i64 1073741824 := by simp only [Mem.Aligned, DType.bytes]; norm_num
    simp only [tokAt, mk, Buf.ptr_addr, DType.bytes] at hv
    norm_num at hv
    obtain ⟨a, ha, rfl⟩ := hv
    rw [zeroMem_read hal, Option.some.injEq] at ha
    subst ha
    have h0 : ivOf (DType.ofBits DType.i64 0) = 0 := by decide
    rw [h0]
    refine ⟨fun _ => ⟨by norm_num [mk], by norm_num [mk]⟩, fun h => ?_⟩
    simp [mk] at h
  all_goals (simp only [mk, DType.bytes]; norm_num)

/-! ## The setup's launches

The host's test cases (source lines 199-226) whose `hidden_dim` is a model dimension (README,
Coverage: hidden 257 and 7 are not), with the ranges `shard_indices` computes, the weight
`[rows, hidden_dim]` contiguous (`randn`), and `BLOCK_H = min(1024, next_power_of_2(hidden_dim))`.
The empty input (`(0,)`, line 207) makes no launch (the host returns at line 124). -/

/-- Vocabulary 128256, TP 8 rank 3, hidden 4096, 8192 `int64` ids. -/
def setup_v128256_tp8 : Inst := mk .i64 8192 4096 1024 4096 16032 48096 64128 0 128256 128256
theorem setup_v128256_tp8_ok : setup_v128256_tp8.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inl rfl) ⟨10, le_refl _, rfl⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- Vocabulary 129280, TP 8 rank 7, hidden 7168, 1 `int64` id. -/
def setup_v129280_tp8 : Inst := mk .i64 1 7168 1024 7168 16160 113120 129280 0 129280 129280
theorem setup_v129280_tp8_ok : setup_v129280_tp8.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inl rfl) ⟨10, le_refl _, rfl⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- Vocabulary 151936, TP 4 rank 0, hidden 2048, ids of shape `(3, 77)` (`int64`). -/
def setup_v151936_tp4 : Inst := mk .i64 231 2048 1024 2048 37984 0 37984 0 151936 151936
theorem setup_v151936_tp4_ok : setup_v151936_tp4.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inl rfl) ⟨10, le_refl _, rfl⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- Vocabulary 152064, TP 2 rank 1, hidden 5120, 513 `int32` ids. -/
def setup_v152064_tp2_i32 : Inst := mk .i32 513 5120 1024 5120 76032 76032 152064 0 152064 152064
theorem setup_v152064_tp2_i32_ok : setup_v152064_tp2_i32.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inr rfl) ⟨10, le_refl _, rfl⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- Vocabulary 50257 plus 3 added, TP 8 rank 7 (47 padding rows), hidden 6144, 300 `int64` ids. -/
def setup_v50257_added_tp8 : Inst := mk .i64 300 6144 1024 6144 6296 44016 50257 47 50260 50260
theorem setup_v50257_added_tp8_ok : setup_v50257_added_tp8.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inl rfl) ⟨10, le_refl _, rfl⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- SETUP.md's other sizes: the code base's test's column slice (a row stride wider than
`hidden_dim`), here hidden 128 of a 256-wide buffer. -/
def setup_column_slice : Inst := mk .i64 64 128 128 256 1000 0 1000 0 1000 1000
theorem setup_column_slice_ok : setup_column_slice.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inl rfl) ⟨7, by norm_num, rfl⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
/-- Serving: hidden 8192, vocabulary 151936 over TP 2 (75968 rows), 65536 `int64` ids. -/
def setup_serving_h8192 : Inst := mk .i64 65536 8192 1024 8192 75968 0 75968 0 151936 151936
theorem setup_serving_h8192_ok : setup_serving_h8192.ok :=
  mk_ok _ _ _ _ _ _ _ _ _ _ _ (Or.inl rfl) ⟨10, le_refl _, rfl⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

end Claims.VocabEmbedding
