import TVA

/-!
# `combine_draft_tokens_i32`: vLLM's input ids and logits indices for speculative decoding

The kernel is `_combine_sampled_and_draft_tokens_kernel` of `source/combine_draft_tokens.py`
(lines 60-118). Its host `combine_sampled_and_draft_tokens` (lines 121-159) launches it with one
program per request of the batch, `BLOCK_SIZE = next_power_of_2(S + N)` (`S` the draft table's
width, `num_speculative_steps`; `N = NUM_NEW_SAMPLED_TOKENS`, `0` or `1`). Program `b` reads its
request-state row `i = idx_mapping[b]`, its number of logits `nl = cu_num_logits[b + 1] -
cu_num_logits[b]` and its query end `qe = query_start_loc[b + 1]`; it writes the request's
`logits_indices`, and, for a request past its prefill, the last sampled token and the draft
tokens into its last query slots of `input_ids`.

What is claimed: every covered launch ends normally; for each request `b`:
- `logits_indices[cu_num_logits[b] + j] = qe - nl + j` for every `j < nl`;
- if `seq_lens[b] > prefill_len[i]` (the request is past its prefill):
  - with `N = 1` and `seq_lens[b] - nl ≥ prefill_len[i]`, `input_ids[qe - nl]` holds the `int32`
    truncation of `last_sampled_tokens[i]`;
  - `input_ids[qe - nd + j]` holds the `int32` truncation of `draft_tokens[i, j]` for every
    `j < nd = nl - N`;
- every other byte is unchanged.

The conditions: `N ≤ 1`; `BLOCK_SIZE` a power of two at most `2^20`; the buffers hold the batch's
entries and are valid; the two outputs share no byte with each other or with an input. For each
request: its row `i` lies in `prefill_len` and `last_sampled_tokens`, its `seq_lens` entry is
nonnegative, `N ≤ nl ≤ BLOCK_SIZE`, `query_start_loc[b] + nl ≤ query_start_loc[b + 1]` (the
request's logits are among its query rows), and, when it has drafts, its draft row lies in
`draft_tokens`; `cu_num_logits` and `query_start_loc` start nonnegative and end within the outputs.
Every token value is covered. The kernel has no float operation, so the specification uses no
device fact.
-/

namespace Claims.CombineDraftTokens

open TVA

/-! ## The model -/

/-- `_combine_sampled_and_draft_tokens_kernel` (source lines 60-118). `input_ids_ptr`,
`query_start_loc_ptr`, `seq_lens_ptr`, `prefill_len_ptr` and `cu_num_logits_ptr` are `*i32`;
`idx_mapping_ptr`, `last_sampled_tokens_ptr`, `draft_tokens_ptr` and `logits_indices_ptr` are
`*i64`. `draft_tokens_stride` is a runtime `int32` (an integer argument in `[0, 2^31)`; at `1` it
is the compile-time constant `1`, whose only use, `req_state_idx * draft_tokens_stride` with the
`int64` `req_state_idx`, builds `1` in `int64`: the same value as the model's `int32` `1`
converted, see `model`). `BLOCK_SIZE` and `NUM_NEW_SAMPLED_TOKENS` are compile-time integers. -/
def combine_sampled_and_draft_tokens_kernel
    (input_ids_ptr : Ptr .i32) (idx_mapping_ptr : Ptr .i64) (last_sampled_tokens_ptr : Ptr .i64)
    (query_start_loc_ptr : Ptr .i32) (seq_lens_ptr : Ptr .i32) (prefill_len_ptr : Ptr .i32)
    (draft_tokens_ptr : Ptr .i64) (draft_tokens_stride : I32) (cu_num_logits_ptr : Ptr .i32)
    (logits_indices_ptr : Ptr .i64) (BLOCK_SIZE NUM_NEW_SAMPLED_TOKENS : Nat) : Kernel where
  body ctx := do
    -- line 75: axis 0 of the 1-D grid, an `int32`
    let batch_idx := tl.program_id ctx 0
    -- line 76: one `int64` element
    let req_state_idx ← tl.load1 (idx_mapping_ptr + batch_idx)
    -- lines 79-80: two `int32` elements; `cu_num_logits_ptr + batch_idx + 1` is
    -- `(ptr + batch_idx) + 1`, Python's left-to-right order, the `1` an `int32`
    let cu_num_logits_start ← tl.load1 (cu_num_logits_ptr + batch_idx)
    let cu_num_logits_end ← tl.load1 (cu_num_logits_ptr + batch_idx + (1 : I32))
    -- line 81: the wrapping `int32` difference
    let num_logits := cu_num_logits_end - cu_num_logits_start
    -- line 82: the compile-time integer beside an `int32` tensor is built in `int32`
    let num_draft_tokens := num_logits - (IntV.ofNat NUM_NEW_SAMPLED_TOKENS : I32)
    -- line 85: `int32` lanes `0, …, BLOCK_SIZE - 1`
    let block := tl.arange 0 BLOCK_SIZE
    -- line 86: one `int32` element, `(ptr + batch_idx) + 1`
    let query_end ← tl.load1 (query_start_loc_ptr + batch_idx + (1 : I32))
    -- line 87: the wrapping `int32` difference
    let logits_start := query_end - num_logits
    -- lines 88-92: the address `(logits_indices_ptr + cu_num_logits_start) + block`; the `int32`
    -- values `logits_start + block` are converted to the pointer's `int64`; the mask is the
    -- signed `int32` comparison `block < num_logits`
    let logits_vals ← tl.convert .i32 .i64 (logits_start + block)
    tl.store (logits_indices_ptr + cu_num_logits_start + block) logits_vals
      (some (block.ltS num_logits))
    -- lines 94-95: one `int32` element each; `prefill_len_ptr + req_state_idx` offsets by the
    -- `int64` row
    let seq_len ← tl.load1 (seq_lens_ptr + batch_idx)
    let prefill_len ← tl.load1 (prefill_len_ptr + req_state_idx)
    -- lines 96-98: a runtime `if` on the `int32` comparison `seq_len <= prefill_len`; the
    -- programs that take it end there, the others run the rest of the body. Only this branch
    -- returns, so the statements after it stay in the subset.
    if seq_len.le prefill_len then
      return ()
    else
      -- line 101: the wrapping `int32` difference
      let first_logit_seq_pos := seq_len - num_logits
      -- lines 102-105: `NUM_NEW_SAMPLED_TOKENS > 0 and first_logit_seq_pos >= prefill_len`. The
      -- first operand is a compile-time value: when it is false (`N = 0`) it ends the
      -- evaluation and the `if` is a compile-time `if` whose branch is not visited; when it is
      -- true (`N = 1`) the condition is the `int32` comparison (`a >= b` is `b <= a`), a runtime
      -- `if`. The `int64` token is converted to the pointer's `int32` (`tl.convert`, which
      -- wraps); the address is `input_ids_ptr + logits_start`.
      (if 0 < NUM_NEW_SAMPLED_TOKENS then
        if prefill_len.le first_logit_seq_pos then do
          let last_token_id ← tl.load1 (last_sampled_tokens_ptr + req_state_idx)
          let last32 ← tl.convert .i64 .i32 (Tensor.scalar last_token_id)
          tl.store1 (input_ids_ptr + logits_start) last32.item
        else pure ()
      else pure ())
      -- lines 108-118: a runtime `if` on `num_draft_tokens > 0` (the `0` an `int32`, `a > b` is
      -- `b < a`)
      if (0 : I32).lt num_draft_tokens then do
        -- line 109: the signed `int32` mask
        let mask := block.ltS num_draft_tokens
        -- lines 110-113: the address `(draft_tokens_ptr + req_state_idx * draft_tokens_stride) +
        -- block`; the `int32` stride is promoted to `int64` for the product with the `int64` row;
        -- a masked load without `other`
        let stride64 := (← tl.convert .i32 .i64 (Tensor.scalar draft_tokens_stride)).item
        let draft_tokens ← tl.load (draft_tokens_ptr + req_state_idx * stride64 + block) (some mask)
        -- lines 114-118: the address `((input_ids_ptr + query_end) - num_draft_tokens) + block`,
        -- left to right (`p - k` is `p + (-k)`, the negation in `int32`); the `int64` tokens are
        -- converted to the pointer's `int32` (`tl.convert`, which wraps)
        let draft32 ← tl.convert .i64 .i32 draft_tokens
        tl.store (input_ids_ptr + query_end - num_draft_tokens + block) draft32 (some mask)
      else pure ()

/-! ## The conditions -/

/-- What the host chooses: one launch through `combine_sampled_and_draft_tokens` (source lines
121-159). The tensors are given by their storage (`Buf`); `draft_tokens` is a `[rows, S]` table
with row stride `dstride` (its `stride(0)`). -/
structure Inst where
  /-- `num_reqs = idx_mapping.shape[0]`: the grid, one program per request. -/
  B : Nat
  /-- `BLOCK_SIZE` (the host's `next_power_of_2(num_speculative_steps + N)`). -/
  BLOCK : Nat
  /-- `NUM_NEW_SAMPLED_TOKENS` (the host's `num_new_sampled_tokens`, `0` or `1`). -/
  N : Nat
  /-- `draft_tokens.stride(0)`, the runtime `draft_tokens_stride`. -/
  dstride : Nat
  /-- `input_ids` (`int32`), written. -/
  input_ids : Buf .i32
  /-- `idx_mapping` (`int64`). -/
  idx_mapping : Buf .i64
  /-- `last_sampled_tokens` (`int64`, `[rows, 1]`). -/
  last_sampled : Buf .i64
  /-- `query_start_loc` (`int32`). -/
  query_start_loc : Buf .i32
  /-- `seq_lens` (`int32`). -/
  seq_lens : Buf .i32
  /-- `prefill_len` (`int32`). -/
  prefill_len : Buf .i32
  /-- `draft_tokens` (`int64`). -/
  draft_tokens : Buf .i64
  /-- `cu_num_logits` (`int32`). -/
  cu_num_logits : Buf .i32
  /-- The output `logits_indices` (`int64`). -/
  logits_indices : Buf .i64

/-- The `int32` value memory `m` holds at element `i` of `b`, as an `Int` (`0` when the bytes
are absent). -/
def rd32 (b : Buf .i32) (m : Mem) (i : Int) : Int :=
  match m.read .i32 (b.ptr i).addr with | some v => v.toInt | none => 0

/-- The `int64` value memory `m` holds at element `i` of `b`, as an `Int` (`0` when the bytes
are absent). -/
def rd64 (b : Buf .i64) (m : Mem) (i : Int) : Int :=
  match m.read .i64 (b.ptr i).addr with | some v => v.toInt | none => 0

/-- The `int32` truncation (two's complement, what storing an `int64` into an `int32` pointer
gives) of the `int64` value memory `m` holds at element `i` of `b`. -/
def rdt64 (b : Buf .i64) (m : Mem) (i : Int) : Int := Int.bmod (rd64 b m i) (2 ^ 32)

/-- Request `b`'s request-state row, `idx_mapping[b]`. -/
def ridx (d : Inst) (m : Mem) (b : Nat) : Int := rd64 d.idx_mapping m b

/-- `cu_num_logits[b]`. -/
def cuAt (d : Inst) (m : Mem) (b : Nat) : Int := rd32 d.cu_num_logits m b

/-- Request `b`'s number of logits, `cu_num_logits[b + 1] - cu_num_logits[b]`. -/
def nlog (d : Inst) (m : Mem) (b : Nat) : Int := cuAt d m (b + 1) - cuAt d m b

/-- Request `b`'s number of draft tokens, `nlog - N`. -/
def ndraft (d : Inst) (m : Mem) (b : Nat) : Int := nlog d m b - d.N

/-- `query_start_loc[b]`. -/
def qsl (d : Inst) (m : Mem) (b : Nat) : Int := rd32 d.query_start_loc m b

/-- Request `b`'s query end, `query_start_loc[b + 1]`. -/
def qend (d : Inst) (m : Mem) (b : Nat) : Int := qsl d m (b + 1)

/-- `seq_lens[b]`. -/
def seqAt (d : Inst) (m : Mem) (b : Nat) : Int := rd32 d.seq_lens m b

/-- Request `b`'s prefill length, `prefill_len[idx_mapping[b]]`. -/
def plAt (d : Inst) (m : Mem) (b : Nat) : Int := rd32 d.prefill_len m (ridx d m b)

/-- Request `b` is past its prefill: `seq_lens[b] > prefill_len[idx_mapping[b]]` (the programs
that do not return at line 96). -/
def Decode (d : Inst) (m : Mem) (b : Nat) : Prop := plAt d m b < seqAt d m b

/-- Request `b` gets its last sampled token: past its prefill, `N = 1`, and its first logit's
position `seq_lens[b] - nlog` at least the prefill length (line 102). -/
def LastOn (d : Inst) (m : Mem) (b : Nat) : Prop :=
  Decode d m b ∧ 0 < d.N ∧ plAt d m b ≤ seqAt d m b - nlog d m b

/-- Request `b`'s entries are in range: its row lies in `prefill_len` and `last_sampled_tokens`;
its `seq_lens` entry is nonnegative (a length); it has between `N` and `BLOCK_SIZE` logits; its
logits are among its query rows (`query_start_loc[b] + nlog ≤ query_start_loc[b + 1]`, the
caller's assert at `vllm/v1/worker/gpu/model_runner.py:1437`); and with drafts, its row of
`ndraft` drafts lies in `draft_tokens`. -/
def ReqOk (d : Inst) (m : Mem) (b : Nat) : Prop :=
  0 ≤ ridx d m b ∧ ridx d m b < d.prefill_len.len ∧ ridx d m b < d.last_sampled.len ∧
    0 ≤ seqAt d m b ∧
    (d.N : Int) ≤ nlog d m b ∧ nlog d m b ≤ d.BLOCK ∧
    qsl d m b + nlog d m b ≤ qend d m b ∧
    (0 < ndraft d m b → ridx d m b * d.dstride + ndraft d m b ≤ d.draft_tokens.len)

/-- The instances covered: a grid CUDA accepts, `N ≤ 1` (the host's assert), a block shape
Triton accepts, a stride that is an `int32` argument, the batch's entries in the index buffers,
and outputs that share no byte with each other or with an input. The inputs may overlap one
another (they are only read). -/
structure Inst.ok (d : Inst) : Prop where
  /-- The grid `(B, 1, 1)` within CUDA's limits, fewer than `2^31` programs. -/
  hgrid : d.B < 2 ^ 31
  /-- `num_new_sampled_tokens` is `0` or `1` (source line 133). -/
  hN : d.N ≤ 1
  /-- `BLOCK_SIZE` is a power of two (`next_power_of_2`) with at most `2^20` lanes. -/
  hblock : ∃ e, e ≤ 20 ∧ d.BLOCK = 2 ^ e
  /-- The stride is an `int32` runtime argument. -/
  hstride : d.dstride < 2 ^ 31
  /-- `idx_mapping` and `seq_lens` have an entry per request, `cu_num_logits` and
  `query_start_loc` the `B + 1` entries the programs read. -/
  hidx : d.B ≤ d.idx_mapping.len
  hseq : d.B ≤ d.seq_lens.len
  hcu : d.B + 1 ≤ d.cu_num_logits.len
  hqsl : d.B + 1 ≤ d.query_start_loc.len
  /-- `input_ids` shares no byte with `logits_indices` or with an input. -/
  hids : Disjoint d.input_ids.bytes d.logits_indices.bytes ∧
    Disjoint d.input_ids.bytes d.idx_mapping.bytes ∧
    Disjoint d.input_ids.bytes d.last_sampled.bytes ∧
    Disjoint d.input_ids.bytes d.query_start_loc.bytes ∧
    Disjoint d.input_ids.bytes d.seq_lens.bytes ∧
    Disjoint d.input_ids.bytes d.prefill_len.bytes ∧
    Disjoint d.input_ids.bytes d.draft_tokens.bytes ∧
    Disjoint d.input_ids.bytes d.cu_num_logits.bytes
  /-- `logits_indices` (a fresh `torch.empty`) shares no byte with an input. -/
  hlog : Disjoint d.logits_indices.bytes d.idx_mapping.bytes ∧
    Disjoint d.logits_indices.bytes d.last_sampled.bytes ∧
    Disjoint d.logits_indices.bytes d.query_start_loc.bytes ∧
    Disjoint d.logits_indices.bytes d.seq_lens.bytes ∧
    Disjoint d.logits_indices.bytes d.prefill_len.bytes ∧
    Disjoint d.logits_indices.bytes d.draft_tokens.bytes ∧
    Disjoint d.logits_indices.bytes d.cu_num_logits.bytes

/-- The model on an instance: the kernel's arguments are the buffers' first elements, the stride
as an `int32`, `BLOCK_SIZE` and `NUM_NEW_SAMPLED_TOKENS`. At `dstride = 1` the stride argument is
the compile-time constant `1`, which its only use (an operand of `*` with the `int64` row) builds
in `int64`: the same value as the model's `int32` `1` converted, so the model is the same. -/
def model (d : Inst) : Kernel :=
  combine_sampled_and_draft_tokens_kernel (d.input_ids.ptr 0) (d.idx_mapping.ptr 0)
    (d.last_sampled.ptr 0) (d.query_start_loc.ptr 0) (d.seq_lens.ptr 0) (d.prefill_len.ptr 0)
    (d.draft_tokens.ptr 0) (IntV.ofNat d.dstride) (d.cu_num_logits.ptr 0)
    (d.logits_indices.ptr 0) d.BLOCK d.N

/-- The host's launch (source line 141): a 1-D grid of `num_reqs` programs; the source's one
block shape `[BLOCK_SIZE]`; `enable_fp_fusion` is Triton's default (`true`); Triton's default
`num_warps = 4` and `num_stages = 3` change no result; `num_ctas = 1`, no `launch_pdl`,
`debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 d.B [[d.BLOCK]] (fusion := true)

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present),
every request's entries are in range (`ReqOk`), `cu_num_logits` starts nonnegative and ends
within `logits_indices`, and `query_start_loc` starts nonnegative and ends within `input_ids`.
Token values, `prefill_len` entries and the initial contents of the outputs are arbitrary. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.input_ids.Valid init ∧ d.idx_mapping.Valid init ∧ d.last_sampled.Valid init ∧
    d.query_start_loc.Valid init ∧ d.seq_lens.Valid init ∧ d.prefill_len.Valid init ∧
    d.draft_tokens.Valid init ∧ d.cu_num_logits.Valid init ∧ d.logits_indices.Valid init ∧
    (∀ b, b < d.B → ReqOk d init b) ∧
    0 ≤ cuAt d init 0 ∧ cuAt d init d.B ≤ d.logits_indices.len ∧
    0 ≤ qsl d init 0 ∧ qsl d init d.B ≤ d.input_ids.len

/-! ## The specification -/

/-- The elements of `logits_indices` the launch writes: request `b`'s
`cu_num_logits[b] + j`, `j < nlog`. -/
def logOut (d : Inst) (init : Mem) : Set Nat :=
  {t | ∃ b, b < d.B ∧ ∃ j : Nat, (j : Int) < nlog d init b ∧ (t : Int) = cuAt d init b + j}

/-- The elements of `input_ids` the launch writes: for each request past its prefill, the last
sampled token's slot `qend - nlog` (when `LastOn`) and the draft slots `qend - ndraft + j`,
`j < ndraft`. -/
def idsOut (d : Inst) (init : Mem) : Set Nat :=
  {t | ∃ b, b < d.B ∧
    ((LastOn d init b ∧ (t : Int) = qend d init b - nlog d init b) ∨
      (Decode d init b ∧ ∃ j : Nat, (j : Int) < ndraft d init b ∧
        (t : Int) = qend d init b - ndraft d init b + j))}

/-- What the launch leaves: for every request `b`,
- `logits_indices[cu_num_logits[b] + j]` holds `qend - nlog + j` for `j < nlog`;
- if `LastOn`, `input_ids[qend - nlog]` holds the `int32` truncation of
  `last_sampled_tokens[idx_mapping[b]]`;
- if past its prefill, `input_ids[qend - ndraft + j]` holds the `int32` truncation of
  `draft_tokens[idx_mapping[b] * dstride + j]` for `j < ndraft`;

and every byte outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b, b < d.B → ∀ j : Nat, (j : Int) < nlog d init b → ∃ w : I64,
      fin.read .i64 (d.logits_indices.ptr (cuAt d init b + j)).addr = some w ∧
      w.toInt = qend d init b - nlog d init b + j) ∧
    (∀ b, b < d.B → LastOn d init b → ∃ w : I32,
      fin.read .i32 (d.input_ids.ptr (qend d init b - nlog d init b)).addr = some w ∧
      w.toInt = rdt64 d.last_sampled init (ridx d init b)) ∧
    (∀ b, b < d.B → Decode d init b → ∀ j : Nat, (j : Int) < ndraft d init b → ∃ w : I32,
      fin.read .i32 (d.input_ids.ptr (qend d init b - ndraft d init b + j)).addr = some w ∧
      w.toInt = rdt64 d.draft_tokens init (ridx d init b * d.dstride + j)) ∧
    ∀ z, z ∉ d.logits_indices.elems (logOut d init) → z ∉ d.input_ids.elems (idsOut d init) →
      fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `_combine_sampled_and_draft_tokens_kernel` ends
normally with `Post`. -/
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

/-- An instance of `B` requests over `R` request-state rows, a draft table of width `S`
(`stride(0)` of a contiguous `[R, S]` tensor: `S`, or `1` when `S = 0`), `N` new sampled tokens,
`BLOCK_SIZE = BLOCK`, an `input_ids` buffer of `T` slots and `L` logits; the buffers one after
another from `2^30` with `2^20`-byte gaps, `input_ids` first and `logits_indices` last. Each takes
8 bytes per element, so every base is a multiple of 8. -/
def mk (B R S N BLOCK T L : Nat) : Inst :=
  let a0 : Int := 2 ^ 30
  let a1 : Int := a0 + T * 8 + 2 ^ 20
  let a2 : Int := a1 + B * 8 + 2 ^ 20
  let a3 : Int := a2 + R * 8 + 2 ^ 20
  let a4 : Int := a3 + (B + 1) * 8 + 2 ^ 20
  let a5 : Int := a4 + B * 8 + 2 ^ 20
  let a6 : Int := a5 + R * 8 + 2 ^ 20
  let a7 : Int := a6 + R * S * 8 + 2 ^ 20
  let a8 : Int := a7 + (B + 1) * 8 + 2 ^ 20
  { B := B, BLOCK := BLOCK, N := N, dstride := if S = 0 then 1 else S,
    input_ids := ⟨a0, T⟩, idx_mapping := ⟨a1, B⟩, last_sampled := ⟨a2, R⟩,
    query_start_loc := ⟨a3, B + 1⟩, seq_lens := ⟨a4, B⟩, prefill_len := ⟨a5, R⟩,
    draft_tokens := ⟨a6, R * S⟩, cu_num_logits := ⟨a7, B + 1⟩, logits_indices := ⟨a8, L⟩ }

theorem mk_ok (B R S N e T L : Nat) (h1 : B < 2 ^ 31) (h2 : N ≤ 1) (h3 : e ≤ 20)
    (h4 : S < 2 ^ 31) : (mk B R S N (2 ^ e) T L).ok := by
  refine ⟨h1, h2, ⟨e, h3, rfl⟩, ?_, le_refl _, le_refl _, le_refl _, le_refl _,
    ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩⟩
  · simp only [mk]; split_ifs <;> omega
  all_goals first
    | (refine bytes_disjoint_of_le ?_; simp only [mk, DType.bytes]; push_cast; nlinarith)
    | (refine (bytes_disjoint_of_le ?_).symm; simp only [mk, DType.bytes]; push_cast; nlinarith)

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

/-- A covered instance and memory: one request with no logit (`N = 0`, no drafts), every buffer
`0`. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 1 1 0 0 1 1 1, zeroMem,
    mk_ok 1 1 0 0 0 1 1 (by norm_num) (by norm_num) (by norm_num) (by norm_num), ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals try (apply valid_zeroMem <;> norm_num [mk, DType.bytes])
  · intro b hb
    simp only [mk] at hb
    have : b = 0 := by omega
    subst this
    simp only [ReqOk, ridx, seqAt, nlog, cuAt, qsl, qend, ndraft, rd32_zeroMem, rd64_zeroMem, mk]
    norm_num
  all_goals simp [cuAt, qsl, rd32_zeroMem, mk]

/-! ## The setup's launches

The host's launches (`SETUP.md`): `num_reqs = B` requests over `R = max_num_reqs` request-state
rows, `num_speculative_steps = S`, `num_new_sampled_tokens = N`, `BLOCK_SIZE =
next_power_of_2(S + N) = 2^e`, an `input_ids` buffer of `max_num_tokens = T` slots and
`total_num_logits = L` logits (at most `B · (S + N)`). -/

/-- Decoding without speculative decoding (the common case): `S = 0`, `N = 1`, `BLOCK_SIZE = 1`,
one logit per request; 256 requests. -/
def setup_decode_b256 : Inst := mk 256 1024 0 1 1 8192 256
theorem setup_decode_b256_ok : setup_decode_b256.ok :=
  mk_ok 256 1024 0 1 0 8192 256 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- The same at `max_num_seqs = 1024` requests and a 16384-token budget. -/
def setup_decode_b1024 : Inst := mk 1024 1024 0 1 1 16384 1024
theorem setup_decode_b1024_ok : setup_decode_b1024.ok :=
  mk_ok 1024 1024 0 1 0 16384 1024 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- One request (a single prompt), `S = 0`. -/
def setup_single : Inst := mk 1 1024 0 1 1 8192 1
theorem setup_single_ok : setup_single.ok :=
  mk_ok 1 1024 0 1 0 8192 1 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- MTP with one speculative token: `S = 1`, `BLOCK_SIZE = 2`; 128 requests. -/
def setup_mtp1 : Inst := mk 128 1024 1 1 2 8192 256
theorem setup_mtp1_ok : setup_mtp1.ok :=
  mk_ok 128 1024 1 1 1 8192 256 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- EAGLE with three speculative tokens: `S = 3`, `BLOCK_SIZE = 4`; 64 requests. -/
def setup_eagle3 : Inst := mk 64 1024 3 1 4 8192 256
theorem setup_eagle3_ok : setup_eagle3.ok :=
  mk_ok 64 1024 3 1 2 8192 256 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- EAGLE-3 with seven speculative tokens: `S = 7`, `BLOCK_SIZE = 8`; 256 requests. -/
def setup_eagle7 : Inst := mk 256 1024 7 1 8 8192 2048
theorem setup_eagle7_ok : setup_eagle7.ok :=
  mk_ok 256 1024 7 1 3 8192 2048 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- n-gram with eight speculative tokens: `S = 8`, `BLOCK_SIZE = 16`; 512 requests. -/
def setup_ngram8 : Inst := mk 512 1024 8 1 16 8192 4608
theorem setup_ngram8_ok : setup_ngram8.ok :=
  mk_ok 512 1024 8 1 4 8192 4608 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- A long draft: `S = 128`, `BLOCK_SIZE = 256`; 32 requests. -/
def setup_spec128 : Inst := mk 32 256 128 1 256 8192 4128
theorem setup_spec128_ok : setup_spec128.ok :=
  mk_ok 32 256 128 1 8 8192 4128 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- No new sampled token (`num_new_sampled_tokens_per_step = 0`, every logit a draft): `S = 4`,
`N = 0`, `BLOCK_SIZE = 4`; 64 requests. -/
def setup_n0_s4 : Inst := mk 64 1024 4 0 4 8192 256
theorem setup_n0_s4_ok : setup_n0_s4.ok :=
  mk_ok 64 1024 4 0 2 8192 256 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- `N = 0` with `S = 7`, `BLOCK_SIZE = 8`; 128 requests. -/
def setup_n0_s7 : Inst := mk 128 1024 7 0 8 8192 896
theorem setup_n0_s7_ok : setup_n0_s7.ok :=
  mk_ok 128 1024 7 0 3 8192 896 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- A large deployment: 8192 requests over 8192 rows, `S = 3`, a `2^20`-token budget. -/
def setup_large : Inst := mk 8192 8192 3 1 4 (2 ^ 20) 32768
theorem setup_large_ok : setup_large.ok :=
  mk_ok 8192 8192 3 1 2 (2 ^ 20) 32768 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- An empty batch (`num_reqs = 0`): no program runs. -/
def setup_empty : Inst := mk 0 1024 3 1 4 8192 0
theorem setup_empty_ok : setup_empty.ok :=
  mk_ok 0 1024 3 1 2 8192 0 (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Claims.CombineDraftTokens
