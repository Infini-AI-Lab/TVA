import TVA

/-!
# `kv_indices_i32`: KV indices for FlashInfer, SGLang

The kernel is `create_flashinfer_kv_indices_triton` of `source/kv_indices.py` (lines 69-150; it
reads the module-level names `triton` and `tl`, lines 65-66). The host `create_kv_indices` (lines
153-186) launches it with one program per request on the `int32` table `req_to_token`
(`[num_slots, max_context_len]`, contiguous), the index buffers `req_pool_indices` and `seq_lens`
(`[bs]`, `int64` or `int32`), the `int32` offsets `kv_indptr` (`[bs + 1]`, the exclusive running
sums of the lengths) and the `int32` output `kv_indices`, with `ENTRY_PAGE_SIZE = 1`,
`TOKEN_BLOCK_PARALLEL = False`, `token_mapping = None`, and `kv_start_idx` either `None` or a
tensor (`[bs]`, `int64` or `int32`) of each request's first position. `SETUP.md` lists the
launches.

What is claimed: every covered launch ends normally; for every request `b` and position
`j < seq_lens[b]`, element `kv_indptr[b] + j` of `kv_indices` holds the `int32` entry
`req_to_token[req_pool_indices[b], s[b] + j]`, where `s[b]` is `kv_start_idx[b]` (`0` without
`kv_start_idx`); every byte outside those elements is unchanged.

The conditions: the index dtypes are `int64` or `int32`; `bs ≤ 8192`, `max_context_len ≤ 2^22`;
the buffers hold the batch's entries and are valid; a given `kv_start_idx` is not at address `0`;
the output shares no byte with the inputs; every request's length and start are nonnegative
context lengths (at most `2^22`) whose window lies in the table; and `kv_indptr` holds the
exclusive running sums of the lengths from a nonnegative start, within the output. Any `int32`
table entries are covered.

The specification is exact: no float operation, no device fact.
-/
namespace Claims.KvIndices

open TVA

/-! ## The model -/

/-- `create_flashinfer_kv_indices_triton` (source lines 69-150) on the covered launches: with
`ENTRY_PAGE_SIZE = 1`, `TOKEN_BLOCK_PARALLEL = False` and `token_mapping = None`, the
compile-time conditions of lines 104, 123, 131 and 148 pick the branches below (the model map
drops the others). `kv_start_idx` is `none` when the host passes `None` (line 117 is then a
compile-time `if` whose branch is not visited) and `some p` for a tensor (line 117 is then a
runtime `if` on `p.addr ≠ 0`). `req_to_token_ptr_stride` is a runtime `int32`
(an integer argument in `[0, 2^31)`; at `1` it is the compile-time constant `1`, whose only use,
`req_pool_index * req_to_token_ptr_stride`, builds `1` in `int64`, the same value, see `model`).
`BLOCK_SIZE: tl.constexpr = 512` (line 102) is the literal `512` below. The table, `kv_indptr` and
the output are `*i32`; the index buffers are `*dp`, `*dl`, `*ds` (`int64` or `int32`). -/
def create_flashinfer_kv_indices_triton (req_to_token_ptr : Ptr .i32) {dp dl ds : DType}
    (req_pool_indices_ptr : Ptr dp) (page_kernel_lens_ptr : Ptr dl)
    (kv_indptr : Ptr .i32) (kv_start_idx : Option (Ptr ds)) (kv_indices_ptr : Ptr .i32)
    (req_to_token_ptr_stride : I32) : Kernel where
  body ctx := do
    -- line 103: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- lines 108-109 (the `else` of the compile-time `if TOKEN_BLOCK_PARALLEL:`): names assigned
    -- Python numbers hold `int32` scalar tensors
    let blk : I32 := IntV.ofInt 0
    let num_blk : I32 := IntV.ofInt 1
    -- line 112: one `int64` (or `int32`) element, `.to(tl.int64)`
    let req_pool_index := (← tl.convert dp .i64
      (Tensor.scalar (← tl.load1 (req_pool_indices_ptr + pid)))).item
    -- line 113: one `int32` element
    let kv_indices_offset ← tl.load1 (kv_indptr + pid)
    -- lines 115-116: names assigned Python numbers hold `int32` scalar tensors
    let kv_start : I32 := IntV.ofInt 0
    let kv_end : I32 := IntV.ofInt 0
    -- lines 117-119: `if kv_start_idx:` — with `None`, a compile-time condition that is false
    -- (lines 118-119 not visited); with a tensor, a runtime `if` on the pointer converted to
    -- `int1` (`p.addr ≠ 0`); the branch assigns both names, each an `int32` scalar, as before it
    let (kv_start, kv_end) ← (match kv_start_idx with
      | none => pure (kv_start, kv_end)
      | some kv_start_idx =>
        if kv_start_idx.addr ≠ 0 then do
          -- line 118: one `int64` (or `int32`) element, `.to(tl.int32)` (`tl.convert`, which
          -- wraps)
          let kv_start := (← tl.convert ds .i32
            (Tensor.scalar (← tl.load1 (kv_start_idx + pid)))).item
          -- line 119
          let kv_end := kv_start
          pure (kv_start, kv_end)
        else pure (kv_start, kv_end))
    -- line 120: `kv_end += …`, the load `.to(tl.int32)` (`tl.convert`, which wraps)
    let kv_end := kv_end + (← tl.convert dl .i32
      (Tensor.scalar (← tl.load1 (page_kernel_lens_ptr + pid)))).item
    -- line 122: `tl.cdiv(kv_end - kv_start, BLOCK_SIZE)` is `(x + 511) // 512` in `int32`:
    -- the wrapping `+` of the constant `511` (Python's `512 - 1`), then the truncating `//`
    let num_loop := (← tl.idiv (Tensor.scalar
        ((kv_end - kv_start) + (IntV.ofNat 511 : I32)))
      (Tensor.scalar (IntV.ofNat 512 : I32))).item
    -- line 127: `for i in range(blk, num_loop, num_blk)`, three runtime `int32` bounds
    let _ ← tl.forRange blk num_loop num_blk () fun i _ => do
      -- line 129: `tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE`: the `int32`
      -- product, then the scalar promoted to `int64` for the sum with the `int64` block
      let iB := i * (IntV.ofNat 512 : I32)
      let i64 := (← tl.convert .i32 .i64 (Tensor.scalar iB)).item
      let a64 ← tl.convert .i32 .i64 (tl.arange 0 512)
      let offset : Tensor [512] I64 := a64 + i64
      -- line 130: the `int32` scalar `kv_end - kv_start` promoted to `int64` for the signed
      -- comparison
      let lim : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (kv_end - kv_start))).item
      let mask := offset.ltS lim
      -- lines 131-138 (`ENTRY_PAGE_SIZE == 1` is the compile-time `True`): the address
      -- `((req_to_token_ptr + req_pool_index * stride) + kv_start) + offset`, left to right; the
      -- `int32` stride promoted to `int64` for the product; a masked load without `other`
      let rsp : I64 := (← tl.convert .i32 .i64 (Tensor.scalar req_to_token_ptr_stride)).item
      let data : Tensor [512] I32 ← tl.load
        (req_to_token_ptr + req_pool_index * rsp + kv_start + offset) (some mask)
      -- lines 139-149: the other branch (`ENTRY_PAGE_SIZE == 1`) and `token_mapping is not
      -- None` (the compile-time `False`) are not visited
      -- line 150: `(kv_indices_ptr + kv_indices_offset) + offset`; `data` is `int32`, the
      -- pointer's dtype
      tl.store (kv_indices_ptr + kv_indices_offset + offset) data (some mask)

/-! ## The conditions -/

/-- What the host chooses: one launch through `create_kv_indices` (lines 153-186). The tensors are
given by their storage (`Buf`); the table is contiguous with row stride `max_context_len` (the
launch's `req_to_token_ptr_stride = req_to_token.shape[1]`, line 179). -/
structure Inst where
  /-- `req_to_token.shape[1]`: the row length, the launch's `req_to_token_ptr_stride`. -/
  max_context_len : Nat
  /-- The batch size: the grid's `x`. -/
  bs : Nat
  /-- The dtype of `req_pool_indices` (`int64`, or `int32`). -/
  dP : DType
  /-- The dtype of `seq_lens` (`int64`, or `int32`). -/
  dL : DType
  /-- `kv_start_idx` is a tensor (`true`) or `None` (`false`). -/
  hasStart : Bool
  /-- The dtype of `kv_start_idx` (`int64`, or `int32`; unused without it). -/
  dS : DType
  /-- The storage of `req_to_token` (`int32`). -/
  req_to_token : Buf .i32
  /-- The storage of `req_pool_indices`. -/
  req_pool_indices : Buf dP
  /-- The storage of `seq_lens` (the kernel's `page_kernel_lens`). -/
  seq_lens : Buf dL
  /-- The storage of `kv_indptr` (`int32`). -/
  kv_indptr : Buf .i32
  /-- The storage of `kv_start_idx` (when `hasStart`). -/
  kv_start_idx : Buf dS
  /-- The storage of the output `kv_indices` (`int32`). -/
  kv_indices : Buf .i32

/-- The signed value of an `int64` or `int32` element (the index dtypes covered; `0` for any
other dtype, which no covered buffer has). -/
def ivOf {dt : DType} (x : dt.Val) : Int :=
  match dt, x with
  | .i64, y => y.toInt
  | .i32, y => y.toInt
  | _, _ => 0

/-- The value element `i` of a buffer holds in `m` (`0` when absent). -/
def valOf {dt : DType} (m : Mem) (b : Buf dt) (i : Nat) : Int :=
  match m.read dt (b.ptr (i : Int)).addr with
  | some x => ivOf x
  | none => 0

/-- Request `b`'s row of the table, `req_pool_indices[b]`. -/
def rpi (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.req_pool_indices b

/-- Request `b`'s length, `seq_lens[b]`. -/
def slen (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.seq_lens b

/-- Request `b`'s output offset, `kv_indptr[b]`. -/
def ind (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.kv_indptr b

/-- Request `b`'s first position: `kv_start_idx[b]` with it, `0` without it. -/
def sidx (d : Inst) (m : Mem) (b : Nat) : Int :=
  if d.hasStart then valOf m d.kv_start_idx b else 0

/-- Request `b`'s length, start and row are in range: a nonnegative length whose `int32` ceiling
`(len + 511) // 512` does not wrap (a limit of the kernel, README, Exclusions; it also makes
the length's `.to(tl.int32)` exact), a start in `[0, 2^31)` (so its `.to(tl.int32)` is exact),
and the row's `len` entries from the start lie in the table's storage (the request's row is a row
of the table, and its window within what the table holds). -/
def ReqOk (d : Inst) (m : Mem) (b : Nat) : Prop :=
  0 ≤ slen d m b ∧ slen d m b + 511 < 2 ^ 31 ∧ 0 ≤ sidx d m b ∧ sidx d m b < 2 ^ 31 ∧
    ∀ j : Nat, (j : Int) < slen d m b →
      0 ≤ rpi d m b * (d.max_context_len : Int) + sidx d m b + (j : Int) ∧
      rpi d m b * (d.max_context_len : Int) + sidx d m b + (j : Int) < (d.req_to_token.len : Int)

/-- Request `b`'s length and start are context lengths (the request's tokens, and the prefix
before its window): each at most `2^22`. -/
def CtxOk (d : Inst) (m : Mem) (b : Nat) : Prop :=
  slen d m b ≤ 2 ^ 22 ∧ sidx d m b ≤ 2 ^ 22

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The index dtypes are `int64` (SGLang's serving buffers) or `int32` (the code base's
  test). -/
  dP_ok : d.dP = .i64 ∨ d.dP = .i32
  dL_ok : d.dL = .i64 ∨ d.dL = .i32
  /-- `kv_start_idx`'s dtype is `int64` (SGLang's `seq_lens - paged_kernel_lens`) or `int32`. -/
  dS_ok : d.dS = .i64 ∨ d.dS = .i32
  /-- The grid `(bs, 1, 1)` within CUDA's limits, fewer than `2^31` programs. -/
  grid_lt : d.bs < 2 ^ 31
  /-- The row length is an `int32` runtime argument. -/
  mcl : d.max_context_len < 2 ^ 31
  /-- `max_context_len` is a context length (a row holds one request's tokens): at most `2^22`. -/
  mcl_le : d.max_context_len ≤ 2 ^ 22
  /-- `bs`, the requests of the batch, is a batch size: at most `8192`. -/
  bs_le : d.bs ≤ 8192
  /-- `kv_indptr` has the `bs + 1` offsets, the index buffers the batch's entries. -/
  plen : d.bs + 1 ≤ d.kv_indptr.len
  rplen : d.bs ≤ d.req_pool_indices.len
  llen : d.bs ≤ d.seq_lens.len
  stlen : d.hasStart = true → d.bs ≤ d.kv_start_idx.len
  /-- With `kv_start_idx`, its address is not `0` (line 117 tests the pointer; the data pointer of
  a CUDA tensor of `bs ≥ 1` elements is never null). -/
  start_nz : d.hasStart = true → d.kv_start_idx.base ≠ 0
  /-- The output shares no byte with the inputs (a fresh `torch.empty`). -/
  disj : Disjoint d.kv_indices.bytes d.req_to_token.bytes ∧
    Disjoint d.kv_indices.bytes d.req_pool_indices.bytes ∧
    Disjoint d.kv_indices.bytes d.seq_lens.bytes ∧
    Disjoint d.kv_indices.bytes d.kv_indptr.bytes ∧
    (d.hasStart = true → Disjoint d.kv_indices.bytes d.kv_start_idx.bytes)

/-- The model on an instance: the kernel's arguments are the buffers' first elements (and
`kv_start_idx` `none` without it) and the row length. At `max_context_len = 1` the stride argument is the compile-time constant `1`, which its
only use (an operand of `*` with the `int64` `req_pool_index`) builds in `int64`: the same value
as the model's `int32` `1` converted, so the model is the same. -/
def model (d : Inst) : Kernel :=
  create_flashinfer_kv_indices_triton (d.req_to_token.ptr 0) (d.req_pool_indices.ptr 0)
    (d.seq_lens.ptr 0) (d.kv_indptr.ptr 0)
    (if d.hasStart then some (d.kv_start_idx.ptr 0) else none) (d.kv_indices.ptr 0)
    (IntV.ofNat d.max_context_len)

/-- The host's launch (line 172): a 1-D grid of `bs` programs; the source's one block shape
`[512]`; `enable_fp_fusion` is Triton's default (`true`); `num_warps = 4`, `num_stages = 3` change
no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 d.bs [[512]] (fusion := true)

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte
present; `kv_start_idx` when given), every request's row, start and length are in range
(`ReqOk`) and are context lengths (`CtxOk`), and `kv_indptr` holds the
exclusive running sums of the lengths from a nonnegative start (the host's `0`), all within the
output's length. Any `int32` table entries. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.req_to_token.Valid init ∧ d.req_pool_indices.Valid init ∧ d.seq_lens.Valid init ∧
    d.kv_indptr.Valid init ∧ d.kv_indices.Valid init ∧
    (d.hasStart = true → d.kv_start_idx.Valid init) ∧
    (∀ b, b < d.bs → ReqOk d init b) ∧
    (∀ b, b < d.bs → CtxOk d init b) ∧
    (∀ b, b < d.bs → ind d init (b + 1) = ind d init b + slen d init b) ∧
    ∀ b, b ≤ d.bs → 0 ≤ ind d init b ∧ ind d init b ≤ (d.kv_indices.len : Int)

/-! ## The specification -/

/-- The output elements the launch writes: request `b`'s positions
`kv_indptr[b] … kv_indptr[b] + seq_lens[b] - 1`, for every `b` of the batch. -/
def kvOut (d : Inst) (init : Mem) : Set Nat :=
  {j : Nat | ∃ b : Nat, b < d.bs ∧ ∃ p : Nat, (p : Int) < slen d init b ∧
    (j : Int) = ind d init b + p}

/-- What the launch leaves: for every request `b` and position `j < seq_lens[b]`, element
`kv_indptr[b] + j` of `kv_indices` holds the initial `int32` value of
`req_to_token[req_pool_indices[b], s[b] + j]` (element
`req_pool_indices[b] * max_context_len + s[b] + j` of the contiguous table, `s[b]` the request's
first position, `sidx`); every byte outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b : Nat, b < d.bs → ∀ j : Nat, (j : Int) < slen d init b → ∃ v : I32,
      init.read .i32
        (d.req_to_token.ptr
          (rpi d init b * (d.max_context_len : Int) + sidx d init b + (j : Int))).addr = some v ∧
      fin.read .i32 (d.kv_indices.ptr (ind d init b + (j : Int))).addr = some v) ∧
    ∀ z, z ∉ d.kv_indices.elems (kvOut d init) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `create_flashinfer_kv_indices_triton` ends normally with
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

/-- An instance with the host's layout: `slots × ctx` table, `bs` requests with index dtypes `dp`,
`dl`, `kv_start_idx` of dtype `ds` when `hs`, an output of `total` elements; the storages one after
another from `2^30` with `2^20`-byte gaps, the output last. -/
def mk (slots ctx bs : Nat) (dp dl ds : DType) (hs : Bool) (total : Nat) : Inst :=
  let t0 : Int := 2 ^ 30
  let p0 : Int := t0 + slots * ctx * 4 + 2 ^ 20
  let l0 : Int := p0 + bs * 8 + 2 ^ 20
  let i0 : Int := l0 + bs * 8 + 2 ^ 20
  let s0 : Int := i0 + (bs + 1) * 4 + 2 ^ 20
  let o0 : Int := s0 + bs * 8 + 2 ^ 20
  { max_context_len := ctx, bs := bs, dP := dp, dL := dl, hasStart := hs, dS := ds,
    req_to_token := ⟨t0, slots * ctx⟩, req_pool_indices := ⟨p0, bs⟩, seq_lens := ⟨l0, bs⟩,
    kv_indptr := ⟨i0, bs + 1⟩, kv_start_idx := ⟨s0, bs⟩, kv_indices := ⟨o0, total⟩ }

theorem mk_ok (slots ctx bs : Nat) (dp dl ds : DType) (hs : Bool) (total : Nat)
    (hp : dp = .i64 ∨ dp = .i32) (hl : dl = .i64 ∨ dl = .i32) (hd : ds = .i64 ∨ ds = .i32)
    (h1 : bs < 2 ^ 31) (h2 : ctx < 2 ^ 31) (h3 : ctx ≤ 2 ^ 22) (h4 : bs ≤ 8192) :
    (mk slots ctx bs dp dl ds hs total).ok := by
  refine ⟨hp, hl, hd, h1, h2, h3, h4, le_refl _, le_refl _, le_refl _, fun _ => le_refl _, fun _ => ?_,
    ?_, ?_, ?_, ?_, fun _ => ?_⟩
  · simp only [mk]; positivity
  · refine (bytes_disjoint_of_le ?_).symm
    simp only [mk, DType.bytes]; push_cast; nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    rcases hp with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    rcases hl with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    simp only [mk, DType.bytes]; push_cast; nlinarith
  · refine (bytes_disjoint_of_le ?_).symm
    rcases hd with rfl | rfl <;> simp only [mk, DType.bytes] <;> push_cast <;> nlinarith

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

/-- Every element of a buffer of an index dtype reads `0` in the all-zero memory. -/
theorem valOf_zeroMem {dt : DType} (hdt : dt = .i64 ∨ dt = .i32) (b : Buf dt) (i : Nat) :
    valOf zeroMem b i = 0 := by
  unfold valOf
  by_cases h : Mem.Aligned dt (b.ptr (i : Int)).addr
  · rw [zeroMem_read h]
    rcases hdt with rfl | rfl <;> decide
  · simp only [Mem.read, h, ↓reduceIte]

/-- A covered instance and memory: the code base's pool of 4096 × 4096 with 37 requests
(`int64` indices and `kv_start_idx`), every buffer `0` (every length and start `0`). -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 4096 4096 37 .i64 .i64 .i64 true 0, zeroMem,
    mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num), ?_⟩
  have hs : ∀ b, slen (mk 4096 4096 37 .i64 .i64 .i64 true 0) zeroMem b = 0 :=
    fun b => valOf_zeroMem (Or.inl rfl) _ _
  have hi : ∀ b, ind (mk 4096 4096 37 .i64 .i64 .i64 true 0) zeroMem b = 0 :=
    fun b => valOf_zeroMem (Or.inr rfl) _ _
  have hst : ∀ b, sidx (mk 4096 4096 37 .i64 .i64 .i64 true 0) zeroMem b = 0 :=
    fun b => by simp only [sidx, mk, ↓reduceIte]; exact valOf_zeroMem (Or.inl rfl) _ _
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, fun _ => ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ?_, ?_, ?_, ?_⟩
  rotate_left 18
  · intro b _
    simp only [ReqOk, hs, hst]
    refine ⟨le_refl _, by norm_num, le_refl _, by norm_num, fun j hj => ?_⟩
    omega
  · intro b _
    simp only [CtxOk, hs, hst]
    norm_num
  · intro b _
    simp only [hi, hs]
    norm_num
  · intro b _
    simp only [hi]
    simp
  all_goals (simp only [mk, DType.bytes]; norm_num)

/-! ## The setup's launches

The host's test cases (source lines 203-235): the table `[num_slots, max_context_len]`, the batch,
the index dtype, `kv_start_idx` or not, and the output's length (the sum of the lengths; for the
random lengths, an upper bound: `Inst.ok` does not depend on it). -/

/-- Pool 4096 × 4096, 1 request, `int32` indices. -/
def setup_pool4096_bs1_i32 : Inst := mk 4096 4096 1 .i32 .i32 .i64 false 4096
theorem setup_pool4096_bs1_i32_ok : setup_pool4096_bs1_i32.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inr rfl) (Or.inr rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Pool 4096 × 4096, 37 requests, `int32` indices. -/
def setup_pool4096_bs37_i32 : Inst := mk 4096 4096 37 .i32 .i32 .i64 false (37 * 4096)
theorem setup_pool4096_bs37_i32_ok : setup_pool4096_bs37_i32.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inr rfl) (Or.inr rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Pool 4096 × 4096, 1786 requests, `int64` indices. -/
def setup_pool4096_bs1786_i64 : Inst := mk 4096 4096 1786 .i64 .i64 .i64 false (1786 * 4096)
theorem setup_pool4096_bs1786_i64_ok : setup_pool4096_bs1786_i64.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Pool 513 × 8192, lengths 1, 511, 512, 513, 1025, 8192 (`int64`). -/
def setup_pool513_lens : Inst := mk 513 8192 6 .i64 .i64 .i64 false 10754
theorem setup_pool513_lens_ok : setup_pool513_lens.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Pool 9 × 131072, lengths 131072, 1, 70000, 4097, 0, 33, 99999, 512 (`int64`). -/
def setup_pool9_long : Inst := mk 9 131072 8 .i64 .i64 .i64 false 305714
theorem setup_pool9_long_ok : setup_pool9_long.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Pool 2049 × 32768, 256 requests (`int64`). -/
def setup_pool2049_bs256 : Inst := mk 2049 32768 256 .i64 .i64 .i64 false (256 * 32768)
theorem setup_pool2049_bs256_ok : setup_pool2049_bs256.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Pool 300 × 4096, 100 requests (`int64`), with `kv_start_idx`: the last `window` positions of
each request. -/
def setup_pool300_window : Inst := mk 300 4096 100 .i64 .i64 .i64 true (100 * 1023)
theorem setup_pool300_window_ok : setup_pool300_window.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Mixed index dtypes (`int64` rows, `int32` lengths), which the host also accepts. -/
def setup_mixed_dtypes : Inst := mk 64 1024 16 .i64 .i32 .i64 false (16 * 1024)
theorem setup_mixed_dtypes_ok : setup_mixed_dtypes.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inr rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- Serving: a 1M-token context row, 4 requests. -/
def setup_context_1m : Inst := mk 4 (2 ^ 20) 4 .i64 .i64 .i64 false (4 * 2 ^ 20)
theorem setup_context_1m_ok : setup_context_1m.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- FlashInfer's sliding-window decode: 64 requests of contexts up to 131072 tokens, each copying
its last `window + 1 = 4097` positions (`kv_start_idx = seq_lens - min(seq_lens, 4097)`,
`int64`). -/
def setup_swa_decode : Inst := mk 64 131072 64 .i64 .i64 .i64 true (64 * 4097)
theorem setup_swa_decode_ok : setup_swa_decode.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inl rfl) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
/-- `int32` lengths and `kv_start_idx` (FlashInfer's planners pass `seq_lens - paged_kernel_lens`
in the lengths' dtype), `int64` rows. -/
def setup_start_i32 : Inst := mk 128 8192 32 .i64 .i32 .i32 true (32 * 8192)
theorem setup_start_i32_ok : setup_start_i32.ok :=
  mk_ok _ _ _ _ _ _ _ _ (Or.inl rfl) (Or.inr rfl) (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end Claims.KvIndices
