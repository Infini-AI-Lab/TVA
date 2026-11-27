import TVA

/-!
# `write_req_to_token_pool_i32`: a batch's KV indices written into `req_to_token`, SGLang

The kernel is `write_req_to_token_pool_triton` of `source/write_req_to_token_pool.py` (lines
56-104; it reads the module-level names `triton` and `tl`, lines 52-53). The host
`write_cache_indices` (lines 107-136) launches it with one program per request of an extend batch
on the `int32` table `req_to_token` (`[rows, max_context_len]`, contiguous, its row length
`max_context_len` passed as the compile-time `req_to_token_ptr_stride`), the `int64` index buffers
`req_pool_indices`, `prefix_lens`, `seq_lens`, `extend_lens` (`[bs]`), the `uint64` data pointers
of the requests' prefix tensors (`int64`, request `b`'s holding its `prefix_lens[b]` cached
locations) and the `int64` new locations `out_cache_loc`. `SETUP.md` lists the launches.

Program `b` writes its row `req_pool_indices[b]`: positions `0 … prefix_lens[b] - 1` from its
prefix tensor, then positions `prefix_lens[b] … seq_lens[b] - 1` from `out_cache_loc`, from
element `c[b] = extend_lens[0] + … + extend_lens[b - 1]` on, each `int64` location stored as an
`int32` (its low 32 bits).

What is claimed: every covered launch ends normally; for every request `b`, element `j` of its
row holds the low 32 bits of `prefix_b[j]` for `j < prefix_lens[b]`, and of
`out_cache_loc[c[b] + j - prefix_lens[b]]` for `prefix_lens[b] ≤ j < seq_lens[b]`; every byte
outside those elements is unchanged.

The conditions: `bs ≤ 8192`, `max_context_len ≤ 2^22`; the buffers hold the batch's entries and
are valid; the table shares no byte with any input; every request has
`0 ≤ prefix_lens[b] ≤ seq_lens[b] ≤ max_context_len` and a row of the table, the requests that
write (`seq_lens[b] > 0`) distinct rows; a request with a prefix has its pointer at a valid
prefix tensor of at least `prefix_lens[b]` elements; every `extend_lens[b]` is in `[0, 2^22]` and
each request's window of `out_cache_loc` lies inside it. Any `int64` locations are covered.

The specification is exact: no float operation, no device fact.
-/
namespace Claims.WriteReqToTokenPool

open TVA

/-! ## The model -/

/-- `write_req_to_token_pool_triton` (source lines 56-104). The table is `*i32`, the index buffers,
`out_cache_loc` and the prefix tensors `*i64`, the prefix pointers `*u64`.
`req_to_token_ptr_stride` is a `tl.constexpr` (a Lean `Nat`), and `BLOCK_SIZE: tl.constexpr = 512`
(line 67) is the literal `512` below. -/
def write_req_to_token_pool_triton (req_to_token_ptr : Ptr .i32) (req_pool_indices : Ptr .i64)
    (prefix_tensors : Ptr .u64) (pre_lens seq_lens extend_lens out_cache_loc : Ptr .i64)
    (req_to_token_ptr_stride : Nat) : Kernel where
  body ctx := do
    -- line 68: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- lines 70-72: one `int64` element each
    let req_pool_index ← tl.load1 (req_pool_indices + pid)
    let pre_len ← tl.load1 (pre_lens + pid)
    let seq_len ← tl.load1 (seq_lens + pid)
    -- line 73: one `uint64` element, `.to(tl.pointer_type(tl.int64))` (`tl.intToPtr`)
    let prefix_tensor : Ptr .i64 := tl.intToPtr .i64 (← tl.load1 (prefix_tensors + pid))
    -- line 76: `tl.cdiv(pre_len, BLOCK_SIZE)` is `(pre_len + 511) // 512` in `int64`: the
    -- constexpr `BLOCK_SIZE - 1` built in `int64` for the wrapping `+`, then the truncating `//`
    let num_loop := (← tl.idiv (Tensor.scalar (pre_len + (IntV.ofNat 511 : I64)))
      (Tensor.scalar (IntV.ofNat 512 : I64))).item
    -- line 77: `for i in range(num_loop)`: the bounds `0`, `num_loop`, `1`, an `int64` induction
    -- variable (the promoted type of the bounds); nothing carried
    let _ ← tl.forRange (IntV.ofInt 0 : I64) num_loop (IntV.ofInt 1 : I64) () fun i _ => do
      -- line 78: the `int32` block `tl.arange(0, BLOCK_SIZE)` promoted to `int64` for the sum
      -- with the `int64` scalar `i * BLOCK_SIZE` (the constexpr built in `int64`)
      let a64 ← tl.convert .i32 .i64 (tl.arange 0 512)
      let offset : Tensor [512] I64 := a64 + i * (IntV.ofNat 512 : I64)
      -- line 79
      let mask := offset.ltS pre_len
      -- line 80: a masked load without `other`
      let value : Tensor [512] I64 ← tl.load (prefix_tensor + offset) (some mask)
      -- lines 81-85: the address `(req_to_token_ptr + req_pool_index * stride) + offset`, the
      -- constexpr stride built in `int64`; the `int64` value converted to the pointer's `int32`
      let v32 ← tl.convert .i64 .i32 value
      tl.store (req_to_token_ptr + req_pool_index * (IntV.ofNat req_to_token_ptr_stride : I64) +
        offset) v32 (some mask)
    -- line 88: `tl.cast(0, tl.int64)`: the Python `0` an `int32` scalar, converted to `int64`
    let cumsum_start : I64 :=
      (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofInt 0 : I32))).item
    -- lines 89-90: `for i in range(pid)`: `int32` bounds and induction variable; the loop
    -- carries `cumsum_start`, to which each trip adds one `int64` element (wrapping)
    let cumsum_start ← tl.forRange (IntV.ofInt 0 : I32) pid (IntV.ofInt 1 : I32) cumsum_start
      fun i cumsum_start => do
        let e ← tl.load1 (extend_lens + i)
        pure (cumsum_start + e)
    -- line 92: `tl.cdiv(seq_len - pre_len, BLOCK_SIZE)`, as line 76
    let num_loop := (← tl.idiv (Tensor.scalar ((seq_len - pre_len) + (IntV.ofNat 511 : I64)))
      (Tensor.scalar (IntV.ofNat 512 : I64))).item
    -- line 93: as line 77
    let _ ← tl.forRange (IntV.ofInt 0 : I64) num_loop (IntV.ofInt 1 : I64) () fun i _ => do
      -- line 94: as line 78
      let a64 ← tl.convert .i32 .i64 (tl.arange 0 512)
      let offset : Tensor [512] I64 := a64 + i * (IntV.ofNat 512 : I64)
      -- line 95
      let mask := offset.ltS (seq_len - pre_len)
      -- line 96: the address `(out_cache_loc + cumsum_start) + offset`; a masked load without
      -- `other`
      let value : Tensor [512] I64 ← tl.load (out_cache_loc + cumsum_start + offset) (some mask)
      -- lines 97-104: the address `((req_to_token_ptr + req_pool_index * stride) + offset) +
      -- pre_len`, the scalar `pre_len` broadcast to the block; the value converted to `int32`
      let v32 ← tl.convert .i64 .i32 value
      tl.store (req_to_token_ptr + req_pool_index * (IntV.ofNat req_to_token_ptr_stride : I64) +
        offset + (bc pre_len : Tensor [512] I64)) v32 (some mask)

/-! ## The conditions -/

/-- What the host chooses: one launch through `write_cache_indices` (lines 107-136). The tensors
are given by their storage (`Buf`); the table is contiguous with row stride `max_context_len` (the
launch's `req_to_token.shape[1]`, line 132). -/
structure Inst where
  /-- `req_to_token.shape[1]`: the row length, the compile-time `req_to_token_ptr_stride`. -/
  max_context_len : Nat
  /-- The batch size: the grid's `x`. -/
  bs : Nat
  /-- The storage of `req_to_token` (`int32`). -/
  req_to_token : Buf .i32
  /-- The storage of `req_pool_indices` (`int64`). -/
  req_pool_indices : Buf .i64
  /-- The storage of `prefix_pointers` (`uint64`), the prefix tensors' data pointers. -/
  prefix_pointers : Buf .u64
  /-- The storage of `prefix_lens` (the kernel's `pre_lens`, `int64`). -/
  prefix_lens : Buf .i64
  /-- The storage of `seq_lens` (`int64`). -/
  seq_lens : Buf .i64
  /-- The storage of `extend_lens` (`int64`). -/
  extend_lens : Buf .i64
  /-- The storage of `out_cache_loc` (`int64`). -/
  out_cache_loc : Buf .i64
  /-- The storage of request `b`'s prefix tensor (`int64`; any buffer for a request without a
  prefix, whose pointer the kernel never follows). -/
  prefixT : Nat → Buf .i64

/-- The signed value element `i` of an `int64` buffer holds in `m` (`0` when absent). -/
def valOf (m : Mem) (b : Buf .i64) (i : Nat) : Int :=
  match m.read .i64 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- The unsigned value element `i` of a `uint64` buffer holds in `m` (`0` when absent). -/
def uvalOf (m : Mem) (b : Buf .u64) (i : Nat) : Int :=
  match m.read .u64 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- Request `b`'s row of the table, `req_pool_indices[b]`. -/
def rpi (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.req_pool_indices b

/-- Request `b`'s prefix length, `prefix_lens[b]`. -/
def pre (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.prefix_lens b

/-- Request `b`'s length, `seq_lens[b]`. -/
def seq (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.seq_lens b

/-- Request `b`'s count of new locations, `extend_lens[b]`. -/
def elen (d : Inst) (m : Mem) (b : Nat) : Int := valOf m d.extend_lens b

/-- Request `b`'s first element of `out_cache_loc`: `extend_lens[0] + … + extend_lens[b - 1]`. -/
def cum (d : Inst) (m : Mem) (b : Nat) : Int := ∑ i ∈ Finset.range b, elen d m i

/-- Request `b`'s lengths and row are in range: `0 ≤ prefix_lens[b] ≤ seq_lens[b] ≤
max_context_len` (the row holds the request's tokens), and its row lies in the table's storage. -/
def ReqOk (d : Inst) (m : Mem) (b : Nat) : Prop :=
  0 ≤ pre d m b ∧ pre d m b ≤ seq d m b ∧ seq d m b ≤ (d.max_context_len : Int) ∧
    0 ≤ rpi d m b ∧
    rpi d m b * (d.max_context_len : Int) + seq d m b ≤ (d.req_to_token.len : Int)

/-- A request with a prefix (`prefix_lens[b] > 0`) has its pointer `prefix_pointers[b]` at its
prefix tensor, which is valid and holds at least `prefix_lens[b]` elements. -/
def PrefixOk (d : Inst) (m : Mem) (b : Nat) : Prop :=
  0 < pre d m b →
    uvalOf m d.prefix_pointers b = (d.prefixT b).base ∧ pre d m b ≤ ((d.prefixT b).len : Int) ∧
      (d.prefixT b).Valid m

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The grid `(bs, 1, 1)` within CUDA's limits, fewer than `2^31` programs. -/
  grid_lt : d.bs < 2 ^ 31
  /-- `bs`, the requests of the batch, is a batch size: at most `8192`. -/
  bs_le : d.bs ≤ 8192
  /-- `max_context_len` is a context length (a row holds one request's tokens): at most `2^22`. -/
  mcl_le : d.max_context_len ≤ 2 ^ 22
  /-- The index buffers and the prefix pointers hold the batch's entries. -/
  rplen : d.bs ≤ d.req_pool_indices.len
  pplen : d.bs ≤ d.prefix_pointers.len
  prelen : d.bs ≤ d.prefix_lens.len
  slen : d.bs ≤ d.seq_lens.len
  elen : d.bs ≤ d.extend_lens.len
  /-- The table shares no byte with the other buffers (separate allocations). -/
  disj : Disjoint d.req_to_token.bytes d.req_pool_indices.bytes ∧
    Disjoint d.req_to_token.bytes d.prefix_pointers.bytes ∧
    Disjoint d.req_to_token.bytes d.prefix_lens.bytes ∧
    Disjoint d.req_to_token.bytes d.seq_lens.bytes ∧
    Disjoint d.req_to_token.bytes d.extend_lens.bytes ∧
    Disjoint d.req_to_token.bytes d.out_cache_loc.bytes
  /-- Nor with any request's prefix tensor (an `int64` tensor of the prefix cache, or a copy). -/
  prefix_disj : ∀ b, b < d.bs → Disjoint d.req_to_token.bytes (d.prefixT b).bytes

/-- The model on an instance: the kernel's arguments are the buffers' first elements and the row
length. -/
def model (d : Inst) : Kernel :=
  write_req_to_token_pool_triton (d.req_to_token.ptr 0) (d.req_pool_indices.ptr 0)
    (d.prefix_pointers.ptr 0) (d.prefix_lens.ptr 0) (d.seq_lens.ptr 0) (d.extend_lens.ptr 0)
    (d.out_cache_loc.ptr 0) d.max_context_len

/-- The host's launch (lines 124-135): a 1-D grid of `bs` programs; the source's one block shape
`[512]`; `enable_fp_fusion` is Triton's default (`true`); `num_warps = 4`, `num_stages = 3` change
no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 d.bs [[512]] (fusion := true)

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present),
every request's lengths and row are in range (`ReqOk`) and its prefix pointer is right when it has
a prefix (`PrefixOk`), every `extend_lens[b]` is in `[0, 2^22]` (a request's new tokens), each
request's `seq_lens[b] - prefix_lens[b]` locations from `c[b]` lie in `out_cache_loc`, and the
requests that write have distinct rows. Any `int64` locations. -/
structure Pre (d : Inst) (init : Mem) : Prop where
  table_valid : d.req_to_token.Valid init
  rpi_valid : d.req_pool_indices.Valid init
  pp_valid : d.prefix_pointers.Valid init
  pre_valid : d.prefix_lens.Valid init
  seq_valid : d.seq_lens.Valid init
  ext_valid : d.extend_lens.Valid init
  out_valid : d.out_cache_loc.Valid init
  req : ∀ b, b < d.bs → ReqOk d init b
  prefixes : ∀ b, b < d.bs → PrefixOk d init b
  ext : ∀ b, b < d.bs → 0 ≤ elen d init b ∧ elen d init b ≤ 2 ^ 22
  window : ∀ b, b < d.bs →
    cum d init b + (seq d init b - pre d init b) ≤ (d.out_cache_loc.len : Int)
  rows : ∀ b b', b < d.bs → b' < d.bs → b ≠ b' → 0 < seq d init b → 0 < seq d init b' →
    rpi d init b ≠ rpi d init b'

/-! ## The specification -/

/-- The table elements the launch writes: positions `0 … seq_lens[b] - 1` of row
`req_pool_indices[b]`, for every request `b` of the batch. -/
def written (d : Inst) (init : Mem) : Set Nat :=
  {e : Nat | ∃ b : Nat, b < d.bs ∧ ∃ p : Nat, (p : Int) < seq d init b ∧
    (e : Int) = rpi d init b * (d.max_context_len : Int) + p}

/-- What the launch leaves: for every request `b`, element `j` of its row (element
`req_pool_indices[b] * max_context_len + j` of the contiguous table) holds the `int32` cast (the
low 32 bits) of the initial `int64` value of `prefix_b[j]` for `j < prefix_lens[b]`, and of
`out_cache_loc[c[b] + (j - prefix_lens[b])]` for `prefix_lens[b] ≤ j < seq_lens[b]`; every byte
outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b : Nat, b < d.bs → ∀ j : Nat, (j : Int) < pre d init b → ∃ v : I64,
      init.read .i64 ((d.prefixT b).ptr (j : Int)).addr = some v ∧
      fin.read .i32
        (d.req_to_token.ptr (rpi d init b * (d.max_context_len : Int) + (j : Int))).addr =
        some (IntV.cast v)) ∧
    (∀ b : Nat, b < d.bs → ∀ j : Nat, pre d init b ≤ (j : Int) → (j : Int) < seq d init b →
      ∃ v : I64,
        init.read .i64
          (d.out_cache_loc.ptr (cum d init b + ((j : Int) - pre d init b))).addr = some v ∧
        fin.read .i32
          (d.req_to_token.ptr (rpi d init b * (d.max_context_len : Int) + (j : Int))).addr =
          some (IntV.cast v)) ∧
    ∀ z, z ∉ d.req_to_token.elems (written d init) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `write_req_to_token_pool_triton` ends normally with
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

/-- An instance with the host's layout: a `rows × ctx` table, `bs` requests, an `out_cache_loc`
of `nOut` locations, and prefix tensors of `maxPre` elements each; the storages one after another
from `2^30` with `2^20`-byte gaps, the prefix tensors last. Each storage takes 8 bytes per element,
so every base is a multiple of 8. -/
def mk (rows ctx bs maxPre nOut : Nat) : Inst :=
  let t0 : Int := 2 ^ 30
  let r0 : Int := t0 + rows * ctx * 8 + 2 ^ 20
  let p0 : Int := r0 + bs * 8 + 2 ^ 20
  let l0 : Int := p0 + bs * 8 + 2 ^ 20
  let s0 : Int := l0 + bs * 8 + 2 ^ 20
  let e0 : Int := s0 + bs * 8 + 2 ^ 20
  let o0 : Int := e0 + bs * 8 + 2 ^ 20
  let q0 : Int := o0 + nOut * 8 + 2 ^ 20
  { max_context_len := ctx, bs := bs, req_to_token := ⟨t0, rows * ctx⟩,
    req_pool_indices := ⟨r0, bs⟩, prefix_pointers := ⟨p0, bs⟩, prefix_lens := ⟨l0, bs⟩,
    seq_lens := ⟨s0, bs⟩, extend_lens := ⟨e0, bs⟩, out_cache_loc := ⟨o0, nOut⟩,
    prefixT := fun b => ⟨q0 + b * maxPre * 8, maxPre⟩ }

theorem mk_ok (rows ctx bs maxPre nOut : Nat) (h1 : bs ≤ 8192) (h2 : ctx ≤ 2 ^ 22) :
    (mk rows ctx bs maxPre nOut).ok := by
  refine ⟨show bs < 2 ^ 31 by omega, h1, h2, le_refl _, le_refl _, le_refl _, le_refl _, le_refl _,
    ⟨?_, ?_, ?_, ?_, ?_, ?_⟩, fun b _ => ?_⟩
  all_goals (refine bytes_disjoint_of_le ?_; simp only [mk, DType.bytes]; push_cast; nlinarith)

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

/-- Every element of an `int64` buffer reads `0` in the all-zero memory. -/
theorem valOf_zeroMem (b : Buf .i64) (i : Nat) : valOf zeroMem b i = 0 := by
  unfold valOf
  by_cases h : Mem.Aligned .i64 (b.ptr (i : Int)).addr
  · rw [zeroMem_read h]
    decide
  · simp only [Mem.read, h, ↓reduceIte]

/-- Every running sum of `extend_lens` is `0` in the all-zero memory. -/
theorem cum_zeroMem (d : Inst) (b : Nat) : cum d zeroMem b = 0 := by
  simp only [cum, elen, valOf_zeroMem, Finset.sum_const_zero]

/-- A buffer is valid in the all-zero memory when it is aligned and inside `[0, 2^64)`. -/
theorem valid_zeroMem {dt : DType} (b : Buf dt) (h0 : 0 ≤ b.base)
    (h1 : b.base + b.len * dt.bytes ≤ 2 ^ 64) (h2 : b.base % dt.bytes = 0) : b.Valid zeroMem :=
  ⟨h0, h1, h2, fun _ _ => by simp [zeroMem]⟩

/-- A covered instance and memory: a pool of 4097 × 4100 with 37 requests, every buffer `0`
(every length `0`, every request on row `0`). -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 4097 4100 37 0 0, zeroMem, mk_ok _ _ _ _ _ (by norm_num) (by norm_num), ?_⟩
  have hz : ∀ (b : Buf .i64) i, valOf zeroMem b i = 0 := valOf_zeroMem
  refine { table_valid := valid_zeroMem _ ?_ ?_ ?_, rpi_valid := valid_zeroMem _ ?_ ?_ ?_,
           pp_valid := valid_zeroMem _ ?_ ?_ ?_, pre_valid := valid_zeroMem _ ?_ ?_ ?_,
           seq_valid := valid_zeroMem _ ?_ ?_ ?_, ext_valid := valid_zeroMem _ ?_ ?_ ?_,
           out_valid := valid_zeroMem _ ?_ ?_ ?_, req := ?_, prefixes := ?_, ext := ?_,
           window := ?_, rows := ?_ }
  rotate_left 21
  · intro b _
    simp only [ReqOk, rpi, pre, seq, hz]
    norm_num
  · intro b _ h
    simp only [pre, hz] at h
    exact absurd h (lt_irrefl _)
  · intro b _
    simp only [elen, hz]
    norm_num
  · intro b _
    rw [cum_zeroMem, seq, pre, hz, hz, sub_self, add_zero]
    exact Int.natCast_nonneg _
  · intro b b' _ _ _ h
    simp only [seq, hz] at h
    exact absurd h (lt_irrefl _)
  all_goals (simp only [mk, DType.bytes]; norm_num)

/-! ## The setup's launches

The host's test cases (source lines 158-184) and serving shapes: the table `[rows, ctx]`
(`rows` the pool's size plus its padding row, `ctx` the model's context length plus `4`), the
batch, the largest prefix, and the length of `out_cache_loc` (an upper bound: `Inst.ok` does not
depend on it). -/

/-- The host's test: pool 64, context 4096, 1 request without a prefix. -/
def setup_test_pool64_bs1 : Inst := mk 65 4100 1 0 4096
theorem setup_test_pool64_bs1_ok : setup_test_pool64_bs1.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- The host's test: pool 256, context 4096, 37 requests, prefixes up to 2048. -/
def setup_test_pool256_bs37 : Inst := mk 257 4100 37 2048 (37 * 2048)
theorem setup_test_pool256_bs37_ok : setup_test_pool256_bs37.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- The host's test: pool 4096, context 4096, 512 requests, prefixes up to 1024, extends up to 32. -/
def setup_test_pool4096_bs512 : Inst := mk 4097 4100 512 1024 (512 * 32)
theorem setup_test_pool4096_bs512_ok : setup_test_pool4096_bs512.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- The host's test: pool 64, context 32768, 8 requests, prefixes up to 20000. -/
def setup_test_ctx32k_bs8 : Inst := mk 65 32772 8 20000 (8 * 8192)
theorem setup_test_ctx32k_bs8_ok : setup_test_ctx32k_bs8.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- The host's test: pool 8, context 131072, 4 requests, prefixes up to 100000. -/
def setup_test_ctx128k_bs4 : Inst := mk 9 131076 4 100000 (4 * 16384)
theorem setup_test_ctx128k_bs4_ok : setup_test_ctx128k_bs4.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- Serving: a chunked prefill of 8192 tokens over 1024 requests of a 128K-context model, pool
2048. -/
def setup_serve_ctx128k_bs1024 : Inst := mk 2049 131076 1024 131072 8192
theorem setup_serve_ctx128k_bs1024_ok : setup_serve_ctx128k_bs1024.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- Serving: a 1M-token context (`2^20 + 4` per row), 8 requests with cached prefixes. -/
def setup_serve_ctx1m : Inst := mk 9 (2 ^ 20 + 4) 8 (2 ^ 20) 16384
theorem setup_serve_ctx1m_ok : setup_serve_ctx1m.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- Serving: 8192 requests in one batch (`bs_le`), pool 8192, context 4096. -/
def setup_serve_bs8192 : Inst := mk 8193 4100 8192 4096 (8192 * 4)
theorem setup_serve_bs8192_ok : setup_serve_bs8192.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)
/-- Speculative decoding: the row length grows by the draft tokens (`32768 + 4 + 8`). -/
def setup_spec_ctx32k : Inst := mk 257 32780 64 32768 (64 * 512)
theorem setup_spec_ctx32k_ok : setup_spec_ctx32k.ok := mk_ok _ _ _ _ _ (by norm_num) (by norm_num)

end Claims.WriteReqToTokenPool
