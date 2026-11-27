import TVA

/-!
# `ep_src2dst_i32`: expert-parallel MoE `src2dst` map after sorting, SGLang

The kernel is `compute_src2dst_triton_kernel` of `source/ep_src2dst.py` (lines 37–59; it reads
the module-level names `triton` and `tl`, lines 33–34). The host `cutlass_w4_run_moe_ep_preproess`
(lines 62–74) sorts the flat `int32` expert ids `topk_ids` stably (`torch.sort`: `sorted_topk_ids`
ascending, `int32`, and the permutation `reorder_ids`, `int64`), allocates the `int32` output
`src2dst` of `topk_ids.numel()` elements and launches the kernel on a 1-D grid of
`cdiv(numel, 512)` programs with `num_toks = numel` and `BLOCK_SIZE = 512`; it launches nothing
when `numel = 0`. `SETUP.md` lists the launches.

Every program first finds `first_nonnegative`, the first sorted position whose id is not negative
(`0` when `sorted_topk_ids[0] ≥ 0`, else a binary search); program `p` then handles the sorted
positions `i = 512·p + l < num_toks`: it reads `reorder_ids[i]` and stores
`i - first_nonnegative` (or `-1` when `i < first_nonnegative`) at `src2dst[reorder_ids[i]]`.

What is claimed: every covered launch ends normally; for every sorted position `i < n`,
element `reorder_ids[i]` of `src2dst` holds `-1` when `sorted_topk_ids[i]` is negative and
`i - c` otherwise, where `c` is the number of negative entries among the first `n` sorted ids;
every byte outside the elements `reorder_ids[i]`, `i < n`, is unchanged.

The conditions: `n < 2^31` (`num_toks` is an `int32`); the buffers hold the `n` entries the
launch reads and are valid; the output shares no byte with an input; the negative entries of
`sorted_topk_ids` come first (true of the host's ascending sort); the entries of `reorder_ids` are
distinct elements of `src2dst` (true of the host's permutation of `[0, n)`). Any `int32` ids are
covered.

The specification is exact: no float operation, no device fact.
-/
namespace Claims.EpSrc2Dst

open TVA

/-! ## The model -/

/-- One trip of the binary search (source lines 45–50) from the state `(left, right)`: the
`int32` `middle = left + (right - left) // 2` (wrapping `-` and `+`, the truncating `//` of the
Python `2` built in `int32`), the unmasked scalar load of `sorted_topk_ids + middle`, and the
runtime `if expert_id < 0` (signed, the Python `0` an `int32` tensor), which assigns `left` or
`right`; `middle` and `expert_id` are new to the body. -/
def searchStep (sorted_topk_ids : Ptr .i32) (st : I32 × I32) : Prog (I32 × I32) := do
  let (left, right) := st
  -- line 45
  let middle := left + (← tl.idiv (Tensor.scalar (right - left))
    (Tensor.scalar (IntV.ofInt 2 : I32))).item
  -- line 46
  let expert_id ← tl.load1 (sorted_topk_ids + middle)
  -- lines 47–50
  if IntV.lt expert_id (IntV.ofInt 0 : I32) then
    -- line 48: the wrapping `int32` `+` of the Python `1`
    let left := middle + (IntV.ofInt 1 : I32)
    pure (left, right)
  else
    -- line 50
    let right := middle
    pure (left, right)

/-- `compute_src2dst_triton_kernel` (source lines 37–59). `sorted_topk_ids` and `src2dst` are
`*i32`, `reorder_ids` is `*i64` (`torch.sort`'s indices); `num_toks` is the `int32` runtime
argument (an integer argument in `[2, 2^31)`, or `0`; at `1` it is the compile-time constant `1`,
which gives the same model, see `model`); `BLOCK_SIZE : tl.constexpr` is a Lean parameter. -/
def compute_src2dst_triton_kernel (sorted_topk_ids : Ptr .i32) (reorder_ids : Ptr .i64)
    (src2dst : Ptr .i32) (num_toks : I32) (BLOCK_SIZE : Nat) : Kernel where
  body ctx := do
    -- line 41: a name assigned a Python number holds an `int32` scalar tensor
    let first_nonnegative : I32 := IntV.ofInt 0
    -- line 42: the unmasked scalar load of element `0`, compared with the Python `0` (an `int32`
    -- tensor, signed `<`): a runtime `if`
    let s0 ← tl.load1 sorted_topk_ids
    let first_nonnegative ← (if IntV.lt s0 (IntV.ofInt 0 : I32) then do
        -- line 43: `left` holds the `int32` `0`, `right` the `int32` `num_toks`
        let left : I32 := IntV.ofInt 0
        let right : I32 := num_toks
        -- line 44: `while left < right:` (signed `int32` `<`); the loop carries `left` and
        -- `right`, the names defined before it that its body assigns
        let (left, _right) ← tl.whileLoop (left, right) (fun st => pure (IntV.lt st.1 st.2))
          (searchStep sorted_topk_ids)
        -- line 51
        let first_nonnegative := left
        pure first_nonnegative
      else pure first_nonnegative)
    -- line 53: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- line 54: the constexpr `BLOCK_SIZE` built in `int32`; wrapping `*` and `+`
    let dst_id := pid * (IntV.ofNat BLOCK_SIZE : I32) + tl.arange 0 BLOCK_SIZE
    -- line 55: the `int32` scalar `num_toks` compared with every lane, signed
    let mask := dst_id.ltS num_toks
    -- line 56: a masked load without `other` (the disabled lanes are unspecified), `int64`
    let src_id ← tl.load (reorder_ids + dst_id) (some mask)
    -- line 58: `dst_id >= first_nonnegative` (signed `int32`, the scalar broadcast) is
    -- `first_nonnegative <= dst_id`; `dst_id - first_nonnegative` wraps in `int32`; the Python
    -- `-1` is built in `int32`, the dtype `tl.where` computes in (`whereTyping?`)
    let packed_id := tl.where (Tensor.le (bc first_nonnegative) dst_id)
      (dst_id - bc first_nonnegative) (bc (IntV.ofInt (-1) : I32))
    -- line 59: the pointers `src2dst + src_id` (`int64` offsets); `packed_id` is already
    -- `int32`, the pointer's dtype
    tl.store (src2dst + src_id) packed_id (some mask)

/-! ## The conditions -/

/-- What the host chooses: one launch through `cutlass_w4_run_moe_ep_preproess` (lines 62–74).
The tensors are given by their storage (`Buf`). -/
structure Inst where
  /-- `num_toks`: `topk_ids.numel()`, the number of routing slots. -/
  n : Nat
  /-- The storage of `sorted_topk_ids` (`int32`, the sorted flat ids). -/
  sorted : Buf .i32
  /-- The storage of `reorder_ids` (`int64`, the sort's indices). -/
  reorder : Buf .i64
  /-- The storage of the output `src2dst` (`int32`). -/
  src2dst : Buf .i32

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `n < 2^31`: Triton passes `num_toks` as an `int32` (or, at `n = 1`, as a constant). The
  grid `cdiv(n, 512)` is then below `2^31` programs and the `int32` positions do not wrap. -/
  n_lt : d.n < 2 ^ 31
  /-- `sorted_topk_ids` holds the `n` sorted ids. -/
  sorted_len : d.n ≤ d.sorted.len
  /-- `reorder_ids` holds the `n` indices. -/
  reorder_len : d.n ≤ d.reorder.len
  /-- The output shares no byte with the sorted ids (a fresh `torch.empty`). -/
  out_sorted : Disjoint d.src2dst.bytes d.sorted.bytes
  /-- The output shares no byte with the indices. -/
  out_reorder : Disjoint d.src2dst.bytes d.reorder.bytes

/-- The signed value of element `i` of an `int32` buffer in `m` (`0` when absent). -/
def val32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- The signed value of element `i` of an `int64` buffer in `m` (`0` when absent). -/
def val64 (m : Mem) (b : Buf .i64) (i : Nat) : Int :=
  match m.read .i64 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- The sorted id at position `i`, `sorted_topk_ids[i]`. -/
def sid (d : Inst) (m : Mem) (i : Nat) : Int := val32 m d.sorted i

/-- The slot that lands at sorted position `i`, `reorder_ids[i]`. -/
def rid (d : Inst) (m : Mem) (i : Nat) : Int := val64 m d.reorder i

/-- The model on an instance: the kernel's arguments are the buffers' first elements, `n` as an
`int32`, and `BLOCK_SIZE = 512`. At `n = 1` Triton passes `num_toks` as the compile-time constant
`1`; its uses, `right = num_toks` (line 43: an assignment makes a Python number an `int32`
tensor) and `dst_id < num_toks` (line 55: a comparison makes it an `int32` tensor first), give
the `int32` `1`, so the model is the same. -/
def model (d : Inst) : Kernel :=
  compute_src2dst_triton_kernel (d.sorted.ptr 0) (d.reorder.ptr 0) (d.src2dst.ptr 0)
    (IntV.ofNat d.n) 512

/-- The host's launch (lines 68–72): a 1-D grid of `triton.cdiv(numel, 512)` programs; the
source's one block shape `[512]` (`tl.arange`, `dst_id`, `mask`, `src_id`, `packed_id`);
`enable_fp_fusion` is Triton's default (`true`; the kernel has no float operation); `num_warps =
4`, `num_stages = 3` change no result; `num_ctas = 1`, no `launch_pdl`, `debug = false`. -/
def launch (d : Inst) : Launch := Launch.grid1 ((d.n + 511) / 512) [[512]] (fusion := true)

/-- The inputs covered: the three buffers are valid (aligned, inside `[0, 2^64)`, every byte
present); the negative entries among the first `n` sorted ids come first (an ascending sort has
this); the first `n` entries of `reorder_ids` are distinct elements of `src2dst` (a permutation of
`[0, n)` with `src2dst` of `n` elements has this). Any `int32` id values. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.sorted.Valid init ∧ d.reorder.Valid init ∧ d.src2dst.Valid init ∧
    (∀ i j : Nat, i ≤ j → j < d.n → sid d init j < 0 → sid d init i < 0) ∧
    (∀ i : Nat, i < d.n → 0 ≤ rid d init i ∧ rid d init i < (d.src2dst.len : Int)) ∧
    (∀ i j : Nat, i < d.n → j < d.n → rid d init i = rid d init j → i = j)

/-! ## The specification -/

/-- The number of negative ids among the first `n` sorted ids: the slots routed to no local
expert. -/
def nneg (d : Inst) (m : Mem) : Nat :=
  ((Finset.range d.n).filter (fun j => sid d m j < 0)).card

/-- The elements of `src2dst` the launch writes: `reorder_ids[i]` for `i < n`. -/
def dstSet (d : Inst) (m : Mem) : Set Nat :=
  {k : Nat | ∃ i : Nat, i < d.n ∧ rid d m i = (k : Int)}

/-- What the launch leaves: for every sorted position `i < n`, element `reorder_ids[i]` of
`src2dst` holds the `int32` `-1` when `sorted_topk_ids[i]` is negative and `i - nneg` otherwise
(the slot's row among the slots routed to a local expert, in sorted order); every byte outside
those elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i : Nat, i < d.n →
      fin.read .i32 (d.src2dst.ptr (rid d init i)).addr =
        some (IntV.ofInt (if sid d init i < 0 then -1 else (i : Int) - (nneg d init : Int)))) ∧
    ∀ z, z ∉ d.src2dst.elems (dstSet d init) → fin z = init z

/-! ## The claim -/

/-- **The claim**: every covered launch of `compute_src2dst_triton_kernel` ends normally with
`Post`. -/
def claim : TVA.Claim where
  Inst := Inst
  ok := Inst.ok
  kernel := model
  launch := launch
  Pre := Pre
  Post := Post

/-! ## Not vacuous -/

/-- The host's launch on `n` slots: `sorted_topk_ids`, `reorder_ids` and a fresh `src2dst` of `n`
elements each, at the 16-byte-aligned addresses `2^34`, `2·2^34` and `3·2^34`. -/
def mk (n : Nat) : Inst where
  n := n
  sorted := ⟨2 ^ 34, n⟩
  reorder := ⟨2 * 2 ^ 34, n⟩
  src2dst := ⟨3 * 2 ^ 34, n⟩

/-- Two buffers whose byte ranges are ordered do not overlap. -/
theorem bytes_disj {d d' : DType} (b : Buf d) (c : Buf d')
    (h : b.base + b.len * d.bytes ≤ c.base ∨ c.base + c.len * d'.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem mk_ok {n : Nat} (hn : n < 2 ^ 31) : (mk n).ok where
  n_lt := hn
  sorted_len := le_refl _
  reorder_len := le_refl _
  out_sorted := bytes_disj _ _ (Or.inr (by simp only [mk, DType.bytes]; push_cast; omega))
  out_reorder := bytes_disj _ _ (Or.inr (by simp only [mk, DType.bytes]; push_cast; omega))

/-- The memory every byte of which is present and `0`. -/
def zeroMem : Mem := fun _ => some 0

/-- A covered instance and memory: one slot, its id `0` and its index `0`. -/
theorem claim_satisfiable : claim.Satisfiable := by
  refine ⟨mk 1, zeroMem, mk_ok (by norm_num), ?_⟩
  have hb4 : (DType.i32.bytes : Int) = 4 := rfl
  have hb8 : (DType.i64.bytes : Int) = 8 := rfl
  have hr : rid (mk 1) zeroMem 0 = 0 := by
    simp only [rid, val64, mk, Mem.read, Mem.Aligned, Buf.ptr, hb8, Nat.cast_zero]
    norm_num
    rfl
  refine ⟨⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩,
    ⟨?_, ?_, ?_, fun _ _ => by simp [zeroMem]⟩, ?_, ?_, ?_⟩
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · simp [mk]
  · simp only [mk, hb8]; norm_num
  · simp only [mk, hb8]; decide
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro i j hij hj _
    have : j = 0 := by simp only [mk] at hj; omega
    subst this
    have : i = 0 := by omega
    subst this
    assumption
  · intro i hi
    have : i = 0 := by simp only [mk] at hi; omega
    subst this
    rw [hr]
    simp [mk]
  · intro i j hi hj _
    simp only [mk] at hi hj
    omega

/-! ## The setup's launches

The host's test cases (source lines 93–100) and serving shapes, `n = num_tokens · topk` slots
(`n = 0` launches nothing and needs no instance). The contents (the ids, the permutation) are
`Pre`'s. -/

/-- The test: 1 token, top-1 (`num_toks = 1`, the compile-time constant). -/
def setup_test_1x1 : Inst := mk 1
theorem setup_test_1x1_ok : setup_test_1x1.ok := mk_ok (by norm_num)
/-- The test: 7 tokens, top-8. -/
def setup_test_7x8 : Inst := mk 56
theorem setup_test_7x8_ok : setup_test_7x8.ok := mk_ok (by norm_num)
/-- The test: 512 tokens, top-8. -/
def setup_test_512x8 : Inst := mk 4096
theorem setup_test_512x8_ok : setup_test_512x8.ok := mk_ok (by norm_num)
/-- The test: 1000 tokens, top-8. -/
def setup_test_1000x8 : Inst := mk 8000
theorem setup_test_1000x8_ok : setup_test_1000x8.ok := mk_ok (by norm_num)
/-- The test: 4096 tokens, top-6. -/
def setup_test_4096x6 : Inst := mk 24576
theorem setup_test_4096x6_ok : setup_test_4096x6.ok := mk_ok (by norm_num)
/-- The test: 300 tokens, top-4, every slot routed to no local expert. -/
def setup_test_300x4 : Inst := mk 1200
theorem setup_test_300x4_ok : setup_test_300x4.ok := mk_ok (by norm_num)
/-- Decode: one token, top-8 (DeepSeek-V3's routing). -/
def setup_decode_1x8 : Inst := mk 8
theorem setup_decode_1x8_ok : setup_decode_1x8.ok := mk_ok (by norm_num)
/-- A prefill chunk: 16384 tokens, top-8. -/
def setup_prefill_16384x8 : Inst := mk 131072
theorem setup_prefill_16384x8_ok : setup_prefill_16384x8.ok := mk_ok (by norm_num)
/-- The largest real batch: `2^22` tokens, top-8 (`2^25` slots). -/
def setup_tokens_4m_x8 : Inst := mk (2 ^ 25)
theorem setup_tokens_4m_x8_ok : setup_tokens_4m_x8.ok := mk_ok (by norm_num)
/-- The kernel's limit: `2^31 - 1` slots. -/
def setup_limit : Inst := mk (2 ^ 31 - 1)
theorem setup_limit_ok : setup_limit.ok := mk_ok (by norm_num)

end Claims.EpSrc2Dst
