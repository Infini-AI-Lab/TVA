import TVA

/-!
# `count_expert_tokens_i32`: the prepared claim for vLLM's `_count_expert_num_tokens`

Source: `source/count_expert_tokens.py`. The kernel is lines
45–74 (the `@triton.jit` function `_count_expert_num_tokens`); it reads the module-level names
`triton` and `tl` (lines 41–42). Its host wrapper `count_expert_num_tokens` (lines 78–107)
allocates `expert_num_tokens` and launches the kernel on a 1-D grid of `num_local_experts`
programs with `BLOCK_SIZE = next_power_of_2(min(topk_ids.numel(), 1024))` and `HAS_EXPERT_MAP =
expert_map is not None`.

One program per local expert scans all of `topk_ids` in blocks of `BLOCK_SIZE`, counts the
entries whose (optionally mapped) id is the program's expert, and stores the count.

The claim (`Statement`): on every covered instance, every launch that runs ends normally;
element `e` of `expert_num_tokens` is the exact number of positions `j < topk_ids.numel()`
routed to local expert `e` — the entry itself without an expert map, `expert_map[entry]` for a
non-negative entry and `-1` for a negative one with a map — for every `e < num_local_experts`;
and every other byte is unchanged.

* The model: `loopBody`, `count_kernel`, `model`, `launch` (the model map is `model-map.json`).
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `routed`, `countOf`, `Post`, `Spec`.
* The assumptions: `Assumptions` (one `[rule]` field, `Compiler.Faithful`; the kernel is
  integer-only, so no device fact is needed).
* Proved here: `consistent`, `satisfiable` and the coverage of the host's test launches
  (`setup_*_ok`). `Statement` itself is the claim to prove.
-/

namespace Claims.CountExpertTokens

open TVA

/-! ## Helpers of the transcription

`topk_ids` may be `int32` or `int64` (the wrapper's assert requires a signed dtype). The model is
one function of the dtype, with the dtype-dependent steps written explicitly; `Inst.ok` restricts
the dtype to the two covered ones. -/

/-- The Python number `-1` as the `other=` of a load from a pointer of dtype `d`: first an `int32`
tensor of its own type, then converted to the pointer's dtype (`int32`: itself; `int64`: the
sign-extended `-1`). -/
def m1 (d : DType) : d.Val :=
  match d with
  | .i32 => IntV.ofInt (-1)
  | .i64 => IntV.ofInt (-1)
  | other => other.ofBits 0

/-- The integer a loaded entry of `topk_ids` holds (`int32`/`int64` two's complement). -/
def entryInt : (d : DType) → d.Val → Int
  | .i32, x => x.toInt
  | .i64, x => x.toInt
  | _, _ => 0

/-- `expert_map + expert_ids` (source line 65): the map's pointer advanced by each loaded id, in
elements (the id's own integer type, `int32` or `int64`). -/
def mapPtrs {B : Nat} (d : DType) (expert_map : Ptr .i32) (ids : Tensor [B] d.Val) :
    Tensor [B] (Ptr .i32) :=
  match d with
  | .i32 => expert_map + ids
  | .i64 => expert_map + ids
  | _ => Tensor.full expert_map

/-- `expert_ids >= 0` (source line 66): the Python number `0` is an `int32` tensor, promoted to
the ids' dtype (`BinOp.typing? .ge`); the signed `>=` is the negation of the signed `<`. -/
def geZero {B : Nat} (d : DType) (ids : Tensor [B] d.Val) : Tensor [B] Bool :=
  match d with
  | .i32 => Tensor.not (ids.ltS (0 : I32))
  | .i64 => Tensor.not (ids.lt (Tensor.full (0 : I64)))
  | _ => Tensor.full false

/-- `expert_ids == curr_expert` without an expert map (source line 69, `BinOp.typing? .eq`): the
`==` in the promoted dtype, the ids' own; the `int32` program id is sign-extended to `int64` when
the ids are `int64`. -/
def eqCurr {B : Nat} (d : DType) (ids : Tensor [B] d.Val) (curr : I32) : Tensor [B] Bool :=
  match d with
  | .i32 => ids.eq (bc curr)
  | .i64 => ids.eq (bc (curr.cast : I64))
  | _ => Tensor.full false

/-- The rest of the loop's body once the ids to count are known (source lines 69–71):
`has_curr_expert = tl.where(expert_ids == curr_expert, 1, 0)` (two Python numbers: `whereTyping?`
computes in `int32`, the constants `1` and `0` built in `int32`), `acc = acc + has_curr_expert`
(the wrapping `int32` addition) and `topk_ids_ptrs += BLOCK_SIZE` (the pointers advanced by the
constexpr, built in `int32`). -/
def tailOf {B : Nat} {d : DType} (acc : Tensor [B] I32) (ptrs : Tensor [B] (Ptr d))
    (eq : Tensor [B] Bool) : Prog (Tensor [B] I32 × Tensor [B] (Ptr d)) :=
  pure (acc + tl.where eq (Tensor.full (1 : I32)) (Tensor.full (0 : I32)),
    ptrs + (Tensor.full (IntV.ofNat B : I32) : Tensor [B] I32))

/-! ## The model -/

/-- One trip of the loop (source lines 62–71): `x` is the trip's `int32` induction variable,
`curr_expert` the program's expert; the state is `(acc, topk_ids_ptrs)`. `offsets` (line 57) is
`tl.arange 0 BLOCK_SIZE`, written where line 62 reads it. `if HAS_EXPERT_MAP:` (line 64) is a
compile-time condition: the model is the Lean `match` on the constexpr, each branch the lines the
front end visits. -/
def loopBody (d : DType) (hasMap : Bool) (BLOCK_SIZE : Nat) (expert_map : Ptr .i32) (topk_numel curr_expert : I32) (x : I32)
    (st : Tensor [BLOCK_SIZE] I32 × Tensor [BLOCK_SIZE] (Ptr d)) :
    Prog (Tensor [BLOCK_SIZE] I32 × Tensor [BLOCK_SIZE] (Ptr d)) := do
  let (acc, topk_ids_ptrs) := st
  -- line 62: `offsets < (topk_numel - x * BLOCK_SIZE)`, all `int32` (wrapping), compared signed
  let mask := (tl.arange 0 BLOCK_SIZE).ltS (topk_numel - x * (IntV.ofNat BLOCK_SIZE : I32))
  -- line 63: the masked load, `other=-1` in the ids' dtype
  let expert_ids ← tl.load topk_ids_ptrs (some mask) (some (Tensor.full (m1 d)))
  match hasMap with
  | true => do
    -- lines 65–67: the gather through the map (`int32`), enabled where the id is `>= 0`,
    -- `other=-1`; `expert_ids` is now the `int32` local id
    let expert_ids ← tl.load (mapPtrs d expert_map expert_ids) (some (geZero d expert_ids))
      (some (Tensor.full (m1 .i32)))
    tailOf acc topk_ids_ptrs (expert_ids.eq (bc curr_expert))
  | false =>
    tailOf acc topk_ids_ptrs (eqCurr d expert_ids curr_expert)

/-- `_count_expert_num_tokens` (source lines 46–74). `topk_ids_ptr` points at `topk_ids` (`int32`
or `int64`), `expert_map` at the map (`int32`; never read when `HAS_EXPERT_MAP` is false, where the
host passes `None`), `expert_num_tokens_ptr` at the output (`int32`); `num_experts` and
`topk_numel` are the `int32` runtime arguments (at the value `1` they are compile-time constants,
which give the same values: README, "Modelling decisions"); `HAS_EXPERT_MAP` and `BLOCK_SIZE` are
`tl.constexpr`, Lean parameters. -/
def count_kernel (d : DType) (hasMap : Bool) (BLOCK_SIZE : Nat) (topk_ids_ptr : Ptr d)
    (expert_map : Ptr .i32) (expert_num_tokens_ptr : Ptr .i32) (num_experts topk_numel : I32) :
    Kernel where
  body ctx := do
    let curr_expert := tl.program_id ctx 0
    let topk_ids_ptrs := topk_ids_ptr + tl.arange 0 BLOCK_SIZE
    let acc := tl.zeros [BLOCK_SIZE] .i32
    -- line 61: `range(tl.cdiv(topk_numel, BLOCK_SIZE))`: `range` with one argument evaluates its
    -- argument twice, the bound being the second value. `tl.cdiv` is the `int32` expression
    -- `(topk_numel + (BLOCK_SIZE - 1)) // BLOCK_SIZE` (a wrapping addition, then `tl.idiv`).
    let _ ← tl.idiv (Tensor.scalar (topk_numel + (IntV.ofNat (BLOCK_SIZE - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE : I32))
    let trips ← tl.idiv (Tensor.scalar (topk_numel + (IntV.ofNat (BLOCK_SIZE - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE : I32))
    let (acc, _) ← tl.forRange 0 trips.item 1 (acc, topk_ids_ptrs)
      (loopBody d hasMap BLOCK_SIZE expert_map topk_numel curr_expert)
    -- line 73: a runtime `if` (signed `int32` `<`)
    if curr_expert.lt num_experts then
      -- line 74: `tl.sum(acc)` of the `int32` block is the wrapping `int32` sum (`foldAll`)
      tl.store1 (expert_num_tokens_ptr + curr_expert)
        (Tensor.foldAll (· + ·) (0 : I32) acc)
    else
      pure ()

/-! ## The conditions -/

/-- What the host chooses: one launch through the wrapper `count_expert_num_tokens`, with the
contents of its two input tensors. -/
structure Inst where
  /-- The dtype of `topk_ids`. -/
  dtype : DType
  /-- `HAS_EXPERT_MAP = expert_map is not None`. -/
  hasMap : Bool
  /-- `num_local_experts`: the grid, the output's length and the argument `num_experts`. -/
  numLocal : Nat
  /-- The number of global experts (the map's length when there is a map). -/
  numGlobal : Nat
  /-- `topk_ids.numel()` (the argument `topk_numel`; the kernel reads `topk_ids` flat). -/
  numel : Nat
  /-- `BLOCK_SIZE` (the wrapper's `next_power_of_2(min(topk_ids.numel(), 1024))`). -/
  B : Nat
  /-- `topk_ids`, contiguous. -/
  ids : Buf dtype
  /-- The entries of `topk_ids` (flat). -/
  idsVal : Nat → dtype.Val
  /-- `expert_map` (unused when `hasMap` is false). -/
  emap : Buf .i32
  /-- The entries of `expert_map`. -/
  mapVal : Nat → I32
  /-- `expert_num_tokens`, allocated fresh by the wrapper. -/
  out : Buf .i32

/-- The instances covered: both paths of `HAS_EXPERT_MAP`, either ids dtype, every block size up
to `2^20`, every size whose `int32` index arithmetic does not wrap, buffers that hold what the
launch touches, and an output sharing no byte with an input. -/
structure Inst.ok (d : Inst) : Prop where
  /-- `topk_ids` is `int32` or `int64`. -/
  dtype_ok : d.dtype = .i32 ∨ d.dtype = .i64
  /-- `BLOCK_SIZE` is a power of two (`tl.arange`). -/
  b_pow2 : ∃ e, d.B = 2 ^ e
  /-- `BLOCK_SIZE ≤ 2^20` (Triton's largest block; the wrapper's is at most `1024`). -/
  b_le : d.B ≤ 2 ^ 20
  /-- `numel + BLOCK_SIZE ≤ 2^31`: `topk_numel` is an `int32` and `tl.cdiv`'s addition
  `topk_numel + (BLOCK_SIZE - 1)` does not wrap (a limit of the kernel: README, Exclusions). -/
  numel_b : d.numel + d.B ≤ 2 ^ 31
  /-- `topk_ids` holds its `numel` entries. -/
  ids_len : d.numel ≤ d.ids.len
  /-- With a map, the map holds the global experts. -/
  emap_len : d.hasMap = true → d.numGlobal ≤ d.emap.len
  /-- The output holds the local experts. -/
  out_len : d.numLocal ≤ d.out.len
  /-- `num_local_experts < 2^31`: the grid and the `int32` argument `num_experts`. -/
  numLocal_lt : d.numLocal < 2 ^ 31
  /-- The output shares no byte with `topk_ids` (the wrapper allocates it). -/
  out_ids : Disjoint d.out.bytes d.ids.bytes
  /-- With a map, the output shares no byte with `expert_map`. -/
  out_emap : d.hasMap = true → Disjoint d.out.bytes d.emap.bytes

/-- The inputs covered: the buffers are valid (aligned, inside `[0, 2^64)`, every byte present)
and hold the instance's entries; with a map, every entry of `topk_ids` is below `numGlobal` (a
global expert id, or negative: a negative entry is masked out of the gather). Without a map the
entries are only compared, so every value is covered. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.ids.Valid init ∧ d.out.Valid init ∧ d.ids.Holds init d.idsVal ∧
  (d.hasMap = true → (d.emap.Valid init ∧ d.emap.Holds init d.mapVal ∧
    ∀ j : Nat, j < d.numel → entryInt d.dtype (d.idsVal j) < d.numGlobal))

/-- The model on an instance: the buffers' first elements, the two scalar arguments as `int32`,
and the constexprs. -/
def model (d : Inst) : Kernel :=
  count_kernel d.dtype d.hasMap d.B (d.ids.ptr 0) (d.emap.ptr 0) (d.out.ptr 0)
    (IntV.ofNat d.numLocal) (IntV.ofNat d.numel)

/-- The wrapper's launch (lines 95–105): a 1-D grid of `num_local_experts` programs, the block
shape `[BLOCK_SIZE]`; `enable_fp_fusion = true` (Triton's default; the kernel has no float
operation), `debug = false`, `num_ctas = 1`, no `launch_pdl`. `num_warps = 4` is performance-only;
`num_stages = 3` changes no outcome of a model without a conflict inside a program. -/
def launch (d : Inst) : Launch := Launch.grid1 d.numLocal [[d.B]]

/-! ## The specification -/

/-- The local expert position `j` is routed to, from the host's inputs: without a map, the entry
itself; with one, `expert_map[entry]` for a non-negative entry and `-1` for a negative one. -/
def routed (d : Inst) (j : Nat) : Int :=
  let a := entryInt d.dtype (d.idsVal j)
  if d.hasMap then (if 0 ≤ a then (d.mapVal a.toNat).toInt else -1) else a

/-- The number of positions `j < numel` routed to expert `e`. -/
def countOf (d : Inst) (e : Nat) : Nat :=
  ((Finset.range d.numel).filter (fun j => routed d j = (e : Int))).card

/-- What the launch leaves: element `e` of `expert_num_tokens` is the exact count `countOf d e`
for every local expert `e`; every byte outside those elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ e : Nat, e < d.numLocal → ∃ c : I32,
      fin.read .i32 (d.out.ptr e).addr = some c ∧ c.toInt = (countOf d e : Int)) ∧
    ∀ z, z ∉ d.out.elems {e | e < d.numLocal} → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the wrapper's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)

/-- **The claim to prove.** -/
def Statement : Prop := ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

/-- An instance of the wrapper's launch with the buffers at fixed, separate addresses (`topk_ids`
at `2^32`, `expert_map` at `2^33`, the output at `2^34`) and zero contents: `dt` the ids' dtype,
`hm` whether a map is passed, `nl` local and `ng` global experts, `numel` entries and
`BLOCK_SIZE = 2^e`. -/
def mk (dt : DType) (hm : Bool) (nl ng numel e : Nat) : Inst where
  dtype := dt
  hasMap := hm
  numLocal := nl
  numGlobal := ng
  numel := numel
  B := 2 ^ e
  ids := ⟨2 ^ 32, numel⟩
  idsVal := fun _ => dt.ofBits 0
  emap := ⟨2 ^ 33, ng⟩
  mapVal := fun _ => (0 : I32)
  out := ⟨2 ^ 34, nl⟩

theorem bytes_disj {d d' : DType} (b : Buf d) (c : Buf d')
    (h : b.base + b.len * d.bytes ≤ c.base ∨ c.base + c.len * d'.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem mk_ok {dt : DType} {hm : Bool} {nl ng numel e : Nat}
    (hdt : dt = .i32 ∨ dt = .i64) (he : e ≤ 20) (hn : numel + 2 ^ e ≤ 2 ^ 31) (hnl : nl < 2 ^ 31)
    (hsz : numel * dt.bytes ≤ 2 ^ 32) (hng : ng * 4 ≤ 2 ^ 32) :
    (mk dt hm nl ng numel e).ok where
  dtype_ok := hdt
  b_pow2 := ⟨e, rfl⟩
  b_le := Nat.pow_le_pow_right (by norm_num) he
  numel_b := hn
  ids_len := le_refl _
  emap_len := fun _ => le_refl _
  out_len := le_refl _
  numLocal_lt := hnl
  out_ids := bytes_disj _ _ (Or.inr (by simp only [mk]; push_cast; omega))
  out_emap := fun _ => bytes_disj _ _ (Or.inr (by simp only [mk, DType.bytes]; push_cast; omega))

/-- A memory in which every byte is present (and `0`). -/
def zeroMem : Mem := fun _ => some 0

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨mk .i32 false 1 1 1 0, zeroMem,
    mk_ok (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num), ?_⟩
  have hb4 : (DType.i32.bytes : Int) = 4 := rfl
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ⟨?_, ?_, ?_, ?_⟩, ?_, ?_⟩
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp [zeroMem]
  · simp [mk]
  · simp only [mk, hb4]; norm_num
  · simp only [mk, hb4]; decide
  · intro x _; simp [zeroMem]
  · intro i hi
    have hi0 : i = 0 := by simp only [mk] at hi; omega
    subst hi0
    simp only [mk, Mem.read, Mem.Aligned, Buf.ptr, hb4, Nat.cast_zero]
    norm_num
    exact ⟨[0, 0, 0, 0], rfl, rfl⟩
  · intro hc; simp [mk] at hc

/-! ## Coverage of the host's launches

Every launch of the host's test (source lines 146–180): each `(num_tokens, topk, experts, ep_size,
dtype)` case at the ranks `0` and `ep_size - 1`, with the map and without it (the test's second
call, `local_ids`), `num_local_experts` as `make_expert_map` gives it and `BLOCK_SIZE` the
wrapper's `next_power_of_2(min(numel, 1024))`. The contents are not part of `Inst.ok`. Names:
`<case>_<tokens>x<topk>_e<experts>_ep<rank>of<ep_size>_<map|nomap>`. -/

def setup_vllm_1x2_e64_ep0of1_map : Inst := mk .i64 true 64 64 2 1
theorem setup_vllm_1x2_e64_ep0of1_map_ok : (setup_vllm_1x2_e64_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_1x2_e64_ep0of1_nomap : Inst := mk .i64 false 64 64 2 1
theorem setup_vllm_1x2_e64_ep0of1_nomap_ok : (setup_vllm_1x2_e64_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_1x6_e64_ep0of2_map : Inst := mk .i64 true 32 64 6 3
theorem setup_vllm_1x6_e64_ep0of2_map_ok : (setup_vllm_1x6_e64_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_1x6_e64_ep0of2_nomap : Inst := mk .i64 false 32 64 6 3
theorem setup_vllm_1x6_e64_ep0of2_nomap_ok : (setup_vllm_1x6_e64_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_1x8_e64_ep0of4_map : Inst := mk .i64 true 16 64 8 3
theorem setup_vllm_1x8_e64_ep0of4_map_ok : (setup_vllm_1x8_e64_ep0of4_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_1x8_e64_ep0of4_nomap : Inst := mk .i64 false 16 64 8 3
theorem setup_vllm_1x8_e64_ep0of4_nomap_ok : (setup_vllm_1x8_e64_ep0of4_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_4x2_e64_ep0of1_map : Inst := mk .i64 true 64 64 8 3
theorem setup_vllm_4x2_e64_ep0of1_map_ok : (setup_vllm_4x2_e64_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_4x2_e64_ep0of1_nomap : Inst := mk .i64 false 64 64 8 3
theorem setup_vllm_4x2_e64_ep0of1_nomap_ok : (setup_vllm_4x2_e64_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_4x6_e64_ep0of2_map : Inst := mk .i64 true 32 64 24 5
theorem setup_vllm_4x6_e64_ep0of2_map_ok : (setup_vllm_4x6_e64_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_4x6_e64_ep0of2_nomap : Inst := mk .i64 false 32 64 24 5
theorem setup_vllm_4x6_e64_ep0of2_nomap_ok : (setup_vllm_4x6_e64_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_4x8_e64_ep0of4_map : Inst := mk .i64 true 16 64 32 5
theorem setup_vllm_4x8_e64_ep0of4_map_ok : (setup_vllm_4x8_e64_ep0of4_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_4x8_e64_ep0of4_nomap : Inst := mk .i64 false 16 64 32 5
theorem setup_vllm_4x8_e64_ep0of4_nomap_ok : (setup_vllm_4x8_e64_ep0of4_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_11x2_e64_ep0of1_map : Inst := mk .i64 true 64 64 22 5
theorem setup_vllm_11x2_e64_ep0of1_map_ok : (setup_vllm_11x2_e64_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_11x2_e64_ep0of1_nomap : Inst := mk .i64 false 64 64 22 5
theorem setup_vllm_11x2_e64_ep0of1_nomap_ok : (setup_vllm_11x2_e64_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_11x6_e64_ep0of2_map : Inst := mk .i64 true 32 64 66 7
theorem setup_vllm_11x6_e64_ep0of2_map_ok : (setup_vllm_11x6_e64_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_11x6_e64_ep0of2_nomap : Inst := mk .i64 false 32 64 66 7
theorem setup_vllm_11x6_e64_ep0of2_nomap_ok : (setup_vllm_11x6_e64_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_11x8_e64_ep0of4_map : Inst := mk .i64 true 16 64 88 7
theorem setup_vllm_11x8_e64_ep0of4_map_ok : (setup_vllm_11x8_e64_ep0of4_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_11x8_e64_ep0of4_nomap : Inst := mk .i64 false 16 64 88 7
theorem setup_vllm_11x8_e64_ep0of4_nomap_ok : (setup_vllm_11x8_e64_ep0of4_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_127x2_e64_ep0of1_map : Inst := mk .i64 true 64 64 254 8
theorem setup_vllm_127x2_e64_ep0of1_map_ok : (setup_vllm_127x2_e64_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_127x2_e64_ep0of1_nomap : Inst := mk .i64 false 64 64 254 8
theorem setup_vllm_127x2_e64_ep0of1_nomap_ok : (setup_vllm_127x2_e64_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_127x6_e64_ep0of2_map : Inst := mk .i64 true 32 64 762 10
theorem setup_vllm_127x6_e64_ep0of2_map_ok : (setup_vllm_127x6_e64_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_127x6_e64_ep0of2_nomap : Inst := mk .i64 false 32 64 762 10
theorem setup_vllm_127x6_e64_ep0of2_nomap_ok : (setup_vllm_127x6_e64_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_127x8_e64_ep0of4_map : Inst := mk .i64 true 16 64 1016 10
theorem setup_vllm_127x8_e64_ep0of4_map_ok : (setup_vllm_127x8_e64_ep0of4_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_127x8_e64_ep0of4_nomap : Inst := mk .i64 false 16 64 1016 10
theorem setup_vllm_127x8_e64_ep0of4_nomap_ok : (setup_vllm_127x8_e64_ep0of4_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_128x2_e64_ep0of1_map : Inst := mk .i64 true 64 64 256 8
theorem setup_vllm_128x2_e64_ep0of1_map_ok : (setup_vllm_128x2_e64_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_128x2_e64_ep0of1_nomap : Inst := mk .i64 false 64 64 256 8
theorem setup_vllm_128x2_e64_ep0of1_nomap_ok : (setup_vllm_128x2_e64_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_128x6_e64_ep0of2_map : Inst := mk .i64 true 32 64 768 10
theorem setup_vllm_128x6_e64_ep0of2_map_ok : (setup_vllm_128x6_e64_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_128x6_e64_ep0of2_nomap : Inst := mk .i64 false 32 64 768 10
theorem setup_vllm_128x6_e64_ep0of2_nomap_ok : (setup_vllm_128x6_e64_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_128x8_e64_ep0of4_map : Inst := mk .i64 true 16 64 1024 10
theorem setup_vllm_128x8_e64_ep0of4_map_ok : (setup_vllm_128x8_e64_ep0of4_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_128x8_e64_ep0of4_nomap : Inst := mk .i64 false 16 64 1024 10
theorem setup_vllm_128x8_e64_ep0of4_nomap_ok : (setup_vllm_128x8_e64_ep0of4_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_3333x2_e64_ep0of1_map : Inst := mk .i64 true 64 64 6666 10
theorem setup_vllm_3333x2_e64_ep0of1_map_ok : (setup_vllm_3333x2_e64_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_3333x2_e64_ep0of1_nomap : Inst := mk .i64 false 64 64 6666 10
theorem setup_vllm_3333x2_e64_ep0of1_nomap_ok : (setup_vllm_3333x2_e64_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_3333x6_e64_ep0of2_map : Inst := mk .i64 true 32 64 19998 10
theorem setup_vllm_3333x6_e64_ep0of2_map_ok : (setup_vllm_3333x6_e64_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_3333x6_e64_ep0of2_nomap : Inst := mk .i64 false 32 64 19998 10
theorem setup_vllm_3333x6_e64_ep0of2_nomap_ok : (setup_vllm_3333x6_e64_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_3333x8_e64_ep0of4_map : Inst := mk .i64 true 16 64 26664 10
theorem setup_vllm_3333x8_e64_ep0of4_map_ok : (setup_vllm_3333x8_e64_ep0of4_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_3333x8_e64_ep0of4_nomap : Inst := mk .i64 false 16 64 26664 10
theorem setup_vllm_3333x8_e64_ep0of4_nomap_ok : (setup_vllm_3333x8_e64_ep0of4_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_7317x2_e64_ep0of1_map : Inst := mk .i64 true 64 64 14634 10
theorem setup_vllm_7317x2_e64_ep0of1_map_ok : (setup_vllm_7317x2_e64_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_7317x2_e64_ep0of1_nomap : Inst := mk .i64 false 64 64 14634 10
theorem setup_vllm_7317x2_e64_ep0of1_nomap_ok : (setup_vllm_7317x2_e64_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_7317x6_e64_ep0of2_map : Inst := mk .i64 true 32 64 43902 10
theorem setup_vllm_7317x6_e64_ep0of2_map_ok : (setup_vllm_7317x6_e64_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_7317x6_e64_ep0of2_nomap : Inst := mk .i64 false 32 64 43902 10
theorem setup_vllm_7317x6_e64_ep0of2_nomap_ok : (setup_vllm_7317x6_e64_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_7317x8_e64_ep0of4_map : Inst := mk .i64 true 16 64 58536 10
theorem setup_vllm_7317x8_e64_ep0of4_map_ok : (setup_vllm_7317x8_e64_ep0of4_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_vllm_7317x8_e64_ep0of4_nomap : Inst := mk .i64 false 16 64 58536 10
theorem setup_vllm_7317x8_e64_ep0of4_nomap_ok : (setup_vllm_7317x8_e64_ep0of4_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_1x1_e32_ep0of2_map : Inst := mk .i64 true 16 32 1 0
theorem setup_numel_1x1_e32_ep0of2_map_ok : (setup_numel_1x1_e32_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_1x1_e32_ep0of2_nomap : Inst := mk .i64 false 16 32 1 0
theorem setup_numel_1x1_e32_ep0of2_nomap_ok : (setup_numel_1x1_e32_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_112x1_e32_ep0of2_map : Inst := mk .i64 true 16 32 112 7
theorem setup_numel_112x1_e32_ep0of2_map_ok : (setup_numel_112x1_e32_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_112x1_e32_ep0of2_nomap : Inst := mk .i64 false 16 32 112 7
theorem setup_numel_112x1_e32_ep0of2_nomap_ok : (setup_numel_112x1_e32_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_1024x1_e32_ep0of2_map : Inst := mk .i64 true 16 32 1024 10
theorem setup_numel_1024x1_e32_ep0of2_map_ok : (setup_numel_1024x1_e32_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_1024x1_e32_ep0of2_nomap : Inst := mk .i64 false 16 32 1024 10
theorem setup_numel_1024x1_e32_ep0of2_nomap_ok : (setup_numel_1024x1_e32_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_1025x1_e32_ep0of2_map : Inst := mk .i64 true 16 32 1025 10
theorem setup_numel_1025x1_e32_ep0of2_map_ok : (setup_numel_1025x1_e32_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_1025x1_e32_ep0of2_nomap : Inst := mk .i64 false 16 32 1025 10
theorem setup_numel_1025x1_e32_ep0of2_nomap_ok : (setup_numel_1025x1_e32_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_2047x1_e32_ep0of2_map : Inst := mk .i64 true 16 32 2047 10
theorem setup_numel_2047x1_e32_ep0of2_map_ok : (setup_numel_2047x1_e32_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_2047x1_e32_ep0of2_nomap : Inst := mk .i64 false 16 32 2047 10
theorem setup_numel_2047x1_e32_ep0of2_nomap_ok : (setup_numel_2047x1_e32_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_8191x1_e32_ep0of2_map : Inst := mk .i64 true 16 32 8191 10
theorem setup_numel_8191x1_e32_ep0of2_map_ok : (setup_numel_8191x1_e32_ep0of2_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_numel_8191x1_e32_ep0of2_nomap : Inst := mk .i64 false 16 32 8191 10
theorem setup_numel_8191x1_e32_ep0of2_nomap_ok : (setup_numel_8191x1_e32_ep0of2_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_dsv3_4096x8_e256_ep0of8_map : Inst := mk .i32 true 32 256 32768 10
theorem setup_dsv3_4096x8_e256_ep0of8_map_ok : (setup_dsv3_4096x8_e256_ep0of8_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_dsv3_4096x8_e256_ep0of8_nomap : Inst := mk .i32 false 32 256 32768 10
theorem setup_dsv3_4096x8_e256_ep0of8_nomap_ok : (setup_dsv3_4096x8_e256_ep0of8_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_dsv3rr_16384x8_e256_ep0of16_map : Inst := mk .i32 true 16 256 131072 10
theorem setup_dsv3rr_16384x8_e256_ep0of16_map_ok : (setup_dsv3rr_16384x8_e256_ep0of16_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_dsv3rr_16384x8_e256_ep0of16_nomap : Inst := mk .i32 false 16 256 131072 10
theorem setup_dsv3rr_16384x8_e256_ep0of16_nomap_ok : (setup_dsv3rr_16384x8_e256_ep0of16_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_qwen_2000x8_e128_ep0of1_map : Inst := mk .i32 true 128 128 16000 10
theorem setup_qwen_2000x8_e128_ep0of1_map_ok : (setup_qwen_2000x8_e128_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_qwen_2000x8_e128_ep0of1_nomap : Inst := mk .i32 false 128 128 16000 10
theorem setup_qwen_2000x8_e128_ep0of1_nomap_ok : (setup_qwen_2000x8_e128_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_uneven_513x8_e128_ep0of3_map : Inst := mk .i32 true 43 128 4104 10
theorem setup_uneven_513x8_e128_ep0of3_map_ok : (setup_uneven_513x8_e128_ep0of3_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_uneven_513x8_e128_ep0of3_nomap : Inst := mk .i32 false 43 128 4104 10
theorem setup_uneven_513x8_e128_ep0of3_nomap_ok : (setup_uneven_513x8_e128_ep0of3_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_uneven_513x8_e128_ep2of3_map : Inst := mk .i32 true 42 128 4104 10
theorem setup_uneven_513x8_e128_ep2of3_map_ok : (setup_uneven_513x8_e128_ep2of3_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_uneven_513x8_e128_ep2of3_nomap : Inst := mk .i32 false 42 128 4104 10
theorem setup_uneven_513x8_e128_ep2of3_nomap_ok : (setup_uneven_513x8_e128_ep2of3_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_mixtral_777x2_e8_ep0of1_map : Inst := mk .i32 true 8 8 1554 10
theorem setup_mixtral_777x2_e8_ep0of1_map_ok : (setup_mixtral_777x2_e8_ep0of1_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_mixtral_777x2_e8_ep0of1_nomap : Inst := mk .i32 false 8 8 1554 10
theorem setup_mixtral_777x2_e8_ep0of1_nomap_ok : (setup_mixtral_777x2_e8_ep0of1_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_one_1x8_e256_ep0of8_map : Inst := mk .i32 true 32 256 8 3
theorem setup_one_1x8_e256_ep0of8_map_ok : (setup_one_1x8_e256_ep0of8_map).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
def setup_one_1x8_e256_ep0of8_nomap : Inst := mk .i32 false 32 256 8 3
theorem setup_one_1x8_e256_ep0of8_nomap_ok : (setup_one_1x8_e256_ep0of8_nomap).ok := mk_ok (by decide) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)

end Claims.CountExpertTokens
