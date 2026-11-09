import TVA

/-!
# `lora_shrink_bf16`: the claim

Source: `source/lora_shrink.py`, vLLM's `_lora_shrink_kernel` (lines
308–423) with the helpers `do_shrink_kernel` (185–305) and `mm_k` (83–182), which the front end
inlines at their calls. The kernel reads the module-level names `triton` and `tl` (lines 79–80)
and the module's `@triton.jit` functions, each bound only by its own `def`. The host is
`lora_shrink` (503–610) with `lora_kernel_meta` (462–500) and `_get_lora_a_ptr` (426–459).

The covered launches are the host's (`SETUP.md`): any number of slices (`SLICE_NUM = 1`, the kernel
reading the weight tensor itself, or several, through a `uint64` table of the slices' weight
addresses), vLLM's default configuration (`BLOCK_K = 256`, `SPLIT_K = 64` below 128 tokens;
`BLOCK_K = 32`, `SPLIT_K = 8` from 128), either value of `EVEN_K`, `USE_GDC = False` and
`STORE_PARTIALS = False`. A program takes one adapter group (axis 2), one slice (axis 1), and one
`32 × 16` tile of the group's rows and of the rank and one of the `SPLIT_K` parts of `K` (axis 0);
it gathers the tile's token indices, multiplies those rows of `inputs` by the adapter's LoRA A rows
over its part of `K` in `bfloat16` with a `float32` accumulator, scales by `scaling` and adds the
`float32` tile into the zeroed output with a relaxed `tl.atomic_add`.

The claim is `Statement`: `∀ K P, Assumptions K P → Spec K`. This file proves only `consistent`,
`satisfiable` and the coverage facts `setup_*_ok`; it does not prove `Statement`.
-/

namespace Claims.LoraShrink

open TVA

/-! ## The model -/

/-- The configuration of the model's `tl.dot` (source lines 152, 166, 177): `bfloat16` operands
into a `float32` accumulator, no `input_precision` (Triton's default `tf32`, which it ignores for
16-bit operands) and no `max_num_imprecise_acc` (`0`). -/
abbrev dotCfg : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-- `_lora_shrink_kernel` (source lines 308–423), with `do_shrink_kernel` (185–305) and `mm_k`
(83–182) inlined at their calls: each helper's body with the call's arguments bound to its
parameters (an annotation on a helper's parameter, `K: tl.constexpr`, `ak_stride: int`, has no
effect: the parameter holds what the call passes, so `mm_k`'s `K` is the kernel's runtime `int32`
`K`).

The kernel's argument `lora_ptr` is a `*bf16` when the host passes one weight tensor
(`SLICE_NUM = 1`, `_get_lora_a_ptr` returns the tensor itself) and a `*uint64` otherwise (the table
of the slices' weight addresses). The model takes it as `lora_ptr` in the first case and as
`lora_tab` in the second; line 226's compile-time branch reads the one the launch passes, and
`model` gives the host's argument in that one (the other is never read).

The compile-time arguments `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `SPLIT_K`, `GROUP_SIZE_M`, `SLICE_NUM`
and `EVEN_K` are parameters, and each compile-time branch on them is a Lean `if`. On every covered
launch the other compile-time arguments are:
* `USE_GDC = False` (line 269, and the host's `use_gdc = False`): no `gdc_wait` (lines 147–148,
  161–162, 172–173), no `gdc_launch_dependents` (273–274);
* `STORE_PARTIALS = False`: lines 343–346, 278–283 and 295–298 are not visited;
* `CAST_TYPE = False` (line 267): lines 150–151, 164–165, 175–176 are not visited, and `b_dtype`
  (line 268) is never read.

`output_split_stride` is read only in the branch `STORE_PARTIALS = False` leaves unvisited (line
282), so the model has no such parameter; `output_d0_stride` is read only when `SLICE_NUM > 1`
(line 286).

An integer argument equal to `1` is a compile-time constant (SEMANTICS.md, "Scalar arguments"):
`input_d1_stride`, `lora_d2_stride` and `output_d2_stride` always (the host's contiguous layouts),
`M` at one token, `K` and `input_d0_stride` at a hidden size of `1`. Each use of such a value in the
source is an operand of `*`, `+`, `-` or a comparison with an `int32` tensor, or of `tl.cdiv` with a
constexpr (Python's `(1 + 31) // 32 = 1`, then an operand of `-` with an `int32` tensor), so the
model's `int32` `1` computes the same values. -/
def loraShrink (input_ptr lora_ptr : Ptr .bf16) (lora_tab : Ptr .u64) (out_ptr : Ptr .f32)
    (M N K : I32)
    (token_indices_sorted_by_lora_ids num_tokens_per_lora lora_token_start_loc lora_ids : Ptr .i32)
    (scaling : F32)
    (input_d0_stride input_d1_stride lora_d0_stride lora_d1_stride lora_d2_stride
      output_d0_stride output_d1_stride output_d2_stride : I32)
    (BLOCK_M BLOCK_N BLOCK_K SPLIT_K GROUP_SIZE_M SLICE_NUM : Nat) (EVEN_K : Bool) : Kernel where
  body ctx := do
    -- line 343: `if STORE_PARTIALS:` is not visited (`STORE_PARTIALS = False`)
    -- line 347: `tl.cdiv(N, BLOCK_N)` is `(N + (BLOCK_N - 1)) // BLOCK_N`: the constexpr
    -- `BLOCK_N - 1` built in `int32`, the wrapping `int32` `+`, the truncating `//` (`tl.idiv`)
    let cta_n_num := (← tl.idiv (Tensor.scalar (N + (IntV.ofNat (BLOCK_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_N : I32))).item
    -- line 348: the same with `M`, `BLOCK_M`
    let cta_m_num := (← tl.idiv (Tensor.scalar (M + (IntV.ofNat (BLOCK_M - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_M : I32))).item
    -- line 350: axis 0 of the grid, an `int32`
    let pid_sk_m_n := tl.program_id ctx 0
    -- line 351: `pid_sk_m_n % SPLIT_K`, the constexpr built in `int32`; the truncating `%`
    let pid_sk := (← tl.irem (Tensor.scalar pid_sk_m_n)
      (Tensor.scalar (IntV.ofNat SPLIT_K : I32))).item
    -- line 353: `pid_sk_m_n // SPLIT_K`, the truncating `//` (`tl.idiv`)
    let pid_m_n := (← tl.idiv (Tensor.scalar pid_sk_m_n)
      (Tensor.scalar (IntV.ofNat SPLIT_K : I32))).item
    -- line 354: the constexpr built in `int32`, the wrapping `int32` product
    let num_pid_in_group := (IntV.ofNat GROUP_SIZE_M : I32) * cta_n_num
    -- line 355: `int32` `//`
    let group_id := (← tl.idiv (Tensor.scalar pid_m_n) (Tensor.scalar num_pid_in_group)).item
    -- line 356: the constexpr built in `int32`, the wrapping `int32` product
    let first_pid_m := group_id * (IntV.ofNat GROUP_SIZE_M : I32)
    -- line 357: Python's `min` with a tensor argument is `tl.minimum`, the constexpr an `int32`
    -- tensor of its own type; on `int32` it is the exact `tl.iminimum`
    let group_size_m := (tl.iminimum (Tensor.scalar (cta_m_num - first_pid_m))
      (Tensor.scalar (IntV.ofNat GROUP_SIZE_M : I32))).item
    -- line 360: two `int32` `%` (`tl.irem`), the inner one first, then the wrapping `+`
    let pid_m := first_pid_m + (← tl.irem (Tensor.scalar (← tl.irem (Tensor.scalar pid_m_n)
      (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
    -- line 361: `pid_m_n % num_pid_in_group` computed again, then `int32` `//`
    let pid_n := (← tl.idiv (Tensor.scalar (← tl.irem (Tensor.scalar pid_m_n)
      (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
    -- line 363: axis 1 of the grid, the slice
    let slice_id := tl.program_id ctx 1
    -- line 364: axis 2 of the grid, the adapter group
    let lora_idx := tl.program_id ctx 2
    -- line 366: a scalar load from the `*int32` buffer
    let lora_id ← tl.load1 (lora_ids + lora_idx)
    -- line 367: `lora_id == -1`, the Python `-1` built in `int32`; lines 368–369 `return`
    if IntV.eq lora_id (IntV.ofInt (-1) : I32) then return () else do
      -- line 371
      let lora_m_size ← tl.load1 (num_tokens_per_lora + lora_idx)
      -- line 373: the constexpr built in `int32`, the wrapping `int32` product
      let cta_m_offset := pid_m * (IntV.ofNat BLOCK_M : I32)
      -- line 374: `cta_m_offset >= lora_m_size` is the signed `lora_m_size <= cta_m_offset`;
      -- lines 375–376 `return`
      if IntV.le lora_m_size cta_m_offset then return () else do
        -- line 379: Python's `min` again (`tl.iminimum`); the wrapping `int32` difference
        let cta_m_len := (tl.iminimum (Tensor.scalar (IntV.ofNat BLOCK_M : I32))
          (Tensor.scalar (lora_m_size - cta_m_offset))).item
        -- line 382
        let lora_m_indices_start ← tl.load1 (lora_token_start_loc + lora_idx)
        -- lines 383–385: two `int32` offsets added to the `*int32` pointer, left to right
        let cta_lora_seq_indices :=
          token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
        -- line 387: `%` by the `int32` scalar `cta_m_len`, broadcast
        let offset_m : Tensor [BLOCK_M] I32 ← tl.irem (tl.arange 0 BLOCK_M) (bc cta_m_len)
        -- line 388: one load per lane, no mask
        let ram : Tensor [BLOCK_M] I32 ← tl.load (cta_lora_seq_indices + offset_m)
        -- ==== `do_shrink_kernel` (lines 185–305), inlined at its call of lines 390–423 ====
        -- lines 226–233: a compile-time branch on `SLICE_NUM`. With one slice `cur_lora_ptr` is
        -- `lora_ptr`; with several, the scalar `uint64` load of entry `slice_id` of the table,
        -- converted to a `*bf16` (`.to(tl.pointer_type(input_ptr.dtype.element_ty))`, the
        -- compile-time `bfloat16` of `input_ptr`'s buffer): `tl.intToPtr`
        let cur_lora_ptr : Ptr .bf16 ← if SLICE_NUM = 1 then pure lora_ptr else do
            let addr ← tl.load1 (lora_tab + slice_id)
            pure (tl.intToPtr .bf16 addr)
        -- line 236: the constexpr built in `int32`, the wrapping `int32` product, then `+`
        let offset_n : Tensor [BLOCK_N] I32 :=
          tl.arange 0 BLOCK_N + pid_n * (IntV.ofNat BLOCK_N : I32)
        -- line 237: `tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)`: both
        -- hints on the value the `%` defines, the pair's promise (groups of `BLOCK_N` consecutive
        -- values, `BLOCK_N` dividing the first of each group)
        let rbn ← tl.max_contiguous_multiple_of
          (← tl.irem offset_n (bc N) : Tensor [BLOCK_N] I32) BLOCK_N BLOCK_N
        -- line 240: the `int32` product `pid_sk * BLOCK_K`, then `+` the `int32` `tl.arange`
        let offset_k : Tensor [BLOCK_K] I32 :=
          pid_sk * (IntV.ofNat BLOCK_K : I32) + tl.arange 0 BLOCK_K
        -- lines 241–246, left to right: `input_ptr + ram[:, None].to(tl.int64) * input_d0_stride`
        -- (the `int64` row index times the `int32` stride promoted to `int64`), then
        -- `+ offset_k[None, :] * input_d1_stride` (an `int32` product, added to the pointers)
        let ram64 : Tensor [BLOCK_M] I64 ← tl.convert .i32 .i64 ram
        let d0_64 : I64 := (← tl.convert .i32 .i64 (Tensor.scalar input_d0_stride)).item
        let a_ptr := (input_ptr + ram64.col BLOCK_K * d0_64) +
          (offset_k * input_d1_stride).row BLOCK_M
        -- lines 247–252, left to right: `cur_lora_ptr + lora_d0_stride * lora_index` (the `int32`
        -- scalar product, `lora_index` the call's `lora_id`), `+ rbn[None, :] * lora_d1_stride`,
        -- `+ offset_k[:, None] * lora_d2_stride` (`int32` products, each added to the pointers)
        let b_ptr := ((cur_lora_ptr + lora_d0_stride * lora_id) +
          rbn.row BLOCK_K * lora_d1_stride) + offset_k.col BLOCK_N * lora_d2_stride
        -- ==== `mm_k` (lines 83–182), inlined at its call of lines 255–271 ====
        -- line 128: `+0.0` in every element
        let accumulator := tl.zeros [BLOCK_M, BLOCK_N] .f32
        -- line 131: `STEP_K = BLOCK_K * SPLIT_K`, a Python integer, built in `int32` where used
        let STEP_K : I32 := IntV.ofNat (BLOCK_K * SPLIT_K)
        -- line 134: `tl.cdiv(K, STEP_K)`: the `int32` sum (wrapping), then the truncating `//`
        let num_iters := (← tl.idiv (Tensor.scalar (K + (STEP_K - 1))) (Tensor.scalar STEP_K)).item
        -- line 136: `range(num_iters)` from the `int32` `0`, step `1`; the loop carries
        -- `accumulator`, `a_ptr` and `b_ptr` (lines 179–180 assign the pointers)
        let (accumulator, _a_ptr, _b_ptr) ← tl.forRange (IntV.ofInt 0 : I32) num_iters
            (IntV.ofInt 1 : I32) (accumulator, a_ptr, b_ptr)
            fun k (accumulator, a_ptr, b_ptr) => do
          -- line 138: `k * STEP_K + base_k`, `base_k = pid_sk * BLOCK_K` (line 270)
          let iter_k := k * STEP_K + pid_sk * (IntV.ofNat BLOCK_K : I32)
          -- line 141
          let block_end := iter_k + (IntV.ofNat BLOCK_K : I32)
          -- line 143: `if EVEN_K` is a compile-time branch; the model keeps it as a Lean `if`
          let accumulator ← if EVEN_K then do
              -- line 146
              let tiled_b : Tensor [BLOCK_K, BLOCK_N] BF16 ← tl.load b_ptr
              -- line 149
              let tiled_a : Tensor [BLOCK_M, BLOCK_K] BF16 ← tl.load a_ptr
              -- line 152: `accumulator += tl.dot(tiled_a, tiled_b)`: `tl.dot_add`
              tl.dot_add dotCfg tiled_a tiled_b accumulator
            else if IntV.le K iter_k then
              -- lines 155–157: `iter_k >= K`: `pass`
              pure accumulator
            else if IntV.le block_end K then do
              -- lines 158–166: the whole block in range, no mask
              let tiled_b : Tensor [BLOCK_K, BLOCK_N] BF16 ← tl.load b_ptr
              let tiled_a : Tensor [BLOCK_M, BLOCK_K] BF16 ← tl.load a_ptr
              tl.dot_add dotCfg tiled_a tiled_b accumulator
            else do
              -- lines 169–177: the partial block, masked; `other=0.0` is the `float32` `+0.0`
              -- (a tensor of its own type), converted to the pointer's dtype `bfloat16`
              let k_offsets : Tensor [BLOCK_K] I32 := tl.arange 0 BLOCK_K
              let mask : Tensor [BLOCK_K] Bool := (iter_k + k_offsets).ltS K
              let tiled_b : Tensor [BLOCK_K, BLOCK_N] BF16 ← tl.load b_ptr (some (mask.col BLOCK_N))
                (some (← tl.convert .f32 .bf16
                  (tl.full [BLOCK_K, BLOCK_N] (FloatV.litF32 (FloatV.ofPat 0)))))
              let tiled_a : Tensor [BLOCK_M, BLOCK_K] BF16 ← tl.load a_ptr (some (mask.row BLOCK_M))
                (some (← tl.convert .f32 .bf16
                  (tl.full [BLOCK_M, BLOCK_K] (FloatV.litF32 (FloatV.ofPat 0)))))
              tl.dot_add dotCfg tiled_a tiled_b accumulator
          -- lines 179–180: the `int32` `STEP_K * stride`, added to every pointer of the block
          let a_ptr := a_ptr +
            (bc (STEP_K * input_d1_stride) : Tensor [BLOCK_M, BLOCK_K] I32)
          let b_ptr := b_ptr +
            (bc (STEP_K * lora_d2_stride) : Tensor [BLOCK_K, BLOCK_N] I32)
          return (accumulator, a_ptr, b_ptr)
        -- line 273: `if USE_GDC:` is not visited
        -- line 276
        let offset_cn : Tensor [BLOCK_N] I32 :=
          tl.arange 0 BLOCK_N + pid_n * (IntV.ofNat BLOCK_N : I32)
        -- line 277
        let offset_cm : Tensor [BLOCK_M] I32 := tl.arange 0 BLOCK_M
        -- lines 278–287 (`STORE_PARTIALS = False`): line 286's conditional expression on the
        -- compile-time `SLICE_NUM == 1`: `out_ptr`, or `out_ptr + slice_id * output_d0_stride` (the
        -- `int32` product added to the pointer)
        let cur_out_ptr : Ptr .f32 :=
          if SLICE_NUM = 1 then out_ptr else out_ptr + slice_id * output_d0_stride
        -- lines 288–292, left to right: `cur_out_ptr + ram[:, None] * output_d1_stride`, then
        -- `+ offset_cn[None, :] * output_d2_stride` (`int32` products, each added to the pointers)
        let c_ptr := (cur_out_ptr + ram.col BLOCK_N * output_d1_stride) +
          offset_cn.row BLOCK_M * output_d2_stride
        -- line 293: two signed `int32` comparisons, broadcast, then `&`
        let c_mask := ((offset_cm.col BLOCK_N).ltS cta_m_len).and ((offset_cn.row BLOCK_M).ltS N)
        -- line 300: `accumulator *= scaling`, the `float32` scalar broadcast: an IEEE product
        let accumulator ← tl.mul accumulator (bc scaling)
        -- line 302: a compile-time branch on `SPLIT_K`
        if SPLIT_K = 1 then
          -- line 303: `accumulator` is already `float32`, the pointer's dtype
          tl.store c_ptr accumulator (some c_mask)
        else do
          -- line 305: an expression statement, so the returned value is never used:
          -- `tl.atomic_add_unused` with the source's `sem="relaxed"` and the default
          -- `scope="gpu"` (its order is `acq_rel`, `MemSem.unused`)
          tl.atomic_add_unused c_ptr accumulator (some c_mask) .relaxed
        return ()

/-! ## The conditions -/

/-- What the host chooses: one launch of `_lora_shrink_kernel` through `lora_shrink`. The metadata
the host builds from its `token_lora_mapping` (`lora_kernel_meta`) is part of the instance as
values: `adapter` is the token-to-adapter map, `n` the number of distinct adapter values,
`groupId`/`groupCnt`/`groupStart` the distinct values in increasing order with their counts and
cumulative sums, and `tIdx` the stably sorted token indices. -/
structure Inst where
  /-- The number of tokens (`inputs.size(0)`, the kernel's `M`). -/
  M : Nat
  /-- The hidden size (the kernel's `K`). -/
  K : Nat
  /-- The rank (the kernel's `N`). -/
  R : Nat
  /-- `max_loras`: the adapter rows of each weight tensor. -/
  maxL : Nat
  /-- The number of slices, `len(lora_a_weights)` (the kernel's `SLICE_NUM`). -/
  S : Nat
  /-- The number of distinct adapter values of the mapping, `-1` included when present. -/
  n : Nat
  /-- `num_active_loras`, the grid's axis 2 (`max_loras + 1`, or the distinct count with
  `specialize_active_lora`). -/
  G : Nat
  /-- The host's `token_lora_mapping`: the adapter of each token, `-1` for none. -/
  adapter : Nat → Int
  /-- The distinct adapter values in increasing order (`-1` first when present). -/
  groupId : Nat → Int
  /-- The number of tokens of group `g`. -/
  groupCnt : Nat → Nat
  /-- The cumulative sums of `groupCnt` from `0`. -/
  groupStart : Nat → Nat
  /-- `token_indices_sorted_by_lora_ids`, as values. -/
  tIdx : Nat → Nat
  /-- `scaling`: the host's Python `float`, a double. The kernel's argument is its `float32`
  rounding, `scaling32 d = FloatV.hostF32 d.scaling` (SEMANTICS.md, "Scalar arguments"). -/
  scaling : F64
  /-- `inputs`: `[M, K]` `bfloat16`, contiguous. -/
  inp : Buf .bf16
  /-- `lora_a_weights[s]` for each slice `s < S`: `[maxL, 1, R, K]` (or `[maxL, R, K]`)
  `bfloat16`, contiguous. -/
  lora : Nat → Buf .bf16
  /-- The `uint64` tensor of the slices' weight addresses that `_get_lora_a_ptr` builds when
  `S > 1` (`[S]`); with one slice the host passes `lora 0` itself and this buffer is not used. -/
  tab : Buf .u64
  /-- The output: `[S, M, R]` `float32`, contiguous, from `torch.zeros`. -/
  out : Buf .f32
  /-- `token_indices_sorted_by_lora_ids`: `[M]` `int32`. -/
  tIdxBuf : Buf .i32
  /-- `num_tokens_per_lora`: `[maxL + 1]` `int32`. -/
  cntBuf : Buf .i32
  /-- `lora_token_start_loc`: `[maxL + 2]` `int32`. -/
  startBuf : Buf .i32
  /-- `lora_ids`: `[maxL + 1]` `int32`. -/
  idBuf : Buf .i32

/-- The kernel's `float32` `scaling` argument: the host's double rounded to nearest even
(`FloatV.hostF32`, C's `(float)`); exact for the `1.0` and `0.5` the host passes. -/
def scaling32 (d : Inst) : F32 := FloatV.hostF32 d.scaling

/-- The host's `BLOCK_K` (source line 555, vLLM's default shrink config): `256` below 128 tokens,
`32` from 128. -/
def bk (d : Inst) : Nat := if d.M < 128 then 256 else 32

/-- The host's `SPLIT_K` (source line 556): `64` below 128 tokens, `8` from 128. -/
def sk (d : Inst) : Nat := if d.M < 128 then 64 else 8

/-- The grid's axis 0 (line 565): `SPLIT_K · cdiv(M, 32) · cdiv(R, 16)`. -/
def grid0 (d : Inst) : Nat := sk d * ((d.M + 31) / 32) * ((d.R + 15) / 16)

/-- Group `g`'s id (line 366's `lora_id`): the `g`-th distinct value, `-1` past them. -/
def idAt (d : Inst) (g : Nat) : Int := if g < d.n then d.groupId g else -1

/-- Group `g`'s token count (line 371's `lora_m_size`): `0` past the distinct ids. -/
def cntAt (d : Inst) (g : Nat) : Nat := if g < d.n then d.groupCnt g else 0

/-- Entry `g` of `lora_token_start_loc` (line 382): the cumulative sum for `g ≤ n`, and `0` past
it (the host's `torch.zeros`). -/
def startAt (d : Inst) (g : Nat) : Nat := if g ≤ d.n then d.groupStart g else 0

/-- The element of the output for slice `s`, token `t`, rank column `j`. -/
def outAt (d : Inst) (s t j : Nat) : Nat := s * (d.M * d.R) + t * d.R + j

/-- The elements of the output's `[S, M, R]` tensor. -/
def outIdx (d : Inst) : Set Nat :=
  {e | ∃ s t j, s < d.S ∧ t < d.M ∧ j < d.R ∧ e = outAt d s t j}

/-- The instances covered: every launch of the host whose rank is a multiple of `16` and whose
`int32` weight and output offsets do not wrap (limits of the kernel, README "Exclusions"), with
hidden sizes up to `2^17`. -/
structure Inst.ok (d : Inst) : Prop where
  /-- **A limit of the kernel.** `R` is a multiple of `BLOCK_N = 16`. Line 237's hint
  `tl.max_contiguous(tl.multiple_of(offset_n % N, 16), 16)` promises that the 16 columns of a
  tile are consecutive and that 16 divides the first; on the last tile of a rank that is not a
  multiple of 16 the columns wrap around `R` and the promise is false, which is undefined
  behaviour (SEMANTICS.md, "Compiler hints"). Ranks 1, 4 and 8 are outside the claim. -/
  R_blk : d.R % 16 = 0
  /-- `R ≥ 1` (with `R_blk`, `R ≥ 16`). -/
  R_pos : 0 < d.R
  /-- `K ≥ 1`. -/
  K_pos : 0 < d.K
  /-- `K ≤ 2^17 = 131072`: every hidden size `SETUP.md` names (2 to 128256). -/
  K_top : d.K ≤ 2 ^ 17
  /-- `R·K < 2^31`: the host's `lora_d0_stride = R·K` is an `int32` argument (`R ≤ 256` and
  `K ≤ 128256` in `SETUP.md` give `R·K < 2^25`). -/
  RK_lt : d.R * d.K < 2 ^ 31
  /-- **A limit of the kernel.** `(M − 1)·R < 2^31`: line 290's `ram[:, None] * output_d1_stride`
  is an `int32` product of a token index and `R`, which wraps beyond. It also makes `M` an
  `int32` and `M + 31` (`tl.cdiv` of line 348) fit in `int32`. -/
  out_wrap : (d.M - 1) * d.R < 2 ^ 31
  /-- **A limit of the kernel.** For each token's adapter `l ≥ 0`, `l·R·K < 2^31`: line 249's
  `lora_d0_stride * lora_index` is an `int32` product, which wraps beyond. -/
  lora_wrap : ∀ t, t < d.M → 0 ≤ d.adapter t → d.adapter t * ((d.R * d.K : Nat) : Int) < 2 ^ 31
  /-- At least one slice. -/
  S_pos : 0 < d.S
  /-- **A limit of the kernel.** With several slices, `(S − 1)·M·R < 2^31`: line 286's
  `slice_id * output_d0_stride` is an `int32` product of the slice and `M·R` (the host's
  `output_d0_stride`, then an `int32` argument), which wraps beyond. -/
  slice_wrap : 1 < d.S → (d.S - 1) * (d.M * d.R) < 2 ^ 31
  /-- The grid's axis 0 is within CUDA's limit (`Launch.Valid`). -/
  grid0_lt : grid0 d < 2 ^ 31
  /-- The grid's axis 1 (the slices) is within CUDA's limit (`Launch.Valid`). -/
  S_lt : d.S < 2 ^ 16
  /-- The grid's axis 2 holds every group. -/
  n_le_G : d.n ≤ d.G
  /-- `num_active_loras ≤ max_loras + 1`. -/
  G_le : d.G ≤ d.maxL + 1
  /-- The grid's axis 2 is within CUDA's limit (`Launch.Valid`). -/
  G_lt : d.G < 2 ^ 16
  /-- Fewer than `2^31` programs (`Launch.Valid`). -/
  grid_size : grid0 d * d.S * d.G < 2 ^ 31
  /-- `scaling` is finite, and so is its `float32` rounding (the host's layers pass `1.0`, its test
  `0.5`). -/
  sc_fin : d.scaling.isFinite = true ∧ (scaling32 d).isFinite = true
  /-- Every token's adapter is in `[-1, maxL)`. -/
  ad_range : ∀ t, t < d.M → -1 ≤ d.adapter t ∧ d.adapter t < (d.maxL : Int)
  /-- Each group `g < n` has a token. -/
  grp_nonempty : ∀ g, g < d.n → ∃ t, t < d.M ∧ d.adapter t = d.groupId g
  /-- The groups' ids are distinct. -/
  grp_inj : ∀ g h, g < d.n → h < d.n → d.groupId g = d.groupId h → g = h
  /-- The groups' ids increase (`torch.unique(…, sorted=True)`). -/
  grp_incr : ∀ g h, h < g → g < d.n → d.groupId h < d.groupId g
  /-- Every token's adapter is a group's id. -/
  grp_cover : ∀ t, t < d.M → ∃ g, g < d.n ∧ d.adapter t = d.groupId g
  /-- `num_tokens_per_lora[g]` is the number of tokens of group `g`. -/
  cnt_def : ∀ g, g < d.n →
    d.groupCnt g = ((Finset.range d.M).filter (fun t => d.adapter t = d.groupId g)).card
  /-- `lora_token_start_loc[0] = 0`. -/
  start_zero : d.groupStart 0 = 0
  /-- `lora_token_start_loc[g + 1] = lora_token_start_loc[g] + num_tokens_per_lora[g]`. -/
  start_succ : ∀ g, d.groupStart (g + 1) = d.groupStart g + d.groupCnt g
  /-- Position `p` of group `g`'s range holds a token of group `g` (the stable sort). -/
  tid_range : ∀ g, g < d.n → ∀ p, d.groupStart g ≤ p → p < d.groupStart g + d.groupCnt g →
    p < d.M ∧ d.tIdx p < d.M ∧ d.adapter (d.tIdx p) = d.groupId g
  /-- `token_indices_sorted_by_lora_ids` is a permutation (injective on `[0, M)`). -/
  tid_inj : ∀ p q, p < d.M → q < d.M → d.tIdx p = d.tIdx q → p = q
  /-- Every token of group `g` occurs in group `g`'s range. -/
  tid_cover : ∀ t, t < d.M → ∀ g, g < d.n → d.adapter t = d.groupId g →
    ∃ p, d.groupStart g ≤ p ∧ p < d.groupStart g + d.groupCnt g ∧ d.tIdx p = t
  /-- `inputs` holds `[M, K]`. -/
  inp_len : ∀ t k, t < d.M → k < d.K → t * d.K + k < d.inp.len
  /-- Each slice's weights hold `[maxL, R, K]`. -/
  lora_len : ∀ s l j k, s < d.S → l < d.maxL → j < d.R → k < d.K →
    l * (d.R * d.K) + j * d.K + k < (d.lora s).len
  /-- With several slices, the address table holds an entry per slice. -/
  tab_len : 1 < d.S → ∀ s, s < d.S → s < d.tab.len
  /-- The output holds `[S, M, R]`. -/
  out_len : ∀ s t j, s < d.S → t < d.M → j < d.R → outAt d s t j < d.out.len
  /-- `token_indices_sorted_by_lora_ids` holds `M` entries. -/
  tid_len : ∀ p, p < d.M → p < d.tIdxBuf.len
  /-- `num_tokens_per_lora` holds an entry per group of the grid. -/
  cnt_len : ∀ g, g < d.G → g < d.cntBuf.len
  /-- `lora_token_start_loc` holds an entry per group of the grid. -/
  start_len : ∀ g, g < d.G → g < d.startBuf.len
  /-- `lora_ids` holds an entry per group of the grid. -/
  id_len : ∀ g, g < d.G → g < d.idBuf.len
  /-- The output shares no byte with `inputs` (the host allocates it fresh). -/
  out_inp : d.out.Disjoint' d.inp
  /-- The output shares no byte with any slice's weights. -/
  out_lora : ∀ s, s < d.S → d.out.Disjoint' (d.lora s)
  /-- With several slices, the output shares no byte with the address table. -/
  out_tab : 1 < d.S → d.out.Disjoint' d.tab
  /-- The output shares no byte with `token_indices_sorted_by_lora_ids`. -/
  out_tid : d.out.Disjoint' d.tIdxBuf
  /-- The output shares no byte with `num_tokens_per_lora`. -/
  out_cnt : d.out.Disjoint' d.cntBuf
  /-- The output shares no byte with `lora_token_start_loc`. -/
  out_start : d.out.Disjoint' d.startBuf
  /-- The output shares no byte with `lora_ids`. -/
  out_id : d.out.Disjoint' d.idBuf

/-- The host's `EVEN_K` (line 562): `K % (BLOCK_K · SPLIT_K) == 0`. -/
def evenK (d : Inst) : Bool := decide (d.K % (bk d * sk d) = 0)

/-- The model on an instance: the buffers' first elements (`lora_ptr` the first slice's weights,
`lora_tab` the address table, read only when `S > 1`), `M`, `R` (`N`), `K`, the metadata
pointers, `scaling`'s `float32` rounding (the kernel's argument), the strides of the host's
contiguous layouts (lines 584–591: the input's `(K, 1)`, the weights' `(R·K, K, 1)` after the
squeeze, the output's `M·R`, `R` and `1`), and the constexprs of lines 553–562 (`BLOCK_M = 32`,
`BLOCK_N = 16`, `BLOCK_K = bk d`, `SPLIT_K = sk d`, `GROUP_SIZE_M = 8`, `SLICE_NUM = S`, `EVEN_K`).
`output_d0_stride = M·R` is read only when `S > 1`, where `slice_wrap` makes it an `int32`. -/
def model (d : Inst) : Kernel :=
  loraShrink (d.inp.ptr 0) ((d.lora 0).ptr 0) (d.tab.ptr 0) (d.out.ptr 0)
    (IntV.ofNat d.M) (IntV.ofNat d.R) (IntV.ofNat d.K)
    (d.tIdxBuf.ptr 0) (d.cntBuf.ptr 0) (d.startBuf.ptr 0) (d.idBuf.ptr 0) (scaling32 d)
    (IntV.ofNat d.K) (IntV.ofNat 1) (IntV.ofNat (d.R * d.K)) (IntV.ofNat d.K) (IntV.ofNat 1)
    (IntV.ofNat (d.M * d.R)) (IntV.ofNat d.R) (IntV.ofNat 1) 32 16 (bk d) (sk d) 8 d.S (evenK d)

/-- The host's launch (lines 564–608): the grid `(SPLIT_K · cdiv(M, 32) · cdiv(R, 16), S,
num_active_loras)`; the source's block shapes; `enable_fp_fusion` at Triton's default (`true`);
`num_ctas = 1`, `launch_pdl = False`, `debug = false`. -/
def launch (d : Inst) : Launch :=
  { grid := ![grid0 d, d.S, d.G],
    blocks := [[32], [16], [bk d], [32, 16], [32, bk d], [bk d, 16], [32, bk d], [bk d, 16]] }

/-- The inputs covered: the buffers valid (with several slices, the address table too, holding each
slice's weight address, as `_get_lora_a_ptr` writes it); the metadata buffers holding the
instance's metadata as `lora_kernel_meta` writes it (`lora_ids`: the distinct ids then `-1`;
`num_tokens_per_lora`: the counts then `0`; `lora_token_start_loc`: the cumulative sums then `0`;
`token_indices_sorted_by_lora_ids`: `tIdx`); the output's `[S, M, R]` tensor `+0.0` (the host's
`torch.zeros`). Any `bfloat16` values in `inputs` and the weights, NaN and infinities included. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.inp.Valid init ∧ (∀ s, s < d.S → (d.lora s).Valid init) ∧ d.out.Valid init ∧
    d.tIdxBuf.Valid init ∧ d.cntBuf.Valid init ∧ d.startBuf.Valid init ∧ d.idBuf.Valid init ∧
    (1 < d.S → d.tab.Valid init ∧
      ∀ s, s < d.S → init.read .u64 (d.tab.ptr s).addr = some (IntV.ofInt (d.lora s).base)) ∧
    d.cntBuf.Holds init (fun g => IntV.ofNat (cntAt d g)) ∧
    d.startBuf.Holds init (fun g => IntV.ofNat (startAt d g)) ∧
    d.idBuf.Holds init (fun g => IntV.ofInt (idAt d g)) ∧
    (∀ p, p < d.M → init.read .i32 (d.tIdxBuf.ptr p).addr = some (IntV.ofNat (d.tIdx p))) ∧
    (∀ s t j, s < d.S → t < d.M → j < d.R → init.read .f32 (d.out.ptr (outAt d s t j)).addr =
      some (FloatV.ofPat 0))

/-! ## The specification -/

/-- The `tl.dot` steps of the `SPLIT_K` programs of a tile together: `cdiv(K, BLOCK_K)`, one per
block of `BLOCK_K` consecutive `k` below `K` (block `b` is step `b / SPLIT_K` of the program of part
`b % SPLIT_K`; a block from `K` on is skipped, lines 155–157). -/
def blocks (d : Inst) : Nat := (d.K + bk d - 1) / bk d

/-- The `tl.dot` steps of the program of part `p < SPLIT_K`: the blocks `b < blocks` with
`b % SPLIT_K = p`. -/
def stepsOf (d : Inst) (p : Nat) : Nat := (blocks d + sk d - 1 - p) / sk d

/-- The width of a chunk of `Rules.Num.DotBF16Chunk` in one `tl.dot` of `K`-extent `BLOCK_K`: `64`
consecutive `k`, or all `BLOCK_K` when `BLOCK_K < 64` (`Rules.Num.chunk16`). -/
def cw (d : Inst) : Nat := min 64 (bk d)

/-- The chunks per `tl.dot`: `BLOCK_K / cw` (`chunks16 BLOCK_K`: `4` at `256`, `1` at `32`). -/
def cpb (d : Inst) : Nat := bk d / cw d

/-- The chunks of the program of part `p`, in the order it adds them: `cpb · stepsOf p`. -/
def nChunks (d : Inst) (p : Nat) : Nat := cpb d * stepsOf d p

/-- Chunk `q` of the program of part `p`: the `k < K` of chunk `q % cpb` of its step `q / cpb`,
whose block is `p + SPLIT_K · (q / cpb)`. -/
def chunkK (d : Inst) (p q : Nat) : Finset Nat :=
  (Finset.range d.K).filter fun k =>
    k / bk d = p + sk d * (q / cpb d) ∧ k % bk d / cw d = q % cpb d

/-- The exact sum of chunk `q` of part `p`. -/
def cSum (d : Inst) (va vb : Nat → BF16) (p q : Nat) : ℚ :=
  ∑ k ∈ chunkK d p q, (va k).toRat * (vb k).toRat

/-- The largest lifted product of chunk `q` of part `p` (`0` for an empty chunk; a `bfloat16`
subnormal counted at `2^-126`), the `Mj` bound `DotBF16Chunk` asks for on the products. -/
def cLift (d : Inst) (va vb : Nat → BF16) (p q : Nat) : ℚ :=
  (chunkK d p q).fold max 0 fun k =>
    Rules.Num.liftMag .bf16 (va k).toRat * Rules.Num.liftMag .bf16 (vb k).toRat

/-- The exact part of the dot product the program of part `p` accumulates. -/
def partSum (d : Inst) (va vb : Nat → BF16) (p : Nat) : ℚ :=
  ∑ q ∈ Finset.range (nChunks d p), cSum d va vb p q

/-- `Rules.Num.DotBF16`'s relative constant for one step of `K`-extent `BLOCK_K`, through
`DotLift.dotAdd_step` (`DotLoop.stepRel relDot16 BLOCK_K`). -/
def rho (d : Inst) : ℚ := DotLoop.stepRel Rules.Num.relDot16 (bk d)

/-- The absolute constant of the same step (`DotLoop.stepAbs absDot16 BLOCK_K`). -/
def alpha (d : Inst) : ℚ := DotLoop.stepAbs Rules.Num.absDot16 (bk d)

/-- The lifted magnitude of part `p`: `Σ liftMag a · liftMag b` over its `k`. -/
def partLift (d : Inst) (va vb : Nat → BF16) (p : Nat) : ℚ :=
  ∑ q ∈ Finset.range (nChunks d p), ∑ k ∈ chunkK d p q,
    Rules.Num.liftMag .bf16 (va k).toRat * Rules.Num.liftMag .bf16 (vb k).toRat

/-- The exact partial sum of the first `g` chunks of part `p` (`pSum p (nChunks p) = partSum p`). -/
def pSum (d : Inst) (va vb : Nat → BF16) (p g : Nat) : ℚ :=
  ∑ q ∈ Finset.range g, cSum d va vb p q

/-- One chunk step of `Rules.Num.DotBF16Chunk` (its first arm, `2^-13`, `2^-20`, `2^-143`), from a
partial value within `e` of the exact partial sum `Pg` to the next, against the next exact partial
sum `Pg'`, `mg` the chunk's largest lifted product: the chunk's loss is at most
`2^-13·Mj + 2^-20·|s j + c_j| + 2^-143` with `Mj ≤ max |Pg| mg + e` and `|s j + c_j| ≤ |Pg'| + e`
(the recursion of `Dot8Chunk.errB`). -/
def cStep (e Pg mg Pg' : ℚ) : ℚ :=
  (1 + Rules.Num.rhoChunk16 + Rules.Num.sigmaChunk16) * e + Rules.Num.rhoChunk16 * max |Pg| mg +
    Rules.Num.sigmaChunk16 * |Pg'| + Rules.Num.alphaChunk16

/-- `n` chunk steps of part `p` from its chunk `g0`, starting within `e`, against the partial sums
`P g` (the part's exact partial sums, or one `tl.dot`'s own from `0`). -/
def cChain (d : Inst) (va vb : Nat → BF16) (p g0 : Nat) (P : Nat → ℚ) (e : ℚ) : Nat → ℚ
  | 0 => e
  | n + 1 => cStep (cChain d va vb p g0 P e n) (P (g0 + n)) (cLift d va vb p (g0 + n))
      (P (g0 + n + 1))

/-- The error after step `t + 1` of part `p`'s loop (line 152, `accumulator += tl.dot(...)`, a
`tl.dot_add` of `BLOCK_K` products, `cpb` chunks), from the error `e` after step `t`, under either
form the library gives it:
* `tl.dot(a, b, acc)`: the step's `cpb` chunk steps continue the accumulator's chain (`cChain`
  from `e` against the part's exact partial sums);
* `tl.dot(a, b)` then the `float32` addition (`AccAdd`, fusable): the product's own chain from `0`
  (within `L` of the step's exact sum `D`), then `FloatV.AddRel.err`: `3u·(|d| + |acc|) + 4η` with
  `|d| ≤ |D| + L` and `|acc| ≤ |P| + e`.
The larger of the two covers both. -/
def tripErr (d : Inst) (va vb : Nat → BF16) (p t : Nat) (e : ℚ) : ℚ :=
  let q := cpb d
  let P := pSum d va vb p
  let L := cChain d va vb p (t * q) (fun g => P g - P (t * q)) 0 q
  max (cChain d va vb p (t * q) P e q)
    (e + L + 3 * Math.Float.Format.binary32.u *
      (|P (t * q + q) - P (t * q)| + L + |P (t * q)| + e) + 4 * Math.Float.Format.binary32.eta)

/-- The error after `t` steps of part `p`, from the exact `+0.0` of `tl.zeros` (line 128). -/
def accErr (d : Inst) (va vb : Nat → BF16) (p : Nat) : Nat → ℚ
  | 0 => 0
  | t + 1 => tripErr d va vb p t (accErr d va vb p t)

/-- The chunk-by-chunk bound on the accumulator of part `p` against `partSum p`
(`Rules.Num.DotBF16Chunk`): the error after the part's `stepsOf p` steps. -/
def errChunk (d : Inst) (va vb : Nat → BF16) (p : Nat) : ℚ :=
  accErr d va vb p (stepsOf d p)

/-- The step-by-step bound on the same accumulator: `DotLoop.bound ρ α (stepsOf p) (partLift p)`
(`DotBF16` per `tl.dot` step through `DotLift.dotAdd_step`, composed by `DotLoop.accum_step`). -/
def errStep (d : Inst) (va vb : Nat → BF16) (p : Nat) : ℚ :=
  DotLoop.bound (rho d) (alpha d) (stepsOf d p) (partLift d va vb p)

/-- The bound on the accumulator of part `p`: the smaller of the two (both hold). -/
def errPart (d : Inst) (va vb : Nat → BF16) (p : Nat) : ℚ :=
  min (errChunk d va vb p) (errStep d va vb p)

/-- The exact dot product of row values `va` and weight-row values `vb` over `k < K`. -/
def dotSum (d : Inst) (va vb : Nat → BF16) : ℚ :=
  ∑ k ∈ Finset.range d.K, (va k).toRat * (vb k).toRat

/-- `Σ liftMag a · liftMag b` (a `bfloat16` subnormal counted at `2^-126`): the room the `tl.dot`s
need is stated on it. -/
def dotMag (d : Inst) (va vb : Nat → BF16) : ℚ :=
  ∑ k ∈ Finset.range d.K,
    Rules.Num.liftMag .bf16 (va k).toRat * Rules.Num.liftMag .bf16 (vb k).toRat

/-- The bound on the `SPLIT_K` accumulators' errors together: `Σ_{p < SPLIT_K} errPart p`. -/
def accTol (d : Inst) (va vb : Nat → BF16) : ℚ :=
  ∑ p ∈ Finset.range (sk d), errPart d va vb p

/-- A bound on the accumulators' magnitudes together: `Σ_{p < SPLIT_K} (|partSum p| + errPart p)`. -/
def partMag (d : Inst) (va vb : Nat → BF16) : ℚ :=
  ∑ p ∈ Finset.range (sk d), (|partSum d va vb p| + errPart d va vb p)

/-- The roundings of the `SPLIT_K` products by the kernel's `float32` `scaling` (`s32`), one
`float32` rounding each (`u = 2^-24`, `η = 2^-150`): `u · |s32| · partMag + SPLIT_K · η`. -/
def mulTol (d : Inst) (va vb : Nat → BF16) : ℚ :=
  Math.Float.Format.binary32.u * |(scaling32 d).toRat| * partMag d va vb +
    sk d * Math.Float.Format.binary32.eta

/-- A bound on the sum of the magnitudes of the `SPLIT_K` scaled parts the atomic additions add:
`|s32| · partMag + mulTol`. -/
def vMag (d : Inst) (va vb : Nat → BF16) : ℚ :=
  |(scaling32 d).toRat| * partMag d va vb + mulTol d va vb

/-- The `SPLIT_K` `float32` atomic additions into the `+0.0` element, in any order
(`Rules.Num.AtomicAddF32`'s `2^-23`, `2^-124`, through `SharedAcc.replay_err_shared`):
`DotLoop.bound 2^-23 2^-124 SPLIT_K vMag`. -/
def atomTol (d : Inst) (va vb : Nat → BF16) : ℚ :=
  DotLoop.bound (2 ^ (-23 : ℤ)) (2 ^ (-124 : ℤ)) (sk d) (vMag d va vb)

/-- The tolerance of an output element against the host's `scaling · S` (`scaling` the host's
double, `S = dotSum`, `s32 = scaling32`): the accumulators' bound scaled by `|s32|`, the products'
roundings, the atomic additions, and the conversion of the host's double to the kernel's `float32`
argument, `|s32 − scaling| · |S|` (`0` for `0.5` and `1.0`; at most `(2^-24·|scaling| + 2^-150)·|S|`
for any double in `float32`'s range, one rounding to nearest). -/
def tol (d : Inst) (va vb : Nat → BF16) : ℚ :=
  |(scaling32 d).toRat| * accTol d va vb + mulTol d va vb + atomTol d va vb +
    |(scaling32 d).toRat - d.scaling.toRat| * |dotSum d va vb|

/-- What the launch leaves in element `(s, t, j)` of the output. A token with no adapter keeps the
`+0.0` the output was allocated with. A token with an adapter `l = adapter t ≥ 0`: when the `K`
values of row `t` of `inputs` and of row `j` of adapter `l`'s weights of slice `s` are finite,
`dotMag ≤ 2^120`, and there is room in `float32` (`vMag + atomTol ≤ maxVal`), the element is
finite and within `tol` of `scaling · S`, `scaling` the host's double, whatever order the atomic
additions ran in. -/
def OutSpec (d : Inst) (init fin : Mem) (s t j : Nat) : Prop :=
  ∃ v : F32, fin.read .f32 (d.out.ptr (outAt d s t j)).addr = some v ∧
    ((d.adapter t < 0 ∧ v = FloatV.ofPat 0) ∨
     (0 ≤ d.adapter t ∧ ∀ va vb : Nat → BF16,
      (∀ k, k < d.K → init.read .bf16 (d.inp.ptr (t * d.K + k)).addr = some (va k) ∧
        (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .bf16
          ((d.lora s).ptr ((d.adapter t).toNat * (d.R * d.K) + j * d.K + k)).addr = some (vb k) ∧
        (vb k).isFinite = true) →
      dotMag d va vb ≤ 2 ^ 120 →
      vMag d va vb + atomTol d va vb ≤ Math.Float.Format.binary32.maxVal →
      v.isFinite = true ∧ |v.toRat - d.scaling.toRat * dotSum d va vb| ≤ tol d va vb))

/-- What the launch leaves: every element of the output's `[S, M, R]` tensor as `OutSpec` says;
every byte outside it unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ s t j, s < d.S → t < d.M → j < d.R → OutSpec d init fin s t j) ∧
    ∀ z, z ∉ d.out.elems (outIdx d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the model's outcomes. The launches are within
  part (A) of its scope (README, Conditions); part (B), the environment, is relied on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotBF16`: `tl.dot` of `bfloat16` operands into `float32`
  (`maxImpreciseAcc = 0`, any `input_precision`), the model's dots: within
  `relDot16 K·L + absDot16 K` of `acc + Σ a·b`, `L = |acc| + Σ liftMag a·liftMag b`. -/
  dot : Rules.Num.DotBF16 P
  /-- [rule] `Num.DotBF16Chunk`: `tl.dot` of `bfloat16` operands into `float32`
  (`maxImpreciseAcc = 0`, any `input_precision`), the model's dots, chunk by chunk: partial values
  from the accumulator to the result, one step per chunk of 64 consecutive `k` (all `K` when
  `K < 64`) in increasing order, each within `min(2^-13·Mj, 2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| +
  2^-143` of the previous value plus the chunk's exact sum (`Rules.Num.DotChunk16Loss`). -/
  dotChunk : Rules.Num.DotBF16Chunk P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` always has a result. -/
  dot_total : Rules.Num.DotTotal P
    { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
  /-- [rule] `Num.AtomicAddF32`: `tl.atomic_add` on `float32` (line 305; `Rules.Num.AtomicAddAcc`
  with `2^-23`, `2^-124`): when the old value and the operand are finite and their exact sum is in
  range, the new value is finite and within `2^-23·|o + v| + 2^-124` of the sum. -/
  atomic : Rules.Num.AtomicAddF32 P
  /-- [rule] `Num.AtomicAddTotal`: a `float32` `tl.atomic_add` (line 305) always has a result. -/
  atomic_total : Rules.Num.AtomicAddTotal P Math.Float.Format.binary32

/-- The claim to prove. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous: the assumptions are consistent -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot,
    ⟨fun d _ => hf d, Platform.sm90ref.dotBF16, Platform.sm90ref.dotBF16Chunk,
      Platform.ref.dotTotal _,
      Platform.sm90ref.atomicAddF32, Platform.ref.atomicAddTotal _⟩⟩

/-! ## Coverage: the host's launches

Two families of instances with `SETUP.md`'s shapes. The host's test draws its mapping at random;
these fix one mapping of the same shape: `oneInst` maps every token to adapter `a`; `twoInst` maps
the first `nm` tokens to `-1` (no adapter) and the rest to `a`, so `token_indices_sorted_by_lora_ids`
is the identity in both (`-1` sorts first). The buffers lie at multiples of `2^40` with the host's
lengths: `inputs` at `1`, the first slice's weights at `2`, the output at `3`, the four metadata
tensors at `4` to `7`, the address table at `8`, and slice `s ≥ 1`'s weights at `8 + s`. -/

/-- The position (in units of `2^40` bytes) of slice `s`'s weights. -/
def loraAt (s : Nat) : Nat := if s = 0 then 2 else 8 + s

/-- The buffers of an instance: `inputs`, the slices' weights, the address table, the output, and
the four metadata tensors, with the host's lengths. -/
structure Lay where
  inp : Buf .bf16
  lora : Nat → Buf .bf16
  tab : Buf .u64
  out : Buf .f32
  tIdxBuf : Buf .i32
  cntBuf : Buf .i32
  startBuf : Buf .i32
  idBuf : Buf .i32

/-- The layout of `M` tokens, hidden `K`, rank `R`, `maxL` adapters, `S` slices. -/
def lay (M K R maxL S : Nat) : Lay where
  inp := ⟨2 ^ 40, M * K⟩
  lora := fun s => ⟨(loraAt s : Int) * 2 ^ 40, maxL * (R * K)⟩
  tab := ⟨8 * 2 ^ 40, S⟩
  out := ⟨3 * 2 ^ 40, S * (M * R)⟩
  tIdxBuf := ⟨4 * 2 ^ 40, M⟩
  cntBuf := ⟨5 * 2 ^ 40, maxL + 1⟩
  startBuf := ⟨6 * 2 ^ 40, maxL + 2⟩
  idBuf := ⟨7 * 2 ^ 40, maxL + 1⟩

/-- Every token on adapter `a`. -/
def oneInst (M K R maxL S G a : Nat) (sc : F64) : Inst where
  M := M
  K := K
  R := R
  maxL := maxL
  S := S
  n := 1
  G := G
  adapter := fun _ => (a : Int)
  groupId := fun _ => (a : Int)
  groupCnt := fun g => if g = 0 then M else 0
  groupStart := fun g => if g = 0 then 0 else M
  tIdx := fun p => p
  scaling := sc
  inp := (lay M K R maxL S).inp
  lora := (lay M K R maxL S).lora
  tab := (lay M K R maxL S).tab
  out := (lay M K R maxL S).out
  tIdxBuf := (lay M K R maxL S).tIdxBuf
  cntBuf := (lay M K R maxL S).cntBuf
  startBuf := (lay M K R maxL S).startBuf
  idBuf := (lay M K R maxL S).idBuf

/-- The first `nm` tokens without an adapter, the others on adapter `a`. -/
def twoInst (M K R maxL S G nm a : Nat) (sc : F64) : Inst where
  M := M
  K := K
  R := R
  maxL := maxL
  S := S
  n := 2
  G := G
  adapter := fun t => if t < nm then -1 else (a : Int)
  groupId := fun g => if g = 0 then -1 else (a : Int)
  groupCnt := fun g => if g = 0 then nm else if g = 1 then M - nm else 0
  groupStart := fun g => if g = 0 then 0 else if g = 1 then nm else M
  tIdx := fun p => p
  scaling := sc
  inp := (lay M K R maxL S).inp
  lora := (lay M K R maxL S).lora
  tab := (lay M K R maxL S).tab
  out := (lay M K R maxL S).out
  tIdxBuf := (lay M K R maxL S).tIdxBuf
  cntBuf := (lay M K R maxL S).cntBuf
  startBuf := (lay M K R maxL S).startBuf
  idBuf := (lay M K R maxL S).idBuf

/-- The sizes of a layout that keep its buffers apart (each below `2^40` bytes). -/
structure LayFits (M K R maxL S : Nat) : Prop where
  inp : M * K * 2 ≤ 2 ^ 40
  lora : maxL * (R * K) * 2 ≤ 2 ^ 40
  out : S * (M * R) * 4 ≤ 2 ^ 40
  tid : M * 4 ≤ 2 ^ 40
  metaB : (maxL + 2) * 4 ≤ 2 ^ 40

theorem lay_disj {d d' : DType} {b : Buf d} {c : Buf d'} (i j : Nat) (hij : i < j)
    (hb : b.base = i * 2 ^ 40) (hc : c.base = j * 2 ^ 40) (hlen : b.len * d.bytes ≤ 2 ^ 40) :
    b.Disjoint' c := by
  unfold Buf.Disjoint'
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  have hl : ((b.len * d.bytes : Nat) : Int) ≤ 2 ^ 40 := by exact_mod_cast hlen
  push_cast at hl
  have hij' : (i : Int) + 1 ≤ j := by exact_mod_cast hij
  rw [hb] at hx
  rw [hc] at hy
  nlinarith

theorem lay_disj' {d d' : DType} {b : Buf d} {c : Buf d'} (i j : Nat) (hij : j < i)
    (hb : b.base = i * 2 ^ 40) (hc : c.base = j * 2 ^ 40) (hlen : c.len * d'.bytes ≤ 2 ^ 40) :
    b.Disjoint' c := by
  unfold Buf.Disjoint'
  exact (lay_disj (b := c) (c := b) j i hij hc hb hlen).symm

/-- The output (at `3`) and slice `s`'s weights (at `loraAt s`) do not overlap. -/
theorem out_lora_disj {M K R maxL S : Nat} (hf : LayFits M K R maxL S) (s : Nat) :
    (lay M K R maxL S).out.Disjoint' ((lay M K R maxL S).lora s) := by
  by_cases hs : s = 0
  · subst hs
    exact lay_disj' 3 2 (by norm_num) rfl (by simp [lay, loraAt])
      (by simpa [DType.bytes, lay] using hf.lora)
  · exact lay_disj 3 (8 + s) (by omega) rfl (by simp [lay, loraAt, hs])
      (by simpa [DType.bytes, lay] using hf.out)

theorem idx_mul_add_lt {t k M K : Nat} (ht : t < M) (hk : k < K) : t * K + k < M * K := by
  have h : (t + 1) * K ≤ M * K := Nat.mul_le_mul_right K ht
  rw [Nat.succ_mul] at h
  omega

theorem lora_idx_lt {l j k maxL R K : Nat} (hl : l < maxL) (hj : j < R) (hk : k < K) :
    l * (R * K) + j * K + k < maxL * (R * K) := by
  have h1 : j * K + k < R * K := idx_mul_add_lt hj hk
  have h2 : (l + 1) * (R * K) ≤ maxL * (R * K) := Nat.mul_le_mul_right _ hl
  rw [Nat.succ_mul] at h2
  omega

theorem out_idx_lt {s t j S M R : Nat} (hs : s < S) (ht : t < M) (hj : j < R) :
    s * (M * R) + t * R + j < S * (M * R) := by
  have h1 : t * R + j < M * R := idx_mul_add_lt ht hj
  have h2 : (s + 1) * (M * R) ≤ S * (M * R) := Nat.mul_le_mul_right _ hs
  rw [Nat.succ_mul] at h2
  omega

/-- The shapes' conditions common to both families. -/
structure CovShape (M K R maxL S G : Nat) (sc : F64) : Prop where
  R_blk : R % 16 = 0
  R_pos : 0 < R
  K_pos : 0 < K
  K_top : K ≤ 2 ^ 17
  RK_lt : R * K < 2 ^ 31
  out_wrap : (M - 1) * R < 2 ^ 31
  S_pos : 0 < S
  slice_wrap : 1 < S → (S - 1) * (M * R) < 2 ^ 31
  grid0 : (if M < 128 then 64 else 8) * ((M + 31) / 32) * ((R + 15) / 16) < 2 ^ 31
  S_lt : S < 2 ^ 16
  G_le : G ≤ maxL + 1
  G_lt : G < 2 ^ 16
  grid_size : (if M < 128 then 64 else 8) * ((M + 31) / 32) * ((R + 15) / 16) * S * G < 2 ^ 31
  sc_fin : sc.isFinite = true ∧ (FloatV.hostF32 sc).isFinite = true
  fits : LayFits M K R maxL S

theorem oneInst_ok {M K R maxL S G a : Nat} {sc : F64} (hs : CovShape M K R maxL S G sc)
    (hM : 0 < M) (hG : 1 ≤ G) (ha : a < maxL) (hw : (a : Int) * ((R * K : Nat) : Int) < 2 ^ 31) :
    (oneInst M K R maxL S G a sc).ok where
  R_blk := hs.R_blk
  R_pos := hs.R_pos
  K_pos := hs.K_pos
  K_top := hs.K_top
  RK_lt := hs.RK_lt
  out_wrap := hs.out_wrap
  lora_wrap := fun _ _ _ => hw
  S_pos := hs.S_pos
  slice_wrap := hs.slice_wrap
  grid0_lt := hs.grid0
  S_lt := hs.S_lt
  n_le_G := hG
  G_le := hs.G_le
  G_lt := hs.G_lt
  grid_size := hs.grid_size
  sc_fin := hs.sc_fin
  ad_range := fun _ _ => ⟨by show (-1 : Int) ≤ (a : Int); omega,
    by show (a : Int) < (maxL : Int); omega⟩
  grp_nonempty := fun _ _ => ⟨0, hM, rfl⟩
  grp_inj := fun g h hg hh _ => by simp only [oneInst] at hg hh; omega
  grp_incr := fun g h hhg hg => by simp only [oneInst] at hg; omega
  grp_cover := fun _ _ => ⟨0, by simp [oneInst], rfl⟩
  cnt_def := fun g hg => by
    have hg0 : g = 0 := by simp only [oneInst] at hg; omega
    subst hg0
    simp [oneInst]
  start_zero := rfl
  start_succ := fun g => by
    by_cases hg : g = 0
    · subst hg; simp [oneInst]
    · simp [oneInst, hg]
  tid_range := fun g hg p h1 h2 => by
    have hg0 : g = 0 := by simp only [oneInst] at hg; omega
    subst hg0
    have h2' : p < M := by simpa [oneInst] using h2
    exact ⟨h2', h2', rfl⟩
  tid_inj := fun _ _ _ _ h => h
  tid_cover := fun t ht g hg _ => by
    have hg0 : g = 0 := by simp only [oneInst] at hg; omega
    subst hg0
    exact ⟨t, by simp [oneInst], by simpa [oneInst] using ht, rfl⟩
  inp_len := fun _ _ ht hk => idx_mul_add_lt ht hk
  lora_len := fun _ _ _ _ _ hl hj hk => lora_idx_lt hl hj hk
  tab_len := fun _ _ hs' => hs'
  out_len := fun _ _ _ hs' ht hj => out_idx_lt hs' ht hj
  tid_len := fun _ hp => hp
  cnt_len := fun g hg => by simp only [oneInst, lay] at hg ⊢; have := hs.G_le; omega
  start_len := fun g hg => by simp only [oneInst, lay] at hg ⊢; have := hs.G_le; omega
  id_len := fun g hg => by simp only [oneInst, lay] at hg ⊢; have := hs.G_le; omega
  out_inp := lay_disj' 3 1 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.inp)
  out_lora := fun s _ => out_lora_disj hs.fits s
  out_tab := fun _ => lay_disj 3 8 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_tid := lay_disj 3 4 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_cnt := lay_disj 3 5 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_start := lay_disj 3 6 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_id := lay_disj 3 7 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)

theorem filter_neg_one {M nm a : Nat} (h : nm ≤ M) :
    ((Finset.range M).filter (fun t => (if t < nm then (-1 : Int) else (a : Int)) = -1)) =
      Finset.range nm := by
  ext t
  simp only [Finset.mem_filter, Finset.mem_range]
  by_cases hc : t < nm
  · simp only [hc, ite_true, and_true, iff_true]; omega
  · simp only [hc, ite_false, iff_false]; omega

theorem filter_a {M nm a : Nat} :
    ((Finset.range M).filter (fun t => (if t < nm then (-1 : Int) else (a : Int)) = (a : Int))) =
      Finset.Ico nm M := by
  ext t
  simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_Ico]
  by_cases hc : t < nm
  · simp only [hc, ite_true]; omega
  · simp only [hc, ite_false, and_true]; omega

theorem twoInst_ok {M K R maxL S G nm a : Nat} {sc : F64} (hs : CovShape M K R maxL S G sc)
    (hnm : 0 < nm) (hnmM : nm < M) (hG : 2 ≤ G) (ha : a < maxL)
    (hw : (a : Int) * ((R * K : Nat) : Int) < 2 ^ 31) :
    (twoInst M K R maxL S G nm a sc).ok where
  R_blk := hs.R_blk
  R_pos := hs.R_pos
  K_pos := hs.K_pos
  K_top := hs.K_top
  RK_lt := hs.RK_lt
  out_wrap := hs.out_wrap
  lora_wrap := fun t _ h0 => by
    show (if t < nm then (-1 : Int) else (a : Int)) * ((R * K : Nat) : Int) < 2 ^ 31
    have h0' : 0 ≤ (if t < nm then (-1 : Int) else (a : Int)) := h0
    split_ifs at h0' ⊢ with hc
    · omega
    · exact hw
  S_pos := hs.S_pos
  slice_wrap := hs.slice_wrap
  grid0_lt := hs.grid0
  S_lt := hs.S_lt
  n_le_G := hG
  G_le := hs.G_le
  G_lt := hs.G_lt
  grid_size := hs.grid_size
  sc_fin := hs.sc_fin
  ad_range := fun t _ => by
    show -1 ≤ (if t < nm then (-1 : Int) else (a : Int)) ∧
      (if t < nm then (-1 : Int) else (a : Int)) < (maxL : Int)
    split_ifs <;> omega
  grp_nonempty := fun g hg => by
    have hg2 : g < 2 := hg
    rcases (by omega : g = 0 ∨ g = 1) with rfl | rfl
    · exact ⟨0, by show 0 < M; omega, by simp [twoInst, hnm]⟩
    · exact ⟨nm, hnmM, by simp [twoInst]⟩
  grp_inj := fun g h hg hh he => by
    have hg2 : g < 2 := hg
    have hh2 : h < 2 := hh
    rcases (by omega : g = 0 ∨ g = 1) with rfl | rfl <;>
      rcases (by omega : h = 0 ∨ h = 1) with rfl | rfl <;> simp [twoInst] at he ⊢
  grp_incr := fun g h hhg hg => by
    have hg2 : g < 2 := hg
    obtain ⟨rfl, rfl⟩ : g = 1 ∧ h = 0 := by omega
    simp [twoInst]
    omega
  grp_cover := fun t _ => by
    by_cases hc : t < nm
    · exact ⟨0, by show 0 < 2; omega, by simp [twoInst, hc]⟩
    · exact ⟨1, by show 1 < 2; omega, by simp [twoInst, hc]⟩
  cnt_def := fun g hg => by
    have hg2 : g < 2 := hg
    rcases (by omega : g = 0 ∨ g = 1) with rfl | rfl
    · show nm = ((Finset.range M).filter
        (fun t => (if t < nm then (-1 : Int) else (a : Int)) = -1)).card
      rw [filter_neg_one hnmM.le, Finset.card_range]
    · show M - nm = ((Finset.range M).filter
        (fun t => (if t < nm then (-1 : Int) else (a : Int)) = (a : Int))).card
      rw [filter_a, Nat.card_Ico]
  start_zero := rfl
  start_succ := fun g => by
    rcases (by omega : g = 0 ∨ g = 1 ∨ 2 ≤ g) with rfl | rfl | h2
    · simp [twoInst]
    · simp [twoInst]; omega
    · have h0 : g ≠ 0 := by omega
      have h1 : g ≠ 1 := by omega
      simp [twoInst, h0, h1]
  tid_range := fun g hg p h1 h2 => by
    have hg2 : g < 2 := hg
    rcases (by omega : g = 0 ∨ g = 1) with rfl | rfl
    · have h2' : p < nm := by simpa [twoInst] using h2
      exact ⟨by show p < M; omega, by show p < M; omega, by simp [twoInst, h2']⟩
    · have h1' : nm ≤ p := by simpa [twoInst] using h1
      have h2' : p < M := by simp [twoInst] at h2; omega
      exact ⟨h2', h2', by simp [twoInst]; omega⟩
  tid_inj := fun _ _ _ _ h => h
  tid_cover := fun t ht g hg he => by
    have hg2 : g < 2 := hg
    have ht' : t < M := ht
    rcases (by omega : g = 0 ∨ g = 1) with rfl | rfl
    · have hc : t < nm := by
        by_contra hc
        simp [twoInst, hc] at he
      exact ⟨t, by simp [twoInst], by simp [twoInst]; omega, rfl⟩
    · have hc : ¬ t < nm := by
        intro hc
        simp [twoInst, hc] at he
      exact ⟨t, by simp [twoInst]; omega, by simp [twoInst]; omega, rfl⟩
  inp_len := fun _ _ ht hk => idx_mul_add_lt ht hk
  lora_len := fun _ _ _ _ _ hl hj hk => lora_idx_lt hl hj hk
  tab_len := fun _ _ hs' => hs'
  out_len := fun _ _ _ hs' ht hj => out_idx_lt hs' ht hj
  tid_len := fun _ hp => hp
  cnt_len := fun g hg => by simp only [twoInst, lay] at hg ⊢; have := hs.G_le; omega
  start_len := fun g hg => by simp only [twoInst, lay] at hg ⊢; have := hs.G_le; omega
  id_len := fun g hg => by simp only [twoInst, lay] at hg ⊢; have := hs.G_le; omega
  out_inp := lay_disj' 3 1 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.inp)
  out_lora := fun s _ => out_lora_disj hs.fits s
  out_tab := fun _ => lay_disj 3 8 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_tid := lay_disj 3 4 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_cnt := lay_disj 3 5 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_start := lay_disj 3 6 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)
  out_id := lay_disj 3 7 (by norm_num) rfl rfl (by simpa [DType.bytes, oneInst, twoInst, lay] using hs.fits.out)

/-- `0.5`, the host test's `scaling` (the double `0x3FE0000000000000`). -/
def scHalf : F64 := FloatV.ofPat 0x3FE0000000000000

/-- `1.0`, the `scaling` vLLM's layers pass (the double `0x3FF0000000000000`). -/
def scOne : F64 := FloatV.ofPat 0x3FF0000000000000

theorem scHalf_fin : scHalf.isFinite = true ∧ (FloatV.hostF32 scHalf).isFinite = true := by
  decide +kernel
theorem scOne_fin : scOne.isFinite = true ∧ (FloatV.hostF32 scOne).isFinite = true := by
  decide +kernel

/-- The shape conditions of a concrete launch, field by field. -/
macro "cov_shape" h:term : tactic =>
  `(tactic| (constructor <;> first
    | exact $h
    | exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩
    | norm_num))

/-- `SETUP.md` case 1: 1 token, hidden 4096, rank 16, 4 adapters, one slice, every token with an
adapter, `num_active_loras = max_loras + 1`; `BLOCK_K = 256`, `SPLIT_K = 64`. -/
def setup_c1 : Inst := oneInst 1 4096 16 4 1 5 0 scHalf

theorem setup_c1_ok : setup_c1.ok :=
  oneInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `SETUP.md` case 4: 128 tokens, hidden 4096, rank 64, 32 adapters, 10 % without an adapter
(13 tokens), the others on adapter 31; `BLOCK_K = 32`, `SPLIT_K = 8`. -/
def setup_c4 : Inst := twoInst 128 4096 64 32 1 33 13 31 scHalf

theorem setup_c4_ok : setup_c4.ok :=
  twoInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `SETUP.md` case 5: 512 tokens, hidden 2049, rank 256, 8 adapters, two slices, 20 % without an
adapter; `K = 2049` takes the masked path (`EVEN_K = False`). -/
def setup_c5 : Inst := twoInst 512 2049 256 8 2 9 102 7 scHalf

theorem setup_c5_ok : setup_c5.ok :=
  twoInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `SETUP.md` case 6: 300 tokens, hidden 14336, rank 32, 4 adapters, three slices,
`specialize_active_lora`: the grid's axis 2 is the number of distinct ids, here `1`. -/
def setup_c6 : Inst := oneInst 300 14336 32 4 3 1 3 scHalf

theorem setup_c6_ok : setup_c6.ok :=
  oneInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `SETUP.md` case 7: 2048 tokens, hidden 8192, rank 16, 128 adapters, 10 % without an adapter
(205 tokens), the others on adapter 127. -/
def setup_c7 : Inst := twoInst 2048 8192 16 128 1 129 205 127 scHalf

theorem setup_c7_ok : setup_c7.ok :=
  twoInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `SETUP.md` case 8: 4096 tokens, hidden 5120, rank 128, 8 adapters. -/
def setup_c8 : Inst := oneInst 4096 5120 128 8 1 9 0 scHalf

theorem setup_c8_ok : setup_c8.ok :=
  oneInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- A vLLM decode step (`scaling = 1.0`): 8 tokens on a merged `qkv_proj` (three slices), hidden
4096, rank 16, 4 adapters; `BLOCK_K = 256`, `SPLIT_K = 64`. -/
def setup_decode_qkv : Inst := oneInst 8 4096 16 4 3 5 1 scOne

theorem setup_decode_qkv_ok : setup_decode_qkv.ok :=
  oneInst_ok (by cov_shape scOne_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- A vLLM prefill step on a merged `gate_up_proj` (two slices, `scaling = 1.0`): 2048 tokens,
hidden 4096, rank 64, 8 adapters, 100 tokens without an adapter; `BLOCK_K = 32`, `SPLIT_K = 8`. -/
def setup_prefill_gate_up : Inst := twoInst 2048 4096 64 8 2 9 100 5 scOne

theorem setup_prefill_gate_up_ok : setup_prefill_gate_up.ok :=
  twoInst_ok (by cov_shape scOne_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- A vLLM layer's launch (`scaling = 1.0`): 32 sequences of 128 tokens, hidden 14336
(`down_proj` of an 8B model, one slice), rank 256, 32 adapters. -/
def setup_layer : Inst := twoInst 4096 14336 256 32 1 33 100 31 scOne

theorem setup_layer_ok : setup_layer.ok :=
  twoInst_ok (by cov_shape scOne_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- `SETUP.md`'s smallest hidden size, `2`, at rank 16 with one adapter. -/
def setup_hidden2 : Inst := oneInst 37 2 16 1 1 2 0 scHalf

theorem setup_hidden2_ok : setup_hidden2.ok :=
  oneInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- Beyond `SETUP.md` (which asks for `max_loras·rank·hidden < 2^31`): the largest
hidden size, `128256`, at rank 256 with 128 adapters, a weight tensor of more than `2^31` elements;
`Inst.ok` covers it for every adapter id up to `65` (`65·256·128256 < 2^31`; adapter `66` is the first
the kernel's `int32` offset wraps on, README "Exclusions"). -/
def setup_hidden_max : Inst := twoInst 64 128256 256 128 1 129 1 65 scOne

theorem setup_hidden_max_ok : setup_hidden_max.ok :=
  twoInst_ok (by cov_shape scOne_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-! ## A memory with a buffer's values -/

/-- Memory `m` with buffer `b` holding `v e` in each element `e`. -/
def fillBuf {dt : DType} (m : Mem) (b : Buf dt) (v : Nat → dt.Val) : Mem := fun x =>
  if b.base ≤ x ∧ x < b.base + b.len * dt.bytes then
    (TVA.Math.Bytes.leBytes dt.bytes (dt.toBits (v ((x - b.base) / dt.bytes).toNat)))[
      ((x - b.base) % dt.bytes).toNat]?
  else m x

theorem fillBuf_frame {dt : DType} (m : Mem) (b : Buf dt) (v : Nat → dt.Val) {x : Int}
    (hx : x ∉ b.bytes) : fillBuf m b v x = m x := by
  unfold fillBuf
  rw [ite_eq_right (by simpa only [Buf.bytes, Set.mem_ofPred_eq] using hx)]

theorem fillBuf_present {dt : DType} {m : Mem} (hm : ∀ x, m x ≠ none) (b : Buf dt)
    (v : Nat → dt.Val) : ∀ x, fillBuf m b v x ≠ none := by
  intro x
  unfold fillBuf
  split_ifs with h
  · have hB := dt.bytes_pos
    have hlt : ((x - b.base) % dt.bytes).toNat <
        (TVA.Math.Bytes.leBytes dt.bytes (dt.toBits (v ((x - b.base) / dt.bytes).toNat))).length := by
      rw [Mem.leBytes_len]
      have h1 := Int.emod_nonneg (x - b.base) (show (dt.bytes : Int) ≠ 0 by omega)
      have h2 := Int.emod_lt_of_pos (x - b.base) (show (0 : Int) < dt.bytes by omega)
      omega
    rw [List.getElem?_eq_getElem hlt]; simp
  · exact hm x

theorem fillBuf_read {dt : DType} {m : Mem} (hm : ∀ x, m x ≠ none) {b : Buf dt}
    (v : Nat → dt.Val) (h0 : 0 ≤ b.base) (h64 : b.base + b.len * dt.bytes ≤ 2 ^ 64)
    (hal : b.base % dt.bytes = 0) {i : Nat} (hi : i < b.len) :
    (fillBuf m b v).read dt (b.ptr ↑i).addr = some (v i) := by
  have hw : m.Writable dt (b.ptr ↑i).addr := b.writable ⟨h0, h64, hal, fun x _ => hm x⟩ hi
  rw [← Mem.read_write_self hw (v i)]
  apply Mem.read_congr
  intro x hx
  have hx' : b.base + (i : Int) * dt.bytes ≤ x ∧ x < b.base + (i : Int) * dt.bytes + dt.bytes := hx
  have hB := dt.bytes_pos
  obtain ⟨r, hr0, hrB, rfl⟩ : ∃ r : Int, 0 ≤ r ∧ r < dt.bytes ∧ x = b.base + (i : Int) * dt.bytes + r :=
    ⟨x - (b.base + (i : Int) * dt.bytes), by omega, by omega, by ring⟩
  have hiL : ((i : Int) + 1) * dt.bytes ≤ (b.len : Int) * dt.bytes :=
    mul_le_mul_of_nonneg_right (by exact_mod_cast hi) (by positivity)
  have hc1 : b.base ≤ b.base + (i : Int) * dt.bytes + r ∧
      b.base + (i : Int) * dt.bytes + r < b.base + b.len * dt.bytes := ⟨by nlinarith, by nlinarith⟩
  have hc2 : (b.ptr ↑i).addr ≤ b.base + (i : Int) * dt.bytes + r ∧
      b.base + (i : Int) * dt.bytes + r < (b.ptr ↑i).addr + dt.bytes := by
    simp only [Buf.ptr_addr]; constructor <;> omega
  unfold fillBuf Mem.write Mem.writeBytes
  rw [ite_eq_left hc1, Mem.leBytes_len, ite_eq_left hc2]
  have hBne : (dt.bytes : Int) ≠ 0 := by omega
  have e1 : (b.base + (i : Int) * dt.bytes + r - b.base) / dt.bytes = i := by
    rw [show b.base + (i : Int) * dt.bytes + r - b.base = r + (i : Int) * dt.bytes by ring,
      Int.add_mul_ediv_right _ _ hBne, Int.ediv_eq_zero_of_lt hr0 hrB, zero_add]
  have e2 : (b.base + (i : Int) * dt.bytes + r - b.base) % dt.bytes = r := by
    rw [show b.base + (i : Int) * dt.bytes + r - b.base = r + (i : Int) * dt.bytes by ring,
      Int.add_mul_emod_self_right, Int.emod_eq_of_lt hr0 hrB]
  rw [e1, e2, Int.toNat_natCast, show b.base + (i : Int) * dt.bytes + r - (b.ptr ↑i).addr = r by
    simp only [Buf.ptr_addr]; ring]

theorem fillBuf_read_other {dt dt' : DType} (m : Mem) (b : Buf dt) (v : Nat → dt.Val) {a : Int}
    (h : a + dt'.bytes ≤ b.base ∨ b.base + b.len * dt.bytes ≤ a) :
    (fillBuf m b v).read dt' a = m.read dt' a :=
  Mem.read_congr fun x hx => fillBuf_frame m b v fun hb => by
    have hx' : a ≤ x ∧ x < a + dt'.bytes := hx
    have hb' : b.base ≤ x ∧ x < b.base + b.len * dt.bytes := hb
    omega

/-- A read of element `i` of a buffer that a `fillBuf` wrote. -/
theorem fillBuf_read_at {dt : DType} {m : Mem} (hm : ∀ x, m x ≠ none) {b : Buf dt}
    (v : Nat → dt.Val) (hv : b.Valid m) {i : Nat} (hi : i < b.len) :
    (fillBuf m b v).read dt (b.ptr ↑i).addr = some (v i) :=
  fillBuf_read hm v hv.1 hv.2.1 hv.2.2.1 hi



/-- The memory every byte is present in, all `0`. -/
def testMem0 : Mem := fun _ => some 0

/-- The zero memory reads as the zero of the dtype. -/
theorem testMem0_readBytes (n : Nat) (a : Int) :
    testMem0.readBytes a n = some (List.replicate n 0) := by
  induction n generalizing a with
  | zero => rfl
  | succ n ih => simp [testMem0, Mem.readBytes, ih, List.replicate_succ]

theorem leValue_replicate_zero (n : Nat) : TVA.Math.Bytes.leValue (List.replicate n 0) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [List.replicate_succ, TVA.Math.Bytes.leValue, ih]

theorem testMem0_read {d : DType} {a : Int} (h : Mem.Aligned d a) :
    testMem0.read d a = some (d.ofBits 0) := by
  unfold Mem.read
  rw [ite_eq_left h, testMem0_readBytes]
  simp [leValue_replicate_zero]

theorem testMem0_read_f32 {a : Int} (h : Mem.Aligned .f32 a) :
    testMem0.read .f32 a = some (FloatV.ofPat 0) :=
  testMem0_read h


/-! ## Not vacuous: a covered instance and a memory satisfying `Pre`

`testInst`: one token on adapter `0`, hidden 512, rank 16, one adapter, one slice,
`num_active_loras = 2`; the
memory is zero except for the four metadata buffers, which hold what `lora_kernel_meta` writes. -/

def testInst : Inst := oneInst 1 512 16 1 1 2 0 scOne

theorem test_ok : testInst.ok :=
  oneInst_ok (by cov_shape scOne_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def tidV (e : Nat) : I32 := IntV.ofNat (testInst.tIdx e)
def cntV (e : Nat) : I32 := IntV.ofNat (cntAt testInst e)
def startV (e : Nat) : I32 := IntV.ofNat (startAt testInst e)
def idV (e : Nat) : I32 := IntV.ofInt (idAt testInst e)

def testMem1 : Mem := fillBuf testMem0 testInst.tIdxBuf tidV
def testMem2 : Mem := fillBuf testMem1 testInst.cntBuf cntV
def testMem3 : Mem := fillBuf testMem2 testInst.startBuf startV
def testMem : Mem := fillBuf testMem3 testInst.idBuf idV

theorem testMem_eq : testMem = fillBuf testMem3 testInst.idBuf idV := rfl
theorem testMem3_eq : testMem3 = fillBuf testMem2 testInst.startBuf startV := rfl
theorem testMem2_eq : testMem2 = fillBuf testMem1 testInst.cntBuf cntV := rfl
theorem testMem1_eq : testMem1 = fillBuf testMem0 testInst.tIdxBuf tidV := rfl

theorem testMem0_present : ∀ x, testMem0 x ≠ none := fun _ => by simp [testMem0]
theorem testMem1_present : ∀ x, testMem1 x ≠ none := fillBuf_present testMem0_present _ _
theorem testMem2_present : ∀ x, testMem2 x ≠ none := fillBuf_present testMem1_present _ _
theorem testMem3_present : ∀ x, testMem3 x ≠ none := fillBuf_present testMem2_present _ _
theorem testMem_present : ∀ x, testMem x ≠ none := fillBuf_present testMem3_present _ _

theorem valid_of {d : DType} {b : Buf d} {m : Mem} (hm : ∀ x, m x ≠ none) (i : Nat)
    (hb : b.base = i * 2 ^ 40) (hi : i < 8) (hlen : b.len * d.bytes ≤ 2 ^ 40) (hal : 16 % d.bytes = 0) :
    b.Valid m := by
  refine ⟨by rw [hb]; positivity, ?_, ?_, fun x _ => hm x⟩
  · rw [hb]
    have : ((b.len * d.bytes : Nat) : Int) ≤ 2 ^ 40 := by exact_mod_cast hlen
    push_cast at this
    have hi' : (i : Int) ≤ 7 := by exact_mod_cast (by omega : i ≤ 7)
    nlinarith
  · rw [hb]
    have h16 : (d.bytes : Int) ∣ 2 ^ 40 := by
      have : d.bytes ∣ 2 ^ 40 := by
        rcases Nat.dvd_of_mod_eq_zero hal with ⟨c, hc⟩
        exact ⟨c * 2 ^ 36, by rw [show (2 : Nat) ^ 40 = 16 * 2 ^ 36 by norm_num, hc]; ring⟩
      exact_mod_cast this
    exact Int.emod_eq_zero_of_dvd (Dvd.dvd.mul_left h16 _)

theorem test_valid {d : DType} {b : Buf d} {m : Mem} (hm : ∀ x, m x ≠ none) (i : Nat)
    (hb : b.base = i * 2 ^ 40) (hi : i < 8) (hlen : b.len * d.bytes ≤ 2 ^ 40)
    (hal : 16 % d.bytes = 0) : b.Valid m := valid_of hm i hb hi hlen hal

theorem read_other {dt dt' : DType} (m : Mem) (b : Buf dt) (v : Nat → dt.Val) {a : Int}
    (i j : Nat) (hij : i < j) (ha : a + dt'.bytes ≤ (i + 1) * 2 ^ 40)
    (hb : b.base = j * 2 ^ 40) :
    (fillBuf m b v).read dt' a = m.read dt' a := by
  refine fillBuf_read_other m b v (Or.inl ?_)
  rw [hb]
  have : ((i : Int) + 1) ≤ j := by exact_mod_cast hij
  nlinarith

theorem test_holds_cnt : testInst.cntBuf.Holds testMem cntV := by
  intro g hg
  have hg' : g < 2 := hg
  have ha : (testInst.cntBuf.ptr ↑g).addr + DType.bytes .i32 ≤ (5 + 1) * 2 ^ 40 := by
    simp [testInst, oneInst, lay, DType.bytes]; omega
  rw [testMem_eq, read_other testMem3 testInst.idBuf idV 5 7 (by norm_num) ha rfl,
    testMem3_eq, read_other testMem2 testInst.startBuf startV 5 6 (by norm_num) ha rfl, testMem2_eq]
  have e1 : testInst.cntBuf.base = ((5 : Nat) : Int) * 2 ^ 40 := rfl
  have e2 : testInst.cntBuf.len * DType.bytes .i32 ≤ 2 ^ 40 := by
    simp [testInst, oneInst, lay, DType.bytes]
  have e3 : 16 % DType.bytes .i32 = 0 := by simp [DType.bytes]
  have hv := test_valid (b := testInst.cntBuf) testMem1_present 5 e1 (by norm_num) e2 e3
  exact fillBuf_read_at (m := testMem1) (b := testInst.cntBuf) (i := g) testMem1_present cntV hv hg

theorem test_holds_start : testInst.startBuf.Holds testMem startV := by
  intro g hg
  have hg' : g < 3 := hg
  have ha : (testInst.startBuf.ptr ↑g).addr + DType.bytes .i32 ≤ (6 + 1) * 2 ^ 40 := by
    simp [testInst, oneInst, lay, DType.bytes]; omega
  rw [testMem_eq, read_other testMem3 testInst.idBuf idV 6 7 (by norm_num) ha rfl, testMem3_eq]
  exact fillBuf_read_at (m := testMem2) (b := testInst.startBuf) (i := g) testMem2_present startV
    (test_valid (b := testInst.startBuf) testMem2_present 6 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes])
      (by simp [DType.bytes])) hg

theorem test_holds_id : testInst.idBuf.Holds testMem idV := by
  intro g hg
  rw [testMem_eq]
  exact fillBuf_read_at (m := testMem3) (b := testInst.idBuf) (i := g) testMem3_present idV
    (test_valid (b := testInst.idBuf) testMem3_present 7 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes])
      (by simp [DType.bytes])) hg

theorem test_read_tid {p : Nat} (hp : p < testInst.M) :
    testMem.read .i32 (testInst.tIdxBuf.ptr ↑p).addr = some (IntV.ofNat (testInst.tIdx p)) := by
  have hp' : p < 1 := hp
  have ha : (testInst.tIdxBuf.ptr ↑p).addr + DType.bytes .i32 ≤ (4 + 1) * 2 ^ 40 := by
    simp [testInst, oneInst, lay, DType.bytes]; omega
  rw [testMem_eq, read_other testMem3 testInst.idBuf idV 4 7 (by norm_num) ha rfl,
    testMem3_eq, read_other testMem2 testInst.startBuf startV 4 6 (by norm_num) ha rfl,
    testMem2_eq, read_other testMem1 testInst.cntBuf cntV 4 5 (by norm_num) ha rfl, testMem1_eq]
  exact fillBuf_read_at (m := testMem0) (b := testInst.tIdxBuf) (i := p) testMem0_present tidV
    (test_valid (b := testInst.tIdxBuf) testMem0_present 4 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes])
      (by simp [DType.bytes])) hp

theorem test_out_zero {s t j : Nat} (hs : s < testInst.S) (ht : t < testInst.M)
    (hj : j < testInst.R) :
    testMem.read .f32 (testInst.out.ptr (outAt testInst s t j)).addr = some (FloatV.ofPat 0) := by
  have hs' : s < 1 := hs
  have ht' : t < 1 := ht
  have hj' : j < 16 := hj
  have hv : testInst.out.Valid testMem0 :=
    test_valid (b := testInst.out) testMem0_present 3 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes])
      (by simp [DType.bytes])
  have hlen : outAt testInst s t j < testInst.out.len := by
    show s * (1 * 16) + t * 16 + j < 1 * (1 * 16); omega
  have ha : (testInst.out.ptr ↑(outAt testInst s t j)).addr + DType.bytes .f32 ≤ (3 + 1) * 2 ^ 40 := by
    simp [testInst, oneInst, lay, DType.bytes, outAt]; omega
  rw [testMem_eq, read_other testMem3 testInst.idBuf idV 3 7 (by norm_num) ha rfl,
    testMem3_eq, read_other testMem2 testInst.startBuf startV 3 6 (by norm_num) ha rfl,
    testMem2_eq, read_other testMem1 testInst.cntBuf cntV 3 5 (by norm_num) ha rfl,
    testMem1_eq, read_other testMem0 testInst.tIdxBuf tidV 3 4 (by norm_num) ha rfl]
  exact testMem0_read_f32 (testInst.out.aligned hv hlen)

theorem test_pre : Pre testInst testMem := by
  have v : ∀ {d : DType} {b : Buf d} (i : Nat), b.base = i * 2 ^ 40 → i < 8 →
      b.len * d.bytes ≤ 2 ^ 40 → 16 % d.bytes = 0 → b.Valid testMem :=
    fun i hb hi hl ha => test_valid testMem_present i hb hi hl ha
  refine ⟨v (b := testInst.inp) 1 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes]) (by simp [DType.bytes]),
    fun s hs => ?_,
    v (b := testInst.out) 3 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes]) (by simp [DType.bytes]),
    v (b := testInst.tIdxBuf) 4 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes]) (by simp [DType.bytes]),
    v (b := testInst.cntBuf) 5 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes]) (by simp [DType.bytes]),
    v (b := testInst.startBuf) 6 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes]) (by simp [DType.bytes]),
    v (b := testInst.idBuf) 7 rfl (by norm_num) (by simp [testInst, oneInst, lay, DType.bytes]) (by simp [DType.bytes]),
    fun h => absurd h (by decide),
    test_holds_cnt, test_holds_start, test_holds_id, fun p hp => test_read_tid hp,
    fun s t j hs ht hj => test_out_zero hs ht hj⟩
  have hs0 : s = 0 := by have : s < 1 := hs; omega
  subst hs0
  exact v (b := testInst.lora 0) 2 (by simp [testInst, oneInst, lay, loraAt]) (by norm_num)
    (by simp [testInst, oneInst, lay, DType.bytes]) (by simp [DType.bytes])

theorem satisfiable : Claim.Satisfiable Inst.ok Pre :=
  ⟨testInst, testMem, test_ok, test_pre⟩

end Claims.LoraShrink
