import TVA

/-!
# `scaled_mm_int8`: the prepared claim

Source: `source/triton_scaled_mm.py` (vLLM v0.33.0's
`model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py`, with the changes
`SETUP.md` lists). The kernel is `scaled_mm_kernel` (lines 55–206); it reads the module-level names
`triton` (its decorator) and `tl` (lines 43–44). Its host `triton_scaled_mm` (lines 211–302) launches
it on an `int8` matrix `A` (`M × K`, the quantised activations), an `int8` matrix `B` (`K × N`, the
quantised weights), `float32` scales (`scale_a` per token or per tensor, `scale_b` per channel or
per tensor), an optional bias, into a fresh `bfloat16` or `float16` matrix `C` (`M × N`), with the
tile its heuristic picks from `M` and `N`; `SETUP.md` lists the launches.

This file holds the model (`scaled_mm_kernel`, `model`, `launch`), the conditions (`Inst`,
`Inst.ok`, `Pre`), the specification (`OutSpec`, `Post`, `Spec`), the assumptions (`Assumptions`)
and the claim `Statement`, which it does not prove. It proves `consistent`, `satisfiable`, and that
each launch family of `SETUP.md` is covered (`setup_*_ok`).
-/

namespace Claims.ScaledMmInt8

open TVA

/-! ## The model -/

/-- `scaled_mm_kernel` (source lines 55–206) on the launches the host makes:
`ACCUMULATOR_DTYPE`, a `tl.constexpr` parameter, holds the dtype `tl.int32` (the host's value for
`int8` inputs, line 268) and means at each use what `tl.int32` means (SEMANTICS.md, "A dtype as a
`tl.constexpr` argument"), so the model writes `.i32` there and has no parameter for it; `USE_TD = False` (line 272), a compile-time condition, so lines 127–149 and
152–157 are not visited and `B_T` (read only there) has no meaning in the model. `bias_ptr` is
`none` when the host passes `None` (a compile-time `None`), and `some p` for a bias tensor. The
output's dtype `od` (`bfloat16` or `float16` on covered launches) is the element type of `c_ptr`
and of `bias_ptr`. The block sizes are Lean parameters (`Nat`).

The pointers: `a_ptr`, `b_ptr` `*int8`; `scale_a_ptr`, `scale_b_ptr` `*float32`; `c_ptr`, `bias_ptr`
`*od`. `M`, `N`, `K` and the six strides are `int32`: an integer argument in `[-2^31, 2^31)` other
than `1` is a runtime `int32`, and each is below `2^31` on covered instances (`Inst.ok`); at `1` an
argument is the compile-time constant `1`, which gives the same model at every use (see `model`). -/
def scaled_mm_kernel (od : DType) (a_ptr b_ptr : Ptr .i8) (scale_a_ptr scale_b_ptr : Ptr .f32)
    (c_ptr : Ptr od) (bias_ptr : Option (Ptr od)) (M N K : I32)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : I32)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K BLOCK_SIZE_SCALE_A BLOCK_SIZE_SCALE_B : Nat) : Kernel where
  body ctx := do
    -- line 81: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- line 83: `tl.cdiv(N, BLOCK_SIZE_N)` is `(N + (BLOCK_SIZE_N - 1)) // BLOCK_SIZE_N`: the
    -- constexpr `BLOCK_SIZE_N - 1` computed by Python, built in `int32` (an operand of `+` with the
    -- `int32` `N`); the wrapping `int32` `+`, then the truncating `//` (`tl.idiv`)
    let num_pid_n := (← tl.idiv (Tensor.scalar (N + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- line 85: `int32` `//`
    let pid_m := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar num_pid_n)).item
    -- line 86: `int32` `%`
    let pid_n := (← tl.irem (Tensor.scalar pid) (Tensor.scalar num_pid_n)).item
    -- lines 88–89: `accumulator_dtype` is the dtype `ACCUMULATOR_DTYPE`, `tl.int32`; `0` in every
    -- element
    let accumulator := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .i32
    -- line 96: the `int32` scalar `pid_m * BLOCK_SIZE_M` (the constexpr built in `int32`) widened
    -- to `int64`, since the other operand, `tl.arange(0, BLOCK_SIZE_M).to(tl.int64)` (the exact
    -- widening), is `int64`; the wrapping `int64` sum
    let offsets_am : Tensor [BLOCK_SIZE_M] I64 :=
      (← tl.convert .i32 .i64 (Tensor.scalar (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)))).item +
        (← tl.convert .i32 .i64 (tl.arange 0 BLOCK_SIZE_M))
    -- line 97: the signed `int64` comparison with the `int32` scalar `M` widened
    let masks_am : Tensor [BLOCK_SIZE_M] Bool :=
      offsets_am.ltS (← tl.convert .i32 .i64 (Tensor.scalar M)).item
    -- line 99: as line 96 with `pid_n`, `BLOCK_SIZE_N`
    let offsets_bn : Tensor [BLOCK_SIZE_N] I64 :=
      (← tl.convert .i32 .i64 (Tensor.scalar (pid_n * (IntV.ofNat BLOCK_SIZE_N : I32)))).item +
        (← tl.convert .i32 .i64 (tl.arange 0 BLOCK_SIZE_N))
    -- line 100: as line 97 with `N`
    let masks_bn : Tensor [BLOCK_SIZE_N] Bool :=
      offsets_bn.ltS (← tl.convert .i32 .i64 (Tensor.scalar N)).item
    -- line 102: the exact widening
    let offsets_k : Tensor [BLOCK_SIZE_K] I64 ← tl.convert .i32 .i64 (tl.arange 0 BLOCK_SIZE_K)
    -- line 103: each `int32` scalar stride widened to `int64` (the other operand is `int64`), the
    -- wrapping `int64` products of the column `offsets_am[:, None]` and the row `offsets_k[None, :]`,
    -- broadcast to `[BLOCK_SIZE_M, BLOCK_SIZE_K]`, and their wrapping `int64` sum
    let offsets_a : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] I64 :=
      ((← tl.convert .i32 .i64 (Tensor.scalar stride_am)).item * offsets_am).col BLOCK_SIZE_K +
        ((← tl.convert .i32 .i64 (Tensor.scalar stride_ak)).item * offsets_k).row BLOCK_SIZE_M
    -- line 104: the same for `b`, `[BLOCK_SIZE_K, BLOCK_SIZE_N]`
    let offsets_b : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] I64 :=
      ((← tl.convert .i32 .i64 (Tensor.scalar stride_bk)).item * offsets_k).col BLOCK_SIZE_N +
        ((← tl.convert .i32 .i64 (Tensor.scalar stride_bn)).item * offsets_bn).row BLOCK_SIZE_K
    -- lines 109–112: `(BLOCK_SIZE_SCALE_A > 1)` is a Python `bool` (both operands compile-time);
    -- `bool * pid_m` builds it in `int32` (`1` or `0`, the dtype of the `int32` scalar `pid_m`),
    -- the wrapping `int32` products (the constexpr `BLOCK_SIZE_M` built in `int32`), added to every
    -- lane of the `int32` `tl.arange(0, BLOCK_SIZE_SCALE_A)`
    let offsets_scale_am : Tensor [BLOCK_SIZE_SCALE_A] I32 :=
      tl.arange 0 BLOCK_SIZE_SCALE_A +
        (IntV.ofNat (if 1 < BLOCK_SIZE_SCALE_A then 1 else 0) : I32) * pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)
    -- line 113: signed `int32` comparison with `M`
    let masks_scale_am : Tensor [BLOCK_SIZE_SCALE_A] Bool := offsets_scale_am.ltS M
    -- lines 115–118: as lines 109–112 for `b`'s scale
    let offsets_scale_bn : Tensor [BLOCK_SIZE_SCALE_B] I32 :=
      tl.arange 0 BLOCK_SIZE_SCALE_B +
        (IntV.ofNat (if 1 < BLOCK_SIZE_SCALE_B then 1 else 0) : I32) * pid_n * (IntV.ofNat BLOCK_SIZE_N : I32)
    -- line 119
    let masks_scale_bn : Tensor [BLOCK_SIZE_SCALE_B] Bool := offsets_scale_bn.ltS N
    -- lines 121–122: pointer additions, in elements
    let a_ptrs := a_ptr + offsets_a
    let b_ptrs := b_ptr + offsets_b
    -- lines 124–125
    let scale_a_ptrs := scale_a_ptr + offsets_scale_am
    let scale_b_ptrs := scale_b_ptr + offsets_scale_bn
    -- lines 127–149: not visited (`USE_TD` is `False`, a compile-time condition)
    -- line 151: `for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))`: the bound
    -- `(K + (BLOCK_SIZE_K - 1)) // BLOCK_SIZE_K` in `int32`, evaluated once, before the loop; the
    -- `int32` bounds `0`, the implicit step `1` and induction variable; the loop carries
    -- `offsets_k`, `a_ptrs`, `b_ptrs`, `accumulator` (assigned in the body, defined before it);
    -- `masks_k`, `masks_a`, `masks_b`, `a`, `b` are first defined in the body and not used after it
    let (_offsets_k, _a_ptrs, _b_ptrs, accumulator) ← tl.forRange (IntV.ofInt 0 : I32)
        (← tl.idiv (Tensor.scalar (K + (IntV.ofNat (BLOCK_SIZE_K - 1) : I32)))
          (Tensor.scalar (IntV.ofNat BLOCK_SIZE_K : I32))).item
        (IntV.ofInt 1 : I32) (offsets_k, a_ptrs, b_ptrs, accumulator)
        fun _k (offsets_k, a_ptrs, b_ptrs, accumulator) => do
      -- line 152: `USE_TD` is `False`: only the `else` branch (lines 159–164) is visited
      -- line 159: the signed `int64` comparison with `K` widened
      let masks_k : Tensor [BLOCK_SIZE_K] Bool :=
        offsets_k.ltS (← tl.convert .i32 .i64 (Tensor.scalar K)).item
      -- line 160: `&` of the column `masks_am[:, None]` and the row `masks_k[None, :]`, broadcast
      let masks_a := (masks_am.col BLOCK_SIZE_K).and (masks_k.row BLOCK_SIZE_M)
      -- line 161: a masked load with no `other`: the disabled lanes hold unspecified values
      let a ← tl.load a_ptrs (some masks_a)
      -- line 163
      let masks_b := (masks_k.col BLOCK_SIZE_N).and (masks_bn.row BLOCK_SIZE_K)
      -- line 164: as line 161
      let b ← tl.load b_ptrs (some masks_b)
      -- line 167: `tl.dot(a, b, accumulator, out_dtype=tl.int32)`: `int8` operands into the
      -- `int32` accumulator, passed as the dot's own accumulator; no `input_precision` (the
      -- default `tf32`, which has no effect on integer operands) and no `max_num_imprecise_acc`
      -- (`0` for non-fp8 operands, `semantic.py:1516-1520`)
      let accumulator ← tl.dot
        { a := .i8, b := .i8, acc := .i32, precision := .tf32, maxImpreciseAcc := 0 } a b accumulator
      -- line 169: `x += y` is `x = x + y`; the constexpr built in `int64` (the tensor's dtype)
      let offsets_k := offsets_k + (IntV.ofNat BLOCK_SIZE_K : I64)
      -- lines 170–171: the constexpr built in `int32` as an operand of `*` with the `int32` scalar
      -- stride, the wrapping `int32` product, added to every pointer of the block
      let a_ptrs := a_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_ak) : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] I32)
      let b_ptrs := b_ptrs +
        (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_bk) : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] I32)
      return (offsets_k, a_ptrs, b_ptrs, accumulator)
    -- line 174: `masks_scale_am[:, None]` (`[BLOCK_SIZE_SCALE_A, 1]`) `&` the `[1, 1]` block
    -- `(tl.arange(0, 1) < 1)[:, None]` broadcast; the Python `1` of the comparison is an `int32`
    -- tensor of its own type, the signed `int32` comparison
    let masks_scale_a : Tensor [BLOCK_SIZE_SCALE_A, 1] Bool :=
      (masks_scale_am.col 1).and
        (Tensor.bcastRows (((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).col 1 : Tensor [1, 1] Bool)
          BLOCK_SIZE_SCALE_A)
    -- line 175: the load of `scale_a_ptrs[:, None]` with the mask (positional), no `other`
    let scale_a ← tl.load (scale_a_ptrs.col 1) (some masks_scale_a)
    -- line 179: `broadcast_to((BLOCK_SIZE_M, 1))`, `Tensor.broadcastTo` (SEMANTICS.md,
    -- `tl.broadcast_to`): Triton accepts the `[BLOCK_SIZE_SCALE_A, 1]` block only when
    -- `Shape.bcastOk` holds (`BLOCK_SIZE_SCALE_A` is `1` or `BLOCK_SIZE_M`, as on every covered
    -- launch) and rejects every other shape on the host, where the model has no outcome (`.ub`)
    let scale_a ← if h : Shape.bcastOk [BLOCK_SIZE_SCALE_A, 1] [BLOCK_SIZE_M, 1] = true then
        pure (scale_a.broadcastTo [BLOCK_SIZE_M, 1] h) else .ub
    -- line 180: `accumulator.to(tl.float32)`, `int32` to `float32` (`tl.itof`); then the IEEE
    -- `float32` product with `scale_a` broadcast along the columns (no addition consumes it before
    -- line 185's product)
    let acc_f : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] F32 ← tl.itof accumulator
    let accumulator ← tl.mul (scale_a.bcastCols BLOCK_SIZE_N) acc_f
    -- line 182: as line 174, with `(tl.arange(0, 1) < 1)[None, :]` (also `[1, 1]`)
    let masks_scale_b : Tensor [BLOCK_SIZE_SCALE_B, 1] Bool :=
      (masks_scale_bn.col 1).and
        (Tensor.bcastRows (((tl.arange 0 1).ltS (IntV.ofInt 1 : I32)).row 1 : Tensor [1, 1] Bool)
          BLOCK_SIZE_SCALE_B)
    -- line 183
    let scale_b ← tl.load (scale_b_ptrs.col 1) (some masks_scale_b)
    -- line 184: `broadcast_to((BLOCK_SIZE_N, 1))`, as line 179
    let scale_b ← if h : Shape.bcastOk [BLOCK_SIZE_SCALE_B, 1] [BLOCK_SIZE_N, 1] = true then
        pure (scale_b.broadcastTo [BLOCK_SIZE_N, 1] h) else .ub
    -- line 185: `scale_b.T` (`[1, BLOCK_SIZE_N]`, `Tensor.trans`) broadcast along the rows;
    -- `accumulator.to(tl.float32)` of a `float32` block is the block itself (`semantic.cast`
    -- returns its input, `semantic.py:815-816`); the IEEE `float32` product
    let accumulator ← tl.mul (Tensor.bcastRows scale_b.trans BLOCK_SIZE_M) accumulator
    -- line 188: `if bias_ptr:` (SEMANTICS.md, "A runtime `if`") — with `None`, a compile-time
    -- condition that is false (lines 189–193 not visited); with a pointer, a runtime `if` on the
    -- pointer converted to `int1`, `p.addr ≠ 0`
    let accumulator ← (match bias_ptr with
      | none => pure accumulator
      | some bias_ptr =>
        if bias_ptr.addr ≠ 0 then do
          -- line 189
          let offsets_bias := offsets_bn
          -- line 190
          let bias_ptrs := bias_ptr + offsets_bias
          -- line 191: as line 100
          let bias_mask : Tensor [BLOCK_SIZE_N] Bool :=
            offsets_bias.ltS (← tl.convert .i32 .i64 (Tensor.scalar N)).item
          -- line 192: the mask positional, no `other`
          let bias ← tl.load bias_ptrs (some bias_mask)
          -- line 193: `accumulator += bias`: `bias` converted to `float32` (`tl.convert`, exact) and
          -- broadcast along the rows; the addition consumes line 185's product, outside this
          -- branch, so it is the contractible `tl.addc`
          let bias32 ← tl.convert od .f32 bias
          tl.addc accumulator (bias32.row BLOCK_SIZE_M)
        else pure accumulator)
    -- line 196: `accumulator.to(c_ptr.type.element_ty)`: a pointer argument's `.type.element_ty`
    -- is its buffer's dtype `od` (SEMANTICS.md, "Values"); `float32` to `od` (`tl.convert`: one
    -- rounding to nearest even)
    let c : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] od.Val ← tl.convert .f32 od accumulator
    -- lines 199–200: as line 96; lines 201–202: `.to(tl.int64)` of an `int64` block is the block
    let offs_cm : Tensor [BLOCK_SIZE_M] I64 :=
      (← tl.convert .i32 .i64 (Tensor.scalar (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)))).item +
        (← tl.convert .i32 .i64 (tl.arange 0 BLOCK_SIZE_M))
    let offs_cn : Tensor [BLOCK_SIZE_N] I64 :=
      (← tl.convert .i32 .i64 (Tensor.scalar (pid_n * (IntV.ofNat BLOCK_SIZE_N : I32)))).item +
        (← tl.convert .i32 .i64 (tl.arange 0 BLOCK_SIZE_N))
    -- line 203: `(c_ptr + stride_cm * offs_cm[:, None]) + stride_cn * offs_cn[None, :]`: the
    -- `int32` scalar strides widened, the `int64` products, two pointer additions, broadcast
    let c_ptrs := c_ptr +
      ((← tl.convert .i32 .i64 (Tensor.scalar stride_cm)).item * offs_cm).col BLOCK_SIZE_N +
      ((← tl.convert .i32 .i64 (Tensor.scalar stride_cn)).item * offs_cn).row BLOCK_SIZE_M
    -- line 204: two signed `int64` comparisons with `M`, `N` widened, broadcast, then `&`
    let c_mask := ((offs_cm.col BLOCK_SIZE_N).ltS (← tl.convert .i32 .i64 (Tensor.scalar M)).item).and
      ((offs_cn.row BLOCK_SIZE_M).ltS (← tl.convert .i32 .i64 (Tensor.scalar N)).item)
    -- line 206: `c` is already of the pointer's dtype
    tl.store c_ptrs c (some c_mask)

/-! ## The conditions -/

/-- The host's tile (lines 249–261, `use_heuristic=True`): `(BLOCK_SIZE_M, BLOCK_SIZE_N,
BLOCK_SIZE_K)` from `M` and `N`. `next_power_of_2_M = max(32, next_power_of_2(M))` is at most `32`,
`64`, `128` exactly when `M` is (for `M ≥ 1`). -/
def heurTile (M N : Nat) : Nat × Nat × Nat :=
  if M ≤ 32 then (if N < 8192 then (64, 64, 256) else (64, 128, 256))
  else if M ≤ 64 then (64, 64, 256)
  else if M ≤ 128 then (64, 128, 128)
  else (128, 128, 128)

/-- What the host chooses: one launch of `scaled_mm_kernel` through `triton_scaled_mm`. -/
structure Inst where
  /-- `M`: the rows of `A` and `C` (`input.shape[0]`). -/
  M : Nat
  /-- `N`: the columns of `B` and `C` (`weight.shape[1]`). -/
  N : Nat
  /-- `K`: the columns of `A`, the rows of `B`. -/
  K : Nat
  /-- `stride_am` (`input.stride(0)`), in elements. -/
  stride_am : Nat
  /-- `stride_ak` (`input.stride(1)`). -/
  stride_ak : Nat
  /-- `stride_bk` (`weight.stride(0)`). -/
  stride_bk : Nat
  /-- `stride_bn` (`weight.stride(1)`). -/
  stride_bn : Nat
  /-- `stride_cm` (`result.stride(0)`). -/
  stride_cm : Nat
  /-- `stride_cn` (`result.stride(1)`). -/
  stride_cn : Nat
  /-- `BLOCK_SIZE_M` (line 263). -/
  BM : Nat
  /-- `BLOCK_SIZE_N`. -/
  BN : Nat
  /-- `BLOCK_SIZE_K`. -/
  BK : Nat
  /-- `scale_a` has one value per row: `BLOCK_SIZE_SCALE_A = BLOCK_SIZE_M` (line 265); otherwise
  one value for all rows and `BLOCK_SIZE_SCALE_A = 1`. -/
  saVec : Bool
  /-- `scale_b` has one value per column: `BLOCK_SIZE_SCALE_B = BLOCK_SIZE_N` (line 266). -/
  sbVec : Bool
  /-- The output's dtype (`out_dtype`), also the bias's. -/
  od : DType
  /-- A bias is passed (`bias` is not `None`). -/
  hasBias : Bool
  /-- `a_ptr`: the storage of `A`, `int8`; element `(i, k)` is element
  `i * stride_am + k * stride_ak` of it. -/
  a : Buf .i8
  /-- `b_ptr`: the storage of `B`, `int8`; element `(k, j)` is `k * stride_bk + j * stride_bn`. -/
  b : Buf .i8
  /-- `scale_a_ptr`: the storage of `scale_a`, `float32`; the kernel reads element `i` for row `i`
  (`saVec`) or element `0`. -/
  sa : Buf .f32
  /-- `scale_b_ptr`: the storage of `scale_b`, `float32`; element `j` for column `j` (`sbVec`) or
  element `0`. -/
  sb : Buf .f32
  /-- `c_ptr`: the storage of `C`; element `(i, j)` is `i * stride_cm + j * stride_cn`. -/
  c : Buf od
  /-- `bias_ptr` (when `hasBias`): the storage of the bias; element `j` for column `j`. -/
  bias : Buf od
  /-- `scale_a.stride(0)`, in elements: row `i`'s scale is element `i * stride_sa` of `scale_a`'s
  storage (the kernel takes no stride of the scales or of the bias: it reads element `i`). -/
  stride_sa : Nat
  /-- `scale_b.stride(0)`: column `j`'s scale is element `j * stride_sb`. -/
  stride_sb : Nat
  /-- `bias.stride(0)`: column `j`'s bias is element `j * stride_bias`. -/
  stride_bias : Nat

/-- `BLOCK_SIZE_SCALE_A` (line 265). -/
def BSA (d : Inst) : Nat := if d.saVec then d.BM else 1

/-- `BLOCK_SIZE_SCALE_B` (line 266). -/
def BSB (d : Inst) : Nat := if d.sbVec then d.BN else 1

/-- The element of `scale_a`'s storage that holds row `i`'s scale: `scale_a[i, 0]`, or the one
value `scale_a[0, 0]`. -/
def saIdx (d : Inst) (i : Nat) : Nat := if d.saVec then i * d.stride_sa else 0

/-- The element of `scale_b`'s storage that holds column `j`'s scale. -/
def sbIdx (d : Inst) (j : Nat) : Nat := if d.sbVec then j * d.stride_sb else 0

/-- The number of elements of `scale_a` the kernel reads: `M` or `1`. -/
def saN (d : Inst) : Nat := if d.saVec then d.M else 1

/-- The number of elements of `scale_b` the kernel reads: `N` or `1`. -/
def sbN (d : Inst) : Nat := if d.sbVec then d.N else 1

/-- The elements of `a`'s storage that make up the matrix `A` (`M × K`). -/
def aIdx (d : Inst) : Set Nat :=
  {e | ∃ i k, i < d.M ∧ k < d.K ∧ e = i * d.stride_am + k * d.stride_ak}

/-- The elements of `b`'s storage that make up the matrix `B` (`K × N`). -/
def bIdx (d : Inst) : Set Nat :=
  {e | ∃ k j, k < d.K ∧ j < d.N ∧ e = k * d.stride_bk + j * d.stride_bn}

/-- The elements of `c`'s storage that make up the matrix `C` (`M × N`). -/
def cIdx (d : Inst) : Set Nat :=
  {e | ∃ i j, i < d.M ∧ j < d.N ∧ e = i * d.stride_cm + j * d.stride_cn}

/-- The instances covered. Shapes, strides and layouts are variables; the bounds keep the kernel's
`int32` arithmetic and its `int32` accumulator from wrapping, the reduction free of the masked
lanes (`K_div`), the grid within CUDA's limit, and the elements of `C` apart. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The output is `bfloat16` or `float16` (the dtype of the activations, `x.dtype`, in the W8A8
  linear). -/
  od_ok : d.od = .bf16 ∨ d.od = .f16
  /-- The tile is the heuristic's (`use_heuristic=True`, every caller's default) or the host's
  default block sizes `(32, 32, 32)` (`use_heuristic=False` with no block sizes, vLLM's
  rounding test). -/
  tile_ok : (d.BM, d.BN, d.BK) = heurTile d.M d.N ∨ (d.BM, d.BN, d.BK) = (32, 32, 32)
  /-- `M, N, K > 0` (the host's assertion, line 226). -/
  M_pos : 0 < d.M
  /-- As `M_pos`. -/
  N_pos : 0 < d.N
  /-- As `M_pos`. -/
  K_pos : 0 < d.K
  /-- `K` is a multiple of `BLOCK_SIZE_K`: otherwise the last trip's loads (lines 161, 164) mask
  the lanes `k ≥ K` with no `other`, whose unspecified values enter the `tl.dot` of every output
  (a kernel limit, README, Exclusions). -/
  K_div : d.K % d.BK = 0
  /-- `K < 2^17`: each product of two `int8` values is at most `2^14` in magnitude, so every
  partial sum of the `int32` accumulator stays below `2^31` (a kernel limit beyond: README). -/
  K_lt : d.K < 2 ^ 17
  /-- `M + BLOCK_SIZE_M ≤ 2^31`: `M` is an `int32`; the rows `pid_m·BLOCK_SIZE_M + i` and the
  `int32` products `pid_m * BLOCK_SIZE_M` (lines 96, 111, 199) stay below `2^31`. -/
  M_le : d.M + d.BM ≤ 2 ^ 31
  /-- `N + BLOCK_SIZE_N ≤ 2^31`: `N` is an `int32`, and `tl.cdiv(N, BLOCK_SIZE_N)` computes
  `N + BLOCK_SIZE_N - 1` in `int32` (line 83). -/
  N_le : d.N + d.BN ≤ 2 ^ 31
  /-- The grid, `cdiv(M, BLOCK_SIZE_M) · cdiv(N, BLOCK_SIZE_N)` programs, has fewer than `2^31`
  programs (`Launch.Valid`; `pid` is an `int32`). -/
  grid_lt : (d.M + d.BM - 1) / d.BM * ((d.N + d.BN - 1) / d.BN) < 2 ^ 31
  /-- Each stride is an `int32` argument. -/
  strides : d.stride_am < 2 ^ 31 ∧ d.stride_ak < 2 ^ 31 ∧ d.stride_bk < 2 ^ 31 ∧
    d.stride_bn < 2 ^ 31 ∧ d.stride_cm < 2 ^ 31 ∧ d.stride_cn < 2 ^ 31
  /-- The `int32` products `BLOCK_SIZE_K * stride_ak` and `BLOCK_SIZE_K * stride_bk` (lines
  170–171) do not wrap. -/
  steps : d.BK * d.stride_ak < 2 ^ 31 ∧ d.BK * d.stride_bk < 2 ^ 31
  /-- Distinct elements of the matrix `C` are distinct elements of its storage. -/
  c_inj : ∀ i j i' j', i < d.M → j < d.N → i' < d.M → j' < d.N →
    i * d.stride_cm + j * d.stride_cn = i' * d.stride_cm + j' * d.stride_cn → i = i' ∧ j = j'
  /-- `a`'s storage holds every element of the matrix `A`. -/
  a_len : ∀ i k, i < d.M → k < d.K → i * d.stride_am + k * d.stride_ak < d.a.len
  /-- `b`'s storage holds every element of the matrix `B`. -/
  b_len : ∀ k j, k < d.K → j < d.N → k * d.stride_bk + j * d.stride_bn < d.b.len
  /-- `c`'s storage holds every element of the matrix `C`. -/
  c_len : ∀ i j, i < d.M → j < d.N → i * d.stride_cm + j * d.stride_cn < d.c.len
  /-- `scale_a`'s storage holds the `M` (or `1`) values the kernel reads. -/
  sa_len : saN d ≤ d.sa.len
  /-- `scale_b`'s storage holds the `N` (or `1`) values. -/
  sb_len : sbN d ≤ d.sb.len
  /-- With a bias, its storage holds `N` values. -/
  bias_len : d.hasBias = true → d.N ≤ d.bias.len
  /-- The vector scales and the bias are contiguous (`stride(0) = 1`): the kernel reads row `i`'s
  scale, column `j`'s scale and bias at element `i` or `j` of their storage (lines 109-125, 189-192),
  which is the tensor's element only then (a kernel limit: README, Exclusions; the host does not
  check it). -/
  scales_contig : (d.saVec = true → d.stride_sa = 1) ∧ (d.sbVec = true → d.stride_sb = 1) ∧
    (d.hasBias = true → d.stride_bias = 1)
  /-- With a bias, its address is not `0` (line 188 tests the pointer; a CUDA tensor's data
  pointer is never null). -/
  bias_nz : d.hasBias = true → d.bias.base ≠ 0
  /-- The matrix `C` shares no byte with the matrix `A`. -/
  c_a : Disjoint (d.c.elems (cIdx d)) (d.a.elems (aIdx d))
  /-- The matrix `C` shares no byte with the matrix `B`. -/
  c_b : Disjoint (d.c.elems (cIdx d)) (d.b.elems (bIdx d))
  /-- The matrix `C` shares no byte with the scales the kernel reads. -/
  c_sa : Disjoint (d.c.elems (cIdx d)) (d.sa.elems {e | e < saN d})
  /-- As `c_sa`. -/
  c_sb : Disjoint (d.c.elems (cIdx d)) (d.sb.elems {e | e < sbN d})
  /-- With a bias, the matrix `C` shares no byte with it. -/
  c_bias : d.hasBias = true → Disjoint (d.c.elems (cIdx d)) (d.bias.elems {e | e < d.N})

/-- The model on an instance: the buffers' first elements, `bias_ptr` `none` without a bias,
`M`, `N`, `K` and the strides as `int32`, and the block sizes of the launch (lines 263–266). An
integer argument equal to `1` is the compile-time constant `1`; every use of `M`, `N`, `K` and of a
stride then makes it an `int32` or `int64` constant of the same value: an operand of `+`, `*` or a
comparison with an `int32` or `int64` tensor builds it in the tensor's dtype, which the model's
widening gives too; `tl.cdiv(1, BLOCK_SIZE_N)` of two constants is Python's `1`, which the name
`num_pid_n` then holds as an `int32` scalar tensor, the value the `int32` computation gives
(`K = 1` is not covered: `K_div`); `BLOCK_SIZE_K * stride` at `stride = 1` is Python's
`BLOCK_SIZE_K`, which the pointer addition makes an `int32`. So the model is the same. -/
def model (d : Inst) : Kernel :=
  scaled_mm_kernel d.od (d.a.ptr 0) (d.b.ptr 0) (d.sa.ptr 0) (d.sb.ptr 0) (d.c.ptr 0)
    (if d.hasBias then some (d.bias.ptr 0) else none)
    (IntV.ofNat d.M) (IntV.ofNat d.N) (IntV.ofNat d.K)
    (IntV.ofNat d.stride_am) (IntV.ofNat d.stride_ak) (IntV.ofNat d.stride_bk) (IntV.ofNat d.stride_bn)
    (IntV.ofNat d.stride_cm) (IntV.ofNat d.stride_cn) d.BM d.BN d.BK (BSA d) (BSB d)

/-- The host's launch (lines 276–300): a 1-D grid of `triton.cdiv(M, BLOCK_SIZE_M) *
triton.cdiv(N, BLOCK_SIZE_N)` programs (lines 241–243); the source's block shapes;
`enable_fp_fusion` is Triton's default (`true`); `grid1` keeps `num_ctas = 1`,
`launch_pdl = False` and `debug = false`. The launch passes no `num_warps` or `num_stages`. -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.M + d.BM - 1) / d.BM * ((d.N + d.BN - 1) / d.BN))
    [[d.BM], [d.BN], [d.BK], [d.BM, 1], [1, d.BK], [d.BM, d.BK], [d.BK, 1], [1, d.BN], [d.BK, d.BN],
      [d.BM, d.BN], [BSA d], [BSA d, 1], [BSB d], [BSB d, 1], [1], [1, 1], [d.BN, 1]]
    (fusion := true)

/-- The inputs covered: the buffers the kernel accesses are valid (aligned, inside `[0, 2^64)`,
every byte present). Any values in them (`int8` values are all finite; any `float32` scales and
any bias values, NaN and infinities included). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.a.Valid init ∧ d.b.Valid init ∧ d.sa.Valid init ∧ d.sb.Valid init ∧ d.c.Valid init ∧
    (d.hasBias = true → d.bias.Valid init)

/-! ## The specification -/

/-- The exact dot product of row values `va` and column values `vb` over `k < K`, an integer. -/
def dotSum (va vb : Nat → I8) (K : Nat) : ℤ := ∑ k ∈ Finset.range K, (va k).toInt * (vb k).toInt

/-- `float32`'s unit roundoff, `2^-24`. -/
def u32 : ℚ := Math.Float.Format.binary32.u

/-- `float32`'s absolute rounding constant, `2^-150`. -/
def eta32 : ℚ := Math.Float.Format.binary32.eta

/-- The largest `float32`. -/
def max32 : ℚ := Math.Float.Format.binary32.maxVal

/-- Error after `accumulator.to(tl.float32)` (line 180) of the exact integer `S`: one rounding. -/
def e1 (S : ℚ) : ℚ := |S| * u32 + eta32

/-- Error of `scale_a * x` (line 180) against `sa·S`: `|sa|·e1` from `x`, plus one rounding of
`sa·x` (`|sa·x| ≤ |sa·S| + |sa|·e1`). -/
def e2 (sa S : ℚ) : ℚ := |sa| * e1 S + (|sa * S| + |sa| * e1 S) * u32 + eta32

/-- Error of `scale_b.T * y` (line 185) against `sa·sb·S`: `|sb|·e2` from `y`, plus one rounding. -/
def e3 (sa sb S : ℚ) : ℚ := |sb| * e2 sa S + (|sa * sb * S| + |sb| * e2 sa S) * u32 + eta32

/-- **The `float32` tolerance** before the output's rounding, against `sa·sb·S + β`: `e3` without
a bias; with one, `e3` plus the contractible addition of line 193 (`FloatV.AddRel.err`:
`3u·(|x| + |β|) + 4η`, with `|x| ≤ |sa·sb·S| + e3`). -/
def tol (hasBias : Bool) (sa sb S β : ℚ) : ℚ :=
  if hasBias then e3 sa sb S + 3 * u32 * (|sa * sb * S| + e3 sa sb S + |β|) + 4 * eta32
  else e3 sa sb S

/-- Room in `float32` for every intermediate value (the operands of each rounding in range, and
the addition's operands below `maxVal/2 − 2η`). -/
def Room32 (sa sb S β : ℚ) : Prop :=
  |sa| * (|S| + e1 S) ≤ max32 ∧ |sb| * (|sa * S| + e2 sa S) ≤ max32 ∧
    |sa * sb * S| + e3 sa sb S + |β| ≤ max32 / 2 - 2 * eta32

/-- The output format's unit roundoff: `2^-8` (`bfloat16`), `2^-11` (`float16`). -/
def outU : DType → ℚ
  | .f16 => Math.Float.Format.binary16.u
  | _ => Math.Float.Format.bfloat16.u

/-- The output format's absolute rounding constant: `2^-134` (`bfloat16`), `2^-25` (`float16`). -/
def outEta : DType → ℚ
  | .f16 => Math.Float.Format.binary16.eta
  | _ => Math.Float.Format.bfloat16.eta

/-- The output format's largest value. -/
def outMax : DType → ℚ
  | .f16 => Math.Float.Format.binary16.maxVal
  | _ => Math.Float.Format.bfloat16.maxVal

/-- `c` is the rounding to nearest even of the `float32` value `v` in the output format, up to NaN
payload. -/
def OutIs : (od : DType) → F32 → od.Val → Prop
  | .bf16, v, c => FloatV.NaNEq (FloatV.cvt v) c
  | .f16, v, c => FloatV.NaNEq (FloatV.cvt v) c
  | _, _, _ => False

/-- The bias value of column `j`: with a bias, the finite value `β` of `bias[j]` (element
`j * stride_bias` of its storage); without one, `β = 0`. -/
def BiasIs (d : Inst) (init : Mem) (j : Nat) (β : ℚ) : Prop :=
  if d.hasBias then
    ∃ x : d.od.Val, init.read d.od (d.bias.ptr ↑(j * d.stride_bias)).addr = some x ∧ d.od.val? x = some β
  else β = 0

/-- What the launch leaves in element `(i, j)` of `C`: some value `c` of the output's dtype; and
when row `i` of `A` and column `j` of `B` hold `int8` values `va k`, `vb k` (`k < K`), the scales
`sa` (`scale_a[i, 0]`, or `scale_a[0, 0]` per tensor: `saIdx`) and `sb` (`scale_b[j, 0]` or
`scale_b[0, 0]`: `sbIdx`) are finite, the bias
value is `β` (finite; `0` without a bias), and there is room in `float32` (`Room32`): with
`S = Σ va k · vb k` (exact), `R = sa·sb·S + β` and `E = tol`,
- `c` is the rounding (to nearest even, up to NaN payload) of a finite `float32` value `v` within `E`
  of `R`;
- when `|R| + E` is at most the output format's largest value, `c` is finite and within `E` plus one
  rounding (`(|R| + E)·u + η` of the output format) of `R`. -/
def OutSpec (d : Inst) (init fin : Mem) (i j : Nat) : Prop :=
  ∃ c : d.od.Val, fin.read d.od (d.c.ptr ↑(i * d.stride_cm + j * d.stride_cn)).addr = some c ∧
    ∀ (va vb : Nat → I8) (sa sb : F32) (β : ℚ),
      (∀ k, k < d.K → init.read .i8 (d.a.ptr ↑(i * d.stride_am + k * d.stride_ak)).addr = some (va k)) →
      (∀ k, k < d.K → init.read .i8 (d.b.ptr ↑(k * d.stride_bk + j * d.stride_bn)).addr = some (vb k)) →
      init.read .f32 (d.sa.ptr ↑(saIdx d i)).addr = some sa → sa.isFinite = true →
      init.read .f32 (d.sb.ptr ↑(sbIdx d j)).addr = some sb → sb.isFinite = true →
      BiasIs d init j β →
      Room32 sa.toRat sb.toRat (dotSum va vb d.K) β →
        ∃ v : F32, v.isFinite = true ∧
          |v.toRat - (sa.toRat * sb.toRat * dotSum va vb d.K + β)| ≤
            tol d.hasBias sa.toRat sb.toRat (dotSum va vb d.K) β ∧
          OutIs d.od v c ∧
          (|sa.toRat * sb.toRat * dotSum va vb d.K + β| + tol d.hasBias sa.toRat sb.toRat (dotSum va vb d.K) β ≤
              outMax d.od →
            ∃ q : ℚ, d.od.val? c = some q ∧
              |q - (sa.toRat * sb.toRat * dotSum va vb d.K + β)| ≤
                tol d.hasBias sa.toRat sb.toRat (dotSum va vb d.K) β +
                  (|sa.toRat * sb.toRat * dotSum va vb d.K + β| +
                    tol d.hasBias sa.toRat sb.toRat (dotSum va vb d.K) β) * outU d.od + outEta d.od)

/-- What the launch leaves: every element of the matrix `C` as `OutSpec` says; every byte outside
the matrix `C` unchanged (`A`, `B`, the scales, the bias, and `C`'s storage outside the matrix). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ i j, i < d.M → j < d.N → OutSpec d init fin i j) ∧
    ∀ z, z ∉ d.c.elems (cIdx d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's launches
  meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotI8`: the `int8 × int8 → int32` `tl.dot` (the model's line 167, any
  `precision`, `maxImpreciseAcc = 0`) is exact, `acc + Σ a·b`, when `|acc| + Σ|a·b| ≤ 2^31 - 1`
  (docs/rules/num.md, "Facts the library states"). -/
  dot : Rules.Num.DotI8 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 167) always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .i8, b := .i8, acc := .i32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## The claim -/

/-- The claim to prove: under the assumptions, the launch meets the specification on every
covered instance and input. Not proved here. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous: the assumptions are consistent -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d,
    Platform.sm90ref.dotI8,
    Platform.ref.dotTotal _⟩⟩

/-! ## Coverage: the launches of `SETUP.md`

`hostInst M N K od hasBias saVec sbVec bT heur` is the instance `triton_scaled_mm` builds for a
contiguous `int8` `A` (`M × K`, strides `(K, 1)`), an `int8` `B` (`K × N`) that is contiguous
(strides `(N, 1)`) or, with `bT`, the transposed view of a contiguous `[N, K]` weight (strides
`(1, K)`, the W8A8 linear's layout), the scales (`M` or `1`, `N` or `1` values), an optional bias of
`N` values, and the fresh contiguous `C` it allocates (strides `(N, 1)`), with the heuristic's tile
(`heur`) or the default `(32, 32, 32)`. The buffers sit at fixed, aligned, far-apart addresses; the
claim covers any addresses `Inst.ok` allows, and these only witness the sizes. -/

/-- The tile of `hostInst`: the heuristic's, or the host's default block sizes. -/
def tileOf (heur : Bool) (M N : Nat) : Nat × Nat × Nat := if heur then heurTile M N else (32, 32, 32)

/-- The host's instance. -/
def hostInst (M N K : Nat) (od : DType) (hasBias saVec sbVec bT heur : Bool) : Inst where
  M := M
  N := N
  K := K
  stride_am := K
  stride_ak := 1
  stride_bk := if bT then 1 else N
  stride_bn := if bT then K else 1
  stride_cm := N
  stride_cn := 1
  BM := (tileOf heur M N).1
  BN := (tileOf heur M N).2.1
  BK := (tileOf heur M N).2.2
  saVec := saVec
  sbVec := sbVec
  od := od
  hasBias := hasBias
  a := ⟨2 ^ 40, M * K⟩
  b := ⟨2 ^ 41, K * N⟩
  sa := ⟨2 ^ 43, if saVec then M else 1⟩
  sb := ⟨2 ^ 44, if sbVec then N else 1⟩
  c := ⟨2 ^ 42, M * N⟩
  bias := ⟨2 ^ 45, N⟩
  stride_sa := 1
  stride_sb := 1
  stride_bias := 1

/-- The four tiles of the heuristic, and the default. -/
theorem tileOf_cases (heur : Bool) (M N : Nat) :
    tileOf heur M N = (64, 64, 256) ∨ tileOf heur M N = (64, 128, 256) ∨
      tileOf heur M N = (64, 128, 128) ∨ tileOf heur M N = (128, 128, 128) ∨
      (heur = false ∧ tileOf heur M N = (32, 32, 32)) := by
  unfold tileOf heurTile
  cases heur
  · simp
  · simp only [ite_true]
    split_ifs <;> simp

/-- `t·K + k < M·K` for `t < M`, `k < K`. -/
theorem row_lt {t M K k : Nat} (ht : t < M) (hk : k < K) : t * K + k < M * K := by
  have h : (t + 1) * K ≤ M * K := Nat.mul_le_mul_right K ht
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Row-major indices of distinct elements differ. -/
theorem rm_inj {N i j i' j' : Nat} (hj : j < N) (hj' : j' < N) (h : i * N + j = i' * N + j') :
    i = i' ∧ j = j' := by
  have hi : i = i' := by
    rcases Nat.lt_trichotomy i i' with hlt | heq | hgt
    · have h1 : (i + 1) * N ≤ i' * N := Nat.mul_le_mul_right N hlt
      rw [Nat.add_mul, Nat.one_mul] at h1
      omega
    · exact heq
    · have h1 : (i' + 1) * N ≤ i * N := Nat.mul_le_mul_right N hgt
      rw [Nat.add_mul, Nat.one_mul] at h1
      omega
  subst hi
  exact ⟨rfl, by omega⟩

/-- Two buffers whose byte ranges are in order share no byte. -/
theorem bytes_disjoint {d₁ d₂ : DType} {x : Buf d₁} {y : Buf d₂}
    (h : x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base) :
    Disjoint x.bytes y.bytes := by
  rw [Set.disjoint_left]
  intro z hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- `⌈M / B⌉ ≤ 32771` for `M ≤ 2^20` and a block of at least `32`. -/
theorem cdiv_le {M B : Nat} (hM : M ≤ 2 ^ 20) (hB : 32 ≤ B) (hB' : B ≤ 128) :
    (M + B - 1) / B ≤ 32771 := by
  have h1 : (M + B - 1) / B ≤ (M + B - 1) / 32 := Nat.div_le_div_left hB (by norm_num)
  have h2 : (M + B - 1) / 32 ≤ 32771 := by omega
  omega

/-- The host's instance is covered when its sizes are within these bounds: `K` a multiple of the
tile's `BLOCK_SIZE_K` and below `2^17`, `M`, `N ≤ 2^20` (which bound the grid and fit the address
layout of `hostInst`), and a `bfloat16` or `float16` output. -/
theorem hostInst_ok {M N K : Nat} {od : DType} {hasBias saVec sbVec bT heur : Bool}
    (hod : od = .bf16 ∨ od = .f16) (hM : 0 < M ∧ M ≤ 2 ^ 20) (hN : 0 < N ∧ N ≤ 2 ^ 20)
    (hK : 0 < K ∧ K < 2 ^ 17) (hdiv : K % (tileOf heur M N).2.2 = 0) :
    (hostInst M N K od hasBias saVec sbVec bT heur).ok := by
  obtain ⟨hM0, hM1⟩ := hM
  obtain ⟨hN0, hN1⟩ := hN
  obtain ⟨hK0, hK1⟩ := hK
  -- the tile's sizes
  have ht : 32 ≤ (tileOf heur M N).1 ∧ (tileOf heur M N).1 ≤ 128 ∧
      32 ≤ (tileOf heur M N).2.1 ∧ (tileOf heur M N).2.1 ≤ 128 ∧
      (tileOf heur M N).2.2 ≤ 256 := by
    rcases tileOf_cases heur M N with h | h | h | h | ⟨-, h⟩ <;> rw [h] <;> norm_num
  obtain ⟨hBM0, hBM1, hBN0, hBN1, hBK1⟩ := ht
  have hMK : M * K ≤ 2 ^ 20 * 2 ^ 17 := Nat.mul_le_mul hM1 hK1.le
  have hKN : K * N ≤ 2 ^ 17 * 2 ^ 20 := Nat.mul_le_mul hK1.le hN1
  have hMN : M * N ≤ 2 ^ 20 * 2 ^ 20 := Nat.mul_le_mul hM1 hN1
  have hBKN : (tileOf heur M N).2.2 * N ≤ 256 * 2 ^ 20 := Nat.mul_le_mul hBK1 hN1
  have hgrid : (M + (tileOf heur M N).1 - 1) / (tileOf heur M N).1 *
      ((N + (tileOf heur M N).2.1 - 1) / (tileOf heur M N).2.1) ≤ 32771 * 32771 :=
    Nat.mul_le_mul (cdiv_le hM1 hBM0 hBM1) (cdiv_le hN1 hBN0 hBN1)
  have hal : ∀ i k, i < M → k < K → i * K + k * 1 < M * K := fun i k hi hk => by
    rw [Nat.mul_one]; exact row_lt hi hk
  have hbl : ∀ k j, k < K → j < N →
      k * (if bT then 1 else N) + j * (if bT then K else 1) < K * N := fun k j hk hj => by
    cases bT
    · simp only [Bool.false_eq_true, ite_false, Nat.mul_one]; exact row_lt hk hj
    · simp only [ite_true, Nat.mul_one]
      have := row_lt hj hk
      rw [Nat.mul_comm N K] at this
      omega
  have hcl : ∀ i j, i < M → j < N → i * N + j * 1 < M * N := fun i j hi hj => by
    rw [Nat.mul_one]; exact row_lt hi hj
  set d := hostInst M N K od hasBias saVec sbVec bT heur with hd
  have hcS : d.c.elems (cIdx d) ⊆ d.c.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, j, hi, hj, rfl⟩; exact hcl i j hi hj)
  have haS : d.a.elems (aIdx d) ⊆ d.a.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, k, hi, hk, rfl⟩; exact hal i k hi hk)
  have hbS : d.b.elems (bIdx d) ⊆ d.b.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨k, j, hk, hj, rfl⟩; exact hbl k j hk hj)
  have hsaS : d.sa.elems {e | e < saN d} ⊆ d.sa.bytes :=
    Buf.elems_sub_bytes _ (fun e he => he)
  have hsbS : d.sb.elems {e | e < sbN d} ⊆ d.sb.bytes :=
    Buf.elems_sub_bytes _ (fun e he => he)
  have hbiS : d.bias.elems {e | e < d.N} ⊆ d.bias.bytes :=
    Buf.elems_sub_bytes _ (fun e he => he)
  have hob : od.bytes = 2 := by rcases hod with rfl | rfl <;> rfl
  have hsaL : (if saVec then M else 1) ≤ M := by split <;> omega
  have hsbL : (if sbVec then N else 1) ≤ N := by split <;> omega
  refine
    { od_ok := hod
      tile_ok := ?_
      M_pos := hM0
      N_pos := hN0
      K_pos := hK0
      K_div := hdiv
      K_lt := hK1
      M_le := by simp only [hd, hostInst]; omega
      N_le := by simp only [hd, hostInst]; omega
      grid_lt := by simp only [hd, hostInst]; omega
      strides := by
        simp only [hd, hostInst]
        refine ⟨by omega, by omega, ?_, ?_, by omega, by omega⟩ <;> cases bT <;> simp <;> omega
      steps := by
        simp only [hd, hostInst]
        refine ⟨by omega, ?_⟩
        cases bT <;> simp <;> omega
      c_inj := fun i j i' j' _ hj _ hj' he =>
        rm_inj hj hj' (by simpa only [hd, hostInst, Nat.mul_one] using he)
      a_len := hal
      b_len := hbl
      c_len := hcl
      sa_len := by simp only [hd, hostInst, saN]; exact le_refl _
      sb_len := by simp only [hd, hostInst, sbN]; exact le_refl _
      bias_len := fun _ => le_refl _
      scales_contig := ⟨fun _ => rfl, fun _ => rfl, fun _ => rfl⟩
      bias_nz := fun _ => show (2 : Int) ^ 45 ≠ 0 by norm_num
      c_a := ?_
      c_b := ?_
      c_sa := ?_
      c_sb := ?_
      c_bias := fun _ => ?_ }
  · cases heur
    · exact Or.inr rfl
    · exact Or.inl rfl
  · refine Disjoint.mono hcS haS (bytes_disjoint ?_)
    change (2 : Int) ^ 42 + ((M * N : Nat) : Int) * ((od.bytes : Nat) : Int) ≤ 2 ^ 40 ∨
      (2 : Int) ^ 40 + ((M * K : Nat) : Int) * ((DType.i8.bytes : Nat) : Int) ≤ 2 ^ 42
    rw [hob]; simp only [DType.bytes]; push_cast; omega
  · refine Disjoint.mono hcS hbS (bytes_disjoint ?_)
    change (2 : Int) ^ 42 + ((M * N : Nat) : Int) * ((od.bytes : Nat) : Int) ≤ 2 ^ 41 ∨
      (2 : Int) ^ 41 + ((K * N : Nat) : Int) * ((DType.i8.bytes : Nat) : Int) ≤ 2 ^ 42
    rw [hob]; simp only [DType.bytes]; push_cast; omega
  · refine Disjoint.mono hcS hsaS (bytes_disjoint ?_)
    change (2 : Int) ^ 42 + ((M * N : Nat) : Int) * ((od.bytes : Nat) : Int) ≤ 2 ^ 43 ∨
      (2 : Int) ^ 43 + (((if saVec then M else 1) : Nat) : Int) * ((DType.f32.bytes : Nat) : Int) ≤ 2 ^ 42
    rw [hob]; simp only [DType.bytes]; push_cast; omega
  · refine Disjoint.mono hcS hsbS (bytes_disjoint ?_)
    change (2 : Int) ^ 42 + ((M * N : Nat) : Int) * ((od.bytes : Nat) : Int) ≤ 2 ^ 44 ∨
      (2 : Int) ^ 44 + (((if sbVec then N else 1) : Nat) : Int) * ((DType.f32.bytes : Nat) : Int) ≤ 2 ^ 42
    rw [hob]; simp only [DType.bytes]; push_cast; omega
  · refine Disjoint.mono hcS hbiS (bytes_disjoint ?_)
    change (2 : Int) ^ 42 + ((M * N : Nat) : Int) * ((od.bytes : Nat) : Int) ≤ 2 ^ 45 ∨
      (2 : Int) ^ 45 + ((N : Nat) : Int) * ((od.bytes : Nat) : Int) ≤ 2 ^ 42
    rw [hob]; push_cast; omega

/-- The shapes `(K, N)` of the Llama-class layers SETUP.md lists (vLLM's
`benchmarks/kernels/weight_shapes.py:16-63`: Mistral-7B, Llama-2-7B, -13B, -70B, Llama-3-8B,
Llama-3.1-8B, -405B, Llama-3.3-70B; and the W8A8 test models TinyLlama-1.1B and Llama-3.2-1B), at
tensor-parallel degrees 1, 2, 4 and 8, whose `K` is a multiple of `256`. -/
def llamaShapes : List (Nat × Nat) :=
  [(256, 2048), (512, 2048), (512, 4096), (1024, 2048), (1024, 4096), (1024, 8192), (1280,
   5120), (1792, 4096), (2048, 320), (2048, 384), (2048, 640), (2048, 768), (2048, 1280), (2048,
   1408), (2048, 1536), (2048, 2048), (2048, 2560), (2048, 2816), (2048, 3072), (2048, 4096),
   (2048, 5632), (2048, 8192), (2048, 11264), (2048, 16384), (2560, 5120), (2816, 2048), (3584,
   4096), (3584, 8192), (4096, 768), (4096, 1536), (4096, 2048), (4096, 2752), (4096, 3072),
   (4096, 3584), (4096, 4096), (4096, 5504), (4096, 6144), (4096, 7168), (4096, 8192), (4096,
   11008), (4096, 12288), (4096, 14336), (4096, 16384), (4096, 22016), (4096, 28672), (5120,
   1920), (5120, 3456), (5120, 3840), (5120, 5120), (5120, 6912), (5120, 7680), (5120, 13824),
   (5120, 15360), (5120, 27648), (5632, 2048), (6656, 16384), (6912, 5120), (7168, 4096), (7168,
   8192), (8192, 1280), (8192, 2048), (8192, 2560), (8192, 5120), (8192, 7168), (8192, 8192),
   (8192, 10240), (8192, 14336), (8192, 16384), (8192, 28672), (8192, 57344), (11008, 4096),
   (13312, 16384), (13824, 5120), (14336, 4096), (14336, 8192), (16384, 2304), (16384, 4608),
   (16384, 9216), (16384, 13312), (16384, 16384), (16384, 18432), (16384, 26624), (16384,
   53248), (16384, 106496), (26624, 16384), (28672, 8192), (53248, 16384)]

/-- The shapes of those layers whose `K` is a multiple of `128` but not of `256` (Llama-2-7B's down
projection at TP2, Llama-2-13B's down projection at TP4 and attention output at TP8, TinyLlama's
down projection at TP4): covered for `M > 64`, where the tile's `BLOCK_SIZE_K` is `128`. -/
def llamaShapes128 : List (Nat × Nat) :=
  [(640, 5120), (1408, 2048), (3456, 5120), (5504, 4096)]

/-- **The Llama-class layers**, every batch `1 ≤ M ≤ 16384` (vLLM's int8 GEMM benchmark,
`benchmarks/kernels/benchmark_int8_gemm.py:96`: 1 to 16384), `bfloat16` or `float16` output, either
scale granularity, with or without a bias, either layout of the weight. -/
theorem setup_llama_ok {M K N : Nat} {od : DType} {hasBias saVec sbVec bT : Bool}
    (hod : od = .bf16 ∨ od = .f16) (hM : 1 ≤ M ∧ M ≤ 16384) (hs : (K, N) ∈ llamaShapes) :
    (hostInst M N K od hasBias saVec sbVec bT true).ok := by
  have hall : ∀ p ∈ llamaShapes, p.1 % 256 = 0 ∧ 0 < p.1 ∧ p.1 < 2 ^ 17 ∧ 0 < p.2 ∧ p.2 ≤ 2 ^ 20 := by
    decide
  obtain ⟨h1, h2, h3, h4, h5⟩ := hall _ hs
  refine hostInst_ok hod ⟨by omega, by omega⟩ ⟨h4, h5⟩ ⟨h2, h3⟩ ?_
  rcases tileOf_cases true M N with h | h | h | h | ⟨h, -⟩
  all_goals first | (rw [h]; simp only; omega) | cases h

/-- **The layers of `llamaShapes128`**, for batches `64 < M ≤ 16384`. -/
theorem setup_llama128_ok {M K N : Nat} {od : DType} {hasBias saVec sbVec bT : Bool}
    (hod : od = .bf16 ∨ od = .f16) (hM : 64 < M ∧ M ≤ 16384) (hs : (K, N) ∈ llamaShapes128) :
    (hostInst M N K od hasBias saVec sbVec bT true).ok := by
  have hall : ∀ p ∈ llamaShapes128, p.1 % 128 = 0 ∧ 0 < p.1 ∧ p.1 < 2 ^ 17 ∧ 0 < p.2 ∧ p.2 ≤ 2 ^ 20 := by
    decide
  obtain ⟨h1, h2, h3, h4, h5⟩ := hall _ hs
  refine hostInst_ok hod ⟨by omega, by omega⟩ ⟨h4, h5⟩ ⟨h2, h3⟩ ?_
  have : (tileOf true M N).2.2 = 128 := by
    simp only [tileOf, heurTile, ite_true]
    split_ifs <;> first | omega | rfl
  rw [this]; exact h1

/-- **vLLM's test** (`tests/kernels/quantization/test_triton_scaled_mm.py:66-123`), its shapes
whose `K` is a multiple of the tile's `BLOCK_SIZE_K`: `(M, N, K) = (64, 971, 1024)` and
`(512, 20486, 1024)`, every flag, either output dtype, either layout. -/
theorem setup_vllm_test_ok {M N K : Nat} {od : DType} {hasBias saVec sbVec bT : Bool}
    (hod : od = .bf16 ∨ od = .f16) (hs : (M, N, K) ∈ [(64, 971, 1024), (512, 20486, 1024)]) :
    (hostInst M N K od hasBias saVec sbVec bT true).ok := by
  simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hs
  rcases hs with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩
  · exact hostInst_ok hod (by norm_num) (by norm_num) (by norm_num) (by decide)
  · exact hostInst_ok hod (by norm_num) (by norm_num) (by norm_num) (by decide)

/-- **vLLM's descriptor test, plain path** (`test_triton_scaled_mm.py:150-178`, `use_td=False`):
`(M, N, K) = (1, 4096, 4096)`, `(64, 4096, 4096)`, `(256, 2048, 4096)`. -/
theorem setup_vllm_plain_ok {M N K : Nat} {od : DType} {hasBias saVec sbVec bT : Bool}
    (hod : od = .bf16 ∨ od = .f16)
    (hs : (M, N, K) ∈ [(1, 4096, 4096), (64, 4096, 4096), (256, 2048, 4096)]) :
    (hostInst M N K od hasBias saVec sbVec bT true).ok := by
  simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hs
  rcases hs with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩
  · exact hostInst_ok hod (by norm_num) (by norm_num) (by norm_num) (by decide)
  · exact hostInst_ok hod (by norm_num) (by norm_num) (by norm_num) (by decide)
  · exact hostInst_ok hod (by norm_num) (by norm_num) (by norm_num) (by decide)

/-- **vLLM's rounding test with `use_heuristic=False`** (`test_triton_scaled_mm.py:126-142`):
`(M, N, K) = (1, 16, 32)`, tile `(32, 32, 32)`, per-tensor scales, a `bfloat16` bias and output,
the contiguous `B` of `expand(...).contiguous()`. -/
def setup_bias_rounding : Inst := hostInst 1 16 32 .bf16 true false false false false
theorem setup_bias_rounding_ok : setup_bias_rounding.ok :=
  hostInst_ok (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by decide)

/-! ## Not vacuous: a covered instance and memory -/

/-- The all-zero memory: every byte present and `0`. -/
def zeroMem : Mem := fun _ => some (0 : TVA.Math.Bytes.Byte)

/-- A buffer at one of `hostInst`'s addresses, inside the address space, is valid in the all-zero
memory. -/
theorem zeroValid {d : DType} (b : Buf d) (hbase : ∃ k, 3 ≤ k ∧ b.base = 2 ^ k)
    (hbound : b.base + b.len * d.bytes ≤ 2 ^ 64) : b.Valid zeroMem := by
  obtain ⟨k, hk, hb⟩ := hbase
  have hdiv : ∀ m : Nat, m ∣ 8 → (b.base) % (m : Int) = 0 := by
    intro m hm
    rw [hb]
    have h8 : (8 : Int) ∣ 2 ^ k := by
      have : (2 : Int) ^ 3 ∣ 2 ^ k := pow_dvd_pow 2 hk
      simpa using this
    exact Int.emod_eq_zero_of_dvd ((Int.natCast_dvd_natCast.mpr hm).trans h8)
  have hbytes : d.bytes ∣ 8 := by cases d <;> decide
  refine ⟨by rw [hb]; positivity, hbound, hdiv _ hbytes, fun x _ => by simp [zeroMem]⟩

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_bias_rounding, zeroMem, setup_bias_rounding_ok, ?_, ?_, ?_, ?_, ?_, fun _ => ?_⟩
  all_goals
    refine zeroValid _ ⟨_, ?_, rfl⟩ ?_
    all_goals norm_num [setup_bias_rounding, hostInst, DType.bytes]
  · change (2 : Int) ^ 42 + ((1 * 16 : Nat) : Int) * 2 ≤ 2 ^ 64
    norm_num
  · change (2 : Int) ^ 45 + ((16 : Nat) : Int) * 2 ≤ 2 ^ 64
    norm_num

end Claims.ScaledMmInt8
