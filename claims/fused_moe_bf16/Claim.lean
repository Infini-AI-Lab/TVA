import TVA

/-!
# `fused_moe_bf16`: the prepared claim

Source: `source/fused_moe.py` (vLLM's `fused_moe_kernel`, unquantised
`bfloat16` path, with the configuration vLLM picks on an H200). The kernel is `fused_moe_kernel`
(lines 109–420) with the helper `write_zeros_to_output` (lines 89–106) it inlines; it reads the
module-level names `triton` and `tl` (lines 85–86). Its hosts, vLLM's two expert GEMMs
`fused_moe_gemm1` (lines 704–766) and `fused_moe_gemm2` (lines 769–833), launch it through
`invoke_fused_moe_triton_kernel` (lines 609–690) on `bfloat16` matrices `A` (`M × K`: the hidden
states, or for the second GEMM the `M·top_k × K` `intermediate_cache2` with the kernel's
`top_k = 1`), `B` (`E × N × K`), into a fresh `bfloat16` `C`, with the routing weights
`topk_weights` (`float32`) and the expert assignment of `_prepare_expert_assignment` (lines
573–606): `sorted_token_ids`, `expert_ids`, `num_tokens_post_padded` from `moe_align_block_size`
(lines 423–482), or, for decode-sized batches, the naive assignment (`naive_block_assignment`:
no `sorted_token_ids`, `expert_ids = topk_ids.view(-1)`), with the tile `try_get_optimal_moe_config`
(lines 560–570) picks: vLLM's tuned H200 file's entry for the batch (`source/configs/`), else
`get_default_config` (lines 485–520).

This file holds the model (`write_zeros_to_output`, `fused_moe_kernel`, `model`, `launch`), the
conditions (`Inst`, `Inst.ok`, `Pre`), the specification (`Post`, `Spec`), the assumptions
(`Assumptions`) and the claim `Statement`, which it does not prove. It proves `consistent`,
`satisfiable`, and that each launch `SETUP.md` names is covered (`setup_*_ok`).

The routing weights need be finite only when `MUL_ROUTED_WEIGHT` (they are read only then);
`Pre` bounds the entries of `sorted_token_ids` below the padded total only from below (an entry at
or above `M · top_k` is masked like the padding) and says nothing of the entries past the padded
total (they are never read). With the naive assignment the entries are those the kernel builds
(`entryOf`). See `README.md`.
-/

namespace Claims.FusedMoe

open TVA

/-! ## The model -/

/-- `write_zeros_to_output` (source lines 89–106). Its `compute_type` is the kernel's
`compute_type: tl.constexpr`, which holds the dtype `tl.bfloat16` on every covered launch (the hosts
set it from the activations' dtype) and means at each use what `tl.bfloat16` means (SEMANTICS.md,
"A dtype as a `tl.constexpr` argument"), so the model writes `.bf16` and has no `compute_type`
parameter. `BLOCK_SIZE_M`, `BLOCK_SIZE_N` are implicit Lean parameters (the source's parameters
`BLOCK_SIZE_M`, `BLOCK_SIZE_N`). -/
def write_zeros_to_output (c_ptr : Ptr .bf16) (stride_cm stride_cn pid_n N : I32)
    {BLOCK_SIZE_M BLOCK_SIZE_N : Nat}
    (offs_token : Tensor [BLOCK_SIZE_M] I64) (token_mask : Tensor [BLOCK_SIZE_M] Bool) :
    Prog Unit := do
  -- line 102: `tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)`, `compute_type` the
  -- dtype `tl.bfloat16`: `bfloat16` `+0.0`
  let accumulator := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .bf16
  -- line 103: the constexpr built in `int32`, the wrapping product, the `int32` `tl.arange`
  let offs_cn : Tensor [BLOCK_SIZE_N] I32 :=
    pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
  -- line 104: `stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]`: the `int32`
  -- scalar `stride_cm` widened to `int64` (the other operand is), the wrapping `int64` product
  -- and sum with the `int32` product `stride_cn * offs_cn[None, :]` widened; the pointer
  -- additions, in elements
  let c_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] (Ptr .bf16) :=
    c_ptr + (← tl.convert .i32 .i64 (Tensor.scalar stride_cm)).item * offs_token.col BLOCK_SIZE_N +
      (← tl.convert .i32 .i64 (offs_cn.row BLOCK_SIZE_M * stride_cn))
  -- line 105: `token_mask[:, None] & (offs_cn[None, :] < N)`: the signed `int32` comparison with
  -- the scalar `N`, then `&` on `int1`
  let c_mask := (token_mask.col BLOCK_SIZE_N).and ((offs_cn.row BLOCK_SIZE_M).ltS N)
  -- line 106: `accumulator` is already `bfloat16`, the pointer's dtype
  tl.store c_ptrs accumulator (some c_mask)

/-- `fused_moe_kernel` (source lines 109–420) on the launches the hosts make: the compile-time
conditions `USE_TD`, `SWAP_AB`, the quantisation flags and `HAS_BIAS` are `False` on every covered
launch, so the model has no parameters for them (the model map drops their lines as `config`);
`compute_type`, a `tl.constexpr` holding the dtype `tl.bfloat16`, is written `.bf16` (SEMANTICS.md,
"A dtype as a `tl.constexpr` argument"). `naive_block_assignment` and `MUL_ROUTED_WEIGHT` are Lean
parameters (`Bool`; the model has both branches of each); `top_k` and the block sizes are Lean
parameters (`Nat`). With `naive_block_assignment` the host passes `sorted_token_ids_ptr = None`, a
compile-time `None` that only the branch not taken names; the model's `sorted_token_ids_ptr` is
then unused.

The pointers: `a_ptr`, `b_ptr`, `c_ptr` `*bfloat16`; `topk_weights_ptr` `*float32`;
`sorted_token_ids_ptr`, `expert_ids_ptr`, `num_tokens_post_padded_ptr` `*int32`. `N`, `K`, `EM`,
`num_valid_tokens` and the seven strides the model uses are `int32`: an integer argument in
`[-2^31, 2^31)` other than `1` is a runtime `int32`, and each is below `2^31` on covered
instances (`Inst.ok`); at `1` an argument is the compile-time constant `1`, which gives the same
model at every use: each use of `EM` is `tl.cdiv`'s `x` with a `tl.constexpr` divisor (at `EM =
1`, Python's `(1 + 127) // 128` is `1`, which the name `num_pid_m` then holds as an `int32`
scalar tensor, the value the `int32` computation gives); the same for `K` (whose `tl.cdiv` feeds
`range`, giving the Python bounds `0`, `1` and the `int32` induction variable of one trip) and
`N` (whose `tl.cdiv` is as `EM`'s, and whose other uses are comparisons with, and the widening
`%` by, `int32`/`int64` tensors, which build the constant in the tensor's dtype and compute the
same value); `num_valid_tokens` and the strides are only compared with, multiplied by or added
to tensors (or are a value of `tl.where`, line 222), which build the constant in the tensor's
dtype. The host passes `num_valid_tokens = A.size(0) · top_k` (`num_tokens`). -/
def fused_moe_kernel (a_ptr b_ptr c_ptr : Ptr .bf16) (topk_weights_ptr : Ptr .f32)
    (sorted_token_ids_ptr expert_ids_ptr num_tokens_post_padded_ptr : Ptr .i32)
    (N K EM num_valid_tokens : I32)
    (stride_am stride_ak stride_be stride_bk stride_bn stride_cm stride_cn : I32)
    (naive_block_assignment : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M top_k : Nat)
    (MUL_ROUTED_WEIGHT : Bool) : Kernel where
  body ctx := do
    -- line 198: axis 0 of the 1-D grid, an `int32`
    let pid := tl.program_id ctx 0
    -- line 199: `tl.cdiv(EM, BLOCK_SIZE_M)` is `(EM + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M`: the
    -- constexpr `BLOCK_SIZE_M - 1` is computed by Python, then built in `int32` (an operand of
    -- `+` with the `int32` `EM`); the wrapping `int32` `+`, then the truncating `//` (`tl.idiv`)
    let num_pid_m := (← tl.idiv (Tensor.scalar (EM + (IntV.ofNat (BLOCK_SIZE_M - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_M : I32))).item
    -- line 200: the same with `N`, `BLOCK_SIZE_N`
    let num_pid_n := (← tl.idiv (Tensor.scalar (N + (IntV.ofNat (BLOCK_SIZE_N - 1) : I32)))
      (Tensor.scalar (IntV.ofNat BLOCK_SIZE_N : I32))).item
    -- line 201: the constexpr times the `int32` scalar, built in `int32`; wrapping `*`
    let num_pid_in_group := (IntV.ofNat GROUP_SIZE_M : I32) * num_pid_n
    -- line 202: `int32` `//` (`tl.idiv`)
    let group_id := (← tl.idiv (Tensor.scalar pid) (Tensor.scalar num_pid_in_group)).item
    -- line 203: `int32` scalar times the constexpr, built in `int32`
    let first_pid_m := group_id * (IntV.ofNat GROUP_SIZE_M : I32)
    -- line 204: Python's `min` with a tensor argument is `tl.minimum`, which makes the constexpr
    -- an `int32` tensor of its own type; on `int32` it is the exact `tl.iminimum`
    let group_size_m := (tl.iminimum (Tensor.scalar (num_pid_m - first_pid_m))
      (Tensor.scalar (IntV.ofNat GROUP_SIZE_M : I32))).item
    -- line 205: two `int32` `%` (`tl.irem`), the inner one first, then the wrapping `+`
    let pid_m := first_pid_m + (← tl.irem (Tensor.scalar (← tl.irem (Tensor.scalar pid)
      (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
    -- line 206: `pid % num_pid_in_group` computed again (the source writes it again), then `//`
    let pid_n := (← tl.idiv (Tensor.scalar (← tl.irem (Tensor.scalar pid)
      (Tensor.scalar num_pid_in_group)).item) (Tensor.scalar group_size_m)).item
    -- line 214: `tl.arange(0, BLOCK_SIZE_M).to(tl.int64)`: the exact widening (`tl.convert`)
    let offs : Tensor [BLOCK_SIZE_M] I64 ← tl.convert .i32 .i64 (tl.arange 0 BLOCK_SIZE_M)
    -- line 215: a scalar load of the `*int32` pointer
    let num_tokens_post_padded : I32 ← tl.load1 num_tokens_post_padded_ptr
    -- lines 216–217: a runtime `if` whose taken branch returns: the programs with
    -- `pid_m * BLOCK_SIZE_M >= num_tokens_post_padded` run nothing after it
    if IntV.le num_tokens_post_padded (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)) then
      return ()
    else do
      -- lines 218–226: `if not naive_block_assignment:` is a compile-time condition (the host
      -- passes `sorted_token_ids is None`, line 684): the model has both branches
      let offs_token : Tensor [BLOCK_SIZE_M] I32 ← if !naive_block_assignment then do
          -- line 219: `offs_token_id = pid_m * BLOCK_SIZE_M + offs`: the `int32` scalar widened
          -- to `int64` (the other operand is), then the wrapping `int64` sum
          let offs_token_id : Tensor [BLOCK_SIZE_M] I64 :=
            (← tl.convert .i32 .i64 (Tensor.scalar (pid_m * (IntV.ofNat BLOCK_SIZE_M : I32)))).item + offs
          -- line 220: an unmasked load of the block's `BLOCK_SIZE_M` entries of `sorted_token_ids`
          tl.load (sorted_token_ids_ptr + offs_token_id)
        else
          -- lines 222–226: `tl.where(offs == 0, pid_m, num_valid_tokens)`: the Python `0`, an
          -- operand of a comparison, is an `int32` tensor of its own type promoted to the `int64`
          -- of `offs` (the value `0`), then `==`; the two values are `int32` scalars (`pid_m`, and
          -- `num_valid_tokens`, at `1` the constant `1` built in `int32`), so the result is the
          -- `int32` block, both broadcast to `[BLOCK_SIZE_M]`
          pure (tl.where (offs.eq (bc (IntV.ofInt 0 : I64))) (bc pid_m) (bc num_valid_tokens))
      -- lines 227–229: `offs_token.to(tl.int64)`: the exact widening
      let offs_token : Tensor [BLOCK_SIZE_M] I64 ← tl.convert .i32 .i64 offs_token
      -- line 231: `offs_token < num_valid_tokens`: the `int32` scalar widened to `int64`, the
      -- signed comparison
      let token_mask : Tensor [BLOCK_SIZE_M] Bool :=
        offs_token.ltS (← tl.convert .i32 .i64 (Tensor.scalar num_valid_tokens)).item
      -- line 233: a scalar load, then `.to(tl.int64)`, the exact widening
      let e32 : I32 ← tl.load1 (expert_ids_ptr + pid_m)
      let off_experts : I64 := (← tl.convert .i32 .i64 (Tensor.scalar e32)).item
      -- lines 234–250: a runtime `if` whose taken branch stores zeros and returns
      if IntV.eq off_experts (IntV.ofInt (-1 : Int) : I64) then
        -- lines 238–249: `write_zeros_to_output` inlined (its body, the call's arguments bound
        -- to its parameters); lines 245–248 pass `BLOCK_SIZE_M`, `BLOCK_SIZE_N` and
        -- `compute_type = tl.bfloat16`
        write_zeros_to_output c_ptr stride_cm stride_cn pid_n N
          (BLOCK_SIZE_M := BLOCK_SIZE_M) (BLOCK_SIZE_N := BLOCK_SIZE_N) offs_token token_mask
        -- line 250
        return ()
      else do
        -- line 252: `(pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N`: the
        -- `int32` scalar product widened to `int64`, the `tl.arange` widened, the wrapping
        -- `int64` sum, then the truncating `int64` `%` with `N` widened (`tl.irem`)
        let n64 : I64 := (← tl.convert .i32 .i64 (Tensor.scalar N)).item
        let offs_bn : Tensor [BLOCK_SIZE_N] I64 ←
          tl.irem ((← tl.convert .i32 .i64 (Tensor.scalar (pid_n * (IntV.ofNat BLOCK_SIZE_N : I32)))).item +
            (← tl.convert .i32 .i64 (tl.arange 0 BLOCK_SIZE_N)))
            (bc n64)
        -- line 253
        let offs_k : Tensor [BLOCK_SIZE_K] I32 := tl.arange 0 BLOCK_SIZE_K
        -- line 255: `tl.static_assert(not (USE_TD and SWAP_AB))`: both are `False` on every
        -- covered launch, so it holds and adds nothing
        -- lines 256–271: not visited (`USE_TD` is `False`, a compile-time condition)
        -- lines 272–280: not visited (`SWAP_AB` is `False`)
        -- lines 281–284: `a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
        -- offs_k[None, :] * stride_ak)`: the `int64` `//` by the constexpr built in `int64`
        -- (truncating), the `int32` scalar `stride_am` widened, the wrapping `int64` product;
        -- the `int32` product `offs_k[None, :] * stride_ak` widened; the wrapping `int64` sum;
        -- the pointer addition, in elements. The source divides `offs_token[:, None]`, a block
        -- of one column; the model divides `offs_token` and broadcasts, the same values
        let a_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] (Ptr .bf16) :=
          a_ptr + (((← tl.idiv offs_token (bc (IntV.ofNat top_k : I64))) *
              (← tl.convert .i32 .i64 (Tensor.scalar stride_am)).item).col BLOCK_SIZE_K +
            (← tl.convert .i32 .i64 (offs_k.row BLOCK_SIZE_M * stride_ak)))
        -- lines 285–289: `b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] *
        -- stride_bk + offs_bn[None, :] * stride_bn)`: the `int64` scalar product; the `int32`
        -- product `offs_k[:, None] * stride_bk` widened; the `int64` product of the `int64` row
        -- of `offs_bn` with the `int32` scalar `stride_bn` widened (`offs_bn` is already
        -- `int64`, so the source's product promotes); two wrapping `int64` sums; the pointer
        -- additions
        let b_ptrs : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] (Ptr .bf16) :=
          b_ptr + off_experts * (← tl.convert .i32 .i64 (Tensor.scalar stride_be)).item +
            ((← tl.convert .i32 .i64 (offs_k.col BLOCK_SIZE_N * stride_bk)) +
              offs_bn.row BLOCK_SIZE_K *
                (← tl.convert .i32 .i64 (Tensor.scalar stride_bn)).item)
        -- lines 290–320: not visited (`use_int8_w8a16`, `use_fp8_w8a8`, `use_int8_w8a8`,
        -- `HAS_BIAS` are `False`, compile-time conditions)
        -- lines 326–329: `SWAP_AB` is `False`, so line 329: `float32` `+0.0` in every element
        let accumulator := tl.zeros [BLOCK_SIZE_M, BLOCK_SIZE_N] .f32
        -- line 330: `for k in range(0, tl.cdiv(K, BLOCK_SIZE_K))`: the bound
        -- `(K + (BLOCK_SIZE_K - 1)) // BLOCK_SIZE_K` in `int32`, evaluated once, before the loop
        -- (a two-argument `range`); the `int32` bounds `0`, the implicit step `1` and induction
        -- variable `k`; the loop carries `a_ptrs`, `b_ptrs`, `accumulator` (assigned in the body,
        -- defined before it)
        let (_a_ptrs, _b_ptrs, accumulator) ← tl.forRange (IntV.ofInt 0 : I32)
            (← tl.idiv (Tensor.scalar (K + (IntV.ofNat (BLOCK_SIZE_K - 1) : I32)))
              (Tensor.scalar (IntV.ofNat BLOCK_SIZE_K : I32))).item
            (IntV.ofInt 1 : I32) (a_ptrs, b_ptrs, accumulator)
            fun k (a_ptrs, b_ptrs, accumulator) => do
          -- lines 333–341: not visited (`USE_TD`, `SWAP_AB` are `False`)
          -- lines 342–346: the mask `token_mask[:, None] & (offs_k[None, :] < K - k *
          -- BLOCK_SIZE_K)` (the signed `int32` comparison broadcast to the pointers' shape, then
          -- `&` on `int1`); `other=0.0`: the `float32` `+0.0` (a tensor of its own type),
          -- broadcast and converted to the pointer's dtype `bfloat16` (`tl.convert`, an implicit
          -- conversion)
          let a ← tl.load a_ptrs
            (some ((token_mask.col BLOCK_SIZE_K).and
              ((offs_k.row BLOCK_SIZE_M).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32)))))
            (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_M, BLOCK_SIZE_K] (FloatV.litF32 (FloatV.ofPat 0)))))
          -- line 347: the mask `offs_k[:, None] < K - k * BLOCK_SIZE_K`; `other=0.0` as above
          let b ← tl.load b_ptrs
            (some ((offs_k.col BLOCK_SIZE_N).ltS (K - k * (IntV.ofNat BLOCK_SIZE_K : I32))))
            (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_SIZE_K, BLOCK_SIZE_N] (FloatV.litF32 (FloatV.ofPat 0)))))
          -- lines 349–372: not visited (the quantisation flags are `False`)
          -- line 373: `accumulator += tl.dot(a, b)`: a source `c + tl.dot(a, b)` whose dot has
          -- the zero accumulator (the default), so the combine pass may rewrite it to
          -- `tl.dot(a, b, c)` (`tl.dot_add`); no `input_precision` (the default `tf32`) and no
          -- `max_num_imprecise_acc` (`0`); the operands are `bfloat16`, the result `float32`
          let accumulator ← tl.dot_add
            { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
            a b accumulator
          -- lines 374–377: `USE_TD` is `False`, so the pointers advance: `x += y` is `x = x +
          -- y`; the constexpr `BLOCK_SIZE_K` built in `int32` as an operand of `*` with the
          -- `int32` scalar stride, then added to every pointer of the block
          let a_ptrs := a_ptrs +
            (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_ak) : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_K] I32)
          let b_ptrs := b_ptrs +
            (bc ((IntV.ofNat BLOCK_SIZE_K : I32) * stride_bk) : Tensor [BLOCK_SIZE_K, BLOCK_SIZE_N] I32)
          return (a_ptrs, b_ptrs, accumulator)
        -- lines 379–397: not visited (`SWAP_AB`, the quantisation flags, `HAS_BIAS` are `False`)
        -- lines 403–409: `MUL_ROUTED_WEIGHT` is a compile-time condition: the model has both
        -- branches. The load's `other=0`: the Python `0` is an `int32` tensor of its own type,
        -- converted to the pointer's dtype `float32` (`tl.convert`, an implicit conversion); the
        -- product `accumulator * moe_weight[:, None]` is an IEEE `float32` product (no product
        -- feeds it)
        if MUL_ROUTED_WEIGHT then
          do
            let moe_weight : Tensor [BLOCK_SIZE_M] F32 ←
              tl.load (topk_weights_ptr + offs_token) (some token_mask)
                (some (← tl.convert .i32 .f32 (tl.full [BLOCK_SIZE_M] (IntV.ofInt 0))))
            let accumulator ← tl.mul accumulator (moe_weight.col BLOCK_SIZE_N)
            -- line 413: `accumulator.to(compute_type)` (`compute_type` the dtype `tl.bfloat16`):
            -- `float32` to `bfloat16`, one rounding to nearest even (`tl.cvt`)
            let c : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] BF16 ← tl.cvt accumulator
            -- lines 417–420: the `int32` scalar `stride_cm` widened (the other operand of the
            -- product is `int64`), the wrapping products and sums, the pointer additions; the
            -- mask as line 105
            let offs_cn : Tensor [BLOCK_SIZE_N] I32 :=
              pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
            let c_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] (Ptr .bf16) :=
              c_ptr + (← tl.convert .i32 .i64 (Tensor.scalar stride_cm)).item * offs_token.col BLOCK_SIZE_N +
                (← tl.convert .i32 .i64 (offs_cn.row BLOCK_SIZE_M * stride_cn))
            let c_mask := (token_mask.col BLOCK_SIZE_N).and ((offs_cn.row BLOCK_SIZE_M).ltS N)
            tl.store c_ptrs c (some c_mask)
        else
          do
            -- line 413
            let c : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] BF16 ← tl.cvt accumulator
            -- lines 417–420
            let offs_cn : Tensor [BLOCK_SIZE_N] I32 :=
              pid_n * (IntV.ofNat BLOCK_SIZE_N : I32) + tl.arange 0 BLOCK_SIZE_N
            let c_ptrs : Tensor [BLOCK_SIZE_M, BLOCK_SIZE_N] (Ptr .bf16) :=
              c_ptr + (← tl.convert .i32 .i64 (Tensor.scalar stride_cm)).item * offs_token.col BLOCK_SIZE_N +
                (← tl.convert .i32 .i64 (offs_cn.row BLOCK_SIZE_M * stride_cn))
            let c_mask := (token_mask.col BLOCK_SIZE_N).and ((offs_cn.row BLOCK_SIZE_M).ltS N)
            tl.store c_ptrs c (some c_mask)

/-! ## The conditions -/

/-- What the host chooses: one launch of `fused_moe_kernel` through `fused_moe_gemm1` or
`fused_moe_gemm2`. The strides the hosts fix (`A` contiguous, `B.stride(-1) = 1`, `C` freshly
allocated as a contiguous `[M, top_k, N]`, or `[M_tok, top_k_ids, N]` for the second GEMM, the same
storage) are the model's values; `B`'s other two strides stay instance variables. For the second
GEMM, `M` is `A.size(0) = M_tok · top_k_ids` and `top_k` is the kernel's `1`, so its flat index
`f = t · top_k_ids + j` is also the row of `A` and of `C`. -/
structure Inst where
  /-- `M`: the rows of `A` (`A.shape[0]`: the tokens, or `M_tok · top_k_ids` for the second
  GEMM). -/
  M : Nat
  /-- `top_k`: the kernel's constexpr `top_k` (`topk_ids.shape[1]` for the first GEMM, `1` for
  the second). -/
  top_k : Nat
  /-- `E`: the local experts (`B.shape[0]`, `w1`'s first dimension). -/
  E : Nat
  /-- `N`: the output features (`B.shape[1]`). -/
  N : Nat
  /-- `K`: the hidden size (`A.shape[1]`, `B.shape[2]`). -/
  K : Nat
  /-- `EM`: the launch's `EM` (`sorted_token_ids.size(0)`, possibly `min`'d, line 510). -/
  EM : Nat
  /-- `stride_be` (`B.stride(0)`), in elements. -/
  stride_be : Nat
  /-- `stride_bn` (`B.stride(1)`), in elements. -/
  stride_bn : Nat
  /-- `MUL_ROUTED_WEIGHT` (`apply_router_weight_on_input` for the first GEMM, its negation for
  the second). -/
  rw : Bool
  /-- `naive_block_assignment` (`sorted_token_ids is None`: `M_tok · top_k_ids · 4 ≤ E_global`
  and no expert map, `_prepare_expert_assignment`). -/
  naive : Bool
  /-- `BLOCK_SIZE_M` (`get_default_config`). -/
  BM : Nat
  /-- `BLOCK_SIZE_N`. -/
  BN : Nat
  /-- `BLOCK_SIZE_K`. -/
  BK : Nat
  /-- `GROUP_SIZE_M`. -/
  GM : Nat
  /-- `a_ptr`: the storage of `A`, `bfloat16`; element `(t, k)` is element `t * K + k` of it
  (`A` is contiguous, `stride_am = K`, `stride_ak = 1`). -/
  a : Buf .bf16
  /-- `b_ptr`: the storage of `B`, `bfloat16`; element `(e, c, k)` is element
  `e * stride_be + c * stride_bn + k` of it (`stride_bk = 1`). -/
  b : Buf .bf16
  /-- `c_ptr`: the storage of `C`, `bfloat16`; element `(t, j, c)` is element
  `(t * top_k + j) * N + c` of it (`C` is a contiguous `[M, top_k, N]`, so `stride_cm =
  C.stride(1) = N`, `stride_cn = 1`). -/
  c : Buf .bf16
  /-- `topk_weights_ptr`: the storage of the routing weights, `float32`, indexed flat by the
  entries of `sorted_token_ids` (the kernel uses no stride of it). -/
  w : Buf .f32
  /-- `topk_weights.size(1)` (`= topk_ids.size(1)`: `top_k` for the first GEMM, `top_k_ids`
  for the second). The flat indices the expert assignment builds are `t · w_cols + j`. -/
  w_cols : Nat
  /-- `topk_weights.stride(0)`, in elements. The kernel reads the weight of flat index
  `f = t · w_cols + j` at element `f` of the storage (line 404, `topk_weights_ptr + offs_token`),
  which is `topk_weights[t, j]` only when `stride(0) = w_cols` (`Inst.ok.w_contig`); the host
  asserts only `stride(1) = 1` (line 623). -/
  w_stride0 : Nat
  /-- `sorted_token_ids_ptr`: the storage of `sorted_token_ids`, `int32` (with `naive`, the host
  passes `None`; the kernel never reads it, and the buffer stands for nothing). -/
  s : Buf .i32
  /-- `expert_ids_ptr`: the storage of `expert_ids`, `int32` (with `naive`, `topk_ids.view(-1)`:
  the expert of flat index `f` is its element `f`). -/
  eid : Buf .i32
  /-- `num_tokens_post_padded_ptr`: the storage of `num_tokens_post_padded` (`[1]`), `int32`. -/
  ntpb : Buf .i32

/-- The flat index of row `(t, j)` of `C`: what `sorted_token_ids` holds for a routed row. -/
def flat (d : Inst) (t j : Nat) : Nat := t * d.top_k + j

/-- `num_valid_tokens` (line 627, `A.size(0) · top_k`): the flat index past the last routed
row. -/
def nv (d : Inst) : Nat := d.M * d.top_k

/-- `stride_cm`: the rows of `C` are `N` elements apart (`C` is a contiguous
`[M, top_k, N]`, whose `stride(1)` is `N` — the host passes `C.stride(1)`, line 666). -/
def stride_cm (d : Inst) : Nat := d.N

/-- The elements of `C`'s storage that make up the matrix `C` (`M × top_k × N`). -/
def cIdx (d : Inst) : Set Nat :=
  {e | ∃ f c, f < nv d ∧ c < d.N ∧ e = f * stride_cm d + c}

/-- The value the kernel loads from `num_tokens_post_padded` (line 215), as a natural number;
`0` when the memory does not hold a nonnegative one (`Pre` does). With `naive` the host's is
`M · top_k · BLOCK_SIZE_M` (line 594). -/
noncomputable def ntpOf (d : Inst) (init : Mem) : Nat :=
  match init.read .i32 (d.ntpb.ptr 0).addr with
  | some v => if 0 ≤ v.toInt then v.toInt.toNat else 0
  | none => 0

/-- The flat index of position `pos` (lane `pos % BLOCK_SIZE_M` of block `pos / BLOCK_SIZE_M`):
without `naive`, the value the kernel loads from `sorted_token_ids` at `pos` (line 220), as an
integer, or the padding value `M · top_k` (which the masks hold) when the memory does not hold one
(`Pre` does, below `ntpOf d init`); with `naive`, the value lines 222–226 build: the block's index
in its first lane, `num_valid_tokens` in the others. -/
noncomputable def entryOf (d : Inst) (init : Mem) (pos : Nat) : Int :=
  if d.naive then (if pos % d.BM = 0 then ((pos / d.BM : Nat) : Int) else nv d)
  else if pos < d.s.len then
    match init.read .i32 (d.s.ptr (pos : Int)).addr with
    | some v => v.toInt
    | none => nv d
  else nv d

/-- The value the kernel loads from `expert_ids` at block `pid_m` (line 233), as an integer;
`0` when the memory does not hold one (`Pre` does, below `ntpOf d init / BLOCK_SIZE_M`). -/
noncomputable def eOf (d : Inst) (init : Mem) (pid_m : Nat) : Int :=
  match init.read .i32 (d.eid.ptr (pid_m : Int)).addr with
  | some v => v.toInt
  | none => 0

/-- The tiles `(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)` the hosts pick on an H200
(`try_get_optimal_moe_config`, lines 560–570): the 56 tiles of vLLM's 22 tuned `bfloat16` files
for `NVIDIA_H200` (`source/configs/`), which include the seven of `get_default_config` (lines
485–520: `(16, 64, 128)` for `M ≤ 32`, `(32, 64, 128)` for `M ≤ 64`, `(32, 128, 64)` for `M ≤ 96`,
`(64, 128, 64)` for `M ≤ 512`, `(128, 128, 64)` above, `GROUP_SIZE_M = 16` when `M // E > 128`).
`BLOCK_SIZE_K` is `64`, `128` or `256`, so each trip of the loop is 1, 2 or 4 chunks of 64. -/
def cfgs : List (Nat × Nat × Nat × Nat) :=
  [(16, 32, 64, 1), (16, 32, 128, 1), (16, 32, 128, 64), (16, 64, 64, 1), (16, 64, 64, 16), (16, 64, 64, 32),
    (16, 64, 64, 64), (16, 64, 128, 1), (16, 64, 128, 16), (16, 64, 128, 32), (16, 64, 128, 64), (16, 64, 256, 1),
    (16, 64, 256, 16), (16, 128, 64, 1), (16, 128, 64, 16), (16, 128, 64, 32), (16, 128, 64, 64), (16, 128, 128, 1),
    (16, 128, 128, 16), (16, 128, 128, 32), (16, 128, 128, 64), (16, 128, 256, 1), (16, 128, 256, 16), (16, 128, 256, 32),
    (16, 256, 128, 1), (16, 256, 128, 16), (32, 64, 64, 1), (32, 64, 64, 32), (32, 64, 128, 1), (32, 128, 64, 1),
    (32, 128, 64, 16), (32, 128, 64, 64), (32, 128, 128, 1), (32, 128, 128, 16), (32, 128, 128, 32), (32, 128, 128, 64),
    (32, 256, 64, 1), (32, 256, 64, 32), (32, 256, 128, 1), (32, 256, 128, 16), (32, 256, 128, 64), (64, 64, 64, 1),
    (64, 64, 64, 64), (64, 128, 64, 1), (64, 128, 64, 16), (64, 128, 64, 32), (64, 128, 64, 64), (64, 128, 128, 1),
    (128, 128, 64, 1), (128, 128, 64, 16), (128, 128, 64, 32), (128, 128, 64, 64), (128, 256, 64, 1), (128, 256, 64, 16),
    (128, 256, 64, 32), (128, 256, 64, 64)]

/-- The instances covered. The bounds keep the kernel's `int32` arithmetic from wrapping, the
`int64` products in range, the tolerance meaningful (`K ≤ 2^15`), and the writes of `C` apart.
The contents of `sorted_token_ids`, `expert_ids` and `num_tokens_post_padded` are conditions of
`Pre` (they are memory, not instance). -/
structure Inst.ok (d : Inst) : Prop where
  /-- The tile is one the hosts pick on an H200 (`cfgs`). -/
  cfg : (d.BM, d.BN, d.BK, d.GM) ∈ cfgs
  /-- `1 ≤ top_k`: the kernel divides by it (line 283). -/
  topk_pos : 0 < d.top_k
  /-- `1 ≤ E`: the expert of a routed block is in `[0, E)` (`Pre`). -/
  E_pos : 0 < d.E
  /-- `1 ≤ N`: `offs_bn`'s `% N` (line 252), the tile mapping and the columns of `C` need a
  nonzero `N`. -/
  N_pos : 0 < d.N
  /-- `1 ≤ K`: the loop bound. -/
  K_pos : 0 < d.K
  /-- `top_k < 2^31`: the constexpr `top_k` is built in `int64` for line 283's `//` by it. -/
  topk_le : d.top_k < 2 ^ 31
  /-- `K + BLOCK_SIZE_K ≤ 2^31`: `K` is an `int32`, and `tl.cdiv(K, BLOCK_SIZE_K)` computes
  `K + BLOCK_SIZE_K − 1` in `int32`. -/
  K_le : d.K + d.BK ≤ 2 ^ 31
  /-- `K ≤ 2^15`: the largest `K` of real use (Grok-1's second GEMM at TP1); the tolerance of
  `Post` grows with `K` (README, Tolerances). -/
  K_le15 : d.K ≤ 2 ^ 15
  /-- `N` is a model dimension (the output features: twice the intermediate size per rank for the
  first GEMM, the hidden size for the second): at most `2^17`. -/
  N_le17 : d.N ≤ 2 ^ 17
  /-- `N + BLOCK_SIZE_N ≤ 2^31`: `N` is an `int32`, and `tl.cdiv(N, BLOCK_SIZE_N)` computes
  `N + BLOCK_SIZE_N − 1` in `int32`. -/
  N_le : d.N + d.BN ≤ 2 ^ 31
  /-- `cdiv(N, BLOCK_SIZE_N) · GROUP_SIZE_M < 2^31`: `num_pid_in_group` (line 201) does not
  wrap. -/
  N_group : (d.N + d.BN - 1) / d.BN * d.GM < 2 ^ 31
  /-- The grid has fewer than `2^31` programs (`Launch.Valid` requires it). -/
  grid_lt : (d.EM + d.BM - 1) / d.BM * ((d.N + d.BN - 1) / d.BN) < 2 ^ 31
  /-- `EM + BLOCK_SIZE_M ≤ 2^31`: `EM` is an `int32`, and `tl.cdiv(EM, BLOCK_SIZE_M)` computes
  `EM + BLOCK_SIZE_M − 1` in `int32`. -/
  EM_le : d.EM + d.BM ≤ 2 ^ 31
  /-- `M · top_k < 2^31`: `num_valid_tokens` is an `int32`. -/
  nv_le : nv d < 2 ^ 31
  /-- `B`'s strides are `int32`. -/
  be : d.stride_be < 2 ^ 31
  /-- As `be`. -/
  bn : d.stride_bn < 2 ^ 31
  /-- `stride_cm < 2^31`: `stride_cm = N` (a contiguous `C`'s `stride(1)`) is an `int32`
  argument of the kernel (line 666), and the `int64` product `stride_cm · offs_token`
  (lines 246, 418) of two `int32`-ranged factors does not wrap. -/
  cm : stride_cm d < 2 ^ 31
  /-- The elements of `B` have `int64` addresses: no `b_ptrs` sum (lines 285–289) of an expert's
  row, a column and an element wraps — also at the loop's last trips, whose lanes reach
  `BLOCK_SIZE_K · steps − 1 ≤ K + BLOCK_SIZE_K − 2` (masked, but their `int64` sums are still
  computed). -/
  b_off : ∀ e c k, e < d.E → c < d.N → k < d.K + d.BK →
    e * d.stride_be + c * d.stride_bn + k < 2 ^ 63
  /-- `A`'s storage holds every element of the matrix `A` (`M × K`, row-major). -/
  a_len : ∀ t k, t < d.M → k < d.K → t * d.K + k < d.a.len
  /-- `B`'s storage holds every element of the matrix `B` (`E × N × K`). -/
  b_len : ∀ e c k, e < d.E → c < d.N → k < d.K →
    e * d.stride_be + c * d.stride_bn + k < d.b.len
  /-- `C`'s storage holds every element of the matrix `C` (`M × top_k × N`). -/
  c_len : ∀ f c, f < nv d → c < d.N → f * stride_cm d + c < d.c.len
  /-- `topk_weights`' storage holds every valid flat index. -/
  w_len : ∀ f, f < nv d → f < d.w.len
  /-- `topk_weights` is contiguous: `stride(0) = size(1)` (with `stride(1) = 1`, which the host
  asserts), so element `f` of its storage is `topk_weights[f / w_cols, f % w_cols]`, the weight of
  the flat index `f` the kernel's flat read (line 404) means. A kernel limit outside real use:
  vLLM's routers return a contiguous `[M, top_k]` tensor (README, Exclusions). -/
  w_contig : d.w_stride0 = d.w_cols
  /-- `sorted_token_ids`' storage holds the `EM` entries the launch passes (without `naive`). -/
  s_len : d.naive = false → d.EM ≤ d.s.len
  /-- `num_tokens_post_padded`'s storage holds its one entry. -/
  ntp_len : 1 ≤ d.ntpb.len
  /-- The matrix `C` shares no byte with `A`. -/
  c_a : Disjoint (d.c.elems (cIdx d)) d.a.bytes
  /-- The matrix `C` shares no byte with `B`. -/
  c_b : Disjoint (d.c.elems (cIdx d)) d.b.bytes
  /-- The matrix `C` shares no byte with `topk_weights`. -/
  c_w : Disjoint (d.c.elems (cIdx d)) d.w.bytes
  /-- The matrix `C` shares no byte with `sorted_token_ids`. -/
  c_s : Disjoint (d.c.elems (cIdx d)) d.s.bytes
  /-- The matrix `C` shares no byte with `expert_ids`. -/
  c_e : Disjoint (d.c.elems (cIdx d)) d.eid.bytes
  /-- The matrix `C` shares no byte with `num_tokens_post_padded`. -/
  c_n : Disjoint (d.c.elems (cIdx d)) d.ntpb.bytes

/-- The inputs covered: the seven buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and the expert assignment is as the host's `_prepare_expert_assignment` builds it
(lines 573–606; `moe_align_block_size`, lines 423–482, or the naive assignment): the padded total
`ntp` in `num_tokens_post_padded`, a multiple of `BLOCK_SIZE_M`, at most `EM`; every entry below
`ntp` (`entryOf`) nonnegative, the valid ones (below `num_valid_tokens`) pairwise distinct (each
routed flat index appears once); the expert of every block below `ntp` in `[0, E)` or `-1`; with
`MUL_ROUTED_WEIGHT`, the routing weights of the valid flat indices finite. Any `bfloat16` values in
`A` and `B`, NaN and infinities included. With `naive`, `s1` and `s2` hold by `entryOf`'s
definition. -/
structure PreC (d : Inst) (init : Mem) : Prop where
  /-- `A` is a valid buffer. -/
  va : d.a.Valid init
  /-- `B` is a valid buffer. -/
  vb : d.b.Valid init
  /-- `C` is a valid buffer. -/
  vc : d.c.Valid init
  /-- `topk_weights` is a valid buffer. -/
  vw : d.w.Valid init
  /-- `sorted_token_ids` is a valid buffer. -/
  vs : d.s.Valid init
  /-- `expert_ids` is a valid buffer. -/
  ve : d.eid.Valid init
  /-- `num_tokens_post_padded` is a valid buffer. -/
  vn : d.ntpb.Valid init
  /-- The value the kernel loads from `num_tokens_post_padded` (line 215) is the nonnegative
  `ntpOf d init`. -/
  ntpEx : ∃ v, init.read .i32 (d.ntpb.ptr 0).addr = some v ∧ 0 ≤ v.toInt ∧
    v.toInt.toNat = ntpOf d init
  /-- The padded total is a multiple of the block size (each expert's list is padded to
  `BLOCK_SIZE_M`; with `naive` it is `M · top_k · BLOCK_SIZE_M`). -/
  ntpBM : d.BM ∣ ntpOf d init
  /-- The padded total is at most `EM`, the launch's `EM`. -/
  ntpEM : ntpOf d init ≤ d.EM
  /-- Below the padded total the entries of `sorted_token_ids` are nonnegative (the host's are
  flat indices `t · top_k + j` or the padding `M · top_k`; an entry at or above `M · top_k` is
  masked, as the padding is). A negative entry would pass the mask `offs_token <
  num_valid_tokens` and address memory before `C`'s row `0`. -/
  s1 : ∀ pos, pos < ntpOf d init → 0 ≤ entryOf d init pos
  /-- The valid entries below the padded total are pairwise distinct (each routed flat index
  appears once). -/
  s2 : ∀ pos pos', pos < ntpOf d init → pos' < ntpOf d init →
    entryOf d init pos < (nv d : Int) → entryOf d init pos' = entryOf d init pos → pos' = pos
  /-- `expert_ids` holds the entries of every block below the padded total. -/
  elen : ∀ pid_m, pid_m * d.BM < ntpOf d init → pid_m < d.eid.len
  /-- The expert of every block below the padded total is a local expert (`[0, E)`) or `-1`. -/
  erange : ∀ pid_m, pid_m * d.BM < ntpOf d init → eOf d init pid_m = -1 ∨
    ∃ z : Nat, eOf d init pid_m = (z : Int) ∧ z < d.E
  /-- With `MUL_ROUTED_WEIGHT`, the weights of the valid flat indices are finite (line 404 loads
  them; without the flag nothing reads them). The host's are softmax probabilities. -/
  wfin : d.rw = true → ∀ f, f < nv d →
    ∃ v, init.read .f32 (d.w.ptr (f : Int)).addr = some v ∧ v.isFinite = true

/-- The inputs covered, as `PreC`. -/
def Pre (d : Inst) (init : Mem) : Prop := PreC d init

/-! ## The specification -/

/-- The magnitude by which the tensor core aligns a `bfloat16` operand value `q`: `0` for `0`,
and otherwise `|q|`, or the smallest normal magnitude `2^-126` when `|q|` is below it (a
subnormal). It is `|q|` for zero and normal values. The library's `Rules.Num.liftMag .bf16`,
which `Rules.Num.DotBF16` counts its magnitudes by. -/
def magB (q : ℚ) : ℚ := Rules.Num.liftMag .bf16 q

/-- The sum of the products' lifted magnitudes, `Σ magB (va k) · magB (vb k)` over `k < K`. With
no subnormal operand it is `Σ |va k · vb k|`. -/
def dotMag (va vb : Nat → BF16) (K : Nat) : ℚ :=
  ∑ k ∈ Finset.range K, magB (va k).toRat * magB (vb k).toRat

/-- The number of `tl.dot` steps of the loop (line 330): `cdiv(K, BLOCK_SIZE_K)`. -/
def steps (BK K : Nat) : Nat := (K + BK - 1) / BK

/-- The product of position `k` of the reduction: `va k · vb k` for `k < K`, and `0` past `K`
(the last trip's lanes `k ≥ K` load the `other=0.0`, `+0`). -/
def prodAt (va vb : Nat → BF16) (K k : Nat) : ℚ := if k < K then (va k).toRat * (vb k).toRat else 0

/-- The lifted product of position `k`: `magB (va k) · magB (vb k)` for `k < K`, `0` past `K`. -/
def liftAt (va vb : Nat → BF16) (K k : Nat) : ℚ :=
  if k < K then magB (va k).toRat * magB (vb k).toRat else 0

/-- The exact sum of chunk `g` of the reduction, the 64 consecutive positions from `64·g`
(`Rules.Num.chunk16`; trip `t` of the loop holds the chunks `t·(BK/64) … t·(BK/64) + BK/64 − 1`). -/
def chunkSum (va vb : Nat → BF16) (K g : Nat) : ℚ :=
  ∑ i ∈ Finset.range 64, prodAt va vb K (64 * g + i)

/-- A bound on every lifted product of chunk `g`: their sum. -/
def chunkLift (va vb : Nat → BF16) (K g : Nat) : ℚ :=
  ∑ i ∈ Finset.range 64, liftAt va vb K (64 * g + i)

/-- The exact partial sum of the first `g` chunks, `Σ_{k < 64·g} va k · vb k`. -/
def preSum (va vb : Nat → BF16) (K g : Nat) : ℚ := ∑ h ∈ Finset.range g, chunkSum va vb K h

/-- The magnitude the chunked accumulation carries: over the `steps BK K · (BK / 64)` chunks of
64 positions that the loop's `tl.dot`s cover (positions past `K` are zeros; chunk `g` is in the
trip that starts at chunk `g − g % (BK / 64)`), the largest of the exact partial sum before the
chunk, the exact partial sum before the chunk's trip, and the chunk's lifted magnitude, summed. It
is at least `dotMag` (each term is at least the chunk's lifted magnitude), so at least
`|dotSum|`. -/
def pathMag (va vb : Nat → BF16) (BK K : Nat) : ℚ :=
  ∑ g ∈ Finset.range (steps BK K * (BK / 64)),
    max (max |preSum va vb K g| |preSum va vb K (g - g % (BK / 64))|) (chunkLift va vb K g)

/-- The tolerance of the `float32` accumulation of row values `va` and column values `vb`:
`2^-12·pathMag + 2^-133`. It is above the chunk-by-chunk bound of `Rules.Num.DotBF16Chunk` along
the exact partial sums, under either form of each trip's `tl.dot_add` (README, Tolerances). -/
def tol (BK K : Nat) (va vb : Nat → BF16) : ℚ := 2 ^ (-12 : ℤ) * pathMag va vb BK K + 2 ^ (-133 : ℤ)

/-- The exact dot product of row values `va` and column values `vb` over `k < K`. -/
def dotSum (va vb : Nat → BF16) (K : Nat) : ℚ := ∑ k ∈ Finset.range K, (va k).toRat * (vb k).toRat

/-- The final conversion (lines 403–413) and store of the GEMM's accumulator `u`, whose distance
to the exact dot product `S` is at most `T`: with `MUL_ROUTED_WEIGHT`, the routing weight's
`float32` product `r` (line 409), then the `bfloat16` rounding of `r` (line 413), stored (line
420); without it, the `bfloat16` rounding of `u` itself. When the product cannot overflow, `r`
and the output are finite; without the weight, when `|S| + T` is within the `bfloat16` range, the
output is finite. -/
def outFin (d : Inst) (init : Mem) (f : Nat) (cv : BF16) (u : F32) (S T : ℚ) : Prop :=
  if d.rw then
    ∃ wv : F32, init.read .f32 (d.w.ptr (f : Int)).addr = some wv ∧ wv.isFinite = true ∧
      ∃ r : F32, FloatV.NaNEq (FloatV.mul u wv) r ∧ FloatV.NaNEq (FloatV.cvt r) cv ∧
        ((|S| + T) * |wv.toRat| ≤ 2 ^ 126 → r.isFinite = true ∧ cv.isFinite = true)
  else
    FloatV.NaNEq (FloatV.cvt u) cv ∧
    (|S| + T ≤ Math.Float.Format.bfloat16.maxVal → cv.isFinite = true)

/-- What the launch leaves in the element of `C` that position `pos`'s entry `f` and column `c`
have (`f * stride_cm + c`), when program `pos / BLOCK_SIZE_M` owns it. With expert `-1` (lines 234–250):
the `bfloat16` `+0.0` of `write_zeros_to_output`. Otherwise (expert `e < E`, lines 252–420): as
`outFin`, for the exact dot product of the token's row of `A` and the expert's column of `B`,
when every input value along them is finite and the products' lifted magnitudes leave room in
`float32` (`dotMag ≤ 2^120`). -/
def outElem (d : Inst) (init fin : Mem) (pos f c : Nat) : Prop :=
  ∃ cv : BF16, fin.read .bf16 (d.c.ptr ↑(f * stride_cm d + c)).addr = some cv ∧
    (if eOf d init (pos / d.BM) = -1 then
      FloatV.NaNEq (FloatV.ofPat 0) cv
    else
      ∃ e : Nat, eOf d init (pos / d.BM) = (e : Int) ∧ e < d.E ∧
        ∀ va vb : Nat → BF16,
          (∀ k, k < d.K → init.read .bf16 (d.a.ptr ↑(f / d.top_k * d.K + k)).addr = some (va k) ∧
            (va k).isFinite = true) →
          (∀ k, k < d.K →
            init.read .bf16 (d.b.ptr ↑(e * d.stride_be + c * d.stride_bn + k)).addr = some (vb k) ∧
            (vb k).isFinite = true) →
          dotMag va vb d.K ≤ 2 ^ 120 →
          ∃ u : F32, u.isFinite = true ∧
            |u.toRat - dotSum va vb d.K| ≤ tol d.BK d.K va vb ∧
            outFin d init f cv u (dotSum va vb d.K) (tol d.BK d.K va vb))

/-- What the launch leaves: every element of the matrix `C` is either unchanged (no position
below the padded total holds its flat index — the row is not routed to a local expert) or as
`outElem` says for the position that holds its flat index; and every byte outside the matrix `C`
is unchanged (`A`, `B`, `topk_weights`, the alignment buffers, and the elements of `C`'s storage
outside the matrix). -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ f c, f < nv d → c < d.N →
    ((∀ pos, pos < ntpOf d init → entryOf d init pos ≠ (f : Int)) →
      ∀ z ∈ d.c.elems {f * stride_cm d + c}, fin z = init z) ∧
    ∀ pos, pos < ntpOf d init → entryOf d init pos = (f : Int) → outElem d init fin pos f c) ∧
    ∀ z, z ∉ d.c.elems (cIdx d) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The model on an instance, the launch, and the assumptions -/

/-- The model on an instance: the buffers' first elements, `N`, `K`, `EM`, `num_valid_tokens` and
the strides as `int32` (the strides the hosts fix: `stride_am = K`, `stride_ak = 1`,
`stride_bk = 1`, `stride_cm = N` (a contiguous `C`'s `stride(1)`), `stride_cn = 1`; see `Inst`),
and the flags and tile of the launch (lines 677–689): `naive_block_assignment = d.naive`,
`BLOCK_SIZE_M = d.BM`, `BLOCK_SIZE_N = d.BN`, `BLOCK_SIZE_K = d.BK`, `GROUP_SIZE_M = d.GM`,
`top_k = d.top_k`, `MUL_ROUTED_WEIGHT = d.rw`. An integer argument equal to `1` is the
compile-time constant `1`; every use of `N`, `K`, `EM`, `num_valid_tokens` and of a stride then
makes it a tensor of the operation's dtype, which computes the same value (the docstring of
`fused_moe_kernel` goes through them). -/
def model (d : Inst) : Kernel :=
  fused_moe_kernel (d.a.ptr 0) (d.b.ptr 0) (d.c.ptr 0) (d.w.ptr 0) (d.s.ptr 0) (d.eid.ptr 0)
    (d.ntpb.ptr 0) (IntV.ofNat d.N) (IntV.ofNat d.K) (IntV.ofNat d.EM) (IntV.ofNat (nv d))
    (IntV.ofNat d.K) (IntV.ofNat 1) (IntV.ofNat d.stride_be) (IntV.ofNat 1) (IntV.ofNat d.stride_bn)
    (IntV.ofNat (stride_cm d)) (IntV.ofNat 1)
    d.naive d.BM d.BN d.BK d.GM d.top_k d.rw

/-- The host's launch (lines 640–690): a 1-D grid of `triton.cdiv(EM, BLOCK_SIZE_M) ·
triton.cdiv(N, BLOCK_SIZE_N)` programs; the source's block shapes; `enable_fp_fusion` is Triton's
default (`true`); `grid1` keeps `num_ctas = 1`, `launch_pdl = False` and `debug = false`.
`num_warps` and `num_stages` (`get_default_config`'s, spread with the configuration) are not
recorded (performance-only, and harmless to a well-behaved model). -/
def launch (d : Inst) : Launch :=
  Launch.grid1 ((d.EM + d.BM - 1) / d.BM * ((d.N + d.BN - 1) / d.BN))
    [[d.BM], [d.BN], [d.BK], [d.BM, d.BK], [d.BK, d.BN], [d.BM, d.BN]]
    (fusion := true)

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The
  rule covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotBF16Chunk`: `tl.dot` of `bfloat16` operands into `float32`
  (`maxImpreciseAcc = 0`, any `input_precision`, the model's line 373), chunk by chunk: partial
  values `s 0 = acc, …, s ⌈K/64⌉ = result`, chunk `j` (64 consecutive `k`) within
  `min (2^-13·Mj) (2·A_j + 2^-22·Mj) + 2^-20·|s j + c_j| + 2^-143` of `s j + c_j`, for every `Mj`
  bounding `|s j|` and the chunk's lifted products (a subnormal operand counted at `2^-126`), with
  room below the `float32` range. -/
  dot : Rules.Num.DotBF16Chunk P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` (line 373) always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }


/-! ## The claim -/

/-- The claim to prove: under the assumptions, the launch meets the specification on every
covered instance and input. Not proved here. -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous: the assumptions are consistent -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotBF16Chunk,
    Platform.ref.dotTotal _⟩⟩

/-! ## Coverage: the launches `SETUP.md` names

`hostInst` is the instance a host builds: `A` contiguous `[Ar, K]` (`Ar` the tokens for the first
GEMM, `M · top_k` for the second), `B` contiguous `[E, N, K]` (`stride_be = N·K`,
`stride_bn = K`), a fresh contiguous `C` of `Ar · top_k` rows of `N`, `topk_weights` of
`Ar · top_k` values (`wcols` per row), and the expert assignment: `sorted_token_ids` of `slen`
entries, `expert_ids` of `elen`, `num_tokens_post_padded` of one, with the launch's `EM` and tile.
`gemm1Inst` and `gemm2Inst` compute these as `fused_moe_gemm1` and `fused_moe_gemm2` do, for the
tile each setup names (the H200's: the tuned file's entry for the batch, else `get_default_config`'s;
the setups' docstrings say which), the naive assignment when `M · top_k · 4 ≤ E_global` and no expert map, the
buffer sizes of `moe_align_block_size` and the `EM` of lines 628–639). The seven buffers sit at
fixed, aligned, far-apart addresses; the claim covers any addresses `Inst.ok` allows, and these
only witness the sizes. -/

/-- The instance for `A`'s rows `Ar`, the kernel's `top_k`, the local experts `E`, `N`, `K`, the
launch's `EM`, the buffers' lengths `slen`, `elen`, `MUL_ROUTED_WEIGHT`, `naive`, the tile `cfg`
and `topk_weights.size(1)`. -/
def hostInst (Ar top_k E N K EM slen elen : Nat) (rw naive : Bool) (cfg : Nat × Nat × Nat × Nat)
    (wcols : Nat) : Inst where
  M := Ar
  top_k := top_k
  E := E
  N := N
  K := K
  EM := EM
  stride_be := N * K
  stride_bn := K
  rw := rw
  naive := naive
  BM := cfg.1
  BN := cfg.2.1
  BK := cfg.2.2.1
  GM := cfg.2.2.2
  a := ⟨2 ^ 32, Ar * K⟩
  b := ⟨2 ^ 36, E * N * K⟩
  c := ⟨2 ^ 40, Ar * top_k * N⟩
  w := ⟨2 ^ 44, Ar * top_k⟩
  w_cols := wcols
  w_stride0 := wcols
  s := ⟨2 ^ 45, slen⟩
  eid := ⟨2 ^ 46, elen⟩
  ntpb := ⟨2 ^ 47, 1⟩

/-- `get_default_config(M, E)` for `bfloat16` (lines 485–520): `(BLOCK_SIZE_M, BLOCK_SIZE_N,
BLOCK_SIZE_K, GROUP_SIZE_M)`. -/
def defaultCfg (M E : Nat) : Nat × Nat × Nat × Nat :=
  let gm := if M / max E 1 > 128 then 16 else 1
  if M ≤ 32 then (16, 64, 128, gm)
  else if M ≤ 64 then (32, 64, 128, gm)
  else if M ≤ 96 then (32, 128, 64, gm)
  else if M ≤ 512 then (64, 128, 64, gm)
  else (128, 128, 64, gm)

/-- `naive_block_assignment` (lines 583–586): no expert map and `M · top_k · 4 ≤ E_global`. -/
def naiveOf (M top_k Eg : Nat) (hasMap : Bool) : Bool := !hasMap && decide (M * top_k * 4 ≤ Eg)

/-- `max_num_tokens_padded` of `moe_align_block_size` (lines 439–443), for `numel` flat indices,
`Eg` experts and the block size `BM`; no buffer with the naive assignment. -/
def slenOf (naive : Bool) (numel Eg BM : Nat) : Nat :=
  if naive then 0
  else if numel < Eg then min (numel * BM) (numel + Eg * (BM - 1)) else numel + Eg * (BM - 1)

/-- The length of `expert_ids`: `cdiv(slen, BM)` (line 444), or `topk_ids.numel()` for the naive
assignment's `topk_ids.view(-1)`. -/
def elenOf (naive : Bool) (numel BM slen : Nat) : Nat :=
  if naive then numel else (slen + BM - 1) / BM

/-- The launch's `EM` (lines 628–639) for `A.size(0) = Ar` and the kernel's `top_k`. -/
def emOf (naive : Bool) (Ar top_k BM slen : Nat) : Nat :=
  if naive then Ar * top_k * BM
  else if Ar < BM then min slen (Ar * top_k * BM) else slen

/-- `fused_moe_gemm1`'s instance for `M` tokens, `top_k`, `E` local and `Eg` global experts, `N`,
`K`, `apply_router_weight_on_input` (`onInput`), whether an expert map is passed, and the tile
`cfg` the host picks (the tuned file's entry for `M`, or `defaultCfg M E`). -/
def gemm1Inst (M top_k E Eg N K : Nat) (onInput hasMap : Bool)
    (cfg : Nat × Nat × Nat × Nat) : Inst :=
  hostInst M top_k E N K
    (emOf (naiveOf M top_k Eg hasMap) M top_k cfg.1
      (slenOf (naiveOf M top_k Eg hasMap) (M * top_k) Eg cfg.1))
    (slenOf (naiveOf M top_k Eg hasMap) (M * top_k) Eg cfg.1)
    (elenOf (naiveOf M top_k Eg hasMap) (M * top_k) cfg.1
      (slenOf (naiveOf M top_k Eg hasMap) (M * top_k) Eg cfg.1))
    onInput (naiveOf M top_k Eg hasMap) cfg top_k

/-- `fused_moe_gemm2`'s instance: `A` is `intermediate_cache2` (`M · top_k` rows of `K`), the
kernel's `top_k` is `1`, `MUL_ROUTED_WEIGHT = not apply_router_weight_on_input`; the tile and the
assignment are those of the `M` tokens. -/
def gemm2Inst (M top_k E Eg N K : Nat) (onInput hasMap : Bool)
    (cfg : Nat × Nat × Nat × Nat) : Inst :=
  hostInst (M * top_k) 1 E N K
    (emOf (naiveOf M top_k Eg hasMap) (M * top_k) 1 cfg.1
      (slenOf (naiveOf M top_k Eg hasMap) (M * top_k) Eg cfg.1))
    (slenOf (naiveOf M top_k Eg hasMap) (M * top_k) Eg cfg.1)
    (elenOf (naiveOf M top_k Eg hasMap) (M * top_k) cfg.1
      (slenOf (naiveOf M top_k Eg hasMap) (M * top_k) Eg cfg.1))
    (!onInput) (naiveOf M top_k Eg hasMap) cfg top_k

/-- `t·K + k < M·K` for `t < M`, `k < K`. -/
theorem row_lt {t M K k : Nat} (ht : t < M) (hk : k < K) : t * K + k < M * K := by
  have h : (t + 1) * K ≤ M * K := Nat.mul_le_mul_right K ht
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- `e·X + X ≤ E·X` for `e < E`. -/
theorem succ_mul_le {e E X : Nat} (h : e < E) : e * X + X ≤ E * X := by
  have := Nat.mul_le_mul_right X h
  rwa [Nat.succ_mul] at this

/-- Two buffers whose byte ranges are in order share no byte. -/
theorem bytes_disjoint {d₁ d₂ : DType} {x : Buf d₁} {y : Buf d₂}
    (h : x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base) :
    Disjoint x.bytes y.bytes := by
  rw [Set.disjoint_left]
  intro z hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The host's instance is covered when its tile is one of `get_default_config`'s and its sizes
meet `Inst.ok`'s numeric bounds and fit the address layout of `hostInst`. -/
theorem hostInst_ok {Ar top_k E N K EM slen elen wcols : Nat} {rw naive : Bool}
    {cfg : Nat × Nat × Nat × Nat}
    (h : cfg ∈ cfgs ∧ 0 < top_k ∧ 0 < E ∧ 0 < N ∧ 0 < K ∧ top_k < 2 ^ 31 ∧ K ≤ 2 ^ 15 ∧
      K + cfg.2.2.1 ≤ 2 ^ 31 ∧ N + cfg.2.1 ≤ 2 ^ 31 ∧
      (N + cfg.2.1 - 1) / cfg.2.1 * cfg.2.2.2 < 2 ^ 31 ∧
      (EM + cfg.1 - 1) / cfg.1 * ((N + cfg.2.1 - 1) / cfg.2.1) < 2 ^ 31 ∧ EM + cfg.1 ≤ 2 ^ 31 ∧
      Ar * top_k < 2 ^ 31 ∧ N * K < 2 ^ 31 ∧ (naive = false → EM ≤ slen) ∧
      Ar * K * 2 ≤ 2 ^ 35 ∧ E * N * K * 2 ≤ 2 ^ 39 ∧ Ar * top_k * N * 2 ≤ 2 ^ 43 ∧
      Ar * top_k * 4 ≤ 2 ^ 44 ∧ slen * 4 ≤ 2 ^ 45 ∧ elen * 4 ≤ 2 ^ 46 ∧ N ≤ 2 ^ 17) :
    (hostInst Ar top_k E N K EM slen elen rw naive cfg wcols).ok := by
  obtain ⟨hc, h1, h2, h3, h4, h5, h6, hK, hN, hG, h9, h10, h11, h12, h13, hA, hB, hC, hW, hS, hE, hN17⟩ := h
  have hcl : ∀ j ∈ cIdx (hostInst Ar top_k E N K EM slen elen rw naive cfg wcols),
      j < (hostInst Ar top_k E N K EM slen elen rw naive cfg wcols).c.len := by
    rintro j ⟨f, c, hf, hc, rfl⟩
    exact row_lt hf hc
  have hcb : ∀ {d' : DType} (x : Buf d'),
      Disjoint (hostInst Ar top_k E N K EM slen elen rw naive cfg wcols).c.bytes x.bytes →
      Disjoint ((hostInst Ar top_k E N K EM slen elen rw naive cfg wcols).c.elems
        (cIdx (hostInst Ar top_k E N K EM slen elen rw naive cfg wcols))) x.bytes :=
    fun x hx => hx.mono_left (Buf.elems_sub_bytes _ hcl)
  have hBK : cfg.2.2.1 ≤ 256 ∧ 1 ≤ cfg.2.1 := by
    simp only [cfgs, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  have hbe : ∀ e c k, e < E → c < N → k < K + cfg.2.2.1 →
      e * (N * K) + c * K + k < E * N * K + 256 + K := by
    intro e c k he hc hk
    have h1 : e * (N * K) + N * K ≤ E * (N * K) := succ_mul_le he
    have h2 : c * K + K ≤ N * K := succ_mul_le hc
    have h3 : E * (N * K) = E * N * K := (Nat.mul_assoc _ _ _).symm
    omega
  have hboff : ∀ e c k, e < E → c < N → k < K + cfg.2.2.1 →
      e * (N * K) + c * K + k < 2 ^ 63 := by
    intro e c k he hc hk
    have := hbe e c k he hc hk
    omega
  have hblen : ∀ e c k, e < E → c < N → k < K → e * (N * K) + c * K + k < E * N * K := by
    intro e c k he hc hk
    have h2 : c * K + k < N * K := row_lt hc hk
    have h1 : e * (N * K) + N * K ≤ E * (N * K) := succ_mul_le he
    have h3 : E * (N * K) = E * N * K := (Nat.mul_assoc _ _ _).symm
    omega
  refine ⟨hc, h1, h2, h3, h4, h5, hK, h6, hN17, hN, hG, h9, h10, h11,
    by simp only [hostInst]; omega, by simp only [hostInst]; omega,
    by simp only [stride_cm, hostInst]; omega,
    hboff, fun t k ht hk => row_lt ht hk, hblen, fun f c hf hc => row_lt hf hc, fun f hf => hf, rfl,
    h13, le_refl 1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals
    refine hcb _ (bytes_disjoint ?_)
    simp only [hostInst, DType.bytes]
    omega

/-! ### The host's test cases (source lines 865–873), both GEMMs -/

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (1, 256, 128, 8, 2)` (line 865): the tile `(16, 64, 128, 1)` of `get_default_config`, the naive assignment. -/
def setup_gemm1_m1_n256_k128_e8_top2 : Inst := gemm1Inst 1 2 8 8 256 128 false false (16, 64, 128, 1)
theorem setup_gemm1_m1_n256_k128_e8_top2_ok : setup_gemm1_m1_n256_k128_e8_top2.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[8, 128, 128]`, `A` the `2 × 128` `intermediate_cache2`). -/
def setup_gemm2_m1_n256_k128_e8_top2 : Inst := gemm2Inst 1 2 8 8 128 128 false false (16, 64, 128, 1)
theorem setup_gemm2_m1_n256_k128_e8_top2_ok : setup_gemm2_m1_n256_k128_e8_top2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (33, 4096, 128, 64, 6)` (line 866): the tile `(32, 64, 128, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_gemm1_m33_n4096_k128_e64_top6 : Inst := gemm1Inst 33 6 64 64 4096 128 false false (32, 64, 128, 1)
theorem setup_gemm1_m33_n4096_k128_e64_top6_ok : setup_gemm1_m33_n4096_k128_e64_top6.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[64, 128, 2048]`, `A` the `198 × 2048` `intermediate_cache2`). -/
def setup_gemm2_m33_n4096_k128_e64_top6 : Inst := gemm2Inst 33 6 64 64 128 2048 false false (32, 64, 128, 1)
theorem setup_gemm2_m33_n4096_k128_e64_top6_ok : setup_gemm2_m33_n4096_k128_e64_top6.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (100, 1022, 511, 8, 2)` (line 867): the tile `(64, 128, 64, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_gemm1_m100_n1022_k511_e8_top2 : Inst := gemm1Inst 100 2 8 8 1022 511 false false (64, 128, 64, 1)
theorem setup_gemm1_m100_n1022_k511_e8_top2_ok : setup_gemm1_m100_n1022_k511_e8_top2.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[8, 511, 511]`, `A` the `200 × 511` `intermediate_cache2`). -/
def setup_gemm2_m100_n1022_k511_e8_top2 : Inst := gemm2Inst 100 2 8 8 511 511 false false (64, 128, 64, 1)
theorem setup_gemm2_m100_n1022_k511_e8_top2_ok : setup_gemm2_m100_n1022_k511_e8_top2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (222, 512, 7168, 256, 8)` (line 868): the tile `(64, 128, 64, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_gemm1_m222_n512_k7168_e256_top8 : Inst := gemm1Inst 222 8 256 256 512 7168 false false (64, 128, 64, 1)
theorem setup_gemm1_m222_n512_k7168_e256_top8_ok : setup_gemm1_m222_n512_k7168_e256_top8.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[256, 7168, 256]`, `A` the `1776 × 256` `intermediate_cache2`). -/
def setup_gemm2_m222_n512_k7168_e256_top8 : Inst := gemm2Inst 222 8 256 256 7168 256 false false (64, 128, 64, 1)
theorem setup_gemm2_m222_n512_k7168_e256_top8_ok : setup_gemm2_m222_n512_k7168_e256_top8.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (1024, 1536, 2048, 128, 8)` (line 869): the tile `(128, 256, 64, 16)` of the tuned file `E=128,N=768`, `moe_align_block_size`. -/
def setup_gemm1_m1024_n1536_k2048_e128_top8 : Inst := gemm1Inst 1024 8 128 128 1536 2048 false false (128, 256, 64, 16)
theorem setup_gemm1_m1024_n1536_k2048_e128_top8_ok : setup_gemm1_m1024_n1536_k2048_e128_top8.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[128, 2048, 768]`, `A` the `8192 × 768` `intermediate_cache2`). -/
def setup_gemm2_m1024_n1536_k2048_e128_top8 : Inst := gemm2Inst 1024 8 128 128 2048 768 false false (128, 256, 64, 16)
theorem setup_gemm2_m1024_n1536_k2048_e128_top8_ok : setup_gemm2_m1024_n1536_k2048_e128_top8.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (2048, 3584, 4096, 8, 2)` (line 870): the tile `(128, 256, 64, 16)` of the tuned file `E=8,N=1792`, `moe_align_block_size`. -/
def setup_gemm1_m2048_n3584_k4096_e8_top2 : Inst := gemm1Inst 2048 2 8 8 3584 4096 false false (128, 256, 64, 16)
theorem setup_gemm1_m2048_n3584_k4096_e8_top2_ok : setup_gemm1_m2048_n3584_k4096_e8_top2.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[8, 4096, 1792]`, `A` the `4096 × 1792` `intermediate_cache2`). -/
def setup_gemm2_m2048_n3584_k4096_e8_top2 : Inst := gemm2Inst 2048 2 8 8 4096 1792 false false (128, 256, 64, 16)
theorem setup_gemm2_m2048_n3584_k4096_e8_top2_ok : setup_gemm2_m2048_n3584_k4096_e8_top2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (64, 2048, 5120, 16, 1)`, the routing weights on the input (line 871): the tile `(16, 64, 128, 1)` of the tuned file `E=16,N=1024`, `moe_align_block_size`. -/
def setup_gemm1_m64_n2048_k5120_e16_top1_rw : Inst := gemm1Inst 64 1 16 16 2048 5120 true false (16, 64, 128, 1)
theorem setup_gemm1_m64_n2048_k5120_e16_top1_rw_ok : setup_gemm1_m64_n2048_k5120_e16_top1_rw.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[16, 5120, 1024]`, `A` the `64 × 1024` `intermediate_cache2`). -/
def setup_gemm2_m64_n2048_k5120_e16_top1_rw : Inst := gemm2Inst 64 1 16 16 5120 1024 true false (16, 64, 128, 1)
theorem setup_gemm2_m64_n2048_k5120_e16_top1_rw_ok : setup_gemm2_m64_n2048_k5120_e16_top1_rw.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (300, 1024, 1024, 64, 6)`, an expert map of 16 local experts (line 872): the tile `(64, 128, 64, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_gemm1_m300_n1024_k1024_e64_top6_ep16 : Inst := gemm1Inst 300 6 16 64 1024 1024 false true (64, 128, 64, 1)
theorem setup_gemm1_m300_n1024_k1024_e64_top6_ep16_ok : setup_gemm1_m300_n1024_k1024_e64_top6_ep16.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[16, 1024, 512]`, `A` the `1800 × 512` `intermediate_cache2`). -/
def setup_gemm2_m300_n1024_k1024_e64_top6_ep16 : Inst := gemm2Inst 300 6 16 64 1024 512 false true (64, 128, 64, 1)
theorem setup_gemm2_m300_n1024_k1024_e64_top6_ep16_ok : setup_gemm2_m300_n1024_k1024_e64_top6_ep16.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The first GEMM of the test case `(M, N, K, E, top_k) = (2, 512, 7168, 256, 8)` (line 873): the tile `(16, 64, 128, 1)` of `get_default_config`, the naive assignment. -/
def setup_gemm1_m2_n512_k7168_e256_top8 : Inst := gemm1Inst 2 8 256 256 512 7168 false false (16, 64, 128, 1)
theorem setup_gemm1_m2_n512_k7168_e256_top8_ok : setup_gemm1_m2_n512_k7168_e256_top8.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Its second GEMM (`w2` `[256, 7168, 256]`, `A` the `16 × 256` `intermediate_cache2`). -/
def setup_gemm2_m2_n512_k7168_e256_top8 : Inst := gemm2Inst 2 8 256 256 7168 256 false false (16, 64, 128, 1)
theorem setup_gemm2_m2_n512_k7168_e256_top8_ok : setup_gemm2_m2_n512_k7168_e256_top8.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-! ### Model layers (SETUP.md): decode, batches, prefill, with the H200's tiles -/

/-- Mixtral-8x7B at TP1 (`E = 8`, `top_k = 2`, hidden `4096`, intermediate `14336`), `M = 1`, the first GEMM (`N = 28672`, `K = 4096`): the tile `(16, 128, 128, 16)` of the tuned file `E=8,N=14336`, the naive assignment. -/
def setup_mixtral_8x7b_m1_gemm1 : Inst := gemm1Inst 1 2 8 8 28672 4096 false false (16, 128, 128, 16)
theorem setup_mixtral_8x7b_m1_gemm1_ok : setup_mixtral_8x7b_m1_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 4096`, `K = 14336`). -/
def setup_mixtral_8x7b_m1_gemm2 : Inst := gemm2Inst 1 2 8 8 4096 14336 false false (16, 128, 128, 16)
theorem setup_mixtral_8x7b_m1_gemm2_ok : setup_mixtral_8x7b_m1_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x7B at TP1 (`E = 8`, `top_k = 2`, hidden `4096`, intermediate `14336`), `M = 16`, the first GEMM (`N = 28672`, `K = 4096`): the tile `(16, 64, 256, 16)` of the tuned file `E=8,N=14336`, `moe_align_block_size`. -/
def setup_mixtral_8x7b_m16_gemm1 : Inst := gemm1Inst 16 2 8 8 28672 4096 false false (16, 64, 256, 16)
theorem setup_mixtral_8x7b_m16_gemm1_ok : setup_mixtral_8x7b_m16_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 4096`, `K = 14336`). -/
def setup_mixtral_8x7b_m16_gemm2 : Inst := gemm2Inst 16 2 8 8 4096 14336 false false (16, 64, 256, 16)
theorem setup_mixtral_8x7b_m16_gemm2_ok : setup_mixtral_8x7b_m16_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x7B at TP1 (`E = 8`, `top_k = 2`, hidden `4096`, intermediate `14336`), `M = 64`, the first GEMM (`N = 28672`, `K = 4096`): the tile `(32, 128, 128, 64)` of the tuned file `E=8,N=14336`, `moe_align_block_size`. -/
def setup_mixtral_8x7b_m64_gemm1 : Inst := gemm1Inst 64 2 8 8 28672 4096 false false (32, 128, 128, 64)
theorem setup_mixtral_8x7b_m64_gemm1_ok : setup_mixtral_8x7b_m64_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 4096`, `K = 14336`). -/
def setup_mixtral_8x7b_m64_gemm2 : Inst := gemm2Inst 64 2 8 8 4096 14336 false false (32, 128, 128, 64)
theorem setup_mixtral_8x7b_m64_gemm2_ok : setup_mixtral_8x7b_m64_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x7B at TP1 (`E = 8`, `top_k = 2`, hidden `4096`, intermediate `14336`), `M = 512`, the first GEMM (`N = 28672`, `K = 4096`): the tile `(128, 256, 64, 16)` of the tuned file `E=8,N=14336`, `moe_align_block_size`. -/
def setup_mixtral_8x7b_m512_gemm1 : Inst := gemm1Inst 512 2 8 8 28672 4096 false false (128, 256, 64, 16)
theorem setup_mixtral_8x7b_m512_gemm1_ok : setup_mixtral_8x7b_m512_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 4096`, `K = 14336`). -/
def setup_mixtral_8x7b_m512_gemm2 : Inst := gemm2Inst 512 2 8 8 4096 14336 false false (128, 256, 64, 16)
theorem setup_mixtral_8x7b_m512_gemm2_ok : setup_mixtral_8x7b_m512_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x7B at TP1 (`E = 8`, `top_k = 2`, hidden `4096`, intermediate `14336`), `M = 40000`, the first GEMM (`N = 28672`, `K = 4096`): the tile `(128, 256, 64, 16)` of the tuned file `E=8,N=14336`, `moe_align_block_size`. -/
def setup_mixtral_8x7b_m40000_gemm1 : Inst := gemm1Inst 40000 2 8 8 28672 4096 false false (128, 256, 64, 16)
theorem setup_mixtral_8x7b_m40000_gemm1_ok : setup_mixtral_8x7b_m40000_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 4096`, `K = 14336`). -/
def setup_mixtral_8x7b_m40000_gemm2 : Inst := gemm2Inst 40000 2 8 8 4096 14336 false false (128, 256, 64, 16)
theorem setup_mixtral_8x7b_m40000_gemm2_ok : setup_mixtral_8x7b_m40000_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x7B at TP2 (intermediate `7168` per rank), `M = 1024`, the first GEMM (`N = 14336`, `K = 4096`): the tile `(128, 256, 64, 64)` of the tuned file `E=8,N=7168`, `moe_align_block_size`. -/
def setup_mixtral_8x7b_tp2_m1024_gemm1 : Inst := gemm1Inst 1024 2 8 8 14336 4096 false false (128, 256, 64, 64)
theorem setup_mixtral_8x7b_tp2_m1024_gemm1_ok : setup_mixtral_8x7b_tp2_m1024_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 4096`, `K = 7168`). -/
def setup_mixtral_8x7b_tp2_m1024_gemm2 : Inst := gemm2Inst 1024 2 8 8 4096 7168 false false (128, 256, 64, 64)
theorem setup_mixtral_8x7b_tp2_m1024_gemm2_ok : setup_mixtral_8x7b_tp2_m1024_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x7B at TP8 (intermediate `1792` per rank), `M = 1`, the first GEMM (`N = 3584`, `K = 4096`): the tile `(16, 64, 256, 1)` of the tuned file `E=8,N=1792`, the naive assignment. -/
def setup_mixtral_8x7b_tp8_m1_gemm1 : Inst := gemm1Inst 1 2 8 8 3584 4096 false false (16, 64, 256, 1)
theorem setup_mixtral_8x7b_tp8_m1_gemm1_ok : setup_mixtral_8x7b_tp8_m1_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 4096`, `K = 1792`). -/
def setup_mixtral_8x7b_tp8_m1_gemm2 : Inst := gemm2Inst 1 2 8 8 4096 1792 false false (16, 64, 256, 1)
theorem setup_mixtral_8x7b_tp8_m1_gemm2_ok : setup_mixtral_8x7b_tp8_m1_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x22B at TP1 (`E = 8`, `top_k = 2`, hidden `6144`, intermediate `16384`), `M = 1`, the first GEMM (`N = 32768`, `K = 6144`): the tile `(16, 64, 128, 1)` of `get_default_config`, the naive assignment. -/
def setup_mixtral_8x22b_m1_gemm1 : Inst := gemm1Inst 1 2 8 8 32768 6144 false false (16, 64, 128, 1)
theorem setup_mixtral_8x22b_m1_gemm1_ok : setup_mixtral_8x22b_m1_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 6144`, `K = 16384`). -/
def setup_mixtral_8x22b_m1_gemm2 : Inst := gemm2Inst 1 2 8 8 6144 16384 false false (16, 64, 128, 1)
theorem setup_mixtral_8x22b_m1_gemm2_ok : setup_mixtral_8x22b_m1_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Mixtral-8x22B at TP1 (`E = 8`, `top_k = 2`, hidden `6144`, intermediate `16384`), `M = 512`, the first GEMM (`N = 32768`, `K = 6144`): the tile `(64, 128, 64, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_mixtral_8x22b_m512_gemm1 : Inst := gemm1Inst 512 2 8 8 32768 6144 false false (64, 128, 64, 1)
theorem setup_mixtral_8x22b_m512_gemm1_ok : setup_mixtral_8x22b_m512_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 6144`, `K = 16384`). -/
def setup_mixtral_8x22b_m512_gemm2 : Inst := gemm2Inst 512 2 8 8 6144 16384 false false (64, 128, 64, 1)
theorem setup_mixtral_8x22b_m512_gemm2_ok : setup_mixtral_8x22b_m512_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Grok-1 at TP1 (`E = 8`, `top_k = 2`, hidden `6144`, intermediate `32768`: the second GEMM's `K = 2^15`), `M = 1`, the first GEMM (`N = 65536`, `K = 6144`): the tile `(16, 64, 128, 1)` of `get_default_config`, the naive assignment. -/
def setup_grok1_m1_gemm1 : Inst := gemm1Inst 1 2 8 8 65536 6144 false false (16, 64, 128, 1)
theorem setup_grok1_m1_gemm1_ok : setup_grok1_m1_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 6144`, `K = 32768`). -/
def setup_grok1_m1_gemm2 : Inst := gemm2Inst 1 2 8 8 6144 32768 false false (16, 64, 128, 1)
theorem setup_grok1_m1_gemm2_ok : setup_grok1_m1_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Grok-1 at TP1 (`E = 8`, `top_k = 2`, hidden `6144`, intermediate `32768`: the second GEMM's `K = 2^15`), `M = 512`, the first GEMM (`N = 65536`, `K = 6144`): the tile `(64, 128, 64, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_grok1_m512_gemm1 : Inst := gemm1Inst 512 2 8 8 65536 6144 false false (64, 128, 64, 1)
theorem setup_grok1_m512_gemm1_ok : setup_grok1_m512_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 6144`, `K = 32768`). -/
def setup_grok1_m512_gemm2 : Inst := gemm2Inst 512 2 8 8 6144 32768 false false (64, 128, 64, 1)
theorem setup_grok1_m512_gemm2_ok : setup_grok1_m512_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- DeepSeek-V3 at TP1 (`E = 256`, `top_k = 8`, hidden `7168`, intermediate `2048`), `M = 1`, the first GEMM (`N = 4096`, `K = 7168`): the tile `(16, 64, 128, 1)` of `get_default_config`, the naive assignment. -/
def setup_deepseek_v3_m1_gemm1 : Inst := gemm1Inst 1 8 256 256 4096 7168 false false (16, 64, 128, 1)
theorem setup_deepseek_v3_m1_gemm1_ok : setup_deepseek_v3_m1_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 7168`, `K = 2048`). -/
def setup_deepseek_v3_m1_gemm2 : Inst := gemm2Inst 1 8 256 256 7168 2048 false false (16, 64, 128, 1)
theorem setup_deepseek_v3_m1_gemm2_ok : setup_deepseek_v3_m1_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- DeepSeek-V3 at TP1 (`E = 256`, `top_k = 8`, hidden `7168`, intermediate `2048`), `M = 8`, the first GEMM (`N = 4096`, `K = 7168`): the tile `(16, 64, 128, 1)` of `get_default_config`, the naive assignment. -/
def setup_deepseek_v3_m8_gemm1 : Inst := gemm1Inst 8 8 256 256 4096 7168 false false (16, 64, 128, 1)
theorem setup_deepseek_v3_m8_gemm1_ok : setup_deepseek_v3_m8_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 7168`, `K = 2048`). -/
def setup_deepseek_v3_m8_gemm2 : Inst := gemm2Inst 8 8 256 256 7168 2048 false false (16, 64, 128, 1)
theorem setup_deepseek_v3_m8_gemm2_ok : setup_deepseek_v3_m8_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- DeepSeek-V3 at TP1 (`E = 256`, `top_k = 8`, hidden `7168`, intermediate `2048`), `M = 9`, the first GEMM (`N = 4096`, `K = 7168`): the tile `(16, 64, 128, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_deepseek_v3_m9_gemm1 : Inst := gemm1Inst 9 8 256 256 4096 7168 false false (16, 64, 128, 1)
theorem setup_deepseek_v3_m9_gemm1_ok : setup_deepseek_v3_m9_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 7168`, `K = 2048`). -/
def setup_deepseek_v3_m9_gemm2 : Inst := gemm2Inst 9 8 256 256 7168 2048 false false (16, 64, 128, 1)
theorem setup_deepseek_v3_m9_gemm2_ok : setup_deepseek_v3_m9_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- DeepSeek-V3 at TP1 (`E = 256`, `top_k = 8`, hidden `7168`, intermediate `2048`), `M = 512`, the first GEMM (`N = 4096`, `K = 7168`): the tile `(64, 128, 64, 1)` of `get_default_config`, `moe_align_block_size`. -/
def setup_deepseek_v3_m512_gemm1 : Inst := gemm1Inst 512 8 256 256 4096 7168 false false (64, 128, 64, 1)
theorem setup_deepseek_v3_m512_gemm1_ok : setup_deepseek_v3_m512_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 7168`, `K = 2048`). -/
def setup_deepseek_v3_m512_gemm2 : Inst := gemm2Inst 512 8 256 256 7168 2048 false false (64, 128, 64, 1)
theorem setup_deepseek_v3_m512_gemm2_ok : setup_deepseek_v3_m512_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- DeepSeek-V3 at TP1 (`E = 256`, `top_k = 8`, hidden `7168`, intermediate `2048`), `M = 40000`, the first GEMM (`N = 4096`, `K = 7168`): the tile `(128, 128, 64, 16)` of `get_default_config`, `moe_align_block_size`. -/
def setup_deepseek_v3_m40000_gemm1 : Inst := gemm1Inst 40000 8 256 256 4096 7168 false false (128, 128, 64, 16)
theorem setup_deepseek_v3_m40000_gemm1_ok : setup_deepseek_v3_m40000_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 7168`, `K = 2048`). -/
def setup_deepseek_v3_m40000_gemm2 : Inst := gemm2Inst 40000 8 256 256 7168 2048 false false (128, 128, 64, 16)
theorem setup_deepseek_v3_m40000_gemm2_ok : setup_deepseek_v3_m40000_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Qwen3-30B-A3B at TP1 (`E = 128`, `top_k = 8`, hidden `2048`, intermediate `768`), `M = 1`, the first GEMM (`N = 1536`, `K = 2048`): the tile `(16, 64, 64, 1)` of the tuned file `E=128,N=768`, the naive assignment. -/
def setup_qwen3_30b_m1_gemm1 : Inst := gemm1Inst 1 8 128 128 1536 2048 false false (16, 64, 64, 1)
theorem setup_qwen3_30b_m1_gemm1_ok : setup_qwen3_30b_m1_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 2048`, `K = 768`). -/
def setup_qwen3_30b_m1_gemm2 : Inst := gemm2Inst 1 8 128 128 2048 768 false false (16, 64, 64, 1)
theorem setup_qwen3_30b_m1_gemm2_ok : setup_qwen3_30b_m1_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Qwen3-30B-A3B at TP1 (`E = 128`, `top_k = 8`, hidden `2048`, intermediate `768`), `M = 4`, the first GEMM (`N = 1536`, `K = 2048`): the tile `(16, 64, 128, 16)` of the tuned file `E=128,N=768`, the naive assignment. -/
def setup_qwen3_30b_m4_gemm1 : Inst := gemm1Inst 4 8 128 128 1536 2048 false false (16, 64, 128, 16)
theorem setup_qwen3_30b_m4_gemm1_ok : setup_qwen3_30b_m4_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 2048`, `K = 768`). -/
def setup_qwen3_30b_m4_gemm2 : Inst := gemm2Inst 4 8 128 128 2048 768 false false (16, 64, 128, 16)
theorem setup_qwen3_30b_m4_gemm2_ok : setup_qwen3_30b_m4_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Qwen3-30B-A3B at TP1 (`E = 128`, `top_k = 8`, hidden `2048`, intermediate `768`), `M = 80`, the first GEMM (`N = 1536`, `K = 2048`): the tile `(16, 256, 128, 1)` of the tuned file `E=128,N=768`, `moe_align_block_size`. -/
def setup_qwen3_30b_m80_gemm1 : Inst := gemm1Inst 80 8 128 128 1536 2048 false false (16, 256, 128, 1)
theorem setup_qwen3_30b_m80_gemm1_ok : setup_qwen3_30b_m80_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 2048`, `K = 768`). -/
def setup_qwen3_30b_m80_gemm2 : Inst := gemm2Inst 80 8 128 128 2048 768 false false (16, 256, 128, 1)
theorem setup_qwen3_30b_m80_gemm2_ok : setup_qwen3_30b_m80_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Qwen3-30B-A3B at TP1 (`E = 128`, `top_k = 8`, hidden `2048`, intermediate `768`), `M = 32768`, the first GEMM (`N = 1536`, `K = 2048`): the tile `(128, 256, 64, 16)` of the tuned file `E=128,N=768`, `moe_align_block_size`. -/
def setup_qwen3_30b_m32768_gemm1 : Inst := gemm1Inst 32768 8 128 128 1536 2048 false false (128, 256, 64, 16)
theorem setup_qwen3_30b_m32768_gemm1_ok : setup_qwen3_30b_m32768_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 2048`, `K = 768`). -/
def setup_qwen3_30b_m32768_gemm2 : Inst := gemm2Inst 32768 8 128 128 2048 768 false false (128, 256, 64, 16)
theorem setup_qwen3_30b_m32768_gemm2_ok : setup_qwen3_30b_m32768_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Llama-4-Scout at TP8 (`E = 16`, `top_k = 1`, hidden `5120`, intermediate `1024` per rank, the routing weights on the input), `M = 1`, the first GEMM (`N = 2048`, `K = 5120`): the tile `(16, 32, 64, 1)` of the tuned file `E=16,N=1024`, the naive assignment. -/
def setup_llama4_scout_tp8_m1_gemm1 : Inst := gemm1Inst 1 1 16 16 2048 5120 true false (16, 32, 64, 1)
theorem setup_llama4_scout_tp8_m1_gemm1_ok : setup_llama4_scout_tp8_m1_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 5120`, `K = 1024`). -/
def setup_llama4_scout_tp8_m1_gemm2 : Inst := gemm2Inst 1 1 16 16 5120 1024 true false (16, 32, 64, 1)
theorem setup_llama4_scout_tp8_m1_gemm2_ok : setup_llama4_scout_tp8_m1_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Llama-4-Scout at TP8 (`E = 16`, `top_k = 1`, hidden `5120`, intermediate `1024` per rank, the routing weights on the input), `M = 64`, the first GEMM (`N = 2048`, `K = 5120`): the tile `(16, 64, 128, 1)` of the tuned file `E=16,N=1024`, `moe_align_block_size`. -/
def setup_llama4_scout_tp8_m64_gemm1 : Inst := gemm1Inst 64 1 16 16 2048 5120 true false (16, 64, 128, 1)
theorem setup_llama4_scout_tp8_m64_gemm1_ok : setup_llama4_scout_tp8_m64_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- The second GEMM (`N = 5120`, `K = 1024`). -/
def setup_llama4_scout_tp8_m64_gemm2 : Inst := gemm2Inst 64 1 16 16 5120 1024 true false (16, 64, 128, 1)
theorem setup_llama4_scout_tp8_m64_gemm2_ok : setup_llama4_scout_tp8_m64_gemm2.ok :=
  hostInst_ok (by norm_num [gemm2Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- Qwen3-30B-A3B, `M = 32768`, with the routing weights on the input: the first GEMM with
`MUL_ROUTED_WEIGHT`. -/
def setup_qwen3_30b_m32768_rw_gemm1 : Inst := gemm1Inst 32768 8 128 128 1536 2048 true false (128, 256, 64, 16)
theorem setup_qwen3_30b_m32768_rw_gemm1_ok : setup_qwen3_30b_m32768_rw_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])

/-- One expert, `top_k = 1`, `M = 300`: `M // E > 128` gives `GROUP_SIZE_M = 16` with the tile
`(64, 128, 64, 16)` of `get_default_config`. -/
def setup_e1_top1_m300_gemm1 : Inst := gemm1Inst 300 1 1 1 384 1024 false false (64, 128, 64, 16)
theorem setup_e1_top1_m300_gemm1_ok : setup_e1_top1_m300_gemm1.ok :=
  hostInst_ok (by norm_num [gemm1Inst, naiveOf, slenOf, elenOf, emOf, cfgs])


/-! ## Not vacuous: a covered instance and memory -/

/-- The all-zero memory: every byte present and `0`. -/
def zeroMem : Mem := fun _ => some (0 : TVA.Math.Bytes.Byte)

/-- An aligned `int32` read of the all-zero memory is `0`. -/
theorem zeroMem_read_i32 (a : Int) (hA : Mem.Aligned DType.i32 a) :
    zeroMem.read DType.i32 a = some (IntV.ofNat 0) := by
  unfold Mem.read
  simp only [hA, ↓reduceIte, show DType.i32.bytes = 4 from rfl]
  simp only [Mem.readBytes, zeroMem]
  rfl

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

/-- The host's first test case with the all-zero memory: `num_tokens_post_padded` reads `0`,
so every program returns at once, and every content condition of `Pre` is vacuous. -/
theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_gemm1_m1_n256_k128_e8_top2, zeroMem, setup_gemm1_m1_n256_k128_e8_top2_ok, ?_⟩
  have hrd : zeroMem.read DType.i32 (setup_gemm1_m1_n256_k128_e8_top2.ntpb.ptr 0).addr =
      some (IntV.ofNat 0) := by
    refine zeroMem_read_i32 _ ?_
    show 0 ≤ _ ∧ _ ∧ _
    simp only [setup_gemm1_m1_n256_k128_e8_top2, gemm1Inst, hostInst, Buf.ptr, DType.bytes]
    norm_num
  have hz : (IntV.ofNat 0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)
  have hntp0 : ntpOf setup_gemm1_m1_n256_k128_e8_top2 zeroMem = 0 := by
    unfold ntpOf
    rw [hrd]
    show (if 0 ≤ (IntV.ofNat 0 : I32).toInt then ((IntV.ofNat 0 : I32).toInt).toNat else 0) = 0
    rw [hz]
    rfl
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ⟨IntV.ofNat 0, hrd, by rw [hz], by rw [hz, hntp0]; rfl⟩,
    by rw [hntp0]; exact dvd_zero _,
    by rw [hntp0]; exact Nat.zero_le _,
    fun pos hp => by rw [hntp0] at hp; exact absurd hp (Nat.not_lt_zero _),
    fun pos pos' hp _ _ _ => by rw [hntp0] at hp; exact absurd hp (Nat.not_lt_zero _),
    fun pid_m hp => by rw [hntp0] at hp; exact absurd hp (Nat.not_lt_zero _),
    fun pid_m hp => by rw [hntp0] at hp; exact absurd hp (Nat.not_lt_zero _),
    fun h => by simp [setup_gemm1_m1_n256_k128_e8_top2, gemm1Inst, hostInst] at h⟩
  all_goals
    refine zeroValid _ ⟨_, ?_, rfl⟩ ?_
    all_goals norm_num [setup_gemm1_m1_n256_k128_e8_top2, gemm1Inst, hostInst, naiveOf, slenOf, elenOf, defaultCfg, DType.bytes]

end Claims.FusedMoe
