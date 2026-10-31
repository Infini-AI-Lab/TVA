import TVA

/-!
# `attn_bwd_f16`: the model, the conditions and the launch

Source: `source/06-fused-attention.py` (tutorial 06). The kernel is `_attn_bwd` (lines 366–500),
with its two helpers `_attn_bwd_dkdv` (lines 266–315) and `_attn_bwd_dq` (lines 319–363), which
the front end inlines at their calls. It reads the module-level names `triton` (its decorators)
and `tl` (lines 20–21), and the module's `@triton.jit` functions `_attn_bwd_dkdv` and
`_attn_bwd_dq`, each bound only by its own `def`. The host's launch is in
`_attention.backward` (lines 608–621).

This module holds the model (`attn_bwd_dkdv`, `attn_bwd_dq`, `attn_bwd`, `model`, `launch`) and the
conditions (`Inst`, `Inst.ok`, `Pre`). The specification and the assumptions are in
`AttnBwd.Spec`.
-/

namespace Kernels.AttnBwd

open TVA

/-! ## The model -/

/-- The configuration of every `tl.dot` of the kernel (lines 293, 303, 307, 310, 345, 353, 358):
`float16` operands (each operand is a `float16` tile: a loaded `float16` block, or the result of
`.to(tl.float16)`), a `float32` accumulator (the default `out_dtype`, and the `float32`
accumulators `dk`, `dv`, `dq`), no `input_precision` (the default `tf32`) and no
`max_num_imprecise_acc` (`0`). -/
abbrev dotCfg : DotCfg := { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-- `_attn_bwd_dkdv` (source lines 266–315), as the front end inlines it at a call: the body with
the call's arguments. The three `tl.constexpr` parameters that give shapes, `BLOCK_M1`,
`BLOCK_N1` and `HEAD_DIM`, come first here (the tensors' types depend on them); the others are in
the source's order. `sm_scale`, `H` and `N_CTX` are passed and not used (the source's body does
not read them). `start_n`, `start_m` and `num_steps` are `int32` scalars at both calls (names that
hold tensors: `start_n = pid * BLOCK_N1`, `start_m`, and `num_steps`, a name assigned a Python
integer or an `int32` quotient), and `MASK` is the compile-time `True` or `False` of the call.

The loop (line 288) carries the names defined before it that its body assigns: `dk`, `dv`,
`offs_m`, `curr_m`, `qT_ptrs`, `do_ptrs`, in that tuple. -/
def attn_bwd_dkdv (BLOCK_M1 BLOCK_N1 HEAD_DIM : Nat) (dk dv : Tensor [BLOCK_N1, HEAD_DIM] F32)
    (Q : Ptr .f16) (k v : Tensor [BLOCK_N1, HEAD_DIM] F16) (_sm_scale : F32) (DO : Ptr .f16)
    (M D : Ptr .f32) (stride_tok stride_d : I32) (_H _N_CTX : I32) (start_n start_m num_steps : I32)
    (MASK : Bool) : Prog (Tensor [BLOCK_N1, HEAD_DIM] F32 × Tensor [BLOCK_N1, HEAD_DIM] F32) := do
  -- line 279: the `int32` scalar `start_m` added to every lane of `tl.arange` (`int32`, wrapping)
  let offs_m : Tensor [BLOCK_M1] I32 := start_m + tl.arange 0 BLOCK_M1
  -- line 280
  let offs_n : Tensor [BLOCK_N1] I32 := start_n + tl.arange 0 BLOCK_N1
  -- line 281
  let offs_k : Tensor [HEAD_DIM] I32 := tl.arange 0 HEAD_DIM
  -- line 282: `(Q + offs_m[None, :] * stride_tok) + offs_k[:, None] * stride_d`: two pointer
  -- additions of `int32` products, broadcast to `[HEAD_DIM, BLOCK_M1]`
  let qT_ptrs := Q + offs_m.row HEAD_DIM * stride_tok + offs_k.col BLOCK_M1 * stride_d
  -- line 283: the same, `[BLOCK_M1, HEAD_DIM]`
  let do_ptrs := DO + offs_m.col HEAD_DIM * stride_tok + offs_k.row BLOCK_M1 * stride_d
  -- line 285: `tl.static_assert`, dropped (it holds)
  -- line 286: `curr_m` holds the `int32` tensor `start_m`
  let curr_m := start_m
  -- line 287: a name assigned the constexpr `BLOCK_M1` holds an `int32` scalar tensor
  let step_m : I32 := IntV.ofNat BLOCK_M1
  -- line 288: `range(num_steps)`: bounds `0` and the `int32` `num_steps` (a name: nothing to
  -- evaluate twice), step `1`; an `int32` induction variable, not used
  let (dk, dv, _offs_m, _curr_m, _qT_ptrs, _do_ptrs) ← tl.forRange (IntV.ofInt 0 : I32) num_steps
      (IntV.ofInt 1 : I32) (dk, dv, offs_m, curr_m, qT_ptrs, do_ptrs)
      fun _blk_idx (dk, dv, _offs_m, curr_m, qT_ptrs, do_ptrs) => do
    -- line 289: no mask
    let qT ← tl.load qT_ptrs
    -- line 291
    let offs_m : Tensor [BLOCK_M1] I32 := curr_m + tl.arange 0 BLOCK_M1
    -- line 292: the `*fp32` pointer plus the `int32` offsets, no mask
    let m ← tl.load (M + offs_m)
    -- line 293: `tl.dot(k, qT)` with the default zero accumulator
    let qkT ← tl.dot dotCfg k qT (tl.zeros [BLOCK_N1, BLOCK_M1] .f32)
    -- line 294: `qkT - m[None, :]`, `m` broadcast to the rows; a `tl.dot` result feeds the
    -- subtraction, so it is the fusable `tl.subc`; then `tl.math.exp2` on `float32`
    let pT ← tl.exp2 (← tl.subc qkT (m.row BLOCK_N1))
    -- lines 296–298: `if MASK` is compile-time; the mask `offs_m[None, :] >= offs_n[:, None]`
    -- (signed `int32`, written `offs_n[:, None] <= offs_m[None, :]`), then
    -- `tl.where(mask, pT, 0.0)`: the `0.0` is built in `pT`'s dtype `float32` (`+0.0`)
    let pT := if MASK then
        let mask := (offs_n.col BLOCK_M1).le (offs_m.row BLOCK_N1)
        tl.where mask pT (tl.full [BLOCK_N1, BLOCK_M1] (FloatV.litF32 (FloatV.ofPat 0)))
      else pT
    -- line 299: no mask
    let «do» ← tl.load do_ptrs
    -- line 301
    let ppT := pT
    -- line 302: `float32` to `float16`, one rounding to nearest even
    let ppT : Tensor [BLOCK_N1, BLOCK_M1] F16 ← tl.cvt ppT
    -- line 303: `dv + tl.dot(ppT, do)`, a `tl.dot` with the default zero accumulator added to
    -- `dv`, its only use: `tl.dot_add`
    let dv ← tl.dot_add dotCfg ppT «do» dv
    -- line 305
    let Di ← tl.load (D + offs_m)
    -- line 307: `tl.dot(v, tl.trans(do))` with the zero accumulator, then `.to(tl.float32)` of a
    -- `float32` tensor (`tl.convert`: the same value)
    let dpT ← tl.convert .f32 .f32 (← tl.dot dotCfg v «do».trans (tl.zeros [BLOCK_N1, BLOCK_M1] .f32))
    -- line 308: `dpT - Di[None, :]` (a `tl.dot` result feeds it: `tl.subc`), then the IEEE
    -- product with `pT`
    let dsT ← tl.mul pT (← tl.subc dpT (Di.row BLOCK_N1))
    -- line 309: `float32` to `float16`
    let dsT : Tensor [BLOCK_N1, BLOCK_M1] F16 ← tl.cvt dsT
    -- line 310: `dk + tl.dot(dsT, tl.trans(qT))`: `tl.dot_add`
    let dk ← tl.dot_add dotCfg dsT qT.trans dk
    -- line 312: `int32` addition
    let curr_m := curr_m + step_m
    -- lines 313–314: the `int32` product `step_m * stride_tok`, added to every pointer
    let qT_ptrs := qT_ptrs + (bc (step_m * stride_tok) : Tensor [HEAD_DIM, BLOCK_M1] I32)
    let do_ptrs := do_ptrs + (bc (step_m * stride_tok) : Tensor [BLOCK_M1, HEAD_DIM] I32)
    return (dk, dv, offs_m, curr_m, qT_ptrs, do_ptrs)
  -- line 315
  return (dk, dv)

/-- `_attn_bwd_dq` (source lines 319–363), as the front end inlines it at a call. The shape
constexprs `BLOCK_M2`, `BLOCK_N2` and `HEAD_DIM` come first; the others are in the source's order.
`H` and `N_CTX` are passed and not used. `m` is the `[BLOCK_M2, 1]` tensor `m[:, None]` of the
kernel's line 466.

The loop (line 342) carries the names defined before it that its body assigns: `dq`, `offs_n`
(assigned at line 349 when `MASK`), `curr_n`, `kT_ptrs`, `vT_ptrs`. -/
def attn_bwd_dq (BLOCK_M2 BLOCK_N2 HEAD_DIM : Nat) (dq : Tensor [BLOCK_M2, HEAD_DIM] F32)
    (q : Tensor [BLOCK_M2, HEAD_DIM] F16) (K V : Ptr .f16) («do» : Tensor [BLOCK_M2, HEAD_DIM] F16)
    (m : Tensor [BLOCK_M2, 1] F32) (D : Ptr .f32) (stride_tok stride_d : I32) (_H _N_CTX : I32)
    (start_m start_n num_steps : I32) (MASK : Bool) : Prog (Tensor [BLOCK_M2, HEAD_DIM] F32) := do
  -- line 331
  let offs_m : Tensor [BLOCK_M2] I32 := start_m + tl.arange 0 BLOCK_M2
  -- line 332
  let offs_n : Tensor [BLOCK_N2] I32 := start_n + tl.arange 0 BLOCK_N2
  -- line 333
  let offs_k : Tensor [HEAD_DIM] I32 := tl.arange 0 HEAD_DIM
  -- lines 334–335: `[HEAD_DIM, BLOCK_N2]` pointer tiles
  let kT_ptrs := K + offs_n.row HEAD_DIM * stride_tok + offs_k.col BLOCK_N2 * stride_d
  let vT_ptrs := V + offs_n.row HEAD_DIM * stride_tok + offs_k.col BLOCK_N2 * stride_d
  -- line 337
  let Di ← tl.load (D + offs_m)
  -- line 339: `tl.static_assert`, dropped (it holds)
  -- line 340
  let curr_n := start_n
  -- line 341: an `int32` scalar tensor
  let step_n : I32 := IntV.ofNat BLOCK_N2
  -- line 342: `range(num_steps)`, as in `attn_bwd_dkdv`
  let (dq, _offs_n, _curr_n, _kT_ptrs, _vT_ptrs) ← tl.forRange (IntV.ofInt 0 : I32) num_steps
      (IntV.ofInt 1 : I32) (dq, offs_n, curr_n, kT_ptrs, vT_ptrs)
      fun _blk_idx (dq, offs_n, curr_n, kT_ptrs, vT_ptrs) => do
    -- lines 343–344
    let kT ← tl.load kT_ptrs
    let vT ← tl.load vT_ptrs
    -- line 345
    let qk ← tl.dot dotCfg q kT (tl.zeros [BLOCK_M2, BLOCK_N2] .f32)
    -- line 346: `qk - m`, `m` (`[BLOCK_M2, 1]`) broadcast to the columns; `tl.subc` (a `tl.dot`
    -- feeds it), then `tl.math.exp2`
    let p ← tl.exp2 (← tl.subc qk (m.bcastCols BLOCK_N2))
    -- lines 348–351: compile-time `if MASK`, whose body assigns `offs_n` and `p`; line 349:
    -- `offs_n = curr_n + tl.arange(0, BLOCK_N2)`
    let offs_n : Tensor [BLOCK_N2] I32 := if MASK then curr_n + tl.arange 0 BLOCK_N2 else offs_n
    -- lines 350–351: the mask `offs_m[:, None] >= offs_n[None, :]` (signed `int32`, written
    -- `offs_n[None, :] <= offs_m[:, None]`), then `tl.where(mask, p, 0.0)`, the `0.0` built in
    -- `float32`
    let p := if MASK then
        let mask := (offs_n.row BLOCK_M2).le (offs_m.col BLOCK_N2)
        tl.where mask p (tl.full [BLOCK_M2, BLOCK_N2] (FloatV.litF32 (FloatV.ofPat 0)))
      else p
    -- line 353: `tl.dot(do, vT)`, then `.to(tl.float32)` of a `float32` tensor
    let dp ← tl.convert .f32 .f32 (← tl.dot dotCfg «do» vT (tl.zeros [BLOCK_M2, BLOCK_N2] .f32))
    -- line 354: `dp - Di[:, None]` (`tl.subc`), then the IEEE product with `p`
    let ds ← tl.mul p (← tl.subc dp (Di.col BLOCK_N2))
    -- line 355
    let ds : Tensor [BLOCK_M2, BLOCK_N2] F16 ← tl.cvt ds
    -- line 358: `dq + tl.dot(ds, tl.trans(kT))`: `tl.dot_add`
    let dq ← tl.dot_add dotCfg ds kT.trans dq
    -- lines 360–362
    let curr_n := curr_n + step_n
    let kT_ptrs := kT_ptrs + (bc (step_n * stride_tok) : Tensor [HEAD_DIM, BLOCK_N2] I32)
    let vT_ptrs := vT_ptrs + (bc (step_n * stride_tok) : Tensor [HEAD_DIM, BLOCK_N2] I32)
    return (dq, offs_n, curr_n, kT_ptrs, vT_ptrs)
  -- line 363
  return dq

/-- The `float64` pattern of the Python float `0.6931471824645996` (`LN2`, line 381). -/
def ln2Pat : Nat := 0x3FE62E4300000000

/-- `_attn_bwd` (source lines 366–500). The pointers are `*fp16` (`Q`, `K`, `V`, `DO`, `DQ`,
`DK`, `DV`: the host passes `float16` tensors) and `*fp32` (`M`, `D`); `sm_scale` is a runtime
`float32` (a Python `float` argument: its double rounded to nearest, `FloatV.hostF32`); the six
integers are `int32` (see `model`); the six constexprs are Lean parameters, and `CAUSAL` the
compile-time `bool` the host passes. `LN2` (line 381) is a compile-time float, used at line 499. -/
def attn_bwd (Q K V : Ptr .f16) (sm_scale : F32) (DO DQ DK DV : Ptr .f16) (M D : Ptr .f32)
    (stride_z stride_h stride_tok stride_d H N_CTX : I32)
    (BLOCK_M1 BLOCK_N1 BLOCK_M2 BLOCK_N2 BLK_SLICE_FACTOR HEAD_DIM : Nat) (CAUSAL : Bool) : Kernel where
  body ctx := do
    -- line 381: `LN2: tl.constexpr = 0.6931471824645996`, a compile-time value (no step)
    -- line 383: axis 2 of the grid, an `int32`
    let bhid := tl.program_id ctx 2
    -- line 384: the `int32` product `bhid * N_CTX`, then `.to(tl.int64)` (sign extension)
    let off_chz : I64 := IntV.cast (bhid * N_CTX)
    -- line 385: `bhid % H` and `bhid // H` (`int32` `%` and `//`, in that order), the `int32`
    -- products and sum, then `.to(tl.int64)`
    let adj : I64 := IntV.cast (stride_h * (← tl.irem (Tensor.scalar bhid) (Tensor.scalar H)).item +
      stride_z * (← tl.idiv (Tensor.scalar bhid) (Tensor.scalar H)).item)
    -- line 386
    let pid := tl.program_id ctx 0
    -- lines 389–397: each pointer plus an `int64` scalar
    let Q := Q + adj
    let K := K + adj
    let V := V + adj
    let DO := DO + adj
    let DQ := DQ + adj
    let DK := DK + adj
    let DV := DV + adj
    let M := M + off_chz
    let D := D + off_chz
    -- line 400
    let offs_k : Tensor [HEAD_DIM] I32 := tl.arange 0 HEAD_DIM
    -- line 402: the `int32` `pid` times the constexpr, built in `int32`
    let start_n := pid * (IntV.ofNat BLOCK_N1 : I32)
    -- line 403: a name assigned the Python `0` holds an `int32` scalar tensor
    let start_m : I32 := IntV.ofInt 0
    -- line 405: a compile-time value (Python's `//` of two constexprs)
    let MASK_BLOCK_M1 := BLOCK_M1 / BLK_SLICE_FACTOR
    -- line 406
    let offs_n : Tensor [BLOCK_N1] I32 := start_n + tl.arange 0 BLOCK_N1
    -- lines 408–409: `+0.0`
    let dv := tl.zeros [BLOCK_N1, HEAD_DIM] .f32
    let dk := tl.zeros [BLOCK_N1, HEAD_DIM] .f32
    -- lines 412–413: `[BLOCK_N1, HEAD_DIM]` tiles, no mask
    let k ← tl.load (K + offs_n.col HEAD_DIM * stride_tok + offs_k.row BLOCK_N1 * stride_d)
    let v ← tl.load (V + offs_n.col HEAD_DIM * stride_tok + offs_k.row BLOCK_N1 * stride_d)
    -- lines 415–429: `if CAUSAL` is compile-time. `num_steps = BLOCK_N1 // MASK_BLOCK_M1` is
    -- Python's `//` of two constexprs, and the name then holds an `int32` scalar tensor; the call
    -- passes `MASK_BLOCK_M1` as the helper's `BLOCK_M1` and `MASK=True`; then `int32` arithmetic
    let (dk, dv, start_m) ← (if CAUSAL then do
        let start_m := start_n
        let num_steps : I32 := IntV.ofNat (BLOCK_N1 / MASK_BLOCK_M1)
        let (dk, dv) ← attn_bwd_dkdv MASK_BLOCK_M1 BLOCK_N1 HEAD_DIM dk dv Q k v sm_scale DO M D
          stride_tok stride_d H N_CTX start_n start_m num_steps true
        let start_m := start_m + num_steps * (IntV.ofNat MASK_BLOCK_M1 : I32)
        pure (dk, dv, start_m)
      else pure (dk, dv, start_m))
    -- line 432: `int32` `-`, then `//` by the constexpr built in `int32` (`tl.idiv`)
    let num_steps := (← tl.idiv (Tensor.scalar (N_CTX - start_m))
      (Tensor.scalar (IntV.ofNat BLOCK_M1 : I32))).item
    -- lines 433–443
    let (dk, dv) ← attn_bwd_dkdv BLOCK_M1 BLOCK_N1 HEAD_DIM dk dv Q k v sm_scale DO M D stride_tok
      stride_d H N_CTX start_n start_m num_steps false
    -- line 445
    let dv_ptrs := DV + offs_n.col HEAD_DIM * stride_tok + offs_k.row BLOCK_N1 * stride_d
    -- line 446: `dv` converted to the pointer's dtype `float16` (one rounding to nearest even)
    tl.store dv_ptrs (← tl.cvt dv)
    -- line 449: `dk * sm_scale`, the `float32` scalar broadcast, an IEEE product
    let dk ← tl.mul dk (bc sm_scale)
    -- line 450
    let dk_ptrs := DK + offs_n.col HEAD_DIM * stride_tok + offs_k.row BLOCK_N1 * stride_d
    -- line 451: converted to `float16`
    tl.store dk_ptrs (← tl.cvt dk)
    -- line 454
    let start_m := pid * (IntV.ofNat BLOCK_M2 : I32)
    -- line 455: an `int32` scalar tensor
    let start_n : I32 := IntV.ofInt 0
    -- line 456: the runtime `int32` `N_CTX` `//` the constexpr, built in `int32`
    let num_steps := (← tl.idiv (Tensor.scalar N_CTX) (Tensor.scalar (IntV.ofNat BLOCK_N2 : I32))).item
    -- line 458: a compile-time value
    let MASK_BLOCK_N2 := BLOCK_N2 / BLK_SLICE_FACTOR
    -- line 459
    let offs_m : Tensor [BLOCK_M2] I32 := start_m + tl.arange 0 BLOCK_M2
    -- line 461
    let q ← tl.load (Q + offs_m.col HEAD_DIM * stride_tok + offs_k.row BLOCK_M2 * stride_d)
    -- line 462
    let dq := tl.zeros [BLOCK_M2, HEAD_DIM] .f32
    -- line 463
    let «do» ← tl.load (DO + offs_m.col HEAD_DIM * stride_tok + offs_k.row BLOCK_M2 * stride_d)
    -- line 465
    let m ← tl.load (M + offs_m)
    -- line 466: `m[:, None]`, shape `[BLOCK_M2, 1]`
    let m := m.unsqueeze1
    -- lines 468–487: compile-time `if CAUSAL`; `num_steps = BLOCK_M2 // MASK_BLOCK_N2` (Python's
    -- `//`, then an `int32` tensor); the third argument of the call is evaluated as an `int32`
    -- expression; then `int32` arithmetic and `//`
    let (dq, start_n, num_steps) ← (if CAUSAL then do
        let end_n := start_m + (IntV.ofNat BLOCK_M2 : I32)
        let num_steps : I32 := IntV.ofNat (BLOCK_M2 / MASK_BLOCK_N2)
        let dq ← attn_bwd_dq BLOCK_M2 MASK_BLOCK_N2 HEAD_DIM dq q K V «do» m D stride_tok stride_d H
          N_CTX start_m (end_n - num_steps * (IntV.ofNat MASK_BLOCK_N2 : I32)) num_steps true
        let end_n := end_n - num_steps * (IntV.ofNat MASK_BLOCK_N2 : I32)
        let num_steps := (← tl.idiv (Tensor.scalar end_n) (Tensor.scalar (IntV.ofNat BLOCK_N2 : I32))).item
        let start_n := end_n - num_steps * (IntV.ofNat BLOCK_N2 : I32)
        pure (dq, start_n, num_steps)
      else pure (dq, start_n, num_steps))
    -- lines 489–496
    let dq ← attn_bwd_dq BLOCK_M2 BLOCK_N2 HEAD_DIM dq q K V «do» m D stride_tok stride_d H N_CTX
      start_m start_n num_steps false
    -- line 498
    let dq_ptrs := DQ + offs_m.col HEAD_DIM * stride_tok + offs_k.row BLOCK_M2 * stride_d
    -- line 499: `dq * LN2`, the compile-time float built in `dq`'s dtype `float32`
    -- (`FloatV.litF32` of the double `0.6931471824645996`), an IEEE product
    let dq ← tl.mul dq (bc (FloatV.litF32 (FloatV.ofPat ln2Pat)))
    -- line 500: converted to `float16`
    tl.store dq_ptrs (← tl.cvt dq)

/-! ## The conditions -/

/-- What the host chooses: one launch of `_attn_bwd` through `_attention.backward` (lines
582–623). The tensors `q`, `arg_k`, `v`, `do`, `dq`, `dk`, `dv` have the shape
`(BATCH, N_HEAD, N_CTX, HEAD_DIM)` and the strides of a contiguous tensor (`do.is_contiguous()`
and `q.stride() == k.stride() == v.stride() == o.stride() == do.stride()`, lines 585–586; `dq`,
`dk`, `dv` are `torch.empty_like` of `q`, `k`, `v`, and `arg_k = k * (…)` keeps `k`'s strides);
`M` and `delta` have the shape `(BATCH, N_HEAD, N_CTX)`, contiguous (`torch.empty` and
`torch.empty_like(M)`). So element `(z, h, r, c)` of a 4-D tensor is element
`((z·N_HEAD + h)·N_CTX + r)·HEAD_DIM + c` of its storage (`eIdx`), and element `(z, h, r)` of
`M` or `delta` is element `(z·N_HEAD + h)·N_CTX + r` (`rIdx`). -/
structure Inst where
  /-- `BATCH` (`q.shape[0]`). -/
  Z : Nat
  /-- `N_HEAD` (`q.shape[1]`), the kernel's `H`. -/
  H : Nat
  /-- `N_CTX` (`q.shape[2]`). -/
  N_CTX : Nat
  /-- `HEAD_DIM` (`ctx.HEAD_DIM`, the last dimension). -/
  HEAD_DIM : Nat
  /-- `CAUSAL` (`ctx.causal`, the `causal` argument of the forward). -/
  CAUSAL : Bool
  /-- `sm_scale` (`ctx.sm_scale`), the Python `float` the host passes, as a double. -/
  sm_scale : F64
  /-- `Q`: the storage of `q`, `float16`. -/
  Q : Buf .f16
  /-- `K`: the storage of `arg_k = k * (sm_scale * RCP_LN2)` (line 597), `float16`. -/
  K : Buf .f16
  /-- `V`: the storage of `v`, `float16`. -/
  V : Buf .f16
  /-- `DO`: the storage of `do`, `float16`. -/
  DO : Buf .f16
  /-- `DQ`: the storage of `dq`, `float16`. -/
  DQ : Buf .f16
  /-- `DK`: the storage of `dk`, `float16`. -/
  DK : Buf .f16
  /-- `DV`: the storage of `dv`, `float16`. -/
  DV : Buf .f16
  /-- `M`: the storage of the forward's `M`, `float32`. -/
  M : Buf .f32
  /-- `D`: the storage of `delta`, `float32`. -/
  D : Buf .f32

/-- The number of heads, `BATCH · N_HEAD`: the grid's axis 2. -/
def G (d : Inst) : Nat := d.Z * d.H

/-- Element `(b, r, c)` of a 4-D tensor (head `b = z·N_HEAD + h`, row `r`, column `c`). -/
def eIdx (d : Inst) (b r c : Nat) : Nat := (b * d.N_CTX + r) * d.HEAD_DIM + c

/-- Element `(b, r)` of `M` or `delta`. -/
def rIdx (d : Inst) (b r : Nat) : Nat := b * d.N_CTX + r

/-- The elements of a 4-D tensor's storage that make up the tensor. -/
def tSet (d : Inst) : Set Nat := {e | e < G d * d.N_CTX * d.HEAD_DIM}

/-- The elements of `M`'s or `delta`'s storage that make up the tensor. -/
def rSet (d : Inst) : Set Nat := {e | e < G d * d.N_CTX}

/-- The bytes of the six input tensors. -/
def inBytes (d : Inst) : Set Int :=
  d.Q.elems (tSet d) ∪ d.K.elems (tSet d) ∪ d.V.elems (tSet d) ∪ d.DO.elems (tSet d) ∪
    d.M.elems (rSet d) ∪ d.D.elems (rSet d)

/-- The bytes of the three output tensors. -/
def outBytes (d : Inst) : Set Int :=
  d.DQ.elems (tSet d) ∪ d.DK.elems (tSet d) ∪ d.DV.elems (tSet d)

/-- The `float32` the kernel receives for `sm_scale`: the double rounded to nearest even. -/
def smF (d : Inst) : F32 := FloatV.hostF32 d.sm_scale

/-- The instances covered. Shapes are variables; the bounds are the host's, those that keep the
kernel's `int32` arithmetic from wrapping, and `N_CTX ≤ 2^14`, the host's largest, where the
tolerances of `Post` are still small (README, Specification). -/
structure Inst.ok (d : Inst) : Prop where
  /-- `HEAD_DIM ∈ {16, 32, 64, 128, 256}`: the forward's assertion (line 512), which made `M`. -/
  hd : d.HEAD_DIM = 16 ∨ d.HEAD_DIM = 32 ∨ d.HEAD_DIM = 64 ∨ d.HEAD_DIM = 128 ∨ d.HEAD_DIM = 256
  /-- `N_CTX % 128 == 0`: the host's assertion (line 599). -/
  n_mod : d.N_CTX % 128 = 0
  /-- `N_CTX ≤ 2^14`: the benchmark's largest (`2^10 … 2^14`, line 718). -/
  n_le : d.N_CTX ≤ 2 ^ 14
  /-- `BATCH · N_HEAD < 2^16`: the grid's axis 2 (`Launch.Valid`). -/
  g_lt : G d < 2 ^ 16
  /-- `BATCH · N_HEAD · N_CTX · HEAD_DIM < 2^31`: the `int32` offset `adj` of line 385 (and
  `stride_z` itself) does not wrap. -/
  size_lt : G d * d.N_CTX * d.HEAD_DIM < 2 ^ 31
  /-- `sm_scale` is a finite `float32` of magnitude at most `2^16` (the host's `0.5` and `1.3`). -/
  sm : (smF d).isFinite = true ∧ |(smF d).toRat| ≤ 2 ^ 16
  /-- Each 4-D tensor's storage holds the tensor. -/
  q_len : G d * d.N_CTX * d.HEAD_DIM ≤ d.Q.len
  /-- As `q_len`. -/
  k_len : G d * d.N_CTX * d.HEAD_DIM ≤ d.K.len
  /-- As `q_len`. -/
  v_len : G d * d.N_CTX * d.HEAD_DIM ≤ d.V.len
  /-- As `q_len`. -/
  do_len : G d * d.N_CTX * d.HEAD_DIM ≤ d.DO.len
  /-- As `q_len`. -/
  dq_len : G d * d.N_CTX * d.HEAD_DIM ≤ d.DQ.len
  /-- As `q_len`. -/
  dk_len : G d * d.N_CTX * d.HEAD_DIM ≤ d.DK.len
  /-- As `q_len`. -/
  dv_len : G d * d.N_CTX * d.HEAD_DIM ≤ d.DV.len
  /-- `M`'s storage holds `M`. -/
  m_len : G d * d.N_CTX ≤ d.M.len
  /-- `delta`'s storage holds `delta`. -/
  d_len : G d * d.N_CTX ≤ d.D.len
  /-- The outputs share no byte with the inputs (`torch.empty_like` allocates them). -/
  out_in : Disjoint (outBytes d) (inBytes d)
  /-- `dq` and `dk` share no byte. -/
  dq_dk : Disjoint (d.DQ.elems (tSet d)) (d.DK.elems (tSet d))
  /-- `dq` and `dv` share no byte. -/
  dq_dv : Disjoint (d.DQ.elems (tSet d)) (d.DV.elems (tSet d))
  /-- `dk` and `dv` share no byte. -/
  dk_dv : Disjoint (d.DK.elems (tSet d)) (d.DV.elems (tSet d))

/-- The model on an instance: the buffers' first elements; `sm_scale` as the kernel receives it;
the strides `q.stride(0…3)` of the contiguous layout, `N_HEAD` and `N_CTX` as `int32`; and the
constexprs of the launch (lines 593–594, 617, 620): `BLOCK_M1 = 32`, `BLOCK_N1 = 128`,
`BLOCK_M2 = 128`, `BLOCK_N2 = 32`, `BLK_SLICE_FACTOR = 2`, `HEAD_DIM`, `CAUSAL`.

`stride_d = q.stride(3) = 1` is the compile-time constant `1`, and so is `N_HEAD` when it is `1`;
every use makes it an `int32` (an operand of `*`, `%` or `//` with an `int32` tensor builds it in
`int32`), so the model writes the `int32` `1`, which gives the same values. The others are
runtime `int32` arguments: they are positive and below `2^31` (`Inst.ok.size_lt`; `N_CTX` and
`stride_z = N_HEAD·N_CTX·HEAD_DIM` are at least `128` and `16·128`). -/
def model (d : Inst) : Kernel :=
  attn_bwd (d.Q.ptr 0) (d.K.ptr 0) (d.V.ptr 0) (smF d) (d.DO.ptr 0) (d.DQ.ptr 0) (d.DK.ptr 0)
    (d.DV.ptr 0) (d.M.ptr 0) (d.D.ptr 0) (IntV.ofNat (d.H * d.N_CTX * d.HEAD_DIM))
    (IntV.ofNat (d.N_CTX * d.HEAD_DIM)) (IntV.ofNat d.HEAD_DIM) (IntV.ofNat 1) (IntV.ofNat d.H)
    (IntV.ofNat d.N_CTX) 32 128 128 32 2 d.HEAD_DIM d.CAUSAL

/-- The block shapes of the source's values on an instance: `tl.arange` and the 1-D offsets
(`[16]`, `[32]`, `[128]`, `[HEAD_DIM]`), their `[:, None]` and `[None, :]` views, the pointer
tiles and loaded tiles, the `tl.dot` operands and results, their transposes, and the masks
(`BLOCK_M1 = 32` and `MASK_BLOCK_M1 = 16` in `_attn_bwd_dkdv`, `BLOCK_N2 = 32` and
`MASK_BLOCK_N2 = 16` in `_attn_bwd_dq`). -/
def blocks (d : Inst) : List Shape :=
  [[16], [32], [128], [d.HEAD_DIM], [1, 16], [1, 32], [16, 1], [32, 1], [128, 1], [1, d.HEAD_DIM],
    [d.HEAD_DIM, 1], [d.HEAD_DIM, 16], [d.HEAD_DIM, 32], [16, d.HEAD_DIM], [32, d.HEAD_DIM],
    [128, 16], [128, 32], [128, d.HEAD_DIM]]

/-- The host's launch (lines 608–621): the grid `(N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)`;
`enable_fp_fusion` is Triton's default (`true`); `num_ctas = 1`, `launch_pdl = False` and
`debug = false` are the defaults. `num_warps=4` and `num_stages=5` are not recorded
(performance-only). -/
def launch (d : Inst) : Launch :=
  { grid := ![d.N_CTX / 128, 1, G d], fusion := true, blocks := blocks d }

/-- The inputs covered: the nine buffers are valid (aligned, inside `[0, 2^64)`, every byte
present). Any values, NaN and infinities included; `Post` states what it needs of them. -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.Q.Valid init ∧ d.K.Valid init ∧ d.V.Valid init ∧ d.DO.Valid init ∧ d.DQ.Valid init ∧
    d.DK.Valid init ∧ d.DV.Valid init ∧ d.M.Valid init ∧ d.D.Valid init

end Kernels.AttnBwd
