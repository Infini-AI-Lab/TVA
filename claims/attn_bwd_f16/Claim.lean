import TVA

/-!
# `attn_bwd_f16`: the prepared claim

Source: `source/06-fused-attention.py` (tutorial 06). The kernel is
`_attn_bwd` (lines 366–500), with its two helpers `_attn_bwd_dkdv` (lines 266–315) and
`_attn_bwd_dq` (lines 319–363), which the front end inlines at their calls. It reads the
module-level names `triton` (its decorators) and `tl` (lines 20–21), and the module's
`@triton.jit` functions `_attn_bwd_dkdv` and `_attn_bwd_dq`, each bound only by its own `def`. The
host's launch is in `_attention.backward` (lines 608–621); the launches to cover are in
`SETUP.md` (the file's test and benchmark).

For one head `b`, write `Q`, `K`, `V`, `dO` for the head's `N_CTX × HEAD_DIM` blocks of the four
`float16` inputs (`K` is the host's `arg_k`), `L` for its row of `M` and `Δ` for its row of `delta`
(all as the launch finds them in memory). For a key row `n` and a query row `m`, with
`Allowed n m` (`CAUSAL` is off, or `n ≤ m`):

* `P[n, m] = 2^(Σ_c K[n, c]·Q[m, c] − L[m])` (`0` when not `Allowed`),
* `dS[n, m] = P[n, m]·(Σ_c V[n, c]·dO[m, c] − Δ[m])`,
* `dV[n, j] = Σ_m P[n, m]·dO[m, j]`,
* `dK[n, j] = sm_scale · Σ_m dS[n, m]·Q[m, j]`,
* `dQ[m, j] = LN2 · Σ_n dS[n, m]·K[n, j]` (`LN2` the kernel's `float32` constant).

`Post` says each stored `dV`, `dK`, `dQ` element is the `float16` rounding of a `float32` value
within an explicit tolerance of these exact values, whenever the head's inputs are finite and the
pairs fit (`HeadV.Ok`), and every byte outside the three outputs is unchanged.

* The model: `attn_bwd_dkdv`, `attn_bwd_dq`, `attn_bwd`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with `HeadV`, `HeadV.Ok`, `OutSpec`, the exact values `dVE`,
  `dKE`, `dQE` and the tolerances `dVTol`, `dKTol`, `dQTol`.
* The assumptions: `Assumptions`.
* The claim: `Statement`.
* Checked here: `consistent`, `satisfiable`, and the coverage facts `canon_ok`, `setup_*_ok`.
-/

namespace Claims.AttnBwd

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
  -- line 285: `tl.static_assert`, not modelled (it holds)
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
  -- line 339: `tl.static_assert`, not modelled (it holds)
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
  /-- `BATCH ≤ 8192`: `BATCH` is a batch size, the number of sequences (README, Exclusions). -/
  z_le : d.Z ≤ 8192
  /-- The kernel's `int32` arithmetic does not wrap: `(BATCH · N_HEAD − 1) · N_CTX · HEAD_DIM <
  2^31` bounds the offset `adj` of line 385 (`stride_h·(bhid % H) + stride_z·(bhid // H)`, at most
  the last head's first element) and `bhid * N_CTX` (line 384), a limit of the kernel;
  `N_HEAD · N_CTX · HEAD_DIM < 2^31` makes `stride_z = q.stride(0)` an `int32` argument, as the model
  takes it (beyond, outside real use: README, Exclusions). -/
  size_lt : (G d - 1) * d.N_CTX * d.HEAD_DIM < 2 ^ 31 ∧ d.H * d.N_CTX * d.HEAD_DIM < 2 ^ 31
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


/-! ## The constants -/

/-- `float32`'s unit roundoff, `2^-24`. -/
noncomputable def u32 : ℝ := (Math.Float.Format.binary32.u : ℝ)

/-- `float32`'s smallest subnormal over two, `2^-150`. -/
noncomputable def eta32 : ℝ := (Math.Float.Format.binary32.eta : ℝ)

/-- `float16`'s unit roundoff, `2^-11`. -/
noncomputable def u16 : ℝ := (Math.Float.Format.binary16.u : ℝ)

/-- `float16`'s smallest subnormal over two, `2^-25`. -/
noncomputable def eta16 : ℝ := (Math.Float.Format.binary16.eta : ℝ)

/-- The relative constant of one `tl.dot` over `HEAD_DIM` products (`Rules.Num.relDot16`,
`(HEAD_DIM + 2)·2^-22`). -/
noncomputable def rH (d : Inst) : ℝ := (Rules.Num.relDot16 d.HEAD_DIM : ℝ)

/-- The absolute constant of one `tl.dot` over `HEAD_DIM` products (`Rules.Num.absDot16`,
`(HEAD_DIM + 2)·2^-126`). -/
noncomputable def aH (d : Inst) : ℝ := (Rules.Num.absDot16 d.HEAD_DIM : ℝ)

/-- The relative error of one accumulation step `acc += tl.dot(a, b)` over at most `32`
products: `DotLoop.stepRel relDot16 32 = 34·2^-22 + 3·2^-24·(1 + 34·2^-22)` (about `8.3·10^-6`). -/
noncomputable def rhoA : ℝ := (DotLoop.stepRel Rules.Num.relDot16 32 : ℝ)

/-- The absolute error of one accumulation step: `DotLoop.stepAbs absDot16 32`
(about `34·2^-126`). -/
noncomputable def alphaA : ℝ := (DotLoop.stepAbs Rules.Num.absDot16 32 : ℝ)

/-- The error after `T` accumulation steps whose products' magnitudes sum to `B`:
`((1 + rhoA)^T − 1)·B + T·(1 + rhoA)^T·alphaA` (`DotLoop.bound`, over the reals). -/
noncomputable def accB (T : ℕ) (B : ℝ) : ℝ := ((1 + rhoA) ^ T - 1) * B + T * (1 + rhoA) ^ T * alphaA

/-- The `float32` constant `LN2` (line 381): `0x3f317218 = 1453635/2097152` (`ln2_val`). -/
def ln2F : F32 := FloatV.litF32 (FloatV.ofPat ln2Pat)

/-! ## The values of a head -/

/-- The values of head `b` the kernel reads: rows `r < N_CTX`, columns `c < HEAD_DIM`. -/
structure HeadV where
  /-- `Q[b, r, c]`. -/
  q : Nat → Nat → F16
  /-- `K[b, r, c]` (`arg_k`). -/
  k : Nat → Nat → F16
  /-- `V[b, r, c]`. -/
  v : Nat → Nat → F16
  /-- `DO[b, r, c]`. -/
  dO : Nat → Nat → F16
  /-- `M[b, r]`. -/
  lse : Nat → F32
  /-- `D[b, r]` (`delta`). -/
  dl : Nat → F32

/-- The initial memory holds `hv` in head `b`. -/
def HeadV.Holds (d : Inst) (init : Mem) (b : Nat) (hv : HeadV) : Prop :=
  ∀ r, r < d.N_CTX →
    (∀ c, c < d.HEAD_DIM →
      init.read .f16 (d.Q.ptr ↑(eIdx d b r c)).addr = some (hv.q r c) ∧
      init.read .f16 (d.K.ptr ↑(eIdx d b r c)).addr = some (hv.k r c) ∧
      init.read .f16 (d.V.ptr ↑(eIdx d b r c)).addr = some (hv.v r c) ∧
      init.read .f16 (d.DO.ptr ↑(eIdx d b r c)).addr = some (hv.dO r c)) ∧
    init.read .f32 (d.M.ptr ↑(rIdx d b r)).addr = some (hv.lse r) ∧
    init.read .f32 (d.D.ptr ↑(rIdx d b r)).addr = some (hv.dl r)

/-- Key row `n` and query row `m` interact: `CAUSAL` is off, or `n ≤ m` (the mask
`offs_m >= offs_n`, lines 297 and 350). -/
abbrev Allowed (d : Inst) (n m : Nat) : Prop := d.CAUSAL = false ∨ n ≤ m

/-- The lifted magnitude of a `float16` value, by which `Rules.Num.DotF16` counts a `tl.dot`
operand (`Rules.Num.liftMag`): `0` for `0`, `|x|` for a normal value, `2^-14` for a subnormal one. -/
noncomputable def lm (x : F16) : ℝ := (Rules.Num.liftMag .f16 x.toRat : ℝ)

/-- A bound on the lifted magnitude of a `float16` value of magnitude at most `B`: `0` when
`B < 2^-24` (the value is then `0`), else `max B 2^-14`. -/
noncomputable def lift16 (B : ℝ) : ℝ := if B < 2 ^ (-24 : ℤ) then 0 else max B (2 ^ (-14 : ℤ))

/-- `Σ_c K[n, c]·Q[m, c]`. -/
noncomputable def sQK (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, (hv.k n c).toReal * (hv.q m c).toReal

/-- `Σ_c lm K[n, c]·lm Q[m, c]`: the lifted magnitude of the dot (`Σ_c |K·Q|` when no operand is
subnormal). -/
noncomputable def aQK (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, lm (hv.k n c) * lm (hv.q m c)

/-- `Σ_c V[n, c]·dO[m, c]`. -/
noncomputable def sVD (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, (hv.v n c).toReal * (hv.dO m c).toReal

/-- `Σ_c lm V[n, c]·lm dO[m, c]`, as `aQK`. -/
noncomputable def aVD (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, lm (hv.v n c) * lm (hv.dO m c)

/-- The error bound of a computed `tl.dot(a, b) − x` (a dot over `HEAD_DIM` products of exact
sum `s` and lifted magnitude sum `a`, then the `float32` subtraction of `x`):
`rH·a + aH + 3u32·(|s| + rH·a + aH + |x|) + 4eta32`. -/
noncomputable def eSub (d : Inst) (s a x : ℝ) : ℝ :=
  rH d * a + aH d + 3 * u32 * (|s| + rH d * a + aH d + |x|) + 4 * eta32

/-- The exact exponent `Σ_c K[n, c]·Q[m, c] − L[m]`. -/
noncomputable def xE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := sQK d hv n m - (hv.lse m).toReal

/-- The bound on the computed exponent's error. -/
noncomputable def eX (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  eSub d (sQK d hv n m) (aQK d hv n m) (hv.lse m).toReal

/-- The exact `dP[n, m] − Δ[m] = Σ_c V[n, c]·dO[m, c] − Δ[m]`. -/
noncomputable def yE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := sVD d hv n m - (hv.dl m).toReal

/-- The bound on its computed value's error. -/
noncomputable def eY (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  eSub d (sVD d hv n m) (aVD d hv n m) (hv.dl m).toReal

/-- A bound on the computed `dP − Δ`'s magnitude. -/
noncomputable def yB (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := |yE d hv n m| + eY d hv n m

/-- A bound on the computed `tl.math.exp2`'s magnitude (`Rules.Num.Exp2F32`'s `2^-22` and
`2^-126`, at the largest exponent the error allows). -/
noncomputable def pBr (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  (1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (xE d hv n m + eX d hv n m) + 2 ^ (-126 : ℤ)

/-- The exact `P[n, m]`. -/
noncomputable def pE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then (2 : ℝ) ^ xE d hv n m else 0

/-- The exact `dS[n, m] = P[n, m]·(dP[n, m] − Δ[m])`. -/
noncomputable def dsE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := pE d hv n m * yE d hv n m

/-- The error bound of the `float16` `P` (line 302): the exponent's and `tl.math.exp2`'s errors
(`pBr − 2^x`), and one rounding to `float16`. `0` when not `Allowed` (the masked lanes are
exactly `0`). -/
noncomputable def p16Err (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then pBr d hv n m - (2 : ℝ) ^ xE d hv n m + u16 * pBr d hv n m + eta16 else 0

/-- A bound on the `float16` `P`'s magnitude. -/
noncomputable def p16B (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then (1 + u16) * pBr d hv n m + eta16 else 0

/-- A bound on the `float16` `P`'s lifted magnitude, the operand of line 303's `tl.dot`. -/
noncomputable def p16L (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := lift16 (p16B d hv n m)

/-- The error bound of the `float32` `dS` (line 308), before its rounding to `float16`. -/
noncomputable def dsErr (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  (pBr d hv n m - (2 : ℝ) ^ xE d hv n m) * yB d hv n m + (2 : ℝ) ^ xE d hv n m * eY d hv n m +
    u32 * (pBr d hv n m * yB d hv n m) + eta32

/-- A bound on the `float32` `dS`'s magnitude. -/
noncomputable def dsB (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  (1 + u32) * (pBr d hv n m * yB d hv n m) + eta32

/-- The error bound of the `float16` `dS` (line 309). `0` when not `Allowed`. -/
noncomputable def ds16Err (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then dsErr d hv n m + u16 * dsB d hv n m + eta16 else 0

/-- A bound on the `float16` `dS`'s magnitude. -/
noncomputable def ds16B (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then (1 + u16) * dsB d hv n m + eta16 else 0

/-- A bound on the `float16` `dS`'s lifted magnitude, the operand of lines 310 and 358's `tl.dot`. -/
noncomputable def ds16L (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := lift16 (ds16B d hv n m)

/-- The inputs of head `b` that `Post`'s accuracy covers: every value finite; `|L|` and `|Δ|` at
most `2^126`; and for every `Allowed` pair, the exponent at most `15` with its error
(`xE + eX ≤ 15`, so `P ≤ 2^15` fits `float16` and `float32`) and `|P·(dP − Δ)|`'s bound at most
`2^15` (it fits `float16`). On real inputs `P ≤ 1` (the forward's `M` is the row's log-sum-exp)
and `|dS|` is small. -/
structure HeadV.Ok (d : Inst) (hv : HeadV) : Prop where
  /-- Finite `Q`. -/
  q : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.q r c).isFinite = true
  /-- Finite `K`. -/
  k : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.k r c).isFinite = true
  /-- Finite `V`. -/
  v : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.v r c).isFinite = true
  /-- Finite `dO`. -/
  dO : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.dO r c).isFinite = true
  /-- Finite `L`, at most `2^126`. -/
  lse : ∀ r, r < d.N_CTX → (hv.lse r).isFinite = true ∧ |(hv.lse r).toRat| ≤ 2 ^ 126
  /-- Finite `Δ`, at most `2^126`. -/
  dl : ∀ r, r < d.N_CTX → (hv.dl r).isFinite = true ∧ |(hv.dl r).toRat| ≤ 2 ^ 126
  /-- Each `Allowed` pair fits. -/
  fits : ∀ n m, n < d.N_CTX → m < d.N_CTX → Allowed d n m →
    xE d hv n m + eX d hv n m ≤ 15 ∧ pBr d hv n m * yB d hv n m ≤ 2 ^ 15

/-! ## The outputs -/

/-- The number of accumulation steps for key row `n` (lines 415–443): `N_CTX/32`; with `CAUSAL`,
`8` masked steps and `(N_CTX − start_n − 128)/32`, where `start_n = 128·⌊n/128⌋`. -/
def stepsKV (d : Inst) (n : Nat) : Nat :=
  if d.CAUSAL then 8 + (d.N_CTX - 128 * (n / 128) - 128) / 32 else d.N_CTX / 32

/-- The number of accumulation steps for query row `m` (lines 468–496): `N_CTX/32`; with
`CAUSAL`, `8` masked steps and `start_m/32`, where `start_m = 128·⌊m/128⌋`. -/
def stepsQ (d : Inst) (m : Nat) : Nat :=
  if d.CAUSAL then 8 + 128 * (m / 128) / 32 else d.N_CTX / 32

/-- The exact `dV[n, j] = Σ_m P[n, m]·dO[m, j]`. -/
noncomputable def dVE (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, pE d hv n m * (hv.dO m j).toReal

/-- The tolerance of `dV[n, j]`'s `float32` value: the `float16` `P`'s errors weighted by
`|dO|`, and the accumulation's error over the bounds of those products' lifted magnitudes
(`p16L·lm dO`). -/
noncomputable def dVTol (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, p16Err d hv n m * |(hv.dO m j).toReal| +
    accB (stepsKV d n) (∑ m ∈ Finset.range d.N_CTX, p16L d hv n m * lm (hv.dO m j))

/-- `Σ_m dS[n, m]·Q[m, j]`. -/
noncomputable def dKS (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, dsE d hv n m * (hv.q m j).toReal

/-- The tolerance of the accumulated `Σ_m dS[n, m]·Q[m, j]`. -/
noncomputable def dKAcc (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, ds16Err d hv n m * |(hv.q m j).toReal| +
    accB (stepsKV d n) (∑ m ∈ Finset.range d.N_CTX, ds16L d hv n m * lm (hv.q m j))

/-- The exact `dK[n, j] = sm_scale · Σ_m dS[n, m]·Q[m, j]` (`sm_scale` as the kernel receives it). -/
noncomputable def dKE (d : Inst) (hv : HeadV) (n j : Nat) : ℝ := (smF d).toReal * dKS d hv n j

/-- The tolerance of `dK[n, j]`'s `float32` value: the accumulation's, scaled, and the rounding
of the product by `sm_scale` (line 449). -/
noncomputable def dKTol (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  |(smF d).toReal| * dKAcc d hv n j +
    u32 * (|(smF d).toReal| * (|dKS d hv n j| + dKAcc d hv n j)) + eta32

/-- `Σ_n dS[n, m]·K[n, j]`. -/
noncomputable def dQS (d : Inst) (hv : HeadV) (m j : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N_CTX, dsE d hv n m * (hv.k n j).toReal

/-- The tolerance of the accumulated `Σ_n dS[n, m]·K[n, j]`. -/
noncomputable def dQAcc (d : Inst) (hv : HeadV) (m j : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N_CTX, ds16Err d hv n m * |(hv.k n j).toReal| +
    accB (stepsQ d m) (∑ n ∈ Finset.range d.N_CTX, ds16L d hv n m * lm (hv.k n j))

/-- The exact `dQ[m, j] = LN2 · Σ_n dS[n, m]·K[n, j]`. -/
noncomputable def dQE (d : Inst) (hv : HeadV) (m j : Nat) : ℝ := ln2F.toReal * dQS d hv m j

/-- The tolerance of `dQ[m, j]`'s `float32` value: the accumulation's, scaled, and the rounding
of the product by `LN2` (line 499). -/
noncomputable def dQTol (d : Inst) (hv : HeadV) (m j : Nat) : ℝ :=
  ln2F.toReal * dQAcc d hv m j + u32 * (ln2F.toReal * (|dQS d hv m j| + dQAcc d hv m j)) + eta32

/-- What the launch leaves in element `(b, r, j)` of the output `B`: some `float16` value `c`;
and when head `b`'s inputs are `hv` and covered (`HeadV.Ok`), `c` is the `float16` rounding (to
nearest even, up to NaN payload) of a finite `float32` value within `T hv` of the exact `E hv`;
and when there is room in `float16` (`|E| + T ≤ 65504`), `c` is finite and within `T` plus one
`float16` rounding (`(|E| + T)·2^-11 + 2^-25`) of `E`. -/
def OutSpec (d : Inst) (init fin : Mem) (B : Buf .f16) (b r j : Nat) (E T : HeadV → ℝ) : Prop :=
  ∃ c : F16, fin.read .f16 (B.ptr ↑(eIdx d b r j)).addr = some c ∧
    ∀ hv : HeadV, hv.Holds d init b → hv.Ok d →
      ∃ v : F32, v.isFinite = true ∧ |v.toReal - E hv| ≤ T hv ∧ FloatV.NaNEq (FloatV.cvt v) c ∧
        (|E hv| + T hv ≤ 65504 →
          c.isFinite = true ∧ |c.toReal - E hv| ≤ T hv + (|E hv| + T hv) * u16 + eta16)

/-- What the launch leaves: every element of `dv`, `dk` and `dq` as `OutSpec` says, and every
byte outside the three outputs unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b r j, b < G d → r < d.N_CTX → j < d.HEAD_DIM →
    OutSpec d init fin d.DV b r j (fun hv => dVE d hv r j) (fun hv => dVTol d hv r j) ∧
    OutSpec d init fin d.DK b r j (fun hv => dKE d hv r j) (fun hv => dKTol d hv r j) ∧
    OutSpec d init fin d.DQ b r j (fun hv => dQE d hv r j) (fun hv => dQTol d hv r j)) ∧
  ∀ z, z ∉ outBytes d → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's launches
  meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotF16`: `tl.dot` of `float16` operands into `float32` (`maxImpreciseAcc = 0`,
  any `input_precision`; lines 293, 303, 307, 310, 345, 353, 358): each output element whose
  operands and accumulator are finite is finite and within `relDot16 K·L + absDot16 K` of
  `acc + Σ a·b`, where `L = |acc| + Σ liftMag a·liftMag b` counts a subnormal operand at `2^-14`
  (`Rules.Num.DotLiftAcc`), with room in `float32`. -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
  /-- [rule] `Num.Exp2F32`: `tl.math.exp2` on `float32` (lines 294, 346) is within
  `2^-22·2^x + 2^-126` of `2^x`, with room in `float32`. -/
  exp2 : Rules.Num.Exp2F32 P
  /-- [rule] `Num.MathTotal`: `tl.math.exp2` on `float32` always has a result. -/
  exp2_total : Rules.Num.MathTotal P .exp2 .binary32


/-- **The claim to prove.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotF16, Platform.ref.dotTotal _,
    Platform.sm90ref.exp2F32, Platform.ref.mathTotal⟩⟩

/-! ## The host's layout

`canon Z H N D c sm` is the host's launch (lines 608–621) for `BATCH = Z`, `N_HEAD = H`,
`N_CTX = N`, `HEAD_DIM = D`, `causal = c` and the Python float `sm_scale = sm`: the nine tensors
contiguous (`q`, `arg_k`, `v`, `do` and `dq`, `dk`, `dv` of `Z·H·N·D` elements, `M` and `delta` of
`Z·H·N`), at `2^40`, `2·2^40`, …, `9·2^40` (inputs first, then `M`, `delta`, then the outputs). -/

def canon (Z H N D : Nat) (c : Bool) (sm : F64) : Inst where
  Z := Z
  H := H
  N_CTX := N
  HEAD_DIM := D
  CAUSAL := c
  sm_scale := sm
  Q := ⟨2 ^ 40, Z * H * N * D⟩
  K := ⟨2 * 2 ^ 40, Z * H * N * D⟩
  V := ⟨3 * 2 ^ 40, Z * H * N * D⟩
  DO := ⟨4 * 2 ^ 40, Z * H * N * D⟩
  M := ⟨5 * 2 ^ 40, Z * H * N⟩
  D := ⟨6 * 2 ^ 40, Z * H * N⟩
  DQ := ⟨7 * 2 ^ 40, Z * H * N * D⟩
  DK := ⟨8 * 2 ^ 40, Z * H * N * D⟩
  DV := ⟨9 * 2 ^ 40, Z * H * N * D⟩

theorem elems_bytes {dt : DType} {B : Buf dt} {S : Set Nat} {x : Int} (hx : x ∈ B.elems S)
    (hS : ∀ j ∈ S, j < B.len) : B.base ≤ x ∧ x < B.base + B.len * dt.bytes :=
  B.elems_sub_bytes hS hx

/-- **The host's launches are covered**: `HEAD_DIM ∈ {16, …, 256}` (line 512), `N_CTX % 128 == 0`
(line 599) up to `2^14`, `BATCH·N_HEAD < 2^16` within the kernel's `int32` offsets (and below
`2^38` elements per tensor, for the layout's spacing), `BATCH ≤ 8192`, either
`causal`, and an `sm_scale` whose `float32` is finite and at most `2^16`. -/
theorem canon_ok (Z H N D : Nat) (c : Bool) (sm : F64)
    (hd : D = 16 ∨ D = 32 ∨ D = 64 ∨ D = 128 ∨ D = 256) (hn : N % 128 = 0) (hnle : N ≤ 2 ^ 14)
    (hg : Z * H < 2 ^ 16) (hsz : (Z * H - 1) * N * D < 2 ^ 31 ∧ H * N * D < 2 ^ 31)
    (hbig : Z * H * N * D < 2 ^ 38) (hz : Z ≤ 8192)
    (hsm : (FloatV.hostF32 sm).isFinite = true ∧ |(FloatV.hostF32 sm).toRat| ≤ 2 ^ 16) :
    (canon Z H N D c sm).ok := by
  have ht : ∀ j ∈ tSet (canon Z H N D c sm), j < Z * H * N * D := by
    intro j hj; simp only [tSet, G, canon, Set.mem_ofPred_eq] at hj; exact hj
  have hr : ∀ j ∈ rSet (canon Z H N D c sm), j < Z * H * N := by
    intro j hj; simp only [rSet, G, canon, Set.mem_ofPred_eq] at hj; exact hj
  have hrn : Z * H * N ≤ Z * H * N * D := Nat.le_mul_of_pos_right _ (by omega)
  have hout : ∀ x ∈ outBytes (canon Z H N D c sm), 7 * 2 ^ 40 ≤ x := by
    intro x hx
    simp only [outBytes, Set.mem_union] at hx
    rcases hx with (h | h) | h
    · have := elems_bytes h ht; simp only [canon] at this; omega
    · have := elems_bytes h ht; simp only [canon] at this; omega
    · have := elems_bytes h ht; simp only [canon] at this; omega
  have hin : ∀ x ∈ inBytes (canon Z H N D c sm), x < 7 * 2 ^ 40 := by
    intro x hx
    simp only [inBytes, Set.mem_union] at hx
    rcases hx with ((((h | h) | h) | h) | h) | h
    · have := elems_bytes h ht; simp only [canon, DType.bytes] at this; omega
    · have := elems_bytes h ht; simp only [canon, DType.bytes] at this; omega
    · have := elems_bytes h ht; simp only [canon, DType.bytes] at this; omega
    · have := elems_bytes h ht; simp only [canon, DType.bytes] at this; omega
    · have := elems_bytes h hr; simp only [canon, DType.bytes] at this; omega
    · have := elems_bytes h hr; simp only [canon, DType.bytes] at this; omega
  refine
    { hd := hd
      n_mod := hn
      n_le := hnle
      g_lt := by simp only [G, canon]; exact hg
      z_le := hz
      size_lt := by simp only [G, canon]; exact hsz
      sm := hsm
      q_len := by simp only [G, canon]; exact le_rfl
      k_len := by simp only [G, canon]; exact le_rfl
      v_len := by simp only [G, canon]; exact le_rfl
      do_len := by simp only [G, canon]; exact le_rfl
      dq_len := by simp only [G, canon]; exact le_rfl
      dk_len := by simp only [G, canon]; exact le_rfl
      dv_len := by simp only [G, canon]; exact le_rfl
      m_len := by simp only [G, canon]; exact le_rfl
      d_len := by simp only [G, canon]; exact le_rfl
      out_in := Set.disjoint_left.mpr fun x hx hy => absurd (hin x hy) (not_lt.mpr (hout x hx))
      dq_dk := ?_
      dq_dv := ?_
      dk_dv := ?_ }
  all_goals
    refine Set.disjoint_left.mpr fun x hx hy => ?_
    have h1 := elems_bytes hx ht
    have h2 := elems_bytes hy ht
    simp only [canon, DType.bytes] at h1 h2; omega

/-! ## Coverage: the launches of `SETUP.md`

The test (`test_op`, lines 631–695, `mode = "bwd"`, `provider = "triton-fp16"`): `Z ∈ {1, 4}`,
`H ∈ {2, 48}`, `N_CTX ∈ {128, 1024, 4096}`, `HEAD_DIM ∈ {64, 128}`, `causal` both,
`sm_scale = 0.5`. The benchmark (`bench_flash_attention`, lines 705–775, `mode = "bwd"`,
`provider = "triton-fp16"`): `BATCH = 4`, `N_HEADS = 32`, `N_CTX = 2^10 … 2^14`,
`HEAD_DIM ∈ {64, 128}`, `causal` both, `sm_scale = 1.3`. -/

/-- The double `0.5` (the test's `sm_scale`). -/
def smTest : F64 := FloatV.ofPat 0x3FE0000000000000
/-- The double `1.3` (the benchmark's `sm_scale`). -/
def smBench : F64 := FloatV.ofPat 0x3FF4CCCCCCCCCCCD

theorem smTest_ok : (FloatV.hostF32 smTest).isFinite = true ∧ |(FloatV.hostF32 smTest).toRat| ≤ 2 ^ 16 := by
  decide +kernel
theorem smBench_ok : (FloatV.hostF32 smBench).isFinite = true ∧ |(FloatV.hostF32 smBench).toRat| ≤ 2 ^ 16 := by
  decide +kernel

/-- One launch of the test. -/
def setup_test (Z H N D : Nat) (c : Bool) : Inst := canon Z H N D c smTest

theorem setup_test_ok : ∀ Z ∈ [1, 4], ∀ H ∈ [2, 48], ∀ N ∈ [128, 1024, 4096], ∀ D ∈ [64, 128],
    ∀ c : Bool, (setup_test Z H N D c).ok := by
  intro Z hZ H hH N hN D hD c
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hZ hH hN hD
  rcases hZ with rfl | rfl <;> rcases hH with rfl | rfl <;> rcases hN with rfl | rfl | rfl <;>
    rcases hD with rfl | rfl <;>
    exact canon_ok _ _ _ _ c smTest (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) smTest_ok

/-- One launch of the benchmark: `BATCH = 4`, `N_HEADS = 32`, `N_CTX = 2^e`. -/
def setup_bench (e D : Nat) (c : Bool) : Inst := canon 4 32 (2 ^ e) D c smBench

theorem setup_bench_ok : ∀ e ∈ [10, 11, 12, 13, 14], ∀ D ∈ [64, 128], ∀ c : Bool,
    (setup_bench e D c).ok := by
  intro e he D hD c
  simp only [List.mem_cons, List.not_mem_nil, or_false] at he hD
  rcases he with rfl | rfl | rfl | rfl | rfl <;> rcases hD with rfl | rfl <;>
    exact canon_ok _ _ _ _ c smBench (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) smBench_ok

/-- The test's largest launch, causal: `Z = 4`, `H = 48`, `N_CTX = 4096`, `HEAD_DIM = 128`. -/
def setup_test_large : Inst := setup_test 4 48 4096 128 true
theorem setup_test_large_ok : setup_test_large.ok :=
  setup_test_ok 4 (by simp) 48 (by simp) 4096 (by simp) 128 (by simp) true

/-- The benchmark's largest launch, causal: `N_CTX = 2^14`, `HEAD_DIM = 128`. -/
def setup_bench_large : Inst := setup_bench 14 128 true
theorem setup_bench_large_ok : setup_bench_large.ok :=
  setup_bench_ok 14 (by simp) 128 (by simp) true

/-- The other head dims the forward admits (line 512), at the benchmark's longest context. -/
theorem setup_headdims_ok : ∀ D ∈ [16, 32, 256], ∀ c : Bool, (canon 4 32 16384 D c smBench).ok := by
  intro D hD c
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hD
  rcases hD with rfl | rfl | rfl <;>
    exact canon_ok _ _ _ _ c smBench (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) smBench_ok

/-! ## Satisfiable

The test's `Z = 1`, `H = 2`, `N_CTX = 128`, `HEAD_DIM = 64`, not causal, in a memory of zeros. -/

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨setup_test 1 2 128 64 false, testMem,
    setup_test_ok 1 (by simp) 2 (by simp) 128 (by simp) 64 (by simp) false, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals
    refine ⟨by norm_num [setup_test, canon], by norm_num [setup_test, canon, DType.bytes],
      by norm_num [setup_test, canon, DType.bytes], fun _ _ => by simp [testMem]⟩

end Claims.AttnBwd
