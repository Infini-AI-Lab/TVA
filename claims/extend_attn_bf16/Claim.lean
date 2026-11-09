import TVA

/-!
# `extend_attn_bf16`: the prepared claim

Source: `source/extend_attention.py`. The kernel is `_fwd_kernel` (lines 99–638); it reads the
module-level names `triton` and `tl` (lines 89–90), and names the helper `tanh` (lines 93–96) only
in branches the covered launches do not visit. The host `extend_attention_fwd` (lines 687–857)
launches it on `bfloat16` `q_extend` (`[total_extend_tokens, num_heads, head_dim]`), `k_extend`,
`v_extend` (`[total_extend_tokens, num_kv_heads, head_dim]`), the KV cache `k_buffer`, `v_buffer`
(`[num_slots, num_kv_heads, head_dim]`), `qo_indptr` (`int64` for an extend batch, `int32` for a
target verification), the `int32` `kv_indptr` (both `[batch + 1]`) and the `int64` `kv_indices`,
with a target verification's tree mask (`custom_mask`, `bool`, and `mask_indptr`, `int64`) or
none, with a layer's sliding window (`SLIDING_WINDOW_SIZE > 0`, and `window_kv_offsets`, `int64`)
or none, and writes `o_extend = torch.empty_like(q_extend)`, on the grid
`(batch, num_heads, cdiv(max_len_extend, BLOCK_M))`, with the H200's block sizes (`BLOCK_M = 128`
for head dims up to 128, `64` up to 256; `BLOCK_N = BLOCK_N_PREFIX = 64`),
`BLOCK_DMODEL = BLOCK_DV = next_power_of_2(head_dim)` and the other flags of SGLang's Triton
backend (`SETUP.md`).

* The model: `fwd_kernel`, `model`, `launch`.
* The conditions: `Inst`, `Inst.ok`, `Pre`.
* The specification: `Post`, `Spec`, with the reference `attnRef` over the positions a row
  attends (`attends`, `keys`), its scale `attnAbs`, the row's score bound `Xrow`, the row's loop trips `tripsPre`, `tripsExt` and the tolerances `tolRel`,
  `tolAbs`.
* The assumptions: `Assumptions`.
* The claim: `Statement`.
* Checked here: `consistent`, `satisfiable`, and the coverage facts `setup_*_ok`.
-/

namespace Claims.ExtendAttn

open TVA TVA.Math.Float

/-! ## Constants of the source -/

/-- The `float32` `+0` (`other=0.0`; `deno == 0.0`, line 607). -/
def zf32 : F32 := FloatV.litF32 (FloatV.ofPat 0)

/-- The `float32` `+∞` (`float("inf")` of line 265, in an arithmetic operation on `float32`). -/
def pinf32 : F32 := FloatV.litF32 (FloatV.ofPat 0x7FF0000000000000)

/-- The `float32` `−∞` (`float("-inf")` of lines 395, 398, 561, 564: a `tl.where` value and a
comparison operand). -/
def ninf32 : F32 := FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000)

/-- The `float32` of `-1e20` (the double `0xc415af1d78b58c40`; a `tl.where` value, lines 398,
564). -/
def neg1e20 : F32 := FloatV.litF32 (FloatV.ofPat 0xC415AF1D78B58C40)

/-- The `float32` `1.0` (a `tl.where` value, line 626). -/
def one32 : F32 := FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)

/-- The double `1.0`, the host's `k_scale` and `v_scale` (Python `float`s, lines 723–724). -/
def one64 : F64 := FloatV.ofPat 0x3FF0000000000000

/-- `BLOCK_N = BLOCK_N_PREFIX = 64`, the H200's choice for head dims up to 256 (lines 663–669;
`FP8_BLOCK_N` is not used: `USE_FP8_PREFIX = USE_FP8_EXTEND = False`, lines 758–764). -/
def BN : Nat := 64

/-- The `tl.dot` configuration of lines 344, 437, 520 and 593: `bfloat16` operands, the
`float32` accumulator (the default, and `out_dtype=tl.float32` at line 520), the default
`input_precision` (`tf32`; no effect on 16-bit operands) and the default
`max_num_imprecise_acc` (`0`, not an fp8 dot). -/
def cfgBF16 : DotCfg :=
  { a := .bf16, b := .bf16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }

/-! ## The types of `qo_indptr`

An extend batch passes an `int64` `qo_indptr`, a target verification an `int32` one
(`SETUP.md`). Triton compiles the kernel for the pointer's element type, and the values computed
from it (`cur_seq_extend_start_idx`, `cur_seq_len_extend`, and every offset and bound built from
them) have that type. `qo32` selects it; `qoD`, `QI`, `loadQo`, `toQ`, `qToI64` give the type and
the operations of each case. -/

/-- The element type of `qo_indptr`: `int32` (`qo32 = true`) or `int64`. -/
def qoD : Bool → DType
  | true => .i32
  | false => .i64

/-- Its width in bits. -/
def qoW : Bool → Nat
  | true => 32
  | false => 64

/-- The integers of `qo_indptr`'s element type (`I32` or `I64`). -/
abbrev QI (qo32 : Bool) : Type := IntV true (qoW qo32)

/-- A scalar load of `qo_indptr` (`tl.load1`), its element an integer of `QI`. -/
def loadQo : (qo32 : Bool) → Ptr (qoD qo32) → Prog (QI qo32)
  | true, p => tl.load1 p
  | false, p => tl.load1 p

/-- An `int32` operand of an operator whose other operand has `qo_indptr`'s type, as the front end
promotes it (`BinOp.typing?`: two signed integers compute in the wider): unchanged when that type
is `int32`, `tl.convert .i32 .i64` when it is `int64`. -/
def toQ {s : Shape} : (qo32 : Bool) → Tensor s I32 → Prog (Tensor s (QI qo32))
  | true, x => pure x
  | false, x => tl.convert .i32 .i64 x

/-- An operand of `qo_indptr`'s type beside an `int64` one: `tl.convert .i32 .i64` when it is
`int32`, unchanged when it is `int64`. -/
def qToI64 {s : Shape} : (qo32 : Bool) → Tensor s (QI qo32) → Prog (Tensor s I64)
  | true, x => tl.convert .i32 .i64 x
  | false, x => pure x

/-- The start of the integer `tl.max` folds (`Tensor.foldRows`, `Tensor.foldAll` with
`IntV.max`): the `int32` minimum, below every element, so each fold is the maximum of its
(nonempty) axis. -/
def imin32 : I32 := IntV.ofInt (-2 ^ 31)

/-! ## The model -/

/-- `_fwd_kernel` (source lines 99–638) on the covered paths. `USE_CUSTOM_MASK` and
`SLIDING_WINDOW_SIZE` are the compile-time flags (`SLIDING_WINDOW_SIZE` the window when it is
positive; `0` stands for the host's `-1`: the kernel uses the value only under
`SLIDING_WINDOW_SIZE > 0`); `qo32` the element type of `qo_indptr`. `Q_Extend`, `K_Extend`,
`V_Extend`, `O_Extend`, `K_Buffer`, `V_Buffer` are `*bf16`, `qo_indptr` `*i32` or `*i64`,
`kv_indptr` `*i32`, `kv_indices` `*i64`, `mask_ptr` `*i1` (a `bool` tensor), `mask_indptr` and
`window_kv_offset_ptr` `*i64` (read only under `USE_CUSTOM_MASK`; without it the host passes
`None`, which the model never reads); `sm_scale`, `k_scale`, `v_scale` are runtime `float32`
scalars (Python `float` arguments: `FloatV.hostF32` of the double); `kv_group_num` and the twelve
strides the kernel reads are `int32` arguments (one equal to `1` is the compile-time constant `1`,
which each use, an operand of `*` or `//` with an `int32` tensor, builds in `int32`, or, beside an
`int64`, in `int64`: the same values); `Lq`, `Lv`, `BLOCK_DMODEL`, `BLOCK_DV`, `BLOCK_M`, `BLOCK_N`,
`BLOCK_N_PREFIX` are compile-time values. The other parameters are compile-time values on every
covered launch (the other flags, `None` pointers) or arguments the covered path never reads
(`stride_lse_*`, `stride_buf_*page`, `stride_buf_*tok`, `compact_batch_size`, `aux0_*`), so the
model has no parameter for them. -/
def fwd_kernel (USE_CUSTOM_MASK qo32 : Bool) (SLIDING_WINDOW_SIZE : Nat)
    (Q_Extend K_Extend V_Extend O_Extend K_Buffer V_Buffer : Ptr .bf16)
    (qo_indptr : Ptr (qoD qo32)) (kv_indptr : Ptr .i32) (kv_indices : Ptr .i64)
    (mask_ptr : Ptr .i1) (mask_indptr window_kv_offset_ptr : Ptr .i64)
    (sm_scale k_scale v_scale : F32) (kv_group_num : I32)
    (stride_qbs stride_qh stride_kbs stride_kh stride_vbs stride_vh stride_obs stride_oh
      stride_buf_kbs stride_buf_kh stride_buf_vbs stride_buf_vh : I32)
    (Lq Lv BLOCK_DMODEL BLOCK_DV BLOCK_M BLOCK_N BLOCK_N_PREFIX : Nat) : Kernel where
  body ctx := do
    -- lines 172, 195–198: `USE_COMPACT_TILE_GRID = False`, a compile-time condition; the
    -- `else` branch: the three grid axes, `int32`
    let cur_seq := tl.program_id ctx 0
    let cur_head := tl.program_id ctx 1
    let cur_block_m := tl.program_id ctx 2
    -- line 199: the `int32` `//` (truncating; undefined for a zero divisor: `kv_pos`, `kv_dvd`)
    let cur_kv_head := (← tl.idiv (Tensor.scalar cur_head) (Tensor.scalar kv_group_num)).item
    -- line 203: a scalar load of `qo_indptr` advanced by the `int32` `cur_seq`
    let cur_seq_extend_start_idx : QI qo32 ← loadQo qo32 (qo_indptr + cur_seq)
    -- line 204: the pointer advanced by `cur_seq`, then by the Python `1` (an `int32`); the
    -- wrapping `-` of `qo_indptr`'s type
    let cur_seq_len_extend : QI qo32 :=
      (← loadQo qo32 (qo_indptr + cur_seq + (IntV.ofInt 1 : I32))) - cur_seq_extend_start_idx
    -- line 205: a scalar load of the `*i32` pointer
    let cur_seq_kv_start_idx : I32 ← tl.load1 (kv_indptr + cur_seq)
    -- line 206: the wrapping `int32` `-`
    let cur_seq_len_prefix : I32 :=
      (← tl.load1 (kv_indptr + cur_seq + (IntV.ofInt 1 : I32))) - cur_seq_kv_start_idx
    -- line 207: the `int32` left operand promoted to `qo_indptr`'s type, the wrapping `+`
    let cur_seq_len : QI qo32 :=
      (← toQ qo32 (Tensor.scalar cur_seq_len_prefix)).item + cur_seq_len_extend
    -- lines 210–211: a runtime `if` whose taken branch returns: `cur_block_m * BLOCK_M` (the
    -- constexpr built in `int32`, a wrapping `int32` product) promoted, the signed `>=`
    if IntV.le cur_seq_len_extend
        (← toQ qo32 (Tensor.scalar (cur_block_m * (IntV.ofNat BLOCK_M : I32)))).item then
      return ()
    else do
      -- lines 213–219: under `USE_CUSTOM_MASK` (compile-time), the scalar load of the `*i64`
      -- `mask_indptr`; `window_kv_offset = 0` (line 217), the Python `0` assigned to a name, an
      -- `int32` scalar tensor, which under `USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0`
      -- (compile-time) the scalar load of the `*i64` `window_kv_offset_ptr` replaces. `cm` is
      -- `some (cur_seq_mask_start_idx, window_kv_offset)` under `USE_CUSTOM_MASK` (`none`
      -- without it: the names are read only under it), with `window_kv_offset` `some` the loaded
      -- `int64`, or `none` for the `int32` `0`
      let cm : Option (I64 × Option I64) ←
        if USE_CUSTOM_MASK then do
          let cur_seq_mask_start_idx : I64 ← tl.load1 (mask_indptr + cur_seq)
          let window_kv_offset : Option I64 ←
            if 0 < SLIDING_WINDOW_SIZE then do
              let w : I64 ← tl.load1 (window_kv_offset_ptr + cur_seq)
              pure (some w)
            else pure none
          pure (some (cur_seq_mask_start_idx, window_kv_offset))
        else pure none
      -- lines 221–223
      let offs_d : Tensor [BLOCK_DMODEL] I32 := tl.arange 0 BLOCK_DMODEL
      let offs_dv : Tensor [BLOCK_DV] I32 := tl.arange 0 BLOCK_DV
      let offs_m : Tensor [BLOCK_M] I32 := tl.arange 0 BLOCK_M
      -- line 224: the `int32` sum promoted, the signed `<` with the scalar of `qo_indptr`'s type
      let mask_m : Tensor [BLOCK_M] Bool :=
        (← toQ qo32 (cur_block_m * (IntV.ofNat BLOCK_M : I32) + offs_m)).ltS cur_seq_len_extend
      -- lines 226–227: the constexprs built in `int32`, signed comparisons
      let mask_d : Tensor [BLOCK_DMODEL] Bool := offs_d.ltS (IntV.ofNat Lq : I32)
      let mask_dv : Tensor [BLOCK_DV] Bool := offs_dv.ltS (IntV.ofNat Lv : I32)
      -- lines 238–243: left to right: the start plus the promoted `int32` product, plus
      -- `offs_m[:, None]` promoted, times `stride_qbs` promoted, plus the `int32` product
      -- `cur_head * stride_qh` promoted, plus `offs_d[None, :]` promoted; wrapping operations of
      -- `qo_indptr`'s type, each term broadcast to the final shape (`x[:, None]` is `.col`,
      -- `x[None, :]` is `.row`)
      let offs_q : Tensor [BLOCK_M, BLOCK_DMODEL] (QI qo32) :=
        ((cur_seq_extend_start_idx +
              (← toQ qo32 (Tensor.scalar (cur_block_m * (IntV.ofNat BLOCK_M : I32)))).item) +
            (← toQ qo32 (offs_m.col BLOCK_DMODEL))) *
            (← toQ qo32 (Tensor.scalar stride_qbs)).item +
          (← toQ qo32 (Tensor.scalar (cur_head * stride_qh))).item +
          (← toQ qo32 (offs_d.row BLOCK_M))
      -- lines 244–246: `other=0.0` is the `float32` `+0`, converted to the pointer's dtype
      let q : Tensor [BLOCK_M, BLOCK_DMODEL] BF16 ←
        tl.load (Q_Extend + offs_q) (some ((mask_m.col BLOCK_DMODEL).and (mask_d.row BLOCK_M)))
          (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_M, BLOCK_DMODEL] zf32)))
      -- lines 259, 261
      let offs_n : Tensor [BLOCK_N] I32 := tl.arange 0 BLOCK_N
      let offs_n_prefix : Tensor [BLOCK_N_PREFIX] I32 := tl.arange 0 BLOCK_N_PREFIX
      -- lines 263–264
      let acc := tl.zeros [BLOCK_M, BLOCK_DV] .f32
      let deno := tl.zeros [BLOCK_M] .f32
      -- line 265: `tl.zeros(...) - float("inf")`: the Python number built in `float32`, IEEE `-`
      let e_max : Tensor [BLOCK_M] F32 ← tl.sub (tl.zeros [BLOCK_M] .f32) (bc pinf32)
      -- lines 267–268: `SKIP_PREFIX = False` (compile-time), so `prefix_end` is
      -- `cur_seq_len_prefix`; `range(0, prefix_end, BLOCK_N_PREFIX)`, an `int32` induction
      -- variable; the loop carries `acc`, `deno`, `e_max`
      let (acc, deno, e_max) ← tl.forRange (IntV.ofInt 0 : I32) cur_seq_len_prefix
          (IntV.ofNat BLOCK_N_PREFIX : I32) (acc, deno, e_max)
        fun start_n (acc, deno, e_max) => do
          -- line 269: a hint on the loop's induction variable: no effect (not modelled)
          -- line 270: `int32`, signed
          let mask_n : Tensor [BLOCK_N_PREFIX] Bool :=
            (start_n + offs_n_prefix).ltS cur_seq_len_prefix
          -- line 272
          let final_mask : Tensor [BLOCK_M, BLOCK_N_PREFIX] Bool :=
            (mask_m.col BLOCK_N_PREFIX).and (mask_n.row BLOCK_M)
          -- lines 273–284: `SKIP_PREFIX_CUSTOM_MASK = True` (compile-time) ends the `and`: not
          -- visited. Lines 285–289: under `SLIDING_WINDOW_SIZE > 0` (compile-time), the `int32`
          -- window mask (`cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]`, left to
          -- right, against `start_n + offs_n_prefix[None, :] + SLIDING_WINDOW_SIZE`, the
          -- constexpr built in `int32`; the signed `<=`), `&=` on `int1` tensors
          let final_mask : Tensor [BLOCK_M, BLOCK_N_PREFIX] Bool :=
            if 0 < SLIDING_WINDOW_SIZE then
              final_mask.and (Tensor.le
                (cur_seq_len_prefix + cur_block_m * (IntV.ofNat BLOCK_M : I32) +
                  offs_m.col BLOCK_N_PREFIX)
                (start_n + offs_n_prefix.row BLOCK_M + (IntV.ofNat SLIDING_WINDOW_SIZE : I32)))
            else final_mask
          -- lines 296–298: `SKIP_TILE = False`, a name assigned a Python `bool`: an `int1` scalar
          -- tensor; under `SLIDING_WINDOW_SIZE > 0` (the `or`'s left side is the compile-time
          -- `False`) it becomes `tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0`:
          -- the `int1` block converted to `int32`, the exact `int32` row maxima, their maximum,
          -- the `==` with the Python `0` (an `int32`)
          let SKIP_TILE : Tensor [] Bool ←
            if 0 < SLIDING_WINDOW_SIZE then do
              let fm : Tensor [BLOCK_M, BLOCK_N_PREFIX] I32 ← tl.convert .i1 .i32 final_mask
              pure (Tensor.scalar (IntV.eq
                (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 fm))
                (IntV.ofInt 0)))
            else pure (Tensor.scalar false)
          -- line 300: a runtime `if` on `not SKIP_TILE` (`Tensor.not` of the `int1` tensor)
          if (Tensor.not SKIP_TILE).item then do
            -- lines 304–308: `IDENTITY_KV_INDICES = False`; the `*i64` pointer advanced by the
            -- three `int32` offsets, left to right; `other=0` (an `int32`) converted to `int64`
            let offs_kv_loc : Tensor [BLOCK_N_PREFIX] I64 ←
              tl.load (kv_indices + cur_seq_kv_start_idx + start_n + offs_n_prefix) (some mask_n)
                (some (← tl.convert .i32 .i64 (tl.full [BLOCK_N_PREFIX] (IntV.ofInt 0 : I32))))
            -- lines 313–319: `PAGE_SIZE == 1`; `offs_kv_loc[None, :] * stride_buf_kbs` in `int64`,
            -- plus the `int32` product `cur_kv_head * stride_buf_kh` widened, plus `offs_d[:, None]`
            -- widened
            let offs_buf_k : Tensor [BLOCK_DMODEL, BLOCK_N_PREFIX] I64 :=
              offs_kv_loc.row BLOCK_DMODEL *
                  (← tl.convert .i32 .i64 (Tensor.scalar stride_buf_kbs)).item +
                (← tl.convert .i32 .i64 (Tensor.scalar (cur_kv_head * stride_buf_kh))).item +
                (← tl.convert .i32 .i64 (offs_d.col BLOCK_N_PREFIX))
            -- lines 329–333
            let k : Tensor [BLOCK_DMODEL, BLOCK_N_PREFIX] BF16 ←
              tl.load (K_Buffer + offs_buf_k)
                (some ((mask_n.row BLOCK_DMODEL).and (mask_d.col BLOCK_N_PREFIX)))
                (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_DMODEL, BLOCK_N_PREFIX] zf32)))
            -- lines 341–344: `IS_GFX1250 = False`; `q.to(k.dtype)`, `bfloat16` to `bfloat16`
            -- (`tl.cvt`: exact); `tl.dot` with the default zero accumulator, its result used by
            -- the product of line 371 (no addition: not the combine pass's shape)
            let q16 : Tensor [BLOCK_M, BLOCK_DMODEL] BF16 ← tl.cvt q
            let qk : Tensor [BLOCK_M, BLOCK_N_PREFIX] F32 ←
              tl.dot cfgBF16 q16 k (tl.zeros [BLOCK_M, BLOCK_N_PREFIX] .f32)
            -- lines 368–371: `USE_EXP2 = False`; `qk *= sm_scale * k_scale`: the IEEE product of
            -- the two `float32` scalars, then the IEEE product (no addition follows)
            let qk ← tl.mul qk (bc (← tl.mul (Tensor.scalar sm_scale) (Tensor.scalar k_scale)).item)
            -- line 395: `float("-inf")` built in `float32`
            let qk := tl.where final_mask qk (bc ninf32)
            -- line 397: the row maximum
            let row_max : Tensor [BLOCK_M] F32 ← tl.max1 true qk
            -- line 398: the ordered `==` with `float32` `−∞`; `-1e20` built in `float32`
            let row_max_fixed := tl.where (row_max.feq (bc ninf32)) (bc neg1e20) row_max
            -- line 399
            let n_e_max : Tensor [BLOCK_M] F32 ← tl.maximum row_max_fixed e_max
            -- lines 401–405: `USE_EXP2 = False`; the IEEE `-` (no product feeds it), `tl.exp`
            let re_scale : Tensor [BLOCK_M] F32 ← tl.exp (← tl.sub e_max n_e_max)
            -- line 406: the IEEE `-` (`qk` is a `tl.where` value), `tl.exp`
            let p : Tensor [BLOCK_M, BLOCK_N_PREFIX] F32 ←
              tl.exp (← tl.sub qk (n_e_max.col BLOCK_N_PREFIX))
            -- line 407: `deno * re_scale + tl.sum(p, 1)`: the IEEE row sum (`p` is not a
            -- product), and the addition the product feeds (`tl.mul_add`)
            let deno ← tl.mul_add deno re_scale (← tl.sum1 p)
            -- lines 409–414: `PAGE_SIZE == 1`; as lines 315–319
            let offs_buf_v : Tensor [BLOCK_N_PREFIX, BLOCK_DV] I64 :=
              offs_kv_loc.col BLOCK_DV *
                  (← tl.convert .i32 .i64 (Tensor.scalar stride_buf_vbs)).item +
                (← tl.convert .i32 .i64 (Tensor.scalar (cur_kv_head * stride_buf_vh))).item +
                (← tl.convert .i32 .i64 (offs_dv.row BLOCK_N_PREFIX))
            -- lines 422–426
            let v : Tensor [BLOCK_N_PREFIX, BLOCK_DV] BF16 ←
              tl.load (V_Buffer + offs_buf_v)
                (some ((mask_n.col BLOCK_DV).and (mask_dv.row BLOCK_N_PREFIX)))
                (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_N_PREFIX, BLOCK_DV] zf32)))
            -- lines 427, 430, 434–437: `USE_FP8_PREFIX = False`, `IS_GFX1250 = False`;
            -- `p.to(v.dtype)`, one rounding to nearest even; `tl.dot` with the default zero
            -- accumulator, its result used by the product of line 438
            let p16 : Tensor [BLOCK_M, BLOCK_N_PREFIX] BF16 ← tl.cvt p
            let dot : Tensor [BLOCK_M, BLOCK_DV] F32 ←
              tl.dot cfgBF16 p16 v (tl.zeros [BLOCK_M, BLOCK_DV] .f32)
            -- line 438: `acc * re_scale[:, None] + dot * v_scale`: two IEEE products feeding the
            -- addition (`tl.addc`)
            let acc ← tl.addc (← tl.mul acc (re_scale.col BLOCK_DV)) (← tl.mul dot (bc v_scale))
            -- line 440
            let e_max := n_e_max
            pure (acc, deno, e_max)
          else
            pure (acc, deno, e_max)
      -- lines 444–448: `IS_CAUSAL = True` (compile-time): only
      -- `tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)` is evaluated: the `int32`
      -- operations, promoted, the exact minimum
      let cur_block_m_end : QI qo32 :=
        (tl.iminimum (Tensor.scalar cur_seq_len_extend)
          (← toQ qo32 (Tensor.scalar ((cur_block_m + (IntV.ofInt 1 : I32)) *
            (IntV.ofNat BLOCK_M : I32))))).item
      -- line 449: `SKIP_EXTEND = False`
      let extend_end := cur_block_m_end
      -- line 454: the Python `0` assigned to a name: an `int32` scalar tensor; lines 455–458:
      -- under `SLIDING_WINDOW_SIZE > 0` (compile-time) it becomes the `int32`
      -- `(tl.maximum(cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0) // BLOCK_N) * BLOCK_N`
      -- (the constexprs and the Python `0` built in `int32`; `tl.imaximum`, the truncating `//`)
      let extend_start : I32 ←
        if 0 < SLIDING_WINDOW_SIZE then do
          let lo : Tensor [] I32 :=
            tl.imaximum (Tensor.scalar (cur_block_m * (IntV.ofNat BLOCK_M : I32) -
              (IntV.ofNat SLIDING_WINDOW_SIZE : I32))) (Tensor.scalar (IntV.ofInt 0 : I32))
          let lo ← tl.idiv lo (Tensor.scalar (IntV.ofNat BLOCK_N : I32))
          pure (lo.item * (IntV.ofNat BLOCK_N : I32))
        else pure (IntV.ofInt 0)
      -- line 459: `range(extend_start, extend_end, BLOCK_N)`: the bounds promote to
      -- `qo_indptr`'s type (`extend_end` has it), so `extend_start` is promoted and the
      -- induction variable has that type; the loop carries `acc`, `deno`, `e_max`
      let (acc, deno, _) ← tl.forRange
          (← toQ qo32 (Tensor.scalar extend_start)).item extend_end
          (IntV.ofNat BLOCK_N : QI qo32) (acc, deno, e_max)
        fun start_n (acc, deno, e_max) => do
          -- line 460: a hint on the loop's induction variable: no effect (not modelled)
          -- line 461: `offs_n` promoted, the sum, the signed `<`
          let mask_n : Tensor [BLOCK_N] Bool :=
            (start_n + (← toQ qo32 offs_n)).ltS cur_block_m_end
          -- line 463
          let final_mask : Tensor [BLOCK_M, BLOCK_N] Bool :=
            (mask_m.col BLOCK_N).and (mask_n.row BLOCK_M)
          -- lines 464–488: `if USE_CUSTOM_MASK:` (`cm` is `some`) or `elif IS_CAUSAL:` (`True`)
          let final_mask : Tensor [BLOCK_M, BLOCK_N] Bool ← match cm with
            | some (cur_seq_mask_start_idx, window_kv_offset) => do
              -- lines 465–475: the `*i1` pointer advanced, left to right, by the `int64`
              -- `cur_seq_mask_start_idx`, by the product of the row index
              -- `cur_block_m * BLOCK_M + offs_m[:, None]` (`int32`) and the row length
              -- `cur_seq_len + window_kv_offset`, then by `window_kv_offset`,
              -- `cur_seq_len_prefix` (`int32`), `start_n` and `offs_n[None, :]` (`int32`); with the
              -- `int32` `0` the row length and the product have `qo_indptr`'s type, with the
              -- loaded `int64` both are `int64`
              let ptrs : Tensor [BLOCK_M, BLOCK_N] (Ptr .i1) ← match window_kv_offset with
                | none => do
                  let wko : I32 := IntV.ofInt 0
                  let row_len : QI qo32 := cur_seq_len + (← toQ qo32 (Tensor.scalar wko)).item
                  let row_off : Tensor [BLOCK_M, BLOCK_N] (QI qo32) :=
                    (← toQ qo32 (cur_block_m * (IntV.ofNat BLOCK_M : I32) + offs_m.col BLOCK_N)) *
                      row_len
                  pure (mask_ptr + cur_seq_mask_start_idx + row_off +
                    (bc wko : Tensor [BLOCK_M, BLOCK_N] _) +
                    (bc cur_seq_len_prefix : Tensor [BLOCK_M, BLOCK_N] _) +
                    (bc start_n : Tensor [BLOCK_M, BLOCK_N] _) + offs_n.row BLOCK_M)
                | some wko => do
                  let row_len : I64 := (← qToI64 qo32 (Tensor.scalar cur_seq_len)).item + wko
                  let row_off : Tensor [BLOCK_M, BLOCK_N] I64 :=
                    (← tl.convert .i32 .i64
                      (cur_block_m * (IntV.ofNat BLOCK_M : I32) + offs_m.col BLOCK_N)) * row_len
                  pure (mask_ptr + cur_seq_mask_start_idx + row_off +
                    (bc wko : Tensor [BLOCK_M, BLOCK_N] _) +
                    (bc cur_seq_len_prefix : Tensor [BLOCK_M, BLOCK_N] _) +
                    (bc start_n : Tensor [BLOCK_M, BLOCK_N] _) + offs_n.row BLOCK_M)
              -- lines 465, 476: the masked load; `other=0` (an `int32`) converted to the
              -- pointer's `int1` (`0 != 0`, false)
              let custom_mask : Tensor [BLOCK_M, BLOCK_N] Bool ←
                tl.load ptrs (some ((mask_m.col BLOCK_N).and (mask_n.row BLOCK_M)))
                  (some (← tl.convert .i32 .i1 (tl.full [BLOCK_M, BLOCK_N] (IntV.ofInt 0 : I32))))
              -- line 478: `custom_mask != 0`: the `int1` block and the Python `0` compute in
              -- `int32` (`int1` counts as unsigned of width 1: the signed `int32` is wider), the
              -- `!=`; then `&` with `mask_m[:, None]` and `mask_n[None, :]`, left to right
              let cm32 : Tensor [BLOCK_M, BLOCK_N] I32 ← tl.convert .i1 .i32 custom_mask
              let custom_mask : Tensor [BLOCK_M, BLOCK_N] Bool :=
                ((cm32.map (fun x => !(IntV.eq x (IntV.ofInt 0)))).and (mask_m.col BLOCK_N)).and
                  (mask_n.row BLOCK_M)
              -- line 479
              pure (final_mask.and custom_mask)
            | none => do
              -- lines 480–483: the `int32` right side promoted, the signed `>=`
              let mask_causual : Tensor [BLOCK_M, BLOCK_N] Bool :=
                Tensor.le (start_n + (← toQ qo32 (offs_n.row BLOCK_M)))
                  (← toQ qo32 (cur_block_m * (IntV.ofNat BLOCK_M : I32) + offs_m.col BLOCK_N))
              -- line 484
              let mask_causual := mask_causual.and ((mask_m.col BLOCK_N).and (mask_n.row BLOCK_M))
              -- line 485
              pure (final_mask.and mask_causual)
          -- lines 490–503: under `SLIDING_WINDOW_SIZE > 0` (compile-time), `IS_CAUSAL = True`:
          -- the `else` branch, `cur_block_m * BLOCK_M + offs_m[:, None]` (`int32`, promoted)
          -- against `start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE` (the constexpr built in
          -- `qo_indptr`'s type), the signed `<=`; `&=`
          let final_mask : Tensor [BLOCK_M, BLOCK_N] Bool ←
            if 0 < SLIDING_WINDOW_SIZE then do
              let window_mask : Tensor [BLOCK_M, BLOCK_N] Bool :=
                Tensor.le
                  (← toQ qo32 (cur_block_m * (IntV.ofNat BLOCK_M : I32) + offs_m.col BLOCK_N))
                  (start_n + (← toQ qo32 (offs_n.row BLOCK_M)) +
                    (IntV.ofNat SLIDING_WINDOW_SIZE : QI qo32))
              pure (final_mask.and window_mask)
            else pure final_mask
          -- lines 505–507: as lines 296–298, under `USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0`
          let SKIP_TILE : Tensor [] Bool ←
            if USE_CUSTOM_MASK || decide (0 < SLIDING_WINDOW_SIZE) then do
              let fm : Tensor [BLOCK_M, BLOCK_N] I32 ← tl.convert .i1 .i32 final_mask
              pure (Tensor.scalar (IntV.eq
                (Tensor.foldAll IntV.max imin32 (Tensor.foldRows IntV.max imin32 fm))
                (IntV.ofInt 0)))
            else pure (Tensor.scalar false)
          -- line 509: a runtime `if` on `not SKIP_TILE`
          if (Tensor.not SKIP_TILE).item then do
            -- lines 511–515: the start plus the induction variable, plus `offs_n[None, :]`
            -- promoted, times `stride_kbs` promoted, plus the `int32` product promoted, plus
            -- `offs_d[:, None]` promoted
            let offs_k : Tensor [BLOCK_DMODEL, BLOCK_N] (QI qo32) :=
              ((cur_seq_extend_start_idx + start_n) +
                    (← toQ qo32 (offs_n.row BLOCK_DMODEL))) *
                  (← toQ qo32 (Tensor.scalar stride_kbs)).item +
                (← toQ qo32 (Tensor.scalar (cur_kv_head * stride_kh))).item +
                (← toQ qo32 (offs_d.col BLOCK_N))
            -- lines 516–518
            let k : Tensor [BLOCK_DMODEL, BLOCK_N] BF16 ←
              tl.load (K_Extend + offs_k) (some ((mask_n.row BLOCK_DMODEL).and (mask_d.col BLOCK_N)))
                (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_DMODEL, BLOCK_N] zf32)))
            -- line 520: `tl.dot` with the default zero accumulator, its result used by the
            -- product of line 537
            let qk : Tensor [BLOCK_M, BLOCK_N] F32 ←
              tl.dot cfgBF16 q k (tl.zeros [BLOCK_M, BLOCK_N] .f32)
            -- lines 534–537: `USE_EXP2 = False`; the IEEE product
            let qk ← tl.mul qk (bc sm_scale)
            -- line 561
            let qk := tl.where final_mask qk (bc ninf32)
            -- line 563
            let row_max : Tensor [BLOCK_M] F32 ← tl.max1 true qk
            -- line 564
            let row_max_fixed := tl.where (row_max.feq (bc ninf32)) (bc neg1e20) row_max
            -- line 565
            let n_e_max : Tensor [BLOCK_M] F32 ← tl.maximum row_max_fixed e_max
            -- lines 567–571
            let re_scale : Tensor [BLOCK_M] F32 ← tl.exp (← tl.sub e_max n_e_max)
            -- line 572
            let p : Tensor [BLOCK_M, BLOCK_N] F32 ← tl.exp (← tl.sub qk (n_e_max.col BLOCK_N))
            -- line 573: as line 407
            let deno ← tl.mul_add deno re_scale (← tl.sum1 p)
            -- lines 575–579: as lines 511–515
            let offs_v : Tensor [BLOCK_N, BLOCK_DV] (QI qo32) :=
              ((cur_seq_extend_start_idx + start_n) +
                    (← toQ qo32 (offs_n.col BLOCK_DV))) *
                  (← toQ qo32 (Tensor.scalar stride_vbs)).item +
                (← toQ qo32 (Tensor.scalar (cur_kv_head * stride_vh))).item +
                (← toQ qo32 (offs_dv.row BLOCK_N))
            -- lines 580–582
            let v : Tensor [BLOCK_N, BLOCK_DV] BF16 ←
              tl.load (V_Extend + offs_v) (some ((mask_n.col BLOCK_DV).and (mask_dv.row BLOCK_N)))
                (some (← tl.convert .f32 .bf16 (tl.full [BLOCK_N, BLOCK_DV] zf32)))
            -- lines 583, 586, 590–593: `USE_FP8_EXTEND = False`, `IS_GFX1250 = False`;
            -- `p.to(v.dtype)`, one rounding to nearest even
            let p16 : Tensor [BLOCK_M, BLOCK_N] BF16 ← tl.cvt p
            -- lines 593–594: `acc * re_scale[:, None] + dot` with `dot` a `tl.dot` of the default
            -- zero accumulator and no other use: `tl.dot_add`, its addend the IEEE product
            let acc ← tl.dot_add cfgBF16 p16 v (← tl.mul acc (re_scale.col BLOCK_DV))
            -- line 596
            let e_max := n_e_max
            pure (acc, deno, e_max)
          else
            pure (acc, deno, e_max)
      -- line 607: the ordered `==` with the `float32` `+0`
      let no_kv := deno.feq (bc zf32)
      -- lines 620–625: as lines 238–243, with `stride_obs`, `stride_oh` and `offs_dv`
      let offs_o : Tensor [BLOCK_M, BLOCK_DV] (QI qo32) :=
        ((cur_seq_extend_start_idx +
              (← toQ qo32 (Tensor.scalar (cur_block_m * (IntV.ofNat BLOCK_M : I32)))).item) +
            (← toQ qo32 (offs_m.col BLOCK_DV))) *
            (← toQ qo32 (Tensor.scalar stride_obs)).item +
          (← toQ qo32 (Tensor.scalar (cur_head * stride_oh))).item +
          (← toQ qo32 (offs_dv.row BLOCK_M))
      -- line 626: `1.0` built in `float32`
      let deno_safe := tl.where no_kv (bc one32) deno
      -- lines 627, 633–638: `STORE_TRANSPOSE = False`; the platform's `float32` `/`, then the
      -- store's conversion to the pointer's `bfloat16` (nearest even)
      let o ← tl.div acc (deno_safe.col BLOCK_DV)
      let o16 : Tensor [BLOCK_M, BLOCK_DV] BF16 ← tl.cvt o
      tl.store (O_Extend + offs_o) o16 (some ((mask_m.col BLOCK_DV).and (mask_dv.row BLOCK_M)))

/-! ## The conditions -/

/-- What the host chooses: one launch of `_fwd_kernel` through `extend_attention_fwd(q_extend,
k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices, max_len_extend,
sm_scale, custom_mask, mask_indptr, sliding_window_size, window_kv_offsets)`. Each tensor is its
storage (`Buf`) and its own strides in elements, the last dimension contiguous. -/
structure Inst where
  /-- `USE_CUSTOM_MASK` (`custom_mask` given): a target verification, whose `qo_indptr` the
  backend builds as `int32`; without it an extend batch, with an `int64` `qo_indptr`. -/
  cm : Bool
  /-- `SLIDING_WINDOW_SIZE` when the layer has a sliding window (`> 0`); `0` for the host's `-1`
  (no window). -/
  sw : Nat
  /-- `qo_indptr.shape[0] - 1`: the grid's `x`. -/
  batch : Nat
  /-- `q_extend.shape[0]` (`= k_extend.shape[0] = v_extend.shape[0]`): the new tokens. -/
  T : Nat
  /-- `k_buffer.shape[0]` (`= v_buffer.shape[0]`): the cache's slots. -/
  S : Nat
  /-- `kv_indices.shape[0]`. -/
  NI : Nat
  /-- `custom_mask.shape[0]` (with the custom mask). -/
  NM : Nat
  /-- `q_extend.shape[1]`: `num_heads`, the grid's `y`. -/
  H : Nat
  /-- `k_extend.shape[1]` (`= v_extend`, `k_buffer`, `v_buffer`): `num_kv_heads`. -/
  HKV : Nat
  /-- The head dimension `Lq = Lk = Lv` (the last dimension of every tensor). -/
  D : Nat
  /-- `BLOCK_DMODEL = BLOCK_DV = triton.next_power_of_2(D)`. -/
  BD : Nat
  /-- `BLOCK_M`. -/
  BM : Nat
  /-- `max_len_extend`. -/
  maxExt : Nat
  /-- `sm_scale`, the Python `float` (a double; by default `1/sqrt(Lq)`, line 744). -/
  sm_scale : F64
  /-- `q_extend.stride(0)`, `.stride(1)`. -/
  sqt : Nat
  sqh : Nat
  /-- `k_extend.stride(0)`, `.stride(1)`. -/
  skt : Nat
  skh : Nat
  /-- `v_extend.stride(0)`, `.stride(1)`. -/
  svt : Nat
  svh : Nat
  /-- `o_extend.stride(0)`, `.stride(1)`. -/
  sot : Nat
  soh : Nat
  /-- `k_buffer.stride(0)`, `.stride(1)` (`_extract_kv_strides`). -/
  sbkt : Nat
  sbkh : Nat
  /-- `v_buffer.stride(0)`, `.stride(1)`. -/
  sbvt : Nat
  sbvh : Nat
  /-- The storages of `q_extend`, `k_extend`, `v_extend`, `o_extend`, `k_buffer`, `v_buffer`. -/
  q : Buf .bf16
  k : Buf .bf16
  v : Buf .bf16
  o : Buf .bf16
  kb : Buf .bf16
  vb : Buf .bf16
  /-- The storages of `qo_indptr` (`int32` with the custom mask, else `int64`), `kv_indptr`,
  `kv_indices`. -/
  qo : Buf (qoD cm)
  kvp : Buf .i32
  kvi : Buf .i64
  /-- The storages of `custom_mask` (`bool`), `mask_indptr` and `window_kv_offsets` (`int64`),
  read only with the custom mask (`window_kv_offsets` only with a window too). -/
  cmask : Buf .i1
  mip : Buf .i64
  wko : Buf .i64

/-- Element `[t, h, e]` of `q_extend`, by its own strides. -/
def qIdx (d : Inst) (t h e : Nat) : Nat := t * d.sqt + h * d.sqh + e
/-- Element `[t, g, e]` of `k_extend`. -/
def kIdx (d : Inst) (t g e : Nat) : Nat := t * d.skt + g * d.skh + e
/-- Element `[t, g, e]` of `v_extend`. -/
def vIdx (d : Inst) (t g e : Nat) : Nat := t * d.svt + g * d.svh + e
/-- Element `[t, h, e]` of `o_extend`. -/
def oIdx (d : Inst) (t h e : Nat) : Nat := t * d.sot + h * d.soh + e
/-- Element `[s, g, e]` of `k_buffer`. -/
def kbIdx (d : Inst) (s g e : Nat) : Nat := s * d.sbkt + g * d.sbkh + e
/-- Element `[s, g, e]` of `v_buffer`. -/
def vbIdx (d : Inst) (s g e : Nat) : Nat := s * d.sbvt + g * d.sbvh + e

/-- `kv_group_num = num_heads // num_kv_heads` (line 746). -/
def kvg (d : Inst) : Nat := d.H / d.HKV

/-- The key/value head of query head `h`: `h // kv_group_num` (the reference's
`repeat_interleave(group)`). -/
def kvh (d : Inst) (h : Nat) : Nat := h / kvg d

/-- The number of query blocks, `cdiv(max_len_extend, BLOCK_M)` (line 771). -/
def nBlocks (d : Inst) : Nat := (d.maxExt + d.BM - 1) / d.BM

/-- The instances covered. -/
structure Inst.ok (d : Inst) : Prop where
  /-- The grid `(batch, num_heads, cdiv(max_len_extend, BLOCK_M))` within CUDA's limits and fewer
  than `2^31` programs (`Launch.Valid`). -/
  batch_lt : d.batch < 2 ^ 31
  H_lt : d.H < 2 ^ 16
  nb_lt : nBlocks d < 2 ^ 16
  grid_lt : d.batch * d.H * nBlocks d < 2 ^ 31
  /-- `num_kv_heads ≥ 1` divides `num_heads` (SETUP.md). -/
  kv_pos : 0 < d.HKV
  kv_dvd : d.HKV ∣ d.H
  /-- The head dimension: `1 ≤ D`, `D ≠ 192`, and `BLOCK_DMODEL` a power of two with
  `D ≤ BLOCK_DMODEL ≤ 256` (`next_power_of_2(D)` for `D ≤ 256` other than `192`; SETUP.md's head
  dims are 13 to 256). -/
  D_pos : 0 < d.D
  /-- `D ≠ 192`: at `Lq = 192` the host launches with `BLOCK_DMODEL = 128`, `BLOCK_DPE = 64`
  (lines 653–655), the RoPE split this claim does not describe (README, Exclusions). -/
  D_ne : d.D ≠ 192
  bd_pow2 : ∃ a, d.BD = 2 ^ a
  D_le : d.D ≤ d.BD
  bd_le : d.BD ≤ 256
  /-- `BLOCK_M`, the H200's choice (lines 664–669): `128` for head dims up to `128`, else `64`. -/
  bm : d.BM = if d.D ≤ 128 then 128 else 64
  /-- `max_len_extend ≤ 2^18`; with the custom mask (target verification's draft tokens per
  request) `≤ 2^12`. -/
  maxExt_le : d.maxExt ≤ 2 ^ 18
  cm_maxExt : d.cm = true → d.maxExt ≤ 2 ^ 12
  /-- `q_extend.shape[0]`, the new tokens of the batch, is a context length: at most `2^22`
  tokens. -/
  T_le : d.T ≤ 2 ^ 22
  /-- `qo_indptr.shape[0] - 1`, the number of requests, is a batch size: at most `8192`. -/
  batch_le : d.batch ≤ 8192
  /-- A window of at most `2^18` positions (`SLIDING_WINDOW_SIZE`, an `int32` constant). -/
  sw_le : d.sw ≤ 2 ^ 18
  /-- `sm_scale` is a finite double in `[2^-64, 2^64]` (by default `1/sqrt(D) ∈ [1/16, 1]`). -/
  scale : d.sm_scale.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ d.sm_scale.toRat ∧
    d.sm_scale.toRat ≤ 2 ^ 64
  /-- The twelve strides are `int32` arguments. -/
  strides_lt : d.sqt < 2 ^ 31 ∧ d.sqh < 2 ^ 31 ∧ d.skt < 2 ^ 31 ∧ d.skh < 2 ^ 31 ∧
    d.svt < 2 ^ 31 ∧ d.svh < 2 ^ 31 ∧ d.sot < 2 ^ 31 ∧ d.soh < 2 ^ 31 ∧
    d.sbkt < 2 ^ 31 ∧ d.sbkh < 2 ^ 31 ∧ d.sbvt < 2 ^ 31 ∧ d.sbvh < 2 ^ 31
  /-- **A limit of the kernel**: its `int32` products `cur_head * stride_qh`,
  `cur_kv_head * stride_kh`, … (lines 241, 317, 412, 513, 577, 623) do not wrap. -/
  heads_i32 : (d.H - 1) * d.sqh < 2 ^ 31 ∧ (d.HKV - 1) * d.skh < 2 ^ 31 ∧
    (d.HKV - 1) * d.svh < 2 ^ 31 ∧ (d.H - 1) * d.soh < 2 ^ 31 ∧
    (d.HKV - 1) * d.sbkh < 2 ^ 31 ∧ (d.HKV - 1) * d.sbvh < 2 ^ 31
  /-- **A limit of the kernel with an `int32` `qo_indptr`** (the custom mask): the element
  offsets of `q_extend`, `k_extend`, `v_extend`, `o_extend` (lines 238–243, 511–515, 575–579,
  620–625) are then `int32` and do not wrap. -/
  cm_i32 : d.cm = true → 0 < d.T →
    (d.T - 1) * d.sqt + (d.H - 1) * d.sqh + d.D ≤ 2 ^ 31 ∧
    (d.T - 1) * d.skt + (d.HKV - 1) * d.skh + d.D ≤ 2 ^ 31 ∧
    (d.T - 1) * d.svt + (d.HKV - 1) * d.svh + d.D ≤ 2 ^ 31 ∧
    (d.T - 1) * d.sot + (d.H - 1) * d.soh + d.D ≤ 2 ^ 31
  /-- Each storage holds its tensor's last element. -/
  q_len : 0 < d.T → (d.T - 1) * d.sqt + (d.H - 1) * d.sqh + d.D ≤ d.q.len
  k_len : 0 < d.T → (d.T - 1) * d.skt + (d.HKV - 1) * d.skh + d.D ≤ d.k.len
  v_len : 0 < d.T → (d.T - 1) * d.svt + (d.HKV - 1) * d.svh + d.D ≤ d.v.len
  o_len : 0 < d.T → (d.T - 1) * d.sot + (d.H - 1) * d.soh + d.D ≤ d.o.len
  kb_len : 0 < d.S → (d.S - 1) * d.sbkt + (d.HKV - 1) * d.sbkh + d.D ≤ d.kb.len
  vb_len : 0 < d.S → (d.S - 1) * d.sbvt + (d.HKV - 1) * d.sbvh + d.D ≤ d.vb.len
  qo_len : d.batch + 1 ≤ d.qo.len
  kvp_len : d.batch + 1 ≤ d.kvp.len
  kvi_len : d.NI ≤ d.kvi.len
  cmask_len : d.cm = true → d.NM ≤ d.cmask.len
  mip_len : d.cm = true → d.batch + 1 ≤ d.mip.len
  wko_len : d.cm = true → 0 < d.sw → d.batch ≤ d.wko.len
  /-- `o_extend`'s elements are distinct: it is row-major in one order or the other
  (`torch.empty_like(q_extend)`: `q_extend`'s strides when it is dense, else contiguous). -/
  o_layout : (d.D ≤ d.soh ∧ d.H * d.soh ≤ d.sot) ∨ (d.D ≤ d.sot ∧ d.T * d.sot ≤ d.soh)
  /-- `o_extend` shares no byte with the inputs (a fresh allocation, line 736). -/
  disj : Disjoint d.o.bytes d.q.bytes ∧ Disjoint d.o.bytes d.k.bytes ∧
    Disjoint d.o.bytes d.v.bytes ∧ Disjoint d.o.bytes d.kb.bytes ∧
    Disjoint d.o.bytes d.vb.bytes ∧ Disjoint d.o.bytes d.qo.bytes ∧
    Disjoint d.o.bytes d.kvp.bytes ∧ Disjoint d.o.bytes d.kvi.bytes ∧
    (d.cm = true → Disjoint d.o.bytes d.cmask.bytes ∧ Disjoint d.o.bytes d.mip.bytes ∧
      (0 < d.sw → Disjoint d.o.bytes d.wko.bytes))

/-- The model on an instance: the launch of lines 784–856, with `k_scale = v_scale = 1.0`;
`USE_CUSTOM_MASK = cm`, `qo_indptr` of `int32` with it, `SLIDING_WINDOW_SIZE = sw`. -/
def model (d : Inst) : Kernel :=
  fwd_kernel d.cm d.cm d.sw (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.o.ptr 0) (d.kb.ptr 0)
    (d.vb.ptr 0) (d.qo.ptr 0) (d.kvp.ptr 0) (d.kvi.ptr 0) (d.cmask.ptr 0) (d.mip.ptr 0)
    (d.wko.ptr 0)
    (FloatV.hostF32 d.sm_scale) (FloatV.hostF32 one64) (FloatV.hostF32 one64) (IntV.ofNat (kvg d))
    (IntV.ofNat d.sqt) (IntV.ofNat d.sqh) (IntV.ofNat d.skt) (IntV.ofNat d.skh)
    (IntV.ofNat d.svt) (IntV.ofNat d.svh) (IntV.ofNat d.sot) (IntV.ofNat d.soh)
    (IntV.ofNat d.sbkt) (IntV.ofNat d.sbkh) (IntV.ofNat d.sbvt) (IntV.ofNat d.sbvh)
    d.D d.D d.BD d.BD d.BM BN BN

/-- The host's launch (line 771): the grid `(batch, num_heads, cdiv(max_len_extend, BLOCK_M))`;
the source's block shapes (`[BM]`: `offs_m`, `mask_m`, `deno`, `e_max`, …; `[BD]`: `offs_d`,
`offs_dv`; `[64]`: `offs_n`, `offs_n_prefix`, `mask_n`, `offs_kv_loc`; `[BM, BD]`: `q`, `acc`,
`offs_o`; `[BD, 64]`: `k`; `[BM, 64]`: `qk`, `p`, the masks; `[64, BD]`: `v`; and the expanded
`[BM, 1]`, `[1, BD]`, `[1, 64]`, `[BD, 1]`, `[64, 1]`); `enable_fp_fusion` Triton's default
(`true`); `num_ctas = 1`, no `launch_pdl`, `debug = false`. `num_warps` and `num_stages = 1`
change no meaning. -/
def launch (d : Inst) : Launch :=
  { grid := ![d.batch, d.H, nBlocks d], fusion := true,
    blocks := [[d.BM], [d.BD], [BN], [d.BM, d.BD], [d.BD, BN], [d.BM, BN], [BN, d.BD],
      [d.BM, 1], [1, d.BD], [1, BN], [d.BD, 1], [BN, 1]] }

/-- The `bfloat16` value at element `i` of a buffer, if present. -/
def rdB (m : Mem) (b : Buf .bf16) (i : Nat) : Option BF16 := m.read .bf16 (b.ptr (i : Int)).addr

/-- Element `i` of a `bfloat16` buffer as a rational (`0` when absent). -/
def valB (m : Mem) (b : Buf .bf16) (i : Nat) : ℚ :=
  match rdB m b i with
  | some x => x.toRat
  | none => 0

/-- Element `i` of a `bfloat16` buffer is a finite value. -/
def finB (m : Mem) (b : Buf .bf16) (i : Nat) : Prop :=
  ∃ x : BF16, rdB m b i = some x ∧ x.isFinite = true

/-- Element `i` of an `int64` buffer as an integer (`0` when absent). -/
def val64 (m : Mem) (b : Buf .i64) (i : Nat) : Int :=
  match m.read .i64 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- Element `i` of an `int32` buffer as an integer (`0` when absent). -/
def val32 (m : Mem) (b : Buf .i32) (i : Nat) : Int :=
  match m.read .i32 (b.ptr (i : Int)).addr with
  | some x => x.toInt
  | none => 0

/-- An element of `qo_indptr`'s type as an integer. -/
def qoInt : (qo32 : Bool) → (qoD qo32).Val → Int
  | true, x => x.toInt
  | false, x => x.toInt

/-- Element `i` of `qo_indptr` (`int32` with the custom mask, else `int64`) as an integer, if
present. -/
def readQo (d : Inst) (m : Mem) (i : Nat) : Option Int :=
  (m.read (qoD d.cm) (d.qo.ptr (i : Int)).addr).map (qoInt d.cm)

/-- Element `i` of `qo_indptr` as an integer (`0` when absent). -/
def valQo (d : Inst) (m : Mem) (i : Nat) : Int := (readQo d m i).getD 0

/-- Request `b`'s first new token, `qo_indptr[b]`. -/
def qsOf (d : Inst) (m : Mem) (b : Nat) : Nat := (valQo d m b).toNat
/-- Request `b`'s number of new tokens, `qo_indptr[b + 1] - qo_indptr[b]`. -/
def extOf (d : Inst) (m : Mem) (b : Nat) : Nat := (valQo d m (b + 1) - valQo d m b).toNat
/-- Where request `b`'s prefix slots start in `kv_indices`, `kv_indptr[b]`. -/
def ksOf (d : Inst) (m : Mem) (b : Nat) : Nat := (val32 m d.kvp b).toNat
/-- Request `b`'s prefix length, `kv_indptr[b + 1] - kv_indptr[b]`. -/
def preOf (d : Inst) (m : Mem) (b : Nat) : Nat := (val32 m d.kvp (b + 1) - val32 m d.kvp b).toNat
/-- The cache slot of request `b`'s prefix token `j`, `kv_indices[kv_indptr[b] + j]`. -/
def slotOf (d : Inst) (m : Mem) (b j : Nat) : Nat := (val64 m d.kvi (ksOf d m b + j)).toNat

/-- Where request `b`'s block of the custom mask starts, `mask_indptr[b]`. -/
def mStart (d : Inst) (m : Mem) (b : Nat) : Nat := (val64 m d.mip b).toNat

/-- Request `b`'s window offset: `window_kv_offsets[b]` with a window, else `0` (the kernel's
`window_kv_offset = 0`). -/
def wOff (d : Inst) (m : Mem) (b : Nat) : Nat :=
  if 0 < d.sw then (val64 m d.wko b).toNat else 0

/-- The element of `custom_mask` for new token `i` of request `b` and its new token `j`: request
`b`'s block is row-major, `ext_b` rows of `pre_b + ext_b + wOff` (the window's offset, then the
prefix, then the new tokens). -/
def maskIdx (d : Inst) (m : Mem) (b i j : Nat) : Nat :=
  mStart d m b + i * (preOf d m b + extOf d m b + wOff d m b) + wOff d m b + preOf d m b + j

/-- That element (a `bool`; `false` when absent). -/
def maskBit (d : Inst) (m : Mem) (b i j : Nat) : Bool :=
  (m.read .i1 (d.cmask.ptr (maskIdx d m b i j : Int)).addr).getD false

/-- `sm_scale` as a real. -/
noncomputable def sig (d : Inst) : ℝ := (d.sm_scale.toRat : ℝ)

/-- Component `e` of the query of new token `i` of request `b`, head `h`. -/
def qv (d : Inst) (m : Mem) (b h i e : Nat) : ℚ := valB m d.q (qIdx d (qsOf d m b + i) h e)

/-- Component `e` of the key at sequence position `j` of request `b`, for query head `h`: the
cached prefix token `j` (`j < pre_b`, through `kv_indices`), or the new token `j − pre_b`. -/
def keyK (d : Inst) (m : Mem) (b h j e : Nat) : ℚ :=
  if j < preOf d m b then valB m d.kb (kbIdx d (slotOf d m b j) (kvh d h) e)
  else valB m d.k (kIdx d (qsOf d m b + (j - preOf d m b)) (kvh d h) e)

/-- Component `e` of the value at sequence position `j` of request `b`, for query head `h`. -/
def keyV (d : Inst) (m : Mem) (b h j e : Nat) : ℚ :=
  if j < preOf d m b then valB m d.vb (vbIdx d (slotOf d m b j) (kvh d h) e)
  else valB m d.v (vIdx d (qsOf d m b + (j - preOf d m b)) (kvh d h) e)

/-- **The exact score** of new token `i` (request `b`, head `h`) against sequence position `j`:
`sm_scale · Σ_e q·k`. -/
noncomputable def score (d : Inst) (m : Mem) (b h i j : Nat) : ℝ :=
  sig d * ∑ e ∈ Finset.range d.D, ((qv d m b h i e : ℚ) : ℝ) * ((keyK d m b h j e : ℚ) : ℝ)

/-- The score's magnitude, `sm_scale · Σ_e liftMag q · liftMag k` (`Rules.Num.liftMag`: a
subnormal counted at `2^-126`), the magnitude `Rules.Num.DotBF16`'s error is relative to. It is
at least `|score|`. -/
noncomputable def scoreMag (d : Inst) (m : Mem) (b h i j : Nat) : ℝ :=
  sig d * ∑ e ∈ Finset.range d.D,
    ((Rules.Num.liftMag .bf16 (qv d m b h i e) * Rules.Num.liftMag .bf16 (keyK d m b h j e) : ℚ) : ℝ)

/-- Whether new token `i` of request `b` attends its sequence position `j`: a prefix position
(`j < pre_b`; the kernel does not read the mask's prefix columns, `SKIP_PREFIX_CUSTOM_MASK`), or
a new token `j − pre_b` that is at most `i` (causal) or, with the custom mask, that its row of the
mask allows; with a window `W = sw > 0`, only if also `pre_b + i ≤ j + W`. -/
def attends (d : Inst) (m : Mem) (b i j : Nat) : Bool :=
  (decide (j < preOf d m b) ||
      (if d.cm then maskBit d m b i (j - preOf d m b) else decide (j ≤ preOf d m b + i))) &&
    (decide (d.sw = 0) || decide (preOf d m b + i ≤ j + d.sw))

/-- The positions new token `i` of request `b` attends, among its `pre_b + ext_b`. -/
def keys (d : Inst) (m : Mem) (b i : Nat) : Finset Nat :=
  (Finset.range (preOf d m b + extOf d m b)).filter fun j => attends d m b i j = true

/-- The row's score bound: the largest `scoreMag` over the positions it attends (and its own
position `pre_b + i`, which it attends on every covered input: `Pre`). -/
noncomputable def Xrow (d : Inst) (m : Mem) (b h i : Nat) : ℝ :=
  (insert (preOf d m b + i) (keys d m b i)).sup' (Finset.insert_nonempty _ _)
    fun j => scoreMag d m b h i j

/-- The requests (SETUP.md): for each `b < batch`, `qo_indptr[b] ≤ qo_indptr[b + 1]` within the
new tokens with at most `max_len_extend` new tokens, `kv_indptr[b] ≤ kv_indptr[b + 1]` within
`kv_indices`, a prefix and new tokens of at most `2^18` positions in all, and every prefix slot
`kv_indices[kv_indptr[b] + j]` (`j < pre_b`) a slot of the cache. -/
def ReqOk (d : Inst) (m : Mem) : Prop :=
  ∀ b, b < d.batch → ∃ (q0 q1 : Int) (k0 k1 : I32),
    readQo d m b = some q0 ∧ readQo d m (b + 1) = some q1 ∧
    m.read .i32 (d.kvp.ptr (b : Int)).addr = some k0 ∧
    m.read .i32 (d.kvp.ptr ((b + 1 : Nat) : Int)).addr = some k1 ∧
    0 ≤ q0 ∧ q0 ≤ q1 ∧ q1 ≤ d.T ∧ q1 - q0 ≤ d.maxExt ∧
    0 ≤ k0.toInt ∧ k0.toInt ≤ k1.toInt ∧ k1.toInt ≤ d.NI ∧
    (k1.toInt - k0.toInt) + (q1 - q0) ≤ 2 ^ 18 ∧
    ∀ j, j < preOf d m b → ∃ s : I64,
      m.read .i64 (d.kvi.ptr ((ksOf d m b + j : Nat) : Int)).addr = some s ∧
        0 ≤ s.toInt ∧ s.toInt < d.S

/-- The custom mask (with `cm`): for each request, `mask_indptr[b] ≥ 0` and, with a window,
`0 ≤ window_kv_offsets[b] ≤ 2^40`, its block within `custom_mask`
(`mask_indptr[b] + ext_b·(pre_b + ext_b + wOff) ≤ NM`), and its tree mask over the new tokens:
each new token attends itself and no later new token (target verification's tree mask: a draft
token sees its ancestors, which come before it). -/
def MaskOk (d : Inst) (m : Mem) : Prop :=
  d.cm = true → ∀ b, b < d.batch →
    (∃ s : I64, m.read .i64 (d.mip.ptr (b : Int)).addr = some s ∧ 0 ≤ s.toInt) ∧
    (0 < d.sw → ∃ w : I64, m.read .i64 (d.wko.ptr (b : Int)).addr = some w ∧
      0 ≤ w.toInt ∧ w.toInt ≤ 2 ^ 40) ∧
    mStart d m b + extOf d m b * (preOf d m b + extOf d m b + wOff d m b) ≤ d.NM ∧
    ∀ i, i < extOf d m b → maskBit d m b i i = true ∧
      ∀ j, i < j → j < extOf d m b → maskBit d m b i j = false

/-- The input values covered: in every request, every query, key and value component of its new
tokens and of its prefix slots is a finite `bfloat16`, every `|v| ≤ 2^100`, and every attended
pair has `scoreMag ≤ 2^12`. -/
def ValsOk (d : Inst) (m : Mem) : Prop :=
  ∀ b, b < d.batch →
    (∀ i h e, i < extOf d m b → h < d.H → e < d.D →
      finB m d.q (qIdx d (qsOf d m b + i) h e)) ∧
    (∀ i g e, i < extOf d m b → g < d.HKV → e < d.D →
      finB m d.k (kIdx d (qsOf d m b + i) g e) ∧ finB m d.v (vIdx d (qsOf d m b + i) g e) ∧
      |valB m d.v (vIdx d (qsOf d m b + i) g e)| ≤ 2 ^ 100) ∧
    (∀ j g e, j < preOf d m b → g < d.HKV → e < d.D →
      finB m d.kb (kbIdx d (slotOf d m b j) g e) ∧ finB m d.vb (vbIdx d (slotOf d m b j) g e) ∧
      |valB m d.vb (vbIdx d (slotOf d m b j) g e)| ≤ 2 ^ 100) ∧
    (∀ h i j, h < d.H → i < extOf d m b → j ∈ keys d m b i → scoreMag d m b h i j ≤ 2 ^ 12)

/-- The inputs covered: the nine buffers are valid (aligned, inside `[0, 2^64)`, every byte
present), and with the custom mask also `custom_mask`, `mask_indptr` and `window_kv_offsets`; the
requests well formed (`ReqOk`), the custom mask a tree mask (`MaskOk`), and their values covered
(`ValsOk`). -/
def Pre (d : Inst) (init : Mem) : Prop :=
  d.q.Valid init ∧ d.k.Valid init ∧ d.v.Valid init ∧ d.o.Valid init ∧ d.kb.Valid init ∧
    d.vb.Valid init ∧ d.qo.Valid init ∧ d.kvp.Valid init ∧ d.kvi.Valid init ∧
    (d.cm = true → d.cmask.Valid init ∧ d.mip.Valid init ∧ (0 < d.sw → d.wko.Valid init)) ∧
    ReqOk d init ∧ MaskOk d init ∧ ValsOk d init

/-! ## The specification -/

/-- **The exact attention** of new token `i` of request `b`, head `h`, component `e`: the
softmax-weighted mean `Σ_j e^{score_j}·v_j / Σ_j e^{score_j}` over the positions it attends. -/
noncomputable def attnRef (d : Inst) (m : Mem) (b h i e : Nat) : ℝ :=
  (∑ j ∈ keys d m b i, Real.exp (score d m b h i j) * ((keyV d m b h j e : ℚ) : ℝ)) /
    ∑ j ∈ keys d m b i, Real.exp (score d m b h i j)

/-- The same weighted mean of `|v|`, the scale of the tolerance (`≥ |attnRef|`). -/
noncomputable def attnAbs (d : Inst) (m : Mem) (b h i e : Nat) : ℝ :=
  (∑ j ∈ keys d m b i, Real.exp (score d m b h i j) * |((keyV d m b h j e : ℚ) : ℝ)|) /
    ∑ j ∈ keys d m b i, Real.exp (score d m b h i j)

/-- `Σ_j |v_j|` over the attended positions, the scale of the absolute floor. -/
noncomputable def vSum (d : Inst) (m : Mem) (b h i e : Nat) : ℝ :=
  ∑ j ∈ keys d m b i, |((keyV d m b h j e : ℚ) : ℝ)|

/-- The prefix trips of request `b`'s programs: `cdiv(pre_b, 64)`. -/
def tripsPre (d : Inst) (m : Mem) (b : Nat) : Nat := (preOf d m b + BN - 1) / BN

/-- The trips over the new tokens of new token `i`'s program:
`cdiv(min(ext_b, (i // BM + 1)·BM), 64)`. -/
def tripsExt (d : Inst) (m : Mem) (b i : Nat) : Nat :=
  (min (extOf d m b) ((i / d.BM + 1) * d.BM) + BN - 1) / BN

/-! ### The tolerances (README, "Tolerances")

`BD = BLOCK_DMODEL`, `Tp` and `Te` the row's prefix and new-token trips (`T = Tp + Te` in all), `X`
the row's score bound. -/

/-- The bound on each computed score's error: the score `tl.dot` (`(BD+2)·2^-22` of `scoreMag`),
the `float32` rounding of `sm_scale`, the products by it (and by `k_scale = 1`), and absolute
floors. -/
noncomputable def dlt (BD : Nat) (X : ℝ) : ℝ := ((BD : ℝ) + 8) * (1 / 2 ^ 22) * X + 1 / 2 ^ 40

/-- The bound on the relative error of each position's coefficient and of the normaliser `deno`:
one `bfloat16` rounding of `p` (`2^-8`), the `float32` roundings and `tl.exp`/`/` errors of each
trip (`2^-20` per trip) and of the row (`8·2^-20`), and `tl.exp`'s error that grows with its
argument (`2^-17·(X + dlt)`). -/
noncomputable def rho (BD T : Nat) (X : ℝ) : ℝ :=
  Real.exp (1 / 2 ^ 8 + ((T : ℝ) + 8) * (1 / 2 ^ 20) + (X + dlt BD X) * (1 / 2 ^ 17)) - 1

/-- The bound on the accumulated `p·v` `tl.dot` errors and the accumulator's roundings, relative
to the coefficients' weighted `|v|`: the prefix stage's dots (zero accumulators) `66·2^-22` once,
its roundings `2·2^-22` per prefix trip, and the new-token stage's `72·2^-22` per trip (`tl.dot`
with the accumulator), compounded. -/
noncomputable def gam (Tp Te : Nat) : ℝ :=
  Real.exp ((66 + 2 * (Tp : ℝ) + 72 * (Te : ℝ)) * (1 / 2 ^ 22)) - 1

/-- The bound on `|acc / deno − attnRef| / attnAbs` before the store. -/
noncomputable def errE (BD Tp Te : Nat) (X : ℝ) : ℝ :=
  Real.exp (2 * dlt BD X) * (rho BD (Tp + Te) X + gam Tp Te * (1 + rho BD (Tp + Te) X)) /
      (1 - rho BD (Tp + Te) X) +
    (Real.exp (2 * dlt BD X) - 1)

/-- **The output's tolerance relative to `attnAbs`**: the store's `bfloat16` rounding and
`errE`. -/
noncomputable def tolRel (BD Tp Te : Nat) (X : ℝ) : ℝ :=
  1 / 2 ^ 8 + (1 + 1 / 2 ^ 8) * errE BD Tp Te X

/-- **The output's absolute floor**: `2^-80·(1 + Σ|v|)` (subnormal and absolute terms). -/
noncomputable def tolAbs (vs : ℝ) : ℝ := (1 / 2 ^ 80) * (1 + vs)

/-- The elements of `o_extend` the launch writes: every new token of every request, every head,
every component. -/
def outElems (d : Inst) (m : Mem) : Set Nat :=
  {x | ∃ b i h e, b < d.batch ∧ i < extOf d m b ∧ h < d.H ∧ e < d.D ∧
    x = oIdx d (qsOf d m b + i) h e}

/-- What the launch leaves: every `o_extend[qo_indptr[b] + i, h, e]` (`b < batch`, `i < ext_b`,
`h < num_heads`, `e < D`) is a finite `bfloat16` within
`tolRel(BD, tripsPre, tripsExt, Xrow)·attnAbs + tolAbs(Σ|v|)` of the exact attention `attnRef`, over the
initial inputs; every byte outside these elements is unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b i h e, b < d.batch → i < extOf d init b → h < d.H → e < d.D →
    ∃ c : BF16, rdB fin d.o (oIdx d (qsOf d init b + i) h e) = some c ∧ c.isFinite = true ∧
      |c.toReal - attnRef d init b h i e| ≤
        tolRel d.BD (tripsPre d init b) (tripsExt d init b i) (Xrow d init b h i) *
            attnAbs d init b h i e +
          tolAbs (vSum d init b h i e)) ∧
  ∀ z, z ∉ d.o.elems (outElems d init) → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's
  launches meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotBF16`: `tl.dot` of `bfloat16` operands into `float32` (lines 344, 437, 520,
  593). -/
  dot : Rules.Num.DotBF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` has a result for all operands. -/
  dot_total : Rules.Num.DotTotal P cfgBF16
  /-- [rule] `Num.ExpF32`: `tl.exp` on `float32` (lines 405, 406, 571, 572). -/
  exp_acc : Rules.Num.ExpF32 P
  /-- [rule] `Num.ExpF32NegInf`: `tl.exp(-∞) = +0` on `float32` (masked lanes; the first trip's
  `re_scale`). -/
  exp_ninf : Rules.Num.ExpF32NegInf P
  /-- [rule] `Num.MathTotal`: `tl.exp` on `float32` has a result for every input. -/
  exp_total : Rules.Num.MathTotal P .exp .binary32
  /-- [rule] `Num.DivFull`: `/` on `float32` is `div.full.f32` (line 636). -/
  div_full : Rules.Num.DivFull P
  /-- [rule] `Num.DivTotal`: `/` on `float32` has a result for every pair of inputs. -/
  div_total : Rules.Num.DivTotal P .binary32

/-- **The claim to prove.** -/
def Statement : Prop := ∀ K P, Assumptions K P → Spec K

/-! ## Not vacuous -/

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.sm90ref model launch
  exact ⟨K, Platform.sm90ref, htot, ⟨fun d _ => hf d, Platform.sm90ref.dotBF16,
    Platform.ref.dotTotal _, Platform.sm90ref.expF32, Platform.sm90ref.expF32NegInf,
    Platform.ref.mathTotal, Platform.ref.divFull, Platform.ref.divTotal _⟩⟩

/-! ## The host's layouts

`canon cm sw batch T S NI NM H HKV D BD BM maxExt sm` is the host's launch for those sizes:
contiguous `q_extend`, `k_extend`, `v_extend` (`[T, H, D]`, `[T, HKV, D]`),
`o_extend = empty_like(q_extend)`, a contiguous cache `k_buffer`, `v_buffer` (`[S, HKV, D]`),
`qo_indptr`, `kv_indptr` of `batch + 1` elements, `kv_indices` of `NI`, `custom_mask` of `NM`,
`mask_indptr` of `batch + 1` and `window_kv_offsets` of `batch`; the twelve storages at `2^40`,
`2·2^40`, …, `12·2^40`. -/

def canon (cm : Bool) (sw batch T S NI NM H HKV D BD BM maxExt : Nat) (sm : F64) : Inst where
  cm := cm
  sw := sw
  batch := batch
  T := T
  S := S
  NI := NI
  NM := NM
  H := H
  HKV := HKV
  D := D
  BD := BD
  BM := BM
  maxExt := maxExt
  sm_scale := sm
  sqt := H * D
  sqh := D
  skt := HKV * D
  skh := D
  svt := HKV * D
  svh := D
  sot := H * D
  soh := D
  sbkt := HKV * D
  sbkh := D
  sbvt := HKV * D
  sbvh := D
  q := ⟨2 ^ 40, T * H * D⟩
  k := ⟨2 * 2 ^ 40, T * HKV * D⟩
  v := ⟨3 * 2 ^ 40, T * HKV * D⟩
  o := ⟨4 * 2 ^ 40, T * H * D⟩
  kb := ⟨5 * 2 ^ 40, S * HKV * D⟩
  vb := ⟨6 * 2 ^ 40, S * HKV * D⟩
  qo := ⟨7 * 2 ^ 40, batch + 1⟩
  kvp := ⟨8 * 2 ^ 40, batch + 1⟩
  kvi := ⟨9 * 2 ^ 40, NI⟩
  cmask := ⟨10 * 2 ^ 40, NM⟩
  mip := ⟨11 * 2 ^ 40, batch + 1⟩
  wko := ⟨12 * 2 ^ 40, batch⟩

/-- Two buffers whose byte ranges do not meet are disjoint. -/
theorem bytes_disj {d1 d2 : DType} (b : Buf d1) (c : Buf d2)
    (h : b.base + b.len * d1.bytes ≤ c.base ∨ c.base + c.len * d2.bytes ≤ b.base) :
    Disjoint b.bytes c.bytes := by
  refine Set.disjoint_left.mpr fun x hx hy => ?_
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

/-- The last element of a contiguous `[t, a, l]` tensor. -/
theorem last_idx (t a l : Nat) (ht : 0 < t) (ha : 0 < a) :
    (t - 1) * (a * l) + (a - 1) * l + l = t * a * l := by
  obtain ⟨t', rfl⟩ : ∃ t', t = t' + 1 := ⟨t - 1, by omega⟩
  obtain ⟨a', rfl⟩ : ∃ a', a = a' + 1 := ⟨a - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  ring

/-- **Every host launch at the code base's sizes is covered**: an extend batch or a target
verification (`cm`), with or without a window of up to `2^18` positions (`sw`), up to `8192`
requests whose grid has fewer than `2^31` programs, `2^20` new tokens, a cache of `2^22` slots, any
number of prefix indices and of mask elements, `1 … 128` heads with any
`num_kv_heads` dividing them, head dims up to `256` other than `192` with the H200's `BLOCK_M`,
`max_len_extend ≤ 2^18` (`≤ 2^12` with the custom mask, whose `T·H·D` is at most `2^31`), and an
`sm_scale` in `[2^-64, 2^64]`. -/
theorem canon_ok (cm : Bool) (sw batch T S NI NM H HKV D BD BM maxExt a : Nat) (sm : F64)
    (hsw : sw ≤ 2 ^ 18) (hb : batch < 2 ^ 31) (hT : T ≤ 2 ^ 20) (hS : S ≤ 2 ^ 22)
    (hH0 : 0 < H) (hH : H ≤ 128) (hkv0 : 0 < HKV) (hkv : HKV ∣ H)
    (hD0 : 0 < D) (hD : D ≤ BD) (hbd : BD = 2 ^ a) (hbd256 : BD ≤ 256)
    (hbm : BM = if D ≤ 128 then 128 else 64) (hm : maxExt ≤ 2 ^ 18)
    (hcm : cm = true → maxExt ≤ 2 ^ 12 ∧ T * H * D ≤ 2 ^ 31)
    (hgrid : batch * H * ((maxExt + BM - 1) / BM) < 2 ^ 31)
    (hsm : sm.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm.toRat ∧ sm.toRat ≤ 2 ^ 64)
    (hD192 : D ≠ 192) (hbat : batch ≤ 8192) :
    (canon cm sw batch T S NI NM H HKV D BD BM maxExt sm).ok := by
  have hkvH : HKV ≤ H := Nat.le_of_dvd hH0 hkv
  have hD256 : D ≤ 256 := le_trans hD hbd256
  have hHD : H * D ≤ 128 * 256 := Nat.mul_le_mul hH hD256
  have hKD : HKV * D ≤ 128 * 256 := Nat.mul_le_mul (le_trans hkvH hH) hD256
  have hH1D : (H - 1) * D ≤ H * D := Nat.mul_le_mul_right _ (Nat.sub_le _ _)
  have hK1D : (HKV - 1) * D ≤ HKV * D := Nat.mul_le_mul_right _ (Nat.sub_le _ _)
  have hq : T * H * D ≤ 2 ^ 20 * 128 * 256 := by
    rw [Nat.mul_assoc, Nat.mul_assoc]; exact Nat.mul_le_mul hT hHD
  have hk : T * HKV * D ≤ 2 ^ 20 * 128 * 256 := by
    rw [Nat.mul_assoc, Nat.mul_assoc]; exact Nat.mul_le_mul hT hKD
  have hkq : T * HKV * D ≤ T * H * D :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_left _ hkvH)
  have hkb : S * HKV * D ≤ 2 ^ 22 * 128 * 256 := by
    rw [Nat.mul_assoc, Nat.mul_assoc]; exact Nat.mul_le_mul hS hKD
  have hnb : (maxExt + BM - 1) / BM ≤ 4097 := by
    by_cases h : D ≤ 128 <;> simp only [h, ite_true, ite_false] at hbm <;> subst hbm <;> omega
  refine
    { batch_lt := by simp only [canon]; omega
      H_lt := by simp only [canon]; omega
      nb_lt := by simp only [canon, nBlocks]; omega
      grid_lt := by simp only [canon, nBlocks]; omega
      kv_pos := hkv0
      kv_dvd := hkv
      D_pos := hD0
      D_ne := hD192
      bd_pow2 := ⟨a, hbd⟩
      D_le := hD
      bd_le := hbd256
      bm := hbm
      maxExt_le := hm
      cm_maxExt := fun h => (hcm h).1
      T_le := by simp only [canon]; omega
      batch_le := hbat
      sw_le := hsw
      scale := hsm
      strides_lt := by simp only [canon]; omega
      heads_i32 := by simp only [canon]; omega
      cm_i32 := fun h h0 => by
        have h2 := (hcm h).2
        simp only [canon] at h0 ⊢
        rw [last_idx _ _ _ h0 hH0, last_idx _ _ _ h0 hkv0]
        omega
      q_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hH0]
      k_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]
      v_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]
      o_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hH0]
      kb_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]
      vb_len := fun h0 => by
        simp only [canon] at h0 ⊢; rw [last_idx _ _ _ h0 hkv0]
      qo_len := le_rfl
      kvp_len := le_rfl
      kvi_len := le_rfl
      cmask_len := fun _ => le_rfl
      mip_len := fun _ => le_rfl
      wko_len := fun _ _ => le_rfl
      o_layout := Or.inl ⟨le_rfl, le_rfl⟩
      disj := by
        refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, fun _ => ⟨?_, ?_, fun _ => ?_⟩⟩ <;>
          exact bytes_disj _ _ (by simp only [canon, DType.bytes]; omega) }

/-! ## Coverage: the code base's tests (`test_triton_attention_kernels.py:172-325, 533-643`),
SETUP.md's model shapes, sliding-window layers and target verification -/

/-- `1/sqrt(256) = 0.0625`, the host's default `sm_scale` at head dim 256. -/
def sm256 : F64 := FloatV.ofPat 0x3fb0000000000000
/-- `1/sqrt(128)` (the double `0x3fb6a09e667f3bcc`). -/
def sm128 : F64 := FloatV.ofPat 0x3fb6a09e667f3bcc
/-- `1/sqrt(96)`. -/
def sm96 : F64 := FloatV.ofPat 0x3fba20bd700c2c3f
/-- `1/sqrt(80)`. -/
def sm80 : F64 := FloatV.ofPat 0x3fbc9f25c5bfedd9
/-- `1/sqrt(64) = 0.125`. -/
def sm64 : F64 := FloatV.ofPat 0x3fc0000000000000
/-- `1/sqrt(13)`. -/
def sm13 : F64 := FloatV.ofPat 0x3fd1c01aa03be896
/-- `1/sqrt(16) = 0.25`. -/
def sm16 : F64 := FloatV.ofPat 0x3fd0000000000000

theorem sm256_ok : sm256.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm256.toRat ∧ sm256.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm128_ok : sm128.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm128.toRat ∧ sm128.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm96_ok : sm96.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm96.toRat ∧ sm96.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm80_ok : sm80.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm80.toRat ∧ sm80.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm64_ok : sm64.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm64.toRat ∧ sm64.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm13_ok : sm13.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm13.toRat ∧ sm13.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem sm16_ok : sm16.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm16.toRat ∧ sm16.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- `canon_ok`'s numeric side conditions, discharged by `norm_num`. -/
macro "canon_tac" : tactic => `(tactic| first | norm_num | (intro h; norm_num at h ⊢) | rfl)

/-- The code base's test at its largest draw: 19 requests, 12 heads on 4 KV heads, prefix and
extend lengths up to `6164` each (`T = 19·6164` new tokens, a cache of `2·19·6164` slots,
`19·6164` prefix indices), head dim 256 (`BLOCK_DMODEL = 256`, `BLOCK_M = 64`). -/
def setup_test256 : Inst := canon false 0 19 117116 234232 117116 0 12 4 256 256 64 6164 sm256
theorem setup_test256_ok : setup_test256.ok :=
  canon_ok false 0 19 117116 234232 117116 0 12 4 256 256 64 6164 8 sm256
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm256_ok
    (by canon_tac) (by canon_tac)

/-- The same at head dim 128 (`BLOCK_M = 128`). -/
def setup_test128 : Inst := canon false 0 19 117116 234232 117116 0 12 4 128 128 128 6164 sm128
theorem setup_test128_ok : setup_test128.ok :=
  canon_ok false 0 19 117116 234232 117116 0 12 4 128 128 128 6164 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- The same at head dim 96 (`BLOCK_DMODEL = 128`). -/
def setup_test96 : Inst := canon false 0 19 117116 234232 117116 0 12 4 96 128 128 6164 sm96
theorem setup_test96_ok : setup_test96.ok :=
  canon_ok false 0 19 117116 234232 117116 0 12 4 96 128 128 6164 7 sm96
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm96_ok
    (by canon_tac) (by canon_tac)

/-- The same at head dim 80 (`BLOCK_DMODEL = 128`). -/
def setup_test80 : Inst := canon false 0 19 117116 234232 117116 0 12 4 80 128 128 6164 sm80
theorem setup_test80_ok : setup_test80.ok :=
  canon_ok false 0 19 117116 234232 117116 0 12 4 80 128 128 6164 7 sm80
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm80_ok
    (by canon_tac) (by canon_tac)

/-- The same at head dim 13 (`BLOCK_DMODEL = 16`). -/
def setup_test13 : Inst := canon false 0 19 117116 234232 117116 0 12 4 13 16 128 6164 sm13
theorem setup_test13_ok : setup_test13.ok :=
  canon_ok false 0 19 117116 234232 117116 0 12 4 13 16 128 6164 4 sm13
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm13_ok
    (by canon_tac) (by canon_tac)

/-- Llama-3-8B's heads (32 on 8 KV heads, head dim 128), a chunked-prefill batch of 8 requests,
8192 new tokens, a cache of `2^20` slots. -/
def setup_llama : Inst := canon false 0 8 8192 1048576 1048576 0 32 8 128 128 128 8192 sm128
theorem setup_llama_ok : setup_llama.ok :=
  canon_ok false 0 8 8192 1048576 1048576 0 32 8 128 128 128 8192 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- Qwen2.5-7B's heads (28 on 4, head dim 128), 4 requests, 16384 new tokens. -/
def setup_qwen : Inst := canon false 0 4 16384 2097152 1048576 0 28 4 128 128 128 16384 sm128
theorem setup_qwen_ok : setup_qwen.ok :=
  canon_ok false 0 4 16384 2097152 1048576 0 28 4 128 128 128 16384 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- Qwen2.5-0.5B's heads (14 on 2, head dim 64), 64 requests, 4096 new tokens. -/
def setup_h64 : Inst := canon false 0 64 4096 1048576 1048576 0 14 2 64 64 128 512 sm64
theorem setup_h64_ok : setup_h64.ok :=
  canon_ok false 0 64 4096 1048576 1048576 0 14 2 64 64 128 512 6 sm64
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm64_ok
    (by canon_tac) (by canon_tac)

/-- Gemma-3-12B's global layers (16 heads on 8, head dim 256), 8 requests, 8192 new tokens. -/
def setup_gemma : Inst := canon false 0 8 8192 1048576 1048576 0 16 8 256 256 64 8192 sm256
theorem setup_gemma_ok : setup_gemma.ok :=
  canon_ok false 0 8 8192 1048576 1048576 0 16 8 256 256 64 8192 8 sm256
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm256_ok
    (by canon_tac) (by canon_tac)

/-- One KV head per rank (Llama-3-70B at tensor parallelism 8: 8 heads on 1, head dim 128), 128
requests. -/
def setup_mqa : Inst := canon false 0 128 16384 1048576 1048576 0 8 1 128 128 128 1024 sm128
theorem setup_mqa_ok : setup_mqa.ok :=
  canon_ok false 0 128 16384 1048576 1048576 0 8 1 128 128 128 1024 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- A 128K context: a prefix of `2^17 − 8192` cached tokens and a chunk of 8192 new ones
(Llama-3-8B's heads). -/
def setup_long : Inst := canon false 0 1 8192 131072 131072 0 32 8 128 128 128 8192 sm128
theorem setup_long_ok : setup_long.ok :=
  canon_ok false 0 1 8192 131072 131072 0 32 8 128 128 128 8192 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- The longest covered request: a prefix of `2^18 − 8192` cached tokens and a chunk of 8192 new
ones (32 heads on 4, head dim 128: a 256K context of the Qwen3 family). -/
def setup_long256 : Inst := canon false 0 1 8192 262144 262144 0 32 4 128 128 128 8192 sm128
theorem setup_long256_ok : setup_long256.ok :=
  canon_ok false 0 1 8192 262144 262144 0 32 4 128 128 128 8192 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- The code base's sliding-window test (`test_triton_attention_kernels.py:533-643`) at its largest
draw: window 127, 19 requests, 64 heads on 8, head dim 128, prefix and extend lengths up to
`6164` (the whole prefix passed, the window masking it). -/
def setup_test_swa : Inst := canon false 127 19 117116 234232 117116 0 64 8 128 128 128 6164 sm128
theorem setup_test_swa_ok : setup_test_swa.ok :=
  canon_ok false 127 19 117116 234232 117116 0 64 8 128 128 128 6164 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- Gemma-3-12B's local layers (window `sliding_window - 1 = 1023`, 16 heads on 8, head dim 256):
8 requests, 8192 new tokens, each request's window prefix (`window_kv_indices`) up to 1023
slots. -/
def setup_gemma_local : Inst := canon false 1023 8 8192 1048576 8184 0 16 8 256 256 64 8192 sm256
theorem setup_gemma_local_ok : setup_gemma_local.ok :=
  canon_ok false 1023 8 8192 1048576 8184 0 16 8 256 256 64 8192 8 sm256
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm256_ok
    (by canon_tac) (by canon_tac)

/-- Target verification of Llama-3-8B (32 heads on 8, head dim 128) with EAGLE's 64 draft tokens
per request (`--speculative-num-draft-tokens 64`): 32 requests, `int32` `qo_indptr`, prefixes up
to `2^18 − 64`, a tree mask of up to `2^29` elements. -/
def setup_verify : Inst := canon true 0 32 2048 4194304 4194304 536870912 32 8 128 128 128 64 sm128
theorem setup_verify_ok : setup_verify.ok :=
  canon_ok true 0 32 2048 4194304 4194304 536870912 32 8 128 128 128 64 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- Target verification of a chain of 3 draft steps (4 tokens per request, EAGLE with top-k 1),
256 requests, Qwen2.5-7B's heads (28 on 4). -/
def setup_verify_chain : Inst := canon true 0 256 1024 4194304 4194304 268435456 28 4 128 128 128 4 sm128
theorem setup_verify_chain_ok : setup_verify_chain.ok :=
  canon_ok true 0 256 1024 4194304 4194304 268435456 28 4 128 128 128 4 7 sm128
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm128_ok
    (by canon_tac) (by canon_tac)

/-- Target verification on Gemma-3-12B's local layers: window 1023, 8 draft tokens per request,
16 requests, the windows' offsets in the tree mask. -/
def setup_verify_swa : Inst := canon true 1023 16 128 1048576 16368 67108864 16 8 256 256 64 8 sm256
theorem setup_verify_swa_ok : setup_verify_swa.ok :=
  canon_ok true 1023 16 128 1048576 16368 67108864 16 8 256 256 64 8 8 sm256
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm256_ok
    (by canon_tac) (by canon_tac)

/-! ## Satisfiable

One request with no prefix and one new token, one head, head dim 16 (`BLOCK_DMODEL = 16`, the
smallest that compiles: `tl.dot` needs `K ≥ 16`), an empty cache, every input `0`,
`qo_indptr = [0, 1]` and `kv_indptr = [0, 0]`. -/

def satInst : Inst := canon false 0 1 1 0 0 0 1 1 16 16 128 1 sm16

theorem satInst_ok : satInst.ok :=
  canon_ok false 0 1 1 0 0 0 1 1 16 16 128 1 4 sm16
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac)
    (by canon_tac) (by canon_tac) (by canon_tac) (by canon_tac) sm16_ok
    (by canon_tac) (by canon_tac)

/-- Every byte `0`, except the low byte of `qo_indptr[1]`, which is `1`. -/
def satMem : Mem := fun x => if x = 7 * 2 ^ 40 + 8 then some 1 else some 0

theorem satMem_qo0 : readQo satInst satMem 0 = some 0 := by
  decide +kernel

theorem satMem_qo1 : readQo satInst satMem (0 + 1) = some 1 := by
  decide +kernel

theorem satMem_kvp0 :
    satMem.read .i32 (satInst.kvp.ptr ((0 : Nat) : Int)).addr = some (IntV.ofInt 0) := by
  decide +kernel

theorem satMem_kvp1 :
    satMem.read .i32 (satInst.kvp.ptr ((0 + 1 : Nat) : Int)).addr = some (IntV.ofInt 0) := by
  decide +kernel

theorem satMem_read_bf16 {i : Int} (hi : 0 ≤ i) (hi' : i < 2 ^ 39) (b : Int)
    (hb : b = 2 ^ 40 ∨ b = 2 * 2 ^ 40 ∨ b = 3 * 2 ^ 40) :
    satMem.read .bf16 (b + i * 2) = some (FloatV.ofPat 0 : BF16) := by
  have h1 : ¬ (b + i * 2 = 7 * 2 ^ 40 + 8) := by rcases hb with rfl | rfl | rfl <;> omega
  have h2 : ¬ (b + i * 2 + 1 = 7 * 2 ^ 40 + 8) := by rcases hb with rfl | rfl | rfl <;> omega
  have hbytes : satMem.readBytes (b + i * 2) 2 = some [0, 0] := by
    simp only [Mem.readBytes, satMem, h1, h2, ite_false]
  have hal : Mem.Aligned .bf16 (b + i * 2) := by
    refine ⟨by rcases hb with rfl | rfl | rfl <;> omega, ?_, ?_⟩
    · simp only [DType.bytes]; rcases hb with rfl | rfl | rfl <;> omega
    · simp only [DType.bytes]; omega
  unfold Mem.read
  rw [show DType.bf16.bytes = 2 from rfl]
  simp only [hal, ite_true, hbytes]
  decide

theorem satB {b : Buf .bf16} (hb : b = satInst.q ∨ b = satInst.k ∨ b = satInst.v) {i : Nat}
    (hi : i < 16) : rdB satMem b i = some (FloatV.ofPat 0) := by
  simp only [rdB, Buf.ptr_addr]
  refine satMem_read_bf16 (by omega) (by omega) _ ?_
  rcases hb with rfl | rfl | rfl <;> simp [satInst, canon]

theorem satV {b : Buf .bf16} (hb : b = satInst.q ∨ b = satInst.k ∨ b = satInst.v) {i : Nat}
    (hi : i < 16) : valB satMem b i = 0 := by
  simp only [valB, satB hb hi]
  decide +kernel

theorem satQs : qsOf satInst satMem 0 = 0 := by
  simp only [qsOf, valQo, satMem_qo0]; decide

theorem satExt : extOf satInst satMem 0 = 1 := by
  simp only [extOf, valQo, satMem_qo0, satMem_qo1]; decide

theorem satPre : preOf satInst satMem 0 = 0 := by
  simp only [preOf, val32, satMem_kvp0, satMem_kvp1]; decide

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨satInst, satMem, satInst_ok, ?_⟩
  have hv : ∀ {dt : DType} (b : Buf dt), 0 ≤ b.base → b.base + b.len * dt.bytes ≤ 2 ^ 64 →
      b.base % dt.bytes = 0 → b.Valid satMem := fun b h1 h2 h3 =>
    ⟨h1, h2, h3, fun x _ => by simp only [satMem]; split <;> simp⟩
  refine ⟨hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]),
    hv _ (by simp [satInst, canon]) (by simp [satInst, canon, DType.bytes, qoD])
      (by simp [satInst, canon, DType.bytes, qoD]), fun h => by simp [satInst, canon] at h, ?_,
    fun h => by simp [satInst, canon] at h, ?_⟩
  · intro b hb
    have hb0 : b = 0 := by simp [satInst, canon] at hb; omega
    subst hb0
    refine ⟨0, 1, IntV.ofInt 0, IntV.ofInt 0, satMem_qo0, satMem_qo1,
      satMem_kvp0, satMem_kvp1, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide, fun j hj => ?_⟩
    rw [satPre] at hj; omega
  · intro b hb
    have hb0 : b = 0 := by simp [satInst, canon] at hb; omega
    subst hb0
    rw [satExt, satPre, satQs]
    have hD : satInst.D = 16 := rfl
    refine ⟨fun i h e hi hh he => ?_, fun i g e hi hg he => ?_, fun j g e hj _ _ => ?_,
      fun h i j hh hi hj => ?_⟩
    · have : i = 0 ∧ h = 0 ∧ e < 16 := by simp [satInst, canon] at hi hh he; omega
      obtain ⟨rfl, rfl, he'⟩ := this
      have hi : qIdx satInst (0 + 0) 0 e = e := by simp [qIdx]
      exact ⟨_, by rw [hi]; exact satB (Or.inl rfl) he', by decide⟩
    · have : i = 0 ∧ g = 0 ∧ e < 16 := by simp [satInst, canon] at hi hg he; omega
      obtain ⟨rfl, rfl, he'⟩ := this
      have hik : kIdx satInst (0 + 0) 0 e = e := by simp [kIdx]
      have hiv : vIdx satInst (0 + 0) 0 e = e := by simp [vIdx]
      refine ⟨⟨_, by rw [hik]; exact satB (Or.inr (Or.inl rfl)) he', by decide⟩,
        ⟨_, by rw [hiv]; exact satB (Or.inr (Or.inr rfl)) he', by decide⟩, ?_⟩
      rw [hiv, satV (Or.inr (Or.inr rfl)) he']; norm_num
    · omega
    · have hj' : j = 0 := by
        simp only [keys, satPre, satExt, Finset.mem_filter, Finset.mem_range] at hj; omega
      have : h = 0 ∧ i = 0 := by simp [satInst, canon] at hh hi; omega
      obtain ⟨rfl, rfl⟩ := this
      subst hj'
      have hqv : ∀ e ∈ Finset.range satInst.D, qv satInst satMem 0 0 0 e = 0 := by
        intro e he
        rw [hD, Finset.mem_range] at he
        simp only [qv, satQs]
        rw [show qIdx satInst (0 + 0) 0 e = e by simp [qIdx]]
        exact satV (Or.inl rfl) he
      have hsum : scoreMag satInst satMem 0 0 0 0 = 0 := by
        rw [scoreMag, Finset.sum_eq_zero fun e he => by
          rw [hqv e he]; simp [Rules.Num.liftMag]]
        ring
      rw [hsum]; norm_num

end Claims.ExtendAttn
